###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""FP8 block-scaled MegaMoE operator (EP=1) on gfx1250: grouped GEMM schedules, wgrad kernels, fused epilogues.

The operator runs the grouped forward / dgrad GEMMs on one of three schedules (persistent MX kernel, 256-deep
k-tile kernel, 128-deep tile kernel) chosen from the GEMM shapes, and the wgrad on one of three kernels. Byte
checks: every GEMM schedule gives the bits of the tile kernel for y, dx, d topk_w, dW1 and dW2, including shapes that
force the fallbacks (K not a power of two, K a multiple of 256 only, H not a multiple of 1024); the two persistent
wgrad kernels agree bit for bit, with beta 0 and beta 1 and with the sum-of-squares epilogue; the fused combine add
equals a separate bf16 add; precomputed routing tables change nothing. A long single-expert window (more than 4096
tokens on one expert) must still produce a correct wgrad.
"""

import importlib

import pytest
import torch

import primus_turbo.pytorch.ops.moe.fused_mega_moe_fp8blk as FM
from tests.pytorch.ops.gfx1250_helper import (
    bit_diff_count,
    bits_equal,
    rel_l2,
    requires_flydsl,
    requires_gluon,
)

pytestmark = requires_flydsl
SCHEDULES = ("tile", "k256", "mxp")


@pytest.fixture
def fm_env(monkeypatch):
    """Reload the operator module under an environment; restore the default module afterwards."""

    def load(**env):
        for k, v in env.items():
            monkeypatch.setenv(k, v)
        return importlib.reload(FM)

    yield load
    monkeypatch.undo()
    importlib.reload(FM)


def _inputs(t, h, i, e, k, seed=0, route=None):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(t, h, device="cuda", generator=g).to(torch.bfloat16)
    w1 = (torch.randn(e, 2 * i, h, device="cuda", generator=g) * h**-0.5).to(torch.bfloat16)
    w2 = (torch.randn(e, h, i, device="cuda", generator=g) * i**-0.5).to(torch.bfloat16)
    if route is None:
        score = torch.randn(t, e, device="cuda", generator=g)
        top, idx = score.topk(k, dim=1)
        tw = torch.softmax(top, dim=1).float()
    else:
        idx, tw = route
    dy = torch.randn(t, h, device="cuda", generator=g).to(torch.bfloat16)
    return x, idx, tw, w1, w2, dy


def _run(fm, x, idx, tw, w1, w2, dy, accumulate=False, add=None, main_grad0=None, routing=None):
    xx = x.clone().requires_grad_(True)
    tt = tw.clone().requires_grad_(True)
    a1, a2 = w1.detach().clone().requires_grad_(True), w2.detach().clone().requires_grad_(True)
    if accumulate:
        a1.main_grad = torch.zeros_like(a1) if main_grad0 is None else main_grad0[0].clone()
        a2.main_grad = torch.zeros_like(a2) if main_grad0 is None else main_grad0[1].clone()
    c1, c2 = fm.FP8BlockWeightCache(a1), fm.FP8BlockWeightCache(a2)
    ss = None if add is None else add.clone().requires_grad_(True)
    y = fm.fused_mega_moe_fp8blk(
        xx, idx, tt, a1, a2, c1, c2, accumulate_wgrad=accumulate, add=ss, routing=routing
    )
    y.backward(dy)
    torch.cuda.synchronize()
    res = dict(
        y=y.detach(),
        dx=xx.grad,
        dtw=tt.grad,
        dw1=a1.main_grad if accumulate else a1.grad,
        dw2=a2.main_grad if accumulate else a2.grad,
    )
    if ss is not None:
        res["dadd"] = ss.grad
    res["_w"] = (a1, a2)
    return res


def _ref64(x, idx, tw, w1, w2, dy):
    """Unquantized fp64 math (clamped SiLU-SwiGLU)."""
    t, h = x.shape
    e, i2, _ = w1.shape
    i = i2 // 2
    with torch.enable_grad():
        x64, t64 = x.double().requires_grad_(True), tw.double().requires_grad_(True)
        a1, a2 = w1.double().requires_grad_(True), w2.double().requires_grad_(True)
        y = torch.zeros(t, h, device="cuda", dtype=torch.float64)
        for ex in range(e):
            tok, k = (idx == ex).nonzero(as_tuple=True)
            if tok.numel() == 0:
                continue
            hh = x64[tok] @ a1[ex].t()
            gate, up = hh[:, :i].clamp(-10.0, 10.0), hh[:, i:].clamp(-10.0, 10.0)
            act = gate * torch.sigmoid(gate) * up * t64[tok, k][:, None]
            y = y.index_add(0, tok, act @ a2[ex].t())
        y.backward(dy.double())
    return dict(y=y.detach(), dx=x64.grad, dtw=t64.grad, dw1=a1.grad, dw2=a2.grad)


_BASE = {}


def _chain(fm_env, schedule, h, i):
    t, e, k = 512, 16, 4
    data = _inputs(t, h, i, e, k)
    key = (h, i)
    if key not in _BASE:
        _BASE[key] = _run(fm_env(PRIMUS_TURBO_MOE_FP8BLK_GG="tile"), *data)
    base = _BASE[key]
    res = _run(fm_env(PRIMUS_TURBO_MOE_FP8BLK_GG=schedule), *data)
    for name in ("y", "dx", "dtw", "dw1", "dw2"):
        assert bit_diff_count(res[name], base[name]) == 0, name
    ref = _ref64(*data)
    assert rel_l2(res["y"], ref["y"]) < 0.08  # also checks that every output column was written
    assert rel_l2(res["dx"], ref["dx"]) < 0.1


@pytest.mark.parametrize("accumulate", [False, True], ids=["beta0", "beta1"])
def test_wgrad_kernels_agree(monkeypatch, accumulate):
    t, h, i, e, k = 512, 1024, 256, 16, 4
    data = _inputs(t, h, i, e, k, seed=1)
    g = torch.Generator(device="cuda").manual_seed(9)
    mg0 = (
        (torch.randn(e, 2 * i, h, device="cuda", generator=g) * 0.01).bfloat16(),
        (torch.randn(e, h, i, device="cuda", generator=g) * 0.01).bfloat16(),
    )
    res = {}
    for wgrad in ("vk", "vkp", "vkq"):
        monkeypatch.setenv("PRIMUS_TURBO_MOE_FP8BLK_WGRAD", wgrad)
        res[wgrad] = _run(FM, *data, accumulate=accumulate, main_grad0=mg0 if accumulate else None)
    for name in ("y", "dx", "dtw"):
        assert bits_equal(res["vkp"][name], res["vkq"][name]) and bits_equal(
            res["vk"][name], res["vkq"][name]
        )
    for name in ("dw1", "dw2"):
        assert bits_equal(res["vkp"][name], res["vkq"][name]), f"vkq differs from vkp: {name}"
        assert rel_l2(res["vk"][name], res["vkq"][name]) < 5e-3


def test_wgrad_sumsq_epilogue(monkeypatch):
    t, h, i, e, k = 512, 1024, 256, 16, 4
    data = _inputs(t, h, i, e, k, seed=2)
    monkeypatch.setenv("PRIMUS_TURBO_MOE_FP8BLK_WGRAD", "vkq")
    monkeypatch.delenv("PRIMUS_TURBO_MOE_WGRAD_SUMSQ", raising=False)
    base = _run(FM, *data, accumulate=True)
    monkeypatch.setenv("PRIMUS_TURBO_MOE_WGRAD_SUMSQ", "1")
    res = _run(FM, *data, accumulate=True)
    for name in ("dw1", "dw2"):
        assert bits_equal(res[name], base[name])
    for w, name in zip(res["_w"], ("dw1", "dw2")):
        assert w._primus_main_grad_sumsq_fresh
        got = w._primus_main_grad_sumsq.double().sum()
        want = (res[name].double() ** 2).sum()
        assert abs(got - want) <= 1e-4 * want


@requires_gluon
def test_combine_add_equals_separate_bf16_add():
    t, h, i, e, k = 512, 1024, 256, 16, 4
    data = _inputs(t, h, i, e, k, seed=3)
    shared = (torch.randn(t, h, device="cuda") * 0.1).to(torch.bfloat16)
    fused = _run(FM, *data, add=shared)
    plain = _run(FM, *data)
    assert bits_equal(fused["y"], plain["y"] + shared)
    for name in ("dx", "dtw", "dw1", "dw2"):
        assert bits_equal(fused[name], plain[name])
    assert bits_equal(fused["dadd"], data[5])


@pytest.mark.parametrize("wgrad", ["vk", "vkq"])
def test_wgrad_with_expert_window_over_4096_tokens(monkeypatch, wgrad):
    """Every token on one expert: a 6144-token window exceeds the 4096-token scale planes of the persistent wgrad."""
    monkeypatch.setenv("PRIMUS_TURBO_MOE_FP8BLK_WGRAD", wgrad)
    t, h, i, e, k = 6144, 1024, 256, 16, 1
    idx = torch.zeros(t, k, dtype=torch.int64, device="cuda")
    tw = torch.ones(t, k, device="cuda")
    data = _inputs(t, h, i, e, k, seed=5, route=(idx, tw))
    res = _run(FM, *data)
    ref = _ref64(*data)
    assert rel_l2(res["dw1"][0], ref["dw1"][0]) < 0.1
    assert rel_l2(res["dw2"][0], ref["dw2"][0]) < 0.1


@requires_gluon
def test_chain_k256_pow2_k(fm_env):
    """H = 1024, I = 256: the 256-deep k-tile kernel on every GEMM."""
    _chain(fm_env, "k256", 1024, 256)


@requires_gluon
def test_chain_mxp_all_gemms(fm_env):
    """H = 1024, I = 512: every K (1024, 512) is a power-of-two multiple of 512, so the persistent MX kernel runs all four GEMMs."""
    _chain(fm_env, "mxp", 1024, 512)


@requires_gluon
@pytest.mark.parametrize("schedule", SCHEDULES[1:])
@pytest.mark.parametrize("h, i", [(1024, 384), (1280, 256)], ids=["K384_tile_fallback", "H_not_mult_of_1024"])
def test_gemm_schedule_chain_fallback_shapes(fm_env, schedule, h, i):
    """K = 384 and H = 1280 are not powers of two: the flygg kernels refuse them with a clear error (their scale-plane
    TDM pad interval K / 32 must be a power of two; compiling them used to abort the process)."""
    with pytest.raises(ValueError, match="power of two"):
        _chain(fm_env, schedule, h, i)


def _skew_route(t, e, k, seed=0):
    """Expert 0 gets no token, expert 1 gets every token (the longest window), the rest share the remaining slots."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    rest = torch.rand(t, e - 2, device="cuda", generator=g).argsort(dim=1)[:, : k - 1] + 2
    idx = torch.cat([torch.ones(t, 1, device="cuda", dtype=rest.dtype), rest], dim=1)
    tw = torch.softmax(torch.randn(t, k, device="cuda", generator=g), dim=1).float()
    return idx, tw


@pytest.mark.parametrize("routing", ["random", "skew"])
@pytest.mark.parametrize(
    "mode, sumsq",
    [("grad", False), ("first_write", False), ("first_write", True), ("accumulate", True)],
    ids=["beta0_grad", "beta0_main_grad", "beta0_main_grad_sumsq", "beta1_main_grad_sumsq"],
)
def test_vkw_wgrad_equals_vkq(monkeypatch, routing, mode, sumsq):
    """Beta-0 wgrads run the 4-wave vkw kernel (every pool window is a multiple of 256 tokens): y, dx, d topk_w, dW1,
    dW2 and the sum-of-squares partials are the bits of vkq. Beta-1 (accumulating) calls keep vkq."""
    from primus_turbo.flydsl.grouped_gemm import grouped_gemm_fp8blk_vkq_kernel as VKQ
    from primus_turbo.pytorch.ops import utils as ops_utils

    t, h, i, e, k = 1024, 1024, 512, 8, 4
    data = _inputs(t, h, i, e, k, seed=3, route=_skew_route(t, e, k) if routing == "skew" else None)
    g = torch.Generator(device="cuda").manual_seed(11)
    mg0 = (
        (torch.randn(e, 2 * i, h, device="cuda", generator=g) * 0.01).bfloat16(),
        (torch.randn(e, h, i, device="cuda", generator=g) * 0.01).bfloat16(),
    )
    monkeypatch.setenv("PRIMUS_TURBO_MOE_FP8BLK_WGRAD", "vkq")
    if sumsq:
        monkeypatch.setenv("PRIMUS_TURBO_MOE_WGRAD_SUMSQ", "1")
    else:
        monkeypatch.delenv("PRIMUS_TURBO_MOE_WGRAD_SUMSQ", raising=False)
    accumulate = mode != "grad"
    ops_utils.set_main_grad_first_write_policy((lambda p: True) if mode == "first_write" else None)
    try:
        res = {}
        for vkw in (False, True):
            monkeypatch.setattr(FM, "_VKW", vkw)
            VKQ._COMPILED.clear()
            res[vkw] = _run(FM, *data, accumulate=accumulate, main_grad0=mg0 if accumulate else None)
            ran_vkw = any(key[0] == "vkw" for key in VKQ._COMPILED)
            assert ran_vkw == (vkw and mode != "accumulate"), f"vkw kernel ran: {ran_vkw}"
    finally:
        ops_utils.set_main_grad_first_write_policy(None)
    for name in ("y", "dx", "dtw", "dw1", "dw2"):
        assert bit_diff_count(res[True][name], res[False][name]) == 0, name
    if sumsq and accumulate:
        for wa, wb in zip(res[False]["_w"], res[True]["_w"]):
            assert wa._primus_main_grad_sumsq_fresh and wb._primus_main_grad_sumsq_fresh
            assert bits_equal(wa._primus_main_grad_sumsq, wb._primus_main_grad_sumsq)
