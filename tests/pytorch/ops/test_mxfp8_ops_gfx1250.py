###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""MXFP8 (MX_BLOCKWISE) through the op layer on gfx1250: gemm_fp8, grouped_gemm_fp8 and grouped_mlp_fp8 on the FlyDSL
MX kernels and their one operand layout (``ops.utils.MX_A`` / ``MX_B`` / ``MX_W``). On the card, against BF16 and
the fp32 emulation of the recipe; host-only, on a faked arch: what gfx1250 refuses or leaves to upstream's path."""

import importlib
import itertools
import sys
import types

import pytest
import torch
import torch.nn.functional as F

import primus_turbo.pytorch.core.quantized_tensor as qt_mod
import primus_turbo.pytorch.kernels.gemm.gemm_fp8_impl as gemm_impl
import primus_turbo.pytorch.kernels.grouped_gemm.grouped_gemm_fp8_impl as grouped_impl
import primus_turbo.pytorch.ops.utils as ops_utils
from primus_turbo.pytorch.core.backend import (
    BackendChoice,
    BackendType,
    GlobalBackendManager,
    PrecisionType,
    dispatch_report,
    reset_dispatch_report,
)
from primus_turbo.pytorch.core.low_precision import (
    Float8QuantConfig,
    Format,
    ScaleDtype,
    ScalingGranularity,
    ScalingRecipe,
    float8_e4m3,
)
from primus_turbo.pytorch.core.quantized_tensor import (
    QuantizedTensor,
    QuantizedTensorPair,
    create_quantized_weight,
)
from primus_turbo.pytorch.core.utils import is_gfx1250
from primus_turbo.pytorch.ops import (
    gemm_fp8,
    grouped_gemm_fp8,
    moe_gather,
    mx_wgrad_accum_supported,
)
from primus_turbo.pytorch.ops.grouped_mlp_fp8 import grouped_mlp_fp8
from primus_turbo.pytorch.ops.utils import MX_A, MX_B, MX_W, mx1250_recipes
from tests.pytorch.ref.mxfp8_ref import dequant, quant_2d, quant_dual, snr_db

# the modules (``primus_turbo.pytorch.ops`` re-exports the functions under the same names)
gemm_fp8_mod = importlib.import_module("primus_turbo.pytorch.ops.gemm_fp8")
grouped_gemm_fp8_mod = importlib.import_module("primus_turbo.pytorch.ops.grouped_gemm_fp8")
grouped_mlp_fp8_mod = importlib.import_module("primus_turbo.pytorch.ops.grouped_mlp_fp8")

on_gfx1250 = pytest.mark.skipif(
    not (torch.cuda.is_available() and is_gfx1250()), reason="gfx1250 MXFP8 kernels"
)
BF16, RAW_W = torch.bfloat16, ScalingRecipe(use_2d_block=True)


def _config(fmt=Format.E4M3):
    return Float8QuantConfig(
        format=fmt, granularity=ScalingGranularity.MX_BLOCKWISE, scale_dtype=ScaleDtype.E8M0, block_size=32
    )


MX = _config()
# (experts, rows, H, I): toy; the DSv3 routed experts at 8192 tokens (top-8 over 32 local experts)
TOY, DSV3 = (4, 1000, 512, 256), (32, 65536, 7168, 2048)


@pytest.fixture(autouse=True)
def _backend_state():
    GlobalBackendManager.reset()
    yield
    GlobalBackendManager.reset()


def _randn(*shape, scale=1.0, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    return (torch.randn(*shape, device="cuda", generator=g) * scale).bfloat16()


def _dense_data(M, N, K, seed=0):
    return _randn(M, K, seed=seed), _randn(N, K, scale=K**-0.5, seed=seed + 1), _randn(M, N, seed=seed + 2)


def _leaves(*tensors):
    return [t.clone().requires_grad_() for t in tensors]


def _check(name, got, ref, snr_min):
    s = snr_db(got, ref)
    cos = F.cosine_similarity(got.double().flatten(), ref.double().flatten(), dim=0).item()
    assert s >= snr_min and cos >= 0.995, f"{name}: SNR {s:.2f} dB, cos {cos:.5f}"


def _cached(w, main_grad_dtype=None, seed=6):
    """The weight as a framework caches it: both MXFP8 copies, and (``main_grad_dtype``) a main_grad to add into."""
    q, q_t = create_quantized_weight(w, float8_e4m3, MX, need_weight_transpose_cache=True)
    assert q.scaling_recipe == MX_W and q_t.scaling_recipe == MX_W
    if main_grad_dtype is not None:
        q.main_grad = _randn(*w.shape, scale=1e-3, seed=seed).to(main_grad_dtype)
        q.grad_added_to_main_grad = False
    return QuantizedTensorPair(q, q_t)


# ---------------------------------------------------------------------------------------------------- dense linear


# toy; M / N / K tails (the quantizer pads K to 128); DSv3's q_down at 1024 tokens
@on_gfx1250
@pytest.mark.parametrize("M, N, K", [(256, 512, 384), (200, 144, 320), (1024, 1536, 7168)])
def test_linear(M, N, K):
    """out / dx / dw against BF16 and against the recipe in fp32 (the same quantization: only the GEMM's
    accumulation order and the bf16 rounding of the result differ)."""
    x, w, dy = _dense_data(M, N, K)
    xg, wg = _leaves(x, w)
    out = gemm_fp8(xg, wg, trans_b=True, out_dtype=BF16, config=MX)
    out.backward(dy)
    xr, sxr, xc, sxc = quant_dual(x)  # activation / gradient: 1x32 rows and columns; weight: 32x32 blocks
    wq, sw, wt, swt = quant_2d(w)
    gr, sgr, gc, sgc = quant_dual(dy)
    mm = lambda a, sa, b, sb: dequant(a, sa) @ dequant(b, sb).t()  # noqa: E731
    emulated = (mm(xr, sxr, wq, sw), mm(gr, sgr, wt, swt), mm(gc, sgc, xc, sxc))
    xf, wf, dyf = x.float(), w.float(), dy.float()
    for name, got, bf, emu in zip(
        ("out", "dx", "dw"), (out, xg.grad, wg.grad), (xf @ wf.t(), dyf @ wf, dyf.t() @ xf), emulated
    ):
        assert got.dtype == BF16 and got.shape == bf.shape, name
        s_bf, s_emu = snr_db(got, bf), snr_db(got, emu)
        assert s_bf >= 25 and s_emu >= 50, f"{name}: SNR vs bf16 {s_bf:.2f} dB, vs the recipe {s_emu:.2f} dB"


@on_gfx1250
def test_cached_weight():
    """A weight quantized ahead by ``create_quantized_weight`` is what the op quantizes a plain weight into: the
    same bits. Without its transposed copy, the op re-quantizes that from the cache's dequantized values."""
    x, w, dy = _dense_data(512, 1536, 1024)
    pair = _cached(w)
    res = []
    for b in (w, pair, pair.data):
        xg = x.clone().requires_grad_()
        out = gemm_fp8(xg, b, trans_b=True, out_dtype=BF16, config=MX)
        out.backward(dy)
        res.append((out, xg.grad))
    plain, cached, no_transpose = res
    assert torch.equal(cached[0], plain[0]) and torch.equal(cached[1], plain[1])
    assert torch.equal(no_transpose[0], plain[0]) and snr_db(no_transpose[1], plain[1]) >= 40
    # as a framework runs it: the cached pair, the wgrad added into its main_grad
    fused, wl = _cached(w, torch.float32), w.clone().requires_grad_()
    main_grad0 = fused.data.main_grad.clone()
    kw = dict(trans_b=True, out_dtype=BF16, config=MX)
    gemm_fp8(x, wl, **kw).backward(dy)
    out = gemm_fp8(x.clone().requires_grad_(), fused, fuse_bgrad_accum_pattern="megatron", **kw)
    out.backward(dy)
    assert fused.data.grad_added_to_main_grad and torch.equal(out, plain[0])
    assert snr_db(fused.data.main_grad.double() - main_grad0.double(), wl.grad) >= 40


@on_gfx1250
def test_direct_path(monkeypatch):
    """An eager MX linear calls the FlyDSL GEMM directly (no torch.library custom op, no dispatcher) and records it
    as a dispatch; a pinned FP8 GEMM backend goes through the dispatcher. The same bits either way."""
    x, w, dy = _dense_data(512, 1536, 1024)

    def run(fuse=None):
        """out, dx and dw: returned, or (``fuse``) added into a zero fp32 main_grad."""
        xg, wg = x.clone().requires_grad_(), torch.nn.Parameter(w.clone())
        wg.main_grad, wg.grad_added_to_main_grad = torch.zeros_like(w, dtype=torch.float32), False
        out = gemm_fp8(xg, wg, trans_b=True, out_dtype=BF16, config=MX, fuse_bgrad_accum_pattern=fuse)
        out.backward(dy)
        return out, xg.grad, wg.main_grad if fuse else wg.grad

    calls = []
    for name in ("gemm_fp8_impl", "gemm_fp8_accum_impl"):
        real = getattr(gemm_fp8_mod, name)
        monkeypatch.setattr(gemm_fp8_mod, name, lambda *a, real=real, **k: calls.append(1) or real(*a, **k))
    reset_dispatch_report()
    direct = run() + run("megatron")
    assert not calls, "the eager MX linear went through the custom ops"
    assert dispatch_report() == [
        {"dispatcher": "GEMMFP8KernelDispatcher", "backend": "FLYDSL", "source": "default", "calls": 6}
    ]
    GlobalBackendManager.set_gemm_backend(BackendType.FLYDSL, PrecisionType.FP8)
    pinned = run() + run("megatron")
    assert len(calls) == 6, "a pinned backend goes through the dispatcher (fwd, dgrad, wgrad; twice)"
    GlobalBackendManager.reset()
    monkeypatch.setattr(gemm_fp8_mod, "mx1250_gemm_direct", lambda *a, **k: None)
    reset_dispatch_report()
    dispatched = run() + run("megatron")
    # FlyDSL as the ops' default backend, the accumulating wgrad included: not a fallback from another one
    assert {(row["backend"], row["source"]) for row in dispatch_report()} == {("FLYDSL", "default")}
    for a, b, c in zip(direct, pinned, dispatched):
        assert torch.equal(a, b) and torch.equal(a, c)


# ------------------------------------------------------------------------------------ wgrad into main_grad (beta=1)


@on_gfx1250
@pytest.mark.parametrize("grouped, main_grad_dtype", [(False, torch.bfloat16), (True, torch.float32)])
def test_wgrad_accum(grouped, main_grad_dtype):
    """``fuse_bgrad_accum_pattern``: the wgrad added into main_grad in the GEMM epilogue (two micro-batches, on top
    of an existing gradient) against the returned wgrad added by the caller."""
    if grouped:
        lens = torch.full((4,), 256, device="cuda", dtype=torch.int64)
        rows, shape = 1024, (4, 512, 384)
    else:
        rows, shape = 512, (768, 384)
    weight = _randn(*shape, scale=1 / 16, seed=1)
    xs = [_randn(rows, shape[-1], scale=1 / 16, seed=2 + i) for i in range(2)]
    dys = [_randn(rows, shape[-2], scale=1 / 16, seed=4 + i) for i in range(2)]
    main_grad0 = _randn(*shape, scale=1 / 16, seed=6).to(main_grad_dtype)
    assert mx_wgrad_accum_supported(grouped, main_grad0)

    def accumulate(fused):
        w = torch.nn.Parameter(weight.clone())
        w.main_grad, w.grad_added_to_main_grad = main_grad0.clone(), False
        kw = dict(trans_b=True, config=MX, fuse_bgrad_accum_pattern="megatron" if fused else None)
        for x, dy in zip(xs, dys):
            out = grouped_gemm_fp8(x, w, lens, **kw) if grouped else gemm_fp8(x, w, out_dtype=BF16, **kw)
            out.backward(dy)
            assert w.grad.shape == w.shape
            if not fused:
                w.main_grad.add_(w.grad.to(main_grad_dtype))
            w.grad = None
        assert w.grad_added_to_main_grad is fused
        return w.main_grad

    fused, unfused = accumulate(True), accumulate(False)
    # the same quantized operands: the epilogue rounds once, the caller's add twice
    torch.testing.assert_close(fused, unfused, rtol=1e-2, atol=1e-2)
    assert not torch.equal(fused, main_grad0), "main_grad was not accumulated into"


# -------------------------------------------------------------------------------------------------- routed experts


def _cases(*axes, dsv3, toy=TOY, big=DSV3):
    """Every value of ``axes`` at the toy shape, plus one case at the DSv3 shape (the ``dsv3`` values)."""

    def case(name, shape, rest):
        return pytest.param(shape, *rest, id="-".join([name, *(str(v).replace("torch.", "") for v in rest)]))

    return [case("toy", toy, rest) for rest in itertools.product(*axes)] + [case("dsv3", big, dsv3)]


def _group_lens(G, M, skew):
    if not skew:
        return torch.full((G,), M // G, dtype=torch.int64, device="cuda")
    w = torch.tensor(
        [(7 * g) % 11 for g in range(G)], dtype=torch.float64
    )  # 0..10: empty experts, ragged sizes
    lens = (w / w.sum() * M).floor().long()
    lens[int(w.argmax())] += M - int(lens.sum())
    return lens.cuda()


def _mlp_data(shape, skew=True):
    G, M, H, I = shape
    x, dy = _randn(M, H, seed=1), _randn(M, H, seed=4)
    w1, w2 = _randn(G, 2 * I, H, scale=H**-0.5, seed=2), _randn(G, H, I, scale=I**-0.5, seed=3)
    probs = torch.rand(M, device="cuda", generator=torch.Generator(device="cuda").manual_seed(5))
    return x, w1, w2, probs, dy, _group_lens(G, M, skew)


def _mlp_bf16(x, w1, w2, lens, probs):
    """The BF16 experts: per expert fc1 -> silu(gate) * up * prob -> fc2."""
    out, o = [], 0
    for g, n in enumerate(lens.tolist()):
        gate, up = (x[o : o + n] @ w1[g].t()).chunk(2, dim=-1)
        act = (F.silu(gate.float()) * up.float() * probs[o : o + n, None]).bfloat16()
        out.append(act @ w2[g].t())
        o += n
    return torch.cat(out)


def _mlp(x, w1, w2, lens, probs, fuse=None):
    return grouped_mlp_fp8(
        x, w1, w2, lens, probs, trans_w1=True, trans_w2=True, config=MX, activation="silu",
        fuse_wgrad_accum_pattern=fuse,
    )  # fmt: skip


def _check_grouped_gemm(shape, skew, config, backend, snr_min=25):
    """fc1-shaped grouped_gemm_fp8: y / dx / dw against per-expert BF16 GEMMs, on ``backend``'s grouped GEMMs."""
    G, M, H, I = shape
    lens = _group_lens(G, M, skew)
    x, w, dy = _randn(M, H, seed=1), _randn(G, 2 * I, H, scale=H**-0.5, seed=2), _randn(M, 2 * I, seed=3)
    xg, wg = _leaves(x, w)
    reset_dispatch_report()
    out = grouped_gemm_fp8(xg, wg, lens, trans_b=True, config=config)
    out.backward(dy)
    ran = {row["backend"] for row in dispatch_report() if row["dispatcher"].startswith("GroupedGEMM")}
    assert ran == {backend}, ran
    ref_y, ref_dx, ref_dw = torch.empty_like(out), torch.empty_like(x), torch.zeros_like(w)
    o = 0
    for g, n in enumerate(lens.tolist()):
        ref_y[o : o + n] = x[o : o + n] @ w[g].t()
        ref_dx[o : o + n] = dy[o : o + n] @ w[g]
        ref_dw[g] = dy[o : o + n].t() @ x[o : o + n]
        o += n
    for name, got, ref in (("y", out, ref_y), ("dx", xg.grad, ref_dx), ("dw", wg.grad, ref_dw)):
        _check(name, got, ref, snr_min)


@on_gfx1250
@pytest.mark.parametrize(
    "shape, skew", [(TOY, False), (TOY, True), ((3, 500, 160, 48), True)], ids=["uniform", "skew", "k160-n96"]
)
def test_grouped_gemm(shape, skew):
    """grouped_gemm_fp8 on the FlyDSL MX GEMMs (also K and N off the 128 grid)."""
    _check_grouped_gemm(shape, skew, MX, "FLYDSL")


@on_gfx1250
def test_grouped_gemm_upstream_path():
    """A call the FlyDSL MX GEMMs do not take (HYBRID: E5M2 gradients, a 2-bit mantissa) runs as on main, on the
    Triton kernels."""
    _check_grouped_gemm(TOY, True, _config(Format.HYBRID), "TRITON", snr_min=20)


@on_gfx1250
@pytest.mark.parametrize("shape, skew", _cases([False, True], dsv3=(True,)))
def test_grouped_mlp(shape, skew):
    """grouped_mlp_fp8 (the GLU and its gradient fused with the dual quantization of their results): y and every
    gradient against BF16."""
    x, w1, w2, probs, dy, lens = _mlp_data(shape, skew)
    leaves, refs = _leaves(x, w1, w2, probs), _leaves(x, w1, w2, probs)
    y = _mlp(leaves[0], leaves[1], leaves[2], lens, leaves[3])
    y.backward(dy)
    ref_y = _mlp_bf16(*refs[:3], lens, refs[3])
    ref_y.backward(dy)
    _check("y", y, ref_y, 20)
    for name, got, ref in zip(("dx", "dw1", "dw2", "dprobs"), leaves, refs):
        _check(name, got.grad, ref.grad, 20)


@on_gfx1250
@pytest.mark.parametrize(
    "shape, main_grad_dtype", _cases([torch.bfloat16, torch.float32], dsv3=(torch.float32,))
)
def test_grouped_mlp_cached(shape, main_grad_dtype):
    """The experts as a framework runs them: w1 / w2 cached by ``create_quantized_weight``, the wgrads accumulated
    into main_grad. y / dx / dprobs are bitwise those of plain weights (the same operands); each main_grad increment
    against the plain run's dw (one rounding against two) and BF16."""
    x, w1, w2, probs, dy, lens = _mlp_data(shape)

    def run(wa, wb, fuse):
        xl, pl = _leaves(x, probs)
        y = _mlp(xl, wa, wb, lens, pl, fuse)
        y.backward(dy)
        return y, xl.grad, pl.grad

    plain = _leaves(w1, w2)
    ref = run(*plain, None)
    cached = [_cached(w, main_grad_dtype) for w in (w1, w2)]
    main_grad0 = [p.data.main_grad.clone() for p in cached]
    for name, a, b in zip(("y", "dx", "dprobs"), run(*cached, "megatron"), ref):
        assert torch.equal(a, b), name
    bf = _leaves(x, w1, w2, probs)
    _mlp_bf16(*bf[:3], lens, bf[3]).backward(dy)
    for name, p, g0, dw, dw_bf in zip(("dw1", "dw2"), cached, main_grad0, plain, bf[1:3]):
        assert p.data.grad_added_to_main_grad, name
        inc = p.data.main_grad.double() - g0.double()
        assert snr_db(inc, dw.grad) >= 40, f"{name}: main_grad increment vs the unfused wgrad"
        _check(name, inc, dw_bf.grad, 20)


def _combine(rows, order, k):
    """Each token's k permuted rows summed in a fixed order (fp32): a deterministic unpermute / permute backward."""
    return rows.float().index_select(0, order).view(-1, k, rows.shape[1]).sum(1).to(rows.dtype)


class _Permute(torch.autograd.Function):
    """Stand-in for a framework's MoE permute: a placeholder without values (``lazy``) or the materialized rows."""

    @staticmethod
    def forward(ctx, x, d2s, order, k, lazy):
        ctx.order, ctx.k = order, k
        if lazy:
            return torch.empty(1, x.shape[1], dtype=x.dtype, device=x.device).expand(d2s.numel(), x.shape[1])
        return x.index_select(0, d2s.long())

    @staticmethod
    def backward(ctx, grad):
        return _combine(grad, ctx.order, ctx.k), None, None, None, None


class _Unpermute(torch.autograd.Function):
    """Stand-in for a framework's MoE unpermute: claims the MLP's output; its backward defers dOut's gather."""

    @staticmethod
    def forward(ctx, y, d2s, row_id_map, order, k, T, E):
        ctx.plan = moe_gather.claim_backward_gather_output(y, row_id_map, None, None, T, E, y.shape[1])
        ctx.d2s = d2s
        return _combine(y, order, k)

    @staticmethod
    def backward(ctx, dout):
        lazy = moe_gather.make_backward_gather(dout.contiguous(), ctx.plan) if ctx.plan is not None else None
        grad = lazy if lazy is not None else dout.index_select(0, ctx.d2s.long())
        return grad, None, None, None, None, None, None


@on_gfx1250
@pytest.mark.parametrize(
    "shape, skew", _cases([False, True], dsv3=(True,), toy=(257, 512, 256, 24), big=(8192, 7168, 2048, 32))
)
def test_grouped_mlp_permute_handoff(shape, skew, monkeypatch):
    """A framework's MoE permute -> grouped_mlp_fp8 -> unpermute with the permute handed over (ops/moe_gather.py;
    top-8 over 24 or 32 experts): x and dOut are quantized straight from the token rows through dest2src, and the
    output and every gradient are bitwise those of the materialized permute."""
    (T, H, I, E), k = shape, 8
    g = torch.Generator(device="cuda").manual_seed(T + H)
    if skew:  # hot experts, empty ones
        w = torch.rand(E, generator=g, device="cuda") ** 4
        w[3:9] = 0
        experts = torch.multinomial(w.expand(T, E), k, generator=g)
    else:
        experts = (torch.arange(T, device="cuda")[:, None] + (E // k) * torch.arange(k, device="cuda")) % E
    rmap = torch.zeros(T, E, dtype=torch.bool, device="cuda").scatter_(1, experts, True)
    probs = (0.1 + torch.rand(T, E, generator=g, device="cuda")) * rmap
    e_idx, t_idx = rmap.T.nonzero(as_tuple=True)  # the permuted rows: expert-major, tokens in order
    d2s, pprobs = t_idx.to(torch.int32), probs[t_idx, e_idx]
    order = torch.argsort(t_idx, stable=True)  # each token's permuted rows, in expert order
    lens = rmap.sum(0)
    row_id_map = torch.zeros(T, 2 * E + 1, dtype=torch.int32, device="cuda")
    x, dout = _randn(T, H, seed=1), _randn(T, H, seed=4)
    w1, w2 = _randn(E, 2 * I, H, scale=H**-0.5, seed=2), _randn(E, H, I, scale=I**-0.5, seed=3)

    gathered, quant = [], grouped_mlp_fp8_mod.grouped_quantize_fp8_with_trans
    monkeypatch.setattr(
        grouped_mlp_fp8_mod,
        "grouped_quantize_fp8_with_trans",
        lambda *a, **kw: gathered.append(kw.get("dest2src") is not None) or quant(*a, **kw),
    )

    def run(handoff):
        leaves = _leaves(x, w1, w2, pprobs)
        xp = _Permute.apply(leaves[0], d2s, order, k, handoff)
        if handoff:
            moe_gather.register_permuted_activation_seam(
                xp, leaves[0].detach(), d2s, pprobs, row_id_map, fuse_backward_permute_quant=True
            )
        y = _mlp(xp, leaves[1], leaves[2], lens, leaves[3])
        out = _Unpermute.apply(y, d2s, row_id_map, order, k, T, E) if handoff else _combine(y, order, k)
        out.backward(dout)
        return [out] + [t.grad for t in leaves]

    ref = run(False)
    assert gathered == [False, False]
    gathered.clear()
    got = run(True)
    assert gathered == [True, True], f"the handoff did not run (dest2src per quantization: {gathered})"
    assert not moe_gather._PERMUTED_ACTIVATION_SEAM_TABLE and not moe_gather._BACKWARD_GATHER_OUTPUTS
    for name, a, b in zip(("out", "dx", "dw1", "dw2", "dprobs"), got, ref):
        assert torch.equal(a, b), name


# ------------------------------------------------------------------------------------- host-only, on a faked arch

_OPS = (gemm_fp8_mod, grouped_gemm_fp8_mod, grouped_mlp_fp8_mod)


@pytest.fixture
def arch(monkeypatch):
    """Fake the arch the op layer sees (its modules' ``is_gfx*``; MXFP8 itself reported as supported)."""

    def fake(name):
        for mod in (ops_utils, qt_mod, gemm_impl, grouped_impl, *_OPS):
            for query in ("is_gfx942", "is_gfx950", "is_gfx1250"):
                if hasattr(mod, query):
                    monkeypatch.setattr(mod, query, lambda query=query: query == f"is_{name}")
            if hasattr(mod, "check_mxfp8_support"):
                monkeypatch.setattr(mod, "check_mxfp8_support", lambda: (True, ""))

    return fake


def _pin(monkeypatch, backend, grouped=False, auto_tune=False):
    choice = BackendChoice(backend=backend, auto_tune=auto_tune)
    getter = "get_grouped_gemm_backend" if grouped else "get_gemm_backend"
    monkeypatch.setattr(GlobalBackendManager, getter, classmethod(lambda cls, precision: choice))


def _empty(*shape, dtype=BF16):
    return torch.empty(*shape, dtype=dtype)


def _qt(shape, recipe, axis=-1):
    """A host-side MXFP8 QuantizedTensor carrying ``recipe``."""
    data = torch.zeros(shape, dtype=float8_e4m3)
    scale = torch.zeros(*shape[:-1], shape[-1] // 32, dtype=torch.uint8)
    meta = dict(shape=torch.Size(shape), orig_dtype=BF16, dest_dtype=float8_e4m3, granularity=MX.granularity)
    return QuantizedTensor(data, scale, block_size=32, scaling_recipe=recipe, quantized_axis=axis, **meta)


REFUSED = "MXFP8 on gfx1250 runs on its FlyDSL kernels"


def test_gfx1250_recipes(arch, monkeypatch):
    """gfx1250 runs E4M3 with bf16 operands and a bf16 output in its MX GEMM layout; anything else is refused with
    one error that says so (or, for a caller with another path, answered with None)."""
    arch("gfx1250")
    a, w, wg = _empty(32, 64), _empty(48, 64), _empty(2, 64, 64)
    qa, qa_t, qw = _qt((32, 64), MX_A), _qt((32, 64), MX_B, -2), _qt((48, 64), MX_W)
    raw = _qt((48, 64), RAW_W), _qt((48, 64), RAW_W, -2)
    assert mx1250_recipes("op", MX, BF16, a, None, w, None) == (MX_A, MX_B, MX_W)
    assert mx1250_recipes("op", MX, BF16, a, None, wg, None, wg, None, glu_width=128) == (MX_A, MX_B, MX_W)
    assert mx1250_recipes("op", MX, BF16, qa, qa_t, qw, None) == (MX_A, MX_B, MX_W)
    refused = {
        "HYBRID (E5M2 gradients)": (_config(Format.HYBRID), BF16, a, None, w),
        "an fp16 output": (MX, torch.float16, a, None, w),
        "an fp16 weight": (MX, BF16, a, None, w.half()),
        "a weight without 16x16 blocks": (MX, BF16, a, None, _empty(40, 64)),
        "grouped weights need N, K % 32": (MX, BF16, a, None, _empty(2, 48, 64)),
        "a weight cached in the raw layout": (MX, BF16, a, None, *raw),
    }
    for what, args in refused.items():
        with pytest.raises(NotImplementedError, match=REFUSED):
            mx1250_recipes("op", *args)
            pytest.fail(f"not refused: {what}")
        assert mx1250_recipes("op", *args, strict=False) is None, what
    # with a fuse pattern: a main_grad the wgrad cannot add into (a view at an element offset)
    unaligned = _param(2, 64, 64, offset=1)
    assert mx1250_recipes("op", MX, BF16, a, None, unaligned, fuse="megatron", strict=False) is None
    with pytest.raises(NotImplementedError, match="intermediate width I of 128"):
        mx1250_recipes("op", MX, BF16, a, None, wg, None, wg, None, glu_width=64)
    for grouped, weight in ((False, w), (True, wg)):
        _pin(monkeypatch, BackendType.TRITON, grouped)
        with pytest.raises(NotImplementedError, match="pinned backend TRITON"):
            mx1250_recipes("op", MX, BF16, a, None, weight)
        _pin(monkeypatch, BackendType.FLYDSL, grouped)
        mx1250_recipes("op", MX, BF16, a, None, weight)


def _param(*shape, main_grad_dtype=torch.float32, offset=0):
    """A weight with a main_grad to add into: a view ``offset`` elements into its buffer (16-byte aligned at 0)."""
    p = torch.nn.Parameter(_empty(*shape))
    p.main_grad = torch.zeros(p.numel() + offset, dtype=main_grad_dtype)[offset:].view(shape)
    p.grad_added_to_main_grad = False
    return p


def test_gfx1250_ops_refuse_up_front(arch, monkeypatch):
    """gemm_fp8, grouped_mlp_fp8 and create_quantized_weight raise that error before anything is quantized. So do
    the two ops for a main_grad the fused wgrad cannot add into, before the weight is marked."""
    arch("gfx1250")
    for mod in _OPS:
        for name in ("quantize_mxfp8_impl", "grouped_quantize_fp8_with_trans", "quantize_fp8_with_trans"):
            if hasattr(mod, name):
                monkeypatch.setattr(mod, name, lambda *a, **k: pytest.fail("quantized before the check"))
    hybrid, lens, offs = _config(Format.HYBRID), torch.tensor([16, 16]), torch.tensor([0, 16, 32])
    x, w = _empty(32, 64), _empty(64, 64)
    w1, w2, probs = _empty(2, 256, 64), _empty(2, 64, 128), torch.empty(32)
    mlp = dict(group_offs=offs, trans_w1=True, trans_w2=True, activation="silu", out_dtype=BF16)
    calls = (
        lambda: gemm_fp8(x, w, trans_b=True, out_dtype=BF16, config=hybrid),
        # the fused GLU's own shape rule (its activation is not padded): I = 64
        lambda: grouped_mlp_fp8(x, _empty(2, 128, 64), _empty(2, 64, 64), lens, probs, config=MX, **mlp),
        lambda: create_quantized_weight(w, float8_e4m3, hybrid),
    )
    for call in calls:
        with pytest.raises(NotImplementedError, match=REFUSED):
            call()

    dense = dict(trans_b=True, out_dtype=BF16, config=MX, fuse_bgrad_accum_pattern="megatron")
    fused = dict(config=MX, fuse_wgrad_accum_pattern="megatron", **mlp)
    accum = (
        lambda ws: gemm_fp8(x, ws[0], **dense),
        lambda ws: grouped_mlp_fp8(x, *ws[1:], lens, probs, **fused),
    )
    # a main_grad at an element offset into a flat gradient buffer (data_ptr % 16 = 4); then an aligned one under a
    # per-op auto-tune request, whose tuner would run the accumulation on main_grad
    for offset in (1, 0):
        ws = [_param(*t.shape, offset=offset) for t in (w, w1, w2)]
        for call in accum:
            with pytest.raises(NotImplementedError, match="main_grad that is bf16 or fp32"):
                call(ws)
        assert not any(p.grad_added_to_main_grad for p in ws), "a weight was marked before the check"
        for grouped in (False, True):
            _pin(monkeypatch, None, grouped, auto_tune=True)


def test_gfx1250_view_refused(arch):
    arch("gfx1250")
    with pytest.raises(AssertionError, match="gfx1250's MX GEMM layout"):
        _qt((64, 64), MX_W).view(32, 128)
    arch("gfx950")
    assert _qt((64, 64), MX_W).view(32, 128).shape == (32, 128)  # there the flags are the CDNA4 shuffles


@pytest.mark.parametrize(
    "name, grouped, choice, main_grad, want",
    [
        ("gfx1250", False, None, dict(main_grad_dtype=torch.bfloat16), True),
        ("gfx1250", True, None, {}, True),
        ("gfx1250", True, None, dict(main_grad_dtype=torch.float16), False),
        ("gfx1250", False, None, dict(offset=1), False),  # not 16-byte aligned
        ("gfx1250", False, dict(backend=BackendType.FLYDSL), {}, True),
        ("gfx1250", False, dict(backend=BackendType.TRITON), {}, False),  # no other backend adds there
        ("gfx1250", True, dict(backend=None, auto_tune=True), {}, False),  # the tuner would add, many times
        ("gfx950", False, None, {}, False),  # answered for gfx1250 only
    ],
)
def test_wgrad_accum_supported(arch, monkeypatch, name, grouped, choice, main_grad, want):
    arch(name)
    if choice is not None:
        _pin(monkeypatch, grouped=grouped, **choice)
    weight = _param(*((2, 64, 64) if grouped else (64, 64)), **main_grad)
    assert mx_wgrad_accum_supported(grouped, weight.main_grad) is want
    if name == "gfx1250" and not main_grad:
        assert not mx_wgrad_accum_supported(grouped, weight.main_grad[..., ::2])  # not contiguous


def _mx_operands(dtype, packed, k=384, groups=()):
    """Host-side MX GEMM operands (a [64, k], its scales, b [*groups, 32, k], its scales): the scales packed
    (gfx1250's layout) or raw."""

    def scale(*lead, rows):
        tail = (-(-rows // 32), -(-k // 128), 32, 4) if packed else (rows, k // 32)
        return torch.empty(*lead, *tail, dtype=torch.uint8)

    a, b = _empty(64, k, dtype=dtype), _empty(*groups, 32, k, dtype=dtype)
    return a, scale(rows=64), b, scale(*groups, rows=32)


def test_gfx1250_backends_take_mx(arch):
    """Which backends take an MXFP8 GEMM on gfx1250: FlyDSL alone the operands in its MX GEMM layout; raw operands
    the Triton grouped kernels, as on main, and no dense backend."""
    arch("gfx1250")
    mx, offs = ScalingGranularity.MX_BLOCKWISE, torch.tensor([0, 32, 64])
    # raw operands in this process's E4M3 dtype (the Triton kernels'), the layout's in gfx1250's
    for packed, dtype, want in (
        (False, float8_e4m3, [set(), {"TRITON"}, {"TRITON"}]),
        (True, torch.float8_e4m3fn, [{"FLYDSL"}] * 3),
    ):
        a, sa, b, sb = _mx_operands(dtype, packed)
        dense = dict(a=a, a_scale_inv=sa, trans_a=False, b=b, b_scale_inv=sb, trans_b=True, trans_c=False)
        a, sa, b, sb = _mx_operands(dtype, packed, groups=(2,))
        grouped = dict(a=a, b=b, a_scales=sa, b_scales=sb, group_lens=offs[1:], group_offs=offs)
        grouped.update(trans_a=False, trans_b=True, num_cu=None)
        a, sa, b, sb = _mx_operands(dtype, packed, k=256)  # the column copies [rows, tokens]
        variable_k = dict(a=a, b=b, a_scales=sa, b_scales=sb, group_lens=offs[1:] * 4, group_offs=offs * 4)
        variable_k.update(trans_a=False, trans_b=False, trans_c=False, num_cu=None)
        tables = (
            (gemm_impl.GEMMFP8KernelDispatcher, dense),
            (grouped_impl.GroupedGEMMFP8KernelDispatcher, grouped),
            (grouped_impl.GroupedGEMMFP8VariableKKernelDispatcher, variable_k),
        )
        for (dispatcher, kwargs), takes in zip(tables, want):
            kwargs.update(out_dtype=BF16, granularity=mx)
            got = {t.name for t, e in dispatcher._backends.items() if e.impl.can_handle(**kwargs)}
            assert got == takes, (packed, dispatcher.__name__, got)


def test_direct_path_bumps_version(arch, monkeypatch):
    """The direct kernel call's write into ``out`` is one mutation to autograd, as the custom op's is."""
    arch("gfx1250")
    kernel = types.SimpleNamespace(gemm_mxfp8_gfx1250=lambda *a, **k: k.get("out", "result"))
    monkeypatch.setitem(sys.modules, "primus_turbo.flydsl.gemm.gemm_mxfp8_kernel_gfx1250", kernel)
    operands = _mx_operands(torch.float8_e4m3fn, packed=True)
    buffer = torch.zeros(64 * 32)
    main_grad = buffer.view(64, 32)
    assert gemm_impl.mx1250_gemm_direct(*operands, BF16) == "result" and buffer._version == 0
    assert gemm_impl.mx1250_gemm_direct(*operands, BF16, main_grad) is main_grad
    assert (main_grad._version, buffer._version) == (1, 1)


class _Recorder:
    """Stand-ins for the quantizers (their kwargs are recorded) and the GEMM custom ops of an op module."""

    def __init__(self, monkeypatch, mod, quantizers, **gemms):
        self.calls = []
        for name, n in quantizers.items():
            monkeypatch.setattr(mod, name, lambda *a, n=n, **kw: self.calls.append(kw) or (_empty(1),) * n)
        for name, result in gemms.items():
            monkeypatch.setattr(mod, name, result)

    def gfx1250(self):
        """Per quantizer call: whether it was asked for gfx1250's MX GEMM layout or its permute handoff."""
        recipes = ("scaling_recipe", "scaling_recipe_for_trans")
        return [
            any(kw.get(r) in (MX_A, MX_B, MX_W) for r in recipes) or "dest2src" in kw or "total_m" in kw
            for kw in self.calls
        ]


_QUANTIZERS = {"grouped_quantize_fp8_with_trans": 8, "quantize_fp8_with_trans": 4}
_never = lambda what: lambda *a, **k: pytest.fail(what)  # noqa: E731


def test_other_archs_keep_upstream_path(arch, monkeypatch):
    """Off gfx1250 the three MXFP8 ops ask the quantizers for none of gfx1250's layouts, run none of its kernels and
    do not look up the permute handoff (a registered placeholder is handed on as it is)."""
    arch("gfx950")
    lens, offs = torch.tensor([16, 16]), torch.tensor([0, 16, 32])
    x, dy = _empty(32, 64).requires_grad_(), _empty(32, 64)
    rec = _Recorder(
        monkeypatch,
        gemm_fp8_mod,
        {"quantize_mxfp8_impl": 4},
        gemm_fp8_impl=lambda *a, **k: _empty(64, 64),
        mx1250_gemm_direct=_never("the gfx1250 direct path ran"),
    )
    square = _empty(64, 64).requires_grad_(), _empty(64, 64).requires_grad_()
    gemm_fp8(*square, trans_b=True, out_dtype=BF16, config=MX).backward(_empty(64, 64))
    assert rec.gfx1250() == [False] * 3  # a, b, grad_out

    rec = _Recorder(
        monkeypatch,
        grouped_gemm_fp8_mod,
        _QUANTIZERS,
        grouped_gemm_fp8_impl=lambda *a, **k: _empty(32, 64),
        grouped_gemm_fp8_variable_k_impl=lambda *a, **k: _empty(2, 64, 64),
    )
    grouped_gemm_fp8(x, _empty(2, 64, 64).requires_grad_(), lens, offs, trans_b=True, config=MX).backward(dy)
    assert rec.gfx1250() == [False] * 3

    w1, w2, probs = _empty(2, 256, 64).requires_grad_(), _empty(2, 64, 128).requires_grad_(), torch.empty(32)
    wgrads = iter((_empty(2, 64, 128), _empty(2, 256, 64)))  # dw2, then dw1
    rec = _Recorder(
        monkeypatch,
        grouped_mlp_fp8_mod,
        _QUANTIZERS,
        grouped_gemm_mxfp8_glu_impl=lambda *a, **k: (_empty(32, 256),) + (torch.empty(1),) * 4,
        grouped_gemm_mxfp8_dglu_impl=lambda *a, **k: (torch.empty(32),) + (torch.empty(1),) * 4,
        grouped_gemm_mxfp8_glu_gfx1250=_never("gfx1250's fused GLU ran"),
        grouped_gemm_fp8_impl=lambda *a, **k: _empty(32, 64),
        grouped_gemm_fp8_variable_k_impl=lambda *a, **k: next(wgrads),
        lookup_permuted_activation_seam=_never("the permute handoff was looked up"),
    )
    mlp = dict(trans_w1=True, trans_w2=True, config=MX, activation="silu")
    grouped_mlp_fp8(x, w1, w2, lens, probs, offs, **mlp).backward(dy)
    assert rec.gfx1250() == [False] * 4  # x, w1, w2, grad_out


def test_weight_cache_layout_by_arch(arch, monkeypatch):
    """create_quantized_weight (MXFP8): on gfx1250 both copies in its MX GEMM layout from one quantizer call, for a
    weight its MXFP8 ops take in that layout; else, and elsewhere, upstream's raw 2-D blocks."""
    import primus_turbo.pytorch.ops.quantization as quantization

    w, calls, generic = torch.nn.Parameter(_empty(64, 64)), [], []

    def dual(weight, dtype, granularity, **kw):
        calls.append(dict(kw, detached=not weight.requires_grad and not torch.is_grad_enabled()))
        code, scale = torch.zeros(64, 64, dtype=float8_e4m3), torch.zeros(2, 1, 32, 4, dtype=torch.uint8)
        return code, scale, code.clone(), scale.clone()

    monkeypatch.setattr(quantization, "quantize_fp8_with_trans", dual)
    monkeypatch.setattr(QuantizedTensor, "quantize", lambda *a, **kw: generic.append(kw["scaling_recipe"]))
    arch("gfx1250")
    q, q_t = create_quantized_weight(w, float8_e4m3, MX, need_weight_transpose_cache=True)
    assert calls == [dict(block_size=32, scaling_recipe=MX_W, scaling_recipe_for_trans=MX_W, detached=True)]
    assert (q.scaling_recipe, q.quantized_axis, q_t.scaling_recipe, q_t.quantized_axis) == (MX_W, 1, MX_W, 0)
    assert create_quantized_weight(w, float8_e4m3, MX)[1] is None and not generic
    # a grouped weight the FlyDSL MX GEMMs do not take (HYBRID; expert dims not % 32): the generic copies
    create_quantized_weight(_empty(2, 64, 64), float8_e4m3, _config(Format.HYBRID))
    create_quantized_weight(_empty(2, 48, 64), float8_e4m3, MX)
    create_quantized_weight(_empty(64), float8_e4m3, MX)  # nor a 1-D one: the generic path's answer
    arch("gfx950")
    create_quantized_weight(w, float8_e4m3, MX)
    assert generic == [RAW_W] * 4 and len(calls) == 2
