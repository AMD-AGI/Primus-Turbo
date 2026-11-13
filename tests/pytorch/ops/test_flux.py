###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import pytest
import torch
import torch.nn.functional as F

import primus_turbo.pytorch.ops.flux as flux_ops
from primus_turbo.pytorch.ops.flux import (
    flux_adaln_dual,
    flux_adaln_single,
    flux_gelu_linear,
)
from tests.pytorch.ops.determinism_helper import (  # noqa: F401
    _det_seed,
    check_determinism,
    make_generator,
    require_gfx950,
)
from tests.pytorch.test_utils import compute_snr

TOL = dict(rtol=2e-2, atol=2e-2)


def _require_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")


def _randn(shape, gen, scale=1.0, dtype=torch.bfloat16):
    return (torch.randn(shape, device="cuda", generator=gen, dtype=torch.float32) * scale).to(dtype)


def _leaf(t):
    return t.detach().clone().requires_grad_(True)


def _assert_snr(ref, got, floor=35.0):
    snr = compute_snr(ref, got)
    assert snr > floor, f"SNR {snr:.1f} dB <= {floor} dB"


# --- flux_gelu_linear -------------------------------------------------------------------------


def _gelu_linear_ref(x, w, b, prefix):
    h = F.gelu(F.linear(x, w, b), approximate="tanh")
    return h if prefix is None else torch.cat([prefix, h], dim=1)


@pytest.mark.parametrize(
    "m,k,n,p",
    [
        (2048, 128, 3072, 0),  # FlyDSL GELU on gfx950
        (2048, 128, 3072, 3072),  # ... with the single-stream block's prefix
        (96, 64, 160, 0),  # unfused GELU: shapes the kernel does not tile
        (96, 64, 160, 32),
    ],
)
def test_flux_gelu_linear(m, k, n, p):
    _require_cuda()
    gen = make_generator(1234)
    x, w, b = _randn((m, k), gen), _randn((n, k), gen, k**-0.5), _randn((n,), gen, 0.1)
    prefix = _randn((m, p), gen) if p else None
    leaves = [_leaf(t) for t in (x, w, b)] + ([_leaf(prefix)] if p else [])
    refs = [_leaf(t) for t in (x, w, b)] + ([_leaf(prefix)] if p else [])

    out = flux_gelu_linear(*leaves[:3], leaves[3] if p else None)
    ref = _gelu_linear_ref(*refs[:3], refs[3] if p else None)
    torch.testing.assert_close(out, ref, **TOL)
    if p:
        assert torch.equal(out[:, :p], prefix)

    grad = _randn(out.shape, gen)
    out.backward(grad)
    ref.backward(grad)
    for got, want in zip(leaves, refs):
        _assert_snr(want.grad, got.grad)


def test_flux_gelu_linear_rejects_mismatched_bias():
    _require_cuda()
    x = torch.zeros((8, 16), device="cuda", dtype=torch.bfloat16)
    w = torch.zeros((32, 16), device="cuda", dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="do not match"):
        flux_gelu_linear(x, w, torch.zeros(31, device="cuda", dtype=torch.bfloat16))


def test_flux_gelu_linear_rejects_fp32():
    _require_cuda()
    x, w, b = (torch.zeros(s, device="cuda") for s in ((8, 16), (32, 16), (32,)))
    with pytest.raises(ValueError, match="BF16"):
        flux_gelu_linear(x, w, b)


# --- flux_adaln_single ------------------------------------------------------------------------


@pytest.fixture(params=["default", "eager"])
def adaln_path(request, monkeypatch):
    """``default``: the op's own path (FlyDSL on gfx950); ``eager``: its eager PyTorch path, as off gfx950."""
    if request.param == "eager":
        monkeypatch.setattr(flux_ops, "is_gfx950", lambda: False)
    return request.param


def _adaln_single_inputs(batch, s_txt, s_img, c, gen):
    x_txt, x_img = _randn((batch, s_txt, c), gen), _randn((batch, s_img, c), gen)
    shift, scale, _ = _randn((batch, 3 * c), gen, 0.5).chunk(3, dim=1)
    return x_txt, x_img, scale, shift


def _adaln_single_ref(x_txt, x_img, scale, shift, eps):
    y = F.layer_norm(torch.cat([x_txt, x_img], dim=1), (x_txt.shape[-1],), eps=eps)
    return y * (1 + scale[:, None]) + shift[:, None]


@pytest.mark.parametrize("batch,s_txt,s_img,c", [(2, 128, 256, 512), (1, 256, 128, 1024)])
def test_flux_adaln_single(batch, s_txt, s_img, c, adaln_path):
    _require_cuda()
    gen = make_generator(555)
    eps = 1e-6
    inputs = _adaln_single_inputs(batch, s_txt, s_img, c, gen)
    leaves, refs = [_leaf(t) for t in inputs], [_leaf(t) for t in inputs]
    r_txt, r_img, y = flux_adaln_single(*leaves, eps)
    want = _adaln_single_ref(*refs, eps)
    assert torch.equal(r_txt, inputs[0]) and torch.equal(r_img, inputs[1])
    torch.testing.assert_close(y, want, **TOL)

    g_y, g_txt, g_img = _randn(y.shape, gen), _randn(r_txt.shape, gen), _randn(r_img.shape, gen)
    torch.autograd.backward((y, r_txt, r_img), (g_y, g_txt, g_img))
    torch.autograd.backward((want, refs[0], refs[1]), (g_y, g_txt, g_img))
    for got, ref in zip(leaves, refs):
        _assert_snr(ref.grad, got.grad)


def test_flux_adaln_single_rejects_untiled_sequence():
    _require_cuda()
    gen = make_generator(5)
    x_txt, x_img, scale, shift = _adaln_single_inputs(1, 100, 128, 512, gen)
    with pytest.raises(ValueError, match="multiples of 128"):
        flux_adaln_single(x_txt, x_img, scale, shift, 1e-6)


# --- flux_adaln_dual --------------------------------------------------------------------------


def _adaln_dual_inputs(batch, s, c, gen):
    streams = []
    for _ in range(2):
        x, a = _randn((batch, s, c), gen), _randn((batch, s, c), gen)
        chunks = _randn((batch, 6 * c), gen, 0.5).chunk(6, dim=1)
        streams += [x, a, chunks[2], chunks[4], chunks[3]]
    return streams


def _adaln_dual_ref(x_img, a_img, g_img, s_img, h_img, x_txt, a_txt, g_txt, s_txt, h_txt, eps):
    out = []
    for x, a, g, s, h in ((x_img, a_img, g_img, s_img, h_img), (x_txt, a_txt, g_txt, s_txt, h_txt)):
        x1 = x + g[:, None] * a
        out += [x1, F.layer_norm(x1, (x.shape[-1],), eps=eps) * (1 + s[:, None]) + h[:, None]]
    return out


@pytest.mark.parametrize("batch,s,c", [(2, 64, 1536), (1, 256, 3072)])
def test_flux_adaln_dual(batch, s, c, adaln_path):
    _require_cuda()
    gen = make_generator(777)
    eps = 1e-6
    inputs = _adaln_dual_inputs(batch, s, c, gen)
    leaves, refs = [_leaf(t) for t in inputs], [_leaf(t) for t in inputs]
    out = flux_adaln_dual(*leaves, eps)
    want = _adaln_dual_ref(*refs, eps)
    for got, ref in zip(out, want):
        torch.testing.assert_close(got, ref, **TOL)

    grads = [_randn(t.shape, gen) for t in out]
    torch.autograd.backward(out, grads)
    torch.autograd.backward(want, grads)
    for got, ref in zip(leaves, refs):
        _assert_snr(ref.grad, got.grad)


def test_flux_adaln_dual_rejects_unequal_streams():
    _require_cuda()
    gen = make_generator(9)
    inputs = _adaln_dual_inputs(1, 64, 1536, gen)
    inputs[5] = inputs[5][:, :32].contiguous()
    inputs[6] = inputs[6][:, :32].contiguous()
    with pytest.raises(ValueError, match="equal length"):
        flux_adaln_dual(*inputs, 1e-6)


# --- determinism ------------------------------------------------------------------------------


@pytest.mark.deterministic
def test_det_flux_adaln(request, det_seed):
    """Both adaLN ops, fwd + bwd."""
    gen = make_generator(det_seed)
    eps = 1e-6
    single0 = _adaln_single_inputs(2, 128, 256, 512, gen)
    dual0 = _adaln_dual_inputs(2, 64, 1536, gen)
    g_y, g_r_txt = _randn((2, 384, 512), gen), _randn((2, 128, 512), gen)
    g_dual = [_randn((2, 64, 1536), gen) for _ in range(4)]

    def run_once(mid):
        single, dual = [_leaf(t) for t in single0], [_leaf(t) for t in dual0]
        r_txt, _, y = flux_adaln_single(*single, eps)
        out_dual = flux_adaln_dual(*dual, eps)
        mid()
        torch.autograd.backward((y, r_txt, *out_dual), (g_y, g_r_txt, *g_dual))
        res = {"y": y, "y_img": out_dual[1], "y_txt": out_dual[3]}
        res.update({f"single{i}": t.grad for i, t in enumerate(single)})
        res.update({f"dual{i}": t.grad for i, t in enumerate(dual)})
        return res

    ref = {"y": _adaln_single_ref(*(t.float() for t in single0), eps)}
    ref["y_img"], ref["y_txt"] = _adaln_dual_ref(*(t.float() for t in dual0), eps)[1::2]
    check_determinism(request.node.nodeid, run_once, ref, snr_floor=35)


@pytest.mark.deterministic
def test_det_flux_gelu_linear(request, det_seed):
    """GELU MLP up-projection with the prefix, fwd + bwd."""
    require_gfx950("the FlyDSL GELU")
    gen = make_generator(det_seed)
    x0, w0, b0 = _randn((2048, 128), gen), _randn((3072, 128), gen, 128**-0.5), _randn((3072,), gen, 0.1)
    p0, g = _randn((2048, 3072), gen), _randn((2048, 6144), gen)

    def run_once(mid):
        x, w, b, p = (_leaf(t) for t in (x0, w0, b0, p0))
        out = flux_gelu_linear(x, w, b, p)
        mid()
        out.backward(g)
        return {"out": out, "dx": x.grad, "dw": w.grad, "db": b.grad, "dp": p.grad}

    ref = {"out": _gelu_linear_ref(x0.float(), w0.float(), b0.float(), p0.float())}
    check_determinism(request.node.nodeid, run_once, ref, snr_floor=35)
