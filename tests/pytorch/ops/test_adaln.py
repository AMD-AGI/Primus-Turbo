###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import pytest
import torch
import torch.nn.functional as F

import primus_turbo.pytorch.ops.normalization as norm_ops
from primus_turbo.pytorch.ops.normalization import adaln_modulate, gated_residual_adaln
from tests.pytorch.ops.determinism_helper import (  # noqa: F401
    _det_seed,
    check_determinism,
    make_generator,
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


@pytest.fixture(params=["default", "eager"])
def adaln_path(request, monkeypatch):
    """``default``: the op's own path (FlyDSL on gfx950); ``eager``: its eager PyTorch path, as off gfx950."""
    if request.param == "eager":
        monkeypatch.setattr(norm_ops, "is_gfx950", lambda: False)
    return request.param


# --- adaln_modulate ---------------------------------------------------------------------------


def _adaln_modulate_inputs(batch, s_txt, s_img, c, gen):
    x_txt, x_img = _randn((batch, s_txt, c), gen), _randn((batch, s_img, c), gen)
    shift, scale, _ = _randn((batch, 3 * c), gen, 0.5).chunk(3, dim=1)
    return x_txt, x_img, scale, shift


def _adaln_modulate_ref(x_txt, x_img, scale, shift, eps):
    y = F.layer_norm(torch.cat([x_txt, x_img], dim=1), (x_txt.shape[-1],), eps=eps)
    return y * (1 + scale[:, None]) + shift[:, None]


@pytest.mark.parametrize("batch,s_txt,s_img,c", [(2, 128, 256, 512), (1, 256, 128, 1024)])
def test_adaln_modulate(batch, s_txt, s_img, c, adaln_path):
    _require_cuda()
    gen = make_generator(555)
    eps = 1e-6
    inputs = _adaln_modulate_inputs(batch, s_txt, s_img, c, gen)
    leaves, refs = [_leaf(t) for t in inputs], [_leaf(t) for t in inputs]
    r_txt, r_img, y = adaln_modulate(*leaves, eps)
    want = _adaln_modulate_ref(*refs, eps)
    assert torch.equal(r_txt, inputs[0]) and torch.equal(r_img, inputs[1])
    torch.testing.assert_close(y, want, **TOL)

    g_y, g_txt, g_img = _randn(y.shape, gen), _randn(r_txt.shape, gen), _randn(r_img.shape, gen)
    torch.autograd.backward((y, r_txt, r_img), (g_y, g_txt, g_img))
    torch.autograd.backward((want, refs[0], refs[1]), (g_y, g_txt, g_img))
    for got, ref in zip(leaves, refs):
        _assert_snr(ref.grad, got.grad)


def test_adaln_modulate_rejects_untiled_sequence():
    _require_cuda()
    gen = make_generator(5)
    x_txt, x_img, scale, shift = _adaln_modulate_inputs(1, 100, 128, 512, gen)
    with pytest.raises(ValueError, match="multiples of 128"):
        adaln_modulate(x_txt, x_img, scale, shift, 1e-6)


# --- gated_residual_adaln ---------------------------------------------------------------------


def _gated_residual_adaln_inputs(batch, s_img, s_txt, c, gen):
    streams = []
    for s in (s_img, s_txt):
        x, a = _randn((batch, s, c), gen), _randn((batch, s, c), gen)
        chunks = _randn((batch, 6 * c), gen, 0.5).chunk(6, dim=1)
        streams += [x, a, chunks[2], chunks[4], chunks[3]]
    return streams


def _gated_residual_adaln_ref(x_img, a_img, g_img, s_img, h_img, x_txt, a_txt, g_txt, s_txt, h_txt, eps):
    out = []
    for x, a, g, s, h in ((x_img, a_img, g_img, s_img, h_img), (x_txt, a_txt, g_txt, s_txt, h_txt)):
        x1 = x + g[:, None] * a
        out += [x1, F.layer_norm(x1, (x.shape[-1],), eps=eps) * (1 + s[:, None]) + h[:, None]]
    return out


@pytest.mark.parametrize(
    "batch,s_img,s_txt,c",
    [(2, 64, 64, 1536), (1, 256, 256, 3072), (2, 256, 512, 3072), (1, 96, 32, 1536)],
)
def test_gated_residual_adaln(batch, s_img, s_txt, c, adaln_path):
    _require_cuda()
    gen = make_generator(777)
    eps = 1e-6
    inputs = _gated_residual_adaln_inputs(batch, s_img, s_txt, c, gen)
    leaves, refs = [_leaf(t) for t in inputs], [_leaf(t) for t in inputs]
    out = gated_residual_adaln(*leaves, eps)
    want = _gated_residual_adaln_ref(*refs, eps)
    for got, ref in zip(out, want):
        torch.testing.assert_close(got, ref, **TOL)

    grads = [_randn(t.shape, gen) for t in out]
    torch.autograd.backward(out, grads)
    torch.autograd.backward(want, grads)
    for got, ref in zip(leaves, refs):
        _assert_snr(ref.grad, got.grad)


def test_gated_residual_adaln_rejects_untiled_sequence():
    _require_cuda()
    gen = make_generator(9)
    inputs = _gated_residual_adaln_inputs(1, 64, 48, 1536, gen)
    with pytest.raises(ValueError, match="multiples of 32"):
        gated_residual_adaln(*inputs, 1e-6)


# --- determinism ------------------------------------------------------------------------------


@pytest.mark.deterministic
def test_det_adaln(request, det_seed):
    """Both adaLN ops, fwd + bwd."""
    gen = make_generator(det_seed)
    eps = 1e-6
    modulate0 = _adaln_modulate_inputs(2, 128, 256, 512, gen)
    gated0 = _gated_residual_adaln_inputs(2, 64, 128, 1536, gen)
    g_y, g_r_txt = _randn((2, 384, 512), gen), _randn((2, 128, 512), gen)
    g_gated = [_randn((2, s, 1536), gen) for s in (64, 64, 128, 128)]

    def run_once(mid):
        modulate, gated = [_leaf(t) for t in modulate0], [_leaf(t) for t in gated0]
        r_txt, _, y = adaln_modulate(*modulate, eps)
        out_gated = gated_residual_adaln(*gated, eps)
        mid()
        torch.autograd.backward((y, r_txt, *out_gated), (g_y, g_r_txt, *g_gated))
        res = {"y": y, "y_img": out_gated[1], "y_txt": out_gated[3]}
        res.update({f"modulate{i}": t.grad for i, t in enumerate(modulate)})
        res.update({f"gated{i}": t.grad for i, t in enumerate(gated)})
        return res

    ref = {"y": _adaln_modulate_ref(*(t.float() for t in modulate0), eps)}
    ref["y_img"], ref["y_txt"] = _gated_residual_adaln_ref(*(t.float() for t in gated0), eps)[1::2]
    check_determinism(request.node.nodeid, run_once, ref, snr_floor=35)
