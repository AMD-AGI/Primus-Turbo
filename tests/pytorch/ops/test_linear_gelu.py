###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import pytest
import torch
import torch.nn.functional as F

from primus_turbo.pytorch.ops.activation import linear_gelu
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


def _linear_gelu_ref(x, w, b, prefix):
    h = F.gelu(F.linear(x, w, b), approximate="tanh")
    return h if prefix is None else torch.cat([prefix, h], dim=1)


@pytest.mark.parametrize(
    "m,k,n,p",
    [
        (2048, 128, 3072, 0),  # FlyDSL GELU on gfx950
        (2048, 128, 3072, 3072),  # ... written after a prefix
        (96, 64, 160, 0),  # unfused GELU: shapes the kernel does not tile
        (96, 64, 160, 32),
    ],
)
def test_linear_gelu(m, k, n, p):
    _require_cuda()
    gen = make_generator(1234)
    x, w, b = _randn((m, k), gen), _randn((n, k), gen, k**-0.5), _randn((n,), gen, 0.1)
    prefix = _randn((m, p), gen) if p else None
    leaves = [_leaf(t) for t in (x, w, b)] + ([_leaf(prefix)] if p else [])
    refs = [_leaf(t) for t in (x, w, b)] + ([_leaf(prefix)] if p else [])

    out = linear_gelu(*leaves[:3], leaves[3] if p else None)
    ref = _linear_gelu_ref(*refs[:3], refs[3] if p else None)
    torch.testing.assert_close(out, ref, **TOL)
    if p:
        assert torch.equal(out[:, :p], prefix)

    grad = _randn(out.shape, gen)
    out.backward(grad)
    ref.backward(grad)
    for got, want in zip(leaves, refs):
        _assert_snr(want.grad, got.grad)


def test_linear_gelu_rejects_mismatched_bias():
    _require_cuda()
    x = torch.zeros((8, 16), device="cuda", dtype=torch.bfloat16)
    w = torch.zeros((32, 16), device="cuda", dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="do not match"):
        linear_gelu(x, w, torch.zeros(31, device="cuda", dtype=torch.bfloat16))


def test_linear_gelu_rejects_fp32():
    _require_cuda()
    x, w, b = (torch.zeros(s, device="cuda") for s in ((8, 16), (32, 16), (32,)))
    with pytest.raises(ValueError, match="BF16"):
        linear_gelu(x, w, b)


@pytest.mark.deterministic
def test_det_linear_gelu(request, det_seed):
    """Linear + GELU with a prefix, fwd + bwd."""
    require_gfx950("the FlyDSL GELU")
    gen = make_generator(det_seed)
    x0, w0, b0 = _randn((2048, 128), gen), _randn((3072, 128), gen, 128**-0.5), _randn((3072,), gen, 0.1)
    p0, g = _randn((2048, 3072), gen), _randn((2048, 6144), gen)

    def run_once(mid):
        x, w, b, p = (_leaf(t) for t in (x0, w0, b0, p0))
        out = linear_gelu(x, w, b, p)
        mid()
        out.backward(g)
        return {"out": out, "dx": x.grad, "dw": w.grad, "db": b.grad, "dp": p.grad}

    ref = {"out": _linear_gelu_ref(x0.float(), w0.float(), b0.float(), p0.float())}
    check_determinism(request.node.nodeid, run_once, ref, snr_floor=35)
