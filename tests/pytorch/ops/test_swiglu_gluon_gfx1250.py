###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Gluon SwiGLU forward / backward + dual 1x128 FP8 quantization of the fp8_block MegaMoE (gfx1250).

``swiglu_fp8blk_fwd_kernel.swf_gl`` must write the bytes of the Triton ``_swiglu_fwd_dual_kernel`` and
``swiglu_fp8blk_bwd_kernel.swb_gl`` the bytes of the quant2 ``swb_q2`` kernel it replaces (q, row scales, qT, compact
column scales and the d w partials) on the live pool rows: random data, edge values (zeros, denormals, the +-10 clamp,
inf / NaN) and padding rows past the real pool. ``swiglu_fwd_dual_blk`` / ``swiglu_bwd_dual_blk`` take the Gluon
kernels only where they reproduce the reference (default activation, compact column scales); other cases keep the
previous kernels.
"""

import pytest
import torch

from tests.pytorch.ops.gfx1250_helper import requires_gluon

pytestmark = requires_gluon

F8 = torch.float8_e4m3fn
SHAPES = [
    (512, 256, 768),
    (2048, 1024, 2304),
    (28672, 2048, 32768),
]  # (live pool rows P, I, allocated pool rows)
EDGE = [
    0.0,
    -0.0,
    1e-39,
    -1e-39,
    1e-30,
    -3e-38,
    9.99,
    10.0,
    10.5,
    -10.0,
    -12.0,
    1e4,
    -1e4,
    float("inf"),
    float("-inf"),
    float("nan"),
    448.0,
    1e-3,
    65504.0,
    3e38,
]


def _qb():
    from primus_turbo.triton.quantization import quantization_fp8blk_moe_kernel as qb

    return qb


def _gen(shape, seed, scale):
    g = torch.Generator(device="cuda").manual_seed(seed)
    return (torch.randn(shape, device="cuda", generator=g) * scale).to(torch.bfloat16)


def _inputs(P, I, PA, seed, edge):
    l1 = _gen((PA, 2 * I), seed, 2.0)  # x2: the +-10 clamp is active on some values
    g = torch.Generator(device="cuda").manual_seed(seed + 1)
    w = torch.rand(PA, device="cuda", generator=g) + 0.5
    w[P:] = 0
    da = _gen((PA, I), seed + 7, 0.05)
    if edge:
        vals = torch.tensor(EDGE, device="cuda")
        for t in (l1, da):
            m = torch.rand(t.shape, device="cuda", generator=g) < 0.03
            idx = torch.randint(0, len(vals), t.shape, device="cuda", generator=g)
            t[m] = vals[idx[m]].to(torch.bfloat16)
        l1[5] = 0
        l1[7, :I] = 0
        l1[9, I:] = -0.0
        l1[:, 3] = 1e-38
        da[6] = 0
        w[11], w[12], w[13] = 0.0, 1e-40, 3e38
    preal = torch.tensor([P], dtype=torch.int32, device="cuda")
    return da, l1, w, preal


def _live_equal(a, b, P):
    q, s, qT, sT = a[:4]
    q2, s2, qT2, sT2 = b[:4]
    u8 = torch.uint8
    return [
        torch.equal(q[:P].view(u8), q2[:P].view(u8)),
        torch.equal(s[:P], s2[:P]),
        torch.equal(qT[:, :P].view(u8), qT2[:, :P].view(u8)),
        torch.equal(sT[:, : P // 128], sT2[:, : P // 128]),
    ]


def _spy(monkeypatch, mod, name):
    calls = []
    orig = getattr(mod, name)
    monkeypatch.setattr(mod, name, lambda *a, **k: (calls.append(1), orig(*a, **k))[1])
    return calls


@pytest.mark.parametrize("edge", [False, True], ids=["random", "edge"])
@pytest.mark.parametrize("seed", [1, 2])
@pytest.mark.parametrize("P, I, PA", SHAPES, ids=[f"P{p}-I{i}" for p, i, _ in SHAPES])
def test_swiglu_fwd_gluon_bits_equal_triton(monkeypatch, P, I, PA, seed, edge):
    from primus_turbo.gluon.activation import swiglu_fp8blk_fwd_kernel as swf

    qb = _qb()
    _, l1, w, preal = _inputs(P, I, PA, seed, edge)
    calls = _spy(monkeypatch, swf, "swf_gl")
    got = qb.swiglu_fwd_dual_blk(l1, w, None, preal=preal, compact_t=True)
    torch.cuda.synchronize()
    assert calls, "swiglu_fwd_dual_blk did not take the Gluon kernel"
    monkeypatch.setattr(qb, "gluon_ok", lambda: False)  # the Triton kernel
    ref = qb.swiglu_fwd_dual_blk(l1, w, None, preal=preal, compact_t=True)
    torch.cuda.synchronize()
    assert all(_live_equal(ref, got, P)), _live_equal(ref, got, P)


@pytest.mark.parametrize("edge", [False, True], ids=["random", "edge"])
@pytest.mark.parametrize("seed", [3, 4])
@pytest.mark.parametrize("P, I, PA", SHAPES, ids=[f"P{p}-I{i}" for p, i, _ in SHAPES])
def test_swiglu_bwd_gluon_bits_equal_swb_q2(monkeypatch, P, I, PA, seed, edge):
    from primus_turbo.flydsl.utils.glu_activation import activation_constexpr
    from primus_turbo.gluon.activation import swiglu_fp8blk_bwd_kernel as swb
    from primus_turbo.gluon.quantization.quantization_fp8blk_dual_kernel import swb_q2

    qb = _qb()
    da, l1, w, preal = _inputs(P, I, PA, seed, edge)
    calls = _spy(monkeypatch, swb, "swb_gl")
    got = qb.swiglu_bwd_dual_blk(da, l1, w, None, preal=preal, compact_t=True, raw_dw=True)
    torch.cuda.synchronize()
    assert calls, "swiglu_bwd_dual_blk did not take the Gluon kernel"
    q, s, qT, sT = qb._dual_alloc(PA, 2 * I, F8, 0, True, l1.device)
    gw = torch.zeros((PA, I // 128), device="cuda", dtype=torch.float32)
    swb_q2(da, l1, w, q, s, qT, sT, gw, preal, *activation_constexpr(None))
    torch.cuda.synchronize()
    assert all(_live_equal((q, s, qT, sT), got, P)), _live_equal((q, s, qT, sT), got, P)
    a, b = gw[:P], got[4][:P]
    differ = a.view(torch.int32) != b.view(torch.int32)
    # inf / NaN inputs: only NaN payload bits may differ (the replaced kernels differ the same way)
    assert not (differ & ~(torch.isnan(a) & torch.isnan(b))).any()


def test_swiglu_gluon_fallbacks(monkeypatch):
    """Cases the Gluon kernels do not reproduce keep the previous kernels."""
    from primus_turbo.flydsl.utils.glu_activation import GLUActivation
    from primus_turbo.gluon.activation import swiglu_fp8blk_bwd_kernel as swb
    from primus_turbo.gluon.activation import swiglu_fp8blk_fwd_kernel as swf

    qb = _qb()
    da, l1, w, preal = _inputs(1024, 512, 1280, 5, False)
    fwd = _spy(monkeypatch, swf, "swf_gl")
    bwd = _spy(monkeypatch, swb, "swb_gl")
    for kw in (dict(activation=GLUActivation.swigluoai()), dict(compact_t=False)):
        qb.swiglu_fwd_dual_blk(l1, w, preal=preal, **{"compact_t": True, **kw})
        qb.swiglu_bwd_dual_blk(da, l1, w, preal=preal, **{"compact_t": True, "raw_dw": True, **kw})
    torch.cuda.synchronize()
    assert not fwd and not bwd
