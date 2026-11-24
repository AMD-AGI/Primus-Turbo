###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Concat-N dY pre-quantization slot (DyPrequantSlot): the op that consumes both concat-N outputs quantizes their
gradients in its own backward launch, together with a column reduction of its own.

Checked: ``dy_prequant_fill`` writes the bytes of ``_quant_cat_cols`` and the reduction of a reference kernel with
the same summation order; the slot is found by output pointers (both orders) and consumed once; the concat backward
uses the pre-quantized dY only when it receives exactly the quantized tensors, and gives the same dX / main_grad
bytes either way.
"""

import pytest
import torch
import triton
import triton.language as tl

from primus_turbo.pytorch.core.low_precision import Float8QuantConfig, Format, ScalingGranularity
from tests.pytorch.ops.gfx1250_helper import bits_equal, requires_gluon

CFG = Float8QuantConfig(format=Format.E4M3, granularity=ScalingGranularity.BLOCKWISE, block_size=128)
F8 = torch.float8_e4m3fn


@triton.jit
def _ref_colsum_kernel(DGP, DGO, P, H: tl.constexpr, BH: tl.constexpr, BP: tl.constexpr):
    """Reference column sum: [BP, BH] tile accumulated over P in steps of BP, then one tl.sum over the tile rows."""
    cols = tl.program_id(0) * BH + tl.arange(0, BH)
    acc = tl.zeros((BP, BH), dtype=tl.float32)
    for p0 in range(0, P, BP):
        pr = p0 + tl.arange(0, BP)
        acc += tl.load(DGP + pr[:, None] * H + cols[None, :], mask=(pr[:, None] < P), other=0.0)
    tl.store(DGO + cols, tl.sum(acc, axis=0).to(DGO.dtype.element_ty))


def _ref_colsum(dgp, dgo):
    P, H = dgp.shape
    _ref_colsum_kernel[(H // 64,)](dgp, dgo, P, H=H, BH=64, BP=64, num_warps=4)


def _use_dsv4up(monkeypatch):
    monkeypatch.setenv("PRIMUS_TURBO_FP8_BLOCK_UE8M0", "1")
    monkeypatch.setenv("PRIMUS_TURBO_FP8_BLOCK_KERNEL", "dsv4up")
    monkeypatch.setenv("PRIMUS_TURBO_DSV4UP_PREFER_G90", "")
    monkeypatch.delenv("PRIMUS_TURBO_FP8_WCACHE", raising=False)


@requires_gluon
@pytest.mark.parametrize("seed", [1, 2, 3])
@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("m, k, h0, h1, p", [(4096, 4096, 512, 1024, 256), (1024, 1024, 512, 512, 100)])
def test_fill_bits_equal_quant_cat_cols_and_reference_reduce(seed, out_dtype, m, k, h0, h1, p):
    from primus_turbo.pytorch.ops import gemm_fp8_concat as gc

    g = torch.Generator(device="cuda").manual_seed(seed)

    def r(*sh):
        return torch.randn(*sh, device="cuda", generator=g)

    x0, x1 = (r(m, h0) * 0.02).bfloat16(), (r(m, h1) * 0.01).bfloat16()
    x1[5, :] = 0  # an all-zero row
    ts = [r(k, h0).to(F8).contiguous(), r(k, h1).to(F8).contiguous()]  # per-weight W^T copies
    dgp0, dgp1 = r(p, h0), r(p, h1)

    ref_q = gc._quant_cat_cols([x0, x1], F8, ts)
    ref_o = [
        torch.full((h0,), 3.0, device="cuda", dtype=out_dtype),
        torch.full((h1,), 3.0, device="cuda", dtype=out_dtype),
    ]
    _ref_colsum(dgp0, ref_o[0])
    _ref_colsum(dgp1, ref_o[1])

    slot = gc.DyPrequantSlot(F8, ts, (0, 0))
    new_o = [
        torch.full((h0,), 5.0, device="cuda", dtype=out_dtype),
        torch.full((h1,), 5.0, device="cuda", dtype=out_dtype),
    ]
    assert gc.dy_prequant_fill(slot, [x0, x1], (dgp0, new_o[0], dgp1, new_o[1], p))
    (qr, er, qt, et), wt = slot.result
    (rqr, rer, rqt, ret), rwt = ref_q
    for a, b in (
        (qr, rqr),
        (er, rer),
        (qt, rqt),
        (et, ret),
        (wt, rwt),
        (new_o[0], ref_o[0]),
        (new_o[1], ref_o[1]),
    ):
        assert bits_equal(a, b)
    assert torch.allclose(new_o[0].float(), dgp0.sum(0).to(out_dtype).float(), rtol=1e-2, atol=1e-2)


@requires_gluon
def test_fill_rejects_unsupported_shapes():
    from primus_turbo.pytorch.ops import gemm_fp8_concat as gc

    ts = [torch.zeros(256, 256, device="cuda").to(F8), torch.zeros(256, 256, device="cuda").to(F8)]
    slot = gc.DyPrequantSlot(F8, ts, (0, 0))
    x = torch.zeros(256, 256, device="cuda", dtype=torch.bfloat16)
    dgp, dgo = torch.zeros(4, 96, device="cuda"), torch.zeros(96, device="cuda")
    assert not gc.dy_prequant_fill(slot, [x, x], (dgp, dgo, dgp, dgo, 4))  # H % 64 != 0
    assert not gc.dy_prequant_fill(
        slot, [x[:100], x[:100]], (dgp[:, :64], dgo[:64], dgp[:, :64], dgo[:64], 4)
    )
    assert slot.result is None


def _concat_setup(monkeypatch):
    from primus_turbo.pytorch.core import fp8blk_weight_cache as wcache
    from primus_turbo.pytorch.kernels.quantization.quantization_impl import (
        quant_fp8_blockwise_for_weight_impl,
    )
    from primus_turbo.pytorch.ops.gemm_fp8_concat import gemm_fp8_block_concat_n_supported

    _use_dsv4up(monkeypatch)
    # The slot carries the per-weight W^T copies, which exist only with the FP8 weight cache (as in training, where
    # the optimizer fills it); VERIFY re-quantizes every cache hit and raises on a byte difference.
    monkeypatch.setenv("PRIMUS_TURBO_FP8_WCACHE", "1")
    monkeypatch.setenv("PRIMUS_TURBO_FP8_WCACHE_VERIFY", "1")
    t, k, n0, n1 = 1024, 1024, 512, 512
    torch.manual_seed(0)
    x = (torch.randn(t, k, device="cuda") * 0.5).bfloat16()
    ws = [(torch.randn(n, k, device="cuda") * 0.02).bfloat16().requires_grad_(True) for n in (n0, n1)]
    for w in ws:  # what the optimizer writes after its update
        q, sc = quant_fp8_blockwise_for_weight_impl(w.detach().contiguous(), F8, block_size=128)
        e = ((sc.contiguous().view(torch.int32) >> 23) & 0xFF).to(torch.uint8)
        cache = wcache.attach(w, F8)
        cache.w.copy_(q)
        cache.scale_inv.copy_(sc)
        cache.e8m0.copy_(e)
        cache.wt.copy_(q.t())
        cache.e8m0_t.copy_(e.t())
        cache.ready = True
    gys = [(torch.randn(t, n, device="cuda") * 0.1).bfloat16() for n in (n0, n1)]
    mg0 = [torch.randn(n, k, device="cuda") * 0.01 for n in (n0, n1)]
    if not gemm_fp8_block_concat_n_supported(x, ws, CFG):
        pytest.skip("concat-N op not enabled for this configuration")
    return x, ws, gys, mg0


def _concat_step(x, ws, gys, mg0, fill):
    """Forward + backward of the concat-N op; ``fill(ys)`` runs between them (the consumer's role)."""
    from primus_turbo.pytorch.ops.gemm_fp8_concat import gemm_fp8_block_concat_n

    for w, m0 in zip(ws, mg0):
        w.main_grad = m0.clone()
        w.grad = None
        w.grad_added_to_main_grad = False
    xc = x.clone().requires_grad_(True)
    ys = gemm_fp8_block_concat_n(xc, ws, CFG, ["megatron", "megatron"])
    grads = fill(ys)
    torch.autograd.backward(list(ys), grads)
    torch.cuda.synchronize()
    return xc.grad.clone(), [w.main_grad.clone() for w in ws]


@requires_gluon
@pytest.mark.parametrize("swapped", [False, True])
@pytest.mark.parametrize("same_tensors", [True, False], ids=["prequantized", "fallback"])
def test_concat_backward_takes_prequantized_dy(monkeypatch, swapped, same_tensors):
    from primus_turbo.pytorch.ops import gemm_fp8_concat as gc

    x, ws, gys, mg0 = _concat_setup(monkeypatch)
    ref = _concat_step(x, ws, gys, mg0, lambda ys: gys)

    calls = []
    orig = gc._quant_cat_cols
    monkeypatch.setattr(gc, "_quant_cat_cols", lambda *a: calls.append(1) or orig(*a))
    dgp = torch.randn(64, 128, device="cuda")
    dgo = [torch.empty(128, device="cuda"), torch.empty(128, device="cuda")]

    def fill(ys):
        slot, sw = gc.take_dy_prequant(ys[1], ys[0]) if swapped else gc.take_dy_prequant(ys[0], ys[1])
        assert slot is not None and sw == swapped
        assert gc.take_dy_prequant(ys[0], ys[1]) == (None, False)  # consumed once
        assert gc.dy_prequant_fill(slot, gys, (dgp, dgo[0], dgp, dgo[1], 64))
        # the backward reuses the result only for exactly the quantized tensors
        return gys if same_tensors else [g.clone() for g in gys]

    dx, mgs = _concat_step(x, ws, gys, mg0, fill)
    assert len(calls) == (0 if same_tensors else 1)
    assert bits_equal(dx, ref[0])
    for a, b in zip(mgs, ref[1]):
        assert bits_equal(a, b)


@requires_gluon
def test_take_dy_prequant_ignores_other_tensors(monkeypatch):
    from primus_turbo.pytorch.ops import gemm_fp8_concat as gc
    from primus_turbo.pytorch.ops.gemm_fp8_concat import gemm_fp8_block_concat_n

    x, ws, _, _ = _concat_setup(monkeypatch)
    with torch.no_grad():
        ys = gemm_fp8_block_concat_n(x, ws, CFG, [None, None])
    other = torch.empty_like(ys[0])
    assert gc.take_dy_prequant(other, ys[1]) == (None, False)
    assert gc.take_dy_prequant(ys[0], ys[1]) == (None, False)  # the failed lookup consumed the slot
