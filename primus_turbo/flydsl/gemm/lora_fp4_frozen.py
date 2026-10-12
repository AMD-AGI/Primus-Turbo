###############################################################################
# SPDX-License-Identifier: Apache-2.0
#
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
###############################################################################
"""Llama-2-70B LoRA (rank 16, alpha 32) on top of the frozen MXFP4 base GEMM.

Folds as much of the adapter's cost as possible into the base GEMM kernel itself,
instead of running `base(x) + scale * ((dropout(x) @ A^T) @ B^T)` as independent
passes the way the bf16-hipBLASLt production path does:

  * Forward: `scale` (= alpha/rank, a power of two here) is folded into the
    `[M, rank]` operand before it reaches the GEMM, so the two extra `[M, N]`
    scale/add passes a naive composition would need disappear -- scaling a bf16
    value by an exact power of two commutes with rounding, so this is free.
  * Forward: `hs @ lora_b^T` and the add onto the base output are fused into the
    base GEMM's own epilogue (`gemm_mxfp4_flydsl_kernel(..., rank_a=hs,
    rank_b=lora_b)` -> `gemm_mxfp4_kernel._rank_prefetch`/`_rank_consume`): one
    v_mfma_f32_16x16x16bf16_1k per 16x16 output sub-block, using the *base's own
    bf16-rounded-then-widened* accumulator as that MFMA's accumulate operand, so
    the result is `bf16(bf16(base) + rank)` -- bit-identical to production's own
    rounding order -- with no separate `[M, N]` read-modify-write pass. See
    `gemm_mxfp4_kernel._rank_consume` for why the base has to be rounded before
    the rank add (seeding the raw fp32 accumulator instead loses a rounding and
    costs real SNR -- measured in this campaign's own history).
  * Backward: the dgrad's `(dh @ A) * mask * keep` term is folded into the base
    dgrad's own Triton accumulate pass (`_k_rank_acc_mask`), again avoiding a
    standalone `[M, K]` pass.

Both A/B gradients (`grad_A = dh^T @ xd`, `grad_B = g^T @ hs`) and the fp32 `dh`
reduction stay as plain bf16 GEMMs: skinny (rank=16) GEMMs at this size are
hipBLASLt-bound, not somewhere a hand-written kernel wins, so they are left
alone rather than fused for fusion's own sake.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

__all__ = ["lora_frozen_linear"]

_RANK_BM = 128
_RANK_BN = 128

_GD_BM = 256
_GD_BN = 128
_GD_NSLICE = 8


@triton.jit
def _k_g_dual(G, B, H, DHP, GBP, M, N, SCALE, R: tl.constexpr, BM: tl.constexpr,
              NSLICE: tl.constexpr, BN: tl.constexpr):
    """One read of grad_out -> DH[M, R] = SCALE * (G @ B) and GB[N, R] = G^T @ H.

    The two reductions run on opposite axes (DH over N, GB over M), so one of them has
    to be carried in registers across the other's loop: DH is, and GB's contribution is
    complete inside each n-chunk for this program's own BM rows, so it goes straight to
    a per-m-block partial with no cross-chunk carry. Both partials are summed by the
    caller. Split partials, not atomics -- the atomic form measured 12x slower.
    """
    pid_m, pid_s = tl.program_id(0), tl.program_id(1)
    rm = pid_m * BM + tl.arange(0, BM)
    rr = tl.arange(0, R)
    hv = tl.load(H + rm[:, None] * R + rr[None, :])
    dh_acc = tl.zeros((BM, R), dtype=tl.float32)
    nper = N // NSLICE
    for n0 in range(pid_s * nper, (pid_s + 1) * nper, BN):
        rn = n0 + tl.arange(0, BN)
        gv = tl.load(G + rm[:, None] * N + rn[None, :])
        bv = tl.load(B + rn[:, None] * R + rr[None, :])
        dh_acc += tl.dot(gv, bv, out_dtype=tl.float32)
        tl.store(GBP + pid_m * N * R + rn[:, None] * R + rr[None, :],
                 tl.dot(tl.trans(gv), hv, out_dtype=tl.float32))
    # scale folded into the accumulator: deletes the standalone `dh * scale` launch,
    # which moves 256 KB and costs ~4.5 us of pure launch latency.
    tl.store(DHP + pid_s * M * R + rm[:, None] * R + rr[None, :], dh_acc * SCALE)


def _g_dual(g, lora_b, hs, scale):
    """(dh, grad_b) from a single pass over grad_out; falls back to two mms off-tile."""
    m, n = g.shape
    r = lora_b.shape[1]
    if m % _GD_BM or n % (_GD_NSLICE * _GD_BN) or r != 16:
        dh = torch.mm(g, lora_b)
        if scale != 1.0:
            dh = dh * scale
        return dh, torch.mm(g.t(), hs)
    gm = m // _GD_BM
    dhp = torch.empty((_GD_NSLICE, m, r), device=g.device, dtype=torch.float32)
    # round-11: tried bf16 here (P0's own named next lever, half the 21 MB fp32 traffic).
    # Measured on the official bench: qkv/proj dB SNR 49.54/49.55, > 0.5 dB below the
    # 50.779/50.781 production floor -- the harness's own gate rejects it (it spends the
    # whole snr_margin_db and then some, not the "exactly equal to baseline" goal.md
    # hoped for). Stays fp32. Next idea on this lever: raise NSLICE instead of narrowing
    # gbp's dtype -- a bigger grid shortens each program's inner loop without touching the
    # partial's precision at all (goal.md P0 "lever (b)").
    gbp = torch.empty((gm, n, r), device=g.device, dtype=torch.float32)
    _k_g_dual[(gm, _GD_NSLICE)](g, lora_b, hs, dhp, gbp, m, n, scale, r,
                                _GD_BM, _GD_NSLICE, _GD_BN, num_warps=8)
    return dhp.sum(0).to(g.dtype), gbp.sum(0).to(g.dtype)


@triton.jit
def _k_rank_acc_mask(OUT, D, A, MASK, M, K, KEEP, R: tl.constexpr, BM: tl.constexpr, BK: tl.constexpr):
    """OUT[M, K] += mask * KEEP * (D[M, R] @ A[R, K])."""
    rm = tl.program_id(0) * BM + tl.arange(0, BM)
    rk = tl.program_id(1) * BK + tl.arange(0, BK)
    rr = tl.arange(0, R)
    d = tl.load(D + rm[:, None] * R + rr[None, :], mask=rm[:, None] < M, other=0.0)
    a = tl.load(A + rr[:, None] * K + rk[None, :], mask=rk[None, :] < K, other=0.0)
    acc = tl.dot(d, a, out_dtype=tl.float32) * KEEP
    off = rm[:, None] * K + rk[None, :]
    msk = (rm[:, None] < M) & (rk[None, :] < K)
    mk = tl.load(MASK + off, mask=msk, other=0).to(tl.int1)
    prev = tl.load(OUT + off, mask=msk, other=0.0)
    tl.store(OUT + off, (tl.where(mk, acc, 0.0) + prev.to(tl.float32)).to(tl.bfloat16), mask=msk)


def _rank_acc_mask(out, d, a, mask, keep):
    m, k = out.shape
    r = d.shape[1]
    grid = (triton.cdiv(m, _RANK_BM), triton.cdiv(k, _RANK_BN))
    _k_rank_acc_mask[grid](out, d, a, mask, m, k, keep, r, _RANK_BM, _RANK_BN)
    return out


def _recipes(use_gradient_sr: bool):
    from primus_turbo.pytorch.core.low_precision import ScalingRecipe

    act_row = ScalingRecipe(use_2d_block=False, use_sr=False, use_rht=False)
    grad_row = ScalingRecipe(use_2d_block=False, use_sr=use_gradient_sr, use_rht=False)
    dropped = ScalingRecipe(use_2d_block=False, use_sr=False, use_rht=False)
    return act_row, grad_row, dropped


def _quantize_row(x, recipe, dropped, config):
    from primus_turbo.pytorch.ops.quantization import quantize_fp4_with_trans

    q, s, _, _ = quantize_fp4_with_trans(
        x,
        torch.float4_e2m1fn_x2,
        config.granularity,
        block_size=config.block_size,
        scaling_recipe=recipe,
        scaling_recipe_for_trans=dropped,
        scale_rounding_mode=config.scale_rounding_mode,
    )
    return q, s


def _gemm_nt(a, a_s, b, b_s, out_dtype, config, rank_a=None, rank_b=None):
    # GEMMFP4FlyDSLBackend.run forwards straight to this with these exact kwargs; going
    # direct is what lets the rank operands through without touching the dispatcher.
    from primus_turbo.flydsl.gemm.gemm_mxfp4_kernel import gemm_mxfp4_flydsl_kernel

    return gemm_mxfp4_flydsl_kernel(
        a,
        a_s,
        b,
        b_s,
        out_dtype=out_dtype,
        trans_c=False,
        beta=0.0,
        out=None,
        scales_prepacked=False,
        k=None,
        rank_a=rank_a,
        rank_b=rank_b,
    )


class _LoRAFrozenFP4(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w_row, w_row_s, w_col, w_col_s, lora_a, lora_b, mask, p, scale, config):
        keep = 1.0 / (1.0 - p)
        act_row, grad_row, dropped = _recipes(config.use_gradient_sr)
        xc = x.contiguous()
        a_q, a_s = _quantize_row(xc, act_row, dropped, config)

        xd = torch.ops.aten.native_dropout_backward(xc, mask, keep)
        # `scale` rides addmm's own alpha instead of a standalone `hs * scale` launch:
        # hipBLASLt folds alpha into the GEMM epilogue (confirmed one kernel, not two, and
        # bit-identical to `mm` then `mul` -- scale is an exact power of two here, so there
        # is no rounding difference either way). beta=0 means `zeros` is never read.
        zeros = torch.zeros(xd.shape[0], lora_a.shape[0], device=xd.device, dtype=xd.dtype)
        hs = torch.addmm(zeros, xd, lora_a.t(), beta=0.0, alpha=scale)
        # Fused: base GEMM + rank-16 adapter product, rounded like production (see module
        # docstring / gemm_mxfp4_kernel._rank_prefetch/_rank_consume).
        out = _gemm_nt(
            a_q, a_s, w_row, w_row_s, torch.bfloat16, config, rank_a=hs, rank_b=lora_b.contiguous()
        )

        ctx.save_for_backward(w_col, w_col_s, xd, hs, lora_a, lora_b, mask)
        ctx.config = config
        ctx.grad_row = grad_row
        ctx.dropped = dropped
        ctx.keep = keep
        ctx.scale = scale
        return out

    @staticmethod
    def backward(ctx, grad_out):
        w_col, w_col_s, xd, hs, lora_a, lora_b, mask = ctx.saved_tensors
        cfg = ctx.config
        g = grad_out.reshape(grad_out.shape[0], -1).contiguous()
        g_q, g_s = _quantize_row(g, ctx.grad_row, ctx.dropped, cfg)
        grad_x = _gemm_nt(g_q, g_s, w_col, w_col_s, torch.bfloat16, cfg)

        # Fused: one read of grad_out feeds both rank-16 reductions, with `scale` folded
        # into dh's accumulator (see `_k_g_dual`).
        dh, grad_b = _g_dual(g, lora_b, hs, ctx.scale)
        grad_a = torch.mm(dh.t(), xd)
        # Fused: dgrad += mask * keep * (dh @ A), folded into the base dgrad's own buffer.
        _rank_acc_mask(grad_x, dh, lora_a, mask, ctx.keep)
        return grad_x, None, None, None, None, grad_a, grad_b, None, None, None, None


def lora_frozen_linear(x, w_row, w_row_s, w_col, w_col_s, lora_a, lora_b, mask, *, p, scale, config):
    """``base(x) + scale * ((x * mask / (1 - p)) @ A^T) @ B^T`` on the frozen MXFP4 base,
    with the adapter folded into the base GEMM's own forward epilogue and backward dgrad
    accumulate (see module docstring)."""
    return _LoRAFrozenFP4.apply(x, w_row, w_row_s, w_col, w_col_s, lora_a, lora_b, mask, p, scale, config)
