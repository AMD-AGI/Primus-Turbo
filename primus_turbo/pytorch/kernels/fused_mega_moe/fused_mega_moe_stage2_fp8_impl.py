###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Two-stage mega MoE stage2 (gate-down) MXFP8 FlyDSL kernel composition.

Stage2 owns the forward SwiGLU + fc2 combine; on the backward, the L2 dgrad, the fused SwiGLU^T
dual-quant, and the dW2 wgrad. The dual-quant emits grad_l1 only as quantized operand pairs, so the
backward returns them for the op layer to hand to stage1 out of band.
"""

from typing import Tuple

import torch

from primus_turbo.flydsl.mega.fp8 import (
    colwise_grouped_meta,
    colwise_requant_fp8in_and_quant_bf16_grouped_flydsl,
    combine_l2_fwd_mxfp8_flydsl_kernel,
    dispatch_l2_dgrad_mxfp8_flydsl_kernel,
    swiglu_bwd_rowcol_dual_quant_mxfp8_flydsl,
    swiglu_mxfp8_flydsl_kernel,
)
from primus_turbo.pytorch.core.backend import BackendType
from primus_turbo.pytorch.core.low_precision import ScalingGranularity
from primus_turbo.pytorch.kernels.fused_mega_moe.mega_moe_fp8_weights import (
    _DW_FP8_FORMAT,
    _w2_fp8_cached,
    _w2t_fp8_cached,
)
from primus_turbo.pytorch.kernels.grouped_gemm.grouped_gemm_fp8_impl import (
    grouped_gemm_fp8_variable_k_impl,
)

_HANDLE_GROUP_LENS = 9  # fp8 dispatch handle index: per-local-expert real token counts
_HANDLE_GROUP_OFFS = 10  # ... and their prefix over the block_m-padded pool

_H_NUM_TILE_BLOCKS = 11  # fp8 dispatch handle index of num_tile_blocks (device real-tile count)

# The L1 comm/preshuffle split is left to the dispatch kernel's signature default, whose measured
# pair is that direction's. L2 combine 32 beats 48 by ~5% on EP8 T=8192 DSv3.
_L2_NUM_COMBINE_CU = 32

__all__ = [
    "fused_mega_moe_stage2_forward_fp8_impl",
    "fused_mega_moe_stage2_backward_fp8_impl",
]


def _dispatch_l2_dgrad_fp8(dy, w2, handle, *, w2t_fp8=None):
    """``w2^T`` prep in front of ``dispatch_l2_dgrad_mxfp8_flydsl_kernel``; the kernel needs no group,
    since the forward's handle already carries the routing."""
    w2t = w2t_fp8 if w2t_fp8 is not None else _w2t_fp8_cached(w2)
    return dispatch_l2_dgrad_mxfp8_flydsl_kernel(dy, w2t, handle)


def _mxfp8_variable_k_wgrad_dw2(a_fp8, b_bf16, group_lens, group_offs, meta=None):
    """dW2 = ``a^T @ b`` (variable-K over the dispatched pool tokens), MXFP8 -> ``[G, H, *]`` bf16.

    ``a`` is the L2-dgrad dispatched-dy pool, requantized colwise directly from fp8 with no bf16
    round-trip; ``b`` is ``act_weighted``. One dual-launch produces both transposed operands."""
    pool_fp8, pool_scale = a_fp8
    if meta is None:
        meta = colwise_grouped_meta(group_lens, group_offs, pool_rows=pool_fp8.shape[0])
    a_t, a_ts, b_t, b_ts, lens_pc, offs_pc = colwise_requant_fp8in_and_quant_bf16_grouped_flydsl(
        pool_fp8,
        pool_scale,
        b_bf16,
        _DW_FP8_FORMAT,
        meta=meta,
    )
    return grouped_gemm_fp8_variable_k_impl(
        a_t,
        b_t,
        a_ts.view(torch.float8_e8m0fnu),
        b_ts.view(torch.float8_e8m0fnu),
        lens_pc.to(torch.int64),
        offs_pc.to(torch.int64),
        trans_a=False,
        trans_b=False,
        trans_c=False,
        out_dtype=torch.bfloat16,
        granularity=ScalingGranularity.MX_BLOCKWISE.value,
        num_cu=None,
        default_backend=BackendType.FLYDSL.value,
    )


def fused_mega_moe_stage2_forward_fp8_impl(
    l1: torch.Tensor,
    w2: torch.Tensor,
    handle: tuple,
    group,
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
) -> torch.Tensor:
    """SwiGLU + mxfp8 quant + grouped fc2 GEMM + combine (nt). Returns y."""
    topk_idx = topk_idx.to(torch.int64)

    # bound swiglu by THIS handle's tile count (per-forward, not shared symm)
    act_fp8, act_a_sp = swiglu_mxfp8_flydsl_kernel(l1, handle[_H_NUM_TILE_BLOCKS])

    w2q, w2s = _w2_fp8_cached(w2)

    # fused grouped fc2 GEMM + mxfp8 epilogue + fp8 combine PUSH + bf16 topk reduce
    y = combine_l2_fwd_mxfp8_flydsl_kernel(
        (w2q, w2s),
        list(handle),
        topk_indices=topk_idx,
        topk_weights=topk_weights if topk_weights.dtype == torch.float32 else topk_weights.to(torch.float32),
        x_fp8=(act_fp8, act_a_sp),
        num_combine_cu=_L2_NUM_COMBINE_CU,
    )
    return y


def fused_mega_moe_stage2_backward_fp8_impl(
    grad_y: torch.Tensor,
    l1: torch.Tensor,
    dispatch_weights: torch.Tensor,
    w2: torch.Tensor,
    handle: tuple,
    group,
    colwise_meta: dict,
) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor], torch.Tensor, torch.Tensor]:
    """L2 dgrad + fused SwiGLU^T dual-quant + dW2 (variable-K wgrad).

    Returns ``(grad_l1_rowwise_fp8, grad_l1_colwise_fp8, grad_gate, dW2)`` -- the rowwise pair feeds
    stage1's L1 dgrad, the colwise pair its dW1. ``colwise_meta`` is stage1's, reused here so the
    dual-quant and dW2 skip a second D2H of the same group offsets.
    """
    group_lens = handle[_HANDLE_GROUP_LENS]
    group_offs = handle[_HANDLE_GROUP_OFFS]
    dy = grad_y.contiguous().to(torch.bfloat16)

    # L2 dgrad: dispatch(dy) + fc2 -> grad_swiglu + the dispatched-dy pool (the dW2 `a` operand)
    grad_swiglu, dispatch_l2_grad_fp8 = _dispatch_l2_dgrad_fp8(dy, w2, handle)

    # SwiGLU^T (re-inject routing weight) + gate grad, fused with grad_l1's rowwise (L1 dgrad) and
    # colwise (dW1) quant -- grad_l1 has no other consumer, so it never reaches HBM
    (
        gl1_q_row,
        gl1_a_sp,
        gl1_q_col,
        gl1_s_col,
        grad_gate,
        act_weighted,
    ) = swiglu_bwd_rowcol_dual_quant_mxfp8_flydsl(
        grad_swiglu,
        l1,
        dispatch_weights,
        _DW_FP8_FORMAT,
        meta=colwise_meta,
    )

    # dW2 = dispatched(dy)^T @ act_weighted -- must run before anything overwrites symm.pool_fp8
    dW2 = _mxfp8_variable_k_wgrad_dw2(
        dispatch_l2_grad_fp8,
        act_weighted,
        group_lens,
        group_offs,
        meta=colwise_meta,
    )

    return (gl1_q_row, gl1_a_sp), (gl1_q_col, gl1_s_col), grad_gate, dW2
