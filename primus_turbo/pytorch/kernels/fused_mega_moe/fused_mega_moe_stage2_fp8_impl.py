###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Two-stage mega MoE stage2 (gate-down) MXFP8 FlyDSL kernel composition.

fp8 sibling of ``fused_mega_moe_stage2_impl``; see ``fused_mega_moe_stage1_fp8_impl`` for the split
rationale. Stage2 owns: forward SwiGLU + mxfp8 quant + fc2 fp8 combine -> y; backward L2 dgrad
(dispatch(dy) + fc2), the fused SwiGLU^T row/col dual-quant, and the dW2 variable-K wgrad.

The dual-quant kernel emits grad_l1 ONLY as the two quantized operands STEP3 and dW1 consume -- no
bf16 grad_l1 ever exists. Those uint8/int32 tensors cannot ride a gradient slot, so the backward
returns them to the op layer, which hands them to stage1 through an opaque state object.
"""

from typing import Tuple

import torch

from primus_turbo.flydsl.mega.fp8 import (
    grouped_gemm_combine_mxfp8_flydsl_kernel,
    swiglu_bwd_rowcol_dual_quant_mxfp8_flydsl,
    swiglu_mxfp8_flydsl_kernel,
)
from primus_turbo.pytorch.kernels.fused_mega_moe.fused_mega_moe_backward_fp8_impl import (
    _DW_FP8_FORMAT,
    _HANDLE_GROUP_LENS,
    _HANDLE_GROUP_OFFS,
    _dispatch_l2_dgrad_mxfp8_flydsl_kernel,
    _mxfp8_variable_k_wgrad_dw2,
)
from primus_turbo.pytorch.kernels.fused_mega_moe.fused_mega_moe_forward_fp8_impl import (
    _H_NUM_TILE_BLOCKS,
    _L2_NUM_COMBINE_CU,
    _w2_fp8_cached,
)

__all__ = [
    "fused_mega_moe_stage2_forward_fp8_impl",
    "fused_mega_moe_stage2_backward_fp8_impl",
]


def fused_mega_moe_stage2_forward_fp8_impl(
    l1: torch.Tensor,
    w2: torch.Tensor,
    handle: tuple,
    group,
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    block_m: int,
    block_n: int,
) -> torch.Tensor:
    """SwiGLU + mxfp8 quant -> fp8 fc2 combine (fp8 GEMM + mxfp8 epilogue + fp8 PUSH + bf16 reduce).

    ``handle`` is stage1's dispatch prologue tuple; its ``num_tile_blocks`` entry bounds the SwiGLU
    to this forward's real tiles.
    """
    topk_idx = topk_idx.to(torch.int64)

    act_fp8, act_a_sp = swiglu_mxfp8_flydsl_kernel(l1, handle[_H_NUM_TILE_BLOCKS])

    w2q, w2s = _w2_fp8_cached(w2)

    y, _ = grouped_gemm_combine_mxfp8_flydsl_kernel(
        None, (w2q, w2s), list(handle), group,
        topk_indices=topk_idx,
        topk_weights=topk_weights if topk_weights.dtype == torch.float32 else topk_weights.to(torch.float32),
        x_fp8=(act_fp8, act_a_sp),
        BM=block_m, BN=block_n,
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
    block_m: int,
    block_n: int,
) -> Tuple[
    Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor], torch.Tensor, torch.Tensor
]:
    """L2 dgrad -> fused SwiGLU^T row/col dual-quant -> dW2 variable-K wgrad.

    Returns ``(grad_l1_rowwise_fp8, grad_l1_colwise_fp8, grad_gate, dW2)``: the rowwise pair feeds
    stage1's STEP3 combine, the colwise pair feeds stage1's dW1, and ``grad_gate`` [P] is scattered
    into ``grad_topk_weights`` there. dW2 comes back in the wgrad's bf16 accumulate dtype.

    ``colwise_meta`` is stage1's (the grouped padded offsets from its pool requant) -- reused here
    so the dual-quant and dW2 skip a second D2H of the same group offsets.
    """
    group_lens = handle[_HANDLE_GROUP_LENS]
    group_offs = handle[_HANDLE_GROUP_OFFS]
    dy = grad_y.contiguous().to(torch.bfloat16)

    # L2 dgrad (fp8 fork): dispatch(dy) + fc2 -> grad_swiglu + the dispatched-dy pool in native
    # rowwise-fp8 (the dW2 `a` operand, requant colwise directly from fp8).
    grad_swiglu, dispatch_l2_grad_fp8 = _dispatch_l2_dgrad_mxfp8_flydsl_kernel(
        dy, w2, group, handle, block_m, block_n
    )

    # SwiGLU^T with the routing weight re-injected, fused with the rowwise (STEP3) and colwise
    # (dW1) mxfp8 quant of grad_l1 -- which has no other consumer, so it never reaches HBM.
    (
        gl1_q_row, gl1_a_sp, gl1_q_col, gl1_s_col, grad_gate, act_weighted,
    ) = swiglu_bwd_rowcol_dual_quant_mxfp8_flydsl(
        grad_swiglu, l1, dispatch_weights, _DW_FP8_FORMAT, meta=colwise_meta,
    )

    # dW2 before anything else overwrites symm.pool_fp8.
    dW2 = _mxfp8_variable_k_wgrad_dw2(
        dispatch_l2_grad_fp8, act_weighted, group_lens, group_offs, meta=colwise_meta,
    )

    return (gl1_q_row, gl1_a_sp), (gl1_q_col, gl1_s_col), grad_gate, dW2
