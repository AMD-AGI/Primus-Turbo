###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Fused mega MoE MXFP8 forward (FlyDSL): L1 dispatch+fc1 (NT) -> SwiGLU+mxfp8 quant -> L2 fp8 combine.

A plain orchestration function, not a custom_op: the fp8 path carries state no schema can hold --
a live symm buffer the backward reuses, plus non-tensor handles.
"""

from typing import Tuple

import torch
from torch.distributed import ProcessGroup

from primus_turbo.flydsl.mega.fp8 import (
    dispatch_grouped_gemm_mxfp8_flydsl_kernel,
    grouped_gemm_combine_mxfp8_flydsl_kernel,
    swiglu_mxfp8_flydsl_kernel,
)
from primus_turbo.pytorch.kernels.fused_mega_moe.fused_mega_moe_backward_fp8_impl import (
    _w1_fp8_cached,
    _w2_fp8_cached,
    prepare_dw1_pool_operand_fp8,
)

__all__ = [
    "fused_mega_moe_forward_fp8_impl",
]

_H_NUM_TILE_BLOCKS = 11  # fp8 dispatch handle index of num_tile_blocks (device real-tile count)

# The L1 comm/preshuffle split comes from the kernel's per-shape table, so nothing is pinned here.
# L2 combine 32 beats 48 by ~5% on EP8 T=8192 DSv3; the combine kernel has no such table yet.
_L2_NUM_COMBINE_CU = 32


def fused_mega_moe_forward_fp8_impl(
    x: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    group: ProcessGroup,
    save_bwd: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor], dict, tuple]:
    """Fused mxfp8 MoE forward: L1 (dispatch + fc1, NT) -> SwiGLU+mxfp8 quant -> L2 fp8 combine.

    Returns ``(y, l1, dispatch_weights, pool_x_colwise, colwise_meta, handle)``, where ``l1`` is the
    fc1 output [P, 2I] and ``handle`` the dispatch prologue tuple. ``topk_idx`` must already be
    int64 (the op layer converts). ``dispatch_weights`` is a LIVE symm-pool view -- clone it before
    a later stage overwrites it.

    ``pool_x_colwise`` / ``colwise_meta`` are dW1's ``b`` operand (None unless ``save_bwd``), built
    here because the fc1-input pool is still live in the symm buffer: requantizing it colwise now
    consumes that view instead of the clone the backward would otherwise need."""
    # ── L1: fused mxfp8 dispatch + fc1 ──
    l1, handle, dispatch_weights, pool_x_fp8 = dispatch_grouped_gemm_mxfp8_flydsl_kernel(
        x,
        _w1_fp8_cached(w1),
        group,
        topk_idx=topk_idx,
        topk_weights=topk_weights,
    )

    act_fp8, act_a_sp = swiglu_mxfp8_flydsl_kernel(l1, handle[_H_NUM_TILE_BLOCKS])

    w2q, w2s = _w2_fp8_cached(w2)

    # ── L2: fp8 combine (fp8 GEMM + mxfp8 epilogue + fp8 PUSH + bf16-out dequant reduce) ──
    y, _ = grouped_gemm_combine_mxfp8_flydsl_kernel(
        (w2q, w2s), list(handle), group,
        topk_indices=topk_idx,
        topk_weights=topk_weights if topk_weights.dtype == torch.float32 else topk_weights.to(torch.float32),
        x_fp8=(act_fp8, act_a_sp),
        num_combine_cu=_L2_NUM_COMBINE_CU,
    )

    pool_x_colwise, colwise_meta = (
        prepare_dw1_pool_operand_fp8(pool_x_fp8, handle) if save_bwd else (None, None)
    )
    return y, l1, dispatch_weights, pool_x_colwise, colwise_meta, handle
