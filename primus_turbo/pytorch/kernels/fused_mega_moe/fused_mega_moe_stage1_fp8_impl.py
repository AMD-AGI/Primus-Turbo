###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Two-stage mega MoE stage1 (gate-up) MXFP8 FlyDSL kernel composition.

Stage1 owns the forward dispatch + fc1 and the fc1-input pool requant for dW1; on the backward, the
L1 dgrad and the dW1 wgrad. Every call below is a helper the fused path already uses.
"""

from typing import Optional, Tuple

import torch

from primus_turbo.flydsl.mega.fp8 import dispatch_grouped_gemm_mxfp8_flydsl_kernel
from primus_turbo.pytorch.kernels.fused_mega_moe.fused_mega_moe_backward_fp8_impl import (
    _l1_dgrad_combine_mxfp8_flydsl_kernel,
    _mxfp8_variable_k_wgrad_dw1,
    _w1_fp8_cached,
    prepare_dw1_pool_operand_fp8,
)

__all__ = [
    "fused_mega_moe_stage1_forward_fp8_impl",
    "fused_mega_moe_stage1_backward_fp8_impl",
]

# fp8 dispatch handle: dispatch_prologue's 11 tables + num_tile_blocks appended by the L1 kernel.
_HANDLE_LEN = 14


def fused_mega_moe_stage1_forward_fp8_impl(
    x: torch.Tensor,
    w1: torch.Tensor,
    group,
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    save_bwd: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, tuple, Optional[Tuple[torch.Tensor, torch.Tensor]], Optional[dict]]:
    """dispatch + grouped fc1 GEMM (nt, mxfp8).

    Returns ``(l1, dispatch_weights, handle, pool_x_colwise, colwise_meta)``. ``dispatch_weights`` is
    cloned because stage2's backward needs it as the SwiGLU^T scale long after later stages overwrite
    the symm buffer. ``pool_x_colwise`` / ``colwise_meta`` (None unless ``save_bwd``) are dW1's ``b``
    operand, requantized here while the fc1-input pool is still live in symm.
    """
    # int64 end-to-end (combine reads topk i64)
    topk_idx = topk_idx.to(torch.int64)
    l1, handle, dispatch_weights, pool_x_fp8 = dispatch_grouped_gemm_mxfp8_flydsl_kernel(
        x,
        _w1_fp8_cached(w1),
        group,
        topk_idx=topk_idx,
        topk_weights=topk_weights,
    )
    assert len(handle) == _HANDLE_LEN, f"fp8 dispatch handle len {len(handle)} != {_HANDLE_LEN}; ABI changed"

    pool_x_colwise, colwise_meta = (
        prepare_dw1_pool_operand_fp8(pool_x_fp8, handle) if save_bwd else (None, None)
    )
    return l1, dispatch_weights.clone(), tuple(handle), pool_x_colwise, colwise_meta


def fused_mega_moe_stage1_backward_fp8_impl(
    grad_l1_rowwise_fp8: Tuple[torch.Tensor, torch.Tensor],
    grad_l1_colwise_fp8: Tuple[torch.Tensor, torch.Tensor],
    grad_gate: torch.Tensor,
    pool_x_colwise_fp8: Tuple[torch.Tensor, torch.Tensor],
    colwise_meta: dict,
    w1: torch.Tensor,
    handle: tuple,
    group,
    topk_idx: torch.Tensor,
    num_tokens: int,
    num_topk: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """L1 dgrad combine + dW1 (variable-K wgrad).

    Returns ``(dx, grad_topk_weights, dW1)``. Both grad_l1 operands arrive pre-quantized from
    stage2's fused SwiGLU^T dual-quant, which is why they come through the state side channel and
    not the ``l1`` gradient slot. The two calls stay serial on the default stream.
    """
    dx, grad_topk_weights = _l1_dgrad_combine_mxfp8_flydsl_kernel(
        w1, group, handle,
        grad_l1_rowwise_fp8=grad_l1_rowwise_fp8,
        grad_gate=grad_gate,
        topk_idx=topk_idx,
        num_tokens=num_tokens,
        num_topk=num_topk,
    )
    dW1 = _mxfp8_variable_k_wgrad_dw1(grad_l1_colwise_fp8, pool_x_colwise_fp8, colwise_meta)
    return dx, grad_topk_weights, dW1
