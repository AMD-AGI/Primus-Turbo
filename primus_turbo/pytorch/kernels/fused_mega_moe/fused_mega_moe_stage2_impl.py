###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Two-stage MegaMoE stage2 (gate-down) FlyDSL kernel composition."""

import torch

from primus_turbo.flydsl.gemm.gemm_bf16_kernel import grouped_gemm_variable_k_bf16
from primus_turbo.flydsl.mega import (
    dispatch_grouped_gemm_bf16_flydsl_kernel,
    grouped_gemm_combine_bf16_flydsl_kernel,
    swiglu_backward_flydsl_kernel,
    swiglu_flydsl_kernel,
)

# dispatch handle layout (see dispatch_prologue ABI).
_H_NUM_TILE_BLOCKS = 8
_H_REAL_COUNT_PER_EXPERT = 6
_H_NUM_TOKENS_PER_EXPERT_PREFIX = 7


def fused_mega_moe_stage2_forward_impl(
    l1_out: torch.Tensor,
    w2: torch.Tensor,
    handle: tuple,
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
) -> torch.Tensor:
    """SwiGLU + grouped L2 GEMM + combine (nt). Returns y."""
    topk_idx = topk_idx.to(torch.int64)

    # bound swiglu by THIS handle's tile count (per-forward, not shared symm)
    act = swiglu_flydsl_kernel(l1_out, num_tile_blocks=handle[_H_NUM_TILE_BLOCKS])

    # fused grouped L2 GEMM + combine PUSH + topk reduce
    y, _ = grouped_gemm_combine_bf16_flydsl_kernel(
        act,
        w2,
        handle,
        topk_indices=topk_idx.contiguous().view(-1),
        topk_weights=topk_weights.to(torch.float32).contiguous().view(-1),
        layout="nt",
    )
    return y


def fused_mega_moe_stage2_backward_impl(
    grad_y: torch.Tensor,
    l1_out: torch.Tensor,
    dispatch_weights: torch.Tensor,
    w2: torch.Tensor,
    handle: tuple,
    group,
):
    """L2 dgrad (nn) + SwiGLU^T + dW2. Returns ``(grad_l1, grad_gate, dW2)``."""
    real_count_per_expert = handle[_H_REAL_COUNT_PER_EXPERT]
    num_tokens_per_expert_prefix = handle[_H_NUM_TOKENS_PER_EXPERT_PREFIX]

    dy = grad_y.contiguous().to(torch.bfloat16)

    # L2 dgrad: cross-rank dispatch PUSH + grouped GEMM (nn)
    grad_swiglu, dispatch_l2_grad, _, _ = dispatch_grouped_gemm_bf16_flydsl_kernel(
        dy,
        w2,
        group,
        handle=handle,
        layout="nn",
    )

    # SwiGLU^T (re-inject routing weight) -> grad wrt l1_out + gate grad + weighted act
    grad_l1, grad_gate, act_weighted = swiglu_backward_flydsl_kernel(
        grad_swiglu,
        l1_out,
        scale=dispatch_weights,
        return_gate=True,
        return_act_w=True,
        num_tile_blocks=handle[_H_NUM_TILE_BLOCKS],
    )

    # dW2 = dispatched(dy)^ @ act_weighted (variable-K)
    dW2 = grouped_gemm_variable_k_bf16(
        dispatch_l2_grad,
        act_weighted,
        num_tokens_per_expert_prefix,
        masked_k=real_count_per_expert,
        trans_c=False,
    )
    return grad_l1, grad_gate, dW2
