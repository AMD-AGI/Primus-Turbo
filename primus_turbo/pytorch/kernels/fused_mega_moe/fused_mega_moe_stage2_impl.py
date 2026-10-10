###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Two-stage MegaMoE stage2 (gate-down) FlyDSL kernel composition."""

from typing import Optional

import torch

from primus_turbo.flydsl.mega import (
    dispatch_grouped_gemm_bf16_flydsl_kernel,
    grouped_gemm_combine_bf16_flydsl_kernel,
)
from primus_turbo.flydsl.mega.bf16.dispatch_prologue_kernel import DispatchHandle
from primus_turbo.flydsl.utils.glu_activation import GLUActivation
from primus_turbo.flydsl.utils.swiglu_kernel import (
    swiglu_backward_flydsl_kernel,
    swiglu_flydsl_kernel,
)
from primus_turbo.pytorch.kernels.fused_mega_moe.fused_mega_moe_backward_impl import compute_dW2


def fused_mega_moe_stage2_forward_impl(
    l1_out: torch.Tensor,
    w2: torch.Tensor,
    handle: DispatchHandle,
    topk_idx: torch.Tensor,
    dispatch_weights: torch.Tensor,
    activation: Optional[GLUActivation] = None,
) -> torch.Tensor:
    """SwiGLU (routing-weighted) + grouped L2 GEMM + combine (nt). Returns y."""
    # bound swiglu by THIS handle's tile count (per-forward, not shared symm)
    act = swiglu_flydsl_kernel(
        l1_out, num_tile_blocks=handle.num_tile_blocks, scale=dispatch_weights, activation=activation
    )

    # fused grouped L2 GEMM + combine PUSH + topk reduce
    y, _ = grouped_gemm_combine_bf16_flydsl_kernel(
        act, w2, handle, topk_indices=topk_idx.contiguous().view(-1), layout="nt"
    )
    return y


def fused_mega_moe_stage2_backward_impl(
    grad_y: torch.Tensor,
    l1_out: torch.Tensor,
    dispatch_weights: torch.Tensor,
    w2: torch.Tensor,
    handle: DispatchHandle,
    group,
    activation: Optional[GLUActivation] = None,
):
    """L2 dgrad (nn) + SwiGLU^T + dW2. Returns ``(grad_l1, grad_gate, dW2)``."""
    dy = grad_y.contiguous().to(torch.bfloat16)

    # L2 dgrad: cross-rank dispatch PUSH + grouped GEMM (nn)
    grad_swiglu, dispatch_l2_grad, _ = dispatch_grouped_gemm_bf16_flydsl_kernel(
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
        num_tile_blocks=handle.num_tile_blocks,
        activation=activation,
    )

    dW2 = compute_dW2(dispatch_l2_grad, act_weighted, handle)
    return grad_l1, grad_gate, dW2
