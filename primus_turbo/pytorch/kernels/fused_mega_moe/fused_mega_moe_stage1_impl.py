###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Two-stage MegaMoE stage1 (gate-up) FlyDSL kernel composition."""

import torch

from primus_turbo.flydsl.mega import (
    dispatch_grouped_gemm_bf16_flydsl_kernel,
    grouped_gemm_combine_bf16_flydsl_kernel,
)
from primus_turbo.flydsl.mega.bf16.dispatch_prologue_kernel import (
    DispatchHandle,
    run_dispatch_prologue,
)


def fused_mega_moe_stage1_forward_impl(
    x: torch.Tensor,
    w1: torch.Tensor,
    group,
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
):
    """dispatch + grouped L1 GEMM (nt). Returns ``(l1_out, dispatch_weights, handle)``."""
    handle, dispatch_weights = run_dispatch_prologue(x, w1, group, topk_idx, topk_weights)
    l1_out, _, _ = dispatch_grouped_gemm_bf16_flydsl_kernel(x, w1, group, handle=handle, layout="nt")
    return l1_out, dispatch_weights, handle


def fused_mega_moe_stage1_backward_impl(
    grad_l1: torch.Tensor,
    saved_x: torch.Tensor,
    w1: torch.Tensor,
    handle: DispatchHandle,
    group,
    topk_idx: torch.Tensor,
    grad_gate: torch.Tensor,
    num_tokens: int,
    num_topk: int,
):
    """L1 dgrad combine + dW1 (variable-K tn wgrad).

    Returns ``(dx, grad_topk_weights, dW1)``. ``grad_gate`` (from stage2's
    SwiGLU^T) is scattered into ``grad_topk_weights`` by the combine kernel.
    """
    topk_indices_flat = topk_idx.contiguous().view(-1)

    # L1 dgrad (grad_l1 @ w1, nn) + combine PUSH + dx reduce + grad_gate scatter
    dx, grad_topk_weights_flat = grouped_gemm_combine_bf16_flydsl_kernel(
        grad_l1,
        w1,
        handle,
        topk_indices=topk_indices_flat,
        grad_gate=grad_gate,
        layout="nn",
    )

    # dW1 = pool(x)^T @ grad_l1 (variable-K tn wgrad; re-dispatch saved x)
    dW1, _, _ = dispatch_grouped_gemm_bf16_flydsl_kernel(
        saved_x,
        grad_l1,
        group,
        handle=handle,
        layout="tn",
        trans_c=True,
    )

    return dx, grad_topk_weights_flat.view(num_tokens, num_topk), dW1
