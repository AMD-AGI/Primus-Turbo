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
from primus_turbo.flydsl.mega.bf16.grouped_gemm_bf16_kernel import (
    grouped_gemm_bf16_variable_k_flydsl_kernel,
)
from primus_turbo.flydsl.utils.glu_activation import GLUActivation
from primus_turbo.pytorch.core.utils import is_gfx1250
from primus_turbo.flydsl.utils.swiglu_kernel import (
    swiglu_backward_flydsl_kernel,
    swiglu_flydsl_kernel,
)
from primus_turbo.pytorch.kernels.fused_mega_moe.mega_moe_bias_impl import (
    fc1_bias_backward,
    fc2_bias_backward,
)


def fused_mega_moe_stage2_forward_impl(
    l1_out: torch.Tensor,
    w2: torch.Tensor,
    handle: DispatchHandle,
    topk_idx: torch.Tensor,
    dispatch_weights: torch.Tensor,
    activation: Optional[GLUActivation] = None,
    fc1_bias: Optional[torch.Tensor] = None,
    fc2_bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """SwiGLU (routing-weighted) + grouped L2 GEMM + combine (nt). Returns y.

    ``fc1_bias`` [E_local, 2I] is added to ``l1_out`` inside the activation, ``fc2_bias`` [E_local, H]
    to each row's L2 output scaled by its routing weight, inside the combine push."""
    if is_gfx1250():
        from primus_turbo.pytorch.kernels.fused_mega_moe import (
            fused_mega_moe_gfx1250_impl as gfx1250,
        )

        return gfx1250.stage2_forward(l1_out, w2, handle, topk_idx, dispatch_weights, activation, fc1_bias, fc2_bias)
    # bound swiglu by THIS handle's tile count (per-forward, not shared symm)
    act = swiglu_flydsl_kernel(
        l1_out,
        num_tile_blocks=handle.num_tile_blocks,
        scale=dispatch_weights,
        activation=activation,
        bias=fc1_bias,
        tile_to_expert=handle.tile_to_expert,
    )

    # fused grouped L2 GEMM + combine PUSH + topk reduce
    y, _ = grouped_gemm_combine_bf16_flydsl_kernel(
        act,
        w2,
        handle,
        topk_indices=topk_idx.contiguous().view(-1),
        layout="nt",
        bias=fc2_bias,
        bias_scale=dispatch_weights,
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
    fc1_bias: Optional[torch.Tensor] = None,
    fc2_bias: Optional[torch.Tensor] = None,
):
    """L2 dgrad (nn) + SwiGLU^T + dW2 (+ the bias grads).

    Returns ``(grad_l1, grad_gate, dW2, d_fc1_bias, d_fc2_bias)``; a bias grad is None without its bias."""
    if is_gfx1250():
        from primus_turbo.pytorch.kernels.fused_mega_moe import (
            fused_mega_moe_gfx1250_impl as gfx1250,
        )

        return gfx1250.stage2_backward(
            grad_y, l1_out, dispatch_weights, w2, handle, activation, fc1_bias, fc2_bias
        )
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
        bias=fc1_bias,
        tile_to_expert=handle.tile_to_expert,
    )

    # dW2 = dispatched(dy)^ @ act_weighted (variable-K)
    dW2 = grouped_gemm_bf16_variable_k_flydsl_kernel(
        dispatch_l2_grad,
        act_weighted,
        handle.num_tokens_per_expert_prefix,
        masked_k=handle.num_tokens_per_expert,
        trans_c=False,
        a_row_idx=handle.pool_row_to_recv_token,
    )
    # the dispatched dy lives in the symm buffer, which stage1's backward re-dispatches x into
    d_fc2_bias = (
        None
        if fc2_bias is None
        else fc2_bias_backward(dispatch_l2_grad, handle, dispatch_weights, fc2_bias, grad_gate)
    )
    d_fc1_bias = None if fc1_bias is None else fc1_bias_backward(grad_l1, handle, fc1_bias.dtype)
    return grad_l1, grad_gate, dW2, d_fc1_bias, d_fc2_bias
