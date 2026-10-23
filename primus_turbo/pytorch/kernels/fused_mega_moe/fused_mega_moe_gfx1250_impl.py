###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""The Mega MoE stages on the gfx1250 kernels (``primus_turbo.flydsl.mega.bf16_gfx1250``).

Same contracts as the gfx950 stage1 / stage2 compositions. The GEMMs read the weights as stored:
NT for the forward, NN (``b_nn``) for the dgrads. The wgrads run the gfx1250 variable-K kernel in
pool order, the dispatched operand read from the dispatch token buffer through the pool's row
table, ordered so the result lands in the weight's own layout.
"""

from typing import Optional

import torch

from primus_turbo.flydsl.grouped_gemm.grouped_gemm_bf16_kernel_gfx1250 import (
    grouped_gemm_bf16_variable_k_flydsl_kernel as _variable_k,
)
from primus_turbo.flydsl.mega.bf16.dispatch_prologue_kernel import (
    DispatchHandle,
    run_dispatch_prologue,
)
from primus_turbo.flydsl.mega.bf16.symm_buffer import get_symm_buffer_for_mega_moe
from primus_turbo.flydsl.mega.bf16_gfx1250.dispatch_gemm_kernel import (
    dispatch_gemm_nt_gfx1250,
    dispatch_only_gfx1250,
)
from primus_turbo.flydsl.mega.bf16_gfx1250.gemm_combine_kernel import (
    gemm_combine_nt_gfx1250,
)
from primus_turbo.flydsl.utils.swiglu_kernel import (
    swiglu_backward_flydsl_kernel,
    swiglu_flydsl_kernel,
)
from primus_turbo.pytorch.kernels.fused_mega_moe.mega_moe_bias_impl import (
    fc1_bias_backward,
    fc2_bias_backward,
)

# the tiles take K in whole k-tiles of 128, or of 64 when 128 does not divide it, at least two
_MIN_TILE_K = 64


def gfx1250_supports(w1: torch.Tensor, w2: torch.Tensor, layout: str = "nt") -> bool:
    """Whether the gfx1250 stages serve these ``[E, 2I, H]`` / ``[E, H, I]`` weights."""
    hidden, inter = w1.shape[2], w2.shape[2]
    return layout == "nt" and all(k % _MIN_TILE_K == 0 and k >= 2 * _MIN_TILE_K for k in (hidden, inter))


def stage1_forward(x, w1, group, topk_idx, topk_weights):
    """dispatch + L1 GEMM. Returns ``(l1_out, dispatch_weights, handle)``."""
    handle, dispatch_weights = run_dispatch_prologue(x, w1, group, topk_idx, topk_weights)
    return dispatch_gemm_nt_gfx1250(x, w1, handle), dispatch_weights, handle


def stage2_forward(
    l1_out: torch.Tensor,
    w2: torch.Tensor,
    handle: DispatchHandle,
    topk_idx: torch.Tensor,
    dispatch_weights: torch.Tensor,
    activation=None,
    fc1_bias: Optional[torch.Tensor] = None,
    fc2_bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Routing-weighted activation (fc1 bias inside) + L2 GEMM + combine (fc2 bias inside) + reduce."""
    act = swiglu_flydsl_kernel(
        l1_out,
        num_tile_blocks=handle.num_tile_blocks,
        scale=dispatch_weights,
        activation=activation,
        bias=fc1_bias,
        tile_to_expert=handle.tile_to_expert,
    )
    return gemm_combine_nt_gfx1250(act, w2, handle, topk_idx, bias=fc2_bias, bias_scale=dispatch_weights)


def stage2_backward(
    grad_y: torch.Tensor,
    l1_out: torch.Tensor,
    dispatch_weights: torch.Tensor,
    w2: torch.Tensor,
    handle: DispatchHandle,
    activation=None,
    fc1_bias: Optional[torch.Tensor] = None,
    fc2_bias: Optional[torch.Tensor] = None,
):
    """dispatch dy + L2 dgrad + activation backward + dW2 (+ the bias grads).

    Returns ``(grad_l1, grad_gate, dW2, d_fc1_bias, d_fc2_bias)``, a bias grad None without its bias."""
    dy = grad_y.contiguous().to(torch.bfloat16)
    grad_act = dispatch_gemm_nt_gfx1250(dy, w2, handle, b_nn=True)
    grad_l1, grad_gate, act_weighted = swiglu_backward_flydsl_kernel(
        grad_act,
        l1_out,
        scale=dispatch_weights,
        return_gate=True,
        return_act_w=True,
        num_tile_blocks=handle.num_tile_blocks,
        activation=activation,
        bias=fc1_bias,
        tile_to_expert=handle.tile_to_expert,
    )
    # the dispatched dy lives in the symm buffer until stage1's backward dispatches x into it
    token_buffer = get_symm_buffer_for_mega_moe().dispatch_token_buffer
    prefix, counts = handle.num_tokens_per_expert_prefix, handle.num_tokens_per_expert
    # dW2[e] = dy_pool[e].T @ act_w[e] -> [E, hidden, inter]
    dW2 = _variable_k(
        token_buffer, act_weighted, prefix, masked_k=counts, a_index=handle.pool_row_to_recv_token
    )
    d_fc2_bias = (
        None if fc2_bias is None else fc2_bias_backward(token_buffer, handle, dispatch_weights, fc2_bias, grad_gate)
    )
    d_fc1_bias = None if fc1_bias is None else fc1_bias_backward(grad_l1, handle, fc1_bias.dtype)
    return grad_l1, grad_gate, dW2.to(w2.dtype), d_fc1_bias, d_fc2_bias


def stage1_backward(grad_l1, saved_x, w1, handle, topk_idx, grad_gate):
    """L1 dgrad + combine + reduce with the routing-weight gradient, then dW1.

    Returns ``(dx, grad_topk_weights [tokens, topk], dW1)``."""
    dx, grad_topk_weights = gemm_combine_nt_gfx1250(grad_l1, w1, handle, topk_idx, grad_gate=grad_gate, b_nn=True)
    # dW1[e] = grad_l1[e].T @ x_pool[e] -> [E, 2 * inter, hidden]; x is dispatched again
    x_recv = dispatch_only_gfx1250(saved_x.contiguous(), handle)
    prefix, counts = handle.num_tokens_per_expert_prefix, handle.num_tokens_per_expert
    dW1 = _variable_k(grad_l1, x_recv, prefix, masked_k=counts, b_index=handle.pool_row_to_recv_token)
    return dx, grad_topk_weights, dW1.to(w1.dtype)
