###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from typing import Optional

import torch
import triton

from primus_turbo.triton.activation.quick_geglu_kernel import (
    quick_geglu_bwd_kernel,
    quick_geglu_fwd_kernel,
    quick_geglu_with_mask_bwd_kernel,
    quick_geglu_with_mask_fwd_kernel,
)

MASK_BLOCK_SIZE = 8192


def quick_geglu_fwd(
    x: torch.Tensor,
    linear_offset: float,
    bias: Optional[torch.Tensor] = None,
    weights: Optional[torch.Tensor] = None,
    row_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    num_tokens, double_hidden_size = x.size()
    out = torch.empty(num_tokens, double_hidden_size // 2, dtype=x.dtype, device=x.device)
    has_bias = bias is not None
    has_weights = weights is not None
    load_width = triton.next_power_of_2(double_hidden_size // 2)

    if row_mask is None:
        grid = (num_tokens,)
        quick_geglu_fwd_kernel[grid](
            x,
            weights,
            bias,
            out,
            linear_offset,
            num_tokens=num_tokens,
            stride_x_token=x.stride(0),
            stride_weights_token=weights.stride(0) if has_weights else 0,
            stride_out_token=out.stride(0),
            LOAD_WIDTH=load_width,
            HAS_BIAS=has_bias,
            HAS_WEIGHTS=has_weights,
        )
    else:
        grid = (MASK_BLOCK_SIZE,)
        quick_geglu_with_mask_fwd_kernel[grid](
            x,
            weights,
            bias,
            row_mask,
            out,
            linear_offset,
            num_tokens=num_tokens,
            stride_x_token=x.stride(0),
            stride_weights_token=weights.stride(0) if has_weights else 0,
            stride_out_token=out.stride(0),
            LOAD_WIDTH=load_width,
            BLOCK_SIZE=MASK_BLOCK_SIZE,
            HAS_BIAS=has_bias,
            HAS_WEIGHTS=has_weights,
        )

    return out


def quick_geglu_bwd(
    grad_out: torch.Tensor,
    x: torch.Tensor,
    linear_offset: float,
    bias: Optional[torch.Tensor] = None,
    weights: Optional[torch.Tensor] = None,
    row_mask: Optional[torch.Tensor] = None,
):
    num_tokens, hidden_size = grad_out.size()
    grad_x = torch.empty_like(x)
    has_bias = bias is not None
    has_weights = weights is not None
    grad_weights = torch.empty_like(weights) if has_weights else None
    load_width = triton.next_power_of_2(hidden_size)

    if row_mask is None:
        grid = (num_tokens,)
        quick_geglu_bwd_kernel[grid](
            grad_out,
            x,
            weights,
            bias,
            grad_x,
            grad_weights,
            linear_offset,
            num_tokens=num_tokens,
            stride_grad_out_token=grad_out.stride(0),
            stride_x_token=x.stride(0),
            stride_weights_token=weights.stride(0) if has_weights else 0,
            stride_grad_x_token=grad_x.stride(0),
            stride_grad_weights_token=grad_weights.stride(0) if has_weights else 0,
            LOAD_WIDTH=load_width,
            HAS_BIAS=has_bias,
            HAS_WEIGHTS=has_weights,
        )
    else:
        grid = (MASK_BLOCK_SIZE,)
        quick_geglu_with_mask_bwd_kernel[grid](
            grad_out,
            x,
            weights,
            bias,
            row_mask,
            grad_x,
            grad_weights,
            linear_offset,
            num_tokens=num_tokens,
            stride_grad_out_token=grad_out.stride(0),
            stride_x_token=x.stride(0),
            stride_weights_token=weights.stride(0) if has_weights else 0,
            stride_grad_x_token=grad_x.stride(0),
            stride_grad_weights_token=grad_weights.stride(0) if has_weights else 0,
            LOAD_WIDTH=load_width,
            BLOCK_SIZE=MASK_BLOCK_SIZE,
            HAS_BIAS=has_bias,
            HAS_WEIGHTS=has_weights,
        )

    return grad_x, grad_weights
