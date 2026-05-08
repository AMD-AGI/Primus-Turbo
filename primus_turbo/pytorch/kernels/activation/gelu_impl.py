###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from typing import Optional

import torch
import triton

from primus_turbo.triton.activation.bias_gelu_kernel import (
    bias_gelu_bwd_kernel,
    bias_gelu_fwd_kernel,
    bias_gelu_with_mask_bwd_kernel,
    bias_gelu_with_mask_fwd_kernel,
)

MASK_BLOCK_SIZE = 8192


def bias_gelu_fwd(
    x: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    row_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    num_tokens, hidden_size = x.size()
    out = torch.empty_like(x)
    has_bias = bias is not None
    load_width = triton.next_power_of_2(hidden_size)

    if row_mask is None:
        grid = (num_tokens,)
        bias_gelu_fwd_kernel[grid](
            x,
            bias,
            out,
            num_tokens=num_tokens,
            stride_x_token=x.stride(0),
            stride_out_token=out.stride(0),
            LOAD_WIDTH=load_width,
            HAS_BIAS=has_bias,
        )
    else:
        grid = (MASK_BLOCK_SIZE,)
        bias_gelu_with_mask_fwd_kernel[grid](
            x,
            bias,
            row_mask,
            out,
            num_tokens=num_tokens,
            stride_x_token=x.stride(0),
            stride_out_token=out.stride(0),
            LOAD_WIDTH=load_width,
            BLOCK_SIZE=MASK_BLOCK_SIZE,
            HAS_BIAS=has_bias,
        )

    return out


def bias_gelu_bwd(
    grad_out: torch.Tensor,
    x: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    row_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    num_tokens, hidden_size = grad_out.size()
    grad_x = torch.empty_like(x)
    has_bias = bias is not None
    load_width = triton.next_power_of_2(hidden_size)

    if row_mask is None:
        grid = (num_tokens,)
        bias_gelu_bwd_kernel[grid](
            grad_out,
            x,
            bias,
            grad_x,
            num_tokens=num_tokens,
            stride_grad_out_token=grad_out.stride(0),
            stride_x_token=x.stride(0),
            stride_grad_x_token=grad_x.stride(0),
            LOAD_WIDTH=load_width,
            HAS_BIAS=has_bias,
        )
    else:
        grid = (MASK_BLOCK_SIZE,)
        bias_gelu_with_mask_bwd_kernel[grid](
            grad_out,
            x,
            bias,
            row_mask,
            grad_x,
            num_tokens=num_tokens,
            stride_grad_out_token=grad_out.stride(0),
            stride_x_token=x.stride(0),
            stride_grad_x_token=grad_x.stride(0),
            LOAD_WIDTH=load_width,
            BLOCK_SIZE=MASK_BLOCK_SIZE,
            HAS_BIAS=has_bias,
        )

    return grad_x
