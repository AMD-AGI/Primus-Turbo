###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from typing import Optional

import torch
import triton

from primus_turbo.triton.activation.swiglu_kernel import (
    swiglu_bwd_kernel,
    swiglu_fwd_kernel,
    swiglu_with_mask_bwd_kernel,
    swiglu_with_mask_fwd_kernel,
)


def swiglu_fwd_with_probs(x: torch.Tensor, probs: torch.Tensor, row_mask: Optional[torch.Tensor] = None):
    num_tokens, double_hidden_size = x.size()

    probs = probs.unsqueeze(-1)

    out = torch.empty(num_tokens, double_hidden_size // 2, dtype=x.dtype, device=x.device)

    if row_mask is None:
        grid = (num_tokens,)
        swiglu_fwd_kernel[grid](
            x,
            probs,
            out,
            num_tokens=num_tokens,
            stride_x_token=x.stride(0),
            stride_probs_token=probs.stride(0),
            stride_out_token=out.stride(0),
            LOAD_WIDTH=triton.next_power_of_2(double_hidden_size // 2),
        )
    else:
        assert row_mask.is_cuda, "row_mask must be a CUDA tensor"

        BLOCK_SIZE = 8192
        grid = (BLOCK_SIZE,)
        swiglu_with_mask_fwd_kernel[grid](
            x,
            probs,
            row_mask,
            out,
            num_tokens=num_tokens,
            stride_x_token=x.stride(0),
            stride_probs_token=probs.stride(0),
            stride_out_token=out.stride(0),
            LOAD_WIDTH=triton.next_power_of_2(double_hidden_size // 2),
            BLOCK_SIZE=BLOCK_SIZE,
        )

    return out


def swiglu_bwd_with_probs(
    grad_out: torch.Tensor,
    x: torch.Tensor,
    probs: torch.Tensor,
    row_mask: Optional[torch.Tensor] = None,
):
    num_tokens, hidden_size = grad_out.size()

    grad_x = torch.empty_like(x)
    grad_probs = torch.empty_like(probs)

    if row_mask is None:
        grid = (num_tokens,)
        swiglu_bwd_kernel[grid](
            grad_out,
            x,
            probs,
            grad_x,
            grad_probs,
            num_tokens=num_tokens,
            stride_grad_out_token=grad_out.stride(0),
            stride_x_token=x.stride(0),
            stride_probs_token=probs.stride(0),
            stride_grad_x_token=grad_x.stride(0),
            stride_grad_probs_token=grad_probs.stride(0),
            LOAD_WIDTH=triton.next_power_of_2(hidden_size),
        )
    else:
        assert row_mask.is_cuda, "tokens_per_expert must be a CUDA tensor"

        BLOCK_SIZE = 8192
        grid = (BLOCK_SIZE,)
        swiglu_with_mask_bwd_kernel[grid](
            grad_out,
            x,
            probs,
            row_mask,
            grad_x,
            grad_probs,
            num_tokens=num_tokens,
            stride_grad_out_token=grad_out.stride(0),
            stride_x_token=x.stride(0),
            stride_probs_token=probs.stride(0),
            stride_grad_x_token=grad_x.stride(0),
            stride_grad_probs_token=grad_probs.stride(0),
            LOAD_WIDTH=triton.next_power_of_2(hidden_size),
            BLOCK_SIZE=BLOCK_SIZE,
        )

    return grad_x, grad_probs
