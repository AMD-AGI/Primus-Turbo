###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from typing import Optional

import torch
import triton

from primus_turbo.triton.activation.bias_geglu_kernel import (
    bias_geglu_bwd_kernel,
    bias_geglu_fwd_kernel,
    bias_geglu_with_mask_bwd_kernel,
    bias_geglu_with_mask_fwd_kernel,
)
from primus_turbo.triton.activation.geglu_kernel import (
    geglu_bwd_kernel,
    geglu_fwd_kernel,
    geglu_with_mask_bwd_kernel,
    geglu_with_mask_fwd_kernel,
)

MASK_BLOCK_SIZE = 8192


# ── Weighted GeGLU (with probs/weights) ──────────────────────────────────────


def geglu_fwd(x: torch.Tensor, probs: torch.Tensor, row_mask: Optional[torch.Tensor] = None):
    num_tokens, double_hidden_size = x.size()
    probs = probs.unsqueeze(-1)
    out = torch.empty(num_tokens, double_hidden_size // 2, dtype=x.dtype, device=x.device)
    load_width = triton.next_power_of_2(double_hidden_size // 2)

    if row_mask is None:
        grid = (num_tokens,)
        geglu_fwd_kernel[grid](
            x,
            probs,
            out,
            num_tokens=num_tokens,
            stride_x_token=x.stride(0),
            stride_probs_token=probs.stride(0),
            stride_out_token=out.stride(0),
            LOAD_WIDTH=load_width,
        )
    else:
        assert row_mask.is_cuda, "row_mask must be a CUDA tensor"
        grid = (MASK_BLOCK_SIZE,)
        geglu_with_mask_fwd_kernel[grid](
            x,
            probs,
            row_mask,
            out,
            num_tokens=num_tokens,
            stride_x_token=x.stride(0),
            stride_probs_token=probs.stride(0),
            stride_out_token=out.stride(0),
            LOAD_WIDTH=load_width,
            BLOCK_SIZE=MASK_BLOCK_SIZE,
        )

    return out


def geglu_bwd(
    grad_out: torch.Tensor,
    x: torch.Tensor,
    probs: torch.Tensor,
    row_mask: Optional[torch.Tensor] = None,
):
    num_tokens, hidden_size = grad_out.size()
    grad_x = torch.empty_like(x)
    grad_probs = torch.empty_like(probs)
    load_width = triton.next_power_of_2(hidden_size)

    if row_mask is None:
        grid = (num_tokens,)
        geglu_bwd_kernel[grid](
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
            LOAD_WIDTH=load_width,
        )
    else:
        assert row_mask.is_cuda, "row_mask must be a CUDA tensor"
        grid = (MASK_BLOCK_SIZE,)
        geglu_with_mask_bwd_kernel[grid](
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
            LOAD_WIDTH=load_width,
            BLOCK_SIZE=MASK_BLOCK_SIZE,
        )

    return grad_x, grad_probs


# ── Bias GeGLU ───────────────────────────────────────────────────────────────


def bias_geglu_fwd(
    x: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    row_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    num_tokens, double_hidden_size = x.size()
    out = torch.empty(num_tokens, double_hidden_size // 2, dtype=x.dtype, device=x.device)
    has_bias = bias is not None
    load_width = triton.next_power_of_2(double_hidden_size // 2)

    if row_mask is None:
        grid = (num_tokens,)
        bias_geglu_fwd_kernel[grid](
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
        bias_geglu_with_mask_fwd_kernel[grid](
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


def bias_geglu_bwd(
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
        bias_geglu_bwd_kernel[grid](
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
        bias_geglu_with_mask_bwd_kernel[grid](
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
