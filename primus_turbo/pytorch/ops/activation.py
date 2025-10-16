###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from typing import Union

import torch

from primus_turbo.pytorch.kernels.activation.geglu_impl import (
    geglu_bwd_with_probs,
    geglu_fwd_with_probs,
)
from primus_turbo.pytorch.kernels.activation.swiglu_impl import (
    swiglu_bwd_with_probs,
    swiglu_fwd_with_probs,
)

__all__ = ["swiglu_with_probs", "geglu_with_probs"]


class GLUWithProbs(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, x: torch.Tensor, probs: torch.Tensor, row_mask: Union[torch.Tensor, None], act_type: str
    ):
        x_origin_shape = x.size()
        probs_origin_shape = probs.size()

        x = x.view(-1, x.size(-1))
        probs = probs.view(-1)

        assert x.size(0) == probs.size(0), "first dimension of x and probs must be the same"
        assert probs.dtype == torch.float32, "probs must be float32"

        SUPPORTED_ACT_TYPES = ["silu", "gelu"]
        assert (
            act_type in SUPPORTED_ACT_TYPES
        ), f"Unsupported act_type: {act_type}. Supported types: {SUPPORTED_ACT_TYPES}"

        if row_mask is not None:
            assert row_mask.is_cuda, "row_mask must be a CUDA tensor"
            assert x.size(0) == row_mask.size(0), "first dimension of x and row_mask must be the same"
            assert row_mask.ndim == 1, "row_mask must be 1D tensor"
            assert row_mask.dtype == torch.int64, "The dtype of row_mask must be torch.int64."

        if act_type == "silu":
            out = swiglu_fwd_with_probs(x, probs, row_mask)
        elif act_type == "gelu":
            out = geglu_fwd_with_probs(x, probs, row_mask)

        ctx.save_for_backward(x, probs, row_mask)
        ctx.act_type = act_type
        ctx.x_origin_shape = x_origin_shape
        ctx.probs_origin_shape = probs_origin_shape

        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        assert grad_output.ndim == 2

        x, probs, row_mask = ctx.saved_tensors

        if ctx.act_type == "silu":
            grad_x, grad_probs = swiglu_bwd_with_probs(grad_output, x, probs, row_mask)
        elif ctx.act_type == "gelu":
            grad_x, grad_probs = geglu_bwd_with_probs(grad_output, x, probs, row_mask)

        return grad_x.view(ctx.x_origin_shape), grad_probs.view(ctx.probs_origin_shape), None, None


def swiglu_with_probs(
    x: torch.Tensor, probs: torch.Tensor, row_mask: Union[torch.Tensor, None]
) -> torch.Tensor:
    return GLUWithProbs.apply(x, probs, row_mask, "silu")


def geglu_with_probs(
    x: torch.Tensor, probs: torch.Tensor, row_mask: Union[torch.Tensor, None]
) -> torch.Tensor:
    return GLUWithProbs.apply(x, probs, row_mask, "gelu")
