###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from typing import Optional, Union

import torch

from primus_turbo.pytorch.kernels.activation.geglu_impl import (
    geglu_bwd_with_probs,
    geglu_fwd_with_probs,
)
from primus_turbo.pytorch.kernels.activation.linear_gelu_impl import (
    linear_gelu_bwd_impl,
    linear_gelu_fwd_impl,
    linear_gelu_shape_error,
)
from primus_turbo.pytorch.kernels.activation.swiglu_impl import (
    swiglu_bwd_with_probs,
    swiglu_fwd_with_probs,
)

__all__ = ["swiglu_with_probs", "geglu_with_probs", "linear_gelu"]


class GLUWithProbs(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        probs: torch.Tensor,
        row_mask: Union[torch.Tensor, None],
        act_type: str,
        clamp_limit: Optional[float] = None,
    ):
        x_origin_shape = x.size()
        probs_origin_shape = probs.size()

        x = x.view(-1, x.size(-1))
        probs = probs.view(-1)

        assert x.size(0) == probs.size(0), "first dimension of x and probs must be the same"
        assert probs.dtype == torch.float32, "probs must be float32"

        SUPPORTED_ACT_TYPES = ["silu", "gelu"]
        assert act_type in SUPPORTED_ACT_TYPES, (
            f"Unsupported act_type: {act_type}. Supported types: {SUPPORTED_ACT_TYPES}"
        )

        if clamp_limit is not None:
            clamp_limit = float(clamp_limit)
            assert clamp_limit > 0.0, f"clamp_limit must be positive, got {clamp_limit}"

        if row_mask is not None:
            assert row_mask.is_cuda, "row_mask must be a CUDA tensor"
            assert x.size(0) == row_mask.size(0), "first dimension of x and row_mask must be the same"
            assert row_mask.ndim == 1, "row_mask must be 1D tensor"
            assert row_mask.dtype == torch.int64, "The dtype of row_mask must be torch.int64."

        if act_type == "silu":
            out = swiglu_fwd_with_probs(x, probs, row_mask, clamp_limit)
        elif act_type == "gelu":
            out = geglu_fwd_with_probs(x, probs, row_mask, clamp_limit)

        ctx.save_for_backward(x, probs, row_mask)
        ctx.act_type = act_type
        ctx.clamp_limit = clamp_limit
        ctx.x_origin_shape = x_origin_shape
        ctx.probs_origin_shape = probs_origin_shape

        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        assert grad_output.ndim == 2

        x, probs, row_mask = ctx.saved_tensors

        if ctx.act_type == "silu":
            grad_x, grad_probs = swiglu_bwd_with_probs(grad_output, x, probs, row_mask, ctx.clamp_limit)
        elif ctx.act_type == "gelu":
            grad_x, grad_probs = geglu_bwd_with_probs(grad_output, x, probs, row_mask, ctx.clamp_limit)

        return grad_x.view(ctx.x_origin_shape), grad_probs.view(ctx.probs_origin_shape), None, None, None


def swiglu_with_probs(
    x: torch.Tensor,
    probs: torch.Tensor,
    row_mask: Union[torch.Tensor, None],
    clamp_limit: Optional[float] = None,
) -> torch.Tensor:
    """``silu(gate) * up * probs``, where ``gate`` / ``up`` are the halves of ``x``'s
    last dim.

    Args:
        clamp_limit: with ``L`` given, the activation becomes
            ``silu(min(gate, L)) * clamp(up, -L, L) * probs``. The clamp backward is
            straight-through, matching ``torch.clamp``. None for no clamp.
    """
    return GLUWithProbs.apply(x, probs, row_mask, "silu", clamp_limit)


def geglu_with_probs(
    x: torch.Tensor,
    probs: torch.Tensor,
    row_mask: Union[torch.Tensor, None],
    clamp_limit: Optional[float] = None,
) -> torch.Tensor:
    """``gelu(gate) * up * probs``, where ``gate`` / ``up`` are the halves of ``x``'s
    last dim.

    Args:
        clamp_limit: with ``L`` given, the activation becomes
            ``gelu(min(gate, L)) * clamp(up, -L, L) * probs``. The clamp backward is
            straight-through, matching ``torch.clamp``. None for no clamp.
    """
    return GLUWithProbs.apply(x, probs, row_mask, "gelu", clamp_limit)


class _LinearGeluFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, prefix):
        out, aux = linear_gelu_fwd_impl(x, weight, bias, prefix)
        ctx.save_for_backward(x, weight, aux)
        ctx.width = 0 if prefix is None else prefix.shape[1]
        ctx.has_prefix = prefix is not None
        return out

    @staticmethod
    def backward(ctx, grad_out):
        x, weight, aux = ctx.saved_tensors
        grad_h, grad_bias = linear_gelu_bwd_impl(grad_out[:, ctx.width :], aux)
        grad_prefix = grad_out[:, : ctx.width] if ctx.has_prefix else None
        return grad_h @ weight, grad_h.t() @ x, grad_bias, grad_prefix


def linear_gelu(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, prefix: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """``cat([prefix, gelu_tanh(x @ weight.T + bias)], dim=1)`` for BF16 ``x [M, K]``, ``weight [N, K]``,
    ``bias [N]`` and an optional ``prefix [M, P]``: an MLP up-projection and tanh GELU, optionally written
    after a prefix, as a parallel DiT block (FLUX's single-stream block) concatenates the attention
    output in front of it for one output projection.

    The GELU is evaluated as Inductor's ``gelu(approximate="tanh")``, operation for operation; the
    backward computes Inductor's ``gelu_backward`` and the bias gradient in one pass. The fused FlyDSL
    GELU runs on gfx950 when M is a multiple of 2048 and N and P are multiples of 3072; other shapes and
    archs take the unfused GELU. Inputs that are not 2-D BF16 CUDA tensors of matching shapes raise
    ``ValueError``.
    """
    why = linear_gelu_shape_error(x, weight, bias, prefix)
    if why is not None:
        raise ValueError(f"linear_gelu: unsupported input ({why})")
    return _LinearGeluFunction.apply(x, weight, bias, prefix)
