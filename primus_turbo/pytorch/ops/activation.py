###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from typing import Optional

import torch

from primus_turbo.pytorch.kernels.activation.geglu_impl import (
    bias_geglu_bwd,
    bias_geglu_fwd,
    geglu_bwd,
    geglu_fwd,
)
from primus_turbo.pytorch.kernels.activation.gelu_impl import (
    bias_gelu_bwd,
    bias_gelu_fwd,
)
from primus_turbo.pytorch.kernels.activation.quick_geglu_impl import (
    quick_geglu_bwd,
    quick_geglu_fwd,
)
from primus_turbo.pytorch.kernels.activation.swiglu_impl import (
    bias_swiglu_bwd,
    bias_swiglu_fwd,
    swiglu_bwd,
    swiglu_fwd,
)

__all__ = [
    "bias_gelu_impl",
    "bias_geglu_impl",
    "bias_swiglu_impl",
    "weighted_bias_swiglu_impl",
    "weighted_bias_geglu_impl",
    "weighted_bias_quick_geglu_impl",
]


def _validate_row_mask(row_mask: torch.Tensor, num_rows: int):
    assert row_mask.is_cuda, "row_mask must be a CUDA tensor"
    assert row_mask.ndim == 1, "row_mask must be 1D tensor"
    assert row_mask.size(0) == num_rows, "first dimension of input and row_mask must be the same"
    assert row_mask.dtype == torch.int64, "row_mask must be torch.int64"


class BiasSwiGLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, bias, row_mask):
        ctx.save_for_backward(input, bias, row_mask)
        return bias_swiglu_fwd(input, bias, row_mask)

    @staticmethod
    def backward(ctx, grad_output):
        input, bias, row_mask = ctx.saved_tensors
        grad_x = bias_swiglu_bwd(grad_output, input, bias, row_mask)
        return grad_x, (grad_x if bias is not None else None), None


def _validate_weight_and_row_mask(
    input: torch.Tensor,
    weights: Optional[torch.Tensor],
    row_mask: Optional[torch.Tensor],
):
    if weights is not None:
        assert input.size(0) == weights.size(0), "first dimension of input and weights must be the same"
        assert weights.dtype == torch.float32, "weights must be float32"
    if row_mask is not None:
        _validate_row_mask(row_mask, input.size(0))


def _clamp_glu_input(input: torch.Tensor, clamp_value: Optional[float]) -> torch.Tensor:
    if clamp_value is None:
        return input
    x_glu, x_linear = input.chunk(2, -1)
    return torch.cat(
        (
            x_glu.clamp(min=None, max=clamp_value),
            x_linear.clamp(min=-clamp_value, max=clamp_value),
        ),
        -1,
    )


class BiasGeLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, bias, row_mask):
        ctx.save_for_backward(input, bias, row_mask)
        return bias_gelu_fwd(input, bias, row_mask)

    @staticmethod
    def backward(ctx, grad_output):
        input, bias, row_mask = ctx.saved_tensors
        grad_x = bias_gelu_bwd(grad_output, input, bias, row_mask)
        return grad_x, (grad_x if bias is not None else None), None


class BiasGeGLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, bias, row_mask):
        ctx.save_for_backward(input, bias, row_mask)
        return bias_geglu_fwd(input, bias, row_mask)

    @staticmethod
    def backward(ctx, grad_output):
        input, bias, row_mask = ctx.saved_tensors
        grad_x = bias_geglu_bwd(grad_output, input, bias, row_mask)
        return grad_x, (grad_x if bias is not None else None), None


class BiasQuickGeGLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, bias, linear_offset, row_mask):
        ctx.save_for_backward(input, bias, row_mask)
        ctx.linear_offset = linear_offset
        return quick_geglu_fwd(input, linear_offset, bias=bias, row_mask=row_mask)

    @staticmethod
    def backward(ctx, grad_output):
        input, bias, row_mask = ctx.saved_tensors
        grad_x, _ = quick_geglu_bwd(grad_output, input, ctx.linear_offset, bias=bias, row_mask=row_mask)
        return grad_x, (grad_x if bias is not None else None), None, None


def bias_gelu_impl(
    input: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    row_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Bias GELU fusion.
        input: [N, D] or [B, S, D]
        bias: Optional [D]
        row_mask: Optional [N] int64
        output: [N, D] or [B, S, D]
    """
    ori_shape = input.shape
    assert len(ori_shape) in [2, 3]

    input = input.view(-1, ori_shape[-1])
    _validate_weight_and_row_mask(input, None, row_mask)

    output = BiasGeLUFunction.apply(input, bias, row_mask)
    return output if len(ori_shape) == 2 else output.view(ori_shape[0], ori_shape[1], -1)


def bias_geglu_impl(
    input: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    row_mask: Optional[torch.Tensor] = None,
    clamp_value: Optional[float] = None,
) -> torch.Tensor:
    """
    Bias GEGLU fusion.
        input: [N, 2H] or [B, S, 2H]
        bias: Optional [2H]
        row_mask: Optional [N] int64
        output: [N, H] or [B, S, H]
    """
    ori_shape = input.shape
    assert len(ori_shape) in [2, 3]

    input = _clamp_glu_input(input, clamp_value)
    input = input.view(-1, ori_shape[-1])
    _validate_weight_and_row_mask(input, None, row_mask)

    output = BiasGeGLUFunction.apply(input, bias, row_mask)
    return output if len(ori_shape) == 2 else output.view(ori_shape[0], ori_shape[1], -1)


def bias_swiglu_impl(
    input: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    row_mask: Optional[torch.Tensor] = None,
    clamp_value: Optional[float] = None,
) -> torch.Tensor:
    """
    Bias SwiGLU fusion.
        input: [N, 2H] or [B, S, 2H]
        bias: Optional [2H]
        row_mask: Optional [N] int64
        output: [N, H] or [B, S, H]
    """
    ori_shape = input.shape
    assert len(ori_shape) in [2, 3]

    input = _clamp_glu_input(input, clamp_value)
    input = input.view(-1, ori_shape[-1])
    _validate_weight_and_row_mask(input, None, row_mask)

    output = BiasSwiGLUFunction.apply(input, bias, row_mask)
    return output if len(ori_shape) == 2 else output.view(ori_shape[0], ori_shape[1], -1)


class WeightedSwiGLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weights, row_mask):
        ctx.save_for_backward(input, weights, row_mask)

        x = input.view(-1, input.size(-1))
        w = weights.view(-1)
        return swiglu_fwd(x, w, row_mask)

    @staticmethod
    def backward(ctx, grad_output):
        input, weights, row_mask = ctx.saved_tensors

        x = input.view(-1, input.size(-1))
        w = weights.view(-1)
        grad_x, grad_w = swiglu_bwd(grad_output, x, w, row_mask)
        return grad_x.view(input.shape), grad_w.view(weights.shape), None


class WeightedGeGLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weights, row_mask):
        ctx.save_for_backward(input, weights, row_mask)

        x = input.view(-1, input.size(-1))
        w = weights.view(-1)
        return geglu_fwd(x, w, row_mask)

    @staticmethod
    def backward(ctx, grad_output):
        input, weights, row_mask = ctx.saved_tensors

        x = input.view(-1, input.size(-1))
        w = weights.view(-1)
        grad_x, grad_w = geglu_bwd(grad_output, x, w, row_mask)
        return grad_x.view(input.shape), grad_w.view(weights.shape), None


class WeightedQuickGeGLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weights, linear_offset, row_mask):
        ctx.save_for_backward(input, weights, row_mask)
        ctx.linear_offset = linear_offset
        return quick_geglu_fwd(input, linear_offset, weights=weights, row_mask=row_mask)

    @staticmethod
    def backward(ctx, grad_output):
        input, weights, row_mask = ctx.saved_tensors
        grad_x, grad_w = quick_geglu_bwd(
            grad_output, input, ctx.linear_offset, weights=weights, row_mask=row_mask
        )
        return grad_x, grad_w, None, None


class WeightedBiasSwiGLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, bias, weights, row_mask):
        ctx.save_for_backward(input, bias, weights, row_mask)

        y = (input + bias).view(-1, input.size(-1))
        w = weights.view(-1)
        return swiglu_fwd(y, w, row_mask)

    @staticmethod
    def backward(ctx, grad_output):
        input, bias, weights, row_mask = ctx.saved_tensors

        y = (input + bias).view(-1, input.size(-1))
        w = weights.view(-1)
        grad_y, grad_w = swiglu_bwd(grad_output, y, w, row_mask)
        grad_y = grad_y.view(input.shape)
        return grad_y, grad_y, grad_w.view(weights.shape), None


class WeightedBiasGeGLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, bias, weights, row_mask):
        ctx.save_for_backward(input, bias, weights, row_mask)

        y = (input + bias).view(-1, input.size(-1))
        w = weights.view(-1)
        return geglu_fwd(y, w, row_mask)

    @staticmethod
    def backward(ctx, grad_output):
        input, bias, weights, row_mask = ctx.saved_tensors

        y = (input + bias).view(-1, input.size(-1))
        w = weights.view(-1)
        grad_y, grad_w = geglu_bwd(grad_output, y, w, row_mask)
        grad_y = grad_y.view(input.shape)
        return grad_y, grad_y, grad_w.view(weights.shape), None


class WeightedBiasQuickGeGLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, bias, weights, linear_offset, row_mask):
        ctx.save_for_backward(input, bias, weights, row_mask)
        ctx.linear_offset = linear_offset
        return quick_geglu_fwd(input, linear_offset, bias=bias, weights=weights, row_mask=row_mask)

    @staticmethod
    def backward(ctx, grad_output):
        input, bias, weights, row_mask = ctx.saved_tensors
        grad_x, grad_w = quick_geglu_bwd(
            grad_output,
            input,
            ctx.linear_offset,
            bias=bias,
            weights=weights,
            row_mask=row_mask,
        )
        return grad_x, grad_x, grad_w, None, None


def weighted_bias_swiglu_impl(
    input: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    weights: Optional[torch.Tensor] = None,
    row_mask: Optional[torch.Tensor] = None,
    clamp_value: Optional[float] = None,
) -> torch.Tensor:
    """
    Token-wise-weighted bias SwiGLU fusion.
        input: [N, 2H] or [B, S, 2H]
        bias: Optional [2H]
        weights: Optional [N, 1]
        row_mask: Optional [N] int64
        output: [N, H] or [B, S, H]
    """
    ori_shape = input.shape
    assert len(ori_shape) in [2, 3]

    input = _clamp_glu_input(input, clamp_value)
    input = input.view(-1, ori_shape[-1])
    _validate_weight_and_row_mask(input, weights, row_mask)

    if weights is not None and bias is not None:
        output = WeightedBiasSwiGLUFunction.apply(input, bias, weights, row_mask)
    elif weights is not None:
        output = WeightedSwiGLUFunction.apply(input, weights, row_mask)
    else:
        output = BiasSwiGLUFunction.apply(input, bias, row_mask)

    return output if len(ori_shape) == 2 else output.view(ori_shape[0], ori_shape[1], -1)


def weighted_bias_geglu_impl(
    input: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    weights: Optional[torch.Tensor] = None,
    row_mask: Optional[torch.Tensor] = None,
    clamp_value: Optional[float] = None,
) -> torch.Tensor:
    """
    Token-wise-weighted bias GEGLU fusion.
        input: [N, 2H] or [B, S, 2H]
        bias: Optional [2H]
        weights: Optional [N, 1]
        row_mask: Optional [N] int64
        output: [N, H] or [B, S, H]
    """
    ori_shape = input.shape
    assert len(ori_shape) in [2, 3]

    input = _clamp_glu_input(input, clamp_value)
    input = input.view(-1, ori_shape[-1])
    _validate_weight_and_row_mask(input, weights, row_mask)

    if weights is not None and bias is not None:
        output = WeightedBiasGeGLUFunction.apply(input, bias, weights, row_mask)
    elif weights is not None:
        output = WeightedGeGLUFunction.apply(input, weights, row_mask)
    else:
        output = BiasGeGLUFunction.apply(input, bias, row_mask)

    return output if len(ori_shape) == 2 else output.view(ori_shape[0], ori_shape[1], -1)


def weighted_bias_quick_geglu_impl(
    input: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    weights: Optional[torch.Tensor] = None,
    row_mask: Optional[torch.Tensor] = None,
    linear_offset: float = 0.0,
    clamp_value: Optional[float] = None,
) -> torch.Tensor:
    """
    Token-wise-weighted bias Quick-GEGLU (sigmoid approximation) fusion.
        input: [N, 2H] or [B, S, 2H]
        bias: Optional [2H]
        weights: Optional [N, 1]
        row_mask: Optional [N] int64
        linear_offset: float
        output: [N, H] or [B, S, H]
    """
    ori_shape = input.shape
    assert len(ori_shape) in [2, 3]

    input = _clamp_glu_input(input, clamp_value)
    input = input.view(-1, ori_shape[-1])
    linear_offset = float(linear_offset)
    _validate_weight_and_row_mask(input, weights, row_mask)

    if weights is not None and bias is not None:
        output = WeightedBiasQuickGeGLUFunction.apply(input, bias, weights, linear_offset, row_mask)
    elif weights is not None:
        output = WeightedQuickGeGLUFunction.apply(input, weights, linear_offset, row_mask)
    else:
        output = BiasQuickGeGLUFunction.apply(input, bias, linear_offset, row_mask)

    return output if len(ori_shape) == 2 else output.view(ori_shape[0], ori_shape[1], -1)
