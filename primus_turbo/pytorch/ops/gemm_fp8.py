###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from typing import Union

import torch

from primus_turbo.pytorch.core.backend import BackendType
from primus_turbo.pytorch.core.low_precision import (
    Float8QuantConfig,
    Format,
    ScalingGranularity,
    ScalingRecipe,
    check_mxfp8_support,
    float8_e4m3,
    float8_e5m2,
)
from primus_turbo.pytorch.core.quantized_tensor import (
    QuantizedTensor,
    check_quantized_tensor,
)
from primus_turbo.pytorch.kernels.gemm.gemm_fp8_impl import gemm_fp8_impl

__all__ = ["gemm_fp8"]


def _get_fp8_dtype(format: Format, is_fwd_stage: bool):
    if format == Format.E4M3:
        return float8_e4m3
    elif format == Format.E5M2:
        return float8_e5m2
    elif format == Format.HYBRID:
        return float8_e4m3 if is_fwd_stage else float8_e5m2
    else:
        raise ValueError(f"Unsupported FP8 format: {format}")


class FP8GemmTensorFunction(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx,
        a: Union[torch.Tensor, QuantizedTensor],
        b: Union[torch.Tensor, QuantizedTensor],
        trans_a: bool,  # trans_a has to be False
        trans_b: bool,
        out_dtype: torch.dtype,
        config: Float8QuantConfig,
    ):
        assert trans_a == False, "trans_a has to be False"

        if isinstance(a, QuantizedTensor):
            check_quantized_tensor(a, config)
            a_fp8 = a
        else:
            a_dtype = _get_fp8_dtype(config.format, True)
            a_fp8 = QuantizedTensor(
                a,
                a_dtype,
                config.granularity,
                block_size=config.block_size,
                keep_trans_cache=False,
            )

        if isinstance(b, QuantizedTensor):
            check_quantized_tensor(b, config)
            b_fp8 = b
        else:
            b_dtype = _get_fp8_dtype(config.format, True)
            b_fp8 = QuantizedTensor(
                b,
                b_dtype,
                config.granularity,
                block_size=config.block_size,
                keep_trans_cache=False,
            )

        out = gemm_fp8_impl(
            a_fp8.data,
            a_fp8.scale_inv,
            trans_a,
            b_fp8.data,
            b_fp8.scale_inv,
            trans_b,
            out_dtype,
            False,
            granularity=config.granularity.value,
            default_backend=BackendType.HIPBLASLT.value,
        )
        ctx.save_for_backward(a_fp8.data, a_fp8.scale_inv, b_fp8.data, b_fp8.scale_inv)
        ctx.trans_a = trans_a
        ctx.trans_b = trans_b
        ctx.out_dtype = out_dtype
        ctx.config = config

        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        if not grad_out.is_contiguous():
            grad_out = grad_out.contiguous()
        a_fp8, a_scale_inv, b_fp8, b_scale_inv = ctx.saved_tensors
        grad_out_dtype = _get_fp8_dtype(ctx.config.format, False)

        grad_out_fp8 = QuantizedTensor(
            grad_out,
            grad_out_dtype,
            ctx.config.granularity,
            keep_trans_cache=False,
        )

        a_grad = gemm_fp8_impl(
            grad_out_fp8.data,
            grad_out_fp8.scale_inv,
            False,
            b_fp8,
            b_scale_inv,
            not ctx.trans_b,
            ctx.out_dtype,
            ctx.trans_a,
            granularity=ctx.config.granularity.value,
            default_backend=BackendType.HIPBLASLT.value,
        )

        b_grad = gemm_fp8_impl(
            a_fp8,
            a_scale_inv,
            not ctx.trans_a,
            grad_out_fp8.data,
            grad_out_fp8.scale_inv,
            False,
            ctx.out_dtype,
            ctx.trans_b,
            granularity=ctx.config.granularity.value,
            default_backend=BackendType.HIPBLASLT.value,
        )

        return (a_grad, b_grad, None, None, None, None)


class FP8GemmRowFunction(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx,
        a: Union[torch.Tensor, QuantizedTensor],
        b: Union[torch.Tensor, QuantizedTensor],
        trans_a: bool,  # trans_a has to be False
        trans_b: bool,
        out_dtype: torch.dtype,
        config: Float8QuantConfig,
    ):
        assert trans_a == False, "trans_a has to be False"

        if isinstance(a, QuantizedTensor):
            check_quantized_tensor(a, config)
            a_fp8 = a
        else:
            a_dtype = _get_fp8_dtype(config.format, True)
            a_fp8 = QuantizedTensor(
                a,
                a_dtype,
                config.granularity,
                block_size=config.block_size,
                keep_trans_cache=True,
            )

        if isinstance(b, QuantizedTensor):
            check_quantized_tensor(b, config)
            b_fp8 = b
        else:
            b_dtype = _get_fp8_dtype(config.format, True)
            b_fp8 = QuantizedTensor(
                b,
                b_dtype,
                config.granularity,
                block_size=config.block_size,
                keep_trans_cache=True,
            )

        # a_fp8.data = axis=1 (row-wise), a_fp8.t() = axis=0 (col-wise)
        a_fp8_row, a_scale_inv_row = a_fp8.data, a_fp8.scale_inv
        a_fp8_col, a_scale_inv_col = a_fp8.t()

        # For b: trans_b=True (NT) -> row is data; trans_b=False (NN) -> row is t
        if trans_b:
            b_fp8_row, b_scale_inv_row = b_fp8.data, b_fp8.scale_inv
            b_fp8_col, b_scale_inv_col = b_fp8.t()
        else:
            b_fp8_row, b_scale_inv_row = b_fp8.t()
            b_fp8_col, b_scale_inv_col = b_fp8.data, b_fp8.scale_inv

        out = gemm_fp8_impl(
            a_fp8_row,
            a_scale_inv_row,
            trans_a,
            b_fp8_row,
            b_scale_inv_row,
            trans_b,
            out_dtype,
            False,
            granularity=config.granularity.value,
            default_backend=BackendType.CK.value,
        )

        ctx.save_for_backward(a_fp8_col, a_scale_inv_col, b_fp8_col, b_scale_inv_col)
        ctx.trans_a = trans_a
        ctx.trans_b = trans_b
        ctx.out_dtype = out_dtype
        ctx.config = config

        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        if not grad_out.is_contiguous():
            grad_out = grad_out.contiguous()
        a_fp8_col, a_scale_inv_col, b_fp8_col, b_scale_inv_col = ctx.saved_tensors
        grad_out_dtype = _get_fp8_dtype(ctx.config.format, False)

        # Quantize grad_out row-wise (axis=1) and col-wise (axis=0) in one shot.
        grad_out_fp8 = QuantizedTensor(
            grad_out,
            grad_out_dtype,
            ctx.config.granularity,
            block_size=ctx.config.block_size,
            keep_trans_cache=True,
        )
        grad_out_fp8_row, grad_out_scale_inv_row = grad_out_fp8.data, grad_out_fp8.scale_inv
        grad_out_fp8_col, grad_out_scale_inv_col = grad_out_fp8.t()

        # NT
        a_grad = gemm_fp8_impl(
            grad_out_fp8_row,
            grad_out_scale_inv_row,
            False,
            b_fp8_col,
            b_scale_inv_col,
            not ctx.trans_b,
            ctx.out_dtype,
            ctx.trans_a,
            granularity=ctx.config.granularity.value,
            default_backend=BackendType.CK.value,
        )

        # TN
        b_grad = gemm_fp8_impl(
            a_fp8_col,
            a_scale_inv_col,
            not ctx.trans_a,
            grad_out_fp8_col,
            grad_out_scale_inv_col,
            False,
            ctx.out_dtype,
            ctx.trans_b,
            granularity=ctx.config.granularity.value,
            default_backend=BackendType.CK.value,
        )

        return (a_grad, b_grad, None, None, None, None)


class FP8GemmBlockFunction(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx,
        a: Union[torch.Tensor, QuantizedTensor],
        b: Union[torch.Tensor, QuantizedTensor],
        trans_a: bool,
        trans_b: bool,
        out_dtype: torch.dtype,
        config: Float8QuantConfig,
    ):
        assert config.granularity == ScalingGranularity.BLOCKWISE
        assert trans_a == False

        # Activation: 1D-block, keep row/col cache for forward+backward.
        if isinstance(a, QuantizedTensor):
            check_quantized_tensor(a, config)
            a_fp8 = a
        else:
            a_dtype = _get_fp8_dtype(config.format, True)
            a_fp8 = QuantizedTensor(
                a,
                a_dtype,
                config.granularity,
                block_size=config.block_size,
                keep_trans_cache=True,
            )

        # Weight: 2D-block; scale is symmetric along row/col so no trans cache needed.
        b_scaling_recipe = ScalingRecipe(use_2d_block=True)
        if isinstance(b, QuantizedTensor):
            check_quantized_tensor(b, config, b_scaling_recipe)
            b_fp8 = b
        else:
            b_dtype = _get_fp8_dtype(config.format, True)
            b_fp8 = QuantizedTensor(
                b,
                b_dtype,
                config.granularity,
                block_size=config.block_size,
                keep_trans_cache=False,
                scaling_recipe=b_scaling_recipe,
            )

        a_fp8_row, a_scale_inv_row = a_fp8.data, a_fp8.scale_inv
        a_fp8_col, a_scale_inv_col = a_fp8.t()

        out = gemm_fp8_impl(
            a_fp8_row,
            a_scale_inv_row,
            trans_a,
            b_fp8.data,
            b_fp8.scale_inv,
            trans_b,
            out_dtype,
            False,
            granularity=config.granularity.value,
            default_backend=BackendType.CK.value,
        )
        ctx.save_for_backward(a_fp8_col, a_scale_inv_col, b_fp8.data, b_fp8.scale_inv)
        ctx.trans_a = trans_a
        ctx.trans_b = trans_b
        ctx.out_dtype = out_dtype
        ctx.config = config
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        if not grad_out.is_contiguous():
            grad_out = grad_out.contiguous()

        a_fp8_col, a_scale_inv_col, b_fp8, b_scale_inv = ctx.saved_tensors
        grad_out_dtype = _get_fp8_dtype(ctx.config.format, False)

        # Quantize grad_out (activation, 1D-block) in both row/col directions at once.
        grad_out_fp8 = QuantizedTensor(
            grad_out,
            grad_out_dtype,
            ctx.config.granularity,
            block_size=ctx.config.block_size,
            keep_trans_cache=True,
        )
        grad_out_fp8_row, grad_out_scale_inv_row = grad_out_fp8.data, grad_out_fp8.scale_inv
        grad_out_fp8_col, grad_out_scale_inv_col = grad_out_fp8.t()

        a_grad = gemm_fp8_impl(
            grad_out_fp8_row,
            grad_out_scale_inv_row,
            False,
            b_fp8,
            b_scale_inv,
            not ctx.trans_b,
            ctx.out_dtype,
            False,
            granularity=ctx.config.granularity.value,
            default_backend=BackendType.CK.value,
        )

        b_grad = gemm_fp8_impl(
            a_fp8_col,
            a_scale_inv_col,
            not ctx.trans_a,
            grad_out_fp8_col,
            grad_out_scale_inv_col,
            False,
            ctx.out_dtype,
            ctx.trans_b,
            granularity=ctx.config.granularity.value,
            default_backend=BackendType.CK.value,
        )

        return a_grad, b_grad, None, None, None, None


class FP8GemmMXFunction(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx,
        a: Union[torch.Tensor, QuantizedTensor],
        b: Union[torch.Tensor, QuantizedTensor],
        trans_a: bool,
        trans_b: bool,
        out_dtype: torch.dtype,
        config: Float8QuantConfig,
    ):
        supported_mxfp8_backend, reason = check_mxfp8_support()
        assert supported_mxfp8_backend, reason

        a_scaling_recipe = ScalingRecipe()
        a_t_scaling_recipe = ScalingRecipe()
        if isinstance(a, QuantizedTensor):
            check_quantized_tensor(a, config, a_scaling_recipe, a_t_scaling_recipe)
            a_fp8 = a
        else:
            a_dtype = _get_fp8_dtype(config.format, True)
            a_fp8 = QuantizedTensor(
                a,
                a_dtype,
                config.granularity,
                block_size=config.block_size,
                keep_trans_cache=True,
                scaling_recipe=a_scaling_recipe,
                scaling_recipe_for_trans=a_t_scaling_recipe,
            )

        b_scaling_recipe = ScalingRecipe(use_2d_block=True)
        b_t_scaling_recipe = ScalingRecipe(use_2d_block=True)
        if isinstance(b, QuantizedTensor):
            check_quantized_tensor(b, config)
            b_fp8 = b
        else:
            b_dtype = _get_fp8_dtype(config.format, True)
            b_fp8 = QuantizedTensor(
                b,
                b_dtype,
                config.granularity,
                block_size=config.block_size,
                keep_trans_cache=True,
                scaling_recipe=b_scaling_recipe,
                scaling_recipe_for_trans=b_t_scaling_recipe,
            )

        # NT layout
        out = gemm_fp8_impl(
            a_fp8.data,
            a_fp8.scale_inv,
            False,
            b_fp8.data,
            b_fp8.scale_inv,
            True,
            out_dtype,
            False,
            granularity=config.granularity.value,
            default_backend=BackendType.TURBO.value,
        )

        a_t_fp8, a_t_scale_inv = a_fp8.t()
        b_t_fp8, b_t_scale_inv = b_fp8.t()
        ctx.save_for_backward(a_t_fp8, a_t_scale_inv, b_t_fp8, b_t_scale_inv)

        ctx.trans_a = trans_a
        ctx.trans_b = trans_b
        ctx.out_dtype = out_dtype
        ctx.config = config
        ctx.a_fp8_dtype = a_fp8.dtype
        ctx.b_fp8_dtype = b_fp8.dtype

        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        a_t_fp8, a_t_scale_inv, b_t_fp8, b_t_scale_inv = ctx.saved_tensors

        grad_out_dtype = _get_fp8_dtype(ctx.config.format, False)
        grad_out = grad_out.view(grad_out.shape[0], -1)

        grad_out_scaling_recipe = ScalingRecipe()
        grad_out_t_scaling_recipe = ScalingRecipe()
        grad_out_fp8 = QuantizedTensor(
            grad_out,
            grad_out_dtype,
            ctx.config.granularity,
            block_size=ctx.config.block_size,
            keep_trans_cache=True,
            scaling_recipe=grad_out_scaling_recipe,
            scaling_recipe_for_trans=grad_out_t_scaling_recipe,
        )

        # NOTE: convert NN layout to NT layout because MXFP8 only supports NT layout.
        grad_a = gemm_fp8_impl(
            grad_out_fp8.data,
            grad_out_fp8.scale_inv,
            False,
            b_t_fp8,
            b_t_scale_inv,
            True,
            ctx.out_dtype,
            False,
            granularity=ctx.config.granularity.value,
            default_backend=BackendType.TURBO.value,
        )

        grad_out_t_fp8, grad_out_t_scale_inv = grad_out_fp8.t()
        # NOTE: convert TN layout to NT layout because MXFP8 only supports NT layout.
        grad_b = gemm_fp8_impl(
            grad_out_t_fp8,
            grad_out_t_scale_inv,
            False,
            a_t_fp8,
            a_t_scale_inv,
            True,
            ctx.out_dtype,
            False,
            granularity=ctx.config.granularity.value,
            default_backend=BackendType.TURBO.value,
        )

        return grad_a, grad_b, None, None, None, None


def gemm_fp8(
    a: Union[torch.Tensor, QuantizedTensor],
    b: Union[torch.Tensor, QuantizedTensor],
    trans_a: bool = False,
    trans_b: bool = False,
    out_dtype: Union[torch.dtype, None] = None,
    config: Union[Float8QuantConfig, None] = None,
) -> torch.Tensor:
    """General matrix multiplication (GEMM) with FP8 quantization, supporting autograd.

    Automatically quantizes inputs to FP8 format during forward and backward passes
    to accelerate training and inference.

    Args:
        a: Input matrix A with shape (M, K), must be 2D tensor
        b: Input matrix B with shape (K, N) or (N, K), must be 2D tensor
        trans_a: Whether to transpose matrix A
        trans_b: Whether to transpose matrix B, if True B shape is (N, K)
        out_dtype: Output data type, defaults to None (auto-inferred)
        config: FP8 quantization config, defaults to None (uses TENSORWISE + E4M3)

    Returns:
        torch.Tensor: Output matrix with shape (M, N)

    Scaling Granularity (config.granularity):
        - TENSORWISE
        - ROWWISE
        - BLOCKWISE
        - MX_BLOCKWISE

    FP8 Format (config.format):
        - E4M3
        - E5M2

    Example::

        >>> # Basic usage
        >>> a = torch.randn(128, 512, device='cuda')
        >>> b = torch.randn(512, 256, device='cuda')
        >>> out = gemm_fp8(a, b)
        >>>
        >>> # ROWWISE quantization
        >>> config = Float8QuantConfig(
        ...     format=Format.E4M3,
        ...     granularity=ScalingGranularity.ROWWISE
        ... )
        >>> out = gemm_fp8(a, b, trans_b=True, config=config)

    """
    assert a.ndim == 2 and b.ndim == 2, "Only 2D tensors are supported"
    if out_dtype is None:
        out_dtype = torch.result_type(a, b)

    if config is None:
        config = Float8QuantConfig()

    args = (a, b, trans_a, trans_b, out_dtype, config)

    if config.granularity == ScalingGranularity.TENSORWISE:
        return FP8GemmTensorFunction.apply(*args)
    elif config.granularity == ScalingGranularity.ROWWISE:
        return FP8GemmRowFunction.apply(*args)
    elif config.granularity == ScalingGranularity.BLOCKWISE:
        return FP8GemmBlockFunction.apply(*args)
    elif config.granularity == ScalingGranularity.MX_BLOCKWISE:
        return FP8GemmMXFunction.apply(*args)
    else:
        raise ValueError(f"Unsupported FP8 ScalingGranularity: {config.granularity}")
