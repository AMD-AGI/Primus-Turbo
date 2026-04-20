###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from typing import Union

import torch

from primus_turbo.pytorch.core.backend import BackendType
from primus_turbo.pytorch.core.low_precision import (
    Float4QuantConfig,
    Format,
    ScalingGranularity,
    ScalingRecipe,
    check_mxfp4_support,
)
from primus_turbo.pytorch.core.quantized_tensor import (
    QuantizedTensor,
    check_quantized_tensor,
)
from primus_turbo.pytorch.kernels.gemm.gemm_fp4_impl import (
    enable_preshuffle,
    gemm_fp4_impl,
)

__all__ = ["gemm_fp4"]


class FP4GemmMXFunction(torch.autograd.Function):
    """
    MXFP4 scaling recipe reference: https://arxiv.org/pdf/2509.25149
    """

    @staticmethod
    def get_fp4_dtype(format: Format):
        if format == Format.E2M1_X2:
            return torch.float4_e2m1fn_x2
        else:
            raise ValueError(f"Unsupported FP4 format: {format}")

    @staticmethod
    def forward(
        ctx,
        a: Union[torch.Tensor, QuantizedTensor],
        b: Union[torch.Tensor, QuantizedTensor],
        trans_a: bool,
        trans_b: bool,
        out_dtype: torch.dtype,
        config: Float4QuantConfig,
    ):
        supported_mxfp4_backend, reason = check_mxfp4_support()
        assert supported_mxfp4_backend, reason

        a_scaling_recipe = ScalingRecipe(
            use_2d_block=False,
            use_sr=False,
            use_rht=False,
            shuffle_scale=enable_preshuffle(),
        )
        a_t_scaling_recipe = ScalingRecipe(
            use_2d_block=False,
            use_sr=False,
            use_rht=True,
            shuffle_scale=enable_preshuffle(),
            shuffle_out=enable_preshuffle(),
        )
        if isinstance(a, QuantizedTensor):
            check_quantized_tensor(a, config, a_scaling_recipe, a_t_scaling_recipe)
            a_fp4 = a
        else:
            a_dtype = FP4GemmMXFunction.get_fp4_dtype(config.format)
            a_fp4 = QuantizedTensor(
                a,
                a_dtype,
                config.granularity,
                block_size=config.block_size,
                keep_trans_cache=True,
                scaling_recipe=a_scaling_recipe,
                scaling_recipe_for_trans=a_t_scaling_recipe,
            )

        b_scaling_recipe = ScalingRecipe(
            use_2d_block=False,
            use_sr=False,
            use_rht=False,
            shuffle_scale=enable_preshuffle(),
        )
        b_t_scaling_recipe = ScalingRecipe(
            use_2d_block=False,
            use_sr=False,
            use_rht=True,
            shuffle_scale=enable_preshuffle(),
            shuffle_out=enable_preshuffle(),
        )
        if isinstance(b, QuantizedTensor):
            check_quantized_tensor(b, config, b_scaling_recipe, b_t_scaling_recipe)
            b_fp4 = b
        else:
            b_dtype = FP4GemmMXFunction.get_fp4_dtype(config.format)
            b_fp4 = QuantizedTensor(
                b,
                b_dtype,
                config.granularity,
                block_size=config.block_size,
                keep_trans_cache=True,
                scaling_recipe=b_scaling_recipe,
                scaling_recipe_for_trans=b_t_scaling_recipe,
            )

        # NT layout
        out = gemm_fp4_impl(
            a_fp4.data,
            a_fp4.scale_inv,
            False,
            b_fp4.data,
            b_fp4.scale_inv,
            True,
            out_dtype,
            False,
            granularity=config.granularity.value,
            default_backend=BackendType.HIPBLASLT.value,
        )

        a_t_fp4, a_t_scale_inv = a_fp4.t()
        b_t_fp4, b_t_scale_inv = b_fp4.t()
        ctx.save_for_backward(a_t_fp4, a_t_scale_inv, b_t_fp4, b_t_scale_inv)

        ctx.trans_a = trans_a
        ctx.trans_b = trans_b
        ctx.out_dtype = out_dtype
        ctx.config = config
        ctx.a_fp4_dtype = a_fp4.real_dtype
        ctx.b_fp4_dtype = b_fp4.real_dtype

        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        a_t_fp4, a_t_scale_inv, b_t_fp4, b_t_scale_inv = ctx.saved_tensors
        grad_out_dtype = FP4GemmMXFunction.get_fp4_dtype(
            ctx.config.format,
        )

        grad_out = grad_out.view(grad_out.shape[0], -1)

        grad_out_scaling_recipe = ScalingRecipe(
            use_2d_block=False,
            use_sr=False,
            use_rht=True,
            shuffle_scale=enable_preshuffle(),
        )
        grad_out_t_scaling_recipe = ScalingRecipe(
            use_2d_block=False,
            use_sr=False,
            use_rht=True,
            shuffle_scale=enable_preshuffle(),
        )
        grad_out_fp4 = QuantizedTensor(
            grad_out,
            grad_out_dtype,
            ctx.config.granularity,
            block_size=ctx.config.block_size,
            keep_trans_cache=True,
            scaling_recipe=grad_out_scaling_recipe,
            scaling_recipe_for_trans=grad_out_t_scaling_recipe,
        )

        # NOTE: convert NN layout to NT layout because MXFP4 only supports NT layout on hipblaslt.
        grad_a = gemm_fp4_impl(
            grad_out_fp4.data,
            grad_out_fp4.scale_inv,
            False,
            b_t_fp4,
            b_t_scale_inv,
            True,
            ctx.out_dtype,
            False,
            granularity=ctx.config.granularity.value,
            default_backend=BackendType.HIPBLASLT.value,
        )

        grad_out_t_fp4, grad_out_t_scale_inv = grad_out_fp4.t()
        # NOTE: convert TN layout to NT layout because MXFP4 only supports NT layout on hipblaslt.
        grad_b = gemm_fp4_impl(
            grad_out_t_fp4,
            grad_out_t_scale_inv,
            False,
            a_t_fp4,
            a_t_scale_inv,
            True,
            ctx.out_dtype,
            False,
            granularity=ctx.config.granularity.value,
            default_backend=BackendType.HIPBLASLT.value,
        )

        return grad_a, grad_b, None, None, None, None


def gemm_fp4(
    a: Union[torch.Tensor, QuantizedTensor],
    b: Union[torch.Tensor, QuantizedTensor],
    trans_a: bool = False,
    trans_b: bool = False,
    out_dtype: Union[torch.dtype, None] = None,
    config: Union[Float4QuantConfig, None] = None,
) -> torch.Tensor:
    """General matrix multiplication (GEMM) with FP4 quantization, supporting autograd.

    Automatically quantizes inputs to FP4 format during forward and backward passes
    to accelerate training and inference. When ``a`` or ``b`` is already a
    :class:`QuantizedTensor`, its quantized data / scale (and cached transpose)
    is reused directly, skipping redundant quantization.

    Args:
        a: Input matrix a with shape (M, K), must be 2D tensor. The A matrix should be activaton.
            Can also be a pre-quantized :class:`QuantizedTensor`.
        b: Input matrix b with shape (K, N) or (N, K), must be 2D tensor. The B matrix should be weight.
            Can also be a pre-quantized :class:`QuantizedTensor`.
        trans_a: Whether to transpose matrix a
        trans_b: Whether to transpose matrix b, if True b shape is (N, K)
        out_dtype: Output data type, defaults to None (auto-inferred)
        config: FP4 quantization config

    Returns:
        torch.Tensor: Output matrix with shape (M, N)

    Scaling Granularity (config.granularity):
        - MX_BLOCKWISE

    FP4 Format (config.format):
        - E2M1_X2

    Example::

        >>> # Basic usage
        >>> a = torch.randn(128, 512, device='cuda')
        >>> b = torch.randn(512, 256, device='cuda')
        >>> out = gemm_fp4(a, b)
        >>>
        >>> # ROWWISE quantization
        >>> config = Float4QuantConfig()
        >>> out = gemm_fp4(a, b, trans_b=True, config=config)

    """
    assert a.ndim == 2 and b.ndim == 2, "Only 2D tensors are supported"
    if out_dtype is None:
        out_dtype = torch.result_type(a, b)

    if config is None:
        config = Float4QuantConfig()

    args = (a, b, trans_a, trans_b, out_dtype, config)

    if config.granularity == ScalingGranularity.MX_BLOCKWISE:
        return FP4GemmMXFunction.apply(*args)
    else:
        raise ValueError(f"Unsupported FP4 ScalingGranularity: {config.granularity}")
