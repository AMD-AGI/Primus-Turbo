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
    MXScalingRecipe,
    ScalingGranularity,
    check_mxfp4_support,
)
from primus_turbo.pytorch.kernels.gemm.gemm_fp4_impl import (
    GEMMFP4HipBLASLtBackend,
    gemm_fp4_impl,
)
from primus_turbo.pytorch.ops.quantization import quantize_fp4

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
        activation: torch.Tensor,
        weight: torch.Tensor,
        trans_activation: bool,
        trans_weight: bool,
        out_dtype: torch.dtype,
        config: Float4QuantConfig,
    ):
        supported_mxfp4_backend, reason = check_mxfp4_support()
        assert supported_mxfp4_backend, reason

        assert (
            config.granularity == ScalingGranularity.MX_BLOCKWISE
        ), "MXFP4 only supports MX_BLOCKWISE granularity"

        activation_dtype = FP4GemmMXFunction.get_fp4_dtype(
            config.format,
        )
        weight_dtype = FP4GemmMXFunction.get_fp4_dtype(
            config.format,
        )

        activation_fp4, activation_scale_inv = quantize_fp4(
            activation,
            activation_dtype,
            config.granularity,
            block_size=config.block_size,
            axis=1,
            padding_align_size=GEMMFP4HipBLASLtBackend.HIPBLASLT_K_MULTIPLE,
            scaling_recipe=MXScalingRecipe(
                use_2d_block=False,
                use_sr=False,
                use_rht=False,
            ),
        )

        weight_fp4, weight_scale_inv = quantize_fp4(
            weight,
            weight_dtype,
            config.granularity,
            block_size=config.block_size,
            axis=1,
            padding_align_size=GEMMFP4HipBLASLtBackend.HIPBLASLT_K_MULTIPLE,
            scaling_recipe=MXScalingRecipe(
                use_2d_block=True,
                use_sr=False,
                use_rht=False,
            ),
        )

        # NT layout
        out = gemm_fp4_impl(
            activation_fp4,
            activation_scale_inv,
            False,
            weight_fp4,
            weight_scale_inv,
            True,
            out_dtype,
            False,
            granularity=config.granularity.value,
            default_backend=BackendType.HIPBLASLT.value,
        )

        ctx.save_for_backward(activation, weight)

        ctx.trans_activation = trans_activation
        ctx.trans_weight = trans_weight
        ctx.out_dtype = out_dtype
        ctx.config = config
        ctx.activation_fp4_dtype = activation_fp4.dtype
        ctx.weight_fp4_dtype = weight_fp4.dtype

        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        activation, weight = ctx.saved_tensors
        grad_out_dtype = FP4GemmMXFunction.get_fp4_dtype(
            ctx.config.format,
        )

        grad_out = grad_out.view(grad_out.shape[0], -1)

        grad_out_fp4, grad_out_scale_inv = quantize_fp4(
            grad_out,
            grad_out_dtype,
            ctx.config.granularity,
            block_size=ctx.config.block_size,
            axis=1,
            padding_align_size=GEMMFP4HipBLASLtBackend.HIPBLASLT_K_MULTIPLE,
            scaling_recipe=MXScalingRecipe(
                use_2d_block=False,
                use_sr=True,
                use_rht=False,
            ),
        )

        grad_out_t_fp4, grad_out_t_scale_inv = quantize_fp4(
            grad_out,
            grad_out_dtype,
            ctx.config.granularity,
            block_size=ctx.config.block_size,
            axis=0,
            padding_align_size=GEMMFP4HipBLASLtBackend.HIPBLASLT_K_MULTIPLE,
            scaling_recipe=MXScalingRecipe(
                use_2d_block=False,
                use_sr=True,
                use_rht=True,
            ),
        )

        activation_t_fp4, activation_t_scale_inv = quantize_fp4(
            activation,
            ctx.activation_fp4_dtype,
            ctx.config.granularity,
            block_size=ctx.config.block_size,
            axis=0,
            padding_align_size=GEMMFP4HipBLASLtBackend.HIPBLASLT_K_MULTIPLE,
            scaling_recipe=MXScalingRecipe(
                use_2d_block=False,
                use_sr=False,
                use_rht=True,
            ),
        )

        weight_t_fp4, weight_t_scale_inv = quantize_fp4(
            weight,
            ctx.weight_fp4_dtype,
            ctx.config.granularity,
            block_size=ctx.config.block_size,
            axis=0,
            padding_align_size=GEMMFP4HipBLASLtBackend.HIPBLASLT_K_MULTIPLE,
            scaling_recipe=MXScalingRecipe(
                use_2d_block=True,
                use_sr=True,
                use_rht=False,
            ),
        )

        # NOTE: convert NN layout to NT layout because MXFP4 only supports NT layout on hipblaslt.
        grad_activation = gemm_fp4_impl(
            grad_out_fp4,
            grad_out_scale_inv,
            False,
            weight_t_fp4,
            weight_t_scale_inv,
            True,
            ctx.out_dtype,
            False,
            granularity=ctx.config.granularity.value,
            default_backend=BackendType.HIPBLASLT.value,
        )

        # NOTE: convert TN layout to NT layout because MXFP4 only supports NT layout on hipblaslt.
        grad_weight = gemm_fp4_impl(
            grad_out_t_fp4,
            grad_out_t_scale_inv,
            False,
            activation_t_fp4,
            activation_t_scale_inv,
            True,
            ctx.out_dtype,
            False,
            granularity=ctx.config.granularity.value,
            default_backend=BackendType.HIPBLASLT.value,
        )

        return grad_activation, grad_weight, None, None, None, None


def gemm_fp4(
    activation: torch.Tensor,
    weight: torch.Tensor,
    trans_activation: bool = False,
    trans_weight: bool = False,
    out_dtype: Union[torch.dtype, None] = None,
    config: Union[Float4QuantConfig, None] = None,
) -> torch.Tensor:
    """General matrix multiplication (GEMM) with FP4 quantization, supporting autograd.

    Automatically quantizes inputs to FP4 format during forward and backward passes
    to accelerate training and inference.

    Args:
        activation: Input matrix activation with shape (M, K), must be 2D tensor
        weight: Input matrix weight with shape (K, N) or (N, K), must be 2D tensor
        trans_activation: Whether to transpose matrix activation
        trans_weight: Whether to transpose matrix weight, if True weight shape is (N, K)
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
        >>> activation = torch.randn(128, 512, device='cuda')
        >>> weight = torch.randn(512, 256, device='cuda')
        >>> out = gemm_fp4(activation, weight)
        >>>
        >>> # ROWWISE quantization
        >>> config = Float4QuantConfig()
        >>> out = gemm_fp4(activation, weight, trans_weight=True, config=config)

    """
    assert activation.ndim == 2 and weight.ndim == 2, "Only 2D tensors are supported"
    if out_dtype is None:
        out_dtype = torch.result_type(activation, weight)

    if config is None:
        config = Float4QuantConfig()

    args = (activation, weight, trans_activation, trans_weight, out_dtype, config)

    if config.granularity == ScalingGranularity.MX_BLOCKWISE:
        return FP4GemmMXFunction.apply(*args)
    else:
        raise ValueError(f"Unsupported FP4 ScalingGranularity: {config.granularity}")
