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
    check_mxfp4_support,
)
from primus_turbo.pytorch.kernels.gemm.gemm_fp4_impl import gemm_fp4_impl
from primus_turbo.pytorch.ops.quantization import quantize_fp4

__all__ = ["gemm_fp4"]


def replicate_scale_inv(scale_inv: torch.Tensor, block_size: int):
    scale_m = scale_inv.size(0) * block_size

    return scale_inv.unsqueeze(1).expand(-1, block_size, -1).reshape(scale_m, -1)


class FP4GemmMXFunction(torch.autograd.Function):

    HIPBLASLT_M_MULTIPLE = 16
    HIPBLASLT_N_MULTIPLE = 16
    HIPBLASLT_K_MULTIPLE = 128

    @staticmethod
    def get_fp4_dtype(format: Format):
        if format == Format.E2M1_X2:
            return torch.float4_e2m1fn_x2
        else:
            raise ValueError(f"Unsupported FP4 format: {format}")

    @staticmethod
    def forward(
        ctx,
        a: torch.Tensor,
        b: torch.Tensor,
        trans_a: bool,
        trans_b: bool,
        out_dtype: torch.dtype,
        config: Float4QuantConfig,
    ):
        supported_mxfp4_backend, reason = check_mxfp4_support()
        assert supported_mxfp4_backend, reason

        assert config.granularity == ScalingGranularity.MX_BLOCKWISE
        assert (
            trans_a == False and trans_b == True
        ), "MXFP4 GEMM only supports trans_a=False and trans_b=True."
        assert (
            a.size(0) % __class__.HIPBLASLT_M_MULTIPLE == 0
            and b.size(0) % __class__.HIPBLASLT_N_MULTIPLE == 0
        ), f"MXFP4 requires M are multiples of {__class__.HIPBLASLT_M_MULTIPLE} and N are multiples of {__class__.HIPBLASLT_N_MULTIPLE}."
        assert (
            a.size(1) % __class__.HIPBLASLT_N_MULTIPLE == 0
            and b.size(1) % __class__.HIPBLASLT_N_MULTIPLE == 0
        ), f"MXFP4 requires K are multiples of {__class__.HIPBLASLT_N_MULTIPLE}."

        a_dtype = FP4GemmMXFunction.get_fp4_dtype(
            config.format,
        )
        b_dtype = FP4GemmMXFunction.get_fp4_dtype(
            config.format,
        )

        a_fp4, a_scale_inv = quantize_fp4(
            a,
            a_dtype,
            config.granularity,
            block_size=config.block_size,
            axis=1,
            padding_align_size=__class__.HIPBLASLT_K_MULTIPLE,
            scaling_recipe=config.scaling_recipe["a_fwd"],
        )

        b_fp4, b_scale_inv = quantize_fp4(
            b,
            b_dtype,
            config.granularity,
            block_size=config.block_size,
            axis=1,
            padding_align_size=__class__.HIPBLASLT_K_MULTIPLE,
            scaling_recipe=config.scaling_recipe["b_fwd"],
        )

        # replicate scale_inv for 2D block
        a_scale_inv = (
            replicate_scale_inv(a_scale_inv, config.block_size)
            if config.scaling_recipe["a_fwd"].use_2d_block
            else a_scale_inv
        )
        b_scale_inv = (
            replicate_scale_inv(b_scale_inv, config.block_size)
            if config.scaling_recipe["b_fwd"].use_2d_block
            else b_scale_inv
        )

        # NT layout
        out = gemm_fp4_impl(
            a_fp4,
            a_scale_inv,
            False,
            b_fp4,
            b_scale_inv,
            True,
            out_dtype,
            False,
            granularity=config.granularity.value,
            default_backend=BackendType.HIPBLASLT.value,
        )

        ctx.save_for_backward(a, b)

        ctx.trans_a = trans_a
        ctx.trans_b = trans_b
        ctx.out_dtype = out_dtype
        ctx.config = config
        ctx.a_fp4_dtype = a_fp4.dtype
        ctx.b_fp4_dtype = b_fp4.dtype

        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        a, b = ctx.saved_tensors
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
            padding_align_size=__class__.HIPBLASLT_K_MULTIPLE,
        )

        grad_out_t = grad_out.T.contiguous()
        grad_out_t_fp4, grad_out_t_scale_inv = quantize_fp4(
            grad_out_t,
            grad_out_dtype,
            ctx.config.granularity,
            block_size=ctx.config.block_size,
            axis=1,
            padding_align_size=__class__.HIPBLASLT_K_MULTIPLE,
            scaling_recipe=ctx.config.scaling_recipe["grad_bwd"],
        )

        a_t = a.T.contiguous()
        a_t_fp4, a_t_scale_inv = quantize_fp4(
            a_t,
            ctx.a_fp4_dtype,
            ctx.config.granularity,
            block_size=ctx.config.block_size,
            axis=1,
            padding_align_size=__class__.HIPBLASLT_K_MULTIPLE,
            scaling_recipe=ctx.config.scaling_recipe["a_bwd"],
        )
        b_t = b.T.contiguous()
        b_t_fp4, b_t_scale_inv = quantize_fp4(
            b_t,
            ctx.b_fp4_dtype,
            ctx.config.granularity,
            block_size=ctx.config.block_size,
            axis=1,
            padding_align_size=__class__.HIPBLASLT_K_MULTIPLE,
            scaling_recipe=ctx.config.scaling_recipe["b_bwd"],
        )

        # replicate scale_inv for 2D block
        grad_out_scale_inv = (
            replicate_scale_inv(grad_out_scale_inv, ctx.config.block_size)
            if ctx.config.scaling_recipe["grad_bwd"].use_2d_block
            else grad_out_scale_inv
        )
        b_t_scale_inv = (
            replicate_scale_inv(b_t_scale_inv, ctx.config.block_size)
            if ctx.config.scaling_recipe["b_bwd"].use_2d_block
            else b_t_scale_inv
        )

        # NOTE: convert NN layout to NT layout because MXFP4 only supports NT layout on hipblaslt.
        grad_a = gemm_fp4_impl(
            grad_out_fp4,
            grad_out_scale_inv,
            False,
            b_t_fp4,
            b_t_scale_inv,
            True,
            ctx.out_dtype,
            False,
            granularity=ctx.config.granularity.value,
            default_backend=BackendType.HIPBLASLT.value,
        )

        # replicate scale_inv for 2D block
        grad_out_t_scale_inv = (
            replicate_scale_inv(grad_out_t_scale_inv, ctx.config.block_size)
            if ctx.config.scaling_recipe["grad_bwd"].use_2d_block
            else grad_out_t_scale_inv
        )
        a_t_scale_inv = (
            replicate_scale_inv(a_t_scale_inv, ctx.config.block_size)
            if ctx.config.scaling_recipe["a_bwd"].use_2d_block
            else a_t_scale_inv
        )

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
    a: torch.Tensor,
    b: torch.Tensor,
    trans_a: bool = False,
    trans_b: bool = False,
    out_dtype: Union[torch.dtype, None] = None,
    config: Union[Float4QuantConfig, None] = None,
) -> torch.Tensor:
    """General matrix multiplication (GEMM) with FP4 quantization, supporting autograd.

    Automatically quantizes inputs to FP4 format during forward and backward passes
    to accelerate training and inference.

    Args:
        a: Input matrix A with shape (M, K), must be 2D tensor
        b: Input matrix B with shape (K, N) or (N, K), must be 2D tensor
        trans_a: Whether to transpose matrix A
        trans_b: Whether to transpose matrix B, if True B shape is (N, K)
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
        out_dtype = torch.bfloat16

    if config is None:
        config = Float4QuantConfig()

    args = (a, b, trans_a, trans_b, out_dtype, config)

    if config.granularity == ScalingGranularity.MX_BLOCKWISE:
        return FP4GemmMXFunction.apply(*args)
    else:
        raise ValueError(f"Unsupported FP4 ScalingGranularity: {config.granularity}")
