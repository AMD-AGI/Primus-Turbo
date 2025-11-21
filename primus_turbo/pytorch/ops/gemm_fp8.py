###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from typing import Union

import torch

from primus_turbo.pytorch.core.low_precision import (
    Float8QuantConfig,
    Format,
    ScalingGranularity,
    check_mxfp8_support,
    float8_e4m3,
    float8_e5m2,
)
from primus_turbo.pytorch.kernels.gemm.gemm_fp8_impl import (
    gemm_fp8_impl,
    quant_fp8_blockwise_for_weight_impl,
)
from primus_turbo.pytorch.kernels.quantization.quantization_impl import (
    quant_fp8_blockwise_impl,
)
from primus_turbo.pytorch.ops.quantization import quantize_fp8

__all__ = ["gemm_fp8"]


class FP8GemmTensorFunction(torch.autograd.Function):

    @staticmethod
    def get_fp8_dtype(format: Format, is_fwd_stage: bool):
        if format == Format.E4M3:
            return float8_e4m3
        elif format == Format.E5M2:
            return float8_e5m2
        else:
            raise ValueError(f"Unsupported FP8 format: {format}")

    @staticmethod
    def forward(
        ctx,
        a: torch.Tensor,
        b: torch.Tensor,
        trans_a: bool,  # trans_a has to be False
        trans_b: bool,
        out_dtype: torch.dtype,
        config: Float8QuantConfig,
    ):
        assert trans_a == False, "trans_a has to be False"
        a_dtype = FP8GemmTensorFunction.get_fp8_dtype(config.format, True)
        b_dtype = FP8GemmTensorFunction.get_fp8_dtype(config.format, True)

        a_fp8, a_scale_inv = quantize_fp8(a, a_dtype, config.granularity)
        b_fp8, b_scale_inv = quantize_fp8(b, b_dtype, config.granularity)

        backend = "hipblaslt" if trans_b else "ck"

        out = gemm_fp8_impl(
            a_fp8,
            a_scale_inv,
            trans_a,
            b_fp8,
            b_scale_inv,
            trans_b,
            out_dtype,
            False,
            backend=backend,
            granularity=config.granularity,
        )
        ctx.save_for_backward(a_fp8, a_scale_inv, b_fp8, b_scale_inv)
        ctx.trans_a = trans_a
        ctx.trans_b = trans_b
        ctx.out_dtype = out_dtype
        ctx.config = config

        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        (a_fp8, a_scale_inv, b_fp8, b_scale_inv) = ctx.saved_tensors
        grad_out_dtype = FP8GemmTensorFunction.get_fp8_dtype(ctx.config.format, False)

        grad_out_fp8, grad_out_scale_inv = quantize_fp8(grad_out, grad_out_dtype, ctx.config.granularity)

        a_grad = gemm_fp8_impl(
            grad_out_fp8,
            grad_out_scale_inv,
            False,
            b_fp8,
            b_scale_inv,
            not ctx.trans_b,
            ctx.out_dtype,
            ctx.trans_a,
            backend="ck",
            granularity=ctx.config.granularity,
        )

        lhs, rhs = (grad_out_fp8, a_fp8) if ctx.trans_b else (a_fp8, grad_out_fp8)
        lhs_scale, rhs_scale = (
            (grad_out_scale_inv, a_scale_inv) if ctx.trans_b else (a_scale_inv, grad_out_scale_inv)
        )

        b_grad = gemm_fp8_impl(
            lhs,
            lhs_scale,
            not ctx.trans_a,
            rhs,
            rhs_scale,
            False,
            ctx.out_dtype,
            ctx.trans_b,
            backend="ck",
            granularity=ctx.config.granularity,
        )

        return (a_grad, b_grad, None, None, None, None)


class FP8GemmRowFunction(torch.autograd.Function):

    @staticmethod
    def get_fp8_dtype(format: Format, is_fwd_stage: bool):
        if format == Format.E4M3:
            return float8_e4m3
        elif format == Format.E5M2:
            return float8_e5m2
        elif format == Format.HYBRID:
            return float8_e4m3 if is_fwd_stage else float8_e5m2
        else:
            raise ValueError(f"Unsupported FP8 format: {format}")

    @staticmethod
    def forward(
        ctx,
        a: torch.Tensor,
        b: torch.Tensor,
        trans_a: bool,  # trans_a has to be False
        trans_b: bool,
        out_dtype: torch.dtype,
        config: Float8QuantConfig,
    ):
        assert trans_a == False, "trans_a has to be False"
        a_dtype = FP8GemmRowFunction.get_fp8_dtype(config.format, True)
        b_dtype = FP8GemmRowFunction.get_fp8_dtype(config.format, True)

        a_fp8_row, a_scale_inv_row = quantize_fp8(a, a_dtype, config.granularity, axis=-1)
        b_fp8_row, b_scale_inv_row = quantize_fp8(
            b, b_dtype, config.granularity, axis=(-1 if trans_b else -2)
        )

        out = gemm_fp8_impl(
            a_fp8_row,
            a_scale_inv_row,
            trans_a,
            b_fp8_row,
            b_scale_inv_row,
            trans_b,
            out_dtype,
            False,
            backend="ck",
            granularity=config.granularity,
        )

        a_fp8_col, a_scale_inv_col = quantize_fp8(a, a_dtype, config.granularity, axis=-2)
        b_fp8_col, b_scale_inv_col = quantize_fp8(
            b, b_dtype, config.granularity, axis=(-2 if trans_b else -1)
        )

        ctx.save_for_backward(a_fp8_col, a_scale_inv_col, b_fp8_col, b_scale_inv_col)
        ctx.trans_a = trans_a
        ctx.trans_b = trans_b
        ctx.out_dtype = out_dtype
        ctx.config = config

        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        (a_fp8_col, a_scale_inv_col, b_fp8_col, b_scale_inv_col) = ctx.saved_tensors
        grad_out_dtype = FP8GemmRowFunction.get_fp8_dtype(ctx.config.format, False)

        grad_out_fp8_row, grad_out_scale_inv_row = quantize_fp8(
            grad_out, grad_out_dtype, ctx.config.granularity, axis=-1
        )

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
            backend="ck",
            granularity=ctx.config.granularity,
        )

        grad_out_fp8_col, grad_out_scale_inv_col = quantize_fp8(
            grad_out, grad_out_dtype, ctx.config.granularity, axis=-2
        )
        lhs, rhs = (grad_out_fp8_col, a_fp8_col) if ctx.trans_b else (a_fp8_col, grad_out_fp8_col)
        lhs_scale, rhs_scale = (
            (grad_out_scale_inv_col, a_scale_inv_col)
            if ctx.trans_b
            else (a_scale_inv_col, grad_out_scale_inv_col)
        )

        # TN
        b_grad = gemm_fp8_impl(
            lhs,
            lhs_scale,
            not ctx.trans_a,
            rhs,
            rhs_scale,
            False,
            ctx.out_dtype,
            ctx.trans_b,
            backend="ck",
            granularity=ctx.config.granularity,
        )

        return (a_grad, b_grad, None, None, None, None)


class FP8GemmBlockFunction(torch.autograd.Function):
    @staticmethod
    def get_fp8_dtype(format: Format, is_fwd_stage: bool):
        if format == Format.E4M3:
            return float8_e4m3
        elif format == Format.E5M2:
            return float8_e5m2
        else:
            raise ValueError(f"Unsupported FP8 format: {format}")

    @staticmethod
    def forward(
        ctx,
        a: torch.Tensor,
        b: torch.Tensor,
        trans_a: bool,
        trans_b: bool,
        out_dtype: torch.dtype,
        config: Float8QuantConfig,
    ):
        assert config.granularity == ScalingGranularity.BLOCKWISE
        assert trans_a == False
        a_dtype = FP8GemmBlockFunction.get_fp8_dtype(config.format, True)
        b_dtype = FP8GemmBlockFunction.get_fp8_dtype(config.format, True)

        a_fp8_row, a_scale_inv_row = quant_fp8_blockwise_impl(
            a, a_dtype, axis=1, block_size=config.block_size
        )
        b_fp8, b_scale_inv = quant_fp8_blockwise_for_weight_impl(b, b_dtype, block_size=config.block_size)

        if not trans_b:
            b_scale_inv = b_scale_inv.transpose(-1, -2)

        out = gemm_fp8_impl(
            a_fp8_row,
            a_scale_inv_row,
            trans_a,
            b_fp8,
            b_scale_inv,
            trans_b,
            out_dtype,
            False,
            backend="ck",
            granularity=config.granularity,
        )
        ctx.save_for_backward(a, b_fp8, b_scale_inv)
        ctx.trans_a = trans_a
        ctx.trans_b = trans_b
        ctx.out_dtype = out_dtype
        ctx.config = config
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        if not grad_out.is_contiguous():
            grad_out = grad_out.contiguous()

        a, b_fp8, b_scale_inv = ctx.saved_tensors
        grad_out_dtype = FP8GemmBlockFunction.get_fp8_dtype(ctx.config.format, False)
        a_dtype = FP8GemmBlockFunction.get_fp8_dtype(ctx.config.format, False)

        # Quantize grad_out in both row-wise and column-wise directions:
        # - row-wise: for dgrad (grad_x)
        # - col-wise: for wgrad (grad_w)
        grad_out_fp8_row, grad_out_scale_inv_row = quant_fp8_blockwise_impl(
            grad_out, grad_out_dtype, -1, ctx.config.block_size
        )
        grad_out_fp8_col, grad_out_scale_inv_col = quant_fp8_blockwise_impl(
            grad_out, grad_out_dtype, -2, ctx.config.block_size
        )

        # TODO: dequant + quant kernel
        a_fp8_col, a_scale_inv_col = quant_fp8_blockwise_impl(
            a, a_dtype, axis=0, block_size=ctx.config.block_size
        )

        a_grad = gemm_fp8_impl(
            grad_out_fp8_row,
            grad_out_scale_inv_row,
            False,
            b_fp8,
            b_scale_inv.transpose(-1, -2),
            not ctx.trans_b,
            ctx.out_dtype,
            False,
            backend="ck",
            granularity=ctx.config.granularity,
        )
        # BGrad
        lhs, rhs = (grad_out_fp8_col, a_fp8_col) if ctx.trans_b else (a_fp8_col, grad_out_fp8_col)
        lhs_scale_inv, rhs_scale_inv = (
            (grad_out_scale_inv_col, a_scale_inv_col)
            if ctx.trans_b
            else (a_scale_inv_col, grad_out_scale_inv_col)
        )

        b_grad = gemm_fp8_impl(
            lhs,
            lhs_scale_inv.transpose(-1, -2),
            not ctx.trans_a,
            rhs,
            rhs_scale_inv.transpose(-1, -2),
            False,
            ctx.out_dtype,
            False,
            backend="ck",
            granularity=ctx.config.granularity,
        )
        return a_grad, b_grad, None, None, None, None


class FP8GemmMXFunction(torch.autograd.Function):

    MXFP8_GEMM_BACKEND = "hipblaslt"
    HIPBLASLT_M_MULTIPLE = 16
    HIPBLASLT_N_MULTIPLE = 16
    HIPBLASLT_K_MULTIPLE = 128

    @staticmethod
    def get_fp8_dtype(format: Format, is_fwd_stage: bool):
        if format == Format.E4M3:
            return float8_e4m3
        elif format == Format.E5M2:
            return float8_e5m2
        elif format == Format.HYBRID:
            return float8_e4m3 if is_fwd_stage else float8_e5m2
        else:
            raise ValueError(f"Unsupported FP8 format: {format}")

    @staticmethod
    def forward(
        ctx,
        a: torch.Tensor,
        b: torch.Tensor,
        trans_a: bool,
        trans_b: bool,
        out_dtype: torch.dtype,
        config: Float8QuantConfig,
    ):
        supported_mxfp8_backend, reason = check_mxfp8_support()
        assert supported_mxfp8_backend, reason

        assert config.granularity == ScalingGranularity.MX_BLOCKWISE
        assert (
            trans_a == False and trans_b == True
        ), "MXFP8 GEMM only supports trans_a=False and trans_b=True."
        assert (
            a.size(0) % __class__.HIPBLASLT_M_MULTIPLE == 0
            and b.size(0) % __class__.HIPBLASLT_N_MULTIPLE == 0
        ), "MXFP8 requires M and N are multiples of 16."

        a_dtype = FP8GemmMXFunction.get_fp8_dtype(config.format, True)
        b_dtype = FP8GemmMXFunction.get_fp8_dtype(config.format, True)

        # NOTE: Padding k dim to 128 multiple for HIPBLASLT.
        k_padding_size = (
            (a.size(1) + __class__.HIPBLASLT_K_MULTIPLE - 1) // __class__.HIPBLASLT_K_MULTIPLE
        ) * __class__.HIPBLASLT_K_MULTIPLE - a.size(1)
        if k_padding_size > 0:
            a = torch.concat(
                [
                    a,
                    torch.zeros(a.size(0), k_padding_size, dtype=a.dtype, device=a.device),
                ],
                dim=1,
            )
            b = torch.concat(
                [
                    b,
                    torch.zeros(b.size(0), k_padding_size, dtype=b.dtype, device=b.device),
                ],
                dim=1,
            )

        a_fp8, a_scale_inv = quantize_fp8(a, a_dtype, config.granularity, block_size=config.block_size)
        b_fp8, b_scale_inv = quantize_fp8(b, b_dtype, config.granularity, block_size=config.block_size)

        # NT layout
        out = gemm_fp8_impl(
            a_fp8,
            a_scale_inv,
            False,
            b_fp8,
            b_scale_inv,
            True,
            out_dtype,
            False,
            backend=__class__.MXFP8_GEMM_BACKEND,
            granularity=config.granularity,
        )

        ctx.save_for_backward(a, b)

        ctx.trans_a = trans_a
        ctx.trans_b = trans_b
        ctx.out_dtype = out_dtype
        ctx.config = config
        ctx.a_fp8_dtype = a_fp8.dtype
        ctx.b_fp8_dtype = b_fp8.dtype
        ctx.k_padding_size = k_padding_size

        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        a, b = ctx.saved_tensors
        grad_out_dtype = FP8GemmMXFunction.get_fp8_dtype(ctx.config.format, False)

        grad_out = grad_out.view(grad_out.shape[0], -1)
        grad_out_t = grad_out.T.contiguous()

        # TODO(ruibzhan): fuse transpose with quantize kernel
        a_t = a.T.contiguous()
        b_t = b.T.contiguous()

        # NOTE: Padding k dim to 128 multiple for HIPBLASLT.
        grad_out_k_padding_size = (
            (grad_out.size(1) + __class__.HIPBLASLT_K_MULTIPLE - 1) // __class__.HIPBLASLT_K_MULTIPLE
        ) * __class__.HIPBLASLT_K_MULTIPLE - grad_out.size(1)
        if grad_out_k_padding_size > 0:
            grad_out = torch.concat(
                [
                    grad_out,
                    torch.zeros(
                        grad_out.size(0),
                        grad_out_k_padding_size,
                        dtype=grad_out.dtype,
                        device=grad_out.device,
                    ),
                ],
                dim=1,
            )
            b_t = torch.concat(
                [
                    b_t,
                    torch.zeros(
                        b_t.size(0),
                        grad_out_k_padding_size,
                        dtype=b_t.dtype,
                        device=b_t.device,
                    ),
                ],
                dim=1,
            )

        # NOTE: Padding k dim to 128 multiple for HIPBLASLT.
        grad_out_t_k_padding_size = (
            (grad_out_t.size(1) + __class__.HIPBLASLT_K_MULTIPLE - 1) // __class__.HIPBLASLT_K_MULTIPLE
        ) * __class__.HIPBLASLT_K_MULTIPLE - grad_out_t.size(1)
        if grad_out_t_k_padding_size > 0:
            grad_out_t = torch.concat(
                [
                    grad_out_t,
                    torch.zeros(
                        grad_out_t.size(0),
                        grad_out_t_k_padding_size,
                        dtype=grad_out_t.dtype,
                        device=grad_out_t.device,
                    ),
                ],
                dim=1,
            )
            a_t = torch.concat(
                [
                    a_t,
                    torch.zeros(
                        a_t.size(0),
                        grad_out_t_k_padding_size,
                        dtype=a_t.dtype,
                        device=a_t.device,
                    ),
                ],
                dim=1,
            )

        grad_out_fp8, grad_out_scale_inv = quantize_fp8(
            grad_out, grad_out_dtype, ctx.config.granularity, block_size=ctx.config.block_size
        )
        grad_out_t_fp8, grad_out_t_scale_inv = quantize_fp8(
            grad_out_t, grad_out_dtype, ctx.config.granularity, block_size=ctx.config.block_size
        )

        a_t_fp8, a_t_scale_inv = quantize_fp8(
            a_t, ctx.a_fp8_dtype, ctx.config.granularity, block_size=ctx.config.block_size
        )
        b_t_fp8, b_t_scale_inv = quantize_fp8(
            b_t, ctx.b_fp8_dtype, ctx.config.granularity, block_size=ctx.config.block_size
        )

        # NOTE: convert NN layout to NT layout because MXFP8 only supports NT layout on hipblaslt.
        grad_a = gemm_fp8_impl(
            grad_out_fp8,
            grad_out_scale_inv,
            False,
            b_t_fp8,
            b_t_scale_inv,
            True,
            ctx.out_dtype,
            False,
            backend=__class__.MXFP8_GEMM_BACKEND,
            granularity=ctx.config.granularity,
        )

        # NOTE: convert TN layout to NT layout because MXFP8 only supports NT layout on hipblaslt.
        grad_b = gemm_fp8_impl(
            grad_out_t_fp8,
            grad_out_t_scale_inv,
            False,
            a_t_fp8,
            a_t_scale_inv,
            True,
            ctx.out_dtype,
            False,
            backend=__class__.MXFP8_GEMM_BACKEND,
            granularity=ctx.config.granularity,
        )

        if ctx.k_padding_size > 0:
            grad_a = grad_a[:, : -ctx.k_padding_size]
            grad_b = grad_b[:, : -ctx.k_padding_size]

        return grad_a, grad_b, None, None, None, None


def gemm_fp8(
    a: torch.Tensor,
    b: torch.Tensor,
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
