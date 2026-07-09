###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from typing import Optional, Union

import torch

from primus_turbo.pytorch.core.backend import (
    BackendType,
)
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
    QuantizedTensorPair,
    check_quantized_tensor,
)
from primus_turbo.pytorch.core.utils import is_gfx942
from primus_turbo.pytorch.kernels.gemm.gemm_fp8_impl import gemm_fp8_impl
from primus_turbo.pytorch.ops.quantization import (
    quantize_fp8,
    quantize_fp8_with_trans,
    quantize_fp8_with_trans_flydsl,
)

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


def _deter_use_nt_layout_gemm_in_bwd(trans_a: bool, trans_b: bool):
    if is_gfx942():
        return False

    # NOTE: the non-NT layout gemm is not optimized for mi350/mi450.
    # Force to use NT layout GEMM in backward for now.
    return trans_a == False and trans_b == True


class FP8GemmTensorFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        a: Union[torch.Tensor, QuantizedTensor],
        b: Union[torch.Tensor, QuantizedTensor],
        a_t: Optional[QuantizedTensor],
        b_t: Optional[QuantizedTensor],
        trans_a: bool,
        trans_b: bool,
        out_dtype: torch.dtype,
        config: Float8QuantConfig,
    ):
        use_nt_layout_gemm_in_bwd = _deter_use_nt_layout_gemm_in_bwd(trans_a, trans_b)

        if isinstance(a, QuantizedTensor):
            quantized_a = a
            check_quantized_tensor(quantized_a, config)
        else:
            a_dtype = _get_fp8_dtype(config.format, True)
            quantized_a = QuantizedTensor.quantize(
                a,
                a_dtype,
                config.granularity,
                axis=-1,
                block_size=config.block_size,
            )

        if use_nt_layout_gemm_in_bwd:
            if a_t is not None and isinstance(a_t, QuantizedTensor):
                quantized_a_t = a_t
            else:
                quantized_a_t = quantized_a.t().contiguous()

        if isinstance(b, QuantizedTensor):
            quantized_b = b
            check_quantized_tensor(quantized_b, config)
        else:
            b_dtype = _get_fp8_dtype(config.format, True)
            quantized_b = QuantizedTensor.quantize(
                b,
                b_dtype,
                config.granularity,
                axis=-1,
                block_size=config.block_size,
            )

        if use_nt_layout_gemm_in_bwd:
            if b_t is not None and isinstance(b_t, QuantizedTensor):
                quantized_b_t = b_t
            else:
                quantized_b_t = quantized_b.t().contiguous()

        out = gemm_fp8_impl(
            quantized_a.qdata,
            quantized_a.scale_inv,
            trans_a,
            quantized_b.qdata,
            quantized_b.scale_inv,
            trans_b,
            out_dtype,
            False,
            granularity=config.granularity.value,
            default_backend=BackendType.HIPBLASLT.value,
        )

        if use_nt_layout_gemm_in_bwd:
            ctx.save_for_backward(
                quantized_a_t.qdata,
                quantized_a_t.scale_inv,
                quantized_b_t.qdata,
                quantized_b_t.scale_inv,
            )
        else:
            ctx.save_for_backward(
                quantized_a.qdata, quantized_a.scale_inv, quantized_b.qdata, quantized_b.scale_inv
            )

        ctx.trans_a = trans_a
        ctx.trans_b = trans_b
        ctx.out_dtype = out_dtype
        ctx.config = config
        ctx.use_nt_layout_gemm_in_bwd = use_nt_layout_gemm_in_bwd

        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        if not grad_out.is_contiguous():
            grad_out = grad_out.contiguous()

        if ctx.use_nt_layout_gemm_in_bwd:
            a_fp8_t, a_t_scale_inv, b_fp8_t, b_t_scale_inv = ctx.saved_tensors
        else:
            a_fp8, a_scale_inv, b_fp8, b_scale_inv = ctx.saved_tensors

        grad_out_dtype = _get_fp8_dtype(ctx.config.format, False)

        quantized_grad_out = QuantizedTensor.quantize(
            grad_out,
            grad_out_dtype,
            ctx.config.granularity,
            axis=-1,
        )

        if ctx.use_nt_layout_gemm_in_bwd:
            a_grad = gemm_fp8_impl(
                quantized_grad_out.qdata,
                quantized_grad_out.scale_inv,
                False,
                b_fp8_t,
                b_t_scale_inv,
                True,
                ctx.out_dtype,
                False,
                granularity=ctx.config.granularity.value,
                default_backend=BackendType.HIPBLASLT.value,
            )
        else:
            a_grad = gemm_fp8_impl(
                quantized_grad_out.qdata,
                quantized_grad_out.scale_inv,
                False,
                b_fp8,
                b_scale_inv,
                not ctx.trans_b,
                ctx.out_dtype,
                ctx.trans_a,
                granularity=ctx.config.granularity.value,
                default_backend=BackendType.HIPBLASLT.value,
            )

        if ctx.use_nt_layout_gemm_in_bwd:
            quantized_grad_out_t = quantized_grad_out.t().contiguous()

            b_grad = gemm_fp8_impl(
                a_fp8_t,
                a_t_scale_inv,
                False,
                quantized_grad_out_t.qdata,
                quantized_grad_out_t.scale_inv,
                True,
                ctx.out_dtype,
                ctx.trans_b,
                granularity=ctx.config.granularity.value,
                default_backend=BackendType.HIPBLASLT.value,
            )
        else:
            b_grad = gemm_fp8_impl(
                a_fp8,
                a_scale_inv,
                not ctx.trans_a,
                quantized_grad_out.qdata,
                quantized_grad_out.scale_inv,
                False,
                ctx.out_dtype,
                ctx.trans_b,
                granularity=ctx.config.granularity.value,
                default_backend=BackendType.HIPBLASLT.value,
            )

        return (
            a_grad,  # a
            b_grad,  # b
            None,  # a_t
            None,  # b_t
            None,  # trans_a
            None,  # trans_b
            None,  # out_dtype
            None,  # config
        )


class FP8GemmRowFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        a: Union[torch.Tensor, QuantizedTensor],
        b: Union[torch.Tensor, QuantizedTensor],
        a_t: Optional[QuantizedTensor],
        b_t: Optional[QuantizedTensor],
        trans_a: bool,
        trans_b: bool,
        out_dtype: torch.dtype,
        config: Float8QuantConfig,
    ):
        assert trans_a == False, "trans_a has to be False"

        if isinstance(a, QuantizedTensor):
            quantized_a = a
            check_quantized_tensor(quantized_a, config, axis=-1)
        else:
            a_dtype = _get_fp8_dtype(config.format, True)
            quantized_a = QuantizedTensor.quantize(
                a,
                a_dtype,
                config.granularity,
                axis=-1,
                block_size=config.block_size,
            )

        if a_t is None:
            quantized_a_t = QuantizedTensor.quantize(
                quantized_a.dequantize(),
                quantized_a.real_dtype,
                config.granularity,
                axis=-2,
                block_size=config.block_size,
            )
        else:
            assert isinstance(a_t, QuantizedTensor)
            quantized_a_t = a_t

        if isinstance(b, QuantizedTensor):
            check_quantized_tensor(b, config, axis=-1 if trans_b else -2)
            quantized_b = b
        else:
            b_dtype = _get_fp8_dtype(config.format, True)
            quantized_b = QuantizedTensor.quantize(
                b,
                b_dtype,
                config.granularity,
                axis=-1 if trans_b else -2,
                block_size=config.block_size,
            )

        if b_t is None:
            quantized_b_t = QuantizedTensor.quantize(
                quantized_b.dequantize(),
                quantized_b.real_dtype,
                config.granularity,
                axis=-2 if trans_b else -1,
                block_size=config.block_size,
            )
        else:
            assert isinstance(b_t, QuantizedTensor)
            quantized_b_t = b_t

        out = gemm_fp8_impl(
            quantized_a.qdata,
            quantized_a.scale_inv,
            trans_a,
            quantized_b.qdata,
            quantized_b.scale_inv,
            trans_b,
            out_dtype,
            False,
            granularity=config.granularity.value,
            default_backend=BackendType.TRITON.value,
        )

        ctx.save_for_backward(
            quantized_a_t.qdata,
            quantized_a_t.scale_inv,
            quantized_b_t.qdata,
            quantized_b_t.scale_inv,
        )
        ctx.trans_a = trans_a
        ctx.trans_b = trans_b
        ctx.out_dtype = out_dtype
        ctx.config = config

        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        if not grad_out.is_contiguous():
            grad_out = grad_out.contiguous()

        a_fp8_t, a_t_scale_inv, b_fp8_t, b_t_scale_inv = ctx.saved_tensors
        grad_out_dtype = _get_fp8_dtype(ctx.config.format, False)

        quantized_grad_out = QuantizedTensor.quantize(
            grad_out,
            grad_out_dtype,
            ctx.config.granularity,
            axis=-1,
            block_size=ctx.config.block_size,
        )

        # NT
        a_grad = gemm_fp8_impl(
            quantized_grad_out.qdata,
            quantized_grad_out.scale_inv,
            False,
            b_fp8_t,
            b_t_scale_inv,
            not ctx.trans_b,
            ctx.out_dtype,
            ctx.trans_a,
            granularity=ctx.config.granularity.value,
            default_backend=BackendType.TRITON.value,
        )

        quantized_grad_out_t = QuantizedTensor.quantize(
            grad_out, grad_out_dtype, ctx.config.granularity, axis=-2, block_size=ctx.config.block_size
        )

        # TN
        b_grad = gemm_fp8_impl(
            a_fp8_t,
            a_t_scale_inv,
            not ctx.trans_a,
            quantized_grad_out_t.qdata,
            quantized_grad_out_t.scale_inv,
            False,
            ctx.out_dtype,
            ctx.trans_b,
            granularity=ctx.config.granularity.value,
            default_backend=BackendType.TRITON.value,
        )

        return (
            a_grad,  # a
            b_grad,  # b
            None,  # a_t
            None,  # b_t
            None,  # trans_a
            None,  # trans_b
            None,  # out_dtype
            None,  # config
        )


class FP8GemmBlockFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        a: Union[torch.Tensor, QuantizedTensor],
        b: Union[torch.Tensor, QuantizedTensor],
        a_t: Optional[QuantizedTensor],
        b_t: Optional[QuantizedTensor],  # not used
        trans_a: bool,
        trans_b: bool,
        out_dtype: torch.dtype,
        config: Float8QuantConfig,
    ):
        assert trans_a == False, "trans_a has to be False"
        a_dtype = _get_fp8_dtype(config.format, True)
        b_dtype = _get_fp8_dtype(config.format, True)

        if isinstance(a, QuantizedTensor):
            check_quantized_tensor(a, config, axis=-1)
            a_row, a_row_scale = a.qdata, a.scale_inv
            if a_t is None:
                a_t = QuantizedTensor.quantize(
                    a.dequantize(),
                    a_dtype,
                    config.granularity,
                    axis=-2,
                    block_size=config.block_size,
                )

            a_col, a_col_scale = a_t.qdata, a_t.scale_inv
        else:
            (
                a_row,
                a_row_scale,
                a_col,
                a_col_scale,
            ) = quantize_fp8_with_trans(a, a_dtype, config.granularity, block_size=config.block_size)

        # --- B side: 2D-block weight, reused unchanged in fwd + bwd. ---
        b_scaling_recipe = ScalingRecipe(use_2d_block=True)
        if isinstance(b, QuantizedTensor):
            check_quantized_tensor(b, config, scaling_recipe=b_scaling_recipe)
            b_row, b_row_scale = b.qdata, b.scale_inv
        else:
            b_row, b_row_scale = quantize_fp8(
                b, b_dtype, config.granularity, block_size=config.block_size, scaling_recipe=b_scaling_recipe
            )
        b_col, b_col_scale = b_row, b_row_scale

        out = gemm_fp8_impl(
            a_row,
            a_row_scale,
            trans_a,
            b_row,
            b_row_scale,
            trans_b,
            out_dtype,
            False,
            granularity=config.granularity.value,
            default_backend=BackendType.TRITON.value,
        )
        ctx.save_for_backward(a_col, a_col_scale, b_col, b_col_scale)

        ctx.trans_a = trans_a
        ctx.trans_b = trans_b
        ctx.out_dtype = out_dtype
        ctx.config = config

        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        a_col, a_col_scale, b_col, b_col_scale = ctx.saved_tensors

        grad_out_dtype = _get_fp8_dtype(ctx.config.format, False)
        grad_out = grad_out.contiguous()

        # Quantize grad_out in both row-wise and column-wise directions:
        # - row-wise: for dgrad (grad_x)
        # - col-wise: for wgrad (grad_w)
        (
            g_row,
            g_row_scale,
            g_col,
            g_col_scale,
        ) = quantize_fp8_with_trans(
            grad_out, grad_out_dtype, ctx.config.granularity, block_size=ctx.config.block_size
        )

        grad_a = gemm_fp8_impl(
            g_row,
            g_row_scale,
            False,
            b_col,
            b_col_scale,
            not ctx.trans_b,
            ctx.out_dtype,
            False,
            granularity=ctx.config.granularity.value,
            default_backend=BackendType.TRITON.value,
        )

        grad_b = gemm_fp8_impl(
            a_col,
            a_col_scale,
            not ctx.trans_a,
            g_col,
            g_col_scale,
            False,
            ctx.out_dtype,
            ctx.trans_b,
            granularity=ctx.config.granularity.value,
            default_backend=BackendType.TRITON.value,
        )

        return (
            grad_a,  # a
            grad_b,  # b
            None,  # a_t
            None,  # b_t
            None,  # trans_a
            None,  # trans_b
            None,  # out_dtype
            None,  # config
        )


class FP8GemmMXFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        a: Union[torch.Tensor, QuantizedTensor],
        b: Union[torch.Tensor, QuantizedTensor],
        a_t: Optional[QuantizedTensor],
        b_t: Optional[QuantizedTensor],
        trans_a: bool,
        trans_b: bool,
        out_dtype: torch.dtype,
        config: Float8QuantConfig,
    ):
        supported_mxfp8_backend, reason = check_mxfp8_support()
        assert supported_mxfp8_backend, reason

        assert trans_a == False and trans_b == True, "trans_a has to be False and trans_b has to be True"

        # Scale preshuffle is NOT done here: the quant emits raw E8M0 [dim, K//32]
        # scales and each GEMM backend implicitly preshuffles right before its own
        # kernel (FlyDSL fuses it into the gemm launch / turbo 16x4 / hipBLASLt vendor).
        fp8_dtype = _get_fp8_dtype(config.format, True)
        granularity = config.granularity
        block_size = config.block_size

        a_scaling_recipe = ScalingRecipe()
        if isinstance(a, QuantizedTensor):
            check_quantized_tensor(a, config, axis=-1, scaling_recipe=a_scaling_recipe)
            a_row, a_row_scale = a.qdata, a.scale_inv
            if a_t is None:
                a_t = QuantizedTensor.quantize(
                    a.dequantize(),
                    a.real_dtype,
                    granularity,
                    axis=-2,
                    block_size=block_size,
                    scaling_recipe=a_scaling_recipe,
                )
            a_col, a_col_scale = a_t.qdata, a_t.scale_inv
        else:
            a_row, a_row_scale, a_col, a_col_scale = quantize_fp8_with_trans_flydsl(a, fp8_dtype)

        b_scaling_recipe = ScalingRecipe(use_2d_block=True)
        if isinstance(b, QuantizedTensor):
            check_quantized_tensor(b, config, axis=-1, scaling_recipe=b_scaling_recipe)
            b_row, b_row_scale = b.qdata, b.scale_inv
            if b_t is None:
                b_t = QuantizedTensor.quantize(
                    b.dequantize(),
                    b.real_dtype,
                    granularity,
                    axis=-2,
                    block_size=block_size,
                    scaling_recipe=b_scaling_recipe,
                )
            b_col, b_col_scale = b_t.qdata, b_t.scale_inv
        else:
            b_row, b_row_scale, b_col, b_col_scale = quantize_fp8_with_trans_flydsl(b, fp8_dtype)

        # NT layout
        out = gemm_fp8_impl(
            a_row,
            a_row_scale,
            False,
            b_row,
            b_row_scale,
            True,
            out_dtype,
            False,
            granularity=granularity.value,
            default_backend=BackendType.FLYDSL.value,
        )

        ctx.save_for_backward(a_col, a_col_scale, b_col, b_col_scale)
        ctx.out_dtype = out_dtype
        ctx.config = config

        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        a_col, a_col_scale, b_col, b_col_scale = ctx.saved_tensors

        grad_out_dtype = _get_fp8_dtype(ctx.config.format, False)
        # reshape (not view): the dual-cast quant kernel asserts a contiguous input;
        # a strided grad_out (e.g. a transpose/slice upstream) would trip it.
        grad_out = grad_out.reshape(grad_out.shape[0], -1).contiguous()

        # Dual-cast grad_out: one kernel emits both the row-wise (grad_a, NN->NT) and
        # col-wise (grad_b, TN->NT) directions -- no second single-direction pass.
        g_row, g_row_scale, g_col, g_col_scale = quantize_fp8_with_trans_flydsl(grad_out, grad_out_dtype)

        grad_a = gemm_fp8_impl(
            g_row,
            g_row_scale,
            False,
            b_col,
            b_col_scale,
            True,
            ctx.out_dtype,
            False,
            granularity=ctx.config.granularity.value,
            default_backend=BackendType.FLYDSL.value,
        )

        grad_b = gemm_fp8_impl(
            g_col,
            g_col_scale,
            False,
            a_col,
            a_col_scale,
            True,
            ctx.out_dtype,
            False,
            granularity=ctx.config.granularity.value,
            default_backend=BackendType.FLYDSL.value,
        )

        return (
            grad_a,  # a
            grad_b,  # b
            None,  # a_t
            None,  # b_t
            None,  # trans_a
            None,  # trans_b
            None,  # out_dtype
            None,  # config
        )


@torch._dynamo.disable(
    recursive=True,
    reason=(
        "FP8 GEMM constructs QuantizedTensor wrapper subclasses inside its "
        "autograd.Function.forward and reads their inner tensors (data / scale_inv). "
        "Dynamo cannot recover Python sources for those graph-internal inner tensors, "
    ),
)
def gemm_fp8(
    a: Union[torch.Tensor, QuantizedTensor, QuantizedTensorPair],
    b: Union[torch.Tensor, QuantizedTensor, QuantizedTensorPair],
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
    if config is None:
        config = Float8QuantConfig()

    if isinstance(a, QuantizedTensorPair):
        a_data, a_data_t = a.data, a.data_t
    else:
        a_data, a_data_t = a, None

    if isinstance(b, QuantizedTensorPair):
        b_data, b_data_t = b.data, b.data_t
    else:
        b_data, b_data_t = b, None

    assert a_data.ndim == 2, "Only 2D tensors are supported"
    assert b_data.ndim == 2, "Only 2D tensors are supported"

    if out_dtype is None:
        out_dtype = torch.promote_types(a_data.dtype, b_data.dtype)

    if config.granularity == ScalingGranularity.TENSORWISE:
        return FP8GemmTensorFunction.apply(
            a_data, b_data, a_data_t, b_data_t, trans_a, trans_b, out_dtype, config
        )
    elif config.granularity == ScalingGranularity.ROWWISE:
        return FP8GemmRowFunction.apply(
            a_data, b_data, a_data_t, b_data_t, trans_a, trans_b, out_dtype, config
        )
    elif config.granularity == ScalingGranularity.BLOCKWISE:
        return FP8GemmBlockFunction.apply(
            a_data, b_data, a_data_t, b_data_t, trans_a, trans_b, out_dtype, config
        )
    elif config.granularity == ScalingGranularity.MX_BLOCKWISE:
        return FP8GemmMXFunction.apply(
            a_data, b_data, a_data_t, b_data_t, trans_a, trans_b, out_dtype, config
        )
    else:
        raise ValueError(f"Unsupported FP8 ScalingGranularity: {config.granularity}")
