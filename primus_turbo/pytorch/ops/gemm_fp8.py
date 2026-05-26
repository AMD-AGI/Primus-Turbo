###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from typing import Optional, Union

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
    QuantizedTensorPair,
    check_quantized_tensor,
)
from primus_turbo.pytorch.kernels.gemm.gemm_fp8_impl import gemm_fp8_impl
from primus_turbo.pytorch.kernels.quantization.quantization_impl import (
    quant_fp8_blockwise_dual_impl,
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
        ctx.save_for_backward(
            quantized_a.qdata, quantized_a.scale_inv, quantized_b.qdata, quantized_b.scale_inv
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
        a_fp8_data, a_scale_inv, b_fp8_data, b_scale_inv = ctx.saved_tensors
        grad_out_dtype = _get_fp8_dtype(ctx.config.format, False)

        quantized_grad_out = QuantizedTensor.quantize(
            grad_out,
            grad_out_dtype,
            ctx.config.granularity,
            axis=-1,
        )

        a_grad = gemm_fp8_impl(
            quantized_grad_out.qdata,
            quantized_grad_out.scale_inv,
            False,
            b_fp8_data,
            b_scale_inv,
            not ctx.trans_b,
            ctx.out_dtype,
            ctx.trans_a,
            granularity=ctx.config.granularity.value,
            default_backend=BackendType.HIPBLASLT.value,
        )

        b_grad = gemm_fp8_impl(
            a_fp8_data,
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

        return (a_grad, b_grad, None, None, None, None)


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
            # B's row-wise axis is (-1 if trans_b else -2); the col-wise / trans
            # cache used by backward is the other axis.
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
            default_backend=BackendType.HIPBLASLT.value,
        )

        # a_fp8.qdata = axis=-1 (row-wise), a_fp8.t() = axis=-2 (col-wise / transposed)
        ctx.save_for_backward(
            quantized_a_t.qdata, quantized_a_t.scale_inv, quantized_b_t.qdata, quantized_b_t.scale_inv
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

        # Quantize grad_out row-wise (axis=-1), then transpose to derive col-wise (axis=-2) version.
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
            default_backend=BackendType.HIPBLASLT.value,
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
            default_backend=BackendType.HIPBLASLT.value,
        )

        # Grads correspond to forward args:
        #   (a, b, a_t, b_t, trans_a, trans_b, out_dtype, config)
        return (a_grad, b_grad, None, None, None, None, None, None)


# TODO(ruibin): Add support for quantized tensor
class FP8GemmBlockFunction(torch.autograd.Function):

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
        from primus_turbo.pytorch.ops.quantization import (
            quant_fp8_blockwise_for_weight_impl,
            quant_fp8_blockwise_impl,
        )

        assert isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor), "a and b must be torch.Tensor"
        assert trans_a == False, "trans_a has to be False"
        a_dtype = _get_fp8_dtype(config.format, True)
        b_dtype = _get_fp8_dtype(config.format, True)
        a_dtype_bwd = _get_fp8_dtype(config.format, False)

        # When forward and backward share the same FP8 dtype (i.e. non-HYBRID
        # formats), fuse forward row-quant with column-quant of the same
        # activation in a single dual kernel pass and save the column result
        # into ctx so the backward can skip its own column-quant launch.
        # This is a kernel fusion (single tensor `a` is read once instead of
        # twice) and does NOT depend on tensor identity across iterations.
        fuse_a_dual = a_dtype == a_dtype_bwd and a.is_contiguous()

        if fuse_a_dual:
            (
                a_fp8_row,
                a_scale_inv_row,
                a_fp8_col,
                a_scale_inv_col,
            ) = quant_fp8_blockwise_dual_impl(a, a_dtype, config.block_size)
        else:
            a_fp8_row, a_scale_inv_row = quant_fp8_blockwise_impl(
                a, a_dtype, axis=1, block_size=config.block_size
            )
            a_fp8_col = None
            a_scale_inv_col = None

        b_fp8, b_scale_inv = quant_fp8_blockwise_for_weight_impl(b, b_dtype, block_size=config.block_size)

        out = gemm_fp8_impl(
            a_fp8_row,
            a_scale_inv_row,
            trans_a,
            b_fp8,
            b_scale_inv,
            trans_b,
            out_dtype,
            False,
            granularity=config.granularity.value,
            default_backend=BackendType.CK.value,
        )
        if fuse_a_dual:
            ctx.save_for_backward(b_fp8, b_scale_inv, a_fp8_col, a_scale_inv_col)
            ctx.has_prequantized_a_col = True
        else:
            ctx.save_for_backward(a, b_fp8, b_scale_inv)
            ctx.has_prequantized_a_col = False
        ctx.trans_a = trans_a
        ctx.trans_b = trans_b
        ctx.out_dtype = out_dtype
        ctx.config = config
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        from primus_turbo.pytorch.ops.quantization import quant_fp8_blockwise_impl

        if not grad_out.is_contiguous():
            grad_out = grad_out.contiguous()

        if ctx.has_prequantized_a_col:
            b_fp8, b_scale_inv, a_fp8_col, a_scale_inv_col = ctx.saved_tensors
        else:
            a, b_fp8, b_scale_inv = ctx.saved_tensors
            a_dtype = _get_fp8_dtype(ctx.config.format, False)
            a_fp8_col, a_scale_inv_col = quant_fp8_blockwise_impl(
                a, a_dtype, axis=0, block_size=ctx.config.block_size
            )
        grad_out_dtype = _get_fp8_dtype(ctx.config.format, False)

        # Quantize grad_out in both row-wise and column-wise directions:
        # - row-wise: for dgrad (grad_x)
        # - col-wise: for wgrad (grad_w)
        (
            grad_out_fp8_row,
            grad_out_scale_inv_row,
            grad_out_fp8_col,
            grad_out_scale_inv_col,
        ) = quant_fp8_blockwise_dual_impl(grad_out, grad_out_dtype, ctx.config.block_size)

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

        a_scaling_recipe = ScalingRecipe()
        if isinstance(a, QuantizedTensor):
            quantized_a = a
            check_quantized_tensor(quantized_a, config, axis=-1, scaling_recipe=a_scaling_recipe)
        else:
            a_dtype = _get_fp8_dtype(config.format, True)
            quantized_a = QuantizedTensor.quantize(
                a,
                a_dtype,
                config.granularity,
                axis=-1,
                block_size=config.block_size,
                scaling_recipe=a_scaling_recipe,
            )

        if a_t is None:
            # MX_BLOCKWISE requires a scaling_recipe; reuse the forward recipe
            # for A's col-wise direction (same recipe as forward).
            quantized_a_t = QuantizedTensor.quantize(
                quantized_a.dequantize(),
                quantized_a.real_dtype,
                config.granularity,
                axis=-2,
                block_size=config.block_size,
                scaling_recipe=a_scaling_recipe,
            )
        else:
            assert isinstance(a_t, QuantizedTensor)
            quantized_a_t = a_t

        b_scaling_recipe = ScalingRecipe(use_2d_block=True)
        if isinstance(b, QuantizedTensor):
            quantized_b = b
            check_quantized_tensor(quantized_b, config, axis=-1, scaling_recipe=b_scaling_recipe)
        else:
            b_dtype = _get_fp8_dtype(config.format, True)
            quantized_b = QuantizedTensor.quantize(
                b,
                b_dtype,
                config.granularity,
                axis=-1,
                block_size=config.block_size,
                scaling_recipe=b_scaling_recipe,
            )

        if b_t is None:
            quantized_b_t = QuantizedTensor.quantize(
                quantized_b.dequantize(),
                quantized_b.real_dtype,
                config.granularity,
                axis=-2,
                block_size=config.block_size,
                scaling_recipe=b_scaling_recipe,
            )
        else:
            assert isinstance(b_t, QuantizedTensor)
            quantized_b_t = b_t

        # NT layout
        out = gemm_fp8_impl(
            quantized_a.qdata,
            quantized_a.scale_inv,
            False,
            quantized_b.qdata,
            quantized_b.scale_inv,
            True,
            out_dtype,
            False,
            granularity=config.granularity.value,
            default_backend=BackendType.TURBO.value,
        )

        ctx.save_for_backward(
            quantized_a_t.qdata, quantized_a_t.scale_inv, quantized_b_t.qdata, quantized_b_t.scale_inv
        )

        ctx.trans_a = trans_a
        ctx.trans_b = trans_b
        ctx.out_dtype = out_dtype
        ctx.config = config

        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        a_fp8_t, a_t_scale_inv, b_fp8_t, b_t_scale_inv = ctx.saved_tensors

        grad_out_dtype = _get_fp8_dtype(ctx.config.format, False)
        grad_out = grad_out.view(grad_out.shape[0], -1)

        grad_out_scaling_recipe = ScalingRecipe()
        quantized_grad_out = QuantizedTensor.quantize(
            grad_out,
            grad_out_dtype,
            ctx.config.granularity,
            block_size=ctx.config.block_size,
            axis=-1,
            scaling_recipe=grad_out_scaling_recipe,
        )

        # NOTE: convert NN layout to NT layout because MXFP8 only supports NT layout.
        grad_a = gemm_fp8_impl(
            quantized_grad_out.qdata,
            quantized_grad_out.scale_inv,
            False,
            b_fp8_t,
            b_t_scale_inv,
            True,
            ctx.out_dtype,
            False,
            granularity=ctx.config.granularity.value,
            default_backend=BackendType.TURBO.value,
        )

        grad_out_t_scaling_recipe = ScalingRecipe()
        quantized_grad_out_t = QuantizedTensor.quantize(
            grad_out,
            grad_out_dtype,
            ctx.config.granularity,
            block_size=ctx.config.block_size,
            axis=-2,
            scaling_recipe=grad_out_t_scaling_recipe,
        )

        # NOTE: convert TN layout to NT layout because MXFP8 only supports NT layout.
        grad_b = gemm_fp8_impl(
            quantized_grad_out_t.qdata,
            quantized_grad_out_t.scale_inv,
            False,
            a_fp8_t,
            a_t_scale_inv,
            True,
            ctx.out_dtype,
            False,
            granularity=ctx.config.granularity.value,
            default_backend=BackendType.TURBO.value,
        )

        # Grads correspond to forward args:
        #   (a, b, a_t, b_t, trans_a, trans_b, out_dtype, config)
        return grad_a, grad_b, None, None, None, None, None, None


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
        return FP8GemmTensorFunction.apply(a_data, b_data, trans_a, trans_b, out_dtype, config)
    elif config.granularity == ScalingGranularity.ROWWISE:
        return FP8GemmRowFunction.apply(
            a_data, b_data, a_data_t, b_data_t, trans_a, trans_b, out_dtype, config
        )
    elif config.granularity == ScalingGranularity.BLOCKWISE:
        # BLOCKWISE does not yet support pre-quantized inputs; preserve the
        # existing assertion behaviour in ``FP8GemmBlockFunction.forward``.
        return FP8GemmBlockFunction.apply(a, b, trans_a, trans_b, out_dtype, config)
    elif config.granularity == ScalingGranularity.MX_BLOCKWISE:
        return FP8GemmMXFunction.apply(
            a_data, b_data, a_data_t, b_data_t, trans_a, trans_b, out_dtype, config
        )
    else:
        raise ValueError(f"Unsupported FP8 ScalingGranularity: {config.granularity}")
