###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from typing import Optional, Union

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
    QuantizedTensorPair,
    check_quantized_tensor,
)
from primus_turbo.pytorch.kernels.gemm.gemm_fp4_impl import gemm_fp4_impl

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
        a_t: Optional[QuantizedTensor],
        b_t: Optional[QuantizedTensor],
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
        )
        if isinstance(a, QuantizedTensor):
            check_quantized_tensor(a, config, scaling_recipe=a_scaling_recipe)
            a_fp4 = a
        else:
            a_dtype = FP4GemmMXFunction.get_fp4_dtype(config.format)
            a_fp4 = QuantizedTensor.quantize(
                a,
                a_dtype,
                config.granularity,
                block_size=config.block_size,
                scaling_recipe=a_scaling_recipe,
                axis=1,
            )

        b_scaling_recipe = ScalingRecipe(
            use_2d_block=True,
            use_sr=False,
            use_rht=False,
        )
        if isinstance(b, QuantizedTensor):
            check_quantized_tensor(b, config, scaling_recipe=b_scaling_recipe)
            b_fp4 = b
        else:
            b_dtype = FP4GemmMXFunction.get_fp4_dtype(config.format)
            b_fp4 = QuantizedTensor.quantize(
                b,
                b_dtype,
                config.granularity,
                block_size=config.block_size,
                scaling_recipe=b_scaling_recipe,
                axis=1,
            )

        # NT layout
        out = gemm_fp4_impl(
            a_fp4.qdata,
            a_fp4.scale_inv,
            False,
            b_fp4.qdata,
            b_fp4.scale_inv,
            True,
            out_dtype,
            False,
            granularity=config.granularity.value,
            default_backend=BackendType.HIPBLASLT.value,
        )

        # Backward needs a col-wise (axis=0) version of A/B with an RHT recipe.
        # If the caller pre-quantized this and passed it via ``a_t`` / ``b_t``,
        # reuse it directly; otherwise derive it from the forward fp4 tensor.
        if a_t is not None:
            quantized_a_t = a_t
        else:
            a_t_scaling_recipe = ScalingRecipe(
                use_2d_block=False,
                use_sr=False,
                use_rht=True,
            )
            quantized_a_t = QuantizedTensor.quantize(
                a_fp4.dequantize(),
                a_fp4.real_dtype,
                config.granularity,
                block_size=config.block_size,
                axis=0,
                scaling_recipe=a_t_scaling_recipe,
            )

        if b_t is not None:
            quantized_b_t = b_t
        else:
            b_t_scaling_recipe = ScalingRecipe(
                use_2d_block=True,
                use_sr=False,
                use_rht=True,
            )
            quantized_b_t = QuantizedTensor.quantize(
                b_fp4.dequantize(),
                b_fp4.real_dtype,
                config.granularity,
                block_size=config.block_size,
                axis=0,
                scaling_recipe=b_t_scaling_recipe,
            )
        ctx.save_for_backward(
            quantized_a_t.qdata, quantized_a_t.scale_inv, quantized_b_t.qdata, quantized_b_t.scale_inv
        )

        ctx.trans_a = trans_a
        ctx.trans_b = trans_b
        ctx.out_dtype = out_dtype
        ctx.config = config
        ctx.a_fp4_dtype = a_fp4.real_dtype
        ctx.b_fp4_dtype = b_fp4.real_dtype

        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        a_fp4_t, a_t_scale_inv, b_fp4_t, b_t_scale_inv = ctx.saved_tensors
        grad_out_dtype = FP4GemmMXFunction.get_fp4_dtype(
            ctx.config.format,
        )

        grad_out = grad_out.view(grad_out.shape[0], -1)

        grad_out_scaling_recipe = ScalingRecipe(
            use_2d_block=False,
            use_sr=ctx.config.use_gradient_sr,
            use_rht=True,
        )

        quantized_grad_out = QuantizedTensor.quantize(
            grad_out,
            grad_out_dtype,
            ctx.config.granularity,
            axis=1,
            block_size=ctx.config.block_size,
            scaling_recipe=grad_out_scaling_recipe,
        )

        # NOTE: convert NN layout to NT layout because MXFP4 only supports NT layout on hipblaslt.
        grad_a = gemm_fp4_impl(
            quantized_grad_out.qdata,
            quantized_grad_out.scale_inv,
            False,
            b_fp4_t,
            b_t_scale_inv,
            True,
            ctx.out_dtype,
            False,
            granularity=ctx.config.granularity.value,
            default_backend=BackendType.HIPBLASLT.value,
        )

        grad_out_t_scaling_recipe = ScalingRecipe(
            use_2d_block=False,
            use_sr=ctx.config.use_gradient_sr,
            use_rht=True,
        )
        quantized_grad_out_t = QuantizedTensor.quantize(
            grad_out,
            grad_out_dtype,
            ctx.config.granularity,
            block_size=ctx.config.block_size,
            axis=0,
            scaling_recipe=grad_out_t_scaling_recipe,
        )

        # NOTE: convert TN layout to NT layout because MXFP4 only supports NT layout on hipblaslt.
        grad_b = gemm_fp4_impl(
            quantized_grad_out_t.qdata,
            quantized_grad_out_t.scale_inv,
            False,
            a_fp4_t,
            a_t_scale_inv,
            True,
            ctx.out_dtype,
            False,
            granularity=ctx.config.granularity.value,
            default_backend=BackendType.HIPBLASLT.value,
        )

        # Grads correspond to forward args:
        #   (a, b, a_t, b_t, trans_a, trans_b, out_dtype, config)
        return grad_a, grad_b, None, None, None, None, None, None


@torch._dynamo.disable(
    recursive=True,
    reason=(
        "FP4 GEMM constructs QuantizedTensor wrapper subclasses inside its "
        "autograd.Function.forward and reads their inner tensors (data / scale_inv). "
        "Dynamo cannot recover Python sources for those graph-internal inner tensors, "
    ),
)
def gemm_fp4(
    a: Union[torch.Tensor, QuantizedTensor, QuantizedTensorPair],
    b: Union[torch.Tensor, QuantizedTensor, QuantizedTensorPair],
    trans_a: bool = False,
    trans_b: bool = False,
    out_dtype: Union[torch.dtype, None] = None,
    config: Union[Float4QuantConfig, None] = None,
) -> torch.Tensor:
    """General matrix multiplication (GEMM) with FP4 quantization, supporting autograd.

    Automatically quantizes inputs to FP4 format during forward and backward passes
    to accelerate training and inference. When ``a`` or ``b`` is already a
    :class:`QuantizedTensor`, its quantized data / scale is reused directly,
    skipping the forward-direction quantization. If a :class:`QuantizedTensorPair`
    wrapper is passed instead, the optional ``data_t`` field is also forwarded
    and reused as the col-wise / RHT transpose cache for backward.

    Args:
        a: Input matrix a with shape (M, K), must be 2D tensor. The A matrix should be activaton.
            Can also be a pre-quantized :class:`QuantizedTensor` (forward only)
            or a :class:`QuantizedTensorPair` carrying both ``data`` and the
            backward-direction ``data_t``.
        b: Input matrix b with shape (K, N) or (N, K), must be 2D tensor. The B matrix should be weight.
            Same pre-quantized variants as ``a`` are accepted.
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
    if config is None:
        config = Float4QuantConfig()

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

    if config.granularity == ScalingGranularity.MX_BLOCKWISE:
        return FP4GemmMXFunction.apply(
            a_data, b_data, a_data_t, b_data_t, trans_a, trans_b, out_dtype, config
        )
    else:
        raise ValueError(f"Unsupported FP4 ScalingGranularity: {config.granularity}")
