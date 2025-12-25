###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from typing import Optional, Tuple

import torch

from primus_turbo.pytorch.core.low_precision import MXScalingRecipe, ScalingGranularity
from primus_turbo.pytorch.kernels.quantization.quantization_impl import (
    dequantize_fp8_rowwise_impl,
    dequantize_fp8_tensorwise_impl,
    dequantize_mxfp4_impl,
    dequantize_mxfp8_impl,
    quantize_fp8_rowwise_impl,
    quantize_fp8_tensorwise_impl,
    quantize_mxfp4_impl,
    quantize_mxfp8_impl,
)

__all__ = ["quantize_fp8", "dequantize_fp8", "quantize_fp4", "dequantize_fp4"]

MX_BLOCK_SIZE = 32


def quantize_fp8(
    x: torch.Tensor,
    out_dtype: torch.dtype,
    granularity: ScalingGranularity,
    *,
    block_size: Optional[int] = None,
    axis: Optional[int] = None,
    scale: Optional[torch.Tensor] = None,
    padding_align_size: Optional[int] = None,
    scaling_recipe: Optional[MXScalingRecipe] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    FP8 Quantize

    NOTE:
        For ROWWISE quantization:
            1. The axis must be specified.

        For MXFP8 quantization:
            1. The x must be 2D tensor.
            2. The axis means direction of quantization. The 0 means along column direction and 1 means along row direction.
            3. The block size must be 32.
            4. The out tensor will be padded in specified axis if padding_align_size is not `None`.
    """
    if granularity == ScalingGranularity.TENSORWISE:
        return quantize_fp8_tensorwise_impl(x, out_dtype, scale)
    elif granularity == ScalingGranularity.ROWWISE:
        assert axis is not None, "The axis must be specified for rowwise FP8 quantization"

        return quantize_fp8_rowwise_impl(x, out_dtype, axis, scale)
    elif granularity == ScalingGranularity.MX_BLOCKWISE:
        assert block_size == MX_BLOCK_SIZE, f"The block size must be {MX_BLOCK_SIZE} for MXFP8 quantization"
        assert scale is None, "The scale is not supported for MXFP8 quantization"

        return quantize_mxfp8_impl(x, out_dtype, axis, block_size, padding_align_size, scaling_recipe)
    else:
        raise NotImplementedError(f"Unknown granularity {granularity}")


def dequantize_fp8(
    x: torch.Tensor,
    out_dtype: torch.dtype,
    granularity: ScalingGranularity,
    *,
    block_size: Optional[int] = None,
    axis: Optional[int] = None,
    scale_inv: torch.Tensor,
    scaling_recipe: Optional[MXScalingRecipe] = None,
):
    """
    FP8 DeQuantize

    NOTE:
        For ROWWISE quantization:
            1. The axis must be specified.

        For MXFP8 quantization:
            1. The x must be 2D tensor.
            2. The axis means direction of de-quantization. The 0 means along column direction and 1 means along row direction.
            3. The block size must be 32.
    """
    if granularity == ScalingGranularity.TENSORWISE:
        return dequantize_fp8_tensorwise_impl(x, out_dtype, scale_inv)
    elif granularity == ScalingGranularity.ROWWISE:
        if axis is None:
            raise ValueError("axis must be specified for rowwise FP8 de-quantization")

        return dequantize_fp8_rowwise_impl(x, out_dtype, axis, scale_inv)
    elif granularity == ScalingGranularity.MX_BLOCKWISE:
        assert block_size == MX_BLOCK_SIZE, f"The block size must be {MX_BLOCK_SIZE} for MXFP8 quantization"

        return dequantize_mxfp8_impl(x, out_dtype, axis, block_size, scale_inv)
    else:
        raise NotImplementedError(f"Unknown granularity {granularity}")


def quantize_fp4(
    x: torch.Tensor,
    out_dtype: torch.dtype,
    granularity: ScalingGranularity,
    *,
    block_size: Optional[int] = None,
    axis: Optional[int] = None,
    scale: Optional[torch.Tensor] = None,
    padding_align_size: Optional[int] = None,
    scaling_recipe: Optional[MXScalingRecipe] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    FP4 Quantize

    NOTE:
        For MXFP4 quantization:
            1. The x must be 2D tensor.
            2. The axis means direction of quantization. The 0 means along column direction and 1 means along row direction.
            3. The block size must be 32.
            4. The out tensor will be padded in specified axis if padding_align_size is not `None`.
            5. The scaling recipe is used to control the quantization behavior.
    """
    if granularity == ScalingGranularity.MX_BLOCKWISE:
        assert scale is None, "The scale is not supported for MXFP4 quantization"
        assert block_size == MX_BLOCK_SIZE, f"The block size must be {MX_BLOCK_SIZE} for MXFP8 quantization"

        return quantize_mxfp4_impl(x, out_dtype, axis, block_size, padding_align_size, scaling_recipe)
    else:
        raise NotImplementedError(f"Unknown granularity {granularity}")


def dequantize_fp4(
    x: torch.Tensor,
    out_dtype: torch.dtype,
    granularity: ScalingGranularity,
    *,
    block_size: Optional[int] = None,
    axis: Optional[int] = None,
    scale_inv: torch.Tensor,
    scaling_recipe: Optional[MXScalingRecipe] = None,
) -> torch.Tensor:
    """
    FP4 DeQuantize

    NOTE:
        For MXFP4 quantization:
            1. The x must be 2D tensor.
            2. The axis means direction of de-quantization. The 0 means along column direction and 1 means along row direction.
            3. The block size must be 32.
            4. The scaling recipe is used to control the de-quantization behavior.
    """
    if granularity == ScalingGranularity.MX_BLOCKWISE:
        assert block_size == MX_BLOCK_SIZE, f"The block size must be {MX_BLOCK_SIZE} for MXFP8 quantization"

        return dequantize_mxfp4_impl(x, out_dtype, axis, block_size, scale_inv)
    else:
        raise NotImplementedError(f"Unknown granularity {granularity}")
