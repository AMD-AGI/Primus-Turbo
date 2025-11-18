###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from typing import Optional, Tuple

import torch


from primus_turbo.pytorch.core.low_precision import ScalingGranularity
from primus_turbo.pytorch.kernels.quantization.quantization_impl import (
    dequantize_fp8_rowwise_impl,
    dequantize_fp8_tensorwise_impl,
    dequantize_mxfp8_impl,
    quantize_fp8_rowwise_impl,
    quantize_fp8_tensorwise_impl,
    quantize_mxfp8_impl,
)

__all__ = ["quantize_fp8", "dequantize_fp8"]

MX_BLOCK_SIZE = 32


def quantize_fp8(
    x: torch.Tensor,
    out_dtype: torch.dtype,
    granularity: ScalingGranularity,
    *,
    block_size: Optional[int] = None,
    axis: Optional[int] = None,
    scale: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    FP8 Quantize
    """
    if granularity == ScalingGranularity.TENSORWISE:
        return quantize_fp8_tensorwise_impl(x, out_dtype, scale)
    elif granularity == ScalingGranularity.ROWWISE:
        assert axis is not None, "The axis must be specified for rowwise FP8 quantization"

        return quantize_fp8_rowwise_impl(x, out_dtype, axis, scale)
    elif granularity == ScalingGranularity.MX_BLOCKWISE:
        assert block_size == MX_BLOCK_SIZE, f"The block size must be {MX_BLOCK_SIZE} for MXFP8 quantization"
        assert scale is None, "The scale is not supported for MXFP8 quantization"

        return quantize_mxfp8_impl(x, out_dtype, block_size)
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
):
    """
    FP8 DeQuantize
    """
    if granularity == ScalingGranularity.TENSORWISE:
        return dequantize_fp8_tensorwise_impl(x, out_dtype, scale_inv)
    elif granularity == ScalingGranularity.ROWWISE:
        if axis is None:
            raise ValueError("axis must be specified for rowwise FP8 de-quantization")

        return dequantize_fp8_rowwise_impl(x, out_dtype, axis, scale_inv)
    elif granularity == ScalingGranularity.MX_BLOCKWISE:
        assert block_size == MX_BLOCK_SIZE, f"The block size must be {MX_BLOCK_SIZE} for MXFP8 quantization"

        return dequantize_mxfp8_impl(x, out_dtype, block_size, scale_inv)
    else:
        raise NotImplementedError(f"Unknown granularity {granularity}")


def quantize_fp4(
    x: torch.Tensor,
    out_dtype: torch.dtype,
    granularity: ScalingGranularity,
    *,
    axis: Optional[int] = None,
    scale: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    FP4 Quantize
    """
    if granularity == ScalingGranularity.MX_BLOCKWISE:
        raise NotImplementedError("MX_BLOCKWISE FP4 quantization is not supported yet")
    else:
        raise NotImplementedError(f"Unknown granularity {granularity}")


def dequantize_fp4(
    x: torch.Tensor,
    out_dtype: torch.dtype,
    granularity: ScalingGranularity,
    *,
    axis: Optional[int] = None,
    scale_inv: torch.Tensor,
):
    """
    FP4 DeQuantize
    """
    if granularity == ScalingGranularity.MX_BLOCKWISE:
        raise NotImplementedError("MX_BLOCKWISE FP4 de-quantization is not supported yet")
    else:
        raise NotImplementedError(f"Unknown granularity {granularity}")
