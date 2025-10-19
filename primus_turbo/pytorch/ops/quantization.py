###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from typing import Optional, Tuple

import torch

from primus_turbo.pytorch.core.float8 import ScalingGranularity
from primus_turbo.pytorch.kernels.quantization_impl import (
    quantize_fp8_rowwise_impl,
    quantize_fp8_tensorwise_impl,
)

__all__ = ["quantize_fp8"]


def quantize_fp8(
    x: torch.Tensor,
    out_dtype: torch.dtype,
    granularity: ScalingGranularity,
    *,
    axis: Optional[int] = None,
    scale: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    FP8 Quantize
    """
    if granularity == ScalingGranularity.TENSORWISE:
        return quantize_fp8_tensorwise_impl(x, out_dtype, scale)
    elif granularity == ScalingGranularity.ROWWISE:
        if axis is None:
            raise ValueError("axis must be specified for rowwise FP8 quantization")
        return quantize_fp8_rowwise_impl(x, out_dtype, axis, scale)
    else:
        raise NotImplementedError(f"Unknown granularity {granularity}")


def dequantize_fp8(
    x: torch.Tensor,
    out_dtype: torch.dtype,
    granularity: ScalingGranularity,
    *,
    axis: Optional[int] = None,
    scale_inv: torch.Tensor,
):
    pass


"""
TODO:
quantize_mxfp8
quantize_mxfp4
"""
