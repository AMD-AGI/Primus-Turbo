###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from typing import Optional

import torch

from primus_turbo.pytorch.core.low_precision import ScalingGranularity


def quantize_fp8_ref(
    x: torch.Tensor,
    out_dtype: torch.dtype,
    granularity: ScalingGranularity,
    axis: Optional[int] = None,
):
    EPS = 1e-12
    if granularity == ScalingGranularity.TENSORWISE:
        return _quantize_fp8_tensorwise_ref(x, out_dtype, EPS)
    elif granularity == ScalingGranularity.ROWWISE:
        if axis is None:
            raise ValueError("axis must be specified for rowwise FP8 quantization")
        return _quantize_fp8_rowwise_ref(x, out_dtype, axis, EPS)
    else:
        raise NotImplementedError(f"Unknown granularity {granularity}")


def _quantize_fp8_tensorwise_ref(x, dtype, EPS=1e-12):
    fp8_max = torch.finfo(dtype).max
    # Compute Scale & Scale-Inv. A 3D [G, M, N] input is quantized per leading
    # group, so its scales are [G, 1]; any other rank shares one scalar scale.
    is_grouped = x.dim() == 3
    if is_grouped:
        x_amax = x.abs().flatten(1).amax(dim=1).to(torch.float32).unsqueeze(-1)
    else:
        x_amax = x.abs().amax().to(torch.float32)
    scale = fp8_max / torch.clamp(x_amax, min=EPS)
    scale_inv = 1.0 / scale
    # Quantize. Scale in fp32: a 0-dim fp32 scale does not promote a half-precision
    # x, so scaling in place would overflow to inf whenever scale > 65504.
    x_scaled = x.to(torch.float32) * (scale.unsqueeze(-1) if is_grouped else scale)
    x_clamped = torch.clamp(x_scaled, -fp8_max, fp8_max)
    return x_clamped.to(dtype), scale.to(torch.float32), scale_inv.to(torch.float32)


def _quantize_fp8_rowwise_ref(x, dtype, axis, EPS=1e-12):
    axis = axis if axis >= 0 else x.dim() + axis
    if axis < 0 or axis >= x.dim():
        raise ValueError(f"axis={axis} is out of bounds for tensor of dimension {x.dim()}")
    fp8_max = torch.finfo(dtype).max
    # Compute Scale
    x_max = torch.amax(x.abs(), dim=axis, keepdim=True).to(torch.float32)

    scale = fp8_max / torch.clamp(x_max, min=EPS)
    scale_inv = 1.0 / scale
    # Quantize
    x_scaled = x * scale
    x_clamped = torch.clamp(x_scaled, -fp8_max, fp8_max)
    return x_clamped.to(dtype), scale.to(torch.float32), scale_inv.to(torch.float32)


def dequantize_fp8_ref(
    x: torch.Tensor,
    out_dtype: torch.dtype,
    granularity: ScalingGranularity,
    *,
    axis: Optional[int] = None,
    scale_inv: torch.Tensor,
):
    scale_inv = scale_inv.to(torch.float32)
    if granularity == ScalingGranularity.TENSORWISE and x.dim() == 3:
        # Per-group tensorwise scales are [G, 1]; broadcast them over [G, M, N].
        scale_inv = scale_inv.unsqueeze(-1)
    y = x.to(torch.float32) * scale_inv
    return y.to(out_dtype)
