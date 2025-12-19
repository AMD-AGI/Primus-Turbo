###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from typing import Optional, Tuple

import jax.numpy as jnp

from primus_turbo.jax.core.low_precision import ScalingGranularity


def quantize_fp8_ref(
    x: jnp.ndarray,
    out_dtype: jnp.dtype,
    granularity: ScalingGranularity,
    axis: Optional[int] = None,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
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
    fp8_max = float(jnp.finfo(dtype).max)
    x_amax = jnp.max(jnp.abs(x)).astype(jnp.float32)
    scale = fp8_max / jnp.maximum(x_amax, EPS)
    scale_inv = 1.0 / scale
    x_scaled = x * scale
    x_clamped = jnp.clip(x_scaled, -fp8_max, fp8_max)
    return x_clamped.astype(dtype), scale.astype(jnp.float32), scale_inv.astype(jnp.float32)


def _quantize_fp8_rowwise_ref(x, dtype, axis, EPS=1e-12):
    axis = axis if axis >= 0 else x.ndim + axis
    if axis < 0 or axis >= x.ndim:
        raise ValueError(f"axis={axis} is out of bounds for tensor of dimension {x.ndim}")
    fp8_max = float(jnp.finfo(dtype).max)
    x_max = jnp.max(jnp.abs(x), axis=axis, keepdims=True).astype(jnp.float32)
    scale = fp8_max / jnp.maximum(x_max, EPS)
    scale_inv = 1.0 / scale
    x_scaled = x * scale
    x_clamped = jnp.clip(x_scaled, -fp8_max, fp8_max)
    return x_clamped.astype(dtype), scale.astype(jnp.float32), scale_inv.astype(jnp.float32)


def dequantize_fp8_ref(
    x: jnp.ndarray,
    out_dtype: jnp.dtype,
    granularity: ScalingGranularity,
    *,
    axis: Optional[int] = None,
    scale_inv: jnp.ndarray,
) -> jnp.ndarray:
    y = x.astype(jnp.float32) * scale_inv.astype(jnp.float32)
    return y.astype(out_dtype)
