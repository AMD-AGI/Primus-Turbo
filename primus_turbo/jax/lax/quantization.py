###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from typing import Optional, Tuple

import jax
import jax.numpy as jnp

from primus_turbo.jax.core.low_precision import ScalingGranularity

from ..primitive.quantization import (
    dequantize_fp8_tensorwise_p,
    quantize_fp8_rowwise_p,
    quantize_fp8_tensorwise_p,
)

__all__ = ["quantize_fp8", "dequantize_fp8"]


def quantize_fp8(
    x: jax.Array,
    out_dtype: jnp.dtype,
    granularity: ScalingGranularity,
    *,
    axis: Optional[int] = None,
    scale: Optional[jax.Array] = None,
) -> Tuple[jax.Array, jax.Array]:
    """FP8 Quantize. Returns: (x_q, scale_inv)"""
    scale_opt = jnp.empty((0,), dtype=jnp.float32) if scale is None else scale

    if granularity == ScalingGranularity.TENSORWISE:
        x_q, scale_inv, _ = quantize_fp8_tensorwise_p.bind(x, scale_opt, out_dtype=out_dtype)
        return x_q, scale_inv

    elif granularity == ScalingGranularity.ROWWISE:
        if axis is None:
            raise ValueError("axis must be specified for rowwise FP8 quantization")
        x_q, scale_inv, _ = quantize_fp8_rowwise_p.bind(x, scale_opt, out_dtype=out_dtype, axis=axis)
        return x_q, scale_inv

    else:
        raise NotImplementedError(f"Unknown granularity {granularity}")


def dequantize_fp8(
    x: jax.Array,
    out_dtype: jnp.dtype,
    granularity: ScalingGranularity,
    *,
    axis: Optional[int] = None,
    scale_inv: jax.Array,
) -> jax.Array:
    """FP8 DeQuantize. Returns: x_dq"""
    if granularity == ScalingGranularity.TENSORWISE:
        return dequantize_fp8_tensorwise_p.bind(x, scale_inv, out_dtype=out_dtype)
    elif granularity == ScalingGranularity.ROWWISE:
        if axis is None:
            raise ValueError("axis must be specified for rowwise FP8 de-quantization")
        raise NotImplementedError("Rowwise dequantization not implemented")
    else:
        raise NotImplementedError(f"Unknown granularity {granularity}")
