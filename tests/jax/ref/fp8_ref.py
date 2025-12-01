###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from __future__ import annotations

from typing import Tuple

import jax.numpy as jnp

from primus_turbo.jax.core.low_precision import (
    Float8QuantConfig,
    Format,
    ScalingGranularity,
    float8_e4m3,
    float8_e5m2,
)
from tests.jax.ref.gemm_ref import grouped_gemm_ref

_FP8_FORMAT_INFO = {
    Format.E4M3: (float8_e4m3, jnp.float32(448.0)),
    Format.E5M2: (float8_e5m2, jnp.float32(57344.0)),
}


def _quantize_tensorwise(x: jnp.ndarray, fp8_dtype, fp8_max: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    x32 = x.astype(jnp.float32)
    amax = jnp.max(jnp.abs(x32))
    scale = jnp.where(amax > 0, fp8_max / amax, 0.0)
    scale_inv = jnp.where(scale > 0, 1.0 / scale, 0.0)
    x_scaled = jnp.clip(x32 * scale, -fp8_max, fp8_max)
    x_fp8 = x_scaled.astype(fp8_dtype)
    return x_fp8, scale_inv.astype(jnp.float32)


def _quantize_rowwise(
    x: jnp.ndarray, fp8_dtype, fp8_max: jnp.ndarray, axis: int
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    axis = axis if axis is not None else -1
    x32 = x.astype(jnp.float32)
    amax = jnp.max(jnp.abs(x32), axis=axis, keepdims=True)
    scale = jnp.where(amax > 0, fp8_max / amax, 0.0)
    scale_inv = jnp.where(scale > 0, 1.0 / scale, 0.0)
    x_scaled = jnp.clip(x32 * scale, -fp8_max, fp8_max)
    x_fp8 = x_scaled.astype(fp8_dtype)
    return x_fp8, scale_inv.astype(jnp.float32)


def _dequantize(fp8_arr: jnp.ndarray, scale_inv: jnp.ndarray) -> jnp.ndarray:
    return fp8_arr.astype(jnp.float32) * scale_inv


def grouped_gemm_fp8_reference(
    a: jnp.ndarray,
    b: jnp.ndarray,
    group_lens: jnp.ndarray,
    *,
    trans_b: bool,
    config: Float8QuantConfig,
) -> jnp.ndarray:
    """Reference FP8 grouped gemm (quantize -> matmul -> dequantize)."""
    if config.format not in _FP8_FORMAT_INFO:
        raise NotImplementedError(f"Reference FP8 format {config.format} is not supported yet")

    fp8_dtype, fp8_max = _FP8_FORMAT_INFO[config.format]

    if config.granularity == ScalingGranularity.TENSORWISE:
        a_fp8, a_scale_inv = _quantize_tensorwise(a, fp8_dtype, fp8_max)
        b_fp8, b_scale_inv = _quantize_tensorwise(b, fp8_dtype, fp8_max)
    elif config.granularity == ScalingGranularity.ROWWISE:
        a_fp8, a_scale_inv = _quantize_rowwise(a, fp8_dtype, fp8_max, axis=-1)
        b_axis = -1 if trans_b else -2
        b_fp8, b_scale_inv = _quantize_rowwise(b, fp8_dtype, fp8_max, axis=b_axis)
    else:
        raise NotImplementedError(f"Reference FP8 is not implemented for {config.granularity}")

    a_deq = _dequantize(a_fp8, a_scale_inv)
    b_deq = _dequantize(b_fp8, b_scale_inv)
    out = grouped_gemm_ref(a_deq, b_deq, group_lens, trans_b=trans_b)
    return out.astype(a.dtype)


__all__ = ["grouped_gemm_fp8_reference"]
