###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""FP8 Quantization Operators for JAX."""

from typing import Optional, Tuple

import jax
import jax.numpy as jnp

from primus_turbo.jax.core.float8 import ScalingGranularity

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
    """
    FP8 Quantize

    Signature matches PyTorch: quantize_fp8(x, out_dtype, granularity, *, axis, scale)
    Returns: (x_q, scale_inv)

    Args:
        x: Input tensor (float32, float16, or bfloat16)
        out_dtype: Output FP8 dtype (e.g., jnp.float8_e4m3fn)
        granularity: Scaling granularity (TENSORWISE or ROWWISE)
        axis: Axis for rowwise quantization (required if granularity is ROWWISE)
        scale: Pre-computed scale (if None, will be computed automatically)

    Returns:
        Tuple of (quantized_tensor, scale_inv)
    """
    # Convert dtype to string
    dtype_str_map = {
        jnp.float8_e4m3fn: "float8_e4m3fn",
        jnp.float8_e4m3fnuz: "float8_e4m3fnuz",
        jnp.float8_e5m2: "float8_e5m2",
        jnp.float8_e5m2fnuz: "float8_e5m2fnuz",
    }
    out_dtype_str = dtype_str_map.get(out_dtype)
    if out_dtype_str is None:
        raise ValueError(f"Unsupported out_dtype: {out_dtype}")

    if granularity == ScalingGranularity.TENSORWISE:
        # Prepare scale_opt (empty array if None, matching PyTorch's c10::optional behavior)
        if scale is None:
            scale_opt = jnp.empty((0,), dtype=jnp.float32)
        else:
            # Ensure scale is a scalar array
            if not isinstance(scale, jax.Array):
                scale = jnp.array(scale, dtype=jnp.float32)
            if scale.ndim == 0:
                scale = scale.reshape(1)
            scale_opt = scale

        # Call primitive: (input, scale_opt) + out_dtype_str -> (output, scale_inv)
        x_q, scale_inv = quantize_fp8_tensorwise_p.bind(x, scale_opt, out_dtype_str=out_dtype_str)

        return x_q, scale_inv

    elif granularity == ScalingGranularity.ROWWISE:
        if axis is None:
            raise ValueError("axis must be specified for rowwise FP8 quantization")

        # Prepare scale_opt
        if scale is None:
            scale_opt = jnp.empty((0,), dtype=jnp.float32)
        else:
            if not isinstance(scale, jax.Array):
                scale = jnp.array(scale, dtype=jnp.float32)
            scale_opt = scale

        # Call primitive: (input, scale_opt) + out_dtype_str, axis -> (output, scale_inv)
        x_q, scale_inv = quantize_fp8_rowwise_p.bind(x, scale_opt, out_dtype_str=out_dtype_str, axis=axis)

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
    """
    FP8 DeQuantize

    Signature matches PyTorch: dequantize_fp8(x, out_dtype, granularity, *, axis, scale_inv)
    Returns: x_dq

    Args:
        x: Input FP8 tensor
        out_dtype: Output dtype (typically float32)
        granularity: Scaling granularity (TENSORWISE or ROWWISE)
        axis: Axis for rowwise dequantization (required if granularity is ROWWISE)
        scale_inv: Inverse scale (1/scale)

    Returns:
        Dequantized tensor
    """
    # For dequantize, out_dtype is always float32
    out_dtype_str = "float32"

    if granularity == ScalingGranularity.TENSORWISE:
        # Ensure scale_inv is a scalar array
        if not isinstance(scale_inv, jax.Array):
            scale_inv = jnp.array(scale_inv, dtype=jnp.float32)
        if scale_inv.ndim == 0:
            scale_inv = scale_inv.reshape(1)

        # Call primitive: (input, scale_inv) + out_dtype_str -> output
        x_dq = dequantize_fp8_tensorwise_p.bind(x, scale_inv, out_dtype_str=out_dtype_str)

        # Cast to output dtype if needed
        if x_dq.dtype != out_dtype:
            x_dq = x_dq.astype(out_dtype)

        return x_dq

    elif granularity == ScalingGranularity.ROWWISE:
        if axis is None:
            raise ValueError("axis must be specified for rowwise FP8 de-quantization")
        # Note: rowwise dequantization is not implemented yet (same as PyTorch)
        raise NotImplementedError(f"Un-impl")

    else:
        raise NotImplementedError(f"Unknown granularity {granularity}")


"""
TODO:
quantize_mxfp8
quantize_mxfp4
"""
