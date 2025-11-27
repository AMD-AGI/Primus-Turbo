###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""FP8 Quantization Operators for JAX."""

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


# Workspace cache for quantization
_quantize_workspace_cache = {}


def _get_quantize_workspace(n, is_rowwise=False, B=1, M=1, N=1, is_row_major=True):
    """Get or create workspace buffer for quantization.

    Args:
        n: Total number of elements (for tensorwise)
        is_rowwise: Whether this is rowwise quantization
        B, M, N: Dimensions for rowwise quantization
        is_row_major: Memory layout for rowwise

    Returns:
        Workspace buffer of appropriate size
    """
    if is_rowwise:
        # Calculate rowwise workspace size
        if is_row_major:
            ws_size = ((B * N * 4 + 255) // 256) * 256  # temp_scale buffer
        else:
            # Col-major: amax + reduce workspace
            # Simplified estimate: B*N*4 for amax + B*M*N*4 for reduce workspace
            amax_size = ((B * N * 4 + 255) // 256) * 256
            reduce_ws_size = ((B * M * N * 4 + 255) // 256) * 256  # Rough estimate
            ws_size = amax_size + reduce_ws_size
        cache_key = ("rowwise", B, M, N, is_row_major)
    else:
        # Tensorwise workspace size
        # amax(256) + reduce_ws + scale(256) + scale_inv(256)
        # Simplified: 256 + n*4 + 512
        ws_size = 256 + ((n * 4 + 255) // 256) * 256 + 512
        cache_key = ("tensorwise", n)

    # Check cache
    if cache_key in _quantize_workspace_cache:
        cached_workspace, cached_size = _quantize_workspace_cache[cache_key]
        if cached_size >= ws_size:
            return cached_workspace

    # Create new workspace
    workspace = jax.device_put(jnp.empty(ws_size, dtype=jnp.uint8))
    _quantize_workspace_cache[cache_key] = (workspace, ws_size)
    return workspace


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

        # Get workspace for tensorwise quantization
        n = x.size
        workspace = _get_quantize_workspace(n, is_rowwise=False)

        # Call primitive: (input, scale_opt, workspace) + out_dtype_str -> (output, scale_inv)
        x_q, scale_inv = quantize_fp8_tensorwise_p.bind(x, scale_opt, workspace, out_dtype_str=out_dtype_str)

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

        # Get workspace for rowwise quantization
        # Estimate B, M, N from shape and axis
        shape = x.shape
        if len(shape) == 2:
            if axis == -1 or axis == 1:
                B, M, N = 1, shape[0], shape[1]
                is_row_major = True
            else:  # axis == 0 or -2
                B, M, N = 1, shape[0], shape[1]
                is_row_major = False
        else:
            # For higher dimensions, use simplified estimate
            B, M, N = 1, x.size // shape[axis], shape[axis]
            is_row_major = axis == len(shape) - 1 or axis == -1

        workspace = _get_quantize_workspace(0, is_rowwise=True, B=B, M=M, N=N, is_row_major=is_row_major)

        # Call primitive: (input, scale_opt, workspace) + out_dtype_str, axis -> (output, scale_inv)
        x_q, scale_inv = quantize_fp8_rowwise_p.bind(
            x, scale_opt, workspace, out_dtype_str=out_dtype_str, axis=axis
        )

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
