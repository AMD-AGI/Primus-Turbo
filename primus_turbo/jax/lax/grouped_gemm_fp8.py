###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from functools import partial
from typing import Optional, Union

import jax
import jax.numpy as jnp

from primus_turbo.jax.core.float8 import (
    Float8QuantConfig,
    Format,
    ScalingGranularity,
    float8_e4m3,
    float8_e5m2,
)
from primus_turbo.jax.lax.quantization import quantize_fp8
from primus_turbo.jax.primitive.grouped_gemm import compute_group_offs_p
from primus_turbo.jax.primitive.grouped_gemm_fp8 import (
    grouped_gemm_fp8_p,
    grouped_gemm_fp8_variable_k_p,
)

__all__ = ["grouped_gemm_fp8"]


def compute_group_offs(group_lens):
    """Compute group offsets from group lengths.

    Args:
        group_lens: Group lengths tensor [bs]

    Returns:
        Group offsets tensor [bs + 1]
    """
    return compute_group_offs_p.bind(group_lens)


# ============================================================================
# TENSORWISE Quantization
# ============================================================================


@partial(jax.custom_vjp, nondiff_argnums=(2, 3, 4, 5, 6))
def _grouped_gemm_fp8_tensorwise(a, b, group_lens, group_offs, trans_b, config, num_cu):
    """Grouped GEMM FP8 with TENSORWISE quantization."""
    # Get FP8 dtype
    a_dtype = float8_e4m3 if config.format == Format.E4M3 else float8_e5m2
    b_dtype = float8_e4m3 if config.format == Format.E4M3 else float8_e5m2
    out_dtype = a.dtype

    # Convert dtype to string for FFI
    dtype_map = {jnp.float16: "float16", jnp.bfloat16: "bfloat16", jnp.float32: "float32"}
    out_dtype_str = dtype_map.get(out_dtype, "float16")

    # Quantize a and b (auto-scale)
    a_fp8, a_scale_inv = quantize_fp8(a, a_dtype, ScalingGranularity.TENSORWISE)
    b_fp8, b_scale_inv = quantize_fp8(b, b_dtype, ScalingGranularity.TENSORWISE)

    # Forward pass
    out = grouped_gemm_fp8_p.bind(
        a_fp8,
        b_fp8,
        a_scale_inv,
        b_scale_inv,
        group_lens,
        group_offs,
        transA=False,
        transB=trans_b,
        num_cu=num_cu if num_cu is not None else -1,
        granularity="TENSORWISE",
        out_dtype_str=out_dtype_str,
    )
    return out


def _grouped_gemm_fp8_tensorwise_fwd(a, b, group_lens, group_offs, trans_b, config, num_cu):
    """Forward pass that saves values for backward."""
    # Get FP8 dtype
    a_dtype = float8_e4m3 if config.format == Format.E4M3 else float8_e5m2
    b_dtype = float8_e4m3 if config.format == Format.E4M3 else float8_e5m2
    out_dtype = a.dtype

    # Convert dtype to string for FFI
    dtype_map = {jnp.float16: "float16", jnp.bfloat16: "bfloat16", jnp.float32: "float32"}
    out_dtype_str = dtype_map.get(out_dtype, "float16")

    # Quantize a and b (auto-scale)
    a_fp8, a_scale_inv = quantize_fp8(a, a_dtype, ScalingGranularity.TENSORWISE)
    b_fp8, b_scale_inv = quantize_fp8(b, b_dtype, ScalingGranularity.TENSORWISE)

    # Forward pass
    out = grouped_gemm_fp8_p.bind(
        a_fp8,
        b_fp8,
        a_scale_inv,
        b_scale_inv,
        group_lens,
        group_offs,
        transA=False,
        transB=trans_b,
        num_cu=num_cu if num_cu is not None else -1,
        granularity="TENSORWISE",
        out_dtype_str=out_dtype_str,
    )

    # Save for backward (don't save dtype - not a JAX type)
    ctx = (a_fp8, b_fp8, a_scale_inv, b_scale_inv, group_lens, group_offs, a, b)
    return out, ctx


def _grouped_gemm_fp8_tensorwise_bwd(group_lens, group_offs, trans_b, config, num_cu, ctx, grad_out):
    """Backward pass for TENSORWISE quantization."""
    a_fp8, b_fp8, a_scale_inv, b_scale_inv, group_lens_saved, group_offs_saved, a, b = ctx

    # Get FP8 dtype for gradients (use same format as forward)
    grad_out_dtype = float8_e4m3 if config.format == Format.E4M3 else float8_e5m2

    # Quantize grad_out (auto-scale)
    grad_out_fp8, grad_out_scale_inv = quantize_fp8(grad_out, grad_out_dtype, ScalingGranularity.TENSORWISE)

    # Compute grad_a: grad_out @ b.T (or grad_out @ b if trans_b)
    dtype_map = {jnp.float16: "float16", jnp.bfloat16: "bfloat16", jnp.float32: "float32"}
    out_dtype_str = dtype_map.get(a.dtype, "float16")

    grad_a = grouped_gemm_fp8_p.bind(
        grad_out_fp8,
        b_fp8,
        grad_out_scale_inv,
        b_scale_inv,
        group_lens_saved,
        group_offs_saved,
        transA=False,
        transB=not trans_b,
        num_cu=num_cu if num_cu is not None else -1,
        granularity="TENSORWISE",
        out_dtype_str=out_dtype_str,
    )

    # Compute grad_b: a.T @ grad_out (variable_k version)
    if trans_b:
        lhs, rhs = grad_out_fp8, a_fp8
        lhs_scale, rhs_scale = grad_out_scale_inv, a_scale_inv
    else:
        lhs, rhs = a_fp8, grad_out_fp8
        lhs_scale, rhs_scale = a_scale_inv, grad_out_scale_inv

    dtype_map_b = {jnp.float16: "float16", jnp.bfloat16: "bfloat16", jnp.float32: "float32"}
    out_dtype_str_b = dtype_map_b.get(b.dtype, "float16")

    grad_b = grouped_gemm_fp8_variable_k_p.bind(
        lhs,
        rhs,
        lhs_scale,
        rhs_scale,
        group_lens_saved,
        group_offs_saved,
        transA=True,
        transB=False,
        num_cu=num_cu if num_cu is not None else -1,
        granularity="TENSORWISE",
        out_dtype_str=out_dtype_str_b,
    )

    return grad_a, grad_b


_grouped_gemm_fp8_tensorwise.defvjp(_grouped_gemm_fp8_tensorwise_fwd, _grouped_gemm_fp8_tensorwise_bwd)


# ============================================================================
# ROWWISE Quantization
# ============================================================================


@partial(jax.custom_vjp, nondiff_argnums=(2, 3, 4, 5, 6))
def _grouped_gemm_fp8_rowwise(a, b, group_lens, group_offs, trans_b, config, num_cu):
    """Grouped GEMM FP8 with ROWWISE quantization."""
    # Get FP8 dtype
    a_dtype = float8_e4m3 if config.format == Format.E4M3 else float8_e5m2
    b_dtype = float8_e4m3 if config.format == Format.E4M3 else float8_e5m2
    out_dtype = a.dtype

    # Convert dtype to string for FFI
    dtype_map = {jnp.float16: "float16", jnp.bfloat16: "bfloat16", jnp.float32: "float32"}
    out_dtype_str = dtype_map.get(out_dtype, "float16")

    # Quantize a and b (row-wise)
    a_fp8_row, a_scale_inv_row = quantize_fp8(a, a_dtype, ScalingGranularity.ROWWISE, axis=-1)
    b_fp8_row, b_scale_inv_row = quantize_fp8(
        b, b_dtype, ScalingGranularity.ROWWISE, axis=(2 if trans_b else 1)
    )

    # Forward pass
    out = grouped_gemm_fp8_p.bind(
        a_fp8_row,
        b_fp8_row,
        a_scale_inv_row,
        b_scale_inv_row,
        group_lens,
        group_offs,
        transA=False,
        transB=trans_b,
        num_cu=num_cu if num_cu is not None else -1,
        granularity="ROWWISE",
        out_dtype_str=out_dtype_str,
    )
    return out


def _grouped_gemm_fp8_rowwise_fwd(a, b, group_lens, group_offs, trans_b, config, num_cu):
    """Forward pass that saves values for backward."""
    # Get FP8 dtype
    a_dtype = float8_e4m3 if config.format == Format.E4M3 else float8_e5m2
    b_dtype = float8_e4m3 if config.format == Format.E4M3 else float8_e5m2
    out_dtype = a.dtype

    # Convert dtype to string for FFI
    dtype_map = {jnp.float16: "float16", jnp.bfloat16: "bfloat16", jnp.float32: "float32"}
    out_dtype_str = dtype_map.get(out_dtype, "float16")

    # Quantize a and b (row-wise for forward)
    a_fp8_row, a_scale_inv_row = quantize_fp8(a, a_dtype, ScalingGranularity.ROWWISE, axis=-1)
    b_fp8_row, b_scale_inv_row = quantize_fp8(
        b, b_dtype, ScalingGranularity.ROWWISE, axis=(-1 if trans_b else -2)
    )
    # Forward pass
    out = grouped_gemm_fp8_p.bind(
        a_fp8_row,
        b_fp8_row,
        a_scale_inv_row,
        b_scale_inv_row,
        group_lens,
        group_offs,
        transA=False,
        transB=trans_b,
        num_cu=num_cu if num_cu is not None else -1,
        granularity="ROWWISE",
        out_dtype_str=out_dtype_str,
    ).astype(out_dtype)

    # Quantize a and b (col-wise for backward)
    a_fp8_col, a_scale_inv_col = quantize_fp8(a, a_dtype, ScalingGranularity.ROWWISE, axis=-2)
    b_fp8_col, b_scale_inv_col = quantize_fp8(
        b, b_dtype, ScalingGranularity.ROWWISE, axis=(-2 if trans_b else -1)
    )

    # Save for backward
    ctx = (a_fp8_col, b_fp8_col, a_scale_inv_col, b_scale_inv_col, group_lens, group_offs, a, b)
    return out, ctx


def _grouped_gemm_fp8_rowwise_bwd(group_lens, group_offs, trans_b, config, num_cu, ctx, grad_out):
    """Backward pass for ROWWISE quantization."""
    a_fp8_col, b_fp8_col, a_scale_inv_col, b_scale_inv_col, group_lens_saved, group_offs_saved, a, b = ctx

    # Get FP8 dtype for gradients
    grad_out_dtype = float8_e4m3 if config.format == Format.E4M3 else float8_e5m2

    # Quantize grad_out (row-wise for grad_a)
    grad_out_fp8_row, grad_out_scale_inv_row = quantize_fp8(
        grad_out, grad_out_dtype, ScalingGranularity.ROWWISE, axis=-1
    )

    # Compute grad_a
    dtype_map = {jnp.float16: "float16", jnp.bfloat16: "bfloat16", jnp.float32: "float32"}
    out_dtype_str = dtype_map.get(a.dtype, "float16")

    grad_a = grouped_gemm_fp8_p.bind(
        grad_out_fp8_row,
        b_fp8_col,
        grad_out_scale_inv_row,
        b_scale_inv_col,
        group_lens_saved,
        group_offs_saved,
        transA=False,
        transB=not trans_b,
        num_cu=num_cu if num_cu is not None else -1,
        granularity="ROWWISE",
        out_dtype_str=out_dtype_str,
    )

    # Quantize grad_out (col-wise for grad_b)
    grad_out_fp8_col, grad_out_scale_inv_col = quantize_fp8(
        grad_out, grad_out_dtype, ScalingGranularity.ROWWISE, axis=-2
    )

    # Compute grad_b
    if trans_b:
        lhs, rhs = grad_out_fp8_col, a_fp8_col
        lhs_scale, rhs_scale = grad_out_scale_inv_col, a_scale_inv_col
    else:
        lhs, rhs = a_fp8_col, grad_out_fp8_col
        lhs_scale, rhs_scale = a_scale_inv_col, grad_out_scale_inv_col

    dtype_map_b = {jnp.float16: "float16", jnp.bfloat16: "bfloat16", jnp.float32: "float32"}
    out_dtype_str_b = dtype_map_b.get(b.dtype, "float16")

    grad_b = grouped_gemm_fp8_variable_k_p.bind(
        lhs,
        rhs,
        lhs_scale,
        rhs_scale,
        group_lens_saved,
        group_offs_saved,
        transA=True,
        transB=False,
        num_cu=num_cu if num_cu is not None else -1,
        granularity="ROWWISE",
        out_dtype_str=out_dtype_str_b,
    )

    return grad_a, grad_b


_grouped_gemm_fp8_rowwise.defvjp(_grouped_gemm_fp8_rowwise_fwd, _grouped_gemm_fp8_rowwise_bwd)


# ============================================================================
# BLOCKWISE Quantization (Placeholder - Not Implemented)
# ============================================================================


@partial(jax.custom_vjp, nondiff_argnums=(2, 3, 4, 5, 6))
def _grouped_gemm_fp8_blockwise(a, b, group_lens, group_offs, trans_b, config, num_cu):
    """Grouped GEMM FP8 with BLOCKWISE quantization.

    Note: BLOCKWISE quantization is not yet implemented in JAX.
    This is a placeholder for future implementation.
    """
    raise NotImplementedError(
        "BLOCKWISE quantization is not yet implemented in JAX. "
        "Please use TENSORWISE or ROWWISE granularity."
    )


def _grouped_gemm_fp8_blockwise_fwd(a, b, group_lens, group_offs, trans_b, config, num_cu):
    raise NotImplementedError("BLOCKWISE quantization not implemented")


def _grouped_gemm_fp8_blockwise_bwd(group_lens, group_offs, trans_b, config, num_cu, ctx, grad_out):
    raise NotImplementedError("BLOCKWISE quantization not implemented")


_grouped_gemm_fp8_blockwise.defvjp(_grouped_gemm_fp8_blockwise_fwd, _grouped_gemm_fp8_blockwise_bwd)


# ============================================================================
# Main Entry Point
# ============================================================================


def grouped_gemm_fp8(
    a: jax.Array,
    b: jax.Array,
    group_lens: jax.Array,
    group_offs: Optional[jax.Array] = None,
    trans_b: bool = True,
    config: Union[Float8QuantConfig, None] = None,
    num_cu: Optional[int] = None,
) -> jax.Array:
    """Grouped GEMM with FP8 quantization.

    Signature matches PyTorch: grouped_gemm_fp8(a, b, group_lens, group_offs, trans_b, config, num_cu)

    This function automatically quantizes input tensors to FP8 based on the config,
    performs grouped matrix multiplication, and returns the result in the original dtype.

    Args:
        a: Input tensor A with shape [m, k] (float16 or bfloat16)
        b: Input tensor B with shape [bs, k, n] or [bs, n, k] if trans_b (float16 or bfloat16)
        group_lens: Group lengths tensor [bs] (int64)
        group_offs: Group offsets tensor [bs + 1] (int64). If None, computed from group_lens
        trans_b: Whether B is transposed (default: True)
        config: FP8 quantization config. If None, uses default (TENSORWISE, E4M3, DYNAMIC)
        num_cu: Number of compute units. If None, uses default (-1)

    Returns:
        Output tensor with shape [m, n] (same dtype as input)

    Raises:
        AssertionError: If input shapes or dtypes are invalid
        NotImplementedError: If BLOCKWISE quantization is requested

    Example:
        >>> import jax.numpy as jnp
        >>> from primus_turbo.jax.lax.grouped_gemm_fp8 import grouped_gemm_fp8
        >>> from primus_turbo.jax.core.float8 import Float8QuantConfig, ScalingGranularity
        >>>
        >>> # Setup
        >>> G, M, K, N = 3, 96, 128, 64
        >>> group_lens = jnp.array([32, 16, 48], dtype=jnp.int64)
        >>> a = jax.random.normal(jax.random.PRNGKey(0), (M, K), dtype=jnp.float16)
        >>> b = jax.random.normal(jax.random.PRNGKey(1), (G, K, N), dtype=jnp.float16)
        >>>
        >>> # TENSORWISE quantization (default)
        >>> out = grouped_gemm_fp8(a, b, group_lens)
        >>>
        >>> # ROWWISE quantization
        >>> config = Float8QuantConfig(granularity=ScalingGranularity.ROWWISE)
        >>> out = grouped_gemm_fp8(a, b, group_lens, config=config)
    """
    supported_dtypes = [jnp.bfloat16, jnp.float16]
    assert a.dtype in supported_dtypes, f"Unsupported dtype {a.dtype}, expected one of {supported_dtypes}"
    assert b.dtype in supported_dtypes, f"Unsupported dtype {b.dtype}, expected one of {supported_dtypes}"

    # Compute group_offs if not provided
    if group_offs is None:
        group_offs = compute_group_offs(group_lens)

    # Use default config if not provided
    if config is None:
        config = Float8QuantConfig()

    # Dispatch based on granularity
    if config.granularity == ScalingGranularity.TENSORWISE:
        return _grouped_gemm_fp8_tensorwise(a, b, group_lens, group_offs, trans_b, config, num_cu)
    elif config.granularity == ScalingGranularity.ROWWISE:
        return _grouped_gemm_fp8_rowwise(a, b, group_lens, group_offs, trans_b, config, num_cu)
    elif config.granularity == ScalingGranularity.BLOCKWISE:
        return _grouped_gemm_fp8_blockwise(a, b, group_lens, group_offs, trans_b, config, num_cu)
    else:
        raise ValueError(f"Unsupported FP8 ScalingGranularity: {config.granularity}")


"""
TODO: MXFP8, MXFP4
"""
