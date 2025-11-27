###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from functools import partial
from typing import Optional, Union

import jax
import jax.numpy as jnp

from primus_turbo.jax.core.low_precision import (
    Float8QuantConfig,
    Format,
    ScalingGranularity,
    float8_e4m3,
    float8_e5m2,
)
from primus_turbo.jax.lax.quantization import quantize_fp8
from primus_turbo.jax.primitive.grouped_gemm.grouped_gemm import (
    compute_group_offs_p,
    get_ck_grouped_gemm_args_sizes,
    get_ck_grouped_gemm_fp8_args_sizes,
)
from primus_turbo.jax.primitive.grouped_gemm.grouped_gemm_fp8 import (
    grouped_gemm_fp8_fused_tensorwise_p,
    grouped_gemm_fp8_p,
    grouped_gemm_fp8_variable_k_p,
)

__all__ = ["grouped_gemm_fp8"]


# Workspace cache for grouped_gemm_fp8
_workspace_cache = {}


def _get_workspace(group_num, is_fp8=False):
    """Get or create workspace buffer for grouped_gemm.

    Args:
        group_num: Number of groups (batch size)
        is_fp8: Whether this is for FP8 operations

    Returns:
        Workspace buffer of appropriate size
    """
    if is_fp8:
        args_size = get_ck_grouped_gemm_fp8_args_sizes(group_num)
    else:
        args_size = get_ck_grouped_gemm_args_sizes(group_num)

    # Check cache
    cache_key = (group_num, is_fp8)
    if cache_key in _workspace_cache:
        cached_workspace, cached_size = _workspace_cache[cache_key]
        if cached_size >= args_size:
            return cached_workspace

    # Create new workspace
    workspace = jax.device_put(jnp.empty(args_size, dtype=jnp.uint8))
    _workspace_cache[cache_key] = (workspace, args_size)
    return workspace


def _get_fused_workspace_size(a_size, b_size, group_num):
    """Calculate workspace size for fused FP8 grouped_gemm.

    Workspace layout: [quant_a_ws][quant_b_ws][a_fp8][b_fp8][a_scale][b_scale][gemm_args]
    This must match the C++ implementation exactly.
    """
    from primus_turbo.jax.primitive.grouped_gemm.grouped_gemm import (
        get_ck_grouped_gemm_fp8_args_sizes,
    )

    # Calculate reduce workspace sizes (matching C++ get_reduce_row_workspace_sizes)
    # BLOCK=256, UNROLL=32, cnt = DIVUP(inner_len, BLOCK * UNROLL) * 2
    # For tensorwise: outer_len=1, inner_len=a_size or b_size
    def get_reduce_ws_size(inner_len):
        BLOCK = 256
        UNROLL = 32
        cnt = ((inner_len + BLOCK * UNROLL - 1) // (BLOCK * UNROLL)) * 2
        if cnt == 1:
            return 0
        return cnt * 1 * 4  # sizeof(float) * cnt * outer_len (outer_len=1)

    reduce_ws_a_size = get_reduce_ws_size(a_size)
    reduce_ws_b_size = get_reduce_ws_size(b_size)

    # Quantization workspace sizes (matching C++ layout)
    # amax(256) + reduce_ws(aligned) + scale(256) + scale_inv(256)
    quant_ws_a_size = 256 + ((reduce_ws_a_size + 255) // 256) * 256 + 256 + 256
    quant_ws_b_size = 256 + ((reduce_ws_b_size + 255) // 256) * 256 + 256 + 256

    # FP8 buffers (aligned)
    a_fp8_size = ((a_size + 255) // 256) * 256
    b_fp8_size = ((b_size + 255) // 256) * 256

    # Scales (256 bytes each, aligned)
    scale_size = 256

    # GEMM args
    gemm_args_size = get_ck_grouped_gemm_fp8_args_sizes(group_num)

    total_size = quant_ws_a_size + quant_ws_b_size + a_fp8_size + b_fp8_size + scale_size * 2 + gemm_args_size
    return total_size


def _expand_rowwise_scale_for_variable_k(scale, group_num):
    """Expand rowwise scale for variable_k kernel.

    Variable_k kernel with ROWWISE requires scales in [group_num, dim] format.

    Args:
        scale: Rowwise scale with shape [dim, 1], [1, dim], or [bs, dim, 1]
        group_num: Number of groups

    Returns:
        Expanded scale with shape [group_num, dim]
    """
    if scale.ndim == 2:
        # Shape [dim, 1] or [1, dim] -> broadcast to [group_num, dim]
        if scale.shape[0] == 1:
            # [1, dim] -> broadcast to [group_num, dim]
            return jnp.broadcast_to(scale, (group_num, scale.shape[1]))
        else:
            # [dim, 1] -> squeeze and broadcast
            scale_1d = jnp.squeeze(scale, axis=-1)  # [dim]
            return jnp.broadcast_to(scale_1d[None, :], (group_num, scale_1d.shape[0]))
    elif scale.ndim == 3:
        # Shape [bs, dim, 1] or [bs, 1, dim] -> reshape to [group_num, dim]
        if scale.shape[-1] == 1:
            return jnp.squeeze(scale, axis=-1)  # [bs, dim]
        else:
            return jnp.squeeze(scale, axis=-2)  # [bs, dim]
    else:
        raise ValueError(f"Unexpected scale shape: {scale.shape}")


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


@partial(jax.custom_vjp, nondiff_argnums=(4, 5, 6))
def _grouped_gemm_fp8_tensorwise(a, b, group_lens, group_offs, trans_b, config, num_cu):
    """Grouped GEMM FP8 with TENSORWISE quantization."""
    # Get FP8 dtype
    out_dtype = a.dtype

    # Convert dtype to string for FFI
    dtype_map = {jnp.float16: "float16", jnp.bfloat16: "bfloat16", jnp.float32: "float32"}
    out_dtype_str = dtype_map.get(out_dtype, "float16")

    # Use fused version for better performance (single FFI call)
    fp8_dtype_str = "e4m3" if config.format == Format.E4M3 else "e5m2"

    # Calculate workspace size for fused operation
    a_size = a.size
    b_size = b.size
    group_num = b.shape[0]
    ws_size = _get_fused_workspace_size(a_size, b_size, group_num)

    # Get or create workspace
    cache_key = ("fused", group_num, a_size, b_size)
    if cache_key in _workspace_cache:
        cached_workspace, cached_size = _workspace_cache[cache_key]
        if cached_size >= ws_size:
            workspace = cached_workspace
        else:
            workspace = jax.device_put(jnp.empty(ws_size, dtype=jnp.uint8))
            _workspace_cache[cache_key] = (workspace, ws_size)
    else:
        workspace = jax.device_put(jnp.empty(ws_size, dtype=jnp.uint8))
        _workspace_cache[cache_key] = (workspace, ws_size)

    # Call fused primitive (quantize + gemm in one FFI call)
    out = grouped_gemm_fp8_fused_tensorwise_p.bind(
        a,
        b,
        group_lens,
        group_offs,
        workspace,
        transA=False,
        transB=trans_b,
        num_cu=num_cu if num_cu is not None else -1,
        fp8_dtype_str=fp8_dtype_str,
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

    # TENSORWISE scales are scalars - pass directly to CK kernel
    workspace = _get_workspace(b_fp8.shape[0], is_fp8=True)
    out = grouped_gemm_fp8_p.bind(
        a_fp8,
        b_fp8,
        a_scale_inv,
        b_scale_inv,
        group_lens,
        group_offs,
        workspace,
        transA=False,
        transB=trans_b,
        num_cu=num_cu if num_cu is not None else -1,
        granularity="TENSORWISE",
        out_dtype_str=out_dtype_str,
    )

    # Save for backward (don't save dtype - not a JAX type)
    ctx = (a_fp8, b_fp8, a_scale_inv, b_scale_inv, group_lens, group_offs, a, b)
    return out, ctx


def _grouped_gemm_fp8_tensorwise_bwd(trans_b, config, num_cu, ctx, grad_out):
    """Backward pass for TENSORWISE quantization."""
    a_fp8, b_fp8, a_scale_inv, b_scale_inv, group_lens_saved, group_offs_saved, a, b = ctx

    # Get FP8 dtype for gradients (use same format as forward)
    grad_out_dtype = float8_e4m3 if config.format == Format.E4M3 else float8_e5m2

    # Quantize grad_out (auto-scale)
    grad_out_fp8, grad_out_scale_inv = quantize_fp8(grad_out, grad_out_dtype, ScalingGranularity.TENSORWISE)

    # Compute grad_a: grad_out @ b.T (or grad_out @ b if trans_b)
    # TENSORWISE scales are scalars - pass directly
    dtype_map = {jnp.float16: "float16", jnp.bfloat16: "bfloat16", jnp.float32: "float32"}
    out_dtype_str = dtype_map.get(a.dtype, "float16")

    workspace = _get_workspace(b_fp8.shape[0], is_fp8=True)
    grad_a = grouped_gemm_fp8_p.bind(
        grad_out_fp8,
        b_fp8,
        grad_out_scale_inv,
        b_scale_inv,
        group_lens_saved,
        group_offs_saved,
        workspace,
        transA=False,
        transB=not trans_b,
        num_cu=num_cu if num_cu is not None else -1,
        granularity="TENSORWISE",
        out_dtype_str=out_dtype_str,
    )

    # Compute grad_b: a.T @ grad_out (variable_k version)
    # After fixing kernel bug, TENSORWISE scalar scales can be passed directly
    if trans_b:
        lhs, rhs = grad_out_fp8, a_fp8
        lhs_scale, rhs_scale = grad_out_scale_inv, a_scale_inv
    else:
        lhs, rhs = a_fp8, grad_out_fp8
        lhs_scale, rhs_scale = a_scale_inv, grad_out_scale_inv

    dtype_map_b = {jnp.float16: "float16", jnp.bfloat16: "bfloat16", jnp.float32: "float32"}
    out_dtype_str_b = dtype_map_b.get(b.dtype, "float16")

    workspace_var = _get_workspace(group_lens_saved.shape[0], is_fp8=True)
    grad_b = grouped_gemm_fp8_variable_k_p.bind(
        lhs,
        rhs,
        lhs_scale,
        rhs_scale,
        group_lens_saved,
        group_offs_saved,
        workspace_var,
        transA=True,
        transB=False,
        num_cu=num_cu if num_cu is not None else -1,
        granularity="TENSORWISE",
        out_dtype_str=out_dtype_str_b,
    )

    # group_lens and group_offs are differentiable args but don't have gradients
    return grad_a, grad_b, None, None


_grouped_gemm_fp8_tensorwise.defvjp(_grouped_gemm_fp8_tensorwise_fwd, _grouped_gemm_fp8_tensorwise_bwd)


# ============================================================================
# ROWWISE Quantization
# ============================================================================


@partial(jax.custom_vjp, nondiff_argnums=(4, 5, 6))
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
        b, b_dtype, ScalingGranularity.ROWWISE, axis=(-1 if trans_b else -2)
    )

    # Forward pass - ROWWISE scales passed directly without expansion
    workspace = _get_workspace(b_fp8_row.shape[0], is_fp8=True)
    out = grouped_gemm_fp8_p.bind(
        a_fp8_row,
        b_fp8_row,
        a_scale_inv_row,
        b_scale_inv_row,
        group_lens,
        group_offs,
        workspace,
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
    workspace = _get_workspace(b_fp8_row.shape[0], is_fp8=True)
    out = grouped_gemm_fp8_p.bind(
        a_fp8_row,
        b_fp8_row,
        a_scale_inv_row,
        b_scale_inv_row,
        group_lens,
        group_offs,
        workspace,
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


def _grouped_gemm_fp8_rowwise_bwd(trans_b, config, num_cu, ctx, grad_out):
    """Backward pass for ROWWISE quantization."""
    a_fp8_col, b_fp8_col, a_scale_inv_col, b_scale_inv_col, group_lens_saved, group_offs_saved, a, b = ctx

    # Get FP8 dtype for gradients
    grad_out_dtype = float8_e4m3 if config.format == Format.E4M3 else float8_e5m2

    # Quantize grad_out (row-wise for grad_a)
    grad_out_fp8_row, grad_out_scale_inv_row = quantize_fp8(
        grad_out, grad_out_dtype, ScalingGranularity.ROWWISE, axis=-1
    )

    # Compute grad_a - ROWWISE scales passed directly
    dtype_map = {jnp.float16: "float16", jnp.bfloat16: "bfloat16", jnp.float32: "float32"}
    out_dtype_str = dtype_map.get(a.dtype, "float16")

    workspace = _get_workspace(b_fp8_col.shape[0], is_fp8=True)
    grad_a = grouped_gemm_fp8_p.bind(
        grad_out_fp8_row,
        b_fp8_col,
        grad_out_scale_inv_row,
        b_scale_inv_col,
        group_lens_saved,
        group_offs_saved,
        workspace,
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

    # Compute grad_b - ROWWISE needs expansion for variable_k
    if trans_b:
        lhs, rhs = grad_out_fp8_col, a_fp8_col
        lhs_scale, rhs_scale = grad_out_scale_inv_col, a_scale_inv_col
    else:
        lhs, rhs = a_fp8_col, grad_out_fp8_col
        lhs_scale, rhs_scale = a_scale_inv_col, grad_out_scale_inv_col

    # Expand rowwise scales for variable_k kernel
    group_num = group_lens_saved.shape[0]
    lhs_scales = _expand_rowwise_scale_for_variable_k(lhs_scale, group_num)
    rhs_scales = _expand_rowwise_scale_for_variable_k(rhs_scale, group_num)

    dtype_map_b = {jnp.float16: "float16", jnp.bfloat16: "bfloat16", jnp.float32: "float32"}
    out_dtype_str_b = dtype_map_b.get(b.dtype, "float16")

    workspace_var = _get_workspace(group_num, is_fp8=True)
    grad_b = grouped_gemm_fp8_variable_k_p.bind(
        lhs,
        rhs,
        lhs_scales,
        rhs_scales,
        group_lens_saved,
        group_offs_saved,
        workspace_var,
        transA=True,
        transB=False,
        num_cu=num_cu if num_cu is not None else -1,
        granularity="ROWWISE",
        out_dtype_str=out_dtype_str_b,
    )

    # group_lens and group_offs are differentiable args but don't have gradients
    return grad_a, grad_b, None, None


_grouped_gemm_fp8_rowwise.defvjp(_grouped_gemm_fp8_rowwise_fwd, _grouped_gemm_fp8_rowwise_bwd)


# ============================================================================
# BLOCKWISE Quantization (Placeholder - Not Implemented)
# ============================================================================


@partial(jax.custom_vjp, nondiff_argnums=(4, 5, 6))
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


def _grouped_gemm_fp8_blockwise_bwd(trans_b, config, num_cu, ctx, grad_out):
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

    This function automatically quantizes input tensors to FP8 based on the config,
    performs grouped matrix multiplication, and returns the result in the original dtype.

    Args:
        a: Input tensor A with shape [bs * m, k] (float16 or bfloat16)
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
