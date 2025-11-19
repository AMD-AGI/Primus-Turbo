###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from functools import partial

import jax

from primus_turbo.jax.primitive.grouped_gemm import (
    compute_group_offs_p,
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


@partial(jax.custom_vjp, nondiff_argnums=(6, 7, 8, 9, 10))
def grouped_gemm_fp8(
    a,
    b,
    a_scales,
    b_scales,
    group_lens,
    group_offs=None,
    transA=False,
    transB=False,
    num_cu=-1,
    out_dtype=None,
    granularity="ROWWISE",
):
    """Grouped GEMM FP8 with automatic differentiation support.

    Args:
        a: Input tensor A (FP8) with shape [m, k] or [k, m] if transA
        b: Input tensor B (FP8) with shape [bs, k, n] or [bs, n, k] if transB
        a_scales: Scaling factors for A
        b_scales: Scaling factors for B
        group_lens: Group lengths tensor [bs]
        group_offs: Group offsets tensor [bs + 1]. If None, computed internally from group_lens
        transA: Whether A is transposed
        transB: Whether B is transposed
        num_cu: Number of compute units
        out_dtype: Output dtype (jnp.float16 or jnp.bfloat16)
        granularity: Quantization granularity ("TENSORWISE" or "ROWWISE")

    Returns:
        Output tensor with shape [m, n]

    Example:
        >>> import jax.numpy as jnp
        >>> from primus_turbo.jax.lax.grouped_gemm_fp8 import grouped_gemm_fp8
        >>> G, K, N = 3, 128, 64
        >>> group_lens = jnp.array([32, 16, 48], dtype=jnp.int64)
        >>> a_fp8 = jax.random.normal(jax.random.PRNGKey(0), (group_lens.sum(), K)).astype(jnp.float8_e4m3fn)
        >>> b_fp8 = jax.random.normal(jax.random.PRNGKey(1), (G, K, N)).astype(jnp.float8_e4m3fn)
        >>> a_scales = jnp.ones((group_lens.sum(),), dtype=jnp.float32)
        >>> b_scales = jnp.ones((G * N,), dtype=jnp.float32)
        >>> out = grouped_gemm_fp8(a_fp8, b_fp8, a_scales, b_scales, group_lens)  # group_offs computed automatically
        >>> out.shape
        (96, 64)
    """
    import jax.numpy as jnp

    if group_offs is None:
        group_offs = compute_group_offs(group_lens)

    if out_dtype is None:
        out_dtype = jnp.float16

    return grouped_gemm_fp8_p.bind(
        a,
        b,
        a_scales,
        b_scales,
        group_lens,
        group_offs,
        transA=transA,
        transB=transB,
        num_cu=num_cu,
        out_dtype=out_dtype,
        granularity=granularity,
    )


def _grouped_gemm_fp8_fwd(
    a, b, a_scales, b_scales, group_lens, group_offs, transA, transB, num_cu, out_dtype, granularity
):
    """Internal forward pass that saves values for backward."""
    if group_offs is None:
        group_offs = compute_group_offs(group_lens)

    c = grouped_gemm_fp8_p.bind(
        a,
        b,
        a_scales,
        b_scales,
        group_lens,
        group_offs,
        transA=transA,
        transB=transB,
        num_cu=num_cu,
        out_dtype=out_dtype,
        granularity=granularity,
    )
    ctx = (a, b, a_scales, b_scales, group_lens, group_offs)
    return c, ctx


def _grouped_gemm_fp8_bwd(transA, transB, num_cu, out_dtype, granularity, ctx, grad_c):
    """Backward pass for grouped GEMM FP8.

    Computes gradients with respect to inputs a and b.
    """
    a, b, a_scales, b_scales, group_lens, group_offs = ctx

    # Quantize grad_c for backward
    # For simplicity, we'll use the same quantization as forward
    # In practice, you might want different quantization for backward
    import jax.numpy as jnp

    # grad_a = grad_c @ b.T (or b if transB)
    # Note: For FP8, we need to re-quantize grad_c
    # This is a simplified version - in practice you'd want proper quantization
    grad_c_scales = jnp.array([1.0], dtype=jnp.float32)  # Simplified
    grad_a = grouped_gemm_fp8_p.bind(
        grad_c,
        b,
        grad_c_scales,
        b_scales,
        group_lens,
        group_offs,
        transA=False,
        transB=not transB,
        num_cu=num_cu,
        out_dtype=out_dtype,
        granularity=granularity,
    )

    # grad_b = a.T @ grad_c (variable_k version)
    if transB:
        lhs, rhs = grad_c, a
        lhs_scales, rhs_scales = grad_c_scales, a_scales
    else:
        lhs, rhs = a, grad_c
        lhs_scales, rhs_scales = a_scales, grad_c_scales

    grad_b = grouped_gemm_fp8_variable_k_p.bind(
        lhs,
        rhs,
        lhs_scales,
        rhs_scales,
        group_lens,
        group_offs,
        transA=True,
        transB=False,
        num_cu=num_cu,
        out_dtype=out_dtype,
        granularity=granularity,
    )

    # a, b, a_scales, b_scales, group_lens, group_offs don't have gradients
    # Only a and b should have gradients (FP8 tensors)
    return grad_a, grad_b, None, None, None, None


grouped_gemm_fp8.defvjp(_grouped_gemm_fp8_fwd, _grouped_gemm_fp8_bwd)
