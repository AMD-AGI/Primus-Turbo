###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from functools import partial

import jax
import jax.numpy as jnp

from primus_turbo.jax.primitive.grouped_gemm.grouped_gemm import (
    compute_group_offs_p,
    get_ck_grouped_gemm_args_sizes,
    grouped_gemm_p,
    grouped_gemm_variable_k_p,
)

__all__ = ["grouped_gemm", "compute_group_offs"]


def compute_group_offs(group_lens):
    """Compute group offsets from group lengths.

    Args:
        group_lens: Group lengths tensor [bs]

    Returns:
        Group offsets tensor [bs + 1]
    """
    return compute_group_offs_p.bind(group_lens)


# Workspace cache for grouped_gemm
# Use a simple dict with (group_num, args_size) as key
_workspace_cache = {}


def _get_workspace(group_num):
    """Get or create workspace buffer for grouped_gemm.

    Args:
        group_num: Number of groups (batch size)

    Returns:
        Workspace buffer of appropriate size
    """
    # Calculate exact size - matching C++ implementation
    # sizeof(ck_tile::GemmTransKernelArg<>) is approximately 200 bytes per group
    # Use C++ calculated size for accuracy
    args_size = get_ck_grouped_gemm_args_sizes(group_num)

    # Check cache
    if group_num in _workspace_cache:
        cached_workspace, cached_size = _workspace_cache[group_num]
        if cached_size >= args_size:
            # Reuse existing buffer if size is sufficient
            return cached_workspace

    # Create new workspace - use device_put to ensure it's on GPU
    workspace = jax.device_put(jnp.empty(args_size, dtype=jnp.uint8))
    _workspace_cache[group_num] = (workspace, args_size)
    return workspace


@partial(jax.custom_vjp, nondiff_argnums=(4, 5, 6))
def grouped_gemm(a, b, group_lens, group_offs=None, transA=False, transB=False, num_cu=-1):
    """Grouped GEMM with automatic differentiation support.

    Args:
        a: Input tensor A with shape [bs * m, k] or [k, bs * m] if transA
        b: Input tensor B with shape [bs, k, n] or [bs, n, k] if transB
        group_lens: Group lengths tensor [bs]
        group_offs: Group offsets tensor [bs + 1]. If None, computed internally from group_lens
        transA: Whether A is transposed
        transB: Whether B is transposed
        num_cu: Number of compute units

    Returns:
        Output tensor with shape [m, n]
    """
    if group_offs is None:
        group_offs = compute_group_offs(group_lens)

    # Get workspace buffer
    group_num = b.shape[0]
    workspace = _get_workspace(group_num)

    return grouped_gemm_p.bind(
        a, b, group_lens, group_offs, workspace, transA=transA, transB=transB, num_cu=num_cu
    )


# Ref: https://docs.jax.dev/en/latest/_autosummary/jax.custom_vjp.defvjp.html
# Input : same input signature as the underlying primal function
# Output: out, ctx
def _grouped_gemm_fwd(a, b, group_lens, group_offs, transA, transB, num_cu):
    """Internal forward pass that saves values for backward."""
    if group_offs is None:
        group_offs = compute_group_offs(group_lens)

    # Get workspace buffer
    group_num = b.shape[0]
    workspace = _get_workspace(group_num)

    c = grouped_gemm_p.bind(
        a, b, group_lens, group_offs, workspace, transA=transA, transB=transB, num_cu=num_cu
    )
    ctx = (a, b, group_lens, group_offs)
    return c, ctx


# input: nondiff_argnums, ctx, grad
# output: input grad
def _grouped_gemm_bwd(transA, transB, num_cu, ctx, grad_c):
    """Backward pass for grouped GEMM.

    Computes gradients with respect to inputs a and b.
    """
    a, b, group_lens, group_offs = ctx

    # Get workspace buffers
    group_num = b.shape[0]
    workspace1 = _get_workspace(group_num)
    workspace2 = _get_workspace(len(group_lens))

    # grad_a = grad_c @ b.T (or b if transB)
    grad_a = grouped_gemm_p.bind(
        grad_c, b, group_lens, group_offs, workspace1, transA=False, transB=not transB, num_cu=num_cu
    )

    # grad_b = a.T @ grad_c (variable_k version)
    # lhs, rhs = (grad_c, a) if transB else (a, grad_c)
    if transB:
        lhs, rhs = grad_c, a
    else:
        lhs, rhs = a, grad_c

    grad_b = grouped_gemm_variable_k_p.bind(
        lhs, rhs, group_lens, group_offs, workspace2, transA=True, transB=False, num_cu=num_cu
    )

    # group_lens, group_offs don't have gradients
    return grad_a, grad_b, None, None


grouped_gemm.defvjp(_grouped_gemm_fwd, _grouped_gemm_bwd)
