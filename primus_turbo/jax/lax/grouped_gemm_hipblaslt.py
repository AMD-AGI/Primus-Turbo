###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from functools import partial

import jax

from primus_turbo.jax.primitive.grouped_gemm.grouped_gemm import compute_group_offs_p
from primus_turbo.jax.primitive.grouped_gemm.grouped_gemm_hipblaslt import (
    grouped_gemm_hipblaslt_p,
    grouped_gemm_variable_k_hipblaslt_p,
)

__all__ = ["grouped_gemm_hipblaslt", "compute_group_offs"]


def compute_group_offs(group_lens):
    """Compute group offsets from group lengths.

    Args:
        group_lens: Group lengths tensor [bs]

    Returns:
        Group offsets tensor [bs + 1]
    """
    return compute_group_offs_p.bind(group_lens)


@partial(jax.custom_vjp, nondiff_argnums=(4, 5, 6))
def grouped_gemm_hipblaslt(a, b, group_lens, group_offs=None, transA=False, transB=False, num_cu=-1):
    """Grouped GEMM with hipBLASLt backend and automatic differentiation support.

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

    return grouped_gemm_hipblaslt_p.bind(
        a, b, group_lens, group_offs, transA=transA, transB=transB, num_cu=num_cu
    )


# Ref: https://docs.jax.dev/en/latest/_autosummary/jax.custom_vjp.defvjp.html
# Input : same input signature as the underlying primal function
# Output: out, ctx
def _grouped_gemm_hipblaslt_fwd(a, b, group_lens, group_offs, transA, transB, num_cu):
    """Internal forward pass that saves values for backward."""
    if group_offs is None:
        group_offs = compute_group_offs(group_lens)

    c = grouped_gemm_hipblaslt_p.bind(
        a, b, group_lens, group_offs, transA=transA, transB=transB, num_cu=num_cu
    )
    ctx = (a, b, group_lens, group_offs)
    return c, ctx


# input: nondiff_argnums, ctx, grad
# output: input grad
def _grouped_gemm_hipblaslt_bwd(transA, transB, num_cu, ctx, grad_c):
    """Backward pass for grouped GEMM with hipBLASLt backend.

    Computes gradients with respect to inputs a and b.
    """
    a, b, group_lens, group_offs = ctx

    # grad_a = grad_c @ b.T (or b if transB)
    grad_a = grouped_gemm_hipblaslt_p.bind(
        grad_c, b, group_lens, group_offs, transA=False, transB=not transB, num_cu=num_cu
    )

    # grad_b = a.T @ grad_c (variable_k version)
    # For transB=True: Forward is C = A @ B.T, so grad_B = grad_C.T @ A
    # For transB=False: Forward is C = A @ B, so grad_B = A.T @ grad_C
    if transB:
        lhs, rhs = grad_c.T, a.T  # Transpose both!
    else:
        lhs, rhs = a.T, grad_c.T  # Transpose both!

    grad_b = grouped_gemm_variable_k_hipblaslt_p.bind(
        lhs, rhs, group_lens, group_offs, transA=True, transB=False, num_cu=num_cu
    )

    # group_lens, group_offs don't have gradients
    return grad_a, grad_b, None, None


grouped_gemm_hipblaslt.defvjp(_grouped_gemm_hipblaslt_fwd, _grouped_gemm_hipblaslt_bwd)
