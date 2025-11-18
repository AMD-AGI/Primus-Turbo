###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from functools import partial

import jax
from jax.core import ShapedArray
from jax.extend.core import Primitive
from jax.interpreters import xla

from primus_turbo.jax.primitive import ABSTRACT_EVAL_TABLE, IMPL_TABLE, LOWERING_TABLE

# ----------------------------------------
# Step-1: Primitive Define
# ----------------------------------------
grouped_gemm_p = Primitive("grouped_gemm")
grouped_gemm_p.multiple_results = False

grouped_gemm_variable_k_p = Primitive("grouped_gemm_variable_k")
grouped_gemm_variable_k_p.multiple_results = False

compute_group_offs_p = Primitive("compute_group_offs")
compute_group_offs_p.multiple_results = False


# ----------------------------------------
# Step-2: Impl
# ----------------------------------------
IMPL_TABLE[grouped_gemm_p] = partial(xla.apply_primitive, grouped_gemm_p)
IMPL_TABLE[grouped_gemm_variable_k_p] = partial(xla.apply_primitive, grouped_gemm_variable_k_p)
IMPL_TABLE[compute_group_offs_p] = partial(xla.apply_primitive, compute_group_offs_p)


# ----------------------------------------
# Step-3: Abstract eval
# ----------------------------------------
def _grouped_gemm_abstract_eval(a, b, group_lens, group_offs, transA, transB, num_cu):
    """Abstract evaluation for grouped_gemm.

    Args:
        a: Input tensor A with shape [m, k] or [k, m] if transA
        b: Input tensor B with shape [bs, k, n] or [bs, n, k] if transB
        group_lens: Group lengths tensor [bs]
        group_offs: Group offsets tensor [bs + 1]
        transA: Whether A is transposed
        transB: Whether B is transposed
        num_cu: Number of compute units

    Returns:
        Output tensor with shape [m, n]
    """
    assert a.dtype == b.dtype, "dtype mismatch between a and b"

    # Calculate output shape based on transpose flags
    m = a.shape[1] if transA else a.shape[0]
    n = b.shape[1] if transB else b.shape[2]

    return ShapedArray((m, n), a.dtype)


ABSTRACT_EVAL_TABLE[grouped_gemm_p] = _grouped_gemm_abstract_eval


def _compute_group_offs_abstract_eval(group_lens):
    """Abstract evaluation for compute_group_offs.

    Args:
        group_lens: Group lengths tensor [bs]

    Returns:
        Group offsets tensor [bs + 1]
    """
    bs = group_lens.shape[0]
    return ShapedArray((bs + 1,), group_lens.dtype)


ABSTRACT_EVAL_TABLE[compute_group_offs_p] = _compute_group_offs_abstract_eval


def _grouped_gemm_variable_k_abstract_eval(a, b, group_lens, group_offs, transA, transB, num_cu):
    """Abstract evaluation for grouped_gemm_variable_k.

    Note: Only supports transA=True, transB=False

    Args:
        a: Input tensor A with shape [k, m] (will be transposed)
        b: Input tensor B with shape [k, n]
        group_lens: Group lengths tensor [bs]
        group_offs: Group offsets tensor [bs + 1]
        transA: Must be True
        transB: Must be False
        num_cu: Number of compute units

    Returns:
        Output tensor with shape [bs, m, n]
    """
    assert a.dtype == b.dtype, "dtype mismatch between a and b"
    assert transA == True and transB == False, "Only transA=True, transB=False supported"

    # For transA=True, transB=False:
    # a: [k, m] (will be transposed to [m, k]), b: [k, n]
    # a.T @ b = [m, k] @ [k, n] = [m, n] for each group
    # output: [bs, m, n]
    bs = group_lens.shape[0]
    a.shape[0]
    m = a.shape[1]
    n = b.shape[1]

    return ShapedArray((bs, m, n), a.dtype)


ABSTRACT_EVAL_TABLE[grouped_gemm_variable_k_p] = _grouped_gemm_variable_k_abstract_eval


# ----------------------------------------
# Step-4: JIT Lowering
# ----------------------------------------
LOWERING_TABLE[grouped_gemm_p] = jax.ffi.ffi_lowering("grouped_gemm")

LOWERING_TABLE[grouped_gemm_variable_k_p] = jax.ffi.ffi_lowering("grouped_gemm_variable_k")

LOWERING_TABLE[compute_group_offs_p] = jax.ffi.ffi_lowering("compute_group_offs")


# ----------------------------------------
# Step-5: Forward & Backward (custom_vjp)
# ----------------------------------------
def grouped_gemm_forward(a, b, group_lens, group_offs, transA=False, transB=False, num_cu=-1):
    """Forward pass for grouped GEMM (with gradient tracking).

    This is the main entry point for grouped GEMM with automatic differentiation support.
    """
    return grouped_gemm_p.bind(a, b, group_lens, group_offs, transA=transA, transB=transB, num_cu=num_cu)


def grouped_gemm_forward_fwd(a, b, group_lens, group_offs, transA=False, transB=False, num_cu=-1):
    """Internal forward pass that saves values for backward."""
    c = grouped_gemm_p.bind(a, b, group_lens, group_offs, transA=transA, transB=transB, num_cu=num_cu)
    return c, (a, b, group_lens, group_offs, transA, transB, num_cu)


def grouped_gemm_backward(res, grad_c):
    """Backward pass for grouped GEMM.

    Computes gradients with respect to inputs a and b.
    """
    a, b, group_lens, group_offs, transA, transB, num_cu = res

    # grad_a = grad_c @ b.T (or b if transB)
    grad_a = grouped_gemm_p.bind(
        grad_c, b, group_lens, group_offs, transA=False, transB=not transB, num_cu=num_cu
    )

    # grad_b = a.T @ grad_c (variable_k version)
    # lhs, rhs = (grad_c, a) if transB else (a, grad_c)
    if transB:
        lhs, rhs = grad_c, a
    else:
        lhs, rhs = a, grad_c

    grad_b = grouped_gemm_variable_k_p.bind(
        lhs, rhs, group_lens, group_offs, transA=True, transB=False, num_cu=num_cu
    )

    # group_lens, group_offs, and scalar params don't have gradients
    return grad_a, grad_b, None, None, None, None, None


# Register custom VJP
grouped_gemm_forward = jax.custom_vjp(grouped_gemm_forward)
grouped_gemm_forward.defvjp(grouped_gemm_forward_fwd, grouped_gemm_backward)


# ----------------------------------------
# Step-6: batching
# ----------------------------------------
# TODO: Add batching support if needed


__all__ = [
    "compute_group_offs_p",
    "grouped_gemm_forward",
    "grouped_gemm_backward",
]
