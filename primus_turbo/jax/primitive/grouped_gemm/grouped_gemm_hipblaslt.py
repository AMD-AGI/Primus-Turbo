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
grouped_gemm_hipblaslt_p = Primitive("grouped_gemm_hipblaslt")
grouped_gemm_hipblaslt_p.multiple_results = False

grouped_gemm_variable_k_hipblaslt_p = Primitive("grouped_gemm_variable_k_hipblaslt")
grouped_gemm_variable_k_hipblaslt_p.multiple_results = False


# ----------------------------------------
# Step-2: Impl
# ----------------------------------------
IMPL_TABLE[grouped_gemm_hipblaslt_p] = partial(xla.apply_primitive, grouped_gemm_hipblaslt_p)
IMPL_TABLE[grouped_gemm_variable_k_hipblaslt_p] = partial(
    xla.apply_primitive, grouped_gemm_variable_k_hipblaslt_p
)


# ----------------------------------------
# Step-3: Abstract eval
# ----------------------------------------
def _grouped_gemm_hipblaslt_abstract_eval(a, b, group_lens, group_offs, transA, transB):
    """Abstract evaluation for grouped_gemm_hipblaslt.

    Args:
        a: Input tensor A with shape [m, k] or [k, m] if transA
        b: Input tensor B with shape [bs, k, n] or [bs, n, k] if transB
        group_lens: Group lengths tensor [bs]
        group_offs: Group offsets tensor [bs + 1]
        transA: Whether A is transposed
        transB: Whether B is transposed

    Returns:
        Output tensor with shape [m, n]
    """
    assert a.dtype == b.dtype, "dtype mismatch between a and b"

    # Calculate output shape based on transpose flags
    m = a.shape[1] if transA else a.shape[0]
    n = b.shape[1] if transB else b.shape[2]

    return ShapedArray((m, n), a.dtype)


ABSTRACT_EVAL_TABLE[grouped_gemm_hipblaslt_p] = _grouped_gemm_hipblaslt_abstract_eval


def _grouped_gemm_variable_k_hipblaslt_abstract_eval(a, b, group_lens, group_offs, transA, transB):
    """Abstract evaluation for grouped_gemm_variable_k_hipblaslt.

    Note: Only supports transA=True, transB=False

    Args:
        a: Input tensor A with shape [m, total_k] (will be transposed)
        b: Input tensor B with shape [n, total_k]
        group_lens: Group lengths tensor [bs]
        group_offs: Group offsets tensor [bs + 1]
        transA: Must be True
        transB: Must be False

    Returns:
        Output tensor with shape [bs, m, n]
    """
    assert a.dtype == b.dtype, "dtype mismatch between a and b"
    assert transA == True and transB == False, "Only transA=True, transB=False supported"

    # For transA=True, transB=False (called with transposed inputs):
    # a: [m, total_tokens], b: [n, total_tokens]
    # For each expert: compute a[:, rows_i] @ b[:, rows_i].T = [m, k_i] @ [k_i, n] = [m, n]
    # output: [bs, m, n]
    bs = group_lens.shape[0]
    m = a.shape[0]  # First dimension is m
    n = b.shape[0]  # First dimension is n

    return ShapedArray((bs, m, n), a.dtype)


ABSTRACT_EVAL_TABLE[grouped_gemm_variable_k_hipblaslt_p] = _grouped_gemm_variable_k_hipblaslt_abstract_eval


# ----------------------------------------
# Step-4: JIT Lowering
# ----------------------------------------
LOWERING_TABLE[grouped_gemm_hipblaslt_p] = jax.ffi.ffi_lowering("grouped_gemm_hipblaslt")

LOWERING_TABLE[grouped_gemm_variable_k_hipblaslt_p] = jax.ffi.ffi_lowering(
    "grouped_gemm_variable_k_hipblaslt"
)


# ----------------------------------------
# Step-5: batching
# ----------------------------------------
# TODO: Add batching support if needed


__all__ = [
    "grouped_gemm_hipblaslt_p",
    "grouped_gemm_variable_k_hipblaslt_p",
]
