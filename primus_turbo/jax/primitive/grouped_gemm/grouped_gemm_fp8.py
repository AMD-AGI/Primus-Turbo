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
grouped_gemm_fp8_p = Primitive("grouped_gemm_fp8")
grouped_gemm_fp8_p.multiple_results = False

grouped_gemm_fp8_variable_k_p = Primitive("grouped_gemm_fp8_variable_k")
grouped_gemm_fp8_variable_k_p.multiple_results = False


# ----------------------------------------
# Step-2: Impl
# ----------------------------------------
IMPL_TABLE[grouped_gemm_fp8_p] = partial(xla.apply_primitive, grouped_gemm_fp8_p)
IMPL_TABLE[grouped_gemm_fp8_variable_k_p] = partial(xla.apply_primitive, grouped_gemm_fp8_variable_k_p)


# ----------------------------------------
# Step-3: Abstract eval
# ----------------------------------------
def _grouped_gemm_fp8_abstract_eval(
    a, b, a_scales, b_scales, group_lens, group_offs, *, transA, transB, num_cu, granularity, out_dtype_str
):
    """Abstract evaluation for grouped_gemm_fp8.

    Args:
        a: Input tensor A (FP8) with shape [m, k] or [k, m] if transA
        b: Input tensor B (FP8) with shape [bs, k, n] or [bs, n, k] if transB
        a_scales: Scaling factors for A
        b_scales: Scaling factors for B
        group_lens: Group lengths tensor [bs]
        group_offs: Group offsets tensor [bs + 1]
        transA: Whether A is transposed
        transB: Whether B is transposed
        num_cu: Number of compute units
        granularity: Quantization granularity ("TENSORWISE" or "ROWWISE")
        out_dtype_str: Output dtype string ("float16" or "bfloat16")

    Returns:
        Output tensor with shape [m, n]
    """
    # Calculate output shape based on transpose flags
    m = a.shape[1] if transA else a.shape[0]
    n = b.shape[1] if transB else b.shape[2]

    # Map string to JAX dtype
    import jax.numpy as jnp

    dtype_map = {"float16": jnp.float16, "bfloat16": jnp.bfloat16}
    out_dtype = dtype_map.get(out_dtype_str, jnp.bfloat16)

    return ShapedArray((m, n), out_dtype)


ABSTRACT_EVAL_TABLE[grouped_gemm_fp8_p] = _grouped_gemm_fp8_abstract_eval


def _grouped_gemm_fp8_variable_k_abstract_eval(
    a, b, a_scales, b_scales, group_lens, group_offs, *, transA, transB, num_cu, granularity, out_dtype_str
):
    """Abstract evaluation for grouped_gemm_fp8_variable_k.

    Note: Only supports transA=True, transB=False

    Args:
        a: Input tensor A (FP8) with shape [k, m] (will be transposed)
        b: Input tensor B (FP8) with shape [k, n]
        a_scales: Scaling factors for A
        b_scales: Scaling factors for B
        group_lens: Group lengths tensor [bs]
        group_offs: Group offsets tensor [bs + 1]
        transA: Must be True
        transB: Must be False
        num_cu: Number of compute units
        granularity: Quantization granularity ("TENSORWISE" or "ROWWISE")
        out_dtype_str: Output dtype string ("float16" or "bfloat16")

    Returns:
        Output tensor with shape [bs, m, n]
    """
    assert transA == True and transB == False, "Only transA=True, transB=False supported"

    # For transA=True, transB=False:
    # a: [k, m] (will be transposed to [m, k]), b: [k, n]
    # a.T @ b = [m, k] @ [k, n] = [m, n] for each group
    # output: [bs, m, n]
    bs = group_lens.shape[0]
    m = a.shape[1]
    n = b.shape[1]

    # Map string to JAX dtype
    import jax.numpy as jnp

    dtype_map = {"float16": jnp.float16, "bfloat16": jnp.bfloat16}
    out_dtype = dtype_map.get(out_dtype_str, jnp.bfloat16)

    return ShapedArray((bs, m, n), out_dtype)


ABSTRACT_EVAL_TABLE[grouped_gemm_fp8_variable_k_p] = _grouped_gemm_fp8_variable_k_abstract_eval


# ----------------------------------------
# Step-4: JIT Lowering
# ----------------------------------------
LOWERING_TABLE[grouped_gemm_fp8_p] = jax.ffi.ffi_lowering("grouped_gemm_fp8")

LOWERING_TABLE[grouped_gemm_fp8_variable_k_p] = jax.ffi.ffi_lowering("grouped_gemm_fp8_variable_k")


# ----------------------------------------
# Step-5: batching
# ----------------------------------------
# TODO: Add batching support if needed


__all__ = [
    "grouped_gemm_fp8_p",
    "grouped_gemm_fp8_variable_k_p",
]
