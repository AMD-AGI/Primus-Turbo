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

grouped_gemm_fp8_fused_tensorwise_p = Primitive("grouped_gemm_fp8_fused_tensorwise")
grouped_gemm_fp8_fused_tensorwise_p.multiple_results = False


# ----------------------------------------
# Step-2: Impl
# ----------------------------------------
IMPL_TABLE[grouped_gemm_fp8_p] = partial(xla.apply_primitive, grouped_gemm_fp8_p)
IMPL_TABLE[grouped_gemm_fp8_variable_k_p] = partial(xla.apply_primitive, grouped_gemm_fp8_variable_k_p)
IMPL_TABLE[grouped_gemm_fp8_fused_tensorwise_p] = partial(
    xla.apply_primitive, grouped_gemm_fp8_fused_tensorwise_p
)


# ----------------------------------------
# Step-3: Abstract eval
# ----------------------------------------
def _grouped_gemm_fp8_abstract_eval(
    a,
    b,
    a_scales,
    b_scales,
    group_lens,
    group_offs,
    workspace,
    *,
    transA,
    transB,
    num_cu,
    granularity,
    out_dtype_str,
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
    a,
    b,
    a_scales,
    b_scales,
    group_lens,
    group_offs,
    workspace,
    *,
    transA,
    transB,
    num_cu,
    granularity,
    out_dtype_str,
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


def _grouped_gemm_fp8_fused_tensorwise_abstract_eval(
    a_fp16,
    b_fp16,
    group_lens,
    group_offs,
    workspace,
    *,
    transA,
    transB,
    num_cu,
    fp8_dtype_str,
    out_dtype_str,
):
    """Abstract evaluation for fused grouped_gemm_fp8_tensorwise.

    Accepts BF16/FP16 inputs, quantizes them internally, then runs grouped_gemm.

    Args:
        a_fp16: Input tensor A (BF16/FP16) with shape [bs * m, k] or [k, bs * m] if transA
        b_fp16: Input tensor B (BF16/FP16) with shape [bs, k, n] or [bs, n, k] if transB
        group_lens: Group lengths tensor [bs]
        group_offs: Group offsets tensor [bs + 1]
        workspace: Workspace buffer
        transA: Whether A is transposed
        transB: Whether B is transposed
        num_cu: Number of compute units
        fp8_dtype_str: FP8 dtype string ("e4m3" or "e5m2")
        out_dtype_str: Output dtype string ("float16" or "bfloat16")

    Returns:
        Output tensor with shape [m, n]
    """
    import jax.numpy as jnp

    b_fp16.shape[0]
    if transA:
        m_total = a_fp16.shape[1]
    else:
        m_total = a_fp16.shape[0]

    # n calculation matches _grouped_gemm_fp8_abstract_eval:
    # if transB=True: b is [bs, N, K], after transpose becomes [bs, K, N], so n = b.shape[1] (original N)
    # if transB=False: b is [bs, K, N], so n = b.shape[2] (N)
    n = b_fp16.shape[1] if transB else b_fp16.shape[2]

    # Output shape: [m_total, n] (flattened across groups)
    dtype_map = {"float16": jnp.float16, "bfloat16": jnp.bfloat16}
    out_dtype = dtype_map.get(out_dtype_str, jnp.bfloat16)

    return ShapedArray((m_total, n), out_dtype)


ABSTRACT_EVAL_TABLE[grouped_gemm_fp8_fused_tensorwise_p] = _grouped_gemm_fp8_fused_tensorwise_abstract_eval


# ----------------------------------------
# Step-4: JIT Lowering
# ----------------------------------------
LOWERING_TABLE[grouped_gemm_fp8_p] = jax.ffi.ffi_lowering("grouped_gemm_fp8")

LOWERING_TABLE[grouped_gemm_fp8_variable_k_p] = jax.ffi.ffi_lowering("grouped_gemm_fp8_variable_k")

LOWERING_TABLE[grouped_gemm_fp8_fused_tensorwise_p] = jax.ffi.ffi_lowering(
    "grouped_gemm_fp8_fused_tensorwise"
)


# ----------------------------------------
# Step-5: batching
# ----------------------------------------
# TODO: Add batching support if needed


__all__ = [
    "grouped_gemm_fp8_p",
    "grouped_gemm_fp8_variable_k_p",
    "grouped_gemm_fp8_fused_tensorwise_p",
]
