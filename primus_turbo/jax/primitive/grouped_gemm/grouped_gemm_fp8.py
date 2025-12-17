###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from functools import partial

import jax
import jax.numpy as jnp
from jax.core import ShapedArray
from jax.extend.core import Primitive
from jax.interpreters import xla

from primus_turbo.jax._C import (
    get_ck_grouped_gemm_fp8_variable_k_workspace_size,
    get_ck_grouped_gemm_fp8_workspace_size,
)
from primus_turbo.jax.primitive import ABSTRACT_EVAL_TABLE, IMPL_TABLE, LOWERING_TABLE

# ----------------------------------------
# Step-1: Primitive Define
# ----------------------------------------
grouped_gemm_fp8_p = Primitive("grouped_gemm_fp8")
grouped_gemm_fp8_p.multiple_results = True

grouped_gemm_fp8_variable_k_p = Primitive("grouped_gemm_fp8_variable_k")
grouped_gemm_fp8_variable_k_p.multiple_results = True


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
    m = a.shape[1] if transA else a.shape[0]
    n = b.shape[1] if transB else b.shape[2]

    dtype_map = {"float16": jnp.float16, "bfloat16": jnp.bfloat16}
    out_dtype = dtype_map.get(out_dtype_str, jnp.bfloat16)

    group_num = group_lens.shape[0]
    ws_size = get_ck_grouped_gemm_fp8_workspace_size(group_num)

    out_aval = ShapedArray((m, n), out_dtype)
    ws_aval = ShapedArray((ws_size,), jnp.uint8)

    return (out_aval, ws_aval)


ABSTRACT_EVAL_TABLE[grouped_gemm_fp8_p] = _grouped_gemm_fp8_abstract_eval


def _grouped_gemm_fp8_variable_k_abstract_eval(
    a, b, a_scales, b_scales, group_lens, group_offs, *, transA, transB, num_cu, granularity, out_dtype_str
):
    assert transA == True and transB == False, "Only transA=True, transB=False supported"

    bs = group_lens.shape[0]
    m = a.shape[1]
    n = b.shape[1]

    dtype_map = {"float16": jnp.float16, "bfloat16": jnp.bfloat16}
    out_dtype = dtype_map.get(out_dtype_str, jnp.bfloat16)

    ws_size = get_ck_grouped_gemm_fp8_variable_k_workspace_size(bs, m, n)

    out_aval = ShapedArray((bs, m, n), out_dtype)
    ws_aval = ShapedArray((ws_size,), jnp.uint8)

    return (out_aval, ws_aval)


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
