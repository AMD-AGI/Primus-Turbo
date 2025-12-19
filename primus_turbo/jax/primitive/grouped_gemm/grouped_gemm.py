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

from primus_turbo.jax._C import get_ck_grouped_gemm_workspace_size
from primus_turbo.jax.primitive import ABSTRACT_EVAL_TABLE, IMPL_TABLE, LOWERING_TABLE

# ----------------------------------------
# Step-1: Primitive Define
# ----------------------------------------
ck_grouped_gemm_p = Primitive("ck_grouped_gemm")
ck_grouped_gemm_p.multiple_results = True

ck_grouped_gemm_variable_k_p = Primitive("ck_grouped_gemm_variable_k")
ck_grouped_gemm_variable_k_p.multiple_results = True

compute_group_offs_p = Primitive("compute_group_offs")
compute_group_offs_p.multiple_results = False


# ----------------------------------------
# Step-2: Impl
# ----------------------------------------
IMPL_TABLE[ck_grouped_gemm_p] = partial(xla.apply_primitive, ck_grouped_gemm_p)
IMPL_TABLE[ck_grouped_gemm_variable_k_p] = partial(xla.apply_primitive, ck_grouped_gemm_variable_k_p)
IMPL_TABLE[compute_group_offs_p] = partial(xla.apply_primitive, compute_group_offs_p)


# ----------------------------------------
# Step-3: Abstract eval
# ----------------------------------------
def _grouped_gemm_abstract_eval(a, b, group_lens, group_offs, transA, transB, num_cu):
    assert a.dtype == b.dtype, "dtype mismatch between a and b"

    m = a.shape[1] if transA else a.shape[0]
    n = b.shape[1] if transB else b.shape[2]

    group_num = group_lens.shape[0]
    ws_size = get_ck_grouped_gemm_workspace_size(group_num)

    out_aval = ShapedArray((m, n), a.dtype)
    ws_aval = ShapedArray((ws_size,), jnp.uint8)

    return (out_aval, ws_aval)


ABSTRACT_EVAL_TABLE[ck_grouped_gemm_p] = _grouped_gemm_abstract_eval


def _compute_group_offs_abstract_eval(group_lens):
    bs = group_lens.shape[0]
    return ShapedArray((bs + 1,), group_lens.dtype)


ABSTRACT_EVAL_TABLE[compute_group_offs_p] = _compute_group_offs_abstract_eval


def _grouped_gemm_variable_k_abstract_eval(a, b, group_lens, group_offs, transA, transB, num_cu):
    assert a.dtype == b.dtype, "dtype mismatch between a and b"
    assert transA == True and transB == False, "Only transA=True, transB=False supported"

    bs = group_lens.shape[0]
    m = a.shape[1] if transA else a.shape[0]
    n = b.shape[0] if transB else b.shape[1]

    ws_size = get_ck_grouped_gemm_workspace_size(bs)

    out_aval = ShapedArray((bs, m, n), a.dtype)
    ws_aval = ShapedArray((ws_size,), jnp.uint8)

    return (out_aval, ws_aval)


ABSTRACT_EVAL_TABLE[ck_grouped_gemm_variable_k_p] = _grouped_gemm_variable_k_abstract_eval


# ----------------------------------------
# Step-4: JIT Lowering
# ----------------------------------------
LOWERING_TABLE[ck_grouped_gemm_p] = jax.ffi.ffi_lowering("ck_grouped_gemm")

LOWERING_TABLE[ck_grouped_gemm_variable_k_p] = jax.ffi.ffi_lowering("ck_grouped_gemm_variable_k")

LOWERING_TABLE[compute_group_offs_p] = jax.ffi.ffi_lowering("compute_group_offs")


# ----------------------------------------
# Step-5: batching
# ----------------------------------------
# TODO: Add batching support if needed


__all__ = [
    "ck_grouped_gemm_p",
    "ck_grouped_gemm_variable_k_p",
    "compute_group_offs_p",
]
