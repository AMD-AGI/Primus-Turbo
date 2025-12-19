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
ck_grouped_gemm_fp8_p = Primitive("ck_grouped_gemm_fp8")
ck_grouped_gemm_fp8_p.multiple_results = True

ck_grouped_gemm_fp8_variable_k_p = Primitive("ck_grouped_gemm_fp8_variable_k")
ck_grouped_gemm_fp8_variable_k_p.multiple_results = True


# ----------------------------------------
# Step-2: Impl
# ----------------------------------------
IMPL_TABLE[ck_grouped_gemm_fp8_p] = partial(xla.apply_primitive, ck_grouped_gemm_fp8_p)
IMPL_TABLE[ck_grouped_gemm_fp8_variable_k_p] = partial(xla.apply_primitive, ck_grouped_gemm_fp8_variable_k_p)


# ----------------------------------------
# Step-3: Abstract eval
# ----------------------------------------
def _grouped_gemm_fp8_abstract_eval(
    a, b, a_scales, b_scales, group_lens, group_offs, *, transA, transB, num_cu, granularity, out_dtype
):
    m = a.shape[1] if transA else a.shape[0]
    n = b.shape[1] if transB else b.shape[2]

    group_num = group_lens.shape[0]
    ws_size = get_ck_grouped_gemm_fp8_workspace_size(group_num)

    out_aval = ShapedArray((m, n), out_dtype)
    ws_aval = ShapedArray((ws_size,), jnp.uint8)

    return (out_aval, ws_aval)


ABSTRACT_EVAL_TABLE[ck_grouped_gemm_fp8_p] = _grouped_gemm_fp8_abstract_eval


def _grouped_gemm_fp8_variable_k_abstract_eval(
    a, b, a_scales, b_scales, group_lens, group_offs, *, transA, transB, num_cu, granularity, out_dtype
):
    assert transA == True and transB == False, "Only transA=True, transB=False supported"

    bs = group_lens.shape[0]
    m = a.shape[1]
    n = b.shape[1]

    ws_size = get_ck_grouped_gemm_fp8_variable_k_workspace_size(bs)

    out_aval = ShapedArray((bs, m, n), out_dtype)
    ws_aval = ShapedArray((ws_size,), jnp.uint8)

    return (out_aval, ws_aval)


ABSTRACT_EVAL_TABLE[ck_grouped_gemm_fp8_variable_k_p] = _grouped_gemm_fp8_variable_k_abstract_eval


# ----------------------------------------
# Step-4: JIT Lowering
# ----------------------------------------
def _grouped_gemm_fp8_lowering(ctx, *args, **kwargs):
    kwargs.pop("out_dtype", None)
    return jax.ffi.ffi_lowering("ck_grouped_gemm_fp8")(ctx, *args, **kwargs)


def _grouped_gemm_fp8_variable_k_lowering(ctx, *args, **kwargs):
    kwargs.pop("out_dtype", None)
    return jax.ffi.ffi_lowering("ck_grouped_gemm_fp8_variable_k")(ctx, *args, **kwargs)


LOWERING_TABLE[ck_grouped_gemm_fp8_p] = _grouped_gemm_fp8_lowering

LOWERING_TABLE[ck_grouped_gemm_fp8_variable_k_p] = _grouped_gemm_fp8_variable_k_lowering


# ----------------------------------------
# Step-5: batching
# ----------------------------------------
# TODO: Add batching support if needed


__all__ = [
    "ck_grouped_gemm_fp8_p",
    "ck_grouped_gemm_fp8_variable_k_p",
]
