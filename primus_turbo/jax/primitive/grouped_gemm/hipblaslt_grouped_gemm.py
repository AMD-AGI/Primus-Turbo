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

from primus_turbo.jax._C import get_hipblaslt_grouped_gemm_workspace_size
from primus_turbo.jax.primitive import ABSTRACT_EVAL_TABLE, IMPL_TABLE, LOWERING_TABLE

# One hipBLASLt grouped GEMM launch; per-group sizes are read on the device. group_lens may be int32 or int64.
hipblaslt_grouped_gemm_p = Primitive("hipblaslt_grouped_gemm")
hipblaslt_grouped_gemm_p.multiple_results = True

IMPL_TABLE[hipblaslt_grouped_gemm_p] = partial(xla.apply_primitive, hipblaslt_grouped_gemm_p)


def _hipblaslt_grouped_gemm_abstract_eval(a, b, group_lens, transA, transB):
    assert a.dtype == b.dtype, "dtype mismatch between a and b"
    assert not (transA and transB), "transA and transB together are not supported"
    if transA:
        # a [M, K], b [M, N] -> [G, K, N]
        out_aval = ShapedArray((group_lens.shape[0], a.shape[1], b.shape[1]), a.dtype)
    else:
        # a [M, K], b [G, K, N] (or [G, N, K] if transB) -> [M, N]
        n = b.shape[1] if transB else b.shape[2]
        out_aval = ShapedArray((a.shape[0], n), a.dtype)
    # Device-side per-group hipBLASLt user arguments.
    ws_aval = ShapedArray((get_hipblaslt_grouped_gemm_workspace_size(group_lens.shape[0]),), jnp.uint8)
    return (out_aval, ws_aval)


ABSTRACT_EVAL_TABLE[hipblaslt_grouped_gemm_p] = _hipblaslt_grouped_gemm_abstract_eval

LOWERING_TABLE[hipblaslt_grouped_gemm_p] = jax.ffi.ffi_lowering("hipblaslt_grouped_gemm")

__all__ = ["hipblaslt_grouped_gemm_p"]
