###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from functools import partial

import jax
from jax.core import Primitive, ShapedArray
from jax.interpreters import xla

from primus_turbo.jax.primitive import ABSTRACT_EVAL_TABLE, IMPL_TABLE, LOWERING_TABLE

# ----------------------------------------
# Step-1: Primitive Define
# ----------------------------------------
rmsnorm_fwd_p = Primitive("rmsnorm_fwd")
rmsnorm_fwd_p.multiple_results = False

rmsnorm_bwd_p = Primitive("rmsnorm_bwd")
rmsnorm_bwd_p.multiple_results = True


# ----------------------------------------
# Step-2: Impl
# ----------------------------------------
IMPL_TABLE[rmsnorm_fwd_p] = partial(xla.apply_primitive, rmsnorm_fwd_p)
IMPL_TABLE[rmsnorm_bwd_p] = partial(xla.apply_primitive, rmsnorm_bwd_p)


# ----------------------------------------
# Step-3: Abstract eval
# ----------------------------------------
def _fwd_abstract_eval(x, gamma, eps):
    assert x.dtype == gamma.dtype, "dtype mismatch"
    assert x.shape[-1] == gamma.shape[0], "last dim mismatch"
    return ShapedArray(x.shape, x.dtype)


ABSTRACT_EVAL_TABLE[rmsnorm_fwd_p] = _fwd_abstract_eval


def _bwd_abstract_eval(ct, x, gamma, eps):
    assert ct.shape == x.shape
    dx = ShapedArray(x.shape, x.dtype)
    # TODO: dgamma = ShapedArray(gamma.shape, gamma.dtype)
    dgamma = ShapedArray(x.shape, x.dtype)
    return dx, dgamma


ABSTRACT_EVAL_TABLE[rmsnorm_bwd_p] = _bwd_abstract_eval

# ----------------------------------------
# Step-4: JIT Lowering
# ----------------------------------------
LOWERING_TABLE[rmsnorm_fwd_p] = jax.ffi.ffi_lowering("rmsnorm_fwd")
LOWERING_TABLE[rmsnorm_bwd_p] = jax.ffi.ffi_lowering("rmsnorm_bwd")

# ----------------------------------------
# Step-5: batching
# ----------------------------------------
# TODO


__all__ = ["rmsnorm_fwd_p", "rmsnorm_bwd_p"]
