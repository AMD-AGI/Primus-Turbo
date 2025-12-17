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

from primus_turbo.jax._C import get_quantize_fp8_tensorwise_workspace_size
from primus_turbo.jax.primitive import ABSTRACT_EVAL_TABLE, IMPL_TABLE, LOWERING_TABLE

__all__ = [
    "quantize_fp8_tensorwise_p",
    "dequantize_fp8_tensorwise_p",
    "quantize_fp8_rowwise_p",
]

# ----------------------------------------
# Step-1: Primitive Define
# ----------------------------------------
quantize_fp8_tensorwise_p = Primitive("quantize_fp8_tensorwise")
quantize_fp8_tensorwise_p.multiple_results = True

dequantize_fp8_tensorwise_p = Primitive("dequantize_fp8_tensorwise")
dequantize_fp8_tensorwise_p.multiple_results = False

quantize_fp8_rowwise_p = Primitive("quantize_fp8_rowwise")
quantize_fp8_rowwise_p.multiple_results = True

# ----------------------------------------
# Step-2: Impl
# ----------------------------------------
IMPL_TABLE[quantize_fp8_tensorwise_p] = partial(xla.apply_primitive, quantize_fp8_tensorwise_p)
IMPL_TABLE[dequantize_fp8_tensorwise_p] = partial(xla.apply_primitive, dequantize_fp8_tensorwise_p)
IMPL_TABLE[quantize_fp8_rowwise_p] = partial(xla.apply_primitive, quantize_fp8_rowwise_p)


# ----------------------------------------
# Step-3: Abstract eval
# ----------------------------------------
def _quantize_fp8_tensorwise_abstract_eval(input_aval, scale_opt_aval, *, out_dtype):
    n = 1
    for dim in input_aval.shape:
        n *= dim
    ws_size = get_quantize_fp8_tensorwise_workspace_size(n)
    return (
        ShapedArray(input_aval.shape, out_dtype),
        ShapedArray((1,), jnp.float32),
        ShapedArray((ws_size,), jnp.uint8),
    )


ABSTRACT_EVAL_TABLE[quantize_fp8_tensorwise_p] = _quantize_fp8_tensorwise_abstract_eval


def _dequantize_fp8_tensorwise_abstract_eval(input_aval, scale_inv_aval, *, out_dtype):
    return ShapedArray(input_aval.shape, out_dtype)


ABSTRACT_EVAL_TABLE[dequantize_fp8_tensorwise_p] = _dequantize_fp8_tensorwise_abstract_eval


def _quantize_fp8_rowwise_abstract_eval(input_aval, scale_opt_aval, *, out_dtype, axis):
    scale_inv_shape = list(input_aval.shape)
    scale_inv_shape[axis] = 1
    return (ShapedArray(input_aval.shape, out_dtype), ShapedArray(tuple(scale_inv_shape), jnp.float32))


ABSTRACT_EVAL_TABLE[quantize_fp8_rowwise_p] = _quantize_fp8_rowwise_abstract_eval


# ----------------------------------------
# Step-4: JIT Lowering
# ----------------------------------------
def _quantize_fp8_tensorwise_lowering(ctx, *args, **kwargs):
    kwargs.pop("out_dtype", None)
    return jax.ffi.ffi_lowering("quantize_fp8_tensorwise")(ctx, *args, **kwargs)


def _dequantize_fp8_tensorwise_lowering(ctx, *args, **kwargs):
    kwargs.pop("out_dtype", None)
    return jax.ffi.ffi_lowering("dequantize_fp8_tensorwise")(ctx, *args, **kwargs)


def _quantize_fp8_rowwise_lowering(ctx, *args, **kwargs):
    kwargs.pop("out_dtype", None)
    return jax.ffi.ffi_lowering("quantize_fp8_rowwise")(ctx, *args, **kwargs)


LOWERING_TABLE[quantize_fp8_tensorwise_p] = _quantize_fp8_tensorwise_lowering
LOWERING_TABLE[dequantize_fp8_tensorwise_p] = _dequantize_fp8_tensorwise_lowering
LOWERING_TABLE[quantize_fp8_rowwise_p] = _quantize_fp8_rowwise_lowering

# ----------------------------------------
# Step-5: batching
# ----------------------------------------
# TODO
