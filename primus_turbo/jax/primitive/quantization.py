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

from primus_turbo.jax.primitive import ABSTRACT_EVAL_TABLE, IMPL_TABLE, LOWERING_TABLE

__all__ = [
    "quantize_fp8_tensorwise_p",
    "dequantize_fp8_tensorwise_p",
    "quantize_fp8_rowwise_p",
]

# ============================================================
# Tensorwise Quantize FP8 Primitive
# ============================================================

quantize_fp8_tensorwise_p = Primitive("quantize_fp8_tensorwise")
quantize_fp8_tensorwise_p.multiple_results = True

# Impl
IMPL_TABLE[quantize_fp8_tensorwise_p] = partial(xla.apply_primitive, quantize_fp8_tensorwise_p)


# Abstract eval
# Matches PyTorch signature: (input, scale_opt) + out_dtype_str (static) -> (output, scale_inv)
def _quantize_fp8_tensorwise_abstract_eval(input_aval, scale_opt_aval, *, out_dtype_str):
    # Convert string to dtype
    dtype_map = {
        "float8_e4m3fn": jnp.float8_e4m3fn,
        "float8_e4m3fnuz": jnp.float8_e4m3fnuz,
        "float8_e5m2": jnp.float8_e5m2,
        "float8_e5m2fnuz": jnp.float8_e5m2fnuz,
    }
    out_dtype = dtype_map.get(out_dtype_str)
    if out_dtype is None:
        raise ValueError(f"Unsupported out_dtype_str: {out_dtype_str}")

    # output: same shape as input but with FP8 dtype
    # scale_inv: scalar float32
    return (ShapedArray(input_aval.shape, out_dtype), ShapedArray((1,), jnp.float32))


quantize_fp8_tensorwise_p.def_abstract_eval(_quantize_fp8_tensorwise_abstract_eval)
ABSTRACT_EVAL_TABLE[quantize_fp8_tensorwise_p] = _quantize_fp8_tensorwise_abstract_eval

# Lowering
LOWERING_TABLE[quantize_fp8_tensorwise_p] = jax.ffi.ffi_lowering("quantize_fp8_tensorwise")

# ============================================================
# Tensorwise Dequantize FP8 Primitive
# ============================================================

dequantize_fp8_tensorwise_p = Primitive("dequantize_fp8_tensorwise")
dequantize_fp8_tensorwise_p.multiple_results = False

# Impl
IMPL_TABLE[dequantize_fp8_tensorwise_p] = partial(xla.apply_primitive, dequantize_fp8_tensorwise_p)


# Abstract eval
# Matches PyTorch signature: (input, scale_inv) + out_dtype_str (static) -> output
def _dequantize_fp8_tensorwise_abstract_eval(input_aval, scale_inv_aval, *, out_dtype_str):
    # output: same shape as input but with float32 dtype
    return ShapedArray(input_aval.shape, jnp.float32)


dequantize_fp8_tensorwise_p.def_abstract_eval(_dequantize_fp8_tensorwise_abstract_eval)
ABSTRACT_EVAL_TABLE[dequantize_fp8_tensorwise_p] = _dequantize_fp8_tensorwise_abstract_eval

# Lowering
LOWERING_TABLE[dequantize_fp8_tensorwise_p] = jax.ffi.ffi_lowering("dequantize_fp8_tensorwise")

# ============================================================
# Rowwise Quantize FP8 Primitive
# ============================================================

quantize_fp8_rowwise_p = Primitive("quantize_fp8_rowwise")
quantize_fp8_rowwise_p.multiple_results = True

# Impl
IMPL_TABLE[quantize_fp8_rowwise_p] = partial(xla.apply_primitive, quantize_fp8_rowwise_p)


# Abstract eval
# Matches PyTorch signature: (input, scale_opt) + out_dtype_str, axis (static) -> (output, scale_inv)
def _quantize_fp8_rowwise_abstract_eval(input_aval, scale_opt_aval, *, out_dtype_str, axis):
    # Convert string to dtype
    dtype_map = {
        "float8_e4m3fn": jnp.float8_e4m3fn,
        "float8_e4m3fnuz": jnp.float8_e4m3fnuz,
        "float8_e5m2": jnp.float8_e5m2,
        "float8_e5m2fnuz": jnp.float8_e5m2fnuz,
    }
    out_dtype = dtype_map.get(out_dtype_str)
    if out_dtype is None:
        raise ValueError(f"Unsupported out_dtype_str: {out_dtype_str}")

    # output: same shape as input but with FP8 dtype
    # scale_inv: same shape as input but with axis dimension = 1
    # E.g., for input [M, K] and axis=-1, scale_inv is [M, 1]
    scale_inv_shape = list(input_aval.shape)
    scale_inv_shape[axis] = 1

    return (ShapedArray(input_aval.shape, out_dtype), ShapedArray(tuple(scale_inv_shape), jnp.float32))


quantize_fp8_rowwise_p.def_abstract_eval(_quantize_fp8_rowwise_abstract_eval)
ABSTRACT_EVAL_TABLE[quantize_fp8_rowwise_p] = _quantize_fp8_rowwise_abstract_eval

# Lowering
LOWERING_TABLE[quantize_fp8_rowwise_p] = jax.ffi.ffi_lowering("quantize_fp8_rowwise")
