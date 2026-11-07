###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
# See LICENSE for license information.
###############################################################################

"""Public errors and abstract shape contracts without launching invalid kernels."""

import jax
import jax.numpy as jnp
import pytest

from primus_turbo.jax.core.low_precision import Float8QuantConfig, ScalingGranularity, float8_e4m3
from primus_turbo.jax.lax import dequantize_fp8, grouped_gemm_fp8, quantize_fp8, rmsnorm
from primus_turbo.jax.primitive._batching import enable_x64


@pytest.mark.parametrize(
    "granularity,axis,error,match",
    [
        (ScalingGranularity.ROWWISE, None, ValueError, "axis must be specified"),
        (ScalingGranularity.BLOCKWISE, None, NotImplementedError, "Unknown granularity"),
        (ScalingGranularity.MX_BLOCKWISE, None, NotImplementedError, "Unknown granularity"),
    ],
)
def test_quantize_rejects_unsupported(granularity, axis, error, match):
    with pytest.raises(error, match=match):
        quantize_fp8(jnp.ones((2, 4), jnp.bfloat16), float8_e4m3, granularity, axis=axis)


@pytest.mark.parametrize(
    "granularity,axis,error,match",
    [
        (ScalingGranularity.ROWWISE, None, ValueError, "axis must be specified"),
        (ScalingGranularity.ROWWISE, -1, NotImplementedError, "Rowwise dequantization"),
        (ScalingGranularity.BLOCKWISE, None, NotImplementedError, "Unknown granularity"),
    ],
)
def test_dequantize_rejects_unsupported(granularity, axis, error, match):
    with pytest.raises(error, match=match):
        dequantize_fp8(
            jnp.ones((2, 4), float8_e4m3),
            jnp.bfloat16,
            granularity,
            axis=axis,
            scale_inv=jnp.ones((1,), jnp.float32),
        )


@pytest.mark.parametrize(
    "granularity,error",
    [(ScalingGranularity.BLOCKWISE, NotImplementedError), (ScalingGranularity.MX_BLOCKWISE, ValueError)],
)
def test_grouped_fp8_unimplemented_granularity(granularity, error):
    a, b = jnp.ones((128, 128), jnp.bfloat16), jnp.ones((1, 128, 128), jnp.bfloat16)
    with enable_x64(True):
        lengths, offsets = jnp.array([128], jnp.int64), jnp.array([0, 128], jnp.int64)
        config = Float8QuantConfig(granularity=granularity, block_size=32)
        with pytest.raises(error, match="BLOCKWISE|Unsupported FP8"):
            grouped_gemm_fp8(a, b, lengths, offsets, config=config)
        if granularity == ScalingGranularity.BLOCKWISE:
            with pytest.raises(NotImplementedError, match="BLOCKWISE"):
                jax.grad(lambda x: grouped_gemm_fp8(x, b, lengths, offsets, config=config).sum())(a)


@pytest.mark.parametrize("a_dtype,b_dtype", [(jnp.float32, jnp.bfloat16), (jnp.bfloat16, jnp.float32)])
def test_grouped_fp8_dtype_validation(a_dtype, b_dtype):
    with pytest.raises(AssertionError, match="Unsupported dtype"):
        grouped_gemm_fp8(jnp.ones((2, 4), a_dtype), jnp.ones((1, 4, 4), b_dtype), jnp.array([2], jnp.int32))


@pytest.mark.parametrize(
    "gamma,error",
    [
        (jax.ShapeDtypeStruct((8,), jnp.float16), "dtype mismatch"),
        (jax.ShapeDtypeStruct((4,), jnp.float32), "last dim mismatch"),
    ],
)
def test_rmsnorm_abstract_errors(gamma, error):
    with pytest.raises(AssertionError, match=error):
        jax.eval_shape(lambda x, g: rmsnorm(x, g, 1e-6), jax.ShapeDtypeStruct((2, 8), jnp.float32), gamma)
