###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import jax
import jax.numpy as jnp
import pytest

from primus_turbo.jax.core.low_precision import (
    ScalingGranularity,
    float8_e4m3,
    float8_e5m2,
)
from primus_turbo.jax.lax.quantization import dequantize_fp8, quantize_fp8
from tests.jax.ref.quantization_ref import dequantize_fp8_ref, quantize_fp8_ref
from tests.jax.test_utils import assert_allclose, get_tolerances


@pytest.mark.parametrize("orig_dtype", [jnp.bfloat16, jnp.float16, jnp.float32])
@pytest.mark.parametrize("dest_dtype", [float8_e4m3, float8_e5m2])
@pytest.mark.parametrize("numel", [6 * 1 * 7168 * 8192])
@pytest.mark.parametrize("dynamic_quantize", [True, False])
@pytest.mark.parametrize("use_jit", [True, False])
@pytest.mark.parametrize("granularity", [ScalingGranularity.TENSORWISE])
def test_quantize_fp8_tensorwise(orig_dtype, dest_dtype, numel, dynamic_quantize, use_jit, granularity):
    key = jax.random.PRNGKey(42)

    x = jax.random.uniform(key, (numel,), dtype=orig_dtype)
    x_ref = jnp.array(x)
    x_fp8_ref, x_scale_ref, x_scale_inv_ref = quantize_fp8_ref(x_ref, dest_dtype, granularity)

    # Quantize
    scale = None
    if dynamic_quantize == False:
        scale = jnp.array(x_scale_ref)

    if use_jit is True:
        quantize_fn = jax.jit(lambda t: quantize_fp8(t, dest_dtype, granularity=granularity, scale=scale))
        x_fp8, x_scale_inv = quantize_fn(x)
    else:
        x_fp8, x_scale_inv = quantize_fp8(x, dest_dtype, granularity=granularity, scale=scale)

    assert_allclose(x_scale_inv_ref, x_scale_inv, **get_tolerances(jnp.float32))
    assert_allclose(
        x_fp8_ref.astype(jnp.float32) * x_scale_inv_ref,
        x_fp8.astype(jnp.float32) * x_scale_inv,
        **get_tolerances(dest_dtype)
    )

    # DeQuantize
    x_dq = dequantize_fp8(x_fp8, orig_dtype, granularity, scale_inv=x_scale_inv)
    x_dq_ref = dequantize_fp8_ref(x_fp8_ref, orig_dtype, granularity, scale_inv=x_scale_inv_ref)
    assert_allclose(x_dq, x_dq_ref, **get_tolerances(dest_dtype))


@pytest.mark.parametrize("orig_dtype", [jnp.bfloat16, jnp.float16, jnp.float32])
@pytest.mark.parametrize("dest_dtype", [float8_e4m3, float8_e5m2])
@pytest.mark.parametrize("axis", [-1, -2, -3, 0, 1, 2])
@pytest.mark.parametrize("B", [1, 4])
@pytest.mark.parametrize("M", [1, 111, 7168])
@pytest.mark.parametrize("N", [1, 111, 4096])
@pytest.mark.parametrize("dynamic_quantize", [True, False])
@pytest.mark.parametrize("use_jit", [True, False])
@pytest.mark.parametrize("granularity", [ScalingGranularity.ROWWISE])
def test_quantize_fp8_rowwise(orig_dtype, dest_dtype, axis, B, M, N, dynamic_quantize, use_jit, granularity):
    key = jax.random.PRNGKey(42)

    x = jax.random.uniform(key, (B, M, N), dtype=orig_dtype)
    x_ref = jnp.array(x)
    x_fp8_ref, x_scale_ref, x_scale_inv_ref = quantize_fp8_ref(x_ref, dest_dtype, granularity, axis)

    scale = None
    if dynamic_quantize == False:
        scale = jnp.array(x_scale_ref)

    if use_jit is True:
        quantize_fn = jax.jit(
            lambda t: quantize_fp8(t, dest_dtype, granularity=granularity, axis=axis, scale=scale)
        )
        x_fp8, x_scale_inv = quantize_fn(x)
    else:
        x_fp8, x_scale_inv = quantize_fp8(x, dest_dtype, granularity=granularity, axis=axis, scale=scale)

    assert_allclose(x_scale_inv_ref, x_scale_inv, **get_tolerances(jnp.float32))
    assert_allclose(
        x_fp8_ref.astype(jnp.float32) * x_scale_inv_ref,
        x_fp8.astype(jnp.float32) * x_scale_inv,
        **get_tolerances(dest_dtype)
    )
