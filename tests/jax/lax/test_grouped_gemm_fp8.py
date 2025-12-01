###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import gc

import jax
import jax.numpy as jnp
import pytest

from primus_turbo.jax.core.low_precision import (
    Float8QuantConfig,
    Format,
    ScalingGranularity,
)
from primus_turbo.jax.lax import grouped_gemm_fp8
from primus_turbo.jax.lax.grouped_gemm import clear_gemm_workspace_cache
from primus_turbo.jax.lax.grouped_gemm_fp8 import clear_workspace_cache
from primus_turbo.jax.lax.quantization import clear_quantize_workspace_cache
from tests.jax.ref.gemm_ref import generate_grouped_gemm_group_lens, grouped_gemm_ref
from tests.jax.test_utils import compute_snr

# Configure JAX for optimal test performance
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_compilation_cache_dir", "/tmp/jax_test_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_enable_compilation_cache", True)
jax.config.update("jax_cpu_enable_async_dispatch", True)


@pytest.fixture(autouse=True)
def cleanup_memory():
    """Clean up GPU memory after each test to prevent OOM."""
    yield
    # Clear workspace caches to free GPU memory
    clear_workspace_cache()
    clear_gemm_workspace_cache()
    clear_quantize_workspace_cache()
    # Clear JAX caches and run garbage collection
    jax.clear_caches()
    gc.collect()


# Generate test arrays (no caching to avoid OOM with many test combinations)
def _get_cached_array(shape_tuple, dtype_str, seed):
    """Generate arrays for test (no caching to prevent OOM)."""
    shape = tuple(shape_tuple)
    dtype = jnp.bfloat16 if dtype_str == "bfloat16" else jnp.float16
    key = jax.random.PRNGKey(seed)
    arr = jax.random.normal(key, shape, dtype=jnp.float32).astype(dtype)
    arr = jax.device_put(arr)
    jax.block_until_ready(arr)
    return arr


def _check_hit_int32_limit(B, M, N, K):
    a_elems = B * M * K
    b_elems = B * N * K
    out_elems = B * M * N
    return max(a_elems, out_elems, b_elems) >= 2**31


@pytest.mark.parametrize("B", [16, 32])
@pytest.mark.parametrize("M", [128, 1024, 4096])
@pytest.mark.parametrize(
    "NK",
    [
        (2048, 1408),
        (1408, 2048),
        (2816, 2048),
        (3072, 5120),
        (5120, 1536),
        (4096, 7168),
    ],
)
@pytest.mark.parametrize("ori_dtype", [jnp.bfloat16, jnp.float16])
@pytest.mark.parametrize("format", [Format.E4M3, Format.E5M2])
@pytest.mark.parametrize("granularity", [ScalingGranularity.TENSORWISE, ScalingGranularity.ROWWISE])
@pytest.mark.parametrize("trans_b", [True, False])
@pytest.mark.parametrize("balance", [False])
def test_grouped_gemm_fp8(B, M, NK, ori_dtype, format, granularity, trans_b, balance):

    N, K = NK

    if _check_hit_int32_limit(B, M, N, K):
        pytest.skip("Shape hits int32 indexing limit (numel >= 2**31).")

    group_lens = generate_grouped_gemm_group_lens(B, M, balance=balance)
    print(
        f"\nB={B}, M={M}, N={N}, K={K}, ori_dtype={ori_dtype}, format={format}, "
        f"granularity={granularity}, trans_b={trans_b}, balance={balance}"
    )

    # Use cached arrays to avoid regenerating data
    dtype_str = "bfloat16" if ori_dtype == jnp.bfloat16 else "float16"
    b_shape = (B, N, K) if trans_b else (B, K, N)

    a = _get_cached_array((B * M, K), dtype_str, seed=0)
    b = _get_cached_array(tuple(b_shape), dtype_str, seed=1)

    # Using cached immutable data
    a_ref = a
    b_ref = b

    # Ref (using float32 for numerical stability)
    a_ref_f32 = a_ref.astype(jnp.float32)
    b_ref_f32 = b_ref.astype(jnp.float32)
    out_ref = grouped_gemm_ref(a_ref_f32, b_ref_f32, group_lens, trans_b)
    print(out_ref.shape, out_ref.dtype)

    # Ref backward
    def loss_fn_ref(a, b):
        return jnp.sum(grouped_gemm_ref(a, b, group_lens, trans_b))

    a_ref_grad, b_ref_grad = jax.grad(loss_fn_ref, argnums=(0, 1))(a_ref_f32, b_ref_f32)
    print(a_ref_grad.shape, b_ref_grad.shape)

    # Turbo forward
    config = Float8QuantConfig(format=format, granularity=granularity)
    out = grouped_gemm_fp8(a, b, group_lens, trans_b=trans_b, config=config)

    # Turbo backward
    def loss_fn(a, b):
        return jnp.sum(grouped_gemm_fp8(a, b, group_lens, trans_b=trans_b, config=config))

    a_grad, b_grad = jax.grad(loss_fn, argnums=(0, 1))(a, b)

    # Validation
    snr_threshold = 25 if format == Format.E4M3 else 20

    out_snr = compute_snr(out_ref.astype(jnp.float32), out.astype(jnp.float32))
    print(f"Out-SNR: {out_snr:.2f} dB")
    assert out_snr > snr_threshold, f"out_snr too low: {out_snr:.2f} dB"

    a_grad_snr = compute_snr(a_ref_grad.astype(jnp.float32), a_grad.astype(jnp.float32))
    print(f"AGrad-SNR: {a_grad_snr:.2f} dB")
    assert a_grad_snr > snr_threshold, f"a_grad_snr too low: {a_grad_snr:.2f} dB"

    b_grad_snr = compute_snr(b_ref_grad.astype(jnp.float32), b_grad.astype(jnp.float32))
    print(f"BGrad-SNR: {b_grad_snr:.2f} dB")
    assert b_grad_snr > snr_threshold, f"b_grad_snr too low: {b_grad_snr:.2f} dB"
