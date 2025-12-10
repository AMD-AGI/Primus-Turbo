###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from functools import lru_cache

import jax
import jax.numpy as jnp
import pytest

from primus_turbo.jax.lax.grouped_gemm import grouped_gemm
from primus_turbo.jax.lax.grouped_gemm_hipblaslt import grouped_gemm_hipblaslt
from tests.jax.ref.gemm_ref import generate_grouped_gemm_group_lens, grouped_gemm_ref
from tests.jax.test_utils import compute_snr

# Configure JAX for optimal test performance
jax.config.update("jax_enable_x64", True)  # Enable once globally
jax.config.update("jax_compilation_cache_dir", "/tmp/jax_test_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_enable_compilation_cache", True)
jax.config.update("jax_cpu_enable_async_dispatch", True)


# Cache test data generation
@lru_cache(maxsize=256)
def _get_cached_array(shape_tuple, dtype_str, seed):
    """Cache generated arrays to avoid regenerating for each test."""
    shape = tuple(shape_tuple)
    dtype = jnp.bfloat16 if dtype_str == "bfloat16" else jnp.float16
    key = jax.random.PRNGKey(seed)
    arr = jax.random.normal(key, shape, dtype=dtype)
    arr = jax.device_put(arr)
    jax.block_until_ready(arr)
    return arr


@pytest.mark.parametrize("B", [16, 32])
@pytest.mark.parametrize("M", [128, 1024, 2048])
@pytest.mark.parametrize(
    "N_K", [(2048, 1536), (2048, 1408), (2816, 2048), (3072, 5120), (5120, 1536), (4096, 7168), (7168, 2048)]
)
@pytest.mark.parametrize("dtype", [jnp.bfloat16, jnp.float16])
@pytest.mark.parametrize("balance", [False])
@pytest.mark.parametrize("trans_b", [True, False])
@pytest.mark.parametrize("reduce_num_cu", [0, 16])
def test_grouped_gemm(B, M, N_K, dtype, balance, trans_b, reduce_num_cu):
    """Test grouped GEMM forward and backward pass."""
    N, K = N_K
    group_lens = generate_grouped_gemm_group_lens(B, M, balance=balance)

    # Use cached arrays to avoid regenerating data
    dtype_str = "bfloat16" if dtype == jnp.bfloat16 else "float16"
    b_shape = (B, N, K) if trans_b else (B, K, N)

    a = _get_cached_array((B * M, K), dtype_str, seed=0)
    b = _get_cached_array(tuple(b_shape), dtype_str, seed=1)

    a_ref = a  # Using cached immutable data
    b_ref = b

    # Compute num_cu
    num_cu = -1 if reduce_num_cu == 0 else reduce_num_cu

    #######################################
    # Forward
    out = grouped_gemm(a, b, group_lens, transB=trans_b, num_cu=num_cu)

    # Reference uses float32 for faster computation
    a_ref_f32 = a_ref.astype(jnp.float32)
    b_ref_f32 = b_ref.astype(jnp.float32)
    out_ref = grouped_gemm_ref(a_ref_f32, b_ref_f32, group_lens, trans_b=trans_b)

    # Check forward results using SNR
    out_snr = compute_snr(out_ref.astype(dtype), out)
    assert out_snr > 20, f"out_snr too low: {out_snr:.2f} dB"

    def loss_fn(a, b):
        return jnp.sum(grouped_gemm(a, b, group_lens, transB=trans_b, num_cu=num_cu))

    def loss_fn_ref(a, b):
        return jnp.sum(grouped_gemm_ref(a, b, group_lens, trans_b=trans_b))

    grad_fn = jax.grad(loss_fn, argnums=(0, 1))
    grad_fn_ref = jax.grad(loss_fn_ref, argnums=(0, 1))

    grad_a, grad_b = grad_fn(a, b)
    grad_a_ref, grad_b_ref = grad_fn_ref(a_ref_f32, b_ref_f32)

    # Check gradients using SNR (convert ref gradients back to original dtype)
    a_grad_snr = compute_snr(grad_a_ref.astype(dtype), grad_a)
    assert a_grad_snr > 20, f"a_grad_snr too low: {a_grad_snr:.2f} dB"

    b_grad_snr = compute_snr(grad_b_ref.astype(dtype), grad_b)
    assert b_grad_snr > 20, f"b_grad_snr too low: {b_grad_snr:.2f} dB"


@pytest.mark.parametrize("B", [16, 32])
@pytest.mark.parametrize("M", [128, 1024, 2048])
@pytest.mark.parametrize(
    "N_K", [(2048, 1536), (2048, 1408), (2816, 2048), (3072, 5120), (5120, 1536), (4096, 7168), (7168, 2048)]
)
@pytest.mark.parametrize("dtype", [jnp.bfloat16, jnp.float16])
@pytest.mark.parametrize("balance", [False])
@pytest.mark.parametrize("trans_b", [True, False])
def test_grouped_gemm_hipblaslt(B, M, N_K, dtype, balance, trans_b):
    """Test grouped GEMM hipBLASLt forward and backward pass."""
    N, K = N_K
    group_lens = generate_grouped_gemm_group_lens(B, M, balance=balance)

    # Use cached arrays to avoid regenerating data
    dtype_str = "bfloat16" if dtype == jnp.bfloat16 else "float16"
    b_shape = (B, N, K) if trans_b else (B, K, N)

    a = _get_cached_array((B * M, K), dtype_str, seed=0)
    b = _get_cached_array(tuple(b_shape), dtype_str, seed=1)

    a_ref = a  # Using cached immutable data
    b_ref = b

    #######################################
    # Forward
    out = grouped_gemm_hipblaslt(a, b, group_lens, transB=trans_b)

    # Reference uses float32 for faster computation
    a_ref_f32 = a_ref.astype(jnp.float32)
    b_ref_f32 = b_ref.astype(jnp.float32)
    out_ref = grouped_gemm_ref(a_ref_f32, b_ref_f32, group_lens, trans_b=trans_b)

    # Check forward results using SNR
    out_snr = compute_snr(out_ref.astype(dtype), out)
    assert out_snr > 20, f"out_snr too low: {out_snr:.2f} dB"

    def loss_fn(a, b):
        return jnp.sum(grouped_gemm_hipblaslt(a, b, group_lens, transB=trans_b))

    def loss_fn_ref(a, b):
        return jnp.sum(grouped_gemm_ref(a, b, group_lens, trans_b=trans_b))

    grad_fn = jax.grad(loss_fn, argnums=(0, 1))
    grad_fn_ref = jax.grad(loss_fn_ref, argnums=(0, 1))

    grad_a, grad_b = grad_fn(a, b)
    grad_a_ref, grad_b_ref = grad_fn_ref(a_ref_f32, b_ref_f32)

    # Check gradients using SNR (convert ref gradients back to original dtype)
    a_grad_snr = compute_snr(grad_a_ref.astype(dtype), grad_a)
    assert a_grad_snr > 20, f"a_grad_snr too low: {a_grad_snr:.2f} dB"

    b_grad_snr = compute_snr(grad_b_ref.astype(dtype), grad_b)
    assert b_grad_snr > 20, f"b_grad_snr too low: {b_grad_snr:.2f} dB"


# ============================================================================
# Multi-GPU Tests for hipBLASLt grouped GEMM
# ============================================================================


def get_num_devices():
    """Get the number of available GPU devices."""
    return len(jax.devices())


@pytest.mark.skipif(get_num_devices() < 2, reason="Need at least 2 GPUs for multi-GPU test")
def test_grouped_gemm_hipblaslt_multi_gpu():
    """
    Test grouped GEMM hipBLASLt on multiple GPUs using jax.pmap.
    This test is designed to reproduce the HIP error 400 (hipErrorInvalidResourceHandle)
    that occurs when global hipBLASLt resources are initialized on one GPU but used on another.
    """
    num_devices = get_num_devices()
    print(f"\n[Multi-GPU Test] Running on {num_devices} GPUs")

    # Test parameters similar to the failing case
    B = 16  # num_experts
    M = 4096  # tokens per expert (approximate)
    N = 2048
    K = 2048
    dtype = jnp.bfloat16
    trans_b = False

    # Generate group lengths
    group_lens = generate_grouped_gemm_group_lens(B, M, balance=False)
    total_tokens = sum(group_lens)

    # Create input data
    key = jax.random.PRNGKey(42)
    key1, key2 = jax.random.split(key)

    a = jax.random.normal(key1, (total_tokens, K), dtype=dtype)
    b_shape = (B, N, K) if trans_b else (B, K, N)
    b = jax.random.normal(key2, b_shape, dtype=dtype)

    # Define the function to run on each device
    def gemm_fn(a, b):
        return grouped_gemm_hipblaslt(a, b, group_lens, transB=trans_b)

    # Run on multiple devices using pmap
    # Replicate the data across devices
    a_replicated = jnp.stack([a] * num_devices)
    b_replicated = jnp.stack([b] * num_devices)

    # pmap the function
    pmap_gemm = jax.pmap(gemm_fn)

    print(f"[Multi-GPU Test] Input shapes: a={a_replicated.shape}, b={b_replicated.shape}")
    print(f"[Multi-GPU Test] Group lens: {group_lens}")
    print(f"[Multi-GPU Test] Total tokens: {total_tokens}")

    # This should trigger the bug if running on multiple GPUs with the buggy code
    try:
        out = pmap_gemm(a_replicated, b_replicated)
        jax.block_until_ready(out)
        print(f"[Multi-GPU Test] Output shape: {out.shape}")
        print("[Multi-GPU Test] SUCCESS - No HIP error occurred")
    except Exception as e:
        print(f"[Multi-GPU Test] FAILED with error: {e}")
        raise

    # Verify correctness
    a_ref_f32 = a.astype(jnp.float32)
    b_ref_f32 = b.astype(jnp.float32)
    out_ref = grouped_gemm_ref(a_ref_f32, b_ref_f32, group_lens, trans_b=trans_b)

    # Check each device's output
    for i in range(num_devices):
        out_snr = compute_snr(out_ref.astype(dtype), out[i])
        print(f"[Multi-GPU Test] Device {i} SNR: {out_snr:.2f} dB")
        assert out_snr > 20, f"Device {i} out_snr too low: {out_snr:.2f} dB"


@pytest.mark.skipif(get_num_devices() < 2, reason="Need at least 2 GPUs for multi-GPU test")
def test_grouped_gemm_hipblaslt_multi_gpu_sequential():
    """
    Test grouped GEMM hipBLASLt by running sequentially on different GPUs.
    This simulates what happens when JAX schedules work on different devices.
    """
    num_devices = get_num_devices()
    print(f"\n[Multi-GPU Sequential Test] Running on {num_devices} GPUs")

    # Test parameters
    B = 16  # num_experts
    M = 2048
    N = 2048
    K = 2048
    dtype = jnp.bfloat16
    trans_b = True

    # Generate group lengths
    group_lens = generate_grouped_gemm_group_lens(B, M, balance=False)
    total_tokens = sum(group_lens)

    devices = jax.devices()[:num_devices]

    results = []
    for i, device in enumerate(devices):
        print(f"[Multi-GPU Sequential Test] Running on device {i}: {device}")

        # Create data on this specific device
        key = jax.random.PRNGKey(i * 100)
        key1, key2 = jax.random.split(key)

        with jax.default_device(device):
            a = jax.random.normal(key1, (total_tokens, K), dtype=dtype)
            b_shape = (B, N, K) if trans_b else (B, K, N)
            b = jax.random.normal(key2, b_shape, dtype=dtype)

            # Run grouped GEMM on this device
            try:
                out = grouped_gemm_hipblaslt(a, b, group_lens, transB=trans_b)
                jax.block_until_ready(out)
                print(f"[Multi-GPU Sequential Test] Device {i} output shape: {out.shape}")
                results.append((True, out, a, b))
            except Exception as e:
                print(f"[Multi-GPU Sequential Test] Device {i} FAILED: {e}")
                results.append((False, str(e), None, None))

    # Check results
    all_passed = True
    for i, (success, out, a, b) in enumerate(results):
        if success:
            a_ref_f32 = a.astype(jnp.float32)
            b_ref_f32 = b.astype(jnp.float32)
            out_ref = grouped_gemm_ref(a_ref_f32, b_ref_f32, group_lens, trans_b=trans_b)
            out_snr = compute_snr(out_ref.astype(dtype), out)
            print(f"[Multi-GPU Sequential Test] Device {i} SNR: {out_snr:.2f} dB")
            assert out_snr > 20, f"Device {i} out_snr too low: {out_snr:.2f} dB"
        else:
            all_passed = False
            print(f"[Multi-GPU Sequential Test] Device {i} failed: {out}")

    assert all_passed, "Some devices failed"
    print("[Multi-GPU Sequential Test] SUCCESS - All devices passed")
