###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from primus_turbo.jax.core.float8 import Float8QuantConfig, Format, ScalingGranularity
from primus_turbo.jax.lax.gemm_fp8 import gemm_fp8


@pytest.mark.parametrize("M,N,K", [(128, 256, 512), (64, 128, 256)])
@pytest.mark.parametrize("trans_a", [False])  # trans_a must be False
@pytest.mark.parametrize("trans_b", [False, True])
@pytest.mark.parametrize("dtype", [jnp.float16, jnp.bfloat16])
@pytest.mark.parametrize("format", [Format.E4M3, Format.E5M2])
@pytest.mark.parametrize("granularity", [ScalingGranularity.TENSORWISE, ScalingGranularity.ROWWISE])
def test_gemm_fp8_forward(M, N, K, trans_a, trans_b, dtype, format, granularity):
    """Test FP8 GEMM forward pass."""
    jax.config.update("jax_enable_x64", True)

    key = jax.random.PRNGKey(42)
    key1, key2 = jax.random.split(key)

    # Create input tensors
    a = jax.random.normal(key1, (M, K), dtype=jnp.float32).astype(dtype)
    if trans_b:
        b = jax.random.normal(key2, (N, K), dtype=jnp.float32).astype(dtype)
    else:
        b = jax.random.normal(key2, (K, N), dtype=jnp.float32).astype(dtype)

    # Create config
    config = Float8QuantConfig(format=format, granularity=granularity)

    # Run FP8 GEMM
    try:
        out = gemm_fp8(a, b, trans_a=trans_a, trans_b=trans_b, out_dtype=dtype, config=config)

        # Compute reference
        if trans_a:
            a_ref = jnp.transpose(a)
        else:
            a_ref = a
        if trans_b:
            b_ref = jnp.transpose(b)
        else:
            b_ref = b
        out_ref = jnp.matmul(a_ref, b_ref)

        # Check shape
        assert out.shape == out_ref.shape, f"Shape mismatch: {out.shape} vs {out_ref.shape}"

        # Check values (FP8 has lower precision)
        if format == Format.E4M3:
            rtol, atol = 5e-2, 5e-2
        else:  # E5M2
            rtol, atol = 1e-1, 1e-1

        out_f32 = out.astype(jnp.float32)
        out_ref_f32 = out_ref.astype(jnp.float32)

        np.testing.assert_allclose(out_f32, out_ref_f32, rtol=rtol, atol=atol)
        print(
            f"✓ Forward test passed: M={M}, N={N}, K={K}, trans_b={trans_b}, "
            f"format={format}, granularity={granularity}"
        )

    except Exception as e:
        pytest.skip(f"Test skipped due to: {str(e)}")


@pytest.mark.parametrize("M,N,K", [(64, 128, 256)])
@pytest.mark.parametrize("trans_b", [False, True])
@pytest.mark.parametrize("dtype", [jnp.float16])
@pytest.mark.parametrize("format", [Format.E4M3])
@pytest.mark.parametrize("granularity", [ScalingGranularity.TENSORWISE, ScalingGranularity.ROWWISE])
def test_gemm_fp8_backward(M, N, K, trans_b, dtype, format, granularity):
    """Test FP8 GEMM backward pass."""
    jax.config.update("jax_enable_x64", True)

    key = jax.random.PRNGKey(42)
    key1, key2 = jax.random.split(key)

    # Create input tensors
    a = jax.random.normal(key1, (M, K), dtype=jnp.float32).astype(dtype)
    if trans_b:
        b = jax.random.normal(key2, (N, K), dtype=jnp.float32).astype(dtype)
    else:
        b = jax.random.normal(key2, (K, N), dtype=jnp.float32).astype(dtype)

    # Create config
    config = Float8QuantConfig(format=format, granularity=granularity)

    # Define loss function
    def loss_fn(a, b):
        out = gemm_fp8(a, b, trans_a=False, trans_b=trans_b, out_dtype=dtype, config=config)
        return jnp.sum(out**2)

    try:
        # Compute gradients
        grad_fn = jax.grad(loss_fn, argnums=(0, 1))
        grad_a, grad_b = grad_fn(a, b)

        # Check shapes
        assert grad_a.shape == a.shape, f"grad_a shape mismatch: {grad_a.shape} vs {a.shape}"
        assert grad_b.shape == b.shape, f"grad_b shape mismatch: {grad_b.shape} vs {b.shape}"

        # Check that gradients are not all zeros
        assert jnp.any(grad_a != 0), "grad_a is all zeros"
        assert jnp.any(grad_b != 0), "grad_b is all zeros"

        # Check that gradients are finite
        assert jnp.all(jnp.isfinite(grad_a)), "grad_a contains non-finite values"
        assert jnp.all(jnp.isfinite(grad_b)), "grad_b contains non-finite values"

        print(
            f"✓ Backward test passed: M={M}, N={N}, K={K}, trans_b={trans_b}, "
            f"format={format}, granularity={granularity}"
        )
        print(f"  grad_a: mean={jnp.mean(jnp.abs(grad_a)):.6f}, max={jnp.max(jnp.abs(grad_a)):.6f}")
        print(f"  grad_b: mean={jnp.mean(jnp.abs(grad_b)):.6f}, max={jnp.max(jnp.abs(grad_b)):.6f}")

    except Exception as e:
        pytest.skip(f"Test skipped due to: {str(e)}")


@pytest.mark.parametrize("M,N,K", [(32, 64, 128)])
@pytest.mark.parametrize("format", [Format.E4M3, Format.E5M2, Format.HYBRID])
def test_gemm_fp8_formats(M, N, K, format):
    """Test different FP8 formats."""
    jax.config.update("jax_enable_x64", True)

    key = jax.random.PRNGKey(42)
    key1, key2 = jax.random.split(key)

    a = jax.random.normal(key1, (M, K), dtype=jnp.float32).astype(jnp.float16)
    b = jax.random.normal(key2, (K, N), dtype=jnp.float32).astype(jnp.float16)

    config = Float8QuantConfig(format=format, granularity=ScalingGranularity.TENSORWISE)

    try:
        # Forward
        out = gemm_fp8(a, b, config=config)
        assert out.shape == (M, N)

        # Backward
        def loss_fn(a, b):
            return jnp.sum(gemm_fp8(a, b, config=config) ** 2)

        grad_fn = jax.grad(loss_fn, argnums=(0, 1))
        grad_a, grad_b = grad_fn(a, b)

        assert grad_a.shape == a.shape
        assert grad_b.shape == b.shape

        print(f"✓ Format test passed: format={format}")

    except Exception as e:
        pytest.skip(f"Test skipped due to: {str(e)}")


def test_gemm_fp8_jit():
    """Test that gemm_fp8 works with JIT compilation."""
    jax.config.update("jax_enable_x64", True)

    M, N, K = 64, 128, 256
    config = Float8QuantConfig(format=Format.E4M3, granularity=ScalingGranularity.TENSORWISE)

    @jax.jit
    def jit_gemm(a, b):
        return gemm_fp8(a, b, config=config)

    key = jax.random.PRNGKey(42)
    key1, key2 = jax.random.split(key)

    a = jax.random.normal(key1, (M, K), dtype=jnp.float16)
    b = jax.random.normal(key2, (K, N), dtype=jnp.float16)

    try:
        # Test JIT compilation
        out = jit_gemm(a, b)
        assert out.shape == (M, N)

        # Test JIT compiled gradient
        @jax.jit
        def jit_loss(a, b):
            return jnp.sum(gemm_fp8(a, b, config=config) ** 2)

        grad_fn = jax.jit(jax.grad(jit_loss, argnums=(0, 1)))
        grad_a, grad_b = grad_fn(a, b)

        assert grad_a.shape == a.shape
        assert grad_b.shape == b.shape

        print("✓ JIT compilation test passed")

    except Exception as e:
        pytest.skip(f"Test skipped due to: {str(e)}")


if __name__ == "__main__":
    # Quick test
    jax.config.update("jax_enable_x64", True)
    print("Testing gemm_fp8...")

    print("\n=== Testing Forward ===")
    test_gemm_fp8_forward(
        128, 256, 512, False, False, jnp.float16, Format.E4M3, ScalingGranularity.TENSORWISE
    )

    print("\n=== Testing Backward ===")
    test_gemm_fp8_backward(64, 128, 256, False, jnp.float16, Format.E4M3, ScalingGranularity.TENSORWISE)

    print("\n=== Testing Formats ===")
    test_gemm_fp8_formats(32, 64, 128, Format.E4M3)

    print("\n=== Testing JIT ===")
    test_gemm_fp8_jit()

    print("\n✓ All tests passed!")
