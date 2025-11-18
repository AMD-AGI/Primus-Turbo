###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from primus_turbo.jax.primitive.grouped_gemm import (
    compute_group_offs_p,
    grouped_gemm_forward,
)


def grouped_gemm(a, b, group_lens, group_offs=None, transA=False, transB=False, num_cu=-1):
    """Grouped GEMM wrapper function."""
    if group_offs is None:
        group_offs = compute_group_offs_p.bind(group_lens)
    return grouped_gemm_forward(a, b, group_lens, group_offs, transA, transB, num_cu)


def grouped_gemm_ref(a, b, group_lens, group_offs, transA=False, transB=False):
    """Reference implementation of grouped GEMM using JAX ops."""
    bs = b.shape[0]
    m = a.shape[0]
    n = b.shape[2] if not transB else b.shape[1]

    # Initialize output
    c = jnp.zeros((m, n), dtype=a.dtype)

    # Process each group
    for i in range(bs):
        start_idx = group_offs[i]
        end_idx = group_offs[i + 1]

        a_slice = a[start_idx:end_idx, :]
        b_slice = b[i, :, :]

        # Apply transpose if needed
        if transA:
            a_slice = a_slice.T
        if transB:
            b_slice = b_slice.T

        # Compute matmul for this group
        c_slice = jnp.matmul(a_slice, b_slice)
        c = c.at[start_idx:end_idx, :].set(c_slice)

    return c


def generate_group_lens(bs, m, balance=True):
    """Generate group lengths similar to PyTorch version."""
    if balance:
        # Balanced groups - all same size
        return jnp.full((bs,), m, dtype=jnp.int64)
    else:
        # Unbalanced groups
        key = jax.random.PRNGKey(42)
        lengths = jax.random.randint(key, (bs,), m // 2, m * 2)
        # Normalize to sum to bs * m
        total = jnp.sum(lengths)
        lengths = (lengths * (bs * m) / total).astype(jnp.int64)
        return lengths


@pytest.mark.parametrize("B", [2, 4])
@pytest.mark.parametrize("M", [128, 256])
@pytest.mark.parametrize("N_K", [(128, 256), (256, 512)])
@pytest.mark.parametrize("dtype", [jnp.float16, jnp.bfloat16])
@pytest.mark.parametrize("trans_b", [True, False])
def test_grouped_gemm(B, M, N_K, dtype, trans_b):
    """Test grouped GEMM forward and backward passes."""
    jax.config.update("jax_enable_x64", True)

    N, K = N_K
    group_lens = generate_group_lens(B, M, balance=True)

    # Create input tensors
    key = jax.random.PRNGKey(0)
    key1, key2 = jax.random.split(key)

    a = jax.random.normal(key1, (B * M, K), dtype=jnp.float32)
    if trans_b:
        b = jax.random.normal(key2, (B, N, K), dtype=jnp.float32)
    else:
        b = jax.random.normal(key2, (B, K, N), dtype=jnp.float32)

    a = a.astype(dtype)
    b = b.astype(dtype)

    group_offs = compute_group_offs_p.bind(group_lens)

    #######################################
    # Forward
    c = grouped_gemm(a, b, group_lens, group_offs, transA=False, transB=trans_b)
    c_ref = grouped_gemm_ref(a, b, group_lens, group_offs, transA=False, transB=trans_b)

    # Check forward results
    if dtype == jnp.float16:
        rtol, atol = 1e-2, 1e-2
    else:  # bfloat16
        rtol, atol = 1e-1, 1e-1

    c_f32 = c.astype(jnp.float32)
    c_ref_f32 = c_ref.astype(jnp.float32)
    np.testing.assert_allclose(c_f32, c_ref_f32, rtol=rtol, atol=atol)

    #######################################
    # Backward
    def loss_fn(a, b):
        return jnp.sum(grouped_gemm(a, b, group_lens, group_offs, transA=False, transB=trans_b))

    def loss_fn_ref(a, b):
        return jnp.sum(grouped_gemm_ref(a, b, group_lens, group_offs, transA=False, transB=trans_b))

    grad_fn = jax.grad(loss_fn, argnums=(0, 1))
    grad_fn_ref = jax.grad(loss_fn_ref, argnums=(0, 1))

    grad_a, grad_b = grad_fn(a, b)
    grad_a_ref, grad_b_ref = grad_fn_ref(a, b)

    # Check backward results
    assert grad_a.shape == a.shape
    assert grad_b.shape == b.shape
    assert not jnp.any(jnp.isnan(grad_a))
    assert not jnp.any(jnp.isnan(grad_b))

    # Compare gradients with reference (looser tolerance for gradients)
    grad_a_f32 = grad_a.astype(jnp.float32)
    grad_a_ref_f32 = grad_a_ref.astype(jnp.float32)
    grad_b_f32 = grad_b.astype(jnp.float32)
    grad_b_ref_f32 = grad_b_ref.astype(jnp.float32)

    np.testing.assert_allclose(grad_a_f32, grad_a_ref_f32, rtol=rtol * 5, atol=atol * 5)
    np.testing.assert_allclose(grad_b_f32, grad_b_ref_f32, rtol=rtol * 5, atol=atol * 5)


if __name__ == "__main__":
    # Quick test
    jax.config.update("jax_enable_x64", True)
    print("Testing grouped_gemm (forward + backward)...")
    test_grouped_gemm(2, 256, (256, 512), jnp.float16, True)
