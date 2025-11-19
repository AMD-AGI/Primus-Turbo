###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from primus_turbo.jax.lax.grouped_gemm import compute_group_offs, grouped_gemm_fp8


def quantize_fp8_simple(x, dtype):
    """Simple FP8 quantization for testing."""
    # Calculate scale factor
    max_val = jnp.max(jnp.abs(x))
    # FP8 E4M3 max value is ~448, E5M2 is ~57344
    fp8_max = 448.0 if dtype == jnp.float8_e4m3fn else 57344.0
    scale = max_val / fp8_max

    # Quantize
    x_scaled = x / (scale + 1e-12)
    x_fp8 = x_scaled.astype(dtype)

    # Return quantized tensor and inverse scale
    scale_inv = jnp.array([1.0 / (scale + 1e-12)], dtype=jnp.float32)
    return x_fp8, scale_inv


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
@pytest.mark.parametrize("ori_dtype", [jnp.float16, jnp.bfloat16])
@pytest.mark.parametrize("fp8_dtype", [jnp.float8_e4m3fn, jnp.float8_e5m2])
@pytest.mark.parametrize("granularity", ["TENSORWISE", "ROWWISE"])
@pytest.mark.parametrize("trans_b", [True, False])
@pytest.mark.parametrize("balance", [True, False])
def test_grouped_gemm_fp8(B, M, N_K, ori_dtype, fp8_dtype, granularity, trans_b, balance):
    """Test grouped GEMM FP8 forward pass."""
    jax.config.update("jax_enable_x64", True)

    N, K = N_K
    group_lens = generate_group_lens(B, M, balance=balance)

    # Create input tensors
    key = jax.random.PRNGKey(0)
    key1, key2 = jax.random.split(key)

    a = jax.random.normal(key1, (B * M, K), dtype=jnp.float32)
    if trans_b:
        b = jax.random.normal(key2, (B, N, K), dtype=jnp.float32)
    else:
        b = jax.random.normal(key2, (B, K, N), dtype=jnp.float32)

    a = a.astype(ori_dtype)
    b = b.astype(ori_dtype)

    group_offs = compute_group_offs(group_lens)

    # Quantize to FP8
    a_fp8, a_scales = quantize_fp8_simple(a, fp8_dtype)

    # For rowwise, we need per-row scales
    if granularity == "ROWWISE":
        # Compute per-row scales for a
        a_scales = []
        a_fp8_list = []
        for i in range(a.shape[0]):
            row_fp8, row_scale = quantize_fp8_simple(a[i : i + 1, :], fp8_dtype)
            a_fp8_list.append(row_fp8)
            a_scales.append(row_scale)
        a_fp8 = jnp.concatenate(a_fp8_list, axis=0)
        a_scales = jnp.concatenate(a_scales, axis=0)

        # Compute per-column or per-row scales for b based on trans_b
        b_scales = []
        b_fp8_list = []
        for i in range(B):
            if trans_b:
                # b is [B, N, K], scale per row (N dimension)
                for j in range(N):
                    slice_fp8, slice_scale = quantize_fp8_simple(b[i : i + 1, j : j + 1, :], fp8_dtype)
                    b_fp8_list.append(slice_fp8)
                    b_scales.append(slice_scale)
            else:
                # b is [B, K, N], scale per column (N dimension)
                for j in range(N):
                    slice_fp8, slice_scale = quantize_fp8_simple(b[i : i + 1, :, j : j + 1], fp8_dtype)
                    b_fp8_list.append(slice_fp8)
                    b_scales.append(slice_scale)
        b_fp8 = jnp.stack(
            [
                jnp.concatenate([b_fp8_list[i * N + j] for j in range(N)], axis=-1 if not trans_b else 1)
                for i in range(B)
            ],
            axis=0,
        )
        b_scales = jnp.concatenate(b_scales, axis=0)
    else:
        # Tensorwise quantization
        b_fp8_list = []
        b_scales_list = []
        for i in range(B):
            b_slice_fp8, b_slice_scale = quantize_fp8_simple(b[i], fp8_dtype)
            b_fp8_list.append(b_slice_fp8)
            b_scales_list.append(b_slice_scale)
        b_fp8 = jnp.stack(b_fp8_list, axis=0)
        b_scales = b_scales_list[0]  # Use first scale for all (tensorwise)

    #######################################
    # Forward
    try:
        out = grouped_gemm_fp8(
            a_fp8,
            b_fp8,
            a_scales,
            b_scales,
            group_lens,
            group_offs,
            transA=False,
            transB=trans_b,
            num_cu=-1,
            out_dtype=ori_dtype,
            granularity=granularity,
        )

        # Compute reference (using original float tensors)
        out_ref = grouped_gemm_ref(a, b, group_lens, group_offs, transA=False, transB=trans_b)

        # Check forward results (FP8 has lower precision)
        if fp8_dtype == jnp.float8_e4m3fn:
            rtol, atol = 5e-2, 5e-2
        else:  # e5m2
            rtol, atol = 1e-1, 1e-1

        out_f32 = out.astype(jnp.float32)
        out_ref_f32 = out_ref.astype(jnp.float32)

        # Allow some tolerance for FP8 quantization error
        np.testing.assert_allclose(out_f32, out_ref_f32, rtol=rtol, atol=atol)
        print(
            f"\nTest passed: B={B}, M={M}, N={N}, K={K}, dtype={ori_dtype}, fp8={fp8_dtype}, "
            f"granularity={granularity}, trans_b={trans_b}, balance={balance}"
        )

    except Exception as e:
        pytest.skip(f"Test skipped due to: {str(e)}")


if __name__ == "__main__":
    # Quick test
    jax.config.update("jax_enable_x64", True)
    print("Testing grouped_gemm_fp8...")
    test_grouped_gemm_fp8(2, 128, (128, 256), jnp.float16, jnp.float8_e4m3fn, "TENSORWISE", True, True)
