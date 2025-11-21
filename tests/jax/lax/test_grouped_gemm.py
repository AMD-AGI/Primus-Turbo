###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import jax
import jax.numpy as jnp
import pytest

from primus_turbo.jax.lax.grouped_gemm import grouped_gemm
from tests.jax.ref.gemm_ref import generate_grouped_gemm_group_lens, grouped_gemm_ref
from tests.jax.test_utils import compute_snr


@pytest.mark.parametrize("B", [16, 32])
@pytest.mark.parametrize("M", [128, 512, 1024, 2048])
@pytest.mark.parametrize(
    "N_K", [(2048, 1536), (2048, 1408), (2816, 2048), (3072, 5120), (5120, 1536), (4096, 7168), (7168, 2048)]
)
@pytest.mark.parametrize("dtype", [jnp.bfloat16, jnp.float16])
@pytest.mark.parametrize("balance", [False])
@pytest.mark.parametrize("trans_b", [True, False])
@pytest.mark.parametrize("reduce_num_cu", [0, 16, 32])
def test_grouped_gemm(B, M, N_K, dtype, balance, trans_b, reduce_num_cu):
    """Test grouped GEMM forward and backward pass."""
    jax.config.update("jax_enable_x64", True)

    N, K = N_K
    group_lens = generate_grouped_gemm_group_lens(B, M, balance=balance)

    key = jax.random.PRNGKey(0)
    key1, key2 = jax.random.split(key)

    b_shape = (B, N, K) if trans_b else (B, K, N)

    a = jax.random.normal(key1, (B * M, K), dtype=jnp.float32)
    b = jax.random.normal(key2, b_shape, dtype=jnp.float32)
    a = a.astype(dtype)
    b = b.astype(dtype)

    a_ref = jnp.array(a, copy=True)
    b_ref = jnp.array(b, copy=True)

    # Compute num_cu
    num_cu = -1 if reduce_num_cu == 0 else reduce_num_cu

    #######################################
    # Forward
    out = grouped_gemm(a, b, group_lens, transB=trans_b, num_cu=num_cu)

    out_ref = grouped_gemm_ref(a_ref, b_ref, group_lens, trans_b=trans_b)

    # Check forward results using SNR
    out_snr = compute_snr(out_ref, out)
    print(f"Out-SNR: {out_snr:.2f} dB")
    assert out_snr > 20, f"out_snr too low: {out_snr:.2f} dB"

    def loss_fn(a, b):
        return jnp.sum(grouped_gemm(a, b, group_lens, transB=trans_b, num_cu=num_cu))

    def loss_fn_ref(a, b):
        return jnp.sum(grouped_gemm_ref(a, b, group_lens, trans_b=trans_b))

    grad_fn = jax.grad(loss_fn, argnums=(0, 1))
    grad_fn_ref = jax.grad(loss_fn_ref, argnums=(0, 1))

    grad_a, grad_b = grad_fn(a, b)
    grad_a_ref, grad_b_ref = grad_fn_ref(a_ref, b_ref)

    # Check gradients using SNR
    a_grad_snr = compute_snr(grad_a_ref, grad_a)
    print(f"AGrad-SNR: {a_grad_snr:.2f} dB")
    assert a_grad_snr > 20, f"a_grad_snr too low: {a_grad_snr:.2f} dB"

    b_grad_snr = compute_snr(grad_b_ref, grad_b)
    print(f"BGrad-SNR: {b_grad_snr:.2f} dB")
    assert b_grad_snr > 20, f"b_grad_snr too low: {b_grad_snr:.2f} dB"
