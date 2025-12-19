###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

from primus_turbo.jax.lax.grouped_gemm import grouped_gemm
from tests.jax.ref.gemm_ref import (
    generate_grouped_gemm_group_lens,
    grouped_gemm_ref_fwd_bwd,
)
from tests.jax.test_utils import assert_allclose, compute_snr, get_tolerances


@pytest.mark.parametrize("B", [16, 32])
@pytest.mark.parametrize("M", [128, 1024, 2048])
@pytest.mark.parametrize(
    "N_K", [(2048, 1536), (2048, 1408), (2816, 2048), (3072, 5120), (5120, 1536), (4096, 7168), (7168, 2048)]
)
@pytest.mark.parametrize("dtype", [jnp.bfloat16, jnp.float16])
@pytest.mark.parametrize("balance", [True, False])
@pytest.mark.parametrize("trans_b", [True, False])
@pytest.mark.parametrize("reduce_num_cu", [0, 16])
def test_grouped_gemm(B, M, N_K, dtype, balance, trans_b, reduce_num_cu):
    N, K = N_K
    group_lens = generate_grouped_gemm_group_lens(B, M, balance=balance)
    b_shape = (B, N, K) if trans_b else (B, K, N)

    key = jax.random.PRNGKey(0)
    a = jax.random.normal(key, (B * M, K), dtype=dtype)
    b = jax.random.normal(key, b_shape, dtype=dtype)

    num_cu = -1 if reduce_num_cu == 0 else reduce_num_cu

    # Forward
    out = grouped_gemm(a, b, group_lens, transB=trans_b, num_cu=num_cu)

    # Backward
    def loss_fn(a, b):
        return jnp.sum(grouped_gemm(a, b, group_lens, transB=trans_b, num_cu=num_cu))

    grad_a, grad_b = jax.grad(loss_fn, argnums=(0, 1))(a, b)

    # Reference
    out_ref, grad_a_ref, grad_b_ref = grouped_gemm_ref_fwd_bwd(a, b, group_lens, trans_b=trans_b)

    snr_threshold = 45 if dtype == jnp.bfloat16 else 50

    tol = get_tolerances(dtype)
    assert_allclose(out, out_ref, **tol)
    assert_allclose(grad_a, grad_a_ref, **tol)
    assert_allclose(grad_b, grad_b_ref, **tol)
    assert compute_snr(out_ref, out) > snr_threshold
    assert compute_snr(grad_a_ref, grad_a) > snr_threshold
    assert compute_snr(grad_b_ref, grad_b) > snr_threshold
