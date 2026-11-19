###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import os

# Each new problem signature is tuned once (in a subprocess); time fewer candidates than the default 256.
os.environ.setdefault("PRIMUS_TURBO_HB_GG_TUNE_CANDIDATES", "16")

import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

from primus_turbo.jax.lax.grouped_gemm_hipblaslt import grouped_gemm_hipblaslt
from tests.jax.ref.gemm_ref import (
    generate_grouped_gemm_group_lens,
    grouped_gemm_ref_fwd_bwd,
)
from tests.jax.test_utils import assert_allclose, compute_snr, get_tolerances


def _fwd_bwd(a, b, group_lens, trans_b):
    def loss_fn(a, b):
        return jnp.sum(grouped_gemm_hipblaslt(a, b, group_lens, transB=trans_b))

    out = grouped_gemm_hipblaslt(a, b, group_lens, transB=trans_b)
    grad_a, grad_b = jax.grad(loss_fn, argnums=(0, 1))(a, b)
    return out, grad_a, grad_b


def _check(got, ref, dtype):
    tol = get_tolerances(dtype)
    snr_threshold = 45 if dtype == jnp.bfloat16 else 50
    for x, x_ref in zip(got, ref):
        assert_allclose(x, x_ref, **tol)
        assert compute_snr(x_ref, x) > snr_threshold


@pytest.mark.parametrize("B", [1, 8, 32])
@pytest.mark.parametrize("M", [1024])
@pytest.mark.parametrize("N_K", [(2048, 1536), (2816, 2048), (4096, 7168)])
@pytest.mark.parametrize("dtype", [jnp.bfloat16, jnp.float16])
@pytest.mark.parametrize("balance", [True, False])
@pytest.mark.parametrize("trans_b", [True, False])
def test_grouped_gemm_hipblaslt(B, M, N_K, dtype, balance, trans_b):
    if B == 1 and not balance:
        pytest.skip("a single group is always balanced")
    N, K = N_K
    group_lens = generate_grouped_gemm_group_lens(B, M, balance=balance)
    b_shape = (B, N, K) if trans_b else (B, K, N)

    key_a, key_b = jax.random.split(jax.random.PRNGKey(0))
    a = jax.random.normal(key_a, (B * M, K), dtype=dtype)
    b = jax.random.normal(key_b, b_shape, dtype=dtype)

    got = _fwd_bwd(a, b, group_lens, trans_b)
    ref = grouped_gemm_ref_fwd_bwd(a, b, group_lens, trans_b=trans_b)
    _check(got, ref, dtype)


@pytest.mark.parametrize("trans_b", [True, False])
@pytest.mark.parametrize("lens_dtype", [jnp.int32, jnp.int64])
def test_grouped_gemm_hipblaslt_empty_groups_and_padding(trans_b, lens_dtype):
    """Empty groups and rows past sum(group_lens): those output and input-gradient rows are zero."""
    dtype = jnp.bfloat16
    N, K = 2048, 1024
    lens = [700, 0, 1300, 0, 512, 900, 0, 588]
    valid, M = sum(lens), 4096  # 96 padding rows
    group_lens = jnp.array(lens, dtype=lens_dtype)
    b_shape = (len(lens), N, K) if trans_b else (len(lens), K, N)

    key_a, key_b = jax.random.split(jax.random.PRNGKey(1))
    a = jax.random.normal(key_a, (M, K), dtype=dtype)
    b = jax.random.normal(key_b, b_shape, dtype=dtype)

    out, grad_a, grad_b = _fwd_bwd(a, b, group_lens, trans_b)
    ref = grouped_gemm_ref_fwd_bwd(a[:valid], b, group_lens, trans_b=trans_b)
    _check((out[:valid], grad_a[:valid], grad_b), ref, dtype)
    assert not jnp.any(out[valid:])
    assert not jnp.any(grad_a[valid:])
    for g, n in enumerate(lens):
        if n == 0:
            assert not jnp.any(grad_b[g])
