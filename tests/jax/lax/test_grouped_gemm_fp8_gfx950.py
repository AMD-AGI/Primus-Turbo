###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Per-tensor e4m3 grouped GEMM on gfx950 (FlyDSL and hipBLASLt backends): forward, dA and dW against the bf16 reference.

The hipBLASLt backend needs a hipBLASLt with FP8 grouped kernels for gfx950 (the stock library has none)."""

import os

os.environ.setdefault("PRIMUS_TURBO_HB_GG_TUNE_CANDIDATES", "16")

import jax
import jax.numpy as jnp
import pytest

from tests.jax.ref.gemm_ref import (
    generate_grouped_gemm_group_lens,
    grouped_gemm_ref_fwd_bwd,
)
from tests.jax.test_utils import compute_snr


def _is_gfx950():
    try:
        return (
            "gfx950" in jax.devices()[0].device_kind.lower()
            or "mi355" in jax.devices()[0].device_kind.lower()
        )
    except Exception:  # pylint: disable=broad-except
        return False


pytestmark = pytest.mark.skipif(not _is_gfx950(), reason="FP8 grouped GEMM backends need gfx950")


def _backend(name):
    if name == "flydsl":
        pytest.importorskip("flydsl")
        from primus_turbo.jax.lax.grouped_gemm_flydsl_fp8 import grouped_gemm_fp8_flydsl

        return grouped_gemm_fp8_flydsl
    from primus_turbo.jax.lax.grouped_gemm_hipblaslt_fp8 import grouped_gemm_fp8_hipblaslt

    return grouped_gemm_fp8_hipblaslt


def _fwd_bwd(fn, a, b, group_lens):
    out = fn(a, b, group_lens)
    grad_a, grad_b = jax.grad(lambda a, b: jnp.sum(fn(a, b, group_lens).astype(jnp.float32)), argnums=(0, 1))(
        a, b
    )
    return out, grad_a, grad_b


@pytest.mark.parametrize("backend", ["flydsl", "hipblaslt"])
@pytest.mark.parametrize("B", [8, 64])
@pytest.mark.parametrize("N_K", [(1408, 2048), (2048, 768), (4096, 4096)])
@pytest.mark.parametrize("balance", [True, False])
def test_grouped_gemm_fp8(backend, B, N_K, balance):
    N, K = N_K
    M = 2048
    fn = _backend(backend)
    group_lens = generate_grouped_gemm_group_lens(B, M, balance=balance)
    key_a, key_b = jax.random.split(jax.random.PRNGKey(0))
    a = (jax.random.normal(key_a, (B * M, K)) * 0.5).astype(jnp.bfloat16)
    b = (jax.random.normal(key_b, (B, K, N)) / K**0.5).astype(jnp.bfloat16)
    got = _fwd_bwd(fn, a, b, group_lens)
    ref = grouped_gemm_ref_fwd_bwd(a, b, group_lens, trans_b=False)
    for x, x_ref in zip(got, ref):
        # e4m3 per-tensor quantisation of both operands: ~3.7e-2 relative error, i.e. ~28 dB
        assert compute_snr(x_ref, x) > 25


@pytest.mark.parametrize("backend", ["flydsl", "hipblaslt"])
def test_grouped_gemm_fp8_empty_groups_and_padding(backend):
    """Empty groups and rows past sum(group_lens): those output and input-gradient rows are zero."""
    fn = _backend(backend)
    N, K = 2048, 1024
    lens = [700, 0, 1300, 0, 512, 900, 0, 588]
    valid, M = sum(lens), 4096  # 96 padding rows
    group_lens = jnp.array(lens, dtype=jnp.int32)
    key_a, key_b = jax.random.split(jax.random.PRNGKey(1))
    a = jax.random.normal(key_a, (M, K), dtype=jnp.bfloat16)
    b = (jax.random.normal(key_b, (len(lens), K, N)) / K**0.5).astype(jnp.bfloat16)
    out, grad_a, grad_b = _fwd_bwd(fn, a, b, group_lens)
    ref = grouped_gemm_ref_fwd_bwd(a[:valid], b, group_lens, trans_b=False)
    for x, x_ref in zip((out[:valid], grad_a[:valid], grad_b), ref):
        assert compute_snr(x_ref, x) > 25
    assert not jnp.any(out[valid:])
    assert not jnp.any(grad_a[valid:])
    for g, n in enumerate(lens):
        if n == 0:
            assert not jnp.any(grad_b[g])
