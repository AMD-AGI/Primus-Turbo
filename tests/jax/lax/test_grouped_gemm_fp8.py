###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import jax
import jax.numpy as jnp
import pytest

from primus_turbo.jax.core.float8 import Float8QuantConfig, Format, ScalingGranularity
from primus_turbo.jax.lax import grouped_gemm_fp8
from tests.jax.ref.gemm_ref import generate_grouped_gemm_group_lens, grouped_gemm_ref
from tests.jax.test_utils import compute_snr


def _check_hit_int32_limit(B, M, N, K):
    a_elems = B * M * K
    b_elems = B * N * K
    out_elems = B * M * N
    return max(a_elems, out_elems, b_elems) >= 2**31


@pytest.mark.parametrize("B", [1, 2, 3, 8, 16, 32, 64])
@pytest.mark.parametrize("M", [128, 256, 512, 1024, 2048, 4096, 8192])
@pytest.mark.parametrize(
    "NK",
    [
        (2048, 1536),
        (2048, 1408),
        (1408, 2048),
        (2816, 2048),
        (3072, 5120),
        (5120, 1536),
        (4096, 7168),
        (7168, 2048),
    ],
)
@pytest.mark.parametrize("ori_dtype", [jnp.bfloat16, jnp.float16])
@pytest.mark.parametrize("format", [Format.E4M3, Format.E5M2])
@pytest.mark.parametrize("granularity", [ScalingGranularity.TENSORWISE, ScalingGranularity.ROWWISE])
@pytest.mark.parametrize("trans_b", [True, False])
@pytest.mark.parametrize("balance", [True, False])
def test_grouped_gemm_fp8(B, M, NK, ori_dtype, format, granularity, trans_b, balance):
    # Enable int64 support
    jax.config.update("jax_enable_x64", True)

    N, K = NK

    if _check_hit_int32_limit(B, M, N, K):
        pytest.skip("Shape hits int32 indexing limit (numel >= 2**31).")

    group_lens = generate_grouped_gemm_group_lens(B, M, balance=balance)
    print(
        f"\nB={B}, M={M}, N={N}, K={K}, ori_dtype={ori_dtype}, format={format}, "
        f"granularity={granularity}, trans_b={trans_b}, balance={balance}"
    )

    b_shape = (B, N, K) if trans_b else (B, K, N)

    key = jax.random.PRNGKey(0)
    key_a, key_b = jax.random.split(key)

    a = jax.random.normal(key_a, (B * M, K), dtype=jnp.float32).astype(ori_dtype)
    b = jax.random.normal(key_b, b_shape, dtype=jnp.float32).astype(ori_dtype)

    # Create reference copies
    a_ref = a.copy()
    b_ref = b.copy()

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

    # Check forward only
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

    print("(Backward tests temporarily disabled for debugging)")
