###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

from primus_turbo.jax.core.low_precision import (
    Float8QuantConfig,
    Format,
    ScalingGranularity,
)
from primus_turbo.jax.core.utils import get_device_compute_capability
from primus_turbo.jax.lax import grouped_gemm_fp8
from tests.jax.ref.gemm_ref import (
    generate_grouped_gemm_group_lens,
    grouped_gemm_ref_fwd_bwd,
)
from tests.jax.test_utils import compute_snr


@pytest.mark.parametrize("B", [16, 32])
@pytest.mark.parametrize("M", [128, 1024, 4096])
@pytest.mark.parametrize(
    "NK",
    [(2048, 1408), (1408, 2048), (2816, 2048), (3072, 5120), (5120, 1536), (4096, 7168)],
)
@pytest.mark.parametrize("ori_dtype", [jnp.bfloat16, jnp.float16])
@pytest.mark.parametrize("format", [Format.E4M3, Format.E5M2])
@pytest.mark.parametrize("granularity", [ScalingGranularity.TENSORWISE, ScalingGranularity.ROWWISE])
@pytest.mark.parametrize("trans_b", [True, False])
@pytest.mark.parametrize("balance", [True, False])
def test_grouped_gemm_fp8(B, M, NK, ori_dtype, format, granularity, trans_b, balance):
    # TODO(xiaobochen-amd): On gfx942, the hipBLASLt path can hang/flake when M <= 512.
    # This has been observed under pytest; root cause not yet identified. MI355 works normally.
    if M <= 512 and get_device_compute_capability() == (9, 4):
        pytest.skip("gfx942: hipBLASLt path can hang/flake when M <= 512")

    N, K = NK
    print(
        f"\nB={B}, M={M}, N={N}, K={K}, dtype={ori_dtype}, format={format}, "
        f"granularity={granularity}, trans_b={trans_b}, balance={balance}"
    )

    # Skip if hits int32 indexing limit
    if max(B * M * K, B * N * K, B * M * N) >= 2**31:
        pytest.skip("Shape hits int32 indexing limit")

    group_lens = generate_grouped_gemm_group_lens(B, M, balance=balance)
    b_shape = (B, N, K) if trans_b else (B, K, N)

    key = jax.random.PRNGKey(0)
    a = jax.random.normal(key, (B * M, K), dtype=jnp.float32).astype(ori_dtype)
    b = jax.random.normal(key, b_shape, dtype=jnp.float32).astype(ori_dtype)

    # Clone for reference
    a_ref = jnp.array(a)
    b_ref = jnp.array(b)

    # Reference forward + backward
    out_ref, grad_a_ref, grad_b_ref = grouped_gemm_ref_fwd_bwd(a_ref, b_ref, group_lens, trans_b=trans_b)

    # FP8 forward
    config = Float8QuantConfig(format=format, granularity=granularity)
    out = grouped_gemm_fp8(a, b, group_lens, trans_b=trans_b, config=config)

    # FP8 backward
    def loss_fn(a, b):
        return jnp.sum(grouped_gemm_fp8(a, b, group_lens, trans_b=trans_b, config=config))

    grad_a, grad_b = jax.grad(loss_fn, argnums=(0, 1))(a, b)

    # Validation
    snr_threshold = 25 if format == Format.E4M3 else 20

    out_snr = compute_snr(out_ref, out)
    print(f"Out-SNR: {out_snr:.2f} dB")
    assert out_snr > snr_threshold, f"out_snr too low: {out_snr:.2f} dB"

    a_grad_snr = compute_snr(grad_a_ref, grad_a)
    print(f"AGrad-SNR: {a_grad_snr:.2f} dB")
    assert a_grad_snr > snr_threshold, f"a_grad_snr too low: {a_grad_snr:.2f} dB"

    b_grad_snr = compute_snr(grad_b_ref, grad_b)
    print(f"BGrad-SNR: {b_grad_snr:.2f} dB")
    assert b_grad_snr > snr_threshold, f"b_grad_snr too low: {b_grad_snr:.2f} dB"
