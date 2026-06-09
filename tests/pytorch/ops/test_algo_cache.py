###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Tests for the hipBLASLt algorithm cache.

Verifies that caching the hipblasLtMatmulAlgo_t across repeat GEMM calls
produces bit-identical results and that different GEMM shapes do not
collide in the cache.
"""

import pytest
import torch

from primus_turbo.pytorch.core.backend import BackendType, GlobalBackendManager
from primus_turbo.pytorch.core.low_precision import (
    Float8QuantConfig,
    Format,
    ScalingGranularity,
)
from primus_turbo.pytorch.ops import gemm, gemm_fp8
from tests.pytorch.test_utils import compute_snr

DEVICE = "cuda:0"
DTYPE = torch.bfloat16


@pytest.fixture(autouse=True)
def _set_hipblaslt_backend():
    """Force hipBLASLt for all tests and reset afterwards."""
    GlobalBackendManager.set_gemm_backend(BackendType.HIPBLASLT)
    GlobalBackendManager.set_auto_tune(False)
    torch.ops.primus_turbo_cpp_extension.hipblaslt_algo_cache_clear()
    yield
    GlobalBackendManager.reset()


class TestAlgoCacheBitExact:
    """Cache hit must produce the same result as cache miss."""

    @pytest.mark.parametrize("m,n,k", [(256, 512, 1024)])
    def test_repeated_call_same_result(self, m, n, k):
        torch.manual_seed(0)
        a_shape = (m, k)
        b_shape = (n, k)

        a = torch.randn(a_shape, dtype=DTYPE, device=DEVICE)
        b = torch.randn(b_shape, dtype=DTYPE, device=DEVICE)

        config = Float8QuantConfig(granularity=ScalingGranularity.TENSORWISE, format=Format.E4M3)

        c1 = gemm_fp8(a, b, False, True, DTYPE, config)
        torch.cuda.synchronize()
        c2 = gemm_fp8(a, b, False, True, DTYPE, config)
        torch.cuda.synchronize()

        assert torch.equal(c1, c2), "Repeat calls with same inputs must be bit-identical"


class TestAlgoCacheMultipleShapes:
    """Different shapes must produce correct results (no key collisions)."""

    SHAPES = [
        (128, 256, 512),
        (256, 512, 1024),
        (512, 1024, 2048),
        (1024, 3072, 768),
    ]

    @pytest.mark.parametrize("m,n,k", SHAPES)
    def test_shape_correctness(self, m, n, k):
        torch.manual_seed(42)
        a = torch.randn(m, k, dtype=DTYPE, device=DEVICE)
        b = torch.randn(n, k, dtype=DTYPE, device=DEVICE)

        config = Float8QuantConfig(granularity=ScalingGranularity.TENSORWISE, format=Format.E4M3)
        c = gemm_fp8(a, b, False, True, DTYPE, config)
        torch.cuda.synchronize()

        c_ref = a @ b.T
        snr = compute_snr(c_ref, c)
        assert snr > 25, f"SNR {snr:.1f} dB too low for shape ({m},{n},{k})"


class TestAlgoCacheBF16:
    """Cache must work correctly for BF16 (non-FP8) GEMM path."""

    @pytest.mark.parametrize("m,n,k", [(256, 512, 1024)])
    def test_bf16_repeated_call_same_result(self, m, n, k):
        torch.manual_seed(0)
        a = torch.randn(m, k, dtype=DTYPE, device=DEVICE)
        b = torch.randn(n, k, dtype=DTYPE, device=DEVICE)

        c1 = gemm(a, b, trans_b=True)
        torch.cuda.synchronize()
        c2 = gemm(a, b, trans_b=True)
        torch.cuda.synchronize()

        assert torch.equal(c1, c2), "BF16 repeat calls must be bit-identical"

    SHAPES = [
        (128, 256, 512),
        (256, 512, 1024),
        (1024, 3072, 768),
    ]

    @pytest.mark.parametrize("m,n,k", SHAPES)
    def test_bf16_shape_correctness(self, m, n, k):
        torch.manual_seed(42)
        a = torch.randn(m, k, dtype=DTYPE, device=DEVICE)
        b = torch.randn(n, k, dtype=DTYPE, device=DEVICE)

        c = gemm(a, b, trans_b=True)
        torch.cuda.synchronize()

        c_ref = a @ b.T
        assert torch.allclose(
            c, c_ref, rtol=1e-2, atol=1e-2
        ), f"BF16 GEMM result mismatch for shape ({m},{n},{k})"


class TestAlgoCacheLayouts:
    """Cache must handle different transpose layouts correctly."""

    @pytest.mark.parametrize("layout", ["NT", "NN"])
    def test_layout(self, layout):
        m, n, k = 256, 512, 1024
        torch.manual_seed(42)

        trans_a = layout[0] == "T"
        trans_b = layout[1] == "T"

        a_shape = (k, m) if trans_a else (m, k)
        b_shape = (n, k) if trans_b else (k, n)

        a = torch.randn(a_shape, dtype=DTYPE, device=DEVICE)
        b = torch.randn(b_shape, dtype=DTYPE, device=DEVICE)

        config = Float8QuantConfig(granularity=ScalingGranularity.TENSORWISE, format=Format.E4M3)
        c = gemm_fp8(a, b, trans_a, trans_b, DTYPE, config)
        torch.cuda.synchronize()

        a_mat = a.T if trans_a else a
        b_mat = b.T if trans_b else b
        c_ref = a_mat @ b_mat
        snr = compute_snr(c_ref, c)
        assert snr > 25, f"SNR {snr:.1f} dB too low for layout {layout}"


class TestAlgoCacheOverflow:
    """Cache overflow (clear) must not break correctness."""

    def test_correctness_after_overflow(self):
        """Insert >256 unique shapes to trigger cache clear, verify correctness holds."""
        config = Float8QuantConfig(granularity=ScalingGranularity.TENSORWISE, format=Format.E4M3)
        n, k = 512, 1024
        for m in range(64, 64 + 300):
            torch.manual_seed(m)
            a = torch.randn(m, k, dtype=DTYPE, device=DEVICE)
            b = torch.randn(n, k, dtype=DTYPE, device=DEVICE)
            c = gemm_fp8(a, b, False, True, DTYPE, config)
            torch.cuda.synchronize()
            c_ref = a.float() @ b.float().T
            snr = compute_snr(c_ref, c)
            assert snr > 20, f"SNR {snr:.1f} dB at m={m} (post-overflow)"
