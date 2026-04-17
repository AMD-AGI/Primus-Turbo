###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""
Tests for the _get_fp8_dtype() helper introduced in:
  - primus_turbo/pytorch/ops/gemm_fp8.py        (PR #257 – refine get_fp8_dtype)
  - primus_turbo/pytorch/ops/grouped_gemm_fp8.py (same refactor)

The HYBRID format branch was absent from FP8GemmTensorFunction before the
refactor and is a regression risk: if the helper raises instead of returning
float8_e4m3 / float8_e5m2 for the HYBRID case, all hybrid-format GEMM calls
will silently break.

These tests are CPU-only and do not require a GPU.
"""

import pytest

from primus_turbo.pytorch.core.low_precision import (
    Format,
    float8_e4m3,
    float8_e5m2,
)

# Import the helpers via their module's private name (they are module-level
# functions, not exported, so we reach them directly).
from primus_turbo.pytorch.ops.gemm_fp8 import _get_fp8_dtype as gemm_fp8_get_fp8_dtype
from primus_turbo.pytorch.ops.grouped_gemm_fp8 import _get_fp8_dtype as grouped_gemm_fp8_get_fp8_dtype


@pytest.mark.parametrize(
    "get_fp8_dtype",
    [gemm_fp8_get_fp8_dtype, grouped_gemm_fp8_get_fp8_dtype],
    ids=["gemm_fp8", "grouped_gemm_fp8"],
)
class TestGetFP8Dtype:
    def test_e4m3_forward(self, get_fp8_dtype):
        assert get_fp8_dtype(Format.E4M3, True) == float8_e4m3

    def test_e4m3_backward(self, get_fp8_dtype):
        assert get_fp8_dtype(Format.E4M3, False) == float8_e4m3

    def test_e5m2_forward(self, get_fp8_dtype):
        assert get_fp8_dtype(Format.E5M2, True) == float8_e5m2

    def test_e5m2_backward(self, get_fp8_dtype):
        assert get_fp8_dtype(Format.E5M2, False) == float8_e5m2

    def test_hybrid_forward_returns_e4m3(self, get_fp8_dtype):
        """HYBRID forward stage must use E4M3 (higher precision for activations)."""
        assert get_fp8_dtype(Format.HYBRID, True) == float8_e4m3

    def test_hybrid_backward_returns_e5m2(self, get_fp8_dtype):
        """HYBRID backward stage must use E5M2 (wider range for gradients)."""
        assert get_fp8_dtype(Format.HYBRID, False) == float8_e5m2

    def test_unsupported_format_raises(self, get_fp8_dtype):
        with pytest.raises(ValueError, match="Unsupported FP8 format"):
            get_fp8_dtype(Format.E2M1_X2, True)
