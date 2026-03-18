###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import pytest

from primus_turbo.pytorch.core.backend import (
    BackendType,
    GlobalBackendManager,
    PrecisionType,
)


@pytest.fixture(autouse=True)
def clean_backend_state(monkeypatch):
    """Reset backend state and clear env vars before/after each test."""
    GlobalBackendManager.reset()
    GlobalBackendManager._extract_backend_from_env.cache_clear()
    for key in (
        "PRIMUS_TURBO_GEMM_BACKEND",
        "PRIMUS_TURBO_GROUPED_GEMM_BACKEND",
        "PRIMUS_TURBO_MOE_DISPATCH_COMBINE_BACKEND",
        "PRIMUS_TURBO_AUTO_TUNE",
    ):
        monkeypatch.delenv(key, raising=False)
    yield
    GlobalBackendManager.reset()
    GlobalBackendManager._extract_backend_from_env.cache_clear()


class TestGlobalBackendManagerEnvVar:

    def test_gemm_backend_single_format(self, monkeypatch):
        """Format 1: PRIMUS_TURBO_GEMM_BACKEND=ck -> all precisions use CK."""
        monkeypatch.setenv("PRIMUS_TURBO_GEMM_BACKEND", "ck")
        assert GlobalBackendManager.get_gemm_backend(PrecisionType.FP4) == BackendType.CK
        assert GlobalBackendManager.get_gemm_backend(PrecisionType.FP8) == BackendType.CK
        assert GlobalBackendManager.get_gemm_backend(PrecisionType.BF16_FP16_FP32) == BackendType.CK

    def test_gemm_backend_per_precision_format(self, monkeypatch):
        """Format 2: fp4:hipblaslt,fp8:ck -> per-precision mapping."""
        monkeypatch.setenv("PRIMUS_TURBO_GEMM_BACKEND", "fp4:hipblaslt,fp8:ck")
        assert GlobalBackendManager.get_gemm_backend(PrecisionType.FP4) == BackendType.HIPBLASLT
        assert GlobalBackendManager.get_gemm_backend(PrecisionType.FP8) == BackendType.CK

    def test_grouped_gemm_backend_env(self, monkeypatch):
        monkeypatch.setenv("PRIMUS_TURBO_GROUPED_GEMM_BACKEND", "hipblaslt")
        assert GlobalBackendManager.get_grouped_gemm_backend(PrecisionType.FP8) == BackendType.HIPBLASLT

    def test_moe_dispatch_combine_backend_env(self, monkeypatch):
        monkeypatch.setenv("PRIMUS_TURBO_MOE_DISPATCH_COMBINE_BACKEND", "triton")
        assert GlobalBackendManager.get_moe_dispatch_combine_backend(PrecisionType.FP8) == BackendType.TRITON

    def test_auto_tune_env_enabled(self, monkeypatch):
        monkeypatch.setenv("PRIMUS_TURBO_AUTO_TUNE", "1")
        assert GlobalBackendManager.auto_tune_enabled() is True

    def test_auto_tune_env_disabled(self, monkeypatch):
        monkeypatch.setenv("PRIMUS_TURBO_AUTO_TUNE", "0")
        assert GlobalBackendManager.auto_tune_enabled() is False

    def test_gemm_backend_other_precision_format(self, monkeypatch):
        """Format 3: fp8:ck,other:hipblaslt -> FP8 uses CK, rest use HIPBLASLT."""
        monkeypatch.setenv("PRIMUS_TURBO_GEMM_BACKEND", "fp8:ck,other:hipblaslt")
        assert GlobalBackendManager.get_gemm_backend(PrecisionType.FP8) == BackendType.CK
        assert GlobalBackendManager.get_gemm_backend(PrecisionType.FP4) == BackendType.HIPBLASLT
        assert GlobalBackendManager.get_gemm_backend(PrecisionType.BF16_FP16_FP32) == BackendType.HIPBLASLT

    def test_gemm_backend_invalid_precision_raises(self, monkeypatch):
        """Invalid precision name should raise AssertionError."""
        monkeypatch.setenv("PRIMUS_TURBO_GEMM_BACKEND", "fp8:ck,invalid:hipblaslt")
        with pytest.raises(AssertionError, match="Precision INVALID not supported"):
            GlobalBackendManager.get_gemm_backend(PrecisionType.FP8)

    def test_returns_none_when_env_not_set(self):
        assert GlobalBackendManager.get_gemm_backend(PrecisionType.FP8) is None
        assert GlobalBackendManager.get_grouped_gemm_backend(PrecisionType.FP8) is None
        assert GlobalBackendManager.get_moe_dispatch_combine_backend(PrecisionType.FP8) is None
        assert GlobalBackendManager.auto_tune_enabled() is False


class TestGlobalBackendManagerFunction:

    @staticmethod
    def _init_gemm_backend():
        GlobalBackendManager._gemm_backend = {p: None for p in PrecisionType}

    @staticmethod
    def _init_grouped_gemm_backend():
        GlobalBackendManager._grouped_gemm_backend = {p: None for p in PrecisionType}

    def test_set_get_gemm_backend(self):
        self._init_gemm_backend()
        GlobalBackendManager.set_gemm_backend(BackendType.CK, PrecisionType.FP8)
        assert GlobalBackendManager.get_gemm_backend(PrecisionType.FP8) == BackendType.CK
        assert GlobalBackendManager.get_gemm_backend(PrecisionType.FP4) is None

    def test_set_get_gemm_backend_multiple_precisions(self):
        self._init_gemm_backend()
        GlobalBackendManager.set_gemm_backend(BackendType.HIPBLASLT, PrecisionType.FP4)
        GlobalBackendManager.set_gemm_backend(BackendType.CK, PrecisionType.FP8)
        GlobalBackendManager.set_gemm_backend(BackendType.HIPBLASLT, PrecisionType.BF16_FP16_FP32)
        assert GlobalBackendManager.get_gemm_backend(PrecisionType.FP4) == BackendType.HIPBLASLT
        assert GlobalBackendManager.get_gemm_backend(PrecisionType.FP8) == BackendType.CK
        assert GlobalBackendManager.get_gemm_backend(PrecisionType.BF16_FP16_FP32) == BackendType.HIPBLASLT

    def test_set_get_grouped_gemm_backend(self):
        self._init_grouped_gemm_backend()
        GlobalBackendManager.set_grouped_gemm_backend(BackendType.CK, PrecisionType.FP8)
        result = GlobalBackendManager.get_grouped_gemm_backend(PrecisionType.FP8)
        assert result is not None

    def test_set_auto_tune(self):
        GlobalBackendManager.set_auto_tune(True)
        assert GlobalBackendManager.auto_tune_enabled() is True
        GlobalBackendManager.set_auto_tune(False)
        assert GlobalBackendManager.auto_tune_enabled() is False

    def test_reset_clears_code_settings(self):
        self._init_gemm_backend()
        GlobalBackendManager.set_gemm_backend(BackendType.CK, PrecisionType.FP8)
        GlobalBackendManager.set_auto_tune(True)

        GlobalBackendManager.reset()

        assert GlobalBackendManager.get_gemm_backend(PrecisionType.FP8) is None
        assert GlobalBackendManager.auto_tune_enabled() is False
