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


class TestMoEDispatchCombineCustomEP:
    """Regression tests for the custom EP backend path added in PR #297.

    The docstring on get_moe_dispatch_combine_backend promises that an env
    value naming a non-BackendType backend (e.g. "UCCL_EP") is tolerated and
    returned as None so the EP registry in moe_dispatch_combine_impl can take
    over. A regression here would turn a controlled None into a KeyError and
    break any user with PRIMUS_TURBO_MOE_DISPATCH_COMBINE_BACKEND=UCCL_EP
    (or future third-party EP backends).
    """

    def test_unknown_custom_ep_name_returns_none(self, monkeypatch):
        monkeypatch.setenv("PRIMUS_TURBO_MOE_DISPATCH_COMBINE_BACKEND", "UCCL_EP")
        assert GlobalBackendManager.get_moe_dispatch_combine_backend(PrecisionType.FP8) is None
        assert GlobalBackendManager.get_moe_dispatch_combine_backend(PrecisionType.BF16_FP16_FP32) is None

    def test_unknown_custom_ep_does_not_leak_to_other_getters(self, monkeypatch):
        """Setting an unknown EP name must not affect the GEMM getters."""
        monkeypatch.setenv("PRIMUS_TURBO_MOE_DISPATCH_COMBINE_BACKEND", "SOMETHING_CUSTOM")
        assert GlobalBackendManager.get_gemm_backend(PrecisionType.FP8) is None
        assert GlobalBackendManager.get_grouped_gemm_backend(PrecisionType.FP8) is None

    def test_valid_backend_still_resolves(self, monkeypatch):
        """Sanity check: the custom-name path does not mask valid backends."""
        monkeypatch.setenv("PRIMUS_TURBO_MOE_DISPATCH_COMBINE_BACKEND", "TURBO")
        assert GlobalBackendManager.get_moe_dispatch_combine_backend(PrecisionType.FP8) == BackendType.TURBO


class TestExtractBackendFromEnvEdgeCases:
    """Coverage for the env-value parser shared by all three GEMM/EP getters.

    These tests run through _extract_backend_from_env via the public getters
    (rather than calling the lru_cache'd helper directly) so they also exercise
    the real resolution path.
    """

    def test_single_value_is_case_insensitive(self, monkeypatch):
        """Format 1 (single backend) must work for lower/upper/mixed case."""
        for value in ("ck", "CK", "Ck"):
            GlobalBackendManager._extract_backend_from_env.cache_clear()
            monkeypatch.setenv("PRIMUS_TURBO_GEMM_BACKEND", value)
            assert GlobalBackendManager.get_gemm_backend(PrecisionType.FP8) == BackendType.CK
            monkeypatch.delenv("PRIMUS_TURBO_GEMM_BACKEND")

    def test_per_precision_value_is_case_insensitive(self, monkeypatch):
        monkeypatch.setenv("PRIMUS_TURBO_GEMM_BACKEND", "FP8:Ck, FP4:HipBlasLt")
        assert GlobalBackendManager.get_gemm_backend(PrecisionType.FP8) == BackendType.CK
        assert GlobalBackendManager.get_gemm_backend(PrecisionType.FP4) == BackendType.HIPBLASLT

    def test_trailing_comma_and_whitespace_tolerated(self, monkeypatch):
        """Empty segments from trailing/extra commas must be skipped."""
        monkeypatch.setenv("PRIMUS_TURBO_GEMM_BACKEND", "fp8:ck,,  ,")
        assert GlobalBackendManager.get_gemm_backend(PrecisionType.FP8) == BackendType.CK
        # Unspecified precisions get None (no 'other:' clause)
        assert GlobalBackendManager.get_gemm_backend(PrecisionType.FP4) is None

    def test_other_only_applies_to_all_precisions(self, monkeypatch):
        """Format 3 with only 'other:<backend>' must fill every precision."""
        monkeypatch.setenv("PRIMUS_TURBO_GEMM_BACKEND", "other:triton")
        assert GlobalBackendManager.get_gemm_backend(PrecisionType.FP8) == BackendType.TRITON
        assert GlobalBackendManager.get_gemm_backend(PrecisionType.FP4) == BackendType.TRITON
        assert GlobalBackendManager.get_gemm_backend(PrecisionType.BF16_FP16_FP32) == BackendType.TRITON

    def test_partial_per_precision_leaves_others_none(self, monkeypatch):
        """Specifying only FP8 and no 'other' returns None for the rest."""
        monkeypatch.setenv("PRIMUS_TURBO_GEMM_BACKEND", "fp8:ck")
        assert GlobalBackendManager.get_gemm_backend(PrecisionType.FP8) == BackendType.CK
        assert GlobalBackendManager.get_gemm_backend(PrecisionType.FP4) is None
        assert GlobalBackendManager.get_gemm_backend(PrecisionType.BF16_FP16_FP32) is None

    def test_bf16_fp16_fp32_alias_to_single_precision(self, monkeypatch):
        """All three float precisions collapse to BF16_FP16_FP32 - last
        key wins (dict overwrite)."""
        monkeypatch.setenv("PRIMUS_TURBO_GEMM_BACKEND", "bf16:ck,fp16:hipblaslt,fp32:triton")
        assert GlobalBackendManager.get_gemm_backend(PrecisionType.BF16_FP16_FP32) == BackendType.TRITON

    def test_invalid_backend_name_raises_key_error_on_gemm(self, monkeypatch):
        """Unknown backend names in format 2 raise - this is a configuration
        error for GEMM paths (MoE path swallows it; see dedicated test above)."""
        monkeypatch.setenv("PRIMUS_TURBO_GEMM_BACKEND", "fp8:NO_SUCH_BACKEND")
        with pytest.raises(KeyError):
            GlobalBackendManager.get_gemm_backend(PrecisionType.FP8)


class TestCodeSettingsOverrideEnv:
    """Code-level set_*_backend() calls must take precedence over env vars
    per the documented priority order in GlobalBackendManager.__doc__."""

    def test_set_gemm_backend_beats_env(self, monkeypatch):
        monkeypatch.setenv("PRIMUS_TURBO_GEMM_BACKEND", "ck")
        GlobalBackendManager.set_gemm_backend(BackendType.HIPBLASLT)
        assert GlobalBackendManager.get_gemm_backend(PrecisionType.FP8) == BackendType.HIPBLASLT

    def test_set_grouped_gemm_backend_beats_env(self, monkeypatch):
        monkeypatch.setenv("PRIMUS_TURBO_GROUPED_GEMM_BACKEND", "ck")
        GlobalBackendManager.set_grouped_gemm_backend(BackendType.HIPBLASLT)
        assert GlobalBackendManager.get_grouped_gemm_backend(PrecisionType.FP8) == BackendType.HIPBLASLT

    def test_set_auto_tune_beats_env(self, monkeypatch):
        monkeypatch.setenv("PRIMUS_TURBO_AUTO_TUNE", "1")
        GlobalBackendManager.set_auto_tune(False)
        assert GlobalBackendManager.auto_tune_enabled() is False

    def test_set_backend_none_reverts_to_env(self, monkeypatch):
        """Passing backend=None clears code override and falls back to env."""
        monkeypatch.setenv("PRIMUS_TURBO_GEMM_BACKEND", "ck")
        GlobalBackendManager.set_gemm_backend(BackendType.HIPBLASLT)
        assert GlobalBackendManager.get_gemm_backend(PrecisionType.FP8) == BackendType.HIPBLASLT
        GlobalBackendManager.set_gemm_backend(None)
        assert GlobalBackendManager.get_gemm_backend(PrecisionType.FP8) == BackendType.CK

    def test_set_grouped_gemm_backend_none_reverts_to_env(self, monkeypatch):
        monkeypatch.setenv("PRIMUS_TURBO_GROUPED_GEMM_BACKEND", "ck")
        GlobalBackendManager.set_grouped_gemm_backend(BackendType.HIPBLASLT)
        assert GlobalBackendManager.get_grouped_gemm_backend(PrecisionType.FP8) == BackendType.HIPBLASLT
        GlobalBackendManager.set_grouped_gemm_backend(None)
        assert GlobalBackendManager.get_grouped_gemm_backend(PrecisionType.FP8) == BackendType.CK


class TestExtractBackendLruCacheHonorsReset:
    """The parser is @lru_cache'd. Tests must not leak cached entries between
    env vars sharing the same value. The module-level fixture clears the cache;
    this test documents that contract."""

    def test_cache_cleared_between_tests(self):
        # Nothing is set; the autouse fixture already cleared the cache.
        # Direct call should produce a fresh computation and succeed.
        result = GlobalBackendManager._extract_backend_from_env("ck")
        assert result[PrecisionType.FP8] == BackendType.CK
