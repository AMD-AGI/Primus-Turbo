###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Backend configuration diagnostics and per-precision overrides."""

import pytest

from primus_turbo.pytorch.core.backend import BackendChoice, BackendType, GlobalBackendManager, PrecisionType


@pytest.fixture(autouse=True)
def restore_manager():
    GlobalBackendManager.reset()
    yield
    GlobalBackendManager.reset()


@pytest.mark.parametrize("family", ["gemm", "grouped_gemm", "sparse_attn"])
def test_partial_mapping_and_empty_entries(family, monkeypatch):
    monkeypatch.setenv(f"PRIMUS_TURBO_{family.upper()}_BACKEND", "fp8:triton,,")
    getter = getattr(GlobalBackendManager, f"get_{family}_backend")
    assert getter(PrecisionType.FP8) == BackendChoice(BackendType.TRITON)
    assert getter(PrecisionType.FP4) is None


@pytest.mark.parametrize("family", ["gemm", "grouped_gemm", "sparse_attn"])
def test_unknown_backend_does_not_silently_select_a_kernel(family, monkeypatch):
    monkeypatch.setenv(f"PRIMUS_TURBO_{family.upper()}_BACKEND", "not_a_backend")
    with pytest.raises(KeyError):
        getattr(GlobalBackendManager, f"get_{family}_backend")(PrecisionType.FP8)


@pytest.mark.parametrize("family", ["gemm", "grouped_gemm", "sparse_attn"])
def test_pinned_backend_cannot_also_request_tuning(family):
    with pytest.raises(AssertionError, match="Backend must be None"):
        getattr(GlobalBackendManager, f"set_{family}_backend")(BackendType.TRITON, auto_tune=True)
