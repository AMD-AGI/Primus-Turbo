###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""The gfx1250 Gluon gate and the fallbacks it selects (host-only: no kernel is launched)."""

import pytest

import primus_turbo.common.envs as envs
from primus_turbo.gluon import gate


@pytest.fixture
def gate_state(monkeypatch):
    """A fresh gate: gfx1250 device, Triton 3.9, cdna5 importable, PRIMUS_GLUON unset."""
    monkeypatch.delenv("PRIMUS_GLUON", raising=False)
    envs.reset_cache("PRIMUS_GLUON")
    monkeypatch.setattr("primus_turbo.pytorch.core.utils.is_gfx1250", lambda: True)
    monkeypatch.setattr(gate, "triton_version", lambda: (3, 9))
    gate.gluon_ok.cache_clear()
    yield monkeypatch
    envs.reset_cache("PRIMUS_GLUON")
    gate.gluon_ok.cache_clear()


def _cdna5_importable():
    try:
        from triton.experimental.gluon.language.amd.cdna5 import tdm  # noqa: F401
    except Exception:
        return False
    return True


def test_old_triton_closes_the_gate(gate_state):
    gate_state.setattr(gate, "triton_version", lambda: (3, 8))
    assert not gate.gluon_ok()
    assert "older" in gate.why_not()


def test_primus_gluon_zero_closes_the_gate(gate_state):
    gate_state.setenv("PRIMUS_GLUON", "0")
    envs.reset_cache("PRIMUS_GLUON")
    assert not gate.gluon_ok()
    assert gate.why_not() == "PRIMUS_GLUON=0"


def test_other_device_closes_the_gate(gate_state):
    gate_state.setattr("primus_turbo.pytorch.core.utils.is_gfx1250", lambda: False)
    assert not gate.gluon_ok()


@pytest.mark.skipif(not _cdna5_importable(), reason="Triton without gluon cdna5")
def test_gate_opens_on_gfx1250_with_new_triton(gate_state):
    assert gate.gluon_ok(), gate.why_not()


def test_require_gluon_raises_with_the_reason(gate_state):
    gate_state.setattr(gate, "triton_version", lambda: (3, 8))
    with pytest.raises(RuntimeError, match="older"):
        gate.require_gluon("the test kernel")


def test_dsv4up_falls_back_to_g90_when_the_gate_is_closed(gate_state):
    from primus_turbo.pytorch.kernels.gemm import gemm_fp8_gluon_gfx1250_impl

    gate_state.setattr(gate, "triton_version", lambda: (3, 8))
    gate_state.setattr(gemm_fp8_gluon_gfx1250_impl, "_AVAILABLE", None)
    assert not gemm_fp8_gluon_gfx1250_impl.available()
    assert not gemm_fp8_gluon_gfx1250_impl.enabled()


def test_dual_quant_uses_triton_when_the_gate_is_closed(gate_state):
    torch = pytest.importorskip("torch")
    from primus_turbo.triton.quantization import quantization_fp8blk_dual_kernel

    gate_state.setattr(gate, "triton_version", lambda: (3, 8))
    x = torch.empty(8, 8, dtype=torch.bfloat16)
    assert not quantization_fp8blk_dual_kernel._q2_use(x)
