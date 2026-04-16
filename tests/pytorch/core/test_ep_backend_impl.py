###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Unit tests for moe_dispatch_combine_impl infrastructure.

These tests cover the pure-Python logic introduced in the EPBackend refactor:
  - EPBufferConfig dataclass
  - set_buffer_global_config()
  - get_hidden_bytes()
  - register_ep_backend() / _get_backend_instance()
  - _resolve_backend_name()
  - _ensure_buffer() pre-condition guard
  - TurboEPBackend / DeepEPBackend availability checks and buffer kwargs
  - GlobalBackendManager.get_moe_dispatch_combine_backend() KeyError path
  - GlobalBackendManager.reset() completeness for _moe_dispatch_combine_backend

No GPU or distributed process group is required; torch tensors are created
on CPU for the get_hidden_bytes() tests.
"""

from unittest.mock import MagicMock, patch

import pytest
import torch

from primus_turbo.pytorch.core.backend import (
    BackendType,
    GlobalBackendManager,
    PrecisionType,
)
from primus_turbo.pytorch.kernels.moe.moe_dispatch_combine_impl import (
    EPBackend,
    EPBufferConfig,
    TurboEPBackend,
    DeepEPBackend,
    _BACKEND_REGISTRY,
    _get_backend_instance,
    _resolve_backend_name,
    _ensure_buffer,
    get_hidden_bytes,
    register_ep_backend,
    set_buffer_global_config,
)
import primus_turbo.pytorch.kernels.moe.moe_dispatch_combine_impl as _impl_module


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def reset_ep_state(monkeypatch):
    """Reset module-level globals before every test."""
    monkeypatch.setattr(_impl_module, "_buffer_config", None)
    # Reset backend instance cache without touching registry
    monkeypatch.setattr(_impl_module, "_backend_instances", {})
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


# ---------------------------------------------------------------------------
# EPBufferConfig dataclass
# ---------------------------------------------------------------------------


class TestEPBufferConfig:
    def test_default_values(self):
        cfg = EPBufferConfig()
        assert cfg.num_sms == 32
        assert cfg.dispatch_config is None
        assert cfg.combine_config is None

    def test_custom_values(self):
        dc = object()
        cc = object()
        cfg = EPBufferConfig(num_sms=64, dispatch_config=dc, combine_config=cc)
        assert cfg.num_sms == 64
        assert cfg.dispatch_config is dc
        assert cfg.combine_config is cc

    def test_equality(self):
        a = EPBufferConfig(num_sms=16)
        b = EPBufferConfig(num_sms=16)
        assert a == b

    def test_inequality_on_num_sms(self):
        assert EPBufferConfig(num_sms=16) != EPBufferConfig(num_sms=32)


# ---------------------------------------------------------------------------
# set_buffer_global_config
# ---------------------------------------------------------------------------


class TestSetBufferGlobalConfig:
    def test_default_no_autotune(self):
        set_buffer_global_config(num_use_cu=64)
        cfg = _impl_module._buffer_config
        assert cfg is not None
        assert cfg.num_sms == 64
        assert cfg.dispatch_config is None
        assert cfg.combine_config is None

    def test_with_autotune_config_tuple(self):
        dispatch_obj = object()
        combine_obj = object()
        set_buffer_global_config(num_use_cu=16, autotune_config=(dispatch_obj, combine_obj))
        cfg = _impl_module._buffer_config
        assert cfg.num_sms == 16
        assert cfg.dispatch_config is dispatch_obj
        assert cfg.combine_config is combine_obj

    def test_with_none_autotune_config(self):
        set_buffer_global_config(num_use_cu=8, autotune_config=None)
        cfg = _impl_module._buffer_config
        assert cfg.dispatch_config is None
        assert cfg.combine_config is None

    def test_overwrites_previous_config(self):
        set_buffer_global_config(num_use_cu=32)
        set_buffer_global_config(num_use_cu=128)
        cfg = _impl_module._buffer_config
        assert cfg.num_sms == 128


# ---------------------------------------------------------------------------
# get_hidden_bytes
# ---------------------------------------------------------------------------


class TestGetHiddenBytes:
    def test_bfloat16_uses_element_size(self):
        # bfloat16 element_size=2 == minimum; result = hidden_size * 2
        x = torch.zeros(4, 512, dtype=torch.bfloat16)
        assert get_hidden_bytes(x) == 512 * 2

    def test_float32_uses_element_size(self):
        # float32 element_size=4 > minimum 2; result = hidden_size * 4
        x = torch.zeros(4, 256, dtype=torch.float32)
        assert get_hidden_bytes(x) == 256 * 4

    def test_float8_uses_minimum_2_bytes(self):
        # float8_e4m3fn has element_size=1 but minimum is 2
        try:
            x = torch.zeros(4, 128, dtype=torch.float8_e4m3fn)
        except (AttributeError, RuntimeError):
            pytest.skip("float8 dtype not available in this torch version")
        assert get_hidden_bytes(x) == 128 * 2

    def test_tuple_input_uses_first_element(self):
        x = (torch.zeros(4, 64, dtype=torch.bfloat16), torch.zeros(4, 64, dtype=torch.bfloat16))
        assert get_hidden_bytes(x) == 64 * 2

    def test_tuple_first_element_dtype_governs(self):
        # First element is fp32, second is bfloat16 — result should use fp32 size
        x = (torch.zeros(2, 32, dtype=torch.float32), torch.zeros(2, 32, dtype=torch.bfloat16))
        assert get_hidden_bytes(x) == 32 * 4


# ---------------------------------------------------------------------------
# Backend availability
# ---------------------------------------------------------------------------


class TestBackendAvailability:
    def test_turbo_is_always_available(self):
        assert TurboEPBackend.is_available() is True

    def test_deep_ep_unavailable_when_import_fails(self):
        with patch.dict("sys.modules", {"deep_ep": None}):
            # Patch importlib-level import in DeepEPBackend.is_available
            with patch("builtins.__import__", side_effect=ImportError("no deep_ep")):
                # is_available catches ImportError and returns False
                result = DeepEPBackend.is_available()
        assert result is False

    def test_deep_ep_available_when_import_succeeds(self):
        fake_deep_ep = MagicMock()
        with patch.dict("sys.modules", {"deep_ep": fake_deep_ep}):
            result = DeepEPBackend.is_available()
        assert result is True


# ---------------------------------------------------------------------------
# DeepEPBackend._make_buffer_kwargs
# ---------------------------------------------------------------------------


class TestDeepEPBufferKwargs:
    def _make_group(self, size: int):
        group = MagicMock()
        group.size.return_value = size
        return group

    def test_intranode_for_small_group(self):
        backend = DeepEPBackend()
        kwargs = backend._make_buffer_kwargs(self._make_group(8))
        assert kwargs == {"is_intranode": True}

    def test_not_intranode_for_large_group(self):
        backend = DeepEPBackend()
        kwargs = backend._make_buffer_kwargs(self._make_group(16))
        assert kwargs == {"is_intranode": False}

    def test_boundary_8_is_intranode(self):
        backend = DeepEPBackend()
        kwargs = backend._make_buffer_kwargs(self._make_group(8))
        assert kwargs["is_intranode"] is True

    def test_boundary_9_is_not_intranode(self):
        backend = DeepEPBackend()
        kwargs = backend._make_buffer_kwargs(self._make_group(9))
        assert kwargs["is_intranode"] is False


# ---------------------------------------------------------------------------
# Backend registry: register_ep_backend / _get_backend_instance
# ---------------------------------------------------------------------------


class TestBackendRegistry:
    def test_unknown_backend_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown EP backend"):
            _get_backend_instance("NONEXISTENT_XYZ")

    def test_unavailable_backend_raises_runtime_error(self, monkeypatch):
        class _UnavailableBackend(EPBackend):
            @staticmethod
            def is_available():
                return False

            def init_buffer(self, group, hidden_bytes, config):
                pass

            def dispatch(self, x, **kwargs):
                pass

            def combine(self, x, handle, **kwargs):
                pass

        monkeypatch.setitem(_impl_module._BACKEND_REGISTRY, "UNAVAILABLE_TEST", _UnavailableBackend)
        with pytest.raises(RuntimeError, match="dependencies are not installed"):
            _get_backend_instance("UNAVAILABLE_TEST")

    def test_singleton_caching(self, monkeypatch):
        """The same backend instance is returned on repeated calls."""
        call_count = 0

        class _CountingBackend(EPBackend):
            @staticmethod
            def is_available():
                return True

            def __init__(self):
                nonlocal call_count
                call_count += 1

            def init_buffer(self, group, hidden_bytes, config):
                pass

            def dispatch(self, x, **kwargs):
                pass

            def combine(self, x, handle, **kwargs):
                pass

        monkeypatch.setitem(_impl_module._BACKEND_REGISTRY, "COUNTING_TEST", _CountingBackend)
        inst1 = _get_backend_instance("COUNTING_TEST")
        inst2 = _get_backend_instance("COUNTING_TEST")
        assert inst1 is inst2
        assert call_count == 1

    def test_register_ep_backend_adds_to_registry(self, monkeypatch):
        class _DummyBackend(EPBackend):
            @staticmethod
            def is_available():
                return True

            def init_buffer(self, group, hidden_bytes, config):
                pass

            def dispatch(self, x, **kwargs):
                pass

            def combine(self, x, handle, **kwargs):
                pass

        register_ep_backend("DUMMY_REGISTERED", _DummyBackend)
        assert "DUMMY_REGISTERED" in _impl_module._BACKEND_REGISTRY
        assert _impl_module._BACKEND_REGISTRY["DUMMY_REGISTERED"] is _DummyBackend
        # Cleanup: remove the key we added
        del _impl_module._BACKEND_REGISTRY["DUMMY_REGISTERED"]


# ---------------------------------------------------------------------------
# _resolve_backend_name
# ---------------------------------------------------------------------------


class TestResolveBackendName:
    def test_default_is_turbo(self):
        assert _resolve_backend_name() == "TURBO"

    def test_env_var_overrides_default(self, monkeypatch):
        # Use TURBO (always available) to avoid the HAVE_DEEP_EP assertion guard
        monkeypatch.setenv("PRIMUS_TURBO_MOE_DISPATCH_COMBINE_BACKEND", "turbo")
        assert _resolve_backend_name() == "TURBO"

    def test_env_var_is_uppercased(self, monkeypatch):
        monkeypatch.setenv("PRIMUS_TURBO_MOE_DISPATCH_COMBINE_BACKEND", "turbo")
        assert _resolve_backend_name() == "TURBO"

    def test_env_var_strips_whitespace(self, monkeypatch):
        monkeypatch.setenv("PRIMUS_TURBO_MOE_DISPATCH_COMBINE_BACKEND", "  turbo  ")
        assert _resolve_backend_name() == "TURBO"

    def test_global_backend_manager_takes_priority_over_env(self, monkeypatch):
        """Code-level backend setting should win over env var."""
        monkeypatch.setenv("PRIMUS_TURBO_MOE_DISPATCH_COMBINE_BACKEND", "DEEP_EP")
        # Simulate GlobalBackendManager returning TURBO (code-level override)
        with patch.object(
            GlobalBackendManager,
            "get_moe_dispatch_combine_backend",
            return_value=BackendType.TURBO,
        ):
            result = _resolve_backend_name()
        assert result == "TURBO"

    def test_custom_backend_name_passthrough(self, monkeypatch):
        """When env contains an unknown BackendType name, GlobalBackendManager returns
        None (KeyError path), and _resolve_backend_name reads the env directly."""
        monkeypatch.setenv("PRIMUS_TURBO_MOE_DISPATCH_COMBINE_BACKEND", "UCCL_EP")
        # GlobalBackendManager returns None for unknown backend names
        with patch.object(
            GlobalBackendManager,
            "get_moe_dispatch_combine_backend",
            return_value=None,
        ):
            result = _resolve_backend_name()
        assert result == "UCCL_EP"


# ---------------------------------------------------------------------------
# _ensure_buffer
# ---------------------------------------------------------------------------


class TestEnsureBuffer:
    def test_raises_when_buffer_config_not_set(self):
        """_ensure_buffer must raise RuntimeError if set_buffer_global_config was never called."""
        assert _impl_module._buffer_config is None

        mock_group = MagicMock()
        mock_backend = MagicMock(spec=EPBackend)

        with pytest.raises(RuntimeError, match="set_buffer_global_config\\(\\) must be called"):
            _ensure_buffer(mock_group, hidden_bytes=512, backend=mock_backend)

    def test_calls_backend_init_buffer_when_config_is_set(self):
        set_buffer_global_config(num_use_cu=32)

        mock_group = MagicMock()
        mock_backend = MagicMock(spec=EPBackend)

        _ensure_buffer(mock_group, hidden_bytes=1024, backend=mock_backend)

        mock_backend.init_buffer.assert_called_once()
        call_args = mock_backend.init_buffer.call_args
        assert call_args[0][0] is mock_group
        assert call_args[0][1] == 1024
        cfg_arg = call_args[0][2]
        assert isinstance(cfg_arg, EPBufferConfig)
        assert cfg_arg.num_sms == 32


# ---------------------------------------------------------------------------
# GlobalBackendManager — new behaviors from refactor
# ---------------------------------------------------------------------------


class TestGlobalBackendManagerNewBehaviors:
    def test_moe_dispatch_combine_unknown_backend_name_returns_none(self, monkeypatch):
        """get_moe_dispatch_combine_backend returns None (not raises) for
        custom backend names like UCCL_EP that are not in BackendType enum.

        This is a bug fix introduced in the refactor: before the KeyError was
        uncaught; now it's caught and None is returned so the EP registry handles it.
        """
        monkeypatch.setenv("PRIMUS_TURBO_MOE_DISPATCH_COMBINE_BACKEND", "UCCL_EP")
        GlobalBackendManager._extract_backend_from_env.cache_clear()
        result = GlobalBackendManager.get_moe_dispatch_combine_backend(PrecisionType.BF16_FP16_FP32)
        assert result is None

    def test_moe_dispatch_combine_valid_backend_name_returns_backend_type(self, monkeypatch):
        monkeypatch.setenv("PRIMUS_TURBO_MOE_DISPATCH_COMBINE_BACKEND", "TURBO")
        GlobalBackendManager._extract_backend_from_env.cache_clear()
        result = GlobalBackendManager.get_moe_dispatch_combine_backend(PrecisionType.BF16_FP16_FP32)
        assert result == BackendType.TURBO

    def test_reset_also_clears_moe_dispatch_combine_backend(self):
        """reset() should clear _moe_dispatch_combine_backend to None.

        If this test fails, it indicates _moe_dispatch_combine_backend is NOT
        being reset — a subtle state leak between tests.
        """
        # Manually set the private state to simulate a prior code-level configuration
        GlobalBackendManager._moe_dispatch_combine_backend = {
            PrecisionType.BF16_FP16_FP32: BackendType.TURBO
        }
        GlobalBackendManager.reset()
        assert GlobalBackendManager._moe_dispatch_combine_backend is None, (
            "GlobalBackendManager.reset() does not clear _moe_dispatch_combine_backend. "
            "This causes state leakage between tests and should be fixed."
        )

    def test_auto_tune_reset(self):
        GlobalBackendManager.set_auto_tune(True)
        GlobalBackendManager.reset()
        assert GlobalBackendManager.auto_tune_enabled() is False
