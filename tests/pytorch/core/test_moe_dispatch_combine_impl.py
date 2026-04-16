###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Unit tests for moe_dispatch_combine_impl.py.

These tests cover the new EPBackend abstraction, backend registry, and
helper utilities introduced in the EP-backend refactor. They do **not**
require a GPU or any compiled C extension — all external dependencies are
stubbed out at import time.
"""

import importlib.util
import os
import sys
import types
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

# ---------------------------------------------------------------------------
# One-time module loading — done at import time so sys.modules stays stable
# ---------------------------------------------------------------------------

_WORKSPACE = Path(__file__).parent.parent.parent.parent


def _stub_modules() -> dict:
    """Return stub modules needed to load backend.py and moe_dispatch_combine_impl.py."""
    common = types.ModuleType("primus_turbo.common")
    logger_mod = types.ModuleType("primus_turbo.common.logger")

    class _Logger:
        def warning(self, msg, *args, **kwargs):
            pass

    logger_mod.logger = _Logger()

    return {
        "primus_turbo": types.ModuleType("primus_turbo"),
        "primus_turbo.common": common,
        "primus_turbo.common.logger": logger_mod,
        "primus_turbo.pytorch": types.ModuleType("primus_turbo.pytorch"),
        "primus_turbo.pytorch.core": types.ModuleType("primus_turbo.pytorch.core"),
    }


def _load(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    m = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = m
    spec.loader.exec_module(m)
    return m


# Inject stubs permanently (for this process)
for _k, _v in _stub_modules().items():
    sys.modules.setdefault(_k, _v)

_backend_mod = _load(
    "primus_turbo.pytorch.core.backend",
    _WORKSPACE / "primus_turbo/pytorch/core/backend.py",
)

_impl_mod = _load(
    "primus_turbo.pytorch.kernels.moe.moe_dispatch_combine_impl",
    _WORKSPACE / "primus_turbo/pytorch/kernels/moe/moe_dispatch_combine_impl.py",
)

GlobalBackendManager = _backend_mod.GlobalBackendManager
PrecisionType = _backend_mod.PrecisionType
BackendType = _backend_mod.BackendType


# ---------------------------------------------------------------------------
# Shared reset fixture
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset():
    """Restore mutable singleton state around every test."""
    GlobalBackendManager.reset()
    GlobalBackendManager._extract_backend_from_env.cache_clear()

    _impl_mod._buffer_config = None
    _impl_mod._backend_instances.clear()
    for key in list(_impl_mod._BACKEND_REGISTRY.keys()):
        if key not in ("TURBO", "DEEP_EP"):
            del _impl_mod._BACKEND_REGISTRY[key]

    yield

    GlobalBackendManager.reset()
    GlobalBackendManager._extract_backend_from_env.cache_clear()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ep_backend(*, available: bool = True):
    """Return a minimal EPBackend subclass."""

    class _FakeBackend(_impl_mod.EPBackend):
        @staticmethod
        def is_available() -> bool:
            return available

        def init_buffer(self, group, hidden_bytes, config):
            pass

        def dispatch(self, x, **kwargs):
            return x, None, None, None, None

        def combine(self, x, handle, **kwargs):
            return x, None

    return _FakeBackend


# ---------------------------------------------------------------------------
# EPBufferConfig
# ---------------------------------------------------------------------------


class TestEPBufferConfig:
    def test_defaults(self):
        cfg = _impl_mod.EPBufferConfig()
        assert cfg.num_sms == 32
        assert cfg.dispatch_config is None
        assert cfg.combine_config is None

    def test_custom_values(self):
        cfg = _impl_mod.EPBufferConfig(num_sms=64, dispatch_config="d", combine_config="c")
        assert cfg.num_sms == 64
        assert cfg.dispatch_config == "d"
        assert cfg.combine_config == "c"


# ---------------------------------------------------------------------------
# set_buffer_global_config
# ---------------------------------------------------------------------------


class TestSetBufferGlobalConfig:
    def test_default_sms_no_autotune(self):
        _impl_mod.set_buffer_global_config(num_use_cu=16)
        assert _impl_mod._buffer_config is not None
        assert _impl_mod._buffer_config.num_sms == 16
        assert _impl_mod._buffer_config.dispatch_config is None
        assert _impl_mod._buffer_config.combine_config is None

    def test_autotune_config_unpacked(self):
        _impl_mod.set_buffer_global_config(num_use_cu=8, autotune_config=("dcfg", "ccfg"))
        assert _impl_mod._buffer_config.num_sms == 8
        assert _impl_mod._buffer_config.dispatch_config == "dcfg"
        assert _impl_mod._buffer_config.combine_config == "ccfg"

    def test_repeated_call_overwrites(self):
        _impl_mod.set_buffer_global_config(num_use_cu=16)
        _impl_mod.set_buffer_global_config(num_use_cu=32)
        assert _impl_mod._buffer_config.num_sms == 32


# ---------------------------------------------------------------------------
# get_hidden_bytes
# ---------------------------------------------------------------------------


class TestGetHiddenBytes:
    def test_bf16_uses_2_bytes(self):
        t = torch.zeros(10, 4096, dtype=torch.bfloat16)
        assert _impl_mod.get_hidden_bytes(t) == 4096 * 2

    def test_fp32_uses_4_bytes(self):
        t = torch.zeros(10, 512, dtype=torch.float32)
        assert _impl_mod.get_hidden_bytes(t) == 512 * 4

    def test_fp8_is_clamped_to_2_bytes(self):
        try:
            t = torch.zeros(10, 256, dtype=torch.float8_e4m3fn)
        except (AttributeError, RuntimeError):
            pytest.skip("torch.float8_e4m3fn not available in this PyTorch build")
        # fp8 is 1 byte per element; the function enforces a minimum of 2
        assert _impl_mod.get_hidden_bytes(t) == 256 * 2

    def test_tuple_input_uses_first_tensor(self):
        t0 = torch.zeros(10, 128, dtype=torch.bfloat16)
        t1 = torch.zeros(10, 64, dtype=torch.float32)
        assert _impl_mod.get_hidden_bytes((t0, t1)) == 128 * 2


# ---------------------------------------------------------------------------
# Backend registry: register_ep_backend / _get_backend_instance
# ---------------------------------------------------------------------------


class TestEPBackendRegistry:
    def test_register_and_retrieve(self):
        Cls = _make_ep_backend()
        _impl_mod.register_ep_backend("MY_BACKEND", Cls)
        assert "MY_BACKEND" in _impl_mod._BACKEND_REGISTRY
        instance = _impl_mod._get_backend_instance("MY_BACKEND")
        assert isinstance(instance, Cls)

    def test_singleton_behavior(self):
        Cls = _make_ep_backend()
        _impl_mod.register_ep_backend("SINGLETON", Cls)
        inst1 = _impl_mod._get_backend_instance("SINGLETON")
        inst2 = _impl_mod._get_backend_instance("SINGLETON")
        assert inst1 is inst2

    def test_unknown_backend_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown EP backend"):
            _impl_mod._get_backend_instance("DOES_NOT_EXIST_XYZ")

    def test_unavailable_backend_raises_runtime_error(self):
        UnavailCls = _make_ep_backend(available=False)
        _impl_mod.register_ep_backend("UNAVAIL", UnavailCls)
        with pytest.raises(RuntimeError, match="dependencies are not installed"):
            _impl_mod._get_backend_instance("UNAVAIL")

    def test_turbo_always_registered(self):
        assert "TURBO" in _impl_mod._BACKEND_REGISTRY

    def test_overwrite_registration(self):
        ClsV1 = _make_ep_backend()
        ClsV2 = _make_ep_backend()
        _impl_mod.register_ep_backend("OVERRIDE", ClsV1)
        _impl_mod.register_ep_backend("OVERRIDE", ClsV2)
        assert _impl_mod._BACKEND_REGISTRY["OVERRIDE"] is ClsV2


# ---------------------------------------------------------------------------
# _resolve_backend_name
# ---------------------------------------------------------------------------


class TestResolveBackendName:
    def test_default_is_turbo(self):
        assert _impl_mod._resolve_backend_name() == "TURBO"

    def test_env_var_sets_name(self):
        # Use TURBO which is always available; DEEP_EP requires the deep_ep package
        with patch.dict(os.environ, {"PRIMUS_TURBO_MOE_DISPATCH_COMBINE_BACKEND": "TURBO"}):
            assert _impl_mod._resolve_backend_name() == "TURBO"

    def test_custom_ep_name_passes_through(self):
        """Names not in BackendType (e.g. UCCL_EP) must be returned as-is (uppercased)."""
        with patch.dict(os.environ, {"PRIMUS_TURBO_MOE_DISPATCH_COMBINE_BACKEND": "uccl_ep"}):
            result = _impl_mod._resolve_backend_name()
        assert result == "UCCL_EP"

    def test_env_var_is_uppercased(self):
        with patch.dict(os.environ, {"PRIMUS_TURBO_MOE_DISPATCH_COMBINE_BACKEND": "turbo"}):
            assert _impl_mod._resolve_backend_name() == "TURBO"


# ---------------------------------------------------------------------------
# get_moe_dispatch_combine_backend — KeyError path guarded by PR
# ---------------------------------------------------------------------------


class TestGetMoeDispatchCombineBackendKeyError:
    """The PR added a try/except KeyError so that custom EP backend names
    (e.g. UCCL_EP) silently return None from GlobalBackendManager instead of
    crashing, allowing the impl registry to handle them.
    """

    def test_unknown_ep_name_returns_none(self):
        with patch.dict(
            os.environ, {"PRIMUS_TURBO_MOE_DISPATCH_COMBINE_BACKEND": "UCCL_EP"}
        ):
            GlobalBackendManager._extract_backend_from_env.cache_clear()
            result = GlobalBackendManager.get_moe_dispatch_combine_backend(
                PrecisionType.BF16_FP16_FP32
            )
        assert result is None, f"Expected None for unknown EP name, got {result}"

    def test_known_backend_type_still_resolves(self):
        with patch.dict(
            os.environ, {"PRIMUS_TURBO_MOE_DISPATCH_COMBINE_BACKEND": "TURBO"}
        ):
            GlobalBackendManager._extract_backend_from_env.cache_clear()
            result = GlobalBackendManager.get_moe_dispatch_combine_backend(
                PrecisionType.BF16_FP16_FP32
            )
        assert result == BackendType.TURBO

    def test_no_env_var_returns_none(self):
        result = GlobalBackendManager.get_moe_dispatch_combine_backend(
            PrecisionType.BF16_FP16_FP32
        )
        assert result is None

    def test_precision_not_in_map_returns_none_with_warning(self):
        """When the env var is set but the queried precision is absent (format 2/3),
        the method should return None after emitting a warning — not crash.
        """
        # fp8:TURBO only covers FP8; querying FP4 returns None
        with patch.dict(
            os.environ, {"PRIMUS_TURBO_MOE_DISPATCH_COMBINE_BACKEND": "fp8:TURBO"}
        ):
            GlobalBackendManager._extract_backend_from_env.cache_clear()
            result = GlobalBackendManager.get_moe_dispatch_combine_backend(PrecisionType.FP4)
        assert result is None


# ---------------------------------------------------------------------------
# _ensure_buffer
# ---------------------------------------------------------------------------


class TestEnsureBuffer:
    def test_raises_without_config(self):
        Cls = _make_ep_backend()
        _impl_mod.register_ep_backend("ENSURE_TEST", Cls)
        backend = _impl_mod._get_backend_instance("ENSURE_TEST")

        assert _impl_mod._buffer_config is None
        fake_group = types.SimpleNamespace(size=lambda: 2)
        with pytest.raises(RuntimeError, match="set_buffer_global_config"):
            _impl_mod._ensure_buffer(fake_group, 128, backend)

    def test_no_raise_after_config_set(self):
        _impl_mod.set_buffer_global_config(num_use_cu=32)

        Cls = _make_ep_backend()
        _impl_mod.register_ep_backend("ENSURE_OK", Cls)
        backend = _impl_mod._get_backend_instance("ENSURE_OK")

        fake_group = types.SimpleNamespace(size=lambda: 2)
        _impl_mod._ensure_buffer(fake_group, 128, backend)
