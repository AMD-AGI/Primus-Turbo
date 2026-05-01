###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Unit tests for the MoE Expert-Parallel backend registry.

PR #297 introduced an extensible backend registry
(``register_ep_backend`` / ``_get_backend_instance`` / ``_resolve_backend_name``)
plus the ``EPBufferConfig`` dataclass and the ``get_hidden_bytes`` helper in
``primus_turbo.pytorch.kernels.moe.moe_dispatch_combine_impl``.

The original change was big and only had multi-GPU integration tests
(``tests/pytorch/modules/test_token_dispatcher.py``) for the happy path.
The pure-Python registry, hidden-bytes math and resolution priority can
regress silently — especially because production callers like the
DeepEP token dispatcher rely on these helpers to pick the right buffer
size and the right backend without any GPU involvement.

These tests run on CPU only and isolate global module state so they are
safe to run in parallel with other tests.
"""

import os

import pytest
import torch

from primus_turbo.pytorch.core.backend import (
    HAVE_DEEP_EP,
    BackendType,
    GlobalBackendManager,
    PrecisionType,
)
from primus_turbo.pytorch.kernels.moe import moe_dispatch_combine_impl as ep_mod
from primus_turbo.pytorch.kernels.moe.moe_dispatch_combine_impl import (
    _DEFAULT_BUFFER_CONFIG,
    EPBufferConfig,
    _get_backend_instance,
    _resolve_backend_name,
    clear_backend_instances,
    get_hidden_bytes,
    register_ep_backend,
    set_buffer_global_config,
)


@pytest.fixture(autouse=True)
def _isolate_state(monkeypatch):
    """Reset global module state between tests.

    * Backend instance cache.
    * ``GlobalBackendManager`` settings.
    * ``PRIMUS_TURBO_MOE_DISPATCH_COMBINE_BACKEND`` env var.
    * Saved buffer config (``_DEFAULT_BUFFER_CONFIG``).
    """
    clear_backend_instances()
    GlobalBackendManager.reset()
    GlobalBackendManager._extract_backend_from_env.cache_clear()
    monkeypatch.delenv("PRIMUS_TURBO_MOE_DISPATCH_COMBINE_BACKEND", raising=False)
    # Snapshot & restore the module-level _buffer_config that
    # set_buffer_global_config mutates.
    saved = ep_mod._buffer_config
    saved_registry = dict(ep_mod._BACKEND_REGISTRY)
    yield
    ep_mod._buffer_config = saved
    ep_mod._BACKEND_REGISTRY.clear()
    ep_mod._BACKEND_REGISTRY.update(saved_registry)
    clear_backend_instances()
    GlobalBackendManager.reset()
    GlobalBackendManager._extract_backend_from_env.cache_clear()


# ---------------------------------------------------------------------------
# get_hidden_bytes
# ---------------------------------------------------------------------------


class TestGetHiddenBytes:

    def test_bf16_uses_actual_element_size(self):
        """bf16 element_size = 2 → matches the floor; result == hidden * 2."""
        x = torch.empty(7, 64, dtype=torch.bfloat16)
        assert get_hidden_bytes(x) == 64 * 2

    def test_fp32_uses_actual_element_size(self):
        x = torch.empty(7, 64, dtype=torch.float32)
        assert get_hidden_bytes(x) == 64 * 4

    def test_uint8_floored_to_two_bytes(self):
        """An fp8 tensor (element_size=1) must still report 2 bytes/elem.

        Without this clamp the buffer for fp8 dispatches would be half the
        size required by the bf16 fallback path and silently corrupt MoE
        outputs.
        """
        x = torch.empty(7, 64, dtype=torch.uint8)
        assert x.element_size() == 1
        assert get_hidden_bytes(x) == 64 * 2

    def test_tuple_input_uses_first_element(self):
        """``dispatch`` may pass a (data, scale) tuple; hidden_bytes is
        derived from the first member only.
        """
        data = torch.empty(7, 128, dtype=torch.bfloat16)
        scale = torch.empty(7, 1, dtype=torch.float32)
        assert get_hidden_bytes((data, scale)) == 128 * 2


# ---------------------------------------------------------------------------
# EPBufferConfig and set_buffer_global_config
# ---------------------------------------------------------------------------


class TestEPBufferConfig:

    def test_defaults(self):
        cfg = EPBufferConfig()
        assert cfg.num_sms == 32
        assert cfg.dispatch_config is None
        assert cfg.combine_config is None

    def test_module_default_constant_matches_class_defaults(self):
        assert _DEFAULT_BUFFER_CONFIG.num_sms == 32
        assert _DEFAULT_BUFFER_CONFIG.dispatch_config is None
        assert _DEFAULT_BUFFER_CONFIG.combine_config is None

    def test_set_buffer_global_config_with_autotune_tuple(self):
        sentinel_dispatch = object()
        sentinel_combine = object()
        set_buffer_global_config(
            num_use_cu=64, autotune_config=(sentinel_dispatch, sentinel_combine)
        )
        cfg = ep_mod._buffer_config
        assert cfg.num_sms == 64
        assert cfg.dispatch_config is sentinel_dispatch
        assert cfg.combine_config is sentinel_combine

    def test_set_buffer_global_config_without_autotune_tuple(self):
        set_buffer_global_config(num_use_cu=24)
        cfg = ep_mod._buffer_config
        assert cfg.num_sms == 24
        assert cfg.dispatch_config is None
        assert cfg.combine_config is None


# ---------------------------------------------------------------------------
# Backend registry & lazy instance creation
# ---------------------------------------------------------------------------


class _FakeBackend:
    """Structural ``EPBackend`` substitute for registry tests.

    Records lifecycle events so tests can assert that the dispatcher cached
    the singleton correctly (one ``__init__`` per backend name) and that
    ``init_buffer`` saw the expected ``EPBufferConfig``.
    """

    init_count = 0
    last_init_buffer_args = None

    def __init__(self):
        type(self).init_count += 1

    @staticmethod
    def is_available() -> bool:
        return True

    def init_buffer(self, group, hidden_bytes, config):
        type(self).last_init_buffer_args = (group, hidden_bytes, config)

    def dispatch(self, x, **kwargs):  # pragma: no cover - registry tests don't call it
        raise AssertionError("dispatch should not be called in registry tests")

    def combine(self, x, handle, **kwargs):  # pragma: no cover
        raise AssertionError("combine should not be called in registry tests")


class _UnavailableBackend(_FakeBackend):

    @staticmethod
    def is_available() -> bool:
        return False


class TestBackendRegistry:

    def test_register_and_get(self):
        _FakeBackend.init_count = 0
        register_ep_backend("FAKE", _FakeBackend)
        inst1 = _get_backend_instance("FAKE")
        inst2 = _get_backend_instance("FAKE")
        assert isinstance(inst1, _FakeBackend)
        # Singleton — second lookup must reuse the cached instance.
        assert inst1 is inst2
        assert _FakeBackend.init_count == 1

    def test_get_unknown_backend_raises(self):
        with pytest.raises(ValueError, match="Unknown EP backend"):
            _get_backend_instance("DOES_NOT_EXIST")

    def test_get_unavailable_backend_raises(self):
        register_ep_backend("UNAVAIL", _UnavailableBackend)
        with pytest.raises(RuntimeError, match="not installed"):
            _get_backend_instance("UNAVAIL")

    def test_clear_backend_instances_drops_singletons(self):
        _FakeBackend.init_count = 0
        register_ep_backend("FAKE", _FakeBackend)
        first = _get_backend_instance("FAKE")
        clear_backend_instances()
        second = _get_backend_instance("FAKE")
        assert first is not second
        assert _FakeBackend.init_count == 2

    def test_default_registry_has_turbo_and_deepep_keys(self):
        """Smoke test: the default registry must always advertise the
        in-tree TURBO backend and the DEEP_EP slot, otherwise the resolver
        would have nothing to dispatch to in production.
        """
        assert "TURBO" in ep_mod._BACKEND_REGISTRY
        assert "DEEP_EP" in ep_mod._BACKEND_REGISTRY


# ---------------------------------------------------------------------------
# _resolve_backend_name priority
# ---------------------------------------------------------------------------


class TestResolveBackendName:

    def test_default_is_turbo(self):
        assert _resolve_backend_name() == "TURBO"

    @pytest.mark.skipif(
        not HAVE_DEEP_EP,
        reason="get_moe_dispatch_combine_backend asserts DeepEP is importable when env=DEEP_EP",
    )
    def test_env_var_override(self, monkeypatch):
        monkeypatch.setenv("PRIMUS_TURBO_MOE_DISPATCH_COMBINE_BACKEND", "deep_ep")
        assert _resolve_backend_name() == "DEEP_EP"

    def test_env_var_strips_and_uppercases(self, monkeypatch):
        monkeypatch.setenv("PRIMUS_TURBO_MOE_DISPATCH_COMBINE_BACKEND", "  uccl_ep  ")
        # Custom names not in BackendType are passed through verbatim
        # (uppercased) so the registry can resolve them.
        assert _resolve_backend_name() == "UCCL_EP"

    def test_global_backend_manager_overrides_env(self, monkeypatch):
        """Code-level setting beats env var (PR #305 priority contract)."""
        monkeypatch.setenv("PRIMUS_TURBO_MOE_DISPATCH_COMBINE_BACKEND", "deep_ep")
        # Initialize the dict so the manager treats it as set.
        GlobalBackendManager._moe_dispatch_combine_backend = {p: BackendType.TURBO for p in PrecisionType}
        try:
            assert _resolve_backend_name() == "TURBO"
        finally:
            GlobalBackendManager._moe_dispatch_combine_backend = None

    def test_env_var_unknown_name_falls_through_to_string(self, monkeypatch):
        """A custom backend name that isn't a BackendType enum value must
        still be returned as the literal env-var string so a downstream
        ``register_ep_backend("UCCL_EP", ...)`` can resolve it.
        """
        monkeypatch.setenv("PRIMUS_TURBO_MOE_DISPATCH_COMBINE_BACKEND", "uccl_ep")
        # GlobalBackendManager.get_moe_dispatch_combine_backend must return
        # None for unknown names; the resolver then reads the env var
        # directly.
        assert (
            GlobalBackendManager.get_moe_dispatch_combine_backend(PrecisionType.BF16_FP16_FP32) is None
        )
        assert _resolve_backend_name() == "UCCL_EP"
