###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Unit tests for ``primus_turbo.pytorch.core.backend`` dispatcher internals.

These tests focus on the pure-Python machinery — ``TuneCache``,
``_format_kwargs``, ``KernelBackend`` registration, dispatcher priority
ordering and error messages — none of which require a real GPU or a built
``primus_turbo._C``.

Coverage gaps motivating these tests:

* ``TuneCache`` LRU semantics (move-to-end on ``get``, capacity warning,
  oldest-eviction) had no direct test.
* ``_format_kwargs`` (added in PR #316 to make dispatcher errors readable)
  had no test, yet it is the message users see when a backend rejects
  inputs — its formatting must remain stable.
* ``AutoKernelDispatcher.dispatch`` priority order:
  user-specified > auto-tune > default > fallback, and the error paths
  for "user-specified backend not registered" / "user-specified backend
  cannot handle inputs" / "no compatible backend".
* ``GlobalBackendManager._extract_backend_from_env`` empty-pair tolerance
  and the "other:" wildcard branch.
"""

from enum import Enum
from typing import Any

import pytest
import torch

from primus_turbo.pytorch.core.backend import (
    AutoKernelDispatcher,
    BackendEntry,
    BackendType,
    GlobalBackendManager,
    KernelBackend,
    PrecisionType,
    TuneCache,
    _format_kwargs,
)


# ---------------------------------------------------------------------------
# _format_kwargs
# ---------------------------------------------------------------------------


class TestFormatKwargs:

    def test_empty(self):
        assert _format_kwargs({}) == ""

    def test_scalar_repr(self):
        out = _format_kwargs({"a": 1, "b": "x"})
        # repr() is used so strings are quoted.
        assert "a=1" in out
        assert "b='x'" in out
        assert out.count(", ") == 1

    def test_tensor_uses_shape_and_dtype(self):
        t = torch.empty(2, 3, dtype=torch.float32)
        out = _format_kwargs({"x": t})
        assert "Tensor(" in out
        # Shape must be preserved verbatim — used in user-facing error msgs.
        assert "shape=torch.Size([2, 3])" in out
        assert "dtype=torch.float32" in out

    def test_enum_uses_typename_dot_name(self):
        out = _format_kwargs({"p": PrecisionType.FP8})
        assert "p=PrecisionType.FP8" in out

    def test_mixed_kwargs_order_preserved(self):
        out = _format_kwargs({"a": 1, "b": PrecisionType.FP4, "c": "z"})
        assert out == "a=1, b=PrecisionType.FP4, c='z'"


# ---------------------------------------------------------------------------
# TuneCache
# ---------------------------------------------------------------------------


class _DummyBackend:
    """Trivial sentinel value; TuneCache stores ``Type[KernelBackend]`` but
    does not actually call into them, so ``object()`` substitutes work.
    """


class TestTuneCache:

    def test_get_missing_returns_none(self):
        cache = TuneCache(capacity=4)
        assert cache.get(("k",)) is None
        assert ("k",) not in cache
        assert len(cache) == 0

    def test_put_then_get(self):
        cache = TuneCache(capacity=4)
        cache.put("k", _DummyBackend)
        assert cache.get("k") is _DummyBackend
        assert "k" in cache
        assert len(cache) == 1

    def test_put_overwrites_existing(self):
        cache = TuneCache(capacity=4)
        cache.put("k", _DummyBackend)
        other = type("Other", (), {})
        cache.put("k", other)
        assert cache.get("k") is other
        assert len(cache) == 1

    def test_lru_eviction_oldest_removed(self):
        cache = TuneCache(capacity=2)
        a, b, c = (type(f"B{i}", (), {}) for i in range(3))
        cache.put("a", a)
        cache.put("b", b)
        cache.put("c", c)  # should evict "a"
        assert cache.get("a") is None
        assert cache.get("b") is b
        assert cache.get("c") is c

    def test_get_promotes_to_most_recent(self):
        """``get`` must call ``move_to_end`` so that recently-read keys
        survive subsequent insertions.  Without this, users running stable
        long-tail workloads would see their entries silently evicted.
        """
        cache = TuneCache(capacity=2)
        a, b, c = (type(f"B{i}", (), {}) for i in range(3))
        cache.put("a", a)
        cache.put("b", b)
        # Touch "a" so it becomes most-recent.
        assert cache.get("a") is a
        cache.put("c", c)  # should evict "b", not "a"
        assert cache.get("a") is a
        assert cache.get("b") is None
        assert cache.get("c") is c

    def test_capacity_overflow_emits_warning(self):
        cache = TuneCache(capacity=1)
        cache.put("a", _DummyBackend)
        with pytest.warns(UserWarning, match="TuneCache capacity"):
            cache.put("b", type("B", (), {}))

    def test_clear(self):
        cache = TuneCache(capacity=4)
        cache.put("a", _DummyBackend)
        cache.put("b", _DummyBackend)
        assert len(cache) == 2
        cache.clear()
        assert len(cache) == 0
        assert cache.get("a") is None


# ---------------------------------------------------------------------------
# AutoKernelDispatcher priority & error paths
# ---------------------------------------------------------------------------


def _make_backend(name: str, *, can_handle: bool = True, return_value: Any = None):
    """Build a throwaway ``KernelBackend`` whose ``execute`` returns a tag."""
    tag = return_value if return_value is not None else f"<{name}>"

    class _B(KernelBackend):
        @staticmethod
        def can_handle(**kwargs):
            return can_handle

        @staticmethod
        def execute(**kwargs):
            return tag

    _B.__name__ = name
    return _B


@pytest.fixture
def dispatcher_cls():
    """Build a fresh ``AutoKernelDispatcher`` subclass per test so cache /
    backend dictionaries are isolated.  ``__init_subclass__`` registers
    the subclass globally; we reset the manager's auto-tune flag too.
    """

    class _D(AutoKernelDispatcher):
        @classmethod
        def make_key(cls, **kwargs):
            return tuple(sorted(kwargs.items()))

    GlobalBackendManager.set_auto_tune(False)
    yield _D
    GlobalBackendManager.set_auto_tune(None)


class TestDispatchPriority:

    def test_user_specified_backend_wins(self, dispatcher_cls):
        a = _make_backend("A", return_value="a")
        b = _make_backend("B", return_value="b")
        dispatcher_cls._backends[BackendType.CK] = BackendEntry(impl=a)
        dispatcher_cls._backends[BackendType.AITER] = BackendEntry(impl=b)

        assert (
            dispatcher_cls.dispatch(default_backend_enum=BackendType.CK, user_backend_enum=BackendType.AITER)
            == "b"
        )

    def test_user_specified_unregistered_raises(self, dispatcher_cls):
        dispatcher_cls._backends[BackendType.CK] = BackendEntry(impl=_make_backend("A"))
        with pytest.raises(ValueError, match="not registered"):
            dispatcher_cls.dispatch(
                default_backend_enum=BackendType.CK,
                user_backend_enum=BackendType.AITER,
            )

    def test_user_specified_cannot_handle_raises_with_kwargs_in_message(self, dispatcher_cls):
        rejecting = _make_backend("Reject", can_handle=False)
        dispatcher_cls._backends[BackendType.AITER] = BackendEntry(impl=rejecting)

        with pytest.raises(ValueError) as exc_info:
            dispatcher_cls.dispatch(
                default_backend_enum=BackendType.AITER,
                user_backend_enum=BackendType.AITER,
                shape=(1, 2),
                dtype="bf16",
            )
        msg = str(exc_info.value)
        # PR #316: error must include the formatted kwargs so the user can
        # see *why* the requested backend rejected them.
        assert "AITER" in msg
        assert "cannot handle the given inputs" in msg
        assert "shape=(1, 2)" in msg
        assert "dtype='bf16'" in msg

    def test_default_backend_used_when_no_user_choice(self, dispatcher_cls):
        a = _make_backend("A", return_value="a")
        b = _make_backend("B", return_value="b")
        dispatcher_cls._backends[BackendType.CK] = BackendEntry(impl=a)
        dispatcher_cls._backends[BackendType.AITER] = BackendEntry(impl=b)

        assert dispatcher_cls.dispatch(default_backend_enum=BackendType.CK) == "a"

    def test_fallback_when_default_cannot_handle(self, dispatcher_cls):
        rejecting = _make_backend("Reject", can_handle=False)
        accepting = _make_backend("Accept", return_value="ok")
        dispatcher_cls._backends[BackendType.CK] = BackendEntry(impl=rejecting)
        dispatcher_cls._backends[BackendType.AITER] = BackendEntry(impl=accepting)

        # CK can't handle → fallback should pick the next compatible one.
        assert dispatcher_cls.dispatch(default_backend_enum=BackendType.CK) == "ok"

    def test_no_compatible_backend_raises(self, dispatcher_cls):
        rejecting = _make_backend("Reject", can_handle=False)
        dispatcher_cls._backends[BackendType.CK] = BackendEntry(impl=rejecting)
        dispatcher_cls._backends[BackendType.AITER] = BackendEntry(impl=rejecting)

        with pytest.raises(ValueError, match="No compatible backend found"):
            dispatcher_cls.dispatch(default_backend_enum=BackendType.CK, foo=1)

    def test_default_missing_falls_back(self, dispatcher_cls):
        """If the default backend isn't even registered, dispatch must
        gracefully fall back rather than raising a KeyError.
        """
        accepting = _make_backend("Accept", return_value="ok")
        dispatcher_cls._backends[BackendType.AITER] = BackendEntry(impl=accepting)

        # CK isn't in _backends at all.
        assert dispatcher_cls.dispatch(default_backend_enum=BackendType.CK) == "ok"


# ---------------------------------------------------------------------------
# GlobalBackendManager._extract_backend_from_env — additional edge cases
# ---------------------------------------------------------------------------


class TestExtractBackendFromEnv:

    def setup_method(self):
        GlobalBackendManager._extract_backend_from_env.cache_clear()

    def teardown_method(self):
        GlobalBackendManager._extract_backend_from_env.cache_clear()

    def test_empty_pair_tolerated(self):
        """Trailing comma must not raise — produced by some shell expansions."""
        result = GlobalBackendManager._extract_backend_from_env("fp8:ck,")
        assert result[PrecisionType.FP8] == BackendType.CK
        # Other precisions have no override → None.
        assert result[PrecisionType.FP4] is None
        assert result[PrecisionType.BF16_FP16_FP32] is None

    def test_whitespace_around_pairs_stripped(self):
        result = GlobalBackendManager._extract_backend_from_env(" fp8 : ck , fp4 : aiter ")
        assert result[PrecisionType.FP8] == BackendType.CK
        assert result[PrecisionType.FP4] == BackendType.AITER

    def test_other_wildcard_applies_to_all_unspecified(self):
        result = GlobalBackendManager._extract_backend_from_env("fp4:hipblaslt,other:aiter")
        assert result[PrecisionType.FP4] == BackendType.HIPBLASLT
        assert result[PrecisionType.FP8] == BackendType.AITER
        assert result[PrecisionType.BF16_FP16_FP32] == BackendType.AITER

    def test_single_backend_format_applies_to_all(self):
        result = GlobalBackendManager._extract_backend_from_env("triton")
        for p in PrecisionType:
            assert result[p] == BackendType.TRITON

    def test_invalid_backend_name_raises(self):
        # Note: the per-precision branch raises ``KeyError`` while the
        # single-backend branch also raises ``KeyError`` on ``BackendType[..]``.
        with pytest.raises(KeyError):
            GlobalBackendManager._extract_backend_from_env("fp8:totally_unknown_backend")

    def test_fp16_alias_maps_to_bf16_fp16_fp32(self):
        result = GlobalBackendManager._extract_backend_from_env("fp16:ck")
        assert result[PrecisionType.BF16_FP16_FP32] == BackendType.CK

    def test_fp32_alias_maps_to_bf16_fp16_fp32(self):
        result = GlobalBackendManager._extract_backend_from_env("fp32:hipblaslt")
        assert result[PrecisionType.BF16_FP16_FP32] == BackendType.HIPBLASLT
