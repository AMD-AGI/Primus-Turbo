###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import json
import os

import pytest

import primus_turbo.common.envs as envs
from primus_turbo.pytorch.core import backend as backend_mod
from primus_turbo.pytorch.core.backend import (
    AutoKernelDispatcher,
    BackendChoice,
    BackendEntry,
    BackendType,
    DispatchSource,
    GlobalBackendManager,
    KernelBackend,
    dispatch_report,
    dump_dispatch_report,
    reset_dispatch_report,
)


class _SmallOnly(KernelBackend):
    @staticmethod
    def can_handle(x, **kwargs) -> bool:
        return x < 10

    @staticmethod
    def execute(x, **kwargs):
        return "small"


class _Any(KernelBackend):
    @staticmethod
    def can_handle(x, **kwargs) -> bool:
        return True

    @staticmethod
    def execute(x, **kwargs):
        return "any"


class _FakeDispatcher(AutoKernelDispatcher):
    _backends = {
        BackendType.TRITON: BackendEntry(_Any, autotune=False),
        BackendType.CK: BackendEntry(_SmallOnly, autotune=False),
    }

    @classmethod
    def make_key(cls, x, **kwargs):
        return (x,)


_DEFAULT = BackendChoice(BackendType.CK)


def _set_env(monkeypatch, name, value):
    monkeypatch.setenv(name, value)
    envs.reset_cache(name)


@pytest.fixture(autouse=True)
def clean_state(monkeypatch):
    for name in ("PRIMUS_TURBO_DISPATCH_REPORT", "PRIMUS_TURBO_STRICT_DISPATCH", "PRIMUS_TURBO_AUTO_TUNE"):
        monkeypatch.delenv(name, raising=False)
    GlobalBackendManager.reset()
    reset_dispatch_report()
    yield
    GlobalBackendManager.reset()
    reset_dispatch_report()


def _totals():
    return {(r["dispatcher"], r["backend"], r["source"]): r["calls"] for r in dispatch_report()}


def test_records_default_user_and_fallback():
    assert _FakeDispatcher.dispatch(_DEFAULT, x=1) == "small"
    assert _FakeDispatcher.dispatch(_DEFAULT, x=2) == "small"
    assert _FakeDispatcher.dispatch(_DEFAULT, BackendChoice(BackendType.TRITON), x=1) == "any"
    assert _FakeDispatcher.dispatch(_DEFAULT, x=100) == "any"

    assert _totals() == {
        ("_FakeDispatcher", "CK", "default"): 2,
        ("_FakeDispatcher", "TRITON", "user"): 1,
        ("_FakeDispatcher", "TRITON", "fallback"): 1,
    }
    assert dispatch_report()[0]["calls"] == 2


def test_resolve_records_like_dispatch():
    assert _FakeDispatcher.resolve(BackendType.CK, x=1) == BackendType.CK
    assert _FakeDispatcher.resolve(BackendType.CK, x=100) == BackendType.TRITON
    assert _totals() == {
        ("_FakeDispatcher", "CK", "default"): 1,
        ("_FakeDispatcher", "TRITON", "fallback"): 1,
    }


def test_rejected_user_backend_is_not_recorded():
    with pytest.raises(ValueError):
        _FakeDispatcher.dispatch(_DEFAULT, BackendChoice(BackendType.CK), x=100)
    assert dispatch_report() == []


def test_autotune_source(monkeypatch):
    monkeypatch.setattr(_FakeDispatcher, "tune", classmethod(lambda cls, **kw: _Any))
    assert _FakeDispatcher.dispatch(BackendChoice(BackendType.CK, auto_tune=True), x=1) == "any"
    assert _totals() == {("_FakeDispatcher", "TRITON", "autotune"): 1}


def test_strict_dispatch_raises_on_fallback(monkeypatch):
    _set_env(monkeypatch, "PRIMUS_TURBO_STRICT_DISPATCH", "1")
    assert _FakeDispatcher.dispatch(_DEFAULT, x=1) == "small"
    with pytest.raises(RuntimeError, match="_FakeDispatcher: default backend CK .*falling back to TRITON"):
        _FakeDispatcher.dispatch(_DEFAULT, x=100)
    with pytest.raises(RuntimeError, match="STRICT_DISPATCH"):
        _FakeDispatcher.resolve(BackendType.CK, x=100)
    assert _totals() == {("_FakeDispatcher", "CK", "default"): 1}


def test_shapes_only_recorded_with_report_env(monkeypatch, tmp_path):
    _FakeDispatcher.dispatch(_DEFAULT, x=1)
    assert dispatch_report(by_shape=True) == []

    _set_env(monkeypatch, "PRIMUS_TURBO_DISPATCH_REPORT", str(tmp_path / "unused.json"))
    _FakeDispatcher.dispatch(_DEFAULT, x=1)
    _FakeDispatcher.dispatch(_DEFAULT, x=1)
    _FakeDispatcher.dispatch(_DEFAULT, x=100)
    assert dispatch_report(by_shape=True) == [
        {"dispatcher": "_FakeDispatcher", "shape": "(1,)", "backend": "CK", "source": "default", "calls": 2},
        {
            "dispatcher": "_FakeDispatcher",
            "shape": "(100,)",
            "backend": "TRITON",
            "source": "fallback",
            "calls": 1,
        },
    ]


def test_shape_capacity(monkeypatch, tmp_path):
    _set_env(monkeypatch, "PRIMUS_TURBO_DISPATCH_REPORT", str(tmp_path / "unused.json"))
    monkeypatch.setattr(backend_mod, "_DISPATCH_SHAPES_CAPACITY", 2)
    for x in range(5):
        _FakeDispatcher.dispatch(_DEFAULT, x=x)
    assert len(dispatch_report(by_shape=True)) == 2
    assert _totals() == {("_FakeDispatcher", "CK", "default"): 5}


def test_dump_and_exit_hook(monkeypatch, tmp_path):
    _FakeDispatcher.dispatch(_DEFAULT, x=1)
    path = tmp_path / "sub" / "report.json"
    dump_dispatch_report(str(path))
    report = json.loads(path.read_text())
    assert report["totals"] == [
        {"dispatcher": "_FakeDispatcher", "backend": "CK", "source": "default", "calls": 1}
    ]
    assert report["shapes"] == []

    _set_env(monkeypatch, "PRIMUS_TURBO_DISPATCH_REPORT", str(tmp_path / "rank{rank}_{pid}.json"))
    monkeypatch.setenv("RANK", "3")
    backend_mod._dump_dispatch_report_at_exit()
    written = tmp_path / f"rank3_{os.getpid()}.json"
    assert json.loads(written.read_text())["totals"][0]["calls"] == 1


def test_dispatch_source_values():
    assert [s.value for s in DispatchSource] == ["user", "autotune", "default", "fallback"]
