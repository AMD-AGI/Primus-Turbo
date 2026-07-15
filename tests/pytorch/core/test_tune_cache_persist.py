###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""CPU-only tests for TuneCache offline persistence (dump_cache / load_cache)."""

import torch

from primus_turbo.pytorch.core.backend import (
    AutoKernelDispatcher,
    BackendEntry,
    BackendType,
    GlobalBackendManager,
    KernelBackend,
)
from primus_turbo.pytorch.core.low_precision import ScalingGranularity


class _DummyTriton(KernelBackend):
    @staticmethod
    def can_handle(**kwargs):
        return True

    @staticmethod
    def execute(**kwargs):
        return None


class _DummyCK(KernelBackend):
    @staticmethod
    def can_handle(**kwargs):
        return True

    @staticmethod
    def execute(**kwargs):
        return None


class _DummyDispatcher(AutoKernelDispatcher):
    # Defined in the class body so __init_subclass__ keeps it (does not reset to {}).
    _backends = {
        BackendType.TRITON: BackendEntry(_DummyTriton),
        BackendType.CK: BackendEntry(_DummyCK),
    }

    @classmethod
    def make_key(cls, m, n, k, dtype, trans, gran):
        # Mix of int / torch.dtype / bool / Enum — exercises the key codec.
        return (m, n, k, dtype, trans, gran)


def test_tune_cache_roundtrip(tmp_path):
    d = _DummyDispatcher
    d._cache.clear()

    k1 = d.make_key(16, 2048, 4096, torch.float8_e4m3fn, True, ScalingGranularity.ROWWISE)
    k2 = d.make_key(32, 2048, 4096, torch.bfloat16, False, ScalingGranularity.BLOCKWISE)
    d._cache.put(k1, _DummyTriton)
    d._cache.put(k2, _DummyCK)

    path = tmp_path / "dummy.json"
    assert d.dump_cache(str(path)) == 2

    # Each entry is a dict with key / backend / perf.
    import json

    entries = json.loads(path.read_text())["entries"]
    assert set(entries[0]) == {"key", "backend", "perf"}

    # Fresh cache, then reload from disk.
    d._cache.clear()
    assert len(d._cache) == 0
    assert d.load_cache(str(path)) == 2

    # Keys must round-trip to the identical hashable -> lookup hits, maps to same impl.
    assert d._cache.get(k1) is _DummyTriton
    assert d._cache.get(k2) is _DummyCK


class _RetTriton(KernelBackend):
    @staticmethod
    def can_handle(**kwargs):
        return True

    @staticmethod
    def execute(**kwargs):
        return "triton"


class _RetCK(KernelBackend):
    @staticmethod
    def can_handle(**kwargs):
        return True

    @staticmethod
    def execute(**kwargs):
        return "ck"


class _DispatchDispatcher(AutoKernelDispatcher):
    _backends = {
        BackendType.TRITON: BackendEntry(_RetTriton),
        BackendType.CK: BackendEntry(_RetCK),
    }

    @classmethod
    def make_key(cls, m):
        return (m,)


def test_dispatch_uses_loaded_cache():
    d = _DispatchDispatcher
    d._cache.clear()
    GlobalBackendManager.set_auto_tune(False)  # deterministic: never profile (CPU-only path)
    try:
        # Empty cache -> falls through to the default backend (behavior unchanged).
        assert d.dispatch(BackendType.CK, m=16) == "ck"

        # Cache says TRITON for m=16 -> lookup wins over the CK default.
        d._cache.put(d.make_key(m=16), _RetTriton)
        assert d.dispatch(BackendType.CK, m=16) == "triton"

        # Un-cached key -> default backend.
        assert d.dispatch(BackendType.CK, m=32) == "ck"
    finally:
        GlobalBackendManager.set_auto_tune(None)


def test_load_unknown_backend_skipped(tmp_path):
    import json

    d = _DummyDispatcher
    d._cache.clear()

    path = tmp_path / "with_unknown.json"
    payload = {
        "dispatcher": "_DummyDispatcher",
        "entries": [
            # AITER not registered here
            {"key": [8, 512, 512, {"__dtype__": "bfloat16"}, False, None], "backend": "AITER", "perf": None},
        ],
    }
    path.write_text(json.dumps(payload))

    # AITER is a valid BackendType but not registered in _DummyDispatcher -> skipped.
    assert d.load_cache(str(path)) == 0
    assert len(d._cache) == 0
