###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""The dispatcher's offline tune-cache contract, on a dummy dispatcher (CPU only).

What the framework promises regardless of op: a dumped cache reloads into the same
lookups, an asset from a foreign build is ignored rather than trusted, and a pinned
config only reaches the backend it was tuned for. Per-op behaviour lives with that op.
"""

import json

import pytest
import torch

from primus_turbo.pytorch.core.backend import (
    AutoKernelDispatcher,
    BackendEntry,
    BackendType,
    GlobalBackendManager,
    KernelBackend,
    TuneEntry,
)
from primus_turbo.pytorch.core.low_precision import ScalingGranularity


class _TritonBackend(KernelBackend):
    @staticmethod
    def can_handle(**kwargs):
        return True

    @staticmethod
    def execute(backend_config=None, **kwargs):
        return ("triton", backend_config)


class _CKBackend(KernelBackend):
    @staticmethod
    def can_handle(**kwargs):
        return True

    @staticmethod
    def execute(backend_config=None, **kwargs):
        return ("ck", backend_config)


class _Dispatcher(AutoKernelDispatcher):
    # Assigned in the class body so __init_subclass__ keeps it instead of resetting to {}.
    _backends = {
        BackendType.TRITON: BackendEntry(_TritonBackend),
        BackendType.CK: BackendEntry(_CKBackend),
    }

    @classmethod
    def make_key(cls, m, dtype=torch.bfloat16, trans=False, gran=None):
        # int / torch.dtype / bool / Enum / None: one of each type the codec must handle.
        return (m, dtype, trans, gran)


@pytest.fixture(autouse=True)
def _isolated_cache():
    """No profiling and no cache carried between tests (the cache is class state)."""
    _Dispatcher._cache.clear()
    GlobalBackendManager.set_auto_tune(False)
    yield
    GlobalBackendManager.set_auto_tune(None)
    _Dispatcher._cache.clear()


def test_dumped_cache_reloads_into_the_same_lookups(tmp_path):
    tuned = _Dispatcher.make_key(16, torch.float8_e4m3fn, True, ScalingGranularity.ROWWISE)
    plain = _Dispatcher.make_key(32, torch.bfloat16, False, None)
    _Dispatcher._cache.put(tuned, TuneEntry(_TritonBackend, {"BLOCK_M": 64, "num_warps": 4}))
    _Dispatcher._cache.put(plain, TuneEntry(_CKBackend))

    path = tmp_path / "cache.json"
    assert _Dispatcher.dump_cache(str(path)) == 2
    entries = json.loads(path.read_text())["entries"]
    assert set(entries[0]) == {"key", "backend", "backend_config", "perf"}

    _Dispatcher._cache.clear()
    assert _Dispatcher.load_cache(str(path)) == 2

    # Rehydrated keys must hash equal to the originals, or every lookup silently misses.
    assert _Dispatcher._cache.get(tuned).backend is _TritonBackend
    assert _Dispatcher._cache.get(tuned).backend_config == {"BLOCK_M": 64, "num_warps": 4}
    assert _Dispatcher._cache.get(plain).backend is _CKBackend
    assert _Dispatcher._cache.get(plain).backend_config is None


def test_entry_naming_an_unregistered_backend_is_skipped(tmp_path):
    """An asset built against a different backend set must not resurrect that backend."""
    path = tmp_path / "alien.json"
    path.write_text(
        json.dumps(
            {
                "dispatcher": _Dispatcher.__name__,
                "entries": [
                    {
                        "key": [8, {"__dtype__": "bfloat16"}, False, None],
                        "backend": BackendType.AITER.name,  # valid enum, not registered here
                        "backend_config": None,
                        "perf": None,
                    }
                ],
            }
        )
    )

    assert _Dispatcher.load_cache(str(path)) == 0
    assert len(_Dispatcher._cache) == 0


def test_cache_outranks_the_default_backend():
    assert _Dispatcher.dispatch(BackendType.CK, m=16)[0] == "ck"

    _Dispatcher._cache.put(_Dispatcher.make_key(m=16), TuneEntry(_TritonBackend))
    assert _Dispatcher.dispatch(BackendType.CK, m=16)[0] == "triton"
    assert _Dispatcher.dispatch(BackendType.CK, m=32)[0] == "ck"  # untuned key keeps the default


def test_pinned_config_only_reaches_the_backend_it_was_tuned_for():
    """backend_config is opaque, so handing one backend another's config would be garbage in."""
    _Dispatcher._cache.put(_Dispatcher.make_key(m=16), TuneEntry(_TritonBackend, {"BLOCK_M": 64}))

    assert _Dispatcher.dispatch(BackendType.CK, BackendType.CK, m=16) == ("ck", None)
    assert _Dispatcher.dispatch(BackendType.CK, BackendType.TRITON, m=16) == ("triton", {"BLOCK_M": 64})
