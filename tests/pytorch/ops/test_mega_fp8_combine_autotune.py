###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""CUDA graph capture of the fp8 combine while its CU split is being tuned online.

The tuner brackets each call with timing events and reads them on a later call. An event pair
recorded during capture fails that read with ``invalid resource handle``, so before
``locked_combine_cu`` any capture that happened before tuning locked in broke -- the first call of
a fresh shape, or any call mid-schedule. ``launch`` stands in for the combine here (one tiny kernel),
so this exercises the tuner's host logic alone and needs a single GPU.
"""

import warnings

import pytest
import torch

from primus_turbo.pytorch.core.utils import is_gfx1250

if is_gfx1250():
    pytest.skip("mega_moe_fused is not supported on gfx1250", allow_module_level=True)

import primus_turbo.flydsl.mega.fp8.combine_autotune as combine_autotune  # noqa: E402
from primus_turbo.flydsl.mega.fp8.grouped_gemm_combine_fp8_kernel import (  # noqa: E402
    _launch_maybe_tuned,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

_FALLBACK = 32


@pytest.fixture(autouse=True)
def _tuning_on(monkeypatch):
    monkeypatch.setattr(combine_autotune, "_ENABLED", True)
    monkeypatch.setattr(combine_autotune, "_VERBOSE", False)
    monkeypatch.setattr(combine_autotune, "_STATE", {})
    monkeypatch.setattr(combine_autotune, "_WARNED_CAPTURE", set())


class _Launch:
    """Records the CU split each call was given and does one real kernel of work."""

    def __init__(self):
        self.buf = torch.zeros(1, device="cuda")
        self.chosen = []

    def __call__(self, cu):
        self.chosen.append(cu)
        self.buf.add_(cu)
        return self.buf


def _call(key, launch):
    return _launch_maybe_tuned(key, None, _FALLBACK, None, launch)


def _capture(key, launch):
    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream), torch.cuda.graph(graph, stream=stream):
        _call(key, launch)
    graph.replay()
    torch.cuda.synchronize()
    return launch.chosen[-1]


def _capture_without_fallback_warning(key, launch):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cu = _capture(key, launch)
    assert not [w for w in caught if "captures the fallback" in str(w.message)]
    return cu


def _schedule_len():
    return len(combine_autotune._CANDIDATES) * combine_autotune._REPS


@pytest.mark.parametrize("eager_calls_before", [0, 3])
def test_capture_before_lock_takes_fallback_and_leaves_tuning_alone(eager_calls_before):
    key = ("capture-before-lock", eager_calls_before)
    launch = _Launch()
    for _ in range(eager_calls_before):
        _call(key, launch)
    before = combine_autotune._STATE.get(key)
    next_idx = before.next_idx if before else 0

    # The graph keeps the fallback for good, so it must say so -- once per key, not per capture.
    with pytest.warns(UserWarning, match=f"captures the fallback {_FALLBACK}"):
        assert _capture(key, launch) == _FALLBACK
    assert _capture_without_fallback_warning(key, launch) == _FALLBACK

    # Tuning is neither advanced nor poisoned by the captured calls: eager calls carry on and lock.
    st = combine_autotune._STATE.get(key)
    assert (st.next_idx if st else 0) == next_idx
    for _ in range(_schedule_len() - next_idx + 1):
        _call(key, launch)
    torch.cuda.synchronize()
    assert combine_autotune._STATE[key].winner in combine_autotune._CANDIDATES


def test_capture_after_lock_takes_the_winner():
    key = ("capture-after-lock",)
    launch = _Launch()
    for _ in range(_schedule_len() + 1):
        _call(key, launch)
    torch.cuda.synchronize()
    winner = combine_autotune._STATE[key].winner
    assert winner is not None

    assert _capture_without_fallback_warning(key, launch) == winner
