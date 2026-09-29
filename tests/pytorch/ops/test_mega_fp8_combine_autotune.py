###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""The fp8 combine's online CU-split tuner: its state machine, and CUDA graph capture.

The EP8 op suites never lock a key -- each shape there sees a handful of calls against a schedule
of 21 -- so the tuner is exercised here directly, on a single GPU:

* State machine, with fake event pairs whose timings and completion are set by the test: the
  round-robin schedule, min-per-candidate (which discards the first-use compile outlier), the
  non-blocking drain leaving unfinished pairs for later, lock-in and winner reuse, the cross-rank
  SUM that makes the winner unanimous, and the disabled / pinned paths that must not measure.
* Capture: the tuner brackets each call with timing events and reads them on a later call. An
  event pair recorded during capture fails that read with ``invalid resource handle``, so before
  ``locked_combine_cu`` any capture that happened before tuning locked in broke. ``launch`` stands
  in for the combine there (one tiny kernel), so only the tuner's host logic is under test.
"""

import warnings

import pytest
import torch

from primus_turbo.pytorch.core.utils import is_gfx1250

if is_gfx1250():
    pytest.skip("mega_moe_fused is not supported on gfx1250", allow_module_level=True)

import primus_turbo.flydsl.mega.fp8.combine_autotune as combine_autotune  # noqa: E402
from primus_turbo.flydsl.mega.fp8.grouped_gemm_combine_fp8_kernel import (  # noqa: E402
    _compile,
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


class _FakeEnd:
    """End event with a preset elapsed time; ``done=False`` models a launch still in flight."""

    def __init__(self, ms, done=True):
        self.ms, self.done, self.synced = ms, done, False

    def query(self):
        return self.done

    def synchronize(self):
        self.synced = self.done = True


class _FakeStart:
    def elapsed_time(self, end):
        assert end.done, "elapsed time read before the end event completed"
        return end.ms


@pytest.fixture
def small_schedule(monkeypatch):
    monkeypatch.setattr(combine_autotune, "_CANDIDATES", (16, 32, 64))
    monkeypatch.setattr(combine_autotune, "_REPS", 2)


def _drive(key, timings, *, group=None, done=True):
    """Choose/observe until the key locks; ``timings[cu]`` is consumed one sample per call.

    Returns ``(winner, candidates in the order they were tried, end events)``."""
    tried, ends = [], []
    while True:
        cu = combine_autotune.choose_combine_cu(key, group=group, default=_FALLBACK)
        if combine_autotune._STATE[key].winner is not None:
            return cu, tried, ends
        tried.append(cu)
        ends.append(_FakeEnd(timings[cu].pop(0), done=done))
        combine_autotune.observe_combine_cu(key, cu, _FakeStart(), ends[-1])


def test_lock_in_takes_the_lowest_min_and_reuses_it(small_schedule):
    key = ("lock-in",)
    # 16's first sample is a compile outlier; its min still makes it the fastest candidate.
    timings = {16: [50.0, 1.0], 32: [2.0, 2.1], 64: [3.0, 1.5]}
    winner, tried, _ = _drive(key, timings)

    assert tried == [16, 32, 64, 16, 32, 64]  # round-robin, every candidate REPS times
    assert winner == 16
    st = combine_autotune._STATE[key]
    assert {cu: min(v) for cu, v in st.samples.items()} == {16: 1.0, 32: 2.0, 64: 1.5}

    # Locked: the winner is reused and further observations are ignored.
    assert combine_autotune.choose_combine_cu(key, default=_FALLBACK) == 16
    combine_autotune.observe_combine_cu(key, 64, _FakeStart(), _FakeEnd(0.1))
    assert st.pending == [] and st.next_idx == len(tried)
    assert combine_autotune.choose_combine_cu(key, default=_FALLBACK) == 16
    assert combine_autotune.locked_combine_cu(key, default=_FALLBACK) == 16


def test_unfinished_launches_are_read_later_not_waited_on(small_schedule):
    key = ("in-flight",)
    timings = {16: [3.0, 3.0], 32: [1.0, 1.0], 64: [2.0, 2.0]}
    winner, tried, ends = _drive(key, timings, done=False)

    # Nothing completed during the schedule, so nothing was folded and nothing was waited on until
    # the deciding call, which waits out every pending pair exactly then.
    assert all(e.synced for e in ends)
    assert winner == 32
    assert combine_autotune._STATE[key].pending == []


@pytest.mark.parametrize(
    "other_rank, want",
    [
        # The locally best split (16) is not the best in sum across ranks (32): SUM, not local min.
        ({16: 5.0, 32: 1.1, 64: 1.3}, 32),
        # A candidate no rank measured stays inf and is never chosen.
        ({16: 0.5, 32: 0.5, 64: float("inf")}, 16),
    ],
)
def test_winner_is_agreed_across_ranks_by_sum(small_schedule, monkeypatch, other_rank, want):
    calls = []

    def fake_all_reduce(t, op=None, group=None):
        calls.append((op, group))
        t += torch.tensor([other_rank[cu] for cu in (16, 32, 64)], dtype=t.dtype, device=t.device)

    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "all_reduce", fake_all_reduce)
    group = object()
    local = {16: [1.0, 1.0], 32: [1.2, 1.2], 64: [1.3, 1.3]}
    winner, _, _ = _drive(("cross-rank", want), local, group=group)

    assert calls == [(torch.distributed.ReduceOp.SUM, group)]  # once per key, at lock-in
    assert winner == want


@pytest.fixture
def no_events(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("a path that must not measure constructed a timing event")

    monkeypatch.setattr(torch.cuda, "Event", refuse)


def test_disabled_tuner_runs_the_fallback_and_measures_nothing(monkeypatch, no_events):
    monkeypatch.setattr(combine_autotune, "_ENABLED", False)
    key = ("disabled",)
    launch = _Launch()

    _call(key, launch)
    assert launch.chosen == [_FALLBACK]
    assert combine_autotune.choose_combine_cu(key, default=_FALLBACK) == _FALLBACK
    assert combine_autotune.locked_combine_cu(key, default=_FALLBACK) == _FALLBACK
    assert key not in combine_autotune._STATE


def test_pinned_split_bypasses_the_tuner(no_events):
    key = ("pinned",)
    launch = _Launch()

    _launch_maybe_tuned(key, 64, _FALLBACK, None, launch)
    assert launch.chosen == [64]
    assert key not in combine_autotune._STATE


@pytest.mark.parametrize("raw", ["0", "-8", "abc"])
def test_env_pin_must_be_a_positive_count(monkeypatch, raw):
    # 0 would mean "no PUSH role" to the combine, with the reduce still waiting for its payload.
    monkeypatch.setenv("PT_MEGA_FP8_L1_COMBINE_CU", raw)
    with pytest.raises(ValueError, match="PT_MEGA_FP8_L1_COMBINE_CU"):
        combine_autotune.env_combine_cu("PT_MEGA_FP8_L1_COMBINE_CU")


def test_env_pin_parses_or_stays_unset(monkeypatch):
    monkeypatch.setenv("PT_MEGA_FP8_L2_COMBINE_CU", "64")
    assert combine_autotune.env_combine_cu("PT_MEGA_FP8_L2_COMBINE_CU") == 64
    monkeypatch.delenv("PT_MEGA_FP8_L2_COMBINE_CU")
    assert combine_autotune.env_combine_cu("PT_MEGA_FP8_L2_COMBINE_CU") is None


def test_candidate_list_must_be_positive(monkeypatch):
    # The tuner launches every candidate for real, so a 0 in the list would hang the tuning call.
    monkeypatch.setenv("PT_MEGA_FP8_COMBINE_CU_CANDIDATES", "16 0 32")
    with pytest.raises(ValueError, match="PT_MEGA_FP8_COMBINE_CU_CANDIDATES"):
        combine_autotune._env_int_list("PT_MEGA_FP8_COMBINE_CU_CANDIDATES", (16,))


_COMPILE_SHAPE = dict(
    out_features=1024,  # the fp8 reduce needs hidden % 1024 == 0
    hidden_size=256,
    num_max_pool_tokens=256,
    BLOCK_M=256,
    BLOCK_N=256,
    combine_slots=256,
    topk=4,
    num_experts=16,
    rank=0,
    num_ranks=8,
    apply_weights=True,
    with_gate=False,
)


@pytest.mark.parametrize("num_combine_cu, num_reduce_cu", [(0, 256), (-1, 0)])
def test_combine_rejects_a_reduce_with_no_push(num_combine_cu, num_reduce_cu):
    """Whatever the source -- env pin, candidate list, or an explicit argument -- the kernel builder
    refuses a split that would leave the reduce spinning, before anything is launched."""
    with pytest.raises(AssertionError, match="num_combine_cu"):
        _compile(num_combine_cu=num_combine_cu, num_reduce_cu=num_reduce_cu, **_COMPILE_SHAPE)


def test_combine_still_builds_the_push_free_isolation_variant():
    # (0 PUSH, 0 reduce) is the GEMM-alone build the benches time; the guard must not refuse it.
    assert _compile(num_combine_cu=0, num_reduce_cu=0, **_COMPILE_SHAPE) is not None


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
