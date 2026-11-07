###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Autotune failures, scratch operands and CUDA-graph dispatch on a real GPU."""

import pytest
import torch

from primus_turbo.pytorch.core.backend import (
    AutoKernelDispatcher,
    BackendChoice,
    BackendEntry,
    BackendType,
    GlobalBackendManager,
    KernelBackend,
    TuneCache,
)
from tests.pytorch.ops.gradient_contract_helpers import caller_thread_backward  # noqa: F401

pytestmark = pytest.mark.usefixtures("caller_thread_backward")


@pytest.fixture
def dispatcher(monkeypatch):
    monkeypatch.setattr(AutoKernelDispatcher, "_subclasses", list(AutoKernelDispatcher._subclasses))

    class Add(KernelBackend):
        @staticmethod
        def can_handle(x, **kwargs):
            return x.numel() > 0

        @staticmethod
        def execute(x, out):
            return out.add_(x)

        @staticmethod
        def tuning_kwargs(x, out):
            return dict(x=x, out=torch.zeros_like(out))

    class Failed(Add):
        @staticmethod
        def execute(**kwargs):
            raise RuntimeError("candidate cannot launch")

    class Dispatch(AutoKernelDispatcher):
        _backends = {BackendType.TRITON: BackendEntry(Failed), BackendType.TURBO: BackendEntry(Add)}
        _cache = TuneCache()
        _warmup_iters = 1
        _profile_iters = 2

        @classmethod
        def make_key(cls, x, **kwargs):
            return tuple(x.shape)

    return Dispatch, Add, Failed


def operands():
    return torch.ones(32, device="cuda"), torch.full((32,), 7.0, device="cuda")


def test_tuning_uses_scratch_and_discards_failing_candidate(dispatcher):
    dispatch, add, _ = dispatcher
    x, out = operands()
    got = dispatch.dispatch(BackendChoice(auto_tune=True), x=x, out=out)
    assert got is out
    torch.testing.assert_close(out, torch.full_like(out, 8.0))
    assert dispatch._cache.get(tuple(x.shape)) is add
    # A cached selection still executes exactly once against the user's buffer.
    dispatch.dispatch(BackendChoice(auto_tune=True), x=x, out=out)
    torch.testing.assert_close(out, torch.full_like(out, 9.0))


def test_all_profile_failures_fall_back_to_nonprofiled_default(dispatcher):
    dispatch, add, _ = dispatcher
    dispatch._backends[BackendType.TURBO] = BackendEntry(add, autotune=False)
    x, out = operands()
    dispatch.dispatch(BackendChoice(backend=BackendType.TURBO, auto_tune=True), x=x, out=out)
    assert dispatch._cache.get(tuple(x.shape)) is None
    torch.testing.assert_close(out, torch.full_like(out, 8.0))


@pytest.mark.parametrize("per_op", [False, True])
def test_resolve_and_execute_share_tuned_backend(dispatcher, per_op):
    dispatch, _, _ = dispatcher
    GlobalBackendManager.set_auto_tune(not per_op)
    x, out = operands()
    selected = dispatch.resolve(
        BackendType.TRITON, BackendChoice(auto_tune=True) if per_op else None, x=x, out=out
    )
    assert selected is BackendType.TURBO
    torch.testing.assert_close(out, torch.full_like(out, 7.0))
    dispatch.dispatch(BackendChoice(), BackendChoice(backend=selected), x=x, out=out)
    torch.testing.assert_close(out, torch.full_like(out, 8.0))


@pytest.mark.parametrize("select", ["dispatch", "resolve"])
def test_unregistered_or_ineligible_backend_is_rejected(dispatcher, select):
    dispatch, _, _ = dispatcher
    x, out = operands()
    if select == "dispatch":
        call = lambda user, inp: dispatch.dispatch(
            BackendChoice(), BackendChoice(backend=user), x=inp, out=out
        )
    else:
        call = lambda user, inp: dispatch.resolve(BackendType.TURBO, user, x=inp, out=out)
    with pytest.raises(ValueError, match="not registered"):
        call(BackendType.AITER, x)
    with pytest.raises(ValueError, match="cannot handle"):
        call(BackendType.TURBO, x[:0])
    with pytest.raises(ValueError, match="No compatible backend"):
        call(None, x[:0])
    torch.testing.assert_close(out, torch.full_like(out, 7.0))


def test_capture_skips_profiling_and_replay_executes_once(dispatcher):
    dispatch, add, _ = dispatcher
    dispatch._backends = {BackendType.TURBO: BackendEntry(add)}
    GlobalBackendManager.set_auto_tune(True)
    x, out = operands()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        add.execute(x, out)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        dispatch.dispatch(BackendChoice(backend=BackendType.TURBO), x=x, out=out)
    assert dispatch._cache.get(tuple(x.shape)) is None
    out.fill_(7)
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(out, torch.full_like(out, 8.0))


def test_pinned_choice_wins_over_autotune(dispatcher):
    dispatch, _, _ = dispatcher
    x, out = operands()
    assert (
        dispatch.resolve(
            BackendType.TRITON, BackendChoice(backend=BackendType.TURBO, auto_tune=True), x=x, out=out
        )
        is BackendType.TURBO
    )
    assert dispatch._cache.get(tuple(x.shape)) is None
    torch.testing.assert_close(out, torch.full_like(out, 7.0))


@pytest.mark.parametrize("select", ["dispatch", "resolve"])
def test_ineligible_default_falls_back_without_tuning(dispatcher, select):
    dispatch, add, _ = dispatcher

    class Reject(add):
        @staticmethod
        def can_handle(**kwargs):
            return False

    dispatch._backends[BackendType.TRITON] = BackendEntry(Reject)
    x, out = operands()
    GlobalBackendManager.set_auto_tune(False)
    if select == "resolve":
        assert dispatch.resolve(BackendType.TRITON, None, x=x, out=out) is BackendType.TURBO
        torch.testing.assert_close(out, torch.full_like(out, 7.0))
    else:
        dispatch.dispatch(BackendChoice(BackendType.TRITON), x=x, out=out)
        torch.testing.assert_close(out, torch.full_like(out, 8.0))
    assert dispatch._cache.get(tuple(x.shape)) is None


def test_no_eligible_tuning_candidates(dispatcher):
    dispatch, _, _ = dispatcher
    x, out = operands()
    assert dispatch.tune(x=x[:0], out=out[:0]) is None
    assert dispatch._enum_for_impl(object) is None
