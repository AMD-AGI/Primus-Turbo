###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""``primus_turbo.flydsl.utils.launch_cache``: one ``flyc.compile`` per key, a key that covers what
the trace bakes in, and no direct ``@flyc.jit`` call on a cache hit."""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
import pytest
import torch
from flydsl.compiler.jit_function import JitFunction
from flydsl.compiler.kernel_function import CompilationContext

from primus_turbo.flydsl.utils import launch_cache
from primus_turbo.flydsl.utils.buffer_ops import (
    buffer_load,
    buffer_store,
    create_buffer_resource,
)

_THREADS = 64


@functools.lru_cache(maxsize=None)
def _axpb_kernel(scale: int):
    @flyc.kernel(known_block_size=[_THREADS, 1, 1])
    def axpb(X: fx.Tensor, Y: fx.Tensor, b: fx.Int32):
        i = fx.block_idx.x * fx.Int32(_THREADS) + fx.thread_idx.x
        v = buffer_load(create_buffer_resource(X, max_size=True), i, vec_width=1, dtype=fx.T.i32)
        buffer_store(v * fx.Int32(scale) + b, create_buffer_resource(Y, max_size=True), i)

    return axpb


@flyc.jit
def _launch_axpb(X, Y, b, grid, scale: fx.Constexpr[int], stream: fx.Stream, extra: fx.Constexpr[int] = 0):
    _axpb_kernel(scale + extra)(X, Y, b).launch(grid=(grid, 1, 1), block=(_THREADS, 1, 1), stream=stream)


@pytest.fixture
def compiles(monkeypatch):
    """Every flyc.compile the launch cache makes, and a fresh cache."""
    made = []
    compile = flyc.compile

    def counted(launch, *args):
        made.append(args)
        return compile(launch, *args)

    monkeypatch.setattr(launch_cache.flyc, "compile", counted)
    monkeypatch.setattr(launch_cache, "_COMPILED", {})
    return made


@pytest.fixture
def direct_calls(monkeypatch):
    """Top-level calls of a @flyc.jit function. flydsl 0.3.4 compiles through one, 0.5 does not."""
    from flydsl._mlir import ir

    calls = []
    call = JitFunction.__call__

    def counted(self, *args, **kwargs):
        if ir.Context.current is None:
            calls.append(self)
        return call(self, *args, **kwargs)

    monkeypatch.setattr(JitFunction, "__call__", counted)
    return calls


def _run(n, scale, b=7, **kwargs):
    x = torch.randint(-1000, 1000, (n,), dtype=torch.int32, device="cuda")
    y = torch.empty_like(x)
    grid = n // _THREADS
    stream = torch.cuda.current_stream()
    if kwargs:
        launch_cache.run_launcher(_launch_axpb, x, y, b, grid, scale, stream=stream, **kwargs)
    else:
        launch_cache.run_launcher(_launch_axpb, x, y, b, grid, scale, stream)
    torch.cuda.synchronize()
    return x, y


def test_one_compile_per_key(compiles, direct_calls):
    for n in (64, 256, 64 * 1024):  # a bare tensor compiles layout-dynamic, b and grid are runtime
        x, y = _run(n, 3)
        torch.testing.assert_close(y, x * 3 + 7)
    assert len(compiles) == 1
    x, y = _run(128, 5)  # a Constexpr value is baked into the trace
    torch.testing.assert_close(y, x * 5 + 7)
    assert len(compiles) == 2
    for _ in range(10):
        x, y = _run(128, 5, b=-2)
        torch.testing.assert_close(y, x * 5 - 2)
    assert len(compiles) == 2
    # only the compiles themselves may go through JitFunction.__call__
    assert len(direct_calls) <= len(compiles)


def test_kwargs_and_defaults(compiles):
    x, y = _run(64, 3, extra=2)
    torch.testing.assert_close(y, x * 5 + 7)
    x, y = _run(64, 3, extra=0)  # the default, spelled out: same key as leaving it out
    torch.testing.assert_close(y, x * 3 + 7)
    x, y = _run(64, 3)
    torch.testing.assert_close(y, x * 3 + 7)
    assert len(compiles) == 2


def test_compile_runs_the_launch_once(compiles):
    """The miss is the launch: an accumulating kernel would fold a second run into its output."""
    x = torch.zeros(64, dtype=torch.int32, device="cuda")
    y = torch.zeros_like(x)
    stream = torch.cuda.current_stream()
    launch_cache.run_launcher(_launch_axpb, y, y, 1, 1, 1, stream)  # y = y * 1 + 1
    torch.cuda.synchronize()
    torch.testing.assert_close(y, x + 1)
    launch_cache.run_launcher(_launch_axpb, y, y, 1, 1, 1, stream)
    torch.cuda.synchronize()
    torch.testing.assert_close(y, x + 2)

    cache = {}
    launch_cache.run_compiled(cache, "k", _launch_axpb, y, y, 1, 1, 1, stream, 0)
    torch.cuda.synchronize()
    torch.testing.assert_close(y, x + 3)
    assert len(cache) == 1


def test_key():
    x = torch.zeros(4, 8, dtype=torch.bfloat16)
    stream = None
    base = launch_cache.launch_key(_launch_axpb, (x, x, 1, 1, 3, stream, 0))
    assert (
        launch_cache.launch_key(
            _launch_axpb, (torch.zeros(16, 2, dtype=torch.bfloat16),) * 2 + (2, 9, 3, stream, 0)
        )
        == base
    )
    assert launch_cache.launch_key(_launch_axpb, (x, x, 1, 1, 4, stream, 0)) != base  # Constexpr
    assert launch_cache.launch_key(_launch_axpb, (x.float(), x, 1, 1, 3, stream, 0)) != base  # dtype
    assert launch_cache.launch_key(_launch_axpb, (x.flatten(), x, 1, 1, 3, stream, 0)) != base  # rank
    assert launch_cache.launch_key(_launch_axpb, (x.t(), x, 1, 1, 3, stream, 0)) != base  # unit-stride axis
    assert launch_cache.launch_key(_launch_axpb, (x, x, True, 1, 3, stream, 0)) != base  # i1 vs i32
    with CompilationContext.compile_hints({"waves_per_eu": 2}):
        assert launch_cache.launch_key(_launch_axpb, (x, x, 1, 1, 3, stream, 0)) != base
    static = launch_cache.launch_key(_launch_axpb, (flyc.from_torch_tensor(x), x, 1, 1, 3, stream, 0))
    assert static != base
    assert static != launch_cache.launch_key(
        _launch_axpb,
        (flyc.from_torch_tensor(torch.zeros(4, 16, dtype=torch.bfloat16)), x, 1, 1, 3, stream, 0),
    )


def test_graph_capture(compiles):
    x = torch.randint(-1000, 1000, (256,), dtype=torch.int32, device="cuda")
    y = torch.empty_like(x)
    stream = torch.cuda.Stream()
    _run(256, 3)  # compiled before the capture
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=stream):
        launch_cache.run_launcher(_launch_axpb, x, y, 7, 4, 3, torch.cuda.current_stream())
    for _ in range(3):
        x.copy_(torch.randint_like(x, -1000, 1000))
        g.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(y, x * 3 + 7)
    assert len(compiles) == 1


def test_plain_callable_passes_through():
    seen = []
    launch_cache.run_launcher(lambda *a, **k: seen.append((a, k)), 1, 2, k=3)
    assert seen == [((1, 2), {"k": 3})]
