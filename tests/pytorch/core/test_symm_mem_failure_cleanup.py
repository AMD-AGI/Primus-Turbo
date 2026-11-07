###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Driver failures must free completed allocations without hiding the original error."""

import ctypes
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from primus_turbo.pytorch.core import symm_mem


@pytest.mark.parametrize("size,pad", [(0, 128), (1024, -1)])
def test_invalid_capacity_fails_before_allocating(size, pad, monkeypatch):
    runtime = Mock()
    monkeypatch.setattr(symm_mem, "get_hip_runtime_lib", runtime)
    with pytest.raises(ValueError, match="must be"):
        symm_mem.SymmetricMemory(object(), size, signal_pad_size=pad)
    runtime.assert_not_called()


@pytest.mark.parametrize(
    "failure_stage", ["buffer_zero", "buffer_exchange", "signal_zero", "signal_exchange"]
)
@pytest.mark.parametrize("free_fails", [False, True])
def test_partial_initialization_frees_owned_pointers(failure_stage, free_fails, monkeypatch):
    lib = Mock()
    lib.hipMalloc.side_effect = [ctypes.c_void_p(1000), ctypes.c_void_p(2000)]
    original = RuntimeError("initialization failed")
    if failure_stage == "buffer_zero":
        lib.hipMemset.side_effect = original
    elif failure_stage == "signal_zero":
        lib.hipMemset.side_effect = [None, original]
    if free_fails:
        lib.hipFree.side_effect = RuntimeError("driver cleanup failed")
    exchange = Mock(
        side_effect=original
        if failure_stage == "buffer_exchange"
        else ([[1000], original] if failure_stage == "signal_exchange" else [[1000], [2000]])
    )
    monkeypatch.setattr(symm_mem, "get_hip_runtime_lib", lambda: lib)
    monkeypatch.setattr(symm_mem.SymmetricMemory, "_rendezvous", exchange)
    group = SimpleNamespace(rank=lambda: 0, size=lambda: 1)
    memory = object.__new__(symm_mem.SymmetricMemory)
    if free_fails:
        with pytest.warns(UserWarning, match="cleanup|hipFree"):
            with pytest.raises(RuntimeError, match="initialization failed") as caught:
                memory.__init__(group, 1024, 128)
    else:
        with pytest.raises(RuntimeError, match="initialization failed") as caught:
            memory.__init__(group, 1024, 128)
    assert caught.value is original
    assert memory.is_destroyed
    assert lib.hipFree.call_count == (2 if failure_stage.startswith("signal") else 1)
    before = lib.hipFree.call_count
    memory.destroy()
    assert lib.hipFree.call_count == before


def test_cleanup_ignores_unopened_peers():
    memory = object.__new__(symm_mem.SymmetricMemory)
    memory.lib = Mock()
    memory.rank = 0
    memory.world_size = 3
    memory.buffer_ptrs = [0, 0, 3000]
    memory.signal_pad_ptrs = []
    memory.buffer_ptrs_dev = memory.signal_pad_ptrs_dev = None
    memory.is_destroyed = False
    memory.destroy()
    memory.lib.hipFree.assert_not_called()
    memory.lib.hipIpcCloseMemHandle.assert_called_once_with(3000)


def test_ipc_close_error_does_not_replace_open_error(monkeypatch):
    memory = object.__new__(symm_mem.SymmetricMemory)
    memory.lib = Mock()
    memory.rank = 0
    memory.world_size = 3
    memory.group = object()
    memory.buffer_ptrs = []
    memory.signal_pad_ptrs = []
    memory.is_destroyed = False
    memory.buffer_ptrs_dev = memory.signal_pad_ptrs_dev = None
    original = RuntimeError("peer open failed")
    memory.lib.hipIpcOpenMemHandle.side_effect = [ctypes.c_void_p(2000), original]
    memory.lib.hipIpcCloseMemHandle.side_effect = RuntimeError("peer close failed")

    def gather(out, value, group):
        out[:] = [b"own", b"one", b"two"]

    monkeypatch.setattr(symm_mem.dist, "all_gather_object", gather)
    with pytest.raises(RuntimeError, match="peer open failed") as caught:
        memory._rendezvous(ctypes.c_void_p(1000))
    assert caught.value is original
    memory.lib.hipIpcCloseMemHandle.assert_called_once_with(2000)
    memory.destroy()
