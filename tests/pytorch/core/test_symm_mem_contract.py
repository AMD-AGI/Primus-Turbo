###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# See LICENSE for license information.
###############################################################################

import ctypes
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from primus_turbo.pytorch.core import symm_mem as impl


@pytest.fixture
def memory():
    # A host-side fixture owns no real device pointers. Tests verify bounds,
    # IPC cleanup and lifetime independently of multi-rank synchronization.
    obj = object.__new__(impl.SymmetricMemory)
    obj.lib = Mock()
    obj.rank, obj.world_size, obj.device = 0, 3, 0
    obj.group = object()
    obj.buffer_size, obj.signal_pad_size = 1024, 128
    obj.buffer_ptrs, obj.signal_pad_ptrs = [1000, 2000, 3000], [4000, 5000, 6000]
    obj.buffer_ptrs_dev = obj.signal_pad_ptrs_dev = None
    obj.is_destroyed = False
    yield obj
    obj.is_destroyed = True


@pytest.mark.parametrize("rank", [-1, 3])
@pytest.mark.parametrize("kind", ["buffer", "signal_pad"])
def test_peer_bounds(memory, rank, kind):
    with pytest.raises(ValueError, match="Invalid peer rank"):
        getattr(memory, "get_" + kind)(rank, (1,), torch.float32)


@pytest.mark.parametrize("kind, shape", [("buffer", (257,)), ("signal_pad", (33,))])
def test_buffer_capacity_includes_storage_offset(memory, kind, shape):
    with pytest.raises(ValueError, match="exceeds the allocated size"):
        getattr(memory, "get_" + kind)(0, shape, torch.float32)
    with pytest.raises(ValueError, match="exceeds the allocated size"):
        getattr(memory, "get_" + kind)(0, (1,), torch.float32, storage_offset=shape[0] - 1)


def test_views_apply_element_offsets_and_default_signal_dtype(memory, monkeypatch):
    view = Mock(return_value=object())
    monkeypatch.setattr(impl, "_tensor_from_device_ptr", view)
    assert memory.get_buffer(1, (2, 3), torch.float32, storage_offset=5) is view.return_value
    view.assert_called_with(2020, (2, 3), torch.float32, 0)
    memory.get_signal_pad(2)
    view.assert_called_with(6000, (32,), torch.uint32, 0)
    memory.get_signal_pad(1, (2,), torch.int64, storage_offset=3)
    view.assert_called_with(5024, (2,), torch.int64, 0)


def test_context_exit_propagates_error_and_frees_each_pointer_once(memory):
    with pytest.raises(RuntimeError, match="body failed"):
        with memory as entered:
            assert entered is memory
            raise RuntimeError("body failed")
    assert memory.is_destroyed
    assert memory.lib.hipFree.call_args_list == [((1000,),), ((4000,),)]
    assert memory.lib.hipIpcCloseMemHandle.call_count == 4
    memory.destroy()
    assert memory.lib.hipFree.call_count == 2


def test_cleanup_continues_after_one_driver_error(memory):
    memory.lib.hipFree.side_effect = [RuntimeError("free failed"), None]
    with pytest.warns(UserWarning, match="cleanup error for rank 0"):
        memory.destroy()
    assert memory.lib.hipIpcCloseMemHandle.call_count == 4
    assert memory.buffer_ptrs == memory.signal_pad_ptrs == []


@pytest.mark.parametrize("pointer", [None, 0])
def test_null_pointer_is_rejected(pointer):
    with pytest.raises(ValueError, match="NULL device pointer"):
        impl.SymmetricMemory._c_void_p_to_int(ctypes.c_void_p(pointer))


def test_failed_ipc_open_closes_already_opened_peers(memory, monkeypatch):
    def gather(outputs, value, group):
        outputs[:] = [b"self", b"peer1", b"peer2"]

    monkeypatch.setattr(impl.dist, "all_gather_object", gather)
    memory.lib.hipIpcOpenMemHandle.side_effect = [ctypes.c_void_p(7000), RuntimeError("open failed")]
    with pytest.raises(RuntimeError, match="open failed"):
        memory._rendezvous(ctypes.c_void_p(1000))
    memory.lib.hipIpcCloseMemHandle.assert_called_once_with(7000)


def test_zero_signal_pad_cannot_barrier(memory):
    memory.signal_pad_size = 0
    with pytest.raises(RuntimeError, match="requires signal_pad_size"):
        memory.barrier()


def test_graph_capture_cannot_grow_workspace(monkeypatch):
    group = SimpleNamespace(group_name="coverage-test")
    monkeypatch.setattr(impl, "_group_name_to_workspace_tensor", {})
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    with pytest.raises(RuntimeError, match="cannot resize.*workspace during CUDA graph capture"):
        impl.get_symm_mem_workspace(group, 1024)


def test_workspace_reuse_and_growth(monkeypatch):
    group = SimpleNamespace(group_name="coverage-test")
    old = Mock(buffer_size=1024, signal_pad_size=128)
    table = {group.group_name: old}
    monkeypatch.setattr(impl, "_group_name_to_workspace_tensor", table)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    construct = Mock(return_value=object())
    monkeypatch.setattr(impl, "SymmetricMemory", construct)
    assert impl.get_symm_mem_workspace(group, 512, 64) is old
    construct.assert_not_called()
    new = impl.get_symm_mem_workspace(group, 2048, 4096)
    construct.assert_called_once_with(group, 2048, signal_pad_size=4096)
    old.destroy.assert_called_once()
    assert table[group.group_name] is new
