###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
# See LICENSE for license information.
###############################################################################

"""Python runtime contracts; driver and coordinator boundaries are mocked.

Real IPC numerical round trips remain in lax/test_*dispatch_combine.py.
"""

import os
from types import SimpleNamespace
from unittest.mock import Mock

import jax
import numpy as np
import pytest

from primus_turbo.common import envs
from primus_turbo.jax.deep_ep import runtime as rt


@pytest.fixture(autouse=True)
def clean_runtime(monkeypatch):
    dep = Mock()
    dep.is_per_process_buffer_ready.return_value = False
    monkeypatch.setattr(rt, "_get_c_deep_ep", lambda: dep)
    mode_name = "PRIMUS_TURBO_JAX_DEEPEP_MODE"
    original_mode = os.environ.get(mode_name)
    envs.set_env(mode_name, None)
    envs.reset_cache()
    rt.reset_runtime()
    try:
        yield dep
    finally:
        rt.reset_runtime()
        envs.set_env(mode_name, original_mode)
        envs.reset_cache()


@pytest.mark.parametrize(
    "raw,mode",
    [
        (None, rt.MODE_INPROC),
        (" INPROC ", rt.MODE_INPROC),
        ("per-process", rt.MODE_PER_PROCESS),
        ("PER_PROCESS", rt.MODE_PER_PROCESS),
    ],
)
def test_modes(raw, mode):
    assert rt.LaunchMode.from_str(raw) is mode
    assert rt.get_target_name("dispatch", launch_mode=int(mode)) == f"dispatch_{mode.mode_name}"


def test_invalid_modes():
    with pytest.raises(ValueError, match="Unsupported JAX DeepEP mode"):
        rt.LaunchMode.from_str("other")
    with pytest.raises(ValueError, match="Unknown JAX DeepEP launch mode"):
        rt.get_target_name("dispatch", launch_mode=99)


def test_mode_lock():
    assert rt.get_launch_mode(lock=True) == 0
    envs.set_env("PRIMUS_TURBO_JAX_DEEPEP_MODE", "per_process")
    with pytest.raises(RuntimeError, match="already locked"):
        rt.get_mode()
    rt.reset_runtime()
    assert rt.get_mode(lock=True) is rt.MODE_PER_PROCESS
    assert rt.get_target_name("combine") == "combine_per_process"


@pytest.mark.parametrize(
    "local,world,explicit,expected",
    [(1, 4, None, "per_process"), (8, 1, None, None), (1, 1, None, None), (1, 4, "inproc", "inproc")],
)
def test_auto_detection(monkeypatch, local, world, explicit, expected):
    monkeypatch.setattr(jax, "local_device_count", lambda: local)
    monkeypatch.setattr(jax, "process_count", lambda: world)
    if explicit:
        envs.set_env("PRIMUS_TURBO_JAX_DEEPEP_MODE", explicit)
    rt.auto_detect_mode()
    assert envs.PRIMUS_TURBO_JAX_DEEPEP_MODE == expected


@pytest.mark.parametrize("ranks,match", [([], "non-empty"), ([2, 2], "unique"), ([0, 1], "not in ep_ranks")])
def test_invalid_ep_group(monkeypatch, ranks, match):
    monkeypatch.setattr(jax, "process_index", lambda: 2)
    with pytest.raises(ValueError, match=match):
        rt.set_ep_group(ranks)


def test_ep_group_identity_and_reset(monkeypatch):
    monkeypatch.setattr(jax, "process_index", lambda: 2)
    monkeypatch.setattr(jax, "process_count", lambda: 16)
    assert rt._ep_group_rank_or_world() == 2
    assert rt._ep_kv_key_suffix() == ""
    rt.set_ep_group([7, 2, 5])
    suffix = rt._ep_kv_key_suffix()
    assert suffix.startswith(":g")
    assert rt.get_ep_group_ranks() == (7, 2, 5)
    assert rt._ep_group_rank_or_world() == 1
    assert rt.MODE_PER_PROCESS.ep_size == 3
    rt.set_ep_group([7, 2, 5])
    assert rt._ep_kv_key_suffix() == suffix
    with pytest.raises(RuntimeError, match="already pinned"):
        rt.set_ep_group([2, 5, 7])
    rt.reset_runtime()
    rt.set_ep_group([2, 5, 7])
    assert rt._ep_kv_key_suffix() != suffix


def test_world_fallback_warning_once(monkeypatch, caplog):
    monkeypatch.setattr(jax, "process_count", lambda: 16)
    assert rt.MODE_PER_PROCESS.ep_size == 16
    assert rt.MODE_PER_PROCESS.ep_size == 16
    assert caplog.text.count("DeepEP EP group not pinned;") == 1


@pytest.mark.parametrize(
    "indices,axis,my_rank,expected",
    [
        ([[0, 1], [2, 3]], "expert", 2, (2, 3)),
        ([[0, 0], [1, 1]], "expert", 1, None),
        ([[0, 1], [2, 3]], "missing", 2, None),
    ],
)
def test_mesh_ep_group(monkeypatch, indices, axis, my_rank, expected):
    monkeypatch.setattr(jax, "process_index", lambda: my_rank)
    devices = np.array([[SimpleNamespace(process_index=i) for i in row] for row in indices], dtype=object)
    rt.pin_ep_group_from_jax_mesh(SimpleNamespace(axis_names=("data", "expert"), devices=devices), axis)
    assert rt.get_ep_group_ranks() == expected


def test_mesh_missing_process(monkeypatch):
    monkeypatch.setattr(jax, "process_index", lambda: 7)
    mesh = SimpleNamespace(
        axis_names=("expert",), devices=np.array([SimpleNamespace(process_index=0)], dtype=object)
    )
    with pytest.raises(RuntimeError, match="not present in mesh"):
        rt.pin_ep_group_from_jax_mesh(mesh)


@pytest.mark.parametrize(
    "mode,size,expected", [("inproc", 8, False), ("per_process", 8, False), ("per_process", 16, True)]
)
def test_internode_detection(monkeypatch, mode, size, expected):
    envs.set_env("PRIMUS_TURBO_JAX_DEEPEP_MODE", mode)
    monkeypatch.setattr(jax, "local_device_count", lambda: size)
    monkeypatch.setattr(jax, "process_count", lambda: size)
    assert rt.get_ep_size() == size
    assert rt.is_internode() is expected


def test_internode_requires_per_process(monkeypatch):
    monkeypatch.setattr(jax, "local_device_count", lambda: 16)
    with pytest.raises(RuntimeError, match="requires per_process"):
        rt.is_internode()
    with pytest.raises(ValueError, match="divisible"):
        rt._check_internode_rank_count(9)
    rt._check_internode_rank_count(16)


def test_kv_allgather_namespaces_and_binary_payload(monkeypatch):
    from jax._src.distributed import global_state

    client = Mock()
    client.blocking_key_value_get.side_effect = ["00ff", "0102", "00ff", "0102"]
    monkeypatch.setattr(global_state, "client", client)
    monkeypatch.setattr(rt, "_kv_allgather_call_id", 0)
    monkeypatch.setattr(rt, "_ep_group_id", "testgroup")
    for call_id in (1, 2):
        result = rt._kv_allgather_bytes("ipc", b"\x00\xff", 2, 0)
        assert result == [bytearray(b"\x00\xff"), bytearray(b"\x01\x02")]
        client.key_value_set.assert_called_with(f"primus_turbo_kvgather:ipc:c{call_id}:gtestgroup:0", "00ff")
        client.blocking_key_value_get.assert_called_with(
            f"primus_turbo_kvgather:ipc:c{call_id}:gtestgroup:1", 60000
        )


def test_barrier_labels_independent_of_ep_group(monkeypatch):
    from jax.experimental import multihost_utils

    barrier = Mock()
    monkeypatch.setattr(multihost_utils, "sync_global_devices", barrier)
    rt._bootstrap_barrier(1)
    barrier.assert_not_called()
    for i, group in enumerate(("group_a", "group_b"), 1):
        monkeypatch.setattr(rt, "_ep_group_id", group)
        rt._bootstrap_barrier(2)
        barrier.assert_called_with(f"primus_turbo.deep_ep.bootstrap.cid={i}")


@pytest.mark.parametrize("rank", [0, 7, 8, 15])
def test_rocshmem_root_matches_nvl_slot(monkeypatch, clean_runtime, rank):
    dep = clean_runtime
    dep.has_rocshmem.return_value = True
    dep.get_unique_id.return_value = b"local"
    gather = Mock(return_value=[bytes([i]) for i in range(16)])
    monkeypatch.setattr(rt, "_kv_allgather_bytes", gather)
    assert rt._get_root_rocshmem_unique_id(dep, rank, 16) == bytes([rank % 8])
    gather.assert_called_once_with("rocshmem_uid", b"local", 16, rank)


def test_rocshmem_missing(clean_runtime):
    clean_runtime.has_rocshmem.return_value = False
    with pytest.raises(RuntimeError, match="requires rocSHMEM"):
        rt._get_root_rocshmem_unique_id(clean_runtime, 0, 16)


@pytest.mark.parametrize("internode,fail_sync", [(False, False), (True, False), (False, True)])
def test_bootstrap_reuse_growth_and_cleanup(monkeypatch, clean_runtime, internode, fail_sync):
    dep = clean_runtime
    monkeypatch.setattr(jax, "process_index", lambda: 0)
    monkeypatch.setattr(jax, "process_count", lambda: 16 if internode else 2)
    barrier = Mock()
    monkeypatch.setattr(rt, "_bootstrap_barrier", barrier)
    monkeypatch.setattr(rt, "_get_root_rocshmem_unique_id", lambda *args: b"root")
    handles = [bytearray(b"handle")] * (16 if internode else 2)
    monkeypatch.setattr(rt, "_kv_allgather_bytes", lambda *args: handles)
    monkeypatch.setattr(rt, "_register_atexit_hook", Mock())
    dep.get_nvl_buffer_size_hint.return_value = 100
    dep.get_rdma_buffer_size_hint.return_value = 200
    dep.create_per_process_buffer.return_value = bytearray(b"handle")
    config = SimpleNamespace(
        num_sms=24,
        num_max_nvl_chunked_send_tokens=8,
        num_max_nvl_chunked_recv_tokens=8,
        num_max_rdma_chunked_send_tokens=8,
        num_max_rdma_chunked_recv_tokens=8,
    )
    if fail_sync:
        dep.sync_per_process_buffer.side_effect = RuntimeError("sync failed")
        with pytest.raises(RuntimeError, match="sync failed"):
            rt._bootstrap_per_process(hidden_bytes=128, config=config)
        dep.destroy_per_process_buffer.assert_called_once()
        assert rt._per_process_nvl_bytes == rt._per_process_rdma_bytes == 0
        return
    rt._bootstrap_per_process(hidden_bytes=128, config=config)
    assert rt._per_process_nvl_bytes == 100
    assert rt._per_process_rdma_bytes == (200 if internode else 0)
    dep.sync_per_process_buffer.assert_called_once_with(*((handles, b"root") if internode else (handles,)))
    dep.is_per_process_buffer_ready.return_value = True
    rt._bootstrap_per_process(hidden_bytes=128, config=config)
    assert barrier.call_count == dep.create_per_process_buffer.call_count == 1
    dep.get_nvl_buffer_size_hint.return_value = 300
    rt._bootstrap_per_process(hidden_bytes=256, config=config)
    assert barrier.call_count == dep.create_per_process_buffer.call_count == 2
    dep.destroy_per_process_buffer.assert_called_once()
    assert rt._per_process_nvl_bytes == 300


def test_runtime_entry_and_atexit(monkeypatch, clean_runtime):
    monkeypatch.setattr(jax, "local_device_count", lambda: 8)
    monkeypatch.setattr(jax, "process_count", lambda: 1)
    rt.ensure_deepep_runtime()
    rt.reset_runtime()
    envs.set_env("PRIMUS_TURBO_JAX_DEEPEP_MODE", "per_process")
    with pytest.raises(ValueError, match="hidden_bytes and config"):
        rt.ensure_deepep_runtime()
    bootstrap = Mock()
    monkeypatch.setattr(rt, "_bootstrap_per_process", bootstrap)
    cfg = object()
    rt.ensure_deepep_runtime(hidden_bytes=256, config=cfg)
    bootstrap.assert_called_once_with(hidden_bytes=256, config=cfg)
    hook = Mock()
    monkeypatch.setattr(rt.atexit, "register", hook)
    monkeypatch.setattr(rt, "_atexit_hook_registered", False)
    rt._register_atexit_hook()
    rt._register_atexit_hook()
    hook.assert_called_once_with(rt._destroy_at_exit)
    clean_runtime.get_source_meta_bytes.return_value = 80
    assert rt.get_source_meta_bytes() == 80
    clean_runtime.is_per_process_buffer_ready.return_value = True
    rt._destroy_at_exit()
    clean_runtime.destroy_per_process_buffer.assert_called_once()
