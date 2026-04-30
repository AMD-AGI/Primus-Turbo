###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from __future__ import annotations

import logging
import os
from enum import IntEnum
from typing import Optional

import jax

_MODE_ENV_VAR = "PRIMUS_TURBO_JAX_DEEPEP_MODE"

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
#  LaunchMode enum
# ---------------------------------------------------------------------------


class LaunchMode(IntEnum):
    INPROC = 0
    PER_PROCESS = 1

    @property
    def mode_name(self) -> str:
        if self is LaunchMode.INPROC:
            return "inproc"
        if self is LaunchMode.PER_PROCESS:
            return "per_process"
        raise ValueError(f"Unsupported JAX DeepEP mode: {self!r}")

    @property
    def ep_size(self) -> int:
        if self is LaunchMode.INPROC:
            return jax.local_device_count()
        return jax.process_count()

    def target_name(self, op_name: str) -> str:
        return f"{op_name}_{self.mode_name}"

    @classmethod
    def from_str(cls, raw_mode: Optional[str]) -> LaunchMode:
        normalized = (raw_mode or "inproc").strip().lower().replace("-", "_")
        if normalized == "inproc":
            return cls.INPROC
        if normalized == "per_process":
            return cls.PER_PROCESS
        raise ValueError(
            f"Unsupported JAX DeepEP mode '{raw_mode}'. "
            f"Expected one of: {[mode.mode_name for mode in cls]}"
        )


MODE_INPROC = LaunchMode.INPROC
MODE_PER_PROCESS = LaunchMode.PER_PROCESS


# ---------------------------------------------------------------------------
#  Global mode state
# ---------------------------------------------------------------------------

_locked_mode: Optional[LaunchMode] = None


def _get_env_mode() -> LaunchMode:
    return LaunchMode.from_str(os.environ.get(_MODE_ENV_VAR))


def _check_locked_mode(mode: LaunchMode) -> None:
    if _locked_mode is not None and _locked_mode != mode:
        raise RuntimeError(
            "JAX DeepEP mode was already locked to "
            f"'{_locked_mode.mode_name}', but {_MODE_ENV_VAR} is now '{mode.mode_name}'. "
            "Set the mode before the first DeepEP call, or reset the runtime state in tests."
        )


def auto_detect_mode() -> None:
    """Set PRIMUS_TURBO_JAX_DEEPEP_MODE if not already set.

    When each JAX process owns exactly one GPU (ONE_GPU_PER_PROCESS),
    DeepEP must use inter-process IPC buffers instead of intra-process
    shared memory.  Call this before the first ``get_mode(lock=True)``
    so that the env var is in place before the mode is locked.
    """
    if os.environ.get(_MODE_ENV_VAR) is not None:
        return
    if jax.local_device_count() == 1 and jax.process_count() > 1:
        os.environ[_MODE_ENV_VAR] = "per_process"
        log.info("Auto-detected per_process mode (1 GPU per process, %d processes)", jax.process_count())


def get_mode(*, lock: bool = False) -> LaunchMode:
    global _locked_mode

    mode = _get_env_mode()
    _check_locked_mode(mode)
    if lock and _locked_mode is None:
        _locked_mode = mode
    return _locked_mode or mode


def get_launch_mode(*, lock: bool = False) -> int:
    return int(get_mode(lock=lock))


def get_ep_size(*, lock: bool = False) -> int:
    return get_mode(lock=lock).ep_size


def get_target_name(
    op_name: str, *, launch_mode: Optional[int] = None, lock: bool = False
) -> str:
    return _resolve_mode(launch_mode, lock=lock).target_name(op_name)


# ---------------------------------------------------------------------------
#  Per-process buffer bootstrap
# ---------------------------------------------------------------------------

_per_process_nvl_bytes: int = 0
_per_process_rdma_bytes: int = 0

NUM_MAX_NVL_PEERS = 8


def _get_c_deep_ep():
    """Lazy import of the pybind ``_C.deep_ep`` submodule."""
    from primus_turbo.jax._C import deep_ep as _dep  # type: ignore[import-untyped]

    return _dep


def is_internode(*, lock: bool = False) -> bool:
    """Return True if ep_size > NUM_MAX_NVL_PEERS (internode communication)."""
    mode = get_mode(lock=lock)
    if mode.ep_size <= NUM_MAX_NVL_PEERS:
        return False
    if mode is not MODE_PER_PROCESS:
        raise RuntimeError(
            "JAX DeepEP internode communication currently requires per_process mode. "
            f"Got mode='{mode.mode_name}' with ep_size={mode.ep_size}."
        )
    return True


def _check_internode_rank_count(num_ranks: int) -> None:
    if num_ranks % NUM_MAX_NVL_PEERS != 0:
        raise ValueError(
            f"Internode DeepEP requires process_count to be divisible by "
            f"NUM_MAX_NVL_PEERS={NUM_MAX_NVL_PEERS}, got {num_ranks}"
        )


def _get_root_rocshmem_unique_id(dep, rank: int, num_ranks: int) -> bytes:
    """Collect and return the root rocSHMEM unique ID for this rank's NVL slot.

    Every process generates a same-sized ID for JAX array allgather, but only
    the ID from ``rdma_rank == 0`` for the same NVL slot is passed to C++.
    """
    if not dep.has_rocshmem():
        raise RuntimeError(
            "Internode DeepEP requires rocSHMEM but it was not available at build time. "
            "Set ROCSHMEM_HOME / MPI_HOME and reinstall."
        )

    import numpy as np
    import jax.numpy as jnp
    from jax.experimental import multihost_utils

    rdma_rank = rank // NUM_MAX_NVL_PEERS

    # Every process generates a fixed-size rocSHMEM unique ID so JAX's array allgather can use a
    # uniform shape. Only the ID from the root RDMA rank of each NVL slot is used below.
    uid_bytes = dep.get_unique_id()

    uid_np = np.frombuffer(uid_bytes, dtype=np.uint8).copy()
    all_uids_jax = multihost_utils.process_allgather(jnp.array(uid_np))
    all_uids_np = np.asarray(all_uids_jax).reshape(num_ranks, -1)

    nvl_rank = rank % NUM_MAX_NVL_PEERS
    root_global_rank = nvl_rank  # rdma_rank==0 on same NVL slot
    root_uid = bytes(all_uids_np[root_global_rank])
    log.info(
        "rocSHMEM root unique ID gathered: rank=%d, rdma_rank=%d",
        rank, rdma_rank,
    )
    return root_uid


def _bootstrap_per_process(*, hidden_bytes: int, config) -> None:
    """Create (or grow) the per-process IPC buffer and exchange handles.

    All processes must call this collectively; the IPC handle allgather acts as
    an implicit barrier.
    """
    global _per_process_nvl_bytes, _per_process_rdma_bytes

    import numpy as np
    import jax.numpy as jnp
    from jax.experimental import multihost_utils

    dep = _get_c_deep_ep()
    rank = jax.process_index()
    num_ranks = jax.process_count()

    internode = num_ranks > NUM_MAX_NVL_PEERS
    if internode:
        _check_internode_rank_count(num_ranks)

    num_nvl_bytes = dep.get_nvl_buffer_size_hint(
        hidden_bytes,
        num_ranks,
        config.num_sms,
        config.num_max_nvl_chunked_send_tokens,
        config.num_max_nvl_chunked_recv_tokens,
        config.num_max_rdma_chunked_send_tokens,
        config.num_max_rdma_chunked_recv_tokens,
    )

    num_rdma_bytes = 0
    if internode:
        num_rdma_bytes = dep.get_rdma_buffer_size_hint(
            hidden_bytes,
            num_ranks,
            config.num_sms,
            config.num_max_nvl_chunked_send_tokens,
            config.num_max_nvl_chunked_recv_tokens,
            config.num_max_rdma_chunked_send_tokens,
            config.num_max_rdma_chunked_recv_tokens,
        )

    if (
        dep.is_per_process_buffer_ready()
        and _per_process_nvl_bytes >= num_nvl_bytes
        and _per_process_rdma_bytes >= num_rdma_bytes
    ):
        return

    if dep.is_per_process_buffer_ready():
        log.info(
            "Growing per-process DeepEP buffer: nvl %d -> %d bytes, rdma %d -> %d bytes",
            _per_process_nvl_bytes,
            num_nvl_bytes,
            _per_process_rdma_bytes,
            num_rdma_bytes,
        )
        dep.destroy_per_process_buffer()
        _per_process_nvl_bytes = 0
        _per_process_rdma_bytes = 0

    root_uid = None
    if internode:
        root_uid = _get_root_rocshmem_unique_id(dep, rank, num_ranks)

    local_ipc_handle: bytearray = dep.create_per_process_buffer(
        rank, num_ranks, num_nvl_bytes, num_rdma_bytes
    )

    local_ipc_handle_np = np.frombuffer(local_ipc_handle, dtype=np.uint8).copy()
    all_ipc_handles_jax = multihost_utils.process_allgather(jnp.array(local_ipc_handle_np))

    all_ipc_handles_np = np.asarray(all_ipc_handles_jax).reshape(num_ranks, -1)
    ipc_handles_list = [bytearray(all_ipc_handles_np[i]) for i in range(num_ranks)]

    if root_uid is None:
        dep.sync_per_process_buffer(ipc_handles_list)
    else:
        dep.sync_per_process_buffer(ipc_handles_list, root_uid)
    _per_process_nvl_bytes = num_nvl_bytes
    _per_process_rdma_bytes = num_rdma_bytes

    log.info(
        "Per-process DeepEP buffer ready: rank=%d, num_ranks=%d, "
        "nvl_bytes=%d, rdma_bytes=%d, internode=%s",
        rank, num_ranks, num_nvl_bytes, num_rdma_bytes, internode,
    )


# ---------------------------------------------------------------------------
#  ensure_deepep_runtime  (public entry point)
# ---------------------------------------------------------------------------


def ensure_deepep_runtime(*, hidden_bytes: Optional[int] = None, config=None) -> None:
    auto_detect_mode()
    mode = get_mode(lock=True)
    if mode is MODE_INPROC:
        return

    if hidden_bytes is None or config is None:
        raise ValueError(
            "hidden_bytes and config are required for per_process mode bootstrap"
        )
    _bootstrap_per_process(hidden_bytes=hidden_bytes, config=config)


# ---------------------------------------------------------------------------
#  Reset
# ---------------------------------------------------------------------------


def reset_runtime() -> None:
    global _locked_mode, _per_process_nvl_bytes, _per_process_rdma_bytes
    _locked_mode = None
    _per_process_nvl_bytes = 0
    _per_process_rdma_bytes = 0
    try:
        dep = _get_c_deep_ep()
        if dep.is_per_process_buffer_ready():
            dep.destroy_per_process_buffer()
    except ImportError:
        pass


# ---------------------------------------------------------------------------
#  Internal helpers
# ---------------------------------------------------------------------------


def _resolve_mode(
    launch_mode: Optional[int], *, lock: bool = False
) -> LaunchMode:
    if launch_mode is None:
        return get_mode(lock=lock)
    try:
        return LaunchMode(launch_mode)
    except ValueError as exc:
        raise ValueError(f"Unknown JAX DeepEP launch mode: {launch_mode}") from exc
