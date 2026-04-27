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


def _get_c_deep_ep():
    """Lazy import of the pybind ``_C.deep_ep`` submodule."""
    from primus_turbo.jax._C import deep_ep as _dep  # type: ignore[import-untyped]

    return _dep


def _bootstrap_per_process(*, hidden_bytes: int, config) -> None:
    """Create (or grow) the per-process IPC buffer and exchange handles.

    All processes must call this collectively; the IPC handle allgather acts as
    an implicit barrier.
    """
    global _per_process_nvl_bytes

    import numpy as np
    import jax.numpy as jnp
    from jax.experimental import multihost_utils

    dep = _get_c_deep_ep()
    rank = jax.process_index()
    num_ranks = jax.process_count()

    needed = dep.get_nvl_buffer_size_hint(
        hidden_bytes,
        num_ranks,
        config.num_sms,
        config.num_max_nvl_chunked_send_tokens,
        config.num_max_nvl_chunked_recv_tokens,
        config.num_max_rdma_chunked_send_tokens,
        config.num_max_rdma_chunked_recv_tokens,
    )

    if dep.is_per_process_buffer_ready() and _per_process_nvl_bytes >= needed:
        return

    if dep.is_per_process_buffer_ready():
        log.info(
            "Growing per-process DeepEP buffer: %d -> %d bytes",
            _per_process_nvl_bytes,
            needed,
        )
        dep.destroy_per_process_buffer()

    local_handle: bytearray = dep.create_per_process_buffer(rank, num_ranks, needed)

    handle_np = np.frombuffer(local_handle, dtype=np.uint8).copy()
    all_handles_jax = multihost_utils.process_allgather(jnp.array(handle_np))

    all_handles_np = np.asarray(all_handles_jax).reshape(num_ranks, -1)
    handles_list = [bytearray(all_handles_np[i]) for i in range(num_ranks)]

    dep.sync_per_process_buffer(handles_list)
    _per_process_nvl_bytes = needed

    log.info(
        "Per-process DeepEP buffer ready: rank=%d, num_ranks=%d, nvl_bytes=%d",
        rank,
        num_ranks,
        needed,
    )


# ---------------------------------------------------------------------------
#  ensure_deepep_runtime  (public entry point)
# ---------------------------------------------------------------------------


def ensure_deepep_runtime(*, hidden_bytes: Optional[int] = None, config=None) -> None:
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
    global _locked_mode, _per_process_nvl_bytes
    _locked_mode = None
    _per_process_nvl_bytes = 0
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
