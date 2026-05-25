###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from __future__ import annotations

import hashlib
import logging
import os
from enum import IntEnum
from typing import List, Optional, Sequence, Tuple

import jax

_MODE_ENV_VAR = "PRIMUS_TURBO_JAX_DEEPEP_MODE"

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
#  EP-group state (decouple "EP rank set" from "JAX world")
# ---------------------------------------------------------------------------
#
# Mirrors the torch ``deep_ep.Buffer(group=ProcessGroup, ...)`` design: the EP
# communication domain may be a strict subset of all JAX processes (e.g. when
# only one mesh axis carries expert parallelism while others carry FSDP / data
# parallelism).  When set, all rank/num_ranks computations below switch to the
# group-relative view.  When unset, behavior falls back to the legacy
# "EP == world" assumption (kept for backward compatibility with existing
# 1-node/proc and single-node 1-GPU/proc users).
#
# Storage is module-global because the C++ Buffer singleton lives in
# ``primus_turbo.jax._C.deep_ep`` and must agree on rank/num_ranks across
# every Python entry point that touches it.

_ep_group_ranks: Optional[Tuple[int, ...]] = None
_ep_group_id: Optional[str] = None  # short hash, used as KV-store key suffix
_ep_fallback_warned: bool = False  # one-shot guard for the "no pin" WARNING


def set_ep_group(ep_ranks: Sequence[int]) -> None:
    """Pin the EP communication domain to a specific ``jax.process_index`` set.

    Must be called collectively (all processes in ``ep_ranks`` pass the same
    ordered tuple) before the first ``get_mode(lock=True)`` / DeepEP call.

    Arguments:
        ep_ranks: ordered list of ``jax.process_index()`` values that form
            this EP group.  Order defines rank-in-group: ``ep_ranks[k]`` is
            EP-rank ``k``.  ``jax.process_index()`` of the calling process
            must appear in the list.
    """
    global _ep_group_ranks, _ep_group_id

    ranks = tuple(int(r) for r in ep_ranks)
    if len(ranks) == 0:
        raise ValueError("ep_ranks must be non-empty")
    if len(set(ranks)) != len(ranks):
        raise ValueError(f"ep_ranks must be unique, got duplicates in {ranks}")
    my_proc = jax.process_index()
    if my_proc not in ranks:
        raise ValueError(
            f"jax.process_index()={my_proc} is not in ep_ranks={ranks}; "
            "every process that calls set_ep_group must include itself in the group."
        )

    # Reject silent reconfiguration.  If the same group is already pinned
    # (collective re-entry from a fresh layer), it's a no-op.
    if _ep_group_ranks is not None and _ep_group_ranks != ranks:
        raise RuntimeError(
            "EP group already pinned; cannot change. "
            f"Existing={_ep_group_ranks}, new={ranks}. "
            "Call reset_runtime() between distinct EP groups."
        )

    _ep_group_ranks = ranks
    # Stable, collision-resistant id derived from the ordered rank tuple
    # (all members compute the same id locally — no cross-process exchange).
    _ep_group_id = hashlib.blake2b(",".join(str(r) for r in ranks).encode(), digest_size=8).hexdigest()
    log.info(
        "Pinned EP group: size=%d, members=%s, id=%s",
        len(ranks),
        ranks,
        _ep_group_id,
    )


def get_ep_group_ranks() -> Optional[Tuple[int, ...]]:
    """Return the pinned EP group (ordered tuple), or None if not set."""
    return _ep_group_ranks


def pin_ep_group_from_jax_mesh(mesh, axis_name: str = "expert") -> None:
    """Derive the EP group from a ``jax.sharding.Mesh`` axis and pin it.

    Convenience wrapper around ``set_ep_group``.  Walks ``mesh.devices`` along
    ``axis_name`` for the calling process's slot, collects the
    ``process_index`` of every EP-axis peer in mesh order, and pins them as
    the EP group.

    No-ops in two cases (logged, no exception):
      * the mesh has no axis named ``axis_name``;
      * the EP axis is intra-process (every entry shares the same
        ``process_index``) — i.e. 1-node/proc mode, where the legacy
        "EP == jax.local_device_count()" path in INPROC mode (or
        "EP == process_count" path in PER_PROCESS mode) is already correct.

    Idempotent: a second call with the same group is a no-op (delegated to
    ``set_ep_group``).  Reconfiguration with a different group raises.

    Must be called collectively by every process in the EP group, before
    the first ``ensure_deepep_runtime`` / dispatch / combine call.
    """
    if axis_name not in mesh.axis_names:
        log.info("EP group not pinned: mesh has no axis named %r.", axis_name)
        return
    # Local import — keep numpy out of the module top-level; matches the
    # rest of this file's "import inside function" style for optional deps.
    import numpy as np  # type: ignore[import-untyped]

    expert_axis_idx = mesh.axis_names.index(axis_name)
    devices_arr = np.asarray(mesh.devices)
    my_proc = jax.process_index()
    coords_my = None
    for idx in np.ndindex(*devices_arr.shape):
        if devices_arr[idx].process_index == my_proc:
            coords_my = idx
            break
    if coords_my is None:
        raise RuntimeError(
            f"jax.process_index()={my_proc} not present in mesh.devices; " "cannot derive EP group."
        )
    ep_ranks: List[int] = []
    for v in range(devices_arr.shape[expert_axis_idx]):
        coords = list(coords_my)
        coords[expert_axis_idx] = v
        ep_ranks.append(int(devices_arr[tuple(coords)].process_index))
    if len(set(ep_ranks)) != len(ep_ranks):
        log.info(
            "EP group not pinned: %r axis is intra-process "
            "(every peer shares process_index=%d); legacy 'EP == world' "
            "semantics are correct here.",
            axis_name,
            my_proc,
        )
        return
    set_ep_group(ep_ranks)


def _ep_group_size_or_world() -> int:
    """EP-aware ep_size.  Falls back to ``jax.process_count()`` when no
    group is pinned (legacy "EP == world" semantics).

    Emits a one-shot WARNING on the fallback path when the JAX world has
    more than one process: in that regime the fallback is correct only when
    the EP axis really spans the whole world; otherwise it silently
    over-sizes IPC/RDMA buffers and inflates ``num_worst_tokens`` by the
    ``world_size / ep_size`` factor.  Call ``pin_ep_group_from_jax_mesh``
    before the first DeepEP entry point to silence it.
    """
    if _ep_group_ranks is not None:
        return len(_ep_group_ranks)

    global _ep_fallback_warned
    nproc = jax.process_count()
    if not _ep_fallback_warned and nproc > 1:
        log.warning(
            "DeepEP EP group not pinned; falling back to ep_size = "
            "jax.process_count() = %d. If the EP axis is a strict subset of "
            "the JAX world (e.g. multi-axis 1-GPU/proc mesh), call "
            "primus_turbo.jax.deep_ep.runtime.pin_ep_group_from_jax_mesh"
            "(mesh) BEFORE the first DeepEP call to avoid over-allocation.",
            nproc,
        )
        _ep_fallback_warned = True
    return nproc


def _ep_group_rank_or_world() -> int:
    """Rank of the calling process within the EP group, or
    ``jax.process_index()`` when no group is pinned."""
    if _ep_group_ranks is None:
        return jax.process_index()
    return _ep_group_ranks.index(jax.process_index())


def _ep_kv_key_suffix() -> str:
    """Per-EP-group KV-store key suffix.  Empty string when no group is pinned
    (preserves byte-for-byte compatibility with legacy single-group runs)."""
    return "" if _ep_group_id is None else f":g{_ep_group_id}"


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
        # PER_PROCESS: prefer the pinned EP group when set; otherwise fall back
        # to ``jax.process_count()`` so legacy single-group runs (where
        # process_count == ep_size by construction) keep working unchanged.
        return _ep_group_size_or_world()

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

    Most users do not need to call this directly: both ``warmup()`` and
    ``ensure_deepep_runtime()`` invoke it on their behalf before the mode
    is locked.  It is exposed publicly for setups that want to detect /
    lock the launch mode early (e.g. before ``set_ep_group``).

    When each JAX process owns exactly one GPU (ONE_GPU_PER_PROCESS),
    DeepEP must use inter-process IPC buffers instead of intra-process
    shared memory.  Call this before the first ``get_mode(lock=True)``
    so that the env var is in place before the mode is locked.

    Heuristic: ``jax.local_device_count() == 1`` is the reliable trigger
    for PER_PROCESS — it's true under any multi-process launcher,
    regardless of whether multiple processes share a node or span nodes,
    or whether the EP communication domain is a strict subset of the
    JAX world (see ``set_ep_group``).
    """
    if os.environ.get(_MODE_ENV_VAR) is not None:
        return
    if jax.local_device_count() == 1 and jax.process_count() > 1:
        os.environ[_MODE_ENV_VAR] = "per_process"
        log.info(
            "Auto-detected per_process mode " "(1 GPU per process, %d JAX processes, ep_group_size=%d)",
            jax.process_count(),
            _ep_group_size_or_world(),
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


def get_target_name(op_name: str, *, launch_mode: Optional[int] = None, lock: bool = False) -> str:
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


def get_source_meta_bytes() -> int:
    """Return the per-token ``recv_src_meta`` byte width used by internode dispatch.

    Public wrapper around the C++ ``deep_ep::internode::get_source_meta_bytes``
    binding. Exposed here so callers (e.g. ``primus_turbo.jax.lax.moe``) don't
    need to reach into the private ``_get_c_deep_ep`` helper or the
    ``primus_turbo.jax._C.deep_ep`` submodule directly.
    """
    return _get_c_deep_ep().get_source_meta_bytes()


def is_internode(*, lock: bool = False) -> bool:
    """Return True if ep_size > NUM_MAX_NVL_PEERS (internode communication).

    ``ep_size`` is the EP-group size (``len(ep_ranks)`` when a group is pinned;
    otherwise ``jax.process_count()``).  This decouples "EP needs RDMA" from
    "JAX world spans multiple nodes" — e.g. a 64-process job with EP-group=8
    is intranode-EP even though the world is multi-node.
    """
    mode = get_mode(lock=lock)
    if mode.ep_size <= NUM_MAX_NVL_PEERS:
        return False
    if mode is not MODE_PER_PROCESS:
        raise RuntimeError(
            "JAX DeepEP internode communication currently requires per_process mode. "
            f"Got mode='{mode.mode_name}' with ep_size={mode.ep_size}."
        )
    return True


_kv_allgather_call_id: int = 0


def _kv_allgather_bytes(
    label: str,
    payload,
    num_ranks: int,
    my_rank: int,
    *,
    rank_to_world: Optional[Sequence[int]] = None,
) -> list[bytearray]:
    """Bytes-level allgather via JAX distributed KV-store.

    Bypasses a known ``jax.experimental.multihost_utils.process_allgather`` bug
    under XLA/RCCL where one random rank's slot in the gathered output is
    silently zero-filled.

    Arguments:
        label, payload, num_ranks, my_rank: as before; ranks here are
            *EP-group-relative* (0..num_ranks-1).
        rank_to_world: optional EP-rank → ``jax.process_index`` mapping used
            only to disambiguate KV keys when multiple non-overlapping EP
            groups co-exist (e.g. 8 EP groups of size 8 sharing a 64-process
            JAX world).  Each group writes/reads under its own key suffix
            (derived from ``_ep_kv_key_suffix()``), so no cross-group races.

    Each call uses a fresh, monotonically increasing call ID so concurrent
    allgathers under different labels don't collide on the same key prefix.
    """
    del rank_to_world  # group identity already encoded via _ep_kv_key_suffix
    global _kv_allgather_call_id

    from jax._src.distributed import global_state as _gs

    client = _gs.client
    _kv_allgather_call_id += 1
    cid = _kv_allgather_call_id
    prefix = f"primus_turbo_kvgather:{label}:c{cid}{_ep_kv_key_suffix()}"
    client.key_value_set(f"{prefix}:{my_rank}", bytes(payload).hex())
    out: list[bytearray] = []
    for r in range(num_ranks):
        hexstr = client.blocking_key_value_get(f"{prefix}:{r}", 60_000)
        out.append(bytearray.fromhex(hexstr))
    return out


def _check_internode_rank_count(num_ranks: int) -> None:
    if num_ranks % NUM_MAX_NVL_PEERS != 0:
        raise ValueError(
            f"Internode DeepEP requires ep_size to be divisible by "
            f"NUM_MAX_NVL_PEERS={NUM_MAX_NVL_PEERS}, got ep_size={num_ranks}"
        )


def _get_root_rocshmem_unique_id(dep, rank: int, num_ranks: int) -> bytes:
    """Collect and return the root rocSHMEM unique ID for this rank's NVL slot.

    Every process generates a same-sized ID for KV-store allgather, but only
    the ID from ``rdma_rank == 0`` for the same NVL slot is passed to C++.
    """
    if not dep.has_rocshmem():
        raise RuntimeError(
            "Internode DeepEP requires rocSHMEM but it was not available at build time. "
            "Set ROCSHMEM_HOME / MPI_HOME and reinstall."
        )

    rdma_rank = rank // NUM_MAX_NVL_PEERS

    uid_bytes = bytes(dep.get_unique_id())

    # KV-store allgather, not multihost_utils.process_allgather: the latter has
    # been observed to silently zero-fill one rank's slot under XLA/RCCL.
    all_uids = _kv_allgather_bytes("rocshmem_uid", uid_bytes, num_ranks, rank)

    nvl_rank = rank % NUM_MAX_NVL_PEERS
    root_global_rank = nvl_rank  # rdma_rank==0 on same NVL slot
    root_uid = bytes(all_uids[root_global_rank])
    log.info(
        "rocSHMEM root unique ID gathered: rank=%d, rdma_rank=%d",
        rank,
        rdma_rank,
    )
    return root_uid


_bootstrap_barrier_call_id = 0


def _bootstrap_barrier(num_ranks: int) -> None:
    """Cross-process barrier before any C++ Buffer / rocSHMEM allocation.

    Why this barrier exists:

    Without it, internode DeepEP bootstrap (``dep.create_per_process_buffer``
    + ``_kv_allgather_bytes`` + ``dep.sync_per_process_buffer`` →
    rocSHMEM endpoint handshake) silently fails on ~50% of multi-host
    runs (3/6 observed under DeepSeek-V3 671B, 64-process, dcn_ep=2,
    ici_ep=8).  Failure mode: a subset of ranks (typically 2–3 of 64)
    silently stop sending heartbeats, JAX coordination service ~30 s
    later marks them unhealthy and ``PollForError`` LOG(FATAL)s the
    remaining ranks; the silently-died ranks leave no core (or only the
    PollForError-victim cores with C++ stacks parked in libc).

    Mechanism (best inference, exact C++ point not pinpointed): rocSHMEM
    RDMA endpoint setup inside ``sync_per_process_buffer`` is a two-way
    handshake; if peer A enters meaningfully earlier than peer B, A's
    handshake retries time out and the C++ side bails with a hang or
    abort that the heartbeat thread observes.  Forcing every rank to
    enter the C++ allocation path within the same global-barrier window
    eliminates the skew.

    Adding the barrier here, *inside* primus_turbo, means callers
    (MaxText, user model code, tests) do not need to know about this
    invariant.  ``sync_global_devices`` is a JAX-distributed-coordinator
    call (KV-store backed), not a GPU/RCCL collective, so it does not
    require GPU readiness and is safe to invoke before any DeepEP C++
    state exists.

    Statistics: 3/6 init crashes pre-fix → 0/12 post-fix
    (P(0/12 | true rate=0.5) ≈ 0.024 %).

    Fast-path note: callers should invoke this *only* when actually
    growing / allocating the buffer; idempotent
    ``is_per_process_buffer_ready`` runs skip the barrier so steady-state
    training does not pay the cost.

    Label design: ``sync_global_devices`` is a **world-wide** barrier
    that uses its label only as a coordinator-key disambiguator; every
    JAX process must call with the *same* label for the barrier to
    complete.  We use a monotonic call counter (incremented in lock-step
    across all processes that hit this code path) so the label cannot
    accidentally diverge between EP groups in multi-EP-group runs.
    Specifically, we do **not** include EP-group-id or buffer-size in
    the label — those are EP-group-local values that would deadlock
    multi-group runs.

    Arguments:
        num_ranks: EP-group size; barrier is a no-op for single-process
            runs (no peers to sync with).
    """
    global _bootstrap_barrier_call_id

    if num_ranks <= 1:
        return
    try:
        # Local import: ``sync_global_devices`` is part of
        # ``jax.experimental`` and may not exist on every JAX version
        # primus_turbo claims to support.  Failure is non-fatal — we
        # log and proceed (callers historically ran without a barrier).
        from jax.experimental.multihost_utils import sync_global_devices
    except ImportError:
        log.warning(
            "jax.experimental.multihost_utils.sync_global_devices unavailable; "
            "DeepEP bootstrap may race on multi-host runs."
        )
        return

    _bootstrap_barrier_call_id += 1
    label = f"primus_turbo.deep_ep.bootstrap.cid={_bootstrap_barrier_call_id}"
    try:
        sync_global_devices(label)
    except Exception:  # pragma: no cover  pylint: disable=broad-except
        # A failed barrier is strictly worse than no barrier — propagate so
        # the caller's existing exception path destroys partial C++ state.
        log.exception("DeepEP bootstrap barrier %r failed", label)
        raise


def _bootstrap_per_process(*, hidden_bytes: int, config) -> None:
    """Create (or grow) the per-process IPC buffer and exchange handles.

    All processes in the EP group must call this collectively; the IPC
    handle allgather acts as an implicit barrier.

    When an EP group is pinned via ``set_ep_group``, ``rank`` and
    ``num_ranks`` here are *EP-group-relative* (so the C++ ``Buffer`` —
    which is sized by these values — only allocates the
    intersection it actually communicates over).  When no group is pinned,
    they fall back to ``jax.process_index()`` / ``jax.process_count()`` so
    legacy single-group runs are unchanged.

    Internally inserts a cross-process barrier (``_bootstrap_barrier``)
    before any C++ ``Buffer`` allocation so callers do not need to wrap
    this call in their own ``sync_global_devices``.  See
    ``_bootstrap_barrier`` docstring for the failure mode that motivated
    it.  Idempotent calls (buffer already large enough) skip the barrier.
    """
    global _per_process_nvl_bytes, _per_process_rdma_bytes

    dep = _get_c_deep_ep()
    rank = _ep_group_rank_or_world()
    num_ranks = _ep_group_size_or_world()

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

    # Fast-path: buffer already large enough.  Skip the cross-process
    # barrier so steady-state training does not pay the cost.
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

    # Slow-path: about to allocate / grow.  Sync all EP-group peers so
    # rocSHMEM RDMA endpoint handshake (inside sync_per_process_buffer)
    # sees all peers within the same global-barrier window.  Without
    # this, ~50 % of multi-host runs silently lose 2–3 ranks here.
    _bootstrap_barrier(num_ranks)

    root_uid = None
    if internode:
        root_uid = _get_root_rocshmem_unique_id(dep, rank, num_ranks)

    try:
        local_ipc_handle: bytearray = dep.create_per_process_buffer(
            rank, num_ranks, num_nvl_bytes, num_rdma_bytes
        )

        # KV-store allgather, not multihost_utils.process_allgather: see
        # _kv_allgather_bytes docstring for the XLA/RCCL zero-fill bug.
        ipc_handles_list = _kv_allgather_bytes("ipc_handle", bytes(local_ipc_handle), num_ranks, rank)

        if root_uid is None:
            dep.sync_per_process_buffer(ipc_handles_list)
        else:
            dep.sync_per_process_buffer(ipc_handles_list, root_uid)
        _per_process_nvl_bytes = num_nvl_bytes
        _per_process_rdma_bytes = num_rdma_bytes
    except BaseException:
        # If bootstrap fails after the local IPC buffer is created, leave the C++
        # singleton in a clean state instead of relying on process teardown.
        try:
            dep.destroy_per_process_buffer()
        finally:
            _per_process_nvl_bytes = 0
            _per_process_rdma_bytes = 0
        raise

    log.info(
        "Per-process DeepEP buffer ready: rank=%d, num_ranks=%d, "
        "nvl_bytes=%d, rdma_bytes=%d, internode=%s, ep_group_id=%s",
        rank,
        num_ranks,
        num_nvl_bytes,
        num_rdma_bytes,
        internode,
        _ep_group_id or "(world)",
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
        raise ValueError("hidden_bytes and config are required for per_process mode bootstrap")
    _bootstrap_per_process(hidden_bytes=hidden_bytes, config=config)


# ---------------------------------------------------------------------------
#  Reset
# ---------------------------------------------------------------------------


def reset_runtime() -> None:
    global _locked_mode, _per_process_nvl_bytes, _per_process_rdma_bytes
    global _ep_group_ranks, _ep_group_id, _ep_fallback_warned
    global _bootstrap_barrier_call_id
    _locked_mode = None
    _per_process_nvl_bytes = 0
    _per_process_rdma_bytes = 0
    _ep_group_ranks = None
    _ep_group_id = None
    _ep_fallback_warned = False
    _bootstrap_barrier_call_id = 0
    try:
        dep = _get_c_deep_ep()
        if dep.is_per_process_buffer_ready():
            dep.destroy_per_process_buffer()
    except ImportError:
        pass


# ---------------------------------------------------------------------------
#  Internal helpers
# ---------------------------------------------------------------------------


def _resolve_mode(launch_mode: Optional[int], *, lock: bool = False) -> LaunchMode:
    if launch_mode is None:
        return get_mode(lock=lock)
    try:
        return LaunchMode(launch_mode)
    except ValueError as exc:
        raise ValueError(f"Unknown JAX DeepEP launch mode: {launch_mode}") from exc
