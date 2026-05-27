###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""User-facing DeepEP dispatch / combine API.

The public surface is intentionally small:

  • :func:`setup` — one-call bootstrap, required exactly once before the
    first :func:`moe_dispatch` / :func:`moe_combine`.  Handles all three
    runtime modes (INPROC, PER_PROCESS intranode, PER_PROCESS internode).
  • :func:`moe_dispatch` / :func:`moe_combine` — the actual MoE all-to-all.
  • :class:`Config` — tuning knobs (advanced).
  • :func:`set_ep_group` — pin EP group without a ``jax.sharding.Mesh``
    (advanced).
  • :func:`get_ep_size` — query EP-group size from the frozen state
    (e.g. when sharding code needs it).
  • :func:`reset_runtime` — tests / multi-job re-entry.

After ``setup()`` returns, the runtime state (mode, ep_size, num_sms,
is_internode, …) is *frozen* — captured into an immutable
:class:`FrozenRuntimeState` snapshot.  Subsequent dispatch/combine calls
read from the snapshot rather than re-querying mutable globals, which
both:

  * **Tightens the contract**: forgetting ``setup()`` raises a clear
    ``RuntimeError`` rather than dying inside C++ with
    ``"per_process buffer not initialized"``.
  * **Simplifies trace path**: each ``moe_dispatch`` no longer reruns
    ``auto_detect_mode`` + 3 ``lock=True`` queries + a buffer-ready check
    on every trace.

Performance: freezing has zero impact at training steady state (the
dispatch/combine Python bodies don't run inside jit-cached jaxprs), and
saves ~5–10 µs per trace.
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from functools import partial
from typing import Optional, Sequence, Tuple, Union

import jax
import jax.core
import jax.numpy as jnp

from primus_turbo.jax.deep_ep import runtime as deep_ep_runtime
from primus_turbo.jax.deep_ep.runtime import (
    MODE_INPROC,  # re-export for callers that introspect mode (e.g. tests)
)
from primus_turbo.jax.deep_ep.runtime import (
    MODE_PER_PROCESS,
    NUM_MAX_NVL_PEERS,
    LaunchMode,
)
from primus_turbo.jax.primitive.moe.moe_combine import (
    moe_combine_p,
    moe_internode_combine_p,
)
from primus_turbo.jax.primitive.moe.moe_dispatch import (
    moe_cached_dispatch_p,
    moe_dispatch_p,
    moe_internode_cached_dispatch_p,
    moe_internode_dispatch_p,
)

from .moe_utils import Config

__all__ = [
    "Config",
    "get_ep_size",
    "is_internode",
    "moe_combine",
    "moe_dispatch",
    "reset_runtime",
    "set_ep_group",
    "setup",
]


# ---------------------------------------------------------------------------
#  Frozen runtime state
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FrozenRuntimeState:
    """Immutable DeepEP runtime snapshot, captured at ``setup()`` time.

    All fields are read by :func:`moe_dispatch` / :func:`moe_combine` on
    every trace; freezing them here means we never have to re-query the
    mutable ``deep_ep.runtime`` globals (mode, EP group, num_sms, ...)
    in the dispatch/combine hot path.

    Re-running ``setup()`` with the same configuration is a no-op (the
    snapshot compares equal).  Different configuration requires
    ``reset_runtime()`` first.
    """

    mode: LaunchMode
    launch_mode: int  # raw int passed as FFI attr
    ep_size: int
    is_internode: bool
    num_sms: int
    source_meta_bytes: int  # cached; 0 for non-internode runs


_frozen: Optional[FrozenRuntimeState] = None
_default_num_sms: int = 64


def _require_frozen() -> FrozenRuntimeState:
    """Return the frozen state or raise a helpful error if ``setup()`` was skipped."""
    if _frozen is None:
        raise RuntimeError(
            "primus_turbo.jax.lax.moe.setup() must be called exactly once before "
            "the first moe_dispatch / moe_combine.  Typical invocation:\n"
            "  from primus_turbo.jax.lax.moe import setup\n"
            "  setup(mesh=mesh, hidden_bytes=emb_dim * 2, num_sms=32)\n"
            "See setup() docstring for details."
        )
    return _frozen


# ---------------------------------------------------------------------------
#  Public re-exports
# ---------------------------------------------------------------------------

# Direct re-export — these are part of the public surface even though they
# live in ``deep_ep.runtime`` (lower layer).
set_ep_group = deep_ep_runtime.set_ep_group


def get_ep_size() -> int:
    """Return the EP-group size as frozen by :func:`setup`.

    Raises ``RuntimeError`` when ``setup()`` has not been called yet.
    Use this for sharding code that needs to know the EP communication
    domain size at module / class construction time.
    """
    return _require_frozen().ep_size


def is_internode() -> bool:
    """Return True iff the EP group spans multiple nodes (``ep_size > 8``).

    Reads from the frozen state captured by :func:`setup`; raises
    ``RuntimeError`` if ``setup()`` has not been called.  Use this when
    handle-interpretation logic (e.g. extracting ``num_recv`` from the
    dispatch handle) needs to branch on internode vs intranode without
    re-querying the lower-level ``deep_ep.runtime`` module.
    """
    return _require_frozen().is_internode


def reset_runtime() -> None:
    """Reset all DeepEP state — frozen snapshot + EP group + C++ buffer.

    Required between distinct ``setup()`` configurations (e.g. tests that
    exercise multiple EP sizes in the same Python process).  After
    ``reset_runtime()``, ``setup()`` may be called again.

    **Caveat — INPROC mode is only partially resettable.**  The C++
    in-process buffer pool is allocated lazily on the first
    :func:`moe_dispatch` and persists for the lifetime of the process;
    it does *not* expose a destroy API.  ``reset_runtime()`` therefore
    only clears the Python-side state (frozen snapshot, locked mode,
    EP-group pin, env-var override) under INPROC — the underlying pool
    keeps whatever shape / dtype hints the first dispatch primed it
    with.  Two practical consequences:

      * Within a single Python process, every INPROC dispatch must be
        compatible with the pool primed by the first one.  Calling
        ``setup()`` again with different ep_size / num_sms is silently
        ineffective for the pool itself.
      * Tearing the runtime down between consecutive INPROC dispatches
        (``dispatch → reset_runtime() → setup() → dispatch``) is *not*
        a no-op even though the Python state ends up equivalent: the
        env-var pop + ``_locked_mode = None`` + re-lock sequence has
        been observed to deadlock the second dispatch when mixed with
        the JAX FFI cache (e.g. bf16 followed by fp8).  Use a single
        module-scope ``setup()`` for INPROC test suites with identical
        configuration; reserve per-test ``reset_runtime()`` for
        PER_PROCESS runs (where the per-process buffer *is* destroyed
        here) or for API-level validation tests.

    Under PER_PROCESS mode the C++ per-process buffer is destroyed via
    ``destroy_per_process_buffer()``, so ``reset_runtime()`` there is
    a full reset.
    """
    global _frozen, _default_num_sms
    _frozen = None
    _default_num_sms = 64
    # Clear PRIMUS_TURBO_JAX_DEEPEP_MODE so auto_detect_mode rediscovers from scratch
    os.environ.pop(deep_ep_runtime._MODE_ENV_VAR, None)
    deep_ep_runtime.reset_runtime()


# ---------------------------------------------------------------------------
#  setup()
# ---------------------------------------------------------------------------


def setup(
    *,
    hidden_bytes: Optional[int] = None,
    mesh: Optional["jax.sharding.Mesh"] = None,
    ep_ranks: Optional[Sequence[int]] = None,
    ep_axis_name: str = "expert",
    num_sms: Optional[int] = None,
    mode: Optional[str] = None,
) -> None:
    """One-call DeepEP bootstrap; required exactly once before dispatch/combine.

    Handles all three runtime modes uniformly:

    ============================= =================== ====== ============= =========
    Mode                          When                EP-grp hidden_bytes  barrier?
    ============================= =================== ====== ============= =========
    INPROC                        1 proc, ≥2 GPUs     no     ignored       no
    PER_PROCESS intranode         N procs, 1 GPU/proc yes    required      no
                                  (all on 1 node, N≤8)
    PER_PROCESS internode         N procs, 1 GPU/proc yes    required      yes
                                  (multi-node, N>8)
    ============================= =================== ====== ============= =========

    Mode is auto-detected from ``jax.local_device_count()`` /
    ``jax.process_count()`` unless ``mode=`` is passed explicitly.

    The EP communication group is derived from:

    1. ``mesh`` on the ``ep_axis_name`` axis, when ``mesh`` is given;
    2. ``ep_ranks`` (explicit ordered ``jax.process_index()`` list)
       when ``ep_ranks`` is given;
    3. ``jax.process_count()`` (legacy ``EP == world``) otherwise.

    For PER_PROCESS internode, an internal ``sync_global_devices`` barrier
    is issued before any C++ buffer allocation to prevent the rocSHMEM
    endpoint handshake race observed at ~50 % rate on multi-host runs
    without it.  Callers do not need to wrap ``setup()`` in their own
    barrier.

    After ``setup()`` returns, the runtime state is *frozen* into a
    :class:`FrozenRuntimeState`.  Subsequent calls with the same
    configuration are no-ops; calls with different configuration raise
    ``RuntimeError`` (call :func:`reset_runtime` first to reconfigure).

    Args:
      hidden_bytes: ``emb_dim * max(dtype_itemsize, 2)``.  Sizes the
        per-process NVL (and, for internode, RDMA) buffer.  **Required
        under PER_PROCESS mode**; ignored under INPROC mode (the
        in-process buffer pool sizes itself lazily on first dispatch,
        matching the pre-strict-freeze behavior where INPROC callers
        never had to pass this value).
      mesh: a ``jax.sharding.Mesh``.  The EP group is derived from the
        ``ep_axis_name`` axis.  Mutually exclusive with ``ep_ranks``.
      ep_ranks: explicit ordered ``jax.process_index()`` list for the EP
        group.  Use when there is no Mesh.  Mutually exclusive with
        ``mesh``.
      ep_axis_name: name of the EP-parallel axis on ``mesh``.  Default
        ``"expert"`` matches MaxText conventions; other frameworks may
        use ``"ep"`` or ``"expert_parallel"``.
      num_sms: override the default ``num_sms`` used by the dispatch /
        combine ``Config``.  ``None`` keeps the library default (64).
        Production multi-host runs typically pass 32 to keep buffer
        sizes from squeezing the XLA prefill memory pool.
      mode: ``"inproc"`` / ``"per_process"`` / ``None`` (auto-detect).

    Raises:
      ValueError: on contradictory configuration (e.g. both ``mesh``
        and ``ep_ranks``, ``mode='inproc'`` for a multi-process
        launch, or ``hidden_bytes=None`` under PER_PROCESS mode).
      RuntimeError: when called twice with different configuration
        without an intervening :func:`reset_runtime`.

    Example::

        from primus_turbo.jax.lax.moe import setup, moe_dispatch, moe_combine

        # PER_PROCESS (1-GPU-per-process, single- or multi-node)
        setup(mesh=mesh, hidden_bytes=emb_dim * 2, num_sms=32)

        # INPROC (single-process, multi-GPU)
        setup()

        # Now dispatch / combine work in all three modes uniformly:
        recv_x, recv_idx, recv_w, handle = moe_dispatch(x, topk_idx, topk_w, num_experts)
        out = moe_combine(expert_out, handle)
    """
    global _frozen, _default_num_sms

    # ----- Early validation: catch contradictory config immediately -----
    if mesh is not None and ep_ranks is not None:
        raise ValueError("Pass either `mesh` or `ep_ranks`, not both.")
    if mode is not None and mode not in ("inproc", "per_process"):
        raise ValueError(f"`mode` must be 'inproc', 'per_process', or None; got {mode!r}")
    nproc = jax.process_count()
    nlocal = jax.local_device_count()
    if mode == "inproc" and nproc > 1:
        raise ValueError(
            f"mode='inproc' is invalid for multi-process launches "
            f"(jax.process_count()={nproc}); use 'per_process' or omit `mode`."
        )
    if mode == "per_process" and nproc == 1 and nlocal > 1:
        raise ValueError(
            f"mode='per_process' is invalid for single-process launches "
            f"(jax.process_count()=1, jax.local_device_count()={nlocal}); "
            f"use 'inproc' or omit `mode`."
        )

    # ----- Mode resolution: env var > explicit mode > heuristic -----
    if mode is not None:
        os.environ[deep_ep_runtime._MODE_ENV_VAR] = mode
    deep_ep_runtime.auto_detect_mode()
    locked_mode = deep_ep_runtime.get_mode(lock=True)

    # ----- EP-group pin (only meaningful under PER_PROCESS) -----
    if locked_mode is MODE_PER_PROCESS:
        if mesh is not None:
            deep_ep_runtime.pin_ep_group_from_jax_mesh(mesh, axis_name=ep_axis_name)
        elif ep_ranks is not None:
            deep_ep_runtime.set_ep_group(ep_ranks)
        # else: legacy "EP == jax.process_count()" path; warning emitted
        # by _ep_group_size_or_world if the world is multi-process.
    elif mesh is not None or ep_ranks is not None:
        warnings.warn(
            "EP-group configuration ignored under INPROC mode "
            "(rank derives from device_id, num_ranks from jax.local_device_count).",
            stacklevel=2,
        )

    # ----- num_sms override -----
    if num_sms is not None:
        _default_num_sms = num_sms

    # ----- Internode divisibility check (before any C++ alloc) -----
    ep_size = deep_ep_runtime.get_ep_size(lock=True)
    # Internode only makes sense under PER_PROCESS: the C++ INPROC buffer
    # pool hard-rejects num_ranks > NUM_MAX_NVL_PEERS and internode FFI
    # handlers require launch_mode==PER_PROCESS. Reject the combination
    # here so the failure is a clear ValueError from setup() instead of an
    # opaque C++ assertion deeper in the stack.
    if locked_mode is not MODE_PER_PROCESS and ep_size > NUM_MAX_NVL_PEERS:
        raise ValueError(
            f"INPROC mode cannot use ep_size={ep_size} > {NUM_MAX_NVL_PEERS}; "
            f"internode / RDMA requires PER_PROCESS mode. Set "
            f"PRIMUS_TURBO_JAX_DEEPEP_MODE=per_process or call "
            f"setup(mode='per_process')."
        )
    is_internode = locked_mode is MODE_PER_PROCESS and ep_size > NUM_MAX_NVL_PEERS
    if is_internode and ep_size % NUM_MAX_NVL_PEERS != 0:
        raise ValueError(
            f"Internode mode requires ep_size %% {NUM_MAX_NVL_PEERS} == 0; " f"got ep_size={ep_size}."
        )

    # ----- Eager warmup -----
    # INPROC: skip — the in-process buffer pool sizes itself lazily on
    #   the first dispatch, exactly matching pre-strict-freeze behavior
    #   where INPROC callers never had to pass hidden_bytes.
    # PER_PROCESS: hidden_bytes is mandatory; trigger _bootstrap_per_process
    #   which internally issues the cross-process barrier before C++
    #   allocation.
    if locked_mode is MODE_PER_PROCESS:
        if hidden_bytes is None:
            raise ValueError(
                "hidden_bytes is required under PER_PROCESS mode. "
                "Pass setup(hidden_bytes=emb_dim * max(dtype_itemsize, 2)); "
                "the value sizes the per-process NVL (and, for internode, "
                "RDMA) buffer."
            )
        config = get_dispatch_config()
        deep_ep_runtime.ensure_deepep_runtime(hidden_bytes=hidden_bytes, config=config)

    # ----- Freeze for subsequent dispatch/combine -----
    new_frozen = FrozenRuntimeState(
        mode=locked_mode,
        launch_mode=int(locked_mode),
        ep_size=ep_size,
        is_internode=is_internode,
        num_sms=_default_num_sms,
        source_meta_bytes=deep_ep_runtime.get_source_meta_bytes() if is_internode else 0,
    )
    if _frozen is not None and _frozen != new_frozen:
        raise RuntimeError(
            "primus_turbo.jax.lax.moe.setup() called with different configuration "
            "than the previous call:\n"
            f"  previous: {_frozen}\n"
            f"  new:      {new_frozen}\n"
            "Call primus_turbo.jax.lax.moe.reset_runtime() between distinct "
            "configurations."
        )
    _frozen = new_frozen


# ---------------------------------------------------------------------------
#  Config helpers (module-internal; not in __all__)
# ---------------------------------------------------------------------------


def get_dispatch_config() -> Config:
    """Return the recommended dispatch ``Config`` for the current ep_size + num_sms.

    Module-internal: callers should not normally need this; the
    ``moe_dispatch`` function uses it as the default when ``config=None``.
    """
    num_ranks = deep_ep_runtime.get_ep_size()
    config_map = {
        2: Config(_default_num_sms, 24, 256, 6, 128),
        4: Config(_default_num_sms, 6, 256, 6, 128),
        8: Config(_default_num_sms, 6, 256, 6, 128),
        16: Config(_default_num_sms, 36, 288, 20, 128),
        24: Config(_default_num_sms, 8, 288, 32, 128),
        32: Config(_default_num_sms, 32, 288, 32, 128),
        64: Config(_default_num_sms, 20, 288, 28, 128),
        128: Config(_default_num_sms, 20, 560, 32, 128),
        144: Config(_default_num_sms, 32, 720, 12, 128),
        160: Config(_default_num_sms, 28, 720, 12, 128),
    }
    assert num_ranks in config_map, f"Unsupported number of EP ranks: {num_ranks}"
    return config_map[num_ranks]


def get_combine_config() -> Config:
    """Return the recommended combine ``Config`` for the current ep_size + num_sms.

    Module-internal: callers should not normally need this; the
    ``moe_combine`` function uses it as the default when ``config=None``.
    """
    num_ranks = deep_ep_runtime.get_ep_size()
    config_map = {
        2: Config(_default_num_sms, 10, 256, 6, 128),
        4: Config(_default_num_sms, 9, 256, 6, 128),
        8: Config(_default_num_sms, 4, 256, 6, 128),
        16: Config(_default_num_sms, 4, 288, 12, 128),
        24: Config(_default_num_sms, 1, 288, 8, 128),
        32: Config(_default_num_sms, 1, 288, 8, 128),
        64: Config(_default_num_sms, 1, 288, 20, 128),
        128: Config(_default_num_sms, 1, 560, 12, 128),
        144: Config(_default_num_sms, 2, 720, 8, 128),
        160: Config(_default_num_sms, 2, 720, 8, 128),
    }
    assert num_ranks in config_map, f"Unsupported number of EP ranks: {num_ranks}"
    return config_map[num_ranks]


# ---------------------------------------------------------------------------
#  moe_dispatch
# ---------------------------------------------------------------------------


def moe_dispatch(
    x: Union[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]],
    topk_idx: jnp.ndarray,
    topk_weights: jnp.ndarray,
    num_experts: int,
    expert_alignment: int = 1,
    config: Optional[Config] = None,
) -> Tuple[Union[Tuple[jnp.ndarray, jnp.ndarray], jnp.ndarray], jnp.ndarray, jnp.ndarray, Tuple]:
    """
    Dispatch tokens to their selected experts in a Mixture of Experts (MoE) model.

    In MoE models, tokens dynamically select their top-k experts based on routing scores. This function
    executes the all-to-all communication required to dispatch tokens from all ranks to the specific ranks
    hosting their chosen experts. It leverages both intra-node communication (e.g., NVLink) and inter-node
    communication (e.g., RDMA) to facilitate efficient expert parallelism across multiple GPUs and nodes.

    Key functionalities of the dispatch process include:
    - Routing tokens based on top-k expert assignments.
    - Performing cross-rank communication to deliver tokens to their designated expert locations.
    - Computing and caching communication layouts to optimize performance.
    - Supporting both standard precision (bfloat16) and low precision (float8) data types.

    Requires :func:`setup` to have been called exactly once before the first invocation.  The runtime
    mode (INPROC / PER_PROCESS intranode / PER_PROCESS internode) and EP-group size are read from the
    frozen state captured at ``setup()`` time and used to select the correct FFI handler.  Calling this
    without ``setup()`` raises :class:`RuntimeError`.

    Args:
        x: A `jnp.ndarray` or a tuple of `jnp.ndarray`s.
            - If a single array, it must have a shape of `[num_tokens, hidden]` and a dtype of `jnp.bfloat16`.
            - If a tuple, the first element must be `[num_tokens, hidden]` with dtype `jnp.float8_e4m3fn`,
              and the second element must be `[num_tokens, hidden // 128]` (where hidden is divisible by 128)
              with dtype `jnp.float32`.
        topk_idx: A `jnp.ndarray` of shape `[num_tokens, num_topk]` and dtype `jnp.int32`, representing
            the expert indices selected by each token. A value of `-1` indicates no selection.
        topk_weights: A `jnp.ndarray` of shape `[num_tokens, num_topk]` and dtype `jnp.float32`,
            representing the routing weights for each token's selected experts.
        num_experts: The total number of experts across all ranks.
        expert_alignment: An integer specifying the alignment for the number of tokens received by each local expert.
        config: An optional performance tuning configuration. If `None`, the default configuration from
            `get_dispatch_config()` is used.

    Returns:
        recv_x: The received tokens, matching the type and structure of the input `x`, but with the
            token dimension updated to reflect the total number of received tokens.
        recv_topk_idx: The received expert indices with shape `[num_recv_tokens, num_topk]`, or `None`
            if `handle` was provided.
        recv_topk_weights: The received expert weights with shape `[num_recv_tokens, num_topk]`, or `None`
            if `handle` was provided.
        handle: A communication handle containing the computed layout information (e.g., `rank_prefix_matrix`,
            `channel_prefix_matrix`, `recv_channel_prefix_matrix`, `recv_src_idx`, `is_token_in_rank`,
            `send_head`). This handle can be passed to subsequent calls to bypass layout recomputation,
            and must be passed to :func:`moe_combine` to undo this dispatch.
    """
    return _moe_dispatch(x, topk_idx, topk_weights, num_experts, expert_alignment, config)


@partial(jax.custom_vjp, nondiff_argnums=(3, 4, 5))
def _moe_dispatch(
    x: Union[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]],
    topk_idx: jnp.ndarray,
    topk_weights: jnp.ndarray,
    num_experts: int,
    expert_alignment: int = 1,
    config: Optional[Config] = None,
) -> Tuple[Union[Tuple[jnp.ndarray, jnp.ndarray], jnp.ndarray], jnp.ndarray, jnp.ndarray, Tuple]:
    return _moe_dispatch_impl(
        x,
        topk_idx=topk_idx,
        topk_weights=topk_weights,
        expert_alignment=expert_alignment,
        num_experts=num_experts,
        config=config,
    )


def _moe_dispatch_impl(
    x: Union[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]],
    handle: Optional[Tuple] = None,
    topk_idx: Optional[jnp.ndarray] = None,
    topk_weights: Optional[jnp.ndarray] = None,
    expert_alignment: int = 1,
    num_experts: Optional[int] = None,
    config: Optional[Config] = None,
) -> Tuple[
    Union[Tuple[jnp.ndarray, jnp.ndarray], jnp.ndarray],
    Optional[jnp.ndarray],
    Optional[jnp.ndarray],
    Optional[Tuple],
]:
    if isinstance(x, tuple):
        x, x_scales = x
    else:
        x_scales = jnp.array([], dtype=jnp.float32)

    assert x.ndim == 2, "x must be a 2D array, but got {}".format(x.ndim)
    num_tokens, _ = x.shape

    # Frozen state: all runtime invariants come from setup().  Forgetting
    # setup() raises here with a clear message rather than dying inside
    # C++ with "per_process buffer not initialized" — and crucially,
    # before get_dispatch_config() (which would otherwise assert on
    # ep_size=1 with a less actionable error).
    state = _require_frozen()
    ep_size = state.ep_size
    launch_mode = state.launch_mode
    num_worst_tokens = num_tokens * ep_size
    config = get_dispatch_config() if config is None else config

    if state.is_internode:
        return _moe_dispatch_impl_internode(
            x,
            x_scales,
            handle,
            topk_idx,
            topk_weights,
            expert_alignment,
            num_experts,
            config,
            ep_size,
            launch_mode,
            num_worst_tokens,
            state,
        )

    if handle is not None:
        assert topk_idx is None and topk_weights is None
        (
            rank_prefix_matrix,
            channel_prefix_matrix,
            recv_channel_prefix_matrix,
            recv_src_idx,
            is_token_in_rank,
            send_head,
        ) = handle
        num_recv_tokens = recv_src_idx.shape[0]
        recv_x, recv_x_scales, _, _, _ = moe_cached_dispatch_p.bind(
            x,
            x_scales,
            is_token_in_rank,
            rank_prefix_matrix,
            channel_prefix_matrix,
            num_recv_tokens=num_recv_tokens,
            expert_alignment=expert_alignment,
            num_worst_tokens=num_worst_tokens,
            ep_size=ep_size,
            launch_mode=launch_mode,
            num_sms=config.num_sms,
            num_max_nvl_chunked_send_tokens=config.num_max_nvl_chunked_send_tokens,
            num_max_nvl_chunked_recv_tokens=config.num_max_nvl_chunked_recv_tokens,
            num_max_rdma_chunked_send_tokens=config.num_max_rdma_chunked_send_tokens,
            num_max_rdma_chunked_recv_tokens=config.num_max_rdma_chunked_recv_tokens,
        )
        recv = (recv_x, recv_x_scales) if x_scales.size > 0 else recv_x
        return recv, None, None, None
    else:
        assert topk_idx is not None and topk_weights is not None
        assert num_experts is not None

        (
            recv_x,
            recv_x_scales,
            recv_topk_idx,
            recv_topk_weights,
            is_token_in_rank,
            num_tokens_per_rank,
            num_tokens_per_expert,
            rank_prefix_matrix,
            channel_prefix_matrix,
            recv_channel_prefix_matrix,
            recv_src_idx,
            send_head,
        ) = moe_dispatch_p.bind(
            x,
            x_scales,
            topk_idx,
            topk_weights,
            num_experts=num_experts,
            expert_alignment=expert_alignment,
            num_worst_tokens=num_worst_tokens,
            ep_size=ep_size,
            launch_mode=launch_mode,
            num_sms=config.num_sms,
            num_max_nvl_chunked_send_tokens=config.num_max_nvl_chunked_send_tokens,
            num_max_nvl_chunked_recv_tokens=config.num_max_nvl_chunked_recv_tokens,
            num_max_rdma_chunked_send_tokens=config.num_max_rdma_chunked_send_tokens,
            num_max_rdma_chunked_recv_tokens=config.num_max_rdma_chunked_recv_tokens,
        )

        handle = (
            rank_prefix_matrix,
            channel_prefix_matrix,
            recv_channel_prefix_matrix,
            recv_src_idx,
            is_token_in_rank,
            send_head,
        )
        return (
            (recv_x, recv_x_scales) if x_scales.size > 0 else recv_x,
            recv_topk_idx,
            recv_topk_weights,
            handle,
        )


def _moe_dispatch_impl_internode(
    x,
    x_scales,
    handle,
    topk_idx,
    topk_weights,
    expert_alignment,
    num_experts,
    config,
    ep_size,
    launch_mode,
    num_worst_tokens,
    state: FrozenRuntimeState,
):
    """Internode dispatch path (ep_size > NUM_MAX_NVL_PEERS)."""
    source_meta_bytes = state.source_meta_bytes

    if handle is not None:
        assert topk_idx is None and topk_weights is None
        (
            is_token_in_rank,
            rdma_channel_prefix_matrix,
            gbl_channel_prefix_matrix,
            _recv_rdma_channel_prefix_matrix,
            recv_rdma_rank_prefix_sum,
            _recv_gbl_channel_prefix_matrix,
            recv_gbl_rank_prefix_sum,
            recv_src_meta,
            send_rdma_head,
            send_nvl_head,
        ) = handle
        num_recv_tokens = recv_src_meta.shape[0]
        num_rdma_recv_tokens = send_nvl_head.shape[0]
        recv_x, recv_x_scales, _, _, _, _ = moe_internode_cached_dispatch_p.bind(
            x,
            x_scales,
            is_token_in_rank,
            rdma_channel_prefix_matrix,
            recv_rdma_rank_prefix_sum,
            gbl_channel_prefix_matrix,
            recv_gbl_rank_prefix_sum,
            num_recv_tokens=num_recv_tokens,
            num_rdma_recv_tokens=num_rdma_recv_tokens,
            expert_alignment=expert_alignment,
            num_worst_tokens=num_worst_tokens,
            ep_size=ep_size,
            launch_mode=launch_mode,
            num_sms=config.num_sms,
            num_max_nvl_chunked_send_tokens=config.num_max_nvl_chunked_send_tokens,
            num_max_nvl_chunked_recv_tokens=config.num_max_nvl_chunked_recv_tokens,
            num_max_rdma_chunked_send_tokens=config.num_max_rdma_chunked_send_tokens,
            num_max_rdma_chunked_recv_tokens=config.num_max_rdma_chunked_recv_tokens,
        )
        recv = (recv_x, recv_x_scales) if x_scales.size > 0 else recv_x
        return recv, None, None, None
    else:
        assert topk_idx is not None and topk_weights is not None
        assert num_experts is not None

        (
            recv_x,
            recv_x_scales,
            recv_topk_idx,
            recv_topk_weights,
            is_token_in_rank,
            num_tokens_per_rank,
            num_tokens_per_rdma_rank,
            num_tokens_per_expert,
            rdma_channel_prefix_matrix,
            recv_rdma_rank_prefix_sum,
            gbl_channel_prefix_matrix,
            recv_gbl_rank_prefix_sum,
            recv_src_meta,
            recv_rdma_channel_prefix_matrix,
            recv_gbl_channel_prefix_matrix,
            send_rdma_head,
            send_nvl_head,
        ) = moe_internode_dispatch_p.bind(
            x,
            x_scales,
            topk_idx,
            topk_weights,
            num_experts=num_experts,
            expert_alignment=expert_alignment,
            num_worst_tokens=num_worst_tokens,
            ep_size=ep_size,
            launch_mode=launch_mode,
            num_sms=config.num_sms,
            num_max_nvl_chunked_send_tokens=config.num_max_nvl_chunked_send_tokens,
            num_max_nvl_chunked_recv_tokens=config.num_max_nvl_chunked_recv_tokens,
            num_max_rdma_chunked_send_tokens=config.num_max_rdma_chunked_send_tokens,
            num_max_rdma_chunked_recv_tokens=config.num_max_rdma_chunked_recv_tokens,
            source_meta_bytes=source_meta_bytes,
        )

        handle = (
            is_token_in_rank,
            rdma_channel_prefix_matrix,
            gbl_channel_prefix_matrix,
            recv_rdma_channel_prefix_matrix,
            recv_rdma_rank_prefix_sum,
            recv_gbl_channel_prefix_matrix,
            recv_gbl_rank_prefix_sum,
            recv_src_meta,
            send_rdma_head,
            send_nvl_head,
        )
        return (
            (recv_x, recv_x_scales) if x_scales.size > 0 else recv_x,
            recv_topk_idx,
            recv_topk_weights,
            handle,
        )


def _moe_dispatch_fwd(
    x: Union[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]],
    topk_idx: jnp.ndarray,
    topk_weights: jnp.ndarray,
    num_experts: int,
    expert_alignment: int = 1,
    config: Optional[Config] = None,
):
    recv_x, recv_topk_idx, recv_topk_weights, handle = _moe_dispatch_impl(
        x,
        topk_idx=topk_idx,
        topk_weights=topk_weights,
        num_experts=num_experts,
        expert_alignment=expert_alignment,
        config=config,
    )

    ctx = (handle,)
    return (recv_x, recv_topk_idx, recv_topk_weights, handle), ctx


def _moe_dispatch_bwd(num_experts, expert_alignment, config, ctx, grad_output):
    (handle,) = ctx
    grad_x, _, grad_topk_weights, _ = grad_output

    if isinstance(grad_x, tuple):
        grad_x_main, _ = grad_x
        grad_x_main, grad_topk_weights = _moe_combine_impl(
            grad_x_main, handle, topk_weights=grad_topk_weights, config=config
        )
        grad_x = (grad_x_main, None)
    else:
        grad_x, grad_topk_weights = _moe_combine_impl(
            grad_x, handle, topk_weights=grad_topk_weights, config=config
        )

    return grad_x, None, grad_topk_weights


_moe_dispatch.defvjp(_moe_dispatch_fwd, _moe_dispatch_bwd)


# ---------------------------------------------------------------------------
#  moe_combine
# ---------------------------------------------------------------------------


def moe_combine(
    x: jnp.ndarray,
    handle: Tuple,
    config: Optional[Config] = None,
) -> jnp.ndarray:
    """
    Combine (reduce) tokens from different experts back to their original ranks in a Mixture of Experts (MoE) model.

    After tokens are processed by their selected experts (via moe_dispatch), this function performs the reverse
    all-to-all communication to gather and aggregate results back to the original token locations. The aggregation
    is performed via addition across all ranks that processed each token. Supports both intranode communication
    via NVLink and internode communication via RDMA.

    This is the complement operation to moe_dispatch and must be called with the handle returned from moe_dispatch
    to ensure correct routing back to the original token positions.

    Requires :func:`setup` to have been called exactly once before the first invocation; the runtime mode and
    EP-group size are read from the frozen state.  Calling this without ``setup()`` raises :class:`RuntimeError`.

    Arguments:
        x: `[num_recv_tokens, hidden]` with `jnp.bfloat16` or `jnp.float8_e4m3fn`, the expert output tokens
            to send back for reducing to their original ranks.
        handle: a required communication handle obtained from the moe_dispatch function, containing layout
            information (rank_prefix_matrix, channel_prefix_matrix, recv_channel_prefix_matrix, recv_src_idx,
            is_token_in_rank, send_head) needed for the reverse communication.
        config: the performance tuning config. If None, will use the default config from get_combine_config().

    Returns:
        combined_x: the reduced tokens from all expert ranks, gathered back to original token positions with
            shape `[num_tokens, hidden]`, aggregated via addition across all ranks.
    """
    return _moe_combine(x, handle, config)


@partial(jax.custom_vjp, nondiff_argnums=(2,))
def _moe_combine(
    x: jnp.ndarray,
    handle: Tuple,
    config: Optional[Config] = None,
) -> jnp.ndarray:
    combine_x, _ = _moe_combine_impl(x, handle, config=config)
    return combine_x


def _moe_combine_impl(
    x: jnp.ndarray,
    handle: Tuple,
    topk_weights: Optional[jnp.ndarray] = None,
    bias: Union[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]] = None,
    config: Optional[Config] = None,
) -> Tuple[jnp.ndarray]:
    state = _require_frozen()
    ep_size = state.ep_size
    launch_mode = state.launch_mode
    config = get_combine_config() if config is None else config

    if state.is_internode:
        return _moe_combine_impl_internode(
            x,
            handle,
            topk_weights,
            config,
            ep_size,
            launch_mode,
        )

    # unpack bias
    bias_0, bias_1 = None, None
    if isinstance(bias, jnp.ndarray):
        bias_0 = bias
    elif isinstance(bias, tuple):
        assert len(bias) == 2
        bias_0, bias_1 = bias

    if topk_weights is None:
        topk_weights = jnp.array([], dtype=jnp.float32)

    if bias_0 is None:
        bias_0 = jnp.array([], dtype=x.dtype)

    if bias_1 is None:
        bias_1 = jnp.array([], dtype=x.dtype)

    rank_prefix_matrix, _, channel_prefix_matrix, src_idx, is_recv_token_in_rank, send_head = handle

    combined_x, combined_topk_weights = moe_combine_p.bind(
        x,
        topk_weights,
        bias_0,
        bias_1,
        src_idx,
        rank_prefix_matrix,
        channel_prefix_matrix,
        send_head,
        ep_size=ep_size,
        launch_mode=launch_mode,
        num_sms=config.num_sms,
        num_max_nvl_chunked_send_tokens=config.num_max_nvl_chunked_send_tokens,
        num_max_nvl_chunked_recv_tokens=config.num_max_nvl_chunked_recv_tokens,
        num_max_rdma_chunked_send_tokens=config.num_max_rdma_chunked_send_tokens,
        num_max_rdma_chunked_recv_tokens=config.num_max_rdma_chunked_recv_tokens,
    )
    return combined_x, combined_topk_weights


def _moe_combine_impl_internode(
    x,
    handle,
    topk_weights,
    config,
    ep_size,
    launch_mode,
):
    """Internode combine path (ep_size > NUM_MAX_NVL_PEERS)."""
    (
        is_token_in_rank,
        _send_rdma_channel_prefix_matrix,
        _send_gbl_channel_prefix_matrix,
        rdma_channel_prefix_matrix,
        rdma_rank_prefix_sum,
        gbl_channel_prefix_matrix,
        gbl_rank_prefix_sum,
        recv_src_meta,
        send_rdma_head,
        send_nvl_head,
    ) = handle

    if topk_weights is None:
        topk_weights = jnp.array([], dtype=jnp.float32)

    combined_x, combined_topk_weights = moe_internode_combine_p.bind(
        x,
        topk_weights,
        recv_src_meta,
        is_token_in_rank,
        rdma_channel_prefix_matrix,
        rdma_rank_prefix_sum,
        gbl_channel_prefix_matrix,
        gbl_rank_prefix_sum,
        send_rdma_head,
        send_nvl_head,
        ep_size=ep_size,
        launch_mode=launch_mode,
        num_sms=config.num_sms,
        num_max_nvl_chunked_send_tokens=config.num_max_nvl_chunked_send_tokens,
        num_max_nvl_chunked_recv_tokens=config.num_max_nvl_chunked_recv_tokens,
        num_max_rdma_chunked_send_tokens=config.num_max_rdma_chunked_send_tokens,
        num_max_rdma_chunked_recv_tokens=config.num_max_rdma_chunked_recv_tokens,
    )
    return combined_x, combined_topk_weights


def _moe_combine_fwd(
    x: jnp.ndarray,
    handle: Tuple,
    config: Optional[Config] = None,
) -> jnp.ndarray:
    combine_x, _ = _moe_combine_impl(x, handle, config=config)
    ctx = (handle,)
    return combine_x, ctx


def _moe_combine_bwd(config, ctx, grad_output):
    (handle,) = ctx

    recv_grad_x, _, _, _ = _moe_dispatch_impl(grad_output, handle=handle, config=config)
    handle_grad = jax.tree_util.tree_map(jnp.zeros_like, handle)
    return recv_grad_x, handle_grad


_moe_combine.defvjp(_moe_combine_fwd, _moe_combine_bwd)
