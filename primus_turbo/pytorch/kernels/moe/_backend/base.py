###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""``EPBackend`` Protocol + ``_DeepEPLikeBackend`` shared base for DeepEP-style backends."""

import functools
import os
import socket
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    Union,
    runtime_checkable,
)

import torch
import torch.distributed as dist

from primus_turbo.common.logger import logger

from ._config import EPBufferConfig

_TOPOLOGY_CACHE: Dict[int, Tuple[int, int]] = {}


def call_once(method):
    flag_attr = "_call_once__" + method.__qualname__.replace(".", "_")

    @functools.wraps(method)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        if getattr(self, flag_attr, False):
            return None
        result = method(self, *args, **kwargs)
        setattr(self, flag_attr, True)
        return result

    return wrapper  # type: ignore[return-value]


def detect_group_topology(group: dist.ProcessGroup) -> Tuple[int, int]:
    """Return ``(node_idx, num_nodes)`` for ``group``, cached per ``id(group)``."""
    cache_key = id(group)
    cached = _TOPOLOGY_CACHE.get(cache_key)
    if cached is not None:
        return cached

    node_token = (
        os.environ.get("NODE_RANK")
        or os.environ.get("GROUP_RANK")
        or os.environ.get("SLURM_NODEID")
        or socket.gethostname()
    )

    world = dist.get_world_size(group)
    node_tokens = [None] * world
    dist.all_gather_object(node_tokens, node_token, group=group)

    token_to_idx: Dict[str, int] = {}
    for token in node_tokens:
        if token not in token_to_idx:
            token_to_idx[token] = len(token_to_idx)

    result = (token_to_idx[node_token], len(token_to_idx))
    _TOPOLOGY_CACHE[cache_key] = result
    return result


def inplace_unique(x: torch.Tensor, num_slots: int) -> None:
    assert x.dim() == 2
    mask = x < 0
    x_padded = x.masked_fill(mask, num_slots)
    bin_count = torch.zeros((x.size(0), num_slots + 1), dtype=x.dtype, device=x.device)
    bin_count.scatter_add_(1, x_padded, torch.ones_like(x_padded))
    bin_count = bin_count[:, :num_slots]
    sorted_bin_count, sorted_bin_idx = torch.sort(bin_count, dim=-1, descending=True)
    sorted_bin_idx.masked_fill_(sorted_bin_count == 0, -1)
    sorted_bin_idx = torch.sort(sorted_bin_idx, descending=True, dim=-1).values
    x[:, :].fill_(-1)
    valid_len = min(num_slots, x.size(1))
    x[:, :valid_len] = sorted_bin_idx[:, :valid_len]


@torch.no_grad()
def bench(
    fn,
    *,
    num_tests: int,
    num_warmup: int,
    l2_flush_mb: int = 256,
    sync_fn: Optional[Callable[[], None]] = None,
) -> float:
    """Time ``fn()`` per iteration with CUDA events; return mean seconds.

    Uses ``torch.cuda.Event`` (HIP events, no CUPTI) rather than the Kineto
    profiler, which races with UCCL's RDMA workers on multi-node ROCm. An
    LLC-sized buffer (``l2_flush_mb``) is zero-filled before each call to evict
    ``fn``'s data from Infinity Cache; one event pair per iteration keeps that
    flush outside the timed window. ``sync_fn`` realigns ranks before each call.
    """
    flush_buf = torch.empty(
        max(1, l2_flush_mb) * 1024 * 1024 // 4,
        dtype=torch.float32,
        device="cuda",
    )

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(num_tests)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(num_tests)]

    for _ in range(num_warmup):
        if sync_fn is not None:
            sync_fn()
        fn()
    torch.cuda.synchronize()

    for i in range(num_tests):
        # Realign ranks before the timed call (outside the timed window).
        if sync_fn is not None:
            sync_fn()
        flush_buf.zero_()
        starts[i].record()
        fn()
        ends[i].record()

    torch.cuda.synchronize()
    total_ms = sum(s.elapsed_time(e) for s, e in zip(starts, ends))
    return (total_ms / num_tests) * 1e-3


def bench_dispatch_combine(
    dispatch_fn: Callable[[], Any],
    combine_fn: Callable[[Any], Any],
    *,
    num_tests: int,
    num_warmup: int,
    sync_fn: Optional[Callable[[], None]] = None,
    l2_flush_mb: int = 256,
    flush_buf: Optional[torch.Tensor] = None,
) -> Tuple[float, float]:
    """Time a dispatch->combine pair; return ``(dispatch_s, combine_s)``.

    ``dispatch_fn()`` returns whatever ``combine_fn(recv)`` needs. They must run
    as a pair: EP backends reuse a one-sided comm buffer that a dispatch-only run
    leaves in a state the combine faults on. ``sync_fn`` realigns ranks before
    each timed iteration, outside the timed window.
    """
    if flush_buf is None:
        flush_buf = torch.empty(max(1, l2_flush_mb) * 1024 * 1024 // 4, dtype=torch.float32, device="cuda")

    for _ in range(num_warmup):
        if sync_fn:
            sync_fn()
        combine_fn(dispatch_fn())
    torch.cuda.synchronize()

    # 3 marks per iter: before dispatch | between | after combine.
    new_events = lambda: [torch.cuda.Event(enable_timing=True) for _ in range(num_tests)]
    pre, mid, post = new_events(), new_events(), new_events()
    for i in range(num_tests):
        if sync_fn:
            sync_fn()
        flush_buf.zero_()
        pre[i].record()
        recv = dispatch_fn()
        mid[i].record()
        combine_fn(recv)
        post[i].record()
    torch.cuda.synchronize()

    disp_ms = sum(a.elapsed_time(b) for a, b in zip(pre, mid)) / num_tests
    comb_ms = sum(a.elapsed_time(b) for a, b in zip(mid, post)) / num_tests
    return disp_ms * 1e-3, comb_ms * 1e-3


def sweep_configs(
    candidates: Sequence[Any],
    dispatch_fn: Callable[[Any], Any],
    combine_fn: Callable[[Any, Any], Any],
    *,
    group: dist.ProcessGroup,
    num_tests: int,
    num_warmup: int,
    on_skip: Optional[Callable[[Any, BaseException], None]] = None,
) -> Tuple[Optional[Any], Optional[Any], float, float]:
    """Benchmark each candidate's dispatch->combine pair; return the best
    dispatch and best combine candidate (chosen independently) and their times.

    ``dispatch_fn(cand) -> recv`` and ``combine_fn(cand, recv)`` are the backend
    launch closures; a candidate whose launch raises is skipped (``on_skip``).
    """

    def _sync() -> None:
        torch.cuda.synchronize()
        dist.barrier(group=group)

    flush_buf = torch.empty(256 * 1024 * 1024 // 4, dtype=torch.float32, device="cuda")
    best_d, best_d_t = None, float("inf")
    best_c, best_c_t = None, float("inf")
    for cand in candidates:
        dist.barrier(group=group)  # realign in case a prior candidate raised on some ranks
        try:
            d_t, c_t = bench_dispatch_combine(
                lambda c=cand: dispatch_fn(c),
                lambda recv, c=cand: combine_fn(c, recv),
                num_tests=num_tests,
                num_warmup=num_warmup,
                sync_fn=_sync,
                flush_buf=flush_buf,
            )
        except Exception as exc:  # noqa: BLE001 - skip bad candidate, keep sweeping
            if on_skip:
                on_skip(cand, exc)
            continue
        if 0 < d_t < best_d_t:
            best_d, best_d_t = cand, d_t
        if 0 < c_t < best_c_t:
            best_c, best_c_t = cand, c_t
    return best_d, best_c, best_d_t, best_c_t


# =========================================================================
# EPBackend Protocol
# =========================================================================


@runtime_checkable
class EPBackend(Protocol):
    """Structural interface for Expert-Parallel communication backends."""

    @staticmethod
    def is_available() -> bool:
        """Return True if this backend's dependencies are importable."""
        ...

    @staticmethod
    def supports_cuda_graph() -> bool:
        """Return True if dispatch/combine are safe inside ``torch.cuda.graph``.

        Override to ``False`` on backends that rely on synchronous HIP APIs
        or per-call device->host plumbing (they would deadlock under capture).
        """
        ...

    def is_initialized(self) -> bool:
        """Return True if the backend is initialized."""
        ...

    def setup_env(self, **overrides: Optional[str]) -> None:
        """Configure backend RDMA/network env vars before init.

        Resolution per var: explicit override > existing env > matching ``NCCL_*``
        > backend default. Backends without network state implement as no-op.
        """
        ...

    def init_buffer(
        self,
        group: dist.ProcessGroup,
        hidden_size: int,
        num_experts: int,
        num_topk: int,
        seqlen: int,
        fp8_dispatch: bool,
        config: "EPBufferConfig",
    ) -> None:
        """(Re-)create the communication buffer if needed.

        Args:
            group: EP process group.
            hidden_size: Per-token hidden dimension (element count).
            num_experts: Total number of experts.
            num_topk: Number of topk experts to dispatch.
            seqlen: Per-rank maximum dispatch tokens.
            fp8_dispatch: Whether the dispatch payload is FP8 (``x`` is a
                ``(fp8_tensor, scales)`` tuple).
            config: Backend-specific tuning config.
        """
        ...

    def dispatch(
        self,
        x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        handle: Optional[tuple] = None,
        topk_idx: Optional[torch.Tensor] = None,
        token_weights: Optional[torch.Tensor] = None,
        num_experts: Optional[int] = None,
        async_finish: bool = False,
        allocate_on_comm_stream: bool = False,
        num_worst_tokens: int = 0,
        config: Optional[Any] = None,
    ) -> Tuple[
        Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[Union[List[int], torch.Tensor]],
        Optional[tuple],
    ]:
        """Dispatch tokens to experts.

        Args:
            x: Input tensor, or ``(fp8_tensor, scales)`` tuple for FP8 path.
            handle: Cached dispatch handle; ``None`` for a primary call.
            topk_idx: ``[num_tokens, num_topk]`` expert indices (primary only).
            token_weights: ``[num_tokens, num_topk]`` expert weights.
            num_experts: Total number of experts (primary only).
            async_finish: If True, return before the kernel finishes.
            allocate_on_comm_stream: Allocate outputs on the comm stream.
            num_worst_tokens: Worst-case receive token count (for padding).
            config: Backend-specific dispatch config.

        Returns:
            ``(recv_x, recv_topk_idx, recv_topk_weights, tokens_per_expert, handle)``.
        """
        ...

    def combine(
        self,
        x: torch.Tensor,
        handle: tuple,
        topk_weights: Optional[torch.Tensor] = None,
        async_finish: bool = False,
        allocate_on_comm_stream: bool = False,
        config: Optional[Any] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Combine expert outputs back to the original token order.

        Args:
            x: Expert-side tensor to be combined.
            handle: Handle returned by the paired ``dispatch`` call.
            topk_weights: Optional per-token expert weights.
            async_finish: If True, return before the kernel finishes.
            allocate_on_comm_stream: Allocate outputs on the comm stream.
            config: Backend-specific combine config.

        Returns:
            ``(combined_x, combined_topk_weights)``.
        """
        ...

    def release_buffer(self) -> None:
        """Release the communication buffer."""
        ...


def _apply_env_with_nccl_fallback(
    mappings: Sequence[Tuple[str, str, Optional[str]]],
    backend_name: str,
) -> None:
    """Set each ``backend_env`` with the first available source:
    explicit value > existing env > ``nccl_env`` > leave unset.
    """
    for backend_env, nccl_env, explicit in mappings:
        if explicit is not None:
            os.environ[backend_env] = str(explicit)
        elif backend_env not in os.environ:
            nccl_val = os.environ.get(nccl_env)
            if nccl_val is not None:
                os.environ[backend_env] = nccl_val
        if backend_env in os.environ:
            logger.info(
                f"[{backend_name} Network Settings] {backend_env}: {os.environ[backend_env]}",
                rank=0,
            )


def _broadcast_from_rank0_int(values: Sequence[int], group: dist.ProcessGroup) -> List[int]:
    """Return rank-0's ``values`` on every rank (via ``all_gather``)."""
    t = torch.tensor(values, dtype=torch.int32, device="cuda")
    gathered = [torch.zeros_like(t) for _ in range(dist.get_world_size(group))]
    dist.all_gather(gathered, t, group=group)
    return gathered[0].tolist()


def _broadcast_from_rank0_float(value: float, group: dist.ProcessGroup) -> float:
    """Float counterpart of :func:`_broadcast_from_rank0_int`."""
    t = torch.tensor([value], dtype=torch.float64, device="cuda")
    gathered = [torch.zeros_like(t) for _ in range(dist.get_world_size(group))]
    dist.all_gather(gathered, t, group=group)
    return float(gathered[0].item())


# =========================================================================
# _DeepEPLikeBackend — shared implementation for DeepEP-compatible backends
# =========================================================================


class _DeepEPLikeBackend:
    """Shared base class for backends that follow the DeepEP Buffer protocol."""

    def __init__(self) -> None:
        self._buffer = None

    # ------------------------------------------------------------------
    # Subclass hooks
    # ------------------------------------------------------------------

    def is_initialized(self) -> bool:
        """Return True if the backend is initialized."""
        return self._buffer is not None

    @staticmethod
    def is_available() -> bool:
        """Return True if backend dependencies are importable."""
        raise NotImplementedError

    @staticmethod
    def supports_cuda_graph() -> bool:
        """DeepEP-style backends run on the caller's stream and are graph-safe."""
        return True

    @staticmethod
    def _get_module():
        """Return the backend Python module exposing ``Buffer``/``Config``/events."""
        raise NotImplementedError

    def _make_buffer_kwargs(self, group: dist.ProcessGroup) -> dict:
        """Extra kwargs forwarded to ``BufferClass(group, nvl, rdma, **kwargs)``."""
        return {}

    def setup_env(self, **overrides: Optional[str]) -> None:  # noqa: ARG002
        """No backend-specific env vars to configure by default.

        Subclasses that need RDMA / socket env wiring (UCCL, ...) override
        this to translate NCCL_* vars into their own namespace.
        """
        return

    # ------------------------------------------------------------------
    # EPBackend interface
    # ------------------------------------------------------------------

    def init_buffer(
        self,
        group: dist.ProcessGroup,
        hidden_size: int,
        num_experts: int,  # noqa: ARG002 - DeepEP sizes by hidden_bytes only.
        num_topk: int,  # noqa: ARG002 - DeepEP sizes by hidden_bytes only.
        seqlen: int,  # noqa: ARG002 - DeepEP sizes by hidden_bytes only.
        fp8_dispatch: bool,
        config: EPBufferConfig,
    ) -> None:
        # Best-effort NCCL-fallback network env; no-op without RDMA, idempotent.
        self.setup_env()

        mod = self._get_module()
        BufferClass = mod.Buffer

        BufferClass.set_num_sms(config.num_sms)

        dispatch_config = config.dispatch_config or BufferClass.get_dispatch_config(group.size())
        combine_config = config.combine_config or BufferClass.get_combine_config(group.size())

        element_size = 1 if fp8_dispatch else 2
        hidden_bytes = hidden_size * max(element_size, 2)

        num_nvl_bytes, num_rdma_bytes = 0, 0
        for cfg in (dispatch_config, combine_config):
            num_nvl_bytes = max(
                cfg.get_nvl_buffer_size_hint(hidden_bytes, group.size()),
                num_nvl_bytes,
            )
            try:
                num_rdma_bytes = max(
                    cfg.get_rdma_buffer_size_hint(hidden_bytes, group.size()),
                    num_rdma_bytes,
                )
            except (RuntimeError, AttributeError):
                pass

        buf_kwargs = self._make_buffer_kwargs(group)
        # Required: GC-based teardown leaks UCCL proxies and hangs.
        buf_kwargs.setdefault("explicitly_destroy", True)

        if (
            self._buffer is None
            # or not isinstance(self._buffer, BufferClass)
            or self._buffer.group != group
            or self._buffer.num_nvl_bytes < num_nvl_bytes
            or self._buffer.num_rdma_bytes < num_rdma_bytes
        ):
            if self._buffer is not None:
                assert isinstance(self._buffer, BufferClass)
            # Tear down old buffer to clear UCCL proxy/IPC state before realloc.
            self.release_buffer()
            self._buffer = BufferClass(group, num_nvl_bytes, num_rdma_bytes, **buf_kwargs)
            logger.info(
                f"[{self.__class__.__name__} init] world_size={group.size()} "
                f"rank={group.rank()} hidden={hidden_size} "
                f"fp8_dispatch={fp8_dispatch} num_nvl_bytes={num_nvl_bytes} "
                f"num_rdma_bytes={num_rdma_bytes} ",
                rank=0,
            )

    # ----- helpers ------------------------------------------------------

    def _get_event_classes(self):
        mod = self._get_module()
        EventOverlapClass = mod.utils.EventOverlap if hasattr(mod, "utils") else mod.EventOverlap
        EventHandleClass = mod.utils.EventHandle if hasattr(mod, "utils") else mod.EventHandle
        return EventOverlapClass, EventHandleClass

    # ----- dispatch / combine -------------------------------------------

    def dispatch(
        self,
        x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        handle: Optional[tuple] = None,
        topk_idx: Optional[torch.Tensor] = None,
        token_weights: Optional[torch.Tensor] = None,
        num_experts: Optional[int] = None,
        async_finish: bool = False,
        allocate_on_comm_stream: bool = False,
        num_worst_tokens: int = 0,
        config: Optional[Any] = None,
    ):
        assert self.is_initialized(), "Backend is not initialized"

        EventOverlapClass, EventHandleClass = self._get_event_classes()
        previous_event = EventOverlapClass(EventHandleClass()) if async_finish else None

        if handle is None:
            assert topk_idx is not None
            assert token_weights is not None
            (
                num_tokens_per_rank,
                num_tokens_per_rdma_rank,
                num_tokens_per_expert,
                is_token_in_rank,
                event,
            ) = self._buffer.get_dispatch_layout(
                topk_idx,
                num_experts,
                previous_event=previous_event,
                async_finish=async_finish,
                allocate_on_comm_stream=allocate_on_comm_stream,
            )

            (
                recv_x,
                recv_token_indices,
                recv_token_probs,
                tokens_per_expert,
                handle,
                after_event,
            ) = self._buffer.dispatch(
                x,
                topk_idx=topk_idx,
                topk_weights=token_weights,
                num_tokens_per_rank=num_tokens_per_rank,
                num_tokens_per_rdma_rank=num_tokens_per_rdma_rank,
                is_token_in_rank=is_token_in_rank,
                num_tokens_per_expert=num_tokens_per_expert,
                previous_event=event,
                async_finish=async_finish,
                allocate_on_comm_stream=allocate_on_comm_stream,
                num_worst_tokens=num_worst_tokens,
                config=config,
            )
        else:
            recv_x, recv_token_indices, recv_token_probs, tokens_per_expert, handle, after_event = (
                self._buffer.dispatch(
                    x,
                    handle=handle,
                    previous_event=previous_event,
                    async_finish=async_finish,
                    allocate_on_comm_stream=allocate_on_comm_stream,
                    config=config,
                )
            )

        if async_finish:
            after_event.current_stream_wait()

        return recv_x, recv_token_indices, recv_token_probs, tokens_per_expert, handle

    def combine(
        self,
        x: torch.Tensor,
        handle: tuple,
        topk_weights: Optional[torch.Tensor] = None,
        async_finish: bool = False,
        allocate_on_comm_stream: bool = False,
        config: Optional[Any] = None,
    ):
        assert self.is_initialized(), "Backend is not initialized"
        EventOverlapClass, EventHandleClass = self._get_event_classes()
        previous_event = EventOverlapClass(EventHandleClass()) if async_finish else None

        combined_x, combined_topk_weights, after_event = self._buffer.combine(
            x,
            handle=handle,
            topk_weights=None if topk_weights is None else topk_weights.float(),
            async_finish=async_finish,
            allocate_on_comm_stream=allocate_on_comm_stream,
            previous_event=previous_event,
            config=config,
        )

        if async_finish:
            after_event.current_stream_wait()

        return combined_x, combined_topk_weights

    def release_buffer(self) -> None:
        """Explicitly destroy the buffer (paired with ``explicitly_destroy=True``)."""
        buf = self._buffer
        if buf is None:
            return
        # Detach first so a partial destroy() can't be reused by init_buffer.
        self._buffer = None
        try:
            # Drain comm-stream work before C++ destroy()'s NVL barrier.
            torch.cuda.synchronize()
            if getattr(buf, "explicitly_destroy", False):
                buf.destroy()
        except Exception as exc:
            logger.warning(
                f"{type(self).__name__}.release_buffer: buffer.destroy() "
                f"raised {type(exc).__name__}: {exc!r}; resources may leak."
            )

    # ----- autotune ------------------------------------------------------

    @torch.no_grad()
    def tune_configs(
        self,
        group: dist.ProcessGroup,
        x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        num_experts: int,
        *,
        topk_idx: Optional[torch.Tensor] = None,
        topk_weights: Optional[torch.Tensor] = None,
        num_sms: int = 32,
        num_tests: int = 20,
        num_topk: Optional[int] = None,
        uniform_dispatch: bool = True,
    ) -> Tuple[Any, Any, float, float]:
        """Sweep ``(nvl_chunk_size, rdma_chunk_size)`` candidates and pick the
        fastest dispatch / combine ``Config`` for this backend.

        Args:
            group: The EP process group.
            x: Dispatch input (Tensor or ``(fp8_tensor, scales)`` tuple).
            num_experts: Total number of experts.
            topk_idx: ``[num_tokens, num_topk]`` expert indices. Required when
                ``uniform_dispatch=False``; only used to derive ``num_topk``
                in uniform mode.
            topk_weights: ``[num_tokens, num_topk]`` expert weights. Required
                when ``uniform_dispatch=False``; ignored in uniform mode.
            num_sms: SM count to use for tuning.
            num_tests: Timed iterations per candidate.
            num_topk: Number of experts per token (uniform mode fallback
                when ``topk_idx`` is not supplied).
            uniform_dispatch: If True, resample a near-uniform ``topk_idx`` /
                ``topk_weights`` internally for more robust tuning.

        Returns:
            ``(best_dispatch_config, best_combine_config, best_dispatch_s, best_combine_s)``
            with times identical on all ranks (rank 0's pick, broadcast).
        """
        mod = self._get_module()
        BufferClass = mod.Buffer
        BufferClass.set_num_sms(num_sms)
        ConfigClass = mod.Config
        ep_size = group.size()
        _, num_nodes = detect_group_topology(group)

        x_tensor = x if isinstance(x, torch.Tensor) else x[0]
        hidden_size = int(x_tensor.size(1))
        seqlen = int(x_tensor.size(0))
        fp8_dispatch = isinstance(x, tuple)
        # Byte count used for RDMA bandwidth accounting.
        hidden_size * max(1 if fp8_dispatch else 2, 2)

        # Resample (topk_idx, topk_weights) when uniform_dispatch is on.
        x_inp = x if isinstance(x, torch.Tensor) else x[0]
        if uniform_dispatch:
            if num_topk is None:
                if topk_idx is None:
                    raise ValueError(
                        "tune_configs(uniform_dispatch=True): need either "
                        "``num_topk`` or a shape-carrying ``topk_idx`` to "
                        "sample a uniform distribution."
                    )
                num_topk = int(topk_idx.size(1))
            if num_topk > num_experts:
                raise ValueError(
                    "tune_configs(uniform_dispatch=True): ``num_topk`` "
                    f"({num_topk}) cannot exceed ``num_experts`` ({num_experts})."
                )
            num_tokens = int(x_inp.size(0))
            device = x_inp.device
            scores = torch.randn((num_tokens, num_experts), dtype=torch.float32, device=device).abs() + 1
            topk_idx = torch.topk(scores, num_topk, dim=-1, largest=True, sorted=False)[1]
            topk_weights = torch.randn((num_tokens, num_topk), dtype=torch.float32, device=device)
        else:
            if topk_idx is None or topk_weights is None:
                raise ValueError(
                    "tune_configs(uniform_dispatch=False): ``topk_idx`` and "
                    "``topk_weights`` are both required."
                )

        # Per-(EP size) NVL/RDMA buffer sizes (chunks), from the upstream UCCL-EP sweep.
        rdma_buffer_size, nvl_buffer_size = 512, (720 if ep_size in (144, 160) else 512)
        if ep_size == 24:
            nvl_buffer_size = 540

        # Sweep ranges; intranode pins rdma_chunk since RDMA is unused.
        nvl_chunk_range = range(1, 8, 1)
        if num_nodes <= 1:
            rdma_chunk_range = (16,)
        else:
            rdma_chunk_range = range(12 if num_nodes == 2 else 8, 33, 4)

        worst_nvl_chunk = max(nvl_chunk_range)
        worst_rdma_chunk = max(
            rdma_chunk_range,
            key=lambda c: -(-rdma_buffer_size // c) * c,
        )
        worst_config = ConfigClass(
            num_sms,
            worst_nvl_chunk,
            nvl_buffer_size,
            worst_rdma_chunk,
            rdma_buffer_size,
        )
        self.init_buffer(
            group,
            hidden_size=hidden_size,
            num_experts=num_experts,
            num_topk=num_topk,
            seqlen=seqlen,
            fp8_dispatch=fp8_dispatch,
            config=EPBufferConfig(
                num_sms=num_sms,
                dispatch_config=worst_config,
                combine_config=worst_config,
            ),
        )

        # Seed one real dispatch so later tune runs can use the cached path.
        (
            num_tokens_per_rank,
            num_tokens_per_rdma_rank,
            num_tokens_per_expert,
            is_token_in_rank,
            _,
        ) = self._buffer.get_dispatch_layout(topk_idx, num_experts)

        topk_weights_f = topk_weights.float() if topk_weights is not None else None
        seed_args = {
            "x": x,
            "num_tokens_per_rank": num_tokens_per_rank,
            "num_tokens_per_rdma_rank": num_tokens_per_rdma_rank,
            "is_token_in_rank": is_token_in_rank,
            "num_tokens_per_expert": num_tokens_per_expert,
            "topk_idx": topk_idx,
            "topk_weights": topk_weights_f,
        }
        # Seed one dispatch for the layout ``handle`` reused below; recv_x dropped.
        _, _, _, _, handle, _ = self._buffer.dispatch(**seed_args)

        num_warmup = max(1, num_tests // 5)

        # Sweep dispatch+combine over the (nvl_chunk, rdma_chunk) grid.
        candidates = [
            (nvl_chunk_size, rdma_chunk_size)
            for nvl_chunk_size in nvl_chunk_range
            for rdma_chunk_size in rdma_chunk_range
        ]

        def _cfg(cand):
            nvl, rdma = cand
            return ConfigClass(num_sms, nvl, nvl_buffer_size, rdma, rdma_buffer_size)

        def _dispatch(cand):
            # recv_x is element [0] of the handle-path dispatch return tuple.
            return self._buffer.dispatch(x=x, handle=handle, config=_cfg(cand))[0]

        def _combine(cand, recv):
            self._buffer.combine(x=recv, handle=handle, config=_cfg(cand))

        def _on_skip(cand, exc):
            if group.rank() == 0:
                logger.debug(f"[tuning] skip invalid cfg nvl/rdma={cand}: {exc!r}")

        best_d, best_c, best_dispatch_time, best_combine_time = sweep_configs(
            candidates,
            _dispatch,
            _combine,
            group=group,
            num_tests=num_tests,
            num_warmup=num_warmup,
            on_skip=_on_skip,
        )
        if best_d is None or best_dispatch_time == float("inf"):
            raise RuntimeError("tune_configs: no valid dispatch config found.")
        if best_c is None or best_combine_time == float("inf"):
            raise RuntimeError("tune_configs: no valid combine config found.")

        # Agree on rank-0's winners + times so every rank uses identical configs.
        d_win = _broadcast_from_rank0_int(list(best_d), group)
        c_win = _broadcast_from_rank0_int(list(best_c), group)
        best_dispatch_time = _broadcast_from_rank0_float(best_dispatch_time, group)
        best_combine_time = _broadcast_from_rank0_float(best_combine_time, group)

        dispatch_config = ConfigClass(num_sms, d_win[0], nvl_buffer_size, d_win[1], rdma_buffer_size)
        combine_config = ConfigClass(num_sms, c_win[0], nvl_buffer_size, c_win[1], rdma_buffer_size)

        if group.rank() == 0:
            logger.debug(
                f"[tuning] Best dispatch: SMs {num_sms}, NVL chunk {d_win[0]}, "
                f"RDMA chunk {d_win[1]}, {best_dispatch_time * 1e6:.2f} us; "
                f"Best combine: NVL chunk {c_win[0]}, RDMA chunk {c_win[1]}, "
                f"{best_combine_time * 1e6:.2f} us"
            )

        return dispatch_config, combine_config, best_dispatch_time, best_combine_time
