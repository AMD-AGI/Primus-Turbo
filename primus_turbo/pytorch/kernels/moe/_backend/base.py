###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""``EPBackend`` Protocol + ``_DeepEPLikeBackend`` shared base for DeepEP-style backends."""

import os
from dataclasses import dataclass
from typing import (
    Any,
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
from primus_turbo.pytorch.kernels.moe.moe_utils import (
    bench_kineto,
    detect_group_topology,
    inplace_unique,
)

from ._config import EPBufferConfig

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
) -> None:
    """Set each ``backend_env`` with the first available source:
    explicit value > existing env > ``nccl_env`` > leave unset.
    """
    for backend_env, nccl_env, explicit in mappings:
        if explicit is not None:
            os.environ[backend_env] = str(explicit)
            continue
        if backend_env in os.environ:
            continue
        nccl_val = os.environ.get(nccl_env)
        if nccl_val is not None:
            os.environ[backend_env] = nccl_val


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


@dataclass
class _DeepEPLikeKernelName:
    dispatch: Union[str, Tuple[str, ...]]
    combine: Union[str, Tuple[str, ...]]


class _DeepEPLikeBackend:
    """Shared base class for backends that follow the DeepEP Buffer protocol."""

    intranode_kernel_names = _DeepEPLikeKernelName(
        dispatch=("intranode::dispatch", "notify_dispatch"), combine=("intranode::combine", "notify_combine")
    )
    internode_kernel_names = _DeepEPLikeKernelName(
        dispatch=("internode::dispatch", "notify"), combine=("internode::combine", "notify")
    )

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
        # Best-effort NCCL-fallback for backend-specific network env vars.
        # No-op for backends without RDMA env settings; idempotent across
        # repeated ``init_buffer`` calls.
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

        if (
            self._buffer is None
            or not isinstance(self._buffer, BufferClass)
            or self._buffer.group != group
            or self._buffer.num_nvl_bytes < num_nvl_bytes
            or self._buffer.num_rdma_bytes < num_rdma_bytes
        ):
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
        """Release the communication buffer."""
        self._buffer = None

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

        kernel_profile_names = self.internode_kernel_names if num_nodes > 1 else self.intranode_kernel_names
        x_tensor = x if isinstance(x, torch.Tensor) else x[0]
        hidden_size = int(x_tensor.size(1))
        seqlen = int(x_tensor.size(0))
        fp8_dispatch = isinstance(x, tuple)
        # Byte count used for RDMA bandwidth accounting.
        hidden_bytes = hidden_size * max(1 if fp8_dispatch else 2, 2)

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

        # Per-(EP size) NVL/RDMA buffer sizes (chunks). Defaults come from
        # the upstream UCCL-EP tuning sweep and are known-good for DeepEP /
        # Turbo on the same shapes; revisit if a new EP topology under-fills
        # the sweep range.
        rdma_buffer_size, nvl_buffer_size = 512, (720 if ep_size in (144, 160) else 512)
        if ep_size == 24:
            nvl_buffer_size = 540

        # Sweep ranges; intranode pins rdma_chunk since RDMA is unused.
        nvl_chunk_range = range(1, 8, 1)
        if num_nodes <= 1:
            rdma_chunk_range = (16,)
        else:
            rdma_chunk_range = range(12 if num_nodes == 2 else 8, 33, 4)

        # Size buffer for the worst-case candidate to avoid reallocation.
        worst_nvl_chunk = max(nvl_chunk_range)
        worst_rdma_chunk = max(rdma_chunk_range)
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
        recv_x, _, _, _, handle, _ = self._buffer.dispatch(**seed_args)

        # Bandwidth accounting; RDMA is N/A on intranode groups.
        is_intranode = num_nodes <= 1
        if is_intranode:
            rdma_send_bytes = 0
        else:
            rdma_idx = topk_idx // (num_experts // num_nodes)
            rdma_idx.masked_fill_(topk_idx == -1, -1)
            inplace_unique(rdma_idx, num_nodes)
            num_rdma_token_sent = rdma_idx.ne(-1).sum().item()
            rdma_send_bytes = num_rdma_token_sent * hidden_bytes
        nvl_recv_bytes = recv_x.numel() * max(recv_x.element_size(), 2)

        # Tune dispatch configs.
        best_time, best_results = 1e10, None
        for nvl_chunk_size in nvl_chunk_range:
            for rdma_chunk_size in rdma_chunk_range:
                config = ConfigClass(
                    num_sms,
                    nvl_chunk_size,
                    nvl_buffer_size,
                    rdma_chunk_size,
                    rdma_buffer_size,
                )
                tune_args = {"x": x, "handle": handle, "config": config}
                t, notify_t = bench_kineto(
                    lambda: self._buffer.dispatch(**tune_args),  # noqa: B023
                    kernel_profile_names.dispatch,
                    suppress_kineto_output=True,
                    num_tests=num_tests,
                )
                if t == 0 or notify_t == 0:
                    continue
                if t + notify_t < best_time:
                    best_time = t + notify_t
                    best_results = (
                        num_sms,
                        nvl_chunk_size,
                        rdma_chunk_size,
                        t,
                        notify_t,
                    )

        if best_results is None:
            raise RuntimeError(
                "tune_configs: no valid dispatch config found " "(all candidates reported zero kernel time)."
            )
        best_dispatch_results = _broadcast_from_rank0_int(best_results[:3], group)
        best_dispatch_time = _broadcast_from_rank0_float(best_time, group)
        if group.rank() == 0:
            dispatch_rdma_bw = "N/A" if is_intranode else f"{rdma_send_bytes / 1e9 / best_time:.2f} GB/s"
            logger.debug(
                f"[tuning] Best dispatch: SMs {best_results[0]}, "
                f"NVL chunk {best_results[1]}, RDMA chunk {best_results[2]}, "
                f"transmit: {best_results[3] * 1e6:.2f} us, "
                f"notify: {best_results[4] * 1e6:.2f} us, "
                f"total: {best_time * 1e6:.2f} us, "
                f"BW: {dispatch_rdma_bw} (RDMA), "
                f"{nvl_recv_bytes / 1e9 / best_time:.2f} GB/s (NVL)"
            )
        dispatch_config = ConfigClass(
            best_dispatch_results[0],
            best_dispatch_results[1],
            nvl_buffer_size,
            best_dispatch_results[2],
            rdma_buffer_size,
        )

        dispatch_args = {
            "x": x,
            "num_tokens_per_rank": num_tokens_per_rank,
            "num_tokens_per_rdma_rank": num_tokens_per_rdma_rank,
            "is_token_in_rank": is_token_in_rank,
            "num_tokens_per_expert": num_tokens_per_expert,
            "config": dispatch_config,
        }
        recv_x, _, _, _, handle, _ = self._buffer.dispatch(**dispatch_args)

        # Combine bandwidth mirrors dispatch (NVL-send / RDMA-recv).
        combine_nvl_send_bytes = recv_x.numel() * max(recv_x.element_size(), 2)
        combine_rdma_recv_bytes = rdma_send_bytes

        # Tune combine configs.
        best_time, best_results = 1e10, None
        for nvl_chunk_size in nvl_chunk_range:
            for rdma_chunk_size in rdma_chunk_range:
                config = ConfigClass(
                    num_sms,
                    nvl_chunk_size,
                    nvl_buffer_size,
                    rdma_chunk_size,
                    rdma_buffer_size,
                )
                tune_args = {"x": recv_x, "handle": handle, "config": config}
                t, notify_t = bench_kineto(
                    lambda: self._buffer.combine(**tune_args),  # noqa: B023
                    kernel_profile_names.combine,
                    suppress_kineto_output=True,
                    num_tests=num_tests,
                )
                if t == 0 or notify_t == 0:
                    continue
                if t + notify_t < best_time:
                    best_time = t + notify_t
                    best_results = (
                        num_sms,
                        nvl_chunk_size,
                        rdma_chunk_size,
                        t,
                        notify_t,
                    )

        if best_results is None:
            raise RuntimeError(
                "tune_configs: no valid combine config found " "(all candidates reported zero kernel time)."
            )
        best_combine_results = _broadcast_from_rank0_int(best_results[:3], group)
        best_combine_time = _broadcast_from_rank0_float(best_time, group)
        if group.rank() == 0:
            combine_rdma_bw = (
                "N/A" if is_intranode else f"{combine_rdma_recv_bytes / 1e9 / best_time:.2f} GB/s"
            )
            logger.debug(
                f"[tuning] Best combine: SMs {best_results[0]}, "
                f"NVL chunk {best_results[1]}, RDMA chunk {best_results[2]}, "
                f"transmit: {best_results[3] * 1e6:.2f} us, "
                f"notify: {best_results[4] * 1e6:.2f} us, "
                f"total: {best_time * 1e6:.2f} us, "
                f"BW: {combine_rdma_bw} (RDMA), "
                f"{combine_nvl_send_bytes / 1e9 / best_time:.2f} GB/s (NVL)"
            )
        combine_config = ConfigClass(
            best_combine_results[0],
            best_combine_results[1],
            nvl_buffer_size,
            best_combine_results[2],
            rdma_buffer_size,
        )

        return dispatch_config, combine_config, best_dispatch_time, best_combine_time
