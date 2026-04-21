###############################################################################
# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Modification Copyright© 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import inspect
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    Type,
    Union,
    runtime_checkable,
)

import torch
import torch.distributed as dist

from primus_turbo.common.logger import logger
from primus_turbo.pytorch.core.backend import (
    GlobalBackendManager,
    PrecisionType,
    TuneCache,
)
from primus_turbo.pytorch.kernels.moe.moe_utils import (
    bench_kineto,
    detect_group_topology,
    inplace_unique,
)

# =========================================================================
# Buffer configuration
# =========================================================================


@dataclass
class EPBufferConfig:
    """Configuration for EP communication buffer initialization.

    Attributes:
        num_sms: Number of SMs to use in high-throughput kernels.
        dispatch_config: Optional user-provided dispatch config (from offline
            benchmarking). When ``None``, the backend's default for the current
            ``ep_size`` is used (``Buffer.get_dispatch_config(ep_size)``).
        combine_config: Optional user-provided combine config.  Same fallback
            behaviour as *dispatch_config*.
    """

    num_sms: int = 32
    dispatch_config: Any = None
    combine_config: Any = None


_DEFAULT_BUFFER_CONFIG = EPBufferConfig(
    num_sms=32,
    dispatch_config=None,
    combine_config=None,
)

_buffer_config: EPBufferConfig = _DEFAULT_BUFFER_CONFIG


def set_buffer_global_config(
    num_use_cu: int = 32,
    autotune_config: Optional[tuple] = None,
) -> None:
    """Store the SM count and optional per-operation configs.

    This is typically called once by the token dispatcher during ``__init__``.

    Args:
        num_use_cu: Number of SMs (compute units) for high-throughput kernels.
        autotune_config: Legacy parameter — a ``(dispatch_config, combine_config)``
            tuple obtained from offline benchmarking.  ``None`` means use the
            backend's built-in defaults for the current EP group size.
    """
    global _buffer_config
    dispatch_cfg, combine_cfg = autotune_config if autotune_config is not None else (None, None)
    _buffer_config = EPBufferConfig(
        num_sms=num_use_cu,
        dispatch_config=dispatch_cfg,
        combine_config=combine_cfg,
    )


# =========================================================================
# EPBackend Protocol
# =========================================================================


@runtime_checkable
class EPBackend(Protocol):
    """Structural (``typing.Protocol``) interface for Expert-Parallel
    communication backends.

    Each backend encapsulates a specific EP library (e.g. in-tree Turbo DeepEP,
    external ``deep_ep``, UCCL-EP, ...) and owns its own buffer lifecycle.
    Adding a new backend is a single-class change plus one
    ``register_ep_backend()`` call — any class that structurally conforms to
    this Protocol is accepted; no explicit inheritance is required.
    """

    @staticmethod
    def is_available() -> bool:
        """Return True if this backend's dependencies are importable."""
        ...

    def init_buffer(
        self,
        group: dist.ProcessGroup,
        hidden_bytes: int,
        config: EPBufferConfig,
    ) -> None:
        """(Re-)create the communication buffer if needed."""
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
    ) -> Tuple[
        Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[Union[List[int], torch.Tensor]],
        Optional[tuple],
    ]:
        """Execute dispatch (layout + send) and return
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
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Execute combine and return ``(combined_x, combined_topk_weights)``."""
        ...

    def release_buffer(self) -> None:
        """Release the communication buffer."""
        ...


def _broadcast_from_rank0_int(values: Sequence[int], group: dist.ProcessGroup) -> List[int]:
    """Return rank-0's ``values`` on every rank (via ``all_gather``).

    Used to reach a globally consistent winning tune config: each rank picks
    its own local optimum, and we arbitrarily adopt rank 0's pick on every
    rank so the runtime config is identical across the group.
    """
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
    """Shared logic for all backends that follow the DeepEP Buffer protocol
    (``get_dispatch_layout`` / ``dispatch`` / ``combine`` / ``set_num_sms`` /
    ``get_dispatch_config`` / ``get_combine_config``).

    This is a plain implementation base class — it does **not** inherit from
    ``EPBackend``. Conformance to the ``EPBackend`` Protocol is checked
    structurally by the type system.

    Subclasses only need to override ``is_available``, ``_get_module``, and
    optionally ``_make_buffer_kwargs`` to supply backend-specific constructor
    arguments.
    """

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

    @staticmethod
    def is_available() -> bool:
        """Return True if this backend's dependencies are importable."""
        raise NotImplementedError

    @staticmethod
    def _get_module():
        """Return the Python module that exposes ``Buffer``, ``Config``,
        ``EventHandle``, ``EventOverlap`` (or a compatible ``utils`` sub-module).
        """
        raise NotImplementedError

    def _make_buffer_kwargs(self, group: dist.ProcessGroup) -> dict:
        """Extra keyword arguments forwarded to ``BufferClass(group, nvl, rdma, **kwargs)``."""
        return {}

    # ------------------------------------------------------------------
    # EPBackend interface
    # ------------------------------------------------------------------

    def init_buffer(
        self,
        group: dist.ProcessGroup,
        hidden_bytes: int,
        config: EPBufferConfig,
    ) -> None:
        mod = self._get_module()
        BufferClass = mod.Buffer

        BufferClass.set_num_sms(config.num_sms)

        dispatch_config = config.dispatch_config or BufferClass.get_dispatch_config(group.size())
        combine_config = config.combine_config or BufferClass.get_combine_config(group.size())

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
    ):
        EventOverlapClass, EventHandleClass = self._get_event_classes()
        buffer = self._buffer
        assert buffer is not None, "init_buffer() must be called before dispatch()"

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
            ) = buffer.get_dispatch_layout(
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
            ) = buffer.dispatch(
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
            )
        else:
            recv_x, recv_token_indices, recv_token_probs, tokens_per_expert, handle, after_event = (
                buffer.dispatch(
                    x,
                    handle=handle,
                    previous_event=previous_event,
                    async_finish=async_finish,
                    allocate_on_comm_stream=allocate_on_comm_stream,
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
    ):
        EventOverlapClass, EventHandleClass = self._get_event_classes()
        buffer = self._buffer
        assert buffer is not None, "init_buffer() must be called before combine()"

        previous_event = EventOverlapClass(EventHandleClass()) if async_finish else None

        combined_x, combined_topk_weights, after_event = buffer.combine(
            x,
            handle=handle,
            topk_weights=None if topk_weights is None else topk_weights.float(),
            async_finish=async_finish,
            allocate_on_comm_stream=allocate_on_comm_stream,
            previous_event=previous_event,
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
        """Tune dispatch / combine ``Config`` for this backend on the given case.

        Follows the benchmark recipe in
        ``benchmark/ops/deep_ep/test_intranode.py``: for each candidate
        ``(nvl_chunk_size, rdma_chunk_size)``, build a ``Config`` and measure
        cached dispatch / combine latency. Each rank picks its own local
        optimum; to reach a globally consistent config we then adopt rank 0's
        pick on every rank (see :func:`_broadcast_from_rank0_int`). The
        returned latencies are rank 0's and are broadcast to every rank so
        the values are identical across the group.

        Args:
            group: The EP process group.
            x: Dispatch input (Tensor or ``(fp8_tensor, scales)`` tuple).
            topk_idx: ``[num_tokens, num_topk]`` expert indices. Required when
                ``uniform_dispatch=False``. In uniform mode this argument is
                only used to derive ``num_topk`` when ``num_topk`` is not
                explicitly provided; its values are ignored and a fresh
                near-uniform ``topk_idx`` is sampled internally.
            topk_weights: ``[num_tokens, num_topk]`` expert weights. Required
                when ``uniform_dispatch=False``; ignored (resampled) in
                uniform mode.
            num_experts: Total number of experts.
            num_sms: SM count to use for tuning (matches runtime setting).
            num_tests: Timed iterations for each candidate.
            num_topk: Number of experts per token. Only consulted in uniform
                mode when ``topk_idx`` is not supplied.
            uniform_dispatch: If ``True`` (default), generate a fresh
                near-uniform ``topk_idx`` / ``topk_weights`` inside the
                tuner — ``scores = |N(0,1)| + 1`` followed by ``topk`` over
                experts, matching the recipe in
                ``benchmark/ops/deep_ep/test_internode.py``. Tuning against
                a uniform dispatch distribution produces a more robust
                config than tuning against a specific, possibly skewed,
                runtime workload.

        Returns:
            ``(best_dispatch_config, best_combine_config, best_dispatch_s, best_combine_s)``
            where the times are seconds and identical on all ranks (rank 0's
            local optimum, broadcast).
        """
        mod = self._get_module()
        ConfigClass = mod.Config
        ep_size = group.size()
        _, num_nodes = detect_group_topology(group)

        kernel_profile_names = self.internode_kernel_names if num_nodes > 1 else self.intranode_kernel_names
        hidden_bytes = get_hidden_bytes(x)

        # --- Resolve (topk_idx, topk_weights) used for tuning ---------------
        # In uniform mode we discard any real-workload values and resample a
        # near-uniform distribution so the tuner does not overfit to a
        # specific, possibly skewed, routing pattern.
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

        # tune config from uccl-ep
        rdma_buffer_size, nvl_buffer_size = 512, (720 if ep_size in (144, 160) else 512)
        if ep_size == 24:
            nvl_buffer_size = 540

        # Tune-candidate sweep ranges (shared by dispatch and combine).
        # On intranode (single-node) groups RDMA is not involved, so the
        # ``rdma_chunk_size`` value is inert: pin it to a single value to
        # avoid running the whole NVL sweep N times for nothing.
        nvl_chunk_range = range(1, 8, 1)
        if num_nodes <= 1:
            rdma_chunk_range = (16,)
        else:
            rdma_chunk_range = range(12 if num_nodes == 2 else 8, 33, 4)

        # Allocate a buffer sized for the worst-case candidate so neither tune
        # loop has to re-allocate (which would invalidate any live ``handle``).
        worst_nvl_chunk = max(nvl_chunk_range)
        worst_rdma_chunk = max(rdma_chunk_range)
        worst_config = ConfigClass(
            num_sms,
            worst_nvl_chunk,
            nvl_buffer_size,
            worst_rdma_chunk,
            rdma_buffer_size,
        )
        # alloc worst-case buffer for tuning, it will be release after finish tuning
        self.init_buffer(
            group,
            hidden_bytes,
            EPBufferConfig(
                num_sms=num_sms,
                dispatch_config=worst_config,
                combine_config=worst_config,
            ),
        )

        # Seed handle: one real dispatch so later runs can use the cached path.
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
        if isinstance(recv_x, tuple):
            recv_x = recv_x[0]

        # Bandwidth bookkeeping. On intranode (``num_nodes == 1``) RDMA does
        # not participate, so we report N/A rather than a meaningless number.
        is_intranode = num_nodes <= 1
        if is_intranode:
            rdma_send_bytes = 0
        else:
            rdma_idx = topk_idx // (num_experts // num_nodes)
            rdma_idx.masked_fill_(topk_idx == -1, -1)
            inplace_unique(rdma_idx, num_nodes)
            num_rdma_token_sent = rdma_idx.ne(-1).sum().item()
            rdma_send_bytes = num_rdma_token_sent * hidden_bytes
        # ``recv_x.numel() * element_size`` is the total NVL-received bytes;
        # ``hidden_bytes`` already includes ``hidden`` so multiplying by
        # ``numel`` would double-count it.
        nvl_recv_bytes = recv_x.numel() * max(recv_x.element_size(), 2)

        # --- Tune dispatch configs -----------------------------------------
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

        # Combine only accepts BF16 input. If the caller tuned with an FP8
        # tuple, re-dispatch with a BF16 surrogate of the same shape so the
        # combine sweep below can use ``recv_x`` directly.
        if isinstance(x, tuple):
            x_for_combine_tune = torch.empty(x[0].shape, dtype=torch.bfloat16, device=x[0].device)
        else:
            x_for_combine_tune = x
        dispatch_args = {
            "x": x_for_combine_tune,
            "num_tokens_per_rank": num_tokens_per_rank,
            "num_tokens_per_rdma_rank": num_tokens_per_rdma_rank,
            "is_token_in_rank": is_token_in_rank,
            "num_tokens_per_expert": num_tokens_per_expert,
            "config": dispatch_config,
        }
        recv_x, _, _, _, handle, _ = self._buffer.dispatch(**dispatch_args)

        # Combine bandwidth accounting: combine sends out over NVL what
        # dispatch received, and receives over RDMA what dispatch sent.
        combine_nvl_send_bytes = recv_x.numel() * max(recv_x.element_size(), 2)
        combine_rdma_recv_bytes = rdma_send_bytes

        # --- Tune combine configs ------------------------------------------
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


# =========================================================================
# Concrete backends
# =========================================================================


class TurboEPBackend(_DeepEPLikeBackend):
    """In-tree Primus-Turbo DeepEP backend (always available)."""

    @staticmethod
    def is_available() -> bool:
        return True

    @staticmethod
    def _get_module():
        import primus_turbo.pytorch.deep_ep as turbo_ep

        return turbo_ep


class DeepEPBackend(_DeepEPLikeBackend):
    """External ``deep_ep`` package backend (optional)."""

    @staticmethod
    def is_available() -> bool:
        try:
            import deep_ep  # noqa: F401

            return True
        except ImportError:
            return False

    @staticmethod
    def _get_module():
        import deep_ep

        return deep_ep

    def _make_buffer_kwargs(self, group: dist.ProcessGroup) -> dict:
        BufferClass = self._get_module().Buffer
        try:
            param = inspect.signature(BufferClass).parameters.get("is_intranode")
        except (TypeError, ValueError):
            param = None
        if param is not None and param.default is False:
            # uccl-ep special handle
            return {"is_intranode": group.size() <= 8}
        return {}


# =========================================================================
# Backend registry
# =========================================================================

_BACKEND_REGISTRY: Dict[str, Type[EPBackend]] = {
    "TURBO": TurboEPBackend,
    "DEEP_EP": DeepEPBackend,
}

_backend_instances: Dict[str, EPBackend] = {}


def register_ep_backend(name: str, cls: Type[EPBackend]) -> None:
    """Register a new EP backend class (e.g. ``UCCL_EP``)."""
    _BACKEND_REGISTRY[name] = cls


def _get_backend_instance(name: str) -> EPBackend:
    """Lazily create and cache a backend singleton."""
    if name not in _backend_instances:
        if name not in _BACKEND_REGISTRY:
            raise ValueError(f"Unknown EP backend '{name}'. Available: {list(_BACKEND_REGISTRY.keys())}")
        cls = _BACKEND_REGISTRY[name]
        if not cls.is_available():
            raise RuntimeError(
                f"EP backend '{name}' is registered but its dependencies are not "
                f"installed. Please install the required package."
            )
        _backend_instances[name] = cls()
    return _backend_instances[name]


# =========================================================================
# Autotuner — finds best (backend, dispatch_config, combine_config) per case
# =========================================================================


@dataclass(frozen=True)
class _EPAutoTuneKey:
    """Shape-based signature used to cache autotune results."""

    num_tokens: int
    hidden: int
    num_topk: int
    num_experts: int
    ep_size: int
    dtype: torch.dtype
    use_fp8: bool


@dataclass
class EPAutoTuneResult:
    """Outcome of :class:`MoEDispatchCombineAutoTuner.tune`.

    Attributes:
        backend_name: Registry name of the winning backend (e.g. ``"TURBO"``).
        dispatch_config: Best dispatch ``Config`` for that backend.
        combine_config: Best combine ``Config`` for that backend.
        dispatch_time_us: Average dispatch latency of the winner (µs).
        combine_time_us: Average combine latency of the winner (µs).
        per_backend: Map ``backend_name -> (d_time_us, c_time_us)`` for every
            candidate backend that completed tuning (for logging / debugging).
        tune_duration_s: Wall-clock seconds spent sweeping all candidate
            backends + configs to produce this result (rank 0 clock).
    """

    backend_name: str
    dispatch_config: Any
    combine_config: Any
    dispatch_time_us: float
    combine_time_us: float
    per_backend: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    tune_duration_s: float = 0.0


class MoEDispatchCombineAutoTuner:
    """Autotuner that selects the best *(backend, dispatch_config, combine_config)*
    for a given MoE dispatch / combine case.

    Design
    ------
    Dispatch and combine come in pairs: a single ``dispatch()`` produces a
    ``handle`` that is later consumed by the paired ``combine()`` (and, in
    autograd, by a reverse-direction ``dispatch()`` during backward). The
    combine input tensor has a different shape from the dispatch input, so
    combine **cannot** reconstruct a shape-based cache key on its own.

    Therefore tuning is driven solely by ``moe_dispatch``:

    1. ``moe_dispatch`` computes a shape-based key from ``(x, topk_idx,
       num_experts, ep_size, dtype)`` and either reuses a cached
       :class:`EPAutoTuneResult` or runs the full *(backend × dispatch ×
       combine)* sweep once.
    2. The chosen :class:`EPAutoTuneResult` is then *bound* to the freshly
       returned ``handle`` via :meth:`register_handle`.
    3. ``moe_combine`` / cached-``moe_dispatch`` / autograd backward use the
       same ``handle`` and simply call :meth:`lookup_handle(handle)`` to pull
       the already-tuned configs out of cache — no measurement, no shape
       lookup on their own input tensors.

    Explicit usage::

        result = MoEDispatchCombineAutoTuner.get_or_tune(
            group, x, topk_idx, topk_weights, num_experts,
        )

    Automatic (env var driven)::

        export PRIMUS_TURBO_AUTO_TUNE=1
        # first dispatch on a new shape triggers tuning, later calls reuse.
    """

    # Shape-based cache: key -> EPAutoTuneResult (re-used across MoE layers
    # that share the same input shape).
    _cache: TuneCache = TuneCache(capacity=1024)

    # Handle-based cache: id(handle) -> EPAutoTuneResult. LRU-bounded so that
    # long-running jobs do not accumulate stale entries.
    _HANDLE_CACHE_MAX: int = 1024
    _handle_cache: "OrderedDict[int, EPAutoTuneResult]" = OrderedDict()

    # Most-recent result: fallback for the (rare) cases where a call is made
    # before any ``moe_dispatch`` has registered a handle mapping.
    _current_result: Optional[EPAutoTuneResult] = None

    @staticmethod
    def make_key(
        x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        topk_idx: Optional[torch.Tensor],
        num_experts: int,
        ep_size: int,
        num_topk: Optional[int] = None,
    ) -> _EPAutoTuneKey:
        inp = x if isinstance(x, torch.Tensor) else x[0]
        num_tokens, hidden = int(inp.size(0)), int(inp.size(1))
        if num_topk is not None:
            resolved_num_topk = int(num_topk)
        elif topk_idx is not None:
            resolved_num_topk = int(topk_idx.size(1))
        else:
            resolved_num_topk = 0
        return _EPAutoTuneKey(
            num_tokens=num_tokens,
            hidden=hidden,
            num_topk=resolved_num_topk,
            num_experts=int(num_experts),
            ep_size=int(ep_size),
            dtype=inp.dtype,
            use_fp8=isinstance(x, tuple),
        )

    # ------------------------------------------------------------------
    # Handle <-> result binding (dispatch/combine pair)
    # ------------------------------------------------------------------

    @classmethod
    def register_handle(cls, handle: Any, result: EPAutoTuneResult) -> None:
        """Bind ``result`` to ``id(handle)`` so the matching ``combine`` /
        cached-``dispatch`` call can recover the tuned configs without any
        shape-based lookup.

        ``handle`` is the tuple returned by the backend's ``dispatch`` call;
        we key by its Python ``id`` because the tuple itself is not hashable
        as a cache key (it contains tensors) and is guaranteed to be unique
        for as long as it is alive — which, under autograd, is exactly the
        span during which combine / backward will need it.
        """
        if handle is None:
            return
        hid = id(handle)
        cache = cls._handle_cache
        if hid in cache:
            cache.move_to_end(hid)
            cache[hid] = result
        else:
            cache[hid] = result
            if len(cache) > cls._HANDLE_CACHE_MAX:
                cache.popitem(last=False)

    @classmethod
    def lookup_handle(cls, handle: Any) -> Optional[EPAutoTuneResult]:
        """Return the :class:`EPAutoTuneResult` previously bound to ``handle``
        via :meth:`register_handle`, or ``None`` if the handle is unknown.
        """
        if handle is None:
            return None
        res = cls._handle_cache.get(id(handle))
        if res is not None:
            cls._handle_cache.move_to_end(id(handle))
        return res

    @classmethod
    def _candidate_backend_names(cls) -> List[str]:
        names: List[str] = []
        for name, impl in _BACKEND_REGISTRY.items():
            try:
                if impl.is_available():
                    names.append(name)
            except Exception:
                continue
        return names

    @classmethod
    @torch.no_grad()
    def tune(
        cls,
        group: dist.ProcessGroup,
        x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        num_experts: int,
        *,
        topk_idx: Optional[torch.Tensor] = None,
        topk_weights: Optional[torch.Tensor] = None,
        num_sms: int = 32,
        candidate_backends: Optional[Sequence[str]] = None,
        num_tests: int = 20,
        num_topk: Optional[int] = None,
        uniform_dispatch: bool = True,
    ) -> EPAutoTuneResult:
        """Tune across candidate backends + configs, return the best result.

        Each candidate backend's :meth:`_DeepEPLikeBackend.tune_configs` is
        called in turn and the one with the smallest *dispatch + combine*
        latency wins. The resulting :class:`EPAutoTuneResult` is **not**
        cached here; use :meth:`get_or_tune` for cached access.

        ``uniform_dispatch`` is forwarded to each backend's ``tune_configs``
        (default ``True``): tuning against a near-uniform topk distribution
        generally yields a more robust config than fitting a specific runtime
        routing pattern. See :meth:`_DeepEPLikeBackend.tune_configs` for
        details.
        """
        names = list(candidate_backends) if candidate_backends is not None else cls._candidate_backend_names()
        if not names:
            raise RuntimeError("MoE autotune: no EP backends are available.")

        # Wall-clock timer for the whole sweep. Each backend's
        # ``tune_configs`` runs real GPU kernels + host-side profiler setup,
        # so this is the user-visible "time spent tuning" number.
        tune_start = time.perf_counter()

        best: Optional[EPAutoTuneResult] = None
        best_total = float("inf")
        per_backend: Dict[str, Tuple[float, float]] = {}

        for name in names:
            if name not in _BACKEND_REGISTRY or not _BACKEND_REGISTRY[name].is_available():
                continue
            backend = _get_backend_instance(name)
            try:
                d_cfg, c_cfg, d_t, c_t = backend.tune_configs(
                    group=group,
                    x=x,
                    num_experts=num_experts,
                    topk_idx=topk_idx,
                    topk_weights=topk_weights,
                    num_sms=num_sms,
                    num_tests=num_tests,
                    num_topk=num_topk,
                    uniform_dispatch=uniform_dispatch,
                )
            except Exception as exc:  # pragma: no cover - defensive
                # Some exceptions (bare ``assert``, no-arg raises, certain
                # pybind-backed C++ errors) have an empty ``str(exc)``; use
                # ``repr`` + ``exc_info`` so the message and traceback are
                # never silently blank.
                logger.warning(
                    f"MoE autotune: backend '{name}' failed: \n" f"{type(exc).__name__}: {exc!r}",
                    exc_info=True,
                )
                continue
            finally:
                # ``tune_configs`` sizes its buffer for the worst-case
                # candidate, and ``init_buffer`` only grows, never shrinks.
                # Drop the buffer here so the next runtime ``init_buffer``
                # (using the best config) reallocates at the right size
                # instead of keeping the worst-case allocation around.
                backend.release_buffer()

            per_backend[name] = (round(d_t * 1e6, 3), round(c_t * 1e6, 3))
            total = d_t + c_t
            if total < best_total:
                best_total = total
                best = EPAutoTuneResult(
                    backend_name=name,
                    dispatch_config=d_cfg,
                    combine_config=c_cfg,
                    dispatch_time_us=round(d_t * 1e6, 3),
                    combine_time_us=round(c_t * 1e6, 3),
                )

        if best is None:
            raise RuntimeError(
                f"MoE autotune: all candidate backends failed ({names}). "
                f"Check logs and EP buffer configuration."
            )
        best.per_backend = per_backend
        best.tune_duration_s = time.perf_counter() - tune_start
        return best

    @classmethod
    @torch.no_grad()
    def get_or_tune(
        cls,
        group: dist.ProcessGroup,
        x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        num_experts: int,
        *,
        topk_idx: Optional[torch.Tensor] = None,
        topk_weights: Optional[torch.Tensor] = None,
        num_sms: int = 32,
        num_topk: Optional[int] = None,
        uniform_dispatch: bool = True,
        **tune_kwargs: Any,
    ) -> EPAutoTuneResult:
        """Return a cached tune result for this shape or run :meth:`tune` now.

        Also updates the process-wide *current* result used by subsequent
        dispatch / combine calls when full input tensors are unavailable
        (e.g. cached-dispatch, combine, backward).

        In uniform mode (default), ``topk_idx`` / ``topk_weights`` may be
        ``None`` as long as ``num_topk`` is provided — the tuner will
        synthesise a near-uniform distribution internally. ``num_topk`` is
        also consulted when building the shape cache key, so the same
        ``(num_tokens, hidden, num_topk, num_experts, ep_size, dtype)``
        signature reuses the cached winner across calls.
        """
        key = cls.make_key(x, topk_idx, num_experts, group.size(), num_topk=num_topk)
        cached = cls._cache.get(key)
        if cached is not None:
            cls._current_result = cached
            return cached

        result = cls.tune(
            group=group,
            x=x,
            num_experts=num_experts,
            topk_idx=topk_idx,
            topk_weights=topk_weights,
            num_sms=num_sms,
            num_topk=num_topk,
            uniform_dispatch=uniform_dispatch,
            **tune_kwargs,
        )
        cls._cache.put(key, result)
        cls._current_result = result
        logger.info(
            f"[MoE AutoTune] key={key} -> backend={result.backend_name} "
            f"dispatch={result.dispatch_time_us:.1f}us combine={result.combine_time_us:.1f}us "
            f"tune_time={result.tune_duration_s:.2f}s "
            f"per_backend={result.per_backend}",
            rank=0,
        )
        return result

    @classmethod
    def current(cls) -> Optional[EPAutoTuneResult]:
        """Return the most recent tune result, or ``None`` if never tuned.

        Only used as a last-resort fallback when both handle-based and
        shape-based lookup fail.
        """
        return cls._current_result

    @classmethod
    def set_current(cls, result: Optional[EPAutoTuneResult]) -> None:
        """Override the current runtime result (mainly for testing)."""
        cls._current_result = result

    @classmethod
    def clear(cls) -> None:
        """Drop every cache (shape, handle, current)."""
        cls._cache.clear()
        cls._handle_cache.clear()
        cls._current_result = None


# =========================================================================
# Backend selection
# =========================================================================

_DEFAULT_BACKEND_NAME = "TURBO"


def _get_backend_name() -> str:
    """
    User-selected backend name, or ``TURBO`` by default.
    """
    bt = GlobalBackendManager.get_moe_dispatch_combine_backend(PrecisionType.BF16_FP16_FP32)
    return bt.name if bt is not None else _DEFAULT_BACKEND_NAME


# =========================================================================
# Utilities
# =========================================================================


def get_hidden_bytes(
    x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
) -> int:
    """Calculate the number of hidden bytes for a tensor.

    Uses at least 2 bytes (bf16 size) so buffers work for both fp8 and bf16
    without reallocation.
    """
    inp = x if isinstance(x, torch.Tensor) else x[0]
    return inp.size(1) * max(inp.element_size(), 2)


def _get_runtime_config(
    x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
    group: dist.ProcessGroup,
    handle: Optional[tuple],
    topk_idx: Optional[torch.Tensor],
    token_weights: Optional[torch.Tensor],
    num_experts: Optional[int],
) -> Tuple[str, EPBufferConfig, Optional[EPAutoTuneResult]]:
    """Determine ``(backend_name, buffer_config, autotune_result)`` for the
    current call.

    Two-branch semantics:

    * **Autotune disabled** — ``backend_name`` comes from the user override
      (code-level :class:`GlobalBackendManager` setting or the
      ``PRIMUS_TURBO_MOE_DISPATCH_COMBINE_BACKEND`` env var), falling back
      to the built-in default. ``buffer_config`` is the user-provided
      :data:`_buffer_config` populated by :func:`set_buffer_global_config`.
      No measurement is performed.

    * **Autotune enabled** — the user's backend override is **ignored**;
      every available EP backend is swept and the fastest *(backend,
      dispatch_config, combine_config)* triple wins. Resolution is layered:

        (a) Handle-keyed lookup: follow-up calls (``moe_combine``, cached
            dispatch, autograd backward) share the same ``handle`` as the
            paired primary ``moe_dispatch`` and retrieve the tuned result
            directly from :meth:`MoEDispatchCombineAutoTuner.lookup_handle`.
        (b) Shape-keyed tune / lookup: a primary ``moe_dispatch`` (no
            handle yet + full inputs) triggers
            :meth:`MoEDispatchCombineAutoTuner.get_or_tune` which either
            reuses a cached :class:`EPAutoTuneResult` for that shape or
            runs the full sweep once.
        (c) Last-resort fallback: reuse
            :meth:`MoEDispatchCombineAutoTuner.current` when neither (a)
            nor (b) applies (e.g. autotune was flipped on mid-run).

    The returned ``autotune_result`` is propagated back to
    :func:`moe_dispatch_impl` so it can
    :meth:`MoEDispatchCombineAutoTuner.register_handle` the freshly
    produced handle; ``None`` means "no autotune was used for this call".
    """
    assert _buffer_config is not None

    # ---- (1) Autotune off: get the user's backend + configs. ----------
    if not GlobalBackendManager.auto_tune_enabled():
        return _get_backend_name(), _buffer_config, None

    # ---- (2) Autotune on: the user's backend choice is ignored. ---------
    result: Optional[EPAutoTuneResult] = None

    # (a) Handle-keyed cache — the dispatch/combine pair shares the same
    # ``handle``, so any non-primary call (combine, cached dispatch,
    # backward) resolves the configs purely from it.
    if handle is not None:
        result = MoEDispatchCombineAutoTuner.lookup_handle(handle)

    # (b) Primary dispatch (no handle yet + full inputs): trigger / reuse
    # the shape-keyed tune result. ``uniform_dispatch`` stays at its default
    # (``True``) so the tuner ignores the runtime routing distribution and
    # tunes against a near-uniform one for robustness.
    if (
        result is None
        and handle is None
        and topk_idx is not None
        and token_weights is not None
        and num_experts is not None
    ):
        result = MoEDispatchCombineAutoTuner.get_or_tune(
            group=group,
            x=x,
            num_experts=int(num_experts),
            topk_idx=topk_idx,
            topk_weights=token_weights,
            num_sms=_buffer_config.num_sms,
        )

    # (c) Last-resort fallback: autotune was enabled without ever running a
    # primary dispatch (e.g. toggled mid-flight); use the most recent result.
    if result is None:
        result = MoEDispatchCombineAutoTuner.current()

    if result is not None:
        cfg = EPBufferConfig(
            num_sms=_buffer_config.num_sms,
            dispatch_config=result.dispatch_config,
            combine_config=result.combine_config,
        )
        return result.backend_name, cfg, result

    # Autotune enabled but nothing to go on yet — fall back to user/default.
    # The next primary ``moe_dispatch`` will populate the cache.
    return _get_backend_name(), _buffer_config, None


# =========================================================================
# Public API — used by ``moe_dispatch_combine.py``
# =========================================================================


def moe_dispatch_impl(
    x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
    group: dist.ProcessGroup,
    handle: Optional[tuple] = None,
    topk_idx: Optional[torch.Tensor] = None,
    token_weights: Optional[torch.Tensor] = None,
    num_experts: Optional[int] = None,
    async_finish: bool = False,
    allocate_on_comm_stream: bool = False,
    num_worst_tokens: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, tuple]:
    name, cfg, result = _get_runtime_config(x, group, handle, topk_idx, token_weights, num_experts)
    backend = _get_backend_instance(name)
    backend.init_buffer(group, get_hidden_bytes(x), cfg)
    outputs = backend.dispatch(
        x,
        handle=handle,
        topk_idx=topk_idx,
        token_weights=token_weights,
        num_experts=num_experts,
        async_finish=async_finish,
        allocate_on_comm_stream=allocate_on_comm_stream,
        num_worst_tokens=num_worst_tokens,
    )

    # Bind the autotune result to the newly returned ``handle`` so the paired
    # combine (or a later cached-dispatch / backward with the same handle)
    # can recover the configs without having to redo any lookup.
    if result is not None:
        new_handle = outputs[-1]
        MoEDispatchCombineAutoTuner.register_handle(new_handle, result)

    return outputs


def moe_combine_impl(
    x: torch.Tensor,
    group: dist.ProcessGroup,
    handle: tuple,
    topk_weights: Optional[torch.Tensor] = None,
    async_finish: bool = False,
    allocate_on_comm_stream: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    # Combine never runs its own tune: it resolves the tuned config purely
    # by ``handle`` (set by the paired :func:`moe_dispatch_impl`) because the
    # combine input ``x`` has a different shape from the dispatch input and
    # therefore cannot reconstruct a shape-based cache key on its own.
    name, cfg, _ = _get_runtime_config(x, group, handle, None, None, None)
    backend = _get_backend_instance(name)
    backend.init_buffer(group, get_hidden_bytes(x), cfg)
    return backend.combine(
        x,
        handle=handle,
        topk_weights=topk_weights,
        async_finish=async_finish,
        allocate_on_comm_stream=allocate_on_comm_stream,
    )
