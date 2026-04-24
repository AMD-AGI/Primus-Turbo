###############################################################################
# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Modification Copyright© 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import dataclasses
import inspect
import os
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache
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
import triton
import triton.language as tl

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


def _compute_expert_token_info_configs() -> List[triton.Config]:
    """Autotune space for :func:`compute_expert_token_info_kernel`."""
    configs: List[triton.Config] = []
    for block_m in (64, 128, 256, 512, 1024):
        for num_warps in (1, 2, 4, 8):
            # Wave=64, cap threads/CTA to 1024 (HW limit on CDNA).
            if num_warps * 64 > 1024:
                continue
            # Keep at least one element per thread in the M dimension.
            if block_m < num_warps * 64:
                continue
            for num_stages in (1, 2):
                configs.append(
                    triton.Config(
                        {"BLOCK_SIZE_M": block_m},
                        num_warps=num_warps,
                        num_stages=num_stages,
                    )
                )
    return configs


@triton.autotune(
    configs=_compute_expert_token_info_configs(),
    key=["num_tokens", "num_topk"],
    reset_to_zero=["num_recv_tokens_per_expert_ptr"],
)
@triton.jit
def compute_expert_token_info_kernel(
    recv_topk_idx_ptr,
    num_recv_tokens_per_expert_ptr,
    num_tokens: tl.int32,
    num_topk: tl.int32,
    num_local_experts: tl.int32,
    INVALID_VALUE: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """Count per-expert received tokens from a ``[num_tokens, num_topk]`` tile."""
    pid = tl.program_id(0)
    row_offs = pid * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    col_offs = tl.arange(0, BLOCK_SIZE_K)
    bin_offs = tl.arange(0, BLOCK_SIZE_N)

    row_mask = row_offs < num_tokens
    col_mask = col_offs < num_topk
    load_mask = row_mask[:, None] & col_mask[None, :]

    # Clamp OOB rows so the computed offset stays in-range; ``load_mask``
    # guarantees these lanes don't contribute.
    safe_row = tl.where(row_mask, row_offs, 0)
    expert_offs = safe_row[:, None] * num_topk + col_offs[None, :]

    topk_experts = tl.load(
        recv_topk_idx_ptr + expert_offs,
        mask=load_mask,
        other=INVALID_VALUE,
    )

    valid = (
        load_mask & (topk_experts != INVALID_VALUE) & (topk_experts >= 0) & (topk_experts < num_local_experts)
    )
    # Clamp the experts of invalid lanes so they land on a legal bin; the
    # ``mask`` passed to ``tl.histogram`` makes sure they are not counted.
    safe_experts = tl.where(valid, topk_experts, 0)

    # ``tl.histogram`` requires a flat 1D input; reshape the 2D tile.
    flat_experts = tl.reshape(safe_experts, [BLOCK_SIZE_M * BLOCK_SIZE_K])
    flat_mask = tl.reshape(valid, [BLOCK_SIZE_M * BLOCK_SIZE_K])

    # CTA-local 1D accumulator of size BLOCK_SIZE_N, initialised to 0 by
    # ``tl.histogram``; bins ``[num_local_experts, BLOCK_SIZE_N)`` are the
    # power-of-two padding and stay zero because ``safe_experts`` is clamped
    # into ``[0, num_local_experts)`` for valid lanes.
    local_counts = tl.histogram(flat_experts, BLOCK_SIZE_N, mask=flat_mask)

    # Flush CTA-local accumulator to global memory with one atomic per bin.
    bin_mask = (bin_offs < num_local_experts) & (local_counts > 0)
    tl.atomic_add(
        num_recv_tokens_per_expert_ptr + bin_offs,
        local_counts,
        sem="relaxed",
        scope="gpu",
        mask=bin_mask,
    )


def compute_expert_token_info(
    recv_topk_idx: torch.Tensor,
    num_local_experts: int,
    invalid_value: int = -1,
) -> torch.Tensor:
    """Count per-expert received tokens from ``recv_topk_idx``.

    Args:
        recv_topk_idx: ``[num_tokens, num_topk]`` tensor of (local) expert
            indices; entries equal to ``invalid_value`` (or outside
            ``[0, num_local_experts)``) are treated as padding and excluded.
        num_local_experts: Number of local expert bins.
        invalid_value: Sentinel marking padded/invalid slots (default ``-1``).

    Returns:
        ``num_recv_tokens_per_expert`` of shape ``[num_local_experts]`` with
        the same dtype as ``recv_topk_idx``; entry ``e`` holds the count of
        valid ``(token, slot)`` pairs whose expert id equals ``e``.
    """
    assert recv_topk_idx.is_cuda, "recv_topk_idx must be a CUDA tensor"
    assert recv_topk_idx.dim() == 2, "recv_topk_idx must be 2D [num_tokens, num_topk]"

    device = recv_topk_idx.device
    num_tokens, num_topk = recv_topk_idx.shape

    num_recv_tokens_per_expert = torch.zeros(num_local_experts, dtype=recv_topk_idx.dtype, device=device)

    if num_tokens == 0 or num_topk == 0 or num_local_experts == 0:
        return num_recv_tokens_per_expert

    # ``tl.histogram`` requires ``num_bins`` to be a power of 2 on AMD Triton.
    block_size_n = triton.next_power_of_2(num_local_experts)

    # ``BLOCK_SIZE_M`` is picked by the autotuner; the grid size depends on it.
    grid = lambda META: (triton.cdiv(num_tokens, META["BLOCK_SIZE_M"]),)  # noqa: E731
    compute_expert_token_info_kernel[grid](
        recv_topk_idx,
        num_recv_tokens_per_expert,
        num_tokens,
        num_topk,
        num_local_experts,
        INVALID_VALUE=invalid_value,
        BLOCK_SIZE_K=triton.next_power_of_2(num_topk),
        BLOCK_SIZE_N=block_size_n,
    )
    return num_recv_tokens_per_expert


@dataclass
class EPBufferConfig:
    """Configuration for EP communication buffer initialization.

    Attributes:
        num_sms: Number of SMs used by high-throughput kernels.
        dispatch_config: Dispatch config; ``None`` means use backend default.
        combine_config: Combine config; ``None`` means use backend default.
    """

    num_sms: int = 32
    dispatch_config: Any = None
    combine_config: Any = None


# Per-backend default buffer configuration. Keys must match the names used
# in ``_BACKEND_REGISTRY`` so lookups can be done by backend name.
_DEFAULT_BUFFER_CONFIG_PER_BACKEND: Dict[str, EPBufferConfig] = {
    "TURBO": EPBufferConfig(
        num_sms=64,
        dispatch_config=None,
        combine_config=None,
    ),
    "DEEP_EP": EPBufferConfig(
        num_sms=64,
        dispatch_config=None,
        combine_config=None,
    ),
    "MORI": EPBufferConfig(num_sms=64, dispatch_config=None, combine_config=None),
}


# Deep copy of the defaults so mutations via ``set_buffer_global_config``
# don't leak into ``_DEFAULT_BUFFER_CONFIG_PER_BACKEND``.
_buffer_config_per_backend: Dict[str, EPBufferConfig] = {
    name: dataclasses.replace(cfg) for name, cfg in _DEFAULT_BUFFER_CONFIG_PER_BACKEND.items()
}


def _get_buffer_config(backend_name: str) -> EPBufferConfig:
    """Return the runtime buffer config for ``backend_name``, or a default."""
    cfg = _buffer_config_per_backend.get(backend_name)
    if cfg is not None:
        return cfg
    default = _DEFAULT_BUFFER_CONFIG_PER_BACKEND.get(backend_name)
    return dataclasses.replace(default) if default is not None else EPBufferConfig()


def set_buffer_global_config(
    num_use_cu: int = 32,
    autotune_config: Optional[tuple] = None,
    backend: Optional[str] = None,
) -> None:
    """Set the EP buffer config for a backend, or for all backends when
    ``backend`` is ``None`` (backward-compat). Accepts SM count and an
    optional ``(dispatch_config, combine_config)`` tuple.
    """
    dispatch_cfg, combine_cfg = autotune_config if autotune_config is not None else (None, None)
    new_cfg = EPBufferConfig(
        num_sms=num_use_cu,
        dispatch_config=dispatch_cfg,
        combine_config=combine_cfg,
    )
    if backend is None:
        targets = set(_buffer_config_per_backend.keys()) | set(_BACKEND_REGISTRY.keys())
        for name in targets:
            _buffer_config_per_backend[name] = dataclasses.replace(new_cfg)
    else:
        _buffer_config_per_backend[backend] = new_cfg


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

    def is_initialized(self) -> bool:
        """Return True if the backend is initialized."""
        ...

    def init_buffer(
        self,
        group: dist.ProcessGroup,
        hidden_size: int,
        num_experts: int,
        num_topk: int,
        seqlen: int,
        fp8_dispatch: bool,
        config: EPBufferConfig,
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
    def _get_module():
        """Return the backend Python module exposing ``Buffer``/``Config``/events."""
        raise NotImplementedError

    def _make_buffer_kwargs(self, group: dist.ProcessGroup) -> dict:
        """Extra kwargs forwarded to ``BufferClass(group, nvl, rdma, **kwargs)``."""
        return {}

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
                f"num_rdma_bytes={num_rdma_bytes} "
                f"dispatch_config={dispatch_config} combine_config={combine_config}",
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

        # Buffer sizes (ported from uccl-ep).
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
            return {"is_intranode": group.size() <= 8}
        return {}


# ==========================================================================
# Mori EP backend helpers
# ==========================================================================
_ENV_MORI_NUM_QP_PER_PE = "PRIMUS_TURBO_MORI_NUM_QP_PER_PE"


class _MoriEpMode(Enum):
    """Mori EP deployment mode (normal dispatch only)."""

    INTRA_NODE = "intra_node"
    INTER_NODE = "inter_node"


@dataclass(frozen=True)
class _MoriDispatchCfg:
    """Mori kernel launch config for a given mode / token budget."""

    kernel_type: Any  # mori.ops.EpDispatchCombineKernelType
    warp_num_per_block: int
    block_num: int
    rdma_block_num: int


def _get_mori_dispatch_configs(
    num_max_dispatch_tokens_per_rank: int,
) -> Dict[_MoriEpMode, _MoriDispatchCfg]:
    """Return per-mode Mori kernel launch configs."""
    import mori.ops

    return {
        _MoriEpMode.INTRA_NODE: _MoriDispatchCfg(
            kernel_type=mori.ops.EpDispatchCombineKernelType.IntraNode,
            warp_num_per_block=16,
            block_num=80,
            rdma_block_num=0,
        ),
        _MoriEpMode.INTER_NODE: _MoriDispatchCfg(
            kernel_type=mori.ops.EpDispatchCombineKernelType.InterNodeV1,
            warp_num_per_block=8,
            block_num=64,
            rdma_block_num=32,
        ),
    }


_MORI_SHMEM_PG_NAME = "mori"


def _register_and_init_mori_shmem(group: dist.ProcessGroup) -> None:
    """Register ``group`` with the Mori SHMEM runtime."""
    import mori.shmem

    assert dist.is_initialized(), "torch.distributed must be initialized before Mori SHMEM init."

    try:
        torch._C._distributed_c10d._register_process_group(_MORI_SHMEM_PG_NAME, group)
    except Exception as e:  # noqa: BLE001 - mori binds raise a mix of exception types.
        if "already registered" in str(e):
            logger.info(
                f"[MORI init] Process group already registered under "
                f"'{_MORI_SHMEM_PG_NAME}'; reusing existing SHMEM binding "
                f"({e}).",
                rank=0,
            )
        else:
            raise
    else:
        mori.shmem.shmem_torch_process_group_init(_MORI_SHMEM_PG_NAME)


def _resolve_mori_dispatch_cfg(
    group: dist.ProcessGroup,
    num_max_dispatch_tokens_per_rank: int,
    config: Optional[EPBufferConfig] = None,
) -> _MoriDispatchCfg:
    """Resolve Mori kernel launch cfg: topology default, overridden by
    ``config.num_sms`` (``block_num``) and ``config.dispatch_config`` (full).
    """
    _, num_nodes = detect_group_topology(group)
    mode = _MoriEpMode.INTRA_NODE if num_nodes <= 1 else _MoriEpMode.INTER_NODE
    base = _get_mori_dispatch_configs(num_max_dispatch_tokens_per_rank)[mode]

    kernel_type = base.kernel_type
    warp_num_per_block = base.warp_num_per_block
    block_num = base.block_num
    rdma_block_num = base.rdma_block_num

    num_sms_override = config.num_sms if config is not None else 0
    full_override = (
        config.dispatch_config
        if config is not None and isinstance(config.dispatch_config, _MoriDispatchCfg)
        else None
    )

    if num_sms_override > 0:
        block_num = int(num_sms_override)

    if full_override is not None:
        kernel_type = full_override.kernel_type
        warp_num_per_block = full_override.warp_num_per_block
        block_num = full_override.block_num
        rdma_block_num = full_override.rdma_block_num

    return _MoriDispatchCfg(
        kernel_type=kernel_type,
        warp_num_per_block=warp_num_per_block,
        block_num=block_num,
        rdma_block_num=rdma_block_num,
    )


@lru_cache(maxsize=8)
def _build_mori_op(
    rank: int,
    world_size: int,
    hidden: int,
    params_dtype: torch.dtype,
    num_local_experts: int,
    num_topk: int,
    num_max_dispatch_tokens_per_rank: int,
    kernel_type: Any,
    warp_num_per_block: int,
    block_num: int,
    rdma_block_num: int,
):
    """Build and cache a :class:`mori.ops.EpDispatchCombineOp`."""
    import mori.ops

    num_qp_per_pe = int(os.environ.get(_ENV_MORI_NUM_QP_PER_PE, "2"))

    common_kwargs = dict(
        data_type=params_dtype,
        rank=rank,
        world_size=world_size,
        hidden_dim=hidden,
        scale_dim=0,
        scale_type_size=0,
        max_token_type_size=max(2, params_dtype.itemsize),
        max_num_inp_token_per_rank=num_max_dispatch_tokens_per_rank,
        num_experts_per_rank=num_local_experts,
        num_experts_per_token=num_topk,
        warp_num_per_block=warp_num_per_block,
        block_num=block_num,
        max_total_recv_tokens=0,
        use_external_inp_buf=True,
        kernel_type=kernel_type,
        gpu_per_node=torch.cuda.device_count(),
        rdma_block_num=rdma_block_num,
        num_qp_per_pe=num_qp_per_pe,
        quant_type="none",
    )

    def _check_mori_compatibility(kwargs: dict) -> dict:
        """Drop kwargs not accepted by the installed Mori config."""
        valid = {f.name for f in dataclasses.fields(mori.ops.EpDispatchCombineConfig)}
        cleaned = dict(kwargs)
        for key in list(cleaned.keys()):
            if key not in valid:
                logger.warning(
                    f"[MORI compat] Dropping incompatible EpDispatchCombineConfig " f"argument '{key}'.",
                    once=True,
                )
                del cleaned[key]
        return cleaned

    common_kwargs = _check_mori_compatibility(common_kwargs)

    logger.info(
        f"[MORI init] world_size={world_size} rank={rank} hidden={hidden} "
        f"dtype={params_dtype} max_inp_tokens={num_max_dispatch_tokens_per_rank} "
        f"num_local_experts={num_local_experts} num_topk={num_topk} "
        f"kernel={kernel_type} block_num={block_num} "
        f"warp_num_per_block={warp_num_per_block} rdma_block_num={rdma_block_num}",
        rank=0,
    )

    config = mori.ops.EpDispatchCombineConfig(**common_kwargs)
    return mori.ops.EpDispatchCombineOp(config)


class MoriEPBackend:
    """ROCm Mori EP backend for MoE dispatch/combine (normal mode)."""

    def __init__(self) -> None:
        self._group: Optional[dist.ProcessGroup] = None
        self._op = None
        self._hidden_size: int = 0
        self._num_local_experts: int = 0
        self._num_topk: int = 0
        self._seqlen: int = 0
        self._fp8_dispatch: bool = False
        self._params_dtype: Optional[torch.dtype] = None
        # Last resolved kernel cfg; used to detect cfg-only rebuilds.
        self._kernel_cfg: Optional[_MoriDispatchCfg] = None

    # ------------------------------------------------------------------
    # EPBackend protocol
    # ------------------------------------------------------------------

    def is_initialized(self) -> bool:
        """Return True if the backend is initialized."""
        return self._op is not None

    @staticmethod
    def is_available() -> bool:
        try:
            import mori  # noqa: F401
            import mori.ops  # noqa: F401
            import mori.shmem  # noqa: F401

            return True
        except ImportError:
            return False

    @staticmethod
    def _derive_params_dtype(fp8_dispatch: bool) -> torch.dtype:
        """Pick the Mori ``data_type`` argument from ``fp8_dispatch``."""
        assert not fp8_dispatch, "Not implemented"
        return torch.bfloat16

    def init_buffer(
        self,
        group: dist.ProcessGroup,
        hidden_size: int,
        num_experts: int,
        num_topk: int,
        seqlen: int,
        fp8_dispatch: bool,
        config: EPBufferConfig,
    ) -> None:
        """Register Mori SHMEM and build/rebuild the Mori op as needed.

        Rebuilds when any shape signature field or kernel cfg changes. Reads
        ``config.num_sms`` as a ``block_num`` override and
        ``config.dispatch_config`` (if a :class:`_MoriDispatchCfg`) as a full
        kernel cfg override; see :func:`_resolve_mori_dispatch_cfg`.
        """
        assert not fp8_dispatch, "not implemented"

        if self._group is not group:
            self._group = group
            _register_and_init_mori_shmem(group)
            logger.info("[MoriEPBackend init] Mori SHMEM initialized.", rank=0)

        num_local_experts = num_experts // group.size()
        params_dtype = self._derive_params_dtype(fp8_dispatch)
        kernel_cfg = _resolve_mori_dispatch_cfg(group, seqlen, config)

        # Rebuild on shape-signature or kernel-cfg change (_build_mori_op is cached).
        needs_rebuild = (
            self._op is None
            or self._hidden_size != hidden_size
            or self._num_local_experts != num_local_experts
            or self._num_topk != num_topk
            or self._seqlen != seqlen
            or self._fp8_dispatch != fp8_dispatch
            or self._params_dtype != params_dtype
            or self._kernel_cfg != kernel_cfg
        )

        self._hidden_size = hidden_size
        self._num_local_experts = num_local_experts
        self._num_topk = num_topk
        self._seqlen = seqlen
        self._fp8_dispatch = fp8_dispatch
        self._params_dtype = params_dtype
        self._kernel_cfg = kernel_cfg

        if needs_rebuild:
            self._op = _build_mori_op(
                rank=group.rank(),
                world_size=group.size(),
                hidden=hidden_size,
                params_dtype=params_dtype,
                num_local_experts=num_local_experts,
                num_topk=num_topk,
                num_max_dispatch_tokens_per_rank=seqlen,
                kernel_type=kernel_cfg.kernel_type,
                warp_num_per_block=kernel_cfg.warp_num_per_block,
                block_num=kernel_cfg.block_num,
                rdma_block_num=kernel_cfg.rdma_block_num,
            )

    def dispatch(
        self,
        x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        handle: Optional[tuple] = None,
        topk_idx: Optional[torch.Tensor] = None,
        token_weights: Optional[torch.Tensor] = None,
        num_experts: Optional[int] = None,
        async_finish: bool = False,  # noqa: ARG002 - API compat only.
        allocate_on_comm_stream: bool = False,  # noqa: ARG002 - API compat only.
        num_worst_tokens: int = 0,  # noqa: ARG002 - API compat only.
        config: Optional[Any] = None,  # noqa: ARG002 - API compat only.
    ):
        assert self.is_initialized(), "Backend is not initialized"
        scale = None
        non_blocking = num_worst_tokens > 0

        if handle is None:
            assert topk_idx is not None
            recv_topk_idx_i32 = topk_idx.to(torch.int32)
        else:
            (recv_topk_idx_i32,) = handle

        recv_x, recv_topk_weights, recv_x_scales, recv_topk_idx, _ = self._op.dispatch(
            x,
            token_weights,
            scale,
            recv_topk_idx_i32,
        )
        num_recv_tokens_per_expert = compute_expert_token_info(
            recv_topk_idx, self._num_local_experts, invalid_value=self._num_local_experts
        )
        if not non_blocking:
            num_recv_tokens_per_expert = num_recv_tokens_per_expert.tolist()
        return recv_x, recv_topk_idx, recv_topk_weights, num_recv_tokens_per_expert, (recv_topk_idx_i32,)

    def combine(
        self,
        x: torch.Tensor,
        handle: tuple,
        topk_weights: Optional[torch.Tensor] = None,
        async_finish: bool = False,  # noqa: ARG002 - API compat only.
        allocate_on_comm_stream: bool = False,  # noqa: ARG002 - API compat only.
        config: Optional[Any] = None,  # noqa: ARG002 - API compat only.
    ):
        assert self.is_initialized(), "Backend is not initialized"

        (topk_idx_i32,) = handle

        combined_x, combined_topk_weights = self._op.combine(
            x.contiguous(),
            topk_weights,
            topk_idx_i32,
        )
        return combined_x, combined_topk_weights

    @torch.no_grad()
    def tune_configs(
        self,
        group: dist.ProcessGroup,  # noqa: ARG002 - stub.
        x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],  # noqa: ARG002
        num_experts: int,  # noqa: ARG002
        *,
        topk_idx: Optional[torch.Tensor] = None,  # noqa: ARG002
        topk_weights: Optional[torch.Tensor] = None,  # noqa: ARG002
        num_sms: int = 32,  # noqa: ARG002
        num_tests: int = 20,  # noqa: ARG002
        num_topk: Optional[int] = None,  # noqa: ARG002
        uniform_dispatch: bool = True,  # noqa: ARG002
    ) -> Tuple[Any, Any, float, float]:
        raise NotImplementedError("tune_configs is not implemented for MoriEPBackend")

    def release_buffer(self) -> None:
        """Drop the fast-path op reference and SHMEM binding state."""
        self._op = None
        self._hidden_size = 0
        self._num_local_experts = 0
        self._num_topk = 0
        self._seqlen = 0
        self._fp8_dispatch = False
        self._params_dtype = None
        self._kernel_cfg = None


# =========================================================================
# Backend registry
# =========================================================================

_BACKEND_REGISTRY: Dict[str, Type[EPBackend]] = {
    "TURBO": TurboEPBackend,
    "DEEP_EP": DeepEPBackend,
    "MORI": MoriEPBackend,
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
    """Result of :class:`MoEDispatchCombineAutoTuner.tune`.

    Attributes:
        backend_name: Winning backend registry name.
        dispatch_config: Best dispatch ``Config``.
        combine_config: Best combine ``Config``.
        dispatch_time_us: Dispatch latency of the winner (us).
        combine_time_us: Combine latency of the winner (us).
        per_backend: ``backend_name -> (d_time_us, c_time_us)`` of every
            candidate that completed tuning.
        tune_duration_s: Wall-clock seconds spent tuning (rank 0 clock).
    """

    backend_name: str
    dispatch_config: Any
    combine_config: Any
    dispatch_time_us: float
    combine_time_us: float
    per_backend: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    tune_duration_s: float = 0.0


class MoEDispatchCombineAutoTuner:
    """Autotuner that picks the best *(backend, dispatch_config, combine_config)*
    for a given MoE dispatch / combine case.

    Tuning is driven by ``moe_dispatch`` on a shape-based key; the resulting
    :class:`EPAutoTuneResult` is bound to the returned ``handle`` so that the
    paired ``moe_combine`` / cached-``moe_dispatch`` / autograd backward can
    reuse it without any further lookup.

    Enable via ``PRIMUS_TURBO_AUTO_TUNE=1`` or by calling
    :meth:`get_or_tune` explicitly.
    """

    # Shape-based cache shared across layers with identical input shape.
    _cache: TuneCache = TuneCache(capacity=1024)

    _HANDLE_CACHE_MAX: int = 1024
    _handle_cache: "OrderedDict[int, Tuple[Any, EPAutoTuneResult]]" = OrderedDict()

    # Fallback for calls made before any handle mapping is registered.
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
        """Bind ``result`` to ``handle`` for paired-call reuse.

        Args:
            handle: Handle returned by the backend's ``dispatch`` call.
            result: Tune result to associate with the handle.
        """
        if handle is None:
            return
        hid = id(handle)
        cache = cls._handle_cache
        if hid in cache:
            cache.move_to_end(hid)
        cache[hid] = (handle, result)
        if len(cache) > cls._HANDLE_CACHE_MAX:
            cache.popitem(last=False)

    @classmethod
    def lookup_handle(cls, handle: Any) -> Optional[EPAutoTuneResult]:
        """Return the result bound to ``handle``, or ``None`` if unknown."""
        if handle is None:
            return None
        entry = cls._handle_cache.get(id(handle))
        if entry is None:
            return None
        stored_handle, res = entry
        if stored_handle is not handle:
            cls._handle_cache.pop(id(handle), None)
            return None
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
        """Tune across candidate backends + configs and return the best.

        Args:
            group: EP process group.
            x: Dispatch input (Tensor or ``(fp8_tensor, scales)`` tuple).
            num_experts: Total number of experts.
            topk_idx: ``[num_tokens, num_topk]`` expert indices (optional in
                uniform mode).
            topk_weights: ``[num_tokens, num_topk]`` expert weights (optional
                in uniform mode).
            num_sms: SM count to use for tuning.
            candidate_backends: Optional explicit backend name list.
            num_tests: Timed iterations per candidate.
            num_topk: Number of experts per token (uniform-mode fallback).
            uniform_dispatch: If True, resample topk internally for robustness.

        Returns:
            Best :class:`EPAutoTuneResult`. Not cached; use :meth:`get_or_tune`.
        """
        names = list(candidate_backends) if candidate_backends is not None else cls._candidate_backend_names()
        if not names:
            raise RuntimeError("MoE autotune: no EP backends are available.")

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
            except NotImplementedError:
                # Backend opts out of config tuning (e.g. Mori has no
                # user-tunable Config sweep space). Skip quietly.
                logger.info(
                    f"MoE autotune: backend '{name}' does not support " f"tune_configs; skipping.",
                    once=True,
                    rank=0,
                )
                continue
            except Exception as exc:  # pragma: no cover - defensive
                # Use ``repr`` so pybind-backed errors aren't silently blank.
                logger.warning(
                    f"MoE autotune: backend '{name}' failed: \n" f"{type(exc).__name__}: {exc!r}",
                    exc_info=True,
                )
                continue
            finally:
                # Drop the worst-case buffer; next init_buffer sizes correctly.
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

        Args:
            group: EP process group.
            x: Dispatch input (Tensor or ``(fp8_tensor, scales)`` tuple).
            num_experts: Total number of experts.
            topk_idx: Optional ``[num_tokens, num_topk]`` expert indices.
            topk_weights: Optional ``[num_tokens, num_topk]`` expert weights.
            num_sms: SM count to use for tuning.
            num_topk: Number of experts per token. Also used in the shape
                cache key, so the same signature reuses the cached winner.
            uniform_dispatch: If True, resample topk internally for robustness.
            **tune_kwargs: Forwarded to :meth:`tune`.
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
        """Return the most recent tune result, or ``None`` if never tuned."""
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
    """Return the user-selected backend name, or ``TURBO`` by default."""
    bt = GlobalBackendManager.get_moe_dispatch_combine_backend(PrecisionType.BF16_FP16_FP32)
    return bt.name if bt is not None else _DEFAULT_BACKEND_NAME


# =========================================================================
# Utilities
# =========================================================================


def _maybe_tune_config(
    x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
    group: dist.ProcessGroup,
    handle: Optional[tuple],
    topk_idx: Optional[torch.Tensor],
    token_weights: Optional[torch.Tensor],
    num_experts: Optional[int],
) -> Tuple[str, EPBufferConfig, Optional[EPAutoTuneResult]]:
    """Tune the config for the given shape or get default config."""
    backend_name = _get_backend_name()
    user_cfg = _get_buffer_config(backend_name)

    # Autotune off: use the user's backend + per-backend config.
    if not GlobalBackendManager.auto_tune_enabled():
        return backend_name, user_cfg, None

    # Autotune on: user's backend choice is ignored when we have a tuned
    # result to fall back to.
    result: Optional[EPAutoTuneResult] = None

    if handle is not None:
        result = MoEDispatchCombineAutoTuner.lookup_handle(handle)
    else:
        if topk_idx is not None and token_weights is not None and num_experts is not None:
            result = MoEDispatchCombineAutoTuner.get_or_tune(
                group=group,
                x=x,
                num_experts=int(num_experts),
                topk_idx=topk_idx,
                topk_weights=token_weights,
                num_sms=user_cfg.num_sms,
            )

        if result is None:
            result = MoEDispatchCombineAutoTuner.current()

    if result is not None:
        winner_cfg = _get_buffer_config(result.backend_name)
        cfg = EPBufferConfig(
            num_sms=winner_cfg.num_sms,
            dispatch_config=result.dispatch_config,
            combine_config=result.combine_config,
        )
        return result.backend_name, cfg, result

    # Nothing cached yet - fall back to user/default.
    return backend_name, user_cfg, None


# =========================================================================
# Public API - used by ``moe_dispatch_combine.py``
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
    """Dispatch tokens to experts via the selected EP backend.

    Args:
        x: Input tensor, or ``(fp8_tensor, scales)`` tuple for FP8 path.
        group: EP process group.
        handle: Cached dispatch handle; ``None`` for a primary call.
        topk_idx: ``[num_tokens, num_topk]`` expert indices (primary only).
        token_weights: ``[num_tokens, num_topk]`` expert weights.
        num_experts: Total number of experts (primary only).
        async_finish: If True, return before the kernel finishes.
        allocate_on_comm_stream: Allocate outputs on the comm stream.
        num_worst_tokens: Worst-case receive token count (for padding).

    Returns:
        ``(recv_x, recv_topk_idx, recv_topk_weights, tokens_per_expert, handle)``.
    """
    name, cfg, result = _maybe_tune_config(x, group, handle, topk_idx, token_weights, num_experts)
    backend = _get_backend_instance(name)
    if num_experts is None or topk_idx is None:
        assert backend.is_initialized(), "Backend is not initialized"
    else:
        # Unpack FP8 ``(tensor, scales)`` tuple to get shape metadata.
        fp8_dispatch = isinstance(x, tuple)
        x_inp = x[0] if fp8_dispatch else x
        backend.init_buffer(
            group,
            hidden_size=x_inp.size(1),
            num_experts=num_experts,
            num_topk=topk_idx.size(1),
            seqlen=x_inp.size(0),
            fp8_dispatch=fp8_dispatch,
            config=cfg,
        )
    outputs = backend.dispatch(
        x,
        handle=handle,
        topk_idx=topk_idx,
        token_weights=token_weights,
        num_experts=num_experts,
        async_finish=async_finish,
        allocate_on_comm_stream=allocate_on_comm_stream,
        num_worst_tokens=num_worst_tokens,
        config=cfg.dispatch_config,
    )

    # Bind the autotune result to the new handle for paired-call reuse.
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
    """Combine expert outputs back to original token order via the selected backend.

    Args:
        x: Expert-side tensor to be combined.
        group: EP process group.
        handle: Handle returned by the paired ``moe_dispatch_impl`` call.
        topk_weights: Optional per-token expert weights.
        async_finish: If True, return before the kernel finishes.
        allocate_on_comm_stream: Allocate outputs on the comm stream.

    Returns:
        ``(combined_x, combined_topk_weights)``.
    """
    name, cfg, _ = _maybe_tune_config(x, group, handle, None, None, None)
    backend = _get_backend_instance(name)
    assert backend.is_initialized(), "Backend is not initialized"
    return backend.combine(
        x,
        handle=handle,
        topk_weights=topk_weights,
        async_finish=async_finish,
        allocate_on_comm_stream=allocate_on_comm_stream,
        config=cfg.combine_config,
    )
