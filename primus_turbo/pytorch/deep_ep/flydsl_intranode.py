###############################################################################
# Copyright (c) 2025 DeepSeek. All rights reserved.
#
# Modification Copyright© 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# The host flow mirrors Buffer::get_dispatch_layout / intranode_dispatch / intranode_combine
# in csrc/pytorch/deep_ep/deep_ep.cpp, ported from DeepEP (https://github.com/deepseek-ai/DeepEP).
#
# See LICENSE for license information.
###############################################################################

"""FlyDSL backend of DeepEP's intranode path (``num_rdma_ranks == 1``).

A runtime exposes the three intranode methods of the C++ ``Buffer`` runtime with the same arguments
and return tuples, so ``Buffer`` can route to either, and owns its own uncached HIP-IPC heap.
``FlyDSLIntranodeRuntime`` holds what both versions share; each version is a subclass:

- ``FlyDSLIntranodeV1Runtime`` (backend ``flydsl``) runs the V1 kernels of ``intranode_v1_kernel``,
  the channel rings of the HIP kernels; its handles are plain tensors, interchangeable with the C++
  backend's.
- ``FlyDSLIntranodeV2Runtime`` (backend ``flydsl_v2``) runs the V2 kernels of ``intranode_v2_kernel``:
  direct writes into fixed slots instead of channel rings. Its outputs match V1's; its handles are its
  own (``send_head`` holds stream indices, and a channel is one block, ``num_sms`` of them).
"""

import contextlib
from typing import NamedTuple, Optional

import torch
import torch.distributed as dist

import primus_turbo.common.envs as envs
from primus_turbo.flydsl.deep_ep.arch import CPP_DEEPEP_ARCHS, default_intranode_backend

# FlyDSLIntranodeV1Runtime only
from primus_turbo.flydsl.deep_ep.intranode_v1_kernel import (
    combine_buffer_layout,
    combine_min_recv_buffer_tokens,
    dispatch_buffer_layout,
    launch_cached_notify_combine,
    launch_combine,
    launch_dispatch,
)

# FlyDSLIntranodeV2Runtime and v2_heap_bytes only
from primus_turbo.flydsl.deep_ep.intranode_v2_kernel import (
    COMBINE_FLAGS_BYTES,
    DATA_OFFSET,
    MAX_EPOCH,
    META_BYTES,
    TAIL_DONE_BIT,
    combine_v2_grabs,
    combine_v2_layout,
    combine_v2_slot_bytes,
    dispatch_v2_layout,
    launch_barrier,
    launch_combine_v2,
    launch_dispatch_v2,
)
from primus_turbo.flydsl.deep_ep.launch_cache import run_launcher
from primus_turbo.flydsl.deep_ep.layout_kernel import get_dispatch_layout_flydsl
from primus_turbo.flydsl.deep_ep.notify_kernel import launch_cached_notify_dispatch, launch_notify_dispatch
from primus_turbo.pytorch.core.symm_mem import SymmetricMemory

from ._compat import NUM_MAX_NVL_PEERS, EventHandle

NUM_MAX_LOCAL_EXPERTS = 1024
SUPPORTED_NUM_RANKS = (2, 4, 8)
INTRANODE_BACKENDS = ("cpp", "flydsl", "flydsl_v2")


def _as_i32(value: int) -> int:
    """``value`` wrapped to a signed 32-bit kernel argument."""
    value &= 0xFFFFFFFF
    return value - (1 << 32) if value >= (1 << 31) else value


def get_intranode_backend(backend: Optional[str] = None) -> str:
    """``backend`` (``cpp`` | ``flydsl`` | ``flydsl_v2``), else ``default_intranode_backend()``; either
    way an unsupported GPU raises."""
    default_backend = default_intranode_backend()
    backend = (backend or default_backend).strip().lower()
    if backend not in INTRANODE_BACKENDS:
        raise ValueError(
            f"Unknown DeepEP intranode backend {backend!r}; expected one of {INTRANODE_BACKENDS}"
        )
    return backend


def _check(cond: bool, msg: str = "") -> None:
    if not cond:
        raise RuntimeError(f"DeepEP FlyDSL intranode check failed: {msg}")


class _DispatchArgs(NamedTuple):
    """What a dispatch launch takes, once the shared part of ``intranode_dispatch`` has run."""

    x: torch.Tensor
    x_scales: Optional[torch.Tensor]
    topk_idx: Optional[torch.Tensor]
    topk_weights: Optional[torch.Tensor]
    is_token_in_rank_u8: torch.Tensor
    rank_prefix_matrix: torch.Tensor
    channel_prefix_matrix: torch.Tensor
    recv_x: torch.Tensor
    recv_x_scales: Optional[torch.Tensor]
    recv_src_idx: torch.Tensor
    recv_topk_idx: Optional[torch.Tensor]
    recv_topk_weights: Optional[torch.Tensor]
    recv_channel_prefix_matrix: torch.Tensor
    send_head: torch.Tensor
    num_tokens: int
    num_recv_tokens: int
    num_worst_tokens: int
    num_channels: int
    hidden_bytes: int
    num_topk: int
    num_experts: int
    num_scales: int
    scale_token_stride: int
    scale_hidden_stride: int
    topk_idx_bytes: int
    config: object


class _CombineArgs(NamedTuple):
    """What a combine launch takes, once the shared part of ``intranode_combine`` has run."""

    x: torch.Tensor
    topk_weights: Optional[torch.Tensor]
    bias_0: Optional[torch.Tensor]
    bias_1: Optional[torch.Tensor]
    src_idx: torch.Tensor
    rank_prefix_matrix: torch.Tensor
    channel_prefix_matrix: torch.Tensor
    send_head: torch.Tensor
    recv_x: torch.Tensor
    recv_topk_weights: Optional[torch.Tensor]
    num_tokens: int
    num_recv_tokens: int
    num_channels: int
    hidden: int
    hidden_bytes: int
    num_topk: int
    config: object


class FlyDSLIntranodeRuntime:
    """Intranode dispatch / combine on FlyDSL kernels over a symmetric uncached heap: what V1
    (``FlyDSLIntranodeV1Runtime``) and V2 (``FlyDSLIntranodeV2Runtime``) share -- the heap, the
    streams, the layout, the checks of the arguments, the notify and the outputs. A subclass sets the
    channel count and what the notify clears, and launches its own dispatch and combine."""

    # notify_dispatch fills every column of the rank prefix matrix, not only this rank's
    FULL_RANK_MATRIX = False

    def __init__(self, group: dist.ProcessGroup, num_nvl_bytes: int, comm_stream: torch.cuda.Stream) -> None:
        self.group = group
        self.rank = group.rank()
        self.num_ranks = group.size()
        _check(
            self.num_ranks in SUPPORTED_NUM_RANKS,
            f"num_ranks must be one of {SUPPORTED_NUM_RANKS}, got {self.num_ranks}",
        )
        _check(num_nvl_bytes > 0, "the FlyDSL intranode backend needs num_nvl_bytes > 0")
        self.num_nvl_bytes = num_nvl_bytes
        self.comm_stream = comm_stream
        self.force_current_stream = envs.PRIMUS_TURBO_EP_FORCE_CURRENT_STREAM

        # The channel protocol leans on every heap access bypassing L2, as the C++
        # buffer's hipDeviceMallocUncached does; the barrier signals likewise.
        self.symm = SymmetricMemory(
            group,
            num_nvl_bytes,
            signal_pad_size=NUM_MAX_NVL_PEERS * 4,
            uncached_signal_pad=True,
            uncached_buffer=True,
        )
        self.buffer_ptrs = self.symm.buffer_ptrs_dev
        self.barrier_ptrs = self.symm.signal_pad_ptrs_dev
        # notify's receive counts land here: [num_recv_tokens, per local expert...]
        self.host_counters = torch.empty(1 + NUM_MAX_LOCAL_EXPERTS, dtype=torch.int32, pin_memory=True)
        self.counters_ready = torch.cuda.Event()
        self._init_state()
        torch.cuda.synchronize()
        group.barrier()

    # ── what a version sets ──────────────────────────────────────────────────

    def _init_state(self) -> None:
        """The version's own state and heap contents, set before every rank meets in ``__init__``."""

    def _num_channels(self, config) -> int:
        """The channels of a dispatch / combine with ``config``; raises where the version cannot run it."""
        raise NotImplementedError

    def _notify_memset_ints(self, num_channels: int) -> int:
        """The int32 words after the rank prefix matrix the notify clears for the dispatch."""
        raise NotImplementedError

    def _launch_dispatch(self, args: _DispatchArgs, launch_stream: torch.cuda.Stream) -> None:
        raise NotImplementedError

    def _check_combine_config(self, config) -> None:
        """Raises for a ``config`` the version's combine cannot run."""

    def _launch_combine(self, args: _CombineArgs, launch_stream: torch.cuda.Stream) -> None:
        raise NotImplementedError

    # ── heap ─────────────────────────────────────────────────────────────────

    def destroy(self) -> None:
        """Collective: every rank of the group calls it."""
        if self.symm is not None:
            torch.cuda.synchronize()
            # a peer's kernels may still write into this heap (the V1 receivers' queue head updates):
            # free it only once every rank's kernels are done
            if dist.is_initialized():
                dist.barrier(group=self.group)
            self.symm.destroy()
            self.symm = None

    def get_local_buffer_tensor(self, dtype: torch.dtype, offset: int) -> torch.Tensor:
        return self.symm.get_buffer(
            self.rank, ((self.num_nvl_bytes - offset * dtype.itemsize) // dtype.itemsize,), dtype, offset
        )

    # ── stream plumbing (Buffer::maybe_fork_stream / maybe_join_stream) ──────

    def _fork(self, previous_event, async_finish: bool, allocate_on_comm_stream: bool):
        compute_stream = torch.cuda.current_stream()
        if self.force_current_stream:
            launch_stream = compute_stream
        else:
            launch_stream = self.comm_stream
            if previous_event is not None:
                with torch.cuda.stream(launch_stream):
                    previous_event.current_stream_wait()
            else:
                launch_stream.wait_stream(compute_stream)
        cross_stream_allocation = allocate_on_comm_stream and launch_stream != compute_stream
        if cross_stream_allocation:
            _check(
                previous_event is not None and async_finish,
                "allocate_on_comm_stream needs previous_event + async",
            )
        return compute_stream, launch_stream, cross_stream_allocation

    @staticmethod
    def _finish(compute_stream, launch_stream, cross_stream_allocation, async_finish, tensors):
        if not async_finish:
            if launch_stream != compute_stream:
                compute_stream.wait_stream(launch_stream)
            return None
        with torch.cuda.stream(launch_stream):
            event = EventHandle()
        for t in tensors:
            if t is not None:
                t.record_stream(launch_stream)
                if cross_stream_allocation:
                    t.record_stream(compute_stream)
        return event

    @staticmethod
    def _alloc_stream(launch_stream, cross_stream_allocation):
        # allocations follow the launch stream only for allocate_on_comm_stream
        return torch.cuda.stream(launch_stream) if cross_stream_allocation else contextlib.nullcontext()

    # ── layout ───────────────────────────────────────────────────────────────

    def get_dispatch_layout(
        self, topk_idx, num_experts, previous_event, async_finish, allocate_on_comm_stream
    ):
        _check(topk_idx.dim() == 2 and topk_idx.is_contiguous(), "topk_idx must be a contiguous 2D tensor")
        _check(num_experts > 0, "num_experts must be positive")
        compute_stream, launch_stream, cross = self._fork(
            previous_event, async_finish, allocate_on_comm_stream
        )
        with self._alloc_stream(launch_stream, cross), torch.cuda.stream(launch_stream):
            num_tokens_per_rank, num_tokens_per_expert, is_token_in_rank = get_dispatch_layout_flydsl(
                topk_idx, self.num_ranks, num_experts
            )
        event = self._finish(
            compute_stream,
            launch_stream,
            cross,
            async_finish,
            (topk_idx, num_tokens_per_rank, num_tokens_per_expert, is_token_in_rank),
        )
        return num_tokens_per_rank, None, num_tokens_per_expert, is_token_in_rank, event

    # ── dispatch ─────────────────────────────────────────────────────────────

    def intranode_dispatch(
        self,
        x,
        x_scales,
        topk_idx,
        topk_weights,
        num_tokens_per_rank,
        is_token_in_rank,
        num_tokens_per_expert,
        cached_num_recv_tokens,
        cached_rank_prefix_matrix,
        cached_channel_prefix_matrix,
        expert_alignment,
        num_worst_tokens,
        config,
        previous_event,
        async_finish,
        allocate_on_comm_stream,
    ):
        num_ranks, rank = self.num_ranks, self.rank
        cached_mode = cached_rank_prefix_matrix is not None
        num_channels = self._num_channels(config)
        if cached_mode:
            _check(cached_channel_prefix_matrix is not None, "cached mode needs the channel prefix matrix")
            _check(
                cached_rank_prefix_matrix.dtype == torch.int32
                and cached_channel_prefix_matrix.dtype == torch.int32
            )
            _check(tuple(cached_rank_prefix_matrix.shape) == (num_ranks, num_ranks))
            _check(tuple(cached_channel_prefix_matrix.shape) == (num_ranks, num_channels))
            _check(cached_rank_prefix_matrix.is_contiguous() and cached_channel_prefix_matrix.is_contiguous())
        else:
            _check(num_tokens_per_rank is not None and num_tokens_per_expert is not None)
            _check(num_tokens_per_rank.dtype == torch.int32 and num_tokens_per_expert.dtype == torch.int32)
            _check(num_tokens_per_expert.dim() == 1 and num_tokens_per_expert.is_contiguous())
            _check(num_tokens_per_expert.size(0) % num_ranks == 0)
            _check(num_tokens_per_expert.size(0) // num_ranks <= NUM_MAX_LOCAL_EXPERTS)
            _check(num_tokens_per_rank.dim() == 1 and num_tokens_per_rank.is_contiguous())
            _check(num_tokens_per_rank.size(0) == num_ranks)
        _check(is_token_in_rank.dtype == torch.bool and is_token_in_rank.is_contiguous())
        _check(x.dim() == 2 and x.is_contiguous(), "x must be a contiguous 2D tensor")
        _check((x.size(1) * x.element_size()) % 16 == 0, "hidden bytes must be a multiple of 16")
        _check(tuple(is_token_in_rank.shape) == (x.size(0), num_ranks))

        num_tokens, hidden = x.shape
        hidden_bytes = hidden * x.element_size()
        num_experts = 0 if cached_mode else num_tokens_per_expert.size(0)
        num_local_experts = num_experts // num_ranks

        num_topk = 0
        _check((topk_idx is None) == (topk_weights is None))
        if topk_idx is not None:
            num_topk = topk_idx.size(1)
            _check(num_experts > 0)
            _check(
                topk_idx.dim() == 2
                and topk_idx.is_contiguous()
                and topk_idx.dtype in (torch.int32, torch.int64)
            )
            _check(topk_weights.dim() == 2 and topk_weights.is_contiguous())
            _check(topk_idx.size(0) == num_tokens and topk_weights.size(0) == num_tokens)
            _check(topk_weights.size(1) == num_topk and topk_weights.dtype == torch.float32)
        topk_idx_bytes = topk_idx.element_size() if topk_idx is not None else 8

        num_scales = scale_token_stride = scale_hidden_stride = 0
        if x_scales is not None:
            _check(x.element_size() == 1, "x_scales comes with a 1-byte x")
            _check(x_scales.dtype in (torch.float32, torch.int32) and x_scales.dim() == 2)
            _check(x_scales.size(0) == num_tokens)
            num_scales = x_scales.size(1)
            scale_token_stride, scale_hidden_stride = x_scales.stride()

        compute_stream, launch_stream, cross = self._fork(
            previous_event, async_finish, allocate_on_comm_stream
        )
        device = x.device
        num_memset_int = self._notify_memset_ints(num_channels)
        num_recv_tokens_per_expert_list = []
        is_token_in_rank_u8 = is_token_in_rank.view(torch.uint8)
        with self._alloc_stream(launch_stream, cross):
            if cached_mode:
                num_recv_tokens = cached_num_recv_tokens
                rank_prefix_matrix = cached_rank_prefix_matrix
                channel_prefix_matrix = cached_channel_prefix_matrix
                run_launcher(
                    launch_cached_notify_dispatch,
                    rank_prefix_matrix,
                    self.buffer_ptrs,
                    self.barrier_ptrs,
                    rank,
                    num_ranks,
                    num_memset_int,
                    launch_stream,
                )
            else:
                rank_prefix_matrix = torch.empty((num_ranks, num_ranks), dtype=torch.int32, device=device)
                channel_prefix_matrix = torch.empty(
                    (num_ranks, num_channels), dtype=torch.int32, device=device
                )
                recv_counters = torch.empty(1 + num_local_experts, dtype=torch.int32, device=device)
                _check(
                    num_ranks * (num_ranks + num_local_experts) * 4 <= self.num_nvl_bytes, "buffer too small"
                )
                run_launcher(
                    launch_notify_dispatch,
                    num_tokens_per_rank,
                    num_tokens_per_expert,
                    is_token_in_rank_u8,
                    channel_prefix_matrix,
                    rank_prefix_matrix,
                    recv_counters,
                    self.buffer_ptrs,
                    self.barrier_ptrs,
                    num_tokens,
                    rank,
                    num_ranks,
                    num_experts,
                    num_channels,
                    num_memset_int,
                    expert_alignment,
                    self.FULL_RANK_MATRIX,
                    launch_stream,
                )
                if num_worst_tokens > 0:
                    # no CPU sync: allocate the worst case
                    num_recv_tokens = num_worst_tokens
                    _check(topk_idx is not None, "num_worst_tokens needs top-k")
                else:
                    # the one D2H of the dispatch, like the C++ path's mapped counters
                    host_counters = self.host_counters[: recv_counters.numel()]
                    with torch.cuda.stream(launch_stream):
                        host_counters.copy_(recv_counters, non_blocking=True)
                    self.counters_ready.record(launch_stream)
                    self.counters_ready.synchronize()
                    num_recv_tokens = int(host_counters[0])
                    num_recv_tokens_per_expert_list = host_counters[1:].tolist()

            recv_x = torch.empty((num_recv_tokens, hidden), dtype=x.dtype, device=device)
            recv_src_idx = torch.empty(num_recv_tokens, dtype=torch.int32, device=device)
            recv_channel_prefix_matrix = torch.empty(
                (num_ranks, num_channels), dtype=torch.int32, device=device
            )
            send_head = torch.empty((num_tokens, num_ranks), dtype=torch.int32, device=device)
            recv_topk_idx = recv_topk_weights = recv_x_scales = None
            if topk_idx is not None:
                recv_topk_idx = torch.empty((num_recv_tokens, num_topk), dtype=topk_idx.dtype, device=device)
                recv_topk_weights = torch.empty(
                    (num_recv_tokens, num_topk), dtype=torch.float32, device=device
                )
            if x_scales is not None:
                recv_x_scales = torch.empty(
                    (num_recv_tokens, num_scales), dtype=x_scales.dtype, device=device
                )

        self._launch_dispatch(
            _DispatchArgs(
                x=x,
                x_scales=x_scales,
                topk_idx=topk_idx,
                topk_weights=topk_weights,
                is_token_in_rank_u8=is_token_in_rank_u8,
                rank_prefix_matrix=rank_prefix_matrix,
                channel_prefix_matrix=channel_prefix_matrix,
                recv_x=recv_x,
                recv_x_scales=recv_x_scales,
                recv_src_idx=recv_src_idx,
                recv_topk_idx=recv_topk_idx,
                recv_topk_weights=recv_topk_weights,
                recv_channel_prefix_matrix=recv_channel_prefix_matrix,
                send_head=send_head,
                num_tokens=num_tokens,
                num_recv_tokens=num_recv_tokens,
                num_worst_tokens=num_worst_tokens,
                num_channels=num_channels,
                hidden_bytes=hidden_bytes,
                num_topk=num_topk,
                num_experts=num_experts,
                num_scales=num_scales,
                scale_token_stride=scale_token_stride,
                scale_hidden_stride=scale_hidden_stride,
                topk_idx_bytes=topk_idx_bytes,
                config=config,
            ),
            launch_stream,
        )
        return self._dispatch_outputs(
            compute_stream,
            launch_stream,
            cross,
            async_finish,
            x,
            x_scales,
            topk_idx,
            topk_weights,
            is_token_in_rank,
            num_tokens_per_rank,
            num_tokens_per_expert,
            num_recv_tokens_per_expert_list,
            rank_prefix_matrix,
            channel_prefix_matrix,
            recv_x,
            recv_x_scales,
            recv_topk_idx,
            recv_topk_weights,
            recv_channel_prefix_matrix,
            recv_src_idx,
            send_head,
        )

    def _dispatch_outputs(
        self,
        compute_stream,
        launch_stream,
        cross,
        async_finish,
        x,
        x_scales,
        topk_idx,
        topk_weights,
        is_token_in_rank,
        num_tokens_per_rank,
        num_tokens_per_expert,
        num_recv_tokens_per_expert_list,
        rank_prefix_matrix,
        channel_prefix_matrix,
        recv_x,
        recv_x_scales,
        recv_topk_idx,
        recv_topk_weights,
        recv_channel_prefix_matrix,
        recv_src_idx,
        send_head,
    ):
        event = self._finish(
            compute_stream,
            launch_stream,
            cross,
            async_finish,
            (
                x,
                is_token_in_rank,
                rank_prefix_matrix,
                channel_prefix_matrix,
                recv_x,
                recv_src_idx,
                recv_channel_prefix_matrix,
                send_head,
                x_scales,
                topk_idx,
                topk_weights,
                num_tokens_per_rank,
                num_tokens_per_expert,
                recv_topk_idx,
                recv_topk_weights,
                recv_x_scales,
            ),
        )
        return (
            recv_x,
            recv_x_scales,
            recv_topk_idx,
            recv_topk_weights,
            num_recv_tokens_per_expert_list,
            rank_prefix_matrix,
            channel_prefix_matrix,
            recv_channel_prefix_matrix,
            recv_src_idx,
            send_head,
            event,
        )

    # ── combine ──────────────────────────────────────────────────────────────

    def intranode_combine(
        self,
        x,
        topk_weights,
        bias_0,
        bias_1,
        src_idx,
        rank_prefix_matrix,
        channel_prefix_matrix,
        send_head,
        config,
        previous_event,
        async_finish,
        allocate_on_comm_stream,
    ):
        num_ranks = self.num_ranks
        _check(x.dim() == 2 and x.is_contiguous(), "x must be a contiguous 2D tensor")
        _check(x.dtype in (torch.bfloat16, torch.float32), f"combine supports bf16 / fp32, got {x.dtype}")
        _check(src_idx.dim() == 1 and src_idx.is_contiguous() and src_idx.dtype == torch.int32)
        _check(send_head.dim() == 2 and send_head.is_contiguous() and send_head.dtype == torch.int32)
        _check(rank_prefix_matrix.is_contiguous() and rank_prefix_matrix.dtype == torch.int32)
        _check(channel_prefix_matrix.is_contiguous() and channel_prefix_matrix.dtype == torch.int32)
        num_channels = self._num_channels(config)

        num_tokens, hidden = x.shape
        num_recv_tokens = send_head.size(0)
        _check(src_idx.size(0) == num_tokens and send_head.size(1) == num_ranks)
        _check(tuple(rank_prefix_matrix.shape) == (num_ranks, num_ranks))
        _check(tuple(channel_prefix_matrix.shape) == (num_ranks, num_channels))
        hidden_bytes = hidden * x.element_size()
        _check(hidden_bytes % 16 == 0, "hidden bytes must be a multiple of 16")
        self._check_combine_config(config)

        compute_stream, launch_stream, cross = self._fork(
            previous_event, async_finish, allocate_on_comm_stream
        )
        num_topk = 0
        recv_topk_weights = None
        with self._alloc_stream(launch_stream, cross):
            if topk_weights is not None:
                _check(topk_weights.dim() == 2 and topk_weights.is_contiguous())
                _check(topk_weights.size(0) == num_tokens and topk_weights.dtype == torch.float32)
                num_topk = topk_weights.size(1)
                recv_topk_weights = torch.empty(
                    (num_recv_tokens, num_topk), dtype=torch.float32, device=x.device
                )
            for bias in (bias_0, bias_1):
                if bias is not None:
                    _check(bias.dim() == 2 and bias.is_contiguous() and bias.dtype == x.dtype)
                    _check(tuple(bias.shape) == (num_recv_tokens, hidden))
            recv_x = torch.empty((num_recv_tokens, hidden), dtype=x.dtype, device=x.device)

        self._launch_combine(
            _CombineArgs(
                x=x,
                topk_weights=topk_weights,
                bias_0=bias_0,
                bias_1=bias_1,
                src_idx=src_idx,
                rank_prefix_matrix=rank_prefix_matrix,
                channel_prefix_matrix=channel_prefix_matrix,
                send_head=send_head,
                recv_x=recv_x,
                recv_topk_weights=recv_topk_weights,
                num_tokens=num_tokens,
                num_recv_tokens=num_recv_tokens,
                num_channels=num_channels,
                hidden=hidden,
                hidden_bytes=hidden_bytes,
                num_topk=num_topk,
                config=config,
            ),
            launch_stream,
        )
        event = self._finish(
            compute_stream,
            launch_stream,
            cross,
            async_finish,
            (
                x,
                src_idx,
                send_head,
                rank_prefix_matrix,
                channel_prefix_matrix,
                recv_x,
                topk_weights,
                recv_topk_weights,
                bias_0,
                bias_1,
            ),
        )
        return recv_x, recv_topk_weights, event


# ── V1: the channel rings of the HIP kernels ─────────────────────────────────


class FlyDSLIntranodeV1Runtime(FlyDSLIntranodeRuntime):
    """Backend ``flydsl``: the V1 kernels of ``intranode_v1_kernel``, whose channel rings, buffer
    layout and handles are the C++ backend's. A channel is a pair of blocks, ``num_sms // 2`` of them."""

    def _num_channels(self, config) -> int:
        _check(config.num_sms % 2 == 0, "num_sms must be even")
        return config.num_sms // 2

    def _notify_memset_ints(self, num_channels: int) -> int:
        # the channel queues' start / end offsets and heads / tails
        return num_channels * self.num_ranks * 4

    def _launch_dispatch(self, args: _DispatchArgs, launch_stream: torch.cuda.Stream) -> None:
        config = args.config
        _, num_bytes = dispatch_buffer_layout(
            self.num_ranks,
            args.num_channels,
            config.num_max_nvl_chunked_recv_tokens,
            args.hidden_bytes,
            args.num_topk,
            args.topk_idx_bytes,
            args.num_scales,
        )
        _check(
            num_bytes <= self.num_nvl_bytes, f"dispatch needs {num_bytes} buffer bytes > {self.num_nvl_bytes}"
        )

        # the kernel reads optional operands only when their count is non-zero
        dummy_i32 = args.rank_prefix_matrix
        dummy_f32 = dummy_i32.view(torch.float32)
        run_launcher(
            launch_dispatch,
            args.recv_x.view(torch.uint8),
            args.recv_x_scales if args.recv_x_scales is not None else dummy_i32,
            args.recv_src_idx,
            args.recv_topk_idx if args.recv_topk_idx is not None else dummy_i32,
            args.recv_topk_weights if args.recv_topk_weights is not None else dummy_f32,
            args.recv_channel_prefix_matrix,
            args.send_head,
            args.x.view(torch.uint8),
            args.x_scales if args.x_scales is not None else dummy_i32,
            args.topk_idx if args.topk_idx is not None else dummy_i32,
            args.topk_weights if args.topk_weights is not None else dummy_f32,
            args.is_token_in_rank_u8,
            args.channel_prefix_matrix,
            self.buffer_ptrs,
            args.num_tokens,
            args.num_worst_tokens,
            args.scale_token_stride,
            args.scale_hidden_stride,
            self.rank,
            self.num_ranks,
            args.num_channels,
            args.hidden_bytes,
            args.num_topk,
            args.num_experts,
            args.num_scales,
            config.num_max_nvl_chunked_send_tokens,
            config.num_max_nvl_chunked_recv_tokens,
            args.topk_idx_bytes,
            launch_stream,
        )

    def _check_combine_config(self, config) -> None:
        min_recv_tokens = combine_min_recv_buffer_tokens(config.num_max_nvl_chunked_send_tokens)
        _check(
            config.num_max_nvl_chunked_recv_tokens >= min_recv_tokens,
            f"combine needs num_max_nvl_chunked_recv_tokens >= {min_recv_tokens} "
            f"for {config.num_max_nvl_chunked_send_tokens} send tokens, or it can deadlock",
        )

    def _launch_combine(self, args: _CombineArgs, launch_stream: torch.cuda.Stream) -> None:
        config = args.config
        num_ranks, num_channels = self.num_ranks, args.num_channels
        _, num_bytes = combine_buffer_layout(
            num_ranks,
            num_channels,
            config.num_max_nvl_chunked_recv_tokens,
            args.hidden_bytes,
            args.num_topk,
        )
        _check(
            num_bytes <= self.num_nvl_bytes, f"combine needs {num_bytes} buffer bytes > {self.num_nvl_bytes}"
        )

        # barrier, reset the queue heads / tails, and patch send_head for skipped tokens
        run_launcher(
            launch_cached_notify_combine,
            args.send_head,
            self.buffer_ptrs,
            self.barrier_ptrs,
            args.num_recv_tokens,
            self.rank,
            num_ranks,
            num_channels,
            num_channels * num_ranks * 2,
            launch_stream,
        )
        if args.num_recv_tokens > 0 or args.num_tokens > 0:
            dummy_f32 = args.rank_prefix_matrix.view(torch.float32)
            run_launcher(
                launch_combine,
                args.recv_x,
                args.recv_topk_weights if args.recv_topk_weights is not None else dummy_f32,
                args.x,
                args.topk_weights if args.topk_weights is not None else dummy_f32,
                args.bias_0 if args.bias_0 is not None else args.x,
                args.bias_1 if args.bias_1 is not None else args.x,
                args.src_idx,
                args.rank_prefix_matrix,
                args.channel_prefix_matrix,
                args.send_head,
                self.buffer_ptrs,
                args.num_recv_tokens,
                self.rank,
                num_ranks,
                num_channels,
                args.hidden,
                args.x.element_size(),
                args.num_topk,
                config.num_max_nvl_chunked_send_tokens,
                config.num_max_nvl_chunked_recv_tokens,
                args.bias_0 is not None,
                args.bias_1 is not None,
                launch_stream,
            )


# ── V2: direct writes into fixed slots ───────────────────────────────────────

# the widest top-k index the dispatch takes (int64)
_V2_TOPK_IDX_BYTES = 8


def v2_heap_bytes(num_max_tokens_per_rank: int, num_ranks: int, hidden_bytes: int, num_topk: int) -> int:
    """Heap bytes the V2 backend needs when no rank sends more than ``num_max_tokens_per_rank`` tokens:
    the dispatch region of a rank may receive every rank's tokens, and the combine region holds each of
    its own tokens coming back from every rank. ``hidden_bytes`` is the bf16 row; an fp8 row with its
    scales is smaller."""
    _check(num_max_tokens_per_rank > 0, "the V2 heap is sized from the maximum number of tokens per rank")
    rows = num_max_tokens_per_rank * num_ranks
    _check(
        rows * 4 <= COMBINE_FLAGS_BYTES,
        f"V2 takes at most {COMBINE_FLAGS_BYTES // 4} tokens x ranks, got {rows}",
    )
    dispatch_row = hidden_bytes + 4 + num_topk * (_V2_TOPK_IDX_BYTES + 4)
    row = max(dispatch_row, combine_v2_slot_bytes(hidden_bytes, num_topk))
    # every region starts 16-byte aligned (dispatch_v2_layout has 5); halves are whole 4 KiB pages
    half = -(-(rows * row + 16 * 5) // 4096) * 4096
    num_bytes = DATA_OFFSET + 2 * half
    assert dispatch_v2_layout(num_bytes, hidden_bytes, num_topk, _V2_TOPK_IDX_BYTES, 0)[1] >= rows
    assert combine_v2_layout(num_bytes, hidden_bytes, num_topk)[1] >= rows
    return num_bytes


class FlyDSLIntranodeV2Runtime(FlyDSLIntranodeRuntime):
    """Backend ``flydsl_v2``: the V2 kernels of ``intranode_v2_kernel``, direct writes into fixed slots
    instead of channel rings. Its outputs match V1's; its handles are its own (``send_head`` holds
    stream indices, and a channel is one block, ``num_sms`` of them)."""

    FULL_RANK_MATRIX = True

    def __init__(self, group: dist.ProcessGroup, num_nvl_bytes: int, comm_stream: torch.cuda.Stream) -> None:
        # V2 moves rows with TDM, which gfx942 / gfx950 lack; its non-TDM path has never run there
        _check(
            default_intranode_backend() != "cpp",
            f"the V2 backend (flydsl_v2) does not run on {' / '.join(CPP_DEEPEP_ARCHS)}; "
            f"leave intranode_backend unset, which takes this GPU's '{default_intranode_backend()}'",
        )
        super().__init__(group, num_nvl_bytes, comm_stream)

    def _init_state(self) -> None:
        _check(self.num_nvl_bytes > DATA_OFFSET, f"the V2 heap needs more than {DATA_OFFSET} bytes")
        # the combine's row flags hold epochs, which start at 1
        self.symm.get_buffer(
            self.rank, ((DATA_OFFSET - META_BYTES) // 4,), torch.int32, META_BYTES // 4
        ).zero_()
        self.dispatch_epoch = 0
        self.combine_epoch = 0
        # the combine's reduce queue; it only ever grows, each combine claims from where the
        # previous one left it
        self.reduce_counter = torch.zeros(1, dtype=torch.int32, device="cuda")
        self.reduce_grabs = 0
        # a combine may push into a peer's combine region only once that peer is done reducing
        # the previous combine: the notify of a dispatch in between is a barrier that ensures it
        self.combine_needs_barrier = False

    @staticmethod
    def _check_not_capturing() -> None:
        # a V2 launch takes host state that every call advances (the epoch its flags carry, the base of
        # the reduce queue): a graph replay would rerun the old values and take stale flags as fresh
        _check(
            not torch.cuda.is_current_stream_capturing(),
            "flydsl_v2 cannot be captured in a CUDA graph: each call advances host-side epochs; "
            "to capture the MoE token exchange on MI455X, use the all-to-all (RCCL) token dispatcher",
        )

    def _num_channels(self, config) -> int:
        self._check_not_capturing()
        return config.num_sms

    def _notify_memset_ints(self, num_channels: int) -> int:
        # V2 has no channel queues to reset
        return 0

    def _launch_dispatch(self, args: _DispatchArgs, launch_stream: torch.cuda.Stream) -> None:
        _, capacity = dispatch_v2_layout(
            self.num_nvl_bytes, args.hidden_bytes, args.num_topk, args.topk_idx_bytes, args.num_scales
        )
        _check(
            args.num_recv_tokens <= capacity,
            f"V2 dispatch receives {args.num_recv_tokens} tokens > the heap's {capacity}; raise num_nvl_bytes",
        )
        _check(
            (args.num_tokens + args.num_channels - 1) // args.num_channels < TAIL_DONE_BIT,
            f"V2 dispatch takes fewer than {TAIL_DONE_BIT} tokens per channel",
        )
        self.dispatch_epoch = self.dispatch_epoch % MAX_EPOCH + 1
        self.combine_needs_barrier = False
        dummy_i32 = args.rank_prefix_matrix
        dummy_f32 = dummy_i32.view(torch.float32)
        run_launcher(
            launch_dispatch_v2,
            args.recv_x.view(torch.uint8),
            args.recv_x_scales if args.recv_x_scales is not None else dummy_i32,
            args.recv_src_idx,
            args.recv_topk_idx if args.recv_topk_idx is not None else dummy_i32,
            args.recv_topk_weights if args.recv_topk_weights is not None else dummy_f32,
            args.recv_channel_prefix_matrix,
            args.send_head,
            args.x.view(torch.uint8),
            args.x_scales if args.x_scales is not None else dummy_i32,
            args.topk_idx if args.topk_idx is not None else dummy_i32,
            args.topk_weights if args.topk_weights is not None else dummy_f32,
            args.is_token_in_rank_u8,
            args.channel_prefix_matrix,
            args.rank_prefix_matrix,
            self.buffer_ptrs,
            args.num_tokens,
            args.num_worst_tokens,
            args.scale_token_stride,
            args.scale_hidden_stride,
            self.rank,
            self.dispatch_epoch,
            self.num_ranks,
            args.num_channels,
            args.hidden_bytes,
            args.num_topk,
            args.num_experts,
            args.num_scales,
            args.topk_idx_bytes,
            self.num_nvl_bytes,
            launch_stream,
        )

    def _launch_combine(self, args: _CombineArgs, launch_stream: torch.cuda.Stream) -> None:
        num_ranks, num_recv_tokens, num_channels = self.num_ranks, args.num_recv_tokens, args.num_channels
        _, capacity = combine_v2_layout(self.num_nvl_bytes, args.hidden_bytes, args.num_topk)
        # a token comes back from at most every rank
        _check(
            num_recv_tokens * num_ranks <= capacity,
            f"V2 combine may receive {num_recv_tokens * num_ranks} rows > the heap's {capacity}; "
            "raise num_nvl_bytes",
        )
        _check(
            num_recv_tokens * num_ranks * 4 <= COMBINE_FLAGS_BYTES,
            f"V2 combine of {num_recv_tokens} tokens needs more than {COMBINE_FLAGS_BYTES} flag bytes",
        )
        self.combine_epoch = self.combine_epoch % (2**31 - 1) + 1
        grab_base = self.reduce_grabs
        self.reduce_grabs += combine_v2_grabs(num_recv_tokens, num_channels)
        if self.combine_needs_barrier:
            run_launcher(launch_barrier, self.barrier_ptrs, self.rank, num_ranks, launch_stream)
        self.combine_needs_barrier = True
        dummy_f32 = args.rank_prefix_matrix.view(torch.float32)
        run_launcher(
            launch_combine_v2,
            args.recv_x,
            args.recv_topk_weights if args.recv_topk_weights is not None else dummy_f32,
            args.x,
            args.topk_weights if args.topk_weights is not None else dummy_f32,
            args.bias_0 if args.bias_0 is not None else args.x,
            args.bias_1 if args.bias_1 is not None else args.x,
            args.rank_prefix_matrix,
            args.send_head,
            self.buffer_ptrs,
            self.reduce_counter,
            args.num_tokens,
            num_recv_tokens,
            self.rank,
            self.combine_epoch,
            _as_i32(grab_base),
            num_ranks,
            num_channels,
            args.hidden,
            args.x.element_size(),
            args.num_topk,
            args.bias_0 is not None,
            args.bias_1 is not None,
            self.num_nvl_bytes,
            launch_stream,
        )
