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

``FlyDSLIntranodeRuntime`` exposes the three intranode methods of the C++ ``Buffer``
runtime with the same arguments and return tuples, so ``Buffer`` can route to either.
It owns its own uncached HIP-IPC heap; the handles it returns are plain tensors and
are interchangeable with the C++ backend's.

``version=2`` (backend ``flydsl_v2``) runs the V2 kernels of ``intranode_v2_kernel``: direct
writes into fixed slots instead of channel rings. Its outputs match V1's; its handles are its own
(``send_head`` holds stream indices, and a channel is one block, ``num_sms`` of them).
"""

import contextlib
import os
from typing import Optional

import torch
import torch.distributed as dist

from primus_turbo.common.constants import ENV_EP_FORCE_CURRENT_STREAM
from primus_turbo.flydsl.deep_ep.common import IS_GFX1250
from primus_turbo.flydsl.deep_ep.intranode_kernel import (
    combine_buffer_layout,
    combine_min_recv_buffer_tokens,
    dispatch_buffer_layout,
    launch_cached_notify_combine,
    launch_cached_notify_dispatch,
    launch_combine,
    launch_dispatch,
    launch_notify_dispatch,
)
from primus_turbo.flydsl.deep_ep.intranode_v2_kernel import (
    COMBINE_FLAGS_BYTES,
    DATA_OFFSET,
    DISPATCH_FLAG_ROWS,
    MAX_EPOCH,
    META_BYTES,
    TAIL_DONE_BIT,
    combine_v2_grabs,
    combine_v2_layout,
    combine_v2_reduce_chunks,
    combine_v2_slot_bytes,
    dispatch_v2_layout,
    dispatch_v2_recv_warps,
    launch_barrier,
    launch_combine_v2,
    launch_dispatch_v2,
)
from primus_turbo.flydsl.deep_ep.launch_cache import run_launcher
from primus_turbo.flydsl.deep_ep.layout_kernel import get_dispatch_layout_flydsl
from primus_turbo.pytorch.core.symm_mem import SymmetricMemory

from ._compat import NUM_MAX_NVL_PEERS, EventHandle, deep_ep_cpp

NUM_MAX_LOCAL_EXPERTS = 1024
SUPPORTED_NUM_RANKS = (2, 4, 8)
INTRANODE_BACKENDS = ("cpp", "flydsl", "flydsl_v2")


def _as_i32(value: int) -> int:
    """``value`` wrapped to a signed 32-bit kernel argument."""
    value &= 0xFFFFFFFF
    return value - (1 << 32) if value >= (1 << 31) else value


def default_intranode_backend() -> str:
    """``flydsl_v2`` on gfx1250; elsewhere ``cpp`` where the C++ DeepEP is loaded and ``flydsl`` where it
    is not."""
    if IS_GFX1250:
        return "flydsl_v2"
    return "cpp" if deep_ep_cpp is not None else "flydsl"


def get_intranode_backend(backend: Optional[str] = None) -> str:
    """``backend``, else ``PRIMUS_TURBO_DEEPEP_INTRANODE_BACKEND`` (``cpp`` | ``flydsl`` | ``flydsl_v2``),
    else ``default_intranode_backend()``."""
    default = default_intranode_backend()
    backend = (backend or os.environ.get("PRIMUS_TURBO_DEEPEP_INTRANODE_BACKEND", default)).strip().lower()
    if backend not in INTRANODE_BACKENDS:
        raise ValueError(
            f"Unknown DeepEP intranode backend {backend!r}; expected one of {INTRANODE_BACKENDS}"
        )
    return backend


def _check(cond: bool, msg: str = "") -> None:
    if not cond:
        raise RuntimeError(f"DeepEP FlyDSL intranode check failed: {msg}")


# the widest top-k index the dispatch takes (int64)
_V2_TOPK_IDX_BYTES = 8
# a cached V2 dispatch needs only a barrier (no peer still copies out of its dispatch region): V2 reads
# neither the notify's heap copy of the rank prefix matrix nor cleared channel queues. 1: the V1 cached
# notify (two barriers and that copy), for comparison
_V2_CACHED_NOTIFY = os.environ.get("PRIMUS_TURBO_DEEPEP_V2_CACHED_NOTIFY", "0") == "1"


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


class FlyDSLIntranodeRuntime:
    """Intranode dispatch / combine on FlyDSL kernels over a symmetric uncached heap."""

    def __init__(
        self, group: dist.ProcessGroup, num_nvl_bytes: int, comm_stream: torch.cuda.Stream, version: int = 1
    ) -> None:
        _check(version in (1, 2), f"version must be 1 or 2, got {version}")
        # V2 has only been built and tested for gfx1250; its non-TDM path has never run
        _check(
            version == 1 or IS_GFX1250,
            "the V2 backend (flydsl_v2) runs on gfx1250 only; on this GPU use intranode_backend="
            f"'{default_intranode_backend()}' (the default) or 'flydsl', or unset "
            "PRIMUS_TURBO_DEEPEP_INTRANODE_BACKEND",
        )
        self.version = version
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
        self.force_current_stream = int(os.environ.get(ENV_EP_FORCE_CURRENT_STREAM, "0")) != 0

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
        if version == 2:
            _check(num_nvl_bytes > DATA_OFFSET, f"the V2 heap needs more than {DATA_OFFSET} bytes")
            # the dispatch's and the combine's row flags hold epochs, which start at 1
            self.symm.get_buffer(
                self.rank, ((DATA_OFFSET - META_BYTES) // 4,), torch.int32, META_BYTES // 4
            ).zero_()
            self.dispatch_epoch = 0
            self.dispatch_flag_epoch = 0
            self.combine_epoch = 0
            # the combine's reduce queue; it only ever grows, each combine claims from where the
            # previous one left it
            self.reduce_counter = torch.zeros(1, dtype=torch.int32, device="cuda")
            self.reduce_grabs = 0
            # a combine may push into a peer's combine region only once that peer is done reducing
            # the previous combine: the notify of a dispatch in between is a barrier that ensures it
            self.combine_needs_barrier = False
        torch.cuda.synchronize()
        group.barrier()

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

    @staticmethod
    def _check_not_capturing() -> None:
        # a V2 launch takes host state that every call advances (the epoch its flags carry, the base of
        # the reduce queue): a graph replay would rerun the old values and take stale flags as fresh
        _check(
            not torch.cuda.is_current_stream_capturing(),
            "flydsl_v2 cannot be captured in a CUDA graph: each call advances host-side epochs; "
            "to capture DeepEP, set PRIMUS_TURBO_DEEPEP_INTRANODE_BACKEND=flydsl (or intranode_backend='flydsl')",
        )

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
        if self.version == 2:
            self._check_not_capturing()
            num_channels = config.num_sms
        else:
            _check(config.num_sms % 2 == 0, "num_sms must be even")
            num_channels = config.num_sms // 2
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
        # V2 has no channel queues to reset
        num_memset_int = 0 if self.version == 2 else num_channels * num_ranks * 4
        num_recv_tokens_per_expert_list = []
        is_token_in_rank_u8 = is_token_in_rank.view(torch.uint8)
        with self._alloc_stream(launch_stream, cross):
            if cached_mode:
                num_recv_tokens = cached_num_recv_tokens
                rank_prefix_matrix = cached_rank_prefix_matrix
                channel_prefix_matrix = cached_channel_prefix_matrix
                if self.version == 2 and not _V2_CACHED_NOTIFY:
                    run_launcher(launch_barrier, self.barrier_ptrs, rank, num_ranks, launch_stream)
                else:
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
                    self.version == 2,
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

        if self.version == 2:
            _, capacity = dispatch_v2_layout(
                self.num_nvl_bytes, hidden_bytes, num_topk, topk_idx_bytes, num_scales
            )
            _check(
                num_recv_tokens <= capacity,
                f"V2 dispatch receives {num_recv_tokens} tokens > the heap's {capacity}; raise num_nvl_bytes",
            )
            _check(
                (num_tokens + num_channels - 1) // num_channels < TAIL_DONE_BIT,
                f"V2 dispatch takes fewer than {TAIL_DONE_BIT} tokens per channel",
            )
            _check(
                dispatch_v2_recv_warps(num_ranks, num_channels) == 0 or num_recv_tokens <= DISPATCH_FLAG_ROWS,
                f"V2 dispatch receives {num_recv_tokens} tokens > its {DISPATCH_FLAG_ROWS} row flags",
            )
            self.dispatch_epoch = self.dispatch_epoch % MAX_EPOCH + 1
            self.dispatch_flag_epoch = self.dispatch_flag_epoch % (2**31 - 1) + 1
            self.combine_needs_barrier = False
            dummy_i32 = rank_prefix_matrix
            dummy_f32 = dummy_i32.view(torch.float32)
            run_launcher(
                launch_dispatch_v2,
                recv_x.view(torch.uint8),
                recv_x_scales if recv_x_scales is not None else dummy_i32,
                recv_src_idx,
                recv_topk_idx if recv_topk_idx is not None else dummy_i32,
                recv_topk_weights if recv_topk_weights is not None else dummy_f32,
                recv_channel_prefix_matrix,
                send_head,
                x.view(torch.uint8),
                x_scales if x_scales is not None else dummy_i32,
                topk_idx if topk_idx is not None else dummy_i32,
                topk_weights if topk_weights is not None else dummy_f32,
                is_token_in_rank_u8,
                channel_prefix_matrix,
                rank_prefix_matrix,
                self.buffer_ptrs,
                num_tokens,
                num_worst_tokens,
                scale_token_stride,
                scale_hidden_stride,
                rank,
                self.dispatch_epoch,
                self.dispatch_flag_epoch,
                num_ranks,
                num_channels,
                hidden_bytes,
                num_topk,
                num_experts,
                num_scales,
                topk_idx_bytes,
                self.num_nvl_bytes,
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

        _, num_bytes = dispatch_buffer_layout(
            num_ranks,
            num_channels,
            config.num_max_nvl_chunked_recv_tokens,
            hidden_bytes,
            num_topk,
            topk_idx_bytes,
            num_scales,
        )
        _check(
            num_bytes <= self.num_nvl_bytes, f"dispatch needs {num_bytes} buffer bytes > {self.num_nvl_bytes}"
        )

        # the kernel reads optional operands only when their count is non-zero
        dummy_i32 = rank_prefix_matrix
        dummy_f32 = dummy_i32.view(torch.float32)
        run_launcher(
            launch_dispatch,
            recv_x.view(torch.uint8),
            recv_x_scales if recv_x_scales is not None else dummy_i32,
            recv_src_idx,
            recv_topk_idx if recv_topk_idx is not None else dummy_i32,
            recv_topk_weights if recv_topk_weights is not None else dummy_f32,
            recv_channel_prefix_matrix,
            send_head,
            x.view(torch.uint8),
            x_scales if x_scales is not None else dummy_i32,
            topk_idx if topk_idx is not None else dummy_i32,
            topk_weights if topk_weights is not None else dummy_f32,
            is_token_in_rank_u8,
            channel_prefix_matrix,
            self.buffer_ptrs,
            num_tokens,
            num_worst_tokens,
            scale_token_stride,
            scale_hidden_stride,
            rank,
            num_ranks,
            num_channels,
            hidden_bytes,
            num_topk,
            num_experts,
            num_scales,
            config.num_max_nvl_chunked_send_tokens,
            config.num_max_nvl_chunked_recv_tokens,
            topk_idx_bytes,
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
        num_ranks, rank = self.num_ranks, self.rank
        _check(x.dim() == 2 and x.is_contiguous(), "x must be a contiguous 2D tensor")
        _check(x.dtype in (torch.bfloat16, torch.float32), f"combine supports bf16 / fp32, got {x.dtype}")
        _check(src_idx.dim() == 1 and src_idx.is_contiguous() and src_idx.dtype == torch.int32)
        _check(send_head.dim() == 2 and send_head.is_contiguous() and send_head.dtype == torch.int32)
        _check(rank_prefix_matrix.is_contiguous() and rank_prefix_matrix.dtype == torch.int32)
        _check(channel_prefix_matrix.is_contiguous() and channel_prefix_matrix.dtype == torch.int32)
        if self.version == 2:
            self._check_not_capturing()
            num_channels = config.num_sms
        else:
            _check(config.num_sms % 2 == 0, "num_sms must be even")
            num_channels = config.num_sms // 2

        num_tokens, hidden = x.shape
        num_recv_tokens = send_head.size(0)
        _check(src_idx.size(0) == num_tokens and send_head.size(1) == num_ranks)
        _check(tuple(rank_prefix_matrix.shape) == (num_ranks, num_ranks))
        _check(tuple(channel_prefix_matrix.shape) == (num_ranks, num_channels))
        hidden_bytes = hidden * x.element_size()
        _check(hidden_bytes % 16 == 0, "hidden bytes must be a multiple of 16")
        if self.version == 1:
            min_recv_tokens = combine_min_recv_buffer_tokens(config.num_max_nvl_chunked_send_tokens)
            _check(
                config.num_max_nvl_chunked_recv_tokens >= min_recv_tokens,
                f"combine needs num_max_nvl_chunked_recv_tokens >= {min_recv_tokens} "
                f"for {config.num_max_nvl_chunked_send_tokens} send tokens, or it can deadlock",
            )

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

        if self.version == 2:
            _, capacity = combine_v2_layout(self.num_nvl_bytes, hidden_bytes, num_topk)
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
            chunks = combine_v2_reduce_chunks(
                num_ranks, hidden, x.element_size(), bias_0 is not None, bias_1 is not None
            )
            self.reduce_grabs += combine_v2_grabs(num_recv_tokens, num_channels, chunks)
            if self.combine_needs_barrier:
                run_launcher(launch_barrier, self.barrier_ptrs, rank, num_ranks, launch_stream)
            self.combine_needs_barrier = True
            dummy_f32 = rank_prefix_matrix.view(torch.float32)
            run_launcher(
                launch_combine_v2,
                recv_x,
                recv_topk_weights if recv_topk_weights is not None else dummy_f32,
                x,
                topk_weights if topk_weights is not None else dummy_f32,
                bias_0 if bias_0 is not None else x,
                bias_1 if bias_1 is not None else x,
                rank_prefix_matrix,
                send_head,
                self.buffer_ptrs,
                self.reduce_counter,
                num_tokens,
                num_recv_tokens,
                rank,
                self.combine_epoch,
                _as_i32(grab_base),
                num_ranks,
                num_channels,
                hidden,
                x.element_size(),
                num_topk,
                bias_0 is not None,
                bias_1 is not None,
                self.num_nvl_bytes,
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

        _, num_bytes = combine_buffer_layout(
            num_ranks,
            num_channels,
            config.num_max_nvl_chunked_recv_tokens,
            hidden_bytes,
            num_topk,
        )
        _check(
            num_bytes <= self.num_nvl_bytes, f"combine needs {num_bytes} buffer bytes > {self.num_nvl_bytes}"
        )

        # barrier, reset the queue heads / tails, and patch send_head for skipped tokens
        run_launcher(
            launch_cached_notify_combine,
            send_head,
            self.buffer_ptrs,
            self.barrier_ptrs,
            num_recv_tokens,
            rank,
            num_ranks,
            num_channels,
            num_channels * num_ranks * 2,
            launch_stream,
        )
        if num_recv_tokens > 0 or num_tokens > 0:
            dummy_f32 = rank_prefix_matrix.view(torch.float32)
            run_launcher(
                launch_combine,
                recv_x,
                recv_topk_weights if recv_topk_weights is not None else dummy_f32,
                x,
                topk_weights if topk_weights is not None else dummy_f32,
                bias_0 if bias_0 is not None else x,
                bias_1 if bias_1 is not None else x,
                src_idx,
                rank_prefix_matrix,
                channel_prefix_matrix,
                send_head,
                self.buffer_ptrs,
                num_recv_tokens,
                rank,
                num_ranks,
                num_channels,
                hidden,
                x.element_size(),
                num_topk,
                config.num_max_nvl_chunked_send_tokens,
                config.num_max_nvl_chunked_recv_tokens,
                bias_0 is not None,
                bias_1 is not None,
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
