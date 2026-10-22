###############################################################################
# SPDX-License-Identifier: Apache-2.0
#
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2025 FlyDSL Project Contributors
#
# Adapted from FlyDSL (https://github.com/ROCm/FlyDSL)
# Modified by the Primus-Turbo team.
#
# This file is distributed under the Apache License 2.0 (see LICENSE-APACHE),
# not the MIT license that covers the rest of Primus-Turbo (see LICENSE).
###############################################################################

"""DeepEP intranode notify / dispatch / combine (FlyDSL).

Same algorithms, buffer layout and channel protocol as the HIP kernels in
``csrc/kernels/deep_ep/intranode.cu``, so every handle tensor (prefix matrices,
``recv_src_idx``, ``send_head``) comes out bit-identical and can be fed to either
backend. A channel is a pair of blocks: the even block sends, the odd one receives.
Within a block the threads split into one group per peer rank, and each group runs
its own ring buffer on the receiver's heap: the sender bumps the tail, the receiver
the head.

Ordering follows the HIP path's cheap fence (gfx942 / gfx950): the heap is uncached,
so a sender that waits for its own stores (``s_waitcnt``) before the tail store has
put the payload in memory, and the receiver reads it with system-scope loads that
miss every cache.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.compiler.ast_rewriter import ASTRewriter
from flydsl.expr import arith, range_constexpr
from flydsl.expr.primitive import get_dyn_shared
from flydsl.expr.primitive import ptrtoint as _fly_ptrtoint

from primus_turbo.flydsl.deep_ep.common import WARP_SIZE, load_index, tensor_addr, to_i64
from primus_turbo.flydsl.utils.buffer_ops import (
    _unwrap_value,
    buffer_load,
    buffer_store,
    create_buffer_resource,
    create_buffer_resource_from_addr,
)
from primus_turbo.flydsl.utils.prims import (
    _readfirstlane_i32,
    _readlane_i32,
    atomic_add,
    cast,
    copy_warp,
    ld,
    memory_fence,
    read_clock,
    spin_timed_out,
    st,
)

NUM_THREADS = 1024
NUM_WARPS = NUM_THREADS // WARP_SIZE
FINISHED_SUM_TAG = 1024
NOTIFY_THREADS = 256
# sc0 | sc1 | nt: a system-scope streaming load, which misses every GPU cache
SYS_LOAD = 0x13
# Bytes of a row one copy_warp call loads before it stores them (1 KiB a warp step). gfx9's vmcnt
# counts stores too, so the loads of the next piece wait for this piece's (remote) stores: small
# pieces keep the sender streaming, as the HIP kernel's 2-step unrolled warp copy does. Measured on
# MI355X (8 ranks, hidden 7168): dispatch is fastest at 2 KiB, the combine sender at 8 KiB.
DISPATCH_COPY_PIECE_BYTES = 2 * 1024
COMBINE_COPY_PIECE_BYTES = 8 * 1024
_INT_MAX = 0x7FFFFFFF


# ── Heap layout (byte offsets from a rank's buffer base) ─────────────────────


def dispatch_buffer_layout(
    num_ranks, num_channels, num_recv_buffer_tokens, hidden_bytes, num_topk, topk_idx_bytes, num_scales
):
    """Region offsets and total bytes of the dispatch heap, as laid out by ``intranode::dispatch``."""
    num_slots = num_channels * num_ranks
    offsets = {}
    offset = num_ranks * num_ranks * 4  # rank prefix matrix
    for name, slot_bytes in (
        ("start_offset", 4),
        ("end_offset", 4),
        ("head", 4),
        ("tail", 4),
        ("x", num_recv_buffer_tokens * hidden_bytes),
        ("src_idx", num_recv_buffer_tokens * 4),
        ("topk_idx", num_recv_buffer_tokens * num_topk * topk_idx_bytes),
        ("topk_weights", num_recv_buffer_tokens * num_topk * 4),
        ("x_scales", num_recv_buffer_tokens * num_scales * 4),
    ):
        offsets[name] = offset
        offset += num_slots * slot_bytes
    return offsets, offset


def combine_buffer_layout(num_ranks, num_channels, num_recv_buffer_tokens, hidden_bytes, num_topk):
    """Region offsets and total bytes of the combine heap, as laid out by ``intranode::combine``."""
    num_slots = num_channels * num_ranks
    offsets = {}
    offset = 0
    for name, slot_bytes in (
        ("head", 4),
        ("tail", 4),
        ("x", num_recv_buffer_tokens * hidden_bytes),
        ("src_idx", num_recv_buffer_tokens * 4),
        ("topk_weights", num_recv_buffer_tokens * num_topk * 4),
    ):
        offsets[name] = offset
        offset += num_slots * slot_bytes
    return offsets, offset


# ── Device helpers ───────────────────────────────────────────────────────────


def _lds_base():
    return _unwrap_value(_fly_ptrtoint(get_dyn_shared()))


def _load_lane_ptr(ptrs_res, rank):
    """Per-lane heap base (i64) from a device pointer table."""
    return buffer_load(ptrs_res, rank, vec_width=1, dtype=fx.T.i64)


def _load_ptr(ptrs_res, rank):
    """Peer heap base (i64) from a device pointer table, in SGPRs (``rank`` must be warp-uniform)."""
    return _uniform_i64(buffer_load(ptrs_res, rank, vec_width=1, dtype=fx.T.i64))


def _ld_sys(addr, idx=0, dtype=None):
    return ld(addr, fx.Int32(idx) if isinstance(idx, int) else idx, scope="sys", dtype=dtype)


def _st_sys(addr, idx, value):
    st(addr, fx.Int32(idx) if isinstance(idx, int) else idx, value, scope="sys")


def _ld_lds(lds_base, idx):
    return ld(lds_base, fx.Int32(idx) if isinstance(idx, int) else idx, scope="workgroup", space=3)


def _st_lds(lds_base, idx, value):
    st(lds_base, fx.Int32(idx) if isinstance(idx, int) else idx, value, scope="workgroup", space=3)


def _load_u8(res, idx):
    return cast(buffer_load(res, idx, vec_width=1, dtype=fx.T.i8), fx.T.i32)


def _sload_u8(res, idx):
    """A warp-uniform byte through the scalar cache. A vector load would put the wait for it on
    vmcnt, which on gfx9 also counts this warp's outstanding (remote) stores, so every flag read
    would wait out the XGMI round trip of the copies before it; a scalar load waits on lgkmcnt."""
    word = buffer_load(res, idx >> fx.Int32(2), vec_width=1, is_scalar=True)
    word = fx.arith.ArithValue(_unwrap_value(word), signed=True)
    return (word >> ((idx & fx.Int32(3)) * fx.Int32(8))) & fx.Int32(0xFF)


def _channel_task_range(num_tokens, num_channels, channel_id):
    """``get_channel_task_range``: the contiguous token range of one channel."""
    per_channel = (num_tokens + fx.Int32(num_channels - 1)) // fx.Int32(num_channels)
    start = arith.select(per_channel * channel_id < num_tokens, per_channel * channel_id, num_tokens)
    end = arith.select(start + per_channel < num_tokens, start + per_channel, num_tokens)
    return start, end


def _addr(base, offset_bytes, index=None, stride_bytes=0):
    """``base + offset_bytes + index * stride_bytes`` in i64."""
    addr = base + fx.Int64(offset_bytes)
    if index is not None:
        addr = addr + to_i64(index) * fx.Int64(stride_bytes)
    return addr


def _copy_row(dst_addr, src_addr, num_bytes, piece_bytes, load_cache_modifier=0):
    """Warp copy of one row between two warp-uniform i64 addresses."""
    # a row base the compiler cannot prove uniform turns every load / store into a waterfall
    dst_addr, src_addr = _uniform_i64(dst_addr), _uniform_i64(src_addr)
    for piece in range(0, num_bytes, piece_bytes):
        copy_warp(
            dst_addr + fx.Int64(piece),
            src_addr + fx.Int64(piece),
            min(piece_bytes, num_bytes - piece),
            load_cache_modifier=load_cache_modifier,
        )


def _uniform_i64(value):
    """``_uniform`` for an i64 (an address), one half at a time."""
    value = fx.arith.ArithValue(_unwrap_value(value), signed=True)
    low = _readfirstlane_i32(cast(value, fx.T.i32))
    high = _readfirstlane_i32(cast(value >> fx.Int64(32), fx.T.i32))
    return (to_i64(high) << fx.Int64(32)) | (to_i64(low) & fx.Int64(0xFFFFFFFF))


def _uniform(value):
    """Pin a warp-uniform i32 into an SGPR. A value the compiler cannot prove uniform (a vector
    load, an LDS read, a bit of a ballot) stays in VGPRs, and every buffer descriptor or branch
    built from it then pays a readfirstlane waterfall."""
    return _readfirstlane_i32(value)


def _ballot(pred):
    """Wave64 mask of the lanes where ``pred`` holds, as i64."""
    return fx.arith.ArithValue(fx.rocdl.ballot(res=fx.T.i64, pred=_unwrap_value(pred)), signed=True)


def _any_lane(pred):
    return _ballot(pred) != fx.Int64(0)


@ASTRewriter.transform
def group_barrier(lds_base, bar_id, num_arrived, num_warps_per_group: int, publish: bool = False):
    """Barrier over the warps of one rank group (``sync_barrier``), on an LDS arrival counter.

    ``num_arrived`` is the count this warp expects before this barrier; returns the next one.
    ``publish`` first waits for the warp's own stores, the cheap fence in front of a tail store."""
    if publish:
        fx.rocdl.s_waitcnt(0)
    memory_fence(order="release", scope="workgroup")
    num_arrived = num_arrived + fx.Int32(num_warps_per_group)
    if fx.lane_id() == fx.Int32(0):
        atomic_add(lds_base, bar_id, fx.Int32(1), "workgroup", 3)
        while _ld_lds(lds_base, bar_id) < num_arrived:
            fx.rocdl.s_sleep(fx.Int32(1))
    memory_fence(order="acquire", scope="workgroup")
    return num_arrived


@ASTRewriter.transform
def barrier_block(barrier_ptrs_res, rank, num_ranks: int, sync_only: bool, tag: str):
    """Cross-rank barrier of one block (``barrier_block``): +TAG on our row, -TAG on the peers'."""
    thread_id = fx.thread_idx.x
    if not sync_only:
        memory_fence(order="release", scope="sys")
        fx.gpu.barrier()
    if thread_id < fx.Int32(num_ranks):
        own_signal = _load_ptr(barrier_ptrs_res, rank)
        peer_signal = _load_lane_ptr(barrier_ptrs_res, thread_id)
        atomic_add(own_signal, thread_id, fx.Int32(FINISHED_SUM_TAG), scope="sys")
        atomic_add(peer_signal, rank, fx.Int32(-FINISHED_SUM_TAG), scope="sys")
        spin_start = read_clock()
        value = _ld_sys(own_signal, thread_id)
        while value > fx.Int32(0):
            if spin_timed_out(spin_start):
                fx.printf(
                    "[DeepEP FlyDSL " + tag + "] barrier timeout: rank={} thread={} value={}\n",
                    rank,
                    thread_id,
                    value,
                )
                spin_start = read_clock()
            value = _ld_sys(own_signal, thread_id)
    fx.gpu.barrier()
    memory_fence(order="acquire", scope="sys")


@ASTRewriter.transform
def wait_nonzero_sys(addr, rank, channel, tag: str):
    """Whole-warp poll of an i32 at ``addr`` until it is non-zero; returns it."""
    spin_start = read_clock()
    value = _ld_sys(addr)
    while value == fx.Int32(0):
        if spin_timed_out(spin_start):
            fx.printf("[DeepEP FlyDSL " + tag + "] timeout: rank={} channel={}\n", rank, channel)
            spin_start = read_clock()
        value = _ld_sys(addr)
    return value


@ASTRewriter.transform
def wait_free_slots(head_addr, tail, num_needed, num_recv_buffer_tokens: int, rank, channel, tag: str):
    """Whole-warp poll of a remote queue head until ``num_needed`` slots are free; returns the head."""
    spin_start = read_clock()
    head = _ld_sys(head_addr)
    while fx.Int32(num_recv_buffer_tokens) - (tail - head) < num_needed:
        if spin_timed_out(spin_start):
            fx.printf("[DeepEP FlyDSL " + tag + "] sender timeout: rank={} channel={}\n", rank, channel)
            spin_start = read_clock()
        head = _ld_sys(head_addr)
    return head


@ASTRewriter.transform
def wait_new_tail(tail_addr, head, rank, channel, tag: str):
    """Whole-warp poll of a local queue tail until it moves past ``head``; returns it."""
    spin_start = read_clock()
    fx.rocdl.s_waitcnt(0)
    tail = _ld_sys(tail_addr)
    while tail == head:
        if spin_timed_out(spin_start):
            fx.printf("[DeepEP FlyDSL " + tag + "] receiver timeout: rank={} channel={}\n", rank, channel)
            spin_start = read_clock()
        fx.rocdl.s_waitcnt(0)
        tail = _ld_sys(tail_addr)
    return tail


# ── notify_dispatch / cached_notify_dispatch ─────────────────────────────────


def _make_notify_dispatch(num_ranks, num_experts, num_channels, num_memset_int, expert_alignment):
    num_experts_per_rank = num_experts // num_ranks

    @flyc.kernel(known_block_size=[NOTIFY_THREADS, 1, 1])
    def notify_dispatch_kernel(
        NUM_TOKENS_PER_RANK: fx.Tensor,
        NUM_TOKENS_PER_EXPERT: fx.Tensor,
        IS_TOKEN_IN_RANK: fx.Tensor,
        CHANNEL_PREFIX_MATRIX: fx.Tensor,
        RANK_PREFIX_MATRIX: fx.Tensor,
        RECV_COUNTERS: fx.Tensor,
        BUFFER_PTRS: fx.Tensor,
        BARRIER_PTRS: fx.Tensor,
        num_tokens: fx.Int32,
        rank: fx.Int32,
    ):
        thread_id = fx.thread_idx.x
        block_id = fx.block_idx.x
        lane_id = thread_id % fx.Int32(WARP_SIZE)
        warp_id = thread_id // fx.Int32(WARP_SIZE)
        buffer_ptrs_res = create_buffer_resource(BUFFER_PTRS, max_size=True)
        barrier_ptrs_res = create_buffer_resource(BARRIER_PTRS, max_size=True)
        lds_base = _lds_base()

        if block_id == fx.Int32(0):
            barrier_block(barrier_ptrs_res, rank, num_ranks, True, "notify_dispatch/enter")

            # peer[rank, t] = tokens we send to t; peer's expert rows likewise
            if thread_id < fx.Int32(num_ranks):
                tokens_res = create_buffer_resource(NUM_TOKENS_PER_RANK, max_size=True)
                experts_res = create_buffer_resource(NUM_TOKENS_PER_EXPERT, max_size=True)
                peer = _load_lane_ptr(buffer_ptrs_res, thread_id)
                _st_sys(
                    peer,
                    rank * fx.Int32(num_ranks) + thread_id,
                    buffer_load(tokens_res, thread_id, vec_width=1, dtype=fx.T.i32),
                )
                for i in range(num_experts_per_rank):
                    count = buffer_load(
                        experts_res,
                        thread_id * fx.Int32(num_experts_per_rank) + fx.Int32(i),
                        vec_width=1,
                        dtype=fx.T.i32,
                    )
                    _st_sys(
                        peer,
                        fx.Int32(num_ranks * num_ranks) + rank * fx.Int32(num_experts_per_rank) + fx.Int32(i),
                        count,
                    )
            barrier_block(barrier_ptrs_res, rank, num_ranks, False, "notify_dispatch/gathered")

            local = _load_ptr(buffer_ptrs_res, rank)
            counters_res = create_buffer_resource(RECV_COUNTERS, max_size=True)
            # inclusive prefix over source ranks, one column per thread
            if thread_id < fx.Int32(num_ranks):
                running = _ld_sys(local, thread_id)
                for i in range(1, num_ranks):
                    running = running + _ld_sys(local, fx.Int32(i * num_ranks) + thread_id)
                    _st_sys(local, fx.Int32(i * num_ranks) + thread_id, running)
                if thread_id == rank:
                    buffer_store(running, counters_res, fx.Int32(0))
            expert = thread_id
            while expert < fx.Int32(num_experts_per_rank):
                total = fx.Int32(0)
                for i in range(num_ranks):
                    total = total + _ld_sys(
                        local, fx.Int32(num_ranks * num_ranks + i * num_experts_per_rank) + expert
                    )
                total = (
                    (total + fx.Int32(expert_alignment - 1))
                    // fx.Int32(expert_alignment)
                    * fx.Int32(expert_alignment)
                )
                buffer_store(total, counters_res, fx.Int32(1) + expert)
                expert = expert + fx.Int32(NOTIFY_THREADS)
            fx.gpu.barrier()

            rank_prefix_res = create_buffer_resource(RANK_PREFIX_MATRIX, max_size=True)
            i = thread_id
            while i < fx.Int32(num_ranks * num_ranks):
                buffer_store(_ld_sys(local, i), rank_prefix_res, i)
                i = i + fx.Int32(NOTIFY_THREADS)
            # clean the channel offsets / queue head and tail for the dispatch
            i = thread_id
            while i < fx.Int32(num_memset_int):
                _st_sys(local, fx.Int32(num_ranks * num_ranks) + i, fx.Int32(0))
                i = i + fx.Int32(NOTIFY_THREADS)
            barrier_block(barrier_ptrs_res, rank, num_ranks, False, "notify_dispatch/cleaned")
        else:
            # per-channel token counts to one destination, then the prefix over channels
            dst_rank = block_id - fx.Int32(1)
            in_rank_res = create_buffer_resource(IS_TOKEN_IN_RANK, max_size=True)
            channel = warp_id
            while channel < fx.Int32(num_channels):
                token_start, token_end = _channel_task_range(num_tokens, num_channels, channel)
                count = fx.Int32(0)
                token = token_start + lane_id
                while token < token_end:
                    count = count + _load_u8(in_rank_res, token * fx.Int32(num_ranks) + dst_rank)
                    token = token + fx.Int32(WARP_SIZE)
                for offset in (32, 16, 8, 4, 2, 1):
                    count = count + fx.shuffle_xor(count, offset, WARP_SIZE)
                if lane_id == fx.Int32(0):
                    _st_lds(lds_base, channel, count)
                channel = channel + fx.Int32(NOTIFY_THREADS // WARP_SIZE)
            fx.gpu.barrier()
            if thread_id == fx.Int32(0):
                prefix_res = create_buffer_resource(CHANNEL_PREFIX_MATRIX, max_size=True)
                running = fx.Int32(0)
                for c in range(num_channels):
                    running = running + _ld_lds(lds_base, c)
                    buffer_store(running, prefix_res, dst_rank * fx.Int32(num_channels) + fx.Int32(c))

    return notify_dispatch_kernel


@flyc.jit
def launch_notify_dispatch(
    num_tokens_per_rank,
    num_tokens_per_expert,
    is_token_in_rank,
    channel_prefix_matrix,
    rank_prefix_matrix,
    recv_counters,
    buffer_ptrs,
    barrier_ptrs,
    num_tokens: fx.Int32,
    rank: fx.Int32,
    num_ranks: fx.Constexpr[int],
    num_experts: fx.Constexpr[int],
    num_channels: fx.Constexpr[int],
    num_memset_int: fx.Constexpr[int],
    expert_alignment: fx.Constexpr[int],
    stream: fx.Stream,
):
    kernel = _make_notify_dispatch(num_ranks, num_experts, num_channels, num_memset_int, expert_alignment)
    kernel(
        num_tokens_per_rank,
        num_tokens_per_expert,
        is_token_in_rank,
        channel_prefix_matrix,
        rank_prefix_matrix,
        recv_counters,
        buffer_ptrs,
        barrier_ptrs,
        num_tokens,
        rank,
    ).launch(grid=(1 + num_ranks, 1, 1), block=(NOTIFY_THREADS, 1, 1), stream=stream, smem=num_channels * 4)


def _make_cached_notify_dispatch(num_ranks, num_memset_int):
    @flyc.kernel(known_block_size=[NOTIFY_THREADS, 1, 1])
    def cached_notify_dispatch_kernel(
        RANK_PREFIX_MATRIX: fx.Tensor,
        BUFFER_PTRS: fx.Tensor,
        BARRIER_PTRS: fx.Tensor,
        rank: fx.Int32,
    ):
        thread_id = fx.thread_idx.x
        buffer_ptrs_res = create_buffer_resource(BUFFER_PTRS, max_size=True)
        barrier_ptrs_res = create_buffer_resource(BARRIER_PTRS, max_size=True)
        barrier_block(barrier_ptrs_res, rank, num_ranks, True, "cached_notify_dispatch/enter")
        local = _load_ptr(buffer_ptrs_res, rank)
        rank_prefix_res = create_buffer_resource(RANK_PREFIX_MATRIX, max_size=True)
        i = thread_id
        while i < fx.Int32(num_ranks * num_ranks):
            _st_sys(local, i, buffer_load(rank_prefix_res, i, vec_width=1, dtype=fx.T.i32))
            i = i + fx.Int32(NOTIFY_THREADS)
        i = thread_id
        while i < fx.Int32(num_memset_int):
            _st_sys(local, fx.Int32(num_ranks * num_ranks) + i, fx.Int32(0))
            i = i + fx.Int32(NOTIFY_THREADS)
        barrier_block(barrier_ptrs_res, rank, num_ranks, False, "cached_notify_dispatch/cleaned")

    return cached_notify_dispatch_kernel


@flyc.jit
def launch_cached_notify_dispatch(
    rank_prefix_matrix,
    buffer_ptrs,
    barrier_ptrs,
    rank: fx.Int32,
    num_ranks: fx.Constexpr[int],
    num_memset_int: fx.Constexpr[int],
    stream: fx.Stream,
):
    kernel = _make_cached_notify_dispatch(num_ranks, num_memset_int)
    kernel(rank_prefix_matrix, buffer_ptrs, barrier_ptrs, rank).launch(
        grid=(1, 1, 1), block=(NOTIFY_THREADS, 1, 1), stream=stream
    )


# ── dispatch ─────────────────────────────────────────────────────────────────


def _make_dispatch(
    num_ranks,
    num_channels,
    hidden_bytes,
    num_topk,
    num_experts,
    num_scales,
    num_max_send_tokens,
    num_recv_buffer_tokens,
    topk_idx_bytes,
):
    assert NUM_WARPS % num_ranks == 0, "every rank group needs whole warps"
    threads_per_rank = NUM_THREADS // num_ranks
    warps_per_rank = threads_per_rank // WARP_SIZE
    num_experts_per_rank = num_experts // num_ranks if num_topk > 0 else 0
    assert num_topk <= WARP_SIZE
    layout, _ = dispatch_buffer_layout(
        num_ranks, num_channels, num_recv_buffer_tokens, hidden_bytes, num_topk, topk_idx_bytes, num_scales
    )
    # LDS: [num_ranks] group barrier counters, then [num_ranks] shared queue tails
    lds_shared_tail = num_ranks

    @flyc.kernel(known_block_size=[NUM_THREADS, 1, 1])
    def dispatch_kernel(
        RECV_X: fx.Tensor,
        RECV_X_SCALES: fx.Tensor,
        RECV_SRC_IDX: fx.Tensor,
        RECV_TOPK_IDX: fx.Tensor,
        RECV_TOPK_WEIGHTS: fx.Tensor,
        RECV_CHANNEL_OFFSET: fx.Tensor,
        SEND_HEAD: fx.Tensor,
        X: fx.Tensor,
        X_SCALES: fx.Tensor,
        TOPK_IDX: fx.Tensor,
        TOPK_WEIGHTS: fx.Tensor,
        IS_TOKEN_IN_RANK: fx.Tensor,
        CHANNEL_PREFIX_MATRIX: fx.Tensor,
        BUFFER_PTRS: fx.Tensor,
        num_tokens: fx.Int32,
        num_worst_tokens: fx.Int32,
        scale_token_stride: fx.Int32,
        scale_hidden_stride: fx.Int32,
        rank: fx.Int32,
    ):
        thread_id = fx.thread_idx.x
        block_id = fx.block_idx.x
        lane_id = thread_id % fx.Int32(WARP_SIZE)
        group = thread_id // fx.Int32(threads_per_rank)
        thread_in_group = thread_id % fx.Int32(threads_per_rank)
        warp_in_group = thread_in_group // fx.Int32(WARP_SIZE)
        channel = block_id // fx.Int32(2)
        lds_base = _lds_base()
        buffer_ptrs_res = create_buffer_resource(BUFFER_PTRS, max_size=True)
        local = _load_ptr(buffer_ptrs_res, rank)

        if thread_id < fx.Int32(2 * num_ranks):
            _st_lds(lds_base, thread_id, fx.Int32(0))
        fx.gpu.barrier()

        if block_id % fx.Int32(2) == fx.Int32(0):
            # ---- sender: this group feeds `group`'s ring for (channel, rank) ----
            peer = _load_ptr(buffer_ptrs_res, group)
            slot = channel * fx.Int32(num_ranks) + rank
            start_offset_addr = _addr(peer, layout["start_offset"], slot, 4)
            end_offset_addr = _addr(peer, layout["end_offset"], slot, 4)
            head_addr = _addr(peer, layout["head"], slot, 4)
            tail_addr = _addr(peer, layout["tail"], slot, 4)
            x_ring = _addr(peer, layout["x"], slot, num_recv_buffer_tokens * hidden_bytes)
            src_idx_ring = create_buffer_resource_from_addr(
                _addr(peer, layout["src_idx"], slot, num_recv_buffer_tokens * 4),
                num_records_bytes=num_recv_buffer_tokens * 4,
            )
            in_rank_res = create_buffer_resource(IS_TOKEN_IN_RANK, max_size=True)
            send_head_res = create_buffer_resource(SEND_HEAD, max_size=True)
            x_base = tensor_addr(X)

            # offsets go out as -value - 1, so a zero still reads as "not yet written"
            if (warp_in_group == fx.Int32(0)) & (lane_id == fx.Int32(0)):
                prefix_res = create_buffer_resource(CHANNEL_PREFIX_MATRIX, max_size=True)
                prefix_row = group * fx.Int32(num_channels)
                prev_channel = arith.select(channel > fx.Int32(0), channel - fx.Int32(1), fx.Int32(0))
                start_value = buffer_load(prefix_res, prefix_row + prev_channel, vec_width=1, dtype=fx.T.i32)
                start_value = arith.select(channel > fx.Int32(0), start_value, fx.Int32(0))
                end_value = buffer_load(prefix_res, prefix_row + channel, vec_width=1, dtype=fx.T.i32)
                _st_sys(start_offset_addr, 0, fx.Int32(-1) - start_value)
                _st_sys(end_offset_addr, 0, fx.Int32(-1) - end_value)

            token_start, token_end = _channel_task_range(num_tokens, num_channels, channel)
            token = token_start
            tail = fx.Int32(0)
            prefetched_head = fx.Int32(0)
            num_arrived = fx.Int32(0)
            while token < token_end:
                # the head only grows, so a stale one only understates the free slots
                if fx.Int32(num_recv_buffer_tokens) - (tail - prefetched_head) < fx.Int32(
                    num_max_send_tokens
                ):
                    prefetched_head = _uniform(
                        wait_free_slots(
                            head_addr,
                            tail,
                            fx.Int32(num_max_send_tokens),
                            num_recv_buffer_tokens,
                            rank,
                            channel,
                            "dispatch",
                        )
                    )
                num_chunk_tokens = fx.Int32(0)
                while (num_chunk_tokens < fx.Int32(num_max_send_tokens)) & (token < token_end):
                    is_selected = _sload_u8(in_rank_res, token * fx.Int32(num_ranks) + group) != fx.Int32(0)
                    if (lane_id == fx.Int32(0)) & (token % fx.Int32(warps_per_rank) == warp_in_group):
                        buffer_store(
                            arith.select(is_selected, tail, fx.Int32(-1)),
                            send_head_res,
                            token * fx.Int32(num_ranks) + group,
                        )
                    if is_selected:
                        dst_slot = tail % fx.Int32(num_recv_buffer_tokens)
                        tail = tail + fx.Int32(1)
                        if tail % fx.Int32(warps_per_rank) == warp_in_group:
                            _copy_row(
                                _addr(x_ring, 0, dst_slot, hidden_bytes),
                                _addr(x_base, 0, token, hidden_bytes),
                                hidden_bytes,
                                DISPATCH_COPY_PIECE_BYTES,
                            )
                            if lane_id == fx.Int32(0):
                                buffer_store(token, src_idx_ring, dst_slot)
                            if fx.const_expr(num_topk > 0):
                                if lane_id < fx.Int32(num_topk):
                                    topk_idx_res = create_buffer_resource(TOPK_IDX, max_size=True)
                                    topk_weights_res = create_buffer_resource(TOPK_WEIGHTS, max_size=True)
                                    topk_idx_ring = create_buffer_resource_from_addr(
                                        _addr(
                                            peer,
                                            layout["topk_idx"],
                                            slot,
                                            num_recv_buffer_tokens * num_topk * topk_idx_bytes,
                                        ),
                                        num_records_bytes=num_recv_buffer_tokens * num_topk * topk_idx_bytes,
                                    )
                                    topk_weights_ring = create_buffer_resource_from_addr(
                                        _addr(
                                            peer,
                                            layout["topk_weights"],
                                            slot,
                                            num_recv_buffer_tokens * num_topk * 4,
                                        ),
                                        num_records_bytes=num_recv_buffer_tokens * num_topk * 4,
                                    )
                                    elem = token * fx.Int32(num_topk) + lane_id
                                    ring_elem = dst_slot * fx.Int32(num_topk) + lane_id
                                    # the expert as a local index of `group`, -1 elsewhere
                                    expert = load_index(topk_idx_res, elem, TOPK_IDX.element_type)
                                    expert_begin = group * fx.Int32(num_experts_per_rank)
                                    is_local = (expert >= expert_begin) & (
                                        expert < expert_begin + fx.Int32(num_experts_per_rank)
                                    )
                                    local_expert = arith.select(is_local, expert - expert_begin, fx.Int32(-1))
                                    if fx.const_expr(topk_idx_bytes == 8):
                                        buffer_store(to_i64(local_expert), topk_idx_ring, ring_elem)
                                    else:
                                        buffer_store(local_expert, topk_idx_ring, ring_elem)
                                    weight = buffer_load(topk_weights_res, elem, vec_width=1, dtype=fx.T.f32)
                                    buffer_store(
                                        arith.select(is_local, weight, fx.Float32(0.0)),
                                        topk_weights_ring,
                                        ring_elem,
                                    )
                            if fx.const_expr(num_scales > 0):
                                x_scales_res = create_buffer_resource(X_SCALES, max_size=True)
                                x_scales_ring = create_buffer_resource_from_addr(
                                    _addr(
                                        peer,
                                        layout["x_scales"],
                                        slot,
                                        num_recv_buffer_tokens * num_scales * 4,
                                    ),
                                    num_records_bytes=num_recv_buffer_tokens * num_scales * 4,
                                )
                                for u in range_constexpr((num_scales + WARP_SIZE - 1) // WARP_SIZE):
                                    scale_idx = lane_id + fx.Int32(u * WARP_SIZE)
                                    if scale_idx < fx.Int32(num_scales):
                                        value = buffer_load(
                                            x_scales_res,
                                            token * scale_token_stride + scale_idx * scale_hidden_stride,
                                            vec_width=1,
                                            dtype=fx.T.i32,
                                        )
                                        buffer_store(
                                            value, x_scales_ring, dst_slot * fx.Int32(num_scales) + scale_idx
                                        )
                        num_chunk_tokens = num_chunk_tokens + fx.Int32(1)
                    token = token + fx.Int32(1)

                # read the next round's head here, so the publish wait covers it too
                prefetched_head = _uniform(_ld_sys(head_addr))
                num_arrived = group_barrier(lds_base, group, num_arrived, warps_per_rank, True)
                if (warp_in_group == fx.Int32(0)) & (lane_id == fx.Int32(0)):
                    _st_sys(tail_addr, 0, tail)
        else:
            # ---- receiver: this group drains the ring `group` fills for (channel, us) ----
            slot = channel * fx.Int32(num_ranks) + group
            start_offset_addr = _addr(local, layout["start_offset"], slot, 4)
            end_offset_addr = _addr(local, layout["end_offset"], slot, 4)
            head_addr = _addr(local, layout["head"], slot, 4)
            tail_addr = _addr(local, layout["tail"], slot, 4)
            x_ring = _addr(local, layout["x"], slot, num_recv_buffer_tokens * hidden_bytes)
            src_idx_ring = _addr(local, layout["src_idx"], slot, num_recv_buffer_tokens * 4)
            recv_x_base = tensor_addr(RECV_X)
            recv_src_idx_res = create_buffer_resource(RECV_SRC_IDX, max_size=True)

            rank_offset = fx.Int32(0)
            if group > fx.Int32(0):
                rank_offset = _uniform(_ld_sys(local, (group - fx.Int32(1)) * fx.Int32(num_ranks) + rank))
            total_offset = fx.Int32(-1) - _uniform(
                wait_nonzero_sys(start_offset_addr, rank, channel, "dispatch/start")
            )
            num_tokens_to_recv = fx.Int32(-1) - _uniform(
                wait_nonzero_sys(end_offset_addr, rank, channel, "dispatch/end")
            )
            num_tokens_to_recv = num_tokens_to_recv - total_offset
            if (warp_in_group == fx.Int32(0)) & (lane_id == fx.Int32(0)):
                recv_channel_offset_res = create_buffer_resource(RECV_CHANNEL_OFFSET, max_size=True)
                buffer_store(total_offset, recv_channel_offset_res, group * fx.Int32(num_channels) + channel)
            total_offset = total_offset + rank_offset

            head = fx.Int32(0)
            num_arrived = fx.Int32(0)
            while num_tokens_to_recv > fx.Int32(0):
                # one warp reads the tail so every warp of the group copies the same tokens
                if warp_in_group == fx.Int32(0):
                    new_tail = wait_new_tail(tail_addr, head, rank, channel, "dispatch")
                    if lane_id == fx.Int32(0):
                        _st_lds(lds_base, fx.Int32(lds_shared_tail) + group, new_tail)
                num_arrived = group_barrier(lds_base, group, num_arrived, warps_per_rank)
                tail = _uniform(_ld_lds(lds_base, fx.Int32(lds_shared_tail) + group))
                num_recv = tail - head

                chunk = warp_in_group
                while chunk < num_recv:
                    ring_slot = (head + chunk) % fx.Int32(num_recv_buffer_tokens)
                    _copy_row(
                        _addr(recv_x_base, 0, total_offset + chunk, hidden_bytes),
                        _addr(x_ring, 0, ring_slot, hidden_bytes),
                        hidden_bytes,
                        DISPATCH_COPY_PIECE_BYTES,
                        load_cache_modifier=SYS_LOAD,
                    )
                    chunk = chunk + fx.Int32(warps_per_rank)
                i = thread_in_group
                while i < num_recv:
                    value = _ld_sys(src_idx_ring, (head + i) % fx.Int32(num_recv_buffer_tokens))
                    buffer_store(value, recv_src_idx_res, total_offset + i)
                    i = i + fx.Int32(threads_per_rank)
                if fx.const_expr(num_topk > 0):
                    topk_idx_ring = _addr(
                        local, layout["topk_idx"], slot, num_recv_buffer_tokens * num_topk * topk_idx_bytes
                    )
                    topk_weights_ring = _addr(
                        local, layout["topk_weights"], slot, num_recv_buffer_tokens * num_topk * 4
                    )
                    recv_topk_idx_res = create_buffer_resource(RECV_TOPK_IDX, max_size=True)
                    recv_topk_weights_res = create_buffer_resource(RECV_TOPK_WEIGHTS, max_size=True)
                    i = thread_in_group
                    while i < num_recv * fx.Int32(num_topk):
                        chunk_idx = i // fx.Int32(num_topk)
                        ring_elem = ((head + chunk_idx) % fx.Int32(num_recv_buffer_tokens)) * fx.Int32(
                            num_topk
                        ) + i % fx.Int32(num_topk)
                        recv_elem = total_offset * fx.Int32(num_topk) + i
                        idx_dtype = fx.T.i64 if topk_idx_bytes == 8 else fx.T.i32
                        buffer_store(
                            _ld_sys(topk_idx_ring, ring_elem, dtype=idx_dtype), recv_topk_idx_res, recv_elem
                        )
                        buffer_store(
                            _ld_sys(topk_weights_ring, ring_elem, dtype=fx.T.f32),
                            recv_topk_weights_res,
                            recv_elem,
                        )
                        i = i + fx.Int32(threads_per_rank)
                if fx.const_expr(num_scales > 0):
                    x_scales_ring = _addr(
                        local, layout["x_scales"], slot, num_recv_buffer_tokens * num_scales * 4
                    )
                    recv_x_scales_res = create_buffer_resource(RECV_X_SCALES, max_size=True)
                    i = thread_in_group
                    while i < num_recv * fx.Int32(num_scales):
                        chunk_idx = i // fx.Int32(num_scales)
                        ring_elem = ((head + chunk_idx) % fx.Int32(num_recv_buffer_tokens)) * fx.Int32(
                            num_scales
                        ) + i % fx.Int32(num_scales)
                        buffer_store(
                            _ld_sys(x_scales_ring, ring_elem),
                            recv_x_scales_res,
                            total_offset * fx.Int32(num_scales) + i,
                        )
                        i = i + fx.Int32(threads_per_rank)

                head = head + num_recv
                total_offset = total_offset + num_recv
                num_arrived = group_barrier(lds_base, group, num_arrived, warps_per_rank)
                if (warp_in_group == fx.Int32(warps_per_rank - 1)) & (lane_id == fx.Int32(0)):
                    _st_sys(head_addr, 0, head)
                num_tokens_to_recv = num_tokens_to_recv - num_recv

        # with a worst-case allocation, the rows past the received ones read as unrouted
        if fx.const_expr(num_topk > 0):
            if num_worst_tokens > fx.Int32(0):
                recv_topk_idx_res = create_buffer_resource(RECV_TOPK_IDX, max_size=True)
                num_recv_tokens = _ld_sys(local, fx.Int32((num_ranks - 1) * num_ranks) + rank)
                i = num_recv_tokens * fx.Int32(num_topk) + block_id * fx.Int32(NUM_THREADS) + thread_id
                while i < num_worst_tokens * fx.Int32(num_topk):
                    if fx.const_expr(topk_idx_bytes == 8):
                        buffer_store(fx.Int64(-1), recv_topk_idx_res, i)
                    else:
                        buffer_store(fx.Int32(-1), recv_topk_idx_res, i)
                    i = i + fx.Int32(2 * num_channels * NUM_THREADS)

    return dispatch_kernel


@flyc.jit
def launch_dispatch(
    recv_x,
    recv_x_scales,
    recv_src_idx,
    recv_topk_idx,
    recv_topk_weights,
    recv_channel_offset,
    send_head,
    x,
    x_scales,
    topk_idx,
    topk_weights,
    is_token_in_rank,
    channel_prefix_matrix,
    buffer_ptrs,
    num_tokens: fx.Int32,
    num_worst_tokens: fx.Int32,
    scale_token_stride: fx.Int32,
    scale_hidden_stride: fx.Int32,
    rank: fx.Int32,
    num_ranks: fx.Constexpr[int],
    num_channels: fx.Constexpr[int],
    hidden_bytes: fx.Constexpr[int],
    num_topk: fx.Constexpr[int],
    num_experts: fx.Constexpr[int],
    num_scales: fx.Constexpr[int],
    num_max_send_tokens: fx.Constexpr[int],
    num_recv_buffer_tokens: fx.Constexpr[int],
    topk_idx_bytes: fx.Constexpr[int],
    stream: fx.Stream,
):
    kernel = _make_dispatch(
        num_ranks,
        num_channels,
        hidden_bytes,
        num_topk,
        num_experts,
        num_scales,
        num_max_send_tokens,
        num_recv_buffer_tokens,
        topk_idx_bytes,
    )
    kernel(
        recv_x,
        recv_x_scales,
        recv_src_idx,
        recv_topk_idx,
        recv_topk_weights,
        recv_channel_offset,
        send_head,
        x,
        x_scales,
        topk_idx,
        topk_weights,
        is_token_in_rank,
        channel_prefix_matrix,
        buffer_ptrs,
        num_tokens,
        num_worst_tokens,
        scale_token_stride,
        scale_hidden_stride,
        rank,
    ).launch(grid=(2 * num_channels, 1, 1), block=(NUM_THREADS, 1, 1), stream=stream, smem=2 * num_ranks * 4)


# ── cached_notify_combine ────────────────────────────────────────────────────


def _make_cached_notify_combine(num_ranks, num_channels, num_memset_int, num_threads):
    # far past any queue position: what a token with no later route to a rank waits for
    num_heuristic_head = 1 << 25

    @flyc.kernel(known_block_size=[num_threads, 1, 1])
    def cached_notify_combine_kernel(
        SEND_HEAD: fx.Tensor,
        BUFFER_PTRS: fx.Tensor,
        BARRIER_PTRS: fx.Tensor,
        num_recv_tokens: fx.Int32,
        rank: fx.Int32,
    ):
        thread_id = fx.thread_idx.x
        block_id = fx.block_idx.x
        lane_id = thread_id % fx.Int32(WARP_SIZE)
        if block_id == fx.Int32(0):
            buffer_ptrs_res = create_buffer_resource(BUFFER_PTRS, max_size=True)
            barrier_ptrs_res = create_buffer_resource(BARRIER_PTRS, max_size=True)
            barrier_block(barrier_ptrs_res, rank, num_ranks, True, "cached_notify_combine/enter")
            local = _load_ptr(buffer_ptrs_res, rank)
            i = thread_id
            while i < fx.Int32(num_memset_int):
                _st_sys(local, i, fx.Int32(0))
                i = i + fx.Int32(num_threads)
            barrier_block(barrier_ptrs_res, rank, num_ranks, False, "cached_notify_combine/cleaned")
        else:
            # a token not sent to a rank gets -(the next sent token's head) - 1, so the combine
            # receiver can still move that rank's queue head past it
            channel = block_id - fx.Int32(1)
            dst_rank = thread_id // fx.Int32(WARP_SIZE)
            if dst_rank < fx.Int32(num_ranks):
                send_head_res = create_buffer_resource(SEND_HEAD, max_size=True)
                token_start, token_end = _channel_task_range(num_recv_tokens, num_channels, channel)
                last_head = fx.Int32(num_heuristic_head)
                token_tail = token_end - fx.Int32(1)
                while token_tail >= token_start:
                    token = token_tail - lane_id
                    is_valid = token >= token_start
                    elem = arith.select(is_valid, token, token_start) * fx.Int32(num_ranks) + dst_rank
                    head = buffer_load(send_head_res, elem, vec_width=1, dtype=fx.T.i32)
                    head = arith.select(is_valid, head, fx.Int32(-1))
                    expected_head = fx.Int32(0)
                    # lane i holds token_tail - i, so walking lanes upward walks tokens backward
                    for i in range_constexpr(WARP_SIZE):
                        lane_head = _readlane_i32(head, i)
                        expected_head = arith.select(
                            (lane_head < fx.Int32(0)) & (lane_id == fx.Int32(i)),
                            fx.Int32(-1) - last_head,
                            expected_head,
                        )
                        last_head = arith.select(lane_head >= fx.Int32(0), lane_head, last_head)
                    if (head < fx.Int32(0)) & is_valid:
                        buffer_store(expected_head, send_head_res, elem)
                    token_tail = token_tail - fx.Int32(WARP_SIZE)

    return cached_notify_combine_kernel


@flyc.jit
def launch_cached_notify_combine(
    send_head,
    buffer_ptrs,
    barrier_ptrs,
    num_recv_tokens: fx.Int32,
    rank: fx.Int32,
    num_ranks: fx.Constexpr[int],
    num_channels: fx.Constexpr[int],
    num_memset_int: fx.Constexpr[int],
    stream: fx.Stream,
):
    num_threads = max(128, WARP_SIZE * num_ranks)
    kernel = _make_cached_notify_combine(num_ranks, num_channels, num_memset_int, num_threads)
    kernel(send_head, buffer_ptrs, barrier_ptrs, num_recv_tokens, rank).launch(
        grid=(1 + num_channels, 1, 1), block=(num_threads, 1, 1), stream=stream
    )


# ── combine ──────────────────────────────────────────────────────────────────


def _make_combine(
    num_ranks,
    num_channels,
    hidden,
    elem_bytes,
    num_topk,
    num_max_send_tokens,
    num_recv_buffer_tokens,
    has_bias_0,
    has_bias_1,
):
    assert NUM_WARPS % num_ranks == 0 and num_ranks <= WARP_SIZE and num_topk <= WARP_SIZE
    hidden_bytes = hidden * elem_bytes
    assert hidden_bytes % 16 == 0
    warps_per_rank = NUM_WARPS // num_ranks
    num_reduce_warps = NUM_WARPS - 1
    layout, _ = combine_buffer_layout(num_ranks, num_channels, num_recv_buffer_tokens, hidden_bytes, num_topk)
    # LDS: [num_ranks] group barrier counters, [NUM_WARPS][num_ranks] per-warp heads,
    # [num_ranks] queue tails, [NUM_WARPS] retired flags
    lds_warp_head = num_ranks
    lds_tail = lds_warp_head + NUM_WARPS * num_ranks
    lds_retired = lds_tail + num_ranks
    lds_words = lds_retired + NUM_WARPS
    # one 16-byte vector per lane per step; passes bound the row loads in flight
    vec_elems = 16 // elem_bytes
    num_vecs = hidden_bytes // 16
    num_steps = (num_vecs + WARP_SIZE - 1) // WARP_SIZE
    steps_per_pass = max(1, 32 // (num_ranks + int(has_bias_0) + int(has_bias_1)))

    @flyc.kernel(known_block_size=[NUM_THREADS, 1, 1])
    def combine_kernel(
        RECV_X: fx.Tensor,
        RECV_TOPK_WEIGHTS: fx.Tensor,
        X: fx.Tensor,
        TOPK_WEIGHTS: fx.Tensor,
        BIAS_0: fx.Tensor,
        BIAS_1: fx.Tensor,
        SRC_IDX: fx.Tensor,
        RANK_PREFIX_MATRIX: fx.Tensor,
        CHANNEL_PREFIX_MATRIX: fx.Tensor,
        SEND_HEAD: fx.Tensor,
        BUFFER_PTRS: fx.Tensor,
        num_recv_tokens: fx.Int32,
        rank: fx.Int32,
    ):
        thread_id = fx.thread_idx.x
        block_id = fx.block_idx.x
        lane_id = thread_id % fx.Int32(WARP_SIZE)
        warp_id = thread_id // fx.Int32(WARP_SIZE)
        channel = block_id // fx.Int32(2)
        lds_base = _lds_base()
        buffer_ptrs_res = create_buffer_resource(BUFFER_PTRS, max_size=True)

        i = thread_id
        while i < fx.Int32(lds_words):
            _st_lds(lds_base, i, fx.Int32(0))
            i = i + fx.Int32(NUM_THREADS)
        fx.gpu.barrier()

        if block_id % fx.Int32(2) == fx.Int32(0):
            # ---- sender: warp w feeds rank (channel + w) % num_ranks ----
            send_rank = (channel + warp_id) % fx.Int32(num_ranks)
            warp_in_rank = warp_id // fx.Int32(num_ranks)
            peer = _load_ptr(buffer_ptrs_res, send_rank)
            slot = channel * fx.Int32(num_ranks) + rank
            head_addr = _addr(peer, layout["head"], slot, 4)
            tail_addr = _addr(peer, layout["tail"], slot, 4)
            x_ring = _addr(peer, layout["x"], slot, num_recv_buffer_tokens * hidden_bytes)
            src_idx_ring = create_buffer_resource_from_addr(
                _addr(peer, layout["src_idx"], slot, num_recv_buffer_tokens * 4),
                num_records_bytes=num_recv_buffer_tokens * 4,
            )
            x_base = tensor_addr(X)
            src_idx_res = create_buffer_resource(SRC_IDX, max_size=True)

            rank_prefix_res = create_buffer_resource(RANK_PREFIX_MATRIX, max_size=True)
            channel_prefix_res = create_buffer_resource(CHANNEL_PREFIX_MATRIX, max_size=True)
            rank_offset = fx.Int32(0)
            if send_rank > fx.Int32(0):
                rank_offset = _uniform(
                    buffer_load(
                        rank_prefix_res,
                        (send_rank - fx.Int32(1)) * fx.Int32(num_ranks) + rank,
                        vec_width=1,
                        dtype=fx.T.i32,
                    )
                )
            num_rank_tokens = (
                _uniform(
                    buffer_load(
                        rank_prefix_res, send_rank * fx.Int32(num_ranks) + rank, vec_width=1, dtype=fx.T.i32
                    )
                )
                - rank_offset
            )
            channel_offset = _uniform(
                buffer_load(
                    channel_prefix_res,
                    send_rank * fx.Int32(num_channels) + channel,
                    vec_width=1,
                    dtype=fx.T.i32,
                )
            )
            channel_end = num_rank_tokens
            if channel < fx.Int32(num_channels - 1):
                channel_end = _uniform(
                    buffer_load(
                        channel_prefix_res,
                        send_rank * fx.Int32(num_channels) + channel + fx.Int32(1),
                        vec_width=1,
                        dtype=fx.T.i32,
                    )
                )
            token = rank_offset + channel_offset
            token_end = rank_offset + channel_end

            tail = fx.Int32(0)
            prefetched_head = fx.Int32(0)
            num_arrived = fx.Int32(0)
            while token < token_end:
                num_round_tokens = arith.select(
                    token_end - token < fx.Int32(num_max_send_tokens),
                    token_end - token,
                    fx.Int32(num_max_send_tokens),
                )
                if fx.Int32(num_recv_buffer_tokens) - (tail - prefetched_head) < num_round_tokens:
                    prefetched_head = _uniform(
                        wait_free_slots(
                            head_addr,
                            tail,
                            num_round_tokens,
                            num_recv_buffer_tokens,
                            rank,
                            channel,
                            "combine",
                        )
                    )
                i = warp_in_rank
                while i < num_round_tokens:
                    dst_slot = (tail + i) % fx.Int32(num_recv_buffer_tokens)
                    _copy_row(
                        _addr(x_ring, 0, dst_slot, hidden_bytes),
                        _addr(x_base, 0, token + i, hidden_bytes),
                        hidden_bytes,
                        COMBINE_COPY_PIECE_BYTES,
                    )
                    if lane_id == fx.Int32(0):
                        buffer_store(
                            buffer_load(src_idx_res, token + i, vec_width=1, dtype=fx.T.i32),
                            src_idx_ring,
                            dst_slot,
                        )
                    if fx.const_expr(num_topk > 0):
                        if lane_id < fx.Int32(num_topk):
                            topk_weights_res = create_buffer_resource(TOPK_WEIGHTS, max_size=True)
                            topk_weights_ring = create_buffer_resource_from_addr(
                                _addr(
                                    peer, layout["topk_weights"], slot, num_recv_buffer_tokens * num_topk * 4
                                ),
                                num_records_bytes=num_recv_buffer_tokens * num_topk * 4,
                            )
                            weight = buffer_load(
                                topk_weights_res,
                                (token + i) * fx.Int32(num_topk) + lane_id,
                                vec_width=1,
                                dtype=fx.T.f32,
                            )
                            buffer_store(weight, topk_weights_ring, dst_slot * fx.Int32(num_topk) + lane_id)
                    i = i + fx.Int32(warps_per_rank)
                token = token + num_round_tokens
                tail = tail + num_round_tokens

                prefetched_head = _uniform(_ld_sys(head_addr))
                num_arrived = group_barrier(lds_base, send_rank, num_arrived, warps_per_rank, True)
                if (warp_in_rank == fx.Int32(0)) & (lane_id == fx.Int32(0)):
                    _st_sys(tail_addr, 0, tail)
        else:
            local = _load_ptr(buffer_ptrs_res, rank)
            if warp_id == fx.Int32(0):
                # ---- queue-head updater: lane r tracks rank r's ring ----
                if lane_id < fx.Int32(num_ranks):
                    head_addr = _addr(local, layout["head"], channel * fx.Int32(num_ranks) + lane_id, 4)
                    tail_addr = _addr(local, layout["tail"], channel * fx.Int32(num_ranks) + lane_id, 4)
                    last_head = fx.Int32(0)
                    num_retired = fx.Int32(0)
                    for w in range_constexpr(1, NUM_WARPS):
                        num_retired = num_retired + _ld_lds(lds_base, lds_retired + w)
                    while num_retired < fx.Int32(num_reduce_warps):
                        fx.rocdl.s_waitcnt(0)
                        _st_lds(lds_base, fx.Int32(lds_tail) + lane_id, _ld_sys(tail_addr))
                        min_head = fx.Int32(_INT_MAX)
                        for w in range_constexpr(1, NUM_WARPS):
                            warp_head = _ld_lds(lds_base, fx.Int32(lds_warp_head + w * num_ranks) + lane_id)
                            is_live = _ld_lds(lds_base, lds_retired + w) == fx.Int32(0)
                            min_head = arith.select(is_live & (warp_head < min_head), warp_head, min_head)
                        if (min_head != fx.Int32(_INT_MAX)) & (min_head > last_head):
                            last_head = min_head
                            _st_sys(head_addr, 0, min_head)
                        num_retired = fx.Int32(0)
                        for w in range_constexpr(1, NUM_WARPS):
                            num_retired = num_retired + _ld_lds(lds_base, lds_retired + w)
            else:
                # ---- reducers: one token per warp, summed over the ranks it went to ----
                x_rings = [
                    _addr(
                        local,
                        layout["x"],
                        channel * fx.Int32(num_ranks) + fx.Int32(r),
                        num_recv_buffer_tokens * hidden_bytes,
                    )
                    for r in range(num_ranks)
                ]
                send_head_res = create_buffer_resource(SEND_HEAD, max_size=True)
                recv_x_base = tensor_addr(RECV_X)
                in_dtype = X.element_type
                f32_vec = fx.T.vec(vec_elems, fx.T.f32)
                in_vec = fx.T.vec(vec_elems, in_dtype.ir_type)
                lane_rank = arith.select(lane_id < fx.Int32(num_ranks), lane_id, fx.Int32(0))

                token_start, token_end = _channel_task_range(num_recv_tokens, num_channels, channel)
                token = token_start + warp_id - fx.Int32(1)
                while token < token_end:
                    expected_head = buffer_load(
                        send_head_res, token * fx.Int32(num_ranks) + lane_rank, vec_width=1, dtype=fx.T.i32
                    )
                    expected_head = arith.select(lane_id < fx.Int32(num_ranks), expected_head, fx.Int32(-1))
                    spin_start = read_clock()
                    pending = (expected_head >= fx.Int32(0)) & (
                        _ld_lds(lds_base, fx.Int32(lds_tail) + lane_rank) <= expected_head
                    )
                    while _any_lane(pending):
                        if spin_timed_out(spin_start):
                            if lane_id == fx.Int32(0):
                                fx.printf(
                                    "[DeepEP FlyDSL combine] receiver timeout: rank={} channel={} token={}\n",
                                    rank,
                                    channel,
                                    token,
                                )
                            spin_start = read_clock()
                        pending = (expected_head >= fx.Int32(0)) & (
                            _ld_lds(lds_base, fx.Int32(lds_tail) + lane_rank) <= expected_head
                        )

                    # rank r's copy of the token, or an empty resource whose loads read 0
                    row_resources = []
                    is_routed = []
                    ring_slots = []
                    for r in range_constexpr(num_ranks):
                        head_r = _readlane_i32(expected_head, r)
                        routed = head_r >= fx.Int32(0)
                        ring_slot = arith.select(routed, head_r, fx.Int32(0)) % fx.Int32(
                            num_recv_buffer_tokens
                        )
                        row_resources.append(
                            create_buffer_resource_from_addr(
                                _addr(x_rings[r], 0, ring_slot, hidden_bytes),
                                num_records_bytes=arith.select(routed, fx.Int32(hidden_bytes), fx.Int32(0)),
                            )
                        )
                        is_routed.append(routed)
                        ring_slots.append(ring_slot)
                    bias_resources = []
                    for has_bias, bias in ((has_bias_0, BIAS_0), (has_bias_1, BIAS_1)):
                        if fx.const_expr(has_bias):
                            bias_resources.append(
                                create_buffer_resource_from_addr(
                                    _addr(tensor_addr(bias), 0, token, hidden_bytes),
                                    num_records_bytes=hidden_bytes,
                                )
                            )
                    out_res = create_buffer_resource_from_addr(
                        _addr(recv_x_base, 0, token, hidden_bytes), num_records_bytes=hidden_bytes
                    )

                    for first_step in range_constexpr(0, num_steps, steps_per_pass):
                        steps = range(first_step, min(first_step + steps_per_pass, num_steps))
                        # element offsets; lanes past the row read 0 and store nothing
                        cols = [
                            (lane_id + fx.Int32(step * WARP_SIZE)) * fx.Int32(vec_elems) for step in steps
                        ]
                        sums = [None] * len(cols)
                        # bias_0 + bias_1 first, then the ranks in ascending order, as the HIP kernel adds them
                        for bias_res in bias_resources:
                            values = [
                                buffer_load(bias_res, col, vec_width=vec_elems, dtype=in_dtype)
                                for col in cols
                            ]
                            for k, value in enumerate(values):
                                term = (
                                    value if fx.const_expr(elem_bytes == 4) else fx.arith.extf(f32_vec, value)
                                )
                                sums[k] = term if sums[k] is None else fx.arith.addf(sums[k], term)
                        for r in range_constexpr(num_ranks):
                            values = [
                                buffer_load(
                                    row_resources[r],
                                    col,
                                    vec_width=vec_elems,
                                    dtype=in_dtype,
                                    cache_modifier=SYS_LOAD,
                                )
                                for col in cols
                            ]
                            for k, value in enumerate(values):
                                term = (
                                    value if fx.const_expr(elem_bytes == 4) else fx.arith.extf(f32_vec, value)
                                )
                                sums[k] = term if sums[k] is None else fx.arith.addf(sums[k], term)
                        for k, col in enumerate(cols):
                            out = (
                                sums[k]
                                if fx.const_expr(elem_bytes == 4)
                                else fx.arith.trunc_f(in_vec, sums[k])
                            )
                            buffer_store(out, out_res, col)

                    if fx.const_expr(num_topk > 0):
                        if lane_id < fx.Int32(num_topk):
                            weight = fx.Float32(0.0)
                            for r in range_constexpr(num_ranks):
                                weights_ring = _addr(
                                    local,
                                    layout["topk_weights"],
                                    channel * fx.Int32(num_ranks) + fx.Int32(r),
                                    num_recv_buffer_tokens * num_topk * 4,
                                )
                                value = _ld_sys(
                                    weights_ring, ring_slots[r] * fx.Int32(num_topk) + lane_id, dtype=fx.T.f32
                                )
                                weight = weight + arith.select(is_routed[r], value, fx.Float32(0.0))
                            recv_topk_weights_res = create_buffer_resource(RECV_TOPK_WEIGHTS, max_size=True)
                            buffer_store(weight, recv_topk_weights_res, token * fx.Int32(num_topk) + lane_id)

                    if lane_id < fx.Int32(num_ranks):
                        next_head = arith.select(
                            expected_head < fx.Int32(0),
                            fx.Int32(-1) - expected_head,
                            expected_head + fx.Int32(1),
                        )
                        _st_lds(
                            lds_base,
                            warp_id * fx.Int32(num_ranks) + fx.Int32(lds_warp_head) + lane_id,
                            next_head,
                        )
                    token = token + fx.Int32(num_reduce_warps)

                if lane_id == fx.Int32(0):
                    _st_lds(lds_base, fx.Int32(lds_retired) + warp_id, fx.Int32(1))

    return combine_kernel, lds_words


@flyc.jit
def launch_combine(
    recv_x,
    recv_topk_weights,
    x,
    topk_weights,
    bias_0,
    bias_1,
    src_idx,
    rank_prefix_matrix,
    channel_prefix_matrix,
    send_head,
    buffer_ptrs,
    num_recv_tokens: fx.Int32,
    rank: fx.Int32,
    num_ranks: fx.Constexpr[int],
    num_channels: fx.Constexpr[int],
    hidden: fx.Constexpr[int],
    elem_bytes: fx.Constexpr[int],
    num_topk: fx.Constexpr[int],
    num_max_send_tokens: fx.Constexpr[int],
    num_recv_buffer_tokens: fx.Constexpr[int],
    has_bias_0: fx.Constexpr[bool],
    has_bias_1: fx.Constexpr[bool],
    stream: fx.Stream,
):
    kernel, lds_words = _make_combine(
        num_ranks,
        num_channels,
        hidden,
        elem_bytes,
        num_topk,
        num_max_send_tokens,
        num_recv_buffer_tokens,
        has_bias_0,
        has_bias_1,
    )
    kernel(
        recv_x,
        recv_topk_weights,
        x,
        topk_weights,
        bias_0,
        bias_1,
        src_idx,
        rank_prefix_matrix,
        channel_prefix_matrix,
        send_head,
        buffer_ptrs,
        num_recv_tokens,
        rank,
    ).launch(grid=(2 * num_channels, 1, 1), block=(NUM_THREADS, 1, 1), stream=stream, smem=lds_words * 4)
