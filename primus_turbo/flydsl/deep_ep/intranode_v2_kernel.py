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

"""DeepEP intranode dispatch / combine, V2 (FlyDSL): direct writes into fixed slots.

V1 streams every (channel, rank) pair through a ring buffer, one chunk per head / tail handshake,
and the receiver copies the ring into ``recv_x``. V2 drops the rings: every token has a fixed slot
on each rank it goes to, so a sender writes it there directly and never waits for a receiver.

Dispatch. A token's slot on rank ``d`` is our rank's offset in ``d``'s receive order (the rank
prefix matrix of the notify) plus its index in our stream of tokens to ``d`` -- the same order the
V1 receiver produces, so ``recv_x`` and every per-token output match V1 exactly. Each block owns
one channel (a contiguous token range); it scans the range a tile at a time for the per-destination
indices (also returned as ``send_head``), then each warp loads a token's row once and stores it to
every rank the token goes to -- our own rank's straight into the outputs, the others' into their
heap. When its rows are visible, the block tells each rank how many rows its channel sent there;
then it copies, from our heap into the outputs, the rows the same channel of every other rank sent
us, as soon as that rank's count is in. With TDM and enough channels (``dispatch_v2_recv_warps``)
the copy overlaps the send: a sender also raises each row's flag on the peer once the row has
landed, and one warp per source rank of every block copies that source's rows as their flags come
up, while the block's other warps send; the block copies whatever is left after its send.

Combine. The rows ``x`` of the dispatch's receive order go back to their source rank, each with
its top-k weights, into a compact region laid out as [destination rank][index in the source's
stream to it], so the slot of (rank ``r``, token ``t``) on the source is
``base_r + send_head[t, r]``; our own rows stay in ``x``. Once a row has landed its pusher raises
the slot's flag to this combine's epoch. Each rank sums the rows of its own tokens (bias first,
then ranks in ascending order, as the HIP and V1 kernels add them) as soon as their flags are up:
warps claim tokens in order from a queue, some from the start of the kernel and the pushers once
they are done, so the reduce overlaps the push and no rank waits for all the others.

On gfx1250 the rows move through LDS with the Tensor Data Mover (one load, a store per
destination); elsewhere a warp copies them through VGPRs.

Ordering is the V1 cheap fence: the heap is uncached, payload stores carry SCOPE_SYS
(``SYS_STORE``), and a warp waits for its own stores before it publishes a count or raises a flag;
the readers load with ``SYS_LOAD``.

Heap layout (byte offsets, the same on every rank): ``[0, META_BYTES)`` the notify's scratch, then
the received channel offsets, the dispatch row counts, the combine's and the dispatch's row flags,
then from ``DATA_OFFSET`` the dispatch region in the first half of the rest and the combine region
in the second half.
"""

import functools
import os

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm, vector
from flydsl.compiler.ast_rewriter import ASTRewriter
from flydsl.expr import arith, range_constexpr, rocdl
from flydsl.expr.rocdl import tdm_ops

from primus_turbo.flydsl.deep_ep.arch import TRAITS
from primus_turbo.flydsl.deep_ep.common import (
    IS_GFX1250,
    SYS_LOAD,
    SYS_STORE,
    WARP_SIZE,
    ld_lds,
    lds_base,
    load_i32,
    load_index,
    read_clock,
    st_lds,
    tensor_addr,
    timed_out,
    timeout_hit,
    wait_mem_all,
)
from primus_turbo.flydsl.deep_ep.intranode_kernel import (
    NOTIFY_THREADS,
    NUM_THREADS,
    NUM_WARPS,
    _addr,
    _channel_task_range,
    _copy_row,
    _ld_sys,
    _load_lane_ptr,
    _load_ptr,
    _load_u8,
    _st_sys,
    _uniform_i64,
    barrier_block,
)
from primus_turbo.flydsl.utils.buffer_ops import (
    buffer_load,
    buffer_store,
    create_buffer_resource,
    create_buffer_resource_from_addr,
)
from primus_turbo.flydsl.utils.prims import (
    _i64,
    _readfirstlane_i32,
    _readlane_i32,
    atomic_add,
)

META_BYTES = 64 * 1024
# int32 [num_ranks][num_channels]: where each source's channel starts in its receive block
RECV_CHANNEL_OFFSET_OFFSET = META_BYTES + 4096
MAX_CHANNELS = 1024
# int32 [num_ranks][num_channels]: how far each source's channel has sent its rows here
TAILS_OFFSET = 128 * 1024
# int32 per combine slot: the epoch of the combine whose row has landed there
COMBINE_FLAGS_OFFSET = 256 * 1024
COMBINE_FLAGS_BYTES = 4 << 20
# i64 per dispatch receive row, ``call << 15 | source rank << 10 | channel`` once the row and its
# metadata have landed; zeroed with the rest of the metadata, so no value of an earlier process or
# layout matches, and the source and channel keep a stream from taking the next stream's rows
DISPATCH_FLAGS_OFFSET = COMBINE_FLAGS_OFFSET + COMBINE_FLAGS_BYTES
DISPATCH_FLAGS_BYTES = 4 << 20
DISPATCH_FLAG_ROWS = DISPATCH_FLAGS_BYTES // 8
DATA_OFFSET = DISPATCH_FLAGS_OFFSET + DISPATCH_FLAGS_BYTES
assert RECV_CHANNEL_OFFSET_OFFSET + 8 * MAX_CHANNELS * 4 <= TAILS_OFFSET
assert TAILS_OFFSET + 8 * MAX_CHANNELS * 4 <= 192 * 1024
# PRIMUS_TURBO_DEEPEP_V2_PHASE_TIMING=1: thread 0 of every block stores realtime-counter stamps (100 MHz)
# into this rank's heap at PHASE_TIMING_OFFSET, int64 [kernel][block][PHASE_EVENTS], dispatch then
# combine; read once per process, so give such a run its own FLYDSL_RUNTIME_CACHE_DIR. The combine
# then waits for all its warps before its last stamp. Off, no code is emitted.
PHASE_TIMING = os.environ.get("PRIMUS_TURBO_DEEPEP_V2_PHASE_TIMING", "0") == "1"
PHASE_TIMING_OFFSET = 192 * 1024
PHASE_EVENTS = 4
assert PHASE_TIMING_OFFSET + 2 * MAX_CHANNELS * PHASE_EVENTS * 8 <= COMBINE_FLAGS_OFFSET

# a warp loads at most this much of a row before storing it, which bounds the VGPRs of a copy
FANOUT_PIECE_BYTES = 8 * 1024
# rows move global -> LDS -> global through the Tensor Data Mover: a warp copy of the row
# through VGPRs caps how much each CU keeps in flight, and with it the bandwidth per CU
USE_TDM = IS_GFX1250
# LDS a block can allocate on gfx1250
MAX_LDS_BYTES = 320 * 1024
# combine: tokens a warp claims from the reduce queue at a time. A claim waits for all its tokens' rows,
# so one late row holds up the claim: one token per claim takes the reduce tail after the last push from
# 34 to 12 us at EP2 / 160 CUs (PRIMUS_TURBO_DEEPEP_V2_REDUCE_GRAB_TOKENS for experiments;
# REDUCE_GRAB_TOKENS x num_ranks lanes must fit a wave)
REDUCE_GRAB_TOKENS = int(os.environ.get("PRIMUS_TURBO_DEEPEP_V2_REDUCE_GRAB_TOKENS", "1"))
# dispatch: from this many channels on (PRIMUS_TURBO_DEEPEP_V2_DISPATCH_RECV_MIN_CHANNELS for experiments;
# 64 is a first guess), num_ranks - 1 warps of each block receive while the others send: the
# per-CU send rate bounds a narrower grid, so there every warp sends
DISPATCH_RECV_MIN_CHANNELS = int(os.environ.get("PRIMUS_TURBO_DEEPEP_V2_DISPATCH_RECV_MIN_CHANNELS", "64"))


def dispatch_v2_recv_warps(num_ranks, num_channels):
    """Warps of each dispatch block that copy the channel's received rows out of the heap while the
    rest of the block sends, one per source rank; 0: the block copies them after its send. With TDM
    only: the VGPR copy overlaps a row's stores with the next row's loads, which the per-row flag
    would serialize."""
    if not USE_TDM or num_ranks < 2 or num_channels < DISPATCH_RECV_MIN_CHANNELS:
        return 0
    return num_ranks - 1


def _tdm_row(lds_addr, global_addr, num_bytes):
    """TDM descriptor moving one ``num_bytes`` row between LDS ``lds_addr`` (i32) and global
    ``global_addr`` (i64), as a 1 x n tile of 8-byte elements; both must be wave-uniform."""
    elems = num_bytes // 8
    assert num_bytes % 8 == 0 and elems < (1 << 16)
    i32 = fx.T.i32
    addr = fx.arith.ArithValue(fx.as_ir_value(global_addr), signed=False)
    lo = fx.arith.trunci(i32, fx.as_ir_value(addr))
    hi = fx.arith.ArithValue(fx.arith.trunci(i32, fx.as_ir_value(addr >> fx.Int64(32))), signed=False)
    hi = hi | fx.Int32(-(1 << 31))  # descriptor type 2 in [31:30]
    group0 = vector.from_elements(
        fx.T.vec(4, i32),
        [fx.as_ir_value(v) for v in (fx.Int32(1), lds_addr, lo, hi)],
    )
    words = (
        3 << 16,  # 8-byte elements
        (elems & 0xFFFF) << 16,  # tensor dim0 [15:0]
        (elems >> 16) | (1 << 16),  # tensor dim0 [31:16], tensor dim1 = 1
        elems << 16,  # tile dim0
        1,  # tile dim1
        elems,  # dim0 stride
        0,
        0,
    )
    group1 = vector.from_elements(fx.T.vec(8, i32), [fx.as_ir_value(fx.Int32(w)) for w in words])
    return tdm_ops.TDMDescriptor2D(dgroup0=group0, dgroup1=group1)


def _stamp(local, kernel: int, block, event: int):
    """One phase stamp (see PHASE_TIMING); the caller picks the thread."""
    slot = fx.Int32((kernel * MAX_CHANNELS) * PHASE_EVENTS + event) + block * fx.Int32(PHASE_EVENTS)
    _st_sys(local + fx.Int64(PHASE_TIMING_OFFSET), slot, read_clock())


def _align(value, alignment):
    return (value + alignment - 1) // alignment * alignment


def _regions(first_offset, capacity, regions):
    layout = {}
    offset = first_offset
    for name, slot_bytes in regions:
        layout[name] = offset
        offset = _align(offset + capacity * slot_bytes, 16)
    return layout, offset


def _half_bytes(num_nvl_bytes):
    return (num_nvl_bytes - DATA_OFFSET) // 2 // 4096 * 4096


@functools.lru_cache(maxsize=None)
def dispatch_v2_layout(num_nvl_bytes, hidden_bytes, num_topk, topk_idx_bytes, num_scales):
    """``({region: offset}, capacity in tokens)`` of the dispatch receive region."""
    half = _half_bytes(num_nvl_bytes)
    regions = (
        ("x", hidden_bytes),
        ("src_idx", 4),
        ("topk_idx", num_topk * topk_idx_bytes),
        ("topk_weights", num_topk * 4),
        ("x_scales", num_scales * 4),
    )
    capacity = max(0, (half - 16 * len(regions)) // sum(b for _, b in regions))
    layout, end = _regions(DATA_OFFSET, capacity, regions)
    assert end <= DATA_OFFSET + half
    return layout, capacity


def combine_v2_slot_bytes(hidden_bytes, num_topk):
    """A combine slot: the row, then its top-k weights, so one TDM store moves both; slots start on
    128-byte boundaries, as TDM stores want."""
    return _align(hidden_bytes + num_topk * 4, 128)


@functools.lru_cache(maxsize=None)
def combine_v2_layout(num_nvl_bytes, hidden_bytes, num_topk):
    """``({region: offset}, capacity in rows)`` of the combine receive region."""
    half = _half_bytes(num_nvl_bytes)
    regions = (("x", combine_v2_slot_bytes(hidden_bytes, num_topk)),)
    capacity = max(0, (half - 16 * len(regions)) // sum(b for _, b in regions))
    layout, end = _regions(DATA_OFFSET + half, capacity, regions)
    assert end <= DATA_OFFSET + 2 * half
    return layout, capacity


# ── Wave helpers ─────────────────────────────────────────────────────────────


def _ballot(pred):
    """Lanes where ``pred`` holds, as a WARP_SIZE-bit mask."""
    mask_type = ir.IntegerType.get_signless(WARP_SIZE)
    return fx.arith.ArithValue(rocdl.ballot(mask_type, fx.as_ir_value(pred)), signed=False)


def _popc(mask):
    count = fx.arith.ArithValue(llvm.intr_ctpop(fx.as_ir_value(mask)), signed=False)
    if WARP_SIZE == 64:
        count = fx.arith.ArithValue(fx.arith.trunci(fx.T.i32, fx.as_ir_value(count)), signed=True)
    return fx.arith.ArithValue(fx.as_ir_value(count), signed=True)


def _lanes_below(mask):
    """Set bits of ``mask`` in the lanes below this one."""
    i32 = fx.Int32.ir_type
    if WARP_SIZE == 32:
        return fx.arith.ArithValue(
            rocdl.mbcnt_lo(i32, fx.as_ir_value(mask), fx.Int32(0).ir_value()), signed=True
        )
    m = fx.arith.ArithValue(fx.as_ir_value(mask), signed=False)
    lo = fx.arith.trunci(fx.T.i32, fx.as_ir_value(m))
    hi = fx.arith.trunci(fx.T.i32, fx.as_ir_value(m >> fx.Int64(32)))
    count = rocdl.mbcnt_lo(i32, lo, fx.Int32(0).ir_value())
    return fx.arith.ArithValue(rocdl.mbcnt_hi(i32, hi, count), signed=True)


def _signed(value):
    return fx.arith.ArithValue(fx.as_ir_value(value), signed=True)


def _uniform_load_i32(res, idx):
    return _readfirstlane_i32(load_i32(res, idx))


def _rank_prefix(rank_prefix_res, num_ranks: int, s: int, r):
    """``rank_prefix[s][r]``, the tokens of ranks ``0..s`` that rank ``r`` receives; 0 for ``s < 0``."""
    if s < 0:
        return fx.Int32(0)
    return _uniform_load_i32(rank_prefix_res, fx.Int32(s * num_ranks) + r)


def _fanout_row(dst_resources, src_res, num_bytes):
    """Warp copy of one row from ``src_res`` to every resource of ``dst_resources``, each load
    issued once. A resource with no records drops its stores, which is how a caller skips a rank."""
    lane_word = (fx.thread_idx.x % fx.Int32(WARP_SIZE)) * fx.Int32(4)
    num_words = num_bytes // 4
    step_words = WARP_SIZE * 4
    piece_words = FANOUT_PIECE_BYTES // 4
    for piece in range_constexpr(0, num_words, piece_words):
        piece_end = min(piece + piece_words, num_words)
        offsets = []
        for word in range_constexpr(piece, piece_end, step_words):
            offset = fx.Int32(word) + lane_word
            if word + step_words > piece_end:
                # lanes past the row read 0 and store nothing
                offset = arith.select(offset < fx.Int32(piece_end), offset, fx.Int32(num_words))
            offsets.append(offset)
        values = [buffer_load(src_res, o, vec_width=4, dtype=fx.T.i32) for o in offsets]
        for dst in dst_resources:
            for o, v in zip(offsets, values):
                buffer_store(v, dst, o, cache_modifier=SYS_STORE)


# ── dispatch ─────────────────────────────────────────────────────────────────

# what a (source rank, channel) stream publishes to its destination: ``epoch << 20 | done << 19 |
# rows sent so far``; the epoch tells this dispatch's values from a previous one's
TAIL_EPOCH_SHIFT = 20
TAIL_DONE_BIT = 1 << 19
TAIL_ROWS_MASK = TAIL_DONE_BIT - 1
MAX_EPOCH = (1 << 11) - 1


@ASTRewriter.transform
def _poll_tails(
    lds,
    lds_recv: int,
    num_ranks: int,
    num_channels: int,
    channel,
    local,
    rank,
    epoch,
    rank_prefix_res,
    recv_channel_offset_res,
):
    """Thread ``s`` reads how far source ``s``'s stream of this channel has got; the first time, also
    where that stream starts in our receive order."""
    thread_id = fx.thread_idx.x
    if (thread_id < fx.Int32(num_ranks)) & (thread_id != rank):
        idx = thread_id * fx.Int32(num_channels) + channel
        value = _ld_sys(local + fx.Int64(TAILS_OFFSET), idx)
        if (value >> fx.Int32(TAIL_EPOCH_SHIFT)) == epoch:
            if ld_lds(lds, fx.Int32(lds_recv + 3 * num_ranks) + thread_id) < fx.Int32(0):
                offset = _ld_sys(local + fx.Int64(RECV_CHANNEL_OFFSET_OFFSET), idx)
                prev = load_i32(
                    rank_prefix_res,
                    arith.select(thread_id > fx.Int32(0), thread_id - fx.Int32(1), fx.Int32(0))
                    * fx.Int32(num_ranks)
                    + rank,
                )
                prev = arith.select(thread_id > fx.Int32(0), prev, fx.Int32(0))
                st_lds(lds, fx.Int32(lds_recv + 3 * num_ranks) + thread_id, prev + offset)
                buffer_store(offset, recv_channel_offset_res, idx)
            # the receive warps may have copied rows of the stream before its count came in
            rows = value & fx.Int32(TAIL_ROWS_MASK)
            copied = ld_lds(lds, fx.Int32(lds_recv) + thread_id)
            st_lds(lds, fx.Int32(lds_recv + num_ranks) + thread_id, arith.select(rows > copied, rows, copied))
            st_lds(
                lds,
                fx.Int32(lds_recv + 2 * num_ranks) + thread_id,
                arith.select((value & fx.Int32(TAIL_DONE_BIT)) != fx.Int32(0), fx.Int32(1), fx.Int32(0)),
            )


@ASTRewriter.transform
def _copy_received_row(
    local,
    layout,
    outputs,
    row,
    hidden_bytes: int,
    num_topk: int,
    num_scales: int,
    idx_dtype,
    row_lds,
    row_buffer_bytes: int,
):
    """The warp copies received row ``row``, with its metadata, from the heap into the outputs."""
    lane_id = fx.thread_idx.x % fx.Int32(WARP_SIZE)
    if fx.const_expr(row_lds is not None):
        for p in range_constexpr(0, hidden_bytes, row_buffer_bytes):
            piece = min(row_buffer_bytes, hidden_bytes - p)
            tdm_ops.tensor_wait(0)
            tdm_ops.tensor_load_2d(
                _tdm_row(row_lds, _uniform_i64(_addr(local, layout["x"] + p, row, hidden_bytes)), piece),
                cache_policy=SYS_LOAD,
            )
            tdm_ops.tensor_wait(0)
            tdm_ops.tensor_store_2d(
                _tdm_row(row_lds, _uniform_i64(_addr(outputs["x"], p, row, hidden_bytes)), piece)
            )
    else:
        _copy_row(
            _addr(outputs["x"], 0, row, hidden_bytes),
            _addr(local, layout["x"], row, hidden_bytes),
            hidden_bytes,
            FANOUT_PIECE_BYTES,
            load_cache_modifier=SYS_LOAD,
        )
    if lane_id == fx.Int32(0):
        _st_sys(outputs["src_idx"], row, _ld_sys(local + fx.Int64(layout["src_idx"]), row))
    if fx.const_expr(num_topk > 0):
        if lane_id < fx.Int32(num_topk):
            elem = row * fx.Int32(num_topk) + lane_id
            _st_sys(
                outputs["topk_idx"],
                elem,
                _ld_sys(local + fx.Int64(layout["topk_idx"]), elem, dtype=idx_dtype),
            )
            _st_sys(
                outputs["topk_weights"],
                elem,
                _ld_sys(local + fx.Int64(layout["topk_weights"]), elem, dtype=fx.T.f32),
            )
    if fx.const_expr(num_scales > 0):
        for u in range_constexpr((num_scales + WARP_SIZE - 1) // WARP_SIZE):
            scale_idx = lane_id + fx.Int32(u * WARP_SIZE)
            if scale_idx < fx.Int32(num_scales):
                elem = row * fx.Int32(num_scales) + scale_idx
                _st_sys(outputs["x_scales"], elem, _ld_sys(local + fx.Int64(layout["x_scales"]), elem))


@ASTRewriter.transform
def _copy_received(
    lds,
    lds_recv: int,
    num_ranks: int,
    local,
    layout,
    outputs,
    hidden_bytes: int,
    num_topk: int,
    num_scales: int,
    idx_dtype,
    row_lds,
    row_buffer_bytes: int,
):
    """The block copies the rows that have arrived and it has not copied yet, with their metadata,
    from the heap into the outputs."""
    thread_id = fx.thread_idx.x
    warp_id = thread_id // fx.Int32(WARP_SIZE)
    cums = []
    pendings = []
    starts = []
    copieds = []
    total = fx.Int32(0)
    for s in range_constexpr(num_ranks):
        copied = ld_lds(lds, fx.Int32(lds_recv + s))
        pending = ld_lds(lds, fx.Int32(lds_recv + num_ranks + s)) - copied
        cums.append(total)
        pendings.append(pending)
        starts.append(ld_lds(lds, fx.Int32(lds_recv + 3 * num_ranks + s)) + copied)
        copieds.append(copied)
        total = total + pending
    j = warp_id
    while j < total:
        row = starts[0] + j
        for s in range_constexpr(1, num_ranks):
            row = arith.select(j >= cums[s], starts[s] + j - cums[s], row)
        _copy_received_row(
            local,
            layout,
            outputs,
            row,
            hidden_bytes,
            num_topk,
            num_scales,
            idx_dtype,
            row_lds,
            row_buffer_bytes,
        )
        j = j + fx.Int32(NUM_WARPS)
    fx.gpu.barrier()
    if thread_id < fx.Int32(num_ranks):
        arrived = pendings[0] + copieds[0]
        for s in range_constexpr(1, num_ranks):
            arrived = arith.select(thread_id == fx.Int32(s), pendings[s] + copieds[s], arrived)
        st_lds(lds, fx.Int32(lds_recv) + thread_id, arrived)
    fx.gpu.barrier()


@ASTRewriter.transform
def _receive_while_sending(
    lds,
    lds_recv: int,
    lds_send_done: int,
    num_ranks: int,
    num_channels: int,
    send_warps: int,
    is_receiver,
    source,
    tile_no,
    channel,
    local,
    rank,
    epoch,
    flag,
    rank_prefix_res,
    recv_channel_offset_res,
    layout,
    outputs,
    hidden_bytes: int,
    num_topk: int,
    num_scales: int,
    idx_dtype,
    row_lds,
    row_buffer_bytes: int,
):
    """A receive warp copies, in order, the rows of ``source``'s stream of this channel whose flags
    read ``flag``, until every send warp of the block is done with tile ``tile_no``; the block's
    receive loop takes the rest from the count it leaves in LDS."""
    lane_id = fx.thread_idx.x % fx.Int32(WARP_SIZE)
    idx = source * fx.Int32(num_channels) + channel
    lds_copied = fx.Int32(lds_recv) + source
    lds_first = fx.Int32(lds_recv + 3 * num_ranks) + source
    flags = local + fx.Int64(DISPATCH_FLAGS_OFFSET)
    prev = _uniform_load_i32(
        rank_prefix_res,
        arith.select(source > fx.Int32(0), (source - fx.Int32(1)) * fx.Int32(num_ranks), fx.Int32(0)) + rank,
    )
    prev = arith.select(source > fx.Int32(0), prev, fx.Int32(0))
    copied = _readfirstlane_i32(ld_lds(lds, lds_copied))
    running = arith.select(is_receiver, fx.Int32(1), fx.Int32(0))
    while running > fx.Int32(0):
        done = ld_lds(
            lds, fx.Int32(lds_send_done) + arith.select(lane_id < fx.Int32(send_warps), lane_id, fx.Int32(0))
        )
        senders_busy = _popc(_ballot((lane_id < fx.Int32(send_warps)) & (done != tile_no)))
        # where the stream starts, from the count its sender publishes before its first row
        if (lane_id == fx.Int32(0)) & (ld_lds(lds, lds_first) < fx.Int32(0)):
            value = _ld_sys(local + fx.Int64(TAILS_OFFSET), idx)
            if (value >> fx.Int32(TAIL_EPOCH_SHIFT)) == epoch:
                offset = _ld_sys(local + fx.Int64(RECV_CHANNEL_OFFSET_OFFSET), idx)
                st_lds(lds, lds_first, prev + offset)
                buffer_store(offset, recv_channel_offset_res, idx)
        first = _readfirstlane_i32(ld_lds(lds, lds_first))
        landed = _ld_sys(
            flags, arith.select(first >= fx.Int32(0), first + copied, fx.Int32(0)), dtype=fx.T.i64
        )
        ready = _readfirstlane_i32(
            arith.select(
                (senders_busy > fx.Int32(0)) & (first >= fx.Int32(0)) & (landed == flag),
                fx.Int32(1),
                fx.Int32(0),
            )
        )
        if ready > fx.Int32(0):
            _copy_received_row(
                local,
                layout,
                outputs,
                first + copied,
                hidden_bytes,
                num_topk,
                num_scales,
                idx_dtype,
                row_lds,
                row_buffer_bytes,
            )
        copied = copied + ready
        if (ready == fx.Int32(0)) & (senders_busy > fx.Int32(0)):
            fx.rocdl.s_sleep(fx.Int32(1))
        running = arith.select(senders_busy > fx.Int32(0), fx.Int32(1), fx.Int32(0))
    if is_receiver & (lane_id == fx.Int32(0)):
        st_lds(lds, lds_copied, copied)


def _make_dispatch_v2(
    num_ranks,
    num_channels,
    hidden_bytes,
    num_topk,
    num_experts,
    num_scales,
    topk_idx_bytes,
    num_nvl_bytes,
):
    assert num_ranks <= WARP_SIZE and num_topk <= WARP_SIZE and num_channels <= MAX_CHANNELS
    assert WARP_SIZE % NUM_WARPS == 0
    num_experts_per_rank = num_experts // num_ranks if num_topk > 0 else 0
    layout, _ = dispatch_v2_layout(num_nvl_bytes, hidden_bytes, num_topk, topk_idx_bytes, num_scales)
    tile = NUM_THREADS
    # LDS: [num_ranks] running stream indices, [num_ranks] the channel's first stream indices,
    # [NUM_WARPS][num_ranks] warp bases, [tile][num_ranks] slots, and per source rank its received
    # stream of this channel: [num_ranks] rows copied, rows arrived, done, first row (-1: unknown)
    lds_running = 0
    lds_chan_start = num_ranks
    lds_warp = 2 * num_ranks
    lds_slot = lds_warp + NUM_WARPS * num_ranks
    lds_recv = lds_slot + tile * num_ranks
    # [NUM_WARPS] the last tile each send warp is done with, for the receive warps
    lds_send_done = lds_recv + 4 * num_ranks
    lds_words = lds_send_done + NUM_WARPS
    recv_warps = dispatch_v2_recv_warps(num_ranks, num_channels)
    send_warps = NUM_WARPS - recv_warps
    # with TDM, a row buffer per warp; a row that does not fit moves through it in pieces
    lds_rows = _align(lds_words * 4, 128)
    row_buffer_bytes = min(_align(hidden_bytes, 128), (MAX_LDS_BYTES - lds_rows) // NUM_WARPS // 128 * 128)
    if USE_TDM:
        lds_words = (lds_rows + NUM_WARPS * row_buffer_bytes) // 4
    grid_threads = num_channels * NUM_THREADS
    idx_dtype = fx.T.i64 if topk_idx_bytes == 8 else fx.T.i32

    @flyc.kernel(known_block_size=[NUM_THREADS, 1, 1])
    def dispatch_v2_kernel(
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
        RANK_PREFIX_MATRIX: fx.Tensor,
        BUFFER_PTRS: fx.Tensor,
        num_tokens: fx.Int32,
        num_worst_tokens: fx.Int32,
        scale_token_stride: fx.Int32,
        scale_hidden_stride: fx.Int32,
        rank: fx.Int32,
        epoch: fx.Int32,
        flag_epoch: fx.Int32,
    ):
        thread_id = fx.thread_idx.x
        channel = fx.block_idx.x
        lane_id = thread_id % fx.Int32(WARP_SIZE)
        warp_id = thread_id // fx.Int32(WARP_SIZE)
        lds = lds_base()
        buffer_ptrs_res = create_buffer_resource(BUFFER_PTRS, max_size=True)
        local = _load_ptr(buffer_ptrs_res, rank)
        # the flag of a row of this call's stream from ``source`` in this channel
        flag_call = _i64(flag_epoch) << fx.Int64(15)
        in_rank_res = create_buffer_resource(IS_TOKEN_IN_RANK, max_size=True)
        send_head_res = create_buffer_resource(SEND_HEAD, max_size=True)
        rank_prefix_res = create_buffer_resource(RANK_PREFIX_MATRIX, max_size=True)
        recv_channel_offset_res = create_buffer_resource(RECV_CHANNEL_OFFSET, max_size=True)
        x_base = tensor_addr(X)
        token_start, token_end = _channel_task_range(num_tokens, num_channels, channel)
        if fx.const_expr(PHASE_TIMING):
            if thread_id == fx.Int32(0):
                _stamp(local, 0, channel, 0)

        # the channel's first index in our stream to each rank, published to that rank before any
        # of its rows; and nothing received yet (our own rows go to the outputs directly)
        if thread_id < fx.Int32(num_ranks):
            prefix_res = create_buffer_resource(CHANNEL_PREFIX_MATRIX, max_size=True)
            prev_channel = arith.select(channel > fx.Int32(0), channel - fx.Int32(1), fx.Int32(0))
            start = load_i32(prefix_res, thread_id * fx.Int32(num_channels) + prev_channel)
            start = arith.select(channel > fx.Int32(0), start, fx.Int32(0))
            st_lds(lds, fx.Int32(lds_running) + thread_id, start)
            st_lds(lds, fx.Int32(lds_chan_start) + thread_id, start)
            idx = rank * fx.Int32(num_channels) + channel
            if thread_id == rank:
                buffer_store(start, recv_channel_offset_res, idx)
            else:
                peer_heap = _load_lane_ptr(buffer_ptrs_res, thread_id)
                _st_sys(peer_heap + fx.Int64(RECV_CHANNEL_OFFSET_OFFSET), idx, start)
                if fx.const_expr(recv_warps > 0):
                    # no rows yet: the receive warps learn where the stream starts before its first row
                    wait_mem_all()
                    _st_sys(peer_heap + fx.Int64(TAILS_OFFSET), idx, epoch << fx.Int32(TAIL_EPOCH_SHIFT))
            st_lds(lds, fx.Int32(lds_recv) + thread_id, fx.Int32(0))
            st_lds(lds, fx.Int32(lds_recv + num_ranks) + thread_id, fx.Int32(0))
            st_lds(
                lds,
                fx.Int32(lds_recv + 2 * num_ranks) + thread_id,
                arith.select(thread_id == rank, fx.Int32(1), fx.Int32(0)),
            )
            st_lds(lds, fx.Int32(lds_recv + 3 * num_ranks) + thread_id, fx.Int32(-1))
        if fx.const_expr(recv_warps > 0):
            if thread_id < fx.Int32(NUM_WARPS):
                st_lds(lds, fx.Int32(lds_send_done) + thread_id, fx.Int32(0))
        fx.gpu.barrier()

        # per destination: where our rank's tokens start in its receive order, and where each
        # output goes -- its heap, or for our own rank the output tensors themselves
        rank_offsets = []
        dst_flags = [
            _load_ptr(buffer_ptrs_res, fx.Int32(d)) + fx.Int64(DISPATCH_FLAGS_OFFSET)
            for d in range(num_ranks if recv_warps > 0 else 0)
        ]
        dst_bases = {name: [] for name in ("x", "src_idx", "topk_idx", "topk_weights", "x_scales")}
        outputs = {
            "x": tensor_addr(RECV_X),
            "src_idx": tensor_addr(RECV_SRC_IDX),
            "topk_idx": tensor_addr(RECV_TOPK_IDX),
            "topk_weights": tensor_addr(RECV_TOPK_WEIGHTS),
            "x_scales": tensor_addr(RECV_X_SCALES),
        }
        for d in range_constexpr(num_ranks):
            peer = _load_ptr(buffer_ptrs_res, fx.Int32(d))
            is_own = rank == fx.Int32(d)
            for name, bases in dst_bases.items():
                bases.append(arith.select(is_own, outputs[name], peer + fx.Int64(layout[name])))
            offset = _uniform_load_i32(
                rank_prefix_res,
                arith.select(rank > fx.Int32(0), (rank - fx.Int32(1)) * fx.Int32(num_ranks), fx.Int32(0))
                + fx.Int32(d),
            )
            rank_offsets.append(arith.select(rank > fx.Int32(0), offset, fx.Int32(0)))
        row_lds = None
        if fx.const_expr(USE_TDM):
            row_lds = (
                fx.arith.ArithValue(lds, signed=True)
                + fx.Int32(lds_rows)
                + _readfirstlane_i32(warp_id) * fx.Int32(row_buffer_bytes)
            )
        tail_base = (epoch << fx.Int32(TAIL_EPOCH_SHIFT)) - ld_lds(lds, fx.Int32(lds_chan_start) + thread_id)

        # receive warp r copies the stream of the r-th other rank
        is_receiver = warp_id >= fx.Int32(send_warps)
        recv_source = warp_id - fx.Int32(send_warps)
        recv_source = arith.select(
            is_receiver,
            recv_source + arith.select(recv_source >= rank, fx.Int32(1), fx.Int32(0)),
            fx.Int32(0),
        )
        tile_no = fx.Int32(0)
        tile_start = token_start
        while tile_start < token_end:
            tile_no = tile_no + fx.Int32(1)
            # ---- the stream index of each token of the tile, per destination ----
            token = tile_start + thread_id
            is_valid = token < token_end
            safe_token = arith.select(is_valid, token, fx.Int32(0))
            flags = []
            belows = []
            for d in range_constexpr(num_ranks):
                flag = is_valid & (
                    _load_u8(in_rank_res, safe_token * fx.Int32(num_ranks) + fx.Int32(d)) != fx.Int32(0)
                )
                mask = _ballot(flag)
                if lane_id == fx.Int32(0):
                    st_lds(lds, fx.Int32(lds_warp + d) + warp_id * fx.Int32(num_ranks), _popc(mask))
                flags.append(flag)
                belows.append(_lanes_below(mask))
            fx.gpu.barrier()
            if thread_id < fx.Int32(num_ranks):
                base = ld_lds(lds, fx.Int32(lds_running) + thread_id)
                for w in range_constexpr(NUM_WARPS):
                    count = ld_lds(lds, fx.Int32(lds_warp + w * num_ranks) + thread_id)
                    st_lds(lds, fx.Int32(lds_warp + w * num_ranks) + thread_id, base)
                    base = base + count
                st_lds(lds, fx.Int32(lds_running) + thread_id, base)
            fx.gpu.barrier()
            for d in range_constexpr(num_ranks):
                warp_base = ld_lds(lds, fx.Int32(lds_warp + d) + warp_id * fx.Int32(num_ranks))
                slot = arith.select(flags[d], warp_base + belows[d], fx.Int32(-1))
                st_lds(lds, fx.Int32(lds_slot + d) + thread_id * fx.Int32(num_ranks), slot)
                if is_valid:
                    buffer_store(slot, send_head_res, token * fx.Int32(num_ranks) + fx.Int32(d))
            fx.gpu.barrier()

            # ---- one warp per token: its row loaded once and stored to every rank it goes to ----
            num_tile_tokens = arith.select(
                token_end - tile_start < fx.Int32(tile), token_end - tile_start, fx.Int32(tile)
            )
            i = arith.select(is_receiver, num_tile_tokens, warp_id)
            while i < num_tile_tokens:
                t = tile_start + i
                rows = []
                routed = []
                for d in range_constexpr(num_ranks):
                    slot = _readfirstlane_i32(ld_lds(lds, fx.Int32(lds_slot + d) + i * fx.Int32(num_ranks)))
                    is_routed = slot >= fx.Int32(0)
                    rows.append(rank_offsets[d] + arith.select(is_routed, slot, fx.Int32(0)))
                    routed.append(is_routed)
                if fx.const_expr(USE_TDM):
                    for p in range_constexpr(0, hidden_bytes, row_buffer_bytes):
                        piece = min(row_buffer_bytes, hidden_bytes - p)
                        # the previous stores are done with the buffer
                        tdm_ops.tensor_wait(0)
                        tdm_ops.tensor_load_2d(
                            _tdm_row(row_lds, _uniform_i64(_addr(x_base, p, t, hidden_bytes)), piece)
                        )
                        tdm_ops.tensor_wait(0)
                        for d in range_constexpr(num_ranks):
                            if routed[d]:
                                tdm_ops.tensor_store_2d(
                                    _tdm_row(
                                        row_lds,
                                        _uniform_i64(_addr(dst_bases["x"][d], p, rows[d], hidden_bytes)),
                                        piece,
                                    ),
                                    cache_policy=SYS_STORE,
                                )
                else:
                    dst_resources = [
                        create_buffer_resource_from_addr(
                            _uniform_i64(_addr(dst_bases["x"][d], 0, rows[d], hidden_bytes)),
                            num_records_bytes=arith.select(routed[d], fx.Int32(hidden_bytes), fx.Int32(0)),
                        )
                        for d in range(num_ranks)
                    ]
                    src_res = create_buffer_resource_from_addr(
                        _uniform_i64(_addr(x_base, 0, t, hidden_bytes)), num_records_bytes=hidden_bytes
                    )
                    _fanout_row(dst_resources, src_res, hidden_bytes)

                for d in range_constexpr(num_ranks):
                    if routed[d]:
                        if lane_id == fx.Int32(0):
                            _st_sys(dst_bases["src_idx"][d], rows[d], t)
                        if fx.const_expr(num_topk > 0):
                            if lane_id < fx.Int32(num_topk):
                                topk_idx_res = create_buffer_resource(TOPK_IDX, max_size=True)
                                topk_weights_res = create_buffer_resource(TOPK_WEIGHTS, max_size=True)
                                elem = t * fx.Int32(num_topk) + lane_id
                                dst_elem = rows[d] * fx.Int32(num_topk) + lane_id
                                expert = load_index(topk_idx_res, elem, TOPK_IDX.element_type)
                                expert_begin = fx.Int32(d * num_experts_per_rank)
                                is_local = (expert >= expert_begin) & (
                                    expert < expert_begin + fx.Int32(num_experts_per_rank)
                                )
                                local_expert = arith.select(is_local, expert - expert_begin, fx.Int32(-1))
                                weight = buffer_load(topk_weights_res, elem, vec_width=1, dtype=fx.T.f32)
                                if fx.const_expr(topk_idx_bytes == 8):
                                    _st_sys(dst_bases["topk_idx"][d], dst_elem, _i64(local_expert))
                                else:
                                    _st_sys(dst_bases["topk_idx"][d], dst_elem, local_expert)
                                _st_sys(
                                    dst_bases["topk_weights"][d],
                                    dst_elem,
                                    arith.select(is_local, weight, fx.Float32(0.0)),
                                )
                        if fx.const_expr(num_scales > 0):
                            x_scales_res = create_buffer_resource(X_SCALES, max_size=True)
                            for u in range_constexpr((num_scales + WARP_SIZE - 1) // WARP_SIZE):
                                scale_idx = lane_id + fx.Int32(u * WARP_SIZE)
                                if scale_idx < fx.Int32(num_scales):
                                    value = load_i32(
                                        x_scales_res,
                                        t * scale_token_stride + scale_idx * scale_hidden_stride,
                                    )
                                    _st_sys(
                                        dst_bases["x_scales"][d],
                                        rows[d] * fx.Int32(num_scales) + scale_idx,
                                        value,
                                    )
                if fx.const_expr(recv_warps > 0):
                    # the row and its metadata have landed on every peer: raise its flags there. The
                    # next row's load waits for these stores anyway (one row buffer per warp)
                    if fx.const_expr(USE_TDM):
                        tdm_ops.tensor_wait(0)
                    wait_mem_all()
                    if lane_id == fx.Int32(0):
                        for d in range_constexpr(num_ranks):
                            if routed[d] & (rank != fx.Int32(d)):
                                _st_sys(
                                    dst_flags[d],
                                    rows[d],
                                    flag_call | _i64(rank * fx.Int32(1 << 10) + channel),
                                )
                i = i + fx.Int32(send_warps)
            if fx.const_expr(recv_warps > 0):
                if (lane_id == fx.Int32(0)) & (warp_id < fx.Int32(send_warps)):
                    st_lds(lds, fx.Int32(lds_send_done) + warp_id, tile_no)
                _receive_while_sending(
                    lds,
                    lds_recv,
                    lds_send_done,
                    num_ranks,
                    num_channels,
                    send_warps,
                    is_receiver,
                    recv_source,
                    tile_no,
                    channel,
                    local,
                    rank,
                    epoch,
                    flag_call | _i64(recv_source * fx.Int32(1 << 10) + channel),
                    rank_prefix_res,
                    recv_channel_offset_res,
                    layout,
                    outputs,
                    hidden_bytes,
                    num_topk,
                    num_scales,
                    idx_dtype,
                    row_lds,
                    row_buffer_bytes,
                )
            tile_start = tile_start + fx.Int32(tile)

        # every warp's rows are visible: our streams are done
        if fx.const_expr(USE_TDM):
            tdm_ops.tensor_wait(0)
        wait_mem_all()
        fx.gpu.barrier()
        if fx.const_expr(PHASE_TIMING):
            if thread_id == fx.Int32(0):
                _stamp(local, 0, channel, 1)
        if (thread_id < fx.Int32(num_ranks)) & (thread_id != rank):
            _st_sys(
                _load_lane_ptr(buffer_ptrs_res, thread_id) + fx.Int64(TAILS_OFFSET),
                rank * fx.Int32(num_channels) + channel,
                tail_base + fx.Int32(TAIL_DONE_BIT) + ld_lds(lds, fx.Int32(lds_running) + thread_id),
            )

        # ---- the rest of what this channel receives ----
        spin_start = read_clock()
        remaining = fx.Int32(1)
        while remaining > fx.Int32(0):
            _poll_tails(
                lds,
                lds_recv,
                num_ranks,
                num_channels,
                channel,
                local,
                rank,
                epoch,
                rank_prefix_res,
                recv_channel_offset_res,
            )
            fx.gpu.barrier()
            _copy_received(
                lds,
                lds_recv,
                num_ranks,
                local,
                layout,
                outputs,
                hidden_bytes,
                num_topk,
                num_scales,
                idx_dtype,
                row_lds,
                row_buffer_bytes,
            )
            remaining = fx.Int32(0)
            for s in range_constexpr(num_ranks):
                remaining = (
                    remaining
                    + fx.Int32(1)
                    - ld_lds(lds, fx.Int32(lds_recv + 2 * num_ranks + s))
                    + ld_lds(lds, fx.Int32(lds_recv + num_ranks + s))
                    - ld_lds(lds, fx.Int32(lds_recv + s))
                )
            if (thread_id == fx.Int32(0)) & timed_out(spin_start):
                fx.printf(
                    "[DeepEP FlyDSL V2 dispatch] receive timeout: rank={} channel={} remaining={}\n",
                    rank,
                    channel,
                    remaining,
                )
                timeout_hit()
                spin_start = read_clock()
            # everyone has read the counts before the next poll rewrites them
            fx.gpu.barrier()
        if fx.const_expr(USE_TDM):
            tdm_ops.tensor_wait(0)
        if fx.const_expr(PHASE_TIMING):
            if thread_id == fx.Int32(0):
                _stamp(local, 0, channel, 2)

        # with a worst-case allocation, the rows past the received ones read as unrouted
        if fx.const_expr(num_topk > 0):
            if num_worst_tokens > fx.Int32(0):
                recv_topk_idx_res = create_buffer_resource(RECV_TOPK_IDX, max_size=True)
                num_recv_tokens = _uniform_load_i32(
                    rank_prefix_res, fx.Int32((num_ranks - 1) * num_ranks) + rank
                )
                i = num_recv_tokens * fx.Int32(num_topk) + channel * fx.Int32(NUM_THREADS) + thread_id
                while i < num_worst_tokens * fx.Int32(num_topk):
                    if fx.const_expr(topk_idx_bytes == 8):
                        buffer_store(fx.Int64(-1), recv_topk_idx_res, i)
                    else:
                        buffer_store(fx.Int32(-1), recv_topk_idx_res, i)
                    i = i + fx.Int32(grid_threads)

    return dispatch_v2_kernel, lds_words


@flyc.jit
def launch_dispatch_v2(
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
    rank_prefix_matrix,
    buffer_ptrs,
    num_tokens: fx.Int32,
    num_worst_tokens: fx.Int32,
    scale_token_stride: fx.Int32,
    scale_hidden_stride: fx.Int32,
    rank: fx.Int32,
    epoch: fx.Int32,
    flag_epoch: fx.Int32,
    num_ranks: fx.Constexpr[int],
    num_channels: fx.Constexpr[int],
    hidden_bytes: fx.Constexpr[int],
    num_topk: fx.Constexpr[int],
    num_experts: fx.Constexpr[int],
    num_scales: fx.Constexpr[int],
    topk_idx_bytes: fx.Constexpr[int],
    num_nvl_bytes: fx.Constexpr[int],
    stream: fx.Stream,
):
    kernel, lds_words = _make_dispatch_v2(
        num_ranks,
        num_channels,
        hidden_bytes,
        num_topk,
        num_experts,
        num_scales,
        topk_idx_bytes,
        num_nvl_bytes,
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
        rank_prefix_matrix,
        buffer_ptrs,
        num_tokens,
        num_worst_tokens,
        scale_token_stride,
        scale_hidden_stride,
        rank,
        epoch,
        flag_epoch,
    ).launch(grid=(num_channels, 1, 1), block=(NUM_THREADS, 1, 1), stream=stream, smem=lds_words * 4)


# ── barrier ──────────────────────────────────────────────────────────────────


def _make_barrier(num_ranks):
    @flyc.kernel(known_block_size=[NOTIFY_THREADS, 1, 1])
    def barrier_kernel(BARRIER_PTRS: fx.Tensor, rank: fx.Int32):
        barrier_ptrs_res = create_buffer_resource(BARRIER_PTRS, max_size=True)
        barrier_block(barrier_ptrs_res, rank, num_ranks, False, "v2/barrier")

    return barrier_kernel


@flyc.jit
def launch_barrier(barrier_ptrs, rank: fx.Int32, num_ranks: fx.Constexpr[int], stream: fx.Stream):
    _make_barrier(num_ranks)(barrier_ptrs, rank).launch(
        grid=(1, 1, 1), block=(NOTIFY_THREADS, 1, 1), stream=stream
    )


# ── combine ──────────────────────────────────────────────────────────────────


def _wait_lds():
    """Wait for this wave's LDS accesses, e.g. before a TDM store reads what it wrote."""
    llvm.inline_asm(
        res=None, operands_=[], asm_string="s_wait_dscnt 0x0", constraints="~{memory}", has_side_effects=True
    )


@ASTRewriter.transform
def _grab_tokens(counter, lds, lds_slot, base):
    """First of the warp's next ``REDUCE_GRAB_TOKENS`` tokens from the reduce queue."""
    lane_id = fx.thread_idx.x % fx.Int32(WARP_SIZE)
    if lane_id == fx.Int32(0):
        st_lds(lds, lds_slot, atomic_add(counter, fx.Int32(0), fx.Int32(1)))
    return (_readfirstlane_i32(ld_lds(lds, lds_slot)) - base) * fx.Int32(REDUCE_GRAB_TOKENS)


@ASTRewriter.transform
def _wait_rows(flags, flag_idx, waits, epoch, rank, token):
    """Spin until every lane in ``waits`` reads ``epoch`` from its row's flag."""
    spin_start = read_clock()
    value = _ld_sys(flags, flag_idx)
    while _popc(_ballot(waits & (value != epoch))) > fx.Int32(0):
        if timed_out(spin_start):
            fx.printf(
                "[DeepEP FlyDSL V2 combine] row timeout: rank={} token={} flag={} epoch={}\n",
                rank,
                token,
                value,
                epoch,
            )
            timeout_hit()
            spin_start = read_clock()
        value = _ld_sys(flags, flag_idx)


# PRIMUS_TURBO_DEEPEP_V2_COMBINE_PUSH_WARPS=n: n push warps per block at every grid size (experiments)
COMBINE_PUSH_WARPS = int(os.environ.get("PRIMUS_TURBO_DEEPEP_V2_COMBINE_PUSH_WARPS", "0"))
# PRIMUS_TURBO_DEEPEP_V2_COMBINE_PUSH_BLOCK_MAJOR=1: the pushers' rows dealt block-major, as before (experiments)
COMBINE_PUSH_BLOCK_MAJOR = os.environ.get("PRIMUS_TURBO_DEEPEP_V2_COMBINE_PUSH_BLOCK_MAJOR", "0") == "1"


def combine_v2_push_warps(num_channels):
    """Warps of each block that push before they reduce; the others reduce from the start. A grid
    of 96+ blocks saturates the links with about 2/3 of its warps pushing, and the rest overlap
    the reduce with the push; a narrower one is bound by how fast each CU sends, so all push (with
    TDM, LDS caps both at 22 slots for hidden 7168; fewer pushers measured slower at every grid)."""
    if COMBINE_PUSH_WARPS > 0:
        return min(COMBINE_PUSH_WARPS, NUM_WARPS)
    return NUM_WARPS * 11 // 16 if num_channels >= 96 else NUM_WARPS


# combine: with one-token claims, a token's reduce is claimed in column chunks of one load pass each,
# so the queue's last tokens are done one pass, not a whole row, after their rows land
# (PRIMUS_TURBO_DEEPEP_V2_REDUCE_CHUNKS for experiments: 0 = one chunk per pass, n = at most n)
REDUCE_CHUNKS = int(os.environ.get("PRIMUS_TURBO_DEEPEP_V2_REDUCE_CHUNKS", "0"))


def _combine_v2_steps(num_ranks, hidden, elem_bytes, has_bias_0, has_bias_1):
    """(WARP_SIZE-lane steps of 16 bytes in a row, steps a reducer loads per pass)."""
    num_steps = (hidden * elem_bytes // 16 + WARP_SIZE - 1) // WARP_SIZE
    steps_per_pass = max(1, TRAITS.combine_loads_in_flight // (num_ranks + int(has_bias_0) + int(has_bias_1)))
    return num_steps, steps_per_pass


def combine_v2_reduce_chunks(num_ranks, hidden, elem_bytes, has_bias_0, has_bias_1):
    """Column chunks a combine token's reduce is claimed in."""
    if REDUCE_GRAB_TOKENS > 1:
        return 1
    num_steps, steps_per_pass = _combine_v2_steps(num_ranks, hidden, elem_bytes, has_bias_0, has_bias_1)
    passes = (num_steps + steps_per_pass - 1) // steps_per_pass
    return passes if REDUCE_CHUNKS <= 0 else min(REDUCE_CHUNKS, passes)


def combine_v2_grabs(num_recv_tokens, num_channels, chunks=1):
    """How far one combine advances the reduce queue counter: every claim of tokens (or of token
    chunks), plus the one claim per warp that finds the queue empty."""
    items = num_recv_tokens * chunks
    return (items + REDUCE_GRAB_TOKENS - 1) // REDUCE_GRAB_TOKENS + num_channels * NUM_WARPS


def _make_combine_v2(
    num_ranks, num_channels, hidden, elem_bytes, num_topk, has_bias_0, has_bias_1, num_nvl_bytes
):
    assert num_ranks <= WARP_SIZE and num_topk <= WARP_SIZE
    hidden_bytes = hidden * elem_bytes
    assert hidden_bytes % 16 == 0
    layout, _ = combine_v2_layout(num_nvl_bytes, hidden_bytes, num_topk)
    slot_bytes = combine_v2_slot_bytes(hidden_bytes, num_topk)
    weights_offset = layout["x"] + hidden_bytes
    # the row and its weights, in whole 8-byte TDM elements
    push_bytes = _align(hidden_bytes + num_topk * 4, 8)
    push_warps = combine_v2_push_warps(num_channels)
    if USE_TDM:
        # each pusher holds a whole slot in LDS: the weights go with the row's store
        push_warps = min(push_warps, (MAX_LDS_BYTES - NUM_WARPS * 4) // slot_bytes)
    grid_push_warps = num_channels * push_warps
    assert grid_push_warps >= num_ranks and REDUCE_GRAB_TOKENS * num_ranks <= WARP_SIZE
    vec_elems = 16 // elem_bytes
    num_steps, steps_per_pass = _combine_v2_steps(num_ranks, hidden, elem_bytes, has_bias_0, has_bias_1)
    chunks = combine_v2_reduce_chunks(num_ranks, hidden, elem_bytes, has_bias_0, has_bias_1)
    # the steps of one chunk; the last chunk's steps past the row load 0 and store nothing
    chunk_steps = (num_steps + chunks - 1) // chunks
    row_buffer_bytes = slot_bytes
    lds_grab = (push_warps * row_buffer_bytes if USE_TDM else 0) // 4
    smem_bytes = lds_grab * 4 + NUM_WARPS * 4

    @flyc.kernel(known_block_size=[NUM_THREADS, 1, 1])
    def combine_v2_kernel(
        RECV_X: fx.Tensor,
        RECV_TOPK_WEIGHTS: fx.Tensor,
        X: fx.Tensor,
        TOPK_WEIGHTS: fx.Tensor,
        BIAS_0: fx.Tensor,
        BIAS_1: fx.Tensor,
        RANK_PREFIX_MATRIX: fx.Tensor,
        SEND_HEAD: fx.Tensor,
        BUFFER_PTRS: fx.Tensor,
        REDUCE_COUNTER: fx.Tensor,
        num_rows: fx.Int32,
        num_recv_tokens: fx.Int32,
        rank: fx.Int32,
        epoch: fx.Int32,
        grab_base: fx.Int32,
    ):
        thread_id = fx.thread_idx.x
        block_id = fx.block_idx.x
        lane_id = thread_id % fx.Int32(WARP_SIZE)
        warp_id = thread_id // fx.Int32(WARP_SIZE)
        buffer_ptrs_res = create_buffer_resource(BUFFER_PTRS, max_size=True)
        local = _load_ptr(buffer_ptrs_res, rank)
        rank_prefix_res = create_buffer_resource(RANK_PREFIX_MATRIX, max_size=True)
        lds = fx.arith.ArithValue(lds_base(), signed=True)
        if fx.const_expr(PHASE_TIMING):
            if thread_id == fx.Int32(0):
                _stamp(local, 1, block_id, 0)

        # per source rank s: where its rows of x begin, and where our rank's rows start
        # in s's region (the tokens s sends to the ranks before ours)
        peers = []
        row_begins = []
        bases = []
        for s in range_constexpr(num_ranks):
            peers.append(_load_ptr(buffer_ptrs_res, fx.Int32(s)))
            row_begins.append(_rank_prefix(rank_prefix_res, num_ranks, s - 1, rank))
            base = fx.Int32(0)
            for r in range_constexpr(num_ranks):
                count = _rank_prefix(rank_prefix_res, num_ranks, s, fx.Int32(r)) - _rank_prefix(
                    rank_prefix_res, num_ranks, s - 1, fx.Int32(r)
                )
                base = base + arith.select(fx.Int32(r) < rank, count, fx.Int32(0))
            bases.append(base)

        # ---- push (the first push_warps warps of each block): each row of x back to its source
        # rank, but for our own rows, which the reduce reads from x; a row's flag goes up once
        # the row is visible ----
        # rows are grouped by source, so a warp sticks to one source: walking them in order would
        # have the whole GPU writing to one peer, over one link, at a time
        x_base = tensor_addr(X)
        # numbered warp-major, so the last, partial round of rows spreads over every block instead of
        # filling the first ones
        if fx.const_expr(COMBINE_PUSH_BLOCK_MAJOR):
            global_warp = block_id * fx.Int32(push_warps) + warp_id
        else:
            global_warp = warp_id * fx.Int32(num_channels) + block_id
        num_peers = num_ranks - 1
        warps_per_peer = grid_push_warps // num_peers
        warp_source = global_warp % fx.Int32(num_peers)
        warp_source = warp_source + arith.select(warp_source >= rank, fx.Int32(1), fx.Int32(0))
        # the rows actually received, not num_rows: under a worst-case allocation x has
        # num_worst_tokens rows, and pushing its tail would land on other ranks' slots
        rows_received = _rank_prefix(rank_prefix_res, num_ranks, num_ranks - 1, rank)
        peer = peers[0]
        row_begin = row_begins[0]
        row_end = row_begins[1] if num_ranks > 1 else rows_received
        base = bases[0]
        for s in range_constexpr(1, num_ranks):
            is_s = warp_source == fx.Int32(s)
            peer = arith.select(is_s, peers[s], peer)
            row_begin = arith.select(is_s, row_begins[s], row_begin)
            row_end = arith.select(is_s, row_begins[s + 1] if s + 1 < num_ranks else rows_received, row_end)
            base = arith.select(is_s, bases[s], base)
        row = row_begin + global_warp // fx.Int32(num_peers)
        is_pusher = (warp_id < fx.Int32(push_warps)) & (global_warp < fx.Int32(warps_per_peer * num_peers))
        row = arith.select(is_pusher, row, row_end)
        peer_flags = peer + fx.Int64(COMBINE_FLAGS_OFFSET)
        row_lds = None
        if fx.const_expr(USE_TDM):
            row_lds = lds + _readfirstlane_i32(warp_id) * fx.Int32(row_buffer_bytes)
        pushed = fx.Int32(-1)
        topk_weights_res = create_buffer_resource(TOPK_WEIGHTS, max_size=True)
        while row < row_end:
            slot = row - row_begin + base
            # the previous row has landed, and is done with the LDS buffer
            if fx.const_expr(USE_TDM):
                tdm_ops.tensor_wait(0)
            else:
                wait_mem_all()
            weight = fx.Float32(0.0)
            if fx.const_expr(num_topk > 0):
                weight = buffer_load(
                    topk_weights_res,
                    row * fx.Int32(num_topk)
                    + arith.select(lane_id < fx.Int32(num_topk), lane_id, fx.Int32(0)),
                    vec_width=1,
                    dtype=fx.T.f32,
                )
            if fx.const_expr(USE_TDM):
                tdm_ops.tensor_load_2d(
                    _tdm_row(row_lds, _uniform_i64(_addr(x_base, 0, row, hidden_bytes)), hidden_bytes)
                )
                if fx.const_expr(num_topk > 0):
                    if lane_id < fx.Int32(num_topk):
                        st_lds(row_lds, fx.Int32(hidden_bytes // 4) + lane_id, weight)
                    _wait_lds()
                tdm_ops.tensor_wait(0)
                tdm_ops.tensor_store_2d(
                    _tdm_row(row_lds, _uniform_i64(_addr(peer, layout["x"], slot, slot_bytes)), push_bytes),
                    cache_policy=SYS_STORE,
                )
            else:
                _copy_row(
                    _addr(peer, layout["x"], slot, slot_bytes),
                    _addr(x_base, 0, row, hidden_bytes),
                    hidden_bytes,
                    FANOUT_PIECE_BYTES,
                    store_cache_modifier=SYS_STORE,
                )
                if fx.const_expr(num_topk > 0):
                    if lane_id < fx.Int32(num_topk):
                        _st_sys(_addr(peer, weights_offset, slot, slot_bytes), lane_id, weight)
            # raise the previous row's flag only now: the store waits to be sent before the warp
            # goes on, which this row's transfer hides
            if (pushed >= fx.Int32(0)) & (lane_id == fx.Int32(0)):
                _st_sys(peer_flags, pushed, epoch)
            pushed = slot
            row = row + fx.Int32(warps_per_peer)
        if fx.const_expr(USE_TDM):
            tdm_ops.tensor_wait(0)
        else:
            wait_mem_all()
        if (pushed >= fx.Int32(0)) & (lane_id == fx.Int32(0)):
            _st_sys(peer_flags, pushed, epoch)
        if fx.const_expr(PHASE_TIMING):
            if thread_id == fx.Int32(0):
                _stamp(local, 1, block_id, 1)

        # ---- reduce: each of our tokens over the ranks it went to, as soon as its rows are in;
        # warps claim tokens in order from a queue, the pushers once they are done ----
        # where rank r's rows start in our region: the tokens we sent to the ranks before r
        own_bases = []
        running = fx.Int32(0)
        for r in range_constexpr(num_ranks):
            own_bases.append(running)
            own_prev = _uniform_load_i32(
                rank_prefix_res,
                arith.select(rank > fx.Int32(0), (rank - fx.Int32(1)) * fx.Int32(num_ranks), fx.Int32(0))
                + fx.Int32(r),
            )
            own_prev = arith.select(rank > fx.Int32(0), own_prev, fx.Int32(0))
            running = (
                running
                + _uniform_load_i32(rank_prefix_res, rank * fx.Int32(num_ranks) + fx.Int32(r))
                - own_prev
            )
        # lane l looks after rank l % num_ranks of the claim's token l / num_ranks
        lane_r = lane_id % fx.Int32(num_ranks)
        lane_base = own_bases[0]
        for r in range_constexpr(1, num_ranks):
            lane_base = arith.select(lane_r == fx.Int32(r), own_bases[r], lane_base)

        send_head_res = create_buffer_resource(SEND_HEAD, max_size=True)
        recv_x_base = tensor_addr(RECV_X)
        local_flags = local + fx.Int64(COMBINE_FLAGS_OFFSET)
        counter = tensor_addr(REDUCE_COUNTER)
        lds_slot = fx.Int32(lds_grab) + warp_id
        in_dtype = X.element_type
        f32_vec = fx.T.vec(vec_elems, fx.T.f32)
        in_vec = fx.T.vec(vec_elems, in_dtype.ir_type)
        num_items = num_recv_tokens * fx.Int32(chunks)
        first = _grab_tokens(counter, lds, lds_slot, grab_base)
        while first < num_items:
            # with chunks, the claim is chunk ``first % chunks`` of token ``first / chunks``
            claim_first = first
            chunk_col = fx.Int32(0)
            if fx.const_expr(chunks > 1):
                claim_first = first // fx.Int32(chunks)
                chunk_col = (first % fx.Int32(chunks)) * fx.Int32(chunk_steps * WARP_SIZE)
            token_end = arith.select(
                claim_first + fx.Int32(REDUCE_GRAB_TOKENS) < num_recv_tokens,
                claim_first + fx.Int32(REDUCE_GRAB_TOKENS),
                num_recv_tokens,
            )
            # the claim's send_head entries in one load, and its rows' flags in one wait
            in_claim = lane_id < (token_end - claim_first) * fx.Int32(num_ranks)
            index = load_i32(
                send_head_res,
                arith.select(in_claim, claim_first * fx.Int32(num_ranks) + lane_id, fx.Int32(0)),
            )
            index = arith.select(in_claim, index, fx.Int32(-1))
            waits = (lane_r != rank) & (index >= fx.Int32(0))
            _wait_rows(
                local_flags,
                arith.select(waits, lane_base + index, fx.Int32(0)),
                waits,
                epoch,
                rank,
                claim_first,
            )
            token = claim_first
            while token < token_end:
                claim_lane = (token - claim_first) * fx.Int32(num_ranks)
                row_resources = []
                is_routed = []
                weight_addrs = []
                for r in range_constexpr(num_ranks):
                    index_r = _readlane_i32(index, claim_lane + fx.Int32(r))
                    routed = index_r >= fx.Int32(0)
                    index_r = arith.select(routed, index_r, fx.Int32(0))
                    # our own rows were not pushed: they are in x
                    is_own = rank == fx.Int32(r)
                    slot = arith.select(is_own, row_begins[r], own_bases[r]) + index_r
                    row_resources.append(
                        create_buffer_resource_from_addr(
                            _uniform_i64(
                                arith.select(
                                    is_own,
                                    _addr(x_base, 0, slot, hidden_bytes),
                                    _addr(local, layout["x"], slot, slot_bytes),
                                )
                            ),
                            num_records_bytes=arith.select(routed, fx.Int32(hidden_bytes), fx.Int32(0)),
                        )
                    )
                    # an unrouted rank's weight is masked out below: read it from our own region,
                    # since x and its weights are empty on a rank that received no tokens
                    weight_addrs.append(
                        arith.select(
                            routed,
                            arith.select(
                                is_own,
                                _addr(tensor_addr(TOPK_WEIGHTS), 0, slot, num_topk * 4),
                                _addr(local, weights_offset, slot, slot_bytes),
                            ),
                            _addr(local, weights_offset, fx.Int32(0), slot_bytes),
                        )
                    )
                    is_routed.append(routed)
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
                for first_step in range_constexpr(0, chunk_steps, steps_per_pass):
                    steps = range(first_step, min(first_step + steps_per_pass, chunk_steps))
                    if fx.const_expr(chunks > 1):
                        cols = [
                            (lane_id + fx.Int32(step * WARP_SIZE) + chunk_col) * fx.Int32(vec_elems)
                            for step in steps
                        ]
                    else:
                        cols = [
                            (lane_id + fx.Int32(step * WARP_SIZE)) * fx.Int32(vec_elems) for step in steps
                        ]
                    sums = [None] * len(cols)
                    sources = [(res, 0) for res in bias_resources]
                    sources += [(row_resources[r], SYS_LOAD) for r in range(num_ranks)]
                    for res, cache_modifier in sources:
                        values = [
                            buffer_load(
                                res, col, vec_width=vec_elems, dtype=in_dtype, cache_modifier=cache_modifier
                            )
                            for col in cols
                        ]
                        for k, value in enumerate(values):
                            term = value if fx.const_expr(elem_bytes == 4) else fx.arith.extf(f32_vec, value)
                            sums[k] = term if sums[k] is None else fx.arith.addf(sums[k], term)
                    for k, col in enumerate(cols):
                        out = sums[k] if fx.const_expr(elem_bytes == 4) else fx.arith.trunc_f(in_vec, sums[k])
                        buffer_store(out, out_res, col)
                if fx.const_expr(num_topk > 0):
                    if (lane_id < fx.Int32(num_topk)) & (chunk_col == fx.Int32(0)):
                        weight = fx.Float32(0.0)
                        for r in range_constexpr(num_ranks):
                            value = _ld_sys(weight_addrs[r], lane_id, dtype=fx.T.f32)
                            weight = weight + arith.select(is_routed[r], value, fx.Float32(0.0))
                        recv_topk_weights_res = create_buffer_resource(RECV_TOPK_WEIGHTS, max_size=True)
                        buffer_store(weight, recv_topk_weights_res, token * fx.Int32(num_topk) + lane_id)
                token = token + fx.Int32(1)
            first = _grab_tokens(counter, lds, lds_slot, grab_base)
        if fx.const_expr(PHASE_TIMING):
            if thread_id == fx.Int32(0):
                _stamp(local, 1, block_id, 2)
            fx.gpu.barrier()
            if thread_id == fx.Int32(0):
                _stamp(local, 1, block_id, 3)

    return combine_v2_kernel, smem_bytes


@flyc.jit
def launch_combine_v2(
    recv_x,
    recv_topk_weights,
    x,
    topk_weights,
    bias_0,
    bias_1,
    rank_prefix_matrix,
    send_head,
    buffer_ptrs,
    reduce_counter,
    num_rows: fx.Int32,
    num_recv_tokens: fx.Int32,
    rank: fx.Int32,
    epoch: fx.Int32,
    grab_base: fx.Int32,
    num_ranks: fx.Constexpr[int],
    num_channels: fx.Constexpr[int],
    hidden: fx.Constexpr[int],
    elem_bytes: fx.Constexpr[int],
    num_topk: fx.Constexpr[int],
    has_bias_0: fx.Constexpr[bool],
    has_bias_1: fx.Constexpr[bool],
    num_nvl_bytes: fx.Constexpr[int],
    stream: fx.Stream,
):
    kernel, smem_bytes = _make_combine_v2(
        num_ranks, num_channels, hidden, elem_bytes, num_topk, has_bias_0, has_bias_1, num_nvl_bytes
    )
    kernel(
        recv_x,
        recv_topk_weights,
        x,
        topk_weights,
        bias_0,
        bias_1,
        rank_prefix_matrix,
        send_head,
        buffer_ptrs,
        reduce_counter,
        num_rows,
        num_recv_tokens,
        rank,
        epoch,
        grab_base,
    ).launch(grid=(num_channels, 1, 1), block=(NUM_THREADS, 1, 1), stream=stream, smem=smem_bytes)
