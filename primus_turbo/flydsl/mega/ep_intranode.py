###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import os
from typing import Optional

import flydsl.expr as fx
from flydsl._mlir.dialects import vector as _vector
from flydsl.compiler.ast_rewriter import ASTRewriter
from flydsl.expr import arith, const_expr, range_constexpr
from flydsl.expr.buffer_ops import (
    buffer_load,
    buffer_store,
    create_buffer_resource_from_addr,
)

from primus_turbo.flydsl.mega.prims import (
    atomic_add,
    cast,
    copy_warp,
    ld,
    read_clock,
    spin_timed_out,
    st,
)
from primus_turbo.flydsl.mega.symm_buffer import SymBuffer, Workspace

_WARP = 64
_BLOCK_THREADS = 512
_PVEC = 8
_NUM_WARPS = _BLOCK_THREADS // _WARP
_L1_BYPASS = 1  # buffer cache_modifier: skip L1 (read fresh L2Y after l2_invalidate)
# Columns batched per reduce step; >1 keeps more gathers in flight before the adds.
_REDUCE_UNROLL = int(os.environ.get("TURBO_REDUCE_UNROLL", "2"))
# Compact live slots per token and branch to a body with exactly that many loads.
_REDUCE_COMPACT = os.environ.get("TURBO_REDUCE_COMPACT", "1") == "1"
# Probe only: push member 0 alone, as if the expert fold already happened upstream.
# Numerically wrong, but the byte counts match an epilogue-folded pool. Never ship on.
_FOLD1_PROBE = os.environ.get("TURBO_COMBINE_FOLD1_PROBE", "0") == "1"
# In-flight gather loads per warp in the dedup reduce. The arity branches share one
# VGPR budget, so the widest one sets it for all; scaling npass with the arity keeps
# every branch at this many loads. 0 disables (every branch uses the flat npass).
_GATHER_INFLIGHT = int(os.environ.get("TURBO_COMBINE_INFLIGHT", "16"))


def _npass_for(arity, out_features, npass):
    """Passes needed to hold ``arity`` members' loads within the in-flight budget."""
    if not _GATHER_INFLIGHT:
        return npass
    num_full_chunks = out_features // (_WARP * _PVEC)
    need = (arity * num_full_chunks + _GATHER_INFLIGHT - 1) // _GATHER_INFLIGHT
    return max(1, min(need, num_full_chunks))


@ASTRewriter.transform
def dispatch_bf16_tile(
    sym: SymBuffer,
    workspace: Workspace,
    thread_index: fx.Int32,
    hidden_size: int,
    input_res: fx.ArithValue,
    expert_send_dst_rank_res: fx.ArithValue,
    expert_send_dst_row_res: fx.ArithValue,
    expert_send_count_res: fx.ArithValue,
    expert_send_offset_res: fx.ArithValue,
    dispatched_token_idx_res: fx.ArithValue,
    source_slot_kind_res: Optional[fx.ArithValue],
    task_index: fx.ArithValue,
    signal: bool = False,
    block_m: int = 0,
    disp_parity: Optional[fx.Int32] = None,
    num_ranks: int = 0,
    num_topk: int = 1,
    skip_duplicates: bool = True,
    direct_unique_slots: bool = False,
    source_rank: int = 0,
    pair_slots: Optional[bool] = None,
):
    hidden_bytes = hidden_size * 2
    assert hidden_bytes % 1024 == 0, "hidden*2 must be a multiple of 1024 bytes -> hidden % 512 == 0"
    hidden_i32 = hidden_bytes // 4  # row stride in i32 words
    # Dedup prologue stores token*topk+k; the plain one stores token.
    use_pair_slots = skip_duplicates if pair_slots is None else pair_slots

    warp_id = thread_index // fx.Int32(_WARP)

    dst_rank = buffer_load(expert_send_dst_rank_res, task_index, vec_width=1, dtype=fx.T.i32())
    dest_row_start = buffer_load(expert_send_dst_row_res, task_index, vec_width=1, dtype=fx.T.i32())
    source_offset = buffer_load(expert_send_offset_res, task_index, vec_width=1, dtype=fx.T.i32())
    token_count = buffer_load(expert_send_count_res, task_index, vec_width=1, dtype=fx.T.i32())
    # hoist workspace-derived values before any dynamic control flow (rewriter can't carry Workspace)
    pool_address = sym.map(workspace.get_dispatch_token_pool_ptr(), dst_rank)
    dispatch_flag_address = sym.map(workspace.get_dispatch_flag_ptr(), dst_rank)
    num_max_pool_blocks = int(workspace.num_max_pool_blocks)

    local_count = (token_count - warp_id + fx.Int32(_NUM_WARPS - 1)) // fx.Int32(_NUM_WARPS)

    for i in range(local_count):
        row_index = warp_id + i * fx.Int32(_NUM_WARPS)
        source_slot = buffer_load(
            dispatched_token_idx_res, source_offset + row_index, vec_width=1, dtype=fx.T.i32()
        )
        source_row = source_slot // fx.Int32(num_topk if use_pair_slots else 1)
        is_duplicate = source_slot != source_slot
        if const_expr(skip_duplicates):
            source_kind = buffer_load(source_slot_kind_res, source_slot, vec_width=1, dtype=fx.T.i32())
            is_duplicate = source_kind == fx.Int32(0)
        should_copy = ~is_duplicate
        if should_copy:
            # dst = peer pool (base addr), src = local input (resource); offsets in i32 words
            if const_expr(direct_unique_slots):
                direct_row = fx.Int32(source_rank * int(workspace.num_max_tokens_per_rank)) + source_row
                copy_warp(
                    pool_address,
                    input_res,
                    hidden_bytes,
                    dst_off=direct_row * fx.Int32(hidden_i32),
                    src_off=source_row * fx.Int32(hidden_i32),
                    load_cache_modifier=19,
                    store_cache_modifier=19,
                )
            else:
                copy_warp(
                    pool_address,
                    input_res,
                    hidden_bytes,
                    dst_off=(dest_row_start + row_index) * fx.Int32(hidden_i32),
                    src_off=source_row * fx.Int32(hidden_i32),
                    load_cache_modifier=19,
                    store_cache_modifier=19,
                )

    if const_expr(signal):
        fx.rocdl.s_waitcnt(0)
        fx.gpu.barrier()
        if thread_index == fx.Int32(0):
            bank = fx.Int32(0) if disp_parity is None else disp_parity * fx.Int32(num_max_pool_blocks)
            local_expert = task_index // fx.Int32(num_ranks)
            atomic_add(dispatch_flag_address, bank + local_expert, fx.Int64(1), scope="sys")


@ASTRewriter.transform
def combine_bf16_tile(
    sym: SymBuffer,
    workspace: Workspace,
    thread_index: fx.Int32,
    task_index: fx.ArithValue,
    recv_dst_rank_res: fx.ArithValue,
    recv_start_row_res: fx.ArithValue,
    recv_count_res: fx.ArithValue,
    origin_slot_res: fx.ArithValue,
    grad_gate_res: Optional[fx.ArithValue] = None,
    signal: bool = False,
    epoch: Optional[fx.Int64] = None,
    bank_offset: Optional[fx.Int32] = None,
    with_gate: bool = False,
):
    # Task-based combine push: one warp sustains one peer's XGMI link (scattered dst_slot)
    out_features = int(workspace.hidden)
    n_slots = int(workspace.num_combine_slots)
    comb_records = n_slots * out_features * 2
    gate_records = n_slots * 4
    cols_per_step = _WARP * _PVEC
    num_full_chunks = out_features // cols_per_step
    tail_cols = out_features % cols_per_step
    row_words = out_features // 2
    full_bytes = num_full_chunks * cols_per_step * 2
    warp_id = thread_index // fx.Int32(_WARP)
    lane_id = thread_index % fx.Int32(_WARP)
    l2_ptr = workspace.get_l2_token_buffer_ptr()

    dst_rank = buffer_load(recv_dst_rank_res, task_index, vec_width=1, dtype=fx.T.i32())
    start_row = buffer_load(recv_start_row_res, task_index, vec_width=1, dtype=fx.T.i32())
    count = buffer_load(recv_count_res, task_index, vec_width=1, dtype=fx.T.i32())
    # hoist workspace-derived values before the dynamic loop (rewriter can't carry Workspace)
    comb_addr = sym.map(workspace.get_combine_token_buffer_ptr(), dst_rank)
    gate_addr = sym.map(workspace.get_combine_gate_ptr(), dst_rank) if with_gate else None
    barrier_addr = sym.map(workspace.get_reduce_flag_ptr(), dst_rank) if signal else None

    row_stride = fx.Int32(_NUM_WARPS)
    row_off = warp_id
    local_count = (count - row_off + row_stride - fx.Int32(1)) // row_stride
    for i in range(local_count):
        row = start_row + row_off + i * row_stride
        slot = buffer_load(origin_slot_res, row, vec_width=1, dtype=fx.T.i32())
        copy_warp(
            comb_addr,
            l2_ptr,
            full_bytes,
            dst_off=slot * fx.Int32(row_words),
            src_off=row * fx.Int32(row_words),
            load_cache_modifier=18,  # sc1|nt: read the same-agent GEMM stage.
            store_cache_modifier=19,  # sc0|sc1|nt: publish to a remote agent.
        )
        if const_expr(tail_cols):
            oob_index = fx.Int32(n_slots) * fx.Int32(out_features)
            slot_base = slot * fx.Int32(out_features)
            row_off = row * fx.Int32(out_features)
            l2_res = create_buffer_resource_from_addr(l2_ptr, num_records_bytes=n_slots * out_features * 2)
            peer = create_buffer_resource_from_addr(comb_addr, num_records_bytes=comb_records)
            col = fx.Int32(num_full_chunks * cols_per_step) + lane_id * fx.Int32(_PVEC)
            in_tail = (lane_id * fx.Int32(_PVEC)) < fx.Int32(tail_cols)
            safe_col = arith.select(in_tail, col, fx.Int32(out_features - _PVEC))
            tail_value = buffer_load(
                l2_res, row_off + safe_col, vec_width=_PVEC, dtype=fx.T.bf16(), cache_modifier=18
            )
            dst = arith.select(in_tail, slot_base + col, oob_index)
            buffer_store(tail_value, peer, dst, cache_modifier=19)
        if const_expr(with_gate):
            gate_value = buffer_load(grad_gate_res, row, vec_width=1, dtype=fx.T.f32())
            gate_peer = create_buffer_resource_from_addr(gate_addr, num_records_bytes=gate_records)
            buffer_store(gate_value, gate_peer, slot, cache_modifier=19)

        if const_expr(signal):
            bank = fx.Int32(0) if bank_offset is None else bank_offset
            # Wait for CM19 payload stores before publishing the relaxed completion flag.
            fx.rocdl.s_waitcnt(0)
            st(barrier_addr, bank + slot, epoch, order="relaxed", scope="sys")


def _member_row_resource(l2_ptr, row, present, row_bytes):
    """One row-sized buffer descriptor. ``present`` false -> num_records 0.

    The pool is >4 GB, so it cannot be addressed by one descriptor; a per-row base
    keeps the offsets 32-bit and lets an absent member be nulled by size instead.
    """
    base = l2_ptr + cast(row, fx.T.i64()) * fx.Int64(row_bytes)
    nbytes = row_bytes if present is None else arith.select(present, fx.Int32(row_bytes), fx.Int32(0))
    return create_buffer_resource_from_addr(base, num_records_bytes=nbytes)


def _gather_reduce_store(
    member_res,
    peer_res,
    weights,
    dst_base,
    lane_col,
    out_features,
    npass,
    oob_store_index,
):
    """Column-wise weighted sum of ``member_res`` -> one bf16 row at ``dst_base``.

    Plain python (no ASTRewriter): every loop here is unrolled at trace time, so the
    accumulator list never has to survive a traced branch.
    """
    f32_vec = fx.T.VectorType.get([_PVEC], fx.T.f32())
    bf16_vec = fx.T.VectorType.get([_PVEC], fx.T.bf16())
    cols_per_step = _WARP * _PVEC
    num_full_chunks = out_features // cols_per_step
    tail_cols = out_features % cols_per_step

    def accumulate(cols):
        accs = [None] * len(cols)
        for res, weight in zip(member_res, weights):
            values = [
                buffer_load(
                    res,
                    col,
                    vec_width=_PVEC,
                    dtype=fx.T.bf16(),
                    cache_modifier=18,  # sc1|nt: read the same-agent GEMM stage.
                )
                for col in cols
            ]
            for i, value in enumerate(values):
                term = fx.arith.extf(f32_vec, value)
                if weight is not None:
                    term = fx.arith.mulf(term, _vector.broadcast(f32_vec, weight))
                accs[i] = term if accs[i] is None else fx.arith.addf(accs[i], term)
        return accs

    # Split into passes so the live accumulator set stays under the GEMM path's VGPR budget.
    chunk_step = max(1, (num_full_chunks + npass - 1) // npass)
    for first in range(0, num_full_chunks, chunk_step):
        cols = [
            fx.Int32(c * cols_per_step) + lane_col
            for c in range(first, min(first + chunk_step, num_full_chunks))
        ]
        accs = accumulate(cols)
        for col, acc in zip(cols, accs):
            buffer_store(
                fx.arith.trunc_f(bf16_vec, acc),
                peer_res,
                dst_base + col,
                cache_modifier=19,  # sc0|sc1|nt: publish to a remote agent.
            )
    if tail_cols:
        col = fx.Int32(num_full_chunks * cols_per_step) + lane_col
        in_tail = lane_col < fx.Int32(tail_cols)
        safe_col = arith.select(in_tail, col, fx.Int32(out_features - _PVEC))
        acc = accumulate([safe_col])[0]
        dst = arith.select(in_tail, dst_base + col, oob_store_index)
        buffer_store(fx.arith.trunc_f(bf16_vec, acc), peer_res, dst, cache_modifier=19)


def _compact_live(live_offs, weights, valid, dead_off, topk):
    """Squeeze the live slots of one token to the front; returns (offs, weights, n).

    Dead slots are already free in bytes (their loads are dropped by num_records),
    but reduce is load-issue bound, not bandwidth bound -- the dropped loads still
    cost a full issue slot. Compacting lets the caller pick a shorter static body.
    """
    offs = [dead_off] * topk
    ws = [None] * topk if weights is None else [fx.Float32(0.0)] * topk
    n = fx.Int32(0)
    for j in range_constexpr(topk):
        for p in range_constexpr(topk):
            take = valid[j] & (n == fx.Int32(p))
            offs[p] = arith.select(take, live_offs[j], offs[p])
            if weights is not None:
                ws[p] = arith.select(take, weights[j], ws[p])
        n = n + arith.select(valid[j], fx.Int32(1), fx.Int32(0))
    return offs, (None if weights is None else ws), n


def _reduce_cols_store(
    comb_local_res,
    output_res,
    slot_offs,
    valid,
    weights,
    out_row,
    lane_col,
    out_features,
    topk,
    unroll,
):
    """Static column loop for one token's topk reduce -> one bf16 row.

    Plain python: everything unrolls at trace time, so ``unroll`` columns' worth of
    gathers (unroll * topk loads) are issued before any of them is consumed. The
    dynamic ``while`` version only kept topk loads in flight.
    """
    f32_vec = fx.T.VectorType.get([_PVEC], fx.T.f32())
    bf16_vec = fx.T.VectorType.get([_PVEC], fx.T.bf16())
    zero_vec = fx.arith.constant_vector(0.0, f32_vec)
    cols_per_step = _WARP * _PVEC
    num_steps = out_features // cols_per_step

    for first in range(0, num_steps, unroll):
        cols = [fx.Int32(s * cols_per_step) + lane_col for s in range(first, min(first + unroll, num_steps))]
        values = [
            [
                buffer_load(
                    comb_local_res,
                    slot_offs[j] + col,
                    vec_width=_PVEC,
                    dtype=fx.T.bf16(),
                    cache_modifier=19,  # sc0|sc1|nt: system-visible non-temporal read.
                )
                for j in range_constexpr(topk)
            ]
            for col in cols
        ]
        for col, vals in zip(cols, values):
            acc = None
            for j in range_constexpr(topk):
                term = fx.arith.extf(f32_vec, vals[j])
                if weights is not None:
                    term = fx.arith.mulf(term, _vector.broadcast(f32_vec, weights[j]))
                if valid is not None:
                    term = fx.arith.select(valid[j], term, zero_vec)
                acc = term if acc is None else fx.arith.addf(acc, term)
            if acc is None:
                acc = zero_vec  # no live slot: the row is all zeros
            buffer_store(fx.arith.trunc_f(bf16_vec, acc), output_res, out_row + col)


@ASTRewriter.transform
def combine_dedup_bf16_tile(
    sym: SymBuffer,
    workspace: Workspace,
    thread_index: fx.Int32,
    task_index: fx.ArithValue,
    recv_dst_rank_res: fx.ArithValue,
    recv_start_row_res: fx.ArithValue,
    recv_count_res: fx.ArithValue,
    origin_slot_res: fx.ArithValue,
    sorted_slot_res: fx.ArithValue,
    key_row_res: fx.ArithValue,
    grad_gate_res: Optional[fx.ArithValue] = None,
    topk: int = 1,
    apply_weights: bool = False,
    signal: bool = False,
    epoch: Optional[fx.Int64] = None,
    bank_offset: Optional[fx.Int32] = None,
    with_gate: bool = False,
    npass: int = 2,
):
    # DeepEP-style sender dedup: the highest pool row of a source token folds every
    # local route of that token into one weighted row and pushes it to the primary
    # slot. Cuts the XGMI write (and the peer's read) by the duplicate ratio.
    out_features = int(workspace.hidden)
    n_slots = int(workspace.num_combine_slots)
    num_pool_rows = int(workspace.num_max_pool_tokens)
    comb_records = n_slots * out_features * 2
    gate_records = n_slots * 4
    row_bytes = out_features * 2
    warp_id = thread_index // fx.Int32(_WARP)
    lane_col = (thread_index % fx.Int32(_WARP)) * fx.Int32(_PVEC)
    oob_row = fx.Int32(num_pool_rows)
    oob_slot = fx.Int32(n_slots)

    dst_rank = buffer_load(recv_dst_rank_res, task_index, vec_width=1, dtype=fx.T.i32())
    start_row = buffer_load(recv_start_row_res, task_index, vec_width=1, dtype=fx.T.i32())
    count = buffer_load(recv_count_res, task_index, vec_width=1, dtype=fx.T.i32())
    # hoist workspace-derived values before the dynamic loop (rewriter can't carry Workspace)
    comb_addr = sym.map(workspace.get_combine_token_buffer_ptr(), dst_rank)
    gate_addr = sym.map(workspace.get_combine_gate_ptr(), dst_rank) if with_gate else None
    barrier_addr = sym.map(workspace.get_reduce_flag_ptr(), dst_rank) if signal else None
    # Exactly-sized resources: an absent member indexes past num_records, so the
    # hardware drops the request and returns 0 instead of moving any bytes.
    l2_ptr = workspace.get_l2_token_buffer_ptr()
    weight_res = (
        create_buffer_resource_from_addr(
            workspace.get_weight_recv_buf_ptr(), num_records_bytes=num_pool_rows * 4
        )
        if apply_weights
        else None
    )
    peer_res = create_buffer_resource_from_addr(comb_addr, num_records_bytes=comb_records)
    gate_peer_res = (
        create_buffer_resource_from_addr(gate_addr, num_records_bytes=gate_records) if with_gate else None
    )

    row_stride = fx.Int32(_NUM_WARPS)
    row_off = warp_id
    local_count = (count - row_off + row_stride - fx.Int32(1)) // row_stride
    # Publishing a flag per row forces a vmcnt drain per row, so a warp never has
    # more than one row of remote stores in flight. Deferred mode pushes the whole
    # chunk, drains once, then publishes; the reduce side sees the rows later.
    for i in range(local_count):
        row = start_row + row_off + i * row_stride
        key = buffer_load(sorted_slot_res, row, vec_width=1, dtype=fx.T.i32())
        key_base = key * fx.Int32(topk)
        # Members are row-descending with a -1 tail: slot 0 pushes, the last valid is primary.
        pusher_row = buffer_load(key_row_res, key_base, vec_width=1, dtype=fx.T.i32())
        if row == pusher_row:
            member_rows = [
                buffer_load(key_row_res, key_base + fx.Int32(k), vec_width=1, dtype=fx.T.i32())
                for k in range_constexpr(topk)
            ]
            primary_row = member_rows[0]
            present = [None] * topk
            safe_rows = [member_rows[0]]
            for k in range_constexpr(topk):
                if k > 0:
                    present[k] = member_rows[k] >= fx.Int32(0)
                    primary_row = arith.select(present[k], member_rows[k], primary_row)
                    safe_rows.append(arith.select(present[k], member_rows[k], oob_row))
            # slot 0 is the pusher itself: always present, so keep its size static.
            member_res = [
                _member_row_resource(l2_ptr, safe_rows[k], present[k], row_bytes)
                for k in range_constexpr(topk)
            ]
            weights = [None] * topk
            if const_expr(apply_weights):
                weights = [buffer_load(weight_res, r, vec_width=1, dtype=fx.T.f32()) for r in safe_rows]
            dst_slot = buffer_load(origin_slot_res, primary_row, vec_width=1, dtype=fx.T.i32())
            dst_base = dst_slot * fx.Int32(out_features)
            oob_store = oob_slot * fx.Int32(out_features)
            if const_expr(topk == 1 or _FOLD1_PROBE):
                _gather_reduce_store(
                    member_res[:1],
                    peer_res,
                    weights[:1],
                    dst_base,
                    lane_col,
                    out_features,
                    _npass_for(1, out_features, npass),
                    oob_store,
                )
            else:
                # An absent member costs no bandwidth but still issues its loads, so
                # branch on the real arity. Routes land ~60% lone, ~30% pairs, ~10% more.
                if member_rows[1] < fx.Int32(0):
                    _gather_reduce_store(
                        member_res[:1],
                        peer_res,
                        weights[:1],
                        dst_base,
                        lane_col,
                        out_features,
                        _npass_for(1, out_features, npass),
                        oob_store,
                    )
                else:
                    if const_expr(topk == 2):
                        _gather_reduce_store(
                            member_res,
                            peer_res,
                            weights,
                            dst_base,
                            lane_col,
                            out_features,
                            _npass_for(2, out_features, npass),
                            oob_store,
                        )
                    else:
                        if member_rows[2] < fx.Int32(0):
                            _gather_reduce_store(
                                member_res[:2],
                                peer_res,
                                weights[:2],
                                dst_base,
                                lane_col,
                                out_features,
                                _npass_for(2, out_features, npass),
                                oob_store,
                            )
                        else:
                            _gather_reduce_store(
                                member_res,
                                peer_res,
                                weights,
                                dst_base,
                                lane_col,
                                out_features,
                                _npass_for(topk, out_features, npass),
                                oob_store,
                            )

            if const_expr(with_gate):
                # Gate is a per-slot scalar: push one per member, before the primary flag.
                # These two resources are max_size, so absent members reuse row 0 (the
                # pusher) and are nullified on the store side instead.
                for k in range_constexpr(topk):
                    is_member = member_rows[k] >= fx.Int32(0)
                    gate_row = arith.select(is_member, member_rows[k], member_rows[0])
                    gate_value = buffer_load(grad_gate_res, gate_row, vec_width=1, dtype=fx.T.f32())
                    member_slot = buffer_load(origin_slot_res, gate_row, vec_width=1, dtype=fx.T.i32())
                    gate_dst = arith.select(is_member, member_slot, oob_slot)
                    buffer_store(gate_value, gate_peer_res, gate_dst, cache_modifier=19)

            if const_expr(signal):
                bank = fx.Int32(0) if bank_offset is None else bank_offset
                # Wait for CM19 payload stores before publishing the relaxed flag.
                fx.rocdl.s_waitcnt(0)
                st(barrier_addr, bank + dst_slot, epoch, order="relaxed", scope="sys")


@ASTRewriter.transform
def topk_reduce_bf16_tile(
    signal: bool,
    apply_weights: bool,
    with_gate: bool,
    thread_index: fx.Int32,
    base_pid: fx.Int32,
    total_warps: fx.Int32,
    topk: int,
    out_features: int,
    num_experts: int,
    rank: int,
    comb_local_res: fx.ArithValue,
    output_res: fx.ArithValue,
    topk_indices_res: fx.ArithValue,
    num_tokens_res: fx.ArithValue,
    barrier_base: fx.ArithValue,
    reduce_bank: fx.Int32,
    topk_weights_res: fx.ArithValue,
    gate_local_res: Optional[fx.ArithValue],
    d_topk_w_res: Optional[fx.ArithValue],
    epoch: fx.Int64,
    dedup: bool = False,
    kind_res: Optional[fx.ArithValue] = None,
    num_combine_slots: int = 0,
):
    assert not dedup or num_combine_slots > 0, "dedup reduce needs num_combine_slots for the OOB sentinel"
    f32_vec = fx.T.VectorType.get([_PVEC], fx.T.f32())
    bf16_vec = fx.T.VectorType.get([_PVEC], fx.T.bf16())
    num_vec_chunks = out_features // _PVEC
    lane_id = thread_index % fx.Int32(_WARP)
    warp_id = thread_index // fx.Int32(_WARP)
    global_warp_id = base_pid * fx.Int32(_NUM_WARPS) + warp_id
    num_tokens = buffer_load(num_tokens_res, fx.Int32(rank), vec_width=1, dtype=fx.T.i32())
    token = global_warp_id
    while token < num_tokens:
        if const_expr(signal):
            # Wait each slot's flag == epoch. Loop MUST stay inline (rewriter needs the control flow).
            for j in range_constexpr(topk):
                slot = token * fx.Int32(topk) + fx.Int32(j)
                topk_index = buffer_load(topk_indices_res, slot, vec_width=1, dtype=fx.T.i64())
                awaited = (topk_index >= fx.Int64(0)) & (topk_index < fx.Int64(num_experts))
                if const_expr(dedup):
                    # kind 0 = duplicate route; the sender folded it into the primary slot.
                    kind = buffer_load(kind_res, slot, vec_width=1, dtype=fx.T.i32())
                    awaited = awaited & (kind != fx.Int32(0))
                if awaited:
                    if lane_id == fx.Int32(0):
                        spin_start = read_clock()
                        fx.rocdl.s_waitcnt(0)
                        flag = ld(
                            barrier_base,
                            reduce_bank + slot,
                            order="relaxed",
                            scope="sys",
                            dtype=fx.T.i64(),
                        )
                        while flag != epoch:
                            fx.rocdl.s_sleep(fx.Int32(1))
                            if spin_timed_out(spin_start):
                                # rank is a compile-time constant, baked into the format string
                                fx.printf(
                                    "[MEGA rank=" + str(rank) + " topk_reduce] combine reduce-flag stuck: "
                                    "GEMM has not written this expert's rows; token={} slot={} expert={} "
                                    "reduce_flag_index={} (seen_flag={} expected_epoch={})\n",
                                    token,
                                    slot,
                                    topk_index,
                                    reduce_bank + slot,
                                    flag,
                                    epoch,
                                )
                                spin_start = read_clock()
                            # re-read the flag each spin iteration (MUST stay inside the while)
                            fx.rocdl.s_waitcnt(0)
                            flag = ld(
                                barrier_base,
                                reduce_bank + slot,
                                order="relaxed",
                                scope="sys",
                                dtype=fx.T.i64(),
                            )
            fx.gpu.barrier()

        token_row_off = token * fx.Int32(topk) * fx.Int32(out_features)
        # Prefetch per-slot validity (scalar, once per token).
        valid = []
        for j in range_constexpr(topk):
            slot = token * fx.Int32(topk) + fx.Int32(j)
            idx = buffer_load(topk_indices_res, slot, vec_width=1, dtype=fx.T.i64())
            ok = (idx >= fx.Int64(0)) & (idx < fx.Int64(num_experts))
            if const_expr(dedup):
                ok = ok & (buffer_load(kind_res, slot, vec_width=1, dtype=fx.T.i32()) != fx.Int32(0))
            valid.append(ok)
        # Point dead slots past num_records: the load is dropped and reads back 0,
        # so a skipped slot costs no combine-buffer bandwidth at all.
        slot_offs = []
        for j in range_constexpr(topk):
            live_off = token_row_off + fx.Int32(j * out_features)
            if const_expr(num_combine_slots > 0):
                live_off = arith.select(valid[j], live_off, fx.Int32(num_combine_slots * out_features))
            slot_offs.append(live_off)
        out_row = token * fx.Int32(out_features)
        hoisted_weights = None
        if const_expr(apply_weights):
            hoisted_weights = [
                buffer_load(
                    topk_weights_res,
                    token * fx.Int32(topk) + fx.Int32(j),
                    vec_width=1,
                    dtype=fx.T.f32(),
                    cache_modifier=19,
                )
                for j in range_constexpr(topk)
            ]
        static_cols = const_expr(out_features % (_WARP * _PVEC) == 0)
        if const_expr(static_cols and _REDUCE_COMPACT and num_combine_slots > 0):
            dead_off = fx.Int32(num_combine_slots * out_features)
            comp_offs, comp_w, n_live = _compact_live(
                [token_row_off + fx.Int32(j * out_features) for j in range_constexpr(topk)],
                hoisted_weights,
                valid,
                dead_off,
                topk,
            )
            # One static body per live-slot count: no dead load is ever issued.
            for k in range_constexpr(topk + 1):
                if n_live == fx.Int32(k):
                    _reduce_cols_store(
                        comb_local_res,
                        output_res,
                        comp_offs[:k],
                        None,
                        None if comp_w is None else comp_w[:k],
                        out_row,
                        lane_id * fx.Int32(_PVEC),
                        out_features,
                        k,
                        _REDUCE_UNROLL,
                    )
        elif const_expr(static_cols):
            _reduce_cols_store(
                comb_local_res,
                output_res,
                slot_offs,
                valid,
                hoisted_weights,
                out_row,
                lane_id * fx.Int32(_PVEC),
                out_features,
                topk,
                _REDUCE_UNROLL,
            )
        else:
            zero_vec = fx.arith.constant_vector(0.0, f32_vec)
            vec_idx = lane_id
            while vec_idx < fx.Int32(num_vec_chunks):
                col = vec_idx * fx.Int32(_PVEC)
                topk_vals = []
                for j in range_constexpr(topk):
                    topk_vals.append(
                        buffer_load(
                            comb_local_res,
                            slot_offs[j] + col,
                            vec_width=_PVEC,
                            dtype=fx.T.bf16(),
                            cache_modifier=19,  # sc0|sc1|nt: system-visible non-temporal read.
                        )
                    )
                acc = None
                for j in range_constexpr(topk):
                    term = fx.arith.extf(f32_vec, topk_vals[j])
                    if const_expr(apply_weights):
                        term = fx.arith.mulf(term, _vector.broadcast(f32_vec, hoisted_weights[j]))
                    term = fx.arith.select(valid[j], term, zero_vec)
                    acc = term if acc is None else fx.arith.addf(acc, term)
                buffer_store(fx.arith.trunc_f(bf16_vec, acc), output_res, out_row + col)
                vec_idx = vec_idx + fx.Int32(_WARP)
        if const_expr(signal and with_gate):
            for j in range_constexpr(topk):
                slot = token * fx.Int32(topk) + fx.Int32(j)
                topk_index = buffer_load(topk_indices_res, slot, vec_width=1, dtype=fx.T.i64())
                if lane_id == fx.Int32(0):
                    gate_v = buffer_load(
                        gate_local_res, slot, vec_width=1, dtype=fx.T.f32(), cache_modifier=19
                    )
                    zero_f = fx.Float32(0.0)
                    v1 = fx.arith.select(topk_index < fx.Int64(num_experts), gate_v, zero_f)
                    d_val = fx.arith.select(topk_index >= fx.Int64(0), v1, zero_f)
                    buffer_store(d_val, d_topk_w_res, slot)
        token = token + total_warps
