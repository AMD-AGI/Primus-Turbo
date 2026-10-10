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

from typing import Optional

import flydsl.expr as fx
from flydsl.compiler.ast_rewriter import ASTRewriter
from flydsl.expr import arith, const_expr, range_constexpr
from flydsl.expr.buffer_ops import (
    buffer_load,
    buffer_store,
    create_buffer_resource_from_addr,
)

from primus_turbo.flydsl.mega.bf16.barrier import spin_until_flag_reaches
from primus_turbo.flydsl.mega.bf16.symm_buffer import SymBuffer, Workspace
from primus_turbo.flydsl.utils.prims import (
    atomic_add,
    cast,
    ceildiv,
    copy_warp,
    st,
)

_WARP = 64
_BLOCK_THREADS = 512
_PVEC = 8
_NUM_WARPS = _BLOCK_THREADS // _WARP
# fold rows a combine warp pushes per group; their index loads and payload stores overlap
_ROW_UNROLL = 4
# per-warp ceiling on bf16x8 loads in flight in _sum_pool_rows; above it the 8-row sum spills
_MAX_LOADS_IN_FLIGHT = 64


def same_rank_order(expert_ids, topk_position, num_experts_per_rank):
    """Return (same_rank_position, num_same_rank_routes) of a valid route among routes to its rank."""
    expert_id = expert_ids[topk_position]
    dst_rank = expert_id // fx.Int32(num_experts_per_rank)
    same_rank_position = fx.Int32(0)
    num_same_rank_routes = fx.Int32(1)
    for other_position, other_expert_id in enumerate(expert_ids):
        if other_position == topk_position:
            continue
        is_same_rank = (other_expert_id >= fx.Int32(0)) & (
            other_expert_id // fx.Int32(num_experts_per_rank) == dst_rank
        )
        # ties break by topk_position so the order stays total
        is_before = (
            other_expert_id <= expert_id if other_position < topk_position else other_expert_id < expert_id
        )
        same_rank_position += arith.select(is_same_rank & is_before, fx.Int32(1), fx.Int32(0))
        num_same_rank_routes += arith.select(is_same_rank, fx.Int32(1), fx.Int32(0))
    return same_rank_position, num_same_rank_routes


def is_primary_route(expert_ids, topk_position, num_experts_per_rank):
    """Return whether the route is first to its rank; only ids < 0 count as invalid, callers bound the top."""
    same_rank_position, _ = same_rank_order(expert_ids, topk_position, num_experts_per_rank)
    return (expert_ids[topk_position] >= fx.Int32(0)) & (same_rank_position == fx.Int32(0))


@ASTRewriter.transform
def dispatch_bf16_tile(
    sym: SymBuffer,
    workspace: Workspace,
    thread_index: fx.Int32,
    hidden_size: int,
    input_res: fx.ArithValue,
    expert_send_dst_rank_res: fx.ArithValue,
    expert_send_count_res: fx.ArithValue,
    expert_send_offset_res: fx.ArithValue,
    dispatched_token_idx_res: fx.ArithValue,
    task_index: fx.ArithValue,
    dispatch_parity: fx.Int32,
    chunk_index: fx.Int32,
    dispatch_chunk_counter_ptr: fx.ArithValue,
    num_chunks: int,
    num_ranks: int,
    rank: int,
):
    hidden_bytes = hidden_size * 2
    assert hidden_bytes % 1024 == 0, "hidden*2 must be a multiple of 1024 bytes -> hidden % 512 == 0"
    hidden_i32 = hidden_bytes // 4  # row stride in i32 words
    row_stride = num_chunks * _NUM_WARPS

    warp_id = chunk_index * fx.Int32(_NUM_WARPS) + thread_index // fx.Int32(_WARP)

    dst_rank = buffer_load(expert_send_dst_rank_res, task_index, vec_width=1, dtype=fx.T.i32())
    source_offset = buffer_load(expert_send_offset_res, task_index, vec_width=1, dtype=fx.T.i32())
    token_count = buffer_load(expert_send_count_res, task_index, vec_width=1, dtype=fx.T.i32())
    # hoist workspace-derived values before any dynamic control flow (rewriter can't carry Workspace)
    token_buffer_address = sym.map(workspace.get_dispatch_token_buffer_ptr(), dst_rank)
    dispatch_flag_address = sym.map(workspace.get_dispatch_flag_ptr(), dst_rank)
    num_max_pool_blocks = int(workspace.num_max_pool_blocks)
    recv_token_base = rank * int(workspace.num_max_tokens_per_rank)

    local_count = (token_count - warp_id + fx.Int32(row_stride - 1)) // fx.Int32(row_stride)

    for i in range(local_count):
        token_idx = buffer_load(
            dispatched_token_idx_res,
            source_offset + warp_id + i * fx.Int32(row_stride),
            vec_width=1,
            dtype=fx.T.i32(),
        )
        # dst = peer dispatch_token_buffer row of this recv token, src = local input; offsets in i32 words
        copy_warp(
            token_buffer_address,
            input_res,
            hidden_bytes,
            dst_off=(fx.Int32(recv_token_base) + token_idx) * fx.Int32(hidden_i32),
            src_off=token_idx * fx.Int32(hidden_i32),
            load_cache_modifier=18,  # sc1|nt: read data produced by the same agent.
            store_cache_modifier=19,
        )

    fx.rocdl.s_waitcnt(0)
    fx.gpu.barrier()
    if thread_index == fx.Int32(0):
        # each chunk's stores retired before its counter add, so the last chunk's flag add publishes them all
        chunk_count_before_add = atomic_add(
            dispatch_chunk_counter_ptr, task_index, fx.Int64(1), scope="agent"
        )
        if chunk_count_before_add == fx.Int64(num_chunks - 1):
            # the last chunk rearms the counter, so every launch starts from 0 whatever its chunk count
            st(dispatch_chunk_counter_ptr, task_index, fx.Int64(0), scope="agent")
            bank = dispatch_parity * fx.Int32(num_max_pool_blocks)
            local_expert = task_index // fx.Int32(num_ranks)
            atomic_add(dispatch_flag_address, bank + local_expert, fx.Int64(1), scope="sys")


@ASTRewriter.transform
def dispatch_bf16_block(
    sym: SymBuffer,
    workspace: Workspace,
    thread_index: fx.Int32,
    block_index: fx.Int32,
    hidden_size: int,
    input_res: fx.ArithValue,
    expert_send_dst_rank_res: fx.ArithValue,
    expert_send_count_res: fx.ArithValue,
    expert_send_offset_res: fx.ArithValue,
    dispatched_token_idx_res: fx.ArithValue,
    dispatch_parity: fx.Int32,
    dispatch_chunk_counter_ptr: fx.ArithValue,
    num_chunks: int,
    num_ranks: int,
    num_experts_per_rank: int,
    rank: int,
):
    """Push this block's chunk of every expert's task to one dst rank, in ascending expert order."""
    dst_rank = block_index % fx.Int32(num_ranks)
    chunk_index = block_index // fx.Int32(num_ranks)
    # ascending expert order: flag[g] complete implies every expert <= g landed (single-flag wait)
    for expert_idx in range(fx.Int32(num_experts_per_rank)):
        dispatch_bf16_tile(
            sym,
            workspace,
            thread_index=thread_index,
            hidden_size=hidden_size,
            input_res=input_res,
            expert_send_dst_rank_res=expert_send_dst_rank_res,
            expert_send_count_res=expert_send_count_res,
            expert_send_offset_res=expert_send_offset_res,
            dispatched_token_idx_res=dispatched_token_idx_res,
            task_index=expert_idx * fx.Int32(num_ranks) + dst_rank,
            dispatch_parity=dispatch_parity,
            chunk_index=chunk_index,
            dispatch_chunk_counter_ptr=dispatch_chunk_counter_ptr,
            num_chunks=num_chunks,
            num_ranks=num_ranks,
            rank=rank,
        )


def _pool_row_resource(l2_ptr, pool_row, row_bytes, is_present=True):
    """Buffer resource over one pool row; an absent row gets num_records 0, so its loads read 0 at no cost."""
    base = l2_ptr + cast(pool_row, fx.T.i64()) * fx.Int64(row_bytes)
    num_records_bytes = (
        row_bytes if is_present is True else arith.select(is_present, fx.Int32(row_bytes), fx.Int32(0))
    )
    return create_buffer_resource_from_addr(base, num_records_bytes=num_records_bytes)


def _sum_pool_rows(pool_rows, l2_ptr, combine_token_res, dst_base, dropped_base, lane_col, hidden):
    """Store the f32 sum of the present pool_rows (ascending, -1 absent) as one bf16 row at dst_base."""
    f32_vec = fx.T.VectorType.get([_PVEC], fx.T.f32())
    bf16_vec = fx.T.VectorType.get([_PVEC], fx.T.bf16())
    row_bytes = hidden * 2
    cols_per_step = _WARP * _PVEC
    num_full_chunks = hidden // cols_per_step
    tail_cols = hidden % cols_per_step
    row_resources = [_pool_row_resource(l2_ptr, pool_rows[0], row_bytes)]
    for pool_row in pool_rows[1:]:
        is_present = pool_row >= fx.Int32(0)
        safe_row = arith.select(is_present, pool_row, pool_rows[0])
        row_resources.append(_pool_row_resource(l2_ptr, safe_row, row_bytes, is_present))

    def accumulate(cols):
        sums = [None] * len(cols)
        for row_res in row_resources:
            # sc1|nt: read the same-agent GEMM stage
            values = [
                buffer_load(row_res, col, vec_width=_PVEC, dtype=fx.T.bf16(), cache_modifier=18)
                for col in cols
            ]
            for i, value in enumerate(values):
                term = fx.arith.extf(f32_vec, value)
                sums[i] = term if sums[i] is None else fx.arith.addf(sums[i], term)
        return sums

    # passes bound the loads in flight so the sum stays inside the GEMM's VGPR budget
    num_passes = max(1, min(num_full_chunks, ceildiv(len(pool_rows) * num_full_chunks, _MAX_LOADS_IN_FLIGHT)))
    chunks_per_pass = max(1, ceildiv(num_full_chunks, num_passes))
    for first_chunk in range(0, num_full_chunks, chunks_per_pass):
        last_chunk = min(first_chunk + chunks_per_pass, num_full_chunks)
        cols = [fx.Int32(chunk * cols_per_step) + lane_col for chunk in range(first_chunk, last_chunk)]
        for col, row_sum in zip(cols, accumulate(cols)):
            # sc0|sc1|nt: publish to a remote agent
            buffer_store(
                fx.arith.trunc_f(bf16_vec, row_sum), combine_token_res, dst_base + col, cache_modifier=19
            )
    if tail_cols:
        col = fx.Int32(num_full_chunks * cols_per_step) + lane_col
        is_in_tail = lane_col < fx.Int32(tail_cols)
        row_sum = accumulate([arith.select(is_in_tail, col, fx.Int32(hidden - _PVEC))])[0]
        dst = arith.select(is_in_tail, dst_base + col, dropped_base)
        buffer_store(fx.arith.trunc_f(bf16_vec, row_sum), combine_token_res, dst, cache_modifier=19)


@ASTRewriter.transform
def combine_bf16_tile(
    sym: SymBuffer,
    workspace: Workspace,
    thread_index: fx.Int32,
    segment_idx: fx.Int32,
    first_fold_row: fx.Int32,
    num_fold_rows: fx.Int32,
    recv_dst_rank_res: fx.ArithValue,
    pool_row_to_route_res: fx.ArithValue,
    pool_row_to_recv_token_res: fx.ArithValue,
    recv_token_to_pool_rows_res: fx.ArithValue,
    epoch: fx.Int64,
    reduce_bank: fx.Int32,
    grad_gate_res: Optional[fx.ArithValue] = None,
    has_grad_gate: bool = False,
):
    """Push each fold row in [first_fold_row, +num_fold_rows) as its recv token's sum to the primary route."""
    hidden = int(workspace.hidden)
    num_topk = int(workspace.num_topk)
    assert num_topk <= _WARP, "lane k pushes the grad_gate of pool row k"
    num_max_routes = int(workspace.num_max_routes)
    row_bytes = hidden * 2
    row_words = hidden // 2
    cols_per_step = _WARP * _PVEC
    num_full_chunks = hidden // cols_per_step
    tail_cols = hidden % cols_per_step
    warp_id = thread_index // fx.Int32(_WARP)
    lane_id = thread_index % fx.Int32(_WARP)
    lane_col = lane_id * fx.Int32(_PVEC)
    dropped_base = fx.Int32(num_max_routes * hidden)
    l2_ptr = workspace.get_l2_token_buffer_ptr()

    dst_rank = buffer_load(recv_dst_rank_res, segment_idx, vec_width=1, dtype=fx.T.i32())
    # hoist workspace-derived values before the dynamic loop (rewriter can't carry Workspace)
    combine_token_addr = sym.map(workspace.get_combine_token_buffer_ptr(), dst_rank)
    reduce_flag_addr = sym.map(workspace.get_reduce_flag_ptr(), dst_rank)
    combine_token_res = create_buffer_resource_from_addr(
        combine_token_addr, num_records_bytes=num_max_routes * row_bytes
    )
    combine_gate_res = (
        create_buffer_resource_from_addr(
            sym.map(workspace.get_combine_gate_ptr(), dst_rank), num_records_bytes=num_max_routes * 4
        )
        if has_grad_gate
        else None
    )

    def push_rows(fold_rows):
        # stage each index level for every row first, so the dependent loads overlap across rows
        recv_tokens = [
            buffer_load(pool_row_to_recv_token_res, fold_row, vec_width=1, dtype=fx.T.i32())
            for fold_row in fold_rows
        ]
        pool_rows_per_row = [
            [
                buffer_load(
                    recv_token_to_pool_rows_res,
                    recv_token * fx.Int32(num_topk) + fx.Int32(k),
                    vec_width=1,
                    dtype=fx.T.i32(),
                )
                for k in range_constexpr(num_topk)
            ]
            for recv_token in recv_tokens
        ]
        # pool_rows[0] is the primary route's row; the sum lands in that route's combine row
        dst_routes = [
            buffer_load(pool_row_to_route_res, pool_rows[0], vec_width=1, dtype=fx.T.i32())
            for pool_rows in pool_rows_per_row
        ]
        for u in range_constexpr(len(fold_rows)):
            push_payload(fold_rows[u], pool_rows_per_row[u], dst_routes[u])
        # payload and grad_gate stores retire before the flags that topk_reduce_bf16_tile waits on
        fx.rocdl.s_waitcnt(0)
        for u in range_constexpr(len(fold_rows)):
            st(reduce_flag_addr, reduce_bank + dst_routes[u], epoch, scope="sys")

    def push_payload(fold_row, pool_rows, dst_route):
        dst_base = dst_route * fx.Int32(hidden)
        second_pool_row = pool_rows[1] if num_topk > 1 else fx.Int32(-1)
        if second_pool_row < fx.Int32(0):
            copy_warp(
                combine_token_addr,
                # i64 row base: fold_row * row_words overflows i32 on large pools
                l2_ptr + cast(fold_row, fx.T.i64()) * fx.Int64(row_bytes),
                num_full_chunks * cols_per_step * 2,
                dst_off=dst_route * fx.Int32(row_words),
                load_cache_modifier=18,  # sc1|nt: read the same-agent GEMM stage.
                store_cache_modifier=19,  # sc0|sc1|nt: publish to a remote agent.
            )
            if const_expr(tail_cols):
                col = fx.Int32(num_full_chunks * cols_per_step) + lane_col
                is_in_tail = lane_col < fx.Int32(tail_cols)
                fold_row_res = _pool_row_resource(l2_ptr, fold_row, row_bytes)
                safe_col = arith.select(is_in_tail, col, fx.Int32(hidden - _PVEC))
                tail_value = buffer_load(
                    fold_row_res, safe_col, vec_width=_PVEC, dtype=fx.T.bf16(), cache_modifier=18
                )
                dst = arith.select(is_in_tail, dst_base + col, dropped_base)
                buffer_store(tail_value, combine_token_res, dst, cache_modifier=19)
        else:
            _sum_pool_rows(pool_rows, l2_ptr, combine_token_res, dst_base, dropped_base, lane_col, hidden)
        if const_expr(has_grad_gate):
            # lane k pushes the grad_gate of pool_rows[k] to that row's own route
            lane_pool_row = pool_rows[0]
            for k in range_constexpr(1, num_topk):
                lane_pool_row = arith.select(lane_id == fx.Int32(k), pool_rows[k], lane_pool_row)
            is_gate_lane = (lane_id < fx.Int32(num_topk)) & (lane_pool_row >= fx.Int32(0))
            gate_row = arith.select(is_gate_lane, lane_pool_row, pool_rows[0])
            gate_value = buffer_load(grad_gate_res, gate_row, vec_width=1, dtype=fx.T.f32())
            gate_route = buffer_load(pool_row_to_route_res, gate_row, vec_width=1, dtype=fx.T.i32())
            gate_dst = arith.select(is_gate_lane, gate_route, fx.Int32(num_max_routes))
            buffer_store(gate_value, combine_gate_res, gate_dst, cache_modifier=19)

    local_count = (num_fold_rows - warp_id + fx.Int32(_NUM_WARPS - 1)) // fx.Int32(_NUM_WARPS)
    num_grouped = (local_count // fx.Int32(_ROW_UNROLL)) * fx.Int32(_ROW_UNROLL)
    for i in range(0, num_grouped, _ROW_UNROLL):
        push_rows(
            [
                first_fold_row + warp_id + (i + fx.Int32(u)) * fx.Int32(_NUM_WARPS)
                for u in range_constexpr(_ROW_UNROLL)
            ]
        )
    for i in range(num_grouped, local_count):
        push_rows([first_fold_row + warp_id + i * fx.Int32(_NUM_WARPS)])


@ASTRewriter.transform
def topk_reduce_bf16_tile(
    thread_index: fx.Int32,
    reduce_block_idx: fx.Int32,
    num_reduce_warps: int,
    num_topk: int,
    hidden: int,
    num_experts: int,
    num_experts_per_rank: int,
    num_max_routes: int,
    rank: int,
    combine_token_res: fx.ArithValue,
    output_res: fx.ArithValue,
    topk_indices_res: fx.ArithValue,
    num_tokens_res: fx.ArithValue,
    reduce_flag_base: fx.ArithValue,
    reduce_bank: fx.Int32,
    epoch: fx.Int64,
    combine_gate_res: Optional[fx.ArithValue] = None,
    grad_topk_weights_res: Optional[fx.ArithValue] = None,
    has_grad_gate: bool = False,
):
    """Sum each token's primary-route combine rows into its output row once their reduce flags reach epoch."""
    f32_vec = fx.T.VectorType.get([_PVEC], fx.T.f32())
    bf16_vec = fx.T.VectorType.get([_PVEC], fx.T.bf16())
    num_vec_chunks = hidden // _PVEC
    lane_id = thread_index % fx.Int32(_WARP)
    warp_id = thread_index // fx.Int32(_WARP)
    num_tokens = buffer_load(num_tokens_res, fx.Int32(rank), vec_width=1, dtype=fx.T.i32())
    token = reduce_block_idx * fx.Int32(_NUM_WARPS) + warp_id
    while token < num_tokens:
        route_base = token * fx.Int32(num_topk)
        expert_ids = []
        for j in range_constexpr(num_topk):
            expert_id = buffer_load(topk_indices_res, route_base + fx.Int32(j), vec_width=1, dtype=fx.T.i64())
            is_valid = (expert_id >= fx.Int64(0)) & (expert_id < fx.Int64(num_experts))
            expert_ids.append(arith.select(is_valid, cast(expert_id, fx.T.i32()), fx.Int32(-1)))
        is_primary = [is_primary_route(expert_ids, j, num_experts_per_rank) for j in range(num_topk)]
        for j in range_constexpr(num_topk):
            if is_primary[j]:
                if lane_id == fx.Int32(0):
                    spin_until_flag_reaches(
                        reduce_flag_base, reduce_bank + route_base + fx.Int32(j), epoch, "sys", rank, "reduce"
                    )
        fx.gpu.barrier()

        # a non-primary route points past num_records, so its load reads 0 and moves no bytes
        route_offsets = [
            arith.select(
                is_primary[j],
                (route_base + fx.Int32(j)) * fx.Int32(hidden),
                fx.Int32(num_max_routes * hidden),
            )
            for j in range(num_topk)
        ]
        vec_idx = lane_id
        while vec_idx < fx.Int32(num_vec_chunks):
            col = vec_idx * fx.Int32(_PVEC)
            row_sum = None
            for j in range_constexpr(num_topk):
                value = buffer_load(
                    combine_token_res,
                    route_offsets[j] + col,
                    vec_width=_PVEC,
                    dtype=fx.T.bf16(),
                    cache_modifier=19,  # sc0|sc1|nt: system-visible non-temporal read.
                )
                term = fx.arith.extf(f32_vec, value)
                row_sum = term if row_sum is None else fx.arith.addf(row_sum, term)
            buffer_store(fx.arith.trunc_f(bf16_vec, row_sum), output_res, token * fx.Int32(hidden) + col)
            vec_idx = vec_idx + fx.Int32(_WARP)
        if const_expr(has_grad_gate):
            # every valid route's dst has a primary route whose flag was awaited above
            for j in range_constexpr(num_topk):
                if lane_id == fx.Int32(0):
                    gate_value = buffer_load(
                        combine_gate_res,
                        route_base + fx.Int32(j),
                        vec_width=1,
                        dtype=fx.T.f32(),
                        cache_modifier=19,
                    )
                    is_valid = expert_ids[j] >= fx.Int32(0)
                    gate_value = arith.select(is_valid, gate_value, fx.Float32(0.0))
                    buffer_store(gate_value, grad_topk_weights_res, route_base + fx.Int32(j))
        token = token + fx.Int32(num_reduce_warps)
