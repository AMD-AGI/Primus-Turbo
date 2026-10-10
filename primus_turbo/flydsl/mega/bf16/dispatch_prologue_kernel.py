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

"""Fused MoE dispatch-prologue kernel (FlyDSL)."""

import functools
import itertools
from typing import NamedTuple

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl.expr.buffer_ops import (
    _unwrap_value,
    buffer_load,
    buffer_store,
    create_buffer_resource,
    create_buffer_resource_from_addr,
    extract_base_index,
)
from flydsl.expr.primitive import get_dyn_shared
from flydsl.expr.primitive import ptrtoint as _fly_ptrtoint

from primus_turbo.flydsl.mega.bf16.barrier import grid_sync, xgmi_barrier
from primus_turbo.flydsl.mega.bf16.ep_intranode import same_rank_order
from primus_turbo.flydsl.mega.bf16.symm_buffer import (
    BLOCK_M,
    TOKEN_DTYPE,
    SymBuffer,
    Workspace,
    get_symm_buffer_for_mega_moe,
)
from primus_turbo.flydsl.mega.tune_utils import (
    Config,
    autotune,
)
from primus_turbo.flydsl.utils.prims import atomic_add, ld, st


class DispatchHandle(NamedTuple):
    """Per-forward routing tables produced by the dispatch prologue."""

    expert_send_dst_rank: torch.Tensor
    expert_send_count: torch.Tensor
    expert_send_offset: torch.Tensor
    dispatched_token_idx: torch.Tensor
    tile_to_expert: torch.Tensor
    num_tokens_per_expert: torch.Tensor
    num_tokens_per_expert_prefix: torch.Tensor
    num_tile_blocks: torch.Tensor
    combine_recv_dst_rank: torch.Tensor
    combine_recv_start_row: torch.Tensor
    combine_recv_fold_count: torch.Tensor
    pool_row_to_route: torch.Tensor
    pool_row_to_recv_token: torch.Tensor
    recv_token_to_pool_rows: torch.Tensor


DISPATCH_HANDLE_DTYPES = DispatchHandle(
    expert_send_dst_rank=torch.int32,
    expert_send_count=torch.int32,
    expert_send_offset=torch.int32,
    dispatched_token_idx=torch.int32,
    tile_to_expert=torch.int32,
    num_tokens_per_expert=torch.int64,
    num_tokens_per_expert_prefix=torch.int64,
    num_tile_blocks=torch.int32,
    combine_recv_dst_rank=torch.int32,
    combine_recv_start_row=torch.int32,
    combine_recv_fold_count=torch.int32,
    pool_row_to_route=torch.int32,
    pool_row_to_recv_token=torch.int32,
    recv_token_to_pool_rows=torch.int32,
)

# SCRATCH holds these per-expert i32 fields; the first six are counts reset at the end of every launch.
_NUM_SCRATCH_FIELDS = 9
_NUM_SCRATCH_COUNT_FIELDS = 6
# LDS holds these per-expert i32 fields.
_NUM_LDS_FIELDS = 6


def _make_dispatch_prologue(
    num_tokens,
    num_topk,
    num_experts,
    num_ranks,
    rank,
    num_experts_per_rank,
    block_m,
    num_max_pool_tokens,
    hidden,
    num_max_tokens_per_rank,
    grid_blocks=64,
    block_threads=256,
):
    total_pairs = num_tokens * num_topk
    grid_stride = grid_blocks * block_threads
    num_pool_blocks = num_max_pool_tokens // block_m
    num_max_recv_tokens = num_ranks * num_max_tokens_per_rank
    expert_count_bytes = num_ranks * 2 * num_experts * 4
    pool_rows_bytes = num_max_pool_tokens * 4
    (
        SCRATCH_NUM_ROUTES,
        SCRATCH_NUM_FOLD_ROWS,
        SCRATCH_NUM_SEND_ROWS,
        SCRATCH_NUM_FOLD_ROWS_RESERVED,
        SCRATCH_NUM_OTHER_ROWS_RESERVED,
        SCRATCH_NUM_SEND_ROWS_RESERVED,
        SCRATCH_SEGMENT_START,
        SCRATCH_SEND_OFFSET,
        SCRATCH_POOL_BASE,
    ) = (field * num_experts for field in range(_NUM_SCRATCH_FIELDS))
    # Phase D reuses the three phase-A counts as within-block positions (routes -> other rows).
    (
        LDS_NUM_ROUTES,
        LDS_NUM_FOLD_ROWS,
        LDS_NUM_SEND_ROWS,
        LDS_FOLD_ROWS_BASE,
        LDS_OTHER_ROWS_BASE,
        LDS_SEND_ROWS_BASE,
    ) = (field * num_experts for field in range(_NUM_LDS_FIELDS))
    phase_a_counts = (
        (LDS_NUM_ROUTES, SCRATCH_NUM_ROUTES),
        (LDS_NUM_FOLD_ROWS, SCRATCH_NUM_FOLD_ROWS),
        (LDS_NUM_SEND_ROWS, SCRATCH_NUM_SEND_ROWS),
    )

    def _ext_i64(v):
        """Sign-extend an fx i32 value to i64 (group_lens/offs stored as int64)."""
        return fx.arith.ArithValue(fx.arith.extsi(fx.T.i64(), _unwrap_value(v)), signed=True)

    @flyc.kernel(known_block_size=[block_threads, 1, 1])
    def dispatch_prologue_kernel(
        TOPK_INDICES: fx.Tensor,
        SCRATCH: fx.Tensor,
        sym_buffer: SymBuffer,
        EXPERT_SEND_DST_RANK: fx.Tensor,
        EXPERT_SEND_COUNT: fx.Tensor,
        EXPERT_SEND_OFFSET: fx.Tensor,
        TILE_TO_EXPERT: fx.Tensor,
        DISPATCHED_TOKEN_IDX: fx.Tensor,
        TOPK_WEIGHT: fx.Tensor,
        NUM_TOKENS_PER_EXPERT: fx.Tensor,
        NUM_TOKENS_PER_EXPERT_PREFIX: fx.Tensor,
        NUM_TILE_BLOCKS: fx.Tensor,
        COMBINE_RECV_DST_RANK: fx.Tensor,
        COMBINE_RECV_START_ROW: fx.Tensor,
        COMBINE_RECV_FOLD_COUNT: fx.Tensor,
    ):
        thread_index = fx.thread_idx.x
        block_index, _, _ = fx.block_idx
        # build the layout from explicit dims (bf16 path -> TOKEN_DTYPE), then hoist workspace-derived
        # region bases before dynamic control flow (rewriter can't carry Workspace)
        workspace = Workspace(
            sym_buffer.get_base_ptr(),
            num_ranks,
            num_experts,
            num_max_tokens_per_rank,
            num_topk,
            hidden,
            token_dtype=TOKEN_DTYPE,
        )
        expert_count_base = workspace.get_expert_count_buffer_ptr()
        pool_row_to_recv_token_base = workspace.get_pool_row_to_recv_token_ptr()
        pool_row_to_recv_token_bytes = workspace.num_pool_row_to_recv_token_entries * 4
        pool_row_to_route_base = workspace.get_pool_row_to_route_ptr()
        recv_token_to_pool_rows_base = workspace.get_recv_token_to_pool_rows_ptr()
        weight_recv_base = workspace.get_weight_recv_buf_ptr()

        lds_base = _unwrap_value(_fly_ptrtoint(get_dyn_shared()))
        topk_resource = create_buffer_resource(TOPK_INDICES, max_size=True)
        idx_load_dtype = TOPK_INDICES.element_type
        idx_is_i64 = fx.const_expr(idx_load_dtype.width == 64)

        def load_expert_id(elem_idx):
            value = buffer_load(topk_resource, elem_idx, vec_width=1, dtype=idx_load_dtype)
            if idx_is_i64:
                value = fx.arith.ArithValue(fx.arith.trunci(fx.T.i32(), _unwrap_value(value)), signed=True)
            return value

        def route_order(route_idx):
            # same_rank_order needs a trace-time topk position, so evaluate all and select this route's
            token_base = (route_idx // fx.Int32(num_topk)) * fx.Int32(num_topk)
            topk_position = route_idx % fx.Int32(num_topk)
            expert_ids = [load_expert_id(token_base + fx.Int32(k)) for k in fx.range_constexpr(num_topk)]
            same_rank_position = fx.Int32(0)
            num_same_rank_routes = fx.Int32(1)
            for k in fx.range_constexpr(num_topk):
                position_k, count_k = same_rank_order(expert_ids, k, num_experts_per_rank)
                is_this_route = topk_position == fx.Int32(k)
                same_rank_position = fx.arith.select(is_this_route, position_k, same_rank_position)
                num_same_rank_routes = fx.arith.select(is_this_route, count_k, num_same_rank_routes)
            return same_rank_position, num_same_rank_routes

        # All global tensors use max_size buffer descriptors.
        scratch_resource = create_buffer_resource(SCRATCH, max_size=True)
        scratch_base = extract_base_index(SCRATCH, address_space=1)

        def load_scratch(field, index):
            return buffer_load(scratch_resource, fx.Int32(field) + index, vec_width=1, dtype=fx.T.i32())

        def load_lds(field, index):
            return ld(lds_base, fx.Int32(field) + index, scope="workgroup", space=3)

        def store_lds(field, index, value):
            st(lds_base, fx.Int32(field) + index, value, scope="workgroup", space=3)

        expert_send_dst_rank_resource = create_buffer_resource(EXPERT_SEND_DST_RANK, max_size=True)
        expert_send_count_resource = create_buffer_resource(EXPERT_SEND_COUNT, max_size=True)
        expert_send_offset_resource = create_buffer_resource(EXPERT_SEND_OFFSET, max_size=True)
        tile_to_expert_resource = create_buffer_resource(TILE_TO_EXPERT, max_size=True)
        dispatched_token_idx_resource = create_buffer_resource(DISPATCHED_TOKEN_IDX, max_size=True)
        topk_weight_resource = create_buffer_resource(TOPK_WEIGHT, max_size=True)
        num_tokens_per_expert_resource = create_buffer_resource(NUM_TOKENS_PER_EXPERT, max_size=True)
        num_tokens_per_expert_prefix_resource = create_buffer_resource(
            NUM_TOKENS_PER_EXPERT_PREFIX, max_size=True
        )

        num_tile_blocks_resource = create_buffer_resource(NUM_TILE_BLOCKS, max_size=True)

        pool_row_to_recv_token_resource = create_buffer_resource_from_addr(
            pool_row_to_recv_token_base, num_records_bytes=pool_row_to_recv_token_bytes
        )
        # combine recv-segment table: one (local_expert, source_rank) entry each
        combine_recv_dst_rank_resource = create_buffer_resource(COMBINE_RECV_DST_RANK, max_size=True)
        combine_recv_start_row_resource = create_buffer_resource(COMBINE_RECV_START_ROW, max_size=True)
        combine_recv_fold_count_resource = create_buffer_resource(COMBINE_RECV_FOLD_COUNT, max_size=True)
        # Init the per-pool-block expert table (sentinel = num_experts_per_rank for unused blocks).
        pool_block_init_idx = block_index * fx.Int32(block_threads) + thread_index
        while pool_block_init_idx < fx.Int32(num_pool_blocks):
            buffer_store(fx.Int32(num_experts_per_rank), tile_to_expert_resource, pool_block_init_idx)
            pool_block_init_idx = pool_block_init_idx + fx.Int32(grid_stride)

        lds_clear_idx = thread_index
        while lds_clear_idx < fx.Int32(3 * num_experts):
            st(lds_base, lds_clear_idx, fx.Int32(0), scope="workgroup", space=3)
            lds_clear_idx = lds_clear_idx + fx.Int32(block_threads)
        fx.gpu.barrier()
        route_idx = block_index * fx.Int32(block_threads) + thread_index
        while route_idx < fx.Int32(total_pairs):
            expert_id = load_expert_id(route_idx)
            if expert_id >= fx.Int32(0):
                same_rank_position, num_same_rank_routes = route_order(route_idx)
                atomic_add(lds_base, fx.Int32(LDS_NUM_ROUTES) + expert_id, fx.Int32(1), "workgroup", 3)
                if same_rank_position == num_same_rank_routes - fx.Int32(1):
                    atomic_add(lds_base, fx.Int32(LDS_NUM_FOLD_ROWS) + expert_id, fx.Int32(1), "workgroup", 3)
                if same_rank_position == fx.Int32(0):
                    atomic_add(lds_base, fx.Int32(LDS_NUM_SEND_ROWS) + expert_id, fx.Int32(1), "workgroup", 3)
            route_idx = route_idx + fx.Int32(grid_stride)
        fx.gpu.barrier()
        lds_flush_idx = thread_index
        while lds_flush_idx < fx.Int32(num_experts):
            for field in fx.range_constexpr(len(phase_a_counts)):
                lds_field, scratch_field = phase_a_counts[field]
                block_count = load_lds(lds_field, lds_flush_idx)
                if block_count > fx.Int32(0):
                    atomic_add(scratch_base, fx.Int32(scratch_field) + lds_flush_idx, block_count, "agent", 1)
            lds_flush_idx = lds_flush_idx + fx.Int32(block_threads)
        grid_sync(workspace, thread_index, block_index, grid_blocks, rank, "dispatch_prologue/A:histogram")

        xgmi_barrier(
            workspace,
            sym_buffer,
            rank,
            num_ranks,
            thread_index,
            block_index,
            True,
            "dispatch_prologue/B1:all-entered",
        )
        if block_index == fx.Int32(0):
            for peer_rank in range(num_ranks):
                peer_c_resource = create_buffer_resource_from_addr(
                    sym_buffer.map(expert_count_base, fx.Int32(peer_rank)),
                    num_records_bytes=expert_count_bytes,
                )
                push_expert_idx = thread_index
                while push_expert_idx < fx.Int32(num_experts):
                    # expert_count[src] = (route counts, fold row counts)
                    for field in fx.range_constexpr(2):
                        count_value = load_scratch(phase_a_counts[field][1], push_expert_idx)
                        buffer_store(
                            count_value,
                            peer_c_resource,
                            fx.Int32((rank * 2 + field) * num_experts) + push_expert_idx,
                        )
                    push_expert_idx = push_expert_idx + fx.Int32(block_threads)
        xgmi_barrier(
            workspace,
            sym_buffer,
            rank,
            num_ranks,
            thread_index,
            block_index,
            False,
            "dispatch_prologue/B3:all-gather-landed",
        )

        if block_index == fx.Int32(0):
            if thread_index < fx.Int32(num_ranks):
                running_pool_offset = fx.Int32(0)
                for local_expert_idx in range(num_experts_per_rank):
                    expert_total_count = fx.Int32(0)
                    for source_rank in range(num_ranks):
                        expert_total_count = expert_total_count + ld(
                            expert_count_base,
                            fx.Int32(source_rank * 2 * num_experts + local_expert_idx)
                            + thread_index * fx.Int32(num_experts_per_rank),
                            scope="sys",
                        )
                    padded_count = (
                        (expert_total_count + fx.Int32(block_m - 1)) // fx.Int32(block_m)
                    ) * fx.Int32(block_m)
                    buffer_store(
                        running_pool_offset,
                        scratch_resource,
                        fx.Int32(SCRATCH_POOL_BASE)
                        + thread_index * fx.Int32(num_experts_per_rank)
                        + fx.Int32(local_expert_idx),
                    )
                    running_pool_offset = running_pool_offset + padded_count
            fx.gpu.barrier()
            expert_idx = thread_index
            while expert_idx < fx.Int32(num_experts):
                preceding_count = fx.Int32(0)
                for source_rank in range(rank):
                    preceding_count = preceding_count + ld(
                        expert_count_base, fx.Int32(source_rank * 2 * num_experts) + expert_idx, scope="sys"
                    )
                pool_base_value = load_scratch(SCRATCH_POOL_BASE, expert_idx)
                buffer_store(
                    pool_base_value + preceding_count,
                    scratch_resource,
                    fx.Int32(SCRATCH_SEGMENT_START) + expert_idx,
                )
                expert_idx = expert_idx + fx.Int32(block_threads)
            fx.gpu.barrier()
            comm_task_idx = thread_index
            while comm_task_idx < fx.Int32(num_experts):
                destination_rank = comm_task_idx % fx.Int32(num_ranks)
                local_expert_idx = comm_task_idx // fx.Int32(num_ranks)
                expert_id = destination_rank * fx.Int32(num_experts_per_rank) + local_expert_idx
                # only primary routes are sent
                count_value = load_scratch(SCRATCH_NUM_SEND_ROWS, expert_id)
                buffer_store(destination_rank, expert_send_dst_rank_resource, comm_task_idx)
                buffer_store(count_value, expert_send_count_resource, comm_task_idx)
                comm_task_idx = comm_task_idx + fx.Int32(block_threads)
            fx.gpu.barrier()
            if thread_index == fx.Int32(0):
                source_offset = fx.Int32(0)
                comm_task_counter = 0
                for local_expert_idx in range(num_experts_per_rank):
                    for destination_rank in range(num_ranks):
                        expert_id = destination_rank * num_experts_per_rank + local_expert_idx
                        count_value = buffer_load(
                            expert_send_count_resource,
                            fx.Int32(comm_task_counter),
                            vec_width=1,
                            dtype=fx.T.i32(),
                        )
                        buffer_store(source_offset, expert_send_offset_resource, fx.Int32(comm_task_counter))
                        buffer_store(
                            source_offset, scratch_resource, fx.Int32(SCRATCH_SEND_OFFSET + expert_id)
                        )
                        source_offset = source_offset + count_value
                        comm_task_counter = comm_task_counter + 1
            if thread_index < fx.Int32(num_experts_per_rank):
                local_expert_idx = thread_index
                expert_pool_base = load_scratch(
                    SCRATCH_POOL_BASE + rank * num_experts_per_rank, local_expert_idx
                )
                source_counts = []
                source_fold_counts = []
                for source_rank in fx.range_constexpr(num_ranks):
                    source_count_idx = (
                        fx.Int32(source_rank * 2 * num_experts + rank * num_experts_per_rank)
                        + local_expert_idx
                    )
                    source_counts.append(ld(expert_count_base, source_count_idx, scope="sys"))
                    source_fold_counts.append(
                        ld(expert_count_base, source_count_idx + fx.Int32(num_experts), scope="sys")
                    )
                expert_total_count = fx.Int32(0)
                for source_rank in fx.range_constexpr(num_ranks):
                    expert_total_count = expert_total_count + source_counts[source_rank]
                padded_count = ((expert_total_count + fx.Int32(block_m - 1)) // fx.Int32(block_m)) * fx.Int32(
                    block_m
                )
                buffer_store(_ext_i64(expert_total_count), num_tokens_per_expert_resource, local_expert_idx)
                buffer_store(
                    _ext_i64(expert_pool_base), num_tokens_per_expert_prefix_resource, local_expert_idx
                )
                num_expert_blocks = padded_count // fx.Int32(block_m)
                base_block_idx = expert_pool_base // fx.Int32(block_m)
                pool_block_offset = fx.Int32(0)
                while pool_block_offset < num_expert_blocks:
                    buffer_store(
                        local_expert_idx, tile_to_expert_resource, base_block_idx + pool_block_offset
                    )
                    pool_block_offset = pool_block_offset + fx.Int32(1)
                within_expert_offset = fx.Int32(0)
                for source_rank in fx.range_constexpr(num_ranks):
                    count_value = source_counts[source_rank]
                    # emit combine recv-segment (push these rows back to source_rank)
                    segment_idx = local_expert_idx * fx.Int32(num_ranks) + fx.Int32(source_rank)
                    buffer_store(fx.Int32(source_rank), combine_recv_dst_rank_resource, segment_idx)
                    buffer_store(
                        expert_pool_base + within_expert_offset, combine_recv_start_row_resource, segment_idx
                    )
                    buffer_store(
                        source_fold_counts[source_rank], combine_recv_fold_count_resource, segment_idx
                    )
                    within_expert_offset = within_expert_offset + count_value
                is_last_expert = local_expert_idx == fx.Int32(num_experts_per_rank - 1)
                # padding rows, plus BLOCK_M rows past the last expert, gather out of bounds and read 0
                sentinel_end = (
                    expert_pool_base
                    + padded_count
                    + fx.arith.select(is_last_expert, fx.Int32(block_m), fx.Int32(0))
                )
                sentinel_row = expert_pool_base + expert_total_count
                while sentinel_row < sentinel_end:
                    buffer_store(fx.Int32(num_max_recv_tokens), pool_row_to_recv_token_resource, sentinel_row)
                    sentinel_row = sentinel_row + fx.Int32(1)
                if is_last_expert:
                    total_rows = expert_pool_base + padded_count
                    buffer_store(total_rows // fx.Int32(block_m), num_tile_blocks_resource, fx.Int32(0))
                    buffer_store(
                        _ext_i64(total_rows),
                        num_tokens_per_expert_prefix_resource,
                        fx.Int32(num_experts_per_rank),
                    )

        grid_sync(workspace, thread_index, block_index, grid_blocks, rank, "dispatch_prologue/C:table-built")

        # Reuse Phase A's per-block counts in LDS (untouched by barriers) -- skip clear + recount.
        reserve_idx = thread_index
        while reserve_idx < fx.Int32(num_experts):
            num_routes = load_lds(LDS_NUM_ROUTES, reserve_idx)
            if num_routes > fx.Int32(0):
                num_fold_rows = load_lds(LDS_NUM_FOLD_ROWS, reserve_idx)
                num_send_rows = load_lds(LDS_NUM_SEND_ROWS, reserve_idx)
                segment_start = load_scratch(SCRATCH_SEGMENT_START, reserve_idx)
                # fold rows lead each segment, so the other rows start after this rank's fold total
                segment_num_fold_rows = load_scratch(SCRATCH_NUM_FOLD_ROWS, reserve_idx)
                send_offset = load_scratch(SCRATCH_SEND_OFFSET, reserve_idx)
                fold_rows_offset = atomic_add(
                    scratch_base,
                    fx.Int32(SCRATCH_NUM_FOLD_ROWS_RESERVED) + reserve_idx,
                    num_fold_rows,
                    "agent",
                    1,
                )
                other_rows_offset = atomic_add(
                    scratch_base,
                    fx.Int32(SCRATCH_NUM_OTHER_ROWS_RESERVED) + reserve_idx,
                    num_routes - num_fold_rows,
                    "agent",
                    1,
                )
                send_rows_offset = atomic_add(
                    scratch_base,
                    fx.Int32(SCRATCH_NUM_SEND_ROWS_RESERVED) + reserve_idx,
                    num_send_rows,
                    "agent",
                    1,
                )
                store_lds(LDS_FOLD_ROWS_BASE, reserve_idx, segment_start + fold_rows_offset)
                store_lds(
                    LDS_OTHER_ROWS_BASE,
                    reserve_idx,
                    segment_start + segment_num_fold_rows + other_rows_offset,
                )
                store_lds(LDS_SEND_ROWS_BASE, reserve_idx, send_offset + send_rows_offset)
                for field in fx.range_constexpr(len(phase_a_counts)):
                    store_lds(phase_a_counts[field][0], reserve_idx, fx.Int32(0))
            reserve_idx = reserve_idx + fx.Int32(block_threads)
        fx.gpu.barrier()
        route_idx = block_index * fx.Int32(block_threads) + thread_index
        while route_idx < fx.Int32(total_pairs):
            expert_id = load_expert_id(route_idx)
            if expert_id >= fx.Int32(0):
                same_rank_position, num_same_rank_routes = route_order(route_idx)
                token_idx = route_idx // fx.Int32(num_topk)
                is_fold = same_rank_position == num_same_rank_routes - fx.Int32(1)
                row_position_idx = (
                    fx.arith.select(is_fold, fx.Int32(LDS_NUM_FOLD_ROWS), fx.Int32(LDS_NUM_ROUTES))
                    + expert_id
                )
                row_base_idx = (
                    fx.arith.select(is_fold, fx.Int32(LDS_FOLD_ROWS_BASE), fx.Int32(LDS_OTHER_ROWS_BASE))
                    + expert_id
                )
                local_position = atomic_add(lds_base, row_position_idx, fx.Int32(1), "workgroup", 3)
                destination_row = ld(lds_base, row_base_idx, scope="workgroup", space=3) + local_position
                if same_rank_position == fx.Int32(0):
                    send_position = atomic_add(
                        lds_base, fx.Int32(LDS_NUM_SEND_ROWS) + expert_id, fx.Int32(1), "workgroup", 3
                    )
                    send_idx = load_lds(LDS_SEND_ROWS_BASE, expert_id) + send_position
                    buffer_store(token_idx, dispatched_token_idx_resource, send_idx)
                routing_weight = buffer_load(topk_weight_resource, route_idx, vec_width=1, dtype=fx.T.f32())
                destination_rank = expert_id // fx.Int32(num_experts_per_rank)
                recv_token_idx = fx.Int32(rank * num_max_tokens_per_rank) + token_idx
                # Symmetric buffers on the destination rank.
                peer_pool_row_to_recv_token_resource = create_buffer_resource_from_addr(
                    sym_buffer.map(pool_row_to_recv_token_base, destination_rank),
                    num_records_bytes=pool_row_to_recv_token_bytes,
                )
                peer_pool_row_to_route_resource = create_buffer_resource_from_addr(
                    sym_buffer.map(pool_row_to_route_base, destination_rank),
                    num_records_bytes=pool_rows_bytes,
                )
                peer_weight_resource = create_buffer_resource_from_addr(
                    sym_buffer.map(weight_recv_base, destination_rank),
                    num_records_bytes=pool_rows_bytes,
                )
                # exact size: the -1 tail stores below drop out of range when not needed
                peer_recv_token_to_pool_rows_resource = create_buffer_resource_from_addr(
                    sym_buffer.map(recv_token_to_pool_rows_base, destination_rank),
                    num_records_bytes=num_max_recv_tokens * num_topk * 4,
                )
                buffer_store(recv_token_idx, peer_pool_row_to_recv_token_resource, destination_row)
                buffer_store(route_idx, peer_pool_row_to_route_resource, destination_row)
                buffer_store(routing_weight, peer_weight_resource, destination_row)
                recv_token_rows_idx = recv_token_idx * fx.Int32(num_topk)
                buffer_store(
                    destination_row,
                    peer_recv_token_to_pool_rows_resource,
                    recv_token_rows_idx + same_rank_position,
                )
                for tail_position in fx.range_constexpr(1, num_topk):
                    is_tail = is_fold & (fx.Int32(tail_position) >= num_same_rank_routes)
                    buffer_store(
                        fx.Int32(-1),
                        peer_recv_token_to_pool_rows_resource,
                        fx.arith.select(
                            is_tail,
                            recv_token_rows_idx + fx.Int32(tail_position),
                            fx.Int32(num_max_recv_tokens * num_topk),
                        ),
                    )
            route_idx = route_idx + fx.Int32(grid_stride)

        grid_sync(workspace, thread_index, block_index, grid_blocks, rank, "dispatch_prologue/D:scatter-done")

        reset_idx = block_index * fx.Int32(block_threads) + thread_index
        while reset_idx < fx.Int32(_NUM_SCRATCH_COUNT_FIELDS * num_experts):
            buffer_store(fx.Int32(0), scratch_resource, reset_idx)
            reset_idx = reset_idx + fx.Int32(grid_stride)

        xgmi_barrier(
            workspace,
            sym_buffer,
            rank,
            num_ranks,
            thread_index,
            block_index,
            False,
            "dispatch_prologue/E:origins-landed",
        )

    # Return the raw KernelFunction; the @flyc.jit launcher below drives launch.
    return dispatch_prologue_kernel


@functools.lru_cache(maxsize=8)
def _dispatch_prologue_scratch_cached(num_experts, device):
    return torch.zeros(_NUM_SCRATCH_FIELDS * num_experts, dtype=torch.int32, device=device)


def get_dispatch_prologue_scratch(num_experts, device="cuda"):
    dev = torch.device(device)
    if dev.type == "cuda" and dev.index is None:
        dev = torch.device("cuda", torch.cuda.current_device())
    return _dispatch_prologue_scratch_cached(int(num_experts), dev)


@autotune(
    configs=[
        Config(num_cu=num_cu, num_threads=num_threads)
        for num_cu, num_threads in itertools.product((32, 64, 96), (256, 512, 1024))
    ],
    rep=5,
    # Retune per shape; topk_idx dtype auto-joins the key via the tensor arg.
    key=[
        "num_tokens",
        "num_topk",
        "num_experts",
        "num_ranks",
        "rank",
        "num_experts_per_rank",
        "block_m",
        "num_max_pool_tokens",
    ],
)
@flyc.jit
def _compiled_dispatch_prologue(
    topk_idx_flat,
    scratch,
    sym_buffer,
    expert_send_dst_rank,
    expert_send_count,
    expert_send_offset,
    tile_to_expert,
    dispatched_token_idx,
    topk_weight_flat,
    num_tokens_per_expert,
    num_tokens_per_expert_prefix,
    num_tile_blocks,
    combine_recv_dst_rank,
    combine_recv_start_row,
    combine_recv_fold_count,
    num_tokens: fx.Constexpr[int],
    num_topk: fx.Constexpr[int],
    num_experts: fx.Constexpr[int],
    num_ranks: fx.Constexpr[int],
    rank: fx.Constexpr[int],
    num_experts_per_rank: fx.Constexpr[int],
    block_m: fx.Constexpr[int],
    num_max_pool_tokens: fx.Constexpr[int],
    hidden: fx.Constexpr[int],
    num_max_tokens_per_rank: fx.Constexpr[int],
    num_cu: fx.Constexpr[int],
    num_threads: fx.Constexpr[int],
    stream: fx.Stream,
):
    kernel = _make_dispatch_prologue(
        num_tokens,
        num_topk,
        num_experts,
        num_ranks,
        rank,
        num_experts_per_rank,
        block_m,
        num_max_pool_tokens,
        hidden,
        num_max_tokens_per_rank,
        grid_blocks=num_cu,
        block_threads=num_threads,
    )
    kernel(
        topk_idx_flat,
        scratch,
        sym_buffer,
        expert_send_dst_rank,
        expert_send_count,
        expert_send_offset,
        tile_to_expert,
        dispatched_token_idx,
        topk_weight_flat,
        num_tokens_per_expert,
        num_tokens_per_expert_prefix,
        num_tile_blocks,
        combine_recv_dst_rank,
        combine_recv_start_row,
        combine_recv_fold_count,
    ).launch(
        grid=(num_cu, 1, 1),
        block=(num_threads, 1, 1),
        stream=stream,
        smem=_NUM_LDS_FIELDS * num_experts * 4,
    )


def dispatch_prologue_flydsl_kernel(symm, topk_idx, topk_weight):
    """Build the routing tables on ``symm``; returns ``(handle, dispatch_weights)``."""
    if topk_idx.dtype not in (torch.int32, torch.int64):
        raise TypeError(f"topk_idx must be int32 or int64, got {topk_idx.dtype}")
    num_tokens, num_topk = topk_idx.shape
    topk_idx_flat = topk_idx.contiguous().view(-1)
    dev = topk_idx.device
    num_experts = symm.num_experts
    num_experts_per_rank = num_experts // symm.num_ranks
    num_max_pool_tokens = symm.num_max_pool_tokens
    scratch = get_dispatch_prologue_scratch(num_experts, device=dev)

    expert_send_dst_rank = torch.empty(num_experts, dtype=torch.int32, device=dev)
    expert_send_count = torch.empty(num_experts, dtype=torch.int32, device=dev)
    expert_send_offset = torch.empty(num_experts, dtype=torch.int32, device=dev)
    tile_to_expert = torch.empty(num_max_pool_tokens // BLOCK_M, dtype=torch.int32, device=dev)
    dispatched_token_idx = torch.empty(num_max_pool_tokens, dtype=torch.int32, device=dev)
    num_tokens_per_expert = torch.empty(num_experts_per_rank, dtype=torch.int64, device=dev)
    num_tokens_per_expert_prefix = torch.empty(num_experts_per_rank + 1, dtype=torch.int64, device=dev)
    num_tile_blocks = torch.empty(1, dtype=torch.int32, device=dev)
    combine_recv_dst_rank = torch.empty(num_experts, dtype=torch.int32, device=dev)
    combine_recv_start_row = torch.empty(num_experts, dtype=torch.int32, device=dev)
    combine_recv_fold_count = torch.empty(num_experts, dtype=torch.int32, device=dev)
    if topk_weight is not None:
        topk_weight_flat = topk_weight.to(torch.float32).contiguous().view(-1)
    else:
        topk_weight_flat = torch.zeros(num_tokens * num_topk, dtype=torch.float32, device=dev)

    _compiled_dispatch_prologue(
        topk_idx_flat=topk_idx_flat,
        scratch=scratch,
        sym_buffer=symm.get_sym_buffer(),
        expert_send_dst_rank=expert_send_dst_rank,
        expert_send_count=expert_send_count,
        expert_send_offset=expert_send_offset,
        tile_to_expert=tile_to_expert,
        dispatched_token_idx=dispatched_token_idx,
        topk_weight_flat=topk_weight_flat,
        num_tokens_per_expert=num_tokens_per_expert,
        num_tokens_per_expert_prefix=num_tokens_per_expert_prefix,
        num_tile_blocks=num_tile_blocks,
        combine_recv_dst_rank=combine_recv_dst_rank,
        combine_recv_start_row=combine_recv_start_row,
        combine_recv_fold_count=combine_recv_fold_count,
        num_tokens=num_tokens,
        num_topk=num_topk,
        num_experts=num_experts,
        num_ranks=symm.num_ranks,
        rank=symm.rank,
        num_experts_per_rank=num_experts_per_rank,
        block_m=BLOCK_M,
        num_max_pool_tokens=num_max_pool_tokens,
        hidden=symm.hidden,
        num_max_tokens_per_rank=symm.num_max_tokens_per_rank,
        stream=torch.cuda.current_stream(),
    )
    # the receive tables and weights live in the shared heap, which the next prologue overwrites
    handle = DispatchHandle(
        expert_send_dst_rank=expert_send_dst_rank,
        expert_send_count=expert_send_count,
        expert_send_offset=expert_send_offset,
        dispatched_token_idx=dispatched_token_idx,
        tile_to_expert=tile_to_expert,
        num_tokens_per_expert=num_tokens_per_expert,
        num_tokens_per_expert_prefix=num_tokens_per_expert_prefix,
        num_tile_blocks=num_tile_blocks,
        combine_recv_dst_rank=combine_recv_dst_rank,
        combine_recv_start_row=combine_recv_start_row,
        combine_recv_fold_count=combine_recv_fold_count,
        pool_row_to_route=symm.pool_row_to_route.clone(),
        pool_row_to_recv_token=symm.pool_row_to_recv_token.clone(),
        recv_token_to_pool_rows=symm.recv_token_to_pool_rows.clone(),
    )
    return handle, symm.weight_recv_buf.clone()


def run_dispatch_prologue(x, w1, group, topk_idx, topk_weights):
    """Activate the symm buffer for this shape and run the prologue on it."""
    num_experts_per_rank = w1.shape[0]
    symm = get_symm_buffer_for_mega_moe(
        group,
        num_experts=num_experts_per_rank * group.size(),
        num_max_tokens_per_rank=x.shape[0],
        num_topk=topk_idx.shape[-1],
        hidden=x.shape[1],
        intermediate_hidden=w1.shape[1] // 2,
    )
    return dispatch_prologue_flydsl_kernel(symm, topk_idx, topk_weights)
