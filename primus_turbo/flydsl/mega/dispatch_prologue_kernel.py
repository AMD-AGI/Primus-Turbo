###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Fused MoE dispatch-prologue kernel (FlyDSL)."""

import functools
import itertools

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl import Config, autotune
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

from primus_turbo.flydsl.mega.barrier import grid_sync, xgmi_barrier
from primus_turbo.flydsl.mega.prims import atomic_add, ld, st
from primus_turbo.flydsl.mega.symm_buffer import SymLayout, sym_map
from primus_turbo.flydsl.mega.tune_utils import _suppress_stdout_stderr


def _make_dispatch_prologue(
    num_tokens,
    num_topk,
    num_experts,
    num_ranks,
    rank,
    experts_per_rank,
    block_m,
    num_max_pool_tokens,
    grid_blocks=64,
    block_threads=256,
):
    total_pairs = num_tokens * num_topk
    grid_stride = grid_blocks * block_threads
    num_pool_blocks = num_max_pool_tokens // block_m
    c_buffer_bytes = num_ranks * num_experts * 4
    origin_buffer_bytes = num_max_pool_tokens * 4
    WS_SEND, WS_WITHIN = 0, num_experts
    WS_START, WS_SROFF, WS_POOLBASE = 2 * num_experts, 3 * num_experts, 4 * num_experts

    def _ext_i64(v):
        """Sign-extend an fx i32 value to i64 (group_lens/offs stored as int64)."""
        return fx.arith.ArithValue(fx.arith.extsi(fx.T.i64(), _unwrap_value(v)), signed=True)

    @flyc.kernel(known_block_size=[block_threads, 1, 1])
    def dispatch_prologue_kernel(
        TOPK_INDICES: fx.Tensor,
        WORKSPACE: fx.Tensor,
        sl: SymLayout,
        EXPERT_SEND_DST_RANK: fx.Tensor,
        EXPERT_SEND_DST_ROW: fx.Tensor,
        EXPERT_SEND_COUNT: fx.Tensor,
        EXPERT_SEND_OFFSET: fx.Tensor,
        TILE_TO_EXPERT: fx.Tensor,
        TILE_EXPECTED: fx.Tensor,
        DISPATCHED_TOKEN_IDX: fx.Tensor,
        TOPK_WEIGHT: fx.Tensor,
        NUM_TOKENS_PER_EXPERT: fx.Tensor,
        NUM_TOKENS_PER_EXPERT_PREFIX: fx.Tensor,
    ):
        thread_index = fx.thread_idx.x
        block_index, _, _ = fx.block_idx

        lds_base = _unwrap_value(_fly_ptrtoint(get_dyn_shared()))
        topk_resource = create_buffer_resource(TOPK_INDICES, max_size=True)
        idx_load_dtype = TOPK_INDICES.element_type
        idx_is_i64 = fx.const_expr(idx_load_dtype.width == 64)

        def load_expert_id(elem_index):
            value = buffer_load(topk_resource, elem_index, vec_width=1, dtype=idx_load_dtype)
            if idx_is_i64:
                value = fx.arith.ArithValue(fx.arith.trunci(fx.T.i32(), _unwrap_value(value)), signed=True)
            return value

        # All global tensors use max_size buffer descriptors.
        workspace_resource = create_buffer_resource(WORKSPACE, max_size=True)
        workspace_base = extract_base_index(WORKSPACE, address_space=1)
        expert_send_dst_rank_resource = create_buffer_resource(EXPERT_SEND_DST_RANK, max_size=True)
        expert_send_dst_row_resource = create_buffer_resource(EXPERT_SEND_DST_ROW, max_size=True)
        expert_send_count_resource = create_buffer_resource(EXPERT_SEND_COUNT, max_size=True)
        expert_send_offset_resource = create_buffer_resource(EXPERT_SEND_OFFSET, max_size=True)
        tile_to_expert_resource = create_buffer_resource(TILE_TO_EXPERT, max_size=True)
        tile_expected_resource = create_buffer_resource(TILE_EXPECTED, max_size=True)
        dispatched_token_idx_resource = create_buffer_resource(DISPATCHED_TOKEN_IDX, max_size=True)
        topk_weight_resource = create_buffer_resource(TOPK_WEIGHT, max_size=True)
        num_tokens_per_expert_resource = create_buffer_resource(NUM_TOKENS_PER_EXPERT, max_size=True)
        num_tokens_per_expert_prefix_resource = create_buffer_resource(
            NUM_TOKENS_PER_EXPERT_PREFIX, max_size=True
        )
        meta_scalars_resource = create_buffer_resource_from_addr(sl.meta_scalars, num_records_bytes=8 * 4)

        my_origin_rank_resource = create_buffer_resource_from_addr(
            sl.pool_src_rank, num_records_bytes=origin_buffer_bytes
        )
        # combine recv-segment table: one (local_expert, source_rank) entry each
        seg_table_bytes = num_experts * 4
        combine_recv_dst_rank_resource = create_buffer_resource_from_addr(
            sl.combine_recv_dst_rank, num_records_bytes=seg_table_bytes
        )
        combine_recv_start_row_resource = create_buffer_resource_from_addr(
            sl.combine_recv_start_row, num_records_bytes=seg_table_bytes
        )
        combine_recv_count_resource = create_buffer_resource_from_addr(
            sl.combine_recv_count, num_records_bytes=seg_table_bytes
        )
        origin_init_index = block_index * fx.Int32(block_threads) + thread_index
        while origin_init_index < fx.Int32(num_max_pool_tokens):
            buffer_store(fx.Int32(-1), my_origin_rank_resource, origin_init_index)
            origin_init_index = origin_init_index + fx.Int32(grid_stride)
        # Fused init of the two per-pool-block tables (same range).
        pool_block_init_index = block_index * fx.Int32(block_threads) + thread_index
        while pool_block_init_index < fx.Int32(num_pool_blocks):
            buffer_store(fx.Int32(0), tile_expected_resource, pool_block_init_index)
            buffer_store(fx.Int32(experts_per_rank), tile_to_expert_resource, pool_block_init_index)
            pool_block_init_index = pool_block_init_index + fx.Int32(grid_stride)

        lds_clear_index = thread_index
        while lds_clear_index < fx.Int32(num_experts):
            st(lds_base, lds_clear_index, fx.Int32(0), scope="workgroup", space=3)
            lds_clear_index = lds_clear_index + fx.Int32(block_threads)
        fx.gpu.barrier()
        pair_index = block_index * fx.Int32(block_threads) + thread_index
        while pair_index < fx.Int32(total_pairs):
            expert_id = load_expert_id(pair_index)
            if expert_id >= fx.Int32(0):
                atomic_add(lds_base, expert_id, fx.Int32(1), "workgroup", 3)
            pair_index = pair_index + fx.Int32(grid_stride)
        fx.gpu.barrier()
        lds_flush_index = thread_index
        while lds_flush_index < fx.Int32(num_experts):
            block_count = ld(lds_base, lds_flush_index, scope="workgroup", space=3)
            if block_count > fx.Int32(0):
                atomic_add(workspace_base, fx.Int32(WS_SEND) + lds_flush_index, block_count, "agent", 1)
            lds_flush_index = lds_flush_index + fx.Int32(block_threads)
        grid_sync(sl, thread_index, block_index, grid_blocks, rank, "dispatch_prologue/A:histogram")

        xgmi_barrier(sl, rank, num_ranks, thread_index, block_index, True, "dispatch_prologue/B1:all-entered")
        if block_index == fx.Int32(0):
            for peer_rank in range(num_ranks):
                peer_c_resource = create_buffer_resource_from_addr(
                    sym_map(sl, sl.expert_count_buffer, fx.Int32(peer_rank)),
                    num_records_bytes=c_buffer_bytes,
                )
                push_expert_index = thread_index
                while push_expert_index < fx.Int32(num_experts):
                    send_count_value = buffer_load(
                        workspace_resource,
                        fx.Int32(WS_SEND) + push_expert_index,
                        vec_width=1,
                        dtype=fx.T.i32(),
                    )
                    buffer_store(
                        send_count_value, peer_c_resource, fx.Int32(rank * num_experts) + push_expert_index
                    )
                    push_expert_index = push_expert_index + fx.Int32(block_threads)
        xgmi_barrier(
            sl, rank, num_ranks, thread_index, block_index, False, "dispatch_prologue/B3:all-gather-landed"
        )

        if block_index == fx.Int32(0):
            own_c_address = sl.expert_count_buffer
            if thread_index < fx.Int32(num_ranks):
                running_pool_offset = fx.Int32(0)
                for local_expert_index in range(experts_per_rank):
                    expert_total_count = fx.Int32(0)
                    for source_rank in range(num_ranks):
                        expert_total_count = expert_total_count + ld(
                            own_c_address,
                            fx.Int32(source_rank * num_experts + local_expert_index)
                            + thread_index * fx.Int32(experts_per_rank),
                            scope="sys",
                        )
                    padded_count = (
                        (expert_total_count + fx.Int32(block_m - 1)) // fx.Int32(block_m)
                    ) * fx.Int32(block_m)
                    buffer_store(
                        running_pool_offset,
                        workspace_resource,
                        fx.Int32(WS_POOLBASE)
                        + thread_index * fx.Int32(experts_per_rank)
                        + fx.Int32(local_expert_index),
                    )
                    running_pool_offset = running_pool_offset + padded_count
            fx.gpu.barrier()
            expert_index = thread_index
            while expert_index < fx.Int32(num_experts):
                preceding_count = fx.Int32(0)
                for source_rank in range(rank):
                    preceding_count = preceding_count + ld(
                        own_c_address, fx.Int32(source_rank * num_experts) + expert_index, scope="sys"
                    )
                pool_base_value = buffer_load(
                    workspace_resource, fx.Int32(WS_POOLBASE) + expert_index, vec_width=1, dtype=fx.T.i32()
                )
                buffer_store(
                    pool_base_value + preceding_count, workspace_resource, fx.Int32(WS_START) + expert_index
                )
                expert_index = expert_index + fx.Int32(block_threads)
            fx.gpu.barrier()
            comm_task_index = thread_index
            while comm_task_index < fx.Int32(num_experts):
                destination_rank = comm_task_index % fx.Int32(num_ranks)
                local_expert_index = comm_task_index // fx.Int32(num_ranks)
                expert_id = destination_rank * fx.Int32(experts_per_rank) + local_expert_index
                count_value = ld(own_c_address, fx.Int32(rank * num_experts) + expert_id, scope="sys")
                start_value = buffer_load(
                    workspace_resource, fx.Int32(WS_START) + expert_id, vec_width=1, dtype=fx.T.i32()
                )
                buffer_store(destination_rank, expert_send_dst_rank_resource, comm_task_index)
                buffer_store(start_value, expert_send_dst_row_resource, comm_task_index)
                buffer_store(count_value, expert_send_count_resource, comm_task_index)
                comm_task_index = comm_task_index + fx.Int32(block_threads)
            fx.gpu.barrier()
            if thread_index == fx.Int32(0):
                source_offset = fx.Int32(0)
                comm_task_counter = 0
                for local_expert_index in range(experts_per_rank):
                    for destination_rank in range(num_ranks):
                        expert_id = destination_rank * experts_per_rank + local_expert_index
                        count_value = buffer_load(
                            expert_send_count_resource,
                            fx.Int32(comm_task_counter),
                            vec_width=1,
                            dtype=fx.T.i32(),
                        )
                        buffer_store(source_offset, expert_send_offset_resource, fx.Int32(comm_task_counter))
                        buffer_store(source_offset, workspace_resource, fx.Int32(WS_SROFF + expert_id))
                        source_offset = source_offset + count_value
                        comm_task_counter = comm_task_counter + 1
                buffer_store(fx.Int32(num_experts), meta_scalars_resource, fx.Int32(2))
            if thread_index < fx.Int32(experts_per_rank):
                local_expert_index = thread_index
                expert_pool_base = buffer_load(
                    workspace_resource,
                    fx.Int32(WS_POOLBASE + rank * experts_per_rank) + local_expert_index,
                    vec_width=1,
                    dtype=fx.T.i32(),
                )
                source_counts = []
                for source_rank in fx.range_constexpr(num_ranks):
                    source_counts.append(
                        ld(
                            own_c_address,
                            fx.Int32(source_rank * num_experts + rank * experts_per_rank)
                            + local_expert_index,
                            scope="sys",
                        )
                    )
                expert_total_count = fx.Int32(0)
                for source_rank in fx.range_constexpr(num_ranks):
                    expert_total_count = expert_total_count + source_counts[source_rank]
                padded_count = ((expert_total_count + fx.Int32(block_m - 1)) // fx.Int32(block_m)) * fx.Int32(
                    block_m
                )
                buffer_store(_ext_i64(padded_count), num_tokens_per_expert_resource, local_expert_index)
                buffer_store(
                    _ext_i64(expert_pool_base), num_tokens_per_expert_prefix_resource, local_expert_index
                )
                num_expert_blocks = padded_count // fx.Int32(block_m)
                base_block_index = expert_pool_base // fx.Int32(block_m)
                pool_block_offset = fx.Int32(0)
                while pool_block_offset < num_expert_blocks:
                    buffer_store(
                        local_expert_index, tile_to_expert_resource, base_block_index + pool_block_offset
                    )
                    pool_block_offset = pool_block_offset + fx.Int32(1)
                within_expert_offset = fx.Int32(0)
                for source_rank in fx.range_constexpr(num_ranks):
                    count_value = source_counts[source_rank]
                    # emit combine recv-segment (push these rows back to source_rank)
                    seg_index = local_expert_index * fx.Int32(num_ranks) + fx.Int32(source_rank)
                    buffer_store(fx.Int32(source_rank), combine_recv_dst_rank_resource, seg_index)
                    buffer_store(
                        expert_pool_base + within_expert_offset, combine_recv_start_row_resource, seg_index
                    )
                    buffer_store(count_value, combine_recv_count_resource, seg_index)
                    if count_value > fx.Int32(0):
                        first_block = (expert_pool_base + within_expert_offset) // fx.Int32(block_m)
                        last_block = (
                            expert_pool_base + within_expert_offset + count_value - fx.Int32(1)
                        ) // fx.Int32(block_m)
                        block_cursor = first_block
                        while block_cursor <= last_block:
                            expected_value = buffer_load(
                                tile_expected_resource, block_cursor, vec_width=1, dtype=fx.T.i32()
                            )
                            buffer_store(expected_value + fx.Int32(1), tile_expected_resource, block_cursor)
                            block_cursor = block_cursor + fx.Int32(1)
                        within_expert_offset = within_expert_offset + count_value
                if local_expert_index == fx.Int32(experts_per_rank - 1):
                    total_rows = expert_pool_base + padded_count
                    buffer_store(total_rows, meta_scalars_resource, fx.Int32(0))
                    buffer_store(total_rows // fx.Int32(block_m), meta_scalars_resource, fx.Int32(1))
                    buffer_store(
                        _ext_i64(total_rows),
                        num_tokens_per_expert_prefix_resource,
                        fx.Int32(experts_per_rank),
                    )

        grid_sync(sl, thread_index, block_index, grid_blocks, rank, "dispatch_prologue/C:table-built")

        # Reuse the per-block histogram Phase A left in LDS (identical pair range,
        # untouched by grid_sync/xgmi_barrier/Phase C) -- skip clear + recount.
        reserve_index = thread_index
        while reserve_index < fx.Int32(num_experts):
            block_expert_count = ld(lds_base, reserve_index, scope="workgroup", space=3)
            if block_expert_count > fx.Int32(0):
                reserved_base = atomic_add(
                    workspace_base,
                    fx.Int32(WS_WITHIN) + reserve_index,
                    block_expert_count,
                    "agent",
                    1,
                )
                st(
                    lds_base,
                    fx.Int32(num_experts) + reserve_index,
                    reserved_base,
                    scope="workgroup",
                    space=3,
                )
                st(lds_base, reserve_index, fx.Int32(0), scope="workgroup", space=3)
            reserve_index = reserve_index + fx.Int32(block_threads)
        fx.gpu.barrier()
        pair_index = block_index * fx.Int32(block_threads) + thread_index
        while pair_index < fx.Int32(total_pairs):
            expert_id = load_expert_id(pair_index)
            if expert_id >= fx.Int32(0):
                token_index = pair_index // fx.Int32(num_topk)
                topk_slot = pair_index % fx.Int32(num_topk)
                local_position = atomic_add(lds_base, expert_id, fx.Int32(1), "workgroup", 3)
                within_expert_position = (
                    ld(lds_base, fx.Int32(num_experts) + expert_id, scope="workgroup", space=3)
                    + local_position
                )
                expert_start = buffer_load(
                    workspace_resource, fx.Int32(WS_START) + expert_id, vec_width=1, dtype=fx.T.i32()
                )
                expert_source_offset = buffer_load(
                    workspace_resource, fx.Int32(WS_SROFF) + expert_id, vec_width=1, dtype=fx.T.i32()
                )
                destination_row = expert_start + within_expert_position
                buffer_store(
                    token_index, dispatched_token_idx_resource, expert_source_offset + within_expert_position
                )
                routing_weight = buffer_load(topk_weight_resource, pair_index, vec_width=1, dtype=fx.T.f32())
                destination_rank = expert_id // fx.Int32(experts_per_rank)
                # Symmetric buffers on the destination rank.
                peer_origin_rank_resource = create_buffer_resource_from_addr(
                    sym_map(sl, sl.pool_src_rank, destination_rank),
                    num_records_bytes=origin_buffer_bytes,
                )
                peer_origin_slot_resource = create_buffer_resource_from_addr(
                    sym_map(sl, sl.pool_src_slot, destination_rank),
                    num_records_bytes=origin_buffer_bytes,
                )
                peer_weight_resource = create_buffer_resource_from_addr(
                    sym_map(sl, sl.weight_recv_buf, destination_rank),
                    num_records_bytes=origin_buffer_bytes,
                )
                buffer_store(fx.Int32(rank), peer_origin_rank_resource, destination_row)
                buffer_store(
                    token_index * fx.Int32(num_topk) + topk_slot,
                    peer_origin_slot_resource,
                    destination_row,
                )
                buffer_store(routing_weight, peer_weight_resource, destination_row)
            pair_index = pair_index + fx.Int32(grid_stride)

        grid_sync(sl, thread_index, block_index, grid_blocks, rank, "dispatch_prologue/D:scatter-done")

        reset_index = block_index * fx.Int32(block_threads) + thread_index
        while reset_index < fx.Int32(num_experts):
            buffer_store(fx.Int32(0), workspace_resource, fx.Int32(WS_SEND) + reset_index)
            buffer_store(fx.Int32(0), workspace_resource, fx.Int32(WS_WITHIN) + reset_index)
            reset_index = reset_index + fx.Int32(grid_stride)

        xgmi_barrier(
            sl, rank, num_ranks, thread_index, block_index, False, "dispatch_prologue/E:origins-landed"
        )

    # Return the raw KernelFunction; the @flyc.jit launcher below drives launch.
    return dispatch_prologue_kernel


@functools.lru_cache(maxsize=8)
def _dispatch_prologue_workspace_cached(num_experts, device):
    return torch.zeros(5 * num_experts, dtype=torch.int32, device=device)


def get_dispatch_prologue_workspace(num_experts, device="cuda"):
    dev = torch.device(device)
    if dev.type == "cuda" and dev.index is None:
        dev = torch.device("cuda", torch.cuda.current_device())
    return _dispatch_prologue_workspace_cached(int(num_experts), dev)


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
        "experts_per_rank",
        "block_m",
        "num_max_pool_tokens",
    ],
)
@flyc.jit
def _compiled_dispatch_prologue(
    topk_idx_flat,
    workspace,
    sym_layout,
    expert_send_dst_rank,
    expert_send_dst_row,
    expert_send_count,
    expert_send_offset,
    tile_to_expert,
    tile_expected,
    dispatched_token_idx,
    topk_weight_flat,
    num_tokens_per_expert,
    num_tokens_per_expert_prefix,
    num_tokens: fx.Constexpr[int],
    num_topk: fx.Constexpr[int],
    num_experts: fx.Constexpr[int],
    num_ranks: fx.Constexpr[int],
    rank: fx.Constexpr[int],
    experts_per_rank: fx.Constexpr[int],
    block_m: fx.Constexpr[int],
    num_max_pool_tokens: fx.Constexpr[int],
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
        experts_per_rank,
        block_m,
        num_max_pool_tokens,
        grid_blocks=num_cu,
        block_threads=num_threads,
    )
    kernel(
        topk_idx_flat,
        workspace,
        sym_layout,
        expert_send_dst_rank,
        expert_send_dst_row,
        expert_send_count,
        expert_send_offset,
        tile_to_expert,
        tile_expected,
        dispatched_token_idx,
        topk_weight_flat,
        num_tokens_per_expert,
        num_tokens_per_expert_prefix,
    ).launch(
        grid=(num_cu, 1, 1),
        block=(num_threads, 1, 1),
        stream=stream,
        smem=2 * num_experts * 4,
    )


def dispatch_prologue(
    topk_idx,
    topk_weight,
    *,
    sym_layout,
    num_tokens,
    num_topk,
    num_experts,
    num_ranks,
    rank,
    experts_per_rank,
    block_m,
    num_max_pool_tokens,
):
    if topk_idx.dtype not in (torch.int32, torch.int64):
        raise TypeError(f"topk_idx must be int32 or int64, got {topk_idx.dtype}")
    topk_idx_flat = topk_idx.contiguous().view(-1)
    dev = topk_idx.device
    workspace = get_dispatch_prologue_workspace(num_experts, device=dev)

    num_max_blocks = num_max_pool_tokens // block_m
    expert_send_dst_rank = torch.empty(num_experts, dtype=torch.int32, device=dev)
    expert_send_dst_row = torch.empty(num_experts, dtype=torch.int32, device=dev)
    expert_send_count = torch.empty(num_experts, dtype=torch.int32, device=dev)
    expert_send_offset = torch.empty(num_experts, dtype=torch.int32, device=dev)
    tile_to_expert = torch.empty(num_max_blocks, dtype=torch.int32, device=dev)
    tile_expected = torch.empty(num_max_blocks, dtype=torch.int32, device=dev)
    dispatched_token_idx = torch.empty(num_max_pool_tokens, dtype=torch.int32, device=dev)
    num_tokens_per_expert = torch.empty(experts_per_rank, dtype=torch.int64, device=dev)
    num_tokens_per_expert_prefix = torch.empty(experts_per_rank + 1, dtype=torch.int64, device=dev)
    if topk_weight is not None:
        topk_weight_flat = topk_weight.to(torch.float32).contiguous().view(-1)
    else:
        topk_weight_flat = torch.zeros(num_tokens * num_topk, dtype=torch.float32, device=dev)

    stream = torch.cuda.current_stream()
    # Silence flydsl autotune progress prints (fd-level, catches JIT output too).
    with _suppress_stdout_stderr():
        _compiled_dispatch_prologue(
            topk_idx_flat=topk_idx_flat,
            workspace=workspace,
            sym_layout=sym_layout,
            expert_send_dst_rank=expert_send_dst_rank,
            expert_send_dst_row=expert_send_dst_row,
            expert_send_count=expert_send_count,
            expert_send_offset=expert_send_offset,
            tile_to_expert=tile_to_expert,
            tile_expected=tile_expected,
            dispatched_token_idx=dispatched_token_idx,
            topk_weight_flat=topk_weight_flat,
            num_tokens_per_expert=num_tokens_per_expert,
            num_tokens_per_expert_prefix=num_tokens_per_expert_prefix,
            num_tokens=num_tokens,
            num_topk=num_topk,
            num_experts=num_experts,
            num_ranks=num_ranks,
            rank=rank,
            experts_per_rank=experts_per_rank,
            block_m=block_m,
            num_max_pool_tokens=num_max_pool_tokens,
            stream=stream,
        )
    return (
        expert_send_dst_rank,
        expert_send_dst_row,
        expert_send_count,
        expert_send_offset,
        dispatched_token_idx,
        tile_to_expert,
        tile_expected,
        num_tokens_per_expert,
        num_tokens_per_expert_prefix,
    )
