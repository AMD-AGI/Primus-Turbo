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
from primus_turbo.flydsl.mega.symm_buffer import TOKEN_DTYPE, SymBuffer, Workspace
from primus_turbo.flydsl.mega.tune_utils import (
    Config,
    autotune,
)


def _make_dispatch_prologue(
    num_tokens,
    num_topk,
    num_experts,
    num_ranks,
    rank,
    experts_per_rank,
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
    c_buffer_bytes = num_ranks * num_experts * 4
    origin_buffer_bytes = num_max_pool_tokens * 4
    SCRATCH_SEND, SCRATCH_WITHIN = 0, num_experts
    SCRATCH_START, SCRATCH_SROFF, SCRATCH_POOLBASE = 2 * num_experts, 3 * num_experts, 4 * num_experts

    def _ext_i64(v):
        """Sign-extend an fx i32 value to i64 (group_lens/offs stored as int64)."""
        return fx.arith.ArithValue(fx.arith.extsi(fx.T.i64(), _unwrap_value(v)), signed=True)

    @flyc.kernel(known_block_size=[block_threads, 1, 1])
    def dispatch_prologue_kernel(
        TOPK_INDICES: fx.Tensor,
        SCRATCH: fx.Tensor,
        sym_buffer: SymBuffer,
        EXPERT_SEND_DST_RANK: fx.Tensor,
        EXPERT_SEND_DST_ROW: fx.Tensor,
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
        COMBINE_RECV_COUNT: fx.Tensor,
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
        pool_src_rank_base = workspace.get_pool_src_rank_ptr()
        pool_src_slot_base = workspace.get_pool_src_slot_ptr()
        weight_recv_base = workspace.get_weight_recv_buf_ptr()

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
        scratch_resource = create_buffer_resource(SCRATCH, max_size=True)
        scratch_base = extract_base_index(SCRATCH, address_space=1)
        expert_send_dst_rank_resource = create_buffer_resource(EXPERT_SEND_DST_RANK, max_size=True)
        expert_send_dst_row_resource = create_buffer_resource(EXPERT_SEND_DST_ROW, max_size=True)
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

        my_origin_rank_resource = create_buffer_resource_from_addr(
            pool_src_rank_base, num_records_bytes=origin_buffer_bytes
        )
        # combine recv-segment table: one (local_expert, source_rank) entry each
        combine_recv_dst_rank_resource = create_buffer_resource(COMBINE_RECV_DST_RANK, max_size=True)
        combine_recv_start_row_resource = create_buffer_resource(COMBINE_RECV_START_ROW, max_size=True)
        combine_recv_count_resource = create_buffer_resource(COMBINE_RECV_COUNT, max_size=True)
        origin_init_index = block_index * fx.Int32(block_threads) + thread_index
        while origin_init_index < fx.Int32(num_max_pool_tokens):
            buffer_store(fx.Int32(-1), my_origin_rank_resource, origin_init_index)
            origin_init_index = origin_init_index + fx.Int32(grid_stride)
        # Init the per-pool-block expert table (sentinel = experts_per_rank for unused blocks).
        pool_block_init_index = block_index * fx.Int32(block_threads) + thread_index
        while pool_block_init_index < fx.Int32(num_pool_blocks):
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
                atomic_add(scratch_base, fx.Int32(SCRATCH_SEND) + lds_flush_index, block_count, "agent", 1)
            lds_flush_index = lds_flush_index + fx.Int32(block_threads)
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
                    num_records_bytes=c_buffer_bytes,
                )
                push_expert_index = thread_index
                while push_expert_index < fx.Int32(num_experts):
                    send_count_value = buffer_load(
                        scratch_resource,
                        fx.Int32(SCRATCH_SEND) + push_expert_index,
                        vec_width=1,
                        dtype=fx.T.i32(),
                    )
                    buffer_store(
                        send_count_value, peer_c_resource, fx.Int32(rank * num_experts) + push_expert_index
                    )
                    push_expert_index = push_expert_index + fx.Int32(block_threads)
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
                for local_expert_index in range(experts_per_rank):
                    expert_total_count = fx.Int32(0)
                    for source_rank in range(num_ranks):
                        expert_total_count = expert_total_count + ld(
                            expert_count_base,
                            fx.Int32(source_rank * num_experts + local_expert_index)
                            + thread_index * fx.Int32(experts_per_rank),
                            scope="sys",
                        )
                    padded_count = (
                        (expert_total_count + fx.Int32(block_m - 1)) // fx.Int32(block_m)
                    ) * fx.Int32(block_m)
                    buffer_store(
                        running_pool_offset,
                        scratch_resource,
                        fx.Int32(SCRATCH_POOLBASE)
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
                        expert_count_base, fx.Int32(source_rank * num_experts) + expert_index, scope="sys"
                    )
                pool_base_value = buffer_load(
                    scratch_resource, fx.Int32(SCRATCH_POOLBASE) + expert_index, vec_width=1, dtype=fx.T.i32()
                )
                buffer_store(
                    pool_base_value + preceding_count,
                    scratch_resource,
                    fx.Int32(SCRATCH_START) + expert_index,
                )
                expert_index = expert_index + fx.Int32(block_threads)
            fx.gpu.barrier()
            comm_task_index = thread_index
            while comm_task_index < fx.Int32(num_experts):
                destination_rank = comm_task_index % fx.Int32(num_ranks)
                local_expert_index = comm_task_index // fx.Int32(num_ranks)
                expert_id = destination_rank * fx.Int32(experts_per_rank) + local_expert_index
                count_value = ld(expert_count_base, fx.Int32(rank * num_experts) + expert_id, scope="sys")
                start_value = buffer_load(
                    scratch_resource, fx.Int32(SCRATCH_START) + expert_id, vec_width=1, dtype=fx.T.i32()
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
                        buffer_store(source_offset, scratch_resource, fx.Int32(SCRATCH_SROFF + expert_id))
                        source_offset = source_offset + count_value
                        comm_task_counter = comm_task_counter + 1
            if thread_index < fx.Int32(experts_per_rank):
                local_expert_index = thread_index
                expert_pool_base = buffer_load(
                    scratch_resource,
                    fx.Int32(SCRATCH_POOLBASE + rank * experts_per_rank) + local_expert_index,
                    vec_width=1,
                    dtype=fx.T.i32(),
                )
                source_counts = []
                for source_rank in fx.range_constexpr(num_ranks):
                    source_counts.append(
                        ld(
                            expert_count_base,
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
                    within_expert_offset = within_expert_offset + count_value
                if local_expert_index == fx.Int32(experts_per_rank - 1):
                    total_rows = expert_pool_base + padded_count
                    buffer_store(total_rows // fx.Int32(block_m), num_tile_blocks_resource, fx.Int32(0))
                    buffer_store(
                        _ext_i64(total_rows),
                        num_tokens_per_expert_prefix_resource,
                        fx.Int32(experts_per_rank),
                    )

        grid_sync(workspace, thread_index, block_index, grid_blocks, rank, "dispatch_prologue/C:table-built")

        # Reuse Phase A's per-block histogram in LDS (untouched by barriers) -- skip clear + recount.
        reserve_index = thread_index
        while reserve_index < fx.Int32(num_experts):
            block_expert_count = ld(lds_base, reserve_index, scope="workgroup", space=3)
            if block_expert_count > fx.Int32(0):
                reserved_base = atomic_add(
                    scratch_base,
                    fx.Int32(SCRATCH_WITHIN) + reserve_index,
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
                    scratch_resource, fx.Int32(SCRATCH_START) + expert_id, vec_width=1, dtype=fx.T.i32()
                )
                expert_source_offset = buffer_load(
                    scratch_resource, fx.Int32(SCRATCH_SROFF) + expert_id, vec_width=1, dtype=fx.T.i32()
                )
                destination_row = expert_start + within_expert_position
                buffer_store(
                    token_index, dispatched_token_idx_resource, expert_source_offset + within_expert_position
                )
                routing_weight = buffer_load(topk_weight_resource, pair_index, vec_width=1, dtype=fx.T.f32())
                destination_rank = expert_id // fx.Int32(experts_per_rank)
                # Symmetric buffers on the destination rank.
                peer_origin_rank_resource = create_buffer_resource_from_addr(
                    sym_buffer.map(pool_src_rank_base, destination_rank),
                    num_records_bytes=origin_buffer_bytes,
                )
                peer_origin_slot_resource = create_buffer_resource_from_addr(
                    sym_buffer.map(pool_src_slot_base, destination_rank),
                    num_records_bytes=origin_buffer_bytes,
                )
                peer_weight_resource = create_buffer_resource_from_addr(
                    sym_buffer.map(weight_recv_base, destination_rank),
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

        grid_sync(workspace, thread_index, block_index, grid_blocks, rank, "dispatch_prologue/D:scatter-done")

        reset_index = block_index * fx.Int32(block_threads) + thread_index
        while reset_index < fx.Int32(num_experts):
            buffer_store(fx.Int32(0), scratch_resource, fx.Int32(SCRATCH_SEND) + reset_index)
            buffer_store(fx.Int32(0), scratch_resource, fx.Int32(SCRATCH_WITHIN) + reset_index)
            reset_index = reset_index + fx.Int32(grid_stride)

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
    return torch.zeros(5 * num_experts, dtype=torch.int32, device=device)


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
        "experts_per_rank",
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
    expert_send_dst_row,
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
    combine_recv_count,
    num_tokens: fx.Constexpr[int],
    num_topk: fx.Constexpr[int],
    num_experts: fx.Constexpr[int],
    num_ranks: fx.Constexpr[int],
    rank: fx.Constexpr[int],
    experts_per_rank: fx.Constexpr[int],
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
        experts_per_rank,
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
        expert_send_dst_row,
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
        combine_recv_count,
    ).launch(
        grid=(num_cu, 1, 1),
        block=(num_threads, 1, 1),
        stream=stream,
        smem=2 * num_experts * 4,
    )


def dispatch_prologue_flydsl_kernel(
    topk_idx,
    topk_weight,
    *,
    sym_buffer,
    num_tokens,
    num_topk,
    num_experts,
    num_ranks,
    rank,
    experts_per_rank,
    block_m,
    num_max_pool_tokens,
    hidden,
    num_max_tokens_per_rank,
):
    if topk_idx.dtype not in (torch.int32, torch.int64):
        raise TypeError(f"topk_idx must be int32 or int64, got {topk_idx.dtype}")
    topk_idx_flat = topk_idx.contiguous().view(-1)
    dev = topk_idx.device
    scratch = get_dispatch_prologue_scratch(num_experts, device=dev)

    num_max_blocks = num_max_pool_tokens // block_m
    expert_send_dst_rank = torch.empty(num_experts, dtype=torch.int32, device=dev)
    expert_send_dst_row = torch.empty(num_experts, dtype=torch.int32, device=dev)
    expert_send_count = torch.empty(num_experts, dtype=torch.int32, device=dev)
    expert_send_offset = torch.empty(num_experts, dtype=torch.int32, device=dev)
    tile_to_expert = torch.empty(num_max_blocks, dtype=torch.int32, device=dev)
    dispatched_token_idx = torch.empty(num_max_pool_tokens, dtype=torch.int32, device=dev)
    num_tokens_per_expert = torch.empty(experts_per_rank, dtype=torch.int64, device=dev)
    num_tokens_per_expert_prefix = torch.empty(experts_per_rank + 1, dtype=torch.int64, device=dev)
    num_tile_blocks = torch.empty(1, dtype=torch.int32, device=dev)
    combine_recv_dst_rank = torch.empty(num_experts, dtype=torch.int32, device=dev)
    combine_recv_start_row = torch.empty(num_experts, dtype=torch.int32, device=dev)
    combine_recv_count = torch.empty(num_experts, dtype=torch.int32, device=dev)
    if topk_weight is not None:
        topk_weight_flat = topk_weight.to(torch.float32).contiguous().view(-1)
    else:
        topk_weight_flat = torch.zeros(num_tokens * num_topk, dtype=torch.float32, device=dev)

    stream = torch.cuda.current_stream()
    _compiled_dispatch_prologue(
        topk_idx_flat=topk_idx_flat,
        scratch=scratch,
        sym_buffer=sym_buffer,
        expert_send_dst_rank=expert_send_dst_rank,
        expert_send_dst_row=expert_send_dst_row,
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
        combine_recv_count=combine_recv_count,
        num_tokens=num_tokens,
        num_topk=num_topk,
        num_experts=num_experts,
        num_ranks=num_ranks,
        rank=rank,
        experts_per_rank=experts_per_rank,
        block_m=block_m,
        num_max_pool_tokens=num_max_pool_tokens,
        hidden=hidden,
        num_max_tokens_per_rank=num_max_tokens_per_rank,
        stream=stream,
    )
    # Handle ABI (indices consumed by fwd combine + bwd dispatch/combine):
    #   0 expert_send_dst_rank   1 expert_send_dst_row   2 expert_send_count
    #   3 expert_send_offset     4 dispatched_token_idx  5 tile_to_expert
    #   6 num_tokens_per_expert  7 num_tokens_per_expert_prefix  8 num_tile_blocks
    #   9 combine_recv_dst_rank  10 combine_recv_start_row  11 combine_recv_count
    # (12 pool_src_slot is appended by the dispatch launcher on the forward path.)
    return (
        expert_send_dst_rank,
        expert_send_dst_row,
        expert_send_count,
        expert_send_offset,
        dispatched_token_idx,
        tile_to_expert,
        num_tokens_per_expert,
        num_tokens_per_expert_prefix,
        num_tile_blocks,
        combine_recv_dst_rank,
        combine_recv_start_row,
        combine_recv_count,
    )
