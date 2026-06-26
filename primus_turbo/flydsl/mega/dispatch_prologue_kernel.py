###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Fused MoE dispatch-prologue kernel (FlyDSL).

One persistent grid-resident kernel that, from ``topk_idx``/``topk_w``, builds the
entire EP dispatch plan over caller-owned (symmetric) buffers:

  * Phase A  -- histogram tokens -> per-expert counts (``SEND_LOCAL``)
  * Phase B  -- cross-rank all-gather of the per-expert counts (``c_buffer``)
  * Phase C  -- serial table build on block 0: pool layout (``pool_base`` /
                ``start_per_expert`` / ``source_offset``), comm tasks
                (``expert_send_dst_rank`` / ``expert_send_dst_row`` / ``expert_send_count``), ``tile_to_expert`` and
                per-pool-block ``tile_expected`` source-rank counts
  * Phase D  -- scatter each (token, topk) pair into its expert region
                (``dispatched_token_idx`` / ``dispatched_topk_slot`` / ``src_token_weight``)
                and push ``origin_rank`` / ``origin_slot`` to the destination rank

All symmetric sub-buffers (cross-rank ``c_buffer`` / ``signal`` / ``origin_rank`` /
``origin_slot`` / ``weight_recv_buf``, plus the device scalars / barrier / profile /
``scoreboard`` / ``barrier_local`` regions) are named by a single ``SymLayout`` struct
(``sym_layout.py``) passed to the kernel by value -- the kernel computes every address
from the struct's two heap bases + per-region byte offsets + per-peer delta tables.
The dispatch plan is returned as a plain tuple handle (DeepEP-style) the
dispatch/combine kernels unpack. Depends only on ``flydsl`` + ``torch``.
"""

import functools
import os as _os

import flydsl.compiler as flyc
import flydsl.expr as fx
import flydsl.expr.buffer_ops as bo
import torch
from flydsl.compiler.ast_rewriter import ASTRewriter
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

from primus_turbo.flydsl.mega import sym_layout as sl_mod
from primus_turbo.flydsl.mega.barrier import grid_sync
from primus_turbo.flydsl.mega.prims import atomic_add, ld, memory_fence, st
from primus_turbo.flydsl.mega.sym_layout import SymLayout

# Set True to emit s_memrealtime phase stamps into the SymLayout ``profile`` region.
# Compile-time constexpr: when False the profile code is not traced (no IR emitted).
ENABLE_PROFILE = False

_BLOCK_THREADS = int(_os.environ.get("PROLOGUE_BT", "256"))  # threads per block
# grid_blocks (== num_cu) is a caller arg (default 64). Fewer blocks => cheaper
# self-resetting grid_sync; 48-64 is the measured sweet spot. Must stay <= num_CU so
# the persistent grid barrier keeps all blocks resident.
_DEFAULT_GRID_BLOCKS = 64


# --------------------------------------------------------------------------- #
# Address-space / scope constants (atomic + fence prims come from prims.py)
# --------------------------------------------------------------------------- #
_llvm = bo.llvm
_SCOPE = "agent"  # device-wide scope (Triton scope="gpu" lowers to this)
_I4 = 4  # int32 byte stride / atomic alignment
_GLOBAL = 1  # LLVM global address space
_LDS = 3  # LLVM LDS (workgroup) address space


# --------------------------------------------------------------------------- #
# LDS (workgroup) int32 scratch -- block-private histogram to slash the global
# atomic contention in Phase A / Phase D (32 counters hit by 32768 atomics).
# --------------------------------------------------------------------------- #
_LDS_SCOPE = "workgroup"


def lds_base_addr():
    """Integer addrspace-3 base of the dynamic shared region (for prims ld/st/atomic_add)."""
    return _unwrap_value(_fly_ptrtoint(get_dyn_shared()))


def _read_realtime():
    """Read the GPU constant-rate realtime counter (s_memrealtime)."""
    op = _llvm.inline_asm(
        fx.T.i64(), [], "s_memrealtime $0\n\ts_waitcnt lgkmcnt(0)", "=s", has_side_effects=True
    )
    return fx.arith.ArithValue(op, signed=False)


# Prologue outputs are returned as a plain positional tuple:
#   (plan, tile_to_expert, tile_expected, origin_rank, origin_slot,
#    num_pool_blocks, max_num_token)
# plan = (dst_rank, dst_offset, count, src_offset, src_tokens, topk_slot, weight)


# --------------------------------------------------------------------------- #
# Cross-rank barrier (only block 0 of each rank handshakes); call at kernel top level.
# Peer signal base addrs come from the SymLayout ``signal`` region + per-peer delta.
# --------------------------------------------------------------------------- #
_FINISHED_SUM_TAG = 1  # 1 suffices for a clean barrier


def barrier_block(sl, rank: int, world_size: int, thread_index, block_index, sync_only: bool = False):
    """Cross-rank barrier over the SymLayout ``signal`` region (main heap)."""
    if not sync_only:
        memory_fence(order="release", scope="sys")  # flush payload before cross-rank signal
    fx.gpu.barrier()  # all pushes landed before any signal
    if block_index == fx.Int32(0):
        if thread_index < fx.Int32(world_size):
            my_signal_base = sl.signal_ptr
            peer_signal_base = sl_mod.map(sl, sl.signal_ptr, thread_index)
            atomic_add(my_signal_base, thread_index, fx.Int32(_FINISHED_SUM_TAG), "sys", _GLOBAL)
            atomic_add(peer_signal_base, fx.Int32(rank), fx.Int32(-_FINISHED_SUM_TAG), "sys", _GLOBAL)
            my_signal_value = ld(my_signal_base, thread_index, scope="sys")
            while my_signal_value > fx.Int32(0):
                my_signal_value = ld(my_signal_base, thread_index, scope="sys")
    fx.gpu.barrier()  # no acquire fence (SIG is uncached)


# Lower if/while in barrier_block to scf.
barrier_block = ASTRewriter.transform(barrier_block)


def _make_dispatch_prologue(
    num_tokens,
    num_topk,
    num_experts,
    world_size,
    rank,
    experts_per_rank,
    block_m,
    pool_capacity,
    grid_blocks=_DEFAULT_GRID_BLOCKS,
    block_threads=_BLOCK_THREADS,
):
    total_pairs = num_tokens * num_topk
    grid_stride = grid_blocks * block_threads
    num_pool_blocks = pool_capacity // block_m  # pool-block capacity
    # scoreboard + barrier_local reset in-kernel (folds out two host zeroing launches):
    combine_slots = num_topk * num_tokens  # per-rank combine slots (= barrier_local len)
    c_buffer_bytes = world_size * num_experts * 4
    origin_buffer_bytes = pool_capacity * 4
    # WORKSPACE = one [5 * num_experts] i32 scratch tensor; named sub-regions by offset.
    WS_SEND, WS_WITHIN = 0, num_experts
    WS_START, WS_SROFF, WS_POOLBASE = 2 * num_experts, 3 * num_experts, 4 * num_experts

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
        DISPATCHED_TOPK_SLOT: fx.Tensor,
        SRC_TOKEN_WEIGHT: fx.Tensor,
        TOPK_WEIGHTS: fx.Tensor,
    ):
        thread_index = fx.thread_idx.x
        block_index, _, _ = fx.block_idx
        if fx.const_expr(ENABLE_PROFILE):
            profile_resource = create_buffer_resource_from_addr(sl.profile_ptr, num_records_bytes=8 * 8)
            is_profiler_thread = block_index == fx.Int32(0)  # block0 records phase boundaries
            if is_profiler_thread:
                if thread_index == fx.Int32(0):
                    buffer_store(_read_realtime(), profile_resource, fx.Int32(0))  # t0 = start

        lds_base = lds_base_addr()  # addrspace-3 base for prims ld/atomic_add (LDS)
        topk_resource = create_buffer_resource(TOPK_INDICES, max_size=True)
        # Load expert id at its native dtype (int32 or int64) -- buffer_load takes an
        # element offset and scales by element bytes, so no stride math is needed.
        # Narrow int64 to i32 for downstream (expert ids fit in i32).
        idx_load_dtype = TOPK_INDICES.element_type
        idx_is_i64 = fx.const_expr(idx_load_dtype.width == 64)

        def load_expert_id(elem_index):
            value = buffer_load(topk_resource, elem_index, vec_width=1, dtype=idx_load_dtype)
            if idx_is_i64:
                value = fx.arith.ArithValue(fx.arith.trunci(fx.T.i32(), _unwrap_value(value)), signed=True)
            return value

        # single scratch tensor; sub-regions addressed via WS_* offsets below
        workspace_resource = create_buffer_resource(WORKSPACE, max_size=True)
        workspace_base = extract_base_index(WORKSPACE, address_space=_GLOBAL)  # for prims atomic_add
        expert_send_dst_rank_resource = create_buffer_resource(EXPERT_SEND_DST_RANK, max_size=True)
        expert_send_dst_row_resource = create_buffer_resource(EXPERT_SEND_DST_ROW, max_size=True)
        expert_send_count_resource = create_buffer_resource(EXPERT_SEND_COUNT, max_size=True)
        expert_send_offset_resource = create_buffer_resource(EXPERT_SEND_OFFSET, max_size=True)
        tile_to_expert_resource = create_buffer_resource(TILE_TO_EXPERT, max_size=True)
        tile_expected_resource = create_buffer_resource(TILE_EXPECTED, max_size=True)
        dispatched_token_idx_resource = create_buffer_resource(DISPATCHED_TOKEN_IDX, max_size=True)
        dispatched_topk_slot_resource = create_buffer_resource(DISPATCHED_TOPK_SLOT, max_size=True)
        src_token_weight_resource = create_buffer_resource(SRC_TOKEN_WEIGHT, max_size=True)
        topk_weights_resource = create_buffer_resource(TOPK_WEIGHTS, max_size=True)
        meta_scalars_resource = create_buffer_resource_from_addr(sl.meta_scalars_ptr, num_records_bytes=8 * 4)

        # ---- Phase 0: init this rank's origin_rank to -1; padding rows stay -1 ----
        my_origin_rank_resource = create_buffer_resource_from_addr(
            sl.origin_rank_ptr, num_records_bytes=origin_buffer_bytes
        )
        origin_init_index = block_index * fx.Int32(block_threads) + thread_index
        while origin_init_index < fx.Int32(pool_capacity):
            buffer_store(fx.Int32(-1), my_origin_rank_resource, origin_init_index)
            origin_init_index = origin_init_index + fx.Int32(grid_stride)
        # Pre-zero TILE_EXPECTED for C4's RMW
        expected_init_index = block_index * fx.Int32(block_threads) + thread_index
        while expected_init_index < fx.Int32(num_pool_blocks):
            buffer_store(fx.Int32(0), tile_expected_resource, expected_init_index)
            expected_init_index = expected_init_index + fx.Int32(grid_stride)
        # Pre-fill TILE_TO_EXPERT with sentinel experts_per_rank (out-of-range id); C4 overwrites valid blocks
        tile_to_group_init_index = block_index * fx.Int32(block_threads) + thread_index
        while tile_to_group_init_index < fx.Int32(num_pool_blocks):
            buffer_store(fx.Int32(experts_per_rank), tile_to_expert_resource, tile_to_group_init_index)
            tile_to_group_init_index = tile_to_group_init_index + fx.Int32(grid_stride)

        if fx.const_expr(ENABLE_PROFILE):
            if is_profiler_thread:
                if thread_index == fx.Int32(0):
                    buffer_store(_read_realtime(), profile_resource, fx.Int32(1))  # after Phase 0 init
        # ---- Phase A: histogram TOPK_INDICES -> SEND_LOCAL; read low word of int64 pair ----
        # Block-private LDS histogram first, then one global atomic per (block, expert):
        # cuts global atomics from total_pairs (~32K) to grid_blocks*num_experts (~2K).
        lds_clear_index = thread_index
        while lds_clear_index < fx.Int32(num_experts):
            st(lds_base, lds_clear_index, fx.Int32(0), scope=_LDS_SCOPE, space=_LDS)
            lds_clear_index = lds_clear_index + fx.Int32(block_threads)
        fx.gpu.barrier()
        pair_index = block_index * fx.Int32(block_threads) + thread_index
        while pair_index < fx.Int32(total_pairs):
            expert_id = load_expert_id(pair_index)
            if expert_id >= fx.Int32(0):
                atomic_add(lds_base, expert_id, fx.Int32(1), _LDS_SCOPE, _LDS)
            pair_index = pair_index + fx.Int32(grid_stride)
        fx.gpu.barrier()
        lds_flush_index = thread_index
        while lds_flush_index < fx.Int32(num_experts):
            block_count = ld(lds_base, lds_flush_index, scope=_LDS_SCOPE, space=_LDS)
            if block_count > fx.Int32(0):
                atomic_add(workspace_base, fx.Int32(WS_SEND) + lds_flush_index, block_count, _SCOPE, _GLOBAL)
            lds_flush_index = lds_flush_index + fx.Int32(block_threads)
        grid_sync(sl, thread_index, block_index, grid_blocks)
        if fx.const_expr(ENABLE_PROFILE):
            if is_profiler_thread:
                if thread_index == fx.Int32(0):
                    buffer_store(_read_realtime(), profile_resource, fx.Int32(2))  # after Phase A

        # ---- Phase B: cross-rank all_gather; B1: all ranks entered ----
        barrier_block(sl, rank, world_size, thread_index, block_index, True)
        if block_index == fx.Int32(0):
            # B2: push my SEND_LOCAL row into every peer's c_buffer at row rank.
            for peer_rank in range(world_size):
                peer_c_resource = create_buffer_resource_from_addr(
                    sl_mod.map(sl, sl.c_buffer_ptr, fx.Int32(peer_rank)), num_records_bytes=c_buffer_bytes
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
        # B3: all ranks pushed + landed
        barrier_block(sl, rank, world_size, thread_index, block_index, False)
        if fx.const_expr(ENABLE_PROFILE):
            if is_profiler_thread:
                if thread_index == fx.Int32(0):
                    buffer_store(_read_realtime(), profile_resource, fx.Int32(3))  # after Phase B

        # ---- Phase C: serial table build on block0 (c_buffer read coherently from own buffer) ----
        if block_index == fx.Int32(0):
            own_c_address = sl.c_buffer_ptr
            # C1 parallel across destination (per-destination local-expert cumsum)
            if thread_index < fx.Int32(world_size):
                running_pool_offset = fx.Int32(0)
                for local_expert_index in range(experts_per_rank):
                    expert_total_count = fx.Int32(0)
                    for source_rank in range(world_size):
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
            # C2 parallel across expert (needs POOL_BASE from C1)
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
            # C3a parallel: destination/start/count per comm task
            comm_task_index = thread_index
            while comm_task_index < fx.Int32(num_experts):
                destination_rank = comm_task_index % fx.Int32(world_size)
                local_expert_index = comm_task_index // fx.Int32(world_size)
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
                # C3b serial prefix sum: exclusive cumsum of count in k-order
                source_offset = fx.Int32(0)
                comm_task_counter = 0
                for local_expert_index in range(experts_per_rank):
                    for destination_rank in range(world_size):
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
            # C4 parallel across local expert: tile_to_expert + tile_expected + total_rows (disjoint pool regions)
            if thread_index < fx.Int32(experts_per_rank):
                local_expert_index = thread_index
                expert_pool_base = buffer_load(
                    workspace_resource,
                    fx.Int32(WS_POOLBASE + rank * experts_per_rank) + local_expert_index,
                    vec_width=1,
                    dtype=fx.T.i32(),
                )
                source_counts = []
                for source_rank in fx.range_constexpr(world_size):
                    source_counts.append(
                        ld(
                            own_c_address,
                            fx.Int32(source_rank * num_experts + rank * experts_per_rank)
                            + local_expert_index,
                            scope="sys",
                        )
                    )
                expert_total_count = fx.Int32(0)
                for source_rank in fx.range_constexpr(world_size):
                    expert_total_count = expert_total_count + source_counts[source_rank]
                padded_count = ((expert_total_count + fx.Int32(block_m - 1)) // fx.Int32(block_m)) * fx.Int32(
                    block_m
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
                for source_rank in fx.range_constexpr(world_size):
                    count_value = source_counts[source_rank]
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

        grid_sync(sl, thread_index, block_index, grid_blocks)
        if fx.const_expr(ENABLE_PROFILE):
            if is_profiler_thread:
                if thread_index == fx.Int32(0):
                    buffer_store(_read_realtime(), profile_resource, fx.Int32(4))  # after Phase C

        # ---- Phase D: scatter pairs (all blocks) ----
        # Two-pass block-private reservation: count this block's pairs per expert in
        # LDS, reserve one contiguous global range per (block, expert) with a single
        # global atomic, then assign positions from an LDS cursor. Cuts the global
        # within-counter atomics from total_pairs (~32K) to grid_blocks*num_experts.
        # Row order within an expert is arbitrary (downstream reads by row), so a
        # block-grouped permutation is correct (validated set-wise by the test).
        d_clear_index = thread_index
        while d_clear_index < fx.Int32(num_experts):
            st(lds_base, d_clear_index, fx.Int32(0), scope=_LDS_SCOPE, space=_LDS)
            d_clear_index = d_clear_index + fx.Int32(block_threads)
        fx.gpu.barrier()
        count_pair_index = block_index * fx.Int32(block_threads) + thread_index
        while count_pair_index < fx.Int32(total_pairs):
            count_expert_id = load_expert_id(count_pair_index)
            if count_expert_id >= fx.Int32(0):
                atomic_add(lds_base, count_expert_id, fx.Int32(1), _LDS_SCOPE, _LDS)
            count_pair_index = count_pair_index + fx.Int32(grid_stride)
        fx.gpu.barrier()
        reserve_index = thread_index
        while reserve_index < fx.Int32(num_experts):
            block_expert_count = ld(lds_base, reserve_index, scope=_LDS_SCOPE, space=_LDS)
            if block_expert_count > fx.Int32(0):
                reserved_base = atomic_add(
                    workspace_base,
                    fx.Int32(WS_WITHIN) + reserve_index,
                    block_expert_count,
                    _SCOPE,
                    _GLOBAL,
                )
                st(
                    lds_base,
                    fx.Int32(num_experts) + reserve_index,
                    reserved_base,
                    scope=_LDS_SCOPE,
                    space=_LDS,
                )  # global base
                st(
                    lds_base, reserve_index, fx.Int32(0), scope=_LDS_SCOPE, space=_LDS
                )  # reset to per-block cursor
            reserve_index = reserve_index + fx.Int32(block_threads)
        fx.gpu.barrier()
        pair_index = block_index * fx.Int32(block_threads) + thread_index
        while pair_index < fx.Int32(total_pairs):
            expert_id = load_expert_id(pair_index)
            if expert_id >= fx.Int32(0):
                token_index = pair_index // fx.Int32(num_topk)
                topk_slot = pair_index % fx.Int32(num_topk)
                local_position = atomic_add(lds_base, expert_id, fx.Int32(1), _LDS_SCOPE, _LDS)
                within_expert_position = (
                    ld(lds_base, fx.Int32(num_experts) + expert_id, scope=_LDS_SCOPE, space=_LDS)
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
                buffer_store(
                    topk_slot, dispatched_topk_slot_resource, expert_source_offset + within_expert_position
                )  # topk slot per pair
                routing_weight = buffer_load(topk_weights_resource, pair_index, vec_width=1, dtype=fx.T.f32())
                buffer_store(
                    routing_weight, src_token_weight_resource, expert_source_offset + within_expert_position
                )  # routing weight per pair
                destination_rank = expert_id // fx.Int32(experts_per_rank)
                peer_origin_rank_resource = create_buffer_resource_from_addr(
                    sl_mod.map(sl, sl.origin_rank_ptr, destination_rank),
                    num_records_bytes=origin_buffer_bytes,
                )
                peer_origin_slot_resource = create_buffer_resource_from_addr(
                    sl_mod.map(sl, sl.origin_slot_ptr, destination_rank),
                    num_records_bytes=origin_buffer_bytes,
                )
                buffer_store(fx.Int32(rank), peer_origin_rank_resource, destination_row)
                # origin_slot = token-major position t*K+k, so the origin rank's combine
                # buffer is a dense [T, K, H] view -> the fused 3-role topk reduce reads
                # comb[token*topk+k] directly (must match the reduce role's [T,K,H] layout).
                buffer_store(
                    token_index * fx.Int32(num_topk) + topk_slot, peer_origin_slot_resource, destination_row
                )
                # ride the routing weight cross-rank to the dest weight_recv_buf[dest_row]
                # (same scatter as origin) -> backward gets per-pool-row weight without all_gather.
                peer_weight_resource = create_buffer_resource_from_addr(
                    sl_mod.map(sl, sl.weight_recv_buf_ptr, destination_rank),
                    num_records_bytes=origin_buffer_bytes,
                )
                buffer_store(routing_weight, peer_weight_resource, destination_row)
            pair_index = pair_index + fx.Int32(grid_stride)

        # ---- Reset the cross-rank signal buffers in-kernel (replaces two host zero launches).
        # scoreboard -> 0 (dispatch handshake); barrier_local -> -1 (combine flags, raised
        # >=0 by role 1). The grid_sync below + Phase E publish these locally + cross-rank.
        scoreboard_resource = create_buffer_resource_from_addr(
            sl.scoreboard_ptr, num_records_bytes=num_pool_blocks * 4
        )
        barrier_local_resource = create_buffer_resource_from_addr(
            sl.barrier_local_ptr, num_records_bytes=combine_slots * 4
        )
        signal_block_index = block_index * fx.Int32(block_threads) + thread_index
        while signal_block_index < fx.Int32(num_pool_blocks):
            buffer_store(fx.Int32(0), scoreboard_resource, signal_block_index)
            signal_block_index = signal_block_index + fx.Int32(grid_stride)
        flag_slot_index = block_index * fx.Int32(block_threads) + thread_index
        while flag_slot_index < fx.Int32(combine_slots):
            buffer_store(fx.Int32(-1), barrier_local_resource, flag_slot_index)
            flag_slot_index = flag_slot_index + fx.Int32(grid_stride)

        grid_sync(sl, thread_index, block_index, grid_blocks)
        if fx.const_expr(ENABLE_PROFILE):
            if is_profiler_thread:
                if thread_index == fx.Int32(0):
                    buffer_store(_read_realtime(), profile_resource, fx.Int32(5))  # after Phase D

        # ---- Post: reset SEND_LOCAL/WITHIN_EXPERT counters for the next launch ----
        reset_index = block_index * fx.Int32(block_threads) + thread_index
        while reset_index < fx.Int32(num_experts):
            buffer_store(fx.Int32(0), workspace_resource, fx.Int32(WS_SEND) + reset_index)
            buffer_store(fx.Int32(0), workspace_resource, fx.Int32(WS_WITHIN) + reset_index)
            reset_index = reset_index + fx.Int32(grid_stride)
        if fx.const_expr(ENABLE_PROFILE):
            if is_profiler_thread:
                if thread_index == fx.Int32(0):
                    buffer_store(_read_realtime(), profile_resource, fx.Int32(6))  # after Post reset

        # ---- Phase E: origins landed cross-rank ----
        barrier_block(sl, rank, world_size, thread_index, block_index, False)
        if fx.const_expr(ENABLE_PROFILE):
            if is_profiler_thread:
                if thread_index == fx.Int32(0):
                    buffer_store(_read_realtime(), profile_resource, fx.Int32(7))  # after Phase E (end)

    @flyc.jit
    def launch(
        topk_indices,
        workspace,
        sym_layout,
        expert_send_dst_rank,
        expert_send_dst_row,
        expert_send_count,
        expert_send_offset,
        tile_to_expert,
        tile_expected,
        dispatched_token_idx,
        dispatched_topk_slot,
        src_token_weight,
        topk_weights,
        stream: fx.Stream = fx.Stream(None),
    ):
        dispatch_prologue_kernel(
            topk_indices,
            workspace,
            sym_layout,
            expert_send_dst_rank,
            expert_send_dst_row,
            expert_send_count,
            expert_send_offset,
            tile_to_expert,
            tile_expected,
            dispatched_token_idx,
            dispatched_topk_slot,
            src_token_weight,
            topk_weights,
        ).launch(
            grid=(grid_blocks, 1, 1),
            block=(block_threads, 1, 1),
            stream=stream,
            smem=2 * num_experts * 4,  # LDS int32: Phase A hist; Phase D cursor+base
        )

    return launch


@functools.lru_cache(maxsize=8)
def _compile(
    num_tokens,
    num_topk,
    num_experts,
    world_size,
    rank,
    experts_per_rank,
    block_m,
    pool_capacity,
    grid_blocks=_DEFAULT_GRID_BLOCKS,
    idx_dtype=None,  # cache-key only: kernel specializes on topk_idx's element_type
):
    return _make_dispatch_prologue(
        num_tokens,
        num_topk,
        num_experts,
        world_size,
        rank,
        experts_per_rank,
        block_m,
        pool_capacity,
        grid_blocks=grid_blocks,
    )


# Module-level fast-launch cache (function/stream-keyed CallState) used when the
# caller does not supply its own launch_cache.
_DEFAULT_LAUNCH_CACHE: dict = {}


@functools.lru_cache(maxsize=8)
def _dispatch_prologue_workspace_cached(num_experts, device):
    # 5 per-expert i32 scratch tables packed in one tensor (see the kernel's WS_*
    # offsets): send_local / within_expert_counter / start_per_expert /
    # source_offset_per_expert / pool_base. The kernel self-resets it each launch.
    return torch.zeros(5 * num_experts, dtype=torch.int32, device=device)


def get_dispatch_prologue_workspace(num_experts, device="cuda"):
    """Cached internal scratch for the prologue kernel (reused across launches; the
    kernel self-resets it each launch, so callers never own or pass it)."""
    dev = torch.device(device)
    if dev.type == "cuda" and dev.index is None:  # pin to a concrete index so the
        dev = torch.device("cuda", torch.cuda.current_device())  # cache key is stable
    return _dispatch_prologue_workspace_cached(int(num_experts), dev)


def dispatch_prologue(
    topk_idx,
    topk_w,
    *,
    sym_layout,
    num_tokens,
    num_topk,
    num_experts,
    world_size,
    rank,
    experts_per_rank,
    block_m,
    pool_capacity,
    launch_cache=None,
    no_cpu_sync=True,
    num_cu=_DEFAULT_GRID_BLOCKS,
):
    """One fused-prologue kernel launch.

    ``sym_layout`` is a single :class:`SymLayout` struct naming every symmetric
    sub-buffer (the two heaps' bases + per-region byte offsets + per-peer delta
    tables); the kernel computes all cross-rank addresses from it. ``num_cu`` is the
    persistent grid block count (must stay <= device CU count).

    The dispatch-plan output tables (expert_send_dst_rank / expert_send_dst_row /
    expert_send_count / expert_send_offset / tile_to_expert / tile_expected /
    dispatched_token_idx / dispatched_topk_slot / src_token_weight) are allocated
    internally and returned -- callers never own them. ``origin_rank`` / ``origin_slot``
    live in ``sym_layout`` (written cross-rank); the return keeps their tuple slots as
    ``None`` for positional compatibility."""
    # Accept int32 or int64; the kernel reads each entry at its native dtype
    # (buffer_load infers element size from the tensor's element_type).
    if topk_idx.dtype not in (torch.int32, torch.int64):
        raise TypeError(f"topk_idx must be int32 or int64, got {topk_idx.dtype}")
    topk_idx_flat = topk_idx.contiguous().view(-1)
    dev = topk_idx.device
    # internal scratch (cached, kernel self-resets) -- not a caller-owned buffer
    workspace = get_dispatch_prologue_workspace(num_experts, device=dev)

    # ---- allocate the plan output tables internally (returned to the caller) ----
    n_mblk = pool_capacity // block_m
    i32 = lambda n: torch.empty(n, dtype=torch.int32, device=dev)
    expert_send_dst_rank, expert_send_dst_row, expert_send_count, expert_send_offset = (
        i32(num_experts) for _ in range(4)
    )
    tile_to_expert, tile_expected = i32(n_mblk), i32(n_mblk)
    dispatched_token_idx, dispatched_topk_slot = i32(pool_capacity), i32(pool_capacity)
    src_token_weight = torch.empty(pool_capacity, dtype=torch.float32, device=dev)
    if topk_w is not None:
        topk_weights_flat = topk_w.to(torch.float32).contiguous().view(-1)
    else:  # no weights given -> internal zeros (topk_w is None branch)
        topk_weights_flat = torch.zeros(num_tokens * num_topk, dtype=torch.float32, device=dev)

    # Default to the module-level launch cache so callers that don't pass one (e.g.
    # the test / custom-op path) still hit the pre-packed CallState fast launch.
    if launch_cache is None:
        launch_cache = _DEFAULT_LAUNCH_CACHE

    stream = torch.cuda.current_stream()
    launch_function = _compile(
        num_tokens,
        num_topk,
        num_experts,
        world_size,
        rank,
        experts_per_rank,
        block_m,
        pool_capacity,
        grid_blocks=int(num_cu),
        idx_dtype=topk_idx.dtype,
    )
    kernel_arguments = (
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
        dispatched_topk_slot,
        src_token_weight,
        topk_weights_flat,
        stream,
    )
    # Fast launch: call pre-packed CallState directly, fall back to normal call
    stream_id = stream.cuda_stream  # raw handle (fresh wrapper each call)
    call_state = None
    if (
        launch_cache is not None
        and launch_cache.get("function") is launch_function
        and launch_cache.get("stream") == stream_id
    ):
        call_state = launch_cache.get("call_state")
    if call_state is not None:
        call_state(kernel_arguments)
    else:
        launch_function(*kernel_arguments[:-1], stream=stream)
        if launch_cache is not None:
            launch_cache["function"], launch_cache["stream"], launch_cache["call_state"] = (
                launch_function,
                stream_id,
                None,
            )
            try:
                call_state_cache = launch_function._call_state_cache
                if len(call_state_cache) == 1:
                    launch_cache["call_state"] = next(iter(call_state_cache.values()))
            except Exception:
                pass
    # DeepEP-style dispatch handle: a flat tuple the dispatch/combine kernels unpack.
    # num_tasks (== num_experts) is derived from dst_rank.numel(); routing tensors last.
    #   (dst_rank, dst_offset, count, src_offset, src_tokens, topk_slot, weight)
    plan = (
        expert_send_dst_rank,
        expert_send_dst_row,
        expert_send_count,
        expert_send_offset,
        dispatched_token_idx,
        dispatched_topk_slot,
        src_token_weight,
    )
    # origin_rank/origin_slot now live in sym_layout; max_num_token from device scalar
    # would need a host readback of sym_layout.meta_scalars, so default to pool_capacity.
    max_num_token = pool_capacity
    return (
        plan,
        tile_to_expert,
        tile_expected,
        None,
        None,
        pool_capacity // block_m,
        max_num_token,
    )
