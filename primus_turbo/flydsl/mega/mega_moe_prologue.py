###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Fused MoE dispatch-prologue kernel (FlyDSL).

One persistent grid-resident kernel that, from ``topk_idx``/``topk_w``, builds the
entire EP dispatch plan over caller-owned (symmetric) buffers:

  * Phase A  -- histogram tokens -> per-expert counts (``SEND_LOCAL``)
  * Phase B  -- cross-rank all-gather of the per-expert counts (``C`` buffer)
  * Phase C  -- serial table build on block 0: pool layout (``pool_base`` /
                ``start_per_expert`` / ``source_offset``), comm tasks
                (``destination`` / ``start`` / ``count``), ``tile_to_group`` and
                per-pool-block ``expected`` source-rank counts
  * Phase D  -- scatter each (token, topk) pair into its expert region
                (``source_tokens`` / ``source_topk_slot`` / ``source_weight``)
                and push ``origin_rank`` / ``origin_slot`` to the destination rank

Self-contained: the comm-handshake prims (grid + cross-rank barriers) and the
``CommTasks`` host dataclass are vendored here, so this module depends only on
``flydsl`` + ``torch`` (no upstream ``kernels`` package).
"""

from __future__ import annotations

import functools
import os as _os
from dataclasses import dataclass
from typing import NamedTuple, Optional

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
    create_llvm_ptr,
    extract_base_index,
    get_element_ptr,
)

_BLOCK_THREADS = int(_os.environ.get("PROLOGUE_BT", "256"))  # threads per block
# Fewer blocks => cheaper sense-reversing grid_barrier (4x per launch); 48-64 is the
# measured sweet spot (T=8192: 150us vs 233us at 256). Must stay <= num_CU so the
# persistent grid_barrier keeps all blocks resident.
_GRID_BLOCKS = int(_os.environ.get("PROLOGUE_G", "64"))


# --------------------------------------------------------------------------- #
# Vendored cross-rank / atomic prims (was kernels.prims)
# --------------------------------------------------------------------------- #
_llvm = bo.llvm
_ORD = _llvm.AtomicOrdering
_SCOPE = "agent"  # device-wide scope (Triton scope="gpu" lowers to this)
_I4 = 4  # int32 byte stride / atomic alignment


def _scope(scope):
    """Map scope name to LLVM syncscope ('sys' -> None = system default)."""
    if scope == "sys":
        return None
    return scope  # 'agent' (or any explicit syncscope string)


def _elem_ptr_i32(tensor, idx):
    """LLVM ptr to int32 element ``tensor[idx]`` (idx an fx/i32 value or int)."""
    base = create_llvm_ptr(extract_base_index(tensor, address_space=1), 1)
    byte_off = _unwrap_value(idx * fx.Int32(_I4))
    return get_element_ptr(base, byte_offset=byte_off, elem_type=fx.T.i8())


def _elem_ptr_i32_from_addr(addr_i64, idx):
    """LLVM ptr to int32 element at ``(addr_i64)[idx]`` (runtime peer base addr)."""
    base = create_llvm_ptr(_unwrap_value(addr_i64), 1)
    byte_off = _unwrap_value(idx * fx.Int32(_I4))
    return get_element_ptr(base, byte_offset=byte_off, elem_type=fx.T.i8())


def _mem_fence():
    """deep_ep cheap fence: s_waitcnt drain + compiler barrier, so a following
    relaxed atomic gets release/acquire ordering."""
    _llvm.inline_asm(fx.T.i32(), [], "s_waitcnt lgkmcnt(0) vmcnt(0)", "=r,~{memory}", has_side_effects=True)


# release/acquire below = relaxed atomic + `_mem_fence()` cheap-fence drain


def st_release(tensor, idx, val, *, scope=_SCOPE):
    """Release store of int32 `val` into `tensor[idx]`: fence drain + relaxed store."""
    ptr = _elem_ptr_i32(tensor, idx)
    _mem_fence()
    _llvm.StoreOp(_unwrap_value(val), ptr, ordering=_ORD.monotonic, syncscope=_scope(scope), alignment=_I4)


def ld_acquire(tensor, idx, *, scope=_SCOPE):
    """Acquire load of int32 `tensor[idx]`: fence drain + relaxed load."""
    ptr = _elem_ptr_i32(tensor, idx)
    _mem_fence()
    op = _llvm.LoadOp(fx.T.i32(), ptr, ordering=_ORD.monotonic, syncscope=_scope(scope), alignment=_I4)
    return fx.arith.ArithValue(op.result, signed=True)


def fence_acquire(*, scope=_SCOPE):
    """REAL acquire fence: invalidates L1 so a reused-buffer consumer reads fresh.
    Load-bearing for grid_barrier table handoff (cheap fence leaves L1 stale)."""
    _llvm.fence(_ORD.acquire, syncscope=_scope(scope))


def fence_release(*, scope=_SCOPE):
    """REAL release fence: L2 writeback so other CUs see prior stores; pair w/ fence_acquire."""
    _llvm.fence(_ORD.release, syncscope=_scope(scope))


def atomic_add(tensor, idx, val, *, release=False, scope=_SCOPE):
    """Atomic int32 add into `tensor[idx]`, returns OLD value; release=True drains first."""
    ptr = _elem_ptr_i32(tensor, idx)
    if release:
        _mem_fence()
    res = _llvm.atomicrmw(
        _llvm.AtomicBinOp.add, ptr, _unwrap_value(val), _ORD.monotonic, syncscope=_scope(scope), alignment=_I4
    )
    return fx.arith.ArithValue(res, signed=True)


def atomic_add_addr(addr_i64, idx, val, *, release=False, scope=_SCOPE):
    """Atomic int32 add to `(addr_i64)[idx]` (runtime peer addr), returns old; release drains first."""
    ptr = _elem_ptr_i32_from_addr(addr_i64, idx)
    if release:
        _mem_fence()
    res = _llvm.atomicrmw(
        _llvm.AtomicBinOp.add, ptr, _unwrap_value(val), _ORD.monotonic, syncscope=_scope(scope), alignment=_I4
    )
    return fx.arith.ArithValue(res, signed=True)


def atomic_cas(tensor, idx, *, expected, desired, scope=_SCOPE):
    """int32 compare-exchange tensor[idx]; returns the OLD value (losers read the winner's).
    First-writer-wins: exactly one racer sees old==expected (the primary)."""
    ptr = _elem_ptr_i32(tensor, idx)
    pair = _llvm.cmpxchg(
        ptr,
        _unwrap_value(fx.Int32(expected)),
        _unwrap_value(desired),
        _ORD.monotonic,
        _ORD.monotonic,
        syncscope=_scope(scope),
        alignment=_I4,
    )
    old = _llvm.extractvalue(fx.T.i32(), pair, [0])  # field 0 = old value
    return fx.arith.ArithValue(old, signed=True)


def _read_realtime():  # [PROF]
    """Read the GPU constant-rate realtime counter (s_memrealtime)."""  # [PROF]
    op = _llvm.inline_asm(
        fx.T.i64(), [], "s_memrealtime $0\n\ts_waitcnt lgkmcnt(0)", "=s", has_side_effects=True
    )  # [PROF]
    return fx.arith.ArithValue(op, signed=False)  # [PROF]


# --------------------------------------------------------------------------- #
# Vendored host dataclass (was kernels.dispatch_flydsl.CommTasks)
# --------------------------------------------------------------------------- #
@dataclass
class CommTasks:
    """Flat per-comm-task push metadata for the fused W>1 overlap Kernel 1."""

    dest: torch.Tensor  # [num_comm] int32
    start: torch.Tensor  # dst pool row start on peer
    cnt: torch.Tensor  # [num_comm] int32
    srcoff: torch.Tensor  # offset into src_tokens
    src_tokens: torch.Tensor  # this rank's token idx, row order
    num_comm: int
    src_k: torch.Tensor = None  # topk-slot k per pair
    src_w: torch.Tensor = None  # routing weight topk_w per pair, source order


class PrologueResult(NamedTuple):
    """Named view of the prologue outputs (a tuple, so old positional unpack still works)."""

    comm_tasks: CommTasks  # flat per-comm-task push metadata
    tile_to_group: torch.Tensor  # [n_mblk] local-expert id per pool block
    expected: torch.Tensor  # [n_mblk] source-rank count per pool block
    origin_rank: torch.Tensor  # [pool_capacity] source rank per pool row (-1 = pad)
    origin_slot: torch.Tensor  # [pool_capacity] source (k*num_tokens+token) per pool row
    num_pool_blocks: int  # pool_capacity // block_m
    max_num_token: int  # pool_capacity (no_cpu_sync) or on-device total_rows
    dedup_src_row: (
        torch.Tensor
    )  # [pool_capacity] -1 = primary (XGMI), >=0 = dest-local copy from that pool row
    source_dedup: torch.Tensor  # [pool_capacity] source-indexed: 1 = secondary (skip push), 0 = primary


# --------------------------------------------------------------------------- #
# Grid-wide + cross-rank barriers; each helper AST-transformed before kernel use
# --------------------------------------------------------------------------- #
_FINISHED_SUM_TAG = 1  # 1 suffices for a clean barrier


def load_relaxed_at_address(base_address, index, *, scope="sys"):
    """Relaxed int32 load at (base_address)[index] (peer-addr analog of ld_relaxed)."""
    element_pointer = _elem_ptr_i32_from_addr(base_address, index)
    load_op = _llvm.LoadOp(
        fx.T.i32(), element_pointer, ordering=_ORD.monotonic, syncscope=_scope(scope), alignment=4
    )
    return fx.arith.ArithValue(load_op.result, signed=True)


def grid_barrier(grid_barrier_state, num_blocks: int, thread_index):
    """Device-wide self-resetting (sense-reversing) barrier over num_blocks blocks."""
    fx.gpu.barrier()  # align threads in this block
    fence_release(scope="agent")  # publish writes to other CUs
    if thread_index == fx.Int32(0):
        sense_before_arrival = ld_acquire(
            grid_barrier_state, fx.Int32(1), scope="agent"
        )  # sense before arriving
        previous_arrival_count = atomic_add(
            grid_barrier_state, fx.Int32(0), fx.Int32(1), release=True, scope="agent"
        )
        if previous_arrival_count == fx.Int32(num_blocks - 1):
            st_release(grid_barrier_state, fx.Int32(0), fx.Int32(0), scope="agent")  # last in: reset counter
            st_release(
                grid_barrier_state, fx.Int32(1), fx.Int32(1) - sense_before_arrival, scope="agent"
            )  # flip sense
        else:
            observed_sense = ld_acquire(grid_barrier_state, fx.Int32(1), scope="agent")
            while observed_sense == sense_before_arrival:
                observed_sense = ld_acquire(grid_barrier_state, fx.Int32(1), scope="agent")
    fx.gpu.barrier()  # broadcast "all arrived"
    fence_acquire(scope="agent")  # invalidate L1 for fresh reads


def barrier_block(
    signal_ptrs, rank: int, world_size: int, thread_index, block_index, sync_only: bool = False
):
    """Cross-rank barrier (only block 0 of each rank handshakes); call at kernel top level."""
    if not sync_only:
        fence_release(scope="sys")  # flush payload before cross-rank signal
    fx.gpu.barrier()  # all pushes landed before any signal
    signal_resource = create_buffer_resource(signal_ptrs, max_size=True)
    if block_index == fx.Int32(0):
        if thread_index < fx.Int32(world_size):
            my_signal_base = buffer_load(signal_resource, fx.Int32(rank), vec_width=1, dtype=fx.T.i64())
            peer_signal_base = buffer_load(signal_resource, thread_index, vec_width=1, dtype=fx.T.i64())
            atomic_add_addr(
                my_signal_base, thread_index, fx.Int32(_FINISHED_SUM_TAG), release=False, scope="sys"
            )
            atomic_add_addr(
                peer_signal_base, fx.Int32(rank), fx.Int32(-_FINISHED_SUM_TAG), release=False, scope="sys"
            )
            my_signal_value = load_relaxed_at_address(my_signal_base, thread_index)
            while my_signal_value > fx.Int32(0):
                my_signal_value = load_relaxed_at_address(my_signal_base, thread_index)
    fx.gpu.barrier()  # no acquire fence (SIG is uncached)


# Lower if/while in these helpers to scf.
grid_barrier = ASTRewriter.transform(grid_barrier)
barrier_block = ASTRewriter.transform(barrier_block)


def _make_prologue(
    num_tokens,
    num_topk,
    num_experts,
    world_size,
    rank,
    experts_per_rank,
    block_m,
    pool_capacity,
    grid_blocks=_GRID_BLOCKS,
    block_threads=_BLOCK_THREADS,
    dedup=False,
):
    total_pairs = num_tokens * num_topk
    grid_stride = grid_blocks * block_threads
    num_pool_blocks = pool_capacity // block_m  # pool-block capacity
    c_buffer_bytes = world_size * num_experts * 4
    origin_buffer_bytes = pool_capacity * 4
    token_rank_table_size = num_tokens * world_size  # dedup table: primary pool row per (token, dest-rank)

    @flyc.kernel(known_block_size=[block_threads, 1, 1])
    def prologue_k(
        TOPK_INDICES: fx.Tensor,
        SEND_LOCAL: fx.Tensor,
        WITHIN_EXPERT_COUNTER: fx.Tensor,
        C_BUFFER_PTRS: fx.Tensor,
        SIGNAL_PTRS: fx.Tensor,
        ORIGIN_RANK_PTRS: fx.Tensor,
        ORIGIN_SLOT_PTRS: fx.Tensor,
        START_PER_EXPERT: fx.Tensor,
        SOURCE_OFFSET_PER_EXPERT: fx.Tensor,
        POOL_BASE: fx.Tensor,
        DESTINATION: fx.Tensor,
        START: fx.Tensor,
        COUNT: fx.Tensor,
        SOURCE_OFFSET_OUT: fx.Tensor,
        TILE_TO_GROUP: fx.Tensor,
        EXPECTED: fx.Tensor,
        SOURCE_TOKENS: fx.Tensor,
        SOURCE_TOPK_SLOT: fx.Tensor,
        SOURCE_WEIGHT: fx.Tensor,
        TOPK_WEIGHTS: fx.Tensor,
        META_SCALARS: fx.Tensor,
        GRID_BARRIER_STATE: fx.Tensor,
        TOKEN_RANK_TABLE: fx.Tensor,
        DEDUP_SRC_ROW_PTRS: fx.Tensor,
        SOURCE_DEDUP: fx.Tensor,
        PROFILE: fx.Tensor,
    ):  # [PROF]
        thread_index = fx.thread_idx.x
        block_index, _, _ = fx.block_idx
        profile_resource = create_buffer_resource(PROFILE, max_size=True)  # [PROF]
        is_profiler_thread = block_index == fx.Int32(0)  # [PROF] block0 records phase boundaries
        if is_profiler_thread:  # [PROF]
            if thread_index == fx.Int32(0):  # [PROF]
                buffer_store(_read_realtime(), profile_resource, fx.Int32(0))  # [PROF] t0 = start

        topk_resource = create_buffer_resource(TOPK_INDICES, max_size=True)
        send_local_resource = create_buffer_resource(SEND_LOCAL, max_size=True)
        within_expert_resource = create_buffer_resource(WITHIN_EXPERT_COUNTER, max_size=True)
        c_buffer_ptrs_resource = create_buffer_resource(C_BUFFER_PTRS, max_size=True)
        origin_rank_ptrs_resource = create_buffer_resource(ORIGIN_RANK_PTRS, max_size=True)
        origin_slot_ptrs_resource = create_buffer_resource(ORIGIN_SLOT_PTRS, max_size=True)
        start_per_expert_resource = create_buffer_resource(START_PER_EXPERT, max_size=True)
        source_offset_per_expert_resource = create_buffer_resource(SOURCE_OFFSET_PER_EXPERT, max_size=True)
        pool_base_resource = create_buffer_resource(POOL_BASE, max_size=True)
        destination_resource = create_buffer_resource(DESTINATION, max_size=True)
        start_resource = create_buffer_resource(START, max_size=True)
        count_resource = create_buffer_resource(COUNT, max_size=True)
        source_offset_out_resource = create_buffer_resource(SOURCE_OFFSET_OUT, max_size=True)
        tile_to_group_resource = create_buffer_resource(TILE_TO_GROUP, max_size=True)
        expected_resource = create_buffer_resource(EXPECTED, max_size=True)
        source_tokens_resource = create_buffer_resource(SOURCE_TOKENS, max_size=True)
        source_topk_slot_resource = create_buffer_resource(SOURCE_TOPK_SLOT, max_size=True)
        source_weight_resource = create_buffer_resource(SOURCE_WEIGHT, max_size=True)
        topk_weights_resource = create_buffer_resource(TOPK_WEIGHTS, max_size=True)
        meta_scalars_resource = create_buffer_resource(META_SCALARS, max_size=True)
        token_rank_table_resource = create_buffer_resource(TOKEN_RANK_TABLE, max_size=True)
        dedup_src_row_ptrs_resource = create_buffer_resource(DEDUP_SRC_ROW_PTRS, max_size=True)
        source_dedup_resource = create_buffer_resource(SOURCE_DEDUP, max_size=True)

        # ---- Phase 0: init this rank's origin_rank to -1; padding rows stay -1 ----
        my_origin_rank_address = buffer_load(
            origin_rank_ptrs_resource, fx.Int32(rank), vec_width=1, dtype=fx.T.i64()
        )
        my_origin_rank_resource = create_buffer_resource_from_addr(
            my_origin_rank_address, num_records_bytes=origin_buffer_bytes
        )
        origin_init_index = block_index * fx.Int32(block_threads) + thread_index
        while origin_init_index < fx.Int32(pool_capacity):
            buffer_store(fx.Int32(-1), my_origin_rank_resource, origin_init_index)
            origin_init_index = origin_init_index + fx.Int32(grid_stride)
        # Init dedup tables only when dedup is enabled (skipped by default).
        if fx.const_expr(dedup):
            # Init this rank's dedup_src_row to -1 (primary default); secondaries overwrite in Phase D.
            my_dedup_address = buffer_load(
                dedup_src_row_ptrs_resource, fx.Int32(rank), vec_width=1, dtype=fx.T.i64()
            )
            my_dedup_resource = create_buffer_resource_from_addr(
                my_dedup_address, num_records_bytes=origin_buffer_bytes
            )
            dedup_init_index = block_index * fx.Int32(block_threads) + thread_index
            while dedup_init_index < fx.Int32(pool_capacity):
                buffer_store(fx.Int32(-1), my_dedup_resource, dedup_init_index)
                dedup_init_index = dedup_init_index + fx.Int32(grid_stride)
            # Init dedup election table to -1 (no primary recorded yet for any (token, dest-rank)).
            token_rank_init_index = block_index * fx.Int32(block_threads) + thread_index
            while token_rank_init_index < fx.Int32(token_rank_table_size):
                buffer_store(fx.Int32(-1), token_rank_table_resource, token_rank_init_index)
                token_rank_init_index = token_rank_init_index + fx.Int32(grid_stride)
        # Pre-zero EXPECTED for C4's RMW
        expected_init_index = block_index * fx.Int32(block_threads) + thread_index
        while expected_init_index < fx.Int32(num_pool_blocks):
            buffer_store(fx.Int32(0), expected_resource, expected_init_index)
            expected_init_index = expected_init_index + fx.Int32(grid_stride)
        # Pre-fill TILE_TO_GROUP with sentinel experts_per_rank (out-of-range id); C4 overwrites valid blocks
        tile_to_group_init_index = block_index * fx.Int32(block_threads) + thread_index
        while tile_to_group_init_index < fx.Int32(num_pool_blocks):
            buffer_store(fx.Int32(experts_per_rank), tile_to_group_resource, tile_to_group_init_index)
            tile_to_group_init_index = tile_to_group_init_index + fx.Int32(grid_stride)

        if is_profiler_thread:  # [PROF]
            if thread_index == fx.Int32(0):  # [PROF]
                buffer_store(_read_realtime(), profile_resource, fx.Int32(1))  # [PROF] after Phase 0 init
        # ---- Phase A: histogram TOPK_INDICES -> SEND_LOCAL; read low word of int64 pair ----
        pair_index = block_index * fx.Int32(block_threads) + thread_index
        while pair_index < fx.Int32(total_pairs):
            expert_id = buffer_load(topk_resource, pair_index + pair_index, vec_width=1, dtype=fx.T.i32())
            if expert_id >= fx.Int32(0):
                atomic_add(SEND_LOCAL, expert_id, fx.Int32(1), release=False, scope="agent")
            pair_index = pair_index + fx.Int32(grid_stride)
        grid_barrier(GRID_BARRIER_STATE, grid_blocks, thread_index)
        if is_profiler_thread:  # [PROF]
            if thread_index == fx.Int32(0):  # [PROF]
                buffer_store(_read_realtime(), profile_resource, fx.Int32(2))  # [PROF] after Phase A

        # ---- Phase B: cross-rank all_gather; B1: all ranks entered ----
        barrier_block(SIGNAL_PTRS, rank, world_size, thread_index, block_index, True)
        if block_index == fx.Int32(0):
            # B2: push my SEND_LOCAL row into every peer's C buffer at row rank.
            for peer_rank in range(world_size):
                peer_c_address = buffer_load(
                    c_buffer_ptrs_resource, fx.Int32(peer_rank), vec_width=1, dtype=fx.T.i64()
                )
                peer_c_resource = create_buffer_resource_from_addr(
                    peer_c_address, num_records_bytes=c_buffer_bytes
                )
                push_expert_index = thread_index
                while push_expert_index < fx.Int32(num_experts):
                    send_count_value = buffer_load(
                        send_local_resource, push_expert_index, vec_width=1, dtype=fx.T.i32()
                    )
                    buffer_store(
                        send_count_value, peer_c_resource, fx.Int32(rank * num_experts) + push_expert_index
                    )
                    push_expert_index = push_expert_index + fx.Int32(block_threads)
        # B3: all ranks pushed + landed
        barrier_block(SIGNAL_PTRS, rank, world_size, thread_index, block_index, False)
        if is_profiler_thread:  # [PROF]
            if thread_index == fx.Int32(0):  # [PROF]
                buffer_store(_read_realtime(), profile_resource, fx.Int32(3))  # [PROF] after Phase B

        # ---- Phase C: serial table build on block0 (C read coherently from own buffer) ----
        if block_index == fx.Int32(0):
            own_c_address = buffer_load(c_buffer_ptrs_resource, fx.Int32(rank), vec_width=1, dtype=fx.T.i64())
            # C1 parallel across destination (per-destination local-expert cumsum)
            if thread_index < fx.Int32(world_size):
                running_pool_offset = fx.Int32(0)
                for local_expert_index in range(experts_per_rank):
                    expert_total_count = fx.Int32(0)
                    for source_rank in range(world_size):
                        expert_total_count = expert_total_count + load_relaxed_at_address(
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
                        pool_base_resource,
                        thread_index * fx.Int32(experts_per_rank) + fx.Int32(local_expert_index),
                    )
                    running_pool_offset = running_pool_offset + padded_count
            fx.gpu.barrier()
            # C2 parallel across expert (needs POOL_BASE from C1)
            expert_index = thread_index
            while expert_index < fx.Int32(num_experts):
                preceding_count = fx.Int32(0)
                for source_rank in range(rank):
                    preceding_count = preceding_count + load_relaxed_at_address(
                        own_c_address, fx.Int32(source_rank * num_experts) + expert_index, scope="sys"
                    )
                pool_base_value = buffer_load(pool_base_resource, expert_index, vec_width=1, dtype=fx.T.i32())
                buffer_store(pool_base_value + preceding_count, start_per_expert_resource, expert_index)
                expert_index = expert_index + fx.Int32(block_threads)
            fx.gpu.barrier()
            # C3a parallel: destination/start/count per comm task
            comm_task_index = thread_index
            while comm_task_index < fx.Int32(num_experts):
                destination_rank = comm_task_index % fx.Int32(world_size)
                local_expert_index = comm_task_index // fx.Int32(world_size)
                expert_id = destination_rank * fx.Int32(experts_per_rank) + local_expert_index
                count_value = load_relaxed_at_address(
                    own_c_address, fx.Int32(rank * num_experts) + expert_id, scope="sys"
                )
                start_value = buffer_load(start_per_expert_resource, expert_id, vec_width=1, dtype=fx.T.i32())
                buffer_store(destination_rank, destination_resource, comm_task_index)
                buffer_store(start_value, start_resource, comm_task_index)
                buffer_store(count_value, count_resource, comm_task_index)
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
                            count_resource, fx.Int32(comm_task_counter), vec_width=1, dtype=fx.T.i32()
                        )
                        buffer_store(source_offset, source_offset_out_resource, fx.Int32(comm_task_counter))
                        buffer_store(source_offset, source_offset_per_expert_resource, fx.Int32(expert_id))
                        source_offset = source_offset + count_value
                        comm_task_counter = comm_task_counter + 1
                buffer_store(fx.Int32(num_experts), meta_scalars_resource, fx.Int32(2))
            # C4 parallel across local expert: tile_to_group + expected + total_rows (disjoint pool regions)
            if thread_index < fx.Int32(experts_per_rank):
                local_expert_index = thread_index
                expert_pool_base = buffer_load(
                    pool_base_resource,
                    fx.Int32(rank * experts_per_rank) + local_expert_index,
                    vec_width=1,
                    dtype=fx.T.i32(),
                )
                source_counts = []
                for source_rank in fx.range_constexpr(world_size):
                    source_counts.append(
                        load_relaxed_at_address(
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
                        local_expert_index, tile_to_group_resource, base_block_index + pool_block_offset
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
                                expected_resource, block_cursor, vec_width=1, dtype=fx.T.i32()
                            )
                            buffer_store(expected_value + fx.Int32(1), expected_resource, block_cursor)
                            block_cursor = block_cursor + fx.Int32(1)
                        within_expert_offset = within_expert_offset + count_value
                if local_expert_index == fx.Int32(experts_per_rank - 1):
                    total_rows = expert_pool_base + padded_count
                    buffer_store(total_rows, meta_scalars_resource, fx.Int32(0))
                    buffer_store(total_rows // fx.Int32(block_m), meta_scalars_resource, fx.Int32(1))

        grid_barrier(GRID_BARRIER_STATE, grid_blocks, thread_index)
        if is_profiler_thread:  # [PROF]
            if thread_index == fx.Int32(0):  # [PROF]
                buffer_store(_read_realtime(), profile_resource, fx.Int32(4))  # [PROF] after Phase C

        # ---- Phase D: scatter pairs (all blocks) ----
        pair_index = block_index * fx.Int32(block_threads) + thread_index
        while pair_index < fx.Int32(total_pairs):
            expert_id = buffer_load(
                topk_resource, pair_index + pair_index, vec_width=1, dtype=fx.T.i32()
            )  # low word of pair
            if expert_id >= fx.Int32(0):
                token_index = pair_index // fx.Int32(num_topk)
                topk_slot = pair_index % fx.Int32(num_topk)
                within_expert_position = atomic_add(
                    WITHIN_EXPERT_COUNTER, expert_id, fx.Int32(1), release=False, scope="agent"
                )
                expert_start = buffer_load(
                    start_per_expert_resource, expert_id, vec_width=1, dtype=fx.T.i32()
                )
                expert_source_offset = buffer_load(
                    source_offset_per_expert_resource, expert_id, vec_width=1, dtype=fx.T.i32()
                )
                destination_row = expert_start + within_expert_position
                buffer_store(
                    token_index, source_tokens_resource, expert_source_offset + within_expert_position
                )
                buffer_store(
                    topk_slot, source_topk_slot_resource, expert_source_offset + within_expert_position
                )  # topk slot per pair
                routing_weight = buffer_load(topk_weights_resource, pair_index, vec_width=1, dtype=fx.T.f32())
                buffer_store(
                    routing_weight, source_weight_resource, expert_source_offset + within_expert_position
                )  # routing weight per pair
                destination_rank = expert_id // fx.Int32(experts_per_rank)
                peer_origin_rank_address = buffer_load(
                    origin_rank_ptrs_resource, destination_rank, vec_width=1, dtype=fx.T.i64()
                )
                peer_origin_slot_address = buffer_load(
                    origin_slot_ptrs_resource, destination_rank, vec_width=1, dtype=fx.T.i64()
                )
                peer_origin_rank_resource = create_buffer_resource_from_addr(
                    peer_origin_rank_address, num_records_bytes=origin_buffer_bytes
                )
                peer_origin_slot_resource = create_buffer_resource_from_addr(
                    peer_origin_slot_address, num_records_bytes=origin_buffer_bytes
                )
                buffer_store(fx.Int32(rank), peer_origin_rank_resource, destination_row)
                # ---- token dedup: first-writer-wins primary election per (token, dest-rank) ----
                # The first pair (per token,rank) keeps dedup_src_row=-1 (primary -> XGMI push);
                # later pairs to the same rank store the primary's pool row (secondary -> dest-local copy).
                # Gated off by default (no dedup): every row is a primary -> full XGMI push.
                if fx.const_expr(dedup):
                    prev_primary_row = atomic_cas(
                        TOKEN_RANK_TABLE,
                        token_index * fx.Int32(world_size) + destination_rank,
                        expected=-1,
                        desired=destination_row,
                        scope="agent",
                    )
                    # source-indexed skip flag (parallels source_tokens): 1 = secondary (skip push)
                    source_dedup_flag = fx.arith.select(
                        prev_primary_row >= fx.Int32(0), fx.Int32(1), fx.Int32(0)
                    )
                    buffer_store(
                        source_dedup_flag,
                        source_dedup_resource,
                        expert_source_offset + within_expert_position,
                    )
                    if prev_primary_row >= fx.Int32(0):
                        peer_dedup_address = buffer_load(
                            dedup_src_row_ptrs_resource, destination_rank, vec_width=1, dtype=fx.T.i64()
                        )
                        peer_dedup_resource = create_buffer_resource_from_addr(
                            peer_dedup_address, num_records_bytes=origin_buffer_bytes
                        )
                        buffer_store(prev_primary_row, peer_dedup_resource, destination_row)
                # origin_slot = token-major position k*T+t, so the origin rank's combine
                # buffer is a dense [K, T, H] view -> the weighted topk reduce reads
                # comb[k, t] directly (must match _weighted_reduce_k's [K,T,H] layout).
                buffer_store(
                    topk_slot * fx.Int32(num_tokens) + token_index, peer_origin_slot_resource, destination_row
                )
            pair_index = pair_index + fx.Int32(grid_stride)
        grid_barrier(GRID_BARRIER_STATE, grid_blocks, thread_index)
        if is_profiler_thread:  # [PROF]
            if thread_index == fx.Int32(0):  # [PROF]
                buffer_store(_read_realtime(), profile_resource, fx.Int32(5))  # [PROF] after Phase D

        # ---- Post: reset SEND_LOCAL/WITHIN_EXPERT_COUNTER for the next launch ----
        reset_index = block_index * fx.Int32(block_threads) + thread_index
        while reset_index < fx.Int32(num_experts):
            buffer_store(fx.Int32(0), send_local_resource, reset_index)
            buffer_store(fx.Int32(0), within_expert_resource, reset_index)
            reset_index = reset_index + fx.Int32(grid_stride)
        if is_profiler_thread:  # [PROF]
            if thread_index == fx.Int32(0):  # [PROF]
                buffer_store(_read_realtime(), profile_resource, fx.Int32(6))  # [PROF] after Post reset

        # ---- Phase E: origins landed cross-rank ----
        barrier_block(SIGNAL_PTRS, rank, world_size, thread_index, block_index, False)
        if is_profiler_thread:  # [PROF]
            if thread_index == fx.Int32(0):  # [PROF]
                buffer_store(_read_realtime(), profile_resource, fx.Int32(7))  # [PROF] after Phase E (end)

    @flyc.jit
    def launch(
        topk_indices,
        send_local,
        within_expert_counter,
        c_buffer_ptrs,
        signal_ptrs,
        origin_rank_ptrs,
        origin_slot_ptrs,
        start_per_expert,
        source_offset_per_expert,
        pool_base,
        destination,
        start,
        count,
        source_offset_out,
        tile_to_group,
        expected,
        source_tokens,
        source_topk_slot,
        source_weight,
        topk_weights,
        meta_scalars,
        grid_barrier_state,
        token_rank_table,
        dedup_src_row_ptrs,
        source_dedup,
        profile,  # [PROF]
        stream: fx.Stream = fx.Stream(None),
    ):
        prologue_k(
            topk_indices,
            send_local,
            within_expert_counter,
            c_buffer_ptrs,
            signal_ptrs,
            origin_rank_ptrs,
            origin_slot_ptrs,
            start_per_expert,
            source_offset_per_expert,
            pool_base,
            destination,
            start,
            count,
            source_offset_out,
            tile_to_group,
            expected,
            source_tokens,
            source_topk_slot,
            source_weight,
            topk_weights,
            meta_scalars,
            grid_barrier_state,
            token_rank_table,
            dedup_src_row_ptrs,
            source_dedup,
            profile,
        ).launch(  # [PROF]
            grid=(grid_blocks, 1, 1), block=(block_threads, 1, 1), stream=stream
        )

    return launch


@functools.lru_cache(maxsize=8)
def _compile(
    num_tokens, num_topk, num_experts, world_size, rank, experts_per_rank, block_m, pool_capacity, dedup=False
):
    return _make_prologue(
        num_tokens,
        num_topk,
        num_experts,
        world_size,
        rank,
        experts_per_rank,
        block_m,
        pool_capacity,
        dedup=dedup,
    )


def mega_moe_prologue(
    topk_idx,
    topk_w,
    *,
    send_local,
    within_expert_counter,
    c_buffer_ptrs,
    signal_ptrs,
    origin_rank_ptrs,
    origin_slot_ptrs,
    start_per_expert,
    source_offset_per_expert,
    pool_base,
    destination,
    start,
    count,
    source_offset_out,
    tile_to_group,
    expected,
    source_tokens,
    source_topk_slot,
    source_weight,
    zero_topk_weights,
    origin_rank,
    origin_slot,
    meta_scalars,
    grid_barrier_state,
    token_rank_table,
    dedup_src_row_ptrs,
    dedup_src_row,
    source_dedup,
    profile,  # [PROF]
    num_tokens,
    num_topk,
    num_experts,
    world_size,
    rank,
    experts_per_rank,
    block_m,
    pool_capacity,
    dedup=False,
    launch_cache=None,
    no_cpu_sync=True,
):
    """One fused-prologue kernel launch over caller-owned buffers (no_cpu_sync controls max_num_token)."""
    # int64 viewed as int32 (free reinterpret); kernel reads the low word
    topk_int32_view = topk_idx.to(torch.int64).contiguous().view(-1).view(torch.int32)
    topk_weights_flat = (
        topk_w.to(torch.float32).contiguous().view(-1) if topk_w is not None else zero_topk_weights
    )

    stream = torch.cuda.current_stream()
    launch_function = _compile(
        num_tokens, num_topk, num_experts, world_size, rank, experts_per_rank, block_m, pool_capacity, dedup
    )
    kernel_arguments = (
        topk_int32_view,
        send_local,
        within_expert_counter,
        c_buffer_ptrs,
        signal_ptrs,
        origin_rank_ptrs,
        origin_slot_ptrs,
        start_per_expert,
        source_offset_per_expert,
        pool_base,
        destination,
        start,
        count,
        source_offset_out,
        tile_to_group,
        expected,
        source_tokens,
        source_topk_slot,
        source_weight,
        topk_weights_flat,
        meta_scalars,
        grid_barrier_state,
        token_rank_table,
        dedup_src_row_ptrs,
        source_dedup,
        profile,  # [PROF]
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
    # num_comm == num_experts always; meta_scalars holds real total_rows/n_mblk on-device
    comm_tasks = CommTasks(
        destination,
        start,
        count,
        source_offset_out,
        source_tokens,
        num_experts,
        src_k=source_topk_slot,
        src_w=source_weight,
    )
    max_num_token = pool_capacity if no_cpu_sync else int(meta_scalars[0].item())
    return PrologueResult(
        comm_tasks,
        tile_to_group,
        expected,
        origin_rank,
        origin_slot,
        pool_capacity // block_m,
        max_num_token,
        dedup_src_row,
        source_dedup,
    )


# --------------------------------------------------------------------------- #
# Ergonomic workspace: owns the buffer-shape contract so callers don't hand-roll
# ~24 tensors. The low-level mega_moe_prologue() above stays the stable primitive.
# --------------------------------------------------------------------------- #
class MegaMoePrologue:
    """Caller-owned workspace + launcher for the fused MoE dispatch prologue.

    ``allocate`` sizes every caller-owned buffer for a fixed config (single
    source of truth for shapes); ``run`` launches over them and returns a
    ``PrologueResult``. Buffers are reused across launches (the kernel self-resets
    its counters), so allocate once and call ``run`` per step.

    The four symmetric tensors (c_buffer/signal/origin_rank/origin_slot) are
    allocated as plain device memory with *self* peer-pointer tables -- correct
    as-is for ``world_size == 1``. For ``world_size > 1`` they must live in
    symmetric (rocSHMEM) memory: assign each peer's base into the matching row of
    the ``*_ptrs`` tables before the first ``run``.
    """

    # symmetric tensors held for lifetime/data_ptr; they reach the kernel via the
    # *_ptrs tables, not as direct kernel args
    _OWNED_NOT_KERNEL = ("c_buffer", "signal")

    def __init__(self, config: dict, buffers: dict):
        self.config = config  # scalar kwargs forwarded to mega_moe_prologue
        self.buffers = buffers  # all owned device tensors (outputs / introspection)
        self.launch_cache: dict = {}
        self._kernel_args = {k: v for k, v in buffers.items() if k not in self._OWNED_NOT_KERNEL}

    @classmethod
    def allocate(
        cls,
        *,
        num_tokens,
        num_topk,
        num_experts,
        pool_capacity,
        world_size=1,
        rank=0,
        block_m=256,
        device="cuda",
        no_cpu_sync=True,
        dedup=False,
    ):
        assert num_experts % world_size == 0, "num_experts must be divisible by world_size"
        assert pool_capacity % block_m == 0, "pool_capacity must be a multiple of block_m"
        experts_per_rank = num_experts // world_size  # derived, never passed in
        n_mblk = pool_capacity // block_m
        i32 = lambda n: torch.zeros(n, dtype=torch.int32, device=device)

        config = dict(
            num_tokens=num_tokens,
            num_topk=num_topk,
            num_experts=num_experts,
            world_size=world_size,
            rank=rank,
            experts_per_rank=experts_per_rank,
            block_m=block_m,
            pool_capacity=pool_capacity,
            no_cpu_sync=no_cpu_sync,
            dedup=dedup,
        )
        buffers = dict(
            # ---- per-rank scratch (counters auto-reset by the kernel each launch) ----
            send_local=i32(num_experts),
            within_expert_counter=i32(num_experts),
            start_per_expert=i32(num_experts),
            source_offset_per_expert=i32(num_experts),
            pool_base=i32(num_experts),
            destination=i32(num_experts),
            start=i32(num_experts),
            count=i32(num_experts),
            source_offset_out=i32(num_experts),
            tile_to_group=i32(n_mblk),
            expected=i32(n_mblk),
            source_tokens=i32(pool_capacity),
            source_topk_slot=i32(pool_capacity),
            source_weight=torch.zeros(pool_capacity, dtype=torch.float32, device=device),
            zero_topk_weights=torch.zeros(num_tokens * num_topk, dtype=torch.float32, device=device),
            meta_scalars=i32(8),
            grid_barrier_state=i32(2),
            # dedup election scratch (local; kernel re-inits to -1 each launch)
            token_rank_table=i32(num_tokens * world_size),
            # source-indexed skip flag (local; written for every pair in Phase D)
            source_dedup=i32(pool_capacity),
            profile=torch.zeros(8, dtype=torch.int64, device=device),  # [PROF] 8 phase-boundary timestamps
            # ---- symmetric tensors (see class docstring for world_size > 1) ----
            c_buffer=i32(world_size * num_experts),
            signal=i32(world_size),
            origin_rank=torch.full((pool_capacity,), -1, dtype=torch.int32, device=device),
            origin_slot=torch.full((pool_capacity,), -1, dtype=torch.int32, device=device),
            dedup_src_row=torch.full((pool_capacity,), -1, dtype=torch.int32, device=device),
        )

        def self_ptr_table(t):  # row `rank` -> own base; peers filled by the runtime
            tab = torch.zeros(world_size, dtype=torch.int64, device=device)
            tab[rank] = t.data_ptr()
            return tab

        buffers.update(
            c_buffer_ptrs=self_ptr_table(buffers["c_buffer"]),
            signal_ptrs=self_ptr_table(buffers["signal"]),
            origin_rank_ptrs=self_ptr_table(buffers["origin_rank"]),
            origin_slot_ptrs=self_ptr_table(buffers["origin_slot"]),
            dedup_src_row_ptrs=self_ptr_table(buffers["dedup_src_row"]),
        )
        return cls(config, buffers)

    def run(self, topk_idx, topk_w: Optional[torch.Tensor] = None) -> PrologueResult:
        # buffer + config names match the primitive's kwargs, so splat (single
        # source of truth = allocate; no per-buffer forwarding to drift)
        num_tokens, num_topk = self.config["num_tokens"], self.config["num_topk"]
        assert topk_idx.shape[-1] == num_topk, "topk_idx last dim must equal num_topk"
        assert (
            topk_idx.numel() == num_tokens * num_topk
        ), "topk_idx token count must match the allocated workspace"
        return mega_moe_prologue(
            topk_idx, topk_w, **self._kernel_args, **self.config, launch_cache=self.launch_cache
        )
