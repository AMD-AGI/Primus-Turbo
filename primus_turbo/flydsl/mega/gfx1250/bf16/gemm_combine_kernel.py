###############################################################################
# SPDX-License-Identifier: Apache-2.0
#
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 FlyDSL Project Contributors
#
# Adapted from FlyDSL (https://github.com/ROCm/FlyDSL)
# Modified by the Primus-Turbo team.
#
# This file is distributed under the Apache License 2.0 (see LICENSE-APACHE),
# not the MIT license that covers the rest of Primus-Turbo (see LICENSE).
###############################################################################

"""Mega MoE fused L2 GEMM + combine for gfx1250: ``out[t] = sum over t's routes of act[p] @ w2[e(p)].T``.

The three roles of the gfx950 ``grouped_gemm_combine_bf16_kernel``, in one launch:

* **GEMM** -- one ``tile_m x tile_n`` tile of the pool, ``l2[pool block] = act @ w2[e].T``, stored
  at device scope into a local staging buffer; once all its waves' stores retire, the workgroup
  bumps the pool block's combine flag.
* **combine** -- takes 16-fold-row chunks in pool order from a counter, waits for the pool blocks
  a chunk depends on, sums each recv token's rows on this rank and pushes the sum to the source
  rank's combine buffer (``combine_bf16_tile``).
* **reduce** -- run by the combine workgroups once no chunk is left: takes token groups from a
  second counter, waits for each token's primary-route rows to land and sums them into ``out``
  (``topk_reduce_bf16_tile``); it also tops up the flags of the pool blocks this call left empty.

Workgroup order: the GEMM tiles, then the tail combine workgroups (one per CU; they start as the
GEMM drains), so nothing that spins can keep a GEMM tile from a CU. ``num_lead_combine`` > 0 moves
that many combine workgroups to after ``lead_gemm_percent`` of the tiles, to push while the GEMM
runs; on MI455X they slow the GEMM by more than their early push saves, so the default has none. The GEMM tile is reached by every workgroup in straight-line code before the
role branches and is masked with ``live`` elsewhere -- no dynamic branch covers the TDM pipeline.
"""

from __future__ import annotations

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl._mlir.dialects import arith as std_arith
from flydsl.expr import arith, const_expr, rocdl
from flydsl.expr.arith import _to_raw as _raw
from flydsl.expr.typing import Constexpr, T
from flydsl.expr.utils.arith import ArithValue

from primus_turbo.flydsl.grouped_gemm.grouped_gemm_bf16_kernel_gfx1250 import (
    _as_ptr,
    _i32_table,
    _require_gfx1250,
)
from primus_turbo.flydsl.mega.gfx1250.bf16.gemm_nt_tile import (
    EB,
    GFX1250_LDS_BYTES,
    NUM_BUFFERS,
    WAVE,
    nt_gemm_tile,
    tile_geometry,
)
from primus_turbo.flydsl.mega.gfx1250.dispatch_prologue_kernel import DispatchHandle
from primus_turbo.flydsl.mega.gfx1250.ep_intranode import (
    combine_bf16_tile,
    spin_until_flag_reaches,
    topk_reduce_bf16_tile,
)
from primus_turbo.flydsl.mega.gfx1250.prims import (
    CPOL_DEVICE_STREAM,
    CPOL_SYSTEM_STREAM,
    atomic_add,
    cast,
    ld,
    st,
    wait_mem_all,
)
from primus_turbo.flydsl.mega.gfx1250.symm_buffer import (
    BLOCK_M,
    TOKEN_DTYPE,
    Workspace,
    get_symm_buffer_for_mega_moe,
)
from primus_turbo.flydsl.utils.buffer_ops import buffer_load, create_buffer_resource_from_addr

# fold rows per combine work item: two rows for each wave of the default 8-wave workgroup
_FOLD_CHUNK_ROWS = 16
# reduce work item: this many workgroup-sized token groups per counter take
_REDUCE_GROUPS_PER_TAKE = 4


def _addr(p):
    return fx.Int64(fx.ptrtoint(p))


@flyc.jit
def _launch_gemm_combine_nt(
    arg_out: fx.Pointer,
    arg_act: fx.Pointer,
    arg_w: fx.Pointer,
    arg_l2: fx.Pointer,
    arg_tile_to_expert: fx.Pointer,
    arg_num_tile_blocks: fx.Pointer,
    arg_recv_dst_rank: fx.Pointer,
    arg_recv_start_row: fx.Pointer,
    arg_recv_fold_count: fx.Pointer,
    arg_pool_row_to_route: fx.Pointer,
    arg_pool_row_to_recv_token: fx.Pointer,
    arg_recv_token_to_pool_rows: fx.Pointer,
    arg_topk_idx: fx.Pointer,
    arg_num_tokens_per_rank: fx.Pointer,
    arg_parity: fx.Pointer,
    arg_combine_expected: fx.Pointer,
    arg_reduce_expected: fx.Pointer,
    arg_chunk_counter: fx.Pointer,
    arg_grad_gate: fx.Pointer,
    arg_grad_topk_weights: fx.Pointer,
    arg_bias: fx.Pointer,
    arg_bias_scale: fx.Pointer,
    sym_buffer,
    stream: fx.Stream,
    grid_size: fx.Int32,
    N: Constexpr[int],
    K: Constexpr[int],
    MAX_TILES: Constexpr[int],
    N_BLOCKS_N: Constexpr[int],
    NUM_LEAD_COMBINE: Constexpr[int],
    NUM_TAIL_COMBINE: Constexpr[int],
    LEAD_GEMM_PERCENT: Constexpr[int],
    NUM_RANKS: Constexpr[int],
    RANK: Constexpr[int],
    NUM_EXPERTS: Constexpr[int],
    NUM_MAX_TOKENS_PER_RANK: Constexpr[int],
    NUM_TOPK: Constexpr[int],
    tile_m: Constexpr[int],
    tile_n: Constexpr[int],
    tile_k: Constexpr[int],
    m_warp: Constexpr[int],
    n_warp: Constexpr[int],
    HAS_GRAD_GATE: Constexpr[bool],
    B_NN: Constexpr[bool],
    HAS_BIAS: Constexpr[bool],
    SCATTER: Constexpr[bool],
):
    block = m_warp * n_warp * WAVE
    arena_bytes = tile_geometry(tile_m, tile_n, tile_k, B_NN)[3]
    if arena_bytes > GFX1250_LDS_BYTES:
        raise ValueError(f"LDS arena {arena_bytes} B exceeds {GFX1250_LDS_BYTES} B")
    NUM_COMBINE = NUM_LEAD_COMBINE + NUM_TAIL_COMBINE
    num_experts_per_rank = NUM_EXPERTS // NUM_RANKS

    @flyc.kernel(known_block_size=[64, 1, 1])
    def epoch_bump(
        arg_parity: fx.Pointer,
        arg_combine_expected: fx.Pointer,
        arg_reduce_expected: fx.Pointer,
        arg_chunk_counter: fx.Pointer,
    ):
        """Flip the parity; the new bank expects N_BLOCKS_N more per pool block and one more per route.

        Also rearms the combine and reduce work counters for this launch."""
        if fx.Int32(fx.thread_idx.x) == fx.Int32(0):
            st(_addr(arg_chunk_counter), fx.Int32(0), fx.Int32(0))
            st(_addr(arg_chunk_counter), fx.Int32(1), fx.Int32(0))
            parity_addr = _addr(arg_parity)
            new_parity = ld(parity_addr, fx.Int32(0), dtype=fx.T.i64) ^ fx.Int64(1)
            st(parity_addr, fx.Int32(0), new_parity)
            bank = cast(new_parity, fx.T.i32)
            combine_addr, reduce_addr = _addr(arg_combine_expected), _addr(arg_reduce_expected)
            st(combine_addr, bank, ld(combine_addr, bank, dtype=fx.T.i64) + fx.Int64(N_BLOCKS_N))
            st(reduce_addr, bank, ld(reduce_addr, bank, dtype=fx.T.i64) + fx.Int64(1))

    @flyc.kernel(known_block_size=[block, 1, 1])
    def gemm_combine_nt(
        arg_out: fx.Pointer,
        arg_act: fx.Pointer,
        arg_w: fx.Pointer,
        arg_l2: fx.Pointer,
        arg_tile_to_expert: fx.Pointer,
        arg_num_tile_blocks: fx.Pointer,
        arg_recv_dst_rank: fx.Pointer,
        arg_recv_start_row: fx.Pointer,
        arg_recv_fold_count: fx.Pointer,
        arg_pool_row_to_route: fx.Pointer,
        arg_pool_row_to_recv_token: fx.Pointer,
        arg_recv_token_to_pool_rows: fx.Pointer,
        arg_topk_idx: fx.Pointer,
        arg_num_tokens_per_rank: fx.Pointer,
        arg_parity: fx.Pointer,
        arg_combine_expected: fx.Pointer,
        arg_reduce_expected: fx.Pointer,
        arg_chunk_counter: fx.Pointer,
        arg_grad_gate: fx.Pointer,
        arg_grad_topk_weights: fx.Pointer,
        arg_bias: fx.Pointer,
        arg_bias_scale: fx.Pointer,
        sym_buffer,
    ):
        arena = fx.SharedAllocator(static=False)
        arena.allocate(arena_bytes)
        base_ptr = arena.base_ptr
        tid = fx.thread_idx.x
        bid = fx.Int32(fx.block_idx.x)

        # token pools are N wide (the model hidden); hoist everything workspace-derived above
        # the branches (the rewriter can't carry Workspace)
        workspace = Workspace(
            sym_buffer.get_base_ptr(),
            NUM_RANKS,
            NUM_EXPERTS,
            NUM_MAX_TOKENS_PER_RANK,
            NUM_TOPK,
            N,
            token_dtype=TOKEN_DTYPE,
        )
        num_max_routes = int(workspace.num_max_routes)
        combine_flag_base = workspace.get_combine_flag_ptr()
        reduce_flag_base = workspace.get_reduce_flag_ptr()
        combine_token_base = workspace.get_combine_token_buffer_ptr()
        combine_gate_base = workspace.get_combine_gate_ptr()
        combine_token_res = create_buffer_resource_from_addr(
            workspace.get_combine_token_buffer_ptr(), num_records_bytes=num_max_routes * N * EB
        )
        combine_gate_res = (
            create_buffer_resource_from_addr(
                workspace.get_combine_gate_ptr(), num_records_bytes=num_max_routes * 4
            )
            if HAS_GRAD_GATE
            else None
        )
        # this launch's epoch, already bumped
        parity = cast(ld(_addr(arg_parity), fx.Int32(0), dtype=fx.T.i64), fx.T.i32)
        combine_bank = parity * fx.Int32(MAX_TILES)
        reduce_bank = parity * fx.Int32(num_max_routes)
        expected_combine = ld(_addr(arg_combine_expected), parity, dtype=fx.T.i64)
        expected_reduce = ld(_addr(arg_reduce_expected), parity, dtype=fx.T.i64)
        l2_base = _addr(fx.recast_iter(fx.Int8, arg_l2))

        num_tiles_m = rocdl.readfirstlane(T.i32, _i32_table(arg_num_tile_blocks)[fx.Int32(0)])
        combine_start = num_tiles_m * fx.Int32(N_BLOCKS_N * LEAD_GEMM_PERCENT // 100)
        tail_combine_start = num_tiles_m * fx.Int32(N_BLOCKS_N) + fx.Int32(NUM_LEAD_COMBINE)
        is_lead_combine = (bid >= combine_start) & (bid < combine_start + fx.Int32(NUM_LEAD_COMBINE))
        is_combine = is_lead_combine | (bid >= tail_combine_start)
        # the combine workgroups also run the reduce, numbered lead first
        combine_idx = arith.select(
            is_lead_combine, bid - combine_start, bid - tail_combine_start + fx.Int32(NUM_LEAD_COMBINE)
        )
        gemm_block_idx = arith.select(bid < combine_start, bid, bid - fx.Int32(NUM_LEAD_COMBINE))

        # ---- GEMM role: straight-line for every workgroup, masked by `live` ----
        is_gemm = arith.select(is_combine, fx.Int32(0), fx.Int32(1)) == fx.Int32(1)
        live = is_gemm & (gemm_block_idx < num_tiles_m * fx.Int32(N_BLOCKS_N))
        gemm_block_idx = live.select(gemm_block_idx, fx.Int32(0))
        block_m = rocdl.readfirstlane(T.i32, gemm_block_idx // fx.Int32(N_BLOCKS_N))
        block_n = rocdl.readfirstlane(T.i32, gemm_block_idx % fx.Int32(N_BLOCKS_N))
        expert = rocdl.readfirstlane(T.i32, _i32_table(arg_tile_to_expert)[block_m])
        expert = live.select(expert, fx.Int32(0))
        blk_n = block_n * fx.Int32(tile_n)
        m_row = block_m * fx.Int32(tile_m)
        c_rows = None
        c_bias = None
        if const_expr(SCATTER):
            # dead tiles and the combine workgroups run the tile too, possibly over an empty pool
            pool_bytes = num_tiles_m * fx.Int32(tile_m * 4)
            recv_token_res = create_buffer_resource_from_addr(
                _addr(arg_pool_row_to_recv_token), num_records_bytes=pool_bytes
            )
            route_res = create_buffer_resource_from_addr(
                _addr(arg_pool_row_to_route), num_records_bytes=pool_bytes
            )
            scale_res = (
                create_buffer_resource_from_addr(_addr(arg_bias_scale), num_records_bytes=pool_bytes)
                if HAS_BIAS
                else None
            )

            def c_rows(lane_row):
                # each row goes straight to its route's combine row on the token's source rank
                pool_row = m_row + lane_row
                recv_token = buffer_load(recv_token_res, pool_row, vec_width=1, dtype=T.i32)
                route = buffer_load(route_res, pool_row, vec_width=1, dtype=T.i32)
                # padding rows carry the num_max_recv_tokens sentinel
                ok = (
                    live
                    & (recv_token >= fx.Int32(0))
                    & (recv_token < fx.Int32(NUM_RANKS * NUM_MAX_TOKENS_PER_RANK))
                    & (route >= fx.Int32(0))
                    & (route < fx.Int32(num_max_routes))
                )
                src_rank = ok.select(recv_token // fx.Int32(NUM_MAX_TOKENS_PER_RANK), fx.Int32(RANK))
                route = ok.select(route, fx.Int32(0))
                addr = sym_buffer.map(combine_token_base, src_rank) + fx.Int64(route) * fx.Int64(N * EB)
                scale = (
                    buffer_load(scale_res, pool_row, vec_width=1, dtype=T.f32)
                    if const_expr(HAS_BIAS)
                    else None
                )
                return addr, ok, scale, (pool_row, src_rank, route, ok)

            if const_expr(HAS_BIAS):
                c_bias = _addr(arg_bias) + fx.Int64(expert) * fx.Int64(N * EB)
        rows_out = nt_gemm_tile(
            base_ptr,
            gA_base=fx.recast_iter(fx.Int8, arg_act),
            lda=K,
            a_row0=m_row,
            a_rows=fx.Int32(0),
            row_idx_table=_i32_table(arg_tile_to_expert),
            row_idx_base=fx.Int32(0),
            gB_base=fx.recast_iter(fx.Int8, arg_w),
            ldb=N if B_NN else K,
            b_slab=fx.Int64(expert) * fx.Int64(N * K * EB),
            blk_n=blk_n,
            c_addr=l2_base + (fx.Int64(m_row) * fx.Int64(N) + fx.Int64(blk_n)) * fx.Int64(EB),
            ldc=N,
            m_rows=fx.Int32(tile_m),
            n_valid=fx.Int32(N) - blk_n,
            i32_k=fx.Int32(K),
            live=live,
            tile_m=tile_m,
            tile_n=tile_n,
            tile_k=tile_k,
            m_warp=m_warp,
            n_warp=n_warp,
            gather=False,
            # scattered rows are read by the reduce on their source rank; the staged ones by
            # combine workgroups on other CUs
            c_cache_modifier=CPOL_SYSTEM_STREAM if SCATTER else CPOL_DEVICE_STREAM,
            b_nn=B_NN,
            i32_n=fx.Int32(N),
            c_rows=c_rows,
            c_bias=c_bias,
        )
        if const_expr(SCATTER):
            pool_row, src_rank, route, row_ok = rows_out[3]
        if const_expr(SCATTER and HAS_GRAD_GATE):
            # one N block per pool block carries each row's gate value to its route
            if block_n == fx.Int32(0):
                if row_ok:
                    gate = buffer_load(
                        create_buffer_resource_from_addr(_addr(arg_grad_gate), num_records_bytes=pool_bytes),
                        pool_row,
                        vec_width=1,
                        dtype=T.f32,
                    )
                    st(
                        sym_buffer.map(combine_gate_base, src_rank),
                        route,
                        ArithValue(std_arith.BitcastOp(T.i32, _raw(gate)).result),
                        scope="sys",
                    )
        # every wave's C stores retire before the flag; two rendezvous, which LLVM would fold
        wait_mem_all()
        fx.gpu.barrier()
        wait_mem_all()
        fx.gpu.barrier()
        if live:
            if fx.Int32(tid) == fx.Int32(0):
                flag_before = atomic_add(combine_flag_base, combine_bank + block_m, fx.Int64(1))
                if const_expr(SCATTER):
                    is_last_n = (flag_before + fx.Int64(1) == expected_combine).select(
                        fx.Int32(1), fx.Int32(0)
                    )
                    st(_addr(base_ptr), fx.Int32(0), is_last_n, scope="workgroup", space=3)
        if const_expr(SCATTER):
            fx.gpu.barrier()
            is_last_n = live & (ld(_addr(base_ptr), fx.Int32(0), scope="workgroup", space=3) == fx.Int32(1))
            # Every N block's stores of this pool block retired before its flag add, so the
            # last one to add publishes the block's rows: their routes' reduce flags go up.
            if is_last_n:
                if row_ok:
                    st(
                        sym_buffer.map(reduce_flag_base, src_rank),
                        reduce_bank + route,
                        expected_reduce,
                        scope="sys",
                    )

        if is_combine:
            lds = _addr(base_ptr)
            if const_expr(not SCATTER):
                # Work items are _FOLD_CHUNK_ROWS-row chunks of the fold rows of all segments, in pool
                # order, taken from one counter: a segment can hold most of the fold rows (a token's
                # fold row sits in its highest local expert's segment), and the lead workgroups, if
                # any, get the early chunks while the tail ones, which start once the GEMM has drained,
                # take what is left. LDS: [0] the taken chunk, [2 + s] segment s's first
                # chunk, [2 + NUM_EXPERTS] the chunk count.
                seg = fx.Int32(tid)
                while seg < fx.Int32(NUM_EXPERTS):
                    fold_count = _i32_table(arg_recv_fold_count)[seg]
                    num_chunks = (fold_count + fx.Int32(_FOLD_CHUNK_ROWS - 1)) // fx.Int32(_FOLD_CHUNK_ROWS)
                    st(lds, fx.Int32(2) + seg, num_chunks, scope="workgroup", space=3)
                    seg = seg + fx.Int32(block)
                fx.gpu.barrier()
                if fx.Int32(tid) == fx.Int32(0):
                    running = fx.Int32(0)
                    for s_idx in range(fx.Int32(NUM_EXPERTS)):
                        num_chunks = ld(lds, fx.Int32(2) + s_idx, scope="workgroup", space=3)
                        st(lds, fx.Int32(2) + s_idx, running, scope="workgroup", space=3)
                        running = running + num_chunks
                    st(lds, fx.Int32(2 + NUM_EXPERTS), running, scope="workgroup", space=3)
                    taken = atomic_add(_addr(arg_chunk_counter), fx.Int32(0), fx.Int32(1))
                    st(lds, fx.Int32(0), taken, scope="workgroup", space=3)
                fx.gpu.barrier()
                total_chunks = rocdl.readfirstlane(
                    T.i32, ld(lds, fx.Int32(2 + NUM_EXPERTS), scope="workgroup", space=3)
                )
                chunk = rocdl.readfirstlane(T.i32, ld(lds, fx.Int32(0), scope="workgroup", space=3))
                # every pool block below next_tile_m is already confirmed, so each is polled once
                next_tile_m = fx.Int32(0)
                segment_idx = fx.Int32(0)
                while chunk < total_chunks:
                    # taken chunks only grow, so the segment cursor only moves forward
                    while ld(lds, fx.Int32(3) + segment_idx, scope="workgroup", space=3) <= chunk:
                        segment_idx = segment_idx + fx.Int32(1)
                    segment_idx = rocdl.readfirstlane(T.i32, segment_idx)
                    first_chunk = ld(lds, fx.Int32(2) + segment_idx, scope="workgroup", space=3)
                    start_row = rocdl.readfirstlane(T.i32, _i32_table(arg_recv_start_row)[segment_idx])
                    fold_count = rocdl.readfirstlane(T.i32, _i32_table(arg_recv_fold_count)[segment_idx])
                    part_offset = (chunk - first_chunk) * fx.Int32(_FOLD_CHUNK_ROWS)
                    num_part_rows = arith.select(
                        fold_count - part_offset < fx.Int32(_FOLD_CHUNK_ROWS),
                        fold_count - part_offset,
                        fx.Int32(_FOLD_CHUNK_ROWS),
                    )
                    # a fold row's other pool rows lie in lower blocks, so wait for the whole prefix
                    last_tile_m = (start_row + part_offset + num_part_rows - fx.Int32(1)) // fx.Int32(BLOCK_M)
                    if fx.Int32(tid) == fx.Int32(0):
                        polled_tile_m = next_tile_m
                        while polled_tile_m <= last_tile_m:
                            spin_until_flag_reaches(
                                combine_flag_base,
                                combine_bank + polled_tile_m,
                                expected_combine,
                                "agent",
                                RANK,
                                "combine",
                            )
                            polled_tile_m = polled_tile_m + fx.Int32(1)
                        # take the next chunk now, so its counter round trip overlaps this one's push
                        taken = atomic_add(_addr(arg_chunk_counter), fx.Int32(0), fx.Int32(1))
                        st(lds, fx.Int32(1), taken, scope="workgroup", space=3)
                    next_tile_m = arith.select(
                        last_tile_m < next_tile_m, next_tile_m, last_tile_m + fx.Int32(1)
                    )
                    wait_mem_all()
                    fx.gpu.barrier()
                    combine_bf16_tile(
                        sym_buffer,
                        workspace,
                        thread_index=tid,
                        segment_idx=segment_idx,
                        first_fold_row=start_row + part_offset,
                        num_fold_rows=num_part_rows,
                        recv_dst_rank_res=create_buffer_resource_from_addr(_addr(arg_recv_dst_rank)),
                        pool_row_to_route_res=create_buffer_resource_from_addr(_addr(arg_pool_row_to_route)),
                        pool_row_to_recv_token_res=create_buffer_resource_from_addr(
                            _addr(arg_pool_row_to_recv_token)
                        ),
                        recv_token_to_pool_rows_res=create_buffer_resource_from_addr(
                            _addr(arg_recv_token_to_pool_rows)
                        ),
                        l2_ptr=l2_base,
                        epoch=expected_reduce,
                        reduce_bank=reduce_bank,
                        grad_gate_res=(
                            create_buffer_resource_from_addr(_addr(arg_grad_gate)) if HAS_GRAD_GATE else None
                        ),
                        has_grad_gate=HAS_GRAD_GATE,
                        bias=(
                            (
                                _addr(arg_bias),
                                create_buffer_resource_from_addr(_addr(arg_tile_to_expert)),
                                create_buffer_resource_from_addr(_addr(arg_bias_scale)),
                            )
                            if HAS_BIAS
                            else None
                        ),
                        block_threads=block,
                    )
                    chunk = rocdl.readfirstlane(T.i32, ld(lds, fx.Int32(1), scope="workgroup", space=3))
                    fx.gpu.barrier()  # every wave has the chunk before thread 0 takes the next one

            # ---- reduce: every chunk is taken by a running workgroup, so the flags waited on come ----
            # pool blocks this call left empty still have to reach the never-reset expected
            num_empty_tiles_m = fx.Int32(MAX_TILES) - num_tiles_m
            num_local_empty_tiles_m = (
                num_empty_tiles_m - combine_idx + fx.Int32(NUM_COMBINE - 1)
            ) // fx.Int32(NUM_COMBINE)
            for empty_iter in range(num_local_empty_tiles_m):
                empty_tile_m = num_tiles_m + combine_idx + empty_iter * fx.Int32(NUM_COMBINE)
                if fx.Int32(tid) == fx.Int32(0):
                    atomic_add(combine_flag_base, combine_bank + empty_tile_m, fx.Int64(N_BLOCKS_N))
            # tokens are taken from a counter too: the workgroups do not all become resident at
            # once (blocks are spread over the XCDs unevenly), and a late one must not hold a
            # static share of the tokens
            group_tokens = block // 64
            take_tokens = _REDUCE_GROUPS_PER_TAKE * group_tokens
            num_tokens = rocdl.readfirstlane(T.i32, _i32_table(arg_num_tokens_per_rank)[fx.Int32(RANK)])
            if fx.Int32(tid) == fx.Int32(0):
                taken = atomic_add(_addr(arg_chunk_counter), fx.Int32(1), fx.Int32(1))
                st(lds, fx.Int32(0), taken, scope="workgroup", space=3)
            fx.gpu.barrier()
            take = rocdl.readfirstlane(T.i32, ld(lds, fx.Int32(0), scope="workgroup", space=3))
            while take * fx.Int32(take_tokens) < num_tokens:
                if fx.Int32(tid) == fx.Int32(0):
                    taken = atomic_add(_addr(arg_chunk_counter), fx.Int32(1), fx.Int32(1))
                    st(lds, fx.Int32(1), taken, scope="workgroup", space=3)
                for group in range(fx.Int32(_REDUCE_GROUPS_PER_TAKE)):
                    topk_reduce_bf16_tile(
                        thread_index=tid,
                        reduce_block_idx=take * fx.Int32(_REDUCE_GROUPS_PER_TAKE) + group,
                        num_reduce_warps=NUM_MAX_TOKENS_PER_RANK,  # one token per warp per group
                        num_topk=NUM_TOPK,
                        hidden=N,
                        num_experts=NUM_EXPERTS,
                        num_experts_per_rank=num_experts_per_rank,
                        num_max_routes=num_max_routes,
                        rank=RANK,
                        combine_token_res=combine_token_res,
                        output_res=create_buffer_resource_from_addr(_addr(arg_out)),
                        topk_indices_res=create_buffer_resource_from_addr(_addr(arg_topk_idx)),
                        num_tokens_res=create_buffer_resource_from_addr(_addr(arg_num_tokens_per_rank)),
                        reduce_flag_base=reduce_flag_base,
                        reduce_bank=reduce_bank,
                        epoch=expected_reduce,
                        combine_gate_res=combine_gate_res,
                        grad_topk_weights_res=(
                            create_buffer_resource_from_addr(_addr(arg_grad_topk_weights))
                            if HAS_GRAD_GATE
                            else None
                        ),
                        has_grad_gate=HAS_GRAD_GATE,
                        block_threads=block,
                        vec_unroll=4,
                        all_routes=SCATTER,
                    )
                fx.gpu.barrier()
                take = rocdl.readfirstlane(T.i32, ld(lds, fx.Int32(1), scope="workgroup", space=3))
                fx.gpu.barrier()  # every wave has the take before thread 0 takes the next one

    epoch_bump(arg_parity, arg_combine_expected, arg_reduce_expected, arg_chunk_counter).launch(
        grid=(1, 1, 1), block=(64, 1, 1), stream=stream
    )
    gemm_combine_nt(
        arg_out,
        arg_act,
        arg_w,
        arg_l2,
        arg_tile_to_expert,
        arg_num_tile_blocks,
        arg_recv_dst_rank,
        arg_recv_start_row,
        arg_recv_fold_count,
        arg_pool_row_to_route,
        arg_pool_row_to_recv_token,
        arg_recv_token_to_pool_rows,
        arg_topk_idx,
        arg_num_tokens_per_rank,
        arg_parity,
        arg_combine_expected,
        arg_reduce_expected,
        arg_chunk_counter,
        arg_grad_gate,
        arg_grad_topk_weights,
        arg_bias,
        arg_bias_scale,
        sym_buffer,
    ).launch(grid=(grid_size, 1, 1), block=(block, 1, 1), stream=stream)


_launch_gemm_combine_nt.compile_hints["llvm_options"] = {"amdgpu-expert-scheduling-mode": True}

_COMPILED: dict = {}


@functools.lru_cache(maxsize=None)
def _chunk_counter(device: torch.device) -> torch.Tensor:
    """The combine chunk and reduce token counters; the launch's epoch bump rearms them."""
    return torch.zeros(2, dtype=torch.int32, device=device)


_CONSTEXPR0 = 25


def gemm_combine_nt_gfx1250(
    act: torch.Tensor,
    w2: torch.Tensor,
    handle: DispatchHandle,
    topk_idx: torch.Tensor,
    *,
    num_lead_combine: int = 0,
    num_tail_combine: int = 0,
    lead_gemm_percent: int = 25,
    tile_m: int = 256,
    tile_n: int = 256,
    tile_k: int = 0,
    m_warp: int = 4,
    n_warp: int = 2,
    grad_gate: torch.Tensor | None = None,
    b_nn: bool = False,
    bias: torch.Tensor | None = None,
    bias_scale: torch.Tensor | None = None,
    scatter: bool = True,
):
    """``act`` [pool rows, K] (pool order, as the L1 + activation left it) -> ``[tokens, N]`` combined,
    ``tokens = topk_idx.shape[0]``.

    With ``grad_gate`` (f32, one per pool row) each row's value also travels to its route and the
    call returns ``(out, grad_topk_weights [tokens, topk])``, the backward's gate gradient.

    ``out[t]`` is the sum of ``act[p] @ w2[e(p)].T`` over the pool rows ``p`` of every route of
    this rank's token ``t``, on whichever rank holds them. No routing weight is applied here.
    ``w2`` is ``[experts, N, K]``, or ``[experts, K, N]`` with ``b_nn`` (``act[p] @ w2[e(p)]``).
    ``bias`` [experts, N] adds ``bias[e(p)] * bias_scale[p]`` to every pool row before the sum,
    ``bias_scale`` being the rows' f32 routing weight. ``tile_k`` 0 is 128 when it divides K, else 64.

    ``scatter`` stores every GEMM row straight into its route's combine row on the token's source
    rank, and the reduce sums all of a token's routes; without it the GEMM stages the rows locally
    and the combine workgroups push one sum per (token, rank), which waits for the token's last
    local expert."""
    _require_gfx1250(act.device)
    symm = get_symm_buffer_for_mega_moe()
    G, N, K = w2.shape
    if b_nn:
        K, N = N, K
        assert N >= tile_n, "an NN block backs its B read off into the row, so N >= tile_n"
    rows = act.shape[0]
    tile_k = tile_k or (128 if K % 128 == 0 else 64)
    assert act.dtype == w2.dtype == torch.bfloat16 and act.is_contiguous() and w2.is_contiguous()
    assert act.shape[1] == K and N == int(symm.hidden), "w2's output dim is the model hidden"
    assert rows == handle.pool_row_to_route.numel(), "act holds the rows dispatched to this rank"
    assert tile_m == BLOCK_M, "a GEMM tile is one Mega pool block"
    assert K % tile_k == 0 and K // tile_k >= NUM_BUFFERS, "K must be a whole number of >= 2 k-tiles"
    assert topk_idx.dim() == 2 and topk_idx.shape[1] == int(symm.num_topk)
    num_tokens = topk_idx.shape[0]
    # the reduce walks num_max_tokens_per_rank tokens; the rows past this call's read as unrouted
    max_tokens = int(symm.num_max_tokens_per_rank)
    topk_flat = torch.full((max_tokens, topk_idx.shape[1]), -1, dtype=torch.int64, device=act.device)
    topk_flat[:num_tokens] = topk_idx
    num_ranks = int(symm.num_ranks)
    assert int(symm.num_experts) // num_ranks == G, "w2 holds this rank's experts"
    num_cu = torch.cuda.get_device_properties(act.device).multi_processor_count
    # lead combine workgroups spin on GEMM tiles, so they must leave CUs for the GEMM to finish
    assert 0 <= num_lead_combine < num_cu
    num_tail_combine = num_tail_combine or num_cu - num_lead_combine
    out = torch.empty(int(symm.num_max_tokens_per_rank), N, dtype=torch.bfloat16, device=act.device)
    has_grad_gate = grad_gate is not None
    if has_grad_gate:
        assert grad_gate.dtype == torch.float32 and grad_gate.numel() == rows, (
            "one f32 gate grad per pool row"
        )
        grad_gate = grad_gate.contiguous()
        grad_topk_weights = torch.empty(int(symm.num_max_routes), dtype=torch.float32, device=act.device)
    has_bias = bias is not None
    if has_bias:
        assert bias.dtype == torch.bfloat16 and tuple(bias.shape) == (G, N), f"bias must be bf16 [{G}, {N}]"
        assert bias_scale is not None and bias_scale.dtype == torch.float32 and bias_scale.numel() == rows
        bias, bias_scale = bias.contiguous(), bias_scale.contiguous()
    # local stage of the L2 output, which only this rank's combine workgroups read; a scatter has none
    l2 = out if scatter else torch.empty(rows, N, dtype=torch.bfloat16, device=act.device)
    args = (
        _as_ptr(out),
        _as_ptr(act),
        _as_ptr(w2),
        _as_ptr(l2),
        _as_ptr(handle.tile_to_expert),
        _as_ptr(handle.num_tile_blocks),
        _as_ptr(handle.combine_recv_dst_rank),
        _as_ptr(handle.combine_recv_start_row),
        _as_ptr(handle.combine_recv_fold_count),
        _as_ptr(handle.pool_row_to_route),
        _as_ptr(handle.pool_row_to_recv_token),
        _as_ptr(handle.recv_token_to_pool_rows),
        _as_ptr(topk_flat),
        _as_ptr(symm.num_tokens_per_rank),
        _as_ptr(symm._combine_parity),
        _as_ptr(symm._combine_expected),
        _as_ptr(symm._reduce_expected),
        _as_ptr(_chunk_counter(act.device)),
        # traced out without a grad gate; any device tensor fills them
        _as_ptr(grad_gate if has_grad_gate else out),
        _as_ptr(grad_topk_weights if has_grad_gate else out),
        _as_ptr(bias if has_bias else out),
        _as_ptr(bias_scale if has_bias else out),
        symm.get_sym_buffer(),
        torch.cuda.current_stream(),
        # GEMM tiles of this call's pool (the prologue's num_tile_blocks), then the combine
        # workgroups: a worst-case grid would queue idle tiles ahead of the tail combine
        rows // BLOCK_M * ((N + tile_n - 1) // tile_n) + num_lead_combine + num_tail_combine,
        N,
        K,
        int(symm.num_max_pool_tokens) // BLOCK_M,
        (N + tile_n - 1) // tile_n,
        num_lead_combine,
        num_tail_combine,
        lead_gemm_percent,
        num_ranks,
        int(symm.rank),
        int(symm.num_experts),
        int(symm.num_max_tokens_per_rank),
        int(symm.num_topk),
        tile_m,
        tile_n,
        tile_k,
        m_warp,
        n_warp,
        has_grad_gate,
        b_nn,
        has_bias,
        scatter,
    )
    key = args[_CONSTEXPR0:]
    fn = _COMPILED.get(key)
    if fn is None:
        fn = flyc.compile(_launch_gemm_combine_nt, *args)
        if fn is not None:
            _COMPILED[key] = fn
    if fn is not None:
        fn(*args)
    if has_grad_gate:
        return out[:num_tokens], grad_topk_weights[: num_tokens * topk_idx.shape[1]].view(num_tokens, -1)
    return out[:num_tokens]
