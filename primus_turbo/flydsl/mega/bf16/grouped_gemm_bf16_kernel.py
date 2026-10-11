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

# MegaMoE-owned snapshot of the shared grouped BF16 variable-K tile as of f6d5ab68
# (last known-good before ed8d7af4 / #486). See gemm_bf16_kernel.py in this package.
#
"""FlyDSL bf16 variable-K GROUPED GEMM — the MoE wgrad operator.

Computes ``out[g] = a[rows_g].T @ b[rows_g]`` for G groups, where A is
[M_total, OUT_M] and B is [M_total, OUT_N] (groups concatenated along the
reduction dim), ``group_k_offsets`` [G+1] int64 gives each group's row start,
and ``masked_k`` [G] int64 gives the per-group VALID row count so the padded
tail is never read.

Grid is exactly ``G * (OUT_M/BLOCK_M) * ceil(OUT_N/BLOCK_N)`` tiles; each WG
maps its pid -> (group_idx, block_m, block_n) and reads m_start/m_end from the
two index tables on-device (no CPU sync). A/B are rebased per group with an
int64 base offset and span, so a worst-case pool cannot wrap int32 before
``make_bf16_buffer_tensor_rebased`` clamps the span into the 32-bit HW
num_records field.

Shares the dense kernel's LDS layout and primitives; see gemm_bf16_kernel.py
for the 4-buffer pipeline / barrier rationale (identical here, except the K
loop is chunked because K is a runtime value).
"""

import functools
from types import SimpleNamespace

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl.compiler.ast_rewriter import ASTRewriter
from flydsl.expr import arith, const_expr, range_constexpr, rocdl
from flydsl.expr.buffer_ops import create_buffer_resource
from flydsl.expr.primitive import get_iter as _get_iter
from flydsl.expr.primitive import ptrtoint as _ptrtoint
from flydsl.expr.typing import Vector as Vec
from flydsl.expr.utils.arith import ArithValue

from primus_turbo.flydsl.grouped_gemm.grouped_gemm_bf16_kernel import (
    _F16,
    _ab_ty,
    _grid_x,
    _load_i32,
    _load_i64_as_i32,
    _ptr_only_view,
    _tail_quad_conds,
)
from primus_turbo.flydsl.mega.bf16.gemm_bf16_kernel import _make_shared_storage
from primus_turbo.flydsl.mega.bf16.gemm_helper import (
    compute_global_swizzle_nn_bf16_wide_row_col,
    load_row_idx_to_lds,
    read_row_idx_from_lds,
)
from primus_turbo.flydsl.utils.gemm_helper import (
    BLOCK_K,
    G2SLoader,
    Mfma16x16x32,
    S2RLoaderTr16x32Bf16Wide,
    StoreCBf16,
    compute_global_swizzle_nn_bf16_wide,
    emit_for,
    emit_if_then,
    group_m_tile_decode,
    make_bf16_buffer_tensor_rebased,
    make_value_attrs,
    wait_barrier,
    wave_lane_with_rank,
    wave_rank_desc_stable,
    xcd_band_remap_pid,
)
from primus_turbo.flydsl.utils.prims import _i64

# Per-wave LDS ring of gather row indices: chunk c + 4 is filled while chunk c runs, after c - 4 is consumed.
LDS_ROW_IDX_CHUNKS = 8
NUM_LDS_ROW_IDX_ENTRIES = 8 * LDS_ROW_IDX_CHUNKS * 4 * 8


@ASTRewriter.transform
def grouped_gemm_bf16_variable_k_tile(
    A,
    B,
    C,
    group_idx,
    block_m,
    block_n,
    m_start,
    m_end,
    lds,
    out_m_rt,
    out_n_rt,
    *,
    G,
    OUT_M,
    OUT_N,
    BLOCK_M,
    BLOCK_N,
    persistent=False,
    ab_ty=fx.BFloat16,
    out_fp16=False,
    c_cache_modifier=0,
    trans_c=False,
    lds_chunk_stride=1152,
    mask_m=None,
    a_row_idx=None,
    b_row_idx=None,
    num_row_idx_entries=None,
    num_gathered_rows=None,
):
    CHUNK = 4
    WGRAD_WAVES = 8  # fixed 8 waves per block
    has_a_row_idx = a_row_idx is not None
    has_b_row_idx = b_row_idx is not None
    has_row_idx = has_a_row_idx or has_b_row_idx
    assert not (has_a_row_idx and has_b_row_idx), "at most one operand is gathered"
    assert NUM_LDS_ROW_IDX_ENTRIES == WGRAD_WAVES * LDS_ROW_IDX_CHUNKS * CHUNK * 8
    assert BLOCK_M >= 128 and BLOCK_N >= 64 and BLOCK_M % 128 == 0 and BLOCK_N % 64 == 0
    N_TILES_A = BLOCK_M // 128
    # A ragged OUT_M over-launches the last M block; a partitioned launch passes mask_m itself.
    MASK_M = (OUT_M % BLOCK_M != 0) if mask_m is None else mask_m
    MASK_N = OUT_N % BLOCK_N != 0
    LDS_BLOCK_M = BLOCK_M // 2
    LDS_BLOCK_N = BLOCK_N // 2
    N_LDS_STEPS_A = (BLOCK_M // 16) // WGRAD_WAVES
    N_LDS_STEPS_B = (BLOCK_N // 16) // WGRAD_WAVES
    N_WAVE_N = WGRAD_WAVES // 2

    lane_id = fx.thread_idx.x % 64
    wave_id = fx.thread_idx.x // 64
    wave_m = wave_id // N_WAVE_N
    wave_n = wave_id % N_WAVE_N

    group_tokens = m_end - m_start
    bf16_ir = fx.BFloat16.ir_type
    # base offset and per-group span (group_tokens * OUT * 2 bytes) can both exceed
    # int32 for a worst-case pool; compute in int64 so the span does not wrap before
    # make_bf16_buffer_tensor_rebased clamps it to the 32-bit HW num_records field.
    a_base_off = _i64(m_start) * fx.Int64(OUT_M * 2)
    b_base_off = _i64(m_start) * fx.Int64(OUT_N * 2)
    a_span = _i64(group_tokens) * _i64(out_m_rt) * fx.Int64(2)
    b_span = _i64(group_tokens) * _i64(out_n_rt) * fx.Int64(2)
    if const_expr(has_a_row_idx):
        a_base_off, a_span = fx.Int64(0), _i64(num_gathered_rows) * fx.Int64(OUT_M * 2)
    if const_expr(has_b_row_idx):
        b_base_off, b_span = fx.Int64(0), _i64(num_gathered_rows) * fx.Int64(OUT_N * 2)
    gA = make_bf16_buffer_tensor_rebased(A, bf16_ir, a_base_off, a_span)
    gB = make_bf16_buffer_tensor_rebased(B, bf16_ir, b_base_off, b_span)
    a_div = fx.logical_divide(gA, fx.make_layout(1, 1))
    b_div = fx.logical_divide(gB, fx.make_layout(1, 1))

    gl_off_a = compute_global_swizzle_nn_bf16_wide(lane_id, wave_id, OUT_M, N_LDS_STEPS_A)
    gl_off_b = compute_global_swizzle_nn_bf16_wide(lane_id, wave_id, OUT_N, N_LDS_STEPS_B)

    a0_off = block_m * BLOCK_M
    a1_off = a0_off + LDS_BLOCK_M
    b0_off = block_n * BLOCK_N
    b1_off = b0_off + LDS_BLOCK_N
    a_k_step = fx.Int32(BLOCK_K) * out_m_rt
    b_k_step = fx.Int32(BLOCK_K) * out_n_rt

    NTA16 = N_TILES_A * 2
    NTB16 = (BLOCK_N // 16) // (2 * N_WAVE_N)
    N_ACCUMS16 = NTA16 * NTB16
    mfma = Mfma16x16x32(NTA16, NTB16, ab_ty)
    a_s2r = S2RLoaderTr16x32Bf16Wide(wave_m, NTA16, chunk_stride=lds_chunk_stride)
    b_s2r = S2RLoaderTr16x32Bf16Wide(wave_n, NTB16, chunk_stride=lds_chunk_stride)
    ACC_VEC_N = 4
    N_ACCUMS_EFF = N_ACCUMS16
    a_g2s = G2SLoader(a_div, gl_off_a, N_LDS_STEPS_A, bf16_ir, wave_id, chunk_stride=lds_chunk_stride)
    b_g2s = G2SLoader(b_div, gl_off_b, N_LDS_STEPS_B, bf16_ir, wave_id, chunk_stride=lds_chunk_stride)

    if const_expr(has_row_idx):
        gathered_steps = N_LDS_STEPS_A if has_a_row_idx else N_LDS_STEPS_B
        gathered_cols = [
            col for _, col in compute_global_swizzle_nn_bf16_wide_row_col(lane_id, wave_id, gathered_steps)
        ]
        gathered_row_stride = OUT_M if has_a_row_idx else OUT_N
        row_idx = a_row_idx if has_a_row_idx else b_row_idx
        row_idx_resource = create_buffer_resource(
            row_idx, num_records_bytes=num_row_idx_entries * fx.Int32(4)
        )
        lds_row_idx_base = fx.Int32(fx.ptrtoint(lds.lds_row_idx.ptr)) + wave_id * (
            LDS_ROW_IDX_CHUNKS * CHUNK * 32
        )

    def _fill_row_idx(chunk):
        """Copy this wave's row indices of chunks ``chunk`` and ``chunk + 1`` into its LDS ring."""
        # chunk is even, so the 2-chunk write never crosses the ring end into the next wave's ring.
        entry = m_start + (chunk * CHUNK + lane_id // 8) * BLOCK_K + wave_id * 8 + lane_id % 8
        load_row_idx_to_lds(
            row_idx_resource, lds_row_idx_base + (chunk % LDS_ROW_IDX_CHUNKS) * (CHUNK * 32), entry * 4
        )

    def _gathered_offsets(k):
        ring_byte_offset = (k % (LDS_ROW_IDX_CHUNKS * CHUNK)) * 32 + ((lane_id // 2) % 8) * 4
        row = read_row_idx_from_lds(lds_row_idx_base + ring_byte_offset) * gathered_row_stride
        return [row + col for col in gathered_cols]

    a_feed = (a_g2s, has_a_row_idx, a_k_step)
    b_feed = (b_g2s, has_b_row_idx, b_k_step)

    def _load(feed, dst, col_offset, k, gathered_offsets):
        g2s, is_gathered, k_step = feed
        if const_expr(is_gathered):
            g2s.gl_offsets = gathered_offsets
            g2s.load(dst, col_offset)
        else:
            g2s.load(dst, col_offset + k * k_step)

    out_ty = fx.Float16 if out_fp16 else fx.BFloat16
    if const_expr(trans_c):
        store_c = StoreCBf16(C, G * OUT_N, OUT_M, out_ty, cache_modifier=c_cache_modifier)
    else:
        store_c = StoreCBf16(C, G * OUT_M, OUT_N, out_ty, cache_modifier=c_cache_modifier)

    acc00 = [fx.make_rmem_tensor(fx.make_layout(ACC_VEC_N, 1), fx.Float32) for _ in range(N_ACCUMS_EFF)]
    acc01 = [fx.make_rmem_tensor(fx.make_layout(ACC_VEC_N, 1), fx.Float32) for _ in range(N_ACCUMS_EFF)]
    acc10 = [fx.make_rmem_tensor(fx.make_layout(ACC_VEC_N, 1), fx.Float32) for _ in range(N_ACCUMS_EFF)]
    acc11 = [fx.make_rmem_tensor(fx.make_layout(ACC_VEC_N, 1), fx.Float32) for _ in range(N_ACCUMS_EFF)]
    for quad in (acc00, acc01, acc10, acc11):
        for reg in quad:
            fx.memref_store_vec(mfma.zero_value, reg)

    # Predicate the over-launched quadrants behind one wave-uniform branch, not a second tile class.
    quad_live = _tail_quad_conds(
        block_m * BLOCK_M + wave_m * (NTA16 * 16),
        block_n * BLOCK_N + wave_n * (NTB16 * 16),
        OUT_M,
        OUT_N,
        LDS_BLOCK_M,
        LDS_BLOCK_N,
        MASK_M,
        MASK_N,
    )

    def _mma_quad(acc, a, b, cond):
        """One accumulator quadrant, skipped whole when its output rows/columns are masked."""

        def _do():
            c = [Vec(fx.memref_load_vec(r)) for r in acc]
            c = mfma.call(a, b, c)
            for idx in range_constexpr(len(acc)):
                fx.memref_store_vec(c[idx], acc[idx])

        if const_expr(cond is None):
            _do()
        else:
            emit_if_then(cond, _do)

    # An empty expert only has to store zero accumulators, so the whole fetch pipeline is skipped.
    def _prologue():
        if const_expr(has_row_idx):
            _fill_row_idx(0)
            _fill_row_idx(2)
        wait_barrier(0)
        offsets0 = _gathered_offsets(0) if const_expr(has_row_idx) else None
        offsets1 = _gathered_offsets(1) if const_expr(has_row_idx) else None
        _load(b_feed, lds.B_lds_cur_0, b0_off, 0, offsets0)
        _load(a_feed, lds.A_lds_cur_0, a0_off, 0, offsets0)
        _load(b_feed, lds.B_lds_cur_1, b1_off, 0, offsets0)
        _load(a_feed, lds.A_lds_cur_1, a1_off, 0, offsets0)
        # Divergent only holds for one tile per WG; a persistent loop needs every wave to
        # stop before the next tile's g2s reuses this LDS. Cf. dense_mma_pipeline_bf16.
        if const_expr(persistent):
            rocdl.s_barrier()
        elif wave_m == 1:
            rocdl.s_barrier()
        wait_barrier(N_LDS_STEPS_A + N_LDS_STEPS_B)
        _load(b_feed, lds.B_lds_next_0, b0_off, 1, offsets1)
        _load(a_feed, lds.A_lds_next_0, a0_off, 1, offsets1)
        _load(b_feed, lds.B_lds_next_1, b1_off, 1, offsets1)
        wait_barrier(N_LDS_STEPS_A + 2 * N_LDS_STEPS_B)

    emit_if_then(group_tokens > 0, _prologue)

    k_iters = (group_tokens + (BLOCK_K - 1)) // BLOCK_K
    n_chunks = (k_iters + (CHUNK - 1)) // CHUNK

    # nested to isolate Python-level buffer rotation from the runtime chunk loop
    def _chunk(chunk_iv, live):
        chunk_idx = ArithValue(chunk_iv)
        a_cur0, a_cur1 = lds.A_lds_cur_0, lds.A_lds_cur_1
        a_next0, a_next1 = lds.A_lds_next_0, lds.A_lds_next_1
        b_cur0, b_cur1 = lds.B_lds_cur_0, lds.B_lds_cur_1
        b_next0, b_next1 = lds.B_lds_next_0, lds.B_lds_next_1
        offsets_after_next = None
        for j in range_constexpr(CHUNK):
            k = chunk_idx * CHUNK + j
            if const_expr(has_row_idx and j == 0):
                # Issued before this step's G2S, so the step's vmcnt waits retire it no later than them.
                emit_if_then(chunk_idx % 2 == 0, lambda: _fill_row_idx(chunk_idx + 4))
            next_offsets = None
            if const_expr(has_a_row_idx):
                next_offsets = _gathered_offsets(k + 1) if const_expr(j == 0) else offsets_after_next
            offsets_after_next = _gathered_offsets(k + 2) if const_expr(has_row_idx) else None
            # 4-buffer pipelined body: interleave s2r/g2s with the 4 mfma quadrants
            b0 = b_s2r.load(b_cur0)
            a0 = a_s2r.load(a_cur0)
            _load(a_feed, a_next1, a1_off, k + 1, next_offsets)
            rocdl.s_barrier()
            rocdl.sched_barrier(0)
            _mma_quad(acc00, a0, b0, live[0, 0])
            rocdl.sched_barrier(0)
            rocdl.s_barrier()
            b1 = b_s2r.load(b_cur1)
            _load(b_feed, b_cur0, b0_off, k + 2, offsets_after_next)
            rocdl.s_barrier()
            rocdl.sched_barrier(0)
            _mma_quad(acc01, a0, b1, live[0, 1])
            rocdl.sched_barrier(0)
            rocdl.s_barrier()
            a1 = a_s2r.load(a_cur1)
            rocdl.s_barrier()
            rocdl.sched_barrier(0)
            _mma_quad(acc10, a1, b0, live[1, 0])
            rocdl.sched_barrier(0)
            rocdl.s_barrier()
            # Both k+2 refills sit in the last phase, most-urgent first; issuing earlier only ages the line.
            _load(a_feed, a_cur0, a0_off, k + 2, offsets_after_next)
            _load(b_feed, b_cur1, b1_off, k + 2, offsets_after_next)
            wait_barrier(2 * N_LDS_STEPS_A + N_LDS_STEPS_B)
            rocdl.sched_barrier(0)
            _mma_quad(acc11, a1, b1, live[1, 1])
            rocdl.sched_barrier(0)
            rocdl.s_barrier()
            a_cur0, a_next0 = a_next0, a_cur0
            a_cur1, a_next1 = a_next1, a_cur1
            b_cur0, b_next0 = b_next0, b_cur0
            b_cur1, b_next1 = b_next1, b_cur1

    # Only ragged boundary tiles branch, and on a workgroup-uniform test so barriers stay matched.
    all_live = {key: None for key in quad_live}
    interior, boundary = None, None
    if const_expr(MASK_M):
        interior = (block_m + 1) * BLOCK_M <= fx.Int32(OUT_M)
        boundary = (block_m + 1) * BLOCK_M > fx.Int32(OUT_M)
    if const_expr(MASK_N):
        n_in = (block_n + 1) * BLOCK_N <= fx.Int32(OUT_N)
        n_bd = (block_n + 1) * BLOCK_N > fx.Int32(OUT_N)
        interior = n_in if interior is None else arith.andi(interior, n_in)
        boundary = n_bd if boundary is None else arith.ori(boundary, n_bd)

    def _loop(live):
        emit_for(n_chunks, lambda iv: _chunk(iv, live))

    if const_expr(interior is None):
        _loop(all_live)
    else:
        emit_if_then(interior, lambda: _loop(all_live))
        emit_if_then(boundary, lambda: _loop(quad_live))

    c00 = [Vec(fx.memref_load_vec(reg)) for reg in acc00]
    c01 = [Vec(fx.memref_load_vec(reg)) for reg in acc01]
    c10 = [Vec(fx.memref_load_vec(reg)) for reg in acc10]
    c11 = [Vec(fx.memref_load_vec(reg)) for reg in acc11]

    if const_expr(trans_c):
        local_m = block_m * BLOCK_M + wave_m * (NTA16 * 16)
        local_n = block_n * BLOCK_N + wave_n * (NTB16 * 16)
        for cfrag, q_row, q_col in (
            (c00, local_m, local_n),
            (c01, local_m, local_n + LDS_BLOCK_N),
            (c10, local_m + LDS_BLOCK_M, local_n),
            (c11, local_m + LDS_BLOCK_M, local_n + LDS_BLOCK_N),
        ):
            for i in range_constexpr(NTA16):
                for j in range_constexpr(NTB16):
                    store_c.store_trans16(
                        [cfrag[i * NTB16 + j]],
                        group_idx,
                        q_row + i * 16,
                        q_col + j * 16,
                        OUT_M,
                        OUT_N,
                        mask_m=MASK_M,
                    )
    else:
        base_row = group_idx * OUT_M + block_m * BLOCK_M + wave_m * (NTA16 * 16)
        row_bound = (group_idx + 1) * OUT_M
        base_col = block_n * BLOCK_N + wave_n * (NTB16 * 16)
        # Both column halves share the band SRD and every row address; only the store immediate differs.
        for cfrags, q_row in (((c00, c01), base_row), ((c10, c11), base_row + LDS_BLOCK_M)):
            store_c.store_band16(cfrags, q_row, base_col, LDS_BLOCK_N, NTA16, NTB16, row_bound, mask_n=MASK_N)


@functools.lru_cache(maxsize=64)
def _compile_grouped_bf16_wgrad(
    OUT_M,
    OUT_N,
    G,
    BLOCK_M=256,
    BLOCK_N=256,
    num_xcd=8,
    waves_per_eu=2,
    agpr_alloc=0,
    ab_ty=fx.BFloat16,
    out_fp16=False,
    trans_c=False,
    # One padded chunk splits the tr16 reader's four lane groups off a single bank half: 128 mod 256.
    lds_chunk_stride=1152,
    group_m=1,
    xcd_band=24,
    cap_cu=0,
    has_a_row_idx=False,
    has_b_row_idx=False,
):
    # See _compile_grouped_bf16_nt: cap_cu = 0 keeps the tuned one-tile-per-WG launch.
    persistent = cap_cu > 0
    N_BLOCKS_M = (OUT_M + BLOCK_M - 1) // BLOCK_M
    N_BLOCKS_N = (OUT_N + BLOCK_N - 1) // BLOCK_N
    TILES_PER_GROUP = N_BLOCKS_M * N_BLOCKS_N
    TOTAL = G * TILES_PER_GROUP
    # A ragged OUT_M over-launches the last M block; its tail rows are dropped at store time.
    MASK_M = N_BLOCKS_M * BLOCK_M > OUT_M
    has_row_idx = has_a_row_idx or has_b_row_idx
    SharedStorage = _make_shared_storage(
        BLOCK_M,
        BLOCK_N,
        chunk_stride=lds_chunk_stride,
        num_lds_row_idx_entries=NUM_LDS_ROW_IDX_ENTRIES if has_row_idx else 0,
    )

    @flyc.kernel(known_block_size=[512, 1, 1])
    def kernel_grouped_variable_k(
        A: fx.Tensor,
        B: fx.Tensor,
        C: fx.Tensor,
        group_k_offsets: fx.Tensor,
        masked_k: fx.Tensor,
        out_m_rt: fx.Int32,
        out_n_rt: fx.Int32,
        # Empty when dense; else a_row_idx or b_row_idx plus num_row_idx_entries and num_gathered_rows.
        row_idx_args,
    ):
        _ = str(fx.thread_idx.x)
        go_base = fx.Int64(_ptrtoint(_get_iter(group_k_offsets)))
        gk_base = fx.Int64(_ptrtoint(_get_iter(masked_k)))
        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        # Dispatch order ranked in the prologue by a lane-resident descending rank, not a host argsort.
        # Tile-independent, so it is hoisted above the persistent loop and ranked once.
        if const_expr(G <= 64):  # one wave's lanes hold the rank; more groups keep their natural order
            lane = fx.Int32(fx.thread_idx.x) % fx.Int32(64)
            in_g = lane < fx.Int32(G)
            k_lane = _load_i32(gk_base, arith.select(in_g, lane, fx.Int32(0)) * fx.Int32(2))
            order_rank = wave_rank_desc_stable(arith.select(in_g, k_lane, fx.Int32(-1)), lane, G)

        # Free function for the same ast-rewriter reason as the NT kernel.
        def _do_tile(pid):
            # Band-cyclic XCD assignment: runs short enough to stay inside one expert, so skew spreads.
            tile = xcd_band_remap_pid(pid, TOTAL, num_xcd, xcd_band)
            group_idx = tile // TILES_PER_GROUP
            if const_expr(G <= 64):
                group_idx = wave_lane_with_rank(order_rank, group_idx)
            local_tile = tile % TILES_PER_GROUP
            if const_expr(trans_c):
                block_n, block_m = group_m_tile_decode(local_tile, N_BLOCKS_N, N_BLOCKS_M, group_m)
            else:
                block_m, block_n = group_m_tile_decode(local_tile, N_BLOCKS_M, N_BLOCKS_N, group_m)
            m_start = _load_i64_as_i32(go_base, group_idx)
            m_end = m_start + _load_i64_as_i32(gk_base, group_idx)
            grouped_gemm_bf16_variable_k_tile(
                A,
                B,
                C,
                group_idx,
                block_m,
                block_n,
                m_start,
                m_end,
                lds,
                out_m_rt,
                out_n_rt,
                G=G,
                OUT_M=OUT_M,
                OUT_N=OUT_N,
                BLOCK_M=BLOCK_M,
                BLOCK_N=BLOCK_N,
                ab_ty=ab_ty,
                out_fp16=out_fp16,
                trans_c=trans_c,
                lds_chunk_stride=lds_chunk_stride,
                mask_m=MASK_M,
                persistent=persistent,
                **vars(row_idx_args),
            )

        if const_expr(persistent):
            for t in range(fx.block_idx.x, TOTAL, fx.grid_dim.x):
                _do_tile(t)
        else:
            _do_tile(fx.block_idx.x)

    @flyc.jit
    def launch_grouped_variable_k(
        A,
        B,
        C,
        group_k_offsets,
        masked_k,
        out_m_rt: fx.Int32,
        out_n_rt: fx.Int32,
        row_idx,
        num_row_idx_entries: fx.Int32,
        num_gathered_rows: fx.Int32,
        stream: fx.Stream,
    ):
        grid_x = fx.Int32(_grid_x(TOTAL, cap_cu))
        # An empty namespace adds no kernel argument, so the dense kernel keeps its signature.
        row_idx_args = SimpleNamespace()
        if const_expr(has_a_row_idx):
            row_idx_args = SimpleNamespace(a_row_idx=row_idx)
        if const_expr(has_b_row_idx):
            row_idx_args = SimpleNamespace(b_row_idx=row_idx)
        if const_expr(has_row_idx):
            row_idx_args.num_row_idx_entries = num_row_idx_entries
            row_idx_args.num_gathered_rows = num_gathered_rows
        kernel_grouped_variable_k(
            A,
            B,
            C,
            group_k_offsets,
            masked_k,
            out_m_rt,
            out_n_rt,
            row_idx_args,
            value_attrs=make_value_attrs(waves_per_eu, agpr_alloc, "512,512"),
        ).launch(grid=(grid_x, 1, 1), block=(512, 1, 1), stream=stream)

    return launch_grouped_variable_k


_COMPILED_GROUPED_GEMM_CACHE = {}


def grouped_gemm_bf16_variable_k_flydsl_kernel(
    a: torch.Tensor,
    b: torch.Tensor,
    group_k_offsets: torch.Tensor,
    masked_k: torch.Tensor = None,
    out_dtype: torch.dtype = torch.bfloat16,
    BLOCK_M: int = 256,
    BLOCK_N: int = 256,
    # An XCD remap reorders tiles across groups, so a skewed token count skews the per-XCD cost.
    num_xcd: int = 8,
    # Super-tile width in M blocks: co-resident workgroups share an A column slice.  Retune with the tile.
    group_m: int = 4,
    # Band-cyclic run length in tiles; it tracks co-residency rather than grid divisibility.
    xcd_band: int = 32,
    # CUs this launch may occupy; None/0 = the whole device, which keeps the tuned
    # one-tile-per-WG launch. A real budget switches to a capped persistent grid.
    cap_cu: int = 0,
    trans_c: bool = False,
    # K row r of a (or b) is read from row a_row_idx[r] (b_row_idx[r]); rows outside a (b) read 0.
    a_row_idx: torch.Tensor = None,
    b_row_idx: torch.Tensor = None,
) -> torch.Tensor:
    """Variable-K grouped wgrad: out[g]=a[g_rows].T@b[g_rows], K=[offsets[g],offsets[g]+masked_k[g])."""
    assert a_row_idx is None or b_row_idx is None, "at most one operand is gathered"
    assert a.dim() == 2 and b.dim() == 2
    assert a_row_idx is not None or b_row_idx is not None or a.shape[0] == b.shape[0]
    assert a.dtype in _F16 and b.dtype == a.dtype, f"16-bit float operands only, got {a.dtype}/{b.dtype}"
    OUT_M = a.shape[1]
    OUT_N = b.shape[1]
    G = group_k_offsets.numel() - 1
    out_fp16 = out_dtype == torch.float16
    out_shape = (G, OUT_N, OUT_M) if trans_c else (G, OUT_M, OUT_N)
    out = torch.empty(out_shape, device=a.device, dtype=out_dtype)
    # index tables loaded as i64 in-kernel
    offsets_i64 = group_k_offsets if group_k_offsets.dtype == torch.int64 else group_k_offsets.to(torch.int64)
    # per-expert valid K length; default = padded span
    if masked_k is None:
        masked_k_i64 = (offsets_i64[1:] - offsets_i64[:-1]).contiguous()
    else:
        assert masked_k.numel() == G, f"masked_k len {masked_k.numel()} != G {G}"
        masked_k_i64 = (masked_k if masked_k.dtype == torch.int64 else masked_k.to(torch.int64)).contiguous()
    row_idx, gathered_operand = (a_row_idx, a) if a_row_idx is not None else (b_row_idx, b)
    if row_idx is None:
        row_idx, num_gathered_rows = masked_k_i64, 0
    else:
        assert row_idx.dtype == torch.int32, f"row index tables are int32, got {row_idx.dtype}"
        num_gathered_rows = gathered_operand.shape[0]
        bytes_with_sentinel = (num_gathered_rows + 1) * gathered_operand.shape[1] * 2
        assert bytes_with_sentinel <= 2**31 - 1, "gathered operand exceeds the 32-bit buffer offset"
    args = (
        _ptr_only_view(a),
        _ptr_only_view(b),
        flyc.from_torch_tensor(out),
        offsets_i64,
        masked_k_i64,
        OUT_M,
        OUT_N,
        _ptr_only_view(row_idx),
        row_idx.numel(),
        num_gathered_rows,
        torch.cuda.current_stream(),
    )
    has_a_row_idx, has_b_row_idx = a_row_idx is not None, b_row_idx is not None
    key = (OUT_M, OUT_N, G, BLOCK_M, BLOCK_N, num_xcd, group_m, xcd_band, a.dtype, out_fp16, trans_c, cap_cu)
    key += (has_a_row_idx, has_b_row_idx)
    compiled = _COMPILED_GROUPED_GEMM_CACHE.get(key)
    if compiled is None:
        launch = _compile_grouped_bf16_wgrad(
            OUT_M,
            OUT_N,
            G,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            num_xcd=num_xcd,
            ab_ty=_ab_ty(a.dtype),
            out_fp16=out_fp16,
            trans_c=trans_c,
            group_m=group_m,
            xcd_band=xcd_band,
            cap_cu=cap_cu,
            has_a_row_idx=has_a_row_idx,
            has_b_row_idx=has_b_row_idx,
        )
        compiled = flyc.compile(launch, *args)
        _COMPILED_GROUPED_GEMM_CACHE[key] = compiled
    compiled(*args)
    return out
