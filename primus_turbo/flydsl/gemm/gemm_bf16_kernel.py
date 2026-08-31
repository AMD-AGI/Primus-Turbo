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

"""Primus-Turbo dense BF16 GEMM kernel (FlyDSL)."""

import flydsl.expr as fx
from flydsl._mlir.dialects import llvm as _llvm
from flydsl.compiler.ast_rewriter import ASTRewriter
from flydsl.expr import arith, const_expr, range_constexpr, rocdl
from flydsl.expr.typing import Vector as Vec

from primus_turbo.flydsl.utils.gemm_helper import (
    BLOCK_K,
    G2SLoader,
    Mfma16x16x32,
    S2RLoader16x16Bf16,
    S2RLoaderTr16x32Bf16Wide,
    StoreCBf16,
    compute_global_swizzle_bf16,
    compute_global_swizzle_nn_bf16_wide,
    emit_if_then,
    make_fp16_bf16_buffer_tensor,
    wait_barrier,
    xcd_remap_pid,
)
from primus_turbo.flydsl.utils.prims import ceildiv

# isort: on


def _make_shared_storage(BLOCK_M, BLOCK_N, chunk_stride=1024):
    a_lds_size = (BLOCK_M // 16) * chunk_stride // 2
    b_lds_size = (BLOCK_N // 16) * chunk_stride // 2

    @fx.struct
    class SharedStorage:
        A_lds_cur_0: fx.Array[fx.BFloat16, a_lds_size, 16]
        A_lds_cur_1: fx.Array[fx.BFloat16, a_lds_size, 16]
        A_lds_next_0: fx.Array[fx.BFloat16, a_lds_size, 16]
        A_lds_next_1: fx.Array[fx.BFloat16, a_lds_size, 16]
        B_lds_cur_0: fx.Array[fx.BFloat16, b_lds_size, 16]
        B_lds_cur_1: fx.Array[fx.BFloat16, b_lds_size, 16]
        B_lds_next_0: fx.Array[fx.BFloat16, b_lds_size, 16]
        B_lds_next_1: fx.Array[fx.BFloat16, b_lds_size, 16]

    return SharedStorage


def _quad_col_conds(base_col, lds_block_n, c_n):
    """Liveness of each accumulator quadrant's column run, keyed (m half, n half).
    A run starting past the output extent is masked away at store time, so skip its MFMA chain.
    Module level: inside the pipeline body a plain ``if`` becomes device control flow."""
    live0 = base_col < c_n
    live1 = base_col + lds_block_n < c_n
    return {(0, 0): live0, (0, 1): live1, (1, 0): live0, (1, 1): live1}


@ASTRewriter.transform
def dense_mma_pipeline_bf16(
    lds,
    a_g2s,
    b_g2s,
    a_s2r,
    b_s2r,
    mfma,
    store_c,
    A0_gl_offset,
    A1_gl_offset,
    B0_gl_offset,
    B1_gl_offset,
    a_k_step,
    b_k_step,
    block_m,
    block_n,
    wave_m,
    wave_n,
    K,
    BLOCK_M,
    BLOCK_N,
    nt_vmcnt,
    pair_cols=False,
    pair_tiles=False,
    quad_conds=None,
    half_n=False,
    n_tiles_a=None,
    n_tiles_b=None,
    wave_hi=None,
    col_safe=False,
    persistent=False,
):
    """Shared 4-quadrant pipelined MMA loop and store epilogue for the fixed-K bf16 tile.
    Keyword flags select the feed and epilogue variants a ragged N needs; each is explained at
    the site that consumes it.  Every caller feeds ``Mfma16x16x32``."""
    MFMA_MN = 16
    K_ITERS = K // BLOCK_K
    assert K_ITERS >= 2, f"K_ITERS={K_ITERS} too small; need K >= {2 * BLOCK_K}"
    N_TILES_A = BLOCK_M // 128 if n_tiles_a is None else n_tiles_a
    N_TILES_B = BLOCK_N // 256 if n_tiles_b is None else n_tiles_b
    WAVE_HI = wave_m if wave_hi is None else wave_hi
    N_ACCUMS = N_TILES_A * N_TILES_B
    LDS_BLOCK_M = BLOCK_M // 2
    LDS_BLOCK_N = BLOCK_N // 2
    # Drain counts are stated in loads still in flight, not derived from the block shape.
    N_LDS_STEPS_A = a_g2s.n_load_steps
    N_LDS_STEPS_B = b_g2s.n_load_steps

    a_cur0 = lds.A_lds_cur_0
    a_cur1 = lds.A_lds_cur_1
    a_next0 = lds.A_lds_next_0
    a_next1 = lds.A_lds_next_1
    b_cur0 = lds.B_lds_cur_0
    b_cur1 = lds.B_lds_cur_1
    b_next0 = lds.B_lds_next_0
    b_next1 = lds.B_lds_next_1

    PRED = quad_conds is not None
    c00_frag = [mfma.zero_value] * N_ACCUMS
    c01_frag = [mfma.zero_value] * N_ACCUMS
    c10_frag = [mfma.zero_value] * N_ACCUMS
    c11_frag = [mfma.zero_value] * N_ACCUMS
    if const_expr(PRED):
        # Memrefs, not SSA: a predicated accumulator would phi out of every branch at register cost.
        acc = [
            [fx.make_rmem_tensor(fx.make_layout(mfma.acc_len, 1), fx.Float32) for _ in range(N_ACCUMS)]
            for _ in range_constexpr(4)
        ]
        for regs in acc:
            for reg in regs:
                fx.memref_store_vec(mfma.zero_value, reg)
        conds = [quad_conds[0, 0], quad_conds[0, 1], quad_conds[1, 0], quad_conds[1, 1]]

    def _mma(q, frag, a, b):
        """One accumulator quadrant, skipped whole when its output columns are masked."""
        if const_expr(not PRED):
            return mfma.call(a, b, frag)

        def _do():
            c = [Vec(fx.memref_load_vec(reg)) for reg in acc[q]]
            c = mfma.call(a, b, c)
            for t in range_constexpr(N_ACCUMS):
                fx.memref_store_vec(c[t], acc[q][t])

        emit_if_then(conds[q], _do)
        return frag

    B1_STEPS = N_LDS_STEPS_B
    LOOP_DRAIN = 2 * N_LDS_STEPS_A + N_LDS_STEPS_B
    ITER_DRAIN = nt_vmcnt
    if const_expr(half_n):
        B1_STEPS = 0
        LOOP_DRAIN = N_LDS_STEPS_A + N_LDS_STEPS_B
        # ``nt_vmcnt`` counts the full body's issue stream, so a narrowed feed retires a different load.
        ITER_DRAIN = N_LDS_STEPS_A

    b_g2s.load(b_cur0, B0_gl_offset + 0 * b_k_step)
    a_g2s.load(a_cur0, A0_gl_offset + 0 * a_k_step)
    if const_expr(not half_n):
        b_g2s.load(b_cur1, B1_gl_offset + 0 * b_k_step)
    a_g2s.load(a_cur1, A1_gl_offset + 0 * a_k_step)

    # One tile per WG: only the high half has to wait here, and the divergence is harmless
    # because the WG ends right after. Inside a persistent tile loop it is not -- the next
    # tile's g2s would overrun LDS the other waves are still reading -- so every wave stops.
    if const_expr(persistent):
        rocdl.s_barrier()
    elif WAVE_HI == 1:
        rocdl.s_barrier()
    wait_barrier(N_LDS_STEPS_A + B1_STEPS)

    b_g2s.load(b_next0, B0_gl_offset + 1 * b_k_step)
    a_g2s.load(a_next0, A0_gl_offset + 1 * a_k_step)
    if const_expr(not half_n):
        b_g2s.load(b_next1, B1_gl_offset + 1 * b_k_step)

    wait_barrier(N_LDS_STEPS_A + N_LDS_STEPS_B + B1_STEPS)

    for k in range_constexpr(K_ITERS - 2):
        b0_frag = b_s2r.load(b_cur0)
        a0_frag = a_s2r.load(a_cur0)
        a_g2s.load(a_next1, A1_gl_offset + (k + 1) * a_k_step)
        rocdl.s_barrier()

        rocdl.sched_barrier(0)
        c00_frag = _mma(0, c00_frag, a0_frag, b0_frag)
        rocdl.sched_barrier(0)
        rocdl.s_barrier()

        if const_expr(not half_n):
            b1_frag = b_s2r.load(b_cur1)
        b_g2s.load(b_cur0, B0_gl_offset + (k + 2) * b_k_step)
        if const_expr(not half_n):
            rocdl.s_barrier()

        if const_expr(not half_n):
            rocdl.sched_barrier(0)
            c01_frag = _mma(1, c01_frag, a0_frag, b1_frag)
            rocdl.sched_barrier(0)
            rocdl.s_barrier()

        a1_frag = a_s2r.load(a_cur1)
        a_g2s.load(a_cur0, A0_gl_offset + (k + 2) * a_k_step)
        rocdl.s_barrier()

        rocdl.sched_barrier(0)
        c10_frag = _mma(2, c10_frag, a1_frag, b0_frag)
        rocdl.sched_barrier(0)
        if const_expr(not half_n):
            rocdl.s_barrier()
            b_g2s.load(b_cur1, B1_gl_offset + (k + 2) * b_k_step)
        wait_barrier(LOOP_DRAIN)

        if const_expr(not half_n):
            rocdl.sched_barrier(0)
            c11_frag = _mma(3, c11_frag, a1_frag, b1_frag)
            rocdl.sched_barrier(0)
            rocdl.s_barrier()

        if const_expr(ITER_DRAIN >= 0):
            _llvm.inline_asm(
                res=None,
                operands_=[],
                asm_string=f"s_waitcnt vmcnt({ITER_DRAIN})",
                constraints="",
                has_side_effects=True,
            )
        a_cur0, a_next0 = a_next0, a_cur0
        a_cur1, a_next1 = a_next1, a_cur1
        b_cur0, b_next0 = b_next0, b_cur0
        b_cur1, b_next1 = b_next1, b_cur1

    k = K_ITERS - 2
    b0_frag = b_s2r.load(b_cur0)
    a0_frag = a_s2r.load(a_cur0)
    rocdl.s_barrier()
    rocdl.sched_barrier(0)
    c00_frag = _mma(0, c00_frag, a0_frag, b0_frag)
    rocdl.sched_barrier(0)
    rocdl.s_barrier()

    if const_expr(not half_n):
        b1_frag = b_s2r.load(b_cur1)
        rocdl.s_barrier()
        rocdl.sched_barrier(0)
        c01_frag = _mma(1, c01_frag, a0_frag, b1_frag)
        rocdl.sched_barrier(0)
        rocdl.s_barrier()

    a1_frag = a_s2r.load(a_cur1)
    rocdl.s_barrier()
    rocdl.sched_barrier(0)
    c10_frag = _mma(2, c10_frag, a1_frag, b0_frag)
    rocdl.sched_barrier(0)
    rocdl.s_barrier()

    b0_frag = b_s2r.load(b_next0)
    a_g2s.load(a_next1, A1_gl_offset + (k + 1) * a_k_step)
    if const_expr(not half_n):
        rocdl.s_barrier()
        rocdl.sched_barrier(0)
        c11_frag = _mma(3, c11_frag, a1_frag, b1_frag)
        rocdl.sched_barrier(0)
        rocdl.s_barrier()

    a_cur0, a_next0 = a_next0, a_cur0
    a_cur1, a_next1 = a_next1, a_cur1
    b_cur0, b_next0 = b_next0, b_cur0
    b_cur1, b_next1 = b_next1, b_cur1

    # Drain before the read: the tail is the first consumer of the last iteration's refill.
    wait_barrier(0)
    a0_frag = a_s2r.load(a_cur0)
    rocdl.sched_barrier(0)
    c00_frag = _mma(0, c00_frag, a0_frag, b0_frag)
    rocdl.sched_barrier(0)
    rocdl.s_barrier()

    if const_expr(not half_n):
        b1_frag = b_s2r.load(b_cur1)
        rocdl.s_barrier()
        rocdl.sched_barrier(0)
        c01_frag = _mma(1, c01_frag, a0_frag, b1_frag)
        rocdl.sched_barrier(0)
        rocdl.s_barrier()

    wave_n_offset = wave_n * (N_TILES_B * MFMA_MN)
    wave_m_offset = wave_m * (N_TILES_A * MFMA_MN)
    base_row = block_m * BLOCK_M + wave_m_offset
    base_col = block_n * BLOCK_N + wave_n_offset

    if const_expr(PRED):
        c00_frag = [Vec(fx.memref_load_vec(reg)) for reg in acc[0]]
        # Half 1 never accumulated, so its quadrants keep the zero fragments the store masks away.
        if const_expr(not half_n):
            c01_frag = [Vec(fx.memref_load_vec(reg)) for reg in acc[1]]
    pair_col = block_n * BLOCK_N + wave_n_offset * 2

    def _store_row(frag_even, frag_odd, row):
        """One m half's quadrants, through the band writer that matches the atom's edge.
        ``half_n`` leaves column half 1 holding the zero fragments, so it is not stored."""
        halves = (frag_even, frag_odd)
        if const_expr(half_n):
            halves = (frag_even,)
        if const_expr(pair_cols):
            store_c.store_band_pair16(frag_even, frag_odd, row, pair_col, N_TILES_B, mask_cols=not col_safe)
        if const_expr(pair_tiles):
            # The live half's column tiles already alternate even/odd, so the pair is its j 0 against j 1.
            store_c.store_band_pair16(
                frag_even[0::2], frag_even[1::2], row, base_col, 1, mask_cols=not col_safe
            )
        if const_expr(not pair_cols and not pair_tiles):
            store_c.store_band16(
                halves,
                row,
                base_col,
                LDS_BLOCK_N,
                N_TILES_A,
                N_TILES_B,
                store_c.c_rows,
                mask_n=not col_safe,
            )

    # m half 0 finalises early, so its stores issue under the remaining matrix work, past the drain.
    _store_row(c00_frag, c01_frag, base_row)

    a1_frag = a_s2r.load(a_cur1)
    rocdl.s_barrier()
    rocdl.sched_barrier(0)
    c10_frag = _mma(2, c10_frag, a1_frag, b0_frag)
    if const_expr(not half_n):
        c11_frag = _mma(3, c11_frag, a1_frag, b1_frag)
    rocdl.sched_barrier(0)
    rocdl.s_barrier()

    if const_expr(PRED):
        c10_frag = [Vec(fx.memref_load_vec(reg)) for reg in acc[2]]
        if const_expr(not half_n):
            c11_frag = [Vec(fx.memref_load_vec(reg)) for reg in acc[3]]

    _store_row(c10_frag, c11_frag, base_row + LDS_BLOCK_M)


def gemm_bf16_nt_tile(
    A,
    B_T,
    C,
    c_m,
    c_n,
    lds,
    block_m=None,
    block_n=None,
    *,
    K,
    BLOCK_M,
    BLOCK_N,
    n_blocks=None,
    GROUP_M=1,
    num_xcd=8,
    persistent=False,
    ab_ty=fx.BFloat16,
    out_fp16=False,
    nt_vmcnt=3,
    b_group_base=None,
    c_cache_modifier=0,
    pair_n=False,
    n_tail=None,
):
    assert BLOCK_M >= 128 and BLOCK_N >= 256 and BLOCK_M % 128 == 0 and BLOCK_N % 256 == 0
    assert K % BLOCK_K == 0, f"bf16 NT needs K % {BLOCK_K} == 0 (got K={K})"
    N_TILES_A = BLOCK_M // 128
    N_TILES_B = BLOCK_N // 256
    LDS_BLOCK_M = BLOCK_M // 2
    LDS_BLOCK_N = BLOCK_N // 2
    N_LDS_STEPS_A = LDS_BLOCK_M // 64
    N_LDS_STEPS_B = LDS_BLOCK_N // 64
    N_LDS_ROUNDS = max(N_LDS_STEPS_A, N_LDS_STEPS_B)

    lane_id = fx.thread_idx.x % 64
    wave_id = fx.thread_idx.x // 64
    wave_m = wave_id // 4
    wave_n = wave_id % 4
    # Two waves share a SIMD (``wave_id % 4``) and must share wave_n to read one B fragment.
    MAIN_GRID = (N_TILES_A, N_TILES_B, wave_m, wave_n)
    TAIL_GRID = (1, 1, wave_id // 2, wave_id % 2)

    if block_m is None:
        num_pid_m = ceildiv(c_m, BLOCK_M)
        pid = xcd_remap_pid(fx.block_idx.x, num_pid_m * n_blocks, num_xcd)
        num_pid_in_group = GROUP_M * n_blocks
        group_id = pid // num_pid_in_group
        pid_in_group = pid % num_pid_in_group
        first_pid_m = group_id * GROUP_M
        remaining_m = num_pid_m - first_pid_m
        group_size_m = arith.select(remaining_m < GROUP_M, remaining_m, fx.Int32(GROUP_M))
        block_m = first_pid_m + (pid_in_group % group_size_m)
        block_n = pid_in_group // group_size_m

    A0_gl_offset = (block_m * BLOCK_M) * K
    A1_gl_offset = (block_m * BLOCK_M + LDS_BLOCK_M) * K
    B0_gl_offset = (block_n * BLOCK_N) * K
    # Column-interleaved feed: LDS half 0/1 hold the block's even and odd output columns, paired at store.
    PAIR_COLS = pair_n and N_TILES_B == 1
    if b_group_base is not None:
        B0_gl_offset = B0_gl_offset + b_group_base

    gA = make_fp16_bf16_buffer_tensor(A)
    gB = make_fp16_bf16_buffer_tensor(B_T)
    a_div = fx.logical_divide(gA, fx.make_layout(1, 1))
    b_div = fx.logical_divide(gB, fx.make_layout(1, 1))

    gl_off_a = compute_global_swizzle_bf16(lane_id, wave_id, K, N_LDS_ROUNDS)
    gl_off_b = compute_global_swizzle_bf16(lane_id, wave_id, K, N_LDS_ROUNDS)

    # The g2s/LDS type only sizes the addressing (2 bytes either way); the operand format is
    # decided by the mfma atom, so fp16 rides the bf16 staging untouched.
    a_g2s = G2SLoader(a_div, gl_off_a, N_LDS_STEPS_A, fx.BFloat16.ir_type, wave_id)
    b_g2s = G2SLoader(b_div, gl_off_b, N_LDS_STEPS_B, fx.BFloat16.ir_type, wave_id)
    _out_ty = fx.Float16 if out_fp16 else fx.BFloat16
    store_c = StoreCBf16(C, c_m, c_n, _out_ty, cache_modifier=c_cache_modifier)

    def _run(pair_cols, grid, half_n, col_safe=False, b_steps=N_LDS_STEPS_B, pair_tiles=False):
        # The bodies differ only in B's column layout, so the loader is re-pointed, not duplicated.
        n_a, n_b, w_m, w_n = grid
        if pair_cols:
            b_g2s.gl_offsets = compute_global_swizzle_bf16(lane_id, wave_id, K, N_LDS_ROUNDS, row_step=2)
        elif pair_tiles:
            assert n_b == 1, "pair_tiles pairs a wave's two 16-column tiles"
            b_g2s.gl_offsets = compute_global_swizzle_bf16(lane_id, wave_id, K, N_LDS_ROUNDS, pair_span=32)
        else:
            b_g2s.gl_offsets = gl_off_b
        b_g2s.n_load_steps = b_steps
        n_a16, n_b16 = 2 * n_a, 2 * n_b
        dense_mma_pipeline_bf16(
            lds,
            a_g2s,
            b_g2s,
            S2RLoader16x16Bf16(w_m, n_a16),
            S2RLoader16x16Bf16(w_n, n_b16),
            Mfma16x16x32(n_a16, n_b16, ab_ty),
            store_c,
            A0_gl_offset,
            A1_gl_offset,
            B0_gl_offset,
            B0_gl_offset + (K if pair_cols else LDS_BLOCK_N * K),
            BLOCK_K,
            BLOCK_K,
            block_m,
            block_n,
            w_m,
            w_n,
            K,
            BLOCK_M,
            BLOCK_N,
            nt_vmcnt,
            pair_cols=pair_cols,
            pair_tiles=pair_tiles,
            half_n=half_n,
            n_tiles_a=n_a16,
            n_tiles_b=n_b16,
            wave_hi=wave_m,
            col_safe=col_safe,
            persistent=persistent,
        )

    TAIL_QUADS = 0 if n_tail is None else ceildiv(n_tail, 32)
    if TAIL_QUADS not in (2, 4):
        _run(PAIR_COLS, MAIN_GRID, False, n_tail == 0)
    else:
        tail_grid = TAIL_GRID if TAIL_QUADS == 2 else MAIN_GRID
        emit_if_then((block_n + 1) * BLOCK_N <= c_n, lambda: _run(PAIR_COLS, MAIN_GRID, False, True))
        emit_if_then(
            (block_n + 1) * BLOCK_N > c_n,
            lambda: _run(False, tail_grid, True, n_tail % 32 == 0, ceildiv(n_tail, 64), pair_tiles=pair_n),
        )


def _gemm_bf16_nn_tn_tile_impl(
    A,
    B,
    C,
    c_m,
    c_n,
    lds,
    block_m,
    block_n,
    *,
    a_transpose,
    K,
    BLOCK_M,
    BLOCK_N,
    n_blocks=None,
    GROUP_M=1,
    num_xcd=8,
    persistent=False,
    ab_ty=fx.BFloat16,
    out_fp16=False,
    nt_vmcnt=3,
    b_group_base=None,
    c_cache_modifier=0,
    n_tail=0,
):
    assert BLOCK_M >= 128 and BLOCK_N >= 256 and BLOCK_M % 128 == 0 and BLOCK_N % 256 == 0
    assert K % BLOCK_K == 0, f"bf16 NN/TN needs K % {BLOCK_K} == 0 (got K={K})"
    LDS_BLOCK_M = BLOCK_M // 2
    LDS_BLOCK_N = BLOCK_N // 2
    N_LDS_STEPS_A = LDS_BLOCK_M // 64
    N_LDS_STEPS_B = LDS_BLOCK_N // 64
    N_LDS_ROUNDS = max(N_LDS_STEPS_A, N_LDS_STEPS_B)
    NTA16 = LDS_BLOCK_M // 32
    NTB16 = LDS_BLOCK_N // 64
    assert NTA16 in (1, 2, 4) and NTB16 in (1, 2, 4), "a wave's tiles must not straddle a chunk group"

    lane_id = fx.thread_idx.x % 64
    wave_id = fx.thread_idx.x // 64
    wave_hi = wave_id // 4
    # Transposing it trades B's free tr16 read for A's dearer ds_read_b128 and loses.
    wave_n = wave_id % 4
    MAIN_GRID = (NTA16, NTB16, wave_hi, wave_n)
    TAIL_GRID = (2, 2, wave_id // 2, wave_id % 2)

    if block_m is None:
        num_pid_m = ceildiv(c_m, BLOCK_M)
        pid = xcd_remap_pid(fx.block_idx.x, num_pid_m * n_blocks, num_xcd)
        num_pid_in_group = GROUP_M * n_blocks
        group_id = pid // num_pid_in_group
        pid_in_group = pid % num_pid_in_group
        first_pid_m = group_id * GROUP_M
        remaining_m = num_pid_m - first_pid_m
        group_size_m = arith.select(remaining_m < GROUP_M, remaining_m, fx.Int32(GROUP_M))
        block_m = first_pid_m + (pid_in_group % group_size_m)
        block_n = pid_in_group // group_size_m

    if a_transpose:
        A0_gl_offset = block_m * BLOCK_M + 0
        A1_gl_offset = block_m * BLOCK_M + LDS_BLOCK_M
        a_k_step = BLOCK_K * c_m
    else:
        A0_gl_offset = (block_m * BLOCK_M) * K
        A1_gl_offset = (block_m * BLOCK_M + LDS_BLOCK_M) * K
        a_k_step = BLOCK_K
    B0_gl_offset = block_n * BLOCK_N + 0
    B1_gl_offset = block_n * BLOCK_N + LDS_BLOCK_N
    b_k_step = BLOCK_K * c_n
    if b_group_base is not None:
        B0_gl_offset = B0_gl_offset + b_group_base
        B1_gl_offset = B1_gl_offset + b_group_base

    gA = make_fp16_bf16_buffer_tensor(A)
    gB = make_fp16_bf16_buffer_tensor(B)
    a_div = fx.logical_divide(gA, fx.make_layout(1, 1))
    b_div = fx.logical_divide(gB, fx.make_layout(1, 1))
    if a_transpose:
        gl_off_a = compute_global_swizzle_nn_bf16_wide(lane_id, wave_id, c_m, N_LDS_STEPS_A)
    else:
        gl_off_a = compute_global_swizzle_bf16(lane_id, wave_id, K, N_LDS_ROUNDS)
    gl_off_b = compute_global_swizzle_nn_bf16_wide(lane_id, wave_id, c_n, N_LDS_STEPS_B)

    a_g2s = G2SLoader(a_div, gl_off_a, N_LDS_STEPS_A, fx.BFloat16.ir_type, wave_id)
    b_g2s = G2SLoader(b_div, gl_off_b, N_LDS_STEPS_B, fx.BFloat16.ir_type, wave_id)
    _out_ty = fx.Float16 if out_fp16 else fx.BFloat16
    store_c = StoreCBf16(C, c_m, c_n, _out_ty, cache_modifier=c_cache_modifier)

    def _run(grid, quad_conds, half_n, col_safe=False, b_steps=N_LDS_STEPS_B):
        n_a16, n_b16, w_m, w_n = grid
        b_g2s.n_load_steps = b_steps
        a_s2r = S2RLoaderTr16x32Bf16Wide(w_m, n_a16) if a_transpose else S2RLoader16x16Bf16(w_m, n_a16)
        dense_mma_pipeline_bf16(
            lds,
            a_g2s,
            b_g2s,
            a_s2r,
            S2RLoaderTr16x32Bf16Wide(w_n, n_b16),
            Mfma16x16x32(n_a16, n_b16, ab_ty),
            store_c,
            A0_gl_offset,
            A1_gl_offset,
            B0_gl_offset,
            B1_gl_offset,
            a_k_step,
            b_k_step,
            block_m,
            block_n,
            w_m,
            w_n,
            K,
            BLOCK_M,
            BLOCK_N,
            nt_vmcnt,
            quad_conds=quad_conds,
            half_n=half_n,
            n_tiles_a=n_a16,
            n_tiles_b=n_b16,
            wave_hi=wave_hi,
            col_safe=col_safe,
            persistent=persistent,
        )

    # Fork a ragged N on the workgroup-uniform column index: barriers stay matched and a feed half drops.
    TAIL_TILES = ceildiv(n_tail, 32)
    if n_tail == 0:
        _run(MAIN_GRID, None, False, True)
    else:
        conds = _quad_col_conds(block_n * BLOCK_N + wave_n * (NTB16 * 16), LDS_BLOCK_N, c_n)
        half_n = n_tail <= LDS_BLOCK_N
        exact = n_tail % 32 == 0
        if TAIL_TILES == 2:
            tail = (TAIL_GRID, None, half_n, exact, ceildiv(n_tail, 64))
        else:
            tail = (MAIN_GRID, conds, half_n, False, N_LDS_STEPS_B)
        emit_if_then((block_n + 1) * BLOCK_N <= c_n, lambda: _run(MAIN_GRID, None, False, True))
        emit_if_then((block_n + 1) * BLOCK_N > c_n, lambda: _run(*tail))


def gemm_bf16_nn_tile(
    A,
    B,
    C,
    c_m,
    c_n,
    lds,
    block_m=None,
    block_n=None,
    *,
    K,
    BLOCK_M,
    BLOCK_N,
    n_blocks=None,
    GROUP_M=1,
    num_xcd=8,
    persistent=False,
    ab_ty=fx.BFloat16,
    out_fp16=False,
    nt_vmcnt=3,
    b_group_base=None,
    c_cache_modifier=0,
    n_tail=0,
):
    _gemm_bf16_nn_tn_tile_impl(
        A,
        B,
        C,
        c_m,
        c_n,
        lds,
        block_m,
        block_n,
        a_transpose=False,
        K=K,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        n_blocks=n_blocks,
        GROUP_M=GROUP_M,
        num_xcd=num_xcd,
        persistent=persistent,
        ab_ty=ab_ty,
        out_fp16=out_fp16,
        nt_vmcnt=nt_vmcnt,
        b_group_base=b_group_base,
        c_cache_modifier=c_cache_modifier,
        n_tail=n_tail,
    )


def gemm_bf16_tn_tile(
    A,
    B,
    C,
    c_m,
    c_n,
    lds,
    block_m=None,
    block_n=None,
    *,
    K,
    BLOCK_M,
    BLOCK_N,
    n_blocks=None,
    GROUP_M=1,
    num_xcd=8,
    persistent=False,
    ab_ty=fx.BFloat16,
    out_fp16=False,
    nt_vmcnt=3,
    b_group_base=None,
):
    _gemm_bf16_nn_tn_tile_impl(
        A,
        B,
        C,
        c_m,
        c_n,
        lds,
        block_m,
        block_n,
        a_transpose=True,
        K=K,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        n_blocks=n_blocks,
        GROUP_M=GROUP_M,
        num_xcd=num_xcd,
        persistent=persistent,
        ab_ty=ab_ty,
        out_fp16=out_fp16,
        nt_vmcnt=nt_vmcnt,
        b_group_base=b_group_base,
    )


def gemm_bf16_tile(layout, *args, **kwargs):
    # static layout dispatch: layout is a compile-time constant.
    # Grouped callers rebase A/C into per-tile views (via make_bf16_fp16_tile_tensor)
    # before calling, so every impl's API stays unchanged here.
    if layout == "nt":
        return gemm_bf16_nt_tile(*args, **kwargs)
    if layout == "nn":
        return gemm_bf16_nn_tile(*args, **kwargs)
    if layout == "tn":
        return gemm_bf16_tn_tile(*args, **kwargs)
    raise ValueError(f"unsupported layout {layout}")
