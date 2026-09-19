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

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl._mlir.dialects import llvm as _llvm
from flydsl.compiler.ast_rewriter import ASTRewriter
from flydsl.expr import arith, const_expr, range_constexpr, rocdl
from flydsl.expr import buffer_ops as _buffer_ops
from flydsl.expr.primitive import get_iter as _get_iter
from flydsl.expr.primitive import ptrtoint as _ptrtoint
from flydsl.expr.typing import Vector as Vec
from flydsl.expr.utils.arith import ArithValue

from primus_turbo.flydsl.utils.gemm_helper import (
    BLOCK_K,
    G2SLoader,
    Mfma16x16x32,
    S2RLoader16x16Bf16,
    S2RLoaderTr16x32Bf16Wide,
    StoreCBf16,
    compile_with_scratch_out,
    compute_global_swizzle_bf16,
    compute_global_swizzle_nn_bf16_wide,
    emit_for,
    emit_if_then,
    group_m_tile_decode,
    make_bf16_buffer_tensor_rebased,
    make_bf16_fp16_tile_tensor,
    make_fp16_bf16_buffer_tensor,
    make_row_band_resource,
    make_value_attrs,
    resolve_accum_out,
    wait_barrier,
    xcd_remap_pid,
)
from primus_turbo.flydsl.utils.prims import _i64, ceildiv

# isort: on


def _make_shared_storage(BLOCK_M, BLOCK_N, chunk_stride=1024, single_n=False):
    """LDS double-buffer struct for one dense tile.

    Standard layout: ``BLOCK_N`` is the tile's full N extent, split into two accumulator
    regions (4 B buffers, each ``BLOCK_N//2`` wide) -- the per-buffer size formula divides by
    16 because it is handed the *doubled* (both-region) width, matching A's own convention
    (A_lds_cur_0 is ``BLOCK_M//2`` wide, sized from the undivided ``BLOCK_M``).

    ``single_n=True`` collapses to ONE region spanning the tile's whole (real, undoubled)
    ``BLOCK_N`` -- only 2 B buffers (cur/next, no second region), each sized directly off
    ``BLOCK_N`` with a ``//8`` divisor (half of ``//16``'s doubling, since this ``BLOCK_N`` is
    already the real per-buffer width, not 2x it). This is what admits a ``BLOCK_N`` that is a
    multiple of 64 but not 128 (e.g. 320), since there is no second-region split to force the
    /2 to land on a 128-boundary."""
    a_lds_size = (BLOCK_M // 16) * chunk_stride // 2

    if single_n:
        b_lds_size = (BLOCK_N // 8) * chunk_stride // 2

        @fx.struct
        class SharedStorage:
            A_lds_cur_0: fx.Array[fx.BFloat16, a_lds_size, 16]
            A_lds_cur_1: fx.Array[fx.BFloat16, a_lds_size, 16]
            A_lds_next_0: fx.Array[fx.BFloat16, a_lds_size, 16]
            A_lds_next_1: fx.Array[fx.BFloat16, a_lds_size, 16]
            B_lds_cur_0: fx.Array[fx.BFloat16, b_lds_size, 16]
            B_lds_next_0: fx.Array[fx.BFloat16, b_lds_size, 16]

        return SharedStorage

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


# ---------------------------------------------------------------------------
# Dense LM-head launcher: forward NT, dgrad NN, and beta=1 wgrad TN.
#
# NT keeps the fixed-K ``dense_mma_pipeline_bf16`` above (K=2880 -> 45 K-iters is a
# sane unroll). NN and TN reduce over K=128256 / 32768, i.e. 2004 / 512 K-iters, which
# ``range_constexpr`` would unroll into an unbuildable kernel, so they run the same
# 4-quadrant body inside a chunked ``emit_for`` (CHUNK k-iterations Python-unrolled per
# runtime trip, so the 4-buffer rotation is the identity at the chunk boundary).
# ---------------------------------------------------------------------------


def _ptr_only_view(t: torch.Tensor) -> torch.Tensor:
    return t.view(torch.int32)


class StoreCBf16Accum(StoreCBf16):
    """``StoreCBf16`` whose band store is a beta=1 read-modify-write (wgrad accumulation).

    Every read-back is issued before the first store waits on one, and the add happens in f32
    before the bf16 cast so the accumulate rounds exactly once (cf. ``StoreCPerTensor._accum``)."""

    def store_band16(
        self, c_frags, base_row, base_col, col_step, n_tiles_a, n_tiles_b, row_bound, mask_n=False,
        col_tile=16,
    ):
        rsrc = make_row_band_resource(self.c_base, base_row, row_bound, self.c_cols, 2)
        lane_col = self.lane_id % 16
        col_ok = [
            (base_col + q * col_step + lane_col < self.c_cols) if mask_n else None
            for q in range(len(c_frags))
        ]
        cols = self.c_cols
        base_e = ((self.lane_id // 16) * 4) * cols + base_col + lane_col
        plan = []
        for ti in range_constexpr(n_tiles_a):
            for r in range_constexpr(4):
                row_e = base_e + (ti * 16 + r) * cols
                for q in range_constexpr(len(c_frags)):
                    for j in range_constexpr(n_tiles_b):
                        plan.append((row_e + q * col_step + j * col_tile, q, ti * n_tiles_b + j, r))
        prev = [
            _buffer_ops.buffer_load(rsrc, off, vec_width=1, dtype=self.out_ty.ir_type, mask=col_ok[q])
            for off, q, _, _ in plan
        ]
        for idx in range_constexpr(len(plan)):
            off, q, t, r = plan[idx]
            val = Vec(c_frags[q][t])[r] + self.out_ty(prev[idx]).to(fx.Float32)
            _buffer_ops.buffer_store(
                val.to(self.out_ty), rsrc, off, mask=col_ok[q], cache_modifier=self.cache_modifier
            )


@ASTRewriter.transform
def dense_mma_chunked_bf16(
    lds,
    a_g2s,
    b_g2s,
    a_s2r,
    b_s2r,
    mfma,
    acc,
    a0_off,
    a1_off,
    b0_off,
    b1_off,
    a_k_step,
    b_k_step,
    n_chunks,
    CHUNK,
    n_steps_a,
    n_steps_b,
    wave_m,
    nt_vmcnt,
    half_n=False,
    a_chunk_src=None,
):
    """Runtime-K twin of ``dense_mma_pipeline_bf16``'s main loop: same 4-buffer, 4-quadrant,
    8-barrier body, but ``CHUNK`` k-iterations per ``emit_for`` trip instead of one fully
    unrolled ``range_constexpr``.  Accumulators live in rmem, not SSA, so nothing has to phi
    through the ``scf.for``.  ``a_chunk_src`` re-bases A's SRD once per chunk for an operand
    whose K span exceeds the 4 GB a single buffer descriptor reaches (wgrad's 8.4 GB A).

    The body's barrier COUNT is load-bearing, not just its placement: the prologue's
    ``wave_m == 1`` barrier leaves the low wave half permanently one rendezvous ahead, so a
    barrier added or dropped on one path re-pairs every later rendezvous between the halves.
    Keep this identical to ``dense_mma_pipeline_bf16``'s loop -- an extra barrier on the
    ``half_n`` path let A1's k+1 refill land while the other half was still reading it,
    which showed up as a few hundred wrong elements in the ragged-N tile on ~half of runs."""
    B1_STEPS = 0 if half_n else n_steps_b
    LOOP_DRAIN = (n_steps_a if half_n else 2 * n_steps_a) + n_steps_b
    # ``nt_vmcnt`` counts the full body's issue stream, so a narrowed feed retires a different load.
    ITER_DRAIN = n_steps_a if half_n else nt_vmcnt
    acc00, acc01, acc10, acc11 = acc
    if const_expr(a_chunk_src is not None):
        # Re-built per emitting region: the chunk loop leaves ``gl_src`` pointing at a value
        # defined inside the loop, which a sibling branch's prologue could not dominate.
        a_g2s.gl_src = a_chunk_src(fx.Int32(0))

    def _mma(dst, a, b):
        c = [Vec(fx.memref_load_vec(r)) for r in dst]
        c = mfma.call(a, b, c)
        for t in range_constexpr(len(dst)):
            fx.memref_store_vec(c[t], dst[t])

    b_g2s.load(lds.B_lds_cur_0, b0_off + 0 * b_k_step)
    a_g2s.load(lds.A_lds_cur_0, a0_off + 0 * a_k_step)
    if const_expr(not half_n):
        b_g2s.load(lds.B_lds_cur_1, b1_off + 0 * b_k_step)
    a_g2s.load(lds.A_lds_cur_1, a1_off + 0 * a_k_step)
    # One tile per WG, so only the high half has to stop here; cf. dense_mma_pipeline_bf16.
    if wave_m == 1:
        rocdl.s_barrier()
    wait_barrier(n_steps_a + B1_STEPS)

    b_g2s.load(lds.B_lds_next_0, b0_off + 1 * b_k_step)
    a_g2s.load(lds.A_lds_next_0, a0_off + 1 * a_k_step)
    if const_expr(not half_n):
        b_g2s.load(lds.B_lds_next_1, b1_off + 1 * b_k_step)
    wait_barrier(n_steps_a + n_steps_b + B1_STEPS)

    # Nested so the Python-level buffer rotation stays out of the runtime chunk loop.
    def _chunk(chunk_iv):
        chunk_idx = ArithValue(chunk_iv)
        if const_expr(a_chunk_src is not None):
            # Re-based per chunk: the chunk's K origin folds into the 64-bit SRD base, so the
            # in-chunk reach (CHUNK+2 k-iters) stays inside a 32-bit buffer offset.
            a_g2s.gl_src = a_chunk_src(chunk_idx)
            k_a0 = fx.Int32(0)
        else:
            k_a0 = chunk_idx * CHUNK
        k_b0 = chunk_idx * CHUNK
        a_cur0, a_cur1 = lds.A_lds_cur_0, lds.A_lds_cur_1
        a_next0, a_next1 = lds.A_lds_next_0, lds.A_lds_next_1
        # The second B region does not exist in the SharedStorage struct when half_n is
        # permanent (single_n's collapsed-to-one-region layout), so it must not be touched
        # even as a dead Python reference -- unlike the tail-fork's half_n, which shares a
        # struct that still allocates (but never uses) the _1 buffers.
        b_cur0 = lds.B_lds_cur_0
        b_next0 = lds.B_lds_next_0
        if const_expr(not half_n):
            b_cur1 = lds.B_lds_cur_1
            b_next1 = lds.B_lds_next_1
        for j in range_constexpr(CHUNK):
            ka = k_a0 + j
            kb = k_b0 + j
            b0 = b_s2r.load(b_cur0)
            a0 = a_s2r.load(a_cur0)
            a_g2s.load(a_next1, a1_off + (ka + 1) * a_k_step)
            rocdl.s_barrier()

            rocdl.sched_barrier(0)
            _mma(acc00, a0, b0)
            rocdl.sched_barrier(0)
            rocdl.s_barrier()

            if const_expr(not half_n):
                b1 = b_s2r.load(b_cur1)
            b_g2s.load(b_cur0, b0_off + (kb + 2) * b_k_step)
            if const_expr(not half_n):
                rocdl.s_barrier()
                rocdl.sched_barrier(0)
                _mma(acc01, a0, b1)
                rocdl.sched_barrier(0)
                rocdl.s_barrier()

            a1 = a_s2r.load(a_cur1)
            a_g2s.load(a_cur0, a0_off + (ka + 2) * a_k_step)
            rocdl.s_barrier()
            rocdl.sched_barrier(0)
            _mma(acc10, a1, b0)
            rocdl.sched_barrier(0)
            if const_expr(not half_n):
                rocdl.s_barrier()
                b_g2s.load(b_cur1, b1_off + (kb + 2) * b_k_step)
            wait_barrier(LOOP_DRAIN)
            if const_expr(not half_n):
                rocdl.sched_barrier(0)
                _mma(acc11, a1, b1)
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
            if const_expr(not half_n):
                b_cur1, b_next1 = b_next1, b_cur1

    emit_for(n_chunks, _chunk)


def dense_bf16_chunked_tile(
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
    CHUNK,
    a_chunk_src=None,
    n_tail=0,
    beta_is_one=False,
    ab_ty=fx.BFloat16,
    nt_vmcnt=3,
    lds_chunk_stride=1024,
    single_n=False,
):
    """One NN/TN output tile over a runtime K loop.  Geometry, swizzles, LDS layout and store
    are the fixed-K ``_gemm_bf16_nn_tn_tile_impl``'s; only the K loop differs.

    ``single_n=True`` collapses the two N accumulator quadrants into one region spanning the
    tile's whole (real) ``BLOCK_N``, permanently running the ``half_n`` feed/store path instead
    of forking it only for a ragged tail.  This frees ``BLOCK_N`` to be any multiple of 64 (not
    just 128), at the cost of the second N region's accumulator throughput -- see
    ``_make_shared_storage``'s docstring for the matching LDS layout."""
    assert BLOCK_M % 128 == 0
    if single_n:
        assert BLOCK_N % 64 == 0, "single_n needs BLOCK_N a multiple of 64 (one wave's tile atom)"
    else:
        assert BLOCK_N % 128 == 0
    assert K % (BLOCK_K * CHUNK) == 0 and CHUNK % 2 == 0
    LDS_BLOCK_M = BLOCK_M // 2
    LDS_BLOCK_N = BLOCK_N if single_n else BLOCK_N // 2
    N_LDS_STEPS_A = LDS_BLOCK_M // 64
    N_LDS_STEPS_B = LDS_BLOCK_N // 64
    N_LDS_ROUNDS = max(N_LDS_STEPS_A, N_LDS_STEPS_B)
    NTA16 = LDS_BLOCK_M // 32
    NTB16 = LDS_BLOCK_N // 64
    N_CHUNKS = (K // BLOCK_K) // CHUNK

    lane_id = fx.thread_idx.x % 64
    wave_id = fx.thread_idx.x // 64
    wave_hi = wave_id // 4
    wave_n = wave_id % 4

    if a_transpose:  # A is [K, M]
        a0_off = block_m * BLOCK_M
        a1_off = a0_off + LDS_BLOCK_M
        a_k_step = BLOCK_K * c_m
        gl_off_a = compute_global_swizzle_nn_bf16_wide(lane_id, wave_id, c_m, N_LDS_STEPS_A)
        a_div = a_chunk_src(fx.Int32(0))
    else:  # A is [M, K], already re-based to this tile's M band by the caller
        a0_off = fx.Int32(0)
        a1_off = fx.Int32(LDS_BLOCK_M * K)
        a_k_step = BLOCK_K
        gl_off_a = compute_global_swizzle_bf16(lane_id, wave_id, K, N_LDS_ROUNDS)
        a_div = fx.logical_divide(make_fp16_bf16_buffer_tensor(A), fx.make_layout(1, 1))
    b0_off = block_n * BLOCK_N
    b1_off = b0_off + LDS_BLOCK_N
    b_k_step = BLOCK_K * c_n
    gl_off_b = compute_global_swizzle_nn_bf16_wide(lane_id, wave_id, c_n, N_LDS_STEPS_B)
    b_div = fx.logical_divide(make_fp16_bf16_buffer_tensor(B), fx.make_layout(1, 1))

    a_g2s = G2SLoader(
        a_div, gl_off_a, N_LDS_STEPS_A, fx.BFloat16.ir_type, wave_id, chunk_stride=lds_chunk_stride
    )
    b_g2s = G2SLoader(
        b_div, gl_off_b, N_LDS_STEPS_B, fx.BFloat16.ir_type, wave_id, chunk_stride=lds_chunk_stride
    )
    store_cls = StoreCBf16Accum if beta_is_one else StoreCBf16
    store_c = store_cls(C, c_m, c_n, fx.BFloat16)

    def _run(n_a16, n_b16, w_m, w_n, half_n, b_steps, mask_n, n_w_n=4):
        b_g2s.n_load_steps = b_steps
        a_s2r = (
            S2RLoaderTr16x32Bf16Wide(w_m, n_a16, chunk_stride=lds_chunk_stride)
            if a_transpose
            else S2RLoader16x16Bf16(w_m, n_a16)
        )
        # Interleaved wave->tile map on B: wave w owns tiles w, w+n_w_n, ... so every LDS
        # offset is a constexpr and n_b16 is free of the 4-tile chunk-group alignment.
        b_s2r = S2RLoaderTr16x32Bf16Wide(w_n, n_b16, chunk_stride=lds_chunk_stride, n_waves=n_w_n)
        mfma = Mfma16x16x32(n_a16, n_b16, ab_ty)

        def _acc_group(need):
            # The second N region's two quadrants (indices 1, 3) are never read from under
            # half_n (dense_mma_chunked_bf16 guards every use with ``if not half_n``), so
            # skip allocating and zeroing their rmem tensors -- with single_n's full-loop
            # half_n this is 20 accumulator tiles/wave, not the tail-fork's 4, so leaving
            # this unconditional would blow the VGPR budget for no live use.
            if not need:
                return []
            return [fx.make_rmem_tensor(fx.make_layout(mfma.acc_len, 1), fx.Float32) for _ in range(n_a16 * n_b16)]

        acc = [_acc_group(True), _acc_group(not half_n), _acc_group(True), _acc_group(not half_n)]
        for quad in acc:
            for reg in quad:
                fx.memref_store_vec(mfma.zero_value, reg)
        dense_mma_chunked_bf16(
            lds,
            a_g2s,
            b_g2s,
            a_s2r,
            b_s2r,
            mfma,
            acc,
            a0_off,
            a1_off,
            b0_off,
            b1_off,
            a_k_step,
            b_k_step,
            N_CHUNKS,
            CHUNK,
            N_LDS_STEPS_A,
            b_steps,
            wave_hi,
            nt_vmcnt,
            half_n=half_n,
            a_chunk_src=a_chunk_src if a_transpose else None,
        )
        c = [[Vec(fx.memref_load_vec(reg)) for reg in quad] for quad in acc]
        base_row = block_m * BLOCK_M + w_m * (n_a16 * 16)
        base_col = block_n * BLOCK_N + w_n * 16
        for q0, q1, row in ((0, 1, base_row), (2, 3, base_row + LDS_BLOCK_M)):
            halves = (c[q0],) if half_n else (c[q0], c[q1])
            store_c.store_band16(
                halves, row, base_col, LDS_BLOCK_N, n_a16, n_b16, c_m, mask_n, col_tile=n_w_n * 16
            )

    # Fork a ragged N on the workgroup-uniform column index: barriers stay matched and a feed half drops.
    if single_n:
        assert n_tail == 0, "single_n expects a BLOCK_N that divides c_n exactly (no tail wired)"
        _run(NTA16, NTB16, wave_hi, wave_n, True, N_LDS_STEPS_B, False)
    elif n_tail == 0:
        _run(NTA16, NTB16, wave_hi, wave_n, False, N_LDS_STEPS_B, False)
    else:
        assert ceildiv(n_tail, 32) == 2, f"n_tail={n_tail}: only the 2-tile tail grid is wired"
        emit_if_then(
            (block_n + 1) * BLOCK_N <= c_n,
            lambda: _run(NTA16, NTB16, wave_hi, wave_n, False, N_LDS_STEPS_B, False),
        )
        emit_if_then(
            (block_n + 1) * BLOCK_N > c_n,
            lambda: _run(
                2, 2, wave_id // 2, wave_id % 2, True, ceildiv(n_tail, 64), n_tail % 32 != 0, 2
            ),
        )


def _dense_a_operand(A, block_m, a_trans, a_tile_elems, a_chunk_bytes, a_total_bytes):
    """This tile's A view, plus (TN only) a per-chunk SRD re-baser.

    TN reads A = [K, M] down its whole 8.4 GB K extent, which no single buffer descriptor
    reaches (``num_records`` is 32-bit), so the chunk's K origin folds into a 64-bit SRD base
    once per chunk.  NN reads A = [M, K] and only needs this tile's M band, so one int64
    re-base per tile is enough and the K offset stays a 32-bit buffer offset."""
    if a_trans:

        def _a_src(chunk_idx):
            off = _i64(chunk_idx) * fx.Int64(a_chunk_bytes)
            return fx.logical_divide(
                make_bf16_buffer_tensor_rebased(
                    A, fx.BFloat16.ir_type, off, fx.Int64(a_total_bytes) - off
                ),
                fx.make_layout(1, 1),
            )

        return A, _a_src
    a_base = fx.Int64(_ptrtoint(_get_iter(A)))
    tile = make_bf16_fp16_tile_tensor(
        a_base, _i64(block_m) * fx.Int64(a_tile_elems * 2), a_tile_elems
    )
    return tile, None


@functools.lru_cache(maxsize=64)
def _compile_dense_bf16_nn_tn(
    layout,
    M,
    N,
    K,
    BLOCK_M=256,
    BLOCK_N=256,
    GROUP_M=1,
    num_xcd=8,
    waves_per_eu=2,
    agpr_alloc=0,
    CHUNK=4,
    beta_is_one=False,
    nt_vmcnt=3,
    lds_chunk_stride=1024,
    single_n=False,
):
    A_TRANS = layout == "tn"
    assert M % BLOCK_M == 0, "the NN/TN store has no ragged-M path"
    N_BLOCKS_M = M // BLOCK_M
    N_BLOCKS_N = ceildiv(N, BLOCK_N)
    TOTAL = N_BLOCKS_M * N_BLOCKS_N
    SharedStorage = _make_shared_storage(BLOCK_M, BLOCK_N, chunk_stride=lds_chunk_stride, single_n=single_n)
    A_TILE_ELEMS = BLOCK_M * K  # NN only: A is [M, K], re-based per M block
    A_CHUNK_BYTES = CHUNK * BLOCK_K * M * 2  # TN only: A is [K, M], re-based per chunk
    A_TOTAL_BYTES = K * M * 2

    @flyc.kernel(known_block_size=[512, 1, 1])
    def kernel_dense_nn_tn(A: fx.Tensor, B: fx.Tensor, C: fx.Tensor):
        _ = str(fx.thread_idx.x)
        lds = fx.SharedAllocator().allocate(SharedStorage).peek()

        def _do_tile(pid):
            tile = xcd_remap_pid(pid, TOTAL, num_xcd)
            block_m, block_n = group_m_tile_decode(tile, N_BLOCKS_M, N_BLOCKS_N, GROUP_M)
            # Module-level so the layout fork stays a Python branch: a nested def inside the
            # kernel is AST-rewritten, and a value defined in an scf.if does not escape it.
            a_arg, a_src = _dense_a_operand(
                A, block_m, A_TRANS, A_TILE_ELEMS, A_CHUNK_BYTES, A_TOTAL_BYTES
            )
            dense_bf16_chunked_tile(
                a_arg,
                B,
                C,
                M,
                N,
                lds,
                block_m,
                block_n,
                a_transpose=A_TRANS,
                K=K,
                BLOCK_M=BLOCK_M,
                BLOCK_N=BLOCK_N,
                CHUNK=CHUNK,
                a_chunk_src=a_src,
                n_tail=0 if single_n else N % BLOCK_N,
                beta_is_one=beta_is_one,
                nt_vmcnt=nt_vmcnt,
                lds_chunk_stride=lds_chunk_stride,
                single_n=single_n,
            )

        _do_tile(fx.block_idx.x)

    @flyc.jit
    def launch_dense_nn_tn(A, B, C, stream: fx.Stream):
        kernel_dense_nn_tn(
            A,
            B,
            C,
            value_attrs=make_value_attrs(waves_per_eu, agpr_alloc, "512,512"),
        ).launch(grid=(TOTAL, 1, 1), block=(512, 1, 1), stream=stream)

    return launch_dense_nn_tn


@functools.lru_cache(maxsize=64)
def _compile_dense_bf16_nt(
    M,
    N,
    K,
    BLOCK_M=256,
    BLOCK_N=256,
    GROUP_M=4,
    num_xcd=8,
    waves_per_eu=2,
    agpr_alloc=0,
    nt_vmcnt=3,
):
    assert M % BLOCK_M == 0, "A/C are rebased per M block; a ragged M needs the masked path"
    N_BLOCKS_M = M // BLOCK_M
    N_BLOCKS_N = (N + BLOCK_N - 1) // BLOCK_N
    TOTAL = N_BLOCKS_M * N_BLOCKS_N
    SharedStorage = _make_shared_storage(BLOCK_M, BLOCK_N)
    A_TILE_BYTES = BLOCK_M * K * 2
    C_TILE_BYTES = BLOCK_M * N * 2

    @flyc.kernel(known_block_size=[512, 1, 1])
    def kernel_dense_nt(A: fx.Tensor, B: fx.Tensor, C: fx.Tensor):
        _ = str(fx.thread_idx.x)
        lds = fx.SharedAllocator().allocate(SharedStorage).peek()

        def _do_tile(pid):
            tile = xcd_remap_pid(pid, TOTAL, num_xcd)
            block_m, block_n = group_m_tile_decode(tile, N_BLOCKS_M, N_BLOCKS_N, GROUP_M)
            a_base = fx.Int64(_ptrtoint(_get_iter(A)))
            c_base = fx.Int64(_ptrtoint(_get_iter(C)))
            # C is up to 8.4 GB here; a single SRD only reaches 4 GB, so the row band is
            # rebased in int64 per tile and the tile then addresses it with block_m = 0.
            a_tile = make_bf16_fp16_tile_tensor(
                a_base, _i64(block_m) * fx.Int64(A_TILE_BYTES), BLOCK_M * K
            )
            c_tile = make_bf16_fp16_tile_tensor(
                c_base, _i64(block_m) * fx.Int64(C_TILE_BYTES), BLOCK_M * N
            )
            gemm_bf16_nt_tile(
                a_tile,
                B,
                c_tile,
                BLOCK_M,
                N,
                lds,
                fx.Int32(0),
                block_n,
                K=K,
                BLOCK_M=BLOCK_M,
                BLOCK_N=BLOCK_N,
                n_blocks=N_BLOCKS_N,
                GROUP_M=GROUP_M,
                num_xcd=num_xcd,
                nt_vmcnt=nt_vmcnt,
                pair_n=N % 2 == 0,
                n_tail=N % BLOCK_N,
            )

        _do_tile(fx.block_idx.x)

    @flyc.jit
    def launch_dense_nt(A, B, C, stream: fx.Stream):
        kernel_dense_nt(
            A,
            B,
            C,
            value_attrs=make_value_attrs(waves_per_eu, agpr_alloc, "512,512"),
        ).launch(grid=(TOTAL, 1, 1), block=(512, 1, 1), stream=stream)

    return launch_dense_nt


_DENSE_BF16_CACHE: dict = {}


# Per-role launch configuration, keyed by layout. Each entry is measured, not inherited:
# see the campaign notes for the sweep behind every value.
_DENSE_BF16_CFG = {
    #        BLOCK_M BLOCK_N GROUP_M num_xcd waves_per_eu agpr_alloc
    "nt": dict(BLOCK_M=256, BLOCK_N=256, GROUP_M=8, num_xcd=8, waves_per_eu=2, agpr_alloc=64),
    "nn": dict(BLOCK_M=256, BLOCK_N=256, GROUP_M=1, num_xcd=8, waves_per_eu=2, agpr_alloc=64),
    # single_n=True: one N accumulator region spanning the whole (real) BLOCK_N=320 instead of
    # two 128-wide regions -- 2880 = 9*320 exactly (vs 2880 = 11*256 + 64 ragged), so this grid
    # is tail-free on both axes.  lds_chunk_stride stays at the unpadded default: at BLOCK_N=320
    # the single-region B buffers already use (320/8)*chunk_stride*2 bytes each (measured
    # group_segment_fixed_size below), and 1152 overflows the 163840 B LDS budget here.
    "tn": dict(
        BLOCK_M=256, BLOCK_N=320, GROUP_M=1, num_xcd=4, waves_per_eu=2, agpr_alloc=0,
        lds_chunk_stride=1024, single_n=True,
    ),
}


def gemm_bf16_flydsl_kernel(
    a: torch.Tensor,
    b: torch.Tensor,
    trans_a: bool = False,
    trans_b: bool = False,
    out_dtype: torch.dtype = torch.bfloat16,
    trans_c: bool = False,
    beta: float = 0.0,
    out: torch.Tensor = None,
    **cfg,
):
    """Dense bf16 GEMM ``op(a) @ op(b)`` for the GPT-OSS LM head (NT / NN / TN).

    ``beta=1`` accumulates into ``out`` in place (bf16, one rounding); ``beta=0`` allocates."""
    assert a.dtype == torch.bfloat16 and b.dtype == torch.bfloat16
    assert not trans_c, "trans_c is not part of the scored LM-head contract"
    assert beta in (0.0, 1.0), f"beta={beta} unsupported (0 or 1)"
    layout = ("t" if trans_a else "n") + ("t" if trans_b else "n")
    if layout == "nt":  # C[M,N] = A[M,K] @ B[N,K]^T
        M, K = a.shape
        N, Kb = b.shape
    elif layout == "nn":  # C[M,N] = A[M,K] @ B[K,N]
        M, K = a.shape
        Kb, N = b.shape
    elif layout == "tn":  # C[M,N] = A[K,M]^T @ B[K,N]
        K, M = a.shape
        Kb, N = b.shape
    else:
        raise ValueError("tt layout is not part of the LM-head contract")
    assert Kb == K, f"K mismatch: {K} vs {Kb}"

    conf = dict(_DENSE_BF16_CFG[layout])
    conf.update(cfg)
    # beta=1.0 with out=None would otherwise accumulate the GEMM result into torch.empty's
    # uninitialized memory (silently, no error) instead of failing loudly like every other
    # accumulate path in this file; resolve_accum_out is the shared beta=1 guard (see its
    # docstring) and asserts beta==0.0 before allocating.
    out = resolve_accum_out(out, beta, (M, N), a.device, out_dtype)
    args = (
        _ptr_only_view(a),
        flyc.from_torch_tensor(b.reshape(-1)),
        _ptr_only_view(out),
        torch.cuda.current_stream(),
    )
    key = (layout, M, N, K, beta, tuple(sorted(conf.items())))
    compiled = _DENSE_BF16_CACHE.get(key)
    if compiled is None:
        if layout == "nt":
            launch = _compile_dense_bf16_nt(M, N, K, **conf)
        else:
            launch = _compile_dense_bf16_nn_tn(layout, M, N, K, beta_is_one=beta == 1.0, **conf)
        # flyc.compile RUNS the kernel once to build its artifact, which a beta=1 build would
        # fold into the caller's buffer as a second GEMM (and make the first call differ from
        # every later one). Compile against a scratch of the same shape instead.
        compiled = compile_with_scratch_out(launch, args, out_index=2)
        _DENSE_BF16_CACHE[key] = compiled
    compiled(*args)
    return out
