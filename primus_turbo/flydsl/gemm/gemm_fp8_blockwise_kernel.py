###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Primus-Turbo blockwise (per-128-block scaled) FP8 GEMM kernel (FlyDSL).

NT / NN / TN layouts on the SAME 256x256 tile, BLOCK_K=128, 8-wave
(wave_m=2 x wave_n=4), mfma_f32_16x16x128_f8f6f4 pipeline as the dense kernel
(``gemm_fp8_kernel.py``); the only difference is that the per-128-contraction-
block scale ``a_scale * b_scale`` is folded into a running f32 accumulator after
each block's mfma instead of one per-tensor scalar at store time. All primitives
are reused from ``flydsl.utils.gemm_helper`` (no external FlyDSL ``kernels``
package dependency).

The three layouts map to the FP8 BLOCKWISE training directions:
  - NT (trans_a=F, trans_b=T): forward  out[M,N]   = a[M,K] @ b[N,K]^T   (CON=K)
  - NN (trans_a=F, trans_b=F): dgrad    grad_a[M,K]= grad_out[M,N] @ b[N,K] (CON=N)
  - TN (trans_a=T, trans_b=F): wgrad    grad_b[N,K]= grad_out[M,N]^T @ a[M,K] (CON=M)

A-side scale is always 1-D (per output row, per contraction block); the launcher
transposes it to ``[scale_con, c_rows]`` so a single index ``kb*c_rows + row``
serves all three. B-side scale is a 2-D weight block grid for fwd/dgrad and a
per-output-column 1-D grid for wgrad — supplied as a per-layout index closure.
"""

import functools

import torch

# isort: off
from primus_turbo.flydsl.utils.gemm_helper import (
    BlockScaleReader,
    G2SLoader,
    Mfma16x16x128,
    S2RLoader,
    S2RLoaderTr,
    StoreCCast,
    ceildiv,
    combine_block_scales,
    compute_global_swizzle,
    compute_global_swizzle_nn,
    make_fp8_buffer_tensor,
    make_value_attrs,
    mask_a_tail,
    mfma_scaled_accumulate,
    wait_barrier,
    xcd_remap_pid,
)
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith
from flydsl.expr import range_constexpr, rocdl

# isort: on

_SCALE_BLOCK = 128


@functools.lru_cache(maxsize=128)
def _compile_blockwise_nt(
    K: int,
    BLOCK_M: int = 256,
    BLOCK_N: int = 256,
    GROUP_M: int = 4,
    num_xcd: int = 8,
    waves_per_eu: int = 2,
    agpr_alloc: int = 0,
    out_fp16: bool = False,
):
    """Forward / NT blockwise kernel. A [M,K], B_T [N,K], C [M,N], CON=K.

    A-scale [scale_k, M] (output-row minor); B-scale [N//128, K//128] 2-D weight
    blocks. Both consumed via runtime-indexed BlockScaleReader."""
    BLOCK_K = _SCALE_BLOCK
    assert BLOCK_M >= 128 and BLOCK_N >= 256 and BLOCK_M % 128 == 0 and BLOCK_N % 256 == 0
    assert K % BLOCK_K == 0, "blockwise GEMM requires K divisible by 128"

    K_ITERS = K // BLOCK_K
    K_TAIL = 0  # K % 128 == 0 enforced above
    SCALE_CON = K // _SCALE_BLOCK
    assert K_ITERS >= 2, "blockwise kernel needs K >= 256 (K_ITERS >= 2)"

    N_TILES_A = BLOCK_M // 64
    N_TILES_B = BLOCK_N // 128
    N_ACCUMS = N_TILES_A * N_TILES_B
    LDS_BLOCK_M = BLOCK_M // 2
    LDS_BLOCK_N = BLOCK_N // 2
    N_LDS_STEPS_A = LDS_BLOCK_M // 64
    N_LDS_STEPS_B = LDS_BLOCK_N // 64
    N_LDS_ROUNDS = max(N_LDS_STEPS_A, N_LDS_STEPS_B)
    a_lds_size = LDS_BLOCK_M * BLOCK_K
    b_lds_size = LDS_BLOCK_N * BLOCK_K

    @fx.struct
    class SharedStorage:
        A_lds_cur_0: fx.Array[fx.Float8E4M3FN, a_lds_size, 16]
        A_lds_cur_1: fx.Array[fx.Float8E4M3FN, a_lds_size, 16]
        A_lds_next_0: fx.Array[fx.Float8E4M3FN, a_lds_size, 16]
        A_lds_next_1: fx.Array[fx.Float8E4M3FN, a_lds_size, 16]
        B_lds_cur_0: fx.Array[fx.Float8E4M3FN, b_lds_size, 16]
        B_lds_cur_1: fx.Array[fx.Float8E4M3FN, b_lds_size, 16]
        B_lds_next_0: fx.Array[fx.Float8E4M3FN, b_lds_size, 16]
        B_lds_next_1: fx.Array[fx.Float8E4M3FN, b_lds_size, 16]

    @flyc.kernel(known_block_size=[512, 1, 1])
    def kernel_blockwise_nt(
        A: fx.Tensor,
        B_T: fx.Tensor,
        C: fx.Tensor,
        A_scale: fx.Tensor,
        B_scale: fx.Tensor,
        c_m: fx.Int32,
        c_n: fx.Int32,
    ):
        F8_IR_t = fx.Float8E4M3FN.ir_type
        n_blocks = ceildiv(c_n, BLOCK_N)

        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        a_cur0 = lds.A_lds_cur_0
        a_cur1 = lds.A_lds_cur_1
        a_next0 = lds.A_lds_next_0
        a_next1 = lds.A_lds_next_1
        b_cur0 = lds.B_lds_cur_0
        b_cur1 = lds.B_lds_cur_1
        b_next0 = lds.B_lds_next_0
        b_next1 = lds.B_lds_next_1

        lane_id = fx.thread_idx.x % 64
        wave_id = fx.thread_idx.x // 64
        wave_m = wave_id // 4
        wave_n = wave_id % 4
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
        B1_gl_offset = (block_n * BLOCK_N + LDS_BLOCK_N) * K

        gA = make_fp8_buffer_tensor(A, F8_IR_t)
        gB = make_fp8_buffer_tensor(B_T, F8_IR_t)
        a_div = fx.logical_divide(gA, fx.make_layout(1, 1))
        b_div = fx.logical_divide(gB, fx.make_layout(1, 1))

        gl_off_a = compute_global_swizzle(lane_id, wave_id, K, N_LDS_ROUNDS, preshuffled=False)
        gl_off_b = compute_global_swizzle(lane_id, wave_id, K, N_LDS_ROUNDS, preshuffled=False)

        mfma = Mfma16x16x128(N_TILES_A, N_TILES_B)
        a_g2s = G2SLoader(a_div, gl_off_a, N_LDS_STEPS_A, F8_IR_t, wave_id)
        b_g2s = G2SLoader(b_div, gl_off_b, N_LDS_STEPS_B, F8_IR_t, wave_id)
        a_s2r = S2RLoader(wave_m, N_TILES_A)
        b_s2r = S2RLoader(wave_n, N_TILES_B)
        _out_ty = fx.Float16 if out_fp16 else fx.BFloat16
        store_c = StoreCCast(C, c_m, c_n, mfma.idx, N_TILES_A, N_TILES_B, _out_ty)

        # Scale readers. A-scale [SCALE_CON, c_m]; B-scale [N//128, SCALE_CON].
        a_sc = BlockScaleReader(A_scale, SCALE_CON * c_m * 4)
        b_sc = BlockScaleReader(B_scale, ceildiv(c_n, _SCALE_BLOCK) * SCALE_CON * 4)

        # Output-tile origin (matches StoreCCast row/col mapping).
        base_row = block_m * BLOCK_M + wave_m * (N_TILES_A * 16)
        base_col = block_n * BLOCK_N + wave_n * (N_TILES_B * 16)
        row_lane = (lane_id // 16) * 4

        def quad_scales(kb, m_off, n_off):
            a_vecs = []
            for ti in range_constexpr(N_TILES_A):
                row = base_row + m_off + ti * 16 + row_lane
                a_vecs.append(a_sc.load4(kb * c_m + row))
            b_scalars = []
            for ni in range_constexpr(N_TILES_B):
                n_blk = (base_col + n_off + ni * 16) // _SCALE_BLOCK
                b_scalars.append(b_sc.load1(n_blk * SCALE_CON + kb))
            return combine_block_scales(a_vecs, b_scalars, N_TILES_A, N_TILES_B)

        c00 = [mfma.zero_value] * N_ACCUMS
        c01 = [mfma.zero_value] * N_ACCUMS
        c10 = [mfma.zero_value] * N_ACCUMS
        c11 = [mfma.zero_value] * N_ACCUMS

        # Prelude: k=0 -> cur, k=1 -> next.
        b_g2s.load(b_cur0, B0_gl_offset + 0 * BLOCK_K)
        a_g2s.load(a_cur0, A0_gl_offset + 0 * BLOCK_K)
        b_g2s.load(b_cur1, B1_gl_offset + 0 * BLOCK_K)
        a_g2s.load(a_cur1, A1_gl_offset + 0 * BLOCK_K)

        if wave_m == 1:
            rocdl.s_barrier()
        wait_barrier(N_LDS_STEPS_A + N_LDS_STEPS_B)

        b_g2s.load(b_next0, B0_gl_offset + 1 * BLOCK_K)
        a_g2s.load(a_next0, A0_gl_offset + 1 * BLOCK_K)
        b_g2s.load(b_next1, B1_gl_offset + 1 * BLOCK_K)

        wait_barrier(N_LDS_STEPS_A + 2 * N_LDS_STEPS_B)

        for k in range_constexpr(K_ITERS - 2):
            s00 = quad_scales(k, 0, 0)
            s01 = quad_scales(k, 0, LDS_BLOCK_N)
            s10 = quad_scales(k, LDS_BLOCK_M, 0)
            s11 = quad_scales(k, LDS_BLOCK_M, LDS_BLOCK_N)

            b0_frag = b_s2r.load(b_cur0)
            a0_frag = a_s2r.load(a_cur0)
            a_g2s.load(a_next1, A1_gl_offset + (k + 1) * BLOCK_K)
            rocdl.s_barrier()

            rocdl.s_setprio(1)
            c00 = mfma_scaled_accumulate(mfma, a0_frag, b0_frag, c00, s00)
            rocdl.s_setprio(0)
            rocdl.s_barrier()

            b1_frag = b_s2r.load(b_cur1)
            b_g2s.load(b_cur0, B0_gl_offset + (k + 2) * BLOCK_K)
            rocdl.s_barrier()

            rocdl.s_setprio(1)
            c01 = mfma_scaled_accumulate(mfma, a0_frag, b1_frag, c01, s01)
            rocdl.s_setprio(0)
            rocdl.s_barrier()

            a1_frag = a_s2r.load(a_cur1)
            a_g2s.load(a_cur0, A0_gl_offset + (k + 2) * BLOCK_K)
            rocdl.s_barrier()

            rocdl.s_setprio(1)
            c10 = mfma_scaled_accumulate(mfma, a1_frag, b0_frag, c10, s10)
            rocdl.s_setprio(0)
            rocdl.s_barrier()

            b_g2s.load(b_cur1, B1_gl_offset + (k + 2) * BLOCK_K)
            wait_barrier(2 * N_LDS_STEPS_A + N_LDS_STEPS_B)

            rocdl.s_setprio(1)
            c11 = mfma_scaled_accumulate(mfma, a1_frag, b1_frag, c11, s11)
            rocdl.s_setprio(0)
            rocdl.s_barrier()

            a_cur0, a_next0 = a_next0, a_cur0
            a_cur1, a_next1 = a_next1, a_cur1
            b_cur0, b_next0 = b_next0, b_cur0
            b_cur1, b_next1 = b_next1, b_cur1

        # Epilog 1 (k = K_ITERS - 2).
        k = K_ITERS - 2
        s00 = quad_scales(k, 0, 0)
        s01 = quad_scales(k, 0, LDS_BLOCK_N)
        s10 = quad_scales(k, LDS_BLOCK_M, 0)
        s11 = quad_scales(k, LDS_BLOCK_M, LDS_BLOCK_N)

        b0_frag = b_s2r.load(b_cur0)
        a0_frag = a_s2r.load(a_cur0)
        rocdl.s_barrier()
        rocdl.s_setprio(1)
        c00 = mfma_scaled_accumulate(mfma, a0_frag, b0_frag, c00, s00)
        rocdl.s_setprio(0)
        rocdl.s_barrier()

        b1_frag = b_s2r.load(b_cur1)
        rocdl.s_barrier()
        rocdl.s_setprio(1)
        c01 = mfma_scaled_accumulate(mfma, a0_frag, b1_frag, c01, s01)
        rocdl.s_setprio(0)
        rocdl.s_barrier()

        a1_frag = a_s2r.load(a_cur1)
        rocdl.s_barrier()
        rocdl.s_setprio(1)
        c10 = mfma_scaled_accumulate(mfma, a1_frag, b0_frag, c10, s10)
        rocdl.s_setprio(0)
        rocdl.s_barrier()

        b0_frag = b_s2r.load(b_next0)
        a_g2s.load(a_next1, A1_gl_offset + (k + 1) * BLOCK_K)
        rocdl.s_barrier()
        rocdl.s_setprio(1)
        c11 = mfma_scaled_accumulate(mfma, a1_frag, b1_frag, c11, s11)
        rocdl.s_setprio(0)
        rocdl.s_barrier()

        a_cur0, a_next0 = a_next0, a_cur0
        a_cur1, a_next1 = a_next1, a_cur1
        b_cur0, b_next0 = b_next0, b_cur0
        b_cur1, b_next1 = b_next1, b_cur1

        # Epilog 2 (k = K_ITERS - 1) -- the K-tail block (no-op mask when K_TAIL=0).
        k = K_ITERS - 1
        s00 = quad_scales(k, 0, 0)
        s01 = quad_scales(k, 0, LDS_BLOCK_N)
        s10 = quad_scales(k, LDS_BLOCK_M, 0)
        s11 = quad_scales(k, LDS_BLOCK_M, LDS_BLOCK_N)

        a0_frag = a_s2r.load(a_cur0)
        a0_frag = mask_a_tail(a0_frag, lane_id, K_TAIL)
        wait_barrier(0)
        rocdl.s_setprio(1)
        c00 = mfma_scaled_accumulate(mfma, a0_frag, b0_frag, c00, s00)
        rocdl.s_setprio(0)
        rocdl.s_barrier()

        b1_frag = b_s2r.load(b_cur1)
        rocdl.s_barrier()
        rocdl.s_setprio(1)
        c01 = mfma_scaled_accumulate(mfma, a0_frag, b1_frag, c01, s01)
        rocdl.s_setprio(0)
        rocdl.s_barrier()

        a1_frag = a_s2r.load(a_cur1)
        a1_frag = mask_a_tail(a1_frag, lane_id, K_TAIL)
        rocdl.s_barrier()
        rocdl.s_setprio(1)
        c10 = mfma_scaled_accumulate(mfma, a1_frag, b0_frag, c10, s10)
        c11 = mfma_scaled_accumulate(mfma, a1_frag, b1_frag, c11, s11)
        rocdl.s_setprio(0)
        rocdl.s_barrier()

        store_c.store(c00, base_row + 0, base_col + 0)
        store_c.store(c01, base_row + 0, base_col + LDS_BLOCK_N)
        store_c.store(c10, base_row + LDS_BLOCK_M, base_col + 0)
        store_c.store(c11, base_row + LDS_BLOCK_M, base_col + LDS_BLOCK_N)

    @flyc.jit
    def launch_blockwise_nt(
        A: fx.Tensor,
        B_T: fx.Tensor,
        C: fx.Tensor,
        A_scale: fx.Tensor,
        B_scale: fx.Tensor,
        c_m: fx.Int32,
        c_n: fx.Int32,
        stream: fx.Stream,
    ):
        grid_x = ceildiv(c_m, BLOCK_M) * ceildiv(c_n, BLOCK_N)
        kernel_blockwise_nt(
            A,
            B_T,
            C,
            A_scale,
            B_scale,
            c_m,
            c_n,
            value_attrs=make_value_attrs(waves_per_eu, agpr_alloc, "512,512"),
        ).launch(grid=(grid_x, 1, 1), block=(512, 1, 1), stream=stream)

    return launch_blockwise_nt


# ──────────────────────────────────────────────────────────────────────


@functools.lru_cache(maxsize=128)
def _compile_blockwise_nn(
    K: int,
    BLOCK_M: int = 256,
    BLOCK_N: int = 256,
    GROUP_M: int = 4,
    num_xcd: int = 8,
    waves_per_eu: int = 2,
    agpr_alloc: int = 0,
    out_fp16: bool = False,
):
    """dgrad / NN blockwise kernel. A [M,CON], B [CON,N], C [M,N], CON is the
    contraction (= the forward N). ``K`` here is the contraction length.

    A-scale [scale_con, M] (output-row minor); B-scale is the forward weight's
    2-D block grid [scale_con, N//128] indexed contraction-major (the launcher
    transposes the [N//128, K//128] weight scale to [K//128, N//128])."""
    BLOCK_K = _SCALE_BLOCK
    assert BLOCK_M >= 128 and BLOCK_N >= 256 and BLOCK_M % 128 == 0 and BLOCK_N % 256 == 0
    assert K % BLOCK_K == 0, "blockwise GEMM requires contraction divisible by 128"

    K_ITERS = K // BLOCK_K
    K_TAIL = 0
    SCALE_CON = K // _SCALE_BLOCK
    assert K_ITERS >= 2

    N_TILES_A = BLOCK_M // 64
    N_TILES_B = BLOCK_N // 128
    N_ACCUMS = N_TILES_A * N_TILES_B
    LDS_BLOCK_M = BLOCK_M // 2
    LDS_BLOCK_N = BLOCK_N // 2
    N_LDS_STEPS_A = LDS_BLOCK_M // 64
    N_LDS_STEPS_B = LDS_BLOCK_N // 64
    N_LDS_ROUNDS = max(N_LDS_STEPS_A, N_LDS_STEPS_B)
    a_lds_size = LDS_BLOCK_M * BLOCK_K
    b_lds_size = LDS_BLOCK_N * BLOCK_K

    @fx.struct
    class SharedStorage:
        A_lds_cur_0: fx.Array[fx.Float8E4M3FN, a_lds_size, 16]
        A_lds_cur_1: fx.Array[fx.Float8E4M3FN, a_lds_size, 16]
        A_lds_next_0: fx.Array[fx.Float8E4M3FN, a_lds_size, 16]
        A_lds_next_1: fx.Array[fx.Float8E4M3FN, a_lds_size, 16]
        B_lds_cur_0: fx.Array[fx.Float8E4M3FN, b_lds_size, 16]
        B_lds_cur_1: fx.Array[fx.Float8E4M3FN, b_lds_size, 16]
        B_lds_next_0: fx.Array[fx.Float8E4M3FN, b_lds_size, 16]
        B_lds_next_1: fx.Array[fx.Float8E4M3FN, b_lds_size, 16]

    @flyc.kernel(known_block_size=[512, 1, 1])
    def kernel_blockwise_nn(
        A: fx.Tensor,
        B: fx.Tensor,
        C: fx.Tensor,
        A_scale: fx.Tensor,
        B_scale: fx.Tensor,
        c_m: fx.Int32,
        c_n: fx.Int32,
    ):
        _ = str(fx.thread_idx.x)
        F8_IR_t = fx.Float8E4M3FN.ir_type
        n_blocks = ceildiv(c_n, BLOCK_N)

        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        a_cur0 = lds.A_lds_cur_0
        a_cur1 = lds.A_lds_cur_1
        a_next0 = lds.A_lds_next_0
        a_next1 = lds.A_lds_next_1
        b_cur0 = lds.B_lds_cur_0
        b_cur1 = lds.B_lds_cur_1
        b_next0 = lds.B_lds_next_0
        b_next1 = lds.B_lds_next_1

        lane_id = fx.thread_idx.x % 64
        wave_id = fx.thread_idx.x // 64
        wave_m = wave_id // 4
        wave_n = wave_id % 4
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
        B0_gl_offset = block_n * BLOCK_N + 0
        B1_gl_offset = block_n * BLOCK_N + LDS_BLOCK_N

        gA = make_fp8_buffer_tensor(A, F8_IR_t)
        gB = make_fp8_buffer_tensor(B, F8_IR_t)
        a_div = fx.logical_divide(gA, fx.make_layout(1, 1))
        b_div = fx.logical_divide(gB, fx.make_layout(1, 1))

        gl_off_a = compute_global_swizzle(lane_id, wave_id, K, N_LDS_ROUNDS, preshuffled=False)
        gl_off_b = compute_global_swizzle_nn(lane_id, wave_id, c_n, N_LDS_ROUNDS)

        mfma = Mfma16x16x128(N_TILES_A, N_TILES_B)
        a_g2s = G2SLoader(a_div, gl_off_a, N_LDS_STEPS_A, F8_IR_t, wave_id)
        b_g2s = G2SLoader(b_div, gl_off_b, N_LDS_STEPS_B, F8_IR_t, wave_id)
        a_s2r = S2RLoader(wave_m, N_TILES_A)
        b_s2r = S2RLoaderTr(wave_n, N_TILES_B, 32, inline_asm=False)
        _out_ty = fx.Float16 if out_fp16 else fx.BFloat16
        store_c = StoreCCast(C, c_m, c_n, mfma.idx, N_TILES_A, N_TILES_B, _out_ty)

        out_blocks_n = ceildiv(c_n, _SCALE_BLOCK)
        a_sc = BlockScaleReader(A_scale, SCALE_CON * c_m * 4)
        b_sc = BlockScaleReader(B_scale, SCALE_CON * out_blocks_n * 4)

        base_row = block_m * BLOCK_M + wave_m * (N_TILES_A * 16)
        base_col = block_n * BLOCK_N + wave_n * (N_TILES_B * 16)
        row_lane = (lane_id // 16) * 4

        def quad_scales(kb, m_off, n_off):
            a_vecs = []
            for ti in range_constexpr(N_TILES_A):
                row = base_row + m_off + ti * 16 + row_lane
                a_vecs.append(a_sc.load4(kb * c_m + row))
            b_scalars = []
            for ni in range_constexpr(N_TILES_B):
                out_blk = (base_col + n_off + ni * 16) // _SCALE_BLOCK
                b_scalars.append(b_sc.load1(kb * out_blocks_n + out_blk))
            return combine_block_scales(a_vecs, b_scalars, N_TILES_A, N_TILES_B)

        c00 = [mfma.zero_value] * N_ACCUMS
        c01 = [mfma.zero_value] * N_ACCUMS
        c10 = [mfma.zero_value] * N_ACCUMS
        c11 = [mfma.zero_value] * N_ACCUMS

        b_g2s.load(b_cur0, B0_gl_offset + 0 * BLOCK_K * c_n)
        a_g2s.load(a_cur0, A0_gl_offset + 0 * BLOCK_K)
        b_g2s.load(b_cur1, B1_gl_offset + 0 * BLOCK_K * c_n)
        a_g2s.load(a_cur1, A1_gl_offset + 0 * BLOCK_K)

        if wave_m == 1:
            rocdl.s_barrier()
        wait_barrier(N_LDS_STEPS_A + N_LDS_STEPS_B)

        b_g2s.load(b_next0, B0_gl_offset + 1 * BLOCK_K * c_n)
        a_g2s.load(a_next0, A0_gl_offset + 1 * BLOCK_K)
        b_g2s.load(b_next1, B1_gl_offset + 1 * BLOCK_K * c_n)

        wait_barrier(N_LDS_STEPS_A + 2 * N_LDS_STEPS_B)

        for k in range_constexpr(K_ITERS - 2):
            s00 = quad_scales(k, 0, 0)
            s01 = quad_scales(k, 0, LDS_BLOCK_N)
            s10 = quad_scales(k, LDS_BLOCK_M, 0)
            s11 = quad_scales(k, LDS_BLOCK_M, LDS_BLOCK_N)

            b0_frag = b_s2r.load(b_cur0)
            a0_frag = a_s2r.load(a_cur0)
            a_g2s.load(a_next1, A1_gl_offset + (k + 1) * BLOCK_K)
            rocdl.s_barrier()
            rocdl.s_setprio(1)
            c00 = mfma_scaled_accumulate(mfma, a0_frag, b0_frag, c00, s00)
            rocdl.s_setprio(0)
            rocdl.s_barrier()

            b1_frag = b_s2r.load(b_cur1)
            b_g2s.load(b_cur0, B0_gl_offset + (k + 2) * BLOCK_K * c_n)
            rocdl.s_barrier()
            rocdl.s_setprio(1)
            c01 = mfma_scaled_accumulate(mfma, a0_frag, b1_frag, c01, s01)
            rocdl.s_setprio(0)
            rocdl.s_barrier()

            a1_frag = a_s2r.load(a_cur1)
            a_g2s.load(a_cur0, A0_gl_offset + (k + 2) * BLOCK_K)
            rocdl.s_barrier()
            rocdl.s_setprio(1)
            c10 = mfma_scaled_accumulate(mfma, a1_frag, b0_frag, c10, s10)
            rocdl.s_setprio(0)
            rocdl.s_barrier()

            b_g2s.load(b_cur1, B1_gl_offset + (k + 2) * BLOCK_K * c_n)
            wait_barrier(2 * N_LDS_STEPS_A + N_LDS_STEPS_B)
            rocdl.s_setprio(1)
            c11 = mfma_scaled_accumulate(mfma, a1_frag, b1_frag, c11, s11)
            rocdl.s_setprio(0)
            rocdl.s_barrier()

            a_cur0, a_next0 = a_next0, a_cur0
            a_cur1, a_next1 = a_next1, a_cur1
            b_cur0, b_next0 = b_next0, b_cur0
            b_cur1, b_next1 = b_next1, b_cur1

        # Epilog 1.
        k = K_ITERS - 2
        s00 = quad_scales(k, 0, 0)
        s01 = quad_scales(k, 0, LDS_BLOCK_N)
        s10 = quad_scales(k, LDS_BLOCK_M, 0)
        s11 = quad_scales(k, LDS_BLOCK_M, LDS_BLOCK_N)

        b0_frag = b_s2r.load(b_cur0)
        a0_frag = a_s2r.load(a_cur0)
        rocdl.s_barrier()
        rocdl.s_setprio(1)
        c00 = mfma_scaled_accumulate(mfma, a0_frag, b0_frag, c00, s00)
        rocdl.s_setprio(0)
        rocdl.s_barrier()

        b1_frag = b_s2r.load(b_cur1)
        rocdl.s_barrier()
        rocdl.s_setprio(1)
        c01 = mfma_scaled_accumulate(mfma, a0_frag, b1_frag, c01, s01)
        rocdl.s_setprio(0)
        rocdl.s_barrier()

        a1_frag = a_s2r.load(a_cur1)
        rocdl.s_barrier()
        rocdl.s_setprio(1)
        c10 = mfma_scaled_accumulate(mfma, a1_frag, b0_frag, c10, s10)
        rocdl.s_setprio(0)
        rocdl.s_barrier()

        b0_frag = b_s2r.load(b_next0)
        a_g2s.load(a_next1, A1_gl_offset + (k + 1) * BLOCK_K)
        rocdl.s_barrier()
        rocdl.s_setprio(1)
        c11 = mfma_scaled_accumulate(mfma, a1_frag, b1_frag, c11, s11)
        rocdl.s_setprio(0)
        rocdl.s_barrier()

        a_cur0, a_next0 = a_next0, a_cur0
        a_cur1, a_next1 = a_next1, a_cur1
        b_cur0, b_next0 = b_next0, b_cur0
        b_cur1, b_next1 = b_next1, b_cur1

        # Epilog 2 -- K-tail block (no-op mask when K_TAIL=0).
        k = K_ITERS - 1
        s00 = quad_scales(k, 0, 0)
        s01 = quad_scales(k, 0, LDS_BLOCK_N)
        s10 = quad_scales(k, LDS_BLOCK_M, 0)
        s11 = quad_scales(k, LDS_BLOCK_M, LDS_BLOCK_N)

        a0_frag = a_s2r.load(a_cur0)
        a0_frag = mask_a_tail(a0_frag, lane_id, K_TAIL)
        wait_barrier(0)
        rocdl.s_setprio(1)
        c00 = mfma_scaled_accumulate(mfma, a0_frag, b0_frag, c00, s00)
        rocdl.s_setprio(0)
        rocdl.s_barrier()

        b1_frag = b_s2r.load(b_cur1)
        rocdl.s_barrier()
        rocdl.s_setprio(1)
        c01 = mfma_scaled_accumulate(mfma, a0_frag, b1_frag, c01, s01)
        rocdl.s_setprio(0)
        rocdl.s_barrier()

        a1_frag = a_s2r.load(a_cur1)
        a1_frag = mask_a_tail(a1_frag, lane_id, K_TAIL)
        rocdl.s_barrier()
        rocdl.s_setprio(1)
        c10 = mfma_scaled_accumulate(mfma, a1_frag, b0_frag, c10, s10)
        c11 = mfma_scaled_accumulate(mfma, a1_frag, b1_frag, c11, s11)
        rocdl.s_setprio(0)
        rocdl.s_barrier()

        store_c.store(c00, base_row + 0, base_col + 0)
        store_c.store(c01, base_row + 0, base_col + LDS_BLOCK_N)
        store_c.store(c10, base_row + LDS_BLOCK_M, base_col + 0)
        store_c.store(c11, base_row + LDS_BLOCK_M, base_col + LDS_BLOCK_N)

    @flyc.jit
    def launch_blockwise_nn(
        A: fx.Tensor,
        B: fx.Tensor,
        C: fx.Tensor,
        A_scale: fx.Tensor,
        B_scale: fx.Tensor,
        c_m: fx.Int32,
        c_n: fx.Int32,
        stream: fx.Stream,
    ):
        grid_x = ceildiv(c_m, BLOCK_M) * ceildiv(c_n, BLOCK_N)
        kernel_blockwise_nn(
            A,
            B,
            C,
            A_scale,
            B_scale,
            c_m,
            c_n,
            value_attrs=make_value_attrs(waves_per_eu, agpr_alloc, "512,512"),
        ).launch(grid=(grid_x, 1, 1), block=(512, 1, 1), stream=stream)

    return launch_blockwise_nn


# ──────────────────────────────────────────────────────────────────────


@functools.lru_cache(maxsize=128)
def _compile_blockwise_tn(
    K: int,
    BLOCK_M: int = 256,
    BLOCK_N: int = 256,
    GROUP_M: int = 4,
    num_xcd: int = 8,
    waves_per_eu: int = 2,
    out_fp16: bool = False,
):
    """wgrad / TN blockwise kernel. A [CON,M], B [CON,N], C [M,N] = A^T @ B; CON
    is the contraction (= the forward M). ``K`` here is the contraction length.

    This is the 1-D x 1-D blockwise case: both operands are column-quantized
    along the contraction, so A-scale [scale_con, M] and B-scale [scale_con, N]
    are both per-output-line (no 2-D weight grid). Both operands are
    contraction-row-major, so each goes through the ds_read_b64_tr_b8 wave-coop
    transpose load (the intrinsic, non-inline-asm path)."""
    BLOCK_K = _SCALE_BLOCK
    assert BLOCK_M >= 128 and BLOCK_N >= 256 and BLOCK_M % 128 == 0 and BLOCK_N % 256 == 0
    assert K % BLOCK_K == 0, "blockwise GEMM requires contraction divisible by 128"

    K_ITERS = K // BLOCK_K
    SCALE_CON = K // _SCALE_BLOCK
    assert K_ITERS >= 2

    N_TILES_A = BLOCK_M // 64
    N_TILES_B = BLOCK_N // 128
    N_ACCUMS = N_TILES_A * N_TILES_B
    LDS_BLOCK_M = BLOCK_M // 2
    LDS_BLOCK_N = BLOCK_N // 2
    # TN A path uses the wave-coop tr8 transpose load (K_log spans [0,128) -> 2
    # G2S rounds / 16K LDS slot); force >= 2 rounds for BM=128.
    N_LDS_STEPS_A = max(LDS_BLOCK_M // 64, 2)
    N_LDS_STEPS_B = LDS_BLOCK_N // 64
    N_LDS_ROUNDS = max(N_LDS_STEPS_A, N_LDS_STEPS_B)
    # Bank-spread LDS chunk stride (=1024+32); shared by the G2S writer and the
    # transpose-read S2RLoaderTr to remove transpose-read bank conflicts.
    _LDS_CS = 1056
    a_lds_size = max(LDS_BLOCK_M * BLOCK_K, 2 * 8 * 1024) // 1024 * _LDS_CS
    b_lds_size = (LDS_BLOCK_N * BLOCK_K) // 1024 * _LDS_CS

    @fx.struct
    class SharedStorage:
        A_lds_cur_0: fx.Array[fx.Float8E4M3FN, a_lds_size, 16]
        A_lds_cur_1: fx.Array[fx.Float8E4M3FN, a_lds_size, 16]
        A_lds_next_0: fx.Array[fx.Float8E4M3FN, a_lds_size, 16]
        A_lds_next_1: fx.Array[fx.Float8E4M3FN, a_lds_size, 16]
        B_lds_cur_0: fx.Array[fx.Float8E4M3FN, b_lds_size, 16]
        B_lds_cur_1: fx.Array[fx.Float8E4M3FN, b_lds_size, 16]
        B_lds_next_0: fx.Array[fx.Float8E4M3FN, b_lds_size, 16]
        B_lds_next_1: fx.Array[fx.Float8E4M3FN, b_lds_size, 16]

    @flyc.kernel(known_block_size=[512, 1, 1])
    def kernel_blockwise_tn(
        A: fx.Tensor,
        B: fx.Tensor,
        C: fx.Tensor,
        A_scale: fx.Tensor,
        B_scale: fx.Tensor,
        c_m: fx.Int32,
        c_n: fx.Int32,
    ):
        _ = str(fx.thread_idx.x)
        F8_IR_t = fx.Float8E4M3FN.ir_type
        n_blocks = ceildiv(c_n, BLOCK_N)

        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        a_cur0 = lds.A_lds_cur_0
        a_cur1 = lds.A_lds_cur_1
        b_cur0 = lds.B_lds_cur_0
        b_cur1 = lds.B_lds_cur_1
        a_next0 = lds.A_lds_next_0
        a_next1 = lds.A_lds_next_1
        b_next0 = lds.B_lds_next_0
        b_next1 = lds.B_lds_next_1

        lane_id = fx.thread_idx.x % 64
        wave_id = fx.thread_idx.x // 64
        wave_m = wave_id // 4
        wave_n = wave_id % 4
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

        # A [CON, M] row-major: stride c_m per contraction row.
        A0_gl_offset = block_m * BLOCK_M + 0
        A1_gl_offset = block_m * BLOCK_M + LDS_BLOCK_M
        # B [CON, N] row-major: stride c_n per contraction row.
        B0_gl_offset = block_n * BLOCK_N + 0
        B1_gl_offset = block_n * BLOCK_N + LDS_BLOCK_N

        gA = make_fp8_buffer_tensor(A, F8_IR_t)
        gB = make_fp8_buffer_tensor(B, F8_IR_t)
        a_div = fx.logical_divide(gA, fx.make_layout(1, 1))
        b_div = fx.logical_divide(gB, fx.make_layout(1, 1))

        gl_off_a = compute_global_swizzle_nn(lane_id, wave_id, c_m, N_LDS_ROUNDS)
        gl_off_b = compute_global_swizzle_nn(lane_id, wave_id, c_n, N_LDS_ROUNDS)

        mfma = Mfma16x16x128(N_TILES_A, N_TILES_B)
        a_g2s = G2SLoader(a_div, gl_off_a, N_LDS_STEPS_A, F8_IR_t, wave_id, chunk_stride=_LDS_CS)
        b_g2s = G2SLoader(b_div, gl_off_b, N_LDS_STEPS_B, F8_IR_t, wave_id, chunk_stride=_LDS_CS)
        a_s2r = S2RLoaderTr(wave_m, N_TILES_A, LDS_BLOCK_M // 2, inline_asm=False, chunk_stride=_LDS_CS)
        b_s2r = S2RLoaderTr(wave_n, N_TILES_B, 32, inline_asm=False, chunk_stride=_LDS_CS)
        _out_ty = fx.Float16 if out_fp16 else fx.BFloat16
        store_c = StoreCCast(C, c_m, c_n, mfma.idx, N_TILES_A, N_TILES_B, _out_ty)

        # 1-D x 1-D scales (both per output line): A [scale_con, c_m],
        # B [scale_con, c_n] (per output column).
        a_sc = BlockScaleReader(A_scale, SCALE_CON * c_m * 4)
        b_sc = BlockScaleReader(B_scale, SCALE_CON * c_n * 4)

        base_row = block_m * BLOCK_M + wave_m * (N_TILES_A * 16)
        base_col = block_n * BLOCK_N + wave_n * (N_TILES_B * 16)
        row_lane = (lane_id // 16) * 4
        col_lane = lane_id % 16

        def quad_scales(kb, m_off, n_off):
            a_vecs = []
            for ti in range_constexpr(N_TILES_A):
                row = base_row + m_off + ti * 16 + row_lane
                a_vecs.append(a_sc.load4(kb * c_m + row))
            b_scalars = []
            for ni in range_constexpr(N_TILES_B):
                out_col = base_col + n_off + ni * 16 + col_lane
                b_scalars.append(b_sc.load1(kb * c_n + out_col))
            return combine_block_scales(a_vecs, b_scalars, N_TILES_A, N_TILES_B)

        c00 = [mfma.zero_value] * N_ACCUMS
        c01 = [mfma.zero_value] * N_ACCUMS
        c10 = [mfma.zero_value] * N_ACCUMS
        c11 = [mfma.zero_value] * N_ACCUMS

        b_g2s.load(b_cur0, B0_gl_offset + 0 * BLOCK_K * c_n)
        a_g2s.load(a_cur0, A0_gl_offset + 0 * BLOCK_K * c_m)
        b_g2s.load(b_cur1, B1_gl_offset + 0 * BLOCK_K * c_n)
        a_g2s.load(a_cur1, A1_gl_offset + 0 * BLOCK_K * c_m)

        if wave_m == 1:
            rocdl.s_barrier()
        wait_barrier(N_LDS_STEPS_A + N_LDS_STEPS_B)

        b_g2s.load(b_next0, B0_gl_offset + 1 * BLOCK_K * c_n)
        a_g2s.load(a_next0, A0_gl_offset + 1 * BLOCK_K * c_m)
        b_g2s.load(b_next1, B1_gl_offset + 1 * BLOCK_K * c_n)

        wait_barrier(N_LDS_STEPS_A + 2 * N_LDS_STEPS_B)

        for k in range_constexpr(K_ITERS - 2):
            s00 = quad_scales(k, 0, 0)
            s01 = quad_scales(k, 0, LDS_BLOCK_N)
            s10 = quad_scales(k, LDS_BLOCK_M, 0)
            s11 = quad_scales(k, LDS_BLOCK_M, LDS_BLOCK_N)

            b0_frag = b_s2r.load(b_cur0, drain=False)
            a0_frag = a_s2r.load(a_cur0)
            a_g2s.load(a_next1, A1_gl_offset + (k + 1) * BLOCK_K * c_m)
            rocdl.s_barrier()
            rocdl.s_setprio(1)
            c00 = mfma_scaled_accumulate(mfma, a0_frag, b0_frag, c00, s00)
            rocdl.s_setprio(0)
            rocdl.s_barrier()

            b1_frag = b_s2r.load(b_cur1)
            b_g2s.load(b_cur0, B0_gl_offset + (k + 2) * BLOCK_K * c_n)
            rocdl.s_barrier()
            rocdl.s_setprio(1)
            c01 = mfma_scaled_accumulate(mfma, a0_frag, b1_frag, c01, s01)
            rocdl.s_setprio(0)
            rocdl.s_barrier()

            a1_frag = a_s2r.load(a_cur1)
            a_g2s.load(a_cur0, A0_gl_offset + (k + 2) * BLOCK_K * c_m)
            rocdl.s_barrier()
            rocdl.s_setprio(1)
            c10 = mfma_scaled_accumulate(mfma, a1_frag, b0_frag, c10, s10)
            rocdl.s_setprio(0)
            rocdl.s_barrier()

            b_g2s.load(b_cur1, B1_gl_offset + (k + 2) * BLOCK_K * c_n)
            wait_barrier(2 * N_LDS_STEPS_A + N_LDS_STEPS_B)
            rocdl.s_setprio(1)
            c11 = mfma_scaled_accumulate(mfma, a1_frag, b1_frag, c11, s11)
            rocdl.s_setprio(0)
            rocdl.s_barrier()

            a_cur0, a_next0 = a_next0, a_cur0
            a_cur1, a_next1 = a_next1, a_cur1
            b_cur0, b_next0 = b_next0, b_cur0
            b_cur1, b_next1 = b_next1, b_cur1

        # Epilog 1.
        k = K_ITERS - 2
        s00 = quad_scales(k, 0, 0)
        s01 = quad_scales(k, 0, LDS_BLOCK_N)
        s10 = quad_scales(k, LDS_BLOCK_M, 0)
        s11 = quad_scales(k, LDS_BLOCK_M, LDS_BLOCK_N)

        b0_frag = b_s2r.load(b_cur0)
        a0_frag = a_s2r.load(a_cur0)
        rocdl.s_barrier()
        rocdl.s_setprio(1)
        c00 = mfma_scaled_accumulate(mfma, a0_frag, b0_frag, c00, s00)
        rocdl.s_setprio(0)
        rocdl.s_barrier()

        b1_frag = b_s2r.load(b_cur1)
        rocdl.s_barrier()
        rocdl.s_setprio(1)
        c01 = mfma_scaled_accumulate(mfma, a0_frag, b1_frag, c01, s01)
        rocdl.s_setprio(0)
        rocdl.s_barrier()

        a1_frag = a_s2r.load(a_cur1)
        rocdl.s_barrier()
        rocdl.s_setprio(1)
        c10 = mfma_scaled_accumulate(mfma, a1_frag, b0_frag, c10, s10)
        rocdl.s_setprio(0)
        rocdl.s_barrier()

        b0_frag = b_s2r.load(b_next0)
        a_g2s.load(a_next1, A1_gl_offset + (k + 1) * BLOCK_K * c_m)
        rocdl.s_barrier()
        rocdl.s_setprio(1)
        c11 = mfma_scaled_accumulate(mfma, a1_frag, b1_frag, c11, s11)
        rocdl.s_setprio(0)
        rocdl.s_barrier()

        a_cur0, a_next0 = a_next0, a_cur0
        a_cur1, a_next1 = a_next1, a_cur1
        b_cur0, b_next0 = b_next0, b_cur0
        b_cur1, b_next1 = b_next1, b_cur1

        # Epilog 2 (final contraction block).
        k = K_ITERS - 1
        s00 = quad_scales(k, 0, 0)
        s01 = quad_scales(k, 0, LDS_BLOCK_N)
        s10 = quad_scales(k, LDS_BLOCK_M, 0)
        s11 = quad_scales(k, LDS_BLOCK_M, LDS_BLOCK_N)

        a0_frag = a_s2r.load(a_cur0)
        wait_barrier(0)
        rocdl.s_setprio(1)
        c00 = mfma_scaled_accumulate(mfma, a0_frag, b0_frag, c00, s00)
        rocdl.s_setprio(0)
        rocdl.s_barrier()

        b1_frag = b_s2r.load(b_cur1)
        rocdl.s_barrier()
        rocdl.s_setprio(1)
        c01 = mfma_scaled_accumulate(mfma, a0_frag, b1_frag, c01, s01)
        rocdl.s_setprio(0)
        rocdl.s_barrier()

        a1_frag = a_s2r.load(a_cur1)
        rocdl.s_barrier()
        rocdl.s_setprio(1)
        c10 = mfma_scaled_accumulate(mfma, a1_frag, b0_frag, c10, s10)
        c11 = mfma_scaled_accumulate(mfma, a1_frag, b1_frag, c11, s11)
        rocdl.s_setprio(0)
        rocdl.s_barrier()

        store_c.store(c00, base_row + 0, base_col + 0)
        store_c.store(c01, base_row + 0, base_col + LDS_BLOCK_N)
        store_c.store(c10, base_row + LDS_BLOCK_M, base_col + 0)
        store_c.store(c11, base_row + LDS_BLOCK_M, base_col + LDS_BLOCK_N)

    @flyc.jit
    def launch_blockwise_tn(
        A: fx.Tensor,
        B: fx.Tensor,
        C: fx.Tensor,
        A_scale: fx.Tensor,
        B_scale: fx.Tensor,
        c_m: fx.Int32,
        c_n: fx.Int32,
        stream: fx.Stream,
    ):
        grid_x = ceildiv(c_m, BLOCK_M) * ceildiv(c_n, BLOCK_N)
        kernel_blockwise_tn(
            A,
            B,
            C,
            A_scale,
            B_scale,
            c_m,
            c_n,
            value_attrs=make_value_attrs(waves_per_eu, 0, "512,512"),
        ).launch(grid=(grid_x, 1, 1), block=(512, 1, 1), stream=stream)

    return launch_blockwise_tn


# ──────────────────────────────────────────────────────────────────────
# Host wrapper: dispatch by (trans_a, trans_b), prepare scales, compile + run.
# ──────────────────────────────────────────────────────────────────────

_COMPILED_BLOCKWISE_CACHE: dict = {}


def _get_compiled(launch, args):
    """Cache the compiled launcher by (launch-id, shapes, dtypes, int args)."""
    key_parts = [id(launch)]
    for a in args:
        if isinstance(a, torch.Tensor):
            key_parts.append((tuple(a.shape), a.dtype))
        elif isinstance(a, int):
            key_parts.append(a)
        else:
            key_parts.append(type(a).__name__)
    key = tuple(key_parts)
    cached = _COMPILED_BLOCKWISE_CACHE.get(key)
    if cached is None:
        cached = flyc.compile(launch, *args)
        _COMPILED_BLOCKWISE_CACHE[key] = cached
    return cached


def _as_i8_flat(t: torch.Tensor) -> torch.Tensor:
    """Zero-copy flat int8 byte view of an fp8 tensor (matches the dense path)."""
    if t.element_size() == 1 and t.dtype != torch.int8:
        return t.contiguous().view(torch.int8).view(-1)
    return t.contiguous().view(-1)


def _f32_flat(t: torch.Tensor, transpose: bool) -> torch.Tensor:
    """Flat fp32 scale buffer; transpose=True maps a row-major [r, c] scale to
    [c, r] so the output line is the minor (contiguous) dim the kernel reads."""
    t = t.transpose(0, 1) if transpose else t
    return t.contiguous().to(torch.float32).view(-1)


def gemm_fp8_blockwise_flydsl_kernel(
    a: torch.Tensor,
    a_scale_inv: torch.Tensor,
    b: torch.Tensor,
    b_scale_inv: torch.Tensor,
    trans_a: bool = False,
    trans_b: bool = True,
    out_dtype: torch.dtype = torch.bfloat16,
    trans_c: bool = False,
) -> torch.Tensor:
    """Blockwise (per-128-block) FP8 GEMM, E4M3 -> bf16/fp16.

    Dispatch by (trans_a, trans_b), matching the three BLOCKWISE training calls:
      - NT (F, T): forward  out[M,N] = a[M,K] @ b[N,K]^T,  a 1-D / b 2-D scale
      - NN (F, F): dgrad     out[M,K] = grad_out[M,N] @ b[N,K], 1-D / 2-D scale
      - TN (T, F): wgrad     out[N,K] = grad_out[M,N]^T @ a[M,K], 1-D / 1-D scale
    ``trans_c=True`` (wgrad) returns the [N, K] transpose of the [K, N] kernel
    output."""
    if out_dtype not in (torch.bfloat16, torch.float16):
        raise NotImplementedError(f"blockwise FlyDSL emits bf16 or fp16, got {out_dtype}")
    assert a.dim() == 2 and b.dim() == 2
    out_fp16 = out_dtype == torch.float16
    stream = torch.cuda.current_stream()

    if (not trans_a) and trans_b:
        # NT forward: a [M,K] (1-D row scale), b [N,K] (2-D weight scale).
        M, K = a.shape
        N = b.shape[0]
        A_scale = _f32_flat(a_scale_inv, transpose=True)   # [K//128, M]
        B_scale = _f32_flat(b_scale_inv, transpose=False)  # [N//128, K//128]
        out = torch.empty((M, N), dtype=out_dtype, device=a.device)
        launch = _compile_blockwise_nt(K=K, out_fp16=out_fp16)
        args = (_as_i8_flat(a), _as_i8_flat(b), out.view(-1), A_scale, B_scale, M, N, stream)
        _get_compiled(launch, args)(*args)
        return out

    if (not trans_a) and (not trans_b):
        # NN dgrad: a=grad_out [M,N] (1-D row scale, contraction N), b [N,K]
        # (2-D weight scale, contraction-major as stored).
        M, N_con = a.shape
        K_out = b.shape[1]
        A_scale = _f32_flat(a_scale_inv, transpose=True)   # [N//128, M]
        B_scale = _f32_flat(b_scale_inv, transpose=False)  # [N//128, K//128]
        out = torch.empty((M, K_out), dtype=out_dtype, device=a.device)
        launch = _compile_blockwise_nn(K=N_con, out_fp16=out_fp16)
        args = (_as_i8_flat(a), _as_i8_flat(b), out.view(-1), A_scale, B_scale, M, K_out, stream)
        _get_compiled(launch, args)(*args)
        return out

    if trans_a and (not trans_b):
        # TN wgrad: a=a_col [M,K] (1-D col scale), b=grad_out_col [M,N] (1-D col
        # scale), contraction M. Kernel output [K,N]; trans_c -> [N,K].
        M_con, K_out = a.shape
        N_out = b.shape[1]
        A_scale = _f32_flat(a_scale_inv, transpose=False)  # [M//128, K]
        B_scale = _f32_flat(b_scale_inv, transpose=False)  # [M//128, N]
        out = torch.empty((K_out, N_out), dtype=out_dtype, device=a.device)
        launch = _compile_blockwise_tn(K=M_con, out_fp16=out_fp16)
        args = (_as_i8_flat(a), _as_i8_flat(b), out.view(-1), A_scale, B_scale, K_out, N_out, stream)
        _get_compiled(launch, args)(*args)
        return out.t().contiguous() if trans_c else out

    raise NotImplementedError(
        f"blockwise FlyDSL GEMM does not support trans_a={trans_a}, trans_b={trans_b}"
    )


__all__ = ["gemm_fp8_blockwise_flydsl_kernel"]
