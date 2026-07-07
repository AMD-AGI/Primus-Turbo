###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Primus-Turbo dense/grouped bf16 GEMM kernels (FlyDSL): NT, NN, TN + TN wgrad.
256-wide tile, BLOCK_K=64, 8-wave (wave_m=2 x wave_n=4), mfma_f32_32x32x16 (dense)
and mfma_f32_16x16x32 (variable-K wgrad), bf16/fp16 out, arbitrary K via native
K-tail. Compute/loader/store primitives are imported from flydsl.utils.gemm_helper
as module globals (@flyc.kernel needs its dependencies as globals)."""

import functools

import torch

# isort: off
# Primitives are vendored in flydsl/utils/gemm_helper.py and imported as module
# globals (mirrors gemm_fp8_kernel.py; the bf16 block sits next to the fp8 one).
from primus_turbo.flydsl.utils.gemm_helper import (
    BLOCK_K,
    G2SLoader,
    Mfma16x16x32,
    Mfma32x32x16,
    S2RLoaderBf16,
    S2RLoaderTr16x32Bf16,
    S2RLoaderTrBf16,
    StoreCBf16,
    ceildiv,
    compute_global_swizzle_bf16,
    compute_global_swizzle_nn_bf16,
    make_bf16_buffer_tensor_rebased,
    make_fp16_bf16_buffer_tensor,
    make_value_attrs,
    wait_barrier,
    xcd_remap_pid,
    _emit_for,
    _emit_if_then,
)
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir.dialects import llvm as _llvm
from flydsl.expr import arith
from flydsl.expr import const_expr, range_constexpr, rocdl
from flydsl.expr.typing import Vector as Vec
from flydsl.expr.utils.arith import ArithValue

# isort: on


WGRAD_WAVES = 8
WGRAD_BLOCK = WGRAD_WAVES * 64
WGRAD_AGPR = 0


def _i64(v):
    # widen an i32 runtime value to i64 (avoids overflow in worst-case base offsets)
    from flydsl.expr.buffer_ops import _unwrap_value

    return ArithValue(arith.extsi(fx.T.i64(), _unwrap_value(v)), signed=True)


def _gemm_geom(K, BLOCK_M, BLOCK_N):
    assert BLOCK_M >= 128 and BLOCK_N >= 256 and BLOCK_M % 128 == 0 and BLOCK_N % 256 == 0
    assert K % BLOCK_K == 0, f"bf16 NT needs K % {BLOCK_K} == 0 (got K={K})"
    K_ITERS = K // BLOCK_K
    assert K_ITERS >= 2, f"K_ITERS={K_ITERS} too small; need K >= {2 * BLOCK_K}"
    N_TILES_A = BLOCK_M // 128
    N_TILES_B = BLOCK_N // 256
    LDS_BLOCK_M = BLOCK_M // 2
    LDS_BLOCK_N = BLOCK_N // 2
    N_LDS_STEPS_A = LDS_BLOCK_M // 64
    N_LDS_STEPS_B = LDS_BLOCK_N // 64
    N_LDS_ROUNDS = max(N_LDS_STEPS_A, N_LDS_STEPS_B)
    return {
        "K_ITERS": K_ITERS,
        "N_TILES_A": N_TILES_A,
        "N_TILES_B": N_TILES_B,
        "N_ACCUMS": N_TILES_A * N_TILES_B,
        "LDS_BLOCK_M": LDS_BLOCK_M,
        "LDS_BLOCK_N": LDS_BLOCK_N,
        "N_LDS_STEPS_A": N_LDS_STEPS_A,
        "N_LDS_STEPS_B": N_LDS_STEPS_B,
        "N_LDS_ROUNDS": N_LDS_ROUNDS,
        "a_lds_size": LDS_BLOCK_M * BLOCK_K,
        "b_lds_size": LDS_BLOCK_N * BLOCK_K,
    }


def _make_shared_storage(BLOCK_M, BLOCK_N):
    a_lds_size = (BLOCK_M // 2) * BLOCK_K
    b_lds_size = (BLOCK_N // 2) * BLOCK_K

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
    out_fp16=False,
    nt_vmcnt=3,
    b_group_base=None,
    c_cache_modifier=0,
):
    g = _gemm_geom(K, BLOCK_M, BLOCK_N)
    K_ITERS = g["K_ITERS"]
    N_TILES_A = g["N_TILES_A"]
    N_TILES_B = g["N_TILES_B"]
    N_ACCUMS = g["N_ACCUMS"]
    LDS_BLOCK_M = g["LDS_BLOCK_M"]
    LDS_BLOCK_N = g["LDS_BLOCK_N"]
    N_LDS_STEPS_A = g["N_LDS_STEPS_A"]
    N_LDS_STEPS_B = g["N_LDS_STEPS_B"]
    N_LDS_ROUNDS = g["N_LDS_ROUNDS"]

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
    B1_gl_offset = (block_n * BLOCK_N + LDS_BLOCK_N) * K
    if b_group_base is not None:
        B0_gl_offset = B0_gl_offset + b_group_base
        B1_gl_offset = B1_gl_offset + b_group_base

    gA = make_fp16_bf16_buffer_tensor(A)
    gB = make_fp16_bf16_buffer_tensor(B_T)
    a_div = fx.logical_divide(gA, fx.make_layout(1, 1))
    b_div = fx.logical_divide(gB, fx.make_layout(1, 1))

    gl_off_a = compute_global_swizzle_bf16(lane_id, wave_id, K, N_LDS_ROUNDS)
    gl_off_b = compute_global_swizzle_bf16(lane_id, wave_id, K, N_LDS_ROUNDS)

    mfma = Mfma32x32x16(N_TILES_A, N_TILES_B)

    a_g2s = G2SLoader(a_div, gl_off_a, N_LDS_STEPS_A, fx.BFloat16.ir_type, wave_id)
    b_g2s = G2SLoader(b_div, gl_off_b, N_LDS_STEPS_B, fx.BFloat16.ir_type, wave_id)
    a_s2r = S2RLoaderBf16(wave_m, N_TILES_A)
    b_s2r = S2RLoaderBf16(wave_n, N_TILES_B)
    _out_ty = fx.Float16 if out_fp16 else fx.BFloat16
    store_c = StoreCBf16(C, c_m, c_n, _out_ty, cache_modifier=c_cache_modifier)

    c00_frag = [mfma.zero_value] * N_ACCUMS
    c01_frag = [mfma.zero_value] * N_ACCUMS
    c10_frag = [mfma.zero_value] * N_ACCUMS
    c11_frag = [mfma.zero_value] * N_ACCUMS

    b_g2s.load(b_cur0, B0_gl_offset + 0 * BLOCK_K)
    a_g2s.load(a_cur0, A0_gl_offset + 0 * BLOCK_K)
    b_g2s.load(b_cur1, B1_gl_offset + 0 * BLOCK_K)
    a_g2s.load(a_cur1, A1_gl_offset + 0 * BLOCK_K)

    _emit_if_then(wave_m == 1, lambda: rocdl.s_barrier())

    wait_barrier(N_LDS_STEPS_A + N_LDS_STEPS_B)

    b_g2s.load(b_next0, B0_gl_offset + 1 * BLOCK_K)
    a_g2s.load(a_next0, A0_gl_offset + 1 * BLOCK_K)
    b_g2s.load(b_next1, B1_gl_offset + 1 * BLOCK_K)

    wait_barrier(N_LDS_STEPS_A + 2 * N_LDS_STEPS_B)

    for k in range_constexpr(K_ITERS - 2):
        b0_frag = b_s2r.load(b_cur0)
        a0_frag = a_s2r.load(a_cur0)
        a_g2s.load(a_next1, A1_gl_offset + (k + 1) * BLOCK_K)
        rocdl.s_barrier()

        rocdl.s_setprio(1)
        c00_frag = mfma.call(a0_frag, b0_frag, c00_frag)
        rocdl.s_setprio(0)
        rocdl.s_barrier()

        b1_frag = b_s2r.load(b_cur1)
        b_g2s.load(b_cur0, B0_gl_offset + (k + 2) * BLOCK_K)
        rocdl.s_barrier()

        rocdl.s_setprio(1)
        c01_frag = mfma.call(a0_frag, b1_frag, c01_frag)
        rocdl.s_setprio(0)
        rocdl.s_barrier()

        a1_frag = a_s2r.load(a_cur1)
        a_g2s.load(a_cur0, A0_gl_offset + (k + 2) * BLOCK_K)
        rocdl.s_barrier()

        rocdl.s_setprio(1)
        c10_frag = mfma.call(a1_frag, b0_frag, c10_frag)
        rocdl.s_setprio(0)
        rocdl.s_barrier()

        b_g2s.load(b_cur1, B1_gl_offset + (k + 2) * BLOCK_K)
        wait_barrier(2 * N_LDS_STEPS_A + N_LDS_STEPS_B)

        rocdl.s_setprio(1)
        c11_frag = mfma.call(a1_frag, b1_frag, c11_frag)
        rocdl.s_setprio(0)
        rocdl.s_barrier()

        if nt_vmcnt >= 0:
            _llvm.inline_asm(
                res=None,
                operands_=[],
                asm_string=f"s_waitcnt vmcnt({nt_vmcnt})",
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

    rocdl.s_setprio(1)
    c00_frag = mfma.call(a0_frag, b0_frag, c00_frag)
    rocdl.s_setprio(0)
    rocdl.s_barrier()

    b1_frag = b_s2r.load(b_cur1)
    rocdl.s_barrier()

    rocdl.s_setprio(1)
    c01_frag = mfma.call(a0_frag, b1_frag, c01_frag)
    rocdl.s_setprio(0)
    rocdl.s_barrier()

    a1_frag = a_s2r.load(a_cur1)
    rocdl.s_barrier()

    rocdl.s_setprio(1)
    c10_frag = mfma.call(a1_frag, b0_frag, c10_frag)
    rocdl.s_setprio(0)
    rocdl.s_barrier()

    b0_frag = b_s2r.load(b_next0)
    a_g2s.load(a_next1, A1_gl_offset + (k + 1) * BLOCK_K)
    rocdl.s_barrier()

    rocdl.s_setprio(1)
    c11_frag = mfma.call(a1_frag, b1_frag, c11_frag)
    rocdl.s_setprio(0)
    rocdl.s_barrier()

    a_cur0, a_next0 = a_next0, a_cur0
    a_cur1, a_next1 = a_next1, a_cur1
    b_cur0, b_next0 = b_next0, b_cur0
    b_cur1, b_next1 = b_next1, b_cur1

    a0_frag = a_s2r.load(a_cur0)
    wait_barrier(0)

    rocdl.s_setprio(1)
    c00_frag = mfma.call(a0_frag, b0_frag, c00_frag)
    rocdl.s_setprio(0)
    rocdl.s_barrier()

    b1_frag = b_s2r.load(b_cur1)
    rocdl.s_barrier()

    rocdl.s_setprio(1)
    c01_frag = mfma.call(a0_frag, b1_frag, c01_frag)
    rocdl.s_setprio(0)
    rocdl.s_barrier()

    a1_frag = a_s2r.load(a_cur1)
    rocdl.s_barrier()

    rocdl.s_setprio(1)
    c10_frag = mfma.call(a1_frag, b0_frag, c10_frag)
    c11_frag = mfma.call(a1_frag, b1_frag, c11_frag)
    rocdl.s_setprio(0)
    rocdl.s_barrier()

    wave_n_offset = wave_n * (N_TILES_B * 32)
    wave_m_offset = wave_m * (N_TILES_A * 32)
    base_row = block_m * BLOCK_M + wave_m_offset
    base_col = block_n * BLOCK_N + wave_n_offset

    store_c.store(c00_frag, base_row + 0, base_col + 0)
    store_c.store(c01_frag, base_row + 0, base_col + LDS_BLOCK_N)
    store_c.store(c10_frag, base_row + LDS_BLOCK_M, base_col + 0)
    store_c.store(c11_frag, base_row + LDS_BLOCK_M, base_col + LDS_BLOCK_N)


def _nn_tn_tile(
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
    out_fp16=False,
    nt_vmcnt=3,
    b_group_base=None,
    c_cache_modifier=0,
):
    g = _gemm_geom(K, BLOCK_M, BLOCK_N)
    K_ITERS = g["K_ITERS"]
    N_TILES_A = g["N_TILES_A"]
    N_TILES_B = g["N_TILES_B"]
    N_ACCUMS = g["N_ACCUMS"]
    LDS_BLOCK_M = g["LDS_BLOCK_M"]
    LDS_BLOCK_N = g["LDS_BLOCK_N"]
    N_LDS_STEPS_A = g["N_LDS_STEPS_A"]
    N_LDS_STEPS_B = g["N_LDS_STEPS_B"]
    N_LDS_ROUNDS = g["N_LDS_ROUNDS"]

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
        gl_off_a = compute_global_swizzle_nn_bf16(lane_id, wave_id, c_m, N_LDS_STEPS_A)
    else:
        gl_off_a = compute_global_swizzle_bf16(lane_id, wave_id, K, N_LDS_ROUNDS)
    gl_off_b = compute_global_swizzle_nn_bf16(lane_id, wave_id, c_n, N_LDS_STEPS_B)

    mfma = Mfma32x32x16(N_TILES_A, N_TILES_B)
    a_g2s = G2SLoader(a_div, gl_off_a, N_LDS_STEPS_A, fx.BFloat16.ir_type, wave_id)
    b_g2s = G2SLoader(b_div, gl_off_b, N_LDS_STEPS_B, fx.BFloat16.ir_type, wave_id)
    a_s2r = S2RLoaderTrBf16(wave_m, N_TILES_A) if a_transpose else S2RLoaderBf16(wave_m, N_TILES_A)
    b_s2r = S2RLoaderTrBf16(wave_n, N_TILES_B)
    _out_ty = fx.Float16 if out_fp16 else fx.BFloat16
    store_c = StoreCBf16(C, c_m, c_n, _out_ty, cache_modifier=c_cache_modifier)

    c00_frag = [mfma.zero_value] * N_ACCUMS
    c01_frag = [mfma.zero_value] * N_ACCUMS
    c10_frag = [mfma.zero_value] * N_ACCUMS
    c11_frag = [mfma.zero_value] * N_ACCUMS

    b_g2s.load(b_cur0, B0_gl_offset + 0 * b_k_step)
    a_g2s.load(a_cur0, A0_gl_offset + 0 * a_k_step)
    b_g2s.load(b_cur1, B1_gl_offset + 0 * b_k_step)
    a_g2s.load(a_cur1, A1_gl_offset + 0 * a_k_step)

    _emit_if_then(wave_m == 1, lambda: rocdl.s_barrier())
    wait_barrier(N_LDS_STEPS_A + N_LDS_STEPS_B)

    b_g2s.load(b_next0, B0_gl_offset + 1 * b_k_step)
    a_g2s.load(a_next0, A0_gl_offset + 1 * a_k_step)
    b_g2s.load(b_next1, B1_gl_offset + 1 * b_k_step)

    wait_barrier(N_LDS_STEPS_A + 2 * N_LDS_STEPS_B)

    for k in range_constexpr(K_ITERS - 2):
        b0_frag = b_s2r.load(b_cur0)
        a0_frag = a_s2r.load(a_cur0)
        a_g2s.load(a_next1, A1_gl_offset + (k + 1) * a_k_step)
        rocdl.s_barrier()

        rocdl.s_setprio(1)
        c00_frag = mfma.call(a0_frag, b0_frag, c00_frag)
        rocdl.s_setprio(0)
        rocdl.s_barrier()

        b1_frag = b_s2r.load(b_cur1)
        b_g2s.load(b_cur0, B0_gl_offset + (k + 2) * b_k_step)
        rocdl.s_barrier()

        rocdl.s_setprio(1)
        c01_frag = mfma.call(a0_frag, b1_frag, c01_frag)
        rocdl.s_setprio(0)
        rocdl.s_barrier()

        a1_frag = a_s2r.load(a_cur1)
        a_g2s.load(a_cur0, A0_gl_offset + (k + 2) * a_k_step)
        rocdl.s_barrier()

        rocdl.s_setprio(1)
        c10_frag = mfma.call(a1_frag, b0_frag, c10_frag)
        rocdl.s_setprio(0)
        rocdl.s_barrier()

        b_g2s.load(b_cur1, B1_gl_offset + (k + 2) * b_k_step)
        wait_barrier(2 * N_LDS_STEPS_A + N_LDS_STEPS_B)

        rocdl.s_setprio(1)
        c11_frag = mfma.call(a1_frag, b1_frag, c11_frag)
        rocdl.s_setprio(0)
        rocdl.s_barrier()

        if nt_vmcnt >= 0:
            _llvm.inline_asm(
                res=None,
                operands_=[],
                asm_string=f"s_waitcnt vmcnt({nt_vmcnt})",
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
    rocdl.s_setprio(1)
    c00_frag = mfma.call(a0_frag, b0_frag, c00_frag)
    rocdl.s_setprio(0)
    rocdl.s_barrier()

    b1_frag = b_s2r.load(b_cur1)
    rocdl.s_barrier()
    rocdl.s_setprio(1)
    c01_frag = mfma.call(a0_frag, b1_frag, c01_frag)
    rocdl.s_setprio(0)
    rocdl.s_barrier()

    a1_frag = a_s2r.load(a_cur1)
    rocdl.s_barrier()
    rocdl.s_setprio(1)
    c10_frag = mfma.call(a1_frag, b0_frag, c10_frag)
    rocdl.s_setprio(0)
    rocdl.s_barrier()

    b0_frag = b_s2r.load(b_next0)
    a_g2s.load(a_next1, A1_gl_offset + (k + 1) * a_k_step)
    rocdl.s_barrier()
    rocdl.s_setprio(1)
    c11_frag = mfma.call(a1_frag, b1_frag, c11_frag)
    rocdl.s_setprio(0)
    rocdl.s_barrier()

    a_cur0, a_next0 = a_next0, a_cur0
    a_cur1, a_next1 = a_next1, a_cur1
    b_cur0, b_next0 = b_next0, b_cur0
    b_cur1, b_next1 = b_next1, b_cur1

    a0_frag = a_s2r.load(a_cur0)
    wait_barrier(0)
    rocdl.s_setprio(1)
    c00_frag = mfma.call(a0_frag, b0_frag, c00_frag)
    rocdl.s_setprio(0)
    rocdl.s_barrier()

    b1_frag = b_s2r.load(b_cur1)
    rocdl.s_barrier()
    rocdl.s_setprio(1)
    c01_frag = mfma.call(a0_frag, b1_frag, c01_frag)
    rocdl.s_setprio(0)
    rocdl.s_barrier()

    a1_frag = a_s2r.load(a_cur1)
    rocdl.s_barrier()
    rocdl.s_setprio(1)
    c10_frag = mfma.call(a1_frag, b0_frag, c10_frag)
    c11_frag = mfma.call(a1_frag, b1_frag, c11_frag)
    rocdl.s_setprio(0)
    rocdl.s_barrier()

    wave_n_offset = wave_n * (N_TILES_B * 32)
    wave_m_offset = wave_m * (N_TILES_A * 32)
    base_row = block_m * BLOCK_M + wave_m_offset
    base_col = block_n * BLOCK_N + wave_n_offset
    store_c.store(c00_frag, base_row + 0, base_col + 0)
    store_c.store(c01_frag, base_row + 0, base_col + LDS_BLOCK_N)
    store_c.store(c10_frag, base_row + LDS_BLOCK_M, base_col + 0)
    store_c.store(c11_frag, base_row + LDS_BLOCK_M, base_col + LDS_BLOCK_N)


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
    out_fp16=False,
    nt_vmcnt=3,
    b_group_base=None,
    c_cache_modifier=0,
):
    _nn_tn_tile(
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
        out_fp16=out_fp16,
        nt_vmcnt=nt_vmcnt,
        b_group_base=b_group_base,
        c_cache_modifier=c_cache_modifier,
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
    out_fp16=False,
    nt_vmcnt=3,
    b_group_base=None,
):
    _nn_tn_tile(
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
        out_fp16=out_fp16,
        nt_vmcnt=nt_vmcnt,
        b_group_base=b_group_base,
    )


@functools.lru_cache(maxsize=256)
def _compile_dense_nt(
    K: int,
    BLOCK_M: int = 256,
    BLOCK_N: int = 256,
    GROUP_M: int = 1,
    waves_per_eu: int = 2,
    agpr_alloc: int = 0,
    nt_vmcnt: int = 4,  # swept: vmcnt=4 > 3 (~1% on L1 NT, gfx950)
    num_xcd: int = 8,
    out_fp16: bool = False,
):
    assert BLOCK_M >= 128 and BLOCK_N >= 256 and BLOCK_M % 128 == 0 and BLOCK_N % 256 == 0
    assert GROUP_M >= 1
    SharedStorage = _make_shared_storage(BLOCK_M, BLOCK_N)

    @flyc.kernel(known_block_size=[512, 1, 1])
    def kernel_dense_nt(
        A: fx.Tensor,
        B_T: fx.Tensor,
        C: fx.Tensor,
        c_m: fx.Int32,
        c_n: fx.Int32,
    ):
        n_blocks = ceildiv(c_n, BLOCK_N)

        lds = fx.SharedAllocator().allocate(SharedStorage).peek()

        gemm_bf16_nt_tile(
            A,
            B_T,
            C,
            c_m,
            c_n,
            lds,
            K=K,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            n_blocks=n_blocks,
            GROUP_M=GROUP_M,
            num_xcd=num_xcd,
            out_fp16=out_fp16,
            nt_vmcnt=nt_vmcnt,
        )

    @flyc.jit
    def launch_dense_nt(
        A: fx.Tensor,
        B_T: fx.Tensor,
        C: fx.Tensor,
        c_m: fx.Int32,
        c_n: fx.Int32,
        stream: fx.Stream,
    ):
        grid_x = ceildiv(c_m, BLOCK_M) * ceildiv(c_n, BLOCK_N)
        kernel_dense_nt(
            A,
            B_T,
            C,
            c_m,
            c_n,
            value_attrs=make_value_attrs(waves_per_eu, agpr_alloc, "512,512"),
        ).launch(grid=(grid_x, 1, 1), block=(512, 1, 1), stream=stream)

    return launch_dense_nt


_COMPILED_DENSE_CACHE: dict = {}


def _get_compiled_dense(launch, args):
    key_parts = [id(launch)]
    for a in args:
        if isinstance(a, torch.Tensor):
            key_parts.append((tuple(a.shape), a.dtype))
        elif isinstance(a, int):
            key_parts.append(a)
        else:
            key_parts.append(type(a).__name__)
    key = tuple(key_parts)
    cached = _COMPILED_DENSE_CACHE.get(key)
    if cached is None:
        cached = flyc.compile(launch, *args)
        _COMPILED_DENSE_CACHE[key] = cached
    return cached


def gemm_bf16_flydsl_kernel(
    a: torch.Tensor,
    b: torch.Tensor,
    trans_a: bool = False,
    trans_b: bool = True,
    out_dtype: torch.dtype = torch.bfloat16,
    trans_c: bool = False,
    BLOCK_M: int = 256,
    GROUP_M: int = 1,
    num_xcd: int = 8,
) -> torch.Tensor:
    assert a.dim() == 2 and b.dim() == 2
    assert a.dtype == torch.bfloat16 and b.dtype == torch.bfloat16
    if out_dtype not in (torch.bfloat16, torch.float16):
        raise NotImplementedError(f"bf16 GEMM emits bf16/fp16, got {out_dtype}")
    out_fp16 = out_dtype == torch.float16

    if (not trans_a) and trans_b:
        M, K_a = a.shape
        N, K_b = b.shape
        assert K_a == K_b, f"NT K mismatch: a {a.shape}, b {b.shape}"
        K = K_a
        out = torch.empty((M, N), dtype=out_dtype, device=a.device)
        launch = _compile_dense_nt(
            K=K, BLOCK_M=BLOCK_M, BLOCK_N=256, GROUP_M=GROUP_M, num_xcd=num_xcd, out_fp16=out_fp16
        )
        args = (
            a.contiguous().view(-1),
            b.contiguous().view(-1),
            out.contiguous().view(-1),
            M,
            N,
            torch.cuda.current_stream(),
        )
        _get_compiled_dense(launch, args)(*args)
        if trans_c:
            return out.t().contiguous()
        return out

    if (not trans_a) and (not trans_b):
        return gemm_bf16_nn_kernel(
            a, b, out_dtype=out_dtype, BLOCK_M=BLOCK_M, GROUP_M=GROUP_M, num_xcd=num_xcd
        )
    if trans_a and (not trans_b):
        return gemm_bf16_tn_kernel(
            a, b, out_dtype=out_dtype, BLOCK_M=BLOCK_M, GROUP_M=GROUP_M, num_xcd=num_xcd
        )
    raise NotImplementedError("mega bf16 GEMM: TT layout not implemented.")


@functools.lru_cache(maxsize=256)
def _compile_dense_nn_tn(
    K,
    BLOCK_M=256,
    BLOCK_N=256,
    GROUP_M=1,
    num_xcd=8,
    nt_vmcnt=3,
    waves_per_eu=2,
    agpr_alloc=0,
    out_fp16=False,
    a_transpose=False,
):
    SharedStorage = _make_shared_storage(BLOCK_M, BLOCK_N)
    tile_fn = gemm_bf16_tn_tile if a_transpose else gemm_bf16_nn_tile

    @flyc.kernel(known_block_size=[512, 1, 1])
    def kernel(A: fx.Tensor, B: fx.Tensor, C: fx.Tensor, c_m: fx.Int32, c_n: fx.Int32):
        _ = str(fx.thread_idx.x)
        n_blocks = ceildiv(c_n, BLOCK_N)
        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        tile_fn(
            A,
            B,
            C,
            c_m,
            c_n,
            lds,
            K=K,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            n_blocks=n_blocks,
            GROUP_M=GROUP_M,
            num_xcd=num_xcd,
            out_fp16=out_fp16,
            nt_vmcnt=nt_vmcnt,
        )

    @flyc.jit
    def launch(A, B, C, c_m: fx.Int32, c_n: fx.Int32, stream: fx.Stream):
        grid_x = ceildiv(c_m, BLOCK_M) * ceildiv(c_n, BLOCK_N)
        kernel(A, B, C, c_m, c_n, value_attrs=make_value_attrs(waves_per_eu, agpr_alloc, "512,512")).launch(
            grid=(grid_x, 1, 1), block=(512, 1, 1), stream=stream
        )

    return launch


def gemm_bf16_nn_kernel(
    a: torch.Tensor,
    b: torch.Tensor,
    out_dtype: torch.dtype = torch.bfloat16,
    BLOCK_M: int = 256,
    GROUP_M: int = 1,
    num_xcd: int = 8,
) -> torch.Tensor:
    assert a.dim() == 2 and b.dim() == 2
    assert a.dtype == torch.bfloat16 and b.dtype == torch.bfloat16
    out_fp16 = out_dtype == torch.float16
    M, K = a.shape
    K_b, N = b.shape
    assert K == K_b, f"NN K mismatch: a {a.shape}, b {b.shape}"
    out = torch.empty((M, N), dtype=out_dtype, device=a.device)
    launch = _compile_dense_nn_tn(
        K=K,
        BLOCK_M=BLOCK_M,
        BLOCK_N=256,
        GROUP_M=GROUP_M,
        num_xcd=num_xcd,
        out_fp16=out_fp16,
        a_transpose=False,
    )
    args = (
        a.contiguous().view(-1),
        b.contiguous().view(-1),
        out.contiguous().view(-1),
        M,
        N,
        torch.cuda.current_stream(),
    )
    _get_compiled_dense(launch, args)(*args)
    return out


def gemm_bf16_tn_kernel(
    a: torch.Tensor,
    b: torch.Tensor,
    out_dtype: torch.dtype = torch.bfloat16,
    BLOCK_M: int = 256,
    GROUP_M: int = 1,
    num_xcd: int = 8,
) -> torch.Tensor:
    assert a.dim() == 2 and b.dim() == 2
    assert a.dtype == torch.bfloat16 and b.dtype == torch.bfloat16
    out_fp16 = out_dtype == torch.float16
    K, M = a.shape
    K_b, N = b.shape
    assert K == K_b, f"TN K mismatch: a {a.shape}, b {b.shape}"
    out = torch.empty((M, N), dtype=out_dtype, device=a.device)
    launch = _compile_dense_nn_tn(
        K=K,
        BLOCK_M=BLOCK_M,
        BLOCK_N=256,
        GROUP_M=GROUP_M,
        num_xcd=num_xcd,
        out_fp16=out_fp16,
        a_transpose=True,
    )
    args = (
        a.contiguous().view(-1),
        b.contiguous().view(-1),
        out.contiguous().view(-1),
        M,
        N,
        torch.cuda.current_stream(),
    )
    _get_compiled_dense(launch, args)(*args)
    return out


def _wgrad_geom(BLOCK_M, BLOCK_N):
    assert BLOCK_M >= 128 and BLOCK_N >= 64 and BLOCK_M % 128 == 0 and BLOCK_N % 64 == 0
    N_TILES_A = BLOCK_M // 128
    N_TILES_B = max(1, BLOCK_N // 256)
    return {
        "N_TILES_A": N_TILES_A,
        "N_TILES_B": N_TILES_B,
        "N_ACCUMS": N_TILES_A * N_TILES_B,
        "LDS_BLOCK_M": BLOCK_M // 2,
        "LDS_BLOCK_N": BLOCK_N // 2,
        "N_LDS_STEPS_A": (BLOCK_M // 16) // WGRAD_WAVES,
        "N_LDS_STEPS_B": (BLOCK_N // 16) // WGRAD_WAVES,
        "a_lds_size": (BLOCK_M // 2) * BLOCK_K,
        "b_lds_size": (BLOCK_N // 2) * BLOCK_K,
    }


def _wgrad_accum_bf16(mfma, a_frags, b_frags, acc_regs):
    c = [Vec(fx.memref_load_vec(r)) for r in acc_regs]
    c = mfma.call(a_frags, b_frags, c)
    for idx in range_constexpr(len(acc_regs)):
        fx.memref_store_vec(c[idx], acc_regs[idx])


def _wgrad_body_4buf_bf16(
    k,
    a_g2s,
    b_g2s,
    a_s2r,
    b_s2r,
    mfma,
    a_cur0,
    a_cur1,
    b_cur0,
    b_cur1,
    a_next0,
    a_next1,
    b_next0,
    b_next1,
    acc00,
    acc01,
    acc10,
    acc11,
    a0_off,
    a1_off,
    b0_off,
    b1_off,
    a_k_step,
    b_k_step,
    n_lds_steps_a,
    n_lds_steps_b,
):
    b0 = b_s2r.load(b_cur0)
    a0 = a_s2r.load(a_cur0)
    a_g2s.load(a_next1, a1_off + (k + 1) * a_k_step)
    rocdl.s_barrier()
    rocdl.s_setprio(1)
    _wgrad_accum_bf16(mfma, a0, b0, acc00)
    rocdl.s_setprio(0)
    rocdl.s_barrier()
    b1 = b_s2r.load(b_cur1)
    b_g2s.load(b_cur0, b0_off + (k + 2) * b_k_step)
    rocdl.s_barrier()
    rocdl.s_setprio(1)
    _wgrad_accum_bf16(mfma, a0, b1, acc01)
    rocdl.s_setprio(0)
    rocdl.s_barrier()
    a1 = a_s2r.load(a_cur1)
    a_g2s.load(a_cur0, a0_off + (k + 2) * a_k_step)
    rocdl.s_barrier()
    rocdl.s_setprio(1)
    _wgrad_accum_bf16(mfma, a1, b0, acc10)
    rocdl.s_setprio(0)
    rocdl.s_barrier()
    b_g2s.load(b_cur1, b1_off + (k + 2) * b_k_step)
    wait_barrier(2 * n_lds_steps_a + n_lds_steps_b)
    rocdl.s_setprio(1)
    _wgrad_accum_bf16(mfma, a1, b1, acc11)
    rocdl.s_setprio(0)
    rocdl.s_barrier()


def gemm_bf16_tn_variable_k_tile(
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
    out_fp16=False,
    c_cache_modifier=0,
    trans_c=False,
):
    CHUNK = 4
    geom = _wgrad_geom(BLOCK_M, BLOCK_N)
    n_tiles_a = geom["N_TILES_A"]
    geom["N_TILES_B"]
    geom["N_ACCUMS"]
    lds_block_m = geom["LDS_BLOCK_M"]
    lds_block_n = geom["LDS_BLOCK_N"]
    n_lds_steps_a = geom["N_LDS_STEPS_A"]
    n_lds_steps_b = geom["N_LDS_STEPS_B"]

    lane_id = fx.thread_idx.x % 64
    wave_id = fx.thread_idx.x // 64
    n_wave_n = WGRAD_WAVES // 2
    wave_m = wave_id // n_wave_n
    wave_n = wave_id % n_wave_n

    group_tokens = m_end - m_start
    bf16_ir = fx.BFloat16.ir_type
    # base offset (m_start * OUT * 2 bytes) can exceed int32 for a worst-case pool;
    # compute it in int64. num_records (per-group span) stays int32-small.
    a_base_off = _i64(m_start) * fx.Int64(OUT_M * 2)
    b_base_off = _i64(m_start) * fx.Int64(OUT_N * 2)
    gA = make_bf16_buffer_tensor_rebased(A, bf16_ir, a_base_off, group_tokens * out_m_rt * fx.Int32(2))
    gB = make_bf16_buffer_tensor_rebased(B, bf16_ir, b_base_off, group_tokens * out_n_rt * fx.Int32(2))
    a_div = fx.logical_divide(gA, fx.make_layout(1, 1))
    b_div = fx.logical_divide(gB, fx.make_layout(1, 1))

    gl_off_a = compute_global_swizzle_nn_bf16(lane_id, wave_id, OUT_M, n_lds_steps_a)
    gl_off_b = compute_global_swizzle_nn_bf16(lane_id, wave_id, OUT_N, n_lds_steps_b)

    a0_off = block_m * BLOCK_M
    a1_off = a0_off + lds_block_m
    b0_off = block_n * BLOCK_N
    b1_off = b0_off + lds_block_n
    a_k_step = fx.Int32(BLOCK_K) * out_m_rt
    b_k_step = fx.Int32(BLOCK_K) * out_n_rt

    nta16 = n_tiles_a * 2
    ntb16 = (BLOCK_N // 16) // (2 * n_wave_n)
    n_accums16 = nta16 * ntb16
    mfma = Mfma16x16x32(nta16, ntb16)
    a_s2r = S2RLoaderTr16x32Bf16(wave_m, nta16)
    b_s2r = S2RLoaderTr16x32Bf16(wave_n, ntb16)
    acc_vec_n = 4
    n_accums_eff = n_accums16
    a_g2s = G2SLoader(a_div, gl_off_a, n_lds_steps_a, bf16_ir, wave_id)
    b_g2s = G2SLoader(b_div, gl_off_b, n_lds_steps_b, bf16_ir, wave_id)
    out_ty = fx.Float16 if out_fp16 else fx.BFloat16
    if trans_c:
        store_c = StoreCBf16(C, G * OUT_N, OUT_M, out_ty, cache_modifier=c_cache_modifier)
    else:
        store_c = StoreCBf16(C, G * OUT_M, OUT_N, out_ty, cache_modifier=c_cache_modifier)

    acc00 = [fx.make_rmem_tensor(fx.make_layout(acc_vec_n, 1), fx.Float32) for _ in range(n_accums_eff)]
    acc01 = [fx.make_rmem_tensor(fx.make_layout(acc_vec_n, 1), fx.Float32) for _ in range(n_accums_eff)]
    acc10 = [fx.make_rmem_tensor(fx.make_layout(acc_vec_n, 1), fx.Float32) for _ in range(n_accums_eff)]
    acc11 = [fx.make_rmem_tensor(fx.make_layout(acc_vec_n, 1), fx.Float32) for _ in range(n_accums_eff)]
    for quad in (acc00, acc01, acc10, acc11):
        for reg in quad:
            fx.memref_store_vec(mfma.zero_value, reg)

    wait_barrier(0)
    b_g2s.load(lds.B_lds_cur_0, b0_off + 0 * b_k_step)
    a_g2s.load(lds.A_lds_cur_0, a0_off + 0 * a_k_step)
    b_g2s.load(lds.B_lds_cur_1, b1_off + 0 * b_k_step)
    a_g2s.load(lds.A_lds_cur_1, a1_off + 0 * a_k_step)
    _emit_if_then(wave_m == 1, lambda: rocdl.s_barrier())
    wait_barrier(n_lds_steps_a + n_lds_steps_b)
    b_g2s.load(lds.B_lds_next_0, b0_off + 1 * b_k_step)
    a_g2s.load(lds.A_lds_next_0, a0_off + 1 * a_k_step)
    b_g2s.load(lds.B_lds_next_1, b1_off + 1 * b_k_step)
    wait_barrier(n_lds_steps_a + 2 * n_lds_steps_b)

    k_iters = (group_tokens + (BLOCK_K - 1)) // BLOCK_K
    n_chunks = (k_iters + (CHUNK - 1)) // CHUNK

    def _chunk(chunk_iv):
        chunk_idx = ArithValue(chunk_iv)
        a_cur0, a_cur1 = lds.A_lds_cur_0, lds.A_lds_cur_1
        a_next0, a_next1 = lds.A_lds_next_0, lds.A_lds_next_1
        b_cur0, b_cur1 = lds.B_lds_cur_0, lds.B_lds_cur_1
        b_next0, b_next1 = lds.B_lds_next_0, lds.B_lds_next_1
        _body = _wgrad_body_4buf_bf16
        for j in range_constexpr(CHUNK):
            k = chunk_idx * CHUNK + j
            _body(
                k,
                a_g2s,
                b_g2s,
                a_s2r,
                b_s2r,
                mfma,
                a_cur0,
                a_cur1,
                b_cur0,
                b_cur1,
                a_next0,
                a_next1,
                b_next0,
                b_next1,
                acc00,
                acc01,
                acc10,
                acc11,
                a0_off,
                a1_off,
                b0_off,
                b1_off,
                a_k_step,
                b_k_step,
                n_lds_steps_a,
                n_lds_steps_b,
            )
            a_cur0, a_next0 = a_next0, a_cur0
            a_cur1, a_next1 = a_next1, a_cur1
            b_cur0, b_next0 = b_next0, b_cur0
            b_cur1, b_next1 = b_next1, b_cur1

    _emit_for(fx.Int32(0), n_chunks, fx.Int32(1), _chunk)

    c00 = [Vec(fx.memref_load_vec(reg)) for reg in acc00]
    c01 = [Vec(fx.memref_load_vec(reg)) for reg in acc01]
    c10 = [Vec(fx.memref_load_vec(reg)) for reg in acc10]
    c11 = [Vec(fx.memref_load_vec(reg)) for reg in acc11]

    def _emit_q(cfrag, q_row, q_col):
        for i in range_constexpr(nta16):
            for j in range_constexpr(ntb16):
                blk = [cfrag[i * ntb16 + j]]
                if trans_c:
                    store_c.store_trans16(blk, group_idx, q_row + i * 16, q_col + j * 16, OUT_M, OUT_N)
                else:
                    store_c.store16(blk, q_row + i * 16, q_col + j * 16)

    if trans_c:
        local_m = block_m * BLOCK_M + wave_m * (nta16 * 16)
        local_n = block_n * BLOCK_N + wave_n * (ntb16 * 16)
        _emit_q(c00, local_m + 0, local_n + 0)
        _emit_q(c01, local_m + 0, local_n + lds_block_n)
        _emit_q(c10, local_m + lds_block_m, local_n + 0)
        _emit_q(c11, local_m + lds_block_m, local_n + lds_block_n)
    else:
        base_row = group_idx * OUT_M + block_m * BLOCK_M + wave_m * (nta16 * 16)
        base_col = block_n * BLOCK_N + wave_n * (ntb16 * 16)
        _emit_q(c00, base_row + 0, base_col + 0)
        _emit_q(c01, base_row + 0, base_col + lds_block_n)
        _emit_q(c10, base_row + lds_block_m, base_col + 0)
        _emit_q(c11, base_row + lds_block_m, base_col + lds_block_n)


def load_go_i32(div, idx):
    atom = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Int32)
    reg = fx.make_rmem_tensor(fx.make_layout(1, 1), fx.Int32)
    fx.copy(atom, fx.slice(div, (None, fx.Int32(idx))), reg)
    return Vec(fx.memref_load_vec(reg))[0]


def load_go_i64(div, idx):
    atom = fx.make_copy_atom(fx.rocdl.BufferCopy64b(), fx.Int64)
    reg = fx.make_rmem_tensor(fx.make_layout(1, 1), fx.Int64)
    fx.copy(atom, fx.slice(div, (None, fx.Int32(idx))), reg)
    v64 = Vec(fx.memref_load_vec(reg))[0]
    return fx.arith.ArithValue(fx.arith.trunci(fx.T.i32(), v64.ir_value()), signed=True)


@functools.lru_cache(maxsize=64)
def _compile_grouped_tn_wgrad_bf16(
    OUT_M,
    OUT_N,
    G,
    BLOCK_M=256,
    BLOCK_N=256,
    num_xcd=8,
    waves_per_eu=2,
    out_fp16=False,
    trans_c=False,
):
    assert OUT_M % BLOCK_M == 0, "OUT_M (unclamped store dim) must divide BLOCK_M"
    N_BLOCKS_M = OUT_M // BLOCK_M
    N_BLOCKS_N = (OUT_N + BLOCK_N - 1) // BLOCK_N
    TILES_PER_GROUP = N_BLOCKS_M * N_BLOCKS_N
    TOTAL = G * TILES_PER_GROUP
    SharedStorage = _make_shared_storage(BLOCK_M, BLOCK_N)

    @flyc.kernel(known_block_size=[WGRAD_BLOCK, 1, 1])
    def kernel(
        A: fx.Tensor,
        B: fx.Tensor,
        C: fx.Tensor,
        group_offs: fx.Tensor,
        out_m_rt: fx.Int32,
        out_n_rt: fx.Int32,
    ):
        _ = str(fx.thread_idx.x)
        go = fx.rocdl.make_buffer_tensor(group_offs, max_size=False, num_records_bytes=(G + 1) * 4)
        go_div = fx.logical_divide(go, fx.make_layout(1, 1))
        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        pid = fx.block_idx.x

        def _do_tile(tile_idx):
            tile = xcd_remap_pid(tile_idx, TOTAL, num_xcd)
            group_idx = tile // TILES_PER_GROUP
            local_tile = tile % TILES_PER_GROUP
            if const_expr(trans_c):
                block_n = local_tile // N_BLOCKS_M
                block_m = local_tile % N_BLOCKS_M
            else:
                block_m = local_tile // N_BLOCKS_N
                block_n = local_tile % N_BLOCKS_N
            m_start = load_go_i32(go_div, group_idx)
            m_end = load_go_i32(go_div, group_idx + 1)
            gemm_bf16_tn_variable_k_tile(
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
                out_fp16=out_fp16,
                trans_c=trans_c,
            )

        _do_tile(pid)

    @flyc.jit
    def launch(A, B, C, group_offs, out_m_rt: fx.Int32, out_n_rt: fx.Int32, stream: fx.Stream):
        grid_x = fx.Int32(TOTAL)
        kernel(
            A,
            B,
            C,
            group_offs,
            out_m_rt,
            out_n_rt,
            value_attrs=make_value_attrs(waves_per_eu, WGRAD_AGPR, f"{WGRAD_BLOCK},{WGRAD_BLOCK}"),
        ).launch(grid=(grid_x, 1, 1), block=(WGRAD_BLOCK, 1, 1), stream=stream)

    return launch


_WGRAD_COMPILED = {}


def grouped_gemm_tn_wgrad_bf16(
    lhs: torch.Tensor,
    rhs: torch.Tensor,
    group_offs: torch.Tensor,
    out_dtype: torch.dtype = torch.bfloat16,
    BLOCK_M: int = 256,
    BLOCK_N: int = 256,
    num_xcd: int = 8,
    trans_c: bool = False,
) -> torch.Tensor:
    assert lhs.dim() == 2 and rhs.dim() == 2 and lhs.shape[0] == rhs.shape[0]
    assert lhs.dtype == torch.bfloat16 and rhs.dtype == torch.bfloat16
    OUT_M = lhs.shape[1]
    OUT_N = rhs.shape[1]
    G = group_offs.numel() - 1
    out_fp16 = out_dtype == torch.float16
    out_shape = (G, OUT_N, OUT_M) if trans_c else (G, OUT_M, OUT_N)
    out = torch.empty(out_shape, device=lhs.device, dtype=out_dtype)
    go32 = group_offs.to(torch.int32) if group_offs.dtype != torch.int32 else group_offs
    launch = _compile_grouped_tn_wgrad_bf16(
        OUT_M,
        OUT_N,
        G,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_xcd=num_xcd,
        out_fp16=out_fp16,
        trans_c=trans_c,
    )
    args = (
        lhs.contiguous().view(-1),
        rhs.contiguous().view(-1),
        out.view(-1),
        go32,
        OUT_M,
        OUT_N,
        torch.cuda.current_stream(),
    )
    key = (OUT_M, OUT_N, G, BLOCK_M, BLOCK_N, out_fp16, trans_c)
    compiled = _WGRAD_COMPILED.get(key)
    if compiled is None:
        compiled = flyc.compile(launch, *args)
        _WGRAD_COMPILED[key] = compiled
    compiled(*args)
    return out


GEMM_TILE = {"nt": gemm_bf16_nt_tile, "nn": gemm_bf16_nn_tile, "tn": gemm_bf16_tn_tile}
