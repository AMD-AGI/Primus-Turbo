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

"""Primus-Turbo dense FP8 GEMM kernel (FlyDSL): NT, NN and TN layouts,
mfma_f32_16x16x128_f8f6f4, per-tensor scale, bf16/fp16 out, arbitrary K (TT unsupported).
Each layout races an 8-wave skeleton against a 4-wave whole-loop per shape."""

import functools
import math
from typing import NamedTuple

import torch

# isort: off
# Primitives are vendored in flydsl/utils/gemm_helper.py (no 3rdparty/FlyDSL
# submodule; flydsl, the compiler, is the only FlyDSL dep) and imported as module
# globals (@flyc.kernel needs its dependencies as globals).
from primus_turbo.flydsl.utils.gemm_helper import (
    resolve_accum_out,
    G2SLoader,
    Mfma16x16x128,
    S2RLoader,
    S2RLoaderTr,
    StoreCPerTensor,
    StoreCPerTensorLineN,
    StoreCPerTensorPairN,
    StoreCPerTensorQuadN,
    XPOSE_SLOT,
    XPOSE_SLOTS,
    _lds_barrier,
    _readfirstlane_i32,
    _robust_time,
    asm_mma_do,
    block_mn,
    ceildiv,
    compute_global_swizzle,
    compute_global_swizzle_nn,
    floordiv_pow2,
    load_per_tensor_scale,
    make_fp8_buffer_tensor_rebased,
    make_row_band_resource,
    make_value_attrs,
    mask_a_tail,
    wait_barrier,
    xcd_remap_pid,
)
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm as _llvm
from flydsl.expr import arith
from flydsl.expr import buffer_ops as _buffer_ops
from flydsl.expr import const_expr, range_constexpr, rocdl
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

# isort: on

# `nt` aux bit: C is write-once, so caching it evicts the A/B band the L2 swizzle keeps.
_CSTORE_AUX = 2

_PICK_RAMP_ITERS = 200  # throwaway launches before timing: the leading candidate else pays the ramp


@functools.lru_cache(maxsize=256)
def _compile_dense_nt(
    K: int,
    BLOCK_M: int = 256,
    BLOCK_N: int = 256,
    GROUP_M: int = 1,
    waves_per_eu: int = 2,
    agpr_alloc: int = 0,
    nt_vmcnt: int = 3,  # end-of-iter s_waitcnt vmcnt(N): N=3 → det=0 (gfx950 G2S buffer_load_lds/ds_read LDS hazard), <=1.1% cost; N>=4 races, N<3 costlier; -1 disables
    num_xcd: int = 8,  # XCD-aware PID remap: cluster same-XCD WGs into contiguous logical tiles for per-XCD L2 reuse (gfx950 MI355X = 8 XCD); 1 disables
    cbsz: int = 0,  # srcA fp8 fmt: 0=E4M3, 1=E5M2
    blgp: int = 0,  # srcB fp8 fmt: 0=E4M3, 1=E5M2
    out_fp16: bool = False,  # StoreCPerTensor out dtype: True -> fp16, else bf16
    pair_n: bool = False,  # fold the n-fragment pair into one dword store (needs even N)
    col_safe: bool = False,  # N % BLOCK_N == 0: drop the epilogue's per-store column clamp
    beta_is_one: bool = False,  # epilogue accumulates (C += acc) instead of overwriting
):
    """Build & cache the (K, BLOCK_M, BLOCK_N, GROUP_M)-specialised NT launch.

    GROUP_M is the super-block tile-id swizzle width for L2 reuse (WGs advance
    block_m first within each GROUP_M x n_blocks band; 1 = row-major). The main
    K-loop barriers are all load-bearing (each guards a compiler-reorder race).
    """
    BLOCK_K = 128
    assert BLOCK_M >= 128 and BLOCK_N >= 256 and BLOCK_M % 128 == 0 and BLOCK_N % 256 == 0
    assert GROUP_M >= 1

    # Odd-K native K-tail: ceil(K/128) iters, the last of length K_TAIL (0 =
    # exact multiple). The tail's invalid K-columns are zeroed on A in Epilog 2
    # via mask_a_tail; G2S tail over-reads clamp to 0 via the buffer SRD bound.
    K_ITERS = (K + BLOCK_K - 1) // BLOCK_K
    K_TAIL = K % BLOCK_K
    assert K_ITERS >= 2, f"K_ITERS={K_ITERS} too small; need K >= 129 (ceil(K/128) >= 2)"

    N_TILES_A = BLOCK_M // 64
    N_TILES_B = BLOCK_N // 128
    N_ACCUMS = N_TILES_A * N_TILES_B
    assert N_ACCUMS > 0

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
    def kernel_dense_nt(
        A: fx.Tensor,
        B_T: fx.Tensor,
        C: fx.Tensor,
        A_scale: fx.Tensor,
        B_scale: fx.Tensor,
        c_m: fx.Int32,
        c_n: fx.Int32,
    ):
        # NT semantics: A is [M, K] row-major K-contig.
        #               B_T is [N, K] row-major K-contig (= B^T storage of [K, N]).
        # Output       C is [M, N] row-major bf16.
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
        # Super-block tile swizzle for L2 reuse; group_size_m clamps the last
        # band so any GROUP_M >= 1 is correct (arith.select = integer min).
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

        # i64 input re-base: fold the per-tile row base (m_row*K, n_row*K) into the
        # SRD base; A/B_T K-contiguous (foldable), k*BLOCK_K small int32 -> no cap.
        a_base = arith.index_cast(T.index, block_m * BLOCK_M) * arith.index(K)
        b_base = arith.index_cast(T.index, block_n * BLOCK_N) * arith.index(K)
        a_nrec = (
            arith.index_cast(T.index, c_m) - arith.index_cast(T.index, block_m * BLOCK_M)
        ) * arith.index(K)
        b_nrec = (
            arith.index_cast(T.index, c_n) - arith.index_cast(T.index, block_n * BLOCK_N)
        ) * arith.index(K)
        A0_gl_offset = 0
        A1_gl_offset = LDS_BLOCK_M * K
        B0_gl_offset = 0
        B1_gl_offset = LDS_BLOCK_N * K

        gA = make_fp8_buffer_tensor_rebased(A, F8_IR_t, a_base, a_nrec)
        gB = make_fp8_buffer_tensor_rebased(B_T, F8_IR_t, b_base, b_nrec)
        a_div = fx.logical_divide(gA, fx.make_layout(1, 1))
        b_div = fx.logical_divide(gB, fx.make_layout(1, 1))

        gl_off_a = compute_global_swizzle(lane_id, wave_id, K, N_LDS_ROUNDS, preshuffled=False)
        gl_off_b = compute_global_swizzle(lane_id, wave_id, K, N_LDS_ROUNDS, preshuffled=False)

        mfma = Mfma16x16x128(N_TILES_A, N_TILES_B)
        if cbsz or blgp:
            _ea = fx.Float8E5M2 if cbsz else fx.Float8E4M3FN
            _eb = fx.Float8E5M2 if blgp else fx.Float8E4M3FN
            mfma.atom = fx.make_mma_atom(fx.rocdl.cdna4.MFMA_Scale(16, 16, 128, _ea, _eb))

        a_g2s = G2SLoader(a_div, gl_off_a, N_LDS_STEPS_A, F8_IR_t, wave_id)
        b_g2s = G2SLoader(b_div, gl_off_b, N_LDS_STEPS_B, F8_IR_t, wave_id)
        a_s2r = S2RLoader(wave_m, N_TILES_A)
        b_s2r = S2RLoader(wave_n, N_TILES_B)
        _out_ty = fx.Float16 if out_fp16 else fx.BFloat16
        store_c = (StoreCPerTensorPairN if pair_n else StoreCPerTensor)(
            A_scale,
            B_scale,
            C,
            c_m,
            c_n,
            mfma.idx,
            N_TILES_A,
            N_TILES_B,
            _out_ty,
            col_safe=col_safe,
            store_aux=_CSTORE_AUX,
            beta_is_one=beta_is_one,
        )

        c00_frag = [mfma.zero_value] * N_ACCUMS
        c01_frag = [mfma.zero_value] * N_ACCUMS
        c10_frag = [mfma.zero_value] * N_ACCUMS
        c11_frag = [mfma.zero_value] * N_ACCUMS

        # Prelude: k=0 → cur, k=1 → next (a_next1 lazily on first main iter).
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

        # Main K-loop. Each iter: s2r {a0,b0,b1,a1} → 4 mma (c00→c01→c10→c11)
        # interleaved with k+1 (a_next1) and k+2 (a_cur0, b_cur0, b_cur1) prefetches.
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
                )  # end-of-iter G2S drain (race fix)
            a_cur0, a_next0 = a_next0, a_cur0
            a_cur1, a_next1 = a_next1, a_cur1
            b_cur0, b_next0 = b_next0, b_cur0
            b_cur1, b_next1 = b_next1, b_cur1

        # Epilog 1 (k = K_ITERS - 2). The a_g2s.load(a_next1, A1 + (k+1)*BLOCK_K)
        # line is the c10/c11 stale-a1 pipeline fix -- without it epilog-2's
        # a1_frag would read older K-iter data and the bottom half of every
        # output tile loses the final K-tile contribution.
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
        a_g2s.load(a_next1, A1_gl_offset + (k + 1) * BLOCK_K)  # stale-a1 fix
        rocdl.s_barrier()

        rocdl.s_setprio(1)
        c11_frag = mfma.call(a1_frag, b1_frag, c11_frag)
        rocdl.s_setprio(0)
        rocdl.s_barrier()

        a_cur0, a_next0 = a_next0, a_cur0
        a_cur1, a_next1 = a_next1, a_cur1
        b_cur0, b_next0 = b_next0, b_cur0
        b_cur1, b_next1 = b_next1, b_cur1

        # K-tail block: mask A so columns past K_TAIL contribute 0. Past wait_barrier(0)
        # nothing writes LDS, so a group store drains in the next group mfma shadow --
        # the two share neither vmcnt nor lgkmcnt.
        a0_frag = a_s2r.load(a_cur0)
        a0_frag = mask_a_tail(a0_frag, lane_id, K_TAIL)
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

        wave_n_offset = wave_n * (N_TILES_B * 16)
        wave_m_offset = wave_m * (N_TILES_A * 16)
        base_row = block_m * BLOCK_M + wave_m_offset
        base_col = block_n * BLOCK_N + wave_n_offset

        a1_frag = a_s2r.load(a_cur1)
        a1_frag = mask_a_tail(a1_frag, lane_id, K_TAIL)
        store_c.store(c00_frag, base_row + 0, base_col + 0)

        rocdl.s_setprio(1)
        c10_frag = mfma.call(a1_frag, b0_frag, c10_frag)
        rocdl.s_setprio(0)
        rocdl.s_barrier()

        store_c.store(c01_frag, base_row + 0, base_col + LDS_BLOCK_N)

        rocdl.s_setprio(1)
        c11_frag = mfma.call(a1_frag, b1_frag, c11_frag)
        rocdl.s_setprio(0)
        rocdl.s_barrier()

        store_c.store(c10_frag, base_row + LDS_BLOCK_M, base_col + 0)
        store_c.store(c11_frag, base_row + LDS_BLOCK_M, base_col + LDS_BLOCK_N)

    @flyc.jit
    def launch_dense_nt(
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
        kernel_dense_nt(
            A,
            B_T,
            C,
            A_scale,
            B_scale,
            c_m,
            c_n,
            value_attrs=make_value_attrs(waves_per_eu, agpr_alloc, "512,512"),
        ).launch(grid=(grid_x, 1, 1), block=(512, 1, 1), stream=stream)

    return launch_dense_nt


# ──────────────────────────────────────────────────────────────────────


@functools.lru_cache(maxsize=128)
def _compile_dense_nn(
    K: int,
    BLOCK_M: int = 256,
    BLOCK_N: int = 256,
    GROUP_M: int = 4,
    group_n: int = 0,  # 0 = 1D GROUP_M swizzle; >0 = 2D band (width group_n), as in NT
    num_xcd: int = 8,  # XCD-aware PID remap for per-XCD L2 reuse (MI355X = 8 XCD); 1 disables. See xcd_remap_pid.
    waves_per_eu: int = 2,
    agpr_alloc: int = 0,
    # Issue ds_read_tr8_b64 as inline asm so the backend skips the auto vmcnt(0)
    # drain; vmcnt_hint supplies the LDS sync. Requires agpr_alloc > 0.
    b_inline_asm_load: bool = False,
    vmcnt_hint: int = 2,
    cbsz: int = 0,  # srcA fp8 fmt: 0=E4M3, 1=E5M2
    blgp: int = 0,  # srcB fp8 fmt: 0=E4M3, 1=E5M2
    out_fp16: bool = False,  # StoreCPerTensor out dtype: True -> fp16, else bf16
    i64_traverse: bool = False,  # B[K,N] traversal via per-load i64 SRD re-base (lifts k*n < 2^32 cap)
    pair_n: bool = False,  # fold the n-fragment pair into one dword store (needs even N)
    col_safe: bool = False,  # N % BLOCK_N == 0: drop the epilogue's per-store column clamp
    beta_is_one: bool = False,  # epilogue accumulates (C += acc) instead of overwriting
):
    """NN-layout fp8 dense kernel. A [M, K], B [K, N], C [M, N].

    ``agpr_alloc`` / ``waves_per_eu`` mirror the NT kernel's knobs; see
    ``make_value_attrs`` for ``agpr_alloc`` encoding (N>0 = exact N AGPRs,
    -N = up to N, 0 = compiler default)."""
    if b_inline_asm_load and agpr_alloc == 0:
        raise ValueError(
            "b_inline_asm_load=True requires agpr_alloc > 0 (a compiler-decided "
            "AGPR count conflicts with the inline-asm operand constraints); "
            "pin AGPR to a nonzero value such as 32."
        )
    BLOCK_K = 128
    assert BLOCK_M >= 128 and BLOCK_N >= 256 and BLOCK_M % 128 == 0 and BLOCK_N % 256 == 0

    # Odd-K native K-tail: ceil iters; final iter masked on A (see NT note).
    K_ITERS = (K + BLOCK_K - 1) // BLOCK_K
    K_TAIL = K % BLOCK_K
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
    b_lds_size = LDS_BLOCK_N * BLOCK_K  # same byte count as NT, different layout

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
    def kernel_dense_nn(
        A: fx.Tensor,
        B: fx.Tensor,
        C: fx.Tensor,
        A_scale: fx.Tensor,
        B_scale: fx.Tensor,
        c_m: fx.Int32,
        c_n: fx.Int32,
    ):
        # Materialize thread_idx.x before S2RLoaderTr lazily uses it inside
        # range_constexpr loops, so the ds_read_tr8_b64 load order is correct.
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
        block_m, block_n = block_mn(pid, num_pid_m, n_blocks, GROUP_M, group_n)

        # i64 input re-base. A[M,K]: fold row base (m_row*K) into SRD. B[K,N]: the
        # k*BLOCK_K*c_n contraction is i64 per load (cn_i), capped at 4GB by num_records.
        m_row = block_m * BLOCK_M
        cn_i = arith.index_cast(T.index, c_n)
        a_base = arith.index_cast(T.index, m_row) * arith.index(K)
        a_nrec = (arith.index_cast(T.index, c_m) - arith.index_cast(T.index, m_row)) * arith.index(K)
        b_base = arith.index_cast(T.index, block_n * BLOCK_N)
        b_nrec = arith.index(K) * cn_i - b_base
        A0_gl_offset = 0
        A1_gl_offset = LDS_BLOCK_M * K
        B0_gl_offset = 0
        B1_gl_offset = LDS_BLOCK_N

        gA = make_fp8_buffer_tensor_rebased(A, F8_IR_t, a_base, a_nrec)
        gB = make_fp8_buffer_tensor_rebased(B, F8_IR_t, b_base, b_nrec)
        a_div = fx.logical_divide(gA, fx.make_layout(1, 1))
        b_div = fx.logical_divide(gB, fx.make_layout(1, 1))

        _nnwz = True  # wave bank-swizzle B; write and read sides must match
        gl_off_a = compute_global_swizzle(lane_id, wave_id, K, N_LDS_ROUNDS, preshuffled=False)
        gl_off_b = compute_global_swizzle_nn(lane_id, wave_id, c_n, N_LDS_ROUNDS, wswz=_nnwz)

        mfma = Mfma16x16x128(N_TILES_A, N_TILES_B)
        if cbsz or blgp:
            # E5M2 / hybrid: rebuild the MFMA atom with per-operand fp8 fmt
            # (cbsz->srcA, blgp->srcB). Same instruction family / frag layout
            # as the default e4m3 atom, so loaders are unchanged.
            _ea = fx.Float8E5M2 if cbsz else fx.Float8E4M3FN
            _eb = fx.Float8E5M2 if blgp else fx.Float8E4M3FN
            mfma.atom = fx.make_mma_atom(fx.rocdl.cdna4.MFMA_Scale(16, 16, 128, _ea, _eb))

        a_g2s = G2SLoader(a_div, gl_off_a, N_LDS_STEPS_A, F8_IR_t, wave_id)
        # B[K,N] is the contraction-traversal operand: in i64 mode re-base its SRD
        # per load (k_offset folds into the i64 base) instead of a 32-bit soffset.
        b_rebase = (B, F8_IR_t, b_base, b_nrec) if i64_traverse else None
        b_g2s = G2SLoader(b_div, gl_off_b, N_LDS_STEPS_B, F8_IR_t, wave_id, rebase=b_rebase)
        a_s2r = S2RLoader(wave_m, N_TILES_A)
        b_s2r = S2RLoaderTr(
            wave_n, N_TILES_B, 32, inline_asm=b_inline_asm_load, vmcnt_hint=vmcnt_hint, wswz=_nnwz
        )
        _out_ty = fx.Float16 if out_fp16 else fx.BFloat16
        store_c = (StoreCPerTensorPairN if pair_n else StoreCPerTensor)(
            A_scale,
            B_scale,
            C,
            c_m,
            c_n,
            mfma.idx,
            N_TILES_A,
            N_TILES_B,
            _out_ty,
            col_safe=col_safe,
            store_aux=_CSTORE_AUX,
            beta_is_one=beta_is_one,
        )

        c00_frag = [mfma.zero_value] * N_ACCUMS
        c01_frag = [mfma.zero_value] * N_ACCUMS
        c10_frag = [mfma.zero_value] * N_ACCUMS
        c11_frag = [mfma.zero_value] * N_ACCUMS

        # Prelude.
        b_g2s.load(b_cur0, B0_gl_offset + arith.index(0 * BLOCK_K) * cn_i)
        a_g2s.load(a_cur0, A0_gl_offset + 0 * BLOCK_K)
        b_g2s.load(b_cur1, B1_gl_offset + arith.index(0 * BLOCK_K) * cn_i)
        a_g2s.load(a_cur1, A1_gl_offset + 0 * BLOCK_K)

        if wave_m == 1:
            rocdl.s_barrier()

        wait_barrier(N_LDS_STEPS_A + N_LDS_STEPS_B)

        b_g2s.load(b_next0, B0_gl_offset + arith.index(1 * BLOCK_K) * cn_i)
        a_g2s.load(a_next0, A0_gl_offset + 1 * BLOCK_K)
        b_g2s.load(b_next1, B1_gl_offset + arith.index(1 * BLOCK_K) * cn_i)

        wait_barrier(N_LDS_STEPS_A + 2 * N_LDS_STEPS_B)

        # Main loop. Emits 7 barriers per K-iter (before/after each MFMA);
        # all are load-bearing — dropping any risks a compiler-reorder race.
        # vmcnt=-1: the trailing wait_barrier already drains g2s (the epilogue keeps its own).
        for k in range_constexpr(K_ITERS - 2):
            b0_frag = b_s2r.load(b_cur0, vmcnt=-1)
            a0_frag = a_s2r.load(a_cur0)
            a_g2s.load(a_next1, A1_gl_offset + (k + 1) * BLOCK_K)
            rocdl.s_barrier()

            rocdl.s_setprio(1)
            c00_frag = mfma.call(a0_frag, b0_frag, c00_frag)
            rocdl.s_setprio(0)
            rocdl.s_barrier()

            b1_frag = b_s2r.load(b_cur1, vmcnt=-1)
            b_g2s.load(b_cur0, B0_gl_offset + arith.index((k + 2) * BLOCK_K) * cn_i)
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

            b_g2s.load(b_cur1, B1_gl_offset + arith.index((k + 2) * BLOCK_K) * cn_i)
            wait_barrier(2 * N_LDS_STEPS_A + N_LDS_STEPS_B)

            rocdl.s_setprio(1)
            c11_frag = mfma.call(a1_frag, b1_frag, c11_frag)
            rocdl.s_setprio(0)
            rocdl.s_barrier()

            a_cur0, a_next0 = a_next0, a_cur0
            a_cur1, a_next1 = a_next1, a_cur1
            b_cur0, b_next0 = b_next0, b_cur0
            b_cur1, b_next1 = b_next1, b_cur1

        # Epilog 1.
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
        a_g2s.load(a_next1, A1_gl_offset + (k + 1) * BLOCK_K)  # stale-a1 fix
        rocdl.s_barrier()

        rocdl.s_setprio(1)
        c11_frag = mfma.call(a1_frag, b1_frag, c11_frag)
        rocdl.s_setprio(0)
        rocdl.s_barrier()

        a_cur0, a_next0 = a_next0, a_cur0
        a_cur1, a_next1 = a_next1, a_cur1
        b_cur0, b_next0 = b_next0, b_cur0
        b_cur1, b_next1 = b_next1, b_cur1

        wave_n_offset = _readfirstlane_i32(wave_n * (N_TILES_B * 16))
        wave_m_offset = _readfirstlane_i32(wave_m * (N_TILES_A * 16))
        base_row = block_m * BLOCK_M + wave_m_offset
        base_col = block_n * BLOCK_N + wave_n_offset

        # Epilog 2 -- K-tail block. Mask A so K-cols >= K_TAIL contribute 0.
        a0_frag = a_s2r.load(a_cur0)
        a0_frag = mask_a_tail(a0_frag, lane_id, K_TAIL)
        wait_barrier(0)

        rocdl.s_setprio(1)
        c00_frag = mfma.call(a0_frag, b0_frag, c00_frag)
        rocdl.s_setprio(0)
        rocdl.s_barrier()

        # Issue each group at its own last mfma: the exposed drain is the burst's tail.
        store_c.store(c00_frag, base_row, base_col)

        b1_frag = b_s2r.load(b_cur1)
        rocdl.s_barrier()

        rocdl.s_setprio(1)
        c01_frag = mfma.call(a0_frag, b1_frag, c01_frag)
        rocdl.s_setprio(0)
        rocdl.s_barrier()

        store_c.store(c01_frag, base_row, base_col + LDS_BLOCK_N)

        a1_frag = a_s2r.load(a_cur1)
        a1_frag = mask_a_tail(a1_frag, lane_id, K_TAIL)
        rocdl.s_barrier()

        rocdl.s_setprio(1)
        c10_frag = mfma.call(a1_frag, b0_frag, c10_frag)
        rocdl.s_setprio(0)

        # Split the tail block's last two groups so the store issues under the second and
        # the matrix core covers a quadrant that used to drain after the final mfma.
        store_c.store(c10_frag, base_row + LDS_BLOCK_M, base_col)

        rocdl.s_setprio(1)
        c11_frag = mfma.call(a1_frag, b1_frag, c11_frag)
        rocdl.s_setprio(0)
        rocdl.s_barrier()

        store_c.store(c11_frag, base_row + LDS_BLOCK_M, base_col + LDS_BLOCK_N)

    @flyc.jit
    def launch_dense_nn(
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
        kernel_dense_nn(
            A,
            B,
            C,
            A_scale,
            B_scale,
            c_m,
            c_n,
            value_attrs=make_value_attrs(waves_per_eu, agpr_alloc, "512,512"),
        ).launch(grid=(grid_x, 1, 1), block=(512, 1, 1), stream=stream)

    return launch_dense_nn


@functools.lru_cache(maxsize=128)
def _compile_dense_tn(
    K: int,
    BLOCK_M: int = 256,
    BLOCK_N: int = 256,
    GROUP_M: int = 4,
    waves_per_eu: int = 2,
    vmcnt_hint: int = 3,
    group_n: int = 0,  # 0 = 1D GROUP_M swizzle; >0 = 2D band (width group_n)
    num_xcd: int = 8,  # XCD-aware PID remap for per-XCD L2 reuse (MI355X = 8 XCD); 1 disables. See xcd_remap_pid.
    cbsz: int = 0,  # srcA fp8 fmt: 0=E4M3, 1=E5M2
    blgp: int = 0,  # srcB fp8 fmt: 0=E4M3, 1=E5M2
    out_fp16: bool = False,  # StoreCPerTensor out dtype: True -> fp16, else bf16
    i64_traverse: bool = False,  # A[K,M] & B[K,N] traversal via per-load i64 SRD re-base (lifts cap)
    beta_is_one: bool = False,  # epilogue accumulates (C += acc) instead of overwriting
):
    """TN-layout fp8 dense kernel: A [K, M], B [K, N], C [M, N] = A^T @ B.
    Both A and B are K-row strided, so both go through the wave-coop
    ds_read_b64_tr_b8 transpose load (the mfma A and B operand register byte
    layouts are identical, so the same S2RLoaderTr feeds both operands).
    Inline-asm tr8 on both operands + asm-inplace MFMA (=a,v,v,0; D aliases C in
    AGPR -> accumulators spill-free, no per-K-iter A-side vmcnt(0) drain)."""
    _a_inline = True
    _b_inline = True
    _asm_mma_mode = "2"  # asm-inplace MFMA (accum in AGPR)
    _inplace = True
    agpr_alloc = 128
    BLOCK_K = 128
    assert BLOCK_M >= 128 and BLOCK_N >= 256 and BLOCK_M % 128 == 0 and BLOCK_N % 256 == 0

    # Odd-K native K-tail: ceil iters. No A-mask needed here -- TN's A [K,M]
    # and B [K,N] are K-row-major, so the tail's invalid K-rows are fully out
    # of bounds and clamp to 0 via the buffer SRD num_records bound.
    K_ITERS = (K + BLOCK_K - 1) // BLOCK_K
    assert K_ITERS >= 2

    N_TILES_A = BLOCK_M // 64
    N_TILES_B = BLOCK_N // 128
    N_ACCUMS = N_TILES_A * N_TILES_B
    LDS_BLOCK_M = BLOCK_M // 2
    LDS_BLOCK_N = BLOCK_N // 2
    # TN A path uses the wave-coop tr8 transpose load, whose K_log spans
    # [0, 128) and needs 2 G2S rounds = a 16K LDS slot. For BM=128 (natural
    # N_LDS_STEPS_A=1, 8K slot) force 2 rounds / 16K slot to match the K=128
    # transpose-load expectation.
    N_LDS_STEPS_A = max(LDS_BLOCK_M // 64, 2)  # ≥ 2 for tr8 K=128
    N_LDS_STEPS_B = LDS_BLOCK_N // 64
    N_LDS_ROUNDS = max(N_LDS_STEPS_A, N_LDS_STEPS_B)
    # Bank-spread LDS chunk stride: 1056 (=1024+32) un-aligns the per-wave chunk
    # base across LDS banks to remove the transpose-read bank conflict; the G2S
    # writer and S2R reader must use the same value.
    _LDS_CS = 1056
    # a_lds_size: N rounds × 8 waves × chunk_stride. Pad to stride.
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

    def _tn_block_mn(pid, num_pid_m, n_blocks, GM, GN):
        """Tile-id -> (block_m, block_n), resolved at trace time. GN==0: 1D
        GROUP_M super-row swizzle (block_m inner). GN>0: 2D band — N split into
        width-GN bands with GROUP_M inside each, keeping both A and B slabs
        L2-resident. Always a bijection."""
        if GN > 0:
            band_tiles = num_pid_m * GN
            band = pid // band_tiles
            pid_in_band = pid % band_tiles
            band_n0 = band * GN
            rem_n = n_blocks - band_n0
            band_w = arith.select(rem_n < GN, rem_n, fx.Int32(GN))
            nig = GM * band_w
            gid = pid_in_band // nig
            pig = pid_in_band % nig
            fpm = gid * GM
            rem_m = num_pid_m - fpm
            gsm = arith.select(rem_m < GM, rem_m, fx.Int32(GM))
            return fpm + (pig % gsm), band_n0 + (pig // gsm)
        nig = GM * n_blocks
        gid = pid // nig
        pig = pid % nig
        fpm = gid * GM
        rem_m = num_pid_m - fpm
        gsm = arith.select(rem_m < GM, rem_m, fx.Int32(GM))
        return fpm + (pig % gsm), pig // gsm

    @flyc.kernel(known_block_size=[512, 1, 1])
    def kernel_dense_tn(
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
        # Swizzle via plain-Python helper (NOT a kernel `if`: @flyc.kernel
        # wraps each if-branch in its own fn so vars defined inside aren't
        # visible after — see prelude note). Helper builds the expr graph
        # for one Python-selected path (1D GROUP_M or 2D band).
        block_m, block_n = _tn_block_mn(pid, num_pid_m, n_blocks, GROUP_M, group_n)

        # i64 input re-base. A[K,M]/B[K,N] K-row-major: fold column base into SRD; the
        # k*BLOCK_K*c_{m,n} traversal is i64 per load (int32 wraps > 2^31), capped at 4GB.
        cm_i = arith.index_cast(T.index, c_m)
        cn_i = arith.index_cast(T.index, c_n)
        a_base = arith.index_cast(T.index, block_m) * arith.index(BLOCK_M)
        b_base = arith.index_cast(T.index, block_n) * arith.index(BLOCK_N)
        a_nrec = arith.index(K) * cm_i - a_base
        b_nrec = arith.index(K) * cn_i - b_base
        A0_gl_offset = 0
        A1_gl_offset = LDS_BLOCK_M
        B0_gl_offset = 0
        B1_gl_offset = LDS_BLOCK_N

        gA = make_fp8_buffer_tensor_rebased(A, F8_IR_t, a_base, a_nrec)
        gB = make_fp8_buffer_tensor_rebased(B, F8_IR_t, b_base, b_nrec)
        a_div = fx.logical_divide(gA, fx.make_layout(1, 1))
        b_div = fx.logical_divide(gB, fx.make_layout(1, 1))

        # Both A+B use NN-style K-strided global swizzle.
        gl_off_a = compute_global_swizzle_nn(lane_id, wave_id, c_m, N_LDS_ROUNDS)
        gl_off_b = compute_global_swizzle_nn(lane_id, wave_id, c_n, N_LDS_ROUNDS)

        mfma = Mfma16x16x128(N_TILES_A, N_TILES_B)
        if _inplace:
            _mm = _asm_mma_mode
            mfma._do_mma = lambda _a, _b, _c, _m=_mm: asm_mma_do(_a, _b, _c, mode=_m, cbsz=cbsz, blgp=blgp)

        # TN: both A[K,M] and B[K,N] are contraction-traversal operands -> re-base
        # both SRDs per load in i64 mode (each k_offset folds into its i64 base).
        a_rebase = (A, F8_IR_t, a_base, a_nrec) if i64_traverse else None
        b_rebase = (B, F8_IR_t, b_base, b_nrec) if i64_traverse else None
        a_g2s = G2SLoader(
            a_div, gl_off_a, N_LDS_STEPS_A, F8_IR_t, wave_id, chunk_stride=_LDS_CS, rebase=a_rebase
        )
        b_g2s = G2SLoader(
            b_div, gl_off_b, N_LDS_STEPS_B, F8_IR_t, wave_id, chunk_stride=_LDS_CS, rebase=b_rebase
        )
        a_s2r = S2RLoaderTr(
            wave_m,
            N_TILES_A,
            LDS_BLOCK_M // 2,
            inline_asm=_a_inline,
            vmcnt_hint=vmcnt_hint,
            chunk_stride=_LDS_CS,
        )
        b_s2r = S2RLoaderTr(
            wave_n, N_TILES_B, 32, inline_asm=_b_inline, vmcnt_hint=vmcnt_hint, chunk_stride=_LDS_CS
        )
        _out_ty = fx.Float16 if out_fp16 else fx.BFloat16
        store_c = StoreCPerTensor(
            A_scale,
            B_scale,
            C,
            c_m,
            c_n,
            mfma.idx,
            N_TILES_A,
            N_TILES_B,
            _out_ty,
            store_aux=_CSTORE_AUX,
            beta_is_one=beta_is_one,
        )

        c00_frag = [mfma.zero_value] * N_ACCUMS
        c01_frag = [mfma.zero_value] * N_ACCUMS
        c10_frag = [mfma.zero_value] * N_ACCUMS
        c11_frag = [mfma.zero_value] * N_ACCUMS

        # Prelude.
        b_g2s.load(b_cur0, B0_gl_offset + arith.index(0 * BLOCK_K) * cn_i)
        a_g2s.load(a_cur0, A0_gl_offset + arith.index(0 * BLOCK_K) * cm_i)
        b_g2s.load(b_cur1, B1_gl_offset + arith.index(0 * BLOCK_K) * cn_i)
        a_g2s.load(a_cur1, A1_gl_offset + arith.index(0 * BLOCK_K) * cm_i)

        if wave_m == 1:
            rocdl.s_barrier()

        wait_barrier(N_LDS_STEPS_A + N_LDS_STEPS_B)

        b_g2s.load(b_next0, B0_gl_offset + arith.index(1 * BLOCK_K) * cn_i)
        a_g2s.load(a_next0, A0_gl_offset + arith.index(1 * BLOCK_K) * cm_i)
        b_g2s.load(b_next1, B1_gl_offset + arith.index(1 * BLOCK_K) * cn_i)

        wait_barrier(N_LDS_STEPS_A + 2 * N_LDS_STEPS_B)

        # Steady loop: per-iter A-half-0/A-half-1 × {b0,b1} MMA interleaved
        # with the next-tile G2S prefetch and one s_barrier per MMA quadrant.
        # All 7 barriers are load-bearing (dropping any races at the
        # MFMA-reorder level under some GROUP_M; gated by long det runs).
        for k in range_constexpr(K_ITERS - 2):
            # b0 drain=False: the b0 reads are covered by the immediately-
            # following a0 load's lgkmcnt(0) before c00 consumes b0, so the
            # b0 loader's own trailing drain is redundant. (b1 keeps its
            # drain — c01 consumes b1 with no covering drain between.)
            b0_frag = b_s2r.load(b_cur0, drain=False)
            a0_frag = a_s2r.load(a_cur0)
            a_g2s.load(a_next1, A1_gl_offset + arith.index((k + 1) * BLOCK_K) * cm_i)
            rocdl.s_barrier()
            rocdl.s_setprio(1)
            c00_frag = mfma.call(a0_frag, b0_frag, c00_frag)
            rocdl.s_setprio(0)
            rocdl.s_barrier()
            b1_frag = b_s2r.load(b_cur1)
            b_g2s.load(b_cur0, B0_gl_offset + arith.index((k + 2) * BLOCK_K) * cn_i)
            rocdl.s_barrier()
            rocdl.s_setprio(1)
            c01_frag = mfma.call(a0_frag, b1_frag, c01_frag)
            rocdl.s_setprio(0)
            rocdl.s_barrier()
            a1_frag = a_s2r.load(a_cur1)
            a_g2s.load(a_cur0, A0_gl_offset + arith.index((k + 2) * BLOCK_K) * cm_i)
            rocdl.s_barrier()
            rocdl.s_setprio(1)
            c10_frag = mfma.call(a1_frag, b0_frag, c10_frag)
            rocdl.s_setprio(0)
            rocdl.s_barrier()
            b_g2s.load(b_cur1, B1_gl_offset + arith.index((k + 2) * BLOCK_K) * cn_i)
            wait_barrier(2 * N_LDS_STEPS_A + N_LDS_STEPS_B)
            rocdl.s_setprio(1)
            c11_frag = mfma.call(a1_frag, b1_frag, c11_frag)
            rocdl.s_setprio(0)
            rocdl.s_barrier()
            a_cur0, a_next0 = a_next0, a_cur0
            a_cur1, a_next1 = a_next1, a_cur1
            b_cur0, b_next0 = b_next0, b_cur0
            b_cur1, b_next1 = b_next1, b_cur1

        # Epilog 1.
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
        a_g2s.load(a_next1, A1_gl_offset + arith.index((k + 1) * BLOCK_K) * cm_i)
        rocdl.s_barrier()
        rocdl.s_setprio(1)
        c11_frag = mfma.call(a1_frag, b1_frag, c11_frag)
        rocdl.s_setprio(0)
        rocdl.s_barrier()

        a_cur0, a_next0 = a_next0, a_cur0
        a_cur1, a_next1 = a_next1, a_cur1
        b_cur0, b_next0 = b_next0, b_cur0
        b_cur1, b_next1 = b_next1, b_cur1

        # Epilog 2.
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

        wave_n_offset = wave_n * (N_TILES_B * 16)
        wave_m_offset = wave_m * (N_TILES_A * 16)
        base_row = block_m * BLOCK_M + wave_m_offset
        base_col = block_n * BLOCK_N + wave_n_offset
        store_c.store(c00_frag, base_row + 0, base_col + 0)
        store_c.store(c01_frag, base_row + 0, base_col + LDS_BLOCK_N)
        store_c.store(c10_frag, base_row + LDS_BLOCK_M, base_col + 0)
        store_c.store(c11_frag, base_row + LDS_BLOCK_M, base_col + LDS_BLOCK_N)

    @flyc.jit
    def launch_dense_tn(
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
        kernel_dense_tn(
            A,
            B,
            C,
            A_scale,
            B_scale,
            c_m,
            c_n,
            value_attrs=make_value_attrs(waves_per_eu, agpr_alloc, "512,512"),
        ).launch(grid=(grid_x, 1, 1), block=(512, 1, 1), stream=stream)

    return launch_dense_tn


# TN 4-wave (occ=1): one tile per WG, whole K loop in one bare-asm region, accumulators in AGPR.
_TN4_BLOCK_K = 128  # K depth of one mfma_f32_16x16x128 pass, so a K-block is one pass deep
_TN4_CS = 1024  # per-wave LDS chunk stride, shared by the G2S writer and the S2R reader
_TN4_PIN = 8  # first VGPR of the pinned operand-fragment window
_TN4_AGPR = 64  # vector<4xf32> accumulators the AGPR file holds; any beyond it take VGPRs
_TN4_RED_VEC = 8  # out_ty elements one lane folds per load (a b128 request)
_TN4_RED_WPT = 8  # reduce workgroups per split-K window tile
_TN4_HAND_S = 2  # split-K slices the in-register handoff covers
# sc0|sc1: nothing pins a tile's two slices to one XCD's L2, so the handoff is device-scope.
_TN4_HAND_AUX = 17
_TN4_ISSUE_PAD = "s_nop 0"  # one wait state between the body's instruction groups
_TN4_ASM_CACHE: dict = {}


class _Tn4Geom(NamedTuple):
    """A macro tile: ``pools`` are (side, width, nbuf), A first and the deepest last so the
    partial phase drain has somewhere to land; ``wgrid`` is how many waves split one along
    each side's axis."""

    bm: int
    bn: int
    pools: tuple
    bstep: int
    # Boundaries taking a wait state, "<before><after>" over m (mfma), d (ds_read), g (g2s).
    pad: tuple = ("mm", "md", "mg", "dm", "dd", "dg", "gm", "gd", "gg")
    wgrid: tuple = (2, 2)  # a square split shares both operands' reads four ways, not one side's
    # Epilogue: which side the fold reorders, n-fragments it folds into one request, store units
    # a quadrant's rows divide into folded and unfolded, units the peel leads by, drain wait.
    gl_side: int = 1
    fold: int = 8
    store_split: int = 2
    store_split_flat: int = 4
    peel_lead: int = 3
    drain_lgkm: int = 12


_TN4_LINE = 128  # global cache line: a g2s row costs one request per line it spans
_TN4_SQUARE = _Tn4Geom(
    256,
    256,
    ((0, 128, 2), (0, 128, 2), (1, 128, 3), (1, 128, 3)),
    0,
)
# Finer grid for shapes the square tile cannot round onto whole CU passes, same operand bytes.
_TN4_RECT = _Tn4Geom(
    384,
    192,
    ((0, 128, 2), (0, 128, 2), (0, 128, 2), (1, 64, 2), (1, 128, 3)),
    1,
)
_TN4_GEOMS = (_TN4_SQUARE, _TN4_RECT)


def _tn4_pad(groups, pad):
    """Space a phase's instruction groups at the boundaries ``pad`` names: at one wave per
    SIMD the memory issue arrives in bursts that back-pressure the matrix pipe. The last group
    of a phase always takes one, since the drain follows it."""
    kinds, out = "mdg", []
    for i, (k, txt) in enumerate(groups):
        out.append(txt)
        if i + 1 == len(groups) or kinds[k] + kinds[groups[i + 1][0]] in pad:
            out.append(_TN4_ISSUE_PAD)
    return out


def _tn4_phases(geom):
    """K-blocks one main-loop pass consumes: the lcm of the pool depths, so a pass always
    ends with every pool back on buffer 0."""
    n = 1
    for _side, _w, nbuf in geom.pools:
        n = n * nbuf // math.gcd(n, nbuf)
    return n


class _Tn4Pool:
    """One LDS operand pool: `width` columns of its side's macro tile, starting at `col`,
    `nbuf` buffers deep. Every other size follows from the width at BLOCK_K / 256 threads."""

    def __init__(self, side, width, nbuf, col, wsplit=2, nthr=256):
        self.side, self.width, self.nbuf, self.col = side, width, nbuf, col
        self.gcol = col  # column the tile actually reads/writes; _tn4_line_cols may rotate it
        self.gq = 0  # rank among the pools that share a global row order; see _tn4_gl_key
        self.tiles = width // (16 * wsplit)  # mfma tiles one wave's share of the pool holds
        self.steps = width * _TN4_BLOCK_K // (nthr * 16)  # dwordx4 G2S steps per buffer
        self.buf = width * _TN4_BLOCK_K  # bytes per buffer
        self.rs = (width // 16) * _TN4_CS  # halves of a tile's transpose reads, BLOCK_K/2 apart
        assert (nbuf - 1) * self.buf + self.rs < 65536, "buffer delta overflows ds offset"


def _tn4_nthr(geom):
    return 64 * geom.wgrid[0] * geom.wgrid[1]


def _tn4_wave_coord(wave_id, geom):
    wn = geom.wgrid[1]
    return wave_id // wn, wave_id % wn


def _tn4_pools(geom):
    """Expand a geometry's pool spec, giving each pool its column offset within its side."""
    wg, nthr = geom.wgrid, _tn4_nthr(geom)
    col, out = [0, 0], []
    for side, width, nbuf in geom.pools:
        out.append(_Tn4Pool(side, width, nbuf, col[side], wg[side], nthr))
        col[side] += width
    assert col == [geom.bm, geom.bn], "pools must cover the macro tile exactly"
    return out


def _tn4_line_cols(pools, geom, off):
    """Point every pool's ``gcol`` at a cache-line-aligned column of its macro tile: rotating
    a misaligned side's pool window by the complement of the misalignment covers the same
    columns off aligned boundaries. Only windows that rotate on a pool boundary take part."""
    for p in pools:
        extent = geom[p.side]
        r = extent % _TN4_LINE
        if r == 0 or any(q.side == p.side and q.col < r < q.col + q.width for q in pools):
            continue
        p.gcol = fx.Int32(
            _readfirstlane_i32(
                arith.select(
                    off[p.side] % fx.Int32(_TN4_LINE) == fx.Int32(0),
                    fx.Int32((p.col + extent - r) % extent),
                    fx.Int32(p.col),
                )
            )
        )


def _tn4_gl_key(p):
    """Identity of a pool's global-swizzle offset set: pools that read the same operand at the
    same LDS width share one, unless a caller gave them different global row orders (``gq``)."""
    return (p.side, p.width, p.gq)


def _tn4_gl_keys(pools):
    """Every distinct global-swizzle offset set, in pool order."""
    keys = []
    for p in pools:
        if _tn4_gl_key(p) not in keys:
            keys.append(_tn4_gl_key(p))
    return keys


def _dense_tn_slice_div(x, s):
    """``x // s`` for a compile-time split-K slice factor: one shift, or one fixed-point
    reciprocal multiply. Exact for every dividend the slice bounds reach (see _TN4_RCP_MAX)."""
    if s & (s - 1) == 0:
        return fx.Int32(floordiv_pow2(x, s))
    return fx.Int32(fx.Int32(x * (-(-(1 << 16) // s))) >> 16)


def _dense_tn_tile_mn(t, NBM, NBN, group_m, group_n):
    """Tile id -> (block_m, block_n) for the whole-loop TN kernel and its reduce. Both go
    through here so a window tile's slices and its fold agree on the tile id."""
    return block_mn(t, fx.Int32(NBM), fx.Int32(NBN), group_m, group_n)


def _dense_tn_reduce_rows(
    ws_base, c_base, M, N, s, base_row, base_col, rows, bn, tid, nthr, out_ty, col_safe
):
    """Sum slice 0's partial, already in C, and the s-1 WS bands of a ``rows`` x ``bn`` block
    back into C. A lane takes a _TN4_RED_VEC-wide run so a row is full-width requests, and the
    bands keep C's row pitch so band j only moves the SRD base."""
    ir_ty = out_ty.ir_type
    f32v = fx.T.VectorType.get([_TN4_RED_VEC], fx.T.f32())
    outv = fx.T.VectorType.get([_TN4_RED_VEC], ir_ty)
    lanes_per_row = bn // _TN4_RED_VEC
    rows_per_pass = nthr // lanes_per_row
    row = tid // fx.Int32(lanes_per_row)
    col = base_col + (tid % fx.Int32(lanes_per_row)) * fx.Int32(_TN4_RED_VEC)
    mask = None if col_safe else col < fx.Int32(N)
    if nthr % lanes_per_row:
        whole = row < fx.Int32(rows_per_pass)
        mask = whole if mask is None else arith.andi(mask, whole)
    base = row * fx.Int32(N) + col
    dst = make_row_band_resource(c_base, base_row, fx.Int32(M), N, 2)
    src = [dst] + [
        make_row_band_resource(ws_base, fx.Int32(j * M) + base_row, fx.Int32((j + 1) * M), N, 2)
        for j in range_constexpr(s - 1)
    ]
    for p in range_constexpr(ceildiv(rows, rows_per_pass)):
        off = base + fx.Int32(p * rows_per_pass * N)
        m = mask
        if (p + 1) * rows_per_pass > rows:
            live = row < fx.Int32(rows - p * rows_per_pass)
            m = live if mask is None else arith.andi(mask, live)
        acc = None
        for j in range_constexpr(s):
            v = arith.extf(
                f32v,
                _buffer_ops.buffer_load(src[j], off, vec_width=_TN4_RED_VEC, dtype=ir_ty, mask=m),
            )
            acc = v if acc is None else arith.addf(acc, v)
        _buffer_ops.buffer_store(arith.trunc_f(outv, acc), dst, off, mask=m, cache_modifier=_CSTORE_AUX)


def _tn4_handoff(split):
    """True when the split-K window settles in-kernel: two slices can hand off, deeper ones
    need the reduce."""
    return split is not None and split[2] == _TN4_HAND_S


def _tn4_handoff_wait(flag, off, want, rows):
    """Spin until this tile's flag reads ``want``, then pass ``rows`` back out: routing the
    scratch descriptor's size through this block keeps the partial's loads below the wait.
    Slice 0 wants the resting value, as does a tile outside the window."""
    r = _llvm.inline_asm(
        ir.Type.parse("!llvm.struct<(i32, i32, i32)>"),
        [arith._to_raw(v) for v in (flag, off, want, rows)],
        "\n".join(
            [
                "1:",
                "buffer_load_dword $0, $4, $3, 0 offen sc0 sc1",
                "s_waitcnt vmcnt(0)",
                "v_readfirstlane_b32 $1, $0",
                "s_cmp_lg_u32 $1, $5",
                "s_cbranch_scc0 2f",
                "s_sleep 8",
                "s_branch 1b",
                "2:",
                "s_mov_b32 $2, $6",
            ]
        ),
        "=&v,=&s,=&s,s,v,s,s",
        has_side_effects=True,
    )
    return fx.Int32(_llvm.extractvalue(T.i32, r, [2]))


# Accumulators a lane hands over per request; the slot is lane-contiguous so a pair rides one.
_TN4_HAND_PACK = 2


def _tn4_frag_off(wave_id, lane_id, q, nacc):
    """Element offset of accumulator ``q``'s pack in a tile's handoff scratch slot; C's own
    layout would scatter a lane's four values over four rows instead."""
    lane_elems = 4 * _TN4_HAND_PACK
    return (wave_id * fx.Int32(nacc) + fx.Int32(q)) * fx.Int32(256) + lane_id * fx.Int32(lane_elems)


def _tn4_publish_frag(frag, rsrc, wave_id, lane_id, q0, nacc, out_ty, aux):
    """Hand this slice's accumulators to the tile's scratch slot, unscaled: the folding slice
    adds them to its own and its store is what applies the per-tensor scale, once. A slice
    that owns no slot takes a zero-record descriptor and its stores drop."""
    n = _TN4_HAND_PACK
    assert len(frag) % n == 0 and q0 % n == 0, "handoff pack must divide the accumulator run"
    for i in range_constexpr(len(frag) // n):
        vals = [frag[n * i + h].to(out_ty) for h in range(n)]
        _buffer_ops.buffer_store(
            Vec.from_elements([v[j] for v in vals for j in range(4)], out_ty),
            rsrc,
            _tn4_frag_off(wave_id, lane_id, q0 + n * i, nacc),
            cache_modifier=aux,
        )


def _tn4_fold_frag(frag, rsrc, wave_id, lane_id, q0, nacc, out_ty, aux):
    """Add the predecessor slice's published accumulators into this one's. A non-folding
    slice reads a zero-record descriptor, which returns zero and leaves its own values."""
    n = _TN4_HAND_PACK
    assert len(frag) % n == 0 and q0 % n == 0, "handoff pack must divide the accumulator run"
    out = []
    for i in range_constexpr(len(frag) // n):
        got = Vec(
            _buffer_ops.buffer_load(
                rsrc,
                _tn4_frag_off(wave_id, lane_id, q0 + n * i, nacc),
                vec_width=4 * n,
                dtype=out_ty.ir_type,
                cache_modifier=aux,
            )
        ).to(fx.Float32)
        for h in range(n):
            add = Vec.from_elements([got[4 * h + j] for j in range(4)], fx.Float32)
            out.append(Vec(frag[n * i + h]) + add)
    return out


def _dense_tn_wave4_asm(geom, cbsz, blgp):
    """Bare-asm K body for one output tile: the mfma quadrants, the ds_read refills that feed
    them and a later K-block's global->LDS writes, rotated over each pool's buffers. Trip
    count and fused tail passes are runtime SGPRs. Returns (asm, constraints, type)."""
    if (geom, cbsz, blgp) in _TN4_ASM_CACHE:
        return _TN4_ASM_CACHE[(geom, cbsz, blgp)]
    pools = _tn4_pools(geom)
    npool = len(pools)
    ap = [i for i, p in enumerate(pools) if p.side == 0]
    bp = [i for i, p in enumerate(pools) if p.side == 1]
    assert ap == list(range(len(ap))), "A pools must lead the pool list"
    nt = pools[0].tiles  # tiles per A pool; the B side may split its extent unevenly
    assert all(pools[i].tiles == nt for i in ap), "A pools must be equally wide"
    na = sum(pools[i].tiles for i in ap)
    nb = sum(pools[i].tiles for i in bp)
    nacc, n_frag = na * nb, na + nb  # accumulators, live operand fragments
    phases = _tn4_phases(geom)
    pad = geom.pad
    ds_sep = f"\n{_TN4_ISSUE_PAD}\n" if "d" in pad else "\n"
    g2s_sep = f"\n{_TN4_ISSUE_PAD}\n" if "g" in pad else "\n"
    mods = f" cbsz:{cbsz} blgp:{blgp}" if (cbsz or blgp) else ""
    frag = [(i, t) for i in range(npool) for t in range(pools[i].tiles)]
    bcol = [(j, t) for j, i in enumerate(bp) for t in range(pools[i].tiles)]
    qoff, o = {}, 0  # accumulator base of each (A pool, B pool) quadrant
    for ai in range(len(ap)):
        for bi in range(len(bp)):
            qoff[(ai, bi)] = o
            o += nt * pools[bp[bi]].tiles

    # Outputs: accumulators, fragments, counter, per-pool soffset; unwritten "=&s" = regalloc hazard.
    o_cnt = nacc + n_frag
    o_wsoff = [o_cnt + 1 + p for p in range(npool)]
    _at = o_cnt + 1 + npool

    def take(n):
        nonlocal _at
        _at += n
        return list(range(_at - n, _at))

    i_base = [take(2 * p.tiles) for p in pools]
    i_gbase = [take(p.nbuf) for p in pools]
    gl = {k: take(k[1] * _TN4_BLOCK_K // (256 * 16)) for k in _tn4_gl_keys(pools)}
    i_rsrc_a, i_rsrc_b = take(1)[0], take(1)[0]
    i_kstep_a, i_kstep_b = take(1)[0], take(1)[0]
    i_nval, i_tail = take(1)[0], take(1)[0]
    i_soff0 = take(npool)
    i_gl = [gl[_tn4_gl_key(p)] for p in pools]
    i_rsrc = [(i_rsrc_a, i_rsrc_b)[p.side] for p in pools]
    i_kstep = [(i_kstep_a, i_kstep_b)[p.side] for p in pools]

    def ds_reads(rbuf, tt):
        # The buffer delta rides the ds_read immediate, so one address pair covers the pool.
        p, ti = frag[tt - nacc]
        bo = rbuf[p] * pools[p].buf
        v = _TN4_PIN + (tt - nacc) * 8
        ptr = (i_base[p][2 * ti], i_base[p][2 * ti + 1])
        return ds_sep.join(
            f"ds_read_b64_tr_b8 v[{v + 2 * j}:{v + 2 * j + 1}], "
            f"${ptr[j % 2]} offset:{bo + (j // 2) * pools[p].rs}"
            for j in range(4)
        )

    def emit_g2s(wbuf):
        # A pools step-interleaved to share the 128B line; B pools last, per the partial drain.
        order = [(p, st) for st in range(pools[0].steps) for p in ap]
        order += [(p, st) for p in bp for st in range(pools[p].steps)]
        return [
            f"s_add_u32 m0, ${i_gbase[p][wbuf[p]]}, {st * _tn4_nthr(geom) // 64 * _TN4_CS}{g2s_sep}"
            f"buffer_load_dwordx4 ${i_gl[p][st]}, ${i_rsrc[p]}, ${o_wsoff[p]} offen lds"
            for p, st in order
        ]

    def mfma_seq():
        # srcA pool outer (this mfma is srcA-movement sensitive); the diagonal spreads refills.
        return [
            (f"v_mfma_f32_16x16x128_f8f6f4 ${q}, ${at}, ${br}, ${q}{mods}", at, br)
            for q, at, br in _tn4_mfma_order(geom, nt, nb, na, nacc, ap, bp, bcol, qoff, pools)
        ]

    def emit_phase(rbuf, wbuf):
        # Refill a fragment right after its last consumer; global writes take the free slots.
        g2sl, mlist = emit_g2s(wbuf), mfma_seq()
        last = {}
        for mi, (_m, at, bt) in enumerate(mlist):
            last[at] = last[bt] = mi
        busy = {mi for mi, (_m, at, bt) in enumerate(mlist) if last[at] == mi or last[bt] == mi}
        free = [mi for mi in range(len(mlist)) if mi not in busy]
        gap = max(len(free) // len(g2sl), 1)
        gslot = {fi: k // gap for k, fi in enumerate(free) if k % gap == 0 and k // gap < len(g2sl)}
        out, gi, refilled = [], 0, set()
        for mi, (ml, at, bt) in enumerate(mlist):
            out.append((0, ml))
            for rt in (at, bt):
                if last[rt] == mi and rt not in refilled:
                    out.append((1, ds_reads(rbuf, rt)))
                    refilled.add(rt)
            if mi in gslot and gi < len(g2sl):
                out.append((2, g2sl[gi]))
                gi += 1
        out += [(2, g) for g in g2sl[gi:]]
        out += [(1, ds_reads(rbuf, tt)) for tt in range(nacc, nacc + n_frag) if tt not in refilled]
        return _tn4_pad(out, pad)

    # A buffer written this phase is read nbuf-1 later, so only a deeper pool stays in flight.
    tailp = [i for i, p in enumerate(pools) if p.nbuf > 2]
    assert tailp == list(range(npool - len(tailp), npool)), "deep pools must be issued last"
    assert all(phases % p.nbuf == 0 for p in pools), "a pass must end on every pool's buf 0"
    n_out = sum(pools[i].steps for i in tailp)
    drain = f"s_waitcnt vmcnt({n_out}) lgkmcnt({geom.drain_lgkm})\ns_barrier"

    def phase_block(ph):
        blk = emit_phase([(ph + 1) % p.nbuf for p in pools], [ph % p.nbuf for p in pools])
        blk.append(drain)
        return blk + [f"s_add_u32 ${o_wsoff[p]}, ${o_wsoff[p]}, ${i_kstep[p]}" for p in range(npool)]

    L = [f"s_mov_b32 ${o_cnt}, 0"]
    L += [f"s_mov_b32 ${o_wsoff[p]}, ${i_soff0[p]}" for p in range(npool)]
    L += [ds_reads([0] * npool, tt) for tt in range(nacc, nacc + n_frag)]
    # Deeper primes wait here to overlap the ds_read issue; the barrier still guards buf0.
    L += [f"s_waitcnt vmcnt({n_out}) lgkmcnt(0)", "s_barrier", "1:"]
    for ph in range(phases):
        L += phase_block(ph)
    L += [
        f"s_add_u32 ${o_cnt}, ${o_cnt}, {phases}",
        f"s_cmp_lt_u32 ${o_cnt}, ${i_nval}",
        "s_cbranch_scc1 1b",
        # A partial drain needs a next phase, which no longer exists past the exit.
        f"s_cmp_eq_u32 ${i_tail}, 0",
        "s_cbranch_scc1 3f",
        "s_waitcnt vmcnt(0) lgkmcnt(0)",
        "s_barrier",
        "3:",
    ]
    for j in range(phases - 1):  # gated single-K-block passes reusing the loop block
        L += [f"s_cmp_le_u32 ${i_tail}, {j}", f"s_cbranch_scc1 {j + 4}f"]
        L += phase_block(j) + [f"{j + 4}:"]
    L.append("s_waitcnt vmcnt(0) lgkmcnt(0)")

    # Accumulators past the AGPR file take a pinned VGPR window above the fragments.
    nag = min(nacc, _TN4_AGPR)
    vacc = _TN4_PIN + n_frag * 8
    L = [f"v_mov_b32 v{vacc + j}, 0" for j in range(4 * (nacc - nag))] + L
    cons = ",".join(
        ["=a"] * nag
        + [f"=&{{v[{vacc + 4 * i}:{vacc + 4 * i + 3}]}}" for i in range(nacc - nag)]
        + [f"=&{{v[{_TN4_PIN + f * 8}:{_TN4_PIN + f * 8 + 7}]}}" for f in range(n_frag)]
        + ["=&s"] * (1 + npool)
        + ["v"] * sum(2 * p.tiles for p in pools)
        + ["s"] * sum(p.nbuf for p in pools)
        + ["v"] * sum(len(g) for g in gl.values())
        + ["s"] * (6 + npool)
        + [str(q) for q in range(nag)]
    )
    st = (
        "!llvm.struct<("
        + ", ".join(["vector<4xf32>"] * nacc + ["vector<8xi32>"] * n_frag + ["i32"] * (1 + npool))
        + ")>"
    )
    _TN4_ASM_CACHE[(geom, cbsz, blgp)] = ("\n".join(L), cons, st)
    return _TN4_ASM_CACHE[(geom, cbsz, blgp)]


# Must stay top-level: nested in @flyc.kernel its asm cache would look like global drift.
def _dense_tn_wave4_tile(
    d,
    *,
    M,
    N,
    K,
    K_ITERS,
    NBM,
    NBN,
    group_m,
    group_n,
    split,
    handoff,
    store_aux,
    lds,
    geom,
    A,
    B,
    C,
    WS,
    FL,
    A_scale,
    B_scale,
    gl_off,
    wave_id,
    wave_m,
    wave_n,
    lane_id,
    cbsz,
    blgp,
    out_ty,
    col_safe,
):
    """Emit one dispatch id's output tile. ``split`` is None or the (lo, n, s) split-K window
    at the grid tail, whose ids carry a (tile, slice) pair: slice 0 writes C and slice j>0 band
    j-1 of WS. Under ``handoff`` every slice writes C and the last folds its predecessor in."""
    row_shift, store_base, c_rows, flag, flag_off, slot_base = (None,) * 6
    k0 = fx.Int32(0)
    ki = fx.Int32(K_ITERS)
    if split is None:
        t = d
    else:
        lo, nwin, s = split
        win_off = fx.Int32(d) - fx.Int32(lo)
        # sid < 0 = whole tile: the ids below the window take the whole K range.
        sid = _dense_tn_slice_div(win_off, nwin)
        tile_in_window = win_off - sid * fx.Int32(nwin)
        if lo:
            pre = win_off < fx.Int32(0)
            t = _readfirstlane_i32(arith.select(pre, d, fx.Int32(lo) + tile_in_window))
            sid = fx.Int32(_readfirstlane_i32(arith.select(pre, fx.Int32(-1), sid)))
            whole = sid < fx.Int32(0)
            slice_id = fx.Int32(arith.select(whole, fx.Int32(0), sid))
        else:
            t = _readfirstlane_i32(fx.Int32(lo) + tile_in_window)
            sid = fx.Int32(_readfirstlane_i32(sid))
            slice_id = sid
        k0 = _dense_tn_slice_div(fx.Int32(K_ITERS) * slice_id, s)
        ki = (
            fx.Int32(
                arith.select(
                    slice_id + fx.Int32(1) < fx.Int32(s),
                    _dense_tn_slice_div(fx.Int32(K_ITERS) * (slice_id + fx.Int32(1)), s),
                    fx.Int32(K_ITERS),
                )
            )
            - k0
        )
        if lo:
            k0 = fx.Int32(arith.select(whole, fx.Int32(0), k0))
            ki = fx.Int32(arith.select(whole, fx.Int32(K_ITERS), ki))
        ki = fx.Int32(_readfirstlane_i32(ki))
        if handoff:
            # Which side of the protocol this id is on; a tile below the window issues neither.
            slot = geom.bm * geom.bn * 2
            pub = sid == fx.Int32(0)
            fold = sid == fx.Int32(1)
            fold_rec = fx.Int32(_readfirstlane_i32(arith.select(fold, fx.Int32(slot), fx.Int32(0))))
            pub_rec = arith.index_cast(
                T.index, fx.Int32(_readfirstlane_i32(arith.select(pub, fx.Int32(slot), fx.Int32(0))))
            )
            slot_base = arith.index_cast(T.index, fx.Int32(_readfirstlane_i32(tile_in_window))) * arith.index(
                slot
            )
            hand_st = _buffer_ops.create_buffer_resource(
                WS, max_size=False, num_records_bytes=pub_rec, base_byte_offset=slot_base
            )
            nrec = arith.index(4 * nwin)
            if lo:
                nrec = arith.select(whole, arith.index(0), nrec)
            flag = _buffer_ops.create_buffer_resource(FL, max_size=False, num_records_bytes=nrec)
            flag_off = fx.Int32(_readfirstlane_i32(tile_in_window)) * fx.Int32(4)
            flag_val = fx.Int32(1) - slice_id
            # The publishing slice writes no C at all, so its output band bases zero records.
            c_rows = fx.Int32(_readfirstlane_i32(arith.select(pub, fx.Int32(0), fx.Int32(M))))
        else:
            # Slice 0 lands in C and slice j>0 in WS rows [(j-1)*M, j*M): s-1 bands, one stride.
            row_shift = _readfirstlane_i32((slice_id - fx.Int32(1)) * fx.Int32(M))
            store_base = _buffer_ops.extract_base_index(WS)
            home = slice_id < fx.Int32(1)  # whole tiles carry slice 0, so this covers them too
            row_shift = _readfirstlane_i32(arith.select(home, fx.Int32(0), fx.Int32(row_shift)))
            store_base = arith.select(home, _buffer_ops.extract_base_index(C), store_base)

    pools = _tn4_pools(geom)
    phases = _tn4_phases(geom)
    apool = [p for p in pools if p.side == 0]
    bpool = [p for p in pools if p.side == 1]
    block_m, block_n = _dense_tn_tile_mn(t, NBM, NBN, group_m, group_n)
    bm_off = _readfirstlane_i32(block_m) * fx.Int32(geom.bm)
    bn_off = _readfirstlane_i32(block_n) * fx.Int32(geom.bn)
    _tn4_line_cols(pools, geom, (bm_off, bn_off))
    n_main = (ki // phases) * phases
    nval = _readfirstlane_i32(n_main)
    tail = _readfirstlane_i32(ki - n_main)

    # A [K,M] / B [K,N] stride K: fold slice row + tile column into the SRD, num_records clamps.
    F8_IR_t = fx.Float8E4M3FN.ir_type
    k_row = arith.index_cast(T.index, k0) * arith.index(_TN4_BLOCK_K)
    rows = arith.index(K) - k_row
    a_base = k_row * arith.index(M) + arith.index_cast(T.index, bm_off)
    a_nrec = arith.maxsi(rows * arith.index(M) - arith.index_cast(T.index, bm_off), arith.index(0))
    b_base = k_row * arith.index(N) + arith.index_cast(T.index, bn_off)
    b_nrec = arith.maxsi(rows * arith.index(N) - arith.index_cast(T.index, bn_off), arith.index(0))
    gA = make_fp8_buffer_tensor_rebased(A, F8_IR_t, a_base, a_nrec)
    gB = make_fp8_buffer_tensor_rebased(B, F8_IR_t, b_base, b_nrec)

    # One loader pair per (operand, LDS width): pools sharing one differ only in start column.
    gdiv = (fx.logical_divide(gA, fx.make_layout(1, 1)), fx.logical_divide(gB, fx.make_layout(1, 1)))
    s2r, g2s = {}, {}
    for p in pools:
        if (p.side, p.width) in s2r:
            continue
        s2r[(p.side, p.width)] = S2RLoaderTr(
            wave_m if p.side == 0 else wave_n,
            n_tiles=p.tiles,
            tile_stride=p.tiles * 16,
            n_waves=_tn4_nthr(geom) // 64,
            chunk_stride=_TN4_CS,
            width=p.width,
            wswz=True,  # wave bank-swizzle (matches gl_off in the kernel body)
        )
        g2s[_tn4_gl_key(p)] = G2SLoader(
            gdiv[p.side],
            gl_off[_tn4_gl_key(p)],
            p.steps,
            F8_IR_t,
            wave_id,
            chunk_stride=_TN4_CS,
        )

    mfma = {p.tiles: Mfma16x16x128(apool[0].tiles, p.tiles) for p in bpool}
    if c_rows is None:
        c_rows = fx.Int32(M) if row_shift is None else fx.Int32(M) + row_shift
    store_c = {
        nb: (StoreCPerTensorPairN if col_safe and nb % 2 == 0 else StoreCPerTensor)(
            A_scale,
            B_scale,
            C,
            c_rows,
            fx.Int32(N),
            m.idx,
            apool[0].tiles,
            nb,
            out_ty,
            col_safe=col_safe,
            store_aux=store_aux,
            c_base=store_base,
        )
        for nb, m in mfma.items()
    }

    a_k = arith.index(_TN4_BLOCK_K) * arith.index(M)
    b_k = arith.index(_TN4_BLOCK_K) * arith.index(N)
    pool_lds = [getattr(lds, f"p{i}") for i in range(len(pools))]
    for b in range(max(p.nbuf for p in pools)):
        for i, p in enumerate(pools):
            if b < p.nbuf:
                kb = b * (a_k, b_k)[p.side]
                col = p.gcol if isinstance(p.gcol, int) else arith.index_cast(T.index, p.gcol)
                g2s[_tn4_gl_key(p)].load(pool_lds[i], col + kb, base_off=fx.Int32(b * p.buf))
    # Covers the buf0 primes only; the deeper ones are waited on inside the asm.
    wait_barrier(sum((p.nbuf - 1) * p.steps for p in pools))

    # A pool's buffers are read the same way, so only buffer 0 needs live address VGPRs.
    ins = [
        v for i, p in enumerate(pools) for pair in s2r[(p.side, p.width)].base_addr(pool_lds[i]) for v in pair
    ]
    ins += [
        rocdl.readfirstlane(
            T.i32,
            fx.Int32(fx.ptrtoint(pool_lds[i].ptr))
            + fx.Int32(b * p.buf)
            + fx.Int32(wave_id) * fx.Int32(_TN4_CS),
        )
        for i, p in enumerate(pools)
        for b in range(p.nbuf)
    ]
    for key in _tn4_gl_keys(pools):
        ins += [fx.Int32(o) for o in gl_off[key]]
    kstep_a = rocdl.readfirstlane(T.i32, fx.Int32(_TN4_BLOCK_K) * fx.Int32(M))
    kstep_b = rocdl.readfirstlane(T.i32, fx.Int32(_TN4_BLOCK_K) * fx.Int32(N))
    ins += [
        _buffer_ops.create_buffer_resource(
            A, max_size=False, num_records_bytes=a_nrec, base_byte_offset=a_base
        ),
        _buffer_ops.create_buffer_resource(
            B, max_size=False, num_records_bytes=b_nrec, base_byte_offset=b_base
        ),
        kstep_a,
        kstep_b,
        nval,
        tail,
    ]
    for p in pools:
        step = fx.Int32(p.nbuf) * (kstep_a, kstep_b)[p.side]
        if isinstance(p.gcol, int):
            soff = step if p.gcol == 0 else fx.Int32(p.gcol) + step
        else:
            soff = p.gcol + step
        ins.append(rocdl.readfirstlane(T.i32, soff))
    nacc = sum(p.tiles for p in apool) * sum(p.tiles for p in bpool)
    ins += [mfma[bpool[0].tiles].zero_value] * min(nacc, _TN4_AGPR)

    asm, cons, st = _dense_tn_wave4_asm(geom, cbsz, blgp)
    r = _llvm.inline_asm(ir.Type.parse(st), [arith._to_raw(v) for v in ins], asm, cons, has_side_effects=True)
    acc_ty = ir.Type.parse("vector<4xf32>")
    res = [Vec(_llvm.extractvalue(acc_ty, r, [q])) for q in range_constexpr(nacc)]

    row_q = bm_off + wave_m * fx.Int32(apool[0].tiles * 16)
    base_row = row_q if row_shift is None else row_q + row_shift

    if handoff:
        _tn4_publish_frag(res, hand_st, wave_id, lane_id, 0, nacc, out_ty, handoff)
        # The raise trails this tile's stores; the second barrier keeps a wave past the poll
        # from resetting the flag under one still spinning.
        wait_barrier(0)
        recs = _tn4_handoff_wait(flag, flag_off, slice_id, fold_rec)
        wait_barrier(0)
        _buffer_ops.buffer_store(
            flag_val,
            flag,
            flag_off,
            mask=lane_id == fx.Int32(0),
            cache_modifier=_TN4_HAND_AUX,
            offset_is_bytes=True,
        )
        hand_ld = _buffer_ops.create_buffer_resource(
            WS,
            max_size=False,
            num_records_bytes=arith.index_cast(T.index, recs),
            base_byte_offset=slot_base,
        )
    q = 0
    for pa in apool:
        for pb in bpool:
            n = pa.tiles * pb.tiles
            col_q = bn_off + wave_n * fx.Int32(pb.tiles * 16)
            frag = res[q : q + n]
            if handoff:
                frag = _tn4_fold_frag(frag, hand_ld, wave_id, lane_id, q, nacc, out_ty, handoff)
            store_c[pb.tiles].store(frag, base_row + pa.gcol, col_q + pb.gcol)
            q += n


_TN4_SPLIT_S = (2, 3, 4)  # slice factors; an odd one is fine, the slices stay co-resident
_TN4_RCP_MAX = 1 << 15  # exactness bound on _dense_tn_slice_div's dividend
_TN4_OUT_ALIGN = 8  # out_ty elements the split-K bands keep aligned at C's row pitch


def _tn4_split_rounds(tiles, n, s, ncu):
    """Rounds the split-K window's ``n * s`` slices occupy."""
    return ceildiv(n * s, ncu) if tiles <= ncu else (1 if s * n <= ncu else s)


def _dense_tn_split(tiles, k_iters, ncu, phases):
    """Split-K window ``(lo, n, s)`` for one dense TN grid, or None. With one uniform K the
    makespan quantizes only on the last partial round, so slice its ``rem`` tiles s ways;
    keep the shortest s that still fits the window inside one round."""
    rem = tiles % ncu
    if rem == 0:
        return None
    best, best_rounds, best_s = 1, 1, 1
    for s in _TN4_SPLIT_S:
        if k_iters < phases * s or k_iters * (s - 1) >= _TN4_RCP_MAX:
            continue  # every slice must keep a whole main-loop pass, and stay exactly divisible
        rounds = _tn4_split_rounds(tiles, rem, s, ncu)
        if rounds * best_s < best_rounds * s:
            best, best_rounds, best_s = s, rounds, s
    return None if best == 1 else (tiles - rem, rem, best)


_NUM_CUS = 0


def _dense_num_cus():
    """Device CU count, memoised: the property query is otherwise on the dispatch path."""
    global _NUM_CUS
    if not _NUM_CUS:
        _NUM_CUS = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
    return _NUM_CUS


def _compile_dense_tn_wave4(
    M: int,
    N: int,
    K: int,
    group_m: int,
    group_n: int,
    geom=_TN4_SQUARE,
    cbsz: int = 0,  # srcA fp8 fmt: 0=E4M3, 1=E5M2
    blgp: int = 0,  # srcB fp8 fmt: 0=E4M3, 1=E5M2
    out_fp16: bool = False,
):
    """4-wave (occ=1) dense TN C[M,N] = A[K,M]^T @ B[K,N] over ``geom``'s macro tile, one
    tile per workgroup. Returns (launch, split-K scratch band count)."""
    BM, BN = geom.bm, geom.bn
    _pools = _tn4_pools(geom)
    phases = _tn4_phases(geom)
    NBM, NBN = ceildiv(M, BM), ceildiv(N, BN)
    TILES = NBM * NBN
    K_ITERS = ceildiv(K, _TN4_BLOCK_K)
    assert K_ITERS >= phases, "4-wave dense TN needs a K of at least one main-loop pass"
    split = _dense_tn_split(TILES, K_ITERS, _dense_num_cus(), phases)
    _out_ty = fx.Float16 if out_fp16 else fx.BFloat16
    # Doubles as the switch and the traffic policy; a closure scalar lands in the jit key.
    _hand = _TN4_HAND_AUX if _tn4_handoff(split) else 0
    # Slice 0 keeps C, so the WS holds s-1 bands and each window tile takes s-1 extra WGs.
    _bands = 0 if split is None else split[2] - 1
    _NWIN = 0 if split is None else split[1]
    _GRID = TILES + _NWIN * (split[2] - 1 if split is not None else 0)

    # ONE field per pool holding its buffers back to back: separate fields may reorder.
    SharedStorage = fx.struct(
        type(
            "SharedStorage",
            (),
            {
                "__annotations__": {
                    f"p{i}": fx.Array[fx.Float8E4M3FN, p.nbuf * p.buf, 16] for i, p in enumerate(_pools)
                }
            },
        )
    )

    @flyc.kernel(known_block_size=[256, 1, 1])
    def kernel_dense_tn_wave4(
        A: fx.Tensor,
        B: fx.Tensor,
        C: fx.Tensor,
        A_scale: fx.Tensor,
        B_scale: fx.Tensor,
        WS: fx.Tensor,
        FL: fx.Tensor,
    ):
        _ = str(fx.thread_idx.x)
        lane_id = fx.thread_idx.x % 64
        wave_id = fx.thread_idx.x // 64
        _dense_tn_wave4_tile(
            fx.block_idx.x,
            M=M,
            N=N,
            K=K,
            K_ITERS=K_ITERS,
            NBM=NBM,
            NBN=NBN,
            group_m=group_m,
            group_n=group_n,
            geom=geom,
            split=split,
            handoff=_hand,
            store_aux=_CSTORE_AUX,
            lds=fx.SharedAllocator().allocate(SharedStorage).peek(),
            A=A,
            B=B,
            C=C,
            WS=WS,
            FL=FL,
            A_scale=A_scale,
            B_scale=B_scale,
            gl_off={
                k: compute_global_swizzle_nn(
                    lane_id,
                    wave_id,
                    M if k[0] == 0 else N,
                    k[1] * _TN4_BLOCK_K // (256 * 16),
                    width=k[1],
                    wswz=True,
                )
                for k in _tn4_gl_keys(_pools)
            },
            wave_id=wave_id,
            wave_m=wave_id // 2,
            wave_n=wave_id % 2,
            lane_id=lane_id,
            cbsz=cbsz,
            blgp=blgp,
            out_ty=_out_ty,
            col_safe=N % BN == 0,
        )

    _ATTRS = make_value_attrs(1, 0, "256,256")
    _LO = 0 if split is None else split[0]
    _S = 0 if split is None else split[2]
    _RED_GRID = 0 if _hand else _NWIN * _TN4_RED_WPT
    _RED_ROWS = BM // _TN4_RED_WPT

    @flyc.kernel(known_block_size=[256, 1, 1])
    def kernel_dense_tn_wave4_reduce(C: fx.Tensor, WS: fx.Tensor):
        """Fold the split-K window's slice bands into C, one row slab per workgroup. It runs
        the whole grid, so the fold spreads over every CU instead of only the window's."""
        tid = fx.thread_idx.x
        wt = _dense_tn_slice_div(fx.Int32(fx.block_idx.x), _TN4_RED_WPT)
        part = fx.Int32(fx.block_idx.x) - wt * fx.Int32(_TN4_RED_WPT)
        block_m, block_n = _dense_tn_tile_mn(
            _readfirstlane_i32(fx.Int32(_LO) + wt), NBM, NBN, group_m, group_n
        )
        _dense_tn_reduce_rows(
            _buffer_ops.extract_base_index(WS),
            _buffer_ops.extract_base_index(C),
            M,
            N,
            _S,
            _readfirstlane_i32(block_m) * fx.Int32(BM) + part * fx.Int32(_RED_ROWS),
            _readfirstlane_i32(block_n) * fx.Int32(BN),
            _RED_ROWS,
            BN,
            tid,
            256,
            _out_ty,
            N % BN == 0,
        )

    @flyc.jit
    def launch_dense_tn_wave4(
        A: fx.Tensor,
        B: fx.Tensor,
        C: fx.Tensor,
        A_scale: fx.Tensor,
        B_scale: fx.Tensor,
        WS: fx.Tensor,
        FL: fx.Tensor,
        stream: fx.Stream,
    ):
        kernel_dense_tn_wave4(A, B, C, A_scale, B_scale, WS, FL, value_attrs=_ATTRS).launch(
            grid=(_GRID, 1, 1), block=(256, 1, 1), stream=stream
        )
        if const_expr(_RED_GRID):
            kernel_dense_tn_wave4_reduce(C, WS).launch(
                grid=(_RED_GRID, 1, 1), block=(256, 1, 1), stream=stream
            )

    return launch_dense_tn_wave4, _bands


# Accumulators live in the AGPR file: moving them to the arch file to save the epilogue's
# v_accvgpr_read trades one hazard for another -- an mfma writing an arch VGPR that a VALU then
# reads needs wait states the occ=1 lane has nothing to fill, and it measured a wash.
_NT4_SQUARE = _Tn4Geom(
    256,
    256,
    ((0, 128, 2), (0, 128, 2), (1, 128, 3), (1, 128, 3)),
    0,
    drain_lgkm=6,  # NT's plain reads leave fewer in flight than the transpose path's
)
_NT4_ASM_CACHE: dict = {}
_NT4_BAND = 64  # B rows one wave's four n-fragments span in a pool


def _nt4_pools(geom, fold):
    """``_tn4_pools`` with the B pools grouped by the epilogue fold: the pools of one group hold
    interleaved columns of one output run, so each gets its own global row order (``gq``) and
    they all address the group's rows from its base."""
    pools = _tn4_pools(geom)
    bpool = [p for p in pools if p.side == geom.gl_side]
    npg = max(1, fold // bpool[0].tiles)  # pools per group
    for i, p in enumerate(bpool):
        p.gq = i % npg
        p.gcol = p.col - p.gq * p.width
    return pools


def _nt4_fold_gl_off(lane_id, wave_id, K, n_rounds, gq, fold):
    """B-side global row order for the folded-N epilogue: sourcing LDS row 16*t + m from group
    row fold*m + t + 4*gq leaves a lane holding adjacent columns, so the epilogue writes
    whole lines. Only the source row moves; the LDS slot and its swizzle are untouched."""
    offs = compute_global_swizzle(lane_id, wave_id, K, n_rounds, preshuffled=False)
    rows_per_round = (fx.block_dim.x // 64) * 8
    out = []
    for r in range_constexpr(n_rounds):
        row = lane_id // 8 + wave_id * 8 + r * rows_per_round
        band = row % _NT4_BAND
        delta = (
            (fold * 16 - _NT4_BAND) * (row // _NT4_BAND)
            + (fold - 1) * (band % 16)
            - 15 * (band // 16)
            + 4 * gq
        )
        out.append(offs[r] + delta * K)
    return out


def _tn4_mfma_order(geom, nt, nb, na, nacc, ap, bp, bcol, qoff, pools):
    """(accumulator, srcA, srcB) of one phase's mfma, in issue order: the srcA pool is outer
    because this mfma is srcA-movement sensitive, and the diagonal spreads the refills."""
    bm, bn = 2, geom.bstep or nb // 2
    n_row_steps, n_col_steps = nt // bm, nb // bn
    for d in range(n_row_steps + n_col_steps - 1):
        for iib in range(n_row_steps):
            if not 0 <= d - iib < n_col_steps:
                continue
            for di in range(bm):
                for ah in range(len(ap)):
                    for dj in range(bn):
                        col, ii = (d - iib) * bn + dj, iib * bm + di
                        bi, bt = bcol[col]
                        yield (
                            qoff[(ah, bi)] + ii * pools[bp[bi]].tiles + bt,
                            nacc + ah * nt + ii,
                            nacc + na + col,
                        )


def _dense_nt_wave4_asm(geom, k_iters, cbsz, blgp, fold, tr_b=False, b_kstep=_TN4_BLOCK_K, carry=False):
    """Bare-asm K body for one NT output tile; ``carry`` hands the closing fills to the next tile."""
    key = (geom, k_iters, cbsz, blgp, fold, tr_b, b_kstep, carry)
    if key in _NT4_ASM_CACHE:
        return _NT4_ASM_CACHE[key]
    pools = _nt4_pools(geom, fold)
    npool = len(pools)
    nwaves = _tn4_nthr(geom) // 64  # the g2s chunks of one step, one per wave
    ap = [i for i, p in enumerate(pools) if p.side == 0]
    bp = [i for i, p in enumerate(pools) if p.side == 1]
    rp = ap + bp
    assert ap == list(range(len(ap))), "A pools must lead the pool list"
    nt = pools[0].tiles  # tiles per A pool; the B side may split its extent unevenly
    assert all(pools[i].tiles == nt for i in ap), "A pools must be equally wide"
    na = sum(pools[i].tiles for i in ap)
    nb = sum(pools[i].tiles for i in bp)
    nacc, n_frag = na * nb, na + nb  # accumulators, live operand fragments
    phases = _tn4_phases(geom)
    n_main = (k_iters // phases) * phases
    tail = k_iters - n_main
    assert n_main >= phases, "the NT whole-loop needs a K of at least one main-loop pass"
    assert not (carry and tail), "a carried ring needs the trip to end on a whole pass"
    pad = geom.pad
    ds_sep = f"\n{_TN4_ISSUE_PAD}\n" if "d" in pad else "\n"
    g2s_sep = f"\n{_TN4_ISSUE_PAD}\n" if "g" in pad else "\n"
    mods = f" cbsz:{cbsz} blgp:{blgp}" if (cbsz or blgp) else ""
    frag = [(i, t) for i in rp for t in range(pools[i].tiles)]
    bcol = [(j, t) for j, i in enumerate(bp) for t in range(pools[i].tiles)]
    qoff, o = {}, 0  # accumulator base of each (A pool, B pool) quadrant
    for ai in range(len(ap)):
        for bi in range(len(bp)):
            qoff[(ai, bi)] = o
            o += nt * pools[bp[bi]].tiles

    o_cnt = nacc + n_frag
    o_wsoff = [o_cnt + 1 + p for p in range(npool)]
    _at = o_cnt + 1 + npool

    def take(n):
        nonlocal _at
        _at += n
        return list(range(_at - n, _at))

    tr = [bool(tr_b) and p.side == 1 for p in pools]
    # A transposed pool carries its column XOR in the address, so every one of its tiles
    # brings a pair; a plain pool's row-tile step is a read immediate and one pair covers it.
    # A plain pool's fragment is one lane's share of a 16xBLOCK_K tile, read as b128 halves.
    reads = 16 * _TN4_BLOCK_K // 64 // 16
    i_base = {i: take(2 * pools[i].tiles if tr[i] else reads) for i in rp}
    i_gbase = [take(p.nbuf) for p in pools]
    gl: dict = {}
    for p in pools:
        if _tn4_gl_key(p) not in gl:
            gl[_tn4_gl_key(p)] = take(p.steps)
    i_rsrc_a, i_rsrc_b = take(1)[0], take(1)[0]
    i_rsrc2 = [take(1)[0], take(1)[0]] if carry else []
    i_soff0 = take(npool)
    cbuf = _nt4_carry_bufs(pools, carry)
    i_pfsoff = [take(len(b)) for b in cbuf]
    i_gl = [gl[_tn4_gl_key(p)] for p in pools]
    i_rsrc = [(i_rsrc_a, i_rsrc_b)[p.side] for p in pools]
    i_pfrsrc = [i_rsrc2[p.side] for p in pools] if carry else []
    ksteps = [(b_kstep if p.side else _TN4_BLOCK_K) for p in pools]

    def ds_reads(rbuf, tt):
        p, ti = frag[tt - nacc]
        v = _TN4_PIN + (tt - nacc) * 8
        bo = rbuf[p] * pools[p].buf
        if tr[p]:
            ptr = (i_base[p][2 * ti], i_base[p][2 * ti + 1])
            return ds_sep.join(
                f"ds_read_b64_tr_b8 v[{v + 2 * j}:{v + 2 * j + 1}], "
                f"${ptr[j % 2]} offset:{bo + (j // 2) * pools[p].rs}"
                for j in range(4)
            )
        bo += ti * 16 * _TN4_BLOCK_K
        assert bo < 65536, "buffer + tile delta overflows the ds offset"
        return ds_sep.join(
            f"ds_read_b128 v[{v + 4 * h}:{v + 4 * h + 3}], ${i_base[p][h]} offset:{bo}" for h in range(reads)
        )

    def src2(q, init):
        # An init phase writes its quadrant from the mfma itself, so nothing zeroes it first.
        return "0" if init else f"${q}"

    def carried(p, left):
        # Past the last phase a fill would fetch a K-block nothing reads; a successor takes it.
        return carry and pools[p].nbuf > left

    def emit_g2s(wbuf, left):
        order = [(p, st) for st in range(pools[0].steps) for p in ap]
        order += [(p, st) for p in bp for st in range(pools[p].steps)]
        assert all(not carried(p, left) or wbuf[p] < len(i_pfsoff[p]) for p, _st in order), (
            "a carried fill must aim at a buffer the next tile top leaves alone"
        )
        return [
            f"s_add_u32 m0, ${i_gbase[p][wbuf[p]]}, {st * nwaves * _TN4_CS}{g2s_sep}"
            f"buffer_load_dwordx4 ${i_gl[p][st]}, "
            f"${i_pfrsrc[p] if carried(p, left) else i_rsrc[p]}, "
            f"${i_pfsoff[p][wbuf[p]] if carried(p, left) else o_wsoff[p]} offen lds"
            for p, st in order
        ]

    def mfma_seq(init):
        return [
            (f"v_mfma_f32_16x16x128_f8f6f4 ${q}, ${at}, ${br}, {src2(q, init)}{mods}", at, br, q)
            for q, at, br in _tn4_mfma_order(geom, nt, nb, na, nacc, ap, bp, bcol, qoff, pools)
        ]

    def emit_phase(rbuf, wbuf, init, left):
        g2sl, mlist = emit_g2s(wbuf, left), mfma_seq(init)
        last = {}
        for mi, (_m, at, bt, _q) in enumerate(mlist):
            last[at] = last[bt] = mi
        busy = {mi for mi, (_m, at, bt, _q) in enumerate(mlist) if last[at] == mi or last[bt] == mi}
        free = [mi for mi in range(len(mlist)) if mi not in busy]
        # Spread over the free slots, not stepped by their ratio, which bares the phase tail.
        gslot = {free[k * len(free) // len(g2sl)]: k for k in range(len(g2sl))}
        out, gi, refilled = [], 0, set()
        for mi, (ml, at, bt, _q) in enumerate(mlist):
            out.append((0, ml))
            for rt in (at, bt):
                if last[rt] == mi and rt not in refilled:
                    refilled.add(rt)
                    out.append((1, ds_reads(rbuf, rt)))
            if mi in gslot and gi < len(g2sl):
                out.append((2, g2sl[gi]))
                gi += 1
        out += [(2, g) for g in g2sl[gi:]]
        out += [(1, ds_reads(rbuf, tt)) for tt in range(nacc, nacc + n_frag) if tt not in refilled]
        return _tn4_pad(out, pad)

    tailp = [i for i, p in enumerate(pools) if p.nbuf > 2]
    assert tailp == list(range(npool - len(tailp), npool)), "deep pools must be issued last"
    assert all(phases % p.nbuf == 0 for p in pools), "a pass must end on every pool's buf 0"
    n_out = sum(pools[i].steps for i in tailp)

    def n_flight(lf):
        """Fills the phase drain may leave outstanding: the trailing run nothing later waits on."""
        n, l = 0, lf
        while True:
            for i in reversed(range(npool)):
                if not (carried(i, l) or (l == lf and i in tailp)):
                    return n
                n += pools[i].steps
            l += 1

    def phase_block(ph, init=False, left=None):
        lf = k_iters if left is None else left
        blk = emit_phase([(ph + 1) % p.nbuf for p in pools], [ph % p.nbuf for p in pools], init, lf)
        blk.append(f"s_waitcnt vmcnt({n_flight(lf)}) lgkmcnt({geom.drain_lgkm})\ns_barrier")
        return blk + [f"s_add_u32 ${o_wsoff[p]}, ${o_wsoff[p]}, {ksteps[p]}" for p in range(npool)]

    def pass_block(first, last, left0=None):
        blk = []
        for ph in range(phases - int(last)):
            blk += phase_block(ph, first and ph == 0, None if left0 is None else left0 - ph)
        return blk

    def loop_block(trip):
        return (
            ["1:"]
            + pass_block(False, False)
            + [
                f"s_add_u32 ${o_cnt}, ${o_cnt}, {phases}",
                f"s_cmp_lt_u32 ${o_cnt}, {trip}",
                "s_cbranch_scc1 1b",
            ]
        )

    L = [f"s_mov_b32 ${o_cnt}, 0"]
    L += [f"s_mov_b32 ${o_wsoff[p]}, ${i_soff0[p]}" for p in range(npool)]
    L += [ds_reads([0] * npool, tt) for tt in range(nacc, nacc + n_frag)]
    L += [f"s_waitcnt vmcnt({n_out}) lgkmcnt(0)", "s_barrier"]
    # The first pass is peeled so its opening phase writes the accumulators rather than
    # accumulating, sparing the zero-init; the closing phase is peeled the other way and
    # handed back, its mfma the window an occ=1 epilogue has nowhere else to sink into.
    if tail:
        L += pass_block(True, False)
        if n_main > phases:
            L += loop_block(n_main - phases)
        L += ["s_waitcnt vmcnt(0) lgkmcnt(0)", "s_barrier"]
        for j in range(tail - 1):
            L += phase_block(j)
    else:
        _tl = phases - 1 if carry else None
        L += pass_block(True, n_main == phases, _tl if n_main == phases else None)
        if n_main > 2 * phases:
            L += loop_block(n_main - 2 * phases)
        if n_main > phases:
            L += pass_block(False, True, _tl)
    L.append("s_waitcnt lgkmcnt(0)" if carry else "s_waitcnt vmcnt(0) lgkmcnt(0)")
    peel = [(q, at - nacc, br - nacc) for _m, at, br, q in mfma_seq(False)]

    nag = min(nacc, _TN4_AGPR)
    vacc = _TN4_PIN + n_frag * 8
    cons = ",".join(
        ["=a"] * nag
        + [f"=&{{v[{vacc + 4 * i}:{vacc + 4 * i + 3}]}}" for i in range(nacc - nag)]
        + [f"=&{{v[{_TN4_PIN + f * 8}:{_TN4_PIN + f * 8 + 7}]}}" for f in range(n_frag)]
        + ["=&s"] * (1 + npool)
        + ["v"] * sum(len(i_base[i]) for i in rp)
        + ["s"] * sum(p.nbuf for p in pools)
        + ["v"] * sum(len(g) for g in gl.values())
        + ["s"] * (2 + len(i_rsrc2) + npool + sum(len(s) for s in i_pfsoff))
    )
    st = (
        "!llvm.struct<("
        + ", ".join(["vector<4xf32>"] * nacc + ["vector<8xi32>"] * n_frag + ["i32"] * (1 + npool))
        + ")>"
    )
    _NT4_ASM_CACHE[key] = ("\n".join(L), cons, st, peel)
    return _NT4_ASM_CACHE[key]


def _nt4_tile_window(d, M, N, K, NBM, NBN, group_m, group_n, num_xcd, geom, tr_b):
    """One dispatch id's operand windows, num_records bounded so read-ahead past the tile drops."""
    pid = xcd_remap_pid(d, NBM * NBN, num_xcd)
    block_m, block_n = block_mn(pid, fx.Int32(NBM), fx.Int32(NBN), group_m, group_n)
    bm_off = _readfirstlane_i32(block_m) * fx.Int32(geom.bm)
    bn_off = _readfirstlane_i32(block_n) * fx.Int32(geom.bn)
    a_base = arith.index_cast(T.index, bm_off) * arith.index(K)
    a_nrec = arith.minui(arith.index(M * K) - a_base, arith.index(geom.bm * K))
    if tr_b:
        b_base = arith.index_cast(T.index, bn_off)
        b_nrec = arith.index(K * N) - b_base
    else:
        b_base = arith.index_cast(T.index, bn_off) * arith.index(K)
        b_nrec = arith.minui(arith.index(N * K) - b_base, arith.index(geom.bn * K))
    return bm_off, bn_off, ((a_base, a_nrec), (b_base, b_nrec))


def _nt4_buf_off(p, b, K, b_kstep, tr_b):
    return p.gcol + b * b_kstep if tr_b and p.side == 1 else p.gcol * K + b * _TN4_BLOCK_K


def _nt4_g2s(A, B, win, pools, gl_off, wave_id):
    src = [
        fx.logical_divide(
            make_fp8_buffer_tensor_rebased(g, fx.Float8E4M3FN.ir_type, base, nrec),
            fx.make_layout(1, 1),
        )
        for g, (base, nrec) in ((A, win[0]), (B, win[1]))
    ]
    g2s = {}
    for p in pools:
        g2s.setdefault(
            _tn4_gl_key(p),
            G2SLoader(
                src[p.side],
                gl_off[_tn4_gl_key(p)],
                p.steps,
                fx.Float8E4M3FN.ir_type,
                wave_id,
                chunk_stride=_TN4_CS,
            ),
        )
    return g2s


def _nt4_prime(pools, pool_lds, g2s, bufs, K, b_kstep, tr_b):
    for b in range(max(p.nbuf for p in pools)):
        for i, p in enumerate(pools):
            if b in bufs[i]:
                g2s[_tn4_gl_key(p)].load(
                    pool_lds[i], _nt4_buf_off(p, b, K, b_kstep, tr_b), base_off=fx.Int32(b * p.buf)
                )


def _nt4_carry_bufs(pools, carry):
    """Buffers the closing phases hand on: all but the last, whose K-block this trip still reads."""
    return [list(range(p.nbuf - 1)) if carry else [] for p in pools]


def _nt4_xpose_lds(p, lds, wave_id, n_waves):
    """Per-wave epilogue scratch in the buffer the tile top re-primes, which its barrier orders."""
    assert n_waves * XPOSE_SLOTS * XPOSE_SLOT <= p.buf, "the transpose scratch must fit the buffer"
    return (
        fx.Int32(fx.ptrtoint(lds.ptr))
        + fx.Int32((p.nbuf - 1) * p.buf)
        + fx.Int32(wave_id) * fx.Int32(XPOSE_SLOTS * XPOSE_SLOT)
    )


def _dense_nt_wave4_tile(
    d,
    *,
    M,
    N,
    K,
    K_ITERS,
    NBM,
    NBN,
    group_m,
    group_n,
    num_xcd,
    store_aux,
    lds,
    geom,
    A,
    B,
    C,
    A_scale,
    B_scale,
    gl_off,
    wave_id,
    wave_m,
    wave_n,
    cbsz,
    blgp,
    out_ty,
    col_safe,
    pair_n,
    fold,
    tr_b=False,
    d_next=None,
    scale=None,
    beta_is_one=False,
):
    """Emit one dispatch id's output tile; ``d_next`` carries the pool ring on, ``scale`` is hoisted."""
    pools = _nt4_pools(geom, fold)
    apool = [p for p in pools if p.side == 0]
    bpool = [p for p in pools if p.side == 1]
    assert len({p.tiles for p in pools if p.side == 1}) == 1, "B pools must be equally wide"
    b_kstep = _TN4_BLOCK_K * N if tr_b else _TN4_BLOCK_K
    pool_lds = [getattr(lds, f"p{i}") for i in range(len(pools))]
    n_waves = _tn4_nthr(geom) // 64
    carry = d_next is not None
    cbuf = _nt4_carry_bufs(pools, carry)
    wargs = (M, N, K, NBM, NBN, group_m, group_n, num_xcd, geom, tr_b)
    bm_off, bn_off, win = _nt4_tile_window(d, *wargs)

    s2r = {}
    for p in pools:
        s2r.setdefault(
            (p.side, p.width),
            S2RLoaderTr(
                wave_n,
                p.tiles,
                p.tiles * 16,
                chunk_stride=_TN4_CS,
                n_waves=n_waves,
                width=p.width,
                wswz=True,  # wave bank-swizzle (matches gl_off in the kernel body)
            )
            if tr_b and p.side == 1
            else S2RLoader(wave_m if p.side == 0 else wave_n, p.tiles),
        )

    # A fraction of a quadrant's rows is the store unit, so the peel has more mfma to hide a
    # store behind. A folded unit leaves few whole-line requests; an unfolded one leaves
    # narrow ones that spread better over the peel when the rows are cut finer.
    nsplit = geom.store_split if fold else geom.store_split_flat
    split = math.gcd(nsplit, apool[0].tiles)
    nfold = fold or bpool[0].tiles
    npg = nfold // bpool[0].tiles  # B pools one store spans
    a_step = apool[0].tiles * bpool[0].tiles // (split * nfold)
    mfma = Mfma16x16x128(a_step, nfold)
    line_n = not fold and col_safe and nfold % 4 == 0
    store_c = (
        StoreCPerTensorQuadN
        if fold
        else (
            StoreCPerTensorLineN
            if line_n
            else (StoreCPerTensorPairN if pair_n and nfold % 2 == 0 else StoreCPerTensor)
        )
    )(
        A_scale,
        B_scale,
        C,
        fx.Int32(M),
        fx.Int32(N),
        mfma.idx,
        a_step,
        nfold,
        out_ty,
        col_safe=col_safe,
        store_aux=store_aux,
        beta_is_one=beta_is_one,
        **(
            {"lds_xpose": _nt4_xpose_lds(pools[0], pool_lds[0], wave_id, n_waves), "scale": scale}
            if line_n
            else {}
        ),
    )

    g2s = _nt4_g2s(A, B, win, pools, gl_off, wave_id)
    _lds_barrier()
    top = [[p.nbuf - 1] if carry else list(range(p.nbuf)) for p in pools]
    _nt4_prime(pools, pool_lds, g2s, top, K, b_kstep, tr_b)
    wait_barrier(sum(p.steps * sum(1 for b in t if b) for p, t in zip(pools, top)))

    ins = []
    for i, p in enumerate(pools):
        addr = s2r[(p.side, p.width)].base_addr(pool_lds[i])
        ins += [v for pair in (addr if tr_b and p.side == 1 else addr[:1]) for v in pair]
    ins += [
        rocdl.readfirstlane(
            T.i32,
            fx.Int32(fx.ptrtoint(pool_lds[i].ptr))
            + fx.Int32(b * p.buf)
            + fx.Int32(wave_id) * fx.Int32(_TN4_CS),
        )
        for i, p in enumerate(pools)
        for b in range(p.nbuf)
    ]
    for key in _tn4_gl_keys(pools):
        ins += [fx.Int32(o) for o in gl_off[key]]
    ins += [
        _buffer_ops.create_buffer_resource(g, max_size=False, num_records_bytes=nrec, base_byte_offset=base)
        for g, (base, nrec) in ((A, win[0]), (B, win[1]))
    ]
    if carry:
        _win2 = _nt4_tile_window(d_next, *wargs)[2]
        ins += [
            _buffer_ops.create_buffer_resource(
                g, max_size=False, num_records_bytes=nrec, base_byte_offset=base
            )
            for g, (base, nrec) in ((A, _win2[0]), (B, _win2[1]))
        ]
    ins += [fx.Int32(_nt4_buf_off(p, p.nbuf, K, b_kstep, tr_b)) for p in pools]
    ins += [fx.Int32(_nt4_buf_off(p, b, K, b_kstep, tr_b)) for i, p in enumerate(pools) for b in cbuf[i]]
    nacc = sum(p.tiles for p in apool) * sum(p.tiles for p in bpool)  # over the pools it reads

    asm, cons, st, peel = _dense_nt_wave4_asm(geom, K_ITERS, cbsz, blgp, fold, tr_b, b_kstep, carry)
    r = _llvm.inline_asm(ir.Type.parse(st), [arith._to_raw(v) for v in ins], asm, cons, has_side_effects=True)
    acc_ty = ir.Type.parse("vector<4xf32>")
    res = [Vec(_llvm.extractvalue(acc_ty, r, [q])) for q in range_constexpr(nacc)]
    # The K body stops one phase short and hands its mfma over here, where they are ordinary
    # SSA values: the scheduler can then interleave the epilogue with them instead of queueing
    # it behind an opaque asm block.
    frag_ty = ir.Type.parse("vector<8xi32>")
    n_frag = sum(p.tiles for p in apool + bpool)
    frg = [Vec(_llvm.extractvalue(frag_ty, r, [nacc + f])) for f in range_constexpr(n_frag)]
    if store_c.stages_lds:
        _lds_barrier()

    base_row = bm_off + wave_m * fx.Int32(apool[0].tiles * 16)
    qoff, o = {}, 0  # accumulator base of each (A pool, B pool) quadrant, as the asm lays them out
    for ai in range(len(apool)):
        for bi, pb in enumerate(bpool):
            qoff[(ai, bi)] = o
            o += apool[0].tiles * pb.tiles
    unit = []
    for ai, pa in enumerate(apool):
        for gi in range(len(bpool) // npg):
            for h in range(pa.tiles // a_step):
                unit.append(
                    (
                        [
                            qoff[(ai, gi * npg + j)] + (h * a_step + ti) * bpool[0].tiles + bt
                            for ti in range(a_step)
                            for j in range(npg)
                            for bt in range(bpool[0].tiles)
                        ],
                        pa.col + h * a_step * 16,
                        bn_off + wave_n * fx.Int32(nfold * 16) + gi * npg * bpool[0].width,
                    )
                )
    # A unit's mfma all precede the store of the previous unit: that store's reads and
    # converts then issue while this unit's mfma occupy the matrix core. Reordering the
    # phase's mfma by unit is free -- they are independent and their refills already ran.
    grp = [[m for m in peel if m[0] in set(acc)] for acc, _row, _col in unit]

    def emit_store(i, tap=None):
        acc, row, col = unit[i]
        store_c.store([res[q] for q in acc], base_row + row, col, **({"tap": tap} if tap else {}))

    def emit_mfma(m):
        q, at, br = m
        res[q] = asm_mma_do(frg[at], frg[br], res[q], mode="2", cbsz=cbsz, blgp=blgp)

    lead = geom.peel_lead if line_n else 0
    if lead:
        # One mfma per output row under the store keeps the matrix core busy underneath it.
        for i in range_constexpr(lead):
            for m in grp[i]:
                emit_mfma(m)
        for i in range_constexpr(len(unit)):
            ahead = list(grp[i + lead]) if i + lead < len(unit) else []

            def tap(pending=ahead):
                if pending:
                    emit_mfma(pending.pop(0))
                    rocdl.sched_barrier(0)

            emit_store(i, tap=tap)
            for m in ahead:  # a group wider than the slots the unit offered
                emit_mfma(m)
    else:
        for i in range_constexpr(len(unit)):
            for m in grp[i]:
                emit_mfma(m)
            if i:
                rocdl.sched_barrier(0)
                emit_store(i - 1)
        emit_store(len(unit) - 1)
    store_c.flush()


@functools.lru_cache(maxsize=128)
def _compile_dense_wave4(
    M: int,
    N: int,
    K: int,
    group_m: int = 4,
    group_n: int = 0,
    num_xcd: int = 8,
    geom=_NT4_SQUARE,
    cbsz: int = 0,  # srcA fp8 fmt: 0=E4M3, 1=E5M2
    blgp: int = 0,  # srcB fp8 fmt: 0=E4M3, 1=E5M2
    out_fp16: bool = False,
    pair_n: bool = False,  # fold the n-fragment pair into one dword store (needs even N)
    col_safe: bool = False,  # N % BLOCK_N == 0: drop the epilogue's per-store column clamp
    beta_is_one: bool = False,  # epilogue accumulates (C += acc) instead of overwriting
    tr_b: bool = False,  # B is K-major (NN), so its side pays the transpose reader
    ring: bool = False,  # carry the pool fills across the tile boundary
):
    """Whole-loop dense NT (``tr_b=False``) or NN over ``geom``'s macro tile: a resident
    workgroup per CU walks a column of tiles. K must be whole BLOCK_K blocks -- a partial one
    would need the read-ahead to mask, and the 8-wave kernel carries the native K-tail."""
    BM, BN = geom.bm, geom.bn
    NTHR = _tn4_nthr(geom)
    phases = _tn4_phases(geom)
    # The whole-line epilogue interleaves the n-fragments by permuting B's global row order,
    # which a K-major B cannot be given; the fold must also divide the wave's n-extent.
    _bt = sum(p.tiles for p in _tn4_pools(geom) if p.side == geom.gl_side)
    fold = 0 if tr_b else (geom.fold if col_safe and _bt % geom.fold == 0 else 0)
    _pools = _nt4_pools(geom, fold)
    NBM, NBN = ceildiv(M, BM), ceildiv(N, BN)
    n_tile = NBM * NBN
    # Every workgroup runs the tile loop the same number of times, so a ragged count repeats
    # the last id. An overwrite store makes that idempotent; an accumulate would fold it
    # in twice, so beta=1 gives up residency and takes one tile per group.
    tiles_per_wg = 1 if beta_is_one else ceildiv(n_tile, min(n_tile, _dense_num_cus()))
    n_wg = ceildiv(n_tile, tiles_per_wg)
    K_ITERS = K // _TN4_BLOCK_K
    assert K % _TN4_BLOCK_K == 0, "the dense whole loop has no K-tail path"
    assert K_ITERS >= phases, "the dense whole loop needs a K of at least one main-loop pass"
    carry = ring and K_ITERS % phases == 0
    _out_ty = fx.Float16 if out_fp16 else fx.BFloat16

    SharedStorage = fx.struct(
        type(
            "SharedStorage",
            (),
            {
                "__annotations__": {
                    f"p{i}": fx.Array[fx.Float8E4M3FN, p.nbuf * p.buf, 16] for i, p in enumerate(_pools)
                }
            },
        )
    )

    @flyc.kernel(known_block_size=[NTHR, 1, 1])
    def kernel_dense_wave4(
        A: fx.Tensor,
        B: fx.Tensor,
        C: fx.Tensor,
        A_scale: fx.Tensor,
        B_scale: fx.Tensor,
    ):
        _ = str(fx.thread_idx.x)
        lane_id = fx.thread_idx.x % 64
        wave_id = fx.thread_idx.x // 64
        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        # A K-major B is filled row by row and read back transposed, so its write side takes
        # the bank swizzle the transpose reader looks for.
        gl_off = {}
        for k in _tn4_gl_keys(_pools):
            side, width, gq = k
            steps = width * _TN4_BLOCK_K // (NTHR * 16)
            if tr_b and side == 1:
                gl_off[k] = compute_global_swizzle_nn(lane_id, wave_id, N, steps, width=width, wswz=True)
            elif fold and side == geom.gl_side:
                gl_off[k] = _nt4_fold_gl_off(lane_id, wave_id, K, steps, gq, fold)
            else:
                gl_off[k] = compute_global_swizzle(lane_id, wave_id, K, steps, preshuffled=False)
        targs = dict(
            M=M,
            N=N,
            K=K,
            K_ITERS=K_ITERS,
            NBM=NBM,
            NBN=NBN,
            group_m=group_m,
            group_n=group_n,
            num_xcd=num_xcd,
            store_aux=_CSTORE_AUX,
            lds=lds,
            geom=geom,
            A=A,
            B=B,
            C=C,
            A_scale=A_scale,
            B_scale=B_scale,
            gl_off=gl_off,
            wave_id=wave_id,
            **dict(zip(("wave_m", "wave_n"), _tn4_wave_coord(wave_id, geom))),
            cbsz=cbsz,
            blgp=blgp,
            out_ty=_out_ty,
            col_safe=col_safe,
            beta_is_one=beta_is_one,
            pair_n=pair_n,
            fold=fold,
            tr_b=tr_b,
        )

        if const_expr(carry):
            w0 = _nt4_tile_window(fx.block_idx.x, M, N, K, NBM, NBN, group_m, group_n, num_xcd, geom, tr_b)
            _nt4_prime(
                _pools,
                [getattr(lds, f"p{i}") for i in range(len(_pools))],
                _nt4_g2s(A, B, w0[2], _pools, gl_off, wave_id),
                _nt4_carry_bufs(_pools, True),
                K,
                _TN4_BLOCK_K * N if tr_b else _TN4_BLOCK_K,
                tr_b,
            )

        scale = load_per_tensor_scale(A_scale, B_scale) if carry else None

        for t in range(fx.Int32(0), fx.Int32(tiles_per_wg), fx.Int32(1)):
            d = fx.block_idx.x + t * n_wg
            dn = d + n_wg if carry else None
            _dense_nt_wave4_tile(
                arith.select(d < fx.Int32(n_tile), d, fx.Int32(n_tile - 1)),
                d_next=arith.select(dn < fx.Int32(n_tile), dn, fx.Int32(n_tile - 1)) if carry else None,
                scale=scale,
                **targs,
            )

    _ATTRS = make_value_attrs(NTHR // 256, 0, f"{NTHR},{NTHR}")

    @flyc.jit
    def launch_dense_wave4(
        A: fx.Tensor,
        B: fx.Tensor,
        C: fx.Tensor,
        A_scale: fx.Tensor,
        B_scale: fx.Tensor,
        c_m: fx.Int32,
        c_n: fx.Int32,
        stream: fx.Stream,
    ):
        _ = c_m, c_n  # the tile count is compile-time here; the grid is the resident set
        kernel_dense_wave4(A, B, C, A_scale, B_scale, value_attrs=_ATTRS).launch(
            grid=(n_wg, 1, 1), block=(NTHR, 1, 1), stream=stream
        )

    return launch_dense_wave4


_compile_dense_nt_wave4 = functools.partial(_compile_dense_wave4, tr_b=False, ring=False)
_compile_dense_nn_wave4 = functools.partial(_compile_dense_wave4, tr_b=True, ring=True)


_COMPILED_DENSE_CACHE: dict = {}

_raw_stream = torch._C._cuda_getCurrentRawStream


def _static_layout(args):
    """Wrap the tensor arguments as static-layout memrefs for a one-time compile: a bare
    torch.Tensor compiles layout-dynamic and re-reads shape/stride per launch, while the
    compiled object is already one per operand geometry."""
    return tuple(flyc.from_torch_tensor(a) if isinstance(a, torch.Tensor) else a for a in args)


def _compile_scratch_out(launch, args, out_index=2):
    """``compile_with_scratch_out`` through ``_static_layout``: the scratch has to be swapped
    in while the arguments are still tensors, and the layout wrap has to be what reaches
    ``flyc.compile``."""
    scratch = torch.zeros_like(args[out_index])
    return flyc.compile(launch, *_static_layout(args[:out_index] + (scratch,) + args[out_index + 1 :]))


def _get_compiled_dense(launch, args):
    """Cache compiled launcher by (shape, stride, dtype, int-arg) tuple. Strides are in the
    key because the compile pins the operand layout; the trailing queue handle is not, as
    it selects where a launch goes and keying on it would recompile per stream."""
    key_parts = [id(launch)]
    for a in args[:-1]:
        if isinstance(a, torch.Tensor):
            key_parts.append((tuple(a.shape), a.stride(), a.dtype))
        elif isinstance(a, int):
            key_parts.append(a)
        else:
            key_parts.append(type(a).__name__)
    key = tuple(key_parts)
    cached = _COMPILED_DENSE_CACHE.get(key)
    if cached is None:
        cached = _compile_scratch_out(launch, args)
        _COMPILED_DENSE_CACHE[key] = cached
    return cached


def _bench_dense_args(args):
    """Launch args with a scratch C, plus that scratch: racing a beta=1 build against the
    caller buffer would fold every timed launch into it, so the winner is recompiled
    with the accumulate epilogue afterwards instead."""
    bench_c = torch.empty_like(args[2])
    return (args[0], args[1], bench_c) + tuple(args[3:]), bench_c


def _dense_beta1_entry(cache, key, tuned):
    """The beta=1 build of an already-tuned config (same config, accumulate epilogue).

    ``tuned[3]`` is the config's compile factory; the compile functions are
    lru_cached, so this only ever traces once per (shape, config).
    """
    bkey = key + (True,)
    entry = cache.get(bkey)
    if entry is None:
        entry = [tuned[3](beta_is_one=True), tuned[1], None, tuned[3]]
        cache[bkey] = entry
    return entry


def _dense_race(cache, key, args, layout, builders, beta_is_one):
    """First-call race of ``builders`` -- (cfg, factory) pairs -- with the winner cached by
    ``key``. A candidate that fails to build, or whose output sample is not finite, is dropped
    before it is timed, so a geometry that does not hold for the shape just leaves the race."""
    tuned = cache.get(key)
    if tuned is None:
        bargs, out_view = _bench_dense_args(args)
        cands = []
        for cfg, build in builders:
            try:
                launch = build(beta_is_one=False)
                c = _get_compiled_dense(launch, bargs)
                c(*bargs)
                torch.cuda.synchronize()
                if not torch.isfinite(out_view.view(-1)[:1024].float()).all().item():
                    continue
                cands.append([launch, cfg, c, build])  # c: compiled, reused eager
            except Exception:
                continue
        if not cands:
            raise RuntimeError(f"{layout} autotune found no working cfg for {key[:3]}")
        cache[key] = tuned = _pick_dense_candidate(cands, bargs)
    return _dense_beta1_entry(cache, key, tuned) if beta_is_one else tuned


def _pick_dense_candidate(cands, args):
    """Fastest of ``cands`` = [[launch, cfg, compiled, factory], ...], sampled twice with the second
    pass reversed and kept at its min, behind a throwaway pass: the leading candidates sit
    closer than one sample's spread, so otherwise clock drift and warm-up do the ranking."""
    for _ in range(_PICK_RAMP_ITERS):
        cands[0][2](*args)
    torch.cuda.synchronize()
    order = list(range(len(cands)))
    ts = [float("inf")] * len(cands)
    for i in order + order[::-1]:
        ts[i] = min(ts[i], _robust_time(cands[i][2], args, warmup=2, reps=2, iters=40))
    return cands[min(order, key=ts.__getitem__)]


def _run_dense(entry, args):
    """Mode-split steady-state launch. entry = [raw @flyc.jit launch, cfg, compiled].
    Eager: run the one-time flyc.compile'd object (skips @flyc.jit's per-call drift-
    check + arg-hash, and the per-call arg-key rebuild). Capture: run the raw closure
    (a flyc.compile'd object regresses under CUDA-graph capture)."""
    if torch.cuda.is_current_stream_capturing():
        entry[0](*args)
    else:
        if entry[2] is None:
            entry[2] = _compile_scratch_out(entry[0], args)
        entry[2](*args)


def _dense_operand(t: torch.Tensor) -> torch.Tensor:
    return t if t.is_contiguous() else t.contiguous()


def _scalar_scale(scale: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Tensorwise scalar -> length-1 fp32 buffer (no broadcast): the kernel applies the
    single value per-tensor, so only an fp32/device cast is needed. A conforming buffer is
    returned as is, since .to()/.reshape() are no-ops on it but still cost two dispatches."""
    if scale.dtype is torch.float32 and scale.shape == (1,) and scale.device == device:
        return scale
    assert scale.numel() == 1, f"per-tensor expects scalar, got {scale.shape}"
    return scale.to(dtype=torch.float32, device=device).reshape(1)


# (BLOCK_M, GROUP_M, group_n, num_xcd, AGPR), keyed on whether the whole loop is eligible:
# it takes two of the four slots when it is, so a shape never races more than four builds.
_NN_CANDIDATES = {
    True: [(256, 1, 0, 4, 32), (256, 4, 4, 8, 44)],
    False: [(256, 4, 4, 8, 44), (256, 4, 8, 8, 44), (256, 1, 0, 4, 32), (128, 4, 0, 8, 48)],
}
# Past this K the per-tile epilogue is amortised enough to race. The resident set is
# the whole device, so a narrow grid band already spans its columns and only the
# XCD group count is left to spread the rows.
_NN4_MIN_K_ITERS = 16
_NN4_BANDS = [(2, 0, 2), (4, 8, 8)]  # (GROUP_M, group_n, num_xcd) the whole loop races
_NN_AUTOTUNE_CACHE: dict = {}


def _autotune_nn_dispatch(
    args, M, N, K, cbsz=0, blgp=0, out_fp16=False, i64_traverse=False, beta_is_one=False
):
    """NN candidates for the shape, raced on first call (see _dense_race). ``i64_traverse``
    re-bases B's SRD per load, lifting the k*n < 2^32 cap."""
    pair_n, col_safe = N % 2 == 0 and not out_fp16, N % 256 == 0
    w4 = not i64_traverse and K % _TN4_BLOCK_K == 0 and K // _TN4_BLOCK_K >= _NN4_MIN_K_ITERS
    builders = [
        (
            (bm, gm, gn, xcd, ag),
            functools.partial(
                _compile_dense_nn,
                K=K,
                BLOCK_M=bm,
                BLOCK_N=256,
                GROUP_M=gm,
                group_n=gn,
                num_xcd=xcd,
                agpr_alloc=ag,
                b_inline_asm_load=True,
                vmcnt_hint=2,
                cbsz=cbsz,
                blgp=blgp,
                out_fp16=out_fp16,
                i64_traverse=i64_traverse,
                pair_n=pair_n,
                col_safe=col_safe,
            ),
        )
        for bm, gm, gn, xcd, ag in _NN_CANDIDATES[w4]
    ]
    # The whole-loop joins the race where it applies: one barrier per K-block and a wave
    # tile twice as wide, against an epilogue occ=1 cannot shadow. Its B side is one
    # 32-bit soffset chain, so a span needing the i64 re-base stays with the 8-wave tiles.
    if w4:
        builders += [
            (
                ("nn4", gm, gn, xcd),
                functools.partial(
                    _compile_dense_nn_wave4,
                    M,
                    N,
                    K,
                    group_m=gm,
                    group_n=gn,
                    num_xcd=xcd,
                    cbsz=cbsz,
                    blgp=blgp,
                    out_fp16=out_fp16,
                    pair_n=pair_n,
                    col_safe=col_safe,
                ),
            )
            for gm, gn, xcd in _NN4_BANDS
        ]
    key = (M, N, K, cbsz, blgp, out_fp16, i64_traverse)
    return _dense_race(_NN_AUTOTUNE_CACHE, key, args, "NN", builders, beta_is_one)


# (BLOCK_M, GROUP_M, num_xcd, AGPR); the live band is a GROUP_M-wide super-row of A. One
# table per regime, four wide: a race compiles every entry, so the list is what first-call
# latency costs. Once the band outgrows an XCD L2 slice, clustering onto XCDs buys no reuse
# and only costs balance, which is why the wide table drops to a plain round-robin.
_NT_BAND_L2_BYTES = 4 << 20
_NT_CANDIDATES = {
    # (whole-loop eligible, band past the L2 slice)
    (True, False): [(128, 4, 8, 32), (256, 4, 8, 32)],
    (True, True): [(128, 4, 8, 48), (256, 2, 1, 32)],
    (False, False): [(256, 4, 8, 64), (256, 4, 8, 32), (128, 4, 8, 48), (128, 4, 8, 32)],
    (False, True): [(256, 4, 8, 32), (128, 4, 8, 48), (256, 2, 1, 32), (256, 4, 1, 32)],
}
# The whole-loop moves a third less LDS traffic per MFMA, which buys clock on a
# power-limited part, against an epilogue occ=1 cannot shadow; a column-safe tile
# writes it a whole line at a time, which pays off even on the short-K shapes.
_NT4_MIN_K_ITERS = 16
_NT4_BANDS = [(2, 1), (4, 8)]  # (GROUP_M, num_xcd) the whole loop races
_NT_AUTOTUNE_CACHE: dict = {}


def _autotune_nt_dispatch(args, M, N, K, cbsz=0, blgp=0, out_fp16=False, beta_is_one=False):
    """NT candidates for the shape, raced on first call (see _dense_race). The 8-wave tiles are
    joined by the 4-wave whole-loop on the long-K shapes whose steady state pays for it."""
    pair_n, col_safe = N % 2 == 0 and not out_fp16, N % 256 == 0
    w4 = K % _TN4_BLOCK_K == 0 and K // _TN4_BLOCK_K >= _NT4_MIN_K_ITERS
    wide = 4 * 256 * K > _NT_BAND_L2_BYTES  # GROUP_M * BLOCK_M * K of the leading cfg
    builders = [
        (
            (bm, gm, xcd, ag),
            functools.partial(
                _compile_dense_nt,
                K=K,
                BLOCK_M=bm,
                BLOCK_N=256,
                GROUP_M=gm,
                agpr_alloc=ag,
                num_xcd=xcd,
                cbsz=cbsz,
                blgp=blgp,
                out_fp16=out_fp16,
                pair_n=pair_n,
                col_safe=col_safe,
            ),
        )
        for bm, gm, xcd, ag in _NT_CANDIDATES[(w4, wide)]
    ]
    if w4:
        builders += [
            (
                ("nt4", gm, xcd),
                functools.partial(
                    _compile_dense_nt_wave4,
                    M,
                    N,
                    K,
                    group_m=gm,
                    num_xcd=xcd,
                    cbsz=cbsz,
                    blgp=blgp,
                    out_fp16=out_fp16,
                    pair_n=pair_n,
                    col_safe=col_safe,
                ),
            )
            for gm, xcd in _NT4_BANDS
        ]
    key = (M, N, K, cbsz, blgp, out_fp16)
    return _dense_race(_NT_AUTOTUNE_CACHE, key, args, "NT", builders, beta_is_one)


_TN_WAVE4_CACHE: dict = {}
_TN4_WS_CACHE: dict = {}
_TN4_FLAG_CACHE: dict = {}
_TN4_PLAN_CACHE: dict = {}
_TN4_MS_MARGIN = 1.05  # makespan spread inside which the model does not order the macro tiles
_TN4_XCD = 8  # XCDs the dispatcher round-robins over; each fills a private L2 slice


def _tn_wave4_supported(N: int, K: int, i64_traverse: bool) -> bool:
    """The whole-loop reaches each operand through one per-tile buffer SRD and its split-K
    bands keep C's row pitch, so spans needing the per-load i64 re-base, vector-unaligned
    output widths, and a K too short for one main-loop pass all go to the 8-wave kernel."""
    min_iters = min(_tn4_phases(g) for g in _TN4_GEOMS)
    return (not i64_traverse) and N % _TN4_OUT_ALIGN == 0 and ceildiv(K, _TN4_BLOCK_K) >= min_iters


def _tn4_band_group(stripes):
    """Snap a group height to a divisor of _TN4_XCD. Only on a divisor do the workgroups one
    XCD owns within a group share a block_m, which is what keeps its resident set a rectangle
    instead of a staircase spanning every row the group covers."""
    return min((1, 2, 4, 8), key=lambda g: abs(math.log(max(stripes, 0.5) / g)))


def _tn_wave4_band(M, N, geom, ncu):
    """(group_m, group_n) sized so one XCD's resident tiles form the rectangle that pulls the
    fewest operand bytes through its L2 slice. A tall grid, where a band would leave a ragged
    last stripe, spreads the stripes inside the group instead."""
    m_blocks, n_blocks = ceildiv(M, geom.bm), ceildiv(N, geom.bn)
    resident = ncu / _TN4_XCD
    stripes = math.sqrt(resident * geom.bm / geom.bn)
    if n_blocks < m_blocks:
        return _tn4_band_group(stripes * _TN4_XCD / n_blocks), n_blocks
    group_m = _tn4_band_group(stripes * m_blocks / resident)
    return group_m, min(_TN4_XCD // group_m, n_blocks)


def _tn4_geom_split(M, N, K, ncu, geom):
    """This geometry's split-K window, at its own BLOCK_K."""
    return _dense_tn_split(
        ceildiv(M, geom.bm) * ceildiv(N, geom.bn), ceildiv(K, _TN4_BLOCK_K), ncu, _tn4_phases(geom)
    )


def _tn4_makespan(M, N, K, ncu, geom):
    """Device time one geometry needs for this shape, in macro-tile cells: full tiles retire
    ncu at a time and the split-K window's slices, being 1/s of a tile each, share the rounds
    they land in. It only orders the candidates; the race is what picks one."""
    tiles = ceildiv(M, geom.bm) * ceildiv(N, geom.bn)
    cells = geom.bm * geom.bn
    split = _tn4_geom_split(M, N, K, ncu, geom)
    if split is None:
        return ceildiv(tiles, ncu) * cells
    lo, n, s = split
    return (lo // ncu) * cells + _tn4_split_rounds(tiles, n, s, ncu) * cells / s


def _tn4_grid(M, N, K, ncu, geom):
    """Workgroups one geometry launches, plus ``(bands, flags)`` in units of the M x N scratch
    band: the reduce route needs s-1 bands, the handoff one fragment-ordered slot and one flag
    per window tile."""
    tiles = ceildiv(M, geom.bm) * ceildiv(N, geom.bn)
    split = _tn4_geom_split(M, N, K, ncu, geom)
    if split is None:
        return tiles, 0, 0
    grid = tiles + split[1] * (split[2] - 1)
    if _tn4_handoff(split):
        return grid, ceildiv(split[1] * geom.bm * geom.bn, M * N), split[1]
    return grid, split[2] - 1, 0


def _tn4_plan(M, N, K, ncu):
    """Whole-loop macro tiles for one TN shape, most promising first, with the scratch bands
    and flags the widest needs. Ordered by predicted makespan, ties going to the smaller grid.
    Memoised because the steady-state launch path reads it on every call."""
    key = (M, N, K, ncu)
    plan = _TN4_PLAN_CACHE.get(key)
    if plan is None:
        scored = []
        for g in _TN4_GEOMS:
            grid, bands, flags = _tn4_grid(M, N, K, ncu, g)
            scored.append((_tn4_makespan(M, N, K, ncu, g), grid, bands, flags, g))
        lo = min(s[0] for s in scored)
        scored.sort(key=lambda s: (s[0] > lo * _TN4_MS_MARGIN, s[1]))
        plan = (
            tuple(s[4] for s in scored),
            max(s[2] for s in scored),
            max(s[3] for s in scored),
        )
        _TN4_PLAN_CACHE[key] = plan
    return plan


def _tn_wave4_workspace(M, N, bands, device, dtype, out):
    """Scratch for the split-K slice partials: ``bands`` bands of M rows at C's row pitch, so
    a slice store only swaps the band SRD's base. Kept per (shape, device) because a fixed
    buffer is what CUDA-graph capture needs; no window -> pass C and allocate nothing."""
    if bands == 0:
        return out
    key = (device.index, dtype, bands * M, N)
    ws = _TN4_WS_CACHE.get(key)
    if ws is None:
        ws = torch.empty((bands * M, N), device=device, dtype=dtype)
        _TN4_WS_CACHE[key] = ws
    return ws


def _tn_wave4_flags(n, device):
    """Handoff flags, one per split-K window tile and zero at rest: slice 0 raises its tile's
    flag once its partial is device-visible and the folding slice clears it again, so one
    buffer serves every launch. Kept per (size, device) for the same reason the scratch is."""
    key = (device.index, max(n, 1))
    fl = _TN4_FLAG_CACHE.get(key)
    if fl is None:
        fl = torch.zeros(max(n, 1), device=device, dtype=torch.int32)
        _TN4_FLAG_CACHE[key] = fl
    return fl


def _tn_wave4_dispatch(args, M, N, K, geoms, cbsz=0, blgp=0, out_fp16=False):
    """First-call race of the whole-loop macro tiles for one TN problem, best (launch, cfg)
    cached by (M,N,K); each is finite-checked before it is timed. The losers stay in the cache
    entry: _get_compiled_dense keys on the launch object, so a freed id could hit a stale one."""
    key = (M, N, K, cbsz, blgp, out_fp16)
    hit = _TN_WAVE4_CACHE.get(key)
    if hit is not None:
        return hit[0]
    out_view = args[2]
    ncu = _dense_num_cus()
    cands = []
    for geom in geoms:
        try:
            group_m, group_n = _tn_wave4_band(M, N, geom, ncu)
            launch, _bands = _compile_dense_tn_wave4(M, N, K, group_m, group_n, geom, cbsz, blgp, out_fp16)
            c = _get_compiled_dense(launch, args)
            c(*args)
            torch.cuda.synchronize()
            if not torch.isfinite(out_view.view(-1)[:1024].float()).all().item():
                continue
            cands.append([launch, (geom.bm, geom.bn, group_m, group_n), c])
        except Exception:
            continue
    if not cands:
        raise RuntimeError(f"TN whole-loop found no working cfg for ({M},{N},{K})")
    best = _pick_dense_candidate(cands, args)
    _TN_WAVE4_CACHE[key] = (best, cands)
    return best


_TN_AUTOTUNE_CACHE: dict = {}


def _autotune_tn_dispatch(
    args, M, N, K, cbsz=0, blgp=0, out_fp16=False, i64_traverse=False, beta_is_one=False
):
    """First-call bench TN candidates, cache best (launch, cfg) by (M,N,K).

    1D GROUP_M=4 with num_xcd 8 vs 1 (XCD-aware PID remap); large
    (HBM-streaming) shapes expose the per-XCD L2 reuse on the hot bench,
    L2-resident shapes pick num_xcd=1. ``i64_traverse`` re-bases A's and B's
    SRDs per load (lifts the k*m / k*n < 2^32 cap; threaded to _compile_dense_tn).
    """
    bm = 256
    builders = [
        (
            (bm, 4, 0, xcd),
            functools.partial(
                _compile_dense_tn,
                K=K,
                BLOCK_M=bm,
                BLOCK_N=256,
                GROUP_M=4,
                vmcnt_hint=3,
                group_n=0,
                num_xcd=xcd,
                cbsz=cbsz,
                blgp=blgp,
                out_fp16=out_fp16,
                i64_traverse=i64_traverse,
            ),
        )
        for xcd in (8, 1)
    ]
    key = (M, N, K, cbsz, blgp, out_fp16, i64_traverse)
    return _dense_race(_TN_AUTOTUNE_CACHE, key, args, "TN", builders, beta_is_one)


def gemm_fp8_tensorwise_flydsl_kernel(
    a: torch.Tensor,
    a_scale_inv: torch.Tensor,
    b: torch.Tensor,
    b_scale_inv: torch.Tensor,
    trans_a: bool = False,
    trans_b: bool = True,
    out_dtype: torch.dtype = torch.bfloat16,
    trans_c: bool = False,
    beta: float = 0.0,
    out: "torch.Tensor | None" = None,
) -> torch.Tensor:
    """Dense FP8 GEMM, per-tensor scaling. Inputs E4M3/E5M2/hybrid, out bf16/fp16,
    arbitrary K (native K-tail). Dispatch by (trans_a, trans_b): NT (F,T), NN
    (F,F, dgrad), TN (T,F) run native; TT (T,T) unsupported. trans_c=True returns
    out.t().contiguous().

    ``beta=1.0`` accumulates into ``out`` (``out += a @ b``) in the epilogue instead
    of overwriting it, and therefore requires ``out``. It is incompatible with
    ``trans_c``, whose transpose happens after the kernel has already written.
    """
    if out_dtype not in (torch.bfloat16, torch.float16):
        raise NotImplementedError(f"FlyDSL wrapper emits bf16 or fp16. Got {out_dtype}.")
    assert a.dim() == 2 and b.dim() == 2
    beta_is_one = beta == 1.0
    assert not (beta_is_one and trans_c), (
        "beta=1.0 cannot be combined with trans_c: the transpose is a post-kernel copy, "
        "so the accumulation would land in a buffer the caller never sees."
    )
    # Element-count threshold past which a contraction-traversal operand's 32-bit
    # soffset wraps (fp8 = 1 byte/elem). At/above it the kernel re-bases the SRD per
    # load in i64; below it the cheaper fixed-base + 32-bit soffset path is used.
    cap = 2**32
    # Per-operand fp8 format -> MFMA cbsz(srcA)/blgp(srcB): 0=E4M3, 1=E5M2.
    cbsz = 1 if a.dtype == torch.float8_e5m2 else 0
    blgp = 1 if b.dtype == torch.float8_e5m2 else 0
    # fp16 vs bf16 output dtype for StoreCPerTensor (both from the f32 accumulator).
    out_fp16 = out_dtype == torch.float16

    if trans_a and (not trans_b):
        # TN native: A [K, M], B [K, N]. Math C = A^T @ B.
        K_a, M = a.shape
        K_b, N = b.shape
        assert K_a == K_b, f"TN K mismatch: a {a.shape}, b {b.shape}"
        K = K_a
        device = a.device
        a_scale_v = _scalar_scale(a_scale_inv, device)
        b_scale_v = _scalar_scale(b_scale_inv, device)
        out = resolve_accum_out(out, beta, (M, N), device, out_dtype)
        # TN both operands traverse K: span k*m / k*n past 2^32 fp8 needs the
        # per-load i64 SRD re-base (else the 32-bit soffset wraps).
        i64_tr = (K * M >= cap) or (K * N >= cap)
        # The split writes a workspace a later pass reduces, so an accumulate on the tile
        # store would be folded in twice; beta=1 takes the 8-wave epilogue instead.
        if _tn_wave4_supported(N, K, i64_tr) and not beta_is_one:
            geoms, bands, flags = _tn4_plan(M, N, K, _dense_num_cus())
            wargs = (
                _dense_operand(a),
                _dense_operand(b),
                out,
                a_scale_v,
                b_scale_v,
                _tn_wave4_workspace(M, N, bands, device, out_dtype, out),
                _tn_wave4_flags(flags, device),
                _raw_stream(device.index),
            )
            _run_dense(_tn_wave4_dispatch(wargs, M, N, K, geoms, cbsz, blgp, out_fp16), wargs)
        else:
            args = (
                _dense_operand(a),
                _dense_operand(b),
                out,
                a_scale_v,
                b_scale_v,
                M,
                N,
                _raw_stream(device.index),
            )
            _run_dense(_autotune_tn_dispatch(args, M, N, K, cbsz, blgp, out_fp16, i64_tr, beta_is_one), args)
        if trans_c:
            return out.t().contiguous()
        return out

    # Dispatch by layout.
    if (not trans_a) and (not trans_b):
        # NN native: A [M, K], B [K, N].
        M, K_a = a.shape
        K_b, N = b.shape
        assert K_a == K_b, f"NN K mismatch: a {a.shape}, b {b.shape}"
        K = K_a
        device = a.device
        a_scale_v = _scalar_scale(a_scale_inv, device)
        b_scale_v = _scalar_scale(b_scale_inv, device)
        out = resolve_accum_out(out, beta, (M, N), device, out_dtype)
        # NN: per-shape runtime autotune over the candidate tiles, caches by
        # (M,N,K). Build args before autotune (it benches against them).
        args = (
            _dense_operand(a),
            _dense_operand(b),
            out,
            a_scale_v,
            b_scale_v,
            M,
            N,
            _raw_stream(device.index),
        )
        # NN: only B[K,N] traverses K; k*n past 2^32 fp8 needs the i64 re-base.
        i64_tr = K * N >= cap
        _run_dense(_autotune_nn_dispatch(args, M, N, K, cbsz, blgp, out_fp16, i64_tr, beta_is_one), args)
    elif (not trans_a) and trans_b:
        # NT native: A [M, K], B [N, K] (B^T storage of [K, N]).
        M, K_a = a.shape
        N, K_b = b.shape
        assert K_a == K_b, f"NT K mismatch: a {a.shape}, b {b.shape}"
        K = K_a
        device = a.device
        a_scale_v = _scalar_scale(a_scale_inv, device)
        b_scale_v = _scalar_scale(b_scale_inv, device)
        out = resolve_accum_out(out, beta, (M, N), device, out_dtype)
        # NT: per-shape runtime autotune over the 8w/v3 candidate tiles, caches
        # by (M,N,K). Build args before autotune (it benches against them).
        args = (
            _dense_operand(a),
            _dense_operand(b),
            out,
            a_scale_v,
            b_scale_v,
            M,
            N,
            _raw_stream(device.index),
        )
        _run_dense(_autotune_nt_dispatch(args, M, N, K, cbsz, blgp, out_fp16, beta_is_one), args)
    else:
        raise NotImplementedError(
            f"FlyDSL fp8 GEMM does not support the TT layout (trans_a={trans_a}, trans_b={trans_b})."
        )
    if trans_c:
        return out.t().contiguous()
    return out
