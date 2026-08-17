###############################################################################
# SPDX-License-Identifier: Apache-2.0
#
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2025 FlyDSL Project Contributors
#
# Adapted from FlyDSL (https://github.com/ROCm/FlyDSL) (kernels/gemm/).
# Modified by the Primus-Turbo team.
#
# This file is distributed under the Apache License 2.0 (see LICENSE-APACHE),
# not the MIT license that covers the rest of Primus-Turbo (see LICENSE).
###############################################################################

"""Primus-Turbo dense BF16 GEMM kernel (FlyDSL)."""

import functools
import math
import os

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl._mlir import ir
from flydsl._mlir.dialects import arith as _std_arith
from flydsl._mlir.dialects import llvm as _llvm
from flydsl.compiler.ast_rewriter import ASTRewriter
from flydsl.expr import arith, const_expr, range_constexpr, rocdl
from flydsl.expr.buffer_ops import (
    _create_i64_constant,
    _unwrap_value,
    create_llvm_ptr,
    get_element_ptr,
)
from flydsl.expr.primitive import get_iter as _get_iter
from flydsl.expr.primitive import ptrtoint as _ptrtoint
from flydsl.expr.typing import Vector as Vec
from flydsl.expr.utils.arith import ArithValue

from primus_turbo.flydsl.utils.gemm_helper import (
    BLOCK_K,
    G2SLoader,
    GatherG2SLoaderBf16,
    GatherVarKG2SLoaderBf16,
    Mfma16x16x32,
    Mfma32x32x16,
    S2RLoaderBf16,
    S2RLoaderTr16x32Bf16,
    S2RLoaderTrBf16,
    StoreCBf16,
    ceildiv,
    compute_global_swizzle_bf16,
    compute_global_swizzle_bf16_rc,
    compute_global_swizzle_nn_bf16,
    compute_global_swizzle_nn_bf16_rc,
    make_bf16_buffer_tensor_rebased,
    make_fp16_bf16_buffer_tensor,
    make_value_attrs,
    wait_barrier,
    xcd_remap_pid,
)

# isort: on

# Epilogue C-store placement inside dense_mma_pipeline_bf16's MFMA tail (see the use
# site for the reasoning and the measurements):
#   0 = batched after the tail (the original), 1 = hoisted into the idle inter-barrier
#   gap BEFORE the next MFMA group, 2 = hoisted into the SHADOW of the next MFMA group
#   (default, measured fastest: 2.3915 / 2.3862 / 2.3705 ms on nt for modes 0/1/2).
# The knob exists only so the arms can be re-run as a same-session A/B; production is 2.
_HOIST_MODE = int(os.environ.get("TURBO_GEMM_HOIST_C", "2"))


def _i64(v):
    # widen an i32 runtime value to i64 (avoids overflow in worst-case base offsets)
    return ArithValue(arith.extsi(fx.T.i64(), _unwrap_value(v)), signed=True)


def _load_i64_as_i32(base, offset):
    # load global i64 at base[offset] and truncate to i32
    ptr = create_llvm_ptr(_unwrap_value(base), 1)  # global address space
    idx = _unwrap_value(offset)
    if isinstance(idx.type, ir.IndexType):
        idx = _unwrap_value(_std_arith.IndexCastOp(fx.T.i64(), idx).result)
    elif isinstance(idx.type, ir.IntegerType) and idx.type.width < 64:
        idx = _unwrap_value(_std_arith.ExtSIOp(fx.T.i64(), idx).result)
    byte_off = _unwrap_value(_std_arith.MulIOp(idx, _create_i64_constant(8)).result)
    elem = get_element_ptr(ptr, byte_offset=byte_off, elem_type=fx.T.i8())
    val = _llvm.LoadOp(fx.T.i64(), elem, ordering=_llvm.AtomicOrdering.monotonic, alignment=8)
    trunc = _std_arith.TruncIOp(fx.T.i32(), val.result)
    return ArithValue(trunc.result, signed=True)


# Staged slot ids per block, in entries. The K loop reads slots from here instead
# of issuing a buffer load per chunk. One direct-to-LDS pass stages SLOT_LDS_PASS
# entries (4 dwords per lane, WGRAD_WAVES waves); only as many passes as the group
# needs are issued, so a small group does not pay for the whole array.
SLOT_LDS_PASS = 8 * 64 * 4
SLOT_LDS_CAP = 2 * SLOT_LDS_PASS


def _make_shared_storage(BLOCK_M, BLOCK_N, slot_lds=False):
    a_lds_size = (BLOCK_M // 2) * BLOCK_K
    b_lds_size = (BLOCK_N // 2) * BLOCK_K

    if slot_lds:

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
            SLOT_lds: fx.Array[fx.Int32, SLOT_LDS_CAP, 16]

        return SharedStorage

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
):
    """Shared 4-quadrant pipelined MMA loop + store epilogue for the fixed-K bf16 tile (NT/NN/TN)."""
    K_ITERS = K // BLOCK_K
    assert K_ITERS >= 2, f"K_ITERS={K_ITERS} too small; need K >= {2 * BLOCK_K}"
    N_TILES_A = BLOCK_M // 128
    N_TILES_B = BLOCK_N // 256
    N_ACCUMS = N_TILES_A * N_TILES_B
    LDS_BLOCK_M = BLOCK_M // 2
    LDS_BLOCK_N = BLOCK_N // 2
    N_LDS_STEPS_A = LDS_BLOCK_M // 64
    N_LDS_STEPS_B = LDS_BLOCK_N // 64

    a_cur0 = lds.A_lds_cur_0
    a_cur1 = lds.A_lds_cur_1
    a_next0 = lds.A_lds_next_0
    a_next1 = lds.A_lds_next_1
    b_cur0 = lds.B_lds_cur_0
    b_cur1 = lds.B_lds_cur_1
    b_next0 = lds.B_lds_next_0
    b_next1 = lds.B_lds_next_1
    a_g2s0, a_g2s1 = a_g2s

    c00_frag = [mfma.zero_value] * N_ACCUMS
    c01_frag = [mfma.zero_value] * N_ACCUMS
    c10_frag = [mfma.zero_value] * N_ACCUMS
    c11_frag = [mfma.zero_value] * N_ACCUMS

    b_g2s.load(b_cur0, B0_gl_offset + 0 * b_k_step)
    a_g2s0.load(a_cur0, A0_gl_offset + 0 * a_k_step)
    b_g2s.load(b_cur1, B1_gl_offset + 0 * b_k_step)
    a_g2s1.load(a_cur1, A1_gl_offset + 0 * a_k_step)

    if wave_m == 1:
        rocdl.s_barrier()
    wait_barrier(N_LDS_STEPS_A + N_LDS_STEPS_B)

    b_g2s.load(b_next0, B0_gl_offset + 1 * b_k_step)
    a_g2s0.load(a_next0, A0_gl_offset + 1 * a_k_step)
    b_g2s.load(b_next1, B1_gl_offset + 1 * b_k_step)

    wait_barrier(N_LDS_STEPS_A + 2 * N_LDS_STEPS_B)

    for k in range_constexpr(K_ITERS - 2):
        b0_frag = b_s2r.load(b_cur0)
        a0_frag = a_s2r.load(a_cur0)
        a_g2s1.load(a_next1, A1_gl_offset + (k + 1) * a_k_step)
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
        a_g2s0.load(a_cur0, A0_gl_offset + (k + 2) * a_k_step)
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

        if const_expr(nt_vmcnt >= 0):
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
    a_g2s1.load(a_next1, A1_gl_offset + (k + 1) * a_k_step)
    rocdl.s_barrier()
    rocdl.s_setprio(1)
    c11_frag = mfma.call(a1_frag, b1_frag, c11_frag)
    rocdl.s_setprio(0)
    rocdl.s_barrier()

    a_cur0, a_next0 = a_next0, a_cur0
    a_cur1, a_next1 = a_next1, a_cur1
    b_cur0, b_next0 = b_next0, b_cur0
    b_cur1, b_next1 = b_next1, b_cur1

    # Address math is needed BEFORE the last MFMA group once the stores are hoisted, and
    # it is loop-invariant scalar/lane arithmetic either way, so sinking it costs nothing.
    wave_n_offset = wave_n * (N_TILES_B * 32)
    wave_m_offset = wave_m * (N_TILES_A * 32)
    base_row = block_m * BLOCK_M + wave_m_offset
    base_col = block_n * BLOCK_N + wave_n_offset

    a0_frag = a_s2r.load(a_cur0)
    wait_barrier(0)
    rocdl.s_setprio(1)
    c00_frag = mfma.call(a0_frag, b0_frag, c00_frag)
    rocdl.s_setprio(0)
    rocdl.s_barrier()

    b1_frag = b_s2r.load(b_cur1)
    # Epilogue store hoist. c00/c01/c10/c11 go final at four different points in this
    # tail, but batching all four stores after it leaves the whole per-workgroup C drain
    # (128 KB, ~4 cycles of address-unit time per instruction) fully exposed after the
    # last MFMA. Issuing each quadrant as soon as ITS last MFMA has retired puts three
    # quarters of that drain in the shadow of the remaining MFMA groups: the matrix pipe
    # and the address unit are separate, so both the store issue and the ack traffic run
    # underneath work that has to happen anyway. Safe by construction -- the only vmcnt
    # wait left in the tail is the wait_barrier(0) ABOVE the first hoist, so nothing
    # downstream force-drains a hoisted store mid-sequence, and the release rendezvous in
    # the caller still waits vmcnt(0) before signalling.
    #
    # Placement matters as much as the hoist. Mode 1 parks a quadrant's stores in the
    # inter-barrier gap BEFORE the next MFMA group, where the wave has nothing else in
    # flight, so only the ack traffic (not the ~4-cycle-per-instruction issue cost) lands
    # in the MFMA shadow. Mode 2 issues them immediately AFTER the group has been handed
    # to the matrix pipe: the MFMAs occupy that pipe for a few hundred cycles while the
    # wave stays free to issue VMEM, so the issue cost overlaps too. Measured on nt:
    # mode 0 2.3915, mode 1 2.3862, mode 2 2.3705 ms -- placement alone is worth 0.66%.
    if const_expr(_HOIST_MODE == 1):
        store_c.store(c00_frag, base_row + 0, base_col + 0)
    rocdl.s_barrier()
    rocdl.s_setprio(1)
    c01_frag = mfma.call(a0_frag, b1_frag, c01_frag)
    rocdl.s_setprio(0)
    # Deliberately OUTSIDE the s_setprio(1) window: mode 2's whole advantage is that the
    # matrix pipe is already loaded when the stores issue, and raising priority here
    # would steal those issue slots back.
    if const_expr(_HOIST_MODE >= 2):
        store_c.store(c00_frag, base_row + 0, base_col + 0)
    rocdl.s_barrier()

    a1_frag = a_s2r.load(a_cur1)
    if const_expr(_HOIST_MODE == 1):
        store_c.store(c01_frag, base_row + 0, base_col + LDS_BLOCK_N)
    rocdl.s_barrier()
    rocdl.s_setprio(1)
    c10_frag = mfma.call(a1_frag, b0_frag, c10_frag)
    c11_frag = mfma.call(a1_frag, b1_frag, c11_frag)
    rocdl.s_setprio(0)
    # Splitting this last MFMA pair across an EXTRA s_barrier -- so c10 goes final a group
    # earlier and gets a full group of store cover too -- is NOT valid: the barrier count
    # through this tail is load-bearing for the surrounding mega-kernel and adding one
    # destroys the result (SNR 51.34 -> 1.21 dB). Stores may move between the existing
    # barriers; the barriers themselves may not be added to or removed.
    if const_expr(_HOIST_MODE >= 2):
        # c01 is independent of the group just issued, so it drains under it; c10 then
        # interlocks on its own MFMA, by which point those stores have covered it.
        store_c.store(c01_frag, base_row + 0, base_col + LDS_BLOCK_N)
        store_c.store(c10_frag, base_row + LDS_BLOCK_M, base_col + 0)
    if const_expr(_HOIST_MODE == 3):
        # Mode 3 = mode 2 plus the last quadrant. c11 is produced by the second MFMA of
        # the pair just issued, so its store cannot dodge that RAW interlock wherever it
        # sits -- but sitting HERE instead of after the barrier hands it the barrier's
        # own resync window (all 8 waves must arrive) as free drain time, and that window
        # is on the critical path to the release rendezvous' vmcnt(0) in the caller. No
        # barrier is added or removed; only the store moves across an existing one.
        # MEASURED AND REJECTED (correct at SNR 51.34, but slower): counterbalanced 2x2,
        # nt 2.3722 -> 2.3870 (+0.62%), nn 3.9766 -> 3.9733 (-0.08%). The barrier window
        # is not free time here -- this wave is the last to arrive as often as not, so
        # what the move actually does is put a stalled store between the MFMA pair and
        # the barrier, delaying every OTHER wave's release. Keep the default at 2.
        store_c.store(c11_frag, base_row + LDS_BLOCK_M, base_col + LDS_BLOCK_N)
    rocdl.s_barrier()

    if const_expr(_HOIST_MODE == 0):
        store_c.store(c00_frag, base_row + 0, base_col + 0)
        store_c.store(c01_frag, base_row + 0, base_col + LDS_BLOCK_N)
    if const_expr(_HOIST_MODE < 2):
        store_c.store(c10_frag, base_row + LDS_BLOCK_M, base_col + 0)
    if const_expr(_HOIST_MODE != 3):
        store_c.store(c11_frag, base_row + LDS_BLOCK_M, base_col + LDS_BLOCK_N)


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
    n_exact=False,
    a_slot_ids=None,
    a_block_m=None,
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

    if a_block_m is None:
        a_block_m = block_m
    if a_slot_ids is None:
        A0_gl_offset = (a_block_m * BLOCK_M) * K
        A1_gl_offset = (a_block_m * BLOCK_M + LDS_BLOCK_M) * K
    else:
        # Gather resolves the row base through slot_ids, so the pipeline's A
        # offset is a pure K column and stays a compile-time constant.
        A0_gl_offset = 0
        A1_gl_offset = 0
    B0_gl_offset = (block_n * BLOCK_N) * K
    B1_gl_offset = (block_n * BLOCK_N + LDS_BLOCK_N) * K
    if b_group_base is not None:
        B0_gl_offset = B0_gl_offset + b_group_base
        B1_gl_offset = B1_gl_offset + b_group_base

    gB = make_fp16_bf16_buffer_tensor(B_T)
    b_div = fx.logical_divide(gB, fx.make_layout(1, 1))
    if a_slot_ids is None:
        gA = make_fp16_bf16_buffer_tensor(A)
        a_div = fx.logical_divide(gA, fx.make_layout(1, 1))

    gl_off_b = compute_global_swizzle_bf16(lane_id, wave_id, K, N_LDS_ROUNDS)

    mfma = Mfma32x32x16(N_TILES_A, N_TILES_B)

    if a_slot_ids is None:
        gl_off_a = compute_global_swizzle_bf16(lane_id, wave_id, K, N_LDS_ROUNDS)
        dense_loader = G2SLoader(a_div, gl_off_a, N_LDS_STEPS_A, fx.BFloat16.ir_type, wave_id)
        a_g2s = (dense_loader, dense_loader)
    else:
        gl_rc_a = compute_global_swizzle_bf16_rc(lane_id, wave_id, N_LDS_ROUNDS)
        a_g2s = (
            GatherG2SLoaderBf16(
                A,
                gl_rc_a,
                N_LDS_STEPS_A,
                wave_id,
                a_slot_ids,
                K,
                a_block_m * fx.Int32(BLOCK_M),
            ),
            GatherG2SLoaderBf16(
                A,
                gl_rc_a,
                N_LDS_STEPS_A,
                wave_id,
                a_slot_ids,
                K,
                a_block_m * fx.Int32(BLOCK_M) + fx.Int32(LDS_BLOCK_M),
            ),
        )
    b_g2s = G2SLoader(b_div, gl_off_b, N_LDS_STEPS_B, fx.BFloat16.ir_type, wave_id)
    a_s2r = S2RLoaderBf16(wave_m, N_TILES_A)
    b_s2r = S2RLoaderBf16(wave_n, N_TILES_B)
    _out_ty = fx.Float16 if out_fp16 else fx.BFloat16
    store_c = StoreCBf16(C, c_m, c_n, _out_ty, cache_modifier=c_cache_modifier, n_exact=n_exact)

    # NT: A is [M,K] row-major, B_T is [N,K] row-major -> both k-steps are BLOCK_K.
    dense_mma_pipeline_bf16(
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
        BLOCK_K,
        BLOCK_K,
        block_m,
        block_n,
        wave_m,
        wave_n,
        K,
        BLOCK_M,
        BLOCK_N,
        nt_vmcnt,
    )


def gemm_bf16_gather_a_tile(*args, a_slot_ids, **kwargs):
    """NT BF16 tile whose logical A rows gather from physical dispatch slots."""
    return gemm_bf16_nt_tile(*args, a_slot_ids=a_slot_ids, **kwargs)


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
    out_fp16=False,
    nt_vmcnt=3,
    b_group_base=None,
    c_cache_modifier=0,
    n_exact=False,
    a_slot_ids=None,
    a_block_m=None,
):
    assert BLOCK_M >= 128 and BLOCK_N >= 256 and BLOCK_M % 128 == 0 and BLOCK_N % 256 == 0
    assert K % BLOCK_K == 0, f"bf16 NN/TN needs K % {BLOCK_K} == 0 (got K={K})"
    # NN's A is [M,K] row-major like NT, so it shares the gather loader. TN's A is
    # transposed, where the gathered index would sit on the contraction axis.
    assert a_slot_ids is None or not a_transpose, "gather-A is only defined for the NN layout here"
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

    if a_block_m is None:
        a_block_m = block_m
    if a_transpose:
        A0_gl_offset = block_m * BLOCK_M + 0
        A1_gl_offset = block_m * BLOCK_M + LDS_BLOCK_M
        a_k_step = BLOCK_K * c_m
    elif a_slot_ids is None:
        A0_gl_offset = (a_block_m * BLOCK_M) * K
        A1_gl_offset = (a_block_m * BLOCK_M + LDS_BLOCK_M) * K
        a_k_step = BLOCK_K
    else:
        # Gather resolves the row base through slot_ids, so the pipeline's A
        # offset is a pure K column and stays a compile-time constant.
        A0_gl_offset = 0
        A1_gl_offset = 0
        a_k_step = BLOCK_K
    B0_gl_offset = block_n * BLOCK_N + 0
    B1_gl_offset = block_n * BLOCK_N + LDS_BLOCK_N
    b_k_step = BLOCK_K * c_n
    if b_group_base is not None:
        B0_gl_offset = B0_gl_offset + b_group_base
        B1_gl_offset = B1_gl_offset + b_group_base

    gB = make_fp16_bf16_buffer_tensor(B)
    b_div = fx.logical_divide(gB, fx.make_layout(1, 1))
    gl_off_b = compute_global_swizzle_nn_bf16(lane_id, wave_id, c_n, N_LDS_STEPS_B)

    mfma = Mfma32x32x16(N_TILES_A, N_TILES_B)
    if a_slot_ids is None:
        gA = make_fp16_bf16_buffer_tensor(A)
        a_div = fx.logical_divide(gA, fx.make_layout(1, 1))
        if a_transpose:
            gl_off_a = compute_global_swizzle_nn_bf16(lane_id, wave_id, c_m, N_LDS_STEPS_A)
        else:
            gl_off_a = compute_global_swizzle_bf16(lane_id, wave_id, K, N_LDS_ROUNDS)
        dense_a_loader = G2SLoader(a_div, gl_off_a, N_LDS_STEPS_A, fx.BFloat16.ir_type, wave_id)
        a_g2s = (dense_a_loader, dense_a_loader)
    else:
        gl_rc_a = compute_global_swizzle_bf16_rc(lane_id, wave_id, N_LDS_ROUNDS)
        a_g2s = (
            GatherG2SLoaderBf16(
                A,
                gl_rc_a,
                N_LDS_STEPS_A,
                wave_id,
                a_slot_ids,
                K,
                a_block_m * fx.Int32(BLOCK_M),
            ),
            GatherG2SLoaderBf16(
                A,
                gl_rc_a,
                N_LDS_STEPS_A,
                wave_id,
                a_slot_ids,
                K,
                a_block_m * fx.Int32(BLOCK_M) + fx.Int32(LDS_BLOCK_M),
            ),
        )
    b_g2s = G2SLoader(b_div, gl_off_b, N_LDS_STEPS_B, fx.BFloat16.ir_type, wave_id)
    a_s2r = S2RLoaderTrBf16(wave_m, N_TILES_A) if a_transpose else S2RLoaderBf16(wave_m, N_TILES_A)
    b_s2r = S2RLoaderTrBf16(wave_n, N_TILES_B)
    _out_ty = fx.Float16 if out_fp16 else fx.BFloat16
    store_c = StoreCBf16(C, c_m, c_n, _out_ty, cache_modifier=c_cache_modifier, n_exact=n_exact)

    dense_mma_pipeline_bf16(
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
    )


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
    n_exact=False,
    a_slot_ids=None,
    a_block_m=None,
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
        a_slot_ids=a_slot_ids,
        a_block_m=a_block_m,
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
        n_exact=n_exact,
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
    nt_vmcnt: int = 3,  # gfx950 G2S LDS hazard: vmcnt>=4 races (nondeterministic); 3 is det
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
            # static-memref JitArgs bake shape into the IR, so shape must be in the key
            shape = getattr(a, "shape", None)
            key_parts.append((type(a).__name__, tuple(shape) if shape is not None else None))
    key = tuple(key_parts)
    cached = _COMPILED_DENSE_CACHE.get(key)
    if cached is None:
        cached = flyc.compile(launch, *args)
        _COMPILED_DENSE_CACHE[key] = cached
    return cached


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
    def kernel_dense_nn_tn(A: fx.Tensor, B: fx.Tensor, C: fx.Tensor, c_m: fx.Int32, c_n: fx.Int32):
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
    def launch_dense_nn_tn(A, B, C, c_m: fx.Int32, c_n: fx.Int32, stream: fx.Stream):
        grid_x = ceildiv(c_m, BLOCK_M) * ceildiv(c_n, BLOCK_N)
        kernel_dense_nn_tn(
            A, B, C, c_m, c_n, value_attrs=make_value_attrs(waves_per_eu, agpr_alloc, "512,512")
        ).launch(grid=(grid_x, 1, 1), block=(512, 1, 1), stream=stream)

    return launch_dense_nn_tn


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


@ASTRewriter.transform
def gemm_bf16_variable_k_tile(
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
    a_slot_ids=None,
    b_slot_ids=None,
    slot_len=None,
    slot_x4=False,
    slot_unroll=1,
    slot_lds=False,
    slot_alu=False,
    slot_u16=False,
):
    CHUNK = 4
    # slot_x4 folds a chunk's slot loads into one dwordx4; without it a wider window
    # still costs one dwordx1 per k, but it is the number of drain points that matters.
    # the staged table is read linearly from LDS, so the interleave buys nothing
    assert not (slot_lds and slot_x4), "slot_lds and slot_x4 are alternatives"
    # slot_lds removes the drain slot_unroll amortizes, and windowing assumes 1-chunk lookahead
    assert not (slot_lds and slot_unroll > 1), "slot_lds implies slot_unroll == 1"
    # probe only: slot_alu ignores the table, so it cannot combine with either scheme
    assert not (slot_alu and (slot_lds or slot_x4)), "slot_alu excludes slot_lds/slot_x4"
    # slot_u16 is the same interleaved table at half width; the others read i32
    assert not (slot_u16 and (slot_x4 or slot_lds or slot_alu)), "slot_u16 excludes the other modes"
    WGRAD_WAVES = 8  # fixed 8 waves per block
    assert BLOCK_M >= 128 and BLOCK_N >= 64 and BLOCK_M % 128 == 0 and BLOCK_N % 64 == 0
    N_TILES_A = BLOCK_M // 128
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

    a0_off = block_m * BLOCK_M
    a1_off = a0_off + LDS_BLOCK_M
    b0_off = block_n * BLOCK_N
    b1_off = b0_off + LDS_BLOCK_N
    a_k_step = fx.Int32(BLOCK_K) * out_m_rt
    b_k_step = fx.Int32(BLOCK_K) * out_n_rt

    NTA16 = N_TILES_A * 2
    NTB16 = (BLOCK_N // 16) // (2 * N_WAVE_N)
    N_ACCUMS16 = NTA16 * NTB16
    mfma = Mfma16x16x32(NTA16, NTB16)
    a_s2r = S2RLoaderTr16x32Bf16(wave_m, NTA16, swz=True)
    b_s2r = S2RLoaderTr16x32Bf16(wave_n, NTB16, swz=True)
    ACC_VEC_N = 4
    N_ACCUMS_EFF = N_ACCUMS16

    a_offs = [a0_off, a1_off]
    b_offs = [b0_off, b1_off]

    def _make_g2s(operand, slot_ids, row_stride, n_steps, base_off, span, cols):
        """Dense rebased loader, or one gather loader serving both LDS halves."""
        if const_expr(slot_ids is None):
            g = make_bf16_buffer_tensor_rebased(operand, bf16_ir, base_off, span)
            gl_off = compute_global_swizzle_nn_bf16(lane_id, wave_id, row_stride, n_steps, swz=True)
            return G2SLoader(fx.logical_divide(g, fx.make_layout(1, 1)), gl_off, n_steps, bf16_ir, wave_id)
        gl_rc = compute_global_swizzle_nn_bf16_rc(lane_id, wave_id, n_steps, WGRAD_WAVES, swz=True)
        x4 = (CHUNK * BLOCK_K, CHUNK) if (slot_x4 or slot_u16) else None
        return GatherVarKG2SLoaderBf16(
            operand,
            gl_rc,
            n_steps,
            wave_id,
            slot_ids,
            slot_len,
            row_stride,
            m_start,
            cols,
            slot_x4=x4,
            slot_u16=slot_u16,
        )

    a_g2s = _make_g2s(A, a_slot_ids, OUT_M, N_LDS_STEPS_A, a_base_off, a_span, a_offs)
    b_g2s = _make_g2s(B, b_slot_ids, OUT_N, N_LDS_STEPS_B, b_base_off, b_span, b_offs)

    # Dense loaders take the fused (column + k * row_stride) offset; gather loaders
    # hold the column base and take voffsets resolved ahead of the MFMA quadrants.
    def _load_a(dst, half, k, voffs):
        if const_expr(a_slot_ids is None):
            a_g2s.load(dst, a_offs[half] + k * a_k_step)
        elif const_expr(slot_u16):
            # the window hands back a thunk: unpack here, not at the window head
            a_g2s.load(dst, half, voffs())
        else:
            a_g2s.load(dst, half, voffs)

    def _load_b(dst, half, k, voffs):
        if const_expr(b_slot_ids is None):
            b_g2s.load(dst, b_offs[half] + k * b_k_step)
        elif const_expr(slot_u16):
            b_g2s.load(dst, half, voffs())
        else:
            b_g2s.load(dst, half, voffs)

    def _voffs(loader, slot_ids, k, wbase=None, wlim=None):
        if const_expr(slot_ids is None):
            return None
        if const_expr(slot_lds):
            return loader.voffsets_lds(lds.SLOT_lds, k * fx.Int32(BLOCK_K) - wbase, wlim)
        if const_expr(slot_alu):
            return loader.voffsets_alu(k * fx.Int32(BLOCK_K))
        return loader.voffsets(k * fx.Int32(BLOCK_K))

    def _voffs_x4(loader, slot_ids, chunk_idx):
        if const_expr(slot_ids is None):
            return None
        return loader.voffsets_x4(chunk_idx)

    def _packs_u16(loader, slot_ids, chunk_idx):
        if const_expr(slot_ids is None):
            return None
        return loader.slot_pack_u16(chunk_idx)

    def _lazy_u16(loader, packs, j):
        """Thunk unpacking k step j at the use site, keeping voffsets short-lived."""
        return lambda: loader.unpack_u16(packs, j)

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

    # A window stages SLOT_LDS_CAP entries but advances by one chunk less, because
    # the K pipeline looks a chunk ahead and must still find those slots staged.
    WIN_CHUNKS = SLOT_LDS_CAP // (CHUNK * BLOCK_K) - 1
    WIN_TOKENS = WIN_CHUNKS * CHUNK * BLOCK_K

    def _win_fills(w):
        """Passes needed for window w: enough to cover it, never more."""
        rem = group_tokens - ArithValue(w) * fx.Int32(WIN_TOKENS)
        n = (rem + fx.Int32(SLOT_LDS_PASS - 1)) // fx.Int32(SLOT_LDS_PASS)
        return ArithValue(
            arith.minsi(arith._to_raw(n), arith._to_raw(fx.Int32(SLOT_LDS_CAP // SLOT_LDS_PASS))),
            signed=True,
        )

    def _win_limit(w):
        return _win_fills(w) * fx.Int32(SLOT_LDS_PASS) - fx.Int32(1)

    if const_expr(slot_lds):
        # Stage before any pool prefetch is in flight: this fill is the tile's only
        # slot-side VMEM traffic, so the drain it costs is paid once, not per chunk.
        slot_g2s = a_g2s if const_expr(a_slot_ids is not None) else b_g2s

        # plain call so the dynamic for does not make slot_g2s/lds loop-carried
        def _fill_pass(base, p):
            slot_g2s.fill_lds(lds.SLOT_lds, base, ArithValue(p), WGRAD_WAVES)

        def _fill_win(w):
            base = ArithValue(w) * fx.Int32(WIN_TOKENS)
            for fill_iv in range(_win_fills(w)):
                _fill_pass(base, fill_iv)
            wait_barrier(0)
            rocdl.s_barrier()

        _fill_win(fx.Int32(0))
        w0_base = fx.Int32(0)
        w0_lim = _win_limit(fx.Int32(0))
    else:
        w0_base = None
        w0_lim = None

    wait_barrier(0)
    av1 = None
    bv1 = None
    if const_expr(slot_x4):
        # one dwordx4 already covers the preamble's k = 0 and k = 1
        aw = _voffs_x4(a_g2s, a_slot_ids, fx.Int32(0))
        bw = _voffs_x4(b_g2s, b_slot_ids, fx.Int32(0))
        av0 = aw[0] if const_expr(a_slot_ids is not None) else None
        bv0 = bw[0] if const_expr(b_slot_ids is not None) else None
        if const_expr(a_slot_ids is not None):
            av1 = aw[1]
        if const_expr(b_slot_ids is not None):
            bv1 = bw[1]
    elif const_expr(slot_u16):
        # one dwordx2 already covers the preamble's k = 0 and k = 1
        ap = _packs_u16(a_g2s, a_slot_ids, fx.Int32(0))
        bp = _packs_u16(b_g2s, b_slot_ids, fx.Int32(0))
        av0 = _lazy_u16(a_g2s, ap, 0) if const_expr(a_slot_ids is not None) else None
        bv0 = _lazy_u16(b_g2s, bp, 0) if const_expr(b_slot_ids is not None) else None
        if const_expr(a_slot_ids is not None):
            av1 = _lazy_u16(a_g2s, ap, 1)
        if const_expr(b_slot_ids is not None):
            bv1 = _lazy_u16(b_g2s, bp, 1)
    else:
        av0 = _voffs(a_g2s, a_slot_ids, fx.Int32(0), w0_base, w0_lim)
        bv0 = _voffs(b_g2s, b_slot_ids, fx.Int32(0), w0_base, w0_lim)
    _load_b(lds.B_lds_cur_0, 0, 0, bv0)
    _load_a(lds.A_lds_cur_0, 0, 0, av0)
    _load_b(lds.B_lds_cur_1, 1, 0, bv0)
    _load_a(lds.A_lds_cur_1, 1, 0, av0)
    if wave_m == 1:
        rocdl.s_barrier()
    wait_barrier(N_LDS_STEPS_A + N_LDS_STEPS_B)
    if const_expr(not slot_x4 and not slot_u16):
        av1 = _voffs(a_g2s, a_slot_ids, fx.Int32(1), w0_base, w0_lim)
        bv1 = _voffs(b_g2s, b_slot_ids, fx.Int32(1), w0_base, w0_lim)
    _load_b(lds.B_lds_next_0, 0, 1, bv1)
    _load_a(lds.A_lds_next_0, 0, 1, av1)
    _load_b(lds.B_lds_next_1, 1, 1, bv1)
    wait_barrier(N_LDS_STEPS_A + 2 * N_LDS_STEPS_B)

    k_iters = (group_tokens + (BLOCK_K - 1)) // BLOCK_K
    n_chunks = (k_iters + (CHUNK - 1)) // CHUNK

    # nested to isolate Python-level buffer rotation from the runtime chunk loop
    def _window(chunk_idx, n_chunk, wbase=None, wlim=None):
        """Slot voffsets for n_chunk consecutive chunks, indexed by k offset.

        Batched here because the body only reads k+1 and k+2: offsets
        1..n_chunk*CHUNK+1 cover the whole span and the VMEM latency is paid once
        per window instead of once per MFMA quadrant. Both halves reuse them.
        """
        av = [None] * (n_chunk * CHUNK + 2)
        bv = [None] * (n_chunk * CHUNK + 2)
        if const_expr(slot_x4):
            # n_chunk+1 dwordx4 span k = c*CHUNK .. c*CHUNK+(n_chunk+1)*CHUNK-1,
            # which contains the window; the last one runs off the table on the
            # final chunk and the SRD clamp turns it into pool row 0, same as the
            # linear path.
            aw = [_voffs_x4(a_g2s, a_slot_ids, chunk_idx + c) for c in range_constexpr(n_chunk + 1)]
            bw = [_voffs_x4(b_g2s, b_slot_ids, chunk_idx + c) for c in range_constexpr(n_chunk + 1)]
            for ko in range_constexpr(1, n_chunk * CHUNK + 2):
                if const_expr(a_slot_ids is not None):
                    av[ko] = aw[ko // CHUNK][ko % CHUNK]
                if const_expr(b_slot_ids is not None):
                    bv[ko] = bw[ko // CHUNK][ko % CHUNK]
        elif const_expr(slot_u16):
            # same span as slot_x4, but each chunk costs CHUNK/2 dwords instead of
            # CHUNK, so a window this wide fits where the i32 one spills
            ap = [_packs_u16(a_g2s, a_slot_ids, chunk_idx + c) for c in range_constexpr(n_chunk + 1)]
            bp = [_packs_u16(b_g2s, b_slot_ids, chunk_idx + c) for c in range_constexpr(n_chunk + 1)]
            for ko in range_constexpr(1, n_chunk * CHUNK + 2):
                if const_expr(a_slot_ids is not None):
                    av[ko] = _lazy_u16(a_g2s, ap[ko // CHUNK], ko % CHUNK)
                if const_expr(b_slot_ids is not None):
                    bv[ko] = _lazy_u16(b_g2s, bp[ko // CHUNK], ko % CHUNK)
        else:
            for ko in range_constexpr(1, n_chunk * CHUNK + 2):
                av[ko] = _voffs(a_g2s, a_slot_ids, chunk_idx * CHUNK + ko, wbase, wlim)
                bv[ko] = _voffs(b_g2s, b_slot_ids, chunk_idx * CHUNK + ko, wbase, wlim)
        return av, bv

    def _steps(k_base, av, bv, off):
        """CHUNK pipelined k steps reading the window at off+j.

        Starts from the LDS buffers in their declared roles and swaps them CHUNK
        (even) times, so the rotation is parity-neutral and no state crosses calls.
        """
        a_cur0, a_cur1 = lds.A_lds_cur_0, lds.A_lds_cur_1
        a_next0, a_next1 = lds.A_lds_next_0, lds.A_lds_next_1
        b_cur0, b_cur1 = lds.B_lds_cur_0, lds.B_lds_cur_1
        b_next0, b_next1 = lds.B_lds_next_0, lds.B_lds_next_1
        for j in range_constexpr(CHUNK):
            k = k_base + j
            jw = off + j
            # 4-buffer pipelined body: interleave s2r/g2s with the 4 mfma quadrants
            b0 = b_s2r.load(b_cur0)
            a0 = a_s2r.load(a_cur0)
            _load_a(a_next1, 1, k + 1, av[jw + 1])
            rocdl.s_barrier()
            rocdl.s_setprio(1)
            c = [Vec(fx.memref_load_vec(r)) for r in acc00]
            c = mfma.call(a0, b0, c)
            for idx in range_constexpr(len(acc00)):
                fx.memref_store_vec(c[idx], acc00[idx])
            rocdl.s_setprio(0)
            rocdl.s_barrier()
            b1 = b_s2r.load(b_cur1)
            _load_b(b_cur0, 0, k + 2, bv[jw + 2])
            rocdl.s_barrier()
            rocdl.s_setprio(1)
            c = [Vec(fx.memref_load_vec(r)) for r in acc01]
            c = mfma.call(a0, b1, c)
            for idx in range_constexpr(len(acc01)):
                fx.memref_store_vec(c[idx], acc01[idx])
            rocdl.s_setprio(0)
            rocdl.s_barrier()
            a1 = a_s2r.load(a_cur1)
            _load_a(a_cur0, 0, k + 2, av[jw + 2])
            rocdl.s_barrier()
            rocdl.s_setprio(1)
            c = [Vec(fx.memref_load_vec(r)) for r in acc10]
            c = mfma.call(a1, b0, c)
            for idx in range_constexpr(len(acc10)):
                fx.memref_store_vec(c[idx], acc10[idx])
            rocdl.s_setprio(0)
            rocdl.s_barrier()
            _load_b(b_cur1, 1, k + 2, bv[jw + 2])
            wait_barrier(2 * N_LDS_STEPS_A + N_LDS_STEPS_B)
            rocdl.s_setprio(1)
            c = [Vec(fx.memref_load_vec(r)) for r in acc11]
            c = mfma.call(a1, b1, c)
            for idx in range_constexpr(len(acc11)):
                fx.memref_store_vec(c[idx], acc11[idx])
            rocdl.s_setprio(0)
            rocdl.s_barrier()
            a_cur0, a_next0 = a_next0, a_cur0
            a_cur1, a_next1 = a_next1, a_cur1
            b_cur0, b_next0 = b_next0, b_cur0
            b_cur1, b_next1 = b_next1, b_cur1

    def _chunk(chunk_iv):
        chunk_idx = ArithValue(chunk_iv)
        av, bv = _window(chunk_idx, 1)
        _steps(chunk_idx * CHUNK, av, bv, 0)

    def _chunk_w(chunk_iv, win_iv):
        """One chunk resolving slots from the LDS window that holds them."""
        chunk_idx = ArithValue(chunk_iv)
        wbase = ArithValue(win_iv) * fx.Int32(WIN_TOKENS)
        av, bv = _window(chunk_idx, 1, wbase, _win_limit(win_iv))
        _steps(chunk_idx * CHUNK, av, bv, 0)

    def _win_lo(w):
        return ArithValue(w) * fx.Int32(WIN_CHUNKS)

    def _win_hi(w):
        hi = (ArithValue(w) + fx.Int32(1)) * fx.Int32(WIN_CHUNKS)
        return ArithValue(arith.minsi(arith._to_raw(hi), arith._to_raw(n_chunks)), signed=True)

    def _n_win(nc):
        return (ArithValue(nc) + fx.Int32(WIN_CHUNKS - 1)) // fx.Int32(WIN_CHUNKS)

    def _chunk_n(chunk_iv):
        # One window per slot_unroll chunks. vmcnt retires in order, so a slot load
        # drains the previous chunk's outstanding pool prefetches; sharing a window
        # divides the number of those drains by slot_unroll.
        chunk_idx = ArithValue(chunk_iv)
        av, bv = _window(chunk_idx, slot_unroll)
        for c in range_constexpr(slot_unroll):
            _steps((chunk_idx + fx.Int32(c)) * CHUNK, av, bv, c * CHUNK)

    if const_expr(slot_lds):
        # Window 0 is peeled: its fill has to precede the preamble, which already
        # resolves k = 0 and k = 1. Later windows refill the same LDS array in place.
        for chunk_iv in range(_win_hi(fx.Int32(0))):
            _chunk_w(chunk_iv, fx.Int32(0))
        for win_iv in range(fx.Int32(1), _n_win(n_chunks)):
            _fill_win(win_iv)
            for chunk_iv in range(_win_lo(win_iv), _win_hi(win_iv)):
                _chunk_w(chunk_iv, win_iv)
    elif const_expr(slot_unroll > 1):
        n_grouped = (n_chunks // slot_unroll) * slot_unroll
        for chunk_iv in range(0, n_grouped, slot_unroll):
            _chunk_n(chunk_iv)
        for chunk_iv in range(n_grouped, n_chunks):
            _chunk(chunk_iv)
    else:
        for chunk_iv in range(n_chunks):
            _chunk(chunk_iv)

    c00 = [Vec(fx.memref_load_vec(reg)) for reg in acc00]
    c01 = [Vec(fx.memref_load_vec(reg)) for reg in acc01]
    c10 = [Vec(fx.memref_load_vec(reg)) for reg in acc10]
    c11 = [Vec(fx.memref_load_vec(reg)) for reg in acc11]

    # Static facts the transposed epilogue needs to pack its stores: every q_row
    # term is a multiple of 16, and the N range is tiled exactly so the per-lane
    # column mask is statically true.
    _trans_m_align = math.gcd(math.gcd(int(BLOCK_M), int(LDS_BLOCK_M)), 16)
    _trans_n_exact = int(OUT_N) % int(BLOCK_N) == 0

    def _emit_q(cfrag, q_row, q_col):
        for i in range_constexpr(NTA16):
            for j in range_constexpr(NTB16):
                blk = [cfrag[i * NTB16 + j]]
                if const_expr(trans_c):
                    store_c.store_trans16(
                        blk,
                        group_idx,
                        q_row + i * 16,
                        q_col + j * 16,
                        OUT_M,
                        OUT_N,
                        m_align=_trans_m_align,
                        n_exact=_trans_n_exact,
                    )
                else:
                    store_c.store16(blk, q_row + i * 16, q_col + j * 16)

    if const_expr(trans_c):
        local_m = block_m * BLOCK_M + wave_m * (NTA16 * 16)
        local_n = block_n * BLOCK_N + wave_n * (NTB16 * 16)
        _emit_q(c00, local_m + 0, local_n + 0)
        _emit_q(c01, local_m + 0, local_n + LDS_BLOCK_N)
        _emit_q(c10, local_m + LDS_BLOCK_M, local_n + 0)
        _emit_q(c11, local_m + LDS_BLOCK_M, local_n + LDS_BLOCK_N)
    else:
        base_row = group_idx * OUT_M + block_m * BLOCK_M + wave_m * (NTA16 * 16)
        base_col = block_n * BLOCK_N + wave_n * (NTB16 * 16)
        _emit_q(c00, base_row + 0, base_col + 0)
        _emit_q(c01, base_row + 0, base_col + LDS_BLOCK_N)
        _emit_q(c10, base_row + LDS_BLOCK_M, base_col + 0)
        _emit_q(c11, base_row + LDS_BLOCK_M, base_col + LDS_BLOCK_N)


@functools.lru_cache(maxsize=64)
def _compile_grouped_variable_k_bf16(
    OUT_M,
    OUT_N,
    G,
    BLOCK_M=256,
    BLOCK_N=256,
    num_xcd=8,
    waves_per_eu=2,
    agpr_alloc=0,
    out_fp16=False,
    trans_c=False,
    gather_a=False,
    gather_b=False,
    slot_x4=False,
    slot_unroll=1,
    slot_lds=False,
    slot_alu=False,
    slot_u16=False,
):
    assert OUT_M % BLOCK_M == 0, "OUT_M (unclamped store dim) must divide BLOCK_M"
    N_BLOCKS_M = OUT_M // BLOCK_M
    N_BLOCKS_N = (OUT_N + BLOCK_N - 1) // BLOCK_N
    TILES_PER_GROUP = N_BLOCKS_M * N_BLOCKS_N
    TOTAL = G * TILES_PER_GROUP
    SharedStorage = _make_shared_storage(BLOCK_M, BLOCK_N, slot_lds=slot_lds)

    @flyc.kernel(known_block_size=[512, 1, 1])
    def kernel_grouped_variable_k(
        A: fx.Tensor,
        B: fx.Tensor,
        C: fx.Tensor,
        group_k_offsets: fx.Tensor,
        masked_k: fx.Tensor,
        A_SLOT_IDS: fx.Tensor,
        slot_len: fx.Int32,
        out_m_rt: fx.Int32,
        out_n_rt: fx.Int32,
    ):
        _ = str(fx.thread_idx.x)
        go_base = fx.Int64(_ptrtoint(_get_iter(group_k_offsets)))
        gk_base = fx.Int64(_ptrtoint(_get_iter(masked_k)))
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
            m_start = _load_i64_as_i32(go_base, group_idx)
            # bound K to valid rows; padding tail never read
            m_end = m_start + _load_i64_as_i32(gk_base, group_idx)
            gemm_bf16_variable_k_tile(
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
                a_slot_ids=A_SLOT_IDS if const_expr(gather_a) else None,
                b_slot_ids=A_SLOT_IDS if const_expr(gather_b) else None,
                slot_len=slot_len,
                slot_x4=slot_x4,
                slot_unroll=slot_unroll,
                slot_lds=slot_lds,
                slot_alu=slot_alu,
                slot_u16=slot_u16,
            )

        _do_tile(pid)

    @flyc.jit
    def launch_grouped_variable_k(
        A,
        B,
        C,
        group_k_offsets,
        masked_k,
        a_slot_ids,
        slot_len: fx.Int32,
        out_m_rt: fx.Int32,
        out_n_rt: fx.Int32,
        stream: fx.Stream,
    ):
        grid_x = fx.Int32(TOTAL)
        kernel_grouped_variable_k(
            A,
            B,
            C,
            group_k_offsets,
            masked_k,
            a_slot_ids,
            slot_len,
            out_m_rt,
            out_n_rt,
            value_attrs=make_value_attrs(waves_per_eu, agpr_alloc, "512,512"),
        ).launch(grid=(grid_x, 1, 1), block=(512, 1, 1), stream=stream)

    return launch_grouped_variable_k


_COMPILED_GROUPED_GEMM_CACHE = {}


def _ptr_only_view(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().view(torch.int32)


def grouped_gemm_variable_k_bf16(
    a: torch.Tensor,
    b: torch.Tensor,
    group_k_offsets: torch.Tensor,
    masked_k: torch.Tensor = None,
    out_dtype: torch.dtype = torch.bfloat16,
    BLOCK_M: int = 256,
    BLOCK_N: int = 256,
    num_xcd: int = 8,
    trans_c: bool = False,
    a_slot_ids: torch.Tensor = None,
    b_slot_ids: torch.Tensor = None,
    slot_x4: bool = False,
    slot_unroll: int = 1,
    slot_lds: bool = False,
    slot_alu: bool = False,
    slot_u16: bool = False,
) -> torch.Tensor:
    """Variable-K grouped wgrad: out[g]=a[g_rows].T@b[g_rows], K=[offsets[g],offsets[g]+masked_k[g]).

    ``a_slot_ids`` makes the A rows indirect: row r reads a[a_slot_ids[r]]. Used
    when the dispatch pool is deduplicated, so a[] holds unique slots while the
    K axis still walks logical route rows."""
    assert a.dim() == 2 and b.dim() == 2
    assert a_slot_ids is None or b_slot_ids is None, "only one operand may be gathered"
    assert a_slot_ids is not None or b_slot_ids is not None or a.shape[0] == b.shape[0]
    assert a.dtype == torch.bfloat16 and b.dtype == torch.bfloat16
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
    launch = _compile_grouped_variable_k_bf16(
        OUT_M,
        OUT_N,
        G,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_xcd=num_xcd,
        out_fp16=out_fp16,
        trans_c=trans_c,
        gather_a=a_slot_ids is not None,
        gather_b=b_slot_ids is not None,
        slot_x4=slot_x4,
        slot_unroll=slot_unroll,
        slot_lds=slot_lds,
        slot_alu=slot_alu,
        slot_u16=slot_u16,
    )
    # static memref: create_buffer_resource needs a real memref, not a raw ptr arg
    slots = a_slot_ids if a_slot_ids is not None else b_slot_ids
    slot_src = slots.contiguous() if slots is not None else masked_k_i64.view(torch.int32)
    slot_arg = flyc.from_torch_tensor(slot_src)
    args = (
        _ptr_only_view(a),
        _ptr_only_view(b),
        flyc.from_torch_tensor(out),
        offsets_i64,
        masked_k_i64,
        slot_arg,
        slot_src.numel(),
        OUT_M,
        OUT_N,
        torch.cuda.current_stream(),
    )
    key = (
        OUT_M,
        OUT_N,
        G,
        BLOCK_M,
        BLOCK_N,
        out_fp16,
        trans_c,
        a_slot_ids is not None,
        b_slot_ids is not None,
        slot_x4,
        slot_unroll,
        slot_lds,
        slot_alu,
        slot_u16,
    )
    compiled = _COMPILED_GROUPED_GEMM_CACHE.get(key)
    if compiled is None:
        compiled = flyc.compile(launch, *args)
        _COMPILED_GROUPED_GEMM_CACHE[key] = compiled
    compiled(*args)
    return out


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
