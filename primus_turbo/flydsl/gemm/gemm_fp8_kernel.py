###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Primus-Turbo dense FP8 GEMM kernel (FlyDSL): NT, NN and TN layouts.
256x256 tile, BLOCK_K=128, 8-wave (wave_m=2 x wave_n=4), mfma_f32_16x16x128_f8f6f4,
per-tensor scale, bf16/fp16 out, arbitrary K via native K-tail (TT unsupported).
Primitives are imported from flydsl.utils.gemm_helper as module globals."""

import functools

import torch

# isort: off
# Primitives are vendored in flydsl/utils/gemm_helper.py (no 3rdparty/FlyDSL
# submodule; flydsl, the compiler, is the only FlyDSL dep) and imported as module
# globals (@flyc.kernel needs its dependencies as globals).
from primus_turbo.flydsl.utils.gemm_helper import (
    G2SLoader,
    Mfma16x16x128,
    S2RLoader,
    S2RLoaderTr,
    StoreCPerTensor,
    asm_mma_do,
    ceildiv,
    compute_global_swizzle,
    compute_global_swizzle_nn,
    make_fp8_buffer_tensor,
    make_value_attrs,
    mask_a_tail,
    wait_barrier,
    xcd_remap_pid,
)
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir.dialects import llvm as _llvm
from flydsl.expr import arith
from flydsl.expr import range_constexpr, rocdl

# isort: on


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
        if cbsz or blgp:
            _ea = fx.Float8E5M2 if cbsz else fx.Float8E4M3FN
            _eb = fx.Float8E5M2 if blgp else fx.Float8E4M3FN
            mfma.atom = fx.make_mma_atom(fx.rocdl.cdna4.MFMA_Scale(16, 16, 128, _ea, _eb))

        a_g2s = G2SLoader(a_div, gl_off_a, N_LDS_STEPS_A, F8_IR_t, wave_id)
        b_g2s = G2SLoader(b_div, gl_off_b, N_LDS_STEPS_B, F8_IR_t, wave_id)
        a_s2r = S2RLoader(wave_m, N_TILES_A)
        b_s2r = S2RLoader(wave_n, N_TILES_B)
        _out_ty = fx.Float16 if out_fp16 else fx.BFloat16
        store_c = StoreCPerTensor(A_scale, B_scale, C, c_m, c_n, mfma.idx, N_TILES_A, N_TILES_B, _out_ty)

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

        # Epilog 2 (k = K_ITERS - 1) -- the K-tail block. Mask the A operand
        # so invalid K-columns (>= K_TAIL) contribute 0. No-op when K_TAIL==0.
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

        a1_frag = a_s2r.load(a_cur1)
        a1_frag = mask_a_tail(a1_frag, lane_id, K_TAIL)
        rocdl.s_barrier()

        rocdl.s_setprio(1)
        c10_frag = mfma.call(a1_frag, b0_frag, c10_frag)
        c11_frag = mfma.call(a1_frag, b1_frag, c11_frag)
        rocdl.s_setprio(0)
        rocdl.s_barrier()

        # Scale + store.
        wave_n_offset = wave_n * (N_TILES_B * 16)
        wave_m_offset = wave_m * (N_TILES_A * 16)
        base_row = block_m * BLOCK_M + wave_m_offset
        base_col = block_n * BLOCK_N + wave_n_offset

        store_c.store(c00_frag, base_row + 0, base_col + 0)
        store_c.store(c01_frag, base_row + 0, base_col + LDS_BLOCK_N)
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
        # Super-block tile swizzle for L2 reuse; group_size_m clamps the last
        # band so any GROUP_M >= 1 is correct (same as NT).
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

        # A: same as NT.
        A0_gl_offset = (block_m * BLOCK_M) * K
        A1_gl_offset = (block_m * BLOCK_M + LDS_BLOCK_M) * K

        # B: NN-specific. B is [K, N] row-major; per WG we load BLOCK_K K-rows
        # × BLOCK_N N-cols, split into 2 N-halves of LDS_BLOCK_N each. K-iter
        # step advances K-rows by BLOCK_K, which in element units is BLOCK_K * c_n.
        B0_gl_offset = block_n * BLOCK_N + 0
        B1_gl_offset = block_n * BLOCK_N + LDS_BLOCK_N

        gA = make_fp8_buffer_tensor(A, F8_IR_t)
        gB = make_fp8_buffer_tensor(B, F8_IR_t)
        a_div = fx.logical_divide(gA, fx.make_layout(1, 1))
        b_div = fx.logical_divide(gB, fx.make_layout(1, 1))

        gl_off_a = compute_global_swizzle(lane_id, wave_id, K, N_LDS_ROUNDS, preshuffled=False)
        gl_off_b = compute_global_swizzle_nn(lane_id, wave_id, c_n, N_LDS_ROUNDS)

        mfma = Mfma16x16x128(N_TILES_A, N_TILES_B)
        if cbsz or blgp:
            # E5M2 / hybrid: rebuild the MFMA atom with per-operand fp8 fmt
            # (cbsz->srcA, blgp->srcB). Same instruction family / frag layout
            # as the default e4m3 atom, so loaders are unchanged.
            _ea = fx.Float8E5M2 if cbsz else fx.Float8E4M3FN
            _eb = fx.Float8E5M2 if blgp else fx.Float8E4M3FN
            mfma.atom = fx.make_mma_atom(fx.rocdl.cdna4.MFMA_Scale(16, 16, 128, _ea, _eb))

        a_g2s = G2SLoader(a_div, gl_off_a, N_LDS_STEPS_A, F8_IR_t, wave_id)
        b_g2s = G2SLoader(b_div, gl_off_b, N_LDS_STEPS_B, F8_IR_t, wave_id)
        a_s2r = S2RLoader(wave_m, N_TILES_A)
        b_s2r = S2RLoaderTr(wave_n, N_TILES_B, 32, inline_asm=b_inline_asm_load, vmcnt_hint=vmcnt_hint)
        _out_ty = fx.Float16 if out_fp16 else fx.BFloat16
        store_c = StoreCPerTensor(A_scale, B_scale, C, c_m, c_n, mfma.idx, N_TILES_A, N_TILES_B, _out_ty)

        c00_frag = [mfma.zero_value] * N_ACCUMS
        c01_frag = [mfma.zero_value] * N_ACCUMS
        c10_frag = [mfma.zero_value] * N_ACCUMS
        c11_frag = [mfma.zero_value] * N_ACCUMS

        # Prelude.
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

        # Main loop. Emits 7 barriers per K-iter (before/after each MFMA);
        # all are load-bearing — dropping any risks a compiler-reorder race.
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
            b_g2s.load(b_cur0, B0_gl_offset + (k + 2) * BLOCK_K * c_n)
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

            b_g2s.load(b_cur1, B1_gl_offset + (k + 2) * BLOCK_K * c_n)
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

        # Epilog 2 -- K-tail block. Mask A so K-cols >= K_TAIL contribute 0.
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

        a1_frag = a_s2r.load(a_cur1)
        a1_frag = mask_a_tail(a1_frag, lane_id, K_TAIL)
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

        # TN A stored [K, M] row-major: stride M per K-row.
        A0_gl_offset = block_m * BLOCK_M + 0
        A1_gl_offset = block_m * BLOCK_M + LDS_BLOCK_M

        # B same as NN: stored [K, N] row-major.
        B0_gl_offset = block_n * BLOCK_N + 0
        B1_gl_offset = block_n * BLOCK_N + LDS_BLOCK_N

        gA = make_fp8_buffer_tensor(A, F8_IR_t)
        gB = make_fp8_buffer_tensor(B, F8_IR_t)
        a_div = fx.logical_divide(gA, fx.make_layout(1, 1))
        b_div = fx.logical_divide(gB, fx.make_layout(1, 1))

        # Both A+B use NN-style K-strided global swizzle.
        gl_off_a = compute_global_swizzle_nn(lane_id, wave_id, c_m, N_LDS_ROUNDS)
        gl_off_b = compute_global_swizzle_nn(lane_id, wave_id, c_n, N_LDS_ROUNDS)

        mfma = Mfma16x16x128(N_TILES_A, N_TILES_B)
        if _inplace:
            _mm = _asm_mma_mode
            mfma._do_mma = lambda _a, _b, _c, _m=_mm: asm_mma_do(_a, _b, _c, mode=_m, cbsz=cbsz, blgp=blgp)

        a_g2s = G2SLoader(a_div, gl_off_a, N_LDS_STEPS_A, F8_IR_t, wave_id, chunk_stride=_LDS_CS)
        b_g2s = G2SLoader(b_div, gl_off_b, N_LDS_STEPS_B, F8_IR_t, wave_id, chunk_stride=_LDS_CS)
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
        store_c = StoreCPerTensor(A_scale, B_scale, C, c_m, c_n, mfma.idx, N_TILES_A, N_TILES_B, _out_ty)

        c00_frag = [mfma.zero_value] * N_ACCUMS
        c01_frag = [mfma.zero_value] * N_ACCUMS
        c10_frag = [mfma.zero_value] * N_ACCUMS
        c11_frag = [mfma.zero_value] * N_ACCUMS

        # Prelude.
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
            a_g2s.load(a_next1, A1_gl_offset + (k + 1) * BLOCK_K * c_m)
            rocdl.s_barrier()
            rocdl.s_setprio(1)
            c00_frag = mfma.call(a0_frag, b0_frag, c00_frag)
            rocdl.s_setprio(0)
            rocdl.s_barrier()
            b1_frag = b_s2r.load(b_cur1)
            b_g2s.load(b_cur0, B0_gl_offset + (k + 2) * BLOCK_K * c_n)
            rocdl.s_barrier()
            rocdl.s_setprio(1)
            c01_frag = mfma.call(a0_frag, b1_frag, c01_frag)
            rocdl.s_setprio(0)
            rocdl.s_barrier()
            a1_frag = a_s2r.load(a_cur1)
            a_g2s.load(a_cur0, A0_gl_offset + (k + 2) * BLOCK_K * c_m)
            rocdl.s_barrier()
            rocdl.s_setprio(1)
            c10_frag = mfma.call(a1_frag, b0_frag, c10_frag)
            rocdl.s_setprio(0)
            rocdl.s_barrier()
            b_g2s.load(b_cur1, B1_gl_offset + (k + 2) * BLOCK_K * c_n)
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
        a_g2s.load(a_next1, A1_gl_offset + (k + 1) * BLOCK_K * c_m)
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


# NN 4-wave kernel removed (consistently slower than 8w; HW caps 1
# wave/SIMD for this layout).


_COMPILED_DENSE_CACHE: dict = {}


def _get_compiled_dense(launch, args):
    """Cache compiled launcher by (shape, dtype, int-arg) tuple."""
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


def _as_i8_flat(t: torch.Tensor) -> torch.Tensor:
    # Zero-copy flat byte view. Recomputed every call (no id()-keyed cache: a
    # freed tensor's id + data_ptr can both be reused, and a recycled pair with a
    # different numel would alias the wrong length). The view ops are ~1us and
    # allocate nothing.
    if t.element_size() == 1 and t.dtype != torch.int8:  # fp8
        return t.contiguous().view(torch.int8).view(-1)
    return t.contiguous().view(-1)


def _scalar_scale(scale: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Tensorwise scalar -> length-1 fp32 buffer (no broadcast). The kernel reads
    the single value and applies it per-tensor, so only an fp32/device cast is
    needed (no per-row/col vector to materialize)."""
    assert scale.numel() == 1, f"per-tensor expects scalar, got {scale.shape}"
    return scale.to(dtype=torch.float32, device=device).reshape(1)


# NN per-shape autotune candidates (BLOCK_M, GROUP_M, num_xcd, AGPR). GROUP_M
# and num_xcd are fixed at the analytic L2 optimum; AGPR must be nonzero (the
# inline-asm ds_read_b64_tr_b8 path needs a pinned AGPR count).
_NN_CANDIDATES = [
    (256, 4, 8, 32),
    (256, 4, 8, 64),
    (128, 4, 8, 48),
]
_NN_AUTOTUNE_CACHE: dict = {}


def _autotune_nn_dispatch(args, M, N, K, cbsz=0, blgp=0, out_fp16=False):
    """First-call bench NN candidates, cache best (launch, cfg) by (M,N,K).

    Runtime micro-benches each (BM, GROUP_M, num_xcd, AG) candidate,
    finite-checks the output, times 2-warmup + 20-iter, and caches the
    fastest by shape.
    """
    import torch as _torch

    key = (M, N, K, cbsz, blgp, out_fp16)
    if key in _NN_AUTOTUNE_CACHE:
        return _NN_AUTOTUNE_CACHE[key]
    out_view = args[2]
    best_us = float("inf")
    best = None
    for bm, gm, xcd, ag in _NN_CANDIDATES:
        # odd-M (M % bm != 0) is fine: the partial last M-tile is
        # bounded by c_m (StoreCPerTensor clamp) and the global SRD (HW OOB
        # clamp on the A G2S load), so no even-tiling filter is needed.
        try:
            # inline-asm ds_read_b64_tr_b8 on by default (drops the per-K-iter
            # compiler-auto vmcnt(0) drains).
            launch = _compile_dense_nn(
                K=K,
                BLOCK_M=bm,
                BLOCK_N=256,
                GROUP_M=gm,
                num_xcd=xcd,
                agpr_alloc=ag,
                b_inline_asm_load=True,
                vmcnt_hint=2,
                cbsz=cbsz,
                blgp=blgp,
                out_fp16=out_fp16,
            )
            c = _get_compiled_dense(launch, args)
            c(*args)
            _torch.cuda.synchronize()
            sample = out_view.view(-1)[:1024].float()
            if not _torch.isfinite(sample).all().item():
                continue
            for _ in range(2):
                c(*args)
            _torch.cuda.synchronize()
            e0 = _torch.cuda.Event(enable_timing=True)
            e1 = _torch.cuda.Event(enable_timing=True)
            _torch.cuda.synchronize()
            e0.record()
            for _ in range(20):
                c(*args)
            e1.record()
            _torch.cuda.synchronize()
            us = e0.elapsed_time(e1) * 1000.0 / 20
            if us < best_us:
                best_us = us
                best = (launch, (bm, gm, xcd, ag))
        except Exception:
            continue
    if best is None:
        raise RuntimeError(f"NN autotune found no working cfg for ({M},{N},{K})")
    _NN_AUTOTUNE_CACHE[key] = best
    return best


# NT per-shape autotune candidates (BLOCK_M, GROUP_M, num_xcd, AGPR). GROUP_M
# and num_xcd are fixed at the analytic L2 optimum; only BLOCK_M and AGPR are
# benched (occupancy/compute effects the hot-cache bench measures reliably).
_NT_CANDIDATES = [
    (256, 4, 8, 64),
    (256, 4, 8, 32),
    (128, 4, 8, 48),
    (128, 4, 8, 32),
]
_NT_AUTOTUNE_CACHE: dict = {}


def _autotune_nt_dispatch(args, M, N, K, cbsz=0, blgp=0, out_fp16=False):
    """First-call bench NT candidates, cache best (launch, cfg) by (M,N,K).

    Runtime micro-benches each (BM, GROUP_M, num_xcd, AG) candidate,
    finite-checks the output, times 2-warmup + 20-iter, and caches the
    fastest by shape.
    """
    import torch as _torch

    key = (M, N, K, cbsz, blgp, out_fp16)
    if key in _NT_AUTOTUNE_CACHE:
        return _NT_AUTOTUNE_CACHE[key]
    out_view = args[2]
    best_us = float("inf")
    best = None
    for bm, gm, xcd, ag in _NT_CANDIDATES:
        # odd-M (M % bm != 0) is fine: the partial last M-tile is
        # bounded by c_m (StoreCPerTensor clamp) and the global SRD (HW OOB
        # clamp on the A G2S load), so no even-tiling filter is needed.
        try:
            launch = _compile_dense_nt(
                K=K,
                BLOCK_M=bm,
                BLOCK_N=256,
                GROUP_M=gm,
                agpr_alloc=ag,
                num_xcd=xcd,
                cbsz=cbsz,
                blgp=blgp,
                out_fp16=out_fp16,
            )
            c = _get_compiled_dense(launch, args)
            c(*args)
            _torch.cuda.synchronize()
            sample = out_view.view(-1)[:1024].float()
            if not _torch.isfinite(sample).all().item():
                continue
            for _ in range(2):
                c(*args)
            _torch.cuda.synchronize()
            e0 = _torch.cuda.Event(enable_timing=True)
            e1 = _torch.cuda.Event(enable_timing=True)
            _torch.cuda.synchronize()
            e0.record()
            for _ in range(20):
                c(*args)
            e1.record()
            _torch.cuda.synchronize()
            us = e0.elapsed_time(e1) * 1000.0 / 20
            if us < best_us:
                best_us = us
                best = (launch, (bm, gm, xcd, ag))
        except Exception:
            continue
    if best is None:
        raise RuntimeError(f"NT autotune found no working cfg for ({M},{N},{K})")
    _NT_AUTOTUNE_CACHE[key] = best
    return best


# TN dispatch: a single inplace-A kernel (inline-asm tr8 on both operands +
# asm_mma=2 → accumulators aliased into AGPR, spill-free, no per-K-iter A-side
# vmcnt(0) drain). Same 1D GROUP_M=4 + XCD-aware PID remap as NT/NN; only the
# num_xcd on/off is benched per shape (L2-resident shapes pick num_xcd=1).


_TN_AUTOTUNE_CACHE: dict = {}


def _autotune_tn_dispatch(args, M, N, K, cbsz=0, blgp=0, out_fp16=False):
    """First-call bench TN candidates, cache best (launch, cfg) by (M,N,K).

    1D GROUP_M=4 with num_xcd 8 vs 1 (XCD-aware PID remap); large
    (HBM-streaming) shapes expose the per-XCD L2 reuse on the hot bench,
    L2-resident shapes pick num_xcd=1.
    """
    import torch as _torch

    key = (M, N, K, cbsz, blgp, out_fp16)
    if key in _TN_AUTOTUNE_CACHE:
        return _TN_AUTOTUNE_CACHE[key]
    # Occupancy routing: BLOCK_M=BLOCK_N=256 yields ceil(M/256)*ceil(N/256)
    # tiles; below NUM_CUS the grid can't fill every CU, so BLOCK_M=128 doubles
    # the M-tile count. Above it the smaller block's per-tile overhead dominates.
    NUM_CUS = 256
    tiles_256 = ((M + 255) // 256) * ((N + 255) // 256)
    bm = 128 if tiles_256 < NUM_CUS else 256
    out_view = args[2]
    best_us = float("inf")
    best = None
    for xcd in (8, 1):
        try:
            launch = _compile_dense_tn(
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
            )
            c = _get_compiled_dense(launch, args)
            c(*args)
            _torch.cuda.synchronize()
            sample = out_view.view(-1)[:1024].float()
            if not _torch.isfinite(sample).all().item():
                continue
            for _ in range(2):
                c(*args)
            _torch.cuda.synchronize()
            e0 = _torch.cuda.Event(enable_timing=True)
            e1 = _torch.cuda.Event(enable_timing=True)
            _torch.cuda.synchronize()
            e0.record()
            for _ in range(20):
                c(*args)
            e1.record()
            _torch.cuda.synchronize()
            us = e0.elapsed_time(e1) * 1000.0 / 20
            if us < best_us:
                best_us = us
                best = (launch, (bm, 4, 0, xcd))
        except Exception:
            continue
    if best is None:
        raise RuntimeError(f"TN autotune found no working cfg for ({M},{N},{K})")
    _TN_AUTOTUNE_CACHE[key] = best
    return best


def gemm_fp8_tensorwise_flydsl_kernel(
    a: torch.Tensor,
    a_scale_inv: torch.Tensor,
    b: torch.Tensor,
    b_scale_inv: torch.Tensor,
    trans_a: bool = False,
    trans_b: bool = True,
    out_dtype: torch.dtype = torch.bfloat16,
    trans_c: bool = False,
) -> torch.Tensor:
    """Dense FP8 GEMM, per-tensor scaling. Inputs E4M3/E5M2/hybrid, out bf16/fp16,
    arbitrary K (native K-tail). Dispatch by (trans_a, trans_b): NT (F,T), NN
    (F,F, dgrad), TN (T,F) run native; TT (T,T) unsupported. trans_c=True returns
    out.t().contiguous()."""
    if out_dtype not in (torch.bfloat16, torch.float16):
        raise NotImplementedError(f"FlyDSL wrapper emits bf16 or fp16. Got {out_dtype}.")
    assert a.dim() == 2 and b.dim() == 2
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
        a_scale_v = _scalar_scale(a_scale_inv, a.device)
        b_scale_v = _scalar_scale(b_scale_inv, a.device)
        out = torch.empty((M, N), dtype=out_dtype, device=a.device)
        # TN: per-shape autotune picks the best candidate cfg, cached by (M,N,K).
        args = (
            _as_i8_flat(a),
            _as_i8_flat(b),
            out.contiguous().view(-1),
            a_scale_v,
            b_scale_v,
            M,
            N,
            torch.cuda.current_stream(),
        )
        launch, _cfg = _autotune_tn_dispatch(args, M, N, K, cbsz, blgp, out_fp16)
        _get_compiled_dense(launch, args)(*args)
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
        a_scale_v = _scalar_scale(a_scale_inv, a.device)
        b_scale_v = _scalar_scale(b_scale_inv, a.device)
        out = torch.empty((M, N), dtype=out_dtype, device=a.device)
        # NN: per-shape runtime autotune over the candidate tiles, caches by
        # (M,N,K). Build args before autotune (it benches against them).
        args = (
            _as_i8_flat(a),
            _as_i8_flat(b),
            out.contiguous().view(-1),
            a_scale_v,
            b_scale_v,
            M,
            N,
            torch.cuda.current_stream(),
        )
        launch, _cfg = _autotune_nn_dispatch(args, M, N, K, cbsz, blgp, out_fp16)
        _get_compiled_dense(launch, args)(*args)
    elif (not trans_a) and trans_b:
        # NT native: A [M, K], B [N, K] (B^T storage of [K, N]).
        M, K_a = a.shape
        N, K_b = b.shape
        assert K_a == K_b, f"NT K mismatch: a {a.shape}, b {b.shape}"
        K = K_a
        a_scale_v = _scalar_scale(a_scale_inv, a.device)
        b_scale_v = _scalar_scale(b_scale_inv, a.device)
        out = torch.empty((M, N), dtype=out_dtype, device=a.device)
        # NT: per-shape runtime autotune over the 8w/v3 candidate tiles, caches
        # by (M,N,K). Build args before autotune (it benches against them).
        args = (
            _as_i8_flat(a),
            _as_i8_flat(b),
            out.contiguous().view(-1),
            a_scale_v,
            b_scale_v,
            M,
            N,
            torch.cuda.current_stream(),
        )
        launch, _cfg = _autotune_nt_dispatch(args, M, N, K, cbsz, blgp, out_fp16)
        _get_compiled_dense(launch, args)(*args)
    else:
        raise NotImplementedError(
            f"FlyDSL fp8 GEMM does not support the TT layout " f"(trans_a={trans_a}, trans_b={trans_b})."
        )
    if trans_c:
        return out.t().contiguous()
    return out
