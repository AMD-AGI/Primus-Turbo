###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Dense FP8 GEMM persistent-kernel variants (FlyDSL).

Moved out of ``primus_turbo/flydsl/gemm/gemm_fp8_kernel.py`` as a baseline
for future grouped-GEMM persistent-thread dispatch. The NT and NN persistent
kernels here are full-functional 1-tile-per-WG-fanout-PERSIST_FACTOR variants
of the corresponding non-persistent kernels in ``gemm_fp8_kernel.py``.

**Status: not active in production dense path.** The dense dispatch in
``gemm_fp8_tensorwise_flydsl_kernel`` does NOT route through these. The
iter>0 wrong-output bug observed in BIG TASK A7-A10 is post-pass07 / LLVM
AMDGPU backend or HW race, not source-controllable from the FlyDSL DSL
layer. Kept here for grouped-GEMM future work where the persistent skeleton
+ work-stealing across experts gives a cleaner separation of concerns.
"""

import functools
import os

from primus_turbo.flydsl.gemm.gemm_fp8_kernel import _FLYDSL_OK

if _FLYDSL_OK:
    import flydsl.compiler as flyc
    import flydsl.expr as fx
    from flydsl._mlir.dialects import llvm as _llvm
    from flydsl.expr import range_constexpr, rocdl
    from kernels.fp8_gemm_utils import (
        G2SLoader,
        Mfma16x16x128,
        S2RLoader,
        StoreC,
        ceildiv,
        compute_global_swizzle,
        make_fp8_buffer_tensor,
        wait_barrier,
    )

    from primus_turbo.flydsl.gemm.gemm_fp8_kernel import (
        S2RLoaderTr,
        compute_global_swizzle_nn,
    )

    @functools.lru_cache(maxsize=128)
    def _compile_dense_nt_persistent(
        K: int,
        BLOCK_M: int = 256,
        BLOCK_N: int = 256,
        PERSIST_FACTOR: int = 2,
        A11_SAMETILE: bool = False,
    ):
        BLOCK_K = 128
        assert BLOCK_M >= 128 and BLOCK_N >= 256 and BLOCK_M % 128 == 0 and BLOCK_N % 256 == 0
        assert K % BLOCK_K == 0
        assert PERSIST_FACTOR >= 1

        K_ITERS = K // BLOCK_K
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
        def kernel_dense_nt_persistent(
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

            lane_id = fx.thread_idx.x % 64
            wave_id = fx.thread_idx.x // 64
            wave_m = wave_id // 4
            wave_n = wave_id % 4

            gA = make_fp8_buffer_tensor(A, F8_IR_t)
            gB = make_fp8_buffer_tensor(B_T, F8_IR_t)
            a_div = fx.logical_divide(gA, fx.make_layout(1, 1))
            b_div = fx.logical_divide(gB, fx.make_layout(1, 1))

            gl_off_a = compute_global_swizzle(lane_id, wave_id, K, N_LDS_ROUNDS, preshuffled=False)
            gl_off_b = compute_global_swizzle(lane_id, wave_id, K, N_LDS_ROUNDS, preshuffled=False)

            # Fix #2: instantiate ALL helpers per iter — rule out cached SSA
            # values in copy_atom / mma_atom / rmem_tensor objects shared
            # across iters. Fix #1 (StoreC-only per-iter) was insufficient.

            # A10 (A9.1 follow-up): outer persistent tile loop emitted as
            # scf.for (runtime loop, single IR body copy) instead of
            # Python-unroll (PF IR body copies). A7-A9 proved odd-pid tiles
            # carry a 0.14-0.21 abs error in the Python-unroll variant; A8
            # IR dump + A9 source-level fixes (StoreC per-iter, ALL helpers
            # per-iter) all failed to dislodge the bug → bug is post-pass07
            # (LLVM AMDGPU backend regalloc/scheduler across doubled body
            # without scf.for region boundary, or pass07/08 SSA merge on
            # doubled body). scf.for gives LLVM an explicit region; backend
            # cannot reuse VGPRs across iters. Single IR body copy also
            # halves kernel binary size + speeds compile.
            #
            # AST rewriter contract (FlyDSL ast_rewriter.py
            # _transform_for_auto): `for tile_iv in range(N):` (NOT
            # `tuple(range(N))` and NOT `range_constexpr(N)`) triggers
            # scf.for lowering. Body-local variables (mfma, a_g2s, a_cur0,
            # c00_frag, etc) are emitted as scf.for body code (NOT
            # loop-carried iter_args) because they are not assigned before
            # the for-loop. iv `tile_iv` is exposed as Int32 i32 SSA.
            for tile_iv in range(PERSIST_FACTOR):
                mfma = Mfma16x16x128(N_TILES_A, N_TILES_B)
                a_g2s = G2SLoader(a_div, gl_off_a, N_LDS_STEPS_A, F8_IR_t, wave_id)
                b_g2s = G2SLoader(b_div, gl_off_b, N_LDS_STEPS_B, F8_IR_t, wave_id)
                a_s2r = S2RLoader(wave_m, N_TILES_A)
                b_s2r = S2RLoader(wave_n, N_TILES_B)
                store_c = StoreC(A_scale, B_scale, C, c_m, c_n, mfma.idx, N_TILES_A, N_TILES_B)
                # Re-bind LDS slot ptrs each iter (inner main-loop Python ptr
                # swaps don't leak across constexpr-unrolled outer iters).
                a_cur0 = lds.A_lds_cur_0
                a_cur1 = lds.A_lds_cur_1
                a_next0 = lds.A_lds_next_0
                a_next1 = lds.A_lds_next_1
                b_cur0 = lds.B_lds_cur_0
                b_cur1 = lds.B_lds_cur_1
                b_next0 = lds.B_lds_next_0
                b_next1 = lds.B_lds_next_1

                # Each WG owns PF consecutive tiles (sequential-stride);
                # PF=2 host-side gate ensures num_tiles % PF == 0.
                # A11 sametile probe (FLYDSL_RCR_A11_SAMETILE=1): force iter1
                # to process iter0's tile by dropping tile_iv from pid. The
                # scf.for loop body STILL runs PF times (tile_iv is still i32
                # SSA iv, just unused for coord derivation); iter1 overwrites
                # iter0's same output. Discriminates two hypotheses:
                #   (a) iter1 wrong because tile_iv → pid → coords IR alias
                #       → sametile output ≡ PF=0 baseline (✓ bit-exact)
                #   (b) iter1 wrong because WG-state pollution (LDS/VGPR/AGPR
                #       /scoreboard from iter0)
                #       → sametile output STILL has 0.21 abs error
                # Python-level ternary: A11_SAMETILE is closure-const Python
                # bool, evaluated at trace time. Avoids AST rewriter
                # converting `if A11_SAMETILE:` into scf.if (which scopes
                # branch-local vars and breaks subsequent uses of pid).
                tile_iv_for_pid = 0 if A11_SAMETILE else tile_iv
                pid = fx.block_idx.x * PERSIST_FACTOR + tile_iv_for_pid
                block_m = pid // n_blocks
                block_n = pid % n_blocks

                A0_gl_offset = (block_m * BLOCK_M) * K
                A1_gl_offset = (block_m * BLOCK_M + LDS_BLOCK_M) * K
                B0_gl_offset = (block_n * BLOCK_N) * K
                B1_gl_offset = (block_n * BLOCK_N + LDS_BLOCK_N) * K

                c00_frag = [mfma.zero_value] * N_ACCUMS
                c01_frag = [mfma.zero_value] * N_ACCUMS
                c10_frag = [mfma.zero_value] * N_ACCUMS
                c11_frag = [mfma.zero_value] * N_ACCUMS

                # Cross-tile sync. A7-A9 + A10.S2 + A10.H1 (60-cycle s_nop
                # after full drain) all verified: v/lgkm/exp full drain
                # +/- mfma-writeback s_nop padding does NOT fix iter>0
                # wrong-output bug. Bug is robust across Python-unroll AND
                # scf.for runtime-loop variants — root cause is post-pass07
                # (HW or LLVM AMDGPU backend) and not source-level drainable.
                # Drain kept as best-effort sync (mirrors implicit kernel-exit
                # drain of non-persistent baseline). No-op semantically on
                # first iter (clean state).
                _llvm.inline_asm(
                    res=None,
                    operands_=[],
                    asm_string="s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)\ns_barrier",
                    constraints="",
                    has_side_effects=True,
                )

                # Prelude (identical to baseline non-persistent).
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

                # Ensure all prelude buffer_load_lds → LDS WRITES are committed
                # before K-loop's ds_read. wait_barrier() only drains vmcnt;
                # buffer_load_lds LDS-write half is tracked by lgkmcnt and is
                # NOT drained otherwise (baseline kernel-launch wave-init delay
                # hides this, but persistent has no such delay between tiles).
                _llvm.inline_asm(
                    res=None,
                    operands_=[],
                    asm_string="s_waitcnt lgkmcnt(0)",
                    constraints="",
                    has_side_effects=True,
                )

                # Main K-loop (identical to baseline).
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

                    a_cur0, a_next0 = a_next0, a_cur0
                    a_cur1, a_next1 = a_next1, a_cur1
                    b_cur0, b_next0 = b_next0, b_cur0
                    b_cur1, b_next1 = b_next1, b_cur1

                # Epilog 1 (identical to baseline, stale-a1 fix included).
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
        def launch_dense_nt_persistent(
            A: fx.Tensor,
            B_T: fx.Tensor,
            C: fx.Tensor,
            A_scale: fx.Tensor,
            B_scale: fx.Tensor,
            c_m: fx.Int32,
            c_n: fx.Int32,
            stream: fx.Stream,
        ):
            num_tiles = ceildiv(c_m, BLOCK_M) * ceildiv(c_n, BLOCK_N)
            grid_x = ceildiv(num_tiles, PERSIST_FACTOR)
            kernel_dense_nt_persistent(
                A,
                B_T,
                C,
                A_scale,
                B_scale,
                c_m,
                c_n,
                value_attrs={"rocdl.waves_per_eu": 2, "rocdl.flat_work_group_size": "512,512"},
            ).launch(grid=(grid_x, 1, 1), block=(512, 1, 1), stream=stream)

        return launch_dense_nt_persistent

    # ──────────────────────────────────────────────────────────────────────
    # NN-layout dense fp8 kernel (used by dgrad path).
    #
    # A is [M, K] row-major K-contig (same as NT). B is [K, N] row-major --
    # contraction sweeps DOWN B's rows so loading B as K-contig per N-col is
    # strided in gmem (we still use BufferCopyLDS128b N-contig load -> N-contig
    # LDS) and the LDS->register path uses ds_read_b64_tr_b8 which HW-transposes
    # 8 K-rows × 8 N-cols per call so each lane ends up with K-contig fragments
    # in the mfma B-operand format.

    @functools.lru_cache(maxsize=128)
    def _compile_dense_nn_persistent(
        K: int,
        BLOCK_M: int = 256,
        BLOCK_N: int = 256,
        PERSIST_FACTOR: int = 2,
        B7_SAMETILE: bool = False,
    ):
        BLOCK_K = 128
        assert BLOCK_M >= 128 and BLOCK_N >= 256 and BLOCK_M % 128 == 0 and BLOCK_N % 256 == 0
        assert K % BLOCK_K == 0
        assert PERSIST_FACTOR >= 1

        K_ITERS = K // BLOCK_K
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
        def kernel_dense_nn_persistent(
            A: fx.Tensor,
            B: fx.Tensor,
            C: fx.Tensor,
            A_scale: fx.Tensor,
            B_scale: fx.Tensor,
            c_m: fx.Int32,
            c_n: fx.Int32,
        ):
            # See `_compile_dense_nn` docstring: forces thread_idx.x
            # materialization before S2RLoaderTr lazy-eval inside loops.
            _ = str(fx.thread_idx.x)
            F8_IR_t = fx.Float8E4M3FN.ir_type

            n_blocks = ceildiv(c_n, BLOCK_N)

            lds = fx.SharedAllocator().allocate(SharedStorage).peek()

            lane_id = fx.thread_idx.x % 64
            wave_id = fx.thread_idx.x // 64
            wave_m = wave_id // 4
            wave_n = wave_id % 4

            gA = make_fp8_buffer_tensor(A, F8_IR_t)
            gB = make_fp8_buffer_tensor(B, F8_IR_t)
            a_div = fx.logical_divide(gA, fx.make_layout(1, 1))
            b_div = fx.logical_divide(gB, fx.make_layout(1, 1))

            gl_off_a = compute_global_swizzle(lane_id, wave_id, K, N_LDS_ROUNDS, preshuffled=False)
            gl_off_b = compute_global_swizzle_nn(lane_id, wave_id, c_n, N_LDS_ROUNDS)

            # Mirror NT A10/A11 design: scf.for outer loop, all helpers per
            # iter to rule out cached SSA values in copy_atom / mma_atom
            # objects across iters.
            for tile_iv in range(PERSIST_FACTOR):
                mfma = Mfma16x16x128(N_TILES_A, N_TILES_B)
                a_g2s = G2SLoader(a_div, gl_off_a, N_LDS_STEPS_A, F8_IR_t, wave_id)
                b_g2s = G2SLoader(b_div, gl_off_b, N_LDS_STEPS_B, F8_IR_t, wave_id)
                a_s2r = S2RLoader(wave_m, N_TILES_A)
                b_s2r = S2RLoaderTr(wave_n, N_TILES_B)
                store_c = StoreC(A_scale, B_scale, C, c_m, c_n, mfma.idx, N_TILES_A, N_TILES_B)

                a_cur0 = lds.A_lds_cur_0
                a_cur1 = lds.A_lds_cur_1
                a_next0 = lds.A_lds_next_0
                a_next1 = lds.A_lds_next_1
                b_cur0 = lds.B_lds_cur_0
                b_cur1 = lds.B_lds_cur_1
                b_next0 = lds.B_lds_next_0
                b_next1 = lds.B_lds_next_1

                # B7 sametile probe (mirror of A11): drop tile_iv from pid
                # so iter1+ overwrites iter0's tile with same coord. The
                # scf.for body STILL runs PF times. See A11 verdict
                # discriminator in `_compile_dense_nt_persistent`.
                tile_iv_for_pid = 0 if B7_SAMETILE else tile_iv
                pid = fx.block_idx.x * PERSIST_FACTOR + tile_iv_for_pid
                block_m = pid // n_blocks
                block_n = pid % n_blocks

                # A: same as NT (row-major M-major stride).
                A0_gl_offset = (block_m * BLOCK_M) * K
                A1_gl_offset = (block_m * BLOCK_M + LDS_BLOCK_M) * K
                # B: NN-specific column-base offsets.
                B0_gl_offset = block_n * BLOCK_N + 0
                B1_gl_offset = block_n * BLOCK_N + LDS_BLOCK_N

                c00_frag = [mfma.zero_value] * N_ACCUMS
                c01_frag = [mfma.zero_value] * N_ACCUMS
                c10_frag = [mfma.zero_value] * N_ACCUMS
                c11_frag = [mfma.zero_value] * N_ACCUMS

                # Cross-tile sync (best-effort drain; A7-A12 proved source
                # drain does NOT fix the iter>0 race in NT — included here
                # to match NT persistent skeleton 1:1).
                _llvm.inline_asm(
                    res=None,
                    operands_=[],
                    asm_string="s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)\ns_barrier",
                    constraints="",
                    has_side_effects=True,
                )

                # Prelude (identical to NN baseline non-persistent).
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

                # Ensure prelude buffer_load_lds → LDS WRITES committed
                # before K-loop's ds_read (mirror NT persistent fix).
                _llvm.inline_asm(
                    res=None,
                    operands_=[],
                    asm_string="s_waitcnt lgkmcnt(0)",
                    constraints="",
                    has_side_effects=True,
                )

                # Main K-loop (identical to NN baseline).
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
        def launch_dense_nn_persistent(
            A: fx.Tensor,
            B: fx.Tensor,
            C: fx.Tensor,
            A_scale: fx.Tensor,
            B_scale: fx.Tensor,
            c_m: fx.Int32,
            c_n: fx.Int32,
            stream: fx.Stream,
        ):
            num_tiles = ceildiv(c_m, BLOCK_M) * ceildiv(c_n, BLOCK_N)
            grid_x = ceildiv(num_tiles, PERSIST_FACTOR)
            kernel_dense_nn_persistent(
                A,
                B,
                C,
                A_scale,
                B_scale,
                c_m,
                c_n,
                value_attrs={"rocdl.waves_per_eu": 2, "rocdl.flat_work_group_size": "512,512"},
            ).launch(grid=(grid_x, 1, 1), block=(512, 1, 1), stream=stream)

        return launch_dense_nn_persistent

else:

    def _compile_dense_nt_persistent(
        K: int,
        BLOCK_M: int = 256,
        BLOCK_N: int = 256,
        PERSIST_FACTOR: int = 2,
        A11_SAMETILE: bool = False,
    ):
        raise ImportError("FlyDSL is not available -- this entry should be gated by can_handle.")

    def _compile_dense_nn_persistent(
        K: int,
        BLOCK_M: int = 256,
        BLOCK_N: int = 256,
        PERSIST_FACTOR: int = 2,
        B7_SAMETILE: bool = False,
    ):
        raise ImportError("FlyDSL is not available -- this entry should be gated by can_handle.")


# ──────────────────────────────────────────────────────────────────────────────
# Opt-in env knobs. Default 0/False → persistent path inactive.
# ──────────────────────────────────────────────────────────────────────────────
_NT_PERSIST_FACTOR = int(os.environ.get("FLYDSL_RCR_PERSIST_FACTOR", "0"))
_NT_A11_SAMETILE = int(os.environ.get("FLYDSL_RCR_A11_SAMETILE", "0")) == 1
_NN_PERSIST_FACTOR = int(os.environ.get("FLYDSL_NN_PERSIST_FACTOR", "0"))
_NN_B7_SAMETILE = int(os.environ.get("FLYDSL_NN_B7_SAMETILE", "0")) == 1


def _pick_persist_factor(num_tiles: int, group_m: int) -> int:
    if _NT_PERSIST_FACTOR <= 1:
        return 0
    if group_m != 1:
        return 0
    if num_tiles % _NT_PERSIST_FACTOR != 0:
        return 0
    return _NT_PERSIST_FACTOR


def _pick_persist_factor_nn(num_tiles: int, group_m: int) -> int:
    if _NN_PERSIST_FACTOR <= 1:
        return 0
    if group_m != 1:
        return 0
    if num_tiles % _NN_PERSIST_FACTOR != 0:
        return 0
    return _NN_PERSIST_FACTOR
