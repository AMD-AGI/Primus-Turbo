###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Primus-Turbo dense FP8 GEMM kernel (FlyDSL).

This file owns the kernel definition end-to-end; it does NOT delegate to
`kernels.fp8_gemm_8wave` in the FlyDSL repo. The lower-level FlyDSL helpers
(G2SLoader, S2RLoader, StoreC, Mfma16x16x128, swizzle math) ARE imported from
`kernels.fp8_gemm_utils` since those are reusable primitives, not the kernel
orchestration itself.

`@flyc.kernel` decorated functions must reference their dependencies as MODULE
globals (not as closure cells from an enclosing factory), so we import FlyDSL
at module load time. If FlyDSL is missing, `_FLYDSL_OK = False` and the
backend's `can_handle` (which calls `flydsl_available()`) returns False -- the
kernel factory + wrapper become unusable stubs.

Algorithm: 8-wave (512 thread WG, wave_m=2 × wave_n=4) NT-layout fp8 GEMM,
BLOCK_M=BLOCK_N=256, BLOCK_K=128, LDS_BLOCK_M=LDS_BLOCK_N=128, 2×2 mma ping-pong
(c00/c01/c10/c11 per wave), `mfma_f32_16x16x128_f8f6f4`. Includes the c10/c11
"stale a_cur1" pipeline fix in epilog 1 so epilog-2's a1_frag reads K_ITERS-1
data instead of older K-iter data.

Constraints:
  - K % 128 == 0     (BLOCK_K=128, K_ITERS compile-time constant, K_ITERS >= 2)
  - out_dtype == torch.bfloat16  (StoreC fixed)
  - per-tensor scale  (a_scale / b_scale scalar fp32, broadcast inside wrapper)
  - All four (trans_a, trans_b) combos supported; trans_c via post-hoc transpose
"""

import functools
import os
import sys

import torch


# ──────────────────────────────────────────────────────────────────────────────
# Module-load-time FlyDSL discovery. We try a direct import first; if FlyDSL is
# installed but its repo root isn't on sys.path (the `kernels.*` package isn't
# pip-installed), infer the root from `flydsl.__file__` and inject it.
# ──────────────────────────────────────────────────────────────────────────────

_FLYDSL_OK = False
try:
    try:
        from kernels.fp8_gemm_utils import (  # noqa: F401  (re-exported names below)
            G2SLoader,
            Mfma16x16x128,
            S2RLoader,
            StoreC,
            ceildiv,
            compute_global_swizzle,
            divmod,
            make_fp8_buffer_tensor,
            pack_i32x4_i32x8,
            swizzle_128,
            wait_barrier,
        )
    except ImportError:
        import flydsl as _flydsl_probe

        _flydsl_root = os.path.abspath(
            os.path.join(os.path.dirname(_flydsl_probe.__file__), "..", "..")
        )
        if _flydsl_root not in sys.path:
            sys.path.insert(0, _flydsl_root)
        from kernels.fp8_gemm_utils import (  # noqa: E402,F401
            G2SLoader,
            Mfma16x16x128,
            S2RLoader,
            StoreC,
            ceildiv,
            compute_global_swizzle,
            divmod,
            make_fp8_buffer_tensor,
            pack_i32x4_i32x8,
            swizzle_128,
            wait_barrier,
        )

    import flydsl.compiler as flyc
    import flydsl.expr as fx
    from flydsl._mlir import ir
    from flydsl._mlir.dialects import llvm as _llvm
    from flydsl.compiler.kernel_function import CompilationContext as _CompilationContext
    from flydsl.expr import buffer_ops as _buffer_ops, const_expr, range_constexpr, rocdl
    from flydsl.expr.utils.arith import ArithValue
    from flydsl.expr.arith import _to_raw as _raw
    from flydsl.expr.typing import T, Vector as Vec

    _FLYDSL_OK = True
except ImportError:
    pass


def flydsl_available() -> bool:
    """Reported via backend can_handle to gate dispatch."""
    return _FLYDSL_OK


# ──────────────────────────────────────────────────────────────────────────────
# Our own dense fp8 NT kernel factory. The @flyc.kernel decorated function
# references the FlyDSL utils as module globals (above) so its co_freevars is
# empty / matches what FlyDSL's AST rewriter expects.
# ──────────────────────────────────────────────────────────────────────────────

if _FLYDSL_OK:

    @functools.lru_cache(maxsize=128)
    def _compile_dense_nt(K: int, BLOCK_M: int = 256, BLOCK_N: int = 256, GROUP_M: int = 1):
        """Build & cache the (K, BLOCK_M, BLOCK_N, GROUP_M)-specialised launch function.

        ``GROUP_M`` controls tile-id swizzle for L2 reuse:
          - ``GROUP_M == 1`` (default): row-major (block_m, block_n) scan = legacy.
          - ``GROUP_M > 1``: Triton-style super-block grouping; WG-ids advance
            ``block_m`` first within a ``GROUP_M × n_blocks`` super-block before
            stepping ``block_n``, so consecutive WGs reuse B-columns in L2.
        """
        BLOCK_K = 128
        assert BLOCK_M >= 128 and BLOCK_N >= 256 and BLOCK_M % 128 == 0 and BLOCK_N % 256 == 0
        assert K % BLOCK_K == 0
        assert GROUP_M >= 1

        K_ITERS = K // BLOCK_K
        assert K_ITERS >= 2, f"K_ITERS={K_ITERS} too small; need K >= 256"

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
            # Triton-style super-block tile-id swizzle. GROUP_M=1 collapses to
            # row-major (block_m = pid // n_blocks, block_n = pid % n_blocks).
            # GROUP_M>1: consecutive WGs share block_n, reusing B-columns in L2.
            pid = fx.block_idx.x
            num_pid_in_group = GROUP_M * n_blocks
            group_id = pid // num_pid_in_group
            pid_in_group = pid % num_pid_in_group
            pid_m_inner = pid_in_group % GROUP_M
            pid_n = pid_in_group // GROUP_M
            block_m = group_id * GROUP_M + pid_m_inner
            block_n = pid_n

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
            store_c = StoreC(A_scale, B_scale, C, c_m, c_n, mfma.idx, N_TILES_A, N_TILES_B)

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

            # Epilog 2 (k = K_ITERS - 1).
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
                value_attrs={"rocdl.waves_per_eu": 2, "rocdl.flat_work_group_size": "512,512"},
            ).launch(grid=(grid_x, 1, 1), block=(512, 1, 1), stream=stream)

        return launch_dense_nt

    # ──────────────────────────────────────────────────────────────────────
    # Persistent NT variant (Session A7 prototype, A6.1 follow-on).
    #
    # A6 PMC diagnosis (RCR_A6_DIAGNOSTIC.md) identified kernel-launch / wave-init
    # overhead as the dominant remaining gap vs hipblaslt: hbl runs ~constant
    # ~215K wave/launch (persistent + work-steal), fly runs (num_tiles * 8) wave
    # /launch (1-WG-per-tile non-persistent), 32x ratio on qwen_down_B16_M4096
    # drives 1.30x kernel-only gap. Source/scheduler/swizzle/tile-autotune
    # levers (A2-A5) all 0 effect; persistent restructure is the only remaining
    # lever per A6 PMC chain.
    #
    # MVP design (this session):
    #   - constexpr PERSIST_FACTOR (PF) wraps full per-tile body in
    #     range_constexpr(PF) — IR is unrolled PF copies, no scf.for state
    #     plumbing. Trades code-size growth (PF*) for WG-launch reduction (PF*).
    #   - Launcher: grid_x = ceildiv(num_tiles, PF). Each WG owns PF
    #     consecutive tiles: pid = block_idx.x * PF + tile_iv.
    #   - Requires num_tiles % PF == 0 (host-side gate; falls back to
    #     non-persistent variant otherwise).
    #   - Per-iter: re-bind LDS ptrs (Python swaps in prior iter), reset acc
    #     fragments, full prelude → main K-loop → epilog 1 → epilog 2 → store.
    #   - Cross-iter sync: rocdl.s_barrier() before next iter's prelude to
    #     ensure prior tile's outstanding LDS state is settled before the
    #     same LDS slots are rewritten.
    # ──────────────────────────────────────────────────────────────────────

    @functools.lru_cache(maxsize=128)
    def _compile_dense_nt_persistent(
        K: int, BLOCK_M: int = 256, BLOCK_N: int = 256, PERSIST_FACTOR: int = 2,
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
    # ──────────────────────────────────────────────────────────────────────

    _LDS_PTR_TYPE = None

    def _inttoptr_lds(byte_addr):
        """Convert an integer byte address to !llvm.ptr<3> (LDS pointer).

        Same shape as the mla kernel helper -- ir.Type.parse'd once, then
        llvm.inttoptr to lift an i64 value into the LDS address space.
        """
        global _LDS_PTR_TYPE
        if _LDS_PTR_TYPE is None:
            _LDS_PTR_TYPE = ir.Type.parse("!llvm.ptr<3>")
        return _llvm.inttoptr(_LDS_PTR_TYPE, _raw(fx.Int64(byte_addr)))

    _gep = _buffer_ops.get_element_ptr

    def _lds_ptr_from_i32(addr_i32, byte_offset=0):
        """Build an LDS pointer (ptr<3>) from an i32 byte address + optional static offset."""
        ptr = _inttoptr_lds(ArithValue(addr_i32).extui(T.i64))
        if byte_offset != 0:
            ptr = _gep(ptr, static_byte_offset=byte_offset)
        return ptr

    def compute_global_swizzle_nn(lane_id, wave_id, N_out, n_rounds):
        """B in NN: [K_inner, N_out] row-major. Each round loads 64 K-rows ×
        128 N-bytes (per wave 8 K-rows × 128 N-bytes via BufferCopyLDS128b).
        Per-round per-lane offset = swizzle_128(k_row, n_col_byte_base) over
        the [K, N] flat byte view with N_out element stride.
        """
        offsets = []
        n_waves = fx.block_dim.x // 64
        for r in range_constexpr(n_rounds):
            k_row = lane_id // 8 + wave_id * 8 + r * (n_waves * 8)
            n_col = (lane_id % 8) * 16
            rs, cs = swizzle_128(k_row, n_col)
            offsets.append(rs * N_out + cs)
        return offsets

    class S2RLoaderTr:
        """LDS -> mfma B-operand transpose-load for NN-layout B (Session 3,
        probe-verified 2026-05-25, see scripts2/probe_transpose_nn_lds.py and
        scripts2/NN_TRANSPOSE_DESIGN.md).

        LDS layout (from G2SLoader + BufferCopyLDS128b + swizzle_128, written
        into a [LDS_BLOCK_N=128, BLOCK_K=128]-sized region) is NOT the natural
        ``LDS[K*128 + N]`` row-major layout assumed by an earlier version of
        this class. The actual per-wave 1024-byte block layout is:
            LDS[W*1024 + step*8192 + L*16 + n_byte]
              = B[K=L//8 + W*8 + step*64, N=(L%8)*16 XOR swz_K + n_byte]
        where W ∈ [0,8) is the writing wave_id, step ∈ {0,1} the LDS load
        round, L ∈ [0,64) the lane within that wave, n_byte ∈ [0,16) the byte
        index within the 16-byte BufferCopyLDS128b chunk, and
            swz_K = ((K % 16) // 2) * 16        (swizzle_128 XOR factor).

        We issue 4 ds_read_tr8_b64 per (lane, mfma_tile) with per-lane ptrs
        derived to match the mfma B operand layout (Session 1 ground truth:
        for B operand of mfma_f32_16x16x128_f8f6f4, lane L wants byte b ∈
        [0,32) to be B[K = (L//16)*16 + (b<16 ? b : 64+b-16),
        N = wave_n*32 + tile_i*16 + L%16]).

        Per-lane ptr formula (I = L//16, L_in_sg = L%16, c ∈ {0,1,2,3},
        K_BASE = [0, 8, 64, 72]):
            K_log = I*16 + K_BASE[c] + L_in_sg//2          ∈ [0, 128)
            r_step = K_log // 64                            ∈ {0, 1}
            W      = (K_log % 64) // 8                     ∈ [0, 8)
            K_mod_8 = K_log % 8                            ∈ [0, 8)
            swz_K  = ((K_log % 16) // 2) * 16              ∈ {0,16,...,112}
            tile_N_start = wave_n*32 + tile_i*16
            j_chunk = (tile_N_start // 16) ^ (swz_K // 16) # un-swizzle N chunk
            ptr = LDS + W*1024 + r_step*8192 + K_mod_8*128
                      + j_chunk*16 + (L_in_sg%2)*8
        """

        _K_BASE = (0, 8, 64, 72)
        TR_TYPE = None  # set lazily inside load()

        def __init__(self, wave_n, n_tiles_b):
            self.wave_n = wave_n
            self.n_tiles_b = n_tiles_b
            self.lane_id = fx.thread_idx.x % 64

        def load(self, lds_src, preshuffled=False):
            """Returns list[N_TILES_B] of i32x8 (32 fp8/lane K-contig at N-col)."""
            assert not preshuffled, "S2RLoaderTr does not support preshuffled"
            if S2RLoaderTr.TR_TYPE is None:
                S2RLoaderTr.TR_TYPE = Vec.make_type(2, fx.Int32)
            tr_type = S2RLoaderTr.TR_TYPE

            base_i32 = fx.Int32(fx.ptrtoint(lds_src.ptr))
            I = self.lane_id // 16
            L_in_sg = self.lane_id % 16

            frag = []
            for tile_i in range_constexpr(self.n_tiles_b):
                calls = []
                for c in range_constexpr(4):
                    K_log = I * 16 + S2RLoaderTr._K_BASE[c] + (L_in_sg // 2)
                    r_step = K_log // 64
                    W = (K_log % 64) // 8
                    K_mod_8 = K_log % 8
                    swz_K = ((K_log % 16) // 2) * 16
                    tile_N_start = self.wave_n * 32 + tile_i * 16
                    j_chunk = (tile_N_start // 16) ^ (swz_K // 16)
                    ptr_offset = (
                        W * 1024
                        + r_step * 8192
                        + K_mod_8 * 128
                        + j_chunk * 16
                        + (L_in_sg % 2) * 8
                    )
                    ptr_i32 = base_i32 + fx.Int32(ptr_offset)
                    ptr = _lds_ptr_from_i32(ptr_i32)
                    v = rocdl.ds_read_tr8_b64(tr_type, ptr).result
                    calls.append(Vec(v))

                # Concat 4 × v2i32 → v8i32. Byte order matches mfma B-operand
                # bytes 0..31 for lane L:
                #   v8i32[0..1] = call 0 → K = (L//16)*16 + L%2*4 + ...   (bytes 0..7)
                #   v8i32[2..3] = call 1 → bytes 8..15
                #   v8i32[4..5] = call 2 → bytes 16..23 (K + 64 jump)
                #   v8i32[6..7] = call 3 → bytes 24..31
                v4_lo = calls[0].shuffle(calls[1], [0, 1, 2, 3])
                v4_hi = calls[2].shuffle(calls[3], [0, 1, 2, 3])
                frag.append(v4_lo.shuffle(v4_hi, list(range(8))))
            return frag

    class S2RLoaderTrA:
        """LDS -> mfma A-operand transpose-load. Mirrors `S2RLoaderTr` but the
        per-wave LDS column stride is parameterized as `n_tiles_a * 16` (not
        the hardcoded 32 of `S2RLoaderTr`).  The hardcoded `wave_n * 32`
        happened to equal `wave_n * (n_tiles_b * 16)` for the NN B case where
        `n_tiles_b == 2`; for CRR A with `n_tiles_a == 4` (BLOCK_M=256,
        wave_m ∈ [0,2)) the correct stride is 64 so the 2 wave_m's tile the
        LDS_BLOCK_M=128 K_out cols without overlap or gaps.

        Lane / byte convention identical to S2RLoaderTr (mfma A and B operands
        share per-lane data format for mfma_f32_16x16x128_f8f6f4).
        """

        _K_BASE = (0, 8, 64, 72)
        TR_TYPE = None

        def __init__(self, wave_idx, n_tiles_a):
            self.wave_idx = wave_idx
            self.n_tiles_a = n_tiles_a
            self.lane_id = fx.thread_idx.x % 64

        def load(self, lds_src, preshuffled=False):
            assert not preshuffled, "S2RLoaderTrA does not support preshuffled"
            if S2RLoaderTrA.TR_TYPE is None:
                S2RLoaderTrA.TR_TYPE = Vec.make_type(2, fx.Int32)
            tr_type = S2RLoaderTrA.TR_TYPE

            base_i32 = fx.Int32(fx.ptrtoint(lds_src.ptr))
            I = self.lane_id // 16
            L_in_sg = self.lane_id % 16
            wave_col_stride = self.n_tiles_a * 16

            frag = []
            for tile_i in range_constexpr(self.n_tiles_a):
                calls = []
                for c in range_constexpr(4):
                    K_log = I * 16 + S2RLoaderTrA._K_BASE[c] + (L_in_sg // 2)
                    r_step = K_log // 64
                    W = (K_log % 64) // 8
                    K_mod_8 = K_log % 8
                    swz_K = ((K_log % 16) // 2) * 16
                    tile_col_start = self.wave_idx * wave_col_stride + tile_i * 16
                    j_chunk = (tile_col_start // 16) ^ (swz_K // 16)
                    ptr_offset = (
                        W * 1024
                        + r_step * 8192
                        + K_mod_8 * 128
                        + j_chunk * 16
                        + (L_in_sg % 2) * 8
                    )
                    ptr_i32 = base_i32 + fx.Int32(ptr_offset)
                    ptr = _lds_ptr_from_i32(ptr_i32)
                    v = rocdl.ds_read_tr8_b64(tr_type, ptr).result
                    calls.append(Vec(v))

                v4_lo = calls[0].shuffle(calls[1], [0, 1, 2, 3])
                v4_hi = calls[2].shuffle(calls[3], [0, 1, 2, 3])
                frag.append(v4_lo.shuffle(v4_hi, list(range(8))))
            return frag

    class S2RLoaderTr32:
        """LDS -> mfma B-operand transpose-load for NN-layout B with
        mfma_f32_32x32x64. Companion of `S2RLoaderTr` (16x16x128).

        Derivation: see scripts2/CRR_C13_S2RLOADER_TR32_DESIGN.md
        (A-operand variant probe-verified 2026-05-26 on chi2762,
        scripts2/probe_transpose_crr32_a_lds.py 0/16384 mismatch).

        Key differences from 16x16x128 S2RLoaderTr:
          - `_K_BASE = (0, 8, 32, 40)`  (vs 16x16 (0, 8, 64, 72) — mfma_K
            halved 128 -> 64).
          - 4 HW transpose sub-groups partition (m_or_n, k_grp) instead of all
            reading same 16-col range:
              I = lane // 16; k_grp = I // 2; mh = I % 2
              n_in_tile = mh * 16 + l_lane     (= lane % 32)
              if byte < 16: k_log = k_grp * 16 + byte
              else:         k_log = k_grp * 16 + 32 + (byte - 16)
          - Per-wave tile spans 32 N_out cols (vs 16 in S2RLoaderTr).
          - Per (lane, call) ptr:
              I       = lane // 16; l_lane = lane % 16
              k_grp   = I // 2; mh = I % 2; i = l_lane // 2
              K_log   = k_grp*16 + _K_BASE[c] + i           ∈ [0, 64)
              r_step  = K_log // 64                          (=0 always)
              W       = (K_log % 64) // 8
              K_mod_8 = K_log % 8
              swz_K   = ((K_log % 16) // 2) * 16
              tile_N_start = wave_n*32 + tile_i*32 + mh*16
              j_chunk = (tile_N_start // 16) ^ (swz_K // 16)
              ptr = LDS + W*1024 + r_step*8192 + K_mod_8*128
                        + j_chunk*16 + (l_lane%2)*8
        """

        _K_BASE = (0, 8, 32, 40)
        TR_TYPE = None

        def __init__(self, wave_n, n_tiles_b):
            self.wave_n = wave_n
            self.n_tiles_b = n_tiles_b
            self.lane_id = fx.thread_idx.x % 64

        def load(self, lds_src, preshuffled=False):
            assert not preshuffled, "S2RLoaderTr32 does not support preshuffled"
            if S2RLoaderTr32.TR_TYPE is None:
                S2RLoaderTr32.TR_TYPE = Vec.make_type(2, fx.Int32)
            tr_type = S2RLoaderTr32.TR_TYPE

            base_i32 = fx.Int32(fx.ptrtoint(lds_src.ptr))
            I = self.lane_id // 16
            l_lane = self.lane_id % 16
            k_grp = I // 2
            mh = I % 2

            frag = []
            for tile_i in range_constexpr(self.n_tiles_b):
                calls = []
                for c in range_constexpr(4):
                    K_log = k_grp * 16 + S2RLoaderTr32._K_BASE[c] + (l_lane // 2)
                    r_step = K_log // 64
                    W = (K_log % 64) // 8
                    K_mod_8 = K_log % 8
                    swz_K = ((K_log % 16) // 2) * 16
                    tile_N_start = self.wave_n * 32 + tile_i * 32 + mh * 16
                    j_chunk = (tile_N_start // 16) ^ (swz_K // 16)
                    ptr_offset = (
                        W * 1024
                        + r_step * 8192
                        + K_mod_8 * 128
                        + j_chunk * 16
                        + (l_lane % 2) * 8
                    )
                    ptr_i32 = base_i32 + fx.Int32(ptr_offset)
                    ptr = _lds_ptr_from_i32(ptr_i32)
                    v = rocdl.ds_read_tr8_b64(tr_type, ptr).result
                    calls.append(Vec(v))

                v4_lo = calls[0].shuffle(calls[1], [0, 1, 2, 3])
                v4_hi = calls[2].shuffle(calls[3], [0, 1, 2, 3])
                frag.append(v4_lo.shuffle(v4_hi, list(range(8))))
            return frag

    class S2RLoaderTrA32:
        """LDS -> mfma A-operand transpose-load for CRR-layout A with
        mfma_f32_32x32x64. Mirrors `S2RLoaderTr32` but per-wave LDS col
        stride parameterized as `n_tiles_a * 32` (vs `S2RLoaderTr32`'s
        hardcoded `wave_n * 32`).

        Probe-verified 2026-05-26 on chi2762
        (scripts2/probe_transpose_crr32_a_lds.py, 0/16384 mismatch).
        See scripts2/CRR_C13_S2RLOADER_TR32_DESIGN.md §4 for full derivation.
        """

        _K_BASE = (0, 8, 32, 40)
        TR_TYPE = None

        def __init__(self, wave_idx, n_tiles_a):
            self.wave_idx = wave_idx
            self.n_tiles_a = n_tiles_a
            self.lane_id = fx.thread_idx.x % 64

        def load(self, lds_src, preshuffled=False):
            assert not preshuffled, "S2RLoaderTrA32 does not support preshuffled"
            if S2RLoaderTrA32.TR_TYPE is None:
                S2RLoaderTrA32.TR_TYPE = Vec.make_type(2, fx.Int32)
            tr_type = S2RLoaderTrA32.TR_TYPE

            base_i32 = fx.Int32(fx.ptrtoint(lds_src.ptr))
            I = self.lane_id // 16
            l_lane = self.lane_id % 16
            k_grp = I // 2
            mh = I % 2
            wave_col_stride = self.n_tiles_a * 32

            frag = []
            for tile_i in range_constexpr(self.n_tiles_a):
                calls = []
                for c in range_constexpr(4):
                    K_log = k_grp * 16 + S2RLoaderTrA32._K_BASE[c] + (l_lane // 2)
                    r_step = K_log // 64
                    W = (K_log % 64) // 8
                    K_mod_8 = K_log % 8
                    swz_K = ((K_log % 16) // 2) * 16
                    tile_col_start = self.wave_idx * wave_col_stride + tile_i * 32 + mh * 16
                    j_chunk = (tile_col_start // 16) ^ (swz_K // 16)
                    ptr_offset = (
                        W * 1024
                        + r_step * 8192
                        + K_mod_8 * 128
                        + j_chunk * 16
                        + (l_lane % 2) * 8
                    )
                    ptr_i32 = base_i32 + fx.Int32(ptr_offset)
                    ptr = _lds_ptr_from_i32(ptr_i32)
                    v = rocdl.ds_read_tr8_b64(tr_type, ptr).result
                    calls.append(Vec(v))

                v4_lo = calls[0].shuffle(calls[1], [0, 1, 2, 3])
                v4_hi = calls[2].shuffle(calls[3], [0, 1, 2, 3])
                frag.append(v4_lo.shuffle(v4_hi, list(range(8))))
            return frag

    @functools.lru_cache(maxsize=128)
    def _compile_dense_nn(
        K: int,
        BLOCK_M: int = 256,
        BLOCK_N: int = 256,
        WAVES_PER_EU: int = 2,
        SETPRIO_HIGH: int = 1,
        SCHED_HINT: int = 0,
        GROUP_M: int = 1,
        MAXNREG: int = 0,
    ):
        """NN-layout fp8 dense kernel. A [M, K], B [K, N], C [M, N].

        Phase B2 knobs (default reproduces B1 baseline behavior):
          WAVES_PER_EU  - rocdl.waves_per_eu attribute (1/2/4)
          SETPRIO_HIGH  - main-loop mfma s_setprio raise value; 0 = no setprio
                          (NN setprio is load-bearing per
                          [[flydsl-nn-setprio-load-bearing]], 0 expected to
                          regress; sweep included for completeness)
          SCHED_HINT    - 0 = off (default); 1 = sched_group_barrier(mfma) anchor
                          after each mfma in main loop; 2 = mfma+vmem+dsrd tight
                          group (mirror A4 NT T3 mask set)

        Phase B3 knob:
          GROUP_M       - Triton-style super-block tile-id swizzle for L2 reuse.
                          GROUP_M=1 (default) = row-major scan = B1 baseline.
                          GROUP_M>1: consecutive WGs share block_n, reusing the
                          B[K, block_n_cols] strip in L2. Mirror of NT GROUP_M.
        """
        BLOCK_K = 128
        assert BLOCK_M >= 128 and BLOCK_N >= 256 and BLOCK_M % 128 == 0 and BLOCK_N % 256 == 0
        assert K % BLOCK_K == 0
        assert GROUP_M >= 1

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
            # Workaround: forces materialization of thread_idx.x in trace IR
            # before downstream S2RLoaderTr lazy-evaluates it inside
            # range_constexpr loops. Without this, flydsl emits an
            # out-of-order load schedule for ds_read_tr8_b64 that yields
            # garbage output (C-SNR ≈ -2.4 dB). Probe-verified ptr formula is
            # correct; bug is in flydsl's trace order, not our kernel.
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
            # B3: Triton-style super-block swizzle. GROUP_M=1 collapses to
            # row-major (matches divmod(pid, n_blocks)).
            pid = fx.block_idx.x
            num_pid_in_group = GROUP_M * n_blocks
            group_id = pid // num_pid_in_group
            pid_in_group = pid % num_pid_in_group
            pid_m_inner = pid_in_group % GROUP_M
            pid_n = pid_in_group // GROUP_M
            block_m = group_id * GROUP_M + pid_m_inner
            block_n = pid_n

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

            a_g2s = G2SLoader(a_div, gl_off_a, N_LDS_STEPS_A, F8_IR_t, wave_id)
            b_g2s = G2SLoader(b_div, gl_off_b, N_LDS_STEPS_B, F8_IR_t, wave_id)
            a_s2r = S2RLoader(wave_m, N_TILES_A)
            b_s2r = S2RLoaderTr(wave_n, N_TILES_B)
            store_c = StoreC(A_scale, B_scale, C, c_m, c_n, mfma.idx, N_TILES_A, N_TILES_B)

            c00_frag = [mfma.zero_value] * N_ACCUMS
            c01_frag = [mfma.zero_value] * N_ACCUMS
            c10_frag = [mfma.zero_value] * N_ACCUMS
            c11_frag = [mfma.zero_value] * N_ACCUMS

            # B2 closure constants — trace at Python time, fold into IR.
            _NN_SP = SETPRIO_HIGH
            _NN_SH = SCHED_HINT

            def _prio_raise():
                if _NN_SP > 0:
                    rocdl.s_setprio(_NN_SP)

            def _prio_drop():
                if _NN_SP > 0:
                    rocdl.s_setprio(0)

            def _sched_after_mfma():
                if _NN_SH == 1:
                    rocdl.sched_group_barrier(rocdl.mask_mfma, N_ACCUMS, 0)
                elif _NN_SH == 2:
                    rocdl.sched_group_barrier(rocdl.mask_mfma, N_ACCUMS, 0)
                    rocdl.sched_group_barrier(rocdl.mask_vmem_rd, 1, 1)
                    rocdl.sched_group_barrier(rocdl.mask_dsrd, 1, 2)

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

            # Main loop.
            for k in range_constexpr(K_ITERS - 2):
                b0_frag = b_s2r.load(b_cur0)
                a0_frag = a_s2r.load(a_cur0)
                a_g2s.load(a_next1, A1_gl_offset + (k + 1) * BLOCK_K)
                rocdl.s_barrier()

                _prio_raise()
                c00_frag = mfma.call(a0_frag, b0_frag, c00_frag)
                _prio_drop()
                _sched_after_mfma()
                rocdl.s_barrier()

                b1_frag = b_s2r.load(b_cur1)
                b_g2s.load(b_cur0, B0_gl_offset + (k + 2) * BLOCK_K * c_n)
                rocdl.s_barrier()

                _prio_raise()
                c01_frag = mfma.call(a0_frag, b1_frag, c01_frag)
                _prio_drop()
                _sched_after_mfma()
                rocdl.s_barrier()

                a1_frag = a_s2r.load(a_cur1)
                a_g2s.load(a_cur0, A0_gl_offset + (k + 2) * BLOCK_K)
                rocdl.s_barrier()

                _prio_raise()
                c10_frag = mfma.call(a1_frag, b0_frag, c10_frag)
                _prio_drop()
                _sched_after_mfma()
                rocdl.s_barrier()

                b_g2s.load(b_cur1, B1_gl_offset + (k + 2) * BLOCK_K * c_n)
                wait_barrier(2 * N_LDS_STEPS_A + N_LDS_STEPS_B)

                _prio_raise()
                c11_frag = mfma.call(a1_frag, b1_frag, c11_frag)
                _prio_drop()
                _sched_after_mfma()
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
                value_attrs=(
                    {
                        "rocdl.waves_per_eu": WAVES_PER_EU,
                        "rocdl.flat_work_group_size": "512,512",
                        "passthrough": [["amdgpu-num-vgpr", str(MAXNREG)]],
                    }
                    if MAXNREG > 0
                    else {"rocdl.waves_per_eu": WAVES_PER_EU, "rocdl.flat_work_group_size": "512,512"}
                ),
            ).launch(grid=(grid_x, 1, 1), block=(512, 1, 1), stream=stream)

        return launch_dense_nn

    # ──────────────────────────────────────────────────────────────────────
    # Session B7: persistent NN variant — sametile probe to confirm/refute
    # whether the A11 iter>0 race (WG-state pollution) reproduces in the
    # NN layout. Structural mirror of `_compile_dense_nt_persistent`:
    #   - scf.for outer loop (`for tile_iv in range(PERSIST_FACTOR)`)
    #   - all helpers (mfma, g2s, s2r, store_c) instantiated per iter
    #   - LDS slot ptrs re-bound each iter
    #   - cross-iter drain ASM (vmcnt/lgkmcnt/expcnt + s_barrier)
    #   - prelude → main K-loop → epilog 1 → epilog 2 → store
    #
    # NN-specific differences from NT persistent:
    #   - B offset is column-based: `block_n * BLOCK_N + {0, LDS_BLOCK_N}`
    #     (no `*K` because B is [K, N] row-major)
    #   - B K-iter step uses `(k + x) * BLOCK_K * c_n` (stride-by-c_n rows)
    #   - B uses `compute_global_swizzle_nn(lane_id, wave_id, c_n, ...)`
    #   - B s2r uses `S2RLoaderTr` (ds_read_b64_tr_b8 transpose-load)
    #   - `_ = str(fx.thread_idx.x)` workaround retained (flydsl trace-order
    #     bug for transposed B reads when thread_idx not pre-materialized)
    #
    # B7 sametile probe: when B7_SAMETILE=True AND PF>=2, iter1+ drops
    # tile_iv from pid (forces same coord as iter0). Same discriminator as
    # A11:
    #   - bit-exact vs PF=0 baseline → tile_iv→pid IR alias (hypothesis a)
    #   - residual abs error      → WG-state pollution (hypothesis b)
    # NN replication of A11 verdict expected with high probability (same
    # HW, same 8-wave WG, same mfma scoreboard, similar LDS slot reuse
    # pattern); but explicit probe required for B7 closure.
    # ──────────────────────────────────────────────────────────────────────

    @functools.lru_cache(maxsize=128)
    def _compile_dense_nn_persistent(
        K: int, BLOCK_M: int = 256, BLOCK_N: int = 256, PERSIST_FACTOR: int = 2,
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

    # ──────────────────────────────────────────────────────────────────────
    # Session C3: CRR (TN/wgrad) dense kernel.
    # C[k_out, n_out] = sum_{m_contract} A[m_contract, k_out] * B[m_contract, n_out]
    # A storage [M_contract, K_out] row-major K_out-contig.
    # B storage [M_contract, N_out] row-major N_out-contig (== NN B byte-identical).
    # C storage [K_out, N_out] row-major N_out-contig.
    # mfma view: mfma_M = K_out (16), mfma_K = M_contract (128), mfma_N = N_out (16).
    # Both A and B need transpose-load (contraction = storage outer dim) → both
    # use S2RLoaderTr; A-path g2s uses compute_global_swizzle_nn with K_out
    # as the row-stride (mirror of NN B with K_inner → M_contract rename).
    # Probe verified: scripts2/probe_transpose_crr_{a,b}_lds.py (3-leg PASS).
    # Kernel structure is a 1:1 mirror of `_compile_dense_nn` with:
    #   - A_gl_offset uses block_m * BLOCK_M (column-start, like NN B), not
    #     (block_m * BLOCK_M) * K
    #   - A K-iter additive = k * BLOCK_K * c_m (row-stride × k), not k * BLOCK_K
    #   - A path uses compute_global_swizzle_nn (stride c_m) + S2RLoaderTr(wave_m)
    # ──────────────────────────────────────────────────────────────────────
    @functools.lru_cache(maxsize=128)
    def _compile_dense_crr(
        K: int,
        BLOCK_M: int = 256,
        BLOCK_N: int = 256,
        WAVES_PER_EU: int = 2,
        SETPRIO_HIGH: int = 1,
        SCHED_HINT: int = 0,
        GROUP_M: int = 1,
        MAXNREG: int = 0,
    ):
        """CRR-layout fp8 dense kernel (TN/wgrad).

        Param naming kept symmetric with NN factory:
          K            - M_contract (contraction extent); compile-time, K % 128 == 0
          BLOCK_M      - K_out tile (mfma_M direction), default 256
          BLOCK_N      - N_out tile (mfma_N direction), default 256
          BLOCK_K      - M_contract tile, fixed 128 (mfma K size)

        Kernel signature inside FlyDSL kernel:
          c_m = K_out   (output row dim, C is [c_m, c_n])
          c_n = N_out   (output col dim)
        """
        BLOCK_K = 128
        assert BLOCK_M >= 128 and BLOCK_N >= 256 and BLOCK_M % 128 == 0 and BLOCK_N % 256 == 0
        assert K % BLOCK_K == 0
        assert GROUP_M >= 1

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
        def kernel_dense_crr(
            A: fx.Tensor,
            B: fx.Tensor,
            C: fx.Tensor,
            A_scale: fx.Tensor,
            B_scale: fx.Tensor,
            c_m: fx.Int32,
            c_n: fx.Int32,
        ):
            # Trace-order workaround (mirror NN; see _compile_dense_nn comment).
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
            pid = fx.block_idx.x
            num_pid_in_group = GROUP_M * n_blocks
            group_id = pid // num_pid_in_group
            pid_in_group = pid % num_pid_in_group
            pid_m_inner = pid_in_group % GROUP_M
            pid_n = pid_in_group // GROUP_M
            block_m = group_id * GROUP_M + pid_m_inner
            block_n = pid_n

            # CRR A: [M_contract, K_out] row-major K_out-contig. We hold a
            # fixed K_out tile [block_m*BLOCK_M : ... + BLOCK_M] and stream
            # M_contract rows. Per-WG starts are K_out column positions.
            A0_gl_offset = block_m * BLOCK_M + 0
            A1_gl_offset = block_m * BLOCK_M + LDS_BLOCK_M

            # CRR B: [M_contract, N_out] row-major N_out-contig. Identical
            # in form to NN B (M_contract == K_inner). Per-WG starts are
            # N_out column positions.
            B0_gl_offset = block_n * BLOCK_N + 0
            B1_gl_offset = block_n * BLOCK_N + LDS_BLOCK_N

            gA = make_fp8_buffer_tensor(A, F8_IR_t)
            gB = make_fp8_buffer_tensor(B, F8_IR_t)
            a_div = fx.logical_divide(gA, fx.make_layout(1, 1))
            b_div = fx.logical_divide(gB, fx.make_layout(1, 1))

            # A path g2s: same shape as NN B's g2s — stride = K_out = c_m.
            gl_off_a = compute_global_swizzle_nn(lane_id, wave_id, c_m, N_LDS_ROUNDS)
            # B path g2s: NN B's exact helper, stride = N_out = c_n.
            gl_off_b = compute_global_swizzle_nn(lane_id, wave_id, c_n, N_LDS_ROUNDS)

            mfma = Mfma16x16x128(N_TILES_A, N_TILES_B)

            a_g2s = G2SLoader(a_div, gl_off_a, N_LDS_STEPS_A, F8_IR_t, wave_id)
            b_g2s = G2SLoader(b_div, gl_off_b, N_LDS_STEPS_B, F8_IR_t, wave_id)
            # A path uses S2RLoaderTrA (parameterized wave-stride =
            # n_tiles_a * 16; needed because wave_m ∈ [0,2) with N_TILES_A=4
            # requires stride 64, not S2RLoaderTr's hardcoded 32 which was
            # specialized for NN B's wave_n ∈ [0,4) × N_TILES_B=2 case).
            # B path = NN B direct reuse (S2RLoaderTr).  Per CRR C1/C2
            # probes (32768/32768 PASS).
            a_s2r = S2RLoaderTrA(wave_m, N_TILES_A)
            b_s2r = S2RLoaderTr(wave_n, N_TILES_B)
            store_c = StoreC(A_scale, B_scale, C, c_m, c_n, mfma.idx, N_TILES_A, N_TILES_B)

            c00_frag = [mfma.zero_value] * N_ACCUMS
            c01_frag = [mfma.zero_value] * N_ACCUMS
            c10_frag = [mfma.zero_value] * N_ACCUMS
            c11_frag = [mfma.zero_value] * N_ACCUMS

            _CRR_SP = SETPRIO_HIGH
            _CRR_SH = SCHED_HINT

            def _prio_raise():
                if _CRR_SP > 0:
                    rocdl.s_setprio(_CRR_SP)

            def _prio_drop():
                if _CRR_SP > 0:
                    rocdl.s_setprio(0)

            def _sched_after_mfma(idx=None):
                if _CRR_SH == 1:
                    rocdl.sched_group_barrier(rocdl.mask_mfma, N_ACCUMS, 0)
                elif _CRR_SH == 2:
                    rocdl.sched_group_barrier(rocdl.mask_mfma, N_ACCUMS, 0)
                    rocdl.sched_group_barrier(rocdl.mask_vmem_rd, 1, 1)
                    rocdl.sched_group_barrier(rocdl.mask_dsrd, 1, 2)
                elif _CRR_SH == 3 and idx == 1:
                    rocdl.sched_group_barrier(rocdl.mask_mfma, N_ACCUMS, 0)
                elif _CRR_SH == 4 and idx == 0:
                    rocdl.sched_group_barrier(rocdl.mask_mfma, N_ACCUMS, 0)
                elif _CRR_SH == 5 and idx == 2:
                    rocdl.sched_group_barrier(rocdl.mask_mfma, N_ACCUMS, 0)

            # Prelude.  A K-iter step in flat A units = BLOCK_K * c_m
            # (M_contract advances by BLOCK_K, with K_out=c_m as row stride).
            # B K-iter step in flat B units = BLOCK_K * c_n (same as NN).
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

            # Main loop (mirror NN ping-pong).
            for k in range_constexpr(K_ITERS - 2):
                b0_frag = b_s2r.load(b_cur0)
                a0_frag = a_s2r.load(a_cur0)
                a_g2s.load(a_next1, A1_gl_offset + (k + 1) * BLOCK_K * c_m)
                rocdl.s_barrier()

                _prio_raise()
                c00_frag = mfma.call(a0_frag, b0_frag, c00_frag)
                _prio_drop()
                _sched_after_mfma(0)
                rocdl.s_barrier()

                b1_frag = b_s2r.load(b_cur1)
                b_g2s.load(b_cur0, B0_gl_offset + (k + 2) * BLOCK_K * c_n)
                rocdl.s_barrier()

                _prio_raise()
                c01_frag = mfma.call(a0_frag, b1_frag, c01_frag)
                _prio_drop()
                _sched_after_mfma(1)
                rocdl.s_barrier()

                a1_frag = a_s2r.load(a_cur1)
                a_g2s.load(a_cur0, A0_gl_offset + (k + 2) * BLOCK_K * c_m)
                rocdl.s_barrier()

                _prio_raise()
                c10_frag = mfma.call(a1_frag, b0_frag, c10_frag)
                _prio_drop()
                _sched_after_mfma(2)
                rocdl.s_barrier()

                b_g2s.load(b_cur1, B1_gl_offset + (k + 2) * BLOCK_K * c_n)
                wait_barrier(2 * N_LDS_STEPS_A + N_LDS_STEPS_B)

                _prio_raise()
                c11_frag = mfma.call(a1_frag, b1_frag, c11_frag)
                _prio_drop()
                _sched_after_mfma(3)
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
            a_g2s.load(a_next1, A1_gl_offset + (k + 1) * BLOCK_K * c_m)  # stale-a1 fix
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
        def launch_dense_crr(
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
            kernel_dense_crr(
                A,
                B,
                C,
                A_scale,
                B_scale,
                c_m,
                c_n,
                value_attrs=(
                    {
                        "rocdl.waves_per_eu": WAVES_PER_EU,
                        "rocdl.flat_work_group_size": "512,512",
                        "passthrough": [["amdgpu-num-vgpr", str(MAXNREG)]],
                    }
                    if MAXNREG > 0
                    else {"rocdl.waves_per_eu": WAVES_PER_EU, "rocdl.flat_work_group_size": "512,512"}
                ),
            ).launch(grid=(grid_x, 1, 1), block=(512, 1, 1), stream=stream)

        return launch_dense_crr

else:

    def _compile_dense_nt(K: int, BLOCK_M: int = 256, BLOCK_N: int = 256):
        raise ImportError("FlyDSL is not available -- this entry should be gated by can_handle.")

    def _compile_dense_nt_persistent(
        K: int, BLOCK_M: int = 256, BLOCK_N: int = 256, PERSIST_FACTOR: int = 2,
        A11_SAMETILE: bool = False,
    ):
        raise ImportError("FlyDSL is not available -- this entry should be gated by can_handle.")

    def _compile_dense_nn(
        K: int,
        BLOCK_M: int = 256,
        BLOCK_N: int = 256,
        WAVES_PER_EU: int = 2,
        SETPRIO_HIGH: int = 1,
        SCHED_HINT: int = 0,
        GROUP_M: int = 1,
    ):
        raise ImportError("FlyDSL is not available -- this entry should be gated by can_handle.")

    def _compile_dense_nn_persistent(
        K: int, BLOCK_M: int = 256, BLOCK_N: int = 256, PERSIST_FACTOR: int = 2,
        B7_SAMETILE: bool = False,
    ):
        raise ImportError("FlyDSL is not available -- this entry should be gated by can_handle.")

    def _compile_dense_crr(
        K: int,
        BLOCK_M: int = 256,
        BLOCK_N: int = 256,
        WAVES_PER_EU: int = 2,
        SETPRIO_HIGH: int = 1,
        SCHED_HINT: int = 0,
        GROUP_M: int = 1,
        MAXNREG: int = 0,
    ):
        raise ImportError("FlyDSL is not available -- this entry should be gated by can_handle.")


# ──────────────────────────────────────────────────────────────────────────────
# Compiled-callable cache. Avoids paying the `flyc.compile` cost on every call
# (single-test benchmarks bind compiled = flyc.compile(...) once and reuse).
# ──────────────────────────────────────────────────────────────────────────────

_COMPILED_DENSE_CACHE: dict = {}


def _get_compiled_dense(launch, args, maxnreg: int = 0):
    """Return a cached compiled launcher whose signature matches `args`.
    Caches by tuple of (shape, dtype) per tensor + int values, so different
    M/N/K trigger a fresh compile but identical shapes reuse the kernel object.

    `maxnreg > 0` injects `--amdgpu-num-vgpr=<maxnreg>` via FlyDSL
    CompilationContext.compile_hints (forces backend to spill VGPR accumulators
    into AGPR when V budget is squeezed). `maxnreg == 0` = no LLVM hint = B3
    baseline behavior.
    """
    key_parts = [id(launch), ("maxnreg", maxnreg)]
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
        if maxnreg > 0:
            with _CompilationContext.compile_hints({"maxnreg": maxnreg}):
                cached = flyc.compile(launch, *args)
        else:
            cached = flyc.compile(launch, *args)
        _COMPILED_DENSE_CACHE[key] = cached
    return cached


# ──────────────────────────────────────────────────────────────────────────────
# Helpers shared with the layer-2 backend.
# ──────────────────────────────────────────────────────────────────────────────


def _as_i8_flat(t: torch.Tensor) -> torch.Tensor:
    if "float8" in str(t.dtype):
        return t.contiguous().view(torch.int8).view(-1)
    return t.contiguous().view(-1)


def _broadcast_scale(scale: torch.Tensor, length: int, device: torch.device) -> torch.Tensor:
    # C5.1 wrap-overhead lever: scalar path uses expand+contiguous on GPU
    # (no host sync), saving ~26us/call vs the previous .item()+torch.full path.
    # Verified correctness-equivalent (produces same (length,) fp32 tensor).
    if scale.numel() == 1:
        s = scale.to(dtype=torch.float32, device=device).view(1)
        return s.expand(length).contiguous()
    if scale.numel() == length:
        return scale.to(dtype=torch.float32, device=device).contiguous().view(-1)
    raise ValueError(
        f"per-tensor wrapper expected scale.numel() in {{1, {length}}}, got {scale.shape}"
    )


def _canonicalize_nt(
    a: torch.Tensor, b: torch.Tensor, trans_a: bool, trans_b: bool
):
    """Coerce (a, b, trans_a, trans_b) to native NT layout:
        A in [M, K] row-major + B in [N, K] row-major (B^T form)
    by host-transposing whichever operand was passed in a non-canonical layout.
    """
    if trans_a:
        a_nt = a.transpose(-1, -2).contiguous()
    else:
        a_nt = a
    if trans_b:
        b_nt = b
    else:
        b_nt = b.transpose(-1, -2).contiguous()
    M, K_a = a_nt.shape
    N, K_b = b_nt.shape
    assert K_a == K_b, f"K mismatch after canonicalize: a_nt {a_nt.shape}, b_nt {b_nt.shape}"
    return a_nt, b_nt, M, N, K_a


# ──────────────────────────────────────────────────────────────────────────────
# A5 per-shape GROUP_M autotune table (verdict: PARTIAL_REJECT — see docstring).
#
# Methodology lesson:
#   rocprofv3 kernel-only probe (single-shape process, GPU 4 / chi2762,
#   200-iter median, /tmp/a5_gm_summary.csv) showed 13/16 K-align shapes
#   winning with GM=4 by 0.31-4.10%. Full 24-shape bench 3-run median REJECTED
#   that conclusion:
#       GM=1 pure              → fly/tri 0.9507 ✓ (winner)
#       GM=4 all shapes        → fly/tri 0.9488 (-0.2pp, noise tie)
#       GM=8 for (3072,4096)   → fly/tri 0.9376 (-1.3pp, regression)
#   Bench run-to-run noise ≈3pp dominates any per-shape probe-claimed win
#   (which were all <5%). Cross-shape L2 pollution + per-shape allocator
#   interference flip probe winners under bench protocol.
#
#   GROUP_M plumbing kept in `_compile_dense_nt` for future per-shape
#   overrides, but no override is active. Default GM=1 (row-major scan)
#   matches A1 baseline behavior and is empirically robust.
# ──────────────────────────────────────────────────────────────────────────────
_NT_GROUP_M_OVERRIDE: dict[tuple[int, int], int] = {}
_NT_GROUP_M_DEFAULT = 1


def _pick_group_m(N: int, K: int) -> int:
    return _NT_GROUP_M_OVERRIDE.get((N, K), _NT_GROUP_M_DEFAULT)


# B3 — per-shape NN GROUP_M autotune table. Default empty + GM=1 reproduces
# B1 baseline behavior. Populated by B3 sweep if any (N, K) shape proves a
# 3-run-bench-validated win > noise floor (~3pp). Mirror of NT GROUP_M
# infra (kept separate because NN tile-id swizzle has different L2 reuse
# pattern: GROUP_M groups WGs along block_m, NN reuses B[K, block_n_cols]
# strip — same conceptual lever but different physical access pattern).
_NN_GROUP_M_OVERRIDE: dict[tuple[int, int], int] = {}
# B3.3 bench-validated (2026-05-25): 3-run median of bench_rrr_3way_tensorwise.py
# × 16 K-align shapes shows GM=4 strictly dominates GM=1 — all 16 shapes
# Δ +1.4 to +7.9pp, 0 regressions, geomean fly/tri 0.8577 → 0.8896 = +3.19pp.
# GM=8 has 3 shapes with -7..-15pp regression, not safe as default.
_NN_GROUP_M_DEFAULT = 4
# B3 validation knob: FLYDSL_NN_GROUP_M_FORCE > 0 overrides per-shape table.
# 0 = honor _NN_GROUP_M_OVERRIDE + _NN_GROUP_M_DEFAULT (production).
_NN_GROUP_M_FORCE = int(os.environ.get("FLYDSL_NN_GROUP_M_FORCE", "0"))


def _pick_group_m_nn(N: int, K: int) -> int:
    if _NN_GROUP_M_FORCE > 0:
        return _NN_GROUP_M_FORCE
    return _NN_GROUP_M_OVERRIDE.get((N, K), _NN_GROUP_M_DEFAULT)


# Session A7 persistent-NT knob. Default 0 = non-persistent baseline (no
# behavior change). Set FLYDSL_RCR_PERSIST_FACTOR=2 to opt-in. Kernel-only KPI
# gate: worst shape qwen_down_B16_M4096 fly/hbl ≥ 1.00. Only activated when:
#   - GROUP_M == 1 (don't mix with super-block swizzle)
#   - num_tiles % PF == 0 (host-side gate; falls back to non-persistent otherwise)
_NT_PERSIST_FACTOR = int(os.environ.get("FLYDSL_RCR_PERSIST_FACTOR", "0"))
# Phase B2 NN-path sched/setprio/waves_per_eu sweep knobs. Default values
# reproduce the B1 baseline kernel (waves_per_eu=2, setprio raise/drop pair,
# no sched_group_barrier hint). Variants are evaluated by `_b2_run_sweep.sh`
# via env override before module import; per A4 NT lessons all three are
# expected to be ≤ noise.
_NN_WAVES_PER_EU = int(os.environ.get("FLYDSL_NN_WAVES_PER_EU", "2"))
_NN_SETPRIO_HIGH = int(os.environ.get("FLYDSL_NN_SETPRIO_HIGH", "1"))
_NN_SCHED_HINT = int(os.environ.get("FLYDSL_NN_SCHED_HINT", "0"))
# B4 — NN VGPR-cap hook via MLIR llvm.passthrough{"amdgpu-num-vgpr"="N"}.
# 0 (default) = no cap, B3 baseline V=256/A=0 mfma accumulator. When >0,
# the attribute reaches LLVM IR attr block #0 (verified by 20_llvm_ir.ll
# dump) and `next_free_vgpr` rounds up to nearest 32-aligned ≤N. **However:
# force-AGPR hypothesis REJECTED**. mfma instructions emitted by DSL are
# `v_mfma_f32_16x16x128_f8f6f4 v[..], v[..], v[..]` (VGPR-dest variants),
# so LLVM regalloc resolves V-cap by spilling to scratch, NOT shifting D
# accumulators to AGPRs. ISA evidence (chi2762, 2026-05-25):
#   mnr=  0 → V=256 A=0 spill=0     scratch=0   (natural)
#   mnr= 96 → V=192 A=0 spill=427   scratch=544 (cap → all-scratch)
#   mnr=112 → V=224 A=0 spill=47    scratch=172 (cap → some-scratch)
#   mnr≥144 → V=256 A=0 spill=0                 (cap ≥ natural = no-op)
# AGPR is NEVER chosen. To actually force AGPR-accum, DSL must emit
# AGPR-dest mfma intrinsic variants (upstream FlyDSL change). Hook
# plumbing kept for future use (or to study scratch-spill perf curve);
# default 0 has 0 overhead and reproduces B3 baseline bit-exactly.
_NN_MAXNREG = int(os.environ.get("FLYDSL_NN_MAXNREG", "0"))
# A11 sametile probe: when 1 AND PF>=2, iter1+ uses iter0's pid (drops
# tile_iv from coord). Only meaningful for diagnostic; correctness is
# "iter1 overwrites iter0 with same coords" so result is undefined for
# non-multiple-of-PF num_tiles. Use scripts2/_a11_same_tile_diag.py.
_NT_A11_SAMETILE = int(os.environ.get("FLYDSL_RCR_A11_SAMETILE", "0")) == 1

# Session B7 persistent-NN knobs (mirror NT). Default 0 = non-persistent
# baseline (no behavior change). Set FLYDSL_NN_PERSIST_FACTOR=2 to opt-in.
# B7_SAMETILE probe: when 1 AND PF>=2, iter1+ drops tile_iv from pid.
# Only activated when:
#   - GROUP_M == 1 (don't mix with super-block swizzle; B3 default GM=4
#     so probe MUST set FLYDSL_NN_GROUP_M_FORCE=1)
#   - num_tiles % PF == 0 (host-side gate; falls back to non-persistent otherwise)
_NN_PERSIST_FACTOR = int(os.environ.get("FLYDSL_NN_PERSIST_FACTOR", "0"))
_NN_B7_SAMETILE = int(os.environ.get("FLYDSL_NN_B7_SAMETILE", "0")) == 1

# Session C3: CRR (TN/wgrad) knobs. Mirror NN defaults; C3 is a skeleton/G0
# correctness gate (no perf sweeps yet — those land in C4/C5/C6).
_CRR_WAVES_PER_EU = int(os.environ.get("FLYDSL_CRR_WAVES_PER_EU", "2"))
_CRR_SETPRIO_HIGH = int(os.environ.get("FLYDSL_CRR_SETPRIO_HIGH", "1"))
_CRR_SCHED_HINT = int(os.environ.get("FLYDSL_CRR_SCHED_HINT", "0"))
_CRR_MAXNREG = int(os.environ.get("FLYDSL_CRR_MAXNREG", "0"))
_CRR_GROUP_M_OVERRIDE: dict[tuple[int, int], int] = {
    # C5.2 GROUP_M sweep wins on large-M_contract shapes (worst-tier per C4 baseline).
    # Key = (N_out, M_contract). Picked by single-shape min-of-3-trial ratio vs Triton TN.
    # Safety: _pick_group_m_crr guards against K_out tile count not divisible by GROUP_M
    # (kernel dispatcher assumes evenly divisible — otherwise OOB writes corrupt C).
    (4096, 32768): 4,  # dsv3 up M_c=32768  K_o=7168 N_o=4096 : +6.3pp vs GM=1
    (4096, 65536): 4,  # dsv3 up M_c=65536  K_o=7168 N_o=4096 : +2.3pp vs GM=1
    (3072, 65536): 4,  # qwen up M_c=65536  K_o=4096 N_o=3072 : +1.8pp vs GM=1
    # C6.1 GROUP_M sweep wins (Δ vs GM=1 > 1.5pp accept threshold)
    (7168,  8192): 2,  # dsv3 down M_c=8192   K_o=2048 N_o=7168 : +2.11pp (M_tiles=8 %2=0)
    (4096, 16384): 2,  # dsv3 up   M_c=16384  K_o=7168 N_o=4096 : +3.01pp (M_tiles=28 %2=0)
    (3072,  8192): 8,  # qwen up   M_c=8192   K_o=4096 N_o=3072 : +3.98pp (M_tiles=16 %8=0)
    (3072, 16384): 4,  # qwen up   M_c=16384  K_o=4096 N_o=3072 : +3.78pp (M_tiles=16 %4=0)
    (3072, 32768): 4,  # qwen up   M_c=32768  K_o=4096 N_o=3072 : +3.44pp (M_tiles=16 %4=0)
}
_CRR_GROUP_M_DEFAULT = 1
_CRR_GROUP_M_FORCE = int(os.environ.get("FLYDSL_CRR_GROUP_M_FORCE", "0"))
_CRR_BLOCK_M = 256  # kept in sync with _compile_dense_crr BLOCK_M default


def _pick_group_m_crr(N: int, K: int, K_out: int = 0) -> int:
    raw = _CRR_GROUP_M_FORCE if _CRR_GROUP_M_FORCE > 0 else \
          _CRR_GROUP_M_OVERRIDE.get((N, K), _CRR_GROUP_M_DEFAULT)
    if raw <= 1 or K_out <= 0:
        return raw
    # Safety: kernel super-block dispatcher requires M_tiles % GROUP_M == 0.
    # Different K_out values can share the same (N_out, M_contract) key, so
    # callers must pass K_out and we fall back to 1 if grouping would over-launch
    # past the valid M-tile range (would write OOB → NaN/corruption).
    m_tiles = (K_out + _CRR_BLOCK_M - 1) // _CRR_BLOCK_M
    if m_tiles % raw != 0:
        return 1
    return raw


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
    """Dense FP8 GEMM (per-tensor scaling).

    Layout dispatch:
      - NT (trans_a=F, trans_b=T): native NT kernel, no host transpose.
      - NN (trans_a=F, trans_b=F): native NN kernel (used by dgrad path), no
        host transpose; uses ds_read_b64_tr_b8 for B-operand load.
      - CRR/TN (trans_a=T, trans_b=F): native CRR kernel (used by wgrad path),
        no host transpose; uses ds_read_b64_tr_b8 for BOTH A and B (both have
        contraction as storage outer dim). Session C3.
      - TT: host-canonicalised to NT (transpose A and B), then NT kernel.
    trans_c=True returned as post-hoc out.t().contiguous() (extra mem copy
    vs Triton which writes [N, M] directly via swapped strides).
    """
    if out_dtype != torch.bfloat16:
        raise NotImplementedError(f"FlyDSL wrapper only emits bf16. Got {out_dtype}.")
    assert a.dim() == 2 and b.dim() == 2

    # Dispatch by layout (NN native path skips canonicalisation).
    if (not trans_a) and (not trans_b):
        # NN native: A [M, K], B [K, N].
        M, K_a = a.shape
        K_b, N = b.shape
        assert K_a == K_b, f"NN K mismatch: a {a.shape}, b {b.shape}"
        K = K_a
        if K % 128 != 0:
            raise NotImplementedError(
                f"FlyDSL dense GEMM requires K % 128 == 0 (BLOCK_K=128). Got K={K}."
            )
        a_scale_v = _broadcast_scale(a_scale_inv, M, a.device)
        b_scale_v = _broadcast_scale(b_scale_inv, N, a.device)
        out = torch.empty((M, N), dtype=out_dtype, device=a.device)
        gm_nn = _pick_group_m_nn(N, K)
        num_tiles_nn = ((M + 255) // 256) * ((N + 255) // 256)
        pf_nn = _pick_persist_factor_nn(num_tiles_nn, gm_nn)
        if pf_nn > 0:
            launch = _compile_dense_nn_persistent(
                K=K, BLOCK_M=256, BLOCK_N=256, PERSIST_FACTOR=pf_nn,
                B7_SAMETILE=_NN_B7_SAMETILE,
            )
        else:
            launch = _compile_dense_nn(
                K=K, BLOCK_M=256, BLOCK_N=256,
                WAVES_PER_EU=_NN_WAVES_PER_EU,
                SETPRIO_HIGH=_NN_SETPRIO_HIGH,
                SCHED_HINT=_NN_SCHED_HINT,
                GROUP_M=gm_nn,
                MAXNREG=_NN_MAXNREG,
            )
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
        _get_compiled_dense(launch, args, maxnreg=_NN_MAXNREG)(*args)
    elif trans_a and (not trans_b):
        # CRR native (TN / wgrad): A [M_contract, K_out], B [M_contract, N_out],
        # C = A^T @ B has shape [K_out, N_out]. Both A and B need transpose-load
        # because contraction = storage outer dim. Probe verified at
        # scripts2/probe_transpose_crr_{a,b}_lds.py (32768/32768 bytes PASS).
        M_a, K_out = a.shape
        M_b, N_out = b.shape
        assert M_a == M_b, f"CRR contraction-M mismatch: a {a.shape}, b {b.shape}"
        M_contract = M_a
        if M_contract % 128 != 0:
            raise NotImplementedError(
                f"FlyDSL CRR requires M_contract % 128 == 0 (BLOCK_K=128). "
                f"Got M_contract={M_contract}."
            )
        # Tensorwise: scale is scalar (numel==1), broadcast to output dims.
        a_scale_v = _broadcast_scale(a_scale_inv, K_out, a.device)
        b_scale_v = _broadcast_scale(b_scale_inv, N_out, a.device)
        out = torch.empty((K_out, N_out), dtype=out_dtype, device=a.device)
        gm_crr = _pick_group_m_crr(N_out, M_contract, K_out)
        launch = _compile_dense_crr(
            K=M_contract, BLOCK_M=256, BLOCK_N=256,
            WAVES_PER_EU=_CRR_WAVES_PER_EU,
            SETPRIO_HIGH=_CRR_SETPRIO_HIGH,
            SCHED_HINT=_CRR_SCHED_HINT,
            GROUP_M=gm_crr,
            MAXNREG=_CRR_MAXNREG,
        )
        args = (
            _as_i8_flat(a),
            _as_i8_flat(b),
            out.contiguous().view(-1),
            a_scale_v,
            b_scale_v,
            K_out,
            N_out,
            torch.cuda.current_stream(),
        )
        _get_compiled_dense(launch, args, maxnreg=_CRR_MAXNREG)(*args)
    else:
        # NT native OR TT via host canonicalisation.
        a_nt, b_nt, M, N, K = _canonicalize_nt(a, b, trans_a, trans_b)
        if K % 128 != 0:
            raise NotImplementedError(
                f"FlyDSL dense GEMM requires K % 128 == 0 (BLOCK_K=128). Got K={K}."
            )
        a_scale_v = _broadcast_scale(a_scale_inv, M, a.device)
        b_scale_v = _broadcast_scale(b_scale_inv, N, a.device)
        out = torch.empty((M, N), dtype=out_dtype, device=a.device)
        gm = _pick_group_m(N, K)
        num_tiles = ((M + 255) // 256) * ((N + 255) // 256)
        pf = _pick_persist_factor(num_tiles, gm)
        if pf > 0:
            launch = _compile_dense_nt_persistent(
                K=K, BLOCK_M=256, BLOCK_N=256, PERSIST_FACTOR=pf,
                A11_SAMETILE=_NT_A11_SAMETILE,
            )
        else:
            launch = _compile_dense_nt(K=K, BLOCK_M=256, BLOCK_N=256, GROUP_M=gm)
        args = (
            _as_i8_flat(a_nt),
            _as_i8_flat(b_nt),
            out.contiguous().view(-1),
            a_scale_v,
            b_scale_v,
            M,
            N,
            torch.cuda.current_stream(),
        )
        _get_compiled_dense(launch, args)(*args)

    if trans_c:
        return out.t().contiguous()
    return out
