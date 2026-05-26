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
  - NT, NN, TT supported (TT via host transpose to NT); trans_c via post-hoc transpose
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

else:

    def _compile_dense_nt(K: int, BLOCK_M: int = 256, BLOCK_N: int = 256):
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
      - TT: host-canonicalised to NT (transpose A and B), then NT kernel.
      - TN: not supported (raises). For wgrad use grouped path.
    trans_c=True returned as post-hoc out.t().contiguous().
    """
    if out_dtype != torch.bfloat16:
        raise NotImplementedError(f"FlyDSL wrapper only emits bf16. Got {out_dtype}.")
    assert a.dim() == 2 and b.dim() == 2

    if trans_a and (not trans_b):
        raise NotImplementedError(
            "FlyDSL dense fp8 GEMM no longer supports TN/CRR layout in this module. "
            "Use the grouped-gemm path for wgrad."
        )

    # Dispatch by layout (NN native path skips canonicalisation).
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
