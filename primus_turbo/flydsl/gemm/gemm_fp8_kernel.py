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

Algorithm: 8-wave (512 thread WG, wave_m=2 × wave_n=4), BLOCK_M=BLOCK_N=256,
BLOCK_K=128, LDS_BLOCK_M=LDS_BLOCK_N=128, 2×2 mma ping-pong (c00/c01/c10/c11
per wave), `mfma_f32_16x16x128_f8f6f4`. NN B-operand load uses
ds_read_b64_tr_b8 transpose-load (S2RLoaderTr); NT shares the FlyDSL repo's
S2RLoader. Includes the c10/c11 "stale a_cur1" pipeline fix in epilog 1.

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


# Module-load-time FlyDSL discovery. `@flyc.kernel` needs FlyDSL utils as
# module globals (not closure cells); inject FlyDSL repo root if missing.

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
            make_fp8_buffer_tensor,
            pack_i32x4_i32x8,
            swizzle_128,
            wait_barrier,
        )

    import flydsl.compiler as flyc
    import flydsl.expr as fx
    from flydsl._mlir import ir
    from flydsl._mlir.dialects import llvm as _llvm
    from flydsl.expr import arith, buffer_ops as _buffer_ops, const_expr, range_constexpr, rocdl
    from flydsl.expr.utils.arith import ArithValue
    from flydsl.expr.arith import _to_raw as _raw
    from flydsl.expr.typing import T, Vector as Vec

    _FLYDSL_OK = True
except ImportError:
    pass


def flydsl_available() -> bool:
    """Reported via backend can_handle to gate dispatch."""
    return _FLYDSL_OK


if _FLYDSL_OK:

    def _make_value_attrs(waves_per_eu, agpr_alloc, fwg):
        """Build kernel value_attrs dict.

        agpr_alloc encoding:
          0       — no AGPR hint (compiler default)
          N > 0   — force EXACT N AGPRs (passthrough "N,N")
          -N      — allow up to N AGPRs (passthrough "0,N")
        """
        d = {"rocdl.waves_per_eu": waves_per_eu, "rocdl.flat_work_group_size": fwg}
        if agpr_alloc != 0:
            if agpr_alloc < 0:
                alloc = f"0,{-agpr_alloc}"
            else:
                alloc = f"{agpr_alloc},{agpr_alloc}"
            d["passthrough"] = [
                ["amdgpu-agpr-alloc", alloc],
                ["amdgpu-mfma-vgpr-form", "false"],
            ]
        return d

    def _asm_mma_do(a, b, c, mode="2"):
        """fp8 16x16x128 MFMA via inline asm. Replaces `mma_atom_call_ssa`
        (FlyDSL core, untouched) to control the dst register class — the lever
        the intrinsic path can't express — and so kill the MFMA dst/srcA WAR
        hazard that makes barrier-reduced TN schedules nondeterministic.

        ``mode`` (param; defaults to env TN_ASM_MMA at the call site):
          1  `=&v,v,v,0`  VGPR accum + early-clobber: dst forced disjoint from
             srcA/srcB. Deterministic but the early-clobber also removes the
             scheduling freedom (and accum eats VGPR → low occupancy).
          2  `=a,v,v,0`   AGPR accum: dst/srcC in AGPR, srcA/srcB in VGPR —
             physically separate register files, so dst can NEVER alias srcA
             (det=0 by construction, no early-clobber needed → full scheduling
             freedom) AND the accumulators leave the VGPR file (higher occ).
        """
        v4f32 = ir.VectorType.get([4], ir.F32Type.get())
        cons = "=a,v,v,0" if str(mode) == "2" else "=&v,v,v,0"
        op = _llvm.InlineAsmOp(
            res=v4f32,
            operands_=[_raw(a), _raw(b), _raw(c)],
            asm_string="v_mfma_f32_16x16x128_f8f6f4 $0, $1, $2, $0",
            constraints=cons,
            has_side_effects=False,
        )
        return Vec(op.result)

    @functools.lru_cache(maxsize=256)
    def _compile_dense_nt(
        K: int,
        BLOCK_M: int = 256,
        BLOCK_N: int = 256,
        GROUP_M: int = 1,
        barrier_mask: int = 0x7F,
        waves_per_eu: int = 2,
        agpr_alloc: int = 0,
        split_barrier: bool = False,
        sched_mask: int = 0,
        nt_vmcnt: int = 3,              # end-of-iter s_waitcnt vmcnt(N): N=3 → det=0 (gfx950 G2S buffer_load_lds/ds_read LDS hazard), <=1.1% cost; N>=4 races, N<3 costlier; -1 disables
    ):
        """Build & cache the (K, BLOCK_M, BLOCK_N, GROUP_M, barrier_mask)-specialised launch.

        ``GROUP_M`` enables Triton-style super-block tile-id swizzle for L2
        reuse: WGs advance ``block_m`` first within each ``GROUP_M × n_blocks``
        band. ``GROUP_M == 1`` = plain row-major scan.

        ``barrier_mask`` (default 0x7F = all 7 barriers ON, the conservative
        original schedule). Bits B1..B7 control the 7 ``_barrier_inline()``
        calls inside the main K-loop (B1/B3/B5 before each MFMA, B2/B4/B6/B7
        after). Probe-tested per shape; removing any barrier risks
        compiler-reorder races, validate SNR ≥ 20 dB.
        """
        BR_B1 = bool(barrier_mask & 0x01)
        BR_B2 = bool(barrier_mask & 0x02)
        BR_B3 = bool(barrier_mask & 0x04)
        BR_B4 = bool(barrier_mask & 0x08)
        BR_B5 = bool(barrier_mask & 0x10)
        BR_B6 = bool(barrier_mask & 0x20)
        BR_B7 = bool(barrier_mask & 0x40)
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

            # sched_mask uses POSITION within K-iter. Each K-iter emits 7
            # barriers (B1..B7). Bit N (N in 0..6) → use sched_barrier(0)
            # for all barriers at position N across the unrolled K-loop.
            # Prologue/epilog barriers always use HW s_barrier (outside the
            # 7-cycle). Caller sets _prologue_offset[0] = barrier_idx after
            # prologue done.
            _barrier_idx = [0]
            _prologue_offset = [-1]   # -1 = still in prologue
            def _barrier_inline():
                idx = _barrier_idx[0]
                _barrier_idx[0] = idx + 1
                if _prologue_offset[0] >= 0:
                    pos = (idx - _prologue_offset[0]) % 7
                    if sched_mask & (1 << pos):
                        rocdl.sched_barrier(0)
                        return
                if split_barrier:
                    rocdl.s_barrier_signal(-1)
                    rocdl.s_barrier_wait(-1)
                else:
                    rocdl.s_barrier()

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
            # Triton-style super-block swizzle for L2 reuse. Tail handled via
            # group_size_m = min(num_pid_m - first_pid_m, GROUP_M) so any
            # GROUP_M ≥ 1 is correct regardless of num_pid_m % GROUP_M.
            # arith.select used since flydsl lacks an integer minimum op.
            num_pid_m = ceildiv(c_m, BLOCK_M)
            pid = fx.block_idx.x
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
                _barrier_inline()

            wait_barrier(N_LDS_STEPS_A + N_LDS_STEPS_B)

            b_g2s.load(b_next0, B0_gl_offset + 1 * BLOCK_K)
            a_g2s.load(a_next0, A0_gl_offset + 1 * BLOCK_K)
            b_g2s.load(b_next1, B1_gl_offset + 1 * BLOCK_K)

            wait_barrier(N_LDS_STEPS_A + 2 * N_LDS_STEPS_B)
            # Mark main-loop start so sched_mask can index barrier position % 7.
            _prologue_offset[0] = _barrier_idx[0]

            # Main K-loop. Each iter: s2r {a0,b0,b1,a1} → 4 mma (c00→c01→c10→c11)
            # interleaved with k+1 (a_next1) and k+2 (a_cur0, b_cur0, b_cur1) prefetches.
            for k in range_constexpr(K_ITERS - 2):
                b0_frag = b_s2r.load(b_cur0)
                a0_frag = a_s2r.load(a_cur0)
                a_g2s.load(a_next1, A1_gl_offset + (k + 1) * BLOCK_K)
                if BR_B1: _barrier_inline()

                rocdl.s_setprio(1)
                c00_frag = mfma.call(a0_frag, b0_frag, c00_frag)
                rocdl.s_setprio(0)
                if BR_B2: _barrier_inline()

                b1_frag = b_s2r.load(b_cur1)
                b_g2s.load(b_cur0, B0_gl_offset + (k + 2) * BLOCK_K)
                if BR_B3: _barrier_inline()

                rocdl.s_setprio(1)
                c01_frag = mfma.call(a0_frag, b1_frag, c01_frag)
                rocdl.s_setprio(0)
                if BR_B4: _barrier_inline()

                a1_frag = a_s2r.load(a_cur1)
                a_g2s.load(a_cur0, A0_gl_offset + (k + 2) * BLOCK_K)
                if BR_B5: _barrier_inline()

                rocdl.s_setprio(1)
                c10_frag = mfma.call(a1_frag, b0_frag, c10_frag)
                rocdl.s_setprio(0)
                if BR_B6: _barrier_inline()

                b_g2s.load(b_cur1, B1_gl_offset + (k + 2) * BLOCK_K)
                wait_barrier(2 * N_LDS_STEPS_A + N_LDS_STEPS_B)

                rocdl.s_setprio(1)
                c11_frag = mfma.call(a1_frag, b1_frag, c11_frag)
                rocdl.s_setprio(0)
                if BR_B7: _barrier_inline()

                if nt_vmcnt >= 0: _llvm.inline_asm(res=None, operands_=[], asm_string=f"s_waitcnt vmcnt({nt_vmcnt})", constraints="", has_side_effects=True)  # end-of-iter G2S drain (race fix)
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
            _barrier_inline()

            rocdl.s_setprio(1)
            c00_frag = mfma.call(a0_frag, b0_frag, c00_frag)
            rocdl.s_setprio(0)
            _barrier_inline()

            b1_frag = b_s2r.load(b_cur1)
            _barrier_inline()

            rocdl.s_setprio(1)
            c01_frag = mfma.call(a0_frag, b1_frag, c01_frag)
            rocdl.s_setprio(0)
            _barrier_inline()

            a1_frag = a_s2r.load(a_cur1)
            _barrier_inline()

            rocdl.s_setprio(1)
            c10_frag = mfma.call(a1_frag, b0_frag, c10_frag)
            rocdl.s_setprio(0)
            _barrier_inline()

            b0_frag = b_s2r.load(b_next0)
            a_g2s.load(a_next1, A1_gl_offset + (k + 1) * BLOCK_K)  # stale-a1 fix
            _barrier_inline()

            rocdl.s_setprio(1)
            c11_frag = mfma.call(a1_frag, b1_frag, c11_frag)
            rocdl.s_setprio(0)
            _barrier_inline()

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
            _barrier_inline()

            b1_frag = b_s2r.load(b_cur1)
            _barrier_inline()

            rocdl.s_setprio(1)
            c01_frag = mfma.call(a0_frag, b1_frag, c01_frag)
            rocdl.s_setprio(0)
            _barrier_inline()

            a1_frag = a_s2r.load(a_cur1)
            _barrier_inline()

            rocdl.s_setprio(1)
            c10_frag = mfma.call(a1_frag, b0_frag, c10_frag)
            c11_frag = mfma.call(a1_frag, b1_frag, c11_frag)
            rocdl.s_setprio(0)
            _barrier_inline()

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
                value_attrs=_make_value_attrs(waves_per_eu, agpr_alloc, "512,512"),
            ).launch(grid=(grid_x, 1, 1), block=(512, 1, 1), stream=stream)

        return launch_dense_nt


    @functools.lru_cache(maxsize=128)
    def _compile_dense_nt_v2(K: int, BLOCK_M: int = 256, BLOCK_N: int = 256, GROUP_M: int = 1,
                              agpr_alloc: int = 0, use_sched_hints: bool = True):
        """V2 NT kernel: 2-stage ping/pong LDS for A+B + single barrier per
        K-iter + sched_* hints (preshuffle_gemm_v2-style structure adapted
        for non-preshuffled B). Same 8-wave 2×4 layout as v1, same MFMA
        layout, same StoreC epilog. Only the K-loop pipeline differs.
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

        # 2-stage ping/pong: each stage has 4 slabs (a_lo, a_hi, b_lo, b_hi)
        # = 8 LDS regions total (same total bytes as v1's cur/next 4-slab,
        # different swap semantics: stages are full K-iter snapshots).
        @fx.struct
        class SharedStorage:
            A0_lo: fx.Array[fx.Float8E4M3FN, a_lds_size, 16]
            A0_hi: fx.Array[fx.Float8E4M3FN, a_lds_size, 16]
            A1_lo: fx.Array[fx.Float8E4M3FN, a_lds_size, 16]
            A1_hi: fx.Array[fx.Float8E4M3FN, a_lds_size, 16]
            B0_lo: fx.Array[fx.Float8E4M3FN, b_lds_size, 16]
            B0_hi: fx.Array[fx.Float8E4M3FN, b_lds_size, 16]
            B1_lo: fx.Array[fx.Float8E4M3FN, b_lds_size, 16]
            B1_hi: fx.Array[fx.Float8E4M3FN, b_lds_size, 16]

        @flyc.kernel(known_block_size=[512, 1, 1])
        def kernel_dense_nt_v2(
            A: fx.Tensor, B_T: fx.Tensor, C: fx.Tensor,
            A_scale: fx.Tensor, B_scale: fx.Tensor,
            c_m: fx.Int32, c_n: fx.Int32,
        ):
            F8_IR_t = fx.Float8E4M3FN.ir_type
            n_blocks = ceildiv(c_n, BLOCK_N)

            lds = fx.SharedAllocator().allocate(SharedStorage).peek()
            A_ld = [[lds.A0_lo, lds.A0_hi], [lds.A1_lo, lds.A1_hi]]
            B_ld = [[lds.B0_lo, lds.B0_hi], [lds.B1_lo, lds.B1_hi]]

            lane_id = fx.thread_idx.x % 64
            wave_id = fx.thread_idx.x // 64
            wave_m = wave_id // 4
            wave_n = wave_id % 4

            num_pid_m = ceildiv(c_m, BLOCK_M)
            pid = fx.block_idx.x
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
            store_c = StoreC(A_scale, B_scale, C, c_m, c_n, mfma.idx, N_TILES_A, N_TILES_B)

            # 4-quadrant c-frags (same as v1, kept for StoreC compat).
            c00 = [mfma.zero_value] * N_ACCUMS
            c01 = [mfma.zero_value] * N_ACCUMS
            c10 = [mfma.zero_value] * N_ACCUMS
            c11 = [mfma.zero_value] * N_ACCUMS

            # Each prefetch issues PREFETCH_VMEM = 2*(N_LDS_STEPS_A+N_LDS_STEPS_B)
            # vmem ops via 4 G2SLoader.load calls. wait_barrier(N) drains to N
            # outstanding vmem.
            PREFETCH_VMEM = 2 * (N_LDS_STEPS_A + N_LDS_STEPS_B)

            def prefetch(stage, k):
                b_g2s.load(B_ld[stage][0], B0_gl_offset + k * BLOCK_K)
                b_g2s.load(B_ld[stage][1], B1_gl_offset + k * BLOCK_K)
                a_g2s.load(A_ld[stage][0], A0_gl_offset + k * BLOCK_K)
                a_g2s.load(A_ld[stage][1], A1_gl_offset + k * BLOCK_K)

            # Prologue: prefetch k=0 → stage 0, k=1 → stage 1.
            # Drain stage-0's prefetch (keep stage-1's PREFETCH_VMEM in flight).
            prefetch(0, 0)
            prefetch(1, 1)
            wait_barrier(PREFETCH_VMEM)
            rocdl.s_barrier()  # cross-wave: ensure all waves see stage-0 LDS

            # Main loop. iter k consumes stage (k%2), prefetches k+2 INTO the
            # SAME stage. v2 prototype (kept for reference) — outperformed by
            # v1's 4-slab cur/next + 7-barrier interleave on benched shapes;
            # 2-stage ping-pong serializes mfma+prefetch.
            for k in range_constexpr(K_ITERS - 2):
                cur = k % 2
                b0 = b_s2r.load(B_ld[cur][0])
                b1 = b_s2r.load(B_ld[cur][1])
                a0 = a_s2r.load(A_ld[cur][0])
                a1 = a_s2r.load(A_ld[cur][1])
                rocdl.s_setprio(1)
                c00 = mfma.call(a0, b0, c00)
                c01 = mfma.call(a0, b1, c01)
                c10 = mfma.call(a1, b0, c10)
                c11 = mfma.call(a1, b1, c11)
                rocdl.s_setprio(0)
                rocdl.s_barrier()
                prefetch(cur, k + 2)
                wait_barrier(PREFETCH_VMEM)

            # Epilog: 2 remaining K-iters from already-prefetched stages.
            for k_ep in range_constexpr(2):
                cur = (K_ITERS - 2 + k_ep) % 2
                b0 = b_s2r.load(B_ld[cur][0])
                b1 = b_s2r.load(B_ld[cur][1])
                a0 = a_s2r.load(A_ld[cur][0])
                a1 = a_s2r.load(A_ld[cur][1])
                rocdl.s_setprio(1)
                c00 = mfma.call(a0, b0, c00)
                c01 = mfma.call(a0, b1, c01)
                c10 = mfma.call(a1, b0, c10)
                c11 = mfma.call(a1, b1, c11)
                rocdl.s_setprio(0)
                if k_ep == 0:
                    rocdl.s_barrier()  # protect stage-(K-1) read from race

            # Scale + store.
            wave_n_offset = wave_n * (N_TILES_B * 16)
            wave_m_offset = wave_m * (N_TILES_A * 16)
            base_row = block_m * BLOCK_M + wave_m_offset
            base_col = block_n * BLOCK_N + wave_n_offset
            store_c.store(c00, base_row + 0,            base_col + 0)
            store_c.store(c01, base_row + 0,            base_col + LDS_BLOCK_N)
            store_c.store(c10, base_row + LDS_BLOCK_M,  base_col + 0)
            store_c.store(c11, base_row + LDS_BLOCK_M,  base_col + LDS_BLOCK_N)

        @flyc.jit
        def launch_dense_nt_v2(
            A: fx.Tensor, B_T: fx.Tensor, C: fx.Tensor,
            A_scale: fx.Tensor, B_scale: fx.Tensor,
            c_m: fx.Int32, c_n: fx.Int32, stream: fx.Stream,
        ):
            grid_x = ceildiv(c_m, BLOCK_M) * ceildiv(c_n, BLOCK_N)
            kernel_dense_nt_v2(
                A, B_T, C, A_scale, B_scale, c_m, c_n,
                value_attrs=_make_value_attrs(2, agpr_alloc, "512,512"),
            ).launch(grid=(grid_x, 1, 1), block=(512, 1, 1), stream=stream)
        return launch_dense_nt_v2


    @functools.lru_cache(maxsize=128)
    def _compile_dense_nt_v3(K: int, BLOCK_M: int = 256, BLOCK_N: int = 256, GROUP_M: int = 1,
                              waves_per_eu: int = 2, sched_mid_barriers: bool = False,
                              add_sched_hints: bool = False, agpr_alloc: int = 0, nt_vmcnt: int = 3):
        """V3 NT kernel: same 8-wave 4-slab cur/next physical structure as
        v1, but each c-quadrant uses _interleaved_cluster pattern (single
        mfma_call_one + single load_one fine-grain interleave, ported from
        fp8_gemm_4wave.py). Hypothesis: same WG-sync barriers as v1 (load-
        bearing for cross-wave LDS), but better mfma/load overlap within
        each barrier-bracketed phase = +2-5% over v1.
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

        # Pre-compute load schedule outside the kernel decorator (Python-only,
        # not subject to flydsl's for→scf.for AST rewriter). For a c-quadrant
        # cluster with N_TILES_A * N_TILES_B mfma_call_one calls, distribute
        # n_g2s_steps g2s.load_one + 2*n_tiles s2r.load_one (= n_s2r) load
        # issues among the mfma slots. Each entry: list of (kind, ...) tuples.
        MFMA_COUNT = N_TILES_A * N_TILES_B
        def _build_schedule(n_g2s_steps, n_tiles_s2r):
            n_s2r = n_tiles_s2r * 2
            load_count = n_g2s_steps + n_s2r
            slots = [[] for _ in range(MFMA_COUNT)]
            for li in range(load_count):
                slot = (li * MFMA_COUNT) // load_count
                if li < n_g2s_steps:
                    slots[slot].append(("g", li))
                else:
                    s_idx = li - n_g2s_steps
                    slots[slot].append(("s", s_idx // 2, s_idx % 2, s_idx))
            return slots
        # Per c-quadrant cluster type (a/b prefetch dim varies).
        SCHED_A_PREFETCH = _build_schedule(N_LDS_STEPS_A, N_TILES_B)  # quad needing a g2s + b s2r
        SCHED_B_PREFETCH = _build_schedule(N_LDS_STEPS_B, N_TILES_A)  # quad needing b g2s + a s2r

        # G2S-only schedules for c10/c11 (no s2r interleave).
        def _build_g2s_only(n_g2s_steps):
            g2s_at = [[] for _ in range(MFMA_COUNT + 1)]
            for step_i in range(n_g2s_steps):
                slot = (step_i * MFMA_COUNT) // max(n_g2s_steps, 1)
                g2s_at[slot].append(step_i)
            return g2s_at
        SCHED_G2S_ONLY_A = _build_g2s_only(N_LDS_STEPS_A)
        SCHED_G2S_ONLY_B = _build_g2s_only(N_LDS_STEPS_B)

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
        def kernel_dense_nt_v3(
            A: fx.Tensor, B_T: fx.Tensor, C: fx.Tensor,
            A_scale: fx.Tensor, B_scale: fx.Tensor,
            c_m: fx.Int32, c_n: fx.Int32,
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
            pid = fx.block_idx.x
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
            store_c = StoreC(A_scale, B_scale, C, c_m, c_n, mfma.idx, N_TILES_A, N_TILES_B)

            # Pre-compute LDS swizzle offsets per fragment (used inside
            # _interleaved_cluster instead of full s2r.load() batch).
            def _lds_swz(s2r):
                lds_swz = []
                for ti in range_constexpr(s2r.n_tiles):
                    row = s2r.wave_idx * (s2r.n_tiles * 16) + ti * 16 + lane_id % 16
                    sub = []
                    for j in range_constexpr(2):
                        col = (lane_id // 16) * 16 + j * 64
                        r, c = swizzle_128(row, col)
                        sub.append(r * BLOCK_K + c)
                    lds_swz.append(sub)
                return lds_swz

            # No-op (g2s-only schedules built at compile time outside kernel).

            # g2s-only cluster: c10/c11 use this — no s2r interleave (saves
            # rt-carry vgpr; next-iter a0/b0 reloaded at top of next iter).
            def _g2s_only_cluster(lds_dst, g2s, k_offset, a_frag, b_frag, c_frag, g2s_at):
                for mma_i in range_constexpr(MFMA_COUNT):
                    i = mma_i // N_TILES_B
                    j = mma_i % N_TILES_B
                    for step_i in g2s_at[mma_i]:
                        g2s.load_one(lds_dst, k_offset, step_i)
                    c_frag[mfma.idx(i, j)] = mfma.call_one(a_frag, b_frag, c_frag, i, j)
                for step_i in g2s_at[MFMA_COUNT]:
                    g2s.load_one(lds_dst, k_offset, step_i)
                return c_frag

            # _interleaved_cluster: compute one c-quadrant (MFMA_COUNT
            # mfma_call_one) interleaved with prefetch loads + s2r reads
            # for NEXT c-quadrant within this iter (rt = next b1/a1 frags).
            # Used for c00/c01 only. Pair-packs s2r halves on-the-fly to
            # free the i32x4 intermediates.
            def _interleaved_cluster(lds_dst, g2s, k_offset,
                                     s2r, lds_src, a_frag, b_frag, c_frag,
                                     schedule):
                lds_swz = _lds_swz(s2r)
                rt = [None] * s2r.n_tiles
                parts = [None, None]  # only hold one pair at a time
                for mma_i in range_constexpr(MFMA_COUNT):
                    i = mma_i // N_TILES_B
                    j = mma_i % N_TILES_B
                    for spec in schedule[mma_i]:
                        if spec[0] == "g":
                            g2s.load_one(lds_dst, k_offset, spec[1])
                        else:
                            _, ti, sub, _s_idx = spec
                            parts[sub] = s2r.load_one(lds_src, lds_swz[ti][sub])
                            if sub == 1:
                                rt[ti] = pack_i32x4_i32x8(parts[0], parts[1])
                                parts[0] = None
                                parts[1] = None
                    c_frag[mfma.idx(i, j)] = mfma.call_one(a_frag, b_frag, c_frag, i, j)
                return c_frag, rt

            c00 = [mfma.zero_value] * N_ACCUMS
            c01 = [mfma.zero_value] * N_ACCUMS
            c10 = [mfma.zero_value] * N_ACCUMS
            c11 = [mfma.zero_value] * N_ACCUMS

            # Prologue: full 8-slab pre-fill (k=0 + k=1), match fp8_gemm_4wave.
            a_g2s.load(a_cur0, A0_gl_offset + 0 * BLOCK_K)
            b_g2s.load(b_cur0, B0_gl_offset + 0 * BLOCK_K)
            b_g2s.load(b_cur1, B1_gl_offset + 0 * BLOCK_K)
            a_g2s.load(a_cur1, A1_gl_offset + 0 * BLOCK_K)
            a_g2s.load(a_next0, A0_gl_offset + 1 * BLOCK_K)
            b_g2s.load(b_next0, B0_gl_offset + 1 * BLOCK_K)
            b_g2s.load(b_next1, B1_gl_offset + 1 * BLOCK_K)
            a_g2s.load(a_next1, A1_gl_offset + 1 * BLOCK_K)

            # Wait for first stage's loads (a_cur0 + b_cur0 = 2 LDS slabs).
            wait_barrier(3 * N_LDS_STEPS_A + 4 * N_LDS_STEPS_B)
            a0_frag = a_s2r.load(a_cur0)
            wait_barrier(3 * N_LDS_STEPS_A + 3 * N_LDS_STEPS_B)
            b0_frag = b_s2r.load(b_cur0)

            # Mid-iter barrier choice: HW s_barrier (default, conservative)
            # or sched_barrier(0) (compiler-only fence, no HW cost).
            # Verified safe via sched_pos.py: B2/B4/B6 positions can become
            # sched_barrier(0) without breaking SNR (waves access non-overlap
            # LDS slices). End-of-iter barrier MUST stay HW (stage swap).
            def _mid_barrier():
                rocdl.s_setprio(0)
                if sched_mid_barriers:
                    rocdl.sched_barrier(0)
                else:
                    rocdl.s_barrier()
                rocdl.s_setprio(1)

            # hot_loop_scheduler — emit sched_mfma/sched_dsrd/sched_vmem
            # hints describing the desired mfma/load interleaving (HBL-style).
            # Per K-iter: 32 mfma_call_one (4 quadrants × N_ACCUMS) + 4 g2s
            # prefetches × N_LDS_STEPS each (vmem) + s2r reads inside clusters.
            def _hot_loop_sched():
                mfma_per_quadrant = N_ACCUMS  # 8 for BM=BN=256
                # Per-iter totals: 4 quadrants × N_ACCUMS = 32 mfma;
                # 4 prefetches × (N_LDS_STEPS_A or B): 2×2 + 2×2 = 8 vmem (BM=BN=256).
                # s2r reads inside _interleaved_cluster: count via schedule.
                # Strategy: per quadrant, declare its mfma + interleave hints.
                for _q in range_constexpr(4):
                    rocdl.sched_mfma(mfma_per_quadrant)
                    rocdl.sched_dsrd(2)        # ~2 ds_read per quadrant
                    rocdl.sched_vmem(2)        # ~2 buffer_load per quadrant
                rocdl.sched_barrier(0)

            for k in range_constexpr(K_ITERS - 2):
                wait_barrier(nt_vmcnt if nt_vmcnt >= 0 else (2 * N_LDS_STEPS_A + 2 * N_LDS_STEPS_B))  # tightened vmcnt = race fix (v3 analog of 8w end-of-iter)
                rocdl.s_setprio(1)
                c00, b1_frag = _interleaved_cluster(
                    a_cur0, a_g2s, A0_gl_offset + (k + 2) * BLOCK_K,
                    b_s2r, b_cur1, a0_frag, b0_frag, c00,
                    SCHED_A_PREFETCH,
                )
                _mid_barrier()
                c01, a1_frag = _interleaved_cluster(
                    b_cur0, b_g2s, B0_gl_offset + (k + 2) * BLOCK_K,
                    a_s2r, a_cur1, a0_frag, b1_frag, c01,
                    SCHED_B_PREFETCH,
                )
                _mid_barrier()
                c10, a0_frag_n = _interleaved_cluster(
                    b_cur1, b_g2s, B1_gl_offset + (k + 2) * BLOCK_K,
                    a_s2r, a_next0, a1_frag, b0_frag, c10,
                    SCHED_B_PREFETCH,
                )
                _mid_barrier()
                c11, b0_frag_n = _interleaved_cluster(
                    a_cur1, a_g2s, A1_gl_offset + (k + 2) * BLOCK_K,
                    b_s2r, b_next0, a1_frag, b1_frag, c11,
                    SCHED_A_PREFETCH,
                )
                rocdl.s_setprio(0)
                if add_sched_hints:
                    _hot_loop_sched()
                a_cur0, a_next0 = a_next0, a_cur0
                a_cur1, a_next1 = a_next1, a_cur1
                b_cur0, b_next0 = b_next0, b_cur0
                b_cur1, b_next1 = b_next1, b_cur1
                a0_frag = a0_frag_n
                b0_frag = b0_frag_n

            # Tail step k_iters - 2 (mirror fp8_gemm_4wave tail).
            wait_barrier(2 * N_LDS_STEPS_A + 2 * N_LDS_STEPS_B)
            b1_frag = b_s2r.load(b_cur1)
            c00 = mfma.call(a0_frag, b0_frag, c00)
            a1_frag = a_s2r.load(a_cur1)
            c01 = mfma.call(a0_frag, b1_frag, c01)
            wait_barrier(1 * N_LDS_STEPS_A + 1 * N_LDS_STEPS_B)
            a0_frag = a_s2r.load(a_next0)
            c10 = mfma.call(a1_frag, b0_frag, c10)
            b0_frag = b_s2r.load(b_next0)
            c11 = mfma.call(a1_frag, b1_frag, c11)

            a_cur0, a_next0 = a_next0, a_cur0
            a_cur1, a_next1 = a_next1, a_cur1
            b_cur0, b_next0 = b_next0, b_cur0
            b_cur1, b_next1 = b_next1, b_cur1

            # Tail step k_iters - 1.
            wait_barrier(0)
            b1_frag = b_s2r.load(b_cur1)
            a1_frag = a_s2r.load(a_cur1)
            c00 = mfma.call(a0_frag, b0_frag, c00)
            c01 = mfma.call(a0_frag, b1_frag, c01)
            c10 = mfma.call(a1_frag, b0_frag, c10)
            c11 = mfma.call(a1_frag, b1_frag, c11)

            wave_n_offset = wave_n * (N_TILES_B * 16)
            wave_m_offset = wave_m * (N_TILES_A * 16)
            base_row = block_m * BLOCK_M + wave_m_offset
            base_col = block_n * BLOCK_N + wave_n_offset
            store_c.store(c00, base_row + 0,            base_col + 0)
            store_c.store(c01, base_row + 0,            base_col + LDS_BLOCK_N)
            store_c.store(c10, base_row + LDS_BLOCK_M,  base_col + 0)
            store_c.store(c11, base_row + LDS_BLOCK_M,  base_col + LDS_BLOCK_N)

        @flyc.jit
        def launch_dense_nt_v3(
            A: fx.Tensor, B_T: fx.Tensor, C: fx.Tensor,
            A_scale: fx.Tensor, B_scale: fx.Tensor,
            c_m: fx.Int32, c_n: fx.Int32, stream: fx.Stream,
        ):
            grid_x = ceildiv(c_m, BLOCK_M) * ceildiv(c_n, BLOCK_N)
            kernel_dense_nt_v3(
                A, B_T, C, A_scale, B_scale, c_m, c_n,
                value_attrs=_make_value_attrs(waves_per_eu, agpr_alloc, "512,512"),
            ).launch(grid=(grid_x, 1, 1), block=(512, 1, 1), stream=stream)
        return launch_dense_nt_v3


    @functools.lru_cache(maxsize=128)
    def _compile_dense_nt_fly8w(
        K: int,
        BLOCK_M: int = 256,
        BLOCK_N: int = 256,
    ):
        """Vanilla FlyDSL-repo 8-wave kernel (kernels.fp8_gemm_8wave).

        Identical pipeline to our `_compile_dense_nt` minus GROUP_M swizzle
        (uses plain divmod(block_idx, n_blocks) = GM=1 row-major). Kept here
        as an A/B baseline for measuring the GROUP_M / BM-128 wins."""
        from kernels.fp8_gemm_8wave import compile_fp8_gemm_8w  # noqa: F401
        return compile_fp8_gemm_8w(
            K=K, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, b_preshuffled=False,
        )

    # NT 4-wave kernel removed 2026-05-28 (unused in override table; user
    # direction to consolidate on 8-wave + v3 only).


    # ──────────────────────────────────────────────────────────────────────

    def _inttoptr_lds(byte_addr):
        """Convert an integer byte address to !llvm.ptr<3> (LDS pointer).

        Same shape as the mla kernel helper -- ir.Type.parse'd per call;
        the parsed Type belongs to the current MLIRContext, and caching
        across compiles fails verification on the 2nd shape.
        """
        return _llvm.inttoptr(ir.Type.parse("!llvm.ptr<3>"), _raw(fx.Int64(byte_addr)))

    _gep = _buffer_ops.get_element_ptr

    def _lds_ptr_from_i32(addr_i32, byte_offset=0):
        """Build an LDS pointer (ptr<3>) from an i32 byte address + optional static offset."""
        ptr = _inttoptr_lds(ArithValue(addr_i32).extui(T.i64))
        if byte_offset != 0:
            ptr = _gep(ptr, static_byte_offset=byte_offset)
        return ptr

    def _packed_ds_read_tr_offsets(base_ptr, byte_offsets, vmcnt_hint=None):
        """Pack len(byte_offsets) ds_read_b64_tr_b8 sharing ONE base ptr VGPR,
        each at a compile-time immediate byte offset (HK CRR
        load_col_from_st_half pattern: 1 addr + offset:N).

        Why 1 shared ptr (not N independent ptrs): the backend needs 1 addr
        VGPR instead of N, avoiding the spill cascade (2026-05-28: N-distinct-
        ptr packs gave spill 14->222 on TN BM=256; this form gives spill 8).
        For S2RLoaderTr the 4 ds_read of one tile split into 2 pairs (c0,c2)
        and (c1,c3), each pair differing by exactly r_step*8192 (verified:
        _K_BASE 0->64 / 8->72 bumps r_step by 1 while W/K_mod_8/swz_K/j_chunk
        stay identical), so the second read of each pair is base+offset:8192
        with NO runtime address component.

        `=&v` early-clobber + `~{memory}` clobber preserve wave-coop
        ds_read_b64_tr_b8 correctness (single-output asm permutes lanes).
        NOTE: ds_read_b64_tr_b8 is async (completes on lgkmcnt). The opaque asm
        hides this from the backend, so the *caller* (S2RLoaderTr.load) must
        emit a trailing `s_waitcnt lgkmcnt(0)` before the consuming mfma —
        otherwise the mfma races the LDS read (K-dependent garbage/NaN).

        Returns list of v2i32 (one per offset)."""
        N = len(byte_offsets)
        v2i32 = ir.VectorType.get([2], ir.IntegerType.get_signless(32))
        struct_t = _llvm.StructType.get_literal([v2i32] * N)
        lines = []
        if vmcnt_hint is not None and vmcnt_hint >= 0:
            lines.append(f"s_waitcnt vmcnt({vmcnt_hint})")
        for k in range(N):
            # ${N} is the single shared input ptr (after N outputs $0..$N-1).
            lines.append(f"ds_read_b64_tr_b8 ${k}, ${N} offset:{byte_offsets[k]}")
        asm = "\n".join(lines)
        constraints = ",".join(["=&v"] * N + ["v"] + ["~{memory}"])
        asm_op = _llvm.InlineAsmOp(
            res=struct_t,
            operands_=[_raw(base_ptr)],
            asm_string=asm,
            constraints=constraints,
            has_side_effects=True,
        )
        return [_llvm.extractvalue(v2i32, asm_op.result, [k]) for k in range(N)]

    class _G2SLoaderStride(G2SLoader):
        """G2SLoader with a tunable per-wave LDS chunk stride (default 1024 =
        identical to base). A stride of 1056 (=1024+32) makes the per-wave chunk
        base NOT bank-aligned on gfx950: W*1056/4 % 64 = W*8, so the wave-step
        dimension W spreads across banks instead of all collapsing to bank 0.
        This halves the TN transpose-read bank conflict (4-way -> 2-way), since
        the 4 lane-groups (lane//16, i.e. W=0,2,4,6) then hit 4 distinct banks.
        The within-chunk data layout (lane*16 from the copy atom) is unchanged;
        only the chunk base is padded by 32 bytes. Read side (S2RLoaderTr*) must
        use the SAME chunk_stride."""
        def __init__(self, *a, chunk_stride=1024, **kw):
            super().__init__(*a, **kw)
            self.chunk_stride = chunk_stride

        def _lds_dst_at(self, lds_dst, step):
            cs = self.chunk_stride
            step_off = self.wave_id * cs + step * (self.n_waves * cs)
            base_i32 = fx.Int32(fx.ptrtoint(lds_dst.ptr))
            sum_i32 = base_i32 + fx.Int32(step_off)
            lds_ptr = fx.inttoptr(self.LdsPtr_t, sum_i32)
            return fx.make_view(lds_ptr, fx.make_layout(1, 1))


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

    def lgkm_barrier():
        """Drain LDS write queue + WG sync. Use after WG-wide LDS write that a
        subsequent LDS read consumes (e.g., LDS2LDS transpose → s2r read).

        Plain ``rocdl.s_barrier()`` (= ``s_barrier`` alone) syncs waves but does
        NOT drain the wave's own LDS write queue; the writes may be in flight
        when ``s_barrier`` completes, then a sibling wave's ``ds_read_b128``
        reads stale data. The fix is ``s_waitcnt lgkmcnt(0)`` BEFORE the
        barrier so each wave commits its LDS ops before signalling.
        """
        _llvm.inline_asm(
            res=None,
            operands_=[],
            asm_string="s_waitcnt lgkmcnt(0)\ns_barrier",
            constraints="",
            has_side_effects=True,
        )

    def compute_global_offsets_nn_identity(lane_id, wave_id, N_out, n_rounds):
        """Identity (no-swizzle) version of compute_global_swizzle_nn.

        Used by the b_lds_transpose path. The LDS destination byte at offset
        (wave_id*1024 + step*8192 + lane_id*16 + b) directly holds
            B[K = lane_id//8 + wave_id*8 + step*64, N = (lane_id%8)*16 + b].
        Makes the LDS2LDSTransposer source-decode trivial (no XOR un-swizzle
        on the read side, since the read side is K-strided per N column rather
        than the wave-coop ds_read_b64_tr_b8 the original swizzle was tuned for).
        """
        offsets = []
        n_waves = fx.block_dim.x // 64
        for r in range_constexpr(n_rounds):
            k_row = lane_id // 8 + wave_id * 8 + r * (n_waves * 8)
            n_col = (lane_id % 8) * 16
            offsets.append(k_row * N_out + n_col)
        return offsets

    class LDS2LDSTransposer:
        """B LDS_temp (K-major identity, written by G2SLoader with
        compute_global_offsets_nn_identity) → LDS_final (N-major, swizzle_128(N,K))
        in-WG transpose, so the main loop reads via per-lane ds_read_b128
        (NT-style S2RLoader) instead of wave-cooperative ds_read_b64_tr_b8.

        Target shape: ONE B sub-block = BLOCK_K K-rows × LDS_BLOCK_N N-cols
        (BLOCK_K=128, LDS_BLOCK_N=128 → 16384 bytes). 8-wave WG = 512 threads.

        Per K-iter cost (MVP, naive 1-byte gather):
        * 16384 bytes / 512 threads / 16 bytes-per-b128 = 2 b128 writes / thread
        * Each output b128 = gather 16 strided bytes from LDS_temp (16 ds_read_u8)
        * Total per B sub-block: 8192 ds_read_u8 + 512 ds_write_b128

        Source decode for target (K, N):
            step = K // 64
            wave_id_src = (K % 64) // 8
            K_mod_8 = K % 8                           # = lane-high bits 3..5
            lane_low = N // 16                        # = lane-low bits 0..2
            lane_id_src = K_mod_8 * 8 + lane_low
            b_in_chunk = N % 16
            offset = wave_id_src*1024 + step*8192 + lane_id_src*16 + b_in_chunk

        Destination encode for target (N_target, K_chunk_base):
            (r_swz, c_swz) = swizzle_128(N_target, K_chunk_base)
            offset = r_swz * BLOCK_K + c_swz          # b128 chunk start
        """

        # Per K-chunk (16 K bytes), pre-computed source decomposition relative
        # to k_local ∈ [0, 16). Used by transpose() body unroll.
        # For K = K_chunk_base + k_local with K_chunk_base % 16 == 0:
        #   step = K_chunk_base // 64       (constant across k_local)
        #   wave_id_src = ((K_chunk_base % 64) // 8) + (k_local // 8)
        #                                   [chunk_idx_low*2 + k_local//8]
        #   K_mod_8 = k_local % 8
        # Plus N_target gives lane_low + b_in_chunk.

        def __init__(self, lds_block_n, block_k):
            assert lds_block_n == 128, "LDS2LDSTransposer hard-wired to LDS_BLOCK_N=128"
            assert block_k == 128, "LDS2LDSTransposer hard-wired to BLOCK_K=128"
            self.LDS_BLOCK_N = lds_block_n
            self.BLOCK_K = block_k
            # tid ∈ [0, 512). Each thread produces 2 b128 writes (32 bytes) per call.
            self.tid = fx.thread_idx.x

        def _gather_b8(self, src_lds_temp, src_off_i32):
            """Read 1 byte from LDS_temp[src_off_i32] via ds_read_u8 (i8 load)."""
            off_tup = fx.make_int_tuple(src_off_i32)
            ptr_off = fx.add_offset(src_lds_temp.ptr, off_tup)
            i8_iter = fx.recast_iter(fx.Uint8, ptr_off)
            view = fx.make_view(i8_iter, fx.make_layout(1, 1))
            return view.load()

        def _store_b128(self, dst_lds_final, dst_off_i32, vec_b128):
            """Write 16 bytes to LDS_final[dst_off_i32] via ds_write_b128."""
            off_tup = fx.make_int_tuple(dst_off_i32)
            ptr_off = fx.add_offset(dst_lds_final.ptr, off_tup)
            i8_iter = fx.recast_iter(fx.Uint8, ptr_off)
            view = fx.make_view(i8_iter, fx.make_layout(16, 1))
            view.store(vec_b128)

        def _store_b8(self, dst_lds_final, dst_off_i32, byte):
            """Write 1 byte to LDS_final[dst_off_i32] via ds_write_b8."""
            off_tup = fx.make_int_tuple(dst_off_i32)
            ptr_off = fx.add_offset(dst_lds_final.ptr, off_tup)
            i8_iter = fx.recast_iter(fx.Uint8, ptr_off)
            view = fx.make_view(i8_iter, fx.make_layout(1, 1))
            view.store(byte)

        def transpose(self, src_lds_temp, dst_lds_final):
            """One B sub-block (BLOCK_K × LDS_BLOCK_N = 128×128 = 16 KiB) transpose.

            Per-thread mapping (linear, iter ∈ {0,1}):
                linear = tid + iter * 512   (∈ [0, 1024) covering 1024 b128 slots)
                n_target  = linear % 128          (N row in LDS_final)
                k_chunk   = linear // 128         (K-chunk idx ∈ [0, 8))
                k_base    = k_chunk * 16
            """
            for iter in range_constexpr(2):
                linear = self.tid + iter * 512
                n_target = linear % 128
                k_chunk = linear // 128
                k_base = k_chunk * 16

                # Source-side per-k_local constants (depend on k_chunk only).
                # Captured outside the inner k_local loop to amortize indexing.
                step = k_chunk // 4                  # ∈ {0, 1}
                chunk_mod_4 = k_chunk % 4            # ∈ [0, 4)
                # Note: wave_id_src = chunk_mod_4 * 2 + k_local // 8 (varies w/ k_local)

                lane_low = n_target // 16
                b_in_chunk = n_target % 16

                # Destination base: swizzle_128(n_target, k_base) gives a 16-byte
                # chunk start. We stream 16 bytes one-by-one inside a RUNTIME
                # loop (plain `range`, not `range_constexpr`) so the AMDGPU
                # backend cannot fold 16 ds_write_b8 back into 1 ds_write_b128
                # (which it does when the loop is unrolled, defeating the
                # whole point of the byte-by-byte streaming).
                #
                # With the runtime loop, peak transposer-live VGPRs = 1 byte
                # per thread (vs 16-17 bytes in the vec16-accumulation
                # variant), which is the actual register-pressure win.
                rs, cs = swizzle_128(n_target, k_base)
                dst_base = rs * 128 + cs
                for k_local in range(16):
                    K_mod_8 = k_local % 8
                    wave_id_src = chunk_mod_4 * 2 + k_local // 8
                    lane_id_src = K_mod_8 * 8 + lane_low
                    src_off = (
                        wave_id_src * 1024
                        + step * 8192
                        + lane_id_src * 16
                        + b_in_chunk
                    )
                    byte = self._gather_b8(src_lds_temp, src_off)
                    self._store_b8(dst_lds_final, dst_base + k_local, byte)

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
        # NOTE: MLIR types are context-scoped; caching TR_TYPE as a class
        # attribute fails on the 2nd compile (different K) because the cached
        # i32 type belongs to the prior MLIR context and Numeric.from_ir_type
        # can't look it up in the new context's map. Always rebuild per-call.

        def __init__(self, wave_n, n_tiles_b, inline_asm=False, vmcnt_hint=2,
                     chunk_stride=1024):
            """``inline_asm=True`` switches ds_read_tr8_b64 from rocdl
            intrinsic to inline-asm block. backend SIInsertWaitcnts treats
            inline asm as opaque (not known wave-coop op) so does NOT
            auto-emit `vmcnt(0)` drain before each tr8 group. Caller-supplied
            `vmcnt_hint` is emitted once per tile_i (before first of 4 ds_read)
            to ensure LDS data sync. This eliminates 446/447 vmcnt(0) sites
            per K=28672 kernel, gaining ~+22% TFLOPS on big K-shapes.

            ``vmcnt_hint=2`` is the validated safe value (1000-run det check
            passed on 5 critical shapes incl. (8192,8192,28672)). vmcnt(6)
            shows occasional race on K=28672. Larger vmcnt = looser sync =
            faster but unsafe; smaller = stricter = closer to baseline drain.

            ⚠ When inline_asm=True, the kernel MUST set agpr_alloc>0 (any
            nonzero pin); compiler-decide AGPR (=0) conflicts with inline
            asm `=v` constraint and produces nan output. See path J debug.
            """
            self.wave_n = wave_n
            self.n_tiles_b = n_tiles_b
            self.lane_id = fx.thread_idx.x % 64
            self.inline_asm = inline_asm
            self.vmcnt_hint = vmcnt_hint
            # Per-wave LDS chunk stride (bank-spread; must match G2S writer).
            # round_stride = n_waves(8) * chunk_stride is the r_step K-sub-round
            # jump (1024->8192 default; 1056->8448 bank-spread).
            self.chunk_stride = chunk_stride
            self.round_stride = 8 * chunk_stride

        def load(self, lds_src, preshuffled=False):
            """Returns list[N_TILES_B] of i32x8 (32 fp8/lane K-contig at N-col).
            inline_asm path emits a single trailing separate lgkmcnt(0) drain;
            correct because mma.call follows at a function-call boundary the
            backend doesn't reorder across."""
            assert not preshuffled, "S2RLoaderTr does not support preshuffled"
            tr_type = Vec.make_type(2, fx.Int32)

            base_i32 = fx.Int32(fx.ptrtoint(lds_src.ptr))
            I = self.lane_id // 16
            L_in_sg = self.lane_id % 16

            # 1-ptr + offset:8192 packing (mirror HK CRR load_col_from_st_half).
            # The 4 ds_read of a tile pair as (c0,c2) and (c1,c3): within each
            # pair the only formula difference is _K_BASE 0->64 (or 8->72),
            # which bumps r_step by exactly 1 (=8192 bytes) and leaves every
            # other term identical. So the 2nd read of each pair is just
            # base+offset:8192 — no runtime address, single shared input ptr.
            def _ptr_off_b(c, tile_i):
                K_log = I * 16 + S2RLoaderTr._K_BASE[c] + (L_in_sg // 2)
                r_step = K_log // 64
                W = (K_log % 64) // 8
                K_mod_8 = K_log % 8
                swz_K = ((K_log % 16) // 2) * 16
                tile_N_start = self.wave_n * 32 + tile_i * 16
                j_chunk = (tile_N_start // 16) ^ (swz_K // 16)
                return (W * self.chunk_stride + r_step * self.round_stride
                        + K_mod_8 * 128 + j_chunk * 16 + (L_in_sg % 2) * 8)

            RS = self.round_stride  # c0->c2 / c1->c3 packed-read jump (r_step+1)
            frag = []
            for tile_i in range_constexpr(self.n_tiles_b):
                if self.inline_asm:
                    p0 = _lds_ptr_from_i32(base_i32 + fx.Int32(_ptr_off_b(0, tile_i)))
                    p1 = _lds_ptr_from_i32(base_i32 + fx.Int32(_ptr_off_b(1, tile_i)))
                    r02 = _packed_ds_read_tr_offsets(p0, [0, RS], vmcnt_hint=self.vmcnt_hint)
                    r13 = _packed_ds_read_tr_offsets(p1, [0, RS], vmcnt_hint=None)
                    # r02 = [c0, c2], r13 = [c1, c3] → reorder to [c0,c1,c2,c3]
                    calls = [Vec(r02[0]), Vec(r13[0]), Vec(r02[1]), Vec(r13[1])]
                else:
                    calls = [Vec(rocdl.ds_read_tr8_b64(
                        tr_type, _lds_ptr_from_i32(base_i32 + fx.Int32(_ptr_off_b(c, tile_i)))
                    ).result) for c in range_constexpr(4)]

                # Concat 4 × v2i32 → v8i32. Byte order matches mfma B-operand
                # bytes 0..31 for lane L:
                #   v8i32[0..1] = call 0 → K = (L//16)*16 + L%2*4 + ...   (bytes 0..7)
                #   v8i32[2..3] = call 1 → bytes 8..15
                #   v8i32[4..5] = call 2 → bytes 16..23 (K + 64 jump)
                #   v8i32[6..7] = call 3 → bytes 24..31
                v4_lo = calls[0].shuffle(calls[1], [0, 1, 2, 3])
                v4_hi = calls[2].shuffle(calls[3], [0, 1, 2, 3])
                frag.append(v4_lo.shuffle(v4_hi, list(range(8))))
            if self.inline_asm:
                # ds_read_b64_tr_b8 completes on lgkmcnt (async LDS read); the
                # opaque asm hides this so the backend won't auto-insert the
                # wait the mfma needs. Single trailing drain (mma.call follows
                # at a call boundary the backend doesn't reorder across).
                _llvm.inline_asm(res=None, operands_=[], asm_string="s_waitcnt lgkmcnt(0)",
                                  constraints="", has_side_effects=True)
            return frag

    class S2RLoaderTr_A:
        """LDS → mfma A-operand wave-coop tr8 load (for TN-layout fp8 GEMM).
        Per HK fp8 CRR (kernel_fp8_layouts.cpp `crr_mma`), mfma A and B
        operand register byte layout is IDENTICAL — both call same
        v_mfma_f32_16x16x128_f8f6f4. HK does
        `reinterpret_cast<A_row_reg>(a_col_reg)` to use B-style transpose
        fragment as A-op. We replicate via per-wave_m S2RLoaderTr variant.
        """

        _K_BASE = (0, 8, 64, 72)

        def __init__(self, wave_m, n_tiles_a, lds_block_m, inline_asm=False, vmcnt_hint=2,
                     chunk_stride=1024, n_waves=8, wave_stride=None):
            """``lds_block_m`` = BLOCK_M // 2 (LDS half-block M-size). Per-wave
            M coverage = lds_block_m / num_wave_m. For 8w with num_wave_m=2:
            BM=256 → lds_block_m=128 → stride=64; BM=128 → lds_block_m=64 → stride=32.
            ``n_waves`` = WG wave count (8 default; 16 for the 1024-thread layout
            where one K-round fills 128 K = 16 waves × 8). ``wave_stride`` overrides
            the M sub-half stride (16w: each wave owns 64 rows → stride=64)."""
            self.wave_m = wave_m
            self.n_tiles_a = n_tiles_a
            self.wave_stride = (lds_block_m // 2) if wave_stride is None else wave_stride
            self.lane_id = fx.thread_idx.x % 64
            self.inline_asm = inline_asm
            self.vmcnt_hint = vmcnt_hint
            self.chunk_stride = chunk_stride       # bank-spread LDS chunk stride
            self.n_waves = n_waves
            self.round_stride = n_waves * chunk_stride   # r_step K-sub-round jump

        def _ptr_off_a(self, c, tile_i, I, L_in_sg):
            K_log = I * 16 + S2RLoaderTr_A._K_BASE[c] + (L_in_sg // 2)
            KW = self.n_waves * 8
            r_step = K_log // KW
            W = (K_log % KW) // 8
            K_mod_8 = K_log % 8
            swz_K = ((K_log % 16) // 2) * 16
            # 8w A: wave_m * wave_stride (= LDS_BLOCK_M / num_wave_m)
            tile_M_start = self.wave_m * self.wave_stride + tile_i * 16
            j_chunk = (tile_M_start // 16) ^ (swz_K // 16)
            return (W * self.chunk_stride + r_step * self.round_stride
                    + K_mod_8 * 128 + j_chunk * 16 + (L_in_sg % 2) * 8)

        # Each inline ds_read_b64_tr_b8 tile issues 4 LDS reads.
        N_LDS_PER_TILE = 4

        def _issue_one(self, lds_src, tile_i):
            """Issue the 4 ds_read_b64_tr_b8 of one A tile WITHOUT draining
            lgkmcnt or assembling the frag. Returns the 4 raw v2i32 Vec. Split
            from assemble so the caller can software-pipeline: issue tile i+1's
            reads, then drain a GRADED lgkmcnt before consuming tile i — so
            tile i+1's LDS reads overlap tile i's mma (mirrors what the backend
            does for the intrinsic path via lgkmcnt(4/8/12), instead of the
            per-tile lgkmcnt(0) full-drain that serialised the inline path)."""
            tr_type = Vec.make_type(2, fx.Int32)
            base_i32 = fx.Int32(fx.ptrtoint(lds_src.ptr))
            I = self.lane_id // 16
            L_in_sg = self.lane_id % 16
            RS = self.round_stride
            if self.inline_asm:
                p0 = _lds_ptr_from_i32(base_i32 + fx.Int32(self._ptr_off_a(0, tile_i, I, L_in_sg)))
                p1 = _lds_ptr_from_i32(base_i32 + fx.Int32(self._ptr_off_a(1, tile_i, I, L_in_sg)))
                r02 = _packed_ds_read_tr_offsets(p0, [0, RS], vmcnt_hint=self.vmcnt_hint)
                r13 = _packed_ds_read_tr_offsets(p1, [0, RS], vmcnt_hint=None)
                return [Vec(r02[0]), Vec(r13[0]), Vec(r02[1]), Vec(r13[1])]
            return [Vec(rocdl.ds_read_tr8_b64(
                tr_type, _lds_ptr_from_i32(base_i32 + fx.Int32(self._ptr_off_a(c, tile_i, I, L_in_sg)))
            ).result) for c in range_constexpr(4)]

        @staticmethod
        def _assemble(calls):
            v4_lo = calls[0].shuffle(calls[1], [0, 1, 2, 3])
            v4_hi = calls[2].shuffle(calls[3], [0, 1, 2, 3])
            return v4_lo.shuffle(v4_hi, list(range(8)))

        @staticmethod
        def _wait_lgkmcnt(n):
            _llvm.inline_asm(res=None, operands_=[], asm_string=f"s_waitcnt lgkmcnt({n})",
                              constraints="", has_side_effects=True)

        def load_one(self, lds_src, tile_i):
            """Single-A-tile wave-coop tr8 load → one i32x8 frag (caps peak
            A-fragment liveness at 1 tile for the interleaved BM=256 inline
            path; see main-loop comment)."""
            calls = self._issue_one(lds_src, tile_i)
            frag = self._assemble(calls)
            if self.inline_asm:
                self._wait_lgkmcnt(0)
            return frag

        def load_pipelined(self, lds_src, consume):
            """Software-pipelined (depth-2) load of all n_tiles_a tiles for the
            inline path. ``consume(tile_i, frag)`` is called once per tile with
            its assembled frag, in order; the caller emits the mma there. Caps
            raw-read liveness at 2 tiles while letting tile i+1's ds_read overlap
            tile i's mma via graded lgkmcnt. Falls back to per-tile drain for the
            intrinsic path (which schedules its own waits)."""
            nt = self.n_tiles_a
            if not self.inline_asm:
                for i in range_constexpr(nt):
                    consume(i, self._assemble(self._issue_one(lds_src, i)))
                return
            raw = self._issue_one(lds_src, 0)
            for i in range_constexpr(nt):
                if i + 1 < nt:
                    raw_next = self._issue_one(lds_src, i + 1)
                    # tile i+1 just issued: 8 LDS in flight; wait until only
                    # tile i+1's 4 remain (tile i retired, LDS completes in order).
                    self._wait_lgkmcnt(self.N_LDS_PER_TILE)
                else:
                    self._wait_lgkmcnt(0)
                consume(i, self._assemble(raw))
                if i + 1 < nt:
                    raw = raw_next

        def load(self, lds_src, preshuffled=False):
            assert not preshuffled
            return [self.load_one(lds_src, tile_i) for tile_i in range_constexpr(self.n_tiles_a)]

    class S2RLoaderTr_4w:
        """4-wave variant of S2RLoaderTr. Geometry differs from 8w version:
        * n_waves = 4 (vs 8) → LDS step jump = 4*1024 = 4096 bytes (vs 8192)
        * step indexes K by n_waves*8 = 32 (vs 64) → K_log decoder uses // 32
        * wave grid is 2×2: wave_j ∈ [0, 2), N_TILES_B = 4 (vs 8w wave_n ∈
          [0, 4), N_TILES_B=2)
        * Per-wave N coverage: wave_j * 64 (vs wave_n * 32)

        LDS layout (from G2SLoader with n_waves=4, N_LDS_STEPS=N_TILES_B):
            LDS[W*1024 + step*4096 + L*16 + n_byte]
              = B[K = L//8 + W*8 + step*32, N = (L%8)*16 XOR swz_K + n_byte]
        where W ∈ [0, 4), step ∈ [0, N_LDS_STEPS), L ∈ [0, 64), n_byte ∈ [0, 16).

        Per-lane ptr formula for byte (K_log, N_log) target:
            r_step = K_log // 32
            W      = (K_log % 32) // 8           ∈ [0, 4)
            K_mod_8 = K_log % 8
            swz_K  = ((K_log % 16) // 2) * 16
            tile_N_start = wave_j * 64 + tile_i * 16
            j_chunk = (tile_N_start // 16) ^ (swz_K // 16)
            ptr = LDS + W*1024 + r_step*4096 + K_mod_8*128
                      + j_chunk*16 + (L_in_sg%2)*8
        """

        _K_BASE = (0, 8, 64, 72)

        def __init__(self, wave_j, n_tiles_b):
            self.wave_j = wave_j
            self.n_tiles_b = n_tiles_b
            self.lane_id = fx.thread_idx.x % 64

        def load_one(self, lds_src, tile_i):
            """Single-tile load: 4 ds_read_tr8_b64 + shuffle into one 32xi8 frag.
            Designed for fine-grain interleave: caller weaves load_one calls
            between mfma issues so the dswr/dsread pipeline overlaps mfma
            issue latency."""
            tr_type = Vec.make_type(2, fx.Int32)
            base_i32 = fx.Int32(fx.ptrtoint(lds_src.ptr))
            I = self.lane_id // 16
            L_in_sg = self.lane_id % 16
            calls = []
            for c in range_constexpr(4):
                K_log = I * 16 + S2RLoaderTr_4w._K_BASE[c] + (L_in_sg // 2)
                r_step = K_log // 32            # 4w: step jump is 32 (not 64)
                W = (K_log % 32) // 8           # 4w: W ∈ [0, 4) (not [0, 8))
                K_mod_8 = K_log % 8
                swz_K = ((K_log % 16) // 2) * 16
                tile_N_start = self.wave_j * 64 + tile_i * 16   # 4w: wave_j * 64
                j_chunk = (tile_N_start // 16) ^ (swz_K // 16)
                ptr_offset = (
                    W * 1024
                    + r_step * 4096           # 4w: step jump 4096 (not 8192)
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
            return v4_lo.shuffle(v4_hi, list(range(8)))

        def load(self, lds_src, preshuffled=False):
            assert not preshuffled, "S2RLoaderTr_4w does not support preshuffled"
            frag = []
            for tile_i in range_constexpr(self.n_tiles_b):
                frag.append(self.load_one(lds_src, tile_i))
            return frag

    @functools.lru_cache(maxsize=128)
    def _compile_dense_nn(
        K: int,
        BLOCK_M: int = 256,
        BLOCK_N: int = 256,
        GROUP_M: int = 4,
        waves_per_eu: int = 2,
        agpr_alloc: int = 0,
        b_lds_transpose: bool = False,
        # 2026-05-28 barrier-mask probe (kept for future tuning): NN main
        # loop emits 7 unconditional ``rocdl.s_barrier()`` per K-iter. Bit
        # position 0..6 controls B1..B7. Default 0x7F = all 7 ON.
        # Empirical: removing any single barrier on ffn_down K=28672 slows
        # latency (B5 cheapest at -0.7%, B7 worst at +12%). Barriers stay on.
        barrier_mask: int = 0x7F,
        # 2026-05-28 path J: replace ds_read_tr8_b64 rocdl intrinsic with
        # inline asm so backend SIInsertWaitcnts treats it as opaque and
        # does NOT auto-emit vmcnt(0) drain. User-supplied vmcnt_hint
        # (validated value 2) gives correct LDS sync without race.
        # Validated +14pp NN/NT geomean on 24-shape Llama set
        # (worst K=28672 NN reaches 1.04x NT). MUST set agpr_alloc>0 when
        # enabling (AGPR=0 + inline asm =v constraint → nan output).
        b_inline_asm_load: bool = False,
        vmcnt_hint: int = 2,
    ):
        """NN-layout fp8 dense kernel. A [M, K], B [K, N], C [M, N].

        ``agpr_alloc`` / ``waves_per_eu`` mirror the NT kernel's knobs; see
        ``_make_value_attrs`` for ``agpr_alloc`` encoding (N>0 = exact N AGPRs,
        -N = up to N, 0 = compiler default).

        ``b_lds_transpose`` (default False, legacy native NN path; see
        NN_G2S_TRANSPOSE_DESIGN.md): if True, switches B to a 2-pass LDS
        scheme so the main loop reads via per-lane ``ds_read_b128``
        (NT-style ``S2RLoader``) instead of wave-cooperative
        ``ds_read_b64_tr_b8`` (``S2RLoaderTr``). The path forks to
        ``_compile_dense_nn_btr`` which inserts ``LDS2LDSTransposer`` between
        the G2S writes and the s2r reads. LDS budget = 160 KB exactly
        (gfx950 cap)."""
        if b_inline_asm_load and agpr_alloc == 0:
            raise ValueError(
                "b_inline_asm_load=True requires agpr_alloc > 0. AGPR=0 "
                "(compiler-decide) combined with inline-asm `=v` constraint "
                "produces nan output. Pin AGPR to 16/32/48 (32 is the "
                "validated default)."
            )
        if b_lds_transpose:
            return _compile_dense_nn_btr(
                K=K, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, GROUP_M=GROUP_M,
                waves_per_eu=waves_per_eu, agpr_alloc=agpr_alloc,
            )
        BLOCK_K = 128
        assert BLOCK_M >= 128 and BLOCK_N >= 256 and BLOCK_M % 128 == 0 and BLOCK_N % 256 == 0
        assert K % BLOCK_K == 0

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
            # Triton-style super-block swizzle for L2 reuse. Tail handling
            # identical to NT — group_size_m = min(remaining_m, GROUP_M) so any
            # GROUP_M is correct regardless of num_pid_m % GROUP_M. Without this
            # tail clamp, GM > num_pid_m emits block_m values past the row tile
            # bound, leaving most valid (m, n) tiles uncovered (SNR ≈ -50 dB).
            num_pid_m = ceildiv(c_m, BLOCK_M)
            pid = fx.block_idx.x
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
            if os.environ.get("TN_ASM_MMA", "0") in ("1", "2"):
                mfma._do_mma = _asm_mma_do

            a_g2s = G2SLoader(a_div, gl_off_a, N_LDS_STEPS_A, F8_IR_t, wave_id)
            b_g2s = G2SLoader(b_div, gl_off_b, N_LDS_STEPS_B, F8_IR_t, wave_id)
            a_s2r = S2RLoader(wave_m, N_TILES_A)
            b_s2r = S2RLoaderTr(wave_n, N_TILES_B,
                                 inline_asm=b_inline_asm_load,
                                 vmcnt_hint=vmcnt_hint)
            store_c = StoreC(A_scale, B_scale, C, c_m, c_n, mfma.idx, N_TILES_A, N_TILES_B)

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

            # Main loop. NT-style optional barrier mask: 7 barrier positions
            # B1..B7. ``barrier_mask`` bit N → emit rocdl.s_barrier at position
            # N; 0 = drop. Probe-tuned per shape via SNR ≥ 20 dB gate. Default
            # 0x7F = all 7 ON (the conservative original schedule).
            BR_B1 = bool(barrier_mask & 0x01)
            BR_B2 = bool(barrier_mask & 0x02)
            BR_B3 = bool(barrier_mask & 0x04)
            BR_B4 = bool(barrier_mask & 0x08)
            BR_B5 = bool(barrier_mask & 0x10)
            BR_B6 = bool(barrier_mask & 0x20)
            BR_B7 = bool(barrier_mask & 0x40)

            for k in range_constexpr(K_ITERS - 2):
                b0_frag = b_s2r.load(b_cur0)
                a0_frag = a_s2r.load(a_cur0)
                a_g2s.load(a_next1, A1_gl_offset + (k + 1) * BLOCK_K)
                if BR_B1: rocdl.s_barrier()

                rocdl.s_setprio(1)
                c00_frag = mfma.call(a0_frag, b0_frag, c00_frag)
                rocdl.s_setprio(0)
                if BR_B2: rocdl.s_barrier()

                b1_frag = b_s2r.load(b_cur1)
                b_g2s.load(b_cur0, B0_gl_offset + (k + 2) * BLOCK_K * c_n)
                if BR_B3: rocdl.s_barrier()

                rocdl.s_setprio(1)
                c01_frag = mfma.call(a0_frag, b1_frag, c01_frag)
                rocdl.s_setprio(0)
                if BR_B4: rocdl.s_barrier()

                a1_frag = a_s2r.load(a_cur1)
                a_g2s.load(a_cur0, A0_gl_offset + (k + 2) * BLOCK_K)
                if BR_B5: rocdl.s_barrier()

                rocdl.s_setprio(1)
                c10_frag = mfma.call(a1_frag, b0_frag, c10_frag)
                rocdl.s_setprio(0)
                if BR_B6: rocdl.s_barrier()

                b_g2s.load(b_cur1, B1_gl_offset + (k + 2) * BLOCK_K * c_n)
                wait_barrier(2 * N_LDS_STEPS_A + N_LDS_STEPS_B)

                rocdl.s_setprio(1)
                c11_frag = mfma.call(a1_frag, b1_frag, c11_frag)
                rocdl.s_setprio(0)
                if BR_B7: rocdl.s_barrier()

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
                value_attrs=_make_value_attrs(waves_per_eu, agpr_alloc, "512,512"),
            ).launch(grid=(grid_x, 1, 1), block=(512, 1, 1), stream=stream)

        return launch_dense_nn


    @functools.lru_cache(maxsize=128)
    def _compile_dense_tn(
        K: int,
        BLOCK_M: int = 256,
        BLOCK_N: int = 256,
        GROUP_M: int = 4,
        waves_per_eu: int = 2,
        agpr_alloc: int = 32,
        barrier_mask: int = 0x7F,
        inline_asm_load: bool = True,   # backward compat: applies to both A and B
        a_inline_asm: int = -1,         # -1 = use inline_asm_load; else 0/1 override
        b_inline_asm: int = -1,         # ditto
        vmcnt_hint: int = 2,
        asm_mma: int = 0,               # 0 = env TN_ASM_MMA; 1/2 = AGPR-asm MFMA (2=AGPR accum)
        group_n: int = 0,               # 0 = 1D GROUP_M swizzle; >0 = 2D band (width group_n)
    ):
        """TN-layout fp8 dense kernel: A [K, M], B [K, N], C [M, N] = A^T @ B.
        Both A and B are K-row strided → wave-coop ds_read_b64_tr_b8 on both
        sides (= NN B path × 2). Per HK CRR insight, mfma A/B operand byte
        layout identical, so S2RLoaderTr_A (mirror of S2RLoaderTr w/ wave_m
        geometry) can feed A operand directly. Path J inline-asm on both."""
        # Resolve per-side inline_asm flags (default to inline_asm_load).
        _a_inline = bool(inline_asm_load) if a_inline_asm < 0 else bool(a_inline_asm)
        _use_interleave = bool(int(os.environ.get("TN_INTERLEAVE", "0")))
        _b_inline = bool(inline_asm_load) if b_inline_asm < 0 else bool(b_inline_asm)
        # Single-buffer LDS: drop the double-buffer `next` slots, halving LDS
        # (132KB→66KB) so 2 workgroups fit per CU (occ 25%→50%). Profiling
        # (MfmaUtil 57%, MemUnitStalled≈0, OccupancyPercent 24%) shows 2515 is
        # occupancy-limited by LDS capacity, NOT mem/MFMA-throughput-bound — the
        # 2nd resident block hides the LDS-read latency the lone block can't.
        # load→barrier→compute→barrier, same buffer reused each K-iter.
        _use_single = bool(int(os.environ.get("TN_SINGLE_BUF", "0")))
        # HK-CRR schedule port: 4 barriers/iter (vs 7) + graded waitcnt, both-B
        # upfront. The 2515→2734 gap is barrier-skew cost; HK's 4-barrier CRR
        # schedule is det=0 by design (load layout matched to barriers), unlike
        # naively dropping barriers from the 7-barrier else loop (leaves WAR
        # gaps → race). Reuses prologue/epilogue/store (same 4 G2S per iter).
        _use_crr = bool(int(os.environ.get("TN_CRR", "0")))
        _CRR_VMCNT = int(os.environ.get("TN_CRR_VMCNT", "-1"))
        _CRR_LGKM = int(os.environ.get("TN_CRR_LGKM", "3"))
        # asm_mma resolved from param (production) or env (probing).
        _asm_mma_mode = str(asm_mma) if asm_mma in (1, 2) else os.environ.get("TN_ASM_MMA", "0")
        _asm_mma = _asm_mma_mode in ("1", "2")
        # asm-inplace (=a,v,v,0; D aliases C in AGPR): move 128 accum/lane OUT of
        # VGPR into AGPR. Eliminates the accvgpr-copy + spill (vspill 18→0) on
        # the asm-MMA path, so the steady loop drops the scratch traffic that
        # was padding the MFMA dependency chain. det0 (no overlap race; AGPR
        # accum disjoint from mfma-src). big-K 2747→2809 (+2.3%), big-N marginal
        # (2D-band already optimal). gated TN_INPLACE; needs an AGPR budget.
        _inplace = bool(int(os.environ.get("TN_INPLACE", "1"))) and _asm_mma
        if _inplace and agpr_alloc == 0:
            agpr_alloc = 128
        # asm-mma mode 1 pins accumulators to VGPR (`v`) so it REQUIRES
        # agpr_alloc==0; mode 2 uses an explicit `a` (AGPR) constraint. Either
        # way bypass the intrinsic-path inline-asm-load agpr guard.
        if (_a_inline or _b_inline) and agpr_alloc == 0 and not _asm_mma:
            raise ValueError(
                "inline_asm requires agpr_alloc > 0 (=v constraint conflict)"
            )
        BLOCK_K = 128
        assert BLOCK_M >= 128 and BLOCK_N >= 256 and BLOCK_M % 128 == 0 and BLOCK_N % 256 == 0
        assert K % BLOCK_K == 0

        K_ITERS = K // BLOCK_K
        assert K_ITERS >= 2

        N_TILES_A = BLOCK_M // 64
        N_TILES_B = BLOCK_N // 128
        N_ACCUMS = N_TILES_A * N_TILES_B
        LDS_BLOCK_M = BLOCK_M // 2
        LDS_BLOCK_N = BLOCK_N // 2
        # TN uses wave-coop tr8 for A path; S2RLoaderTr_A formula reads
        # K_log ∈ [0, 128) requiring 2 G2S rounds = 16K LDS slot.
        # For BM=128 (natural N_LDS_STEPS_A=1, 8K slot), force to 2 rounds
        # and 16K slot to match S2RLoaderTr_A K=128 expectation.
        N_LDS_STEPS_A = max(LDS_BLOCK_M // 64, 2)  # ≥ 2 for tr8 K=128
        N_LDS_STEPS_B = LDS_BLOCK_N // 64
        N_LDS_ROUNDS = max(N_LDS_STEPS_A, N_LDS_STEPS_B)
        # Bank-spread LDS chunk stride (gfx950 bank-conflict fix). 1024 = off
        # (bank-0-aligned per-wave chunk → transpose-read bank conflict). 1056
        # = +32B pad → W*1056/4%64 = W*8 spreads the wave-step dim across banks.
        # PMC: SQ_LDS_BANK_CONFLICT/IDX_ACTIVE 0.50 → 0.00 (fully eliminated);
        # 24-shape TN/NT geomean 0.811 → 0.861 (+5.0pp). Default ON; env can
        # override. Both G2S writer (_G2SLoaderStride) and S2R reader use it.
        _LDS_CS = int(os.environ.get("FLYDSL_TN_CHUNK_STRIDE", "1056"))
        # a_lds_size: N rounds × 8 waves × chunk_stride. Pad to stride.
        a_lds_size = max(LDS_BLOCK_M * BLOCK_K, 2 * 8 * 1024) // 1024 * _LDS_CS
        b_lds_size = (LDS_BLOCK_N * BLOCK_K) // 1024 * _LDS_CS

        if _use_single:
            @fx.struct
            class SharedStorage:
                A_lds_cur_0: fx.Array[fx.Float8E4M3FN, a_lds_size, 16]
                A_lds_cur_1: fx.Array[fx.Float8E4M3FN, a_lds_size, 16]
                B_lds_cur_0: fx.Array[fx.Float8E4M3FN, b_lds_size, 16]
                B_lds_cur_1: fx.Array[fx.Float8E4M3FN, b_lds_size, 16]
        else:
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
            """Tile-id -> (block_m, block_n). Plain Python so the GN>0 branch is
            resolved at trace time (not as a kernel `if`). GN==0: 1D GROUP_M
            super-row swizzle (block_m inner, all-N sweep). GN>0: 2D band — N
            split into width-GN bands, GROUP_M 1D inside each band. Both dims
            blocked → A[block_m] reused GN×, B[block_n] reused GROUP_M× so the
            working set (GROUP_M·A_slab + GN·B_slab) stays L2-resident, cutting
            the big-N B re-stream. Always a bijection (full bands take
            num_pid_m·GN pids; remainder = narrower last band). det-neutral."""
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
            A: fx.Tensor, B: fx.Tensor, C: fx.Tensor,
            A_scale: fx.Tensor, B_scale: fx.Tensor,
            c_m: fx.Int32, c_n: fx.Int32,
        ):
            _ = str(fx.thread_idx.x)
            F8_IR_t = fx.Float8E4M3FN.ir_type
            n_blocks = ceildiv(c_n, BLOCK_N)
            lds = fx.SharedAllocator().allocate(SharedStorage).peek()
            a_cur0 = lds.A_lds_cur_0; a_cur1 = lds.A_lds_cur_1
            b_cur0 = lds.B_lds_cur_0; b_cur1 = lds.B_lds_cur_1
            # Ternary (NOT an `if` stmt) so handles are always bound in the kernel
            # scope: @flyc.kernel wraps each `if` branch in its own function, so a
            # var defined inside one `if` is invisible to a sibling `if`. Python
            # short-circuit avoids touching the missing next-slot fields when single.
            a_next0 = a_cur0 if _use_single else lds.A_lds_next_0
            a_next1 = a_cur1 if _use_single else lds.A_lds_next_1
            b_next0 = b_cur0 if _use_single else lds.B_lds_next_0
            b_next1 = b_cur1 if _use_single else lds.B_lds_next_1

            lane_id = fx.thread_idx.x % 64
            wave_id = fx.thread_idx.x // 64
            wave_m = wave_id // 4
            wave_n = wave_id % 4

            num_pid_m = ceildiv(c_m, BLOCK_M)
            pid = fx.block_idx.x
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
                mfma._do_mma = (lambda _a, _b, _c, _m=_mm: _asm_mma_do(_a, _b, _c, mode=_m))

            a_g2s = _G2SLoaderStride(a_div, gl_off_a, N_LDS_STEPS_A, F8_IR_t, wave_id,
                                      chunk_stride=_LDS_CS)
            b_g2s = _G2SLoaderStride(b_div, gl_off_b, N_LDS_STEPS_B, F8_IR_t, wave_id,
                                      chunk_stride=_LDS_CS)
            a_s2r = S2RLoaderTr_A(wave_m, N_TILES_A, LDS_BLOCK_M,
                                   inline_asm=_a_inline, vmcnt_hint=vmcnt_hint,
                                   chunk_stride=_LDS_CS)
            b_s2r = S2RLoaderTr(wave_n, N_TILES_B,
                                 inline_asm=_b_inline, vmcnt_hint=vmcnt_hint,
                                 chunk_stride=_LDS_CS)
            store_c = StoreC(A_scale, B_scale, C, c_m, c_n, mfma.idx, N_TILES_A, N_TILES_B)

            c00_frag = [mfma.zero_value] * N_ACCUMS
            c01_frag = [mfma.zero_value] * N_ACCUMS
            c10_frag = [mfma.zero_value] * N_ACCUMS
            c11_frag = [mfma.zero_value] * N_ACCUMS

            # Prelude.
            b_g2s.load(b_cur0, B0_gl_offset + 0 * BLOCK_K * c_n)
            a_g2s.load(a_cur0, A0_gl_offset + 0 * BLOCK_K * c_m)
            b_g2s.load(b_cur1, B1_gl_offset + 0 * BLOCK_K * c_n)
            a_g2s.load(a_cur1, A1_gl_offset + 0 * BLOCK_K * c_m)

            if _use_single:
                # Symmetric prelude (every wave runs the same barrier): just
                # drain cur(0) fully + sync. No `next` slot, no wave_m-asymmetric
                # barrier (that desyncs the single-buffer loop → race).
                wait_barrier(0)
            else:
                if wave_m == 1:
                    rocdl.s_barrier()

                wait_barrier(N_LDS_STEPS_A + N_LDS_STEPS_B)

                b_g2s.load(b_next0, B0_gl_offset + 1 * BLOCK_K * c_n)
                a_g2s.load(a_next0, A0_gl_offset + 1 * BLOCK_K * c_m)
                b_g2s.load(b_next1, B1_gl_offset + 1 * BLOCK_K * c_n)

                wait_barrier(N_LDS_STEPS_A + 2 * N_LDS_STEPS_B)

            BR_B1 = bool(barrier_mask & 0x01)
            BR_B2 = bool(barrier_mask & 0x02)
            BR_B3 = bool(barrier_mask & 0x04)
            BR_B4 = bool(barrier_mask & 0x08)
            BR_B5 = bool(barrier_mask & 0x10)
            BR_B6 = bool(barrier_mask & 0x20)
            BR_B7 = bool(barrier_mask & 0x40)

            # Interleaved A-load half (inline path): software-pipeline (depth-2)
            # the per-tile inline ds_read + mma. Caps A raw-read liveness at 2
            # tiles (no 350-VGPR spill that otherwise cripples inline_both at
            # BM=256) AND, crucially, uses GRADED lgkmcnt so tile i+1's LDS read
            # overlaps tile i's mma — the per-tile lgkmcnt(0) full-drain was the
            # real serialiser (2243 lgkmcnt(0) vs b_inline's 898 + graded
            # lgkmcnt(4/8/12)), not the vmcnt(0) drain (inline path: vmcnt0=1).
            def _mma_half(a_lds, bx0, bx1, cq0, cq1):
                def _consume(i, a_frag):
                    rocdl.s_setprio(1)
                    for j in range_constexpr(N_TILES_B):
                        cq0[mfma.idx(i, j)] = mfma._do_mma(a_frag, bx0[j], cq0[mfma.idx(i, j)])
                    for j in range_constexpr(N_TILES_B):
                        cq1[mfma.idx(i, j)] = mfma._do_mma(a_frag, bx1[j], cq1[mfma.idx(i, j)])
                    rocdl.s_setprio(0)
                a_s2r.load_pipelined(a_lds, _consume)
                return cq0, cq1

            if _use_single:
                # Single-buffer steady loop: full K_ITERS, same cur buffer reused.
                #   wait_barrier(0): cur(k) G2S landed (vmcnt0) + cross-wave sync
                #   s2r + 4-quadrant mma from cur(k)
                #   s_barrier: all waves done reading cur(k) before overwrite
                #   tail G2S: prefetch cur(k+1) into the SAME buffer
                # 2 barriers/iter, no `next` slots → LDS halved → 2 blocks/CU.
                for k in range_constexpr(K_ITERS):
                    wait_barrier(0)
                    b0_frag = b_s2r.load(b_cur0)
                    b1_frag = b_s2r.load(b_cur1)
                    a0_frag = a_s2r.load(a_cur0)
                    a1_frag = a_s2r.load(a_cur1)
                    rocdl.s_setprio(1)
                    c00_frag = mfma.call(a0_frag, b0_frag, c00_frag)
                    c01_frag = mfma.call(a0_frag, b1_frag, c01_frag)
                    c10_frag = mfma.call(a1_frag, b0_frag, c10_frag)
                    c11_frag = mfma.call(a1_frag, b1_frag, c11_frag)
                    rocdl.s_setprio(0)
                    rocdl.s_barrier()
                    if k + 1 < K_ITERS:
                        b_g2s.load(b_cur0, B0_gl_offset + (k + 1) * BLOCK_K * c_n)
                        a_g2s.load(a_cur0, A0_gl_offset + (k + 1) * BLOCK_K * c_m)
                        b_g2s.load(b_cur1, B1_gl_offset + (k + 1) * BLOCK_K * c_n)
                        a_g2s.load(a_cur1, A1_gl_offset + (k + 1) * BLOCK_K * c_m)
            elif _use_interleave and _a_inline:
                for k in range_constexpr(K_ITERS - 2):
                    b0_frag = b_s2r.load(b_cur0)
                    b1_frag = b_s2r.load(b_cur1)
                    a_g2s.load(a_next1, A1_gl_offset + (k + 1) * BLOCK_K * c_m)
                    rocdl.s_barrier()
                    c00_frag, c01_frag = _mma_half(a_cur0, b0_frag, b1_frag, c00_frag, c01_frag)
                    rocdl.s_barrier()
                    b_g2s.load(b_cur0, B0_gl_offset + (k + 2) * BLOCK_K * c_n)
                    a_g2s.load(a_cur0, A0_gl_offset + (k + 2) * BLOCK_K * c_m)
                    rocdl.s_barrier()
                    c10_frag, c11_frag = _mma_half(a_cur1, b0_frag, b1_frag, c10_frag, c11_frag)
                    b_g2s.load(b_cur1, B1_gl_offset + (k + 2) * BLOCK_K * c_n)
                    wait_barrier(2 * N_LDS_STEPS_A + N_LDS_STEPS_B)
                    rocdl.s_barrier()
                    a_cur0, a_next0 = a_next0, a_cur0
                    a_cur1, a_next1 = a_next1, a_cur1
                    b_cur0, b_next0 = b_next0, b_cur0
                    b_cur1, b_next1 = b_next1, b_cur1

                # Epilog 1.
                k = K_ITERS - 2
                b0_frag = b_s2r.load(b_cur0)
                b1_frag = b_s2r.load(b_cur1)
                rocdl.s_barrier()
                c00_frag, c01_frag = _mma_half(a_cur0, b0_frag, b1_frag, c00_frag, c01_frag)
                rocdl.s_barrier()
                b0_next_frag = b_s2r.load(b_next0)
                a_g2s.load(a_next1, A1_gl_offset + (k + 1) * BLOCK_K * c_m)
                rocdl.s_barrier()
                c10_frag, c11_frag = _mma_half(a_cur1, b0_frag, b1_frag, c10_frag, c11_frag)
                rocdl.s_barrier()

                a_cur0, a_next0 = a_next0, a_cur0
                a_cur1, a_next1 = a_next1, a_cur1
                b_cur0, b_next0 = b_next0, b_cur0
                b_cur1, b_next1 = b_next1, b_cur1

                # Epilog 2.
                b1_frag = b_s2r.load(b_cur1)
                wait_barrier(0)
                c00_frag, c01_frag = _mma_half(a_cur0, b0_next_frag, b1_frag, c00_frag, c01_frag)
                rocdl.s_barrier()
                c10_frag, c11_frag = _mma_half(a_cur1, b0_next_frag, b1_frag, c10_frag, c11_frag)
                rocdl.s_barrier()
            else:
                if _use_crr:
                    # HK-CRR 4-barrier steady loop (port of kernel_fp8_layouts.cpp
                    # crr_kernel main loop). Both B halves loaded upfront; A half-0
                    # × {b0,b1} then A half-1 × {b0,b1}; only 4 s_barrier/iter with
                    # graded lgkm/vmcnt hints + lgkmcnt(0) right before each MMA
                    # group. Same 4 G2S per iter as the 7-barrier loop → shared epilogs.
                    _crr_vmcnt = (2 * N_LDS_STEPS_A + N_LDS_STEPS_B) if _CRR_VMCNT < 0 else _CRR_VMCNT
                    def _lgkm0():
                        _llvm.inline_asm(res=None, operands_=[], asm_string="s_waitcnt lgkmcnt(0)",
                                          constraints="", has_side_effects=True)
                    def _lgkm_barrier(n):
                        _llvm.inline_asm(res=None, operands_=[],
                                          asm_string=f"s_waitcnt lgkmcnt({n})\ns_barrier",
                                          constraints="", has_side_effects=True)
                    for k in range_constexpr(K_ITERS - 2):
                        b0_frag = b_s2r.load(b_cur0)
                        b1_frag = b_s2r.load(b_cur1)
                        a0_frag = a_s2r.load(a_cur0)
                        a_g2s.load(a_next1, A1_gl_offset + (k + 1) * BLOCK_K * c_m)
                        _lgkm_barrier(_CRR_LGKM)            # barrier 1
                        _lgkm0()
                        rocdl.s_setprio(1)
                        c00_frag = mfma.call(a0_frag, b0_frag, c00_frag)
                        c01_frag = mfma.call(a0_frag, b1_frag, c01_frag)
                        rocdl.s_setprio(0)
                        rocdl.s_barrier()                   # barrier 2 (MID)
                        a1_frag = a_s2r.load(a_cur1)
                        a_g2s.load(a_cur0, A0_gl_offset + (k + 2) * BLOCK_K * c_m)
                        b_g2s.load(b_cur1, B1_gl_offset + (k + 2) * BLOCK_K * c_n)
                        wait_barrier(_crr_vmcnt)            # barrier 3 (vmcnt + s_barrier)
                        _lgkm0()
                        rocdl.s_setprio(1)
                        c10_frag = mfma.call(a1_frag, b0_frag, c10_frag)
                        c11_frag = mfma.call(a1_frag, b1_frag, c11_frag)
                        rocdl.s_setprio(0)
                        rocdl.s_barrier()                   # barrier 4
                        b_g2s.load(b_cur0, B0_gl_offset + (k + 2) * BLOCK_K * c_n)
                        a_cur0, a_next0 = a_next0, a_cur0
                        a_cur1, a_next1 = a_next1, a_cur1
                        b_cur0, b_next0 = b_next0, b_cur0
                        b_cur1, b_next1 = b_next1, b_cur1
                else:
                    for k in range_constexpr(K_ITERS - 2):
                        b0_frag = b_s2r.load(b_cur0)
                        a0_frag = a_s2r.load(a_cur0)
                        a_g2s.load(a_next1, A1_gl_offset + (k + 1) * BLOCK_K * c_m)
                        if BR_B1: rocdl.s_barrier()
                        rocdl.s_setprio(1)
                        c00_frag = mfma.call(a0_frag, b0_frag, c00_frag)
                        rocdl.s_setprio(0)
                        if BR_B2: rocdl.s_barrier()
                        b1_frag = b_s2r.load(b_cur1)
                        b_g2s.load(b_cur0, B0_gl_offset + (k + 2) * BLOCK_K * c_n)
                        if BR_B3: rocdl.s_barrier()
                        rocdl.s_setprio(1)
                        c01_frag = mfma.call(a0_frag, b1_frag, c01_frag)
                        rocdl.s_setprio(0)
                        if BR_B4: rocdl.s_barrier()
                        a1_frag = a_s2r.load(a_cur1)
                        a_g2s.load(a_cur0, A0_gl_offset + (k + 2) * BLOCK_K * c_m)
                        if BR_B5: rocdl.s_barrier()
                        rocdl.s_setprio(1)
                        c10_frag = mfma.call(a1_frag, b0_frag, c10_frag)
                        rocdl.s_setprio(0)
                        if BR_B6: rocdl.s_barrier()
                        b_g2s.load(b_cur1, B1_gl_offset + (k + 2) * BLOCK_K * c_n)
                        wait_barrier(2 * N_LDS_STEPS_A + N_LDS_STEPS_B)
                        rocdl.s_setprio(1)
                        c11_frag = mfma.call(a1_frag, b1_frag, c11_frag)
                        rocdl.s_setprio(0)
                        if BR_B7: rocdl.s_barrier()
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
            A: fx.Tensor, B: fx.Tensor, C: fx.Tensor,
            A_scale: fx.Tensor, B_scale: fx.Tensor,
            c_m: fx.Int32, c_n: fx.Int32, stream: fx.Stream,
        ):
            grid_x = ceildiv(c_m, BLOCK_M) * ceildiv(c_n, BLOCK_N)
            kernel_dense_tn(
                A, B, C, A_scale, B_scale, c_m, c_n,
                value_attrs=_make_value_attrs(waves_per_eu, agpr_alloc, "512,512"),
            ).launch(grid=(grid_x, 1, 1), block=(512, 1, 1), stream=stream)
        return launch_dense_tn


    @functools.lru_cache(maxsize=128)
    def _compile_dense_nn_btr(
        K: int,
        BLOCK_M: int = 256,
        BLOCK_N: int = 256,
        GROUP_M: int = 4,
        waves_per_eu: int = 2,
        agpr_alloc: int = 0,
    ):
        """NN B-LDS-transpose variant of ``_compile_dense_nn``.

        Replaces the wave-cooperative ``ds_read_b64_tr_b8`` (``S2RLoaderTr``)
        in the main loop with a 2-pass LDS scheme:
          1. G2S writes to ``B_lds_temp_*`` using ``compute_global_offsets_nn_identity``
             (no swizzle in HBM offset → LDS_temp is plain K-major).
          2. ``LDS2LDSTransposer`` transposes ``B_lds_temp_*`` → ``B_lds_final_*``
             with ``swizzle_128(N, K)`` layout matching what NT ``S2RLoader``
             expects (probe-verified mismatch=0 on chi2762 gfx950).
          3. Main loop reads ``B_lds_final_*`` via per-lane ``ds_read_b128``
             (``S2RLoader`` shared with NT path).

        Motivation (ISA evidence, K=28672):
            NN: 3584 ``ds_read_b64_tr_b8`` + 3584 ``ds_read_b128`` + 447
                ``s_waitcnt vmcnt(0)`` (LLVM-conservative drain before each
                wave-coop tr8 group)
            NT: 5376 ``ds_read_b128`` + 1 ``s_waitcnt vmcnt(0)``
            Eliminating the 447 vmcnt(0) drains is the only known lever to
            close the NN/NT = 90.5% → 97% gap on gfx950 (cdna4) within source.

        LDS budget audit (BLOCK_K=128, LDS_BLOCK_M=LDS_BLOCK_N=128):
            4× A_lds = 64 KB
            4× B_lds_temp (cur/next × 0/1) = 64 KB
            2× B_lds_final (0/1; single K-iter slot, overwritten each iter) = 32 KB
            Total = 160 KB exactly (gfx950 cap)

        See NN_G2S_TRANSPOSE_DESIGN.md (Session 7+ plan).
        """
        BLOCK_K = 128
        assert BLOCK_M >= 128 and BLOCK_N >= 256 and BLOCK_M % 128 == 0 and BLOCK_N % 256 == 0
        assert K % BLOCK_K == 0
        # LDS2LDSTransposer is hard-wired to BLOCK_K=128 and LDS_BLOCK_N=128.
        # If we need other block sizes later, generalize the transposer first.
        assert BLOCK_N == 256 and BLOCK_M >= 128, (
            f"BTR path requires BLOCK_N=256 (LDS_BLOCK_N=128), got {BLOCK_N}"
        )

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
        b_lds_size = LDS_BLOCK_N * BLOCK_K   # 16 KB per buffer

        @fx.struct
        class SharedStorage:
            A_lds_cur_0:       fx.Array[fx.Float8E4M3FN, a_lds_size, 16]
            A_lds_cur_1:       fx.Array[fx.Float8E4M3FN, a_lds_size, 16]
            A_lds_next_0:      fx.Array[fx.Float8E4M3FN, a_lds_size, 16]
            A_lds_next_1:      fx.Array[fx.Float8E4M3FN, a_lds_size, 16]
            # B temp: K-major identity layout (G2S writes here)
            B_lds_temp_cur_0:  fx.Array[fx.Float8E4M3FN, b_lds_size, 16]
            B_lds_temp_cur_1:  fx.Array[fx.Float8E4M3FN, b_lds_size, 16]
            B_lds_temp_next_0: fx.Array[fx.Float8E4M3FN, b_lds_size, 16]
            B_lds_temp_next_1: fx.Array[fx.Float8E4M3FN, b_lds_size, 16]
            # B final: N-major swizzle_128 layout (transposed; S2RLoader reads here)
            B_lds_final_0:     fx.Array[fx.Float8E4M3FN, b_lds_size, 16]
            B_lds_final_1:     fx.Array[fx.Float8E4M3FN, b_lds_size, 16]

        @flyc.kernel(known_block_size=[512, 1, 1])
        def kernel_dense_nn_btr(
            A: fx.Tensor,
            B: fx.Tensor,
            C: fx.Tensor,
            A_scale: fx.Tensor,
            B_scale: fx.Tensor,
            c_m: fx.Int32,
            c_n: fx.Int32,
        ):
            # Matches _compile_dense_nn workaround.
            _ = str(fx.thread_idx.x)
            F8_IR_t = fx.Float8E4M3FN.ir_type

            n_blocks = ceildiv(c_n, BLOCK_N)

            lds = fx.SharedAllocator().allocate(SharedStorage).peek()
            a_cur0 = lds.A_lds_cur_0
            a_cur1 = lds.A_lds_cur_1
            a_next0 = lds.A_lds_next_0
            a_next1 = lds.A_lds_next_1
            b_temp_cur0 = lds.B_lds_temp_cur_0
            b_temp_cur1 = lds.B_lds_temp_cur_1
            b_temp_next0 = lds.B_lds_temp_next_0
            b_temp_next1 = lds.B_lds_temp_next_1
            b_final0 = lds.B_lds_final_0
            b_final1 = lds.B_lds_final_1

            lane_id = fx.thread_idx.x % 64
            wave_id = fx.thread_idx.x // 64
            wave_m = wave_id // 4
            wave_n = wave_id % 4

            num_pid_m = ceildiv(c_m, BLOCK_M)
            pid = fx.block_idx.x
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
            # BTR uses IDENTITY offsets (no swizzle_128) for B; LDS_temp ends up
            # plain K-major which is what LDS2LDSTransposer's source-decode
            # formula expects.
            gl_off_b = compute_global_offsets_nn_identity(lane_id, wave_id, c_n, N_LDS_ROUNDS)

            mfma = Mfma16x16x128(N_TILES_A, N_TILES_B)

            a_g2s = G2SLoader(a_div, gl_off_a, N_LDS_STEPS_A, F8_IR_t, wave_id)
            b_g2s = G2SLoader(b_div, gl_off_b, N_LDS_STEPS_B, F8_IR_t, wave_id)
            a_s2r = S2RLoader(wave_m, N_TILES_A)
            # KEY DIFFERENCE: NT-style S2RLoader reads ds_read_b128 from
            # swizzle_128 N-major LDS (the transpose target).
            b_s2r = S2RLoader(wave_n, N_TILES_B)
            transposer = LDS2LDSTransposer(LDS_BLOCK_N, BLOCK_K)
            store_c = StoreC(A_scale, B_scale, C, c_m, c_n, mfma.idx, N_TILES_A, N_TILES_B)


            c00_frag = [mfma.zero_value] * N_ACCUMS
            c01_frag = [mfma.zero_value] * N_ACCUMS
            c10_frag = [mfma.zero_value] * N_ACCUMS
            c11_frag = [mfma.zero_value] * N_ACCUMS

            # Prelude.
            b_g2s.load(b_temp_cur0, B0_gl_offset + 0 * BLOCK_K * c_n)
            a_g2s.load(a_cur0, A0_gl_offset + 0 * BLOCK_K)
            b_g2s.load(b_temp_cur1, B1_gl_offset + 0 * BLOCK_K * c_n)
            a_g2s.load(a_cur1, A1_gl_offset + 0 * BLOCK_K)

            # NOTE: the native NN kernel has `if wave_m == 1: rocdl.s_barrier()`
            # here as a pipeline-stagger trick. We CANNOT use it in BTR because
            # the transposer is a WG-cooperative LDS write (all 8 waves must
            # participate, including wave_m=1). The stagger barrier desyncs
            # wave_m=1 from wave_m=0 such that wave_m=1 waves miss the
            # transposer call entirely (probe-verified: K_chunk={2,3,6,7}
            # writes from waves 4-7 missing, causing the NaN-at-odd-N bug).

            wait_barrier(N_LDS_STEPS_A + N_LDS_STEPS_B)

            b_g2s.load(b_temp_next0, B0_gl_offset + 1 * BLOCK_K * c_n)
            a_g2s.load(a_next0, A0_gl_offset + 1 * BLOCK_K)
            b_g2s.load(b_temp_next1, B1_gl_offset + 1 * BLOCK_K * c_n)

            wait_barrier(N_LDS_STEPS_A + 2 * N_LDS_STEPS_B)

            # Drain ALL b_temp loads before transpose reads from them.
            wait_barrier(0)
            transposer.transpose(b_temp_cur0, b_final0)
            transposer.transpose(b_temp_cur1, b_final1)
            lgkm_barrier()

            # Main loop.
            for k in range_constexpr(K_ITERS - 2):
                b0_frag = b_s2r.load(b_final0)
                a0_frag = a_s2r.load(a_cur0)
                a_g2s.load(a_next1, A1_gl_offset + (k + 1) * BLOCK_K)
                rocdl.s_barrier()

                rocdl.s_setprio(1)
                c00_frag = mfma.call(a0_frag, b0_frag, c00_frag)
                rocdl.s_setprio(0)
                rocdl.s_barrier()

                b1_frag = b_s2r.load(b_final1)
                b_g2s.load(b_temp_cur0, B0_gl_offset + (k + 2) * BLOCK_K * c_n)
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

                b_g2s.load(b_temp_cur1, B1_gl_offset + (k + 2) * BLOCK_K * c_n)
                wait_barrier(2 * N_LDS_STEPS_A + N_LDS_STEPS_B)

                rocdl.s_setprio(1)
                c11_frag = mfma.call(a1_frag, b1_frag, c11_frag)
                rocdl.s_setprio(0)
                rocdl.s_barrier()

                a_cur0, a_next0 = a_next0, a_cur0
                a_cur1, a_next1 = a_next1, a_cur1
                b_temp_cur0, b_temp_next0 = b_temp_next0, b_temp_cur0
                b_temp_cur1, b_temp_next1 = b_temp_next1, b_temp_cur1

                # After swap, b_temp_cur now holds K-iter (k+1)'s data (was
                # b_temp_next before swap). Drain b_temp loads before
                # transpose reads, then re-transpose into b_final so next
                # iter's b_s2r.load sees the correct b_final. lgkm_barrier
                # (not plain s_barrier) is required — see prelude comment.
                wait_barrier(0)
                transposer.transpose(b_temp_cur0, b_final0)
                transposer.transpose(b_temp_cur1, b_final1)
                lgkm_barrier()

            # Epilog 1.
            k = K_ITERS - 2
            b0_frag = b_s2r.load(b_final0)
            a0_frag = a_s2r.load(a_cur0)
            rocdl.s_barrier()

            rocdl.s_setprio(1)
            c00_frag = mfma.call(a0_frag, b0_frag, c00_frag)
            rocdl.s_setprio(0)
            rocdl.s_barrier()

            b1_frag = b_s2r.load(b_final1)
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

            # Same stale-a1 fix as native NN: prefetch a1 for K_ITERS-1 so it
            # lands before epilog 2's a_s2r.load. We drop the original NN's
            # b0_frag preload from b_next0 because in BTR the iter-(K_ITERS-1)
            # B data is not yet in b_final — we'd need to transpose first, and
            # that comes after the swap below.
            a_g2s.load(a_next1, A1_gl_offset + (k + 1) * BLOCK_K)
            rocdl.s_barrier()

            rocdl.s_setprio(1)
            c11_frag = mfma.call(a1_frag, b1_frag, c11_frag)
            rocdl.s_setprio(0)
            rocdl.s_barrier()

            a_cur0, a_next0 = a_next0, a_cur0
            a_cur1, a_next1 = a_next1, a_cur1
            b_temp_cur0, b_temp_next0 = b_temp_next0, b_temp_cur0
            b_temp_cur1, b_temp_next1 = b_temp_next1, b_temp_cur1

            # Transpose K-iter (K_ITERS-1)'s B (now in b_temp_cur post-swap)
            # into b_final for epilog 2. Drain b_temp loads before transpose
            # (same rationale as prelude / main-loop transpose calls).
            wait_barrier(0)
            transposer.transpose(b_temp_cur0, b_final0)
            transposer.transpose(b_temp_cur1, b_final1)
            lgkm_barrier()

            # Epilog 2.
            b0_frag = b_s2r.load(b_final0)
            a0_frag = a_s2r.load(a_cur0)
            wait_barrier(0)

            rocdl.s_setprio(1)
            c00_frag = mfma.call(a0_frag, b0_frag, c00_frag)
            rocdl.s_setprio(0)
            rocdl.s_barrier()

            b1_frag = b_s2r.load(b_final1)
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
        def launch_dense_nn_btr(
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
            kernel_dense_nn_btr(
                A,
                B,
                C,
                A_scale,
                B_scale,
                c_m,
                c_n,
                value_attrs=_make_value_attrs(waves_per_eu, agpr_alloc, "512,512"),
            ).launch(grid=(grid_x, 1, 1), block=(512, 1, 1), stream=stream)

        return launch_dense_nn_btr


    @functools.lru_cache(maxsize=8)
    def _compile_nn_btr_prelude_probe():
        """Probe: run the FULL BTR prelude (4 G2S + 2 transposes) and dump
        b_final0 to HBM. Compare against expected to localize uncovered bytes.

        Differs from _compile_nn_btr_probe: this exercises the prelude in
        kernel context (multiple G2S loads, full LDS budget, register
        pressure), to surface bugs that only manifest under that pressure.
        """
        BLOCK_K = 128
        LDS_BLOCK_N = 128
        LDS_BLOCK_M = 128
        a_lds_size = LDS_BLOCK_M * BLOCK_K
        b_lds_size = LDS_BLOCK_N * BLOCK_K

        @fx.struct
        class _PreludeStorage:
            A_lds_cur_0:       fx.Array[fx.Float8E4M3FN, a_lds_size, 16]
            A_lds_cur_1:       fx.Array[fx.Float8E4M3FN, a_lds_size, 16]
            A_lds_next_0:      fx.Array[fx.Float8E4M3FN, a_lds_size, 16]
            A_lds_next_1:      fx.Array[fx.Float8E4M3FN, a_lds_size, 16]
            B_lds_temp_cur_0:  fx.Array[fx.Float8E4M3FN, b_lds_size, 16]
            B_lds_temp_cur_1:  fx.Array[fx.Float8E4M3FN, b_lds_size, 16]
            B_lds_temp_next_0: fx.Array[fx.Float8E4M3FN, b_lds_size, 16]
            B_lds_temp_next_1: fx.Array[fx.Float8E4M3FN, b_lds_size, 16]
            B_lds_final_0:     fx.Array[fx.Float8E4M3FN, b_lds_size, 16]
            B_lds_final_1:     fx.Array[fx.Float8E4M3FN, b_lds_size, 16]

        @flyc.kernel(known_block_size=[512, 1, 1])
        def kernel_nn_btr_prelude_probe(
            B_in: fx.Tensor,        # fp8e4m3 [BLOCK_K * 2, LDS_BLOCK_N * 2] flat
            A_in: fx.Tensor,        # fp8e4m3 [LDS_BLOCK_M * 2, BLOCK_K * 2] flat
            dump_final0: fx.Tensor, # uint8 [16384]
            dump_final1: fx.Tensor, # uint8 [16384]
        ):
            _ = str(fx.thread_idx.x)
            F8_IR_t = fx.Float8E4M3FN.ir_type

            lds = fx.SharedAllocator().allocate(_PreludeStorage).peek()
            a_cur0 = lds.A_lds_cur_0
            a_cur1 = lds.A_lds_cur_1
            a_next0 = lds.A_lds_next_0
            a_next1 = lds.A_lds_next_1
            b_temp_cur0 = lds.B_lds_temp_cur_0
            b_temp_cur1 = lds.B_lds_temp_cur_1
            b_temp_next0 = lds.B_lds_temp_next_0
            b_temp_next1 = lds.B_lds_temp_next_1
            b_final0 = lds.B_lds_final_0
            b_final1 = lds.B_lds_final_1

            lane_id = fx.thread_idx.x % 64
            wave_id = fx.thread_idx.x // 64
            wave_m = wave_id // 4

            # K = 2*BLOCK_K so the prelude's 2 K-iter loads are valid.
            K_total = 2 * BLOCK_K
            N_total = 2 * LDS_BLOCK_N
            BLOCK_M = LDS_BLOCK_M * 2
            BLOCK_N = LDS_BLOCK_N * 2
            N_LDS_STEPS_A = LDS_BLOCK_M // 64
            N_LDS_STEPS_B = LDS_BLOCK_N // 64
            N_LDS_ROUNDS = max(N_LDS_STEPS_A, N_LDS_STEPS_B)

            gA = make_fp8_buffer_tensor(A_in, F8_IR_t)
            gB = make_fp8_buffer_tensor(B_in, F8_IR_t)
            a_div = fx.logical_divide(gA, fx.make_layout(1, 1))
            b_div = fx.logical_divide(gB, fx.make_layout(1, 1))

            gl_off_a = compute_global_swizzle(lane_id, wave_id, K_total, N_LDS_ROUNDS, preshuffled=False)
            gl_off_b = compute_global_offsets_nn_identity(lane_id, wave_id, N_total, N_LDS_ROUNDS)

            a_g2s = G2SLoader(a_div, gl_off_a, N_LDS_STEPS_A, F8_IR_t, wave_id)
            b_g2s = G2SLoader(b_div, gl_off_b, N_LDS_STEPS_B, F8_IR_t, wave_id)
            transposer = LDS2LDSTransposer(LDS_BLOCK_N, BLOCK_K)

            # MIRROR the BTR prelude:
            A0_gl_offset = 0
            A1_gl_offset = LDS_BLOCK_M * K_total
            B0_gl_offset = 0
            B1_gl_offset = LDS_BLOCK_N

            b_g2s.load(b_temp_cur0, B0_gl_offset + 0 * BLOCK_K * N_total)
            a_g2s.load(a_cur0, A0_gl_offset + 0 * BLOCK_K)
            b_g2s.load(b_temp_cur1, B1_gl_offset + 0 * BLOCK_K * N_total)
            a_g2s.load(a_cur1, A1_gl_offset + 0 * BLOCK_K)

            wait_barrier(N_LDS_STEPS_A + N_LDS_STEPS_B)

            b_g2s.load(b_temp_next0, B0_gl_offset + 1 * BLOCK_K * N_total)
            a_g2s.load(a_next0, A0_gl_offset + 1 * BLOCK_K)
            b_g2s.load(b_temp_next1, B1_gl_offset + 1 * BLOCK_K * N_total)

            wait_barrier(N_LDS_STEPS_A + 2 * N_LDS_STEPS_B)
            wait_barrier(0)

            transposer.transpose(b_temp_cur0, b_final0)
            transposer.transpose(b_temp_cur1, b_final1)
            lgkm_barrier()

            # Phase 3: dump LDS_final0/1 to HBM linearly.
            tid = fx.thread_idx.x
            gDump0 = make_fp8_buffer_tensor(dump_final0, F8_IR_t)
            gDump1 = make_fp8_buffer_tensor(dump_final1, F8_IR_t)
            d0_div = fx.logical_divide(gDump0, fx.make_layout(1, 1))
            d1_div = fx.logical_divide(gDump1, fx.make_layout(1, 1))

            for iter in range_constexpr(2):
                lin_off = (tid + iter * 512) * 16

                src_tup = fx.make_int_tuple(lin_off)
                src_ptr = fx.add_offset(b_final0.ptr, src_tup)
                src_i8 = fx.recast_iter(fx.Uint8, src_ptr)
                src_view = fx.make_view(src_i8, fx.make_layout(16, 1))
                v0 = src_view.load()
                reg0 = fx.make_rmem_tensor(fx.make_layout(16, 1), fx.Uint8)
                fx.memref_store_vec(v0, reg0)
                store_atom0 = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.Uint8)
                fx.copy(store_atom0, reg0, fx.slice(d0_div, (None, fx.Int32(lin_off))))

                src_ptr1 = fx.add_offset(b_final1.ptr, src_tup)
                src_i8_1 = fx.recast_iter(fx.Uint8, src_ptr1)
                src_view1 = fx.make_view(src_i8_1, fx.make_layout(16, 1))
                v1 = src_view1.load()
                reg1 = fx.make_rmem_tensor(fx.make_layout(16, 1), fx.Uint8)
                fx.memref_store_vec(v1, reg1)
                store_atom1 = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.Uint8)
                fx.copy(store_atom1, reg1, fx.slice(d1_div, (None, fx.Int32(lin_off))))

        @flyc.jit
        def launch_nn_btr_prelude_probe(
            B_in: fx.Tensor,
            A_in: fx.Tensor,
            dump_final0: fx.Tensor,
            dump_final1: fx.Tensor,
            stream: fx.Stream,
        ):
            kernel_nn_btr_prelude_probe(
                B_in, A_in, dump_final0, dump_final1,
                value_attrs=_make_value_attrs(2, 32, "512,512"),
            ).launch(grid=(1, 1, 1), block=(512, 1, 1), stream=stream)

        return launch_nn_btr_prelude_probe


    @functools.lru_cache(maxsize=8)
    def _compile_nn_btr_probe():
        """Probe kernel for NN b_lds_transpose primitive.

        Tests the chain: G2SLoader(identity offsets) → LDS_temp → LDS2LDSTransposer
        → LDS_final (swizzle_128 N-major).

        Setup: 1 WG (512 threads), reads one 128×128 B tile from HBM `B_in`,
        writes the resulting LDS_final layout to HBM `lds_dump` for host check.

        Host verification:
            for N ∈ [0, 128), K_chunk ∈ [0, 8), k_local ∈ [0, 16):
                K = K_chunk * 16 + k_local
                (r, c) = swizzle_128(N, K_chunk * 16)
                expected = B_in[K, N]
                actual   = lds_dump[r * 128 + c + k_local]
                assert expected == actual

        Pass = mismatch == 0.
        """
        BLOCK_K = 128
        LDS_BLOCK_N = 128
        b_lds_size = BLOCK_K * LDS_BLOCK_N      # 16384 bytes

        @fx.struct
        class _ProbeStorage:
            B_lds_temp:  fx.Array[fx.Float8E4M3FN, b_lds_size, 16]
            B_lds_final: fx.Array[fx.Float8E4M3FN, b_lds_size, 16]

        @flyc.kernel(known_block_size=[512, 1, 1])
        def kernel_nn_btr_probe(
            B_in: fx.Tensor,     # fp8e4m3 [BLOCK_K=128, LDS_BLOCK_N=128] flat
            lds_dump: fx.Tensor, # uint8 [16384] flat output
        ):
            _ = str(fx.thread_idx.x)
            F8_IR_t = fx.Float8E4M3FN.ir_type

            lds = fx.SharedAllocator().allocate(_ProbeStorage).peek()
            b_temp = lds.B_lds_temp
            b_final = lds.B_lds_final

            lane_id = fx.thread_idx.x % 64
            wave_id = fx.thread_idx.x // 64

            gB = make_fp8_buffer_tensor(B_in, F8_IR_t)
            b_div = fx.logical_divide(gB, fx.make_layout(1, 1))

            # n_rounds = 1: single 64-K-row pass. With 8 waves × 8 K-rows/wave
            # × 2 steps = 128 K-rows = full BLOCK_K. n_load_steps = LDS_BLOCK_N
            # // 64 = 2.
            n_load_steps = LDS_BLOCK_N // 64
            gl_offsets = compute_global_offsets_nn_identity(
                lane_id, wave_id, LDS_BLOCK_N, n_load_steps
            )
            b_g2s = G2SLoader(b_div, gl_offsets, n_load_steps, F8_IR_t, wave_id)
            transposer = LDS2LDSTransposer(LDS_BLOCK_N, BLOCK_K)

            # Phase 1: HBM → LDS_temp (identity K-major)
            b_g2s.load(b_temp, 0)
            wait_barrier(0)

            # Phase 2: LDS_temp → LDS_final (swizzle_128 N-major) via transpose
            transposer.transpose(b_temp, b_final)
            rocdl.s_barrier()

            # Phase 3: dump LDS_final to HBM lds_dump (linear copy).
            # 16384 bytes / 512 threads = 32 bytes/thread = 2 × b128 writes.
            tid = fx.thread_idx.x
            gDump = make_fp8_buffer_tensor(lds_dump, F8_IR_t)
            d_div = fx.logical_divide(gDump, fx.make_layout(1, 1))
            for iter in range_constexpr(2):
                lin_off = (tid + iter * 512) * 16
                # Read 16 bytes from b_final[lin_off]
                src_tup = fx.make_int_tuple(lin_off)
                src_ptr = fx.add_offset(b_final.ptr, src_tup)
                src_i8 = fx.recast_iter(fx.Uint8, src_ptr)
                src_view = fx.make_view(src_i8, fx.make_layout(16, 1))
                v = src_view.load()
                # Write 16 bytes to lds_dump[lin_off]
                dst_slice = fx.slice(d_div, (None, fx.Int32(lin_off)))
                # Use a copy_atom for byte-store, or store via view:
                # Simpler — use raw memref store on a buffer tensor view.
                # Following StoreC's pattern via fx.copy + buffer_load atom not
                # applicable for store; use direct view store:
                # Build a register tensor and store via BufferCopy128b.
                reg = fx.make_rmem_tensor(fx.make_layout(16, 1), fx.Uint8)
                fx.memref_store_vec(v, reg)
                store_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.Uint8)
                fx.copy(store_atom, reg, dst_slice)

        @flyc.jit
        def launch_nn_btr_probe(
            B_in: fx.Tensor,
            lds_dump: fx.Tensor,
            stream: fx.Stream,
        ):
            kernel_nn_btr_probe(
                B_in, lds_dump,
                value_attrs=_make_value_attrs(2, 0, "512,512"),
            ).launch(grid=(1, 1, 1), block=(512, 1, 1), stream=stream)

        return launch_nn_btr_probe


    # NN 4-wave kernel removed 2026-05-28 (5 levers exhausted, all 30-45%
    # slower than 8w on big shapes; HW caps 1 wave/SIMD for this layout).


else:

    def _compile_dense_nt(K: int, BLOCK_M: int = 256, BLOCK_N: int = 256):
        raise ImportError("FlyDSL is not available -- this entry should be gated by can_handle.")

    def _compile_dense_nn(
        K: int,
        BLOCK_M: int = 256,
        BLOCK_N: int = 256,
        GROUP_M: int = 4,
        waves_per_eu: int = 2,
        agpr_alloc: int = 0,
    ):
        raise ImportError("FlyDSL is not available -- this entry should be gated by can_handle.")

    def _compile_nn_btr_probe():
        raise ImportError("FlyDSL is not available -- this entry should be gated by can_handle.")

    def _compile_nn_btr_prelude_probe():
        raise ImportError("FlyDSL is not available -- this entry should be gated by can_handle.")



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


_AS_I8_CACHE: dict = {}


def _as_i8_flat(t: torch.Tensor) -> torch.Tensor:
    # Cache view conversion by (id(t), data_ptr) — repeated calls with same
    # tensor object (bench / inference) skip the view ops (~1us each saved).
    key = id(t)
    cached = _AS_I8_CACHE.get(key)
    if cached is not None and cached[1] == t.data_ptr():
        return cached[0]
    if t.element_size() == 1 and t.dtype != torch.int8:  # fp8
        v = t.contiguous().view(torch.int8).view(-1)
    else:
        v = t.contiguous().view(-1)
    _AS_I8_CACHE[key] = (v, t.data_ptr())
    if len(_AS_I8_CACHE) > 64:
        _AS_I8_CACHE.pop(next(iter(_AS_I8_CACHE)))
    return v


# Cache broadcasted scale buffers keyed by id(scale_tensor) + length. Repeat
# bench / inference loop reuses the same scale tensor → cache hit avoids
# per-call GPU-copy (saves ~5us each = 10us per fwd dispatch on M=4096 shape).
# Safety: if caller mutates the scale tensor's value in-place between calls,
# the cached buffer is stale. We invalidate on data_ptr mismatch.
_SCALE_BUF_CACHE: dict = {}


def _broadcast_scale(scale: torch.Tensor, length: int, device: torch.device) -> torch.Tensor:
    """Tensorwise scalar → (length,) fp32 buffer. Cache keyed by (id(scale),
    length) avoids GPU copy when same scale tensor reused (bench/inference)."""
    assert scale.numel() == 1, f"per-tensor expects scalar, got {scale.shape}"
    key = (id(scale), length)
    cached = _SCALE_BUF_CACHE.get(key)
    if cached is not None and cached[1] == scale.data_ptr():
        return cached[0]
    buf = torch.empty(length, dtype=torch.float32, device=device)
    buf.copy_(scale.to(dtype=torch.float32, device=device).expand(length))
    _SCALE_BUF_CACHE[key] = (buf, scale.data_ptr())
    if len(_SCALE_BUF_CACHE) > 64:
        # Simple bounded cache.
        _SCALE_BUF_CACHE.pop(next(iter(_SCALE_BUF_CACHE)))
    return buf


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




# NN super-block GROUP_M (Triton-style tile-id swizzle for L2 reuse).
# B3.3 bench-validated (2026-05-25, 3-run median × 16 K-align shapes): GM=4
# beats GM=1 on every shape (Δ +1.4 to +7.9pp, geomean +3.19pp), GM=8 has
# 3 shapes with -7..-15pp regression so not safe as default.
_NN_GROUP_M = 4

# Per-shape NN tile override. Populated by scripts2/_nn_safe_sweep.py
# (2026-05-26, post-swizzle-fix re-sweep). 5-run cold-SNR ≥ 30 dB
# validation per cfg before timing; us_min from those passing cfgs.
# All cfgs verified correct after the swizzle tail-handling bug fix at
# kernel_dense_nn line ~1186. Geomean fly/tri = 1.133, 0/24 SNR fail.
# Format: (BM, GM, AGPR_alloc) — implicit "8w" kernel kind.
# Extended format: ("8w" | "4w", BM, GM, AG[, use_xcd_remap]) for future
# kernel-kind variants. Dispatcher accepts both 3-tuple (legacy 8w) and
# tagged-tuple forms; see _pick_tile_nn + dispatcher in
# gemm_fp8_tensorwise_flydsl_kernel.
_NN_DEFAULT_CFG = (256, 4, 0)
_NN_TILE_OVERRIDE: dict = {
    # 2026-05-28 path J fix: any AGPR=0 entry promoted to AGPR=32 because
    # inline-asm ds_read_b64_tr_b8 (default ON in dispatcher) requires
    # nonzero AGPR pin to avoid `=v` constraint / AGPR-decide compiler
    # conflict (produces nan output). 32 is universal safe; per-shape
    # could be tuned to 16/48 but +/- < 2% perf.
    (2048, 12288,  4096): (256,  1, 32),
    (4096, 12288,  4096): (256,  1, 32),
    (8192, 12288,  4096): (256,  1, 32),
    (2048,  4096,  4096): (128,  2, 48),
    (4096,  4096,  4096): (256,  2, 32),
    (8192,  4096,  4096): (256,  2, 32),
    (2048, 11008,  4096): (256,  2, 32),
    (4096, 11008,  4096): (256, 16, 32),
    (8192, 11008,  4096): (256, 32, 32),
    (2048,  4096, 11008): (128,  4, 48),
    (4096,  4096, 11008): (256,  2, 32),
    (8192,  4096, 11008): (256,  4, 32),
    (2048, 10240,  8192): (256,  1, 32),
    (4096, 10240,  8192): (256,  1, 32),
    (8192, 10240,  8192): (256,  1, 32),
    (2048,  8192,  8192): (256,  2, 32),
    (4096,  8192,  8192): (256,  1, 32),
    (8192,  8192,  8192): (256,  2, 32),
    (2048, 28672,  8192): (256,  1, 32),
    (4096, 28672,  8192): (256, 16, 32),
    (8192, 28672,  8192): (256, 32, 32),
    (2048,  8192, 28672): (256,  1, 32),
    (4096,  8192, 28672): (256,  1, 32),
    (8192,  8192, 28672): (256,  1, 32),
}


def _pick_tile_nn(M: int, N: int, K: int) -> tuple:
    """Returns (BM, GM, AGPR_alloc) for the shape, defaulting to BM=256 GM=4."""
    return _NN_TILE_OVERRIDE.get((M, N, K), _NN_DEFAULT_CFG)



# NT super-block GROUP_M autotune table. Populated by
# scripts2/_autotune_llama_nt.py (2026-05-26, GM ∈ {1,2,4,8,16} × Llama
# 2 7b + 70b × M ∈ {2048,4096,8192} on MI355X). 7 of 8 (N, K) pairs win at
# GM=16 (+5..+9pp geomean); 70b qkv (10240, 8192) stays GM=1.
_NT_DEFAULT_CFG = ("8w", 256, 1)  # (kind, BLOCK_M, GROUP_M)
# Per-shape tile config, keyed by (M, N, K). Two kernel families:
#   ("8w", BLOCK_M, GROUP_M)            — _compile_dense_nt (8-wave, 2×4 wave)
#   ("4w", BLOCK_M, BLOCK_N, use_xcd)   — _compile_dense_nt_4w (FlyDSL 4-wave,
#                                          2×2 wave, XCD-aware swizzle)
# Populated by scripts2/_autotune_llama_nt.py.
_NT_TILE_OVERRIDE: dict = {
    # 2026-05-28: 4wave kernel variant dropped; v3 KEPT (BM=128 fine-grain
    # interleave wins +1-5% on small-M shapes).
    # cfg = ("8w", BM, GM) or ("8w", BM, GM, AGPR_ALLOC)
    #     = ("v3", BM, GM)
    # AGPR_ALLOC encoding (passthrough amdgpu-agpr-alloc):
    #   N>0 = force exact N AGPRs;  -N = allow up to N
    (2048,  4096,  4096):  ("v3", 128, 1),
    (2048,  4096,  11008): ("8w", 128, 2),
    (2048,  8192,  8192):  ("8w", 256, 1, 48),
    (2048,  10240, 8192):  ("v3", 128, 1),
    (2048,  11008, 4096):  ("v3", 128, 16),
    (2048,  12288, 4096):  ("v3", 128, 1),
    (4096,  4096,  4096):  ("8w", 256, 2, 48),
    (4096,  4096,  11008): ("8w", 256, 2, 32),
    (4096,  8192,  8192):  ("8w", 256, 2, 32),
    (4096,  11008, 4096):  ("8w", 256, 16, 48),
    (4096,  12288, 4096):  ("8w", 256, 1),
    (4096,  28672, 8192):  ("8w", 256, 32, 48),  # 0.904 (huge N, structural cap)
    (8192,  4096,  4096):  ("8w", 256, 2, 48),
    (8192,  4096,  11008): ("8w", 256, 2),
    (8192,  8192,  28672): ("8w", 256, 1, 16),
    (8192,  11008, 4096):  ("8w", 256, 16),
    (8192,  28672, 8192):  ("8w", 256, 32, 64),  # gm=32 unlocked 0.946
}


def _pick_tile_nt(M: int, N: int, K: int) -> tuple:
    """Returns ('8w'|'4w', ...) tile config for the shape, defaulting to
    8-wave BM=256 GM=1 if not in the per-shape table."""
    return _NT_TILE_OVERRIDE.get((M, N, K), _NT_DEFAULT_CFG)


# TN per-shape autotune. First-call benches small candidate set, caches by
# (M, N, K). Candidates use packed B inline asm (a_inline=0, b_inline=1):
# 1-ptr+offset:8192 ds_read_b64_tr_b8 + =&v + memory clobber + lgkmcnt drain.
# Verified 2026-05-28: spill 14->8, SNR all-K pass, TN/NT 0.735->0.804.
# A-side stays intrinsic (a_inline spills to 206).
_TN_CANDIDATES = [
    (256, 1, 32),
    (256, 2, 32),
    (256, 16, 32),
]
# NOTE: GM=8/32 candidates tested 2026-05-29 — no benefit (worst 28672 shapes
# stay 0.73; unlike NT, TN's big-N shapes are bound by the double wave-coop
# ds_read_b64_tr_b8, not L2 super-block reuse). Kept set minimal.
# BM=128 (inline_both / a_inline=1) candidates dropped 2026-05-30: spill-free
# but small-tile compute always loses to BM=256 b_inline on big-N/K shapes.
_TN_AUTOTUNE_CACHE: dict = {}


def _autotune_tn_dispatch(args, M, N, K):
    """First-call bench TN candidates, cache best (launch, cfg) by (M,N,K)."""
    import torch as _torch
    key = (M, N, K)
    if key in _TN_AUTOTUNE_CACHE:
        return _TN_AUTOTUNE_CACHE[key]
    out_view = args[2]
    best_us = float("inf"); best = None
    # 2026-05-30 STRICT det=0 audit: ALL barrier-dropped masks are intermittently
    # racy at the MFMA-reorder level (~1e-5, snr stays 55.6) under some
    # GM/occupancy — exposed only by long det gates:
    #   0x5A vh2 : det~5e-6  (was shipped; this is the BUG being fixed here)
    #   0x6A/0x6E/0x4E : raced as late as run 192 (8-run checks missed it)
    #   0x4A vh3 : 2731 TF (+8.8%), det=0 over 1000-run at GM∈{1,2} but GM=16
    #              raced at run 612 (1.2e-5)
    # The current loop's racy-ceiling (0x00, no barriers) is only ~2734 TF, so no
    # barrier config can reach the 2800 target anyway — that needs the A-side tr8
    # vmcnt(0) drain removed (raises the ceiling), not barrier removal. Until a
    # strict-det=0 drain-removal lands, ship ONLY the full-barrier 0x7F (robust
    # det=0 across all GM, 1000-run gated). See scripts2/_tn_e2e_det.py.
    for bm, gm, ag in _TN_CANDIDATES:
        if M % bm != 0:
            continue
        for _bmask in [0x7F]:
            try:
                # A-side stays intrinsic (a_inline @ BM=256 spills V to 350,
                # @ BM=128 no-spill but small-tile loses); B-side path-J only.
                launch = _compile_dense_tn(
                    K=K, BLOCK_M=bm, BLOCK_N=256, GROUP_M=gm,
                    agpr_alloc=ag, inline_asm_load=False,
                    a_inline_asm=0, b_inline_asm=1, barrier_mask=_bmask,
                )
                c = _get_compiled_dense(launch, args)
                c(*args); _torch.cuda.synchronize()
                sample = out_view.view(-1)[:1024].float()
                if not _torch.isfinite(sample).all().item():
                    continue
                for _ in range(2): c(*args)
                _torch.cuda.synchronize()
                e0 = _torch.cuda.Event(enable_timing=True); e1 = _torch.cuda.Event(enable_timing=True)
                _torch.cuda.synchronize(); e0.record()
                for _ in range(20): c(*args)
                e1.record(); _torch.cuda.synchronize()
                us = e0.elapsed_time(e1)*1000.0/20
                if us < best_us:
                    best_us = us; best = (launch, (bm, gm, ag, _bmask))
            except Exception:
                continue
    # 2D super-block (band) swizzle for big-N: when N is large the 1D GROUP_M
    # sweep re-streams all of B per M-group → L2 hit ~51% (vs ~66% on square
    # shapes), the dominant gap to the big-K ceiling (rocprof: latency-bound,
    # dep-wait 63%, VMEM 3% — NOT bandwidth). Tiling N into width-GN bands
    # blocks both operands into L2 (GM·A_slab + GN·B_slab resident). On
    # 8192x28672x8192 GM4×GN14 = 2706 TF vs 1D 2411 (+12%), det=0 over 1000
    # runs (pure tile permutation). GN≈n_blocks/8 (band count = #XCD) is the
    # sweet spot. Only helps when M is tall enough for GROUP_M banding to pay
    # (M small → 1D wins, so bench-gated). See scripts2/_tn_bigN_2d_confirm.py.
    n_blocks = (N + 255) // 256
    # Gate on M too: 2D banding only pays when there are enough M-blocks for
    # GROUP_M grouping (M small → too few groups, 1D wins). mpid>=16 (M>=4096).
    if n_blocks >= 32 and (M // 256) >= 16:
        for gm2 in (4, 2):
            for gn2 in (n_blocks // 8, n_blocks // 16):
                if gn2 < 2 or M % 256 != 0:
                    continue
                try:
                    launch = _compile_dense_tn(
                        K=K, BLOCK_M=256, BLOCK_N=256, GROUP_M=gm2,
                        agpr_alloc=32, inline_asm_load=False,
                        a_inline_asm=0, b_inline_asm=1, barrier_mask=0x7F,
                        group_n=gn2,
                    )
                    c = _get_compiled_dense(launch, args)
                    c(*args); _torch.cuda.synchronize()
                    sample = out_view.view(-1)[:1024].float()
                    if not _torch.isfinite(sample).all().item():
                        continue
                    for _ in range(2): c(*args)
                    _torch.cuda.synchronize()
                    e0 = _torch.cuda.Event(enable_timing=True); e1 = _torch.cuda.Event(enable_timing=True)
                    _torch.cuda.synchronize(); e0.record()
                    for _ in range(20): c(*args)
                    e1.record(); _torch.cuda.synchronize()
                    us = e0.elapsed_time(e1)*1000.0/20
                    if us < best_us:
                        best_us = us; best = (launch, (256, gm2, 32, 0x7F))
                except Exception:
                    continue
        # big-N inplace candidate: inline-A + asm_mma=2 → asm-inplace MFMA
        # (accum→AGPR, spill 18→0). Marginal over the intrinsic-A 2D-band path
        # (2657→2691, ~+1.3%), det0 over 3x2000 fresh-data. Bench picks if
        # fastest (no-op when 2D-band intrinsic-A wins).
        for gm2 in (4, 2):
            for gn2 in (n_blocks // 8, n_blocks // 16):
                if gn2 < 2 or M % 256 != 0:
                    continue
                try:
                    launch = _compile_dense_tn(
                        K=K, BLOCK_M=256, BLOCK_N=256, GROUP_M=gm2,
                        agpr_alloc=0, inline_asm_load=False,
                        a_inline_asm=1, b_inline_asm=1, barrier_mask=0x7F,
                        vmcnt_hint=3, asm_mma=2, group_n=gn2,
                    )
                    c = _get_compiled_dense(launch, args)
                    c(*args); _torch.cuda.synchronize()
                    if not _torch.isfinite(out_view.view(-1)[:1024].float()).all().item():
                        continue
                    for _ in range(2): c(*args)
                    _torch.cuda.synchronize()
                    e0 = _torch.cuda.Event(enable_timing=True); e1 = _torch.cuda.Event(enable_timing=True)
                    _torch.cuda.synchronize(); e0.record()
                    for _ in range(20): c(*args)
                    e1.record(); _torch.cuda.synchronize()
                    us = e0.elapsed_time(e1)*1000.0/20
                    if us < best_us:
                        best_us = us; best = (launch, (256, gm2, 0, 0x7F))
                except Exception:
                    continue
    # Big-K (K>=28672) drain-removal win config: BOTH A+B path-J (removes the
    # A-side tr8 vmcnt(0) drain the 0x7F comment above flags as the ceiling),
    # agpr_alloc=0 (compiler-decided AGPR; guard bypassed via asm_mma=2),
    # GROUP_M=1, vmcnt_hint=3, full-barrier 0x7F. Validated det=0 over 1000-run
    # (scripts2/_v3.py / _safe.py) = 2756 TF on 8192x8192x28672 (+6.8% vs the
    # A-intrinsic candidates above). vmcnt_hint=3 is the det=0 sweet spot:
    # vh=2 also det=0 but 2710-2734; vh>=4 RACES (det 1.5e-5 at 500-run). bench
    # picks this only if fastest (loses on N-major / non-big-K → not added there).
    # winK + (optional) 2D band: GROUP_M=1/group_n=0 is the original big-K win
    # (2768 TF); stacking the 2D band swizzle on top recovers the last L2 slack
    # even when it's already good — 8192x8192x28672 winK GM4×GN4 = 2799 TF
    # (+1.1%, hits the ~2800 ceiling), det=0 over 500-run. Benched, fastest wins.
    if K >= 28672 and M % 256 == 0:
        _wk = [(1, 0)]
        if n_blocks >= 32:
            _wk += [(4, n_blocks // 8), (4, n_blocks // 4)]
        for _wgm, _wgn in _wk:
            try:
                wlaunch = _compile_dense_tn(
                    K=K, BLOCK_M=256, BLOCK_N=256, GROUP_M=_wgm,
                    agpr_alloc=0, inline_asm_load=False,
                    a_inline_asm=1, b_inline_asm=1, barrier_mask=0x7F,
                    vmcnt_hint=3, asm_mma=2, group_n=_wgn,
                )
                c = _get_compiled_dense(wlaunch, args)
                c(*args); _torch.cuda.synchronize()
                sample = out_view.view(-1)[:1024].float()
                if not _torch.isfinite(sample).all().item():
                    continue
                for _ in range(2): c(*args)
                _torch.cuda.synchronize()
                e0 = _torch.cuda.Event(enable_timing=True); e1 = _torch.cuda.Event(enable_timing=True)
                _torch.cuda.synchronize(); e0.record()
                for _ in range(20): c(*args)
                e1.record(); _torch.cuda.synchronize()
                us = e0.elapsed_time(e1)*1000.0/20
                if us < best_us:
                    best_us = us; best = (wlaunch, (256, _wgm, 0, 0x7F))
            except Exception:
                continue
    if best is None:
        raise RuntimeError(f"TN autotune found no working cfg for ({M},{N},{K})")
    if os.environ.get("FLYDSL_TN_VERBOSE"):
        print(f"[TN autotune] ({M},{N},{K}) -> cfg(BM,GM,AG)={best[1]} us={best_us:.1f}", flush=True)
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
        # TN native: A [K, M], B [K, N]. Math C = A^T @ B.
        K_a, M = a.shape
        K_b, N = b.shape
        assert K_a == K_b, f"TN K mismatch: a {a.shape}, b {b.shape}"
        K = K_a
        if K % 128 != 0:
            raise NotImplementedError(
                f"FlyDSL dense GEMM requires K % 128 == 0 (BLOCK_K=128). Got K={K}."
            )
        a_scale_v = _broadcast_scale(a_scale_inv, M, a.device)
        b_scale_v = _broadcast_scale(b_scale_inv, N, a.device)
        out = torch.empty((M, N), dtype=out_dtype, device=a.device)
        # TN: intrinsic-only (path J inline_asm doesn't apply, see
        # commit 10470beb). Per-shape autotune over _TN_CANDIDATES picks
        # best (BM, GM, AGPR) for this shape, caches by (M,N,K).
        args = (
            _as_i8_flat(a), _as_i8_flat(b), out.contiguous().view(-1),
            a_scale_v, b_scale_v, M, N, torch.cuda.current_stream(),
        )
        launch, _cfg = _autotune_tn_dispatch(args, M, N, K)
        _get_compiled_dense(launch, args)(*args)
        if trans_c:
            return out.t().contiguous()
        return out

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
        cfg = _pick_tile_nn(M, N, K)
        # Two cfg tuple shapes supported:
        #   (BM, GM, AG)                — legacy 8-wave (most current overrides)
        #   ("8w"|"4w", BM, GM, AG[, ...]) — tagged kernel-kind
        if isinstance(cfg[0], str):
            kind = cfg[0]
            bm = cfg[1]; gm = cfg[2]; ag = cfg[3] if len(cfg) >= 4 else 0
        else:
            kind = "8w"
            bm = cfg[0]; gm = cfg[1]; ag = cfg[2]
        if kind == "8w":
            # path J: inline-asm ds_read_b64_tr_b8 ON by default.
            # Eliminates 446/447 compiler-auto vmcnt(0) drain per K-iter
            # honest 24-shape NN/NT geomean 0.897 → 0.96 (full-pull FlyDSL,
            # alternating-order bench). Worst K=28672 shape NN/NT 0.821 → 0.96+.
            launch = _compile_dense_nn(K=K, BLOCK_M=bm, BLOCK_N=256, GROUP_M=gm,
                                       agpr_alloc=ag, b_inline_asm_load=True,
                                       vmcnt_hint=2)
        else:
            raise ValueError(
                f"Unknown NN tile kind: {kind!r} (only '8w' supported; "
                f"4w dropped 2026-05-28)"
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
        _get_compiled_dense(launch, args)(*args)
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
        cfg = _pick_tile_nt(M, N, K)
        # 2026-05-28: 4wave kernel dropped (NT 4w + NN 4w both); v3 kept.
        kind = cfg[0]
        if kind == "8w":
            # cfg = ("8w", BM, GM) or ("8w", BM, GM, AGPR_ALLOC)
            bm = cfg[1]; gm = cfg[2]; ag = cfg[3] if len(cfg) >= 4 else 0
            launch = _compile_dense_nt(K=K, BLOCK_M=bm, BLOCK_N=256, GROUP_M=gm, agpr_alloc=ag)
        elif kind == "v3":
            _, bm, gm = cfg
            launch = _compile_dense_nt_v3(K=K, BLOCK_M=bm, BLOCK_N=256, GROUP_M=gm)
        else:
            raise ValueError(
                f"Unknown NT tile kind: {kind!r} (only '8w' / 'v3' supported; "
                f"4w dropped 2026-05-28)"
            )
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
