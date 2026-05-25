# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""8-wave FP8 NN-layout matmul (sibling of fp8_gemm_8wave).

Computes C[M,N] = A[M,K] @ B[K,N] where both A and B are row-major fp8:
  - A is K-contig (same as the NT kernel's A path)
  - B is N-contig per K-row (contraction sweeps DOWN B's rows)

Used by Primus-Turbo gemm_fp8 dgrad. The contraction direction of B differs
from the NT kernel's [N, K] storage, so the global swizzle for B is rewritten
and the LDS->register path eventually uses ds_read_tr8_b64 (added in M1+).

Milestone status:
    M0 (this file) -- skeleton: compiles, B g2s uses NN swizzle, B s2r is a
                      *stub* reusing the NT S2RLoader so output is garbage but
                      the kernel can be JIT'd end-to-end. Verify with FLYDSL
                      IR dump / dispatch-only smoke.
    M1 -- replace B s2r with ds_read_tr8_b64 + SSA swap_pair (correctness probe).
    M2+ -- enable full pipeline and per-K bring-up.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm as _llvm
from flydsl.expr import const_expr, range_constexpr, rocdl
from flydsl.expr.arith import _to_raw as _raw
from flydsl.expr.typing import T, Vector as Vec
from flydsl.expr.utils.arith import ArithValue

from kernels.fp8_gemm_utils import (
    G2SLoader,
    Mfma16x16x128,
    S2RLoader,
    StoreC,
    ceildiv,
    compute_global_swizzle,
    divmod,
    make_fp8_buffer_tensor,
    swizzle_128,
    wait_barrier,
)


# Session 3 (HW B-transpose) — 4× ds_read_tr8_b64, 0 ds_bpermute.
# Replaces the NT-style S2RLoader stub for B. See
# scripts2/NN_TRANSPOSE_DESIGN.md + scripts2/probe_transpose_nn_lds.py
# (probe PASS 2026-05-25: 0/32768 mismatches across K- and N-pattern).
#
# Layout assumption (matches G2SLoader + compute_global_swizzle_nn +
# BufferCopyLDS128b on a [BLOCK_K=128, LDS_BLOCK_N=128] B tile):
#   LDS holds per-(wave_writer, step) 1024-byte blocks: lane L_writer in
#   wave W_writer step r writes the GMEM data it loaded into
#   LDS[W_writer*1024 + r*8192 + L_writer*16 + 0..15], where its read
#   came from B[K=L_writer//8 + W_writer*8 + r*64,
#               N=(L_writer%8)*16 XOR swz_K..+15].
#
# Per-lane ptr derivation for the 4-call transpose (I = lane//16,
# L_in_sg = lane%16, K_BASE = [0, 8, 64, 72]):
#   K_log = I*16 + K_BASE[c] + L_in_sg//2          ∈ [0, 128)
#   r_step = K_log // 64                            ∈ {0, 1}
#   W      = (K_log % 64) // 8                     ∈ [0, 8)
#   K_mod_8 = K_log % 8                            ∈ [0, 8)
#   swz_K  = ((K_log % 16) // 2) * 16              ∈ {0,16,32,...,112}
#   tile_N_start = wave_n*32 + tile_i*16           target mfma N
#   j_chunk = (tile_N_start // 16) ^ (swz_K // 16)  un-swizzled N chunk
#   ptr = LDS + W*1024 + r_step*8192 + K_mod_8*128
#             + j_chunk*16 + (L_in_sg%2)*8
_LDS_PTR_TYPE = None


def _lds_ptr_from_i32(addr_i32):
    global _LDS_PTR_TYPE
    if _LDS_PTR_TYPE is None:
        _LDS_PTR_TYPE = ir.Type.parse("!llvm.ptr<3>")
    return _llvm.inttoptr(_LDS_PTR_TYPE, _raw(ArithValue(addr_i32).extui(T.i64)))


class S2RLoaderTrV2:
    """Transpose-load B via 4× ds_read_tr8_b64 (Option A, 0 bpermute).

    API mirrors :class:`S2RLoader`: ``load(lds_src, preshuffled=False)`` ->
    ``list[v8i32]`` of length ``n_tiles``. Each ``v8i32`` is the mfma B
    fragment for tile ``tile_i`` (covers 16 N-cols at K=[0,128)).
    """

    _K_BASE = (0, 8, 64, 72)

    def __init__(self, wave_n, n_tiles):
        self.wave_n = wave_n
        self.n_tiles = n_tiles
        self.lane = fx.thread_idx.x % 64

    def load(self, lds_src, preshuffled=False):
        assert not preshuffled, "S2RLoaderTrV2 does not support preshuffled"
        base_i32 = fx.Int32(fx.ptrtoint(lds_src.ptr))
        I = self.lane // 16
        L_in_sg = self.lane % 16
        v2i32_t = Vec.make_type(2, fx.Int32)

        frags = []
        for tile_i in range_constexpr(self.n_tiles):
            calls = []
            for c in range_constexpr(4):
                K_log = I * 16 + S2RLoaderTrV2._K_BASE[c] + (L_in_sg // 2)
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
                v = rocdl.ds_read_tr8_b64(v2i32_t, ptr).result
                calls.append(Vec(v))

            # Concat 4 × v2i32 → v8i32. Byte order:
            #   v8i32[0..1] = call 0 → mfma bytes 0..7   (K=16I+0..7)
            #   v8i32[2..3] = call 1 → mfma bytes 8..15  (K=16I+8..15)
            #   v8i32[4..5] = call 2 → mfma bytes 16..23 (K=16I+64..71)
            #   v8i32[6..7] = call 3 → mfma bytes 24..31 (K=16I+72..79)
            v4_lo = calls[0].shuffle(calls[1], [0, 1, 2, 3])
            v4_hi = calls[2].shuffle(calls[3], [0, 1, 2, 3])
            frags.append(v4_lo.shuffle(v4_hi, list(range(8))))
        return frags


# ──────────────────────────────────────────────────────────────────────────────
# NN-specific global swizzle for B.
#
# B is [K_inner, N_out] row-major. Per WG we load a [BLOCK_K=128, LDS_BLOCK_N=128]
# tile (one half-N of the BLOCK_N=256 tile). Each lane handles 16 fp8 bytes
# along N (BufferCopyLDS128b lane width); waves cover 8 K-rows each; rounds
# stack 8*n_waves = 64 K-rows per round, n_rounds=2 covers 128 K-rows.
#
# Offset within the [K, N] flat byte view:
#   k_row * N_out + n_col
# Apply swizzle_128 on (k_row, n_col) to stagger the LDS write banks (same
# 16-row XOR pattern the NT kernel uses for its A side).
# ──────────────────────────────────────────────────────────────────────────────
def compute_global_swizzle_nn(lane_id, wave_id, N_out, n_rounds):
    offsets = []
    n_waves = fx.block_dim.x // 64
    for r in range_constexpr(n_rounds):
        k_row = lane_id // 8 + wave_id * 8 + r * (n_waves * 8)
        n_col = (lane_id % 8) * 16
        rs, cs = swizzle_128(k_row, n_col)
        offsets.append(rs * N_out + cs)
    return offsets


def compile_fp8_gemm_8w_nn(*, K: int, BLOCK_M: int = 256, BLOCK_N: int = 256):
    BLOCK_K = 128

    assert BLOCK_M >= 128 and BLOCK_N >= 256 and BLOCK_M % 128 == 0 and BLOCK_N % 256 == 0
    assert K % BLOCK_K == 0

    K_ITERS = K // BLOCK_K

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
    b_lds_size = LDS_BLOCK_N * BLOCK_K  # same byte count as NT, just different layout

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
    def kernel_gemm_nn(
        A: fx.Tensor,
        B: fx.Tensor,
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
        block_m, block_n = divmod(fx.block_idx.x, n_blocks)

        # === A path: unchanged from NT (A is [M, K] K-contig) ===
        A0_gl_offset = (block_m * BLOCK_M) * K
        A1_gl_offset = (block_m * BLOCK_M + LDS_BLOCK_M) * K

        # === B path: NN-specific ===
        # B is [K, N] row-major (stride c_n). Per WG we load BLOCK_K=128 K-rows ×
        # BLOCK_N=256 N-cols, split into 2 N-halves (LDS_BLOCK_N=128 each).
        # K-iter step advances K-rows by BLOCK_K, which in byte/element units is
        # BLOCK_K * c_n.
        B_K_STEP = BLOCK_K  # element stride per K-iter; G2SLoader sees this × c_n
        # Base byte offset to (K=0, N_start=block_n*BLOCK_N) for B-half 0 / 1.
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
        # Session 3: HW B-transpose via 4× ds_read_tr8_b64 (probe-verified
        # ptr formula, see S2RLoaderTrV2 docstring above).
        b_s2r = S2RLoaderTrV2(wave_n, N_TILES_B)

        store_c = StoreC(A_scale, B_scale, C, c_m, c_n, mfma.idx, N_TILES_A, N_TILES_B)

        c00_frag = [mfma.zero_value] * N_ACCUMS
        c01_frag = [mfma.zero_value] * N_ACCUMS
        c10_frag = [mfma.zero_value] * N_ACCUMS
        c11_frag = [mfma.zero_value] * N_ACCUMS

        # Prelude: load k=0 and k=1 tiles into cur/next slots.
        # For B in NN, the per-iter byte step is BLOCK_K K-rows × c_n stride.
        b_g2s.load(b_cur0, B0_gl_offset + 0 * B_K_STEP * c_n)
        a_g2s.load(a_cur0, A0_gl_offset + 0 * BLOCK_K)
        b_g2s.load(b_cur1, B1_gl_offset + 0 * B_K_STEP * c_n)
        a_g2s.load(a_cur1, A1_gl_offset + 0 * BLOCK_K)

        if wave_m == 1:
            rocdl.s_barrier()

        wait_barrier(N_LDS_STEPS_A + N_LDS_STEPS_B)

        b_g2s.load(b_next0, B0_gl_offset + 1 * B_K_STEP * c_n)
        a_g2s.load(a_next0, A0_gl_offset + 1 * BLOCK_K)
        b_g2s.load(b_next1, B1_gl_offset + 1 * B_K_STEP * c_n)

        wait_barrier(N_LDS_STEPS_A + 2 * N_LDS_STEPS_B)

        for k in range_constexpr(K_ITERS - 2):
            b0_frag = b_s2r.load(b_cur0, preshuffled=False)
            a0_frag = a_s2r.load(a_cur0)
            a_g2s.load(a_next1, A1_gl_offset + (k + 1) * BLOCK_K)
            rocdl.s_barrier()

            rocdl.s_setprio(1)
            c00_frag = mfma.call(a0_frag, b0_frag, c00_frag)
            rocdl.s_setprio(0)
            rocdl.s_barrier()

            b1_frag = b_s2r.load(b_cur1, preshuffled=False)
            b_g2s.load(b_cur0, B0_gl_offset + (k + 2) * B_K_STEP * c_n)
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

            b_g2s.load(b_cur1, B1_gl_offset + (k + 2) * B_K_STEP * c_n)
            wait_barrier(2 * N_LDS_STEPS_A + N_LDS_STEPS_B)

            rocdl.s_setprio(1)
            c11_frag = mfma.call(a1_frag, b1_frag, c11_frag)
            rocdl.s_setprio(0)
            rocdl.s_barrier()

            a_cur0, a_next0 = a_next0, a_cur0
            a_cur1, a_next1 = a_next1, a_cur1
            b_cur0, b_next0 = b_next0, b_cur0
            b_cur1, b_next1 = b_next1, b_cur1

        # Epilog 1 (k = K_ITERS - 2) — inherits the c10/c11 stale-a1 fix.
        k = K_ITERS - 2
        b0_frag = b_s2r.load(b_cur0, preshuffled=False)
        a0_frag = a_s2r.load(a_cur0)
        rocdl.s_barrier()

        rocdl.s_setprio(1)
        c00_frag = mfma.call(a0_frag, b0_frag, c00_frag)
        rocdl.s_setprio(0)
        rocdl.s_barrier()

        b1_frag = b_s2r.load(b_cur1, preshuffled=False)
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

        b0_frag = b_s2r.load(b_next0, preshuffled=False)
        # Stale-a1 fix (same bug as NT kernel): make sure a_next1 holds k+1 data
        # before the swap so epilog 2's a1_frag reads fresh values.
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

        # Epilog 2 (k = K_ITERS - 1)
        a0_frag = a_s2r.load(a_cur0)
        wait_barrier(0)

        rocdl.s_setprio(1)
        c00_frag = mfma.call(a0_frag, b0_frag, c00_frag)
        rocdl.s_setprio(0)
        rocdl.s_barrier()

        b1_frag = b_s2r.load(b_cur1, preshuffled=False)
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
    def launch_gemm_nn(
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
        kernel_gemm_nn(
            A,
            B,
            C,
            A_scale,
            B_scale,
            c_m,
            c_n,
            value_attrs={"rocdl.waves_per_eu": 2, "rocdl.flat_work_group_size": "512,512"},
        ).launch(grid=(grid_x, 1, 1), block=(512, 1, 1), stream=stream)

    return launch_gemm_nn
