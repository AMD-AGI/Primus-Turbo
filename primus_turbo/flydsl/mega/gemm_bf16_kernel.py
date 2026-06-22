###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Primus-Turbo dense BF16 GEMM kernel (FlyDSL): NT, NN and TN layouts.

A faithful port of the hand-rolled fp8 software pipeline (gemm/gemm_fp8_kernel.py)
to bf16. The software-pipeline arrangement is UNCHANGED: 256x256 tile, 8-wave
(wave_m=2 x wave_n=4), the prelude double-buffer, the 4-quadrant main K-loop
(c00->c01->c10->c11 with interleaved G2S prefetch + s_barrier per MMA), and the
two epilogs. Only the dtype-specific bits change for bf16:

  * MFMA 16x16x128 (f8f6f4) -> Mfma32x32x16 (v_mfma_f32_32x32x16_bf16). Each
    k-iter still issues ONE mfma.call per quadrant; the call carries an internal
    NK_SUB (=BLOCK_K/16) K-subloop so the macro pipeline is untouched.
  * BLOCK_K = 64 bf16 = 128 BYTES, keeping the LDS row byte layout identical to
    fp8 (128-byte rows + swizzle_128) so G2SLoader is reused verbatim.
  * S2RLoader / S2RLoaderTr produce bf16 mfma operands (8 bf16 = Vec(8,bf16) per
    lane per K-subtile) instead of fp8 i32x8.
  * Per-tensor scale dropped (plain bf16); the store casts f32->bf16/fp16.
  * No K-tail mask: K must be a multiple of BLOCK_K (DSv3 shapes qualify).

Primitives are defined here (bf16-specific) and reuse the byte-agnostic fp8
helpers (G2SLoader, swizzle_128, ceildiv, xcd_remap_pid, wait_barrier,
make_value_attrs) from flydsl.utils.gemm_helper.
"""

import functools

import torch

# isort: off
from primus_turbo.flydsl.utils.gemm_helper import (
    G2SLoader,
    ceildiv,
    make_value_attrs,
    swizzle_128,
    wait_barrier,
    xcd_remap_pid,
    _lds_ptr_from_i32,
)
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir.dialects import fly as fly_dialect
from flydsl._mlir.dialects import llvm as _llvm
from flydsl._mlir import ir
from flydsl.expr import arith
from flydsl.expr import range_constexpr, rocdl
from flydsl.expr.typing import Vector as Vec
from flydsl.expr.arith import _to_raw as _raw

from primus_turbo.flydsl.common.tile_spec import _emit_if_then

# isort: on

BLOCK_K = 64  # bf16 elements per K-iter = 128 bytes (= fp8 byte row)
INST_K = 16  # MFMA 32x32x16 instruction K
NK_SUB = BLOCK_K // INST_K  # K-subtiles per k-iter inside one mfma.call


# ───────────────────────────────────────────────────────────────────────
# bf16 buffer tensor + global-load swizzles (byte-identical to the fp8 path,
# only the per-lane element offset is halved because bf16 = 2 bytes).
# ───────────────────────────────────────────────────────────────────────
def make_fp16_bf16_buffer_tensor(arg, bf16_ir_t=None):
    """bf16/fp16 buffer-resource view. arg is already a 2-byte tensor, so the
    descriptor adapts to its extent (max_size=False)."""
    return fx.rocdl.make_buffer_tensor(arg, max_size=False)


def compute_global_swizzle_bf16(lane_id, wave_id, K, n_rounds):
    """Per-lane global-load offsets for a [rows, K] bf16 K-contiguous operand.
    Mirrors compute_global_swizzle (non-preshuffled) in BYTES: 8 lanes cover one
    128-byte K-chunk (= 64 bf16), 8 rows/wave/round. The swizzled byte column is
    halved back to a bf16 element offset."""
    offsets = []
    n_waves = fx.block_dim.x // 64
    for r in range_constexpr(n_rounds):
        row = lane_id // 8 + wave_id * 8 + r * (n_waves * 8)
        col_byte = (lane_id % 8) * 16  # 16 bytes = 8 bf16 per lane
        _, c = swizzle_128(row, col_byte)
        offsets.append(row * K + c // 2)
    return offsets


# ───────────────────────────────────────────────────────────────────────
# NN/TN transpose-read primitives. A K-major operand ([K,N] for NN B; [K,M] and
# [K,N] for TN A/B) is brought in by a coalesced async G2S and transposed on the
# LDS read with ds_read_b64_tr_b16 (gfx950's tr16 intrinsic is not selectable, so
# inline asm). The G2S swizzle and the S2R loader are a co-designed matched pair.
# ───────────────────────────────────────────────────────────────────────
def compute_global_swizzle_nn_bf16(lane_id, wave_id, c_n, n_steps):
    """Per-lane gmem offsets for a K-major operand [K, N_or_M] (NN B, TN A/B).

    Matched to ``S2RLoaderTrBf16`` + the async G2S chunk layout: chunk
    (wave + step*8) holds sub-block (n_tile*4 + ks); each lane copies 8 contiguous
    elems (n) at (k=ks*16+kk, n=n_tile*32 + g*16 + (lane%2)*8), kk=(lane%32)//2,
    g=lane//32. Coalesced read; the 8-elem chunk lands transposed for tr16."""
    offsets = []
    n_waves = fx.block_dim.x // 64
    kk = (lane_id % 32) // 2
    g = lane_id // 32
    n_in = g * 16 + (lane_id % 2) * 8
    for step in range_constexpr(n_steps):
        idx = wave_id + step * n_waves
        n_tile = idx // 4
        ks = idx % 4
        offsets.append((ks * 16 + kk) * c_n + n_tile * 32 + n_in)
    return offsets


def _packed_ds_read_tr16(base_ptr, byte_offsets):
    """Pack N ds_read_b64_tr_b16 onto ONE shared base-ptr VGPR in a single inline-asm
    block (1 memory barrier + 1 addr VGPR vs N). Returns N x Vec(4,bf16)."""
    n = len(byte_offsets)
    v2i32 = ir.VectorType.get([2], ir.IntegerType.get_signless(32))
    struct_t = _llvm.StructType.get_literal([v2i32] * n)
    asm = "\n".join(f"ds_read_b64_tr_b16 ${k}, ${n} offset:{byte_offsets[k]}" for k in range(n))
    constraints = ",".join(["=&v"] * n + ["v"] + ["~{memory}"])
    op = _llvm.InlineAsmOp(
        res=struct_t,
        operands_=[_raw(base_ptr)],
        asm_string=asm,
        constraints=constraints,
        has_side_effects=True,
    )
    return [Vec(_llvm.extractvalue(v2i32, op.result, [k])).bitcast(fx.BFloat16) for k in range(n)]


class S2RLoaderTrBf16:
    """K-major LDS sub-block -> 32x32x16 bf16 operand via ds_read_b64_tr_b16.
    Reads the layout written by compute_global_swizzle_nn_bf16 + G2SLoader. Per
    lane per ks: 2 tr16 reads -> Vec(8,bf16) = operand[row, kblk*8 .. kblk*8+7]."""

    def __init__(self, wave_idx, n_tiles, sub_stride=512, nk_sub=NK_SUB):
        self.lane_id = fx.thread_idx.x % 64
        self.wave_idx = wave_idx
        self.n_tiles = n_tiles
        self.sub_stride = sub_stride  # bf16 per (tile,ks) sub-block (>=512; pad for bank-spread)
        self.nk_sub = nk_sub

    def load(self, lds_src):
        frag = []
        m = self.lane_id % 32
        kblk = self.lane_id // 32
        g = m // 16
        ml = m % 16
        base_i32 = fx.Int32(fx.ptrtoint(lds_src.ptr))
        for i in range_constexpr(self.n_tiles):
            tile = self.wave_idx * self.n_tiles + i
            subs = []
            for ks in range_constexpr(self.nk_sub):
                sub_base = (tile * self.nk_sub + ks) * self.sub_stride + g * 256 + kblk * 128 + ml * 4
                ptr = _lds_ptr_from_i32(base_i32 + sub_base * 2)
                r0, r1 = _packed_ds_read_tr16(ptr, [0, 64 * 2])
                subs.append(r0.shuffle(r1, list(range(8))))
            frag.append(subs)
        return frag


# ───────────────────────────────────────────────────────────────────────
# MFMA 32x32x16 bf16. accum = 16 f32 / lane; A,B operands = 8 bf16 / lane per
# K-subtile. call() keeps the fp8 "one call per quadrant" contract by looping
# NK_SUB K-subtiles internally.
# ───────────────────────────────────────────────────────────────────────
class Mfma32x32x16:
    # NOTE (reviewed 2026-06-18): an AGPR-accumulator variant (inline-asm
    # v_mfma_f32_32x32x16_bf16 "=a,v,v,0") was tried to cut arch-VGPR pressure
    # (218 VGPR, no spill). It was both INCORRECT (cos 0.93) and SLOWER
    # (504 vs 1340 TF). The default fly mma (VGPR accumulator) is the keeper.
    def __init__(self, n_tiles_a, n_tiles_b, nk_sub=NK_SUB):
        self.atom = fx.make_mma_atom(fx.rocdl.MFMA(32, 32, 16, fx.BFloat16))
        self.accum_type = Vec.make_type(16, fx.Float32)
        self.zero_value = Vec.filled(16, 0.0, fx.Float32)
        self.n_tiles_a = n_tiles_a
        self.n_tiles_b = n_tiles_b
        self.nk_sub = nk_sub

    def idx(self, i, j):
        return i * self.n_tiles_b + j

    def _do_mma(self, a, b, c):
        return fly_dialect.mma_atom_call_ssa([self.accum_type], self.atom, a, b, c)

    def call(self, a, b, c):
        # a: [n_tiles_a][nk_sub] Vec(8,bf16); b: [n_tiles_b][nk_sub] Vec(8,bf16)
        # c: [n_tiles_a*n_tiles_b] Vec(16,f32)
        assert len(a) == self.n_tiles_a
        assert len(b) == self.n_tiles_b
        for i in range_constexpr(self.n_tiles_a):
            for j in range_constexpr(self.n_tiles_b):
                acc = c[self.idx(i, j)]
                for ks in range_constexpr(self.nk_sub):
                    acc = self._do_mma(a[i][ks], b[j][ks], acc)
                c[self.idx(i, j)] = acc
        return c


# ───────────────────────────────────────────────────────────────────────
# Shared-to-register loaders. The LDS byte layout is the fp8 flat [rows, 128B]
# swizzle_128 layout; each lane pulls 8 contiguous bf16 (16 bytes) per K-subtile.
# ───────────────────────────────────────────────────────────────────────
def _load8_bf16(lds_src, byte_off):
    """Load 8 contiguous bf16 (16 bytes) from LDS at a byte offset -> Vec(8,bf16)."""
    i8 = fx.recast_iter(fx.Uint8, lds_src.ptr)
    p = fx.add_offset(i8, fx.make_int_tuple(byte_off))
    v = fx.make_view(p, fx.make_layout(16, 1)).load()  # 16 u8
    return v.bitcast(fx.BFloat16)  # 8 bf16


class S2RLoaderBf16:
    """Plain swizzled LDS -> mfma operand for a K-contiguous operand (NT A and B).
    Per lane: m = lane%32 (the operand row/col), kblk = lane//32 selects the upper
    or lower 8 K of each 16-K mfma subtile."""

    def __init__(self, wave_idx, n_tiles, nk_sub=NK_SUB):
        self.lane_id = fx.thread_idx.x % 64
        self.wave_idx = wave_idx
        self.n_tiles = n_tiles
        self.nk_sub = nk_sub

    def load(self, lds_src):
        frag = []
        m = self.lane_id % 32
        kblk = self.lane_id // 32
        for i in range_constexpr(self.n_tiles):
            row = self.wave_idx * (self.n_tiles * 32) + i * 32 + m
            subs = []
            for ks in range_constexpr(self.nk_sub):
                col_byte = (ks * 16 + kblk * 8) * 2  # bf16 K -> byte col
                _, cs = swizzle_128(row, col_byte)
                subs.append(_load8_bf16(lds_src, row * 128 + cs))
            frag.append(subs)
        return frag


# ───────────────────────────────────────────────────────────────────────
# Dispatch wrappers below only wire up NT for now (NN/TN added once NT validates).
# ───────────────────────────────────────────────────────────────────────
class StoreCBf16:
    """Cast + store the f32 accumulator (16 / lane per 32x32 tile) to bf16/fp16 C.
    Accumulator layout (v_mfma_f32_32x32x16): for acc index r in [0,16),
    n = lane%32, m = (r//4)*8 + (lane//32)*4 + (r%4). Columns past c_cols clamp
    to an OOB index (HW drops the store)."""

    def __init__(self, C, c_rows, c_cols, out_ty):
        self.c_rows = c_rows
        self.c_cols = c_cols
        self.lane_id = fx.thread_idx.x % 64
        self.out_ty = out_ty
        c_nbytes = c_rows * c_cols * 2
        gC = fx.rocdl.make_buffer_tensor(C, max_size=False, num_records_bytes=c_nbytes)
        self.c_div = fx.logical_divide(gC, fx.make_layout(1, 1))
        self.out_atom_1 = fx.make_copy_atom(fx.rocdl.BufferCopy16b(), out_ty)
        self.reg_out_1 = fx.make_rmem_tensor(fx.make_layout(1, 1), out_ty)

    def _store_one(self, value, c_index):
        fx.memref_store_vec(Vec.filled(1, value, self.out_ty), self.reg_out_1)
        fx.copy(self.out_atom_1, self.reg_out_1, fx.slice(self.c_div, (None, fx.Int32(c_index))))

    def store(self, c_frag, base_row, base_col):
        oob = fx.Int32(self.c_rows * self.c_cols)
        n = self.lane_id % 32
        m_hi = (self.lane_id // 32) * 4
        col = base_col + n
        col_valid = col < self.c_cols
        for ti in range_constexpr(len(c_frag)):
            acc = Vec(c_frag[ti])
            for r in range_constexpr(16):
                row = base_row + ti * 32 + (r // 4) * 8 + m_hi + (r % 4)
                c_index = row * self.c_cols + col
                self._store_one(acc[r].to(self.out_ty), arith.select(col_valid, c_index, oob))


# ───────────────────────────────────────────────────────────────────────
# Tile geometry + shared storage (derived from the dense bf16 kernel). Shared by
# the dense launcher here and the fused dispatch / combine grouped GEMM kernels.
# ───────────────────────────────────────────────────────────────────────
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
    """Double-buffered A/B LDS struct (byte layout identical to the dense kernel)."""
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


# ───────────────────────────────────────────────────────────────────────
# gemm_bf16_nt_tile — the dense bf16 NT software pipeline, for ONE tile
# (block_m, block_n). Core code copied VERBATIM from ``kernel_dense_nt`` below so
# the emitted IR is identical; ``kernel_dense_nt`` and the fused dispatch / combine
# kernels all build on this single closure. ``b_group_base`` (elements) shifts the
# B operand to the per-expert weight slab (grouped GEMM); None for the dense case.
# Runtime control flow uses _emit_if_then because this is a plain helper (no
# @flyc.kernel AST rewrite here).
# ───────────────────────────────────────────────────────────────────────
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

    # Dense self-schedule (block_m/block_n omitted): the XCD-swizzle + GROUP_M map.
    # Emitted here (after the wave-ids) so the dense kernel's IR matches kernel_dense_nt
    # exactly. The fused dispatch/combine kernels pass their own block_m/block_n.
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
    store_c = StoreCBf16(C, c_m, c_n, _out_ty)

    c00_frag = [mfma.zero_value] * N_ACCUMS
    c01_frag = [mfma.zero_value] * N_ACCUMS
    c10_frag = [mfma.zero_value] * N_ACCUMS
    c11_frag = [mfma.zero_value] * N_ACCUMS

    # Prelude.
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

    # Main K-loop (4-quadrant, mirrors fp8 NT exactly).
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

    # Epilog 2 (last K-iter; K % BLOCK_K == 0 so no A-tail mask).
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

    # Store.
    wave_n_offset = wave_n * (N_TILES_B * 32)
    wave_m_offset = wave_m * (N_TILES_A * 32)
    base_row = block_m * BLOCK_M + wave_m_offset
    base_col = block_n * BLOCK_N + wave_n_offset

    store_c.store(c00_frag, base_row + 0, base_col + 0)
    store_c.store(c01_frag, base_row + 0, base_col + LDS_BLOCK_N)
    store_c.store(c10_frag, base_row + LDS_BLOCK_M, base_col + 0)
    store_c.store(c11_frag, base_row + LDS_BLOCK_M, base_col + LDS_BLOCK_N)


# ───────────────────────────────────────────────────────────────────────
# NN / TN bf16 tiles. Same 4-quadrant pipeline as NT; the only change is the
# K-major operand path: coalesced async G2S (compute_global_swizzle_nn_bf16) +
# tr16 transpose-read S2R (S2RLoaderTrBf16). NN transposes B only (A = NT); TN
# transposes both A and B. Perf on DSv3: NN ~99%, TN ~95% of NT (TN's gap is the
# gfx950 tr16 b64 read being half-width vs NT's b128, i.e. 2x the LDS-read slots).
# ───────────────────────────────────────────────────────────────────────
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

    # B is always K-major [K,N]; A is K-major [K,M] for TN, M-major [M,K] for NN.
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
    if b_group_base is not None:  # grouped GEMM: shift B to this expert's [K,N] block
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
    store_c = StoreCBf16(C, c_m, c_n, _out_ty)

    c00_frag = [mfma.zero_value] * N_ACCUMS
    c01_frag = [mfma.zero_value] * N_ACCUMS
    c10_frag = [mfma.zero_value] * N_ACCUMS
    c11_frag = [mfma.zero_value] * N_ACCUMS

    # Prelude.
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

    # Main K-loop (4-quadrant).
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

    # Epilog 2 (last K-iter; K % BLOCK_K == 0).
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
):
    """NN tile: A [M,K] (NT), B [K,N] (transpose-read). C [M,N] = A @ B."""
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
    """TN tile: A [K,M], B [K,N] (both transpose-read). C [M,N] = A^T @ B."""
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


# ───────────────────────────────────────────────────────────────────────
# NT-layout bf16 dense kernel: A [M,K], B_T [N,K] (= B^T of [K,N]), C [M,N].
# Software pipeline arrangement copied verbatim from the fp8 NT kernel.
# ───────────────────────────────────────────────────────────────────────
@functools.lru_cache(maxsize=256)
def _compile_dense_nt(
    K: int,
    BLOCK_M: int = 256,
    BLOCK_N: int = 256,
    GROUP_M: int = 1,
    waves_per_eu: int = 2,
    agpr_alloc: int = 0,
    nt_vmcnt: int = 3,
    num_xcd: int = 8,
    out_fp16: bool = False,
):
    assert BLOCK_M >= 128 and BLOCK_N >= 256 and BLOCK_M % 128 == 0 and BLOCK_N % 256 == 0
    assert GROUP_M >= 1
    # geometry / K-tile validation lives in _gemm_geom (used by gemm_bf16_nt_tile).
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


# ───────────────────────────────────────────────────────────────────────
# Host dispatch (NT only for now).
# ───────────────────────────────────────────────────────────────────────
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
    """Dense bf16 GEMM, hand-rolled fp8-style pipeline. NT/NN/TN (tr16 transpose-read
    for the K-major operands; NN ~99%, TN ~95% of NT on DSv3 shapes)."""
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


# ───────────────────────────────────────────────────────────────────────
# NN / TN dense launchers + host entry points (share gemm_bf16_nn/tn_tile).
# ───────────────────────────────────────────────────────────────────────
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
    """Compile a dense NN (a_transpose=False) or TN (a_transpose=True) bf16 launcher."""
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
    a: torch.Tensor,  # [M, K] bf16
    b: torch.Tensor,  # [K, N] bf16
    out_dtype: torch.dtype = torch.bfloat16,
    BLOCK_M: int = 256,
    GROUP_M: int = 1,
    num_xcd: int = 8,
) -> torch.Tensor:
    """Dense bf16 NN GEMM: C[M,N] = A[M,K] @ B[K,N]. B transpose-read (tr16)."""
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
    a: torch.Tensor,  # [K, M] bf16
    b: torch.Tensor,  # [K, N] bf16
    out_dtype: torch.dtype = torch.bfloat16,
    BLOCK_M: int = 256,
    GROUP_M: int = 1,
    num_xcd: int = 8,
) -> torch.Tensor:
    """Dense bf16 TN GEMM: C[M,N] = A[K,M]^T @ B[K,N]. Both operands transpose-read."""
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
