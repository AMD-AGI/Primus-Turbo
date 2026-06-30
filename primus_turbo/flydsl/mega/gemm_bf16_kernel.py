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
import os

import torch

# isort: off
from primus_turbo.flydsl.utils.gemm_helper import (
    G2SLoader,
    ceildiv,
    make_bf16_buffer_tensor_rebased,
    make_value_attrs,
    swizzle_128,
    wait_barrier,
    xcd_remap_pid,
    _lds_ptr_from_i32,
)
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr.buffer_ops import buffer_store, create_buffer_resource
from flydsl._mlir.dialects import fly as fly_dialect
from flydsl._mlir.dialects import llvm as _llvm
from flydsl._mlir import ir
from flydsl.expr import arith
from flydsl.expr import const_expr, range_constexpr, rocdl
from flydsl.expr.typing import Vector as Vec
from flydsl.expr.arith import _to_raw as _raw
from flydsl.expr.utils.arith import ArithValue

from primus_turbo.flydsl.common.tile_spec import _emit_for, _emit_if_then

# isort: on

BLOCK_K = 64  # bf16 elements per K-iter = 128 bytes (= fp8 byte row)
INST_K = 16  # MFMA 32x32x16 instruction K
NK_SUB = BLOCK_K // INST_K  # K-subtiles per k-iter inside one mfma.call
INST_K32 = 32  # MFMA 16x16x32 instruction K (gfx950) — wgrad TN K=32 path
NK_SUB32 = BLOCK_K // INST_K32  # = 2 K=32 subtiles per BLOCK_K=64 k-iter

# TN wgrad warp count. Round-2 lever-1: 4 waves (down from 8) doubles the per-wave
# output tile (8x8 16-tiles) so each transpose-read fragment feeds 2x more MFMAs,
# raising MFMA/LDS-read 1.33 -> 2.0 to attack the LDS issue-port bound. Env override
# MEGA_WGRAD_WAVES=8 restores the round-1 ship.
WGRAD_WAVES = int(os.environ.get("MEGA_WGRAD_WAVES", "8"))
WGRAD_BLOCK = WGRAD_WAVES * 64  # threads per WG for the wgrad kernel
# Allow compiler to place spilled accumulators in AGPR (occ-1: 256 VGPR + 256 AGPR).
# Negative => 'allow up to N' hint (NOT the disproven hand-rolled AGPR-pin engine).
WGRAD_AGPR = int(os.environ.get("MEGA_WGRAD_AGPR", "0"))


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

    def load(self, lds_src, base_off=0):
        # base_off (bytes) selects the LDS double-buffer stage (matches G2SLoader).
        frag = []
        m = self.lane_id % 32
        kblk = self.lane_id // 32
        g = m // 16
        ml = m % 16
        base_i32 = fx.Int32(fx.ptrtoint(lds_src.ptr)) + base_off
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
# MFMA 16x16x32 bf16 (gfx950). accum = 4 f32 / lane; A,B operands = 8 bf16 / lane
# per K=32 subtile. Derived for the TN wgrad path only. Acc lane layout:
#   c[v] = C[m = (lane//16)*4 + v, n = lane%16], v in 0..3.
# Operand lane layout (canonical CDNA mfma_16x16x32):
#   a[v] = A[m = lane%16, k = (lane//16)*8 + v]; b[v] = B[k=(lane//16)*8+v, n=lane%16].
# call() loops nk_sub (=BLOCK_K/32) K=32 subtiles internally, mirroring the 32x32
# "one call per quadrant" contract.
# ───────────────────────────────────────────────────────────────────────
class Mfma16x16x32:
    def __init__(self, n_tiles_a, n_tiles_b, nk_sub=NK_SUB32):
        self.atom = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 32, fx.BFloat16))
        self.accum_type = Vec.make_type(4, fx.Float32)
        self.zero_value = Vec.filled(4, 0.0, fx.Float32)
        self.n_tiles_a = n_tiles_a  # number of 16-row A m-tiles
        self.n_tiles_b = n_tiles_b  # number of 16-col B n-tiles
        self.nk_sub = nk_sub

    def idx(self, i, j):
        return i * self.n_tiles_b + j

    def _do_mma(self, a, b, c):
        return fly_dialect.mma_atom_call_ssa([self.accum_type], self.atom, a, b, c)

    def call(self, a, b, c):
        # a: [n_tiles_a][nk_sub] Vec(8,bf16); b: [n_tiles_b][nk_sub] Vec(8,bf16)
        # c: [n_tiles_a*n_tiles_b] Vec(4,f32)
        assert len(a) == self.n_tiles_a
        assert len(b) == self.n_tiles_b
        for i in range_constexpr(self.n_tiles_a):
            for j in range_constexpr(self.n_tiles_b):
                acc = c[self.idx(i, j)]
                for ks in range_constexpr(self.nk_sub):
                    acc = self._do_mma(a[i][ks], b[j][ks], acc)
                c[self.idx(i, j)] = acc
        return c


class S2RLoaderTr16x32Bf16:
    """K-major LDS sub-block -> 16x16x32 bf16 operand via ds_read_b64_tr_b16.

    Reuses the SAME LDS byte layout written by compute_global_swizzle_nn_bf16 +
    G2SLoader (16-K x 32-N sub-blocks, sub_stride bf16 each, 4 sub-blocks per
    BLOCK_K=64). A K=32 subtile = two adjacent 16-K sub-blocks. Each 32-N sub-block
    feeds TWO 16-row operand m-tiles (g16 = 0 / 1). Output (per 16-row m-tile, per
    K=32 subtile) is Vec(8,bf16): for lane L, a[v] = A[m=L%16, k=(L//16)*8+v].

    n_tiles32 = number of 16-row m-tiles owned by this wave = 2 * (32x32 n_tiles).
    """

    def __init__(self, wave_idx, n_tiles32, sub_stride=512, nk_sub=NK_SUB32):
        self.lane_id = fx.thread_idx.x % 64
        self.wave_idx = wave_idx
        self.n_tiles32 = n_tiles32  # 16-row m-tiles for this wave
        self.sub_stride = sub_stride
        self.nk_sub = nk_sub

    def load(self, lds_src, base_off=0):
        frag = []
        octet = self.lane_id // 16  # 0..3 -> k-octet within the K=32 subtile
        mm = self.lane_id % 16  # output row within the 16-tile
        base_i32 = fx.Int32(fx.ptrtoint(lds_src.ptr)) + base_off
        # octet -> (which 16-K sub-block of the K=32 pair, kblk half)
        s_in_pair = octet // 2  # 0 or 1: first/second 16-K sub-block
        kb = octet % 2  # 0 or 1: lower/upper 8 K of that 16-K sub-block
        for i in range_constexpr(self.n_tiles32):
            # m-tile i: original 32-N sub-block = i//2, row-half g16 = i%2.
            orig_tile = self.wave_idx * self.n_tiles32 + i
            n32_block = orig_tile // 2  # 32-N sub-block index
            g16 = orig_tile % 2  # 0 -> rows 0..15, 1 -> rows 16..31 of the 32-N block
            subs = []
            for ks in range_constexpr(self.nk_sub):
                # the two 16-K sub-blocks of this K=32 subtile
                sub16 = (n32_block * NK_SUB + ks * 2) + s_in_pair
                sub_base = sub16 * self.sub_stride + g16 * 256 + kb * 128 + mm * 4
                ptr = _lds_ptr_from_i32(base_i32 + sub_base * 2)
                r0, r1 = _packed_ds_read_tr16(ptr, [0, 64 * 2])
                subs.append(r0.shuffle(r1, list(range(8))))
            frag.append(subs)
        return frag


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

    def __init__(self, C, c_rows, c_cols, out_ty, cache_modifier=0):
        self.c_rows = c_rows
        self.c_cols = c_cols
        self.lane_id = fx.thread_idx.x % 64
        self.out_ty = out_ty
        self.cache_modifier = cache_modifier  # nonzero -> write-through C store (cross-CU L2Y)
        c_nbytes = c_rows * c_cols * 2
        gC = fx.rocdl.make_buffer_tensor(C, max_size=False, num_records_bytes=c_nbytes)
        self.c_div = fx.logical_divide(gC, fx.make_layout(1, 1))
        self.out_atom_1 = fx.make_copy_atom(fx.rocdl.BufferCopy16b(), out_ty)
        self.reg_out_1 = fx.make_rmem_tensor(fx.make_layout(1, 1), out_ty)
        # write-through path: buffer_store with a cache modifier (the copy atom can't carry one)
        self.c_rsrc = (
            create_buffer_resource(C, max_size=False, num_records_bytes=c_nbytes) if cache_modifier else None
        )

    def _store_one(self, value, c_index):
        if self.cache_modifier:
            buffer_store(value, self.c_rsrc, fx.Int32(c_index), cache_modifier=self.cache_modifier)
        else:
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

    def store_trans(self, c_frag, group_idx, base_m, base_n, out_m, out_n):
        """Transposed store: result tile (local m, local n) -> out[group, n, m] in a
        [G*out_n, out_m] buffer (index (group*out_n + n)*out_m + m). The mma lane owns
        one n (=lane%32) and 16 m (the transposed-contiguous dim). Coalescing comes from
        the block_m-fastest tile schedule (neighbouring tiles write contiguous transposed
        rows); a per-tile LDS CShuffle was tried and is perf-neutral, so kept scalar."""
        oob = fx.Int32(self.c_rows * self.c_cols)
        n = self.lane_id % 32
        m_hi = (self.lane_id // 32) * 4
        glob_n = base_n + n
        n_valid = glob_n < out_n
        row_base = (group_idx * out_n + glob_n) * out_m
        for ti in range_constexpr(len(c_frag)):
            acc = Vec(c_frag[ti])
            for r in range_constexpr(16):
                m = base_m + ti * 32 + (r // 4) * 8 + m_hi + (r % 4)
                c_index = row_base + m
                self._store_one(acc[r].to(self.out_ty), arith.select(n_valid, c_index, oob))

    def store16(self, c_frag, base_row, base_col):
        """Coalesced store for the 16x16x32 acc layout (4 f32/lane).
        c_frag is a list of 16x16 m-tiles; tile t covers output rows
        [base_row + t*16 .. +15], cols [base_col .. base_col+15].
        For lane L: n = base_col + L%16, m = base_row + t*16 + (L//16)*4 + v."""
        oob = fx.Int32(self.c_rows * self.c_cols)
        n = self.lane_id % 16
        m_hi = (self.lane_id // 16) * 4
        col = base_col + n
        col_valid = col < self.c_cols
        for ti in range_constexpr(len(c_frag)):
            acc = Vec(c_frag[ti])
            for r in range_constexpr(4):
                row = base_row + ti * 16 + m_hi + r
                c_index = row * self.c_cols + col
                self._store_one(acc[r].to(self.out_ty), arith.select(col_valid, c_index, oob))

    def store_trans16(self, c_frag, group_idx, base_m, base_n, out_m, out_n):
        """Transposed store for 16x16x32 acc -> out[group, n, m] in [G*out_n, out_m].
        Lane L owns n = base_n + L%16 and 4 m = base_m + ti*16 + (L//16)*4 + r."""
        oob = fx.Int32(self.c_rows * self.c_cols)
        n = self.lane_id % 16
        m_hi = (self.lane_id // 16) * 4
        glob_n = base_n + n
        n_valid = glob_n < out_n
        row_base = (group_idx * out_n + glob_n) * out_m
        for ti in range_constexpr(len(c_frag)):
            acc = Vec(c_frag[ti])
            for r in range_constexpr(4):
                m = base_m + ti * 16 + m_hi + r
                c_index = row_base + m
                self._store_one(acc[r].to(self.out_ty), arith.select(n_valid, c_index, oob))


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
    store_c = StoreCBf16(C, c_m, c_n, _out_ty, cache_modifier=c_cache_modifier)

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
    store_c = StoreCBf16(C, c_m, c_n, _out_ty, cache_modifier=c_cache_modifier)

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
    c_cache_modifier=0,
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


# ───────────────────────────────────────────────────────────────────────
# TN wgrad (variable-K grouped GEMM): C[g] = lhs[g]^T @ rhs[g]. lhs [M_total,
# OUT_M], rhs [M_total, OUT_N], out [G, OUT_M, OUT_N]. Contraction = the per-group
# token count (runtime k_iters); OUT_M/OUT_N fixed. bf16 port of the fp8 wgrad in
# flydsl/grouped_gemm/gemm_fp8_grouped_kernel.py, reusing the bf16 transpose-read
# pipeline (both operands K-major -> ds_read_b64_tr_b16). Per-group SRD num_records
# clamp handles the K-tail (over-read past the group's last token -> 0).
# ───────────────────────────────────────────────────────────────────────
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
    """One quadrant's mma accumulate, reading/writing rmem accs in place (so the
    value survives the runtime chunk scf.for boundary)."""
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
    """One K-tile of the 4-buffer distance-2 pipeline (bf16 port of the dense
    NN/TN main loop): read cur tile k, prefetch tile k+1/k+2 into next/cur, mma.
    Caller swaps cur<->next after. Over-read past the group SRD-clamped to 0."""
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
    """One wgrad output tile: C[group_idx][block_m, block_n] = lhs[g]^T @ rhs[g].
    Runtime K-loop over the group's [m_start, m_end) tokens. Both operands are
    K-major (token-row) so both transpose-read (tr16). 4-buffer distance-2 pipeline
    in an even-CHUNK scf.for (the ping-pong resets at each chunk boundary).
    trans_c stores the result transposed into a [G*OUT_N, OUT_M] buffer."""
    CHUNK = 8  # even: 4-buffer ping-pong resets at chunk boundary (sweep: 8/16/32 perf-neutral)
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

    # Per-group rebased SRD: base folds m_start, num_records bounds the K-tail
    # (over-read past the group's last token -> 0).
    group_tokens = m_end - m_start
    bf16_ir = fx.BFloat16.ir_type
    # base/num_records use runtime out_*_rt (compile-time OUT_* would const-fold the
    # m_start*OUT multiply to i16 and overflow for groups past the first).
    gA = make_bf16_buffer_tensor_rebased(
        A, bf16_ir, m_start * out_m_rt * fx.Int32(2), group_tokens * out_m_rt * fx.Int32(2)
    )
    gB = make_bf16_buffer_tensor_rebased(
        B, bf16_ir, m_start * out_n_rt * fx.Int32(2), group_tokens * out_n_rt * fx.Int32(2)
    )
    a_div = fx.logical_divide(gA, fx.make_layout(1, 1))
    b_div = fx.logical_divide(gB, fx.make_layout(1, 1))

    gl_off_a = compute_global_swizzle_nn_bf16(lane_id, wave_id, OUT_M, n_lds_steps_a)
    gl_off_b = compute_global_swizzle_nn_bf16(lane_id, wave_id, OUT_N, n_lds_steps_b)

    a0_off = block_m * BLOCK_M
    a1_off = a0_off + lds_block_m
    b0_off = block_n * BLOCK_N
    b1_off = b0_off + lds_block_n
    a_k_step = fx.Int32(BLOCK_K) * out_m_rt  # runtime i32 (k*a_k_step must not const-fold to i16)
    b_k_step = fx.Int32(BLOCK_K) * out_n_rt

    # 16x16x32 MFMA path (the keeper): 2x more m/n tiles (16-row/col), acc = 4 f32/lane.
    nta16 = n_tiles_a * 2
    ntb16 = (BLOCK_N // 16) // (2 * n_wave_n)  # n-tiles per wave (8/n_wave_n for BN=256)
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

    # Per-tile entry: drain the previous tile's outstanding C stores (vmcnt 0) so the
    # prelude's graded wait_barrier counts are relative to a clean vmcnt, + WAR barrier
    # vs the previous tile's LDS reads (persistent grid reuses the LDS across tiles).
    wait_barrier(0)
    # Prelude: tile 0 -> cur, tile 1 -> next.
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

    # 2D (i,j) 16x16 tiles per quadrant: acc[i*ntb16+j] -> rows i*16, cols j*16.
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
    """Read group_offs[idx] from an int32 [G+1] buffer view (uniform across the WG)."""
    atom = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Int32)
    reg = fx.make_rmem_tensor(fx.make_layout(1, 1), fx.Int32)
    fx.copy(atom, fx.slice(div, (None, fx.Int32(idx))), reg)
    return Vec(fx.memref_load_vec(reg))[0]


def load_go_i64(div, idx):
    """Read group_offs[idx] from an int64 [G+1] buffer view; return i32 (values fit i32)."""
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
    """Persistent bf16 wgrad: fixed grid of WGs strides the (group, block_m, block_n)
    tile space. C is the stacked [G*OUT_M, OUT_N] output ([G*OUT_N, OUT_M] if trans_c)."""
    assert OUT_M % BLOCK_M == 0, "OUT_M (unclamped store dim) must divide BLOCK_M"
    N_BLOCKS_M = OUT_M // BLOCK_M
    N_BLOCKS_N = (OUT_N + BLOCK_N - 1) // BLOCK_N  # ceil-div: last N block partial, store n-clamps
    TILES_PER_GROUP = N_BLOCKS_M * N_BLOCKS_N
    TOTAL = G * TILES_PER_GROUP
    # Non-persistent grid=TOTAL (one tile/WG): correct + faster than the strided
    # persistent loop (which has a multi-tile-per-WG accumulator-reset bug; only needed
    # for CU-capped comm overlap, deferred).
    SharedStorage = _make_shared_storage(BLOCK_M, BLOCK_N)  # 4-buffer (cur/next) A+B

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
                # trans output is m-contiguous; vary block_m fastest so neighbouring
                # tiles write contiguous transposed rows (L2/DRAM write locality).
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

        _do_tile(pid)  # one tile per WG (grid = TOTAL)

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
    lhs: torch.Tensor,  # [M_total, OUT_M] bf16
    rhs: torch.Tensor,  # [M_total, OUT_N] bf16
    group_offs: torch.Tensor,  # [G+1] int32
    out_dtype: torch.dtype = torch.bfloat16,
    BLOCK_M: int = 256,
    BLOCK_N: int = 256,
    num_xcd: int = 8,
    trans_c: bool = False,
) -> torch.Tensor:
    """bf16 variable-K wgrad: out[g] = lhs[offs[g]:offs[g+1]]^T @ rhs[offs[g]:offs[g+1]].
    out [G, OUT_M, OUT_N], or [G, OUT_N, OUT_M] when trans_c (= W1/W2-native layout)."""
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
