###############################################################################
# SPDX-License-Identifier: Apache-2.0
#
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 FlyDSL Project Contributors
#
# Adapted from FlyDSL (https://github.com/ROCm/FlyDSL)
# Modified by the Primus-Turbo team.
#
# This file is distributed under the Apache License 2.0 (see LICENSE-APACHE),
# not the MIT license that covers the rest of Primus-Turbo (see LICENSE).
###############################################################################

"""MXFP8 quantization for gfx1250 (MI455X): E4M3 codes, one E8M0 scale per 32 elements, the smallest 2^e with
amax / 2^e <= 448 (nothing saturates). ``tests/pytorch/ref/mxfp8_ref.py`` is the bitwise reference.

The outputs are operands of the gfx1250 MX GEMM: a row copy ``[rows, cols]`` of x and / or a column copy (of x
transposed), the copy's last dim padded to a multiple of 128 (zero codes, byte-0 scales). Scales are packed,
``[ceil(rows/32), cols/128, 32, 4]`` uint8 viewed as ``float8_e8m0fnu``: one int32 is a row's 4 scales of a
128-wide step, pad bytes 0. Codes are row-major (A operands) or, with the ``layout`` bit ``SHUF_ROW`` / ``SHUF_COL``
of the row / column copy, in 16x16 blocks ``[rows/16][cols/16][16][16]``, same shape, rows % 16 == 0 (B operands).
Only ``quant_mxfp8_rows`` writes raw scales ``[rows, cols/32]``: MegaMoE's wire format. Batched inputs put the
batch dim first. One kernel family, one 128 x 128 bf16 tile per workgroup (8 waves):

* the tile comes in by one TDM load into LDS rows padded by 16 bytes; rows and columns past the
  tensor (or past an expert's tokens) are zero-filled by the descriptor's bounds (the input's last row in a
  partial last k-tile: masked 16-byte loads, which address nothing past the data; see ``TAIL``);
* rows phase: a lane takes half a 32-block of a row (two ``ds_load_b128``), |x| max on the bf16
  bits, the other half's max by DPP, the E8M0 byte, codes by
  ``v_cvt_scalef32_pk8_fp8_bf16``, one 16-byte store (8 lanes fill a 128-byte line);
* cols phase: the same on half a 32-block of a column, read by two ``ds_load_tr16_b128`` (the
  LDS transposes 8 x 8 bf16 blocks), giving the transposed copy ``q_col [cols, rows]``; its codes
  (staged in the consumed bf16 tile) and all scale bytes go through LDS, then out as whole 128-byte
  lines and one scale dword per row;
* ``2d`` (weights): one scale per 32 x 32 block, its max from a pre-pass into LDS; both copies use
  it, so ``q_t == q.T``;
* ``grouped`` (MoE): the row copy per expert zero-padded to 32 rows, the column copy to 128 tokens (pad
  rows / blocks: scale 127, Turbo's grouped contract); HIP's 8-tuple with both padded lens / offsets.
* producers of the tile besides a plain load: fused GLU / dGLU (MoE), gather (permute), gather of E4M3 rows with
  raw E8M0 scales (``P_FP8``, MegaMoE's dispatched rows: dequantized exactly, the row copy keeps their codes and
  scales, the column copy re-quantizes).
"""

from __future__ import annotations

import math

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm as _llvm
from flydsl._mlir.dialects import rocdl as _rocdl
from flydsl._mlir.dialects import vector as _vector
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.arith import ArithValue
from flydsl.expr.arith import _to_raw as _raw
from flydsl.expr.typing import Constexpr, T
from flydsl.expr.typing import Vector as Vec

from primus_turbo.flydsl.utils.buffer_ops import (
    buffer_load,
    buffer_store,
    create_buffer_resource_from_addr,
)
from primus_turbo.flydsl.utils.gemm_epilogue_helper import (
    _check_activation,
    _check_clamp_limit,
    _glu_act,
    _glu_act_grad,
    _glu_clamp,
    _glu_kept,
)
from primus_turbo.flydsl.utils.prims import _row16_sum_f32

I32, I64 = fx.Int32, fx.Int64
BM = BK = 128  # tile rows x columns
PITCH = 2 * BK + 16  # LDS bytes per bf16 tile row
CPITCH = BM + 32  # LDS bytes per staged column-copy row (2-way bank conflicts on the stage writes)
NT = 256  # 8 waves
# TDM group-1 word 0: 2-byte elements, LDS padding 4 dwords after every 64 (one tile row)
TDM_W0 = (1 << 16) | (1 << 20) | (5 << 22) | (3 << 25)
P_DENSE, P_GROUPED, P_GLU, P_DGLU, P_GATHER, P_FP8 = range(6)  # producers of the bf16 tile
O_ROWS, O_COLS = 1, 2  # outputs
SHUF_ROW, SHUF_COL = 4, 8  # ``layout`` bits: the row / column copy's codes in 16x16 blocks
_PACK = 1  # the kernel's layout bit of a row copy with packed scales (all but the wire rows)
PAD_SCALE = 127


def _addr(p):
    return I64(fx.ptrtoint(p))


def _rsrc(a, nbytes):
    """Buffer resource at byte address ``a`` with an access bound: a masked store (offset 0x7FFFFFFF)
    is dropped only when ``nbytes`` < 2^31, never without a bound."""
    return create_buffer_resource_from_addr(a, num_records_bytes=nbytes)


def _lds_ptr(a):
    return _llvm.inttoptr(ir.Type.parse("!llvm.ptr<3>"), _raw(a))


def _lds_v4(a):
    return Vec(_llvm.LoadOp(T.vec(4, T.i32), _lds_ptr(a), alignment=16).result)


def _lds_tr16(a):
    return Vec(_rocdl.ds_load_tr16_b128(ir.VectorType.get([8], T.i16), _lds_ptr(a))).bitcast(I32)


def _lds_st(a, v, align):
    _llvm.StoreOp(_raw(v), _lds_ptr(a), alignment=align)


def _i8(v):
    return ArithValue(_raw(v)).trunci(T.i8)


def _dpp(v, ctrl):
    return I32(_rocdl.update_dpp(T.i32, _raw(v), _raw(v), ctrl, 0xF, 0xF, False))


def _max(a, b):
    return (a > b).select(a, b)


def _amax(words):
    """max |x| bits (bf16, in an i32) of the bf16 pairs in ``words`` (vector<4xi32> list): unsigned max
    of the bits shifted left by one (drops the sign, one packed shift per pair), shifted back."""
    m = None
    for w in words:
        a = Vec(w).bitcast(fx.Uint16) << fx.Uint16(1)
        m = a if m is None else (a > m).select(a, m)
    return m.reduce("max").to(I32) >> I32(1)


def _e8m0(amax):
    """E8M0 byte of a block from its max |x| bf16 bits: exponent - 8, one more when the mantissa is above 1.75
    (448 = 1.75 * 2^8), so amax / 2^(byte - 127) <= 448; Inf / NaN -> 0xFF."""
    byte = ((amax + I32(0x1F)) >> I32(7)) - I32(8)
    byte = _max(byte, I32(0))
    return (amax >= I32(0x7F80)).select(I32(0xFF), byte)


def _cvt16(words, byte, swap=None):
    """16 bf16 (two vector<4xi32>) / 2^(byte - 127) -> 16 E4M3 codes (vector<4xi32>; ``swap``: the
    words' halves in reverse order). The conversion uses the scale's exponent field only (0 reads as
    2^-127) and keeps denormals."""
    scale = I32(byte << I32(23)).bitcast(fx.Float32)
    out = [
        Vec(_rocdl.cvt_scalef32_pk8_fp8_bf16(T.vec(2, T.i32), _raw(Vec(w).bitcast(fx.BFloat16)), _raw(scale)))
        for w in words
    ]
    if const_expr(swap is None):
        return out[0].shuffle(out[1], [0, 1, 2, 3])
    return swap.select(out[1].shuffle(out[0], [0, 1, 2, 3]), out[0].shuffle(out[1], [0, 1, 2, 3]))


def _tdm_load(lds, gaddr, cols, rows, stride):
    """TDM load of a BM x BK bf16 tile at ``gaddr`` (row stride ``stride`` elements) into LDS at
    ``lds``; ``cols`` / ``rows`` (from the tile start) bound it, the rest of the tile is zero."""
    g0 = [I32(1), lds, I32(gaddr), I32(gaddr >> I64(32)) | I32(-(1 << 31))]
    g1 = [
        I32(TDM_W0),
        (cols & I32(0xFFFF)) << I32(16),
        ((cols >> I32(16)) & I32(0xFFFF)) | ((rows & I32(0xFFFF)) << I32(16)),
        ((rows >> I32(16)) & I32(0xFFFF)) | I32(BK << 16),
        I32(BM),
        I32(stride),
        I32(0),
        I32(0),
    ]
    vec = [_vector.from_elements(T.vec(len(g), T.i32), [_raw(v) for v in g]) for g in (g0, g1)]
    z4, z8 = _raw(Vec.filled(4, 0, I32)), _raw(Vec.filled(8, 0, I32))
    rocdl.tensor_load_to_lds(vec[0], vec[1], z4, z4, z8)


def _masked_tile(lds, gaddr, stride, rows, cols, zero, t):
    """``_tdm_load``'s tile by masked 16-byte async loads, one per lane and pass: the 8 columns from ``c`` of row
    ``r`` at ``gaddr + (r * stride + c) * 2`` when r < ``rows`` and c < ``cols`` (a multiple of 8), else 16 bytes of
    the zero buffer ``zero``. Nothing past the data is addressed. The caller waits for them (asynccnt)."""
    for i in range_constexpr(BM * BK // 8 // NT):
        q = t + I32(i * NT)
        r, c = q // I32(16), (q % I32(16)) * I32(8)
        ok = (r < rows) & (c < cols)
        ga = ok.select(gaddr + (I64(r) * I64(stride) + I64(c)) * I64(2), zero)
        gp = _llvm.inttoptr(ir.Type.parse("!llvm.ptr<1>"), _raw(ga))
        _rocdl.global_load_async_to_lds_b128(gp, _lds_ptr(lds + r * I32(PITCH) + c * I32(2)), 0)


def _group_scan(arg_go, mc0, G, lane):
    """Expert of the column-copy tile at padded token ``mc0``, lane-parallel: lane l of chunk c takes
    expert 32 c + l, DPP prefix sums give the offsets of the column copy (experts padded to 128 tokens)
    and of the row copy (to 32 rows), a ballot the tile's expert. Returns (first input row, real rows,
    first row-copy row, [(lane's expert, (len, offset) of the column copy, (len, offset) of the rows)]
    per chunk); expert G's offsets are the padded totals."""
    rs = _rsrc(_addr(arg_go), I32(8 * (G + 1)))
    acc, acc32, in0, nreal, rb32, per = I32(0), I32(0), I32(0), I32(0), I32(0), []

    def scan(v):  # inclusive prefix over the wave: row_shr within 16-lane rows, then row 0's total
        for k in (1, 2, 4, 8):
            v = v + I32(_rocdl.update_dpp(T.i32, _raw(I32(0)), _raw(v), 0x110 + k, 0xF, 0xF, True))
        return v + (lane >= I32(16)).select(I32(rocdl.readlane(T.i32, v, 15)), I32(0))

    for ch in range_constexpr(G // 32 + 1):
        g = I32(ch * 32) + lane
        o0 = I32(buffer_load(rs, g * I32(2), vec_width=1, dtype=T.i32))
        ln = (g < I32(G)).select(
            I32(buffer_load(rs, g * I32(2) + I32(2), vec_width=1, dtype=T.i32)) - o0, I32(0)
        )
        pl, pl32 = ((ln + I32(127)) >> I32(7)) << I32(7), ((ln + I32(31)) >> I32(5)) << I32(5)
        x, x32 = scan(pl), scan(pl32)
        excl, excl32 = acc + x - pl, acc32 + x32 - pl32
        hit = rocdl.ballot(T.i32, (mc0 >= excl) & (mc0 < excl + pl))
        src = I32(
            _llvm.call_intrinsic(T.i32, "llvm.cttz.i32", [_raw(I32(hit)), _raw(fx.Boolean(False))], [], [])
        )
        found = I32(hit) != I32(0)
        in0 = found.select(I32(rocdl.readlane(T.i32, o0 + mc0 - excl, src)), in0)
        nreal = found.select(I32(rocdl.readlane(T.i32, ln - (mc0 - excl), src)), nreal)
        rb32 = found.select(I32(rocdl.readlane(T.i32, excl32 + mc0 - excl, src)), rb32)
        per.append((g, (pl, excl), (pl32, excl32)))
        acc, acc32 = acc + I32(rocdl.readlane(T.i32, x, 31)), acc32 + I32(rocdl.readlane(T.i32, x32, 31))
    return in0, nreal, rb32, per


def _act8(g, u, prob, act, clamp):
    """GLU forward on 8 columns (f32), the gfx950 phase-2 math: clamp, silu as g / (1 + exp2(-g log2 e))
    (gelu via its sigmoid form), times up, times the row's prob; the caller rounds to bf16."""
    gc, uc, _, _ = _glu_clamp(g, u, clamp if clamp else None, lanes=8)
    if const_expr(act == "silu"):
        ex = Vec.from_elements(
            [
                fx.Float32(rocdl.exp2(T.f32, _raw(v * fx.Float32(-1.4426950408889634))))
                for v in [gc[i] for i in range(8)]
            ],
            fx.Float32,
        )
        gate = gc / (ex + fx.Float32(1.0))
    else:
        gate = _glu_act(gc, act, lanes=8)
    return (gate * uc) * prob


def _dact8(d, g, u, prob, act, clamp):
    """GLU backward on 8 columns, the gfx950 phase-2 math: (dgate, dup, sum of d * f(g) * up)."""
    gc, uc, kept_g, kept_u = _glu_clamp(g, u, clamp if clamp else None, lanes=8)
    a, da = _glu_act_grad(gc, act, lanes=8)
    gp = fx.Float32(0.0)
    for i in range_constexpr(8):
        gp = gp + d[i] * a[i] * uc[i]
    dp = d * prob
    return _glu_kept(dp * uc * da, kept_g, lanes=8), _glu_kept(dp * a, kept_u, lanes=8), gp


def _bf8(v):
    return v.to(fx.BFloat16).bitcast(I32)


@flyc.jit
def _launch(
    arg_x: fx.Pointer,
    arg_x2: fx.Pointer,
    arg_qr: fx.Pointer,
    arg_sr: fx.Pointer,
    arg_qc: fx.Pointer,
    arg_sc: fx.Pointer,
    arg_go: fx.Pointer,
    arg_lc: fx.Pointer,
    arg_oc: fx.Pointer,
    arg_p: fx.Pointer,
    arg_gp: fx.Pointer,
    arg_dbg: fx.Pointer,
    arg_z: fx.Pointer,
    stream: fx.Stream,
    i32_m: fx.Int32,
    i32_mc: fx.Int32,
    i32_grid: fx.Int32,
    i32_batch: fx.Int32,
    i32_t: fx.Int32,
    K: Constexpr[int],
    PROD: Constexpr[int],
    OUT: Constexpr[int],
    B2D: Constexpr[int],
    G: Constexpr[int],
    ACT: Constexpr[str],
    CLAMP: Constexpr[float],
    DBG: Constexpr[int],
    ZPROB: Constexpr[int],
    LAY: Constexpr[int],
):
    NKT = (K + BK - 1) // BK
    # A TDM load addresses whole BK-column tile rows (its inner extent bounds the data, not the addressing): in a
    # partial last k-tile each row is read up to 254 bytes past its data, into the rows that follow. With K >= 64
    # that is at most one row, so only the input's last row reads past the input: the tile holding it is loaded by
    # masked 16-byte loads instead (K % 8 == 0: whole 16-byte chunks). Inputs with K % 8 != 0 or K < 64 (a row
    # reads several rows ahead) get storage past their end from the entries instead (``_check``).
    TAIL = K % BK != 0 and K % 8 == 0
    EXP = PROD != P_DENSE  # the MoE producers: expert-padded copies
    HALVES = 2 if PROD == P_DGLU else 1  # dGLU quantizes dgate (columns [0, K)) and dup ([K, 2K))
    KV = NKT * BK  # columns of a half in the row copy: K padded to 128 (zero codes, byte-0 scales)
    OW = HALVES * KV  # row-copy width
    KBO, RT = OW // 32, OW // BK  # 32-blocks / 128-column steps of a row
    NTILE = {P_GLU: 2, P_DGLU: 3}.get(PROD, 1)
    TILE = BM * PITCH
    O_SR = NTILE * TILE  # scale bytes: the row copy's [BM][4], then the column copy's [BK][4]
    O_SC, O_TAB = O_SR + BM * 4, O_SR + (BM + BK) * 4
    PR, SR, SC = LAY & _PACK, LAY & SHUF_ROW, LAY & SHUF_COL

    @flyc.kernel(name=f"mxfp8_quant_gfx1250_p{PROD}o{OUT}", known_block_size=[NT, 1, 1])
    def kern(
        arg_x: fx.Pointer,
        arg_x2: fx.Pointer,
        arg_qr: fx.Pointer,
        arg_sr: fx.Pointer,
        arg_qc: fx.Pointer,
        arg_sc: fx.Pointer,
        arg_go: fx.Pointer,
        arg_lc: fx.Pointer,
        arg_oc: fx.Pointer,
        arg_p: fx.Pointer,
        arg_gp: fx.Pointer,
        arg_dbg: fx.Pointer,
        arg_z: fx.Pointer,
        i32_m: fx.Int32,
        i32_mc: fx.Int32,
        i32_t: fx.Int32,
    ):
        arena = fx.SharedAllocator(static=False)
        arena.allocate(O_TAB + 64)
        base = I32(I64(fx.ptrtoint(arena.base_ptr)))
        t = I32(fx.thread_idx.x)
        lane = t % I32(32)
        wave = rocdl.readfirstlane(T.i32, t // I32(32))
        pid = I32(fx.block_idx.x)
        bat = I32(fx.block_idx.y)
        mt = pid // I32(NKT)
        kt = pid - mt * I32(NKT)
        k0 = kt * I32(BK)
        mc0 = mt * I32(BM)  # the tile's first row in the column copy's token space
        if const_expr(EXP):
            in0, nreal, rb32, per = _group_scan(arg_go, mc0, G, lane)
            if (pid == I32(0)) & (wave == I32(0)):  # padded lens / offsets (int64 [2, G + 1]) of both copies
                for a, w in ((arg_lc, 1), (arg_oc, 2)):  # column copy (128 tokens), row copy (32 rows)
                    lo = _rsrc(_addr(a), I32(16 * (G + 1)))
                    for g, *pairs in per:
                        ln, ex = pairs[w - 1]
                        buffer_store(Vec.from_elements([ln, I32(0)], I32), lo, g * I32(2), mask=g < I32(G))
                        buffer_store(
                            Vec.from_elements([ex, I32(0)], I32),
                            lo,
                            (g + I32(G + 1)) * I32(2),
                            mask=g <= I32(G),
                        )
        else:
            in0, rb32 = mc0, I32(0)
            nreal = i32_m - mc0
        nreal = (nreal > I32(BM)).select(I32(BM), nreal)
        if nreal > I32(0):
            row0 = I64(bat) * I64(i32_m) + I64(in0)  # the tile's first input (= output) row
            cols = I32(K) - k0
            # ---- producer: the bf16 tile(s) into LDS, rows / columns past the data zero
            if const_expr(PROD in (P_DENSE, P_GROUPED)):
                if const_expr(TAIL):
                    if (kt == I32(NKT - 1)) & (in0 + nreal == i32_m):  # see TAIL
                        src = _addr(arg_x) + (row0 * I64(K) + I64(k0)) * I64(2)
                        _masked_tile(base, src, K, nreal, cols, _addr(arg_z), t)
                        rocdl.s_wait_asynccnt(0)
                    elif wave == I32(0):
                        _tdm_load(base, _addr(arg_x) + (row0 * I64(K) + I64(k0)) * I64(2), cols, nreal, K)
                        rocdl.s_wait_tensorcnt(0)
                elif wave == I32(0):
                    _tdm_load(base, _addr(arg_x) + (row0 * I64(K) + I64(k0)) * I64(2), cols, nreal, K)
                    rocdl.s_wait_tensorcnt(0)
            elif const_expr(PROD in (P_GLU, P_DGLU)):  # gate / up halves of l1, then dact (K % 128 == 0)
                l1 = _addr(arg_x2 if PROD == P_DGLU else arg_x) + (row0 * I64(2 * K) + I64(k0)) * I64(2)
                if wave < I32(NTILE):
                    src = [l1, l1 + I64(2 * K), _addr(arg_x) + (row0 * I64(K) + I64(k0)) * I64(2)]
                    for w in range_constexpr(NTILE):
                        if wave == I32(w):
                            _tdm_load(base + I32(w * TILE), src[w], cols, nreal, K if w == 2 else 2 * K)
                    rocdl.s_wait_tensorcnt(0)
            elif const_expr(PROD == P_GATHER):
                # row r is x[dest2src[in0 + r]]: one 16-byte async load per lane and chunk
                ri = _rsrc(_addr(arg_x2), i32_m * I32(4))
                rp = _rsrc(_addr(arg_p), i32_m * I32(4))
                for i in range_constexpr(BM * BK // 8 // NT):
                    q = t + I32(i * NT)
                    r, c16 = q // I32(16), q % I32(16)
                    src = I32(buffer_load(ri, in0 + r, vec_width=1, dtype=T.i32))
                    ok = (r < nreal) & (src >= I32(0)) & (src < i32_t) & (k0 + c16 * I32(8) < I32(K))
                    if const_expr(ZPROB):  # zero the rows whose prob is 0 (the permute's padding rule)
                        ok = ok & (
                            fx.Float32(buffer_load(rp, in0 + r, vec_width=1, dtype=T.f32)) != fx.Float32(0.0)
                        )
                    ga = ok.select(
                        _addr(arg_x) + (I64(src) * I64(K) + I64(k0 + c16 * I32(8))) * I64(2), _addr(arg_z)
                    )
                    gp = _llvm.inttoptr(ir.Type.parse("!llvm.ptr<1>"), _raw(ga))
                    _rocdl.global_load_async_to_lds_b128(
                        gp, _lds_ptr(base + r * I32(PITCH) + c16 * I32(16)), 0
                    )
                rocdl.s_wait_asynccnt(0)
            else:  # P_FP8: row r is the E4M3 row x[x2[in0 + r]] (16 codes per lane and chunk) times its E8M0 scales
                # (p, raw [t, K / 32]), exact in bf16; an out-of-range row is zero. The rows phase takes the staged
                # scales back, so the row copy keeps the source codes (a re-derived scale can differ by one step).
                ri = _rsrc(_addr(arg_x2), i32_m * I32(4))
                qs = _rsrc(_addr(arg_x), i32_t * I32(K))
                ss = _rsrc(_addr(arg_p), i32_t * I32(K // 32))
                for i in range_constexpr(BM * BK // 16 // NT):
                    q = t + I32(i * NT)
                    r, c16 = q // I32(8), q % I32(8)
                    src = I32(buffer_load(ri, in0 + r, vec_width=1, dtype=T.i32))
                    ok = (r < nreal) & (src >= I32(0)) & (src < i32_t)
                    at = src * I32(K // 4) + kt * I32(BK // 4) + c16 * I32(4)
                    codes = Vec(buffer_load(qs, at, vec_width=4, dtype=T.i32, mask=ok))
                    sw = I32(buffer_load(ss, src * I32(K // BK) + kt, vec_width=1, dtype=T.i32, mask=ok))
                    if c16 == I32(0):
                        _lds_st(base + I32(O_SR) + r * I32(4), sw, 4)
                    # 8 codes -> f32 at scale 2^0 (gfx1250's v_cvt_scale_pk8_f32_fp8, as aiter's MegaMoE combine), times
                    # the block's exact power of two (byte 0: 2^-127 as a denormal, 0xFF: NaN, the reference's values)
                    b = (sw >> ((c16 >> I32(1)) * I32(8))) & I32(0xFF)
                    scale = (
                        (b << I32(23)) | ((b == I32(0)) | (b == I32(0xFF))).select(I32(0x400000), I32(0))
                    ).bitcast(fx.Float32)
                    for h in range_constexpr(2):
                        pk8 = Vec.from_elements([codes[2 * h], codes[2 * h + 1]], I32)
                        v8 = Vec(rocdl.cvt_scale_pk8_f32_fp8(T.vec(8, T.f32), _raw(pk8), _raw(I32(127)), 0))
                        _lds_st(base + r * I32(PITCH) + c16 * I32(32) + I32(16 * h), _bf8(v8 * scale), 16)
            gpu.barrier()
            if const_expr(PROD in (P_GLU, P_DGLU)):  # elementwise GLU / dGLU, rounded to bf16 in place
                rp = _rsrc(_addr(arg_p), i32_m * I32(4))
                gpr = _rsrc(_addr(arg_gp) + I64(kt) * I64(i32_m) * I64(4), i32_m * I32(4))
                if const_expr(DBG):  # the bf16 values, for the tests
                    dbg = _rsrc(_addr(arg_dbg) + row0 * I64(OW * 2), nreal * I32(OW * 2))
                for i in range_constexpr(BM * BK // 8 // NT):
                    q = t + I32(i * NT)
                    r, c8 = q // I32(16), (q % I32(16)) * I32(8)
                    off = base + r * I32(PITCH) + c8 * I32(2)
                    prob = fx.Float32(buffer_load(rp, in0 + r, vec_width=1, dtype=T.f32))
                    gv, uv = [
                        Vec(_lds_v4(off + I32(w * TILE))).bitcast(fx.BFloat16).to(fx.Float32)
                        for w in range(2)
                    ]
                    if const_expr(PROD == P_GLU):
                        res = [_bf8(_act8(gv, uv, prob, ACT, CLAMP))]
                    else:
                        dv = Vec(_lds_v4(off + I32(2 * TILE))).bitcast(fx.BFloat16).to(fx.Float32)
                        dg, du, gsum = _dact8(dv, gv, uv, prob, ACT, CLAMP)
                        res = [_bf8(dg), _bf8(du)]
                        # lane 15 of each 16-lane row: the row's sum over the tile
                        gsum = _row16_sum_f32(gsum)
                        buffer_store(gsum, gpr, in0 + r, mask=(r < nreal) & (q % I32(16) == I32(15)))
                    for w in range_constexpr(len(res)):
                        _lds_st(off + I32(w * TILE), res[w], 16)
                        if const_expr(DBG):
                            at = (r * I32(OW) + I32(w * K) + k0 + c8) * I32(2)
                            buffer_store(res[w], dbg, at, mask=r < nreal, offset_is_bytes=True)
                gpu.barrier()
            if const_expr(B2D):  # 32 x 32 block maxima: 16 lanes per block, 2 rows each
                bi, sub = t // I32(16), t % I32(16)
                a0 = base + ((bi // I32(4)) * I32(32) + sub * I32(2)) * I32(PITCH) + (bi % I32(4)) * I32(64)
                a = _amax([_lds_v4(a0 + I32(r * PITCH + 16 * q)) for r in range(2) for q in range(4)])
                for ctrl in (0xB1, 0x4E, 0x164, 0x168):  # xor 1, 2, 4, 8
                    a = _max(a, _dpp(a, ctrl))
                _lds_st(base + I32(O_TAB) + bi * I32(4), a, 4)
                gpu.barrier()

            def blk_amax(words, tab, partner):
                if const_expr(B2D):
                    return I32(
                        _llvm.LoadOp(T.i32, _lds_ptr(base + I32(O_TAB) + tab * I32(4)), alignment=4).result
                    )
                a = _amax(words)
                return _max(a, _dpp(a, partner))

            for hf in range_constexpr(HALVES):
                tb = base + I32((hf if PROD == P_DGLU else 0) * TILE)  # the bf16 tile of this half
                sb = base + I32((2 - 2 * hf if PROD == P_DGLU else 0) * TILE)  # a consumed tile: the stage
                ocol = I32(hf * K)  # the half's first column of the row copy / row of the column copy
                # ---- rows phase: lane = (row, 32-block kb, half h), 4 rows per wave and pass; codes
                # straight to q_row (8 lanes = one 128-byte line), scale bytes to LDS. 16x16 codes: 16 rows x
                # one kb per wave and pass, so each 16-lane half stores a whole 256-byte block. Expert row copies
                # hold whole 32-row groups: rows past the tokens are zero codes, scale 127.
                nrows = ((nreal + I32(31)) >> I32(5)) << I32(5) if EXP else nreal
                rowbase = I64(rb32) if EXP else I64(bat) * I64(i32_m) + I64(in0)
                if const_expr(OUT & O_ROWS):
                    qr = _rsrc(_addr(arg_qr) + rowbase * I64(OW), nrows * I32(OW))
                    h = lane % I32(2)
                    for p in range_constexpr(BM // 32):
                        if const_expr(SR):
                            u = I32(p * 8) + wave
                            r, kb = (u >> I32(2)) * I32(16) + lane // I32(2), u & I32(3)
                        else:
                            r, kb = I32(p * 32) + wave * I32(4) + lane // I32(8), (lane % I32(8)) // I32(2)
                        col = k0 + kb * I32(32) + h * I32(16)
                        a0 = tb + r * I32(PITCH) + (kb * I32(2) + h) * I32(32)
                        words = [_lds_v4(a0), _lds_v4(a0 + I32(16))]
                        if const_expr(PROD == P_FP8):  # the source row's scale, staged by the producer
                            sw = _llvm.LoadOp(T.i32, _lds_ptr(base + I32(O_SR) + r * I32(4)), alignment=4)
                            byte = (I32(sw.result) >> (kb * I32(8))) & I32(0xFF)
                        else:
                            byte = _e8m0(blk_amax(words, (r // I32(32)) * I32(4) + kb, 0xB1))
                        if const_expr(EXP):  # pad rows: 127 over the real columns, 0 in the K padding
                            byte = ((r >= nreal) & (k0 + kb * I32(32) < I32(K))).select(I32(PAD_SCALE), byte)
                        codes = _cvt16(words, byte)
                        ok = (r < nrows) & (col < I32(KV))
                        if const_expr(SR):
                            b16 = (r >> I32(4)) * I32(OW // 16) + ((ocol + col) >> I32(4))
                            off = b16 * I32(256) + (r & I32(15)) * I32(16)
                        else:
                            off = r * I32(OW) + ocol + col
                        buffer_store(codes, qr, off, mask=ok, offset_is_bytes=True)
                        if h == I32(0):
                            _lds_st(base + I32(O_SR) + r * I32(4) + kb, _i8(byte), 1)
                # ---- cols phase: lane = (column of an 8-column chunk, 16 rows of a 64-row slab); codes and
                # scale bytes staged in LDS. Groups 16 rows apart hit the same banks whatever the row pitch:
                # odd groups read their second 8 rows first (2-way conflicts, the minimum).
                if const_expr(OUT & O_COLS):
                    g, j = lane // I32(8), lane % I32(8)
                    odd = (g % I32(2)) == I32(1)
                    first = odd.select(I32(8 * PITCH), I32(0))
                    staged = []
                    for cp in range_constexpr(BK // 64):
                        c = (I32(cp * 8) + wave) * I32(8) + j
                        for rp in range_constexpr(BM // 64):
                            m = I32(rp * 64) + g * I32(16)  # the lane's first row in the tile
                            a0 = tb + (m + j) * I32(PITCH) + (c - j) * I32(2)
                            words = [_lds_tr16(a0 + first), _lds_tr16(a0 + I32(8 * PITCH) - first)]
                            blk = I32(rp * 2) + g // I32(2)
                            byte = _e8m0(blk_amax(words, blk * I32(4) + c // I32(32), 0x168))
                            if const_expr(EXP):  # blocks of padding only
                                byte = (blk * I32(32) >= nreal).select(I32(PAD_SCALE), byte)
                            at = c * I32(CPITCH) + m
                            staged.append((at, _cvt16(words, byte, odd), c * I32(4) + blk, byte))
                    gpu.barrier()  # every wave is done with the tile: stage the column copy in a consumed one
                    for qo, codes, so, byte in staged:
                        _lds_st(sb + qo, codes, 16)
                        if g % I32(2) == I32(0):
                            _lds_st(base + I32(O_SC) + so, _i8(byte), 1)
                gpu.barrier()

                # ---- stores from LDS: the column copy (whole 128-token tiles) by 128-byte lines (16x16 blocks:
                # 256 bytes), scales one row per lane (packed: a 32-row group's 128 bytes per 32 lanes)
                ncol = I32(K) - k0
                ncol = (ncol > I32(BK)).select(I32(BK), ncol)
                mcb = (i32_mc + I32(31)) // I32(32)
                cbase = I64(bat) * I64(HALVES * K) + I64(ocol) + I64(k0)  # the tile's first column-copy row
                if const_expr(OUT & O_COLS):
                    qc = _rsrc(_addr(arg_qc) + cbase * I64(i32_mc), ncol * i32_mc)
                    for i in range_constexpr(BK * BM // 16 // NT):
                        idx = t + I32(i * NT)
                        if const_expr(SC):  # 16 lanes per 16x16 block: 256 contiguous bytes
                            c, q16 = (
                                (idx // I32(128)) * I32(16) + idx % I32(16),
                                ((idx // I32(16)) % I32(8)) * I32(16),
                            )
                            b16 = (c >> I32(4)) * (i32_mc >> I32(4)) + ((mc0 + q16) >> I32(4))
                            off = b16 * I32(256) + (c & I32(15)) * I32(16)
                        else:
                            c, q16 = idx // I32(8), (idx % I32(8)) * I32(16)
                            off = c * i32_mc + mc0 + q16
                        v = _lds_v4(sb + c * I32(CPITCH) + q16)
                        buffer_store(v, qc, off, mask=c < ncol, offset_is_bytes=True)
                # lanes 0..127: row rs of the row copy's scales; 128..255: of the column copy's
                rs = t % I32(BM)
                sbyte = I32(_llvm.LoadOp(T.i32, _lds_ptr(base + I32(O_SR) + t * I32(4)), alignment=4).result)
                if t < I32(BM):
                    if const_expr(OUT & O_ROWS):
                        kbg = (ocol + k0) // I32(32)
                        if const_expr(PR):
                            # rows past the data in the last 32-row group: 0 (2d blocks would not be)
                            gbase = (
                                I64(rb32 >> I32(5))
                                if EXP
                                else I64(bat) * I64((i32_m + I32(31)) >> I32(5)) + I64(in0 >> I32(5))
                            )
                            n32 = ((nreal + I32(31)) >> I32(5)) << I32(5)
                            sr = _rsrc(_addr(arg_sr) + gbase * I64(RT * 128), (n32 >> I32(5)) * I32(RT * 128))
                            grp = (rs >> I32(5)) * I32(RT) + kbg // I32(4)
                            off = grp * I32(128) + (rs & I32(31)) * I32(4)
                            val = (rs < nrows).select(sbyte, I32(0))
                            buffer_store(val, sr, off, mask=rs < n32, offset_is_bytes=True)
                        else:  # raw: a row's 4 bytes of this k-tile
                            sr = _rsrc(_addr(arg_sr) + rowbase * I64(KBO), nrows * I32(KBO))
                            off = rs * I32(KBO) + kbg
                            buffer_store(sbyte, sr, off, mask=rs < nrows, offset_is_bytes=True)
                elif const_expr(OUT & O_COLS):
                    # rows past the columns of x: 0 (as mxfp8_ref.pack_scales pads)
                    ct = (mcb + I32(3)) >> I32(2)  # 128-token steps of the column copy
                    gbase = I64(bat) * I64((HALVES * K + 31) // 32) + I64((ocol + k0) >> I32(5))
                    c32 = ((ncol + I32(31)) >> I32(5)) << I32(5)
                    sc = _rsrc(_addr(arg_sc) + gbase * I64(ct) * I64(128), (c32 >> I32(5)) * ct * I32(128))
                    off = ((rs >> I32(5)) * ct + mt) * I32(128) + (rs & I32(31)) * I32(4)
                    val = (rs < ncol).select(sbyte, I32(0))
                    buffer_store(val, sc, off, mask=rs < c32, offset_is_bytes=True)
                if const_expr(hf + 1 < HALVES):
                    gpu.barrier()  # the scale bytes and the stage are rewritten by the next half

    kern(
        arg_x,
        arg_x2,
        arg_qr,
        arg_sr,
        arg_qc,
        arg_sc,
        arg_go,
        arg_lc,
        arg_oc,
        arg_p,
        arg_gp,
        arg_dbg,
        arg_z,
        i32_m,
        i32_mc,
        i32_t,
    ).launch(grid=(i32_grid, i32_batch, 1), block=(NT, 1, 1), stream=stream)


_COMPILED: dict = {}
_E8 = getattr(torch, "float8_e8m0fnu", torch.uint8)
_ZERO: dict = {}
_RAW_STREAM = getattr(torch._C, "_cuda_getCurrentRawStream", None)


def _ptr(t):
    return flyc.from_c_void_p(fx.Int8, t.data_ptr(), assumed_align=16)


def _ceil128(n):
    return -(-n // BK) * BK


def _run(
    prod,
    out,
    lay,
    x,
    outs,
    m,
    mc,
    k,
    b2d=0,
    go=None,
    x2=None,
    p=None,
    gp=None,
    dbg=None,
    t=0,
    act="silu",
    clamp=0.0,
):
    """Launch one kernel of the family: ``mc`` / 128 token tiles x ceil(``k`` / 128). ``outs`` = (q_row, s_row,
    q_col, s_col, lens / offs of the column copy, of the row copy (int64 [2, G + 1]: lens, offsets)), any None. The
    compiled launcher of a constexpr tail is called with the tensors' data_ptr() ints (no per-call pointer
    objects); compiling runs the call itself."""
    batch = 1 if x.ndim == 2 else x.shape[0]
    grid = (mc // BM) * -(-k // BK)
    if grid == 0 or batch == 0:
        return
    if BK * mc >= 2**31:
        raise ValueError(
            f"mxfp8_quant_gfx1250: 128 x {mc} tokens overflow the column copy's 32-bit byte offsets"
        )
    G, zprob = 0 if go is None else go.shape[0] - 1, int(prod == P_GATHER and p is not None)
    tail = (k, prod, out, b2d, G, act, float(clamp or 0.0), int(dbg is not None), zprob, lay)
    dev = torch.cuda.current_device()
    # one device (the outputs are on x's), so the compiling call and the later ones launch on the same stream
    for a in (x, x2, go, p, gp, dbg):
        if a is not None and a.get_device() != dev:
            raise ValueError(
                f"mxfp8_quant_gfx1250: every tensor must be on the current device cuda:{dev}, got one on {a.device}"
            )
    zero = _ZERO.get(dev)
    if zero is None:
        zero = _ZERO[dev] = torch.zeros(64, dtype=torch.uint8, device=x.device)
    tensors = (x, x2, *outs[:4], go, *outs[4:], p, gp, dbg, zero)
    scalars = (m, mc, grid, batch, t, *tail)
    fn = _COMPILED.get(tail)
    if fn is not None:
        xp = x.data_ptr()
        stream = _RAW_STREAM(dev) if _RAW_STREAM is not None else torch.cuda.current_stream()
        fn(*[xp if a is None else a.data_ptr() for a in tensors], stream, *scalars)
        return
    ptrs = [_ptr(x if a is None else a) for a in tensors]
    fn = flyc.compile(_launch, *ptrs, torch.cuda.current_stream(), *scalars)  # compiles, then runs this call
    if fn is not None:  # (None under COMPILE_ONLY)
        _COMPILED[tail] = fn


def _slack(t):
    """Bytes of ``t``'s storage past its last element (``t`` contiguous)."""
    return t.untyped_storage().nbytes() - (t.storage_offset() + t.numel()) * t.element_size()


def _check(x, out_dtype, ndims):
    """``x`` as the kernels read it: contiguous, 16-byte aligned and, where the masked loads do not cover its partial
    last k-tile (K % 128 != 0 with K % 8 != 0 or K < 64, see ``TAIL`` in ``_launch``), with 256 bytes of storage
    past its end (a copy otherwise): the TDM loads of that k-tile address up to 254 bytes past the data."""
    if x.ndim not in ndims or not x.is_cuda or x.dtype != torch.bfloat16:
        raise ValueError(
            f"mxfp8_quant_gfx1250: bf16 CUDA input of {ndims} dims, got {x.dtype} {tuple(x.shape)}"
        )
    if out_dtype != torch.float8_e4m3fn:
        raise ValueError(f"mxfp8_quant_gfx1250: E4M3 codes only, got {out_dtype}")
    K = x.shape[-1]
    room = BK if K % BK and (K % 8 or K < BK // 2) else 0  # elements
    if x.is_contiguous() and x.data_ptr() % 16 == 0 and (not room or _slack(x) >= 2 * room):
        return x
    buf = torch.empty(x.numel() + room, dtype=x.dtype, device=x.device)
    return buf[: x.numel()].view(x.shape).copy_(x)


# Storage past every GEMM operand the quantizers write: a TDM load's inner extent clamps the data, not the
# addressing, so the gfx1250 MX GEMM's two-step K loop may address up to 2048 bytes past an operand's end (the next
# 16 rows of a B operand's 128-wide step).
GEMM_SLACK = 4096
_ALIGN = 256  # every carved tensor starts 256-byte aligned
_ESIZE = {torch.float8_e4m3fn: 1, _E8: 1, torch.uint8: 1, torch.int64: 8}
_PLANS: dict = {}


def _carve(specs, device):
    """The tensors of ``specs`` (``(shape, dtype)`` each) carved from one ``torch.empty``: contiguous, 256-byte
    aligned, each followed by ``GEMM_SLACK`` bytes of the allocation. The offsets are planned once per ``specs``.
    (One allocation per copy, not per dual cast: a copy's codes and scales live and die together, while the row
    and column copies of an activation do not -- the column copy is kept for the wgrad.)"""
    plan = _PLANS.get(specs)
    if plan is None:
        off, parts = 0, []
        for shape, dtype in specs:
            size = _ESIZE[dtype]
            parts.append(
                (dtype, shape, tuple(math.prod(shape[i + 1 :]) for i in range(len(shape))), off // size)
            )
            off = -(-(off + math.prod(shape) * size + GEMM_SLACK) // _ALIGN) * _ALIGN
        if len(_PLANS) > 4096:  # many distinct shapes: start over rather than grow without bound
            _PLANS.clear()
        plan = _PLANS[specs] = (off, tuple(parts))
    total, parts = plan
    buf = torch.empty(total, dtype=torch.uint8, device=device)
    views = {torch.uint8: buf}
    out = []
    for dtype, shape, strides, eoff in parts:
        v = views.get(dtype)
        if v is None:
            v = views[dtype] = buf.view(dtype)
        out.append(v.as_strided(shape, strides, eoff))
    return out


def _bufs(lead, rows, cols, shuf, like, raw=False):
    """A copy's E4M3 codes [*lead, rows, cols] (16x16 blocks when ``shuf``) and E8M0 scales, packed [*lead,
    ceil(rows/32), cols/128, 32, 4] or (``raw``) [*lead, rows, cols/32], in one allocation (``_carve``)."""
    if shuf and rows % 16:
        raise ValueError(f"mxfp8_quant_gfx1250: codes in 16x16 blocks need rows % 16 == 0, got {rows}")
    sshape = (rows, cols // 32) if raw else (-(-rows // 32), cols // BK, 32, 4)
    lead = tuple(lead)
    return _carve(((lead + (rows, cols), torch.float8_e4m3fn), (lead + sshape, _E8)), like.device)


def _copies(rows, cols, layout):
    """(the kernel's OUT, its LAY) of the copies asked for."""
    if layout & ~(SHUF_ROW | SHUF_COL) or not (rows or cols):
        raise ValueError(
            f"mxfp8_quant_gfx1250: layout is SHUF_ROW | SHUF_COL and a copy must be asked for, got layout "
            f"{layout}, rows {rows}, cols {cols}"
        )
    out = (O_ROWS if rows else 0) | (O_COLS if cols else 0)
    return out, _PACK | (layout & ((SHUF_ROW if rows else 0) | (SHUF_COL if cols else 0)))


def _dense(x, out_dtype, blk_2d, out, lay):
    x = _check(x, out_dtype, (2, 3))
    *b, M, K = x.shape
    Mc = _ceil128(M)
    qr, sr = _bufs(b, M, _ceil128(K), lay & SHUF_ROW, x, not lay & _PACK) if out & O_ROWS else (None, None)
    qc, sc = _bufs(b, K, Mc, lay & SHUF_COL, x) if out & O_COLS else (None, None)
    _run(P_DENSE, out, lay, x, (qr, sr, qc, sc, None, None), M, Mc, K, int(blk_2d))
    return qr, sr, qc, sc


def quant_mxfp8(x, out_dtype=torch.float8_e4m3fn, blk_2d=False, rows=True, cols=True, layout=0):
    """Dual cast of x [M, K] or [B, M, K] bf16, read once: (q_row [.., M, K'], s_row, q_col [.., K, M'], s_col) in
    ``layout``, K' = ceil128(K), M' = ceil128(M). ``blk_2d``: 32 x 32 blocks (weights), one scale per block shared
    by both copies; otherwise 1 x 32 blocks along each copy's rows. ``rows`` / ``cols`` False leave that copy out
    (None)."""
    return _dense(x, out_dtype, blk_2d, *_copies(rows, cols, layout))


def quant_mxfp8_rows(x, out_dtype=torch.float8_e4m3fn):
    """MegaMoE's wire format, the row copy of x [M, K] with raw scales: (q [M, K'], s [M, K' / 32] e8m0)."""
    return _dense(x, out_dtype, False, O_ROWS, 0)[:2]


def _grouped(prod, x, group_lens, group_offs, k, M, layout, rows=True, cols=True, **kw):
    G = group_lens.shape[0]
    if group_offs.dtype != torch.int64 or group_offs.shape[0] != G + 1 or G == 0 or k == 0:
        raise ValueError(
            f"mxfp8_quant_gfx1250: group_offs must be int64 [G + 1] for G > 0 experts of width K > 0, got "
            f"{group_offs.dtype} {tuple(group_offs.shape)}, G = {G}, K = {k}"
        )
    out, lay = _copies(rows, cols, layout)
    if lay & SHUF_ROW:
        raise ValueError("mxfp8_quant_gfx1250: an expert row copy is a GEMM A operand: no SHUF_ROW")
    OW = (2 if prod == P_DGLU else 1) * _ceil128(k)  # dGLU (k % 128 == 0): dgate | dup
    Mr, Mp = -(-(M + 32 * G) // 32) * 32, -(-(M + BM * G) // BM) * BM  # bounds of the padded rows / tokens
    qr, sr = _bufs((), Mr, OW, False, x) if rows else (None, None)
    qc, sc = _bufs((), OW if prod == P_DGLU else k, Mp, lay & SHUF_COL, x) if cols else (None, None)
    lc, lr = _carve((((2, G + 1), torch.int64), ((2, G + 1), torch.int64)), x.device)
    _run(prod, out, lay, x, (qr, sr, qc, sc, lc, lr), M, Mp, k, go=group_offs.contiguous(), **kw)
    return qr, sr, qc, sc, lr[0, :G], lr[1], lc[0, :G], lc[1]


def grouped_quant_mxfp8(
    x, group_lens, group_offs, out_dtype=torch.float8_e4m3fn, rows=True, cols=True, layout=0
):
    """MoE dual cast of x [M, K] (experts' tokens consecutive, ``group_offs`` [G + 1] int64), HIP's
    ``grouped_quantize_mxfp8_dual`` contract: the row copy per expert padded to 32 rows, the column copy per expert
    padded to 128 tokens (pad rows / blocks: zero codes, scale 127). Returns (q_row [Mr, K'], s_row, q_col [K, Mp],
    s_col, lens_row [G], offs_row [G + 1], lens_col [G], offs_col [G + 1]) with K' = ceil128(K), Mr = ceil32(M +
    32 G), Mp = ceil128(M + 128 G): bounds (no host sync); rows past offs_row[G] / columns past offs_col[G] are
    not written, nor are tokens past group_offs[G]. ``rows`` / ``cols`` False leave that copy out (None); a row
    copy's codes are row-major (no ``SHUF_ROW``)."""
    x = _check(x, out_dtype, (2,))
    return _grouped(P_GROUPED, x, group_lens, group_offs, x.shape[1], x.shape[0], layout, rows, cols)


def _glu_args(l1, M, I, probs, vals, width, activation, clamp_limit):
    """The fused GLU / dGLU's inputs checked (what bounds the kernel's loads and stores) -> its ``_run`` keywords."""
    if tuple(l1.shape) != (M, 2 * I) or I % BK:
        raise ValueError(
            f"mxfp8_quant_gfx1250: l1 must be [M, 2I] with I % 128 == 0, got {tuple(l1.shape)}, I = {I}"
        )
    if tuple(probs.shape) != (M,) or probs.dtype != torch.float32:
        raise ValueError(
            f"mxfp8_quant_gfx1250: probs must be fp32 [{M}], got {probs.dtype} {tuple(probs.shape)}"
        )
    v = vals
    if v is not None and (tuple(v.shape) != (M, width) or v.dtype != torch.bfloat16 or not v.is_contiguous()):
        raise ValueError(
            f"mxfp8_quant_gfx1250: the bf16 output must be contiguous bf16 [{M}, {width}], got {v.dtype} "
            f"{tuple(v.shape)}"
        )
    act, clamp = _check_activation(activation), _check_clamp_limit(clamp_limit)
    return dict(p=probs.contiguous(), dbg=vals, act=act, clamp=clamp)


def grouped_glu_quant_mxfp8(
    l1,
    probs,
    group_lens,
    group_offs,
    activation="silu",
    clamp_limit=None,
    out_dtype=torch.float8_e4m3fn,
    act_out=None,
    layout=0,
    cols=True,
):
    """GLU forward + MoE dual cast: act = bf16(f(min(gate, L)) * clamp(up, -L, L) * probs[row]) from
    l1 [M, 2I] bf16 (gate | up, Turbo's fc1 layout; I % 128 == 0; probs [M] fp32), quantized as
    ``grouped_quant_mxfp8`` would quantize it (the 8-tuple, act [M, I]). ``cols`` False leaves the column copy out
    (None). ``act_out`` [M, I] bf16: also receives act (the tests check the quantization against it)."""
    l1 = _check(l1, out_dtype, (2,))
    M, I = l1.shape[0], l1.shape[1] // 2
    kw = _glu_args(l1, M, I, probs, act_out, I, activation, clamp_limit)
    return _grouped(P_GLU, l1, group_lens, group_offs, I, M, layout, cols=cols, **kw)


def grouped_dglu_quant_mxfp8(
    grad_act,
    l1,
    probs,
    group_lens,
    group_offs,
    activation="silu",
    clamp_limit=None,
    out_dtype=torch.float8_e4m3fn,
    grad_out=None,
    layout=0,
):
    """GLU backward + MoE dual cast: from grad_act [M, I] bf16 (the fc2 dgrad; I % 128 == 0), l1 [M, 2I] and
    probs [M]: grad_l1 = [dgate | dup] [M, 2I] (bf16, then quantized like ``grouped_quant_mxfp8``; the column
    copy [2I, Mp]) and grad_probs [M] fp32 = sum_j grad_act * f(gate) * up (per-128-column partials,
    summed in a fixed order). Returns (grad_probs, *the 8-tuple). ``grad_out`` [M, 2I] bf16: also receives
    grad_l1 (the tests check the quantization against it)."""
    grad_act, l1 = _check(grad_act, out_dtype, (2,)), _check(l1, out_dtype, (2,))
    M, I = grad_act.shape
    kw = _glu_args(l1, M, I, probs, grad_out, 2 * I, activation, clamp_limit)
    part = torch.zeros(I // BK, M, dtype=torch.float32, device=l1.device)
    out = _grouped(P_DGLU, grad_act, group_lens, group_offs, I, M, layout, x2=l1, gp=part, **kw)
    return (part.sum(0), *out)


def _row_map(what, idx, M):
    if idx.dtype != torch.int32 or idx.numel() < M:
        raise ValueError(
            f"mxfp8_quant_gfx1250: {what} must be int32 with at least {M} entries, got {idx.dtype} [{idx.numel()}]"
        )
    return idx.contiguous()


def grouped_gather_quant_mxfp8(
    x,
    dest2src,
    group_lens,
    group_offs,
    permuted_probs=None,
    out_dtype=torch.float8_e4m3fn,
    total_m=None,
    layout=0,
):
    """Permute (gather) + MoE dual cast: row d of the permuted activation is x[dest2src[d]] (x [T, H]
    bf16, H % 8 == 0, dest2src [M] int32, read by per-row async loads); a row whose dest2src is out of [0, T), or
    whose ``permuted_probs`` is exactly 0 (the routing padding rule), is zero. Returns the 8-tuple of
    ``grouped_quant_mxfp8`` applied to the permuted [M, H] activation. Serves the backward too
    (dOut rows -> dY2, no probs)."""
    x = _check(x, out_dtype, (2,))
    T, H = x.shape
    M = int(total_m if total_m is not None else dest2src.shape[0])
    pp = permuted_probs
    if H % 8:
        raise ValueError(f"mxfp8_quant_gfx1250: the gather loads 16-byte row chunks, H % 8 == 0, got {H}")
    if pp is not None and (pp.dtype != torch.float32 or pp.numel() < M):
        raise ValueError(
            f"mxfp8_quant_gfx1250: permuted_probs must be fp32 with at least {M} entries, got {pp.dtype} [{pp.numel()}]"
        )
    pp = None if pp is None else pp.contiguous()
    d2s = _row_map("dest2src", dest2src, M)
    return _grouped(P_GATHER, x, group_lens, group_offs, H, M, layout, x2=d2s, p=pp, t=T)


def grouped_gather_fp8_quant_mxfp8(q, s, row_map, group_lens, group_offs, total_m=None, cols=True, layout=0):
    """Gather of MX rows + MoE dual cast (MegaMoE: the dispatched rows into pool order): row d is the E4M3 row
    ``q[row_map[d]]`` with its raw E8M0 scales ``s[row_map[d]]`` (``q [T, K]``, ``s [T, K / 32]``, ``row_map`` int32;
    out of [0, T): a zero row). The row copy keeps each row's codes and scales, the column copy quantizes the
    dequantized values, exact (up to E4M3 subnormals). Returns the 8-tuple of ``grouped_quant_mxfp8``; ``cols``
    False leaves the column copy out (None)."""
    T, K = q.shape
    if q.dtype != torch.float8_e4m3fn or not (q.is_contiguous() and s.is_contiguous()):
        raise ValueError(f"mxfp8_quant_gfx1250: contiguous E4M3 rows and scales, got {q.dtype}")
    if K % BK or tuple(s.shape) != (T, K // 32) or T * K >= 2**31:
        raise ValueError(
            f"mxfp8_quant_gfx1250: q [T, K] with K % 128 == 0 and T * K < 2**31, s [T, K / 32], got "
            f"{tuple(q.shape)}, {tuple(s.shape)}"
        )
    M = int(total_m if total_m is not None else row_map.shape[0])
    aux = dict(x2=_row_map("row_map", row_map, M), p=s.view(torch.uint8), t=T)
    return _grouped(P_FP8, q, group_lens, group_offs, K, M, layout, cols=cols, **aux)
