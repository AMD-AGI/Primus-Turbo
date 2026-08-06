###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""FlyDSL MXFP8 (per-1x32 E8M0) quantization kernels for the fused mega MoE.

Every operand the mxfp8 mega GEMMs consume is block-scaled with one E8M0 byte per 32
elements along the contracted dim.  This module holds all of it: the shared block math,
the rowwise (along-K) quant used by activations and weights, and the colwise (along-M)
transpose-quant the variable-K wgrads need.

E8M0 + fp8 math (mirrors the production HIP kernel
``csrc/kernels/quantization/quantization_mxfp8.hip``, ``compute_tile_scale``)::

    exp = ((float_as_uint(amax) + round_add) >> 23 & 0x1ff) - 127 - target_pow2  # round-even
    exp = clamp(exp, -127, 128);  e8m0_byte = exp + 127;  scale = 2^exp
    q_i = round_to_fp8(x_i / scale)                       # soft-clamped to the fp8 max first

``round_add``/``target_pow2``/the clamp follow the output encoding: E4M3 (``cvt_pk_fp8_f32``,
max 448, target 2^8) or E5M2 (``cvt_pk_bf8_f32``, max 57344, target 2^15; the dW2 default,
matching the grad range).

ROWWISE (along K): one thread owns one 1x32 K-block, so there is no cross-lane reduction.
Optionally fuses the A-scale ScaleS2R preshuffle (``pack=4`` variant repacks through LDS) so
the GEMM's scale operand comes straight out of the quant kernel.

COLWISE (along M, transposed out): dW2/dW1 contract over M, so their operands need the E8M0
block to group 32 consecutive M rows and the fp8 output to be the transpose ``[F, M]``.  The
production C++ ``grouped_quantize_mxfp8_dual`` also emits the (here unused) rowwise half; at
the DSv3 dW2 shape that costs ~1.0 ms and makes fp8 dW2 net-negative vs bf16.  Emitting only
the colwise operand roughly halves the write traffic (~0.61 ms for both dW2 operands, 1.6x,
byte-exact vs the dual).  Each thread owns one output column and privately reduces its 32
M-values; consecutive threads own consecutive columns so the reads stay coalesced.  One
workgroup handles one (32-M block, ``BT``-wide F-tile): the kernel is memory-*latency* bound
(VALU and VMEM issue are <1% of runtime), so the large grid is what hides the strided-load
latency -- an earlier single-workgroup-per-column version left ~2 waves/SIMD and ran 1.4x
slower.  A fp8-in variant fuses the dequant->requant round-trip for the L2-dgrad pool operand.
"""

import functools
from typing import Optional

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl._mlir.dialects import vector as _vector
from flydsl.expr import arith, range_constexpr
from flydsl.expr import math as fmath
from flydsl.expr.buffer_ops import buffer_load, buffer_store, create_buffer_resource
from flydsl.expr.rocdl import cvt_f32_fp8, cvt_pk_bf8_f32, cvt_pk_fp8_f32
from flydsl.expr.typing import Vector as Vec

from primus_turbo.flydsl.mega.fp8.gemm_helper import (
    _PRESHUF_KT,
    build_preshuffle_ab_kernel,
    ceildiv,
)

MXFP8_BLOCK = 32
_BLK = 32  # mxfp8 block (elements per E8M0 scale)
_VEC = 8   # sub-vector width for the bf16 load / f32 compute
# Match grouped GEMM / gemm_mxfp8_nt_tile (_SCALE_PACK).
_SCALE_PACK = 4


# ── shared MXFP8 block math ───────────────────────────────────────────────────────────────


def _mxfp8_words_from_f32_subvecs(fvs):
    """Quantize one 1x32 f32 block given as ``subs`` vectors of width ``_VEC`` to MXFP8.

    Returns ``(words, biased)`` (same layout as ``_quant_block_words``). Used by the standalone
    quant kernel and the fused SwiGLU+mxfp8 epilogue."""
    f32v = fx.T.VectorType.get([_VEC], fx.T.f32())
    neg1 = fx.arith.constant_vector(-1.0, f32v)
    lim = fx.arith.constant_vector(448.0, f32v)
    neglim = fx.arith.constant_vector(-448.0, f32v)
    subs = _BLK // _VEC  # 4

    sub_amax = []
    for s in range_constexpr(subs):
        fv = fvs[s]
        av = fx.arith.maximumf(fv, fx.arith.mulf(fv, neg1))  # |fv|
        sub_amax.append(
            fx.arith.ArithValue(_vector.reduction(fx.T.f32(), _vector.CombiningKind.MAXIMUMF, av))
        )
    amax = sub_amax[0]
    for s in range_constexpr(1, subs):
        amax = fx.arith.maximumf(amax, sub_amax[s])

    # E8M0 scale: round-even exponent, target 2^8, clamp to E8M0 range.
    amax_bits = fx.arith.ArithValue(amax).bitcast(fx.T.i32())
    t = amax_bits + fx.Int32(1 << 19)
    exp = ((t >> fx.Int32(23)) & fx.Int32(0x1FF)) - fx.Int32(127 + 8)
    exp = fx.arith.select(exp < fx.Int32(-127), fx.Int32(-127), exp)
    exp = fx.arith.select(exp > fx.Int32(128), fx.Int32(128), exp)
    biased = fx.arith.ArithValue(exp) + fx.Int32(127)
    # Build 1/scale from the exponent bits, as _e8m0_quant_pack does, rather than dividing by
    # `bits(biased << 23)`. The two agree bit-for-bit on every scale a finite input can produce,
    # but they differ on the one an all-zero block produces: there amax is 0, exp clamps to -127
    # and biased is 0, so the float form is 1.0/0.0 = inf and every element quantizes to 0*inf =
    # NaN. A token the loss masks out has an exactly-zero gradient row, so training hits this
    # where fixed random benchmark data never does. 2^(127-biased) is also what the stored E8M0
    # byte actually means, which the divide-by-zero form was not.
    inv_scale = fx.arith.ArithValue((fx.Int32(254) - biased) << fx.Int32(23)).bitcast(fx.T.f32())
    inv_v = _vector.broadcast(f32v, arith._to_raw(inv_scale))

    words = []
    for s in range_constexpr(subs):
        qraw = fx.arith.mulf(fvs[s], inv_v)
        qf = Vec(fx.arith.minimumf(fx.arith.maximumf(qraw, neglim), lim))  # soft-clamp to fp8 max
        e = [qf[i] for i in range_constexpr(_VEC)]
        w0 = cvt_pk_fp8_f32(fx.T.i32(), e[0], e[1], fx.Int32(0), False)
        w0 = cvt_pk_fp8_f32(fx.T.i32(), e[2], e[3], w0, True)
        w1 = cvt_pk_fp8_f32(fx.T.i32(), e[4], e[5], fx.Int32(0), False)
        w1 = cvt_pk_fp8_f32(fx.T.i32(), e[6], e[7], w1, True)
        words.append(w0)
        words.append(w1)
    return words, biased


def _quant_block_words(xr, base_elem):
    """Quantize one 1x32 bf16 block at ``xr[base_elem : base_elem+32]`` to MXFP8.

    Returns ``(words, biased)``: ``words`` = 8 i32 packing the 32 fp8 (E4M3) values
    (4 fp8/word, via the HW ``cvt_pk_fp8_f32``); ``biased`` = the E8M0 scale byte in an
    i32. Mirrors the production ``compute_tile_scale`` (round-even exp, target 2^8,
    soft-clamp before cvt). Shared by the standalone quant kernel and the fused push."""
    f32v = fx.T.VectorType.get([_VEC], fx.T.f32())
    subs = _BLK // _VEC  # 4

    fvs = []
    for s in range_constexpr(subs):
        vv = buffer_load(xr, base_elem + fx.Int32(s * _VEC), vec_width=_VEC, dtype=fx.T.bf16())
        fvs.append(fx.arith.extf(f32v, vv))
    return _mxfp8_words_from_f32_subvecs(fvs)


def _e8m0_quant_pack(vals, round_add, target_pow2, lo, hi, cvt, zero_i32):
    """Scalar-input MXFP8 block math: 32 f32 ``vals`` -> (8 packed i32 fp8 words, E8M0 biased byte).

    Encoding-parametric twin of ``_mxfp8_words_from_f32_subvecs`` (which is hard-wired to E4M3
    and takes ``_VEC``-wide vectors).  Used by the colwise kernels, whose 32 values come from a
    strided gather rather than a contiguous vector load."""
    amax = None
    for fv in vals:
        a = fmath.absf(fv)
        amax = a if amax is None else fx.arith.maximumf(amax, a)
    amax_bits = fx.arith.ArithValue(amax).bitcast(fx.T.i32())
    t = amax_bits + fx.Int32(round_add)
    exp = ((t >> fx.Int32(23)) & fx.Int32(0x1FF)) - fx.Int32(127 + target_pow2)
    exp = fx.arith.select(exp < fx.Int32(-127), fx.Int32(-127), exp)
    exp = fx.arith.select(exp > fx.Int32(128), fx.Int32(128), exp)
    biased = fx.arith.ArithValue(exp) + fx.Int32(127)
    # scale = 2^(biased-127) is an exact power of two, so 1/scale = 2^(127-biased) = float bits
    # ((254 - biased) << 23). Bit-identical to 1.0/scale (IEEE div by pow2 is exact) but replaces
    # a VALU fdiv sequence with one sub+shift. Valid for every finite scale (exp<128).
    inv_scale = fx.arith.ArithValue((fx.Int32(254) - biased) << fx.Int32(23)).bitcast(fx.T.f32())
    qs = []
    for fv in vals:
        q = fmath.clampf(fx.arith.ArithValue(fv) * inv_scale, lo, hi)
        qs.append(fx.arith._to_raw(q))
    words = []
    for wi in range_constexpr(_BLK // 4):
        j = wi * 4
        w = cvt(fx.T.i32(), qs[j], qs[j + 1], zero_i32, False)
        w = cvt(fx.T.i32(), qs[j + 2], qs[j + 3], w, True)
        words.append(w)
    return words, biased


def _e8m0_broadcast_i32(biased):
    """Broadcast the E8M0 byte (i32) into all 4 bytes of an i32 (ScaleS2R operand)."""
    bb = fx.arith.ArithValue(biased) & fx.Int32(0xFF)
    return bb | (bb << fx.Int32(8)) | (bb << fx.Int32(16)) | (bb << fx.Int32(24))


# ── A-scale (ScaleS2R) preshuffle slot indices ────────────────────────────────────────────


def _preshuffle_a_idx(dest_row, b, K128):
    """ScaleS2R layout-1 slot for row ``dest_row``, micro-block ``b``:
    ``((grp*K128 + gk)*64 + (g*16+r))*4 + s``, grp=row//64, s=(row%64)//16, r=row%16,
    gk=b//4, g=b%4. Returns the i32 element index into the broadcast a_sp buffer."""
    grp = dest_row // fx.Int32(64)
    s_row = (dest_row % fx.Int32(64)) // fx.Int32(16)
    r_row = dest_row % fx.Int32(16)
    gk = b // fx.Int32(4)
    g = b % fx.Int32(4)
    lane = g * fx.Int32(16) + r_row
    return ((grp * fx.Int32(K128) + gk) * fx.Int32(64) + lane) * fx.Int32(4) + s_row


def _preshuffle_a_pack4_idx(dest_row, kkp, g, K128p):
    """ScaleS2R pack=4 slot for row ``dest_row``, packed-K index ``kkp``, micro-group ``g``."""
    grp = dest_row // fx.Int32(64)
    s_row = (dest_row % fx.Int32(64)) // fx.Int32(16)
    r_row = dest_row % fx.Int32(16)
    lane = g * fx.Int32(16) + r_row
    return ((grp * fx.Int32(K128p) + kkp) * fx.Int32(64) + lane) * fx.Int32(4) + s_row


# ── rowwise (along-K) quant ───────────────────────────────────────────────────────────────


@functools.lru_cache(maxsize=32)
def _compile_quant(K: int, BT: int = 256, preshuffle: bool = False):
    assert K % _BLK == 0, f"K={K} must be a multiple of {_BLK}"
    n_blk = K // _BLK
    K128 = K // 128
    K_fp8_i32 = K // 4
    blk_i32 = _BLK // 4

    @flyc.kernel(known_block_size=[BT, 1, 1])
    def kern(X: fx.Tensor, Q: fx.Tensor, S: fx.Tensor, c_m: fx.Int32):
        tid = fx.thread_idx.x
        row = fx.block_idx.x

        xr = create_buffer_resource(X, max_size=True)
        qr = create_buffer_resource(Q, max_size=True)
        sr = create_buffer_resource(S, max_size=True)

        b = tid
        while b < fx.Int32(n_blk):
            base = row * fx.Int32(K) + b * fx.Int32(_BLK)
            words, biased = _quant_block_words(xr, base)

            if preshuffle:
                buffer_store(_e8m0_broadcast_i32(biased), sr, _preshuffle_a_idx(row, b, K128))
            else:
                buffer_store(fx.arith.ArithValue(biased).trunci(fx.T.i8()), sr, row * fx.Int32(n_blk) + b)

            base_i32 = row * fx.Int32(K_fp8_i32) + b * fx.Int32(blk_i32)
            for wi in range_constexpr(blk_i32):
                buffer_store(words[wi], qr, base_i32 + fx.Int32(wi))

            b = b + fx.Int32(BT)

    @flyc.jit
    def launch(X: fx.Tensor, Q: fx.Tensor, S: fx.Tensor, M: int, stream: fx.Stream = fx.Stream(None)):
        kern(X, Q, S, M).launch(grid=(M, 1, 1), block=(BT, 1, 1), stream=stream)

    return launch


@functools.lru_cache(maxsize=32)
def _compile_quant_preshuffle_pack4(K: int, BT: int = 256, scale_pack: int = 4):
    """Quant bf16 row + fused pack-4 A-scale preshuffle (single kernel, LDS repack)."""
    assert K % _BLK == 0, f"K={K} must be a multiple of {_BLK}"
    n_blk = K // _BLK
    K128 = K // 128
    K128p = ceildiv(K128, scale_pack)
    K_fp8_i32 = K // 4
    blk_i32 = _BLK // 4

    @fx.struct
    class RowSmem:
        raw: fx.Array[fx.Int8, n_blk, 16]

    @flyc.kernel(known_block_size=[BT, 1, 1])
    def kern(X: fx.Tensor, Q: fx.Tensor, S: fx.Tensor, c_m: fx.Int32):
        tid = fx.thread_idx.x
        row = fx.block_idx.x
        smem = fx.SharedAllocator().allocate(RowSmem).peek().raw

        xr = create_buffer_resource(X, max_size=True)
        qr = create_buffer_resource(Q, max_size=True)
        sr = create_buffer_resource(S, max_size=True)

        b = tid
        while b < fx.Int32(n_blk):
            base = row * fx.Int32(K) + b * fx.Int32(_BLK)
            words, biased = _quant_block_words(xr, base)
            smem[b] = fx.arith.ArithValue(biased).trunci(fx.T.i8())

            base_i32 = row * fx.Int32(K_fp8_i32) + b * fx.Int32(blk_i32)
            for wi in range_constexpr(blk_i32):
                buffer_store(words[wi], qr, base_i32 + fx.Int32(wi))

            b = b + fx.Int32(BT)

        fx.rocdl.s_barrier()
        grp = row // fx.Int32(64)
        r_row = row % fx.Int32(16)
        s_row = (row % fx.Int32(64)) // fx.Int32(16)
        n_out = K128p * 4
        n_rounds = ceildiv(n_out, BT)
        for pi in range_constexpr(n_rounds):
            idx = tid + pi * BT
            if idx < fx.Int32(n_out):
                kkp = idx // fx.Int32(4)
                g = idx % fx.Int32(4)
                lane = g * fx.Int32(16) + r_row
                packed = fx.Int32(0)
                for bb in range_constexpr(scale_pack):
                    ki = kkp * fx.Int32(scale_pack) + fx.Int32(bb)
                    raw_b = ki * fx.Int32(4) + g
                    scale_byte = fx.arith.ArithValue(smem[raw_b]).extui(fx.T.i32())
                    packed = packed | ((scale_byte & fx.Int32(0xFF)) << (fx.Int32(bb) * fx.Int32(8)))
                    out_idx = ((grp * fx.Int32(K128p) + kkp) * fx.Int32(64) + lane) * fx.Int32(4) + s_row
                buffer_store(packed, sr, out_idx)

    @flyc.jit
    def launch(X: fx.Tensor, Q: fx.Tensor, S: fx.Tensor, M: int, stream: fx.Stream = fx.Stream(None)):
        kern(X, Q, S, M).launch(grid=(M, 1, 1), block=(BT, 1, 1), stream=stream)

    return launch


_QUANT_COMPILED: dict = {}
_BSCALE_PS_COMPILED: dict = {}


def preshuffle_b_scale(b_scale: torch.Tensor, G: int, N: int, K: int, *, pack: int = _SCALE_PACK):
    """Host preshuffle of a grouped weight E8M0 scale into the ScaleBComb layout-3 ``b_sp``.

    ``b_scale`` = raw E8M0 [G, N, K//32] (or [G*N, K//32]) uint8 -> ``b_sp`` int32
    [b_ngrp*K128*256], b_ngrp=ceildiv(G*N,256)*4, read by ``ScaleBComb``. Runs the shared
    ``build_preshuffle_ab_kernel`` (B region only; A is a 64-row dummy). Weights are static,
    so callers cache the result per (G,N,K)."""
    GN = G * N
    K128 = K // 128
    K128p = ceildiv(K128, pack)
    dev = b_scale.device
    b_raw = b_scale.contiguous().reshape(GN, K // 32).view(torch.int32).reshape(-1)
    a_ngrp = 1
    a_blocks = a_ngrp * ceildiv(K128, _PRESHUF_KT)
    b_ngrp = ((GN + 255) // 256) * 4
    a_raw = torch.zeros(64 * K128, dtype=torch.int32, device=dev)  # dummy A (64 rows)
    a_sp = torch.zeros(a_ngrp * K128p * 256, dtype=torch.int32, device=dev)
    b_sp = torch.zeros(b_ngrp * K128p * 256, dtype=torch.int32, device=dev)
    pre_kern, n_kt = build_preshuffle_ab_kernel(K128, pack=pack)

    @flyc.jit
    def _launch(a_raw, b_raw, a_sp, b_sp, a_blocks: fx.Int32, a_ngrp: fx.Int32, b_ngrp: fx.Int32,
                stream: fx.Stream = fx.Stream(None)):
        pre_kern(a_raw, b_raw, a_sp, b_sp, fx.Int32(64), fx.Int32(GN), a_blocks, a_ngrp, b_ngrp).launch(
            grid=(a_blocks + b_ngrp * n_kt, 1, 1), block=(256, 1, 1), stream=stream
        )

    args = (a_raw, b_raw, a_sp, b_sp, a_blocks, a_ngrp, b_ngrp, torch.cuda.current_stream())
    ck = (GN, K128, pack)
    compiled = _BSCALE_PS_COMPILED.get(ck)
    if compiled is None:
        compiled = flyc.compile(_launch, *args)
        _BSCALE_PS_COMPILED[ck] = compiled
    compiled(*args)
    return b_sp


def quantize_rowwise_mxfp8_flydsl(x: torch.Tensor, preshuffle: bool = False, scale_pack: int = _SCALE_PACK):
    """Rowwise MXFP8 quant of ``x`` [M, K] bf16 in one FlyDSL kernel.

    ``preshuffle=False``: returns ``(q fp8 [M,K], s uint8 [M, K//32])`` raw E8M0.
    ``preshuffle=True``: returns ``(q, a_sp)`` with A-scale in the
    ScaleS2R broadcast layout; ``scale_pack=4`` (default) fuses pack-4 preshuffle into the quant
    kernel (no separate host preshuffle pass)."""
    assert x.dim() == 2 and x.dtype == torch.bfloat16
    M, K = x.shape
    x = x.contiguous()
    q = torch.empty((M, K), dtype=torch.float8_e4m3fn, device=x.device)
    q_i32 = q.view(torch.int32)
    if preshuffle:
        K128 = K // 128
        K128p = ceildiv(K128, scale_pack)
        a_ngrp = ceildiv(M, 64)
        a_sp = torch.zeros(a_ngrp * K128p * 256, dtype=torch.int32, device=x.device)
        if scale_pack == 4:
            launch = _compile_quant_preshuffle_pack4(int(K), scale_pack=int(scale_pack))
        else:
            launch = _compile_quant(int(K), preshuffle=True)
        args = (x, q_i32, a_sp, M, torch.cuda.current_stream())
        ck = (M, K, True, int(scale_pack))
        compiled = _QUANT_COMPILED.get(ck)
        if compiled is None:
            compiled = flyc.compile(launch, *args)
            _QUANT_COMPILED[ck] = compiled
        compiled(*args)
        return q, a_sp
    s = torch.empty((M, K // _BLK), dtype=torch.uint8, device=x.device)
    launch = _compile_quant(int(K), preshuffle=False)
    args = (x, q_i32, s, M, torch.cuda.current_stream())
    ck = (M, K, False)
    compiled = _QUANT_COMPILED.get(ck)
    if compiled is None:
        compiled = flyc.compile(launch, *args)
        _QUANT_COMPILED[ck] = compiled
    compiled(*args)
    return q, s


def quantize_grouped_weight_mxfp8_flydsl(w: torch.Tensor):
    """Per-group MXFP8 quant of grouped weights ``[G, N, K]`` along K (block=32), E4M3.

    Rowwise-along-K quant is per-row independent, so group boundaries don't matter:
    ``[G, N, K] -> [G*N, K]`` and run the rowwise kernel above (~5.9 TB/s, near HBM peak vs the
    generic ~2.3 TB/s), then reshape back -- one kernel instead of a ``G``-launch Python loop
    (~2 ms at G=32 / DSv3 w1, a static-weight cost otherwise paid every step). The scale is
    viewed as ``float8_e8m0fnu`` (byte-identical raw E8M0). Returns
    ``(w_fp8 [G,N,K] e4m3, w_scale [G,N,K//32] e8m0)``."""
    assert w.dim() == 3, f"expected 3D [G,N,K], got {tuple(w.shape)}"
    G, N, K = w.shape
    q, s = quantize_rowwise_mxfp8_flydsl(w.reshape(G * N, K))  # q e4m3 [G*N,K], s uint8 [G*N,K//32]
    return q.view(G, N, K), s.view(torch.float8_e8m0fnu).view(G, N, K // MXFP8_BLOCK)


# ── colwise (along-M) transpose-quant, bf16 in ────────────────────────────────────────────


@functools.lru_cache(maxsize=64)
def _compile_colwise_quant_grouped(F: int, is_e5m2: bool, BT: int = 256):
    """Grouped colwise MXFP8 transpose-quant.  One workgroup == one (padded 32-M block,
    BT-wide F-tile).  ``blk2grp[pmb]`` -> group g; the block's 32 padded-M rows map to input
    rows ``group_offs[g] + (pmb*32 - offs_pc[g] + i)`` when that local row < ``group_lens[g]``,
    else they are pad (value 0 => fp8 0).  Runtime row strides (padded total M) come in as the
    ``mpad_i32`` / ``npblk`` scalar args."""
    assert F % BT == 0, f"F={F} must be a multiple of BT={BT}"
    blk_i32 = _BLK // 4
    n_ftile = F // BT
    fp8_max = 57344.0 if is_e5m2 else 448.0
    cvt = cvt_pk_bf8_f32 if is_e5m2 else cvt_pk_fp8_f32
    mbits = 2 if is_e5m2 else 3
    round_add = 1 << (22 - mbits)
    target_pow2 = 15 if is_e5m2 else 8

    @flyc.kernel(known_block_size=[BT, 1, 1])
    def kern(
        X: fx.Tensor, Q: fx.Tensor, S: fx.Tensor,
        BLK2GRP: fx.Tensor, LENS: fx.Tensor, OFFS: fx.Tensor, OFFS_PC: fx.Tensor,
        mpad_i32: fx.Int32, npblk: fx.Int32,
    ):
        tid = fx.thread_idx.x
        bid = fx.block_idx.x
        pmb = bid // fx.Int32(n_ftile)                       # padded 32-M block index
        f = (bid % fx.Int32(n_ftile)) * fx.Int32(BT) + tid   # output column (input free-col)

        xr = create_buffer_resource(X, max_size=True)         # bf16 [total_M, F]
        qr = create_buffer_resource(Q, max_size=True)         # fp8 [F, total_M_pad] i32 [F, mpad_i32]
        sr = create_buffer_resource(S, max_size=True)         # raw uint8 [F, npblk]
        b2g = create_buffer_resource(BLK2GRP, max_size=True)  # i32 [npblk] -> group id
        lr = create_buffer_resource(LENS, max_size=True)      # i32 [G] unpadded lens
        ofr = create_buffer_resource(OFFS, max_size=True)     # i32 [G+1] unpadded row offsets
        opr = create_buffer_resource(OFFS_PC, max_size=True)  # i32 [G+1] padded block-row offsets

        lo = fx.arith.constant(-fp8_max, type=fx.T.f32())
        hi = fx.arith.constant(fp8_max, type=fx.T.f32())
        zero_i32 = fx.arith.constant(0, type=fx.T.i32())

        # workgroup-uniform group metadata (pmb == same for all threads in the WG).
        g = buffer_load(b2g, pmb, vec_width=1, dtype=fx.T.i32())
        offs_pc_g = buffer_load(opr, g, vec_width=1, dtype=fx.T.i32())
        in_off_g = buffer_load(ofr, g, vec_width=1, dtype=fx.T.i32())
        len_g = buffer_load(lr, g, vec_width=1, dtype=fx.T.i32())
        m_local0 = pmb * fx.Int32(_BLK) - fx.arith.ArithValue(offs_pc_g)  # M-offset within group

        vals = []
        for i in range_constexpr(_BLK):
            m_local = fx.arith.ArithValue(m_local0) + fx.Int32(i)
            real = m_local < fx.arith.ArithValue(len_g)
            # clamp pad rows to the group's first row (in-bounds); select 0 for the value.
            m_eff = fx.arith.select(real, m_local, fx.Int32(0))
            row = fx.arith.ArithValue(in_off_g) + fx.arith.ArithValue(m_eff)
            v = buffer_load(xr, row * fx.Int32(F) + f, vec_width=1, dtype=fx.T.bf16())
            fv = fx.arith.select(real, fx.arith.extf(fx.T.f32(), v), fx.Float32(0.0))
            vals.append(fx.arith._to_raw(fv))

        words, biased = _e8m0_quant_pack(vals, round_add, target_pow2, lo, hi, cvt, zero_i32)
        base_i32 = f * fx.arith.ArithValue(mpad_i32) + pmb * fx.Int32(blk_i32)
        buffer_store(Vec.from_elements(words[0:4], fx.Int32).ir_value(), qr, base_i32)
        buffer_store(Vec.from_elements(words[4:8], fx.Int32).ir_value(), qr, base_i32 + fx.Int32(4))
        buffer_store(fx.arith.ArithValue(biased).trunci(fx.T.i8()), sr,
                     f * fx.arith.ArithValue(npblk) + pmb)

    @flyc.jit
    def launch(X, Q, S, BLK2GRP, LENS, OFFS, OFFS_PC, mpad_i32, npblk, n_pblk,
               stream: fx.Stream = fx.Stream(None)):
        kern(X, Q, S, BLK2GRP, LENS, OFFS, OFFS_PC, mpad_i32, npblk).launch(
            grid=(n_pblk * n_ftile, 1, 1), block=(BT, 1, 1), stream=stream)

    return launch


_WEIGHT_GENERATION = [0]


def advance_weight_generation() -> None:
    """Invalidate every cached fp8 weight derivative. Call once per optimizer step.

    A weight update has to invalidate THREE caches, and none of them can detect it on its own: the
    quantized weight (was keyed on ``w._version``, which Megatron's precision-aware optimizer never
    bumps) and the flattened / preshuffled copies the GEMM actually contracts (keyed on
    ``w1q.data_ptr()``, which does not move when the quantization is rewritten in place -- and did
    not move even when it was reallocated, because the allocator hands back the block it just
    freed). All three missing the update is why the fp8 experts trained on their step-0 weights.

    Megatron already publishes the right signal: the pipeline schedule calls
    ``model.set_is_first_microbatch()`` on the first microbatch of each step, which is exactly when
    the weight has changed and no microbatch of this step has consumed it yet. Driving one counter
    from there beats three caches each guessing.
    """
    _WEIGHT_GENERATION[0] += 1


def weight_generation() -> int:
    """The current weight generation; include it in any cache key derived from a weight."""
    return _WEIGHT_GENERATION[0]


def colwise_grouped_meta(
    group_lens: torch.Tensor, group_offs: torch.Tensor, pool_rows: Optional[int] = None
):
    """Precompute the (device) grouping metadata for the grouped colwise quant.  Both dW2
    operands share the same group structure, so compute this ONCE and pass to both calls
    (avoids re-running cumsum/repeat_interleave twice).

    ``pool_rows`` is the M extent of the pool the groups live in. Given it, the padded total is
    bounded without reading the device: each group is padded to 128 here while the pool already
    pads each group to BLOCK_M=256, and 128 divides 256, so the 128-padded total can never exceed
    the pool. Taking the bound instead of the exact value removes a D2H per call -- which does not
    just cost the copy but blocks until every kernel already queued has retired, so in a training
    step it serialises the whole pipeline (it measured ~1 s per call, 72% of the op's host time;
    the bf16 path takes no host read at all). Rows past the real groups are masked by ``len_g`` in
    the kernels and never read downstream, which bound per-group offsets from ``offs_pc``.
    """
    dev = group_lens.device
    lens = group_lens.to(torch.int32)
    lens_pc = ((lens + 127) // 128) * 128                                   # pad each group M to 128
    offs_pc = torch.cat(
        [torch.zeros(1, dtype=torch.int32, device=dev), torch.cumsum(lens_pc, 0)]
    ).to(torch.int32)
    if pool_rows is None:
        total_M_pad = int(offs_pc[-1].item())                               # D2H (sizes output/grid)
    else:
        assert int(pool_rows) % _BLK == 0, f"pool_rows {pool_rows} must be a multiple of {_BLK}"
        total_M_pad = int(pool_rows)
    n_pblk = total_M_pad // _BLK
    # group id per 32-M block via searchsorted on the block offsets (fixed output size =>
    # no hidden D2H, unlike repeat_interleave which syncs to size its dynamic output). Blocks past
    # the last real group land on index G, which none of the per-group tables below can hold, so
    # clamp them onto the last group; their rows fail the len_g mask anyway.
    offs32 = group_offs.to(torch.int32)
    blk2grp = (
        torch.searchsorted(
            offs_pc // _BLK, torch.arange(n_pblk, dtype=torch.int32, device=dev), right=True
        )
        - 1
    ).clamp_(0, lens.numel() - 1).to(torch.int32)
    grp = blk2grp
    offs_pc_g = offs_pc[grp]
    in_off_g = offs32[grp]
    len_g = lens[grp]
    m_local0 = torch.arange(n_pblk, dtype=torch.int32, device=dev) * _BLK - offs_pc_g
    # Per-pmb WG metadata [offs_pc_g, in_off_g, len_g, m_local0] — avoids dependent VMEM chain + barrier.
    pmb_meta = torch.stack([offs_pc_g, in_off_g, len_g, m_local0], dim=1).contiguous()
    return {
        "lens": lens, "lens_pc": lens_pc, "offs_pc": offs_pc,
        "offs32": offs32, "blk2grp": blk2grp, "pmb_meta": pmb_meta,
        "total_M_pad": total_M_pad, "n_pblk": n_pblk,
    }


def colwise_quant_mxfp8_grouped_flydsl(
    x: torch.Tensor, out_dtype: torch.dtype,
    group_lens: torch.Tensor = None, group_offs: torch.Tensor = None,
    meta: dict = None, BT: int = 256,
):
    """Grouped colwise (along-M) MXFP8 transpose-quant, per-group M padded to 128.

    Args:
        x: bf16 ``[total_M, F]`` (groups stacked along M).
        group_lens/group_offs: int ``[G]`` / ``[G+1]`` (unpadded) -- used if ``meta`` is None.
        meta: precomputed ``colwise_grouped_meta`` (share across both dW2 operands).

    Returns ``(q, s, lens_pc, offs_pc)`` (drop-in for the wgrad's colwise operand):
        q: fp8 ``[F, total_M_pad]``   s: uint8 ``[F, total_M_pad//32]``.
    """
    assert x.dim() == 2 and x.dtype == torch.bfloat16, "x must be bf16 [total_M, F]"
    M, F = x.shape
    is_e5m2 = out_dtype == torch.float8_e5m2
    if meta is None:
        meta = colwise_grouped_meta(group_lens, group_offs, pool_rows=M)
    total_M_pad, n_pblk = meta["total_M_pad"], meta["n_pblk"]
    q = torch.empty((F, total_M_pad), dtype=out_dtype, device=x.device)
    s = torch.empty((F, n_pblk), dtype=torch.uint8, device=x.device)
    while F % BT != 0:
        BT //= 2
    _compile_colwise_quant_grouped(F, is_e5m2, BT)(
        x, q, s, meta["blk2grp"], meta["lens"], meta["offs32"], meta["offs_pc"],
        total_M_pad // 4, n_pblk, n_pblk,
    )
    return q, s, meta["lens_pc"], meta["offs_pc"]


# ── fp8-in fused dequant->colwise-requant (a-branch producer fusion) ─────────────────────────
# The dW2 `a` operand (`dispatch_l2_grad`) originates from the backward L2-dgrad fp8 pool, which is
# ROWWISE MXFP8 (E4M3 [P,H] + per-1x32-H E8M0 scale [P,H//32]) because that layout is what the
# L2 dgrad MMA contracts. dW2, however, contracts over P and needs the operand quantized
# COLWISE (along P). The current path dequants the pool to a bf16 `dispatch_l2_grad` (HBM
# round-trip) then re-quantizes it colwise. This kernel FUSES both: it reads the rowwise-fp8 pool
# directly, dequants in-register (cvt_f32_fp8 * 2^(e8m0-127)), and emits the colwise (transposed)
# fp8 operand dW2 wants -- eliminating the bf16 intermediate. Numerically identical to
# dequant->requant (the pool fp8 dequants exactly), just without the bf16 materialization.


@functools.lru_cache(maxsize=64)
def _compile_colwise_requant_grouped_fp8in(F: int, is_e5m2_out: bool, BT: int = 128, MB: int = 4):
    """Grouped fp8-in colwise MXFP8 transpose-requant with an LDS OUTPUT-transpose stage.

    One workgroup owns (``MB`` padded 32-M blocks) x (BT-wide F-tile).  Each thread owns one output
    column ``f`` and colwise-quantizes its ``MB*32`` M-values (decode ROWWISE-fp8 E4M3 + raw E8M0
    ``2^(e8m0-127)``, then per-32-M-block amax/requant).

    KEY (write coalescing): the naive path has each thread write its column's fp8 to the transposed
    output ``[F, total_M_pad]`` -- consecutive threads (columns) hit consecutive output ROWS (stride
    ``mpad``), so a wavefront scatters to 64 different cache lines (``TCC_WRREQ`` 3.6x ``RDREQ``).
    Here the computed fp8 is first staged in LDS (``[BT col][MB*8 i32]``), then written back with
    threads re-mapped so 32 consecutive lanes emit 32 consecutive M-i32 of ONE row (128 B, one full
    line) -- turning the scattered per-lane writes into coalesced full-line writes."""
    assert F % BT == 0, f"F={F} must be a multiple of BT={BT}"
    assert F % _BLK == 0, f"F={F} must be a multiple of {_BLK} (rowwise scale columns)"
    blk_i32 = _BLK // 4                # fp8 i32 words per 32-block (=8)
    n_ftile = F // BT
    F32 = F // _BLK                    # rowwise E8M0 scale columns (per input row)
    RPT = MB * blk_i32                 # output i32 per column (MB blocks * 8 words)
    TILE_I32 = BT * RPT                # LDS out-tile size
    assert TILE_I32 % BT == 0
    n_wr = TILE_I32 // BT              # coalesced-write iterations (= RPT)
    SCPT = BT // _BLK                  # rowwise scale columns per F-tile (shared 32-wide)
    MROWS = MB * _BLK                  # M-rows per workgroup
    SC_N = MROWS * SCPT                # scale slab entries (= MB*BT)
    fp8_max = 57344.0 if is_e5m2_out else 448.0
    cvt = cvt_pk_bf8_f32 if is_e5m2_out else cvt_pk_fp8_f32
    mbits = 2 if is_e5m2_out else 3
    round_add = 1 << (22 - mbits)
    target_pow2 = 15 if is_e5m2_out else 8

    @fx.struct
    class Smem:
        outq: fx.Array[fx.Int32, TILE_I32, 16]  # [BT col][MB*8 i32] fp8 output tile (for coalesced write)
        scale: fx.Array[fx.Int32, SC_N, 16]     # [MROWS][SCPT] staged rowwise E8M0 (cut 32x reload)

    @flyc.kernel(known_block_size=[BT, 1, 1])
    def kern(
        XQ: fx.Tensor, XS: fx.Tensor, Q: fx.Tensor, S: fx.Tensor,
        BLK2GRP: fx.Tensor, LENS: fx.Tensor, OFFS: fx.Tensor, OFFS_PC: fx.Tensor,
        mpad_i32: fx.Int32, npblk: fx.Int32,
    ):
        tid = fx.thread_idx.x
        bid = fx.block_idx.x
        mwg = bid // fx.Int32(n_ftile)                       # M-workgroup (owns MB blocks)
        ftile = bid % fx.Int32(n_ftile)
        f = ftile * fx.Int32(BT) + tid                       # output column (input free-col)
        f_base = ftile * fx.Int32(BT)
        pmb0 = mwg * fx.Int32(MB)                             # first padded 32-M block

        xqr = create_buffer_resource(XQ, max_size=True)       # fp8 (i8 view) [total_M, F]
        xsr = create_buffer_resource(XS, max_size=True)       # raw E8M0 uint8 [total_M, F//32]
        qr = create_buffer_resource(Q, max_size=True)         # fp8 [F, total_M_pad] i32 [F, mpad_i32]
        sr = create_buffer_resource(S, max_size=True)         # raw uint8 [F, npblk]
        b2g = create_buffer_resource(BLK2GRP, max_size=True)  # i32 [npblk] -> group id
        lr = create_buffer_resource(LENS, max_size=True)      # i32 [G] unpadded lens
        ofr = create_buffer_resource(OFFS, max_size=True)     # i32 [G+1] unpadded row offsets
        opr = create_buffer_resource(OFFS_PC, max_size=True)  # i32 [G+1] padded block-row offsets
        lds = fx.SharedAllocator().allocate(Smem).peek()
        outq_lds = lds.outq
        scale_lds = lds.scale

        lo = fx.arith.constant(-fp8_max, type=fx.T.f32())
        hi = fx.arith.constant(fp8_max, type=fx.T.f32())
        zero_i32 = fx.arith.constant(0, type=fx.T.i32())

        g = buffer_load(b2g, pmb0, vec_width=1, dtype=fx.T.i32())
        offs_pc_g = buffer_load(opr, g, vec_width=1, dtype=fx.T.i32())
        in_off_g = buffer_load(ofr, g, vec_width=1, dtype=fx.T.i32())
        len_g = buffer_load(lr, g, vec_width=1, dtype=fx.T.i32())
        m_local0 = pmb0 * fx.Int32(_BLK) - fx.arith.ArithValue(offs_pc_g)
        scol_base = ftile * fx.Int32(SCPT)          # first rowwise scale col of this F-tile
        scol_local = tid // fx.Int32(_BLK)          # this column's scol within the tile

        def row_of(r):
            m_local = fx.arith.ArithValue(m_local0) + r
            real = m_local < fx.arith.ArithValue(len_g)
            m_eff = fx.arith.select(real, m_local, fx.Int32(0))
            return real, fx.arith.ArithValue(in_off_g) + fx.arith.ArithValue(m_eff)

        # ── stage the rowwise E8M0 scale slab (MROWS x SCPT) in LDS once (SC_N entries, MB/thread) ──
        for p in range_constexpr(MB):
            e = tid + fx.Int32(p * BT)
            r = e // fx.Int32(SCPT)
            j = e % fx.Int32(SCPT)
            _, srow = row_of(r)
            se_ld = buffer_load(xsr, srow * fx.Int32(F32) + (scol_base + j), vec_width=1, dtype=fx.T.i8())
            se_ld_i32 = fx.arith.ArithValue(se_ld).extui(fx.T.i32())
            fx.make_view(fx.add_offset(scale_lds.ptr, fx.make_int_tuple(e)), fx.make_layout(1, 1)).store(
                Vec.from_elements([fx.arith._to_raw(se_ld_i32)], fx.Int32))
        fx.gpu.barrier()

        # ── compute: each column's MB blocks -> fp8 words into LDS out-tile + scale to global ──
        for mb in range_constexpr(MB):
            vals = []
            for i in range_constexpr(_BLK):
                r = mb * _BLK + i
                real, row = row_of(fx.Int32(r))
                qb = buffer_load(xqr, row * fx.Int32(F) + f, vec_width=1, dtype=fx.T.i8())
                qb_i32 = fx.arith.ArithValue(qb).extui(fx.T.i32())
                fq = cvt_f32_fp8(fx.T.f32(), fx.arith._to_raw(qb_i32), 0)
                sv_s = Vec(fx.make_view(fx.add_offset(scale_lds.ptr,
                           fx.make_int_tuple(fx.Int32(r * SCPT) + scol_local)), fx.make_layout(1, 1)).load())
                sc = (fx.arith.ArithValue(fx.Int32(sv_s[0])) << fx.Int32(23)).bitcast(fx.T.f32())
                dv = fx.arith.mulf(fx.arith.ArithValue(fq), sc)
                fv = fx.arith.select(real, dv, fx.Float32(0.0))
                vals.append(fx.arith._to_raw(fv))
            words, biased = _e8m0_quant_pack(vals, round_add, target_pow2, lo, hi, cvt, zero_i32)
            lds_base = tid * fx.Int32(RPT) + fx.Int32(mb * blk_i32)
            for w in range_constexpr(blk_i32):
                fx.make_view(fx.add_offset(outq_lds.ptr, fx.make_int_tuple(lds_base + fx.Int32(w))),
                             fx.make_layout(1, 1)).store(Vec.from_elements([words[w]], fx.Int32))
            buffer_store(fx.arith.ArithValue(biased).trunci(fx.T.i8()), sr,
                         f * fx.arith.ArithValue(npblk) + (pmb0 + fx.Int32(mb)))
        fx.gpu.barrier()

        # ── coalesced write: 32 consecutive lanes -> 32 consecutive M-i32 of one output row ──
        for it in range_constexpr(n_wr):
            k = tid + fx.Int32(it * BT)
            col = k // fx.Int32(RPT)
            j = k % fx.Int32(RPT)
            sv = Vec(fx.make_view(fx.add_offset(outq_lds.ptr, fx.make_int_tuple(k)),
                                  fx.make_layout(1, 1)).load())
            gi = (f_base + col) * fx.arith.ArithValue(mpad_i32) + pmb0 * fx.Int32(blk_i32) + j
            buffer_store(fx.Int32(sv[0]), qr, gi)

    @flyc.jit
    def launch(XQ, XS, Q, S, BLK2GRP, LENS, OFFS, OFFS_PC, mpad_i32, npblk, n_mwg,
               stream: fx.Stream = fx.Stream(None)):
        kern(XQ, XS, Q, S, BLK2GRP, LENS, OFFS, OFFS_PC, mpad_i32, npblk).launch(
            grid=(n_mwg * n_ftile, 1, 1), block=(BT, 1, 1), stream=stream)

    return launch


def colwise_requant_mxfp8_grouped_fp8in_flydsl(
    q_in: torch.Tensor, s_in: torch.Tensor, out_dtype: torch.dtype,
    group_lens: torch.Tensor = None, group_offs: torch.Tensor = None,
    meta: dict = None, BT: int = 256,
):
    """Grouped fp8-in colwise (along-M) MXFP8 transpose-requant, per-group M padded to 128.

    Drop-in replacement for ``colwise_quant_mxfp8_grouped_flydsl`` when the operand is already the
    L2-dgrad rowwise-fp8 pool (fusing the dequant->requant round-trip).

    Args:
        q_in: rowwise-fp8 (E4M3) ``[total_M, F]`` (the dispatched-dy pool).
        s_in: raw E8M0 rowwise scale, uint8 ``[total_M, F//32]``.
        out_dtype: colwise output fp8 encoding (``float8_e5m2`` default dW2 / ``float8_e4m3fn``).
        group_lens/group_offs: int ``[G]`` / ``[G+1]`` (unpadded) -- used if ``meta`` is None.
        meta: precomputed ``colwise_grouped_meta`` (share across both dW2 operands).

    Returns ``(q, s, lens_pc, offs_pc)`` (identical layout to ``colwise_quant_mxfp8_grouped_flydsl``):
        q: fp8 ``[F, total_M_pad]``   s: uint8 ``[F, total_M_pad//32]``.
    """
    assert q_in.dim() == 2 and s_in.dim() == 2, "q_in [total_M, F], s_in [total_M, F//32]"
    M, F = q_in.shape
    assert s_in.shape[1] == F // _BLK, f"s_in cols {s_in.shape[1]} != F//32 {F // _BLK}"
    is_e5m2_out = out_dtype == torch.float8_e5m2
    if meta is None:
        meta = colwise_grouped_meta(group_lens, group_offs, pool_rows=M)
    total_M_pad, n_pblk = meta["total_M_pad"], meta["n_pblk"]
    q = torch.empty((F, total_M_pad), dtype=out_dtype, device=q_in.device)
    s = torch.empty((F, n_pblk), dtype=torch.uint8, device=q_in.device)
    while F % BT != 0:
        BT //= 2
    # MB 32-M blocks per workgroup (contiguous in the transposed output); n_pblk is a multiple of 4
    # (per-group M padded to 128), so MB|n_pblk and each WG's blocks stay within one group.
    MB = 4 if n_pblk % 4 == 0 else (2 if n_pblk % 2 == 0 else 1)
    n_mwg = n_pblk // MB
    xq = q_in.view(torch.uint8)
    xs = s_in.contiguous().view(torch.uint8)
    _compile_colwise_requant_grouped_fp8in(F, is_e5m2_out, BT, MB)(
        xq, xs, q, s, meta["blk2grp"], meta["lens"], meta["offs32"], meta["offs_pc"],
        total_M_pad // 4, n_pblk, n_mwg,
    )
    return q, s, meta["lens_pc"], meta["offs_pc"]


# ── dW2 dual-launch: fp8-in pool requant (a) + bf16 act colwise-quant (b) ────────────────────
# Independent grids (requant ``n_mwg*(H/BT)`` heavy + quant ``n_pblk*(I/BT)`` light LDS=0), same
# ``meta``, back-to-back on one stream.  Byte-exact to the two shipped kernels; avoids the
# single-WG serial-tail fusion that held 36KB LDS through the act phase.


@functools.lru_cache(maxsize=64)
def _compile_dw2_colwise_dual_launch(
    F_a: int, F_b: int, is_e5m2_out: bool, BT_a: int, BT_b: int, MB: int,
):
    """Return a launch closure that dispatches requant then quant on the same stream."""
    launch_a = _compile_colwise_requant_grouped_fp8in(F_a, is_e5m2_out, BT_a, MB)
    launch_b = _compile_colwise_quant_grouped(F_b, is_e5m2_out, BT_b)

    def launch(
        XQ, XS, QA, SA, XB, QB, SB,
        BLK2GRP, LENS, OFFS, OFFS_PC, mpad_i32, npblk, n_mwg,
        stream: fx.Stream = fx.Stream(None),
    ):
        launch_a(
            XQ, XS, QA, SA, BLK2GRP, LENS, OFFS, OFFS_PC, mpad_i32, npblk, n_mwg, stream=stream,
        )
        launch_b(
            XB, QB, SB, BLK2GRP, LENS, OFFS, OFFS_PC, mpad_i32, npblk, npblk, stream=stream,
        )

    return launch


def colwise_requant_fp8in_and_quant_bf16_grouped_flydsl(
    q_in: torch.Tensor, s_in: torch.Tensor, x_bf16: torch.Tensor, out_dtype: torch.dtype,
    group_lens: torch.Tensor = None, group_offs: torch.Tensor = None,
    meta: dict = None, BT: int = 256,
):
    """dW2 colwise operands: pool rowwise-fp8 requant (a) + bf16 act quant (b), dual-launch.

    Uses two independent kernels (heavy requant grid + light quant grid) on one stream with
    shared ``meta``.  Returns ``(a_t, a_ts, b_t, b_ts, lens_pc, offs_pc)`` -- byte-exact to
    ``colwise_requant_mxfp8_grouped_fp8in_flydsl`` +
    ``colwise_quant_mxfp8_grouped_flydsl``."""
    assert q_in.dim() == 2 and s_in.dim() == 2 and x_bf16.dim() == 2
    assert x_bf16.dtype == torch.bfloat16
    M, F_a = q_in.shape
    M_b, F_b = x_bf16.shape
    assert M == M_b, f"pool rows {M} != act rows {M_b}"
    assert s_in.shape[1] == F_a // _BLK
    is_e5m2_out = out_dtype == torch.float8_e5m2
    if meta is None:
        meta = colwise_grouped_meta(group_lens, group_offs, pool_rows=M)
    total_M_pad, n_pblk = meta["total_M_pad"], meta["n_pblk"]
    q_a = torch.empty((F_a, total_M_pad), dtype=out_dtype, device=q_in.device)
    s_a = torch.empty((F_a, n_pblk), dtype=torch.uint8, device=q_in.device)
    q_b = torch.empty((F_b, total_M_pad), dtype=out_dtype, device=q_in.device)
    s_b = torch.empty((F_b, n_pblk), dtype=torch.uint8, device=q_in.device)
    BT_a = BT
    while F_a % BT_a != 0:
        BT_a //= 2
    BT_b = BT
    while F_b % BT_b != 0:
        BT_b //= 2
    MB = 4 if n_pblk % 4 == 0 else (2 if n_pblk % 2 == 0 else 1)
    n_mwg = n_pblk // MB
    x_bf16 = x_bf16.contiguous()
    _compile_dw2_colwise_dual_launch(F_a, F_b, is_e5m2_out, BT_a, BT_b, MB)(
        q_in.view(torch.uint8), s_in.contiguous().view(torch.uint8), q_a, s_a,
        x_bf16, q_b, s_b,
        meta["blk2grp"], meta["lens"], meta["offs32"], meta["offs_pc"],
        total_M_pad // 4, n_pblk, n_mwg,
    )
    return q_a, s_a, q_b, s_b, meta["lens_pc"], meta["offs_pc"]


# ── FUSED rowwise + colwise dual-quant (one read of grad_l1 -> both operands) ─────────────────
# grad_l1 [P, F] bf16 is needed BOTH rowwise-preshuffled (E4M3, the L1 fc1-dgrad) and colwise-grouped
# (E5M2, dW1 wgrad). Reading it twice (the two shipped kernels) costs an extra HBM read; this fuses
# both from ONE read via a 32xBT bf16 tile staged in LDS -> colwise reads down columns, rowwise reads
# across each 32-feature block. Rowwise ``q`` matches ``quantize_rowwise_mxfp8_flydsl``;
# ``a_sp`` pack=4 fused in-kernel: one workgroup/pool M-block loops all F-tiles then packs (single launch).


@functools.lru_cache(maxsize=64)
def _compile_rowcol_dual_pack_grouped(F: int, BT: int = 256):
    """Pack=4 a_sp preshuffle for grouped rowcol dual-quant (reads s_raw written by quant kernel)."""
    n_blk = F // _BLK
    K128p = ceildiv(F // 128, 4)
    n_out_pack = K128p * 4
    n_row_pack_slots = _BLK * n_out_pack
    n_pack_rounds = (n_row_pack_slots + BT - 1) // BT

    @flyc.kernel(known_block_size=[BT, 1, 1])
    def pack_kern(
        ASP: fx.Tensor, SRAW: fx.Tensor, PMB_META: fx.Tensor,
    ):
        tid = fx.thread_idx.x
        bid = fx.block_idx.x
        pmb = bid // fx.Int32(n_pack_rounds)
        pi = bid % fx.Int32(n_pack_rounds)

        aspr = create_buffer_resource(ASP, max_size=True)
        srr = create_buffer_resource(SRAW, max_size=True)
        pmbr = create_buffer_resource(PMB_META, max_size=True)
        mb = pmb * fx.Int32(4)
        in_off_g = buffer_load(pmbr, mb + fx.Int32(1), vec_width=1, dtype=fx.T.i32())
        len_g = buffer_load(pmbr, mb + fx.Int32(2), vec_width=1, dtype=fx.T.i32())
        m_local0 = fx.arith.ArithValue(buffer_load(pmbr, mb + fx.Int32(3), vec_width=1, dtype=fx.T.i32()))

        flat = tid + pi * BT
        if flat < fx.Int32(n_row_pack_slots):
            row_i = flat // fx.Int32(n_out_pack)
            pidx = flat % fx.Int32(n_out_pack)
            m_local = fx.arith.ArithValue(m_local0) + fx.arith.ArithValue(row_i)
            if m_local < fx.arith.ArithValue(len_g):
                global_row = fx.arith.ArithValue(in_off_g) + m_local
                kkp = pidx // fx.Int32(4)
                g_out = pidx % fx.Int32(4)
                packed = fx.Int32(0)
                for bb in range_constexpr(4):
                    raw_b = (kkp * fx.Int32(4) + fx.Int32(bb)) * fx.Int32(4) + g_out
                    scale_byte = buffer_load(
                        srr, global_row * fx.Int32(n_blk) + raw_b, vec_width=1, dtype=fx.T.i8())
                    b_i32 = fx.arith.extui(fx.T.i32(), scale_byte)
                    packed = packed | ((b_i32 & fx.Int32(0xFF)) << (fx.Int32(bb) * fx.Int32(8)))
                buffer_store(packed, aspr, _preshuffle_a_pack4_idx(global_row, kkp, g_out, K128p))

    @flyc.jit
    def launch_pack(ASP, SRAW, PMB_META, n_pblk, stream: fx.Stream = fx.Stream(None)):
        pack_kern(ASP, SRAW, PMB_META).launch(
            grid=(n_pblk * n_pack_rounds, 1, 1), block=(BT, 1, 1), stream=stream)

    return launch_pack
