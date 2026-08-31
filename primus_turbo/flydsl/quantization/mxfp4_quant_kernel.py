###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2025 FlyDSL Project Contributors
# Adapted from FlyDSL (https://github.com/ROCm/FlyDSL); see LICENSE-APACHE for the Apache-2.0 terms.
#
# See LICENSE for license information.
###############################################################################
"""Pure-FlyDSL MXFP4 activation quant kernels (no ``import torch`` at module top).

Bit-exact replacement for the C++ ``quantize_mxfp4_dual`` for the scored
``preshuffle=False`` recipes. This module is data-quant only; it never fuses
quant into the GEMM.

Numerics reproduce ``csrc/kernels/quantization/quantization_mxfp4.cu`` exactly:
  * e8m0 scale via ``compute_tile_scale`` (all-int32 recipe),
  * native ``rocdl.cvt_scalef32_pk_fp4_f32`` pair-form cvt (dst_sel chaining),
  * RHT = fixed H16 = H4 (within a 4-block) then H4 (across the 4 blocks) done
    fully IN-REGISTER (each thread owns a whole 32-elem microblock = 2 H16
    groups), bit-identical to the C++ distributed ds_swizzle version.
"""

import os

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith, buffer_ops, math, range_constexpr, rocdl
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec
from flydsl.expr.utils.arith import _to_raw as _raw

from primus_turbo.flydsl.utils.gemm_helper import xcd_remap_pid

_OOB = 0x7FFFFFFF  # word offset past any SRD -> buffer_load returns 0 / buffer_store dropped

# ---- fused NT B-scale preshuffle (packed layout written in place) -------------------
# The E8M0 this kernel emits for a weight operand is consumed by EXACTLY one thing: the
# grouped NT GEMM (ROW_SC -> fwd's b_scale, COL_SC -> dgrad's b_scale), and only after the
# merged preshuffle's B segment has repacked it into the lane-contiguous layout
# `ScaleS2RPacked` reads. That segment is a pure ~33 MB read + 36 MB write round trip of a
# tensor this kernel has just written -- so write the packed layout directly and the
# segment's 1107 workgroups (x2 launches per operator) are DELETED, exactly the way C23's
# `df` deleted the A segment.
#
# Packed byte address of the E8M0 for (operand row `row`, 32-element block `c`), per expert:
#     off = ((wi*KK + kk)*256 + g*64 + r*4 + last) * 4 + t
#     grp = row//64, t = (row%64)//16, r = row%16
#     rr  = (grp//2)%2, wi = 2*(grp//4) + grp%2          # inverse of _mxfp4_grp_from(mode=1)
#     kk  = c//8,  sb = (c//4)%2,  g = c%4,  last = 2*rr + sb
# (forward map of the preshuffle's B segment; verified byte-identical against the shipped
# kernel over both shipped geometries before any timing was taken.)
# FLYDSL_NT_BSFUSE=0 reverts BOTH this and the GEMM's b_pre_grid=0, which must agree.
_BSFUSE_ENV = os.environ.get("FLYDSL_NT_BSFUSE", "0") == "1"


def bsc_pack_geom(rows, kbytes):
    """(N_SCALE, K128, KK, nbytes) of the packed NT B-scale layout for one expert.

    ``rows`` = the operand's free-dim rows, ``kbytes`` = E8M0 bytes per row (= K_op/32).
    Mirrors the GEMM's own N_SCALE = ceil(rows,256)*256 and K128 = ceil(K_op,256)*256/128.
    """
    rows = int(rows)
    kbytes = int(kbytes)
    K256 = ((kbytes * 32 + 255) // 256) * 256
    K128 = K256 // 128
    N_SCALE = ((rows + 255) // 256) * 256
    return N_SCALE, K128, K128 // 2, N_SCALE * K128 * 4


def bsc_fused(rows, kbytes):
    """True when the B-scale is emitted in the packed NT layout. The GEMM applies the
    SAME predicate (plus a numel check) on its own (N, k_real/32); the two must agree."""
    if not _BSFUSE_ENV:
        return False
    rows, kbytes = int(rows), int(kbytes)
    if rows <= 0 or kbytes <= 0 or rows % 64 or kbytes % 4:
        return False
    _NS, _K128, _KK, nbytes = bsc_pack_geom(rows, kbytes)
    # the packed tensor must be a DIFFERENT size from the canonical one, so the consumer
    # can tell the two apart by numel with no side channel.
    return nbytes != rows * kbytes


def _bsc_off(row, c, KK):
    """Device-side packed BYTE offset (within one expert) for E8M0 (row, block c)."""
    I32 = fx.Int32
    grp = row >> I32(6)
    t = (row >> I32(4)) & I32(3)
    r = row & I32(15)
    rr = (grp >> I32(1)) & I32(1)
    wi = ((grp >> I32(2)) << I32(1)) | (grp & I32(1))
    kk = c >> I32(3)
    sb = (c >> I32(2)) & I32(1)
    g = c & I32(3)
    dw = (wi * I32(KK) + kk) * I32(256) + g * I32(64) + r * I32(4) + I32(2) * rr + sb
    return (dw << I32(2)) + t


BLK = 256
MB = 32  # MXFP4 micro-block size (elements per e8m0 scale)


def _abs_i32(fbits):
    return fbits & 0x7FFFFFFF


def _imax(a, b):
    return arith.select(a < b, b, a)


# ---- Stochastic rounding (gradient SR) ----------------------------------------
# SR can't be bit-exact vs the C++ dual (different thread<->element tiling), and it
# is random by design; the goal is an unbiased, decorrelated rounding. Each launch
# gets a distinct seed (host counter, mirroring the C++ atomic counter) and each
# micro-block a grid-unique id (col salted apart from row).
_SR_COL_SALT = 0x5BD1E995  # decorrelate the col-wise operand from the row-wise one
_SR_COUNTER = [0]


def _next_sr_seed():
    """Per-launch u32 seed: distinct each launch, reproducible within a process for
    a fixed call order (matches the C++ ``global_sr_counter`` semantics)."""
    s = _SR_COUNTER[0] & 0xFFFFFFFF
    _SR_COUNTER[0] = (_SR_COUNTER[0] + 1) & 0xFFFFFFFF
    return s


def _sr_hash(seed):
    """Integer avalanche hash for SR seeds (same shape as the C++ ``sr_hash``).
    Uses ``>>`` (the fx numeric shift accepts a python int; ``shrui`` does not);
    arithmetic vs logical shift is irrelevant here - SR only needs the resulting
    seeds well-distributed and decorrelated across micro-blocks/pairs."""
    seed = (seed ^ 61) ^ (seed >> 16)
    seed = seed * 9
    seed = seed ^ (seed >> 4)
    seed = seed * 0x27D4EB2D
    seed = seed ^ (seed >> 15)
    return seed


def _compute_scale_native(amax_bits, exp_up=0):
    """e8m0 scale, all-int32 (matches compute_tile_scale). Returns
    (scale_native_f32bits_i32, scale_e8m0_biased_i32).

    ``exp_up`` supports the folded-RHT-scale path: the caller's values (and hence
    ``amax_bits``) are 2**exp_up times the true ones because the trailing ``*0.25``
    of each ``_rht16`` was dropped. Subtracting ``exp_up`` from the extracted
    exponent recovers the *identical* e8m0 byte, and the native scale handed to the
    cvt is scaled up by the same power of two, so ``v/scale`` -- and therefore every
    fp4 nibble -- is bit-identical. Both scalings are exact powers of two.
    ``amax_bits <= 0x7fffffff`` bounds the extracted field at 256, so ``biased``
    tops out at 254-exp_up and ``biased+exp_up`` never overflows the exponent."""
    val_to_add = 1 << 21  # 1 << (23 - 1 - 1)
    hp_exp_mask = 0x1FF  # (1 << 9) - 1
    extracted = ((amax_bits + val_to_add) >> 23) & hp_exp_mask
    extracted = extracted - 127 - 2 - exp_up  # - hp_exp_bias - FP4_TARGET_MAX_POW2
    extracted = _imax(extracted, -127)
    extracted = arith.select(extracted < 128, extracted, 128)
    biased = extracted + 127  # 0..255
    native_bits = (biased + exp_up) << 23  # 2^(biased+exp_up-127) as f32 bits
    return native_bits, biased


def _h4(v0, v1, v2, v3):
    """One H4 butterfly, same float order as rht16_inplace stage-1 / cross-lane."""
    a0 = v0 + v1
    a1 = v0 - v1
    a2 = v2 + v3
    a3 = v2 - v3
    return a0 + a2, a1 + a3, a0 - a2, a1 - a3


def _rht16(v, post_scale=True):
    """In-register H16 = H4(local) then H4(across 4 blocks), * 0.25.
    ``v`` is a list of 16 f32 Values, element index e = 4*block + local.
    ``post_scale=False`` drops the trailing ``*0.25`` (16 ``v_mul_f32`` per H16,
    ~13% of the grouped quant's VALU); the caller must then pass ``exp_up=2`` to
    ``_compute_scale_native``, which recovers bit-identical output."""
    o = [None] * 16
    for b in range_constexpr(4):
        y0, y1, y2, y3 = _h4(v[4 * b + 0], v[4 * b + 1], v[4 * b + 2], v[4 * b + 3])
        o[4 * b + 0] = y0
        o[4 * b + 1] = y1
        o[4 * b + 2] = y2
        o[4 * b + 3] = y3
    r = [None] * 16
    for lc in range_constexpr(4):
        y0, y1, y2, y3 = _h4(o[0 * 4 + lc], o[1 * 4 + lc], o[2 * 4 + lc], o[3 * 4 + lc])
        if post_scale:
            y0, y1, y2, y3 = y0 * 0.25, y1 * 0.25, y2 * 0.25, y3 * 0.25
        r[0 * 4 + lc] = y0
        r[1 * 4 + lc] = y1
        r[2 * 4 + lc] = y2
        r[3 * 4 + lc] = y3
    return r


def _cvt_microblock_to_fp4(vf, scale_native_f32, seed=None):
    """32 f32 Values -> 4 i32 words (8 fp4 each). Pair-form cvt, dst_sel chaining.
    ``seed`` (an i32 Value) switches to the stochastic-rounding converter; one
    per-thread seed drives all pairs of the micro-block (mirrors the C++ path).
    Same packing as the plain path (SR op is its exact analog)."""
    words = []
    for wi in range_constexpr(4):
        acc = fx.Int32(0)
        for pair in range_constexpr(4):
            i = wi * 8 + pair * 2
            if seed is None:
                acc = rocdl.cvt_scalef32_pk_fp4_f32(T.i32, acc, vf[i], vf[i + 1], scale_native_f32, pair)
            else:
                # SR op's dst_sel/oldVdst chaining misbehaves (bytes 1-2 corrupt);
                # mirror the C++ path exactly: one rng for all pairs (per-thread seed),
                # each pair -> byte 0 with old=0, then OR-shift into place.
                src = _raw(
                    Vec.from_elements([fx.Float32(_raw(vf[i])), fx.Float32(_raw(vf[i + 1]))], fx.Float32)
                )
                # The llvm.amdgcn.cvt.scalef32.sr.pk.fp4.f32 intrinsic consumes the SR
                # seed one bit lower than the raw v_cvt asm the C++ path uses, which
                # halves the round-up probability. Shift left by 1 to realign; drops
                # only one LSB of a full-entropy hash so the distribution is unaffected.
                b = rocdl.cvt_scalef32_sr_pk_fp4_f32(
                    T.i32, _raw(fx.Int32(0)), src, _raw(seed << 1), scale_native_f32, 0
                )
                acc = acc | ((fx.Int32(b) & 0xFF) << (pair * 8))
        words.append(acc)
    return words


def _store_words_vec4(rsrc, off, words):
    """One b128 (vec4) buffer_store of 4 contiguous i32 fp4-packed words, instead
    of 4 scalar b32 stores (4x fewer store instructions, same bytes/values)."""
    buffer_ops.buffer_store(Vec.from_elements(list(words), fx.Int32), rsrc, off)


def _lds_store_vec4(lds_ptr, off, vec):
    fx.make_view(fx.add_offset(lds_ptr, fx.make_int_tuple(off)), fx.make_layout(4, 1)).store(vec)


def _lds_load1(lds_ptr, off):
    return fx.make_view(fx.add_offset(lds_ptr, fx.make_int_tuple(off)), fx.make_layout(1, 1)).load()[0]


def _lds_load_vec4(lds_ptr, off):
    return fx.make_view(fx.add_offset(lds_ptr, fx.make_int_tuple(off)), fx.make_layout(4, 1)).load()


def _lds_store1(lds_ptr, off, val):
    fx.make_view(fx.add_offset(lds_ptr, fx.make_int_tuple(off)), fx.make_layout(1, 1)).store(
        Vec.from_elements([val], fx.Int32)
    )


def _rht16_pair(v, post_scale=True):
    """The two H16 groups of one 32-element microblock, done together as
    ``vector<2xf32>`` so gfx950's packed-FP32 pipe issues both in one instruction.

    Group 0 (elements 0..15) is lane 0 and group 1 (elements 16..31) is lane 1 of
    every 2-vector, so *the two lanes never interact* -- every op is exactly the
    scalar op ``_rht16`` would have emitted on that element, in the same order, on
    the same operands. Bit-exactness is therefore structural, not an FP-reassoc
    argument. 128 scalar adds per microblock become 64 ``v_pk_add_f32``."""
    p = [Vec.from_elements([v[i], v[i + 16]], fx.Float32) for i in range_constexpr(16)]
    o = [None] * 16
    for b in range_constexpr(4):
        y0, y1, y2, y3 = _h4(p[4 * b + 0], p[4 * b + 1], p[4 * b + 2], p[4 * b + 3])
        o[4 * b + 0] = y0
        o[4 * b + 1] = y1
        o[4 * b + 2] = y2
        o[4 * b + 3] = y3
    r = [None] * 16
    for lc in range_constexpr(4):
        y0, y1, y2, y3 = _h4(o[0 * 4 + lc], o[1 * 4 + lc], o[2 * 4 + lc], o[3 * 4 + lc])
        if post_scale:
            y0, y1, y2, y3 = y0 * 0.25, y1 * 0.25, y2 * 0.25, y3 * 0.25
        r[0 * 4 + lc] = y0
        r[1 * 4 + lc] = y1
        r[2 * 4 + lc] = y2
        r[3 * 4 + lc] = y3
    return [r[i][0] for i in range_constexpr(16)] + [r[i][1] for i in range_constexpr(16)]


def _microblock_vf(vbits, use_rht, fold_scale=False):
    """32 f32-bit i32 values -> list of 32 f32 Values (post-RHT if enabled).
    ``fold_scale`` drops the RHT's trailing ``*0.25``; see ``_rht16``. It returns
    ``vf_exp_up(use_rht, fold_scale)`` extra binary exponents that the caller must
    hand to ``_compute_scale_native``."""
    vf = [Vec.from_elements([b], fx.Int32).bitcast(fx.Float32)[0] for b in vbits]
    if use_rht:
        if fold_scale:  # packed path: both H16 in <2 x float>, no trailing *0.25
            vf = _rht16_pair(vf, post_scale=False)
        else:
            vf = _rht16(vf[0:16]) + _rht16(vf[16:32])
    return vf


def vf_exp_up(use_rht, fold_scale=True):
    """The ``exp_up`` that pairs with ``_microblock_vf(..., fold_scale=...)``."""
    return 2 if (use_rht and fold_scale) else 0


def _microblock_amax(vf):
    """int-max over abs bits of 32 f32 Values (matches C++ fabs-reduce, bit-exact)."""
    amax = fx.Int32(0)
    for i in range_constexpr(32):
        b = Vec.from_elements([vf[i]], fx.Float32).bitcast(fx.Int32)[0]
        amax = _imax(amax, _abs_i32(b))
    return amax


def _microblock_amax_f(vf):
    """Same value as ``_microblock_amax``, computed in the float pipe so the abs
    becomes a free VOP3 source modifier instead of 32 ``v_and_b32``.

    For finite inputs the ordering of |x| by float compare and by the AND-ed bit
    pattern is identical, and ``max`` returns one of its operands, so the returned
    bit pattern is the same. The only divergences are NaN (float max quiets it
    away; the int path would return a NaN pattern -> exponent 255) and a
    denormal-flushing max (returns +0 instead of the denormal pattern) -- and any
    amax below 2**-126 clamps to the same e8m0 byte 0 either way, so the flush is
    invisible in the output."""
    cur = math.absf(vf[0])
    for i in range_constexpr(1, 32):
        cur = cur.maximumf(math.absf(vf[i]))
    return Vec.from_elements([cur], fx.Float32).bitcast(fx.Int32)[0]


def _finish_microblock(vbits, use_rht, seed=None):
    """32 f32-bit i32 values -> (4 fp4 i32 words, scale_e8m0 i8-ready i32).
    ``seed`` (i32 Value) enables stochastic rounding in the final cvt (amax/scale
    stay deterministic)."""
    vf = _microblock_vf(vbits, use_rht, fold_scale=True)
    amax = _microblock_amax_f(vf)
    native_bits, biased = _compute_scale_native(amax, exp_up=vf_exp_up(use_rht))
    words = _cvt_microblock_to_fp4(vf, arith.bitcast(T.f32, native_bits), seed)
    return words, biased


# ---- fused-dual tile geometry (shared by the 2D and batched-3D kernels) ----
# Tile rows/cols for the fused dual. _TR drives the COL_OUT write granule
# (_TR/8 i32 per feature's R-run): at _TR=64 that is 32 bytes, i.e. every byte of
# the 265 MB colwise weight output leaves in a sub-cacheline burst. _TR must divide
# both shipped N (5760 and 2880) and _TR*_TC/32 must divide BLK=256.
# _TC also sets the K tail: ceil(2880/_TC)*_TC.
_TR = 96  # tile rows (R dim)
_TC = 256  # tile cols (C dim)
_TCW = _TC // 2  # 128 i32 words per tile row
_NW = _TR * _TCW  # 8192 i32 words in LDS
_RMB = _TR // 32  # col m-microblocks per tile
_NLOAD = (_NW + BLK * 4 - 1) // (BLK * 4)  # vec4 loads per thread
_RROWTASK = (_TR * (_TC // 32)) // BLK  # row tasks per thread
_RMBC = _TC // 32  # row micro-blocks along C (== 8)
_NSCR = _TR * _RMBC  # LDS amax scratch elems (64x8 = 512 i32 = 2KB)


_TILE_DEFAULT = (_TR, _TC)


def _set_tile_geom(tr, tc):
    """Re-derive every fused-dual tile constant for (tr, tc).

    Set immediately around the trace+compile of ONE kernel and restored afterwards;
    compilation is serialised in-process, and (tr, tc) is part of both compile-cache keys.
    """
    global _TR, _TC, _TCW, _NW, _RMB, _NLOAD, _RROWTASK, _RMBC, _NSCR
    _TR, _TC = tr, tc
    _TCW = _TC // 2
    _NW = _TR * _TCW
    _RMB = _TR // 32
    _NLOAD = (_NW + BLK * 4 - 1) // (BLK * 4)
    _RROWTASK = (_TR * (_TC // 32)) // BLK
    _RMBC = _TC // 32
    _NSCR = _TR * _RMBC


def _pick_tile_geom(N, K):
    """Per-shape tile geometry for the batched weight quant.

    Three things are decided by (_TR, _TC) and the two shipped weight shapes want different
    answers:
      * COL_OUT write granule = _TR/8 i32. At _TR=96 that is 48 bytes -- SUB-CACHELINE, and
        the profile measures 1.28x write amplification (761 MB written for 596 MB of unique
        output) on a kernel that is bandwidth bound at 64% of HBM peak. _TR=128 gives
        exactly 64 bytes.
      * legality: _TR must divide R (the weight's N), _TR*_TC/32 must divide BLK=256, and
        _TC must BE 256 (the col phase is one thread per column).
    N=5760 admits _TR=128 and so gets a full-cacheline COL_OUT granule; N=2880 does not
    (2880 % 128 = 64) and keeps _TR=96. Hence per-shape rather than one global.
    """
    # _TC is pinned to BLK: the col cast phase maps exactly one thread per tile column
    # (c_col = tid), so _TC != 256 silently corrupts the colwise output -- observed as
    # dgrad SNR 0.28 dB while fwd and wgrad stayed clean. Only _TR is free.
    for tr, tc in ((128, 256), (96, 256)):
        if N % tr == 0 and (tr * (tc // 32)) % BLK == 0 and K % 32 == 0:
            return tr, tc
    return _TILE_DEFAULT


def _make_dual_struct(need_scr):
    if need_scr:

        @fx.struct
        class _DualSS:
            buf: fx.Array[fx.Int32, _NW, 16]
            scr: fx.Array[fx.Int32, _NSCR, 16]

    else:

        @fx.struct
        class _DualSS:
            buf: fx.Array[fx.Int32, _NW, 16]

    return _DualSS


def _emit_dual_body(
    row_rht,
    col_rht,
    row_2d,
    col_2d,
    lds,
    tid,
    X,
    ROW_OUT,
    ROW_SC,
    COL_OUT,
    COL_SC,
    R,
    C,
    bid,
    gx=0,
    gro=0,
    grsc=0,
    gco=0,
    gcsc=0,
    gmul=1,
    padded=False,
    ncblk=None,
    CP=None,
    RP=None,
    col_locality=False,
    batched=False,
    row_sr=False,
    col_sr=False,
    sr_seed=None,
    sr_gbid=None,
    rpk=None,
    cpk=None,
):
    """Emit one fused-dual tile (rowwise + colwise-transpose mxfp4 cast) for block
    ``bid``. ``row_2d``/``col_2d`` pick the C++ ``USE_2D_BLOCK`` amax geometry; the
    batched-3D kernel passes per-expert base offsets ``gx/gro/grsc/gco/gcsc`` and
    ``gmul=G`` to widen the SRDs over the whole 3D tensor (R,C stay per-expert).
    ``padded`` (non-256 K / non-128 N): X is the real [R,C] but outputs use K_pad=CP /
    N_pad=RP cols (caller zero-inits so pad stays 0, matching HIP); loads past real C
    mask to 0 and writes past K_pad / real-C rows go to _OOB so the store drops them."""
    if ncblk is None:
        ncblk = C // _TC
    cpad = CP if padded else C  # row-out column extent (K_pad)
    rpad = RP if padded else R  # col-out column extent (N_pad)
    # Block order = which output's partial stores L2 can combine. col_locality (C>R):
    # row-tile-fastest so blocks writing the same col-out rows run back-to-back and L2
    # merges the scattered transpose stores; else col-tile-fastest keeps row-out coalesced.
    if col_locality:
        nrblk = R // _TR
        cblk = bid // nrblk
        rblk = bid % nrblk
    else:
        rblk = bid // ncblk
        cblk = bid % ncblk
    r0 = rblk * _TR
    c0w = cblk * _TCW  # i32-word base along C

    # Re-base each SRD in int64 with per-tile/per-expert num_records: a whole-tensor SRD's
    # num_records (full bytes) overflows the 32-bit field past 4GB (high rows/experts OOB) and
    # the per-row voffset overflows int32. 2D folds this tile's row (r0)/col (cblk*_TC) base;
    # batched-3D folds the per-expert base (small experts keep r0/c0 in the offsets). _row0/
    # _col0 drop the folded base from the additive offsets below.
    _fold = not batched
    _row0 = fx.Int32(0) if _fold else r0
    _col0 = fx.Int32(0) if _fold else cblk * _TC

    def _srd(t, elem_off, elem_bytes, nrec_bytes):
        base = arith.index_cast(T.i64, buffer_ops.extract_base_index(t))
        boff = arith.index_cast(T.i64, arith.index_cast(T.index, elem_off) * arith.index(elem_bytes))
        raw = arith._to_raw(base + boff)
        r = rocdl.readfirstlane(res=raw.type, src=raw)  # pin the SRD base to an SGPR
        base_v = r.result if hasattr(r, "result") else r
        nr = arith.minui(arith.index_cast(T.index, nrec_bytes), arith.index(0x7FFFFFFF))
        return buffer_ops.create_buffer_resource_from_addr(base_v, num_records_bytes=nr)

    if batched:
        rsrc = _srd(X, gx, 4, R * (C >> 1) * 4)
        orsrc = _srd(ROW_OUT, gro, 4, R * (cpad >> 3) * 4)
        rscrsrc = _srd(ROW_SC, grsc, 1, fx.Int32(rpk[3]) if rpk else R * (cpad >> 5))
        corsrc = _srd(COL_OUT, gco, 4, C * (rpad >> 3) * 4)
        cscrsrc = _srd(COL_SC, gcsc, 1, fx.Int32(cpk[3]) if cpk else C * (rpad >> 5))
        gx = gro = grsc = gco = gcsc = 0  # expert bases folded into the SRDs above
    else:
        r0i = arith.index_cast(T.index, r0)
        c0i = arith.index_cast(T.index, cblk * _TC)
        rsrc = _srd(X, r0i * arith.index_cast(T.index, C >> 1), 4, _TR * (C >> 1) * 4)
        orsrc = _srd(ROW_OUT, r0i * arith.index_cast(T.index, cpad >> 3), 4, _TR * (cpad >> 3) * 4)
        rscrsrc = _srd(ROW_SC, r0i * arith.index_cast(T.index, cpad >> 5), 1, _TR * (cpad >> 5))
        corsrc = _srd(COL_OUT, c0i * arith.index_cast(T.index, rpad >> 3), 4, _TC * (rpad >> 3) * 4)
        cscrsrc = _srd(COL_SC, c0i * arith.index_cast(T.index, rpad >> 5), 1, _TC * (rpad >> 5))

    # ---- coalesced tile load -> LDS ----
    for chunk in range_constexpr(_NLOAD):
        tw = chunk * (BLK * 4) + tid * 4
        tr = tw // _TCW
        wc = tw % _TCW
        goff = (_row0 + tr) * (C >> 1) + c0w + wc + gx
        if padded:
            # mask cols past real C -> OOB load returns 0 (rows always valid: R%64==0,
            # tile is 64 rows, rblk covers exactly R/64 tiles).
            goff = arith.select((c0w + wc) < (C >> 1), goff, fx.Int32(_OOB))
        vec = buffer_ops.buffer_load(rsrc, goff, vec_width=4, dtype=T.i32)
        _lds_store_vec4(lds.buf.ptr, tw, vec)
    # DS writes must retire before any thread reads the tile (a bare s_barrier
    # does NOT wait for LDS); fx.barrier() emits the waitcnt + barrier.
    fx.barrier()

    # Per-micro-block SR seeds: grid-unique block id folds the tile + loop task so
    # every micro-block in the launch draws an independent seed (col salted apart
    # from row). Constexpr row_sr/col_sr -> the plain (seed=None) IR when SR is off.
    _gbid = bid if sr_gbid is None else sr_gbid

    def _row_seed(k):
        if not row_sr:
            return None
        return _sr_hash(sr_seed ^ (_gbid * (BLK * _RROWTASK) + (k * BLK + tid)))

    def _col_seed(mmb):
        if not col_sr:
            return None
        return _sr_hash((sr_seed ^ _SR_COL_SALT) ^ (_gbid * (BLK * _RMB) + (mmb * BLK + tid)))

    # ---- ROW phase: 32-elem microblocks along C, contiguous LDS (vec4 reads) ----
    if row_2d:
        # 2D-block amax: the scale spans a whole 32x32 tile = the 32 rows that
        # share one 32-col micro-block. Pass 1: each thread computes its own
        # micro-block amax (RHT'd) and writes it to LDS scratch, keeping the
        # RHT'd vals in registers. Barrier. Pass 2: each thread max-reduces the
        # 32 amax of its tile, then quantizes its held vals with the tile scale.
        vf_hold = []
        meta = []
        for k in range_constexpr(_RROWTASK):
            task = k * BLK + tid
            r_row = task // _RMBC
            cmb = task % _RMBC
            base_w = r_row * _TCW + cmb * 16
            rbits = []
            for q in range_constexpr(4):
                v4 = _lds_load_vec4(lds.buf.ptr, base_w + q * 4)
                for j in range_constexpr(4):
                    word = v4[j]
                    rbits.append(word << 16)
                    rbits.append(word & 0xFFFF0000)
            vf = _microblock_vf(rbits, row_rht, fold_scale=True)
            _lds_store1(lds.scr.ptr, r_row * _RMBC + cmb, _microblock_amax_f(vf))
            vf_hold.append(vf)
            meta.append((r_row, cmb))
        fx.barrier()
        for k in range_constexpr(_RROWTASK):
            r_row, cmb = meta[k]
            vf = vf_hold[k]
            row_base = (r_row // 32) * 32  # tile's first row within the LDS tile
            tile_amax = fx.Int32(0)
            for i in range_constexpr(32):
                tile_amax = _imax(tile_amax, _lds_load1(lds.scr.ptr, (row_base + i) * _RMBC + cmb))
            native_bits, rbiased = _compute_scale_native(tile_amax, exp_up=vf_exp_up(row_rht))
            rwords = _cvt_microblock_to_fp4(vf, arith.bitcast(T.f32, native_bits), _row_seed(k))
            grow = _row0 + r_row
            gcmb = cblk * _RMBC + cmb
            ob = grow * (cpad >> 3) + gcmb * 4 + gro
            sc = _bsc_off(grow, gcmb, rpk[2]) if rpk else grow * (cpad >> 5) + gcmb + grsc
            if padded:
                wok = gcmb < (cpad >> 5)  # rows always valid (R%64==0)
                ob = arith.select(wok, ob, fx.Int32(_OOB))
                sc = arith.select(wok, sc, fx.Int32(_OOB))
            _store_words_vec4(orsrc, ob, rwords)
            buffer_ops.buffer_store(arith.trunci(T.i8, rbiased & 0xFF), rscrsrc, sc)
    else:
        for k in range_constexpr(_RROWTASK):
            task = k * BLK + tid
            r_row = task // (_TC // 32)
            cmb = task % (_TC // 32)
            base_w = r_row * _TCW + cmb * 16
            rbits = []
            for q in range_constexpr(4):
                v4 = _lds_load_vec4(lds.buf.ptr, base_w + q * 4)
                for j in range_constexpr(4):
                    word = v4[j]
                    rbits.append(word << 16)
                    rbits.append(word & 0xFFFF0000)
            rwords, rbiased = _finish_microblock(rbits, row_rht, _row_seed(k))
            grow = _row0 + r_row
            gcmb = cblk * (_TC // 32) + cmb
            ob = grow * (cpad >> 3) + gcmb * 4 + gro
            sc = _bsc_off(grow, gcmb, rpk[2]) if rpk else grow * (cpad >> 5) + gcmb + grsc
            if padded:
                wok = gcmb < (cpad >> 5)  # rows always valid (R%64==0)
                ob = arith.select(wok, ob, fx.Int32(_OOB))
                sc = arith.select(wok, sc, fx.Int32(_OOB))
            _store_words_vec4(orsrc, ob, rwords)
            buffer_ops.buffer_store(arith.trunci(T.i8, rbiased & 0xFF), rscrsrc, sc)

    # ---- COL phase: thread = column, 32-row microblocks (strided LDS reads) ----
    c_col = tid
    half = c_col & 1
    cw = c_col >> 1
    if col_2d:
        # 2D-block amax: the scale spans a whole 32x32 tile = the 32 columns
        # that share one 32-row micro-block. Reuse the LDS amax scratch (freed
        # after the row phase); a barrier before pass 1 protects the WAR on scr.
        fx.barrier()
        cvf_hold = []
        for mmb in range_constexpr(_RMB):
            row0 = mmb * 32
            cbits = []
            for row in range_constexpr(32):
                word = _lds_load1(lds.buf.ptr, (row0 + row) * _TCW + cw)
                fb = arith.select(half != 0, word & fx.Int32(-65536), word << 16)
                cbits.append(fb)
            vf = _microblock_vf(cbits, col_rht, fold_scale=True)
            _lds_store1(lds.scr.ptr, mmb * _TC + c_col, _microblock_amax_f(vf))
            cvf_hold.append(vf)
        fx.barrier()
        col_base = (c_col // 32) * 32  # tile's first column within the LDS tile
        for mmb in range_constexpr(_RMB):
            vf = cvf_hold[mmb]
            tile_amax = fx.Int32(0)
            for i in range_constexpr(32):
                tile_amax = _imax(tile_amax, _lds_load1(lds.scr.ptr, mmb * _TC + col_base + i))
            native_bits, cbiased = _compute_scale_native(tile_amax, exp_up=vf_exp_up(col_rht))
            cwords = _cvt_microblock_to_fp4(vf, arith.bitcast(T.f32, native_bits), _col_seed(mmb))
            gcol = _col0 + c_col
            gmmb = rblk * _RMB + mmb
            cob = gcol * (rpad >> 3) + gmmb * 4 + gco
            csoff = _bsc_off(gcol, gmmb, cpk[2]) if cpk else gcol * (rpad >> 5) + gmmb + gcsc
            if padded:
                cok = gcol < C  # col-out has real-C rows; drop pad-K rows
                cob = arith.select(cok, cob, fx.Int32(_OOB))
                csoff = arith.select(cok, csoff, fx.Int32(_OOB))
            _store_words_vec4(corsrc, cob, cwords)
            buffer_ops.buffer_store(arith.trunci(T.i8, cbiased & 0xFF), cscrsrc, csoff)
    else:
        for mmb in range_constexpr(_RMB):
            row0 = mmb * 32
            cbits = []
            for row in range_constexpr(32):
                word = _lds_load1(lds.buf.ptr, (row0 + row) * _TCW + cw)
                fb = arith.select(half != 0, word & fx.Int32(-65536), word << 16)
                cbits.append(fb)
            cwords, cbiased = _finish_microblock(cbits, col_rht, _col_seed(mmb))
            gcol = _col0 + c_col
            gmmb = rblk * _RMB + mmb
            cob = gcol * (rpad >> 3) + gmmb * 4 + gco
            csoff = _bsc_off(gcol, gmmb, cpk[2]) if cpk else gcol * (rpad >> 5) + gmmb + gcsc
            if padded:
                cok = gcol < C  # col-out has real-C rows; drop pad-K rows
                cob = arith.select(cok, cob, fx.Int32(_OOB))
                csoff = arith.select(cok, csoff, fx.Int32(_OOB))
            _store_words_vec4(corsrc, cob, cwords)
            buffer_ops.buffer_store(arith.trunci(T.i8, cbiased & 0xFF), cscrsrc, csoff)


def _build_dual_kernel(
    row_rht, col_rht, row_2d=False, col_2d=False, col_locality=False, row_sr=False, col_sr=False
):
    """Single-recipe fused LDS dual (one coalesced 32x256 tile load feeds both the
    rowwise and colwise-transpose casts). Thin wrapper over ``_emit_dual_body``.
    ``col_locality`` (set for C>R shapes) flips the block order to combine the
    transpose stores; see ``_emit_dual_body``. ``row_sr``/``col_sr`` enable
    stochastic rounding on that direction (uses the per-launch ``SR_SEED``)."""
    _DualSS = _make_dual_struct(bool(row_2d or col_2d))

    @flyc.kernel(known_block_size=[BLK, 1, 1])
    def _dual_kernel(
        X: fx.Tensor,  # int32 view [R, C/2]
        ROW_OUT: fx.Tensor,  # int32 view [R, C/8]
        ROW_SC: fx.Tensor,  # uint8 [R, C/32]
        COL_OUT: fx.Tensor,  # int32 view [C, R/8]
        COL_SC: fx.Tensor,  # uint8 [C, R/32]
        R: fx.Int32,
        C: fx.Int32,
        SR_SEED: fx.Int32,  # per-launch stochastic-rounding seed (0 when SR off)
    ):
        lds = fx.SharedAllocator().allocate(_DualSS).peek()
        tid = fx.thread_idx.x
        _emit_dual_body(
            row_rht,
            col_rht,
            row_2d,
            col_2d,
            lds,
            tid,
            X,
            ROW_OUT,
            ROW_SC,
            COL_OUT,
            COL_SC,
            R,
            C,
            fx.block_idx.x,
            col_locality=col_locality,
            row_sr=row_sr,
            col_sr=col_sr,
            sr_seed=SR_SEED,
            sr_gbid=fx.block_idx.x,
        )

    return _dual_kernel


def _build_dual_launch(
    row_rht, col_rht, row_2d=False, col_2d=False, col_locality=False, row_sr=False, col_sr=False
):
    kern = _build_dual_kernel(row_rht, col_rht, row_2d, col_2d, col_locality, row_sr, col_sr)

    @flyc.jit
    def _dual_launch(
        X: fx.Tensor,
        ROW_OUT: fx.Tensor,
        ROW_SC: fx.Tensor,
        COL_OUT: fx.Tensor,
        COL_SC: fx.Tensor,
        R: fx.Int32,
        C: fx.Int32,
        SR_SEED: fx.Int32,
        grid_x: fx.Int32,
        stream: fx.Stream,
    ):
        kern(X, ROW_OUT, ROW_SC, COL_OUT, COL_SC, R, C, SR_SEED).launch(
            grid=(grid_x, 1, 1), block=(BLK, 1, 1), stream=stream
        )

    return _dual_launch


_DUAL_LAUNCH = {}
_DUAL_COMPILED = {}


def dual_eligible(R, C, row_recipe, col_recipe):
    """True if the FlyDSL fused dual can wholesale-replace the C++ dual for these
    recipes/dims (no preshuffle, dims aligned -> no padding). Both the per-microblock
    (2d=F) and the 2d-block (2d=T weight) amax geometries are supported and bit-exact
    vs C++ (non-SR); SR is supported (unbiased, not bit-exact). Shuffled recipes still
    fall back."""
    return (
        not row_recipe.shuffle_scale
        and not row_recipe.shuffle_out
        and not col_recipe.shuffle_scale
        and not col_recipe.shuffle_out
        and (R % 128 == 0)
        and (C % 256 == 0)
    )


def flydsl_dual_quant(
    x_bf16, fp4_dtype, row_rht, col_rht, row_2d=False, col_2d=False, row_sr=False, col_sr=False
):
    """Fused rowwise + colwise-transpose mxfp4 cast (one bf16 read). Returns
    (row_data, row_scale, col_data, col_scale) in C++-compatible dtypes/shapes.
    ``row_sr``/``col_sr`` request stochastic rounding on that direction."""
    import torch

    R, C = x_bf16.shape
    dev = x_bf16.device
    x_i32 = x_bf16.view(torch.int32)  # [R, C/2]
    ro = torch.empty((R, C // 8), dtype=torch.int32, device=dev)
    rs = torch.empty((R, C // 32), dtype=torch.uint8, device=dev)
    co = torch.empty((C, R // 8), dtype=torch.int32, device=dev)
    cs = torch.empty((C, R // 32), dtype=torch.uint8, device=dev)
    fn, grid_x = get_dual_cast(R, C, row_rht, col_rht, row_2d, col_2d, row_sr, col_sr)
    sr_seed = _next_sr_seed() if (row_sr or col_sr) else 0
    fn(x_i32, ro, rs, co, cs, R, C, sr_seed, grid_x, torch.cuda.current_stream())
    row_data = ro.view(torch.uint8).view(fp4_dtype)  # [R, C/2] fp4
    col_data = co.view(torch.uint8).view(fp4_dtype)  # [C, R/2] fp4
    row_scale = rs.view(torch.float8_e8m0fnu)
    col_scale = cs.view(torch.float8_e8m0fnu)
    return row_data, row_scale, col_data, col_scale


def get_dual_cast(R, C, row_rht, col_rht, row_2d=False, col_2d=False, row_sr=False, col_sr=False):
    """Return (compiled_fn, grid_x) for the fused dual at
    (R, C, row_rht, col_rht, row_2d, col_2d, row_sr, col_sr).
    Requires R % 128 == 0 and C % 256 == 0 (no scale/output padding)."""
    col_locality = int(C) > int(R)  # C>R (down-proj): combine transpose stores
    lk = (bool(row_rht), bool(col_rht), bool(row_2d), bool(col_2d), col_locality, bool(row_sr), bool(col_sr))
    raw = _DUAL_LAUNCH.get(lk)
    if raw is None:
        raw = _build_dual_launch(
            bool(row_rht), bool(col_rht), bool(row_2d), bool(col_2d), col_locality, bool(row_sr), bool(col_sr)
        )
        _DUAL_LAUNCH[lk] = raw
    key = (
        int(R),
        int(C),
        bool(row_rht),
        bool(col_rht),
        bool(row_2d),
        bool(col_2d),
        bool(row_sr),
        bool(col_sr),
    )
    ent = _DUAL_COMPILED.get(key)
    if ent is None:
        import torch

        x = torch.zeros((R, C // 2), dtype=torch.int32, device="cuda")
        ro = torch.zeros((R, C // 8), dtype=torch.int32, device="cuda")
        rs = torch.zeros((R, C // 32), dtype=torch.uint8, device="cuda")
        co = torch.zeros((C, R // 8), dtype=torch.int32, device="cuda")
        cs = torch.zeros((C, R // 32), dtype=torch.uint8, device="cuda")
        grid_x = (R // _TR) * (C // _TC)
        stream = torch.cuda.current_stream()
        fn = flyc.compile(raw, x, ro, rs, co, cs, R, C, 0, grid_x, stream)
        ent = (fn, grid_x)
        _DUAL_COMPILED[key] = ent
    return ent


# ---- Batched-3D dual quant: [G,N,K] weight, all experts in ONE launch (G x the
# blocks -> fills the GPU even for small per-expert N, where the 2D dense kernel is
# occupancy-starved and drops to ~2.5 TB/s). Reuses _emit_dual_body per-tile with
# per-expert base offsets; SRDs cover the whole 3D (gmul=G). ----
def _build_dual3_kernel(
    row_rht, col_rht, row_2d=False, col_2d=False, padded=False, col_locality=False, row_sr=False,
    col_sr=False, rpk=None, cpk=None
):
    _DualSS = _make_dual_struct(bool(row_2d or col_2d))

    @flyc.kernel(known_block_size=[BLK, 1, 1])
    def _dual3_kernel(
        X: fx.Tensor,  # int32 view [G, R, C/2] (real)
        ROW_OUT: fx.Tensor,  # int32 view [G, R, CP/8]
        ROW_SC: fx.Tensor,  # uint8 [G, R, CP/32]
        COL_OUT: fx.Tensor,  # int32 view [G, C, RP/8]
        COL_SC: fx.Tensor,  # uint8 [G, C, RP/32]
        R: fx.Int32,
        C: fx.Int32,
        G: fx.Int32,
        CP: fx.Int32,  # K_pad (row-out cols); == C when aligned
        RP: fx.Int32,  # N_pad (col-out cols); == R when aligned
        SR_SEED: fx.Int32,
    ):
        lds = fx.SharedAllocator().allocate(_DualSS).peek()
        tid = fx.thread_idx.x
        cpad = CP if padded else C
        rpad = RP if padded else R
        ncblk = ((C + _TC - 1) // _TC) if padded else (C // _TC)  # ceil over real C (incl tail)
        tpg = (R // _TR) * ncblk  # tiles per expert
        # Gather each XCD's workgroups into one contiguous tile range so its COL_OUT
        # writes (265 MB of transposed weight, this kernel's dominant store stream)
        # land in one DRAM region instead of being strided across all 32 experts by
        # the hardware's bid%8 distribution. The grouped quant already does this.
        _pid = xcd_remap_pid(fx.block_idx.x, fx.Int32(tpg) * G, 8)
        g = _pid // tpg
        lbid = _pid - g * tpg
        _emit_dual_body(
            row_rht,
            col_rht,
            row_2d,
            col_2d,
            lds,
            tid,
            X,
            ROW_OUT,
            ROW_SC,
            COL_OUT,
            COL_SC,
            R,
            C,
            lbid,
            # per-expert element bases in index (64-bit): g * per_expert_elems overflows
            # int32 for large-G MoE (e.g. G=64: 63 * N*K/2 > 2^31); _emit_dual_body folds
            # these into per-expert int64 SRD bases.
            gx=arith.index_cast(T.index, g) * arith.index_cast(T.index, R * (C >> 1)),
            gro=arith.index_cast(T.index, g) * arith.index_cast(T.index, R * (cpad >> 3)),
            grsc=arith.index_cast(T.index, g)
            * arith.index_cast(T.index, fx.Int32(rpk[3]) if rpk else R * (cpad >> 5)),
            gco=arith.index_cast(T.index, g) * arith.index_cast(T.index, C * (rpad >> 3)),
            gcsc=arith.index_cast(T.index, g)
            * arith.index_cast(T.index, fx.Int32(cpk[3]) if cpk else C * (rpad >> 5)),
            gmul=G,
            padded=padded,
            ncblk=ncblk,
            CP=CP,
            RP=RP,
            col_locality=col_locality,
            batched=True,
            # global bid so different experts (same lbid) get independent seeds
            row_sr=row_sr,
            col_sr=col_sr,
            sr_seed=SR_SEED,
            sr_gbid=_pid,
            rpk=rpk,
            cpk=cpk,
        )

    return _dual3_kernel


def _build_dual3_launch(
    row_rht, col_rht, row_2d=False, col_2d=False, padded=False, col_locality=False, row_sr=False,
    col_sr=False, rpk=None, cpk=None
):
    kern = _build_dual3_kernel(
        row_rht, col_rht, row_2d, col_2d, padded, col_locality, row_sr, col_sr, rpk, cpk
    )

    @flyc.jit
    def _dual3_launch(X, ROW_OUT, ROW_SC, COL_OUT, COL_SC, R, C, G, CP, RP, SR_SEED, grid_x, stream):
        kern(X, ROW_OUT, ROW_SC, COL_OUT, COL_SC, R, C, G, CP, RP, SR_SEED).launch(
            grid=(grid_x, 1, 1), block=(BLK, 1, 1), stream=stream
        )

    return _dual3_launch


_DUAL3_LAUNCH = {}
_DUAL3_COMPILED = {}


def dual3_eligible(N, K, row_recipe, col_recipe):
    """True if the batched-3D FlyDSL dual can replace the C++ dual for a [G,N,K]
    weight (no preshuffle). Handles non-256 K / non-128 N via K_pad/N_pad (bit-exact
    vs the HIP dual whose pad is all-zero; SR is unbiased, not bit-exact). Needs
    N%64==0 (row/col tiling) and K%64==0 (32-microblock + vec4-aligned tail load mask)."""
    return (
        not row_recipe.shuffle_scale
        and not row_recipe.shuffle_out
        and not col_recipe.shuffle_scale
        and not col_recipe.shuffle_out
        and (N % 64 == 0)
        and (K % 64 == 0)
    )


def get_dual3_cast(N, K, G, row_rht, col_rht, row_2d=False, col_2d=False, row_sr=False, col_sr=False):
    """(compiled_fn, grid_x, K_pad, N_pad, padded) for the batched-3D dual at
    (N,K,G,recipes). K_pad=ceil(K/128)*128 (row-out), N_pad=ceil(N/128)*128 (col-out);
    `padded` when K not a 256-tile multiple or N not 128-multiple."""
    Kp = ((K + 127) // 128) * 128
    Np = ((N + 127) // 128) * 128
    tr, tc = _pick_tile_geom(int(N), int(K))
    padded = (K % tc != 0) or (N % 128 != 0)
    col_locality = int(K) > int(N)  # K>N: combine transpose stores (col-out)
    # packed NT B-scale geometry (row operand is [N, Kp], col operand is [K, Np])
    rpk = bsc_pack_geom(N, Kp // 32) if bsc_fused(N, Kp // 32) else None
    cpk = bsc_pack_geom(K, Np // 32) if bsc_fused(K, Np // 32) else None
    lk = (
        bool(row_rht),
        bool(col_rht),
        bool(row_2d),
        bool(col_2d),
        padded,
        col_locality,
        bool(row_sr),
        bool(col_sr),
    )
    lk = lk + (tr, tc, rpk, cpk)
    _saved = (_TR, _TC)
    _set_tile_geom(tr, tc)
    raw = _DUAL3_LAUNCH.get(lk)
    if raw is None:
        raw = _build_dual3_launch(
            bool(row_rht),
            bool(col_rht),
            bool(row_2d),
            bool(col_2d),
            padded,
            col_locality,
            bool(row_sr),
            bool(col_sr),
            rpk,
            cpk,
        )
        _DUAL3_LAUNCH[lk] = raw
    key = (int(N), int(K), int(G), *lk)
    if _DUAL3_COMPILED.get(key) is not None:
        _set_tile_geom(*_saved)
        return _DUAL3_COMPILED[key]
    ent = _DUAL3_COMPILED.get(key)
    if ent is None:
        import torch

        # trace-only tensors (shapes drive the compile, contents never read)
        x = torch.empty((G, N, K // 2), dtype=torch.int32, device="cuda")
        ro = torch.empty((G, N, Kp // 8), dtype=torch.int32, device="cuda")
        rs = (
            torch.empty(G * rpk[3], dtype=torch.uint8, device="cuda")
            if rpk
            else torch.empty((G, N, Kp // 32), dtype=torch.uint8, device="cuda")
        )
        co = torch.empty((G, K, Np // 8), dtype=torch.int32, device="cuda")
        cs = (
            torch.empty(G * cpk[3], dtype=torch.uint8, device="cuda")
            if cpk
            else torch.empty((G, K, Np // 32), dtype=torch.uint8, device="cuda")
        )
        ncblk = ((K + tc - 1) // tc) if padded else (K // tc)
        grid_x = (N // tr) * ncblk * G
        fn = flyc.compile(raw, x, ro, rs, co, cs, N, K, G, Kp, Np, 0, grid_x, torch.cuda.current_stream())
        ent = (fn, grid_x, Kp, Np, padded, rpk, cpk)
        _DUAL3_COMPILED[key] = ent
    _set_tile_geom(*_saved)
    return ent


def flydsl_dual_quant_batched(
    x3d, fp4_dtype, row_rht, col_rht, row_2d=False, col_2d=False, row_sr=False, col_sr=False
):
    """Batched-3D fused rowwise + colwise-transpose mxfp4 dual cast for a [G,N,K]
    weight in ONE launch. Returns C++-compatible per-expert
    (row_data [G,N,K/2], row_scale [G,N,K/32], col_data [G,K,N/2], col_scale [G,K,N/32]).
    ``row_sr``/``col_sr`` request stochastic rounding on that direction."""
    import torch

    G, N, K = x3d.shape
    dev = x3d.device
    x_i32 = x3d.contiguous().view(torch.int32)  # [G, N, K/2]
    fn, grid_x, Kp, Np, padded, rpk, cpk = get_dual3_cast(
        N, K, G, row_rht, col_rht, row_2d, col_2d, row_sr, col_sr
    )
    # Outputs sized on K_pad/N_pad; the pad regions must read back all-0 to match the HIP
    # dual (the GEMM contracts over the PADDED extent, so pad garbage would corrupt it).
    # Zeroing the whole buffer to achieve that costs ~570 MB of memset per weight quant
    # (G,N,Kp/8 + G,K,Np/8 i32 for GG1) to protect at most a few percent of tail columns.
    # Allocate uninitialised and zero only the tail slices past the real K / N extent.
    ro = torch.empty((G, N, Kp // 8), dtype=torch.int32, device=dev)
    co = torch.empty((G, K, Np // 8), dtype=torch.int32, device=dev)
    # Packed NT B-scale: the tile grid covers only the REAL rows/cols, so the slots the
    # preshuffle used to zero (the 256-row pad quartet, and the E8M0 columns past the real
    # extent) are never stored to -- allocate zeroed instead of adding a fill dispatch.
    rs = (
        torch.zeros(G * rpk[3], dtype=torch.uint8, device=dev)
        if rpk
        else torch.empty((G, N, Kp // 32), dtype=torch.uint8, device=dev)
    )
    cs = (
        torch.zeros(G * cpk[3], dtype=torch.uint8, device=dev)
        if cpk
        else torch.empty((G, K, Np // 32), dtype=torch.uint8, device=dev)
    )
    if padded:
        if Kp != K:
            ro[..., K // 8 :].zero_()
            if rpk is None:
                rs[..., K // 32 :].zero_()
        if Np != N:
            co[..., N // 8 :].zero_()
            if cpk is None:
                cs[..., N // 32 :].zero_()
    sr_seed = _next_sr_seed() if (row_sr or col_sr) else 0
    fn(x_i32, ro, rs, co, cs, N, K, G, Kp, Np, sr_seed, grid_x, torch.cuda.current_stream())
    return (
        ro.view(torch.uint8).view(fp4_dtype),
        rs.view(torch.float8_e8m0fnu),
        co.view(torch.uint8).view(fp4_dtype),
        cs.view(torch.float8_e8m0fnu),
    )
