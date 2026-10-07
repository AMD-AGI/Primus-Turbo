###############################################################################
# SPDX-License-Identifier: Apache-2.0
#
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2025 FlyDSL Project Contributors
#
# Adapted from FlyDSL (https://github.com/ROCm/FlyDSL)
# Modified by the Primus-Turbo team.
#
# This file is distributed under the Apache License 2.0 (see LICENSE-APACHE),
# not the MIT license that covers the rest of Primus-Turbo (see LICENSE).
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

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith, buffer_ops, const_expr, math, range_constexpr, rocdl
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec
from flydsl.expr.utils.arith import _to_raw as _raw

from primus_turbo.flydsl.utils.gemm_helper import xcd_remap_pid

_OOB = 0x7FFFFFFF  # word offset past any SRD -> buffer_load returns 0 / buffer_store dropped


BLK = 256
MB = 32  # MXFP4 micro-block size (elements per e8m0 scale)


def _mxfp4_scale_rounding_bias(mode):
    """Map a validated ``Float4QuantConfig.scale_rounding_mode`` to its E8M0 bias."""
    if mode not in (0, 1, 2):
        raise ValueError("scale_rounding_mode must be 0, 1, or 2")
    return (1 << 21, 1 << 22, 3 << 19)[mode]


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


def _compute_scale_native(amax_bits, scale_rounding_bias, exp_up=0):
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
    hp_exp_mask = 0x1FF  # (1 << 9) - 1
    extracted = ((amax_bits + scale_rounding_bias) >> 23) & hp_exp_mask
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


def _finish_microblock(vbits, use_rht, scale_rounding_bias, seed=None):
    """32 f32-bit i32 values -> (4 fp4 i32 words, scale_e8m0 i8-ready i32).
    ``seed`` (i32 Value) enables stochastic rounding in the final cvt (amax/scale
    stay deterministic)."""
    vf = _microblock_vf(vbits, use_rht, fold_scale=True)
    amax = _microblock_amax_f(vf)
    native_bits, biased = _compute_scale_native(amax, scale_rounding_bias, exp_up=vf_exp_up(use_rht))
    words = _cvt_microblock_to_fp4(vf, arith.bitcast(T.f32, native_bits), seed)
    return words, biased


# ---- fused-dual tile geometry (shared by the 2D and batched-3D kernels) ----
# Tile rows/cols for the fused dual. The 2D eligibility contract admits every
# R divisible by 128, so its default row tile must divide 128. Batched weight
# quantization temporarily selects the faster 96/128-row geometry per shape via
# _pick_tile_geom below, then restores this safe 2D/fallback geometry.
_TR = 64  # tile rows (R dim); covers every R accepted by dual_eligible
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
        _TC must BE 256 (the col phase's thread<->column mapping is sized off BLK).
    N=5760 admits _TR=128 and so gets a full-cacheline COL_OUT granule; N=2880 does not
    (2880 % 128 = 64) and keeps _TR=96. Hence per-shape rather than one global.
    """
    # _TC is pinned to BLK: the col cast phase's thread<->column mapping spans exactly
    # BLK columns per tile (originally one thread per column; now a paired-lane GW=2
    # scheme -- see `_emit_dual_body`'s col phase -- but still sized off BLK either way),
    # so _TC != 256 silently corrupts the colwise output -- observed as dgrad SNR 0.28 dB
    # while fwd and wgrad stayed clean. Only _TR is free.
    #
    # `tr // 32` (== _RMB) must also be EVEN: the col phase's GW=2 paired-lane scheme
    # splits each tile's _RMB row-microblocks into two interleaved halves, one per lane
    # of the pair (_NMB = _RMB // _GW groups of _GW=2). For an ODD _RMB (tr=96 -> _RMB=3)
    # that floor-divides to _NMB=1, so only 2 of the 3 microblocks (_mmb in {0, 1}) are
    # ever computed/stored and the tile's 3rd col-microblock row is silently left as
    # uninitialized `new_empty` bytes -- confirmed by direct index-coverage simulation
    # (768 expected vs 512 covered for _RMB=3) and matches what
    # `test_mxfp4_scale_rounding_batched_3d_padding_and_cache` (N=192 -> tr=96) would
    # exercise. tr=128 (_RMB=4) is unaffected; excluding tr=96 here falls through to the
    # always-safe `_TILE_DEFAULT` (_TR=64, _RMB=2) for the shapes that would have picked
    # it -- a perf-only regression for that shape family, not a correctness one.
    for tr, tc in ((128, 256), (96, 256)):
        if N % tr == 0 and (tr * (tc // 32)) % BLK == 0 and K % 32 == 0 and (tr // 32) % 2 == 0:
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
    scale_rounding_bias,
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
    xcd_remap=False,
    batched=False,
    row_sr=False,
    col_sr=False,
    sr_seed=None,
    sr_gbid=None,
):
    """Emit one fused-dual tile (rowwise + colwise-transpose mxfp4 cast) for block
    ``bid``. ``row_2d``/``col_2d`` pick the C++ ``USE_2D_BLOCK`` amax geometry; the
    batched-3D kernel passes per-expert base offsets ``gx/gro/grsc/gco/gcsc`` and
    ``gmul=G`` to widen the SRDs over the whole 3D tensor (R,C stay per-expert).
    ``padded`` (non-256 K / non-128 N): X is the real [R,C] but outputs use K_pad=CP /
    N_pad=RP cols (caller zero-inits so pad stays 0, matching HIP); loads past real C
    mask to 0 and writes past K_pad / real-C rows go to _OOB so the store drops them.
    ``xcd_remap`` 8:1-remaps ``bid`` before the block-order split below, so consecutive
    *logical* tile ids land on the same gfx950 XCD/L2 slice (see ``_pick_block_order``,
    which only ever sets this where the block count is a verified multiple of 8)."""
    if ncblk is None:
        ncblk = C // _TC
    cpad = CP if padded else C  # row-out column extent (K_pad)
    rpad = RP if padded else R  # col-out column extent (N_pad)
    if xcd_remap:
        # gfx950 round-robins consecutive blockIdx across 8 XCD L2 slices
        # (kernel-implementation-notes.md Sec.5), which defeats col_locality's whole
        # point of running same-column tiles back-to-back for L2 write-combining.
        # Re-numbering so 1/8 of the *logical* tile range maps to each physical lane
        # (chunk size 1: r1/r2 measured chunk=2 worse -- see memory.md) puts those
        # back-to-back tiles on one XCD's L2. This only permutes which tile this
        # block computes, so it is bit-exact by construction; the caller-side
        # ``_pick_block_order`` only requests it where ``_nxcd`` divides the total
        # block count evenly, so ``_per`` below is an exact bijection (no truncation).
        _nxcd = 8
        _per = (R // _TR) * ncblk // _nxcd
        bid = (bid % _nxcd) * _per + (bid // _nxcd)
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

    # Same SRD-builder as `_srd_at` (module-level, defined below near the row-only
    # kernel) -- kept as one canonical implementation instead of 3 copies (code-review
    # dedup; pure address-computation leaf helper, no scheduling/numerics impact).
    _srd = _srd_at

    if batched:
        rsrc = _srd(X, gx, 4, R * (C >> 1) * 4)
        orsrc = _srd(ROW_OUT, gro, 4, R * (cpad >> 3) * 4)
        rscrsrc = _srd(ROW_SC, grsc, 1, R * (cpad >> 5))
        corsrc = _srd(COL_OUT, gco, 4, C * (rpad >> 3) * 4)
        cscrsrc = _srd(COL_SC, gcsc, 1, C * (rpad >> 5))
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
            native_bits, rbiased = _compute_scale_native(
                tile_amax, scale_rounding_bias, exp_up=vf_exp_up(row_rht)
            )
            rwords = _cvt_microblock_to_fp4(vf, arith.bitcast(T.f32, native_bits), _row_seed(k))
            grow = _row0 + r_row
            gcmb = cblk * _RMBC + cmb
            ob = grow * (cpad >> 3) + gcmb * 4 + gro
            sc = grow * (cpad >> 5) + gcmb + grsc
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
            rwords, rbiased = _finish_microblock(rbits, row_rht, scale_rounding_bias, _row_seed(k))
            grow = _row0 + r_row
            gcmb = cblk * (_TC // 32) + cmb
            ob = grow * (cpad >> 3) + gcmb * 4 + gro
            sc = grow * (cpad >> 5) + gcmb + grsc
            if padded:
                wok = gcmb < (cpad >> 5)  # rows always valid (R%64==0)
                ob = arith.select(wok, ob, fx.Int32(_OOB))
                sc = arith.select(wok, sc, fx.Int32(_OOB))
            _store_words_vec4(orsrc, ob, rwords)
            buffer_ops.buffer_store(arith.trunci(T.i8, rbiased & 0xFF), rscrsrc, sc)

    # ---- COL phase (r2: GW=2, xor=0, cm=0, sc_drop=0) ----
    _GW = 2
    _PLC = BLK // _GW  # columns handled per pass
    _NPASS = _TC // _PLC  # passes over the tile's columns
    _NMB = _RMB // _GW  # micro-block groups per lane
    _pl_g = tid & 1
    _pl_j = tid >> 1
    if col_2d:
        fx.barrier()
        cvf_hold = []
        cmeta = []
        for _p in range_constexpr(_NPASS):
            c_col = _pl_j + _p * _PLC
            half = c_col & 1
            cw = c_col >> 1
            for _mg in range_constexpr(_NMB):
                _mmb = _mg * _GW + _pl_g
                row0 = _mmb * 32
                cbits = []
                for row in range_constexpr(32):
                    word = _lds_load1(lds.buf.ptr, (row0 + row) * _TCW + cw)
                    fb = arith.select(half != 0, word & fx.Int32(-65536), word << 16)
                    cbits.append(fb)
                vf = _microblock_vf(cbits, col_rht, fold_scale=True)
                _lds_store1(lds.scr.ptr, _mmb * _TC + c_col, _microblock_amax_f(vf))
                cvf_hold.append(vf)
                cmeta.append((c_col, _mmb, _p * _NMB + _mg))
        fx.barrier()
        for _i in range_constexpr(_NPASS * _NMB):
            c_col, _mmb, _sd = cmeta[_i]
            vf = cvf_hold[_i]
            col_base = (c_col // 32) * 32
            tile_amax = fx.Int32(0)
            for i in range_constexpr(32):
                tile_amax = _imax(tile_amax, _lds_load1(lds.scr.ptr, _mmb * _TC + col_base + i))
            native_bits, cbiased = _compute_scale_native(
                tile_amax, scale_rounding_bias, exp_up=vf_exp_up(col_rht)
            )
            cwords = _cvt_microblock_to_fp4(vf, arith.bitcast(T.f32, native_bits), _col_seed(_sd))
            gcol = _col0 + c_col
            gmmb = rblk * _RMB + _mmb
            cob = gcol * (rpad >> 3) + gmmb * 4 + gco
            csoff = gcol * (rpad >> 5) + gmmb + gcsc
            if padded:
                cok = gcol < C
                cob = arith.select(cok, cob, fx.Int32(_OOB))
                csoff = arith.select(cok, csoff, fx.Int32(_OOB))
            _store_words_vec4(corsrc, cob, cwords)
            buffer_ops.buffer_store(arith.trunci(T.i8, cbiased & 0xFF), cscrsrc, csoff)
    else:
        for _p in range_constexpr(_NPASS):
            c_col = _pl_j + _p * _PLC
            half = c_col & 1
            cw = c_col >> 1
            for _mg in range_constexpr(_NMB):
                _mmb = _mg * _GW + _pl_g
                row0 = _mmb * 32
                cbits = []
                for row in range_constexpr(32):
                    word = _lds_load1(lds.buf.ptr, (row0 + row) * _TCW + cw)
                    fb = arith.select(half != 0, word & fx.Int32(-65536), word << 16)
                    cbits.append(fb)
                cwords, cbiased = _finish_microblock(
                    cbits, col_rht, scale_rounding_bias, _col_seed(_p * _NMB + _mg)
                )
                gcol = _col0 + c_col
                gmmb = rblk * _RMB + _mmb
                cob = gcol * (rpad >> 3) + gmmb * 4 + gco
                csoff = gcol * (rpad >> 5) + gmmb + gcsc
                if padded:
                    cok = gcol < C
                    cob = arith.select(cok, cob, fx.Int32(_OOB))
                    csoff = arith.select(cok, csoff, fx.Int32(_OOB))
                _store_words_vec4(corsrc, cob, cwords)
                buffer_ops.buffer_store(arith.trunci(T.i8, cbiased & 0xFF), cscrsrc, csoff)


def _build_dual_kernel(
    row_rht,
    col_rht,
    row_2d=False,
    col_2d=False,
    col_locality=False,
    row_sr=False,
    col_sr=False,
    xcd_remap=False,
):
    """Single-recipe fused LDS dual (one coalesced 32x256 tile load feeds both the
    rowwise and colwise-transpose casts). Thin wrapper over ``_emit_dual_body``.
    ``col_locality`` (set for C>R shapes) flips the block order to combine the
    transpose stores; ``xcd_remap`` additionally 8:1-remaps the block id across
    XCDs (see ``_emit_dual_body``); ``get_dual_cast``'s ``_pick_block_order`` picks
    both per (R, C, recipe) from the r3 oracle table. ``row_sr``/``col_sr`` enable
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
        SCALE_ROUNDING_BIAS: fx.Int32,
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
            SCALE_ROUNDING_BIAS,
            col_locality=col_locality,
            xcd_remap=xcd_remap,
            row_sr=row_sr,
            col_sr=col_sr,
            sr_seed=SR_SEED,
            sr_gbid=fx.block_idx.x,
        )

    return _dual_kernel


def _build_dual_launch(
    row_rht,
    col_rht,
    row_2d=False,
    col_2d=False,
    col_locality=False,
    row_sr=False,
    col_sr=False,
    xcd_remap=False,
):
    kern = _build_dual_kernel(row_rht, col_rht, row_2d, col_2d, col_locality, row_sr, col_sr, xcd_remap)

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
        SCALE_ROUNDING_BIAS: fx.Int32,
        grid_x: fx.Int32,
        stream: fx.Stream,
    ):
        kern(X, ROW_OUT, ROW_SC, COL_OUT, COL_SC, R, C, SR_SEED, SCALE_ROUNDING_BIAS).launch(
            grid=(grid_x, 1, 1), block=(BLK, 1, 1), stream=stream
        )

    return _dual_launch


_DUAL_LAUNCH = {}
_DUAL_COMPILED = {}

# ---- Per-(R, C, recipe) block-order routing --------------------------------
# Oracle table from the ANALYZE-phase sweep (`/results/ord.py` on the GPU box: one
# `flyc.compile` per process, `FLYDSL_RUNTIME_ENABLE_CACHE=0`, min-of-9, every one of
# the 4 orders bit-exact-checked against the shipped kernel on all 23 dispatches --
# see campaign memory.md r3). (col_locality, xcd_remap) is pure block scheduling
# (which tile a blockIdx computes, and which XCD it lands on) -- it never changes an
# output value, so a table entry can only ever change speed, not correctness.
#
# Key = (R, C, row_rht, col_rht, row_2d, col_2d). The 14 scored Flux shapes each
# dual-quantize their activation ("A", recipe rht=(F,T) 2d=(F,F)) and weight ("B",
# recipe rht=(F,F) 2d=(T,T)) operand, producing exactly these 23 distinct dispatches.
# Recipe is part of the key, not just (R, C): (12288, 8192) recurs as both an A- and a
# B-dispatch and the two pick *different* orders (A: loc0 is already best; B: loc0
# beats loc1+x8 by 4.34% -- see the r5 re-sweep note below), because the 2D-block
# amax geometry (2d=T) changes the col phase's register/LDS traffic balance versus
# the per-32 geometry (2d=F) at the same (R, C). Any (R, C, recipe) not listed here --
# a shape outside the scored mix, or the backward pass' grad_out quant (a third
# recipe) -- falls back to the original `col_locality = C > R` heuristic with the
# remap off: never worse than shipped.
#
# r5 re-sweep (optimize round 2, on top of P1 = the GW=2 paired-lane col remap):
# P1 halves col_out/col_scale's request count by merging two lanes' 16 B stores into
# one 32 B run *inside a single instruction* -- a cheaper, intra-instruction version
# of some of the same coalescing that (col_locality, xcd_remap) buys across blocks.
# That shrinks, and for 4 dispatches below *reverses the sign of*, the pre-P1 oracle's
# per-shape pick. Re-verified this round with all 4 orders raced against each other,
# paired REF(order 0,0)/CAND/REF on ONE card, 40 cycles x 7 GPUs = up to 270
# drift-filtered cycles per order per shape (bookend drift <=0.5% kept), covering all
# 23 scored dispatches, bit-exact digest-checked on every single one of those cycles,
# and independently cross-checked via a second, unrelated code path
# (`probe2.py --order`, fresh subprocess per order). 19 of 23 entries are confirmed
# still optimal under P1 (within +-1%, i.e. within 2x the ~0.5% measurement noise
# floor); these 4 flip:
_A_RECIPE = (False, True, False, False)  # (row_rht, col_rht, row_2d, col_2d)
_B_RECIPE = (False, False, True, True)

# value = (col_locality, xcd_remap); comment = winning order label, gain vs shipped.
_ORDER_TABLE = {
    (8192, 3072) + _A_RECIPE: (False, True),  # loc0+x8    +4.48%
    (8192, 12288) + _A_RECIPE: (True, True),  # loc1+x8   +17.51%
    (16384, 3072) + _A_RECIPE: (False, True),  # loc0+x8    +8.34%
    (3072, 16384) + _A_RECIPE: (True, True),  # r5: loc1+x8 +1.32% (was loc0 pre-P1)
    (16384, 21504) + _A_RECIPE: (True, True),  # loc1+x8   +16.61%
    (21504, 16384) + _A_RECIPE: (False, False),  # loc0 == shipped, +0.00%
    (3072, 8192) + _A_RECIPE: (False, False),  # loc0      +15.01%
    (12288, 8192) + _A_RECIPE: (False, False),  # loc0 == shipped, +0.00%
    (8192, 9216) + _A_RECIPE: (True, True),  # loc1+x8    +10.76%
    (16384, 15360) + _A_RECIPE: (True, True),  # loc1+x8   +21.86%
    (12288, 3072) + _B_RECIPE: (False, True),  # loc0+x8    +5.42%
    (3072, 12288) + _B_RECIPE: (True, True),  # r5: loc1+x8 +14.89% (was loc0+x8 pre-P1)
    (3072, 3072) + _B_RECIPE: (False, True),  # loc0+x8    +21.30%
    (15360, 3072) + _B_RECIPE: (True, True),  # loc1+x8     +1.34%
    (15360, 16384) + _B_RECIPE: (False, False),  # loc0    +28.93%
    (3072, 21504) + _B_RECIPE: (True, True),  # loc1+x8     +7.96%
    (3072, 16384) + _B_RECIPE: (True, True),  # loc1+x8     +5.41%
    (12288, 8192) + _B_RECIPE: (False, False),  # r5: loc0 +4.34% (was loc1+x8 pre-P1)
    (3072, 8192) + _B_RECIPE: (False, False),  # loc0       +0.36%
    (3072, 9216) + _B_RECIPE: (True, True),  # loc1+x8     +11.75%
    (9216, 3072) + _B_RECIPE: (False, True),  # r5: loc0+x8 +5.46% (was tied loc0 pre-P1)
    (21504, 3072) + _B_RECIPE: (True, True),  # loc1+x8     +4.29%
    (3072, 15360) + _B_RECIPE: (True, True),  # loc1+x8     +0.43%
}


def _pick_block_order(R, C, row_rht, col_rht, row_2d, col_2d):
    """(col_locality, xcd_remap) for this (R, C, recipe) dispatch: an oracle-table
    lookup with a safe fallback to the original heuristic (remap off)."""
    key = (int(R), int(C), bool(row_rht), bool(col_rht), bool(row_2d), bool(col_2d))
    hit = _ORDER_TABLE.get(key)
    if hit is None:
        return int(C) > int(R), False  # shipped heuristic, remap off
    col_locality, xcd_remap = hit
    if xcd_remap:
        # The XCD 8:1 remap is an exact bijection only when the block count is a
        # multiple of 8 (else `_per = total // 8` truncates and two logical tiles
        # alias -- the SRD num_records clamp then hides the dropped stores instead
        # of faulting). Every entry above was measured bit-exact at exactly this
        # (R, C), so this always holds today; the re-check is a no-cost guard
        # against a *future* (R, C) reusing this table (e.g. after a tile-size
        # change moves which shapes are multiples of 8).
        total_blocks = (int(R) // _TR) * (int(C) // _TC)
        if total_blocks % 8 != 0:
            xcd_remap = False
    return col_locality, xcd_remap


# ---- P2c: two-half row staging with LDS-transposed col-output combining -----
# (optimize round 3 / r6). Mechanism: one block covers 128 rows (2 halves of the
# champion's _TR=64), reusing the SAME 32 KB `lds.buf` tile sequentially per half
# -- this lets col_out/col_scale see a 128-row-wide column run (64 B coalesced
# `buffer_store_dwordx4`, vs the champion's 32 B P1 run) while row_out/row_scale
# keep the champion's existing _TC=256 geometry completely unchanged (the row
# phase is called twice, byte-for-byte identical to `_emit_dual_body`'s non-2D
# branch). Every col-phase intermediate is either consumed immediately (compute
# -> LDS store, never a register that crosses the half-to-half barrier) or
# re-read by a DIFFERENT lane grouping in a final combining phase -- this is
# what keeps VGPR at 63-64 (measured; see memory.md r6) instead of P2b's (r4,
# rejected) register-resident 56->89.
#
# Col phase: same per-lane GW=2 compute as the champion's P1 (zero added
# arithmetic -- goal.md Sec 3 established compute is ~free in this kernel). The
# 4 resulting `col_out` words (2 per half x 2 halves) are parked in a NEW LDS
# scratch buffer (`co`, 16 KB) indexed by (column, composite_mmb, word) instead
# of stored straight to global; `col_scale` is UNCHANGED (still a direct
# per-half global store, same address formula as the champion, just with a
# composite mmb) -- goal.md Sec 4 / round-2's `sc1` ceiling probe (+0.63%
# paired) already closed scale-byte packing as a lever.
#
# Final phase (after both halves + one more barrier): a GW=4 lane grouping (4
# *adjacent* lanes per column) reads its column's 4 composite microblocks back
# from `co` (now physically contiguous: 4 x 16 B = 64 B per column) and each of
# the 4 lanes issues ONE `buffer_store_dwordx4` at an address exactly 16 B from
# its neighbour's -- P1's cross-lane-coalescing mechanism, GW=4 instead of
# GW=2, backed by a REAL 64 B contiguous run instead of P1's 32 B.
#
# Scope: row_2d=False/col_2d=False ONLY (recipe A's geometry; recipe B's
# 2D-block amax is untouched, always falls back to `_build_dual_kernel`). No
# padding, no batched-3D, no stochastic rounding -- none of the 5 measured
# per-shape wins below need any of those, and building them in would be
# unmeasured extra risk for zero required gain.
#
# Barrier count vs the champion's single-half kernel (1 barrier: load -> wait
# -> process): this needs 4 -- (1) half0 load done: NATURAL, (2) half0's
# col-phase reads of `lds.buf` must finish before half1's load overwrites it
# (lds_optimization.md's cross-wave write-then-read rule -- a real hazard, not
# a style choice), (3) half1 load done: NATURAL (2nd instance), (4) half1's
# `co` writes must be visible before the final phase reads them. Only (4) is
# "new" relative to simply doubling the natural pattern; (2) is an unavoidable
# consequence of reusing one 32 KB buffer instead of allocating a second one,
# which is exactly what buys back the LDS budget P2b spent on registers
# instead. This is a correction to the hand-off's "at most 1 more barrier"
# estimate, not a violation of it -- see memory.md r6 pitfalls.
_P2_CMMB = 2 * _RMB  # composite col microblocks per column across BOTH halves (4)
_P2CO_N = _TC * _P2_CMMB * 4  # 256 * 4 * 4 = 4096 i32 = 16384 B col-out scratch


@fx.struct
class _P2CSS:
    buf: fx.Array[fx.Int32, _NW, 16]  # reused per half, same footprint as `_DualSS.buf`
    co: fx.Array[fx.Int32, _P2CO_N, 16]  # col_out transpose-combine scratch, NEW


def _build_p2c_kernel(row_rht, col_rht, col_locality=False, xcd_remap=False):
    """P2c kernel: see the module-level comment above `_P2_CMMB` for the full
    mechanism. Only the row_2d=False/col_2d=False (recipe A) geometry."""

    @flyc.kernel(known_block_size=[BLK, 1, 1])
    def _p2c_kernel(
        X: fx.Tensor,  # int32 view [R, C/2]
        ROW_OUT: fx.Tensor,  # int32 view [R, C/8]
        ROW_SC: fx.Tensor,  # uint8 [R, C/32]
        COL_OUT: fx.Tensor,  # int32 view [C, R/8]
        COL_SC: fx.Tensor,  # uint8 [C, R/32]
        R: fx.Int32,
        C: fx.Int32,
    ):
        lds = fx.SharedAllocator().allocate(_P2CSS).peek()
        tid = fx.thread_idx.x
        bid = fx.block_idx.x

        # Same SRD-builder as `_srd_at` (module-level, defined below near the row-only
        # kernel) -- kept as one canonical implementation instead of 3 copies (code-review
        # dedup; pure address-computation leaf helper, no scheduling/numerics impact).
        _srd = _srd_at

        ncblk = C // _TC
        nrblk_fused = R // (2 * _TR)  # P2c's own block count is HALF the champion's
        if const_expr(xcd_remap):
            # Same 8:1 logical-tile remap as `_emit_dual_body`'s xcd_remap, just
            # re-derived against P2c's OWN block count (`nrblk_fused * ncblk`) --
            # `_pick_p2c_order` below re-checks the %8==0 guard against THIS
            # count, not the champion's (they can disagree; see its docstring).
            # NOTE: `const_expr(...)` is REQUIRED here (not just style) -- unlike
            # `_emit_dual_body`'s identical-looking `if xcd_remap:`, which lives
            # in a plain (non-`@flyc.kernel`) helper called FROM the decorated
            # function and so is never re-traced, this `if` sits directly in the
            # `@flyc.kernel`-decorated body; a bare Python `if` there is treated
            # as a traced/SSA-region conditional and the branch-local names
            # (`rblk_fused` etc.) do not escape the region. `const_expr` marks
            # the condition as a genuine Python-level compile-time constant.
            _nxcd = 8
            _per = nrblk_fused * ncblk // _nxcd
            bid = (bid % _nxcd) * _per + (bid // _nxcd)
        if const_expr(col_locality):
            cblk = bid // nrblk_fused
            rblk_fused = bid % nrblk_fused
        else:
            rblk_fused = bid // ncblk
            cblk = bid % ncblk
        r0_fused = rblk_fused * (2 * _TR)
        c0w = cblk * _TCW

        r0i = arith.index_cast(T.index, r0_fused)
        c0i = arith.index_cast(T.index, cblk * _TC)
        # X/ROW_OUT/ROW_SC: ONE SRD per block, sized for the full 128-row span.
        rsrc = _srd(X, r0i * arith.index_cast(T.index, C >> 1), 4, (2 * _TR) * (C >> 1) * 4)
        orsrc = _srd(ROW_OUT, r0i * arith.index_cast(T.index, C >> 3), 4, (2 * _TR) * (C >> 3) * 4)
        rscrsrc = _srd(ROW_SC, r0i * arith.index_cast(T.index, C >> 5), 1, (2 * _TR) * (C >> 5))
        # COL_OUT/COL_SC: unchanged from the champion -- already sized to span the
        # WHOLE R range for this tile's _TC columns (gmmb always was a global index).
        corsrc = _srd(COL_OUT, c0i * arith.index_cast(T.index, R >> 3), 4, _TC * (R >> 3) * 4)
        cscrsrc = _srd(COL_SC, c0i * arith.index_cast(T.index, R >> 5), 1, _TC * (R >> 5))

        _pl_g = tid & 1
        _pl_j = tid >> 1

        for half in range_constexpr(2):
            _row0 = half * _TR  # Python int (0 or 64): local row offset within the SRD

            # ---- coalesced tile load -> lds.buf (this half's 64 rows) ----
            for chunk in range_constexpr(_NLOAD):
                tw = chunk * (BLK * 4) + tid * 4
                tr = tw // _TCW
                wc = tw % _TCW
                goff = (_row0 + tr) * (C >> 1) + c0w + wc
                vec = buffer_ops.buffer_load(rsrc, goff, vec_width=4, dtype=T.i32)
                _lds_store_vec4(lds.buf.ptr, tw, vec)
            fx.barrier()

            # ---- row phase: byte-for-byte the champion's non-2D branch ----
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
                rwords, rbiased = _finish_microblock(rbits, row_rht, fx.Int32(1 << 21))
                grow = _row0 + r_row  # local row (0..127) within the 128-row SRD window
                gcmb = cblk * _RMBC + cmb
                ob = grow * (C >> 3) + gcmb * 4
                sc = grow * (C >> 5) + gcmb
                _store_words_vec4(orsrc, ob, rwords)
                buffer_ops.buffer_store(arith.trunci(T.i8, rbiased & 0xFF), rscrsrc, sc)

            # ---- col phase: champion's GW=2 compute, park cwords in LDS `co` ----
            for _p in range_constexpr(_TC // (BLK // 2)):  # _NPASS, GW=2
                c_col = _pl_j + _p * (BLK // 2)
                bhalf = c_col & 1
                cw = c_col >> 1
                row0 = _pl_g * 32
                cbits = []
                for row in range_constexpr(32):
                    word = _lds_load1(lds.buf.ptr, (row0 + row) * _TCW + cw)
                    fb = arith.select(bhalf != 0, word & fx.Int32(-65536), word << 16)
                    cbits.append(fb)
                cwords, cbiased = _finish_microblock(cbits, col_rht, fx.Int32(1 << 21))
                composite_mmb = half * 2 + _pl_g  # 0..3 across both halves (GW=2)
                gmmb = rblk_fused * _P2_CMMB + composite_mmb  # global microblock (spans full R)
                csoff = c_col * (R >> 5) + gmmb
                buffer_ops.buffer_store(arith.trunci(T.i8, cbiased & 0xFF), cscrsrc, csoff)
                co_idx = c_col * (_P2_CMMB * 4) + composite_mmb * 4
                _lds_store_vec4(lds.co.ptr, co_idx, Vec.from_elements(list(cwords), fx.Int32))

            if const_expr(half == 0):
                # half0's col-phase reads of lds.buf MUST finish before half1's
                # tile-load overwrites it (cross-wave write-then-read hazard;
                # lds_optimization.md's LDS Op Async Model rule 2).
                fx.barrier()

        # half1's `co` writes must be visible before the combined read below.
        fx.barrier()

        # ---- final phase: GW=4 combined store, one real 64 B run per column ----
        pf_g = tid & 3
        pf_j = tid >> 2
        for _fp in range_constexpr(_TC // (BLK // 4)):
            col = pf_j + _fp * (BLK // 4)
            gmmb = rblk_fused * _P2_CMMB + pf_g
            cob = col * (R >> 3) + gmmb * 4
            co_idx = col * (_P2_CMMB * 4) + pf_g * 4
            v4 = _lds_load_vec4(lds.co.ptr, co_idx)
            cwords = [v4[j] for j in range_constexpr(4)]
            _store_words_vec4(corsrc, cob, cwords)

    return _p2c_kernel


def _build_p2c_launch(row_rht, col_rht, col_locality=False, xcd_remap=False):
    kern = _build_p2c_kernel(row_rht, col_rht, col_locality, xcd_remap)

    @flyc.jit
    def _p2c_launch(X, ROW_OUT, ROW_SC, COL_OUT, COL_SC, R, C, SR_SEED, SCALE_ROUNDING_BIAS, grid_x, stream):
        # SR_SEED is accepted-but-unused so this launch wrapper's call signature
        # is IDENTICAL to `_dual_launch`'s: `flydsl_dual_quant` has one call site
        # for whichever kernel `get_dual_cast` picked, and it always passes
        # `sr_seed` positionally (0 whenever P2c is eligible, since `_p2c_eligible`
        # requires `not row_sr and not col_sr`). P2c never needs SR.
        kern(X, ROW_OUT, ROW_SC, COL_OUT, COL_SC, R, C).launch(
            grid=(grid_x, 1, 1), block=(BLK, 1, 1), stream=stream
        )

    return _p2c_launch


_P2C_LAUNCH = {}
_P2C_COMPILED = {}

# Measured (paired REF/CAND/REF, N=9 reps x 7 GPUs, r6): P2c wins decisively
# where its halved-but-fatter grid (`(R//128)*(C//256)` blocks, vs the
# champion's `(R//64)*(C//256)`) is still big enough to hide the LDS-driven
# occupancy drop (48 KB/block -> 3 WGs/CU, vs the champion's 32 KB -> 5
# WGs/CU); it regresses where the grid is too small to hide that. All 10 real
# recipe-A dispatches were measured -- `_ORDER_TABLE` has exactly 10 entries
# with `_A_RECIPE`, so this is a complete sweep, not a partial one:
#
#   shape (R,C)      grid_x(P2c)   paired delta        verdict
#   (8192, 3072)          768      +3.0% / +3.1%       REGRESS -> champion
#   (16384, 3072)        1536      +1.8% / +2.5-3.4%   REGRESS -> champion
#   (3072, 16384)        1536      +6.7%               REGRESS -> champion
#   (8192, 9216)         2304      +5.3%               REGRESS -> champion
#   (3072, 8192)          768      +3.1%               REGRESS -> champion
#   (8192, 12288)        3072      -0.8% (x2 waves)    WIN -> P2c
#   (12288, 8192)        3072      -1.7%               WIN -> P2c
#   (16384, 15360)       7680      -3.9% / -8.2%       WIN -> P2c
#   (16384, 21504)      10752      -7.5% / -11.3%      WIN -> P2c
#   (21504, 16384)      10752      -3.3% / -3.5%       WIN -> P2c
#
# Any (R, C, row_rht, col_rht) not in this dict (every recipe-B dispatch, every
# regressing shape above, and any future/unlisted shape) falls back to the
# unchanged champion kernel -- never worse than shipped, same fallback
# discipline as `_ORDER_TABLE`.
_P2C_ELIGIBLE = {
    (8192, 12288) + _A_RECIPE[:2]: True,
    (12288, 8192) + _A_RECIPE[:2]: True,
    (16384, 15360) + _A_RECIPE[:2]: True,
    (16384, 21504) + _A_RECIPE[:2]: True,
    (21504, 16384) + _A_RECIPE[:2]: True,
}


def _static_layout(*tensors):
    """Wrap the tensor warm-up args as static-layout memrefs for the one-time
    ``flyc.compile`` call in each ``get_*_cast``. ``flyc.from_torch_tensor``
    builds a ``dynamic_layout=False`` ``TorchTensorJitArg``; confirmed against
    the live ``jit_argument.py`` (``TorchTensorJitArg.__c_abi_spec__``): when
    ``is_layout_dynamic`` is False the returned ABI slot list is just
    ``[(c_void_p, ptr_fill)]`` (``ptr_fill`` does ``s.value =
    t.data_ptr()``) -- the second slot (a generated ``fill`` that reads
    ``t.shape``/``t.stride()`` and ``codec.pack_into``s them every call) is
    only appended when ``is_layout_dynamic`` is True, i.e. only for a bare
    (unwrapped) tensor arg. One fewer ABI fill per tensor gets unrolled into
    the compiled dispatch closure -- pure call overhead, identical GPU binary.

    Safe here because none of this kernel's 5 tensor params are ever read for
    shape/stride: every address is one of `_srd()`'s SRDs, built from `R`/`C`
    (explicit `fx.Int32` scalars) plus `buffer_ops.extract_base_index(t)`,
    which lowers to `memref.extract_aligned_pointer_as_index` -- a bare
    base-pointer read that does not consult the memref's static-vs-dynamic
    layout at all (confirmed against FlyDSL's own `buffer_ops.py`). Only the
    ONE warm-up compile call needs this wrap; every real per-call launch in
    `flydsl_dual_quant`/`flydsl_dual_quant_batched` keeps passing plain
    tensors unchanged -- `TorchTensorJitArg.__c_abi_spec__`'s `ptr_fill` does
    `t = a.torch_tensor if hasattr(a, "torch_tensor") else a`, so a bare
    tensor is accepted at call time whether or not the compiled slot is
    static. The production dispatcher (`quantization_impl.py`) already
    asserts `x.is_contiguous()` before either function is reached, so the
    baked-in static stride always matches the real call-time tensor. Same
    pattern as `gemm_fp8_kernel.py`'s shipped `_static_layout`.

    Provenance: proposed in campaign 20260919_081500 r3 (host-cost analysis),
    prototyped r6 (isolated probes: -1.3..-2.3 us/call, poisoned bit-exact on
    dispatches covering both P2c-eligible and recipe-B/2D-block-amax shapes)
    and r7 (stacked with P3/P4 below), but neither round's edit survived to a
    committed keep -- r6's own official re-verification read flat/slightly
    negative (kept_ratio 0.9988) despite the round's self-reported -0.85%, and
    r7's session crashed before any official read. r9 re-applies this fresh
    and re-measures end to end on the real op rather than trusting either
    prior self-report; see this round's ROUND_REPORT for the actual number."""
    return tuple(flyc.from_torch_tensor(t) for t in tensors)


def _p2c_eligible(R, C, row_rht, col_rht, row_2d, col_2d, row_sr, col_sr):
    """True iff this dispatch should route through the P2c kernel instead of the
    champion `_build_dual_kernel`. P2c only implements the non-2D, non-padded,
    non-SR geometry (recipe A's), and only the 5 shapes measured to win above --
    everything else must fall back (see the table above)."""
    if row_2d or col_2d or row_sr or col_sr:
        return False
    return _P2C_ELIGIBLE.get((int(R), int(C), bool(row_rht), bool(col_rht)), False)


def _pick_p2c_order(R, C, row_rht, col_rht):
    """Reuse `_ORDER_TABLE` (this file's own oracle) for (R, C, row_rht, col_rht,
    row_2d=False, col_2d=False) -- the only geometry P2c implements. The table's
    `xcd_remap` bit was guarded against the CHAMPION's block count
    `(R//_TR)*ncblk`; P2c's block covers 2*_TR rows, so its own block count is
    HALF that and can have different low bits (a count that is a multiple of 8
    need not have its half also be a multiple of 8). Re-derive the guard
    against P2c's own count instead of trusting the champion's already-applied
    bit -- goal.md Sec 5's "keep that guard in anything new" applies verbatim."""
    key = (int(R), int(C), bool(row_rht), bool(col_rht), False, False)
    hit = _ORDER_TABLE.get(key)
    if hit is None:
        return int(C) > int(R), False  # same fallback heuristic as the champion
    col_locality, xcd_remap = hit
    if xcd_remap:
        nrblk_fused = int(R) // (2 * _TR)
        ncblk = int(C) // _TC
        if (nrblk_fused * ncblk) % 8 != 0:
            xcd_remap = False
    return col_locality, xcd_remap


def get_p2c_cast(R, C, row_rht, col_rht):
    """Return (compiled_fn, grid_x) for the P2c fused dual at (R, C, row_rht,
    col_rht). Caller (`get_dual_cast`) must have already checked
    `_p2c_eligible`. Same R%128==0/C%256==0 alignment requirement as
    `dual_eligible` -- P2c introduces no new constraint."""
    col_locality, xcd_remap = _pick_p2c_order(R, C, row_rht, col_rht)
    lk = (bool(row_rht), bool(col_rht), bool(col_locality), bool(xcd_remap))
    raw = _P2C_LAUNCH.get(lk)
    if raw is None:
        raw = _build_p2c_launch(bool(row_rht), bool(col_rht), bool(col_locality), bool(xcd_remap))
        _P2C_LAUNCH[lk] = raw
    key = (int(R), int(C)) + lk
    ent = _P2C_COMPILED.get(key)
    if ent is None:
        import torch

        x = torch.zeros((R, C // 2), dtype=torch.int32, device="cuda")
        ro = torch.zeros((R, C // 8), dtype=torch.int32, device="cuda")
        rs = torch.zeros((R, C // 32), dtype=torch.uint8, device="cuda")
        co = torch.zeros((C, R // 8), dtype=torch.int32, device="cuda")
        cs = torch.zeros((C, R // 32), dtype=torch.uint8, device="cuda")
        grid_x = (R // (2 * _TR)) * (C // _TC)
        stream = torch.cuda.current_stream()
        fn = flyc.compile(raw, *_static_layout(x, ro, rs, co, cs), R, C, 0, 1 << 21, grid_x, stream)
        ent = (fn, grid_x)
        _P2C_COMPILED[key] = ent
    return ent


# ---- Row-only H16 + mxfp4 cast (this campaign's scored deliverable) --------
# First prototyped as an ANALYZE r1 probe and reverted at the end of THAT probe;
# re-landed for real starting at the round that added `flydsl_quant_mxfp4_h16`
# below, and iterated on since (see that function's docstring / goal.md O1-O2/O7).
# This block is live, shipped code -- not a leftover prototype -- even though
# `MXFP4Linear` does not call `flydsl_quant_mxfp4_h16` yet (wiring it in is an
# explicitly out-of-scope follow-up task per goal.md; see this round's
# ROUND_REPORT / campaign-lessons.md-style note before deleting this as "dead").
#
# MXFP4Linear consumes only the ROW pack; the fused dual additionally computes
# and stores the colwise-transpose pack, which is half this kernel's output
# bytes and all of its strided stores.
#
# Dropping it makes the cast a flat stream: microblock mb owns X words
# [16mb, 16mb+16), ROW_OUT words [4mb, 4mb+4) and ROW_SC byte mb -- exact for
# every C % 32 == 0, because rows are contiguous and C/32 microblocks tile a row
# with no remainder. So no LDS, no barrier and no integer division, and every
# access is wave-contiguous (4 KB load / 1 KB + 64 B store per wave).
def _srd_at(t, elem_off, elem_bytes, nrec_bytes):
    base = arith.index_cast(T.i64, buffer_ops.extract_base_index(t))
    boff = arith.index_cast(T.i64, arith.index_cast(T.index, elem_off) * arith.index(elem_bytes))
    raw = arith._to_raw(base + boff)
    r = rocdl.readfirstlane(res=raw.type, src=raw)
    base_v = r.result if hasattr(r, "result") else r
    nr = arith.minui(arith.index_cast(T.index, nrec_bytes), arith.index(0x7FFFFFFF))
    return buffer_ops.create_buffer_resource_from_addr(base_v, num_records_bytes=nr)


_RQ_BLK = 128  # row-only flat kernel block size (independent of the dual's BLK)


def _build_rowq_kernel(row_rht):
    @flyc.kernel(known_block_size=[_RQ_BLK, 1, 1])
    def _rowq_kernel(
        X: fx.Tensor,  # int32 view [R, C/2]
        ROW_OUT: fx.Tensor,  # int32 view [R, C/8]
        ROW_SC: fx.Tensor,  # uint8 [R, C/32]
        SCALE_ROUNDING_BIAS: fx.Int32,
    ):
        tid = fx.thread_idx.x
        mb0 = arith.index_cast(T.index, fx.block_idx.x * _RQ_BLK)
        xsrc = _srd_at(X, mb0 * arith.index(16), 4, fx.Int32(_RQ_BLK * 16 * 4))
        osrc = _srd_at(ROW_OUT, mb0 * arith.index(4), 4, fx.Int32(_RQ_BLK * 4 * 4))
        ssrc = _srd_at(ROW_SC, mb0, 1, fx.Int32(_RQ_BLK))
        rbits = []
        for q in range_constexpr(4):
            v4 = buffer_ops.buffer_load(xsrc, tid * 16 + q * 4, vec_width=4, dtype=T.i32)
            for j in range_constexpr(4):
                word = v4[j]
                rbits.append(word << 16)
                rbits.append(word & 0xFFFF0000)
        words, biased = _finish_microblock(rbits, row_rht, SCALE_ROUNDING_BIAS)
        buffer_ops.buffer_store(Vec.from_elements(list(words), fx.Int32), osrc, tid * 4, cache_modifier=2)
        buffer_ops.buffer_store(arith.trunci(T.i8, biased & 0xFF), ssrc, tid)

    return _rowq_kernel


def _build_rowq_launch(row_rht):
    kern = _build_rowq_kernel(row_rht)

    @flyc.jit
    def _rowq_launch(X, ROW_OUT, ROW_SC, BIAS: fx.Int32, grid_x: fx.Int32, stream: fx.Stream):
        kern(X, ROW_OUT, ROW_SC, BIAS).launch(grid=(grid_x, 1, 1), block=(_RQ_BLK, 1, 1), stream=stream)

    return _rowq_launch


_ROWQ_LAUNCH = {}
_ROWQ_COMPILED = {}
_ROWQ_PLAN = {}


def rowq_eligible(R, C):
    return (int(C) % 32 == 0) and ((int(R) * int(C) // 32) % _RQ_BLK == 0)


def get_rowq_cast(R, C, row_rht):
    raw = _ROWQ_LAUNCH.get(bool(row_rht))
    if raw is None:
        raw = _build_rowq_launch(bool(row_rht))
        _ROWQ_LAUNCH[bool(row_rht)] = raw
    key = (int(R), int(C), bool(row_rht))
    ent = _ROWQ_COMPILED.get(key)
    if ent is None:
        import torch

        x = torch.zeros((R, C // 2), dtype=torch.int32, device="cuda")
        ro = torch.zeros((R, C // 8), dtype=torch.int32, device="cuda")
        rs = torch.zeros((R, C // 32), dtype=torch.uint8, device="cuda")
        grid_x = (R * C // 32) // _RQ_BLK
        fn = flyc.compile(raw, x, ro, rs, 1 << 21, grid_x, torch.cuda.current_stream())
        ent = (fn, grid_x)
        _ROWQ_COMPILED[key] = ent
    return ent


def _make_rowq_plan(R, C, row_rht, fp4_dtype, scale_rounding_mode=0):
    """Same generative-closure shape as ``_make_plan``: hoist the shapes, the
    compiled fn, the grid and the raw-stream lookup out of the per-call path.

    OPTIMIZE r4 (O7): ``ro``/``rs`` are allocated directly in their FINAL
    caller-facing dtype (``fp4_dtype``, itemsize 1 byte -- 2 packed fp4 nibbles
    per byte -- and ``float8_e8m0fnu``, itemsize 1 byte) instead of
    (``int32``, ``uint8``) + a post-call ``.view().view()`` chain, and ``x_bf16``
    is passed to ``fn`` un-viewed. This is safe because
    ``TorchTensorJitArg.__c_abi_spec__``'s per-call ``ptr_fill`` only ever reads
    ``t.data_ptr()`` -- dtype/``element_bits`` is fixed once at ``flyc.compile``
    time from ``get_rowq_cast``'s int32/uint8 warmup tensors (untouched here)
    and never re-checked against the tensor passed to a live call. The kernel
    body itself only ever addresses memory through ``_srd_at``'s explicit
    ``R``/``C``-derived byte offsets (never the memref's own reported
    shape/stride -- same fact ``_static_layout``'s docstring already
    establishes for the dual kernel's warmup wrap), so a differently-dtyped
    but same-byte-count tensor is address-identical. ``ro_shape`` in
    ``fp4_dtype`` elements is ``C // 2`` (itemsize 1) instead of ``C // 8``
    (int32 itemsize 4) for the exact same ``R * (C // 8) * 4`` total bytes.
    Removes 3 of the wrapper's 4 per-call ``.view()`` calls plus the caller's
    input-side view -- 0 left in the hot path. Measured (this round, isolated,
    interleaved, bit-exact vs the pre-change wrapper and vs
    ``flydsl_dual_quant``): +14.3% at 3072x3072 (host overhead is ~1/7 of that
    shape's ~9-10us GPU time), +1.9% at 8192x3072, -0.6% (noise) at
    16384x21504 where GPU time (~150us) dwarfs any host cost. See this round's
    ROUND_REPORT for the official-ruler confirmation."""
    import torch

    fn, grid_x = get_rowq_cast(R, C, row_rht)
    ro_shape = (R, C // 2)  # fp4_dtype elements (1 B each) == C//8 i32 words (4 B each)
    rs_shape = (R, C // 32)
    bias = _mxfp4_scale_rounding_bias(scale_rounding_mode)
    e8m0 = torch.float8_e8m0fnu
    raw_stream = torch._C._cuda_getCurrentRawStream

    def _plan(x_bf16):
        ro = x_bf16.new_empty(ro_shape, dtype=fp4_dtype)
        rs = x_bf16.new_empty(rs_shape, dtype=e8m0)
        fn(x_bf16, ro, rs, bias, grid_x, raw_stream(x_bf16.device.index))
        return ro, rs

    return _plan


def flydsl_quant_mxfp4_h16(x_bf16, fp4_dtype, scale_rounding_mode=0):
    """Rowwise mxfp4 cast with the deterministic in-kernel H16 (``_rht16``), i.e.
    exactly what ``MXFP4Linear`` needs: bit-identical to
    ``flydsl_dual_quant(x, fp4_dtype, True, False)[:2]`` without computing or
    storing the discarded colwise pack."""
    R, C = x_bf16.shape
    if not rowq_eligible(R, C):
        row, scale, _, _ = flydsl_dual_quant(x_bf16, fp4_dtype, True, False)
        return row, scale
    key = (int(R), int(C), True, fp4_dtype, int(scale_rounding_mode))
    plan = _ROWQ_PLAN.get(key)
    if plan is None:
        plan = _make_rowq_plan(R, C, True, fp4_dtype, scale_rounding_mode)
        _ROWQ_PLAN[key] = plan
    return plan(x_bf16)


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


_PLAN = {}


def _make_plan(R, C, row_rht, col_rht, row_2d, col_2d, row_sr, col_sr, scale_rounding_mode=0):
    """Build a per-key generative closure for one (R, C, recipe) dispatch of
    ``flydsl_dual_quant``. Called ONCE per unique key (on a ``_PLAN`` miss); the
    returned closure is cached and reused for every future call with that key.

    This cache sits STRICTLY ABOVE ``get_dual_cast`` -- it calls it exactly once
    per key and stores whatever ``(fn, grid_x)`` that call returns. It never
    re-implements ``_p2c_eligible``/``_pick_block_order``/``_pick_p2c_order``:
    ``get_dual_cast`` already owns that routing (including its own delegation to
    ``get_p2c_cast`` when ``_P2C_ELIGIBLE``), so a future change to the routing
    tables is picked up automatically on the next cold key, and every key already
    cached here keeps the routing decision ``get_dual_cast`` made the first time
    that key was seen -- correct because ``_ORDER_TABLE``/``_P2C_ELIGIBLE`` are
    static dicts built once at import time and never mutated at runtime, so
    ``get_dual_cast`` is a pure function of this key.

    The closure bakes in the 4 output shape tuples and dtypes (recomputed from
    scratch on every call in the un-cached path) plus the
    ``torch._C._cuda_getCurrentRawStream`` lookup, so a ``_PLAN`` hit inside
    ``flydsl_dual_quant`` costs one dict lookup + 4 allocations + 1 dispatch,
    instead of also re-deriving the P2C-eligibility check, the block-order
    oracle lookup, and the ``_DUAL_LAUNCH``/``_DUAL_COMPILED`` dict lookups that
    an uncached call to ``get_dual_cast`` performs every time.

    Deliberately does NOT bake in ``device``: unlike the shape/dtype/fn/grid_x
    values (pure functions of the key), device is a property of the actual
    call-time tensor, not the key. The returned closure takes ``x_bf16`` and
    allocates with ``x_bf16.new_empty(...)``, which reads device/layout off that
    argument on every call, so a process with tensors on more than one device
    still lands each output on the right device even though the compiled-fn /
    grid_x / shape plan is shared across devices for the same (R, C, recipe).

    ``sr_seed`` is NOT hoisted (only the ``need_sr`` boolean predicate is) --
    ``_next_sr_seed()`` mutates a global counter and must advance once per real
    launch for the stochastic-rounding seed sequence to stay decorrelated; only
    the branch condition (a pure function of the key) is safe to precompute.

    Key = (R, C, row_rht, col_rht, row_2d, col_2d, row_sr, col_sr) -- pure shape
    + compile-time-flag values, no tensor identity -- bucket K1 (kernel-internal
    per-shape compile cache), same bucket as ``_DUAL_COMPILED``; Rule 11's
    id(Q/K/V) ban does not apply.

    Provenance: proposed campaign 20260919_081500 r3, prototyped r6, landed in
    an uncommitted r7 that crashed before an official re-verification bench.
    r9 re-applies fresh; see this round's ROUND_REPORT for the measured number."""
    import torch

    fn, grid_x = get_dual_cast(R, C, row_rht, col_rht, row_2d, col_2d, row_sr, col_sr)
    ro_shape = (R, C // 8)
    rs_shape = (R, C // 32)
    co_shape = (C, R // 8)
    cs_shape = (C, R // 32)
    need_sr = bool(row_sr or col_sr)
    i32 = torch.int32
    u8 = torch.uint8
    raw_stream = torch._C._cuda_getCurrentRawStream

    def _plan(x_bf16):
        x_i32 = x_bf16.view(i32)  # [R, C/2]
        # new_empty (vs torch.empty(..., device=dev)) inherits device/layout off
        # x_bf16 directly instead of parsing a device= kwarg. Measured campaign
        # 20260919_081500 r3: -0.91 us per 4-alloc batch.
        ro = x_bf16.new_empty(ro_shape, dtype=i32)
        rs = x_bf16.new_empty(rs_shape, dtype=u8)
        co = x_bf16.new_empty(co_shape, dtype=i32)
        cs = x_bf16.new_empty(cs_shape, dtype=u8)
        sr_seed = _next_sr_seed() if need_sr else 0
        # Raw-int stream: same rationale as the r5 landing (isinstance(raw, int)
        # is fx.Stream's fast-path fill; _cuda_getCurrentRawStream(idx) is the
        # primitive current_stream(idx) itself wraps, so semantics -- including
        # `with torch.cuda.stream(s):` -- are unchanged).
        bias = _mxfp4_scale_rounding_bias(scale_rounding_mode)
        fn(x_i32, ro, rs, co, cs, R, C, sr_seed, bias, grid_x, raw_stream(x_bf16.device.index))
        return ro, rs, co, cs

    return _plan


def flydsl_dual_quant(
    x_bf16,
    fp4_dtype,
    row_rht,
    col_rht,
    row_2d=False,
    col_2d=False,
    row_sr=False,
    col_sr=False,
    scale_rounding_mode=0,
):
    """Fused rowwise + colwise-transpose mxfp4 cast (one bf16 read). Returns
    (row_data, row_scale, col_data, col_scale) in C++-compatible dtypes/shapes.
    ``row_sr``/``col_sr`` request stochastic rounding on that direction."""
    import torch

    R, C = x_bf16.shape
    key = (
        int(R),
        int(C),
        bool(row_rht),
        bool(col_rht),
        bool(row_2d),
        bool(col_2d),
        bool(row_sr),
        bool(col_sr),
        int(scale_rounding_mode),
    )
    plan = _PLAN.get(key)
    if plan is None:
        plan = _make_plan(R, C, row_rht, col_rht, row_2d, col_2d, row_sr, col_sr, scale_rounding_mode)
        _PLAN[key] = plan
    ro, rs, co, cs = plan(x_bf16)
    row_data = ro.view(torch.uint8).view(fp4_dtype)  # [R, C/2] fp4
    col_data = co.view(torch.uint8).view(fp4_dtype)  # [C, R/2] fp4
    row_scale = rs.view(torch.float8_e8m0fnu)
    col_scale = cs.view(torch.float8_e8m0fnu)
    return row_data, row_scale, col_data, col_scale


def get_dual_cast(R, C, row_rht, col_rht, row_2d=False, col_2d=False, row_sr=False, col_sr=False):
    """Return (compiled_fn, grid_x) for the fused dual at
    (R, C, row_rht, col_rht, row_2d, col_2d, row_sr, col_sr).
    Requires R % 128 == 0 and C % 256 == 0 (no scale/output padding)."""
    if _p2c_eligible(R, C, row_rht, col_rht, row_2d, col_2d, row_sr, col_sr):
        # P2c (r6): measured win on this exact (R, C, recipe) -- see
        # `_P2C_ELIGIBLE`'s comment. Note the P2c launch signature has no
        # SR_SEED slot (P2c never runs with SR requested); `flydsl_dual_quant`
        # always passes `sr_seed=0` in that case, so dropping the arg here is
        # transparent to every caller.
        return get_p2c_cast(R, C, row_rht, col_rht)
    col_locality, xcd_remap = _pick_block_order(R, C, row_rht, col_rht, row_2d, col_2d)
    lk = (
        bool(row_rht),
        bool(col_rht),
        bool(row_2d),
        bool(col_2d),
        col_locality,
        bool(row_sr),
        bool(col_sr),
        xcd_remap,
    )
    raw = _DUAL_LAUNCH.get(lk)
    if raw is None:
        raw = _build_dual_launch(
            bool(row_rht),
            bool(col_rht),
            bool(row_2d),
            bool(col_2d),
            col_locality,
            bool(row_sr),
            bool(col_sr),
            xcd_remap,
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
        fn = flyc.compile(raw, *_static_layout(x, ro, rs, co, cs), R, C, 0, 1 << 21, grid_x, stream)
        ent = (fn, grid_x)
        _DUAL_COMPILED[key] = ent
    return ent


# ---- Batched-3D dual quant: [G,N,K] weight, all experts in ONE launch (G x the
# blocks -> fills the GPU even for small per-expert N, where the 2D dense kernel is
# occupancy-starved and drops to ~2.5 TB/s). Reuses _emit_dual_body per-tile with
# per-expert base offsets; SRDs cover the whole 3D (gmul=G). ----
def _build_dual3_kernel(
    row_rht,
    col_rht,
    row_2d=False,
    col_2d=False,
    padded=False,
    col_locality=False,
    row_sr=False,
    col_sr=False,
    scale_rounding_bias=1 << 21,
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
            fx.Int32(scale_rounding_bias),
            # per-expert element bases in index (64-bit): g * per_expert_elems overflows
            # int32 for large-G MoE (e.g. G=64: 63 * N*K/2 > 2^31); _emit_dual_body folds
            # these into per-expert int64 SRD bases.
            gx=arith.index_cast(T.index, g) * arith.index_cast(T.index, R * (C >> 1)),
            gro=arith.index_cast(T.index, g) * arith.index_cast(T.index, R * (cpad >> 3)),
            grsc=arith.index_cast(T.index, g) * arith.index_cast(T.index, R * (cpad >> 5)),
            gco=arith.index_cast(T.index, g) * arith.index_cast(T.index, C * (rpad >> 3)),
            gcsc=arith.index_cast(T.index, g) * arith.index_cast(T.index, C * (rpad >> 5)),
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
        )

    return _dual3_kernel


def _build_dual3_launch(
    row_rht,
    col_rht,
    row_2d=False,
    col_2d=False,
    padded=False,
    col_locality=False,
    row_sr=False,
    col_sr=False,
    scale_rounding_bias=1 << 21,
):
    kern = _build_dual3_kernel(
        row_rht,
        col_rht,
        row_2d,
        col_2d,
        padded,
        col_locality,
        row_sr,
        col_sr,
        scale_rounding_bias,
    )

    @flyc.jit
    def _dual3_launch(
        X,
        ROW_OUT,
        ROW_SC,
        COL_OUT,
        COL_SC,
        R,
        C,
        G,
        CP,
        RP,
        SR_SEED,
        grid_x,
        stream,
    ):
        kern(
            X,
            ROW_OUT,
            ROW_SC,
            COL_OUT,
            COL_SC,
            R,
            C,
            G,
            CP,
            RP,
            SR_SEED,
        ).launch(grid=(grid_x, 1, 1), block=(BLK, 1, 1), stream=stream)

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


def get_dual3_cast(
    N,
    K,
    G,
    row_rht,
    col_rht,
    row_2d=False,
    col_2d=False,
    row_sr=False,
    col_sr=False,
    scale_rounding_bias=1 << 21,
):
    """(compiled_fn, grid_x, K_pad, N_pad, padded) for the batched-3D dual at
    (N,K,G,recipes). K_pad=ceil(K/128)*128 (row-out), N_pad=ceil(N/128)*128 (col-out);
    `padded` when K not a 256-tile multiple or N not 128-multiple."""
    Kp = ((K + 127) // 128) * 128
    Np = ((N + 127) // 128) * 128
    tr, tc = _pick_tile_geom(int(N), int(K))
    padded = (K % tc != 0) or (N % 128 != 0)
    col_locality = int(K) > int(N)  # K>N: combine transpose stores (col-out)
    # The optimized batched launcher is not stable with one more dynamic scalar
    # argument in FlyDSL 0.2.4. Specialize its bias instead; a process normally
    # selects one mode, and even runtime switching creates at most three variants.
    lk = (
        bool(row_rht),
        bool(col_rht),
        bool(row_2d),
        bool(col_2d),
        padded,
        col_locality,
        bool(row_sr),
        bool(col_sr),
        int(scale_rounding_bias),
    )
    lk = lk + (tr, tc)
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
            int(scale_rounding_bias),
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
        rs = torch.empty((G, N, Kp // 32), dtype=torch.uint8, device="cuda")
        co = torch.empty((G, K, Np // 8), dtype=torch.int32, device="cuda")
        cs = torch.empty((G, K, Np // 32), dtype=torch.uint8, device="cuda")
        ncblk = ((K + tc - 1) // tc) if padded else (K // tc)
        grid_x = (N // tr) * ncblk * G
        fn = flyc.compile(
            raw,
            *_static_layout(x, ro, rs, co, cs),
            N,
            K,
            G,
            Kp,
            Np,
            0,
            grid_x,
            torch.cuda.current_stream(),
        )
        ent = (fn, grid_x, Kp, Np, padded)
        _DUAL3_COMPILED[key] = ent
    _set_tile_geom(*_saved)
    return ent


_PLAN3 = {}


def _make_plan3(N, K, G, row_rht, col_rht, row_2d, col_2d, row_sr, col_sr, scale_rounding_mode=0):
    """Batched-3D twin of ``_make_plan`` -- see its docstring for the full
    rationale (cache sits above ``get_dual3_cast`` only, never re-implements its
    padding/routing math; device is never baked in, only shape/dtype/fn/grid_x/
    Kp/Np/padded, all pure functions of this key). Key adds ``G`` (expert count)
    ahead of the 2D key's fields; not on the 14-shape Flux ruler today (dense
    MLP, no MoE), same production dispatch path as ``_make_plan``."""
    import torch

    scale_rounding_bias = _mxfp4_scale_rounding_bias(scale_rounding_mode)
    fn, grid_x, Kp, Np, padded = get_dual3_cast(
        N, K, G, row_rht, col_rht, row_2d, col_2d, row_sr, col_sr, scale_rounding_bias
    )
    ro_shape = (G, N, Kp // 8)
    rs_shape = (G, N, Kp // 32)
    co_shape = (G, K, Np // 8)
    cs_shape = (G, K, Np // 32)
    # Masked loads make the kernel write the complete K_pad row-output tail. Only
    # the col-output N_pad suffix has no producer threads, so allocate without a
    # full-tensor memset and clear that small unwritten suffix explicitly.
    need_kpad = bool(padded and Kp != K)
    need_npad = bool(padded and Np != N)
    k8, k32 = K // 8, K // 32
    n8, n32 = N // 8, N // 32
    need_sr = bool(row_sr or col_sr)
    i32 = torch.int32
    u8 = torch.uint8
    raw_stream = torch._C._cuda_getCurrentRawStream

    def _plan(x3d):
        x_i32 = x3d.contiguous().view(i32)  # [G, N, K/2]
        ro = x_i32.new_empty(ro_shape, dtype=i32)
        rs = x_i32.new_empty(rs_shape, dtype=u8)
        co = x_i32.new_empty(co_shape, dtype=i32)
        cs = x_i32.new_empty(cs_shape, dtype=u8)
        if need_kpad:
            ro[..., k8:].zero_()
            rs[..., k32:].zero_()
        if need_npad:
            co[:, :, n8:].zero_()
            cs[:, :, n32:].zero_()
        sr_seed = _next_sr_seed() if need_sr else 0
        fn(x_i32, ro, rs, co, cs, N, K, G, Kp, Np, sr_seed, grid_x, raw_stream(x_i32.device.index))
        return ro, rs, co, cs

    return _plan


def flydsl_dual_quant_batched(
    x3d,
    fp4_dtype,
    row_rht,
    col_rht,
    row_2d=False,
    col_2d=False,
    row_sr=False,
    col_sr=False,
    scale_rounding_mode=0,
):
    """Batched-3D fused rowwise + colwise-transpose mxfp4 dual cast for a [G,N,K]
    weight in ONE launch. Returns C++-compatible per-expert
    (row_data [G,N,K/2], row_scale [G,N,K/32], col_data [G,K,N/2], col_scale [G,K,N/32]).
    ``row_sr``/``col_sr`` request stochastic rounding on that direction."""
    import torch

    G, N, K = x3d.shape
    key = (
        int(N),
        int(K),
        int(G),
        bool(row_rht),
        bool(col_rht),
        bool(row_2d),
        bool(col_2d),
        bool(row_sr),
        bool(col_sr),
        int(scale_rounding_mode),
    )
    plan = _PLAN3.get(key)
    if plan is None:
        plan = _make_plan3(N, K, G, row_rht, col_rht, row_2d, col_2d, row_sr, col_sr, scale_rounding_mode)
        _PLAN3[key] = plan
    ro, rs, co, cs = plan(x3d)
    return (
        ro.view(torch.uint8).view(fp4_dtype),
        rs.view(torch.float8_e8m0fnu),
        co.view(torch.uint8).view(fp4_dtype),
        cs.view(torch.float8_e8m0fnu),
    )
