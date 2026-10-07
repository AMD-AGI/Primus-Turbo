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

import os

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith, buffer_ops, const_expr, math, range_constexpr, rocdl
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec
from flydsl.expr.utils.arith import _to_raw as _raw

from primus_turbo.flydsl.utils.gemm_helper import xcd_remap_pid, xcd_remap_pid_blocked
from primus_turbo.flydsl.utils.prims import _DPP_QUAD_SWAP1, _dpp_add_f32, _row16_sum_f32, atomic_add

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


def _cvt_microblock_to_fp6(vf, scale_f32):
    """32 f32 Values -> 6 i32 words: a little-endian stream of 32 E2M3 codes. gfx950's
    2xpk16 convert interleaves its sources (code 2i from src0[i], 2i+1 from src1[i])."""
    src = [
        _raw(Vec.from_elements([fx.Float32(_raw(vf[2 * i + h])) for i in range_constexpr(16)], fx.Float32))
        for h in range_constexpr(2)
    ]
    v = Vec(rocdl.cvt_scalef32_2xpk16_fp6_f32(T.vec(6, T.i32), src[0], src[1], _raw(scale_f32)))
    return [v[i] for i in range_constexpr(6)]


def _finish_microblock_fp6(vbits, use_rht, scale_rounding_bias):
    """`_finish_microblock` with an E2M3 convert: (6 fp6 i32 words, scale_e8m0 i8-ready i32)."""
    vf = _microblock_vf(vbits, use_rht, fold_scale=True)
    amax = _microblock_amax_f(vf)
    native_bits, biased = _compute_scale_native(amax, scale_rounding_bias, exp_up=vf_exp_up(use_rht))
    return _cvt_microblock_to_fp6(vf, arith.bitcast(T.f32, native_bits)), biased


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
    skip_row=False,
    legpack=False,
    XPAD=None,
    XOFF=None,
    row_fp6=False,
    row_bias=None,
    XSTR=None,
    gelu=False,
    BIAS=None,
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
    which only ever sets this where the block count is a verified multiple of 8).

    ``skip_row`` (optimize round 9 / N2): when True, the ROW phase is never
    traced and ``ROW_OUT``/``ROW_SC`` are never touched (may be ``None``) --
    used by the col-only transpose-read kernel (``_build_colq_kernel``) for a
    caller that only wants the col pack (e.g. weight's Dgrad operand, whose
    row pack Fprop already gets separately from the cheap flat
    ``_rowq_kernel``, so computing a row half here too would be wasted work).
    Every existing caller passes ``skip_row=False`` (the default), so this is
    a pure addition: the traced IR for every current caller is unchanged.

    ``legpack`` (optimize round 10 / P6): when True, this call packs a COLUMN
    SLICE ``X[:, XOFF:XOFF+C]`` of a wider ``[R, XPAD]`` BF16 tensor into the
    matching column slice of a SHARED ``[R, XPAD/8]``-row-out /
    ``[XPAD, R/8]``-col-out pair -- ``C`` keeps meaning THIS call's own local
    leg width (grid sizing / ``ncblk`` / every row- and col-phase LOCAL loop
    bound are all unchanged), while ``XPAD`` is the wider buffer's true
    per-row stride and ``XOFF`` is this leg's starting column within it. Two
    (or more) legpack calls at increasing ``XOFF`` therefore build up exactly
    what one monolithic ``C=XPAD`` dual call would have, without a
    concatenation step on either side. Both ``XPAD`` and ``XOFF`` MUST be
    traced ``fx.Int32`` kernel arguments (never plain Python ints) -- the
    same requirement ``CP``/``RP`` already have, for the same reason: a raw
    Python int reaching ``arith.index_cast`` on this (non-batched) SRD path
    raises ``AttributeError: 'int' object has no attribute '_CAPIPtr'``.

    Independent of ``padded``/``CP``/``RP`` (the pre-existing OOB-tail-mask
    for a single non-tile-aligned dispatch): every leg width this campaign
    uses (3072/9216/12288) is already a multiple of ``_TC``, so the
    ``padded`` masking selects would be pure overhead here and are
    deliberately left off (``legpack`` calls always pass ``padded=False``).
    The column offset folds into the COL_OUT SRD's *base* (via ``c0i``), not
    into a post-hoc store offset like ``gco`` -- a leg offset is typically
    far larger than one tile's ``num_records`` bound, so adding it after the
    SRD is built would be silently dropped as OOB instead of landing in the
    right slice.

    ``row_fp6`` makes the row half the A6W4 activation operand instead: E2M3 codes with
    ``row_bias`` as its scale rounding bias, stored FP8-padded (24 code bytes + 8 zero bytes
    per micro-block, ROW_OUT a [R, C] byte tensor) with ROW_SC in shuffle_scale_w4 order.
    The col half is unchanged. Only the plain 2D path (legpack included) supports it.

    ``XSTR`` (legpack only) makes X this leg's own tensor with row stride ``XSTR``, rather than
    the shared ``[R, XPAD]`` tensor read at column ``XOFF``. ``gelu`` applies `_gelu1_torch` to
    the loaded bf16 and rounds back to bf16 before the LDS store, so both halves see the bytes a
    materialised torch ``gelu(x, approximate="tanh")`` would have held. ``BIAS`` (with ``gelu``)
    is this leg's bf16 bias, added in f32 before the GELU: Inductor's fused ``gelu(x + bias)``."""
    if row_fp6 and (batched or padded or row_2d or row_sr or skip_row):
        raise ValueError("row_fp6 supports only the plain, unpadded, non-SR 2D dual")
    if (XSTR is not None or gelu) and not legpack:
        raise ValueError("XSTR and gelu are legpack-only")
    if BIAS is not None and not gelu:
        raise ValueError("BIAS is only added ahead of the GELU")
    if ncblk is None:
        ncblk = C // _TC
    xpad = XPAD if legpack else C  # X's true per-row stride (elements)
    xstr = xpad if XSTR is None else XSTR
    cpad = CP if padded else (xpad if legpack else C)  # row-out column extent (K_pad)
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
        if not skip_row:
            orsrc = _srd(ROW_OUT, gro, 4, R * (cpad >> 3) * 4)
            rscrsrc = _srd(ROW_SC, grsc, 1, R * (cpad >> 5))
        corsrc = _srd(COL_OUT, gco, 4, C * (rpad >> 3) * 4)
        cscrsrc = _srd(COL_SC, gcsc, 1, C * (rpad >> 5))
        gx = gro = grsc = gco = gcsc = 0  # expert bases folded into the SRDs above
    else:
        r0i = arith.index_cast(T.index, r0)
        _colbase = cblk * _TC
        if legpack:
            # COL_OUT is indexed by ORIGINAL column (its own [C_total, R/8]
            # "row" axis), so this leg's global start must move the SRD's
            # BASE, not a post-hoc store offset (see the docstring above).
            _colbase = _colbase + XOFF
        c0i = arith.index_cast(T.index, _colbase)
        rsrc = _srd(X, r0i * arith.index_cast(T.index, xstr >> 1), 4, _TR * (xstr >> 1) * 4)
        if legpack:
            # X's own column start (word units) and ROW_OUT's matching start
            # (word/byte units for data/scale). Plain overwrites, not adds:
            # `gx`/`gro`/`grsc` are always the unused 0 default on every
            # legpack call (legpack is never combined with the batched
            # per-expert offset mechanism these three otherwise serve).
            gx = fx.Int32(0) if XSTR is not None else XOFF >> 1
            gro = XOFF >> 3
            grsc = XOFF >> 5
        if row_fp6:
            orsrc = _srd(ROW_OUT, r0i * arith.index_cast(T.index, xpad >> 2), 4, _TR * xpad)
            rscrsrc = _srd(ROW_SC, arith.index(0), 1, R * (xpad >> 5))
        elif not skip_row:
            orsrc = _srd(ROW_OUT, r0i * arith.index_cast(T.index, cpad >> 3), 4, _TR * (cpad >> 3) * 4)
            rscrsrc = _srd(ROW_SC, r0i * arith.index_cast(T.index, cpad >> 5), 1, _TR * (cpad >> 5))
        corsrc = _srd(COL_OUT, c0i * arith.index_cast(T.index, rpad >> 3), 4, _TC * (rpad >> 3) * 4)
        cscrsrc = _srd(COL_SC, c0i * arith.index_cast(T.index, rpad >> 5), 1, _TC * (rpad >> 5))

    if BIAS is not None:
        bsrc = _srd(BIAS, arith.index(0), 4, (C >> 1) * 4)
    # ---- coalesced tile load -> LDS ----
    for chunk in range_constexpr(_NLOAD):
        tw = chunk * (BLK * 4) + tid * 4
        tr = tw // _TCW
        wc = tw % _TCW
        goff = (_row0 + tr) * (xstr >> 1) + c0w + wc + gx
        if padded:
            # mask cols past real C -> OOB load returns 0 (rows always valid: R%64==0,
            # tile is 64 rows, rblk covers exactly R/64 tiles).
            goff = arith.select((c0w + wc) < (C >> 1), goff, fx.Int32(_OOB))
        vec = buffer_ops.buffer_load(rsrc, goff, vec_width=4, dtype=T.i32)
        if const_expr(gelu):
            if BIAS is not None:
                bvec = buffer_ops.buffer_load(bsrc, c0w + wc, vec_width=4, dtype=T.i32)
            words = []
            for q in range_constexpr(4):
                v2 = _bf16_pair_to_f32(vec[q])
                g0, g1 = fx.Float32(v2[0]), fx.Float32(v2[1])
                gelu1 = _gelu1_torch_bf16
                if BIAS is not None:
                    b2 = _bf16_pair_to_f32(bvec[q])
                    g0, g1 = g0 + fx.Float32(b2[0]), g1 + fx.Float32(b2[1])
                    gelu1 = _gelu1_torch
                words.append(rocdl.cvt_pk_bf16_f32(gelu1(g0), gelu1(g1)))
            vec = Vec.from_elements(words, fx.Int32)
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
    # Skipped entirely (not even traced) when skip_row=True -- a plain Python
    # bool resolved at trace time (same pattern as row_2d/col_2d/batched
    # elsewhere in this function), so the col-only kernel's IR never
    # references ROW_OUT/ROW_SC/orsrc/rscrsrc at all.
    if not skip_row:
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
                if row_fp6:
                    rwords, rbiased = _finish_microblock_fp6(rbits, row_rht, row_bias)
                    gcmb = cblk * (_TC // 32) + cmb
                    if legpack:
                        gcmb = gcmb + grsc
                    ow = r_row * (xpad >> 2) + gcmb * 8
                    _store_words_vec4(orsrc, ow, rwords[0:4])
                    _store_words_vec4(orsrc, ow + 4, [rwords[4], rwords[5], fx.Int32(0), fx.Int32(0)])
                    # shuffle_scale_w4: (r, c) -> [r/32, c/8, c%4, r%16, (c/4)%2, (r/16)%2]
                    gr = r0 + r_row
                    so = (
                        ((((gr >> 5) * (xpad >> 8) + (gcmb >> 3)) * 4 + (gcmb & 3)) * 16 + (gr & 15)) * 2
                        + ((gcmb >> 2) & 1)
                    ) * 2 + ((gr >> 4) & 1)
                    buffer_ops.buffer_store(arith.trunci(T.i8, rbiased & 0xFF), rscrsrc, so)
                    continue
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

# ---- H16 fusion recipe: (True, True, False, False) ------------------------
# This is the recipe key both `flydsl_quant_mxfp4_h16_dual` (round 8/N1, X and grad_out)
# and `flydsl_quant_mxfp4_h16_col` (round 9/N2, weight's Dgrad operand)
# ACTUALLY dispatch under -- neither is `_A_RECIPE` or `_B_RECIPE`, so before
# this round every one of their calls fell back to the shipped
# `col_locality = C > R` heuristic with `xcd_remap` off (confirmed round 7/9:
# `p2c_eligible: false`, `xcd_remap: false` on every H16-recipe dispatch).
# `_build_colq_kernel` traces `_emit_dual_body(..., skip_row=True, ...)` with
# `col_rht=True`, and `get_colq_cast` calls `_pick_block_order(R, C, True,
# col_rht, False, False)` -- i.e. it looks up this EXACT SAME key (row_rht is
# hardcoded True there too, even though the row phase is never traced), so
# one set of entries below serves both kernels with no dispatch-code change.
_H16_RECIPE = (True, True, False, False)

# The 7 (R, C) pairs below are every real shape this recipe hits in the 4
# traced Flux layers (qkv 3072->9216, proj 3072->3072, mlp_up 3072->12288,
# mlp_down 12288->3072) at the profiled 8192-token stack (round 7 probe 1/4):
#   dual (X activation + grad_out, `flydsl_quant_mxfp4_h16_dual`):
#     (8192, 3072)   X: qkv, proj, mlp_up   | grad_out: proj, mlp_down (5 call
#                    sites/step -- the highest-multiplicity shape here)
#     (8192, 9216)   grad_out: qkv
#     (8192, 12288)  X: mlp_down            | grad_out: mlp_up
#   colq (weight's Dgrad operand, `flydsl_quant_mxfp4_h16_col`, row phase
#   skipped -- weight's OWN row pack comes from the separate flat
#   `_rowq_kernel`, never from this recipe):
#     (9216, 3072)   weight: qkv
#     (3072, 3072)   weight: proj
#     (12288, 3072)  weight: mlp_up
#     (3072, 12288)  weight: mlp_down
#
# Measured on `/results/r10/order_sweep.py` (this round's own "ord.py"-style
# oracle: one `flyc.compile` per candidate per process,
# `FLYDSL_RUNTIME_ENABLE_CACHE=0`, GPUs 0-6 fanned one shape per card, torch.
# equal-bit-exact-checked against `flydsl_quant_mxfp4_h16` on every candidate
# BEFORE any timing). Two independent passes per shape: an initial 4-way
# sweep (N=36 warm-interleaved + N=9 cold-MALL-flush-interleaved, rotating
# which candidate starts each cycle) then a fresh-process 2-(or 3-)way
# confirmation race restricted to {shipped default, winner} at N=60-80 warm /
# N=15-20 cold. Both passes and both regimes agreed on every shape except
# (8192, 12288) (see its entry below). Gains are `1 - new/shipped` on the
# CONFIRMATION pass, reported as "warm/cold":
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
    # r10 H16-recipe entries (see the block comment above _H16_RECIPE).
    (8192, 3072) + _H16_RECIPE: (False, True),  # loc0+x8 dual X*3+G*2/step +3.6%/+3.4%
    (8192, 9216) + _H16_RECIPE: (True, True),  # loc1+x8 dual G:qkv        +9.1%/+15.2%
    # (8192, 12288): dual X:mlp_down + G:mlp_up. loc1+x8 wins warm (+13.3%)
    # but loc0 wins cold (+15.1%) -- the one shape where the two regimes
    # disagree on WHICH non-default order is better (both clearly beat the
    # shipped loc1 default in both regimes: warm +7.3%/+13.3%, cold +15.1%/
    # +12.6%). Picked loc0 (the cold winner): real training calls this once
    # per huge (8192x12288-element) tensor per step, interleaved with a full
    # model's worth of other kernels between repeats -- MALL-flushed cold is
    # the more representative regime for a shape this large, not the
    # back-to-back-repeat-friendly warm regime the isolated timing loop uses.
    (8192, 12288) + _H16_RECIPE: (False, False),  # loc0    dual (see note)  +7.3%/+15.1%
    (9216, 3072) + _H16_RECIPE: (True, True),  # loc1+x8 colq weight:qkv     +8.2%/+6.4%
    (3072, 3072) + _H16_RECIPE: (True, True),  # loc1+x8 colq weight:proj   +2.6%/+5.0%
    (12288, 3072) + _H16_RECIPE: (True, True),  # loc1+x8 colq weight:mlp_up  +8.6%/+7.2%
    (3072, 12288) + _H16_RECIPE: (False, False),  # loc0  colq weight:mlp_down +11.4%/+16.5%
    # r11/P1 entries (optimize round 4): the three real single-block dual
    # shapes this recipe never had a table row for -- `_pick_block_order` fell
    # back to the shipped heuristic (`col_locality = C > R`, `xcd_remap=False`)
    # on every one of these until now. Raced against the shipped default with
    # the r2-corrected protocol (N=16 warm-interleaved + N=11
    # cold-MALL-flush-interleaved, rotating which candidate starts each
    # cycle, medians -- supersedes r1's min-of-25 sequential race, which
    # systematically overstated all three of these by a further 4-13 points).
    # All 4 candidates per shape re-verified `torch.equal` against
    # `flydsl_quant_mxfp4_h16` (both row and col pack) this round on GPUs 0-6
    # before picking a winner; (1,1) wins both the warm AND the cold regime
    # at all three, so there is no cold/warm tiebreak needed here (contrast
    # the (8192, 12288) entry above, which does need one):
    (16384, 15360) + _H16_RECIPE: (True, True),  # loc1+x8 dual single l2 X      +19.79%/+19.39%
    (16384, 21504) + _H16_RECIPE: (True, True),  # loc1+x8 dual single l1 G      +16.60%/+15.22%
    (16384, 3072) + _H16_RECIPE: (True, True),  # loc1+x8 dual single l1 X/l2 G  +3.18%/+5.33%
    # r10/P6 entries: the two LEG widths the new column-slice legpack dual
    # (`get_dual_legpack_cast`) dispatches per real Flux split -- (16384,12288)
    # from the 15360=3072+12288 split, (16384,9216) from the 21504=9216+12288
    # split. Neither leg width previously had a table row (no monolithic H16
    # call anywhere hits these exact (R,C) pairs -- every production M=16384
    # dim is 15360/3072/21504, so this is a pure addition with zero collateral
    # callers), so `_pick_block_order` was falling back to the shipped
    # heuristic (`C > R`, remap off) -> (False, False) for both, which this
    # round's own race (`/results/r10/probe3_order_race.py`, canary-filtered
    # N=6-11 kept samples per candidate) found is the LOSING order at both:
    (16384, 12288) + _H16_RECIPE: (True, True),  # loc1+x8 legpack leg (15360=3072+12288 split)  +6.19%
    (16384, 9216) + _H16_RECIPE: (True, True),  # loc1+x8 legpack leg (21504=9216+12288 split)   +24.97%
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


def _buffer_atomic_add_f32(rsrc, elem_off, val):
    """Atomic HBM accumulate of one f32 through an SRD -- BUFFER_ATOMIC_ADD_F32
    (``rocdl.raw_ptr_buffer_atomic_fadd``), addressed exactly like this file's
    own ``buffer_ops.buffer_store`` for a scalar f32 (``elem_off`` in
    ELEMENTS, scaled to bytes here to match that call's convention). Same
    V#/MUBUF range-check hardware as every other SRD access in this file: an
    out-of-range offset is a silent no-op, not a fault -- the guarantee the
    P7 fold's odd-lane ``_OOB`` drop already relies on for ``buffer_store``,
    confirmed to hold for this atomic op too on an isolated probe (optimize
    round 10 / P16(a): SNR 125-140 dB vs a fixed-order f32 reference sum at
    the real per-column contention level, `/results/opt7/p16a_atomic_probe.py`).
    """
    off_bytes = elem_off * fx.Int32(4)
    rocdl.raw_ptr_buffer_atomic_fadd(val, rsrc, off_bytes, 0, 0)


_RQ_BLK = 128  # row-only flat kernel block size (independent of the dual's BLK)

# ---- P1+P2: grid-class blocked XCD remap for the row-only flat kernel ------
# (optimize round 3). `xcd_remap_pid_blocked` takes its `total_pids` as a
# Python int -- it does Python-level `if full == 0` / `full == total_pids`
# compile-time branches internally (gemm_helper.py) -- so `grid_x` must be
# baked into the compiled kernel body, not passed as a runtime launch
# argument. The flat kernel's ONLY shape-visible quantity is
# `grid_x = R*C//32//_RQ_BLK`: every address in the body is computed from
# `bid*_RQ_BLK` onward with no other reference to R or C, so transposing
# (R, C) leaves the compiled kernel byte-for-byte identical. The table
# therefore keys on grid_x, not (R, C).
#
# Values are the per-grid-class remap run length B, each independently
# confirmed this round with a STRICTLY-INTERLEAVED single-shot A/B, N=25-30
# cycles, alternating which side goes first every cycle so shared-box session
# drift cancels instead of biasing whichever candidate is measured later
# (kb/flydsl/pitfalls.md "Benchmark Noise on a Shared GPU Box"). A same-round
# sequential cold sweep (measure B=0 once, then each candidate once, in a
# fixed order) produced sign-flipped and 3-8x-inflated deltas on several of
# these exact grids on this box -- e.g. grid 9216 measured +3.3% sequential
# but -1.6% interleaved, and grid 12288 measured -8.3% sequential but -1.75%
# interleaved -- so only the interleaved numbers are trusted for this table;
# see this round's ROUND_REPORT for the full before/after comparison.
#
# grid_x 11520 (15360x3072 / 3072x15360) and 16128 (3072x21504 / 21504x3072)
# are DELIBERATELY OMITTED (fall back to B=0, the unmodified prologue), not
# just untested: 11520 showed no B in {4, 8, 16} with a repeatable interleaved
# win, and 16128's two REAL ruler call sites disagree in SIGN at every B in
# {4, 8, 16, 32, 64, 128, 256} tried -- one improves, the other regresses by
# more -- so no single compile-time-baked B serves both real dispatches that
# share that grid_x. This reconfirms (with both physical shapes, not one) the
# prior round's memory note that 3072x21504 must not enter the table.
_XCD_TABLE = {
    86016: 256,  # 16384x21504, 21504x16384 -- interleaved ratio 0.969-0.979
    61440: 256,  # 15360x16384, 16384x15360 -- interleaved ratio 0.973-0.975
    24576: 64,  # 12288x8192, 8192x12288 -- interleaved ratio 0.987
    18432: 16,  # 8192x9216 -- interleaved ratio 0.984
    12288: 16,  # 16384x3072, 3072x16384 -- interleaved ratio 0.983 (32 ties; 64/128 regress)
    9216: 16,  # 12288x3072, 3072x12288 -- interleaved ratio 0.984 (32 ties)
    6912: 16,  # 3072x9216, 9216x3072 -- interleaved ratio 0.990
    6144: 16,  # 8192x3072, 3072x8192 -- interleaved ratio 0.973
}


def _pick_xcd_block(grid_x):
    """Interleaved-verified remap length B for this `grid_x`, or 0 (no remap,
    original prologue) for every grid_x not in `_XCD_TABLE` -- a shape that
    misses the table is guaranteed byte-identical to the pre-P1 kernel."""
    return _XCD_TABLE.get(int(grid_x), 0)


def _build_rowq_kernel(row_rht, xb=0, grid_x=0, row_sr=False):
    @flyc.kernel(known_block_size=[_RQ_BLK, 1, 1])
    def _rowq_kernel(
        X: fx.Tensor,  # int32 view [R, C/2]
        ROW_OUT: fx.Tensor,  # int32 view [R, C/8]
        ROW_SC: fx.Tensor,  # uint8 [R, C/32]
        SCALE_ROUNDING_BIAS: fx.Int32,
        SR_SEED: fx.Int32,
    ):
        tid = fx.thread_idx.x
        bid = fx.block_idx.x
        if const_expr(xb > 0):
            # Pure index permutation (gemm_helper.xcd_remap_pid_blocked): keeps
            # `xb` consecutive flat-kernel blocks (`xb * 8 KiB` of input) on the
            # same gfx950 XCD instead of letting the dispatcher round-robin
            # every block across all 8 L2 slices. Bijection over [0, grid_x),
            # so this changes nothing about the numerics -- only which
            # physical block computes which logical microblock range. Shapes
            # whose grid_x misses `_XCD_TABLE` get xb=0 and skip this
            # entirely (bid stays the raw, unremapped block id).
            bid = xcd_remap_pid_blocked(bid, grid_x, 8, xb)
        mb0 = arith.index_cast(T.index, bid * _RQ_BLK)
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
        # One thread owns one micro-block here (unlike the dual, where a thread
        # loops over several), so the grid-unique id is just its flat index. Note
        # `bid` is the XCD-remapped block above, which is fine and in fact better:
        # the remap is a bijection over [0, grid_x), so every micro-block still
        # draws a distinct seed. Constexpr row_sr -> the plain (seed=None) IR when
        # SR is off, so this costs nothing unless asked for.
        seed = _sr_hash(SR_SEED ^ (bid * _RQ_BLK + tid)) if const_expr(row_sr) else None
        words, biased = _finish_microblock(rbits, row_rht, SCALE_ROUNDING_BIAS, seed)
        buffer_ops.buffer_store(Vec.from_elements(list(words), fx.Int32), osrc, tid * 4, cache_modifier=2)
        buffer_ops.buffer_store(arith.trunci(T.i8, biased & 0xFF), ssrc, tid)

    return _rowq_kernel


def _build_rowq_launch(row_rht, xb=0, grid_x=0, row_sr=False):
    kern = _build_rowq_kernel(row_rht, xb, grid_x, row_sr)

    @flyc.jit
    def _rowq_launch(X, ROW_OUT, ROW_SC, BIAS: fx.Int32, SR_SEED: fx.Int32, gx: fx.Int32, stream: fx.Stream):
        kern(X, ROW_OUT, ROW_SC, BIAS, SR_SEED).launch(grid=(gx, 1, 1), block=(_RQ_BLK, 1, 1), stream=stream)

    return _rowq_launch


_ROWQ_LAUNCH = {}
_ROWQ_COMPILED = {}
_ROWQ_PLAN = {}


def rowq_eligible(R, C):
    return (int(C) % 32 == 0) and ((int(R) * int(C) // 32) % _RQ_BLK == 0)


def get_rowq_cast(R, C, row_rht, row_sr=False):
    grid_x = (R * C // 32) // _RQ_BLK
    xb = _pick_xcd_block(grid_x)
    # Cache key adds (xb, grid_x) -- but ONLY bakes grid_x in when xb > 0.
    # The compiled kernel body only reads `grid_x` inside the
    # `const_expr(xb > 0)` branch (it's dead/unreferenced code otherwise), so
    # every shape that lands on xb=0 is byte-identical regardless of its
    # grid_x and must share ONE launch closure -- exactly like the original,
    # pre-P1 shape-agnostic universal binary. Without this normalization,
    # grid_x would fragment the xb=0 shapes (2304, 11520, 16128, ...) into one
    # extra compile each, which only adds cold-JIT-compile overhead on a
    # fresh-cache run and buys nothing (measured this round: an early
    # official-bench A/B before this fix showed the two FIRST-processed
    # shapes regress 1-2% -- consistent with extra front-loaded compiles
    # stealing GPU boost-clock ramp time from a cache-cleared run -- while
    # every other shape still improved; folding xb=0 into one shared entry
    # removes that fan-out.) Two shapes sharing a grid_x with xb>0 (e.g.
    # 16384x3072 and 3072x16384, both grid_x=12288) still correctly reuse ONE
    # launch closure too, since xb>0 keys on the real grid_x.
    baked_grid_x = grid_x if xb > 0 else 0
    lk = (bool(row_rht), xb, baked_grid_x, bool(row_sr))
    raw = _ROWQ_LAUNCH.get(lk)
    if raw is None:
        raw = _build_rowq_launch(bool(row_rht), xb, baked_grid_x, bool(row_sr))
        _ROWQ_LAUNCH[lk] = raw
    key = (int(R), int(C), bool(row_rht), bool(row_sr))
    ent = _ROWQ_COMPILED.get(key)
    if ent is None:
        import torch

        x = torch.zeros((R, C // 2), dtype=torch.int32, device="cuda")
        ro = torch.zeros((R, C // 8), dtype=torch.int32, device="cuda")
        rs = torch.zeros((R, C // 32), dtype=torch.uint8, device="cuda")
        fn = flyc.compile(raw, x, ro, rs, 1 << 21, 0, grid_x, torch.cuda.current_stream())
        ent = (fn, grid_x)
        _ROWQ_COMPILED[key] = ent
    return ent


def _make_rowq_plan(R, C, row_rht, fp4_dtype, scale_rounding_mode=0, row_sr=False):
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

    fn, grid_x = get_rowq_cast(R, C, row_rht, row_sr)
    ro_shape = (R, C // 2)  # fp4_dtype elements (1 B each) == C//8 i32 words (4 B each)
    rs_shape = (R, C // 32)
    bias = _mxfp4_scale_rounding_bias(scale_rounding_mode)
    e8m0 = torch.float8_e8m0fnu
    raw_stream = torch._C._cuda_getCurrentRawStream

    def _plan(x_bf16):
        ro = x_bf16.new_empty(ro_shape, dtype=fp4_dtype)
        rs = x_bf16.new_empty(rs_shape, dtype=e8m0)
        # A fresh seed per launch, not per plan: the plan is cached and reused for
        # every call at this shape, so hoisting the seed would make every step of
        # training reuse one draw and turn SR back into a fixed dither. Same
        # reasoning `_make_plan` spells out for the dual.
        sr_seed = _next_sr_seed() if row_sr else 0
        fn(x_bf16, ro, rs, bias, sr_seed, grid_x, raw_stream(x_bf16.device.index))
        return ro, rs

    return _plan


def flydsl_quant_mxfp4_h16(x_bf16, fp4_dtype, scale_rounding_mode=0, sr=False):
    """Rowwise mxfp4 cast with the deterministic in-kernel H16 (``_rht16``), i.e.
    exactly what ``MXFP4Linear`` needs: bit-identical to
    ``flydsl_dual_quant(x, fp4_dtype, True, False)[:2]`` without computing or
    storing the discarded colwise pack.

    ``sr`` requests stochastic rounding, which makes the quantization error
    zero-mean instead of a repeatable bias (see ``flydsl_quant_mxfp4_h16_dual``
    for the measurements). Bit-exactness against the C++ reference applies to
    ``sr=False`` only -- SR is random by design."""
    R, C = x_bf16.shape
    if not rowq_eligible(R, C):
        row, scale, _, _ = flydsl_dual_quant(x_bf16, fp4_dtype, True, False, row_sr=sr)
        return row, scale
    key = (int(R), int(C), True, fp4_dtype, int(scale_rounding_mode), bool(sr))
    plan = _ROWQ_PLAN.get(key)
    if plan is None:
        plan = _make_rowq_plan(R, C, True, fp4_dtype, scale_rounding_mode, bool(sr))
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


# ---- H16 fusion Linear: one-read row+col pack ------------------------------
# (optimize round 8 / N1). The fusion Linear (`mxfp4_linear.py`'s
# `_MXFP4LinearFunction`) packs each of X and grad_out TWICE per step, in two
# orientations, from one logical buffer: once as-is (row pack, along its own
# last dim -- what the fprop/dgrad GEMM contracts over) and once as
# `t().contiguous()` (col pack, along the OTHER dim -- what the wgrad GEMM
# contracts over). Both calls already resolve to `flydsl_quant_mxfp4_h16`
# whenever the recipe matches (see that function's docstring), so the second
# orientation costs a full transpose-copy kernel (measured 0.26-1.03 TB/s,
# 5-36x the pack it feeds -- see this round's ROUND_REPORT) plus a second
# independent pack launch, when one `flydsl_dual_quant(x, fp4, True, True)`
# call reads `x` ONCE and produces both packs directly (bit-exact, verified
# below and re-verified this round on GPU).
#
# `flydsl_quant_mxfp4_h16` itself is NOT touched by any of this -- this is a
# pure additive wrapper that reuses it (for the fallback) and `flydsl_dual_quant`
# (for the fused path), both already defined above.
_H16_DUAL_PLAN = {}


def _make_h16_dual_plan(R, C, fp4_dtype, scale_rounding_mode=0, row_sr=False, col_sr=False):
    """Generative closure for ``flydsl_quant_mxfp4_h16_dual``, same shape as
    ``_make_rowq_plan``/``_make_plan``: hoist the eligibility check and the
    fused-vs-fallback branch out of the per-call path so a cache hit costs one
    dict lookup instead of re-deriving the branch every call.

    Key = (R, C, fp4_dtype, scale_rounding_mode) -- pure shape/dtype/flag
    values, no tensor identity -- bucket K1 (kernel-internal per-shape compile
    cache), same bucket as ``_ROWQ_PLAN``/``_PLAN``. Rule 11's id(activation)
    ban does not apply: nothing here is keyed on ``id(x_bf16)`` or
    ``x_bf16.data_ptr()``, only on the shape/dtype the caller passes.
    """
    if R % 128 != 0 or C % 256 != 0:
        if row_sr or col_sr:
            # The row-only kernel this falls back to has no SR path, so honouring
            # the request is impossible here. Refuse rather than quietly packing
            # these shapes round-to-nearest: a run where some layers are
            # stochastic and others are not measures neither recipe.
            raise NotImplementedError(
                f"stochastic rounding requested at (R={R}, C={C}), which is not "
                "dual-eligible (needs R%128==0 and C%256==0); the row-only "
                "fallback pack has no SR path"
            )

        # Same fallback a caller would hand-roll without this function: two
        # independent flydsl_quant_mxfp4_h16 calls, one per orientation. The
        # fused dual kernel's own grid math (`(R // _TR) * (C // _TC)`,
        # `_TR=64`/`_TC=256`) has no masking path for a ragged tail, so any
        # (R, C) that `dual_eligible` would reject must not reach it.
        def _plan(x_bf16):
            row, row_scale = flydsl_quant_mxfp4_h16(x_bf16, fp4_dtype, scale_rounding_mode)
            col, col_scale = flydsl_quant_mxfp4_h16(x_bf16.t().contiguous(), fp4_dtype, scale_rounding_mode)
            return row, row_scale, col, col_scale

        return _plan

    def _plan(x_bf16):
        return flydsl_dual_quant(
            x_bf16,
            fp4_dtype,
            True,
            True,
            row_sr=row_sr,
            col_sr=col_sr,
            scale_rounding_mode=scale_rounding_mode,
        )

    return _plan


def flydsl_quant_mxfp4_h16_dual(x_bf16, fp4_dtype, scale_rounding_mode=0, row_sr=False, col_sr=False):
    """One-read row+col H16 pack for the fusion Linear.

    Produces both the row pack (``flydsl_quant_mxfp4_h16(x_bf16, ...)``, along
    ``x_bf16``'s own last dim) and the col pack
    (``flydsl_quant_mxfp4_h16(x_bf16.t().contiguous(), ...)``, along the
    transpose's last dim) from a SINGLE bf16 read via
    ``flydsl_dual_quant(x_bf16, fp4_dtype, True, True)``, eliminating the
    ``.t().contiguous()`` transpose copy that would otherwise sit between the
    two independent calls.

    Bit-identical to calling ``flydsl_quant_mxfp4_h16`` on each orientation
    separately: the dual kernel's row phase never reads ``col_rht`` (see
    ``_emit_dual_body`` -- the row phase's amax/quantize loop has no
    reference to it), so ``(row_data, row_scale)`` at ``(row_rht=True,
    col_rht=True)`` is byte-for-byte what the row-only path already computes
    from ``x_bf16``. The col phase applies the identical ``_rht16`` +
    ``_finish_microblock`` machinery to the same data, addressed
    transpose-wise, that a row-only pack would apply to
    ``x_bf16.t().contiguous()``. Verified ``torch.equal`` on every
    dual-eligible ruler shape and the real Flux operand shapes this round
    (see ROUND_REPORT); ``dual_eligible``'s own docstring already documents
    the underlying dual kernel as bit-exact vs the C++ reference (non-SR).

    Falls back to two separate ``flydsl_quant_mxfp4_h16`` calls -- exactly
    what a caller would do without this function, so every (R, C) is
    supported, not just the dual-aligned ones -- whenever ``R % 128 != 0`` or
    ``C % 256 != 0`` (the fused dual kernel's own alignment requirement; see
    ``dual_eligible``).

    ``row_sr``/``col_sr`` request stochastic rounding on that orientation, which
    the underlying dual kernel implements with MI355X's native
    ``cvt_scalef32_sr_pk_fp4_f32`` convert -- inside a pack that already has the
    data resident, so it is free: measured 0.90x and 1.11x against sr=off at
    2048x3072 and 8192x12288. It costs ~3 dB of instantaneous SNR (19.02 -> 15.98)
    to make the error zero-mean: the share surviving averaging over N draws goes
    from a flat 0.77 to 1/sqrt(N), and the output shrinkage from -1.12% to -0.21%.
    Note the two are NOT bit-identical to each other by construction, so the
    bit-exactness claims above apply to the sr=False path.

    Returns ``(row_data, row_scale, col_data, col_scale)``.
    """
    R, C = x_bf16.shape
    key = (int(R), int(C), fp4_dtype, int(scale_rounding_mode), bool(row_sr), bool(col_sr))
    plan = _H16_DUAL_PLAN.get(key)
    if plan is None:
        plan = _make_h16_dual_plan(R, C, fp4_dtype, scale_rounding_mode, row_sr, col_sr)
        _H16_DUAL_PLAN[key] = plan
    return plan(x_bf16)


# ---- H16 dual: two-leg column-slice packing (optimize round 10 / P6) ------
# Charter's P6: give the dual an input-side stride so a wide activation that is
# itself the materialized result of a `torch.cat` of two producers (e.g. Flux
# SingleStreamBlock linear2's X = cat(FA_out[*, 3072], GELU(linear1_C)[*, 12288]))
# can be packed as two independently-scheduled column-slice launches instead of
# one monolithic dual -- each leg writes DIRECTLY into the matching slice of one
# shared row-out/col-out pair, so no concatenation kernel runs on either side.
# `_emit_dual_body`'s `legpack`/`XPAD`/`XOFF` above are the shared mechanism;
# this section is the thin per-leg kernel/launch/dispatch wrapper around it,
# built the same generative-closure way as `_build_dual_kernel`/`_build_dual_
# launch`/`get_dual_cast`/`_make_plan` above (same docstrings' rationale
# applies verbatim: R/XPAD/leg-C/XOFF are all traced kernel arguments, never
# baked into the compiled kernel, so one compiled kernel per (leg-C, recipe,
# order) serves every (R, XPAD, XOFF) that shares those).
def _build_dual_legpack_kernel(row_rht, col_rht, col_locality=False, xcd_remap=False):
    """Leg-pack twin of `_build_dual_kernel`: only the non-2D-amax H16 recipe
    (`row_2d=col_2d=False`) is needed -- weight's batched/2D-amax geometry
    never goes through this path -- so those two are hardcoded rather than
    threaded as extra params."""
    _DualSS = _make_dual_struct(False)

    @flyc.kernel(known_block_size=[BLK, 1, 1])
    def _dual_legpack_kernel(
        X: fx.Tensor,  # int32 view [R, XPAD/2] -- the FULL (unsliced) activation
        ROW_OUT: fx.Tensor,  # int32 view [R, XPAD/8] -- shared full-width row pack
        ROW_SC: fx.Tensor,  # uint8 [R, XPAD/32]
        COL_OUT: fx.Tensor,  # int32 view [XPAD, R/8] -- shared full-height col pack
        COL_SC: fx.Tensor,  # uint8 [XPAD, R/32]
        R: fx.Int32,
        C: fx.Int32,  # THIS leg's own local width
        XPAD: fx.Int32,  # the shared buffers' true/full width
        XOFF: fx.Int32,  # this leg's starting column (elements) within XPAD
        SCALE_ROUNDING_BIAS: fx.Int32,
    ):
        lds = fx.SharedAllocator().allocate(_DualSS).peek()
        tid = fx.thread_idx.x
        _emit_dual_body(
            row_rht,
            col_rht,
            False,
            False,
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
            legpack=True,
            XPAD=XPAD,
            XOFF=XOFF,
        )

    return _dual_legpack_kernel


def _build_dual_legpack_launch(row_rht, col_rht, col_locality=False, xcd_remap=False):
    kern = _build_dual_legpack_kernel(row_rht, col_rht, col_locality, xcd_remap)

    @flyc.jit
    def _dual_legpack_launch(
        X: fx.Tensor,
        ROW_OUT: fx.Tensor,
        ROW_SC: fx.Tensor,
        COL_OUT: fx.Tensor,
        COL_SC: fx.Tensor,
        R: fx.Int32,
        C: fx.Int32,
        XPAD: fx.Int32,
        XOFF: fx.Int32,
        SCALE_ROUNDING_BIAS: fx.Int32,
        grid_x: fx.Int32,
        stream: fx.Stream,
    ):
        kern(X, ROW_OUT, ROW_SC, COL_OUT, COL_SC, R, C, XPAD, XOFF, SCALE_ROUNDING_BIAS).launch(
            grid=(grid_x, 1, 1), block=(BLK, 1, 1), stream=stream
        )

    return _dual_legpack_launch


_DUAL_LEGPACK_LAUNCH = {}
_DUAL_LEGPACK_COMPILED = {}


def get_dual_legpack_cast(R, C, XPAD, row_rht, col_rht):
    """(compiled_fn, grid_x) for one leg of a two-leg column-slice dual pack:
    packs ``C`` columns of an ``[R, XPAD]``-strided BF16 tensor into a shared
    ``[R, XPAD/8]`` row-out / ``[XPAD, R/8]`` col-out pair. ``XOFF`` (this
    leg's starting column) is a per-CALL runtime argument, not part of the
    compile key -- one compiled kernel serves every leg of every (R, XPAD)
    split that shares (C, recipe, order), exactly mirroring how
    ``get_dual_cast`` already treats R/C as runtime arguments rather than
    compile-time constants.

    Order (``col_locality``/``xcd_remap``) is looked up via the SAME
    ``_pick_block_order``/``_ORDER_TABLE`` oracle ``get_dual_cast`` uses,
    keyed on THIS leg's own (R, C) -- a leg is a real, independently-tileable
    (R, C) dispatch, so the existing table applies unchanged. Neither real
    leg shape has a table entry yet (this round adds none -- that would be a
    second mechanism), so both fall back to the table's own safe heuristic."""
    col_locality, xcd_remap = _pick_block_order(R, C, row_rht, col_rht, False, False)
    lk = (bool(row_rht), bool(col_rht), col_locality, xcd_remap)
    raw = _DUAL_LEGPACK_LAUNCH.get(lk)
    if raw is None:
        raw = _build_dual_legpack_launch(bool(row_rht), bool(col_rht), col_locality, xcd_remap)
        _DUAL_LEGPACK_LAUNCH[lk] = raw
    key = (int(R), int(C), int(XPAD), bool(row_rht), bool(col_rht))
    ent = _DUAL_LEGPACK_COMPILED.get(key)
    if ent is None:
        import torch

        x = torch.zeros((R, XPAD // 2), dtype=torch.int32, device="cuda")
        ro = torch.zeros((R, XPAD // 8), dtype=torch.int32, device="cuda")
        rs = torch.zeros((R, XPAD // 32), dtype=torch.uint8, device="cuda")
        co = torch.zeros((XPAD, R // 8), dtype=torch.int32, device="cuda")
        cs = torch.zeros((XPAD, R // 32), dtype=torch.uint8, device="cuda")
        grid_x = (R // _TR) * (C // _TC)
        stream = torch.cuda.current_stream()
        fn = flyc.compile(raw, *_static_layout(x, ro, rs, co, cs), R, C, XPAD, 0, 1 << 21, grid_x, stream)
        ent = (fn, grid_x)
        _DUAL_LEGPACK_COMPILED[key] = ent
    return ent


_H16_DUAL_LEGS_PLAN = {}


def _make_h16_dual_legs_plan(R, leg_widths, fp4_dtype, scale_rounding_mode=0):
    """Generative closure for ``flydsl_quant_mxfp4_h16_dual_legs``, same shape
    as ``_make_h16_dual_plan``: hoist the per-leg (fn, grid_x) lookups and the
    shared output-buffer shapes out of the per-call path."""
    import torch

    C = sum(leg_widths)
    scale_rounding_bias = _mxfp4_scale_rounding_bias(scale_rounding_mode)
    legs = []
    xoff = 0
    for w in leg_widths:
        fn, grid_x = get_dual_legpack_cast(R, w, C, True, True)
        legs.append((fn, grid_x, int(w), int(xoff)))
        xoff += w
    ro_shape = (R, C // 8)
    rs_shape = (R, C // 32)
    co_shape = (C, R // 8)
    cs_shape = (C, R // 32)
    i32 = torch.int32
    u8 = torch.uint8
    raw_stream = torch._C._cuda_getCurrentRawStream

    def _plan(x_bf16):
        x_i32 = x_bf16.view(i32)  # [R, C/2], the FULL (unsliced) activation
        ro = x_bf16.new_empty(ro_shape, dtype=i32)
        rs = x_bf16.new_empty(rs_shape, dtype=u8)
        co = x_bf16.new_empty(co_shape, dtype=i32)
        cs = x_bf16.new_empty(cs_shape, dtype=u8)
        stream = raw_stream(x_bf16.device.index)
        for fn, grid_x, w, off in legs:
            fn(x_i32, ro, rs, co, cs, R, w, C, off, scale_rounding_bias, grid_x, stream)
        return ro, rs, co, cs

    return _plan


def flydsl_quant_mxfp4_h16_dual_legs(x_bf16, leg_widths, fp4_dtype, scale_rounding_mode=0):
    """Two-(or-more)-leg column-slice variant of ``flydsl_quant_mxfp4_h16_dual``
    (optimize round 10 / P6): packs each ``x_bf16[:, off:off+w]`` column slice
    with its OWN dual-kernel launch, each writing directly into the matching
    slice of ONE shared ``(row_data, row_scale, col_data, col_scale)`` output
    set -- no separate per-leg buffers and no concatenation kernel afterward.

    Bit-identical to ``flydsl_quant_mxfp4_h16_dual(x_bf16, ...)``: every leg
    traces the SAME ``_emit_dual_body`` H16 recipe (``row_rht=col_rht=True,
    row_2d=col_2d=False``) as the monolithic dual, addressed through
    ``XPAD``/``XOFF`` instead of implicitly starting at column 0 of a
    ``C == XPAD``-wide buffer -- see ``_emit_dual_body``'s ``legpack``
    docstring section for the address-equivalence argument. Verified
    ``torch.equal`` against ``flydsl_quant_mxfp4_h16_dual`` at both real Flux
    leg splits (see this round's ROUND_REPORT).

    Motivation (probe D / this round's own re-measurement): splitting one
    wide dual launch into two tiles the XCDs better at these shapes than one
    wide launch does -- not a correctness change, a pure scheduling one
    (like ``_ORDER_TABLE``'s col_locality/xcd_remap), achieved here by
    shrinking the grid instead of reordering it.

    ``leg_widths`` must sum to ``x_bf16.shape[1]``; every entry must be a
    multiple of ``_TC`` (256, true for both real Flux splits used this
    round). Caller is responsible for the eligibility gate (``R % 128 == 0``
    plus the multiple-of-256 leg constraint); this function does not fall
    back to a monolithic or per-leg-independent pack on its own."""
    R, C = x_bf16.shape
    assert sum(leg_widths) == C, f"leg_widths {leg_widths} must sum to C={C}"
    key = (int(R), tuple(int(w) for w in leg_widths), fp4_dtype, int(scale_rounding_mode))
    plan = _H16_DUAL_LEGS_PLAN.get(key)
    if plan is None:
        plan = _make_h16_dual_legs_plan(R, leg_widths, fp4_dtype, scale_rounding_mode)
        _H16_DUAL_LEGS_PLAN[key] = plan
    import torch

    ro, rs, co, cs = plan(x_bf16)
    row_data = ro.view(torch.uint8).view(fp4_dtype)
    col_data = co.view(torch.uint8).view(fp4_dtype)
    row_scale = rs.view(torch.float8_e8m0fnu)
    col_scale = cs.view(torch.float8_e8m0fnu)
    return row_data, row_scale, col_data, col_scale


# ---- H16 dual + fused bias-gradient partial sum ----------------------------
# (optimize round 5 / P2). `_MXFP4LinearFunction.backward` (`mxfp4_linear.py`)
# reads `grad_2d` TWICE: once through `_quantize_mxfp4_h16_dual_op` (the row+
# col pack above) and once through `grad_2d.sum(0)` (the bias gradient) -- a
# second full BF16 HBM read of the exact tensor the dual just streamed
# through LDS one instruction earlier. `grad_2d.sum(0)` alone is 16.76
# ms/step under the scored (compiled-block) run -- LARGER than the 16.6
# ms/step of dual-read this campaign's charter targets (goal.md Sec A.2/C.P2).
#
# The dual's col phase already walks every row of a column's 32-row
# microblock through registers one at a time (`fb`, the loaded bf16 value
# shifted into the high 16 bits of an i32 -- `bitcast(f32, fb)` of that IS the
# bf16 value widened to f32 exactly, because that is how bf16->f32 conversion
# works: bf16 is simply the top 16 bits of an f32). Accumulating those 32
# values into a running f32 sum is 32 free `v_add_f32` on data already
# resident; the only NEW HBM traffic this adds is one f32 per
# (row-microblock, column) -- a `[R/32, C]` partial-sum buffer, R*C/8 bytes,
# vs the full 2*R*C bytes `grad_2d.sum(0)` reads today.
#
# The accumulator's live range is the load-loop ONLY -- summed BEFORE
# `_finish_microblock`, never after. An earlier prototype kept all 32
# `cbits` live to sum them AFTER `_finish_microblock` and paid +24.7..48.5%
# on the dual: `_dual_kernel_0` is VGPR 48 / AGPR 40, deeply memory-stalled
# (`SQ_WAIT_ANY/SQ_BUSY`=9.3), and 32 extra live i32 pushes it across the
# 256/48 -> 5-waves-per-SIMD occupancy cliff to 3 waves (goal.md Sec D). A
# single live f32 scalar costs +5.9..16.2% instead (validated prototype
# `/results/r3/dual_bias3.py`, landed here verbatim -- see
# kb/flydsl/lds_optimization.md's occupancy guidance and this kernel's own
# PMC census for why VALU is nearly free on this kernel but live VGPRs are
# not).
# ---- tanh-approximate dGELU, folded into the bf16 load ---------------------
_DG_KK = 0.044715
_DG_KB = 0.7978845608028654
_DG_K3K = 0.134145  # 3 * _DG_KK, Inductor's own folded literal
_DG_TWO_LOG2E = 2.8853900817779268
_DG_KB_TWO_LOG2E = _DG_KB * _DG_TWO_LOG2E
_DG_LOAD_GROUP = 8  # fused-load hoist depth; must divide _NLOAD (== 8) --
# optimize round 2 (P2) introduced this knob at 4: hoist `group` (x2 when
# dgelu) `buffer_load_dwordx4`s above any dGELU/LDS-store work per group, so
# more VMEM is in flight before the first `s_waitcnt`. group 1 (P0 alone) ->
# group 4 was a real cut on the fused legs at the time P2 landed, but round 2
# ALSO measured group 8 == group 4 without `nt` (226.73 vs 225.93 us) --
# the extra in-flight lines were thrashing L2 against each other with
# nothing to stop them.
#
# Optimize round 7 (P2b), on top of P15 (`nt`, below): with the two INPUT
# loads marked non-temporal that thrashing is gone, so the deeper hoist pays.
# Re-measured this round with the campaign's own interleaved canary-filtered
# race/plan harnesses, group 4 (this file's PRIOR value) vs group 8, both at
# ld_cm=2, bsum_fold=False, bit-identical outputs (mm 1.811e-06 both):
#   race.py  (kernel sum, leg0+leg1): 314.39 -> 307.32 us   (-2.25%)
#   plan2.py (full plan call):        336.21 -> 331.66 us   (-1.35%)
# group 2 and group 1 (also re-measured this round, same harnesses) are both
# WORSE than group 4 by +3.4-5.7%, so 8 is not "more hoist is free" -- 2 and 1
# lose, 4 was a local optimum, and only 8 (with `nt` already in the tree)
# beats it. Official ruler (bench.sh, card 7, this round): see ROUND_REPORT.
# _NLOAD == 8 (see its definition above), so 8 is the deepest hoist this tile
# geometry can express -- there is no larger divisor to try next on this
# knob; the next lever on leg1 is P13 (the end-of-block store drain), not a
# further retune of this one.
#
# P7 (BSUM fold) is a separate, later lever -- NOT part of this round's
# change; `_DG_BSUM_FOLD` stays False (r5 found its bias wrong on the fused
# legs; a correctness fix is still open, tracked separately).
#
# ---- gate/dgate two-plane body's OWN load-group constant (campaign
# 20261003_041039, optimize round 7 / P4) -- a SEPARATE knob from
# `_DG_LOAD_GROUP` above on purpose, even though it lands on the same value.
# The two mechanisms never co-occur at runtime: `gate`/`dgate` are only ever
# True from `_make_gate_plan`'s call into `_emit_dual_bias_body`, which
# always passes `XSTR=None`, while the dgelu legs' own call
# (`_build_dual_bias_legs_dgelu_kernel`) always leaves `gate`/`dgate` at
# their `False` default -- so this knob can never change `_DG_LOAD_GROUP`'s
# own tuned value either way. Keeping it as its own constant makes that true
# BY CONSTRUCTION instead of by coincidence, and gives a later round a
# single named knob to retune if the two paths' optimal depth ever diverges.
#
# Mechanism, re-measured THIS round on the live, already-stacked (P1+P2+P3)
# tree (goal.md Sec C.4/C.5 ran the same probe on an isolated prototype
# first): the gate/dgate body's load loop issues one `buffer_load_dwordx4`
# per chunk for dY, and -- whenever `dgate` -- a SECOND `buffer_load_dwordx4`
# for `y` at the SAME offset, so at full hoist there are up to 16 outstanding
# vec4 loads per tile instead of 8 (dY alone). rocprofv3 `--pmc` on this
# exact kernel (card 0, `double_proj`/`single_linear2`, 11 steady-state
# calls, group 1 vs group 8) shows it is stall-bound, not instruction-bound:
# SQ_INSTS_VALU/VMEM/LDS and SQ_LDS_BANK_CONFLICT are BIT-IDENTICAL between
# group 1 and group 8 at both shapes (dp 7293936/193632/362496/1622016 both;
# sl 14920752/408288/724992/3244032 both; SQ_WAVES also unchanged, 6408 /
# 14568) -- only `SQ_WAIT_ANY` moves, taking WAIT/BUSY from 9.25 -> 6.31 at
# dp and 10.45 -> 7.60 at sl. Deepening the hoist therefore only reorders
# issue (more VMEM in flight before the first `s_waitcnt`); it is bit-exact
# BY CONSTRUCTION, confirmed empirically every run this round (`bit_exact`/
# `bias_bit_exact` true, `max_mismatch_frac 0.0`, both SNRs identical to
# baseline). Resource footprint is unchanged by the hoist depth itself:
# `_dual_bias_gate_tile_kernel_0` is VGPR 76 + Accum 12 = 88, SGPR 112,
# LDS 32768, Scratch 0 at BOTH group 1 and group 8 (rocprofv3 dispatch
# record) -- comfortably inside the <=102/<=32768/==0 budget.
#
# Gated on `(gate or dgate)` specifically, NOT applied unconditionally: the
# SAME hoist on the plain one-plane production pack (`gate=False,
# dgate=False`, i.e. the body `_build_dual_bias_tile_launch` compiles)
# measured as a LOSS, not a win (goal.md Sec D.2, `CAND_PGRP` probe) --
# that body only ever has 8 outstanding loads at full hoist already (no
# second Y stream to overlap), so deepening its grouping just delays the
# first `s_waitcnt` with nothing new in flight to hide behind it. The
# production pack's own `_grp` (the trailing `1` below) must stay untouched,
# and this round's own `bench.sh` runs confirm it did: `pack_ms` sits at
# 3.627-3.647 ms before and after this change, inside the box's own +-0.6%
# noise band.
#
# Official ruler (`bash bench.sh`, card 7, weighted `ms` over all 3 cases),
# a monotone sweep of `_GATE_DGATE_LOAD_GROUP` on the live, already-stacked
# tree: group 1 (== the pre-P4 behaviour) 5.06996 / 5.07262 (2 samples,
# reproducing this round's fresh pre-edit measurement, as it must -- group 1
# is textually the old expression), group 2 5.08326 (a tie with group 1,
# both inside noise), group 4 5.00004, **group 8 4.86058 / 4.86248 / 4.88566
# / 4.90314 / 4.91416 / 4.91568 (median 4.8944)** -- a -3.5% cut vs this
# round's pre-edit baseline. Every sample at every group value measured
# `bit_exact: true`, `bias_bit_exact: true`, `max_mismatch_frac: 0.0`, and
# both SNRs exactly equal to baseline. Per-case (group 1 -> 8): double_proj
# 0.03408 -> 0.0329 (-3.5%), double_fc2 0.0351 -> 0.03312 (-5.6%),
# single_linear2 0.06424 -> 0.06194 (-3.6%) -- a uniform win on all three
# shapes, unlike P5's shape-dependent `nt` sign flip, so this lever needs no
# per-shape table.
_GATE_DGATE_LOAD_GROUP = 8  # == _NLOAD -- the deepest hoist this tile
# geometry can express (same ceiling `_DG_LOAD_GROUP`'s own comment above
# already derives for the dgelu legs; `_NLOAD // _grp` would skip chunks
# entirely for any `_grp` that does not divide `_NLOAD`, so there is no
# larger value to try -- confirmed structurally and not re-swept at 16).
_DG_LD_CM = 2  # cache policy on the two fused INPUT loads -- optimize round 3
# (P15). Bit 1 of the MUBUF aux field is `nt` (non-temporal); 0 is the
# ordinary policy (every other `buffer_load` in this file omits the kwarg
# entirely and gets it by default). `d_act`/
# `d_qkv` (the `xsrc` load) and `preact` (the `prsrc` load, leg1 only) are
# each read exactly once per element and never re-read within the kernel,
# but at group 4 every 128 B line they pull into the CU's L2 slice competes
# with the lines the STORE side is actively merging (the partial `col_out`/
# `row_out` runs into 64 B lines) and with the BSUM plane the col phase just
# wrote a moment earlier. rocprofv3 on the champion (goal.md Sec R3.1)
# measured L2 hit rate 0.420 with `TCC_EA0_RDREQ_32B_sum == 0` -- every read
# is a full-line miss and the ONLY hits are write-side line merges -- so
# these streamed reads are pure eviction pressure on lines that matter, not
# reuse of their own. Marking them `nt` does not move a single byte
# (`TCC_EA0_RDREQ_sum` is bit-for-bit identical with and without it,
# confirmed again this round); the win is latency and reduced L2 pollution,
# not bandwidth. The mirror knob on the OUTPUT stores (`st_cm=2`) is a
# measured LOSS (+33%, it defeats exactly the write-combining above) and is
# deliberately NOT wired to anything here -- this knob only ever reaches the
# two `buffer_ops.buffer_load` calls below, never a `buffer_store`.
#
# `_ldcm` (computed from this knob inside `_emit_dual_bias_body`) collapses
# to 0 whenever `_own_in` is False, i.e. whenever `XSTR` was not passed --
# the ONLY caller that leaves it unset is the plain, non-dgelu legpack body
# (`_build_dual_bias_legpack_kernel`, the shipped `_dual_bias_legpack_
# kernel_0` that IS the production baseline the ruler's `baseline_ms` and
# `pack_ms` call). That kernel's ISA is therefore unaffected BY CONSTRUCTION
# regardless of this knob's value -- re-verified byte-for-byte this round
# via the `bee661fc7f283a2957cf` / 46035 B hash (see this round's
# ROUND_REPORT for the fresh run).
# ANALYZE-only store/load ablation knob (timing only -- a non-empty value
# produces WRONG results). Comma-separated subset of
# {ro, rs, co, cs, bs, pr}: drop that output store, or the preact load.
_DG_SKIP = ""
# Fold the `_GW` per-column microblock partials with one DPP neighbour add, so
# BSUM is [R/(32*_GW), XPAD] instead of [R/32, XPAD]: halves the BSUM store and
# halves stage-2's read. Only valid with _GW == 2 (the shipped col grouping).
# Held OFF (P7 lands in a later round) -- see the _DG_LOAD_GROUP note above.
_DG_BSUM_FOLD = False
# P16(a) (optimize round 10): instead of storing this leg's per-(row-
# microblock, column) partial `acc` to a `[R/32, XPAD]` BSUM plane for a
# SEPARATE stage-2 kernel to reduce later, atomically accumulate it straight
# into a `[_DG_BIAS_SHARDS, XPAD]` mini-bias plane via BUFFER_ATOMIC_ADD_F32
# -- shrinking (not deleting -- see `_DG_BIAS_SHARDS` below) the BSUM store,
# the stage-2 read, AND stage-2's own data volume by R/(32*_DG_BIAS_SHARDS).
# Independent of `_DG_BSUM_FOLD`/P7 on purpose: P7's DPP pre-fold of the two
# `_GW` microblock partials into one is a separate, currently-broken lever
# (r5 found its bias wrong on the fused legs -- see the `_DG_BSUM_FOLD`
# comment above); gating the atomic on it would inherit that open bug.
#
# MEASURED (r10): a naive SINGLE-SLOT atomic (`_DG_BIAS_SHARDS=1`, straight
# into one `[XPAD]` vector, no stage-2 at all) was wired into the real fused
# leg kernel and raced end-to-end against champion with the plan-level
# canary-filtered harness -- SNR/byte-match are perfect (bias_snr_db 55.649,
# identical to champion, stable over 3 independent atomic-order trials;
# `/results/opt7/p16a_smoke.py`), but plan time REGRESSED +167.65% (354.29 ->
# 948.24 us/call, N=16 canary-kept cycles, `/results/opt7/p16a_race.py`).
# rocprofv3 (`/results/opt7/pmc_champ`, `/results/opt7/pmc_atomic`) roots
# this in the leg kernel itself, not some side effect: `_dual_bias_legs_
# dgelu_kernel_0` alone went 169.85 -> 448.0 us avg (the ENTIRE regression;
# stage-2's now-deleted 27.9 us/call is a rounding error against that), and
# TCC_EA0_WRREQ_sum went UP 43.6% (3.91M -> 5.61M/call) rather than down --
# atomics from different wavefronts hitting the SAME small address cannot
# coalesce the way this file's per-row-microblock BSUM stores do (each
# wavefront owns a unique row, so those coalesce fine), and
# TCC_EA0_ATOMIC_LEVEL_sum/TCC_EA0_ATOMIC_sum (the file's own documented
# average-in-flight-atomics formula) is ~2749 -- a deep backlog. This is
# exactly `kb/campaign-lessons.md` lesson 7's "single-slot global atomic"
# failure mode (measured there at +168 us/15360 blk, ~linear in grid size)
# and `hardware/gfx950/kernel-implementation-notes.md` SS5's own prescribed
# fix ("bias each warp's per-lane byte offset... so concurrent warps target
# disjoint packed slots and never collide") -- lesson 7 measured sharding to
# 256 slots at ~+0.3 us, i.e. free. See `_DG_BIAS_SHARDS`, which IS the
# applied fix: `_DG_BIAS_ATOMIC=True` with `_DG_BIAS_SHARDS>=2` measures as a
# real win (this round's ROUND_REPORT), so this knob defaults ON. Set False
# to cleanly revert to the legacy full-BSUM-plane store + full-P stage-2.
_DG_BIAS_ATOMIC = True
# Shard count for `_DG_BIAS_ATOMIC`'s mini-bias plane: each block's atomic
# lands in row `gmmb & (_DG_BIAS_SHARDS - 1)` instead of row `gmmb` (must be
# a power of 2 -- enforced by the bitmask, no modulo instruction needed).
# `gmmb` ranges over R/32 values (512 today), so `_DG_BIAS_SHARDS=32` maps
# 16 distinct row-microblocks onto each shard: 32x fewer BSUM rows than the
# legacy full plane (P16(a)'s `_DG_BIAS_SHARDS=1` IS the measured-regressed
# single-slot case above; this is lesson 7's fix, not a different mechanism).
# Stage-2 still runs, but on a `[_DG_BIAS_SHARDS, XPAD]` plane instead of
# `[R/32, XPAD]` -- far less data for stage-2 to read, on top of whatever
# contention the sharding itself removes from the leg kernels' atomics.
#
# MEASURED (r10, plan-level canary-filtered race vs champion, N=16 then
# confirmed at N=24, `/results/opt7/p16a_shard_sweep2.py` /
# `p16a_shard_final.py`): shards={2,4,8,16} are ALL wins and tied within
# noise (-8.2%..-4.9% first pass, -6.6%..-6.0% at N=24) -- the atomic
# contention collapses as soon as there is more than one slot (shards=1 is
# the measured +154%..+168% regression above), and the remaining curve is
# governed by stage-2's own read cost creeping up with more shard rows, not
# by residual contention. 4 is picked over the measured-best 2 for headroom
# against contention on hardware/conditions this round's probes did not
# cover (bench.sh's card 7 pin), at a cost within this round's own noise
# band against 2.
_DG_BIAS_SHARDS = 4


def _dgelu_pair(dy, x):
    """<2 x f32> tanh-GELU backward, collapsed through r = 1/(exp2(2*log2e*u)+1).

    `flydsl.expr.math.tanh` has no libcall on this target, so tanh is built from
    exp2 + rcp. With t = 1 - 2r the two halves collapse: 0.5*(1+t) = 1-r and
    1-t^2 = 4r(1-r), so dg = (1-r) * (1 + x*r*(2*idv)). exp2 saturating to +inf
    gives r -> 0 -> dg -> 1 and exp2 -> 0 gives r -> 1 -> dg -> 0, i.e. both
    tails are correct with no clamp (the algebraically equivalent 1/(1+s) form
    instead evaluates inf*0 and NaNs).
    """
    x_sq = x * x
    a = (x + x_sq * x * _DG_KK) * _DG_KB_TWO_LOG2E
    e0 = math.exp2(fx.Float32(a[0]))
    e1 = math.exp2(fx.Float32(a[1]))
    rv = Vec.from_elements(
        [
            fx.Float32(rocdl.rcp(T.f32, _raw(e0 + fx.Float32(1.0)))),
            fx.Float32(rocdl.rcp(T.f32, _raw(e1 + fx.Float32(1.0)))),
        ],
        fx.Float32,
    )
    ld = rv * (-1.0) + 1.0  # 1 - r
    idv2 = x_sq * (_DG_K3K * 2.0 * _DG_KB) + (2.0 * _DG_KB)  # 2 * idv
    return (dy * ld) * ((x * rv) * idv2 + 1.0)


def _gelu_pair(x):
    """<2 x f32> tanh-GELU, through the same r = 1/(exp2(2*log2e*u)+1) as `_dgelu_pair`:
    0.5*x*(1+tanh(u)) = x*(1-r). Saturating exp2 gives x at +inf and -0.0 at -inf, the
    values torch's formula produces there."""
    a = (x + x * x * x * _DG_KK) * _DG_KB_TWO_LOG2E
    e0 = math.exp2(fx.Float32(a[0]))
    e1 = math.exp2(fx.Float32(a[1]))
    rv = Vec.from_elements(
        [
            fx.Float32(rocdl.rcp(T.f32, _raw(e0 + fx.Float32(1.0)))),
            fx.Float32(rocdl.rcp(T.f32, _raw(e1 + fx.Float32(1.0)))),
        ],
        fx.Float32,
    )
    return x * (rv * (-1.0) + 1.0)


def _struct_f32(bits):
    import struct

    return struct.unpack("<f", struct.pack("<I", bits))[0]


# ocml's __ocml_tanh_f32 polynomial for |x| < 0.625 (from the image's ocml.bc), highest order first.
_OCML_TANH_C = tuple(_struct_f32(b) for b in (0xBBBAC73D, 0x3CA908C9, 0xBD5C1C4E, 0x3E088382, 0xBEAAAA99))
_TORCH_GELU_KBETA = 0.7978845608028654  # M_SQRT2 * M_2_SQRTPI * 0.5, rounded to f32 by the IR


def _tanh_ocml(v):
    """f32 tanh, instruction for instruction __ocml_tanh_f32 (what torch's tanh lowers to on ROCm)."""
    y = math.absf(v)
    x2 = v * v
    p = fx.Float32(_OCML_TANH_C[0])
    for c in _OCML_TANH_C[1:]:
        p = math.fma(x2, p, fx.Float32(c))
    small = math.fma(x2, y * p, y)
    e = math.exp(y * 2.0)
    big = math.fma(fx.Float32(-2.0), fx.Float32(rocdl.rcp(T.f32, _raw(e + fx.Float32(1.0)))), fx.Float32(1.0))
    return math.copysign(arith.select(y < fx.Float32(0.625), small, big), v)


def _gelu1_torch_bf16(x):
    """One f32 tanh-GELU whose bf16 rounding matches torch's (eager and Inductor), ties included,
    for every bf16 ``x`` (verified exhaustively) -- not for general f32 inputs, see `_gelu1_torch`.
    t = tanh(u) is formed without cancellation on each side: 1 - 2r for u >= 0, 2er - 1 for u < 0
    (r = 1/(e+1), e = exp(2u)); picking the side by e keeps r's denormal flush from mattering.
    About a third cheaper in the legs pack than `_gelu1_torch`, whose GELU is ALU-bound."""
    a = (x + x * x * x * _DG_KK) * _DG_KB_TWO_LOG2E
    e = math.exp2(a)
    r = fx.Float32(rocdl.rcp(T.f32, _raw(e + fx.Float32(1.0))))
    t = arith.select(e < fx.Float32(1.0), (e * r) * 2.0 - fx.Float32(1.0), fx.Float32(1.0) - r * 2.0)
    return (x * 0.5) * (t + fx.Float32(1.0))


def _gelu1_torch(x):
    """One f32 tanh-GELU equal to torch's (eager ATen and Inductor, both of which contract
    x + kKappa*x^3 into an FMA) for every input: verified exhaustively over bf16 and on 1e8
    f32 sums of two bf16, which is what a GELU of a bf16 GEMM output plus a bf16 bias sees."""
    inner = math.fma(fx.Float32(_DG_KK), (x * x) * x, x) * _TORCH_GELU_KBETA
    return (x * 0.5) * (_tanh_ocml(inner) + fx.Float32(1.0))


def _bf16_pair_to_f32(word):
    """The two bf16 halves of one i32 as <2 x f32> (low half first, matching the
    pack's own `word << 16` / `word & 0xFFFF0000` convention)."""
    return Vec.from_elements(
        [
            Vec.from_elements([word << 16], fx.Int32).bitcast(fx.Float32)[0],
            Vec.from_elements([word & 0xFFFF0000], fx.Int32).bitcast(fx.Float32)[0],
        ],
        fx.Float32,
    )


def _emit_dual_bias_body(
    lds,
    tid,
    X,
    ROW_OUT,
    ROW_SC,
    COL_OUT,
    COL_SC,
    BSUM,
    R,
    C,
    bid,
    scale_rounding_bias,
    col_locality=False,
    xcd_remap=False,
    legpack=False,
    XPAD=None,
    XOFF=None,
    dgelu=False,
    PR=None,
    XSTR=None,
    PRSTR=None,
    tile_bsum=False,
    fold_nrblk=None,
    BIAS=None,
    CNT=None,
    ROW_SCP=None,
    COL_SCP=None,
    GATE=None,
    Y=None,
    YBIAS=None,
    DGS=None,
    gate=False,
    dgate=False,
    lmb2=1,
    y_ld_cm=0,
):
    """`_emit_dual_body`'s `(row_rht=True, col_rht=True, row_2d=False,
    col_2d=False, batched=False, padded=False, skip_row=False)` path --
    exactly the `_H16_RECIPE` geometry `flydsl_quant_mxfp4_h16_dual`
    dispatches to -- verbatim, plus a per-(row-microblock, column) sum
    accumulated inside the col phase's 32-row load loop and stored to
    `BSUM` (`[R/32, C]` f32; one coalesced run of `C` contiguous f32 per
    row-microblock, so adjacent lanes' `c_col` stores land in the same
    cacheline).

    Only ever called for the H16 fusion recipe -- the sole caller
    (`get_dual_bias_cast`) always builds this with `row_rht=col_rht=True`
    baked in, so unlike `_emit_dual_body` those are not parameters here.
    `batched`/`padded`/`skip_row`/stochastic-rounding are not supported:
    `_dual_pack_eligible` (`mxfp4_linear.py`) already requires SR and
    randomized-RHT off before this recipe is even reachable, and `grad_2d`
    (the sole caller-side tensor) is never batched-3D or padded, so none of
    those branches would ever execute -- omitting them keeps this body a
    direct, auditable copy of the ONE path that matters instead of
    re-parameterizing a path nothing exercises.

    ``legpack``/``XPAD``/``XOFF`` (optimize round 10 / P6, extended to the
    bias-fused recipe): identical contract to `_emit_dual_body`'s own
    ``legpack`` -- packs the column slice ``X[:, XOFF:XOFF+C]`` of a wider
    ``[R, XPAD]`` tensor into the matching slice of a SHARED row-out/col-out
    pair, here ALSO extended to `BSUM`: a legpack `BSUM` is logically
    `[R/32, XPAD]` (row-stride `XPAD`, not this call's own local `C`), so
    each leg's partial-sum row lands at the correct absolute column and a
    SINGLE stage-2 reduce over the full `[R/32, XPAD]` buffer (called once,
    after every leg's kernel has run -- see `_make_h16_dual_bias_legs_plan`)
    folds both legs' partials with no stage-2 changes at all. Same read-vs-
    write asymmetry as `_emit_dual_body`: X's own SRD stride and the load
    loop use ``xpad`` (this leg needs to skip past the OTHER leg's columns
    within the SAME shared physical row), ROW_OUT/ROW_SC/BSUM's leg offset
    is a bounded post-hoc add (``_row_off``/``_rowsc_off``, or folded into
    BSUM's own SRD base via ``c0i`` alongside COL_OUT/COL_SC), exactly per
    `_emit_dual_body`'s docstring rationale.

    ``GATE``/``gate``/``lmb2`` (optimize round 4 / P1): scale dY by
    ``gate[b, c]`` at the bf16 load, before anything reaches LDS, so the ROW
    and COL phases both read the identical staged ``G = bf16(gate * dY)`` a
    materialised-G pack would have produced. ``Y``/``YBIAS``/``DGS``/
    ``dgate`` (optimize round 5 / P2): read the GEMM output ``y`` and its
    ``[C]`` bias at the SAME bf16 load (same ``goff`` as dY, so the load
    stays vectorised and in the col phase's existing row order -- the
    precondition `sb-dgelu-legpack` established for a fused producer to cost
    less than the traffic it deletes), and accumulate
    ``fp32(dY) * (fp32(y) + fp32(y_bias))`` per column into ``DGS``
    (``[R/_TR, C]`` f32, the SAME per-tile-column grain ``BSUM`` has under
    ``tile_bsum=True``) -- so the gate gradient's reduction no longer
    re-reads dY from HBM at all; it rides the pack's own load.

    ``y_ld_cm`` (optimize round 8 / P5): cache-policy bit for the ``Y`` load
    ONLY, separate from ``_ldcm`` (which keeps gating dY's own load and the
    dgelu legs' loads unchanged). The caller (``_build_gate_tile_launch``)
    threads this from a per-``(R, C)`` table, since ``y`` is read exactly
    once per element here and whether marking it non-temporal helps or hurts
    depends on how oversubscribed the L2 is at that shape -- see
    ``_GATE_Y_NT_TABLE`` / ``_pick_gate_y_cm`` below.
    """
    ncblk = C // _TC
    _skip = set(s for s in _DG_SKIP.split(",") if s) if dgelu else set()
    _bfold = bool(_DG_BSUM_FOLD) and XSTR is not None
    # `_DG_BIAS_ATOMIC` takes priority over `_bfold` below, which masks the
    # fold's known-wrong bias only while the atomic defaults on.
    assert not _bfold, (
        "_DG_BSUM_FOLD is known-wrong on the fused-legs dGELU body "
        "(bias_snr_db ~3dB); do not enable it until the DPP pre-fold's bias bug is fixed"
    )
    # P16(a): atomic-accumulate straight into a [_bshards, XPAD] mini-bias
    # plane instead of the full [R/32, XPAD] BSUM -- see the module-level
    # comment on `_DG_BIAS_ATOMIC`/`_DG_BIAS_SHARDS` for why this is
    # independent of `_bfold`, and why sharding (not a single slot) is
    # required to avoid atomic serialisation (lesson 7).
    _batomic = bool(_DG_BIAS_ATOMIC) and XSTR is not None
    _bshards = int(_DG_BIAS_SHARDS) if _batomic else 1
    assert _bshards & (_bshards - 1) == 0, "_DG_BIAS_SHARDS must be a power of 2"
    _bmask = fx.Int32(_bshards - 1)
    xpad = XPAD if legpack else C  # shared buffers' true/full width
    # P0: the INPUT plane's row stride, decoupled from the OUTPUT plane's
    # `xpad`. Production passes `XSTR=None` and keeps the single-plane
    # read-vs-write asymmetry `_emit_dual_body`'s legpack docstring describes
    # (input stride == xpad, input column offset == XOFF), which makes every
    # expression below textually identical to the pre-P0 body. The fused legs
    # instead read their OWN [R, XSTR] tensor whose base pointer already
    # carries the column offset (they arrive as `t[:, off:]` views), so their
    # input column offset is 0.
    xstr = xpad if XSTR is None else XSTR
    _own_in = XSTR is not None
    # P15: non-temporal cache policy on the two INPUT loads below (see the
    # module-level comment on `_DG_LD_CM`). `_own_in` is already the exact
    # Python-level gate the rest of this function uses to tell the fused-legs
    # body apart from the plain legpack body it shares this code with, so
    # reusing it here is what keeps the plain body's ISA untouched: `_ldcm`
    # is a plain Python int (0 or 2) fixed at trace time, never a runtime
    # branch.
    _ldcm = int(_DG_LD_CM) if _own_in else 0
    if xcd_remap:
        # Same 8:1 XCD remap as `_emit_dual_body` -- see that function's
        # docstring; `_pick_block_order` only ever requests it where the
        # block count is a verified multiple of 8, so this bijection is exact.
        _nxcd = 8
        _per = (R // _TR) * ncblk // _nxcd
        bid = (bid % _nxcd) * _per + (bid // _nxcd)
    if col_locality:
        nrblk = R // _TR
        cblk = bid // nrblk
        rblk = bid % nrblk
    else:
        rblk = bid // ncblk
        cblk = bid % ncblk
    r0 = rblk * _TR
    c0w = cblk * _TCW

    r0i = arith.index_cast(T.index, r0)
    _colbase = cblk * _TC
    if legpack:
        # BSUM/COL_OUT/COL_SC are all indexed by ORIGINAL column, so this
        # leg's global start must move each of their SRD BASEs -- same
        # reasoning as `_emit_dual_body`'s `legpack` docstring.
        _colbase = _colbase + XOFF
    c0i = arith.index_cast(T.index, _colbase)
    rsrc = _srd_at(X, r0i * arith.index_cast(T.index, xstr >> 1), 4, _TR * (xstr >> 1) * 4)
    if gate:
        # GATE is [B, xpad] bf16 viewed as i32, one row per batch sample
        # (optimize round 4 / P1). The whole 64-row tile belongs to sample
        # `rblk // lmb2` (lmb2 = L / _TR, L = R / B); `_make_gate_plan` only
        # builds this path when L % _TR == 0, falling back to a torch gate
        # multiply otherwise.
        gsrc = _srd_at(GATE, fx.Int32(0), 4, (R // (lmb2 * _TR)) * (xpad >> 1) * 4)
    if dgate:
        # Y is dY's own GEMM output -- same [R, xpad] bf16 layout as X, read
        # at the SAME `goff` the load loop already computes for X below, so
        # dgate's partial sum rides the load the col phase's `acc` already
        # needs instead of a separate re-read (optimize round 5 / P2).
        # YBIAS is [xpad] bf16, one vector per tile like GATE. DGS is
        # [R/_TR, xpad] f32 (the SAME per-tile-column grain BSUM has under
        # `tile_bsum=True`) normally, or a PERSISTENT [B, xpad] accumulator
        # when `_DGATE_ACC_ATOMIC` (optimize round 11 / P9, see
        # `_dgate_acc_plane`) -- either way re-based by `c0i` the same way,
        # only the row COUNT (and hence the SRD's valid range) differs.
        ysrc = _srd_at(Y, r0i * arith.index_cast(T.index, xstr >> 1), 4, _TR * (xstr >> 1) * 4)
        ybsrc = _srd_at(YBIAS, fx.Int32(0), 4, (xpad >> 1) * 4)
        _dg_nrows = (R // (lmb2 * _TR)) if _DGATE_ACC_ATOMIC else (R >> 6)
        dgsrc = _srd_at(DGS, c0i, 4, _dg_nrows * xpad * 4)
    if const_expr(dgelu):
        prsrc = _srd_at(PR, r0i * arith.index_cast(T.index, PRSTR >> 1), 4, _TR * (PRSTR >> 1) * 4)
    orsrc = _srd_at(ROW_OUT, r0i * arith.index_cast(T.index, xpad >> 3), 4, _TR * (xpad >> 3) * 4)
    rscrsrc = _srd_at(ROW_SC, r0i * arith.index_cast(T.index, xpad >> 5), 1, _TR * (xpad >> 5))
    corsrc = _srd_at(COL_OUT, c0i * arith.index_cast(T.index, R >> 3), 4, _TC * (R >> 3) * 4)
    cscrsrc = _srd_at(COL_SC, c0i * arith.index_cast(T.index, R >> 5), 1, _TC * (R >> 5))
    # Same tile-column-based SRD base as COL_OUT/COL_SC above (re-based so the
    # in-body offset only needs the WITHIN-tile column, exactly the `_fold`
    # trick `_emit_dual_body` already uses for every other SRD here); `BSUM`
    # is logically `[R/32, xpad]` row-major, f32 (legpack: `xpad` is the
    # SHARED full width, not this leg's own local `C`), so this SRD's
    # `num_records` covers the whole tensor (cheap and always safe -- there
    # is no padding path here to need a tighter per-tile bound, unlike
    # COL_OUT/COL_SC). P16(a): when `_batomic`, the SAME `BSUM` parameter
    # instead carries the `[_bshards, xpad]` mini-bias plane (`_bshards` rows
    # instead of R/32) -- same c0i-based re-basing, a much smaller
    # `num_records` whenever `_bshards < R/32` (the sharded case).
    if _batomic:
        biasrc = _srd_at(BSUM, c0i, 4, _bshards * xpad * 4)
    else:
        bsrc = _srd_at(BSUM, c0i, 4, (R >> 5) * xpad * 4)
    # ROW_OUT/ROW_SC's leg offset: a bounded post-hoc add inside the SRD's
    # existing per-tile bound (which is sized off `xpad`, giving exactly
    # enough slack for this -- see `_emit_dual_body`'s `legpack` docstring),
    # not a base-offset fold like COL_OUT/BSUM above.
    _row_off = (XOFF >> 3) if legpack else 0
    _rowsc_off = (XOFF >> 5) if legpack else 0
    _pk_row, _pk_col = [], []

    # `_DG_LOAD_GROUP` hoists `group` (x2 when dgelu) `buffer_load_dwordx4`
    # above any dGELU/LDS-store work, so more VMEM is in flight before the
    # first `s_waitcnt`. group == 1 reproduces the pre-P0 straight-line loop
    # exactly. `_GATE_DGATE_LOAD_GROUP` (module comment above, campaign
    # 20261003_041039 optimize round 7 / P4) does the same hoist for the
    # gate/dgate two-plane body, gated on `(gate or dgate)` instead of
    # `XSTR is not None` so the two knobs can never collide; the production
    # one-plane pack (`gate=False, dgate=False, XSTR=None`) still falls
    # through to the untouched `1`.
    _grp = _GATE_DGATE_LOAD_GROUP if (gate or dgate) else (_DG_LOAD_GROUP if XSTR is not None else 1)
    if gate or dgate:
        # Thread `tid` owns i32 words `_wc0.._wc0+3` of EVERY row it loads --
        # the load loop's `wc = tw % _TCW` below is chunk-invariant because
        # `_TCW | BLK * 4` -- so the gate / y_bias vector for its 8 columns
        # is one `dwordx4` per TILE, not one per chunk.
        _wc0 = (tid * 4) % _TCW
    if gate:
        _gw = buffer_ops.buffer_load(
            gsrc, (rblk // lmb2) * (xpad >> 1) + c0w + _wc0, vec_width=4, dtype=T.i32
        )
    if dgate:
        # y_bias, like gate, is one vector per tile (optimize round 5 / P2).
        # `_dacc` holds this thread's running fp32 partial sum, over the
        # tile's rows, for each of its 8 columns (4 lanes x 2 f32),
        # accumulated BEFORE `_finish_microblock` -- same early-accumulate
        # shape as BSUM's own `acc` (see the module comment above this
        # function for why that keeps the live range to one f32, not 32).
        _ybw = buffer_ops.buffer_load(ybsrc, c0w + _wc0, vec_width=4, dtype=T.i32)
        _dacc = [
            Vec.from_elements([fx.Float32(0.0), fx.Float32(0.0)], fx.Float32) for _ in range_constexpr(4)
        ]
    for _b in range_constexpr(_NLOAD // _grp):
        staged = []
        for _j in range_constexpr(_grp):
            chunk = _b * _grp + _j
            tw = chunk * (BLK * 4) + tid * 4
            tr = tw // _TCW
            wc = tw % _TCW
            goff = tr * (xstr >> 1) + c0w + wc
            if legpack and not _own_in:
                goff = goff + (XOFF >> 1)
            vec = buffer_ops.buffer_load(rsrc, goff, vec_width=4, dtype=T.i32, cache_modifier=_ldcm)
            yvec = None
            if dgate:
                # Same `goff` as dY's own load above -- same row, same 8
                # columns, already vectorised and in the col phase's
                # existing order (optimize round 5 / P2; the precondition
                # `sb-dgelu-legpack` established for a fused read to cost
                # less than the traffic it deletes). `y_ld_cm` (optimize
                # round 8 / P5) is a SEPARATE cache-policy knob from `_ldcm`
                # above: dY's own load (and the dgelu legs' loads) keep
                # `_ldcm` unconditionally; only THIS read picks up the
                # per-shape table (`_GATE_Y_NT_TABLE` / `_pick_gate_y_cm`).
                yvec = buffer_ops.buffer_load(ysrc, goff, vec_width=4, dtype=T.i32, cache_modifier=y_ld_cm)
            pvec = None
            if const_expr(dgelu):
                pvec = (
                    vec
                    if "pr" in _skip
                    else buffer_ops.buffer_load(
                        prsrc,
                        tr * (PRSTR >> 1) + c0w + wc,
                        vec_width=4,
                        dtype=T.i32,
                        cache_modifier=_ldcm,
                    )
                )
            staged.append((tw, vec, pvec, yvec))
        for tw, vec, pvec, yvec in staged:
            if gate or dgate:
                # G = bf16(dY * gate) and/or dgate's running fp32 partial are
                # both computed HERE, at the load, before anything reaches
                # LDS -- so the ROW and COL phases below read the identical
                # staged G a materialised-G pack would have produced
                # (optimize round 4 / P1), and dgate's accumulator never
                # re-reads dY from HBM (optimize round 5 / P2). `d` (dY as
                # f32) is shared by both uses, computed once per word.
                words = []
                for q in range_constexpr(4):
                    d = _bf16_pair_to_f32(vec[q])
                    if dgate:
                        _dacc[q] = _dacc[q] + d * (_bf16_pair_to_f32(yvec[q]) + _bf16_pair_to_f32(_ybw[q]))
                    if gate:
                        g = d * _bf16_pair_to_f32(_gw[q])
                        words.append(rocdl.cvt_pk_bf16_f32(g[0], g[1]))
                if gate:
                    vec = Vec.from_elements(words, fx.Int32)
            if const_expr(dgelu):
                # G = bf16(d_act * dgelu'(preact)) computed here, at the load,
                # so BOTH the ROW and the COL phase below read it out of the
                # same staged LDS tile and stay byte-identical to packing a
                # materialised G.
                words = []
                for q in range_constexpr(4):
                    g = _dgelu_pair(_bf16_pair_to_f32(vec[q]), _bf16_pair_to_f32(pvec[q]))
                    words.append(rocdl.cvt_pk_bf16_f32(g[0], g[1]))
                vec = Vec.from_elements(words, fx.Int32)
            _lds_store_vec4(lds.buf.ptr, tw, vec)
    fx.barrier()

    # ---- ROW phase: byte-for-byte `_emit_dual_body`'s (row_2d=False,
    # padded=False, batched=False, row_sr=False) path -- untouched, no bias
    # fusion here (the bias gradient sums over ROWS, which is what the COL
    # phase already walks per column; fusing into the row phase would sum
    # over COLUMNS instead, the wrong axis).
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
        rwords, rbiased = _finish_microblock(rbits, True, scale_rounding_bias, None)
        gcmb = cblk * (_TC // 32) + cmb
        if "ro" not in _skip:
            _store_words_vec4(orsrc, r_row * (xpad >> 3) + gcmb * 4 + _row_off, rwords)
        if "rs" not in _skip:
            buffer_ops.buffer_store(
                arith.trunci(T.i8, rbiased & 0xFF), rscrsrc, r_row * (xpad >> 5) + gcmb + _rowsc_off
            )
        if ROW_SCP is not None:
            _pk_row.append((r_row, cmb, rbiased & 0xFF))

    # ---- COL phase + fused per-microblock column sum ----
    # Same GW=2 paired-lane compute as `_emit_dual_body`'s (col_2d=False)
    # branch -- zero added arithmetic to the pack itself, only the running
    # `acc` below is new.
    _GW = 2
    _PLC = BLK // _GW
    _NPASS = _TC // _PLC
    _NMB = _RMB // _GW
    _pl_g = tid & 1
    _pl_j = tid >> 1
    assert not tile_bsum or (_NMB == 1 and XSTR is None)
    for _p in range_constexpr(_NPASS):
        c_col = _pl_j + _p * _PLC
        half = c_col & 1
        cw = c_col >> 1
        for _mg in range_constexpr(_NMB):
            _mmb = _mg * _GW + _pl_g
            row0 = _mmb * 32
            cbits = []
            acc = fx.Float32(0.0)
            for row in range_constexpr(32):
                word = _lds_load1(lds.buf.ptr, (row0 + row) * _TCW + cw)
                fb = arith.select(half != 0, word & fx.Int32(-65536), word << 16)
                cbits.append(fb)
                # Accumulate INSIDE the load loop, before `_finish_microblock`
                # -- see the module-level comment above this function for why
                # that keeps the accumulator's live range to one f32 instead
                # of 32.
                acc = acc + arith.bitcast(T.f32, fb)
            cwords, cbiased = _finish_microblock(cbits, True, scale_rounding_bias, None)
            gmmb = rblk * _RMB + _mmb
            cob = c_col * (R >> 3) + gmmb * 4
            csoff = c_col * (R >> 5) + gmmb
            if "co" not in _skip:
                _store_words_vec4(corsrc, cob, cwords)
            if "cs" not in _skip:
                buffer_ops.buffer_store(arith.trunci(T.i8, cbiased & 0xFF), cscrsrc, csoff)
            if COL_SCP is not None:
                _pk_col.append((c_col, _mmb, cbiased & 0xFF))
            if "bs" not in _skip:
                if tile_bsum:
                    # Lanes 2j and 2j+1 hold the tile's two microblocks of the same
                    # column, so both end up with the same tile sum and store the
                    # same bits. sc1: the strip's last CTA may sit on another XCD.
                    acc = _dpp_add_f32(acc, _DPP_QUAD_SWAP1)
                    buffer_ops.buffer_store(
                        acc, bsrc, rblk * xpad + c_col, cache_modifier=16 if fold_nrblk else 0
                    )
                elif _batomic:
                    # P16(a) + lesson 7: every lane atomically adds its own
                    # raw 32-row partial into shard `gmmb & (_bshards-1)` of
                    # the mini-bias plane, NOT a single shared slot -- up to
                    # `_bshards` DISJOINT addresses per column instead of
                    # one, so concurrent blocks with different `gmmb` mostly
                    # land on different cache lines (lesson 7's fix for the
                    # measured single-slot regression; see the module-level
                    # `_DG_BIAS_ATOMIC` comment for the numbers). Same
                    # non-deterministic-ORDER/deterministic-VALUE SNR
                    # argument as the single-slot case: reduction order
                    # across shards (here) and within a shard (stage-2)
                    # already differs from torch's own sum() order, same as
                    # `_bfold`'s DPP fold and stage-2's DPP-row scan.
                    _buffer_atomic_add_f32(biasrc, (gmmb & _bmask) * xpad + c_col, acc)
                elif _bfold:
                    # The two microblock partials for THIS column live in lanes
                    # tid and tid^1 (`_mmb = _mg*_GW + (tid & 1)`, same
                    # `_pl_j` -> same `c_col`). One DPP quad_perm[1,0,3,2] add
                    # folds them; the odd lane then addresses past BSUM's SRD
                    # so its store is dropped with no divergent branch.
                    tot = _dpp_add_f32(acc, _DPP_QUAD_SWAP1)
                    goff_b = rblk * xpad + c_col
                    buffer_ops.buffer_store(tot, bsrc, arith.select(_pl_g == 0, goff_b, fx.Int32(_OOB)))
                else:
                    buffer_ops.buffer_store(acc, bsrc, gmmb * xpad + c_col)

    if ROW_SCP is not None:
        _store_packed_scales(
            lds, tid, _pk_row, _pk_col, ROW_SCP, COL_SCP, R, xpad, r0, rblk, cblk, _colbase, _rowsc_off
        )

    if dgate:
        _fold_dgate_partials(lds, tid, _dacc, dgsrc, rblk, xpad, lmb2)

    if fold_nrblk is not None:
        _bias_fold_tail(lds, tid, bsrc, BIAS, CNT, c0i, _colbase, xpad, fold_nrblk)


_PK_SCR = 2048  # LDS words; clear of `_bias_fold_tail`'s [0, _FOLD_FLAG]
_DGS_SCR = 4096  # LDS words for the dgate fold (optimize round 5 / P2);
# clear of `_PK_SCR` (2048..3071) and `_bias_fold_tail`'s arrival flag
# (word `_FOLD_FLAG` == 1024) -- same dead-staged-tile reuse precedent as
# `_store_packed_scales` below, just at a different offset.


def _fold_dgate_partials(lds, tid, dacc, dgsrc, rblk, xpad, lmb2=1):
    """Fold the `dgate` partial sums held by every lane (optimize round 5 /
    P2) into one fp32-per-column row of `DGS` (`[R/_TR, xpad]`, the exact
    grain `BSUM` has under `tile_bsum=True`). Each thread holds 8 fp32
    partials -- its own 8 columns (`_wc0.._wc0+3` word-pairs), accumulated
    over all `_TR` tile rows by `dacc` in the load loop. The 8 threads that
    together cover one column's full set of row-groups are 32 lanes apart
    (`tid`, `tid+32`, ..., `tid+224`: different waves), so the fold goes
    through LDS -- words `[_DGS_SCR, _DGS_SCR+2048)` REUSE the staged tile
    (dead by this point in program order: the col phase above has already
    issued every read of `buf` it will ever make, and the leading barrier
    below -- like `_store_packed_scales`'s own -- makes that visible to every
    wave before any thread overwrites those words), so this needs NO new
    LDS: the 5-workgroup/CU budget is untouched.

    `lmb2` / `_DGATE_ACC_ATOMIC` (optimize round 11 / P9): when the module
    flag is on, `DGS` is the PERSISTENT `[B, xpad]` accumulator
    (`_dgate_acc_plane`) and every one of the `lmb2` row-tile-blocks that
    share one batch sample `b = rblk // lmb2` must land on the SAME row, so
    the final write is `_buffer_atomic_add_f32` instead of an exclusive
    `buffer_store` -- correctness relies on nothing else (not even a
    different thread in THIS block) ever touching that row concurrently,
    which holds because every OTHER thread owns a disjoint column. Off, this
    reproduces the pre-P9 per-call `[R/_TR, xpad]` plane write byte-for-byte
    (`rblk` is already that plane's own row; `lmb2` is unused).
    """
    fx.barrier()
    _dgo = _DGS_SCR + (tid // 32) * _TC + ((tid * 4) % _TCW) * 2
    for _h in range_constexpr(2):
        _lds_store_vec4(
            lds.buf.ptr,
            _dgo + _h * 4,
            Vec.from_elements(
                [
                    arith.bitcast(T.i32, _raw(fx.Float32(dacc[2 * _h + (_k >> 1)][_k & 1])))
                    for _k in range_constexpr(4)
                ],
                fx.Int32,
            ),
        )
    fx.barrier()
    _ds = fx.Float32(arith.bitcast(T.f32, _raw(_lds_load1(lds.buf.ptr, _DGS_SCR + tid))))
    for _w in range_constexpr(1, 8):
        _ds = _ds + fx.Float32(arith.bitcast(T.f32, _raw(_lds_load1(lds.buf.ptr, _DGS_SCR + _w * _TC + tid))))
    if _DGATE_ACC_ATOMIC:
        _buffer_atomic_add_f32(dgsrc, (rblk // lmb2) * xpad + tid, _ds)
    else:
        buffer_ops.buffer_store(_ds, dgsrc, rblk * xpad + tid)


def _store_packed_scales(
    lds, tid, pk_row, pk_col, ROW_SCP, COL_SCP, R, xpad, r0, rblk, cblk, colbase, rowsc_off
):
    """Write this tile's row and col scales in the FP4 GEMM's packed A layout
    (`mxfp4_packed_scale_offset`). A 64-row x 256-col tile's 512 row scales fill
    exactly 128 packed dwords, and so do its 512 col scales, so the bytes go
    through LDS (one word each, at `4 * dword + byte`) and leave as whole dwords
    instead of 1024 scattered byte stores. Needs `_TR == 64`, `_TC == 256`."""
    fx.barrier()
    for r_row, cmb, v in pk_row:
        d = (r_row & 15) * 8 + (cmb & 3) * 2 + (cmb >> 2)
        _lds_store1(lds.buf.ptr, _PK_SCR + d * 4 + (r_row >> 4), v)
    for c_col, mmb, v in pk_col:
        d = ((c_col >> 6) * 2 + mmb) * 16 + (c_col & 15)
        _lds_store1(lds.buf.ptr, _PK_SCR + 512 + d * 4 + ((c_col >> 4) & 3), v)
    fx.barrier()
    w = [_lds_load1(lds.buf.ptr, _PK_SCR + tid * 4 + t) for t in range(4)]
    word = w[0] | (w[1] << 8) | (w[2] << 16) | (w[3] << 24)
    d = tid & 127
    # row half: d = (r, g, sub); dim = R rows, K = xpad
    r, g, sub = d >> 3, (d >> 1) & 3, d & 1
    kk = cblk + (rowsc_off >> 3)
    row_dw = (((r0 >> 7) * (xpad >> 8) + kk) * 64 + r) * 4 + g * 64 + ((r0 >> 6) & 1) * 2 + sub
    # col half: d = (wi_l, rr, gl, r); dim = xpad cols, K = R
    r, gl, rr, wil = d & 15, (d >> 4) & 1, (d >> 5) & 1, d >> 6
    g = (rblk & 1) * 2 + gl
    col_dw = (
        ((((colbase >> 7) + wil) * (R >> 8) + (rblk >> 2)) * 64 + r) * 4 + g * 64 + rr * 2 + ((rblk >> 1) & 1)
    )
    rsrc = _srd_at(ROW_SCP, fx.Int32(0), 4, R * (xpad >> 5))
    csrc = _srd_at(COL_SCP, fx.Int32(0), 4, R * (xpad >> 5))
    is_col = tid >= 128
    buffer_ops.buffer_store(word, rsrc, row_dw, mask=tid < 128)
    buffer_ops.buffer_store(word, csrc, col_dw, mask=is_col)


_FOLD_FLAG = 4 * _TC  # LDS word past the reduce's [4, _TC] f32 partials
_FOLD_U = 8  # vec4 loads in flight per thread in the last CTA's reduce


def _scf_if(cond, then_fn):
    from flydsl._mlir import ir
    from flydsl._mlir.dialects import scf

    op = scf.IfOp(_raw(cond), [], has_else=False)
    with ir.InsertionPoint(op.regions[0].blocks[0]):
        then_fn()
        scf.YieldOp([])


def _wait_vmcnt0():
    from flydsl._mlir.dialects import llvm

    llvm.inline_asm(fx.T.i32(), [], "s_waitcnt vmcnt(0)", "=r,~{memory}", has_side_effects=True)


def _bias_fold_tail(lds, tid, bsrc, BIAS, CNT, c0i, colbase, xpad, nrblk):
    """Stage 2 of the bias reduce, run by whichever CTA of a `_TC`-column strip
    stores its tile sums last. `CNT` holds one int32 arrival counter per strip,
    zeroed once and never reset: every call adds exactly `nrblk` (a power of
    two) per strip, so the arrival index is the count mod `nrblk`.

    The reduce reads the strip's `[nrblk, _TC]` tile sums coalesced (lane l of
    wave w: columns 4l..4l+3, rows w, w+4, ...), folds the four waves through
    LDS in wave order, and writes the bf16 bias directly. Deterministic: the
    order never depends on which CTA arrives last."""
    from flydsl._mlir import ir
    from flydsl._mlir.dialects import scf

    _wait_vmcnt0()
    fx.barrier()
    key = colbase // _TC
    cnt = buffer_ops.extract_base_index(CNT)

    def _arrive():
        _lds_store1(lds.buf.ptr, _FOLD_FLAG, atomic_add(cnt, key, fx.Int32(1)))

    _scf_if(tid == fx.Int32(0), _arrive)
    fx.barrier()
    old = _lds_load1(lds.buf.ptr, _FOLD_FLAG)
    last = (old & fx.Int32(nrblk - 1)) == fx.Int32(nrblk - 1)

    def _reduce():
        wave = tid // 64
        c4 = (tid % 64) * 4
        z = _raw(fx.Float32(0.0))
        loop = scf.ForOp(
            _raw(arith.index(0)), _raw(arith.index(nrblk // 4)), _raw(arith.index(_FOLD_U)), [z] * 4
        )
        with ir.InsertionPoint(loop.body):
            k0 = fx.Int32(arith.index_cast(T.i32, loop.induction_variable))
            a = [fx.Float32(v) for v in loop.inner_iter_args]
            vs = [
                Vec(
                    buffer_ops.buffer_load(
                        bsrc,
                        (wave + (k0 + u) * 4) * xpad + c4,
                        vec_width=4,
                        dtype=T.f32,
                        cache_modifier=16,
                    )
                )
                for u in range_constexpr(_FOLD_U)
            ]
            for v in vs:
                a = [a[i] + v[i] for i in range(4)]
            scf.YieldOp([_raw(x) for x in a])
        a = [fx.Float32(v) for v in loop.results]
        _lds_store_vec4(
            lds.buf.ptr,
            wave * _TC + c4,
            Vec.from_elements([arith.bitcast(T.i32, _raw(x)) for x in a], fx.Int32),
        )
        fx.barrier()
        s = fx.Float32(arith.bitcast(T.f32, _raw(_lds_load1(lds.buf.ptr, tid))))
        for w in range_constexpr(1, 4):
            s = s + fx.Float32(arith.bitcast(T.f32, _raw(_lds_load1(lds.buf.ptr, tid + w * _TC))))
        osrc = _srd_at(BIAS, c0i, 2, fx.Int32(_TC * 2))
        buffer_ops.buffer_store(arith.bitcast(T.i16, arith.truncf(T.bf16, _raw(s))), osrc, tid)

    _scf_if(last, _reduce)


def _build_dual_bias_kernel(col_locality=False, xcd_remap=False):
    """H16-recipe fused dual + bias-partial kernel. Thin wrapper, same shape
    as `_build_dual_kernel`, over `_emit_dual_bias_body`."""
    _DualSS = _make_dual_struct(False)

    @flyc.kernel(known_block_size=[BLK, 1, 1])
    def _dual_bias_kernel(
        X: fx.Tensor,
        ROW_OUT: fx.Tensor,
        ROW_SC: fx.Tensor,
        COL_OUT: fx.Tensor,
        COL_SC: fx.Tensor,
        BSUM: fx.Tensor,  # float32 [R/32, C] partial column sums
        R: fx.Int32,
        C: fx.Int32,
        SCALE_ROUNDING_BIAS: fx.Int32,
    ):
        lds = fx.SharedAllocator().allocate(_DualSS).peek()
        _emit_dual_bias_body(
            lds,
            fx.thread_idx.x,
            X,
            ROW_OUT,
            ROW_SC,
            COL_OUT,
            COL_SC,
            BSUM,
            R,
            C,
            fx.block_idx.x,
            SCALE_ROUNDING_BIAS,
            col_locality=col_locality,
            xcd_remap=xcd_remap,
        )

    return _dual_bias_kernel


def _build_dual_bias_launch(col_locality=False, xcd_remap=False):
    kern = _build_dual_bias_kernel(col_locality, xcd_remap)

    @flyc.jit
    def _dual_bias_launch(
        X: fx.Tensor,
        ROW_OUT: fx.Tensor,
        ROW_SC: fx.Tensor,
        COL_OUT: fx.Tensor,
        COL_SC: fx.Tensor,
        BSUM: fx.Tensor,
        R: fx.Int32,
        C: fx.Int32,
        SCALE_ROUNDING_BIAS: fx.Int32,
        grid_x: fx.Int32,
        stream: fx.Stream,
    ):
        kern(X, ROW_OUT, ROW_SC, COL_OUT, COL_SC, BSUM, R, C, SCALE_ROUNDING_BIAS).launch(
            grid=(grid_x, 1, 1), block=(BLK, 1, 1), stream=stream
        )

    return _dual_bias_launch


_DUAL_BIAS_LAUNCH = {}
_DUAL_BIAS_COMPILED = {}


def get_dual_bias_cast(R, C):
    """(compiled_fn, grid_x) for the H16-recipe fused dual+bias kernel at
    (R, C). Requires `dual_eligible`'s own alignment (R % 128 == 0,
    C % 256 == 0); callers must check that themselves (same contract as
    `get_dual_cast`). Reuses the SAME `_ORDER_TABLE` oracle lookup as the
    plain H16 dual/colq kernels (`_pick_block_order(R, C, True, True, False,
    False)`) so this kernel gets the identical tuned block schedule -- block
    order never changes a value, only which physical block computes which
    logical tile, so any measured delta against the plain dual is the bias
    fusion's cost alone, not a different schedule."""
    col_locality, xcd_remap = _pick_block_order(R, C, True, True, False, False)
    lk = (col_locality, xcd_remap)
    raw = _DUAL_BIAS_LAUNCH.get(lk)
    if raw is None:
        raw = _build_dual_bias_launch(col_locality, xcd_remap)
        _DUAL_BIAS_LAUNCH[lk] = raw
    key = (int(R), int(C))
    ent = _DUAL_BIAS_COMPILED.get(key)
    if ent is None:
        import torch

        x = torch.zeros((R, C // 2), dtype=torch.int32, device="cuda")
        ro = torch.zeros((R, C // 8), dtype=torch.int32, device="cuda")
        rs = torch.zeros((R, C // 32), dtype=torch.uint8, device="cuda")
        co = torch.zeros((C, R // 8), dtype=torch.int32, device="cuda")
        cs = torch.zeros((C, R // 32), dtype=torch.uint8, device="cuda")
        bs = torch.zeros((R // 32, C), dtype=torch.float32, device="cuda")
        grid_x = (R // _TR) * (C // _TC)
        stream = torch.cuda.current_stream()
        fn = flyc.compile(raw, *_static_layout(x, ro, rs, co, cs, bs), R, C, 1 << 21, grid_x, stream)
        ent = (fn, grid_x)
        _DUAL_BIAS_COMPILED[key] = ent
    return ent


# ---- Bias-fused dual: two-leg column-slice packing (optimize round 10 / P6,
# extended to the bias recipe). Thin legpack wrapper over
# `_emit_dual_bias_body`'s new `legpack`/`XPAD`/`XOFF` params, mirroring
# `_build_dual_legpack_kernel`/`_launch`/`get_dual_legpack_cast` above --
# see those docstrings for the shared rationale. The one structural
# difference from the plain-dual legpack: `BSUM` needs NO legpack-specific
# stage-2 kernel at all -- each leg just writes its own column-slice of one
# shared `[R/32, XPAD]` buffer (row-stride `XPAD`, per `_emit_dual_bias_
# body`'s docstring), and the EXISTING `get_stage2_bias_reduce` runs ONCE,
# after both legs, directly on that full buffer (see
# `_make_h16_dual_bias_legs_plan` below).
def _build_dual_bias_legpack_kernel(col_locality=False, xcd_remap=False):
    _DualSS = _make_dual_struct(False)

    @flyc.kernel(known_block_size=[BLK, 1, 1])
    def _dual_bias_legpack_kernel(
        X: fx.Tensor,
        ROW_OUT: fx.Tensor,
        ROW_SC: fx.Tensor,
        COL_OUT: fx.Tensor,
        COL_SC: fx.Tensor,
        BSUM: fx.Tensor,  # float32 [R/32, XPAD] shared partial column sums
        R: fx.Int32,
        C: fx.Int32,  # this leg's own local width
        XPAD: fx.Int32,
        XOFF: fx.Int32,
        SCALE_ROUNDING_BIAS: fx.Int32,
    ):
        lds = fx.SharedAllocator().allocate(_DualSS).peek()
        _emit_dual_bias_body(
            lds,
            fx.thread_idx.x,
            X,
            ROW_OUT,
            ROW_SC,
            COL_OUT,
            COL_SC,
            BSUM,
            R,
            C,
            fx.block_idx.x,
            SCALE_ROUNDING_BIAS,
            col_locality=col_locality,
            xcd_remap=xcd_remap,
            legpack=True,
            XPAD=XPAD,
            XOFF=XOFF,
        )

    return _dual_bias_legpack_kernel


def _build_dual_bias_legpack_launch(col_locality=False, xcd_remap=False):
    kern = _build_dual_bias_legpack_kernel(col_locality, xcd_remap)

    @flyc.jit
    def _dual_bias_legpack_launch(
        X: fx.Tensor,
        ROW_OUT: fx.Tensor,
        ROW_SC: fx.Tensor,
        COL_OUT: fx.Tensor,
        COL_SC: fx.Tensor,
        BSUM: fx.Tensor,
        R: fx.Int32,
        C: fx.Int32,
        XPAD: fx.Int32,
        XOFF: fx.Int32,
        SCALE_ROUNDING_BIAS: fx.Int32,
        grid_x: fx.Int32,
        stream: fx.Stream,
    ):
        kern(X, ROW_OUT, ROW_SC, COL_OUT, COL_SC, BSUM, R, C, XPAD, XOFF, SCALE_ROUNDING_BIAS).launch(
            grid=(grid_x, 1, 1), block=(BLK, 1, 1), stream=stream
        )

    return _dual_bias_legpack_launch


_DUAL_BIAS_LEGPACK_LAUNCH = {}
_DUAL_BIAS_LEGPACK_COMPILED = {}


def get_dual_bias_legpack_cast(R, C, XPAD):
    """(compiled_fn, grid_x) for one leg of a two-leg column-slice bias-fused
    dual pack -- same contract as `get_dual_legpack_cast`, plus a shared
    `[R/32, XPAD]` BSUM buffer instead of `[R/32, C]`. Order is looked up on
    THIS leg's own (R, C) via the same `_ORDER_TABLE` oracle (the bias fusion
    changes zero scheduling decisions, just like the plain dual's `get_dual_
    bias_cast` reuses the plain dual's own lookup)."""
    col_locality, xcd_remap = _pick_block_order(R, C, True, True, False, False)
    lk = (col_locality, xcd_remap)
    raw = _DUAL_BIAS_LEGPACK_LAUNCH.get(lk)
    if raw is None:
        raw = _build_dual_bias_legpack_launch(col_locality, xcd_remap)
        _DUAL_BIAS_LEGPACK_LAUNCH[lk] = raw
    key = (int(R), int(C), int(XPAD))
    ent = _DUAL_BIAS_LEGPACK_COMPILED.get(key)
    if ent is None:
        import torch

        x = torch.zeros((R, XPAD // 2), dtype=torch.int32, device="cuda")
        ro = torch.zeros((R, XPAD // 8), dtype=torch.int32, device="cuda")
        rs = torch.zeros((R, XPAD // 32), dtype=torch.uint8, device="cuda")
        co = torch.zeros((XPAD, R // 8), dtype=torch.int32, device="cuda")
        cs = torch.zeros((XPAD, R // 32), dtype=torch.uint8, device="cuda")
        bs = torch.zeros((R // 32, XPAD), dtype=torch.float32, device="cuda")
        grid_x = (R // _TR) * (C // _TC)
        stream = torch.cuda.current_stream()
        fn = flyc.compile(raw, *_static_layout(x, ro, rs, co, cs, bs), R, C, XPAD, 0, 1 << 21, grid_x, stream)
        ent = (fn, grid_x)
        _DUAL_BIAS_LEGPACK_COMPILED[key] = ent
    return ent


# ---- Tile-sum bias: `tile_bsum` halves BSUM to one row per tile (R/_TR), and
# the bias comes out as bf16 with no cast launch. Two ways to finish it:
#   "tail":   `_bias_fold_tail` inside the dual kernel -- no stage-2 launch, but
#             every CTA pays a store drain + device-scope atomic round trip;
#   "stage2": `_stage2_bias_bf16_kernel` over the tile sums.
# Measured on MI355X at the Flux shapes, "tail" only wins from ~8K columns up
# (narrow grids have too few CTAs per CU to hide the per-CTA arrival).
_FOLD_TAIL_MIN_C = 8192


def bias_fold_mode(R, XPAD):
    if os.environ.get("PRIMUS_MXFP4_BIAS_FOLD", "1") != "1":
        return None
    nrblk = R // _TR
    if XPAD >= _FOLD_TAIL_MIN_C and nrblk & (nrblk - 1) == 0 and nrblk % (4 * _FOLD_U) == 0:
        return "tail"
    return "stage2"


def _emit_b_scale_repack(tid, lbid, nbx, bdim, kx, kw, BX_RAW, BX_OUT, BW_RAW, BW_OUT):
    """The GEMM scale preshuffle's B map (``_CSTORE`` interleave, unpadded), for the two
    backward B operands at once: blocks [0, nbx) repack BX (``[bdim, kx/32]``), the rest
    BW (``[bdim, kw/32]``). Same per-thread gather as ``_build_mxfp4_preshuffle_kernel_ab``."""
    from primus_turbo.flydsl.gemm.gemm_mxfp4_kernel import _mxfp4_pack_cell

    is_w = lbid >= nbx
    local = arith.select(is_w, lbid - nbx, lbid)
    k128 = arith.select(is_w, kw >> 7, kx >> 7)
    x_in = _srd_at(BX_RAW, fx.Int32(0), 4, bdim * (kx >> 5))
    x_out = _srd_at(BX_OUT, fx.Int32(0), 4, bdim * (kx >> 5))
    w_in = _srd_at(BW_RAW, fx.Int32(0), 4, bdim * (kw >> 5))
    w_out = _srd_at(BW_OUT, fx.Int32(0), 4, bdim * (kw >> 5))
    rin = arith.select(is_w, w_in, x_in)
    rout = arith.select(is_w, w_out, x_out)
    gid = local * BLK + tid
    ok = gid < bdim * k128 // 16
    kk_n = k128 >> 1
    r = gid % 16
    e2 = gid // 16
    kk = e2 % kk_n
    wi = e2 // kk_n
    dws = []
    for r_region in range_constexpr(2):
        grp = (wi >> 1) * 4 + (wi & 1) + 2 * r_region
        for t in range_constexpr(4):
            row = grp * 64 + r * 4 + t
            dws.append(
                Vec(buffer_ops.buffer_load(rin, row * k128 + kk * 2, vec_width=2, dtype=T.i32, mask=ok))
            )
    words = _mxfp4_pack_cell(dws, 2, 4, 4)
    base = ((wi * kk_n + kk) * 64 + r) * 4
    for g in range_constexpr(4):
        buffer_ops.buffer_store(Vec.from_elements(words[g]), rout, base + g * 64, mask=ok)


def _build_dual_bias_tile_launch(col_locality, xcd_remap, fold_nrblk, legpack, packed):
    _DualSS = _make_dual_struct(False)

    @flyc.kernel(known_block_size=[BLK, 1, 1])
    def _dual_bias_tile_kernel(
        X: fx.Tensor,
        ROW_OUT: fx.Tensor,
        ROW_SC: fx.Tensor,
        COL_OUT: fx.Tensor,
        COL_SC: fx.Tensor,
        BSUM: fx.Tensor,  # float32 [R/_TR, XPAD] per-tile column sums
        BIAS: fx.Tensor,  # int16 view of the bf16 [XPAD] bias
        CNT: fx.Tensor,  # int32 [XPAD/_TC] arrival counters
        ROW_SCP: fx.Tensor,  # packed: GEMM-layout copies of ROW_SC / COL_SC (A operands)
        COL_SCP: fx.Tensor,
        BX_RAW: fx.Tensor,  # packed: canonical B scales in, GEMM layout out
        BX_OUT: fx.Tensor,
        BW_RAW: fx.Tensor,
        BW_OUT: fx.Tensor,
        R: fx.Int32,
        C: fx.Int32,
        XPAD: fx.Int32,
        XOFF: fx.Int32,
        SCALE_ROUNDING_BIAS: fx.Int32,
        GRID_MAIN: fx.Int32,
        NBX: fx.Int32,
        BDIM: fx.Int32,
    ):
        lds = fx.SharedAllocator().allocate(_DualSS).peek()
        tid = fx.thread_idx.x

        def _main(bid):
            _emit_dual_bias_body(
                lds,
                tid,
                X,
                ROW_OUT,
                ROW_SC,
                COL_OUT,
                COL_SC,
                BSUM,
                R,
                C,
                bid,
                SCALE_ROUNDING_BIAS,
                col_locality=col_locality,
                xcd_remap=xcd_remap,
                legpack=legpack,
                XPAD=XPAD if legpack else None,
                XOFF=XOFF if legpack else None,
                tile_bsum=True,
                fold_nrblk=fold_nrblk,
                BIAS=BIAS,
                CNT=CNT,
                ROW_SCP=ROW_SCP if packed else None,
                COL_SCP=COL_SCP if packed else None,
            )

        if const_expr(packed):
            bid = rocdl.readfirstlane(T.i32, fx.block_idx.x)
            _scf_if(bid < GRID_MAIN, lambda: _main(bid))
            _scf_if(
                bid >= GRID_MAIN,
                lambda: _emit_b_scale_repack(
                    tid, bid - GRID_MAIN, NBX, BDIM, R, XPAD, BX_RAW, BX_OUT, BW_RAW, BW_OUT
                ),
            )
        else:
            _main(fx.block_idx.x)

    @flyc.jit
    def _dual_bias_tile_launch(
        X: fx.Tensor,
        ROW_OUT: fx.Tensor,
        ROW_SC: fx.Tensor,
        COL_OUT: fx.Tensor,
        COL_SC: fx.Tensor,
        BSUM: fx.Tensor,
        BIAS: fx.Tensor,
        CNT: fx.Tensor,
        ROW_SCP: fx.Tensor,
        COL_SCP: fx.Tensor,
        BX_RAW: fx.Tensor,
        BX_OUT: fx.Tensor,
        BW_RAW: fx.Tensor,
        BW_OUT: fx.Tensor,
        R: fx.Int32,
        C: fx.Int32,
        XPAD: fx.Int32,
        XOFF: fx.Int32,
        SCALE_ROUNDING_BIAS: fx.Int32,
        GRID_MAIN: fx.Int32,
        NBX: fx.Int32,
        BDIM: fx.Int32,
        grid_x: fx.Int32,
        stream: fx.Stream,
    ):
        _dual_bias_tile_kernel(
            X,
            ROW_OUT,
            ROW_SC,
            COL_OUT,
            COL_SC,
            BSUM,
            BIAS,
            CNT,
            ROW_SCP,
            COL_SCP,
            BX_RAW,
            BX_OUT,
            BW_RAW,
            BW_OUT,
            R,
            C,
            XPAD,
            XOFF,
            SCALE_ROUNDING_BIAS,
            GRID_MAIN,
            NBX,
            BDIM,
        ).launch(grid=(grid_x, 1, 1), block=(BLK, 1, 1), stream=stream)

    return _dual_bias_tile_launch


_DUAL_BIAS_TILE_LAUNCH = {}
_DUAL_BIAS_TILE_COMPILED = {}


def get_dual_bias_tile_cast(R, C, XPAD, legpack, tail, bdim=None):
    """(compiled_fn, grid_x) for the tile-sum dual kernel over the column slice
    `[:, XOFF:XOFF+C]` of an `[R, XPAD]` tensor (`XPAD == C`, XOFF 0 unless
    `legpack`). Same block order as `get_dual_bias_cast`. `bdim` (the backward
    B operands' row count) selects the packed-scale build; `grid_x` is the main
    grid only, the caller appends the B-repack blocks."""
    col_locality, xcd_remap = _pick_block_order(R, C, True, True, False, False)
    fold_nrblk = R // _TR if tail else None
    packed = bdim is not None
    lk = (col_locality, xcd_remap, fold_nrblk, bool(legpack), packed)
    raw = _DUAL_BIAS_TILE_LAUNCH.get(lk)
    if raw is None:
        raw = _build_dual_bias_tile_launch(col_locality, xcd_remap, fold_nrblk, bool(legpack), packed)
        _DUAL_BIAS_TILE_LAUNCH[lk] = raw
    key = (int(R), int(C), int(XPAD), bool(legpack), bool(tail), bdim)
    ent = _DUAL_BIAS_TILE_COMPILED.get(key)
    if ent is None:
        import torch

        def z(*shape, dtype=torch.uint8):
            return torch.zeros(shape, dtype=dtype, device="cuda")

        x = z(R, XPAD // 2, dtype=torch.int32)
        ro = z(R, XPAD // 8, dtype=torch.int32)
        rs = z(R, XPAD // 32)
        co = z(XPAD, R // 8, dtype=torch.int32)
        cs = z(XPAD, R // 32)
        bs = z(R // _TR, XPAD, dtype=torch.float32)
        bias = z(XPAD, dtype=torch.int16)
        cnt = z(XPAD // _TC, dtype=torch.int32)
        if packed:
            extra = (
                z(R, XPAD // 32),
                z(XPAD, R // 32),
                z(bdim, R // 32),
                z(bdim, R // 32),
                z(bdim, XPAD // 32),
                z(bdim, XPAD // 32),
            )
        else:
            extra = (cnt,) * 6
        grid_x = (R // _TR) * (C // _TC)
        stream = torch.cuda.current_stream()
        fn = flyc.compile(
            raw,
            *_static_layout(x, ro, rs, co, cs, bs, bias, cnt, *extra),
            R,
            C,
            XPAD,
            0,
            1 << 21,
            grid_x,
            0,
            0 if bdim is None else bdim,
            grid_x,
            stream,
        )
        ent = (fn, grid_x)
        _DUAL_BIAS_TILE_COMPILED[key] = ent
    return ent


_BIAS_FOLD_COUNTERS = {}


def _bias_fold_counters(R, XPAD, device, stream):
    """Arrival counters for one (shape, stream). Launches on one stream are
    serialized, which is what keeps every strip's count a multiple of `nrblk`
    between calls; two streams must never share a buffer."""
    import torch

    key = (int(R), int(XPAD), device, stream)
    buf = _BIAS_FOLD_COUNTERS.get(key)
    if buf is None:
        buf = torch.zeros((XPAD // _TC,), dtype=torch.int32, device=device)
        _BIAS_FOLD_COUNTERS[key] = buf
    return buf


_H16_DUAL_BIAS_LEGS_PLAN = {}


def _make_h16_dual_bias_legs_plan(R, leg_widths, fp4_dtype, scale_rounding_mode=0):
    """Generative closure for `flydsl_quant_mxfp4_h16_dual_bias_legs`, same
    shape as `_make_h16_dual_legs_plan` plus the stage-2 bias fold -- called
    ONCE on the full `[R/32, XPAD]` BSUM buffer after every leg's kernel has
    written its own column-slice (see the module comment above `_build_dual_
    bias_legpack_kernel`)."""
    import torch

    C = sum(leg_widths)
    scale_rounding_bias = _mxfp4_scale_rounding_bias(scale_rounding_mode)
    mode = bias_fold_mode(R, C)
    if mode is not None:
        return _make_h16_dual_bias_tile_plan(R, leg_widths, fp4_dtype, scale_rounding_bias, mode)
    legs = []
    xoff = 0
    for w in leg_widths:
        fn, grid_x = get_dual_bias_legpack_cast(R, w, C)
        legs.append((fn, grid_x, int(w), int(xoff)))
        xoff += w
    ro_shape = (R, C // 8)
    rs_shape = (R, C // 32)
    co_shape = (C, R // 8)
    cs_shape = (C, R // 32)
    bs_shape = (R // 32, C)
    i32 = torch.int32
    u8 = torch.uint8
    f32 = torch.float32
    e8m0 = torch.float8_e8m0fnu
    raw_stream = torch._C._cuda_getCurrentRawStream
    P = R // 32
    s2_fn, s2_gx = get_stage2_bias_reduce(P, C)

    def _plan(x_bf16):
        x_i32 = x_bf16.view(i32)
        ro = x_bf16.new_empty(ro_shape, dtype=i32)
        rs = x_bf16.new_empty(rs_shape, dtype=u8)
        co = x_bf16.new_empty(co_shape, dtype=i32)
        cs = x_bf16.new_empty(cs_shape, dtype=u8)
        bs = x_bf16.new_empty(bs_shape, dtype=f32)
        stream = raw_stream(x_bf16.device.index)
        for fn, grid_x, w, off in legs:
            fn(x_i32, ro, rs, co, cs, bs, R, w, C, off, scale_rounding_bias, grid_x, stream)
        bias_f32 = x_bf16.new_empty((C,), dtype=f32)
        s2_fn(bs, bias_f32, P, C, s2_gx, stream)
        bias = bias_f32.to(x_bf16.dtype)
        return (
            ro.view(u8).view(fp4_dtype),
            rs.view(e8m0),
            co.view(u8).view(fp4_dtype),
            cs.view(e8m0),
            bias,
        )

    return _plan


def _make_h16_dual_bias_tile_plan(R, leg_widths, fp4_dtype, scale_rounding_bias, mode, bdim=None):
    import torch

    C = sum(leg_widths)
    legpack = len(leg_widths) > 1
    tail = mode == "tail"
    packed = bdim is not None
    legs = []
    xoff = 0
    for w in leg_widths:
        fn, grid_x = get_dual_bias_tile_cast(R, w, C, legpack, tail, bdim)
        legs.append((fn, grid_x, int(w), int(xoff)))
        xoff += w
    # The B repack rides on the first leg's launch.
    nbx = nbw = 0
    if packed:
        nbx = -(-bdim * (R // 128) // (16 * BLK))
        nbw = -(-bdim * (C // 128) // (16 * BLK))
    i32 = torch.int32
    u8 = torch.uint8
    e8m0 = torch.float8_e8m0fnu
    raw_stream = torch._C._cuda_getCurrentRawStream
    if not tail:
        s2_fn, s2_gx = get_stage2_bias_reduce(R // _TR, C, bf16_out=True)

    def _plan(x_bf16, bx_scale=None, bw_scale=None):
        x_i32 = x_bf16.view(i32)
        ro = x_bf16.new_empty((R, C // 8), dtype=i32)
        rs = x_bf16.new_empty((R, C // 32), dtype=u8)
        co = x_bf16.new_empty((C, R // 8), dtype=i32)
        cs = x_bf16.new_empty((C, R // 32), dtype=u8)
        bs = x_bf16.new_empty((R // _TR, C), dtype=torch.float32)
        bias = x_bf16.new_empty((C,))
        stream = raw_stream(x_bf16.device.index)
        cnt = _bias_fold_counters(R, C, x_bf16.device, stream)
        bias_i16 = bias.view(torch.int16)
        if packed:
            rsp = torch.empty_like(rs)
            csp = torch.empty_like(cs)
            bxr = bx_scale.contiguous().view(u8)
            bwr = bw_scale.contiguous().view(u8)
            bxp = torch.empty_like(bxr)
            bwp = torch.empty_like(bwr)
            extra = (rsp, csp, bxr, bxp, bwr, bwp)
        else:
            extra = (cnt,) * 6
        for i, (fn, grid_x, w, off) in enumerate(legs):
            nb = nbx + nbw if i == 0 else 0
            fn(
                x_i32,
                ro,
                rs,
                co,
                cs,
                bs,
                bias_i16,
                cnt,
                *extra,
                R,
                w,
                C,
                off,
                scale_rounding_bias,
                grid_x,
                nbx,
                bdim or 0,
                grid_x + nb,
                stream,
            )
        if not tail:
            s2_fn(bs, bias_i16, R // _TR, C, s2_gx, stream)
        out = (
            ro.view(u8).view(fp4_dtype),
            rs.view(e8m0),
            co.view(u8).view(fp4_dtype),
            cs.view(e8m0),
            bias,
        )
        if packed:
            out = out + (rsp.view(e8m0), csp.view(e8m0), bxp.view(e8m0), bwp.view(e8m0))
        return out

    return _plan


_H16_DUAL_BIAS_PACKED_PLAN = {}


# Resolved once, eagerly, by `warm_packed_scale_api` below. They are read from inside
# `_MXFP4LinearFunction.backward`, which is traced, and everything that resolves them -- the
# module import, the `functools.cache` dict write, upstream's own lazy `_MXFP4_NCU.append` --
# is a mutation that tracing rejects. "unset" means the warm-up never ran, which disables the
# feature rather than resolving it in the wrong place.
_PACKED_API = "unset"
_PACKED_GEMM_MOD = None

# Upstream's packed-scale interface is OFF by default. The overlay interface is unaffected: it
# is detected the same way it always was and needs none of this, because it takes the pack at
# any shape and asks the layout nothing.
_TURBO_PACKED = os.environ.get("FLUX_MXFP4_TURBO_PACKED", "0") == "1"


def warm_packed_scale_api():
    """Resolve the packed-scale interface and warm what reading it would otherwise mutate.

    Called from `_init_turbo`, which is eager. Three separate things here are side effects that
    a traced caller may not perform, and all three have failed a run: importing the GEMM module,
    filling a `functools.cache`, and -- the one that took run 37184679793 -- upstream's
    `_mxfp4_ncu()`, which appends the device's CU count to a module-level list on first call and
    so reads as "Attempted to mutate ListVariable(length=0)" under tracing. Calling it once here
    leaves the list populated, after which every helper below is pure arithmetic.

    "overlay" is the P3 copy's `a_scale_packed=`/`b_scale_packed=` kwargs; "turbo" is upstream's
    own equivalent, `scales_prepacked=True` reached through gemm_fp4_impl's `preshuffled=`.
    Preferring the overlay when both are present keeps a P3 image on the path it was measured
    with. "turbo" additionally requires the three layout helpers `_turbo_packed_layout_agrees`
    asks: without them the layout cannot be checked, and an unchecked packed scale is silent
    corruption rather than an error.
    """
    global _PACKED_API, _PACKED_GEMM_MOD
    import primus_turbo.flydsl.gemm.gemm_mxfp4_kernel as g

    _PACKED_GEMM_MOD = g
    if hasattr(g, "mxfp4_packed_scale_offset"):
        _PACKED_API = "overlay"
    elif _TURBO_PACKED and all(
        hasattr(g, n)
        for n in (
            "preshuffle_mxfp4_scales",
            "mxfp4_packed_scale_block_n",
            "mxfp4_packed_scale_block_m",
            "mxfp4_packed_scale_ilv",
        )
    ):
        _PACKED_API = "turbo"
        g._mxfp4_ncu()
    else:
        _PACKED_API = None
    return _PACKED_API


def _gemm_takes_packed_scales():
    return _PACKED_API not in (None, "unset")


def _turbo_packed_layout_agrees(M, N, K):
    """Does our packed layout equal the one upstream's GEMM reads back for this shape?

    Upstream tiles the packed layout and picks the tile from the shape alone, so the two agree
    on some shapes and not others -- a 256 tile matches byte for byte, a narrower one does not
    even match in size (M=2048 N=3072 K=12288 gives upstream a third more extent). Reading one
    layout as the other yields plausible numbers rather than an error, so this is asked rather
    than assumed.

    Asked through the three helpers upstream exposes for precisely this caller -- its own
    docstring says "a direct-pack caller needs this ... the tile is what sets B's packed scale
    group" -- so this is their API answering about their layout, not a copy of their tile
    heuristics that would rot the next time one is retuned.

    Pure Python on purpose, and uncached on purpose. This is reached from inside a compiled
    region, where every way of making it cheaper is a side effect: the first version allocated
    scales and compared bytes against `preshuffle_mxfp4_scales` (run 37182610367), the second
    memoised with `functools.cache` and imported the module here (runs 37183081545, 37184679793).
    It is a handful of integer divisions; it does not need either. The byte-exact comparison
    still exists as an offline check (.tmp_cmp/packmatch.py), which is where it belongs.
    """
    g = _PACKED_GEMM_MOD

    if g.mxfp4_packed_scale_block_n(M, N, K) != 256 or g.mxfp4_packed_scale_block_m(M, N, K) != 256:
        return False
    # The interleave only exists on the 256-wide tile and is further conditional on the C store
    # folding, which depends on K. Our pack always writes the interleaved layout, so an
    # un-interleaved shape is not ours to serve. accum=False: the packed scales only reach
    # `_mxfp4_mm`, never the accumulating GEMM.
    Kw = (K + 255) // 256 * 256
    ilv = g.mxfp4_packed_scale_ilv(
        Kw, out_fp16=False, accum=False, k_real=(None if K == Kw else K), block_n=256
    )
    return ilv == getattr(g, "_MXFP4_PACK_ILV", 4)


def dual_bias_packed_eligible(R, C, bdim):
    if not (
        bias_fold_mode(R, C) is not None
        and R % 256 == 0
        and C % 256 == 0
        and bdim % 256 == 0
        and _gemm_takes_packed_scales()
    ):
        return False
    if _PACKED_API == "turbo":
        # The two backward GEMMs these scales feed -- Dgrad (R, bdim, C) and Wgrad (C, bdim, R)
        # -- get a pack each, so EITHER qualifying is worth producing them for, and the GEMM
        # that cannot read ours falls back to canonical on its own in `_mxfp4_mm`. Demanding
        # both would switch the pack off almost everywhere: upstream's tile is narrower at the
        # large K the Wgrad contracts over, so of the twelve shapes this recipe issues, the
        # seven that qualify are nearly all Dgrad.
        return _turbo_packed_layout_agrees(R, bdim, C) or _turbo_packed_layout_agrees(C, bdim, R)
    return True


def flydsl_quant_mxfp4_h16_dual_bias_packed(
    x_bf16, leg_widths, fp4_dtype, bx_scale, bw_scale, scale_rounding_mode=0
):
    """`flydsl_quant_mxfp4_h16_dual_bias(_legs)` for a Linear backward, plus the scales of
    both backward GEMMs in the FP4 GEMM's packed layout (`gemm_mxfp4_flydsl_kernel`'s
    `a_scale_packed`/`b_scale_packed`), so neither GEMM launches its scale preshuffle:
    row/col scale as the Dgrad/Wgrad A operands, and `bx_scale` (activation col pack,
    `[bdim, R/32]`) / `bw_scale` (weight col pack, `[bdim, C/32]`) repacked as the
    Wgrad/Dgrad B operands. Returns the five usual outputs plus
    `(row_scale_p, col_scale_p, bx_scale_p, bw_scale_p)`. Caller checks
    `dual_bias_packed_eligible` first."""
    R, C = x_bf16.shape
    bdim = int(bx_scale.shape[0])
    assert sum(leg_widths) == C and tuple(bw_scale.shape) == (bdim, C // 32)
    assert tuple(bx_scale.shape) == (bdim, R // 32)
    key = (int(R), tuple(int(w) for w in leg_widths), bdim, fp4_dtype, int(scale_rounding_mode))
    plan = _H16_DUAL_BIAS_PACKED_PLAN.get(key)
    if plan is None:
        plan = _make_h16_dual_bias_tile_plan(
            R,
            tuple(leg_widths),
            fp4_dtype,
            _mxfp4_scale_rounding_bias(scale_rounding_mode),
            bias_fold_mode(R, C),
            bdim,
        )
        _H16_DUAL_BIAS_PACKED_PLAN[key] = plan
    return plan(x_bf16, bx_scale, bw_scale)


# ---- Gate-fused dual-bias-packed: gate*dY folded into the bf16 load -------
# (optimize round 4 / P1.) Every gated residual in Flux is
# `x = x + gate * linear(h)`; on the way back the compiled graph currently
# materialises `G = gate * dY` (Inductor), packs G (this file), then
# re-reads dY for `dgate` (Inductor). `_emit_dual_bias_body`'s `gate`/`GATE`/
# `lmb2` params above scale dY by `gate[b, c]` at the SAME bf16 load the
# plain dual-bias-packed body already does, before anything reaches LDS, so
# the materialised `bf16(gate * dY)` is never written at all -- the ROW and
# COL phases both read the identical staged G out of LDS a materialised-G
# pack would have produced.
#
# (Optimize round 5 / P2.) The SAME load ALSO now accumulates `dgate`'s
# partial sum: `_emit_dual_bias_body`'s `Y`/`YBIAS`/`DGS`/`dgate` params read
# the GEMM output and its bias at the identical bf16 load and fold
# `fp32(dY) * (fp32(y) + fp32(y_bias))` per column into `DGS`
# (`[R/_TR, C]` f32), so the gate gradient's reduction -- which used to
# re-read the full `[R, C]` dY and `y` a second time (Inductor) -- now reads
# neither on its own; it only has to fold `DGS` down to `[B, C]`
# (`get_merged_fold` below, merged into one launch with the bias stage-2
# reduce as of optimize round 6 / P3).
#
# Thin wrapper, same shape as `_build_dual_bias_tile_launch(legpack=False,
# packed=True, fold_nrblk=None)` plus GATE/Y/YBIAS/DGS tensors -- a SEPARATE
# launch builder, so the production dual-bias-packed launch's trace, cache
# key and ISA are untouched: the `gate=False, dgate=False` default path
# through `_emit_dual_bias_body` is textually identical to the pre-existing
# body (every new line is behind `if gate:`/`if dgate:`), and
# `_build_dual_bias_tile_launch` itself is not modified at all.
#
# ---- P5: per-shape non-temporal cache policy on `y`'s load ONLY (optimize
# round 8). `y` (the GEMM output dgate's accumulator reads alongside dY, at
# the SAME `goff`) is read exactly once per element and never re-read within
# the kernel -- the same shape of argument `_DG_LD_CM` above already uses
# for the dgelu legs' X/preact loads -- but unlike `_GATE_DGATE_LOAD_GROUP`
# (P4, a uniform win at all three scored shapes) this one measures with
# OPPOSITE signs at the two dY shapes this recipe hits (goal.md Sec C.5/D.1,
# isolated prototype; re-confirmed THIS round on the live, P4-stacked tree,
# N=12 interleaved A/B per shape, kb/flydsl/pitfalls.md protocol): `nt`
# (cache_modifier=2) on `y` is a clean win at `single_linear2`'s
# (16384, 3072) -- median 0.06107ms -> 0.06022ms, -1.4% -- that shape's 96
# MiB-per-call `y` plane oversubscribes the L2 (44.6% hit rate measured in
# goal.md Sec C.4), so evicting it on first use helps -- and a clean LOSS at
# `double_proj`/`double_fc2`'s shared (8192, 3072) dY shape (dp: median
# 0.03324ms -> 0.03474ms, +4.5%; fc2: 0.03349ms -> 0.03627ms, +8.3%), where
# `y`'s lines still have reuse value the `nt` hint throws away. A single
# module constant (`_GATE_DGATE_LOAD_GROUP`'s own idiom) cannot win both
# signs at once; this has to be a per-`(R, C)` table lookup, exactly like
# `_ORDER_TABLE` / `_pick_block_order` above.
#
# cache_modifier=3 (GLC+SLC, i.e. `nt` plus the L1-bypass coherency bit) was
# swept alongside 0/2 this round at all three shapes: it tracks 2's sign
# everywhere (dp +4.7%, fc2 +8.2% vs 0 -- same loss; sl2 -2.7% vs 0 -- same
# win) and does not beat 2 head-to-head at sl2 (median 0.06021ms vs
# 0.06031ms, within noise) -- the extra GLC bit buys nothing here, so 2
# stays the table's value, not 3.
#
# `y_ld_cm` (threaded through `_emit_dual_bias_body`, above) only ever
# reaches the SECOND (`Y`) `buffer_load` inside the gate/dgate body's load
# loop -- `_ldcm` keeps gating dY's own load (and the dgelu legs' loads)
# completely unchanged, so this knob cannot affect `bit_exact` /
# `bias_bit_exact` by construction (a cache-policy bit never changes a
# loaded value) and cannot affect any OTHER caller: every other
# `_emit_dual_bias_body` call site leaves the new `y_ld_cm` parameter at its
# default (0, ordinary policy == pre-P5 behaviour), and the parameter is
# dead code whenever `dgate=False` (the only case `yvec` is ever loaded at
# all) regardless of its value.
#
# cache_modifier encoding, same as `_DG_LD_CM` above: bit 1 of the MUBUF aux
# field is `nt` (non-temporal); 0 is the ordinary policy.
_GATE_Y_NT_TABLE = {
    (8192, 3072): 0,  # double_proj / double_fc2 share this dY shape -- `nt`
    # on y measured a LOSS here; keep the ordinary policy.
    (16384, 3072): 2,  # single_linear2 -- `nt` on y measured a WIN; evicting
    # y's lines on first use helps the dY/output lines that ARE reused.
}


def _pick_gate_y_cm(R, C):
    """cache_modifier for the gate/dgate body's Y load only -- oracle
    lookup against `_GATE_Y_NT_TABLE`, falling back to 0 (ordinary policy,
    == pre-P5 behaviour) for any (R, C) this table has not measured, so an
    unseen shape never regresses relative to the pre-P5 baseline."""
    return _GATE_Y_NT_TABLE.get((int(R), int(C)), 0)


# ---- P9: persistent [B, C] f32 dgate accumulator (optimize round 11) ------
# `DGS` was a FRESH `[R/_TR, C]` f32 plane every call (`_make_gate_plan`):
# every one of the `lmb2` row-tile-blocks that share one batch sample owns
# its OWN exclusive row (no contention, plain `buffer_store`), and the
# merged fold's dgate half (`_build_merged_fold_launch`) then reduces `lmb2`
# rows per sample with a `range_constexpr(grp)` strided-read loop. That is
# `(R/_TR - B)` extra rows written and read every call for nothing but
# feeding a loop the atomic below can skip entirely.
#
# Collapsing `DGS` to a PERSISTENT `[B, C]` buffer and having the `lmb2`
# contributing blocks ATOMICALLY add into the SAME row `b = rblk // lmb2`
# removes both: the tile kernel's write touches only `B` distinct rows
# (fewer unique cachelines live across the kernel's whole run, same
# footprint-shrink argument the already-shipped `_DG_BIAS_ATOMIC` mini-bias
# plane above relies on), and the fold's dgate block drops its `grp`-deep
# loop to a single load -- the sum is already complete in HBM.
#
# Contention: `lmb2` (4 or 8 at this file's three scored shapes) writers can
# land on the SAME (b, c) address. That is STRICTLY FEWER than the
# already-shipped `_DG_BIAS_ATOMIC`/`_DG_BIAS_SHARDS` mechanism tolerates
# (shards 2/4/8/16, i.e. `(R/32)/_DG_BIAS_SHARDS` realized contenders per
# shard -- tens, not single digits -- and STILL a measured win; only
# `_DG_BIAS_SHARDS=1`, a single GLOBAL slot with ALL `R/32` writers landing
# on literally one address, regressed +154..+168%, see that mechanism's own
# module comment above `_dgelu_pair`). `[B, C]` is a semantically-exact
# shard -- one row per real output sample, never folded further -- so unlike
# that mini-bias plane this needs NO stage-2 reduction loop at all.
#
# `DGS` is cached per `(B, C, device, stream)`, same shape as
# `_bias_fold_counters` above and the same invariant: launches on one stream
# are serialized, so the merged fold kernel's own self-clear (write 0.0 back
# to every cell it just read, in the SAME pass that emits bf16 `dgate` --
# see `_build_merged_fold_launch`'s `_dgate_block`) is what keeps the plane
# zeroed for the NEXT call with no per-call `zero_()`; two streams must
# never share a buffer. Zeroed once, at allocation, for the very first call
# ever made at a given shape.
#
# `dgate` is SNR-gated, not bit-exact (`task.yaml`'s own
# `dgate_snr_margin_db`), so this flag can only ever change the SUMMATION
# ORDER of an already-non-bit-exact output across the `lmb2` partials --
# the identical non-determinism `_buffer_atomic_add_f32`'s own docstring
# already accepts for the bias leg, at a per-address contention level this
# one is even lower than.
_DGATE_ACC_ATOMIC = os.environ.get("PRIMUS_TURBO_DGATE_ACC_ATOMIC", "1") != "0"
# False reproduces the pre-P9 per-call [R/_TR, C] plane byte-for-byte (every
# new branch below is gated on this). Env-var override (default True) only
# to let THIS round's own A/B script flip it through the unmodified bench.py
# ruler via a fresh process per sample -- no effect on the orchestrator's
# own plain `bash bench.sh` invocation, which always measures True.
_DGATE_ACC_PLANES = {}


def _dgate_acc_plane(B, C, device, stream):
    """Persistent `[B, C]` f32 dgate accumulator for one (shape, stream)
    (optimize round 11 / P9) -- same caching shape as `_bias_fold_counters`
    above: allocate-and-zero once, then hand back the SAME tensor on every
    later call for this exact key."""
    import torch

    key = (int(B), int(C), device, stream)
    buf = _DGATE_ACC_PLANES.get(key)
    if buf is None:
        buf = torch.zeros((B, C), dtype=torch.float32, device=device)
        _DGATE_ACC_PLANES[key] = buf
    return buf


def _build_gate_tile_launch(col_locality, xcd_remap, lmb2, y_nt):
    _DualSS = _make_dual_struct(False)

    @flyc.kernel(known_block_size=[BLK, 1, 1])
    def _dual_bias_gate_tile_kernel(
        X: fx.Tensor,
        ROW_OUT: fx.Tensor,
        ROW_SC: fx.Tensor,
        COL_OUT: fx.Tensor,
        COL_SC: fx.Tensor,
        BSUM: fx.Tensor,  # float32 [R/_TR, C] per-tile column sums (of G, not dY)
        BIAS: fx.Tensor,  # int16 view of the bf16 [C] bias (sum of G over rows)
        CNT: fx.Tensor,  # int32 [C/_TC] arrival counters
        ROW_SCP: fx.Tensor,  # packed: GEMM-layout copies of ROW_SC / COL_SC (A operands)
        COL_SCP: fx.Tensor,
        BX_RAW: fx.Tensor,  # packed: canonical B scales in, GEMM layout out
        BX_OUT: fx.Tensor,
        BW_RAW: fx.Tensor,
        BW_OUT: fx.Tensor,
        GATE: fx.Tensor,  # bf16 [B, C] viewed i32, one row per batch sample
        Y: fx.Tensor,  # bf16 [R, C] viewed i32, the GEMM output (dgate input)
        YBIAS: fx.Tensor,  # bf16 [C] viewed i32, Y's bias (dgate input)
        DGS: fx.Tensor,  # float32 dgate partial sums: [R/_TR, C] normally,
        # a persistent [B, C] accumulator when `_DGATE_ACC_ATOMIC` (P9)
        R: fx.Int32,
        C: fx.Int32,
        XPAD: fx.Int32,
        XOFF: fx.Int32,
        SCALE_ROUNDING_BIAS: fx.Int32,
        GRID_MAIN: fx.Int32,
        NBX: fx.Int32,
        BDIM: fx.Int32,
    ):
        lds = fx.SharedAllocator().allocate(_DualSS).peek()
        tid = fx.thread_idx.x

        def _main(bid):
            _emit_dual_bias_body(
                lds,
                tid,
                X,
                ROW_OUT,
                ROW_SC,
                COL_OUT,
                COL_SC,
                BSUM,
                R,
                C,
                bid,
                SCALE_ROUNDING_BIAS,
                col_locality=col_locality,
                xcd_remap=xcd_remap,
                legpack=False,
                XPAD=None,
                XOFF=None,
                tile_bsum=True,
                fold_nrblk=None,
                BIAS=BIAS,
                CNT=CNT,
                ROW_SCP=ROW_SCP,
                COL_SCP=COL_SCP,
                GATE=GATE,
                Y=Y,
                YBIAS=YBIAS,
                DGS=DGS,
                gate=True,
                dgate=True,
                lmb2=lmb2,
                y_ld_cm=y_nt,
            )

        bid = rocdl.readfirstlane(T.i32, fx.block_idx.x)
        _scf_if(bid < GRID_MAIN, lambda: _main(bid))
        _scf_if(
            bid >= GRID_MAIN,
            lambda: _emit_b_scale_repack(
                tid, bid - GRID_MAIN, NBX, BDIM, R, XPAD, BX_RAW, BX_OUT, BW_RAW, BW_OUT
            ),
        )

    @flyc.jit
    def _dual_bias_gate_tile_launch(
        X: fx.Tensor,
        ROW_OUT: fx.Tensor,
        ROW_SC: fx.Tensor,
        COL_OUT: fx.Tensor,
        COL_SC: fx.Tensor,
        BSUM: fx.Tensor,
        BIAS: fx.Tensor,
        CNT: fx.Tensor,
        ROW_SCP: fx.Tensor,
        COL_SCP: fx.Tensor,
        BX_RAW: fx.Tensor,
        BX_OUT: fx.Tensor,
        BW_RAW: fx.Tensor,
        BW_OUT: fx.Tensor,
        GATE: fx.Tensor,
        Y: fx.Tensor,
        YBIAS: fx.Tensor,
        DGS: fx.Tensor,
        R: fx.Int32,
        C: fx.Int32,
        XPAD: fx.Int32,
        XOFF: fx.Int32,
        SCALE_ROUNDING_BIAS: fx.Int32,
        GRID_MAIN: fx.Int32,
        NBX: fx.Int32,
        BDIM: fx.Int32,
        grid_x: fx.Int32,
        stream: fx.Stream,
    ):
        _dual_bias_gate_tile_kernel(
            X,
            ROW_OUT,
            ROW_SC,
            COL_OUT,
            COL_SC,
            BSUM,
            BIAS,
            CNT,
            ROW_SCP,
            COL_SCP,
            BX_RAW,
            BX_OUT,
            BW_RAW,
            BW_OUT,
            GATE,
            Y,
            YBIAS,
            DGS,
            R,
            C,
            XPAD,
            XOFF,
            SCALE_ROUNDING_BIAS,
            GRID_MAIN,
            NBX,
            BDIM,
        ).launch(grid=(grid_x, 1, 1), block=(BLK, 1, 1), stream=stream)

    return _dual_bias_gate_tile_launch


_GATE_TILE_LAUNCH = {}
_GATE_TILE_COMPILED = {}


def get_gate_tile_cast(R, C, bdim, lmb2):
    """(compiled_fn, grid_x) for the dual-bias-packed tile kernel with the
    gate multiply AND the dgate partial sum fused at the bf16 load (optimize
    round 4 / P1, round 5 / P2), plus a per-shape cache-policy choice for
    the Y load (optimize round 8 / P5). Same `_ORDER_TABLE` oracle lookup as
    `get_dual_bias_tile_cast` -- the fusion changes zero scheduling
    decisions, so any measured delta is the fused load and fold alone; the
    P5 `y_nt` lookup is an independent oracle over the SAME `(R, C)` key,
    not a scheduling decision either."""
    col_locality, xcd_remap = _pick_block_order(R, C, True, True, False, False)
    y_nt = _pick_gate_y_cm(R, C)
    lk = (col_locality, xcd_remap, lmb2, y_nt)
    raw = _GATE_TILE_LAUNCH.get(lk)
    if raw is None:
        raw = _build_gate_tile_launch(col_locality, xcd_remap, lmb2, y_nt)
        _GATE_TILE_LAUNCH[lk] = raw
    key = (int(R), int(C), int(bdim), int(lmb2))
    ent = _GATE_TILE_COMPILED.get(key)
    if ent is None:
        import torch

        def z(*shape, dtype=torch.uint8):
            return torch.zeros(shape, dtype=dtype, device="cuda")

        x = z(R, C // 2, dtype=torch.int32)
        ro = z(R, C // 8, dtype=torch.int32)
        rs = z(R, C // 32)
        co = z(C, R // 8, dtype=torch.int32)
        cs = z(C, R // 32)
        bs = z(R // _TR, C, dtype=torch.float32)
        bias = z(C, dtype=torch.int16)
        cnt = z(C // _TC, dtype=torch.int32)
        extra = (
            z(R, C // 32),
            z(C, R // 32),
            z(bdim, R // 32),
            z(bdim, R // 32),
            z(bdim, C // 32),
            z(bdim, C // 32),
        )
        g = z(R // (_TR * lmb2), C // 2, dtype=torch.int32)
        y = z(R, C // 2, dtype=torch.int32)
        yb = z(C // 2, dtype=torch.int32)
        # P9: the trial buffer's row count is cosmetic -- `_static_layout`
        # only ever extracts a base pointer for this family of kernels (see
        # its own docstring), never a shape/stride -- but matching the real
        # allocation shape here keeps this call site honest to read.
        _dg_rows = (R // (_TR * lmb2)) if _DGATE_ACC_ATOMIC else (R // _TR)
        dgs = z(_dg_rows, C, dtype=torch.float32)
        grid_x = (R // _TR) * (C // _TC)
        fn = flyc.compile(
            raw,
            *_static_layout(x, ro, rs, co, cs, bs, bias, cnt, *extra, g, y, yb, dgs),
            R,
            C,
            C,
            0,
            1 << 21,
            grid_x,
            0,
            bdim,
            grid_x,
            torch.cuda.current_stream(),
        )
        ent = (fn, grid_x)
        _GATE_TILE_COMPILED[key] = ent
    return ent


# ---- merged fold: dgate's [P, C] f32 partial sums -> [B, C] bf16, AND the
# bias stage-2 reduce's [P, C] f32 -> [C] bf16, in ONE launch (optimize
# round 6 / P3). Round 5 / P2 landed these as two separate launches on
# purpose (different output shapes, `[B, C]` vs `[C]`, and the dgate fold
# needs no cross-lane communication unlike the bias reduce's 16-lane DPP
# column fold) and deferred merging them to a later round; `bs` and `dgs`
# are nonetheless the SAME shape (`[P, C]`, `P = R / _TR`), so one grid can
# carry both reductions and the gate path pays only ONE extra launch's
# dispatch floor instead of two (`campaign-lessons.md`: a bare-minimum
# kernel still costs ~2.0-2.1us of dispatch on this box).
#
# Blocks `bid < NDG` (`NDG = (P // grp) * ncb == B * C / _DG_FOLD_BLK`) fold
# the dgate plane -- the SAME body as the old standalone `_dgate_fold_kernel`
# (one block owns `_DG_FOLD_BLK` contiguous columns of ONE batch sample,
# every load a coalesced run, the store one bf16 per lane). Blocks
# `bid >= NDG` run `_stage2_bias_kernel`'s body UNCHANGED (16-lane DPP split
# of the P axis + `_row16_sum_f32`, lane 15 writes), re-based on
# `lbid = bid - NDG` so its column numbering is unaffected by sharing the
# grid with the dgate blocks -- `bias_bit_exact` depends on this being an
# exact copy, not a re-derivation. Grid = `NDG + ceil(C / _S2_RPB)`.
_DG_FOLD_BLK = 256  # matches `_S2_BLK`; C % _DG_FOLD_BLK == 0 at every shape
# `dual_bias_packed_eligible` admits (C % 256 == 0 is already required for
# the production pack's own tile geometry, `_TC == 256`).


def _build_merged_fold_launch(grp, ncb):
    """One compiled kernel, two block roles selected by `bid` vs the
    runtime `NDG` threshold: `bid < NDG` is a dgate-fold block (`grp = L /
    _TR` tile-rows per sample, `ncb = C // _DG_FOLD_BLK` column-blocks per
    sample), `bid >= NDG` is a bias stage-2 block. `grp`/`ncb` are baked in
    at trace time (one compiled kernel per `(ncb, grp)` pair, same as the
    old `_build_dgate_fold_launch`); `NDG` itself is a runtime arg so the
    grid split can still vary per call without re-tracing."""

    @flyc.kernel(known_block_size=[_DG_FOLD_BLK, 1, 1])
    def _merged_fold_kernel(
        DGS: fx.Tensor,
        DG: fx.Tensor,
        BS: fx.Tensor,
        BIAS: fx.Tensor,
        P: fx.Int32,
        C: fx.Int32,
        NDG: fx.Int32,
    ):
        tid = fx.thread_idx.x
        bid = fx.block_idx.x

        def _dgate_block():
            # Verbatim `_dgate_fold_kernel` body (optimize round 5 / P2),
            # extended with the P9 (optimize round 11) atomic-accumulator
            # path: when `_DGATE_ACC_ATOMIC`, `DGS` is ALREADY `[B, C]` --
            # the tile kernel's atomics finished the whole reduction in HBM
            # -- so this block reads the one complete sum instead of
            # looping `grp` strided reads, then clears that cell back to
            # 0.0 so the PERSISTENT buffer is ready for the next call
            # (`_dgate_acc_plane`; launches on one stream are serialized,
            # same invariant `_bias_fold_counters` already relies on).
            # `_wait_vmcnt0()` forces the read to land before the clear
            # issues: both are plain (non-atomic) ops to the SAME address,
            # and this block is the ONLY writer ever touching this exact
            # (b, col) cell during the fold phase (one block owns each
            # (b, col-run) pair), so no atomic is needed for the clear
            # itself -- only ordering against this block's own prior read.
            b = bid // ncb
            col = (bid % ncb) * _DG_FOLD_BLK + tid
            off = b * C + col
            osrc = _srd_at(DG, fx.Int32(0), 2, (P // grp) * C * 2)
            if _DGATE_ACC_ATOMIC:
                ssrc = _srd_at(DGS, fx.Int32(0), 4, (P // grp) * C * 4)
                acc = fx.Float32(buffer_ops.buffer_load(ssrc, off, vec_width=1, dtype=T.f32))
                buffer_ops.buffer_store(arith.bitcast(T.i16, arith.truncf(T.bf16, _raw(acc))), osrc, off)
                _wait_vmcnt0()
                buffer_ops.buffer_store(fx.Float32(0.0), ssrc, off)
            else:
                ssrc = _srd_at(DGS, fx.Int32(0), 4, P * C * 4)
                acc = fx.Float32(0.0)
                for g in range_constexpr(grp):
                    acc = acc + fx.Float32(
                        buffer_ops.buffer_load(ssrc, (b * grp + g) * C + col, vec_width=1, dtype=T.f32)
                    )
                buffer_ops.buffer_store(arith.bitcast(T.i16, arith.truncf(T.bf16, _raw(acc))), osrc, off)

        def _bias_block():
            # Verbatim `_stage2_bias_kernel` body (its `bf16_out=True`
            # case), re-based on `lbid` instead of `bid` -- same 16-lane
            # DPP split of P, same `_row16_sum_f32`, same lane-15 write.
            lbid = bid - NDG
            wave = tid // 64
            lane_in_wave = tid % 64
            dpp_row = lane_in_wave // 16
            lane_in_row = lane_in_wave % 16
            col = lbid * _S2_RPB + wave * 4 + dpp_row
            bsrc = _srd_at(BS, fx.Int32(0), 4, P * C * 4)
            osrc = _srd_at(BIAS, fx.Int32(0), 2, C * 2)
            p_idx = arith.index_cast(T.index, P)
            c_idx = arith.index_cast(T.index, C)
            col_idx = arith.index_cast(T.index, col)
            lane16 = arith.index_cast(T.index, lane_in_row)
            xinit = [fx.Float32(0.0), fx.Int32(0)]
            xres = xinit
            for _it, ia in range(fx.Index(0), p_idx, fx.Index(16), init=xinit):
                acc = ia[0]
                row = _it + lane16
                off = row * c_idx + col_idx
                v = buffer_ops.buffer_load(bsrc, off, vec_width=1, dtype=T.f32)
                acc = acc + fx.Float32(v)
                xres = yield [acc, ia[1]]
            total = _row16_sum_f32(xres[0])
            total = arith.bitcast(T.i16, arith.truncf(T.bf16, _raw(total)))
            if lane_in_row == fx.Int32(15):
                buffer_ops.buffer_store(total, osrc, col_idx)

        _scf_if(bid < NDG, _dgate_block)
        _scf_if(bid >= NDG, _bias_block)

    @flyc.jit
    def _merged_fold_launch(
        DGS,
        DG,
        BS,
        BIAS,
        P: fx.Int32,
        C: fx.Int32,
        NDG: fx.Int32,
        gx: fx.Int32,
        stream: fx.Stream,
    ):
        _merged_fold_kernel(DGS, DG, BS, BIAS, P, C, NDG).launch(
            grid=(gx, 1, 1), block=(_DG_FOLD_BLK, 1, 1), stream=stream
        )

    return _merged_fold_launch


_MERGED_FOLD_LAUNCH = {}
_MERGED_FOLD_COMPILED = {}


def get_merged_fold(P, C, grp):
    """Compile-and-cache the merged dgate+bias fold for this exact (P, C,
    grp); same shape-only key convention as `get_stage2_bias_reduce`. `P =
    R / _TR`, `grp = L / _TR`, so `P // grp == B` (checked by the caller's
    own `L % _TR == 0` precondition, see `_make_gate_plan`). Returns `(fn,
    grid_x, ndg)` -- the caller passes `ndg` straight back in as `NDG` on
    every call, so the kernel never has to recompute its own grid split."""
    ncb = C // _DG_FOLD_BLK
    lk = (ncb, grp)
    raw = _MERGED_FOLD_LAUNCH.get(lk)
    if raw is None:
        raw = _build_merged_fold_launch(grp, ncb)
        _MERGED_FOLD_LAUNCH[lk] = raw
    key = (int(P), int(C), int(grp))
    ent = _MERGED_FOLD_COMPILED.get(key)
    if ent is None:
        import torch

        # P9: cosmetic row count for the SAME reason as `get_gate_tile_
        # cast`'s own trial `dgs` above -- only the base pointer matters.
        _dg_rows = (P // grp) if _DGATE_ACC_ATOMIC else P
        dgs = torch.zeros((_dg_rows, C), dtype=torch.float32, device="cuda")
        dg = torch.zeros((P // grp, C), dtype=torch.int16, device="cuda")
        bs = torch.zeros((P, C), dtype=torch.float32, device="cuda")
        bias = torch.zeros((C,), dtype=torch.int16, device="cuda")
        ndg = (P // grp) * ncb
        gx = ndg + (C + _S2_RPB - 1) // _S2_RPB
        fn = flyc.compile(raw, dgs, dg, bs, bias, P, C, ndg, gx, torch.cuda.current_stream())
        ent = (fn, gx, ndg)
        _MERGED_FOLD_COMPILED[key] = ent
    return ent


def _gate_dgate_eager(dy, y, y_bias, b: int):
    """`dgate[b, c] = sum_l dy[b, l, c] * (y[b, l, c] + y_bias[c])`, fp32
    accumulation throughout (both inputs and the bias cast up BEFORE the
    add/multiply/sum). This fp32 order hits the compiled backward's own
    dgate SNR exactly; naive bf16 arithmetic (no cast) measures several dB
    below it against the fp64 reference (optimize round 4 ROUND_REPORT).
    The view, the fp32 upcast and the final `.bfloat16()` are all INSIDE
    this traced function -- same shape as `bench.py`'s own `dgate_producer`
    -- so `torch.compile` fuses the whole chain into ONE reduction kernel
    with no extra eager op before or after it (measured faster than viewing
    outside the traced function, optimize round 4 ROUND_REPORT).

    As of optimize round 5 / P2 this is ONLY reached by `_make_gate_plan`'s
    `L % _TR != 0` fallback (the main path folds `dgate` inside the gate-
    fused tile kernel itself, see `_fold_dgate_partials` / `get_dgate_
    fold`) -- kept unfused-input-shaped so the fallback stays correct at ANY
    `L` a future caller might pass, not just the three `_TR`-aligned ruler
    shapes. Only ever called through `_gate_dgate_reduce`'s `torch.compile`
    wrapper: eager (uncompiled), this is 4-5 unfused kernel launches over
    full `[R, C]`-sized fp32 intermediates; compiled, Inductor fuses it into
    one reduction kernel the same shape as `bench.py`'s own
    `compiled_dgate_producer` (the `dgate_ms` reference the ruler reports).
    """
    r, c = dy.shape
    l = r // b
    return (dy.view(b, l, c).float() * (y.view(b, l, c).float() + y_bias.float())).sum(1).bfloat16()


_GATE_DGATE_COMPILED = None


def _gate_dgate_reduce(dy, y, y_bias, b):
    """`torch.compile`'s `_gate_dgate_eager` ONCE per process (lazily --
    this module avoids a top-level `import torch`) and reuses the wrapper
    across every shape; `dynamic=False` means Dynamo keeps one specialized
    compiled artifact per distinct (dy.shape, b) under the same wrapper, so
    a shape seen before hits its cache directly instead of re-tracing."""
    global _GATE_DGATE_COMPILED
    import torch

    if _GATE_DGATE_COMPILED is None:
        _GATE_DGATE_COMPILED = torch.compile(_gate_dgate_eager, dynamic=False)
    return _GATE_DGATE_COMPILED(dy, y, y_bias, b)


_GATE_PLAN = {}


def _make_gate_plan(R, C, B, bdim, fp4_dtype, scale_rounding_mode, scale_rounding_bias):
    """Generative closure for `flydsl_quant_mxfp4_h16_dual_bias_gate`. The
    fused kernel path needs the WHOLE 64-row tile to belong to one batch
    sample (`rblk // lmb2` picks the gate row, see `_emit_dual_bias_body`);
    `task.yaml` only guarantees `L = R / B` is a multiple of 32, so this
    falls back to an unfused torch gate multiply feeding the existing
    production pack whenever L is not ALSO a multiple of `_TR` (64) --
    correct at any L, just without the producer-kernel deletion P1 is
    after. All three ruler shapes (L = 256, 256, 512) take the fused path.
    """
    import torch

    L = R // B

    if L % _TR != 0:

        def _plan_fallback(dy, gate, y, y_bias, bx_scale, bw_scale):
            g = (dy.view(B, L, C) * gate.view(B, 1, C)).view(R, C)
            out = flydsl_quant_mxfp4_h16_dual_bias_packed(
                g, (C,), fp4_dtype, bx_scale, bw_scale, scale_rounding_mode=scale_rounding_mode
            )
            dgate = _gate_dgate_reduce(dy, y, y_bias, B)
            return out + (dgate,)

        return _plan_fallback

    lmb2 = L // _TR
    P = R // _TR
    fn, grid_x = get_gate_tile_cast(R, C, bdim, lmb2)
    nbx = -(-bdim * (R // 128) // (16 * BLK))
    nbw = -(-bdim * (C // 128) // (16 * BLK))
    # Merged bias stage-2 + dgate fold (optimize round 6 / P3): `bs` and
    # `dgs` are both `[P, C]` f32 (`P // grp == B` exactly, since `P =
    # R/_TR`, `grp = lmb2 = L/_TR` and `R = B*L`), so one launch folds both
    # planes -- round 5 / P2 had this as two separate launches (`s2_fn` /
    # `dg_fn`); see `get_merged_fold` for the single-kernel body.
    mg_fn, mg_gx, mg_ndg = get_merged_fold(P, C, lmb2)
    i32, u8, e8m0 = torch.int32, torch.uint8, torch.float8_e8m0fnu
    raw_stream = torch._C._cuda_getCurrentRawStream

    def _plan(dy, gate, y, y_bias, bx_scale, bw_scale):
        x_i32 = dy.view(i32)
        stream = raw_stream(dy.device.index)
        ro = dy.new_empty((R, C // 8), dtype=i32)
        rs = dy.new_empty((R, C // 32), dtype=u8)
        co = dy.new_empty((C, R // 8), dtype=i32)
        cs = dy.new_empty((C, R // 32), dtype=u8)
        bs = dy.new_empty((R // _TR, C), dtype=torch.float32)
        # P9 (optimize round 11): a PERSISTENT [B, C] accumulator cached per
        # (shape, stream) instead of a fresh [R/_TR, C] plane every call --
        # see `_dgate_acc_plane`. The tile kernel below atomically adds into
        # it; the merged fold clears it back to 0.0 in the same pass that
        # emits bf16 `dgate`, so there is no per-call `zero_()` here.
        if _DGATE_ACC_ATOMIC:
            dgs = _dgate_acc_plane(B, C, dy.device, stream)
        else:
            dgs = dy.new_empty((P, C), dtype=torch.float32)
        bias = dy.new_empty((C,))
        dgate = dy.new_empty((B, C))
        cnt = _bias_fold_counters(R, C, dy.device, stream)
        bias_i16 = bias.view(torch.int16)
        dgate_i16 = dgate.view(torch.int16)
        rsp, csp = torch.empty_like(rs), torch.empty_like(cs)
        bxr = bx_scale.contiguous().view(u8)
        bwr = bw_scale.contiguous().view(u8)
        bxp, bwp = torch.empty_like(bxr), torch.empty_like(bwr)
        fn(
            x_i32,
            ro,
            rs,
            co,
            cs,
            bs,
            bias_i16,
            cnt,
            rsp,
            csp,
            bxr,
            bxp,
            bwr,
            bwp,
            gate.view(i32),
            y.view(i32),
            y_bias.view(i32),
            dgs,
            R,
            C,
            C,
            0,
            scale_rounding_bias,
            grid_x,
            nbx,
            bdim,
            grid_x + nbx + nbw,
            stream,
        )
        # One launch now folds BOTH `bs` (bias stage-2) and `dgs` (dgate) --
        # both were written by the SAME tile kernel call above, and neither
        # dY nor y is re-read (optimize round 5 / P2, merged round 6 / P3).
        mg_fn(dgs, dgate_i16, bs, bias_i16, P, C, mg_ndg, mg_gx, stream)
        return (
            ro.view(u8).view(fp4_dtype),
            rs.view(e8m0),
            co.view(u8).view(fp4_dtype),
            cs.view(e8m0),
            bias,
            rsp.view(e8m0),
            csp.view(e8m0),
            bxp.view(e8m0),
            bwp.view(e8m0),
            dgate,
        )

    return _plan


def flydsl_quant_mxfp4_h16_dual_bias_gate(
    dy, gate, y, y_bias, fp4_dtype, bx_scale, bw_scale, scale_rounding_mode=0
):
    """Fused `gate * dY` + MXFP4 dual-bias-packed pack, plus the gate
    gradient, for the gated Linears' backward (optimize round 4 / P1, round
    5 / P2, round 6 / P3): `bf16(gate * dY)` is produced at the tile's bf16
    load, inside the pack, and never materialised on its own; `dgate`'s
    partial sum (`fp32(dY) * (fp32(y) + fp32(y_bias))`, summed over each
    batch sample's rows) is accumulated at the SAME load and folded by one
    launch shared with the bias stage-2 reduce (round 6 / P3: one launch
    does both `[P, C] -> [C]` and `[P, C] -> [B, C]` instead of two), so
    neither dY nor y is read a second time. As of round 11 / P9 the dgate
    partial plane itself is a PERSISTENT `[B, C]` f32 accumulator
    (`_dgate_acc_plane`) that the tile kernel atomically adds into and the
    merged fold self-clears in the same pass it emits bf16 `dgate` --
    see `_DGATE_ACC_ATOMIC` -- instead of a fresh `[R/_TR, C]` plane folded
    by a `grp`-deep read loop every call. First nine outputs are byte-
    identical to `flydsl_quant_mxfp4_h16_dual_bias_packed(bf16(gate * dy),
    (C,), fp4_dtype, bx_scale, bw_scale)`. `dgate` is `[B, C]` bf16, matching
    the compiled backward's own `dy, mm (unbiased), bias` reduction within
    `dgate_snr_margin_db`. Falls back to an unfused torch gate multiply plus
    an unfused (but still `torch.compile`'d) dgate reduction whenever
    `L = R / B` is not a multiple of 64 -- see `_make_gate_plan`.
    """
    R, C = dy.shape
    B = gate.shape[0]
    bdim = int(bx_scale.shape[0])
    assert tuple(gate.shape) == (B, C), f"gate is {tuple(gate.shape)}, want ({B}, {C})"
    assert tuple(y.shape) == (R, C), f"y is {tuple(y.shape)}, want ({R}, {C})"
    assert tuple(y_bias.shape) == (C,), f"y_bias is {tuple(y_bias.shape)}, want ({C},)"
    key = (int(R), int(C), int(B), bdim, fp4_dtype, int(scale_rounding_mode))
    plan = _GATE_PLAN.get(key)
    if plan is None:
        plan = _make_gate_plan(
            int(R),
            int(C),
            int(B),
            bdim,
            fp4_dtype,
            int(scale_rounding_mode),
            _mxfp4_scale_rounding_bias(scale_rounding_mode),
        )
        _GATE_PLAN[key] = plan
    return plan(dy, gate, y, y_bias, bx_scale, bw_scale)


def flydsl_quant_mxfp4_h16_dual_bias_legs(x_bf16, leg_widths, fp4_dtype, scale_rounding_mode=0):
    """Two-(or-more)-leg column-slice variant of
    `flydsl_quant_mxfp4_h16_dual_bias` (optimize round 10 / P6 extended to
    the bias recipe): each `x_bf16[:, off:off+w]` column slice gets its OWN
    dual-bias-kernel launch, writing directly into the matching slice of ONE
    shared `(row_data, row_scale, col_data, col_scale, BSUM)` set, then a
    SINGLE stage-2 reduce folds the complete `BSUM` into `bias` -- no
    concatenation kernel anywhere, no legpack-specific stage-2 code.

    Bit-identical (row/row_scale/col/col_scale) to `flydsl_quant_mxfp4_h16_
    dual_bias(x_bf16, ...)`; `bias` is `allclose` vs its `bias` output for
    the same non-bit-exactness reason `flydsl_quant_mxfp4_h16_dual_bias`
    itself is only `allclose` vs `x_bf16.sum(0)` (different reduction
    order) -- see this round's ROUND_REPORT for the measured SNR.

    ``leg_widths`` must sum to ``x_bf16.shape[1]``; every entry must be a
    multiple of ``_TC``. Caller is responsible for the eligibility gate."""
    R, C = x_bf16.shape
    assert sum(leg_widths) == C, f"leg_widths {leg_widths} must sum to C={C}"
    key = (int(R), tuple(int(w) for w in leg_widths), fp4_dtype, int(scale_rounding_mode))
    plan = _H16_DUAL_BIAS_LEGS_PLAN.get(key)
    if plan is None:
        plan = _make_h16_dual_bias_legs_plan(R, leg_widths, fp4_dtype, scale_rounding_mode)
        _H16_DUAL_BIAS_LEGS_PLAN[key] = plan
    return plan(x_bf16)


# ---- P0: dGELU + cat folded into the two-leg bias legpack -----------------
# `flydsl_quant_mxfp4_h16_dual_bias_legs(cat(d_qkv, dgelu(d_act, preact)), ...)`
# with the bf16 `cat` never materialised. Each leg reads its OWN input tensor
# in place through `_emit_dual_bias_body`'s new `XSTR` input stride while both
# still write the shared [R, XPAD] row/col/BSUM plane at `XOFF`; leg1 reads a
# second tensor (`PR`) and computes the tanh dGELU at the bf16 load, before
# anything reaches LDS. Both the ROW and the COL phase then read that same
# staged tile, so both packs are byte-identical to packing a materialised G by
# construction -- the only numeric question is whether the in-kernel dGELU
# rounds to the same bf16 bits as Inductor's, which the campaign's mismatch
# gate prices directly.
_DUAL_BIAS_LEGS_DGELU_LAUNCH = {}
_DUAL_BIAS_LEGS_DGELU_COMPILED = {}


def _build_dual_bias_legs_dgelu_kernel(col_locality=False, xcd_remap=False, dgelu=False):
    _DualSS = _make_dual_struct(False)

    @flyc.kernel(known_block_size=[BLK, 1, 1])
    def _dual_bias_legs_dgelu_kernel(
        X: fx.Tensor,
        PR: fx.Tensor,
        ROW_OUT: fx.Tensor,
        ROW_SC: fx.Tensor,
        COL_OUT: fx.Tensor,
        COL_SC: fx.Tensor,
        BSUM: fx.Tensor,
        R: fx.Int32,
        C: fx.Int32,
        XPAD: fx.Int32,
        XOFF: fx.Int32,
        XSTR: fx.Int32,
        PRSTR: fx.Int32,
        SCALE_ROUNDING_BIAS: fx.Int32,
    ):
        lds = fx.SharedAllocator().allocate(_DualSS).peek()
        _emit_dual_bias_body(
            lds,
            fx.thread_idx.x,
            X,
            ROW_OUT,
            ROW_SC,
            COL_OUT,
            COL_SC,
            BSUM,
            R,
            C,
            fx.block_idx.x,
            SCALE_ROUNDING_BIAS,
            col_locality=col_locality,
            xcd_remap=xcd_remap,
            legpack=True,
            XPAD=XPAD,
            XOFF=XOFF,
            dgelu=dgelu,
            PR=PR,
            XSTR=XSTR,
            PRSTR=PRSTR,
        )

    return _dual_bias_legs_dgelu_kernel


def _build_dual_bias_legs_dgelu_launch(col_locality=False, xcd_remap=False, dgelu=False):
    kern = _build_dual_bias_legs_dgelu_kernel(col_locality, xcd_remap, dgelu)

    @flyc.jit
    def _dual_bias_legs_dgelu_launch(
        X: fx.Tensor,
        PR: fx.Tensor,
        ROW_OUT: fx.Tensor,
        ROW_SC: fx.Tensor,
        COL_OUT: fx.Tensor,
        COL_SC: fx.Tensor,
        BSUM: fx.Tensor,
        R: fx.Int32,
        C: fx.Int32,
        XPAD: fx.Int32,
        XOFF: fx.Int32,
        XSTR: fx.Int32,
        PRSTR: fx.Int32,
        SCALE_ROUNDING_BIAS: fx.Int32,
        grid_x: fx.Int32,
        stream: fx.Stream,
    ):
        kern(
            X,
            PR,
            ROW_OUT,
            ROW_SC,
            COL_OUT,
            COL_SC,
            BSUM,
            R,
            C,
            XPAD,
            XOFF,
            XSTR,
            PRSTR,
            SCALE_ROUNDING_BIAS,
        ).launch(grid=(grid_x, 1, 1), block=(BLK, 1, 1), stream=stream)

    return _dual_bias_legs_dgelu_launch


def get_dual_bias_legs_dgelu_cast(R, C, XPAD, XSTR, PRSTR, dgelu):
    """(compiled_fn, grid_x) for one leg of the fused dGELU legpack. Order comes
    from the SAME `_ORDER_TABLE` oracle the production legpack uses for this
    leg's own (R, C) -- the fused leg keeps the production block geometry."""
    col_locality, xcd_remap = _pick_block_order(R, C, True, True, False, False)
    lk = (
        col_locality,
        xcd_remap,
        bool(dgelu),
        _DG_LOAD_GROUP,
        _DG_SKIP,
        _DG_BSUM_FOLD,
        _DG_LD_CM,
        _DG_BIAS_ATOMIC,
        _DG_BIAS_SHARDS,
    )
    raw = _DUAL_BIAS_LEGS_DGELU_LAUNCH.get(lk)
    if raw is None:
        raw = _build_dual_bias_legs_dgelu_launch(col_locality, xcd_remap, bool(dgelu))
        _DUAL_BIAS_LEGS_DGELU_LAUNCH[lk] = raw
    key = (
        int(R),
        int(C),
        int(XPAD),
        int(XSTR),
        int(PRSTR),
        bool(dgelu),
        _DG_LOAD_GROUP,
        _DG_LD_CM,
        _DG_BIAS_ATOMIC,
        _DG_BIAS_SHARDS,
    )
    ent = _DUAL_BIAS_LEGS_DGELU_COMPILED.get(key)
    if ent is None:
        import torch

        x = torch.zeros((R, XSTR // 2), dtype=torch.int32, device="cuda")
        pr = torch.zeros((R, PRSTR // 2), dtype=torch.int32, device="cuda")
        ro = torch.zeros((R, XPAD // 8), dtype=torch.int32, device="cuda")
        rs = torch.zeros((R, XPAD // 32), dtype=torch.uint8, device="cuda")
        co = torch.zeros((XPAD, R // 8), dtype=torch.int32, device="cuda")
        cs = torch.zeros((XPAD, R // 32), dtype=torch.uint8, device="cuda")
        bs = torch.zeros((R // 32, XPAD), dtype=torch.float32, device="cuda")
        grid_x = (R // _TR) * (C // _TC)
        stream = torch.cuda.current_stream()
        fn = flyc.compile(
            raw,
            *_static_layout(x, pr, ro, rs, co, cs, bs),
            R,
            C,
            XPAD,
            0,
            XSTR,
            PRSTR,
            1 << 21,
            grid_x,
            stream,
        )
        ent = (fn, grid_x)
        _DUAL_BIAS_LEGS_DGELU_COMPILED[key] = ent
    return ent


_H16_DUAL_BIAS_LEGS_DGELU_PLAN = {}


def _make_h16_dual_bias_legs_dgelu_plan(R, w0, w1, s0, sa, sp, fp4_dtype, scale_rounding_mode):
    import torch

    C = w0 + w1
    scale_rounding_bias = _mxfp4_scale_rounding_bias(scale_rounding_mode)
    fn0, gx0 = get_dual_bias_legs_dgelu_cast(R, w0, C, s0, s0, False)
    fn1, gx1 = get_dual_bias_legs_dgelu_cast(R, w1, C, sa, sp, True)
    ro_shape = (R, C // 8)
    rs_shape = (R, C // 32)
    co_shape = (C, R // 8)
    cs_shape = (C, R // 32)
    i32 = torch.int32
    u8 = torch.uint8
    f32 = torch.float32
    e8m0 = torch.float8_e8m0fnu
    raw_stream = torch._C._cuda_getCurrentRawStream
    # P16(a) + lesson 7: atomic mode's `bs` is a `[_DG_BIAS_SHARDS, C]`
    # mini-bias plane (sharded, not a single `[C]` slot -- see the
    # module-level `_DG_BIAS_ATOMIC`/`_DG_BIAS_SHARDS` comments for why a
    # single slot regressed 2.68x in this exact plan). Stage-2 ALWAYS still
    # runs -- this shrinks its input, it does not delete it -- so this
    # branch only picks `P`/`bs_shape`; the call sequence below is otherwise
    # identical between atomic and legacy paths. Kept mutually exclusive
    # with `_DG_BSUM_FOLD` (see the module-level `_DG_BIAS_ATOMIC` comment).
    _batomic = bool(_DG_BIAS_ATOMIC)
    if _batomic:
        P = int(_DG_BIAS_SHARDS)
    else:
        P = R // (64 if _DG_BSUM_FOLD else 32)
    bs_shape = (P, C)
    s2_fn, s2_gx = get_stage2_bias_reduce(P, C)

    def _plan(d_qkv, d_act, preact):
        q32 = d_qkv.view(i32)
        a32 = d_act.view(i32)
        p32 = preact.view(i32)
        ro = d_qkv.new_empty(ro_shape, dtype=i32)
        rs = d_qkv.new_empty(rs_shape, dtype=u8)
        co = d_qkv.new_empty(co_shape, dtype=i32)
        cs = d_qkv.new_empty(cs_shape, dtype=u8)
        # Atomic mode accumulates INTO `bs` (multiple row-microblocks alias
        # onto each shard), so it must start at zero; the legacy [P, C] BSUM
        # plane is fully overwritten 1 store/cell by the kernels below, so
        # `new_empty` is safe and cheaper there.
        bs = d_qkv.new_zeros(bs_shape, dtype=f32) if _batomic else d_qkv.new_empty(bs_shape, dtype=f32)
        stream = raw_stream(d_qkv.device.index)
        fn0(q32, q32, ro, rs, co, cs, bs, R, w0, C, 0, s0, s0, scale_rounding_bias, gx0, stream)
        fn1(a32, p32, ro, rs, co, cs, bs, R, w1, C, w0, sa, sp, scale_rounding_bias, gx1, stream)
        bias_f32 = d_qkv.new_empty((C,), dtype=f32)
        s2_fn(bs, bias_f32, P, C, s2_gx, stream)
        bias = bias_f32.to(d_qkv.dtype)
        return (
            ro.view(u8).view(fp4_dtype),
            rs.view(e8m0),
            co.view(u8).view(fp4_dtype),
            cs.view(e8m0),
            bias,
        )

    return _plan


def flydsl_quant_mxfp4_h16_dual_bias_legs_dgelu(d_qkv, d_act, preact, fp4_dtype, scale_rounding_mode=0):
    """`flydsl_quant_mxfp4_h16_dual_bias_legs(cat(d_qkv, dgelu_tanh(d_act,
    preact)), (d_qkv.shape[1], d_act.shape[1]), ...)` with the bf16 cat never
    written. `d_act`/`preact` may be column-slice views; they are read in place
    through their own row strides."""
    R, w0 = d_qkv.shape
    w1 = d_act.shape[1]
    s0, sa, sp = d_qkv.stride(0), d_act.stride(0), preact.stride(0)
    assert d_act.shape[0] == R and preact.shape == d_act.shape
    assert d_qkv.stride(1) == 1 and d_act.stride(1) == 1 and preact.stride(1) == 1
    key = (
        int(R),
        int(w0),
        int(w1),
        int(s0),
        int(sa),
        int(sp),
        fp4_dtype,
        int(scale_rounding_mode),
    )
    plan = _H16_DUAL_BIAS_LEGS_DGELU_PLAN.get(key)
    if plan is None:
        plan = _make_h16_dual_bias_legs_dgelu_plan(
            int(R), int(w0), int(w1), int(s0), int(sa), int(sp), fp4_dtype, scale_rounding_mode
        )
        _H16_DUAL_BIAS_LEGS_DGELU_PLAN[key] = plan
    return plan(d_qkv, d_act, preact)


# ---- The dGELU leg on its own: the double blocks' MLP-up G ----------------
# `flydsl_quant_mxfp4_h16_dual_bias(dgelu_tanh(d_act, preact), ...)`: the fused
# legpack's leg1 launched alone across the full width (XOFF=0, XPAD=C), so the
# bf16 G [8192, 12288] the double blocks' img_mlp/txt_mlp backward otherwise
# writes and re-reads is never materialised. Same kernel, same knobs, same
# `_ORDER_TABLE` entry as the plain dual-bias pack at this (R, C).
_H16_DUAL_BIAS_DGELU_PLAN = {}


def _make_h16_dual_bias_dgelu_plan(R, C, sa, sp, fp4_dtype, scale_rounding_mode):
    import torch

    scale_rounding_bias = _mxfp4_scale_rounding_bias(scale_rounding_mode)
    fn, gx = get_dual_bias_legs_dgelu_cast(R, C, C, sa, sp, True)
    ro_shape = (R, C // 8)
    rs_shape = (R, C // 32)
    co_shape = (C, R // 8)
    cs_shape = (C, R // 32)
    i32 = torch.int32
    u8 = torch.uint8
    f32 = torch.float32
    e8m0 = torch.float8_e8m0fnu
    raw_stream = torch._C._cuda_getCurrentRawStream
    _batomic = bool(_DG_BIAS_ATOMIC)
    if _batomic:
        P = int(_DG_BIAS_SHARDS)
    else:
        P = R // (64 if _DG_BSUM_FOLD else 32)
    bs_shape = (P, C)
    s2_fn, s2_gx = get_stage2_bias_reduce(P, C)

    def _plan(d_act, preact):
        a32 = d_act.view(i32)
        p32 = preact.view(i32)
        ro = d_act.new_empty(ro_shape, dtype=i32)
        rs = d_act.new_empty(rs_shape, dtype=u8)
        co = d_act.new_empty(co_shape, dtype=i32)
        cs = d_act.new_empty(cs_shape, dtype=u8)
        bs = d_act.new_zeros(bs_shape, dtype=f32) if _batomic else d_act.new_empty(bs_shape, dtype=f32)
        stream = raw_stream(d_act.device.index)
        fn(a32, p32, ro, rs, co, cs, bs, R, C, C, 0, sa, sp, scale_rounding_bias, gx, stream)
        bias_f32 = d_act.new_empty((C,), dtype=f32)
        s2_fn(bs, bias_f32, P, C, s2_gx, stream)
        bias = bias_f32.to(d_act.dtype)
        return (
            ro.view(u8).view(fp4_dtype),
            rs.view(e8m0),
            co.view(u8).view(fp4_dtype),
            cs.view(e8m0),
            bias,
        )

    return _plan


def flydsl_quant_mxfp4_h16_dual_bias_dgelu(d_act, preact, fp4_dtype, scale_rounding_mode=0):
    """`flydsl_quant_mxfp4_h16_dual_bias(gelu_backward(d_act, preact, "tanh"), ...)`
    with the bf16 dGELU never written. Needs R % 128 == 0 and C % 256 == 0 (the
    dual kernel's tiling); the caller checks."""
    R, C = d_act.shape
    sa, sp = d_act.stride(0), preact.stride(0)
    assert preact.shape == d_act.shape
    assert d_act.stride(1) == 1 and preact.stride(1) == 1
    key = (int(R), int(C), int(sa), int(sp), fp4_dtype, int(scale_rounding_mode))
    plan = _H16_DUAL_BIAS_DGELU_PLAN.get(key)
    if plan is None:
        plan = _make_h16_dual_bias_dgelu_plan(
            int(R), int(C), int(sa), int(sp), fp4_dtype, scale_rounding_mode
        )
        _H16_DUAL_BIAS_DGELU_PLAN[key] = plan
    return plan(d_act, preact)


# ---- Stage-2 bias fold: [R/32, C] f32 partials -> [C] ---------------------
# (optimize round 6 / P3). Replaces the stock `bs.sum(0)` in
# `_make_h16_dual_bias_plan._plan` below. `bs` is P = R//32 rows of C columns
# (row-major, C contiguous); ATen's generic `sum(dim=0)` on this shape
# measured only 0.28-3.16 TB/s achieved BW across the real G_SHAPES
# (goal.md Sec C.P2/E.6) despite the natural reduction axis (P, the outer/
# strided dim) needing no cross-lane traffic structurally -- first-principles
# analysis pointed at ATen's generic-reduction kernel selection, not an HBM
# limit, as the likely bottleneck; this round's measurements below confirm
# a hand-written kernel recovers most of that gap.
#
# One 16-lane DPP row (a hardware-fixed group of 16 physically-adjacent
# lanes within a 64-lane wave) owns ONE output column: the 16 lanes split
# the P (row) axis by stride 16, each accumulating a scalar partial in a
# runtime loop, then `prims._row16_sum_f32` (4x ROW_SHR adds, no LDS) folds
# the 16 partials so lane 15 of the row holds the column total, which does
# the single writeback -- the "lane-fold, single writeback" mechanism this
# round's directive named, generalized to whatever P actually is (a
# compile-time-fixed 16-way split of a runtime-bounded loop) rather than a
# hardcoded 32.
#
# Measured this round (14-shape sweep -- all 6 real G_SHAPES, all 6 real
# X_SHAPES repurposed as synthetic [P, C] stress shapes, 2 alignment edge
# cases -- GPU 0, CUDA-event median of 13/25 cycles, interleaved-start-order
# vs baseline to cancel shared-box drift; see this round's ROUND_REPORT):
# +22.6% to +46.9% wall-clock over `bs.sum(0)` at every shape, mean +39.8%,
# worst-case +23.3%. Two alternative hand-written designs were measured and
# rejected: a scalar 1-column/lane loop and a vec4 4-column/lane loop, both
# with no cross-lane op, LOST to `bs.sum(0)` by 130-220% at every shape --
# at these small (<=44 MB) working sets the P=256/512 shapes have too few
# blocks/threads in flight to hide per-thread memory latency serially over
# the whole P axis, and splitting P across lanes (this design) is what
# fixes that, not coalescing (both losing designs were more coalesced than
# this one). A 64-lane (full-wave) fold was also measured: it wins big on
# narrow-C shapes but LOSES by up to -89.5% on wide-C shapes (fewer, more
# expensive blocks), so the 16-lane split here is the shape-robust choice.
#
# rocprofv3 kernel-dispatch timing (independent of the CUDA-event harness)
# confirms the direction and magnitude at the pure-kernel level: narrow
# shape (P=256, C=3072) 9231 ns / 0.34 TB/s -> 3742 ns / 0.84 TB/s; wide
# shape (P=512, C=21504) 31861 ns / 1.38 TB/s -> 22778 ns / 1.93 TB/s.
#
# Both dimensions rely on the same SRD-range-check masking `_srd_at`'s other
# callers in this file already lean on (module header, "buffer_load returns
# 0 / buffer_store dropped"): a row >= P always computes a byte offset
# >= nrec_bytes (since `(row-P)*C + col >= 0` whenever `row >= P` and
# `col >= 0`), so the load reads 0 instead of real memory, and a column >= C
# is dropped by BIAS's own SRD the same way -- no explicit bounds branch
# needed, even though the eligibility gate below only guarantees
# `C % 256 == 0` (every real R % 128 == 0 shape reachable here gives P a
# multiple of 16 anyway, but the masking holds unconditionally).
_S2_BLK = 256  # 16 DPP-rows (== output columns) per block; block-size-swept
_S2_RPB = _S2_BLK // 16  # this round: 128/256/512/1024 all measured, 256 won


def _build_stage2_bias_launch(bf16_out):
    ob = 2 if bf16_out else 4

    @flyc.kernel(known_block_size=[_S2_BLK, 1, 1])
    def _stage2_bias_kernel(BS: fx.Tensor, BIAS: fx.Tensor, P: fx.Int32, C: fx.Int32):
        tid = fx.thread_idx.x
        bid = fx.block_idx.x
        wave = tid // 64
        lane_in_wave = tid % 64
        dpp_row = lane_in_wave // 16
        lane_in_row = lane_in_wave % 16
        col = bid * _S2_RPB + wave * 4 + dpp_row
        bsrc = _srd_at(BS, fx.Int32(0), 4, P * C * 4)
        osrc = _srd_at(BIAS, fx.Int32(0), ob, C * ob)
        p_idx = arith.index_cast(T.index, P)
        c_idx = arith.index_cast(T.index, C)
        col_idx = arith.index_cast(T.index, col)
        lane16 = arith.index_cast(T.index, lane_in_row)
        xinit = [fx.Float32(0.0), fx.Int32(0)]
        xres = xinit
        for _it, ia in range(fx.Index(0), p_idx, fx.Index(16), init=xinit):
            acc = ia[0]
            row = _it + lane16
            off = row * c_idx + col_idx
            v = buffer_ops.buffer_load(bsrc, off, vec_width=1, dtype=T.f32)
            acc = acc + fx.Float32(v)
            xres = yield [acc, ia[1]]
        total = _row16_sum_f32(xres[0])
        if const_expr(bf16_out):
            total = arith.bitcast(T.i16, arith.truncf(T.bf16, _raw(total)))
        if lane_in_row == fx.Int32(15):
            buffer_ops.buffer_store(total, osrc, col_idx)

    @flyc.jit
    def _stage2_bias_launch(BS, BIAS, P: fx.Int32, C: fx.Int32, gx: fx.Int32, stream: fx.Stream):
        _stage2_bias_kernel(BS, BIAS, P, C).launch(grid=(gx, 1, 1), block=(_S2_BLK, 1, 1), stream=stream)

    return _stage2_bias_launch


_STAGE2_BIAS_LAUNCH = {False: _build_stage2_bias_launch(False), True: _build_stage2_bias_launch(True)}
_STAGE2_BIAS_COMPILED = {}


def get_stage2_bias_reduce(P, C, bf16_out=False):
    """Compile-and-cache the stage-2 `[P, C] f32 -> [C] f32` reduce for this
    exact (P, C), same bucket-K1 shape-only key convention as
    `_ROWQ_COMPILED`/`_DUAL_BIAS_COMPILED`. `bf16_out`: BIAS is an int16 view
    of a bf16 `[C]` instead."""
    key = (int(P), int(C), bool(bf16_out))
    ent = _STAGE2_BIAS_COMPILED.get(key)
    if ent is None:
        import torch

        bs = torch.zeros((P, C), dtype=torch.float32, device="cuda")
        bias = torch.zeros((C,), dtype=torch.int16 if bf16_out else torch.float32, device="cuda")
        gx = (C + _S2_RPB - 1) // _S2_RPB
        fn = flyc.compile(
            _STAGE2_BIAS_LAUNCH[bool(bf16_out)], bs, bias, P, C, gx, torch.cuda.current_stream()
        )
        ent = (fn, gx)
        _STAGE2_BIAS_COMPILED[key] = ent
    return ent


_H16_DUAL_BIAS_PLAN = {}


def _make_h16_dual_bias_plan(R, C, fp4_dtype, scale_rounding_mode=0):
    """Generative closure for `flydsl_quant_mxfp4_h16_dual_bias`, same shape
    as `_make_h16_dual_plan`: hoist the eligibility check and the fused-vs-
    fallback branch out of the per-call path.

    Key = (R, C, fp4_dtype, scale_rounding_mode) -- pure shape/dtype/flag
    values, no tensor identity -- bucket K1, same as `_H16_DUAL_PLAN`. Rule
    11's id(activation) ban does not apply."""
    if R % 128 != 0 or C % 256 != 0:
        # Same fallback an unfused caller would hand-roll without this
        # function: two independent `flydsl_quant_mxfp4_h16` calls plus the
        # plain torch reduction -- exactly today's `grad_2d.sum(0)`, so
        # correctness never regresses for a shape this kernel does not cover.
        def _plan(x_bf16):
            row, row_scale = flydsl_quant_mxfp4_h16(x_bf16, fp4_dtype, scale_rounding_mode)
            col, col_scale = flydsl_quant_mxfp4_h16(x_bf16.t().contiguous(), fp4_dtype, scale_rounding_mode)
            return row, row_scale, col, col_scale, x_bf16.sum(0)

        return _plan

    import torch

    mode = bias_fold_mode(R, C)
    if mode is not None:
        return _make_h16_dual_bias_tile_plan(
            R, (C,), fp4_dtype, _mxfp4_scale_rounding_bias(scale_rounding_mode), mode
        )
    fn, grid_x = get_dual_bias_cast(R, C)
    ro_shape = (R, C // 8)
    rs_shape = (R, C // 32)
    co_shape = (C, R // 8)
    cs_shape = (C, R // 32)
    bs_shape = (R // 32, C)
    i32 = torch.int32
    u8 = torch.uint8
    f32 = torch.float32
    e8m0 = torch.float8_e8m0fnu
    raw_stream = torch._C._cuda_getCurrentRawStream
    bias_rounding = _mxfp4_scale_rounding_bias(scale_rounding_mode)
    P = R // 32
    s2_fn, s2_gx = get_stage2_bias_reduce(P, C)

    def _plan(x_bf16):
        x_i32 = x_bf16.view(i32)
        ro = x_bf16.new_empty(ro_shape, dtype=i32)
        rs = x_bf16.new_empty(rs_shape, dtype=u8)
        co = x_bf16.new_empty(co_shape, dtype=i32)
        cs = x_bf16.new_empty(cs_shape, dtype=u8)
        bs = x_bf16.new_empty(bs_shape, dtype=f32)
        fn(x_i32, ro, rs, co, cs, bs, R, C, bias_rounding, grid_x, raw_stream(x_bf16.device.index))
        # Stage-2 fold [R/32, C] f32 partials -> [C]: hand-written 16-lane
        # DPP row fold (optimize round 6 / P3), replacing stock `bs.sum(0)`
        # (measured 0.28-3.16 TB/s achieved BW -- goal.md Sec C.P2/E.6).
        # See `_stage2_bias_kernel`'s comment above for the mechanism and
        # this round's ROUND_REPORT for the measured gain.
        bias_f32 = x_bf16.new_empty((C,), dtype=f32)
        s2_fn(bs, bias_f32, P, C, s2_gx, raw_stream(x_bf16.device.index))
        bias = bias_f32.to(x_bf16.dtype)
        return (
            ro.view(u8).view(fp4_dtype),
            rs.view(e8m0),
            co.view(u8).view(fp4_dtype),
            cs.view(e8m0),
            bias,
        )

    return _plan


def flydsl_quant_mxfp4_h16_dual_bias(x_bf16, fp4_dtype, scale_rounding_mode=0):
    """`flydsl_quant_mxfp4_h16_dual` plus the bias-gradient column sum, fused
    into the SAME dual-kernel launch's col phase (optimize round 5 / P2 --
    see `_emit_dual_bias_body`'s docstring for the mechanism).

    Returns `(row_data, row_scale, col_data, col_scale, bias)`. The first
    four are bit-exact (`torch.equal`) vs `flydsl_quant_mxfp4_h16_dual` at
    every dual-eligible shape -- the bias fusion only adds a store, it never
    perturbs the row/col pack math. `bias` is `allclose` (not `torch.equal`)
    vs `x_bf16.sum(0)`: a 32-way kernel-side tree plus a stock-torch stage-2
    fold is a different reduction order than `torch.sum`'s own, the same
    non-bit-exactness `torch.sum`'s two internal strategies already show
    against each other (see this round's ROUND_REPORT for the measured SNR).

    Falls back to two separate `flydsl_quant_mxfp4_h16` calls plus a plain
    `x_bf16.sum(0)` whenever `R % 128 != 0` or `C % 256 != 0` (the fused dual
    kernel's own alignment requirement -- see `dual_eligible`), exactly what
    a caller would do without this function.
    """
    R, C = x_bf16.shape
    key = (int(R), int(C), fp4_dtype, int(scale_rounding_mode))
    plan = _H16_DUAL_BIAS_PLAN.get(key)
    if plan is None:
        plan = _make_h16_dual_bias_plan(R, C, fp4_dtype, scale_rounding_mode)
        _H16_DUAL_BIAS_PLAN[key] = plan
    return plan(x_bf16)


# ---- Col-only (transpose-read) H16 pack: weight's Dgrad operand ------------
# (optimize round 9 / N2). Profiling the real in-image `mxfp4_linear.py` fwd+bwd
# (this round's ROUND_REPORT, /results/r9/p1_rank.json) confirmed that of the 4
# syntactic `.t().contiguous()` call sites in that file, only ONE ever executes
# under the locked recipe (fuse_x/fuse_g/x_is_quantized are always True, so the
# other 3 are dead code): `weight.t().contiguous()` feeding the Dgrad GEMM's B
# operand. That single transpose copy is 22.87% of a 4-layer Flux Linear
# stack's fwd+bwd device time on its own -- the single largest non-GEMM item.
#
# `flydsl_quant_mxfp4_h16_dual` (N1, above) already eliminates the analogous
# transpose for X/grad_out by reading the tensor once and producing both packs.
# Reusing it as-is for weight would work (bit-exact) but wastes half the read:
# weight's OWN row pack already comes from a separate, cheap flat
# `_rowq_kernel` call in Fprop (`_quantize_mxfp4_h16_op(weight)`), so a dual
# call here would recompute a row pack nobody reads. `_emit_dual_body`'s new
# `skip_row=True` mode (see its docstring) traces ONLY the col phase, so this
# kernel does exactly the work the old code's `weight.t().contiguous()` +
# row-only pack of the transpose did -- one coalesced HBM read of `weight`
# feeding the LDS tile, then the col-phase transpose-oriented amax+quantize+
# store -- with zero row-phase compute or LDS scratch for it.
def _build_colq_kernel(col_rht, col_locality=False, xcd_remap=False, col_sr=False):
    """Col-only kernel: emit ONLY the col-phase (transpose-read) mxfp4 pack
    from `_emit_dual_body`'s shared tile-load infrastructure (`skip_row=True`),
    dropping ROW_OUT/ROW_SC from the launch signature entirely rather than
    passing unused tensors. `col_rht`/`col_locality`/`xcd_remap` have the same
    meaning as in `_build_dual_kernel`; row_2d is fixed False (matches the
    H16-fusion recipe `flydsl_quant_mxfp4_h16_dual` already uses -- see that
    function's docstring for why `col_2d=False` is the correct recipe for a
    row-only-pack-of-the-transpose-equivalent result)."""
    col_2d = False
    _DualSS = _make_dual_struct(col_2d)

    @flyc.kernel(known_block_size=[BLK, 1, 1])
    def _colq_kernel(
        X: fx.Tensor,  # int32 view [R, C/2] (weight, untransposed)
        COL_OUT: fx.Tensor,  # int32 view [C, R/8]
        COL_SC: fx.Tensor,  # uint8 [C, R/32]
        R: fx.Int32,
        C: fx.Int32,
        SCALE_ROUNDING_BIAS: fx.Int32,
        SR_SEED: fx.Int32,  # per-launch stochastic-rounding seed (0 when SR off)
    ):
        lds = fx.SharedAllocator().allocate(_DualSS).peek()
        tid = fx.thread_idx.x
        _emit_dual_body(
            False,  # row_rht: dead under skip_row=True, never traced
            col_rht,
            False,  # row_2d: dead under skip_row=True, never traced
            col_2d,
            lds,
            tid,
            X,
            None,  # ROW_OUT: never referenced (skip_row=True)
            None,  # ROW_SC: never referenced (skip_row=True)
            COL_OUT,
            COL_SC,
            R,
            C,
            fx.block_idx.x,
            SCALE_ROUNDING_BIAS,
            col_locality=col_locality,
            xcd_remap=xcd_remap,
            skip_row=True,
            col_sr=col_sr,
            sr_seed=SR_SEED,
        )

    return _colq_kernel


def _build_colq_launch(col_rht, col_locality=False, xcd_remap=False, col_sr=False):
    kern = _build_colq_kernel(col_rht, col_locality, xcd_remap, col_sr)

    @flyc.jit
    def _colq_launch(
        X: fx.Tensor,
        COL_OUT: fx.Tensor,
        COL_SC: fx.Tensor,
        R: fx.Int32,
        C: fx.Int32,
        SCALE_ROUNDING_BIAS: fx.Int32,
        SR_SEED: fx.Int32,
        grid_x: fx.Int32,
        stream: fx.Stream,
    ):
        kern(X, COL_OUT, COL_SC, R, C, SCALE_ROUNDING_BIAS, SR_SEED).launch(
            grid=(grid_x, 1, 1), block=(BLK, 1, 1), stream=stream
        )

    return _colq_launch


_COLQ_LAUNCH = {}
_COLQ_COMPILED = {}
_COLQ_PLAN = {}


def colq_eligible(R, C):
    """Same alignment requirement as the fused dual (`dual_eligible`): the tile
    grid math (`(R // _TR) * (C // _TC)`) has no ragged-tail masking, so
    `R % 128 == 0` and `C % 256 == 0` are required here too (this is the exact
    same `_emit_dual_body`/`_TR`/`_TC` machinery, just with the row phase
    skipped -- the col phase's addressing and tile constraints are unchanged).
    Every ruler-relevant (N, K) weight shape satisfies this (confirmed this
    round); shapes that don't just fall back (see `_make_colq_plan`), same as
    today's behavior, so being exactly as strict as `dual_eligible` costs
    nothing in practice and keeps one single source of truth for the
    constraint instead of two independently-drifting copies."""
    return (int(R) % 128 == 0) and (int(C) % 256 == 0)


def get_colq_cast(R, C, col_rht, col_sr=False):
    """Return (compiled_fn, grid_x) for the col-only kernel at (R, C, col_rht).

    Reuses `_pick_block_order(R, C, True, col_rht, False, False)` directly
    instead of a separate `_pick_colq_order`/oracle table: the col-only
    kernel's grid/address pattern for COL_OUT (block order, XCD remap) is
    byte-for-byte identical to the plain dual kernel's at the same recipe, so
    whatever order is best for the dual kernel's col phase is best here too.
    `_ORDER_TABLE` carries dedicated colq entries under `_H16_RECIPE` for the
    4 measured weight shapes (see the `_H16_RECIPE` block comment above
    `_ORDER_TABLE`) -- those hit directly since this call passes the same
    literal key. Any other (R, C) misses the table and falls back to the
    safe shipped heuristic (`col_locality = C > R`, `xcd_remap = False`) --
    the same fallback `flydsl_quant_mxfp4_h16_dual` already hits for an
    untuned shape, not a new gap this round introduces."""
    col_locality, xcd_remap = _pick_block_order(R, C, True, col_rht, False, False)
    lk = (bool(col_rht), col_locality, xcd_remap, bool(col_sr))
    raw = _COLQ_LAUNCH.get(lk)
    if raw is None:
        raw = _build_colq_launch(bool(col_rht), col_locality, xcd_remap, bool(col_sr))
        _COLQ_LAUNCH[lk] = raw
    key = (int(R), int(C), bool(col_rht), bool(col_sr))
    ent = _COLQ_COMPILED.get(key)
    if ent is None:
        import torch

        x = torch.zeros((R, C // 2), dtype=torch.int32, device="cuda")
        co = torch.zeros((C, R // 8), dtype=torch.int32, device="cuda")
        cs = torch.zeros((C, R // 32), dtype=torch.uint8, device="cuda")
        grid_x = (R // _TR) * (C // _TC)
        stream = torch.cuda.current_stream()
        # (SCALE_ROUNDING_BIAS, SR_SEED), matching the dual launch. The SR_SEED
        # slot used to be absent here on the grounds that weight's col pack never
        # wants stochastic rounding; it now can (FLUX_FP4_SR_PASSES), and the
        # weight is in fact the operand whose round-to-nearest error repeats most,
        # since it is re-derived from a slowly moving parameter every step. Neither
        # warmup value is read at real-call time -- `_make_colq_plan`'s `_plan`
        # passes the real bias and a fresh seed on every call.
        fn = flyc.compile(raw, *_static_layout(x, co, cs), R, C, 1 << 21, 0, grid_x, stream)
        ent = (fn, grid_x)
        _COLQ_COMPILED[key] = ent
    return ent


def _make_colq_plan(R, C, col_rht, fp4_dtype, scale_rounding_mode=0, col_sr=False):
    """Generative closure for `flydsl_quant_mxfp4_h16_col`, same shape as
    `_make_rowq_plan`/`_make_h16_dual_plan`: hoist the eligibility check, the
    fused-vs-fallback branch, the output shape tuples/dtypes and the
    `torch._C._cuda_getCurrentRawStream` lookup out of the per-call path.

    Key = (R, C, col_rht, fp4_dtype, scale_rounding_mode) -- pure shape/dtype/
    flag values, no tensor identity -- bucket K1, same as every other `*_PLAN`
    cache in this file; Rule 11's id(weight) ban does not apply."""
    if not colq_eligible(R, C):
        # Exactly what today's call site does without this function: transpose
        # then row-pack. Correct for any (R, C), not just the colq-aligned ones.
        # The row-only pack carries SR too, so this honours col_sr rather than
        # silently dropping it.
        def _plan(x_bf16):
            return flydsl_quant_mxfp4_h16(x_bf16.t().contiguous(), fp4_dtype, scale_rounding_mode, col_sr)

        return _plan

    import torch

    fn, grid_x = get_colq_cast(R, C, col_rht, col_sr)
    # co/cs shapes mirror `_make_rowq_plan`'s itemsize-folded ro_shape: the
    # kernel addresses COL_OUT/COL_SC purely via `_srd_at`-computed R/C-derived
    # byte offsets (never the memref's reported shape/dtype -- same fact
    # `_static_layout`'s docstring establishes), so allocating directly in the
    # caller-facing fp4_dtype/e8m0 dtype (itemsize 1B each) instead of
    # (int32, uint8) + a post-call `.view().view()` chain is safe and removes
    # every per-call view from this path.
    co_shape = (C, R // 2)  # fp4_dtype elements == R//8 i32 words, same total bytes
    cs_shape = (C, R // 32)
    bias = _mxfp4_scale_rounding_bias(scale_rounding_mode)
    e8m0 = torch.float8_e8m0fnu
    raw_stream = torch._C._cuda_getCurrentRawStream

    def _plan(x_bf16):
        co = x_bf16.new_empty(co_shape, dtype=fp4_dtype)
        cs = x_bf16.new_empty(cs_shape, dtype=e8m0)
        # Fresh seed per launch, not per plan -- see `_make_plan`.
        sr_seed = _next_sr_seed() if col_sr else 0
        fn(x_bf16, co, cs, R, C, bias, sr_seed, grid_x, raw_stream(x_bf16.device.index))
        return co, cs

    return _plan


def flydsl_quant_mxfp4_h16_col(x_bf16, fp4_dtype, scale_rounding_mode=0, sr=False):
    """Colwise-transpose-read H16 mxfp4 cast: bit-identical to
    ``flydsl_quant_mxfp4_h16(x_bf16.t().contiguous(), fp4_dtype,
    scale_rounding_mode)`` but reads ``x_bf16`` ONCE via the dual kernel's
    coalesced LDS-tile-staged load instead of materializing a
    ``.t().contiguous()`` HBM transpose copy first and then packing that.

    Today's only caller is weight's Dgrad operand in the real Linear
    (``mxfp4_linear.py``'s ``backward()``, previously
    ``weight.t().contiguous()`` fed to ``_quantize_and_mm``); weight's OWN row
    pack (Fprop) keeps going through the separate, already-optimal flat
    ``_rowq_kernel`` via a plain ``flydsl_quant_mxfp4_h16(weight, ...)`` call,
    untouched by this function.

    Bit-exactness argument: this traces `_emit_dual_body` with
    ``skip_row=True``, ``col_rht=True``, ``row_2d=col_2d=False`` -- the exact
    same col-phase recipe (``row_rht=True, col_rht=True, row_2d=False,
    col_2d=False``) that ``flydsl_quant_mxfp4_h16_dual`` already runs and that
    function's own docstring already establishes produces a col pack
    byte-for-byte identical to ``flydsl_quant_mxfp4_h16`` on the transposed
    tensor (the col phase's IR never references ``row_rht``, only ``col_rht``,
    so skipping the row phase cannot change the col phase's output). Verified
    ``torch.equal`` against ``flydsl_quant_mxfp4_h16(x.t().contiguous(), ...)``
    on every ruler weight (N, K) shape plus the real Flux layer weight shapes
    this round (see ROUND_REPORT).

    Falls back to ``flydsl_quant_mxfp4_h16(x_bf16.t().contiguous(), fp4_dtype,
    scale_rounding_mode)`` -- exactly what a caller would do without this
    function -- whenever ``not colq_eligible(R, C)``.

    Returns ``(col_data, col_scale)`` in C++-compatible dtype/shape, i.e. what
    ``flydsl_quant_mxfp4_h16(x_bf16.t().contiguous(), ...)`` returns.
    """
    R, C = x_bf16.shape
    key = (int(R), int(C), True, fp4_dtype, int(scale_rounding_mode), bool(sr))
    plan = _COLQ_PLAN.get(key)
    if plan is None:
        plan = _make_colq_plan(R, C, True, fp4_dtype, scale_rounding_mode, bool(sr))
        _COLQ_PLAN[key] = plan
    return plan(x_bf16)


# ---- A6W4 activation dual: FP6 fprop operand + MXFP4 Wgrad col pack from one tile read ----


def _build_a6w4_dual_launch(col_locality=False, xcd_remap=False):
    _DualSS = _make_dual_struct(False)

    @flyc.kernel(known_block_size=[BLK, 1, 1])
    def _a6w4_dual_kernel(
        X: fx.Tensor,  # int32 view [R, C/2]
        ROW_OUT: fx.Tensor,  # int32 view [R, C/4]: FP8-padded FP6
        ROW_SC: fx.Tensor,  # uint8 [R*C/32], shuffle_scale_w4
        COL_OUT: fx.Tensor,  # int32 view [C, R/8]
        COL_SC: fx.Tensor,  # uint8 [C, R/32]
        R: fx.Int32,
        C: fx.Int32,
        ROW_BIAS: fx.Int32,
        COL_BIAS: fx.Int32,
    ):
        lds = fx.SharedAllocator().allocate(_DualSS).peek()
        tid = fx.thread_idx.x
        _emit_dual_body(
            True,
            True,
            False,
            False,
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
            COL_BIAS,
            col_locality=col_locality,
            xcd_remap=xcd_remap,
            row_fp6=True,
            row_bias=ROW_BIAS,
        )

    @flyc.jit
    def _a6w4_dual_launch(
        X: fx.Tensor,
        ROW_OUT: fx.Tensor,
        ROW_SC: fx.Tensor,
        COL_OUT: fx.Tensor,
        COL_SC: fx.Tensor,
        R: fx.Int32,
        C: fx.Int32,
        ROW_BIAS: fx.Int32,
        COL_BIAS: fx.Int32,
        grid_x: fx.Int32,
        stream: fx.Stream,
    ):
        _a6w4_dual_kernel(X, ROW_OUT, ROW_SC, COL_OUT, COL_SC, R, C, ROW_BIAS, COL_BIAS).launch(
            grid=(grid_x, 1, 1), block=(BLK, 1, 1), stream=stream
        )

    return _a6w4_dual_launch


_A6W4_DUAL = {}


def a6w4_act_dual_eligible(R, C):
    return colq_eligible(R, C)


def flydsl_quant_a6w4_act_dual(x_bf16):
    """One read of a bf16 [R, C] activation for both of its A6W4-recipe packs:
      (aq [R, C] uint8, sa [R*C/32] uint8)  the A6W4 fprop operand, the same bytes as
                                             a6w4_quant.quant_act_a6w4 (E2M3, scale bias 0)
      (co [C, R/2] fp4, cs [C, R/32] e8m0)  the Wgrad col pack, the same bytes as
                                             flydsl_quant_mxfp4_h16_col(x, fp4, 0)
    Requires a6w4_act_dual_eligible(R, C)."""
    import torch

    R, C = x_bf16.shape
    key = (int(R), int(C))
    ent = _A6W4_DUAL.get(key)
    if ent is None:
        col_locality, xcd_remap = _pick_block_order(R, C, True, True, False, False)
        raw = _build_a6w4_dual_launch(col_locality, xcd_remap)
        x = torch.zeros((R, C // 2), dtype=torch.int32, device="cuda")
        ro = torch.zeros((R, C // 4), dtype=torch.int32, device="cuda")
        rs = torch.zeros((R * C // 32,), dtype=torch.uint8, device="cuda")
        co = torch.zeros((C, R // 8), dtype=torch.int32, device="cuda")
        cs = torch.zeros((C, R // 32), dtype=torch.uint8, device="cuda")
        grid_x = (R // _TR) * (C // _TC)
        fn = flyc.compile(
            raw, *_static_layout(x, ro, rs, co, cs), R, C, 0, 1 << 21, grid_x, torch.cuda.current_stream()
        )
        ent = (fn, grid_x)
        _A6W4_DUAL[key] = ent
    fn, grid_x = ent
    aq = x_bf16.new_empty((R, C), dtype=torch.uint8)
    sa = x_bf16.new_empty((R * C // 32,), dtype=torch.uint8)
    co = x_bf16.new_empty((C, R // 2), dtype=torch.float4_e2m1fn_x2)
    cs = x_bf16.new_empty((C, R // 32), dtype=torch.float8_e8m0fnu)
    fn(
        x_bf16,
        aq,
        sa,
        co,
        cs,
        R,
        C,
        0,
        _mxfp4_scale_rounding_bias(0),
        grid_x,
        torch._C._cuda_getCurrentRawStream(x_bf16.device.index),
    )
    return aq, sa, co, cs


def _build_a6w4_dual_legs_launch(col_locality, xcd_remap, gelu, has_bias):
    _DualSS = _make_dual_struct(False)

    @flyc.kernel(known_block_size=[BLK, 1, 1])
    def _a6w4_dual_legs_kernel(
        X: fx.Tensor,
        BIAS: fx.Tensor,  # int32 view of this leg's bf16 bias [C]; unread unless has_bias
        ROW_OUT: fx.Tensor,
        ROW_SC: fx.Tensor,
        COL_OUT: fx.Tensor,
        COL_SC: fx.Tensor,
        R: fx.Int32,
        C: fx.Int32,
        XPAD: fx.Int32,
        XOFF: fx.Int32,
        XSTR: fx.Int32,
        ROW_BIAS: fx.Int32,
        COL_BIAS: fx.Int32,
    ):
        lds = fx.SharedAllocator().allocate(_DualSS).peek()
        _emit_dual_body(
            True,
            True,
            False,
            False,
            lds,
            fx.thread_idx.x,
            X,
            ROW_OUT,
            ROW_SC,
            COL_OUT,
            COL_SC,
            R,
            C,
            fx.block_idx.x,
            COL_BIAS,
            col_locality=col_locality,
            xcd_remap=xcd_remap,
            legpack=True,
            XPAD=XPAD,
            XOFF=XOFF,
            XSTR=XSTR,
            gelu=gelu,
            BIAS=BIAS if has_bias else None,
            row_fp6=True,
            row_bias=ROW_BIAS,
        )

    @flyc.jit
    def _a6w4_dual_legs_launch(
        X: fx.Tensor,
        BIAS: fx.Tensor,
        ROW_OUT: fx.Tensor,
        ROW_SC: fx.Tensor,
        COL_OUT: fx.Tensor,
        COL_SC: fx.Tensor,
        R: fx.Int32,
        C: fx.Int32,
        XPAD: fx.Int32,
        XOFF: fx.Int32,
        XSTR: fx.Int32,
        ROW_BIAS: fx.Int32,
        COL_BIAS: fx.Int32,
        grid_x: fx.Int32,
        stream: fx.Stream,
    ):
        _a6w4_dual_legs_kernel(
            X, BIAS, ROW_OUT, ROW_SC, COL_OUT, COL_SC, R, C, XPAD, XOFF, XSTR, ROW_BIAS, COL_BIAS
        ).launch(grid=(grid_x, 1, 1), block=(BLK, 1, 1), stream=stream)

    return _a6w4_dual_legs_launch


_A6W4_DUAL_LEGS = {}


def a6w4_act_dual_legs_eligible(R, widths):
    return int(R) % 256 == 0 and sum(widths) % 1024 == 0 and all(int(w) % 256 == 0 for w in widths)


def flydsl_quant_a6w4_act_dual_legs(xs, gelus, biases=None):
    """`flydsl_quant_a6w4_act_dual(cat([gelu(x + b) if g else x ...], 1))` without writing the cat or
    the GELU output: one launch per leg, each reading its own [R, w] tensor (unit column stride, any
    row stride) into its columns of the shared outputs. GELU is torch's tanh-approximate GELU, exact
    (see `_gelu1_torch`); a leg's bf16 bias, if given, is added in f32 first, as Inductor's fused
    ``gelu(x + bias)`` does. Requires a6w4_act_dual_legs_eligible(R, widths)."""
    import torch

    R = int(xs[0].shape[0])
    widths = tuple(int(x.shape[1]) for x in xs)
    C = sum(widths)
    biases = biases or (None,) * len(xs)
    x0 = xs[0]
    aq = x0.new_empty((R, C), dtype=torch.uint8)
    sa = x0.new_empty((R * C // 32,), dtype=torch.uint8)
    co = x0.new_empty((C, R // 2), dtype=torch.float4_e2m1fn_x2)
    cs = x0.new_empty((C, R // 32), dtype=torch.float8_e8m0fnu)
    st = torch._C._cuda_getCurrentRawStream(x0.device.index)
    col_bias = _mxfp4_scale_rounding_bias(0)
    xoff = 0
    for x, g, b in zip(xs, gelus, biases):
        assert x.shape[0] == R and x.stride(1) == 1 and x.dtype == torch.bfloat16
        w, s = int(x.shape[1]), int(x.stride(0))
        if b is not None:
            # Named, not bare: this fired 32 times across a 4-node run and said only
            # "AssertionError", which cost a whole job to narrow down.
            assert g, f"leg of width {w} has a bias but no GELU"
            assert b.shape == (w,), f"bias {tuple(b.shape)} is not the leg's width ({w},)"
            assert b.dtype == torch.bfloat16, f"bias dtype {b.dtype} is not bfloat16"
            assert b.is_contiguous(), f"bias {tuple(b.shape)} strides {b.stride()} not contiguous"
        key = (R, w, C, s, bool(g), b is not None)
        ent = _A6W4_DUAL_LEGS.get(key)
        if ent is None:
            col_locality, xcd_remap = _pick_block_order(R, w, True, True, False, False)
            raw = _build_a6w4_dual_legs_launch(col_locality, xcd_remap, bool(g), b is not None)
            grid_x = (R // _TR) * (w // _TC)
            fn = flyc.compile(
                raw,
                *_static_layout(
                    torch.zeros((R, s // 2), dtype=torch.int32, device="cuda"),
                    torch.zeros((w // 2,), dtype=torch.int32, device="cuda"),
                    torch.zeros((R, C // 4), dtype=torch.int32, device="cuda"),
                    torch.zeros((R * C // 32,), dtype=torch.uint8, device="cuda"),
                    torch.zeros((C, R // 8), dtype=torch.int32, device="cuda"),
                    torch.zeros((C, R // 32), dtype=torch.uint8, device="cuda"),
                ),
                R,
                w,
                C,
                0,
                s,
                0,
                1 << 21,
                grid_x,
                torch.cuda.current_stream(),
            )
            ent = (fn, grid_x, x0.new_zeros((w,), dtype=torch.bfloat16))
            _A6W4_DUAL_LEGS[key] = ent
        fn, grid_x, no_bias = ent
        fn(x, no_bias if b is None else b, aq, sa, co, cs, R, w, C, xoff, s, 0, col_bias, grid_x, st)
        xoff += w
    return aq, sa, co, cs


# ---- FP8-row + MXFP4-col dual, with this step's amax riding along -------------------
#
# The FP8 forward reads every operand THREE times, and each of those passes is already at the
# bandwidth it can achieve, so none of them can be made faster on its own -- the only way to win
# is to delete passes. Measured in a 900-step trace on one node (GBS 256, fp8 fprop + mxfp4
# backward), all three moving the SAME 53.55 GB/step:
#
#   tensorwise_amax_partial_kernel      12.64 ms/step   4.24 TB/s   reduce to one scalar
#   quantize_tensorwise_pad_row_kernel  16.16 ms/step               apply the scale -> fp8 row
#   _colq_kernel_0                      19.11 ms/step   2.80 TB/s   transposed read -> mxfp4 col
#
# 47.91 ms of a 455 ms step spent reading one tensor three times. This kernel does all of it in
# ONE read: cast to FP8 with a scale the caller supplies, pack the MXFP4 column format, and
# reduce this step's amax so the caller has next step's scale for free.
#
# Measured against the three passes it replaces, summed over the five real Flux operand shapes:
#   three passes  33.51 ms/step
#   this kernel   16.93 ms/step
# with the FP8 pack, the MXFP4 column pack, the column scales AND the amax all bit-exact against
# quantize_fp8 and flydsl_quant_mxfp4_h16_col.
#
# Deleting passes pays twice here, because the step is bandwidth-limited rather than
# kernel-limited: the fp8 forward GEMM costs 95.0 ms/step in training but only 70.0 ms standalone
# with cold operands, and that 25 ms penalty tracks bytes-written-per-FLOP exactly (1.66x on the
# short-K shapes that write most, 0.95x on the large-K ones that barely write). The GEMM is
# losing HBM to the very passes this kernel removes.
#
# Why the caller supplies the scale instead of this kernel computing it: a tensorwise scale needs
# the whole tensor before the first output byte can be written, which is precisely why the amax
# is a separate pass today. Feeding in the PREVIOUS step's scale breaks that dependency. That is
# delayed scaling, and it is the ONLY numerics change here -- the cast itself is bit-identical.
# Measured cost of the staleness: perturbing an activation by 1%/5%/20% moved its amax by
# -0.56%/+0.00%/+2.79%, and requantising with the pre-perturbation scale clipped 0, 0 and 1
# elements out of 25M while leaving relative error at 2.652e-02 against a fresh 2.650e-02.
#
# The scale is a TENSOR and not a float on purpose. The caller keeps it in a device buffer, and
# converting that to a Python float would sync once per Linear per step -- 228 syncs, costing far
# more than this kernel saves.
#
# Two alternatives were measured and rejected, so they do not need retrying. Primus-Turbo has no
# delayed-scaling recipe: ScalingStrategy has exactly one member, DYNAMIC, and its source carries
# "# DELAYED_SCALING = auto() # TODO: undetermined". And torch.compile cannot stand in for this
# kernel: asked to produce the scaled cast and the amax from one function it emits two kernels
# and two reads, landing 0.43 ms/step WORSE than the production chain.
_FP8_SAT = 448.0  # e4m3 saturation bound, matching primus_turbo's fp8_params


def _emit_fp8_dual_body(
    lds,
    tid,
    X,
    ROW_OUT,
    COL_OUT,
    COL_SC,
    AMAXP,
    SCALE,
    R,
    C,
    bid,
    scale_rounding_bias,
    col_locality=False,
    xcd_remap=False,
    invert_scale=False,
    amax_mode="thread",
    legpack=False,
    XPAD=None,
    XOFF=None,
    XSTR=None,
    gelu=False,
):
    """`_emit_dual_bias_body`'s geometry with two substitutions and nothing else changed:

      ROW phase   was an MXFP4 pack writing an E8M0 scale byte per microblock; now a tensorwise
                  FP8 cast using ``SCALE``, writing 32 bytes per microblock and no scale.
      reduction   was a per-column bias sum into BSUM; now a per-thread amax into ``AMAXP``,
                  which the caller finishes with a plain torch ``amax`` over ``grid*BLK`` floats
                  (3 MB for the largest shape -- negligible beside the operand).

    The COL phase is copied verbatim, RHT included, because its output has to stay bit-exact with
    ``flydsl_quant_mxfp4_h16_col``: that is what Wgrad consumes.

    The FP8 row pack takes NO RHT. The Hadamard rotation belongs to the MXFP4 recipe and the FP8
    fprop does not apply one. Row and col each rebuild their own values from the LDS tile, so they
    can disagree about RHT without interfering -- the same independence `_emit_dual_bias_body`
    relies on for its own row/col recipes.

    ``legpack``/``XPAD``/``XOFF``/``XSTR`` are `_emit_dual_bias_body`'s fused-legs contract: this
    launch packs one ``[R, C]`` leg, read from its own tensor with row stride ``XSTR``, into
    columns ``XOFF:XOFF+C`` of shared ``[R, XPAD]`` outputs. ``gelu`` applies tanh-GELU to the
    loaded bf16 and rounds back to bf16 before the LDS store, so the row cast, the col pack and
    the amax all see the same values a materialised ``gelu(x)`` would have given them.
    """
    IRI = fx.Int32.ir_type
    ncblk = C // _TC
    xpad = XPAD if legpack else C
    xstr = xpad if XSTR is None else XSTR
    if xcd_remap:
        # Same 8:1 XCD remap as `_emit_dual_body`; a bijection over [0, grid_x), so it changes
        # only which block computes which tile, never a value.
        _nxcd = 8
        _per = (R // _TR) * ncblk // _nxcd
        bid = (bid % _nxcd) * _per + (bid // _nxcd)
    if col_locality:
        nrblk = R // _TR
        cblk = bid // nrblk
        rblk = bid % nrblk
    else:
        rblk = bid // ncblk
        cblk = bid % ncblk
    r0 = rblk * _TR
    c0w = cblk * _TCW
    r0i = arith.index_cast(T.index, r0)
    if legpack:
        c0i = arith.index_cast(T.index, cblk * _TC + XOFF)
    else:
        c0i = arith.index_cast(T.index, cblk * _TC)

    # Every thread loads the same dword, so this is a broadcast that stays resident in cache.
    scsrc = _srd_at(SCALE, arith.index(0), 4, fx.Int32(4))
    scale = buffer_ops.buffer_load(scsrc, 0, vec_width=1, dtype=T.f32)
    if invert_scale:
        # `SCALE` holds the delayed `scale_inv` (amax/448) instead of the forward scale.
        # `arith.divf` on f32 is a correctly-rounded IEEE divide here (verified bit-exact
        # against torch's `1.0 / t` over 262144 adversarial values), which is what the
        # convention demands -- a `v_rcp_f32` would shift the scale by an ulp and flip
        # about 1% of the FP8 rounding ties.
        scale = fx.Float32(1.0) / fx.Float32(scale)

    rsrc = _srd_at(X, r0i * arith.index_cast(T.index, xstr >> 1), 4, _TR * (xstr >> 1) * 4)
    # FP8 is one byte per element, so a row is `C >> 2` i32 words, where the MXFP4 row pack at two
    # elements per byte used `C >> 3`.
    orsrc = _srd_at(ROW_OUT, r0i * arith.index_cast(T.index, xpad >> 2), 4, _TR * (xpad >> 2) * 4)
    corsrc = _srd_at(COL_OUT, c0i * arith.index_cast(T.index, R >> 3), 4, _TC * (R >> 3) * 4)
    cscrsrc = _srd_at(COL_SC, c0i * arith.index_cast(T.index, R >> 5), 1, _TC * (R >> 5))
    # L3: how many f32 this block contributes to AMAXP. "thread" is the shipped layout
    # (one per thread); "wave" pre-reduces with 6 DPP ops (64x fewer); "block" folds the
    # four waves together after the col phase has freed the LDS tile (256x fewer).
    # `amax_mode` is a compile-time Python str, so this plain call resolves at trace time --
    # see `_amax_per_block` below for the single source of truth on this mapping.
    _NAP = _amax_per_block(amax_mode)
    absrc = _srd_at(AMAXP, arith.index_cast(T.index, bid * _NAP), 4, fx.Int32(_NAP * 4))

    for chunk in range_constexpr(_NLOAD):
        tw = chunk * (BLK * 4) + tid * 4
        tr = tw // _TCW
        wc = tw % _TCW
        goff = tr * (xstr >> 1) + c0w + wc
        vec = buffer_ops.buffer_load(rsrc, goff, vec_width=4, dtype=T.i32)
        if const_expr(gelu):
            words = []
            for q in range_constexpr(4):
                g = _gelu_pair(_bf16_pair_to_f32(vec[q]))
                words.append(rocdl.cvt_pk_bf16_f32(g[0], g[1]))
            vec = Vec.from_elements(words, fx.Int32)
        _lds_store_vec4(lds.buf.ptr, tw, vec)
    fx.barrier()

    # ---- ROW phase: tensorwise FP8 cast with the supplied scale, plus this thread's amax ----
    amax_acc = fx.Int32(0)
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
                # A bf16 pair in one i32, low half first so the fp8 bytes land in memory order.
                rbits.append(word << 16)
                rbits.append(word & 0xFFFF0000)
        vf = _microblock_vf(rbits, False)
        amax_acc = _imax(amax_acc, _microblock_amax(vf))
        words = []
        for g in range_constexpr(8):
            # cvt_pk_fp8_f32 packs two f32 into one half of a dword; two calls fill four bytes,
            # so 32 elements become 8 words. Saturating to +-448 before the cvt matches the HIP
            # cast and keeps a boundary round from emitting a NaN code.
            z = fx.Int32(0)
            qs = [
                math.clampf(vf[4 * g + i] * scale, fx.Float32(-_FP8_SAT), fx.Float32(_FP8_SAT))
                for i in range(4)
            ]
            w = fx.Int32(rocdl.cvt_pk_fp8_f32(IRI, qs[0], qs[1], z, 0))
            w = fx.Int32(rocdl.cvt_pk_fp8_f32(IRI, qs[2], qs[3], w, 1))
            words.append(w)
        gcmb = cblk * (_TC // 32) + cmb
        off = r_row * (xpad >> 2) + gcmb * 8
        if legpack:
            off = off + (XOFF >> 2)
        _store_words_vec4(orsrc, off, words[0:4])
        _store_words_vec4(orsrc, off + 4, words[4:8])
    if amax_mode == "thread":
        buffer_ops.buffer_store(Vec.from_elements([amax_acc], fx.Int32).bitcast(fx.Float32)[0], absrc, tid)
    elif amax_mode == "wave":
        # 6 DPP ops, no LDS, no barrier, no change in liveness -- the reduce happens
        # exactly where the store used to be. Lane 63 of each wave holds the wave max.
        # This is a plain helper, not an `@flyc.kernel`, so a runtime `if` cannot lower
        # here; mask by sending the other 63 lanes' offset past the SRD instead, the
        # same `_OOB` idiom `_emit_dual_body` uses for its own tail stores.
        _wm = _wave_max_i32(amax_acc)
        _woff = arith.select(tid % 64 == 63, tid // 64, fx.Int32(_OOB))
        buffer_ops.buffer_store(Vec.from_elements([_wm], fx.Int32).bitcast(fx.Float32)[0], absrc, _woff)

    # ---- COL phase: `_emit_dual_bias_body`'s col phase verbatim, minus its bias accumulator --
    _GW = 2
    _PLC = BLK // _GW
    _NPASS = _TC // _PLC
    _NMB = _RMB // _GW
    _pl_g = tid & 1
    _pl_j = tid >> 1
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
            cwords, cbiased = _finish_microblock(cbits, True, scale_rounding_bias, None)
            gmmb = rblk * _RMB + _mmb
            _store_words_vec4(corsrc, c_col * (R >> 3) + gmmb * 4, cwords)
            buffer_ops.buffer_store(arith.trunci(T.i8, cbiased & 0xFF), cscrsrc, c_col * (R >> 5) + gmmb)
    if amax_mode == "block":
        # The col phase is the last reader of `lds.buf`, so its first BLK dwords are
        # dead scratch now -- a block reduce with NO extra LDS array, which would have
        # cost a block/CU (32 KB -> 34 KB). `amax_acc` stays live across the col phase:
        # one extra VGPR against the ~15 spare the ISA dump measured.
        fx.barrier()
        _bm = _block_max_i32_p(lds.buf.ptr, tid, amax_acc, BLK)
        _boff = arith.select(tid == 63, fx.Int32(0), fx.Int32(_OOB))
        buffer_ops.buffer_store(Vec.from_elements([_bm], fx.Int32).bitcast(fx.Float32)[0], absrc, _boff)


def _build_fp8_dual_kernel(col_locality=False, xcd_remap=False, invert_scale=False, amax_mode="thread"):
    """Thin wrapper over `_emit_fp8_dual_body`, same shape as `_build_dual_bias_kernel`."""
    _DualSS = _make_dual_struct(False)

    @flyc.kernel(known_block_size=[BLK, 1, 1])
    def _fp8_dual_kernel(
        X: fx.Tensor,
        ROW_OUT: fx.Tensor,
        COL_OUT: fx.Tensor,
        COL_SC: fx.Tensor,
        AMAXP: fx.Tensor,
        SCALE: fx.Tensor,
        R: fx.Int32,
        C: fx.Int32,
        SCALE_ROUNDING_BIAS: fx.Int32,
    ):
        lds = fx.SharedAllocator().allocate(_DualSS).peek()
        _emit_fp8_dual_body(
            lds,
            fx.thread_idx.x,
            X,
            ROW_OUT,
            COL_OUT,
            COL_SC,
            AMAXP,
            SCALE,
            R,
            C,
            fx.block_idx.x,
            SCALE_ROUNDING_BIAS,
            col_locality=col_locality,
            xcd_remap=xcd_remap,
            invert_scale=invert_scale,
            amax_mode=amax_mode,
        )

    return _fp8_dual_kernel


def _build_fp8_dual_launch(col_locality=False, xcd_remap=False, invert_scale=False, amax_mode="thread"):
    kern = _build_fp8_dual_kernel(col_locality, xcd_remap, invert_scale, amax_mode)

    @flyc.jit
    def _fp8_dual_launch(
        X,
        ROW_OUT,
        COL_OUT,
        COL_SC,
        AMAXP,
        SCALE,
        R: fx.Int32,
        C: fx.Int32,
        SCALE_ROUNDING_BIAS: fx.Int32,
        grid_x: fx.Int32,
        stream: fx.Stream,
    ):
        kern(X, ROW_OUT, COL_OUT, COL_SC, AMAXP, SCALE, R, C, SCALE_ROUNDING_BIAS).launch(
            grid=(grid_x, 1, 1), block=(BLK, 1, 1), stream=stream
        )

    return _fp8_dual_launch


_FP8_DUAL_LAUNCH = {}
_FP8_DUAL_COMPILED = {}
_FP8_DUAL_PLAN = {}


def _amax_per_block(amax_mode):
    """AMAXP f32 per block for each L3 reduction depth. `thread` is the shipped
    layout; `wave` is a 64x shrink; `block` is 256x."""
    return {"thread": BLK, "wave": BLK // 64, "block": 1}[str(amax_mode)]


def fp8_dual_eligible(R, C):
    """Same alignment contract as `dual_eligible`, which this kernel's tiling inherits.

    R % 64 and C % 256 would suffice for the tiling itself, but keeping `dual_eligible`'s
    R % 128 means this shares the exact geometry that is already validated bit-exact, with no
    new edge case to reason about. Every Flux fprop operand clears it: R of 8192/16384/21504 and
    C of 3072/12288/15360."""
    return (int(R) % 128 == 0) and (int(C) % 256 == 0)


def get_fp8_dual_cast(R, C, invert_scale=False, amax_mode="thread"):
    """(compiled_fn, grid_x) for the FP8+MXFP4 dual at (R, C). Callers must check
    `fp8_dual_eligible` first -- same contract as `get_dual_bias_cast`.

    Reuses `_pick_block_order(R, C, True, True, False, False)`, the same tuned oracle the plain
    H16 dual and colq kernels use, so any measured delta against them is this kernel's doing and
    not a different block schedule."""
    col_locality, xcd_remap = _pick_block_order(R, C, True, True, False, False)
    lk = (col_locality, xcd_remap, bool(invert_scale), str(amax_mode))
    raw = _FP8_DUAL_LAUNCH.get(lk)
    if raw is None:
        raw = _build_fp8_dual_launch(col_locality, xcd_remap, bool(invert_scale), str(amax_mode))
        _FP8_DUAL_LAUNCH[lk] = raw
    key = (int(R), int(C), bool(invert_scale), str(amax_mode))
    ent = _FP8_DUAL_COMPILED.get(key)
    if ent is None:
        import torch

        grid_x = (R // _TR) * (C // _TC)
        fn = flyc.compile(
            raw,
            *_static_layout(
                torch.zeros((R, C // 2), dtype=torch.int32, device="cuda"),
                torch.zeros((R, C // 4), dtype=torch.int32, device="cuda"),
                torch.zeros((C, R // 8), dtype=torch.int32, device="cuda"),
                torch.zeros((C, R // 32), dtype=torch.uint8, device="cuda"),
                torch.zeros((grid_x * _amax_per_block(amax_mode),), dtype=torch.float32, device="cuda"),
                torch.zeros((1,), dtype=torch.float32, device="cuda"),
            ),
            R,
            C,
            1 << 21,
            grid_x,
            torch.cuda.current_stream(),
        )
        ent = (fn, grid_x)
        _FP8_DUAL_COMPILED[key] = ent
    return ent


def _make_fp8_dual_plan(R, C, fp4_dtype, fp8_dtype, scale_rounding_mode=0):
    """Generative closure, same shape as `_make_colq_plan`: hoist the shapes, dtypes, compiled
    fn, grid and raw-stream lookup out of the per-call path."""
    import torch

    fn, grid_x = get_fp8_dual_cast(R, C)
    ro_shape = (R, C)  # fp8, one byte per element
    co_shape = (C, R // 2)  # fp4_dtype elements == R//8 i32 words, same bytes
    cs_shape = (C, R // 32)
    ap_shape = (grid_x * BLK,)
    bias = _mxfp4_scale_rounding_bias(scale_rounding_mode)
    e8m0 = torch.float8_e8m0fnu
    raw_stream = torch._C._cuda_getCurrentRawStream

    def _plan(x_bf16, scale_t):
        ro = x_bf16.new_empty(ro_shape, dtype=fp8_dtype)
        co = x_bf16.new_empty(co_shape, dtype=fp4_dtype)
        cs = x_bf16.new_empty(cs_shape, dtype=e8m0)
        ap = x_bf16.new_empty(ap_shape, dtype=torch.float32)
        fn(x_bf16, ro, co, cs, ap, scale_t, R, C, bias, grid_x, raw_stream(x_bf16.device.index))
        return ro, co, cs, ap

    return _plan


def flydsl_quant_fp8_mxfp4_dual(x_bf16, fp4_dtype, fp8_dtype, scale_t, scale_rounding_mode=0):
    """One read of ``x_bf16`` -> (fp8 row pack, mxfp4 col pack, col scales, amax partials).

    ``scale_t`` is a 1-element f32 device tensor holding the FORWARD scale (``448/amax``), which
    for a delayed-scaling caller is the previous step's. ``amax partials`` is one f32 per thread;
    reduce it with ``.amax()`` to get this step's amax, from which the next scale follows. Keeping
    the reduction on device is the point -- see this section's header for why a float scale would
    cost 228 syncs per step.

    Equivalent to, and bit-exact against, this triple:

        q, si = quantize_fp8(x, fp8_dtype, TENSORWISE)      # given the same scale
        cq, cs = flydsl_quant_mxfp4_h16_col(x, fp4_dtype)
        amax = x.abs().amax()

    Requires `fp8_dual_eligible(R, C)`; there is no fallback here, because the caller has to
    decide whether a non-eligible shape reverts to the three separate calls (which also means
    reverting to a FRESH scale, so the fallback is not numerically identical and cannot be hidden
    inside this function)."""
    R, C = x_bf16.shape
    key = (int(R), int(C), fp4_dtype, fp8_dtype, int(scale_rounding_mode))
    plan = _FP8_DUAL_PLAN.get(key)
    if plan is None:
        plan = _make_fp8_dual_plan(R, C, fp4_dtype, fp8_dtype, scale_rounding_mode)
        _FP8_DUAL_PLAN[key] = plan
    return plan(x_bf16, scale_t)


# ---- Delayed-scale dual quant (production entry point: `flydsl_quant_fp8_mxfp4_dual_delayed`
# below) --------------------------------------------------------------------------------
# Folds the amax reduce / clamp / `* fl(1/448)` that callers of `flydsl_quant_fp8_mxfp4_dual`
# used to do in eager torch into two small FlyDSL finalize launches, so the only host-side
# work left is `new_empty` + kernel launches. `flydsl_quant_fp8_mxfp4_dual` above is
# untouched and stays the production baseline / correctness ruler (goal.md's L0-L3 stack).
import struct as _struct

_INV448 = _struct.unpack("<f", _struct.pack("<f", 1.0 / 448.0))[0]
assert _struct.unpack("<I", _struct.pack("<f", _INV448))[0] == 0x3B124925
_AMAX_CLAMP_MIN = 1e-12
_S1_AMAX_BLK = 256
_S2_AMAX_BLK = 1024
_DPP_SHR = 0x110
_DPP_BCAST15 = 0x142
_DPP_BCAST31 = 0x143


def _dpp_max_i32(acc, ctrl, row_mask=0xF):
    r = rocdl.update_dpp(_raw(acc).type, _raw(fx.Int32(0)), _raw(acc), ctrl, row_mask, 0xF, True)
    return _imax(acc, fx.Int32(r.result if hasattr(r, "result") else r))


def _wave_max_i32(v):
    for _sh in (1, 2, 4, 8):
        v = _dpp_max_i32(v, _DPP_SHR + _sh)
    v = _dpp_max_i32(v, _DPP_BCAST15, row_mask=0xA)
    return _dpp_max_i32(v, _DPP_BCAST31, row_mask=0xC)


def _make_amax_struct(n):
    @fx.struct
    class _AmaxSS:
        scr: fx.Array[fx.Int32, n, 16]

    return _AmaxSS


def _block_max_i32_p(ptr, tid, m, nthread):
    """Block-wide int max through `nthread` dwords of the caller's LDS. Lane 63 of
    every wave ends up holding the block max; `tid == 63` is the canonical writer.

    The caller owns the barrier BEFORE this (its own last reader of `ptr` must be
    done); the barrier after the store is issued here."""
    _lds_store1(ptr, tid, m)
    fx.barrier()
    lane = tid % fx.Int32(64)
    acc = _lds_load1(ptr, lane)
    for _q in range_constexpr(1, nthread // 64):
        acc = _imax(acc, _lds_load1(ptr, lane + fx.Int32(64 * _q)))
    return _wave_max_i32(acc)  # lane 63 holds the block max


def _block_max_i32(lds, tid, m, nthread):
    return _block_max_i32_p(lds.scr.ptr, tid, m, nthread)


def _build_amax_s1(npart, gx1):
    """`npart` f32 abs-bit partials -> one f32 per block into MID[gx1]."""
    _SS = _make_amax_struct(_S1_AMAX_BLK)
    rep = npart // (gx1 * _S1_AMAX_BLK * 4)

    @flyc.kernel(known_block_size=[_S1_AMAX_BLK, 1, 1])
    def _amax_s1_kernel(AP: fx.Tensor, MID: fx.Tensor):
        tid = fx.thread_idx.x
        bid = fx.block_idx.x
        lds = fx.SharedAllocator().allocate(_SS).peek()
        apsrc = _srd_at(AP, fx.Int32(0), 4, fx.Int32(npart * 4))
        msrc = _srd_at(MID, fx.Int32(0), 4, fx.Int32(gx1 * 4))
        m = fx.Int32(0)
        for _c in range_constexpr(rep):
            off = (bid * fx.Int32(rep) + fx.Int32(_c)) * fx.Int32(_S1_AMAX_BLK * 4) + tid * 4
            v = buffer_ops.buffer_load(apsrc, off, vec_width=4, dtype=T.i32)
            for _j in range_constexpr(4):
                m = _imax(m, _abs_i32(fx.Int32(v[_j])))
        bm = _block_max_i32(lds, tid, m, _S1_AMAX_BLK)
        if tid == fx.Int32(63):
            buffer_ops.buffer_store(Vec.from_elements([bm], fx.Int32).bitcast(fx.Float32)[0], msrc, bid)

    @flyc.jit
    def _amax_s1_launch(AP, MID, stream: fx.Stream):
        _amax_s1_kernel(AP, MID).launch(grid=(gx1, 1, 1), block=(_S1_AMAX_BLK, 1, 1), stream=stream)

    return _amax_s1_launch


def _build_amax_s2(nmid):
    """MID[nmid] -> next_scale_inv = maximumf(amax, 1e-12) * fl(1/448).

    The scale MUST be applied as a multiply by the f32-rounded reciprocal, not as
    `arith.divf(x, 448.0)`: torch lowers `tensor / <python float>` to
    `BUnaryFunctor<MulFunctor>(fl(1/448))`, and an IEEE divide lands one ulp away on
    over half of all inputs -- measured, and enough to fail the ruler's bit-exactness
    gate on seed 0 of every shape.
    """
    _SS = _make_amax_struct(_S2_AMAX_BLK)
    per = (nmid + _S2_AMAX_BLK - 1) // _S2_AMAX_BLK

    @flyc.kernel(known_block_size=[_S2_AMAX_BLK, 1, 1])
    def _amax_s2_kernel(MID: fx.Tensor, OUT: fx.Tensor):
        tid = fx.thread_idx.x
        lds = fx.SharedAllocator().allocate(_SS).peek()
        msrc = _srd_at(MID, fx.Int32(0), 4, fx.Int32(nmid * 4))
        osrc = _srd_at(OUT, fx.Int32(0), 4, fx.Int32(4))
        m = fx.Int32(0)
        for _c in range_constexpr(per):
            m = _imax(
                m,
                _abs_i32(
                    fx.Int32(
                        buffer_ops.buffer_load(
                            msrc, tid + fx.Int32(_c * _S2_AMAX_BLK), vec_width=1, dtype=T.i32
                        )
                    )
                ),
            )
        bm = _block_max_i32(lds, tid, m, _S2_AMAX_BLK)
        f = Vec.from_elements([bm], fx.Int32).bitcast(fx.Float32)[0]
        f = f.maximumf(fx.Float32(_AMAX_CLAMP_MIN))
        if tid == fx.Int32(63):
            buffer_ops.buffer_store(f * fx.Float32(_INV448), osrc, 0)

    @flyc.jit
    def _amax_s2_launch(MID, OUT, stream: fx.Stream):
        _amax_s2_kernel(MID, OUT).launch(grid=(1, 1, 1), block=(_S2_AMAX_BLK, 1, 1), stream=stream)

    return _amax_s2_launch


def _pick_amax_grid(npart):
    """Largest stage-1 grid in [64, 256] that divides `npart` into whole vec4 rounds."""
    for gx1 in (256, 128, 64):
        if npart % (gx1 * _S1_AMAX_BLK * 4) == 0:
            return gx1
    return 1


_FP8_DUAL_DELAYED_PLAN = {}


def _l3_amax_mode():
    """L3 reduction depth for the delayed path, from `FLUX_L3_AMAX_MODE`.

    `wave` (default) pre-reduces each wave's amax with 6 DPP ops where the per-thread
    store used to be: AMAXP shrinks 64x, the stage-1 launch disappears, and the ISA adds
    15 VALU with no extra barrier and no extra LDS. `thread` is the shipped AMAXP layout
    plus the two-stage reduce, which recovers the previous champion exactly. `block`
    folds the four waves together through the dead LDS tile after the col phase -- 256x
    smaller AMAXP, but two extra barriers per block, which measured slower on the
    60-blocks/CU shape. Env-selected so one file state can be benched in every mode with
    the kernel as the only variable, and so `flydsl_quant_fp8_mxfp4_dual` (which never
    passes `amax_mode`) keeps compiling a byte-identical kernel."""
    import os

    m = os.environ.get("FLUX_L3_AMAX_MODE", "wave").strip().lower()
    if m not in ("thread", "wave", "block"):
        raise ValueError(f"FLUX_L3_AMAX_MODE must be thread|wave|block, got {m!r}")
    return m


def _make_fp8_dual_delayed_plan(R, C, fp4_dtype, fp8_dtype, scale_rounding_mode=0):
    import torch

    amax_mode = _l3_amax_mode()
    fn, grid_x = get_fp8_dual_cast(R, C, invert_scale=True, amax_mode=amax_mode)
    ro_shape = (R, C)
    co_shape = (C, R // 2)
    cs_shape = (C, R // 32)
    npart = grid_x * _amax_per_block(amax_mode)
    ap_shape = (npart,)
    bias = _mxfp4_scale_rounding_bias(scale_rounding_mode)
    e8m0 = torch.float8_e8m0fnu
    raw_stream = torch._C._cuda_getCurrentRawStream

    # `thread` still needs stage 1 to fold grid_x*256 partials down to something one
    # block can finish. `wave`/`block` already left grid_x*4 / grid_x behind, which the
    # finalize block reads directly -- one launch fewer.
    if amax_mode == "thread":
        gx1 = _pick_amax_grid(npart)
        mid_shape = (gx1,)
        s1 = flyc.compile(
            _build_amax_s1(npart, gx1),
            torch.zeros((npart,), dtype=torch.float32, device="cuda"),
            torch.zeros(mid_shape, dtype=torch.float32, device="cuda"),
            torch.cuda.current_stream(),
        )
        nfinal = gx1
    else:
        gx1 = 0
        mid_shape = None
        s1 = None
        nfinal = npart
    s2 = flyc.compile(
        _build_amax_s2(nfinal),
        torch.zeros((nfinal,), dtype=torch.float32, device="cuda"),
        torch.zeros((1,), dtype=torch.float32, device="cuda"),
        torch.cuda.current_stream(),
    )

    def _plan(x_bf16, prev_scale_inv):
        ro = x_bf16.new_empty(ro_shape, dtype=fp8_dtype)
        co = x_bf16.new_empty(co_shape, dtype=fp4_dtype)
        cs = x_bf16.new_empty(cs_shape, dtype=e8m0)
        ap = x_bf16.new_empty(ap_shape, dtype=torch.float32)
        nxt = x_bf16.new_empty((1,), dtype=torch.float32)
        st = raw_stream(x_bf16.device.index)
        fn(x_bf16, ro, co, cs, ap, prev_scale_inv, R, C, bias, grid_x, st)
        if s1 is None:
            s2(ap, nxt, st)
        else:
            mid = x_bf16.new_empty(mid_shape, dtype=torch.float32)
            s1(ap, mid, st)
            s2(mid, nxt, st)
        return ro, co, cs, nxt

    return _plan


def flydsl_quant_fp8_mxfp4_dual_delayed(x_bf16, fp4_dtype, fp8_dtype, prev_scale_inv, scale_rounding_mode=0):
    """`flydsl_quant_fp8_mxfp4_dual` with the delayed-scale arithmetic folded in.

    Takes the PREVIOUS scale_inv (``amax/448``) and returns the next one, so the caller
    keeps no eager math. Byte-identical to

        row_q, col_q, col_s, amax_p = flydsl_quant_fp8_mxfp4_dual(x, fp4, fp8, 1.0/prev)
        next_inv = (amax_p.amax().float().clamp(min=1e-12) / 448.0).reshape(1)
    """
    R, C = x_bf16.shape
    key = (int(R), int(C), fp4_dtype, fp8_dtype, int(scale_rounding_mode))
    plan = _FP8_DUAL_DELAYED_PLAN.get(key)
    if plan is None:
        plan = _make_fp8_dual_delayed_plan(R, C, fp4_dtype, fp8_dtype, scale_rounding_mode)
        _FP8_DUAL_DELAYED_PLAN[key] = plan
    return plan(x_bf16, prev_scale_inv)


# ---- Delayed-scale dual quant of cat(legs), optionally GELU'd per leg ----------------------
# The forward mirror of `flydsl_quant_mxfp4_h16_dual_bias_legs_dgelu`: one launch per leg, each
# reading its own [R, w] tensor (a column-slice view is fine) and writing its columns of the
# shared row pack, col pack and col scales. Every output element depends on one column of one
# leg, so the packs equal the single-tensor kernel's on the materialised cat. The amax is a max
# over both legs' partials, so it does too.
def _build_fp8_dual_legs_kernel(col_locality, xcd_remap, amax_mode, gelu):
    _DualSS = _make_dual_struct(False)

    @flyc.kernel(known_block_size=[BLK, 1, 1])
    def _fp8_dual_legs_kernel(
        X: fx.Tensor,
        ROW_OUT: fx.Tensor,
        COL_OUT: fx.Tensor,
        COL_SC: fx.Tensor,
        AMAXP: fx.Tensor,
        SCALE: fx.Tensor,
        R: fx.Int32,
        C: fx.Int32,
        XPAD: fx.Int32,
        XOFF: fx.Int32,
        XSTR: fx.Int32,
        SCALE_ROUNDING_BIAS: fx.Int32,
    ):
        lds = fx.SharedAllocator().allocate(_DualSS).peek()
        _emit_fp8_dual_body(
            lds,
            fx.thread_idx.x,
            X,
            ROW_OUT,
            COL_OUT,
            COL_SC,
            AMAXP,
            SCALE,
            R,
            C,
            fx.block_idx.x,
            SCALE_ROUNDING_BIAS,
            col_locality=col_locality,
            xcd_remap=xcd_remap,
            invert_scale=True,
            amax_mode=amax_mode,
            legpack=True,
            XPAD=XPAD,
            XOFF=XOFF,
            XSTR=XSTR,
            gelu=gelu,
        )

    return _fp8_dual_legs_kernel


def _build_fp8_dual_legs_launch(col_locality, xcd_remap, amax_mode, gelu):
    kern = _build_fp8_dual_legs_kernel(col_locality, xcd_remap, amax_mode, gelu)

    @flyc.jit
    def _fp8_dual_legs_launch(
        X,
        ROW_OUT,
        COL_OUT,
        COL_SC,
        AMAXP,
        SCALE,
        R: fx.Int32,
        C: fx.Int32,
        XPAD: fx.Int32,
        XOFF: fx.Int32,
        XSTR: fx.Int32,
        SCALE_ROUNDING_BIAS: fx.Int32,
        grid_x: fx.Int32,
        stream: fx.Stream,
    ):
        kern(X, ROW_OUT, COL_OUT, COL_SC, AMAXP, SCALE, R, C, XPAD, XOFF, XSTR, SCALE_ROUNDING_BIAS).launch(
            grid=(grid_x, 1, 1), block=(BLK, 1, 1), stream=stream
        )

    return _fp8_dual_legs_launch


_FP8_DUAL_LEGS_LAUNCH = {}
_FP8_DUAL_LEGS_COMPILED = {}
_FP8_DUAL_LEGS_PLAN = {}


def get_fp8_dual_legs_cast(R, C, XPAD, XSTR, gelu, amax_mode):
    """(compiled_fn, grid_x) for one leg. Block order is the production oracle's for this leg's
    own (R, C), as the backward legpack does."""
    col_locality, xcd_remap = _pick_block_order(R, C, True, True, False, False)
    lk = (col_locality, xcd_remap, str(amax_mode), bool(gelu))
    raw = _FP8_DUAL_LEGS_LAUNCH.get(lk)
    if raw is None:
        raw = _build_fp8_dual_legs_launch(col_locality, xcd_remap, str(amax_mode), bool(gelu))
        _FP8_DUAL_LEGS_LAUNCH[lk] = raw
    key = (int(R), int(C), int(XPAD), int(XSTR), bool(gelu), str(amax_mode))
    ent = _FP8_DUAL_LEGS_COMPILED.get(key)
    if ent is None:
        import torch

        grid_x = (R // _TR) * (C // _TC)
        fn = flyc.compile(
            raw,
            *_static_layout(
                torch.zeros((R, XSTR // 2), dtype=torch.int32, device="cuda"),
                torch.zeros((R, XPAD // 4), dtype=torch.int32, device="cuda"),
                torch.zeros((XPAD, R // 8), dtype=torch.int32, device="cuda"),
                torch.zeros((XPAD, R // 32), dtype=torch.uint8, device="cuda"),
                torch.zeros((grid_x * _amax_per_block(amax_mode),), dtype=torch.float32, device="cuda"),
                torch.zeros((1,), dtype=torch.float32, device="cuda"),
            ),
            R,
            C,
            XPAD,
            0,
            XSTR,
            1 << 21,
            grid_x,
            torch.cuda.current_stream(),
        )
        ent = (fn, grid_x)
        _FP8_DUAL_LEGS_COMPILED[key] = ent
    return ent


def fp8_dual_legs_eligible(R, widths):
    return int(R) % 128 == 0 and all(int(w) % _TC == 0 for w in widths)


def _make_fp8_dual_legs_plan(R, widths, strides, gelus, fp4_dtype, fp8_dtype, scale_rounding_mode):
    import torch

    amax_mode = _l3_amax_mode()
    C = sum(widths)
    legs = []
    npart = 0
    xoff = 0
    for w, s, g in zip(widths, strides, gelus):
        fn, gx = get_fp8_dual_legs_cast(R, w, C, s, g, amax_mode)
        n = gx * _amax_per_block(amax_mode)
        legs.append((fn, gx, w, s, xoff, npart, n))
        npart += n
        xoff += w
    if amax_mode == "thread":
        gx1 = _pick_amax_grid(npart)
        s1 = flyc.compile(
            _build_amax_s1(npart, gx1),
            torch.zeros((npart,), dtype=torch.float32, device="cuda"),
            torch.zeros((gx1,), dtype=torch.float32, device="cuda"),
            torch.cuda.current_stream(),
        )
        nfinal = gx1
    else:
        gx1, s1, nfinal = 0, None, npart
    s2 = flyc.compile(
        _build_amax_s2(nfinal),
        torch.zeros((nfinal,), dtype=torch.float32, device="cuda"),
        torch.zeros((1,), dtype=torch.float32, device="cuda"),
        torch.cuda.current_stream(),
    )
    bias = _mxfp4_scale_rounding_bias(scale_rounding_mode)
    e8m0 = torch.float8_e8m0fnu
    raw_stream = torch._C._cuda_getCurrentRawStream

    def _plan(xs, prev_scale_inv):
        x0 = xs[0]
        ro = x0.new_empty((R, C), dtype=fp8_dtype)
        co = x0.new_empty((C, R // 2), dtype=fp4_dtype)
        cs = x0.new_empty((C, R // 32), dtype=e8m0)
        ap = x0.new_empty((npart,), dtype=torch.float32)
        nxt = x0.new_empty((1,), dtype=torch.float32)
        st = raw_stream(x0.device.index)
        for x, (fn, gx, w, s, off, p0, n) in zip(xs, legs):
            fn(x, ro, co, cs, ap[p0 : p0 + n], prev_scale_inv, R, w, C, off, s, bias, gx, st)
        if s1 is None:
            s2(ap, nxt, st)
        else:
            mid = x0.new_empty((gx1,), dtype=torch.float32)
            s1(ap, mid, st)
            s2(mid, nxt, st)
        return ro, co, cs, nxt

    return _plan


def flydsl_quant_fp8_mxfp4_dual_legs_delayed(
    xs, gelus, fp4_dtype, fp8_dtype, prev_scale_inv, scale_rounding_mode=0
):
    """`flydsl_quant_fp8_mxfp4_dual_delayed(cat([gelu(x) if g else x for x, g in zip(xs,
    gelus)], 1), ...)` without writing the cat or the GELU output. GELU is tanh-approximate,
    computed in f32 and rounded to bf16 before packing. Each leg must have unit column stride;
    its row stride is read in place."""
    R = int(xs[0].shape[0])
    widths = tuple(int(x.shape[1]) for x in xs)
    strides = tuple(int(x.stride(0)) for x in xs)
    for x in xs:
        assert x.shape[0] == R and x.stride(1) == 1 and x.dtype == xs[0].dtype
    key = (R, widths, strides, tuple(bool(g) for g in gelus), fp4_dtype, fp8_dtype, int(scale_rounding_mode))
    plan = _FP8_DUAL_LEGS_PLAN.get(key)
    if plan is None:
        plan = _make_fp8_dual_legs_plan(R, widths, strides, key[3], fp4_dtype, fp8_dtype, scale_rounding_mode)
        _FP8_DUAL_LEGS_PLAN[key] = plan
    return plan(xs, prev_scale_inv)


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
