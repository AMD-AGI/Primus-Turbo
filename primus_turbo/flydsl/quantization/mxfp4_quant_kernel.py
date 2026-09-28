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
from flydsl.expr import arith, buffer_ops, math, range_constexpr, rocdl
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec
from flydsl.expr.utils.arith import _to_raw as _raw

from primus_turbo.flydsl.utils.gemm_helper import mxfp4_packed_scale_byte, xcd_remap_pid

_OOB = 0x7FFFFFFF  # word offset past any SRD -> buffer_load returns 0 / buffer_store dropped

# Cache policy for the fused dual's source tile load (bit0 sc0, bit1 nt, bit4 sc1), set
# around the trace of ONE kernel by ``_pick_load_aux`` and part of both compile-cache keys.
_LOAD_AUX = 0

_MALL_BYTES = 256 << 20  # MI355X last-level cache


def _pick_load_aux(elems):
    """Source-tile cache policy for a dual over ``elems`` elements.

    Every line of the source is read exactly once by exactly one workgroup -- a tile is 512 B
    of a 128 B-aligned row, so no two of them share a line -- yet at 2 of the 3.0625 bytes per
    element the kernel moves it is two thirds of the traffic, so it owns two thirds of the
    cache. What does have reuse is the other third: eight column tiles each write 8 B of the
    same 64 B ROW_SC line, and the quantised planes are read straight back by the next kernel
    in the step. ``nt`` on the load stops the one stream nobody looks at twice from evicting
    them, and measures +3.0 to +14.6% once the working set is past the last-level cache.
    Below that the source itself is resident and the hint throws away real reuse (-5.3% at
    98 MB, -7.3% at 196 MB); far past it the streams have no residency left to protect and it
    turns negative again (-5.9% at 1372 MB, -9.4% at 1960 MB), so it is spent in the band
    where it pays.
    """
    fp = elems * 49 // 16  # in (2 B/elem) + all four output planes (1.0625 B/elem)
    return 2 if _MALL_BYTES <= fp <= 7 * _MALL_BYTES // 2 else 0


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


def _imin(a, b):
    return arith.select(a < b, a, b)


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


def _microblock_vf(vbits, use_rht, fold_scale=False, scales=None):
    """32 f32-bit i32 values -> list of 32 f32 Values (post-RHT if enabled).
    ``fold_scale`` drops the RHT's trailing ``*0.25``; see ``_rht16``. ``scales``
    is applied before RHT so RMSNorm can fold rstd*gamma in without a bf16 trip."""
    vf = [Vec.from_elements([b], fx.Int32).bitcast(fx.Float32)[0] for b in vbits]
    if scales is not None:
        vf = [vf[i] * scales[i] for i in range_constexpr(32)]
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


def _finish_microblock(vbits, use_rht, scale_rounding_bias, seed=None, scales=None):
    """32 f32-bit i32 values -> (4 fp4 i32 words, scale_e8m0 i8-ready i32).
    ``seed`` (i32 Value) enables stochastic rounding in the final cvt (amax/scale
    stay deterministic); ``scales`` folds a per-element factor in before RHT."""
    vf = _microblock_vf(vbits, use_rht, fold_scale=True, scales=scales)
    amax = _microblock_amax_f(vf)
    native_bits, biased = _compute_scale_native(amax, scale_rounding_bias, exp_up=vf_exp_up(use_rht))
    words = _cvt_microblock_to_fp4(vf, arith.bitcast(T.f32, native_bits), seed)
    return words, biased


# ---- fused-dual tile geometry (shared by the 2D and batched-3D kernels) ----
# Tile rows/cols for the fused dual. Both entry points temporarily select the faster
# 96/128-row geometry per shape via _pick_tile_geom below and restore these afterwards;
# this pair is the fallback for an R that divides neither, and must divide every R the
# eligibility contract admits.
_TR = 64  # tile rows (R dim); covers every R accepted by dual_eligible
_TC = 256  # tile cols (C dim)
_TCW = _TC // 2  # 128 i32 words per tile row
_NW = _TR * _TCW  # 8192 i32 words in LDS
_RMB = _TR // 32  # col m-microblocks per tile
_NLOAD = (_NW + BLK * 4 - 1) // (BLK * 4)  # vec4 loads per thread
_RROWTASK = (_TR * (_TC // 32)) // BLK  # row tasks per thread
_RMBC = _TC // 32  # row micro-blocks along C (== 8)
_SCP = _TR + 8  # row-phase scratch pitch; +8 dwords keeps the strided write conflict-free
_NSCR = max(_RMB * _TC, _RMBC * _SCP)  # LDS amax scratch, sized for both phases
# Scaled dual: half-height tile so buf+gamma+rstd LDS stays under 32 KB (5 WG/CU).
_RMS_TR = 32
_RMS_TC = _TC
_RMS_NW = _RMS_TR * (_RMS_TC // 2)  # 4096 i32 = 16 KB


_TILE_DEFAULT = (_TR, _TC)


def _set_tile_geom(tr, tc):
    """Re-derive every fused-dual tile constant for (tr, tc).

    Set immediately around the trace+compile of ONE kernel and restored afterwards;
    compilation is serialised in-process, and (tr, tc) is part of both compile-cache keys.
    """
    global _TR, _TC, _TCW, _NW, _RMB, _NLOAD, _RROWTASK, _RMBC, _SCP, _NSCR
    _TR, _TC = tr, tc
    _TCW = _TC // 2
    _NW = _TR * _TCW
    _RMB = _TR // 32
    _NLOAD = (_NW + BLK * 4 - 1) // (BLK * 4)
    _RROWTASK = (_TR * (_TC // 32)) // BLK
    _RMBC = _TC // 32
    _SCP = _TR + 8
    _NSCR = max(_RMB * _TC, _RMBC * _SCP)


def _pick_tile_geom(N, K):
    """Per-shape tile geometry for the fused dual (N = the R dim, K = the C dim).

    Three things are decided by (_TR, _TC) and the shipped shapes want different answers:
      * COL_OUT write granule = _TR/8 i32. At _TR=96 that is 48 bytes -- SUB-CACHELINE, and
        which the profile sees as write amplification on a kernel that is bandwidth bound.
        _TR=128 gives exactly 64 bytes.
      * legality: _TR must divide N, _TR*_TC/32 must divide BLK=256, and _TC must BE 256
        (the col phase is one thread per column).
    The 2D path asks about R_pad, a 128-multiple by construction, so it always lands on the
    128-row tile. Batched-3D tiles the real N and so still drops to 96 rows at N=2880.
    """
    # _TC is pinned to BLK: the col cast phase maps exactly one thread per tile column
    # (c_col = tid), so _TC != 256 silently corrupts the colwise output -- and only the
    # colwise one, so dgrad goes wrong while fwd and wgrad stay clean. Only _TR is free.
    for tr, tc in ((128, 256), (96, 256)):
        if N % tr == 0 and (tr * (tc // 32)) % BLK == 0 and K % 32 == 0:
            return tr, tc
    return _TILE_DEFAULT


def _co_wide(tr, tc):
    """True when a run of lanes can cover whole 64 B col-out lines.

    A thread owns tr/32 micro-block results of 16 B each for its column; they are consecutive
    in COL_OUT, so four lanes hold exactly one cacheline. See ``_emit_dual_body``.
    """
    return tr // 32 == 4 and tc == BLK


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
    rowpad=False,
    col_locality=False,
    batched=False,
    row_sr=False,
    col_sr=False,
    sr_seed=None,
    sr_gbid=None,
    rmsnorm_scale=False,
    RSTD=None,
    GAMMA=None,
    tile_tr=None,
    tile_tc=None,
    pack_row=None,
    pack_col=None,
):
    """Emit one fused-dual tile (rowwise + colwise-transpose mxfp4 cast) for block
    ``bid``. ``row_2d``/``col_2d`` pick the C++ ``USE_2D_BLOCK`` amax geometry; the
    batched-3D kernel passes per-expert base offsets ``gx/gro/grsc/gco/gcsc`` and
    ``gmul=G`` to widen the SRDs over the whole 3D tensor (R,C stay per-expert).
    ``padded`` (non-256 K / non-128 N): X is the real [R,C] but the row-out runs to the
    column-tile grid and the scale/col-out to K_pad=CP / N_pad=RP (the kernel writes literal
    zero over every pad column, matching HIP); loads past real C mask to 0 and writes past
    those extents go to _OOB so the store drops them.
    ``rowpad`` (R != RP): row tiles cover R_pad, so the trailing ones are partly or wholly
    past the real R -- the SRD bound drops their row-side work and the col side writes zero."""
    if ncblk is None:
        ncblk = C // _TC
    cpad = CP if padded else C  # row-out column extent (K_pad)
    rpad = RP if padded else R  # col-out column extent (N_pad)
    # ROW_OUT is allocated on the column-tile grid rather than on K_pad, so an fp4 row is a
    # whole number of 128 B lines wide for every C and every row starts on a line. A row whose
    # width is 64 mod 128 costs its consumer's g2s an extra request per row, and the GEMM takes
    # the allocated width as ``row_bytes``, so the alignment is the quantiser's to give. The
    # scale keeps the K_pad width and therefore still carries the true contraction.
    dpad = cpad if (batched or not padded) else ncblk * _TC
    # Block order = which output's partial stores L2 can combine. col_locality: row-tile-fastest
    # so blocks writing the same col-out rows run back-to-back and L2 merges the scattered
    # transpose stores; else col-tile-fastest keeps the tile load and row-out on the same rows.
    # Only worth it while a col-out store is a partial line -- see ``_co_wide``.
    if col_locality:
        # 2D tiles the row dim over R_pad (its last tile may be short); batched-3D tiles the
        # real R, whose geometry is chosen to divide it.
        nrblk = (R if batched else rpad) // _TR
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
        # A whole-slab SRD gives both of these as host literals, which have no traced value
        # for index_cast to unwrap.
        _ix = lambda v: arith.index(v) if isinstance(v, int) else arith.index_cast(T.index, v)
        boff = arith.index_cast(T.i64, _ix(elem_off) * arith.index(elem_bytes))
        raw = arith._to_raw(base + boff)
        r = rocdl.readfirstlane(res=raw.type, src=raw)  # pin the SRD base to an SGPR
        base_v = r.result if hasattr(r, "result") else r
        if isinstance(nrec_bytes, int):
            nrec_bytes = arith.index(nrec_bytes)
        nr = arith.minui(arith.index_cast(T.index, nrec_bytes), arith.index(0x7FFFFFFF))
        return buffer_ops.create_buffer_resource_from_addr(base_v, num_records_bytes=nr)

    if batched:
        rsrc = _srd(X, gx, 4, R * (C >> 1) * 4)
        orsrc = _srd(ROW_OUT, gro, 4, R * (dpad >> 3) * 4)
        rscrsrc = _srd(ROW_SC, grsc, 1, R * (cpad >> 5))
        corsrc = _srd(COL_OUT, gco, 4, C * (rpad >> 3) * 4)
        cscrsrc = _srd(COL_SC, gcsc, 1, C * (rpad >> 5))
        gx = gro = grsc = gco = gcsc = 0  # expert bases folded into the SRDs above
    else:
        r0i = arith.index_cast(T.index, r0)
        c0i = arith.index_cast(T.index, cblk * _TC)
        # Row tiles cover R_pad, so the last one can run past the real R. Bounding the three
        # row-indexed SRDs by what is actually left makes the hardware do the masking: loads
        # past R read 0 and stores past R are dropped, with no per-access predicate.
        trow = _imin(R - r0, fx.Int32(_TR))
        rsrc = _srd(X, r0i * arith.index_cast(T.index, C >> 1), 4, trow * (C >> 1) * 4)
        orsrc = _srd(ROW_OUT, r0i * arith.index_cast(T.index, dpad >> 3), 4, trow * (dpad >> 3) * 4)
        if pack_row is None:
            rscrsrc = _srd(ROW_SC, r0i * arith.index_cast(T.index, cpad >> 5), 1, trow * (cpad >> 5))
        else:
            # The packed layout scatters a tile's rows over the whole slab, so this SRD cannot
            # be re-based per tile the way the canonical one is. Every term of num_records has
            # to come off a traced extent: a host literal is folded in without keying the
            # compile cache, and the next shape inherits this one's bound.
            _qm = ((R + 255) >> 8) << 8
            rscrsrc = _srd(ROW_SC, grsc, 1, _qm * ((C >> 7) * 4))
        corsrc = _srd(COL_OUT, c0i * arith.index_cast(T.index, rpad >> 3), 4, _TC * (rpad >> 3) * 4)
        if pack_col is None:
            cscrsrc = _srd(COL_SC, c0i * arith.index_cast(T.index, rpad >> 5), 1, _TC * (rpad >> 5))
        else:
            _qn = ((C + 255) >> 8) << 8
            cscrsrc = _srd(COL_SC, gcsc, 1, _qn * ((R >> 7) * 4))

    rstd_rsrc = None
    gamma_rsrc = None
    _rmsnorm_scales_row = None
    _rmsnorm_scales_col = None
    if rmsnorm_scale:
        # X is xpr; y = xpr * rstd * gamma is applied in-register before amax/RHT.
        r0i = arith.index_cast(T.index, r0)
        c0i = arith.index_cast(T.index, cblk * _TC)
        rstd_rsrc = _srd(RSTD, r0i, 4, arith.index(_TR * 4))
        gamma_rsrc = _srd(GAMMA, c0i, 4, arith.index(_TC * 4))

        def _f32_from_i32bits(bits):
            return Vec.from_elements([bits], fx.Int32).bitcast(fx.Float32)[0]

        def _rmsnorm_scales_row(r_row, cmb):
            rstd_v = _f32_from_i32bits(_lds_load1(lds.rbits.ptr, r_row))
            scales = []
            base = cmb * 32
            for q in range_constexpr(8):
                v4 = _lds_load_vec4(lds.gbits.ptr, base + q * 4)
                for j in range_constexpr(4):
                    scales.append(rstd_v * _f32_from_i32bits(v4[j]))
            return scales

        def _rmsnorm_scales_col(mmb, c_col):
            gv = _f32_from_i32bits(_lds_load1(lds.gbits.ptr, c_col))
            scales = []
            row0 = mmb * 32
            for q in range_constexpr(8):
                v4 = _lds_load_vec4(lds.rbits.ptr, row0 + q * 4)
                for j in range_constexpr(4):
                    scales.append(_f32_from_i32bits(v4[j]) * gv)
            return scales

    # ---- coalesced tile load -> LDS ----
    for chunk in range_constexpr(_NLOAD):
        tw = chunk * (BLK * 4) + tid * 4
        tr = tw // _TCW
        wc = tw % _TCW
        goff = (_row0 + tr) * (C >> 1) + c0w + wc + gx
        if padded:
            # mask cols past real C -> OOB load returns 0; rows past real R are already
            # bounded out by this tile's SRD.
            goff = arith.select((c0w + wc) < (C >> 1), goff, fx.Int32(_OOB))
        vec = buffer_ops.buffer_load(rsrc, goff, vec_width=4, dtype=T.i32, cache_modifier=_LOAD_AUX)
        _lds_store_vec4(lds.buf.ptr, tw, vec)
    if rmsnorm_scale:
        # One cooperative fill of the 256-col gamma tile and 64-row rstd tile.
        # Replaces 32 scalar global loads per microblock (same values reused
        # across rows / cols of this tile).
        gv = fx.Float32(buffer_ops.buffer_load(gamma_rsrc, _col0 + tid, vec_width=1, dtype=T.f32))
        _lds_store1(lds.gbits.ptr, tid, Vec.from_elements([gv], fx.Float32).bitcast(fx.Int32)[0])
        roff = arith.select(tid < _TR, _row0 + tid, fx.Int32(_OOB))
        rv = fx.Float32(buffer_ops.buffer_load(rstd_rsrc, roff, vec_width=1, dtype=T.f32))
        _lds_store1(lds.rbits.ptr, tid, Vec.from_elements([rv], fx.Float32).bitcast(fx.Int32)[0])
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

    def _zero_pad_cmb(gcmb, words, biased):
        """Force an all-pad row micro-block to literal zero.

        ROW_OUT runs to the column-tile grid, so the trailing micro-blocks quantise columns
        that do not exist; the same reasoning as ``_zero_pad_mb`` on the col side applies, and
        it also covers the K_pad range the caller used to memset.
        """
        cok = gcmb < (C >> 5)
        z = arith.constant(0)
        return [arith.select(cok, w, z) for w in words], arith.select(cok, biased, z)

    def _row_store_offs(gcmb, ob, sc):
        """Drop the stores this tile does not own: ROW_OUT to the tile grid, the scale to
        K_pad. Rows are bounded by the SRD, so only the column axis needs a predicate."""
        ob = arith.select(gcmb < (dpad >> 5), ob, fx.Int32(_OOB))
        return ob, arith.select(gcmb < (cpad >> 5), sc, fx.Int32(_OOB))

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
            # Scratch is micro-block-column major: the reduction below runs along the ROW axis,
            # so keeping that axis contiguous turns its 32 reads into 8 vec4 ones.
            _lds_store1(lds.scr.ptr, cmb * _SCP + r_row, _microblock_amax_f(vf))
            vf_hold.append(vf)
            meta.append((r_row, cmb))
        fx.barrier()
        for k in range_constexpr(_RROWTASK):
            r_row, cmb = meta[k]
            vf = vf_hold[k]
            row_base = (r_row // 32) * 32  # tile's first row within the LDS tile
            tile_amax = fx.Int32(0)
            for q in range_constexpr(8):
                v4 = _lds_load_vec4(lds.scr.ptr, cmb * _SCP + row_base + q * 4)
                for j in range_constexpr(4):
                    tile_amax = _imax(tile_amax, v4[j])
            native_bits, rbiased = _compute_scale_native(
                tile_amax, scale_rounding_bias, exp_up=vf_exp_up(row_rht)
            )
            rwords = _cvt_microblock_to_fp4(vf, arith.bitcast(T.f32, native_bits), _row_seed(k))
            grow = _row0 + r_row
            gcmb = cblk * _RMBC + cmb
            ob = grow * (dpad >> 3) + gcmb * 4 + gro
            sc = (
                grow * (cpad >> 5) + gcmb + grsc
                if pack_row is None
                # The canonical store is tile-relative (its SRD is re-based per tile); the
                # packed slab is addressed whole, so this one needs the global row.
                else mxfp4_packed_scale_byte(
                    r0 + r_row,
                    gcmb,
                    k128=pack_row["k128"],
                    kk=C >> 8,  # (C // 128) // 2, traced
                    b_ilv=pack_row["b_ilv"],
                    is_b=pack_row["is_b"],
                )
            )
            if padded:
                rwords, rbiased = _zero_pad_cmb(gcmb, rwords, rbiased)
                ob, sc = _row_store_offs(gcmb, ob, sc)
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
            rwords, rbiased = _finish_microblock(
                rbits,
                row_rht,
                scale_rounding_bias,
                seed=_row_seed(k),
                scales=_rmsnorm_scales_row(r_row, cmb) if rmsnorm_scale else None,
            )
            grow = _row0 + r_row
            gcmb = cblk * (_TC // 32) + cmb
            ob = grow * (dpad >> 3) + gcmb * 4 + gro
            sc = (
                grow * (cpad >> 5) + gcmb + grsc
                if pack_row is None
                # The canonical store is tile-relative (its SRD is re-based per tile); the
                # packed slab is addressed whole, so this one needs the global row.
                else mxfp4_packed_scale_byte(
                    r0 + r_row,
                    gcmb,
                    k128=pack_row["k128"],
                    kk=C >> 8,  # (C // 128) // 2, traced
                    b_ilv=pack_row["b_ilv"],
                    is_b=pack_row["is_b"],
                )
            )
            if padded:
                rwords, rbiased = _zero_pad_cmb(gcmb, rwords, rbiased)
                ob, sc = _row_store_offs(gcmb, ob, sc)
            _store_words_vec4(orsrc, ob, rwords)
            buffer_ops.buffer_store(arith.trunci(T.i8, rbiased & 0xFF), rscrsrc, sc)

    # ---- COL phase: thread = column, 32-row microblocks (strided LDS reads) ----
    c_col = tid
    half = c_col & 1
    cw = c_col >> 1
    # `gcol` below is tile-relative when the tile's col base is folded into the SRD (2D) and
    # global when it is not (batched-3D); the pad mask needs the true global column either way.
    c_glob = cblk * _TC + c_col
    # Lifting the lane's bf16 half of a packed word into f32 position is a pure byte
    # permutation, so one v_perm_b32 replaces the (and, shl, cndmask) triple on each of the
    # 128 elements a thread casts. Selector byte 12 emits a literal zero and 4..7 pick the
    # packed word's own bytes; only the selector depends on the lane, and it is invariant.
    hsel = arith.select(half != 0, fx.Int32(0x07060C0C), fx.Int32(0x05040C0C))

    def _col_half(word):
        return fx.Int32(rocdl.perm_b32(word, word, hsel))

    # A col scale byte lands one per lane, `rpad/32` bytes from its neighbour's, so every one
    # of them is its own write request. The _RMB a thread owns are consecutive in memory and
    # 4-aligned when the row tile is a 128-multiple (rpad always is), so pack four per dword.
    cs_packed = pack_col is None and _RMB % 4 == 0

    def _store_col_scales(biased):
        base = ((_col0 + c_col) * (rpad >> 5) + rblk * _RMB + gcsc) >> 2
        for q in range_constexpr(_RMB // 4):
            w = biased[4 * q] & 0xFF
            for j in range_constexpr(1, 4):
                w = w | ((biased[4 * q + j] & 0xFF) << (8 * j))
            off = base + q
            if padded:
                off = arith.select(c_glob < C, off, fx.Int32(_OOB))
            buffer_ops.buffer_store(w, cscrsrc, off)

    def _zero_pad_mb(gmmb, words, biased):
        """Force an all-pad col micro-block to literal zero.

        Row tiles run to R_pad, so the trailing ones quantise rows that do not exist. The HIP
        dual writes zero over that range and the GEMM contracts over the padded extent, so the
        cvt of an all-zero block (its e8m0 scale is 0) must not be trusted to land there.
        """
        mok = gmmb < (R >> 5)
        z = arith.constant(0)
        return [arith.select(mok, w, z) for w in words], arith.select(mok, biased, z)

    # A lane owns one column's _RMB micro-blocks, i.e. _RMB*16 = 64 CONSECUTIVE col-out bytes,
    # but a buffer_store is 16 B wide, so the four of them hit the same 64 B line four times:
    # PMC reads 256 write requests per wave for 4096 B of payload, a 4x amplification, and
    # col-out alone is 256 of the kernel's 416 write requests per wave. Give each line to one
    # quad of lanes instead -- lanes 4a..4a+3 take micro-blocks 0..3 of the same column -- and
    # every request becomes a whole line. The regroup is one LDS round trip through the (now
    # dead) tile buffer; both phases sit at the b128 floor of 8 words per bank.
    co_wide = _co_wide(_TR, _TC)

    def _store_col_out(held):
        """Write the tile's _RMB colwise micro-blocks per column."""
        if not co_wide:
            for mmb in range_constexpr(_RMB):
                cob = (_col0 + c_col) * (rpad >> 3) + (rblk * _RMB + mmb) * 4 + gco
                if padded:
                    cob = arith.select(c_glob < C, cob, fx.Int32(_OOB))
                _store_words_vec4(corsrc, cob, held[mmb])
            return
        fx.barrier()  # the col phase above is still reading lds.buf
        for mmb in range_constexpr(_RMB):
            _lds_store_vec4(
                lds.buf.ptr, mmb * (BLK * 4) + tid * 4, Vec.from_elements(list(held[mmb]), fx.Int32)
            )
        fx.barrier()
        mm, cq = tid & (_RMB - 1), tid >> 2
        for q in range_constexpr(_RMB):
            cc = cq + q * (BLK // _RMB)
            v = _lds_load_vec4(lds.buf.ptr, mm * (BLK * 4) + cc * 4)
            cob = (_col0 + cc) * (rpad >> 3) + (rblk * _RMB + mm) * 4 + gco
            if padded:
                cob = arith.select((cblk * _TC + cc) < C, cob, fx.Int32(_OOB))
            _store_words_vec4(corsrc, cob, [v[j] for j in range_constexpr(4)])

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
        cbs, cws = [], []
        for mmb in range_constexpr(_RMB):
            vf = cvf_hold[mmb]
            tile_amax = fx.Int32(0)
            for q in range_constexpr(8):
                v4 = _lds_load_vec4(lds.scr.ptr, mmb * _TC + col_base + q * 4)
                for j in range_constexpr(4):
                    tile_amax = _imax(tile_amax, v4[j])
            native_bits, cbiased = _compute_scale_native(
                tile_amax, scale_rounding_bias, exp_up=vf_exp_up(col_rht)
            )
            cwords = _cvt_microblock_to_fp4(vf, arith.bitcast(T.f32, native_bits), _col_seed(mmb))
            gcol = _col0 + c_col
            gmmb = rblk * _RMB + mmb
            if rowpad:
                cwords, cbiased = _zero_pad_mb(gmmb, cwords, cbiased)
            csoff = (
                gcol * (rpad >> 5) + gmmb + gcsc
                if pack_col is None
                else mxfp4_packed_scale_byte(
                    cblk * _TC + c_col,
                    gmmb,
                    k128=pack_col["k128"],
                    kk=R >> 8,  # (R // 128) // 2, traced
                    b_ilv=pack_col["b_ilv"],
                    is_b=pack_col["is_b"],
                )
            )
            if padded:
                csoff = arith.select(c_glob < C, csoff, fx.Int32(_OOB))
            cws.append(cwords)
            if cs_packed:
                cbs.append(cbiased)
            else:
                buffer_ops.buffer_store(arith.trunci(T.i8, cbiased & 0xFF), cscrsrc, csoff)
        if cs_packed:
            _store_col_scales(cbs)
        _store_col_out(cws)
    else:
        cbs, cws = [], []
        for mmb in range_constexpr(_RMB):
            row0 = mmb * 32
            cbits = []
            for row in range_constexpr(32):
                word = _lds_load1(lds.buf.ptr, (row0 + row) * _TCW + cw)
                cbits.append(_col_half(word))
            cwords, cbiased = _finish_microblock(cbits, col_rht, scale_rounding_bias, _col_seed(mmb))
            gcol = _col0 + c_col
            gmmb = rblk * _RMB + mmb
            if rowpad:
                cwords, cbiased = _zero_pad_mb(gmmb, cwords, cbiased)
            csoff = (
                gcol * (rpad >> 5) + gmmb + gcsc
                if pack_col is None
                else mxfp4_packed_scale_byte(
                    cblk * _TC + c_col,
                    gmmb,
                    k128=pack_col["k128"],
                    kk=R >> 8,  # (R // 128) // 2, traced
                    b_ilv=pack_col["b_ilv"],
                    is_b=pack_col["is_b"],
                )
            )
            if padded:
                csoff = arith.select(c_glob < C, csoff, fx.Int32(_OOB))
            cws.append(cwords)
            if cs_packed:
                cbs.append(cbiased)
            else:
                buffer_ops.buffer_store(arith.trunci(T.i8, cbiased & 0xFF), cscrsrc, csoff)
        if cs_packed:
            _store_col_scales(cbs)
        _store_col_out(cws)


def _build_dual_kernel(
    row_rht,
    col_rht,
    row_2d=False,
    col_2d=False,
    col_locality=False,
    row_sr=False,
    col_sr=False,
    pack_row=None,
    pack_col=None,
    padded=False,
    rowpad=False,
):
    """Single-recipe fused LDS dual (one coalesced 32x256 tile load feeds both the
    rowwise and colwise-transpose casts). Thin wrapper over ``_emit_dual_body``.
    ``col_locality`` flips the block order to combine partial-line transpose
    stores; see ``_emit_dual_body``. ``row_sr``/``col_sr`` enable
    stochastic rounding on that direction (uses the per-launch ``SR_SEED``).
    ``padded`` sizes the outputs on CP/RP and masks the ragged tail, as in the 3D path."""
    _DualSS = _make_dual_struct(bool(row_2d or col_2d))

    @flyc.kernel(known_block_size=[BLK, 1, 1])
    def _dual_kernel(
        X: fx.Tensor,  # int32 view [R, C/2]
        ROW_OUT: fx.Tensor,  # int32 view [R, CP/8]
        ROW_SC: fx.Tensor,  # uint8 [R, CP/32]
        COL_OUT: fx.Tensor,  # int32 view [C, RP/8]
        COL_SC: fx.Tensor,  # uint8 [C, RP/32]
        R: fx.Int32,
        C: fx.Int32,
        CP: fx.Int32,  # C_pad (row-out cols); == C when aligned
        RP: fx.Int32,  # R_pad (col-out cols); == R when aligned
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
            padded=padded,
            ncblk=(((C + _TC - 1) // _TC) if padded else (C // _TC)),
            CP=CP,
            RP=RP,
            rowpad=rowpad,
            col_locality=col_locality,
            row_sr=row_sr,
            col_sr=col_sr,
            sr_seed=SR_SEED,
            pack_row=pack_row,
            pack_col=pack_col,
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
    pack_row=None,
    pack_col=None,
    padded=False,
    rowpad=False,
):
    kern = _build_dual_kernel(
        row_rht,
        col_rht,
        row_2d,
        col_2d,
        col_locality,
        row_sr,
        col_sr,
        pack_row,
        pack_col,
        padded,
        rowpad,
    )

    @flyc.jit
    def _dual_launch(
        X: fx.Tensor,
        ROW_OUT: fx.Tensor,
        ROW_SC: fx.Tensor,
        COL_OUT: fx.Tensor,
        COL_SC: fx.Tensor,
        R: fx.Int32,
        C: fx.Int32,
        CP: fx.Int32,
        RP: fx.Int32,
        SR_SEED: fx.Int32,
        SCALE_ROUNDING_BIAS: fx.Int32,
        grid_x: fx.Int32,
        stream: fx.Stream,
    ):
        kern(X, ROW_OUT, ROW_SC, COL_OUT, COL_SC, R, C, CP, RP, SR_SEED, SCALE_ROUNDING_BIAS).launch(
            grid=(grid_x, 1, 1), block=(BLK, 1, 1), stream=stream
        )

    return _dual_launch


_DUAL_LAUNCH = {}
_DUAL_COMPILED = {}


def dual_eligible(R, C, row_recipe, col_recipe):
    """True if the FlyDSL fused dual can wholesale-replace the C++ dual for these
    recipes/dims (no preshuffle). Both the per-microblock (2d=F) and the 2d-block
    (2d=T weight) amax geometries are supported and bit-exact vs C++ (non-SR); SR is
    supported (unbiased, not bit-exact). Shuffled recipes still fall back.
    Non-256 C / non-128 R go through C_pad/R_pad exactly as the batched-3D path does,
    so the bound is the tiling itself: R%64 (row tile) and C%64 (32-microblock plus
    vec4-aligned tail load mask)."""
    return (
        not row_recipe.shuffle_scale
        and not row_recipe.shuffle_out
        and not col_recipe.shuffle_scale
        and not col_recipe.shuffle_out
        and (R % 64 == 0)
        and (C % 64 == 0)
    )


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
    pack_row=None,
    pack_col=None,
):
    """Fused rowwise + colwise-transpose mxfp4 cast (one bf16 read). Returns
    (row_data, row_scale, col_data, col_scale) in C++-compatible dtypes/shapes.
    ``row_sr``/``col_sr`` request stochastic rounding on that direction."""
    import torch

    R, C = x_bf16.shape
    dev = x_bf16.device
    x_i32 = x_bf16.view(torch.int32)  # [R, C/2]
    fn, grid_x, CP, RP, padded, DP = get_dual_cast(
        R, C, row_rht, col_rht, row_2d, col_2d, row_sr, col_sr, pack_row, pack_col
    )
    # Outputs are sized on the pad extents, the same contract as the batched-3D dual: the pad
    # must read back all-0 to match the HIP dual (the GEMM contracts over the padded extent).
    # The row-out runs to DP, a whole number of 128 B lines per row so the GEMM's g2s gets one
    # request per row; the kernel writes its pad, so nothing here has to memset.
    ro = torch.empty((R, DP // 8), dtype=torch.int32, device=dev)
    co = torch.empty((C, RP // 8), dtype=torch.int32, device=dev)

    def _scale_out(pack, dim, k128):
        # A packed slab is byte-addressed and flat; the canonical one keeps its [dim, K/32].
        if pack is None:
            return torch.empty((dim, k128 * 4), dtype=torch.uint8, device=dev)
        return torch.empty((dim + 255) // 256 * 256 * k128 * 4, dtype=torch.uint8, device=dev)

    rs = _scale_out(pack_row, R, CP // 128)
    cs = _scale_out(pack_col, C, RP // 128)
    sr_seed = _next_sr_seed() if (row_sr or col_sr) else 0
    fn(
        x_i32,
        ro,
        rs,
        co,
        cs,
        R,
        C,
        CP,
        RP,
        sr_seed,
        _mxfp4_scale_rounding_bias(scale_rounding_mode),
        grid_x,
        torch.cuda.current_stream(),
    )
    row_data = ro.view(torch.uint8).view(fp4_dtype)  # [R, DP/2] fp4
    col_data = co.view(torch.uint8).view(fp4_dtype)  # [C, RP/2] fp4
    # A packed slab is the GEMM's own i32 layout, so it goes back as i32 -- that is what the
    # backend looks at to recognise it. The canonical one stays e8m0.
    row_scale = rs.view(torch.int32) if pack_row else rs.view(torch.float8_e8m0fnu)
    col_scale = cs.view(torch.int32) if pack_col else cs.view(torch.float8_e8m0fnu)
    return row_data, row_scale, col_data, col_scale


def get_dual_cast(
    R,
    C,
    row_rht,
    col_rht,
    row_2d=False,
    col_2d=False,
    row_sr=False,
    col_sr=False,
    pack_row=None,
    pack_col=None,
):
    """Return (compiled_fn, grid_x, CP, RP, padded, DP) for the fused dual at
    (R, C, row_rht, col_rht, row_2d, col_2d, row_sr, col_sr). CP=ceil(C/128)*128 (row scale),
    RP=ceil(R/128)*128 (col-out), DP=ceil(C/_TC)*_TC (row-out data, so its fp4 row pitch is a
    whole number of 128 B lines); ``padded`` when C is not a _TC-tile multiple or R is not a
    128-multiple. Requires R % 64 == 0 and C % 64 == 0. The row tile is picked per shape by
    ``_pick_tile_geom`` (COL_OUT's write granule is _TR/2 bytes), as in the batched path."""
    CP = ((int(C) + 127) // 128) * 128
    RP = ((int(R) + 127) // 128) * 128
    rowpad = RP != int(R)  # row tiles cover R_pad; the trailing ones run past the real R
    # pack_row changes the emitted store, so it keys both caches.
    pk = None if pack_row is None else tuple(sorted(pack_row.items()))
    pc = None if pack_col is None else tuple(sorted(pack_col.items()))
    tr, tc = _pick_tile_geom(RP, int(C))  # R_pad is always a 128-multiple -> the 128-row tile
    DP = ((int(C) + tc - 1) // tc) * tc
    col_locality = int(C) > int(R) and not _co_wide(tr, tc)
    global _LOAD_AUX
    saved, saved_aux = (_TR, _TC), _LOAD_AUX
    _set_tile_geom(tr, tc)
    _LOAD_AUX = _pick_load_aux(int(R) * int(C))
    try:
        padded = (int(C) % _TC != 0) or (int(R) % 128 != 0)
        lk = (
            bool(row_rht),
            bool(col_rht),
            bool(row_2d),
            bool(col_2d),
            col_locality,
            bool(row_sr),
            bool(col_sr),
            pk,
            pc,
            padded,
            rowpad,
            tr,
            tc,
            _LOAD_AUX,
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
                pack_row,
                pack_col,
                padded,
                rowpad,
            )
            _DUAL_LAUNCH[lk] = raw
        key = (int(R), int(C), *lk)
        ent = _DUAL_COMPILED.get(key)
        if ent is None:
            import torch

            x = torch.zeros((R, C // 2), dtype=torch.int32, device="cuda")
            ro = torch.zeros((R, DP // 8), dtype=torch.int32, device="cuda")
            rs = (
                torch.zeros((R, CP // 32), dtype=torch.uint8, device="cuda")
                if pack_row is None
                # byte-addressed like the canonical slab: the packed store is still one i8 per
                # thread, it just lands somewhere else.
                else torch.zeros(pack_row["qm"] * pack_row["k128"] * 4, dtype=torch.uint8, device="cuda")
            )
            co = torch.zeros((C, RP // 8), dtype=torch.int32, device="cuda")
            cs = (
                torch.zeros((C, RP // 32), dtype=torch.uint8, device="cuda")
                if pack_col is None
                else torch.zeros(pack_col["qn"] * pack_col["k128"] * 4, dtype=torch.uint8, device="cuda")
            )
            ncblk = ((C + _TC - 1) // _TC) if padded else (C // _TC)
            grid_x = (RP // _TR) * ncblk
            stream = torch.cuda.current_stream()
            fn = flyc.compile(raw, x, ro, rs, co, cs, R, C, CP, RP, 0, 1 << 21, grid_x, stream)
            ent = (fn, grid_x, CP, RP, padded, DP)
            _DUAL_COMPILED[key] = ent
    finally:
        _set_tile_geom(*saved)
        _LOAD_AUX = saved_aux
    return ent


_RMSNORM_DUAL_LAUNCH = {}
_RMSNORM_DUAL_COMPILED = {}


def _build_rmsnorm_dual_kernel(row_rht, col_rht, col_locality=False):
    """Dual of residual-sum ``xpr`` with in-register ``rstd*gamma`` (activation recipe)."""

    @fx.struct
    class _RmsDualSS:
        buf: fx.Array[fx.Int32, _RMS_NW, 16]
        gbits: fx.Array[fx.Int32, _RMS_TC, 16]
        rbits: fx.Array[fx.Int32, BLK, 16]

    _DualSS = _RmsDualSS

    @flyc.kernel(known_block_size=[BLK, 1, 1])
    def _rmsnorm_dual_kernel(
        X: fx.Tensor,
        ROW_OUT: fx.Tensor,
        ROW_SC: fx.Tensor,
        COL_OUT: fx.Tensor,
        COL_SC: fx.Tensor,
        RSTD: fx.Tensor,
        GAMMA: fx.Tensor,
        R: fx.Int32,
        C: fx.Int32,
        SR_SEED: fx.Int32,
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
            sr_seed=SR_SEED,
            sr_gbid=fx.block_idx.x,
            rmsnorm_scale=True,
            RSTD=RSTD,
            GAMMA=GAMMA,
            tile_tr=_RMS_TR,
            tile_tc=_RMS_TC,
        )

    return _rmsnorm_dual_kernel


def _build_rmsnorm_dual_launch(row_rht, col_rht, col_locality=False):
    kern = _build_rmsnorm_dual_kernel(row_rht, col_rht, col_locality)

    @flyc.jit
    def _rmsnorm_dual_launch(
        X: fx.Tensor,
        ROW_OUT: fx.Tensor,
        ROW_SC: fx.Tensor,
        COL_OUT: fx.Tensor,
        COL_SC: fx.Tensor,
        RSTD: fx.Tensor,
        GAMMA: fx.Tensor,
        R: fx.Int32,
        C: fx.Int32,
        SR_SEED: fx.Int32,
        SCALE_ROUNDING_BIAS: fx.Int32,
        grid_x: fx.Int32,
        stream: fx.Stream,
    ):
        kern(X, ROW_OUT, ROW_SC, COL_OUT, COL_SC, RSTD, GAMMA, R, C, SR_SEED, SCALE_ROUNDING_BIAS).launch(
            grid=(grid_x, 1, 1), block=(BLK, 1, 1), stream=stream
        )

    return _rmsnorm_dual_launch


def flydsl_rmsnorm_dual_quant(xpr_bf16, rstd_f32, gamma_f32, fp4_dtype, col_rht=True, scale_rounding_mode=0):
    """MXFP4 dual of ``y = xpr * rstd[:,None] * gamma`` without a BF16 ``y`` load.

    ``xpr_bf16`` is ``[R, C]`` bf16 (residual sum). ``rstd_f32`` is ``[R]``.
    ``gamma_f32`` is ``[C]``. Row recipe is no-RHT; col recipe is RHT when
    ``col_rht`` (matches dense MLP / QKV activation ``quantize_fp4_with_trans``).

    This is close to, but NOT the same as, quantizing a materialised BF16 ``y``:
    ``rstd*gamma`` is applied in f32 registers, so the bf16 rounding of ``y`` never
    happens. Measured on [8192, 4096] that moves 1.1% of the output bytes (and some
    scales). It is the more accurate of the two; a test must compare with a tolerance,
    not with ``torch.equal`` against ``quantize_fp4_with_trans(y)``.
    """
    import torch

    R, C = xpr_bf16.shape
    rstd_f32 = rstd_f32.reshape(-1).contiguous()
    gamma_f32 = gamma_f32.reshape(-1).contiguous()
    assert rstd_f32.shape == (R,) and gamma_f32.shape == (C,)
    assert xpr_bf16.dtype == torch.bfloat16
    assert rstd_f32.dtype == torch.float32 and gamma_f32.dtype == torch.float32
    # R % 256, not the kernel's own R % 128: the col outputs below are [C, R/8] unpadded
    # while the GEMM's activation buffers round M up to 256.
    assert R % 256 == 0 and C % 256 == 0
    dev = xpr_bf16.device
    x_i32 = xpr_bf16.contiguous().view(torch.int32)
    ro = torch.empty((R, C // 8), dtype=torch.int32, device=dev)
    rs = torch.empty((R, C // 32), dtype=torch.uint8, device=dev)
    co = torch.empty((C, R // 8), dtype=torch.int32, device=dev)
    cs = torch.empty((C, R // 32), dtype=torch.uint8, device=dev)
    fn, grid_x = get_rmsnorm_dual_cast(R, C, False, bool(col_rht))
    fn(
        x_i32,
        ro,
        rs,
        co,
        cs,
        rstd_f32.contiguous(),
        gamma_f32.contiguous(),
        R,
        C,
        0,
        _mxfp4_scale_rounding_bias(scale_rounding_mode),
        grid_x,
        torch.cuda.current_stream(),
    )
    row_data = ro.view(torch.uint8).view(fp4_dtype)
    col_data = co.view(torch.uint8).view(fp4_dtype)
    row_scale = rs.view(torch.float8_e8m0fnu)
    col_scale = cs.view(torch.float8_e8m0fnu)
    return row_data, row_scale, col_data, col_scale


def get_rmsnorm_dual_cast(R, C, row_rht, col_rht):
    col_locality = int(C) > int(R)
    lk = (bool(row_rht), bool(col_rht), col_locality)
    raw = _RMSNORM_DUAL_LAUNCH.get(lk)
    if raw is None:
        raw = _build_rmsnorm_dual_launch(bool(row_rht), bool(col_rht), col_locality)
        _RMSNORM_DUAL_LAUNCH[lk] = raw
    key = (int(R), int(C), bool(row_rht), bool(col_rht))
    ent = _RMSNORM_DUAL_COMPILED.get(key)
    if ent is None:
        import torch

        x = torch.zeros((R, C // 2), dtype=torch.int32, device="cuda")
        ro = torch.zeros((R, C // 8), dtype=torch.int32, device="cuda")
        rs = torch.zeros((R, C // 32), dtype=torch.uint8, device="cuda")
        co = torch.zeros((C, R // 8), dtype=torch.int32, device="cuda")
        cs = torch.zeros((C, R // 32), dtype=torch.uint8, device="cuda")
        rstd = torch.zeros((R,), dtype=torch.float32, device="cuda")
        gamma = torch.zeros((C,), dtype=torch.float32, device="cuda")
        grid_x = (R // _RMS_TR) * (C // _RMS_TC)
        stream = torch.cuda.current_stream()
        fn = flyc.compile(raw, x, ro, rs, co, cs, rstd, gamma, R, C, 0, 1 << 21, grid_x, stream)
        ent = (fn, grid_x)
        _RMSNORM_DUAL_COMPILED[key] = ent
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
    col_locality = int(K) > int(N) and not _co_wide(tr, tc)
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
    global _LOAD_AUX
    _saved, _saved_aux = (_TR, _TC), _LOAD_AUX
    _set_tile_geom(tr, tc)
    _LOAD_AUX = _pick_load_aux(int(G) * int(N) * int(K))
    lk = lk + (tr, tc, _LOAD_AUX)
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
        _LOAD_AUX = _saved_aux
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
            x,
            ro,
            rs,
            co,
            cs,
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
    _LOAD_AUX = _saved_aux
    return ent


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
    dev = x3d.device
    x_i32 = x3d.contiguous().view(torch.int32)  # [G, N, K/2]
    scale_rounding_bias = _mxfp4_scale_rounding_bias(scale_rounding_mode)
    fn, grid_x, Kp, Np, padded = get_dual3_cast(
        N,
        K,
        G,
        row_rht,
        col_rht,
        row_2d,
        col_2d,
        row_sr,
        col_sr,
        scale_rounding_bias,
    )
    # Outputs sized on K_pad/N_pad; the pad regions must read back all-0 to match the HIP
    # dual (the GEMM contracts over the PADDED extent, so pad garbage would corrupt it).
    # Zeroing the whole buffer to achieve that costs ~570 MB of memset per weight quant
    # (G,N,Kp/8 + G,K,Np/8 i32 for GG1) to protect at most a few percent of tail columns.
    # Allocate uninitialised and zero only the tail slices past the real K / N extent.
    ro = torch.empty((G, N, Kp // 8), dtype=torch.int32, device=dev)
    co = torch.empty((G, K, Np // 8), dtype=torch.int32, device=dev)
    rs = torch.empty((G, N, Kp // 32), dtype=torch.uint8, device=dev)
    cs = torch.empty((G, K, Np // 32), dtype=torch.uint8, device=dev)
    if padded:
        if Kp != K:
            ro[..., K // 8 :].zero_()
            rs[..., K // 32 :].zero_()
        if Np != N:
            co[..., N // 8 :].zero_()
            cs[..., N // 32 :].zero_()
    sr_seed = _next_sr_seed() if (row_sr or col_sr) else 0
    fn(
        x_i32,
        ro,
        rs,
        co,
        cs,
        N,
        K,
        G,
        Kp,
        Np,
        sr_seed,
        grid_x,
        torch.cuda.current_stream(),
    )
    return (
        ro.view(torch.uint8).view(fp4_dtype),
        rs.view(torch.float8_e8m0fnu),
        co.view(torch.uint8).view(fp4_dtype),
        cs.view(torch.float8_e8m0fnu),
    )
