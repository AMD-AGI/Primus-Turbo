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
from flydsl.expr import arith, buffer_ops, range_constexpr, rocdl
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec
from flydsl.expr.utils.arith import _to_raw as _raw

_OOB = 0x7FFFFFFF  # word offset past any SRD -> buffer_load returns 0 / buffer_store dropped
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


def _compute_scale_native(amax_bits):
    """e8m0 scale, all-int32 (matches compute_tile_scale). Returns
    (scale_native_f32bits_i32, scale_e8m0_biased_i32)."""
    val_to_add = 1 << 21  # 1 << (23 - 1 - 1)
    hp_exp_mask = 0x1FF  # (1 << 9) - 1
    extracted = ((amax_bits + val_to_add) >> 23) & hp_exp_mask
    extracted = extracted - 127 - 2  # - hp_exp_bias - FP4_TARGET_MAX_POW2
    extracted = _imax(extracted, -127)
    extracted = arith.select(extracted < 128, extracted, 128)
    biased = extracted + 127  # 0..255
    native_bits = biased << 23  # 2^(biased-127) as f32 bits
    return native_bits, biased


def _h4(v0, v1, v2, v3):
    """One H4 butterfly, same float order as rht16_inplace stage-1 / cross-lane."""
    a0 = v0 + v1
    a1 = v0 - v1
    a2 = v2 + v3
    a3 = v2 - v3
    return a0 + a2, a1 + a3, a0 - a2, a1 - a3


def _rht16(v):
    """In-register H16 = H4(local) then H4(across 4 blocks), * 0.25.
    ``v`` is a list of 16 f32 Values, element index e = 4*block + local."""
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
        r[0 * 4 + lc] = y0 * 0.25
        r[1 * 4 + lc] = y1 * 0.25
        r[2 * 4 + lc] = y2 * 0.25
        r[3 * 4 + lc] = y3 * 0.25
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


def _microblock_vf(vbits, use_rht):
    """32 f32-bit i32 values -> list of 32 f32 Values (post-RHT if enabled)."""
    vf = [Vec.from_elements([b], fx.Int32).bitcast(fx.Float32)[0] for b in vbits]
    if use_rht:
        vf = _rht16(vf[0:16]) + _rht16(vf[16:32])
    return vf


def _microblock_amax(vf):
    """int-max over abs bits of 32 f32 Values (matches C++ fabs-reduce, bit-exact)."""
    amax = fx.Int32(0)
    for i in range_constexpr(32):
        b = Vec.from_elements([vf[i]], fx.Float32).bitcast(fx.Int32)[0]
        amax = _imax(amax, _abs_i32(b))
    return amax


# ---- packed-pair microblock (bf16 source, no RHT / no SR) ----------------------
# The 16-bit source never has to be widened to f32: bf16->f32 is exact, so both the
# amax and the fp4 cvt can run straight on the packed LDS word. That removes the
# per-element extract and halves the abs+max tree, the two biggest VALU blocks of the
# cast (96 of the 104 VALU per microblock measured in the ISA; 50 after).
_ABS16X2 = 0x7FFF7FFF  # clears both sign bits of a packed 16-bit-float pair


def _packed_amax_bits(words):
    """f32 bit-pattern of max|v| over the 2*len(words) bf16 packed in ``words``.
    Bit-identical to the int-max over the unpacked abs f32 bits: clearing the sign
    leaves the magnitude order equal to the unsigned 16-bit integer order, and the
    bf16->f32 widening (<<16) is monotone, so the max commutes with both. Reducing on
    the pair (v_pk_max_u16) halves both the abs mask and the max tree."""
    v = [Vec.from_elements([w & _ABS16X2], fx.Int32).bitcast(fx.Int16) for w in words]
    m = v[0]
    for i in range_constexpr(1, len(v)):
        m = arith.maxui(m, v[i])
    m32 = fx.Int32(m.bitcast(fx.Int32)[0])
    return _imax(m32 & 0xFFFF, m32 >> 16) << 16


def _cvt_microblock_bf16(words, scale_native_f32):
    """``len(words)`` packed bf16 pairs (element 2i in the low half of word i) -> fp4 i32
    words, same packing / dst_sel chaining as ``_cvt_microblock_to_fp4``. The bf16 form
    of the cvt consumes the LDS word directly, so the pair is never widened to f32."""
    out = []
    for wi in range_constexpr(len(words) // 4):
        acc = fx.Int32(0)
        for pair in range_constexpr(4):
            src = Vec.from_elements([words[wi * 4 + pair]], fx.Int32).bitcast(fx.BFloat16)
            acc = fx.Int32(
                rocdl.cvt_scalef32_pk_fp4_bf16(T.i32, _raw(acc), _raw(src), scale_native_f32, pair)
            )
        out.append(acc)
    return out


def _finish_microblock_bf16(words):
    """Packed-pair fast path of ``_finish_microblock``: packed bf16 words -> (fp4 i32
    words, scale_e8m0 i8-ready i32). bf16 source only, no RHT, no SR."""
    native_bits, biased = _compute_scale_native(_packed_amax_bits(words))
    return _cvt_microblock_bf16(words, arith.bitcast(T.f32, native_bits)), biased


def _finish_microblock(vbits, use_rht, seed=None, scales=None):
    """32 f32-bit i32 values -> (4 fp4 i32 words, scale_e8m0 i8-ready i32).
    ``seed`` (i32 Value) enables stochastic rounding in the final cvt (amax/scale
    stay deterministic). ``scales`` (32 f32 Values) is applied after unpack and
    before RHT so RMSNorm ``y = xpr * rstd * gamma`` can feed this dual without
    a BF16 ``y`` round-trip."""
    if scales is None:
        vf = _microblock_vf(vbits, use_rht)
    else:
        vf = [Vec.from_elements([b], fx.Int32).bitcast(fx.Float32)[0] for b in vbits]
        vf = [vf[i] * scales[i] for i in range_constexpr(32)]
        if use_rht:
            vf = _rht16(vf[0:16]) + _rht16(vf[16:32])
    amax = _microblock_amax(vf)
    native_bits, biased = _compute_scale_native(amax)
    words = _cvt_microblock_to_fp4(vf, arith.bitcast(T.f32, native_bits), seed)
    return words, biased


# ---- fused-dual tile geometry (shared by the 2D and batched-3D kernels) ----
_TR = 64  # tile rows (R dim); 2 col-microblocks/tile, 32KB LDS (occ 2)
_TC = 256  # tile cols (C dim)
_TCW = _TC // 2  # 128 i32 words per tile row
_NW = _TR * _TCW  # 8192 i32 words in LDS
_RMB = _TR // 32  # col m-microblocks per tile
_NLOAD = (_NW + BLK * 4 - 1) // (BLK * 4)  # vec4 loads per thread
_RROWTASK = (_TR * (_TC // 32)) // BLK  # row tasks per thread
_RMBC = _TC // 32  # row micro-blocks along C (== 8)
_NSCR = _TR * _RMBC  # LDS amax scratch elems (64x8 = 512 i32 = 2KB)
# Scaled dual: half-height tile so buf+gamma+rstd LDS stays under 32 KB (5 WG/CU).
_RMS_TR = 32
_RMS_TC = _TC
_RMS_NW = _RMS_TR * (_RMS_TC // 2)  # 4096 i32 = 16 KB


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
    rmsnorm_scale=False,
    RSTD=None,
    GAMMA=None,
    tile_tr=None,
    tile_tc=None,
):
    """Emit one fused-dual tile (rowwise + colwise-transpose mxfp4 cast) for block
    ``bid``. ``row_2d``/``col_2d`` pick the C++ ``USE_2D_BLOCK`` amax geometry; the
    batched-3D kernel passes per-expert base offsets ``gx/gro/grsc/gco/gcsc`` and
    ``gmul=G`` to widen the SRDs over the whole 3D tensor (R,C stay per-expert).
    ``padded`` (non-256 K / non-128 N): X is the real [R,C] but outputs use K_pad=CP /
    N_pad=RP cols (caller zero-inits so pad stays 0, matching HIP); loads past real C
    mask to 0 and writes past K_pad / real-C rows go to _OOB so the store drops them.
    ``rmsnorm_scale``: X is residual-sum ``xpr``; multiply by ``RSTD[row]*GAMMA[col]``
    (f32) after unpack and before RHT so this dual matches ``quantize(y)``.
    ``tile_tr``/``tile_tc`` override the module tile (rmsnorm dual uses a shorter
    row tile so gamma/rstd LDS does not drop CU occupancy)."""
    tile_r = _TR if tile_tr is None else int(tile_tr)
    tile_c = _TC if tile_tc is None else int(tile_tc)
    tile_cw = tile_c // 2
    tile_rmb = tile_r // 32
    tile_nload = (tile_r * tile_cw + BLK * 4 - 1) // (BLK * 4)
    tile_rrowtask = (tile_r * (tile_c // 32)) // BLK
    tile_rmbc = tile_c // 32
    if ncblk is None:
        ncblk = C // tile_c
    cpad = CP if padded else C  # row-out column extent (K_pad)
    rpad = RP if padded else R  # col-out column extent (N_pad)
    # Block order = which output's partial stores L2 can combine. col_locality (C>R):
    # row-tile-fastest so blocks writing the same col-out rows run back-to-back and L2
    # merges the scattered transpose stores; else col-tile-fastest keeps row-out coalesced.
    if col_locality:
        nrblk = R // tile_r
        cblk = bid // nrblk
        rblk = bid % nrblk
    else:
        rblk = bid // ncblk
        cblk = bid % ncblk
    r0 = rblk * tile_r
    c0w = cblk * tile_cw  # i32-word base along C

    # Re-base each SRD in int64 with per-tile/per-expert num_records: a whole-tensor SRD's
    # num_records (full bytes) overflows the 32-bit field past 4GB (high rows/experts OOB) and
    # the per-row voffset overflows int32. 2D folds this tile's row (r0)/col (cblk*tile_c) base;
    # batched-3D folds the per-expert base (small experts keep r0/c0 in the offsets). _row0/
    # _col0 drop the folded base from the additive offsets below.
    _fold = not batched
    _row0 = fx.Int32(0) if _fold else r0
    _col0 = fx.Int32(0) if _fold else cblk * tile_c

    def _srd(t, elem_off, elem_bytes, nrec_bytes):
        base = arith.index_cast(T.i64, buffer_ops.extract_base_index(t))
        boff = arith.index_cast(T.i64, arith.index_cast(T.index, elem_off) * arith.index(elem_bytes))
        raw = arith._to_raw(base + boff)
        r = rocdl.readfirstlane(res=raw.type, src=raw)  # pin the SRD base to an SGPR
        base_v = r.result if hasattr(r, "result") else r
        if isinstance(nrec_bytes, int):
            nrec_bytes = arith.index(nrec_bytes)
        nr = arith.minui(arith.index_cast(T.index, nrec_bytes), arith.index(0x7FFFFFFF))
        return buffer_ops.create_buffer_resource_from_addr(base_v, num_records_bytes=nr)

    if batched:
        rsrc = _srd(X, gx, 4, R * (C >> 1) * 4)
        orsrc = _srd(ROW_OUT, gro, 4, R * (cpad >> 3) * 4)
        rscrsrc = _srd(ROW_SC, grsc, 1, R * (cpad >> 5))
        corsrc = _srd(COL_OUT, gco, 4, C * (rpad >> 3) * 4)
        cscrsrc = _srd(COL_SC, gcsc, 1, C * (rpad >> 5))
        gx = gro = grsc = gco = gcsc = 0  # expert bases folded into the SRDs above
    else:
        r0i = arith.index_cast(T.index, r0)
        c0i = arith.index_cast(T.index, cblk * tile_c)
        rsrc = _srd(X, r0i * arith.index_cast(T.index, C >> 1), 4, tile_r * (C >> 1) * 4)
        orsrc = _srd(ROW_OUT, r0i * arith.index_cast(T.index, cpad >> 3), 4, tile_r * (cpad >> 3) * 4)
        rscrsrc = _srd(ROW_SC, r0i * arith.index_cast(T.index, cpad >> 5), 1, tile_r * (cpad >> 5))
        corsrc = _srd(COL_OUT, c0i * arith.index_cast(T.index, rpad >> 3), 4, tile_c * (rpad >> 3) * 4)
        cscrsrc = _srd(COL_SC, c0i * arith.index_cast(T.index, rpad >> 5), 1, tile_c * (rpad >> 5))

    rstd_rsrc = None
    gamma_rsrc = None
    _rmsnorm_scales_row = None
    _rmsnorm_scales_col = None
    if rmsnorm_scale:
        # X is xpr; y = xpr * rstd * gamma is applied in-register before amax/RHT.
        r0i = arith.index_cast(T.index, r0)
        c0i = arith.index_cast(T.index, cblk * tile_c)
        rstd_rsrc = _srd(RSTD, r0i, 4, arith.index(tile_r * 4))
        gamma_rsrc = _srd(GAMMA, c0i, 4, arith.index(tile_c * 4))

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
    for chunk in range_constexpr(tile_nload):
        tw = chunk * (BLK * 4) + tid * 4
        tr = tw // tile_cw
        wc = tw % tile_cw
        goff = (_row0 + tr) * (C >> 1) + c0w + wc + gx
        if padded:
            # mask cols past real C -> OOB load returns 0 (rows always valid: R%64==0,
            # tile is 64 rows, rblk covers exactly R/64 tiles).
            goff = arith.select((c0w + wc) < (C >> 1), goff, fx.Int32(_OOB))
        vec = buffer_ops.buffer_load(rsrc, goff, vec_width=4, dtype=T.i32)
        _lds_store_vec4(lds.buf.ptr, tw, vec)
    if rmsnorm_scale:
        # One cooperative fill of the 256-col gamma tile and 64-row rstd tile.
        # Replaces 32 scalar global loads per microblock (same values reused
        # across rows / cols of this tile).
        gv = fx.Float32(buffer_ops.buffer_load(gamma_rsrc, _col0 + tid, vec_width=1, dtype=T.f32))
        _lds_store1(lds.gbits.ptr, tid, Vec.from_elements([gv], fx.Float32).bitcast(fx.Int32)[0])
        roff = arith.select(tid < tile_r, _row0 + tid, fx.Int32(_OOB))
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
        return _sr_hash(sr_seed ^ (_gbid * (BLK * tile_rrowtask) + (k * BLK + tid)))

    def _col_seed(mmb):
        if not col_sr:
            return None
        return _sr_hash((sr_seed ^ _SR_COL_SALT) ^ (_gbid * (BLK * tile_rmb) + (mmb * BLK + tid)))

    # ---- ROW phase: 32-elem microblocks along C, contiguous LDS (vec4 reads) ----
    if row_2d:
        # 2D-block amax: the scale spans a whole 32x32 tile = the 32 rows that
        # share one 32-col micro-block. Pass 1: each thread computes its own
        # micro-block amax (RHT'd) and writes it to LDS scratch, keeping the
        # RHT'd vals in registers. Barrier. Pass 2: each thread max-reduces the
        # 32 amax of its tile, then quantizes its held vals with the tile scale.
        vf_hold = []
        meta = []
        for k in range_constexpr(tile_rrowtask):
            task = k * BLK + tid
            r_row = task // tile_rmbc
            cmb = task % tile_rmbc
            base_w = r_row * tile_cw + cmb * 16
            rbits = []
            for q in range_constexpr(4):
                v4 = _lds_load_vec4(lds.buf.ptr, base_w + q * 4)
                for j in range_constexpr(4):
                    word = v4[j]
                    rbits.append(word << 16)
                    rbits.append(word & 0xFFFF0000)
            vf = _microblock_vf(rbits, row_rht)
            _lds_store1(lds.scr.ptr, r_row * tile_rmbc + cmb, _microblock_amax(vf))
            vf_hold.append(vf)
            meta.append((r_row, cmb))
        fx.barrier()
        for k in range_constexpr(tile_rrowtask):
            r_row, cmb = meta[k]
            vf = vf_hold[k]
            row_base = (r_row // 32) * 32  # tile's first row within the LDS tile
            tile_amax = fx.Int32(0)
            for i in range_constexpr(32):
                tile_amax = _imax(tile_amax, _lds_load1(lds.scr.ptr, (row_base + i) * tile_rmbc + cmb))
            native_bits, rbiased = _compute_scale_native(tile_amax)
            rwords = _cvt_microblock_to_fp4(vf, arith.bitcast(T.f32, native_bits), _row_seed(k))
            grow = _row0 + r_row
            gcmb = cblk * tile_rmbc + cmb
            ob = grow * (cpad >> 3) + gcmb * 4 + gro
            sc = grow * (cpad >> 5) + gcmb + grsc
            if padded:
                wok = gcmb < (cpad >> 5)  # rows always valid (R%64==0)
                ob = arith.select(wok, ob, fx.Int32(_OOB))
                sc = arith.select(wok, sc, fx.Int32(_OOB))
            _store_words_vec4(orsrc, ob, rwords)
            buffer_ops.buffer_store(arith.trunci(T.i8, rbiased & 0xFF), rscrsrc, sc)
    else:
        for k in range_constexpr(tile_rrowtask):
            task = k * BLK + tid
            r_row = task // (tile_c // 32)
            cmb = task % (tile_c // 32)
            base_w = r_row * tile_cw + cmb * 16
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
                _row_seed(k),
                _rmsnorm_scales_row(r_row, cmb) if rmsnorm_scale else None,
            )
            grow = _row0 + r_row
            gcmb = cblk * (tile_c // 32) + cmb
            ob = grow * (cpad >> 3) + gcmb * 4 + gro
            sc = grow * (cpad >> 5) + gcmb + grsc
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
        for mmb in range_constexpr(tile_rmb):
            row0 = mmb * 32
            cbits = []
            for row in range_constexpr(32):
                word = _lds_load1(lds.buf.ptr, (row0 + row) * tile_cw + cw)
                fb = arith.select(half != 0, word & fx.Int32(-65536), word << 16)
                cbits.append(fb)
            vf = _microblock_vf(cbits, col_rht)
            _lds_store1(lds.scr.ptr, mmb * tile_c + c_col, _microblock_amax(vf))
            cvf_hold.append(vf)
        fx.barrier()
        col_base = (c_col // 32) * 32  # tile's first column within the LDS tile
        for mmb in range_constexpr(tile_rmb):
            vf = cvf_hold[mmb]
            tile_amax = fx.Int32(0)
            for i in range_constexpr(32):
                tile_amax = _imax(tile_amax, _lds_load1(lds.scr.ptr, mmb * tile_c + col_base + i))
            native_bits, cbiased = _compute_scale_native(tile_amax)
            cwords = _cvt_microblock_to_fp4(vf, arith.bitcast(T.f32, native_bits), _col_seed(mmb))
            gcol = _col0 + c_col
            gmmb = rblk * tile_rmb + mmb
            cob = gcol * (rpad >> 3) + gmmb * 4 + gco
            csoff = gcol * (rpad >> 5) + gmmb + gcsc
            if padded:
                cok = gcol < C  # col-out has real-C rows; drop pad-K rows
                cob = arith.select(cok, cob, fx.Int32(_OOB))
                csoff = arith.select(cok, csoff, fx.Int32(_OOB))
            _store_words_vec4(corsrc, cob, cwords)
            buffer_ops.buffer_store(arith.trunci(T.i8, cbiased & 0xFF), cscrsrc, csoff)
    else:
        for mmb in range_constexpr(tile_rmb):
            row0 = mmb * 32
            cbits = []
            for row in range_constexpr(32):
                word = _lds_load1(lds.buf.ptr, (row0 + row) * tile_cw + cw)
                fb = arith.select(half != 0, word & fx.Int32(-65536), word << 16)
                cbits.append(fb)
            cwords, cbiased = _finish_microblock(
                cbits,
                col_rht,
                _col_seed(mmb),
                _rmsnorm_scales_col(mmb, c_col) if rmsnorm_scale else None,
            )
            gcol = _col0 + c_col
            gmmb = rblk * tile_rmb + mmb
            cob = gcol * (rpad >> 3) + gmmb * 4 + gco
            csoff = gcol * (rpad >> 5) + gmmb + gcsc
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
        grid_x: fx.Int32,
        stream: fx.Stream,
    ):
        kern(X, ROW_OUT, ROW_SC, COL_OUT, COL_SC, RSTD, GAMMA, R, C, SR_SEED).launch(
            grid=(grid_x, 1, 1), block=(BLK, 1, 1), stream=stream
        )

    return _rmsnorm_dual_launch


def flydsl_rmsnorm_dual_quant(xpr_bf16, rstd_f32, gamma_f32, fp4_dtype, col_rht=True):
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
        fn = flyc.compile(raw, x, ro, rs, co, cs, rstd, gamma, R, C, 0, grid_x, stream)
        ent = (fn, grid_x)
        _RMSNORM_DUAL_COMPILED[key] = ent
    return ent


# ---- Batched-3D dual quant: [G,N,K] weight, all experts in ONE launch (G x the
# blocks -> fills the GPU even for small per-expert N, where the 2D dense kernel is
# occupancy-starved and drops to ~2.5 TB/s). Reuses _emit_dual_body per-tile with
# per-expert base offsets; SRDs cover the whole 3D (gmul=G). ----
def _build_dual3_kernel(
    row_rht, col_rht, row_2d=False, col_2d=False, padded=False, col_locality=False, row_sr=False, col_sr=False
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
        g = fx.block_idx.x // tpg
        lbid = fx.block_idx.x - g * tpg
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
            sr_gbid=fx.block_idx.x,
        )

    return _dual3_kernel


def _build_dual3_launch(
    row_rht, col_rht, row_2d=False, col_2d=False, padded=False, col_locality=False, row_sr=False, col_sr=False
):
    kern = _build_dual3_kernel(row_rht, col_rht, row_2d, col_2d, padded, col_locality, row_sr, col_sr)

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
    padded = (K % _TC != 0) or (N % 128 != 0)
    col_locality = int(K) > int(N)  # K>N: combine transpose stores (col-out)
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
        )
        _DUAL3_LAUNCH[lk] = raw
    key = (int(N), int(K), int(G), *lk)
    ent = _DUAL3_COMPILED.get(key)
    if ent is None:
        import torch

        x = torch.zeros((G, N, K // 2), dtype=torch.int32, device="cuda")
        ro = torch.zeros((G, N, Kp // 8), dtype=torch.int32, device="cuda")
        rs = torch.zeros((G, N, Kp // 32), dtype=torch.uint8, device="cuda")
        co = torch.zeros((G, K, Np // 8), dtype=torch.int32, device="cuda")
        cs = torch.zeros((G, K, Np // 32), dtype=torch.uint8, device="cuda")
        ncblk = ((K + _TC - 1) // _TC) if padded else (K // _TC)
        grid_x = (N // _TR) * ncblk * G
        fn = flyc.compile(raw, x, ro, rs, co, cs, N, K, G, Kp, Np, 0, grid_x, torch.cuda.current_stream())
        ent = (fn, grid_x, Kp, Np, padded)
        _DUAL3_COMPILED[key] = ent
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
    fn, grid_x, Kp, Np, padded = get_dual3_cast(N, K, G, row_rht, col_rht, row_2d, col_2d, row_sr, col_sr)
    # Outputs sized on K_pad/N_pad and must match the HIP dual (pad regions all-0).
    # The masked loads already make the kernel write the whole K_pad row-output tail, so
    # only the col-output N_pad suffix is left without producer threads: allocate without
    # a full-tensor memset and clear that small suffix instead (from #468).
    ro = torch.empty((G, N, Kp // 8), dtype=torch.int32, device=dev)
    rs = torch.empty((G, N, Kp // 32), dtype=torch.uint8, device=dev)
    co = torch.empty((G, K, Np // 8), dtype=torch.int32, device=dev)
    cs = torch.empty((G, K, Np // 32), dtype=torch.uint8, device=dev)
    if padded and Np > N:
        co[:, :, N // 8 :].zero_()
        cs[:, :, N // 32 :].zero_()
    sr_seed = _next_sr_seed() if (row_sr or col_sr) else 0
    fn(x_i32, ro, rs, co, cs, N, K, G, Kp, Np, sr_seed, grid_x, torch.cuda.current_stream())
    return (
        ro.view(torch.uint8).view(fp4_dtype),
        rs.view(torch.float8_e8m0fnu),
        co.view(torch.uint8).view(fp4_dtype),
        cs.view(torch.float8_e8m0fnu),
    )
