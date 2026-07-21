###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
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

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith, buffer_ops, range_constexpr, rocdl
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

_OOB = 0x7FFFFFFF  # word offset past any SRD -> buffer_load returns 0 / buffer_store dropped
BLK = 256
MB = 32  # MXFP4 micro-block size (elements per e8m0 scale)


def _abs_i32(fbits):
    return fbits & 0x7FFFFFFF


def _imax(a, b):
    return arith.select(a < b, b, a)


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


def _cvt_microblock_to_fp4(vf, scale_native_f32):
    """32 f32 Values -> 4 i32 words (8 fp4 each). Pair-form cvt, dst_sel chaining."""
    words = []
    for wi in range_constexpr(4):
        acc = fx.Int32(0)
        for pair in range_constexpr(4):
            i = wi * 8 + pair * 2
            acc = rocdl.cvt_scalef32_pk_fp4_f32(T.i32, acc, vf[i], vf[i + 1], scale_native_f32, pair)
        words.append(acc)
    return words


def _build_row_kernel(use_rht):
    """Factory: fresh @flyc.kernel with RHT baked in as a Python constant."""

    @flyc.kernel(known_block_size=[BLK, 1, 1])
    def _row_cast_kernel(
        X: fx.Tensor,  # int32 view of bf16 [R, C], shape [R, C/2]
        OUT: fx.Tensor,  # int32 view of fp4 bytes [R, C/2], shape [R, C/8]
        SCALE: fx.Tensor,  # uint8 [R, C/32]
        R: fx.Int32,
        C: fx.Int32,
    ):
        g = fx.block_idx.x * fx.Int32(BLK) + fx.thread_idx.x
        mb_per_row = C >> 5  # C / 32
        row = g // mb_per_row
        mb = g % mb_per_row
        # ---- load 16 i32 (32 bf16) from X (i32 view: offsets in i32 words) ----
        x_words = R * (C >> 1)
        rsrc = buffer_ops.create_buffer_resource(X, max_size=False, num_records_bytes=x_words * 4)
        i32base = row * (C >> 1) + mb * 16
        vbits = []  # 32 f32-bit i32 values
        for c in range_constexpr(4):
            wv = buffer_ops.buffer_load(rsrc, i32base + c * 4, vec_width=4, dtype=T.i32)
            for j in range_constexpr(4):
                w = wv[j]
                vbits.append(w << 16)  # even bf16
                vbits.append(w & 0xFFFF0000)  # odd bf16
        vf = [Vec.from_elements([b], fx.Int32).bitcast(fx.Float32)[0] for b in vbits]
        if use_rht:
            vf = _rht16(vf[0:16]) + _rht16(vf[16:32])
        # ---- amax over 32 (int-max on abs bits) ----
        amax = fx.Int32(0)
        for i in range_constexpr(32):
            b = Vec.from_elements([vf[i]], fx.Float32).bitcast(fx.Int32)[0]
            amax = _imax(amax, _abs_i32(b))
        native_bits, biased = _compute_scale_native(amax)
        scale_native = arith.bitcast(T.f32, native_bits)
        words = _cvt_microblock_to_fp4(vf, scale_native)
        # ---- store fp4: OUT i32 view [R, C/8]; offset = row*(C/8) + mb*4 + c ----
        out_words = R * (C >> 3)
        orsrc = buffer_ops.create_buffer_resource(OUT, max_size=False, num_records_bytes=out_words * 4)
        outbase = row * (C >> 3) + mb * 4
        for c in range_constexpr(4):
            buffer_ops.buffer_store(words[c], orsrc, outbase + c)
        # ---- store scale byte: SCALE [R, C/32] u8; offset = row*(C/32) + mb ----
        sc_nbytes = R * mb_per_row
        srsrc = buffer_ops.create_buffer_resource(SCALE, max_size=False, num_records_bytes=sc_nbytes)
        buffer_ops.buffer_store(arith.trunci(T.i8, biased & 0xFF), srsrc, row * mb_per_row + mb)

    return _row_cast_kernel


def _build_row_launch(use_rht):
    kern = _build_row_kernel(use_rht)

    @flyc.jit
    def _row_cast_launch(
        X: fx.Tensor,
        OUT: fx.Tensor,
        SCALE: fx.Tensor,
        R: fx.Int32,
        C: fx.Int32,
        grid_x: fx.Int32,
        stream: fx.Stream,
    ):
        kern(X, OUT, SCALE, R, C).launch(grid=(grid_x, 1, 1), block=(BLK, 1, 1), stream=stream)

    return _row_cast_launch


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


def _finish_microblock(vbits, use_rht):
    """32 f32-bit i32 values -> (4 fp4 i32 words, scale_e8m0 i8-ready i32)."""
    vf = _microblock_vf(vbits, use_rht)
    amax = _microblock_amax(vf)
    native_bits, biased = _compute_scale_native(amax)
    words = _cvt_microblock_to_fp4(vf, arith.bitcast(T.f32, native_bits))
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

    # Output SRDs are re-based in int64 with per-tile / per-expert num_records. A whole-tensor
    # SRD would need num_records = full bytes, which overflows the 32-bit num_records field
    # once the operand exceeds 4GB (high rows/experts OOB -> garbage) and drives a per-row byte
    # voffset past int32. 2D folds THIS tile's row (r0) / col (cblk*_TC) base -> num_records
    # per-tile; batched-3D folds the per-EXPERT base (gx/... , experts small so r0/c0 stay in
    # the offsets). ``_row0``/``_col0`` drop the folded base from the additive offsets below.
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
            vf = _microblock_vf(rbits, row_rht)
            _lds_store1(lds.scr.ptr, r_row * _RMBC + cmb, _microblock_amax(vf))
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
            native_bits, rbiased = _compute_scale_native(tile_amax)
            rwords = _cvt_microblock_to_fp4(vf, arith.bitcast(T.f32, native_bits))
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
            rwords, rbiased = _finish_microblock(rbits, row_rht)
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
            vf = _microblock_vf(cbits, col_rht)
            _lds_store1(lds.scr.ptr, mmb * _TC + c_col, _microblock_amax(vf))
            cvf_hold.append(vf)
        fx.barrier()
        col_base = (c_col // 32) * 32  # tile's first column within the LDS tile
        for mmb in range_constexpr(_RMB):
            vf = cvf_hold[mmb]
            tile_amax = fx.Int32(0)
            for i in range_constexpr(32):
                tile_amax = _imax(tile_amax, _lds_load1(lds.scr.ptr, mmb * _TC + col_base + i))
            native_bits, cbiased = _compute_scale_native(tile_amax)
            cwords = _cvt_microblock_to_fp4(vf, arith.bitcast(T.f32, native_bits))
            gcol = _col0 + c_col
            gmmb = rblk * _RMB + mmb
            cob = gcol * (rpad >> 3) + gmmb * 4 + gco
            csoff = gcol * (rpad >> 5) + gmmb + gcsc
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
            cwords, cbiased = _finish_microblock(cbits, col_rht)
            gcol = _col0 + c_col
            gmmb = rblk * _RMB + mmb
            cob = gcol * (rpad >> 3) + gmmb * 4 + gco
            csoff = gcol * (rpad >> 5) + gmmb + gcsc
            if padded:
                cok = gcol < C  # col-out has real-C rows; drop pad-K rows
                cob = arith.select(cok, cob, fx.Int32(_OOB))
                csoff = arith.select(cok, csoff, fx.Int32(_OOB))
            _store_words_vec4(corsrc, cob, cwords)
            buffer_ops.buffer_store(arith.trunci(T.i8, cbiased & 0xFF), cscrsrc, csoff)


def _build_dual_kernel(row_rht, col_rht, row_2d=False, col_2d=False, col_locality=False):
    """Single-recipe fused LDS dual (one coalesced 32x256 tile load feeds both the
    rowwise and colwise-transpose casts). Thin wrapper over ``_emit_dual_body``.
    ``col_locality`` (set for C>R shapes) flips the block order to combine the
    transpose stores; see ``_emit_dual_body``."""
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
        )

    return _dual_kernel


def _build_dual_launch(row_rht, col_rht, row_2d=False, col_2d=False, col_locality=False):
    kern = _build_dual_kernel(row_rht, col_rht, row_2d, col_2d, col_locality)

    @flyc.jit
    def _dual_launch(
        X: fx.Tensor,
        ROW_OUT: fx.Tensor,
        ROW_SC: fx.Tensor,
        COL_OUT: fx.Tensor,
        COL_SC: fx.Tensor,
        R: fx.Int32,
        C: fx.Int32,
        grid_x: fx.Int32,
        stream: fx.Stream,
    ):
        kern(X, ROW_OUT, ROW_SC, COL_OUT, COL_SC, R, C).launch(
            grid=(grid_x, 1, 1), block=(BLK, 1, 1), stream=stream
        )

    return _dual_launch


_DUAL_LAUNCH = {}
_DUAL_COMPILED = {}


def dual_eligible(R, C, row_recipe, col_recipe):
    """True if the FlyDSL fused dual can wholesale-replace the C++ dual for these
    recipes/dims (SR off, no preshuffle, dims aligned -> no padding). Both the
    per-microblock (2d=F) and the 2d-block (2d=T weight) amax geometries are
    supported and bit-exact vs C++; SR / shuffled recipes still fall back."""
    return (
        not row_recipe.use_sr
        and not col_recipe.use_sr
        and not row_recipe.shuffle_scale
        and not row_recipe.shuffle_out
        and not col_recipe.shuffle_scale
        and not col_recipe.shuffle_out
        and (R % 128 == 0)
        and (C % 256 == 0)
    )


def flydsl_dual_quant(x_bf16, fp4_dtype, row_rht, col_rht, row_2d=False, col_2d=False):
    """Fused rowwise + colwise-transpose mxfp4 cast (one bf16 read). Returns
    (row_data, row_scale, col_data, col_scale) in C++-compatible dtypes/shapes."""
    import torch

    R, C = x_bf16.shape
    dev = x_bf16.device
    x_i32 = x_bf16.view(torch.int32)  # [R, C/2]
    ro = torch.empty((R, C // 8), dtype=torch.int32, device=dev)
    rs = torch.empty((R, C // 32), dtype=torch.uint8, device=dev)
    co = torch.empty((C, R // 8), dtype=torch.int32, device=dev)
    cs = torch.empty((C, R // 32), dtype=torch.uint8, device=dev)
    fn, grid_x = get_dual_cast(R, C, row_rht, col_rht, row_2d, col_2d)
    fn(x_i32, ro, rs, co, cs, R, C, grid_x, torch.cuda.current_stream())
    row_data = ro.view(torch.uint8).view(fp4_dtype)  # [R, C/2] fp4
    col_data = co.view(torch.uint8).view(fp4_dtype)  # [C, R/2] fp4
    row_scale = rs.view(torch.float8_e8m0fnu)
    col_scale = cs.view(torch.float8_e8m0fnu)
    return row_data, row_scale, col_data, col_scale


def get_dual_cast(R, C, row_rht, col_rht, row_2d=False, col_2d=False):
    """Return (compiled_fn, grid_x) for the fused dual at
    (R, C, row_rht, col_rht, row_2d, col_2d).
    Requires R % 128 == 0 and C % 256 == 0 (no scale/output padding)."""
    col_locality = int(C) > int(R)  # C>R (down-proj): combine transpose stores
    lk = (bool(row_rht), bool(col_rht), bool(row_2d), bool(col_2d), col_locality)
    raw = _DUAL_LAUNCH.get(lk)
    if raw is None:
        raw = _build_dual_launch(bool(row_rht), bool(col_rht), bool(row_2d), bool(col_2d), col_locality)
        _DUAL_LAUNCH[lk] = raw
    key = (int(R), int(C), bool(row_rht), bool(col_rht), bool(row_2d), bool(col_2d))
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
        fn = flyc.compile(raw, x, ro, rs, co, cs, R, C, grid_x, stream)
        ent = (fn, grid_x)
        _DUAL_COMPILED[key] = ent
    return ent


# ---- Batched-3D dual quant: [G,N,K] weight, all experts in ONE launch (G x the
# blocks -> fills the GPU even for small per-expert N, where the 2D dense kernel is
# occupancy-starved and drops to ~2.5 TB/s). Reuses _emit_dual_body per-tile with
# per-expert base offsets; SRDs cover the whole 3D (gmul=G). ----
def _build_dual3_kernel(row_rht, col_rht, row_2d=False, col_2d=False, padded=False, col_locality=False):
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
        )

    return _dual3_kernel


def _build_dual3_launch(row_rht, col_rht, row_2d=False, col_2d=False, padded=False, col_locality=False):
    kern = _build_dual3_kernel(row_rht, col_rht, row_2d, col_2d, padded, col_locality)

    @flyc.jit
    def _dual3_launch(X, ROW_OUT, ROW_SC, COL_OUT, COL_SC, R, C, G, CP, RP, grid_x, stream):
        kern(X, ROW_OUT, ROW_SC, COL_OUT, COL_SC, R, C, G, CP, RP).launch(
            grid=(grid_x, 1, 1), block=(BLK, 1, 1), stream=stream
        )

    return _dual3_launch


_DUAL3_LAUNCH = {}
_DUAL3_COMPILED = {}


def dual3_eligible(N, K, row_recipe, col_recipe):
    """True if the batched-3D FlyDSL dual can replace the C++ dual for a [G,N,K]
    weight (SR off, no preshuffle). Handles non-256 K / non-128 N via K_pad/N_pad
    (bit-exact vs the HIP dual whose pad is all-zero). Needs N%64==0 (row/col tiling)
    and K%64==0 (32-microblock + vec4-aligned tail load mask)."""
    return (
        not row_recipe.use_sr
        and not col_recipe.use_sr
        and not row_recipe.shuffle_scale
        and not row_recipe.shuffle_out
        and not col_recipe.shuffle_scale
        and not col_recipe.shuffle_out
        and (N % 64 == 0)
        and (K % 64 == 0)
    )


def get_dual3_cast(N, K, G, row_rht, col_rht, row_2d=False, col_2d=False):
    """(compiled_fn, grid_x, K_pad, N_pad, padded) for the batched-3D dual at
    (N,K,G,recipes). K_pad=ceil(K/128)*128 (row-out), N_pad=ceil(N/128)*128 (col-out);
    `padded` when K not a 256-tile multiple or N not 128-multiple."""
    Kp = ((K + 127) // 128) * 128
    Np = ((N + 127) // 128) * 128
    padded = (K % _TC != 0) or (N % 128 != 0)
    col_locality = int(K) > int(N)  # K>N: combine transpose stores (col-out)
    lk = (bool(row_rht), bool(col_rht), bool(row_2d), bool(col_2d), padded, col_locality)
    raw = _DUAL3_LAUNCH.get(lk)
    if raw is None:
        raw = _build_dual3_launch(
            bool(row_rht), bool(col_rht), bool(row_2d), bool(col_2d), padded, col_locality
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
        fn = flyc.compile(raw, x, ro, rs, co, cs, N, K, G, Kp, Np, grid_x, torch.cuda.current_stream())
        ent = (fn, grid_x, Kp, Np, padded)
        _DUAL3_COMPILED[key] = ent
    return ent


def flydsl_dual_quant_batched(x3d, fp4_dtype, row_rht, col_rht, row_2d=False, col_2d=False):
    """Batched-3D fused rowwise + colwise-transpose mxfp4 dual cast for a [G,N,K]
    weight in ONE launch. Returns C++-compatible per-expert
    (row_data [G,N,K/2], row_scale [G,N,K/32], col_data [G,K,N/2], col_scale [G,K,N/32])."""
    import torch

    G, N, K = x3d.shape
    dev = x3d.device
    x_i32 = x3d.contiguous().view(torch.int32)  # [G, N, K/2]
    fn, grid_x, Kp, Np, padded = get_dual3_cast(N, K, G, row_rht, col_rht, row_2d, col_2d)
    # Outputs sized on K_pad/N_pad; zeros so pad regions match the HIP dual (all-0).
    alloc = torch.zeros if padded else torch.empty
    ro = alloc((G, N, Kp // 8), dtype=torch.int32, device=dev)
    rs = alloc((G, N, Kp // 32), dtype=torch.uint8, device=dev)
    co = alloc((G, K, Np // 8), dtype=torch.int32, device=dev)
    cs = alloc((G, K, Np // 32), dtype=torch.uint8, device=dev)
    fn(x_i32, ro, rs, co, cs, N, K, G, Kp, Np, grid_x, torch.cuda.current_stream())
    return (
        ro.view(torch.uint8).view(fp4_dtype),
        rs.view(torch.float8_e8m0fnu),
        co.view(torch.uint8).view(fp4_dtype),
        cs.view(torch.float8_e8m0fnu),
    )


# raw @flyc.jit launchers (rht off / on), built once
_ROW_LAUNCH = {False: _build_row_launch(False), True: _build_row_launch(True)}
_ROW_COMPILED = {}


def get_row_cast(R, C, use_rht):
    """Return (compiled_fn, grid_x) for the row cast at (R, C, use_rht)."""
    key = (int(R), int(C), bool(use_rht))
    ent = _ROW_COMPILED.get(key)
    if ent is None:
        import torch

        x = torch.zeros((R, C // 2), dtype=torch.int32, device="cuda")
        out = torch.zeros((R, C // 8), dtype=torch.int32, device="cuda")
        sc = torch.zeros((R, C // 32), dtype=torch.uint8, device="cuda")
        grid_x = (R * (C // 32) + BLK - 1) // BLK
        stream = torch.cuda.current_stream()
        raw = _ROW_LAUNCH[bool(use_rht)]
        fn = flyc.compile(raw, x, out, sc, R, C, grid_x, stream)
        ent = (fn, grid_x)
        _ROW_COMPILED[key] = ent
    return ent
