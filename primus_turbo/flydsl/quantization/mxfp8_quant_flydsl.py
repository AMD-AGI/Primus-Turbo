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

# Dual-cast mxfp8 quant emitting raw row-major E8M0 scales, bit-identical to the HIP
# quantize_mxfp8_dual. One coalesced tile load -> LDS feeds two concurrent halves: ROW casts fp8 +
# row E8M0, COL casts the transpose + stages it in LDS for a coalesced write-back + col E8M0.
# GEMM-side scale preshuffle is fused in the GEMM launch, not here.
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir.dialects import llvm as _llvm
from flydsl.expr import arith, range_constexpr, rocdl
from flydsl.expr import buffer_ops as bo
from flydsl.expr import math as fm
from flydsl.expr.arith import ArithValue
from flydsl.expr.typing import T as _T
from flydsl.expr.typing import Vector as Vec

from primus_turbo.flydsl.utils.gemm_helper import (
    group_m_tile_decode,
    make_row_band_resource,
    make_row_band_resource_div,
    xcd_band_remap_pid,
)

# Scale stores are byte-granular and rely on L2 merging the write masks into whole lines.
_CM = 1  # glc
# Data-store policy: `nt` speeds the cast up but the consuming GEMM hands it all back.
_CMD = 1  # glc
_OOB = 0x7FFFFFFF  # past-end buffer offset -> HW drops the access


def _cvt_pk_bf8_f32(res, src_a, src_b, old, word_sel, **kw):
    # flydsl.expr.rocdl wraps cvt_pk_fp8_f32 (snake-case + _to_ir on operands) but
    # re-exports cvt_pk_bf8_f32 as the raw generated op (no operand conversion). Mirror
    # the fp8 wrapper so e5m2 (bf8) can be called with flydsl exprs like e4m3.
    from flydsl._mlir.dialects.rocdl import cvt_pk_bf8_f32 as _op
    from flydsl.expr.rocdl import _to_ir

    return _op(res=res, src_a=_to_ir(src_a), src_b=_to_ir(src_b), old=_to_ir(old), word_sel=word_sel, **kw)


def fp8_params(out_fp8):
    """Per-output-format quant constants (val_to_add round-even bump, ep_sub bias, fp8 sat bound,
    packed f32->fp8 cvt), mirroring the C++ compute_tile_scale / cvt_f32x4_to_fp8x4."""
    if out_fp8 == "e5m2":
        return (1 << 20, 142, 57344.0, _cvt_pk_bf8_f32)
    return (1 << 19, 135, 448.0, rocdl.cvt_pk_fp8_f32)


def _sat(v, bound):
    # clamp a scalar f32 to +-fp8_max so a boundary round can't emit a NaN code;
    # matches the HIP saturating fp8 cast.
    F32 = fx.Float32
    return fm.clampf(v, F32(-bound), F32(bound))


def _ep(amax, va, sub):
    I32 = fx.Int32
    # flydsl >=0.2.2: bitcast takes a Numeric subclass (not .ir_type) and returns it.
    ai = amax.bitcast(fx.Int32) + I32(va)
    ep = ((ai >> I32(23)) & I32(0x1FF)) - I32(sub)
    ep = (ep < I32(-127)).select(I32(-127), ep)
    ep = (ep > I32(128)).select(I32(128), ep)
    return I32(ep)


def _raw_scale_dword(free, blk, scale_n):
    """Raw E8M0 (plain row-major [free, contract//32]) byte address split into
    (dword, jbyte) for the shared scale store. off = free*scale_n + blk; a dword can
    straddle two free rows written by different workgroups, hence the byte store."""
    I32 = fx.Int32
    off = free * I32(scale_n) + blk
    return off >> I32(2), off & I32(3)


def _store_scale(buf, buf_bytes, dword, jbyte, e8_i32, pack, ok=None, cm=_CM, base_byte=0):
    """Write one E8M0 scale byte. pack==1: broadcast to all 4 bytes of i32[dword] (dword owned by
    one workgroup -> plain dword store). pack>1: native byte store (BUFFER_STORE_BYTE, no dword
    RMW) so ``pack`` bytes packed from DIFFERENT workgroups don't race. ``ok`` masks the K-tail
    (pack==1 redirects the offset OOB; pack>1 uses the store mask). ``base_byte`` (batched):
    per-batch dword-aligned byte base so B scale buffers share one resource."""
    I32 = fx.Int32
    _has_base = not (isinstance(base_byte, int) and base_byte == 0)
    if pack == 1:
        rsrc = bo.create_buffer_resource(buf, max_size=False, num_records_bytes=I32(buf_bytes))
        bcast = e8_i32 | (e8_i32 << I32(8)) | (e8_i32 << I32(16)) | (e8_i32 << I32(24))
        # pack==1 offset is in dword units; buf_bytes (== 4*elems) is safely past the end.
        off = dword if ok is None else ok.select(dword, I32(buf_bytes))
        if _has_base:
            off = off + (base_byte >> I32(2))  # base_byte is dword-aligned
        bo.buffer_store(bcast, rsrc, off, cache_modifier=cm)
    else:
        # pack>1: byte-granular store (L2 byte-write-mask merges, no dword RMW) so the dword's
        # bytes from different M-block workgroups don't need a cross-CU atomicrmw. ``ok`` masks OOB.
        # The scale plane costs LINES, not stores -- line sharing is what ``xcd_band`` cuts.
        rsrc = bo.create_buffer_resource(buf, max_size=False, num_records_bytes=I32(buf_bytes))
        byte_off = (dword << I32(2)) | jbyte
        if _has_base:
            byte_off = byte_off + base_byte
        val_i8 = ArithValue(e8_i32 & I32(255)).trunci(_T.i8)
        bo.buffer_store(val_i8, rsrc, byte_off, mask=ok, offset_is_bytes=True, cache_modifier=cm)


def in_elt(dtype):
    """Map a torch high-precision input dtype -> flydsl element class. The quant math is all f32;
    only the LDS tile + global load bit-width depends on it, so only bf16/fp16 are supported."""
    import torch

    if dtype == torch.float16:
        return fx.Float16
    if dtype == torch.bfloat16:
        return fx.BFloat16
    raise ValueError(f"unsupported input dtype {dtype}; expected bfloat16 or float16")


def raw_scale_int32(free, contract):
    """int32 count to hold a raw E8M0 scale of logical shape [free, contract//32] (1
    byte/block, row-major, dword-packed). The caller views the int32 buffer as uint8
    [free, contract//32]; free*contract//32 is padded up to a 4-multiple for the view."""
    nbytes = free * (contract // 32)
    return (nbytes + 3) // 4


def _decode_pid(pid, nbk):
    """pid -> (br, bk), row-major. The dense path walks tiles through ``group_m_tile_decode``
    so it can trade the two output planes' run lengths off against each other; the grouped
    path has one row block per expert slab and nothing to trade."""
    br = pid // nbk
    return br, pid - br * nbk


def _ceil128(d):
    return ((d + 127) // 128) * 128


def _warp_reduce_max(val, lane, masks, IRI):
    """Butterfly max-reduce a per-lane f32 across each 2^k-lane group via ds_bpermute
    (dst[l] = src[l ^ mask]): masks=[1,2,4] reduces an 8-lane row-block (1D per-32 scale),
    +[8,16,32] the full 64-lane warp (2D 32x32 tile scale). max is order-independent, so the
    result is bit-identical to the HIP warp_reduce_max_8/64 -- and needs no barrier/LDS."""
    I32 = fx.Int32
    F32 = fx.Float32
    v = val
    for m in masks:
        idx = (lane ^ I32(m)) << I32(2)
        other = I32(rocdl.ds_bpermute(IRI, idx, v.bitcast(I32))).bitcast(F32)
        v = (v > other).select(v, other)
    return v


_WR2 = [1]
_WR8 = [1, 2, 4]
_WR64 = [1, 2, 4, 8, 16, 32]
# Lanes sharing one 32x32 block: the ROW half fixes lane bit 1, the COL half lane bit 5.
_WR32_ROW = [1, 4, 8, 16, 32]
_WR32_COL = [1, 2, 4, 8, 16]


def _row_pass_scales(secp, chunks, lane, va, ep_sub, IRI):
    """(E8M0 exponent, 1/scale) per 16B store half. With ``secp`` the lane pair sharing a
    scale block trades partial amax, which is order-free. Plain Python: values must stay
    in the caller's scope."""
    F32 = fx.Float32
    if not secp:
        a = F32(0.0)
        for i in range(8):
            ai = fm.absf(chunks[i]).reduce("max")
            a = (a > ai).select(a, ai)
        ep = _ep(a, va, ep_sub)
        return [(ep, F32(1.0) / fm.exp2(ep.to(F32)))] * 2
    out = []
    for v in range(2):
        a = F32(0.0)
        for i in range(4):
            ai = fm.absf(chunks[4 * v + i]).reduce("max")
            a = (a > ai).select(a, ai)
        ep = _ep(_warp_reduce_max(a, lane, _WR2, IRI), va, ep_sub)
        out.append((ep, F32(1.0) / fm.exp2(ep.to(F32))))
    return out


def _row_scale_slot(secp, kb, nkb, scales, z):
    """(scale block inside the tile row, biased E8M0 byte) for the row half's scale store.
    ``secp``: the lane's two halves sit in different blocks, so pick the lane<->block
    bijection that needs no cross-lane move."""
    I32 = fx.Int32
    if not secp:
        return kb, scales[0][0] + I32(127)
    odd = kb & I32(1)
    return odd * I32(nkb // 2) + (kb >> I32(1)), (odd == z).select(scales[0][0], scales[1][0]) + I32(127)


def _blk_row_scales(is_2d, chunks, lane, va, ep_sub, IRI):
    """(E8M0 exponent, 1/scale) for the flat 2d row half's two 16B store halves. A lane owns
    16 columns of two rows 16 apart, so ``is_2d`` maxes over the 32 lanes that share the
    32x32 block, and the 1d form only over the column pair completing each row's 32 block."""
    F32 = fx.Float32
    part = []
    for v in range(2):
        a = F32(0.0)
        for i in range(4):
            ai = fm.absf(chunks[4 * v + i]).reduce("max")
            a = (a > ai).select(a, ai)
        part.append(a)
    if is_2d:
        a = (part[0] > part[1]).select(part[0], part[1])
        ep = _ep(_warp_reduce_max(a, lane, _WR32_ROW, IRI), va, ep_sub)
        return [(ep, F32(1.0) / fm.exp2(ep.to(F32)))] * 2
    out = []
    for v in range(2):
        ep = _ep(_warp_reduce_max(part[v], lane, _WR2, IRI), va, ep_sub)
        out.append((ep, F32(1.0) / fm.exp2(ep.to(F32))))
    return out


def _tile_scales(is_2d, lamax, lamaxes, lane, va, ep_sub, IRI):
    """Per-pass (E8M0 exponent, 1/scale) for the 4 passes of a warp's 32x32 chunk. 2d-block: one
    warp_reduce_64 over the whole chunk, shared by all 4 passes; 1d: warp_reduce_8 per pass (per-32
    block). The ``is_2d`` branch runs in plain Python (this helper is not AST-rewritten) so the
    scale stays in the caller's scope -- a kernel-body ``if`` would confine it to a scf.if closure."""
    F32 = fx.Float32
    if is_2d:
        a = _warp_reduce_max(lamax, lane, _WR64, IRI)
        ep = _ep(a, va, ep_sub)
        return [(ep, F32(1.0) / fm.exp2(ep.to(F32)))] * 4
    out = []
    for p in range(4):
        a = _warp_reduce_max(lamaxes[p], lane, _WR8, IRI)
        ep = _ep(a, va, ep_sub)
        out.append((ep, F32(1.0) / fm.exp2(ep.to(F32))))
    return out


# Long-run walk for casts scattered on BOTH padded dims: krun lengthens the read and row
# runs, an XCD-owned pid band lengthens the col write.
_QD_KRUN = 3
_QD_LONG_BAND = 8
_QD_LONG_STRIDE = 8192
# Write-sector completion for the ROW half: at 64B write-request granularity its two 16B
# stores each half-fill a sector, because a lane owns 32 consecutive columns.
_QD_SEC_PAIR = True
# Same completion for the 2d-block halves, whose stores are a lane-wide dword (see flat_2d).
_QD_2D_FLAT = True


def _qdual_tile_cfg(M, K, Mp, Kp):
    """Per-shape tile config for the dual-cast quant kernel (all bit-identical). Default bm=128,bk=128 gives
    BOTH ROW and COL a full 128B write burst (write-burst granularity is the limiter, not
    occupancy). Large-K small-M (Kp>=8192, Mp<=4096) uses bm=128,bk=64 + a column-major walk
    to avoid DRAM-channel camping. Shapes scattered on both axes take the long-run config."""
    sec = dict(sec_pair=_QD_SEC_PAIR, flat_2d=_QD_2D_FLAT)
    if Kp >= 8192 and Mp <= 4096:
        return dict(bm=128, bk=64, pad_extra=4, grid_gm=0, krun=1, xcd_band=0, **sec)
    bm, bk = 128, 128
    long_run = Mp >= _QD_LONG_STRIDE and Kp >= _QD_LONG_STRIDE and (Kp // bk) % _QD_KRUN == 0
    if long_run:
        walk = dict(grid_gm=0, xcd_band=_QD_LONG_BAND, krun=_QD_KRUN)
    else:
        walk = dict(grid_gm=1, xcd_band=0, krun=1)
    return dict(bm=bm, bk=bk, nth=1024, pad_extra=4, **sec, **walk)


# ============================================================================
# Dense dual-cast quant, batched over B experts (B=1 covers the plain 2D case). Grid = B*NBM*NBK
# quantizes all experts of a uniform [B, M, K] input in ONE launch (no per-expert Python loop).
# Each workgroup handles one (batch, M-tile, K-tile) and rebases the input + 4 output bands by
# batch id + a per-batch scale byte base (all 0 for B=1).


def compile_qdual(
    B,
    M,
    K,
    elt=None,
    out_fp8="e4m3",
    Mp=None,
    Kp=None,
    cm_row=_CM,
    cm_col=_CM,
    cm_data_row=_CMD,
    cm_data_col=_CMD,
    pad_extra=4,
    grid_gm=1,
    xcd_band=0,
    num_xcd=8,
    bm=64,
    bk=128,
    nth=512,
    krun=1,
    row_2d=False,
    col_2d=False,
    sec_pair=False,
    flat_2d=True,
):
    """Dense dual-cast mxfp8 quant, batched over B experts (B=1 = plain 2D). Tile [bm x bk]
    (bm*bk==16*nth); ROW half writes fp8+E8M0, COL half stages the transpose through LDS for a
    coalesced write-back (see file-header contract). Real-dim per-expert buffers (row [B,M,Kp],
    col [B,K,Mp]); each workgroup does one (batch, M-tile, K-tile), masking pad rows/cols. Tile
    shape trades the full-128B-burst side; ``grid_gm`` is the row-block depth of the pid walk
    (1 = row-major, 0 = column-major, k = k row blocks per column block) and
    ``xcd_band`` (0 = off) hands each XCD runs of that many consecutive tiles, without which
    the hardware's round-robin pid->XCD split scatters every super-tile across all eight L2s.
    ``cm_row``/``cm_col`` are the E8M0 scale-store policies, ``cm_data_row``/``cm_data_col``
    the fp8 data-store ones; they are separate axes (see _CMD). ``krun`` is how many consecutive
    K-tiles one workgroup walks (see _QD_KRUN). ``sec_pair`` (see _QD_SEC_PAIR) hands each ROW
    lane two 16B units CCOLM apart instead of one 32-column run, so a store instruction spans
    whole 64B sectors; it needs the 4+ lanes per row of bk>=128 and the non-2d layout.
    ``flat_2d`` (see _QD_2D_FLAT) gives the 2d-block halves the same completion by handing a
    wave two whole 32x32 blocks instead of one chunk per warp, keeping the block amax
    wave-local; it needs bm,bk>=64 and is bit-identical to the chunk-per-warp form."""
    if elt is None:
        elt = fx.BFloat16
    va, ep_sub, sat_bnd, cvt = fp8_params(out_fp8)
    Mp = M if Mp is None else Mp
    Kp = K if Kp is None else Kp
    assert Kp % 128 == 0 and Mp % 128 == 0 and Mp >= M and Kp >= K
    # tile = bm*bk elems = nth*16 (each half maps onto nth//2 threads, 32 elems/thread).
    assert bm * bk == 16 * nth and bm % 32 == 0 and bk % 32 == 0
    BMv, BKv, NTHv = bm, bk, nth
    HALF = NTHv // 2
    PAD_L = BKv + pad_extra
    SCMASK = (K % 4) != 0
    LMASK = (Kp != K) and not SCMASK
    # vec8 load path when K%8==0: an 8-col block never straddles the K-pad boundary.
    USE_V8 = (not SCMASK) and (K % 8 == 0)
    NBK = Kp // BKv
    NBM = Mp // BMv
    assert NBK % krun == 0, f"krun={krun} must divide NBK={NBK}"
    NBKG = NBK // krun  # K-tile groups: one workgroup walks krun tiles of a group
    NPB = NBM * NBKG  # workgroups per batch
    GRID_GM = NBM if grid_gm == 0 else min(grid_gm, NBM)
    VPR = BKv // 4
    NKB = BKv // 32
    NMB = BMv // 32
    DWPC = BMv // 4
    LDSC_DW = BKv * BMv // 4
    CW_ITERS_V = LDSC_DW // 4 // NTHv
    assert DWPC % 4 == 0
    SCALEN_ROW = Kp // 32
    SCALEN_COL = Mp // 32
    # 2d-block warp-reduce layout: each 64-lane warp owns one 32x32 chunk (8 rows_in_warp x 8
    # threads_per_row x 4 passes), reducing the tile amax with ds_bpermute (no barrier). The
    # half's HALF threads map to NWARPS warps; NKB*NMB chunks split into ROUNDS per warp.
    NWARPS = HALF // 64
    ROUNDS = (NKB * NMB) // NWARPS
    assert HALF % 64 == 0 and (NKB * NMB) % NWARPS == 0
    # Real-dim (unpadded-row) outputs -> no host .contiguous() copy: row buffers hold M real rows,
    # col buffers K real rows; tile padding rows HW-drop via the per-batch band num_records, and the
    # scale byte stores add a boundary mask so they can't spill into the next batch. Per-batch scale
    # byte counts are dword-aligned (Kp//32, Mp//32 mult of 4) -> base_byte keeps dword-packing.
    _use_2d = row_2d or col_2d
    FLAT2D = _use_2d and flat_2d and NKB >= 2 and NMB >= 2
    NCP = NKB // 2  # 64-column pairs a ROW-half wave walks
    NMP = NMB // 2  # 64-row pairs a COL-half wave walks
    SECP = sec_pair and not _use_2d and NKB >= 4
    CCOLM, VSTEP = (16, NKB * 16) if SECP else (32, 16)
    PB_A_BYTES = M * (Kp // 32)
    PB_AT_BYTES = K * (Mp // 32)
    SP_A_BYTES = PB_A_BYTES * B
    SP_AT_BYTES = PB_AT_BYTES * B

    @flyc.kernel(known_block_size=[NTHv, 1, 1])
    def kern(X: fx.Tensor, Qr: fx.Tensor, ASp: fx.Tensor, AtQd: fx.Tensor, AtSp: fx.Tensor):
        I32 = fx.Int32
        BF = elt.ir_type
        F32 = fx.Float32
        z = I32(0)
        IRI = fx.Int32.ir_type

        @fx.struct
        class Smem:
            tile: fx.Array[elt, BMv * PAD_L, 16]
            ldsc: fx.Array[fx.Int32, LDSC_DW, 16]

        sm = fx.SharedAllocator().allocate(Smem).peek()
        tile = sm.tile
        ldsc = sm.ldsc
        t = fx.thread_idx.x
        pid = fx.block_idx.x
        batch = pid // I32(NPB)
        pib = pid - batch * I32(NPB)
        ptile = xcd_band_remap_pid(pib, NPB, num_xcd, xcd_band)
        br, bkg = group_m_tile_decode(ptile, NBM, NBKG, GRID_GM)
        # per-batch scale byte bases (dword-aligned; keeps the dword-packing invariant)
        base_row_b = batch * I32(PB_A_BYTES)
        base_col_b = batch * I32(PB_AT_BYTES)
        # input band rebased to batch's [M, K] slot: rows past M (padding) read OOB -> 0.
        rx = make_row_band_resource(
            bo.extract_base_index(X), batch * I32(M), (batch + I32(1)) * I32(M), I32(K), 2
        )
        half = t // I32(HALF)
        lt = t - half * I32(HALF)
        # band holds K real rows/batch: transposed stores with gc>=K land OOB and HW-drop.
        raqd = make_row_band_resource(
            bo.extract_base_index(AtQd), batch * I32(K), (batch + I32(1)) * I32(K), I32(Mp), 1
        )
        mbase = br * I32(BMv)
        # Walk krun consecutive K-tiles of one M-block; existing barriers order the reuse.
        for kr in range_constexpr(krun):
            bk = bkg * I32(krun) + I32(kr)
            gbase = (br * I32(BMv)) * I32(K) + bk * I32(BKv)
            if USE_V8:
                VPR8 = BKv // 8
                ITERS8 = BMv * BKv // 8 // NTHv
                for ls in range_constexpr(ITERS8):
                    lin = t + I32(ls * NTHv)
                    lr = lin // I32(VPR8)
                    cv = (lin - lr * I32(VPR8)) * I32(8)
                    pbase = lr * I32(PAD_L) + cv
                    ioff = gbase + lr * I32(K) + cv
                    if LMASK:
                        ioff = (bk * I32(BKv) + cv < I32(K)).select(ioff, I32(_OOB))
                    v = bo.buffer_load(rx, ioff, vec_width=8, dtype=BF)
                    p = fx.add_offset(tile.ptr, fx.make_int_tuple(pbase))
                    fx.make_view(p, fx.make_layout(8, 1)).store(Vec(v))
            else:
                ITERS = BMv * BKv // 4 // NTHv
                for ls in range_constexpr(ITERS):
                    lin = t + I32(ls * NTHv)
                    lr = lin // I32(VPR)
                    cv = (lin - lr * I32(VPR)) * I32(4)
                    pbase = lr * I32(PAD_L) + cv
                    if SCMASK:
                        for j in range_constexpr(4):
                            gcj = bk * I32(BKv) + cv + I32(j)
                            vj = bo.buffer_load(
                                rx,
                                gbase + lr * I32(K) + cv + I32(j),
                                vec_width=1,
                                dtype=BF,
                                mask=(gcj < I32(K)),
                            )
                            pj = fx.add_offset(tile.ptr, fx.make_int_tuple(pbase + I32(j)))
                            fx.make_view(pj, fx.make_layout(1, 1)).store(Vec.from_elements([vj], elt))
                    else:
                        ioff = gbase + lr * I32(K) + cv
                        if LMASK:
                            ioff = (bk * I32(BKv) + cv < I32(K)).select(ioff, I32(_OOB))
                        v = bo.buffer_load(rx, ioff, vec_width=4, dtype=BF)
                        p = fx.add_offset(tile.ptr, fx.make_int_tuple(pbase))
                        fx.make_view(p, fx.make_layout(4, 1)).store(Vec(v))
            _llvm.inline_asm(
                res=None,
                operands_=[],
                asm_string="s_waitcnt vmcnt(0) lgkmcnt(0)",
                constraints="",
                has_side_effects=True,
            )
            rocdl.s_barrier()
            if FLAT2D:
                lane = t & I32(63)
                wv = lt >> I32(6)
                if half == z:
                    rb = wv // I32(NCP)
                    cp = wv - rb * I32(NCP)
                    rq = lane >> I32(2)
                    qc = lane & I32(3)
                    col0 = cp * I32(64) + qc * I32(16)
                    row0 = rb * I32(32) + rq
                    chunks = []
                    for v in range_constexpr(2):
                        loff = (row0 + I32(16 * v)) * I32(PAD_L) + col0
                        for i in range_constexpr(4):
                            p = fx.add_offset(tile.ptr, fx.make_int_tuple(loff + I32(4 * i)))
                            chunks.append(Vec(fx.make_view(p, fx.make_layout(4, 1)).load()).to(F32))
                    scales = _blk_row_scales(row_2d, chunks, lane, va, ep_sub, IRI)
                    rqr = make_row_band_resource(
                        bo.extract_base_index(Qr), batch * I32(M), (batch + I32(1)) * I32(M), I32(Kp), 1
                    )
                    kcol = bk * I32(NKB) + cp * I32(2) + (qc >> I32(1))
                    scale_ok = (qc & I32(1)) == z
                    for v in range_constexpr(2):
                        ep_v, inv_v = scales[v]
                        grow = br * I32(BMv) + row0 + I32(16 * v)
                        words = []
                        for i in range_constexpr(4):
                            qf = chunks[4 * v + i] * inv_v
                            word = I32(cvt(IRI, _sat(qf[0], sat_bnd), _sat(qf[1], sat_bnd), z, 0))
                            word = I32(cvt(IRI, _sat(qf[2], sat_bnd), _sat(qf[3], sat_bnd), word, 1))
                            words.append(word)
                        bo.buffer_store(
                            Vec.from_elements(words, fx.Int32).ir_value(),
                            rqr,
                            grow * I32(Kp) + bk * I32(BKv) + col0,
                            cache_modifier=cm_data_row,
                            offset_is_bytes=True,
                        )
                        dword, jbyte = _raw_scale_dword(grow, kcol, SCALEN_ROW)
                        _store_scale(
                            ASp,
                            SP_A_BYTES,
                            dword,
                            jbyte,
                            ep_v + I32(127),
                            4,
                            ok=(grow < I32(M)) & scale_ok,
                            cm=cm_row,
                            base_byte=base_row_b,
                        )
                else:
                    ckb = wv // I32(NMP)
                    mp = wv - ckb * I32(NMP)
                    c = ckb * I32(32) + (lane & I32(31))
                    mblk = mp * I32(2) + (lane >> I32(5))
                    base = fx.add_offset(tile.ptr, fx.make_int_tuple(c + mblk * I32(32) * I32(PAD_L)))
                    cv2 = Vec(fx.make_view(base, fx.make_layout(32, PAD_L)).load()).to(F32)
                    ca = fm.absf(cv2).reduce("max")
                    camax = (F32(0.0) > ca).select(F32(0.0), ca)
                    if col_2d:
                        camax = _warp_reduce_max(camax, lane, _WR32_COL, IRI)
                    cep = _ep(camax, va, ep_sub)
                    cinv = F32(1.0) / fm.exp2(cep.to(F32))
                    cq = cv2 * cinv
                    cwords = []
                    for wi in range_constexpr(8):
                        word = I32(
                            cvt(IRI, _sat(cq[4 * wi + 0], sat_bnd), _sat(cq[4 * wi + 1], sat_bnd), z, 0)
                        )
                        word = I32(
                            cvt(IRI, _sat(cq[4 * wi + 2], sat_bnd), _sat(cq[4 * wi + 3], sat_bnd), word, 1)
                        )
                        cwords.append(word)
                    csbase = c * I32(DWPC) + mblk * I32(8)
                    for v in range_constexpr(2):
                        sp = fx.add_offset(ldsc.ptr, fx.make_int_tuple(csbase + I32(4 * v)))
                        fx.make_view(sp, fx.make_layout(4, 1)).store(
                            Vec.from_elements(cwords[4 * v : 4 * v + 4], fx.Int32)
                        )
                    gc = bk * I32(BKv) + c
                    mcol = br * I32(BMv // 32) + mblk
                    dword_bc, jbyte_bc = _raw_scale_dword(gc, mcol, SCALEN_COL)
                    _store_scale(
                        AtSp,
                        SP_AT_BYTES,
                        dword_bc,
                        jbyte_bc,
                        cep + I32(127),
                        4,
                        ok=gc < I32(K),
                        cm=cm_col,
                        base_byte=base_col_b,
                    )
            elif _use_2d:
                # 2d-capable warp-reduce layout: one 32x32 chunk per warp, both the per-32 and
                # the shared 32x32 scale via ds_bpermute, so no Phase-0 pass or extra barrier.
                lane = t & I32(63)
                wp = lt >> I32(6)  # warp within this half
                riw = lane >> I32(3)  # row_in_warp 0..7
                tir = lane & I32(7)  # thread_in_row 0..7
                if half == z:
                    rqr = make_row_band_resource(
                        bo.extract_base_index(Qr), batch * I32(M), (batch + I32(1)) * I32(M), I32(Kp), 1
                    )
                    for rnd in range_constexpr(ROUNDS):
                        ci = wp + I32(rnd * NWARPS)
                        cm2 = ci // I32(NKB)  # M-chunk 0..NMB-1
                        cn2 = ci - cm2 * I32(NKB)  # K-chunk 0..NKB-1
                        tcol0 = cn2 * I32(32) + (tir << I32(2))
                        vals = []
                        lamaxes = []
                        lamax = F32(0.0)
                        for p in range_constexpr(4):
                            trow = cm2 * I32(32) + I32(p * 8) + riw
                            pp = fx.add_offset(tile.ptr, fx.make_int_tuple(trow * I32(PAD_L) + tcol0))
                            v4 = Vec(fx.make_view(pp, fx.make_layout(4, 1)).load()).to(F32)
                            vals.append(v4)
                            a = fm.absf(v4).reduce("max")
                            lamaxes.append(a)
                            lamax = (lamax > a).select(lamax, a)
                        scales = _tile_scales(row_2d, lamax, lamaxes, lane, va, ep_sub, IRI)
                        kcol = bk * I32(BKv // 32) + cn2
                        for p in range_constexpr(4):
                            ep_p, inv_p = scales[p]
                            qf = vals[p] * inv_p
                            word = I32(cvt(IRI, _sat(qf[0], sat_bnd), _sat(qf[1], sat_bnd), z, 0))
                            word = I32(cvt(IRI, _sat(qf[2], sat_bnd), _sat(qf[3], sat_bnd), word, 1))
                            grow = br * I32(BMv) + cm2 * I32(32) + I32(p * 8) + riw
                            gcol = bk * I32(BKv) + cn2 * I32(32) + (tir << I32(2))
                            bo.buffer_store(
                                word,
                                rqr,
                                grow * I32(Kp) + gcol,
                                cache_modifier=cm_data_row,
                                offset_is_bytes=True,
                            )
                            dword, jbyte = _raw_scale_dword(grow, kcol, SCALEN_ROW)
                            _store_scale(
                                ASp,
                                SP_A_BYTES,
                                dword,
                                jbyte,
                                ep_p + I32(127),
                                4,
                                ok=(grow < I32(M)) & (tir == z),
                                cm=cm_row,
                                base_byte=base_row_b,
                            )
                else:
                    for rnd in range_constexpr(ROUNDS):
                        ci = wp + I32(rnd * NWARPS)
                        ckc = ci // I32(NMB)  # K-feature chunk 0..NKB-1
                        cmc = ci - ckc * I32(NMB)  # M chunk 0..NMB-1
                        m0 = cmc * I32(32) + (tir << I32(2))
                        vals = []
                        lamaxes = []
                        lamax = F32(0.0)
                        for p in range_constexpr(4):
                            loc_c = ckc * I32(32) + I32(p * 8) + riw
                            base = fx.add_offset(tile.ptr, fx.make_int_tuple(m0 * I32(PAD_L) + loc_c))
                            v4 = Vec(fx.make_view(base, fx.make_layout(4, PAD_L)).load()).to(F32)
                            vals.append(v4)
                            a = fm.absf(v4).reduce("max")
                            lamaxes.append(a)
                            lamax = (lamax > a).select(lamax, a)
                        scales = _tile_scales(col_2d, lamax, lamaxes, lane, va, ep_sub, IRI)
                        for p in range_constexpr(4):
                            cep_p, cinv_p = scales[p]
                            cq = vals[p] * cinv_p
                            word = I32(cvt(IRI, _sat(cq[0], sat_bnd), _sat(cq[1], sat_bnd), z, 0))
                            word = I32(cvt(IRI, _sat(cq[2], sat_bnd), _sat(cq[3], sat_bnd), word, 1))
                            loc_c = ckc * I32(32) + I32(p * 8) + riw
                            # stage into the c-major LDS for the shared coalesced write-back.
                            sp = fx.add_offset(
                                ldsc.ptr, fx.make_int_tuple(loc_c * I32(DWPC) + cmc * I32(8) + tir)
                            )
                            fx.make_view(sp, fx.make_layout(1, 1)).store(Vec.from_elements([word], fx.Int32))
                            gc = bk * I32(BKv) + loc_c
                            mcol = br * I32(BMv // 32) + cmc
                            dword_bc, jbyte_bc = _raw_scale_dword(gc, mcol, SCALEN_COL)
                            _store_scale(
                                AtSp,
                                SP_AT_BYTES,
                                dword_bc,
                                jbyte_bc,
                                cep_p + I32(127),
                                4,
                                ok=(gc < I32(K)) & (tir == z),
                                cm=cm_col,
                                base_byte=base_col_b,
                            )
            else:
                if half == z:
                    row = lt // I32(NKB)
                    kb = lt - row * I32(NKB)
                    ccol = kb * I32(CCOLM)
                    loff = row * I32(PAD_L) + ccol
                    chunks = []
                    for v in range_constexpr(2):
                        for i in range_constexpr(4):
                            p = fx.add_offset(tile.ptr, fx.make_int_tuple(loff + I32(v * VSTEP + 4 * i)))
                            chunks.append(Vec(fx.make_view(p, fx.make_layout(4, 1)).load()).to(F32))
                    scales = _row_pass_scales(SECP, chunks, t & I32(63), va, ep_sub, IRI)
                    grow = br * I32(BMv) + row
                    row_ok = grow < I32(M)
                    rqr = make_row_band_resource(
                        bo.extract_base_index(Qr), batch * I32(M), (batch + I32(1)) * I32(M), I32(Kp), 1
                    )
                    words = []
                    for v in range_constexpr(2):
                        for i in range_constexpr(4):
                            qf = chunks[4 * v + i] * scales[v][1]
                            word = I32(cvt(IRI, _sat(qf[0], sat_bnd), _sat(qf[1], sat_bnd), z, 0))
                            word = I32(cvt(IRI, _sat(qf[2], sat_bnd), _sat(qf[3], sat_bnd), word, 1))
                            words.append(word)
                    row_byte0 = grow * I32(Kp) + bk * I32(BKv) + ccol
                    for v in range_constexpr(2):
                        v4 = Vec.from_elements(words[4 * v : 4 * v + 4], fx.Int32)
                        bo.buffer_store(
                            v4.ir_value(),
                            rqr,
                            row_byte0 + I32(v * VSTEP),
                            cache_modifier=cm_data_row,
                            offset_is_bytes=True,
                        )
                    sblk, e8 = _row_scale_slot(SECP, kb, NKB, scales, z)
                    kcol = bk * I32(NKB) + sblk
                    dword, jbyte = _raw_scale_dword(grow, kcol, SCALEN_ROW)
                    _store_scale(
                        ASp,
                        SP_A_BYTES,
                        dword,
                        jbyte,
                        e8,
                        4,
                        ok=row_ok,
                        cm=cm_row,
                        base_byte=base_row_b,
                    )
                else:
                    c = lt // I32(NMB)
                    mblk = lt - c * I32(NMB)
                    base = fx.add_offset(tile.ptr, fx.make_int_tuple(c + I32(mblk * 32) * I32(PAD_L)))
                    cv2 = Vec(fx.make_view(base, fx.make_layout(32, PAD_L)).load()).to(F32)
                    ca = fm.absf(cv2).reduce("max")
                    camax = (F32(0.0) > ca).select(F32(0.0), ca)
                    cep = _ep(camax, va, ep_sub)
                    cinv = F32(1.0) / fm.exp2(cep.to(F32))
                    cq = cv2 * cinv
                    cwords = []
                    for wi in range_constexpr(8):
                        word = I32(
                            cvt(IRI, _sat(cq[4 * wi + 0], sat_bnd), _sat(cq[4 * wi + 1], sat_bnd), z, 0)
                        )
                        word = I32(
                            cvt(IRI, _sat(cq[4 * wi + 2], sat_bnd), _sat(cq[4 * wi + 3], sat_bnd), word, 1)
                        )
                        cwords.append(word)
                    csbase = c * I32(DWPC) + mblk * I32(8)
                    for v in range_constexpr(2):
                        sp = fx.add_offset(ldsc.ptr, fx.make_int_tuple(csbase + I32(4 * v)))
                        fx.make_view(sp, fx.make_layout(4, 1)).store(
                            Vec.from_elements(cwords[4 * v : 4 * v + 4], fx.Int32)
                        )
                    gc = bk * I32(BKv) + c
                    mcol = br * I32(BMv // 32) + mblk
                    col_ok = gc < I32(K)
                    dword_bc, jbyte_bc = _raw_scale_dword(gc, mcol, SCALEN_COL)
                    _store_scale(
                        AtSp,
                        SP_AT_BYTES,
                        dword_bc,
                        jbyte_bc,
                        cep + I32(127),
                        4,
                        ok=col_ok,
                        cm=cm_col,
                        base_byte=base_col_b,
                    )
            _llvm.inline_asm(
                res=None,
                operands_=[],
                asm_string="s_waitcnt lgkmcnt(0)",
                constraints="",
                has_side_effects=True,
            )
            rocdl.s_barrier()
            for it in range_constexpr(CW_ITERS_V):
                lo = (t + I32(it * NTHv)) * I32(4)
                cc = lo // I32(DWPC)
                dwi0 = lo - cc * I32(DWPC)
                rp = fx.add_offset(ldsc.ptr, fx.make_int_tuple(lo))
                v4 = Vec(fx.make_view(rp, fx.make_layout(4, 1)).load())
                gc = bk * I32(BKv) + cc
                bo.buffer_store(
                    v4.ir_value(),
                    raqd,
                    gc * I32(Mp) + mbase + dwi0 * I32(4),
                    cache_modifier=cm_data_col,
                    offset_is_bytes=True,
                )

    @flyc.jit
    def launch(
        X: fx.Tensor, Qr: fx.Tensor, ASp: fx.Tensor, AtQd: fx.Tensor, AtSp: fx.Tensor, stream: fx.Stream
    ):
        grid = B * NPB
        kern(X, Qr, ASp, AtQd, AtSp).launch(grid=(grid, 1, 1), block=(NTHv, 1, 1), stream=stream)

    return launch


_RAW_QDUAL_BATCHED_CACHE: dict = {}


def quant_mxfp8_raw_batched(x_3d, out_dtype, row_2d=False, col_2d=False):
    """Batched raw-E8M0 dual-cast mxfp8 quant for a uniform [B, M, K] input (grouped-gemm weight
    path), all B experts in ONE launch. Returns quant_mxfp8_raw's 4-tuple stacked to [B, ...]:
    row_fp8 [B, M, Kp] / row_scale [B, M, Kp//32] e8m0 / col_fp8 [B, K, Mp] / col_scale, with
    Kp=ceil(K/128)*128, Mp=ceil(M/128)*128. Bit-identical to the HIP dual-cast per expert."""
    import flydsl.compiler as _flyc
    import torch

    assert x_3d.ndim == 3 and x_3d.is_contiguous()
    assert x_3d.is_cuda and x_3d.dtype in (torch.bfloat16, torch.float16), (
        "quant expects a CUDA bf16/fp16 tensor"
    )
    assert "float8" in str(out_dtype), f"out_dtype must be an fp8 dtype, got {out_dtype}"
    B, M, K = int(x_3d.shape[0]), int(x_3d.shape[1]), int(x_3d.shape[2])
    Mp, Kp = _ceil128(M), _ceil128(K)
    out_fp8 = "e5m2" if out_dtype == torch.float8_e5m2 else "e4m3"

    # Real-dim (unpadded-row) buffers: the kernel writes exactly the consumer-read regions,
    # so the outputs need no host slice + .contiguous() copy (row fp8 [B,M,Kp] keeps the
    # K-pad columns like HIP; col fp8 [B,K,Mp] keeps the N/M-pad columns).
    Qr = torch.empty(B, M, Kp, dtype=out_dtype, device=x_3d.device)
    AtQd = torch.empty(B, K, Mp, dtype=out_dtype, device=x_3d.device)
    ASp = torch.empty(B * M * (Kp // 32) // 4, dtype=torch.int32, device=x_3d.device)
    AtSp = torch.empty(B * K * (Mp // 32) // 4, dtype=torch.int32, device=x_3d.device)
    stream = torch.cuda.current_stream()

    # Tile / pid-walk / store-policy are build parameters, so they belong in the cache key.
    cfg = _qdual_tile_cfg(M, K, Mp, Kp)
    key = (B, M, K, Mp, Kp, x_3d.dtype, out_dtype, row_2d, col_2d) + tuple(sorted(cfg.items()))
    comp = _RAW_QDUAL_BATCHED_CACHE.get(key)
    if comp is None:
        launch = compile_qdual(
            B,
            M,
            K,
            elt=in_elt(x_3d.dtype),
            out_fp8=out_fp8,
            Mp=Mp,
            Kp=Kp,
            row_2d=row_2d,
            col_2d=col_2d,
            **cfg,
        )
        comp = _flyc.compile(launch, x_3d, Qr, ASp, AtQd, AtSp, stream)
        _RAW_QDUAL_BATCHED_CACHE[key] = comp
    comp(x_3d, Qr, ASp, AtQd, AtSp, stream)

    e8 = getattr(torch, "float8_e8m0fnu", torch.uint8)
    row_fp8 = Qr
    row_scale = ASp.view(torch.uint8).view(B, M, Kp // 32).view(e8)
    col_fp8 = AtQd
    col_scale = AtSp.view(torch.uint8).view(B, K, Mp // 32).view(e8)
    return row_fp8, row_scale, col_fp8, col_scale


# ============================================================================
# Grouped dual-cast quant (per-group M zero-pad, offs-driven) -- bit-compatible FlyDSL replacement
# for the HIP grouped_quantize_mxfp8_dual. Grid tiles the col-128 padded M extent (bm divides 128
# => each tile lives in one group); real rows are remapped from the tight input to the row-64 /
# col-128 output, pad rows emit zero data / E8M0=127. Output layout: see grouped_quant_mxfp8_raw.

_GQD_BM, _GQD_BK, _GQD_NTH = 64, 128, 512


def grouped_qdual_grid(M_pad_col, N):
    """Grid (in workgroups) for ``compile_grouped_qdual``'s kernel at this runtime M_pad_col.
    The M extent is a launch argument, so the host owns the tiling arithmetic."""
    return (M_pad_col // _GQD_BM) * (_ceil128(N) // _GQD_BK)


def _load_i32_at(div, idx):
    """Read one int32 scalar at element ``idx`` (runtime fx value OR python const) from an
    i32 logical view. Mirrors the grouped GEMM's ``_load_i32`` (copy-atom -> rmem -> scalar)."""
    if isinstance(idx, int):
        idx = fx.Int32(idx)
    atom = fx.make_copy_atom(rocdl.BufferCopy32b(), fx.Int32)
    reg = fx.make_rmem_tensor(fx.make_layout(1, 1), fx.Int32)
    fx.copy(atom, fx.slice(div, (None, idx)), reg)
    return Vec(fx.memref_load_vec(reg))[0]


def _e8_or_one(amax, ep):
    """E8M0 biased byte: normally ep+127; when the block amax is 0 force 127 (=1.0),
    matching the HIP grouped zero-amax safe path (pad rows / all-zero real blocks)."""
    I32 = fx.Int32
    return (amax == fx.Float32(0.0)).select(I32(127), ep + I32(127))


def _col_store_res(band, base_index, tile_feat_base, gc, feat_local, num_feat, stride, tail_off):
    """64-bit-re-based SRD + i32 voffset for the col-major [num_feat, stride] store, picked at
    trace time (a kernel ``if`` would be rewritten to scf.if and drop the branch-local rsrc).
    band: uniform feature-band base (cheap, needs BKv*stride < 2^31); else per-feature-row
    divergent base (waterfalls, any size). gc>=num_feat bases 0 records -> HW drop."""
    if band:
        return make_row_band_resource(
            base_index, tile_feat_base, num_feat, stride, 1
        ), feat_local * stride + tail_off
    return make_row_band_resource_div(base_index, gc, num_feat, stride, 1), tail_off


def compile_grouped_qdual(
    N,
    G,
    elt=None,
    out_fp8="e4m3",
    pad_extra=4,
    col_data_band=True,
    col_scale_band=True,
):
    """Compile the grouped dual-cast mxfp8 quant. Tile [bm=64 x bk=128] (bm=64 divides the
    128-aligned col-pad boundary so each tile stays in one group). Each WG computes its per-tile
    group metadata (RB/RO/RE/RIE) inline via an O(G) offset scan (no prologue kernel)."""
    if elt is None:
        elt = fx.BFloat16
    va, ep_sub, sat_bnd, cvt = fp8_params(out_fp8)
    N_pad = _ceil128(N)
    assert N % 32 == 0
    BMv, BKv, NTHv = _GQD_BM, _GQD_BK, _GQD_NTH
    assert BMv * BKv == 16 * NTHv and 128 % BMv == 0 and BMv % 32 == 0 and BKv % 32 == 0
    HALF = NTHv // 2
    PAD_L = BKv + pad_extra
    LMASK = N_pad != N
    NBK = N_pad // BKv
    NKB = BKv // 32
    NMB = BMv // 32
    DWPC = BMv // 4
    LDSC_DW = BKv * BMv // 4
    CW_ITERS_V = LDSC_DW // 4 // NTHv
    assert DWPC % 4 == 0
    SCALEN_ROW = N_pad // 32
    # Transposed-store re-base: cheap feature-band when the i32 voffset (< BKv*stride) can't
    # overflow, else per-feature-row divergent (waterfalls) for any size. See _col_store_res.
    COL_DATA_BAND = col_data_band
    COL_SCALE_BAND = col_scale_band

    @flyc.kernel(known_block_size=[NTHv, 1, 1])
    def kern(
        X: fx.Tensor,
        Qr: fx.Tensor,
        ASp: fx.Tensor,
        AtQd: fx.Tensor,
        AtSp: fx.Tensor,
        GO: fx.Tensor,  # tight per-group offs (int32 view of int64 [G+1])
        LR: fx.Tensor,  # OUT: 64-padded per-group lens (int64 [G])
        GR: fx.Tensor,  # OUT: 64-padded per-group offs (int64 [G+1])
        LC: fx.Tensor,  # OUT: 128-padded per-group lens (int64 [G])
        GC: fx.Tensor,  # OUT: 128-padded per-group offs (int64 [G+1])
        m_pad_row: fx.Int32,  # rows of Qr / ASp (64-padded M extent)
        m_pad_col: fx.Int32,  # cols of AtQd (128-padded M extent)
        scalen_col: fx.Int32,  # cols of AtSp = m_pad_col // 32
    ):
        I32 = fx.Int32
        BF = elt.ir_type
        F32 = fx.Float32
        z = I32(0)
        IRI = fx.Int32.ir_type

        @fx.struct
        class Smem:
            tile: fx.Array[elt, BMv * PAD_L, 16]
            ldsc: fx.Array[fx.Int32, LDSC_DW, 16]

        sm = fx.SharedAllocator().allocate(Smem).peek()
        tile = sm.tile
        ldsc = sm.ldsc
        t = fx.thread_idx.x
        pid = fx.block_idx.x
        br, bkc = _decode_pid(pid, I32(NBK))
        base_m = br * I32(BMv)

        # ---- per-tile group metadata computed INLINE (no pad/meta prologue kernels):
        # each WG does the O(G) 64/128-padded-offset scan from GO (loaded to registers
        # first, no dependent load chain), yielding in_rebase / rowbase_out / real_end /
        # in_end. The pid==0 WG also emits the padded lens/offs outputs (threads t<=G). ----
        go_t = rocdl.make_buffer_tensor(GO, max_size=False, num_records_bytes=(G + 1) * 8)
        go_div = fx.logical_divide(go_t, fx.make_layout(1, 1))
        go_vals = [_load_i32_at(go_div, 2 * g) for g in range_constexpr(G + 1)]
        found = z
        go_orig_g = z
        go_orig_g1 = z
        go_row_g = z
        go_col_g = z
        acc_row = z
        acc_col = z
        cap_lr = z
        cap_lc = z
        cap_or = z
        cap_oc = z
        for g in range_constexpr(G):
            prev = go_vals[g]
            nxt = go_vals[g + 1]
            ln = nxt - prev
            lrow = ((ln + I32(63)) // I32(64)) * I32(64)
            lcol = ((ln + I32(127)) // I32(128)) * I32(128)
            inq = (base_m >= acc_col) & (base_m < acc_col + lcol)
            go_col_g = arith.select(inq, acc_col, go_col_g)
            go_orig_g = arith.select(inq, prev, go_orig_g)
            go_orig_g1 = arith.select(inq, nxt, go_orig_g1)
            go_row_g = arith.select(inq, acc_row, go_row_g)
            found = arith.select(inq, I32(1), found)
            atg = t == I32(g)
            cap_lr = arith.select(atg, lrow, cap_lr)
            cap_lc = arith.select(atg, lcol, cap_lc)
            cap_or = arith.select(atg, acc_row, cap_or)  # offs before group g
            cap_oc = arith.select(atg, acc_col, cap_oc)
            acc_row = acc_row + lrow
            acc_col = acc_col + lcol
        cap_or = arith.select(t == I32(G), acc_row, cap_or)  # offs_row[G] = total padded
        cap_oc = arith.select(t == I32(G), acc_col, cap_oc)
        isreal = found == I32(1)
        mrel = base_m - go_col_g
        in_rebase = arith.select(isreal, go_orig_g + mrel, z)  # abs input row for local row 0
        rowbase_out = arith.select(isreal, go_row_g + mrel, z)  # row-64 output base
        real_end = arith.select(isreal, go_col_g + (go_orig_g1 - go_orig_g), base_m)  # real-row end
        in_end = arith.select(isreal, go_orig_g1, z)  # abs input row end of this group
        if pid == z:  # one WG writes the padded lens/offs outputs (num_records masks t>G)
            lr_r = bo.create_buffer_resource(LR, max_size=False, num_records_bytes=I32(G * 8))
            gr_r = bo.create_buffer_resource(GR, max_size=False, num_records_bytes=I32((G + 1) * 8))
            lc_r = bo.create_buffer_resource(LC, max_size=False, num_records_bytes=I32(G * 8))
            gc_r = bo.create_buffer_resource(GC, max_size=False, num_records_bytes=I32((G + 1) * 8))
            bo.buffer_store(cap_lr, lr_r, 2 * t)
            bo.buffer_store(z, lr_r, 2 * t + I32(1))
            bo.buffer_store(cap_or, gr_r, 2 * t)
            bo.buffer_store(z, gr_r, 2 * t + I32(1))
            bo.buffer_store(cap_lc, lc_r, 2 * t)
            bo.buffer_store(z, lc_r, 2 * t + I32(1))
            bo.buffer_store(cap_oc, gc_r, 2 * t)
            bo.buffer_store(z, gc_r, 2 * t + I32(1))

        # ---- load tile: remap TIGHT input row -> LDS, zero pad rows / OOB cols ----
        # Input band bounded to THIS group's real rows [in_rebase, in_end): rows past the group
        # HW-drop -> the hot load loop needs no per-vec4 (grow<real_end) select (which would make
        # every addr depend on the RE global load -> serialized).
        rx = make_row_band_resource(bo.extract_base_index(X), in_rebase, in_end, I32(N), 2)
        # vec8 (16B) loads: N%32==0 => each 8-col block is one side of the N-pad boundary, so a
        # single masked base redirect is exact.
        VPR8 = BKv // 8
        ITERS8 = BMv * BKv // 8 // NTHv
        for ls in range_constexpr(ITERS8):
            lin = t + I32(ls * NTHv)
            lr = lin // I32(VPR8)
            cv = (lin - lr * I32(VPR8)) * I32(8)
            pbase = lr * I32(PAD_L) + cv
            fcol = bkc * I32(BKv) + cv
            ioff = lr * I32(N) + fcol
            if LMASK:
                ioff = (fcol < I32(N)).select(ioff, I32(_OOB))
            v = bo.buffer_load(rx, ioff, vec_width=8, dtype=BF)
            p = fx.add_offset(tile.ptr, fx.make_int_tuple(pbase))
            fx.make_view(p, fx.make_layout(8, 1)).store(Vec(v))
        _llvm.inline_asm(
            res=None,
            operands_=[],
            asm_string="s_waitcnt vmcnt(0) lgkmcnt(0)",
            constraints="",
            has_side_effects=True,
        )
        rocdl.s_barrier()
        half = t // I32(HALF)
        lt = t - half * I32(HALF)
        if half == z:
            # ROW half: (row, kb) = (lt//NKB, lt%NKB); bm rows x NKB K-blocks of 32.
            row = lt // I32(NKB)
            kb = lt - row * I32(NKB)
            loff = row * I32(PAD_L) + kb * I32(32)
            chunks = []
            for i in range_constexpr(8):
                p = fx.add_offset(tile.ptr, fx.make_int_tuple(loff + I32(4 * i)))
                chunks.append(Vec(fx.make_view(p, fx.make_layout(4, 1)).load()).to(F32))
            amax = F32(0.0)
            for i in range_constexpr(8):
                a = fm.absf(chunks[i]).reduce("max")
                amax = (amax > a).select(amax, a)
            grow = base_m + row
            row_ok = grow < real_end
            gcol0 = bkc * I32(BKv) + kb * I32(32)
            ep = _ep(amax, va, ep_sub)
            inv = F32(1.0) / fm.exp2(ep.to(F32))
            # Re-base Qr at this tile's output row so the i32 voffset spans only the tile
            # (base-0 row_out*N_pad overflows once M_pad_row*N_pad > 2^31).
            rqr = make_row_band_resource(bo.extract_base_index(Qr), rowbase_out, m_pad_row, I32(N_pad), 1)
            words = []
            for wi in range_constexpr(8):
                qf = chunks[wi] * inv
                word = I32(cvt(IRI, _sat(qf[0], sat_bnd), _sat(qf[1], sat_bnd), z, 0))
                word = I32(cvt(IRI, _sat(qf[2], sat_bnd), _sat(qf[3], sat_bnd), word, 1))
                words.append(word)
            # Coalesce 8 fp8 words (32 contiguous cols) into 2 vec4 (16B) stores.
            row_byte0 = row * I32(N_pad) + gcol0  # local row within the [rowbase_out, ...) band
            for v in range_constexpr(2):
                off = row_ok.select(row_byte0 + I32(16 * v), I32(_OOB))
                v4 = Vec.from_elements(words[4 * v : 4 * v + 4], fx.Int32)
                bo.buffer_store(v4.ir_value(), rqr, off, cache_modifier=_CM, offset_is_bytes=True)
            kcol = bkc * I32(BKv // 32) + kb
            # Row scale byte matrix [M_pad_row, SCALEN_ROW]: same re-base at rowbase_out.
            rasp = make_row_band_resource(
                bo.extract_base_index(ASp), rowbase_out, m_pad_row, I32(SCALEN_ROW), 1
            )
            e8b = _e8_or_one(amax, ep)
            bo.buffer_store(
                ArithValue(e8b & I32(255)).trunci(_T.i8),
                rasp,
                row * I32(SCALEN_ROW) + kcol,
                mask=row_ok,
                offset_is_bytes=True,
                cache_modifier=_CM,
            )
        else:
            # COL half: (c, mblk) = (lt//NMB, lt%NMB); c = feature col, mblk = M-block of 32.
            c = lt // I32(NMB)
            mblk = lt - c * I32(NMB)
            base = fx.add_offset(tile.ptr, fx.make_int_tuple(c + I32(mblk * 32) * I32(PAD_L)))
            cv2 = Vec(fx.make_view(base, fx.make_layout(32, PAD_L)).load()).to(F32)
            ca = fm.absf(cv2).reduce("max")
            camax = (F32(0.0) > ca).select(F32(0.0), ca)
            cep = _ep(camax, va, ep_sub)
            cinv = F32(1.0) / fm.exp2(cep.to(F32))
            cq = cv2 * cinv
            cwords = []
            for wi in range_constexpr(8):
                word = I32(cvt(IRI, _sat(cq[4 * wi + 0], sat_bnd), _sat(cq[4 * wi + 1], sat_bnd), z, 0))
                word = I32(cvt(IRI, _sat(cq[4 * wi + 2], sat_bnd), _sat(cq[4 * wi + 3], sat_bnd), word, 1))
                cwords.append(word)
            # stage the 8 contiguous col words as 2 vec4 LDS stores (c-major, bm M-bytes/K-col)
            csbase = c * I32(DWPC) + mblk * I32(8)
            for v in range_constexpr(2):
                sp = fx.add_offset(ldsc.ptr, fx.make_int_tuple(csbase + I32(4 * v)))
                fx.make_view(sp, fx.make_layout(4, 1)).store(
                    Vec.from_elements(cwords[4 * v : 4 * v + 4], fx.Int32)
                )
            gc = bkc * I32(BKv) + c
            mcol = br * I32(BMv // 32) + mblk
            # Col scale byte matrix [N, SCALEN_COL], transposed re-base (base-0 overflows once
            # N*SCALEN_COL > 2^31).
            ce8b = _e8_or_one(camax, cep)
            catsp, csoff = _col_store_res(
                COL_SCALE_BAND,
                bo.extract_base_index(AtSp),
                bkc * I32(BKv),
                gc,
                c,
                I32(N),
                scalen_col,
                mcol,
            )
            bo.buffer_store(
                ArithValue(ce8b & I32(255)).trunci(_T.i8),
                catsp,
                csoff,
                offset_is_bytes=True,
                cache_modifier=_CM,
            )
        _llvm.inline_asm(
            res=None, operands_=[], asm_string="s_waitcnt lgkmcnt(0)", constraints="", has_side_effects=True
        )
        rocdl.s_barrier()
        # Coalesced transposed col write from the LDS stage. AtQd is col-major [N, M_pad_col],
        # transposed re-base (base-0 gc*M_pad_col overflows once N*M_pad_col > 2^31).
        for it in range_constexpr(CW_ITERS_V):
            lo = (t + I32(it * NTHv)) * I32(4)
            cc = lo // I32(DWPC)
            dwi0 = lo - cc * I32(DWPC)
            rp = fx.add_offset(ldsc.ptr, fx.make_int_tuple(lo))
            v4 = Vec(fx.make_view(rp, fx.make_layout(4, 1)).load())
            gc = bkc * I32(BKv) + cc
            raqd, off = _col_store_res(
                COL_DATA_BAND,
                bo.extract_base_index(AtQd),
                bkc * I32(BKv),
                gc,
                cc,
                I32(N),
                m_pad_col,
                base_m + dwi0 * I32(4),
            )
            bo.buffer_store(
                v4.ir_value(),
                raqd,
                off,
                cache_modifier=_CM,
                offset_is_bytes=True,
            )

    @flyc.jit
    def launch(
        X: fx.Tensor,
        Qr: fx.Tensor,
        ASp: fx.Tensor,
        AtQd: fx.Tensor,
        AtSp: fx.Tensor,
        GO: fx.Tensor,
        GR: fx.Tensor,
        GC: fx.Tensor,
        LR: fx.Tensor,
        LC: fx.Tensor,
        m_pad_row: fx.Int32,
        m_pad_col: fx.Int32,
        scalen_col: fx.Int32,
        grid: fx.Int32,
        stream: fx.Stream,
    ):
        # Single kernel: per-tile group metadata computed inline (no pad/meta prologue),
        # the pid==0 WG emits the padded lens/offs outputs (LR/GR/LC/GC).
        kern(X, Qr, ASp, AtQd, AtSp, GO, LR, GR, LC, GC, m_pad_row, m_pad_col, scalen_col).launch(
            grid=(grid, 1, 1), block=(NTHv, 1, 1), stream=stream
        )

    return launch


_GROUPED_QDUAL_CACHE: dict = {}


def quant_mxfp8_raw(x, out_dtype, row_2d=False, col_2d=False):
    """FlyDSL raw-E8M0 dual-cast mxfp8 quant for an ARBITRARY 2D [M,K] input (NO host padding),
    bit-for-bit matching the C++ quantize_mxfp8_dual layout: fp8 [M,Kp] row / [K,Mp] col
    (Kp=ceil(K/128)*128, Mp=ceil(M/128)*128) + plain row-major E8M0 scales viewed as
    float8_e8m0fnu. Returns (row_fp8, row_scale, col_fp8, col_scale). Delegates to the B-batched
    path with B=1 (one shared quant kernel; the 2D result is the B=0 slice)."""
    assert x.ndim == 2, f"quant_mxfp8_raw expects 2D, got {x.ndim}D"
    row_fp8, row_scale, col_fp8, col_scale = quant_mxfp8_raw_batched(
        x.unsqueeze(0), out_dtype, row_2d=row_2d, col_2d=col_2d
    )
    return row_fp8[0], row_scale[0], col_fp8[0], col_scale[0]


def grouped_quant_mxfp8_raw(x, group_lens, group_offs, out_dtype):
    """FlyDSL grouped dual-cast mxfp8 quant, drop-in for the HIP grouped_quantize_mxfp8_dual
    (non-shuffle, per-row/col E8M0). ``x`` [total_M, N] bf16/fp16; group_lens/group_offs [G]/[G+1]
    int64 GPU (tight). Returns the HIP 8-tuple: (row fp8 [M_pad_row, N_pad], row e8m0, col fp8
    [N, M_pad_col] transposed, col e8m0, lens/offs_padded_row [G]/[G+1], lens/offs_padded_col)."""
    import flydsl.compiler as _flyc
    import torch

    assert x.ndim == 2 and x.is_contiguous()
    assert x.is_cuda and x.dtype in (torch.bfloat16, torch.float16), (
        "grouped quant expects a CUDA bf16/fp16 tensor"
    )
    assert group_lens.is_cuda and group_offs.is_cuda, "group_lens/group_offs must be CUDA tensors"
    assert "float8" in str(out_dtype), f"out_dtype must be an fp8 dtype, got {out_dtype}"
    total_M, N = int(x.shape[0]), int(x.shape[1])
    G = int(group_lens.shape[0])
    N_pad = _ceil128(N)
    M_pad_row = ((total_M + G * 64) + 63) // 64 * 64
    M_pad_col = ((total_M + G * 128) + 127) // 128 * 128
    grid = grouped_qdual_grid(M_pad_col, N)
    out_fp8 = "e5m2" if out_dtype == torch.float8_e5m2 else "e4m3"

    # Padded per-group lens/offs are filled ON-DEVICE by the quant kernel (no host torch launches).
    lens_row = torch.empty(G, dtype=torch.int64, device=x.device)
    lens_col = torch.empty(G, dtype=torch.int64, device=x.device)
    offs_row = torch.empty(G + 1, dtype=torch.int64, device=x.device)
    offs_col = torch.empty(G + 1, dtype=torch.int64, device=x.device)

    Qr = torch.empty(M_pad_row, N_pad, dtype=out_dtype, device=x.device)
    AtQd = torch.empty(N, M_pad_col, dtype=out_dtype, device=x.device)
    ASp = torch.empty(raw_scale_int32(M_pad_row, N_pad), dtype=torch.int32, device=x.device)
    AtSp = torch.empty(raw_scale_int32(N, M_pad_col), dtype=torch.int32, device=x.device)

    # int32 views of the int64 [G+1] offs (low word carries the value; token offsets < 2^31). The
    # kernel reads GO + fills the padded lens/offs (gr/gc/lr/lc) on-device (no host metadata ops).
    go = group_offs.to(torch.int64).view(torch.int32)
    gr = offs_row.view(torch.int32)
    gc = offs_col.view(torch.int32)
    lr = lens_row.view(torch.int32)
    lc = lens_col.view(torch.int32)

    scalen_col = M_pad_col // 32
    col_data_band = _GQD_BK * M_pad_col < (1 << 31)
    col_scale_band = _GQD_BK * scalen_col < (1 << 31)
    key = (N, G, x.dtype, out_dtype, col_data_band, col_scale_band)
    comp = _GROUPED_QDUAL_CACHE.get(key)
    stream = torch.cuda.current_stream()
    if comp is None:
        launch = compile_grouped_qdual(
            N,
            G,
            elt=in_elt(x.dtype),
            out_fp8=out_fp8,
            col_data_band=col_data_band,
            col_scale_band=col_scale_band,
        )
        comp = _flyc.compile(
            launch, x, Qr, ASp, AtQd, AtSp, go, gr, gc, lr, lc, M_pad_row, M_pad_col, scalen_col, grid, stream
        )
        _GROUPED_QDUAL_CACHE[key] = comp
    comp(x, Qr, ASp, AtQd, AtSp, go, gr, gc, lr, lc, M_pad_row, M_pad_col, scalen_col, grid, stream)

    e8 = getattr(torch, "float8_e8m0fnu", torch.uint8)
    rowwise_scale = ASp.view(torch.uint8)[: M_pad_row * (N_pad // 32)].view(M_pad_row, N_pad // 32).view(e8)
    colwise_scale = AtSp.view(torch.uint8)[: N * (M_pad_col // 32)].view(N, M_pad_col // 32).view(e8)
    return (
        Qr,
        rowwise_scale,
        AtQd,
        colwise_scale,
        lens_row,
        offs_row,
        lens_col,
        offs_col,
    )
