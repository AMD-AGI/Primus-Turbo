###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

# Dual-cast mxfp8 quant emitting raw E8M0 scales. Tile [bm x bk] (=16*nth threads): all threads
# coalesced-load the input -> LDS, then the two halves run concurrently -- ROW half writes fp8 +
# row E8M0 (Qr/ASp), COL half casts + stages the transpose in LDS for a coalesced write-back
# (AtQd) + col E8M0 (AtSp). Bit-identical to the HIP quantize_mxfp8_dual:
#   Qr [Mp,Kp] fp8 row cast / ASp row E8M0 [Mp,Kp//32]
#   AtQd [Kp,Mp] fp8 col cast (transposed) / AtSp col E8M0 [Kp,Mp//32]
# Scales are raw row-major (1 byte/block, dword-packed; cross-workgroup dwords use byte stores to
# avoid RMW races). GEMM-side preshuffle is fused in the GEMM launch, not here.
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir.dialects import llvm as _llvm
from flydsl.expr import arith, range_constexpr, rocdl
from flydsl.expr import buffer_ops as bo
from flydsl.expr import math as fm
from flydsl.expr.arith import ArithValue
from flydsl.expr.typing import T as _T
from flydsl.expr.typing import Vector as Vec

from primus_turbo.flydsl.utils.gemm_helper import make_row_band_resource, make_row_band_resource_div

_CM = 1  # glc: bypass L2 on output stores


def _cvt_pk_bf8_f32(res, src_a, src_b, old, word_sel, **kw):
    # flydsl.expr.rocdl wraps cvt_pk_fp8_f32 (snake-case + _to_ir on operands) but
    # re-exports cvt_pk_bf8_f32 as the raw generated op (no operand conversion). Mirror
    # the fp8 wrapper so e5m2 (bf8) can be called with flydsl exprs like e4m3.
    from flydsl._mlir.dialects.rocdl import cvt_pk_bf8_f32 as _op
    from flydsl.expr.rocdl import _to_ir

    return _op(res=res, src_a=_to_ir(src_a), src_b=_to_ir(src_b), old=_to_ir(old), word_sel=word_sel, **kw)


def fp8_params(out_fp8):
    """Per-output-format quant constants, mirroring the C++ compute_tile_scale /
    cvt_f32x4_to_fp8x4. Returns (val_to_add, ep_sub, sat_bound, cvt_intrinsic):
      val_to_add = 1 << (23 - mbits - 1)  (round-to-even bump; e4m3 mbits=3, e5m2=2)
      ep_sub     = 127 + target_max_pow2  (e4m3 8 -> 135, e5m2 15 -> 142)
      sat_bound  = fp8 max finite (e4m3 448, e5m2 57344)
      cvt        = packed f32->fp8 (e4m3) / f32->bf8 (e5m2) rocdl intrinsic."""
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
        rsrc = bo.create_buffer_resource(buf, max_size=False, num_records_bytes=I32(buf_bytes))
        byte_off = (dword << I32(2)) | jbyte
        if _has_base:
            byte_off = byte_off + base_byte
        val_i8 = ArithValue(e8_i32 & I32(255)).trunci(_T.i8)
        bo.buffer_store(val_i8, rsrc, byte_off, mask=ok, offset_is_bytes=True, cache_modifier=cm)


def in_elt(dtype):
    """Map a torch high-precision input dtype -> flydsl element class. The quant math
    is all f32; only the LDS tile + global load bit-width/interpretation depend on it,
    so bf16 and fp16 are both supported (anything else is unsupported -> caller gates)."""
    import torch

    return fx.Float16 if dtype == torch.float16 else fx.BFloat16


def raw_scale_int32(free, contract):
    """int32 count to hold a raw E8M0 scale of logical shape [free, contract//32] (1
    byte/block, row-major, dword-packed). The caller views the int32 buffer as uint8
    [free, contract//32]; free*contract//32 is padded up to a 4-multiple for the view."""
    nbytes = free * (contract // 32)
    return (nbytes + 3) // 4


def _decode_pid(pid, nbk, nbm, swz):
    """pid -> (br, bk). swz=False: row-major (br=pid//nbk). swz=True: column-major
    (bk=pid//nbm) so consecutive pids vary br. Python-level (NOT in the kernel body) so the
    compile-time branch isn't rewritten into runtime control flow by the AST rewriter."""
    if swz:
        bk = pid // nbm
        return pid - bk * nbm, bk
    br = pid // nbk
    return br, pid - br * nbk


def _ceil128(d):
    return ((d + 127) // 128) * 128


def compile_qdual(
    M,
    K,
    elt=None,
    out_fp8="e4m3",
    Mp=None,
    Kp=None,
    cm_row=_CM,
    cm_col=_CM,
    pad_extra=4,
    grid_swz=False,
    bm=64,
    bk=128,
    nth=512,
):
    """Compile the dual-cast mxfp8 quant. Tile [bm x bk] with bm*bk==16*nth (each half maps to
    nth//2 threads); the transposed COL fp8 is staged through LDS + written back coalesced. Tile
    shape trades which output gets the full 128B write burst (bm=64,bk=128: ROW full/COL half;
    bm=128,bk=64: COL full/ROW half). pad_extra breaks the COL-read LDS bank conflict; grid_swz
    uses column-major pids (anti DRAM-channel camping). Output is bit-identical across configs."""
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
    NBK = Kp // BKv  # exact (Kp % 128 == 0)
    NBM = Mp // BMv
    VPR = BKv // 4  # vec4 per row
    NKB = BKv // 32  # K-blocks (of 32) per row
    NMB = BMv // 32  # M-blocks (of 32) per K-col
    DWPC = BMv // 4  # dwords per K-col staged in LDS
    LDSC_DW = BKv * BMv // 4
    CW_ITERS_V = LDSC_DW // 4 // NTHv  # vec4 col-write iters/thread
    assert DWPC % 4 == 0  # vec4 col write keeps 4 dwords inside one K-col
    SCALEN_ROW = Kp // 32
    SCALEN_COL = Mp // 32
    SP_A_BYTES = raw_scale_int32(Mp, Kp) * 4
    SP_AT_BYTES = raw_scale_int32(Kp, Mp) * 4

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
        br, bk = _decode_pid(pid, I32(NBK), I32(NBM), grid_swz)
        rx = make_row_band_resource(bo.extract_base_index(X), z, I32(M), I32(K), 2)
        gbase = (br * I32(BMv)) * I32(K) + bk * I32(BKv)
        # vec8 (16B) load path when K%8==0: each 8-col block is one side of the K-pad boundary, so
        # a single masked base redirect is exact -- halves the load/LDS-fill instruction count.
        USE_V8 = (not SCMASK) and (K % 8 == 0)
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
                    ioff = (bk * I32(BKv) + cv < I32(K)).select(ioff, I32(0x7FFFFFFF))
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
                            rx, gbase + lr * I32(K) + cv + I32(j), vec_width=1, dtype=BF, mask=(gcj < I32(K))
                        )
                        pj = fx.add_offset(tile.ptr, fx.make_int_tuple(pbase + I32(j)))
                        fx.make_view(pj, fx.make_layout(1, 1)).store(Vec.from_elements([vj], elt))
                else:
                    ioff = gbase + lr * I32(K) + cv
                    if LMASK:
                        ioff = (bk * I32(BKv) + cv < I32(K)).select(ioff, I32(0x7FFFFFFF))
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
        half = t // I32(HALF)
        lt = t - half * I32(HALF)
        if half == z:
            # ROW half: thread -> (row, kb) = (lt//NKB, lt%NKB); bm rows x NKB K-blocks of 32.
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
            grow = br * I32(BMv) + row
            gcol0 = bk * I32(BKv) + kb * I32(32)
            ep = _ep(amax, va, ep_sub)
            inv = F32(1.0) / fm.exp2(ep.to(F32))
            rqr = make_row_band_resource(bo.extract_base_index(Qr), z, I32(Mp), I32(Kp), 1)
            words = []
            for wi in range_constexpr(8):
                qf = chunks[wi] * inv
                word = I32(cvt(IRI, _sat(qf[0], sat_bnd), _sat(qf[1], sat_bnd), z, 0))
                word = I32(cvt(IRI, _sat(qf[2], sat_bnd), _sat(qf[3], sat_bnd), word, 1))
                words.append(word)
            # Coalesce 8 scalar 4B stores -> 2 vec4 (16B) stores (32 contiguous cols).
            row_byte0 = grow * I32(Kp) + gcol0
            for v in range_constexpr(2):
                v4 = Vec.from_elements(words[4 * v : 4 * v + 4], fx.Int32)
                bo.buffer_store(
                    v4.ir_value(), rqr, row_byte0 + I32(16 * v), cache_modifier=cm_row, offset_is_bytes=True
                )
            kcol = bk * I32(BKv // 32) + kb
            dword, jbyte = _raw_scale_dword(grow, kcol, SCALEN_ROW)
            _store_scale(ASp, SP_A_BYTES, dword, jbyte, ep + I32(127), 4, None, cm=cm_row)
        else:
            # COL half: thread -> (c, mblk) = (lt//NMB, lt%NMB); c = K-col, mblk = M-block of 32.
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
            # stage the 8 contiguous col words as 2 vec4 LDS stores (c-major; bm M-bytes/K-col)
            csbase = c * I32(DWPC) + mblk * I32(8)
            for v in range_constexpr(2):
                sp = fx.add_offset(ldsc.ptr, fx.make_int_tuple(csbase + I32(4 * v)))
                fx.make_view(sp, fx.make_layout(4, 1)).store(
                    Vec.from_elements(cwords[4 * v : 4 * v + 4], fx.Int32)
                )
            gc = bk * I32(BKv) + c
            mcol = br * I32(BMv // 32) + mblk
            dword_bc, jbyte_bc = _raw_scale_dword(gc, mcol, SCALEN_COL)
            _store_scale(AtSp, SP_AT_BYTES, dword_bc, jbyte_bc, cep + I32(127), 4, None, cm=cm_col)
        _llvm.inline_asm(
            res=None, operands_=[], asm_string="s_waitcnt lgkmcnt(0)", constraints="", has_side_effects=True
        )
        rocdl.s_barrier()
        # Coalesced transposed write: each thread emits ONE vec4 (16B) store of 4 contiguous
        # M-dwords of a K-col (DWPC%4==0 keeps them inside one K-col); consecutive lanes walk M
        # so the wave coalesces one bm-byte burst per K-col.
        raqd = make_row_band_resource(bo.extract_base_index(AtQd), z, I32(Kp), I32(Mp), 1)
        mbase = br * I32(BMv)
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
                cache_modifier=cm_col,
                offset_is_bytes=True,
            )

    @flyc.jit
    def launch(
        X: fx.Tensor, Qr: fx.Tensor, ASp: fx.Tensor, AtQd: fx.Tensor, AtSp: fx.Tensor, stream: fx.Stream
    ):
        grid = NBM * NBK
        kern(X, Qr, ASp, AtQd, AtSp).launch(grid=(grid, 1, 1), block=(NTHv, 1, 1), stream=stream)

    return launch


_RAW_QDUAL_CACHE: dict = {}


def _qdual_tile_cfg(M, K, Mp, Kp):
    """Per-shape tile config for compile_qdual (all bit-identical; measured on MI355X/gfx950 over
    Llama shapes). Default bm=128,bk=128 (1024 thr): BOTH ROW and COL get full 128B write bursts,
    beating the bm=64 half-line-COL default 1.07-1.17x (write-burst granularity is the limiter,
    not occupancy). Large-K small-M (Kp>=8192, Mp<=4096) prefers bm=128,bk=64 + grid swizzle (anti
    DRAM-channel camping). PT_QDUAL_CFG='bm,bk,nth' overrides for A/B."""
    import os

    _ov = os.environ.get("PT_QDUAL_CFG")
    if _ov:
        b, k, n = (int(x) for x in _ov.split(","))
        return dict(bm=b, bk=k, nth=n, pad_extra=4)
    if Kp >= 8192 and Mp <= 4096:
        return dict(bm=128, bk=64, pad_extra=4, grid_swz=True)
    return dict(bm=128, bk=128, nth=1024, pad_extra=4)


# ============================================================================
# Batched dense dual-cast quant: same [bm x bk] tile / dual-cast body as ``compile_qdual`` plus a
# batch dim so all B experts of a uniform [B, M, K] weight are quantized in ONE launch (grid =
# B*NBM*NBK), replacing the per-expert Python loop + stack. Uniform shapes -> no group search;
# each workgroup rebases the input + 4 output bands by batch id + adds a per-batch scale byte base.
# Per-expert bit-identical to ``compile_qdual``.


def compile_qdual_batched(
    B,
    M,
    K,
    elt=None,
    out_fp8="e4m3",
    Mp=None,
    Kp=None,
    cm_row=_CM,
    cm_col=_CM,
    pad_extra=4,
    grid_swz=False,
    bm=64,
    bk=128,
    nth=512,
):
    """Batched ``compile_qdual``: quantize B experts of [B, M, K] in one launch.
    Buffers are stacked per-expert (row fp8 [B, Mp, Kp], col fp8 [B, Kp, Mp], raw
    E8M0 row/col scales [B, ...]); each workgroup handles one (batch, M-tile, K-tile).
    See ``compile_qdual`` for the tile / dual-cast math; the only additions are the
    per-batch rebase of the input + 4 output bands and the per-batch scale byte base."""
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
    NBK = Kp // BKv
    NBM = Mp // BMv
    NPB = NBM * NBK  # tiles per batch
    VPR = BKv // 4
    NKB = BKv // 32
    NMB = BMv // 32
    DWPC = BMv // 4
    LDSC_DW = BKv * BMv // 4
    CW_ITERS_V = LDSC_DW // 4 // NTHv
    assert DWPC % 4 == 0
    SCALEN_ROW = Kp // 32
    SCALEN_COL = Mp // 32
    # Real-dim (unpadded-row) outputs -> no host .contiguous() copy: row buffers hold M real rows,
    # col buffers K real rows; tile padding rows HW-drop via the per-batch band num_records, and the
    # scale byte stores add a boundary mask so they can't spill into the next batch. Per-batch scale
    # byte counts are dword-aligned (Kp//32, Mp//32 mult of 4) -> base_byte keeps dword-packing.
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
        br, bk = _decode_pid(pib, I32(NBK), I32(NBM), grid_swz)
        # per-batch scale byte bases (dword-aligned; keeps the dword-packing invariant)
        base_row_b = batch * I32(PB_A_BYTES)
        base_col_b = batch * I32(PB_AT_BYTES)
        # input band rebased to batch's [M, K] slot: rows past M (padding) read OOB -> 0.
        rx = make_row_band_resource(
            bo.extract_base_index(X), batch * I32(M), (batch + I32(1)) * I32(M), I32(K), 2
        )
        gbase = (br * I32(BMv)) * I32(K) + bk * I32(BKv)
        # vec8 (16B) load path when K%8==0: each 8-col block is one side of the K-pad boundary, so
        # a single masked base redirect is exact -- halves the load/LDS-fill instruction count.
        USE_V8 = (not SCMASK) and (K % 8 == 0)
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
                    ioff = (bk * I32(BKv) + cv < I32(K)).select(ioff, I32(0x7FFFFFFF))
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
                            rx, gbase + lr * I32(K) + cv + I32(j), vec_width=1, dtype=BF, mask=(gcj < I32(K))
                        )
                        pj = fx.add_offset(tile.ptr, fx.make_int_tuple(pbase + I32(j)))
                        fx.make_view(pj, fx.make_layout(1, 1)).store(Vec.from_elements([vj], elt))
                else:
                    ioff = gbase + lr * I32(K) + cv
                    if LMASK:
                        ioff = (bk * I32(BKv) + cv < I32(K)).select(ioff, I32(0x7FFFFFFF))
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
        half = t // I32(HALF)
        lt = t - half * I32(HALF)
        if half == z:
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
            grow = br * I32(BMv) + row
            gcol0 = bk * I32(BKv) + kb * I32(32)
            row_ok = grow < I32(M)
            ep = _ep(amax, va, ep_sub)
            inv = F32(1.0) / fm.exp2(ep.to(F32))
            # band holds M real rows/batch: grow>=M fp8 stores land OOB and HW-drop.
            rqr = make_row_band_resource(
                bo.extract_base_index(Qr), batch * I32(M), (batch + I32(1)) * I32(M), I32(Kp), 1
            )
            words = []
            for wi in range_constexpr(8):
                qf = chunks[wi] * inv
                word = I32(cvt(IRI, _sat(qf[0], sat_bnd), _sat(qf[1], sat_bnd), z, 0))
                word = I32(cvt(IRI, _sat(qf[2], sat_bnd), _sat(qf[3], sat_bnd), word, 1))
                words.append(word)
            row_byte0 = grow * I32(Kp) + gcol0
            for v in range_constexpr(2):
                v4 = Vec.from_elements(words[4 * v : 4 * v + 4], fx.Int32)
                bo.buffer_store(
                    v4.ir_value(), rqr, row_byte0 + I32(16 * v), cache_modifier=cm_row, offset_is_bytes=True
                )
            kcol = bk * I32(BKv // 32) + kb
            dword, jbyte = _raw_scale_dword(grow, kcol, SCALEN_ROW)
            _store_scale(
                ASp, SP_A_BYTES, dword, jbyte, ep + I32(127), 4, ok=row_ok, cm=cm_row, base_byte=base_row_b
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
            res=None, operands_=[], asm_string="s_waitcnt lgkmcnt(0)", constraints="", has_side_effects=True
        )
        rocdl.s_barrier()
        # band holds K real rows/batch: transposed stores with gc>=K land OOB and HW-drop.
        raqd = make_row_band_resource(
            bo.extract_base_index(AtQd), batch * I32(K), (batch + I32(1)) * I32(K), I32(Mp), 1
        )
        mbase = br * I32(BMv)
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
                cache_modifier=cm_col,
                offset_is_bytes=True,
            )

    @flyc.jit
    def launch(
        X: fx.Tensor, Qr: fx.Tensor, ASp: fx.Tensor, AtQd: fx.Tensor, AtSp: fx.Tensor, stream: fx.Stream
    ):
        grid = B * NBM * NBK
        kern(X, Qr, ASp, AtQd, AtSp).launch(grid=(grid, 1, 1), block=(NTHv, 1, 1), stream=stream)

    return launch


_RAW_QDUAL_BATCHED_CACHE: dict = {}


def quant_mxfp8_raw_batched(x_3d, out_dtype):
    """Batched raw-E8M0 dual-cast mxfp8 quant for a uniform [B, M, K] input (grouped-gemm
    weight path). Quantizes all B experts in ONE FlyDSL launch (no Python loop / stack) and
    returns the SAME 4-tuple as ``quant_mxfp8_raw`` per expert, stacked to [B, ...]:
      row_fp8 [B, M, Kp] fp8, row_scale [B, M, Kp//32] e8m0,
      col_fp8 [B, K, Mp] fp8, col_scale [B, K, Mp//32] e8m0
    (Kp=ceil(K/128)*128, Mp=ceil(M/128)*128). Per-expert bit-identical to the removed
    per-expert loop (standard MX 1x32 blocks), matching the HIP dual-cast output contract."""
    import flydsl.compiler as _flyc
    import torch

    assert x_3d.ndim == 3 and x_3d.is_contiguous()
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

    key = (B, M, K, Mp, Kp, x_3d.dtype, out_dtype)
    comp = _RAW_QDUAL_BATCHED_CACHE.get(key)
    if comp is None:
        launch = compile_qdual_batched(
            B, M, K, elt=in_elt(x_3d.dtype), out_fp8=out_fp8, Mp=Mp, Kp=Kp, **_qdual_tile_cfg(M, K, Mp, Kp)
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
# Grouped dual-cast quant (per-group M zero-pad, offs-driven) -- FlyDSL replacement for the HIP
# ``grouped_quantize_mxfp8_dual``, bit-compatible (raw row/col-major E8M0, no preshuffle):
#   rowwise_output [M_pad_row, N_pad] fp8 / rowwise_scale [M_pad_row, N_pad//32] e8m0
#   colwise_output [N, M_pad_col] fp8 (transposed) / colwise_scale [N, M_pad_col//32] e8m0
# Grid tiles the col-128 padded M extent (bm divides 128 => each tile lives in one group). Real
# rows are remapped from the TIGHT input to the row-64 / col-128 output; pad rows are zero data /
# E8M0=127 (HIP zero-amax safe path); consumers read only the padded-offs regions.


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
        return make_row_band_resource(base_index, tile_feat_base, num_feat, stride, 1), feat_local * stride + tail_off
    return make_row_band_resource_div(base_index, gc, num_feat, stride, 1), tail_off


def compile_grouped_qdual(
    total_M,
    N,
    G,
    M_pad_row,
    M_pad_col,
    elt=None,
    out_fp8="e4m3",
    pad_extra=4,
):
    """Compile the grouped dual-cast mxfp8 quant. Tile [bm=64 x bk=128] (== the fast dense qdual
    config; bm=64 divides the 128-aligned col-pad boundaries so each tile stays in one group).
    Per-tile group metadata (RB/RO/RE/RIE [NBM]) is computed by a fused ``meta`` prologue (O(G)
    search once per tile) so the main kernel does uniform scalar loads and the host does zero
    per-call metadata ops. pad/meta/kern launch back-to-back in one jit stub. Shapes are baked."""
    if elt is None:
        elt = fx.BFloat16
    va, ep_sub, sat_bnd, cvt = fp8_params(out_fp8)
    N_pad = _ceil128(N)
    assert N % 32 == 0
    assert M_pad_col % 128 == 0 and M_pad_row % 64 == 0
    # bm=64,bk=128 (512 thr): the robust grouped choice -- the bm=128 dense big-tile is neutral/
    # worse here (the 4 per-tile scalar metadata loads need occupancy to hide, bm=128 starves it).
    import os

    _ov = os.environ.get("PT_GQDUAL_CFG")  # "bm,bk,nth" A/B override
    if _ov:
        bm, bk, nth = (int(v) for v in _ov.split(","))
    else:
        bm, bk, nth = 64, 128, 512
    assert bm * bk == 16 * nth and 128 % bm == 0 and bm % 32 == 0 and bk % 32 == 0
    BMv, BKv, NTHv = bm, bk, nth
    HALF = NTHv // 2
    PAD_L = BKv + pad_extra
    LMASK = N_pad != N
    NBK = N_pad // BKv
    NBM = M_pad_col // BMv
    NKB = BKv // 32
    NMB = BMv // 32
    DWPC = BMv // 4
    LDSC_DW = BKv * BMv // 4
    CW_ITERS_V = LDSC_DW // 4 // NTHv
    assert DWPC % 4 == 0
    SCALEN_ROW = N_pad // 32
    SCALEN_COL = M_pad_col // 32
    # Transposed-store re-base: cheap feature-band when the i32 voffset (< BKv*stride) can't
    # overflow, else per-feature-row divergent (waterfalls) for any size. See _col_store_res.
    COL_DATA_BAND = BKv * M_pad_col < (1 << 31)
    COL_SCALE_BAND = BKv * SCALEN_COL < (1 << 31)
    META_BLK = 256
    META_GRID = (NBM + META_BLK - 1) // META_BLK

    @flyc.kernel(known_block_size=[1, 1, 1])
    def pad(GO: fx.Tensor, LR: fx.Tensor, ORow: fx.Tensor, LC: fx.Tensor, OCol: fx.Tensor):
        # Fused padded-layout prologue (1 thread, mirrors HIP compute_padded_layout_gpu): from the
        # tight int64 offs fill lens/offs_row (64-aligned) + lens/offs_col (128-aligned), replacing
        # ~7 tiny torch launches. int64 written as (low=val, high=0) i32 pairs (offsets < 2^31).
        I32 = fx.Int32
        z = I32(0)
        go_t = rocdl.make_buffer_tensor(GO, max_size=False, num_records_bytes=(G + 1) * 8)
        go_div = fx.logical_divide(go_t, fx.make_layout(1, 1))
        lr_r = bo.create_buffer_resource(LR, max_size=False, num_records_bytes=I32(G * 8))
        or_r = bo.create_buffer_resource(ORow, max_size=False, num_records_bytes=I32((G + 1) * 8))
        lc_r = bo.create_buffer_resource(LC, max_size=False, num_records_bytes=I32(G * 8))
        oc_r = bo.create_buffer_resource(OCol, max_size=False, num_records_bytes=I32((G + 1) * 8))
        bo.buffer_store(z, or_r, I32(0))
        bo.buffer_store(z, or_r, I32(1))
        bo.buffer_store(z, oc_r, I32(0))
        bo.buffer_store(z, oc_r, I32(1))
        acc_row = z
        acc_col = z
        prev = _load_i32_at(go_div, 0)
        for g in range_constexpr(G):
            nxt = _load_i32_at(go_div, 2 * (g + 1))
            ln = nxt - prev
            lrow = ((ln + I32(63)) // I32(64)) * I32(64)
            lcol = ((ln + I32(127)) // I32(128)) * I32(128)
            bo.buffer_store(lrow, lr_r, I32(2 * g))
            bo.buffer_store(z, lr_r, I32(2 * g + 1))
            bo.buffer_store(lcol, lc_r, I32(2 * g))
            bo.buffer_store(z, lc_r, I32(2 * g + 1))
            acc_row = acc_row + lrow
            acc_col = acc_col + lcol
            bo.buffer_store(acc_row, or_r, I32(2 * (g + 1)))
            bo.buffer_store(z, or_r, I32(2 * (g + 1) + 1))
            bo.buffer_store(acc_col, oc_r, I32(2 * (g + 1)))
            bo.buffer_store(z, oc_r, I32(2 * (g + 1) + 1))
            prev = nxt

    @flyc.kernel(known_block_size=[META_BLK, 1, 1])
    def meta(
        GO: fx.Tensor,
        GR: fx.Tensor,
        GC: fx.Tensor,
        RB: fx.Tensor,
        RO: fx.Tensor,
        RE: fx.Tensor,
        RIE: fx.Tensor,
    ):
        # Prologue: one thread per M-tile bt does the O(G) group search over the int32-view int64
        # offs and writes (RB,RO,RE,RIE): RB=abs input row of local row 0, RO=row-64 output base,
        # RE=real-row end in col-pad space (OOB tile -> base_m), RIE=abs input row end of the group
        # (bounds the input band so past-group rows HW-drop). bt>=NBM OOB-drops on the store.
        I32 = fx.Int32
        bt = fx.block_idx.x * I32(META_BLK) + fx.thread_idx.x
        base_m = bt * I32(BMv)
        go_t = rocdl.make_buffer_tensor(GO, max_size=False, num_records_bytes=(G + 1) * 8)
        go_div = fx.logical_divide(go_t, fx.make_layout(1, 1))
        gr_t = rocdl.make_buffer_tensor(GR, max_size=False, num_records_bytes=(G + 1) * 8)
        gr_div = fx.logical_divide(gr_t, fx.make_layout(1, 1))
        gc_t = rocdl.make_buffer_tensor(GC, max_size=False, num_records_bytes=(G + 1) * 8)
        gc_div = fx.logical_divide(gc_t, fx.make_layout(1, 1))
        found = I32(0)
        go_orig_g = I32(0)
        go_orig_g1 = I32(0)
        go_row_g = I32(0)
        go_col_g = I32(0)
        for g in range_constexpr(G):
            c0 = _load_i32_at(gc_div, 2 * g)  # low word of int64 offs[g]
            c1 = _load_i32_at(gc_div, 2 * g + 2)
            inq = (base_m >= c0) & (base_m < c1)
            go_col_g = arith.select(inq, c0, go_col_g)
            go_orig_g = arith.select(inq, _load_i32_at(go_div, 2 * g), go_orig_g)
            go_orig_g1 = arith.select(inq, _load_i32_at(go_div, 2 * g + 2), go_orig_g1)
            go_row_g = arith.select(inq, _load_i32_at(gr_div, 2 * g), go_row_g)
            found = arith.select(inq, I32(1), found)
        isreal = found == I32(1)
        mrel = base_m - go_col_g
        rb = arith.select(isreal, go_orig_g + mrel, I32(0))
        ro = arith.select(isreal, go_row_g + mrel, I32(0))
        re = arith.select(isreal, go_col_g + (go_orig_g1 - go_orig_g), base_m)
        rie = arith.select(isreal, go_orig_g1, I32(0))
        rb_rsrc = bo.create_buffer_resource(RB, max_size=False, num_records_bytes=I32(NBM * 4))
        ro_rsrc = bo.create_buffer_resource(RO, max_size=False, num_records_bytes=I32(NBM * 4))
        re_rsrc = bo.create_buffer_resource(RE, max_size=False, num_records_bytes=I32(NBM * 4))
        rie_rsrc = bo.create_buffer_resource(RIE, max_size=False, num_records_bytes=I32(NBM * 4))
        bo.buffer_store(rb, rb_rsrc, bt)
        bo.buffer_store(ro, ro_rsrc, bt)
        bo.buffer_store(re, re_rsrc, bt)
        bo.buffer_store(rie, rie_rsrc, bt)

    @flyc.kernel(known_block_size=[NTHv, 1, 1])
    def kern(
        X: fx.Tensor,
        Qr: fx.Tensor,
        ASp: fx.Tensor,
        AtQd: fx.Tensor,
        AtSp: fx.Tensor,
        RB: fx.Tensor,  # per M-tile input row rebase (go_orig_g + mrel), int32 [NBM]
        RO: fx.Tensor,  # per M-tile row-64 output base   (go_row_g  + mrel), int32 [NBM]
        RE: fx.Tensor,  # per M-tile real-row end in col-pad space (go_col_g + Mg), int32 [NBM]
        RIE: fx.Tensor,  # per M-tile abs INPUT row end (go_orig_g1); bounds the input band [NBM]
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
        br, bkc = _decode_pid(pid, I32(NBK), I32(NBM), False)
        base_m = br * I32(BMv)

        # ---- per-tile group metadata (host-precomputed, indexed by M-tile br) ----
        rb_t = rocdl.make_buffer_tensor(RB, max_size=False, num_records_bytes=NBM * 4)
        rb_div = fx.logical_divide(rb_t, fx.make_layout(1, 1))
        ro_t = rocdl.make_buffer_tensor(RO, max_size=False, num_records_bytes=NBM * 4)
        ro_div = fx.logical_divide(ro_t, fx.make_layout(1, 1))
        re_t = rocdl.make_buffer_tensor(RE, max_size=False, num_records_bytes=NBM * 4)
        re_div = fx.logical_divide(re_t, fx.make_layout(1, 1))
        rie_t = rocdl.make_buffer_tensor(RIE, max_size=False, num_records_bytes=NBM * 4)
        rie_div = fx.logical_divide(rie_t, fx.make_layout(1, 1))
        in_rebase = _load_i32_at(rb_div, br)  # abs input row for this tile's local row 0
        rowbase_out = _load_i32_at(ro_div, br)  # row-64 output base for local row 0
        real_end = _load_i32_at(re_div, br)  # real-row end (col-pad space); grow>=end -> pad
        in_end = _load_i32_at(rie_div, br)  # abs input row end of this group (go_orig_g1)

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
                ioff = (fcol < I32(N)).select(ioff, I32(0x7FFFFFFF))
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
            rqr = make_row_band_resource(bo.extract_base_index(Qr), rowbase_out, I32(M_pad_row), I32(N_pad), 1)
            words = []
            for wi in range_constexpr(8):
                qf = chunks[wi] * inv
                word = I32(cvt(IRI, _sat(qf[0], sat_bnd), _sat(qf[1], sat_bnd), z, 0))
                word = I32(cvt(IRI, _sat(qf[2], sat_bnd), _sat(qf[3], sat_bnd), word, 1))
                words.append(word)
            # Coalesce 8 fp8 words (32 contiguous cols) into 2 vec4 (16B) stores.
            row_byte0 = row * I32(N_pad) + gcol0  # local row within the [rowbase_out, ...) band
            for v in range_constexpr(2):
                off = row_ok.select(row_byte0 + I32(16 * v), I32(0x7FFFFFFF))
                v4 = Vec.from_elements(words[4 * v : 4 * v + 4], fx.Int32)
                bo.buffer_store(v4.ir_value(), rqr, off, cache_modifier=_CM, offset_is_bytes=True)
            kcol = bkc * I32(BKv // 32) + kb
            # Row scale byte matrix [M_pad_row, SCALEN_ROW]: same re-base at rowbase_out.
            rasp = make_row_band_resource(bo.extract_base_index(ASp), rowbase_out, I32(M_pad_row), I32(SCALEN_ROW), 1)
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
                COL_SCALE_BAND, bo.extract_base_index(AtSp), bkc * I32(BKv), gc, c, I32(N), I32(SCALEN_COL), mcol
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
                I32(M_pad_col),
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
        RB: fx.Tensor,
        RO: fx.Tensor,
        RE: fx.Tensor,
        RIE: fx.Tensor,
        LR: fx.Tensor,
        LC: fx.Tensor,
        stream: fx.Stream,
    ):
        # 0) pad_layout fills GR/GC (=offs_row/col) + LR/LC (=lens_row/col) from tight GO;
        # 1) meta prologue fills RB/RO/RE/RIE (reads GR/GC); 2) main quant reads them.
        pad(GO, LR, GR, LC, GC).launch(grid=(1, 1, 1), block=(1, 1, 1), stream=stream)
        meta(GO, GR, GC, RB, RO, RE, RIE).launch(
            grid=(META_GRID, 1, 1), block=(META_BLK, 1, 1), stream=stream
        )
        grid = NBM * NBK
        kern(X, Qr, ASp, AtQd, AtSp, RB, RO, RE, RIE).launch(
            grid=(grid, 1, 1), block=(NTHv, 1, 1), stream=stream
        )

    return launch


_GROUPED_QDUAL_CACHE: dict = {}


def quant_mxfp8_raw(x, out_dtype):
    """FlyDSL raw-E8M0 dual-cast mxfp8 quant for an ARBITRARY 2D [M,K] input (NO host padding),
    bit-for-bit matching the C++ quantize_mxfp8_dual layout: fp8 [M,Kp] row / [K,Mp] col
    (Kp=ceil(K/128)*128, Mp=ceil(M/128)*128) + plain row-major E8M0 scales [free, contract//32]
    viewed as float8_e8m0fnu. Returns (row_fp8, row_scale, col_fp8, col_scale)."""
    import flydsl.compiler as _flyc
    import torch

    assert x.ndim == 2, f"quant_mxfp8_raw expects 2D, got {x.ndim}D"

    out_fp8 = "e5m2" if out_dtype == torch.float8_e5m2 else "e4m3"
    M, K = x.shape
    Mp, Kp = _ceil128(M), _ceil128(K)
    key = (M, K, Mp, Kp, x.dtype, out_dtype)
    comp = _RAW_QDUAL_CACHE.get(key)
    Qr = torch.empty(Mp, Kp, dtype=out_dtype, device=x.device)
    AtQd = torch.empty(Kp, Mp, dtype=out_dtype, device=x.device)
    # No zero-init needed: every in-range scale byte is written exactly once by a native
    # byte store, and the consumer reads only [:nbytes] (the dword-alignment pad is unread).
    ASp = torch.empty(raw_scale_int32(Mp, Kp), dtype=torch.int32, device=x.device)
    AtSp = torch.empty(raw_scale_int32(Kp, Mp), dtype=torch.int32, device=x.device)
    stream = torch.cuda.current_stream()
    if comp is None:
        launch = compile_qdual(
            M,
            K,
            elt=in_elt(x.dtype),
            out_fp8=out_fp8,
            Mp=Mp,
            Kp=Kp,
            **_qdual_tile_cfg(M, K, Mp, Kp),
        )
        comp = _flyc.compile(launch, x, Qr, ASp, AtQd, AtSp, stream)
        _RAW_QDUAL_CACHE[key] = comp
    comp(x, Qr, ASp, AtQd, AtSp, stream)
    e8 = getattr(torch, "float8_e8m0fnu", torch.uint8)
    row_fp8 = Qr[:M, :Kp].contiguous()
    row_scale = ASp.view(torch.uint8)[: Mp * (Kp // 32)].view(Mp, Kp // 32)[:M].contiguous().view(e8)
    col_fp8 = AtQd[:K, :Mp].contiguous()
    col_scale = AtSp.view(torch.uint8)[: Kp * (Mp // 32)].view(Kp, Mp // 32)[:K].contiguous().view(e8)
    return row_fp8, row_scale, col_fp8, col_scale


def grouped_quant_mxfp8_raw(x, group_lens, group_offs, out_dtype):
    """FlyDSL grouped dual-cast mxfp8 quant, drop-in for the HIP
    ``grouped_quantize_mxfp8_dual`` (non-shuffle, per-row/col E8M0).  Returns the same
    8-tuple contract:
      (rowwise_output [M_pad_row, N_pad] fp8, rowwise_scale [M_pad_row, N_pad//32] e8m0,
       colwise_output [N, M_pad_col]     fp8, colwise_scale [N, M_pad_col//32] e8m0,
       group_lens_padded_rowwise [G], group_offs_padded_rowwise [G+1],
       group_lens_padded_colwise [G], group_offs_padded_colwise [G+1]).

    ``x`` [total_M, N] bf16/fp16 contiguous; ``group_lens`` [G] / ``group_offs`` [G+1] int64 GPU
    (tight). M_pad_row / M_pad_col use the host upper bounds (cdiv(total_M + G*align, align)*align)
    like the HIP op, so the grid over-covers and OOB tiles emit all-zero fp8 / E8M0=127. The padded
    per-group lens/offs are filled on-device by the fused ``pad`` prologue (no D2H)."""
    import flydsl.compiler as _flyc
    import torch

    assert x.ndim == 2 and x.is_contiguous()
    assert x.dtype in (torch.bfloat16, torch.float16)
    total_M, N = int(x.shape[0]), int(x.shape[1])
    G = int(group_lens.shape[0])
    N_pad = _ceil128(N)
    M_pad_row = ((total_M + G * 64) + 63) // 64 * 64
    M_pad_col = ((total_M + G * 128) + 127) // 128 * 128
    out_fp8 = "e5m2" if out_dtype == torch.float8_e5m2 else "e4m3"

    # Padded per-group lens/offs are filled ON-DEVICE by the fused ``pad`` prologue (replacing ~7
    # tiny host torch launches). Allocate uninitialized; ``pad`` fills them before meta/kern read.
    lens_row = torch.empty(G, dtype=torch.int64, device=x.device)
    lens_col = torch.empty(G, dtype=torch.int64, device=x.device)
    offs_row = torch.empty(G + 1, dtype=torch.int64, device=x.device)
    offs_col = torch.empty(G + 1, dtype=torch.int64, device=x.device)

    Qr = torch.empty(M_pad_row, N_pad, dtype=out_dtype, device=x.device)
    AtQd = torch.empty(N, M_pad_col, dtype=out_dtype, device=x.device)
    ASp = torch.empty(raw_scale_int32(M_pad_row, N_pad), dtype=torch.int32, device=x.device)
    AtSp = torch.empty(raw_scale_int32(N, M_pad_col), dtype=torch.int32, device=x.device)

    # int32 views of the int64 [G+1] offs (low word carries the value; token offsets < 2^31). The
    # fused ``meta`` prologue reads these + fills RB/RO/RE/RIE on-device (no host metadata ops).
    go = group_offs.to(torch.int64).view(torch.int32)
    gr = offs_row.view(torch.int32)
    gc = offs_col.view(torch.int32)
    lr = lens_row.view(torch.int32)
    lc = lens_col.view(torch.int32)
    NBM = M_pad_col // 64
    RB = torch.empty(NBM, dtype=torch.int32, device=x.device)
    RO = torch.empty(NBM, dtype=torch.int32, device=x.device)
    RE = torch.empty(NBM, dtype=torch.int32, device=x.device)
    RIE = torch.empty(NBM, dtype=torch.int32, device=x.device)

    key = (total_M, N, G, M_pad_row, M_pad_col, x.dtype, out_dtype)
    comp = _GROUPED_QDUAL_CACHE.get(key)
    stream = torch.cuda.current_stream()
    if comp is None:
        launch = compile_grouped_qdual(
            total_M, N, G, M_pad_row, M_pad_col, elt=in_elt(x.dtype), out_fp8=out_fp8
        )
        comp = _flyc.compile(launch, x, Qr, ASp, AtQd, AtSp, go, gr, gc, RB, RO, RE, RIE, lr, lc, stream)
        _GROUPED_QDUAL_CACHE[key] = comp
    comp(x, Qr, ASp, AtQd, AtSp, go, gr, gc, RB, RO, RE, RIE, lr, lc, stream)

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
