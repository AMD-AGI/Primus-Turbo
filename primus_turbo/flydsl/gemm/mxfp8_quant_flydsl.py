# Dual-cast mxfp8 quant emitting raw E8M0 scales. Tile [bm x bk] (bm*bk==8192), 512
# threads/block: all 512 coalesced-load bm×bk input -> LDS, then halves run concurrently:
#   ROW half (0..255):   fp8 + raw E8M0 row scale                  -> Qr, ASp
#   COL half (256..511): cast fp8 + stage bm M-bytes/K-col -> LDS; raw E8M0 col scale -> AtSp
# then all 512 threads re-read the staged LDS and write the transposed col fp8 -> AtQd with
# coalesced bm-byte bursts (the transposed store is the bandwidth bottleneck; staging through
# LDS turns each K-col's scattered per-lane store into one wide burst).
#
# Outputs are bit-identical to the removed HIP quantize_mxfp8 / quantize_mxfp8_dual:
#   Qr   [Mp, Kp]   fp8 e4m3/e5m2  -- row cast (fwd A operand)
#   ASp  raw E8M0   row scale [Mp, Kp//32], 1 byte/block, row-major (viewed as uint8)
#   AtQd [Kp, Mp]   fp8 e4m3/e5m2  -- col cast (bwd at operand, stored transposed)
#   AtSp raw E8M0   col scale [Kp, Mp//32], 1 byte/block, row-major (viewed as uint8)
# A scale dword packs 4 adjacent blocks; when those come from different workgroups each is
# written with a native byte store (BUFFER_STORE_BYTE -- byte-granular, no dword RMW), so
# the writers don't race. The gemm-side preshuffle (layout-1 A / b-comb) is fused inside
# each GEMM launch, NOT here.
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir.dialects import llvm as _llvm
from flydsl.expr import buffer_ops as bo
from flydsl.expr import math as fm
from flydsl.expr import range_constexpr, rocdl
from flydsl.expr.arith import ArithValue
from flydsl.expr.typing import T as _T
from flydsl.expr.typing import Vector as Vec

from primus_turbo.flydsl.utils.gemm_helper import make_row_band_resource

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


def _store_scale(buf, buf_bytes, dword, jbyte, e8_i32, pack, ok=None, cm=_CM):
    """Write one E8M0 scale. pack==1: broadcast to all 4 bytes of i32[dword] (each dword is
    owned by a single workgroup -> a plain dword store is safe). pack>1: the dword is
    byte-packed across writers in DIFFERENT workgroups, so write the single byte with a
    native byte store (BUFFER_STORE_BYTE -- byte-granular, no dword RMW; the L2 byte-write-
    mask merges the writers' bytes). ``ok`` (K-tail predicate) masks the write when the
    32-col block is past K (pack==1 redirects the offset OOB; pack>1 uses the store mask)."""
    I32 = fx.Int32
    if pack == 1:
        rsrc = bo.create_buffer_resource(buf, max_size=False, num_records_bytes=I32(buf_bytes))
        bcast = e8_i32 | (e8_i32 << I32(8)) | (e8_i32 << I32(16)) | (e8_i32 << I32(24))
        # pack==1 offset is in dword units; buf_bytes (== 4*elems) is safely past the end.
        off = dword if ok is None else ok.select(dword, I32(buf_bytes))
        bo.buffer_store(bcast, rsrc, off, cache_modifier=cm)
    else:
        # pack>1: write the single E8M0 byte at its exact byte offset with a native byte
        # store (BUFFER_STORE_BYTE is byte-granular -- the L2 byte-write-mask merges, with
        # no dword read-modify-write), so the ``pack`` bytes of one dword that come from
        # DIFFERENT workgroups don't race. This replaces a cross-workgroup atomicrmw whose
        # serialisation across CUs dominated the col-scale write (the 4 col bytes of a dword
        # are produced by 4 distinct M-block workgroups). ``ok`` (K-tail) masks the store OOB.
        rsrc = bo.create_buffer_resource(buf, max_size=False, num_records_bytes=I32(buf_bytes))
        byte_off = (dword << I32(2)) | jbyte
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
):
    """Compile the dual-cast mxfp8 quant. Tile [bm x bk] with the constraint bm*bk==8192 so
    both halves map cleanly onto 256 threads; the transposed COL fp8 is staged through LDS
    and written back with coalesced bursts. The two tile shapes trade which output gets a
    full 128B cache-line write:
      (bm=64,  bk=128): COL burst 64B (half line),  ROW write run 128B (full line)  <- default
      (bm=128, bk=64) : COL burst 128B (full line), ROW write run 64B  (half line)
    pad_extra pads the LDS row stride to break the bank conflict the COL half hits reading 32
    M-values at stride bk; grid_swz uses a column-major pid so co-resident workgroups write
    distinct AtQd M-stripes (anti DRAM-channel camping). Kp%128==0 => bk in {64,128} divides
    Kp => no K-tail. Output is bit-identical across all configs."""
    if elt is None:
        elt = fx.BFloat16
    va, ep_sub, sat_bnd, cvt = fp8_params(out_fp8)
    Mp = M if Mp is None else Mp
    Kp = K if Kp is None else Kp
    assert Kp % 128 == 0 and Mp % 128 == 0 and Mp >= M and Kp >= K
    assert bm * bk == 8192 and bm % 32 == 0 and bk % 32 == 0
    BMv, BKv, NTHv = bm, bk, 512
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
        half = t // I32(256)
        lt = t - half * I32(256)
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
            for wi in range_constexpr(8):
                qf = chunks[wi] * inv
                word = I32(cvt(IRI, _sat(qf[0], sat_bnd), _sat(qf[1], sat_bnd), z, 0))
                word = I32(cvt(IRI, _sat(qf[2], sat_bnd), _sat(qf[3], sat_bnd), word, 1))
                bo.buffer_store(
                    word,
                    rqr,
                    grow * I32(Kp) + gcol0 + I32(4 * wi),
                    cache_modifier=cm_row,
                    offset_is_bytes=True,
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
            for wi in range_constexpr(8):
                word = I32(cvt(IRI, _sat(cq[4 * wi + 0], sat_bnd), _sat(cq[4 * wi + 1], sat_bnd), z, 0))
                word = I32(cvt(IRI, _sat(cq[4 * wi + 2], sat_bnd), _sat(cq[4 * wi + 3], sat_bnd), word, 1))
                # stage to ldsc[c*DWPC + mblk*8 + wi] (c-major; bm contiguous M-bytes/K-col)
                sp = fx.add_offset(ldsc.ptr, fx.make_int_tuple(c * I32(DWPC) + mblk * I32(8) + I32(wi)))
                fx.make_view(sp, fx.make_layout(1, 1)).store(Vec.from_elements([word], fx.Int32))
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
    """Per-shape tile config for compile_qdual (all bit-identical). Measured on MI355X
    (gfx950) over Llama shapes: the bm=64 default (full-line ROW write) is the robust choice
    and never regresses; the bm=128 variant (full-line COL write) + grid swizzle wins only
    when K is large and M is small (e.g. 4096x11008)."""
    if Kp >= 8192 and Mp <= 4096:
        return dict(bm=128, bk=64, pad_extra=4, grid_swz=True)
    return dict(bm=64, bk=128, pad_extra=4)


def quant_mxfp8_raw(x, out_dtype, with_trans, axis):
    """FlyDSL raw-E8M0 mxfp8 quant, bit-for-bit matching the C++ quantize_mxfp8 /
    quantize_mxfp8_dual layout for an ARBITRARY [M,K] input -- NO host padding. The kernel
    masks the real dims and writes the consumer's aligned buffers (fp8 [M,Kp] row / [K,Mp]
    col with Kp=ceil(K/128)*128, Mp=ceil(M/128)*128; scales plain row-major E8M0
    [free, contract//32], 1 byte/block) viewed as float8_e8m0fnu -- exactly what
    QuantizedTensor / dequantize_mxfp8 consume.

    with_trans=False single-direction: axis=1 -> rowwise (row fp8/scale), axis=0 -> colwise
    (col fp8/scale). with_trans=True dual: (row_fp8, row_scale, col_fp8, col_scale). The dual
    kernel always computes both directions (the single path discards the unused half -- this
    is the non-hot public/pre-quant API, not the fused gemm fast path).

    A 3D [G,M,N] batched input (grouped-gemm weight path) is quantized per-group and stacked
    -> [G, ...], matching the C++ quantize_mxfp8_dual is_batched output."""
    import flydsl.compiler as _flyc
    import torch

    if x.ndim == 3:
        per = [quant_mxfp8_raw(x[g], out_dtype, with_trans, axis) for g in range(x.shape[0])]
        return tuple(torch.stack([o[i] for o in per], 0) for i in range(len(per[0])))
    assert x.ndim == 2, f"quant_mxfp8_raw expects 2D/3D, got {x.ndim}D"

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
            M, K, elt=in_elt(x.dtype), out_fp8=out_fp8, Mp=Mp, Kp=Kp, **_qdual_tile_cfg(M, K, Mp, Kp)
        )
        comp = _flyc.compile(launch, x, Qr, ASp, AtQd, AtSp, stream)
        _RAW_QDUAL_CACHE[key] = comp
    comp(x, Qr, ASp, AtQd, AtSp, stream)
    e8 = getattr(torch, "float8_e8m0fnu", torch.uint8)
    row_fp8 = Qr[:M, :Kp].contiguous()
    row_scale = ASp.view(torch.uint8)[: Mp * (Kp // 32)].view(Mp, Kp // 32)[:M].contiguous().view(e8)
    col_fp8 = AtQd[:K, :Mp].contiguous()
    col_scale = AtSp.view(torch.uint8)[: Kp * (Mp // 32)].view(Kp, Mp // 32)[:K].contiguous().view(e8)
    if with_trans:
        return row_fp8, row_scale, col_fp8, col_scale
    if axis == 1:
        return row_fp8, row_scale
    return col_fp8, col_scale
