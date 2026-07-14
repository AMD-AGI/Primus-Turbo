###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
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

from primus_turbo.flydsl.utils.gemm_helper import make_row_band_resource, make_row_band_resource_div

_CM = 1  # glc: bypass L2 on output stores
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


def _qdual_tile_cfg(M, K, Mp, Kp):
    """Per-shape tile config for the dual-cast quant kernel (all bit-identical). Default bm=128,bk=128 gives
    BOTH ROW and COL a full 128B write burst (write-burst granularity is the limiter, not
    occupancy). Large-K small-M (Kp>=8192, Mp<=4096) uses bm=128,bk=64 + grid swizzle to avoid
    DRAM-channel camping."""
    if Kp >= 8192 and Mp <= 4096:
        return dict(bm=128, bk=64, pad_extra=4, grid_swz=True)
    return dict(bm=128, bk=128, nth=1024, pad_extra=4)


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
    pad_extra=4,
    grid_swz=False,
    bm=64,
    bk=128,
    nth=512,
):
    """Dense dual-cast mxfp8 quant, batched over B experts (B=1 = plain 2D). Tile [bm x bk]
    (bm*bk==16*nth); ROW half writes fp8+E8M0, COL half stages the transpose through LDS for a
    coalesced write-back (see file-header contract). Real-dim per-expert buffers (row [B,M,Kp],
    col [B,K,Mp]); each workgroup does one (batch, M-tile, K-tile), masking pad rows/cols. Tile
    shape trades the full-128B-burst side; grid_swz = column-major pids (anti DRAM camping)."""
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
                            rx, gbase + lr * I32(K) + cv + I32(j), vec_width=1, dtype=BF, mask=(gcj < I32(K))
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

    key = (B, M, K, Mp, Kp, x_3d.dtype, out_dtype)
    comp = _RAW_QDUAL_BATCHED_CACHE.get(key)
    if comp is None:
        launch = compile_qdual(
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
# Grouped dual-cast quant (per-group M zero-pad, offs-driven) -- bit-compatible FlyDSL replacement
# for the HIP grouped_quantize_mxfp8_dual. Grid tiles the col-128 padded M extent (bm divides 128
# => each tile lives in one group); real rows are remapped from the tight input to the row-64 /
# col-128 output, pad rows emit zero data / E8M0=127. Output layout: see grouped_quant_mxfp8_raw.


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
    total_M,
    N,
    G,
    M_pad_row,
    M_pad_col,
    elt=None,
    out_fp8="e4m3",
    pad_extra=4,
):
    """Compile the grouped dual-cast mxfp8 quant. Tile [bm=64 x bk=128] (bm=64 divides the
    128-aligned col-pad boundary so each tile stays in one group). Each WG computes its per-tile
    group metadata (RB/RO/RE/RIE) inline via an O(G) offset scan (no prologue kernel). Shapes baked."""
    if elt is None:
        elt = fx.BFloat16
    va, ep_sub, sat_bnd, cvt = fp8_params(out_fp8)
    N_pad = _ceil128(N)
    assert N % 32 == 0
    assert M_pad_col % 128 == 0 and M_pad_row % 64 == 0
    # bm=64,bk=128 (512 thr): the robust grouped choice -- the bm=128 dense big-tile is neutral/
    # worse here (the 4 per-tile scalar metadata loads need occupancy to hide, bm=128 starves it).
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
            rqr = make_row_band_resource(
                bo.extract_base_index(Qr), rowbase_out, I32(M_pad_row), I32(N_pad), 1
            )
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
                bo.extract_base_index(ASp), rowbase_out, I32(M_pad_row), I32(SCALEN_ROW), 1
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
                I32(SCALEN_COL),
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
        LR: fx.Tensor,
        LC: fx.Tensor,
        stream: fx.Stream,
    ):
        # Single kernel: per-tile group metadata computed inline (no pad/meta prologue),
        # the pid==0 WG emits the padded lens/offs outputs (LR/GR/LC/GC).
        grid = NBM * NBK
        kern(X, Qr, ASp, AtQd, AtSp, GO, LR, GR, LC, GC).launch(
            grid=(grid, 1, 1), block=(NTHv, 1, 1), stream=stream
        )

    return launch


_GROUPED_QDUAL_CACHE: dict = {}


def quant_mxfp8_raw(x, out_dtype):
    """FlyDSL raw-E8M0 dual-cast mxfp8 quant for an ARBITRARY 2D [M,K] input (NO host padding),
    bit-for-bit matching the C++ quantize_mxfp8_dual layout: fp8 [M,Kp] row / [K,Mp] col
    (Kp=ceil(K/128)*128, Mp=ceil(M/128)*128) + plain row-major E8M0 scales viewed as
    float8_e8m0fnu. Returns (row_fp8, row_scale, col_fp8, col_scale). Delegates to the B-batched
    path with B=1 (one shared quant kernel; the 2D result is the B=0 slice)."""
    assert x.ndim == 2, f"quant_mxfp8_raw expects 2D, got {x.ndim}D"
    row_fp8, row_scale, col_fp8, col_scale = quant_mxfp8_raw_batched(x.unsqueeze(0), out_dtype)
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

    key = (total_M, N, G, M_pad_row, M_pad_col, x.dtype, out_dtype)
    comp = _GROUPED_QDUAL_CACHE.get(key)
    stream = torch.cuda.current_stream()
    if comp is None:
        launch = compile_grouped_qdual(
            total_M, N, G, M_pad_row, M_pad_col, elt=in_elt(x.dtype), out_fp8=out_fp8
        )
        comp = _flyc.compile(launch, x, Qr, ASp, AtQd, AtSp, go, gr, gc, lr, lc, stream)
        _GROUPED_QDUAL_CACHE[key] = comp
    comp(x, Qr, ASp, AtQd, AtSp, go, gr, gc, lr, lc, stream)

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
