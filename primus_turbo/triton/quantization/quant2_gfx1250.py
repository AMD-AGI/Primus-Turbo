###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""quant2 (PRIMUS_QUANT2_T2 / _DQ / _SWB, default off): Gluon (Triton main, gluon cdna5) dual FP8 activation
quant core for gfx1250, byte-identical to quant_g90._quant_dual_t2_kernel, quant_blk_gfx1250._dual_1x128_128x1_kernel and
quant_blk_gluon_gfx1250._swb_tdm_kernel.

Per 128x128 bf16 tile (persistent programs, NS-stage TDM prefetch ring in LDS, rows padded by 16 B):
  * row pass: one lane per row (ds_load_b128), in-lane 1x128 amax, quantize, regroup in-warp to 16 B x 8 lanes per
    row (convert_layout) and store coalesced;
  * column pass: one lane per column via transposed LDS reads (ds_load_tr16_b128), in-lane 128x1 amax, quantize,
    regroup in-warp and store qT rows coalesced.
No cross-lane or cross-warp amax reductions; amax is order-independent, so bytes equal the int kernels'.

Conventions (CONV constexpr):
  0 = G90 (quant_g90._quant_dual_t2_kernel): amax floored at 1e-4, inf amax -> multiplier 0 / byte 255,
      q clamped to +-FP8_MAX; scale layouts SL 1 (g90 byte) or 2 (dsv4up packed).
  1 = FLY (quant_blk_gfx1250._dual_1x128_128x1_kernel): zero block -> e = 0, e clamped to [-126, 126], no clamp;
      row scales SL 0 (x4 int32 words) or 2 (packed), column scales compact bytes (CT) or SL.
"""
import os

import torch
import triton
from triton.experimental import gluon
import triton.experimental.gluon.language as gl
from triton.experimental.gluon.language.amd.cdna5 import tdm


@gluon.jit
def _e_g90(amax, MAX_EXP: gl.constexpr):
    """quant_epi_gfx1250.ue8m0_exp_g90"""
    a = gl.maximum(amax, 1e-4)
    bits = a.to(gl.int32, bitcast=True)
    fld = (bits >> 23) & 0xFF
    e = fld - (127 + MAX_EXP) + ((bits & 0x7FFFFF) > 0x600000).to(gl.int32)
    return gl.where(fld == 0xFF, 128, e)


@gluon.jit
def _p2neg_g90(e):
    """quant_epi_gfx1250.pow2_neg"""
    return gl.where(e > 127, 0, (127 - e) << 23).to(gl.float32, bitcast=True)


@gluon.jit
def _e_fly(amax, MAX_EXP: gl.constexpr):
    """quant_epi_gfx1250.ue8m0_exp_fly"""
    bits = amax.to(gl.int32, bitcast=True)
    fld = (bits >> 23) & 0xFF
    e = fld - (127 + MAX_EXP) + ((bits & 0x7FFFFF) > 0x600000).to(gl.int32)
    e = gl.where(fld == 0xFF, 126, e)
    e = gl.where(amax > 0, e, 0)
    return gl.minimum(gl.maximum(e, -126), 126)


@gluon.jit
def _p2neg_fly(e):
    return ((127 - e) << 23).to(gl.float32, bitcast=True)


@gluon.jit
def _store_scale(S, r, kb, e, KB, SL: gl.constexpr):
    """quant_epi_gfx1250.store_scale: SL 0 (x4 int32 [R, KB]), 1 (byte [R, KB]), 2 (dsv4up packed)."""
    r64 = r.to(gl.int64)
    if SL == 0:
        gl.store(S.to(gl.pointer_type(gl.int32)) + r64 * KB + kb, ((e + 127) & 0xFF) * 0x01010101)
    elif SL == 1:
        gl.store(S + r64 * KB + kb, ((e + 127) & 0xFF).to(gl.uint8))
    else:
        w = (r64 >> 7) * (KB * 128) + kb * 128 + (r64 & 31) * 4 + ((r64 >> 5) & 3)
        gl.store(S.to(gl.pointer_type(gl.int32)) + w, ((e + 127) & 0xFF) * 0x01010101)


@gluon.jit
def _quant(x, e_axis: gl.constexpr, FP8_MAX: gl.constexpr, MAX_EXP: gl.constexpr, CONV: gl.constexpr, dty):
    """x fp32 [128, 128] -> (fp8 q, exponent per slice along the other axis)."""
    if CONV == 0:
        e = _e_g90(gl.max(gl.abs(x), axis=e_axis), MAX_EXP)
        if e_axis == 1:
            q = gl.clamp(x * _p2neg_g90(e)[:, None], -FP8_MAX, FP8_MAX)
        else:
            q = gl.clamp(x * _p2neg_g90(e)[None, :], -FP8_MAX, FP8_MAX)
    else:
        e = _e_fly(gl.max(gl.abs(x), axis=e_axis), MAX_EXP)
        if e_axis == 1:
            q = x * _p2neg_fly(e)[:, None]
        else:
            q = x * _p2neg_fly(e)[None, :]
    return q.to(dty), e


@gluon.jit
def _q2_tile(src, tm, tn, rv_r, rv_c, QR, ER, QT, ET, M, N, FP8_MAX: gl.constexpr, MAX_EXP: gl.constexpr,
             CONV: gl.constexpr, SLR: gl.constexpr, SLT: gl.constexpr, CTT: gl.constexpr, ZM: gl.constexpr,
             NW: gl.constexpr):
    """One [128 rows (tile tm of M), 128 cols (tile tn of N)] bf16 tile in LDS `src`: QR [M, N] + ER (block tn),
    QT [N, M] + ET (block tm). ZM: zero the rows whose rv_* flag is False (gathered padding rows)."""
    gl.static_assert(NW == 4)
    LR: gl.constexpr = gl.BlockedLayout([1, 128], [32, 1], [NW, 1], [1, 0])  # lane = row
    LC: gl.constexpr = gl.BlockedLayout([128, 1], [1, 32], [1, NW], [0, 1])  # lane = column
    # coalesced store layouts (16 contiguous bytes per lane, 8 lanes per 128-byte row) with the same warp -> rows /
    # columns map as LR / LC, so the regrouping stays inside each warp
    LSR: gl.constexpr = gl.DistributedLinearLayout(
        reg_bases=[[0, 1], [0, 2], [0, 4], [0, 8], [4, 0], [8, 0], [16, 0]],
        lane_bases=[[0, 16], [0, 32], [0, 64], [1, 0], [2, 0]], warp_bases=[[32, 0], [64, 0]], block_bases=[],
        shape=[128, 128])
    LSC: gl.constexpr = gl.DistributedLinearLayout(
        reg_bases=[[1, 0], [2, 0], [4, 0], [8, 0], [0, 4], [0, 8], [0, 16]],
        lane_bases=[[16, 0], [32, 0], [64, 0], [0, 1], [0, 2]], warp_bases=[[0, 32], [0, 64]], block_bases=[],
        shape=[128, 128])
    # ---- row pass (1x128 along N)
    x = src.load(LR).to(gl.float32)
    if ZM:
        x = gl.where(rv_r[:, None], x, 0.0)
    q, e = _quant(x, 1, FP8_MAX, MAX_EXP, CONV, QR.dtype.element_ty)
    q = gl.convert_layout(q, LSR)
    m_s = tm * 128 + gl.arange(0, 128, layout=gl.SliceLayout(1, LSR))
    n_s = tn * 128 + gl.arange(0, 128, layout=gl.SliceLayout(0, LSR))
    gl.store(QR + m_s[:, None].to(gl.int64) * N + n_s[None, :], q)
    m_r = tm * 128 + gl.arange(0, 128, layout=gl.SliceLayout(1, LR))
    _store_scale(ER, m_r, tn, e, N // 128, SLR)
    # ---- column pass (128x1 along M), stored transposed
    xc = src.load(LC).to(gl.float32)
    if ZM:
        xc = gl.where(rv_c[:, None], xc, 0.0)
    qc, ec = _quant(xc, 0, FP8_MAX, MAX_EXP, CONV, QT.dtype.element_ty)
    qc = gl.convert_layout(qc, LSC)
    m_s2 = tm * 128 + gl.arange(0, 128, layout=gl.SliceLayout(1, LSC))
    n_s2 = tn * 128 + gl.arange(0, 128, layout=gl.SliceLayout(0, LSC))
    gl.store(QT + n_s2[None, :].to(gl.int64) * M + m_s2[:, None], qc)
    n_c = tn * 128 + gl.arange(0, 128, layout=gl.SliceLayout(0, LC))
    if CTT:
        gl.store(ET + n_c.to(gl.int64) * (M // 128) + tm, ((ec + 127) & 0xFF).to(gl.uint8))
    else:
        _store_scale(ET, n_c, tm, ec, M // 128, SLT)


@gluon.jit
def _q2_dense_kernel(X, QR, ER, QT, ET, M, N, stride_x, FP8_MAX: gl.constexpr, MAX_EXP: gl.constexpr,
                     CONV: gl.constexpr, SLR: gl.constexpr, SLT: gl.constexpr, NW: gl.constexpr,
                     NPROG: gl.constexpr, NS: gl.constexpr):
    """persistent dense dual quant of X [M, N]: program pid walks tiles pid, pid + NPROG, ... (tokens fastest)."""
    SH: gl.constexpr = gl.PaddedSharedLayout.with_identity_for([[128, 8]], [128, 128], [1, 0])
    pid = gl.program_id(0)
    nbm = M // 128
    ntiles = nbm * (N // 128)
    bufs = gl.allocate_shared_memory(gl.bfloat16, [NS, 128, 128], SH)
    xd = tdm.make_tensor_descriptor(base=X, shape=(M, N), strides=(stride_x, 1), block_shape=(128, 128), layout=SH)
    n_my = (ntiles - pid + NPROG - 1) // NPROG
    for s in gl.static_range(NS - 1):
        t = gl.minimum(pid + s * NPROG, ntiles - 1)
        tdm.async_load(xd, [(t % nbm) * 128, (t // nbm) * 128], bufs.index(s))
    for it in range(0, n_my):
        t = pid + it * NPROG
        tl_ = gl.minimum(t + (NS - 1) * NPROG, ntiles - 1)  # tail iterations re-load a valid tile (never read)
        tdm.async_wait(NS - 2)  # this tile's load (the oldest outstanding) is complete for this wave ...
        gl.barrier()  # ... and for all waves; the slot re-filled below was released by the previous iteration's barrier
        tdm.async_load(xd, [(tl_ % nbm) * 128, (tl_ // nbm) * 128], bufs.index((it + NS - 1) % NS))
        _q2_tile(bufs.index(it % NS), t % nbm, t // nbm, 0, 0, QR, ER, QT, ET, M, N, FP8_MAX, MAX_EXP, CONV, SLR,
                 SLT, 0, 0, NW)
        gl.barrier()
    tdm.async_wait(0)


@gluon.jit
def _q2_gather_kernel(X, ROWMAP, Q, S, QT, ST, T, P, C, stride_x, PREAL, FP8_MAX: gl.constexpr,
                      MAX_EXP: gl.constexpr, SL: gl.constexpr, CT: gl.constexpr, ZM: gl.constexpr, NW: gl.constexpr,
                      NPROG: gl.constexpr, NS: gl.constexpr):
    """persistent MegaMoE dispatch gather + dual quant (FLY): pool row p reads X[ROWMAP[p]] (negative = padding row
    = zeros) for the live pool tiles (< PREAL rows). ZM 0: padding rows gathered from the out-of-bounds row T (TDM
    zero fill); ZM 1: padding rows gathered from row 0 and zeroed in registers."""
    SH: gl.constexpr = gl.PaddedSharedLayout.with_identity_for([[128, 8]], [128, 128], [1, 0])
    IL2: gl.constexpr = gl.BlockedLayout([128 // NW, 1], [1, 32], [NW, 1], [1, 0])
    IL: gl.constexpr = gl.SliceLayout(1, IL2)  # every lane of a warp holds the warp's row ids (TDM gather)
    LR: gl.constexpr = gl.BlockedLayout([1, 128], [32, 1], [NW, 1], [1, 0])
    LC: gl.constexpr = gl.BlockedLayout([128, 1], [1, 32], [1, NW], [0, 1])
    pid = gl.program_id(0)
    nct = C // 128
    npt_real = gl.load(PREAL) // 128
    npt = gl.maximum(npt_real, 1)  # (no live tile: the prologue gathers tile 0, nothing is written)
    ntiles = npt_real * nct
    bufs = gl.allocate_shared_memory(gl.bfloat16, [NS, 128, 128], SH)
    xd = tdm.make_tensor_descriptor(base=X, shape=(T, C), strides=(stride_x, 1), block_shape=(128, 128), layout=SH)
    n_my = (ntiles - pid + NPROG - 1) // NPROG
    for s in gl.static_range(NS - 1):
        t = gl.maximum(gl.minimum(pid + s * NPROG, ntiles - 1), 0)
        r = gl.load(ROWMAP + (t % npt) * 128 + gl.arange(0, 128, layout=IL))
        if ZM:
            r = gl.maximum(r, 0)
        else:
            r = gl.where(r >= 0, r, T)
        tdm.async_gather(tdm.update_tensor_descriptor(xd, add_offsets=[0, (t // npt) * 128]), r.to(gl.int16),
                         bufs.index(s))
    for it in range(0, n_my):
        t = pid + it * NPROG
        tl_ = gl.minimum(t + (NS - 1) * NPROG, ntiles - 1)  # tail iterations re-gather a valid tile (never read)
        rn = gl.load(ROWMAP + (tl_ % npt) * 128 + gl.arange(0, 128, layout=IL))
        if ZM:
            rn = gl.maximum(rn, 0)
        else:
            rn = gl.where(rn >= 0, rn, T)
        pt = t % npt
        rv_r = 0
        rv_c = 0
        if ZM:
            rv_r = gl.load(ROWMAP + pt * 128 + gl.arange(0, 128, layout=gl.SliceLayout(1, LR))) >= 0
            rv_c = gl.load(ROWMAP + pt * 128 + gl.arange(0, 128, layout=gl.SliceLayout(1, LC))) >= 0
        tdm.async_wait(NS - 2)
        gl.barrier()
        tdm.async_gather(tdm.update_tensor_descriptor(xd, add_offsets=[0, (tl_ // npt) * 128]), rn.to(gl.int16),
                         bufs.index((it + NS - 1) % NS))
        _q2_tile(bufs.index(it % NS), pt, t // npt, rv_r, rv_c, Q, S, QT, ST, P, C, FP8_MAX, MAX_EXP, 1, SL, SL,
                 CT, ZM, NW)
        gl.barrier()
    tdm.async_wait(0)


_MARKED = set()


def _mark(kind, *key):
    """PRIMUS_QUANT2_MARK=<file>: append one line per (kind, shape, layout) the first time it runs (in-step check)."""
    p = os.environ.get("PRIMUS_QUANT2_MARK")
    if p and (kind, key) not in _MARKED:
        _MARKED.add((kind, key))
        with open(p, "a") as f:
            f.write(f"{kind} {key}\n")


LAST = None  # last compiled kernel (tests)
_NCU = {}


def _fp8_max_exp(fmax):
    import math
    m, e = math.frexp(fmax)
    assert m == 0.875
    return e - 1


def _ncu(dev):
    n = _NCU.get(dev.index)
    if n is None:
        n = _NCU[dev.index] = torch.cuda.get_device_properties(dev).multi_processor_count
    return n


def _scale_buf(R, K, sl, dev):
    if sl == 2:
        return torch.empty((R // 128, K // 32 * 128), dtype=torch.uint8, device=dev)
    return torch.empty((R, K // 128), dtype=torch.uint8, device=dev)


def q2_dense(x, dtype=torch.float8_e4m3fn, row_layout="g90", t_layout="g90", out=None, nw=4, ns=2, wpc=3,
             nprog=None):
    """Same outputs as quant_g90.quant_dual_t_e8m0_v2(x, dtype, row_layout=, t_layout=) (with_row, with_t, no sr):
    returns (q_row, None, e8m0_row, q_t, e8m0_t)."""
    assert x.dim() == 2 and x.stride(1) == 1 and x.dtype == torch.bfloat16
    M, N = x.shape
    assert M % 128 == 0 and N % 128 == 0
    slr = 2 if row_layout in ("packed", "dsv4", 2) else 1
    slt = 2 if t_layout in ("packed", "dsv4", 2) else 1
    dev = x.device
    if out is None:
        qr = torch.empty((M, N), dtype=dtype, device=dev)
        er = _scale_buf(M, N, slr, dev)
        qt = torch.empty((N, M), dtype=dtype, device=dev)
        et = _scale_buf(N, M, slt, dev)
    else:
        qr, er, qt, et = out
    nprog = min(nprog or _ncu(dev) * wpc, (M // 128) * (N // 128))
    fmax = float(torch.finfo(dtype).max)
    for t in (x, qr, er, qt, et):
        assert t.data_ptr() != 0
    _mark("dense", M, N, slr, slt, str(dtype))
    global LAST
    LAST = _q2_dense_kernel[(nprog,)](x, qr, er, qt, et, M, N, x.stride(0), FP8_MAX=fmax,
                                      MAX_EXP=_fp8_max_exp(fmax), CONV=0, SLR=slr, SLT=slt, NW=nw, NPROG=nprog,
                                      NS=ns, num_warps=nw)
    return qr, None, er, qt, et


def q2_gather(x, P, rowmap, preal, dtype=torch.float8_e4m3fn, compact_t=False, scale_layout="x4", out=None, nw=4,
              ns=2, wpc=3, nprog=None, zm=0):
    """Same outputs as quant_blk_gfx1250.quant_dual_blk(x, P, rowmap, preal=preal, compact_t=, scale_layout=)
    (live pool tiles only; tiles at or past preal are not written, like the int kernel)."""
    C = x.shape[1]
    T = x.shape[0]
    assert x.dim() == 2 and x.stride(1) == 1 and x.dtype == torch.bfloat16 and P % 128 == 0 and C % 128 == 0
    assert T < 32768 and rowmap.dtype == torch.int32 and preal.dtype == torch.int32
    sl = 2 if scale_layout in ("packed", "dsv4", 2) else 0
    dev = x.device
    if out is None:
        q = torch.empty((P, C), device=dev, dtype=dtype)
        s = (torch.empty((P // 128, C // 32 * 128), device=dev, dtype=torch.uint8) if sl == 2 else
             torch.empty((P, C // 32), device=dev, dtype=torch.uint8))
        qT = torch.empty((C, P), device=dev, dtype=dtype)
        if compact_t:
            sT = torch.empty((C, P // 128), device=dev, dtype=torch.uint8)
        else:
            sT = (torch.empty((C // 128, P // 32 * 128), device=dev, dtype=torch.uint8) if sl == 2 else
                  torch.empty((C, P // 32), device=dev, dtype=torch.uint8))
    else:
        q, s, qT, sT = out
    nprog = nprog or _ncu(dev) * wpc
    fmax = float(torch.finfo(dtype).max)
    for t in (x, rowmap, preal, q, s, qT, sT):
        assert t.data_ptr() != 0
    _mark("gather", T, C, P, sl, int(compact_t), str(dtype))
    global LAST
    LAST = _q2_gather_kernel[(nprog,)](x, rowmap, q, s, qT, sT, T, P, C, x.stride(0), preal, FP8_MAX=fmax,
                                       MAX_EXP=_fp8_max_exp(fmax), SL=sl, CT=int(compact_t), ZM=zm, NW=nw,
                                       NPROG=nprog, NS=ns, num_warps=nw)
    return q, s, qT, sT


# ----------------------------------------------------------------------------------------------------------------
# fp32-input variant (values computed in registers, e.g. the MegaMoE SwiGLU backward): the producer stores its fp32
# tile column-major into LDS (4 consecutive rows per thread -> ds_store_b128), then a column pass (lane = half a
# column, ds_load_b128) and a row pass (lane = half a row, ds_load_b32) quantize it, 8 warps, FLY convention.
@gluon.jit
def _q2f_tile(cbuf, p0, kbq, cq0, ct0, pt, P, CQ, Q, S, QT, ST, MAX_EXP: gl.constexpr, NW: gl.constexpr):
    """cbuf: [128 rows, 128 cols] fp32 in LDS (column-major). Rows are pool rows p0 + i; row-major copy Q [P, CQ]
    columns cq0 + j with x4 scale block kbq; transposed copy QT [*, P] rows ct0 + j with compact ST [*, P/128]."""
    gl.static_assert(NW == 8)
    # lane = half a row: lane bit 0 -> column bit 2 so the two halves' b32 reads hit different LDS banks
    LR8: gl.constexpr = gl.DistributedLinearLayout(
        reg_bases=[[0, 1], [0, 2], [0, 8], [0, 16], [0, 32], [0, 64]],
        lane_bases=[[0, 4], [1, 0], [2, 0], [4, 0], [8, 0]], warp_bases=[[16, 0], [32, 0], [64, 0]], block_bases=[],
        shape=[128, 128])
    LC8: gl.constexpr = gl.BlockedLayout([64, 1], [2, 16], [1, 8], [0, 1])  # lane = half a column
    LSR8: gl.constexpr = gl.DistributedLinearLayout(
        reg_bases=[[0, 1], [0, 2], [0, 4], [0, 8], [4, 0], [8, 0]],
        lane_bases=[[0, 16], [0, 32], [0, 64], [1, 0], [2, 0]], warp_bases=[[16, 0], [32, 0], [64, 0]], block_bases=[],
        shape=[128, 128])
    LSC8: gl.constexpr = gl.DistributedLinearLayout(
        reg_bases=[[1, 0], [2, 0], [4, 0], [8, 0], [0, 4], [0, 8]],
        lane_bases=[[16, 0], [32, 0], [64, 0], [0, 1], [0, 2]], warp_bases=[[0, 16], [0, 32], [0, 64]], block_bases=[],
        shape=[128, 128])
    # ---- column pass (128x1 along the pool rows) -> QT rows, compact ST
    xc = cbuf.load(LC8)
    ec = _e_fly(gl.max(gl.abs(xc), axis=0), MAX_EXP)
    qc = gl.convert_layout((xc * _p2neg_fly(ec)[None, :]).to(QT.dtype.element_ty), LSC8)
    r_s = p0 + gl.arange(0, 128, layout=gl.SliceLayout(1, LSC8))
    c_s = ct0 + gl.arange(0, 128, layout=gl.SliceLayout(0, LSC8))
    gl.store(QT + c_s[None, :].to(gl.int64) * P + r_s[:, None], qc)
    c_c = ct0 + gl.arange(0, 128, layout=gl.SliceLayout(0, LC8))
    gl.store(ST + c_c.to(gl.int64) * (P // 128) + pt, ((ec + 127) & 0xFF).to(gl.uint8))
    # ---- row pass (1x128 along the channels) -> Q, x4 S
    xr = cbuf.load(LR8)
    er = _e_fly(gl.max(gl.abs(xr), axis=1), MAX_EXP)
    qr = gl.convert_layout((xr * _p2neg_fly(er)[:, None]).to(Q.dtype.element_ty), LSR8)
    r_s2 = p0 + gl.arange(0, 128, layout=gl.SliceLayout(1, LSR8))
    c_s2 = cq0 + gl.arange(0, 128, layout=gl.SliceLayout(0, LSR8))
    gl.store(Q + r_s2[:, None].to(gl.int64) * CQ + c_s2[None, :], qr)
    r_r = p0 + gl.arange(0, 128, layout=gl.SliceLayout(1, LR8))
    gl.store(S.to(gl.pointer_type(gl.int32)) + r_r.to(gl.int64) * (CQ // 128) + kbq, ((er + 127) & 0xFF) * 0x01010101)


@gluon.jit
def _sigmoid(x):
    return 1 / (1 + gl.exp(-x))  # tl.sigmoid's expression (as quant_blk_gluon_gfx1250._sigmoid)


@gluon.jit
def _swb_q2_kernel(DA, L1, W, Q, S, QT, ST, GW, P, I, PREAL, ALPHA: gl.constexpr, OFF: gl.constexpr,
                   GLO: gl.constexpr, GHI: gl.constexpr, UCL: gl.constexpr, NPROG: gl.constexpr, NW: gl.constexpr):
    """quant_blk_gluon_gfx1250._swb_tdm_kernel (same loads, math, gw sums in the same layout) with the two dual
    quants on the quant2 fp32 core (column-major LDS tile, lane-per-half-row / half-column amax)."""
    L: gl.constexpr = gl.BlockedLayout(size_per_thread=[4, 16], threads_per_warp=[32, 1], warps_per_cta=[1, NW],
                                       order=[1, 0])
    SH: gl.constexpr = gl.PaddedSharedLayout.with_identity_for([[128, 8]], [128, 128], [1, 0])
    SC: gl.constexpr = gl.PaddedSharedLayout.with_identity_for([[128, 4]], [128, 128], [0, 1])  # column-major fp32
    pid = gl.program_id(0)
    nct = I // 128
    ntiles = (gl.load(PREAL) // 128) * nct
    gbuf = gl.allocate_shared_memory(gl.bfloat16, [2, 128, 128], SH)
    ubuf = gl.allocate_shared_memory(gl.bfloat16, [2, 128, 128], SH)
    dbuf = gl.allocate_shared_memory(gl.bfloat16, [2, 128, 128], SH)
    cbuf = gl.allocate_shared_memory(gl.float32, [128, 128], SC)
    l1d = tdm.make_tensor_descriptor(base=L1, shape=(P, 2 * I), strides=(2 * I, 1), block_shape=(128, 128), layout=SH)
    dad = tdm.make_tensor_descriptor(base=DA, shape=(P, I), strides=(I, 1), block_shape=(128, 128), layout=SH)
    n_my = (ntiles - pid + NPROG - 1) // NPROG
    t0 = gl.minimum(pid, ntiles - 1)
    tdm.async_load(l1d, [(t0 // nct) * 128, (t0 % nct) * 128], gbuf.index(0))
    tdm.async_load(l1d, [(t0 // nct) * 128, I + (t0 % nct) * 128], ubuf.index(0))
    tdm.async_load(dad, [(t0 // nct) * 128, (t0 % nct) * 128], dbuf.index(0))
    for it in range(0, n_my):
        slot = it % 2
        t = pid + it * NPROG
        tn = gl.minimum(t + NPROG, ntiles - 1)  # the last iteration re-loads a valid tile (never read)
        tdm.async_wait(0)
        gl.barrier()
        tdm.async_load(l1d, [(tn // nct) * 128, (tn % nct) * 128], gbuf.index(1 - slot))
        tdm.async_load(l1d, [(tn // nct) * 128, I + (tn % nct) * 128], ubuf.index(1 - slot))
        tdm.async_load(dad, [(tn // nct) * 128, (tn % nct) * 128], dbuf.index(1 - slot))
        pt = t // nct
        pc = t % nct
        p = pt * 128 + gl.arange(0, 128, layout=gl.SliceLayout(1, L))
        g0 = gbuf.index(slot).load(L).to(gl.float32)
        u0 = ubuf.index(slot).load(L).to(gl.float32)
        d = dbuf.index(slot).load(L).to(gl.float32)
        w = gl.load(W + p)
        g = gl.minimum(gl.maximum(g0, GLO), GHI)
        u = gl.minimum(gl.maximum(u0, -UCL), UCL)
        sg = _sigmoid(ALPHA * g)
        silu = g * sg
        up = u + OFF
        gl.store(GW + p.to(gl.int64) * (I // 128) + pc, gl.sum(d * silu * up, axis=1))
        dw = d * w[:, None]
        gin = (g0 >= GLO) & (g0 <= GHI)
        uin = (u0 >= -UCL) & (u0 <= UCL)
        d_up = gl.where(uin, dw * silu, 0.0)
        d_gate = gl.where(gin, dw * up * (sg + ALPHA * g * sg * (1.0 - sg)), 0.0)
        cbuf.store(d_gate)
        gl.barrier()
        _q2f_tile(cbuf, pt * 128, pc, pc * 128, pc * 128, pt, P, 2 * I, Q, S, QT, ST, 8, NW)
        gl.barrier()
        cbuf.store(d_up)
        gl.barrier()
        _q2f_tile(cbuf, pt * 128, pc + I // 128, I + pc * 128, I + pc * 128, pt, P, 2 * I, Q, S, QT, ST, 8, NW)
        gl.barrier()  # every wave is done with this stage and cbuf before the next iteration overwrites them
    tdm.async_wait(0)


def swb_q2(d_act, l1, w, q, s, qT, sT, gw, preal, alpha, off, glo, ghi, ucl, nprog=None):
    """quant_blk_gluon_gfx1250.swiglu_bwd_dual_tdm's contract (outputs allocated by swiglu_bwd_dual_blk)."""
    P, I = d_act.shape
    assert l1.shape == (P, 2 * I) and P % 128 == 0 and I % 128 == 0
    for t in (d_act, l1, w, q, s, qT, sT, gw, preal):
        assert t.is_cuda and t.is_contiguous() and t.data_ptr() != 0
    nprog = nprog or _ncu(d_act.device)
    _mark("swb", P, I)
    global LAST
    LAST = _swb_q2_kernel[(nprog,)](d_act, l1, w, q, s, qT, sT, gw, P, I, preal, ALPHA=alpha, OFF=off, GLO=glo,
                                    GHI=ghi, UCL=ucl, NPROG=nprog, NW=8, num_warps=8)

