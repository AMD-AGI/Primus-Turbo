###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""DeepSeek-V3/V4 FP8 recipe quantization for gfx1250: E4M3 with power-of-two (UE8M0) scales per 1x128
activation tile and per 128x128 weight block.

Scales are emitted as the MX planes the gfx1250 WMMA-scale GEMMs read: one E8M0 byte per 32 K-elements, the
block-128 exponent repeated 4x (written as one int32), so the GEMM applies them in hardware.

  e = ceil(log2(amax / 448)), q = x * 2^-e (exact power-of-two scaling, one RNE rounding to E4M3),
  byte = e + 127; an all-zero block gets e = 0.
The exponent comes from the amax bits (no division; quant_epi_gfx1250), bit-identical to the former
``amax / FP8_MAX`` form. ``scale_layout`` = "x4" (default, flygg MX planes [R, K/32]) or "packed" (dsv4up
pack_scale BK256 layout [R/128, K/32*128]); same bytes, different order.
"""

import os

import torch
import triton
import triton.language as tl

from primus_turbo.triton.quantization.quant_epi_gfx1250 import (
    SL_PACKED,
    pow2_neg_fly,
    scale_layout_id,
    store_scale,
    ue8m0_exp_fly,
)

FP8_MAX = 448.0
MX = 32
# fuse-moe: Gluon TDM-streamed SwiGLU backward (quant_blk_gluon_gfx1250; needs Triton main with gluon cdna5)
_SWB_TDM = os.environ.get("PRIMUS_FUSE_MOE_SWB_TDM", "0") == "1"
# quant2: MegaMoE dispatch gather + dual quant on the Gluon quant2 core (quant2_gfx1250.q2_gather; byte-identical)
_Q2_DQ = os.environ.get("PRIMUS_QUANT2_DQ", "0") == "1"
# quant2: the SWB_TDM kernel's two dual quants on the quant2 fp32 core (quant2_gfx1250.swb_q2; byte-identical)
_Q2_SWB = os.environ.get("PRIMUS_QUANT2_SWB", "0") == "1"


@triton.jit
def _ue8m0_exp(amax, FP8_MAX: tl.constexpr):
    """smallest e with amax <= FP8_MAX * 2^e, 0 for amax == 0, from the amax bits (no division)."""
    if FP8_MAX == 448.0:
        e = ue8m0_exp_fly(amax, 8)
    else:
        tl.static_assert(FP8_MAX == 57344.0)
        e = ue8m0_exp_fly(amax, 15)
    return e


def _scale_alloc(R: int, K: int, sl: int, dev):
    """E8M0 bytes of a 1x128-tiled [R, K] operand in layout sl (same byte count for both)."""
    if sl == SL_PACKED:
        assert R % 128 == 0
        return torch.empty((R // 128, K // MX * 128), device=dev, dtype=torch.uint8)
    return torch.empty((R, K // MX), device=dev, dtype=torch.uint8)


@triton.jit
def _act_1x128_kernel(X, Q, S, M, K, stride_x, stride_q, R: tl.constexpr, FP8_MAX: tl.constexpr,
                      SL: tl.constexpr = 0):
    """grid (cdiv(M, R), K // 128): R rows x one 128-wide tile; scales [M, K/32] uint8 viewed as int32 [M, K/128]."""
    pid_m = tl.program_id(0)
    kb = tl.program_id(1)
    rows = pid_m * R + tl.arange(0, R)
    cols = kb * 128 + tl.arange(0, 128)
    rm = rows < M
    x = tl.load(X + rows[:, None].to(tl.int64) * stride_x + cols[None, :], mask=rm[:, None], other=0.0).to(tl.float32)
    amax = tl.max(tl.abs(x), axis=1)
    e = _ue8m0_exp(amax, FP8_MAX)
    q = (x * pow2_neg_fly(e)[:, None]).to(Q.dtype.element_ty)  # 2^-e
    tl.store(Q + rows[:, None].to(tl.int64) * stride_q + cols[None, :], q, mask=rm[:, None])
    store_scale(S, rows, kb, e, M, K // 128, SL, mask=rm)


def quant_act_1x128(x: torch.Tensor, dtype=torch.float8_e4m3fn, rows_per_prog: int = 16, scale_layout="x4"):
    """x [M, K] bf16 (K % 128 == 0) -> (q [M, K] fp8, s [M, K/32] uint8 E8M0 repeated per 128)."""
    assert x.dim() == 2 and x.shape[1] % 128 == 0 and x.stride(1) == 1
    M, K = x.shape
    sl = scale_layout_id(scale_layout)
    q = torch.empty((M, K), device=x.device, dtype=dtype)
    s = _scale_alloc(M, K, sl, x.device)
    _act_1x128_kernel[(triton.cdiv(M, rows_per_prog), K // 128)](
        x, q, s, M, K, x.stride(0), q.stride(0), R=rows_per_prog, FP8_MAX=torch.finfo(dtype).max, SL=sl,
        num_warps=4
    )
    return q, s


@triton.jit
def _w_128x128_kernel(W, Q, S, QT, ST, N, K, FP8_MAX: tl.constexpr, TRANS: tl.constexpr):
    """grid (N // 128, K // 128, G): one 128x128 block of expert g."""
    nb = tl.program_id(0)
    kb = tl.program_id(1)
    g = tl.program_id(2).to(tl.int64)
    r = tl.arange(0, 128)
    base = g * N * K
    w = tl.load(W + base + (nb * 128 + r)[:, None].to(tl.int64) * K + kb * 128 + r[None, :]).to(tl.float32)
    amax = tl.max(tl.max(tl.abs(w), axis=1), axis=0)
    e = _ue8m0_exp(amax, FP8_MAX)
    q = (w * pow2_neg_fly(e)).to(Q.dtype.element_ty)
    tl.store(Q + base + (nb * 128 + r)[:, None].to(tl.int64) * K + kb * 128 + r[None, :], q)
    word = ((e + 127) & 0xFF) * 0x01010101
    sv = tl.zeros((128,), dtype=tl.int32) + word
    # S [G, N, K/32] as int32 [G, N, K/128]
    tl.store(S.to(tl.pointer_type(tl.int32)) + g * N * (K // 128) + (nb * 128 + r).to(tl.int64) * (K // 128) + kb, sv)
    if TRANS:
        # QT [G, K, N] (= this block transposed), ST [G, K, N/32] as int32 [G, K, N/128]
        tl.store(QT + base + (kb * 128 + r)[:, None].to(tl.int64) * N + nb * 128 + r[None, :], tl.trans(q))
        tl.store(ST.to(tl.pointer_type(tl.int32)) + g * K * (N // 128) + (kb * 128 + r).to(tl.int64) * (N // 128) + nb, sv)


def quant_w_128x128(w: torch.Tensor, dtype=torch.float8_e4m3fn, transposed: bool = False, out=None):
    """w [G, N, K] bf16 (N, K % 128 == 0) -> (q [G, N, K], s [G, N, K/32]) and, with ``transposed``, also
    (qT [G, K, N], sT [G, K, N/32]) for the NT form of the dgrad (a 128x128 block's scale is its transpose's)."""
    if w.dim() == 2:
        w = w.unsqueeze(0)
    G, N, K = w.shape
    assert N % 128 == 0 and K % 128 == 0 and w.is_contiguous()
    if out is not None:  # fixed buffers (e.g. a per-layer cache refreshed after every optimizer step)
        q, s, qT, sT = out if transposed else (*out, None, None)
        qT = q if qT is None else qT
        sT = s if sT is None else sT
    else:
        q = torch.empty((G, N, K), device=w.device, dtype=dtype)
        s = torch.empty((G, N, K // MX), device=w.device, dtype=torch.uint8)
        qT = torch.empty((G, K, N), device=w.device, dtype=dtype) if transposed else q
        sT = torch.empty((G, K, N // MX), device=w.device, dtype=torch.uint8) if transposed else s
    _w_128x128_kernel[(N // 128, K // 128, G)](w, q, s, qT, sT, N, K, FP8_MAX=torch.finfo(dtype).max,
                                               TRANS=int(transposed), num_warps=8)
    return (q, s, qT, sT) if transposed else (q, s)


def dequant(q: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """fp32 reference: q [..., R, K] fp8 with s [..., R, K/32] E8M0 -> q * 2^(s - 127) per 32-block."""
    e = s.to(torch.int32) - 127
    f = torch.pow(2.0, e.float()).repeat_interleave(MX, dim=-1)
    return q.float() * f


@triton.jit
def _dual_1x128_128x1_kernel(X, ROWMAP, Q, S, QT, ST, P, C, stride_x, PREAL, HAS_MAP: tl.constexpr, BT: tl.constexpr,
                             BC: tl.constexpr, FP8_MAX: tl.constexpr, HAS_PREAL: tl.constexpr, TMODE: tl.constexpr = 1,
                             CT: tl.constexpr = 0, SL: tl.constexpr = 0):
    """grid (P // BT, C // BC), BT == BC == 128: one 128 pool rows x 128 channels tile quantized both ways.
    rows: pool row p reads X[ROWMAP[p]] (HAS_MAP; a negative entry is a padding row -> zeros) or X[p].
      Q  [P, C]  fp8, S  [P, C/32] (1x128 tiles, exponent x4 as int32 per 128 channels)
      QT [C, P]  fp8, ST [C, P/32] (128x1 tiles along the pool rows, exponent x4 per 128 rows)"""
    pt = tl.program_id(0)
    pc = tl.program_id(1)
    if HAS_PREAL:
        # fixed (worst-case) pool for CUDA graphs: tiles past the real pool rows (device count) do nothing
        if pt * BT >= tl.load(PREAL):
            return
    p = pt * BT + tl.arange(0, BT)
    c = pc * BC + tl.arange(0, BC)
    if HAS_MAP:
        src = tl.load(ROWMAP + p)
        valid = src >= 0
        src = tl.where(valid, src, 0)
    else:
        src = p
        valid = p < P
    x = tl.load(X + src[:, None].to(tl.int64) * stride_x + c[None, :], mask=valid[:, None], other=0.0).to(tl.float32)
    _quant_tile_dual(x, p, c, pc, c, pt, P, C, Q, S, QT, ST, FP8_MAX, TMODE, CT, SL)


def _dual_alloc(P: int, C: int, dtype, sl: int, compact_t: bool, dev):
    q = torch.empty((P, C), device=dev, dtype=dtype)
    s = _scale_alloc(P, C, sl, dev)
    qT = torch.empty((C, P), device=dev, dtype=dtype)
    sT = torch.empty((C, P // 128), device=dev, dtype=torch.uint8) if compact_t else _scale_alloc(C, P, sl, dev)
    return q, s, qT, sT


def quant_dual_blk(x: torch.Tensor, P: int, rowmap: torch.Tensor = None, dtype=torch.float8_e4m3fn, nw: int = 8,
                   preal: torch.Tensor = None, tmode: int = 1, compact_t: bool = False, scale_layout="x4"):
    """x [rows, C] bf16 -> pool-ordered (q [P, C], s [P, C/32], qT [C, P], sT [C, P/32]); P % 128 == 0.
    ``rowmap`` int32 [P]: source row of each pool row (negative = padding, quantized as zeros): the dispatch
    (permute) gather is fused, no permuted bf16 copy is written.
    ``scale_layout="packed"``: s [P/128, C/32*128] and sT [C/128, P/32*128] in the dsv4up packed layout."""
    C = x.shape[1]
    assert P % 128 == 0 and C % 128 == 0 and x.stride(1) == 1
    sl = scale_layout_id(scale_layout)
    if (_Q2_DQ and rowmap is not None and preal is not None and sl in (0, 2) and x.dtype == torch.bfloat16
            and x.shape[0] < 32768 and rowmap.dtype == torch.int32 and preal.dtype == torch.int32
            and rowmap.is_contiguous() and preal.is_contiguous() and (x.stride(0) * 2) % 16 == 0
            and x.data_ptr() % 16 == 0):
        from primus_turbo.triton.quantization.quant2_gfx1250 import q2_gather

        return q2_gather(x, P, rowmap, preal, dtype=dtype, compact_t=compact_t, scale_layout=sl)
    q, s, qT, sT = _dual_alloc(P, C, dtype, sl, compact_t, x.device)
    _dual_1x128_128x1_kernel[(P // 128, C // 128)](
        x, rowmap if rowmap is not None else x, q, s, qT, sT, P, C, x.stride(0), preal if preal is not None else x,
        HAS_MAP=int(rowmap is not None), BT=128, BC=128, FP8_MAX=torch.finfo(dtype).max,
        HAS_PREAL=int(preal is not None), TMODE=tmode, CT=int(compact_t), SL=sl, num_warps=nw)
    return q, s, qT, sT


@triton.jit
def _quant_tile_dual(v, p, cq, pcq, c, pt, P, C, Q, S, QT, ST, FP8_MAX: tl.constexpr, TMODE: tl.constexpr = 1,
                     CT: tl.constexpr = 0, SL: tl.constexpr = 0):
    """quantize one [128 pool rows, 128 cols] fp32 tile both ways (see _dual_1x128_128x1_kernel).
    cq / pcq: columns / 128-block of the tile in the row-major copy Q [P, C] (and S);
    c: the same channels as rows of the transposed copy QT [C, P] (and ST). They differ only for the
    [gate128|up128]-interleaved fc1 layout, where Q follows the interleaved W1 K order and QT stays canonical."""
    av = tl.abs(v)
    er = _ue8m0_exp(tl.max(av, axis=1), FP8_MAX)
    qr = (v * pow2_neg_fly(er)[:, None]).to(Q.dtype.element_ty)
    tl.store(Q + p[:, None].to(tl.int64) * C + cq[None, :], qr)
    store_scale(S, p, pcq, er, P, C // 128, SL)
    ec = _ue8m0_exp(tl.max(av, axis=0), FP8_MAX)
    qc = (v * pow2_neg_fly(ec)[None, :]).to(QT.dtype.element_ty)
    if TMODE == 0:
        tl.store(QT + c[:, None].to(tl.int64) * P + p[None, :], tl.trans(qc))
    elif TMODE == 1:
        # 4 consecutive pool rows of a column packed into one int32 before the transpose: 4-byte stores
        u = tl.reshape(qc.to(tl.uint8, bitcast=True).to(tl.int32), (32, 4, 128))
        w = tl.sum(u << (tl.arange(0, 4) * 8)[None, :, None], axis=1)  # [32, 128]
        pw = pt * 32 + tl.arange(0, 32)
        tl.store(QT.to(tl.pointer_type(tl.int32)) + c[:, None].to(tl.int64) * (P // 4) + pw[None, :], tl.trans(w))
    else:
        tl.store(QT + c[None, :].to(tl.int64) * P + p[:, None], qc)
    if CT:  # compact: one byte per 128 pool rows, ST [C, P/128] (the persistent wgrad repeats it in-kernel)
        tl.store(ST + c.to(tl.int64) * (P // 128) + pt, ((ec + 127) & 0xFF).to(tl.uint8))
    else:
        store_scale(ST, c, pt, ec, C, P // 128, SL)


@triton.jit
def _swiglu_fwd_dual_kernel(L1, W, Q, S, QT, ST, P, I, PREAL, ALPHA: tl.constexpr, OFF: tl.constexpr,
                            GLO: tl.constexpr, GHI: tl.constexpr, UCL: tl.constexpr, FP8_MAX: tl.constexpr,
                            HAS_PREAL: tl.constexpr, CT: tl.constexpr = 0, SL: tl.constexpr = 0,
                            ILV: tl.constexpr = 0):
    """act_w[p, c] = w[p] * g * sigmoid(alpha g) * (u + off), g/u the clamped gate/up halves of l1 [P, 2I];
    quantized 1x128 (fc2 A operand) and 128x1 transposed (dW2 B operand). grid (P // 128, I // 128).
    ILV: l1 columns interleaved per 128 ([gate blk j | up blk j] at 256 j), the fc1-epilogue W1 order."""
    pt = tl.program_id(0)
    pc = tl.program_id(1)
    if HAS_PREAL:
        if pt * 128 >= tl.load(PREAL):
            return
    p = pt * 128 + tl.arange(0, 128)
    c = pc * 128 + tl.arange(0, 128)
    rowp = L1 + p[:, None].to(tl.int64) * (2 * I)
    if ILV:
        gc = pc * 256 + tl.arange(0, 128)
        uc = gc + 128
    else:
        gc = c
        uc = I + c
    g = tl.load(rowp + gc[None, :]).to(tl.float32)
    u = tl.load(rowp + uc[None, :]).to(tl.float32)
    w = tl.load(W + p)
    act = swiglu_act_w(g, u, w, ALPHA, OFF, GLO, GHI, UCL)
    _quant_tile_dual(act, p, c, pc, c, pt, P, I, Q, S, QT, ST, FP8_MAX, 1, CT, SL)


@triton.jit
def swiglu_act_w(g, u, w, ALPHA: tl.constexpr, OFF: tl.constexpr, GLO: tl.constexpr, GHI: tl.constexpr,
                 UCL: tl.constexpr):
    """clamped SwiGLU x routing weight on fp32 tiles g, u [BR, BC] (bf16-exact values), w [BR]; the one
    expression shared by the standalone kernel and the fc1 GEMM epilogue (keeps them bit-identical)."""
    g = tl.minimum(tl.maximum(g, GLO), GHI)
    u = tl.minimum(tl.maximum(u, -UCL), UCL)
    return g * tl.sigmoid(ALPHA * g) * (u + OFF) * w[:, None]


def swiglu_fwd_dual_blk(l1: torch.Tensor, w: torch.Tensor, activation=None, dtype=torch.float8_e4m3fn, preal=None,
                        compact_t: bool = False, scale_layout="x4", interleaved: bool = False):
    """l1 [P, 2I] bf16 (pool order), w [P] fp32 routing weight per pool row (0 on padding rows) ->
    (q [P, I], s [P, I/32], qT [I, P], sT [I, P/32]) of the weighted activation.
    ``interleaved``: l1 columns in the [gate128|up128] order of an interleaved W1 (see interleave_w1)."""
    from primus_turbo.flydsl.utils.glu_activation import activation_constexpr
    alpha, off, glo, ghi, ucl = activation_constexpr(activation)
    P, I2 = l1.shape
    I = I2 // 2
    sl = scale_layout_id(scale_layout)
    q, s, qT, sT = _dual_alloc(P, I, dtype, sl, compact_t, l1.device)
    _swiglu_fwd_dual_kernel[(P // 128, I // 128)](l1, w, q, s, qT, sT, P, I, preal if preal is not None else w,
                                                  ALPHA=alpha, OFF=off, GLO=glo, GHI=ghi, UCL=ucl,
                                                  FP8_MAX=torch.finfo(dtype).max, HAS_PREAL=int(preal is not None),
                                                  CT=int(compact_t), SL=sl, ILV=int(interleaved), num_warps=8)
    return q, s, qT, sT


@triton.jit
def _swiglu_bwd_dual_kernel(DA, L1, W, Q, S, QT, ST, GW, P, I, PREAL, ALPHA: tl.constexpr, OFF: tl.constexpr,
                            GLO: tl.constexpr, GHI: tl.constexpr, UCL: tl.constexpr, FP8_MAX: tl.constexpr,
                            HAS_PREAL: tl.constexpr, CT: tl.constexpr = 0, SL: tl.constexpr = 0,
                            ILV: tl.constexpr = 0):
    """d act_w [P, I] -> grad_l1 [P, 2I] (gate | up), quantized 1x128 (L1 dgrad A) and 128x1 transposed (dW1 A),
    plus per-(row, 128-col block) partials of d w = sum_c d act_w * act. grid (P // 128, I // 128).
    ILV: l1 is read and the row-major grad_l1 copy is written in the interleaved [gate128|up128] order (the K order
    of the interleaved W1^T); the transposed copy keeps canonical rows (gate c, up I + c) so dW1 stays canonical."""
    pt = tl.program_id(0)
    pc = tl.program_id(1)
    if HAS_PREAL:
        if pt * 128 >= tl.load(PREAL):
            return
    p = pt * 128 + tl.arange(0, 128)
    c = pc * 128 + tl.arange(0, 128)
    rowp = L1 + p[:, None].to(tl.int64) * (2 * I)
    if ILV:
        gc = pc * 256 + tl.arange(0, 128)
        uc = gc + 128
        gb = 2 * pc
        ub = 2 * pc + 1
    else:
        gc = c
        uc = I + c
        gb = pc
        ub = pc + I // 128
    g0 = tl.load(rowp + gc[None, :]).to(tl.float32)
    u0 = tl.load(rowp + uc[None, :]).to(tl.float32)
    d = tl.load(DA + p[:, None].to(tl.int64) * I + c[None, :]).to(tl.float32)
    w = tl.load(W + p)
    g = tl.minimum(tl.maximum(g0, GLO), GHI)
    u = tl.minimum(tl.maximum(u0, -UCL), UCL)
    sg = tl.sigmoid(ALPHA * g)
    silu = g * sg
    up = u + OFF
    tl.store(GW + p.to(tl.int64) * (I // 128) + pc, tl.sum(d * silu * up, axis=1))
    dw = d * w[:, None]
    # clamp gradient passes on the bounds (the kernels' convention)
    gin = (g0 >= GLO) & (g0 <= GHI)
    uin = (u0 >= -UCL) & (u0 <= UCL)
    d_up = tl.where(uin, dw * silu, 0.0)
    d_gate = tl.where(gin, dw * up * (sg + ALPHA * g * sg * (1.0 - sg)), 0.0)
    _quant_tile_dual(d_gate, p, gc, gb, c, pt, P, 2 * I, Q, S, QT, ST, FP8_MAX, 1, CT, SL)
    _quant_tile_dual(d_up, p, uc, ub, c + I, pt, P, 2 * I, Q, S, QT, ST, FP8_MAX, 1, CT, SL)


def swiglu_bwd_dual_blk(d_act: torch.Tensor, l1: torch.Tensor, w: torch.Tensor, activation=None,
                        dtype=torch.float8_e4m3fn, preal=None, compact_t: bool = False, scale_layout="x4",
                        interleaved: bool = False, raw_dw: bool = False):
    """-> (q [P, 2I], s [P, 2I/32], qT [2I, P], sT [2I, P/32]) of grad_l1, and d w [P] (fp32).
    ``interleaved``: l1 and q in the [gate128|up128] column order; qT/sT rows stay canonical."""
    from primus_turbo.flydsl.utils.glu_activation import activation_constexpr
    alpha, off, glo, ghi, ucl = activation_constexpr(activation)
    P, I = d_act.shape
    dev = d_act.device
    sl = scale_layout_id(scale_layout)
    q, s, qT, sT = _dual_alloc(P, 2 * I, dtype, sl, compact_t, dev)
    # rows past the real pool keep zero partials (their d w is never read)
    # raw_dw (fuse track, PRIMUS_FUSE_MOE_BWD): return the per-(row, 128-col block) partials [P, I/128] unsummed and
    # unzeroed (rows past the real pool hold garbage; route_dw_partials reads only routed rows)
    gw = (torch.empty if raw_dw else torch.zeros)((P, I // 128), device=dev, dtype=torch.float32)
    if (_SWB_TDM and preal is not None and compact_t and sl == 0 and not interleaved and dtype == torch.float8_e4m3fn
            and d_act.dtype == torch.bfloat16 and l1.dtype == torch.bfloat16 and w.dtype == torch.float32):
        # fuse-moe (PRIMUS_FUSE_MOE_SWB_TDM=1): Gluon TDM-streamed version, byte-identical outputs
        if _Q2_SWB and preal.dtype == torch.int32:
            from primus_turbo.triton.quantization.quant2_gfx1250 import swb_q2 as swiglu_bwd_dual_tdm
        else:
            from primus_turbo.triton.quantization.quant_blk_gluon_gfx1250 import swiglu_bwd_dual_tdm

        swiglu_bwd_dual_tdm(d_act.contiguous(), l1.contiguous(), w.contiguous(), q, s, qT, sT, gw, preal,
                            alpha, off, glo, ghi, ucl)
        return q, s, qT, sT, (gw if raw_dw else gw.sum(1))
    _swiglu_bwd_dual_kernel[(P // 128, I // 128)](d_act, l1, w, q, s, qT, sT, gw, P, I,
                                                  preal if preal is not None else w, ALPHA=alpha, OFF=off, GLO=glo,
                                                  GHI=ghi, UCL=ucl, FP8_MAX=torch.finfo(dtype).max,
                                                  HAS_PREAL=int(preal is not None), CT=int(compact_t), SL=sl,
                                                  ILV=int(interleaved), num_warps=8)
    return q, s, qT, sT, (gw if raw_dw else gw.sum(1))


def interleave_perm(n2: int, dev=None) -> torch.Tensor:
    """row order of the [gate128|up128]-interleaved W1 [2I, H]: new row 256 j + i <- gate row 128 j + i,
    new row 256 j + 128 + i <- up row I + 128 j + i."""
    I = n2 // 2
    assert I % 128 == 0
    j = torch.arange(I // 128, device=dev).view(-1, 1, 1)
    h = torch.arange(2, device=dev).view(1, -1, 1)
    i = torch.arange(128, device=dev).view(1, 1, -1)
    return (h * I + j * 128 + i).reshape(-1)


def interleave_w1(w1: torch.Tensor) -> torch.Tensor:
    """[..., 2I, H] (gate rows then up rows) -> interleaved rows (128-row blocks, so 128x128 weight-quant blocks map
    1:1: quantizing the interleaved weight == interleaving the quantized blocks and their scales)."""
    return w1.index_select(-2, interleave_perm(w1.shape[-2], w1.device))


@triton.jit
def _combine_kernel(Y, INV, OUT, H, K: tl.constexpr, BH: tl.constexpr, ADD=None, HAS_ADD: tl.constexpr = 0):
    """out[t] = sum_k Y[INV[t*K + k]] (bf16 rows, fp32 sum). grid (T, H // BH).
    HAS_ADD: out[t] = bf16(fp32(bf16(sum)) + fp32(ADD[t])), i.e. exactly ``combine + ADD`` as two bf16 ops (the
    routed + shared expert add of the MoE layer, fuse track)."""
    t = tl.program_id(0)
    h = tl.program_id(1) * BH + tl.arange(0, BH)
    acc = tl.zeros((BH,), dtype=tl.float32)
    for k in tl.static_range(K):
        r = tl.load(INV + t * K + k).to(tl.int64)
        acc += tl.load(Y + r * H + h).to(tl.float32)
    o = acc.to(OUT.dtype.element_ty)
    if HAS_ADD:
        o = (o.to(tl.float32) + tl.load(ADD + t.to(tl.int64) * H + h).to(tl.float32)).to(OUT.dtype.element_ty)
    tl.store(OUT + t.to(tl.int64) * H + h, o)


def combine_rows(y: torch.Tensor, inv: torch.Tensor, T: int, K: int, bh: int = 1024, add: torch.Tensor = None):
    """y [P, H] pool rows (routing weight already applied), inv int32 [T*K] pool row of route (t, k) -> [T, H].
    ``add`` [T, H] (same dtype, contiguous): fused ``+ add`` (bit-identical to a separate torch add)."""
    H = y.shape[1]
    out = torch.empty((T, H), device=y.device, dtype=y.dtype)
    if add is not None:
        assert add.shape == (T, H) and add.dtype == y.dtype and add.is_contiguous() and add.data_ptr() != 0
    _combine_kernel[(T, H // bh)](y, inv, out, H, K=K, BH=bh, ADD=add if add is not None else y,
                                  HAS_ADD=int(add is not None), num_warps=4)
    return out


@triton.jit
def _route_count_kernel(IDX, CNT, ROWMAP, RW, R, P, E: tl.constexpr, BLOCK: tl.constexpr, PB: tl.constexpr):
    """grid (cdiv(R, BLOCK),): per-chunk expert histogram CNT [nchunk, E]; also resets the padding defaults of
    the pool tables (ROWMAP -1, RW 0) over this program's share of P."""
    ch = tl.program_id(0)
    e_ar = tl.arange(0, E)
    lr = ch * BLOCK + tl.arange(0, BLOCK)
    idx = tl.load(IDX + lr, mask=lr < R, other=-1).to(tl.int32)
    tl.store(CNT + ch * E + e_ar, tl.sum((idx[:, None] == e_ar[None, :]).to(tl.int32), axis=0))
    pr = ch * PB + tl.arange(0, PB)
    tl.store(ROWMAP + pr, tl.full((PB,), -1, tl.int32), mask=pr < P)
    tl.store(RW + pr, tl.zeros((PB,), tl.float32), mask=pr < P)


@triton.jit
def _route_offs_kernel(CNT, BASE, POFFS, NCH, E: tl.constexpr, NCP: tl.constexpr, ALIGN: tl.constexpr):
    """one program: padded expert offsets POFFS [E+1] and each chunk's first pool row per expert BASE [nchunk, E]."""
    e_ar = tl.arange(0, E)
    cr = tl.arange(0, NCP)
    m = (cr < NCH)[:, None]
    c = tl.load(CNT + cr[:, None] * E + e_ar[None, :], mask=m, other=0)
    cnt = tl.sum(c, axis=0)
    plen = tl.maximum((cnt + ALIGN - 1) // ALIGN, 1) * ALIGN
    pend = tl.cumsum(plen, 0)
    pstart = pend - plen
    tl.store(POFFS + 1 + e_ar, pend)
    tl.store(POFFS + e_ar, pstart, mask=e_ar == 0)
    excl = tl.cumsum(c, 0) - c
    tl.store(BASE + cr[:, None] * E + e_ar[None, :], pstart[None, :] + excl, mask=m)


@triton.jit
def _route_scatter_kernel(IDX, TW, BASE, ROWMAP, RW, INV, R, E: tl.constexpr, K: tl.constexpr, BLOCK: tl.constexpr):
    """grid (nchunk,): pool row of every route (token order kept within an expert) and the tables at it."""
    ch = tl.program_id(0)
    e_ar = tl.arange(0, E)
    r = ch * BLOCK + tl.arange(0, BLOCK)
    m = r < R
    idx = tl.load(IDX + r, mask=m, other=-1).to(tl.int32)
    oh = (idx[:, None] == e_ar[None, :]).to(tl.int32)
    base = tl.load(BASE + ch * E + e_ar)
    pos = tl.sum(oh * (base[None, :] + tl.cumsum(oh, 0) - 1), axis=1)
    tl.store(ROWMAP + pos, r // K, mask=m)
    tl.store(RW + pos, tl.load(TW + r, mask=m, other=0.0).to(tl.float32), mask=m)
    tl.store(INV + r, pos, mask=m)


def build_routing_blk(topk_idx: torch.Tensor, topk_w: torch.Tensor, num_experts: int, P: int, align: int = 256,
                      block: int = 128):
    """Device-only, capturable pool tables -> (rowmap_tok int32 [P], route_w f32 [P], inv int32 [T*K],
    poffs int32 [E+1], preal = poffs[E:]). Each expert ceil to ``align`` rows, at least ``align``."""
    T, K = topk_idx.shape
    R = T * K
    E = num_experts
    dev = topk_idx.device
    nch = triton.cdiv(R, block)
    pb = triton.next_power_of_2(triton.cdiv(P, nch))
    idx = topk_idx.contiguous()
    poffs = torch.empty(E + 1, dtype=torch.int32, device=dev)
    rowmap = torch.empty(P, dtype=torch.int32, device=dev)
    rw = torch.empty(P, dtype=torch.float32, device=dev)
    inv = torch.empty(R, dtype=torch.int32, device=dev)
    cnt = torch.empty((nch, E), dtype=torch.int32, device=dev)
    base = torch.empty((nch, E), dtype=torch.int32, device=dev)
    _route_count_kernel[(nch,)](idx, cnt, rowmap, rw, R, P, E=E, BLOCK=block, PB=pb, num_warps=4)
    _route_offs_kernel[(1,)](cnt, base, poffs, nch, E=E, NCP=triton.next_power_of_2(nch), ALIGN=align, num_warps=4)
    _route_scatter_kernel[(nch,)](idx, topk_w.contiguous(), base, rowmap, rw, inv, R, E=E, K=K, BLOCK=block,
                                  num_warps=4)
    return rowmap, rw, inv, poffs, poffs[E:]


@triton.jit
def _gw4(GW, p, m, NC: tl.constexpr, j: tl.constexpr):
    x0 = tl.load(GW + p * NC + 4 * j, mask=m, other=0.0)
    x1 = tl.load(GW + p * NC + 4 * j + 1, mask=m, other=0.0)
    x2 = tl.load(GW + p * NC + 4 * j + 2, mask=m, other=0.0)
    x3 = tl.load(GW + p * NC + 4 * j + 3, mask=m, other=0.0)
    return (x0 + x1) + (x2 + x3)


@triton.jit
def _route_dw_kernel(GW, INV, OUT, R, NC: tl.constexpr, BLOCK: tl.constexpr):
    """OUT[r] = sum_c GW[INV[r], c]: d topk_weight of route r from the SwiGLU-bwd partials. NC 16 / 8: the pairwise
    tree torch's gw.sum(1) uses for those widths (bit-identical, checked); other widths sequential."""
    r = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = r < R
    p = tl.load(INV + r, mask=m, other=0).to(tl.int64)
    if NC == 16:
        acc = (_gw4(GW, p, m, NC, 0) + _gw4(GW, p, m, NC, 1)) + (_gw4(GW, p, m, NC, 2) + _gw4(GW, p, m, NC, 3))
    elif NC == 8:
        acc = _gw4(GW, p, m, NC, 0) + _gw4(GW, p, m, NC, 1)
    else:
        acc = tl.zeros((BLOCK,), dtype=tl.float32)
        for c in tl.static_range(NC):
            acc += tl.load(GW + p * NC + c, mask=m, other=0.0)
    tl.store(OUT + r, acc.to(OUT.dtype.element_ty), mask=m)


def route_dw_partials(gw: torch.Tensor, inv: torch.Tensor, out_dtype=torch.float32) -> torch.Tensor:
    """gw [P, NC] fp32 partials of d w per pool row (swiglu_bwd_dual_blk raw_dw), inv int32 [R] pool row of each
    route -> d topk_weights [R] in one launch (replaces zeros + sum(1) + inv.long() + index + cast; bit-identical to
    ``gw.sum(1)[inv.long()]`` for NC 16 and 8)."""
    R = inv.numel()
    NC = gw.shape[1]
    out = torch.empty(R, device=gw.device, dtype=out_dtype)
    _route_dw_kernel[(triton.cdiv(R, 1024),)](gw, inv, out, R, NC=NC, BLOCK=1024, num_warps=4)
    return out


@triton.jit
def _lpt_order_kernel(POFFS, PERM, G, GP: tl.constexpr):
    """one program: PERM = experts by window length descending, ties by index (= stable argsort, descending)."""
    g = tl.arange(0, GP)
    v = g < G
    ln = tl.where(v, tl.load(POFFS + 1 + g, mask=v, other=0) - tl.load(POFFS + g, mask=v, other=0), -1)
    before = (ln[None, :] > ln[:, None]) | ((ln[None, :] == ln[:, None]) & (g[None, :] < g[:, None]))
    rank = tl.sum(before.to(tl.int32), axis=1)
    tl.store(PERM + rank, g.to(tl.int32), mask=v)


def expert_order_lpt_fused(poffs_i32: torch.Tensor) -> torch.Tensor:
    """Same int32 [G] order as grouped_gemm_fp8blk_vkp's ``expert_order_lpt`` (stable descending argsort of the
    window lengths) in one launch instead of seven."""
    G = poffs_i32.numel() - 1
    perm = torch.empty(G, device=poffs_i32.device, dtype=torch.int32)
    _lpt_order_kernel[(1,)](poffs_i32, perm, G, GP=triton.next_power_of_2(G), num_warps=4)
    return perm
