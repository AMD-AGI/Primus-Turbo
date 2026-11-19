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
"""

import torch
import triton
import triton.language as tl

FP8_MAX = 448.0
MX = 32


@triton.jit
def _ue8m0_exp(amax, FP8_MAX: tl.constexpr):
    """smallest e with amax <= FP8_MAX * 2^e (exact on the fp32 bit pattern), 0 for amax == 0."""
    r = amax / FP8_MAX
    bits = r.to(tl.int32, bitcast=True)
    e = ((bits >> 23) & 0xFF) - 127
    mant = bits & 0x7FFFFF
    e = tl.where(mant != 0, e + 1, e)  # round the exponent up unless r is an exact power of two
    e = tl.where(amax > 0, e, 0)
    return tl.minimum(tl.maximum(e, -126), 126)


@triton.jit
def _act_1x128_kernel(X, Q, S, M, K, stride_x, stride_q, R: tl.constexpr, FP8_MAX: tl.constexpr):
    """grid (cdiv(M, R), K // 128): R rows x one 128-wide tile; scales [M, K/32] uint8 viewed as int32 [M, K/128]."""
    pid_m = tl.program_id(0)
    kb = tl.program_id(1)
    rows = pid_m * R + tl.arange(0, R)
    cols = kb * 128 + tl.arange(0, 128)
    rm = rows < M
    x = tl.load(X + rows[:, None].to(tl.int64) * stride_x + cols[None, :], mask=rm[:, None], other=0.0).to(tl.float32)
    amax = tl.max(tl.abs(x), axis=1)
    e = _ue8m0_exp(amax, FP8_MAX)
    inv = ((127 - e) << 23).to(tl.float32, bitcast=True)  # 2^-e
    q = (x * inv[:, None]).to(Q.dtype.element_ty)
    tl.store(Q + rows[:, None].to(tl.int64) * stride_q + cols[None, :], q, mask=rm[:, None])
    b = (e + 127) & 0xFF
    tl.store(S.to(tl.pointer_type(tl.int32)) + rows.to(tl.int64) * (K // 128) + kb, b * 0x01010101, mask=rm)


def quant_act_1x128(x: torch.Tensor, dtype=torch.float8_e4m3fn, rows_per_prog: int = 16):
    """x [M, K] bf16 (K % 128 == 0) -> (q [M, K] fp8, s [M, K/32] uint8 E8M0 repeated per 128)."""
    assert x.dim() == 2 and x.shape[1] % 128 == 0 and x.stride(1) == 1
    M, K = x.shape
    q = torch.empty((M, K), device=x.device, dtype=dtype)
    s = torch.empty((M, K // MX), device=x.device, dtype=torch.uint8)
    _act_1x128_kernel[(triton.cdiv(M, rows_per_prog), K // 128)](
        x, q, s, M, K, x.stride(0), q.stride(0), R=rows_per_prog, FP8_MAX=torch.finfo(dtype).max, num_warps=4
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
    inv = ((127 - e) << 23).to(tl.float32, bitcast=True)
    q = (w * inv).to(Q.dtype.element_ty)
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
                             CT: tl.constexpr = 0):
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
    _quant_tile_dual(x, p, c, pt, pc, P, C, Q, S, QT, ST, FP8_MAX, TMODE, CT)


def quant_dual_blk(x: torch.Tensor, P: int, rowmap: torch.Tensor = None, dtype=torch.float8_e4m3fn, nw: int = 8,
                   preal: torch.Tensor = None, tmode: int = 1, compact_t: bool = False):
    """x [rows, C] bf16 -> pool-ordered (q [P, C], s [P, C/32], qT [C, P], sT [C, P/32]); P % 128 == 0.
    ``rowmap`` int32 [P]: source row of each pool row (negative = padding, quantized as zeros)."""
    C = x.shape[1]
    assert P % 128 == 0 and C % 128 == 0 and x.stride(1) == 1
    dev = x.device
    q = torch.empty((P, C), device=dev, dtype=dtype)
    s = torch.empty((P, C // MX), device=dev, dtype=torch.uint8)
    qT = torch.empty((C, P), device=dev, dtype=dtype)
    sT = torch.empty((C, P // (128 if compact_t else MX)), device=dev, dtype=torch.uint8)
    _dual_1x128_128x1_kernel[(P // 128, C // 128)](
        x, rowmap if rowmap is not None else x, q, s, qT, sT, P, C, x.stride(0), preal if preal is not None else x,
        HAS_MAP=int(rowmap is not None), BT=128, BC=128, FP8_MAX=torch.finfo(dtype).max,
        HAS_PREAL=int(preal is not None), TMODE=tmode, CT=int(compact_t), num_warps=nw)
    return q, s, qT, sT


@triton.jit
def _quant_tile_dual(v, p, c, pt, pc, P, C, Q, S, QT, ST, FP8_MAX: tl.constexpr, TMODE: tl.constexpr = 1,
                     CT: tl.constexpr = 0):
    """quantize one [128 pool rows, 128 cols] fp32 tile both ways (see _dual_1x128_128x1_kernel)."""
    av = tl.abs(v)
    er = _ue8m0_exp(tl.max(av, axis=1), FP8_MAX)
    qr = (v * (((127 - er) << 23).to(tl.float32, bitcast=True))[:, None]).to(Q.dtype.element_ty)
    tl.store(Q + p[:, None].to(tl.int64) * C + c[None, :], qr)
    tl.store(S.to(tl.pointer_type(tl.int32)) + p.to(tl.int64) * (C // 128) + pc, ((er + 127) & 0xFF) * 0x01010101)
    ec = _ue8m0_exp(tl.max(av, axis=0), FP8_MAX)
    qc = (v * (((127 - ec) << 23).to(tl.float32, bitcast=True))[None, :]).to(QT.dtype.element_ty)
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
        tl.store(ST.to(tl.pointer_type(tl.int32)) + c.to(tl.int64) * (P // 128) + pt, ((ec + 127) & 0xFF) * 0x01010101)


@triton.jit
def _swiglu_fwd_dual_kernel(L1, W, Q, S, QT, ST, P, I, PREAL, ALPHA: tl.constexpr, OFF: tl.constexpr,
                            GLO: tl.constexpr, GHI: tl.constexpr, UCL: tl.constexpr, FP8_MAX: tl.constexpr,
                            HAS_PREAL: tl.constexpr, CT: tl.constexpr = 0):
    """act_w[p, c] = w[p] * g * sigmoid(alpha g) * (u + off), g/u the clamped gate/up halves of l1 [P, 2I];
    quantized 1x128 (fc2 A operand) and 128x1 transposed (dW2 B operand). grid (P // 128, I // 128)."""
    pt = tl.program_id(0)
    pc = tl.program_id(1)
    if HAS_PREAL:
        if pt * 128 >= tl.load(PREAL):
            return
    p = pt * 128 + tl.arange(0, 128)
    c = pc * 128 + tl.arange(0, 128)
    rowp = L1 + p[:, None].to(tl.int64) * (2 * I)
    g = tl.load(rowp + c[None, :]).to(tl.float32)
    u = tl.load(rowp + I + c[None, :]).to(tl.float32)
    w = tl.load(W + p)
    g = tl.minimum(tl.maximum(g, GLO), GHI)
    u = tl.minimum(tl.maximum(u, -UCL), UCL)
    act = g * tl.sigmoid(ALPHA * g) * (u + OFF) * w[:, None]
    _quant_tile_dual(act, p, c, pt, pc, P, I, Q, S, QT, ST, FP8_MAX, 1, CT)


def swiglu_fwd_dual_blk(l1: torch.Tensor, w: torch.Tensor, activation=None, dtype=torch.float8_e4m3fn, preal=None,
                        compact_t: bool = False):
    """l1 [P, 2I] bf16 (pool order), w [P] fp32 routing weight per pool row (0 on padding rows) ->
    (q [P, I], s [P, I/32], qT [I, P], sT [I, P/32]) of the weighted activation."""
    from primus_turbo.flydsl.utils.glu_activation import activation_constexpr
    alpha, off, glo, ghi, ucl = activation_constexpr(activation)
    P, I2 = l1.shape
    I = I2 // 2
    dev = l1.device
    q = torch.empty((P, I), device=dev, dtype=dtype)
    s = torch.empty((P, I // MX), device=dev, dtype=torch.uint8)
    qT = torch.empty((I, P), device=dev, dtype=dtype)
    sT = torch.empty((I, P // (128 if compact_t else MX)), device=dev, dtype=torch.uint8)
    _swiglu_fwd_dual_kernel[(P // 128, I // 128)](l1, w, q, s, qT, sT, P, I, preal if preal is not None else w,
                                                  ALPHA=alpha, OFF=off, GLO=glo, GHI=ghi, UCL=ucl,
                                                  FP8_MAX=torch.finfo(dtype).max, HAS_PREAL=int(preal is not None),
                                                  CT=int(compact_t), num_warps=8)
    return q, s, qT, sT


@triton.jit
def _swiglu_bwd_dual_kernel(DA, L1, W, Q, S, QT, ST, GW, P, I, PREAL, ALPHA: tl.constexpr, OFF: tl.constexpr,
                            GLO: tl.constexpr, GHI: tl.constexpr, UCL: tl.constexpr, FP8_MAX: tl.constexpr,
                            HAS_PREAL: tl.constexpr, CT: tl.constexpr = 0):
    """d act_w [P, I] -> grad_l1 [P, 2I] (gate | up), quantized 1x128 (L1 dgrad A) and 128x1 transposed (dW1 A),
    plus per-(row, 128-col block) partials of d w = sum_c d act_w * act. grid (P // 128, I // 128)."""
    pt = tl.program_id(0)
    pc = tl.program_id(1)
    if HAS_PREAL:
        if pt * 128 >= tl.load(PREAL):
            return
    p = pt * 128 + tl.arange(0, 128)
    c = pc * 128 + tl.arange(0, 128)
    rowp = L1 + p[:, None].to(tl.int64) * (2 * I)
    g0 = tl.load(rowp + c[None, :]).to(tl.float32)
    u0 = tl.load(rowp + I + c[None, :]).to(tl.float32)
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
    _quant_tile_dual(d_gate, p, c, pt, pc, P, 2 * I, Q, S, QT, ST, FP8_MAX, 1, CT)
    _quant_tile_dual(d_up, p, c + I, pt, pc + I // 128, P, 2 * I, Q, S, QT, ST, FP8_MAX, 1, CT)


def swiglu_bwd_dual_blk(d_act: torch.Tensor, l1: torch.Tensor, w: torch.Tensor, activation=None,
                        dtype=torch.float8_e4m3fn, preal=None, compact_t: bool = False):
    """-> (q [P, 2I], s [P, 2I/32], qT [2I, P], sT [2I, P/32]) of grad_l1, and d w [P] (fp32)."""
    from primus_turbo.flydsl.utils.glu_activation import activation_constexpr
    alpha, off, glo, ghi, ucl = activation_constexpr(activation)
    P, I = d_act.shape
    dev = d_act.device
    q = torch.empty((P, 2 * I), device=dev, dtype=dtype)
    s = torch.empty((P, 2 * I // MX), device=dev, dtype=torch.uint8)
    qT = torch.empty((2 * I, P), device=dev, dtype=dtype)
    sT = torch.empty((2 * I, P // (128 if compact_t else MX)), device=dev, dtype=torch.uint8)
    # rows past the real pool keep zero partials (their d w is never read)
    gw = torch.zeros((P, I // 128), device=dev, dtype=torch.float32)
    _swiglu_bwd_dual_kernel[(P // 128, I // 128)](d_act, l1, w, q, s, qT, sT, gw, P, I,
                                                  preal if preal is not None else w, ALPHA=alpha, OFF=off, GLO=glo,
                                                  GHI=ghi, UCL=ucl, FP8_MAX=torch.finfo(dtype).max,
                                                  HAS_PREAL=int(preal is not None), CT=int(compact_t), num_warps=8)
    return q, s, qT, sT, gw.sum(1)


@triton.jit
def _combine_kernel(Y, INV, OUT, H, K: tl.constexpr, BH: tl.constexpr):
    """out[t] = sum_k Y[INV[t*K + k]] (bf16 rows, fp32 sum). grid (T, H // BH)."""
    t = tl.program_id(0)
    h = tl.program_id(1) * BH + tl.arange(0, BH)
    acc = tl.zeros((BH,), dtype=tl.float32)
    for k in tl.static_range(K):
        r = tl.load(INV + t * K + k).to(tl.int64)
        acc += tl.load(Y + r * H + h).to(tl.float32)
    tl.store(OUT + t.to(tl.int64) * H + h, acc.to(OUT.dtype.element_ty))


def combine_rows(y: torch.Tensor, inv: torch.Tensor, T: int, K: int, bh: int = 1024):
    """y [P, H] pool rows (routing weight already applied), inv int32 [T*K] pool row of route (t, k) -> [T, H]."""
    H = y.shape[1]
    out = torch.empty((T, H), device=y.device, dtype=y.dtype)
    _combine_kernel[(T, H // bh)](y, inv, out, H, K=K, BH=bh, num_warps=4)
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
