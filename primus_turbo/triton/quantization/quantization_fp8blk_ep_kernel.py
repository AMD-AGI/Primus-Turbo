###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Expert-parallel (EP>1) data movement for the FP8-block MegaMoE (fused_mega_moe_fp8blk_ep) on gfx1250.

DeepSeek recipe: tokens leave their rank as E4M3 rows with one UE8M0 byte per 1x128 tile (FP8 dispatch); the expert
outputs come back as BF16 rows, one per route (BF16 combine). All buffers are fixed-shape (per-peer worst case) and
every table is built on the device, so nothing here syncs with the host.

  src_layout_ep       source rank: dispatch slots [T, R] (token order per destination), per-destination counts, each
                      route's combine slot, and the per-slot route metadata [R, CAP, 3K] (local expert id / -1,
                      fp32 weight bits, combine slot) -- three chunked Triton kernels (count / offsets / scatter).
  quant_rows_send     source rank: x [T, H] -> 1x128 quant (the exact row quant of quant_dual_blk) written straight
                      into the per-destination send slots [R, CAP, H] (+ [R, CAP, H/128] scale bytes); one token is
                      sent once per destination rank (deduplicated), not once per route. ``quant=False``: bf16 rows.
  build_routing_blk_masked
                      receiving rank: pool tables of quantization_fp8blk_moe_kernel.build_routing_blk over its local experts for
                      routes given as [N, K] local expert ids, -1 = route of another rank / empty slot (skipped);
                      with the received combine slots also the pool row -> combine slot map.
  gather_dual_fp8     receiving rank: pool-ordered FP8 operands from the received FP8 rows: the 1x128 copy is the
                      received bytes (gather only), the 128x1 (token-direction) copy for the wgrads is re-quantized
                      from the dequantized FP8 rows (DeepSeek-V3: "dequantize, transpose, re-quantize into 128x1").
  pack_pool_rows      receiving rank: live pool rows -> their combine slots (grid over the live pool rows only).
"""

import torch
import triton
import triton.language as tl

from primus_turbo.triton.quantization.quantization_fp8blk_moe_kernel import (
    _dual_alloc,
    _route_count_kernel,
    _route_offs_kernel,
    _ue8m0_exp,
)
from primus_turbo.triton.quantization.quantization_fp8blk_utils import (
    pow2_neg_fly,
    pow2_pos,
    scale_layout_id,
    store_scale,
)

__all__ = [
    "src_layout_ep",
    "quant_rows_send",
    "build_routing_blk_masked",
    "gather_dual_fp8",
    "pack_pool_rows",
]


@triton.jit
def _src_onehot(
    IDX, ch, T, K: tl.constexpr, KP: tl.constexpr, R: tl.constexpr, EL: tl.constexpr, BLOCK: tl.constexpr
):
    t = ch * BLOCK + tl.arange(0, BLOCK)
    k = tl.arange(0, KP)
    m = (t[:, None] < T) & (k[None, :] < K)
    idx = tl.load(IDX + t[:, None].to(tl.int64) * K + k[None, :], mask=m, other=0)
    dst = tl.where(m, idx // EL, -1).to(tl.int32)  # [BLOCK, KP] rank of each route, -1 off the table
    d = tl.arange(0, R)
    oh = (dst[:, :, None] == d[None, None, :]).to(tl.int32)  # [BLOCK, KP, R]
    return t, k, m, idx, dst, d, oh


@triton.jit
def _src_count_kernel(
    IDX, CT, CR, T, K: tl.constexpr, KP: tl.constexpr, R: tl.constexpr, EL: tl.constexpr, BLOCK: tl.constexpr
):
    """grid (nchunk,): per chunk of BLOCK tokens, tokens sent to each rank CT [nchunk, R] and routes to it CR."""
    ch = tl.program_id(0)
    t, k, m, idx, dst, d, oh = _src_onehot(IDX, ch, T, K, KP, R, EL, BLOCK)
    tl.store(CT + ch * R + d, tl.sum(tl.max(oh, axis=1), axis=0))
    tl.store(CR + ch * R + d, tl.sum(tl.sum(oh, axis=1), axis=0))


@triton.jit
def _src_offs_kernel(CT, CR, BT, BR, CNT, NCH, R: tl.constexpr, NCP: tl.constexpr):
    """one program: exclusive prefix over the chunks (BT / BR [nchunk, R]) and the per-rank token totals CNT [R]."""
    c = tl.arange(0, NCP)
    d = tl.arange(0, R)
    m = (c < NCH)[:, None]
    off = c[:, None] * R + d[None, :]
    ct = tl.load(CT + off, mask=m, other=0)
    cr = tl.load(CR + off, mask=m, other=0)
    tl.store(BT + off, tl.cumsum(ct, 0) - ct, mask=m)
    tl.store(BR + off, tl.cumsum(cr, 0) - cr, mask=m)
    tl.store(CNT + d, tl.sum(ct, axis=0))


@triton.jit
def _src_scatter_kernel(
    IDX,
    TW,
    BT,
    BR,
    SLOT,
    CINV,
    META,
    T,
    CAP,
    CCAP,
    K: tl.constexpr,
    KP: tl.constexpr,
    R: tl.constexpr,
    EL: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """grid (nchunk,): SLOT [T, R] (slot of token t in rank d's dispatch block, token order; -1 not sent),
    CINV [T*K] (combine slot d * CCAP + j of route (t, k): its rank in (t, k) order among the routes to d) and the
    metadata row META[d, slot, :] = [local expert id or -1 (K), weight fp32 bits or 0 (K), combine pos or -1 (K)]."""
    ch = tl.program_id(0)
    t, k, m, idx, dst, d, oh = _src_onehot(IDX, ch, T, K, KP, R, EL, BLOCK)
    has = tl.max(oh, axis=1)  # [BLOCK, R]
    bt = tl.load(BT + ch * R + d)
    slot = tl.where(has > 0, bt[None, :] + tl.cumsum(has, 0) - 1, -1)
    tl.store(SLOT + t[:, None].to(tl.int64) * R + d[None, :], slot, mask=(t < T)[:, None])
    br = tl.load(BR + ch * R + d)
    cs = tl.reshape(tl.cumsum(tl.reshape(oh, (BLOCK * KP, R)), 0), (BLOCK, KP, R))  # (t, k) order
    rpos = tl.sum(oh * (br[None, None, :] + cs - 1), axis=2)  # [BLOCK, KP] (one rank matches a live route)
    tl.store(CINV + t[:, None].to(tl.int64) * K + k[None, :], dst * CCAP + rpos, mask=m)
    sel = oh > 0
    w = tl.load(TW + t[:, None].to(tl.int64) * K + k[None, :], mask=m, other=0.0).to(tl.float32)
    loc = (idx - dst * EL).to(tl.int32)
    row = d[None, None, :] * CAP + tl.where(has > 0, slot, 0)[:, None, :]
    ok = (has > 0)[:, None, :] & (k < K)[None, :, None] & (t < T)[:, None, None]
    ptr = META + row.to(tl.int64) * (3 * K) + k[None, :, None]
    tl.store(ptr, tl.where(sel, loc[:, :, None], -1), mask=ok)
    tl.store(ptr + K, tl.where(sel, w[:, :, None], 0.0).to(tl.int32, bitcast=True), mask=ok)
    tl.store(ptr + 2 * K, tl.where(sel, rpos[:, :, None], -1), mask=ok)


def src_layout_ep(
    topk_idx: torch.Tensor, topk_w: torch.Tensor, R: int, E_l: int, cap: int, ccap: int, block: int = 128
):
    """Source rank of an EP group of R ranks (rank d owns global experts [d*E_l, (d+1)*E_l)): -> (slot int32 [T, R],
    cinv int32 [T*K], meta int32 [R, cap, 3K], cnt int32 [R]). Metadata rows past cnt[d] are left unwritten.
    Device-only, fixed shapes."""
    T, K = topk_idx.shape
    assert R & (R - 1) == 0, "EP size must be a power of two (tl.arange)"
    dev = topk_idx.device
    idx = topk_idx.contiguous()
    tw = topk_w.contiguous()
    nch = max(1, triton.cdiv(T, block))
    KP = triton.next_power_of_2(K)
    ct = torch.empty((nch, R), dtype=torch.int32, device=dev)
    cr = torch.empty_like(ct)
    bt = torch.empty_like(ct)
    br = torch.empty_like(ct)
    cnt = torch.empty(R, dtype=torch.int32, device=dev)
    slot = torch.empty((T, R), dtype=torch.int32, device=dev)
    cinv = torch.empty(T * K, dtype=torch.int32, device=dev)
    meta = torch.empty((R, cap, 3 * K), dtype=torch.int32, device=dev)
    kw = dict(K=K, KP=KP, R=R, EL=E_l, BLOCK=block, num_warps=4)
    _src_count_kernel[(nch,)](idx, ct, cr, T, **kw)
    _src_offs_kernel[(1,)](ct, cr, bt, br, cnt, nch, R=R, NCP=triton.next_power_of_2(nch), num_warps=4)
    _src_scatter_kernel[(nch,)](idx, tw, bt, br, slot, cinv, meta, T, cap, ccap, **kw)
    return slot, cinv, meta, cnt


@triton.jit
def _quant_rows_send_kernel(
    X,
    SLOT,
    Q,
    S,
    T,
    H,
    stride_x,
    CAP,
    R: tl.constexpr,
    BT: tl.constexpr,
    QUANT: tl.constexpr,
    FP8_MAX: tl.constexpr,
):
    """grid (cdiv(T, BT), H // 128). SLOT int32 [T, R]: slot of token t in destination d's block (-1: not sent).
    QUANT: Q [R, CAP, H] fp8 + S [R, CAP, H/128] uint8 (E8M0 byte per 1x128 tile); else Q [R, CAP, H] bf16 copy."""
    pt = tl.program_id(0)
    pc = tl.program_id(1)
    t = pt * BT + tl.arange(0, BT)
    c = pc * 128 + tl.arange(0, 128)
    tm = t < T
    v = tl.load(X + t[:, None].to(tl.int64) * stride_x + c[None, :], mask=tm[:, None], other=0.0)
    if QUANT:
        v = v.to(tl.float32)
        # the row half of quantization_fp8blk_moe_kernel._quant_tile_dual, expression for expression (bit-identical bytes)
        av = tl.abs(v)
        er = _ue8m0_exp(tl.max(av, axis=1), FP8_MAX)
        q = (v * pow2_neg_fly(er)[:, None]).to(Q.dtype.element_ty)
        b = ((er + 127) & 0xFF).to(tl.uint8)
    else:
        q = v
    for d in tl.static_range(R):
        sl = tl.load(SLOT + t.to(tl.int64) * R + d, mask=tm, other=-1)
        ok = sl >= 0
        row = d * CAP + tl.where(ok, sl, 0).to(tl.int64)
        tl.store(Q + row[:, None] * H + c[None, :], q, mask=ok[:, None])
        if QUANT:
            tl.store(S + row * (H // 128) + pc, b, mask=ok)


def quant_rows_send(
    x: torch.Tensor,
    slot: torch.Tensor,
    cap: int,
    quant: bool = True,
    q=None,
    s=None,
    dtype=torch.float8_e4m3fn,
    bt: int = 64,
):
    """x [T, H] bf16, slot int32 [T, R] (-1 = token not sent to that rank) -> send buffers
    q [R, cap, H] (fp8, or bf16 when ``quant`` is False) and s [R, cap, H/128] uint8 (None when not ``quant``).
    Slots no token maps to are left as they are (never read)."""
    T, H = x.shape
    R = slot.shape[1]
    assert H % 128 == 0 and x.stride(1) == 1 and slot.dtype == torch.int32 and slot.is_contiguous()
    dev = x.device
    if q is None:
        q = torch.empty((R, cap, H), device=dev, dtype=dtype if quant else x.dtype)
    if quant and s is None:
        s = torch.empty((R, cap, H // 128), device=dev, dtype=torch.uint8)
    if T == 0:
        return q, s
    _quant_rows_send_kernel[(triton.cdiv(T, bt), H // 128)](
        x,
        slot,
        q,
        s if quant else q,
        T,
        H,
        x.stride(0),
        cap,
        R=R,
        BT=bt,
        QUANT=int(quant),
        FP8_MAX=torch.finfo(dtype).max if quant else 448.0,
        num_warps=4,
    )
    return q, s


@triton.jit
def _route_scatter_masked_kernel(
    IDX,
    TW,
    CS,
    BASE,
    ROWMAP,
    RW,
    INV,
    P2S,
    R,
    TW_STRIDE,
    CAP,
    CCAP,
    E: tl.constexpr,
    K: tl.constexpr,
    BLOCK: tl.constexpr,
    HAS_CS: tl.constexpr,
):
    """quantization_fp8blk_moe_kernel._route_scatter_kernel with routes of id < 0 skipped (INV -1 for them). Route r = (n, k) reads
    its weight at TW[n * TW_STRIDE + k]; HAS_CS: also P2S[pool row] = (n // CAP) * CCAP + CS[n * TW_STRIDE + k], the
    combine slot of the route (source rank n // CAP)."""
    ch = tl.program_id(0)
    e_ar = tl.arange(0, E)
    r = ch * BLOCK + tl.arange(0, BLOCK)
    m = r < R
    idx = tl.load(IDX + r, mask=m, other=-1).to(tl.int32)
    oh = (idx[:, None] == e_ar[None, :]).to(tl.int32)
    base = tl.load(BASE + ch * E + e_ar)
    pos = tl.sum(oh * (base[None, :] + tl.cumsum(oh, 0) - 1), axis=1)
    live = m & (idx >= 0)
    n = r // K
    off = n.to(tl.int64) * TW_STRIDE + r % K
    tl.store(ROWMAP + pos, n, mask=live)
    tl.store(RW + pos, tl.load(TW + off, mask=live, other=0.0).to(tl.float32), mask=live)
    tl.store(INV + r, tl.where(live, pos, -1), mask=m)
    if HAS_CS:
        cs = tl.load(CS + off, mask=live, other=0)
        tl.store(P2S + pos, (n // CAP) * CCAP + cs, mask=live)


def build_routing_blk_masked(
    local_idx: torch.Tensor,
    local_w: torch.Tensor,
    num_experts: int,
    P: int,
    align: int = 256,
    block: int = None,
    cslot: torch.Tensor = None,
    cap: int = 0,
    ccap: int = 0,
):
    """build_routing_blk for [N, K] routes given as local expert ids in [0, E) or -1 (no route): -> (rowmap int32 [P]
    (row n of the route, -1 = padding), route_w f32 [P], inv int32 [N*K] (-1 for no route), poffs int32 [E+1],
    preal = poffs[E:], p2s). Same pool layout rules (and the same count / offset kernels) as EP=1: each expert ceil to
    ``align`` rows, at least ``align``; routes of one expert in row-major (n, k) order (whatever ``block``, the chunk
    size, which only trades the chunk kernels' width against the single-program offset kernel). Device-only, capturable.
    ``local_w`` / ``cslot``: [N, K] views with unit column stride and a common row stride (e.g. columns of the
    received metadata). ``cslot`` given: p2s int32 [P] = combine slot (n // cap) * ccap + cslot[n, k] of each pool
    row's route, -1 for padding rows; else p2s is None.
    P must hold every live route plus E * align rows of padding."""
    N, K = local_idx.shape
    R = N * K
    E = num_experts
    assert E & (E - 1) == 0, "local expert count must be a power of two (tl.arange)"
    assert local_w.dtype == torch.float32 and local_w.stride(1) == 1 and local_w.shape == (N, K)
    if cslot is not None:
        assert cslot.dtype == torch.int32 and cslot.stride() == local_w.stride() and cap > 0 and ccap > 0
    dev = local_idx.device
    if block is None:  # at most ~1024 chunks: the offsets kernel scans all chunks in one program
        block = max(128, triton.next_power_of_2(triton.cdiv(R, 1024)))
    nw = 4 if block <= 128 else 8
    nch = triton.cdiv(R, block)
    pb = triton.next_power_of_2(triton.cdiv(P, nch))
    idx = local_idx.contiguous()
    poffs = torch.empty(E + 1, dtype=torch.int32, device=dev)
    rowmap = torch.empty(P, dtype=torch.int32, device=dev)
    rw = torch.empty(P, dtype=torch.float32, device=dev)
    inv = torch.empty(R, dtype=torch.int32, device=dev)
    cnt = torch.empty((nch, E), dtype=torch.int32, device=dev)
    base = torch.empty((nch, E), dtype=torch.int32, device=dev)
    p2s = torch.full((P,), -1, dtype=torch.int32, device=dev) if cslot is not None else None
    _route_count_kernel[(nch,)](idx, cnt, rowmap, rw, R, P, E=E, BLOCK=block, PB=pb, num_warps=nw)
    _route_offs_kernel[(1,)](
        cnt, base, poffs, nch, E=E, NCP=triton.next_power_of_2(nch), ALIGN=align, num_warps=4
    )
    _route_scatter_masked_kernel[(nch,)](
        idx,
        local_w,
        cslot if cslot is not None else local_w,
        base,
        rowmap,
        rw,
        inv,
        p2s if p2s is not None else rowmap,
        R,
        local_w.stride(0),
        max(cap, 1),
        ccap,
        E=E,
        K=K,
        BLOCK=block,
        HAS_CS=int(cslot is not None),
        num_warps=nw,
    )
    return rowmap, rw, inv, poffs, poffs[E:], p2s


@triton.jit
def _gather_dual_fp8_kernel(
    XQ,
    XS,
    ROWMAP,
    Q,
    S,
    QT,
    ST,
    P,
    C,
    PREAL,
    FP8_MAX: tl.constexpr,
    HAS_PREAL: tl.constexpr,
    CT: tl.constexpr = 0,
    SL: tl.constexpr = 0,
):
    """grid (P // 128, C // 128). Pool row p <- received row ROWMAP[p] (negative: padding row, zeros):
    Q  [P, C] fp8 = the received bytes, S [P, C/32] = the received E8M0 byte (x4 word / layout SL),
    QT [C, P] fp8 + ST: 128x1 tiles along the pool rows of the dequantized rows (q * 2^e, exact in fp32), with the
    column half of quantization_fp8blk_moe_kernel._quant_tile_dual (TMODE 1 stores, CT compact scales)."""
    pt = tl.program_id(0)
    pc = tl.program_id(1)
    if HAS_PREAL:
        if pt * 128 >= tl.load(PREAL):
            return
    p = pt * 128 + tl.arange(0, 128)
    c = pc * 128 + tl.arange(0, 128)
    src = tl.load(ROWMAP + p)
    valid = src >= 0
    src = tl.where(valid, src, 0).to(tl.int64)
    qb = tl.load(XQ + src[:, None] * C + c[None, :], mask=valid[:, None], other=0)  # uint8 bytes
    er = tl.load(XS + src * (C // 128) + pc, mask=valid, other=127).to(tl.int32) - 127
    tl.store(Q + p[:, None].to(tl.int64) * C + c[None, :], qb)
    store_scale(S, p, pc, er, P, C // 128, SL)
    v = qb.to(tl.float8e4nv, bitcast=True).to(tl.float32) * pow2_pos(er)[:, None]
    ec = _ue8m0_exp(tl.max(tl.abs(v), axis=0), FP8_MAX)
    qc = (v * pow2_neg_fly(ec)[None, :]).to(QT.dtype.element_ty)
    u = tl.reshape(qc.to(tl.uint8, bitcast=True).to(tl.int32), (32, 4, 128))
    w = tl.sum(
        u << (tl.arange(0, 4) * 8)[None, :, None], axis=1
    )  # [32, 128]: 4 pool rows of a column per int32
    pw = pt * 32 + tl.arange(0, 32)
    tl.store(QT.to(tl.pointer_type(tl.int32)) + c[:, None].to(tl.int64) * (P // 4) + pw[None, :], tl.trans(w))
    if CT:
        tl.store(ST + c.to(tl.int64) * (P // 128) + pt, ((ec + 127) & 0xFF).to(tl.uint8))
    else:
        store_scale(ST, c, pt, ec, C, P // 128, SL)


def gather_dual_fp8(
    xq: torch.Tensor,
    xs: torch.Tensor,
    P: int,
    rowmap: torch.Tensor,
    preal: torch.Tensor = None,
    compact_t: bool = False,
    scale_layout="x4",
    nw: int = 8,
):
    """xq [N, C] fp8 received rows, xs [N, C/128] uint8 their E8M0 bytes -> pool-ordered (q [P, C], s [P, C/32],
    qT [C, P], sT [C, P/32] or [C, P/128] when ``compact_t``): quant_dual_blk's outputs and layouts, with the row
    quant taken over unchanged and the 128x1 quant re-derived from the FP8 values."""
    N, C = xq.shape
    assert P % 128 == 0 and C % 128 == 0 and xq.is_contiguous() and xs.is_contiguous()
    assert xs.dtype == torch.uint8 and xs.shape == (N, C // 128) and rowmap.dtype == torch.int32
    sl = scale_layout_id(scale_layout)
    q, s, qT, sT = _dual_alloc(P, C, xq.dtype, sl, compact_t, xq.device)
    _gather_dual_fp8_kernel[(P // 128, C // 128)](
        xq.view(torch.uint8),
        xs,
        rowmap,
        q.view(torch.uint8),
        s,
        qT,
        sT,
        P,
        C,
        preal if preal is not None else rowmap,
        FP8_MAX=torch.finfo(xq.dtype).max,
        HAS_PREAL=int(preal is not None),
        CT=int(compact_t),
        SL=sl,
        num_warps=nw,
    )
    return q, s, qT, sT


@triton.jit
def _pack_pool_rows_kernel(
    SRC, P2S, OUT, EXT, EXT_OUT, PREAL, C, BR: tl.constexpr, BC: tl.constexpr, HAS_EXT: tl.constexpr
):
    """grid (P // BR, C // BC): OUT[P2S[p]] = SRC[p] for the live pool rows p < PREAL with P2S[p] >= 0
    (EXT_OUT[P2S[p]] = EXT[p] too when HAS_EXT); tiles past PREAL exit at once."""
    pt = tl.program_id(0)
    pc = tl.program_id(1)
    if pt * BR >= tl.load(PREAL):
        return
    p = pt * BR + tl.arange(0, BR)
    s = tl.load(P2S + p)
    ok = s >= 0
    s64 = tl.where(ok, s, 0).to(tl.int64)
    h = pc * BC + tl.arange(0, BC)
    v = tl.load(SRC + p[:, None].to(tl.int64) * C + h[None, :], mask=ok[:, None])
    tl.store(OUT + s64[:, None] * C + h[None, :], v, mask=ok[:, None])
    if HAS_EXT:
        if pc == 0:
            tl.store(EXT_OUT + s64, tl.load(EXT + p, mask=ok), mask=ok)


def pack_pool_rows(
    src: torch.Tensor,
    p2s: torch.Tensor,
    preal: torch.Tensor,
    nslots: int,
    ext: torch.Tensor = None,
    br: int = 16,
    bc: int = 1024,
):
    """src [P, C] pool rows -> out [nslots, C] with out[p2s[p]] = src[p] (live pool rows; other slots unwritten);
    ``ext`` [P] fp32 -> ext_out [nslots] the same way (else None)."""
    P, C = src.shape
    assert (
        src.is_contiguous() and p2s.dtype == torch.int32 and p2s.numel() == P and P % br == 0 and C % 128 == 0
    )
    bc = min(bc, C)
    while C % bc:
        bc //= 2
    out = torch.empty((nslots, C), device=src.device, dtype=src.dtype)
    ext_out = None
    if ext is not None:
        assert ext.dtype == torch.float32 and ext.numel() == P
        ext_out = torch.empty(nslots, device=src.device, dtype=torch.float32)
    _pack_pool_rows_kernel[(P // br, C // bc)](
        src,
        p2s,
        out,
        ext if ext is not None else src,
        ext_out if ext_out is not None else out,
        preal,
        C,
        BR=br,
        BC=bc,
        HAS_EXT=int(ext is not None),
        num_warps=4,
    )
    return out, ext_out
