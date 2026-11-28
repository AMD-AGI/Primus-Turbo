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

import os

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
def _dp_src_kernel(
    X,
    IDX,
    TW,
    Q,
    PAY,
    ITR,
    NTR,
    T,
    H,
    stride_x,
    W,
    K: tl.constexpr,
    KP: tl.constexpr,
    R: tl.constexpr,
    EL: tl.constexpr,
    BT: tl.constexpr,
    ROUTES: tl.constexpr,
    FP8_MAX: tl.constexpr,
):
    """grid (cdiv(T, BT), H // 128). The DeepEP source side in one launch: Q [T, H] fp8 = the 1x128 row quant of
    quant_rows_send (same expressions, same bytes), its E8M0 bytes into PAY int32 [T, W] (byte pc of each row: the
    s.view(int32) words); ROUTES (column block 0): PAY[t, H/512 + k] = global expert id, PAY[t, H/512 + K + k] =
    fp32 route weight bits, ITR u8 [T, R] (token has a route on rank d) and NTR [R] += tokens per rank (int atomics,
    exact; NTR zeroed by the caller)."""
    pt = tl.program_id(0)
    pc = tl.program_id(1)
    t = pt * BT + tl.arange(0, BT)
    c = pc * 128 + tl.arange(0, 128)
    tm = t < T
    v = tl.load(X + t[:, None].to(tl.int64) * stride_x + c[None, :], mask=tm[:, None], other=0.0)
    v = v.to(tl.float32)
    av = tl.abs(v)
    er = _ue8m0_exp(tl.max(av, axis=1), FP8_MAX)
    q = (v * pow2_neg_fly(er)[:, None]).to(Q.dtype.element_ty)
    b = ((er + 127) & 0xFF).to(tl.uint8)
    tl.store(Q + t[:, None].to(tl.int64) * H + c[None, :], q, mask=tm[:, None])
    tl.store(PAY.to(tl.pointer_type(tl.uint8)) + t.to(tl.int64) * (W * 4) + pc, b, mask=tm)
    if ROUTES:
        if pc == 0:
            k = tl.arange(0, KP)
            m = tm[:, None] & (k[None, :] < K)
            off = t[:, None].to(tl.int64) * K + k[None, :]
            idx = tl.load(IDX + off, mask=m, other=0)
            w = tl.load(TW + off, mask=m, other=0.0).to(tl.float32)
            prow = PAY + t[:, None].to(tl.int64) * W + H // 512 + k[None, :]
            tl.store(prow, idx.to(tl.int32), mask=m)
            tl.store(prow + K, w.to(tl.int32, bitcast=True), mask=m)
            dst = tl.where(m, idx // EL, -1).to(tl.int32)
            d = tl.arange(0, R)
            has = tl.max((dst[:, :, None] == d[None, None, :]).to(tl.int32), axis=1)  # [BT, R]
            tl.store(ITR + t[:, None].to(tl.int64) * R + d[None, :], has.to(tl.uint8), mask=tm[:, None])
            tl.atomic_add(NTR + d, tl.sum(has, axis=0))


def dp_src_pack(x: torch.Tensor, idx: torch.Tensor = None, tw: torch.Tensor = None, R: int = 1, E_l: int = 1,
                dtype=torch.float8_e4m3fn, bt: int = 64):
    """DeepEP source side: x [T, H] bf16 -> q [T, H] fp8 (quant_rows_send's row quant) and the payload int32 [T, W]
    (E8M0 bytes in the first H/512 words; with ``idx`` int [T, K] global expert ids / ``tw`` [T, K] weights also the
    ids and the fp32 weight bits, W = H/512 + 2K). With idx: also itr bool [T, R], ntr int32 [R] (tokens per rank).
    Returns (q, payload) or (q, payload, itr, ntr)."""
    T, H = x.shape
    assert H % 512 == 0 and x.stride(1) == 1
    dev = x.device
    nsw = H // 512
    routes = idx is not None
    K = idx.shape[1] if routes else 0
    W = nsw + 2 * K
    q = torch.empty((T, H), device=dev, dtype=dtype)
    pay = torch.empty((T, W), device=dev, dtype=torch.int32)
    if routes:
        idx = idx.contiguous()
        tw = tw.contiguous()
        assert R & (R - 1) == 0 and idx.shape == tw.shape
        itr = torch.empty((T, R), device=dev, dtype=torch.bool)
        ntr = torch.zeros(R, device=dev, dtype=torch.int32)
    else:
        itr = ntr = None
    if T > 0:
        _dp_src_kernel[(triton.cdiv(T, bt), H // 128)](
            x, idx if routes else pay, tw if routes else pay, q, pay,
            itr.view(torch.uint8) if routes else pay, ntr if routes else pay,
            T, H, x.stride(0), W,
            K=max(K, 1), KP=triton.next_power_of_2(max(K, 1)), R=R, EL=E_l, BT=bt, ROUTES=int(routes),
            FP8_MAX=torch.finfo(dtype).max, num_warps=4,
        )
    return (q, pay, itr, ntr) if routes else (q, pay)


@triton.jit
def _dp_route_load(RP, W, IOFF, CNTP, ME_EL, r, m, K: tl.constexpr, EL: tl.constexpr):
    """Route r = (n, k) of DeepEP received row n: (n, local expert id or -1, live). Rows past the live count
    CNTP[0] are unwritten DeepEP slots (masked)."""
    n = r // K
    cnt = tl.load(CNTP)
    live0 = m & (n < cnt)
    g = tl.load(RP + n.to(tl.int64) * W + IOFF + r % K, mask=live0, other=-1)
    loc = g - ME_EL
    ok = live0 & (loc >= 0) & (loc < EL)
    return n, tl.where(ok, loc, -1), ok


@triton.jit
def _dp_route_count_kernel(
    RP, W, IOFF, CNTP, ME_EL, CNT, ROWMAP, RW, R, P, NCH,
    K: tl.constexpr, E: tl.constexpr, BLOCK: tl.constexpr, PB: tl.constexpr,
):
    """_route_count_meta_kernel over the DeepEP payload; CNT [E, NCH] (expert-major: the offsets kernel reads a
    contiguous column per expert); resets ROWMAP -1 / RW 0 over this program's share of P."""
    ch = tl.program_id(0)
    e_ar = tl.arange(0, E)
    r = ch * BLOCK + tl.arange(0, BLOCK)
    m = r < R
    n, idx, ok = _dp_route_load(RP, W, IOFF, CNTP, ME_EL, r, m, K, E)
    tl.store(CNT + e_ar * NCH + ch, tl.sum((idx[:, None] == e_ar[None, :]).to(tl.int32), axis=0))
    pr = ch * PB + tl.arange(0, PB)
    tl.store(ROWMAP + pr, tl.full((PB,), -1, tl.int32), mask=pr < P)
    tl.store(RW + pr, tl.zeros((PB,), tl.float32), mask=pr < P)


@triton.jit
def _dp_route_offs_kernel(CNT, BASE, TOT, NCH, NCP: tl.constexpr):
    """grid (E,): expert e's exclusive prefix over the chunks BASE[e, :] and its route total TOT[e] (one program per
    expert instead of _route_offs_kernel's single program over all [nchunk, E])."""
    e = tl.program_id(0)
    c_ar = tl.arange(0, NCP)
    m = c_ar < NCH
    c = tl.load(CNT + e * NCH + c_ar, mask=m, other=0)
    tl.store(BASE + e * NCH + c_ar, tl.cumsum(c, 0) - c, mask=m)
    tl.store(TOT + e, tl.sum(c, 0))


@triton.jit
def _dp_route_scatter_kernel(
    RP, W, IOFF, WOFF, CNTP, ME_EL, BASE, TOT, POFFS, ROWMAP, RW, INV, R, NCH, WOK, WCAP,
    E: tl.constexpr, K: tl.constexpr, BLOCK: tl.constexpr, ALIGN: tl.constexpr, HAS_WOK: tl.constexpr,
):
    """_route_scatter_meta_kernel over the DeepEP payload, with the padded expert starts (the _route_offs_kernel
    formula) recomputed from TOT in every program; program 0 writes POFFS."""
    ch = tl.program_id(0)
    e_ar = tl.arange(0, E)
    tot = tl.load(TOT + e_ar)
    plen = tl.maximum((tot + ALIGN - 1) // ALIGN, 1) * ALIGN
    pend = tl.cumsum(plen, 0)
    pstart = pend - plen
    if ch == 0:
        tl.store(POFFS + 1 + e_ar, pend)
        tl.store(POFFS + e_ar, pstart, mask=e_ar == 0)
        if HAS_WOK:  # the wgrad window check (max expert window <= WCAP) as one flag
            tl.store(WOK, (tl.max(plen, 0) <= WCAP).to(tl.uint8))
    r = ch * BLOCK + tl.arange(0, BLOCK)
    m = r < R
    n, idx, live = _dp_route_load(RP, W, IOFF, CNTP, ME_EL, r, m, K, E)
    oh = (idx[:, None] == e_ar[None, :]).to(tl.int32)
    base = pstart + tl.load(BASE + e_ar * NCH + ch)
    pos = tl.sum(oh * (base[None, :] + tl.cumsum(oh, 0) - 1), axis=1)
    tl.store(ROWMAP + pos, n, mask=live)
    tw = tl.load(RP + n.to(tl.int64) * W + WOFF + r % K, mask=live, other=0).to(tl.float32, bitcast=True)
    tl.store(RW + pos, tw, mask=live)
    tl.store(INV + r, tl.where(live, pos, -1), mask=m)


def build_routing_dp(
    rp: torch.Tensor,
    ioff: int,
    woff: int,
    cntp: torch.Tensor,
    me: int,
    N: int,
    K: int,
    num_experts: int,
    P: int,
    align: int = 256,
    block: int = None,
    wcap: int = 0,
):
    """build_routing_blk_meta straight from DeepEP's received payload rp int32 [N, W] (global expert ids at column
    ``ioff``, fp32 weight bits at ``woff``; rank ``me`` owns experts [me*E, (me+1)*E)), live rows n < cntp[0] (a
    one-element view of the rank prefix matrix): no materialized metadata, and the chunk prefix in E parallel
    programs. Same outputs, bit for bit, as build_routing_blk_meta over the equivalent metadata (p2s None).
    ``wcap`` > 0: also a bool [1] flag, max(poffs[1:] - poffs[:-1]) <= wcap (7th output; else None)."""
    E = num_experts
    assert E & (E - 1) == 0, "local expert count must be a power of two (tl.arange)"
    assert rp.dtype == torch.int32 and rp.is_contiguous() and rp.shape[0] == N and cntp.dtype == torch.int32
    W = rp.shape[1]
    R = N * K
    dev = rp.device
    if block is None:  # same chunking rule as build_routing_blk_masked (layout is chunking-independent)
        block = max(128, min(512, triton.next_power_of_2(triton.cdiv(R, 128))))
    nw = 4 if block <= 128 else 8
    nch = triton.cdiv(R, block)
    pb = triton.next_power_of_2(triton.cdiv(P, nch))
    poffs = torch.empty(E + 1, dtype=torch.int32, device=dev)
    rowmap = torch.empty(P, dtype=torch.int32, device=dev)
    rw = torch.empty(P, dtype=torch.float32, device=dev)
    inv = torch.empty(R, dtype=torch.int32, device=dev)
    cnt = torch.empty((E, nch), dtype=torch.int32, device=dev)
    base = torch.empty((E, nch), dtype=torch.int32, device=dev)
    tot = torch.empty(E, dtype=torch.int32, device=dev)
    me_el = me * E
    wok = torch.empty(1, dtype=torch.bool, device=dev) if wcap > 0 else None
    _dp_route_count_kernel[(nch,)](
        rp, W, ioff, cntp, me_el, cnt, rowmap, rw, R, P, nch, K=K, E=E, BLOCK=block, PB=pb, num_warps=nw
    )
    _dp_route_offs_kernel[(E,)](cnt, base, tot, nch, NCP=triton.next_power_of_2(nch), num_warps=4)
    _dp_route_scatter_kernel[(nch,)](
        rp, W, ioff, woff, cntp, me_el, base, tot, poffs, rowmap, rw, inv, R, nch,
        wok.view(torch.uint8) if wok is not None else tot, max(wcap, 0), E=E, K=K, BLOCK=block,
        ALIGN=align, HAS_WOK=int(wok is not None), num_warps=nw,
    )
    return rowmap, rw, inv, poffs, poffs[E:], None, wok


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
    if block is None:  # at most ~128 chunks: the offsets kernel scans all chunks (x E experts) in one program
        block = max(128, min(512, triton.next_power_of_2(triton.cdiv(R, 128))))
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
def _meta_route(META, META_OFF, SRC_STRIDE, CNTS, CNT_OFF, CNT_STRIDE, CAP, r, m, K: tl.constexpr):
    """Route r = (n, k) of received slot n = (source, slot): its metadata row offset and whether the slot is live
    (slot < the source's count; rows past the count are unwritten)."""
    n = r // K
    src = n // CAP
    slot = n % CAP
    c = tl.load(CNTS + CNT_OFF + src * CNT_STRIDE, mask=m, other=0)
    live = m & (slot < c)
    row = META_OFF + src.to(tl.int64) * SRC_STRIDE + slot.to(tl.int64) * (3 * K)
    return n, row, live


@triton.jit
def _route_count_meta_kernel(
    META,
    META_OFF,
    SRC_STRIDE,
    CNTS,
    CNT_OFF,
    CNT_STRIDE,
    CAP,
    CNT,
    ROWMAP,
    RW,
    P2S,
    R,
    P,
    K: tl.constexpr,
    E: tl.constexpr,
    BLOCK: tl.constexpr,
    PB: tl.constexpr,
    HAS_CS: tl.constexpr,
):
    """_route_count_kernel over the received route metadata (local expert ids in columns [0, K) of each live slot's
    row, -1 for dead slots); also resets ROWMAP -1, RW 0 (and P2S -1) over this program's share of P."""
    ch = tl.program_id(0)
    e_ar = tl.arange(0, E)
    r = ch * BLOCK + tl.arange(0, BLOCK)
    m = r < R
    n, row, live = _meta_route(META, META_OFF, SRC_STRIDE, CNTS, CNT_OFF, CNT_STRIDE, CAP, r, m, K)
    idx = tl.load(META + row + r % K, mask=live, other=-1)
    tl.store(CNT + ch * E + e_ar, tl.sum((idx[:, None] == e_ar[None, :]).to(tl.int32), axis=0))
    pr = ch * PB + tl.arange(0, PB)
    tl.store(ROWMAP + pr, tl.full((PB,), -1, tl.int32), mask=pr < P)
    tl.store(RW + pr, tl.zeros((PB,), tl.float32), mask=pr < P)
    if HAS_CS:
        tl.store(P2S + pr, tl.full((PB,), -1, tl.int32), mask=pr < P)


@triton.jit
def _route_scatter_meta_kernel(
    META,
    META_OFF,
    SRC_STRIDE,
    CNTS,
    CNT_OFF,
    CNT_STRIDE,
    CAP,
    BASE,
    ROWMAP,
    RW,
    INV,
    P2S,
    R,
    CCAP,
    E: tl.constexpr,
    K: tl.constexpr,
    BLOCK: tl.constexpr,
    HAS_CS: tl.constexpr,
):
    """_route_scatter_masked_kernel over the received route metadata: ids in columns [0, K), route weights (fp32
    bits) in [K, 2K), combine slots in [2K, 3K) of each live slot's row."""
    ch = tl.program_id(0)
    e_ar = tl.arange(0, E)
    r = ch * BLOCK + tl.arange(0, BLOCK)
    m = r < R
    n, row, live0 = _meta_route(META, META_OFF, SRC_STRIDE, CNTS, CNT_OFF, CNT_STRIDE, CAP, r, m, K)
    kk = r % K
    idx = tl.load(META + row + kk, mask=live0, other=-1)
    oh = (idx[:, None] == e_ar[None, :]).to(tl.int32)
    base = tl.load(BASE + ch * E + e_ar)
    pos = tl.sum(oh * (base[None, :] + tl.cumsum(oh, 0) - 1), axis=1)
    live = live0 & (idx >= 0)
    tl.store(ROWMAP + pos, n, mask=live)
    tw = tl.load(META + row + K + kk, mask=live, other=0).to(tl.float32, bitcast=True)
    tl.store(RW + pos, tw, mask=live)
    tl.store(INV + r, tl.where(live, pos, -1), mask=m)
    if HAS_CS:
        cs = tl.load(META + row + 2 * K + kk, mask=live, other=0)
        tl.store(P2S + pos, (n // CAP) * CCAP + cs, mask=live)


def build_routing_blk_meta(
    meta: torch.Tensor,
    meta_off: int,
    src_stride: int,
    cnts: torch.Tensor,
    cnt_off: int,
    cnt_stride: int,
    N: int,
    K: int,
    num_experts: int,
    P: int,
    cap: int,
    ccap: int = 0,
    align: int = 256,
    block: int = None,
):
    """build_routing_blk_masked straight from the received route metadata, without materializing the masked id /
    weight / slot columns: ``meta`` int32 (flat), the [cap, 3K] metadata block of source s at element
    ``meta_off + s * src_stride`` (columns: local expert id or -1, route weight as fp32 bits, combine slot);
    ``cnts`` int32 (flat), source s's live slot count at ``cnt_off + s * cnt_stride``. N = sources * cap received
    slots. ``ccap`` > 0: also p2s. Same outputs, bit for bit, as build_routing_blk_masked of the masked columns."""
    R = N * K
    E = num_experts
    assert E & (E - 1) == 0, "local expert count must be a power of two (tl.arange)"
    assert meta.dtype == torch.int32 and cnts.dtype == torch.int32 and meta.is_contiguous() and cnts.is_contiguous()
    dev = meta.device
    if block is None:  # same chunking rule as build_routing_blk_masked (layout is chunking-independent)
        block = max(128, min(512, triton.next_power_of_2(triton.cdiv(R, 128))))
    nw = 4 if block <= 128 else 8
    nch = triton.cdiv(R, block)
    pb = triton.next_power_of_2(triton.cdiv(P, nch))
    has_cs = int(ccap > 0)
    poffs = torch.empty(E + 1, dtype=torch.int32, device=dev)
    rowmap = torch.empty(P, dtype=torch.int32, device=dev)
    rw = torch.empty(P, dtype=torch.float32, device=dev)
    inv = torch.empty(R, dtype=torch.int32, device=dev)
    cnt = torch.empty((nch, E), dtype=torch.int32, device=dev)
    base = torch.empty((nch, E), dtype=torch.int32, device=dev)
    p2s = torch.empty(P, dtype=torch.int32, device=dev) if has_cs else None
    src = (meta, meta_off, src_stride, cnts, cnt_off, cnt_stride, cap)
    _route_count_meta_kernel[(nch,)](
        *src, cnt, rowmap, rw, p2s if has_cs else rowmap, R, P, K=K, E=E, BLOCK=block, PB=pb, HAS_CS=has_cs,
        num_warps=nw,
    )
    _route_offs_kernel[(1,)](
        cnt, base, poffs, nch, E=E, NCP=triton.next_power_of_2(nch), ALIGN=align, num_warps=4
    )
    _route_scatter_meta_kernel[(nch,)](
        *src, base, rowmap, rw, inv, p2s if has_cs else rowmap, R, max(ccap, 1), E=E, K=K, BLOCK=block,
        HAS_CS=has_cs, num_warps=nw,
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
    XS_STRIDE,
    FP8_MAX: tl.constexpr,
    HAS_PREAL: tl.constexpr,
    CT: tl.constexpr = 0,
    SL: tl.constexpr = 0,
    SQ: tl.constexpr = 7,
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
    er = tl.load(XS + src * XS_STRIDE + pc, mask=valid, other=127).to(tl.int32) - 127
    if SQ & 1:
        tl.store(Q + p[:, None].to(tl.int64) * C + c[None, :], qb)
    if SQ & 2:
        store_scale(S, p, pc, er, P, C // 128, SL)
    if SQ & 4:
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


@triton.jit
def _gather_dual_fp8_tile_kernel(
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
    XS_STRIDE,
    FP8_MAX: tl.constexpr,
    HAS_PREAL: tl.constexpr,
    CT: tl.constexpr,
    SL: tl.constexpr,
    RB: tl.constexpr,
    CB: tl.constexpr,
    SQ: tl.constexpr = 7,
):
    """_gather_dual_fp8_kernel on a (RB*128) x (CB*128) tile: grid (P // (RB*128), C // (CB*128)). Per element the
    same math (row scale of its 128-column block, column scale over its 128-row block), so bit-identical; the wider
    tile gives longer contiguous runs for the gathered row loads / Q stores (CB) and the QT stores (RB)."""
    BR: tl.constexpr = RB * 128
    BC: tl.constexpr = CB * 128
    pt = tl.program_id(0)
    pc = tl.program_id(1)
    if HAS_PREAL:
        if pt * BR >= tl.load(PREAL):
            return
    p = pt * BR + tl.arange(0, BR)
    c = pc * BC + tl.arange(0, BC)
    kb = pc * CB + tl.arange(0, CB)
    src = tl.load(ROWMAP + p)
    valid = src >= 0
    src = tl.where(valid, src, 0).to(tl.int64)
    qb = tl.load(XQ + src[:, None] * C + c[None, :], mask=valid[:, None], other=0)  # uint8 bytes
    er = tl.load(XS + src[:, None] * XS_STRIDE + kb[None, :], mask=valid[:, None], other=127).to(tl.int32) - 127
    if SQ & 1:
        tl.store(Q + p[:, None].to(tl.int64) * C + c[None, :], qb)
    if SQ & 2:
        store_scale(S, p[:, None], kb[None, :], er, P, C // 128, SL)
    if SQ & 4:
        v = tl.reshape(qb.to(tl.float8e4nv, bitcast=True).to(tl.float32), (BR, CB, 128)) * pow2_pos(er)[:, :, None]
        v = tl.reshape(v, (RB, 128, BC))
        ec = _ue8m0_exp(tl.max(tl.abs(v), axis=1), FP8_MAX)  # [RB, BC]
        qc = tl.reshape(v * pow2_neg_fly(ec)[:, None, :], (BR, BC)).to(QT.dtype.element_ty)
        u = tl.reshape(qc.to(tl.uint8, bitcast=True).to(tl.int32), (BR // 4, 4, BC))
        w = tl.sum(u << (tl.arange(0, 4) * 8)[None, :, None], axis=1)  # [BR/4, BC]: 4 pool rows of a column per int32
        pw = pt * (BR // 4) + tl.arange(0, BR // 4)
        tl.store(QT.to(tl.pointer_type(tl.int32)) + c[:, None].to(tl.int64) * (P // 4) + pw[None, :], tl.trans(w))
        rb = pt * RB + tl.arange(0, RB)
        if CT:
            tl.store(ST + c[:, None].to(tl.int64) * (P // 128) + rb[None, :], tl.trans((ec + 127) & 0xFF).to(tl.uint8))
        else:
            store_scale(ST, c[:, None], rb[None, :], tl.trans(ec), C, P // 128, SL)


def _gather_tile():
    """PRIMUS_TURBO_GATHER_TILE="RB,CB[,num_warps]" of gather_dual_fp8 (default "2,1,8": 256 x 128 tiles, 256-B QT
    runs; MBS4 EP4 shape 262 -> 236 us/call, bit-identical; "1,1" = the 128 x 128 _gather_dual_fp8_kernel)."""
    v = os.environ.get("PRIMUS_TURBO_GATHER_TILE", "2,1,8").split(",")
    return int(v[0]), int(v[1]), int(v[2]) if len(v) > 2 else 8


def gather_dual_fp8(
    xq: torch.Tensor,
    xs: torch.Tensor,
    P: int,
    rowmap: torch.Tensor,
    preal: torch.Tensor = None,
    compact_t: bool = False,
    scale_layout="x4",
    nw: int = 8,
    store_q: bool = True,
    parts: str = "qst",
    out=None,
):
    """xq [N, C] fp8 received rows, xs [N, C/128] uint8 their E8M0 bytes -> pool-ordered (q [P, C], s [P, C/32],
    qT [C, P], sT [C, P/32] or [C, P/128] when ``compact_t``): quant_dual_blk's outputs and layouts, with the row
    quant taken over unchanged and the 128x1 quant re-derived from the FP8 values. ``store_q=False``: q is not
    written (returned as None) -- for a consumer that gathers the rows itself (mxgp_fp8blk_nt(rowmap=...)).
    ``parts``: which outputs are written ("q" rows, "s" row scales, "t" the 128x1 copy + its scales; the others are
    not allocated and returned as None unless ``out`` = (s, qT, sT) gives the buffers, e.g. the 128x1 copy computed
    later on a side stream into buffers allocated now). Every written byte is the same as in the full call."""
    N, C = xq.shape
    assert P % 128 == 0 and C % 128 == 0 and xq.is_contiguous()
    # xs: [N, C/128] with unit column stride and any row stride (e.g. the scale bytes inside a DeepEP payload row)
    assert xs.dtype == torch.uint8 and xs.shape == (N, C // 128) and xs.stride(1) == 1 and rowmap.dtype == torch.int32
    sl = scale_layout_id(scale_layout)
    if not store_q:
        parts = parts.replace("q", "")
    m = ("q" in parts) * 1 + ("s" in parts) * 2 + ("t" in parts) * 4
    if out is not None:
        q = None
        s, qT, sT = out
    else:
        q, s, qT, sT = _dual_alloc(P, C, xq.dtype, sl, compact_t, xq.device, alloc_q=bool(m & 1))
        if not m & 2:
            s = None
        if not m & 4:
            qT = sT = None
    dummy = next(t for t in (q, s, qT, sT) if t is not None)
    qa = q if q is not None else dummy
    s_, qT_, sT_ = (t if t is not None else dummy for t in (s, qT, sT))
    rb, cb, tnw = _gather_tile()
    # wide tiles for the x4 / byte scale layouts (validated); the packed layout keeps the 128 x 128 kernel
    if (rb, cb) != (1, 1) and sl in (0, 1) and P % (rb * 128) == 0 and C % (cb * 128) == 0:
        _gather_dual_fp8_tile_kernel[(P // (rb * 128), C // (cb * 128))](
            xq.view(torch.uint8),
            xs,
            rowmap,
            qa.view(torch.uint8),
            s_,
            qT_,
            sT_,
            P,
            C,
            preal if preal is not None else rowmap,
            xs.stride(0),
            FP8_MAX=torch.finfo(xq.dtype).max,
            HAS_PREAL=int(preal is not None),
            CT=int(compact_t),
            SL=sl,
            RB=rb,
            CB=cb,
            SQ=m,
            num_warps=tnw,
        )
        return q, s, qT, sT
    _gather_dual_fp8_kernel[(P // 128, C // 128)](
        xq.view(torch.uint8),
        xs,
        rowmap,
        qa.view(torch.uint8),
        s_,
        qT_,
        sT_,
        P,
        C,
        preal if preal is not None else rowmap,
        xs.stride(0),
        FP8_MAX=torch.finfo(xq.dtype).max,
        HAS_PREAL=int(preal is not None),
        CT=int(compact_t),
        SL=sl,
        SQ=m,
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


@triton.jit
def _presum_routes_kernel(
    ROWS, INV, OUT, H, EXT, P2S, EXT_OUT, K: tl.constexpr, BH: tl.constexpr, HAS_EXT: tl.constexpr = 0
):
    """grid (N, H // BH). OUT[n] = sum over the live routes k of received token row n of ROWS[INV[n*K+k]] (fp32, k
    order, one bf16 rounding); rows without a live route are left unwritten (no source reads them).
    HAS_EXT: the first column block also writes EXT_OUT[P2S[p]] = EXT[p] for each live route's pool row p."""
    n = tl.program_id(0)
    h = tl.program_id(1) * BH + tl.arange(0, BH)
    # Branch-free: each route's row load is masked (no route -> +0.0), so the K row loads are in flight together
    # instead of each waiting on its index test. Same bits: acc starts at +0.0 and an fp32 sum never turns into -0.0
    # by adding +0.0 (round to nearest), so the masked adds leave it unchanged; the live routes add in k order.
    acc = tl.zeros((BH,), dtype=tl.float32)
    live = 0
    for k in tl.static_range(K):
        p = tl.load(INV + n * K + k)
        acc += tl.load(ROWS + tl.maximum(p, 0).to(tl.int64) * H + h, mask=p >= 0, other=0.0).to(tl.float32)
        live += (p >= 0).to(tl.int32)
    if HAS_EXT != 0:
        if tl.program_id(1) == 0:
            for k in tl.static_range(K):
                p = tl.load(INV + n * K + k)
                q = tl.maximum(p, 0)
                if HAS_EXT == 1:
                    tl.store(EXT_OUT + tl.load(P2S + q), tl.load(EXT + q), mask=p >= 0)
                if HAS_EXT == 2:  # per received route: EXT_OUT[n, k] = EXT[p] (0 for no route on this rank)
                    tl.store(EXT_OUT + n * K + k, tl.load(EXT + q, mask=p >= 0, other=0.0))
    if live > 0:
        tl.store(OUT + n.to(tl.int64) * H + h, acc.to(OUT.dtype.element_ty))


def presum_routes(
    rows: torch.Tensor,
    inv: torch.Tensor,
    N: int,
    K: int,
    bh: int = 1024,
    ext: torch.Tensor = None,
    p2s: torch.Tensor = None,
    nslots: int = 0,
):
    """Expert rank of the EP combine: rows [P, H] pool rows (routing weight applied), inv int32 [N*K] pool row of
    received route (n, k) or -1 -> out [N, H]: per received token, the sum of its routes on this rank.
    ``ext`` fp32 [P] (with ``p2s``, ``nslots``): also -> ext_out [nslots] by combine slot (scatter_pool_ext of the
    live routes; other slots unwritten); ``ext`` without ``p2s``: ext_out fp32 [N, K] = ext[inv] (0 where inv < 0).
    Returns out, or (out, ext_out)."""
    P, H = rows.shape
    assert rows.is_contiguous() and inv.dtype == torch.int32 and inv.numel() == N * K and H % 128 == 0
    bh = min(bh, H & -H)
    out = torch.empty((N, H), device=rows.device, dtype=rows.dtype)
    ext_out = None
    mode = 0
    if ext is not None:
        assert ext.dtype == torch.float32 and ext.numel() == P and ext.is_contiguous()
        if p2s is None:
            mode = 2
            ext_out = torch.empty((N, K), device=rows.device, dtype=torch.float32)
        else:
            assert p2s.numel() == P
            mode = 1
            ext_out = torch.empty(nslots, device=rows.device, dtype=torch.float32)
    _presum_routes_kernel[(N, H // bh)](
        rows,
        inv,
        out,
        H,
        ext if ext is not None else rows,
        p2s if p2s is not None and ext is not None else inv,
        ext_out if ext is not None else out,
        K=K,
        BH=bh,
        HAS_EXT=mode,
        num_warps=int(os.environ.get("PRIMUS_TURBO_EP_PRESUM_NW", "2")),  # 2: 134 -> 112 us at the MBS4 EP4 shape
    )
    return out if ext is None else (out, ext_out)


@triton.jit
def _combine_ranks_kernel(
    RECV, SLOT, OUT, H, CAP, RWS, CINV, DTW, R: tl.constexpr, BH: tl.constexpr, K: tl.constexpr = 1,
    HAS_EXT: tl.constexpr = 0,
):
    """grid (T, H // BH). OUT[t] = sum over destination ranks r that token t was sent to (SLOT[t, r] >= 0) of
    RECV[r, SLOT[t, r]] (fp32, rank order, one bf16 rounding). HAS_EXT: the first column block also gathers
    DTW[t, k] = RWS[CINV[t*K+k]] (the per-route values received by combine slot)."""
    t = tl.program_id(0)
    h = tl.program_id(1) * BH + tl.arange(0, BH)
    acc = tl.zeros((BH,), dtype=tl.float32)
    for r in tl.static_range(R):
        s = tl.load(SLOT + t * R + r)
        if s >= 0:
            acc += tl.load(RECV + (r * CAP + s).to(tl.int64) * H + h).to(tl.float32)
    tl.store(OUT + t.to(tl.int64) * H + h, acc.to(OUT.dtype.element_ty))
    if HAS_EXT:
        if tl.program_id(1) == 0:
            for k in tl.static_range(K):
                tl.store(DTW + t * K + k, tl.load(RWS + tl.load(CINV + t * K + k)))


def combine_ranks(
    recv: torch.Tensor,
    slot: torch.Tensor,
    T: int,
    R: int,
    cap: int,
    bh: int = 1024,
    rws: torch.Tensor = None,
    cinv: torch.Tensor = None,
    K: int = 1,
):
    """Source rank of the EP combine: recv [R*cap, H] (block r: the per-token partial sums from expert rank r),
    slot int32 [T, R] -> y [T, H]. ``rws`` fp32 [R*ccap] (per-route values by combine slot) with ``cinv`` int32
    [T*K]: also -> [T, K] = rws[cinv]. Returns y, or (y, [T, K])."""
    H = recv.shape[1]
    assert recv.is_contiguous() and slot.dtype == torch.int32 and slot.shape == (T, R) and H % 128 == 0
    bh = min(bh, H & -H)
    out = torch.empty((T, H), device=recv.device, dtype=recv.dtype)
    dtw = None
    if rws is not None:
        assert rws.dtype == torch.float32 and rws.is_contiguous() and cinv.dtype == torch.int32 and cinv.numel() == T * K
        dtw = torch.empty((T, K), device=recv.device, dtype=torch.float32)
    _combine_ranks_kernel[(T, H // bh)](
        recv,
        slot.contiguous(),
        out,
        H,
        cap,
        rws if rws is not None else recv,
        cinv if rws is not None else slot,
        dtw if rws is not None else out,
        R=R,
        BH=bh,
        K=K,
        HAS_EXT=int(rws is not None),
        num_warps=4,
    )
    return out if rws is None else (out, dtw)


def scatter_pool_ext(ext: torch.Tensor, p2s: torch.Tensor, nslots: int) -> torch.Tensor:
    """ext [P] fp32 per pool row -> [nslots] by combine slot (p2s, -1 = padding row); other slots unwritten/zero."""
    out = torch.zeros(nslots + 1, device=ext.device, dtype=torch.float32)
    idx = torch.where(p2s >= 0, p2s, nslots).long()
    out.index_put_((idx,), ext)
    return out[:nslots]
