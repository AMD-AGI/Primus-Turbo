###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Sort-free, deterministic dKV gather for a "window + pool" top-k map (V4 CSA, B == 1).

The map of token ``t`` is ``W`` sliding-window slots (slot ``s`` = kv row ``t - W + 1 + s``,
``-1`` before the sequence start) followed by pool slots holding kv rows ``n_raw + p``
(``0 <= p < P``) or ``-1``, each pool row at most once per token (an indexer top-k).

Instead of the CSR inverse (a radix sort of every entry), the window rows are reduced in
closed form (row ``j`` sums slot ``j - t + W - 1`` of tokens ``t = j .. j + W - 1``) and the
pool picks are scattered into a dense ``pos[p, t]`` slot table (int16, ``P x T``), which the
pool reduction scans in token order. Every row sums its entries in ascending token order,
segment partials in segment order: bitwise reproducible.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _pool_pos_kernel(Idx, Pos, t0, n_tok, T, TOPK, W, n_raw, P, BLOCK: tl.constexpr):
    # One program per token of this chunk: pos[p, t] = slot of pool row p in token t's map.
    i = tl.program_id(0)
    t = t0 + i
    offs = tl.arange(0, BLOCK)
    m = offs < TOPK - W
    r = tl.load(Idx + i.to(tl.int64) * TOPK + W + offs, mask=m, other=-1)
    p = r - n_raw
    ok = m & (r >= 0) & (p >= 0) & (p < P)
    tl.store(Pos + p.to(tl.int64) * T + t, (W + offs).to(tl.int16), mask=ok)


@triton.jit
def _window_gather_kernel(Interm, Acc, t0, t1, W, TOPK, stride_acc, WP: tl.constexpr, DB: tl.constexpr):
    # kv row j (window part), columns [c * DB, (c + 1) * DB): += sum over the tokens t of this
    # chunk with j in t's window, all W rows loaded at once (ascending t).
    j = t0 - W + 1 + tl.program_id(0)
    c = tl.program_id(1)
    if j >= 0:
        offs_d = c * DB + tl.arange(0, DB)
        t = j + tl.arange(0, WP)
        m = (t >= t0) & (t < t1) & (t < j + W)
        ent = (t - t0).to(tl.int64) * TOPK + (j - t + W - 1)
        rows = tl.load(Interm + ent[:, None] * 512 + offs_d[None, :], mask=m[:, None], other=0.0)
        acc = tl.sum(rows.to(tl.float32), axis=0)
        a_ptr = Acc + j.to(tl.int64) * stride_acc + offs_d
        tl.store(a_ptr, tl.load(a_ptr) + acc)


@triton.jit
def _pool_count_kernel(Pos, Cnt, t0, t1, T, BLOCK: tl.constexpr):
    # Cnt[p] = number of tokens in [t0, t1) that picked pool row p.
    p = tl.program_id(0)
    c = tl.zeros([BLOCK], dtype=tl.int32)
    for tb in range(t0, t1, BLOCK):
        t = tb + tl.arange(0, BLOCK)
        sl = tl.load(Pos + p.to(tl.int64) * T + t, mask=t < t1, other=-1)
        c += (sl >= 0).to(tl.int32)
    tl.store(Cnt + p, tl.sum(c, axis=0))


@triton.jit
def _pool_fill_kernel(Pos, Off, Ent, t0, t1, T, TOPK, BLOCK: tl.constexpr):
    # Ent[Off[p] ..] = interm row (t - t0) * TOPK + slot of every pick of row p, ascending t.
    p = tl.program_id(0)
    base = tl.load(Off + p)
    for tb in range(t0, t1, BLOCK):
        t = tb + tl.arange(0, BLOCK)
        sl = tl.load(Pos + p.to(tl.int64) * T + t, mask=t < t1, other=-1).to(tl.int32)
        ok = sl >= 0
        oki = ok.to(tl.int32)
        pre = tl.cumsum(oki, axis=0) - oki
        tl.store(Ent + base + pre, (t - t0) * TOPK + sl, mask=ok)
        base += tl.sum(oki, axis=0)


@triton.jit
def _pool_fill_scan_kernel(Pos, Cnt, Off, Ent, t0, t1, T, TOPK, P, BLOCK: tl.constexpr, PB: tl.constexpr):
    # _pool_fill_kernel with the exclusive scan of Cnt done in place (exact integer sums): writes
    # Off[p + 1] (and Off[0]), so the separate zeros + cumsum + copies are not needed.
    p = tl.program_id(0)
    i = tl.arange(0, PB)
    base = tl.sum(tl.load(Cnt + i, mask=i < p, other=0), axis=0)
    tl.store(Off + p + 1, base + tl.load(Cnt + p))
    if p == 0:
        tl.store(Off, 0)
    for tb in range(t0, t1, BLOCK):
        t = tb + tl.arange(0, BLOCK)
        sl = tl.load(Pos + p.to(tl.int64) * T + t, mask=t < t1, other=-1).to(tl.int32)
        ok = sl >= 0
        oki = ok.to(tl.int32)
        pre = tl.cumsum(oki, axis=0) - oki
        tl.store(Ent + base + pre, (t - t0) * TOPK + sl, mask=ok)
        base += tl.sum(oki, axis=0)


@triton.jit
def _pool_gather_kernel(Interm, Ent, Off, Side, NSEG, SEG: tl.constexpr, DB: tl.constexpr):
    # Partial of pool row p over its entries [s * SEG, (s + 1) * SEG), columns block c -> side[p, s].
    p = tl.program_id(0)
    s = tl.program_id(1)
    c = tl.program_id(2)
    e0 = tl.load(Off + p) + s * SEG
    e1 = tl.load(Off + p + 1)
    offs_d = c * DB + tl.arange(0, DB)
    if e0 < e1:
        e = e0 + tl.arange(0, SEG)
        m = e < e1
        ent = tl.load(Ent + e, mask=m, other=0).to(tl.int64)
        rows = tl.load(Interm + ent[:, None] * 512 + offs_d[None, :], mask=m[:, None], other=0.0)
        acc = tl.sum(rows.to(tl.float32), axis=0)
    else:
        acc = tl.zeros([DB], dtype=tl.float32)
    tl.store(Side + (p.to(tl.int64) * NSEG + s) * 512 + offs_d, acc)


@triton.jit
def _pool_fixup_kernel(Acc, Side, n_raw, NSEG, stride_acc, D: tl.constexpr):
    p = tl.program_id(0)
    offs_d = tl.arange(0, D)
    a_ptr = Acc + (n_raw + p).to(tl.int64) * stride_acc + offs_d
    a = tl.load(a_ptr)
    for s in range(0, NSEG):
        a += tl.load(Side + (p.to(tl.int64) * NSEG + s) * D + offs_d)
    tl.store(a_ptr, a)


def band_pool_positions(topk_indices, window, n_raw, pool):
    """Dense ``pos[p, t]`` (int16, -1 = not picked) of the pool part of a window + pool map."""
    T, topk = topk_indices.shape
    assert topk - window <= 32767 and topk_indices.dtype == torch.int32
    pos = torch.full((pool, T), -1, dtype=torch.int16, device=topk_indices.device)
    if T and topk > window and pool:
        _pool_pos_kernel[(T,)](
            topk_indices,
            pos,
            0,
            T,
            T,
            topk,
            window,
            n_raw,
            pool,
            BLOCK=triton.next_power_of_2(topk - window),
            num_warps=4,
        )
    return pos


def dkv_band_gather_acc(
    interm, pos, acc, t0, t1, window, n_raw, pool, d=512, seg_t=256, db=128, nw=8, fused_scan=False
):
    """``acc[:, :d] +=`` the dKV of tokens ``[t0, t1)`` from their ``[t1 - t0, topk, d]`` bf16
    intermediate, for a window + pool map (see the module docstring). ``fused_scan``: the per-pool-row
    offsets are scanned inside the fill kernel (same values; two kernels instead of six)."""
    n = t1 - t0
    if n <= 0:
        return
    assert d == 512 and d % db == 0
    topk = interm.shape[1]
    T = pos.shape[1] if pool else t1
    _window_gather_kernel[(n + window - 1, d // db)](
        interm, acc, t0, t1, window, topk, acc.stride(0), WP=triton.next_power_of_2(window), DB=db, num_warps=nw
    )
    if pool and topk > window:
        cnt = torch.empty(pool, dtype=torch.int32, device=acc.device)
        _pool_count_kernel[(pool,)](pos, cnt, t0, t1, T, BLOCK=1024, num_warps=4)
        ent = torch.empty(n * (topk - window), dtype=torch.int32, device=acc.device)
        if fused_scan:
            off = torch.empty(pool + 1, dtype=torch.int32, device=acc.device)
            _pool_fill_scan_kernel[(pool,)](
                pos, cnt, off, ent, t0, t1, T, topk, pool, BLOCK=1024, PB=triton.next_power_of_2(pool), num_warps=4
            )
        else:
            off = torch.zeros(pool + 1, dtype=torch.int32, device=acc.device)
            off[1:] = torch.cumsum(cnt, 0)
            _pool_fill_kernel[(pool,)](pos, off, ent, t0, t1, T, topk, BLOCK=1024, num_warps=4)
        nseg = triton.cdiv(n, seg_t)
        side = torch.empty(pool, nseg, d, dtype=torch.float32, device=acc.device)
        _pool_gather_kernel[(pool, nseg, d // db)](interm, ent, off, side, nseg, SEG=seg_t, DB=db, num_warps=nw)
        _pool_fixup_kernel[(pool,)](acc, side, n_raw, nseg, acc.stride(0), D=d, num_warps=4)


@triton.jit
def _ntile_kernel(Idx, Out, n_tok, TOPK, BN, BLOCK: tl.constexpr, ROWS: tl.constexpr):
    # Out[t] = ceil((1 + last slot of token t holding a valid index) / BN) (0 when none).
    r = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    s = tl.arange(0, BLOCK)
    m = (r < n_tok)[:, None] & (s < TOPK)[None, :]
    v = tl.load(Idx + r.to(tl.int64)[:, None] * TOPK + s[None, :], mask=m, other=-1)
    last = tl.max(tl.where(v >= 0, s[None, :] + 1, 0), axis=1)
    tl.store(Out + r, (last + BN - 1) // BN, mask=r < n_tok)


def walked_tiles(topk_indices, bn):
    """Per token: the number of ``bn``-slot tiles up to its last valid slot (int32)."""
    T, topk = topk_indices.shape
    out = torch.empty(T, dtype=torch.int32, device=topk_indices.device)
    rows = 8
    if T:
        _ntile_kernel[(triton.cdiv(T, rows),)](
            topk_indices, out, T, topk, bn, BLOCK=triton.next_power_of_2(topk), ROWS=rows, num_warps=4
        )
    return out
