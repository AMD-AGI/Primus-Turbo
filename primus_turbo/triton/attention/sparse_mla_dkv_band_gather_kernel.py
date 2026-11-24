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
def _pool_fill_scan_kernel(Pos, Cnt, Off, Ent, t0, t1, T, TOPK, P, BLOCK: tl.constexpr, PB: tl.constexpr):
    # Ent[Off[p] ..] = interm row (t - t0) * TOPK + slot of every pick of row p, ascending t, with the exclusive
    # scan of Cnt done in place (exact integer sums): writes Off[p + 1] (and Off[0]).
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
    return pool_positions_into(pos, topk_indices, window, n_raw, pool)


def dkv_band_gather_acc(interm, pos, acc, t0, t1, window, n_raw, pool, d=512, seg_t=256, db=128, nw=8):
    """``acc[:, :d] +=`` the dKV of tokens ``[t0, t1)`` from their ``[t1 - t0, topk, d]`` bf16
    intermediate, for a window + pool map (see the module docstring); the per-pool-row offsets are
    scanned inside the fill kernel."""
    n = t1 - t0
    if n <= 0:
        return
    assert d == 512 and d % db == 0
    topk = interm.shape[1]
    T = pos.shape[1] if pool else t1
    _window_gather_kernel[(n + window - 1, d // db)](
        interm,
        acc,
        t0,
        t1,
        window,
        topk,
        acc.stride(0),
        WP=triton.next_power_of_2(window),
        DB=db,
        num_warps=nw,
    )
    if pool and topk > window:
        cnt = torch.empty(pool, dtype=torch.int32, device=acc.device)
        _pool_count_kernel[(pool,)](pos, cnt, t0, t1, T, BLOCK=1024, num_warps=4)
        ent = torch.empty(n * (topk - window), dtype=torch.int32, device=acc.device)
        off = torch.empty(pool + 1, dtype=torch.int32, device=acc.device)
        _pool_fill_scan_kernel[(pool,)](
            pos,
            cnt,
            off,
            ent,
            t0,
            t1,
            T,
            topk,
            pool,
            BLOCK=1024,
            PB=triton.next_power_of_2(pool),
            num_warps=4,
        )
        nseg = triton.cdiv(n, seg_t)
        side = torch.empty(pool, nseg, d, dtype=torch.float32, device=acc.device)
        _pool_gather_kernel[(pool, nseg, d // db)](
            interm, ent, off, side, nseg, SEG=seg_t, DB=db, num_warps=nw
        )
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


# ----------------------------------------------------------------------------------------------------------------------
# Single-chunk band backward (T tokens, kv rows [0, T) raw + T + p
# pool): every dKV row has exactly one writer (window gather for raw rows, fixup for pool rows), so the gather writes
# the final bf16 dKV itself -- no fp32 zeros, no fp32 -> bf16 cast. Values: 0.0 + x then RNE to bf16, as before.
# The pos fill (-1) rides in the _ntile launch.
@triton.jit
def _ntile_fill_kernel(
    Idx, Out, n_tok, TOPK, BN, Pos, NPOS, NT, BLOCK: tl.constexpr, ROWS: tl.constexpr, FB: tl.constexpr
):
    pid = tl.program_id(0)
    if pid < NT:
        r = pid * ROWS + tl.arange(0, ROWS)
        s = tl.arange(0, BLOCK)
        m = (r < n_tok)[:, None] & (s < TOPK)[None, :]
        v = tl.load(Idx + r.to(tl.int64)[:, None] * TOPK + s[None, :], mask=m, other=-1)
        last = tl.max(tl.where(v >= 0, s[None, :] + 1, 0), axis=1)
        tl.store(Out + r, (last + BN - 1) // BN, mask=r < n_tok)
    else:
        o = (pid - NT).to(tl.int64) * FB + tl.arange(0, FB)
        tl.store(Pos + o, tl.full([FB], -1, tl.int16), mask=o < NPOS)


def walked_tiles_fill_pos(topk_indices, bn, pool):
    """(walked_tiles(topk_indices, bn), int16 [pool, T] filled with -1) from one launch."""
    T, topk = topk_indices.shape
    out = torch.empty(T, dtype=torch.int32, device=topk_indices.device)
    pos = torch.empty((pool, T), dtype=torch.int16, device=topk_indices.device)
    rows, fb = 8, 4096
    nt = triton.cdiv(T, rows)
    npos = pool * T
    assert T > 0 and npos > 0 and topk_indices.is_contiguous()
    _ntile_fill_kernel[(nt + triton.cdiv(npos, fb),)](
        topk_indices,
        out,
        T,
        topk,
        bn,
        pos,
        npos,
        nt,
        BLOCK=triton.next_power_of_2(topk),
        ROWS=rows,
        FB=fb,
        num_warps=4,
    )
    return out, pos


def pool_positions_into(pos, topk_indices, window, n_raw, pool):
    """band_pool_positions into a pre-filled (-1) ``pos``."""
    T, topk = topk_indices.shape
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


# Single-chunk band path (T == n_raw): the pool picks are counted and listed per pool row in one pass (row p's entries
# at the fixed offset p * T, no cross-row scan), then ONE launch reduces the window rows and the pool rows (each pool
# row sums its 256-entry segments in order inside one program: no fp32 side buffer, no fixup launch) and also writes
# the contiguous bf16 dpool. Bytes of ``zeros fp32 -> dkv_band_gather_acc -> .to(bf16)``: the same per-segment tl.sum
# tiles in the same entry order, partials added in order from 0.0, trailing all-zero partials skipped (exact: the
# accumulator is never -0.0).
@triton.jit
def _pool_fill_cnt_kernel(Pos, Cnt, Ent, T, TOPK, BLOCK: tl.constexpr):
    p = tl.program_id(0)
    row = p.to(tl.int64) * T
    base = tl.full([], 0, tl.int32)
    for tb in range(0, T, BLOCK):
        t = tb + tl.arange(0, BLOCK)
        sl = tl.load(Pos + row + t, mask=t < T, other=-1).to(tl.int32)
        ok = sl >= 0
        oki = ok.to(tl.int32)
        pre = tl.cumsum(oki, axis=0) - oki
        tl.store(Ent + row + base + pre, t * TOPK + sl, mask=ok)
        base += tl.sum(oki, axis=0)
    tl.store(Cnt + p, base)


@triton.jit
def _band_gather_all_kernel(
    Interm,
    Ent,
    Cnt,
    Out,
    Dpool,
    T,
    W,
    TOPK,
    n_raw,
    P,
    stride_out,
    WP: tl.constexpr,
    DB: tl.constexpr,
    D: tl.constexpr,
    RP: tl.constexpr,
    SEG: tl.constexpr,
    NCB: tl.constexpr,
):
    # 1-D grid: (P pool rows, heavy first) x NCB column blocks, then T window rows x NCB.
    pid = tl.program_id(0)
    r = pid // NCB
    c = pid % NCB
    offs_d = c * DB + tl.arange(0, DB)
    if r < P:
        p = r
        n = tl.load(Cnt + p)
        row = p.to(tl.int64) * T
        acc = tl.zeros([DB], dtype=tl.float32)
        for e0 in range(0, n, SEG):
            e = e0 + tl.arange(0, SEG)
            m = e < n
            ent = tl.load(Ent + row + e, mask=m, other=0).to(tl.int64)
            rows = tl.load(Interm + ent[:, None] * 512 + offs_d[None, :], mask=m[:, None], other=0.0)
            acc += tl.sum(rows.to(tl.float32), axis=0)
        o_ptr = Out + (n_raw + p).to(tl.int64) * stride_out
        v = acc.to(Out.dtype.element_ty)
        tl.store(o_ptr + offs_d, v)
        tl.store(Dpool + p.to(tl.int64) * D + offs_d, v)
    else:
        j = r - P
        t = j + tl.arange(0, WP)
        m = (t < T) & (t < j + W)
        ent = t.to(tl.int64) * TOPK + (j - t + W - 1)
        rows = tl.load(Interm + ent[:, None] * 512 + offs_d[None, :], mask=m[:, None], other=0.0)
        acc = tl.sum(rows.to(tl.float32), axis=0)
        acc = tl.where(acc == 0.0, 0.0, acc)  # 0.0 + acc (the zeroed fp32 accumulator)
        o_ptr = Out + j.to(tl.int64) * stride_out
        tl.store(o_ptr + offs_d, acc.to(Out.dtype.element_ty))
    if c == 0:
        z = tl.arange(0, RP)
        tl.store(o_ptr + D + z, tl.zeros([RP], dtype=Out.dtype.element_ty))


def pool_fill_cnt(pos, T, topk):
    """(cnt int32 [P], ent int32 [P * T]): ent[p * T + k] = interm row t * topk + slot of the k-th pick of pool row p
    (ascending t), from the dense ``pos`` table (int16 [P, T])."""
    P = pos.shape[0]
    assert pos.shape[1] == T and pos.is_contiguous() and P * T < 2**31 and T * topk < 2**31
    cnt = torch.empty(P, dtype=torch.int32, device=pos.device)
    ent = torch.empty(P * T, dtype=torch.int32, device=pos.device)
    _pool_fill_cnt_kernel[(P,)](pos, cnt, ent, T, topk, BLOCK=1024, num_warps=4)
    return cnt, ent


def dkv_band_gather_all(
    interm, cnt, ent, out, window, n_raw, pool, d=512, seg_t=256, db=128, nw=8, dpool=None
):
    """``out [n_raw + pool, d + rp]`` (bf16) = the final dKV of ALL tokens ``[0, T)`` (single chunk, T == n_raw) from
    ``pool_fill_cnt``'s (cnt, ent), in one launch; also writes ``dpool`` (bf16 [pool, d], contiguous) =
    ``out[n_raw:, :d]``. Returns dpool."""
    T = interm.shape[0]
    topk = interm.shape[1]
    rp = out.shape[1] - d
    assert (
        d == 512
        and d % db == 0
        and T == n_raw
        and pool > 0
        and topk > window
        and out.shape[0] == n_raw + pool
    )
    assert rp > 0 and rp & (rp - 1) == 0 and out.stride(1) == 1 and interm.is_contiguous()
    assert cnt.numel() == pool and ent.numel() == pool * T
    if dpool is None:
        dpool = torch.empty(pool, d, dtype=out.dtype, device=out.device)
    ncb = d // db
    _band_gather_all_kernel[((pool + T) * ncb,)](
        interm,
        ent,
        cnt,
        out,
        dpool,
        T,
        window,
        topk,
        n_raw,
        pool,
        out.stride(0),
        WP=triton.next_power_of_2(window),
        DB=db,
        D=d,
        RP=rp,
        SEG=seg_t,
        NCB=ncb,
        num_warps=nw,
    )
    return dpool
