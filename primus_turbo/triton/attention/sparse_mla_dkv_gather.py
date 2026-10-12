###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Load-balanced, deterministic dKV gather for the sparse-MLA backward.

``acc[k, :D] += sum(interm[e] for e in the CSR row k)``. Rows with at most ``SEG`` entries get
one program each. Heavier rows (e.g. the compressed pool) are split into fixed blocks of ``SEG``
entries of the kv-sorted list so they no longer serialize on one program: a heavy row inside a
block is added to ``acc`` directly, a heavy row that crosses blocks leaves one partial per block
in ``side`` that the fix-up kernel sums in block order (fixed order: bitwise reproducible).
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _sum_entries(Interm, InvData, s, t, D: tl.constexpr, BLOCK_K: tl.constexpr):
    offs_d = tl.arange(0, D)
    offs_k = tl.arange(0, BLOCK_K)
    acc = tl.zeros([D], dtype=tl.float32)
    for e0 in range(s, t, BLOCK_K):
        e = e0 + offs_k
        m = e < t
        ent = tl.load(InvData + e, mask=m, other=0).to(tl.int64)
        rows = tl.load(Interm + ent[:, None] * D + offs_d[None, :], mask=m[:, None], other=0.0)
        acc += tl.sum(rows.to(tl.float32), axis=0)
    return acc


@triton.jit
def _row_of(InvPtr, x, num_kv, LOG_KV: tl.constexpr):
    # Largest r with inv_ptr[r] <= x, for x < inv_ptr[num_kv]: the non-empty row holding entry x.
    lo = 0
    hi = num_kv
    for _ in tl.static_range(LOG_KV):
        mid = (lo + hi) // 2
        go = tl.load(InvPtr + mid) <= x
        lo = tl.where(go, mid, lo)
        hi = tl.where(go, hi, mid)
    return lo


@triton.jit
def _dkv_gather_main(
    Interm,  # [rows, D] bf16
    Keys,  # [E] sorted kv row per entry (-1 = invalid, first)
    InvPtr,  # [num_kv + 1] int32
    InvData,  # [E] int32 interm row per entry
    Acc,  # [num_kv, stride_acc] fp32
    Side,  # [num_blocks, 2, D] fp32
    E,
    num_kv,
    stride_acc,
    D: tl.constexpr,
    SEG: tl.constexpr,
    ROW_BK: tl.constexpr,
    BLOCK_K: tl.constexpr,
    LOG_KV: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_d = tl.arange(0, D)
    if pid < num_kv:
        p0 = tl.load(InvPtr + pid)
        p1 = tl.load(InvPtr + pid + 1)
        if p1 - p0 <= SEG:
            a_ptr = Acc + pid.to(tl.int64) * stride_acc + offs_d
            a = tl.load(a_ptr)
            tl.store(a_ptr, a + _sum_entries(Interm, InvData, p0, p1, D, ROW_BK))
    else:
        b = pid - num_kv
        b0 = b * SEG
        b1 = tl.minimum(b0 + SEG, E)
        lo = tl.maximum(b0, tl.load(InvPtr))
        if lo < b1:
            r = tl.load(Keys + lo).to(tl.int32)
            p0 = tl.load(InvPtr + r)
            p1 = tl.load(InvPtr + r + 1)
            # Visit only the block's non-empty rows: with chunked tokens the kv rows touched by
            # a chunk can leave long runs of empty rows (e.g. between the window and the pool).
            while p0 < b1:
                if p1 - p0 > SEG:
                    acc = _sum_entries(Interm, InvData, tl.maximum(p0, lo), tl.minimum(p1, b1), D, BLOCK_K)
                    if (p0 >= b0) & (p1 <= b1):
                        a_ptr = Acc + r.to(tl.int64) * stride_acc + offs_d
                        tl.store(a_ptr, tl.load(a_ptr) + acc)
                    elif p0 < b0:
                        tl.store(Side + (b * 2) * D + offs_d, acc)
                    else:
                        tl.store(Side + (b * 2 + 1) * D + offs_d, acc)
                if p1 < b1:
                    # Next non-empty row: r + 1 if it holds entries, else the row holding p1.
                    p2 = tl.load(InvPtr + r + 2)
                    if p2 > p1:
                        r = r + 1
                    else:
                        r = _row_of(InvPtr, p1, num_kv, LOG_KV)
                    p0 = p1
                    p1 = tl.load(InvPtr + r + 1)
                else:
                    p0 = b1


@triton.jit
def _dkv_gather_fixup(Acc, Side, InvPtr, stride_acc, D: tl.constexpr, SEG: tl.constexpr):
    r = tl.program_id(0)
    p0 = tl.load(InvPtr + r)
    p1 = tl.load(InvPtr + r + 1)
    bf = p0 // SEG
    bl = (p1 - 1) // SEG
    if (p1 - p0 > SEG) & (bl > bf):
        offs_d = tl.arange(0, D)
        a_ptr = Acc + r.to(tl.int64) * stride_acc + offs_d
        a = tl.load(a_ptr) + tl.load(Side + (bf * 2 + 1) * D + offs_d)
        for bb in range(bf + 1, bl + 1):
            a += tl.load(Side + (bb * 2) * D + offs_d)
        tl.store(a_ptr, a)


def dkv_gather_acc(interm, keys, inv_ptr, inv_data, acc, d=512, seg=256, row_bk=32, block_k=32, num_warps=4):
    """``acc[k, :d] += sum of interm rows listed for kv row k`` (CSR from a stable sort: ``keys``
    are the sorted kv rows, ``inv_ptr`` the row pointers, ``inv_data`` the interm rows)."""
    E = inv_data.numel()
    num_kv = inv_ptr.numel() - 1
    if E == 0 or num_kv == 0:
        return
    nb = triton.cdiv(E, seg)
    side = torch.empty(nb, 2, d, dtype=torch.float32, device=acc.device)
    _dkv_gather_main[(num_kv + nb,)](
        interm,
        keys,
        inv_ptr,
        inv_data,
        acc,
        side,
        E,
        num_kv,
        acc.stride(0),
        D=d,
        SEG=seg,
        ROW_BK=row_bk,
        BLOCK_K=block_k,
        LOG_KV=max(1, num_kv.bit_length()),
        num_warps=num_warps,
    )
    _dkv_gather_fixup[(num_kv,)](acc, side, inv_ptr, acc.stride(0), D=d, SEG=seg, num_warps=4)
