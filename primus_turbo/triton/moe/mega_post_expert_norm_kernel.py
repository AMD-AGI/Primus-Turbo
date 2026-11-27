###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Per-expert post norm with the routing weight, over the Mega MoE pool rows (Triton).

For every pool row ``y`` (the L2 GEMM output of one routed token at one expert), one norm weight
``w`` shared by the experts and the row's routing weight ``p``, in this order and rounding::

    s   = round(y.float() * (1 + w.float()))
    n   = round(s.float() * rsqrt(mean(s.float()**2) + eps))
    out = round(n.float() * p.float())

``round`` is a cast to ``y.dtype``. A pool row is padding when its receive-token index is the
dispatch prologue's sentinel (``>= num_recv``): the forward writes it as 0 and the backward gives
it no gradient. The backward reads the row's gradient straight from the dispatch token buffer
through the same index, so the dispatched dy is never gathered into pool order first::

    dp  = sum(g.float() * n.float())
    gn  = round(g.float() * p)
    ds  = round(r * gn - s * r**3 * mean(gn * s))     # r = rstd
    dy  = round(ds * (1 + w))
    dw  = sum_rows(ds * y.float())                    # per-program partials, then a sum
"""

import triton
import triton.language as tl


@triton.jit
def mega_post_norm_fwd_kernel(
    Y,
    W,
    P,
    RECV_IDX,
    OUT,
    RSTD,
    num_recv,
    H,
    stride_y,
    stride_o,
    eps,
    BLOCK_H: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK_H)
    live = tl.load(RECV_IDX + row) < num_recv
    mask = (cols < H) & live
    dt = Y.dtype.element_ty
    y = tl.load(Y + row * stride_y + cols, mask=mask, other=0.0).to(tl.float32)
    wp1 = 1.0 + tl.load(W + cols, mask=cols < H, other=0.0).to(tl.float32)
    s = (y * wp1).to(dt).to(tl.float32)
    ms = tl.sum(s * s, axis=0) / H
    r = tl.where(live, tl.math.div_rn(1.0, tl.sqrt_rn(ms + eps)), 0.0)
    n = (s * r).to(dt).to(tl.float32)
    n = n * tl.load(P + row).to(tl.float32)
    tl.store(OUT + row * stride_o + cols, n.to(OUT.dtype.element_ty), mask=cols < H)
    tl.store(RSTD + row, r)


@triton.jit
def mega_post_norm_bwd_kernel(
    Y,
    W,
    P,
    RECV_IDX,
    G,
    RSTD,
    DY,
    DP,
    DWP,
    T,
    num_recv,
    H,
    tiles_per_prog,
    stride_y,
    stride_g,
    stride_dy,
    BLOCK_T: tl.constexpr,
    BLOCK_H: tl.constexpr,
    HAS_DW: tl.constexpr,
):
    # each program walks ``tiles_per_prog`` tiles of BLOCK_T pool rows and keeps its own fp32
    # partial of dw (one row of DWP); the host sums the partials
    pid = tl.program_id(0)
    cols = tl.arange(0, BLOCK_H)
    cmask = cols < H
    dt = Y.dtype.element_ty
    wp1 = 1.0 + tl.load(W + cols, mask=cmask, other=0.0).to(tl.float32)
    acc = tl.zeros([BLOCK_H], dtype=tl.float32)
    first = pid.to(tl.int64) * tiles_per_prog * BLOCK_T
    offs_t = tl.arange(0, BLOCK_T).to(tl.int64)
    for k in range(0, tiles_per_prog):
        rows = first + k * BLOCK_T + offs_t
        in_range = rows < T
        idx = tl.load(RECV_IDX + rows, mask=in_range, other=num_recv).to(tl.int64)
        live = in_range & (idx < num_recv)
        m = live[:, None] & cmask[None, :]
        y = tl.load(Y + rows[:, None] * stride_y + cols[None, :], mask=m, other=0.0).to(tl.float32)
        g = tl.load(G + idx[:, None] * stride_g + cols[None, :], mask=m, other=0.0).to(tl.float32)
        r = tl.load(RSTD + rows, mask=live, other=0.0)
        p = tl.load(P + rows, mask=live, other=0.0).to(tl.float32)
        s = (y * wp1[None, :]).to(dt).to(tl.float32)
        n = (s * r[:, None]).to(dt).to(tl.float32)
        tl.store(DP + rows, tl.sum(g * n, axis=1).to(DP.dtype.element_ty), mask=in_range)
        gn = (g * p[:, None]).to(dt).to(tl.float32)
        c = r * r * r * (tl.sum(gn * s, axis=1) / H)
        ds = (r[:, None] * gn - s * c[:, None]).to(dt).to(tl.float32)
        tl.store(
            DY + rows[:, None] * stride_dy + cols[None, :],
            (ds * wp1[None, :]).to(DY.dtype.element_ty),
            mask=in_range[:, None] & cmask[None, :],
        )
        if HAS_DW:
            acc += tl.sum(ds * y, axis=0)  # padding rows add 0: their y and g are 0
    if HAS_DW:
        tl.store(DWP + pid.to(tl.int64) * H + cols, acc, mask=cmask)
