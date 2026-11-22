###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Attention-sink gradient of sparse MLA: ``dsink[h] = -sum_t exp(sink[h] - lse[t, h]) * delta[t, h]``."""

import torch
import triton
import triton.language as tl


@triton.jit
def _sparse_mla_dsink_kernel(
    lse_ptr,
    delta_ptr,
    sink_ptr,
    part_ptr,
    T,
    H: tl.constexpr,
    BH: tl.constexpr,
    BT: tl.constexpr,
    CH: tl.constexpr,
):
    h = tl.program_id(0) * BH + tl.arange(0, BH)
    c = tl.program_id(1)
    sink = tl.load(sink_ptr + h)
    acc = tl.zeros([BT, BH], dtype=tl.float32)
    for t0 in tl.range(c * CH, c * CH + CH, BT, num_stages=3):
        t = t0 + tl.arange(0, BT)
        off = t[:, None] * H + h[None, :]
        m = t[:, None] < T
        lse = tl.load(lse_ptr + off, mask=m, other=float("inf"))
        delta = tl.load(delta_ptr + off, mask=m, other=0.0)
        acc += tl.exp(sink[None, :] - lse) * delta
    tl.store(part_ptr + c * H + h, -tl.sum(acc, axis=0))


def sparse_mla_dsink(attn_sink, lse, delta, block_h=16, block_t=256, chunk=1024, num_warps=4):
    """fp32 ``[H]`` from fp32 ``attn_sink [H]``, ``lse [T, H]`` and ``delta [T, H]``: per-chunk
    partials, then a fixed-order sum (deterministic)."""
    T, H = lse.shape
    assert lse.is_contiguous() and delta.is_contiguous() and delta.shape == lse.shape and H % block_h == 0
    n_c = triton.cdiv(T, chunk)
    part = torch.empty(n_c, H, dtype=torch.float32, device=lse.device)
    _sparse_mla_dsink_kernel[(H // block_h, n_c)](
        lse, delta, attn_sink, part, T, H=H, BH=block_h, BT=block_t, CH=chunk, num_warps=num_warps
    )
    return part.sum(0)


# ----------------------------------------------------------------------------------------------------------------------
# PRIMUS_LAUNCH_DSINK (launch line, default off): the sink gradient in ONE launch, written straight into the sink
# parameter's ``main_grad``. Replaces dsink kernel + ``part.sum(0)`` + autograd's bf16 cast + the DDP hook copy / add.
# Same partials (same tiles / loop / tl.sum per chunk), the chunk partials combined in torch's order for a 4-ish row
# dim-0 sum (((0 + p0) + p1) + ...), rounded to bf16 once, then overwrite (first write) or bf16 add like ``add_``.
_MG_TARGET = [None]


def set_main_grad_target(mg, accumulate):
    """The next ``sparse_mla_bwd_gfx1250`` call writes its dsink into ``mg`` (bf16 [H]) instead of returning it."""
    _MG_TARGET[0] = (mg, bool(accumulate))


def take_main_grad_target():
    t, _MG_TARGET[0] = _MG_TARGET[0], None
    return t


@triton.jit
def _sparse_mla_dsink_mg_kernel(
    lse_ptr,
    delta_ptr,
    sink_ptr,
    mg_ptr,
    T,
    NC,
    H: tl.constexpr,
    BH: tl.constexpr,
    BT: tl.constexpr,
    CH: tl.constexpr,
    ACC: tl.constexpr,
):
    h = tl.program_id(0) * BH + tl.arange(0, BH)
    sink = tl.load(sink_ptr + h).to(tl.float32)
    tot = tl.zeros([BH], dtype=tl.float32)
    acc = tl.zeros([BT, BH], dtype=tl.float32)
    # one pipelined loop over all NC * CH rows (the per-chunk accumulators restart at each chunk boundary, so every
    # chunk partial is the original kernel's; adding 0.0 between boundaries leaves tot unchanged: it is never -0.0)
    for i in tl.range(0, NC * (CH // BT), num_stages=3):
        t = i * BT + tl.arange(0, BT)
        off = t[:, None] * H + h[None, :]
        m = t[:, None] < T
        lse = tl.load(lse_ptr + off, mask=m, other=float("inf"))
        delta = tl.load(delta_ptr + off, mask=m, other=0.0)
        acc += tl.exp(sink[None, :] - lse) * delta
        last = (i + 1) % (CH // BT) == 0
        part = 0.0 - tl.sum(acc, axis=0)
        tot = tot + tl.where(last, part, 0.0)
        acc = tl.where(last, 0.0, acc)
    g = tot.to(tl.bfloat16)
    if ACC:
        g = (tl.load(mg_ptr + h).to(tl.float32) + g.to(tl.float32)).to(tl.bfloat16)
    tl.store(mg_ptr + h, g)


@triton.jit
def _sparse_mla_dsink_fin_kernel(part_ptr, mg_ptr, NC, H: tl.constexpr, ACC: tl.constexpr):
    h = tl.arange(0, H)
    tot = tl.zeros([H], dtype=tl.float32)
    for c in range(0, NC):
        tot = tot + tl.load(part_ptr + c * H + h)
    g = tot.to(tl.bfloat16)
    if ACC:
        g = (tl.load(mg_ptr + h).to(tl.float32) + g.to(tl.float32)).to(tl.bfloat16)
    tl.store(mg_ptr + h, g)


def sparse_mla_dsink_mg2(attn_sink, lse, delta, mg, accumulate, block_h=16, block_t=256, chunk=1024, num_warps=4):
    """PRIMUS_LAUNCH_DSINK=2: the original partial kernel (H/16 x chunks programs) + one tiny finalize into
    ``mg`` (((0 + p0) + p1) + ..., bf16 RNE, overwrite / add): 2 launches instead of 4, full parallelism."""
    T, H = lse.shape
    assert lse.is_contiguous() and delta.is_contiguous() and delta.shape == lse.shape and H % block_h == 0
    assert mg.dtype == torch.bfloat16 and mg.is_contiguous() and mg.numel() == H and H & (H - 1) == 0
    n_c = triton.cdiv(T, chunk)
    part = torch.empty(n_c, H, dtype=torch.float32, device=lse.device)
    _sparse_mla_dsink_kernel[(H // block_h, n_c)](
        lse, delta, attn_sink, part, T, H=H, BH=block_h, BT=block_t, CH=chunk, num_warps=num_warps
    )
    _sparse_mla_dsink_fin_kernel[(1,)](part, mg, n_c, H=H, ACC=bool(accumulate), num_warps=1)


def sparse_mla_dsink_mg(attn_sink, lse, delta, mg, accumulate, block_h=16, block_t=256, chunk=1024, num_warps=4):

    """``mg (bf16 [H]) = bf16(dsink)`` (or ``+=``); ``attn_sink`` fp32 or bf16 [H]. One launch, H // block_h programs."""
    T, H = lse.shape
    assert lse.is_contiguous() and delta.is_contiguous() and delta.shape == lse.shape and H % block_h == 0
    assert mg.dtype == torch.bfloat16 and mg.is_contiguous() and mg.numel() == H and attn_sink.numel() == H
    for x in (lse, delta, attn_sink, mg):
        assert x.is_cuda and x.data_ptr() != 0
    n_c = triton.cdiv(T, chunk)
    _sparse_mla_dsink_mg_kernel[(H // block_h,)](
        lse, delta, attn_sink, mg, T, n_c, H=H, BH=block_h, BT=block_t, CH=chunk, ACC=bool(accumulate),
        num_warps=num_warps
    )
