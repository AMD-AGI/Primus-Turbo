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


def _dsink_parts(attn_sink, lse, delta, block_h, block_t, chunk, num_warps):
    """Per-chunk fp32 partials ``[n_chunks, H]`` of the sink gradient and ``n_chunks``."""
    T, H = lse.shape
    assert lse.is_contiguous() and delta.is_contiguous() and delta.shape == lse.shape and H % block_h == 0
    n_c = triton.cdiv(T, chunk)
    part = torch.empty(n_c, H, dtype=torch.float32, device=lse.device)
    _sparse_mla_dsink_kernel[(H // block_h, n_c)](
        lse, delta, attn_sink, part, T, H=H, BH=block_h, BT=block_t, CH=chunk, num_warps=num_warps
    )
    return part, n_c


def sparse_mla_dsink(attn_sink, lse, delta, block_h=16, block_t=256, chunk=1024, num_warps=4):
    """fp32 ``[H]`` from fp32 ``attn_sink [H]``, ``lse [T, H]`` and ``delta [T, H]``: per-chunk
    partials, then a fixed-order sum (deterministic)."""
    part, _ = _dsink_parts(attn_sink, lse, delta, block_h, block_t, chunk, num_warps)
    return part.sum(0)


# ----------------------------------------------------------------------------------------------------------------------
# The sink gradient written straight into the sink parameter's ``main_grad``: the partial kernel above plus one
# tiny finalize launch, replacing ``part.sum(0)``, autograd's bf16 cast and the DDP hook copy / add. The chunk
# partials are combined in torch's order for a short dim-0 sum (((0 + p0) + p1) + ...), rounded to bf16 once,
# then written (first write) or added in bf16 like ``add_``.
# ``sparse_mla_bwd_gfx1250(..., dsink_main_grad=(mg, accumulate))`` selects this path.


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


def sparse_mla_dsink_mg2(
    attn_sink, lse, delta, mg, accumulate, block_h=16, block_t=256, chunk=1024, num_warps=4
):
    """The partial kernel (H/16 x chunks programs) + one tiny finalize into ``mg`` (((0 + p0) + p1) + ...,
    bf16 RNE, overwrite / add): 2 launches."""
    H = lse.shape[1]
    assert mg.dtype == torch.bfloat16 and mg.is_contiguous() and mg.numel() == H and H & (H - 1) == 0
    part, n_c = _dsink_parts(attn_sink, lse, delta, block_h, block_t, chunk, num_warps)
    _sparse_mla_dsink_fin_kernel[(1,)](part, mg, n_c, H=H, ACC=bool(accumulate), num_warps=1)
