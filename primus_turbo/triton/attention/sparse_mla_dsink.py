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
