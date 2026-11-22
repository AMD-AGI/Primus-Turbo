###############################################################################
# SPDX-License-Identifier: MIT
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
###############################################################################
"""bf16tune: deterministic split-K on top of bf16g2_mm (gfx1250, Gluon TDM + WMMA).

  out = a @ b   (2-D, torch.mm semantics; a [M,K] K- or M-contiguous, b [K,N] K- or N-contiguous, bf16)

K is cut into S equal slices; slice s is batch entry s of one bf16g2_mm launch whose output is an fp32 workspace
[S, M, N] (the fp32 accumulator stored unrounded). A Triton kernel then sums the S partials in a fixed order
(((p0 + p1) + p2) + ...) in fp32 and rounds once to bf16. No atomics: same bits run to run.
S = 1 is a plain bf16g2_mm (no workspace, no reduce).
"""

import torch
import triton
import triton.language as tl

from primus_turbo.gluon.gemm.bf16g2_gemm import BK, TILES, bf16g2_mm


@triton.jit
def _splitk_reduce_kernel(ws_ptr, out_ptr, M, N, stride_om, S: tl.constexpr, BLOCK_M: tl.constexpr,
                          BLOCK_N: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (rm[:, None] < M) & (rn[None, :] < N)
    off = rm[:, None].to(tl.int64) * N + rn[None, :]
    mn = M.to(tl.int64) * N
    acc = tl.load(ws_ptr + off, mask=mask, other=0.0)
    for s in tl.static_range(1, S):
        acc += tl.load(ws_ptr + s * mn + off, mask=mask, other=0.0)
    tl.store(out_ptr + rm[:, None].to(tl.int64) * stride_om + rn[None, :], acc.to(out_ptr.dtype.element_ty),
             mask=mask)


def splitk_supported(a, b, out, splits, tile="256x256"):
    if a.dim() != 2 or b.dim() != 2 or out.dim() != 2:
        return False
    if not (a.dtype == b.dtype == out.dtype == torch.bfloat16) or out.stride(1) != 1:
        return False
    if not ((a.stride(1) == 1 or a.stride(0) == 1) and (b.stride(0) == 1 or b.stride(1) == 1)):
        return False
    K = a.shape[1]
    if K % splits:
        return False
    ks = K // splits
    _, _, ns = TILES[tile]
    return ks % int(BK.value) == 0 and ks // int(BK.value) >= ns + 1


def bf16g2_splitk_mm(a, b, out, splits, tile="256x256", red=(32, 128), **kw):
    """out = a @ b with K split `splits` ways (fixed-order fp32 reduce). Returns out."""
    M, K = a.shape
    N = b.shape[1]
    assert b.shape[0] == K and out.shape == (M, N), (a.shape, b.shape, out.shape)
    if splits == 1:
        return bf16g2_mm(a, b, out, tile=tile, **kw)
    assert K % splits == 0, (K, splits)
    ks = K // splits
    a3 = a.as_strided((splits, M, ks), (ks * a.stride(1), a.stride(0), a.stride(1)))
    b3 = b.as_strided((splits, ks, N), (ks * b.stride(0), b.stride(0), b.stride(1)))
    ws = torch.empty(splits, M, N, device=a.device, dtype=torch.float32)  # per call: safe across streams
    bf16g2_mm(a3, b3, ws, tile=tile, **kw)
    bm, bn = red
    grid = (triton.cdiv(M, bm), triton.cdiv(N, bn))
    _splitk_reduce_kernel[grid](ws, out, M, N, out.stride(0), S=splits, BLOCK_M=bm, BLOCK_N=bn, num_warps=4)
    return out
