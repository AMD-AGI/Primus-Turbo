###############################################################################
# SPDX-License-Identifier: MIT
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
###############################################################################

"""UE8M0 128-block scale bytes -> the packed MX block-32 layout of the dsv4up Gluon GEMM, in one launch.

Source: E8M0 bytes (value = 2**(code - 127)), either 1x128 per-row scales src[R, K/128] (rows_per_scale = 1)
or 128x128 block scales src[R/128, K/128] (rows_per_scale = 128); any 2D strides (e.g. a transposed view).
Destination: packed[R/128, K/32*128] uint8 with
    packed[c, kc*512 + j*16 + i*4 + s] = src[(c*128 + i*32 + j) // rows_per_scale, kc]
which equals upstream mxfp_gemm_cdna5.pack_scale(src replicated x4 along K (x128 along rows for blocks)), so the
kernel's block-32 scales are the 128-block scales (exact for power-of-two scales).
Traffic: reads the (tiny, cache-resident) source, writes K/32 bytes per row -- 4x the 1x128 scale bytes.
Several (src, dst) jobs share one launch (one program column per job).
"""

import torch
import triton
import triton.language as tl

_BKC = 8  # 128-wide K chunks per program (8 * 512 B = 4 KiB written per program)


@triton.jit
def pack_job(src_ptr, dst_ptr, s0, s1, KB, RPS, pid, n_kcb, BKC: tl.constexpr):
    """Program ``pid`` of one pack job: BKC 128-wide K chunks of 128-row block ``pid // n_kcb`` (s0 / s1: source strides,
    KB = K / 128, RPS = rows per source scale). Callable from other Triton kernels that fuse the packing."""
    # the 4 bytes s = 0..3 of every (j, i) are the same code: one gathered byte -> one int32 (code * 0x01010101)
    c = pid // n_kcb
    kc = (pid % n_kcb) * BKC + tl.arange(0, BKC)
    t = tl.arange(0, 128)  # int32 word t = j*4 + i of a 512-byte kc chunk
    row = c * 128 + (t % 4) * 32 + t // 4
    srow = row // RPS
    kmask = kc < KB
    v = tl.load(
        src_ptr + srow[None, :].to(tl.int64) * s0 + kc[:, None].to(tl.int64) * s1,
        mask=kmask[:, None],
        other=127,
    )
    w = v.to(tl.int32) * 0x01010101
    dst = (
        dst_ptr.to(tl.pointer_type(tl.int32))
        + c.to(tl.int64) * KB * 128
        + kc[:, None].to(tl.int64) * 128
        + t[None, :]
    )
    tl.store(dst, w, mask=kmask[:, None])


@triton.jit
def _pack_scales_kernel(
    s0p,
    d0p,
    a0,
    b0,
    k0,
    r0,
    n0,
    c0,
    s1p,
    d1p,
    a1,
    b1,
    k1,
    r1,
    n1,
    c1,
    s2p,
    d2p,
    a2,
    b2,
    k2,
    r2,
    n2,
    c2,
    s3p,
    d3p,
    a3,
    b3,
    k3,
    r3,
    n3,
    c3,
    BKC: tl.constexpr,
):
    pid = tl.program_id(0)
    job = tl.program_id(1)
    if job == 0:
        if pid < n0:
            pack_job(s0p, d0p, a0, b0, k0, r0, pid, c0, BKC)
    elif job == 1:
        if pid < n1:
            pack_job(s1p, d1p, a1, b1, k1, r1, pid, c1, BKC)
    elif job == 2:
        if pid < n2:
            pack_job(s2p, d2p, a2, b2, k2, r2, pid, c2, BKC)
    else:
        if pid < n3:
            pack_job(s3p, d3p, a3, b3, k3, r3, pid, c3, BKC)


def pack_scales(jobs):
    """jobs: list of (src_e8m0 uint8 2D tensor, rows_per_scale (1 | 128), R, K).  Returns the packed tensors
    [R/128, K*4] (None for a None job).  R % 128 == 0, K % 128 == 0.  At most 4 non-None jobs per launch."""
    outs = [None] * len(jobs)
    live = [(i, j) for i, j in enumerate(jobs) if j is not None]
    for base in range(0, len(live), 4):
        chunk = live[base : base + 4]
        args, max_prog = [], 0
        for i, (src, rps, R, K) in chunk:
            assert src.dtype == torch.uint8 and src.dim() == 2 and R % 128 == 0 and K % 128 == 0
            KB = K // 128
            assert tuple(src.shape) == ((R // rps), KB), (tuple(src.shape), rps, R, K)
            dst = torch.empty((R // 128, K * 4), dtype=torch.uint8, device=src.device)
            outs[i] = dst
            n_kcb = triton.cdiv(KB, _BKC)
            nprog = (R // 128) * n_kcb
            max_prog = max(max_prog, nprog)
            args += [src, dst, src.stride(0), src.stride(1), KB, rps, nprog, n_kcb]
        while len(args) < 32:  # unused job slots
            args += [args[0], args[1], 0, 0, 0, 1, 0, 1]
        _pack_scales_kernel[(max_prog, len(chunk))](*args, BKC=_BKC, num_warps=4)
    return outs
