###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Bit-preserving reconstruction of gathered MXFP4 strips."""

import triton
import triton.language as tl


@triton.jit
def assemble_mxfp4_strips_kernel(
    wire,
    strip_bases,
    output,
    G: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    RD: tl.constexpr,
    RS: tl.constexpr,
    CD: tl.constexpr,
    CS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    lane = tl.arange(0, BLOCK)
    strips: tl.constexpr = G * N // 32
    kr: tl.constexpr = triton.cdiv(K, 128) * 64
    ks: tl.constexpr = triton.cdiv(K, 128) * 4
    nr: tl.constexpr = triton.cdiv(N, 128) * 64
    ns: tl.constexpr = triton.cdiv(N, 128) * 4
    rb: tl.constexpr = triton.cdiv(RD, BLOCK)
    sb: tl.constexpr = triton.cdiv(RS, BLOCK)
    cb: tl.constexpr = triton.cdiv(CD, BLOCK)
    if pid < rb:
        index = pid * BLOCK + lane
        row = index // kr
        column = index % kr
        strip = row // 32
        valid = (index < RD) & (column < K // 2)
        base = tl.load(strip_bases + strip, mask=valid, other=0)
        value = tl.load(wire + base + row % 32 * (K // 2) + column, mask=valid, other=0)
        tl.store(output + index, value, mask=index < RD)
    elif pid < rb + sb:
        index = (pid - rb) * BLOCK + lane
        row = index // ks
        column = index % ks
        strip = row // 32
        valid = (index < RS) & (column < K // 32)
        base = tl.load(strip_bases + strips + strip, mask=valid, other=0)
        value = tl.load(wire + base + row % 32 * (K // 32) + column, mask=valid, other=0)
        tl.store(output + RD + index, value, mask=index < RS)
    elif pid < rb + sb + cb:
        index = (pid - rb - sb) * BLOCK + lane
        group = index // (K * nr)
        row = index // nr % K
        column = index % nr
        strip = group * (N // 32) + column // 16
        valid = (index < CD) & (column < N // 2)
        base = tl.load(strip_bases + 2 * strips + strip, mask=valid, other=0)
        value = tl.load(wire + base + row * 16 + column % 16, mask=valid, other=0)
        tl.store(output + RD + RS + index, value, mask=index < CD)
    else:
        index = (pid - rb - sb - cb) * BLOCK + lane
        group = index // (K * ns)
        row = index // ns % K
        column = index % ns
        strip = group * (N // 32) + column
        valid = (index < CS) & (column < N // 32)
        base = tl.load(strip_bases + 3 * strips + strip, mask=valid, other=0)
        value = tl.load(wire + base + row, mask=valid, other=0)
        tl.store(output + RD + RS + CD + index, value, mask=index < CS)
