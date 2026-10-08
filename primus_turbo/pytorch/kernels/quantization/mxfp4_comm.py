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
    SHARED_2D: tl.constexpr = False,
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
    # Traverse column operands in 64x64 logical tiles. Reading a long output
    # row would visit many distant strip allocations one tiny fragment at a
    # time; tiled reconstruction keeps neighboring source rows together.
    cb: tl.constexpr = G * triton.cdiv(K, 64) * (nr // 32)
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
        if SHARED_2D:
            source = base + column
        else:
            source = base + row % 32 * (K // 32) + column
        value = tl.load(wire + source, mask=valid, other=0)
        tl.store(output + RD + index, value, mask=index < RS)
    elif pid < rb + sb + cb:
        tile = pid - rb - sb
        n_tile = tile % (nr // 32)
        k_tile = tile // (nr // 32) % triton.cdiv(K, 64)
        tile_group = tile // ((nr // 32) * triton.cdiv(K, 64))
        out_row = k_tile * 64 + tl.arange(0, 64)
        out_column = n_tile * 32 + tl.arange(0, 32)
        if SHARED_2D:
            in_row = n_tile * 64 + tl.arange(0, 64)
            in_column = k_tile * 32 + tl.arange(0, 32)
            tile_strip = tile_group * (N // 32) + in_row // 32
            tile_base = tl.load(strip_bases + tile_strip, mask=in_row < N, other=0)
            tile_source = tile_base[:, None] + (in_row % 32)[:, None] * (K // 2) + in_column[None, :]
            packed_rows = tl.load(
                wire + tile_source, mask=(in_row[:, None] < N) & (in_column[None, :] < K // 2), other=0
            )
            values = tl.reshape(tl.join(packed_rows & 15, packed_rows >> 4), (64, 64))
            even, odd = tl.split(tl.reshape(tl.trans(values), (64, 32, 2)))
            packed_columns = even | (odd << 4)
        else:
            tile_strip = tile_group * (N // 32) + out_column // 16
            tile_base = tl.load(strip_bases + 2 * strips + tile_strip, mask=out_column < N // 2, other=0)
            tile_source = tile_base[None, :] + out_row[:, None] * 16 + (out_column % 16)[None, :]
            packed_columns = tl.load(
                wire + tile_source, mask=(out_row[:, None] < K) & (out_column[None, :] < N // 2), other=0
            )
        destination = tile_group * K * nr + out_row[:, None] * nr + out_column[None, :]
        tl.store(output + RD + RS + destination, packed_columns, mask=out_row[:, None] < K)
    else:
        index = (pid - rb - sb - cb) * BLOCK + lane
        group = index // (K * ns)
        row = index // ns % K
        column = index % ns
        strip = group * (N // 32) + column
        valid = (index < CS) & (column < N // 32)
        if SHARED_2D:
            base = tl.load(strip_bases + strips + strip, mask=valid, other=0)
            source = base + row // 32
        else:
            base = tl.load(strip_bases + 3 * strips + strip, mask=valid, other=0)
            source = base + row
        value = tl.load(wire + source, mask=valid, other=0)
        tl.store(output + RD + RS + CD + index, value, mask=index < CS)
