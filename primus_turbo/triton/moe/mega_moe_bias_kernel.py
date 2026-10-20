###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Expert-bias gradients over the Mega MoE pool.

The pool holds each local expert's dispatched rows in whole ``POOL_BLOCK``-row tiles, padded past
the expert's row count; a row ``r`` of tile ``t`` is real iff ``r - prefix[e] < count[e]`` for
``e = tile_to_expert[t]``. Each reduction runs in two deterministic phases: one program per
(tile, row group) writes a partial row, then each expert sums its own tiles' partials in order.
"""

import triton
import triton.language as tl


@triton.jit
def _tile_rows_valid(rows, tile, TILE_TO_EXPERT, PREFIX, COUNT):
    expert = tl.load(TILE_TO_EXPERT + tile)
    start = tl.load(PREFIX + expert).to(tl.int32)
    count = tl.load(COUNT + expert).to(tl.int32)
    return expert, (rows - start) < count


@triton.jit
def expert_rows_sum_partial_kernel(
    X,  # [rows, C] pool-row major
    TILE_TO_EXPERT,
    PREFIX,  # [E_local + 1] i64, pool row offsets
    COUNT,  # [E_local] i64, real rows per expert
    PARTIAL,  # [num_tiles, C] f32
    C,
    stride_x,
    POOL_BLOCK: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    """PARTIAL[t] = the sum of tile t's real rows of X."""
    tile = tl.program_id(0)
    cols = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
    col_mask = cols < C
    acc = tl.zeros((BLOCK_C,), dtype=tl.float32)
    for r0 in range(0, POOL_BLOCK, BLOCK_R):
        rows = tile * POOL_BLOCK + r0 + tl.arange(0, BLOCK_R)
        _, valid = _tile_rows_valid(rows, tile, TILE_TO_EXPERT, PREFIX, COUNT)
        # pad rows hold garbage (the activation backward leaves them unwritten), so mask, not multiply
        x = tl.load(
            X + rows.to(tl.int64)[:, None] * stride_x + cols[None, :],
            mask=valid[:, None] & col_mask[None, :],
            other=0.0,
        )
        acc += tl.sum(x.to(tl.float32), axis=0)
    tl.store(PARTIAL + tile.to(tl.int64) * C + cols, acc, mask=col_mask)


@triton.jit
def fc2_bias_backward_partial_kernel(
    DY,  # [num_recv_tokens, H] the dispatched output gradient
    POOL_ROW_TO_RECV_TOKEN,
    SCALE,  # [rows] f32 routing weight per pool row
    TILE_TO_EXPERT,
    PREFIX,
    COUNT,
    BIAS,  # [E_local, H]
    GRAD_GATE,  # [rows] f32, accumulated in place
    PARTIAL,  # [num_tiles * GROUPS, H] f32
    H,
    stride_dy,
    POOL_BLOCK: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """For the real rows r of one row group: GRAD_GATE[r] += dy_r . b_e, PARTIAL = sum_r p_r dy_r.

    The forward adds p_r * b_e to row r's output, so b_e's gradient is sum_r p_r dy_r and p_r's
    picks up dy_r . b_e. A program owns whole rows, so the gate update needs no atomics."""
    tile = tl.program_id(0)
    group = tl.program_id(1)
    groups: tl.constexpr = POOL_BLOCK // BLOCK_R
    rows = tile * POOL_BLOCK + group * BLOCK_R + tl.arange(0, BLOCK_R)
    expert, valid = _tile_rows_valid(rows, tile, TILE_TO_EXPERT, PREFIX, COUNT)
    tokens = tl.load(POOL_ROW_TO_RECV_TOKEN + rows, mask=valid, other=0).to(tl.int64)
    weights = tl.load(SCALE + rows, mask=valid, other=0.0)
    dot = tl.zeros((BLOCK_R,), dtype=tl.float32)
    partial_row = (tile.to(tl.int64) * groups + group) * H
    for h0 in range(0, H, BLOCK_H):
        cols = h0 + tl.arange(0, BLOCK_H)
        col_mask = cols < H
        dy = tl.load(
            DY + tokens[:, None] * stride_dy + cols[None, :],
            mask=valid[:, None] & col_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        bias = tl.load(BIAS + expert.to(tl.int64) * H + cols, mask=col_mask, other=0.0).to(tl.float32)
        dot += tl.sum(dy * bias[None, :], axis=1)
        tl.store(PARTIAL + partial_row + cols, tl.sum(dy * weights[:, None], axis=0), mask=col_mask)
    gate = tl.load(GRAD_GATE + rows, mask=valid, other=0.0)
    tl.store(GRAD_GATE + rows, gate + dot, mask=valid)


@triton.jit
def expert_partials_sum_kernel(
    PARTIAL,  # [num_tiles * PARTS_PER_TILE, C] f32
    PREFIX,
    OUT,  # [E_local, C]
    C,
    POOL_BLOCK: tl.constexpr,
    PARTS_PER_TILE: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    """OUT[e] = the in-order sum of expert e's partial rows (its tiles are contiguous)."""
    expert = tl.program_id(0)
    cols = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
    col_mask = cols < C
    first = (tl.load(PREFIX + expert) // POOL_BLOCK * PARTS_PER_TILE).to(tl.int32)
    last = (tl.load(PREFIX + expert + 1) // POOL_BLOCK * PARTS_PER_TILE).to(tl.int32)
    acc = tl.zeros((BLOCK_C,), dtype=tl.float32)
    for j in range(first, last):
        acc += tl.load(PARTIAL + j.to(tl.int64) * C + cols, mask=col_mask, other=0.0)
    tl.store(OUT + expert.to(tl.int64) * C + cols, acc.to(OUT.dtype.element_ty), mask=col_mask)
