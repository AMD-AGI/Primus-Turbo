###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Tiled 2-D transpose into a contiguous output: ``y = x.T``.

``y.copy_(x.t())`` on a row-major ``x`` runs torch's generic strided copy, which reads or
writes with a large stride. This kernel loads a ``BLOCK_R x BLOCK_C`` tile coalesced along
``x``'s unit-stride dimension and stores it transposed, coalesced along ``y``'s rows. It is
pure data movement, so the result is bit-identical to torch's copy.
:mod:`primus_turbo.pytorch.core.mm_layout_workaround` uses it for its operand copies. The C++
op ``transpose_2d`` (``csrc/pytorch/transpose``) does not fit there: it takes only contiguous
input and allocates its own output, while the workaround transposes strided views (row and
column blocks) into preallocated scratch buffers.
"""

import torch
import triton
import triton.language as tl

__all__ = ["transpose_into"]


@triton.jit
def _transpose_kernel(
    x_ptr,
    y_ptr,
    R,
    C,
    stride_xr,
    stride_xc,
    BLOCK_R: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    # One program per tile of x, row tiles fastest: the launch order of a 2-D
    # (row tiles, column tiles) grid, without the tighter limit on a grid's second dimension.
    pid = tl.program_id(0)
    num_tiles_r = tl.cdiv(R, BLOCK_R)
    pid_r = pid % num_tiles_r
    pid_c = pid // num_tiles_r
    # int64 offsets: an operand may hold more than 2**31 elements.
    rows = pid_r.to(tl.int64) * BLOCK_R + tl.arange(0, BLOCK_R).to(tl.int64)
    cols = pid_c.to(tl.int64) * BLOCK_C + tl.arange(0, BLOCK_C).to(tl.int64)
    mask = (rows[:, None] < R) & (cols[None, :] < C)
    tile = tl.load(x_ptr + rows[:, None] * stride_xr + cols[None, :] * stride_xc, mask=mask)
    # y is (C, R) row-major: y[c, r] = x[r, c].
    tl.store(y_ptr + cols[:, None] * R + rows[None, :], tl.trans(tile), mask=tl.trans(mask))


def transpose_into(
    y: torch.Tensor,
    x: torch.Tensor,
    block_r: int = 64,
    block_c: int = 64,
    num_warps: int = 4,
) -> torch.Tensor:
    """Write ``x.T`` into ``y`` and return ``y``.

    Args:
        y: Contiguous ``(C, R)`` output with ``x``'s dtype and device. Must not overlap ``x``.
        x: ``(R, C)`` input with any strides, e.g. a transposed view or a column slice.
        block_r, block_c, num_warps: Tile shape, ``block_r x block_c`` elements of ``x`` per
            program (powers of two), and warps per program.
    """
    if x.dim() != 2:
        raise ValueError(f"transpose_into: x must be 2-D, got shape {tuple(x.shape)}")
    rows, cols = x.shape
    if y.shape != (cols, rows) or not y.is_contiguous():
        raise ValueError(
            f"transpose_into: y must be a contiguous {(cols, rows)} tensor, got shape "
            f"{tuple(y.shape)} with strides {y.stride()}"
        )
    if y.dtype != x.dtype or y.device != x.device:
        raise ValueError(f"transpose_into: y ({y.dtype}, {y.device}) must match x ({x.dtype}, {x.device})")
    if x.numel() == 0:
        return y
    grid = (triton.cdiv(rows, block_r) * triton.cdiv(cols, block_c),)
    with torch.cuda.device(x.device):
        _transpose_kernel[grid](
            x,
            y,
            rows,
            cols,
            x.stride(0),
            x.stride(1),
            BLOCK_R=block_r,
            BLOCK_C=block_c,
            num_warps=num_warps,
        )
    return y
