###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Triton tanh-GELU backward with the bias gradient's per-program column sums, for FLUX's MLP."""

import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def flux_gelu_bwd_bias_kernel(
    g_ptr,
    h_ptr,
    dh_ptr,
    part_ptr,
    n_rows,
    n_cols,
    g_stride,
    rows_per_prog,
    h_stride,
    dh_stride,
    BLOCK_R: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid_r = tl.program_id(0)
    cols = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
    col_mask = cols < n_cols
    row_end = tl.minimum((pid_r + 1) * rows_per_prog, n_rows)
    acc = tl.zeros([BLOCK_R, BLOCK_C], dtype=tl.float32)
    for row0 in range(pid_r * rows_per_prog, row_end, BLOCK_R):
        rows = (row0 + tl.arange(0, BLOCK_R)).to(tl.int64)
        mask = (rows < row_end)[:, None] & col_mask[None, :]
        g = tl.load(g_ptr + rows[:, None] * g_stride + cols[None, :], mask=mask, other=0.0).to(tl.float32)
        h = tl.load(h_ptr + rows[:, None] * h_stride + cols[None, :], mask=mask, other=0.0).to(tl.float32)
        # Inductor's gelu_backward(approximate="tanh"), operation for operation.
        h2 = h * h
        t = libdevice.tanh((h + h2 * h * 0.044715) * 0.7978845608028654)
        dh = g * ((t + 1.0) * 0.5 + (h * 0.5) * (1.0 - t * t) * ((h2 * 0.134145 + 1.0) * 0.7978845608028654))
        dh = dh.to(dh_ptr.dtype.element_ty)
        tl.store(dh_ptr + rows[:, None] * dh_stride + cols[None, :], dh, mask=mask)
        acc += dh.to(tl.float32)
    tl.store(part_ptr + pid_r * n_cols + cols, tl.sum(acc, axis=0), mask=col_mask)
