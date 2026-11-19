###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Column sums of FP32 partial-sum buffers, which fused kernels write for their bias and modulation
gradients: over all rows, or per sample. A fixed reduction order keeps them deterministic."""

import triton
import triton.language as tl


@triton.jit
def colsum_kernel(src_ptr, out_ptr, n_rows, n_cols, BLOCK_R: tl.constexpr, BLOCK_C: tl.constexpr):
    # out[c] = sum_r src[r, c] for the FP32 partials; faster here than torch's column sum.
    cols = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)
    col_mask = cols < n_cols
    acc = tl.zeros([BLOCK_R, BLOCK_C], dtype=tl.float32)
    for r0 in range(0, n_rows, BLOCK_R):
        rows = (r0 + tl.arange(0, BLOCK_R)).to(tl.int64)
        acc += tl.load(
            src_ptr + rows[:, None] * n_cols + cols[None, :],
            mask=(rows < n_rows)[:, None] & col_mask[None, :],
            other=0.0,
        )
    tl.store(out_ptr + cols, tl.sum(acc, axis=0).to(out_ptr.dtype.element_ty), mask=col_mask)


@triton.jit
def sample_colsum_kernel(src_ptr, out_ptr, n_parts, n_cols, BLOCK_P: tl.constexpr, BLOCK_C: tl.constexpr):
    # out[b, c] = sum_p src[b, p, c] for the FP32 per-wave partials.
    b = tl.program_id(0).to(tl.int64)
    cols = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
    col_mask = cols < n_cols
    acc = tl.zeros([BLOCK_P, BLOCK_C], dtype=tl.float32)
    for p0 in range(0, n_parts, BLOCK_P):
        p = p0 + tl.arange(0, BLOCK_P)
        acc += tl.load(
            src_ptr + (b * n_parts + p)[:, None] * n_cols + cols[None, :],
            mask=(p < n_parts)[:, None] & col_mask[None, :],
            other=0.0,
        )
    tl.store(out_ptr + b * n_cols + cols, tl.sum(acc, axis=0).to(out_ptr.dtype.element_ty), mask=col_mask)
