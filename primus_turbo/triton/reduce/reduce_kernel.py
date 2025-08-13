###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import triton
import triton.language as tl


@triton.jit
def kernel_consumer_reduce_async_tp(
    c_ptr,  # [M, N]
    out_ptr,  # [M_per_rank, N]
    # shape of matrix
    M_per_rank,
    N,
    rank,
    num_ranks: tl.constexpr,
    REDUCE_AVG: tl.constexpr,
    # tile size
    BLOCK_SIZE: tl.constexpr,
):
    """
    Only used within asyncTP scenarios to enable overlap between matmul and reduce_scatter.
    Currently supporting bf16 and fp16 data types.
    """

    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs = tl.where(offs < M_per_rank * N, offs, 0)
    out_ptrs = out_ptr + offs

    accum = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    for i in range(0, num_ranks):
        cur_rank = (i + rank + 1) % num_ranks
        c_ptrs = c_ptr + offs + cur_rank * M_per_rank * N
        data = tl.load(c_ptrs)
        accum += data

    if REDUCE_AVG:
        accum /= num_ranks
    tl.store(out_ptrs, accum.to(out_ptrs.dtype.element_ty))
