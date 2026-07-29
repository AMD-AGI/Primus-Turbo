// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
//
// See LICENSE for license information.
#pragma once

#include "primus_turbo/common.h"
#include "primus_turbo/device/reduce.cuh"
#include "primus_turbo/device/utils.cuh"
#include <hip/hip_runtime.h>

namespace primus_turbo {

// Round 1: one warp per row of the [total_m, inner_len] input. A row never spans
// two blocks, so the groups do not matter yet and no partial has to be merged
// across blocks.
template <template <class> class ReduceOp, typename InType, typename ComputeType, int BLOCK_SIZE>
__launch_bounds__(BLOCK_SIZE) __global__
    void reduce_grouped_row_per_row_kernel(const InType *__restrict__ input,
                                           ComputeType *__restrict__ row_partial,
                                           const int64_t total_m, const int64_t inner_len) {
    constexpr int UNROLL         = 16 / sizeof(InType);
    constexpr int ROWS_PER_BLOCK = BLOCK_SIZE / THREADS_PER_WARP;

    const int64_t row = static_cast<int64_t>(blockIdx.x) * ROWS_PER_BLOCK +
                        static_cast<int64_t>(threadIdx.x / THREADS_PER_WARP);
    if (row >= total_m)
        return;

    const int     lane    = threadIdx.x % THREADS_PER_WARP;
    const InType *row_ptr = input + row * inner_len;
    ComputeType   local   = ReduceOp<ComputeType>::init();

    // Row bases are multiples of inner_len, so the packed load is only safe when
    // both the input and the row stride preserve the pack alignment.
    const bool packed = (inner_len % UNROLL == 0) &&
                        (reinterpret_cast<uintptr_t>(row_ptr) % (sizeof(InType) * UNROLL) == 0);
    if (packed) {
        InType ld_regs[UNROLL];
        for (int64_t col = static_cast<int64_t>(lane) * UNROLL; col < inner_len;
             col += THREADS_PER_WARP * UNROLL) {
            load_data<InType, UNROLL>(row_ptr + col, ld_regs);
#pragma unroll
            for (int i = 0; i < UNROLL; ++i) {
                local = ReduceOp<ComputeType>::op(local, static_cast<ComputeType>(ld_regs[i]));
            }
        }
    } else {
        for (int64_t col = lane; col < inner_len; col += THREADS_PER_WARP) {
            local = ReduceOp<ComputeType>::op(local, static_cast<ComputeType>(row_ptr[col]));
        }
    }

    // A warp maps to exactly one row, so the reduction is never entered with part
    // of the wavefront retired by the bounds check above.
    local = WarpReduce<ReduceOp, ComputeType>(local);
    if (lane == 0) {
        row_partial[row] = local;
    }
}

// Round 2: one block per group folds the row partials of its rows into
// output[group]. Reading the partials back instead of merging them with
// cross-block atomics keeps the result independent of the block order.
template <template <class> class ReduceOp, typename ComputeType, typename OutType, int BLOCK_SIZE>
__launch_bounds__(BLOCK_SIZE) __global__
    void reduce_grouped_row_per_group_kernel(const ComputeType *__restrict__ row_partial,
                                             OutType *__restrict__ output,
                                             const int64_t *__restrict__ group_offs,
                                             const int64_t group_num, const int64_t total_m) {
    const int64_t group = static_cast<int64_t>(blockIdx.x);

    // Rows below the first / above the last offset fold into the first / last
    // group, so every row of the input feeds exactly one output.
    const int64_t row_beg = (group == 0) ? 0 : group_offs[group];
    const int64_t end     = (group + 1 == group_num) ? total_m : group_offs[group + 1];
    const int64_t row_end = (end < total_m) ? end : total_m;

    ComputeType local = ReduceOp<ComputeType>::init();
    for (int64_t row = row_beg + threadIdx.x; row < row_end; row += BLOCK_SIZE) {
        local = ReduceOp<ComputeType>::op(local, row_partial[row]);
    }
    const ComputeType ret = BlockReduce<ReduceOp, ComputeType>(local);

    // An empty group keeps the reduce identity.
    if (threadIdx.x == 0) {
        output[group] = static_cast<OutType>(ret);
    }
}

} // namespace primus_turbo
