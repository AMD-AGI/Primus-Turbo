// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
//
// See LICENSE for license information.

#include "primus_turbo/device/reduce.cuh"
#include "primus_turbo/reduce.h"
#include "reduce_row.cuh"

namespace primus_turbo {

using namespace primus_turbo::dtype;

template <template <class> class ReduceOp, typename InType, typename OutType, typename ComputeType>
void reduce_row_impl(const InType *input, OutType *output, const int64_t &outer_len,
                     const int64_t &inner_len, const int64_t workspace_sizes, void *workspace,
                     hipStream_t stream) {
    const int     BLOCK      = 256;
    const int     UNROLL     = 16;
    const int64_t TILE_ELEMS = BLOCK * UNROLL;
    if (inner_len <= TILE_ELEMS) {
        const int64_t grid_i = DIVUP<int64_t>(inner_len, BLOCK * UNROLL);
        const int64_t grid_o = outer_len;
        const dim3    grid(grid_i, grid_o, 1);
        reduce_row_kernel<ReduceOp, InType, OutType, ComputeType, UNROLL>
            <<<grid, BLOCK, 0, stream>>>(input, output, outer_len, inner_len);
        return;
    }

    // Multi round reduce
    const int64_t tiles        = DIVUP<int64_t>(inner_len, TILE_ELEMS);
    const int64_t max_partials = tiles;
    const int64_t need_elems   = 2 * outer_len * max_partials; // ping-pong
    PRIMUS_TURBO_CHECK(need_elems * sizeof(ComputeType) >= workspace_sizes,
                       "workspace too small for ping-pong");
    auto *ping = reinterpret_cast<ComputeType *>(workspace);
    auto *pong = ping + outer_len * max_partials;

    // Frist round
    {
        const dim3 grid(tiles, outer_len, 1);
        reduce_row_kernel<ReduceOp, InType, ComputeType, ComputeType, UNROLL>
            <<<grid, BLOCK, 0, stream>>>(input, ping, outer_len, inner_len);
    }

    int64_t cur_inner = tiles;
    while (cur_inner > TILE_ELEMS) {
        const int64_t next_tiles = DIVUP<int64_t>(cur_inner, TILE_ELEMS);
        const dim3    grid(next_tiles, outer_len, 1);
        reduce_row_kernel<ReduceOp, ComputeType, ComputeType, ComputeType, UNROLL>
            <<<grid, BLOCK, 0, stream>>>(ping, pong, outer_len, cur_inner);
        std::swap(ping, pong);
        cur_inner = next_tiles;
    }

    // Last round
    {
        const dim3 grid(DIVUP<int64_t>(cur_inner, TILE_ELEMS), outer_len, 1);
        reduce_row_kernel<ReduceOp, ComputeType, OutType, ComputeType, UNROLL>
            <<<grid, BLOCK, 0, stream>>>(ping, output, outer_len, cur_inner);
    }
}

// TODO: Refactor
template <typename InType, typename OutType, typename ComputeType>
void reduce_row(PrimusTurboReduceOp reduce_op, const InType *input, OutType *output,
                const int64_t &outer_len, const int64_t &inner_len, const int64_t workspace_sizes,
                void *workspace, hipStream_t stream) {
    switch (reduce_op) {
    case PrimusTurboReduceOp::REDUCE_MAX:
        reduce_row_impl<MaxOp, InType, OutType, ComputeType>(input, output, outer_len, inner_len,
                                                             workspace_sizes, workspace, stream);
        return;
    case PrimusTurboReduceOp::REDUCE_ABS_MAX:
        reduce_row_impl<AbsMaxOp, InType, OutType, ComputeType>(input, output, outer_len, inner_len,
                                                                workspace_sizes, workspace, stream);
        return;
    }
    PRIMUS_TURBO_CHECK(false, "Unsupported reduce op");
}

template void reduce_row<bfloat16, float, float>(PrimusTurboReduceOp reduce_op,
                                                 const bfloat16 *input, float *output,
                                                 const int64_t &outer_len, const int64_t &inner_len,
                                                 const int64_t workspace_sizes, void *workspace,
                                                 hipStream_t stream);

} // namespace primus_turbo
