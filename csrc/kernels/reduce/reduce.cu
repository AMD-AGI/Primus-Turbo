// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
//
// See LICENSE for license information.

#include "primus_turbo/device/reduce.cuh"
#include "primus_turbo/reduce.h"
#include "reduce_col.cuh"
#include "reduce_row.cuh"

namespace primus_turbo {

using namespace primus_turbo::dtype;

template <template <class> class ReduceOp, typename InType, typename OutType, typename ComputeType>
void reduce_row_impl(const InType *input, OutType *output, const int64_t &outer_len,
                     const int64_t &inner_len, const int64_t workspace_sizes, void *workspace,
                     hipStream_t stream) {
    constexpr int     BLOCK_SIZE = 256;
    constexpr int     UNROLL     = 32;
    constexpr int64_t TILE_ELEMS = BLOCK_SIZE * UNROLL;
    if (inner_len <= TILE_ELEMS) {
        const int64_t grid_i = DIVUP<int64_t>(inner_len, BLOCK_SIZE * UNROLL);
        const int64_t grid_o = outer_len;
        const dim3    grid(grid_i, grid_o, 1);
        reduce_row_kernel<ReduceOp, InType, OutType, ComputeType, BLOCK_SIZE, UNROLL>
            <<<grid, BLOCK_SIZE, 0, stream>>>(input, output, outer_len, inner_len);
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
        reduce_row_kernel<ReduceOp, InType, ComputeType, ComputeType, BLOCK_SIZE, UNROLL>
            <<<grid, BLOCK_SIZE, 0, stream>>>(input, ping, outer_len, inner_len);
    }

    int64_t cur_inner = tiles;
    while (cur_inner > TILE_ELEMS) {
        const int64_t next_tiles = DIVUP<int64_t>(cur_inner, TILE_ELEMS);
        const dim3    grid(next_tiles, outer_len, 1);
        reduce_row_kernel<ReduceOp, ComputeType, ComputeType, ComputeType, BLOCK_SIZE, UNROLL>
            <<<grid, BLOCK_SIZE, 0, stream>>>(ping, pong, outer_len, cur_inner);
        std::swap(ping, pong);
        cur_inner = next_tiles;
    }

    // Last round
    {
        const dim3 grid(DIVUP<int64_t>(cur_inner, TILE_ELEMS), outer_len, 1);
        reduce_row_kernel<ReduceOp, ComputeType, OutType, ComputeType, BLOCK_SIZE, UNROLL>
            <<<grid, BLOCK_SIZE, 0, stream>>>(ping, output, outer_len, cur_inner);
    }
}

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
    default:
        PRIMUS_TURBO_CHECK(false, "Unsupported reduce op");
        return;
    }
}

template <template <class> class ReduceOp, typename InType, typename OutType, typename ComputeType>
void reduce_col_impl(const InType *input, OutType *output, const int64_t &batch, const int64_t &m,
                     const int64_t &n, const int64_t workspace_sizes, void *workspace,
                     hipStream_t stream) {
    const int32_t BLOCK_SIZE = 256;
    const int32_t NUM_WARP   = BLOCK_SIZE / THREADS_PER_WARP;
    const int32_t UNROLL_M   = 8;
    const int32_t UNROLL_N   = sizeof(uint4) / sizeof(OutType);

    const int64_t grid_x = DIVUP<int64_t>(n, THREADS_PER_WARP * UNROLL_N);
    const int64_t grid_z = batch;

    // Single
    if (NUM_WARP * UNROLL_M >= m) {
        const int64_t grid_y = DIVUP<int64_t>(m, NUM_WARP * UNROLL_M);
        const dim3    grid_dim(grid_x, grid_y, grid_z);
        reduce_col_kernel<ReduceOp, InType, OutType, ComputeType, BLOCK_SIZE, UNROLL_M, UNROLL_N>
            <<<grid_dim, BLOCK_SIZE, 0, stream>>>(input, output, m, n);
        return;
    }

    // Multi round reduce
    int64_t next_m = DIVUP<int64_t>(m, NUM_WARP * UNROLL_M);
    auto   *ping   = reinterpret_cast<ComputeType *>(workspace);
    auto   *pong   = ping + batch * next_m * n;

    // Frist round
    {
        const dim3 grid_dim(grid_x, next_m, grid_z);
        reduce_col_kernel<ReduceOp, InType, ComputeType, ComputeType, BLOCK_SIZE, UNROLL_M,
                          UNROLL_N><<<grid_dim, BLOCK_SIZE, 0, stream>>>(input, ping, m, n);
    }

    int64_t cur_m = next_m;
    while (cur_m > NUM_WARP * UNROLL_M) {
        next_m = DIVUP<int64_t>(cur_m, NUM_WARP * UNROLL_M);
        const dim3 grid_dim(grid_x, next_m, grid_z);
        reduce_col_kernel<ReduceOp, ComputeType, ComputeType, ComputeType, BLOCK_SIZE, UNROLL_M,
                          UNROLL_N><<<grid_dim, BLOCK_SIZE, 0, stream>>>(ping, pong, cur_m, n);
        std::swap(ping, pong);
        cur_m = next_m;
    }

    // last
    {
        const dim3 grid_dim(grid_x, 1, grid_z);
        reduce_col_kernel<ReduceOp, ComputeType, OutType, ComputeType, BLOCK_SIZE, UNROLL_M,
                          UNROLL_N><<<grid_dim, BLOCK_SIZE, 0, stream>>>(ping, output, cur_m, n);
    }
}

template <typename InType, typename OutType, typename ComputeType>
void reduce_col(PrimusTurboReduceOp reduce_op, const InType *input, OutType *output,
                const int64_t &batch, const int64_t &m, const int64_t &n,
                const int64_t workspace_sizes, void *workspace, hipStream_t stream) {
    switch (reduce_op) {
    case PrimusTurboReduceOp::REDUCE_ABS_MAX:
        reduce_col_impl<AbsMaxOp, InType, OutType, ComputeType>(input, output, batch, m, n,
                                                                workspace_sizes, workspace, stream);
        return;
    default:
        PRIMUS_TURBO_CHECK(false, "Unsupported reduce op");
        return;
    }
}

#define DECL_REDUCE_ROW_INSTANCE(InType, OutType, ComputeType)                                     \
    template void reduce_row<InType, OutType, ComputeType>(                                        \
        PrimusTurboReduceOp reduce_op, const InType *input, OutType *output,                       \
        const int64_t &outer_len, const int64_t &inner_len, const int64_t workspace_sizes,         \
        void *workspace, hipStream_t stream);

DECL_REDUCE_ROW_INSTANCE(dtype::float16, dtype::float32, dtype::float32)
DECL_REDUCE_ROW_INSTANCE(dtype::bfloat16, dtype::float32, dtype::float32)
DECL_REDUCE_ROW_INSTANCE(dtype::float32, dtype::float32, dtype::float32)

#undef DECL_REDUCE_ROW_INSTANCE

#define DECL_REDUCE_COL_INSTANCE(InType, OutType, ComputeType)                                     \
    template void reduce_col<InType, OutType, ComputeType>(                                        \
        PrimusTurboReduceOp reduce_op, const InType *input, OutType *output, const int64_t &batch, \
        const int64_t &m, const int64_t &n, const int64_t workspace_sizes, void *workspace,        \
        hipStream_t stream);

DECL_REDUCE_COL_INSTANCE(dtype::float16, dtype::float32, dtype::float32)
DECL_REDUCE_COL_INSTANCE(dtype::bfloat16, dtype::float32, dtype::float32)
DECL_REDUCE_COL_INSTANCE(dtype::float32, dtype::float32, dtype::float32)

#undef DECL_REDUCE_COL_INSTANCE

} // namespace primus_turbo
