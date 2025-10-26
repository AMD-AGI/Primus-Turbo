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

//**********************************************************
//**********************************************************

template <template <class> class ReduceOp, typename InType, typename OutType, typename ComputeType,
          int BLOCK_SIZE, int UNROLL_M, int UNROLL_N>
__launch_bounds__(BLOCK_SIZE) __global__
    void reduce_col_kernel(const InType *__restrict__ input_ptr, OutType *__restrict__ output_ptr,
                           const int64_t m, const int64_t n) {
    const int warp_id = threadIdx.x / THREADS_PER_WARP;
    const int lane_id = threadIdx.x % THREADS_PER_WARP;

    const int bid_x    = blockIdx.x;
    const int bid_y    = blockIdx.y;
    const int bid_z    = blockIdx.z;
    const int NUM_WARP = BLOCK_SIZE / THREADS_PER_WARP;

    const int64_t reduce_m = DIVUP<int64_t>(m, NUM_WARP * UNROLL_M);

    const int64_t offset_n      = bid_x * THREADS_PER_WARP * UNROLL_N + lane_id * UNROLL_N;
    const int64_t offset_m      = bid_y * NUM_WARP * UNROLL_M + warp_id * UNROLL_M;
    const int64_t offset_input  = bid_z * m * n + offset_m * n + offset_n;
    const int64_t offset_output = bid_z * reduce_m * n + bid_y * n + offset_n;

    // if (warp_id == 0) {
    //     printf("lane_id=%d, offset_m=%d, offset_n=%d offset_input=%d offset_output=%d\n",
    //     lane_id,
    //            offset_m, offset_n, offset_input, offset_output);
    // }

    input_ptr += offset_input;

    const InType init_intype = ReduceOp<InType>::init();
    InType       ld_regs[UNROLL_M][UNROLL_N];
    ComputeType  reduce_regs[UNROLL_N];
#pragma unroll
    for (int mi = 0; mi < UNROLL_M; ++mi) {
#pragma unroll
        for (int ni = 0; ni < UNROLL_N; ++ni) {
            ld_regs[mi][ni] = init_intype;
        }
    }

#pragma unroll
    for (int i = 0; i < UNROLL_N; ++i) {
        reduce_regs[i] = ReduceOp<ComputeType>::init();
    }

    const bool full_tile_m = (offset_m + UNROLL_M) <= m;
    const bool full_tile_n = (offset_n + UNROLL_N) <= n;
    if (full_tile_m && full_tile_n) {
#pragma unroll
        for (int mi = 0; mi < UNROLL_M; ++mi) {
            load_data<InType, UNROLL_N>(input_ptr + mi * n, ld_regs[mi]);
        }
    } else {
        // TODO
        const int32_t m_remaining = static_cast<int32_t>(m - offset_m);
        const int32_t m_valid     = m_remaining > UNROLL_M ? UNROLL_M : m_remaining;
        for (int mi = 0; mi < m_valid; ++mi) {
            if (full_tile_n) {
                load_data<InType, UNROLL_N>(input_ptr + mi * n, ld_regs[mi]);
            } else {
                for (int ni = 0; ni < UNROLL_N; ++ni) {
                    ld_regs[mi][ni] = (offset_n + ni) < n ? input_ptr[mi * n + ni] : init_intype;
                }
            }
        }
    }

#pragma unroll
    for (int mi = 0; mi < UNROLL_M; ++mi) {
#pragma unroll
        for (int ni = 0; ni < UNROLL_N; ++ni) {
            reduce_regs[ni] = ReduceOp<ComputeType>::op(reduce_regs[ni],
                                                        static_cast<ComputeType>(ld_regs[mi][ni]));
        }
    }

    // if (lane_id == 0) {
    //     printf("warp_id=%d: %f %f %f %f  -> %f\n", warp_id,
    //     static_cast<ComputeType>(ld_regs[0][0]),
    //            static_cast<ComputeType>(ld_regs[1][0]), static_cast<ComputeType>(ld_regs[2][0]),
    //            static_cast<ComputeType>(ld_regs[3][0]), reduce_regs[0], );
    // }

    __shared__ ComputeType smem[NUM_WARP][THREADS_PER_WARP * UNROLL_N];
    store_data<ComputeType, UNROLL_N>(&smem[warp_id][lane_id * UNROLL_N], reduce_regs);
    __syncthreads();
    if (warp_id == 0) {
        ComputeType lds_regs[UNROLL_N];
#pragma unroll
        for (int i = 0; i < NUM_WARP; ++i) {
            load_data<ComputeType, UNROLL_N>(&smem[i][lane_id * UNROLL_N], lds_regs);
#pragma unroll
            for (int ni = 0; ni < UNROLL_N; ++ni) {
                reduce_regs[ni] = ReduceOp<ComputeType>::op(lds_regs[ni], reduce_regs[ni]);
            }
        }

        OutType st_regs[UNROLL_N];
#pragma unroll
        for (int i = 0; i < UNROLL_N; ++i) {
            st_regs[i] = static_cast<OutType>(reduce_regs[i]);
        }

        if (full_tile_n) {
            store_data<OutType, UNROLL_N>(output_ptr + offset_output, st_regs);
        } else {
            for (int ni = 0; ni < UNROLL_N; ++ni) {
                if (offset_n + ni < n) {
                    output_ptr[offset_output + ni] = st_regs[ni];
                }
            }
        }
    }

    // TODO: Opt
    // #pragma unroll
    //     for (int ni = 0; ni < UNROLL_N; ++ni) {
    //         ComputeType regs[UNROLL_M];
    // #pragma unroll
    //         for (int mi = 0; mi < UNROLL_M; ++mi) {
    //             regs[mi] = static_cast<ComputeType>(ld_regs[mi][ni]);
    //         }
    //     }
}

template <template <class> class ReduceOp, typename InType, typename OutType, typename ComputeType>
void reduce_col_impl(const InType *input, OutType *output, const int64_t &batch, const int64_t &m,
                     const int64_t &n, const int64_t workspace_sizes, void *workspace,
                     hipStream_t stream) {
    const int32_t BLOCK_SIZE = 512;
    const int32_t NUM_WARP   = BLOCK_SIZE / THREADS_PER_WARP;
    const int32_t UNROLL_M   = 4;
    const int32_t UNROLL_N   = sizeof(uint4) / sizeof(OutType);

    const int64_t grid_x = DIVUP<int64_t>(n, THREADS_PER_WARP * UNROLL_N);
    const int64_t grid_z = batch;

    // Single
    if (NUM_WARP * UNROLL_M >= m) {
        const int64_t grid_y = DIVUP<int64_t>(m, NUM_WARP * UNROLL_M);
        const dim3    grid_dim(grid_x, grid_y, grid_z);
        // printf("g.x=%d  g.y=%d  g.z=%d\n", grid_dim.x, grid_dim.y, grid_dim.z);
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
        // printf("F g.x=%d  g.y=%d  g.z=%d\n", grid_dim.x, grid_dim.y, grid_dim.z);
        reduce_col_kernel<ReduceOp, InType, ComputeType, ComputeType, BLOCK_SIZE, UNROLL_M,
                          UNROLL_N><<<grid_dim, BLOCK_SIZE, 0, stream>>>(input, ping, m, n);
    }

    int64_t cur_m = next_m;
    while (cur_m > NUM_WARP * UNROLL_M) {
        next_m = DIVUP<int64_t>(cur_m, NUM_WARP * UNROLL_M);
        const dim3 grid_dim(grid_x, next_m, grid_z);
        // printf("M g.x=%d  g.y=%d  g.z=%d\n", grid_dim.x, grid_dim.y, grid_dim.z);
        reduce_col_kernel<ReduceOp, ComputeType, ComputeType, ComputeType, BLOCK_SIZE, UNROLL_M,
                          UNROLL_N><<<grid_dim, BLOCK_SIZE, 0, stream>>>(ping, pong, cur_m, n);
        std::swap(ping, pong);
        cur_m = next_m;
    }

    // last
    {
        const dim3 grid_dim(grid_x, 1, grid_z);
        // printf("L g.x=%d  g.y=%d  g.z=%d\n", grid_dim.x, grid_dim.y, grid_dim.z);
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

#define DECL_REDUCE_COL_INSTANCE(InType, OutType, ComputeType)                                     \
    template void reduce_col<InType, OutType, ComputeType>(                                        \
        PrimusTurboReduceOp reduce_op, const InType *input, OutType *output, const int64_t &batch, \
        const int64_t &m, const int64_t &n, const int64_t workspace_sizes, void *workspace,        \
        hipStream_t stream);

DECL_REDUCE_COL_INSTANCE(dtype::float16, dtype::float32, dtype::float32)
DECL_REDUCE_COL_INSTANCE(dtype::bfloat16, dtype::float32, dtype::float32)
DECL_REDUCE_COL_INSTANCE(dtype::float32, dtype::float32, dtype::float32)
#undef DECL_REDUCE_COL_INSTANCE

template void reduce_row<float, float, float>(PrimusTurboReduceOp reduce_op, const float *input,
                                              float *output, const int64_t &outer_len,
                                              const int64_t &inner_len,
                                              const int64_t workspace_sizes, void *workspace,
                                              hipStream_t stream);

template void reduce_row<float16, float, float>(PrimusTurboReduceOp reduce_op, const float16 *input,
                                                float *output, const int64_t &outer_len,
                                                const int64_t &inner_len,
                                                const int64_t workspace_sizes, void *workspace,
                                                hipStream_t stream);

template void reduce_row<bfloat16, float, float>(PrimusTurboReduceOp reduce_op,
                                                 const bfloat16 *input, float *output,
                                                 const int64_t &outer_len, const int64_t &inner_len,
                                                 const int64_t workspace_sizes, void *workspace,
                                                 hipStream_t stream);

} // namespace primus_turbo
