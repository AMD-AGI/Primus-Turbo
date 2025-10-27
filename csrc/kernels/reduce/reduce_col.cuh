// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
//
// See LICENSE for license information.
#pragma once

#include "primus_turbo/common.h"
#include "primus_turbo/device/reduce.cuh"
#include "primus_turbo/device/utils.cuh"
#include <hip/hip_runtime.h>

namespace primus_turbo {

template <template <class> class ReduceOp, typename InType, typename OutType, typename ComputeType,
          int BLOCK_SIZE, int UNROLL_M, int UNROLL_N>
__launch_bounds__(BLOCK_SIZE) __global__
    void reduce_col_kernel(const InType *__restrict__ input_ptr, OutType *__restrict__ output_ptr,
                           const int64_t m, const int64_t n) {
    const int warp_id = threadIdx.x / THREADS_PER_WARP;
    const int lane_id = threadIdx.x % THREADS_PER_WARP;

    const int     bid_x    = blockIdx.x;
    const int     bid_y    = blockIdx.y;
    const int     bid_z    = blockIdx.z;
    constexpr int NUM_WARP = BLOCK_SIZE / THREADS_PER_WARP;

    const int64_t reduce_m = DIVUP<int64_t>(m, NUM_WARP * UNROLL_M);

    const int64_t offset_n      = bid_x * THREADS_PER_WARP * UNROLL_N + lane_id * UNROLL_N;
    const int64_t offset_m      = bid_y * NUM_WARP * UNROLL_M + warp_id * UNROLL_M;
    const int64_t offset_input  = bid_z * m * n + offset_m * n + offset_n;
    const int64_t offset_output = bid_z * reduce_m * n + bid_y * n + offset_n;

    input_ptr += offset_input;

    const InType init_intype = ReduceOp<InType>::init();
    InType       ld_regs[UNROLL_M][UNROLL_N];
    ComputeType  reduce_regs[UNROLL_N];

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
        for (int mi = 0; mi < UNROLL_M; ++mi) {
#pragma unroll
            for (int ni = 0; ni < UNROLL_N; ++ni) {
                ld_regs[mi][ni] = ((offset_m + mi < m) && (offset_n + ni < n))
                                      ? input_ptr[mi * n + ni]
                                      : init_intype;
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

    // TODO: Opt reduce in smem
    if (NUM_WARP > 1) {
        __shared__ ComputeType smem[NUM_WARP - 1][THREADS_PER_WARP * UNROLL_N];

        if (warp_id != 0) {
            store_data<ComputeType, UNROLL_N>(&smem[warp_id - 1][lane_id * UNROLL_N], reduce_regs);
        }

        __syncthreads();

        if (warp_id == 0) {
            ComputeType lds_regs[UNROLL_N];
#pragma unroll
            for (int i = 0; i < NUM_WARP - 1; ++i) {
                load_data<ComputeType, UNROLL_N>(&smem[i][lane_id * UNROLL_N], lds_regs);
#pragma unroll
                for (int ni = 0; ni < UNROLL_N; ++ni) {
                    reduce_regs[ni] = ReduceOp<ComputeType>::op(lds_regs[ni], reduce_regs[ni]);
                }
            }
        }
    }

    if (warp_id == 0) {
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
}

} // namespace primus_turbo
