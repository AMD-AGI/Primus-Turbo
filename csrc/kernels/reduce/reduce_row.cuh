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
          int UNROLL>
__global__ void reduce_row_kernel(const InType *input, OutType *output, const int64_t outer_len,
                                  const int64_t inner_len) {
    static constexpr int UNROLL_N = 16 / sizeof(InType);
    static constexpr int UNROLL_M = UNROLL / UNROLL_N;
    static_assert(UNROLL_N * UNROLL_M == UNROLL, "UNROLL_N * UNROLL_M must equal UNROLL");

    const int BLOCKSIZE = blockDim.x;
    const int blockid_x = blockIdx.x;
    const int blockid_y = blockIdx.y;
    const int tid       = threadIdx.x;

    const InType init_intype = ReduceOp<InType>::init();
    InType       ld_regs[UNROLL_M][UNROLL_N];

    const int64_t tile_elems  = static_cast<int64_t>(BLOCKSIZE) * UNROLL_N;
    const int64_t block_start = static_cast<int64_t>(blockid_x) * BLOCKSIZE * UNROLL;
    const InType *input_ptr   = input + blockid_y * inner_len + block_start;

    const bool full_tile = block_start + BLOCKSIZE * UNROLL <= inner_len;
    if (full_tile) {
#pragma unroll
        for (int mi = 0; mi < UNROLL_M; ++mi) {
            const int64_t offset = mi * tile_elems + tid * UNROLL_N;
            load_data<InType, UNROLL_N>(input_ptr + offset, ld_regs[mi]);
        }
    } else {
        for (int mi = 0; mi < UNROLL_M; ++mi) {
            const int64_t offset = mi * tile_elems + tid * UNROLL_N;
            #pragma unroll
            for (int ni = 0; ni < UNROLL_N; ++ni) {
                const int64_t g = block_start + offset + ni;
                ld_regs[mi][ni] = (g < inner_len) ? input_ptr[offset + ni] : init_intype;
            }
        }
    }

    const ComputeType init_ctype = ReduceOp<ComputeType>::init();
    ComputeType       reduce_regs[UNROLL_N];
#pragma unroll
    for (int j = 0; j < UNROLL_N; ++j) {
        ComputeType reg = init_ctype;
#pragma unroll
        for (int i = 0; i < UNROLL_M; ++i) {
            reg = ReduceOp<ComputeType>::op(reg, static_cast<ComputeType>(ld_regs[i][j]));
        }
        reduce_regs[j] = reg;
    }

    ComputeType ret = init_ctype;
// TODO: Opt
#pragma unroll
    for (int ni = 0; ni < UNROLL_N; ++ni) {
        ret = ReduceOp<ComputeType>::op(reduce_regs[ni], ret);
    }

    ret = BlockReduce<ReduceOp, ComputeType>(ret);

    OutType *output_ptr = output + blockid_y * gridDim.x + blockid_x;
    // Save
    if (tid == 0) {
        output_ptr[0] = static_cast<OutType>(ret);
    }
}

} // namespace primus_turbo
