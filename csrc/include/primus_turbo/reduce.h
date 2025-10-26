// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
//
// See LICENSE for license information.
#pragma once

#include "primus_turbo/common.h"
#include <hip/hip_runtime.h>

namespace primus_turbo {

enum class PrimusTurboReduceOp {
    REDUCE_SUM     = 0,
    REDUCE_MAX     = 1,
    REDUCE_MIN     = 2,
    REDUCE_ABS_MAX = 3,
};

template <typename ComputeType>
int64_t get_reduce_row_workspace_sizes(const int64_t &outer_len, const int64_t &inner_len) {
    const int     BLOCK  = 256;
    const int     UNROLL = 32;
    const int64_t cnt = DIVUP<int64_t>(inner_len, BLOCK * UNROLL) * 2; // For multi rounds ping-pong
    return (cnt == 1 ? 0 : sizeof(ComputeType) * cnt * outer_len);
}

template <typename ComputeType>
int64_t get_reduce_col_workspace_sizes(const int64_t batch, const int64_t m, const int64_t n) {
    const int BLOCK    = 512;
    const int NUM_WARP = BLOCK / THREADS_PER_WARP;
    const int UNROLL_M = 4;

    const int64_t cnt =
        batch * DIVUP<int64_t>(m, NUM_WARP * UNROLL_M) * n * 2; // For multi rounds ping-pong.
    return cnt * sizeof(ComputeType);
}

template <typename InType, typename OutType, typename ComputeType>
void reduce_row(PrimusTurboReduceOp reduce_op, const InType *input, OutType *output,
                const int64_t &outer_len, const int64_t &inner_len, const int64_t workspace_sizes,
                void *workspace, hipStream_t stream);

template <typename InType, typename OutType, typename ComputeType>
void reduce_col(PrimusTurboReduceOp reduce_op, const InType *input, OutType *output,
                const int64_t &batch, const int64_t &m, const int64_t &n,
                const int64_t workspace_sizes, void *workspace, hipStream_t stream);

} // namespace primus_turbo
