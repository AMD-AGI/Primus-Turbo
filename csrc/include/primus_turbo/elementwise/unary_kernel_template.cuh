// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
//
// See LICENSE for license information.

#pragma once

#include "primus_turbo/common.h"
#include <hip/hip_runtime.h>

namespace primus_turbo {

// TODO: Opt performance
template <typename InT, typename OutT, typename Op>
__global__ void unary_kernel(const InT *__restrict__ x, OutT *__restrict__ y, int64_t n, Op op) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        y[idx] = static_cast<OutT>(op(x[idx]));
    }
}

} // namespace primus_turbo
