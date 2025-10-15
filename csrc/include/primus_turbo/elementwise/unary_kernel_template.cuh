// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
//
// See LICENSE for license information.

#pragma once

#include "primus_turbo/common.h"
#include "primus_turbo/device/utils.cuh"
#include <hip/hip_runtime.h>

namespace primus_turbo {

template <int BLOCK, int UNROLL, typename InT, typename OutT, typename Op>
__launch_bounds__(BLOCK) __global__
    void unary_kernel(const InT *__restrict__ x, OutT *__restrict__ y, Op op,
                      PackedEltwiseConfig pack_cfg) {
    int64_t tid = static_cast<int64_t>(blockIdx.x) * BLOCK + threadIdx.x;

    if (tid < pack_cfg.nPack) {
        InT  ld_regs[UNROLL];
        OutT st_regs[UNROLL];
        load_data<InT, UNROLL>(x + tid * UNROLL, ld_regs);
#pragma unroll
        for (int i = 0; i < UNROLL; ++i) {
            st_regs[i] = static_cast<OutT>(op(ld_regs[i]));
        }
        store_data<OutT, UNROLL>(y + tid * UNROLL, st_regs);
    } else if (UNROLL > 1 && tid < pack_cfg.nThread) {
        const int64_t idx = tid + pack_cfg.nPack * (UNROLL - 1);
        y[idx]            = static_cast<OutT>(op(x[idx]));
    }
}

} // namespace primus_turbo
