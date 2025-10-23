// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
//
// See LICENSE for license information.

#pragma once

#include "primus_turbo/common.h"
#include "primus_turbo/device/utils.cuh"
#include <hip/hip_runtime.h>

namespace primus_turbo {

// TODO: normal kernel

// Boradcast kernel

template <int NumDim, int BLOCK, int UNROLL>
PRIMUS_TURBO_DEVICE void
binary_broadcast_offset(const Array<IntDivMod<int64_t>, NumDim> &z_div_mod,
                        const Array<int64_t, NumDim>            &x_stride,
                        const Array<int64_t, NumDim> &y_stride, const int64_t (&z_offset)[UNROLL],
                        int64_t (&x_offset)[UNROLL], int64_t (&y_offset)[UNROLL]) {
    int64_t div_tmp[UNROLL];
#pragma unroll
    for (int i = 0; i < UNROLL; ++i) {
        x_offset[i] = 0;
        y_offset[i] = 0;
        div_tmp[i]  = z_offset[i];
    }

#pragma unroll
    for (int i = 0; i < NumDim; ++i) {
#pragma unroll
        for (int j = 0; j < UNROLL; ++j) {
            auto    dm  = z_div_mod[i].div_mod(div_tmp[j]);
            int64_t idx = dm.mod;
            div_tmp[j]  = dm.div;
            x_offset[j] += idx * x_stride[i];
            y_offset[j] += idx * y_stride[i];
        }
    }
}

// TODO: Opt Perf
template <int NumDim, int BLOCK, int UNROLL, typename InT0, typename InT1, typename OutT,
          typename ComputeType, typename OpType>
__launch_bounds__(BLOCK) __global__
    void binary_broadcast_kernel(const InT0 *__restrict__ x, const InT1 *__restrict__ y,
                                 OutT *__restrict__ z, const int64_t n,
                                 const Array<IntDivMod<int64_t>, NumDim> z_div_mod,
                                 const Array<int64_t, NumDim>            x_stride,
                                 const Array<int64_t, NumDim> y_stride, OpType op) {
    int64_t x_offset[UNROLL];
    int64_t y_offset[UNROLL];
    int64_t z_offset[UNROLL];

    const int64_t z_offset_start = static_cast<int64_t>(blockIdx.x) * BLOCK * UNROLL + threadIdx.x;
#pragma unroll
    for (int i = 0; i < UNROLL; ++i) {
        z_offset[i] = z_offset_start + i * BLOCK;
    }

    binary_broadcast_offset<NumDim, BLOCK, UNROLL>(z_div_mod, x_stride, y_stride, z_offset,
                                                   x_offset, y_offset);

    InT0 x_regs[UNROLL];
    InT1 y_regs[UNROLL];
    OutT z_regs[UNROLL];

    int64_t z_tile_count = n > z_offset_start ? DIVUP<int64_t>(n - z_offset_start, BLOCK) : 0;
    if (z_tile_count >= UNROLL) {
#pragma unroll
        for (int i = 0; i < UNROLL; ++i) {
            x_regs[i] = x[x_offset[i]];
            y_regs[i] = y[y_offset[i]];
        }

#pragma unroll
        for (int i = 0; i < UNROLL; ++i) {
            ComputeType val =
                op(static_cast<ComputeType>(x_regs[i]), static_cast<ComputeType>(y_regs[i]));
            z_regs[i] = static_cast<OutT>(val);
        }

#pragma unroll
        for (int i = 0; i < UNROLL; ++i) {
            z[z_offset[i]] = z_regs[i];
        }
    } else {
#pragma unroll
        for (int i = 0; i < UNROLL; ++i) {
            if (i < z_tile_count) {
                x_regs[i] = x[x_offset[i]];
                y_regs[i] = y[y_offset[i]];
            }
        }

#pragma unroll
        for (int i = 0; i < UNROLL; ++i) {
            ComputeType val =
                op(static_cast<ComputeType>(x_regs[i]), static_cast<ComputeType>(y_regs[i]));
            z_regs[i] = static_cast<OutT>(val);
        }

#pragma unroll
        for (int i = 0; i < UNROLL; ++i) {
            if (i < z_tile_count) {
                z[z_offset[i]] = z_regs[i];
            }
        }
    }
}

} // namespace primus_turbo
