// Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// See LICENSE for license information.

#pragma once

#include "primus_turbo/common.h"
#include <hip/hip_runtime.h>
#include <optional>

namespace primus_turbo {

namespace detail {
// Memory layout shuffle parameters (for GEMM optimization)
constexpr int MXFP4_SHUFFLE_BN     = 16; // Block size for N dimension
constexpr int MXFP4_SHUFFLE_BK     = 32; // Block size for K dimension
constexpr int MXFP4_SHUFFLE_K_ELEM = 16; // Elements per K sub-block
} // namespace detail

void shuffle_e8m0_scale(uint8_t *scale, uint8_t *shuffled_scale, int64_t scale_M, int64_t scale_N,
                        int64_t scale_M_pad, int64_t scale_N_pad, hipStream_t stream);

template <typename DType>
void shuffle_weight(DType *weight, DType *shuffled_weight, int64_t weight_M, int64_t weight_N,
                    hipStream_t stream);

} // namespace primus_turbo
