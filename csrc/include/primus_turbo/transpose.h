// Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// See LICENSE for license information.

#pragma once

#include "primus_turbo/common.h"
#include <hip/hip_runtime.h>

namespace primus_turbo {

// *************** Transpose ***************
// Batched 2D transpose of the last two dims of a contiguous buffer, for any
// element dtype :
//   in [batch, M, N] -> out [batch, N, M] (batch == 1 for 2D). Output contiguous.
void transpose_2d_impl(const void *x, void *y, const int64_t batch, const int64_t M,
                       const int64_t N, const int64_t itemsize, hipStream_t stream);

} // namespace primus_turbo
