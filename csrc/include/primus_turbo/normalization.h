// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
//
// See LICENSE for license information.

#pragma once

#include "primus_turbo/common.h"
#include "primus_turbo/device/reduce.cuh"
#include "primus_turbo/device/utils.cuh"

namespace primus_turbo {

template <typename T>
void rmsnorm_fwd_impl(const T *input, const T *gamma, T *output, const int64_t inner_len,
                      const int64_t outer_len, const float epsilon, hipStream_t stream);

template <typename T>
void rmsnorm_bwd_impl(const T *input, const T *gamma, const T *output_grad, T *input_grad,
                      T *gamma_grad, const int64_t inner_len, const int64_t outer_len,
                      const float epsilon, hipStream_t stream);

} // namespace primus_turbo
