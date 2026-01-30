// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
//
// See LICENSE for license information.

#pragma once

#include "primus_turbo/common.h"
#include <hip/hip_runtime.h>

namespace primus_turbo {

template <typename T>
void compute_scale_from_amax(const T *amax, const T q_max, T *scale, T *scale_inv, const int64_t n,
                             hipStream_t stream, const float eps = 1e-12);

// *************** Quantize ***************
template <typename FType, typename QType, typename ComputeType = float>
void quantize_tensorwise_impl(const FType *x, const float *scale, QType *y, const int64_t n,
                              hipStream_t stream);

template <typename FType, typename QType, typename ComputeType = float,
          bool PreComputeScale = false>
void quantize_rowwise_row_major_impl(const FType *x, float *scale, float *scale_inv, QType *y,
                                     const int64_t outer_len, const int64_t inner_len,
                                     hipStream_t stream);

template <typename FType, typename QType, typename ComputeType = float>
void quantize_rowwise_col_major_impl(const FType *x, float *scale, float *scale_inv, QType *y,
                                     const int64_t batch, const int64_t m, const int64_t n,
                                     hipStream_t stream);

// *************** DeQuantize ***************
template <typename FType, typename QType, typename ComputeType = float>
void dequantize_tensorwise_impl(const QType *x, const float *scale_inv, FType *y, const int64_t n,
                                hipStream_t stream);

// *************** Fused Cast + Transpose Tensorwise ***************
// Uses quantize_tensorwise_impl + transpose kernel
template <typename FType, typename QType, typename ComputeType = float>
void cast_transpose_tensorwise_impl(const FType *input, QType *output, QType *output_t,
                                    float *scale, float *scale_inv, float *amax, const int64_t M,
                                    const int64_t N, const float q_max, hipStream_t stream);

// Transpose kernel for FP8 tensors
template <int TILE_DIM, int BLOCK_ROWS, typename QType>
void transpose_fp8_kernel(const QType *input, QType *output, int64_t M, int64_t N,
                          hipStream_t stream);

} // namespace primus_turbo
