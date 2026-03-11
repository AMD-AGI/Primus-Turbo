// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
//
// See LICENSE for license information.

#pragma once

#include "primus_turbo/common.h"
#include <hip/hip_runtime.h>
#include <optional>

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

namespace detail {

// MXFP4 format: each scale covers 32 elements
constexpr int MXFP4_BLOCK_SIZE = 32;

// Memory layout shuffle parameters (for GEMM optimization)
constexpr int MXFP4_SHUFFLE_BN     = 16; // Block size for N dimension
constexpr int MXFP4_SHUFFLE_BK     = 32; // Block size for K dimension
constexpr int MXFP4_SHUFFLE_K_ELEM = 16; // Elements per K sub-block

struct MXScalingRecipe {
    bool use_2d_block   = false;
    bool use_sr         = false;
    bool use_rht        = false;
    bool shuffle_scale  = false;
    bool shuffle_output = false;
};

constexpr int FP32_MANTISSA_BITS     = 23;
constexpr int FP32_EXPONENT_BITS     = 8;
constexpr int FP32_EXPONENT_EXP_BIAS = 127;

constexpr int FP4_MANTISSA_BITS   = 1;
constexpr int FP4_EXPONENT_BITS   = 2;
constexpr int FP4_TARGET_MAX_POW2 = 2;

constexpr int E8M0_EXPONENT_BIAS = 127;

} // namespace detail

template <typename DType>
void quantize_mxfp4_dual_shuffle_impl(
    const DType *input, dtype::float4x2_e2m1 *rowwise_output, uint8_t *rowwise_scale,
    dtype::float4x2_e2m1 *colwise_output, uint8_t *colwise_scale, int M, int N,
    int rowwise_scale_stride, int colwise_scale_stride, int rowwise_scale_N,
    int rowwise_scale_M_pad, int rowwise_scale_N_pad, int colwise_scale_M, int colwise_scale_N,
    int colwise_scale_M_pad, int colwise_scale_N_pad, detail::MXScalingRecipe rowwise_recipe,
    detail::MXScalingRecipe colwise_recipe, hipStream_t stream);

// *************** DeQuantize ***************
template <typename FType, typename QType, typename ComputeType = float>
void dequantize_tensorwise_impl(const QType *x, const float *scale_inv, FType *y, const int64_t n,
                                hipStream_t stream);

} // namespace primus_turbo
