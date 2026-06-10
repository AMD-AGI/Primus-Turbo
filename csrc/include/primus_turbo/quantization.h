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

// Segment-padded group offsets (each segment rounded up to block_size), on-device.
template <typename IndexType>
void compute_padded_group_offs(const IndexType *group_lens_ptr, IndexType *padded_lens_ptr,
                               IndexType *padded_offs_ptr, const int64_t group_num,
                               const IndexType block_size, hipStream_t stream);

// Fused single-pass row + segment-padded col blockwise FP8 quant (grouped fwd/bwd).
template <typename FType, typename QType>
void quantize_blockwise_segment_m_row_col_impl(const FType *x, QType *y_row, QType *y_col_padded,
                                               float *scales_row, float *scales_col_padded,
                                               const int64_t *group_offs,
                                               const int64_t *padded_group_offs, const int64_t M_in,
                                               const int64_t N, const int64_t M_padded_max,
                                               const int num_groups, const float fp8_max,
                                               hipStream_t stream);

// Blockwise FP8 weight quant: [B, M, N] (or [M, N]), one scalar scale per [128,128] tile.
template <typename FType, typename QType>
void quantize_blockwise_for_weight_impl(const FType *w, QType *w_fp8, float *w_scales_inv,
                                        const int64_t B, const int64_t M, const int64_t N,
                                        const float fp8_max, hipStream_t stream);

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

enum class QuantizeMode { ROWWISE, COLWISE };

// MX format: each scale covers 32 elements
constexpr int MXFP4_BLOCK_SIZE = 32;
constexpr int MXFP8_BLOCK_SIZE = 32;

// Padding alignment expected for the public ``padding_align_size`` op argument.
// Must stay in sync with ``MXFP4_PADDING_ALIGN_SIZE`` / ``MXFP8_PADDING_ALIGN_SIZE``
// declared in ``primus_turbo/pytorch/core/low_precision.py``.
constexpr int MXFP4_PADDING_ALIGN_SIZE = 128;
constexpr int MXFP8_PADDING_ALIGN_SIZE = 128;

struct ScalingRecipe {
    bool use_2d_block = false;
    bool use_sr       = false;
    bool use_rht      = false;

    bool shuffle_scale = false;
    bool shuffle_out   = false;
};

constexpr int FP32_MANTISSA_BITS     = 23;
constexpr int FP32_EXPONENT_BITS     = 8;
constexpr int FP32_EXPONENT_EXP_BIAS = 127;

constexpr int FP4_MANTISSA_BITS   = 1;
constexpr int FP4_EXPONENT_BITS   = 2;
constexpr int FP4_TARGET_MAX_POW2 = 2;

constexpr int FP8E5M2_MANTISSA_BITS   = 2;
constexpr int FP8E5M2_EXPONENT_BITS   = 5;
constexpr int FP8E5M2_TARGET_MAX_POW2 = 15;

constexpr int FP8E4M3_MANTISSA_BITS = 3;
constexpr int FP8E4M3_EXPONENT_BITS = 4;
// NOTE: The max value of fp8 e4m3 ocp is 448.
constexpr float FP8E4M3_MAX             = 448.0;
constexpr int   FP8E4M3_TARGET_MAX_POW2 = 8;
// NOTE: The max value of fp8 e4m3 fnuz is 240.
constexpr float FP8E4M3_FNUZ_MAX             = 240.0;
constexpr int   FP8E4M3_FNUZ_TARGET_MAX_POW2 = 7;

constexpr int E8M0_EXPONENT_BIAS = 127;

} // namespace detail

template <typename DType>
void quantize_mxfp4_dual_impl(const DType *input, dtype::float4x2_e2m1 *rowwise_output,
                              uint8_t *rowwise_scale, dtype::float4x2_e2m1 *colwise_output,
                              uint8_t *colwise_scale, int M, int N, int M_pad, int N_pad,
                              int rowwise_scale_stride, int colwise_scale_stride,
                              int rowwise_scale_N, int rowwise_scale_M_pad, int rowwise_scale_N_pad,
                              int colwise_scale_M, int colwise_scale_N, int colwise_scale_M_pad,
                              int colwise_scale_N_pad, detail::ScalingRecipe rowwise_recipe,
                              detail::ScalingRecipe colwise_recipe, hipStream_t stream);

template <typename DType>
void quantize_mxfp4_impl(const DType *input, dtype::float4x2_e2m1 *output, uint8_t *scale,
                         detail::QuantizeMode mode, int M, int N, int M_pad, int N_pad,
                         int scale_stride, int scale_N, int scale_M_pad, int scale_N_pad,
                         detail::ScalingRecipe recipe, hipStream_t stream);

template <typename IType, typename OType>
void quantize_mxfp8_dual_impl(const IType *input, OType *rowwise_output, uint8_t *rowwise_scale,
                              OType *colwise_output, uint8_t *colwise_scale, int M, int N,
                              int M_pad, int N_pad, int rowwise_scale_stride,
                              int colwise_scale_stride, int rowwise_scale_N,
                              int rowwise_scale_M_pad, int rowwise_scale_N_pad, int colwise_scale_M,
                              int colwise_scale_N, int colwise_scale_M_pad, int colwise_scale_N_pad,
                              detail::ScalingRecipe rowwise_recipe,
                              detail::ScalingRecipe colwise_recipe, hipStream_t stream);

template <typename IType, typename OType>
void quantize_mxfp8_impl(const IType *input, OType *output, uint8_t *scale,
                         detail::QuantizeMode mode, int M, int N, int M_pad, int N_pad,
                         int scale_stride, int scale_N, int scale_M_pad, int scale_N_pad,
                         detail::ScalingRecipe recipe, hipStream_t stream);

// *************** DeQuantize ***************
template <typename FType, typename QType, typename ComputeType = float>
void dequantize_tensorwise_impl(const QType *x, const float *scale_inv, FType *y, const int64_t n,
                                hipStream_t stream);

// Rowwise dequantize when the per-row dim is the innermost (last) dim.
// scale_inv has shape [outer_len] (one scalar per row).
template <typename FType, typename QType, typename ComputeType = float>
void dequantize_rowwise_row_major_impl(const QType *x, const float *scale_inv, FType *y,
                                       const int64_t outer_len, const int64_t inner_len,
                                       hipStream_t stream);

// Rowwise dequantize when the per-row dim is a middle dim.
// Input is viewed as [B, M, N], scale_inv has shape [B, N] (broadcast across M).
template <typename FType, typename QType, typename ComputeType = float>
void dequantize_rowwise_col_major_impl(const QType *x, const float *scale_inv, FType *y,
                                       const int64_t batch, const int64_t m, const int64_t n,
                                       hipStream_t stream);

} // namespace primus_turbo
