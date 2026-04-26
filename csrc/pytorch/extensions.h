// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
//
// See LICENSE for license information.

#pragma once

#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <ATen/miopen/Handle.h>
#include <ATen/native/DispatchStub.h>
#include <c10/macros/Macros.h>
#include <torch/extension.h>
#include <torch/torch.h>

#include <ATen/cuda/CUDAGraphsUtils.cuh>

#include "deep_ep/deep_ep.hpp"
#include "primus_turbo/common.h"

namespace primus_turbo::pytorch {

//==================================================================
//  Quantization
//==================================================================

std::vector<at::Tensor> quantize_fp8_tensorwise(const at::Tensor          input,
                                                const at::ScalarType      dest_dtype,
                                                c10::optional<at::Tensor> scale_opt);

std::vector<at::Tensor> quantize_fp8_tensorwise_meta(const at::Tensor          input,
                                                     const at::ScalarType      dest_dtype,
                                                     c10::optional<at::Tensor> scale_opt);

std::vector<at::Tensor> quantize_fp8_rowwise(const at::Tensor     input,
                                             const at::ScalarType dest_dtype, const int64_t axis,
                                             c10::optional<at::Tensor> scale_opt);

std::vector<at::Tensor> quantize_fp8_rowwise_meta(const at::Tensor          input,
                                                  const at::ScalarType      dest_dtype,
                                                  const int64_t             axis,
                                                  c10::optional<at::Tensor> scale_opt);

at::Tensor dequantize_fp8_tensorwise(const at::Tensor input, const at::Tensor scale_inv,
                                     const at::ScalarType dest_dtype);

at::Tensor dequantize_fp8_tensorwise_meta(const at::Tensor input, const at::Tensor scale_inv,
                                          const at::ScalarType dest_dtype);

std::vector<at::Tensor>
quantize_mxfp4_dual(const at::Tensor input, const at::ScalarType dest_dtype,
                    const bool rowwise_use_2d_block, const bool rowwise_use_sr,
                    const bool rowwise_use_rht, const bool colwise_use_2d_block,
                    const bool colwise_use_sr, const bool colwise_use_rht,
                    const bool shuffle_rowwise_scale = false, const bool shuffle_rowwise = false,
                    const bool shuffle_colwise_scale = false, const bool shuffle_colwise = false);

std::vector<at::Tensor> quantize_mxfp4_dual_meta(
    const at::Tensor input, const at::ScalarType dest_dtype, const bool rowwise_use_2d_block,
    const bool rowwise_use_sr, const bool rowwise_use_rht, const bool colwise_use_2d_block,
    const bool colwise_use_sr, const bool colwise_use_rht, const bool shuffle_rowwise_scale = false,
    const bool shuffle_rowwise = false, const bool shuffle_colwise_scale = false,
    const bool shuffle_colwise = false);

std::vector<at::Tensor> quantize_mxfp4(const at::Tensor input, const at::ScalarType dest_dtype,
                                       const int64_t axis, const bool use_2d_block,
                                       const bool use_sr, const bool use_rht,
                                       const bool shuffle_scale = false,
                                       const bool shuffle_out   = false);

std::vector<at::Tensor> quantize_mxfp4_meta(const at::Tensor input, const at::ScalarType dest_dtype,
                                            const int64_t axis, const bool use_2d_block,
                                            const bool use_sr, const bool use_rht,
                                            const bool shuffle_scale = false,
                                            const bool shuffle_out   = false);

std::vector<at::Tensor>
quantize_mxfp8_dual(const at::Tensor input, const at::ScalarType dest_dtype,
                    const bool rowwise_use_2d_block, const bool colwise_use_2d_block,
                    const bool shuffle_rowwise_scale = false, const bool shuffle_rowwise = false,
                    const bool shuffle_colwise_scale = false, const bool shuffle_colwise = false);

std::vector<at::Tensor> quantize_mxfp8_dual_meta(
    const at::Tensor input, const at::ScalarType dest_dtype, const bool rowwise_use_2d_block,
    const bool colwise_use_2d_block, const bool shuffle_rowwise_scale = false,
    const bool shuffle_rowwise = false, const bool shuffle_colwise_scale = false,
    const bool shuffle_colwise = false);

std::vector<at::Tensor> quantize_mxfp8(const at::Tensor input, const at::ScalarType dest_dtype,
                                       const int64_t axis, const bool use_2d_block,
                                       const bool shuffle_scale = false,
                                       const bool shuffle_out   = false);

std::vector<at::Tensor> quantize_mxfp8_meta(const at::Tensor input, const at::ScalarType dest_dtype,
                                            const int64_t axis, const bool use_2d_block,
                                            const bool shuffle_scale = false,
                                            const bool shuffle_out   = false);

//==================================================================
//  Shuffle
//==================================================================

at::Tensor shuffle_scale_impl(const at::Tensor scale, at::IntArrayRef layout);

at::Tensor shuffle_scale_impl_meta(const at::Tensor scale, at::IntArrayRef layout);

at::Tensor shuffle_weight_impl(const at::Tensor weight, at::IntArrayRef layout);

at::Tensor shuffle_weight_impl_meta(const at::Tensor weight, at::IntArrayRef layout);

//==================================================================
//  GEMM
//==================================================================

at::Tensor hipblaslt_gemm(at::Tensor A, at::Tensor B, const at::ScalarType out_dtype, bool transA,
                          bool transB, bool transC);

at::Tensor hipblaslt_gemm_meta(at::Tensor A, at::Tensor B, const at::ScalarType out_dtype,
                               bool transA, bool transB, bool transC);

at::Tensor hipblaslt_gemm_fp8(at::Tensor A, at::Tensor scaleA_inv, at::Tensor B,
                              at::Tensor scaleB_inv, const at::ScalarType out_dtype, bool transA,
                              bool transB, bool transC, const std::string &granularity);

at::Tensor hipblaslt_gemm_fp8_meta(at::Tensor A, at::Tensor scaleA_inv, at::Tensor B,
                                   at::Tensor scaleB_inv, const at::ScalarType out_dtype,
                                   bool transA, bool transB, bool transC,
                                   const std::string &granularity);

at::Tensor hipblaslt_gemm_fp4(at::Tensor A, at::Tensor scaleA_inv, at::Tensor B,
                              at::Tensor scaleB_inv, const at::ScalarType out_dtype, bool transA,
                              bool transB, bool transC, const std::string &granularity);

at::Tensor hipblaslt_gemm_fp4_meta(at::Tensor A, at::Tensor scaleA_inv, at::Tensor B,
                                   at::Tensor scaleB_inv, const at::ScalarType out_dtype,
                                   bool transA, bool transB, bool transC,
                                   const std::string &granularity);

at::Tensor ck_gemm_fp8(at::Tensor &a, at::Tensor &b, at::Tensor &a_scales, at::Tensor &b_scales,
                       const bool transA, const bool transB, at::ScalarType out_dtype,
                       const std::string &granularity);

at::Tensor ck_gemm_fp8_meta(at::Tensor &a, at::Tensor &b, at::Tensor &a_scales,
                            at::Tensor &b_scales, const bool transA, const bool transB,
                            at::ScalarType out_dtype, const std::string &granularity);

at::Tensor turbo_gemm_fp8(at::Tensor A, at::Tensor scaleA_inv, at::Tensor B, at::Tensor scaleB_inv,
                          const at::ScalarType out_dtype, bool transA, bool transB, bool transC,
                          const std::string &granularity);

at::Tensor turbo_gemm_fp8_meta(at::Tensor A, at::Tensor scaleA_inv, at::Tensor B,
                               at::Tensor scaleB_inv, const at::ScalarType out_dtype, bool transA,
                               bool transB, bool transC, const std::string &granularity);

//==================================================================
//  Normalization
//==================================================================

at::Tensor rmsnorm_fwd(const at::Tensor &input, const at::Tensor &gamma, const double eps);

at::Tensor rmsnorm_fwd_meta(const at::Tensor &input, const at::Tensor &gamma, const double eps);

std::vector<at::Tensor> rmsnorm_bwd(const at::Tensor &input, const at::Tensor &gamma,
                                    const at::Tensor &grad_output, const double eps);

std::vector<at::Tensor> rmsnorm_bwd_meta(const at::Tensor &input, const at::Tensor &gamma,
                                         const at::Tensor &grad_output, const double eps);

//==================================================================
//  Grouped GEMM
//==================================================================

at::Tensor ck_grouped_gemm(at::Tensor &a, at::Tensor &b, at::Tensor &group_lens,
                           at::Tensor &group_offs, const bool transA, const bool transB,
                           c10::optional<int64_t> num_cu);

at::Tensor ck_grouped_gemm_meta(at::Tensor &a, at::Tensor &b, at::Tensor &group_lens,
                                at::Tensor &group_offs, const bool transA, const bool transB,
                                c10::optional<int64_t> num_cu);

at::Tensor ck_grouped_gemm_variable_k(at::Tensor &a, at::Tensor &b, at::Tensor &group_lens,
                                      at::Tensor &group_offs, const bool transA, const bool transB,
                                      c10::optional<int64_t> num_cu);

at::Tensor ck_grouped_gemm_variable_k_meta(at::Tensor &a, at::Tensor &b, at::Tensor &group_lens,
                                           at::Tensor &group_offs, const bool transA,
                                           const bool transB, c10::optional<int64_t> num_cu);

at::Tensor ck_grouped_gemm_fp8(at::Tensor &a, at::Tensor &b, at::Tensor &a_scales,
                               at::Tensor &b_scales, at::Tensor &group_lens, at::Tensor &group_offs,
                               const bool transA, const bool transB, at::ScalarType out_dtype,
                               const std::string &granularity, c10::optional<int64_t> num_cu);

at::Tensor ck_grouped_gemm_fp8_meta(at::Tensor &a, at::Tensor &b, at::Tensor &a_scales,
                                    at::Tensor &b_scales, at::Tensor &group_lens,
                                    at::Tensor &group_offs, const bool transA, const bool transB,
                                    at::ScalarType out_dtype, const std::string &granularity,
                                    c10::optional<int64_t> num_cu);

at::Tensor ck_grouped_gemm_fp8_variable_k(at::Tensor &a, at::Tensor &b, at::Tensor &a_scales,
                                          at::Tensor &b_scales, at::Tensor &group_lens,
                                          at::Tensor &group_offs, const bool transA,
                                          const bool transB, at::ScalarType out_dtype,
                                          const std::string     &granularity,
                                          c10::optional<int64_t> num_cu);

at::Tensor ck_grouped_gemm_fp8_variable_k_meta(at::Tensor &a, at::Tensor &b, at::Tensor &a_scales,
                                               at::Tensor &b_scales, at::Tensor &group_lens,
                                               at::Tensor &group_offs, const bool transA,
                                               const bool transB, at::ScalarType out_dtype,
                                               const std::string     &granularity,
                                               c10::optional<int64_t> num_cu);

at::Tensor hipblaslt_grouped_gemm(at::Tensor &a, at::Tensor &b, at::Tensor &group_lens,
                                  at::Tensor &group_offs, const bool transA, const bool transB,
                                  const bool pre_sync);

at::Tensor hipblaslt_grouped_gemm_meta(at::Tensor &a, at::Tensor &b, at::Tensor &group_lens,
                                       at::Tensor &group_offs, const bool transA, const bool transB,
                                       const bool pre_sync);

at::Tensor hipblaslt_grouped_gemm_fp8(at::Tensor &a, at::Tensor &b, at::Tensor &a_scales,
                                      at::Tensor &b_scales, at::Tensor &group_lens,
                                      at::Tensor &group_offs, const bool transA, const bool transB,
                                      at::ScalarType out_dtype, const std::string &granularity,
                                      const bool pre_sync);

at::Tensor hipblaslt_grouped_gemm_fp8_meta(at::Tensor &a, at::Tensor &b, at::Tensor &a_scales,
                                           at::Tensor &b_scales, at::Tensor &group_lens,
                                           at::Tensor &group_offs, const bool transA,
                                           const bool transB, at::ScalarType out_dtype,
                                           const std::string &granularity, const bool pre_sync);

at::Tensor turbo_grouped_gemm_fp8(at::Tensor &a, at::Tensor &b, at::Tensor &a_scales,
                                  at::Tensor &b_scales, at::Tensor &group_lens,
                                  at::Tensor &group_offs, const bool transA, const bool transB,
                                  at::ScalarType out_dtype, const std::string &granularity);

at::Tensor turbo_grouped_gemm_fp8_meta(at::Tensor &a, at::Tensor &b, at::Tensor &a_scales,
                                       at::Tensor &b_scales, at::Tensor &group_lens,
                                       at::Tensor &group_offs, const bool transA, const bool transB,
                                       at::ScalarType out_dtype, const std::string &granularity);

at::Tensor turbo_grouped_gemm_variable_k_fp8(at::Tensor &lhs, at::Tensor &lhs_scales,
                                             at::Tensor &rhs, at::Tensor &rhs_scales,
                                             at::Tensor &group_lens, at::Tensor &group_offs,
                                             at::ScalarType out_dtype,
                                             const std::string &granularity);

at::Tensor turbo_grouped_gemm_variable_k_fp8_meta(at::Tensor &lhs, at::Tensor &lhs_scales,
                                                  at::Tensor &rhs, at::Tensor &rhs_scales,
                                                  at::Tensor &group_lens, at::Tensor &group_offs,
                                                  at::ScalarType out_dtype,
                                                  const std::string &granularity);

at::Tensor grouped_gemm_compute_offs(at::Tensor &group_lens);

at::Tensor grouped_gemm_compute_offs_meta(at::Tensor &group_lens);

//==================================================================
//  Runtime
//==================================================================

int64_t create_stream_with_cu_masks(const int device_id, const std::vector<uint32_t> &cu_masks);

void destroy_stream(const int device_id, const int64_t stream_ptr);

} // namespace primus_turbo::pytorch
