// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
//
// See LICENSE for license information.

#pragma once

#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/hip/HIPContext.h>
#include <ATen/hip/HIPGeneratorImpl.h>
#include <ATen/miopen/Handle.h>
#include <ATen/native/DispatchStub.h>
#include <c10/macros/Macros.h>
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>
#include <torch/csrc/distributed/c10d/Work.hpp>
#include <torch/extension.h>
#include <torch/torch.h>

#include <ATen/hip/HIPGraphsUtils.cuh>

#include "primus_turbo/common.h"

#include "deep_ep/deep_ep.hpp"

namespace primus_turbo::pytorch {

/* Quantization */
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

/* GEMM */

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

at::Tensor ck_gemm_fp8(at::Tensor &a, at::Tensor &b, at::Tensor &a_scales, at::Tensor &b_scales,
                       const bool transA, const bool transB, at::ScalarType out_dtype,
                       const std::string &granularity);

at::Tensor ck_gemm_fp8_meta(at::Tensor &a, at::Tensor &b, at::Tensor &a_scales,
                            at::Tensor &b_scales, const bool transA, const bool transB,
                            at::ScalarType out_dtype, const std::string &granularity);

std::vector<torch::Tensor> rendezvous_shmem(const std::string          &group_name,
                                            const std::vector<int64_t> &shape,
                                            c10::ScalarType             dtype);
/* Normalization */
at::Tensor rmsnorm_fwd(const at::Tensor &input, const at::Tensor &gamma, const double eps);

at::Tensor rmsnorm_fwd_meta(const at::Tensor &input, const at::Tensor &gamma, const double eps);

std::vector<at::Tensor> rmsnorm_bwd(const at::Tensor &input, const at::Tensor &gamma,
                                    const at::Tensor &grad_output, const double eps);

std::vector<at::Tensor> rmsnorm_bwd_meta(const at::Tensor &input, const at::Tensor &gamma,
                                         const at::Tensor &grad_output, const double eps);

/* Grouped Gemm */
at::Tensor grouped_gemm(at::Tensor &a, at::Tensor &b, at::Tensor &group_lens,
                        at::Tensor &group_offs, const bool transA, const bool transB,
                        c10::optional<int64_t> num_cu);

at::Tensor grouped_gemm_meta(at::Tensor &a, at::Tensor &b, at::Tensor &group_lens,
                             at::Tensor &group_offs, const bool transA, const bool transB,
                             c10::optional<int64_t> num_cu);

at::Tensor grouped_gemm_variable_k(at::Tensor &a, at::Tensor &b, at::Tensor &group_lens,
                                   at::Tensor &group_offs, const bool transA, const bool transB,
                                   c10::optional<int64_t> num_cu);

at::Tensor grouped_gemm_variable_k_meta(at::Tensor &a, at::Tensor &b, at::Tensor &group_lens,
                                        at::Tensor &group_offs, const bool transA,
                                        const bool transB, c10::optional<int64_t> num_cu);

at::Tensor grouped_gemm_fp8(at::Tensor &a, at::Tensor &b, at::Tensor &a_scales,
                            at::Tensor &b_scales, at::Tensor &group_lens, at::Tensor &group_offs,
                            const bool transA, const bool transB, at::ScalarType out_dtype,
                            const std::string &granularity, c10::optional<int64_t> num_cu);

at::Tensor grouped_gemm_fp8_meta(at::Tensor &a, at::Tensor &b, at::Tensor &a_scales,
                                 at::Tensor &b_scales, at::Tensor &group_lens,
                                 at::Tensor &group_offs, const bool transA, const bool transB,
                                 at::ScalarType out_dtype, const std::string &granularity,
                                 c10::optional<int64_t> num_cu);

at::Tensor grouped_gemm_fp8_variable_k(at::Tensor &a, at::Tensor &b, at::Tensor &a_scales,
                                       at::Tensor &b_scales, at::Tensor &group_lens,
                                       at::Tensor &group_offs, const bool transA, const bool transB,
                                       at::ScalarType out_dtype, const std::string &granularity,
                                       c10::optional<int64_t> num_cu);

at::Tensor grouped_gemm_fp8_variable_k_meta(at::Tensor &a, at::Tensor &b, at::Tensor &a_scales,
                                            at::Tensor &b_scales, at::Tensor &group_lens,
                                            at::Tensor &group_offs, const bool transA,
                                            const bool transB, at::ScalarType out_dtype,
                                            const std::string     &granularity,
                                            c10::optional<int64_t> num_cu);

at::Tensor grouped_gemm_compute_offs(at::Tensor &group_lens);

at::Tensor grouped_gemm_compute_offs_meta(at::Tensor &group_lens);

/* Runtime */
int64_t create_stream_with_cu_masks(const int device_id, const std::vector<uint32_t> &cu_masks);

void destroy_stream(const int device_id, const int64_t stream_ptr);

/* Torch Distributed */
c10::intrusive_ptr<c10d::Work> dma_all_gather_into_tensor(at::Tensor       output_tensor,
                                                          const at::Tensor input_tensor,
                                                          c10::intrusive_ptr<c10d::ProcessGroup> pg,
                                                          const std::string &group_tag);

} // namespace primus_turbo::pytorch
