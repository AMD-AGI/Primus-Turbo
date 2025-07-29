#pragma once

#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/hip/HIPContext.h>
#include <ATen/hip/HIPGeneratorImpl.h>
#include <ATen/miopen/Handle.h>
#include <ATen/native/DispatchStub.h>
#include <c10/macros/Macros.h>
#include <torch/extension.h>
#include <torch/torch.h>

#include <ATen/hip/HIPGraphsUtils.cuh>

#include "primus_turbo/common.h"

#include "deep_ep/deep_ep.hpp"

namespace primus_turbo::pytorch {

/* Quantize */

at::Tensor fp8_quantize(const at::Tensor input, const at::Tensor scale,
                        const at::ScalarType dest_dtype);

at::Tensor fp8_quantize_meta(const at::Tensor input, const at::Tensor scale,
                             const at::ScalarType dest_dtype);

at::Tensor fp8_dequantize(const at::Tensor input, const at::Tensor scale_inv,
                          const at::ScalarType dest_dtype);

at::Tensor fp8_dequantize_meta(const at::Tensor input, const at::Tensor scale_inv,
                               const at::ScalarType dest_dtype);

/* GEMM */

at::Tensor hipblaslt_gemm(at::Tensor A, at::Tensor scaleA_inv, at::Tensor B, at::Tensor scaleB_inv,
                          const at::ScalarType out_dtype, bool transA, bool transB, bool transC);

at::Tensor hipblaslt_gemm_meta(at::Tensor A, at::Tensor scaleA_inv, at::Tensor B,
                               at::Tensor scaleB_inv, const at::ScalarType out_dtype, bool transA,
                               bool transB, bool transC);

torch::Tensor gemm_fp8_blockwise(torch::Tensor &a, torch::Tensor &a_scales, torch::Tensor &b,
                                 torch::Tensor &b_scales, torch::Tensor &c, const bool transA,
                                 const bool transB, const int64_t block_size);

torch::Tensor gemm_fp8_blockwise_meta(torch::Tensor &a, torch::Tensor &a_scales, torch::Tensor &b,
                                      torch::Tensor &b_scales, torch::Tensor &c, const bool transA,
                                      const bool transB, const int64_t block_size);

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

} // namespace primus_turbo::pytorch
