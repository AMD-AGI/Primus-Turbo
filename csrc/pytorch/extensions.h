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

namespace primus_turbo::pytorch {

at::Tensor fp8_quantize(const at::Tensor input, const at::Tensor scale,
                        const at::ScalarType dest_dtype);

at::Tensor fp8_quantize_meta(const at::Tensor input, const at::Tensor scale,
                             const at::ScalarType dest_dtype);

at::Tensor fp8_dequantize(const at::Tensor input, const at::Tensor scale_inv,
                          const at::ScalarType dest_dtype);

at::Tensor fp8_dequantize_meta(const at::Tensor input, const at::Tensor scale_inv,
                               const at::ScalarType dest_dtype);

torch::Tensor gemm(torch::Tensor a, torch::Tensor b);

torch::Tensor gemm_meta(torch::Tensor a, torch::Tensor b);

torch::Tensor gemm_fp8_blockwise(torch::Tensor &a, torch::Tensor &a_scales, torch::Tensor &b,
                                 torch::Tensor &b_scales, torch::Tensor &c, const bool transA,
                                 const bool transB, const int64_t block_size);

torch::Tensor gemm_fp8_blockwise_meta(torch::Tensor &a, torch::Tensor &a_scales, torch::Tensor &b,
                                      torch::Tensor &b_scales, torch::Tensor &c, const bool transA,
                                      const bool transB, const int64_t block_size);

} // namespace primus_turbo::pytorch
