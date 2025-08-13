// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
//
// See LICENSE for license information.

#include "primus_turbo/normalization.h"

#include "../extensions.h"

namespace primus_turbo::pytorch {

using namespace primus_turbo::dtype;

at::Tensor rmsnorm_fwd(const at::Tensor &input, const at::Tensor &gamma, const double eps) {
    TORCH_CHECK(input.is_contiguous(), "rmsnorm_fwd: input must be contiguous.");
    TORCH_CHECK(gamma.is_contiguous(), "rmsnorm_fwd: gamma must be contiguous.");

    const int64_t inner_len = gamma.numel();
    const int64_t outer_len = input.numel() / inner_len;
    auto          output    = at::empty_like(input);

    TORCH_CHECK(input.numel() % inner_len == 0, "input.numel() must be divisible by gamma.numel()");

    auto stream = at::cuda::getCurrentCUDAStream();
    if (input.scalar_type() == at::kFloat) {
        rmsnorm_fwd_impl<float>(input.data_ptr<float>(), gamma.data_ptr<float>(),
                                output.data_ptr<float>(), inner_len, outer_len,
                                static_cast<float>(eps), stream);
    } else if (input.scalar_type() == at::kHalf) {
        rmsnorm_fwd_impl<float16>(reinterpret_cast<float16 *>(input.data_ptr()),
                                  reinterpret_cast<float16 *>(gamma.data_ptr()),
                                  reinterpret_cast<float16 *>(output.data_ptr()), inner_len,
                                  outer_len, static_cast<float>(eps), stream);
    } else if (input.scalar_type() == at::kBFloat16) {
        rmsnorm_fwd_impl<bfloat16>(reinterpret_cast<bfloat16 *>(input.data_ptr()),
                                   reinterpret_cast<bfloat16 *>(gamma.data_ptr()),
                                   reinterpret_cast<bfloat16 *>(output.data_ptr()), inner_len,
                                   outer_len, static_cast<float>(eps), stream);
    } else {
        PRIMUS_TURBO_ERROR("RMSNorm only support : [float32, float16, bfloat16]");
    }
    return output;
}

std::vector<at::Tensor> rmsnorm_bwd(const at::Tensor &input, const at::Tensor &gamma,
                                    const at::Tensor &grad_output, const double eps) {
    TORCH_CHECK(input.is_contiguous(), "rmsnorm_bwd: input must be contiguous.");
    TORCH_CHECK(gamma.is_contiguous(), "rmsnorm_bwd: gamma must be contiguous.");
    TORCH_CHECK(grad_output.is_contiguous(), "rmsnorm_bwd: grad_output must be contiguous.");

    const int64_t inner_len = gamma.numel();
    const int64_t outer_len = input.numel() / inner_len;

    TORCH_CHECK(input.numel() % inner_len == 0, "input.numel() must be divisible by gamma.numel()");

    auto input_grad = at::empty_like(input);
    auto gamma_grad = at::empty_like(input);

    auto stream = at::cuda::getCurrentCUDAStream();
    if (input.scalar_type() == at::kFloat) {
        rmsnorm_bwd_impl<float>(input.data_ptr<float>(), gamma.data_ptr<float>(),
                                grad_output.data_ptr<float>(), input_grad.data_ptr<float>(),
                                gamma_grad.data_ptr<float>(), inner_len, outer_len,
                                static_cast<float>(eps), stream);
    } else if (input.scalar_type() == at::kHalf) {
        rmsnorm_bwd_impl<float16>(reinterpret_cast<float16 *>(input.data_ptr()),
                                  reinterpret_cast<float16 *>(gamma.data_ptr()),
                                  reinterpret_cast<float16 *>(grad_output.data_ptr()),
                                  reinterpret_cast<float16 *>(input_grad.data_ptr()),
                                  reinterpret_cast<float16 *>(gamma_grad.data_ptr()), inner_len,
                                  outer_len, static_cast<float>(eps), stream);
    } else if (input.scalar_type() == at::kBFloat16) {
        rmsnorm_bwd_impl<bfloat16>(reinterpret_cast<bfloat16 *>(input.data_ptr()),
                                   reinterpret_cast<bfloat16 *>(gamma.data_ptr()),
                                   reinterpret_cast<bfloat16 *>(grad_output.data_ptr()),
                                   reinterpret_cast<bfloat16 *>(input_grad.data_ptr()),
                                   reinterpret_cast<bfloat16 *>(gamma_grad.data_ptr()), inner_len,
                                   outer_len, static_cast<float>(eps), stream);
    } else {
        PRIMUS_TURBO_ERROR("RMSNorm only support : [float32, float16, bfloat16]");
    }

    return {input_grad, gamma_grad};
}

} // namespace primus_turbo::pytorch
