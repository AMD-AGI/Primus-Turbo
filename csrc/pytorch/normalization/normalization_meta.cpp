// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
//
// See LICENSE for license information.

#include <torch/extension.h>

namespace primus_turbo::pytorch {

std::vector<at::Tensor> rmsnorm_fwd_meta(const at::Tensor &input, const at::Tensor &gamma,
                                         const double eps) {
    const int64_t inner_len = gamma.numel();
    const int64_t outer_len = input.numel() / inner_len;
    auto          output    = at::empty_like(input, at::device(at::kMeta));
    auto          rs = at::empty({outer_len}, input.options().device(at::kMeta).dtype(at::kFloat));
    return {output, rs};
}

std::vector<at::Tensor> rmsnorm_bwd_meta(const at::Tensor &input, const at::Tensor &gamma,
                                         const at::Tensor &grad_output, const at::Tensor &rs,
                                         const double eps) {
    auto grad_input = at::empty_like(input, at::device(at::kMeta));
    auto grad_gamma = at::empty_like(gamma, at::device(at::kMeta));
    return {grad_input, grad_gamma};
}

} // namespace primus_turbo::pytorch
