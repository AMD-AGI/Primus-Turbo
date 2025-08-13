// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
//
// See LICENSE for license information.

#include <torch/extension.h>

namespace primus_turbo::pytorch {

at::Tensor rmsnorm_fwd_meta(const at::Tensor &input, const at::Tensor &gamma, const double eps) {
    return at::empty_like(input, at::device(at::kMeta));
}

std::vector<at::Tensor> rmsnorm_bwd_meta(const at::Tensor &input, const at::Tensor &gamma,
                                         const at::Tensor &grad_output, const double eps) {

    at::Tensor grad_input = at::empty_like(input, at::device(at::kMeta));
    at::Tensor grad_gamma = at::empty_like(input, at::device(at::kMeta));

    return {grad_input, grad_gamma};
}

} // namespace primus_turbo::pytorch
