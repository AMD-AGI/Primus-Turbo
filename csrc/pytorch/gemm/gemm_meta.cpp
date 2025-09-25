// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
//
// See LICENSE for license information.

#include <torch/extension.h>

#include "../extensions.h"

namespace primus_turbo::pytorch {

at::Tensor hipblaslt_gemm_meta(at::Tensor A, at::Tensor scaleA_inv, at::Tensor B,
                               at::Tensor scaleB_inv, const at::ScalarType out_dtype, bool transA,
                               bool transB, bool transC) {
    const int64_t m = transA ? A.size(1) : A.size(0);
    const int64_t n = transB ? B.size(0) : B.size(1);

    std::vector<int64_t> c_shape = transC ? std::vector<int64_t>{n, m} : std::vector<int64_t>{m, n};
    at::TensorOptions    options = at::TensorOptions().dtype(out_dtype).device(at::kMeta);
    return at::empty(c_shape, options);
}

at::Tensor gemm_fp8_meta(at::Tensor &a, at::Tensor &b, at::Tensor &a_scales, at::Tensor &b_scales,
                         const bool transA, const bool transB, at::ScalarType out_dtype,
                         const std::string &granularity, c10::optional<int64_t> num_cu) {
    const int64_t m      = transA ? a.size(1) : a.size(0);
    const int64_t n      = transB ? B.size(0) : B.size(1);
    at::Tensor c      = at::empty({m, n}, at::dtype(out_dtype).device(at::kMeta);
    return c;
}

} // namespace primus_turbo::pytorch
