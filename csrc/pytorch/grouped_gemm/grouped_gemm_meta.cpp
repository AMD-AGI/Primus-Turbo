// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
//
// See LICENSE for license information.

#include <torch/extension.h>

#include "../extensions.h"

namespace primus_turbo::pytorch {

at::Tensor grouped_gemm_meta(at::Tensor &a, at::Tensor &b, at::Tensor &group_lens,
                             at::Tensor &group_offs, const bool transA, const bool transB) {
    // TODO: out-datatype
    const int64_t m = transA ? a.size(1) : a.size(0);
    const int64_t n = transB ? b.size(1) : b.size(2);
    return at::empty(std::vector<int64_t>{m, n}, a.options().device(at::kMeta));
}

at::Tensor grouped_gemm_variable_k_meta(at::Tensor &a, at::Tensor &b, at::Tensor &group_lens,
                                        at::Tensor &group_offs, const bool transA,
                                        const bool transB) {
    const int64_t bs = group_lens.numel();
    const int64_t m  = transA ? a.size(1) : a.size(0);
    const int64_t n  = transB ? b.size(0) : b.size(1);
    return at::empty({bs, m, n}, a.options().device(at::kMeta));
}

} // namespace primus_turbo::pytorch
