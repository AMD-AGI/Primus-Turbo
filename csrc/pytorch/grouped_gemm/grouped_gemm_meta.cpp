// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
//
// See LICENSE for license information.

#include <torch/extension.h>

#include "pytorch/extensions.h"

namespace primus_turbo::pytorch {

//==================================================================
//  CK Grouped GEMM Meta
//==================================================================

at::Tensor ck_grouped_gemm_meta(at::Tensor &a, at::Tensor &b, at::Tensor &group_lens,
                                at::Tensor &group_offs, const bool transA, const bool transB,
                                c10::optional<int64_t> num_cu) {
    // TODO: out-datatype
    const int64_t m = transA ? a.size(1) : a.size(0);
    const int64_t n = transB ? b.size(1) : b.size(2);
    return at::empty(std::vector<int64_t>{m, n}, a.options().device(at::kMeta));
}

at::Tensor ck_grouped_gemm_variable_k_meta(at::Tensor &a, at::Tensor &b, at::Tensor &group_lens,
                                           at::Tensor &group_offs, const bool transA,
                                           const bool transB, c10::optional<int64_t> num_cu) {
    const int64_t bs = group_lens.numel();
    const int64_t m  = transA ? a.size(1) : a.size(0);
    const int64_t n  = transB ? b.size(0) : b.size(1);
    return at::empty({bs, m, n}, a.options().device(at::kMeta));
}

at::Tensor ck_grouped_gemm_fp8_meta(at::Tensor &a, at::Tensor &b, at::Tensor &a_scales,
                                    at::Tensor &b_scales, at::Tensor &group_lens,
                                    at::Tensor &group_offs, const bool transA, const bool transB,
                                    at::ScalarType out_dtype, const std::string &granularity,
                                    c10::optional<int64_t> num_cu) {
    const int64_t m      = transA ? a.size(1) : a.size(0);
    const int64_t n      = transB ? b.size(1) : b.size(2);
    at::Tensor    output = at::empty({m, n}, at::dtype(out_dtype).device(at::kMeta));
    return output;
}

at::Tensor ck_grouped_gemm_fp8_variable_k_meta(at::Tensor &a, at::Tensor &b, at::Tensor &a_scales,
                                               at::Tensor &b_scales, at::Tensor &group_lens,
                                               at::Tensor &group_offs, const bool transA,
                                               const bool transB, at::ScalarType out_dtype,
                                               const std::string     &granularity,
                                               c10::optional<int64_t> num_cu) {
    const int64_t bs     = group_lens.numel();
    const int64_t m      = transA ? a.size(1) : a.size(0);
    const int64_t n      = transB ? b.size(0) : b.size(1);
    at::Tensor    output = at::empty({bs, m, n}, at::dtype(out_dtype).device(at::kMeta));
    return output;
}

//==================================================================
//  hipBLASLt Grouped GEMM Meta
//==================================================================
at::Tensor hipblaslt_grouped_gemm_meta(at::Tensor &a, at::Tensor &b, at::Tensor &group_lens,
                                       at::Tensor &group_offs, const bool transA,
                                       const bool transB) {
    if (transA) {
        const int64_t bs = group_lens.numel();
        const int64_t m  = transA ? a.size(1) : a.size(0);
        const int64_t n  = transB ? b.size(0) : b.size(1);
        return at::empty({bs, m, n}, a.options().device(at::kMeta));
    } else {
        const int64_t m = transA ? a.size(1) : a.size(0);
        const int64_t n = transB ? b.size(1) : b.size(2);
        return at::empty(std::vector<int64_t>{m, n}, a.options().device(at::kMeta));
    }
}

at::Tensor hipblaslt_grouped_gemm_fp8_meta(at::Tensor &a, at::Tensor &b, at::Tensor &a_scales,
                                           at::Tensor &b_scales, at::Tensor &group_lens,
                                           at::Tensor &group_offs, const bool transA,
                                           const bool transB, at::ScalarType out_dtype,
                                           const std::string &granularity) {
    if (transA) {
        const int64_t bs = group_lens.numel();
        const int64_t m  = a.size(1);
        const int64_t n  = transB ? b.size(0) : b.size(1);
        return at::empty({bs, m, n}, at::dtype(out_dtype).device(at::kMeta));
    } else {
        const int64_t m = a.size(0);
        const int64_t n = transB ? b.size(1) : b.size(2);
        return at::empty({m, n}, at::dtype(out_dtype).device(at::kMeta));
    }
}

at::Tensor grouped_gemm_compute_offs_meta(at::Tensor &group_lens) {
    at::Tensor group_offs = at::empty({group_lens.numel() + 1}, group_lens.options());
    return group_offs;
}

} // namespace primus_turbo::pytorch
