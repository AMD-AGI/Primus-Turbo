// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
//
// See LICENSE for license information.

#include "primus_turbo/quantization.h"
#include "primus_turbo/shuffle.h"
#include "pytorch/extensions.h"

namespace primus_turbo::pytorch {

using namespace primus_turbo::detail;

at::Tensor shuffle_scale_impl_meta(const at::Tensor scale, at::IntArrayRef layout) {
    std::function<int64_t(int64_t, int64_t)> cdiv = [](int64_t a, int64_t b) -> int64_t {
        return (a + b - 1) / b;
    };

    PRIMUS_TURBO_CHECK(scale.scalar_type() == at::kFloat8_e8m0fnu, "Scale must be Float8_e8m0fnu.");
    PRIMUS_TURBO_CHECK(scale.dim() == 2, "Scale must be 2D");

    const int64_t M = scale.size(0);
    const int64_t N = scale.size(1);

    int64_t scale_M_pad = cdiv(M, 256) * 256;
    int64_t scale_N_pad = cdiv(N, 8) * 8;

    return at::empty({scale_M_pad, scale_N_pad},
                     at::TensorOptions().dtype(at::kByte).device(at::kMeta))
        .view(at::kFloat8_e8m0fnu);
}

at::Tensor shuffle_weight_impl_meta(const at::Tensor weight, at::IntArrayRef layout) {
    PRIMUS_TURBO_CHECK(weight.scalar_type() == at::kFloat4_e2m1fn_x2,
                       "Weight must be Float4_e2m1fn_x2.");
    PRIMUS_TURBO_CHECK(weight.dim() == 2, "Weight must be 2D");
    PRIMUS_TURBO_CHECK(weight.is_contiguous(), "Weight must be contiguous");

    const int64_t M = weight.size(0);
    const int64_t N = weight.size(1);

    PRIMUS_TURBO_CHECK(M % MXFP4_SHUFFLE_BN == 0, "M must be divisible by ", MXFP4_SHUFFLE_BN,
                       " for shuffled FP4. But got M=", M);
    PRIMUS_TURBO_CHECK((N / 2) % MXFP4_SHUFFLE_BK == 0, "N/2 must be divisible by ",
                       MXFP4_SHUFFLE_BK, " for shuffled FP4. But got N/2=", N / 2);

    return at::empty({M, N}, at::TensorOptions().dtype(at::kByte).device(at::kMeta))
        .view(at::kFloat4_e2m1fn_x2);
}

} // namespace primus_turbo::pytorch
