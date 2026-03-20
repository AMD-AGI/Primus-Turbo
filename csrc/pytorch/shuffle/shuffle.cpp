// Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// See LICENSE for license information.

#include "primus_turbo/shuffle.h"
#include "primus_turbo/quantization.h"
#include "pytorch/extensions.h"
#include "pytorch/utils.h"

namespace primus_turbo::pytorch {

using namespace primus_turbo::dtype;

at::Tensor shuffle_scale_impl(const at::Tensor scale, at::IntArrayRef layout) {
    using namespace primus_turbo::detail;

    std::function<int64_t(int64_t, int64_t)> cdiv = [](int64_t a, int64_t b) -> int64_t {
        return (a + b - 1) / b;
    };

    PRIMUS_TURBO_CHECK(scale.is_cuda(), "Scale must be a CUDA tensor");
    PRIMUS_TURBO_CHECK(scale.scalar_type() == at::kFloat8_e8m0fnu, "Scale must be Float8_e8m0fnu.");
    PRIMUS_TURBO_CHECK(scale.dim() == 2, "Scale must be 2D");

    PRIMUS_TURBO_CHECK(layout.size() == 2, "layout must have exactly 2 elements");

    const int tile_m = layout[0];
    const int tile_n = layout[1];

    int64_t M           = scale.size(0);
    int64_t scale_M_pad = cdiv(M, 256) * 256;
    int64_t scale_N     = scale.size(1);
    int64_t scale_N_pad = cdiv(scale_N, 8) * 8;

    auto device = scale.device();
    auto stream = at::cuda::getCurrentCUDAStream();

    at::Tensor shuffled_scale =
        at::empty({scale_M_pad, scale_N_pad}, at::TensorOptions().dtype(at::kByte).device(device));

    shuffle_e8m0_scale(reinterpret_cast<uint8_t *>(scale.data_ptr()),
                       reinterpret_cast<uint8_t *>(shuffled_scale.data_ptr()), tile_m, tile_n, M,
                       scale_N, scale_M_pad, scale_N_pad, stream);

    return shuffled_scale.view(at::kFloat8_e8m0fnu);
}

at::Tensor shuffle_weight_impl(const at::Tensor weight, at::IntArrayRef layout) {
    using namespace primus_turbo::detail;

    PRIMUS_TURBO_CHECK(weight.is_cuda(), "Weight must be a CUDA tensor");
    PRIMUS_TURBO_CHECK(weight.scalar_type() == at::kFloat4_e2m1fn_x2,
                       "Weight must be Float4_e2m1fn_x2.");
    PRIMUS_TURBO_CHECK(weight.dim() == 2, "Weight must be 2D");
    PRIMUS_TURBO_CHECK(weight.is_contiguous(), "Weight must be contiguous");

    int64_t M = weight.size(0);
    int64_t N = weight.size(1);

    PRIMUS_TURBO_CHECK(M % MXFP4_SHUFFLE_BN == 0, "M must be divisible by ", MXFP4_SHUFFLE_BN,
                       " for shuffled FP4. But got M=", M);
    PRIMUS_TURBO_CHECK(N % MXFP4_SHUFFLE_BK == 0, "N must be divisible by ", MXFP4_SHUFFLE_BK,
                       " for shuffled FP4. But got N=", N);
    PRIMUS_TURBO_CHECK(layout.size() == 2, "layout must have exactly 2 elements");

    const int tile_m = layout[0];
    const int tile_n = layout[1];

    auto device = weight.device();
    auto stream = at::cuda::getCurrentCUDAStream();

    // packed 2 fp4 values in N dimension
    at::Tensor shuffled_weight =
        at::empty({M, N}, at::TensorOptions().dtype(at::kByte).device(device));

    TORCH_TYPE_SWITCH_FP4(weight.scalar_type(), DType, {
        shuffle_weight<DType>(reinterpret_cast<DType *>(weight.data_ptr()),
                              reinterpret_cast<DType *>(shuffled_weight.data_ptr()), tile_m, tile_n,
                              M, N, stream);
    });

    return shuffled_weight.view(at::kFloat4_e2m1fn_x2);
}

} // namespace primus_turbo::pytorch
