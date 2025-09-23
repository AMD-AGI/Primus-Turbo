// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
//
// See LICENSE for license information.

#include "../extensions.h"
#include "../type_traits.h"
#include "primus_turbo/arch.h"
#include "primus_turbo/grouped_gemm.h"

namespace primus_turbo::pytorch {

at::Tensor grouped_gemm_fp8(at::Tensor &a, at::Tensor &b, at::Tensor &a_scales,
                            at::Tensor &b_scales, const bool transA, const bool transB,
                            at::ScalarType out_dtype, const std::string &granularity,
                            c10::optional<int64_t> num_cu) {

    // Check
    PRIMUS_TURBO_CHECK(is_8bit_floating_point_dtype(a.scalar_type()));
    PRIMUS_TURBO_CHECK(is_8bit_floating_point_dtype(b.scalar_type()));
    PRIMUS_TURBO_CHECK(a.scalar_type() == b.scalar_type(), "a and b dtype mismatch");
    PRIMUS_TURBO_CHECK(out_dtype == at::kBFloat16 || out_dtype == at::kHalf,
                       "out_dtype must be kBFloat16 or kHalf");

    // Determine output tensor size based on transA and transB
    const int64_t bs = b.size(0);
    const int64_t m  = transA ? a.size(1) : a.size(0);
    const int64_t n  = transB ? b.size(1) : b.size(2);
    const int64_t k  = transA ? a.size(0) : a.size(1);

    // Process Scale
    at::Tensor aq_tensor;
    at::Tensor bq_tensor;
    if (granularity == "TENSORWISE") {
        aq_tensor = a_scales.reshape({1, 1}).expand({m, 1});
        bq_tensor = b_scales.reshape({1, 1, 1}).expand({bs, 1, n});
    } else {
        aq_tensor = a_scales.clone();
        bq_tensor = b_scales.clone();
    }
    aq_tensor = aq_tensor.contiguous();
    bq_tensor = bq_tensor.contiguous();

    at::Tensor c      = at::empty({m, n}, at::dtype(out_dtype).device(at::kCUDA));
    auto       stream = at::cuda::getCurrentCUDAStream();

    if (a.dtype() == at::kFloat8_e4m3fnuz || a.dtype() == at::kFloat8_e4m3fn) {
        using AType = typename TorchToCKTileType<at::kFloat8_e4m3fnuz>::type;
        using BType = AType;

        if (out_dtype == at::kBFloat16) {
            using CType = typename TorchToCKTileType<at::kBFloat16>::type;
            auto params = make_ck_groued_gemm_fp8_params<AType, BType, CType, float>(
                args_tensor.data_ptr(), a, b, c, aq_tensor, bq_tensor, group_lens, group_offs,
                transA, transB, bs, m, n, k, stream, get_grouped_gemm_num_cu(num_cu));
            ck_grouped_gemm_fp8<AType, BType, CType, float>(params);
        } else if (out_dtype == at::kHalf) {
            using CType = typename TorchToCKTileType<at::kHalf>::type;
            auto params = make_ck_groued_gemm_fp8_params<AType, BType, CType, float>(
                args_tensor.data_ptr(), a, b, c, aq_tensor, bq_tensor, group_lens, group_offs,
                transA, transB, bs, m, n, k, stream, get_grouped_gemm_num_cu(num_cu));
            ck_grouped_gemm_fp8<AType, BType, CType, float>(params);
        } else {
            PRIMUS_TURBO_CHECK(false, "Unsupported out_dtype for fp8 e4m3");
        }
    } else if (a.dtype() == at::kFloat8_e5m2fnuz || a.dtype() == at::kFloat8_e5m2) {
        using AType = typename TorchToCKTileType<at::kFloat8_e5m2fnuz>::type;
        using BType = AType;

        if (out_dtype == at::kBFloat16) {
            using CType = typename TorchToCKTileType<at::kBFloat16>::type;
            auto params = make_ck_groued_gemm_fp8_params<AType, BType, CType, float>(
                args_tensor.data_ptr(), a, b, c, aq_tensor, bq_tensor, group_lens, group_offs,
                transA, transB, bs, m, n, k, stream, get_grouped_gemm_num_cu(num_cu));
            ck_grouped_gemm_fp8<AType, BType, CType, float>(params);
        } else if (out_dtype == at::kHalf) {
            using CType = typename TorchToCKTileType<at::kHalf>::type;
            auto params = make_ck_groued_gemm_fp8_params<AType, BType, CType, float>(
                args_tensor.data_ptr(), a, b, c, aq_tensor, bq_tensor, group_lens, group_offs,
                transA, transB, bs, m, n, k, stream, get_grouped_gemm_num_cu(num_cu));
            ck_grouped_gemm_fp8<AType, BType, CType, float>(params);
        } else {
            PRIMUS_TURBO_CHECK(false, "Unsupported out_dtype for fp8 e5m2");
        }
    } else {
        PRIMUS_TURBO_CHECK(false, "GroupedGemmFp8 only support fp8/bf8");
    }

    return c;
}

} // namespace primus_turbo::pytorch
