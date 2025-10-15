// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
//
// See LICENSE for license information.

#include "../extensions.h"
#include "../type_traits.h"
#include "primus_turbo/arch.h"
#include "primus_turbo/gemm.h"

namespace primus_turbo::pytorch {

template <typename AType, typename BType, typename CType, typename ACCType>
inline CKGemmFP8Params<AType, BType, CType, ACCType>
make_ck_gemm_fp8_params(const at::Tensor &a, const at::Tensor &b, at::Tensor &c,
                        const at::Tensor &a_scales, const at::Tensor &b_scales, bool transA,
                        bool transB, int32_t m, int32_t n, int32_t k, hipStream_t stream,
                        uint32_t num_cu) {
    CKGemmFP8Params<AType, BType, CType, ACCType> params;
    params.a_ptr  = reinterpret_cast<const AType *>(a.data_ptr());
    params.b_ptr  = reinterpret_cast<const BType *>(b.data_ptr());
    params.c_ptr  = reinterpret_cast<CType *>(c.data_ptr());
    params.aq_ptr = reinterpret_cast<const ACCType *>(a_scales.data_ptr());
    params.bq_ptr = reinterpret_cast<const ACCType *>(b_scales.data_ptr());
    params.transA = transA;
    params.transB = transB;
    params.m      = m;
    params.n      = n;
    params.k      = k;
    params.stream = stream;
    params.num_cu = num_cu;
    return params;
}

uint32_t get_gemm_num_cu(c10::optional<int64_t> num_cu) {
    auto    stream     = at::cuda::getCurrentCUDAStream();
    int32_t cus        = get_multi_processor_count(stream.device_index());
    int32_t num_cu_val = num_cu.has_value() ? num_cu.value() : -1;
    return num_cu_val <= 0 ? uint32_t(cus) : uint32_t(std::min(num_cu_val, cus));
}

at::Tensor gemm_fp8(at::Tensor &a, at::Tensor &b, at::Tensor &a_scales, at::Tensor &b_scales,
                    const bool transA, const bool transB, at::ScalarType out_dtype,
                    const std::string &granularity, c10::optional<int64_t> num_cu) {

    // Check
    PRIMUS_TURBO_CHECK(is_8bit_floating_point_dtype(a.scalar_type()));
    PRIMUS_TURBO_CHECK(is_8bit_floating_point_dtype(b.scalar_type()));
    PRIMUS_TURBO_CHECK(a.scalar_type() == b.scalar_type(), "a and b dtype mismatch");
    PRIMUS_TURBO_CHECK(out_dtype == at::kBFloat16 || out_dtype == at::kHalf,
                       "out_dtype must be kBFloat16 or kHalf");
    PRIMUS_TURBO_CHECK(granularity == "TENSORWISE" || granularity == "ROWWISE",
                       "granularity must be 'TENSORWISE' or 'ROWWISE'");

    // Determine output tensor size based on transA and transB
    const int64_t m = transA ? a.size(1) : a.size(0);
    const int64_t k = transA ? a.size(0) : a.size(1);
    const int64_t n = transB ? b.size(0) : b.size(1);
    PRIMUS_TURBO_CHECK(k % 128 == 0, "Inner dimension K must be multiple of 128");

    at::Tensor aq_tensor = a_scales.contiguous();
    at::Tensor bq_tensor = b_scales.contiguous();

    at::Tensor c      = at::empty({m, n}, at::dtype(out_dtype).device(at::kCUDA));
    auto       stream = at::cuda::getCurrentCUDAStream();

    if (a.dtype() == at::kFloat8_e4m3fnuz || a.dtype() == at::kFloat8_e4m3fn) {
        using AType = typename TorchToCKTileType<at::kFloat8_e4m3fnuz>::type;
        using BType = AType;

        if (out_dtype == at::kBFloat16) {
            using CType = typename TorchToCKTileType<at::kBFloat16>::type;
            auto params = CKGemmFP8Params<AType, BType, CType, float>(
                static_cast<const AType *>(a.data_ptr()), static_cast<const BType *>(b.data_ptr()),
                static_cast<CType *>(c.data_ptr()),
                static_cast<const float *>(aq_tensor.data_ptr()),
                static_cast<const float *>(bq_tensor.data_ptr()), transA, transB, m, n, k, stream,
                get_gemm_num_cu(num_cu));
            if (granularity == "TENSORWISE")
                ck_gemm_fp8<AType, BType, CType, float, ck_tile::QuantType::TensorQuant>(params);
            else
                ck_gemm_fp8<AType, BType, CType, float, ck_tile::QuantType::RowColQuant>(params);
        } else if (out_dtype == at::kHalf) {
            using CType = typename TorchToCKTileType<at::kHalf>::type;
            auto params = CKGemmFP8Params<AType, BType, CType, float>(
                static_cast<const AType *>(a.data_ptr()), static_cast<const BType *>(b.data_ptr()),
                static_cast<CType *>(c.data_ptr()),
                static_cast<const float *>(aq_tensor.data_ptr()),
                static_cast<const float *>(bq_tensor.data_ptr()), transA, transB, m, n, k, stream,
                get_gemm_num_cu(num_cu));
            if (granularity == "TENSORWISE")
                ck_gemm_fp8<AType, BType, CType, float, ck_tile::QuantType::TensorQuant>(params);
            else
                ck_gemm_fp8<AType, BType, CType, float, ck_tile::QuantType::RowColQuant>(params);
        } else {
            PRIMUS_TURBO_CHECK(false, "Unsupported out_dtype for fp8 e4m3");
        }
    } else if (a.dtype() == at::kFloat8_e5m2fnuz || a.dtype() == at::kFloat8_e5m2) {
        using AType = typename TorchToCKTileType<at::kFloat8_e5m2fnuz>::type;
        using BType = AType;

        if (out_dtype == at::kBFloat16) {
            using CType = typename TorchToCKTileType<at::kBFloat16>::type;
            auto params = CKGemmFP8Params<AType, BType, CType, float>(
                static_cast<const AType *>(a.data_ptr()), static_cast<const BType *>(b.data_ptr()),
                static_cast<CType *>(c.data_ptr()),
                static_cast<const float *>(aq_tensor.data_ptr()),
                static_cast<const float *>(bq_tensor.data_ptr()), transA, transB, m, n, k, stream,
                get_gemm_num_cu(num_cu));
            if (granularity == "TENSORWISE")
                ck_gemm_fp8<AType, BType, CType, float, ck_tile::QuantType::TensorQuant>(params);
            else
                ck_gemm_fp8<AType, BType, CType, float, ck_tile::QuantType::RowColQuant>(params);
        } else if (out_dtype == at::kHalf) {
            using CType = typename TorchToCKTileType<at::kHalf>::type;
            auto params = CKGemmFP8Params<AType, BType, CType, float>(
                static_cast<const AType *>(a.data_ptr()), static_cast<const BType *>(b.data_ptr()),
                static_cast<CType *>(c.data_ptr()),
                static_cast<const float *>(aq_tensor.data_ptr()),
                static_cast<const float *>(bq_tensor.data_ptr()), transA, transB, m, n, k, stream,
                get_gemm_num_cu(num_cu));
            if (granularity == "TENSORWISE")
                ck_gemm_fp8<AType, BType, CType, float, ck_tile::QuantType::TensorQuant>(params);
            else
                ck_gemm_fp8<AType, BType, CType, float, ck_tile::QuantType::RowColQuant>(params);
        } else {
            PRIMUS_TURBO_CHECK(false, "Unsupported out_dtype for fp8 e5m2");
        }
    } else {
        PRIMUS_TURBO_CHECK(false, "GroupedGemmFp8 only support fp8/bf8");
    }

    return c;
}

} // namespace primus_turbo::pytorch
