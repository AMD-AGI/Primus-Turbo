// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
//
// See LICENSE for license information.

#include "ck_tile/host/hip_check_error.hpp"

#include "ck_gemm_kernel.h"
#include "ck_tile/ops/gemm_quant.hpp"
#include "primus_turbo/gemm.h"
namespace primus_turbo {

template <typename ADataType, typename BDataType, typename CDataType, typename AccDataType>
ck_tile::QuantGemmKernelArgs
compute_gemm_args(const CKGemmFP8Params<ADataType, BDataType, CDataType, AccDataType> &params,
                  const ck_tile::index_t strideA, const ck_tile::index_t strideB,
                  const ck_tile::index_t strideC, const ck_tile::index_t strideAQ,
                  const ck_tile::index_t strideBQ) {
    ck_tile::QuantGemmKernelArgs args;

    args.a_ptr  = params.a_ptr;
    args.b_ptr  = params.b_ptr;
    args.c_ptr  = params.c_ptr;
    args.aq_ptr = params.aq_ptr;
    args.bq_ptr = params.bq_ptr;

    args.M = params.m;
    args.N = params.n;
    args.K = params.k;

    args.QK_A = 1;
    args.QK_B = 1;

    args.stride_A = strideA;
    args.stride_B = strideB;
    args.stride_C = strideC;

    args.stride_AQ = strideAQ;
    args.stride_BQ = strideBQ;

    // We're not using split-k, so k_batch is 1
    args.k_batch = 1;

    return args;
}

template <typename ADataType, typename BDataType, typename CDataType, typename AccDataType,
          ck_tile::QuantType QuantMode>
void ck_gemm_fp8(const CKGemmFP8Params<ADataType, BDataType, CDataType, AccDataType> &params) {
    const ck_tile::index_t k_batch = 1;
    const bool             splitk  = k_batch > 1;

    const ck_tile::index_t strideA  = params.transA ? params.m : params.k;
    const ck_tile::index_t strideB  = params.transB ? params.k : params.n;
    const ck_tile::index_t strideC  = params.n;
    const ck_tile::index_t strideAQ = 1;
    const ck_tile::index_t strideBQ = 1;

    const auto                             stream_cfg = ck_tile::stream_config{params.stream};
    std::unique_ptr<CKGemmRunnerInterFace> runner;
    using CLayout = RowMajor;
    if (!params.transA && !params.transB) { // NN
        using ALayout = RowMajor;
        using BLayout = RowMajor;
        runner        = get_ck_gemm_instance<ADataType, BDataType, CDataType, AccDataType, ALayout,
                                             BLayout, CLayout, QuantMode>(params.m, params.n, params.k);
    } else if (!params.transA && params.transB) { // NT
        using ALayout = RowMajor;
        using BLayout = ColMajor;
        runner        = get_ck_gemm_instance<ADataType, BDataType, CDataType, AccDataType, ALayout,
                                             BLayout, CLayout, QuantMode>(params.m, params.n, params.k);
    } else if (params.transA && !params.transB) { // TN
        using ALayout = ColMajor;
        using BLayout = RowMajor;
        runner        = get_ck_gemm_instance<ADataType, BDataType, CDataType, AccDataType, ALayout,
                                             BLayout, CLayout, QuantMode>(params.m, params.n, params.k);
    } else {
        PRIMUS_TURBO_CHECK(false, "CKGroupedGemm only support NN and NT");
    }
    auto args = compute_gemm_args(params, strideA, strideB, strideC, strideAQ, strideBQ);
    runner->run(stream_cfg, args, params.num_cu);
}

// fp8 * fp8 -> fp16 - RowColQuant
template void ck_gemm_fp8<ck_tile::fp8_t, ck_tile::fp8_t, ck_tile::half_t, float,
                          ck_tile::QuantType::RowColQuant>(
    const CKGemmFP8Params<ck_tile::fp8_t, ck_tile::fp8_t, ck_tile::half_t, float> &params);
template void ck_gemm_fp8<ck_tile::bf8_t, ck_tile::bf8_t, ck_tile::half_t, float,
                          ck_tile::QuantType::RowColQuant>(
    const CKGemmFP8Params<ck_tile::bf8_t, ck_tile::bf8_t, ck_tile::half_t, float> &params);
template void ck_gemm_fp8<ck_tile::fp8_t, ck_tile::fp8_t, ck_tile::bfloat16_t, float,
                          ck_tile::QuantType::RowColQuant>(
    const CKGemmFP8Params<ck_tile::fp8_t, ck_tile::fp8_t, ck_tile::bfloat16_t, float> &params);
template void ck_gemm_fp8<ck_tile::bf8_t, ck_tile::bf8_t, ck_tile::bfloat16_t, float,
                          ck_tile::QuantType::RowColQuant>(
    const CKGemmFP8Params<ck_tile::bf8_t, ck_tile::bf8_t, ck_tile::bfloat16_t, float> &params);

// fp8 * fp8 -> fp16 - TensorQuant
template void ck_gemm_fp8<ck_tile::fp8_t, ck_tile::fp8_t, ck_tile::half_t, float,
                          ck_tile::QuantType::TensorQuant>(
    const CKGemmFP8Params<ck_tile::fp8_t, ck_tile::fp8_t, ck_tile::half_t, float> &params);
template void ck_gemm_fp8<ck_tile::bf8_t, ck_tile::bf8_t, ck_tile::half_t, float,
                          ck_tile::QuantType::TensorQuant>(
    const CKGemmFP8Params<ck_tile::bf8_t, ck_tile::bf8_t, ck_tile::half_t, float> &params);
template void ck_gemm_fp8<ck_tile::fp8_t, ck_tile::fp8_t, ck_tile::bfloat16_t, float,
                          ck_tile::QuantType::TensorQuant>(
    const CKGemmFP8Params<ck_tile::fp8_t, ck_tile::fp8_t, ck_tile::bfloat16_t, float> &params);
template void ck_gemm_fp8<ck_tile::bf8_t, ck_tile::bf8_t, ck_tile::bfloat16_t, float,
                          ck_tile::QuantType::TensorQuant>(
    const CKGemmFP8Params<ck_tile::bf8_t, ck_tile::bf8_t, ck_tile::bfloat16_t, float> &params);
} // namespace primus_turbo
