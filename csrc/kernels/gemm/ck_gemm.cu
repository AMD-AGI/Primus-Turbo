// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
//
// See LICENSE for license information.

#include "ck_gemm_kernel_instance_factory.h"
#include "ck_gemm_kernel_template.h"
#include "ck_tile/ops/gemm_quant.hpp"
#include "primus_turbo/gemm.h"
namespace primus_turbo {

template <typename ADataType, typename BDataType, typename CDataType, typename AccDataType>
ck_tile::QuantGemmKernelArgs
compute_gemm_args(const CKGemmFP8Params<ADataType, BDataType, CDataType, AccDataType> &params,
                  const ck_tile::index_t strideA, const ck_tile::index_t strideB,
                  const ck_tile::index_t strideC, const ck_tile::index_t strideAQ,
                  const ck_tile::index_t strideBQ, const ck_tile::index_t QK_A,
                  const ck_tile::index_t QK_B) {
    ck_tile::QuantGemmKernelArgs args;

    args.a_ptr  = params.a_ptr;
    args.b_ptr  = params.b_ptr;
    args.c_ptr  = params.c_ptr;
    args.aq_ptr = params.aq_ptr;
    args.bq_ptr = params.bq_ptr;

    args.M = params.m;
    args.N = params.n;
    args.K = params.k;

    args.QK_A = QK_A;
    args.QK_B = QK_B;

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
void ck_gemm_fp8_impl(const CKGemmFP8Params<ADataType, BDataType, CDataType, AccDataType> &params) {
    const ck_tile::index_t k_batch = 1;
    const bool             splitk  = k_batch > 1;

    const ck_tile::index_t strideA = params.transA ? params.m : params.k;
    const ck_tile::index_t strideB = params.transB ? params.k : params.n;
    const ck_tile::index_t strideC = params.n;

    // Calculate proper strides and QK values for quantization scales
    ck_tile::index_t strideAQ, strideBQ, QK_A, QK_B;
    if constexpr (QuantMode == ck_tile::QuantType::ABQuantGrouped) {
        // AQuantGroupSize: <1, 1, 128> => A scale shape: (M, AQK), AQLayout is RowMajor
        // AQK = K / 128 = number of K-dimension scale groups per row
        const ck_tile::index_t AQK = (params.k + 127) / 128;
        QK_A                       = AQK; // QK_A is the K-dimension size of the scale tensor
        strideAQ                   = AQK; // RowMajor stride = number of columns

        // BQuantGroupSize: <1, 128, 128> => B scale shape: (BQK, BQN), BQLayout is ColumnMajor
        // BQK = K / 128, BQN = N / 128
        const ck_tile::index_t BQK = (params.k + 127) / 128;
        QK_B                       = BQK; // QK_B is the K-dimension size of the B scale tensor
        strideBQ                   = BQK; // ColumnMajor stride = number of rows
    } else {
        // For RowColQuant and TensorQuant, QK_A and QK_B should be 1
        QK_A     = 1;
        QK_B     = 1;
        strideAQ = 1;
        strideBQ = 1;
    }

    const auto                             stream_cfg = ck_tile::stream_config{params.stream};
    std::unique_ptr<CKGemmRunnerInterFace> runner;
    using CLayout = RowMajor;
    if (!params.transA && !params.transB) { // NN
        using ALayout = RowMajor;
        using BLayout = RowMajor;
        runner        = get_ck_gemm_instance<ADataType, BDataType, CDataType, AccDataType, ALayout,
                                             BLayout, CLayout, QuantMode>(params.m, params.n, params.k);
    } else if (!params.transA && params.transB) { // NT (RowMajor × ColMajor)
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
        PRIMUS_TURBO_CHECK(false, "CK Gemm only support NN, TN and NT");
    }
    auto args =
        compute_gemm_args(params, strideA, strideB, strideC, strideAQ, strideBQ, QK_A, QK_B);
    runner->run(stream_cfg, args);
}

// fp8 * fp8 -> fp16 - RowColQuant
template void ck_gemm_fp8_impl<ck_tile::fp8_t, ck_tile::fp8_t, ck_tile::half_t, float,
                               ck_tile::QuantType::RowColQuant>(
    const CKGemmFP8Params<ck_tile::fp8_t, ck_tile::fp8_t, ck_tile::half_t, float> &params);
// bf8 * bf8 -> fp16 - RowColQuant
template void ck_gemm_fp8_impl<ck_tile::bf8_t, ck_tile::bf8_t, ck_tile::half_t, float,
                               ck_tile::QuantType::RowColQuant>(
    const CKGemmFP8Params<ck_tile::bf8_t, ck_tile::bf8_t, ck_tile::half_t, float> &params);
// fp8 * fp8 -> bf16 - RowColQuant
template void ck_gemm_fp8_impl<ck_tile::fp8_t, ck_tile::fp8_t, ck_tile::bfloat16_t, float,
                               ck_tile::QuantType::RowColQuant>(
    const CKGemmFP8Params<ck_tile::fp8_t, ck_tile::fp8_t, ck_tile::bfloat16_t, float> &params);
// bf8 * bf8 -> bf16 - RowColQuant
template void ck_gemm_fp8_impl<ck_tile::bf8_t, ck_tile::bf8_t, ck_tile::bfloat16_t, float,
                               ck_tile::QuantType::RowColQuant>(
    const CKGemmFP8Params<ck_tile::bf8_t, ck_tile::bf8_t, ck_tile::bfloat16_t, float> &params);

// fp8 * fp8 -> fp16 - TensorQuant
template void ck_gemm_fp8_impl<ck_tile::fp8_t, ck_tile::fp8_t, ck_tile::half_t, float,
                               ck_tile::QuantType::TensorQuant>(
    const CKGemmFP8Params<ck_tile::fp8_t, ck_tile::fp8_t, ck_tile::half_t, float> &params);
// bf8 * bf8 -> fp16 - TensorQuant
template void ck_gemm_fp8_impl<ck_tile::bf8_t, ck_tile::bf8_t, ck_tile::half_t, float,
                               ck_tile::QuantType::TensorQuant>(
    const CKGemmFP8Params<ck_tile::bf8_t, ck_tile::bf8_t, ck_tile::half_t, float> &params);
// fp8 * fp8 -> bf16 - TensorQuant
template void ck_gemm_fp8_impl<ck_tile::fp8_t, ck_tile::fp8_t, ck_tile::bfloat16_t, float,
                               ck_tile::QuantType::TensorQuant>(
    const CKGemmFP8Params<ck_tile::fp8_t, ck_tile::fp8_t, ck_tile::bfloat16_t, float> &params);
// bf8 * bf8 -> bf16 - TensorQuant
template void ck_gemm_fp8_impl<ck_tile::bf8_t, ck_tile::bf8_t, ck_tile::bfloat16_t, float,
                               ck_tile::QuantType::TensorQuant>(
    const CKGemmFP8Params<ck_tile::bf8_t, ck_tile::bf8_t, ck_tile::bfloat16_t, float> &params);

// fp8 * fp8 -> fp16 - ABQuantGrouped
template void ck_gemm_fp8_impl<ck_tile::fp8_t, ck_tile::fp8_t, ck_tile::half_t, float,
                               ck_tile::QuantType::ABQuantGrouped>(
    const CKGemmFP8Params<ck_tile::fp8_t, ck_tile::fp8_t, ck_tile::half_t, float> &params);
// bf8 * bf8 -> fp16 - ABQuantGrouped
template void ck_gemm_fp8_impl<ck_tile::bf8_t, ck_tile::bf8_t, ck_tile::half_t, float,
                               ck_tile::QuantType::ABQuantGrouped>(
    const CKGemmFP8Params<ck_tile::bf8_t, ck_tile::bf8_t, ck_tile::half_t, float> &params);
// fp8 * fp8 -> bf16 - ABQuantGrouped
template void ck_gemm_fp8_impl<ck_tile::fp8_t, ck_tile::fp8_t, ck_tile::bfloat16_t, float,
                               ck_tile::QuantType::ABQuantGrouped>(
    const CKGemmFP8Params<ck_tile::fp8_t, ck_tile::fp8_t, ck_tile::bfloat16_t, float> &params);
// bf8 * bf8 -> bf16 - ABQuantGrouped
template void ck_gemm_fp8_impl<ck_tile::bf8_t, ck_tile::bf8_t, ck_tile::bfloat16_t, float,
                               ck_tile::QuantType::ABQuantGrouped>(
    const CKGemmFP8Params<ck_tile::bf8_t, ck_tile::bf8_t, ck_tile::bfloat16_t, float> &params);
} // namespace primus_turbo
