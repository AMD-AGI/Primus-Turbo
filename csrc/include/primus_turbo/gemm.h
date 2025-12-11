// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
//
// See LICENSE for license information.

#pragma once
#include "ck_tile/ops/gemm_quant/pipeline/tile_gemm_quant_traits.hpp"
#include <cstdint>
#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt.h>
#include <stdexcept>
namespace primus_turbo {

//==================================================================
//  HipBLASLt Type Utils
//==================================================================

inline size_t hipblaslt_dtype_bytes(hipDataType dtype) {
    switch (dtype) {
    case HIP_R_64F:
        return 8;
    case HIP_R_32F:
        return 4;
    case HIP_R_16F:
    case HIP_R_16BF:
        return 2;
    case HIP_R_8F_E4M3_FNUZ:
    case HIP_R_8F_E5M2_FNUZ:
    case HIP_R_8F_E4M3:
    case HIP_R_8F_E5M2:
        return 1;
    default:
        throw std::runtime_error("Unsupported hipDataType");
    }
}

//==================================================================
//  HipBLASLt GEMM
//==================================================================

int64_t get_hipblaslt_workspace_size_in_byte();

void hipblaslt_gemm_impl(const void *A, const hipDataType A_type, const int64_t rows_a,
                         const int64_t cols_a, const int64_t lda, const void *scaleA_inv,
                         hipblasOperation_t transA, const void *B, const hipDataType B_type,
                         const int64_t rows_b, const int64_t cols_b, const int64_t ldb,
                         const void *scaleB_inv, hipblasOperation_t transB, void *D,
                         const hipDataType D_type, const int64_t rows_d, const int64_t cols_d,
                         const int64_t ldd, void *workspace, const int64_t workspace_size,
                         const bool use_fp8, const hipblasLtMatmulMatrixScale_t scale_mode,
                         hipblasLtHandle_t handle, hipStream_t stream);

//==================================================================
//  CK GEMM
//==================================================================

template <typename AType, typename BType, typename CType, typename ACCType = float>
struct CKGemmFP8Params {
    const AType   *a_ptr  = nullptr;
    const BType   *b_ptr  = nullptr;
    CType         *c_ptr  = nullptr;
    const ACCType *aq_ptr = nullptr;
    const ACCType *bq_ptr = nullptr;

    bool transA = false;
    bool transB = false;

    int32_t m = 0;
    int32_t n = 0;
    int32_t k = 0;

    hipStream_t stream = nullptr;
};

template <typename ADataType, typename BDataType, typename CDataType, typename AccDataType,
          ck_tile::QuantType QuantMode>
void ck_gemm_fp8_impl(const CKGemmFP8Params<ADataType, BDataType, CDataType, AccDataType> &params);

} // namespace primus_turbo
