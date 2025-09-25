// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
//
// See LICENSE for license information.

#pragma once
#include <cstdint>
#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt.h>
#include <stdexcept>

namespace primus_turbo {

// *************** HipBlasLt ***************
int64_t get_hipblaslt_workspace_size_in_byte();

void hipblaslt_gemm_impl(const void *A, const hipDataType A_type, const int64_t lda,
                         const void *scaleA_inv, hipblasOperation_t transA, const void *B,
                         const hipDataType B_type, const int64_t ldb, const void *scaleB_inv,
                         hipblasOperation_t transB, void *D, const hipDataType D_type,
                         const int64_t ldd, const int64_t m, const int64_t n, const int64_t k,
                         void *workspace, const int64_t workspace_size, const bool use_fp8,
                         const bool use_rowwise, hipblasLtHandle_t handle, hipStream_t stream);
// *****************************************

template <typename AType, typename BType, typename CType, typename ACCType> struct CKGemmFP8Params {
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
    uint32_t    num_cu = 0;
};

template <typename ADataType, typename BDataType, typename CDataType, typename AccDataType>
void ck_gemm_fp8(const CKGemmFP8Params<ADataType, BDataType, CDataType, AccDataType> &params);

} // namespace primus_turbo
