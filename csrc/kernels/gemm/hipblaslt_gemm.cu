// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
//
// See LICENSE for license information.

#include "primus_turbo/common.h"
#include "primus_turbo/gemm.h"

namespace primus_turbo {

int64_t get_hipblaslt_workspace_size_in_byte() {
    GPUArch arch = get_current_arch();
    switch (arch) {
    case GPUArch::GFX950:
        return 67108864; // 64 MiB
    case GPUArch::GFX942:
    case GPUArch::UNKNOWN:
        return 33554432; // 32 MiB
    }
}

void hipblaslt_gemm_impl(const void *A, const hipDataType A_type, const int64_t lda,
                         const void *scaleA_inv, hipblasOperation_t transA, const void *B,
                         const hipDataType B_type, const int64_t ldb, const void *scaleB_inv,
                         hipblasOperation_t transB, void *D, const hipDataType D_type,
                         const int64_t ldd, const int64_t m, const int64_t n, const int64_t k,
                         void *workspace, const int64_t workspace_size, const bool use_fp8,
                         hipblasLtMatmulMatrixScale_t scale_mode, hipblasLtHandle_t handle,
                         hipStream_t stream) {
    hipblasLtMatmulDesc_t       operation_desc = nullptr;
    hipblasLtMatrixLayout_t     A_desc = nullptr, B_desc = nullptr, D_desc = nullptr;
    hipblasLtMatmulPreference_t preference        = nullptr;
    hipblasLtEpilogue_t         epilogue          = HIPBLASLT_EPILOGUE_DEFAULT;
    hipblasComputeType_t        gemm_compute_type = HIPBLAS_COMPUTE_32F;

    PRIMUS_TURBO_CHECK_HIPBLAS(hipblasLtMatrixLayoutCreate(
        &A_desc, A_type, transA == HIPBLAS_OP_N ? m : k, transA == HIPBLAS_OP_N ? k : m, lda));
    PRIMUS_TURBO_CHECK_HIPBLAS(hipblasLtMatrixLayoutCreate(
        &B_desc, B_type, transB == HIPBLAS_OP_N ? k : n, transB == HIPBLAS_OP_N ? n : k, ldb));
    PRIMUS_TURBO_CHECK_HIPBLAS(hipblasLtMatrixLayoutCreate(&D_desc, D_type, m, n, ldd));

    PRIMUS_TURBO_CHECK_HIPBLAS(
        hipblasLtMatmulDescCreate(&operation_desc, gemm_compute_type, HIP_R_32F));
    PRIMUS_TURBO_CHECK_HIPBLAS(hipblasLtMatmulDescSetAttribute(
        operation_desc, HIPBLASLT_MATMUL_DESC_TRANSA, &transA, sizeof(transA)));
    PRIMUS_TURBO_CHECK_HIPBLAS(hipblasLtMatmulDescSetAttribute(
        operation_desc, HIPBLASLT_MATMUL_DESC_TRANSB, &transB, sizeof(transB)));
    PRIMUS_TURBO_CHECK_HIPBLAS(hipblasLtMatmulDescSetAttribute(
        operation_desc, HIPBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue)));

    if (use_fp8) {
        PRIMUS_TURBO_CHECK(scaleA_inv != nullptr);
        PRIMUS_TURBO_CHECK(scaleB_inv != nullptr);

        hipblasLtMatmulDescAttributes_t scaleA_inv_ptr_desc = HIPBLASLT_MATMUL_DESC_A_SCALE_POINTER;
        hipblasLtMatmulDescAttributes_t scaleB_inv_ptr_desc = HIPBLASLT_MATMUL_DESC_B_SCALE_POINTER;

        PRIMUS_TURBO_CHECK_HIPBLAS(hipblasLtMatmulDescSetAttribute(
            operation_desc, HIPBLASLT_MATMUL_DESC_A_SCALE_MODE, &scale_mode, sizeof(scale_mode)));
        PRIMUS_TURBO_CHECK_HIPBLAS(hipblasLtMatmulDescSetAttribute(
            operation_desc, HIPBLASLT_MATMUL_DESC_B_SCALE_MODE, &scale_mode, sizeof(scale_mode)));

        PRIMUS_TURBO_CHECK_HIPBLAS(hipblasLtMatmulDescSetAttribute(
            operation_desc, scaleA_inv_ptr_desc, &scaleA_inv, sizeof(scaleA_inv)));
        PRIMUS_TURBO_CHECK_HIPBLAS(hipblasLtMatmulDescSetAttribute(
            operation_desc, scaleB_inv_ptr_desc, &scaleB_inv, sizeof(scaleB_inv)));
    }

    const int                                     request_solutions = 1;
    std::vector<hipblasLtMatmulHeuristicResult_t> algos(request_solutions);
    int                                           returnedAlgoCount = 0;

    PRIMUS_TURBO_CHECK_HIPBLAS(hipblasLtMatmulPreferenceCreate(&preference));
    PRIMUS_TURBO_CHECK_HIPBLAS(
        hipblasLtMatmulPreferenceSetAttribute(preference, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                              &workspace_size, sizeof(workspace_size)));

    PRIMUS_TURBO_CHECK_HIPBLAS(hipblasLtMatmulAlgoGetHeuristic(
        handle, operation_desc, A_desc, B_desc, D_desc, D_desc, preference, request_solutions,
        algos.data(), &returnedAlgoCount));
    PRIMUS_TURBO_CHECK(returnedAlgoCount > 0,
                       "hipBLASLt: no valid algorithm found for current matmul config");

    const float alpha = 1.0;
    const float beta  = 0.0;
    // clang-format off
    PRIMUS_TURBO_CHECK_HIPBLAS(hipblasLtMatmul(
        handle,
        operation_desc,
        &alpha,
        A, A_desc,
        B, B_desc,
        &beta,
        D, D_desc,
        D, D_desc,
        &algos[0].algo,
        workspace, workspace_size,
        stream));
    // clang-format on

    PRIMUS_TURBO_CHECK_HIPBLAS(hipblasLtMatrixLayoutDestroy(D_desc));
    PRIMUS_TURBO_CHECK_HIPBLAS(hipblasLtMatrixLayoutDestroy(B_desc));
    PRIMUS_TURBO_CHECK_HIPBLAS(hipblasLtMatrixLayoutDestroy(A_desc));
    PRIMUS_TURBO_CHECK_HIPBLAS(hipblasLtMatmulDescDestroy(operation_desc));
    PRIMUS_TURBO_CHECK_HIPBLAS(hipblasLtMatmulPreferenceDestroy(preference));
}

} // namespace primus_turbo
