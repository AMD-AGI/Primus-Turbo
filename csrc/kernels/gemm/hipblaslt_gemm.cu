// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
//
// See LICENSE for license information.

#include <hipblaslt/hipblaslt-ext.hpp>
#include <map>
#include <mutex>
#include <string>
#include <tuple>

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

void hipblaslt_gemm_impl(const void *A, const hipDataType A_type, const int64_t rows_a,
                         const int64_t cols_a, const int64_t lda, const void *scaleA_inv,
                         hipblasOperation_t transA, const void *B, const hipDataType B_type,
                         const int64_t rows_b, const int64_t cols_b, const int64_t ldb,
                         const void *scaleB_inv, hipblasOperation_t transB, void *D,
                         const hipDataType D_type, const int64_t rows_d, const int64_t cols_d,
                         const int64_t ldd, void *workspace, const int64_t workspace_size,
                         const bool use_low_precision, hipblasLtMatmulMatrixScale_t scale_mode,
                         hipblasLtHandle_t handle, hipStream_t stream) {
    hipblasLtMatmulDesc_t       operation_desc = nullptr;
    hipblasLtMatrixLayout_t     A_desc = nullptr, B_desc = nullptr, D_desc = nullptr;
    hipblasLtMatmulPreference_t preference        = nullptr;
    hipblasLtEpilogue_t         epilogue          = HIPBLASLT_EPILOGUE_DEFAULT;
    hipblasComputeType_t        gemm_compute_type = HIPBLAS_COMPUTE_32F;

    PRIMUS_TURBO_CHECK_HIPBLAS(hipblasLtMatrixLayoutCreate(&A_desc, A_type, rows_a, cols_a, lda));
    PRIMUS_TURBO_CHECK_HIPBLAS(hipblasLtMatrixLayoutCreate(&B_desc, B_type, rows_b, cols_b, ldb));
    PRIMUS_TURBO_CHECK_HIPBLAS(hipblasLtMatrixLayoutCreate(&D_desc, D_type, rows_d, cols_d, ldd));

    PRIMUS_TURBO_CHECK_HIPBLAS(
        hipblasLtMatmulDescCreate(&operation_desc, gemm_compute_type, HIP_R_32F));
    PRIMUS_TURBO_CHECK_HIPBLAS(hipblasLtMatmulDescSetAttribute(
        operation_desc, HIPBLASLT_MATMUL_DESC_TRANSA, &transA, sizeof(transA)));
    PRIMUS_TURBO_CHECK_HIPBLAS(hipblasLtMatmulDescSetAttribute(
        operation_desc, HIPBLASLT_MATMUL_DESC_TRANSB, &transB, sizeof(transB)));
    PRIMUS_TURBO_CHECK_HIPBLAS(hipblasLtMatmulDescSetAttribute(
        operation_desc, HIPBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue)));

    if (use_low_precision) {
        if (scale_mode == HIPBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0) {
            PRIMUS_TURBO_CHECK(
                is_gfx950(),
                "The HIPBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0 only support on gfx950.");
        }
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
    // Cap the workspace hint for low-precision (FP8/FP4) paths to steer
    // hipBLASLt away from the Stream-K persistent kernel that intermittently
    // stalls under multi-stream grouped GEMM (~500us -> 300ms..25s on MI355X).
    int64_t pref_workspace_size = use_low_precision ? int64_t{8 * 1024 * 1024} : workspace_size;
    PRIMUS_TURBO_CHECK_HIPBLAS(
        hipblasLtMatmulPreferenceSetAttribute(preference, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                              &pref_workspace_size, sizeof(pref_workspace_size)));

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

bool hipblaslt_gemm_is_streamk(hipblasLtHandle_t handle, const hipDataType A_type,
                               const int64_t rows_a, const int64_t cols_a, const int64_t lda,
                               hipblasOperation_t transA, const hipDataType B_type,
                               const int64_t rows_b, const int64_t cols_b, const int64_t ldb,
                               hipblasOperation_t transB, const hipDataType D_type,
                               const int64_t rows_d, const int64_t cols_d, const int64_t ldd,
                               const bool                         use_low_precision,
                               const hipblasLtMatmulMatrixScale_t scale_mode) {
    auto key = std::make_tuple(rows_a, cols_a, rows_b, cols_b, rows_d, cols_d, (int) A_type,
                               (int) B_type, (int) D_type, (int) transA, (int) transB,
                               (int) use_low_precision, (int) scale_mode);
    static std::map<decltype(key), bool> cache;
    static std::mutex                    cache_mtx;
    {
        std::lock_guard<std::mutex> lock(cache_mtx);
        if (auto it = cache.find(key); it != cache.end()) return it->second;
    }

    // Heuristic call must mirror hipblaslt_gemm_impl (same workspace hint -> same algo).
    hipblasLtMatmulDesc_t       op = nullptr;
    hipblasLtMatrixLayout_t     A_desc = nullptr, B_desc = nullptr, D_desc = nullptr;
    hipblasLtMatmulPreference_t pref     = nullptr;
    hipblasLtEpilogue_t         epi      = HIPBLASLT_EPILOGUE_DEFAULT;
    int64_t                     pref_ws  = use_low_precision ? int64_t{8 * 1024 * 1024}
                                                             : get_hipblaslt_workspace_size_in_byte();
    PRIMUS_TURBO_CHECK_HIPBLAS(hipblasLtMatrixLayoutCreate(&A_desc, A_type, rows_a, cols_a, lda));
    PRIMUS_TURBO_CHECK_HIPBLAS(hipblasLtMatrixLayoutCreate(&B_desc, B_type, rows_b, cols_b, ldb));
    PRIMUS_TURBO_CHECK_HIPBLAS(hipblasLtMatrixLayoutCreate(&D_desc, D_type, rows_d, cols_d, ldd));
    PRIMUS_TURBO_CHECK_HIPBLAS(hipblasLtMatmulDescCreate(&op, HIPBLAS_COMPUTE_32F, HIP_R_32F));
    auto set_attr = [&](hipblasLtMatmulDescAttributes_t a, const auto &v) {
        PRIMUS_TURBO_CHECK_HIPBLAS(hipblasLtMatmulDescSetAttribute(op, a, &v, sizeof(v)));
    };
    set_attr(HIPBLASLT_MATMUL_DESC_TRANSA, transA);
    set_attr(HIPBLASLT_MATMUL_DESC_TRANSB, transB);
    set_attr(HIPBLASLT_MATMUL_DESC_EPILOGUE, epi);
    if (use_low_precision) {
        set_attr(HIPBLASLT_MATMUL_DESC_A_SCALE_MODE, scale_mode);
        set_attr(HIPBLASLT_MATMUL_DESC_B_SCALE_MODE, scale_mode);
    }
    PRIMUS_TURBO_CHECK_HIPBLAS(hipblasLtMatmulPreferenceCreate(&pref));
    PRIMUS_TURBO_CHECK_HIPBLAS(hipblasLtMatmulPreferenceSetAttribute(
        pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &pref_ws, sizeof(pref_ws)));

    hipblasLtMatmulHeuristicResult_t algo{};
    int                              n_algo = 0;
    PRIMUS_TURBO_CHECK_HIPBLAS(hipblasLtMatmulAlgoGetHeuristic(
        handle, op, A_desc, B_desc, D_desc, D_desc, pref, 1, &algo, &n_algo));

    // Tensile Stream-K kernel names carry GSUAMBSK (gfx942 FP8) or CMS (gfx950, see 7743364).
    bool is_sk = false;
    if (n_algo > 0) {
        const auto name = hipblaslt_ext::getKernelNameFromAlgo(handle, algo.algo);
        is_sk = name.find("GSUAMBSK") != std::string::npos ||
                name.find("_CMS_") != std::string::npos;
    }

    PRIMUS_TURBO_CHECK_HIPBLAS(hipblasLtMatrixLayoutDestroy(D_desc));
    PRIMUS_TURBO_CHECK_HIPBLAS(hipblasLtMatrixLayoutDestroy(B_desc));
    PRIMUS_TURBO_CHECK_HIPBLAS(hipblasLtMatrixLayoutDestroy(A_desc));
    PRIMUS_TURBO_CHECK_HIPBLAS(hipblasLtMatmulDescDestroy(op));
    PRIMUS_TURBO_CHECK_HIPBLAS(hipblasLtMatmulPreferenceDestroy(pref));

    std::lock_guard<std::mutex> lock(cache_mtx);
    cache[key] = is_sk;
    return is_sk;
}

} // namespace primus_turbo
