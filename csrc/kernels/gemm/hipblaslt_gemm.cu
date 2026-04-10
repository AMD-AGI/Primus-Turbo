// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
//
// See LICENSE for license information.

#include <unordered_map>

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

// ── algo cache ───────────────────────────────────────────────────────────────
// hipblasLtMatmulAlgoGetHeuristic() can cost 100 ms+ on first encounter of a
// new shape.  Cache the winning algo per (shape, dtype, trans, handle) tuple
// so subsequent calls for the same problem are free.

struct AlgoKey {
    int64_t rows_a, cols_a, lda;
    int64_t rows_b, cols_b, ldb;
    int64_t rows_d, cols_d, ldd;
    hipDataType A_type, B_type, D_type;
    hipblasOperation_t transA, transB;
    bool use_low_precision;
    hipblasLtMatmulMatrixScale_t scale_mode;
    hipblasLtHandle_t handle;

    bool operator==(const AlgoKey &o) const {
        return rows_a == o.rows_a && cols_a == o.cols_a && lda == o.lda &&
               rows_b == o.rows_b && cols_b == o.cols_b && ldb == o.ldb &&
               rows_d == o.rows_d && cols_d == o.cols_d && ldd == o.ldd &&
               A_type == o.A_type && B_type == o.B_type && D_type == o.D_type &&
               transA == o.transA && transB == o.transB &&
               use_low_precision == o.use_low_precision && scale_mode == o.scale_mode &&
               handle == o.handle;
    }
};

struct AlgoKeyHash {
    size_t operator()(const AlgoKey &k) const {
        size_t s = 0;
        auto hc  = [&s](auto v) {
            s ^= std::hash<decltype(v)>{}(v) + 0x9e3779b9u + (s << 6) + (s >> 2);
        };
        hc(k.rows_a); hc(k.cols_a); hc(k.lda);
        hc(k.rows_b); hc(k.cols_b); hc(k.ldb);
        hc(k.rows_d); hc(k.cols_d); hc(k.ldd);
        hc(static_cast<int>(k.A_type));
        hc(static_cast<int>(k.B_type));
        hc(static_cast<int>(k.D_type));
        hc(static_cast<int>(k.transA));
        hc(static_cast<int>(k.transB));
        hc(k.use_low_precision);
        hc(static_cast<int>(k.scale_mode));
        hc(reinterpret_cast<uintptr_t>(k.handle));
        return s;
    }
};

static thread_local std::unordered_map<AlgoKey, hipblasLtMatmulHeuristicResult_t, AlgoKeyHash>
    algo_cache;
// ─────────────────────────────────────────────────────────────────────────────

void hipblaslt_gemm_impl(const void *A, const hipDataType A_type, const int64_t rows_a,
                         const int64_t cols_a, const int64_t lda, const void *scaleA_inv,
                         hipblasOperation_t transA, const void *B, const hipDataType B_type,
                         const int64_t rows_b, const int64_t cols_b, const int64_t ldb,
                         const void *scaleB_inv, hipblasOperation_t transB, void *D,
                         const hipDataType D_type, const int64_t rows_d, const int64_t cols_d,
                         const int64_t ldd, void *workspace, const int64_t workspace_size,
                         const bool use_low_precision, hipblasLtMatmulMatrixScale_t scale_mode,
                         hipblasLtHandle_t handle, hipStream_t stream) {
    hipblasLtMatmulDesc_t   operation_desc = nullptr;
    hipblasLtMatrixLayout_t A_desc = nullptr, B_desc = nullptr, D_desc = nullptr;
    hipblasLtEpilogue_t     epilogue          = HIPBLASLT_EPILOGUE_DEFAULT;
    hipblasComputeType_t    gemm_compute_type = HIPBLAS_COMPUTE_32F;

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

    // Look up algo cache; run heuristic search only on first encounter of this shape.
    AlgoKey key{rows_a,  cols_a,   lda,   rows_b,    cols_b,    ldb,
                rows_d,  cols_d,   ldd,   A_type,    B_type,    D_type,
                transA,  transB,   use_low_precision, scale_mode, handle};

    hipblasLtMatmulHeuristicResult_t algo_result{};
    auto it = algo_cache.find(key);
    if (it != algo_cache.end()) {
        algo_result = it->second;
    } else {
        const int request_solutions = 1;
        int returnedAlgoCount = 0;
        hipblasLtMatmulPreference_t preference = nullptr;

        PRIMUS_TURBO_CHECK_HIPBLAS(hipblasLtMatmulPreferenceCreate(&preference));
        PRIMUS_TURBO_CHECK_HIPBLAS(hipblasLtMatmulPreferenceSetAttribute(
            preference, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
            &workspace_size, sizeof(workspace_size)));

        PRIMUS_TURBO_CHECK_HIPBLAS(hipblasLtMatmulAlgoGetHeuristic(
            handle, operation_desc, A_desc, B_desc, D_desc, D_desc, preference,
            request_solutions, &algo_result, &returnedAlgoCount));
        PRIMUS_TURBO_CHECK(returnedAlgoCount > 0,
                           "hipBLASLt: no valid algorithm found for current matmul config");

        PRIMUS_TURBO_CHECK_HIPBLAS(hipblasLtMatmulPreferenceDestroy(preference));

        algo_cache[key] = algo_result;
    }

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
        &algo_result.algo,
        workspace, workspace_size,
        stream));
    // clang-format on

    PRIMUS_TURBO_CHECK_HIPBLAS(hipblasLtMatrixLayoutDestroy(D_desc));
    PRIMUS_TURBO_CHECK_HIPBLAS(hipblasLtMatrixLayoutDestroy(B_desc));
    PRIMUS_TURBO_CHECK_HIPBLAS(hipblasLtMatrixLayoutDestroy(A_desc));
    PRIMUS_TURBO_CHECK_HIPBLAS(hipblasLtMatmulDescDestroy(operation_desc));
}

} // namespace primus_turbo
