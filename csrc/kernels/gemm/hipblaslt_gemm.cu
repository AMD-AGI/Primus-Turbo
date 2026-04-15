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

// ── Algo cache key: optimistic M-invariant fast path ─────────────────────────
// Reusing an algo across different M values avoids re-autotuning every expert
// shape in grouped GEMM. Some BF16 hipBLASLt configs still require an exact-M
// override, so an exact-shape cache is checked first and populated on fallback.
struct AlgoKey {
    int64_t                      cols_a;              // K (M-invariant)
    int64_t                      rows_b, cols_b, ldb; // weight matrix shape (fixed per layer)
    int64_t                      cols_d;              // N (M-invariant)
    hipDataType                  A_type, B_type, D_type;
    hipblasOperation_t           transA, transB;
    bool                         use_low_precision;
    hipblasLtMatmulMatrixScale_t scale_mode;
    hipblasLtHandle_t            handle;

    bool operator==(const AlgoKey &o) const {
        return cols_a == o.cols_a && rows_b == o.rows_b && cols_b == o.cols_b && ldb == o.ldb &&
               cols_d == o.cols_d && A_type == o.A_type && B_type == o.B_type &&
               D_type == o.D_type && transA == o.transA && transB == o.transB &&
               use_low_precision == o.use_low_precision && scale_mode == o.scale_mode &&
               handle == o.handle;
    }
};

struct AlgoKeyHash {
    size_t operator()(const AlgoKey &k) const {
        size_t s  = 0;
        auto   hc = [&s](auto v) {
            s ^= std::hash<decltype(v)>{}(v) + 0x9e3779b9u + (s << 6) + (s >> 2);
        };
        hc(k.cols_a);
        hc(k.rows_b);
        hc(k.cols_b);
        hc(k.ldb);
        hc(k.cols_d);
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

// ── Descriptor cache key: includes full shape (M-dependent) ─────────────────
// Matrix layouts must match exact dimensions, so descriptors are cached per
// full shape. Same-shape experts (balanced routing) still get cache hits.
struct DescKey {
    int64_t                      rows_a, cols_a, lda;
    int64_t                      rows_b, cols_b, ldb;
    int64_t                      rows_d, cols_d, ldd;
    hipDataType                  A_type, B_type, D_type;
    hipblasOperation_t           transA, transB;
    bool                         use_low_precision;
    hipblasLtMatmulMatrixScale_t scale_mode;
    hipblasLtHandle_t            handle;

    bool operator==(const DescKey &o) const {
        return rows_a == o.rows_a && cols_a == o.cols_a && lda == o.lda && rows_b == o.rows_b &&
               cols_b == o.cols_b && ldb == o.ldb && rows_d == o.rows_d && cols_d == o.cols_d &&
               ldd == o.ldd && A_type == o.A_type && B_type == o.B_type && D_type == o.D_type &&
               transA == o.transA && transB == o.transB &&
               use_low_precision == o.use_low_precision && scale_mode == o.scale_mode &&
               handle == o.handle;
    }
};

struct DescKeyHash {
    size_t operator()(const DescKey &k) const {
        size_t s  = 0;
        auto   hc = [&s](auto v) {
            s ^= std::hash<decltype(v)>{}(v) + 0x9e3779b9u + (s << 6) + (s >> 2);
        };
        hc(k.rows_a);
        hc(k.cols_a);
        hc(k.lda);
        hc(k.rows_b);
        hc(k.cols_b);
        hc(k.ldb);
        hc(k.rows_d);
        hc(k.cols_d);
        hc(k.ldd);
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

struct DescCache {
    hipblasLtMatrixLayout_t A_desc  = nullptr;
    hipblasLtMatrixLayout_t B_desc  = nullptr;
    hipblasLtMatrixLayout_t D_desc  = nullptr;
    hipblasLtMatmulDesc_t   op_desc = nullptr;
};

static thread_local std::unordered_map<DescKey, DescCache, DescKeyHash> desc_cache;
static thread_local std::unordered_map<DescKey, hipblasLtMatmulHeuristicResult_t, DescKeyHash>
    exact_algo_cache;
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
    // Look up or create cached descriptors for this exact shape.
    DescKey dkey{rows_a,     cols_a, lda,    rows_b, cols_b,
                 ldb,        rows_d, cols_d, ldd,    A_type,
                 B_type,     D_type, transA, transB, use_low_precision,
                 scale_mode, handle};

    auto &dc = desc_cache[dkey];
    if (dc.op_desc == nullptr) {
        hipblasLtEpilogue_t  epilogue          = HIPBLASLT_EPILOGUE_DEFAULT;
        hipblasComputeType_t gemm_compute_type = HIPBLAS_COMPUTE_32F;

        PRIMUS_TURBO_CHECK_HIPBLAS(
            hipblasLtMatrixLayoutCreate(&dc.A_desc, A_type, rows_a, cols_a, lda));
        PRIMUS_TURBO_CHECK_HIPBLAS(
            hipblasLtMatrixLayoutCreate(&dc.B_desc, B_type, rows_b, cols_b, ldb));
        PRIMUS_TURBO_CHECK_HIPBLAS(
            hipblasLtMatrixLayoutCreate(&dc.D_desc, D_type, rows_d, cols_d, ldd));

        PRIMUS_TURBO_CHECK_HIPBLAS(
            hipblasLtMatmulDescCreate(&dc.op_desc, gemm_compute_type, HIP_R_32F));
        PRIMUS_TURBO_CHECK_HIPBLAS(hipblasLtMatmulDescSetAttribute(
            dc.op_desc, HIPBLASLT_MATMUL_DESC_TRANSA, &transA, sizeof(transA)));
        PRIMUS_TURBO_CHECK_HIPBLAS(hipblasLtMatmulDescSetAttribute(
            dc.op_desc, HIPBLASLT_MATMUL_DESC_TRANSB, &transB, sizeof(transB)));
        PRIMUS_TURBO_CHECK_HIPBLAS(hipblasLtMatmulDescSetAttribute(
            dc.op_desc, HIPBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue)));

        if (use_low_precision) {
            if (scale_mode == HIPBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0) {
                PRIMUS_TURBO_CHECK(
                    is_gfx950(),
                    "The HIPBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0 only support on gfx950.");
            }
            PRIMUS_TURBO_CHECK_HIPBLAS(hipblasLtMatmulDescSetAttribute(
                dc.op_desc, HIPBLASLT_MATMUL_DESC_A_SCALE_MODE, &scale_mode, sizeof(scale_mode)));
            PRIMUS_TURBO_CHECK_HIPBLAS(hipblasLtMatmulDescSetAttribute(
                dc.op_desc, HIPBLASLT_MATMUL_DESC_B_SCALE_MODE, &scale_mode, sizeof(scale_mode)));
        }
    }

    hipblasLtMatmulDesc_t   operation_desc = dc.op_desc;
    hipblasLtMatrixLayout_t A_desc = dc.A_desc, B_desc = dc.B_desc, D_desc = dc.D_desc;

    // Scale pointers change per call — must update every time for FP8.
    if (use_low_precision) {
        PRIMUS_TURBO_CHECK(scaleA_inv != nullptr);
        PRIMUS_TURBO_CHECK(scaleB_inv != nullptr);
        PRIMUS_TURBO_CHECK_HIPBLAS(
            hipblasLtMatmulDescSetAttribute(operation_desc, HIPBLASLT_MATMUL_DESC_A_SCALE_POINTER,
                                            &scaleA_inv, sizeof(scaleA_inv)));
        PRIMUS_TURBO_CHECK_HIPBLAS(
            hipblasLtMatmulDescSetAttribute(operation_desc, HIPBLASLT_MATMUL_DESC_B_SCALE_POINTER,
                                            &scaleB_inv, sizeof(scaleB_inv)));
    }

    auto autotune_algo = [&]() -> hipblasLtMatmulHeuristicResult_t {
        static constexpr int             kMaxCandidates = 8;
        static constexpr int             kWarmupIters   = 10;
        static constexpr int             kBenchIters    = 30;
        hipblasLtMatmulHeuristicResult_t candidates[kMaxCandidates];
        hipblasLtMatmulHeuristicResult_t tuned_algo{};
        int                              returnedAlgoCount = 0;
        hipblasLtMatmulPreference_t      preference        = nullptr;

        PRIMUS_TURBO_CHECK_HIPBLAS(hipblasLtMatmulPreferenceCreate(&preference));
        PRIMUS_TURBO_CHECK_HIPBLAS(hipblasLtMatmulPreferenceSetAttribute(
            preference, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace_size,
            sizeof(workspace_size)));

        PRIMUS_TURBO_CHECK_HIPBLAS(hipblasLtMatmulAlgoGetHeuristic(
            handle, operation_desc, A_desc, B_desc, D_desc, D_desc, preference, kMaxCandidates,
            candidates, &returnedAlgoCount));
        PRIMUS_TURBO_CHECK(returnedAlgoCount > 0,
                           "hipBLASLt: no valid algorithm found for current matmul config");

        PRIMUS_TURBO_CHECK_HIPBLAS(hipblasLtMatmulPreferenceDestroy(preference));

        if (returnedAlgoCount == 1) {
            tuned_algo = candidates[0];
        } else {
            // Autotune: benchmark each candidate, pick fastest.
            const float a1 = 1.0f, b0 = 0.0f;
            hipEvent_t  ev_start, ev_stop;
            PRIMUS_TURBO_CHECK_HIP(hipEventCreate(&ev_start));
            PRIMUS_TURBO_CHECK_HIP(hipEventCreate(&ev_stop));

            float best_ms  = 1e30f;
            int   best_idx = 0;
            for (int c = 0; c < returnedAlgoCount; ++c) {
                // warm-up
                for (int w = 0; w < kWarmupIters; ++w) {
                    (void) hipblasLtMatmul(handle, operation_desc, &a1, A, A_desc, B, B_desc, &b0,
                                           D, D_desc, D, D_desc, &candidates[c].algo, workspace,
                                           workspace_size, stream);
                }

                PRIMUS_TURBO_CHECK_HIP(hipEventRecord(ev_start, stream));
                for (int i = 0; i < kBenchIters; ++i) {
                    (void) hipblasLtMatmul(handle, operation_desc, &a1, A, A_desc, B, B_desc, &b0,
                                           D, D_desc, D, D_desc, &candidates[c].algo, workspace,
                                           workspace_size, stream);
                }
                PRIMUS_TURBO_CHECK_HIP(hipEventRecord(ev_stop, stream));
                PRIMUS_TURBO_CHECK_HIP(hipEventSynchronize(ev_stop));

                float ms = 0;
                PRIMUS_TURBO_CHECK_HIP(hipEventElapsedTime(&ms, ev_start, ev_stop));
                if (ms < best_ms) {
                    best_ms  = ms;
                    best_idx = c;
                }
            }
            PRIMUS_TURBO_CHECK_HIP(hipEventDestroy(ev_start));
            PRIMUS_TURBO_CHECK_HIP(hipEventDestroy(ev_stop));
            tuned_algo = candidates[best_idx];
        }

        return tuned_algo;
    };

    // Check exact-shape overrides first, then fall back to the optimistic
    // M-invariant cache to avoid repeated autotune on similar expert shapes.
    AlgoKey akey{cols_a,
                 rows_b,
                 cols_b,
                 ldb,
                 cols_d,
                 A_type,
                 B_type,
                 D_type,
                 transA,
                 transB,
                 use_low_precision,
                 scale_mode,
                 handle};

    hipblasLtMatmulHeuristicResult_t algo_result{};
    bool                             used_shared_algo_cache = false;
    auto                             exact_it               = exact_algo_cache.find(dkey);
    if (exact_it != exact_algo_cache.end()) {
        algo_result = exact_it->second;
    } else {
        auto it = algo_cache.find(akey);
        if (it != algo_cache.end()) {
            algo_result            = it->second;
            used_shared_algo_cache = true;
        } else {
            algo_result      = autotune_algo();
            algo_cache[akey] = algo_result;
        }
    }

    const float     alpha = 1.0;
    const float     beta  = 0.0;
    hipblasStatus_t status =
        hipblasLtMatmul(handle, operation_desc, &alpha, A, A_desc, B, B_desc, &beta, D, D_desc, D,
                        D_desc, &algo_result.algo, workspace, workspace_size, stream);

    if (status == HIPBLAS_STATUS_INVALID_VALUE && used_shared_algo_cache) {
        // Some configs still need an exact-shape algo even when N/K/trans/dtype
        // match. Retune for this descriptor and remember the exact override.
        algo_result            = autotune_algo();
        exact_algo_cache[dkey] = algo_result;
        status =
            hipblasLtMatmul(handle, operation_desc, &alpha, A, A_desc, B, B_desc, &beta, D, D_desc,
                            D, D_desc, &algo_result.algo, workspace, workspace_size, stream);
    }

    PRIMUS_TURBO_CHECK_HIPBLAS(status);

    // Descriptors are cached — no destroy needed per call.
}

} // namespace primus_turbo
