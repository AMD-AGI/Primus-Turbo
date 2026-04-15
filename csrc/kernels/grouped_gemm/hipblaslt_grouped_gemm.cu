// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
//
// See LICENSE for license information.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <set>
#include <tuple>
#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt.h>

#include "primus_turbo/gemm.h"
#include "primus_turbo/grouped_gemm.h"

namespace primus_turbo {

static constexpr size_t kMaxNumStreams = 8;

// Per-expert token threshold: above this, individual GEMMs are large enough
// to saturate the GPU, so serial dispatch avoids resource contention.
// Below this, GEMMs are small and benefit from concurrent multi-stream
// dispatch to overlap kernel launch overhead.
static constexpr size_t kSerialThreshold = 512;

std::int64_t get_hipblaslt_grouped_gemm_workspace_size() {
    // Multi-stream path needs one workspace per stream.
    return kMaxNumStreams * get_hipblaslt_workspace_size_in_byte();
}

class HipblasltGroupedGemm {
public:
    HipblasltGroupedGemm() {
        PRIMUS_TURBO_CHECK_HIPBLAS(hipblasLtCreate(&handle_));
        PRIMUS_TURBO_CHECK_HIP(hipEventCreateWithFlags(&sync_event_, hipEventDisableTiming));

        for (size_t i = 0; i < kMaxNumStreams; ++i) {
            PRIMUS_TURBO_CHECK_HIPBLAS(hipblasLtCreate(&par_handles_[i]));
            PRIMUS_TURBO_CHECK_HIP(
                hipStreamCreateWithPriority(&par_streams_[i], hipStreamNonBlocking, -1));
            PRIMUS_TURBO_CHECK_HIP(
                hipEventCreateWithFlags(&par_events_[i], hipEventDisableTiming));
        }
    }

    ~HipblasltGroupedGemm() {
        if (sync_event_) (void)hipEventDestroy(sync_event_);
        for (size_t i = 0; i < kMaxNumStreams; ++i) {
            if (par_streams_[i]) (void)hipStreamDestroy(par_streams_[i]);
            if (par_handles_[i]) (void)hipblasLtDestroy(par_handles_[i]);
            if (par_events_[i])  (void)hipEventDestroy(par_events_[i]);
        }
        if (handle_) (void)hipblasLtDestroy(handle_);
    }

    void check(const HipblasltGroupedGemmParams &params) {
        PRIMUS_TURBO_CHECK(params.a_shape.size() == 2);
        if (params.transA) {
            PRIMUS_TURBO_CHECK(params.b_shape.size() == 2);
            PRIMUS_TURBO_CHECK(params.c_shape.size() == 3);
            PRIMUS_TURBO_CHECK(params.a_shape[0] == params.b_shape[0]);
            PRIMUS_TURBO_CHECK(params.c_shape[0] == params.group_num);
            PRIMUS_TURBO_CHECK(params.c_shape[1] == params.a_shape[1]);
            PRIMUS_TURBO_CHECK(params.c_shape[2] == params.b_shape[1]);
        } else {
            PRIMUS_TURBO_CHECK(params.b_shape.size() == 3);
            PRIMUS_TURBO_CHECK(params.c_shape.size() == 2);
            PRIMUS_TURBO_CHECK(params.b_shape[0] == params.group_num);
        }
    }

    void run(const HipblasltGroupedGemmParams &params, const bool pre_sync) {
        if (pre_sync) {
            PRIMUS_TURBO_CHECK_HIP(hipStreamSynchronize(params.stream));
        }

        check(params);
        compute_args(params);

        const size_t num_gemms = gemm_ptrs_.size();
        if (num_gemms == 0) return;

        // Warm the hipBLASLt algo cache with balanced M (total_M / group_num)
        // so the tuned algo is representative of the average expert size,
        // not biased by whichever expert happens to be dispatched first.
        warm_algo_cache(params, num_gemms);

        // Determine max per-expert token count to choose dispatch strategy.
        size_t max_tokens = 0;
        for (size_t i = 0; i < num_gemms; ++i) {
            // fwd/dgrad: cols_c = token count; wgrad: cols_a = token count
            size_t tokens = params.transA ? cols_b_[i] : cols_c_[i];
            max_tokens = std::max(max_tokens, tokens);
        }

        if (max_tokens >= kSerialThreshold || num_gemms <= kMaxNumStreams) {
            dispatch_serial(params, num_gemms);
        } else {
            dispatch_parallel(params, num_gemms);
        }
    }

private:
    void warm_algo_cache(const HipblasltGroupedGemmParams &params, size_t num_gemms) {
        // Use (rows_b, cols_a, transA, transB) as a compact shape key.
        // This matches the M-invariant part of AlgoKey in hipblaslt_gemm.cu.
        auto shape_key = std::make_tuple(rows_b_[0], cols_a_[0],
                                         params.transA, params.transB);
        if (warmed_shapes_.count(shape_key)) return;

        // Compute balanced M = total_tokens / group_num.
        int64_t total_tokens = 0;
        for (size_t i = 0; i < num_gemms; ++i) {
            total_tokens += params.transA ? cols_b_[i] : cols_c_[i];
        }
        const int64_t balanced_m = total_tokens / params.group_num;
        if (balanced_m <= 0) return;

        // Call hipblaslt_gemm_impl with balanced M to trigger algo tuning
        // with a representative workload size.
        const int64_t bal_cols_a = params.transA ? rows_a_[0] : balanced_m;
        const int64_t bal_cols_b = params.transA ? balanced_m : cols_b_[0];
        const int64_t bal_cols_d = params.transA ? cols_c_[0] : balanced_m;
        // clang-format off
        hipblaslt_gemm_impl(
            gemm_ptrs_[0].b_ptr, params.b_type, rows_b_[0], bal_cols_b, ld_b_[0],
            gemm_ptrs_[0].b_scale_ptr,
            params.transB ? HIPBLAS_OP_T : HIPBLAS_OP_N,
            gemm_ptrs_[0].a_ptr, params.a_type, rows_a_[0], bal_cols_a, ld_a_[0],
            gemm_ptrs_[0].a_scale_ptr,
            params.transA ? HIPBLAS_OP_T : HIPBLAS_OP_N,
            gemm_ptrs_[0].c_ptr, params.c_type, rows_c_[0], bal_cols_d, ld_c_[0],
            params.workspace, get_hipblaslt_workspace_size_in_byte(),
            params.use_low_precision,
            params.scale_mode,
            handle_,
            params.stream
        );
        // clang-format on
        PRIMUS_TURBO_CHECK_HIP(hipStreamSynchronize(params.stream));
        warmed_shapes_.insert(shape_key);
    }

    void dispatch_serial(const HipblasltGroupedGemmParams &params, size_t num_gemms) {
        for (size_t idx = 0; idx < num_gemms; ++idx) {
            // clang-format off
            hipblaslt_gemm_impl(
                gemm_ptrs_[idx].b_ptr, params.b_type, rows_b_[idx], cols_b_[idx], ld_b_[idx],
                gemm_ptrs_[idx].b_scale_ptr,
                params.transB ? HIPBLAS_OP_T : HIPBLAS_OP_N,
                gemm_ptrs_[idx].a_ptr, params.a_type, rows_a_[idx], cols_a_[idx], ld_a_[idx],
                gemm_ptrs_[idx].a_scale_ptr,
                params.transA ? HIPBLAS_OP_T : HIPBLAS_OP_N,
                gemm_ptrs_[idx].c_ptr, params.c_type, rows_c_[idx], cols_c_[idx], ld_c_[idx],
                params.workspace, get_hipblaslt_workspace_size_in_byte(),
                params.use_low_precision,
                params.scale_mode,
                handle_,
                params.stream
            );
            // clang-format on
        }
    }

    void dispatch_parallel(const HipblasltGroupedGemmParams &params, size_t num_gemms) {
        const size_t num_streams = std::min(kMaxNumStreams, num_gemms);
        char *workspace_base = static_cast<char *>(params.workspace);

        // Fork: make compute streams wait for the caller's stream.
        PRIMUS_TURBO_CHECK_HIP(hipEventRecord(sync_event_, params.stream));
        for (size_t s = 0; s < num_streams; ++s) {
            PRIMUS_TURBO_CHECK_HIP(hipStreamWaitEvent(par_streams_[s], sync_event_, 0));
        }

        // Round-robin dispatch across streams.
        for (size_t idx = 0; idx < num_gemms; ++idx) {
            const size_t s = idx % kMaxNumStreams;
            void *ws = workspace_base + s * get_hipblaslt_workspace_size_in_byte();
            // clang-format off
            hipblaslt_gemm_impl(
                gemm_ptrs_[idx].b_ptr, params.b_type, rows_b_[idx], cols_b_[idx], ld_b_[idx],
                gemm_ptrs_[idx].b_scale_ptr,
                params.transB ? HIPBLAS_OP_T : HIPBLAS_OP_N,
                gemm_ptrs_[idx].a_ptr, params.a_type, rows_a_[idx], cols_a_[idx], ld_a_[idx],
                gemm_ptrs_[idx].a_scale_ptr,
                params.transA ? HIPBLAS_OP_T : HIPBLAS_OP_N,
                gemm_ptrs_[idx].c_ptr, params.c_type, rows_c_[idx], cols_c_[idx], ld_c_[idx],
                ws, get_hipblaslt_workspace_size_in_byte(),
                params.use_low_precision,
                params.scale_mode,
                par_handles_[s],
                par_streams_[s]
            );
            // clang-format on
        }

        // Join: caller's stream waits for all compute streams.
        for (size_t s = 0; s < num_streams; ++s) {
            PRIMUS_TURBO_CHECK_HIP(hipEventRecord(par_events_[s], par_streams_[s]));
            PRIMUS_TURBO_CHECK_HIP(hipStreamWaitEvent(params.stream, par_events_[s], 0));
        }
    }

    struct GemmPtr {
        const void *a_ptr       = nullptr;
        const void *a_scale_ptr = nullptr;
        const void *b_ptr       = nullptr;
        const void *b_scale_ptr = nullptr;
        void       *c_ptr       = nullptr;
    };

    void compute_args(const HipblasltGroupedGemmParams &params) {

        group_lens_host_.resize(params.group_num);
        if (params.group_lens_on_host) {
            std::memcpy(group_lens_host_.data(), params.group_lens_ptr,
                        params.group_num * sizeof(int64_t));
        } else {
            PRIMUS_TURBO_CHECK_HIP(hipMemcpy(group_lens_host_.data(), params.group_lens_ptr,
                                             params.group_num * sizeof(int64_t),
                                             hipMemcpyDeviceToHost));
        }

        int valid_group_num = 0;
        for (size_t i = 0; i < params.group_num; ++i) {
            valid_group_num += group_lens_host_[i] > 0 ? 1 : 0;
        }

        const char *a_ptr = static_cast<const char *>(params.a_ptr);
        const char *b_ptr = static_cast<const char *>(params.b_ptr);
        char       *c_ptr = static_cast<char *>(params.c_ptr);

        const int64_t b_expert_stride = params.transA
            ? 0
            : get_dim(params.b_shape, -1) * get_dim(params.b_shape, -2) *
                  hipblaslt_dtype_bytes(params.b_type);

        gemm_ptrs_.resize(valid_group_num);
        ld_a_.resize(valid_group_num);
        ld_b_.resize(valid_group_num);
        ld_c_.resize(valid_group_num);
        rows_a_.resize(valid_group_num);
        cols_a_.resize(valid_group_num);
        rows_b_.resize(valid_group_num);
        cols_b_.resize(valid_group_num);
        rows_c_.resize(valid_group_num);
        cols_c_.resize(valid_group_num);

        int write_idx = 0;
        for (size_t i = 0; i < params.group_num; ++i) {
            int64_t len = group_lens_host_[i];

            if (params.transA && len == 0) {
                int64_t c_rows_t = get_dim(params.c_shape, -1);
                int64_t c_cols_t = get_dim(params.c_shape, -2);
                int64_t c_size_t = c_rows_t * c_cols_t * hipblaslt_dtype_bytes(params.c_type);
                PRIMUS_TURBO_CHECK_HIP(hipMemsetAsync(c_ptr, 0, c_size_t, params.stream));
                c_ptr += c_size_t;
            }

            if (len == 0)
                continue;

            gemm_ptrs_[write_idx].a_ptr = a_ptr;
            gemm_ptrs_[write_idx].b_ptr = params.transA
                ? b_ptr
                : static_cast<const char *>(params.b_ptr) + static_cast<int64_t>(i) * b_expert_stride;
            gemm_ptrs_[write_idx].c_ptr = c_ptr;
            if (params.use_low_precision) {
                gemm_ptrs_[write_idx].a_scale_ptr = params.a_scale_ptr;
                gemm_ptrs_[write_idx].b_scale_ptr = params.b_scale_ptr;
            }

            ld_a_[write_idx] = get_dim(params.a_shape, -1);
            ld_b_[write_idx] = get_dim(params.b_shape, -1);
            ld_c_[write_idx] = get_dim(params.c_shape, -1);
            rows_a_[write_idx] = get_dim(params.a_shape, -1);
            cols_a_[write_idx] = len;
            rows_b_[write_idx] = get_dim(params.b_shape, -1);
            cols_b_[write_idx] = params.transA ? len : get_dim(params.b_shape, -2);
            rows_c_[write_idx] = get_dim(params.c_shape, -1);
            cols_c_[write_idx] = params.transA ? get_dim(params.c_shape, -2) : len;

            a_ptr += rows_a_[write_idx] * cols_a_[write_idx] * hipblaslt_dtype_bytes(params.a_type);
            if (params.transA) {
                b_ptr += rows_b_[write_idx] * cols_b_[write_idx] * hipblaslt_dtype_bytes(params.b_type);
            }
            c_ptr += rows_c_[write_idx] * cols_c_[write_idx] * hipblaslt_dtype_bytes(params.c_type);
            write_idx++;
        }
    }

    template <typename T> T get_dim(const std::vector<T> &shape, int idx) {
        if (idx < 0) {
            idx += static_cast<int>(shape.size());
        }
        return shape.at(idx);
    }

    // Primary handle for serial dispatch
    hipblasLtHandle_t handle_{};

    // Parallel dispatch infrastructure
    hipblasLtHandle_t par_handles_[kMaxNumStreams]{};
    hipStream_t       par_streams_[kMaxNumStreams]{};
    hipEvent_t        par_events_[kMaxNumStreams]{};
    hipEvent_t        sync_event_{nullptr};

    // Gemm Pointers
    std::vector<GemmPtr> gemm_ptrs_;

    // host-side copy of group_lens (avoids GPU cache coherence issues)
    std::vector<int64_t> group_lens_host_;

    // Leading dimensions
    std::vector<int64_t> ld_a_;
    std::vector<int64_t> ld_b_;
    std::vector<int64_t> ld_c_;

    // rows and cols of matrices
    std::vector<int64_t> rows_a_;
    std::vector<int64_t> cols_a_;
    std::vector<int64_t> rows_b_;
    std::vector<int64_t> cols_b_;
    std::vector<int64_t> rows_c_;
    std::vector<int64_t> cols_c_;

    // Track which (N,K,transA,transB) shapes have been warmed for algo tuning
    using ShapeKey = std::tuple<int64_t, int64_t, bool, bool>;
    std::set<ShapeKey> warmed_shapes_;
};

void hipblaslt_grouped_gemm(const HipblasltGroupedGemmParams &params, const bool pre_sync) {
    static thread_local HipblasltGroupedGemm instance;
    instance.run(params, pre_sync);
}

} // namespace primus_turbo
