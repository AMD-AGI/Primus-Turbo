// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
//
// See LICENSE for license information.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt.h>

#include "primus_turbo/gemm.h"
#include "primus_turbo/grouped_gemm.h"

namespace primus_turbo {

static constexpr size_t kMaxNumStreams = 4;

std::int64_t get_hipblaslt_grouped_gemm_workspace_size() {
    return kMaxNumStreams * get_hipblaslt_workspace_size_in_byte();
}

class HipblasltGroupedGemm {
public:
    HipblasltGroupedGemm() {
        PRIMUS_TURBO_CHECK_HIP(hipEventCreateWithFlags(&sync_event_, hipEventDisableTiming));

        for (size_t i = 0; i < kMaxNumStreams; ++i) {
            PRIMUS_TURBO_CHECK_HIPBLAS(hipblasLtCreate(&handles_[i]));
            PRIMUS_TURBO_CHECK_HIP(
                hipStreamCreateWithPriority(&compute_streams_[i], hipStreamNonBlocking, -1));
            PRIMUS_TURBO_CHECK_HIP(
                hipEventCreateWithFlags(&hipblaslt_events_[i], hipEventDisableTiming));
        }
        workspaces_.resize(kMaxNumStreams);
    }

    ~HipblasltGroupedGemm() {
        if (sync_event_ != nullptr) {
            (void) hipEventDestroy(sync_event_);
        }

        for (size_t i = 0; i < kMaxNumStreams; ++i) {
            if (compute_streams_[i] != nullptr) {
                (void) hipStreamDestroy(compute_streams_[i]);
            }
            if (handles_[i] != nullptr) {
                (void) hipblasLtDestroy(handles_[i]);
            }
            if (hipblaslt_events_[i] != nullptr) {
                (void) hipEventDestroy(hipblaslt_events_[i]);
            }
        }
    }

    void check(const HipblasltGroupedGemmParams &params) {
        PRIMUS_TURBO_CHECK(params.a_shape.size() == 2);
        if (params.transA) {
            // For a * grad_c = grad_b
            // [m, k]^T * [m, n] = [b, k, n]
            PRIMUS_TURBO_CHECK(params.b_shape.size() == 2);
            PRIMUS_TURBO_CHECK(params.c_shape.size() == 3);
            PRIMUS_TURBO_CHECK(params.a_shape[0] == params.b_shape[0]);
            PRIMUS_TURBO_CHECK(params.c_shape[0] == params.group_num);
            PRIMUS_TURBO_CHECK(params.c_shape[1] == params.a_shape[1]);
            PRIMUS_TURBO_CHECK(params.c_shape[2] == params.b_shape[1]);
        } else {
            // For a * b = c and grad_c * b = grad_a
            PRIMUS_TURBO_CHECK(params.b_shape.size() == 3);
            PRIMUS_TURBO_CHECK(params.c_shape.size() == 2);
            PRIMUS_TURBO_CHECK(params.b_shape[0] == params.group_num);
        }
    }

    void run(const HipblasltGroupedGemmParams &params) {
        // Check
        check(params);
        // Compute arguments
        compute_args(params);

        const size_t num_gemms{gemm_ptrs_.size()};
        const size_t num_stream_used{std::min<size_t>(kMaxNumStreams, num_gemms)};

        if (num_gemms > 0) {
            // wait for current stream to finish
            PRIMUS_TURBO_CHECK_HIP(hipEventRecord(sync_event_, params.stream));

            for (size_t s = 0; s < num_stream_used; ++s) {
                PRIMUS_TURBO_CHECK_HIP(hipStreamWaitEvent(compute_streams_[s], sync_event_, 0));
            }

            for (size_t idx = 0; idx < num_gemms; ++idx) {
                const auto stream_idx = idx % kMaxNumStreams;
                auto       stream     = compute_streams_[stream_idx];
                auto       handle     = handles_[stream_idx];
                auto       workspace  = workspaces_[stream_idx];
                // clang-format off
                hipblaslt_gemm_impl(
                    gemm_ptrs_[idx].b_ptr, params.b_type, rows_b_[idx], cols_b_[idx], ld_b_[idx],
                    gemm_ptrs_[idx].b_scale_ptr,
                    params.transB ? HIPBLAS_OP_T : HIPBLAS_OP_N,
                    gemm_ptrs_[idx].a_ptr, params.a_type, rows_a_[idx], cols_a_[idx], ld_a_[idx],
                    gemm_ptrs_[idx].a_scale_ptr,
                    params.transA ? HIPBLAS_OP_T : HIPBLAS_OP_N,
                    gemm_ptrs_[idx].c_ptr, params.c_type, rows_c_[idx], cols_c_[idx], ld_c_[idx],
                    workspace, get_hipblaslt_workspace_size_in_byte(),
                    params.use_low_precision,
                    params.scale_mode,
                    handle,
                    stream
                );
                // clang-format on
            }

            // record events on compute streams
            for (size_t s = 0; s < num_stream_used; ++s) {
                PRIMUS_TURBO_CHECK_HIP(hipEventRecord(hipblaslt_events_[s], compute_streams_[s]));
            }

            // wait for all compute streams to finish
            for (size_t s = 0; s < num_stream_used; ++s) {
                PRIMUS_TURBO_CHECK_HIP(hipStreamWaitEvent(params.stream, hipblaslt_events_[s], 0));
            }
        }
    }

private:
    struct GemmPtr {
        const void *a_ptr       = nullptr;
        const void *a_scale_ptr = nullptr;
        const void *b_ptr       = nullptr;
        const void *b_scale_ptr = nullptr;
        void       *c_ptr       = nullptr;
    };

    void compute_args(const HipblasltGroupedGemmParams &params) {

        int valid_group_num = 0;
        for (size_t i = 0; i < params.group_num; ++i) {
            valid_group_num += params.group_lens_ptr[i] > 0 ? 1 : 0;
        }

        const char *a_ptr = static_cast<const char *>(params.a_ptr);
        const char *b_ptr = static_cast<const char *>(params.b_ptr);
        char       *c_ptr = static_cast<char *>(params.c_ptr);
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
            int64_t len = params.group_lens_ptr[i];

            // For grad_b (transA), if the group len is 0, set the output memory to 0
            // c shape is [group_num, k, n], so each group's c size is k * n
            if (params.transA && len == 0) {
                int64_t c_rows_t = get_dim(params.c_shape, -1);
                int64_t c_cols_t = get_dim(params.c_shape, -2);
                int64_t c_size_t = c_rows_t * c_cols_t * hipblaslt_dtype_bytes(params.c_type);
                PRIMUS_TURBO_CHECK_HIP(hipMemsetAsync(c_ptr, 0, c_size_t, params.stream));
                c_ptr += c_size_t;
            }

            if (len == 0)
                continue;

            // pointers
            gemm_ptrs_[write_idx].a_ptr = a_ptr;
            gemm_ptrs_[write_idx].b_ptr = b_ptr;
            gemm_ptrs_[write_idx].c_ptr = c_ptr;
            if (params.use_low_precision) {
                // TODO(xiaobochen): support variable scale mode
                gemm_ptrs_[write_idx].a_scale_ptr = params.a_scale_ptr;
                gemm_ptrs_[write_idx].b_scale_ptr = params.b_scale_ptr;
            }

            // leading dimension
            ld_a_[write_idx] = get_dim(params.a_shape, -1);
            ld_b_[write_idx] = get_dim(params.b_shape, -1);
            ld_c_[write_idx] = get_dim(params.c_shape, -1);
            // rows and cols of matrices
            rows_a_[write_idx] = get_dim(params.a_shape, -1);
            cols_a_[write_idx] = len;
            rows_b_[write_idx] = get_dim(params.b_shape, -1);
            cols_b_[write_idx] = params.transA ? len : get_dim(params.b_shape, -2);
            rows_c_[write_idx] = get_dim(params.c_shape, -1);
            cols_c_[write_idx] = params.transA ? get_dim(params.c_shape, -2) : len;
            // advance the pointers
            a_ptr += rows_a_[write_idx] * cols_a_[write_idx] * hipblaslt_dtype_bytes(params.a_type);
            b_ptr += rows_b_[write_idx] * cols_b_[write_idx] * hipblaslt_dtype_bytes(params.b_type);
            c_ptr += rows_c_[write_idx] * cols_c_[write_idx] * hipblaslt_dtype_bytes(params.c_type);
            write_idx++;
        }

        char *workspace_ptr = static_cast<char *>(params.workspace);
        for (size_t i = 0; i < kMaxNumStreams; ++i) {
            workspaces_[i] = workspace_ptr + i * get_hipblaslt_workspace_size_in_byte();
        }
    }

    template <typename T> T get_dim(const std::vector<T> &shape, int idx) {
        if (idx < 0) {
            idx += static_cast<int>(shape.size());
        }
        return shape.at(idx);
    }

    // Handles, events, streams, heuristic, epilogue
    hipblasLtHandle_t   handles_[kMaxNumStreams]{};
    hipEvent_t          sync_event_{nullptr};
    hipStream_t         compute_streams_[kMaxNumStreams]{};
    hipEvent_t          hipblaslt_events_[kMaxNumStreams]{};
    hipblasLtEpilogue_t epilogue_{HIPBLASLT_EPILOGUE_DEFAULT};

    // Gemm Pointers
    std::vector<GemmPtr> gemm_ptrs_;

    // workspace
    std::vector<void *> workspaces_;

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
};

void hipblaslt_grouped_gemm(const HipblasltGroupedGemmParams &params) {
    // TODO: This uses a single static instance, which may be risky under multi-threaded usage.
    // If/when needed, add proper thread-safety (e.g., per-thread instances or locking).
    static HipblasltGroupedGemm instance;
    instance.run(params);
}

} // namespace primus_turbo
