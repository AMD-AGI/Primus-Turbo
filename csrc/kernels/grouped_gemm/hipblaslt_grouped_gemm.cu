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

std::int64_t get_hipblaslt_grouped_gemm_workspace_size() {
    // Single-stream dispatch: only one workspace needed.
    return get_hipblaslt_workspace_size_in_byte();
}

class HipblasltGroupedGemm {
public:
    HipblasltGroupedGemm() {
        PRIMUS_TURBO_CHECK_HIPBLAS(hipblasLtCreate(&handle_));
    }

    ~HipblasltGroupedGemm() {
        if (handle_) {
            (void)hipblasLtDestroy(handle_);
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

    void run(const HipblasltGroupedGemmParams &params, const bool pre_sync) {
        // Always synchronize before compute_args: group_lens_ptr is a device
        // pointer read directly from the CPU.  Without this sync, writes made
        // by upstream GPU kernels (e.g. the MoE router) may not be visible to
        // the host, causing stale-zero reads and silent no-ops.
        PRIMUS_TURBO_CHECK_HIP(hipStreamSynchronize(params.stream));
        (void) pre_sync; // kept for API compatibility

        // Check
        check(params);
        // Compute arguments
        compute_args(params);

        const size_t num_gemms{gemm_ptrs_.size()};

        // Dispatch all expert GEMMs sequentially on the caller's stream.
        // This avoids GPU resource contention that occurs when 4 concurrent
        // streams each launch large GEMMs competing for CUs/L2/register file.
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

private:
    struct GemmPtr {
        const void *a_ptr       = nullptr;
        const void *a_scale_ptr = nullptr;
        const void *b_ptr       = nullptr;
        const void *b_scale_ptr = nullptr;
        void       *c_ptr       = nullptr;
    };

    void compute_args(const HipblasltGroupedGemmParams &params) {

        // Copy group_lens from device to host via hipMemcpy so the ROCm runtime
        // handles GPU cache flushing / coherence correctly.  Direct CPU
        // dereference of a device pointer can read stale L2-cached data even
        // after hipStreamSynchronize, causing valid_group_num=0 and skipping the
        // algo-cache warm-up, which then triggers a 500+ ms heuristic search on
        // the first *timed* iteration and produces catastrophically low TFLOPS.
        group_lens_host_.resize(params.group_num);
        PRIMUS_TURBO_CHECK_HIP(hipMemcpy(group_lens_host_.data(), params.group_lens_ptr,
                                         params.group_num * sizeof(int64_t),
                                         hipMemcpyDeviceToHost));

        int valid_group_num = 0;
        for (size_t i = 0; i < params.group_num; ++i) {
            valid_group_num += group_lens_host_[i] > 0 ? 1 : 0;
        }

        const char *a_ptr = static_cast<const char *>(params.a_ptr);
        const char *b_ptr = static_cast<const char *>(params.b_ptr);  // sequential ptr (wgrad only)
        char       *c_ptr = static_cast<char *>(params.c_ptr);

        // For fwd/dgrad (transA=False): b is [G, K, N] — each expert has a fixed K×N weight block.
        // Use absolute per-expert offset (b_base + i * b_expert_stride) so that skipping cold
        // experts (len==0) does not shift the pointer and corrupt subsequent hot experts.
        // For wgrad (transA=True): b is flat [M_total, OUT_N] — advance sequentially per hot group.
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
            // fwd/dgrad: absolute per-expert offset avoids desync on cold experts
            // wgrad:     sequential advance through the flat b tensor
            gemm_ptrs_[write_idx].b_ptr = params.transA
                ? b_ptr
                : static_cast<const char *>(params.b_ptr) + static_cast<int64_t>(i) * b_expert_stride;
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
            if (params.transA) {
                // wgrad only: advance b_ptr through the flat tensor
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

    // Single handle (algo caching is inside hipblaslt_gemm_impl)
    hipblasLtHandle_t handle_{};

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
};

void hipblaslt_grouped_gemm(const HipblasltGroupedGemmParams &params, const bool pre_sync) {
    static thread_local HipblasltGroupedGemm instance;
    instance.run(params, pre_sync);
}

} // namespace primus_turbo
