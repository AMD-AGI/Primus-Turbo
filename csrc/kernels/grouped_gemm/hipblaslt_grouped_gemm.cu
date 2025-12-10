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

std::int64_t get_hipblaslt_grouped_gemm_workspace_size(const int64_t group_num) {
    return group_num * get_hipblaslt_workspace_size_in_byte();
}

template <typename ADataType, typename BDataType, typename CDataType, typename AccDataType = float>
class HipblasltGroupedGemm {
public:
    HipblasltGroupedGemm() {
        printf("Init HipblasltGroupedGemm\n");
        PRIMUS_TURBO_CHECK_HIP(hipEventCreate(&sync_event_));

        handles_[0]         = nullptr;
        compute_streams_[0] = nullptr;
        for (size_t i = 0; i < kDefaultInitKmaxNumStream; ++i) {
            if (i > 0) {
                PRIMUS_TURBO_CHECK_HIPBLAS(hipblasLtCreate(&handles_[i]));
                PRIMUS_TURBO_CHECK_HIP(
                    hipStreamCreateWithPriority(&compute_streams_[i], hipStreamNonBlocking, -1));
                // PRIMUS_TURBO_CHECK_HIPBLAS(hipblasSetStream(handles_[i], compute_streams_[i]));
            }
            PRIMUS_TURBO_CHECK_HIP(hipEventCreate(&hipblaslt_events_[i]));
        }
    }

    ~HipblasltGroupedGemm() {
        if (sync_event_ != nullptr) {
            (void) hipEventDestroy(sync_event_);
        }

        for (size_t i = 0; i < kDefaultInitKmaxNumStream; ++i) {
            if (i > 0) {
                (void) hipStreamDestroy(compute_streams_[i]);
                (void) hipblasLtDestroy(handles_[i]);
            }
            (void) hipEventDestroy(hipblaslt_events_[i]);
        }
        printf("Destroy HipblasltGroupedGemm\n");
    }

    void check(const HipblasltGroupedGemmParams<ADataType, BDataType, CDataType> &params) {
        printf("HipblasltGroupedGemm::check\n");

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

    void run(const HipblasltGroupedGemmParams<ADataType, BDataType, CDataType> &params,
             hipblasLtHandle_t                                                  handle) {
        check(params);
        compute_args(params);
        printf("HipblasltGroupedGemm::run\n");
    }

private:
    struct GemmPtr {
        const ADataType *a_ptr;
        const BDataType *b_ptr;
        CDataType       *c_ptr;
    };

    void compute_args(const HipblasltGroupedGemmParams<ADataType, BDataType, CDataType> &params) {

        int valid_group_num = 0;
        for (size_t i = 0; i < params.group_num; ++i) {
            valid_group_num += params.group_lens_ptr[i] > 0 ? 1 : 0;
        }

        const ADataType *a_ptr = params.a_ptr;
        const BDataType *b_ptr = params.b_ptr;
        CDataType       *c_ptr = params.c_ptr;
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
            if (len == 0)
                continue;

            // pointers
            gemm_ptrs_[write_idx].a_ptr = a_ptr;
            gemm_ptrs_[write_idx].b_ptr = b_ptr;
            gemm_ptrs_[write_idx].c_ptr = c_ptr;
            // leading dimension
            ld_a_[write_idx] = get_dim(params.a_shape, -1);
            ld_b_[write_idx] = get_dim(params.b_shape, -1);
            ld_c_[write_idx] = get_dim(params.c_shape, -1);
            // rows and cols of matrices
            rows_a_[write_idx] = len;
            cols_a_[write_idx] = get_dim(params.a_shape, -1);
            rows_b_[write_idx] = params.transA ? len : get_dim(params.b_shape, -2);
            cols_b_[write_idx] = get_dim(params.b_shape, -1);
            rows_c_[write_idx] = params.transA ? get_dim(params.c_shape, -2) : len;
            cols_c_[write_idx] = get_dim(params.c_shape, -1);
            // advance the pointers
            a_ptr += rows_a_[write_idx] * cols_a_[write_idx];
            b_ptr += rows_b_[write_idx] * cols_b_[write_idx];
            c_ptr += rows_c_[write_idx] * cols_c_[write_idx];
            write_idx++;
        }

        // For grad_b, if the group len is 0, set the local memory to 0
        for (size_t i = 0; i < params.group_num; ++i) {
            if (params.transA && params.group_lens_ptr[i] == 0) {
                PRIMUS_TURBO_CHECK_HIP(hipMemsetAsync(gemm_ptrs_[i].c_ptr, 0,
                                                      rows_c_[i] * cols_c_[i] * sizeof(CDataType),
                                                      params.stream));
            }
        }
    }

    template <typename T> T get_dim(const std::vector<T> &shape, int idx) {
        if (idx < 0) {
            idx += static_cast<int>(shape.size());
        }
        return shape.at(idx);
    }

    //
    static constexpr size_t kDefaultInitKmaxNumStream{4};

    // Handles, events, streams, heuristic, epilogue
    hipblasLtHandle_t   handles_[kDefaultInitKmaxNumStream];
    hipEvent_t          sync_event_{nullptr};
    hipStream_t         compute_streams_[kDefaultInitKmaxNumStream];
    hipEvent_t          hipblaslt_events_[kDefaultInitKmaxNumStream];
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

template <typename ADataType, typename BDataType, typename CDataType, typename AccDataType>
void hipblaslt_grouped_gemm(
    const HipblasltGroupedGemmParams<ADataType, BDataType, CDataType> &params,
    hipblasLtHandle_t                                                  handle) {
    static HipblasltGroupedGemm<ADataType, BDataType, CDataType, AccDataType> instance;
    instance.run(params, handle);
    printf("HHHH hipblaslt_grouped_gemm\n");
}

template void hipblaslt_grouped_gemm<dtype::float16, dtype::float16, dtype::float16>(
    const HipblasltGroupedGemmParams<dtype::float16, dtype::float16, dtype::float16> &params,
    hipblasLtHandle_t                                                                 handle);

} // namespace primus_turbo
