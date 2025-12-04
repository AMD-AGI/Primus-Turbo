// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
//
// See LICENSE for license information.
//
// Grouped GEMM hipBLASLt Implementation for PyTorch
// Uses primus_turbo::hipblaslt_gemm_impl for better performance

#include "../extensions.h"
#include "../type_traits.h"
#include "primus_turbo/gemm.h"
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt.h>
#include <mutex>

#define NUM_STREAM 4

#define HIP_CHECK(cmd)                                                                             \
    do {                                                                                           \
        hipError_t status = (cmd);                                                                 \
        if (status != hipSuccess) {                                                                \
            std::fprintf(stderr, "HIP error %d at %s:%d\n", status, __FILE__, __LINE__);           \
            std::abort();                                                                          \
        }                                                                                          \
    } while (0)

#define HIPBLASLT_CHECK(cmd)                                                                       \
    do {                                                                                           \
        hipblasStatus_t status = (cmd);                                                            \
        if (status != HIPBLAS_STATUS_SUCCESS) {                                                    \
            std::fprintf(stderr, "hipBLASLt error %d at %s:%d\n", status, __FILE__, __LINE__);     \
            std::abort();                                                                          \
        }                                                                                          \
    } while (0)

namespace primus_turbo::pytorch {

// Global hipBLASLt state
static hipblasLtHandle_t g_hipblaslt_handle[NUM_STREAM];
static hipStream_t       g_hipblaslt_stream[NUM_STREAM];
static hipEvent_t        g_hipblaslt_event[NUM_STREAM];
static void             *g_hipblaslt_workspace[NUM_STREAM];
static int64_t           g_hipblaslt_workspace_size = 0;
static std::once_flag    g_hipblaslt_once_flag;

void init_hipblaslt_grouped() {
    g_hipblaslt_workspace_size = primus_turbo::get_hipblaslt_workspace_size_in_byte();
    for (int i = 0; i < NUM_STREAM; i++) {
        HIP_CHECK(hipStreamCreateWithFlags(&g_hipblaslt_stream[i], hipStreamNonBlocking));
        HIPBLASLT_CHECK(hipblasLtCreate(&g_hipblaslt_handle[i]));
        HIP_CHECK(hipEventCreate(&g_hipblaslt_event[i]));
        HIP_CHECK(hipMalloc(&g_hipblaslt_workspace[i], g_hipblaslt_workspace_size));
    }
}

inline void ensure_hipblaslt_initialized() {
    std::call_once(g_hipblaslt_once_flag, init_hipblaslt_grouped);
}

inline void hipblaslt_streams_wait_current(hipStream_t current_stream) {
    HIP_CHECK(hipEventRecord(g_hipblaslt_event[0], current_stream));
    for (int s = 0; s < NUM_STREAM; s++) {
        HIP_CHECK(hipStreamWaitEvent(g_hipblaslt_stream[s], g_hipblaslt_event[0]));
    }
}

inline void hipblaslt_current_wait_streams(hipStream_t current_stream) {
    for (int s = 0; s < NUM_STREAM; s++) {
        HIP_CHECK(hipEventRecord(g_hipblaslt_event[s], g_hipblaslt_stream[s]));
    }
    for (int s = 0; s < NUM_STREAM; s++) {
        HIP_CHECK(hipStreamWaitEvent(current_stream, g_hipblaslt_event[s]));
    }
}

static hipDataType get_hipblaslt_dtype(const at::ScalarType t) {
    switch (t) {
    case at::kHalf:
        return HIP_R_16F;
    case at::kFloat:
        return HIP_R_32F;
    case at::kBFloat16:
        return HIP_R_16BF;
    default:
        PRIMUS_TURBO_ERROR("Invalid type for hipblaslt grouped gemm");
    }
}

// Helper function using hipblaslt_gemm_impl
// Row-major: a[m,k], b[b_rows,b_cols], c[m,n]
// trans_b: if true, b is [n,k] (transposed), else b is [k,n]
void HipblasltGemm(hipblasLtHandle_t handle, hipStream_t stream, void *workspace,
                   int64_t workspace_size, void *a, int64_t m, int64_t k, void *b, int64_t b_rows,
                   int64_t b_cols, bool trans_b, void *c, int64_t n, hipDataType dtype,
                   int64_t lda_override = -1, int64_t ldb_override = -1) {
    int64_t lda = (lda_override >= 0) ? lda_override : k;
    int64_t ldb = (ldb_override >= 0) ? ldb_override : b_cols;
    int64_t ldc = n;

    // hipblaslt expects col-major, but we have row-major
    // For row-major C[m,n] = A[m,k] @ B[k,n]:
    //   col-major equivalent: C^T[n,m] = B^T[n,k] @ A^T[k,m]
    // So we call hipblaslt with (B, A) and swap dimensions
    hipblasOperation_t trans_a_op = HIPBLAS_OP_N;
    hipblasOperation_t trans_b_op = trans_b ? HIPBLAS_OP_T : HIPBLAS_OP_N;

    primus_turbo::hipblaslt_gemm_impl(b, dtype, ldb, nullptr, trans_b_op, a, dtype, lda, nullptr,
                                      trans_a_op, c, dtype, ldc, n, m, k, workspace, workspace_size,
                                      false, HIPBLASLT_MATMUL_MATRIX_SCALE_END, handle, stream);
}

void HipblasltGroupedGemmImpl(void *a_ptr, void *b_ptr, void *c_ptr, const int64_t *batch_sizes,
                              int64_t num_experts, int64_t k, int64_t n, int64_t b_rows,
                              int64_t b_cols, bool trans_b, hipStream_t stream, hipDataType dtype) {
    ensure_hipblaslt_initialized();

    size_t elem_size = (dtype == HIP_R_32F) ? 4 : 2;
    char  *a         = reinterpret_cast<char *>(a_ptr);
    char  *b         = reinterpret_cast<char *>(b_ptr);
    char  *c         = reinterpret_cast<char *>(c_ptr);

    hipblaslt_streams_wait_current(stream);

    for (int64_t i = 0; i < num_experts; ++i) {
        int64_t m = batch_sizes[i];

        if (m == 0) {
            b += b_rows * b_cols * elem_size;
            continue;
        }

        int stream_idx = i % NUM_STREAM;

        HipblasltGemm(g_hipblaslt_handle[stream_idx], g_hipblaslt_stream[stream_idx],
                      g_hipblaslt_workspace[stream_idx], g_hipblaslt_workspace_size, a, m, k, b,
                      b_rows, b_cols, trans_b, c, n, dtype);

        a += m * k * elem_size;
        b += b_rows * b_cols * elem_size;
        c += m * n * elem_size;
    }

    hipblaslt_current_wait_streams(stream);
}

// ============================================================================
// PyTorch API
// ============================================================================

at::Tensor grouped_gemm_hipblaslt(at::Tensor &a, at::Tensor &b, at::Tensor &group_lens,
                                  at::Tensor &group_offs, const bool transA, const bool transB) {
    auto out_dtype = a.scalar_type();

    // Check
    PRIMUS_TURBO_CHECK(is_16bit_floating_point_dtype(a.scalar_type()),
                       "grouped_gemm_hipblaslt only supports float16 and bfloat16");
    PRIMUS_TURBO_CHECK(is_16bit_floating_point_dtype(b.scalar_type()),
                       "grouped_gemm_hipblaslt only supports float16 and bfloat16");
    PRIMUS_TURBO_CHECK(group_lens.scalar_type() == at::kLong);
    PRIMUS_TURBO_CHECK(group_offs.scalar_type() == at::kLong);
    PRIMUS_TURBO_CHECK(a.scalar_type() == b.scalar_type(), "a and b dtype mismatch");
    PRIMUS_TURBO_CHECK(!transA, "grouped_gemm_hipblaslt does not support transA=True");

    // Get dimensions
    const int64_t bs = b.size(0);
    const int64_t m  = a.size(0);
    const int64_t n  = transB ? b.size(1) : b.size(2);
    const int64_t k  = a.size(1);

    // Calculate b dimensions
    const int64_t b_rows = transB ? n : k;
    const int64_t b_cols = transB ? k : n;

    // Create output tensor
    at::Tensor c = at::empty({m, n}, at::dtype(out_dtype).device(at::kCUDA));

    // Get stream and dtype
    auto        stream = at::cuda::getCurrentCUDAStream();
    hipDataType dtype  = get_hipblaslt_dtype(a.scalar_type());

    // Call implementation
    HipblasltGroupedGemmImpl(a.data_ptr(), b.data_ptr(), c.data_ptr(),
                             group_lens.data_ptr<int64_t>(), bs, k, n, b_rows, b_cols, transB,
                             stream, dtype);

    return c;
}

at::Tensor grouped_gemm_variable_k_hipblaslt(at::Tensor &a, at::Tensor &b, at::Tensor &group_lens,
                                             at::Tensor &group_offs, const bool transA,
                                             const bool transB) {
    auto out_dtype = a.scalar_type();

    // Check
    PRIMUS_TURBO_CHECK(is_16bit_floating_point_dtype(a.scalar_type()),
                       "grouped_gemm_variable_k_hipblaslt only supports float16 and bfloat16");
    PRIMUS_TURBO_CHECK(is_16bit_floating_point_dtype(b.scalar_type()),
                       "grouped_gemm_variable_k_hipblaslt only supports float16 and bfloat16");
    PRIMUS_TURBO_CHECK(group_lens.scalar_type() == at::kLong);
    PRIMUS_TURBO_CHECK(group_offs.scalar_type() == at::kLong);
    PRIMUS_TURBO_CHECK(a.scalar_type() == b.scalar_type(), "a and b dtype mismatch");
    PRIMUS_TURBO_CHECK(transA && !transB,
                       "grouped_gemm_variable_k_hipblaslt only supports transA=True, transB=False");

    ensure_hipblaslt_initialized();

    // Get dimensions
    // a: [M, total_tokens], b: [N, total_tokens]
    // output: [num_experts, M, N]
    const int64_t num_experts  = group_lens.numel();
    const int64_t M            = a.size(0);
    const int64_t total_tokens = a.size(1);
    const int64_t N            = b.size(0);

    // Create output tensor
    at::Tensor c = at::empty({num_experts, M, N}, at::dtype(out_dtype).device(at::kCUDA));

    // Get stream and dtype
    auto        stream = at::cuda::getCurrentCUDAStream();
    hipDataType dtype  = get_hipblaslt_dtype(a.scalar_type());

    size_t elem_size = 2; // float16 and bfloat16 are 2 bytes
    char  *a_ptr     = reinterpret_cast<char *>(a.data_ptr());
    char  *b_ptr     = reinterpret_cast<char *>(b.data_ptr());
    char  *c_ptr     = reinterpret_cast<char *>(c.data_ptr());

    const int64_t *batch_sizes_ptr = group_lens.data_ptr<int64_t>();

    hipblaslt_streams_wait_current(stream);

    int64_t start = 0;
    for (int64_t i = 0; i < num_experts; ++i) {
        int64_t k_i = batch_sizes_ptr[i];

        if (k_i == 0) {
            size_t output_size = M * N * elem_size;
            HIP_CHECK(hipMemsetAsync(c_ptr + i * output_size, 0, output_size, stream));
            start += k_i;
            continue;
        }

        int stream_idx = i % NUM_STREAM;

        // a_sub = a[:, start:start+k_i] is [M, k_i]
        // b_sub = b[:, start:start+k_i] is [N, k_i]
        // c[i] = a_sub @ b_sub.T = [M, k_i] @ [k_i, N] = [M, N]
        HipblasltGemm(g_hipblaslt_handle[stream_idx], g_hipblaslt_stream[stream_idx],
                      g_hipblaslt_workspace[stream_idx], g_hipblaslt_workspace_size,
                      a_ptr + start * elem_size, M, k_i, b_ptr + start * elem_size, N, k_i, true,
                      c_ptr + i * M * N * elem_size, N, dtype,
                      total_tokens,  // lda: physical row stride for a
                      total_tokens); // ldb: physical row stride for b

        start += k_i;
    }

    hipblaslt_current_wait_streams(stream);

    return c;
}

} // namespace primus_turbo::pytorch
