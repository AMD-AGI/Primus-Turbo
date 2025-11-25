// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
//
// See LICENSE for license information.
//
// Grouped GEMM hipBLASLt Implementation
// Simpler alternative to CK-based implementation with full JIT support

#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include "../extensions.h"

#define NUM_STREAM 4

#define HIP_CHECK(cmd)                                                        \
  do {                                                                        \
    hipError_t status = (cmd);                                                \
    if (status != hipSuccess) {                                               \
      std::fprintf(stderr, "HIP error %d at %s:%d\n", status, __FILE__,       \
                   __LINE__);                                                 \
      std::abort();                                                           \
    }                                                                         \
  } while (0)

#define HIPBLAS_CHECK(cmd)                                                    \
  do {                                                                        \
    hipblasStatus_t status = (cmd);                                           \
    if (status != HIPBLAS_STATUS_SUCCESS) {                                   \
      std::fprintf(stderr, "hipBLAS error %d at %s:%d\n", status, __FILE__,   \
                   __LINE__);                                                 \
      std::abort();                                                           \
    }                                                                         \
  } while (0)

namespace primus_turbo::jax {

// Global hipBLAS state for hipBLASLt backend
static hipblasHandle_t g_hipblaslt_handle[NUM_STREAM];
static hipStream_t g_hipblaslt_stream[NUM_STREAM];
static hipEvent_t g_hipblaslt_event[NUM_STREAM];
static std::once_flag g_hipblaslt_once_flag;

void init_hipblaslt() {
  for (int i = 0; i < NUM_STREAM; i++) {
    HIP_CHECK(hipStreamCreateWithFlags(&g_hipblaslt_stream[i], hipStreamNonBlocking));
    HIPBLAS_CHECK(hipblasCreate(&g_hipblaslt_handle[i]));
    HIPBLAS_CHECK(hipblasSetStream(g_hipblaslt_handle[i], g_hipblaslt_stream[i]));
    HIP_CHECK(hipEventCreate(&g_hipblaslt_event[i]));
  }
}

inline void ensure_hipblaslt_initialized() {
  std::call_once(g_hipblaslt_once_flag, init_hipblaslt);
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

void HipblasltGemm(hipblasHandle_t handle, void *a, int64_t a_rows,
                   int64_t a_cols, bool trans_a, void *b, int64_t b_rows,
                   int64_t b_cols, bool trans_b, void *c, int64_t c_rows,
                   int64_t c_cols, hipDataType dtype = HIP_R_16BF,
                   int64_t lda_override = -1, int64_t ldb_override = -1) {
  int m = trans_b ? b_rows : b_cols;
  int k = trans_b ? b_cols : b_rows;
  int n = trans_a ? a_cols : a_rows;

  // Leading dimension is the physical stride in memory (row-major)
  int lda = (lda_override >= 0) ? lda_override : a_cols;
  int ldb = (ldb_override >= 0) ? ldb_override : b_cols;
  
  hipblasOperation_t transpose_a = trans_a ? HIPBLAS_OP_T : HIPBLAS_OP_N;
  hipblasOperation_t transpose_b = trans_b ? HIPBLAS_OP_T : HIPBLAS_OP_N;

  float alpha = 1.0f, beta = 0.0f;
  HIPBLAS_CHECK(hipblasGemmEx(handle, transpose_b, transpose_a, m, n, k,
                              &alpha, b, dtype, ldb, a, dtype, lda,
                              &beta, c, dtype, c_cols,
                              HIPBLAS_COMPUTE_32F, HIPBLAS_GEMM_DEFAULT));
}

void HipblasltGroupedGemm(void *a_ptr, void *b_ptr, void *c_ptr,
                          const int64_t *batch_sizes, int64_t num_experts,
                          int64_t k, int64_t n, 
                          int64_t b_rows, int64_t b_cols, bool trans_b,
                          hipStream_t stream, hipDataType dtype = HIP_R_16BF) {
  ensure_hipblaslt_initialized();

  size_t elem_size = 2;  // Both float16 and bfloat16 are 2 bytes
  char *a = reinterpret_cast<char*>(a_ptr);
  char *b = reinterpret_cast<char*>(b_ptr);
  char *c = reinterpret_cast<char*>(c_ptr);

  hipblaslt_streams_wait_current(stream);

  for (int64_t i = 0; i < num_experts; ++i) {
    int64_t m = batch_sizes[i];
    
    if (m == 0) {
      b += b_rows * b_cols * elem_size;
      continue;
    }
    
    int stream_idx = i % NUM_STREAM;
    
    HipblasltGemm(g_hipblaslt_handle[stream_idx], 
                  a, m, k, false,
                  b, b_rows, b_cols, trans_b, 
                  c, m, n,
                  dtype);
    
    a += m * k * elem_size;
    b += b_rows * b_cols * elem_size;
    c += m * n * elem_size;
  }

  hipblaslt_current_wait_streams(stream);
}

// ============================================================================
// XLA FFI Handlers for Hipblaslt Backend
// ============================================================================

ffi::Error GroupedGemmHipblasltFFI(
    hipStream_t stream,
    ffi::AnyBuffer a,
    ffi::AnyBuffer b,
    ffi::AnyBuffer batch_sizes,
    ffi::AnyBuffer group_offs,  // Added to match signature
    ffi::Result<ffi::AnyBuffer> c,
    bool trans_a,
    bool trans_b,
    int64_t num_cu
) {
    // Check input types
    if (a.element_type() != b.element_type()) {
        return ffi::Error(ffi::ErrorCode::kInvalidArgument, "a and b dtype mismatch");
    }
    
    if (a.element_type() != ffi::BF16 && a.element_type() != ffi::F16) {
        return ffi::Error(ffi::ErrorCode::kInvalidArgument, "Only bfloat16 and float16 supported");
    }
    
    if (batch_sizes.element_type() != ffi::S64 || group_offs.element_type() != ffi::S64) {
        return ffi::Error(ffi::ErrorCode::kInvalidArgument, "batch_sizes and group_offs must be int64");
    }
    
    // Get dimensions
    const int32_t num_experts = static_cast<int32_t>(b.dimensions()[0]);
    const int64_t total_tokens = static_cast<int64_t>(a.dimensions()[0]);
    const int32_t K = static_cast<int32_t>(a.dimensions()[1]);
    const int32_t N = trans_b ? b.dimensions()[1] : b.dimensions()[2];
    
    // Calculate b dimensions
    const int64_t b_rows = trans_b ? N : K;
    const int64_t b_cols = trans_b ? K : N;

    // Determine datatype
    hipDataType dtype = (a.element_type() == ffi::F16) ? HIP_R_16F : HIP_R_16BF;

    // Call grouped GEMM
    HipblasltGroupedGemm(
        const_cast<void*>(a.untyped_data()),
        const_cast<void*>(b.untyped_data()),
        c->untyped_data(),
        batch_sizes.typed_data<int64_t>(),
        num_experts,
        K, N, b_rows, b_cols, trans_b,
        stream, dtype
    );
    
    return ffi::Error::Success();
}

// Variable K Grouped GEMM FFI Handler (Hipblaslt Backend)
ffi::Error GroupedGemmVariableKHipblasltFFI(
    hipStream_t stream,
    ffi::AnyBuffer a,              // [m, total_k]
    ffi::AnyBuffer b,              // [n, total_k]
    ffi::AnyBuffer batch_sizes,    // [num_experts]
    ffi::AnyBuffer group_offs,     // [num_experts + 1] - Added to match signature
    ffi::Result<ffi::AnyBuffer> c, // [num_experts, m, n]
    bool trans_a,                  // must be True
    bool trans_b,                  // must be False
    int64_t num_cu
) {
    // Check constraint: only supports trans_a=True, trans_b=False
    if (!trans_a || trans_b) {
        return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                          "variable_k only supports trans_a=True, trans_b=False");
    }
    
    // Check input types
    if (a.element_type() != b.element_type()) {
        return ffi::Error(ffi::ErrorCode::kInvalidArgument, "a and b dtype mismatch");
    }
    
    if (a.element_type() != ffi::BF16 && a.element_type() != ffi::F16) {
        return ffi::Error(ffi::ErrorCode::kInvalidArgument, "Only bfloat16 and float16 supported");
    }
    
    if (batch_sizes.element_type() != ffi::S64 || group_offs.element_type() != ffi::S64) {
        return ffi::Error(ffi::ErrorCode::kInvalidArgument, "batch_sizes and group_offs must be int64");
    }
    
    // Get dimensions
    // Python passes transposed inputs:
    // a: [M, total_tokens], b: [N, total_tokens]
    // output: [num_experts, M, N]
    const int32_t num_experts = static_cast<int32_t>(batch_sizes.element_count());
    const int32_t M = static_cast<int32_t>(a.dimensions()[0]);
    const int32_t total_tokens = static_cast<int32_t>(a.dimensions()[1]);
    const int32_t N = static_cast<int32_t>(b.dimensions()[0]);
    
    ensure_hipblaslt_initialized();
    
    // Determine datatype
    hipDataType dtype = (a.element_type() == ffi::F16) ? HIP_R_16F : HIP_R_16BF;
    
    size_t elem_size = 2;  // Both float16 and bfloat16 are 2 bytes
    char *a_ptr = reinterpret_cast<char*>(const_cast<void*>(a.untyped_data()));
    char *b_ptr = reinterpret_cast<char*>(const_cast<void*>(b.untyped_data()));
    char *c_ptr = reinterpret_cast<char*>(c->untyped_data());
    
    const int64_t *batch_sizes_ptr = batch_sizes.typed_data<int64_t>();
    
    hipblaslt_streams_wait_current(stream);
    
    int64_t start = 0;
    for (int32_t i = 0; i < num_experts; ++i) {
        int64_t k_i = batch_sizes_ptr[i];
        
        if (k_i == 0) {
            size_t output_size = M * N * elem_size;
            HIP_CHECK(hipMemsetAsync(c_ptr + i * output_size, 0, output_size, stream));
            start += k_i;
            continue;
        }
        
        int stream_idx = i % NUM_STREAM;
        
        // Row-major memory layout with TRANSPOSED inputs:
        // a: [M, total_tokens], element a[row, col] at offset (row * total_tokens + col) * elem_size
        // b: [N, total_tokens], element b[row, col] at offset (row * total_tokens + col) * elem_size
        //
        // To access COLUMN slice [:, start:start+k_i]:
        // - First element a[0, start] is at offset: start * elem_size
        // - First element b[0, start] is at offset: start * elem_size
        // - Leading dimension (stride between rows): total_tokens
        //
        // We want to compute: c[i] = a_sub @ b_sub.T
        // where a_sub = a[:, start:start+k_i] is [M, k_i]
        //       b_sub = b[:, start:start+k_i] is [N, k_i]
        //       c[i] = a_sub @ b_sub.T = [M, k_i] @ [k_i, N] = [M, N]
        HipblasltGemm(
            g_hipblaslt_handle[stream_idx],
            a_ptr + start * elem_size,       // Pointer to a[0, start]
            M, k_i, false,                   // a_sub: [M, k_i], no transpose
            b_ptr + start * elem_size,       // Pointer to b[0, start]
            N, k_i, true,                    // b_sub: [N, k_i], transpose it to [k_i, N]
            c_ptr + i * M * N * elem_size,   // Output c[i]: [M, N]
            M, N,
            dtype,         // datatype (float16 or bfloat16)
            total_tokens,  // lda: physical row stride in memory for a
            total_tokens   // ldb: physical row stride in memory for b
        );
        
        start += k_i;
    }
    
    hipblaslt_current_wait_streams(stream);
    
    return ffi::Error::Success();
}

// Register Hipblaslt FFI Handlers
XLA_FFI_DEFINE_HANDLER_SYMBOL(
    GroupedGemmHipblasltHandler, GroupedGemmHipblasltFFI,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<hipStream_t>>()
        .Arg<ffi::AnyBuffer>()  // a
        .Arg<ffi::AnyBuffer>()  // b
        .Arg<ffi::AnyBuffer>()  // batch_sizes
        .Arg<ffi::AnyBuffer>()  // group_offs
        .Ret<ffi::AnyBuffer>()  // c
        .Attr<bool>("transA")
        .Attr<bool>("transB")
        .Attr<int64_t>("num_cu")
);

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    GroupedGemmVariableKHipblasltHandler, GroupedGemmVariableKHipblasltFFI,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<hipStream_t>>()
        .Arg<ffi::AnyBuffer>()  // a
        .Arg<ffi::AnyBuffer>()  // b
        .Arg<ffi::AnyBuffer>()  // batch_sizes
        .Arg<ffi::AnyBuffer>()  // group_offs
        .Ret<ffi::AnyBuffer>()  // c
        .Attr<bool>("transA")
        .Attr<bool>("transB")
        .Attr<int64_t>("num_cu")
);

} // namespace primus_turbo::jax

