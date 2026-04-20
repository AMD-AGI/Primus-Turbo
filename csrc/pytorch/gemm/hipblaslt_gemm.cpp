// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
//
// See LICENSE for license information.

#include "primus_turbo/gemm.h"

#include "../extensions.h"
#include "../type_traits.h"

namespace primus_turbo::pytorch {

static at::Tensor hipblaslt_gemm_core(at::Tensor A, at::Tensor B, const at::ScalarType out_dtype,
                                      bool transA, bool transB, bool transC, int64_t algo_index = 0,
                                      int *out_algo_count = nullptr) {
    PRIMUS_TURBO_CHECK(is_floating_point_dtype(A.scalar_type()));
    PRIMUS_TURBO_CHECK(is_floating_point_dtype(B.scalar_type()));
    PRIMUS_TURBO_CHECK(A.scalar_type() == B.scalar_type(), "A and B dtype mismatch");
    PRIMUS_TURBO_CHECK(is_floating_point_dtype(out_dtype));

    // contiguous check
    PRIMUS_TURBO_CHECK(A.is_contiguous(), "A must be contiguous");
    PRIMUS_TURBO_CHECK(B.is_contiguous(), "B must be contiguous");

    // shape check
    PRIMUS_TURBO_CHECK(A.dim() == 2 && B.dim() == 2, "A, B must be 2D tensors");

    if (transC) {
        std::swap(A, B);
        std::tie(transA, transB) = std::make_tuple(!transB, !transA);
    }

    const int64_t m = transA ? A.size(1) : A.size(0);
    const int64_t k = transA ? A.size(0) : A.size(1);
    const int64_t n = transB ? B.size(0) : B.size(1);

    // rows/cols
    const int64_t rows_a = transA ? m : k;
    const int64_t cols_a = transA ? k : m;
    const int64_t rows_b = transB ? k : n;
    const int64_t cols_b = transB ? n : k;
    const int64_t rows_d = n;
    const int64_t cols_d = m;

    // NOTE: The leading dimension is col-major.
    int64_t lda, ldb, ldd;
    if (!transA && transB) { // NT
        PRIMUS_TURBO_CHECK(A.size(1) == B.size(1), "tensor size mismatch");
        lda = k;
        ldb = k;
        ldd = n;
    } else if (!transA && !transB) { // NN
        PRIMUS_TURBO_CHECK(A.size(1) == B.size(0), "tensor size mismatch");
        lda = k;
        ldb = n;
        ldd = n;
    } else if (transA && !transB) { // TN
        PRIMUS_TURBO_CHECK(A.size(0) == B.size(0), "tensor size mismatch");
        lda = m;
        ldb = n;
        ldd = n;
    } else {
        PRIMUS_TURBO_ERROR("Not support layout.");
    }

    auto               handle            = at::cuda::getCurrentCUDABlasLtHandle();
    hipblasOperation_t trans_operation_A = transA ? HIPBLAS_OP_T : HIPBLAS_OP_N;
    hipblasOperation_t trans_operation_B = transB ? HIPBLAS_OP_T : HIPBLAS_OP_N;
    const hipDataType  A_type            = get_hipblaslt_dtype(A.scalar_type());
    const hipDataType  B_type            = get_hipblaslt_dtype(B.scalar_type());
    const hipDataType  D_type            = get_hipblaslt_dtype(out_dtype);
    const int64_t      workspace_size    = get_hipblaslt_workspace_size_in_byte();

    if (out_algo_count) {
        // clang-format off
        *out_algo_count = hipblaslt_gemm_get_algo_count(
            static_cast<const void *>(B.data_ptr()), B_type,
            rows_b, cols_b, ldb, nullptr, trans_operation_B,
            static_cast<const void *>(A.data_ptr()), A_type,
            rows_a, cols_a, lda, nullptr, trans_operation_A,
            D_type, rows_d, cols_d, ldd,
            workspace_size, false, HIPBLASLT_MATMUL_MATRIX_SCALE_END, handle);
        // clang-format on
        return {};
    }

    at::Tensor        C      = at::empty({m, n}, torch::dtype(out_dtype).device(at::kCUDA));
    auto              stream = at::hip::getCurrentHIPStream();
    const hipDataType C_type = get_hipblaslt_dtype(C.scalar_type());
    at::Tensor workspace = at::empty({workspace_size}, torch::dtype(at::kByte).device(at::kCUDA));

    // clang-format off
    // NOTE: hipblaslt expects tensor in col-major but torch Tensor is in row-major.
    // Swapping A&B that are essentially computing C^T = B^T @ A^T.
    hipblaslt_gemm_impl(
        static_cast<const void *>(B.data_ptr()), B_type,
        rows_b, cols_b, ldb,
        nullptr,
        trans_operation_B,
        static_cast<const void *>(A.data_ptr()), A_type,
        rows_a, cols_a, lda,
        nullptr,
        trans_operation_A,
        static_cast<void *>(C.data_ptr()), C_type,
        rows_d, cols_d, ldd,
        static_cast<void *>(workspace.data_ptr()), workspace_size,
        false,
        HIPBLASLT_MATMUL_MATRIX_SCALE_END,
        handle, stream, static_cast<int>(algo_index));
    // clang-format on

    return C;
}

static at::Tensor hipblaslt_gemm_fp8_core(at::Tensor A, at::Tensor scaleA_inv, at::Tensor B,
                                          at::Tensor scaleB_inv, const at::ScalarType out_dtype,
                                          bool transA, bool transB, bool transC,
                                          const std::string &granularity, int64_t algo_index = 0,
                                          int *out_algo_count = nullptr) {
    const bool use_fp8 = is_8bit_floating_point_dtype(A.scalar_type()) &&
                         is_8bit_floating_point_dtype(B.scalar_type());

    PRIMUS_TURBO_CHECK(use_fp8, "A and B must be FP8 Tensor.");

    // scale mode
    hipblasLtMatmulMatrixScale_t scale_mode = HIPBLASLT_MATMUL_MATRIX_SCALE_END;
    if (granularity == "TENSORWISE") {
        scale_mode = HIPBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F;
    } else if (granularity == "MX_BLOCKWISE") {
        scale_mode = HIPBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0;
    } else {
        PRIMUS_TURBO_ERROR("Invalid granularity.");
    }

    PRIMUS_TURBO_CHECK(is_8bit_floating_point_dtype(A.scalar_type()));
    PRIMUS_TURBO_CHECK(is_8bit_floating_point_dtype(B.scalar_type()));
    PRIMUS_TURBO_CHECK(is_16bit_floating_point_dtype(out_dtype));
    if (granularity != "MX_BLOCKWISE") {
        PRIMUS_TURBO_CHECK(scaleA_inv.scalar_type() == at::kFloat);
        PRIMUS_TURBO_CHECK(scaleB_inv.scalar_type() == at::kFloat);
    } else {
        // scaling factor is e8m0 format.
        PRIMUS_TURBO_CHECK(scaleA_inv.scalar_type() == at::kFloat8_e8m0fnu);
        PRIMUS_TURBO_CHECK(scaleB_inv.scalar_type() == at::kFloat8_e8m0fnu);
    }

    // contiguous check
    PRIMUS_TURBO_CHECK(A.is_contiguous(), "A must be contiguous");
    PRIMUS_TURBO_CHECK(B.is_contiguous(), "B must be contiguous");

    // shape check
    PRIMUS_TURBO_CHECK(A.dim() == 2 && B.dim() == 2, "A, B must be 2D tensors");

    if (transC) {
        std::swap(A, B);
        std::swap(scaleA_inv, scaleB_inv);
        std::tie(transA, transB) = std::make_tuple(!transB, !transA);
    }

    const int64_t m = transA ? A.size(1) : A.size(0);
    const int64_t k = transA ? A.size(0) : A.size(1);
    const int64_t n = transB ? B.size(0) : B.size(1);

    // rows/cols
    const int64_t rows_a = transA ? m : k;
    const int64_t cols_a = transA ? k : m;
    const int64_t rows_b = transB ? k : n;
    const int64_t cols_b = transB ? n : k;
    const int64_t rows_d = n;
    const int64_t cols_d = m;

    // NOTE: The leading dimension is col-major.
    int64_t lda, ldb, ldd;
    if (!transA && transB) { // NT
        PRIMUS_TURBO_CHECK(A.size(1) == B.size(1), "tensor size mismatch");
        lda = k;
        ldb = k;
        ldd = n;
    } else if (!transA && !transB) { // NN
        PRIMUS_TURBO_CHECK(A.size(1) == B.size(0), "tensor size mismatch");
        lda = k;
        ldb = n;
        ldd = n;
    } else if (transA && !transB) { // TN
        PRIMUS_TURBO_CHECK(A.size(0) == B.size(0), "tensor size mismatch");
        lda = m;
        ldb = n;
        ldd = n;
    } else {
        PRIMUS_TURBO_ERROR("Not support layout.");
    }

    // MXFP8 extra check, ref:
    // https://rocm.docs.amd.com/projects/hipBLASLt/en/latest/reference/api-reference.html#supported-data-types
    if (granularity == "MX_BLOCKWISE") {
        PRIMUS_TURBO_CHECK(n % 16 == 0);
        PRIMUS_TURBO_CHECK(m % 16 == 0);
        PRIMUS_TURBO_CHECK(k % 128 == 0);

        PRIMUS_TURBO_CHECK(!transA && transB);

        PRIMUS_TURBO_CHECK(scaleA_inv.dim() == 2, "Scale A must be a 2-D tensor.");
        PRIMUS_TURBO_CHECK(scaleB_inv.dim() == 2, "Scale B must be a 2-D tensor.");
    }

    auto               handle            = at::cuda::getCurrentCUDABlasLtHandle();
    hipblasOperation_t trans_operation_A = transA ? HIPBLAS_OP_T : HIPBLAS_OP_N;
    hipblasOperation_t trans_operation_B = transB ? HIPBLAS_OP_T : HIPBLAS_OP_N;
    const hipDataType  A_type            = get_hipblaslt_dtype(A.scalar_type());
    const hipDataType  B_type            = get_hipblaslt_dtype(B.scalar_type());
    const hipDataType  D_type            = get_hipblaslt_dtype(out_dtype);
    const int64_t      workspace_size    = get_hipblaslt_workspace_size_in_byte();

    if (out_algo_count) {
        // clang-format off
        *out_algo_count = hipblaslt_gemm_get_algo_count(
            static_cast<const void *>(B.data_ptr()), B_type,
            rows_b, cols_b, ldb, nullptr, trans_operation_B,
            static_cast<const void *>(A.data_ptr()), A_type,
            rows_a, cols_a, lda, nullptr, trans_operation_A,
            D_type, rows_d, cols_d, ldd,
            workspace_size, false, HIPBLASLT_MATMUL_MATRIX_SCALE_END, handle);
        // clang-format on
        return {};
    }

    at::Tensor        C      = at::empty({m, n}, torch::dtype(out_dtype).device(at::kCUDA));
    auto              stream = at::hip::getCurrentHIPStream();
    const hipDataType C_type = get_hipblaslt_dtype(C.scalar_type());
    at::Tensor workspace = at::empty({workspace_size}, torch::dtype(at::kByte).device(at::kCUDA));

    // clang-format off
    // NOTE: hipblaslt expects tensor in col-major but torch Tensor is in row-major.
    // Swapping A&B that are essentially computing C^T = B^T @ A^T.
    hipblaslt_gemm_impl(
        static_cast<const void *>(B.data_ptr()), B_type,
        rows_b, cols_b, ldb,
        static_cast<const void*>(scaleB_inv.data_ptr()),
        trans_operation_B,
        static_cast<const void *>(A.data_ptr()), A_type,
        rows_a, cols_a, lda,
        static_cast<const void*>(scaleA_inv.data_ptr()),
        trans_operation_A,
        static_cast<void *>(C.data_ptr()), C_type,
        rows_d, cols_d, ldd,
        static_cast<void *>(workspace.data_ptr()), workspace_size,
        use_fp8,
        scale_mode,
        handle, stream);
    // clang-format on

    return C;
}

static at::Tensor hipblaslt_gemm_fp4_core(at::Tensor A, at::Tensor scaleA_inv, at::Tensor B,
                                          at::Tensor scaleB_inv, const at::ScalarType out_dtype,
                                          bool transA, bool transB, bool transC,
                                          const std::string &granularity, int64_t algo_index = 0,
                                          int *out_algo_count = nullptr) {
    const bool use_fp4 = is_4bit_floating_point_dtype(A.scalar_type()) &&
                         is_4bit_floating_point_dtype(B.scalar_type());

    PRIMUS_TURBO_CHECK(use_fp4, "A and B must be FP4 Tensor.");

    // scale mode
    hipblasLtMatmulMatrixScale_t scale_mode = HIPBLASLT_MATMUL_MATRIX_SCALE_END;
    if (granularity == "MX_BLOCKWISE") {
        scale_mode = HIPBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0;
    } else {
        PRIMUS_TURBO_ERROR("Invalid granularity.");
    }

    PRIMUS_TURBO_CHECK(is_4bit_floating_point_dtype(A.scalar_type()));
    PRIMUS_TURBO_CHECK(is_4bit_floating_point_dtype(B.scalar_type()));
    PRIMUS_TURBO_CHECK(is_16bit_floating_point_dtype(out_dtype));

    if (granularity == "MX_BLOCKWISE") {
        // scaling factor is e8m0 format.
        PRIMUS_TURBO_CHECK(scaleA_inv.scalar_type() == at::kFloat8_e8m0fnu);
        PRIMUS_TURBO_CHECK(scaleB_inv.scalar_type() == at::kFloat8_e8m0fnu);
    }

    // contiguous check
    PRIMUS_TURBO_CHECK(A.is_contiguous(), "A must be contiguous");
    PRIMUS_TURBO_CHECK(B.is_contiguous(), "B must be contiguous");

    // shape check
    PRIMUS_TURBO_CHECK(A.dim() == 2 && B.dim() == 2, "A, B must be 2D tensors");

    if (transC) {
        std::swap(A, B);
        std::swap(scaleA_inv, scaleB_inv);
        std::tie(transA, transB) = std::make_tuple(!transB, !transA);
    }

    // NOTE: The k dim is packed for FP4.
    const int64_t m = transA ? A.size(1) : A.size(0);
    const int64_t k = transA ? A.size(0) * 2 : A.size(1) * 2;
    const int64_t n = transB ? B.size(0) : B.size(1);

    // rows/cols
    const int64_t rows_a = transA ? m : k;
    const int64_t cols_a = transA ? k : m;
    const int64_t rows_b = transB ? k : n;
    const int64_t cols_b = transB ? n : k;
    const int64_t rows_d = n;
    const int64_t cols_d = m;

    // NOTE: The leading dimension is col-major.
    int64_t lda, ldb, ldd;
    if (!transA && transB) { // NT
        PRIMUS_TURBO_CHECK(A.size(1) == B.size(1), "tensor size mismatch");
        lda = k;
        ldb = k;
        ldd = n;
    } else if (!transA && !transB) { // NN
        PRIMUS_TURBO_CHECK(A.size(1) == B.size(0), "tensor size mismatch");
        lda = k;
        ldb = n;
        ldd = n;
    } else if (transA && !transB) { // TN
        PRIMUS_TURBO_CHECK(A.size(0) == B.size(0), "tensor size mismatch");
        lda = m;
        ldb = n;
        ldd = n;
    } else {
        PRIMUS_TURBO_ERROR("Not support layout.");
    }

    // MXFP4 extra check, ref:
    // https://rocm.docs.amd.com/projects/hipBLASLt/en/latest/reference/api-reference.html#supported-data-types
    if (granularity == "MX_BLOCKWISE") {
        PRIMUS_TURBO_CHECK(n % 16 == 0);
        PRIMUS_TURBO_CHECK(m % 16 == 0);
        PRIMUS_TURBO_CHECK(k % 128 == 0);

        PRIMUS_TURBO_CHECK(!transA && transB);

        PRIMUS_TURBO_CHECK(scaleA_inv.dim() == 2, "Scale A must be a 2-D tensor.");
        PRIMUS_TURBO_CHECK(scaleB_inv.dim() == 2, "Scale B must be a 2-D tensor.");
    }

    auto               handle            = at::cuda::getCurrentCUDABlasLtHandle();
    hipblasOperation_t trans_operation_A = transA ? HIPBLAS_OP_T : HIPBLAS_OP_N;
    hipblasOperation_t trans_operation_B = transB ? HIPBLAS_OP_T : HIPBLAS_OP_N;
    const hipDataType  A_type            = get_hipblaslt_dtype(A.scalar_type());
    const hipDataType  B_type            = get_hipblaslt_dtype(B.scalar_type());
    const hipDataType  D_type            = get_hipblaslt_dtype(out_dtype);
    const int64_t      workspace_size    = get_hipblaslt_workspace_size_in_byte();

    if (out_algo_count) {
        // clang-format off
        *out_algo_count = hipblaslt_gemm_get_algo_count(
            static_cast<const void *>(B.data_ptr()), B_type,
            rows_b, cols_b, ldb, nullptr, trans_operation_B,
            static_cast<const void *>(A.data_ptr()), A_type,
            rows_a, cols_a, lda, nullptr, trans_operation_A,
            D_type, rows_d, cols_d, ldd,
            workspace_size, false, HIPBLASLT_MATMUL_MATRIX_SCALE_END, handle);
        // clang-format on
        return {};
    }

    at::Tensor        C      = at::empty({m, n}, torch::dtype(out_dtype).device(at::kCUDA));
    auto              stream = at::hip::getCurrentHIPStream();
    const hipDataType C_type = get_hipblaslt_dtype(C.scalar_type());
    at::Tensor workspace = at::empty({workspace_size}, torch::dtype(at::kByte).device(at::kCUDA));

    // clang-format off
    // NOTE: hipblaslt expects tensor in col-major but torch Tensor is in row-major.
    // Swapping A&B that are essentially computing C^T = B^T @ A^T.
    hipblaslt_gemm_impl(
        static_cast<const void *>(B.data_ptr()), B_type,
        rows_b, cols_b, ldb,
        static_cast<const void*>(scaleB_inv.data_ptr()),
        trans_operation_B,
        static_cast<const void *>(A.data_ptr()), A_type,
        rows_a, cols_a, lda,
        static_cast<const void*>(scaleA_inv.data_ptr()),
        trans_operation_A,
        static_cast<void *>(C.data_ptr()), C_type,
        rows_d, cols_d, ldd,
        static_cast<void *>(workspace.data_ptr()), workspace_size,
        use_fp4,
        scale_mode,
        handle, stream, static_cast<int>(algo_index));
    // clang-format on

    return C;
}

// ---- Public API: thin wrappers around *_core ----

at::Tensor hipblaslt_gemm(at::Tensor A, at::Tensor B, const at::ScalarType out_dtype, bool transA,
                          bool transB, bool transC, int64_t algo_index) {
    return hipblaslt_gemm_core(A, B, out_dtype, transA, transB, transC, algo_index);
}

int64_t hipblaslt_gemm_algo_count(at::Tensor A, at::Tensor B, const at::ScalarType out_dtype,
                                  bool transA, bool transB, bool transC) {
    int count = 0;
    hipblaslt_gemm_core(A, B, out_dtype, transA, transB, transC, 0, &count);
    return count;
}

at::Tensor hipblaslt_gemm_fp8(at::Tensor A, at::Tensor scaleA_inv, at::Tensor B,
                              at::Tensor scaleB_inv, const at::ScalarType out_dtype, bool transA,
                              bool transB, bool transC, const std::string &granularity,
                              int64_t algo_index) {
    return hipblaslt_gemm_fp8_core(A, scaleA_inv, B, scaleB_inv, out_dtype, transA, transB, transC,
                                   granularity, algo_index);
}

int64_t hipblaslt_gemm_fp8_algo_count(at::Tensor A, at::Tensor scaleA_inv, at::Tensor B,
                                      at::Tensor scaleB_inv, const at::ScalarType out_dtype,
                                      bool transA, bool transB, bool transC,
                                      const std::string &granularity) {
    int count = 0;
    hipblaslt_gemm_fp8_core(A, scaleA_inv, B, scaleB_inv, out_dtype, transA, transB, transC,
                            granularity, 0, &count);
    return count;
}

at::Tensor hipblaslt_gemm_fp4(at::Tensor A, at::Tensor scaleA_inv, at::Tensor B,
                              at::Tensor scaleB_inv, const at::ScalarType out_dtype, bool transA,
                              bool transB, bool transC, const std::string &granularity,
                              int64_t algo_index) {
    return hipblaslt_gemm_fp4_core(A, scaleA_inv, B, scaleB_inv, out_dtype, transA, transB, transC,
                                   granularity, algo_index);
}

int64_t hipblaslt_gemm_fp4_algo_count(at::Tensor A, at::Tensor scaleA_inv, at::Tensor B,
                                      at::Tensor scaleB_inv, const at::ScalarType out_dtype,
                                      bool transA, bool transB, bool transC,
                                      const std::string &granularity) {
    int count = 0;
    hipblaslt_gemm_fp4_core(A, scaleA_inv, B, scaleB_inv, out_dtype, transA, transB, transC,
                            granularity, 0, &count);
    return count;
}

} // namespace primus_turbo::pytorch
