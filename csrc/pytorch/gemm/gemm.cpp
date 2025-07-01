#include "primus_turbo/gemm.h"

#include "../extensions.h"

namespace primus_turbo::pytorch {

static hipDataType get_hipblaslt_dtype(const at::ScalarType t) {
    switch (t) {
    case at::kHalf:
        return HIP_R_16F;
    case at::kFloat:
        return HIP_R_32F;
    case at::kBFloat16:
        return HIP_R_16BF;
    case at::kFloat8_e4m3fnuz:
        return HIP_R_8F_E4M3_FNUZ;
    case at::kFloat8_e4m3fn:
        return HIP_R_8F_E4M3;
    case at::kFloat8_e5m2fnuz:
        return HIP_R_8F_E5M2_FNUZ;
    case at::kFloat8_e5m2:
        return HIP_R_8F_E5M2;
    default:
        AT_ERROR("Invalid type");
    }
}

static inline bool is_16bit_floating_point_dtype(at::ScalarType dtype) {
    return dtype == at::kHalf || dtype == at::kBFloat16;
}

static inline bool is_floating_point_dtype(at::ScalarType dtype) {
    return dtype == at::kHalf || dtype == at::kBFloat16 || dtype == at::kFloat;
}

at::Tensor gemm(const at::Tensor A, const at::Tensor B, const at::ScalarType out_dtype,
                const bool transA, const bool transB) {
    PRIMUS_TURBO_CHECK(is_16bit_floating_point_dtype(A.scalar_type()));
    PRIMUS_TURBO_CHECK(is_16bit_floating_point_dtype(B.scalar_type()));
    PRIMUS_TURBO_CHECK(A.scalar_type() == B.scalar_type(), "A and B dtype mismatch");
    PRIMUS_TURBO_CHECK(is_floating_point_dtype(out_dtype));

    PRIMUS_TURBO_CHECK(A.is_contiguous(), "A must be contiguous");
    PRIMUS_TURBO_CHECK(B.is_contiguous(), "B must be contiguous");

    PRIMUS_TURBO_CHECK(A.dim() == 2 && B.dim() == 2, "A, B must be 2D tensors");

    const int64_t m = transA ? A.size(1) : A.size(0);
    const int64_t k = transA ? A.size(0) : A.size(1);
    const int64_t n = transB ? B.size(0) : B.size(1);

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

    at::Tensor C = at::empty({m, n}, torch::dtype(out_dtype).device(at::kCUDA));

    auto stream = at::hip::getCurrentHIPStream();
    auto handle = at::cuda::getCurrentCUDABlasLtHandle();

    hipblasOperation_t trans_operation_A = transA ? HIPBLAS_OP_T : HIPBLAS_OP_N;
    hipblasOperation_t trans_operation_B = transB ? HIPBLAS_OP_T : HIPBLAS_OP_N;
    const hipDataType  A_type            = get_hipblaslt_dtype(A.scalar_type());
    const hipDataType  B_type            = get_hipblaslt_dtype(B.scalar_type());
    const hipDataType  C_type            = get_hipblaslt_dtype(C.scalar_type());

    const int64_t workspace_size = get_hipblaslt_workspace_size_in_byte();
    at::Tensor workspace = at::empty({workspace_size}, torch::dtype(at::kByte).device(at::kCUDA));

    // clang-format off
    // NOTE: hipblaslt expects tensor in col-major but torch Tensor is in row-major.
    // Swapping A&B that are essentially computing C^T = B^T @ A^T.
    hipblaslt_gemm(
        static_cast<const void *>(B.data_ptr()), B_type, ldb, trans_operation_B,
        static_cast<const void *>(A.data_ptr()), A_type, lda, trans_operation_A,
        static_cast<void *>(C.data_ptr()), C_type, ldd,
        n, m, k,
        static_cast<void *>(workspace.data_ptr()), workspace_size,
        handle, stream);
    // clang-format on

    return C;
}

} // namespace primus_turbo::pytorch
