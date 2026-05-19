// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
//
// See LICENSE for license information.

#include "primus_turbo/gemm.h"

#include "../extensions.h"
#include "../type_traits.h"
#include "../utils.h"

namespace primus_turbo::pytorch {

// ── MX_BLOCKWISE launch ──

static at::Tensor launch_mxfp8(at::Tensor A, at::Tensor scaleA_inv, at::Tensor B,
                               at::Tensor scaleB_inv, const at::ScalarType out_dtype, bool transA,
                               bool transB, bool transC) {
    PRIMUS_TURBO_CHECK(scaleA_inv.scalar_type() == at::kFloat8_e8m0fnu, "Scale A must be E8M0.");
    PRIMUS_TURBO_CHECK(scaleB_inv.scalar_type() == at::kFloat8_e8m0fnu, "Scale B must be E8M0.");
    PRIMUS_TURBO_CHECK(scaleA_inv.dim() == 2, "Scale A must be 2D.");
    PRIMUS_TURBO_CHECK(scaleB_inv.dim() == 2, "Scale B must be 2D.");

    PRIMUS_TURBO_CHECK(!transA && transB && !transC,
                       "turbo_gemm_fp8 MX_BLOCKWISE only supports NT layout.");

    const int64_t m = A.size(0);
    const int64_t k = A.size(1);
    const int64_t n = B.size(0);
    PRIMUS_TURBO_CHECK(k == B.size(1), "K dimension mismatch.");

    PRIMUS_TURBO_CHECK(m % 16 == 0, "M must be multiple of 16.");
    PRIMUS_TURBO_CHECK(n % 16 == 0, "N must be multiple of 16.");
    PRIMUS_TURBO_CHECK(k % 128 == 0, "K must be multiple of 128.");
    PRIMUS_TURBO_CHECK(k >= 384, "K must be >= 384.");

    at::Tensor C = at::empty({m, n}, torch::dtype(out_dtype).device(A.device()));

    const size_t ws_size = primus_turbo::turbo_gemm_mxfp8_workspace_size(m, n, k);
    at::Tensor   workspace =
        at::empty({(int64_t) ws_size}, torch::dtype(at::kByte).device(A.device()));

    auto stream = at::cuda::getCurrentCUDAStream();

    TORCH_TYPE_SWITCH_FP8(
        A.scalar_type(), AType,
        TORCH_TYPE_SWITCH_FP8(
            B.scalar_type(), BType,
            TORCH_TYPE_SWITCH_FP16_BF16(
                out_dtype, CType,
                primus_turbo::turbo_gemm_mxfp8_impl<AType, BType, CType>(
                    reinterpret_cast<const AType *>(A.data_ptr()),
                    reinterpret_cast<const BType *>(B.data_ptr()),
                    reinterpret_cast<const dtype::float8_e8m0 *>(scaleA_inv.data_ptr()),
                    reinterpret_cast<const dtype::float8_e8m0 *>(scaleB_inv.data_ptr()),
                    reinterpret_cast<CType *>(C.data_ptr()), m, n, k, workspace.data_ptr(), ws_size,
                    stream);
                return C;)))

    PRIMUS_TURBO_ERROR("Unsupported dtype combination for turbo_gemm_fp8 MX_BLOCKWISE.");
    return C;
}

// ── BLOCKWISE launch (DeepSeek-V3 style 1×128 + 128×128, FP32 scales) ──
//
// A scale: [M, ceil(K/128)] FP32, contiguous (rowwise-1×128, K-major).
// B scale: [ceil(N/128), ceil(K/128)] FP32, contiguous (block-128×128).
// Forward NT only. Falls back to caller (Triton/hipBLASLt) when shapes do not
// satisfy the M/N/K-multiple-of-128 constraint of the turbo kernel.

static at::Tensor launch_blockwise_fp8(at::Tensor A, at::Tensor scaleA_inv, at::Tensor B,
                                       at::Tensor scaleB_inv, const at::ScalarType out_dtype,
                                       bool transA, bool transB, bool transC) {
    PRIMUS_TURBO_CHECK(scaleA_inv.scalar_type() == at::kFloat, "Scale A must be FP32.");
    PRIMUS_TURBO_CHECK(scaleB_inv.scalar_type() == at::kFloat, "Scale B must be FP32.");
    PRIMUS_TURBO_CHECK(scaleA_inv.dim() == 2, "Scale A must be 2D.");
    PRIMUS_TURBO_CHECK(scaleB_inv.dim() == 2, "Scale B must be 2D.");
    PRIMUS_TURBO_CHECK(scaleA_inv.is_contiguous(), "Scale A must be contiguous.");
    PRIMUS_TURBO_CHECK(scaleB_inv.is_contiguous(), "Scale B must be contiguous.");

    PRIMUS_TURBO_CHECK(!transA && transB && !transC,
                       "turbo_gemm_fp8 BLOCKWISE only supports NT layout.");

    const int64_t m = A.size(0);
    const int64_t k = A.size(1);
    const int64_t n = B.size(0);
    PRIMUS_TURBO_CHECK(k == B.size(1), "K dimension mismatch.");

    PRIMUS_TURBO_CHECK(primus_turbo::turbo_gemm_blockwise_fp8_supported(m, n, k),
                       "BLOCKWISE turbo path requires M/N/K all multiples of 128.");

    // Scale shape sanity: K-major.
    const int64_t scale_cols = (k + 127) / 128;
    PRIMUS_TURBO_CHECK(scaleA_inv.size(0) == m && scaleA_inv.size(1) == scale_cols,
                       "Scale A shape must be [M, ceil(K/128)].");
    const int64_t b_scale_rows = (n + 127) / 128;
    PRIMUS_TURBO_CHECK(scaleB_inv.size(0) == b_scale_rows && scaleB_inv.size(1) == scale_cols,
                       "Scale B shape must be [ceil(N/128), ceil(K/128)].");

    at::Tensor C = at::empty({m, n}, torch::dtype(out_dtype).device(A.device()));

    auto stream = at::cuda::getCurrentCUDAStream();

    TORCH_TYPE_SWITCH_FP8(
        A.scalar_type(), AType,
        TORCH_TYPE_SWITCH_FP8(
            B.scalar_type(), BType,
            TORCH_TYPE_SWITCH_FP16_BF16(
                out_dtype, CType,
                primus_turbo::turbo_gemm_blockwise_fp8_impl<AType, BType, CType>(
                    reinterpret_cast<const AType *>(A.data_ptr()),
                    reinterpret_cast<const BType *>(B.data_ptr()),
                    reinterpret_cast<const float *>(scaleA_inv.data_ptr()),
                    reinterpret_cast<const float *>(scaleB_inv.data_ptr()),
                    reinterpret_cast<CType *>(C.data_ptr()), m, n, k, stream);
                return C;)))

    PRIMUS_TURBO_ERROR("Unsupported dtype combination for turbo_gemm_fp8 BLOCKWISE.");
    return C;
}

// ── Top-level entry point ──

at::Tensor turbo_gemm_fp8(at::Tensor A, at::Tensor scaleA_inv, at::Tensor B, at::Tensor scaleB_inv,
                          const at::ScalarType out_dtype, bool transA, bool transB, bool transC,
                          const std::string &granularity) {
    PRIMUS_TURBO_CHECK(is_8bit_floating_point_dtype(A.scalar_type()), "A must be FP8.");
    PRIMUS_TURBO_CHECK(is_8bit_floating_point_dtype(B.scalar_type()), "B must be FP8.");
    PRIMUS_TURBO_CHECK(is_16bit_floating_point_dtype(out_dtype) || out_dtype == at::kFloat,
                       "out_dtype must be fp16, bf16, or fp32.");
    PRIMUS_TURBO_CHECK(A.is_contiguous(), "A must be contiguous");
    PRIMUS_TURBO_CHECK(B.is_contiguous(), "B must be contiguous");
    PRIMUS_TURBO_CHECK(A.dim() == 2 && B.dim() == 2, "A, B must be 2D tensors");

    if (granularity == "MX_BLOCKWISE") {
        return launch_mxfp8(A, scaleA_inv, B, scaleB_inv, out_dtype, transA, transB, transC);
    }
    if (granularity == "BLOCKWISE") {
        return launch_blockwise_fp8(A, scaleA_inv, B, scaleB_inv, out_dtype, transA, transB,
                                    transC);
    }

    PRIMUS_TURBO_ERROR("turbo_gemm_fp8: unsupported granularity '" + granularity + "'.");
    return at::Tensor();
}

} // namespace primus_turbo::pytorch
