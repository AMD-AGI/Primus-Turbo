// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
//
// See LICENSE for license information.

#include "primus_turbo/grouped_gemm.h"
#include "primus_turbo/gemm.h"

#include "../extensions.h"
#include "../type_traits.h"
#include "../utils.h"

namespace primus_turbo::pytorch {

// ── MX_BLOCKWISE launch ──
//
// Variable-M grouped GEMM, NT layout:
//   A: [total_M, K] FP8                     (groups concatenated along M)
//   B: [group_num, N, K] FP8                (per-group weight, N-major)
//   C: [total_M, N] FP16/BF16
//   a_scales: [total_M, K/32] E8M0          (groups concatenated along M)
//   b_scales: [group_num, N, K/32] E8M0
//   group_lens: [group_num] int64           (M_g per group, sum = total_M)
//   group_offs: [group_num+1] int64         (cumulative row offsets)

static at::Tensor launch_mxfp8_grouped(at::Tensor &a, at::Tensor &a_scales, at::Tensor &b,
                                       at::Tensor &b_scales, at::Tensor &group_lens,
                                       at::Tensor &group_offs, const at::ScalarType out_dtype,
                                       bool transA, bool transB) {
    PRIMUS_TURBO_CHECK(a_scales.scalar_type() == at::kFloat8_e8m0fnu, "Scale A must be E8M0.");
    PRIMUS_TURBO_CHECK(b_scales.scalar_type() == at::kFloat8_e8m0fnu, "Scale B must be E8M0.");
    PRIMUS_TURBO_CHECK(a_scales.dim() == 2, "Scale A must be 2D [total_M, K/32].");
    PRIMUS_TURBO_CHECK(b_scales.dim() == 3, "Scale B must be 3D [group_num, N, K/32].");
    PRIMUS_TURBO_CHECK(group_lens.scalar_type() == at::kLong, "group_lens must be int64.");
    PRIMUS_TURBO_CHECK(group_offs.scalar_type() == at::kLong, "group_offs must be int64.");

    PRIMUS_TURBO_CHECK(!transA && transB,
                       "turbo_grouped_gemm_fp8 MX_BLOCKWISE only supports NT layout.");

    PRIMUS_TURBO_CHECK(a.dim() == 2, "A must be 2D [total_M, K].");
    PRIMUS_TURBO_CHECK(b.dim() == 3, "B must be 3D [group_num, N, K].");

    const int64_t total_m   = a.size(0);
    const int64_t k         = a.size(1);
    const int64_t group_num = b.size(0);
    const int64_t n         = b.size(1);
    PRIMUS_TURBO_CHECK(k == b.size(2), "K dimension mismatch between A and B.");
    PRIMUS_TURBO_CHECK(group_lens.numel() == group_num,
                       "group_lens.numel() must equal group_num (B.size(0)).");
    PRIMUS_TURBO_CHECK(group_offs.numel() == group_num + 1,
                       "group_offs.numel() must equal group_num + 1.");

    PRIMUS_TURBO_CHECK(n % 16 == 0, "N must be multiple of 16.");
    PRIMUS_TURBO_CHECK(k % 128 == 0, "K must be multiple of 128.");
    PRIMUS_TURBO_CHECK(k >= 384, "K must be >= 384.");
    PRIMUS_TURBO_CHECK(total_m % 16 == 0,
                       "total_M must be multiple of 16 (per-group M_g preshuffle alignment).");

    at::Tensor c = at::empty({total_m, n}, torch::dtype(out_dtype).device(a.device()));

    auto stream = at::cuda::getCurrentCUDAStream();

    // Compute group metadata host-side for conservative per-group MXFP8 launches.
    at::Tensor    lens_cpu  = group_lens.cpu();
    const int64_t *lens_data = lens_cpu.data_ptr<int64_t>();
    at::Tensor    offs_cpu  = group_offs.cpu();
    const int64_t *offs_data = offs_cpu.data_ptr<int64_t>();
    int64_t        max_m     = 0;
    for (int g = 0; g < group_num; ++g) {
        max_m = std::max(max_m, lens_data[g]);
    }
    const int64_t scale_cols = k / 32;
    const size_t  ws_size = primus_turbo::turbo_gemm_mxfp8_workspace_size(max_m, n, k);
    at::Tensor    workspace =
        at::empty({(int64_t) ws_size}, torch::dtype(at::kByte).device(a.device()));

    TORCH_TYPE_SWITCH_FP8(
        a.scalar_type(), AType,
        TORCH_TYPE_SWITCH_FP8(
            b.scalar_type(), BType,
            TORCH_TYPE_SWITCH_FP16_BF16(
                out_dtype, CType,
                auto *a_ptr = reinterpret_cast<const AType *>(a.data_ptr());
                auto *b_ptr = reinterpret_cast<const BType *>(b.data_ptr());
                auto *c_ptr = reinterpret_cast<CType *>(c.data_ptr());
                auto *a_scale_ptr =
                    reinterpret_cast<const dtype::float8_e8m0 *>(a_scales.data_ptr());
                auto *b_scale_ptr =
                    reinterpret_cast<const dtype::float8_e8m0 *>(b_scales.data_ptr());
                for (int64_t g = 0; g < group_num; ++g) {
                    const int64_t m_g = lens_data[g];
                    if (m_g <= 0) {
                        continue;
                    }
                    const int64_t off = offs_data[g];
                    primus_turbo::turbo_gemm_mxfp8_impl<AType, BType, CType>(
                        a_ptr + off * k,
                        b_ptr + g * n * k,
                        a_scale_ptr + off * scale_cols,
                        b_scale_ptr + g * n * scale_cols,
                        c_ptr + off * n,
                        m_g, n, k, workspace.data_ptr(), ws_size, stream);
                }
                return c;)))

    PRIMUS_TURBO_ERROR("Unsupported dtype combination for turbo_grouped_gemm_fp8 MX_BLOCKWISE.");
    return c;
}

// ── Wgrad variable-K MX_BLOCKWISE launch ──
// Inputs (col-quantized, transposed FP8):
//   lhs (dC^T): (N, total_M), lhs_scales: (N, total_M/32)
//   rhs (A^T):  (K, total_M), rhs_scales: (K, total_M/32)
// Output dB:    (G, N, K)
static at::Tensor launch_mxfp8_grouped_wgrad(at::Tensor &lhs, at::Tensor &lhs_scales,
                                             at::Tensor &rhs, at::Tensor &rhs_scales,
                                             at::Tensor &group_lens, at::Tensor &group_offs,
                                             const at::ScalarType out_dtype) {
    PRIMUS_TURBO_CHECK(lhs_scales.scalar_type() == at::kFloat8_e8m0fnu, "Scale LHS must be E8M0.");
    PRIMUS_TURBO_CHECK(rhs_scales.scalar_type() == at::kFloat8_e8m0fnu, "Scale RHS must be E8M0.");
    PRIMUS_TURBO_CHECK(lhs.dim() == 2, "LHS (dC^T) must be 2D [N, total_M].");
    PRIMUS_TURBO_CHECK(rhs.dim() == 2, "RHS (A^T)  must be 2D [K, total_M].");
    PRIMUS_TURBO_CHECK(group_lens.scalar_type() == at::kLong, "group_lens must be int64.");
    PRIMUS_TURBO_CHECK(group_offs.scalar_type() == at::kLong, "group_offs must be int64.");

    const int64_t n         = lhs.size(0);
    const int64_t total_m   = lhs.size(1);
    const int64_t k         = rhs.size(0);
    const int64_t group_num = group_lens.numel();
    PRIMUS_TURBO_CHECK(rhs.size(1) == total_m, "LHS and RHS must agree on total_M.");
    PRIMUS_TURBO_CHECK(group_offs.numel() == group_num + 1,
                       "group_offs.numel() must equal group_num + 1.");
    PRIMUS_TURBO_CHECK(n % 16 == 0, "N must be multiple of 16 (preshuffle row alignment).");
    PRIMUS_TURBO_CHECK(k % 16 == 0, "K must be multiple of 16 (preshuffle row alignment).");
    PRIMUS_TURBO_CHECK(total_m % 128 == 0,
                       "total_M must be multiple of 128 (MX preshuffled scale col-block "
                       "alignment); also required: each per-group M_g % 128 == 0.");

    at::Tensor db = at::empty({group_num, n, k}, torch::dtype(out_dtype).device(lhs.device()));

    const size_t ws_size =
        primus_turbo::turbo_grouped_gemm_mxfp8_wgrad_workspace_size(total_m, n, k);
    at::Tensor workspace =
        at::empty({(int64_t) ws_size}, torch::dtype(at::kByte).device(lhs.device()));

    auto stream = at::cuda::getCurrentCUDAStream();

    TORCH_TYPE_SWITCH_FP8(
        lhs.scalar_type(), AType,
        TORCH_TYPE_SWITCH_FP8(
            rhs.scalar_type(), BType,
            TORCH_TYPE_SWITCH_FP16_BF16(
                out_dtype, CType,
                primus_turbo::TurboGroupedGemmMXFP8WgradParams<AType, BType, CType> params;
                params.lhs_ptr = reinterpret_cast<const AType *>(lhs.data_ptr());
                params.rhs_ptr = reinterpret_cast<const BType *>(rhs.data_ptr());
                params.db_ptr  = reinterpret_cast<CType *>(db.data_ptr());
                params.lhs_scale_ptr =
                    reinterpret_cast<const dtype::float8_e8m0 *>(lhs_scales.data_ptr());
                params.rhs_scale_ptr =
                    reinterpret_cast<const dtype::float8_e8m0 *>(rhs_scales.data_ptr());
                params.group_lens_ptr = reinterpret_cast<const int64_t *>(group_lens.data_ptr());
                params.group_offs_ptr = reinterpret_cast<const int64_t *>(group_offs.data_ptr());
                params.group_num = (int32_t) group_num; params.total_m = (int32_t) total_m;
                params.n = (int32_t) n; params.k = (int32_t) k;
                params.workspace = workspace.data_ptr(); params.workspace_size = ws_size;
                params.stream    = stream;
                primus_turbo::turbo_grouped_gemm_mxfp8_wgrad_impl<AType, BType, CType>(params);
                return db;)))

    PRIMUS_TURBO_ERROR("Unsupported dtype combination for turbo_grouped_gemm_fp8_wgrad MX_BLOCKWISE.");
    return db;
}

// ── Top-level entry point ──

at::Tensor turbo_grouped_gemm_fp8(at::Tensor &a, at::Tensor &b, at::Tensor &a_scales,
                                  at::Tensor &b_scales, at::Tensor &group_lens,
                                  at::Tensor &group_offs, const bool transA, const bool transB,
                                  at::ScalarType out_dtype, const std::string &granularity) {
    PRIMUS_TURBO_CHECK(is_8bit_floating_point_dtype(a.scalar_type()), "A must be FP8.");
    PRIMUS_TURBO_CHECK(is_8bit_floating_point_dtype(b.scalar_type()), "B must be FP8.");
    PRIMUS_TURBO_CHECK(is_16bit_floating_point_dtype(out_dtype), "out_dtype must be fp16 or bf16.");
    PRIMUS_TURBO_CHECK(a.is_contiguous(), "A must be contiguous.");
    PRIMUS_TURBO_CHECK(b.is_contiguous(), "B must be contiguous.");
    PRIMUS_TURBO_CHECK(a_scales.is_contiguous(), "a_scales must be contiguous.");
    PRIMUS_TURBO_CHECK(b_scales.is_contiguous(), "b_scales must be contiguous.");

    if (granularity == "MX_BLOCKWISE") {
        return launch_mxfp8_grouped(a, a_scales, b, b_scales, group_lens, group_offs, out_dtype,
                                    transA, transB);
    }

    PRIMUS_TURBO_ERROR("turbo_grouped_gemm_fp8: unsupported granularity '" + granularity + "'.");
    return at::Tensor();
}

at::Tensor turbo_grouped_gemm_fp8_meta(at::Tensor &a, at::Tensor &b, at::Tensor &a_scales,
                                       at::Tensor &b_scales, at::Tensor &group_lens,
                                       at::Tensor &group_offs, const bool transA, const bool transB,
                                       at::ScalarType out_dtype, const std::string &granularity) {
    const int64_t total_m = a.size(0);
    const int64_t n       = transB ? b.size(1) : b.size(2);
    return at::empty({total_m, n}, at::dtype(out_dtype).device(at::kMeta));
}

// ── Wgrad top-level entry ──

at::Tensor turbo_grouped_gemm_fp8_wgrad(at::Tensor &lhs, at::Tensor &lhs_scales, at::Tensor &rhs,
                                        at::Tensor &rhs_scales, at::Tensor &group_lens,
                                        at::Tensor &group_offs, at::ScalarType out_dtype,
                                        const std::string &granularity) {
    PRIMUS_TURBO_CHECK(is_8bit_floating_point_dtype(lhs.scalar_type()), "LHS must be FP8.");
    PRIMUS_TURBO_CHECK(is_8bit_floating_point_dtype(rhs.scalar_type()), "RHS must be FP8.");
    PRIMUS_TURBO_CHECK(is_16bit_floating_point_dtype(out_dtype), "out_dtype must be fp16 or bf16.");
    PRIMUS_TURBO_CHECK(lhs.is_contiguous(), "LHS must be contiguous.");
    PRIMUS_TURBO_CHECK(rhs.is_contiguous(), "RHS must be contiguous.");
    PRIMUS_TURBO_CHECK(lhs_scales.is_contiguous(), "lhs_scales must be contiguous.");
    PRIMUS_TURBO_CHECK(rhs_scales.is_contiguous(), "rhs_scales must be contiguous.");

    if (granularity == "MX_BLOCKWISE") {
        return launch_mxfp8_grouped_wgrad(lhs, lhs_scales, rhs, rhs_scales, group_lens, group_offs,
                                          out_dtype);
    }
    PRIMUS_TURBO_ERROR("turbo_grouped_gemm_fp8_wgrad: unsupported granularity '" + granularity +
                       "'.");
    return at::Tensor();
}

at::Tensor turbo_grouped_gemm_fp8_wgrad_meta(at::Tensor &lhs, at::Tensor &lhs_scales,
                                             at::Tensor &rhs, at::Tensor &rhs_scales,
                                             at::Tensor &group_lens, at::Tensor &group_offs,
                                             at::ScalarType out_dtype,
                                             const std::string &granularity) {
    const int64_t n         = lhs.size(0);
    const int64_t k         = rhs.size(0);
    const int64_t group_num = group_lens.numel();
    return at::empty({group_num, n, k}, at::dtype(out_dtype).device(at::kMeta));
}

} // namespace primus_turbo::pytorch
