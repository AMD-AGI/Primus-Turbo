// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
//
// See LICENSE for license information.

#include "primus_turbo/arch.h"
#include "primus_turbo/grouped_gemm.h"
#include "turbo/turbo_grouped_gemm_mxfp8_kernel.h"
#include "turbo/turbo_grouped_gemm_mxfp8_wgrad_kernel.h"
#include <hip/hip_runtime.h>

namespace primus_turbo {

// Workspace layout:
//   [ A_scale_preshuf : total_M * scale_cols * uint32 ]
//   [ B_scale_preshuf : group_num * N * scale_cols * uint32 ]  (skipped when
//                                                               b_scale_preshuffled)
size_t turbo_grouped_gemm_mxfp8_workspace_size(int32_t total_m, int32_t group_num, int32_t n,
                                               int32_t k, bool b_scale_preshuffled) {
    constexpr int32_t MX_BLOCK_SIZE = 32;
    const int32_t     scale_cols    = (k + MX_BLOCK_SIZE - 1) / MX_BLOCK_SIZE;
    const size_t      a_scale_bytes = (size_t) total_m * scale_cols * sizeof(uint32_t);
    if (b_scale_preshuffled) {
        return a_scale_bytes;
    }
    const size_t b_scale_bytes = (size_t) group_num * (size_t) n * scale_cols * sizeof(uint32_t);
    return a_scale_bytes + b_scale_bytes;
}

void turbo_preshuffle_mxfp8_scale_16x4_launch(const uint8_t *in_ptr, uint32_t *out_ptr,
                                              int rows, int cols, hipStream_t stream) {
    turbo::preshuffle_scale_16x4_kernel<uint8_t, uint32_t>
        <<<(uint32_t) (rows / 16), 64, 0, stream>>>(in_ptr, out_ptr, rows, cols);
}

// ── Public API ──

template <typename AType, typename BType, typename CType>
void turbo_grouped_gemm_mxfp8_impl(const TurboGroupedGemmMXFP8Params<AType, BType, CType> &params) {
    constexpr int32_t MX_BLOCK_SIZE = 32;
    const int32_t     total_m       = params.total_m;
    const int32_t     group_num     = params.group_num;
    const int32_t     n             = params.n;
    const int32_t     k             = params.k;
    const int32_t     scale_cols    = (k + MX_BLOCK_SIZE - 1) / MX_BLOCK_SIZE;
    const size_t      a_scale_bytes = (size_t) total_m * scale_cols * sizeof(uint32_t);

    auto *a_scale_preshuf = reinterpret_cast<uint32_t *>(params.workspace);
    auto *a_scale_raw     = reinterpret_cast<const uint8_t *>(params.a_scale_ptr);
    // If B is already preshuffled (Python-side weight cache) we only
    // preshuffle A; otherwise A+B are fused into a single dual launch.
    const uint32_t *b_scale_preshuf;
    if (params.b_scale_preshuffled) {
        b_scale_preshuf = reinterpret_cast<const uint32_t *>(params.b_scale_ptr);
        turbo::preshuffle_scale_16x4_kernel<uint8_t, uint32_t>
            <<<total_m / 16, 64, 0, params.stream>>>(a_scale_raw, a_scale_preshuf, total_m,
                                                     scale_cols);
    } else {
        auto *b_scale_preshuf_workspace = reinterpret_cast<uint32_t *>(
            reinterpret_cast<uint8_t *>(params.workspace) + a_scale_bytes);
        auto *b_scale_raw = reinterpret_cast<const uint8_t *>(params.b_scale_ptr);
        turbo::preshuffle_scale_16x4_dual_kernel<uint8_t, uint32_t>
            <<<(total_m + group_num * n) / 16, 64, 0, params.stream>>>(
                a_scale_raw, a_scale_preshuf, total_m, b_scale_raw, b_scale_preshuf_workspace,
                scale_cols);
        b_scale_preshuf = b_scale_preshuf_workspace;
    }

    const int32_t grid_m = params.grid_x;
    const int32_t grid_n = (n + 255) / 256;
    dim3          grid(256);
    dim3          block(256);
    turbo::turbo_grouped_gemm_mxfp8_256x256x128_16x16x128_4wave_persistent_kernel<AType, BType,
                                                                                   CType>
        <<<grid, block, 0, params.stream>>>(
            params.a_ptr, params.b_ptr, a_scale_preshuf, b_scale_preshuf, params.c_ptr,
            params.group_lens_ptr, params.group_offs_ptr, group_num, (uint32_t) n, (uint32_t) k,
            grid_m, grid_n);
}

// ── Explicit instantiations ──

#define INSTANTIATE_TURBO_GROUPED_GEMM(A, B, C)                                                    \
    template void turbo_grouped_gemm_mxfp8_impl<A, B, C>(                                          \
        const TurboGroupedGemmMXFP8Params<A, B, C> &);

INSTANTIATE_TURBO_GROUPED_GEMM(dtype::float8_e4m3, dtype::float8_e4m3, dtype::float16)
INSTANTIATE_TURBO_GROUPED_GEMM(dtype::float8_e4m3, dtype::float8_e4m3, dtype::bfloat16)
INSTANTIATE_TURBO_GROUPED_GEMM(dtype::float8_e5m2, dtype::float8_e5m2, dtype::float16)
INSTANTIATE_TURBO_GROUPED_GEMM(dtype::float8_e5m2, dtype::float8_e5m2, dtype::bfloat16)
INSTANTIATE_TURBO_GROUPED_GEMM(dtype::float8_e4m3, dtype::float8_e5m2, dtype::float16)
INSTANTIATE_TURBO_GROUPED_GEMM(dtype::float8_e4m3, dtype::float8_e5m2, dtype::bfloat16)
INSTANTIATE_TURBO_GROUPED_GEMM(dtype::float8_e5m2, dtype::float8_e4m3, dtype::float16)
INSTANTIATE_TURBO_GROUPED_GEMM(dtype::float8_e5m2, dtype::float8_e4m3, dtype::bfloat16)

#undef INSTANTIATE_TURBO_GROUPED_GEMM

// ────────────────────────────────────────────────────────────────────
//  Wgrad variable-K path
// ────────────────────────────────────────────────────────────────────

// Workspace layout:
//   [ LHS_scale_preshuf : N         * (total_M / 32) * uint32 ]
//   [ RHS_scale_preshuf : K         * (total_M / 32) * uint32 ]
size_t turbo_grouped_gemm_mxfp8_wgrad_workspace_size(int32_t total_m, int32_t n, int32_t k) {
    constexpr int32_t MX_BLOCK_SIZE = 32;
    const int32_t     scale_cols    = (total_m + MX_BLOCK_SIZE - 1) / MX_BLOCK_SIZE;
    const size_t      lhs_scale_bytes = (size_t) n * scale_cols * sizeof(uint32_t);
    const size_t      rhs_scale_bytes = (size_t) k * scale_cols * sizeof(uint32_t);
    return lhs_scale_bytes + rhs_scale_bytes;
}

template <typename AType, typename BType, typename CType>
void turbo_grouped_gemm_mxfp8_wgrad_impl(
    const TurboGroupedGemmMXFP8WgradParams<AType, BType, CType> &params) {
    constexpr int32_t MX_BLOCK_SIZE = 32;
    const int32_t     total_m       = params.total_m;
    const int32_t     group_num     = params.group_num;
    const int32_t     n             = params.n;
    const int32_t     k             = params.k;
    const int32_t     scale_cols    = (total_m + MX_BLOCK_SIZE - 1) / MX_BLOCK_SIZE;
    const size_t      lhs_scale_bytes = (size_t) n * scale_cols * sizeof(uint32_t);

    auto *lhs_scale_preshuf = reinterpret_cast<uint32_t *>(params.workspace);
    auto *rhs_scale_preshuf = reinterpret_cast<uint32_t *>(
        reinterpret_cast<uint8_t *>(params.workspace) + lhs_scale_bytes);

    // Fused LHS+RHS preshuffle (both share scale_cols).
    auto *lhs_scale_raw = reinterpret_cast<const uint8_t *>(params.lhs_scale_ptr);
    auto *rhs_scale_raw = reinterpret_cast<const uint8_t *>(params.rhs_scale_ptr);
    turbo::preshuffle_scale_16x4_dual_kernel<uint8_t, uint32_t>
        <<<(n + k) / 16, 64, 0, params.stream>>>(lhs_scale_raw, lhs_scale_preshuf, n,
                                                  rhs_scale_raw, rhs_scale_preshuf, scale_cols);

    const int32_t grid_n = (n + 255) / 256;
    const int32_t grid_k = (k + 255) / 256;
    dim3          grid(256);
    dim3          block(256);
    turbo::turbo_grouped_gemm_mxfp8_wgrad_256x256x128_16x16x128_4wave_persistent_kernel<AType,
                                                                                         BType,
                                                                                         CType>
        <<<grid, block, 0, params.stream>>>(
            params.lhs_ptr, params.rhs_ptr, lhs_scale_preshuf, rhs_scale_preshuf, params.db_ptr,
            params.group_lens_ptr, params.group_offs_ptr, group_num, (uint32_t) total_m,
            (uint32_t) n, (uint32_t) k, grid_n, grid_k);
}

#define INSTANTIATE_TURBO_GROUPED_GEMM_WGRAD(A, B, C)                                              \
    template void turbo_grouped_gemm_mxfp8_wgrad_impl<A, B, C>(                                    \
        const TurboGroupedGemmMXFP8WgradParams<A, B, C> &);

INSTANTIATE_TURBO_GROUPED_GEMM_WGRAD(dtype::float8_e4m3, dtype::float8_e4m3, dtype::float16)
INSTANTIATE_TURBO_GROUPED_GEMM_WGRAD(dtype::float8_e4m3, dtype::float8_e4m3, dtype::bfloat16)
INSTANTIATE_TURBO_GROUPED_GEMM_WGRAD(dtype::float8_e5m2, dtype::float8_e5m2, dtype::float16)
INSTANTIATE_TURBO_GROUPED_GEMM_WGRAD(dtype::float8_e5m2, dtype::float8_e5m2, dtype::bfloat16)
INSTANTIATE_TURBO_GROUPED_GEMM_WGRAD(dtype::float8_e5m2, dtype::float8_e4m3, dtype::float16)
INSTANTIATE_TURBO_GROUPED_GEMM_WGRAD(dtype::float8_e5m2, dtype::float8_e4m3, dtype::bfloat16)
INSTANTIATE_TURBO_GROUPED_GEMM_WGRAD(dtype::float8_e4m3, dtype::float8_e5m2, dtype::float16)
INSTANTIATE_TURBO_GROUPED_GEMM_WGRAD(dtype::float8_e4m3, dtype::float8_e5m2, dtype::bfloat16)

#undef INSTANTIATE_TURBO_GROUPED_GEMM_WGRAD

} // namespace primus_turbo
