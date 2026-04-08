// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
//
// See LICENSE for license information.

#include "primus_turbo/gemm.h"
#include "turbo/turbo_gemm_mxfp8_kernel.h"
#include <hip/hip_runtime.h>

namespace primus_turbo {

// ── Workspace size ──

size_t turbo_gemm_mxfp8_workspace_size(int32_t m, int32_t n, int32_t k) {
    constexpr int32_t MX_BLOCK_SIZE = 32;
    const int32_t     scale_cols    = (k + MX_BLOCK_SIZE - 1) / MX_BLOCK_SIZE;
    const size_t      a_scale_bytes = (size_t) m * scale_cols * sizeof(uint32_t);
    const size_t      b_scale_bytes = (size_t) n * scale_cols * sizeof(uint32_t);
    return a_scale_bytes + b_scale_bytes;
}

// ── Public API ──

template <typename AType, typename BType, typename CType>
void turbo_gemm_mxfp8_impl(const AType *a_ptr, const BType *b_ptr,
                           const dtype::float8_e8m0 *a_scale_ptr,
                           const dtype::float8_e8m0 *b_scale_ptr, CType *c_ptr, int32_t m,
                           int32_t n, int32_t k, void *workspace, size_t workspace_size,
                           hipStream_t stream) {
    constexpr int32_t MX_BLOCK_SIZE = 32;
    const int32_t     scale_cols    = (k + MX_BLOCK_SIZE - 1) / MX_BLOCK_SIZE;
    const size_t      a_scale_bytes = (size_t) m * scale_cols * sizeof(uint32_t);

    auto *a_scale_preshuf = reinterpret_cast<uint32_t *>(workspace);
    auto *b_scale_preshuf =
        reinterpret_cast<uint32_t *>(reinterpret_cast<uint8_t *>(workspace) + a_scale_bytes);

    // E8M0 raw bytes → uint32_t (zero-extend) + preshuffle
    auto *a_scale_raw = reinterpret_cast<const uint8_t *>(a_scale_ptr);
    auto *b_scale_raw = reinterpret_cast<const uint8_t *>(b_scale_ptr);
    turbo::preshuffle_scale_16x4_kernel<uint8_t, uint32_t>
        <<<m / 16, 64, 0, stream>>>(a_scale_raw, a_scale_preshuf, m, scale_cols);
    turbo::preshuffle_scale_16x4_kernel<uint8_t, uint32_t>
        <<<n / 16, 64, 0, stream>>>(b_scale_raw, b_scale_preshuf, n, scale_cols);

    // Launch GEMM kernel
    constexpr int BLOCK_M = 256, BLOCK_N = 256;
    dim3          grid((m + BLOCK_M - 1) / BLOCK_M, (n + BLOCK_N - 1) / BLOCK_N);
    dim3          block(256);
    turbo::turbo_gemm_mxfp8_256x256x128_16x16x128_4wave_kernel<AType, BType, CType>
        <<<grid, block, 0, stream>>>(a_ptr, b_ptr, a_scale_preshuf, b_scale_preshuf, c_ptr, m, n,
                                     k);
}

// ── Explicit instantiations ──

#define INSTANTIATE_TURBO_GEMM(A, B, C)                                                            \
    template void turbo_gemm_mxfp8_impl<A, B, C>(const A *, const B *, const dtype::float8_e8m0 *, \
                                                 const dtype::float8_e8m0 *, C *, int32_t,         \
                                                 int32_t, int32_t, void *, size_t, hipStream_t);

INSTANTIATE_TURBO_GEMM(dtype::float8_e4m3, dtype::float8_e4m3, dtype::float16)
INSTANTIATE_TURBO_GEMM(dtype::float8_e4m3, dtype::float8_e4m3, dtype::bfloat16)
INSTANTIATE_TURBO_GEMM(dtype::float8_e5m2, dtype::float8_e5m2, dtype::float16)
INSTANTIATE_TURBO_GEMM(dtype::float8_e5m2, dtype::float8_e5m2, dtype::bfloat16)
INSTANTIATE_TURBO_GEMM(dtype::float8_e4m3, dtype::float8_e5m2, dtype::float16)
INSTANTIATE_TURBO_GEMM(dtype::float8_e4m3, dtype::float8_e5m2, dtype::bfloat16)
INSTANTIATE_TURBO_GEMM(dtype::float8_e5m2, dtype::float8_e4m3, dtype::float16)
INSTANTIATE_TURBO_GEMM(dtype::float8_e5m2, dtype::float8_e4m3, dtype::bfloat16)

#undef INSTANTIATE_TURBO_GEMM

} // namespace primus_turbo
