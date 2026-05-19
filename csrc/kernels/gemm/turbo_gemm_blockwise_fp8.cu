// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
//
// See LICENSE for license information.
//
// Launcher for blockwise FP8 GEMM (DeepSeek-V3 style 1×128 / 128×128 FP32
// scales) on gfx950. The kernel is in turbo/turbo_gemm_blockwise_fp8_kernel.h
// and follows candidate F' from
// docs/kernel_optimize/primus_turbo_blockwise_fp8_gemm_design.md
// (128M × 128N × 128K tile, 4 warps, 16×16×128 unscaled MFMA, software
// promotion accumulator).

#include "primus_turbo/gemm.h"
#include "turbo/turbo_gemm_blockwise_fp8_kernel.h"
#include <hip/hip_runtime.h>

namespace primus_turbo {

// ── Support check ──
// The kernel assumes K is a multiple of BLOCK_K (128) so the K-loop can be
// fully unrolled per blockwise scale, and M/N multiples of 128 so the tile
// edges align with the FP8 scale grid (1×128 for A, 128×128 for B).
bool turbo_gemm_blockwise_fp8_supported(int32_t m, int32_t n, int32_t k) {
    constexpr int32_t BLOCK = 128;
    return (m % BLOCK == 0) && (n % BLOCK == 0) && (k % BLOCK == 0) && m > 0 && n > 0 && k > 0;
}

// ── Launcher ──
template <typename AType, typename BType, typename CType>
void turbo_gemm_blockwise_fp8_impl(const AType *a_ptr, const BType *b_ptr,
                                   const float *a_scale_ptr, const float *b_scale_ptr,
                                   CType *c_ptr, int32_t m, int32_t n, int32_t k,
                                   hipStream_t stream) {
    constexpr int BLOCK_M = 128, BLOCK_N = 128;
    dim3          grid((m + BLOCK_M - 1) / BLOCK_M, (n + BLOCK_N - 1) / BLOCK_N);
    dim3          block(256);
    turbo::turbo_gemm_blockwise_fp8_128x128x128_16x16x128_4wave_kernel<AType, BType, CType>
        <<<grid, block, 0, stream>>>(a_ptr, b_ptr, a_scale_ptr, b_scale_ptr, c_ptr,
                                     static_cast<uint32_t>(m), static_cast<uint32_t>(n),
                                     static_cast<uint32_t>(k));
}

// ── Explicit instantiations ──
//
// FP8 e4m3 / e5m2 inputs paired with FP16 / BF16 outputs. Mixed-format A/B
// pairs cover NT GEMM where A and B may carry different FP8 encodings (rare
// in practice but used in some grouped-GEMM flows).
#define INSTANTIATE_TURBO_GEMM_BLOCKWISE(A, B, C)                                                  \
    template void turbo_gemm_blockwise_fp8_impl<A, B, C>(const A *, const B *, const float *,     \
                                                         const float *, C *, int32_t, int32_t,    \
                                                         int32_t, hipStream_t);

INSTANTIATE_TURBO_GEMM_BLOCKWISE(dtype::float8_e4m3, dtype::float8_e4m3, dtype::float16)
INSTANTIATE_TURBO_GEMM_BLOCKWISE(dtype::float8_e4m3, dtype::float8_e4m3, dtype::bfloat16)
INSTANTIATE_TURBO_GEMM_BLOCKWISE(dtype::float8_e5m2, dtype::float8_e5m2, dtype::float16)
INSTANTIATE_TURBO_GEMM_BLOCKWISE(dtype::float8_e5m2, dtype::float8_e5m2, dtype::bfloat16)
INSTANTIATE_TURBO_GEMM_BLOCKWISE(dtype::float8_e4m3, dtype::float8_e5m2, dtype::float16)
INSTANTIATE_TURBO_GEMM_BLOCKWISE(dtype::float8_e4m3, dtype::float8_e5m2, dtype::bfloat16)
INSTANTIATE_TURBO_GEMM_BLOCKWISE(dtype::float8_e5m2, dtype::float8_e4m3, dtype::float16)
INSTANTIATE_TURBO_GEMM_BLOCKWISE(dtype::float8_e5m2, dtype::float8_e4m3, dtype::bfloat16)

#undef INSTANTIATE_TURBO_GEMM_BLOCKWISE

} // namespace primus_turbo
