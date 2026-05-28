// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
//
// See LICENSE for license information.

#pragma once

#include "primus_turbo/common.h"
#include "primus_turbo/platform.h"

namespace primus_turbo {

// CTA fan-out per CU for the rmsnorm grid. The launcher computes
// `target_ctas = num_cus * RMSNORM_CTAS_PER_CU` and uses it as the cap on
// kernel grid size and (for backward) on the dgamma_part workspace rows.
//
// Trade-off: more CTAs expose more parallelism per call, but also grow the
// fp32 dgamma_part scratch (`target_ctas * cols * 4 bytes`) and the cost of
// the finalize reduction. 4 was empirically the best compromise on CDNA3
// across the bench grid (cols 64..8192, tokens 1k..64k); going lower starves
// large workloads, going higher wastes scratch on small ones.
constexpr int RMSNORM_CTAS_PER_CU = 4;

// Warps packed per CTA in the warp-per-row fast path (small inner_len). Used
// by both fwd and bwd stage 0, and by the host-side dgamma_part sizing. Must
// divide RMSNORM_CTAS_PER_CU so that `target_ctas / RMSNORM_WARPS_PER_BLOCK`
// is exact — keeps the workspace upper bound and the actual `n_parts` in
// agreement without a separate round-up.
constexpr int RMSNORM_WARPS_PER_BLOCK = 4;
static_assert(RMSNORM_CTAS_PER_CU % RMSNORM_WARPS_PER_BLOCK == 0,
              "RMSNORM_CTAS_PER_CU must be a multiple of RMSNORM_WARPS_PER_BLOCK so that the "
              "dgamma_part workspace bound matches the warp-per-row grid math.");

// Block-size selector shared by fwd and bwd block-per-row kernels: smallest
// multiple of warp size that covers `cols / unroll` active threads, capped at
// MAX_THREADS_PER_BLOCK.
inline int rmsnorm_pick_blocksize(int64_t cols, int unroll) {
    int64_t needed = (cols + unroll - 1) / unroll;
    if (needed > MAX_THREADS_PER_BLOCK)
        needed = MAX_THREADS_PER_BLOCK;
    int rounded =
        static_cast<int>(((needed + THREADS_PER_WARP - 1) / THREADS_PER_WARP) * THREADS_PER_WARP);
    if (rounded < THREADS_PER_WARP)
        rounded = THREADS_PER_WARP;
    return rounded;
}

// CDNA hard occupancy ceiling: 32 wavefronts per CU. (CDNA1/2/3/4 all set
// this; NVIDIA equivalents are 64 warps × 32-thread / 1024-thread limit which
// works out to the same 2 CTAs/SM at block=1024.) Block size sets a hard
// upper bound on CTAs/CU regardless of register/smem pressure.
constexpr int RMSNORM_MAX_WAVES_PER_CU = 32;

// Effective CTAs-per-CU for an rmsnorm launch: the desired
// RMSNORM_CTAS_PER_CU (scratch/parallelism trade-off) clamped to the
// architectural occupancy ceiling for this block size. At block=1024 (used
// for cols ≥ 8K bf16/fp16 or cols ≥ 4K fp32) the ceiling is 2 CTAs/CU, so
// over-allocating dgamma_part to fit 4 CTAs/CU just wastes HBM bandwidth.
// Mirrors TE's `cudaOccupancyMaxActiveBlocksPerMultiprocessor` flow without
// the runtime API call, since our kernel resource footprint is regular.
inline int rmsnorm_effective_ctas_per_cu(int block_threads) {
    const int waves_per_cta   = (block_threads + THREADS_PER_WARP - 1) / THREADS_PER_WARP;
    const int occupancy_limit = RMSNORM_MAX_WAVES_PER_CU / waves_per_cta;
    const int effective =
        occupancy_limit < RMSNORM_CTAS_PER_CU ? occupancy_limit : RMSNORM_CTAS_PER_CU;
    return effective < 1 ? 1 : effective;
}

// Forward: writes y = gamma * x / sqrt(mean(x^2) + eps), and rs = 1/sqrt(...)
// per-row (fp32). Saving rs lets backward skip recomputing the mean square.
// `target_ctas` caps the kernel grid; each block grid-strides over rows.
template <typename T>
void rmsnorm_fwd_impl(const T *input, const T *gamma, T *output, float *rs, const int64_t inner_len,
                      const int64_t outer_len, const float epsilon, const int64_t target_ctas,
                      hipStream_t stream);

// Backward stage 0: writes dx and a partial dgamma of shape [n_parts, cols] (fp32).
// `n_parts` is the actual grid x-dim used (≤ target_ctas); the caller passes
// it back to finalize. The caller must ensure `dgamma_part` has at least
// `target_ctas` rows allocated.
template <typename T>
int64_t rmsnorm_bwd_stage0_impl(const T *input, const T *gamma, const T *grad_out, const float *rs,
                                T *grad_in, float *dgamma_part, const int64_t inner_len,
                                const int64_t outer_len, const int64_t target_ctas,
                                hipStream_t stream);

// Backward stage 1: reduce dgamma_part [n_parts, cols] -> dgamma [cols] in T.
template <typename T>
void rmsnorm_bwd_finalize_impl(const float *dgamma_part, T *dgamma, const int64_t cols,
                               const int64_t n_parts, hipStream_t stream);

} // namespace primus_turbo
