// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
//
// See LICENSE for license information.

#pragma once

#include <hip/hip_fp8.h>
#include <type_traits>

#include "primus_turbo/dtype.h"

namespace primus_turbo::device {

// ── FP8 format code mapping ──
// Maps C++ FP8 types to cbsz/blgp encoding for v_mfma_scale_f32_*_f8f6f4:
//   0 = FP8 e4m3,  1 = FP8 e5m2
//   2 = FP6 e2m3,  3 = FP6 e3m2,  4 = FP4 e2m1  (future)
template <typename T>
inline constexpr int fp8_format_code =
    (std::is_same_v<T, __hip_fp8_e5m2> || std::is_same_v<T, dtype::float8_e5m2>) ? 1 : 0;

// ── v_mfma_scale_f32_16x16x128_f8f6f4 (gfx950) ──
// Scaled MFMA: D = A * B * scale, with microscaling (MX) support.
//   AType/BType: FP8 element types (determines cbsz/blgp encoding)
//   M=16, N=16, K=128, output f32x4
template <typename AType, typename BType> struct mfma_scale_f32_16x16x128_f8f6f4 {
    static constexpr int cbsz = fp8_format_code<AType>;
    static constexpr int blgp = fp8_format_code<BType>;

    // Pinned registers, accumulator in AGPR.
    // PIN_A/PIN_B: VGPR start for A/B data (8 VGPRs each)
    // PIN_ACC:     AGPR start for accumulator (4 AGPRs)
    // PIN_SA/PIN_SB: VGPR for A/B scale (1 VGPR each)
    // clang-format off
    template <int PIN_A, int PIN_B, int PIN_ACC, int PIN_SA, int PIN_SB>
    __device__ __forceinline__ static void run_pinned_acc_agpr() {
#if defined(__gfx950__)
        if constexpr (cbsz == 0 && blgp == 0)
            asm volatile("v_mfma_scale_f32_16x16x128_f8f6f4 a[%0:%1], v[%2:%3], v[%4:%5], a[%0:%1], v[%6], v[%7] op_sel_hi:[0,0,0]"
                : : "n"(PIN_ACC), "n"(PIN_ACC + 3), "n"(PIN_A), "n"(PIN_A + 7), "n"(PIN_B), "n"(PIN_B + 7), "n"(PIN_SA), "n"(PIN_SB));
        else if constexpr (cbsz == 1 && blgp == 0)
            asm volatile("v_mfma_scale_f32_16x16x128_f8f6f4 a[%0:%1], v[%2:%3], v[%4:%5], a[%0:%1], v[%6], v[%7] op_sel_hi:[0,0,0] cbsz:1"
                : : "n"(PIN_ACC), "n"(PIN_ACC + 3), "n"(PIN_A), "n"(PIN_A + 7), "n"(PIN_B), "n"(PIN_B + 7), "n"(PIN_SA), "n"(PIN_SB));
        else if constexpr (cbsz == 0 && blgp == 1)
            asm volatile("v_mfma_scale_f32_16x16x128_f8f6f4 a[%0:%1], v[%2:%3], v[%4:%5], a[%0:%1], v[%6], v[%7] op_sel_hi:[0,0,0] blgp:1"
                : : "n"(PIN_ACC), "n"(PIN_ACC + 3), "n"(PIN_A), "n"(PIN_A + 7), "n"(PIN_B), "n"(PIN_B + 7), "n"(PIN_SA), "n"(PIN_SB));
        else
            asm volatile("v_mfma_scale_f32_16x16x128_f8f6f4 a[%0:%1], v[%2:%3], v[%4:%5], a[%0:%1], v[%6], v[%7] op_sel_hi:[0,0,0] cbsz:1 blgp:1"
                : : "n"(PIN_ACC), "n"(PIN_ACC + 3), "n"(PIN_A), "n"(PIN_A + 7), "n"(PIN_B), "n"(PIN_B + 7), "n"(PIN_SA), "n"(PIN_SB));
#else
        static_assert(false, "mfma_scale_f32_16x16x128_f8f6f4 requires gfx950");
#endif
    }

    // Pinned registers, accumulator in VGPR.
    template <int PIN_A, int PIN_B, int PIN_ACC, int PIN_SA, int PIN_SB>
    __device__ __forceinline__ static void run_pinned_acc_vgpr() {
#if defined(__gfx950__)
        if constexpr (cbsz == 0 && blgp == 0)
            asm volatile("v_mfma_scale_f32_16x16x128_f8f6f4 v[%0:%1], v[%2:%3], v[%4:%5], v[%0:%1], v[%6], v[%7] op_sel_hi:[0,0,0]"
                : : "n"(PIN_ACC), "n"(PIN_ACC + 3), "n"(PIN_A), "n"(PIN_A + 7), "n"(PIN_B), "n"(PIN_B + 7), "n"(PIN_SA), "n"(PIN_SB));
        else if constexpr (cbsz == 1 && blgp == 0)
            asm volatile("v_mfma_scale_f32_16x16x128_f8f6f4 v[%0:%1], v[%2:%3], v[%4:%5], v[%0:%1], v[%6], v[%7] op_sel_hi:[0,0,0] cbsz:1"
                : : "n"(PIN_ACC), "n"(PIN_ACC + 3), "n"(PIN_A), "n"(PIN_A + 7), "n"(PIN_B), "n"(PIN_B + 7), "n"(PIN_SA), "n"(PIN_SB));
        else if constexpr (cbsz == 0 && blgp == 1)
            asm volatile("v_mfma_scale_f32_16x16x128_f8f6f4 v[%0:%1], v[%2:%3], v[%4:%5], v[%0:%1], v[%6], v[%7] op_sel_hi:[0,0,0] blgp:1"
                : : "n"(PIN_ACC), "n"(PIN_ACC + 3), "n"(PIN_A), "n"(PIN_A + 7), "n"(PIN_B), "n"(PIN_B + 7), "n"(PIN_SA), "n"(PIN_SB));
        else
            asm volatile("v_mfma_scale_f32_16x16x128_f8f6f4 v[%0:%1], v[%2:%3], v[%4:%5], v[%0:%1], v[%6], v[%7] op_sel_hi:[0,0,0] cbsz:1 blgp:1"
                : : "n"(PIN_ACC), "n"(PIN_ACC + 3), "n"(PIN_A), "n"(PIN_A + 7), "n"(PIN_B), "n"(PIN_B + 7), "n"(PIN_SA), "n"(PIN_SB));
#else
        static_assert(false, "mfma_scale_f32_16x16x128_f8f6f4 requires gfx950");
#endif
    }
    // clang-format on

    // Builtin path: compiler-managed registers.
    // A/B are packed FP8 data (8 x int32 = 128 FP8 elements).
    // c is the accumulator (input & output). Returns updated accumulator.
    __device__ __forceinline__ static dtype::float32x4 run(dtype::int32x8 a, dtype::int32x8 b,
                                                           dtype::float32x4 c, uint32_t scale_a,
                                                           uint32_t scale_b) {
#if defined(__gfx950__)
        return __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(a, b, c, cbsz, blgp, 0, scale_a, 0,
                                                                scale_b);
#else
        static_assert(false, "mfma_scale_f32_16x16x128_f8f6f4 requires gfx950");
        return c;
#endif
    }
};

// ── v_mfma_f32_16x16x128_f8f6f4 (gfx950, no scale) ──
// Unscaled FP8/F8F6F4 MFMA used by blockwise FP8 GEMM where the FP32 scale is
// applied via a software promotion accumulator outside the MFMA pipeline.
//   AType/BType: FP8 element types (determines cbsz/blgp encoding)
//   M=16, N=16, K=128, output f32x4
template <typename AType, typename BType> struct mfma_f32_16x16x128_f8f6f4 {
    static constexpr int cbsz = fp8_format_code<AType>;
    static constexpr int blgp = fp8_format_code<BType>;

    // Pinned registers, accumulator in AGPR.
    // PIN_A/PIN_B: VGPR start for A/B data (8 VGPRs each)
    // PIN_ACC:     AGPR start for accumulator (4 AGPRs)
    // clang-format off
    template <int PIN_A, int PIN_B, int PIN_ACC>
    __device__ __forceinline__ static void run_pinned_acc_agpr() {
#if defined(__gfx950__)
        if constexpr (cbsz == 0 && blgp == 0)
            asm volatile("v_mfma_f32_16x16x128_f8f6f4 a[%0:%1], v[%2:%3], v[%4:%5], a[%0:%1]"
                : : "n"(PIN_ACC), "n"(PIN_ACC + 3), "n"(PIN_A), "n"(PIN_A + 7), "n"(PIN_B), "n"(PIN_B + 7));
        else if constexpr (cbsz == 1 && blgp == 0)
            asm volatile("v_mfma_f32_16x16x128_f8f6f4 a[%0:%1], v[%2:%3], v[%4:%5], a[%0:%1] cbsz:1"
                : : "n"(PIN_ACC), "n"(PIN_ACC + 3), "n"(PIN_A), "n"(PIN_A + 7), "n"(PIN_B), "n"(PIN_B + 7));
        else if constexpr (cbsz == 0 && blgp == 1)
            asm volatile("v_mfma_f32_16x16x128_f8f6f4 a[%0:%1], v[%2:%3], v[%4:%5], a[%0:%1] blgp:1"
                : : "n"(PIN_ACC), "n"(PIN_ACC + 3), "n"(PIN_A), "n"(PIN_A + 7), "n"(PIN_B), "n"(PIN_B + 7));
        else
            asm volatile("v_mfma_f32_16x16x128_f8f6f4 a[%0:%1], v[%2:%3], v[%4:%5], a[%0:%1] cbsz:1 blgp:1"
                : : "n"(PIN_ACC), "n"(PIN_ACC + 3), "n"(PIN_A), "n"(PIN_A + 7), "n"(PIN_B), "n"(PIN_B + 7));
#else
        static_assert(false, "mfma_f32_16x16x128_f8f6f4 requires gfx950");
#endif
    }
    // clang-format on

    // Pinned registers, accumulator in VGPR. Caller must zero-initialise the
    // 4 destination VGPRs before the first call to avoid accumulating residual
    // values; this variant is used by blockwise FP8 to keep the inner partial
    // in VGPR and dodge the MAI→VALU AGPR read-after-write hazard.
    // clang-format off
    template <int PIN_A, int PIN_B, int PIN_ACC>
    __device__ __forceinline__ static void run_pinned_acc_vgpr() {
#if defined(__gfx950__)
        if constexpr (cbsz == 0 && blgp == 0)
            asm volatile("v_mfma_f32_16x16x128_f8f6f4 v[%0:%1], v[%2:%3], v[%4:%5], v[%0:%1]"
                : : "n"(PIN_ACC), "n"(PIN_ACC + 3), "n"(PIN_A), "n"(PIN_A + 7), "n"(PIN_B), "n"(PIN_B + 7));
        else if constexpr (cbsz == 1 && blgp == 0)
            asm volatile("v_mfma_f32_16x16x128_f8f6f4 v[%0:%1], v[%2:%3], v[%4:%5], v[%0:%1] cbsz:1"
                : : "n"(PIN_ACC), "n"(PIN_ACC + 3), "n"(PIN_A), "n"(PIN_A + 7), "n"(PIN_B), "n"(PIN_B + 7));
        else if constexpr (cbsz == 0 && blgp == 1)
            asm volatile("v_mfma_f32_16x16x128_f8f6f4 v[%0:%1], v[%2:%3], v[%4:%5], v[%0:%1] blgp:1"
                : : "n"(PIN_ACC), "n"(PIN_ACC + 3), "n"(PIN_A), "n"(PIN_A + 7), "n"(PIN_B), "n"(PIN_B + 7));
        else
            asm volatile("v_mfma_f32_16x16x128_f8f6f4 v[%0:%1], v[%2:%3], v[%4:%5], v[%0:%1] cbsz:1 blgp:1"
                : : "n"(PIN_ACC), "n"(PIN_ACC + 3), "n"(PIN_A), "n"(PIN_A + 7), "n"(PIN_B), "n"(PIN_B + 7));
#else
        static_assert(false, "mfma_f32_16x16x128_f8f6f4 requires gfx950");
#endif
    }
    // clang-format on

    // Builtin path: compiler-managed registers.
    __device__ __forceinline__ static dtype::float32x4 run(dtype::int32x8 a, dtype::int32x8 b,
                                                           dtype::float32x4 c) {
#if defined(__gfx950__)
        // v_mfma_scale_f32_16x16x128_f8f6f4 with scale=0 is canonicalized to the
        // unscaled MFMA by the LLVM optimizer (PR #116724).
        return __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(a, b, c, cbsz, blgp, 0, 0u, 0, 0u);
#else
        static_assert(false, "mfma_f32_16x16x128_f8f6f4 requires gfx950");
        return c;
#endif
    }
};

// ── Zero a 4-VGPR range via v_mov_b32 ──
// Used to prepare the inner accumulator (VGPR-resident) for a fresh MFMA.
template <int PIN_VGPR>
__device__ __forceinline__ static void zero_vgpr_4() {
    asm volatile("v_mov_b32_e32 v[%0], 0" : : "n"(PIN_VGPR + 0));
    asm volatile("v_mov_b32_e32 v[%0], 0" : : "n"(PIN_VGPR + 1));
    asm volatile("v_mov_b32_e32 v[%0], 0" : : "n"(PIN_VGPR + 2));
    asm volatile("v_mov_b32_e32 v[%0], 0" : : "n"(PIN_VGPR + 3));
}

// ── Blockwise FP8 promotion accumulator (16×16×128 fragment) ──
// For blockwise FP8, FP32 scale cannot be consumed by the MFMA. Each MFMA
// produces an inner partial that lives in 4 VGPR; a post-MFMA promotion step
// folds the FP32 scale into a FP32 outer accumulator in 4 AGPR. The pattern
// per fragment is:
//
//   outer[i] += (a_scale[i] * b_scale) * inner[i]    for i in [0..3]
//
// Lane mapping for 16×16×128 fragment: each lane holds 4 FP32 partials at
// row = (lane_id / 16) × 4 + i and col = lane_id % 16. The four lane-local
// scales are pre-multiplied (a_scale × b_scale) so promotion only needs FMA.
//
// PIN_OUTER:          AGPR start for outer accumulator (4 AGPRs, in/out).
// PIN_INNER_VGPR:     VGPR start for inner partial (4 VGPRs, MFMA output).
// PIN_TMP_OUTER:      4-VGPR scratch range for AGPR→VGPR shuttling of outer.
// PIN_COMBINED_SCALE: VGPR holding 4 FP32 combined scales for the lane.
template <int PIN_OUTER, int PIN_INNER_VGPR, int PIN_TMP_OUTER, int PIN_COMBINED_SCALE>
__device__ __forceinline__ static void apply_scale_promotion_16x16() {
#if defined(__gfx950__)
    // clang-format off
    // Read outer accumulator AGPR -> VGPR.
    asm volatile("v_accvgpr_read_b32 v[%0], a[%1]" : : "n"(PIN_TMP_OUTER + 0), "n"(PIN_OUTER + 0));
    asm volatile("v_accvgpr_read_b32 v[%0], a[%1]" : : "n"(PIN_TMP_OUTER + 1), "n"(PIN_OUTER + 1));
    asm volatile("v_accvgpr_read_b32 v[%0], a[%1]" : : "n"(PIN_TMP_OUTER + 2), "n"(PIN_OUTER + 2));
    asm volatile("v_accvgpr_read_b32 v[%0], a[%1]" : : "n"(PIN_TMP_OUTER + 3), "n"(PIN_OUTER + 3));
    // outer += combined_scale * inner_vgpr   (FP32 fma).
    asm volatile("v_fmac_f32_e32 v[%0], v[%1], v[%2]"
        : : "n"(PIN_TMP_OUTER + 0), "n"(PIN_COMBINED_SCALE + 0), "n"(PIN_INNER_VGPR + 0));
    asm volatile("v_fmac_f32_e32 v[%0], v[%1], v[%2]"
        : : "n"(PIN_TMP_OUTER + 1), "n"(PIN_COMBINED_SCALE + 1), "n"(PIN_INNER_VGPR + 1));
    asm volatile("v_fmac_f32_e32 v[%0], v[%1], v[%2]"
        : : "n"(PIN_TMP_OUTER + 2), "n"(PIN_COMBINED_SCALE + 2), "n"(PIN_INNER_VGPR + 2));
    asm volatile("v_fmac_f32_e32 v[%0], v[%1], v[%2]"
        : : "n"(PIN_TMP_OUTER + 3), "n"(PIN_COMBINED_SCALE + 3), "n"(PIN_INNER_VGPR + 3));
    // Write outer back AGPR.
    asm volatile("v_accvgpr_write_b32 a[%0], v[%1]" : : "n"(PIN_OUTER + 0), "n"(PIN_TMP_OUTER + 0));
    asm volatile("v_accvgpr_write_b32 a[%0], v[%1]" : : "n"(PIN_OUTER + 1), "n"(PIN_TMP_OUTER + 1));
    asm volatile("v_accvgpr_write_b32 a[%0], v[%1]" : : "n"(PIN_OUTER + 2), "n"(PIN_TMP_OUTER + 2));
    asm volatile("v_accvgpr_write_b32 a[%0], v[%1]" : : "n"(PIN_OUTER + 3), "n"(PIN_TMP_OUTER + 3));
    // clang-format on
#else
    static_assert(false, "apply_scale_promotion_16x16 requires gfx950");
#endif
}

} // namespace primus_turbo::device
