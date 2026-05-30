// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
//
// See LICENSE for license information.

#pragma once

#include <hip/hip_fp8.h>
#include <type_traits>

#include "primus_turbo/dtype.h"

namespace primus_turbo::device {

// ── FP8/FP6/FP4 format code mapping ──
// Maps C++ low-precision types to cbsz/blgp encoding for v_mfma_scale_f32_*_f8f6f4:
//   0 = FP8 e4m3,  1 = FP8 e5m2
//   2 = FP6 e2m3,  3 = FP6 e3m2,  4 = FP4 e2m1
// e4m3 is the default (0). e5m2 -> 1. FP4 e2m1 -> 4 (the FP4-weight path: A stays
// FP8 e4m3, B is float4x2_e2m1 so blgp=4 selects the 4-bit operand on the K=128 MFMA).
// FP6 (e2m3/e3m2) codes 2/3 are reserved but their C++ types are not defined in this
// tree, so they are not mapped here (would not compile).
template <typename T>
inline constexpr int fp8_format_code =
    (std::is_same_v<T, __hip_fp8_e5m2> || std::is_same_v<T, dtype::float8_e5m2>)       ? 1
    : (std::is_same_v<T, __hip_fp4x2_e2m1> || std::is_same_v<T, dtype::float4x2_e2m1>) ? 4
                                                                                       : 0;

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

// ── v_mfma_scale_f32_32x32x64_f8f6f4 (gfx950) ──
// Scaled MFMA: D = A * B * scale, with microscaling (MX) support.
//   AType/BType: FP8 element types (determines cbsz/blgp encoding)
//   M=32, N=32, K=64, output f32x16 per lane (1024 / 64 lanes = 16).
//
// Preferred default over the 16x16x128 variant for L1/L2 GEMM tiles
// because the accumulator footprint is 16 AGPRs/lane (vs 4 AGPRs/lane
// for 16x16x128 — but the 16x16x128 variant needs 64 calls/lane for a
// 128x128 output tile, exhausting all 256 AGPRs and forcing 1 wave/SIMD
// occupancy).  With 32x32x64 the same 128x128 tile needs 16 calls/lane
// × 16 floats = 256 AGPRs — same total footprint per-tile, but
// distributed across fewer MFMA calls so pipeline slack and
// occupancy improve.  See TODO.md item 4 for the AGPR budget rationale.
template <typename AType, typename BType> struct mfma_scale_f32_32x32x64_f8f6f4 {
    static constexpr int cbsz = fp8_format_code<AType>;
    static constexpr int blgp = fp8_format_code<BType>;

    // Builtin path: compiler-managed registers.
    // A/B are packed FP8 data (8 x int32 = 128 FP8 elements; for K=64
    // the lower 4 dwords are consumed but the intrinsic accepts the
    // same int32x8 operand type as the 16x16x128 variant).
    // c is the accumulator (input & output) of 16 floats per lane.
    __device__ __forceinline__ static dtype::float32x16 run(dtype::int32x8 a, dtype::int32x8 b,
                                                            dtype::float32x16 c, uint32_t scale_a,
                                                            uint32_t scale_b) {
#if defined(__gfx950__)
        return __builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4(a, b, c, cbsz, blgp, 0, scale_a, 0,
                                                               scale_b);
#else
        static_assert(false, "mfma_scale_f32_32x32x64_f8f6f4 requires gfx950");
        return c;
#endif
    }
};

} // namespace primus_turbo::device
