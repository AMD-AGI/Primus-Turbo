/***************************************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE for license information.
 **************************************************************************************************/

// Device-side MXFP4 (E2M1) group emit for AITER's A6W4 / A4W6 blob layout.
//
// A header because two packers need the same routine: the weight packer in
// quantization_mxfp4_gfx950.cu, and the MXFP6 packer's hybrid mode, which emits an MXFP6
// row direction and an MXFP4 column direction from one staged tile. That hybrid is what
// makes wgrad eligible for a mixed-format GEMM, since wgrad contracts the token dimension
// and its operands are a gradient and an activation rather than the weight.
//
// Only the emit lives here. Tiling, staging and launch geometry stay with each kernel,
// because they are tuned per packer and have no business being shared.
//
// gfx950-only: it uses the hardware FP4 conversion.

#pragma once

#include <hip/hip_bf16.h>
#include <hip/hip_runtime.h>

namespace primus_turbo {
namespace mxfp4_emit {

using bfloat16 = hip_bfloat16;

// Layout of AITER's packed MXFP4 blob. Shares MXFP6's 256-row tile, 128-K tile, guard
// tiles and E8M0 scale plane; differs only in the code plane, 16 compact bytes per group
// against MXFP6's 24 split over C0 and C1.
constexpr int kGroupSize       = 32;
constexpr int kTileRows        = 256;
constexpr int kKTile           = 128;
constexpr int kGroupsPerKTile  = kKTile / kGroupSize;
constexpr int kPackedTileBytes = 16384; // MXFP6's is 24576
constexpr int kScaleTileBytes  = 1024;
constexpr int kBytesPerBlock   = 16;

// 1/sqrt(32) rounded to bf16, not to the nearest float: the reference packers apply the
// rotation as a bf16 dot, so their normalisation carries bf16 precision.
constexpr float kHadamard32Norm = 0.1767578125f;
// E2M1 RCEIL block scale, ceil_pow2(amax / max_pos) with max_pos = 6.0.
constexpr float kFp4InvMaxPos = 1.0f / 6.0f;

using uint4_t  = uint32_t __attribute__((ext_vector_type(4)));
using bf16x2_t = __bf16 __attribute__((ext_vector_type(2)));

#ifndef MXFP4_ABLATE_BF16
#define MXFP4_ABLATE_BF16 0
#endif
#ifndef MXFP4_ABLATE_CVT
#define MXFP4_ABLATE_CVT 0
#endif

/*
 * Rotate, quantize and store one 32-value group.
 *
 * One thread owns the whole group, as in `mxfp6_emit_group` and for the same measured
 * reason: spreading a group over four lanes needs 42 cross-lane exchanges, and three of
 * the four lanes then discard their work at the store.
 *
 * The butterfly network is walked in the same order the four-lane reference walks it
 * (h = 1, 2, 4, 8, 16), so every floating-point addition sees the same operands and the
 * result is bit-exact rather than merely equivalent. The amax reduction is a max tree and
 * is order-independent.
 */
__device__ __forceinline__ void mxfp4_emit_group(float (&values)[kGroupSize],
                                                 const int64_t out_row, const int32_t group,
                                                 const int32_t nk_pad,
                                                 uint8_t *__restrict__ packed,
                                                 uint8_t *__restrict__ packed_scale) {
    // The normalisation goes in FIRST, before the butterfly -- which is where AITER's
    // MXFP4 packer puts it, and NOT where the MXFP6 packer puts it (MXFP6 multiplies the
    // rotated result at the end). The two orders differ in floating point, so copying the
    // MXFP6 emit here produces a blob that is close but not equal, and `gemm_a6w4` has no
    // way to detect the difference. Matching AITER is the contract.
#pragma unroll
    for (int i = 0; i < kGroupSize; ++i)
        values[i] *= kHadamard32Norm;

#pragma unroll
    for (int stage = 0; stage < 5; ++stage) {
        const int h = 1 << stage;
#pragma unroll
        for (int pair = 0; pair < kGroupSize / 2; ++pair) {
            const int   butterfly = pair / h;
            const int   offset    = pair % h;
            const int   i0        = butterfly * (2 * h) + offset;
            const int   i1        = i0 + h;
            const float x0        = values[i0];
            const float x1        = values[i1];
            values[i0]            = x0 + x1;
            values[i1]            = x0 - x1;
        }
    }

    // Ablation hooks. Diagnostics only -- each one makes the blob wrong, and they exist to
    // attribute the packer's time rather than to be shipped. See RESULTS_mxfp4_packer.md.
#ifndef MXFP4_ABLATE_BF16
#define MXFP4_ABLATE_BF16 0
#endif
#ifndef MXFP4_ABLATE_CVT
#define MXFP4_ABLATE_CVT 0
#endif
    // Round the rotated values through bf16. Also MXFP4-specific: the MXFP6 packer feeds
    // the conversion full fp32. E2M1 has so few levels that the extra rounding costs
    // nothing, but it moves values sitting near a code boundary, so omitting it shifts
    // roughly 1% of codes and a fifth of a percent of the block scales.
    // Round to bf16 and keep the group in that form for the rest of the emit -- sixteen
    // VGPRs of bf16x2 rather than thirty-two of f32.
    //
    // This is a register-pressure change, not a rounding change: AITER rounds here too,
    // and takes its amax from the rounded values, so the codes are identical either way.
    // What it buys is occupancy. The FP4 conversion consumes two values per call and
    // feeds its own accumulator, so a whole group is a chain of sixteen calls and every
    // value stays live across it -- where MXFP6 consumes all 32 in one
    // cvt_scalef32_2xpk16_fp6_f32 and frees them immediately. Holding the group as f32
    // cost 81 VGPRs and 5 waves/SIMD against the MXFP6 packer's 68 and 7, which on a
    // memory-bound kernel is the missing latency hiding. Forcing it with __launch_bounds__
    // only trades the registers for spills: 7 waves spills 8-19 VGPRs and measures 1.67x.
    bf16x2_t pairs[kGroupSize / 2];
#pragma unroll
    for (int i = 0; i < kGroupSize / 2; ++i) {
#if MXFP4_ABLATE_BF16
        pairs[i][0] = static_cast<__bf16>(0.0f);
        pairs[i][1] = static_cast<__bf16>(0.0f);
#else
        pairs[i][0] = static_cast<__bf16>(values[2 * i]);
        pairs[i][1] = static_cast<__bf16>(values[2 * i + 1]);
#endif
    }

    // Seeded at 1e-10 rather than 0, matching AITER's group_amax under RoundUp: it floors
    // the scale for an all-zero group so RCEIL cannot emit byte 0 there. Above that floor
    // it has no effect, so it never perturbs a real weight.
    float amax = 1.0e-10f;
#pragma unroll
    for (int i = 0; i < kGroupSize / 2; ++i) {
        amax = fmaxf(amax, fabsf(static_cast<float>(pairs[i][0])));
        amax = fmaxf(amax, fabsf(static_cast<float>(pairs[i][1])));
    }

    // RCEIL: ceil_pow2(amax / 6). Bump the exponent whenever any mantissa bit survives
    // the divide, which is what makes it a ceiling rather than a truncation.
    const uint32_t scaled   = __builtin_bit_cast(uint32_t, amax * kFp4InvMaxPos);
    uint32_t       exponent = (scaled >> 23) & 0xFFu;
    if (exponent < 0xFFu && (scaled & 0x7FFFFFu))
        exponent += 1;
    const uint8_t scale_exp = static_cast<uint8_t>(exponent);

    // E8M0 byte zero means the minimum scale 2^-127. An f32 with a zero exponent field is
    // numeric zero, so feed the hardware the normal/subnormal boundary instead.
    const uint32_t scale_bits =
        scale_exp == 0 ? 0x00400000u : static_cast<uint32_t>(scale_exp) << 23;
    const float conversion_scale = __builtin_bit_cast(float, scale_bits);

    // Four uint32 words, 8 values each, matching the four lanes of the reference packer:
    // word w carries values[8w .. 8w+8) and lands at byte offset 4w within the group.
    //
    // The conversion's last argument selects which 8-bit field of the accumulator the
    // pair lands in and must be a literal -- the compiler rejects an ordinary loop
    // induction variable there even under `#pragma unroll`, because the constraint is
    // checked in the frontend before unrolling. Hence the explicit four, matching the
    // reference packer's `static_for` over the same range.
    uint4_t words = {0u, 0u, 0u, 0u};
#if defined(__gfx950__) && !MXFP4_ABLATE_CVT
    if (amax != 0.0f) {
#pragma unroll
        for (int w = 0; w < 4; ++w) {
            const int p    = 4 * w;
            uint32_t  word = 0;
            word = __builtin_amdgcn_cvt_scalef32_pk_fp4_bf16(word, pairs[p + 0],
                                                             conversion_scale, 0);
            word = __builtin_amdgcn_cvt_scalef32_pk_fp4_bf16(word, pairs[p + 1],
                                                             conversion_scale, 1);
            word = __builtin_amdgcn_cvt_scalef32_pk_fp4_bf16(word, pairs[p + 2],
                                                             conversion_scale, 2);
            word = __builtin_amdgcn_cvt_scalef32_pk_fp4_bf16(word, pairs[p + 3],
                                                             conversion_scale, 3);
            words[w] = word;
        }
    }
#endif

    const int32_t tile_row  = static_cast<int32_t>(out_row / kTileRows);
    const int32_t rem       = static_cast<int32_t>(out_row % kTileRows);
    const int32_t row_block = rem / 16;
    const int32_t row16     = rem % 16;
    const int32_t step      = group / kGroupsPerKTile;
    const int32_t k_group   = group % kGroupsPerKTile;
    const int32_t block     = row_block * 64 + k_group * 16 + row16;
    const int64_t tile_base = (static_cast<int64_t>(tile_row) * nk_pad + step) * kPackedTileBytes;
    *reinterpret_cast<uint4_t *>(packed + tile_base + block * kBytesPerBlock) = words;

    const int32_t scale_upper = rem / 128;
    const int32_t scale_sub   = (rem % 128) / 16;
    const int64_t scale_address =
        (static_cast<int64_t>(tile_row) * nk_pad + step) * kScaleTileBytes + scale_upper * 512 +
        k_group * 128 + row16 * 8 + scale_sub;
    packed_scale[scale_address] = scale_exp;
}

} // namespace mxfp4_emit
} // namespace primus_turbo
