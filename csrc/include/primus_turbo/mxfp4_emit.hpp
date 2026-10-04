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

// 1/sqrt(32) at fp32 precision, the same constant as the MXFP6 packer: A6W4 pairs an
// MXFP4 operand with an MXFP6 one, so both rotations must carry the same normalisation
// for the product to be scaled by 32 c^2 = 1 (see quantization_mxfp6_gfx950.cu).
constexpr float kHadamard32Norm = 0.17677669529663687f;
// E2M1 RCEIL block scale, ceil_pow2(amax / max_pos) with max_pos = 6.0.
constexpr float kFp4InvMaxPos = 1.0f / 6.0f;

using uint4_t  = uint32_t __attribute__((ext_vector_type(4)));
using bf16x2_t = __bf16 __attribute__((ext_vector_type(2)));

// Where an emitted group lands. The quantization is identical for all three; only the store
// address differs, and in each layout a group's 16 code bytes stay one contiguous store.
//   A6W4Blob : AITER's A6W4 / A4W6 operand blob (256-row tiles, 128-K tiles, guard tiles).
//   A4W4A    : the A operand of AITER's f4gemm A4W4 kernels -- plain row-major fp4x2,
//              [rows, K/2], with the e8m0 scales in shuffle_scale()'s layout.
//   A4W4B    : the B operand of those kernels -- shuffle_weight(layout=(16, 16)): rows in
//              blocks of 16, each 32-byte K block stored as two 16-row x 16-byte halves --
//              with the same scale layout.
// For the two A4W4 layouts `nk` is ceil(K, 256) / 128 (no guard tiles), so a row is 64 * nk
// bytes and the scale plane has 4 * nk columns, a multiple of 8 as shuffle_scale requires.
enum class Layout { A6W4Blob, A4W4A, A4W4B, Plain, Fly };

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
// Stochastic rounding's random words. One avalanche hash per group, keyed by the launch seed
// and the group's (row, group) position, then a Weyl step per value pair, so no two pairs in
// a group share a word (Turbo's MXFP4 quantizer reuses one word per thread).
__device__ __forceinline__ uint32_t sr_mix(uint32_t x) {
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

// SR: round each value up or down to its FP4 neighbours with probability proportional to
// distance (v_cvt_scalef32_sr_pk_fp4_bf16), so a code is unbiased. The scale is unchanged.
// FlyDSL's packed-scale byte offset for (row, kblk): a port of `mxfp4_packed_scale_byte`
// (flydsl/utils/gemm_helper.py), checked byte-for-byte against `preshuffle_mxfp4_scales` on the
// Flux training shapes. n_sub = 2, nd = ng = 4; ku = 2 when (k128 / 2) is even, else 1.
struct FlyPackArgs {
    int32_t is_b, nt, ilv, k128, rows;
};
__device__ __forceinline__ int64_t fly_scale_byte(const int64_t row, const int32_t kblk,
                                                  const FlyPackArgs &f) {
    if (f.nt == 4 && (f.ilv == 0 || f.ilv == 4)) {
        // Every packer in use takes this path: the 256-wide N tile (nt = 4; the 192-wide one races)
        // with interleave 0 or 4. All divisors are then powers of two -- gspan 64, ilv 4, nw = 2 *
        // ku in {2, 4} -- so the mapping below is the generic one with its runtime divides and
        // modulos (three software division sequences per scale byte on gfx950) replaced by shifts
        // and masks. Same byte for every input.
        const uint32_t r32 = static_cast<uint32_t>(row);
        const uint32_t kk  = static_cast<uint32_t>(f.k128) >> 1;
        const uint32_t ku_shift =
            (kk & 1u) == 0 ? 1u : 0u;            // ku = 2 when (k128 / 2) is even, else 1
        const uint32_t nw_shift = ku_shift + 1u; // nw = 2 * ku
        const uint32_t kdw = static_cast<uint32_t>(kblk) >> 2, g = static_cast<uint32_t>(kblk) & 3u;
        const uint32_t kh = kdw >> nw_shift, rem = kdw & ((1u << nw_shift) - 1u);
        const uint32_t u = rem >> 1, lo = rem & 1u;
        const uint32_t grp = r32 >> 6, loc = r32 & 63u;
        uint32_t       wi, r_region;
        if (f.is_b) {
            r_region = (grp & 3u) >> 1;
            wi       = (grp >> 2) * 2u + (grp & 1u);
        } else {
            wi       = grp >> 1;
            r_region = grp & 1u;
        }
        const uint32_t r    = f.ilv ? (loc >> 2) : (loc & 15u);
        const uint32_t t    = f.ilv ? (loc & 3u) : (loc >> 4);
        const uint32_t last = r_region * 2u + lo;
        const int64_t  base = ((int64_t(wi) * kk + int64_t(kh << ku_shift)) * 64 + r) * 4;
        return (base + int64_t(u) * 256 + int64_t(g) * 64 + last) * 4 + t;
    }
    const int32_t ku = ((f.k128 / 2) % 2 == 0) ? 2 : 1;
    const int32_t nw = 2 * ku, kk = f.k128 / 2;
    const int32_t kdw = kblk / 4, g = kblk % 4;
    const int32_t kh = kdw / nw, rem = kdw % nw;
    const int32_t u = rem / 2, lo = rem % 2;
    const int64_t gspan = 16 * f.nt;
    const int64_t grp = row / gspan, loc = row % gspan;
    int64_t       wi, r_region;
    if (f.is_b) {
        const int64_t blk = grp / 4, off = grp % 4;
        r_region = off / 2;
        wi       = blk * 2 + off % 2;
    } else {
        wi       = grp / 2;
        r_region = grp % 2;
    }
    int64_t r, t;
    if (f.ilv) {
        r = loc / f.ilv;
        t = loc % f.ilv;
    } else {
        t = loc / 16;
        r = loc % 16;
    }
    const int64_t last = r_region * 2 + lo;
    const int64_t base = ((wi * kk + int64_t(kh) * ku) * 64 + r) * 4;
    return (base + int64_t(u) * 256 + int64_t(g) * 64 + last) * 4 + t;
}

template <Layout LAYOUT = Layout::A6W4Blob, bool SR = false>
__device__ __forceinline__ void
mxfp4_emit_group(float (&values)[kGroupSize], const int64_t out_row, const int32_t group,
                 const int32_t nk_pad, uint8_t *__restrict__ packed,
                 uint8_t *__restrict__ packed_scale, const uint32_t sr_seed = 0,
                 const FlyPackArgs fly = {}) {
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
    if (SR && amax != 0.0f) {
        // The SR conversion does not preserve the accumulator's other bytes the way the RTN
        // one does (probed on gfx950: with sel=1 byte 0 is cleared and the upper half holds
        // stale register contents), so each pair converts into byte 0 of a fresh register and
        // is shifted into place. The instruction takes its random bits from the top of the seed
        // (bit 31 decides the first value of a pair, bit 30 the second), hence a fresh
        // well-mixed word per pair.
        uint32_t rng = sr_mix(sr_seed ^ sr_mix(static_cast<uint32_t>(out_row) * 0x9e3779b1u ^
                                               static_cast<uint32_t>(group)));
#pragma unroll
        for (int w = 0; w < 4; ++w) {
            uint32_t b[4];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                // Weyl step: one add per pair. Marginally uniform from the hashed base, which
                // is all unbiased rounding needs; a xorshift per pair costs noticeably more packer
                // time.
                rng += 0x9e3779b9u;
                // Inline asm, not the builtin: the compiler folds builtin + mask + shift back
                // into the byte-select form (old = word, sel = j), which this instruction
                // does not honour -- measured, only every fourth byte survived. Not volatile:
                // it is pure, and the scheduler should interleave it.
                const uint32_t src = __builtin_bit_cast(uint32_t, pairs[4 * w + j]);
                asm("v_cvt_scalef32_sr_pk_fp4_bf16 %0, %1, %2, %3"
                    : "=v"(b[j])
                    : "v"(src), "v"(rng), "v"(conversion_scale));
            }
            // Byte 0 of each result, assembled with three v_perm_b32 (selector bytes 0-3 pick
            // from the second operand, 4-7 from the first, 0x0c is zero).
            const uint32_t lo = __builtin_amdgcn_perm(b[1], b[0], 0x0c0c0400u);
            const uint32_t hi = __builtin_amdgcn_perm(b[3], b[2], 0x0c0c0400u);
            words[w]          = __builtin_amdgcn_perm(hi, lo, 0x05040100u);
        }
    } else if (amax != 0.0f) {
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

    if constexpr (LAYOUT == Layout::A6W4Blob) {
        const int32_t tile_row  = static_cast<int32_t>(out_row / kTileRows);
        const int32_t rem       = static_cast<int32_t>(out_row % kTileRows);
        const int32_t row_block = rem / 16;
        const int32_t row16     = rem % 16;
        const int32_t step      = group / kGroupsPerKTile;
        const int32_t k_group   = group % kGroupsPerKTile;
        const int32_t block     = row_block * 64 + k_group * 16 + row16;
        const int64_t tile_base =
            (static_cast<int64_t>(tile_row) * nk_pad + step) * kPackedTileBytes;
        *reinterpret_cast<uint4_t *>(packed + tile_base + block * kBytesPerBlock) = words;

        const int32_t scale_upper = rem / 128;
        const int32_t scale_sub   = (rem % 128) / 16;
        const int64_t scale_address =
            (static_cast<int64_t>(tile_row) * nk_pad + step) * kScaleTileBytes + scale_upper * 512 +
            k_group * 128 + row16 * 8 + scale_sub;
        packed_scale[scale_address] = scale_exp;
    } else {
        // Codes. Row stride 64 * nk bytes; group g is bytes [16 g, 16 g + 16) of its row.
        const int64_t row_bytes = static_cast<int64_t>(nk_pad) * 64;
        int64_t       address;
        if constexpr (LAYOUT == Layout::A4W4A || LAYOUT == Layout::Plain || LAYOUT == Layout::Fly) {
            address = out_row * row_bytes + static_cast<int64_t>(group) * kBytesPerBlock;
        } else {
            // shuffle_weight((16, 16)): view (N/16, 16, Kb/32, 2, 16) -> permute (0, 2, 3, 1, 4).
            const int64_t n0     = out_row / 16;
            const int64_t r16    = out_row % 16;
            const int64_t kblock = group / 2; // 32-byte K block
            const int64_t half   = group % 2; // which 16 bytes of it
            address = (((n0 * (row_bytes / 32) + kblock) * 2 + half) * 16 + r16) * kBytesPerBlock;
        }
        *reinterpret_cast<uint4_t *>(packed + address) = words;

        // Scales. Plain: row-major [rows, K/32]. Otherwise shuffle_scale(): view (Mp/32, 2, 16,
        // Sp/8, 2, 4) -> permute (0, 3, 5, 2, 4, 1), with Sp = K/32 = 4 * nk.
        const int64_t sp = static_cast<int64_t>(nk_pad) * 4;
        if constexpr (LAYOUT == Layout::Plain) {
            packed_scale[out_row * sp + group] = scale_exp;
            return;
        }
        if constexpr (LAYOUT == Layout::Fly) {
            if (out_row < fly.rows && group < fly.k128 * 4)
                packed_scale[fly_scale_byte(out_row, group, fly)] = scale_exp;
            return;
        }
        const int64_t i0 = out_row / 32, i1 = (out_row % 32) / 16, i2 = out_row % 16;
        const int64_t j0 = group / 8, j1 = (group % 8) / 4, j2 = group % 4;
        const int64_t scale_address =
            ((((i0 * (sp / 8) + j0) * 4 + j2) * 16 + i2) * 2 + j1) * 2 + i1;
        packed_scale[scale_address] = scale_exp;
    }
}

} // namespace mxfp4_emit
} // namespace primus_turbo
