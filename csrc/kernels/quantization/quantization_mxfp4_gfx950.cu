/******************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE for license information.
 ******************************************************************************/

/*
 * MXFP4 (E2M1) quantize + pack into AITER's A6W4 weight-operand blob layout.
 *
 * Why this exists next to the MXFP6 packer rather than inside it
 * --------------------------------------------------------------
 * A6W4 contracts MXFP6 activations against MXFP4 weights. The activation side is
 * unchanged -- `gemm_a6w4` consumes exactly the blob `quantize_mxfp6_*` already produces,
 * because upstream built A6W4 on the same H32-rotated packing contract. Only the weight
 * operand is new.
 *
 * Training needs that weight in *both* contraction directions every microbatch: row for
 * the forward (`out = x @ w.T`, contracting K) and column for dgrad (`grad_x = g @ w`,
 * contracting N). AITER ships `quant_mxfp4_gemm`, but it packs the row direction only, so
 * the column direction there means materialising `w.T` first. That is the same transpose
 * the MXFP6 column packer was written to avoid, and the reason this file exists.
 *
 * wgrad is deliberately absent. `grad_w = g_col @ x_col` contracts M, so neither operand
 * is the weight and there is no MXFP4 blob to produce -- that GEMM stays A6W6.
 *
 * Relationship to the MXFP6 packer
 * --------------------------------
 * The blob geometry is shared: same 256-row tile, same 128-K tile, same two guard K
 * tiles, same E8M0 scale plane and the *same* block index arithmetic. Only two things
 * differ, and both are local to `mxfp4_emit_group`:
 *
 *   * the code plane is 16 compact bytes per group instead of MXFP6's 24 split across a
 *     C0 plane and a C1 plane, so `kPackedTileBytes` is 16384 rather than 24576;
 *   * the E8M0 scale is `ceil_pow2(amax / 6)` -- RCEIL against E2M1's max_pos of 6.0 --
 *     where MXFP6 derives its exponent straight from the amax against E2M3's 7.5.
 *
 * The Hadamard is bit-for-bit the MXFP6 one, including running all five butterfly stages
 * in-lane rather than splitting the last two across lanes, and including the bf16-rounded
 * 1/sqrt(32). Both properties are load-bearing: the blob has to match what AITER's own
 * packer would have produced, and `test_quantize_mxfp4_parity` pins that.
 *
 * Not ported: the extreme-group repair
 * ------------------------------------
 * AITER's MXFP4 packer carries the `hadamard32<true>` + `kHadamardSafetyShift` repair for
 * groups whose rotation overflows fp32. It is not reproduced here, matching the MXFP6
 * packer's existing position -- internal measurements confine the MXFP6 divergence to
 * inputs at or above 2^125. MXFP4's narrower
 * range means that threshold has to be re-measured rather than assumed to carry over;
 * `probe_extreme_group.py` is the tool. Until then the bound is: weights at or above
 * ~1e37 pack differently from AITER, and no Flux weight is within thirty orders of that.
 */

#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <type_traits>

#include "primus_turbo/arch.h"
#include "primus_turbo/common.h"
#include "primus_turbo/quantization.h"

namespace primus_turbo {

using bfloat16 = hip_bfloat16;

namespace {

// ---------------------------------------------------------------------------
// Layout constants. These describe AITER's packed weight blob and cannot be retuned:
// the A6W4 assembly derives its strides from them. Everything except the two code-plane
// entries is shared with MXFP6 by construction.
// ---------------------------------------------------------------------------
constexpr int kGroupSize       = 32;  // values per E8M0 scale, and the Hadamard size
constexpr int kTileRows        = 256; // rows per packed tile
constexpr int kKTile           = 128; // K values per packed tile
constexpr int kGroupsPerKTile  = kKTile / kGroupSize; // 4
constexpr int kPackedTileBytes = 16384;               // MXFP6's is 24576
constexpr int kScaleTileBytes  = 1024;
constexpr int kBytesPerBlock   = 16; // 32 values x 4 bits, one compact plane

// Same constant, same reason, as the MXFP6 packer: 1/sqrt(32) rounded to bf16 and NOT to
// the nearest float, because the reference packers apply the rotation as a bf16 dot.
constexpr float kHadamard32Norm = 0.1767578125f;

// E2M1 RCEIL block scale: ceil_pow2(amax / max_pos) with max_pos = 6.0. This is
// MxDtypeConfig<FP4_E2M1>::inv_max_pos under MxScaleRoundMode::RoundUp in AITER's
// csrc/include/mx_quant_utils.h, which is its default round mode.
constexpr float kFp4InvMaxPos = 1.0f / 6.0f;

// Blocking. Overridable so the shape can be swept without editing, exactly as the MXFP6
// packer parametrises MXFP6_TILE_M / MXFP6_TILE_N / MXFP6_THREADS_PER_BLOCK. The defaults
// below are the measured winners; see RESULTS_mxfp4_packer.md for the sweep.
//
// TILE_M and TILE_N are multiples of kGroupSize so a staged patch holds whole 32-value
// groups both ways, and divide 256 so a 256-aligned operand tiles exactly.
#ifndef MXFP4_TILE_M
#define MXFP4_TILE_M 64
#endif
#ifndef MXFP4_TILE_N
#define MXFP4_TILE_N 128
#endif
#ifndef MXFP4_THREADS_PER_BLOCK
#define MXFP4_THREADS_PER_BLOCK 256
#endif
constexpr int TILE_M            = MXFP4_TILE_M;
constexpr int kDefaultTileN     = MXFP4_TILE_N;
constexpr int THREADS_PER_BLOCK = MXFP4_THREADS_PER_BLOCK;

using uint4_t = uint32_t __attribute__((ext_vector_type(4)));
// Two bf16 in one VGPR, which is the form the FP4 conversion actually wants.
using bf16x2_t = __bf16 __attribute__((ext_vector_type(2)));

// 16-byte staging vector: the widest load that keeps one thread on one contiguous run of
// a row. s_tile rows are TILE_N * 2 bytes, a multiple of 16, so the shared-memory store is
// aligned whenever local_n is a multiple of kStageVec.
constexpr int kStageVec = 8;
using stage_vec_t = uint16_t __attribute__((ext_vector_type(kStageVec)));

// Pad each LDS row so the column gather does not serialise on one bank. A compact
// TILE_N=64 row of uint16 is 128 bytes, which is exactly the width of the 32 4-byte banks,
// so `s_tile[i][n]` for consecutive `i` lands in the *same* bank every time -- a 32-way
// conflict on the direction that exists to avoid materialising a transpose. Eight uint16
// of padding shifts consecutive rows 4 banks apart and keeps every row 16-byte aligned for
// the vector staging store. The MXFP6 packer pads for the same reason (LDS_PAD there).
constexpr int LDS_PAD = 8;

// Direct-to-LDS staging. `buffer_load_lds` writes HBM straight into LDS without the data
// passing through VGPRs, which is what lets the MXFP6 packer sit at HBM bandwidth. Without
// it this kernel reached only about two thirds of MXFP6's bandwidth on the same read and a smaller
// write -- the gap was never the arithmetic (ablating the whole FP4 conversion and the
// bf16 rounding moved the total by 0.7%) and never LDS bank conflicts.
//
// The instruction lays each wave's lane payloads out contiguously, so it needs the compact
// pitch; the padded pitch is kept for the fallback, where the column gather would otherwise
// serialise on one bank.
#ifndef MXFP4_ASYNC_STAGE
#define MXFP4_ASYNC_STAGE 1
#endif
constexpr bool kAsyncStage = MXFP4_ASYNC_STAGE && TILE_M == 64 &&
                             (kDefaultTileN == 64 || kDefaultTileN == 128) &&
                             THREADS_PER_BLOCK == 256;
constexpr int LDS_PITCH = kAsyncStage ? MXFP4_TILE_N : (MXFP4_TILE_N + LDS_PAD);

using as3_uint32_ptr = uint32_t __attribute__((address_space(3))) *;
using int32x4_t      = int32_t __attribute__((ext_vector_type(4)));

// Clang's raw_ptr builtin only accepts 1/2/4-byte widths, while the LLVM intrinsic and the
// gfx950 ISA accept a 16-byte lane payload. Declared the same way the MXFP6 packer does.
extern "C" __device__ void llvm_amdgcn_raw_buffer_load_lds(
    int32x4_t resource, as3_uint32_ptr lds_base, int size, int voffset, int soffset, int offset,
    int aux) __asm("llvm.amdgcn.raw.buffer.load.lds");

__device__ __forceinline__ int32x4_t make_buffer_resource_vec(const void *ptr, uint32_t bytes) {
    const uint64_t address = reinterpret_cast<uint64_t>(ptr);
    return int32x4_t{static_cast<int32_t>(address), static_cast<int32_t>(address >> 32),
                     static_cast<int32_t>(bytes), 0x00020000};
}

__device__ __forceinline__ void async_load_lds_16(int32x4_t resource, as3_uint32_ptr lds_base,
                                                  int32_t byte_offset) {
    llvm_amdgcn_raw_buffer_load_lds(resource, lds_base, 16, byte_offset, 0, 0, 0);
}

constexpr int ceil_div(const int x, const int m) {
    return (x + m - 1) / m;
}

// Identical to the MXFP6 packer's: an fp16 input is rounded through bf16 first, because
// the Hadamard is a bf16 dot with fp32 accumulate and skipping that drifts the codes.
template <typename DType> __device__ __forceinline__ float to_dot_operand(const uint16_t bits) {
    if constexpr (std::is_same_v<DType, bfloat16>) {
        const uint32_t widened = static_cast<uint32_t>(bits) << 16;
        return __builtin_bit_cast(float, widened);
    } else {
        const float value = __half2float(__builtin_bit_cast(half, bits));
        return static_cast<float>(static_cast<bfloat16>(value));
    }
}

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

/*
 * Stage a TILE_M x TILE_N patch in LDS, then emit row groups, column groups or both.
 *
 * The grid covers the operand padded to 256 in both dimensions rather than its logical
 * extent. That is not wasted work: the blob has to contain a well-defined encoding of zero
 * on the padding, and letting the out-of-bounds reads fall out of the LDS zero-fill
 * produces it for free -- the alternative is memsetting the whole blob on the host, which
 * costs more than the packing does.
 */
template <typename DType, bool DO_ROW, bool DO_COL, int TILE_N = kDefaultTileN>
__global__ __launch_bounds__(THREADS_PER_BLOCK) void quantize_mxfp4_kernel(
    const DType *__restrict__ input, uint8_t *__restrict__ row_packed,
    uint8_t *__restrict__ row_scale, uint8_t *__restrict__ col_packed,
    uint8_t *__restrict__ col_scale, const int32_t M, const int32_t N,
    const int32_t row_nk_pad, const int32_t col_nk_pad) {
    static_assert(TILE_N % kGroupSize == 0, "a staged patch must hold whole groups both ways");

    __shared__ uint16_t s_tile[TILE_M][LDS_PITCH];

    const int32_t tile_m = blockIdx.y * TILE_M;
    const int32_t tile_n = blockIdx.x * TILE_N;

    // Stage in 16-byte chunks. A scalar 2-byte load per element leaves this kernel about
    // 3x off HBM bandwidth -- measured 74 us against the MXFP6 packer's 26.6 us on
    // 12288x3072, for *less* traffic -- because the packer is memory bound and eight
    // half-width loads cost eight issues where one costs one. TILE_N and every Flux weight
    // dimension are multiples of kStageVec, so the vector path takes every full tile and
    // the scalar tail only runs on a ragged edge.
    const uint16_t *in16 = reinterpret_cast<const uint16_t *>(input);
    constexpr int   kChunksPerRow = TILE_N / kStageVec;

    // A block whose tile lies wholly inside the operand can take the direct-to-LDS path.
    // A ragged edge one cannot: `buffer_load_lds` has no per-lane predication here, and the
    // out-of-range rows have to read as zero rather than as whatever follows the tensor.
    const bool async_full_tile = tile_m + TILE_M <= M && tile_n + TILE_N <= N;
    if constexpr (kAsyncStage) {
        if (async_full_tile) {
            // Each pass stages THREADS_PER_BLOCK * kStageVec elements, i.e. that many
            // halves laid out as whole rows of TILE_N.
            constexpr int kAsyncStageRows = THREADS_PER_BLOCK * kStageVec / TILE_N;
            constexpr int kAsyncStages    = TILE_M / kAsyncStageRows;
            constexpr int kWaveRows       = 64 * kStageVec / TILE_N;
            constexpr int kVecsPerRow     = TILE_N / kStageVec;
            static_assert(TILE_M % kAsyncStageRows == 0, "TILE_M must divide into whole stages");

            const auto input_resource_vec = make_buffer_resource_vec(
                input, static_cast<uint32_t>(int64_t(M) * N * sizeof(DType)));
            const int wave = threadIdx.x >> 6;
#pragma unroll
            for (int stage = 0; stage < kAsyncStages; ++stage) {
                const int local_m  = threadIdx.x / kVecsPerRow;
                const int local_n  = (threadIdx.x % kVecsPerRow) * kStageVec;
                const int global_m = tile_m + stage * kAsyncStageRows + local_m;
                const int byte_offset =
                    static_cast<int>((int64_t(global_m) * N + tile_n + local_n) * sizeof(DType));
                const uintptr_t tile_lds_base = reinterpret_cast<uintptr_t>(
                    &s_tile[stage * kAsyncStageRows + wave * kWaveRows][0]);
                async_load_lds_16(input_resource_vec,
                                  reinterpret_cast<as3_uint32_ptr>(tile_lds_base), byte_offset);
            }
            __syncthreads();
            goto staged;
        }
    }

    for (int idx = threadIdx.x; idx < TILE_M * kChunksPerRow; idx += THREADS_PER_BLOCK) {
        const int     local_m  = idx / kChunksPerRow;
        const int     local_n  = (idx % kChunksPerRow) * kStageVec;
        const int32_t global_m = tile_m + local_m;
        const int32_t global_n = tile_n + local_n;

        stage_vec_t staged;
        if (global_m < M && global_n + kStageVec <= N) {
            staged = *reinterpret_cast<const stage_vec_t *>(
                &in16[static_cast<int64_t>(global_m) * N + global_n]);
        } else {
#pragma unroll
            for (int i = 0; i < kStageVec; ++i)
                staged[i] = (global_m < M && global_n + i < N)
                                ? in16[static_cast<int64_t>(global_m) * N + global_n + i]
                                : uint16_t{0};
        }
        *reinterpret_cast<stage_vec_t *>(&s_tile[local_m][local_n]) = staged;
    }
    __syncthreads();

staged:;

    const int slot = threadIdx.x;

    // Row direction: contract along N. Each staged row contributes TILE_N/32 groups.
    if constexpr (DO_ROW) {
        constexpr int kBlocksPerRow = TILE_N / kGroupSize;
        constexpr int kRowGroups    = TILE_M * kBlocksPerRow;
#pragma unroll
        for (int gi = slot; gi < kRowGroups; gi += THREADS_PER_BLOCK) {
            const int local_m  = gi / kBlocksPerRow;
            const int k_block  = gi % kBlocksPerRow;
            const int n_offset = k_block * kGroupSize;

            float values[kGroupSize];
#pragma unroll
            for (int i = 0; i < kGroupSize; ++i)
                values[i] = to_dot_operand<DType>(s_tile[local_m][n_offset + i]);

            mxfp4_emit_group(values, tile_m + local_m, tile_n / kGroupSize + k_block, row_nk_pad,
                             row_packed, row_scale);
        }
    }

    // Column direction: contract along M, i.e. pack the rows of x.T. Same emit, different
    // gather -- this is the transpose that no longer has to be materialised.
    if constexpr (DO_COL) {
        constexpr int kBlocksPerCol = TILE_M / kGroupSize;
        constexpr int kColGroups    = TILE_N * kBlocksPerCol;
#pragma unroll
        for (int gi = slot; gi < kColGroups; gi += THREADS_PER_BLOCK) {
            const int local_n  = gi / kBlocksPerCol;
            const int k_block  = gi % kBlocksPerCol;
            const int m_offset = k_block * kGroupSize;

            float values[kGroupSize];
#pragma unroll
            for (int i = 0; i < kGroupSize; ++i)
                values[i] = to_dot_operand<DType>(s_tile[m_offset + i][local_n]);

            mxfp4_emit_group(values, tile_n + local_n, tile_m / kGroupSize + k_block, col_nk_pad,
                             col_packed, col_scale);
        }
    }
}

struct launch_geometry {
    int  row_nk_pad;
    int  col_nk_pad;
    dim3 grid;
    dim3 block;
};

// Cover the operand padded to whole 256-row tiles in both directions: M is the row count
// of the row-direction blob and the K extent of the column-direction one, and vice versa
// for N, so both are rounded up before tiling. Rounding K to 256 rather than its own 128
// granularity can push one K-tile past the blob's real extent; the two mandatory guard
// tiles absorb it, exactly as in the MXFP6 packer.
launch_geometry geometry_for(const int M, const int N) {
    const int m_padded = ceil_div(M, kTileRows) * kTileRows;
    const int n_padded = ceil_div(N, kTileRows) * kTileRows;
    return {ceil_div(N, kKTile) + MXFP4_GUARD_K_TILES, ceil_div(M, kKTile) + MXFP4_GUARD_K_TILES,
            dim3(ceil_div(n_padded, kDefaultTileN), ceil_div(m_padded, TILE_M)),
            dim3(THREADS_PER_BLOCK)};
}

} // namespace

template <typename DType>
void quantize_mxfp4_impl(const DType *input, uint8_t *row_packed, uint8_t *row_scale,
                         uint8_t *col_packed, uint8_t *col_scale, const int32_t M, const int32_t N,
                         const MXFP4Direction direction, hipStream_t stream) {
    const launch_geometry g = geometry_for(M, N);

    switch (direction) {
    case MXFP4Direction::Row:
        quantize_mxfp4_kernel<DType, true, false><<<g.grid, g.block, 0, stream>>>(
            input, row_packed, row_scale, nullptr, nullptr, M, N, g.row_nk_pad, g.col_nk_pad);
        break;
    case MXFP4Direction::Col:
        quantize_mxfp4_kernel<DType, false, true><<<g.grid, g.block, 0, stream>>>(
            input, nullptr, nullptr, col_packed, col_scale, M, N, g.row_nk_pad, g.col_nk_pad);
        break;
    case MXFP4Direction::Dual:
        quantize_mxfp4_kernel<DType, true, true><<<g.grid, g.block, 0, stream>>>(
            input, row_packed, row_scale, col_packed, col_scale, M, N, g.row_nk_pad, g.col_nk_pad);
        break;
    }
}

template void quantize_mxfp4_impl<bfloat16>(const bfloat16 *, uint8_t *, uint8_t *, uint8_t *,
                                            uint8_t *, const int32_t, const int32_t,
                                            const MXFP4Direction, hipStream_t);
template void quantize_mxfp4_impl<half>(const half *, uint8_t *, uint8_t *, uint8_t *, uint8_t *,
                                        const int32_t, const int32_t, const MXFP4Direction,
                                        hipStream_t);

} // namespace primus_turbo
