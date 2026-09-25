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
#include "primus_turbo/mxfp4_emit.hpp"
#include "primus_turbo/quantization.h"

namespace primus_turbo {

using bfloat16 = hip_bfloat16;

namespace {

// Layout constants and the group emit are shared with the MXFP6 packer's hybrid mode.
using namespace mxfp4_emit;

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

            mxfp4_emit::mxfp4_emit_group(values, tile_m + local_m, tile_n / kGroupSize + k_block, row_nk_pad,
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

            mxfp4_emit::mxfp4_emit_group(values, tile_n + local_n, tile_m / kGroupSize + k_block, col_nk_pad,
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
