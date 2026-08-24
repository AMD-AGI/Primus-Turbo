/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

// Fused MXFP6 (E2M3) quantize + pack into AITER's mxfp6_c0c1_256_padk2 layout,
// in both contraction directions from a single read of the input.
//
// Why this exists
// ---------------
// Training needs every tensor packed along two axes (fprop contracts K, dgrad
// contracts N, wgrad contracts M). AITER only ships a row-direction packer, so the
// column direction previously went through `pack(x.t().contiguous())`. Profiling a
// Flux 12B step attributed 82% of the MXFP6-vs-MXFP4 step-time gap to exactly that
// materialised transpose -- 420 ms of a 512 ms gap -- dwarfing both the GEMM
// difference and the packing itself.
//
// The fix is to never materialise it. A block stages one TILE_M x TILE_N patch of the
// input in LDS with coalesced reads, then packs rows and columns out of that patch.
// Everything downstream of the gather -- Hadamard, scale, conversion, addressing -- is
// identical for the two directions, because packing a column of `x` is by definition
// packing a row of `x.T`.
//
// Bit-exactness
// -------------
// The per-group math is a deliberate transliteration of AITER's `quant_mxfp6_group`
// (csrc/kernels/quant_mxfp6_gemm.cu), down to the H8-then-two-lane-butterflies
// factorisation of H32 and the gfx950 conversion intrinsic. That is not incidental:
// the row direction must reproduce AITER's packer byte for byte, since that packer is
// the oracle the A6W6 assembly was validated against, and the E2M3 rounding of an exact
// tie differs between plausible implementations (`floor(x + 0.5)` disagrees with the
// hardware's round-to-nearest-even, which is what the intrinsic does). Reusing the
// intrinsic makes the agreement structural rather than something tests have to police.

#include <hip/hip_runtime.h>

#include "primus_turbo/common.h"
#include "primus_turbo/quantization.h"

namespace primus_turbo {

using namespace primus_turbo::dtype;

namespace {

// ---------------------------------------------------------------------------
// Layout constants. These describe AITER's packed blob and cannot be retuned:
// the A6W6 assembly derives its strides from them.
// ---------------------------------------------------------------------------
constexpr int kGroupSize       = 32;  // values per E8M0 scale, and the Hadamard size
constexpr int kThreadsPerGroup = 4;
constexpr int kValuesPerThread = kGroupSize / kThreadsPerGroup; // 8
constexpr int kTileRows        = 256; // rows per packed tile
constexpr int kKTile           = 128; // K values per packed tile
constexpr int kGroupsPerKTile  = kKTile / kGroupSize; // 4
constexpr int kPackedTileBytes = 24576;
constexpr int kScaleTileBytes  = 1024;
constexpr int kC1PlaneOffset   = 16384; // byte offset of the C1 plane within a tile
constexpr int kC0BytesPerBlock = 16;    // of 24 bytes per group, 16 land in C0
constexpr int kC1BytesPerBlock = 8;

// 1/sqrt(32) rounded to bf16, NOT the nearest float. The reference packers apply the
// rotation as a bf16 dot, so their normalisation carries bf16 precision; using the exact
// fp32 constant instead shifts every rotated value by ~1e-4 relative, which is enough to
// push values sitting just under a rounding boundary over it and change ~0.1% of codes.
constexpr float kHadamard32Norm = 0.1767578125f;

// ---------------------------------------------------------------------------
// Blocking. TILE_M/TILE_N are multiples of kGroupSize so every staged patch holds
// whole 32-value groups in both directions, and divide 256 so a 256-aligned operand
// tiles exactly.
// ---------------------------------------------------------------------------
constexpr int TILE_M           = 64;
constexpr int TILE_N           = 64;
constexpr int THREADS_PER_BLOCK = 256;
constexpr int GROUP_SLOTS      = THREADS_PER_BLOCK / kThreadsPerGroup; // 64

// Pad the LDS row pitch to 33 dwords so the column-direction gather, which walks the
// pitch, spreads across banks instead of piling onto one.
constexpr int LDS_PAD   = 2;
constexpr int LDS_PITCH = TILE_N + LDS_PAD; // 66 uint16 = 33 dwords

using packed_fp6x32_t = uint32_t __attribute__((ext_vector_type(6)));
using uint4_t         = uint32_t __attribute__((ext_vector_type(4)));
using uint2_t         = uint32_t __attribute__((ext_vector_type(2)));
using float16_t       = float __attribute__((ext_vector_type(16)));

__device__ __forceinline__ float swap_adjacent_lane(float value) {
    return __shfl_xor(value, 1);
}

__device__ __forceinline__ float swap_lane_distance_two(float value) {
    return __shfl_xor(value, 2);
}

// The packer's Hadamard is a bf16 dot with fp32 accumulate, so an fp16 input has to be
// rounded through bf16 first or the codes drift from AITER's.
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
 * Quantize one 32-value group, cooperatively across `kThreadsPerGroup` adjacent lanes,
 * and scatter it into the packed blob.
 *
 * `values` holds this lane's 8 contiguous elements of the group: lane L owns group
 * elements [L*8, L*8+8). The two cross-lane butterflies below depend on that mapping
 * and on the four lanes being a lane-aligned quad, so callers must keep the whole quad
 * on the same control-flow path.
 *
 * `out_row` is the row index in the packed operand and `group` the 32-block index along
 * the contraction axis; the caller decides what those mean, which is the only thing that
 * distinguishes the row direction from the column direction.
 */
__device__ __forceinline__ void mxfp6_emit_group(float (&values)[kValuesPerThread],
                                                 const int32_t lane, const int64_t out_row,
                                                 const int32_t group, const int32_t nk_pad,
                                                 uint8_t *__restrict__ packed,
                                                 uint8_t *__restrict__ packed_scale) {
    // H32 = H8 within each lane, then butterflies against lane^1 and lane^2.
#pragma unroll
    for (int stage = 0; stage < 3; ++stage) {
        const int h = 1 << stage;
#pragma unroll
        for (int pair = 0; pair < kValuesPerThread / 2; ++pair) {
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
#pragma unroll
    for (int i = 0; i < kValuesPerThread; ++i) {
        const float peer = swap_adjacent_lane(values[i]);
        values[i]        = (lane & 1) == 0 ? values[i] + peer : peer - values[i];
    }
#pragma unroll
    for (int i = 0; i < kValuesPerThread; ++i) {
        const float peer = swap_lane_distance_two(values[i]);
        values[i]        = (lane < 2 ? values[i] + peer : peer - values[i]) * kHadamard32Norm;
    }

    float local_amax = 0.0f;
#pragma unroll
    for (int i = 0; i < kValuesPerThread; ++i)
        local_amax = fmaxf(local_amax, fabsf(values[i]));
    local_amax       = fmaxf(local_amax, swap_adjacent_lane(local_amax));
    const float amax = fmaxf(local_amax, swap_lane_distance_two(local_amax));

    // E8M0 scale: take the amax exponent directly out of the fp32 bits. The -129 (rather
    // than -127) leaves two binades of headroom so the largest magnitude lands inside
    // E2M3's 7.5 rather than clipping.
    int32_t scale_unbiased;
    if (amax == 0.0f) {
        scale_unbiased = 0;
    } else {
        const uint32_t exponent = (__builtin_bit_cast(uint32_t, amax) >> 23) & 0xFFu;
        scale_unbiased          = exponent == 0u
                                      ? -127
                                      : (exponent == 0xFFu ? 127 : static_cast<int32_t>(exponent) - 129);
        scale_unbiased          = scale_unbiased < -127 ? -127 : scale_unbiased;
        scale_unbiased          = scale_unbiased > 127 ? 127 : scale_unbiased;
    }
    const uint8_t scale_exp = static_cast<uint8_t>(scale_unbiased + 127);

    // Gather the quad's 32 values into the even/odd operands the conversion expects. The
    // instruction interleaves src0/src1, so this yields the plain sequential 6-bit stream.
    float16_t even;
    float16_t odd;
#pragma unroll
    for (int i = 0; i < kValuesPerThread / 2; ++i) {
        const float v0_even = values[2 * i];
        const float v0_odd  = values[2 * i + 1];
        const float v1_even = swap_adjacent_lane(v0_even);
        const float v1_odd  = swap_adjacent_lane(v0_odd);
        const float v2_even = swap_lane_distance_two(v0_even);
        const float v2_odd  = swap_lane_distance_two(v0_odd);
        const float v3_even = swap_adjacent_lane(v2_even);
        const float v3_odd  = swap_adjacent_lane(v2_odd);
        even[i]             = v0_even;
        odd[i]              = v0_odd;
        even[4 + i]         = v1_even;
        odd[4 + i]          = v1_odd;
        even[8 + i]         = v2_even;
        odd[8 + i]          = v2_odd;
        even[12 + i]        = v3_even;
        odd[12 + i]         = v3_odd;
    }

    if (lane != 0)
        return;

    const uint32_t scale_bits =
        scale_exp == 0 ? 0x00400000u : static_cast<uint32_t>(scale_exp) << 23;
    const float mx_scale = __builtin_bit_cast(float, scale_bits);
#if defined(__gfx950__)
    const packed_fp6x32_t fp6 =
        amax == 0.0f ? packed_fp6x32_t{}
                     : __builtin_amdgcn_cvt_scalef32_2xpk16_fp6_f32(even, odd, mx_scale);
#else
    const packed_fp6x32_t fp6{};
#endif

    // Scatter: 24 bytes per group, split 16/8 across the C0 and C1 planes.
    const int32_t tile_row  = static_cast<int32_t>(out_row / kTileRows);
    const int32_t rem       = static_cast<int32_t>(out_row % kTileRows);
    const int32_t row_block = rem / 16;
    const int32_t row16     = rem % 16;
    const int32_t step      = group / kGroupsPerKTile;
    const int32_t k_group   = group % kGroupsPerKTile;
    const int32_t block     = row_block * 64 + k_group * 16 + row16;
    const int64_t tile_base = (static_cast<int64_t>(tile_row) * nk_pad + step) * kPackedTileBytes;
    const int64_t c0_base   = tile_base + block * kC0BytesPerBlock;
    const int64_t c1_base   = tile_base + kC1PlaneOffset + block * kC1BytesPerBlock;
    *reinterpret_cast<uint4_t *>(packed + c0_base) = *reinterpret_cast<const uint4_t *>(&fp6);
    *reinterpret_cast<uint2_t *>(packed + c1_base) =
        *reinterpret_cast<const uint2_t *>(reinterpret_cast<const uint8_t *>(&fp6) + 16);

    const int32_t scale_upper = rem / 128;
    const int32_t scale_sub   = (rem % 128) / 16;
    const int64_t scale_address =
        (static_cast<int64_t>(tile_row) * nk_pad + step) * kScaleTileBytes + scale_upper * 512 +
        k_group * 128 + row16 * 8 + scale_sub;
    packed_scale[scale_address] = scale_exp;
}

/*
 * Fused dual MXFP6 packer.
 *
 * The grid covers the operand padded to 256 in both dimensions rather than its logical
 * extent, so the padded rows and columns are packed too. That is not wasted work: the
 * blob has to contain a well-defined encoding of zero there, and letting the OOB reads
 * fall out of the LDS zero-fill produces it for free -- otherwise the host would have to
 * memset the whole blob, which costs more than the packing.
 */
template <typename DType, bool DO_ROW, bool DO_COL>
__global__ __launch_bounds__(THREADS_PER_BLOCK) void quantize_mxfp6_dual_kernel(
    const DType *__restrict__ input, uint8_t *__restrict__ row_packed,
    uint8_t *__restrict__ row_scale, uint8_t *__restrict__ col_packed,
    uint8_t *__restrict__ col_scale, const int32_t M, const int32_t N,
    const int32_t row_nk_pad, const int32_t col_nk_pad) {
    __shared__ uint16_t s_tile[TILE_M][LDS_PITCH];

    const int32_t tile_m = blockIdx.y * TILE_M;
    const int32_t tile_n = blockIdx.x * TILE_N;

    // Stage the patch. Reads are coalesced along N; anything outside the logical tensor
    // becomes zero, which is what the padded region of the blob must encode.
    {
        const auto *__restrict__ input_u16 = reinterpret_cast<const uint16_t *>(input);
        constexpr int VEC                  = 8;
        constexpr int ELEMS                = TILE_M * TILE_N;
#pragma unroll
        for (int base = threadIdx.x * VEC; base < ELEMS; base += THREADS_PER_BLOCK * VEC) {
            const int local_m = base / TILE_N;
            const int local_n = base % TILE_N;
            const int global_m = tile_m + local_m;
            const int global_n = tile_n + local_n;

            uint16_t staged[VEC] = {0, 0, 0, 0, 0, 0, 0, 0};
            if (global_m < M) {
                if (global_n + VEC <= N) {
                    *reinterpret_cast<uint4 *>(staged) = *reinterpret_cast<const uint4 *>(
                        &input_u16[static_cast<int64_t>(global_m) * N + global_n]);
                } else {
#pragma unroll
                    for (int i = 0; i < VEC; ++i) {
                        staged[i] = (global_n + i < N)
                                        ? input_u16[static_cast<int64_t>(global_m) * N + global_n + i]
                                        : uint16_t{0};
                    }
                }
            }
#pragma unroll
            for (int i = 0; i < VEC; ++i)
                s_tile[local_m][local_n + i] = staged[i];
        }
    }
    __syncthreads();

    const int32_t lane = threadIdx.x & (kThreadsPerGroup - 1);
    const int32_t slot = threadIdx.x / kThreadsPerGroup;

    // Row direction: contract along N. Each staged row contributes TILE_N/32 groups.
    if constexpr (DO_ROW) {
        constexpr int kBlocksPerRow = TILE_N / kGroupSize;
        constexpr int kRowGroups    = TILE_M * kBlocksPerRow;
#pragma unroll
        for (int gi = slot; gi < kRowGroups; gi += GROUP_SLOTS) {
            const int local_m  = gi / kBlocksPerRow;
            const int k_block  = gi % kBlocksPerRow;
            const int n_offset = k_block * kGroupSize + lane * kValuesPerThread;

            float values[kValuesPerThread];
#pragma unroll
            for (int i = 0; i < kValuesPerThread; ++i)
                values[i] = to_dot_operand<DType>(s_tile[local_m][n_offset + i]);

            mxfp6_emit_group(values, lane, tile_m + local_m, tile_n / kGroupSize + k_block,
                             row_nk_pad, row_packed, row_scale);
        }
    }

    // Column direction: contract along M, i.e. pack the rows of x.T. Same emit, different
    // gather -- this is the transpose that no longer has to be materialised.
    if constexpr (DO_COL) {
        constexpr int kBlocksPerCol = TILE_M / kGroupSize;
        constexpr int kColGroups    = TILE_N * kBlocksPerCol;
#pragma unroll
        for (int gi = slot; gi < kColGroups; gi += GROUP_SLOTS) {
            const int local_n  = gi / kBlocksPerCol;
            const int k_block  = gi % kBlocksPerCol;
            const int m_offset = k_block * kGroupSize + lane * kValuesPerThread;

            float values[kValuesPerThread];
#pragma unroll
            for (int i = 0; i < kValuesPerThread; ++i)
                values[i] = to_dot_operand<DType>(s_tile[m_offset + i][local_n]);

            mxfp6_emit_group(values, lane, tile_n + local_n, tile_m / kGroupSize + k_block,
                             col_nk_pad, col_packed, col_scale);
        }
    }
}

constexpr int ceil_div(const int x, const int m) { return (x + m - 1) / m; }

} // namespace

template <typename DType>
void quantize_mxfp6_impl(const DType *input, uint8_t *row_packed, uint8_t *row_scale,
                         uint8_t *col_packed, uint8_t *col_scale, const int M, const int N,
                         const MXFP6Direction direction, hipStream_t stream) {
    const int row_nk_pad = ceil_div(N, kKTile) + MXFP6_GUARD_K_TILES;
    const int col_nk_pad = ceil_div(M, kKTile) + MXFP6_GUARD_K_TILES;

    // Cover the operand padded to whole 256-row tiles in both directions: M is the row
    // count of the row-direction blob and the K extent of the column-direction one, and
    // vice versa for N, so both have to be rounded up before tiling.
    //
    // Rounding K up to 256 rather than to its own 128 granularity can push one K-tile
    // past the blob's real extent. That is in bounds, not an overrun: the two mandatory
    // guard tiles absorb it, and the excess is at most one tile because ceil(k, 256) and
    // ceil(k, 128) differ by at most 128. Those writes are dead, like everything else in
    // the guard region.
    const int m_padded = ceil_div(M, kTileRows) * kTileRows;
    const int n_padded = ceil_div(N, kTileRows) * kTileRows;
    const dim3 grid(ceil_div(n_padded, TILE_N), ceil_div(m_padded, TILE_M));
    const dim3 block(THREADS_PER_BLOCK);

    switch (direction) {
    case MXFP6Direction::Row:
        quantize_mxfp6_dual_kernel<DType, true, false><<<grid, block, 0, stream>>>(
            input, row_packed, row_scale, col_packed, col_scale, M, N, row_nk_pad, col_nk_pad);
        break;
    case MXFP6Direction::Col:
        quantize_mxfp6_dual_kernel<DType, false, true><<<grid, block, 0, stream>>>(
            input, row_packed, row_scale, col_packed, col_scale, M, N, row_nk_pad, col_nk_pad);
        break;
    case MXFP6Direction::Dual:
        quantize_mxfp6_dual_kernel<DType, true, true><<<grid, block, 0, stream>>>(
            input, row_packed, row_scale, col_packed, col_scale, M, N, row_nk_pad, col_nk_pad);
        break;
    }
    PRIMUS_TURBO_CHECK_HIP(hipGetLastError());
}

template void quantize_mxfp6_impl<bfloat16>(const bfloat16 *, uint8_t *, uint8_t *, uint8_t *,
                                            uint8_t *, const int, const int, const MXFP6Direction,
                                            hipStream_t);
template void quantize_mxfp6_impl<float16>(const float16 *, uint8_t *, uint8_t *, uint8_t *,
                                           uint8_t *, const int, const int, const MXFP6Direction,
                                           hipStream_t);

} // namespace primus_turbo
