#include "primus_turbo/quantization.h"
#include "primus_turbo/shuffle.h"

namespace primus_turbo {

constexpr int THREADS_PER_BLOCK = 1024;

using namespace primus_turbo::dtype;
using namespace primus_turbo::detail;

/*
 * Vectorized E8M0 Scale Shuffle Kernel.
 * --------------------------------------
 *
 * The shuffle rearranges (32 × 8) tiles.  Within each tile:
 *   input  (i1, i2, i4, i5) → output (i5, i2, i4, i1)
 *   where  row = i0*32 + i1*16 + i2,  col = i3*8 + i4*4 + i5
 *
 * The 4 output bytes at a fixed (i5, i2) are contiguous (i4*2+i1 = 0..3),
 * sourced from 2 rows × 2 col-halves.  Each thread gathers these 4 bytes,
 * packs into uint32_t, and writes once.
 *
 * Thread mapping: 64 threads per (32×8) tile, one per (i5, i2) pair.
 * Writes are fully coalesced: 64 threads × 4 B = 256 B contiguous per tile.
 */
__global__ __launch_bounds__(THREADS_PER_BLOCK) void shuffle_e8m0_scale_kernel(
    uint8_t *__restrict__ x, uint8_t *__restrict__ y, int64_t M, int64_t N, int64_t M_pad,
    int64_t N_pad) {
    constexpr int THREADS_PER_TILE = 64;
    constexpr int TILES_PER_BLOCK  = THREADS_PER_BLOCK / THREADS_PER_TILE;

    const int n_tiles_col = static_cast<int>(N_pad >> 3);
    const int total_tiles = static_cast<int>(M_pad >> 5) * n_tiles_col;

    const int t_in_tile  = threadIdx.x & 63;
    const int local_tile = threadIdx.x >> 6;
    const int tile_idx   = static_cast<int>(blockIdx.x) * TILES_PER_BLOCK + local_tile;

    if (tile_idx >= total_tiles)
        return;

    const int i0 = tile_idx / n_tiles_col;
    const int i3 = tile_idx - i0 * n_tiles_col;

    const int i5 = t_in_tile >> 4;
    const int i2 = t_in_tile & 0xF;

    const int row0 = (i0 << 5) + i2;
    const int row1 = row0 + 16;
    const int col0 = (i3 << 3) + i5;
    const int col1 = col0 + 4;

    const bool r0_ok = row0 < M;
    const bool r1_ok = row1 < M;
    const bool c0_ok = col0 < N;
    const bool c1_ok = col1 < N;

    const int64_t r0_base = static_cast<int64_t>(row0) * N;
    const int64_t r1_base = static_cast<int64_t>(row1) * N;

    const uint8_t pad = E8M0_EXPONENT_BIAS;
    uint8_t       b0  = (r0_ok && c0_ok) ? x[r0_base + col0] : pad; // i1=0, i4=0
    uint8_t       b1  = (r1_ok && c0_ok) ? x[r1_base + col0] : pad; // i1=1, i4=0
    uint8_t       b2  = (r0_ok && c1_ok) ? x[r0_base + col1] : pad; // i1=0, i4=1
    uint8_t       b3  = (r1_ok && c1_ok) ? x[r1_base + col1] : pad; // i1=1, i4=1

    uint32_t packed = static_cast<uint32_t>(b0) | (static_cast<uint32_t>(b1) << 8) |
                      (static_cast<uint32_t>(b2) << 16) | (static_cast<uint32_t>(b3) << 24);

    const int64_t tile_base = static_cast<int64_t>(tile_idx) << 8;
    *reinterpret_cast<uint32_t *>(&y[tile_base + (i5 << 6) + (i2 << 2)]) = packed;
}

/*
 * Vectorized FP4 Weight Shuffle Kernel.
 * --------------------------
 * Each (BN=16, BK=32-byte) tile is reordered from row-major into two
 * contiguous (16×16-byte) sub-blocks.
 *
 * Thread mapping: 16 threads per tile, one thread per row_in_block.
 *   - Load  32 B  (2 × uint4) from the source row
 *   - Store first  16 B into sub-block 0
 *   - Store second 16 B into sub-block 1
 *
 */
__global__ __launch_bounds__(THREADS_PER_BLOCK) void shuffle_fp4_weight_kernel(
    uint8_t *__restrict__ x, uint8_t *__restrict__ y, int64_t M, int64_t N) {
    constexpr int THREADS_PER_TILE = 16;
    constexpr int TILES_PER_BLOCK  = THREADS_PER_BLOCK / THREADS_PER_TILE;

    const int K_packed    = static_cast<int>(N >> 1); // N / 2
    const int k_blocks    = K_packed >> 5;
    const int total_tiles = static_cast<int>(M >> 4) * k_blocks;

    const int row_in_block = threadIdx.x & 0xF; // threadIdx.x % 16
    const int local_tile   = threadIdx.x >> 4;  // threadIdx.x / 16
    const int tile_idx     = static_cast<int>(blockIdx.x) * TILES_PER_BLOCK + local_tile;

    if (tile_idx >= total_tiles)
        return;

    const int nb = tile_idx / k_blocks;
    const int kb = tile_idx - nb * k_blocks;

    const int64_t src =
        static_cast<int64_t>(nb * THREADS_PER_TILE + row_in_block) * K_packed + kb * 32;
    const int64_t dst = static_cast<int64_t>(nb) * (K_packed << 4) + kb * 512 + (row_in_block << 4);

    uint4 lo = *reinterpret_cast<const uint4 *>(&x[src]);
    uint4 hi = *reinterpret_cast<const uint4 *>(&x[src + 16]);

    *reinterpret_cast<uint4 *>(&y[dst])       = lo;
    *reinterpret_cast<uint4 *>(&y[dst + 256]) = hi;
}

void shuffle_e8m0_scale(uint8_t *scale, uint8_t *shuffled_scale, int64_t scale_M, int64_t scale_N,
                        int64_t scale_M_pad, int64_t scale_N_pad, hipStream_t stream) {
    constexpr int TILES_PER_BLOCK = THREADS_PER_BLOCK / 64;
    int64_t       total_tiles     = (scale_M_pad >> 5) * (scale_N_pad >> 3);
    dim3          grid((total_tiles + TILES_PER_BLOCK - 1) / TILES_PER_BLOCK);
    dim3          block(THREADS_PER_BLOCK);

    shuffle_e8m0_scale_kernel<<<grid, block, 0, stream>>>(scale, shuffled_scale, scale_M, scale_N,
                                                          scale_M_pad, scale_N_pad);
}

template <typename DType>
void shuffle_weight(DType *weight, DType *shuffled_weight, int64_t weight_M, int64_t weight_N,
                    hipStream_t stream) {
    if constexpr (std::is_same_v<DType, float4x2_e2m1>) {
        constexpr int TILES_PER_BLOCK = THREADS_PER_BLOCK / 16;
        int64_t       total_tiles     = (weight_M >> 4) * (weight_N >> 4) / 4;
        dim3          grid((total_tiles + TILES_PER_BLOCK - 1) / TILES_PER_BLOCK);
        dim3          block(THREADS_PER_BLOCK);
        shuffle_fp4_weight_kernel<<<grid, block, 0, stream>>>(
            reinterpret_cast<uint8_t *>(weight), reinterpret_cast<uint8_t *>(shuffled_weight),
            weight_M, weight_N);
    } else {
        PRIMUS_TURBO_ERROR("Unsupported weight type");
    }
}

template void shuffle_weight<dtype::float4x2_e2m1>(dtype::float4x2_e2m1 *weight,
                                                   dtype::float4x2_e2m1 *shuffled_weight,
                                                   int64_t weight_M, int64_t weight_N,
                                                   hipStream_t stream);

} // namespace primus_turbo
