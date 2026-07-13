// Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// See LICENSE for license information.
//
// Batched 2D transpose of the last two dims, for arbitrary element dtype.

#include "primus_turbo/common.h"
#include "primus_turbo/transpose.h"

namespace primus_turbo {

namespace {

// grid.z is limited to 65535 on the HIP driver; larger batches are covered by a
// batch grid-stride loop inside the kernel.
constexpr int64_t MAX_BATCH         = 65535;
constexpr int     THREADS_PER_BLOCK = 256; // shared by both paths

constexpr int VEC_BYTES = 16; // bytes per vectorized op (int4), VEC path only

// VEC path swizzle over 16-byte strips of ELEMS_PER_STRIP = VEC_BYTES/sizeof(T)
// elements:
//   strip'(row, col) = (col / ELEMS_PER_STRIP) ^ (row / ELEMS_PER_STRIP)
//   (STRIPS_PER_ROW = TILE / ELEMS_PER_STRIP strips per tile row)
// keeping each 16B strip contiguous/aligned while making the corner-turn gather
// bank-conflict-free. Both constants are powers of two, so /, %, * fold to shifts.
template <int ELEMS_PER_STRIP, int STRIPS_PER_ROW>
__device__ __forceinline__ int swz(const int row, const int col) {
    return (((col / ELEMS_PER_STRIP) ^ ((row / ELEMS_PER_STRIP) & (STRIPS_PER_ROW - 1))) *
            ELEMS_PER_STRIP) +
           (col % ELEMS_PER_STRIP);
}

// 16-byte POD element for itemsize == 16 dtypes (e.g. complex128).
struct alignas(16) Bytes16 {
    uint64_t lo;
    uint64_t hi;
};

// Single kernel, two compile-time-selected paths (see file header).
//   VEC_PATH == true  : sizeof(T) in {1,2,4}, TILE == 64, requires that both M and
//                       N are a multiple of ELEMS_PER_STRIP (= 16 / sizeof(T)).
//   VEC_PATH == false : coalesced tiled transpose for any T / shape.
template <typename T, int TILE, bool VEC_PATH>
__global__ __launch_bounds__(THREADS_PER_BLOCK) void transpose_2d_kernel(const T *__restrict__ x,
                                                                         T *__restrict__ y,
                                                                         const int M, const int N,
                                                                         const int64_t batch) {
    // Diagonal block remap: shift the column-block by the row-block index so the
    // transposed writes of concurrently-running blocks spread across DRAM
    // channels instead of camping on a few. Bijective per row-block.
    const int bx  = (blockIdx.x + blockIdx.y) % gridDim.x;
    const int r0  = blockIdx.y * TILE; // i (row / M) base
    const int c0  = bx * TILE;         // j (col / N) base
    const int tid = threadIdx.x;

    if constexpr (VEC_PATH) {
        constexpr int ELEMS_PER_STRIP = VEC_BYTES / (int) sizeof(T); // elements per 16B strip
        constexpr int STRIPS_PER_ROW  = TILE / ELEMS_PER_STRIP;      // 16B strips per tile row/col
        constexpr int STRIPS_PER_TILE = TILE * STRIPS_PER_ROW;       // total strips per tile
        __shared__ __align__(16) T smem[TILE][TILE];                 // swizzled

        for (int64_t b = blockIdx.z; b < batch; b += gridDim.z) {
            const int64_t base = b * (int64_t) M * N; // shared by in (i*N+j) and out (j*M+i)

            // Load: contiguous 16B strip along j -> swizzled smem strip.
            for (int idx = tid; idx < STRIPS_PER_TILE; idx += THREADS_PER_BLOCK) {
                const int li  = idx / STRIPS_PER_ROW;
                const int lj0 = (idx % STRIPS_PER_ROW) * ELEMS_PER_STRIP;
                const int gi  = r0 + li;
                const int gj0 = c0 + lj0;
                if (gi < M && gj0 < N) {
                    const int sc = swz<ELEMS_PER_STRIP, STRIPS_PER_ROW>(li, lj0);
                    *reinterpret_cast<int4 *>(&smem[li][sc]) =
                        *reinterpret_cast<const int4 *>(&x[base + (int64_t) gi * N + gj0]);
                }
            }
            __syncthreads();

            // Store: corner-turn gather of ELEMS_PER_STRIP rows -> contiguous 16B strip along i.
            for (int idx = tid; idx < STRIPS_PER_TILE; idx += THREADS_PER_BLOCK) {
                const int lj  = idx / STRIPS_PER_ROW;
                const int li0 = (idx % STRIPS_PER_ROW) * ELEMS_PER_STRIP;
                const int gj  = c0 + lj;
                const int gi0 = r0 + li0;
                if (gj < N && gi0 < M) {
                    // sc is constant over the ELEMS_PER_STRIP gathered rows.
                    const int sc = swz<ELEMS_PER_STRIP, STRIPS_PER_ROW>(li0, lj);
                    int4      v;
                    T        *vt = reinterpret_cast<T *>(&v);
#pragma unroll
                    for (int k = 0; k < ELEMS_PER_STRIP; ++k) {
                        vt[k] = smem[li0 + k][sc];
                    }
                    *reinterpret_cast<int4 *>(&y[base + (int64_t) gj * M + gi0]) = v;
                }
            }
            __syncthreads();
        }
    } else {
        constexpr int GROWS = THREADS_PER_BLOCK / TILE; // tile rows covered per pass
        constexpr int STRIP = TILE / GROWS;
        __shared__ T  smem[TILE][TILE + 4]; // +4 pads away corner-turn bank conflicts

        const int tx = tid % TILE; // fast dim (column on load, row i on store)
        const int tg = tid / TILE; // row group; strides by GROWS

        for (int64_t b = blockIdx.z; b < batch; b += gridDim.z) {
            const int64_t base = b * (int64_t) M * N;

            // Load: tx -> column (coalesced along N). Read the whole strip into
            // registers first (many in-flight global loads) then commit to smem.
            {
                const int gj = c0 + tx;
                if (gj < N) {
                    T reg[STRIP];
#pragma unroll
                    for (int s = 0; s < STRIP; ++s) {
                        const int gi = r0 + tg + s * GROWS;
                        if (gi < M) {
                            reg[s] = x[base + (int64_t) gi * N + gj];
                        }
                    }
#pragma unroll
                    for (int s = 0; s < STRIP; ++s) {
                        const int gi = r0 + tg + s * GROWS;
                        if (gi < M) {
                            smem[tg + s * GROWS][tx] = reg[s];
                        }
                    }
                }
            }
            __syncthreads();

            // Store: tx -> row i (coalesced along M, the output inner dim).
            {
                const int gi = r0 + tx;
                if (gi < M) {
                    for (int c = tg; c < TILE; c += GROWS) {
                        const int gj = c0 + c;
                        if (gj < N) {
                            y[base + (int64_t) gj * M + gi] = smem[tx][c];
                        }
                    }
                }
            }
            __syncthreads();
        }
    }
}

template <typename T, int TILE, bool VEC_PATH>
void launch(const void *x, void *y, const int64_t batch, const int64_t M, const int64_t N,
            const unsigned int grid_z, hipStream_t stream) {
    const dim3 grid(static_cast<unsigned int>(DIVUP<int64_t>(N, TILE)),
                    static_cast<unsigned int>(DIVUP<int64_t>(M, TILE)), grid_z);
    transpose_2d_kernel<T, TILE, VEC_PATH><<<grid, dim3(THREADS_PER_BLOCK), 0, stream>>>(
        reinterpret_cast<const T *>(x), reinterpret_cast<T *>(y), static_cast<int>(M),
        static_cast<int>(N), batch);
}

} // namespace

// Batched 2D transpose (last two dims) of a contiguous buffer of `itemsize`-byte
// elements: in [batch, M, N] -> out [batch, N, M]. batch == 1 for 2D input.
void transpose_2d_impl(const void *x, void *y, const int64_t batch, const int64_t M,
                       const int64_t N, const int64_t itemsize, hipStream_t stream) {
    if (batch == 0 || M == 0 || N == 0) {
        return;
    }
    const unsigned int grid_z = static_cast<unsigned int>(batch < MAX_BATCH ? batch : MAX_BATCH);

    // VEC path handles 1/2/4-byte elements when both trailing dims are a multiple
    // of the strip width (16 / itemsize elements); otherwise fall back to generic.
    switch (itemsize) {
    case 1:
        if ((M % 16) == 0 && (N % 16) == 0) {
            launch<uint8_t, 64, true>(x, y, batch, M, N, grid_z, stream);
        } else {
            launch<uint8_t, 64, false>(x, y, batch, M, N, grid_z, stream);
        }
        break;
    case 2:
        if ((M % 8) == 0 && (N % 8) == 0) {
            launch<uint16_t, 64, true>(x, y, batch, M, N, grid_z, stream);
        } else {
            launch<uint16_t, 64, false>(x, y, batch, M, N, grid_z, stream);
        }
        break;
    case 4:
        if ((M % 4) == 0 && (N % 4) == 0) {
            launch<uint32_t, 64, true>(x, y, batch, M, N, grid_z, stream);
        } else {
            launch<uint32_t, 32, false>(x, y, batch, M, N, grid_z, stream);
        }
        break;
    case 8:
        launch<uint64_t, 32, false>(x, y, batch, M, N, grid_z, stream);
        break;
    case 16:
        launch<Bytes16, 16, false>(x, y, batch, M, N, grid_z, stream);
        break;
    default:
        PRIMUS_TURBO_CHECK(false, "transpose_2d: unsupported element size (bytes)");
    }
}

} // namespace primus_turbo
