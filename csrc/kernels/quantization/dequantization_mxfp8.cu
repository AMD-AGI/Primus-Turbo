// Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// See LICENSE for license information.
#include "primus_turbo/common.h"
#include "primus_turbo/device/utils.cuh"
#include "primus_turbo/quantization.h"

namespace primus_turbo {

using namespace primus_turbo::dtype;
using namespace primus_turbo::detail;

// The dequant ops split into two access patterns with different optimal tiles:
//
//  * Rowwise / grouped-rowwise are pure element-wise (row-major in, row-major
//    out). They are bandwidth bound, so the tile is wide and each thread
//    processes several VEC-column chunks (the ``for ci ...`` grid-stride loop
//    runs TOTAL_CHUNKS/nthreads times) to expose instruction-level parallelism
//    and hide global-memory latency without relying solely on occupancy.
//
//  * Colwise / grouped-colwise transpose through shared memory (one
//    __syncthreads per tile). A wider tile amortizes that barrier over more
//    work while keeping the coalesced transposed store (threadIdx.x -> output
//    row, stride 1).
//
// BLOCK_M is kept a multiple of both the gfx950 (64) and gfx1250 (32)
// wavefront sizes so the colwise transposed store stays coalesced on both.

// Rowwise (element-wise) path: each thread converts one 128-bit chunk
// (RW_ELEMS fp8 -> RW_ELEMS out). 16 fp8 columns are 16-aligned, so they never
// cross a 32-wide MXFP8 scale block => a single E8M0 lookup per thread. The
// grid keeps a high thread count (one thread per 16-column chunk) so enough
// memory requests are in flight to hide latency for this bandwidth/latency
// bound op.
constexpr int RW_ELEMS   = 32; // fp8 columns per thread (== block_size => one scale)
constexpr int RW_LVEC    = 16; // fp8 per 128-bit load
constexpr int RW_THREADS = 256;
// RW_TX == 32 so a single 32-lane wavefront covers one contiguous row segment
// (RW_TX * RW_ELEMS columns) => fully-coalesced load/store fronts. RW_ELEMS ==
// block_size keeps all of a thread's columns inside one E8M0 scale block, and
// everything unrolls at compile time. n_cols is a multiple of block_size and
// each start column is a multiple of RW_ELEMS, so a thread's chunk is either
// fully in range or skipped -- no partial/over-read.
constexpr int RW_TX   = 32;                 // threads along columns
constexpr int RW_TY   = RW_THREADS / RW_TX; // thread-rows per block
constexpr int RW_ROWS = 2;                  // rows processed per thread (MLP)

// Colwise (shared-memory transpose) tile. The output is contiguous along the
// original-row dim, so phase 2 has each thread read VEC consecutive rows from
// smem and emit a single 128-bit store (vectorized transposed write) instead of
// VEC scalar 2-byte stores. BLOCK_M must be a multiple of VEC.
constexpr int CW_THREADS = 512;
constexpr int CW_BLOCK_M = 64;  // rows per block (transposed-store contiguous dim)
constexpr int CW_BLOCK_N = 128; // cols per block
constexpr int CW_LVEC    = 16;  // fp8 per 128-bit phase-1 load (16 cols share one scale)

// ---------------------------------------------------------------------------
// MXFP8 de-quantization kernel (rowwise + colwise in one entry).
// ---------------------------------------------------------------------------
// Rowwise (element-wise) dequant: one 128-bit chunk of RW_ELEMS fp8 per thread.
template <typename OType, typename QType>
__global__ void dequantize_mxfp8_rowwise_kernel(
    const QType *__restrict__ x, OType *__restrict__ y, const int64_t stride_x_row,
    const int64_t stride_y_row, const int n_rows, const int n_cols,
    const uint8_t *__restrict__ scale_inv, const int64_t stride_scale_row,
    const int64_t stride_scale_col, const int scale_n_rows, const int scale_n_cols,
    const int block_size) {
    constexpr int VEC  = 16 / static_cast<int>(sizeof(OType)); // out elems per 128-bit store
    const int     row0 = (blockIdx.y * RW_TY + threadIdx.y) * RW_ROWS;
    const int     col  = (blockIdx.x * RW_TX + threadIdx.x) * RW_ELEMS;
    if (col >= n_cols)
        return;
    const int col_block = col / block_size;

    // Issue all RW_ROWS row loads first (independent streams -> more MLP), then
    // convert + store.
    QType x_reg[RW_ROWS][RW_ELEMS];
    float scale[RW_ROWS];
    bool  valid[RW_ROWS];
#pragma unroll
    for (int r = 0; r < RW_ROWS; ++r) {
        const int row = row0 + r;
        valid[r]      = row < n_rows;
        if (!valid[r])
            continue;
        uint8_t e8m0 = static_cast<uint8_t>(E8M0_EXPONENT_BIAS);
        if (row < scale_n_rows && col_block < scale_n_cols)
            e8m0 = scale_inv[static_cast<int64_t>(row) * stride_scale_row +
                             static_cast<int64_t>(col_block) * stride_scale_col];
        scale[r]             = e8m0_to_scale(e8m0);
        const int64_t x_base = static_cast<int64_t>(row) * stride_x_row + col;
#pragma unroll
        for (int b = 0; b < RW_ELEMS; b += RW_LVEC)
            load_data<QType, RW_LVEC>(x + x_base + b, x_reg[r] + b);
    }
#pragma unroll
    for (int r = 0; r < RW_ROWS; ++r) {
        if (!valid[r])
            continue;
        const int64_t y_base = static_cast<int64_t>(row0 + r) * stride_y_row + col;
#pragma unroll
        for (int base = 0; base < RW_ELEMS; base += VEC) {
            OType y_reg[VEC];
#pragma unroll
            for (int i = 0; i < VEC; ++i)
                y_reg[i] = static_cast<OType>(static_cast<float>(x_reg[r][base + i]) * scale[r]);
            store_data<OType, VEC>(y + y_base + base, y_reg);
        }
    }
}

// Colwise dequant: shared-memory transpose then coalesced transposed store.
template <typename OType, typename QType>
__global__ void dequantize_mxfp8_colwise_kernel(
    const QType *__restrict__ x, OType *__restrict__ y, const int64_t stride_x_row,
    const int64_t stride_y_row, const int64_t stride_y_col, const int n_rows, const int n_cols,
    const uint8_t *__restrict__ scale_inv, const int64_t stride_scale_row,
    const int64_t stride_scale_col, const int scale_n_rows, const int scale_n_cols,
    const int block_size) {
    constexpr int BLOCK_M        = CW_BLOCK_M;
    constexpr int BLOCK_N        = CW_BLOCK_N;
    constexpr int VEC            = 16 / static_cast<int>(sizeof(OType)); // store width
    constexpr int CVEC           = CW_LVEC;                              // phase-1 load width (fp8)
    constexpr int CHUNKS_PER_ROW = BLOCK_N / CVEC;
    constexpr int TOTAL_CHUNKS   = BLOCK_M * CHUNKS_PER_ROW;

    const int nthreads = blockDim.x * blockDim.y;
    const int tid      = threadIdx.y * blockDim.x + threadIdx.x;
    const int c0       = blockIdx.x * BLOCK_N;
    const int r0       = blockIdx.y * BLOCK_M;

    __shared__ OType s_tile[BLOCK_M][BLOCK_N + 1]; // +1 mitigates bank conflicts

    // Phase 1: 128-bit read + dequant -> row-major smem (s_tile[row][col]).
    for (int ci = tid; ci < TOTAL_CHUNKS; ci += nthreads) {
        const int local_row = ci / CHUNKS_PER_ROW;
        const int local_col = (ci % CHUNKS_PER_ROW) * CVEC;
        const int grow      = r0 + local_row;
        const int gcol      = c0 + local_col;
        if (grow < n_rows && gcol < n_cols) {
            const int col_block = gcol / block_size; // CVEC (16) cols share one scale
            uint8_t   e8m0      = static_cast<uint8_t>(E8M0_EXPONENT_BIAS);
            if (grow < scale_n_rows && col_block < scale_n_cols) {
                e8m0 = scale_inv[static_cast<int64_t>(grow) * stride_scale_row +
                                 static_cast<int64_t>(col_block) * stride_scale_col];
            }
            const float scale = e8m0_to_scale(e8m0);
            QType       x_reg[CVEC];
            load_data<QType, CVEC>(x + static_cast<int64_t>(grow) * stride_x_row + gcol, x_reg);
#pragma unroll
            for (int i = 0; i < CVEC; ++i) {
                s_tile[local_row][local_col + i] =
                    static_cast<OType>(static_cast<float>(x_reg[i]) * scale);
            }
        }
    }
    __syncthreads();

    // Phase 2: vectorized transposed write. Output is contiguous along the
    // original-row dim (stride_y_col == 1), so each task gathers VEC consecutive
    // rows from smem (fixed col) and stores them as one 128-bit chunk.
    constexpr int R_CHUNKS = BLOCK_M / VEC; // r-chunks along output-contiguous dim
    for (int idx = tid; idx < R_CHUNKS * BLOCK_N; idx += nthreads) {
        const int rc     = idx % R_CHUNKS; // consecutive tids -> consecutive output rows
        const int lc     = idx / R_CHUNKS; // local column
        const int r_base = rc * VEC;
        const int orow   = r0 + r_base;
        const int ocol   = c0 + lc;
        if (ocol >= n_cols)
            continue;
        const int64_t y_off =
            static_cast<int64_t>(orow) * stride_y_col + static_cast<int64_t>(ocol) * stride_y_row;
        if (stride_y_col == 1 && orow + VEC <= n_rows) {
            OType reg[VEC];
#pragma unroll
            for (int i = 0; i < VEC; ++i)
                reg[i] = s_tile[r_base + i][lc];
            store_data<OType, VEC>(y + y_off, reg);
        } else {
#pragma unroll
            for (int i = 0; i < VEC; ++i) {
                if (orow + i < n_rows)
                    y[y_off + i * stride_y_col] = s_tile[r_base + i][lc];
            }
        }
    }
}

// Map compact M row -> padded M row in grouped layout.
__device__ __forceinline__ int64_t grouped_compact_to_padded_m(int            compact_m,
                                                               const int64_t *group_offs,
                                                               const int64_t *group_offs_padded,
                                                               int            G) {
    for (int g = 0; g < G; ++g) {
        if (compact_m >= group_offs[g] && compact_m < group_offs[g + 1]) {
            return group_offs_padded[g] + (compact_m - group_offs[g]);
        }
    }
    return compact_m;
}

// Map padded M row -> compact M row; returns -1 for padding rows.
__device__ __forceinline__ int64_t grouped_padded_to_compact_m(int            padded_m,
                                                               const int64_t *group_offs,
                                                               const int64_t *group_offs_padded,
                                                               int            G) {
    for (int g = 0; g < G; ++g) {
        const int64_t pad_start = group_offs_padded[g];
        if (padded_m >= pad_start && padded_m < group_offs_padded[g + 1]) {
            const int64_t offset_in_group = padded_m - pad_start;
            const int64_t group_len       = group_offs[g + 1] - group_offs[g];
            if (offset_in_group < group_len) {
                return group_offs[g] + offset_in_group;
            }
            return -1;
        }
    }
    return -1;
}

// Rowwise grouped dequant: grid covers compact [total_M, n_cols], reads padded input rows.
template <typename OType, typename QType>
__global__ void grouped_dequantize_mxfp8_rowwise_kernel(
    const QType *__restrict__ x, OType *__restrict__ y, const int64_t stride_x_row,
    const int64_t stride_y_row, const int total_M, const int n_cols,
    const uint8_t *__restrict__ scale_inv, const int64_t stride_scale_row,
    const int64_t stride_scale_col, const int scale_n_rows, const int scale_n_cols,
    const int64_t *__restrict__ group_offs, const int64_t *__restrict__ group_offs_padded, int G,
    const int block_size) {
    constexpr int VEC  = 16 / static_cast<int>(sizeof(OType));
    const int     row0 = (blockIdx.y * RW_TY + threadIdx.y) * RW_ROWS; // compact row
    const int     col0 = (blockIdx.x * RW_TX + threadIdx.x) * RW_ELEMS;
    if (col0 >= n_cols)
        return;
    const int col_block = col0 / block_size;

    QType x_reg[RW_ROWS][RW_ELEMS];
    float scale[RW_ROWS];
    bool  valid[RW_ROWS];
#pragma unroll
    for (int r = 0; r < RW_ROWS; ++r) {
        const int row = row0 + r;
        valid[r]      = row < total_M;
        if (!valid[r])
            continue;
        const int64_t padded_row =
            grouped_compact_to_padded_m(row, group_offs, group_offs_padded, G);
        uint8_t e8m0 = static_cast<uint8_t>(E8M0_EXPONENT_BIAS);
        if (padded_row < scale_n_rows && col_block < scale_n_cols)
            e8m0 = scale_inv[padded_row * stride_scale_row +
                             static_cast<int64_t>(col_block) * stride_scale_col];
        scale[r]             = e8m0_to_scale(e8m0);
        const int64_t x_base = padded_row * stride_x_row + col0;
#pragma unroll
        for (int b = 0; b < RW_ELEMS; b += RW_LVEC)
            load_data<QType, RW_LVEC>(x + x_base + b, x_reg[r] + b);
    }
#pragma unroll
    for (int r = 0; r < RW_ROWS; ++r) {
        if (!valid[r])
            continue;
        const int64_t y_base = static_cast<int64_t>(row0 + r) * stride_y_row + col0;
#pragma unroll
        for (int base = 0; base < RW_ELEMS; base += VEC) {
            OType y_reg[VEC];
#pragma unroll
            for (int i = 0; i < VEC; ++i)
                y_reg[i] = static_cast<OType>(static_cast<float>(x_reg[r][base + i]) * scale[r]);
            store_data<OType, VEC>(y + y_base + base, y_reg);
        }
    }
}

// Colwise grouped dequant: phase 1 identical to dequantize_mxfp8 colwise; phase 2 writes compact
// rows.
template <typename OType, typename QType>
__global__ void grouped_dequantize_mxfp8_colwise_kernel(
    const QType *__restrict__ x, OType *__restrict__ y, const int64_t stride_x_row,
    const int64_t stride_y_row, const int total_M, const int m_padded, const int n_cols,
    const uint8_t *__restrict__ scale_inv, const int64_t stride_scale_row,
    const int64_t stride_scale_col, const int scale_n_rows, const int scale_n_cols,
    const int64_t *__restrict__ group_offs, const int64_t *__restrict__ group_offs_padded, int G,
    const int block_size) {
    constexpr int BLOCK_M        = CW_BLOCK_M;
    constexpr int BLOCK_N        = CW_BLOCK_N;
    constexpr int VEC            = 16 / static_cast<int>(sizeof(OType)); // store width
    constexpr int CVEC           = CW_LVEC;                              // phase-1 load width (fp8)
    constexpr int CHUNKS_PER_ROW = BLOCK_N / CVEC;
    constexpr int TOTAL_CHUNKS   = BLOCK_M * CHUNKS_PER_ROW;

    const int nthreads = blockDim.x * blockDim.y;
    const int tid      = threadIdx.y * blockDim.x + threadIdx.x;
    const int c0       = blockIdx.x * BLOCK_N;
    const int r0       = blockIdx.y * BLOCK_M;

    __shared__ OType s_tile[BLOCK_M][BLOCK_N + 1];

    // Phase 1: dequant padded input [n_cols, m_padded] into smem tile.
    for (int ci = tid; ci < TOTAL_CHUNKS; ci += nthreads) {
        const int local_row = ci / CHUNKS_PER_ROW;
        const int local_col = (ci % CHUNKS_PER_ROW) * CVEC;
        const int grow      = r0 + local_row;
        const int gcol      = c0 + local_col;
        if (grow < n_cols && gcol < m_padded) {
            const int col_block = gcol / block_size;
            uint8_t   e8m0      = static_cast<uint8_t>(E8M0_EXPONENT_BIAS);
            if (grow < scale_n_rows && col_block < scale_n_cols) {
                e8m0 = scale_inv[static_cast<int64_t>(grow) * stride_scale_row +
                                 static_cast<int64_t>(col_block) * stride_scale_col];
            }
            const float scale = e8m0_to_scale(e8m0);
            QType       x_reg[CVEC];
            load_data<QType, CVEC>(x + static_cast<int64_t>(grow) * stride_x_row + gcol, x_reg);
#pragma unroll
            for (int i = 0; i < CVEC; ++i) {
                s_tile[local_row][local_col + i] =
                    static_cast<OType>(static_cast<float>(x_reg[i]) * scale);
            }
        }
    }
    __syncthreads();

    // Phase 2: vectorized scatter. Compact output is contiguous along n (the
    // input-row / output-col dim), so each task gathers VEC consecutive n values
    // from smem (fixed padded-M col) and emits one 128-bit store.
    constexpr int N_CHUNKS = BLOCK_M / VEC; // n-chunks along output-contiguous dim
    for (int idx = tid; idx < N_CHUNKS * BLOCK_N; idx += nthreads) {
        const int rc           = idx % N_CHUNKS; // consecutive tids -> consecutive n
        const int lc           = idx / N_CHUNKS; // local padded-M col
        const int n_base       = rc * VEC;
        const int n            = r0 + n_base;
        const int m_padded_row = c0 + lc;
        if (m_padded_row >= m_padded)
            continue;
        const int64_t compact_m =
            grouped_padded_to_compact_m(m_padded_row, group_offs, group_offs_padded, G);
        if (compact_m < 0 || compact_m >= total_M)
            continue;
        const int64_t y_off = static_cast<int64_t>(compact_m) * stride_y_row + n;
        if (n + VEC <= n_cols) {
            OType reg[VEC];
#pragma unroll
            for (int i = 0; i < VEC; ++i)
                reg[i] = s_tile[n_base + i][lc];
            store_data<OType, VEC>(y + y_off, reg);
        } else {
#pragma unroll
            for (int i = 0; i < VEC; ++i) {
                if (n + i < n_cols)
                    y[y_off + i] = s_tile[n_base + i][lc];
            }
        }
    }
}

template <typename OType, typename QType>
void grouped_dequantize_mxfp8_impl(const QType *x, OType *y, const int64_t stride_x_row,
                                   const int64_t stride_y_row, const int total_M, const int n_rows,
                                   const int n_cols, const uint8_t *scale_inv,
                                   const int64_t stride_scale_row, const int64_t stride_scale_col,
                                   const int scale_n_rows, const int scale_n_cols,
                                   const int64_t *group_offs, const int64_t *group_offs_padded,
                                   int G, int block_size, bool use_rowwise, hipStream_t stream) {
    if (total_M == 0 || n_cols == 0)
        return;

    if (use_rowwise) {
        const int  cols_per_block = RW_TX * RW_ELEMS;
        const int  rows_per_block = RW_TY * RW_ROWS;
        const dim3 block(RW_TX, RW_TY);
        const dim3 grid((n_cols + cols_per_block - 1) / cols_per_block,
                        (total_M + rows_per_block - 1) / rows_per_block);
        grouped_dequantize_mxfp8_rowwise_kernel<OType, QType><<<grid, block, 0, stream>>>(
            x, y, stride_x_row, stride_y_row, total_M, n_cols, scale_inv, stride_scale_row,
            stride_scale_col, scale_n_rows, scale_n_cols, group_offs, group_offs_padded, G,
            block_size);
    } else {
        const dim3 block(CW_BLOCK_M, CW_THREADS / CW_BLOCK_M);
        const dim3 grid((n_rows + CW_BLOCK_N - 1) / CW_BLOCK_N,
                        (n_cols + CW_BLOCK_M - 1) / CW_BLOCK_M);
        grouped_dequantize_mxfp8_colwise_kernel<OType, QType><<<grid, block, 0, stream>>>(
            x, y, stride_x_row, stride_y_row, total_M, n_rows, n_cols, scale_inv, stride_scale_row,
            stride_scale_col, scale_n_rows, scale_n_cols, group_offs, group_offs_padded, G,
            block_size);
    }
}

// Map compact M row -> padded M row in grouped layout.
__device__ __forceinline__ int64_t grouped_compact_to_padded_m(int            compact_m,
                                                               const int64_t *group_offs,
                                                               const int64_t *group_offs_padded,
                                                               int            G) {
    for (int g = 0; g < G; ++g) {
        if (compact_m >= group_offs[g] && compact_m < group_offs[g + 1]) {
            return group_offs_padded[g] + (compact_m - group_offs[g]);
        }
    }
    return compact_m;
}

// Map padded M row -> compact M row; returns -1 for padding rows.
__device__ __forceinline__ int64_t grouped_padded_to_compact_m(int            padded_m,
                                                               const int64_t *group_offs,
                                                               const int64_t *group_offs_padded,
                                                               int            G) {
    for (int g = 0; g < G; ++g) {
        const int64_t pad_start = group_offs_padded[g];
        if (padded_m >= pad_start && padded_m < group_offs_padded[g + 1]) {
            const int64_t offset_in_group = padded_m - pad_start;
            const int64_t group_len       = group_offs[g + 1] - group_offs[g];
            if (offset_in_group < group_len) {
                return group_offs[g] + offset_in_group;
            }
            return -1;
        }
    }
    return -1;
}

// Rowwise grouped dequant: grid covers compact [total_M, n_cols], reads padded input rows.
template <typename OType, typename QType>
__global__ void grouped_dequantize_mxfp8_rowwise_kernel(
    const QType *__restrict__ x, OType *__restrict__ y, const int64_t stride_x_row,
    const int64_t stride_y_row, const int total_M, const int n_cols,
    const uint8_t *__restrict__ scale_inv, const int64_t stride_scale_row,
    const int64_t stride_scale_col, const int scale_n_rows, const int scale_n_cols,
    const int64_t *__restrict__ group_offs, const int64_t *__restrict__ group_offs_padded, int G,
    const int block_size) {
    constexpr int VEC            = 16 / static_cast<int>(sizeof(OType));
    constexpr int CHUNKS_PER_ROW = BLOCK_N / VEC;
    constexpr int TOTAL_CHUNKS   = BLOCK_M * CHUNKS_PER_ROW;

    const int nthreads = blockDim.x * blockDim.y;
    const int tid      = threadIdx.y * blockDim.x + threadIdx.x;
    const int c0       = blockIdx.x * BLOCK_N;
    const int r0       = blockIdx.y * BLOCK_M;

    for (int ci = tid; ci < TOTAL_CHUNKS; ci += nthreads) {
        const int row = r0 + ci / CHUNKS_PER_ROW;
        const int col = c0 + (ci % CHUNKS_PER_ROW) * VEC;
        if (row < total_M && col < n_cols) {
            const int64_t padded_row =
                grouped_compact_to_padded_m(row, group_offs, group_offs_padded, G);
            const int col_block = col / block_size;
            uint8_t   e8m0      = static_cast<uint8_t>(E8M0_EXPONENT_BIAS);
            if (padded_row < scale_n_rows && col_block < scale_n_cols) {
                e8m0 = scale_inv[static_cast<int64_t>(padded_row) * stride_scale_row +
                                 static_cast<int64_t>(col_block) * stride_scale_col];
            }
            const float scale = e8m0_to_scale(e8m0);
            QType       x_reg[VEC];
            OType       y_reg[VEC];
            load_data<QType, VEC>(x + static_cast<int64_t>(padded_row) * stride_x_row + col, x_reg);
#pragma unroll
            for (int i = 0; i < VEC; ++i) {
                y_reg[i] = static_cast<OType>(static_cast<float>(x_reg[i]) * scale);
            }
            store_data<OType, VEC>(y + static_cast<int64_t>(row) * stride_y_row + col, y_reg);
        }
    }
}

// Colwise grouped dequant: phase 1 identical to dequantize_mxfp8 colwise; phase 2 writes compact
// rows.
template <typename OType, typename QType>
__global__ void grouped_dequantize_mxfp8_colwise_kernel(
    const QType *__restrict__ x, OType *__restrict__ y, const int64_t stride_x_row,
    const int64_t stride_y_row, const int total_M, const int m_padded, const int n_cols,
    const uint8_t *__restrict__ scale_inv, const int64_t stride_scale_row,
    const int64_t stride_scale_col, const int scale_n_rows, const int scale_n_cols,
    const int64_t *__restrict__ group_offs, const int64_t *__restrict__ group_offs_padded, int G,
    const int block_size) {
    constexpr int VEC            = 16 / static_cast<int>(sizeof(OType));
    constexpr int CHUNKS_PER_ROW = BLOCK_N / VEC;
    constexpr int TOTAL_CHUNKS   = BLOCK_M * CHUNKS_PER_ROW;
    constexpr int COL_STEP       = THREADS_PER_BLOCK / BLOCK_M;

    const int nthreads = blockDim.x * blockDim.y;
    const int tid      = threadIdx.y * blockDim.x + threadIdx.x;
    const int c0       = blockIdx.x * BLOCK_N;
    const int r0       = blockIdx.y * BLOCK_M;

    __shared__ OType s_tile[BLOCK_M][BLOCK_N + 1];

    // Phase 1: dequant padded input [n_cols, m_padded] into smem tile.
    for (int ci = tid; ci < TOTAL_CHUNKS; ci += nthreads) {
        const int local_row = ci / CHUNKS_PER_ROW;
        const int local_col = (ci % CHUNKS_PER_ROW) * VEC;
        const int grow      = r0 + local_row;
        const int gcol      = c0 + local_col;
        if (grow < n_cols && gcol < m_padded) {
            const int col_block = gcol / block_size;
            uint8_t   e8m0      = static_cast<uint8_t>(E8M0_EXPONENT_BIAS);
            if (grow < scale_n_rows && col_block < scale_n_cols) {
                e8m0 = scale_inv[static_cast<int64_t>(grow) * stride_scale_row +
                                 static_cast<int64_t>(col_block) * stride_scale_col];
            }
            const float scale = e8m0_to_scale(e8m0);
            QType       x_reg[VEC];
            load_data<QType, VEC>(x + static_cast<int64_t>(grow) * stride_x_row + gcol, x_reg);
#pragma unroll
            for (int i = 0; i < VEC; ++i) {
                s_tile[local_row][local_col + i] =
                    static_cast<OType>(static_cast<float>(x_reg[i]) * scale);
            }
        }
    }
    __syncthreads();

    // Phase 2: scatter transposed tile into compact [total_M, n_cols].
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int n  = r0 + tx;
#pragma unroll
    for (int j = 0; j < BLOCK_N; j += COL_STEP) {
        const int m_padded_row = c0 + ty + j;
        if (n < n_cols && m_padded_row < m_padded) {
            const int64_t compact_m =
                grouped_padded_to_compact_m(m_padded_row, group_offs, group_offs_padded, G);
            if (compact_m >= 0 && compact_m < total_M) {
                y[static_cast<int64_t>(compact_m) * stride_y_row + static_cast<int64_t>(n)] =
                    s_tile[tx][ty + j];
            }
        }
    }
}

template <typename OType, typename QType>
void grouped_dequantize_mxfp8_impl(const QType *x, OType *y, const int64_t stride_x_row,
                                   const int64_t stride_y_row, const int total_M, const int n_rows,
                                   const int n_cols, const uint8_t *scale_inv,
                                   const int64_t stride_scale_row, const int64_t stride_scale_col,
                                   const int scale_n_rows, const int scale_n_cols,
                                   const int64_t *group_offs, const int64_t *group_offs_padded,
                                   int G, int block_size, bool use_rowwise, hipStream_t stream) {
    if (total_M == 0 || n_cols == 0)
        return;

    const dim3 block(BLOCK_M, THREADS_PER_BLOCK / BLOCK_M);
    if (use_rowwise) {
        const dim3 grid((n_cols + BLOCK_N - 1) / BLOCK_N, (total_M + BLOCK_M - 1) / BLOCK_M);
        grouped_dequantize_mxfp8_rowwise_kernel<OType, QType><<<grid, block, 0, stream>>>(
            x, y, stride_x_row, stride_y_row, total_M, n_cols, scale_inv, stride_scale_row,
            stride_scale_col, scale_n_rows, scale_n_cols, group_offs, group_offs_padded, G,
            block_size);
    } else {
        const dim3 grid((n_rows + BLOCK_N - 1) / BLOCK_N, (n_cols + BLOCK_M - 1) / BLOCK_M);
        grouped_dequantize_mxfp8_colwise_kernel<OType, QType><<<grid, block, 0, stream>>>(
            x, y, stride_x_row, stride_y_row, total_M, n_rows, n_cols, scale_inv, stride_scale_row,
            stride_scale_col, scale_n_rows, scale_n_cols, group_offs, group_offs_padded, G,
            block_size);
    }
}

template <typename OType, typename QType>
void dequantize_mxfp8_impl(const QType *x, OType *y, const int64_t stride_x_row,
                           const int64_t stride_x_col, const int64_t stride_y_row,
                           const int64_t stride_y_col, const int n_rows, const int n_cols,
                           const uint8_t *scale_inv, const int64_t stride_scale_row,
                           const int64_t stride_scale_col, const int scale_n_rows,
                           const int scale_n_cols, const int block_size, const bool use_rowwise,
                           hipStream_t stream) {
    (void) stride_x_col; // input is contiguous along columns (stride == 1)
    if (n_rows == 0 || n_cols == 0)
        return;

    // Each mode uses its own kernel + tile geometry (see the RW_*/CW_* constants).
    if (use_rowwise) {
        const int  cols_per_block = RW_TX * RW_ELEMS;
        const int  rows_per_block = RW_TY * RW_ROWS;
        const dim3 block(RW_TX, RW_TY);
        const dim3 grid((n_cols + cols_per_block - 1) / cols_per_block,
                        (n_rows + rows_per_block - 1) / rows_per_block);
        dequantize_mxfp8_rowwise_kernel<OType, QType><<<grid, block, 0, stream>>>(
            x, y, stride_x_row, stride_y_row, n_rows, n_cols, scale_inv, stride_scale_row,
            stride_scale_col, scale_n_rows, scale_n_cols, block_size);
    } else {
        const dim3 block(CW_BLOCK_M, CW_THREADS / CW_BLOCK_M);
        const dim3 grid((n_cols + CW_BLOCK_N - 1) / CW_BLOCK_N,
                        (n_rows + CW_BLOCK_M - 1) / CW_BLOCK_M);
        dequantize_mxfp8_colwise_kernel<OType, QType><<<grid, block, 0, stream>>>(
            x, y, stride_x_row, stride_y_row, stride_y_col, n_rows, n_cols, scale_inv,
            stride_scale_row, stride_scale_col, scale_n_rows, scale_n_cols, block_size);
    }
}

// ---------------------------------------------------------------------------
// Explicit instantiations
// ---------------------------------------------------------------------------
#define DECL_DEQUANT_MXFP8_INSTANCE(OType, QType)                                                  \
    template void dequantize_mxfp8_impl<OType, QType>(                                             \
        const QType *x, OType *y, const int64_t stride_x_row, const int64_t stride_x_col,          \
        const int64_t stride_y_row, const int64_t stride_y_col, const int n_rows,                  \
        const int n_cols, const uint8_t *scale_inv, const int64_t stride_scale_row,                \
        const int64_t stride_scale_col, const int scale_n_rows, const int scale_n_cols,            \
        const int block_size, const bool use_rowwise, hipStream_t stream);

DECL_DEQUANT_MXFP8_INSTANCE(dtype::float16, dtype::float8_e4m3)
DECL_DEQUANT_MXFP8_INSTANCE(dtype::float16, dtype::float8_e5m2)
DECL_DEQUANT_MXFP8_INSTANCE(dtype::bfloat16, dtype::float8_e4m3)
DECL_DEQUANT_MXFP8_INSTANCE(dtype::bfloat16, dtype::float8_e5m2)
DECL_DEQUANT_MXFP8_INSTANCE(dtype::float32, dtype::float8_e4m3)
DECL_DEQUANT_MXFP8_INSTANCE(dtype::float32, dtype::float8_e5m2)

#undef DECL_DEQUANT_MXFP8_INSTANCE

#define DECL_GROUPED_DEQUANT_MXFP8_INSTANCE(OType, QType)                                          \
    template void grouped_dequantize_mxfp8_impl<OType, QType>(                                     \
        const QType *x, OType *y, const int64_t stride_x_row, const int64_t stride_y_row,          \
        const int total_M, const int n_rows, const int n_cols, const uint8_t *scale_inv,           \
        const int64_t stride_scale_row, const int64_t stride_scale_col, const int scale_n_rows,    \
        const int scale_n_cols, const int64_t *group_offs, const int64_t *group_offs_padded,       \
        int G, int block_size, bool use_rowwise, hipStream_t stream);

DECL_GROUPED_DEQUANT_MXFP8_INSTANCE(dtype::float16, dtype::float8_e4m3)
DECL_GROUPED_DEQUANT_MXFP8_INSTANCE(dtype::float16, dtype::float8_e5m2)
DECL_GROUPED_DEQUANT_MXFP8_INSTANCE(dtype::bfloat16, dtype::float8_e4m3)
DECL_GROUPED_DEQUANT_MXFP8_INSTANCE(dtype::bfloat16, dtype::float8_e5m2)
DECL_GROUPED_DEQUANT_MXFP8_INSTANCE(dtype::float32, dtype::float8_e4m3)
DECL_GROUPED_DEQUANT_MXFP8_INSTANCE(dtype::float32, dtype::float8_e5m2)

#undef DECL_GROUPED_DEQUANT_MXFP8_INSTANCE

} // namespace primus_turbo
