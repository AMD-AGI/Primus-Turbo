// Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// See LICENSE for license information.
#include "primus_turbo/common.h"
#include "primus_turbo/device/utils.cuh"
#include "primus_turbo/quantization.h"

namespace primus_turbo {

using namespace primus_turbo::dtype;
using namespace primus_turbo::detail;

// Each block processes one BLOCK_M x BLOCK_N tile; THREADS_PER_BLOCK threads are
// laid out as blockDim (BLOCK_M, THREADS_PER_BLOCK / BLOCK_M). BLOCK_M == 64 so a
// full 64-lane wavefront covers one tile row (128B coalesced transposed writes /
// 64B coalesced reads in the colwise transpose).
constexpr int WARP_SIZE         = 64;  // AMD wavefront size
constexpr int THREADS_PER_BLOCK = 512; // 8 warps per block
constexpr int WARPS_PER_BLOCK   = THREADS_PER_BLOCK / WARP_SIZE;
constexpr int BLOCK_M           = 64; // rows per block (tile height)
constexpr int BLOCK_N           = 64; // cols per block (tile width)

// ---------------------------------------------------------------------------
// MXFP8 de-quantization kernel (rowwise + colwise in one entry).
// ---------------------------------------------------------------------------
template <typename OType, typename QType, bool USE_ROWWISE>
__global__ void dequantize_mxfp8_kernel(const QType *__restrict__ x, OType *__restrict__ y,
                                        const int64_t stride_x_row, const int64_t stride_y_row,
                                        const int64_t stride_y_col, const int n_rows,
                                        const int     n_cols, const uint8_t *__restrict__ scale_inv,
                                        const int64_t stride_scale_row,
                                        const int64_t stride_scale_col, const int scale_n_rows,
                                        const int scale_n_cols, const int block_size) {
    constexpr int VEC            = 16 / static_cast<int>(sizeof(OType)); // cols per chunk
    constexpr int CHUNKS_PER_ROW = BLOCK_N / VEC;                        // VEC-chunks per tile row
    constexpr int TOTAL_CHUNKS   = BLOCK_M * CHUNKS_PER_ROW;
    constexpr int COL_STEP       = THREADS_PER_BLOCK / BLOCK_M; // == blockDim.y

    // Both modes share the same launch: a BLOCK_M x BLOCK_N tile per block,
    // blockDim (BLOCK_M, COL_STEP). Threads walk the tile in VEC-column chunks
    // via a flattened tid.
    const int nthreads = blockDim.x * blockDim.y;
    const int tid      = threadIdx.y * blockDim.x + threadIdx.x;
    const int c0       = blockIdx.x * BLOCK_N;
    const int r0       = blockIdx.y * BLOCK_M;

    if constexpr (USE_ROWWISE) {
        // Read VEC contiguous columns + cast/scale -> contiguous store.
        for (int ci = tid; ci < TOTAL_CHUNKS; ci += nthreads) {
            const int row = r0 + ci / CHUNKS_PER_ROW;
            const int col = c0 + (ci % CHUNKS_PER_ROW) * VEC;
            if (row < n_rows && col < n_cols) {
                const int col_block = col / block_size;
                uint8_t   e8m0      = static_cast<uint8_t>(E8M0_EXPONENT_BIAS);
                if (row < scale_n_rows && col_block < scale_n_cols) {
                    e8m0 = scale_inv[static_cast<int64_t>(row) * stride_scale_row +
                                     static_cast<int64_t>(col_block) * stride_scale_col];
                }
                const float scale = e8m0_to_scale(e8m0);
                QType       x_reg[VEC];
                OType       y_reg[VEC];
                load_data<QType, VEC>(x + static_cast<int64_t>(row) * stride_x_row + col, x_reg);
#pragma unroll
                for (int i = 0; i < VEC; ++i) {
                    y_reg[i] = static_cast<OType>(static_cast<float>(x_reg[i]) * scale);
                }
                store_data<OType, VEC>(y + static_cast<int64_t>(row) * stride_y_row + col, y_reg);
            }
        }
    } else {
        __shared__ OType s_tile[BLOCK_M][BLOCK_N + 1]; // +1 mitigates bank conflicts

        // Phase 1: vectorized read + dequant -> row-major smem (s_tile[row][col]).
        for (int ci = tid; ci < TOTAL_CHUNKS; ci += nthreads) {
            const int local_row = ci / CHUNKS_PER_ROW;
            const int local_col = (ci % CHUNKS_PER_ROW) * VEC;
            const int grow      = r0 + local_row;
            const int gcol      = c0 + local_col;
            if (grow < n_rows && gcol < n_cols) {
                const int col_block = gcol / block_size; // VEC cols share one scale
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

        // Phase 2: coalesced transposed write (threadIdx.x -> output row).
        const int tx   = threadIdx.x;
        const int ty   = threadIdx.y;
        const int orow = r0 + tx;
#pragma unroll
        for (int j = 0; j < BLOCK_N; j += COL_STEP) {
            const int ocol = c0 + ty + j;
            if (orow < n_rows && ocol < n_cols) {
                y[static_cast<int64_t>(orow) * stride_y_col +
                  static_cast<int64_t>(ocol) * stride_y_row] = s_tile[tx][ty + j];
            }
        }
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

    // One kernel, one launch geometry for both modes: a BLOCK_M x BLOCK_N tile
    // per block, blockDim (BLOCK_M, THREADS_PER_BLOCK / BLOCK_M).
    const dim3 block(BLOCK_M, THREADS_PER_BLOCK / BLOCK_M);
    const dim3 grid((n_cols + BLOCK_N - 1) / BLOCK_N, (n_rows + BLOCK_M - 1) / BLOCK_M);
    if (use_rowwise) {
        dequantize_mxfp8_kernel<OType, QType, true><<<grid, block, 0, stream>>>(
            x, y, stride_x_row, stride_y_row, stride_y_col, n_rows, n_cols, scale_inv,
            stride_scale_row, stride_scale_col, scale_n_rows, scale_n_cols, block_size);
    } else {
        dequantize_mxfp8_kernel<OType, QType, false><<<grid, block, 0, stream>>>(
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

} // namespace primus_turbo
