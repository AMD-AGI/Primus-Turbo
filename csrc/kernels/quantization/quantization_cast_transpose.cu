// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
//
// See LICENSE for license information.
//
// Fused cast + transpose + abs-max FP8 kernels (TE-style delayed scaling).
//
// Given a 2D row-major input [M, N], produces:
//   cast_out  [M, N]  -- FP8 quantized, row-major
//   trans_out [N, M]  -- FP8 quantized, transposed (contiguous)
//   amax      scalar  -- abs-max of the un-scaled input (optional)

#include <algorithm>

#include "primus_turbo/common.h"
#include "primus_turbo/quantization.h"

namespace primus_turbo {

using namespace primus_turbo::dtype;

PRIMUS_TURBO_DEVICE void atomicMaxFloat(float *addr, float val) {
    if (val <= 0.0f)
        return;
    unsigned int *addr_as_uint = reinterpret_cast<unsigned int *>(addr);
    unsigned int  old          = __float_as_uint(*addr);
    unsigned int  assumed;
    do {
        assumed = old;
        if (__uint_as_float(assumed) >= val)
            break;
        old = atomicCAS(addr_as_uint, assumed, __float_as_uint(val));
    } while (assumed != old);
}

template <int TILE_DIM, typename FType, typename QType, bool COMPUTE_AMAX,
          typename ComputeType = float>
__launch_bounds__(TILE_DIM * (TILE_DIM / 4)) __global__
    void cast_transpose_with_amax_kernel(const FType *__restrict__ input,
                                         const float *__restrict__ scale_ptr,
                                         QType *__restrict__ cast_out,
                                         QType *__restrict__ trans_out,
                                         float *__restrict__ amax_ptr,
                                         const int64_t rows,
                                         const int64_t cols) {
    constexpr int BLOCK_ROWS = TILE_DIM / 4;
    constexpr int SMEM_COLS  = TILE_DIM + 1;
    __shared__ uint8_t smem_raw[TILE_DIM * SMEM_COLS * sizeof(QType)];
    auto *smem = reinterpret_cast<QType(*)[SMEM_COLS]>(smem_raw);

    const ComputeType scale    = static_cast<ComputeType>(scale_ptr[0]);
    const ComputeType CLIP_MIN = static_cast<ComputeType>(std::numeric_limits<QType>::lowest());
    const ComputeType CLIP_MAX = static_cast<ComputeType>(std::numeric_limits<QType>::max());

    const int64_t total_tiles_x = (cols + TILE_DIM - 1) / TILE_DIM;
    const int64_t total_tiles_y = (rows + TILE_DIM - 1) / TILE_DIM;
    const int64_t total_tiles   = total_tiles_x * total_tiles_y;

    ComputeType local_amax = 0.0f;

    for (int64_t tile_id = blockIdx.x; tile_id < total_tiles; tile_id += gridDim.x) {
        const int64_t tile_x = tile_id % total_tiles_x;
        const int64_t tile_y = tile_id / total_tiles_x;

        const int64_t base_col = tile_x * TILE_DIM;
        const int64_t base_row = tile_y * TILE_DIM;

        const int64_t col = base_col + threadIdx.x;

#pragma unroll
        for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
            const int64_t row = base_row + threadIdx.y + j;
            if (row < rows && col < cols) {
                const ComputeType val = static_cast<ComputeType>(input[row * cols + col]);
                if (COMPUTE_AMAX) {
                    local_amax = fmaxf(local_amax, fabsf(val));
                }
                const QType q = static_cast<QType>(
                    fmaxf(fminf(val * scale, CLIP_MAX), CLIP_MIN));
                cast_out[row * cols + col] = q;
                smem[threadIdx.y + j][threadIdx.x] = q;
            }
        }

        __syncthreads();

        const int64_t t_col = base_row + threadIdx.x;
#pragma unroll
        for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
            const int64_t t_row = base_col + threadIdx.y + j;
            if (t_row < cols && t_col < rows) {
                trans_out[t_row * rows + t_col] = smem[threadIdx.x][threadIdx.y + j];
            }
        }

        __syncthreads();
    }

    if (COMPUTE_AMAX) {
        constexpr int BLOCK_SIZE_2D = TILE_DIM * BLOCK_ROWS;
        const int linear_tid = threadIdx.y * TILE_DIM + threadIdx.x;
        auto *amax_smem = reinterpret_cast<float*>(smem_raw);
        amax_smem[linear_tid] = local_amax;
        __syncthreads();
        for (int s = BLOCK_SIZE_2D / 2; s > 0; s >>= 1) {
            if (linear_tid < s) {
                amax_smem[linear_tid] = fmaxf(amax_smem[linear_tid],
                                               amax_smem[linear_tid + s]);
            }
            __syncthreads();
        }
        if (linear_tid == 0) {
            atomicMaxFloat(amax_ptr, amax_smem[0]);
        }
    }
}

template <typename FType, typename QType, typename ComputeType>
void cast_transpose_with_amax_impl(const FType *input, const float *scale,
                                   QType *cast_out, QType *trans_out,
                                   float *amax_out, const int64_t rows,
                                   const int64_t cols, hipStream_t stream) {
    constexpr int TILE_DIM   = 32;
    constexpr int BLOCK_ROWS = TILE_DIM / 4;  // 8
    const dim3 block(TILE_DIM, BLOCK_ROWS);    // 256 threads

    const int64_t total_tiles_x = (cols + TILE_DIM - 1) / TILE_DIM;
    const int64_t total_tiles_y = (rows + TILE_DIM - 1) / TILE_DIM;
    const int64_t total_tiles   = total_tiles_x * total_tiles_y;
    constexpr int MAX_BLOCKS    = 1024;
    const int n_blocks = static_cast<int>(
        std::min(total_tiles, static_cast<int64_t>(MAX_BLOCKS)));

    if (amax_out != nullptr) {
        cast_transpose_with_amax_kernel<TILE_DIM, FType, QType, true, ComputeType>
            <<<n_blocks, block, 0, stream>>>(input, scale, cast_out, trans_out,
                                             amax_out, rows, cols);
    } else {
        cast_transpose_with_amax_kernel<TILE_DIM, FType, QType, false, ComputeType>
            <<<n_blocks, block, 0, stream>>>(input, scale, cast_out, trans_out,
                                             amax_out, rows, cols);
    }
}

// ---------------------------------------------------------------------------
// Explicit instantiations
// ---------------------------------------------------------------------------
#define DECL_CAST_TRANSPOSE_INSTANCE(FType, QType)                                                 \
    template void cast_transpose_with_amax_impl<FType, QType, float>(                              \
        const FType *input, const float *scale, QType *cast_out, QType *trans_out,                 \
        float *amax_out, const int64_t rows, const int64_t cols, hipStream_t stream);

DECL_CAST_TRANSPOSE_INSTANCE(dtype::float16, dtype::float8_e4m3)
DECL_CAST_TRANSPOSE_INSTANCE(dtype::float16, dtype::float8_e5m2)
DECL_CAST_TRANSPOSE_INSTANCE(dtype::bfloat16, dtype::float8_e4m3)
DECL_CAST_TRANSPOSE_INSTANCE(dtype::bfloat16, dtype::float8_e5m2)
DECL_CAST_TRANSPOSE_INSTANCE(dtype::float32, dtype::float8_e4m3)
DECL_CAST_TRANSPOSE_INSTANCE(dtype::float32, dtype::float8_e5m2)

#undef DECL_CAST_TRANSPOSE_INSTANCE

} // namespace primus_turbo
