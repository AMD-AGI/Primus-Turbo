// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
//
// See LICENSE for license information.

#include "primus_turbo/common.h"
#include "primus_turbo/device/reduce.cuh"
#include "primus_turbo/device/utils.cuh"
#include "primus_turbo/quantization.h"

namespace primus_turbo {

using namespace primus_turbo::dtype;

// Segment-padded group offsets (each segment rounded up to block_size). Used by the
// fused segment-m row+col quant below; computed on-device to avoid host sync.
template <typename IndexType>
__global__ void compute_padded_group_offs_device(const IndexType *group_lens_ptr,
                                                 IndexType       *padded_lens_ptr,
                                                 IndexType *padded_offs_ptr, const int group_num,
                                                 const IndexType block_size) {
    const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx == 0)
        padded_offs_ptr[0] = 0;
    if (idx < group_num) {
        IndexType cumsum = 0;
        for (int i = 0; i < idx; ++i) {
            cumsum += ((group_lens_ptr[i] + block_size - 1) / block_size) * block_size;
        }
        const IndexType padded_self =
            ((group_lens_ptr[idx] + block_size - 1) / block_size) * block_size;
        padded_lens_ptr[idx]     = padded_self;
        padded_offs_ptr[idx + 1] = cumsum + padded_self;
    }
}

template <typename IndexType>
void compute_padded_group_offs(const IndexType *group_lens_ptr, IndexType *padded_lens_ptr,
                               IndexType *padded_offs_ptr, const int64_t group_num,
                               const IndexType block_size, hipStream_t stream) {
    const int threads_per_block = 256;
    const int blocks = static_cast<int>((group_num + threads_per_block - 1) / threads_per_block);
    compute_padded_group_offs_device<IndexType>
        <<<dim3(blocks), dim3(threads_per_block), 0, stream>>>(
            group_lens_ptr, padded_lens_ptr, padded_offs_ptr, static_cast<int>(group_num),
            block_size);
}

template void compute_padded_group_offs<int64_t>(const int64_t *group_lens_ptr,
                                                 int64_t *padded_lens_ptr, int64_t *padded_offs_ptr,
                                                 const int64_t group_num, const int64_t block_size,
                                                 hipStream_t stream);

// Single-pass fused row + segment-padded col blockwise FP8 quant.
// One bf16 read of x [M_in, N]; vals cached in vregs (64 fp32/thread).
// Row amax: 16-thread sub-warp xor reduce → write row FP8 + scale per round.
// Col amax: per-thread col_amax[8] over 8 rounds → LDS reduce across 16 row-slots
// → broadcast scale; col-padded FP8 written from regs. ~9KB LDS, 3-4 blocks/CU.
template <typename FType, typename QType>
__launch_bounds__(256) __global__ void quant_fp8_blockwise_segment_m_row_col_kernel(
    const FType *__restrict__ x_ptr, QType *__restrict__ x_fp8_row_ptr,
    QType *__restrict__ x_fp8_col_padded_ptr, float *__restrict__ x_scales_row_ptr,
    float *__restrict__ x_scales_col_padded_ptr, const int64_t *__restrict__ group_offs_ptr,
    const int64_t *__restrict__ padded_group_offs_ptr, const int64_t M_in, const int64_t N,
    const int num_groups, const float fp8_max) {
    constexpr int BLOCK_SIZE      = 128;
    constexpr int PACK            = 8;
    constexpr int THREADS_PER_ROW = BLOCK_SIZE / PACK;           // 16
    constexpr int ROWS_PER_ROUND  = 256 / THREADS_PER_ROW;       // 16
    constexpr int ROUNDS          = BLOCK_SIZE / ROWS_PER_ROUND; // 8

    const int64_t pid_m         = blockIdx.x;
    const int64_t col_blk       = blockIdx.y;
    const int     tid           = threadIdx.x;
    const int     pack_idx      = tid % THREADS_PER_ROW;
    const int     load_row_base = tid / THREADS_PER_ROW;

    const int64_t M_padded        = padded_group_offs_ptr[num_groups];
    const int64_t pad_block_start = pid_m * BLOCK_SIZE;
    if (pad_block_start >= M_padded)
        return;

    int group_id = 0;
#pragma unroll 1
    for (int g = 0; g < num_groups; ++g) {
        if (pad_block_start >= padded_group_offs_ptr[g] &&
            pad_block_start < padded_group_offs_ptr[g + 1])
            group_id = g;
    }
    const int64_t orig_start = group_offs_ptr[group_id];
    const int64_t orig_end   = group_offs_ptr[group_id + 1];
    const int64_t pad_start  = padded_group_offs_ptr[group_id];

    const int64_t col_start = col_blk * BLOCK_SIZE;
    const float   clip_lo   = -static_cast<float>(fp8_max);
    const float   clip_hi   = static_cast<float>(fp8_max);

    float vals[ROUNDS][PACK];
    float col_amax_local[PACK];
#pragma unroll
    for (int i = 0; i < PACK; ++i)
        col_amax_local[i] = 0.f;

    __shared__ bool s_valid_row[BLOCK_SIZE];

#pragma unroll
    for (int r = 0; r < ROUNDS; ++r) {
        const int     local_m  = load_row_base + r * ROWS_PER_ROUND;
        const int     local_n  = pack_idx * PACK;
        const int64_t in_m     = orig_start + (pad_block_start + local_m - pad_start);
        const bool    valid_in = (in_m >= orig_start) && (in_m < orig_end) && (in_m < M_in);
        if (pack_idx == 0)
            s_valid_row[local_m] = valid_in;

        const int64_t gn = col_start + local_n;
        FType         buf[PACK];
        if (valid_in && gn + PACK <= N) {
            load_data<FType, PACK>(x_ptr + in_m * N + gn, buf);
        } else {
#pragma unroll
            for (int i = 0; i < PACK; ++i) {
                const int64_t cn = gn + i;
                buf[i] = (valid_in && cn < N) ? x_ptr[in_m * N + cn] : static_cast<FType>(0.f);
            }
        }

        float row_amax = 0.f;
#pragma unroll
        for (int i = 0; i < PACK; ++i) {
            const float v  = static_cast<float>(buf[i]);
            const float av = fabsf(v);
            vals[r][i]     = v;
            row_amax       = fmaxf(row_amax, av);
            if (valid_in)
                col_amax_local[i] = fmaxf(col_amax_local[i], av);
        }
#pragma unroll
        for (int offset = THREADS_PER_ROW >> 1; offset > 0; offset >>= 1) {
            row_amax = fmaxf(row_amax, __shfl_xor(row_amax, offset, THREADS_PER_ROW));
        }
        const float row_scale = static_cast<float>(fp8_max) / fmaxf(row_amax, 1e-4f);

        if (valid_in) {
            QType out[PACK];
#pragma unroll
            for (int i = 0; i < PACK; ++i) {
                out[i] = static_cast<QType>(fmaxf(fminf(vals[r][i] * row_scale, clip_hi), clip_lo));
            }
            if (gn + PACK <= N) {
                store_data<QType, PACK>(x_fp8_row_ptr + in_m * N + gn, out);
            } else {
#pragma unroll
                for (int i = 0; i < PACK; ++i) {
                    const int64_t cn = gn + i;
                    if (cn < N)
                        x_fp8_row_ptr[in_m * N + cn] = out[i];
                }
            }
            if (pack_idx == 0) {
                // Pshuffled [N_blocks, M_in]: matches the persistent fwd GEMM scale order.
                x_scales_row_ptr[col_blk * M_in + in_m] = 1.0f / row_scale;
            }
        }
    }

    __shared__ float s_col_partial[ROWS_PER_ROUND][BLOCK_SIZE]; // 8KB
#pragma unroll
    for (int i = 0; i < PACK; ++i) {
        s_col_partial[load_row_base][pack_idx * PACK + i] = col_amax_local[i];
    }
    __syncthreads();

    __shared__ float s_col_scale[BLOCK_SIZE];
    if (tid < BLOCK_SIZE) {
        float col_amax = 0.f;
#pragma unroll
        for (int rs = 0; rs < ROWS_PER_ROUND; ++rs) {
            col_amax = fmaxf(col_amax, s_col_partial[rs][tid]);
        }
        const float col_scale = static_cast<float>(fp8_max) / fmaxf(col_amax, 1e-4f);
        s_col_scale[tid]      = col_scale;
        const int64_t gn      = col_start + tid;
        if (gn < N)
            x_scales_col_padded_ptr[pid_m * N + gn] = 1.0f / col_scale;
    }
    __syncthreads();

#pragma unroll
    for (int r = 0; r < ROUNDS; ++r) {
        const int     local_m = load_row_base + r * ROWS_PER_ROUND;
        const int     local_n = pack_idx * PACK;
        const int64_t out_m   = pad_block_start + local_m;
        const int64_t gn      = col_start + local_n;
        const bool    valid   = s_valid_row[local_m];

        QType out[PACK];
#pragma unroll
        for (int i = 0; i < PACK; ++i) {
            const float v = valid ? vals[r][i] : 0.f;
            out[i] =
                static_cast<QType>(fmaxf(fminf(v * s_col_scale[local_n + i], clip_hi), clip_lo));
        }
        if (out_m < M_padded && gn + PACK <= N) {
            store_data<QType, PACK>(x_fp8_col_padded_ptr + out_m * N + gn, out);
        } else if (out_m < M_padded) {
#pragma unroll
            for (int i = 0; i < PACK; ++i) {
                const int64_t cn = gn + i;
                if (cn < N)
                    x_fp8_col_padded_ptr[out_m * N + cn] = out[i];
            }
        }
    }
}

template <typename FType, typename QType>
void quantize_blockwise_segment_m_row_col_impl(const FType *x, QType *y_row, QType *y_col_padded,
                                               float *scales_row, float *scales_col_padded,
                                               const int64_t *group_offs,
                                               const int64_t *padded_group_offs, const int64_t M_in,
                                               const int64_t N, const int64_t M_padded_max,
                                               const int num_groups, const float fp8_max,
                                               hipStream_t stream) {
    constexpr int BLOCK_SIZE = 128;
    const int64_t m_blocks   = (M_padded_max + BLOCK_SIZE - 1) / BLOCK_SIZE;
    const int64_t n_blocks   = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3          grid(m_blocks, n_blocks);
    quant_fp8_blockwise_segment_m_row_col_kernel<FType, QType><<<grid, dim3(256), 0, stream>>>(
        x, y_row, y_col_padded, scales_row, scales_col_padded, group_offs, padded_group_offs, M_in,
        N, num_groups, fp8_max);
}

#define DECL_QUANT_BLOCKWISE_SEGM_INSTANCE(FType, QType)                                           \
    template void quantize_blockwise_segment_m_row_col_impl<FType, QType>(                         \
        const FType *x, QType *y_row, QType *y_col_padded, float *scales_row,                      \
        float *scales_col_padded, const int64_t *group_offs, const int64_t *padded_group_offs,     \
        const int64_t M_in, const int64_t N, const int64_t M_padded_max, const int num_groups,     \
        const float fp8_max, hipStream_t stream);
DECL_QUANT_BLOCKWISE_SEGM_INSTANCE(dtype::bfloat16, dtype::float8_e4m3)
DECL_QUANT_BLOCKWISE_SEGM_INSTANCE(dtype::bfloat16, dtype::float8_e5m2)
DECL_QUANT_BLOCKWISE_SEGM_INSTANCE(dtype::float16, dtype::float8_e4m3)
DECL_QUANT_BLOCKWISE_SEGM_INSTANCE(dtype::float16, dtype::float8_e5m2)
#undef DECL_QUANT_BLOCKWISE_SEGM_INSTANCE

// Weight blockwise FP8 quant: 3D weight [B, M, N], one scalar scale per [128,128]
// tile. 256 threads/block; vec-8 packed loads; BlockReduce<AbsMax>.
template <typename FType, typename QType>
__launch_bounds__(256) __global__
    void quant_fp8_blockwise_for_weight_kernel(const FType *__restrict__ w_ptr,
                                               QType *__restrict__ w_fp8_ptr,
                                               float *__restrict__ w_scales_inv_ptr,
                                               const int64_t M, const int64_t N,
                                               const float fp8_max) {
    constexpr int BLOCK_SIZE      = 128;
    constexpr int PACK            = 8;
    constexpr int THREADS_PER_ROW = BLOCK_SIZE / PACK;           // 16
    constexpr int ROWS_PER_ROUND  = 256 / THREADS_PER_ROW;       // 16
    constexpr int ROUNDS          = BLOCK_SIZE / ROWS_PER_ROUND; // 8

    const int64_t bid           = blockIdx.x;
    const int64_t row_blk       = blockIdx.y;
    const int64_t col_blk       = blockIdx.z;
    const int     tid           = threadIdx.x;
    const int     pack_idx      = tid % THREADS_PER_ROW;
    const int     load_row_base = tid / THREADS_PER_ROW;

    const int64_t row_start = row_blk * BLOCK_SIZE;
    const int64_t col_start = col_blk * BLOCK_SIZE;
    const int64_t batch_off = bid * M * N;

    float vals[ROUNDS][PACK];
    float amax = 0.f;

#pragma unroll
    for (int r = 0; r < ROUNDS; ++r) {
        const int     local_m = load_row_base + r * ROWS_PER_ROUND;
        const int     local_n = pack_idx * PACK;
        const int64_t gm      = row_start + local_m;
        const int64_t gn      = col_start + local_n;
        FType         buf[PACK];
        if (gm < M && gn + PACK <= N) {
            load_data<FType, PACK>(w_ptr + batch_off + gm * N + gn, buf);
        } else {
#pragma unroll
            for (int i = 0; i < PACK; ++i) {
                const int64_t cn = gn + i;
                buf[i] =
                    (gm < M && cn < N) ? w_ptr[batch_off + gm * N + cn] : static_cast<FType>(0.f);
            }
        }
#pragma unroll
        for (int i = 0; i < PACK; ++i) {
            const float v = static_cast<float>(buf[i]);
            vals[r][i]    = v;
            amax          = fmaxf(amax, fabsf(v));
        }
    }

    amax = BlockReduce<AbsMaxOp, float>(amax);
    // Clamp eps matches Triton reference (tl.maximum(w_tile_max, 1e-4)).
    const float scale   = static_cast<float>(fp8_max) / fmaxf(amax, 1e-4f);
    const float clip_lo = -static_cast<float>(fp8_max);
    const float clip_hi = static_cast<float>(fp8_max);

    if (tid == 0) {
        const int64_t sn = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
        const int64_t sm = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
        w_scales_inv_ptr[bid * sm * sn + row_blk * sn + col_blk] = 1.0f / scale;
    }

#pragma unroll
    for (int r = 0; r < ROUNDS; ++r) {
        const int     local_m = load_row_base + r * ROWS_PER_ROUND;
        const int     local_n = pack_idx * PACK;
        const int64_t gm      = row_start + local_m;
        const int64_t gn      = col_start + local_n;
        QType         out[PACK];
#pragma unroll
        for (int i = 0; i < PACK; ++i) {
            out[i] = static_cast<QType>(fmaxf(fminf(vals[r][i] * scale, clip_hi), clip_lo));
        }
        if (gm < M && gn + PACK <= N) {
            store_data<QType, PACK>(w_fp8_ptr + batch_off + gm * N + gn, out);
        } else if (gm < M) {
#pragma unroll
            for (int i = 0; i < PACK; ++i) {
                const int64_t cn = gn + i;
                if (cn < N)
                    w_fp8_ptr[batch_off + gm * N + cn] = out[i];
            }
        }
    }
}

template <typename FType, typename QType>
void quantize_blockwise_for_weight_impl(const FType *w, QType *w_fp8, float *w_scales_inv,
                                        const int64_t B, const int64_t M, const int64_t N,
                                        const float fp8_max, hipStream_t stream) {
    constexpr int BLOCK_SIZE = 128;
    const int64_t m_blocks   = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    const int64_t n_blocks   = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3          grid(B, m_blocks, n_blocks);
    quant_fp8_blockwise_for_weight_kernel<FType, QType>
        <<<grid, dim3(256), 0, stream>>>(w, w_fp8, w_scales_inv, M, N, fp8_max);
}

#define DECL_QUANT_BLOCKWISE_FOR_WEIGHT_INSTANCE(FType, QType)                                     \
    template void quantize_blockwise_for_weight_impl<FType, QType>(                                \
        const FType *w, QType *w_fp8, float *w_scales_inv, const int64_t B, const int64_t M,       \
        const int64_t N, const float fp8_max, hipStream_t stream);
DECL_QUANT_BLOCKWISE_FOR_WEIGHT_INSTANCE(dtype::bfloat16, dtype::float8_e4m3)
DECL_QUANT_BLOCKWISE_FOR_WEIGHT_INSTANCE(dtype::bfloat16, dtype::float8_e5m2)
DECL_QUANT_BLOCKWISE_FOR_WEIGHT_INSTANCE(dtype::float16, dtype::float8_e4m3)
DECL_QUANT_BLOCKWISE_FOR_WEIGHT_INSTANCE(dtype::float16, dtype::float8_e5m2)
#undef DECL_QUANT_BLOCKWISE_FOR_WEIGHT_INSTANCE

} // namespace primus_turbo
