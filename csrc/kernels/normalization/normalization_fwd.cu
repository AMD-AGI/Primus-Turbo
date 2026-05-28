// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
//
// See LICENSE for license information.

#include <algorithm>

#include "primus_turbo/device/reduce.cuh"
#include "primus_turbo/device/utils.cuh"
#include "primus_turbo/normalization.h"

namespace primus_turbo {

using namespace primus_turbo::dtype;

// LDGS = number of vectorized load strides per row this kernel statically
// handles. Each thread holds LDGS * UNROLL elements in registers across the
// two passes — single-load forward.
//
//   stride per pass = blockDim.x * UNROLL elements
//   row coverage    = LDGS * stride
template <typename T, int LDGS, int UNROLL>
__global__ void rmsnorm_fwd_kernel(const T *__restrict__ input, const T *__restrict__ gamma,
                                   T *__restrict__ output, float *__restrict__ rs_out,
                                   const int64_t inner_len, const int64_t outer_len,
                                   const float epsilon) {
    const int     tid          = threadIdx.x;
    const int     blocksize    = blockDim.x;
    const int64_t start_offset = (int64_t) tid * UNROLL;
    const int64_t stride       = (int64_t) blocksize * UNROLL;

    // Load gamma once per CTA lifetime; reused across all rows this CTA handles.
    T gamma_regs[LDGS][UNROLL];
#pragma unroll
    for (int i = 0; i < LDGS; ++i) {
        const int64_t offset = start_offset + (int64_t) i * stride;
        if (offset < inner_len) {
            load_data<T, UNROLL>(gamma + offset, gamma_regs[i]);
        } else {
#pragma unroll
            for (int j = 0; j < UNROLL; ++j)
                gamma_regs[i][j] = T(0);
        }
    }

    const float inv_n = 1.0f / static_cast<float>(inner_len);

    // Grid-stride over rows.
    for (int64_t row = blockIdx.x; row < outer_len; row += gridDim.x) {
        const T *input_ptr  = input + row * inner_len;
        T       *output_ptr = output + row * inner_len;

        // Pass 1: load x into registers, accumulate squared sum.
        T     x_regs[LDGS][UNROLL];
        float local_sq = 0.0f;
#pragma unroll
        for (int i = 0; i < LDGS; ++i) {
            const int64_t offset = start_offset + (int64_t) i * stride;
            if (offset < inner_len) {
                load_data<T, UNROLL>(input_ptr + offset, x_regs[i]);
#pragma unroll
                for (int j = 0; j < UNROLL; ++j) {
                    const float v = static_cast<float>(x_regs[i][j]);
                    local_sq += v * v;
                }
            } else {
#pragma unroll
                for (int j = 0; j < UNROLL; ++j)
                    x_regs[i][j] = T(0);
            }
        }

        // BlockReduce's trailing __syncthreads guards the smem write, not the
        // read in `return smem[0]`. The leading sync prevents a fast warp from
        // re-entering BlockReduce on the next row and overwriting smem[0]
        // before slower warps have finished reading it.
        __syncthreads();
        const float mean_sq = BlockReduce<SumOp, float>(local_sq) * inv_n;
        const float rs      = rsqrtf(mean_sq + epsilon);

        if (tid == 0) {
            rs_out[row] = rs;
        }

        // Pass 2: write y = (x * rs) * gamma. x is already in registers — no reload.
        T y_regs[UNROLL];
#pragma unroll
        for (int i = 0; i < LDGS; ++i) {
            const int64_t offset = start_offset + (int64_t) i * stride;
            if (offset < inner_len) {
#pragma unroll
                for (int j = 0; j < UNROLL; ++j) {
                    const float v = static_cast<float>(x_regs[i][j]) * rs *
                                    static_cast<float>(gamma_regs[i][j]);
                    y_regs[j] = static_cast<T>(v);
                }
                store_data<T, UNROLL>(output_ptr + offset, y_regs);
            }
        }
    }
}

// Warp-per-row variant: when cols fit in a single warp's vector load
// (THREADS_PER_WARP * UNROLL elements), pack multiple rows per CTA, one per
// warp. No __syncthreads, no shared memory — pure warp shuffles.
template <typename T, int UNROLL>
__global__ void rmsnorm_fwd_warp_per_row_kernel(const T *__restrict__ input,
                                                const T *__restrict__ gamma, T *__restrict__ output,
                                                float *__restrict__ rs_out, const int64_t inner_len,
                                                const int64_t outer_len, const float epsilon) {
    const int     warps_per_block = blockDim.x / THREADS_PER_WARP;
    const int     warp_id         = threadIdx.x / THREADS_PER_WARP;
    const int     lane            = threadIdx.x % THREADS_PER_WARP;
    const int64_t start_offset    = (int64_t) lane * UNROLL;

    // Load gamma once per CTA-life (per warp; same data, but cheap).
    T gamma_regs[UNROLL];
    if (start_offset < inner_len) {
        load_data<T, UNROLL>(gamma + start_offset, gamma_regs);
    } else {
#pragma unroll
        for (int j = 0; j < UNROLL; ++j)
            gamma_regs[j] = T(0);
    }

    const float inv_n = 1.0f / static_cast<float>(inner_len);

    const int64_t global_warp_id = (int64_t) blockIdx.x * warps_per_block + warp_id;
    const int64_t total_warps    = (int64_t) gridDim.x * warps_per_block;

    for (int64_t row = global_warp_id; row < outer_len; row += total_warps) {
        const T *input_ptr  = input + row * inner_len;
        T       *output_ptr = output + row * inner_len;

        T     x_regs[UNROLL];
        float local_sq = 0.0f;
        if (start_offset < inner_len) {
            load_data<T, UNROLL>(input_ptr + start_offset, x_regs);
#pragma unroll
            for (int j = 0; j < UNROLL; ++j) {
                const float v = static_cast<float>(x_regs[j]);
                local_sq += v * v;
            }
        } else {
#pragma unroll
            for (int j = 0; j < UNROLL; ++j)
                x_regs[j] = T(0);
        }

        local_sq       = WarpReduce<SumOp, float>(local_sq);
        const float rs = rsqrtf(local_sq * inv_n + epsilon);

        if (lane == 0)
            rs_out[row] = rs;

        if (start_offset < inner_len) {
            T y_regs[UNROLL];
#pragma unroll
            for (int j = 0; j < UNROLL; ++j) {
                const float v =
                    static_cast<float>(x_regs[j]) * rs * static_cast<float>(gamma_regs[j]);
                y_regs[j] = static_cast<T>(v);
            }
            store_data<T, UNROLL>(output_ptr + start_offset, y_regs);
        }
    }
}

template <typename T, int UNROLL>
static void launch_fwd(const T *input, const T *gamma, T *output, float *rs,
                       const int64_t inner_len, const int64_t outer_len, const float epsilon,
                       const int block, const int grid, hipStream_t stream) {
    // Dispatch on LDGS (compile-time row-coverage factor). LDGS=1 covers cols
    // up to block * UNROLL (= 8192 for block=1024, UNROLL=8 in bf16).
    const int64_t span = (int64_t) block * UNROLL;
    PRIMUS_TURBO_CHECK(inner_len <= 8 * span,
                       "rmsnorm fwd: inner_len exceeds LDGS=8 capacity for this dtype + block; "
                       "the kernel's LDGS dispatch chain caps at 8 — extend it or pick a larger "
                       "block.");
    if (inner_len <= span) {
        rmsnorm_fwd_kernel<T, 1, UNROLL>
            <<<grid, block, 0, stream>>>(input, gamma, output, rs, inner_len, outer_len, epsilon);
    } else if (inner_len <= 2 * span) {
        rmsnorm_fwd_kernel<T, 2, UNROLL>
            <<<grid, block, 0, stream>>>(input, gamma, output, rs, inner_len, outer_len, epsilon);
    } else if (inner_len <= 4 * span) {
        rmsnorm_fwd_kernel<T, 4, UNROLL>
            <<<grid, block, 0, stream>>>(input, gamma, output, rs, inner_len, outer_len, epsilon);
    } else {
        rmsnorm_fwd_kernel<T, 8, UNROLL>
            <<<grid, block, 0, stream>>>(input, gamma, output, rs, inner_len, outer_len, epsilon);
    }
}

template <typename T>
void rmsnorm_fwd_impl(const T *input, const T *gamma, T *output, float *rs, const int64_t inner_len,
                      const int64_t outer_len, const float epsilon, const int64_t target_ctas,
                      hipStream_t stream) {
    constexpr int UNROLL = sizeof(uint4) / sizeof(T);

    const bool aligned = (inner_len % UNROLL == 0);

    // Fast path: cols fit in a single warp's vector tile. Pack
    // RMSNORM_WARPS_PER_BLOCK rows per CTA, one row per warp. No syncthreads,
    // no smem.
    const int64_t warp_span = (int64_t) THREADS_PER_WARP * UNROLL;
    if (aligned && inner_len <= warp_span) {
        constexpr int W             = RMSNORM_WARPS_PER_BLOCK;
        const int     block         = W * THREADS_PER_WARP;
        const int64_t blocks_needed = (outer_len + W - 1) / W;
        const int64_t max_blocks    = (target_ctas + W - 1) / W;
        const int     grid          = static_cast<int>(std::min(blocks_needed, max_blocks));
        rmsnorm_fwd_warp_per_row_kernel<T, UNROLL>
            <<<grid, block, 0, stream>>>(input, gamma, output, rs, inner_len, outer_len, epsilon);
        return;
    }

    // Unaligned path uses UNROLL=1; block must still fit the row
    // (block * UNROLL * LDGS_MAX must cover inner_len, and LDGS_MAX = 8).
    const int block = rmsnorm_pick_blocksize(inner_len, aligned ? UNROLL : 1);
    const int grid  = static_cast<int>(std::min<int64_t>(outer_len, target_ctas));

    if (aligned) {
        launch_fwd<T, UNROLL>(input, gamma, output, rs, inner_len, outer_len, epsilon, block, grid,
                              stream);
    } else {
        launch_fwd<T, 1>(input, gamma, output, rs, inner_len, outer_len, epsilon, block, grid,
                         stream);
    }
}

template void rmsnorm_fwd_impl<float>(const float *, const float *, float *, float *, const int64_t,
                                      const int64_t, const float, const int64_t, hipStream_t);
template void rmsnorm_fwd_impl<float16>(const float16 *, const float16 *, float16 *, float *,
                                        const int64_t, const int64_t, const float, const int64_t,
                                        hipStream_t);
template void rmsnorm_fwd_impl<bfloat16>(const bfloat16 *, const bfloat16 *, bfloat16 *, float *,
                                         const int64_t, const int64_t, const float, const int64_t,
                                         hipStream_t);

} // namespace primus_turbo
