/***************************************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE for license information.
 **************************************************************************************************/

#include "primus_turbo/deoscillation.h"

#include <cmath>

namespace primus_turbo {

namespace {

constexpr int kThreads = 256;

__global__ void weight_deosc_update_kernel(const dtype::bfloat16 *__restrict__ current,
                                           const dtype::bfloat16 *__restrict__ current_qdq,
                                           const dtype::bfloat16 *__restrict__ previous,
                                           const dtype::bfloat16 *__restrict__ previous_qdq,
                                           float *__restrict__ dist, float *__restrict__ dist_qdq,
                                           int64_t numel) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= numel)
        return;

    // PyTorch evaluates BF16 - BF16 in BF16 before promoting the absolute
    // delta into the FP32 accumulator. Preserve that intermediate rounding.
    const dtype::bfloat16 delta     = current[index] - previous[index];
    const dtype::bfloat16 delta_qdq = current_qdq[index] - previous_qdq[index];

    dist[index] += fabsf(static_cast<float>(delta));
    dist_qdq[index] += fabsf(static_cast<float>(delta_qdq));
}

__global__ void weight_deosc_close_kernel(float *__restrict__ master,
                                          dtype::bfloat16 *__restrict__ previous,
                                          const dtype::bfloat16 *__restrict__ current_qdq,
                                          float *__restrict__ dist, float *__restrict__ dist_qdq,
                                          unsigned long long *__restrict__ reset_count,
                                          int64_t numel, float ratio_threshold, float eps,
                                          bool collect_count) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    bool          reset = false;

    if (index < numel) {
        const float distance     = dist[index];
        const float distance_qdq = dist_qdq[index];
        // Match torch.clamp(min=eps): retain NaN rather than using fmaxf,
        // whose NaN handling differs from PyTorch's elementwise clamp.
        const float denominator = distance < eps ? eps : distance;
        const float ratio       = distance_qdq / denominator;
        reset                   = distance > 0.0f && ratio >= ratio_threshold;

        if (reset) {
            const dtype::bfloat16 snapped = current_qdq[index];
            master[index]                 = static_cast<float>(snapped);
            previous[index]               = snapped;
        }

        dist[index]     = 0.0f;
        dist_qdq[index] = 0.0f;
    }

    if (collect_count) {
        // Reduce locally so diagnostic logging performs one global atomic per
        // thread block instead of one atomic per snapped element.
        __shared__ unsigned int block_counts[kThreads];
        block_counts[threadIdx.x] = reset ? 1U : 0U;
        __syncthreads();
        for (int stride = kThreads / 2; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride)
                block_counts[threadIdx.x] += block_counts[threadIdx.x + stride];
            __syncthreads();
        }
        if (threadIdx.x == 0 && block_counts[0] != 0)
            atomicAdd(reset_count, static_cast<unsigned long long>(block_counts[0]));
    }
}

} // namespace

void weight_deosc_update(const dtype::bfloat16 *current, const dtype::bfloat16 *current_qdq,
                         const dtype::bfloat16 *previous, const dtype::bfloat16 *previous_qdq,
                         float *dist, float *dist_qdq, int64_t numel, hipStream_t stream) {
    if (numel == 0)
        return;
    const dim3 block(kThreads);
    const dim3 grid(DIVUP<int64_t>(numel, kThreads));
    weight_deosc_update_kernel<<<grid, block, 0, stream>>>(current, current_qdq, previous,
                                                           previous_qdq, dist, dist_qdq, numel);
}

void weight_deosc_close(float *master, dtype::bfloat16 *previous,
                        const dtype::bfloat16 *current_qdq, float *dist, float *dist_qdq,
                        int64_t *reset_count, int64_t numel, float ratio_threshold, float eps,
                        bool collect_count, hipStream_t stream) {
    if (numel == 0)
        return;
    const dim3 block(kThreads);
    const dim3 grid(DIVUP<int64_t>(numel, kThreads));
    weight_deosc_close_kernel<<<grid, block, 0, stream>>>(
        master, previous, current_qdq, dist, dist_qdq,
        reinterpret_cast<unsigned long long *>(reset_count), numel, ratio_threshold, eps,
        collect_count);
}

} // namespace primus_turbo
