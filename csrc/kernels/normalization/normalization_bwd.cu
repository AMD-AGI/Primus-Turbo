// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
//
// See LICENSE for license information.

#include "primus_turbo/device/reduce.cuh"
#include "primus_turbo/device/utils.cuh"
#include "primus_turbo/normalization.h"

namespace primus_turbo {

using namespace primus_turbo::dtype;

// https://github.com/gashon/rms-norm-triton-kernel/blob/main/triton_util.py

// TODO: one scan
// TODO: gamma_grad reduce sum
template <typename T, int UNROLL>
__global__ void rmsnorm_bwd_two_scan_stage_0_kernel(const T *input, const T *gamma,
                                                    const T *output_grad, T *input_grad,
                                                    T *gamma_grad, const int64_t inner_len,
                                                    const float epsilon) {
    const int BLOCKSIZE = blockDim.x;
    const int bid       = blockIdx.x;
    const int warp_id   = threadIdx.x / THREADS_PER_WARP;
    const int lane_id   = threadIdx.x % THREADS_PER_WARP;

    const T *input_ptr       = input + bid * inner_len;
    const T *output_grad_ptr = output_grad + bid * inner_len;
    const T *gamma_ptr       = gamma;
    T       *input_grad_ptr  = input_grad + bid * inner_len;
    T       *gamma_grad_ptr  = gamma_grad + bid * inner_len;

    T ld_input_regs[UNROLL];
    T ld_outgrad_regs[UNROLL];
    T ld_gamma_regs[UNROLL];

    float     local_squares_sum = 0.0f;
    float     local_dot_sum     = 0.0f;
    const int start_offset      = warp_id * THREADS_PER_WARP * UNROLL + lane_id * UNROLL;
    for (int64_t offset = start_offset; offset < inner_len; offset += (BLOCKSIZE * UNROLL)) {
        load_data<T, UNROLL>(input_ptr + offset, ld_input_regs);
        load_data<T, UNROLL>(output_grad_ptr + offset, ld_outgrad_regs);
        load_data<T, UNROLL>(gamma_ptr + offset, ld_gamma_regs);

#pragma unroll
        for (int i = 0; i < UNROLL; ++i) {
            const float x  = static_cast<float>(ld_input_regs[i]);
            const float dy = static_cast<float>(ld_outgrad_regs[i]);
            const float g  = static_cast<float>(ld_gamma_regs[i]);
            local_squares_sum += (x * x);
            local_dot_sum += x * dy * g;
        }
    }

    const float mean_square =
        BlockReduce<SumOp, float>(local_squares_sum) / static_cast<float>(inner_len);
    const float inv_std  = rsqrtf(mean_square + epsilon);
    const float inv_std3 = inv_std * inv_std * inv_std;
    __syncthreads();
    const float dot_sum = BlockReduce<SumOp, float>(local_dot_sum) / static_cast<float>(inner_len);
    const float coeff   = dot_sum * inv_std3;

    T dx_regs[UNROLL];
    T dg_regs[UNROLL];
    for (int64_t offset = start_offset; offset < inner_len; offset += (BLOCKSIZE * UNROLL)) {
        load_data<T, UNROLL>(input_ptr + offset, ld_input_regs);
        load_data<T, UNROLL>(output_grad_ptr + offset, ld_outgrad_regs);
        load_data<T, UNROLL>(gamma_ptr + offset, ld_gamma_regs);

#pragma unroll
        for (int i = 0; i < UNROLL; ++i) {
            const float x  = static_cast<float>(ld_input_regs[i]);
            const float dy = static_cast<float>(ld_outgrad_regs[i]);
            const float g  = static_cast<float>(ld_gamma_regs[i]);
            const float dx = dy * g * inv_std - coeff * x;
            const float dg = x * dy * inv_std;
            dx_regs[i]     = static_cast<T>(dx);
            dg_regs[i]     = static_cast<T>(dg);
        }
        store_data<T, UNROLL>(input_grad_ptr + offset, dx_regs);
        store_data<T, UNROLL>(gamma_grad_ptr + offset, dg_regs);
    }
}

template <typename T>
void rmsnorm_bwd_impl(const T *input, const T *gamma, const T *output_grad, T *input_grad,
                      T *gamma_grad, const int64_t inner_len, const int64_t outer_len,
                      const float epsilon, hipStream_t stream) {
    const dim3    block_dim(THREADS_PER_WARP, 1, 1);
    const dim3    grid_dim(outer_len, 1, 1);
    constexpr int UNROLL = sizeof(uint4) / sizeof(T);
    if (inner_len % UNROLL == 0) {
        rmsnorm_bwd_two_scan_stage_0_kernel<T, UNROLL><<<grid_dim, block_dim, 0, stream>>>(
            input, gamma, output_grad, input_grad, gamma_grad, inner_len, epsilon);
    } else {
        rmsnorm_bwd_two_scan_stage_0_kernel<T, 1><<<grid_dim, block_dim, 0, stream>>>(
            input, gamma, output_grad, input_grad, gamma_grad, inner_len, epsilon);
    }
}

template void rmsnorm_bwd_impl<float>(const float *input, const float *gamma,
                                      const float *output_grad, float *input_grad,
                                      float *gamma_grad, const int64_t inner_len,
                                      const int64_t outer_len, const float epsilon,
                                      hipStream_t stream);

template void rmsnorm_bwd_impl<float16>(const float16 *input, const float16 *gamma,
                                        const float16 *output_grad, float16 *input_grad,
                                        float16 *gamma_grad, const int64_t inner_len,
                                        const int64_t outer_len, const float epsilon,
                                        hipStream_t stream);

template void rmsnorm_bwd_impl<bfloat16>(const bfloat16 *input, const bfloat16 *gamma,
                                         const bfloat16 *output_grad, bfloat16 *input_grad,
                                         bfloat16 *gamma_grad, const int64_t inner_len,
                                         const int64_t outer_len, const float epsilon,
                                         hipStream_t stream);

} // namespace primus_turbo
