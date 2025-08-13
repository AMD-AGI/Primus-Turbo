// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
//
// See LICENSE for license information.

#include "primus_turbo/normalization.h"

namespace primus_turbo {

using namespace primus_turbo::dtype;

// TODO: one scan

template <typename T, int UNROLL>
__global__ void rmsnorm_fwd_two_scan_kernel(const T *input, const T *gamma, T *output,
                                            const int64_t inner_len, const float epsilon) {
    const int BLOCKSIZE = blockDim.x;
    const int bid       = blockIdx.x;
    const int warp_id   = threadIdx.x / THREADS_PER_WARP;
    const int lane_id   = threadIdx.x % THREADS_PER_WARP;

    const T *input_ptr  = input + bid * inner_len;
    const T *gamma_ptr  = gamma;
    T       *output_ptr = output + bid * inner_len;

    const int start_offset = warp_id * THREADS_PER_WARP * UNROLL + lane_id * UNROLL;
    T         ld_input_regs[UNROLL];
    float     local_squares_sum = 0.0f;
    for (int64_t offset = start_offset; offset < inner_len; offset += (BLOCKSIZE * UNROLL)) {
        load_data<T, UNROLL>(input_ptr + offset, ld_input_regs);
#pragma unroll
        for (int i = 0; i < UNROLL; ++i) {
            const float val = static_cast<float>(ld_input_regs[i]);
            local_squares_sum += (val * val);
        }
    }

    const float mean_square =
        BlockReduce<SumOp, float>(local_squares_sum) / static_cast<float>(inner_len);
    const float norm_factor = rsqrtf(mean_square + epsilon);

    T ld_gamma_regs[UNROLL];
    T st_regs[UNROLL];
    for (int64_t offset = start_offset; offset < inner_len; offset += (BLOCKSIZE * UNROLL)) {
        load_data<T, UNROLL>(input_ptr + offset, ld_input_regs);
        load_data<T, UNROLL>(gamma_ptr + offset, ld_gamma_regs);

#pragma unroll
        for (int i = 0; i < UNROLL; ++i) {
            float val = static_cast<float>(ld_input_regs[i]) * norm_factor *
                        static_cast<float>(ld_gamma_regs[i]);
            st_regs[i] = static_cast<T>(val);
        }
        store_data<T, UNROLL>(output_ptr + offset, st_regs);
    }
}

template <typename T>
void rmsnorm_fwd_impl(const T *input, const T *gamma, T *output, const int64_t inner_len,
                      const int64_t outer_len, const float epsilon, hipStream_t stream) {
    const dim3    block_dim(MAX_THREADS_PER_BLOCK, 1, 1);
    const dim3    grid_dim(outer_len, 1, 1);
    constexpr int UNROLL = sizeof(uint4) / sizeof(T);
    if (inner_len % UNROLL == 0) {
        rmsnorm_fwd_two_scan_kernel<T, UNROLL>
            <<<grid_dim, block_dim, 0, stream>>>(input, gamma, output, inner_len, epsilon);
    } else {
        rmsnorm_fwd_two_scan_kernel<T, 1>
            <<<grid_dim, block_dim, 0, stream>>>(input, gamma, output, inner_len, epsilon);
    }
}

template void rmsnorm_fwd_impl<float>(const float *input, const float *gamma, float *output,
                                      const int64_t inner_len, const int64_t outer_len,
                                      const float epsilon, hipStream_t stream);

template void rmsnorm_fwd_impl<float16>(const float16 *input, const float16 *gamma, float16 *output,
                                        const int64_t inner_len, const int64_t outer_len,
                                        const float epsilon, hipStream_t stream);

template void rmsnorm_fwd_impl<bfloat16>(const bfloat16 *input, const bfloat16 *gamma,
                                         bfloat16 *output, const int64_t inner_len,
                                         const int64_t outer_len, const float epsilon,
                                         hipStream_t stream);

} // namespace primus_turbo
