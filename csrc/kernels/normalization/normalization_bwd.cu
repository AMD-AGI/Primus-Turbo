#include "primus_turbo/normalization.h"

namespace primus_turbo {

using namespace primus_turbo::dtype;

// https://github.com/gashon/rms-norm-triton-kernel/blob/main/triton_util.py

// !!! This is a demo impl. It`s performance is bad.
// TODO: Opt
template <typename T>
__global__ void rmsnorm_bwd_kernel(const T *input, const T *gamma, const T *output_grad,
                                   T *input_grad, T *gamma_grad, const int64_t inner_len,
                                   const float epsilon) {
    const int32_t bid     = blockIdx.x;
    const int32_t warp_id = threadIdx.x / THREADS_PER_WARP; // in this case, the warp_id always == 0
    const int32_t lane_id = threadIdx.x % THREADS_PER_WARP;

    const T *input_ptr       = input + bid * inner_len;
    const T *gamma_ptr       = gamma;
    const T *output_grad_ptr = output_grad + bid * inner_len;
    T       *input_grad_ptr  = input_grad + bid * inner_len;
    T       *gamma_grad_ptr  = gamma_grad + bid * inner_len;

    //
    float local_squares_sum = 0.0f;
    float local_dot_sum     = 0.f;
    for (int64_t i = lane_id; i < inner_len; i += THREADS_PER_WARP) {
        float x  = static_cast<float>(input_ptr[i]);
        float dy = static_cast<float>(output_grad_ptr[i]);
        float g  = static_cast<float>(gamma_ptr[i]);
        local_squares_sum += (x * x);
        local_dot_sum += x * dy * g;
    }

    float mean_square = WarpReduce<SumOp, float>(local_squares_sum) / static_cast<float>(inner_len);
    const float inv_std  = rsqrtf(mean_square + epsilon);
    const float inv_std3 = inv_std * inv_std * inv_std;

    const float dot_sum = WarpReduce<SumOp, float>(local_dot_sum) / static_cast<float>(inner_len);
    const float coeff   = dot_sum * inv_std3;

    //
    for (int64_t i = lane_id; i < inner_len; i += THREADS_PER_WARP) {
        float x  = static_cast<float>(input_ptr[i]);
        float dy = static_cast<float>(output_grad_ptr[i]);
        float g  = static_cast<float>(gamma_ptr[i]);

        gamma_grad_ptr[i] = static_cast<T>(x * dy * inv_std);

        float dx          = dy * g * inv_std - coeff * x;
        input_grad_ptr[i] = static_cast<T>(dx);
    }
}

template <typename T>
void rmsnorm_bwd_impl(const T *input, const T *gamma, const T *output_grad, T *input_grad,
                      T *gamma_grad, const int64_t inner_len, const int64_t outer_len,
                      const float epsilon, hipStream_t stream) {
    const dim3 block_dim(THREADS_PER_WARP, 1, 1);
    const dim3 grid_dim(outer_len, 1, 1);
    rmsnorm_bwd_kernel<T><<<grid_dim, block_dim, 0, stream>>>(input, gamma, output_grad, input_grad,
                                                              gamma_grad, inner_len, epsilon);
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
