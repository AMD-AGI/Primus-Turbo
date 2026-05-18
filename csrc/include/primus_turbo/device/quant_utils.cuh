// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
//
// See LICENSE for license information.
//
// Quant helpers shared by tensorwise and rowwise kernels.
//
//   * QuantOpBase / QuantOp  : the (x * scale) -> clamp(min,max) op used by all
//                              FP8 quant paths.
//   * compute_scale_from_amax_device_kernel : per-element amax -> scale on device.
//   * compute_scale_from_amax_kernel        : __global__ over a [n] amax buffer.
//   * compute_scale_from_amax               : host wrapper that launches the
//                                             kernel above.
//
// All entries are header-only template definitions; the corresponding host
// wrappers are launched directly from including .cu files, and the explicit
// instantiation of `compute_scale_from_amax<float>` lives in
// quantization_tensorwise.cu so the symbol is exported by libprimus_turbo.

#pragma once

#include <hip/hip_runtime.h>

#include "primus_turbo/common.h"
#include "primus_turbo/quantization.h"

namespace primus_turbo {

// ---------------------------------------------------------------------------
// QuantOp: y = clamp(x * scale, clip_min, clip_max)
// ---------------------------------------------------------------------------
template <typename ComputeType = float> struct QuantOpBase {
    static PRIMUS_TURBO_HOST_DEVICE ComputeType quant(const ComputeType x, const ComputeType scale,
                                                      const ComputeType clip_min,
                                                      const ComputeType clip_max) {
        const ComputeType v = x * scale;
        return fmax(fmin(v, clip_max), clip_min);
    }
};

template <typename ComputeType = float> struct QuantOp : QuantOpBase<ComputeType> {
    ComputeType clip_min;
    ComputeType clip_max;

    PRIMUS_TURBO_HOST_DEVICE ComputeType operator()(const ComputeType x,
                                                    const ComputeType scale) const {
        return QuantOpBase<ComputeType>::quant(x, scale, clip_min, clip_max);
    }
};

// ---------------------------------------------------------------------------
// amax -> scale conversion
// ---------------------------------------------------------------------------
template <typename T = float>
PRIMUS_TURBO_DEVICE T compute_scale_from_amax_device_kernel(const T amax, const T q_max,
                                                            const float eps) {
    float amax_t = fmax(static_cast<float>(amax), eps);
    return static_cast<T>(static_cast<float>(q_max) / amax_t);
}

template <typename T>
__global__ void compute_scale_from_amax_kernel(const T *amax_ptr, const T q_max, T *scale_ptr,
                                               T *scale_inv_ptr, const int64_t n, const float eps) {
    int64_t tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (tid < n) {
        float amax         = static_cast<float>(amax_ptr[tid]);
        amax               = fmax(amax, eps);
        float scale        = static_cast<float>(q_max) / amax;
        float scale_inv    = 1.0f / scale;
        scale_ptr[tid]     = static_cast<T>(scale);
        scale_inv_ptr[tid] = static_cast<T>(scale_inv);
    }
}

// Host wrapper. Defined here so that any TU including this header can launch
// the kernel; the explicit instantiation of `compute_scale_from_amax<float>`
// lives in quantization_tensorwise.cu so external linkers can find a single
// symbol for it. (The default value for `eps` is declared in
// primus_turbo/quantization.h.)
template <typename T>
void compute_scale_from_amax(const T *amax, const T q_max, T *scale, T *scale_inv, const int64_t n,
                             hipStream_t stream, const float eps) {
    constexpr int64_t BLOCK_SIZE = 512;
    const int64_t     GRID_SIZE  = DIVUP<int64_t>(n, BLOCK_SIZE);
    compute_scale_from_amax_kernel<T>
        <<<GRID_SIZE, BLOCK_SIZE, 0, stream>>>(amax, q_max, scale, scale_inv, n, eps);
}

} // namespace primus_turbo
