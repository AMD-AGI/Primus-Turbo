// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
//
// See LICENSE for license information.

#include "primus_turbo/quantization.h"
#include "jax/extensions.h"
#include "jax/ffi.h"
#include "jax/utils.h"
#include "primus_turbo/reduce.h"

#include <functional>
#include <numeric>

namespace primus_turbo::jax {

using namespace primus_turbo::dtype;

// Simple GPU kernel for computing reciprocal (scale_inv = 1.0 / scale)
__global__ void reciprocal_kernel(const float *scale, float *scale_inv, int64_t n) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        scale_inv[idx] = 1.0f / scale[idx];
    }
}

// Helper function to compute scale_inv on GPU
inline void compute_scale_inv_gpu(const float *scale, float *scale_inv, int64_t n,
                                  hipStream_t stream) {
    constexpr int threads = 256;
    int           blocks  = (n + threads - 1) / threads;
    hipLaunchKernelGGL(reciprocal_kernel, dim3(blocks), dim3(threads), 0, stream, scale, scale_inv,
                       n);
}

float get_float8_max(ffi::DataType dtype) {
    switch (dtype) {
    case ffi::F8E4M3FN:
        return 448.0f;
    case ffi::F8E4M3FNUZ:
        return 240.0f;
    case ffi::F8E5M2:
    case ffi::F8E5M2FNUZ:
        return 57344.0f;
    default:
        return 1.0f;
    }
}

// Align size to 128 bytes for optimal GPU memory access
constexpr int64_t kWorkspaceAlignment = 128;

inline int64_t align_size(int64_t size) {
    return (size + kWorkspaceAlignment - 1) / kWorkspaceAlignment * kWorkspaceAlignment;
}

// Workspace size for tensorwise quantization (when scale is not provided)
// Layout: [amax (aligned) | reduce_workspace (aligned) | scale (aligned)]
int64_t GetQuantizeFP8TensorwiseWorkspaceSize(int64_t n) {
    const int64_t amax_size      = align_size(sizeof(float));
    const int64_t reduce_ws_size = align_size(get_reduce_row_workspace_sizes<float>(1, n));
    const int64_t scale_size     = align_size(sizeof(float));
    return amax_size + reduce_ws_size + scale_size;
}

// Workspace size for rowwise quantization (when scale is not provided)
int64_t GetQuantizeFP8RowwiseWorkspaceSize(const std::vector<int64_t> &shape, int64_t axis) {
    int64_t ndim         = static_cast<int64_t>(shape.size());
    int64_t valid_axis   = (axis >= 0) ? axis : ndim + axis;
    bool    is_row_major = (valid_axis == ndim - 1);

    if (is_row_major) {
        // Row-major: workspace = [temp_scale (aligned)]
        int64_t n = 1;
        for (auto dim : shape)
            n *= dim;
        int64_t outer_len = n / shape[valid_axis];
        return align_size(outer_len * sizeof(float));
    } else {
        // Col-major: workspace = [amax (aligned) | reduce_ws (aligned) | temp_scale (aligned)]
        int64_t B = 1, M = shape[valid_axis], N = 1;
        for (int64_t i = 0; i < valid_axis; ++i)
            B *= shape[i];
        for (int64_t i = valid_axis + 1; i < ndim; ++i)
            N *= shape[i];

        const int64_t amax_size      = align_size(B * N * sizeof(float));
        const int64_t reduce_ws_size = align_size(get_reduce_col_workspace_sizes<float>(B, M, N));
        const int64_t scale_size     = align_size(B * N * sizeof(float));
        return amax_size + reduce_ws_size + scale_size;
    }
}

// Helper to compute BMN dimensions for rowwise quantization
inline void compute_quantize_fp8_rowwise_bmn(const std::vector<int64_t> &shape, int64_t axis,
                                             int64_t &B, int64_t &M, int64_t &N) {
    const int64_t ndim = static_cast<int64_t>(shape.size());
    auto          prod = [&](int64_t start, int64_t end) {
        return std::accumulate(shape.begin() + start, shape.begin() + end, int64_t{1},
                                        std::multiplies<int64_t>());
    };
    B = prod(0, axis);
    M = shape[axis];
    N = prod(axis + 1, ndim);
}

// Tensorwise Quantize FP8 FFI
ffi::Error QuantizeFP8TensorwiseFFI(hipStream_t stream, ffi::AnyBuffer input,
                                    ffi::Buffer<ffi::DataType::F32>              scale_opt,
                                    ffi::Result<ffi::AnyBuffer>                  output,
                                    ffi::Result<ffi::Buffer<ffi::DataType::F32>> scale_inv_out,
                                    ffi::Result<ffi::AnyBuffer>                  workspace) {

    const int64_t n             = input.element_count();
    auto          input_dtype   = input.element_type();
    auto          output_dtype  = output->element_type();
    auto          output_buf    = output->untyped_data();
    auto          input_buf     = input.untyped_data();
    float        *scale_inv_ptr = scale_inv_out->typed_data();

    bool has_scale = scale_opt.element_count() > 0;

    if (has_scale) {
        const float *scale_ptr = scale_opt.typed_data();
        compute_scale_inv_gpu(scale_ptr, scale_inv_ptr, 1, stream);

        FFI_TYPE_SWITCH_FP16_BF16_FP32(input_dtype, FType, {
            FFI_TYPE_SWITCH_FP8(output_dtype, QType, {
                quantize_tensorwise_impl<FType, QType>(
                    reinterpret_cast<const FType *>(input_buf), scale_ptr,
                    reinterpret_cast<QType *>(output_buf), n, stream);
            });
        });
    } else {
        const int64_t amax_size      = align_size(sizeof(float));
        const int64_t reduce_ws_size = align_size(get_reduce_row_workspace_sizes<float>(1, n));

        uint8_t *ws_ptr        = workspace->typed_data<uint8_t>();
        float   *amax_ptr      = reinterpret_cast<float *>(ws_ptr);
        void    *reduce_ws_ptr = ws_ptr + amax_size;
        float   *scale_ptr     = reinterpret_cast<float *>(ws_ptr + amax_size + reduce_ws_size);

        float fp8_max = get_float8_max(output_dtype);

        FFI_TYPE_SWITCH_FP16_BF16_FP32(input_dtype, InT, {
            reduce_row<InT, float, float>(PrimusTurboReduceOp::REDUCE_ABS_MAX,
                                          const_cast<InT *>(input.typed_data<InT>()), amax_ptr, 1,
                                          n, reduce_ws_size, reduce_ws_ptr, stream);
        });

        compute_scale_from_amax<float>(amax_ptr, fp8_max, scale_ptr, scale_inv_ptr, 1, stream);

        FFI_TYPE_SWITCH_FP16_BF16_FP32(input_dtype, FType, {
            FFI_TYPE_SWITCH_FP8(output_dtype, QType, {
                quantize_tensorwise_impl<FType, QType>(
                    reinterpret_cast<const FType *>(input_buf), scale_ptr,
                    reinterpret_cast<QType *>(output_buf), n, stream);
            });
        });
    }

    return ffi::Error::Success();
}

// Tensorwise Dequantize FP8 FFI
ffi::Error DequantizeFP8TensorwiseFFI(hipStream_t stream, ffi::AnyBuffer input,
                                      ffi::Buffer<ffi::DataType::F32> scale_inv,
                                      ffi::Result<ffi::AnyBuffer>     output) {
    FFI_TYPE_SWITCH_FP16_BF16_FP32(output->element_type(), FType, {
        FFI_TYPE_SWITCH_FP8(input.element_type(), QType, {
            dequantize_tensorwise_impl<FType, QType>(
                reinterpret_cast<const QType *>(input.untyped_data()), scale_inv.typed_data(),
                reinterpret_cast<FType *>(output->untyped_data()), input.element_count(), stream);
        });
    });
    return ffi::Error::Success();
}

// Rowwise Quantize FP8 FFI
ffi::Error QuantizeFP8RowwiseFFI(hipStream_t stream, ffi::AnyBuffer input, int64_t axis,
                                 ffi::Buffer<ffi::DataType::F32>              scale_opt,
                                 ffi::Result<ffi::AnyBuffer>                  output,
                                 ffi::Result<ffi::Buffer<ffi::DataType::F32>> scale_inv_out,
                                 ffi::Result<ffi::AnyBuffer>                  workspace) {

    auto                 input_shape = input.dimensions();
    std::vector<int64_t> shape_vec(input_shape.begin(), input_shape.end());

    auto input_dtype  = input.element_type();
    auto output_dtype = output->element_type();

    // Compute valid axis
    int64_t valid_axis = (axis >= 0) ? axis : static_cast<int64_t>(input_shape.size()) + axis;
    if (valid_axis < 0 || valid_axis >= static_cast<int64_t>(input_shape.size())) {
        return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                          "Invalid axis for rowwise quantization");
    }

    bool is_row_major = (valid_axis == static_cast<int64_t>(input_shape.size()) - 1);

    // Compute scale shape
    std::vector<int64_t> scale_shape = shape_vec;
    scale_shape[valid_axis]          = 1;

    bool has_scale = scale_opt.element_count() > 0;

    auto output_buf    = output->untyped_data();
    auto scale_inv_ptr = scale_inv_out->typed_data();
    auto input_buf     = input.untyped_data();

    if (has_scale) {
        float *scale_ptr = const_cast<float *>(scale_opt.typed_data());

        // Compute scale_inv = 1.0 / scale on GPU
        compute_scale_inv_gpu(scale_ptr, scale_inv_ptr, scale_opt.element_count(), stream);

        // Quantize
        if (is_row_major) {
            const int64_t inner_len     = input_shape[valid_axis];
            const int64_t outer_len_val = input.element_count() / inner_len;

            FFI_TYPE_SWITCH_FP16_BF16_FP32(input_dtype, FType, {
                FFI_TYPE_SWITCH_FP8(output_dtype, QType, {
                    quantize_rowwise_row_major_impl<FType, QType, float32, true>(
                        reinterpret_cast<const FType *>(input_buf), scale_ptr, scale_inv_ptr,
                        reinterpret_cast<QType *>(output_buf), outer_len_val, inner_len, stream);
                });
            });
        } else {
            int64_t B, M, N;
            compute_quantize_fp8_rowwise_bmn(shape_vec, valid_axis, B, M, N);

            FFI_TYPE_SWITCH_FP16_BF16_FP32(input_dtype, FType, {
                FFI_TYPE_SWITCH_FP8(output_dtype, QType, {
                    quantize_rowwise_col_major_impl<FType, QType, float32>(
                        reinterpret_cast<const FType *>(input_buf), scale_ptr, scale_inv_ptr,
                        reinterpret_cast<QType *>(output_buf), B, M, N, stream);
                });
            });
        }
    } else {
        // Auto-scale: compute scale from amax
        if (is_row_major) {
            const int64_t inner_len     = input_shape[valid_axis];
            const int64_t outer_len_val = input.element_count() / inner_len;

            // Use workspace for temp_scale
            float *temp_scale = reinterpret_cast<float *>(workspace->untyped_data());

            FFI_TYPE_SWITCH_FP16_BF16_FP32(input_dtype, FType, {
                FFI_TYPE_SWITCH_FP8(output_dtype, QType, {
                    quantize_rowwise_row_major_impl<FType, QType, float32, false>(
                        reinterpret_cast<const FType *>(input_buf), temp_scale, scale_inv_ptr,
                        reinterpret_cast<QType *>(output_buf), outer_len_val, inner_len, stream);
                });
            });
        } else {
            // Col-major: requires reduce-col to compute amax, then compute scale
            int64_t B, M, N;
            compute_quantize_fp8_rowwise_bmn(shape_vec, valid_axis, B, M, N);

            // Workspace layout: [amax (aligned) | reduce_workspace (aligned) | temp_scale
            // (aligned)]
            const int64_t amax_size = align_size(B * N * sizeof(float));
            const int64_t reduce_ws_size =
                align_size(get_reduce_col_workspace_sizes<float>(B, M, N));

            uint8_t *ws_ptr        = workspace->typed_data<uint8_t>();
            float   *amax_ptr      = reinterpret_cast<float *>(ws_ptr);
            void    *reduce_ws_ptr = ws_ptr + amax_size;
            float   *temp_scale    = reinterpret_cast<float *>(ws_ptr + amax_size + reduce_ws_size);

            // Step 1: Reduce-Col to compute amax
            FFI_TYPE_SWITCH_FP16_BF16_FP32(input_dtype, InT, {
                reduce_col<InT, float, float>(PrimusTurboReduceOp::REDUCE_ABS_MAX,
                                              reinterpret_cast<const InT *>(input_buf), amax_ptr, B,
                                              M, N, reduce_ws_size, reduce_ws_ptr, stream);
            });

            // Step 2: Compute scale from amax
            float fp8_max = get_float8_max(output_dtype);
            compute_scale_from_amax<float>(amax_ptr, fp8_max, temp_scale, scale_inv_ptr, B * N,
                                           stream);

            // Step 3: Quantize using computed scale
            FFI_TYPE_SWITCH_FP16_BF16_FP32(input_dtype, FType, {
                FFI_TYPE_SWITCH_FP8(output_dtype, QType, {
                    quantize_rowwise_col_major_impl<FType, QType, float>(
                        reinterpret_cast<const FType *>(input_buf), temp_scale, scale_inv_ptr,
                        reinterpret_cast<QType *>(output_buf), B, M, N, stream);
                });
            });
        }
    }

    return ffi::Error::Success();
}

// Register FFI handlers
XLA_FFI_DEFINE_HANDLER_SYMBOL(QuantizeFP8TensorwiseHandler, QuantizeFP8TensorwiseFFI,
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::PlatformStream<hipStream_t>>() // stream
                                  .Arg<ffi::AnyBuffer>()                   // input (F32/F16/BF16)
                                  .Arg<ffi::Buffer<ffi::DataType::F32>>()  // scale_opt
                                  .Ret<ffi::AnyBuffer>()                   // output (fp8)
                                  .Ret<ffi::Buffer<ffi::DataType::F32>>()  // scale_inv
                                  .Ret<ffi::AnyBuffer>()                   // workspace
);

XLA_FFI_DEFINE_HANDLER_SYMBOL(DequantizeFP8TensorwiseHandler, DequantizeFP8TensorwiseFFI,
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::PlatformStream<hipStream_t>>() // stream
                                  .Arg<ffi::AnyBuffer>()                   // input (fp8)
                                  .Arg<ffi::Buffer<ffi::DataType::F32>>()  // scale_inv
                                  .Ret<ffi::AnyBuffer>()                   // output
);

XLA_FFI_DEFINE_HANDLER_SYMBOL(QuantizeFP8RowwiseHandler, QuantizeFP8RowwiseFFI,
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::PlatformStream<hipStream_t>>() // stream
                                  .Arg<ffi::AnyBuffer>()                   // input (F32/F16/BF16)
                                  .Attr<int64_t>("axis")                   // axis
                                  .Arg<ffi::Buffer<ffi::DataType::F32>>()  // scale_opt
                                  .Ret<ffi::AnyBuffer>()                   // output (fp8)
                                  .Ret<ffi::Buffer<ffi::DataType::F32>>()  // scale_inv
                                  .Ret<ffi::AnyBuffer>()                   // workspace
);

} // namespace primus_turbo::jax
