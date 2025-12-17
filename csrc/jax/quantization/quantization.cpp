// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
//
// See LICENSE for license information.

#include "primus_turbo/quantization.h"
#include "jax/extensions.h"
#include "jax/ffi.h"
#include "jax/utils.h"
#include "primus_turbo/reduce.h"

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

// TODO: Check correctness
float get_float8_max(ffi::DataType dtype) {
    if (dtype == ffi::F8E4M3FN) {
        return 448.0f;
    } else if (dtype == ffi::F8E4M3FNUZ) {
        return 240.0f;
    } else if (dtype == ffi::F8E5M2) {
        return 57344.0f;
    } else if (dtype == ffi::F8E5M2FNUZ) {
        return 57344.0f;
    } else {
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

// Helper to compute BMN dimensions for rowwise quantization
inline void compute_quantize_fp8_rowwise_bmn(const std::vector<int64_t> &input_shape, int64_t axis,
                                             int64_t &B, int64_t &M, int64_t &N) {
    B = 1;
    for (int64_t i = 0; i < axis; ++i) {
        B *= input_shape[i];
    }
    M = input_shape[axis];
    N = 1;
    for (size_t i = axis + 1; i < input_shape.size(); ++i) {
        N *= input_shape[i];
    }
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
                                 ffi::Result<ffi::Buffer<ffi::DataType::F32>> scale_inv_out) {

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

            // The kernel writes to scale_ptr in auto-scale mode (PreComputeScale=false)
            // Allocate a temporary buffer for it
            float     *temp_scale = nullptr;
            hipError_t alloc_err  = hipMalloc(&temp_scale, outer_len_val * sizeof(float));
            if (alloc_err != hipSuccess) {
                return ffi::Error(ffi::ErrorCode::kInternal,
                                  "Failed to allocate temp scale buffer");
            }

            FFI_TYPE_SWITCH_FP16_BF16_FP32(input_dtype, FType, {
                FFI_TYPE_SWITCH_FP8(output_dtype, QType, {
                    quantize_rowwise_row_major_impl<FType, QType, float32, false>(
                        reinterpret_cast<const FType *>(input_buf), temp_scale, scale_inv_ptr,
                        reinterpret_cast<QType *>(output_buf), outer_len_val, inner_len, stream);
                });
            });

            hipFree(temp_scale);
        } else {
            // Col-major: requires reduce-col to compute amax, then compute scale
            int64_t B, M, N;
            compute_quantize_fp8_rowwise_bmn(shape_vec, valid_axis, B, M, N);

            // Allocate amax buffer (same shape as scale_inv)
            float     *amax_ptr       = nullptr;
            hipError_t amax_alloc_err = hipMalloc(&amax_ptr, B * N * sizeof(float));
            if (amax_alloc_err != hipSuccess) {
                return ffi::Error(ffi::ErrorCode::kInternal, "Failed to allocate amax buffer");
            }

            // Allocate workspace for reduce
            const int64_t ws_size       = get_reduce_col_workspace_sizes<float>(B, M, N);
            void         *workspace_ptr = nullptr;
            hipError_t    ws_alloc_err  = hipMalloc(&workspace_ptr, ws_size);
            if (ws_alloc_err != hipSuccess) {
                hipFree(amax_ptr);
                return ffi::Error(ffi::ErrorCode::kInternal, "Failed to allocate workspace");
            }

            // Step 1: Reduce-Col to compute amax
            FFI_TYPE_SWITCH_FP16_BF16_FP32(input_dtype, InT, {
                reduce_col<InT, float, float>(PrimusTurboReduceOp::REDUCE_ABS_MAX,
                                              reinterpret_cast<const InT *>(input_buf), amax_ptr, B,
                                              M, N, ws_size, workspace_ptr, stream);
            });

            // Allocate temp_scale buffer
            float     *temp_scale      = nullptr;
            hipError_t scale_alloc_err = hipMalloc(&temp_scale, B * N * sizeof(float));
            if (scale_alloc_err != hipSuccess) {
                hipFree(amax_ptr);
                hipFree(workspace_ptr);
                return ffi::Error(ffi::ErrorCode::kInternal,
                                  "Failed to allocate temp scale buffer");
            }

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

            // Cleanup
            hipFree(amax_ptr);
            hipFree(workspace_ptr);
            hipFree(temp_scale);
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
);

} // namespace primus_turbo::jax
