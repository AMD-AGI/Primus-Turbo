// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
//
// See LICENSE for license information.

#include "primus_turbo/quantization.h"
#include "../extensions.h"
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

// Calculate workspace size for tensorwise quantization
inline int64_t get_quantize_fp8_tensorwise_workspace_size(int64_t n) {
    // For auto-scale path:
    // - amax_ptr: sizeof(float)
    // - reduce workspace: get_reduce_row_workspace_sizes<float>(1, n)
    // - scale_ptr: sizeof(float)
    // - scale_inv_ptr: sizeof(float)
    int64_t reduce_ws_size = get_reduce_row_workspace_sizes<float>(1, n);
    // Align to 256 bytes for each buffer
    int64_t total_size = 0;
    total_size += 256;                                  // amax_ptr (aligned)
    total_size += ((reduce_ws_size + 255) / 256) * 256; // reduce workspace (aligned)
    total_size += 256;                                  // scale_ptr (aligned)
    total_size += 256;                                  // scale_inv_ptr (aligned)
    return total_size;
}

// Calculate workspace size for rowwise quantization
inline int64_t get_quantize_fp8_rowwise_workspace_size(int64_t B, int64_t M, int64_t N,
                                                       bool is_row_major) {
    if (is_row_major) {
        // Row-major: temp_scale buffer
        return ((B * N * sizeof(float) + 255) / 256) * 256;
    } else {
        // Col-major: amax + reduce workspace
        int64_t amax_size      = B * N * sizeof(float);
        int64_t reduce_ws_size = get_reduce_col_workspace_sizes<float>(B, M, N);
        return ((amax_size + 255) / 256) * 256 + ((reduce_ws_size + 255) / 256) * 256;
    }
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

// Tensorwise Quantize FP8 FFI
// Input parameters: input (F32/F16/BF16), out_dtype_str (string), scale_opt (F32 buffer, optional),
// workspace Output parameters (via ffi::Result): output, scale_inv Return value: ffi::Error
// (status)
ffi::Error QuantizeFP8TensorwiseFFI(ffi::AnyBuffer input, std::string_view out_dtype_str,
                                    ffi::Buffer<ffi::DataType::F32> scale_opt,
                                    ffi::AnyBuffer workspace, ffi::Result<ffi::AnyBuffer> output,
                                    ffi::Result<ffi::Buffer<ffi::DataType::F32>> scale_inv_out) {
    hipStream_t stream = nullptr;

    auto    input_shape = input.dimensions();
    int64_t n           = 1;
    for (auto dim : input_shape) {
        n *= dim;
    }

    auto input_dtype = input.element_type();

    // Check if scale is provided (scale_opt.numel() > 0)
    bool has_scale = (scale_opt.dimensions().size() > 0 && scale_opt.dimensions()[0] > 0);

    if (has_scale) {
        // Use provided scale (GPU memory)
        const float *scale_ptr = scale_opt.typed_data();

        // Compute scale_inv on GPU (1.0 / scale) and write to output
        float *scale_inv_ptr = scale_inv_out->typed_data();
        compute_scale_inv_gpu(scale_ptr, scale_inv_ptr, 1, stream);

        // Quantize - dispatch based on input dtype
        auto output_buf = output->untyped_data();
        auto input_buf  = input.untyped_data();

        if (out_dtype_str == "float8_e4m3fn" || out_dtype_str == "float8_e4m3fnuz") {
            if (input_dtype == ffi::F32) {
                quantize_tensorwise_impl<float32, float8_e4m3_t, float32>(
                    reinterpret_cast<const float32 *>(input_buf), scale_ptr,
                    reinterpret_cast<float8_e4m3_t *>(output_buf), n, stream);
            } else if (input_dtype == ffi::F16) {
                quantize_tensorwise_impl<float16, float8_e4m3_t, float32>(
                    reinterpret_cast<const float16 *>(input_buf), scale_ptr,
                    reinterpret_cast<float8_e4m3_t *>(output_buf), n, stream);
            } else if (input_dtype == ffi::BF16) {
                quantize_tensorwise_impl<bfloat16, float8_e4m3_t, float32>(
                    reinterpret_cast<const bfloat16 *>(input_buf), scale_ptr,
                    reinterpret_cast<float8_e4m3_t *>(output_buf), n, stream);
            } else {
                return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                                  "Unsupported input dtype (expected F32, F16, or BF16)");
            }
        } else if (out_dtype_str == "float8_e5m2" || out_dtype_str == "float8_e5m2fnuz") {
            if (input_dtype == ffi::F32) {
                quantize_tensorwise_impl<float32, float8_e5m2_t, float32>(
                    reinterpret_cast<const float32 *>(input_buf), scale_ptr,
                    reinterpret_cast<float8_e5m2_t *>(output_buf), n, stream);
            } else if (input_dtype == ffi::F16) {
                quantize_tensorwise_impl<float16, float8_e5m2_t, float32>(
                    reinterpret_cast<const float16 *>(input_buf), scale_ptr,
                    reinterpret_cast<float8_e5m2_t *>(output_buf), n, stream);
            } else if (input_dtype == ffi::BF16) {
                quantize_tensorwise_impl<bfloat16, float8_e5m2_t, float32>(
                    reinterpret_cast<const bfloat16 *>(input_buf), scale_ptr,
                    reinterpret_cast<float8_e5m2_t *>(output_buf), n, stream);
            } else {
                return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                                  "Unsupported input dtype (expected F32, F16, or BF16)");
            }
        } else {
            return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                              "Unsupported FP8 dtype for quantization");
        }
    } else {
        // Auto-compute scale from amax
        ffi::DataType fp8_dtype;
        if (out_dtype_str == "float8_e4m3fn") {
            fp8_dtype = ffi::DataType::F8E4M3FN;
        } else if (out_dtype_str == "float8_e4m3fnuz") {
            fp8_dtype = ffi::DataType::F8E4M3FNUZ;
        } else if (out_dtype_str == "float8_e5m2") {
            fp8_dtype = ffi::DataType::F8E5M2;
        } else {
            fp8_dtype = ffi::DataType::F8E5M2FNUZ;
        }
        float         fp8_max = get_float8_max(fp8_dtype);
        const int64_t ws_size = get_reduce_row_workspace_sizes<float>(1, n);

        void *workspace_ptr = workspace.untyped_data();

        // Layout workspace buffer:
        // [amax_ptr: 256 bytes] [reduce_ws: aligned] [scale_ptr: 256 bytes]
        char  *ws_char       = reinterpret_cast<char *>(workspace_ptr);
        float *amax_ptr      = reinterpret_cast<float *>(ws_char);
        void  *reduce_ws_ptr = ws_char + 256;
        float *scale_ptr = reinterpret_cast<float *>(ws_char + 256 + ((ws_size + 255) / 256) * 256);

        // Use output buffer directly for scale_inv (no extra copy needed)
        float *scale_inv_ptr = scale_inv_out->typed_data();

        // Dispatch based on input dtype
        if (input_dtype == ffi::DataType::F32) {
            reduce_row<float, float, float>(PrimusTurboReduceOp::REDUCE_ABS_MAX,
                                            const_cast<float *>(input.typed_data<float>()),
                                            amax_ptr, 1, n, ws_size, reduce_ws_ptr, stream);
        } else if (input_dtype == ffi::DataType::F16) {
            reduce_row<float16, float, float>(PrimusTurboReduceOp::REDUCE_ABS_MAX,
                                              const_cast<float16 *>(input.typed_data<float16>()),
                                              amax_ptr, 1, n, ws_size, reduce_ws_ptr, stream);
        } else if (input_dtype == ffi::DataType::BF16) {
            reduce_row<bfloat16, float, float>(PrimusTurboReduceOp::REDUCE_ABS_MAX,
                                               const_cast<bfloat16 *>(input.typed_data<bfloat16>()),
                                               amax_ptr, 1, n, ws_size, reduce_ws_ptr, stream);
        }

        // Compute scale and scale_inv, write directly to output
        compute_scale_from_amax<float>(amax_ptr, fp8_max, scale_ptr, scale_inv_ptr, 1, stream);

        // Now quantize with the computed scale
        auto output_buf = output->untyped_data();
        auto input_buf  = input.untyped_data();

        if (out_dtype_str == "float8_e4m3fn" || out_dtype_str == "float8_e4m3fnuz") {
            if (input_dtype == ffi::DataType::F32) {
                quantize_tensorwise_impl<float32, float8_e4m3_t, float32>(
                    reinterpret_cast<const float32 *>(input_buf), scale_ptr,
                    reinterpret_cast<float8_e4m3_t *>(output_buf), n, stream);
            } else if (input_dtype == ffi::DataType::F16) {
                quantize_tensorwise_impl<float16, float8_e4m3_t, float32>(
                    reinterpret_cast<const float16 *>(input_buf), scale_ptr,
                    reinterpret_cast<float8_e4m3_t *>(output_buf), n, stream);
            } else if (input_dtype == ffi::DataType::BF16) {
                quantize_tensorwise_impl<bfloat16, float8_e4m3_t, float32>(
                    reinterpret_cast<const bfloat16 *>(input_buf), scale_ptr,
                    reinterpret_cast<float8_e4m3_t *>(output_buf), n, stream);
            }
        } else if (out_dtype_str == "float8_e5m2" || out_dtype_str == "float8_e5m2fnuz") {
            if (input_dtype == ffi::DataType::F32) {
                quantize_tensorwise_impl<float32, float8_e5m2_t, float32>(
                    reinterpret_cast<const float32 *>(input_buf), scale_ptr,
                    reinterpret_cast<float8_e5m2_t *>(output_buf), n, stream);
            } else if (input_dtype == ffi::DataType::F16) {
                quantize_tensorwise_impl<float16, float8_e5m2_t, float32>(
                    reinterpret_cast<const float16 *>(input_buf), scale_ptr,
                    reinterpret_cast<float8_e5m2_t *>(output_buf), n, stream);
            } else if (input_dtype == ffi::DataType::BF16) {
                quantize_tensorwise_impl<bfloat16, float8_e5m2_t, float32>(
                    reinterpret_cast<const bfloat16 *>(input_buf), scale_ptr,
                    reinterpret_cast<float8_e5m2_t *>(output_buf), n, stream);
            }
        }
    }

    return ffi::Error::Success();
}

// Tensorwise Dequantize FP8 FFI
// Output parameters (via ffi::Result): output
// Return value: ffi::Error (status)
ffi::Error DequantizeFP8TensorwiseFFI(ffi::AnyBuffer                               input,
                                      ffi::Buffer<ffi::DataType::F32>              scale_inv,
                                      std::string_view                             out_dtype_str,
                                      ffi::Result<ffi::Buffer<ffi::DataType::F32>> output) {
    hipStream_t stream = nullptr;

    auto    input_shape = input.dimensions();
    int64_t n           = 1;
    for (auto dim : input_shape) {
        n *= dim;
    }

    auto input_buf       = input.untyped_data();
    auto input_elem_type = input.element_type();

    // Dispatch based on element type
    if (input_elem_type == ffi::F8E4M3FN || input_elem_type == ffi::F8E4M3FNUZ) {
        dequantize_tensorwise_impl<float32, float8_e4m3_t, float32>(
            reinterpret_cast<const float8_e4m3_t *>(input_buf), scale_inv.typed_data(),
            output->typed_data(), n, stream);
    } else if (input_elem_type == ffi::F8E5M2 || input_elem_type == ffi::F8E5M2FNUZ) {
        dequantize_tensorwise_impl<float32, float8_e5m2_t, float32>(
            reinterpret_cast<const float8_e5m2_t *>(input_buf), scale_inv.typed_data(),
            output->typed_data(), n, stream);
    } else {
        return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                          "Unsupported input dtype for dequantization");
    }

    return ffi::Error::Success();
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

// Rowwise Quantize FP8 FFI
// Output parameters (via ffi::Result): output, scale_inv
// Return value: ffi::Error (status)
ffi::Error QuantizeFP8RowwiseFFI(ffi::AnyBuffer input, std::string_view out_dtype_str, int64_t axis,
                                 ffi::Buffer<ffi::DataType::F32> scale_opt,
                                 ffi::AnyBuffer workspace, ffi::Result<ffi::AnyBuffer> output,
                                 ffi::Result<ffi::Buffer<ffi::DataType::F32>> scale_inv_out) {
    hipStream_t stream = nullptr;

    auto                 input_shape = input.dimensions();
    std::vector<int64_t> shape_vec(input_shape.begin(), input_shape.end());

    auto input_dtype = input.element_type();

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

    // Check if scale is provided
    bool has_scale = (scale_opt.dimensions().size() > 0 && scale_opt.dimensions()[0] > 0);

    auto output_buf    = output->untyped_data();
    auto scale_inv_ptr = scale_inv_out->typed_data();
    auto input_buf     = input.untyped_data();

    if (has_scale) {
        // Use provided scale (GPU memory)
        const float *scale_ptr = scale_opt.typed_data();

        // Compute scale_inv: scale_inv = 1.0 / scale (element-wise)
        // Use the actual size of scale_opt, not scale_shape
        int64_t scale_numel = 1;
        for (auto dim : scale_opt.dimensions()) {
            scale_numel *= dim;
        }

        // Use GPU kernel for element-wise reciprocal
        // Need to cast away const because kernel expects non-const pointer
        float *scale_ptr_nonconst = const_cast<float *>(scale_ptr);
        compute_scale_inv_gpu(scale_ptr_nonconst, scale_inv_ptr, scale_numel, stream);

        // Quantize
        if (is_row_major) {
            const int64_t inner_len = input_shape[valid_axis];
            int64_t       outer_len = 1;
            for (auto dim : input_shape) {
                outer_len *= dim;
            }
            const int64_t outer_len_val = outer_len / inner_len;

            // Need to cast away const because kernel expects non-const pointer
            // But kernel won't modify scale when has_scale=true
            float *scale_ptr_nonconst = const_cast<float *>(scale_ptr);

            if (out_dtype_str == "float8_e4m3fn" || out_dtype_str == "float8_e4m3fnuz") {
                if (input_dtype == ffi::F32) {
                    quantize_rowwise_row_major_impl<float32, float8_e4m3_t, float32, true>(
                        reinterpret_cast<const float32 *>(input_buf), scale_ptr_nonconst,
                        scale_inv_ptr, reinterpret_cast<float8_e4m3_t *>(output_buf), outer_len_val,
                        inner_len, stream);
                } else if (input_dtype == ffi::F16) {
                    quantize_rowwise_row_major_impl<float16, float8_e4m3_t, float32, true>(
                        reinterpret_cast<const float16 *>(input_buf), scale_ptr_nonconst,
                        scale_inv_ptr, reinterpret_cast<float8_e4m3_t *>(output_buf), outer_len_val,
                        inner_len, stream);
                } else if (input_dtype == ffi::BF16) {
                    quantize_rowwise_row_major_impl<bfloat16, float8_e4m3_t, float32, true>(
                        reinterpret_cast<const bfloat16 *>(input_buf), scale_ptr_nonconst,
                        scale_inv_ptr, reinterpret_cast<float8_e4m3_t *>(output_buf), outer_len_val,
                        inner_len, stream);
                } else {
                    return ffi::Error(ffi::ErrorCode::kInvalidArgument, "Unsupported input dtype");
                }
            } else if (out_dtype_str == "float8_e5m2" || out_dtype_str == "float8_e5m2fnuz") {
                if (input_dtype == ffi::F32) {
                    quantize_rowwise_row_major_impl<float32, float8_e5m2_t, float32, true>(
                        reinterpret_cast<const float32 *>(input_buf), scale_ptr_nonconst,
                        scale_inv_ptr, reinterpret_cast<float8_e5m2_t *>(output_buf), outer_len_val,
                        inner_len, stream);
                } else if (input_dtype == ffi::F16) {
                    quantize_rowwise_row_major_impl<float16, float8_e5m2_t, float32, true>(
                        reinterpret_cast<const float16 *>(input_buf), scale_ptr_nonconst,
                        scale_inv_ptr, reinterpret_cast<float8_e5m2_t *>(output_buf), outer_len_val,
                        inner_len, stream);
                } else if (input_dtype == ffi::BF16) {
                    quantize_rowwise_row_major_impl<bfloat16, float8_e5m2_t, float32, true>(
                        reinterpret_cast<const bfloat16 *>(input_buf), scale_ptr_nonconst,
                        scale_inv_ptr, reinterpret_cast<float8_e5m2_t *>(output_buf), outer_len_val,
                        inner_len, stream);
                } else {
                    return ffi::Error(ffi::ErrorCode::kInvalidArgument, "Unsupported input dtype");
                }
            } else {
                return ffi::Error(ffi::ErrorCode::kInvalidArgument, "Unsupported FP8 dtype");
            }
        } else {
            // Col-major quantization
            int64_t B, M, N;
            compute_quantize_fp8_rowwise_bmn(shape_vec, valid_axis, B, M, N);

            // Need to cast away const because kernel expects non-const pointer
            float *scale_ptr_nonconst = const_cast<float *>(scale_ptr);

            if (out_dtype_str == "float8_e4m3fn" || out_dtype_str == "float8_e4m3fnuz") {
                if (input_dtype == ffi::F32) {
                    quantize_rowwise_col_major_impl<float32, float8_e4m3_t, float32>(
                        reinterpret_cast<const float32 *>(input_buf), scale_ptr_nonconst,
                        scale_inv_ptr, reinterpret_cast<float8_e4m3_t *>(output_buf), B, M, N,
                        stream);
                } else if (input_dtype == ffi::F16) {
                    quantize_rowwise_col_major_impl<float16, float8_e4m3_t, float32>(
                        reinterpret_cast<const float16 *>(input_buf), scale_ptr_nonconst,
                        scale_inv_ptr, reinterpret_cast<float8_e4m3_t *>(output_buf), B, M, N,
                        stream);
                } else if (input_dtype == ffi::BF16) {
                    quantize_rowwise_col_major_impl<bfloat16, float8_e4m3_t, float32>(
                        reinterpret_cast<const bfloat16 *>(input_buf), scale_ptr_nonconst,
                        scale_inv_ptr, reinterpret_cast<float8_e4m3_t *>(output_buf), B, M, N,
                        stream);
                } else {
                    return ffi::Error(ffi::ErrorCode::kInvalidArgument, "Unsupported input dtype");
                }
            } else if (out_dtype_str == "float8_e5m2" || out_dtype_str == "float8_e5m2fnuz") {
                if (input_dtype == ffi::F32) {
                    quantize_rowwise_col_major_impl<float32, float8_e5m2_t, float32>(
                        reinterpret_cast<const float32 *>(input_buf), scale_ptr_nonconst,
                        scale_inv_ptr, reinterpret_cast<float8_e5m2_t *>(output_buf), B, M, N,
                        stream);
                } else if (input_dtype == ffi::F16) {
                    quantize_rowwise_col_major_impl<float16, float8_e5m2_t, float32>(
                        reinterpret_cast<const float16 *>(input_buf), scale_ptr_nonconst,
                        scale_inv_ptr, reinterpret_cast<float8_e5m2_t *>(output_buf), B, M, N,
                        stream);
                } else if (input_dtype == ffi::BF16) {
                    quantize_rowwise_col_major_impl<bfloat16, float8_e5m2_t, float32>(
                        reinterpret_cast<const bfloat16 *>(input_buf), scale_ptr_nonconst,
                        scale_inv_ptr, reinterpret_cast<float8_e5m2_t *>(output_buf), B, M, N,
                        stream);
                } else {
                    return ffi::Error(ffi::ErrorCode::kInvalidArgument, "Unsupported input dtype");
                }
            } else {
                return ffi::Error(ffi::ErrorCode::kInvalidArgument, "Unsupported FP8 dtype");
            }
        }
    } else {
        // Auto-scale: compute scale from amax
        if (is_row_major) {
            const int64_t inner_len = input_shape[valid_axis];
            int64_t       outer_len = 1;
            for (auto dim : input_shape) {
                outer_len *= dim;
            }
            const int64_t outer_len_val = outer_len / inner_len;

            void  *workspace_ptr = workspace.untyped_data();
            float *temp_scale    = reinterpret_cast<float *>(workspace_ptr);

            if (out_dtype_str == "float8_e4m3fn" || out_dtype_str == "float8_e4m3fnuz") {
                if (input_dtype == ffi::F32) {
                    quantize_rowwise_row_major_impl<float32, float8_e4m3_t, float32, false>(
                        reinterpret_cast<const float32 *>(input_buf), temp_scale, scale_inv_ptr,
                        reinterpret_cast<float8_e4m3_t *>(output_buf), outer_len_val, inner_len,
                        stream);
                } else if (input_dtype == ffi::F16) {
                    quantize_rowwise_row_major_impl<float16, float8_e4m3_t, float32, false>(
                        reinterpret_cast<const float16 *>(input_buf), temp_scale, scale_inv_ptr,
                        reinterpret_cast<float8_e4m3_t *>(output_buf), outer_len_val, inner_len,
                        stream);
                } else if (input_dtype == ffi::BF16) {
                    quantize_rowwise_row_major_impl<bfloat16, float8_e4m3_t, float32, false>(
                        reinterpret_cast<const bfloat16 *>(input_buf), temp_scale, scale_inv_ptr,
                        reinterpret_cast<float8_e4m3_t *>(output_buf), outer_len_val, inner_len,
                        stream);
                } else {
                    return ffi::Error(ffi::ErrorCode::kInvalidArgument, "Unsupported input dtype");
                }
            } else if (out_dtype_str == "float8_e5m2" || out_dtype_str == "float8_e5m2fnuz") {
                if (input_dtype == ffi::F32) {
                    quantize_rowwise_row_major_impl<float32, float8_e5m2_t, float32, false>(
                        reinterpret_cast<const float32 *>(input_buf), temp_scale, scale_inv_ptr,
                        reinterpret_cast<float8_e5m2_t *>(output_buf), outer_len_val, inner_len,
                        stream);
                } else if (input_dtype == ffi::F16) {
                    quantize_rowwise_row_major_impl<float16, float8_e5m2_t, float32, false>(
                        reinterpret_cast<const float16 *>(input_buf), temp_scale, scale_inv_ptr,
                        reinterpret_cast<float8_e5m2_t *>(output_buf), outer_len_val, inner_len,
                        stream);
                } else if (input_dtype == ffi::BF16) {
                    quantize_rowwise_row_major_impl<bfloat16, float8_e5m2_t, float32, false>(
                        reinterpret_cast<const bfloat16 *>(input_buf), temp_scale, scale_inv_ptr,
                        reinterpret_cast<float8_e5m2_t *>(output_buf), outer_len_val, inner_len,
                        stream);
                } else {
                    return ffi::Error(ffi::ErrorCode::kInvalidArgument, "Unsupported input dtype");
                }
            } else {
                return ffi::Error(ffi::ErrorCode::kInvalidArgument, "Unsupported FP8 dtype");
            }
        } else {
            // Col-major: requires reduce-col to compute amax, then compute scale
            int64_t B, M, N;
            compute_quantize_fp8_rowwise_bmn(shape_vec, valid_axis, B, M, N);

            void *workspace_ptr = workspace.untyped_data();
            char *ws_char       = reinterpret_cast<char *>(workspace_ptr);

            int64_t amax_size = ((B * N * sizeof(float) + 255) / 256) * 256;
            float  *amax_ptr  = reinterpret_cast<float *>(ws_char);

            const int64_t ws_size              = get_reduce_col_workspace_sizes<float>(B, M, N);
            void         *reduce_workspace_ptr = ws_char + amax_size;

            // Step 1: Reduce-Col to compute amax
            if (input_dtype == ffi::F32) {
                reduce_col<float32, float, float>(PrimusTurboReduceOp::REDUCE_ABS_MAX,
                                                  reinterpret_cast<const float32 *>(input_buf),
                                                  amax_ptr, B, M, N, ws_size, reduce_workspace_ptr,
                                                  stream);
            } else if (input_dtype == ffi::F16) {
                reduce_col<float16, float, float>(PrimusTurboReduceOp::REDUCE_ABS_MAX,
                                                  reinterpret_cast<const float16 *>(input_buf),
                                                  amax_ptr, B, M, N, ws_size, reduce_workspace_ptr,
                                                  stream);
            } else if (input_dtype == ffi::BF16) {
                reduce_col<bfloat16, float, float>(PrimusTurboReduceOp::REDUCE_ABS_MAX,
                                                   reinterpret_cast<const bfloat16 *>(input_buf),
                                                   amax_ptr, B, M, N, ws_size, reduce_workspace_ptr,
                                                   stream);
            } else {
                return ffi::Error(ffi::ErrorCode::kInvalidArgument, "Unsupported input dtype");
            }

            // temp_scale buffer: kernel writes scale output here
            // We can reuse part of workspace or scale_inv_ptr directly
            float *temp_scale = scale_inv_ptr;

            // Step 2: Compute scale from amax
            ffi::DataType fp8_dtype;
            if (out_dtype_str == "float8_e4m3fn") {
                fp8_dtype = ffi::DataType::F8E4M3FN;
            } else if (out_dtype_str == "float8_e4m3fnuz") {
                fp8_dtype = ffi::DataType::F8E4M3FNUZ;
            } else if (out_dtype_str == "float8_e5m2") {
                fp8_dtype = ffi::DataType::F8E5M2;
            } else {
                fp8_dtype = ffi::DataType::F8E5M2FNUZ;
            }
            float fp8_max = get_float8_max(fp8_dtype);

            compute_scale_from_amax<float>(amax_ptr, fp8_max, temp_scale, scale_inv_ptr, B * N,
                                           stream);

            // Step 3: Quantize using computed scale
            if (out_dtype_str == "float8_e4m3fn" || out_dtype_str == "float8_e4m3fnuz") {
                if (input_dtype == ffi::F32) {
                    quantize_rowwise_col_major_impl<float32, float8_e4m3_t, float>(
                        reinterpret_cast<const float32 *>(input_buf), temp_scale, scale_inv_ptr,
                        reinterpret_cast<float8_e4m3_t *>(output_buf), B, M, N, stream);
                } else if (input_dtype == ffi::F16) {
                    quantize_rowwise_col_major_impl<float16, float8_e4m3_t, float>(
                        reinterpret_cast<const float16 *>(input_buf), temp_scale, scale_inv_ptr,
                        reinterpret_cast<float8_e4m3_t *>(output_buf), B, M, N, stream);
                } else if (input_dtype == ffi::BF16) {
                    quantize_rowwise_col_major_impl<bfloat16, float8_e4m3_t, float>(
                        reinterpret_cast<const bfloat16 *>(input_buf), temp_scale, scale_inv_ptr,
                        reinterpret_cast<float8_e4m3_t *>(output_buf), B, M, N, stream);
                }
            } else if (out_dtype_str == "float8_e5m2" || out_dtype_str == "float8_e5m2fnuz") {
                if (input_dtype == ffi::F32) {
                    quantize_rowwise_col_major_impl<float32, float8_e5m2_t, float>(
                        reinterpret_cast<const float32 *>(input_buf), temp_scale, scale_inv_ptr,
                        reinterpret_cast<float8_e5m2_t *>(output_buf), B, M, N, stream);
                } else if (input_dtype == ffi::F16) {
                    quantize_rowwise_col_major_impl<float16, float8_e5m2_t, float>(
                        reinterpret_cast<const float16 *>(input_buf), temp_scale, scale_inv_ptr,
                        reinterpret_cast<float8_e5m2_t *>(output_buf), B, M, N, stream);
                } else if (input_dtype == ffi::BF16) {
                    quantize_rowwise_col_major_impl<bfloat16, float8_e5m2_t, float>(
                        reinterpret_cast<const bfloat16 *>(input_buf), temp_scale, scale_inv_ptr,
                        reinterpret_cast<float8_e5m2_t *>(output_buf), B, M, N, stream);
                }
            }
        }
    }

    return ffi::Error::Success();
}

// Register FFI handlers
XLA_FFI_DEFINE_HANDLER_SYMBOL(QuantizeFP8TensorwiseHandler, QuantizeFP8TensorwiseFFI,
                              ffi::Ffi::Bind()
                                  .Arg<ffi::AnyBuffer>()                   // input (F32/F16/BF16)
                                  .Attr<std::string_view>("out_dtype_str") // dest_dtype
                                  .Arg<ffi::Buffer<ffi::DataType::F32>>()  // scale_opt
                                  .Arg<ffi::AnyBuffer>()                   // workspace
                                  .Ret<ffi::AnyBuffer>()                   // output (fp8)
                                  .Ret<ffi::Buffer<ffi::DataType::F32>>()  // scale_inv
);

XLA_FFI_DEFINE_HANDLER_SYMBOL(DequantizeFP8TensorwiseHandler, DequantizeFP8TensorwiseFFI,
                              ffi::Ffi::Bind()
                                  .Arg<ffi::AnyBuffer>()                   // input (fp8)
                                  .Arg<ffi::Buffer<ffi::DataType::F32>>()  // scale_inv
                                  .Attr<std::string_view>("out_dtype_str") // dest_dtype
                                  .Ret<ffi::Buffer<ffi::DataType::F32>>()  // output
);

XLA_FFI_DEFINE_HANDLER_SYMBOL(QuantizeFP8RowwiseHandler, QuantizeFP8RowwiseFFI,
                              ffi::Ffi::Bind()
                                  .Arg<ffi::AnyBuffer>()                   // input (F32/F16/BF16)
                                  .Attr<std::string_view>("out_dtype_str") // dest_dtype
                                  .Attr<int64_t>("axis")                   // axis
                                  .Arg<ffi::Buffer<ffi::DataType::F32>>()  // scale_opt
                                  .Arg<ffi::AnyBuffer>()                   // workspace
                                  .Ret<ffi::AnyBuffer>()                   // output (fp8)
                                  .Ret<ffi::Buffer<ffi::DataType::F32>>()  // scale_inv
);

} // namespace primus_turbo::jax
