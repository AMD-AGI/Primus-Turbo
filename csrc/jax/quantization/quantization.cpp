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
// Signature matches PyTorch: quantize_fp8_tensorwise(input, dest_dtype, scale_opt)
// Output parameters (via ffi::Result): output, scale_inv
// Return value: ffi::Error (status)
ffi::Error QuantizeFP8TensorwiseFFI(ffi::Buffer<ffi::DataType::F32>              input,
                                    std::string_view                             out_dtype_str,
                                    ffi::Buffer<ffi::DataType::F32>              scale_opt,
                                    ffi::Result<ffi::AnyBuffer>                  output,
                                    ffi::Result<ffi::Buffer<ffi::DataType::F32>> scale_inv_out) {
    hipStream_t stream = nullptr;

    auto    input_shape = input.dimensions();
    int64_t n           = 1;
    for (auto dim : input_shape) {
        n *= dim;
    }

    // Check if scale is provided (scale_opt.numel() > 0)
    // This matches PyTorch's c10::optional<at::Tensor> behavior
    bool has_scale = (scale_opt.dimensions().size() > 0 && scale_opt.dimensions()[0] > 0);

    float scale_inv_val;

    if (has_scale) {
        // Use provided scale (GPU memory)
        const float *scale_ptr = scale_opt.typed_data();

        // Copy scale to host to compute scale_inv
        float scale_val_host;
        hipMemcpy(&scale_val_host, scale_ptr, sizeof(float), hipMemcpyDeviceToHost);
        scale_inv_val = 1.0f / scale_val_host;

        // Quantize
        auto output_buf = output->untyped_data();

        if (out_dtype_str == "float8_e4m3fn" || out_dtype_str == "float8_e4m3fnuz") {
            quantize_tensorwise_impl<float32, float8_e4m3_t, float32>(
                input.typed_data(), scale_ptr, reinterpret_cast<float8_e4m3_t *>(output_buf), n,
                stream);
        } else if (out_dtype_str == "float8_e5m2" || out_dtype_str == "float8_e5m2fnuz") {
            quantize_tensorwise_impl<float32, float8_e5m2_t, float32>(
                input.typed_data(), scale_ptr, reinterpret_cast<float8_e5m2_t *>(output_buf), n,
                stream);
        } else {
            return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                              "Unsupported FP8 dtype for quantization");
        }
    } else {
        // TODO: Auto-compute scale from amax (requires GPU reduce kernel)
        return ffi::Error(ffi::ErrorCode::kUnimplemented,
                          "Auto-scale computation not yet implemented in JAX quantize_fp8. "
                          "Please provide scale explicitly.");
    }

    // Copy scale_inv to output
    scale_inv_out->typed_data()[0] = scale_inv_val;

    return ffi::Error::Success();
}

// Tensorwise Dequantize FP8 FFI
// Signature matches PyTorch: dequantize_fp8_tensorwise(input, scale_inv, dest_dtype)
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
// Signature matches PyTorch: quantize_fp8_rowwise(input, dest_dtype, axis, scale_opt)
// Output parameters (via ffi::Result): output, scale_inv
// Return value: ffi::Error (status)
ffi::Error QuantizeFP8RowwiseFFI(ffi::Buffer<ffi::DataType::F32> input,
                                 std::string_view out_dtype_str, int64_t axis,
                                 ffi::Buffer<ffi::DataType::F32>              scale_opt,
                                 ffi::Result<ffi::AnyBuffer>                  output,
                                 ffi::Result<ffi::Buffer<ffi::DataType::F32>> scale_inv_out) {
    hipStream_t stream = nullptr;

    auto                 input_shape = input.dimensions();
    std::vector<int64_t> shape_vec(input_shape.begin(), input_shape.end());

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
                quantize_rowwise_row_major_impl<float32, float8_e4m3_t, float32, true>(
                    input.typed_data(), scale_ptr_nonconst, scale_inv_ptr,
                    reinterpret_cast<float8_e4m3_t *>(output_buf), outer_len_val, inner_len,
                    stream);
            } else if (out_dtype_str == "float8_e5m2" || out_dtype_str == "float8_e5m2fnuz") {
                quantize_rowwise_row_major_impl<float32, float8_e5m2_t, float32, true>(
                    input.typed_data(), scale_ptr_nonconst, scale_inv_ptr,
                    reinterpret_cast<float8_e5m2_t *>(output_buf), outer_len_val, inner_len,
                    stream);
            } else {
                return ffi::Error(ffi::ErrorCode::kInvalidArgument, "Unsupported FP8 dtype");
            }
        } else {
            int64_t B, M, N;
            compute_quantize_fp8_rowwise_bmn(shape_vec, valid_axis, B, M, N);

            // Need to cast away const because kernel expects non-const pointer
            float *scale_ptr_nonconst = const_cast<float *>(scale_ptr);

            if (out_dtype_str == "float8_e4m3fn" || out_dtype_str == "float8_e4m3fnuz") {
                quantize_rowwise_col_major_impl<float32, float8_e4m3_t, float32>(
                    input.typed_data(), scale_ptr_nonconst, scale_inv_ptr,
                    reinterpret_cast<float8_e4m3_t *>(output_buf), B, M, N, stream);
            } else if (out_dtype_str == "float8_e5m2" || out_dtype_str == "float8_e5m2fnuz") {
                quantize_rowwise_col_major_impl<float32, float8_e5m2_t, float32>(
                    input.typed_data(), scale_ptr_nonconst, scale_inv_ptr,
                    reinterpret_cast<float8_e5m2_t *>(output_buf), B, M, N, stream);
            } else {
                return ffi::Error(ffi::ErrorCode::kInvalidArgument, "Unsupported FP8 dtype");
            }
        }
    } else {
        // Auto-compute scale from amax
        if (is_row_major) {
            const int64_t inner_len = input_shape[valid_axis];
            int64_t       outer_len = 1;
            for (auto dim : input_shape) {
                outer_len *= dim;
            }
            const int64_t outer_len_val = outer_len / inner_len;

            if (out_dtype_str == "float8_e4m3fn" || out_dtype_str == "float8_e4m3fnuz") {
                quantize_rowwise_row_major_impl<float32, float8_e4m3_t, float32, false>(
                    input.typed_data(),
                    nullptr, // scale computed internally
                    scale_inv_ptr, reinterpret_cast<float8_e4m3_t *>(output_buf), outer_len_val,
                    inner_len, stream);
            } else if (out_dtype_str == "float8_e5m2" || out_dtype_str == "float8_e5m2fnuz") {
                quantize_rowwise_row_major_impl<float32, float8_e5m2_t, float32, false>(
                    input.typed_data(), nullptr, scale_inv_ptr,
                    reinterpret_cast<float8_e5m2_t *>(output_buf), outer_len_val, inner_len,
                    stream);
            } else {
                return ffi::Error(ffi::ErrorCode::kInvalidArgument, "Unsupported FP8 dtype");
            }
        } else {
            // Not implemented yet - requires GPU reduce kernel
            return ffi::Error(ffi::ErrorCode::kUnimplemented,
                              "Auto-scale for col-major rowwise quantization not yet implemented. "
                              "Please provide scale explicitly.");
        }
    }

    return ffi::Error::Success();
}

// Register FFI handlers
XLA_FFI_DEFINE_HANDLER_SYMBOL(QuantizeFP8TensorwiseHandler, QuantizeFP8TensorwiseFFI,
                              ffi::Ffi::Bind()
                                  .Arg<ffi::Buffer<ffi::DataType::F32>>()  // input
                                  .Attr<std::string_view>("out_dtype_str") // dest_dtype
                                  .Arg<ffi::Buffer<ffi::DataType::F32>>()  // scale_opt
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
                                  .Arg<ffi::Buffer<ffi::DataType::F32>>()  // input
                                  .Attr<std::string_view>("out_dtype_str") // dest_dtype
                                  .Attr<int64_t>("axis")                   // axis
                                  .Arg<ffi::Buffer<ffi::DataType::F32>>()  // scale_opt
                                  .Ret<ffi::AnyBuffer>()                   // output (fp8)
                                  .Ret<ffi::Buffer<ffi::DataType::F32>>()  // scale_inv
);

} // namespace primus_turbo::jax
