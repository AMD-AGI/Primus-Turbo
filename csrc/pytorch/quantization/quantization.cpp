// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
//
// See LICENSE for license information.

#include "primus_turbo/quantization.h"
#include "primus_turbo/reduce.h"
#include "primus_turbo/shuffle.h"
#include "pytorch/extensions.h"
#include "pytorch/utils.h"

namespace primus_turbo::pytorch {

using namespace primus_turbo::dtype;

// TODO: Check correctness
float get_float8_max(const at::ScalarType dtype) {
    switch (dtype) {
    case at::kFloat8_e4m3fn:
        return 448.0f;
    case at::kFloat8_e4m3fnuz:
        return 240.0f;
    case at::kFloat8_e5m2:
        return 57344.0f;
    case at::kFloat8_e5m2fnuz:
        return 57344.0f;
    default:
        PRIMUS_TURBO_CHECK(false, "Unsupported FP8 type");
        return 1.0f;
    }
}

inline bool is_torch_fp8(const at::ScalarType dtype) {
    return dtype == at::kFloat8_e4m3fn || dtype == at::kFloat8_e4m3fnuz ||
           dtype == at::kFloat8_e5m2 || dtype == at::kFloat8_e5m2fnuz;
}

std::vector<at::Tensor> quantize_fp8_tensorwise(const at::Tensor          input,
                                                const at::ScalarType      dest_dtype,
                                                c10::optional<at::Tensor> scale_opt) {
    PRIMUS_TURBO_CHECK(input.scalar_type() == at::kBFloat16 || input.scalar_type() == at::kHalf ||
                       input.scalar_type() == at::kFloat);
    PRIMUS_TURBO_CHECK(is_torch_fp8(dest_dtype));
    auto stream = at::cuda::getCurrentCUDAStream();

    at::Tensor scale     = torch::empty({}, input.options().dtype(at::kFloat));
    at::Tensor scale_inv = torch::empty({}, input.options().dtype(at::kFloat));

    if (scale_opt.has_value()) {
        scale = scale_opt.value();
        PRIMUS_TURBO_CHECK(scale.numel() == 1, "tensorwise scale must be scalar tensor");
        scale_inv = 1.0f / scale;
    } else {
        // Reduce
        auto          amax      = torch::empty({}, input.options().dtype(at::kFloat));
        const int64_t ws_size   = get_reduce_row_workspace_sizes<float>(1, input.numel());
        auto          workspace = torch::empty({ws_size}, input.options().dtype(at::kByte));
        TORCH_TYPE_SWITCH_FP16_BF16_FP32(input.scalar_type(), InT, {
            reduce_row<InT, float, float>(
                PrimusTurboReduceOp::REDUCE_ABS_MAX, reinterpret_cast<InT *>(input.data_ptr()),
                amax.data_ptr<float>(), 1, input.numel(), ws_size, workspace.data_ptr(), stream);
        });

        // Compute Scale
        const float fp8_max = get_float8_max(dest_dtype);
        compute_scale_from_amax<float>(reinterpret_cast<const float *>(amax.data_ptr()), fp8_max,
                                       reinterpret_cast<float *>(scale.data_ptr()),
                                       reinterpret_cast<float *>(scale_inv.data_ptr()),
                                       amax.numel(), stream);
    }

    // Quantize
    at::Tensor output = torch::empty_like(input, torch::dtype(dest_dtype).device(input.device()));
    TORCH_TYPE_SWITCH_FP16_BF16_FP32(input.scalar_type(), FType, {
        TORCH_TYPE_SWITCH_FP8(output.scalar_type(), QType, {
            quantize_tensorwise_impl<FType, QType>(
                reinterpret_cast<const FType *>(input.data_ptr()),
                reinterpret_cast<const float *>(scale.data_ptr()),
                reinterpret_cast<QType *>(output.data_ptr()), input.numel(), stream);
        });
    });

    return {output, scale_inv};
}

inline void compute_quantize_fp8_rowwise_bmn(const std::vector<int64_t> &shape, int64_t axis,
                                             int64_t &B, int64_t &M, int64_t &N) {
    const int64_t ndim = static_cast<int64_t>(shape.size());
    if (ndim == 0) {
        B = M = N = 1;
        return;
    }
    PRIMUS_TURBO_CHECK(axis >= 0 && axis < ndim);

    auto prod = [](const std::vector<int64_t> &v, int64_t start, int64_t end) {
        return std::accumulate(v.begin() + start, v.begin() + end, int64_t{1},
                               std::multiplies<int64_t>());
    };
    B = prod(shape, 0, axis);
    M = shape[axis];
    N = prod(shape, axis + 1, ndim);
}

std::vector<at::Tensor> quantize_fp8_rowwise(const at::Tensor     input,
                                             const at::ScalarType dest_dtype, const int64_t axis,
                                             c10::optional<at::Tensor> scale_opt) {
    PRIMUS_TURBO_CHECK(input.scalar_type() == at::kBFloat16 || input.scalar_type() == at::kHalf ||
                       input.scalar_type() == at::kFloat);
    PRIMUS_TURBO_CHECK(is_torch_fp8(dest_dtype));

    const int64_t valid_axis = (axis >= 0) ? axis : input.dim() + axis;
    PRIMUS_TURBO_CHECK(valid_axis >= 0 && valid_axis < input.dim());
    const bool is_row_major = valid_axis == (input.dim() - 1);

    std::vector<int64_t> input_shape(input.sizes().begin(), input.sizes().end());
    std::vector<int64_t> scale_shape(input.sizes().begin(), input.sizes().end());
    scale_shape[valid_axis] = 1;
    auto scale              = at::empty(scale_shape, input.options().dtype(at::kFloat));
    auto scale_inv          = at::empty(scale_shape, input.options().dtype(at::kFloat));
    auto output             = at::empty_like(input, input.options().dtype(dest_dtype));

    auto        stream  = at::cuda::getCurrentCUDAStream();
    const float fp8_max = get_float8_max(dest_dtype);
    if (scale_opt.has_value()) {
        PRIMUS_TURBO_CHECK(scale_opt.value().sizes() == at::IntArrayRef(scale_shape));

        scale     = scale_opt.value();
        scale_inv = 1.0f / scale;

        if (is_row_major) {
            const int64_t inner_len = input.sizes()[valid_axis];
            const int64_t outer_len = input.numel() / inner_len;
            TORCH_TYPE_SWITCH_FP16_BF16_FP32(input.scalar_type(), FType, {
                TORCH_TYPE_SWITCH_FP8(output.scalar_type(), QType, {
                    quantize_rowwise_row_major_impl<FType, QType, float, true>(
                        reinterpret_cast<const FType *>(input.data_ptr()),
                        reinterpret_cast<float *>(scale.data_ptr()),
                        reinterpret_cast<float *>(scale_inv.data_ptr()),
                        reinterpret_cast<QType *>(output.data_ptr()), outer_len, inner_len, stream);
                });
            });
        } else {
            int64_t B, M, N;
            compute_quantize_fp8_rowwise_bmn(input_shape, valid_axis, B, M, N);

            TORCH_TYPE_SWITCH_FP16_BF16_FP32(input.scalar_type(), FType, {
                TORCH_TYPE_SWITCH_FP8(output.scalar_type(), QType, {
                    quantize_rowwise_col_major_impl<FType, QType, float>(
                        reinterpret_cast<const FType *>(input.data_ptr()),
                        reinterpret_cast<float *>(scale.data_ptr()),
                        reinterpret_cast<float *>(scale_inv.data_ptr()),
                        reinterpret_cast<QType *>(output.data_ptr()), B, M, N, stream);
                });
            });
        }
    } else {
        if (is_row_major) {
            const int64_t inner_len = input.sizes()[valid_axis];
            const int64_t outer_len = input.numel() / inner_len;
            TORCH_TYPE_SWITCH_FP16_BF16_FP32(input.scalar_type(), FType, {
                TORCH_TYPE_SWITCH_FP8(output.scalar_type(), QType, {
                    quantize_rowwise_row_major_impl<FType, QType, float, false>(
                        reinterpret_cast<const FType *>(input.data_ptr()),
                        reinterpret_cast<float *>(scale.data_ptr()),
                        reinterpret_cast<float *>(scale_inv.data_ptr()),
                        reinterpret_cast<QType *>(output.data_ptr()), outer_len, inner_len, stream);
                });
            });
        } else {
            int64_t B, M, N;
            compute_quantize_fp8_rowwise_bmn(input_shape, valid_axis, B, M, N);

            // AMAX Reduce-Col
            auto          amax      = at::empty_like(scale);
            const int64_t ws_size   = get_reduce_col_workspace_sizes<float>(B, M, N);
            auto          workspace = torch::empty({ws_size}, input.options().dtype(at::kByte));
            TORCH_TYPE_SWITCH_FP16_BF16_FP32(input.scalar_type(), InT, {
                reduce_col<InT, float, float>(PrimusTurboReduceOp::REDUCE_ABS_MAX,
                                              reinterpret_cast<const InT *>(input.data_ptr()),
                                              amax.data_ptr<float>(), B, M, N, ws_size,
                                              workspace.data_ptr(), stream);
            });

            // Scale
            compute_scale_from_amax<float>(reinterpret_cast<const float *>(amax.data_ptr()),
                                           fp8_max, reinterpret_cast<float *>(scale.data_ptr()),
                                           reinterpret_cast<float *>(scale_inv.data_ptr()),
                                           amax.numel(), stream);
            // Quant
            TORCH_TYPE_SWITCH_FP16_BF16_FP32(input.scalar_type(), FType, {
                TORCH_TYPE_SWITCH_FP8(output.scalar_type(), QType, {
                    quantize_rowwise_col_major_impl<FType, QType, float>(
                        reinterpret_cast<const FType *>(input.data_ptr()),
                        reinterpret_cast<float *>(scale.data_ptr()),
                        reinterpret_cast<float *>(scale_inv.data_ptr()),
                        reinterpret_cast<QType *>(output.data_ptr()), B, M, N, stream);
                });
            });
        }
    }
    return {output, scale_inv};
}

// De-Quantize
at::Tensor dequantize_fp8_tensorwise(const at::Tensor input, const at::Tensor scale_inv,
                                     const at::ScalarType dest_dtype) {
    PRIMUS_TURBO_CHECK(dest_dtype == at::kBFloat16 || dest_dtype == at::kHalf ||
                       dest_dtype == at::kFloat);
    PRIMUS_TURBO_CHECK(is_torch_fp8(input.scalar_type()));
    PRIMUS_TURBO_CHECK(scale_inv.numel() == 1, "tensorwise scale_inv must be scalar tensor");
    auto stream = at::cuda::getCurrentCUDAStream();

    at::Tensor output = torch::empty_like(input, torch::dtype(dest_dtype).device(input.device()));
    TORCH_TYPE_SWITCH_FP16_BF16_FP32(output.scalar_type(), FType, {
        TORCH_TYPE_SWITCH_FP8(input.scalar_type(), QType, {
            dequantize_tensorwise_impl<FType, QType>(
                reinterpret_cast<const QType *>(input.data_ptr()),
                reinterpret_cast<const float *>(scale_inv.data_ptr()),
                reinterpret_cast<FType *>(output.data_ptr()), input.numel(), stream);
        });
    });

    return output;
}

at::Tensor dequantize_fp8_rowwise(const at::Tensor input, const at::Tensor scale_inv,
                                  const int64_t axis, const at::ScalarType dest_dtype) {
    PRIMUS_TURBO_CHECK(dest_dtype == at::kBFloat16 || dest_dtype == at::kHalf ||
                       dest_dtype == at::kFloat);
    PRIMUS_TURBO_CHECK(is_torch_fp8(input.scalar_type()));
    PRIMUS_TURBO_CHECK(scale_inv.scalar_type() == at::kFloat,
                       "rowwise scale_inv must be float32 tensor");
    PRIMUS_TURBO_CHECK(input.is_contiguous(), "input must be contiguous");
    PRIMUS_TURBO_CHECK(scale_inv.is_contiguous(), "scale_inv must be contiguous");

    const int64_t valid_axis = (axis >= 0) ? axis : input.dim() + axis;
    PRIMUS_TURBO_CHECK(valid_axis >= 0 && valid_axis < input.dim(),
                       "rowwise dequantize axis out of range");

    std::vector<int64_t> expected_scale_shape(input.sizes().begin(), input.sizes().end());
    expected_scale_shape[valid_axis] = 1;
    PRIMUS_TURBO_CHECK(scale_inv.sizes() == at::IntArrayRef(expected_scale_shape),
                       "scale_inv shape must match input shape with size 1 along axis");

    const bool is_row_major = valid_axis == (input.dim() - 1);

    auto       stream = at::cuda::getCurrentCUDAStream();
    at::Tensor output = torch::empty_like(input, torch::dtype(dest_dtype).device(input.device()));

    if (is_row_major) {
        const int64_t inner_len = input.sizes()[valid_axis];
        const int64_t outer_len = input.numel() / inner_len;
        TORCH_TYPE_SWITCH_FP16_BF16_FP32(output.scalar_type(), FType, {
            TORCH_TYPE_SWITCH_FP8(input.scalar_type(), QType, {
                dequantize_rowwise_row_major_impl<FType, QType, float>(
                    reinterpret_cast<const QType *>(input.data_ptr()),
                    reinterpret_cast<const float *>(scale_inv.data_ptr()),
                    reinterpret_cast<FType *>(output.data_ptr()), outer_len, inner_len, stream);
            });
        });
    } else {
        int64_t              B, M, N;
        std::vector<int64_t> input_shape(input.sizes().begin(), input.sizes().end());
        compute_quantize_fp8_rowwise_bmn(input_shape, valid_axis, B, M, N);
        TORCH_TYPE_SWITCH_FP16_BF16_FP32(output.scalar_type(), FType, {
            TORCH_TYPE_SWITCH_FP8(input.scalar_type(), QType, {
                dequantize_rowwise_col_major_impl<FType, QType, float>(
                    reinterpret_cast<const QType *>(input.data_ptr()),
                    reinterpret_cast<const float *>(scale_inv.data_ptr()),
                    reinterpret_cast<FType *>(output.data_ptr()), B, M, N, stream);
            });
        });
    }

    return output;
}

// Quantize MXFP4 with dual mode
std::vector<at::Tensor> quantize_mxfp4_dual(
    const at::Tensor input, const at::ScalarType dest_dtype, const int64_t padding_align_size,
    const bool rowwise_use_2d_block, const bool rowwise_use_sr, const bool rowwise_use_rht,
    const bool colwise_use_2d_block, const bool colwise_use_sr, const bool colwise_use_rht,
    const bool shuffle_rowwise_scale, const bool shuffle_rowwise, const bool shuffle_colwise_scale,
    const bool shuffle_colwise) {
    using namespace primus_turbo::detail;

    std::function<int64_t(int64_t, int64_t)> cdiv = [](int64_t a, int64_t b) -> int64_t {
        return (a + b - 1) / b;
    };

    PRIMUS_TURBO_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    PRIMUS_TURBO_CHECK(input.scalar_type() == at::kBFloat16 || input.scalar_type() == at::kHalf,
                       "Input must be BFloat16 or Half");
    PRIMUS_TURBO_CHECK(input.dim() == 2, "Input must be 2D");
    PRIMUS_TURBO_CHECK(input.is_contiguous(), "Input must be contiguous");
    PRIMUS_TURBO_CHECK(dest_dtype == at::kFloat4_e2m1fn_x2, "Output must be Float4_e2m1fn_x2.");
    // Guard the public op argument against zero/negative values (would otherwise
    // divide-by-zero in cdiv below) and lock it to the expected MXFP4 constant.
    PRIMUS_TURBO_CHECK(padding_align_size == MXFP4_PADDING_ALIGN_SIZE,
                       "padding_align_size must be ", MXFP4_PADDING_ALIGN_SIZE,
                       " for MXFP4. But got padding_align_size=", padding_align_size);

    const int64_t M = input.size(0);
    const int64_t N = input.size(1);

    const int64_t M_pad = cdiv(M, padding_align_size) * padding_align_size;
    const int64_t N_pad = cdiv(N, padding_align_size) * padding_align_size;

    PRIMUS_TURBO_CHECK(N % MXFP4_BLOCK_SIZE == 0, "N must be divisible by ", MXFP4_BLOCK_SIZE);

    if (shuffle_rowwise) {
        PRIMUS_TURBO_CHECK(M % MXFP4_SHUFFLE_BN == 0, "M must be divisible by ", MXFP4_SHUFFLE_BN,
                           " for shuffled rowwise FP4. But got M=", M);
    }
    if (shuffle_colwise) {
        PRIMUS_TURBO_CHECK(N % MXFP4_SHUFFLE_BN == 0, "N must be divisible by ", MXFP4_SHUFFLE_BN,
                           " for shuffled colwise FP4. But got N=", N);
    }

    auto device = input.device();
    auto stream = at::cuda::getCurrentCUDAStream();

    int64_t rowwise_scale_M_pad = cdiv(M, 256) * 256;
    int64_t rowwise_scale_N     = cdiv(N_pad, MXFP4_BLOCK_SIZE);
    int64_t rowwise_scale_N_pad = cdiv(rowwise_scale_N, 8) * 8;

    int64_t    rowwise_scale_stride = 1;
    at::Tensor rowwise_scale;
    if (shuffle_rowwise_scale) {
        rowwise_scale = at::full({rowwise_scale_M_pad, rowwise_scale_N_pad}, E8M0_EXPONENT_BIAS,
                                 at::TensorOptions().dtype(at::kByte).device(device));
        rowwise_scale_stride = rowwise_scale.stride(0);
    } else {
        rowwise_scale =
            at::empty({M, rowwise_scale_N}, at::TensorOptions().dtype(at::kByte).device(device));
        rowwise_scale_stride = rowwise_scale_N;
    }

    // packed 2 fp4 values in N dimension
    at::Tensor rowwise_output =
        at::empty({M, N_pad / 2}, at::TensorOptions().dtype(at::kByte).device(device));

    int64_t colwise_scale_M_pad = cdiv(N, 256) * 256;
    int64_t colwise_scale_N     = cdiv(M_pad, MXFP4_BLOCK_SIZE);
    int64_t colwise_scale_N_pad = cdiv(colwise_scale_N, 8) * 8;

    at::Tensor colwise_scale;
    int        colwise_scale_stride = 1;
    if (shuffle_colwise_scale) {
        colwise_scale = at::full({colwise_scale_M_pad, colwise_scale_N_pad}, E8M0_EXPONENT_BIAS,
                                 at::TensorOptions().dtype(at::kByte).device(device));
        colwise_scale_stride = colwise_scale.stride(0);
    } else {
        colwise_scale =
            at::empty({N, colwise_scale_N}, at::TensorOptions().dtype(at::kByte).device(device));
        colwise_scale_stride = colwise_scale_N;
    }

    // packed 2 fp4 values in N dimension
    at::Tensor colwise_output =
        at::empty({N, M_pad / 2}, at::TensorOptions().dtype(at::kByte).device(device));

    TORCH_TYPE_SWITCH_FP16_BF16(input.scalar_type(), DType, {
        quantize_mxfp4_dual_impl<DType>(
            reinterpret_cast<DType *>(input.data_ptr()),
            reinterpret_cast<dtype::float4x2_e2m1 *>(rowwise_output.data_ptr()),
            rowwise_scale.data_ptr<uint8_t>(),
            reinterpret_cast<dtype::float4x2_e2m1 *>(colwise_output.data_ptr()),
            colwise_scale.data_ptr<uint8_t>(), M, N, M_pad, N_pad, rowwise_scale_stride,
            colwise_scale_stride, rowwise_scale_N, rowwise_scale_M_pad, rowwise_scale_N_pad, N,
            colwise_scale_N, colwise_scale_M_pad, colwise_scale_N_pad,
            ScalingRecipe(rowwise_use_2d_block, rowwise_use_sr, rowwise_use_rht,
                          shuffle_rowwise_scale, shuffle_rowwise),
            ScalingRecipe(colwise_use_2d_block, colwise_use_sr, colwise_use_rht,
                          shuffle_colwise_scale, shuffle_colwise),
            stream);
    });

    return {rowwise_output.view(at::kFloat4_e2m1fn_x2), rowwise_scale.view(at::kFloat8_e8m0fnu),
            colwise_output.view(at::kFloat4_e2m1fn_x2), colwise_scale.view(at::kFloat8_e8m0fnu)};
}

// Quantize MXFP4 with single mode (rowwise or colwise)
std::vector<at::Tensor> quantize_mxfp4(const at::Tensor input, const at::ScalarType dest_dtype,
                                       const int64_t axis, const int64_t padding_align_size,
                                       const bool use_2d_block, const bool use_sr,
                                       const bool use_rht, const bool shuffle_scale,
                                       const bool shuffle_out) {
    using namespace primus_turbo::detail;

    auto cdiv = [](int64_t a, int64_t b) -> int64_t { return (a + b - 1) / b; };

    PRIMUS_TURBO_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    PRIMUS_TURBO_CHECK(input.scalar_type() == at::kBFloat16 || input.scalar_type() == at::kHalf,
                       "Input must be BFloat16 or Half");
    PRIMUS_TURBO_CHECK(input.dim() == 2, "Input must be 2D");
    PRIMUS_TURBO_CHECK(input.is_contiguous(), "Input must be contiguous");
    PRIMUS_TURBO_CHECK(dest_dtype == at::kFloat4_e2m1fn_x2, "Output must be Float4_e2m1fn_x2.");
    PRIMUS_TURBO_CHECK(axis == 0 || axis == 1, "Axis must be 0 or 1");
    // Guard the public op argument against zero/negative values (would otherwise
    // divide-by-zero in cdiv below) and lock it to the expected MXFP4 constant.
    PRIMUS_TURBO_CHECK(padding_align_size == MXFP4_PADDING_ALIGN_SIZE,
                       "padding_align_size must be ", MXFP4_PADDING_ALIGN_SIZE,
                       " for MXFP4. But got padding_align_size=", padding_align_size);

    QuantizeMode mode       = (axis == 0) ? QuantizeMode::COLWISE : QuantizeMode::ROWWISE;
    const bool   is_rowwise = (mode == QuantizeMode::ROWWISE);

    const int64_t M     = input.size(0);
    const int64_t N     = input.size(1);
    const int64_t M_pad = cdiv(M, padding_align_size) * padding_align_size;
    const int64_t N_pad = cdiv(N, padding_align_size) * padding_align_size;

    PRIMUS_TURBO_CHECK(N % MXFP4_BLOCK_SIZE == 0, "N must be divisible by ", MXFP4_BLOCK_SIZE);

    if (shuffle_out) {
        if (is_rowwise) {
            PRIMUS_TURBO_CHECK(M % MXFP4_SHUFFLE_BN == 0, "M must be divisible by ",
                               MXFP4_SHUFFLE_BN, " for shuffled rowwise FP4. But got M=", M);
        } else {
            PRIMUS_TURBO_CHECK(N % MXFP4_SHUFFLE_BN == 0, "N must be divisible by ",
                               MXFP4_SHUFFLE_BN, " for shuffled colwise FP4. But got N=", N);
        }
    }

    auto device = input.device();
    auto stream = at::cuda::getCurrentCUDAStream();

    // Scale dimensions depend on quantization direction:
    //   Rowwise: scale has shape [M, cdiv(N_pad,32)], shuffle-padded to [scale_M_pad, scale_N_pad]
    //   Colwise: scale has shape [N, cdiv(M_pad,32)], shuffle-padded to [scale_M_pad, scale_N_pad]
    int64_t scale_outer = is_rowwise ? M : N;
    int64_t scale_N = is_rowwise ? cdiv(N_pad, MXFP4_BLOCK_SIZE) : cdiv(M_pad, MXFP4_BLOCK_SIZE);
    int64_t scale_M_pad = cdiv(scale_outer, 256) * 256;
    int64_t scale_N_pad = cdiv(scale_N, 8) * 8;

    int64_t    scale_stride = 1;
    at::Tensor scale_tensor;
    if (shuffle_scale) {
        scale_tensor = at::full({scale_M_pad, scale_N_pad}, E8M0_EXPONENT_BIAS,
                                at::TensorOptions().dtype(at::kByte).device(device));
        scale_stride = scale_tensor.stride(0);
    } else {
        scale_tensor =
            at::empty({scale_outer, scale_N}, at::TensorOptions().dtype(at::kByte).device(device));
        scale_stride = scale_N;
    }

    // Output FP4 tensor (2 FP4 values packed per byte):
    //   Rowwise: [M, N_pad/2]    Colwise: [N, M_pad/2]
    int64_t    output_rows = is_rowwise ? M : N;
    int64_t    output_cols = is_rowwise ? (N_pad / 2) : (M_pad / 2);
    at::Tensor output =
        at::empty({output_rows, output_cols}, at::TensorOptions().dtype(at::kByte).device(device));

    TORCH_TYPE_SWITCH_FP16_BF16(input.scalar_type(), DType, {
        quantize_mxfp4_impl<DType>(
            reinterpret_cast<DType *>(input.data_ptr()),
            reinterpret_cast<dtype::float4x2_e2m1 *>(output.data_ptr()),
            scale_tensor.data_ptr<uint8_t>(), mode, M, N, M_pad, N_pad, scale_stride, scale_N,
            scale_M_pad, scale_N_pad,
            ScalingRecipe(use_2d_block, use_sr, use_rht, shuffle_scale, shuffle_out), stream);
    });

    return {output.view(at::kFloat4_e2m1fn_x2), scale_tensor.view(at::kFloat8_e8m0fnu)};
}

// Quantize MXFP8 with dual mode
std::vector<at::Tensor>
quantize_mxfp8_dual(const at::Tensor input, const at::ScalarType dest_dtype,
                    const int64_t padding_align_size, const bool rowwise_use_2d_block,
                    const bool colwise_use_2d_block, const bool shuffle_rowwise_scale,
                    const bool shuffle_rowwise, const bool shuffle_colwise_scale,
                    const bool shuffle_colwise) {
    using namespace primus_turbo::detail;

    std::function<int64_t(int64_t, int64_t)> cdiv = [](int64_t a, int64_t b) -> int64_t {
        return (a + b - 1) / b;
    };

    PRIMUS_TURBO_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    PRIMUS_TURBO_CHECK(input.scalar_type() == at::kBFloat16 || input.scalar_type() == at::kHalf,
                       "Input must be BFloat16 or Half");
    PRIMUS_TURBO_CHECK(input.dim() == 2, "Input must be 2D");
    PRIMUS_TURBO_CHECK(input.is_contiguous(), "Input must be contiguous");
    PRIMUS_TURBO_CHECK(dest_dtype == at::kFloat8_e4m3fn || dest_dtype == at::kFloat8_e5m2,
                       "Output must be Float8_e4m3fn or Float8_e5m2");
    // Guard the public op argument against zero/negative values (would otherwise
    // divide-by-zero in cdiv below) and lock it to the expected MXFP8 constant.
    PRIMUS_TURBO_CHECK(padding_align_size == MXFP8_PADDING_ALIGN_SIZE,
                       "padding_align_size must be ", MXFP8_PADDING_ALIGN_SIZE,
                       " for MXFP8. But got padding_align_size=", padding_align_size);

    const int64_t M = input.size(0);
    const int64_t N = input.size(1);

    const int64_t M_pad = cdiv(M, padding_align_size) * padding_align_size;
    const int64_t N_pad = cdiv(N, padding_align_size) * padding_align_size;

    PRIMUS_TURBO_CHECK(N % MXFP8_BLOCK_SIZE == 0, "N must be divisible by ", MXFP8_BLOCK_SIZE);

    if (shuffle_rowwise) {
        PRIMUS_TURBO_CHECK(M % MXFP8_SHUFFLE_BN == 0, "M must be divisible by ", MXFP8_SHUFFLE_BN,
                           " for shuffled rowwise FP8. But got M=", M);
    }
    if (shuffle_colwise) {
        PRIMUS_TURBO_CHECK(N % MXFP8_SHUFFLE_BN == 0, "N must be divisible by ", MXFP8_SHUFFLE_BN,
                           " for shuffled colwise FP8. But got N=", N);
    }

    auto device = input.device();
    auto stream = at::cuda::getCurrentCUDAStream();

    int64_t rowwise_scale_M_pad = cdiv(M, 256) * 256;
    int64_t rowwise_scale_N     = cdiv(N_pad, MXFP8_BLOCK_SIZE);
    int64_t rowwise_scale_N_pad = cdiv(rowwise_scale_N, 8) * 8;

    int64_t    rowwise_scale_stride = 1;
    at::Tensor rowwise_scale;
    if (shuffle_rowwise_scale) {
        rowwise_scale = at::full({rowwise_scale_M_pad, rowwise_scale_N_pad}, E8M0_EXPONENT_BIAS,
                                 at::TensorOptions().dtype(at::kByte).device(device));
        rowwise_scale_stride = rowwise_scale.stride(0);
    } else {
        rowwise_scale =
            at::empty({M, rowwise_scale_N}, at::TensorOptions().dtype(at::kByte).device(device));
        rowwise_scale_stride = rowwise_scale_N;
    }

    at::Tensor rowwise_output =
        at::empty({M, N_pad}, at::TensorOptions().dtype(at::kByte).device(device));

    int64_t colwise_scale_M_pad = cdiv(N, 256) * 256;
    int64_t colwise_scale_N     = cdiv(M_pad, MXFP8_BLOCK_SIZE);
    int64_t colwise_scale_N_pad = cdiv(colwise_scale_N, 8) * 8;

    at::Tensor colwise_scale;
    int        colwise_scale_stride = 1;
    if (shuffle_colwise_scale) {
        colwise_scale = at::full({colwise_scale_M_pad, colwise_scale_N_pad}, E8M0_EXPONENT_BIAS,
                                 at::TensorOptions().dtype(at::kByte).device(device));
        colwise_scale_stride = colwise_scale.stride(0);
    } else {
        colwise_scale =
            at::empty({N, colwise_scale_N}, at::TensorOptions().dtype(at::kByte).device(device));
        colwise_scale_stride = colwise_scale_N;
    }

    at::Tensor colwise_output =
        at::empty({N, M_pad}, at::TensorOptions().dtype(at::kByte).device(device));

    TORCH_TYPE_SWITCH_FP16_BF16(
        input.scalar_type(), IType, {TORCH_TYPE_SWITCH_FP8(dest_dtype, OType, {
            quantize_mxfp8_dual_impl<IType, OType>(
                reinterpret_cast<IType *>(input.data_ptr()),
                reinterpret_cast<OType *>(rowwise_output.data_ptr()),
                rowwise_scale.data_ptr<uint8_t>(),
                reinterpret_cast<OType *>(colwise_output.data_ptr()),
                colwise_scale.data_ptr<uint8_t>(), M, N, M_pad, N_pad, rowwise_scale_stride,
                colwise_scale_stride, rowwise_scale_N, rowwise_scale_M_pad, rowwise_scale_N_pad, N,
                colwise_scale_N, colwise_scale_M_pad, colwise_scale_N_pad,
                ScalingRecipe(rowwise_use_2d_block, false, false, shuffle_rowwise_scale,
                              shuffle_rowwise),
                ScalingRecipe(colwise_use_2d_block, false, false, shuffle_colwise_scale,
                              shuffle_colwise),
                stream);
        })});

    return {rowwise_output.view(dest_dtype), rowwise_scale.view(at::kFloat8_e8m0fnu),
            colwise_output.view(dest_dtype), colwise_scale.view(at::kFloat8_e8m0fnu)};
}

// Quantize MXFP8 with single mode (rowwise or colwise)
std::vector<at::Tensor> quantize_mxfp8(const at::Tensor input, const at::ScalarType dest_dtype,
                                       const int64_t axis, const int64_t padding_align_size,
                                       const bool use_2d_block, const bool shuffle_scale,
                                       const bool shuffle_out) {
    using namespace primus_turbo::detail;

    auto cdiv = [](int64_t a, int64_t b) -> int64_t { return (a + b - 1) / b; };

    PRIMUS_TURBO_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    PRIMUS_TURBO_CHECK(input.scalar_type() == at::kBFloat16 || input.scalar_type() == at::kHalf,
                       "Input must be BFloat16 or Half");
    PRIMUS_TURBO_CHECK(input.dim() == 2, "Input must be 2D");
    PRIMUS_TURBO_CHECK(input.is_contiguous(), "Input must be contiguous");
    PRIMUS_TURBO_CHECK(dest_dtype == at::kFloat8_e4m3fn || dest_dtype == at::kFloat8_e5m2,
                       "Output must be Float8_e4m3fn or Float8_e5m2");
    PRIMUS_TURBO_CHECK(axis == 0 || axis == 1, "Axis must be 0 or 1");
    // Guard the public op argument against zero/negative values (would otherwise
    // divide-by-zero in cdiv below) and lock it to the expected MXFP8 constant.
    PRIMUS_TURBO_CHECK(padding_align_size == MXFP8_PADDING_ALIGN_SIZE,
                       "padding_align_size must be ", MXFP8_PADDING_ALIGN_SIZE,
                       " for MXFP8. But got padding_align_size=", padding_align_size);

    QuantizeMode mode       = (axis == 0) ? QuantizeMode::COLWISE : QuantizeMode::ROWWISE;
    const bool   is_rowwise = (mode == QuantizeMode::ROWWISE);

    const int64_t M     = input.size(0);
    const int64_t N     = input.size(1);
    const int64_t M_pad = cdiv(M, padding_align_size) * padding_align_size;
    const int64_t N_pad = cdiv(N, padding_align_size) * padding_align_size;

    PRIMUS_TURBO_CHECK(N % MXFP8_BLOCK_SIZE == 0, "N must be divisible by ", MXFP8_BLOCK_SIZE);

    if (shuffle_out) {
        if (is_rowwise) {
            PRIMUS_TURBO_CHECK(M % MXFP8_SHUFFLE_BN == 0, "M must be divisible by ",
                               MXFP8_SHUFFLE_BN, " for shuffled rowwise FP8. But got M=", M);
        } else {
            PRIMUS_TURBO_CHECK(N % MXFP8_SHUFFLE_BN == 0, "N must be divisible by ",
                               MXFP8_SHUFFLE_BN, " for shuffled colwise FP8. But got N=", N);
        }
    }

    auto device = input.device();
    auto stream = at::cuda::getCurrentCUDAStream();

    // Scale dimensions depend on quantization direction:
    //   Rowwise: scale has shape [M, cdiv(N_pad,32)], shuffle-padded to [scale_M_pad, scale_N_pad]
    //   Colwise: scale has shape [N, cdiv(M_pad,32)], shuffle-padded to [scale_M_pad, scale_N_pad]
    int64_t scale_outer = is_rowwise ? M : N;
    int64_t scale_N = is_rowwise ? cdiv(N_pad, MXFP8_BLOCK_SIZE) : cdiv(M_pad, MXFP8_BLOCK_SIZE);
    int64_t scale_M_pad = cdiv(scale_outer, 256) * 256;
    int64_t scale_N_pad = cdiv(scale_N, 8) * 8;

    int64_t    scale_stride = 1;
    at::Tensor scale_tensor;
    if (shuffle_scale) {
        scale_tensor = at::full({scale_M_pad, scale_N_pad}, E8M0_EXPONENT_BIAS,
                                at::TensorOptions().dtype(at::kByte).device(device));
        scale_stride = scale_tensor.stride(0);
    } else {
        scale_tensor =
            at::empty({scale_outer, scale_N}, at::TensorOptions().dtype(at::kByte).device(device));
        scale_stride = scale_N;
    }

    // Output FP8 tensor (1 byte per element):
    //   Rowwise: [M, N_pad]    Colwise: [N, M_pad]
    int64_t    output_rows = is_rowwise ? M : N;
    int64_t    output_cols = is_rowwise ? N_pad : M_pad;
    at::Tensor output =
        at::empty({output_rows, output_cols}, at::TensorOptions().dtype(at::kByte).device(device));

    TORCH_TYPE_SWITCH_FP16_BF16(
        input.scalar_type(), IType, {TORCH_TYPE_SWITCH_FP8(dest_dtype, OType, {
            quantize_mxfp8_impl<IType, OType>(
                reinterpret_cast<IType *>(input.data_ptr()),
                reinterpret_cast<OType *>(output.data_ptr()), scale_tensor.data_ptr<uint8_t>(),
                mode, M, N, M_pad, N_pad, scale_stride, scale_N, scale_M_pad, scale_N_pad,
                ScalingRecipe(use_2d_block, false, false, shuffle_scale, shuffle_out), stream);
        })});

    return {output.view(dest_dtype), scale_tensor.view(at::kFloat8_e8m0fnu)};
}

// Fused single-pass row + segment-padded col blockwise FP8 quant for grouped GEMM.
// One bf16/fp16 read of `input` [M, N] emits the row-wise scaled tensor (fwd/dgrad)
// and the segment-padded col-wise scaled tensor (variable-K wgrad). Row scales are
// pshuffled [N_blocks, M] to match the persistent GEMM's coalesced scale reads.
std::vector<at::Tensor> quantize_fp8_blockwise_segment_m_row_col(const at::Tensor     input,
                                                                 const at::ScalarType dest_dtype,
                                                                 const int64_t        block_size,
                                                                 const at::Tensor     group_lens,
                                                                 const at::Tensor     group_offs) {
    PRIMUS_TURBO_CHECK(input.scalar_type() == at::kBFloat16 || input.scalar_type() == at::kHalf);
    PRIMUS_TURBO_CHECK(is_torch_fp8(dest_dtype));
    PRIMUS_TURBO_CHECK(input.dim() == 2);
    PRIMUS_TURBO_CHECK(block_size == 128);
    for (const auto &t : {group_lens, group_offs}) {
        PRIMUS_TURBO_CHECK(t.is_cuda(), "group_lens / group_offs must be CUDA tensors");
        PRIMUS_TURBO_CHECK(t.scalar_type() == at::kLong, "group_lens / group_offs must be int64");
        PRIMUS_TURBO_CHECK(t.is_contiguous(), "group_lens / group_offs must be contiguous");
    }

    const int64_t M          = input.size(0);
    const int64_t N          = input.size(1);
    const int     num_groups = static_cast<int>(group_lens.size(0));

    auto stream = at::cuda::getCurrentCUDAStream();

    // Segment-padded group offsets on-device (avoids div + zeros + cumsum host ops).
    auto var_k_group_lens = at::empty({num_groups}, group_lens.options());
    auto var_k_group_offs = at::empty({num_groups + 1}, group_lens.options());
    compute_padded_group_offs<int64_t>(reinterpret_cast<const int64_t *>(group_lens.data_ptr()),
                                       reinterpret_cast<int64_t *>(var_k_group_lens.data_ptr()),
                                       reinterpret_cast<int64_t *>(var_k_group_offs.data_ptr()),
                                       num_groups, block_size, stream);

    const int64_t M_padded_max = M + num_groups * block_size;

    // Kernel mask-writes cover every position read downstream, so skip zero-init.
    // Row scales emitted pshuffled [N_blocks, M] to match the fwd GEMM layout.
    auto x_fp8_row        = at::empty({M, N}, input.options().dtype(dest_dtype));
    auto x_fp8_col_padded = at::empty({M_padded_max, N}, input.options().dtype(dest_dtype));
    auto x_scales_row =
        at::empty({(N + block_size - 1) / block_size, M}, input.options().dtype(at::kFloat));
    auto x_scales_col_padded = at::empty({(M_padded_max + block_size - 1) / block_size, N},
                                         input.options().dtype(at::kFloat));

    const float fp8_max = get_float8_max(dest_dtype);
    TORCH_TYPE_SWITCH_FP16_BF16(input.scalar_type(), FType, {
        TORCH_TYPE_SWITCH_FP8(x_fp8_row.scalar_type(), QType, {
            quantize_blockwise_segment_m_row_col_impl<FType, QType>(
                reinterpret_cast<const FType *>(input.data_ptr()),
                reinterpret_cast<QType *>(x_fp8_row.data_ptr()),
                reinterpret_cast<QType *>(x_fp8_col_padded.data_ptr()),
                reinterpret_cast<float *>(x_scales_row.data_ptr()),
                reinterpret_cast<float *>(x_scales_col_padded.data_ptr()),
                reinterpret_cast<const int64_t *>(group_offs.data_ptr()),
                reinterpret_cast<const int64_t *>(var_k_group_offs.data_ptr()), M, N, M_padded_max,
                num_groups, fp8_max, stream);
        });
    });

    return {x_fp8_row,           x_fp8_col_padded, x_scales_row,
            x_scales_col_padded, var_k_group_lens, var_k_group_offs};
}

// Blockwise FP8 weight quant: 2D [M, N] or 3D [B, M, N], one scalar scale per [128,128] tile.
std::vector<at::Tensor> quantize_fp8_blockwise_for_weight(const at::Tensor     input,
                                                          const at::ScalarType dest_dtype,
                                                          const int64_t        block_size) {
    PRIMUS_TURBO_CHECK(input.scalar_type() == at::kBFloat16 || input.scalar_type() == at::kHalf);
    PRIMUS_TURBO_CHECK(is_torch_fp8(dest_dtype));
    PRIMUS_TURBO_CHECK(input.dim() == 2 || input.dim() == 3,
                       "weight quant requires 2D or 3D input");
    PRIMUS_TURBO_CHECK(block_size == 128, "only block_size=128 currently supported");

    const bool    is_2d    = (input.dim() == 2);
    const int64_t B        = is_2d ? 1 : input.size(0);
    const int64_t M        = is_2d ? input.size(0) : input.size(1);
    const int64_t N        = is_2d ? input.size(1) : input.size(2);
    const int64_t m_blocks = (M + block_size - 1) / block_size;
    const int64_t n_blocks = (N + block_size - 1) / block_size;

    std::vector<int64_t> out_shape =
        is_2d ? std::vector<int64_t>{M, N} : std::vector<int64_t>{B, M, N};
    std::vector<int64_t> scale_shape = is_2d ? std::vector<int64_t>{m_blocks, n_blocks}
                                             : std::vector<int64_t>{B, m_blocks, n_blocks};
    auto                 output      = at::empty(out_shape, input.options().dtype(dest_dtype));
    auto                 scale_inv   = at::empty(scale_shape, input.options().dtype(at::kFloat));

    auto        stream  = at::cuda::getCurrentCUDAStream();
    const float fp8_max = get_float8_max(dest_dtype);
    TORCH_TYPE_SWITCH_FP16_BF16(input.scalar_type(), FType, {
        TORCH_TYPE_SWITCH_FP8(output.scalar_type(), QType, {
            quantize_blockwise_for_weight_impl<FType, QType>(
                reinterpret_cast<const FType *>(input.data_ptr()),
                reinterpret_cast<QType *>(output.data_ptr()),
                reinterpret_cast<float *>(scale_inv.data_ptr()), B, M, N, fp8_max, stream);
        });
    });
    return {output, scale_inv};
}

} // namespace primus_turbo::pytorch
