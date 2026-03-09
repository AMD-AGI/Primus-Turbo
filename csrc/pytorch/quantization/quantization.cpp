// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
//
// See LICENSE for license information.

#include "primus_turbo/quantization.h"
#include "primus_turbo/reduce.h"
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

// Quantize MXFP4 Dual with scale and output shuffle
std::vector<at::Tensor> quantize_mxfp4_dual_shuffle(
    const at::Tensor input, const at::ScalarType dest_dtype, const bool shuffle_rowwise_scale,
    const bool shuffle_rowwise_output, const bool rowwise_use_2d_block, const bool rowwise_use_sr,
    const bool rowwise_use_rht, const bool shuffle_colwise_scale, const bool shuffle_colwise_output,
    const bool colwise_use_2d_block, const bool colwise_use_sr, const bool colwise_use_rht) {
    using namespace primus_turbo::detail;

    std::function<int64_t(int64_t, int64_t)> cdiv = [](int64_t a, int64_t b) -> int64_t {
        return (a + b - 1) / b;
    };

    PRIMUS_TURBO_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    PRIMUS_TURBO_CHECK(input.scalar_type() == at::kBFloat16 || input.scalar_type() == at::kHalf,
                       "Input must be BFloat16 or Half");
    PRIMUS_TURBO_CHECK(input.dim() == 2, "Input must be 2D");
    PRIMUS_TURBO_CHECK(input.is_contiguous(), "Input must be contiguous");
    PRIMUS_TURBO_CHECK(dest_dtype == at::kFloat4_e2m1fn_x2, "Output must be Float4_e2m1fn_x2");

    const int64_t M = input.size(0);
    const int64_t N = input.size(1);

    PRIMUS_TURBO_CHECK(N % MXFP4_BLOCK_SIZE == 0, "N must be divisible by 32");

    if (shuffle_rowwise_output) {
        PRIMUS_TURBO_CHECK(M % MXFP4_SHUFFLE_BN == 0,
                           "M must be divisible by 16 for shuffled rowwise FP4");
        PRIMUS_TURBO_CHECK((N / 2) % MXFP4_SHUFFLE_BK == 0,
                           "N/2 must be divisible by 32 for shuffled rowwise FP4");
    }
    if (shuffle_colwise_output) {
        PRIMUS_TURBO_CHECK(N % MXFP4_SHUFFLE_BN == 0,
                           "N must be divisible by 16 for shuffled colwise FP4");
        PRIMUS_TURBO_CHECK((M / 2) % MXFP4_SHUFFLE_BK == 0,
                           "M/2 must be divisible by 32 for shuffled colwise FP4");
    }

    auto device = input.device();
    auto stream = at::cuda::getCurrentCUDAStream();

    int64_t rowwise_scale_M_pad = cdiv(M, 256) * 256;
    int64_t rowwise_scale_N     = cdiv(N, MXFP4_BLOCK_SIZE);
    int64_t rowwise_scale_N_pad = cdiv(rowwise_scale_N, 8) * 8;

    int64_t    rowwise_scale_stride = 1;
    at::Tensor rowwise_scale;
    if (shuffle_rowwise_scale) {
        rowwise_scale        = at::empty({rowwise_scale_M_pad, rowwise_scale_N_pad},
                                         at::TensorOptions().dtype(at::kByte).device(device));
        rowwise_scale_stride = rowwise_scale.stride(0);
    } else {
        rowwise_scale =
            at::empty({M, rowwise_scale_N}, at::TensorOptions().dtype(at::kByte).device(device));
    }

    // packed 2 fp4 values in N dimension
    at::Tensor rowwise_output =
        at::empty({M, N / 2}, at::TensorOptions().dtype(at::kByte).device(device));

    int64_t colwise_scale_M_pad = cdiv(N, 256) * 256;
    int64_t colwise_scale_N     = cdiv(M, MXFP4_BLOCK_SIZE);
    int64_t colwise_scale_N_pad = cdiv(colwise_scale_N, 8) * 8;

    at::Tensor colwise_scale;
    int        colwise_scale_stride = 1;
    if (shuffle_colwise_scale) {
        colwise_scale        = at::empty({colwise_scale_M_pad, colwise_scale_N_pad},
                                         at::TensorOptions().dtype(at::kByte).device(device));
        colwise_scale_stride = colwise_scale.stride(0);
    } else {
        colwise_scale =
            at::empty({N, colwise_scale_N}, at::TensorOptions().dtype(at::kByte).device(device));
    }

    // packed 2 fp4 values in N dimension
    at::Tensor colwise_output =
        at::empty({N, M / 2}, at::TensorOptions().dtype(at::kByte).device(device));

    TORCH_TYPE_SWITCH_FP16_BF16(input.scalar_type(), DType, {
        quantize_mxfp4_dual_shuffle_impl<DType>(
            reinterpret_cast<DType *>(input.data_ptr()),
            reinterpret_cast<dtype::float4x2_e2m1 *>(rowwise_output.data_ptr()),
            rowwise_scale.data_ptr<uint8_t>(),
            reinterpret_cast<dtype::float4x2_e2m1 *>(colwise_output.data_ptr()),
            colwise_scale.data_ptr<uint8_t>(), M, N, rowwise_scale_stride, colwise_scale_stride,
            rowwise_scale_N, rowwise_scale_M_pad, rowwise_scale_N_pad, N, colwise_scale_N,
            colwise_scale_M_pad, colwise_scale_N_pad,
            MXScalingRecipe(rowwise_use_2d_block, rowwise_use_sr, rowwise_use_rht,
                            shuffle_rowwise_scale, shuffle_rowwise_output),
            MXScalingRecipe(colwise_use_2d_block, colwise_use_sr, colwise_use_rht,
                            shuffle_colwise_scale, shuffle_colwise_output),
            stream);
    });

    return {rowwise_output.view(at::kFloat4_e2m1fn_x2), rowwise_scale.view(at::kFloat8_e8m0fnu),
            colwise_output.view(at::kFloat4_e2m1fn_x2), colwise_scale.view(at::kFloat8_e8m0fnu)};
}

} // namespace primus_turbo::pytorch
