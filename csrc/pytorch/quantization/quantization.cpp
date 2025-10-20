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

std::vector<at::Tensor> quantize_fp8_rowwise(const at::Tensor     input,
                                             const at::ScalarType dest_dtype, const int64_t axis,
                                             c10::optional<at::Tensor> scale_opt) {
    PRIMUS_TURBO_CHECK(input.scalar_type() == at::kBFloat16 || input.scalar_type() == at::kHalf ||
                       input.scalar_type() == at::kFloat);
    PRIMUS_TURBO_CHECK(is_torch_fp8(dest_dtype));

    const int64_t valid_axis = (axis >= 0) ? axis : input.dim() + axis;
    PRIMUS_TURBO_CHECK(valid_axis >= 0 && valid_axis < input.dim());
    const bool is_row_major = valid_axis == (input.dim() - 1);

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
            // TODO: reduce-col + binary kernel
            auto output_scaled  = input * scale;
            auto output_clamped = at::clamp(output_scaled, -fp8_max, fp8_max);
            output              = output_clamped.to(dest_dtype);
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
            // TODO: reduce-col + binary kernel
            auto amax = input.abs().amax(valid_axis, true).to(at::kFloat);
            compute_scale_from_amax<float>(reinterpret_cast<const float *>(amax.data_ptr()),
                                           fp8_max, reinterpret_cast<float *>(scale.data_ptr()),
                                           reinterpret_cast<float *>(scale_inv.data_ptr()),
                                           amax.numel(), stream);
            auto output_scaled  = input * scale;
            auto output_clamped = at::clamp(output_scaled, -fp8_max, fp8_max);
            output              = output_clamped.to(dest_dtype);
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

} // namespace primus_turbo::pytorch
