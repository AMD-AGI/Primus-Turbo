// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
//
// See LICENSE for license information.

#include "../extensions.h"
#include "../utils.h"
#include "primus_turbo/reduce.h"

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

std::vector<at::Tensor> quantize_fp8_tensorwise(const at::Tensor     input,
                                                const at::ScalarType dest_dtype) {
    PRIMUS_TURBO_CHECK(input.scalar_type() == at::kBFloat16 || input.scalar_type() == at::kHalf ||
                       input.scalar_type() == at::kFloat);
    PRIMUS_TURBO_CHECK(is_torch_fp8(dest_dtype));
    auto stream = at::cuda::getCurrentCUDAStream();

    // Reduce
    auto          input_max = torch::empty({}, input.options().dtype(at::kFloat));
    const int64_t ws_size   = get_reduce_row_workspace_sizes<float>(1, input.numel());
    auto          workspace = torch::empty({ws_size}, input.options().dtype(at::kByte));
    TORCH_TYPE_SWITCH_FLOAT(input.scalar_type(), InT, {
        reduce_row<InT, float, float>(
            PrimusTurboReduceOp::REDUCE_ABS_MAX, reinterpret_cast<InT *>(input.data_ptr()),
            input_max.data_ptr<float>(), 1, input.numel(), ws_size, workspace.data_ptr(), stream);
    });

    input_max = input_max.clamp_min(1e-12f);
    // Compute Scale
    const float fp8_max   = get_float8_max(dest_dtype);
    auto        scale     = fp8_max / input_max;
    auto        scale_inv = 1.0f / scale;

    // Quantize
    // TODO: refactor
    auto output = fp8_quantize(input, scale, dest_dtype);
    return {output, scale_inv};
}

std::vector<at::Tensor> quantize_fp8_rowwise(const at::Tensor     input,
                                             const at::ScalarType dest_dtype, const int64_t axis) {
    PRIMUS_TURBO_CHECK(input.scalar_type() == at::kBFloat16 || input.scalar_type() == at::kHalf ||
                       input.scalar_type() == at::kFloat);
    PRIMUS_TURBO_CHECK(is_torch_fp8(dest_dtype));

    const float fp8_max = get_float8_max(dest_dtype);

    const int64_t valid_axis = (axis >= 0) ? axis : input.dim() + axis;
    PRIMUS_TURBO_CHECK(valid_axis >= 0 && valid_axis < input.dim());

    // TODO: Opt Reduce
    // ReduceMax
    auto x_max = input.abs().amax(valid_axis, true).to(at::kFloat);
    x_max      = at::clamp(x_max, 1e-8f, std::numeric_limits<float>::infinity());
    // Compute Scale
    auto scale     = fp8_max / x_max;
    auto scale_inv = 1.0f / scale;

    // Quantize
    auto x_scaled  = input * scale;
    auto x_clamped = at::clamp(x_scaled, -fp8_max, fp8_max);
    auto x_fp8     = x_clamped.to(dest_dtype);
    return {x_fp8, scale_inv};
}

} // namespace primus_turbo::pytorch
