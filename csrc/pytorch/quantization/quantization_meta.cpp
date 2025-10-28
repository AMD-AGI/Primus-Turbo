// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
//
// See LICENSE for license information.

#include "pytorch/extensions.h"

namespace primus_turbo::pytorch {

std::vector<at::Tensor> quantize_fp8_tensorwise_meta(const at::Tensor          input,
                                                     const at::ScalarType      dest_dtype,
                                                     c10::optional<at::Tensor> scale_opt) {
    auto input_fp8 = at::empty_like(input, at::dtype(dest_dtype).device(at::kMeta));
    auto scale_inv = at::empty({}, input.options().dtype(at::kFloat).device(at::kMeta));
    return {input_fp8, scale_inv};
}

std::vector<at::Tensor> quantize_fp8_rowwise_meta(const at::Tensor          input,
                                                  const at::ScalarType      dest_dtype,
                                                  const int64_t             axis,
                                                  c10::optional<at::Tensor> scale_opt) {
    const int64_t valid_axis = (axis >= 0) ? axis : input.dim() + axis;
    PRIMUS_TURBO_CHECK(valid_axis >= 0 && valid_axis < input.dim());
    auto input_fp8 = at::empty_like(input, at::dtype(dest_dtype).device(at::kMeta));

    std::vector<int64_t> scale_inv_shape(input.sizes().begin(), input.sizes().end());
    scale_inv_shape[valid_axis] = 1;
    auto scale_inv =
        at::empty(scale_inv_shape, input.options().dtype(at::kFloat).device(at::kMeta));
    return {input_fp8, scale_inv};
}

at::Tensor dequantize_fp8_tensorwise_meta(const at::Tensor input, const at::Tensor scale_inv,
                                          const at::ScalarType dest_dtype) {
    at::Tensor output = at::empty_like(input, at::dtype(dest_dtype).device(at::kMeta));
    return output;
}

} // namespace primus_turbo::pytorch
