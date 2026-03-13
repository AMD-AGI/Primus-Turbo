// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
//
// See LICENSE for license information.

#include "primus_turbo/quantization.h"
#include "primus_turbo/shuffle.h"
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

std::vector<at::Tensor> quantize_mxfp4_dual_meta(
    const at::Tensor input, const at::ScalarType dest_dtype, const bool rowwise_use_2d_block,
    const bool rowwise_use_sr, const bool rowwise_use_rht, const bool colwise_use_2d_block,
    const bool colwise_use_sr, const bool colwise_use_rht, const bool shuffle_rowwise_scale,
    const bool shuffle_rowwise, const bool shuffle_colwise_scale, const bool shuffle_colwise) {
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

    const int64_t M_pad = cdiv(M, MXFP4_PADDING_ALIGN_SIZE) * MXFP4_PADDING_ALIGN_SIZE;
    const int64_t N_pad = cdiv(N, MXFP4_PADDING_ALIGN_SIZE) * MXFP4_PADDING_ALIGN_SIZE;

    PRIMUS_TURBO_CHECK(N % MXFP4_BLOCK_SIZE == 0, "N must be divisible by 32");

    if (shuffle_rowwise) {
        PRIMUS_TURBO_CHECK(M % MXFP4_SHUFFLE_BN == 0,
                           "M must be divisible by 16 for shuffled rowwise FP4");
        PRIMUS_TURBO_CHECK((N / 2) % MXFP4_SHUFFLE_BK == 0,
                           "N/2 must be divisible by 32 for shuffled rowwise FP4");
    }
    if (shuffle_colwise) {
        PRIMUS_TURBO_CHECK(N % MXFP4_SHUFFLE_BN == 0,
                           "N must be divisible by 16 for shuffled colwise FP4");
        PRIMUS_TURBO_CHECK((M / 2) % MXFP4_SHUFFLE_BK == 0,
                           "M/2 must be divisible by 32 for shuffled colwise FP4");
    }

    int64_t rowwise_scale_M_pad = cdiv(M, 256) * 256;
    int64_t rowwise_scale_N     = cdiv(N_pad, MXFP4_BLOCK_SIZE);
    int64_t rowwise_scale_N_pad = cdiv(rowwise_scale_N, 8) * 8;

    at::Tensor rowwise_scale;
    if (shuffle_rowwise_scale) {
        rowwise_scale = at::empty({rowwise_scale_M_pad, rowwise_scale_N_pad},
                                  at::TensorOptions().dtype(at::kByte).device(at::kMeta));
    } else {
        rowwise_scale =
            at::empty({M, rowwise_scale_N}, at::TensorOptions().dtype(at::kByte).device(at::kMeta));
    }

    // packed 2 fp4 values in N dimension
    at::Tensor rowwise_output =
        at::empty({M, N_pad / 2}, at::TensorOptions().dtype(at::kByte).device(at::kMeta));

    int64_t colwise_scale_M_pad = cdiv(N, 256) * 256;
    int64_t colwise_scale_N     = cdiv(M_pad, MXFP4_BLOCK_SIZE);
    int64_t colwise_scale_N_pad = cdiv(colwise_scale_N, 8) * 8;

    at::Tensor colwise_scale;
    if (shuffle_colwise_scale) {
        colwise_scale = at::empty({colwise_scale_M_pad, colwise_scale_N_pad},
                                  at::TensorOptions().dtype(at::kByte).device(at::kMeta));
    } else {
        colwise_scale =
            at::empty({N, colwise_scale_N}, at::TensorOptions().dtype(at::kByte).device(at::kMeta));
    }

    // packed 2 fp4 values in N dimension
    at::Tensor colwise_output =
        at::empty({N, M_pad / 2}, at::TensorOptions().dtype(at::kByte).device(at::kMeta));

    return {rowwise_output, rowwise_scale, colwise_output, colwise_scale};
}

} // namespace primus_turbo::pytorch
