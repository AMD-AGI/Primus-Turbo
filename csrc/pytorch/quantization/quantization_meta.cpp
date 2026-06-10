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

at::Tensor dequantize_fp8_rowwise_meta(const at::Tensor input, const at::Tensor scale_inv,
                                       const int64_t axis, const at::ScalarType dest_dtype) {
    const int64_t valid_axis = (axis >= 0) ? axis : input.dim() + axis;
    PRIMUS_TURBO_CHECK(valid_axis >= 0 && valid_axis < input.dim());
    at::Tensor output = at::empty_like(input, at::dtype(dest_dtype).device(at::kMeta));
    return output;
}

std::vector<at::Tensor> quantize_mxfp4_dual_meta(
    const at::Tensor input, const at::ScalarType dest_dtype, const int64_t padding_align_size,
    const bool rowwise_use_2d_block, const bool rowwise_use_sr, const bool rowwise_use_rht,
    const bool colwise_use_2d_block, const bool colwise_use_sr, const bool colwise_use_rht,
    const bool shuffle_rowwise_scale, const bool shuffle_rowwise, const bool shuffle_colwise_scale,
    const bool shuffle_colwise) {
    using namespace primus_turbo::detail;

    std::function<int64_t(int64_t, int64_t)> cdiv = [](int64_t a, int64_t b) -> int64_t {
        return (a + b - 1) / b;
    };

    PRIMUS_TURBO_CHECK(input.scalar_type() == at::kBFloat16 || input.scalar_type() == at::kHalf,
                       "Input must be BFloat16 or Half");
    PRIMUS_TURBO_CHECK(input.dim() == 2, "Input must be 2D");
    PRIMUS_TURBO_CHECK(input.is_contiguous(), "Input must be contiguous");
    PRIMUS_TURBO_CHECK(dest_dtype == at::kFloat4_e2m1fn_x2, "Output must be Float4_e2m1fn_x2");
    // Guard the public op argument against zero/negative values (would otherwise
    // divide-by-zero in cdiv below) and lock it to the expected MXFP4 constant.
    PRIMUS_TURBO_CHECK(padding_align_size == MXFP4_PADDING_ALIGN_SIZE,
                       "padding_align_size must be ", MXFP4_PADDING_ALIGN_SIZE,
                       " for MXFP4. But got padding_align_size=", padding_align_size);

    const int64_t M = input.size(0);
    const int64_t N = input.size(1);

    const int64_t M_pad = cdiv(M, padding_align_size) * padding_align_size;
    const int64_t N_pad = cdiv(N, padding_align_size) * padding_align_size;

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

    return {rowwise_output.view(at::kFloat4_e2m1fn_x2), rowwise_scale.view(at::kFloat8_e8m0fnu),
            colwise_output.view(at::kFloat4_e2m1fn_x2), colwise_scale.view(at::kFloat8_e8m0fnu)};
}

std::vector<at::Tensor> quantize_mxfp4_meta(const at::Tensor input, const at::ScalarType dest_dtype,
                                            const int64_t axis, const int64_t padding_align_size,
                                            const bool use_2d_block, const bool use_sr,
                                            const bool use_rht, const bool shuffle_scale,
                                            const bool shuffle_out) {
    using namespace primus_turbo::detail;

    auto cdiv = [](int64_t a, int64_t b) -> int64_t { return (a + b - 1) / b; };

    PRIMUS_TURBO_CHECK(input.scalar_type() == at::kBFloat16 || input.scalar_type() == at::kHalf,
                       "Input must be BFloat16 or Half");
    PRIMUS_TURBO_CHECK(input.dim() == 2, "Input must be 2D");
    PRIMUS_TURBO_CHECK(dest_dtype == at::kFloat4_e2m1fn_x2, "Output must be Float4_e2m1fn_x2");
    PRIMUS_TURBO_CHECK(axis == 0 || axis == 1, "Axis must be 0 or 1");
    // Guard the public op argument against zero/negative values (would otherwise
    // divide-by-zero in cdiv below) and lock it to the expected MXFP4 constant.
    PRIMUS_TURBO_CHECK(padding_align_size == MXFP4_PADDING_ALIGN_SIZE,
                       "padding_align_size must be ", MXFP4_PADDING_ALIGN_SIZE,
                       " for MXFP4. But got padding_align_size=", padding_align_size);

    const bool    is_rowwise = (axis == 1);
    const int64_t M          = input.size(0);
    const int64_t N          = input.size(1);
    const int64_t M_pad      = cdiv(M, padding_align_size) * padding_align_size;
    const int64_t N_pad      = cdiv(N, padding_align_size) * padding_align_size;

    int64_t scale_outer = is_rowwise ? M : N;
    int64_t scale_N = is_rowwise ? cdiv(N_pad, MXFP4_BLOCK_SIZE) : cdiv(M_pad, MXFP4_BLOCK_SIZE);
    int64_t scale_M_pad = cdiv(scale_outer, 256) * 256;
    int64_t scale_N_pad = cdiv(scale_N, 8) * 8;

    at::Tensor scale_tensor;
    if (shuffle_scale) {
        scale_tensor = at::empty({scale_M_pad, scale_N_pad},
                                 at::TensorOptions().dtype(at::kByte).device(at::kMeta));
    } else {
        scale_tensor = at::empty({scale_outer, scale_N},
                                 at::TensorOptions().dtype(at::kByte).device(at::kMeta));
    }

    int64_t    output_rows = is_rowwise ? M : N;
    int64_t    output_cols = is_rowwise ? (N_pad / 2) : (M_pad / 2);
    at::Tensor output      = at::empty({output_rows, output_cols},
                                       at::TensorOptions().dtype(at::kByte).device(at::kMeta));

    return {output.view(at::kFloat4_e2m1fn_x2), scale_tensor.view(at::kFloat8_e8m0fnu)};
}

std::vector<at::Tensor>
quantize_mxfp8_dual_meta(const at::Tensor input, const at::ScalarType dest_dtype,
                         const int64_t padding_align_size, const bool rowwise_use_2d_block,
                         const bool colwise_use_2d_block, const bool shuffle_rowwise_scale,
                         const bool shuffle_rowwise, const bool shuffle_colwise_scale,
                         const bool shuffle_colwise) {
    using namespace primus_turbo::detail;

    std::function<int64_t(int64_t, int64_t)> cdiv = [](int64_t a, int64_t b) -> int64_t {
        return (a + b - 1) / b;
    };

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
                           " for shuffled rowwise FP8");
    }
    if (shuffle_colwise) {
        PRIMUS_TURBO_CHECK(N % MXFP8_SHUFFLE_BN == 0, "N must be divisible by ", MXFP8_SHUFFLE_BN,
                           " for shuffled colwise FP8");
    }

    int64_t rowwise_scale_M_pad = cdiv(M, 256) * 256;
    int64_t rowwise_scale_N     = cdiv(N_pad, MXFP8_BLOCK_SIZE);
    int64_t rowwise_scale_N_pad = cdiv(rowwise_scale_N, 8) * 8;

    at::Tensor rowwise_scale;
    if (shuffle_rowwise_scale) {
        rowwise_scale = at::empty({rowwise_scale_M_pad, rowwise_scale_N_pad},
                                  at::TensorOptions().dtype(at::kByte).device(at::kMeta));
    } else {
        rowwise_scale =
            at::empty({M, rowwise_scale_N}, at::TensorOptions().dtype(at::kByte).device(at::kMeta));
    }

    at::Tensor rowwise_output =
        at::empty({M, N_pad}, at::TensorOptions().dtype(at::kByte).device(at::kMeta));

    int64_t colwise_scale_M_pad = cdiv(N, 256) * 256;
    int64_t colwise_scale_N     = cdiv(M_pad, MXFP8_BLOCK_SIZE);
    int64_t colwise_scale_N_pad = cdiv(colwise_scale_N, 8) * 8;

    at::Tensor colwise_scale;
    if (shuffle_colwise_scale) {
        colwise_scale = at::empty({colwise_scale_M_pad, colwise_scale_N_pad},
                                  at::TensorOptions().dtype(at::kByte).device(at::kMeta));
    } else {
        colwise_scale =
            at::empty({N, colwise_scale_N}, at::TensorOptions().dtype(at::kByte).device(at::kMeta));
    }

    at::Tensor colwise_output =
        at::empty({N, M_pad}, at::TensorOptions().dtype(at::kByte).device(at::kMeta));

    return {rowwise_output.view(dest_dtype), rowwise_scale.view(at::kFloat8_e8m0fnu),
            colwise_output.view(dest_dtype), colwise_scale.view(at::kFloat8_e8m0fnu)};
}

std::vector<at::Tensor> quantize_mxfp8_meta(const at::Tensor input, const at::ScalarType dest_dtype,
                                            const int64_t axis, const int64_t padding_align_size,
                                            const bool use_2d_block, const bool shuffle_scale,
                                            const bool shuffle_out) {
    using namespace primus_turbo::detail;

    auto cdiv = [](int64_t a, int64_t b) -> int64_t { return (a + b - 1) / b; };

    PRIMUS_TURBO_CHECK(input.scalar_type() == at::kBFloat16 || input.scalar_type() == at::kHalf,
                       "Input must be BFloat16 or Half");
    PRIMUS_TURBO_CHECK(input.dim() == 2, "Input must be 2D");
    PRIMUS_TURBO_CHECK(dest_dtype == at::kFloat8_e4m3fn || dest_dtype == at::kFloat8_e5m2,
                       "Output must be Float8_e4m3fn or Float8_e5m2");
    PRIMUS_TURBO_CHECK(axis == 0 || axis == 1, "Axis must be 0 or 1");
    PRIMUS_TURBO_CHECK(input.is_contiguous(), "Input must be contiguous");
    // Guard the public op argument against zero/negative values (would otherwise
    // divide-by-zero in cdiv below) and lock it to the expected MXFP8 constant.
    PRIMUS_TURBO_CHECK(padding_align_size == MXFP8_PADDING_ALIGN_SIZE,
                       "padding_align_size must be ", MXFP8_PADDING_ALIGN_SIZE,
                       " for MXFP8. But got padding_align_size=", padding_align_size);

    const bool    is_rowwise = (axis == 1);
    const int64_t M          = input.size(0);
    const int64_t N          = input.size(1);
    const int64_t M_pad      = cdiv(M, padding_align_size) * padding_align_size;
    const int64_t N_pad      = cdiv(N, padding_align_size) * padding_align_size;

    PRIMUS_TURBO_CHECK(N % MXFP8_BLOCK_SIZE == 0, "N must be divisible by ", MXFP8_BLOCK_SIZE);

    if (shuffle_out) {
        if (is_rowwise) {
            PRIMUS_TURBO_CHECK(M % MXFP8_SHUFFLE_BN == 0, "M must be divisible by ",
                               MXFP8_SHUFFLE_BN, " for shuffled rowwise FP8");
        } else {
            PRIMUS_TURBO_CHECK(N % MXFP8_SHUFFLE_BN == 0, "N must be divisible by ",
                               MXFP8_SHUFFLE_BN, " for shuffled colwise FP8");
        }
    }

    int64_t scale_outer = is_rowwise ? M : N;
    int64_t scale_N = is_rowwise ? cdiv(N_pad, MXFP8_BLOCK_SIZE) : cdiv(M_pad, MXFP8_BLOCK_SIZE);
    int64_t scale_M_pad = cdiv(scale_outer, 256) * 256;
    int64_t scale_N_pad = cdiv(scale_N, 8) * 8;

    at::Tensor scale_tensor;
    if (shuffle_scale) {
        scale_tensor = at::empty({scale_M_pad, scale_N_pad},
                                 at::TensorOptions().dtype(at::kByte).device(at::kMeta));
    } else {
        scale_tensor = at::empty({scale_outer, scale_N},
                                 at::TensorOptions().dtype(at::kByte).device(at::kMeta));
    }

    int64_t    output_rows = is_rowwise ? M : N;
    int64_t    output_cols = is_rowwise ? N_pad : M_pad;
    at::Tensor output      = at::empty({output_rows, output_cols},
                                       at::TensorOptions().dtype(at::kByte).device(at::kMeta));

    return {output.view(dest_dtype), scale_tensor.view(at::kFloat8_e8m0fnu)};
}

std::vector<at::Tensor> quantize_fp8_blockwise_segment_m_row_col_meta(
    const at::Tensor input, const at::ScalarType dest_dtype, const int64_t block_size,
    const at::Tensor group_lens, const at::Tensor group_offs) {
    const int64_t M            = input.size(0);
    const int64_t N            = input.size(1);
    const int64_t num_groups   = group_lens.size(0);
    const int64_t M_padded_max = M + num_groups * block_size;
    auto          fp8_meta     = at::dtype(dest_dtype).device(at::kMeta);
    auto          fp32_meta    = at::dtype(at::kFloat).device(at::kMeta);
    auto          i64_meta     = at::dtype(at::kLong).device(at::kMeta);
    return {
        at::empty({M, N}, fp8_meta),
        at::empty({M_padded_max, N}, fp8_meta),
        at::empty({(N + block_size - 1) / block_size, M}, fp32_meta), // pshuffled
        at::empty({(M_padded_max + block_size - 1) / block_size, N}, fp32_meta),
        at::empty({num_groups}, i64_meta),
        at::empty({num_groups + 1}, i64_meta),
    };
}

std::vector<at::Tensor> quantize_fp8_blockwise_for_weight_meta(const at::Tensor     input,
                                                               const at::ScalarType dest_dtype,
                                                               const int64_t        block_size) {
    PRIMUS_TURBO_CHECK(input.dim() == 2 || input.dim() == 3);
    const bool    is_2d     = (input.dim() == 2);
    const int64_t B         = is_2d ? 1 : input.size(0);
    const int64_t M         = is_2d ? input.size(0) : input.size(1);
    const int64_t N         = is_2d ? input.size(1) : input.size(2);
    const int64_t m_blocks  = (M + block_size - 1) / block_size;
    const int64_t n_blocks  = (N + block_size - 1) / block_size;
    auto          fp8_meta  = at::dtype(dest_dtype).device(at::kMeta);
    auto          fp32_meta = at::dtype(at::kFloat).device(at::kMeta);
    if (is_2d) {
        return {at::empty({M, N}, fp8_meta), at::empty({m_blocks, n_blocks}, fp32_meta)};
    }
    return {at::empty({B, M, N}, fp8_meta), at::empty({B, m_blocks, n_blocks}, fp32_meta)};
}

} // namespace primus_turbo::pytorch
