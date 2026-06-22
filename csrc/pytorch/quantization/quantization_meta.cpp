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

at::Tensor dequantize_mxfp8_meta(const at::Tensor input, const at::Tensor scale_inv,
                                 const int64_t axis, const int64_t block_size,
                                 const at::ScalarType dest_dtype) {
    PRIMUS_TURBO_CHECK(input.dim() == 2, "Input must be 2D");
    PRIMUS_TURBO_CHECK(axis == 0 || axis == 1, "Axis must be 0 or 1");
    const int64_t num_rows   = input.size(0);
    const int64_t row_length = input.size(1);
    at::Tensor    output =
        (axis == 1) ? at::empty({num_rows, row_length}, at::dtype(dest_dtype).device(at::kMeta))
                       : at::empty({row_length, num_rows}, at::dtype(dest_dtype).device(at::kMeta));
    return output;
}

at::Tensor dequantize_mxfp4_meta(const at::Tensor input, const at::Tensor scale_inv,
                                 const int64_t axis, const int64_t block_size,
                                 const at::ScalarType dest_dtype) {
    PRIMUS_TURBO_CHECK(input.dim() == 2, "Input must be 2D");
    PRIMUS_TURBO_CHECK(axis == 0 || axis == 1, "Axis must be 0 or 1");
    const int64_t num_rows = input.size(0);
    // ``input`` packs 2 FP4 values per byte in the last dim.
    const int64_t row_length = input.size(1) * 2;
    at::Tensor    output =
        (axis == 1) ? at::empty({num_rows, row_length}, at::dtype(dest_dtype).device(at::kMeta))
                       : at::empty({row_length, num_rows}, at::dtype(dest_dtype).device(at::kMeta));
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
    PRIMUS_TURBO_CHECK(padding_align_size == MXFP4_K_DIM_PADDING_ALIGN_SIZE,
                       "padding_align_size must be ", MXFP4_K_DIM_PADDING_ALIGN_SIZE,
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
    PRIMUS_TURBO_CHECK(padding_align_size == MXFP4_K_DIM_PADDING_ALIGN_SIZE,
                       "padding_align_size must be ", MXFP4_K_DIM_PADDING_ALIGN_SIZE,
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

std::vector<at::Tensor> quantize_mxfp8_dual_meta(
    const at::Tensor input, const at::ScalarType dest_dtype, const int64_t padding_align_size,
    const bool rowwise_use_2d_block, const bool colwise_use_2d_block,
    const bool shuffle_rowwise_scale, const bool shuffle_rowwise, const bool shuffle_colwise_scale,
    const bool shuffle_colwise, const int64_t preshuffle_layout, const int64_t preshuffle_n_tiles,
    const int64_t col_preshuffle_layout, const int64_t col_preshuffle_n_tiles) {
    using namespace primus_turbo::detail;

    std::function<int64_t(int64_t, int64_t)> cdiv = [](int64_t a, int64_t b) -> int64_t {
        return (a + b - 1) / b;
    };

    PRIMUS_TURBO_CHECK(input.scalar_type() == at::kBFloat16 || input.scalar_type() == at::kHalf,
                       "Input must be BFloat16 or Half");
    PRIMUS_TURBO_CHECK(input.is_contiguous(), "Input must be contiguous");
    PRIMUS_TURBO_CHECK(dest_dtype == at::kFloat8_e4m3fn || dest_dtype == at::kFloat8_e5m2,
                       "Output must be Float8_e4m3fn or Float8_e5m2");
    // Guard the public op argument against zero/negative values (would otherwise
    // divide-by-zero in cdiv below) and lock it to the expected MXFP8 constant.
    PRIMUS_TURBO_CHECK(padding_align_size == MXFP8_K_DIM_PADDING_ALIGN_SIZE,
                       "padding_align_size must be ", MXFP8_K_DIM_PADDING_ALIGN_SIZE,
                       " for MXFP8. But got padding_align_size=", padding_align_size);

    // Mirror the CUDA impl: 2D keeps ``G == 1`` and the 2D layout; 3D carries a
    // leading group dim ``G`` on every per-group buffer and returns 3D views.
    int64_t G, M, N;
    if (input.dim() == 2) {
        G = 1;
        M = input.size(0);
        N = input.size(1);
    } else if (input.dim() == 3) {
        G = input.size(0);
        M = input.size(1);
        N = input.size(2);
    } else {
        PRIMUS_TURBO_ERROR("Input must be 2D or 3D");
    }

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

    const bool    is_batched = (input.dim() == 3);
    const int64_t Gout       = is_batched ? G : 1;

    int64_t rowwise_scale_M_pad = cdiv(M, 256) * 256;
    int64_t rowwise_scale_N     = cdiv(N_pad, MXFP8_BLOCK_SIZE);
    int64_t rowwise_scale_N_pad = cdiv(rowwise_scale_N, 8) * 8;

    at::Tensor rowwise_scale;
    if (shuffle_rowwise_scale) {
        rowwise_scale = at::empty({Gout * rowwise_scale_M_pad, rowwise_scale_N_pad},
                                  at::TensorOptions().dtype(at::kByte).device(at::kMeta));
    } else {
        rowwise_scale = at::empty({Gout * M, rowwise_scale_N},
                                  at::TensorOptions().dtype(at::kByte).device(at::kMeta));
    }

    at::Tensor rowwise_output =
        at::empty({Gout * M, N_pad}, at::TensorOptions().dtype(at::kByte).device(at::kMeta));

    int64_t colwise_scale_M_pad = cdiv(N, 256) * 256;
    int64_t colwise_scale_N     = cdiv(M_pad, MXFP8_BLOCK_SIZE);
    int64_t colwise_scale_N_pad = cdiv(colwise_scale_N, 8) * 8;

    at::Tensor colwise_scale;
    if (shuffle_colwise_scale) {
        colwise_scale = at::empty({Gout * colwise_scale_M_pad, colwise_scale_N_pad},
                                  at::TensorOptions().dtype(at::kByte).device(at::kMeta));
    } else {
        colwise_scale = at::empty({Gout * N, colwise_scale_N},
                                  at::TensorOptions().dtype(at::kByte).device(at::kMeta));
    }

    at::Tensor colwise_output =
        at::empty({Gout * N, M_pad}, at::TensorOptions().dtype(at::kByte).device(at::kMeta));

    if (is_batched) {
        const int64_t rowwise_scale_rows = shuffle_rowwise_scale ? rowwise_scale_M_pad : M;
        const int64_t colwise_scale_rows = shuffle_colwise_scale ? colwise_scale_M_pad : N;
        return {rowwise_output.view({G, M, N_pad}).view(dest_dtype),
                rowwise_scale.view({G, rowwise_scale_rows, -1}).view(at::kFloat8_e8m0fnu),
                colwise_output.view({G, N, M_pad}).view(dest_dtype),
                colwise_scale.view({G, colwise_scale_rows, -1}).view(at::kFloat8_e8m0fnu)};
    }

    return {rowwise_output.view(dest_dtype), rowwise_scale.view(at::kFloat8_e8m0fnu),
            colwise_output.view(dest_dtype), colwise_scale.view(at::kFloat8_e8m0fnu)};
}

std::vector<at::Tensor>
grouped_quantize_mxfp8_dual_meta(const at::Tensor input, const at::Tensor group_lens,
                                 const at::Tensor group_offs, const at::ScalarType dest_dtype,
                                 const bool rowwise_use_2d_block, const bool colwise_use_2d_block,
                                 const bool shuffle_rowwise_scale, const bool shuffle_rowwise,
                                 const bool shuffle_colwise_scale, const bool shuffle_colwise) {
    using namespace primus_turbo::detail;
    auto cdiv = [](int64_t a, int64_t b) -> int64_t { return (a + b - 1) / b; };

    PRIMUS_TURBO_CHECK(input.scalar_type() == at::kBFloat16 || input.scalar_type() == at::kHalf,
                       "Input must be BFloat16 or Half");
    PRIMUS_TURBO_CHECK(input.dim() == 2, "Input must be 2D");
    PRIMUS_TURBO_CHECK(dest_dtype == at::kFloat8_e4m3fn || dest_dtype == at::kFloat8_e5m2,
                       "Output must be Float8_e4m3fn or Float8_e5m2");

    constexpr int64_t ALIGN = MXFP8_K_DIM_PADDING_ALIGN_SIZE;

    const int64_t     G         = group_lens.size(0);
    const int64_t     total_M   = input.size(0);
    const int64_t     N         = input.size(1);
    constexpr int64_t ROW_ALIGN = MXFP8_GROUP_M_PADDING_ALIGN_SIZE; // = 32
    const int64_t     M_pad_col = cdiv(total_M + G * ALIGN, ALIGN) * ALIGN;
    const int64_t     M_pad_row = cdiv(total_M + G * ROW_ALIGN, ROW_ALIGN) * ROW_ALIGN;
    const int64_t     N_pad     = cdiv(N, ALIGN) * ALIGN;

    int64_t rowwise_scale_M_pad = cdiv(M_pad_row, 256) * 256;
    int64_t rowwise_scale_N     = cdiv(N_pad, MXFP8_BLOCK_SIZE);
    int64_t rowwise_scale_N_pad = cdiv(rowwise_scale_N, 8) * 8;

    at::Tensor rowwise_scale;
    if (shuffle_rowwise_scale) {
        rowwise_scale = at::empty({rowwise_scale_M_pad, rowwise_scale_N_pad},
                                  at::TensorOptions().dtype(at::kByte).device(at::kMeta));
    } else {
        rowwise_scale = at::empty({M_pad_row, rowwise_scale_N},
                                  at::TensorOptions().dtype(at::kByte).device(at::kMeta));
    }

    at::Tensor rowwise_output =
        at::empty({M_pad_row, N_pad}, at::TensorOptions().dtype(at::kByte).device(at::kMeta));

    int64_t colwise_scale_M_pad = cdiv(N, 256) * 256;
    int64_t colwise_scale_N     = cdiv(M_pad_col, MXFP8_BLOCK_SIZE);
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
        at::empty({N, M_pad_col}, at::TensorOptions().dtype(at::kByte).device(at::kMeta));

    auto group_lens_padded_colwise =
        at::empty_like(group_lens, group_lens.options().device(at::kMeta));
    auto group_offs_padded_colwise = at::empty({G + 1}, group_offs.options().device(at::kMeta));
    auto group_lens_padded_rowwise =
        at::empty_like(group_lens, group_lens.options().device(at::kMeta));
    auto group_offs_padded_rowwise = at::empty({G + 1}, group_offs.options().device(at::kMeta));

    return {rowwise_output.view(dest_dtype), rowwise_scale.view(at::kFloat8_e8m0fnu),
            colwise_output.view(dest_dtype), colwise_scale.view(at::kFloat8_e8m0fnu),
            group_lens_padded_rowwise,       group_offs_padded_rowwise,
            group_lens_padded_colwise,       group_offs_padded_colwise};
}

std::vector<at::Tensor> quantize_mxfp8_meta(const at::Tensor input, const at::ScalarType dest_dtype,
                                            const int64_t axis, const int64_t padding_align_size,
                                            const bool use_2d_block, const bool shuffle_scale,
                                            const bool shuffle_out) {
    using namespace primus_turbo::detail;

    auto cdiv = [](int64_t a, int64_t b) -> int64_t { return (a + b - 1) / b; };

    PRIMUS_TURBO_CHECK(input.scalar_type() == at::kBFloat16 || input.scalar_type() == at::kHalf,
                       "Input must be BFloat16 or Half");
    PRIMUS_TURBO_CHECK(dest_dtype == at::kFloat8_e4m3fn || dest_dtype == at::kFloat8_e5m2,
                       "Output must be Float8_e4m3fn or Float8_e5m2");
    PRIMUS_TURBO_CHECK(axis == 0 || axis == 1, "Axis must be 0 or 1");
    PRIMUS_TURBO_CHECK(input.is_contiguous(), "Input must be contiguous");
    // Guard the public op argument against zero/negative values (would otherwise
    // divide-by-zero in cdiv below) and lock it to the expected MXFP8 constant.
    PRIMUS_TURBO_CHECK(padding_align_size == MXFP8_K_DIM_PADDING_ALIGN_SIZE,
                       "padding_align_size must be ", MXFP8_K_DIM_PADDING_ALIGN_SIZE,
                       " for MXFP8. But got padding_align_size=", padding_align_size);

    // Mirror the CUDA impl: ``axis`` is locked to {0, 1}. 2D maps axis==0 ->
    // COLWISE / axis==1 -> ROWWISE; 3D maps axis==1 -> COLWISE, otherwise
    // ROWWISE, with a leading group dim ``G`` and 3D views on the output.
    bool    is_rowwise;
    int64_t G, M, N;
    if (input.dim() == 2) {
        is_rowwise = (axis != 0); // axis==0 -> COLWISE, axis==1 -> ROWWISE
        G          = 1;
        M          = input.size(0);
        N          = input.size(1);
    } else if (input.dim() == 3) {
        is_rowwise = (axis != 1); // axis==1 -> COLWISE, otherwise ROWWISE
        G          = input.size(0);
        M          = input.size(1);
        N          = input.size(2);
    } else {
        PRIMUS_TURBO_ERROR("Input must be 2D or 3D");
    }

    const int64_t M_pad = cdiv(M, padding_align_size) * padding_align_size;
    const int64_t N_pad = cdiv(N, padding_align_size) * padding_align_size;

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

    const bool    is_batched = (input.dim() == 3);
    const int64_t Gout       = is_batched ? G : 1;

    int64_t scale_outer = is_rowwise ? M : N;
    int64_t scale_N = is_rowwise ? cdiv(N_pad, MXFP8_BLOCK_SIZE) : cdiv(M_pad, MXFP8_BLOCK_SIZE);
    int64_t scale_M_pad = cdiv(scale_outer, 256) * 256;
    int64_t scale_N_pad = cdiv(scale_N, 8) * 8;

    at::Tensor scale_tensor;
    if (shuffle_scale) {
        scale_tensor = at::empty({Gout * scale_M_pad, scale_N_pad},
                                 at::TensorOptions().dtype(at::kByte).device(at::kMeta));
    } else {
        scale_tensor = at::empty({Gout * scale_outer, scale_N},
                                 at::TensorOptions().dtype(at::kByte).device(at::kMeta));
    }

    int64_t    output_rows = is_rowwise ? M : N;
    int64_t    output_cols = is_rowwise ? N_pad : M_pad;
    at::Tensor output      = at::empty({Gout * output_rows, output_cols},
                                       at::TensorOptions().dtype(at::kByte).device(at::kMeta));

    if (is_batched) {
        const int64_t scale_rows = shuffle_scale ? scale_M_pad : scale_outer;
        return {output.view({G, output_rows, output_cols}).view(dest_dtype),
                scale_tensor.view({G, scale_rows, -1}).view(at::kFloat8_e8m0fnu)};
    }

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
