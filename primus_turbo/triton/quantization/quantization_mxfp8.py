###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################


import triton
import triton.language as tl

FP32_EXPONENT_BIAS = tl.constexpr(127)
FP32_MANTISSA_BITS = tl.constexpr(23)


@triton.jit
def exp2f_rcp_triton(biased_exp: tl.uint8) -> tl.float32:
    biased_exp_f32 = biased_exp.to(tl.float32)
    exp_val = FP32_EXPONENT_BIAS - biased_exp_f32
    result = tl.exp2(exp_val)
    final_result = tl.where(biased_exp == 0, 1.0, result)
    return final_result


@triton.jit
def float_to_e8m0_triton(val: tl.float32) -> tl.uint8:
    is_nan = val != val
    is_inf = tl.abs(val) == float("inf")
    is_zero = val == 0.0

    result_e8m0 = tl.zeros(val.shape, dtype=tl.uint8)  # Placeholder
    val_u32 = tl.cast(val, tl.uint32, bitcast=True)

    # Extract exponent and mantissa
    exponent_raw = (val_u32 >> FP32_MANTISSA_BITS) & 0xFF
    mantissa = val_u32 & 0x7FFFFF

    # Round up exponent and deal with satfinite.
    # (mantissa > 0 && exponent != 0xFE) && !(exponent == 0 && mantissa <= 0x400000)
    cond1 = mantissa > 0
    cond2 = exponent_raw != 0xFE
    cond3_part1 = exponent_raw == 0
    cond3_part2 = mantissa <= 0x400000
    cond3 = cond3_part1 & cond3_part2

    round_up_condition = (cond1 & cond2) & ~cond3

    # Increment exponent if the condition is true
    calculated_exponent = tl.where(round_up_condition, exponent_raw + 1, exponent_raw)

    # Priority: NaN -> Inf -> Zero -> Calculated Exponent
    result_e8m0 = tl.where(is_nan, tl.full(val.shape, 0xFF, dtype=tl.uint8), result_e8m0)
    result_e8m0 = tl.where(~is_nan & is_inf, tl.full(val.shape, 0xFE, dtype=tl.uint8), result_e8m0)
    result_e8m0 = tl.where(~is_nan & ~is_inf & is_zero, tl.full(val.shape, 0x00, dtype=tl.uint8), result_e8m0)
    result_e8m0 = tl.where(~is_nan & ~is_inf & ~is_zero, calculated_exponent, result_e8m0)

    return result_e8m0


@triton.jit
def quantize_mxfp8_kernel(
    x_ptr,
    y_ptr,
    stride_x_row,
    stride_x_col,
    stride_y_row,
    stride_y_col,
    n_rows,
    n_cols,
    padded_n_rows,
    padded_n_cols,
    scale_inv_ptr,
    stride_scale_inv_row,
    stride_scale_inv_col,
    scale_n_rows,
    scale_n_cols,
    max_fp8: tl.constexpr,
    BLOCK_X: tl.constexpr,
    BLOCK_Y: tl.constexpr,
    GROUP_Y: tl.constexpr,
    TRANS: tl.constexpr,
    MXFP8_BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)

    num_pid_along_Y = tl.cdiv(padded_n_rows, BLOCK_Y)
    num_pid_along_X = tl.cdiv(padded_n_cols, BLOCK_X)
    num_pid_in_group = GROUP_Y * num_pid_along_X

    group_id = pid // num_pid_in_group
    group_size = min(num_pid_along_Y - group_id * GROUP_Y, GROUP_Y)
    pid_m = group_id * GROUP_Y + ((pid % num_pid_in_group) % group_size)
    pid_n = (pid % num_pid_in_group) // group_size

    global_offset_Y_base = pid_m.to(tl.int64) * BLOCK_Y
    global_offset_X_base = pid_n.to(tl.int64) * BLOCK_X

    num_chunks_in_block_Y = BLOCK_Y // MXFP8_BLOCK_SIZE
    num_chunks_in_block_X = BLOCK_X // MXFP8_BLOCK_SIZE
    max_norm_rcp = 1.0 / max_fp8

    for chunk_id_y in range(0, num_chunks_in_block_Y):
        offsets_Y = global_offset_Y_base + chunk_id_y * MXFP8_BLOCK_SIZE + tl.arange(0, MXFP8_BLOCK_SIZE)
        for chunk_id_x in range(0, num_chunks_in_block_X):
            offsets_X = global_offset_X_base + chunk_id_x * MXFP8_BLOCK_SIZE + tl.arange(0, MXFP8_BLOCK_SIZE)
            x_ptr_current_chunk = (
                x_ptr + offsets_Y[:, None] * stride_x_row + offsets_X[None, :] * stride_x_col
            )
            load_mask = (offsets_Y < n_rows)[:, None] & (offsets_X < n_cols)[None, :]
            # (MXFP8_BLOCK_SIZE, MXFP8_BLOCK_SIZE)
            x_chunk = tl.load(x_ptr_current_chunk, mask=load_mask, other=0.0).to(tl.float32)

            if not TRANS:
                # Row-wise
                subwarp_amax_rowwise = tl.max(tl.abs(x_chunk), axis=-1, keep_dims=True)
                biased_exponent_rowwise = float_to_e8m0_triton(subwarp_amax_rowwise * max_norm_rcp)

                scale_offset_X = (pid_n * num_chunks_in_block_X) + chunk_id_x
                scale_inv_store_offsets = (
                    offsets_Y[:, None] * stride_scale_inv_row
                ) + scale_offset_X * stride_scale_inv_col
                scale_inv_store_mask = (offsets_Y < scale_n_rows)[:, None] & (scale_offset_X < scale_n_cols)
                tl.store(
                    scale_inv_ptr + scale_inv_store_offsets,
                    biased_exponent_rowwise,
                    mask=scale_inv_store_mask,
                )

                block_inverse_scale_rowwise = exp2f_rcp_triton(biased_exponent_rowwise)
                y_chunk_scaled = x_chunk * block_inverse_scale_rowwise

                store_mask = (offsets_Y < padded_n_rows)[:, None] & (offsets_X < padded_n_cols)[None, :]
                y_ptr_current_chunk = (
                    y_ptr + offsets_Y[:, None] * stride_y_row + offsets_X[None, :] * stride_y_col
                )
                tl.store(
                    y_ptr_current_chunk,
                    y_chunk_scaled.to(y_ptr.type.element_ty),
                    mask=store_mask,
                )
            else:
                # Col-wise
                subwarp_amax_colwise = tl.max(tl.abs(x_chunk), axis=0, keep_dims=True)
                biased_exponent_colwise = float_to_e8m0_triton(subwarp_amax_colwise * max_norm_rcp)

                scale_offset_Y = (pid_m * num_chunks_in_block_Y) + chunk_id_y
                scale_inv_store_offsets = scale_offset_Y * stride_scale_inv_col + (
                    offsets_X[None, :] * stride_scale_inv_row
                )
                scale_inv_store_mask = (scale_offset_Y < scale_n_rows) & (offsets_X < scale_n_cols)[None, :]
                tl.store(
                    scale_inv_ptr + scale_inv_store_offsets,
                    biased_exponent_colwise,
                    mask=scale_inv_store_mask,
                )

                block_inverse_scale_colwise = exp2f_rcp_triton(biased_exponent_colwise)
                y_chunk_scaled = x_chunk * block_inverse_scale_colwise
                store_mask = (offsets_Y < padded_n_rows)[:, None] & (offsets_X < padded_n_cols)[None, :]
                y_ptr_current_chunk = (
                    y_ptr + offsets_Y[:, None] * stride_y_col + offsets_X[None, :] * stride_y_row
                )
                tl.store(y_ptr_current_chunk, y_chunk_scaled.to(y_ptr.type.element_ty), mask=store_mask)


@triton.jit
def dequantize_mxfp8_kernel(
    x_ptr,
    y_ptr,
    stride_x_row,
    stride_x_col,
    stride_y_row,
    stride_y_col,
    n_rows,
    n_cols,
    scale_inv_ptr,
    stride_scale_inv_row,
    stride_scale_inv_col,
    scale_n_rows,
    scale_n_cols,
    BLOCK_X: tl.constexpr,
    BLOCK_Y: tl.constexpr,
    GROUP_Y: tl.constexpr,
    TRANS: tl.constexpr,
    MXFP8_BLOCK_SIZE: tl.constexpr,
):

    pid = tl.program_id(0)

    num_pid_along_Y = tl.cdiv(n_rows, BLOCK_Y)
    num_pid_along_X = tl.cdiv(n_cols, BLOCK_X)
    num_pid_in_group = GROUP_Y * num_pid_along_X

    group_id = pid // num_pid_in_group
    group_size = min(num_pid_along_Y - group_id * GROUP_Y, GROUP_Y)
    pid_m = group_id * GROUP_Y + ((pid % num_pid_in_group) % group_size)
    pid_n = (pid % num_pid_in_group) // group_size

    global_offset_Y_base = pid_m.to(tl.int64) * BLOCK_Y
    global_offset_X_base = pid_n.to(tl.int64) * BLOCK_X

    num_chunks_in_block_Y = BLOCK_Y // MXFP8_BLOCK_SIZE
    num_chunks_in_block_X = BLOCK_X // MXFP8_BLOCK_SIZE

    for chunk_id_y in range(0, num_chunks_in_block_Y):
        offsets_Y = global_offset_Y_base + chunk_id_y * MXFP8_BLOCK_SIZE + tl.arange(0, MXFP8_BLOCK_SIZE)
        for chunk_id_x in range(0, num_chunks_in_block_X):
            offsets_X = global_offset_X_base + chunk_id_x * MXFP8_BLOCK_SIZE + tl.arange(0, MXFP8_BLOCK_SIZE)
            x_ptr_current_chunk = (
                x_ptr + offsets_Y[:, None] * stride_x_row + offsets_X[None, :] * stride_x_col
            )
            load_mask = (offsets_Y < n_rows)[:, None] & (offsets_X < n_cols)[None, :]
            x_chunk = tl.load(x_ptr_current_chunk, mask=load_mask)

            scale_offset_X = (pid_n * num_chunks_in_block_X) + chunk_id_x
            scale_inv_load_offsets = (
                offsets_Y[:, None] * stride_scale_inv_row
            ) + scale_offset_X * stride_scale_inv_col
            scale_inv_load_mask = (offsets_Y < scale_n_rows)[:, None] & (scale_offset_X < scale_n_cols)

            biased_exponent = tl.load(
                scale_inv_ptr + scale_inv_load_offsets, mask=scale_inv_load_mask, other=127
            )

            block_scale = tl.exp2(biased_exponent.to(tl.float32) - 127)
            y_chunk_scaled = x_chunk.to(tl.float32) * block_scale

            if not TRANS:
                store_mask = (offsets_Y < n_rows)[:, None] & (offsets_X < n_cols)[None, :]
                y_ptr_current_chunk = (
                    y_ptr + offsets_Y[:, None] * stride_x_row + offsets_X[None, :] * stride_x_col
                )
                tl.store(y_ptr_current_chunk, y_chunk_scaled.to(y_ptr.type.element_ty), mask=store_mask)
            else:
                store_mask = (offsets_Y < n_rows)[:, None] & (offsets_X < n_cols)[None, :]
                y_ptr_current_chunk = (
                    y_ptr + offsets_Y[:, None] * stride_y_col + offsets_X[None, :] * stride_y_row
                )
                tl.store(y_ptr_current_chunk, y_chunk_scaled.to(y_ptr.type.element_ty), mask=store_mask)
