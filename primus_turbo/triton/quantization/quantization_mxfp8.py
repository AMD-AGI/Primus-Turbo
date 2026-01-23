###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################


import triton
import triton.language as tl

from primus_turbo.triton.utils.quantization_helper import (
    E8M0_EXPONENT_BIAS,
    calculate_fp8_e8m0_scale,
    scale_e8m0_to_rcp,
)


@triton.jit
def quantize_mxfp8_kernel(
    x_ptr,
    y_rowwise_ptr,
    y_colwise_ptr,
    stride_x_row,
    stride_x_col,
    stride_y_rowwise_row,
    stride_y_rowwise_col,
    stride_y_colwise_row,
    stride_y_colwise_col,
    n_rows,
    n_cols,
    padded_n_rows,
    padded_n_cols,
    scale_inv_rowwise_ptr,
    scale_inv_colwise_ptr,
    stride_scale_inv_rowwise_row,
    stride_scale_inv_rowwise_col,
    stride_scale_inv_colwise_row,
    stride_scale_inv_colwise_col,
    scale_inv_rowwise_n_rows,
    scale_inv_rowwise_n_cols,
    scale_inv_colwise_n_rows,
    scale_inv_colwise_n_cols,
    BLOCK_X: tl.constexpr,
    BLOCK_Y: tl.constexpr,
    GROUP_Y: tl.constexpr,
    USE_ROWWISE: tl.constexpr,
    USE_COLWISE: tl.constexpr,
    ROWWISE_USE_2D_BLOCK: tl.constexpr,
    COLWISE_USE_2D_BLOCK: tl.constexpr,
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

            if USE_ROWWISE:
                # Row-wise
                tl.device_assert(scale_inv_rowwise_ptr is not None)
                tl.device_assert(y_rowwise_ptr is not None)

                if ROWWISE_USE_2D_BLOCK:
                    biased_exponent = calculate_fp8_e8m0_scale(
                        x_chunk, axis=None, FP8_DTYPE=y_rowwise_ptr.type.element_ty
                    )
                else:
                    biased_exponent = calculate_fp8_e8m0_scale(
                        x_chunk, axis=-1, FP8_DTYPE=y_rowwise_ptr.type.element_ty
                    )

                scale_offset_X = (pid_n * num_chunks_in_block_X) + chunk_id_x
                scale_inv_store_offsets = (
                    offsets_Y[:, None] * stride_scale_inv_rowwise_row
                ) + scale_offset_X * stride_scale_inv_rowwise_col
                scale_inv_store_mask = (offsets_Y < scale_inv_rowwise_n_rows)[:, None] & (
                    scale_offset_X < scale_inv_rowwise_n_cols
                )
                tl.store(
                    scale_inv_rowwise_ptr + scale_inv_store_offsets,
                    biased_exponent,
                    mask=scale_inv_store_mask,
                )

                block_inverse_scale = scale_e8m0_to_rcp(biased_exponent)
                y_chunk_scaled = x_chunk * block_inverse_scale

                store_mask = (offsets_Y < padded_n_rows)[:, None] & (offsets_X < padded_n_cols)[None, :]
                y_ptr_current_chunk = (
                    y_rowwise_ptr
                    + offsets_Y[:, None] * stride_y_rowwise_row
                    + offsets_X[None, :] * stride_y_rowwise_col
                )
                tl.store(
                    y_ptr_current_chunk,
                    y_chunk_scaled.to(y_rowwise_ptr.type.element_ty),
                    mask=store_mask,
                )

            if USE_COLWISE:
                # Col-wise
                tl.device_assert(scale_inv_colwise_ptr is not None)
                tl.device_assert(y_colwise_ptr is not None)

                if COLWISE_USE_2D_BLOCK:
                    biased_exponent = calculate_fp8_e8m0_scale(
                        x_chunk, axis=None, FP8_DTYPE=y_colwise_ptr.type.element_ty
                    )
                else:
                    biased_exponent = calculate_fp8_e8m0_scale(
                        x_chunk, axis=0, FP8_DTYPE=y_colwise_ptr.type.element_ty
                    )

                scale_offset_Y = (pid_m * num_chunks_in_block_Y) + chunk_id_y
                scale_inv_store_offsets = scale_offset_Y * stride_scale_inv_colwise_col + (
                    offsets_X[None, :] * stride_scale_inv_colwise_row
                )
                scale_inv_store_mask = (scale_offset_Y < scale_inv_colwise_n_rows) & (
                    offsets_X < scale_inv_colwise_n_cols
                )[None, :]
                tl.store(
                    scale_inv_colwise_ptr + scale_inv_store_offsets,
                    biased_exponent,
                    mask=scale_inv_store_mask,
                )

                block_inverse_scale = scale_e8m0_to_rcp(biased_exponent)
                y_chunk_scaled = x_chunk * block_inverse_scale
                store_mask = (offsets_Y < padded_n_rows)[:, None] & (offsets_X < padded_n_cols)[None, :]
                y_ptr_current_chunk = (
                    y_colwise_ptr
                    + offsets_Y[:, None] * stride_y_colwise_col
                    + offsets_X[None, :] * stride_y_colwise_row
                )
                tl.store(
                    y_ptr_current_chunk, y_chunk_scaled.to(y_colwise_ptr.type.element_ty), mask=store_mask
                )


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
    USE_ROWWISE: tl.constexpr,
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
                scale_inv_ptr + scale_inv_load_offsets, mask=scale_inv_load_mask, other=E8M0_EXPONENT_BIAS
            )

            block_scale = tl.exp2(biased_exponent.to(tl.float32) - E8M0_EXPONENT_BIAS)
            y_chunk_scaled = x_chunk.to(tl.float32) * block_scale

            if USE_ROWWISE:
                # Row-wise
                store_mask = (offsets_Y < n_rows)[:, None] & (offsets_X < n_cols)[None, :]
                y_ptr_current_chunk = (
                    y_ptr + offsets_Y[:, None] * stride_y_row + offsets_X[None, :] * stride_y_col
                )
                tl.store(y_ptr_current_chunk, y_chunk_scaled.to(y_ptr.type.element_ty), mask=store_mask)
            else:
                # Col-wise
                store_mask = (offsets_Y < n_rows)[:, None] & (offsets_X < n_cols)[None, :]
                y_ptr_current_chunk = (
                    y_ptr + offsets_Y[:, None] * stride_y_col + offsets_X[None, :] * stride_y_row
                )
                tl.store(y_ptr_current_chunk, y_chunk_scaled.to(y_ptr.type.element_ty), mask=store_mask)
