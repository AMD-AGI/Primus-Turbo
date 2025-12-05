###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################


import triton
import triton.language as tl

from primus_turbo.triton.utils.quantization_helper import exp2f_rcp, float_to_e8m0


@triton.jit
def quantize_mxfp4_kernel(
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
    USE_2D_BLOCK: tl.constexpr,
    USE_SR: tl.constexpr,
    MXFP4_BLOCK_SIZE: tl.constexpr,
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

    num_chunks_in_block_Y = BLOCK_Y // MXFP4_BLOCK_SIZE
    num_chunks_in_block_X = BLOCK_X // MXFP4_BLOCK_SIZE
    max_norm_rcp = 1.0 / max_fp8

    for chunk_id_y in range(0, num_chunks_in_block_Y):
        offsets_Y = global_offset_Y_base + chunk_id_y * MXFP4_BLOCK_SIZE + tl.arange(0, MXFP4_BLOCK_SIZE)
        for chunk_id_x in range(0, num_chunks_in_block_X):
            offsets_X = global_offset_X_base + chunk_id_x * MXFP4_BLOCK_SIZE + tl.arange(0, MXFP4_BLOCK_SIZE)
            x_ptr_current_chunk = (
                x_ptr + offsets_Y[:, None] * stride_x_row + offsets_X[None, :] * stride_x_col
            )
            load_mask = (offsets_Y < n_rows)[:, None] & (offsets_X < n_cols)[None, :]
            # (MXFP4_BLOCK_SIZE, MXFP4_BLOCK_SIZE)
            x_chunk = tl.load(x_ptr_current_chunk, mask=load_mask, other=0.0).to(tl.float32)

            # Row-wise
            subwarp_amax = tl.max(tl.abs(x_chunk), axis=-1, keep_dims=True)
            biased_exponent = float_to_e8m0(subwarp_amax * max_norm_rcp)

            scale_offset_X = (pid_n * num_chunks_in_block_X) + chunk_id_x
            scale_inv_store_offsets = (
                offsets_Y[:, None] * stride_scale_inv_row
            ) + scale_offset_X * stride_scale_inv_col
            scale_inv_store_mask = (offsets_Y < scale_n_rows)[:, None] & (scale_offset_X < scale_n_cols)
            tl.store(
                scale_inv_ptr + scale_inv_store_offsets,
                biased_exponent,
                mask=scale_inv_store_mask,
            )

            block_inverse_scale_rowwise = exp2f_rcp(biased_exponent)
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
           