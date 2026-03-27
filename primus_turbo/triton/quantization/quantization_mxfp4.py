###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################


import triton
import triton.language as tl


@triton.jit
def dequantize_mxfp4_kernel(
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
    MXFP4_BLOCK_SIZE: tl.constexpr,
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

    num_chunks_in_block_Y = BLOCK_Y // MXFP4_BLOCK_SIZE
    num_chunks_in_block_X = BLOCK_X // MXFP4_BLOCK_SIZE

    for chunk_id_y in range(0, num_chunks_in_block_Y):
        offsets_Y = global_offset_Y_base + chunk_id_y * MXFP4_BLOCK_SIZE + tl.arange(0, MXFP4_BLOCK_SIZE)
        for chunk_id_x in range(0, num_chunks_in_block_X):
            x_offsets_X = (
                pid_n.to(tl.int64) * BLOCK_X // 2
                + chunk_id_x * MXFP4_BLOCK_SIZE // 2
                + tl.arange(0, MXFP4_BLOCK_SIZE // 2)
            )
            offsets_X = global_offset_X_base + chunk_id_x * MXFP4_BLOCK_SIZE + tl.arange(0, MXFP4_BLOCK_SIZE)
            x_ptr_current_chunk = (
                x_ptr + offsets_Y[:, None] * stride_x_row + x_offsets_X[None, :] * stride_x_col
            )
            load_mask = (offsets_Y < n_rows)[:, None] & (x_offsets_X < n_cols // 2)[None, :]
            x_chunk = tl.load(x_ptr_current_chunk, mask=load_mask)

            scale_offset_X = (pid_n * num_chunks_in_block_X) + chunk_id_x
            scale_inv_load_offsets = (
                offsets_Y[:, None] * stride_scale_inv_row
            ) + scale_offset_X * stride_scale_inv_col
            scale_inv_load_mask = (offsets_Y < scale_n_rows)[:, None] & (scale_offset_X < scale_n_cols)

            biased_exponent = tl.load(
                scale_inv_ptr + scale_inv_load_offsets, mask=scale_inv_load_mask, other=127
            )
            scale = (biased_exponent.to(tl.uint32) << 23).to(tl.float32, bitcast=True)

            x_chunk = x_chunk.to(tl.uint16)
            out_dtype = y_ptr.type.element_ty
            tl.static_assert(out_dtype == tl.bfloat16 or out_dtype == tl.float16 or out_dtype == tl.float32)

            if out_dtype == tl.bfloat16:
                y_chunk = tl.inline_asm_elementwise(
                    asm="""
                    v_cvt_scalef32_pk_bf16_fp4 $0, $1, $2 op_sel:[0,0];
                    """,
                    constraints="=&v,v,v",
                    args=[x_chunk, scale],
                    dtype=tl.uint32,
                    is_pure=True,
                    pack=1,
                )
                y_chunk1 = (y_chunk >> 16).to(tl.uint16).to(out_dtype, bitcast=True)
                y_chunk0 = (y_chunk & 0x0000FFFF).to(tl.uint16).to(out_dtype, bitcast=True)
            elif out_dtype == tl.float16:
                y_chunk = tl.inline_asm_elementwise(
                    asm="""
                    v_cvt_scalef32_pk_f16_fp4 $0, $1, $2 op_sel:[0,0];
                    """,
                    constraints="=&v,v,v",
                    args=[x_chunk, scale],
                    dtype=tl.uint32,
                    is_pure=True,
                    pack=1,
                )
                y_chunk1 = (y_chunk >> 16).to(tl.uint16).to(out_dtype, bitcast=True)
                y_chunk0 = (y_chunk & 0x0000FFFF).to(tl.uint16).to(out_dtype, bitcast=True)
            elif out_dtype == tl.float32:
                y_chunk = tl.inline_asm_elementwise(
                    asm="""
                    v_cvt_scalef32_pk_f32_fp4 $0, $1, $2 op_sel:[0,0];
                    """,
                    constraints="=&v,v,v",
                    args=[x_chunk, scale],
                    dtype=tl.uint64,
                    is_pure=True,
                    pack=1,
                )
                y_chunk1 = (y_chunk >> 32).to(tl.uint32).to(out_dtype, bitcast=True)
                y_chunk0 = (y_chunk & 0x00000000FFFFFFFF).to(tl.uint32).to(out_dtype, bitcast=True)

            y_chunk_scaled = tl.join(y_chunk0, y_chunk1).reshape(MXFP4_BLOCK_SIZE, MXFP4_BLOCK_SIZE)

            if USE_ROWWISE:
                store_mask = (offsets_Y < n_rows)[:, None] & (offsets_X < n_cols)[None, :]
                y_ptr_current_chunk = (
                    y_ptr + offsets_Y[:, None] * stride_y_row + offsets_X[None, :] * stride_y_col
                )
                tl.store(y_ptr_current_chunk, y_chunk_scaled.to(y_ptr.type.element_ty), mask=store_mask)
            else:
                store_mask = (offsets_Y < n_rows)[:, None] & (offsets_X < n_cols)[None, :]
                y_ptr_current_chunk = (
                    y_ptr + offsets_Y[:, None] * stride_y_col + offsets_X[None, :] * stride_y_row
                )
                tl.store(y_ptr_current_chunk, y_chunk_scaled.to(y_ptr.type.element_ty), mask=store_mask)
