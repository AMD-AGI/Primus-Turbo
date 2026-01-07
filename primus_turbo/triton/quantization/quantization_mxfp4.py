###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################


import triton
import triton.language as tl

from primus_turbo.triton.utils.quantization_helper import calculate_e8m0_scale
from primus_turbo.triton.utils.triton_lang_helper import generate_randval_4x


@triton.jit
def pack_and_scale_fp32_to_fp4(
    x_chunk, scale, philox_seed, philox_offset, USE_SR: tl.constexpr, MXFP4_BLOCK_SIZE: tl.constexpr
):
    x_chunk0, x_chunk1 = tl.split(x_chunk.reshape(MXFP4_BLOCK_SIZE, MXFP4_BLOCK_SIZE // 2, 2))

    if not USE_SR:
        # NOTE: dtype must be uint16 for the ISA to work.
        y_chunk_scaled = tl.inline_asm_elementwise(
            asm="""
            v_cvt_scalef32_pk_fp4_f32 $0, $1, $2, $3 op_sel:[0,0,0,0];
            """,
            constraints="=&v,v,v,v",
            args=[x_chunk0, x_chunk1, scale],
            dtype=tl.uint16,
            is_pure=True,
            pack=1,
        )
    else:
        randval, _, _, _ = generate_randval_4x(
            MXFP4_BLOCK_SIZE, MXFP4_BLOCK_SIZE // 2, philox_seed, philox_offset
        )
        # Pack two float32 values (x_chunk0 and x_chunk1) into a single 64-bit value.
        # This is required for the stochastic rounding instruction, which expects the two values packed.
        x_chunk = (x_chunk1.to(tl.uint32, bitcast=True).to(tl.uint64) << 32) | x_chunk0.to(
            tl.uint32, bitcast=True
        )
        # NOTE: dtype must be uint16 for the ISA to work.
        y_chunk_scaled = tl.inline_asm_elementwise(
            asm="""
            v_cvt_scalef32_sr_pk_fp4_f32 $0, $1, $2, $3 op_sel:[0,0,0,0];
            """,
            constraints="=&v,v,v,v",
            args=[x_chunk, randval, scale],
            dtype=tl.uint16,
            is_pure=True,
            pack=1,
        )

    y_chunk_scaled = (y_chunk_scaled & 0x00FF).to(tl.uint8)

    return y_chunk_scaled


@triton.jit
def quantize_mxfp4_kernel(
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
    hadamard_matrix_ptr,
    hadamard_matrix_size,
    rowwise_philox_seed,
    rowwise_philox_offset,
    colwise_philox_seed,
    colwise_philox_offset,
    BLOCK_X: tl.constexpr,
    BLOCK_Y: tl.constexpr,
    GROUP_Y: tl.constexpr,
    USE_ROWWISE: tl.constexpr,
    USE_COLWISE: tl.constexpr,
    ROWWISE_USE_2D_BLOCK: tl.constexpr,
    COLWISE_USE_2D_BLOCK: tl.constexpr,
    ROWWISE_USE_SR: tl.constexpr,
    COLWISE_USE_SR: tl.constexpr,
    ROWWISE_USE_RHT: tl.constexpr,
    COLWISE_USE_RHT: tl.constexpr,
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

    if ROWWISE_USE_RHT or COLWISE_USE_RHT:
        # NOTE: Hadamard matrix is 2D matrix whose the rows and columns are the same size and both are the power of 2.
        tl.device_assert(
            hadamard_matrix_size == MXFP4_BLOCK_SIZE, "Hadamard matrix size must be equal to MXFP4_BLOCK_SIZE"
        )
        offset_hadamard_matrix_x = tl.arange(0, MXFP4_BLOCK_SIZE)
        offset_hadamard_matrix_y = tl.arange(0, MXFP4_BLOCK_SIZE)
        hadamard_matrix = tl.load(
            hadamard_matrix_ptr
            + offset_hadamard_matrix_y[:, None] * MXFP4_BLOCK_SIZE
            + offset_hadamard_matrix_x[None, :]
        ).to(tl.float32)

        if USE_COLWISE:
            hadamard_matrix_t = tl.trans(hadamard_matrix, (1, 0))

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

            if USE_ROWWISE:
                # Row-wise
                if ROWWISE_USE_RHT:
                    x_chunk = tl.dot(x_chunk, hadamard_matrix)

                if ROWWISE_USE_2D_BLOCK:
                    biased_exponent = calculate_e8m0_scale(x_chunk, axis=None)
                else:
                    biased_exponent = calculate_e8m0_scale(x_chunk, axis=-1)

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

                scale = (biased_exponent.to(tl.uint32) << 23).to(tl.float32, bitcast=True)

                y_chunk_scaled = pack_and_scale_fp32_to_fp4(
                    x_chunk,
                    scale,
                    rowwise_philox_seed,
                    rowwise_philox_offset,
                    ROWWISE_USE_SR,
                    MXFP4_BLOCK_SIZE,
                )

                y_offsets_X = (
                    pid_n.to(tl.int64) * BLOCK_X // 2
                    + chunk_id_x * MXFP4_BLOCK_SIZE // 2
                    + tl.arange(0, MXFP4_BLOCK_SIZE // 2)
                )
                store_mask = (offsets_Y < padded_n_rows)[:, None] & (y_offsets_X < padded_n_cols // 2)[
                    None, :
                ]
                y_ptr_current_chunk = (
                    y_rowwise_ptr
                    + offsets_Y[:, None] * stride_y_rowwise_row
                    + y_offsets_X[None, :] * stride_y_rowwise_col
                )
                tl.store(
                    y_ptr_current_chunk,
                    y_chunk_scaled,
                    mask=store_mask,
                )

            if USE_COLWISE:
                # Col-wise
                if COLWISE_USE_RHT:
                    x_chunk = tl.dot(hadamard_matrix_t, x_chunk)

                if COLWISE_USE_2D_BLOCK:
                    biased_exponent = calculate_e8m0_scale(x_chunk, axis=None)
                else:
                    biased_exponent = calculate_e8m0_scale(x_chunk, axis=0)

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

                # NOTE: Transpose x chunk to Row-wise because of ISA requirement
                x_chunk = tl.trans(x_chunk, (1, 0))

                scale = (biased_exponent.to(tl.uint32) << 23).to(tl.float32, bitcast=True)
                # NOTE: Transpose scale to Row-wise because of x_chunk is Row-wise
                scale = tl.trans(scale, (1, 0))

                y_chunk_scaled = pack_and_scale_fp32_to_fp4(
                    x_chunk,
                    scale,
                    colwise_philox_seed,
                    colwise_philox_offset,
                    COLWISE_USE_SR,
                    MXFP4_BLOCK_SIZE,
                )

                # Transpose y chunk to Col-wise
                y_chunk_scaled = tl.trans(y_chunk_scaled, (1, 0)).reshape(
                    MXFP4_BLOCK_SIZE // 2, MXFP4_BLOCK_SIZE
                )
                y_offsets_Y = (
                    pid_m.to(tl.int64) * BLOCK_Y // 2
                    + chunk_id_y * MXFP4_BLOCK_SIZE // 2
                    + tl.arange(0, MXFP4_BLOCK_SIZE // 2)
                )
                store_mask = (y_offsets_Y < padded_n_rows // 2)[:, None] & (offsets_X < padded_n_cols)[
                    None, :
                ]
                y_ptr_current_chunk = (
                    y_colwise_ptr
                    + offsets_X[None, :] * stride_y_colwise_row
                    + y_offsets_Y[:, None] * stride_y_colwise_col
                )
                tl.store(
                    y_ptr_current_chunk,
                    y_chunk_scaled,
                    mask=store_mask,
                )


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
