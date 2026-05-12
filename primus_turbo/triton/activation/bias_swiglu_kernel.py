###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import triton
import triton.language as tl


@triton.jit
def bias_swiglu_fwd_kernel(
    x_ptr,
    bias_ptr,
    out_ptr,
    num_tokens: tl.constexpr,
    stride_x_token: tl.constexpr,
    stride_out_token: tl.constexpr,
    LOAD_WIDTH: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    pid = tl.program_id(0)
    compute_type = tl.float32
    data_type = x_ptr.dtype.element_ty

    half_stride = stride_x_token // 2

    row_idx = pid
    row_mask = row_idx < num_tokens
    col_off = tl.arange(0, LOAD_WIDTH)
    col_mask = col_off < half_stride
    mask = row_mask & col_mask

    up_ptr = x_ptr + row_idx * stride_x_token
    down_ptr = up_ptr + half_stride

    up = tl.load(up_ptr + col_off, mask=mask).to(compute_type)
    down = tl.load(down_ptr + col_off, mask=mask).to(compute_type)

    if HAS_BIAS:
        bias_up = tl.load(bias_ptr + col_off, mask=col_mask).to(compute_type)
        bias_down = tl.load(bias_ptr + half_stride + col_off, mask=col_mask).to(compute_type)
        up = up + bias_up
        down = down + bias_down

    silu = tl.fdiv(up, (1.0 + tl.exp(-up)))
    out = silu * down
    tl.store(out_ptr + row_idx * stride_out_token + col_off, out.to(data_type), mask=mask)


@triton.jit
def bias_swiglu_with_mask_fwd_kernel(
    x_ptr,
    bias_ptr,
    row_mask_ptr,
    out_ptr,
    num_tokens: tl.constexpr,
    stride_x_token,
    stride_out_token,
    LOAD_WIDTH: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    pid = tl.program_id(0)
    compute_type = tl.float32
    data_type = x_ptr.dtype.element_ty
    idx_type = tl.int64

    half_stride = stride_x_token // 2
    col_off = tl.arange(0, LOAD_WIDTH)
    col_mask = col_off < half_stride

    if HAS_BIAS:
        bias_up = tl.load(bias_ptr + col_off, mask=col_mask).to(compute_type)
        bias_down = tl.load(bias_ptr + half_stride + col_off, mask=col_mask).to(compute_type)

    loop = (num_tokens + BLOCK_SIZE - 1) // BLOCK_SIZE
    for i in range(0, loop):
        row_idx = (i * BLOCK_SIZE + pid).to(idx_type)
        row_mask = row_idx < num_tokens

        extra_mask = tl.load(row_mask_ptr + row_idx, mask=row_mask)
        mask = (row_mask & col_mask) & (extra_mask != 0)

        up_ptr = x_ptr + row_idx * stride_x_token
        down_ptr = up_ptr + half_stride

        up = tl.load(up_ptr + col_off, mask=mask).to(compute_type)
        down = tl.load(down_ptr + col_off, mask=mask).to(compute_type)

        if HAS_BIAS:
            up = up + bias_up
            down = down + bias_down

        silu = tl.fdiv(up, (1.0 + tl.exp(-up)))
        out = silu * down
        tl.store(out_ptr + row_idx * stride_out_token + col_off, out.to(data_type), mask=mask)


@triton.jit
def bias_swiglu_bwd_kernel(
    grad_out_ptr,
    x_ptr,
    bias_ptr,
    grad_x_ptr,
    num_tokens: tl.constexpr,
    stride_grad_out_token: tl.constexpr,
    stride_x_token: tl.constexpr,
    stride_grad_x_token: tl.constexpr,
    LOAD_WIDTH: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    pid = tl.program_id(0)
    compute_type = tl.float32
    grad_data_type = grad_x_ptr.dtype.element_ty

    half_stride = stride_x_token // 2

    row_idx = pid
    row_mask = row_idx < num_tokens
    col_off = tl.arange(0, LOAD_WIDTH)
    col_mask = col_off < half_stride
    mask = row_mask & col_mask

    up_ptr = x_ptr + row_idx * stride_x_token
    down_ptr = up_ptr + half_stride

    up = tl.load(up_ptr + col_off, mask=mask).to(compute_type)
    down = tl.load(down_ptr + col_off, mask=mask).to(compute_type)

    if HAS_BIAS:
        bias_up = tl.load(bias_ptr + col_off, mask=col_mask).to(compute_type)
        bias_down = tl.load(bias_ptr + half_stride + col_off, mask=col_mask).to(compute_type)
        up = up + bias_up
        down = down + bias_down

    sigmoid = tl.sigmoid(up)
    silu = sigmoid * up

    g = tl.load(grad_out_ptr + row_idx * stride_grad_out_token + col_off, mask=mask).to(compute_type)

    grad_down = g * silu
    grad_silu = sigmoid * (1.0 + up * (1.0 - sigmoid))
    grad_up = g * down * grad_silu

    tl.store(
        grad_x_ptr + row_idx * stride_grad_x_token + col_off,
        grad_up.to(grad_data_type),
        mask=mask,
    )
    tl.store(
        grad_x_ptr + row_idx * stride_grad_x_token + half_stride + col_off,
        grad_down.to(grad_data_type),
        mask=mask,
    )


@triton.jit
def bias_swiglu_with_mask_bwd_kernel(
    grad_out_ptr,
    x_ptr,
    bias_ptr,
    row_mask_ptr,
    grad_x_ptr,
    num_tokens: tl.constexpr,
    stride_grad_out_token,
    stride_x_token,
    stride_grad_x_token,
    LOAD_WIDTH: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    pid = tl.program_id(0)
    compute_type = tl.float32
    grad_data_type = grad_x_ptr.dtype.element_ty
    idx_type = tl.int64

    half_stride = stride_x_token // 2
    col_off = tl.arange(0, LOAD_WIDTH)
    col_mask = col_off < half_stride

    if HAS_BIAS:
        bias_up = tl.load(bias_ptr + col_off, mask=col_mask).to(compute_type)
        bias_down = tl.load(bias_ptr + half_stride + col_off, mask=col_mask).to(compute_type)

    loop = (num_tokens + BLOCK_SIZE - 1) // BLOCK_SIZE
    for i in range(0, loop):
        row_idx = (i * BLOCK_SIZE + pid).to(idx_type)
        row_mask = row_idx < num_tokens

        extra_mask = tl.load(row_mask_ptr + row_idx, mask=row_mask)
        mask = (row_mask & col_mask) & (extra_mask != 0)

        up_ptr = x_ptr + row_idx * stride_x_token
        down_ptr = up_ptr + half_stride

        up = tl.load(up_ptr + col_off, mask=mask).to(compute_type)
        down = tl.load(down_ptr + col_off, mask=mask).to(compute_type)

        if HAS_BIAS:
            up = up + bias_up
            down = down + bias_down

        sigmoid = tl.sigmoid(up)
        silu = sigmoid * up

        g = tl.load(grad_out_ptr + row_idx * stride_grad_out_token + col_off, mask=mask).to(compute_type)

        grad_down = g * silu
        grad_silu = sigmoid * (1.0 + up * (1.0 - sigmoid))
        grad_up = g * down * grad_silu

        tl.store(
            grad_x_ptr + row_idx * stride_grad_x_token + col_off,
            grad_up.to(grad_data_type),
            mask=mask,
        )
        tl.store(
            grad_x_ptr + row_idx * stride_grad_x_token + half_stride + col_off,
            grad_down.to(grad_data_type),
            mask=mask,
        )
