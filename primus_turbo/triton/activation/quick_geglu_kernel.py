###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import triton
import triton.language as tl

from primus_turbo.triton.utils.gelu import quick_gelu


@triton.jit
def quick_geglu_fwd_kernel(
    x_ptr,
    weights_ptr,
    bias_ptr,
    out_ptr,
    linear_offset,
    num_tokens: tl.constexpr,
    stride_x_token: tl.constexpr,
    stride_weights_token: tl.constexpr,
    stride_out_token: tl.constexpr,
    LOAD_WIDTH: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    HAS_WEIGHTS: tl.constexpr,
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

    out = quick_gelu(up) * (down + linear_offset)

    if HAS_WEIGHTS:
        w = tl.load(weights_ptr + row_idx * stride_weights_token)
        out = out * w

    tl.store(out_ptr + row_idx * stride_out_token + col_off, out.to(data_type), mask=mask)


@triton.jit
def quick_geglu_with_mask_fwd_kernel(
    x_ptr,
    weights_ptr,
    bias_ptr,
    row_mask_ptr,
    out_ptr,
    linear_offset,
    num_tokens: tl.constexpr,
    stride_x_token,
    stride_weights_token,
    stride_out_token,
    LOAD_WIDTH: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    HAS_WEIGHTS: tl.constexpr,
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

        out = quick_gelu(up) * (down + linear_offset)

        if HAS_WEIGHTS:
            w = tl.load(weights_ptr + row_idx * stride_weights_token, mask=row_mask)
            out = out * w

        tl.store(out_ptr + row_idx * stride_out_token + col_off, out.to(data_type), mask=mask)


@triton.jit
def quick_geglu_bwd_kernel(
    grad_out_ptr,
    x_ptr,
    weights_ptr,
    bias_ptr,
    grad_x_ptr,
    grad_weights_ptr,
    linear_offset,
    num_tokens: tl.constexpr,
    stride_grad_out_token: tl.constexpr,
    stride_x_token: tl.constexpr,
    stride_weights_token: tl.constexpr,
    stride_grad_x_token: tl.constexpr,
    stride_grad_weights_token: tl.constexpr,
    LOAD_WIDTH: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    HAS_WEIGHTS: tl.constexpr,
):
    pid = tl.program_id(0)
    compute_type = tl.float32
    grad_x_dtype = grad_x_ptr.dtype.element_ty
    tl.int64

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

    sigmoid_out = tl.sigmoid(1.702 * up)
    quick_gelu_up = up * sigmoid_out

    g = tl.load(grad_out_ptr + row_idx * stride_grad_out_token + col_off, mask=mask).to(compute_type)

    if HAS_WEIGHTS:
        activation_val = quick_gelu_up * (down + linear_offset)
        grad_w = tl.sum(g * activation_val, dtype=compute_type)
        grad_w_dtype = grad_weights_ptr.dtype.element_ty
        tl.store(
            grad_weights_ptr + row_idx * stride_grad_weights_token,
            grad_w.to(grad_w_dtype),
            mask=row_mask,
        )
        w = tl.load(weights_ptr + row_idx * stride_weights_token).to(compute_type)
        g = g * w

    dy_up = g * sigmoid_out * (1.0 + 1.702 * up * (1.0 - sigmoid_out)) * (down + linear_offset)
    dy_down = g * quick_gelu_up

    tl.store(
        grad_x_ptr + row_idx * stride_grad_x_token + col_off,
        dy_up.to(grad_x_dtype),
        mask=mask,
    )
    tl.store(
        grad_x_ptr + row_idx * stride_grad_x_token + half_stride + col_off,
        dy_down.to(grad_x_dtype),
        mask=mask,
    )


@triton.jit
def quick_geglu_with_mask_bwd_kernel(
    grad_out_ptr,
    x_ptr,
    weights_ptr,
    bias_ptr,
    row_mask_ptr,
    grad_x_ptr,
    grad_weights_ptr,
    linear_offset,
    num_tokens: tl.constexpr,
    stride_grad_out_token,
    stride_x_token,
    stride_weights_token,
    stride_grad_x_token,
    stride_grad_weights_token,
    LOAD_WIDTH: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    HAS_WEIGHTS: tl.constexpr,
):
    pid = tl.program_id(0)
    compute_type = tl.float32
    grad_x_dtype = grad_x_ptr.dtype.element_ty
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

        sigmoid_out = tl.sigmoid(1.702 * up)
        quick_gelu_up = up * sigmoid_out

        g = tl.load(grad_out_ptr + row_idx * stride_grad_out_token + col_off, mask=mask).to(compute_type)

        if HAS_WEIGHTS:
            activation_val = quick_gelu_up * (down + linear_offset)
            grad_w = tl.sum(g * activation_val, dtype=compute_type)
            grad_w_dtype = grad_weights_ptr.dtype.element_ty
            tl.store(
                grad_weights_ptr + row_idx * stride_grad_weights_token,
                grad_w.to(grad_w_dtype),
                mask=row_mask,
            )
            w = tl.load(weights_ptr + row_idx * stride_weights_token, mask=row_mask).to(compute_type)
            g = g * w

        dy_up = g * sigmoid_out * (1.0 + 1.702 * up * (1.0 - sigmoid_out)) * (down + linear_offset)
        dy_down = g * quick_gelu_up

        tl.store(
            grad_x_ptr + row_idx * stride_grad_x_token + col_off,
            dy_up.to(grad_x_dtype),
            mask=mask,
        )
        tl.store(
            grad_x_ptr + row_idx * stride_grad_x_token + half_stride + col_off,
            dy_down.to(grad_x_dtype),
            mask=mask,
        )
