###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import triton
import triton.language as tl

from primus_turbo.triton.utils.gelu import gelu_bwd_none, gelu_none


@triton.jit
def geglu_with_mask_fwd_kernel(
    # pointers
    x_ptr,
    probs_ptr,
    row_mask_ptr,
    out_ptr,
    # sizes
    num_tokens: tl.constexpr,
    # strides
    stride_x_token,
    stride_probs_token,
    stride_out_token,
    # metas
    LOAD_WIDTH: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)

    compute_type = tl.float32
    data_type = x_ptr.dtype.element_ty
    idx_type = tl.int64

    half_stride_x_token = stride_x_token // 2
    loop = (num_tokens + BLOCK_SIZE - 1) // BLOCK_SIZE
    for i in range(0, loop):
        row_idx = (i * BLOCK_SIZE + pid).to(idx_type)
        row_mask = row_idx < num_tokens
        col_off = tl.arange(0, LOAD_WIDTH)
        col_mask = col_off < half_stride_x_token

        extra_mask = tl.load(row_mask_ptr + row_idx, mask=row_mask)
        mask = (row_mask & col_mask) & (extra_mask != 0)

        up_ptr = x_ptr + row_idx * stride_x_token
        down_ptr = up_ptr + half_stride_x_token

        up = tl.load(up_ptr + col_off, mask=mask).to(compute_type)
        down = tl.load(down_ptr + col_off, mask=mask).to(compute_type)

        up = gelu_none(up)
        out = up * down

        probs = tl.load(probs_ptr + row_idx * stride_probs_token)
        out = out * probs

        tl.store(out_ptr + row_idx * stride_out_token + col_off, out.to(data_type), mask=mask)


@triton.jit
def geglu_with_mask_bwd_kernel(
    # pointers
    grad_out_ptr,
    x_ptr,
    probs_ptr,
    row_mask_ptr,
    grad_x_ptr,
    grad_probs_ptr,
    # sizes
    num_tokens: tl.constexpr,
    # strides
    stride_grad_out_token,
    stride_x_token,
    stride_probs_token,
    stride_grad_x_token,
    stride_grad_probs_token,
    # metas
    LOAD_WIDTH: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)

    compute_type = tl.float32
    grad_x_data_type = grad_x_ptr.dtype.element_ty
    grad_probs_data_type = grad_probs_ptr.dtype.element_ty
    idx_type = tl.int64

    half_stride_x_token = stride_x_token // 2
    loop = (num_tokens + BLOCK_SIZE - 1) // BLOCK_SIZE
    for i in range(0, loop):
        row_idx = (i * BLOCK_SIZE + pid).to(idx_type)
        row_mask = row_idx < num_tokens
        col_off = tl.arange(0, LOAD_WIDTH)
        col_mask = col_off < half_stride_x_token

        extra_mask = tl.load(row_mask_ptr + row_idx, mask=row_mask)
        mask = (row_mask & col_mask) & (extra_mask != 0)

        up_ptr = x_ptr + row_idx * stride_x_token
        down_ptr = up_ptr + half_stride_x_token

        up = tl.load(up_ptr + col_off, mask=mask).to(compute_type)
        down = tl.load(down_ptr + col_off, mask=mask).to(compute_type)

        gelu = gelu_none(up)

        grad_out = tl.load(grad_out_ptr + row_idx * stride_grad_out_token + col_off, mask=mask).to(
            compute_type
        )

        grad_probs = grad_out * gelu * down
        grad_probs_sum = tl.sum(grad_probs, dtype=compute_type)

        tl.store(
            grad_probs_ptr + row_idx * stride_grad_probs_token,
            grad_probs_sum.to(grad_probs_data_type),
            mask=row_mask,
        )

        probs = tl.load(probs_ptr + row_idx * stride_probs_token).to(compute_type)

        grad_out_with_probs = grad_out * probs
        grad_down = grad_out_with_probs * gelu
        grad_gelu = gelu_bwd_none(up, grad_out)
        grad_up = grad_out_with_probs * down * grad_gelu

        tl.store(
            grad_x_ptr + row_idx * stride_grad_x_token + col_off, grad_up.to(grad_x_data_type), mask=mask
        )
        tl.store(
            grad_x_ptr + row_idx * stride_grad_x_token + stride_grad_x_token // 2 + col_off,
            grad_down.to(grad_x_data_type),
            mask=mask,
        )


@triton.jit
def geglu_fwd_kernel(
    # pointers
    x_ptr,
    probs_ptr,
    out_ptr,
    # sizes
    num_tokens: tl.constexpr,
    # strides
    stride_x_token: tl.constexpr,
    stride_probs_token: tl.constexpr,
    stride_out_token: tl.constexpr,
    # metas
    LOAD_WIDTH: tl.constexpr,
):
    pid = tl.program_id(0)

    compute_type = tl.float32
    data_type = x_ptr.dtype.element_ty

    half_stride_x_token = stride_x_token // 2

    row_idx = pid
    row_mask = row_idx < num_tokens
    col_off = tl.arange(0, LOAD_WIDTH)
    col_mask = col_off < half_stride_x_token

    mask = row_mask & col_mask

    up_ptr = x_ptr + row_idx * stride_x_token
    down_ptr = up_ptr + half_stride_x_token

    up = tl.load(up_ptr + col_off, mask=mask).to(compute_type)
    down = tl.load(down_ptr + col_off, mask=mask).to(compute_type)

    up = gelu_none(up)
    out = up * down

    probs = tl.load(probs_ptr + row_idx * stride_probs_token)
    out = out * probs

    tl.store(out_ptr + row_idx * stride_out_token + col_off, out.to(data_type), mask=mask)


@triton.jit
def geglu_bwd_kernel(
    # pointers
    grad_out_ptr,
    x_ptr,
    probs_ptr,
    grad_x_ptr,
    grad_probs_ptr,
    # sizes
    num_tokens: tl.constexpr,
    # strides
    stride_grad_out_token: tl.constexpr,
    stride_x_token: tl.constexpr,
    stride_probs_token: tl.constexpr,
    stride_grad_x_token: tl.constexpr,
    stride_grad_probs_token: tl.constexpr,
    # metas
    LOAD_WIDTH: tl.constexpr,
):
    pid = tl.program_id(0)

    compute_type = tl.float32
    grad_x_data_type = grad_x_ptr.dtype.element_ty
    grad_probs_data_type = grad_probs_ptr.dtype.element_ty

    half_stride_x_token = stride_x_token // 2

    row_idx = pid
    row_mask = row_idx < num_tokens
    col_off = tl.arange(0, LOAD_WIDTH)
    col_mask = col_off < half_stride_x_token

    mask = row_mask & col_mask

    up_ptr = x_ptr + row_idx * stride_x_token
    down_ptr = up_ptr + half_stride_x_token

    up = tl.load(up_ptr + col_off, mask=mask).to(compute_type)
    down = tl.load(down_ptr + col_off, mask=mask).to(compute_type)

    gelu = gelu_none(up)

    grad_out = tl.load(grad_out_ptr + row_idx * stride_grad_out_token + col_off, mask=mask).to(compute_type)

    grad_probs = grad_out * gelu * down
    grad_probs_sum = tl.sum(grad_probs, dtype=compute_type)

    tl.store(
        grad_probs_ptr + row_idx * stride_grad_probs_token,
        grad_probs_sum.to(grad_probs_data_type),
        mask=row_mask,
    )

    probs = tl.load(probs_ptr + row_idx * stride_probs_token).to(compute_type)

    grad_out_with_probs = grad_out * probs
    grad_down = grad_out_with_probs * gelu
    grad_gelu = gelu_bwd_none(up, grad_out)
    grad_up = grad_out_with_probs * down * grad_gelu

    tl.store(grad_x_ptr + row_idx * stride_grad_x_token + col_off, grad_up.to(grad_x_data_type), mask=mask)
    tl.store(
        grad_x_ptr + row_idx * stride_grad_x_token + stride_grad_x_token // 2 + col_off,
        grad_down.to(grad_x_data_type),
        mask=mask,
    )
