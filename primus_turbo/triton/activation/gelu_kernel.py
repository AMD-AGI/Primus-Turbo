###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import triton
import triton.language as tl

from primus_turbo.triton.utils.gelu import gelu_bwd_none, gelu_none


@triton.jit
def gelu_fwd_kernel(
    x_ptr,
    probs_ptr,
    out_ptr,
    num_tokens: tl.constexpr,
    stride_x_token: tl.constexpr,
    stride_probs_token: tl.constexpr,
    stride_out_token: tl.constexpr,
    LOAD_WIDTH: tl.constexpr,
):
    pid = tl.program_id(0)
    compute_type = tl.float32
    data_type = x_ptr.dtype.element_ty

    row_idx = pid
    row_mask = row_idx < num_tokens
    col_off = tl.arange(0, LOAD_WIDTH)
    col_mask = col_off < stride_x_token
    mask = row_mask & col_mask

    x = tl.load(x_ptr + row_idx * stride_x_token + col_off, mask=mask).to(compute_type)
    out = gelu_none(x)

    probs = tl.load(probs_ptr + row_idx * stride_probs_token)
    out = out * probs

    tl.store(out_ptr + row_idx * stride_out_token + col_off, out.to(data_type), mask=mask)


@triton.jit
def gelu_with_mask_fwd_kernel(
    x_ptr,
    probs_ptr,
    row_mask_ptr,
    out_ptr,
    num_tokens: tl.constexpr,
    stride_x_token,
    stride_probs_token,
    stride_out_token,
    LOAD_WIDTH: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    compute_type = tl.float32
    data_type = x_ptr.dtype.element_ty
    idx_type = tl.int64

    loop = (num_tokens + BLOCK_SIZE - 1) // BLOCK_SIZE
    for i in range(0, loop):
        row_idx = (i * BLOCK_SIZE + pid).to(idx_type)
        row_mask = row_idx < num_tokens
        col_off = tl.arange(0, LOAD_WIDTH)
        col_mask = col_off < stride_x_token

        extra_mask = tl.load(row_mask_ptr + row_idx, mask=row_mask)
        mask = (row_mask & col_mask) & (extra_mask != 0)

        x = tl.load(x_ptr + row_idx * stride_x_token + col_off, mask=mask).to(compute_type)
        out = gelu_none(x)

        probs = tl.load(probs_ptr + row_idx * stride_probs_token, mask=row_mask)
        out = out * probs

        tl.store(out_ptr + row_idx * stride_out_token + col_off, out.to(data_type), mask=mask)


@triton.jit
def gelu_bwd_kernel(
    grad_out_ptr,
    x_ptr,
    probs_ptr,
    grad_x_ptr,
    grad_probs_ptr,
    num_tokens: tl.constexpr,
    stride_grad_out_token: tl.constexpr,
    stride_x_token: tl.constexpr,
    stride_probs_token: tl.constexpr,
    stride_grad_x_token: tl.constexpr,
    stride_grad_probs_token: tl.constexpr,
    LOAD_WIDTH: tl.constexpr,
):
    pid = tl.program_id(0)
    compute_type = tl.float32
    grad_x_data_type = grad_x_ptr.dtype.element_ty
    grad_probs_data_type = grad_probs_ptr.dtype.element_ty

    row_idx = pid
    row_mask = row_idx < num_tokens
    col_off = tl.arange(0, LOAD_WIDTH)
    col_mask = col_off < stride_x_token
    mask = row_mask & col_mask

    x = tl.load(x_ptr + row_idx * stride_x_token + col_off, mask=mask).to(compute_type)
    gelu_out = gelu_none(x)

    grad_out = tl.load(grad_out_ptr + row_idx * stride_grad_out_token + col_off, mask=mask).to(compute_type)

    # grad_probs = sum(grad_out * gelu(x))
    grad_probs = grad_out * gelu_out
    grad_probs_sum = tl.sum(grad_probs, dtype=compute_type)
    tl.store(
        grad_probs_ptr + row_idx * stride_grad_probs_token,
        grad_probs_sum.to(grad_probs_data_type),
        mask=row_mask,
    )

    # grad_x = grad_out * probs * gelu'(x)
    probs = tl.load(probs_ptr + row_idx * stride_probs_token).to(compute_type)
    grad_x = gelu_bwd_none(x, grad_out * probs)

    tl.store(
        grad_x_ptr + row_idx * stride_grad_x_token + col_off,
        grad_x.to(grad_x_data_type),
        mask=mask,
    )


@triton.jit
def gelu_with_mask_bwd_kernel(
    grad_out_ptr,
    x_ptr,
    probs_ptr,
    row_mask_ptr,
    grad_x_ptr,
    grad_probs_ptr,
    num_tokens: tl.constexpr,
    stride_grad_out_token,
    stride_x_token,
    stride_probs_token,
    stride_grad_x_token,
    stride_grad_probs_token,
    LOAD_WIDTH: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    compute_type = tl.float32
    grad_x_data_type = grad_x_ptr.dtype.element_ty
    grad_probs_data_type = grad_probs_ptr.dtype.element_ty
    idx_type = tl.int64

    loop = (num_tokens + BLOCK_SIZE - 1) // BLOCK_SIZE
    for i in range(0, loop):
        row_idx = (i * BLOCK_SIZE + pid).to(idx_type)
        row_mask = row_idx < num_tokens
        col_off = tl.arange(0, LOAD_WIDTH)
        col_mask = col_off < stride_x_token

        extra_mask = tl.load(row_mask_ptr + row_idx, mask=row_mask)
        mask = (row_mask & col_mask) & (extra_mask != 0)

        x = tl.load(x_ptr + row_idx * stride_x_token + col_off, mask=mask).to(compute_type)
        gelu_out = gelu_none(x)

        grad_out = tl.load(grad_out_ptr + row_idx * stride_grad_out_token + col_off, mask=mask).to(
            compute_type
        )

        grad_probs = grad_out * gelu_out
        grad_probs_sum = tl.sum(grad_probs, dtype=compute_type)
        tl.store(
            grad_probs_ptr + row_idx * stride_grad_probs_token,
            grad_probs_sum.to(grad_probs_data_type),
            mask=row_mask,
        )

        probs = tl.load(probs_ptr + row_idx * stride_probs_token, mask=row_mask).to(compute_type)
        grad_x = gelu_bwd_none(x, grad_out * probs)

        tl.store(
            grad_x_ptr + row_idx * stride_grad_x_token + col_off,
            grad_x.to(grad_x_data_type),
            mask=mask,
        )
