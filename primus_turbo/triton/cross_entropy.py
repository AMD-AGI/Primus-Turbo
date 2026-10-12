###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Modifications Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Adapted from NVIDIA TransformerEngine commit
# 76ae4b0981849d4b85a528d26f39981974b409f8,
# transformer_engine/common/triton/cross_entropy.py.
# Changes: retain TP=1 kernels; rename profiler symbols; widen strided addressing.
# See LICENSE-APACHE for the upstream license.

"""Fused cross-entropy loss/statistics and deferred gradient kernels."""

import triton
import triton.language as tl


@triton.jit
def turbo_cross_entropy_forward_kernel(
    X_ptr,
    X_stride_0,
    X_stride_1,
    X_stride_2,
    saved_input_ptr,
    Y_ptr,
    loss_ptr,
    stats_ptr,
    n_non_ignore,
    n_cols,
    n_rows_1,
    ignore_idx,
    label_smoothing: tl.constexpr,
    COUNT_NON_IGNORE: tl.constexpr,
    COPY_INPUT: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Compute single-rank loss/statistics and optionally preserve the input."""

    row = tl.program_id(0).to(tl.int64)
    row_0 = row // n_rows_1
    row_1 = row - row_0 * n_rows_1
    X_ptr += row_0 * X_stride_0 + row_1 * X_stride_1
    saved_input_ptr += row * n_cols

    y = tl.load(Y_ptr + row)
    if COUNT_NON_IGNORE:
        if y != ignore_idx:
            tl.atomic_add(n_non_ignore, 1)

    m = float("-inf")
    d = 0.0
    x_sum = 0.0
    for i in range(0, n_cols, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols
        if COPY_INPUT:
            # Preserving input also supports views whose vocabulary stride
            # spans more than 2**31 elements. Overwrite mode is contiguous.
            x = tl.load(X_ptr + offsets.to(tl.int64) * X_stride_2, mask=mask, other=float("-inf"))
            tl.store(saved_input_ptr + offsets, x, mask=mask)
        else:
            x = tl.load(X_ptr + offsets * X_stride_2, mask=mask, other=float("-inf"))
        x = x.to(tl.float32)
        block_max = tl.max(x)
        m_new = tl.maximum(m, block_max)
        d = d * tl.exp(m - m_new) + tl.sum(tl.exp(x - m_new))
        m = m_new
        if label_smoothing > 0:
            x_sum += tl.sum(tl.where(mask, x, 0.0))

    tl.store(stats_ptr + row * 2, m)
    tl.store(stats_ptr + row * 2 + 1, d)

    if y == ignore_idx:
        tl.store(loss_ptr + row, 0.0)
        return

    x_y = float("-inf")
    if y >= 0:
        if y < n_cols:
            x_y = tl.load(X_ptr + y * X_stride_2).to(tl.float32)

    loss = -(x_y - m - tl.log(d))
    if label_smoothing > 0:
        eps = label_smoothing / n_cols
        smooth_loss = -eps * x_sum + label_smoothing * (m + tl.log(d))
        loss = loss * (1 - label_smoothing) + smooth_loss
    tl.store(loss_ptr + row, loss)


@triton.jit
def turbo_cross_entropy_backward_kernel(
    saved_input_ptr,
    Y_ptr,
    stats_ptr,
    n_non_ignore_ptr,
    grad_output_ptr,
    grad_output_stride,
    rank,
    world_size,
    n_cols,
    ignore_idx,
    reduce_loss: tl.constexpr,
    label_smoothing: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Reconstruct the FP32 derivative and store it in the saved input buffer."""

    row = tl.program_id(0).to(tl.int64)
    saved_input_ptr += row * n_cols
    y = tl.load(Y_ptr + row)

    if y == ignore_idx:
        for i in range(0, n_cols, BLOCK_SIZE):
            offsets = i + tl.arange(0, BLOCK_SIZE)
            tl.store(saved_input_ptr + offsets, 0.0, mask=offsets < n_cols)
        return

    m = tl.load(stats_ptr + row * 2)
    d = tl.load(stats_ptr + row * 2 + 1)
    grad_output = tl.load(grad_output_ptr + row * grad_output_stride).to(tl.float32)
    if reduce_loss:
        grad_output /= tl.load(n_non_ignore_ptr)

    eps = label_smoothing / (n_cols * world_size)
    vocab_start = rank * n_cols
    target_col = y - vocab_start
    target_is_local = (y >= vocab_start) & (y < vocab_start + n_cols)

    for i in range(0, n_cols, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols
        x = tl.load(saved_input_ptr + offsets, mask=mask, other=float("-inf")).to(tl.float32)
        grad = tl.exp(x - m) / d - eps
        is_target = target_is_local & (offsets == target_col)
        grad -= tl.where(is_target, 1 - label_smoothing, 0.0)
        tl.store(saved_input_ptr + offsets, grad * grad_output, mask=mask)
