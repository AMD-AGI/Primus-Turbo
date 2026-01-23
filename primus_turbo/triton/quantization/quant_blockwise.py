###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import triton
import triton.language as tl


@triton.jit
def compute_scale_and_quant(x_tile, x_tile_abs, axis, FP8_MAX):
    x_tile_max = tl.max(x_tile_abs, axis=axis, keep_dims=True)
    x_tile_max = tl.maximum(x_tile_max, 1e-4)
    x_scales_tile = FP8_MAX / x_tile_max
    x_fp8_tile = x_tile * x_scales_tile
    x_fp8_tile = tl.clamp(x_fp8_tile, min=-FP8_MAX, max=FP8_MAX)
    return x_fp8_tile, x_scales_tile


# Unified blockwise quantize kernel with segment-aware offset mapping
@triton.jit
def quant_fp8_blockwise_kernel(
    x_ptr,
    x_fp8_ptr,
    x_scales_ptr,
    N,  # N dimension
    M_out,  # Output M (may be padded)
    group_offs_ptr,  # Original group offsets [B+1]
    padded_group_offs_ptr,  # Padded group offsets [B+1]
    num_groups,  # Number of groups
    BLOCK_SIZE: tl.constexpr,
    FP8_MAX: tl.constexpr,
    AXIS: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    # Find which group this block belongs to
    group_id = 0
    for g in range(num_groups):
        padded_start = tl.load(padded_group_offs_ptr + g)
        padded_end = tl.load(padded_group_offs_ptr + g + 1)
        block_start = pid_m * BLOCK_SIZE
        if block_start >= padded_start and block_start < padded_end:
            group_id = g

    # Get group boundaries
    orig_group_start = tl.load(group_offs_ptr + group_id)
    orig_group_end = tl.load(group_offs_ptr + group_id + 1)
    padded_group_start = tl.load(padded_group_offs_ptr + group_id)

    # Calculate offsets
    offs_m_out = tl.cast(pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE), tl.int64)
    offs_n = tl.cast(pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE), tl.int64)

    # Map output offset to input offset
    offs_m_in = orig_group_start + (offs_m_out - padded_group_start)

    # Load mask: valid if within original group and within N
    load_mask = (
        (offs_m_in[:, None] >= orig_group_start)
        & (offs_m_in[:, None] < orig_group_end)
        & (offs_n[None, :] < N)
    )

    # Load from input
    x_ptrs = x_ptr + offs_m_in[:, None] * N + offs_n[None, :]
    x_tile = tl.load(x_ptrs, mask=load_mask, other=0.0).to(tl.float32)
    x_tile_abs = tl.abs(x_tile)

    x_fp8_tile, x_scales_tile = compute_scale_and_quant(x_tile, x_tile_abs, AXIS, FP8_MAX)

    # Store to output
    x_fp8_ptrs = x_fp8_ptr + offs_m_out[:, None] * N + offs_n[None, :]
    store_mask = offs_n[None, :] < N
    tl.store(x_fp8_ptrs, x_fp8_tile.to(x_fp8_ptr.dtype.element_ty), mask=store_mask)

    # Store scale (different layout for axis=0 vs axis=1)
    if AXIS == 1:
        scale_offs = offs_m_out * tl.cdiv(N, BLOCK_SIZE) + pid_n
        scale_mask = offs_m_out < M_out
    else:
        scale_offs = pid_m * N + offs_n
        scale_mask = offs_n < N

    x_scales_tile_inv = tl.reshape(1.0 / x_scales_tile, BLOCK_SIZE)
    tl.store(
        x_scales_ptr + scale_offs,
        x_scales_tile_inv,
        mask=scale_mask,
    )


# w_ptr         [B, M, N]
# w_fp8_ptr     [B, M, N] FP8
# w_scales_ptr  [B, M // BLOCK_SIZE, N // BLOCK_SIZE] FP32
@triton.jit
def quant_fp8_blockwise_for_weight_kernel(
    w_ptr,
    w_fp8_ptr,
    w_scales_ptr,
    M,
    N,
    BLOCK_SIZE: tl.constexpr,
    FP8_MAX: tl.constexpr,
):
    bid = tl.program_id(axis=0)
    pid_m = tl.program_id(axis=1)
    pid_n = tl.program_id(axis=2)

    batch_offset_w = bid * M * N
    batch_offset_scales = bid * tl.cdiv(M, BLOCK_SIZE) * tl.cdiv(N, BLOCK_SIZE)

    offs_m = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs_n = pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)

    # Load [BLOCK_SIZE, BLOCK_SIZE]
    w_ptrs = w_ptr + batch_offset_w + offs_m[:, None] * N + offs_n[None, :]
    w_tile = tl.load(w_ptrs, mask=mask, other=0.0).to(tl.float32)

    w_tile_abs = tl.abs(w_tile)
    w_tile_max = tl.max(w_tile_abs)  # [1]
    w_tile_max = tl.maximum(w_tile_max, 1e-4)
    w_scales = FP8_MAX / w_tile_max
    w_fp8_tile = w_tile * w_scales
    w_fp8_tile = tl.clamp(w_fp8_tile, min=-FP8_MAX, max=FP8_MAX)

    # Store
    w_fp8_ptrs = w_fp8_ptr + batch_offset_w + offs_m[:, None] * N + offs_n[None, :]
    tl.store(w_fp8_ptrs, w_fp8_tile.to(w_fp8_ptr.dtype.element_ty), mask=mask)
    # Store scale
    scale_offs = batch_offset_scales + pid_m * tl.cdiv(N, BLOCK_SIZE) + pid_n
    w_scales_inv = 1.0 / w_scales
    tl.store(w_scales_ptr + scale_offs, w_scales_inv)
