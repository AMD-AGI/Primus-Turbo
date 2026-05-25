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


# Standard blockwise quantize kernel
@triton.jit
def quant_fp8_blockwise_kernel(
    x_ptr,
    x_fp8_ptr,
    x_scales_ptr,
    M,
    N,
    BLOCK_SIZE: tl.constexpr,
    FP8_MAX: tl.constexpr,
    AXIS: tl.constexpr,
    PSHUFFLE_SCALES: tl.constexpr = False,  # if True, write scales in transposed-block layout
):
    """Blockwise FP8 quant. PSHUFFLE_SCALES=True writes scales pre-transposed to the GEMM layout."""
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    offs_m = tl.cast(pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE), tl.int64)
    offs_n = tl.cast(pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE), tl.int64)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)

    # Load [BLOCK_SIZE, BLOCK_SIZE]
    x_ptrs = x_ptr + offs_m[:, None] * N + offs_n[None, :]
    x_tile = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)
    x_tile_abs = tl.abs(x_tile)

    x_fp8_tile, x_scales_tile = compute_scale_and_quant(x_tile, x_tile_abs, AXIS, FP8_MAX)

    # Store output
    x_fp8_ptrs = x_fp8_ptr + offs_m[:, None] * N + offs_n[None, :]
    tl.store(x_fp8_ptrs, x_fp8_tile.to(x_fp8_ptr.dtype.element_ty), mask=mask)

    # Store scale
    if AXIS == 1:
        if PSHUFFLE_SCALES:
            # Layout [K//block, M]: row index = pid_n, col index = offs_m
            scale_offs = pid_n * M + offs_m
        else:
            # Layout [M, K//block]: row index = offs_m, col index = pid_n
            scale_offs = offs_m * tl.cdiv(N, BLOCK_SIZE) + pid_n
        scale_mask = offs_m < M
    else:
        if PSHUFFLE_SCALES:
            # Layout [N, M//block]: row index = offs_n, col index = pid_m
            scale_offs = offs_n * tl.cdiv(M, BLOCK_SIZE) + pid_m
        else:
            # Layout [M//block, N]: row index = pid_m, col index = offs_n
            scale_offs = pid_m * N + offs_n
        scale_mask = offs_n < N
    x_scales_tile_inv = tl.reshape(1.0 / x_scales_tile, BLOCK_SIZE)
    tl.store(
        x_scales_ptr + scale_offs,
        x_scales_tile_inv,
        mask=scale_mask,
    )


# Fused row + col-xpose blockwise quant for grad_out: one bf16 read serves dgrad-NT and K-contig wgrad.
@triton.jit
def quant_fp8_blockwise_row_col_xpose_kernel(
    x_ptr,
    x_fp8_row_ptr,
    x_fp8_col_T_ptr,
    x_scales_row_ptr,
    x_scales_col_ptr,
    M, N,
    BLOCK_SIZE: tl.constexpr,
    FP8_MAX: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    offs_m = tl.cast(pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE), tl.int64)
    offs_n = tl.cast(pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE), tl.int64)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)

    x = tl.load(x_ptr + offs_m[:, None] * N + offs_n[None, :], mask=mask, other=0.0).to(tl.float32)
    x_abs = tl.abs(x)

    row_max = tl.maximum(tl.max(x_abs, axis=1, keep_dims=True), 1e-4)
    row_scale = FP8_MAX / row_max
    x_fp8_row = tl.clamp(x * row_scale, min=-FP8_MAX, max=FP8_MAX).to(x_fp8_row_ptr.dtype.element_ty)

    col_max = tl.maximum(tl.max(x_abs, axis=0, keep_dims=True), 1e-4)
    col_scale = FP8_MAX / col_max
    x_fp8_col = tl.clamp(x * col_scale, min=-FP8_MAX, max=FP8_MAX).to(x_fp8_col_T_ptr.dtype.element_ty)

    tl.store(x_fp8_row_ptr + offs_m[:, None] * N + offs_n[None, :], x_fp8_row, mask=mask)
    tl.store(x_fp8_col_T_ptr + offs_n[:, None] * M + offs_m[None, :], x_fp8_col.T, mask=mask.T)

    # Row scales pre-shuffled to [N//block, M].
    row_scale_inv = tl.reshape(1.0 / row_scale, BLOCK_SIZE)
    tl.store(x_scales_row_ptr + pid_n * M + offs_m, row_scale_inv, mask=offs_m < M)

    col_scale_inv = tl.reshape(1.0 / col_scale, BLOCK_SIZE)
    tl.store(x_scales_col_ptr + pid_m * N + offs_n, col_scale_inv, mask=offs_n < N)


# Blockwise quantize with segment padding
# Reads from original tensor, writes to padded tensor with segment alignment
@triton.jit
def quant_fp8_blockwise_segment_m_kernel(
    x_ptr,  # Input tensor [M_in, N]
    x_fp8_ptr,  # Output tensor [M_out, N] (padded)
    x_scales_ptr,  # Output scales [M_out // BLOCK_SIZE, N]
    group_offs_ptr,  # Original group offsets [B+1]
    padded_group_offs_ptr,  # Padded group offsets [B+1]
    N,
    num_groups,
    BLOCK_SIZE: tl.constexpr,
    FP8_MAX: tl.constexpr,
):
    """
    Each program handles one BLOCK_SIZE x BLOCK_SIZE tile in the output (padded) space.
    Maps back to input space using group offsets.
    """
    pid_m = tl.program_id(axis=0)  # Block index in M dimension (padded)
    pid_n = tl.program_id(axis=1)  # Block index in N dimension

    # Get actual M_padded for bounds checking (last element of padded_group_offs)
    M_padded = tl.load(padded_group_offs_ptr + num_groups)

    # Early exit for out-of-bounds blocks (when using upper bound allocation)
    block_start = pid_m * BLOCK_SIZE
    if block_start >= M_padded:
        return

    # Find which group this block belongs to by searching padded_group_offs
    group_id = 0
    for g in range(num_groups):
        padded_start = tl.load(padded_group_offs_ptr + g)
        padded_end = tl.load(padded_group_offs_ptr + g + 1)
        if block_start >= padded_start and block_start < padded_end:
            group_id = g

    # Get group boundaries
    orig_group_start = tl.load(group_offs_ptr + group_id)
    orig_group_end = tl.load(group_offs_ptr + group_id + 1)
    padded_group_start = tl.load(padded_group_offs_ptr + group_id)

    # Calculate offsets in padded output space
    offs_m_out = tl.cast(pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE), tl.int64)
    offs_n = tl.cast(pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE), tl.int64)

    # Map output offset to input offset
    offs_m_in = orig_group_start + (offs_m_out - padded_group_start)

    # Mask: valid if within original group and within N
    mask = (
        (offs_m_in[:, None] >= orig_group_start)
        & (offs_m_in[:, None] < orig_group_end)
        & (offs_n[None, :] < N)
    )

    # Load from input (using input offsets)
    x_ptrs = x_ptr + offs_m_in[:, None] * N + offs_n[None, :]
    x_tile = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)
    x_tile_abs = tl.abs(x_tile)

    # Compute scale and quantize
    x_fp8_tile, x_scales_tile = compute_scale_and_quant(x_tile, x_tile_abs, 0, FP8_MAX)

    # Store to padded output (using output offsets)
    x_fp8_ptrs = x_fp8_ptr + offs_m_out[:, None] * N + offs_n[None, :]
    # Output mask: check both M and N bounds
    out_mask = (offs_m_out[:, None] < M_padded) & (offs_n[None, :] < N)
    tl.store(x_fp8_ptrs, x_fp8_tile.to(x_fp8_ptr.dtype.element_ty), mask=out_mask)

    # Store scale
    scale_offs = pid_m * N + offs_n
    scale_mask = (pid_m < tl.cdiv(M_padded, BLOCK_SIZE)) & (offs_n < N)
    x_scales_tile_inv = tl.reshape(1.0 / x_scales_tile, BLOCK_SIZE)
    tl.store(
        x_scales_ptr + scale_offs,
        x_scales_tile_inv,
        mask=scale_mask,
    )


# Fused row + segment-padded-col quant of grouped grad_out: one bf16 read
# serves dgrad (row-wise) and variable-K wgrad (col-wise, segment-padded).
@triton.jit
def quant_fp8_blockwise_segment_m_row_col_kernel(
    x_ptr,
    x_fp8_row_ptr,
    x_fp8_col_padded_ptr,
    x_scales_row_ptr,
    x_scales_col_padded_ptr,
    group_offs_ptr,
    padded_group_offs_ptr,
    M_in,
    N,
    num_groups,
    BLOCK_SIZE: tl.constexpr,
    FP8_MAX: tl.constexpr,
):
    """Each program handles one [BLOCK, BLOCK] tile in padded output space; row
    outputs are emitted only for valid input rows."""
    pid_m = tl.program_id(axis=0)   # padded M index
    pid_n = tl.program_id(axis=1)

    M_padded = tl.load(padded_group_offs_ptr + num_groups)
    block_start = pid_m * BLOCK_SIZE
    if block_start >= M_padded:
        return

    # Find group containing this padded tile
    group_id = 0
    for g in range(num_groups):
        ps = tl.load(padded_group_offs_ptr + g)
        pe = tl.load(padded_group_offs_ptr + g + 1)
        if block_start >= ps and block_start < pe:
            group_id = g

    orig_start = tl.load(group_offs_ptr + group_id)
    orig_end = tl.load(group_offs_ptr + group_id + 1)
    pad_start = tl.load(padded_group_offs_ptr + group_id)

    offs_m_out = tl.cast(pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE), tl.int64)
    offs_n = tl.cast(pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE), tl.int64)
    offs_m_in = orig_start + (offs_m_out - pad_start)

    mask_in = (
        (offs_m_in[:, None] >= orig_start)
        & (offs_m_in[:, None] < orig_end)
        & (offs_n[None, :] < N)
    )

    x = tl.load(
        x_ptr + offs_m_in[:, None] * N + offs_n[None, :], mask=mask_in, other=0.0
    ).to(tl.float32)
    x_abs = tl.abs(x)

    col_max = tl.maximum(tl.max(x_abs, axis=0, keep_dims=True), 1e-4)
    col_scale = FP8_MAX / col_max
    x_fp8_col = tl.clamp(x * col_scale, min=-FP8_MAX, max=FP8_MAX).to(
        x_fp8_col_padded_ptr.dtype.element_ty
    )

    row_max = tl.maximum(tl.max(x_abs, axis=1, keep_dims=True), 1e-4)
    row_scale = FP8_MAX / row_max
    x_fp8_row = tl.clamp(x * row_scale, min=-FP8_MAX, max=FP8_MAX).to(
        x_fp8_row_ptr.dtype.element_ty
    )

    out_mask_pad = (offs_m_out[:, None] < M_padded) & (offs_n[None, :] < N)
    tl.store(
        x_fp8_col_padded_ptr + offs_m_out[:, None] * N + offs_n[None, :],
        x_fp8_col,
        mask=out_mask_pad,
    )

    tl.store(
        x_fp8_row_ptr + offs_m_in[:, None] * N + offs_n[None, :],
        x_fp8_row,
        mask=mask_in,
    )

    col_scale_inv = tl.reshape(1.0 / col_scale, BLOCK_SIZE)
    col_scale_mask = (pid_m < tl.cdiv(M_padded, BLOCK_SIZE)) & (offs_n < N)
    tl.store(
        x_scales_col_padded_ptr + pid_m * N + offs_n,
        col_scale_inv,
        mask=col_scale_mask,
    )

    row_scale_inv = tl.reshape(1.0 / row_scale, BLOCK_SIZE)
    row_scale_mask = (offs_m_in >= orig_start) & (offs_m_in < orig_end) & (offs_m_in < M_in)
    # Pshuffled [N_blocks, M_in]: matches the persistent fwd GEMM's scale order.
    tl.store(
        x_scales_row_ptr + pid_n * M_in + offs_m_in,
        row_scale_inv,
        mask=row_scale_mask,
    )


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
