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
    """Quant FP8 blockwise. PSHUFFLE_SCALES: when True, write scales already
    in the layout the GEMM kernel needs (saves a runtime .T.contiguous()):

      AXIS=1 (row-wise): default scales layout [M, K//block]; pre-shuffled is [K//block, M]
      AXIS=0 (col-wise): default scales layout [M//block, N]; pre-shuffled is [N, M//block]

    For AXIS=1 the GEMM wrappers already do A_scales.T.contiguous() — pre-shuffle
    eliminates that runtime transpose. AXIS=0 is unaffected by current GEMM
    wrappers, but we keep the symmetry for future use.
    """
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


# Blockwise quantize with fused transpose: single bf16 read, dual fp8 write.
# Useful for wgrad, where the col-wise quantized inputs (axis=0) need to be
# read along the M dim by the kernel; pre-transposing makes M-dim contiguous
# → kernel can use the K_CONTIGUOUS=True path → avoids strided FP8 loads.
@triton.jit
def quant_fp8_blockwise_with_xpose_kernel(
    x_ptr,        # bf16/fp16 input  [M, N]
    x_fp8_ptr,    # fp8 output       [M, N]  (regular layout)
    x_fp8_T_ptr,  # fp8 output       [N, M]  (transposed layout)
    x_scales_ptr, # fp32 scales — see PSHUFFLE_SCALES for layout
    M, N,
    BLOCK_SIZE: tl.constexpr,
    FP8_MAX: tl.constexpr,
    AXIS: tl.constexpr,
    PSHUFFLE_SCALES: tl.constexpr = False,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    offs_m = tl.cast(pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE), tl.int64)
    offs_n = tl.cast(pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE), tl.int64)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)

    # Single bf16 read
    x_ptrs = x_ptr + offs_m[:, None] * N + offs_n[None, :]
    x_tile = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)
    x_tile_abs = tl.abs(x_tile)

    x_fp8_tile, x_scales_tile = compute_scale_and_quant(x_tile, x_tile_abs, AXIS, FP8_MAX)
    x_fp8_tile_t = x_fp8_tile.to(x_fp8_ptr.dtype.element_ty)

    # Write 1: regular [M, N]
    x_fp8_ptrs = x_fp8_ptr + offs_m[:, None] * N + offs_n[None, :]
    tl.store(x_fp8_ptrs, x_fp8_tile_t, mask=mask)

    # Write 2: transposed [N, M]
    x_fp8_T_ptrs = x_fp8_T_ptr + offs_n[:, None] * M + offs_m[None, :]
    tl.store(x_fp8_T_ptrs, x_fp8_tile_t.T, mask=mask.T)

    # Scales — pre-shuffled layout when PSHUFFLE_SCALES=True (saves a runtime
    # .T.contiguous() in the GEMM wrapper)
    if AXIS == 1:
        if PSHUFFLE_SCALES:
            scale_offs = pid_n * M + offs_m
        else:
            scale_offs = offs_m * tl.cdiv(N, BLOCK_SIZE) + pid_n
        scale_mask = offs_m < M
    else:
        if PSHUFFLE_SCALES:
            scale_offs = offs_n * tl.cdiv(M, BLOCK_SIZE) + pid_m
        else:
            scale_offs = pid_m * N + offs_n
        scale_mask = offs_n < N
    x_scales_tile_inv = tl.reshape(1.0 / x_scales_tile, BLOCK_SIZE)
    tl.store(x_scales_ptr + scale_offs, x_scales_tile_inv, mask=scale_mask)


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


# Same as quant_fp8_blockwise_segment_m_kernel but ALSO writes the FP8 transposed
# output [N, M_padded_max] in the same bf16-read pass. The transposed layout has
# the M_padded dim contiguous → the variable-K bwd kernel can use A_K_CONTIGUOUS=True
# path, replacing strided loads.
@triton.jit
def quant_fp8_blockwise_segment_m_with_xpose_kernel(
    x_ptr,
    x_fp8_ptr,
    x_fp8_T_ptr,    # [N, M_padded_max] transposed FP8 output
    x_scales_ptr,
    group_offs_ptr,
    padded_group_offs_ptr,
    N,
    M_padded_max,    # constant for transpose stride
    num_groups,
    BLOCK_SIZE: tl.constexpr,
    FP8_MAX: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    M_padded = tl.load(padded_group_offs_ptr + num_groups)

    block_start = pid_m * BLOCK_SIZE
    if block_start >= M_padded:
        return

    group_id = 0
    for g in range(num_groups):
        padded_start = tl.load(padded_group_offs_ptr + g)
        padded_end = tl.load(padded_group_offs_ptr + g + 1)
        if block_start >= padded_start and block_start < padded_end:
            group_id = g

    orig_group_start = tl.load(group_offs_ptr + group_id)
    orig_group_end = tl.load(group_offs_ptr + group_id + 1)
    padded_group_start = tl.load(padded_group_offs_ptr + group_id)

    offs_m_out = tl.cast(pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE), tl.int64)
    offs_n = tl.cast(pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE), tl.int64)
    offs_m_in = orig_group_start + (offs_m_out - padded_group_start)

    mask = (
        (offs_m_in[:, None] >= orig_group_start)
        & (offs_m_in[:, None] < orig_group_end)
        & (offs_n[None, :] < N)
    )

    x_ptrs = x_ptr + offs_m_in[:, None] * N + offs_n[None, :]
    x_tile = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)
    x_tile_abs = tl.abs(x_tile)

    x_fp8_tile, x_scales_tile = compute_scale_and_quant(x_tile, x_tile_abs, 0, FP8_MAX)
    x_fp8_tile_t = x_fp8_tile.to(x_fp8_ptr.dtype.element_ty)

    # Write 1: regular [M_padded_max, N]
    x_fp8_ptrs = x_fp8_ptr + offs_m_out[:, None] * N + offs_n[None, :]
    out_mask = (offs_m_out[:, None] < M_padded) & (offs_n[None, :] < N)
    tl.store(x_fp8_ptrs, x_fp8_tile_t, mask=out_mask)

    # Write 2: transposed [N, M_padded_max] — M_padded dim contiguous
    x_fp8_T_ptrs = x_fp8_T_ptr + offs_n[:, None] * M_padded_max + offs_m_out[None, :]
    tl.store(x_fp8_T_ptrs, x_fp8_tile_t.T, mask=out_mask.T)

    scale_offs = pid_m * N + offs_n
    scale_mask = (pid_m < tl.cdiv(M_padded, BLOCK_SIZE)) & (offs_n < N)
    x_scales_tile_inv = tl.reshape(1.0 / x_scales_tile, BLOCK_SIZE)
    tl.store(x_scales_ptr + scale_offs, x_scales_tile_inv, mask=scale_mask)


# w_ptr         [B, M, N]
# w_fp8_ptr     [B, M, N] FP8
# w_scales_ptr  [B, M // BLOCK_SIZE, N // BLOCK_SIZE] FP32
@triton.jit
def quant_fp8_blockwise_for_weight_with_xpose_kernel(
    w_ptr,            # bf16/fp16 input  [B, M, N]
    w_fp8_ptr,        # fp8 output       [B, M, N]
    w_fp8_T_ptr,      # fp8 transposed   [B, N, M]
    w_scales_ptr,     # fp32 scales      [B, ceil(M/BLOCK), ceil(N/BLOCK)]
    M, N,
    BLOCK_SIZE: tl.constexpr,
    FP8_MAX: tl.constexpr,
):
    """Fused weight quant + transpose. Single bf16 read, dual fp8 write.

    Produces both [B, M, N] and [B, N, M] FP8 layouts so the caller can
    use the regular layout for FWD (B @ A^T pattern) and the transposed
    for DGRAD (grad_out @ B pattern, TN/RCR layout). Scales are produced
    once in [B, M_blocks, N_blocks] layout; the transposed view is just
    `w_scales.permute(0, 2, 1)` (zero-copy) on the caller side.
    """
    bid = tl.program_id(axis=0)
    pid_m = tl.program_id(axis=1)
    pid_n = tl.program_id(axis=2)

    batch_offset_w = bid * M * N
    batch_offset_scales = bid * tl.cdiv(M, BLOCK_SIZE) * tl.cdiv(N, BLOCK_SIZE)

    offs_m = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs_n = pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)

    # Single bf16 read
    w_ptrs = w_ptr + batch_offset_w + offs_m[:, None] * N + offs_n[None, :]
    w_tile = tl.load(w_ptrs, mask=mask, other=0.0).to(tl.float32)

    # One scale per [BLOCK_SIZE, BLOCK_SIZE] tile
    w_tile_max = tl.maximum(tl.max(tl.abs(w_tile)), 1e-4)
    w_scales = FP8_MAX / w_tile_max
    w_fp8_tile = tl.clamp(w_tile * w_scales, min=-FP8_MAX, max=FP8_MAX)
    w_fp8_tile_t = w_fp8_tile.to(w_fp8_ptr.dtype.element_ty)

    # Write 1: original layout [B, M, N]
    w_fp8_ptrs = w_fp8_ptr + batch_offset_w + offs_m[:, None] * N + offs_n[None, :]
    tl.store(w_fp8_ptrs, w_fp8_tile_t, mask=mask)

    # Write 2: transposed layout [B, N, M] — same data, swap (m, n) → (n, m)
    w_fp8_T_ptrs = w_fp8_T_ptr + batch_offset_w + offs_n[:, None] * M + offs_m[None, :]
    tl.store(w_fp8_T_ptrs, w_fp8_tile_t.T, mask=mask.T)

    # Single scales write
    scale_offs = batch_offset_scales + pid_m * tl.cdiv(N, BLOCK_SIZE) + pid_n
    tl.store(w_scales_ptr + scale_offs, 1.0 / w_scales)


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
