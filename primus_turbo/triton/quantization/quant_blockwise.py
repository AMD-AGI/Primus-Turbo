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
):
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
        scale_offs = offs_m * tl.cdiv(N, BLOCK_SIZE) + pid_n
        scale_mask = offs_m < M
    else:
        scale_offs = pid_m * N + offs_n
        scale_mask = offs_n < N
    x_scales_tile_inv = tl.reshape(1.0 / x_scales_tile, BLOCK_SIZE)
    tl.store(
        x_scales_ptr + scale_offs,
        x_scales_tile_inv,
        mask=scale_mask,
    )


@triton.jit
def quant_fp8_blockwise_dual_kernel(
    x_ptr,
    x_fp8_row_ptr,
    x_scales_row_ptr,
    x_fp8_col_ptr,
    x_scales_col_ptr,
    M,
    N,
    BLOCK_SIZE: tl.constexpr,
    FP8_MAX: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    offs_m = tl.cast(pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE), tl.int64)
    offs_n = tl.cast(pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE), tl.int64)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)

    x_ptrs = x_ptr + offs_m[:, None] * N + offs_n[None, :]
    x_tile = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)
    x_tile_abs = tl.abs(x_tile)

    x_fp8_row_tile, x_scales_row_tile = compute_scale_and_quant(x_tile, x_tile_abs, 1, FP8_MAX)
    x_fp8_col_tile, x_scales_col_tile = compute_scale_and_quant(x_tile, x_tile_abs, 0, FP8_MAX)

    x_fp8_row_ptrs = x_fp8_row_ptr + offs_m[:, None] * N + offs_n[None, :]
    tl.store(x_fp8_row_ptrs, x_fp8_row_tile.to(x_fp8_row_ptr.dtype.element_ty), mask=mask)

    x_fp8_col_ptrs = x_fp8_col_ptr + offs_m[:, None] * N + offs_n[None, :]
    tl.store(x_fp8_col_ptrs, x_fp8_col_tile.to(x_fp8_col_ptr.dtype.element_ty), mask=mask)

    row_scale_offs = offs_m * tl.cdiv(N, BLOCK_SIZE) + pid_n
    row_scale_mask = offs_m < M
    x_scales_row_tile_inv = tl.reshape(1.0 / x_scales_row_tile, BLOCK_SIZE)
    tl.store(
        x_scales_row_ptr + row_scale_offs,
        x_scales_row_tile_inv,
        mask=row_scale_mask,
    )

    col_scale_offs = pid_m * N + offs_n
    col_scale_mask = offs_n < N
    x_scales_col_tile_inv = tl.reshape(1.0 / x_scales_col_tile, BLOCK_SIZE)
    tl.store(
        x_scales_col_ptr + col_scale_offs,
        x_scales_col_tile_inv,
        mask=col_scale_mask,
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
    """Fused row + segment-padded col quant for grouped GEMM (dgrad + variable-K wgrad).

    One bf16 read of x produces: row-wise scaled output (for the fwd/dgrad GEMM) and
    segment-padded col-wise scaled output (for the variable-K wgrad GEMM). Each program
    handles one [BLOCK, BLOCK] tile in padded output space; row outputs are emitted only
    for valid input rows. Row scales are written pre-shuffled as [N_blocks, M_in] so the
    persistent fwd GEMM reads them by stride with no runtime .T.contiguous().
    """
    pid_m = tl.program_id(axis=0)  # padded M index
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

    mask_in = (offs_m_in[:, None] >= orig_start) & (offs_m_in[:, None] < orig_end) & (offs_n[None, :] < N)

    x = tl.load(x_ptr + offs_m_in[:, None] * N + offs_n[None, :], mask=mask_in, other=0.0).to(tl.float32)
    x_abs = tl.abs(x)

    col_max = tl.maximum(tl.max(x_abs, axis=0, keep_dims=True), 1e-4)
    col_scale = FP8_MAX / col_max
    x_fp8_col = tl.clamp(x * col_scale, min=-FP8_MAX, max=FP8_MAX).to(x_fp8_col_padded_ptr.dtype.element_ty)

    row_max = tl.maximum(tl.max(x_abs, axis=1, keep_dims=True), 1e-4)
    row_scale = FP8_MAX / row_max
    x_fp8_row = tl.clamp(x * row_scale, min=-FP8_MAX, max=FP8_MAX).to(x_fp8_row_ptr.dtype.element_ty)

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
    # Pshuffled [N_blocks, M_in]: matches the persistent fwd GEMM's scale order, so the
    # GEMM reads scales coalesced with no runtime .T.contiguous(). The HIP fast path
    # (quantize_fp8_blockwise_segment_m_row_col) emits the same layout.
    tl.store(
        x_scales_row_ptr + pid_n * M_in + offs_m_in,
        row_scale_inv,
        mask=row_scale_mask,
    )
