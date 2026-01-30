###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Optimized FP8 tensorwise quantization with fused transpose."""

import torch
import triton
import triton.language as tl


@triton.jit
def _fast_amax_kernel(
    x_ptr,
    partial_amax_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Optimized amax kernel."""
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    local_amax = tl.max(tl.abs(x))
    tl.store(partial_amax_ptr + pid, local_amax)


@triton.jit
def _final_reduce_and_scale_kernel(
    partial_amax_ptr,
    scale_ptr,
    n_partials,
    fp8_max: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Final reduction and scale computation in one kernel."""
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < n_partials
    partial = tl.load(partial_amax_ptr + offs, mask=mask, other=0.0)
    amax = tl.max(partial)
    # Compute scale = fp8_max / max(amax, eps)
    eps = 1e-12
    amax_clamped = tl.maximum(amax, eps)
    scale = fp8_max / amax_clamped
    tl.store(scale_ptr, scale)


def compute_scale_fast(x: torch.Tensor, fp8_max: float) -> torch.Tensor:
    """Compute scale using fused Triton kernels."""
    n_elements = x.numel()
    BLOCK_SIZE = 16384
    n_blocks = triton.cdiv(n_elements, BLOCK_SIZE)

    partial_amax = torch.empty(n_blocks, dtype=torch.float32, device=x.device)
    scale = torch.empty(1, dtype=torch.float32, device=x.device)

    # First pass: partial amax
    _fast_amax_kernel[(n_blocks,)](x, partial_amax, n_elements, BLOCK_SIZE=BLOCK_SIZE)

    # Second pass: final reduce + scale (single block)
    # Find next power of 2 >= n_blocks for final reduction
    final_block_size = 1
    while final_block_size < n_blocks:
        final_block_size *= 2
    final_block_size = min(final_block_size, 1024)  # Cap at 1024

    _final_reduce_and_scale_kernel[(1,)](
        partial_amax, scale, n_blocks, fp8_max=fp8_max, BLOCK_SIZE=final_block_size
    )

    return scale


def fast_amax(x: torch.Tensor) -> torch.Tensor:
    """Compute amax using optimized two-level Triton reduction."""
    n_elements = x.numel()
    BLOCK_SIZE = 16384
    n_blocks = triton.cdiv(n_elements, BLOCK_SIZE)

    partial_amax = torch.empty(n_blocks, dtype=torch.float32, device=x.device)
    _fast_amax_kernel[(n_blocks,)](x, partial_amax, n_elements, BLOCK_SIZE=BLOCK_SIZE)

    return partial_amax.max()


# TE-style fused kernel: cast + transpose + atomic amax
# Uses prev_amax to compute scale internally - no extra kernel needed!
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 256}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 128}, num_warps=8, num_stages=2),
    ],
    key=["M", "N"],
)
@triton.jit
def _cast_transpose_with_amax_kernel(
    x_ptr,
    y_ptr,
    y_t_ptr,
    prev_amax_ptr,  # Input: amax from previous iteration (or 0 for first call)
    amax_ptr,  # Output: amax for next iteration
    scale_inv_ptr,
    M,
    N,
    stride_xm,
    stride_xn,
    stride_ym,
    stride_yn,
    stride_ytm,
    stride_ytn,
    FP8_MAX: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    TE-style fused kernel: cast + transpose + atomic amax.
    Computes scale from prev_amax internally - completely fused!
    """
    pid = tl.program_id(0)

    # Simple 2D grid
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    # Compute scale from prev_amax (fused inside kernel!)
    prev_amax = tl.load(prev_amax_ptr)
    eps = 1e-12
    prev_amax_clamped = tl.maximum(prev_amax, eps)
    scale = FP8_MAX / prev_amax_clamped

    # Block offsets
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Bounds check
    mask = (rm[:, None] < M) & (rn[None, :] < N)

    # Load input (coalesced along N)
    x_ptrs = x_ptr + rm[:, None] * stride_xm + rn[None, :] * stride_xn
    x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)

    # Quantize
    x_scaled = x * scale
    x_fp8 = tl.clamp(x_scaled, -FP8_MAX, FP8_MAX).to(y_ptr.dtype.element_ty)

    # Store regular output (coalesced along N)
    y_ptrs = y_ptr + rm[:, None] * stride_ym + rn[None, :] * stride_yn
    tl.store(y_ptrs, x_fp8, mask=mask)

    # Store transposed output - transpose mask for correct bounds
    t_mask = (rn[:, None] < N) & (rm[None, :] < M)
    y_t_ptrs = y_t_ptr + rn[:, None] * stride_ytm + rm[None, :] * stride_ytn
    # Transpose the data: x_fp8 is [BLOCK_M, BLOCK_N], we need [BLOCK_N, BLOCK_M]
    tl.store(y_t_ptrs, tl.trans(x_fp8), mask=t_mask)

    # Atomic amax for next iteration
    local_amax = tl.max(tl.abs(x))
    tl.atomic_max(amax_ptr, local_amax, sem="relaxed")

    # Store scale_inv (only first block)
    if pid == 0:
        tl.store(scale_inv_ptr, 1.0 / scale)


# Simple cast+transpose kernel without amax (for when scale is pre-computed)
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8),
    ],
    key=["M", "N"],
)
@triton.jit
def _cast_transpose_kernel(
    x_ptr,
    y_ptr,
    y_t_ptr,
    scale_ptr,
    scale_inv_ptr,
    M,
    N,
    stride_xm,
    stride_xn,
    stride_ym,
    stride_yn,
    stride_ytm,
    stride_ytn,
    FP8_MAX: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Fast FP8 cast + transpose kernel.
    """
    pid = tl.program_id(0)

    # Simple 2D grid
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    # Load scale
    scale = tl.load(scale_ptr)

    # Block offsets
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Bounds check
    mask = (rm[:, None] < M) & (rn[None, :] < N)

    # Load input
    x_ptrs = x_ptr + rm[:, None] * stride_xm + rn[None, :] * stride_xn
    x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)

    # Quantize
    x_scaled = x * scale
    x_fp8 = tl.clamp(x_scaled, -FP8_MAX, FP8_MAX).to(y_ptr.dtype.element_ty)

    # Store regular output
    y_ptrs = y_ptr + rm[:, None] * stride_ym + rn[None, :] * stride_yn
    tl.store(y_ptrs, x_fp8, mask=mask)

    # Store transposed: y_t[n, m] = x_fp8[m, n]
    y_t_ptrs = y_t_ptr + rn[None, :] * stride_ytm + rm[:, None] * stride_ytn
    tl.store(y_t_ptrs, x_fp8, mask=mask)

    # Store scale_inv (only first block)
    if pid == 0:
        tl.store(scale_inv_ptr, 1.0 / scale)


def quantize_fp8_tensorwise_with_transpose(
    x: torch.Tensor,
    out_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Quantize a 2D tensor to FP8 and return both the quantized tensor and its transpose.

    Args:
        x: Input tensor of shape (M, N)
        out_dtype: Output FP8 dtype (torch.float8_e4m3fn or torch.float8_e5m2)

    Returns:
        x_fp8: Quantized tensor of shape (M, N)
        scale_inv: Inverse scale tensor
        x_t_fp8: Transposed quantized tensor of shape (N, M)
    """
    assert x.ndim == 2, "Input must be 2D tensor"
    assert x.is_contiguous(), "Input must be contiguous"

    M, N = x.shape
    device = x.device
    fp8_max = torch.finfo(out_dtype).max

    # Compute scale (2 Triton kernels: amax + reduce)
    scale = compute_scale_fast(x, fp8_max)

    # Allocate outputs
    x_fp8 = torch.empty((M, N), dtype=out_dtype, device=device)
    x_t_fp8 = torch.empty((N, M), dtype=out_dtype, device=device)
    scale_inv = torch.empty(1, dtype=torch.float32, device=device)

    # Launch cast+transpose kernel
    grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),)

    _cast_transpose_kernel[grid](
        x,
        x_fp8,
        x_t_fp8,
        scale,
        scale_inv,
        M,
        N,
        x.stride(0),
        x.stride(1),
        x_fp8.stride(0),
        x_fp8.stride(1),
        x_t_fp8.stride(0),
        x_t_fp8.stride(1),
        FP8_MAX=fp8_max,
    )

    return x_fp8, scale_inv, x_t_fp8


def quantize_fp8_delayed_scaling(
    x: torch.Tensor,
    out_dtype: torch.dtype,
    prev_amax: torch.Tensor | None = None,
    next_amax: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    TE-style delayed scaling quantization - FULLY FUSED with persistent buffers!

    Args:
        x: Input tensor of shape (M, N)
        out_dtype: Output FP8 dtype
        prev_amax: Amax from previous iteration. If None, compute synchronously (first call).
        next_amax: Pre-allocated buffer for next amax (persistent, no allocation needed).

    Returns:
        x_fp8: Quantized tensor of shape (M, N)
        scale_inv: Inverse scale tensor
        x_t_fp8: Transposed quantized tensor of shape (N, M)
    """
    assert x.ndim == 2, "Input must be 2D tensor"
    assert x.is_contiguous(), "Input must be contiguous"

    M, N = x.shape
    device = x.device
    fp8_max = torch.finfo(out_dtype).max

    # First call: compute amax synchronously using full reduction
    if prev_amax is None:
        scale = compute_scale_fast(x, fp8_max)
        # Extract amax from scale: amax = fp8_max / scale
        prev_amax = torch.tensor([fp8_max], dtype=torch.float32, device=device) / scale

    # Allocate next_amax if not provided (for backward compatibility)
    if next_amax is None:
        next_amax = torch.zeros(1, dtype=torch.float32, device=device)

    # Allocate outputs
    x_fp8 = torch.empty((M, N), dtype=out_dtype, device=device)
    x_t_fp8 = torch.empty((N, M), dtype=out_dtype, device=device)
    scale_inv = torch.empty(1, dtype=torch.float32, device=device)

    # Autotune grid
    grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),)

    # Launch FULLY FUSED kernel: scale computation + cast + transpose + amax
    _cast_transpose_with_amax_kernel[grid](
        x,
        x_fp8,
        x_t_fp8,
        prev_amax,
        next_amax,
        scale_inv,
        M,
        N,
        x.stride(0),
        x.stride(1),
        x_fp8.stride(0),
        x_fp8.stride(1),
        x_t_fp8.stride(0),
        x_t_fp8.stride(1),
        FP8_MAX=fp8_max,
    )

    return x_fp8, scale_inv, x_t_fp8


@triton.jit
def _amax_to_scale_kernel(
    amax_ptr,
    scale_ptr,
    fp8_max: tl.constexpr,
):
    """Convert amax to scale in a single Triton kernel."""
    amax = tl.load(amax_ptr)
    eps = 1e-12
    amax_clamped = tl.maximum(amax, eps)
    scale = fp8_max / amax_clamped
    tl.store(scale_ptr, scale)


def amax_to_scale(amax: torch.Tensor, fp8_max: float) -> torch.Tensor:
    """Convert amax to scale using Triton kernel."""
    scale = torch.empty(1, dtype=torch.float32, device=amax.device)
    _amax_to_scale_kernel[(1,)](amax, scale, fp8_max=fp8_max)
    return scale
