from typing import Tuple

import torch
import triton

from primus_turbo.triton.gemm.gemm_fp8_kernel import (
    gemm_fp8_blockwise_nn_kernel,
    gemm_fp8_blockwise_nt_kernel,
    gemm_fp8_blockwise_tn_kernel,
)
from primus_turbo.triton.quantize.quant_blockwise import (
    quant_fp8_blockwise_for_act_grad_kernel,
    quant_fp8_blockwise_for_weight_kernel,
    quant_fp8_blockwise_kernel,
)


def ceil_div(a, b):
    return (a + b - 1) // b


def quant_fp8_blockwise_impl(
    x: torch.Tensor,
    dtype: torch.dtype,
    axis: int,
    block_size: int = 128,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantization for fp8 blockwise.

    Quantizes a 2D tensor using blockwise scale along the specified axis.
    Assumes `x` is contiguous and 2D.

    Returns:
        x_fp8: FP8-quantized tensor.
        x_scales: Per-block scale tensor in float32.
    """

    assert x.is_contiguous() and x.dim() == 2, "Input must be 2D and contiguous"
    assert axis in (-2, -1, 0, 1), f"axis must be 0 or 1 (or -1, -2), got {axis}"
    axis = axis % 2  # Convert negative axis to positive

    M, N = x.shape
    x_fp8 = torch.empty((M, N), dtype=dtype, device=x.device)
    scales_shape = (triton.cdiv(M, block_size), N) if axis == 0 else (M, triton.cdiv(N, block_size))
    x_scales = torch.empty(scales_shape, dtype=torch.float32, device=x.device)

    grid = (triton.cdiv(M, block_size), triton.cdiv(N, block_size))
    quant_fp8_blockwise_kernel[grid](
        x,
        x_fp8,
        x_scales,
        M,
        N,
        block_size,
        torch.finfo(dtype).max,
        axis,
    )
    return x_fp8, x_scales


def quant_fp8_blockwise_for_weight_impl(
    w: torch.Tensor,
    dtype: torch.dtype,
    block_size: int = 128,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantization for fp8 blockwise (weight).

    Quantizes a 2D weight tensor using blockwise scales along both axes.
    Assumes `w` is contiguous and 2D.

    Returns:
        w_fp8: FP8-quantized weight tensor.
        w_scales: Per-block scale tensor in float32.
    """

    assert w.is_contiguous()
    M, N = w.shape
    w_fp8 = torch.empty(M, N, dtype=dtype, device=w.device)
    w_scales = torch.empty(
        ceil_div(M, block_size),
        ceil_div(N, block_size),
        dtype=torch.float32,
        device=w.device,
    )
    grid = (triton.cdiv(M, block_size), triton.cdiv(N, block_size))
    quant_fp8_blockwise_for_weight_kernel[grid](
        w,
        w_fp8,
        w_scales,
        M,
        N,
        block_size,
        torch.finfo(dtype).max,
    )
    return w_fp8, w_scales


def quant_fp8_blockwise_for_act_grad_impl(
    x: torch.Tensor,
    dtype: torch.dtype,
    block_size: int = 128,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Quantization for fp8 blockwise (activation grad).

    Quantizes a 2D tensor both row-wise and column-wise for activation gradients.
    Assumes `x` is contiguous and 2D.

    Returns:
        x_fp8_row: Row-wise quantized tensor.
        x_scales_row: Row-wise scale tensor (float32).
        x_fp8_col: Column-wise quantized tensor.
        x_scales_col: Column-wise scale tensor (float32).
    """
    assert x.is_contiguous()
    M, N = x.shape
    x_fp8_row = torch.empty(M, N, dtype=dtype, device=x.device)
    x_scales_row = torch.empty(M, ceil_div(N, block_size), dtype=torch.float32, device=x.device)
    x_fp8_col = torch.empty(M, N, dtype=dtype, device=x.device)
    x_scales_col = torch.empty(ceil_div(M, block_size), N, dtype=torch.float32, device=x.device)
    grid = (triton.cdiv(M, block_size), triton.cdiv(N, block_size))
    quant_fp8_blockwise_for_act_grad_kernel[grid](
        x,
        x_fp8_row,
        x_scales_row,
        x_fp8_col,
        x_scales_col,
        M,
        N,
        block_size,
        torch.finfo(dtype).max,
    )
    return x_fp8_row, x_scales_row, x_fp8_col, x_scales_col


# For FWD NT
def gemm_fp8_blockwise_nt_impl(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scales: torch.Tensor,
    b_scales: torch.Tensor,
    out_dtype: torch.dtype,
    block_size: int = 128,
) -> torch.Tensor:
    """
    Blockwise FP8 GEMM for A @ B.T (NT layout).
    """
    assert a.is_contiguous() and b.is_contiguous(), "Inputs must be contiguous"
    assert a_scales.is_contiguous() and b_scales.is_contiguous(), "Scales must be contiguous"
    M, K = a.shape
    N, K1 = b.shape
    assert K == K1, f"Incompatible K dims: {K} vs {K1}"

    out = torch.empty(M, N, dtype=out_dtype, device=a.device)

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]),
        triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )
    gemm_fp8_blockwise_nt_kernel[grid](a, b, out, a_scales, b_scales, M, N, K, BLOCK_SIZE_K=block_size)
    return out


# For DGrad NN
def gemm_fp8_blockwise_nn_impl(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scales: torch.Tensor,
    b_scales: torch.Tensor,
    out_dtype: torch.dtype,
    block_size: int = 128,
) -> torch.Tensor:
    """
    Blockwise FP8 GEMM for A @ B (NN layout).
    """
    assert a.is_contiguous() and b.is_contiguous(), "Inputs must be contiguous"
    assert a_scales.is_contiguous() and b_scales.is_contiguous(), "Scales must be contiguous"

    M, K = a.shape
    K1, N = b.shape
    assert K == K1, f"Incompatible K dims: {K} vs {K1}"

    out = torch.empty(M, N, dtype=out_dtype, device=a.device)
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]),
        triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )
    gemm_fp8_blockwise_nn_kernel[grid](a, b, out, a_scales, b_scales, M, N, K, BLOCK_SIZE_K=block_size)
    return out


# For WGrad TN
def gemm_fp8_blockwise_tn_impl(
    a: torch.Tensor,  # [k, m]
    b: torch.Tensor,  # [k, n]
    a_scales: torch.Tensor,  # [k//block_size, m]
    b_scales: torch.Tensor,  # [k//block_size, n]
    out_dtype: torch.dtype,
    block_size: int = 128,
) -> torch.Tensor:
    """
    Blockwise FP8 GEMM for A.T @ B (TN layout).
    """
    assert a.is_contiguous() and b.is_contiguous(), "Inputs must be contiguous"
    assert a_scales.is_contiguous() and b_scales.is_contiguous(), "Scales must be contiguous"

    K, M = a.shape
    K1, N = b.shape
    assert K == K1, f"Incompatible K dims: {K} vs {K1}"

    out = torch.empty(M, N, dtype=out_dtype, device=a.device)
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]),
        triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )
    gemm_fp8_blockwise_tn_kernel[grid](a, b, out, a_scales, b_scales, M, N, K, BLOCK_SIZE_K=block_size)
    return out
