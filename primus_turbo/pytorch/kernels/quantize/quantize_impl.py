###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from typing import Optional, Tuple

import torch
import triton
from torch.library import triton_op, wrap_triton

from primus_turbo.triton.quantize.quant_blockwise import (
    quant_fp8_blockwise_kernel,
    quant_fp8_blockwise_segment_m_kernel,
)
from primus_turbo.triton.quantize.quantize_mxfp8 import (
    dequantize_mxfp8_kernel,
    quantize_mxfp8_kernel,
)


def quantize_fp8_tensorwise_impl(
    x: torch.Tensor, out_dtype: torch.dtype, scale: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize FP8 Tensor-Wise
    """
    x_fp8, scale_inv = torch.ops.primus_turbo_cpp_extension.quantize_fp8_tensorwise(x, out_dtype, scale)
    return x_fp8, scale_inv


def quantize_fp8_rowwise_impl(
    x: torch.Tensor, out_dtype: torch.dtype, axis: int, scale: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize FP8 Row-Wise
    """
    if not x.is_contiguous():
        x = x.contiguous()
    x_fp8, scale_inv = torch.ops.primus_turbo_cpp_extension.quantize_fp8_rowwise(x, out_dtype, axis, scale)
    return x_fp8, scale_inv


def dequantize_fp8_tensorwise_impl(x: torch.Tensor, out_dtype: torch.dtype, scale_inv: torch.Tensor):
    """
    DeQuantize FP8 Tensor-Wise
    """
    return torch.ops.primus_turbo_cpp_extension.dequantize_fp8_tensorwise(x, scale_inv, out_dtype)


def dequantize_fp8_rowwise_impl(x: torch.Tensor, out_dtype: torch.dtype, axis: int, scale_inv: torch.Tensor):
    """
    DeQuantize FP8 Row-Wise
    """
    raise NotImplementedError(f"Un-impl")


@triton_op("primus_turbo::quant_fp8_blockwise_impl", mutates_args=())
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
    wrap_triton(quant_fp8_blockwise_kernel)[grid](
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


@quant_fp8_blockwise_impl.register_fake
def quant_fp8_blockwise_impl_meta(
    x: torch.Tensor,
    dtype: torch.dtype,
    axis: int,
    block_size: int = 128,
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.dim() == 2, "Input must be 2D"
    assert axis in (-2, -1, 0, 1), f"axis must be 0 or 1 (or -1, -2), got {axis}"
    axis = axis % 2  # Convert negative axis to positive
    M, N = x.shape
    x_fp8 = torch.empty((M, N), dtype=dtype, device=x.device)
    scales_shape = (triton.cdiv(M, block_size), N) if axis == 0 else (M, triton.cdiv(N, block_size))
    x_scales = torch.empty(scales_shape, dtype=torch.float32, device=x.device)
    return x_fp8, x_scales


def quant_fp8_blockwise_segment_m_impl(
    x: torch.Tensor,
    batch_size: int,
    seg_lens: torch.Tensor,
    seg_indptr: torch.Tensor,
    scales_seg_indptr: torch.Tensor,
    dtype: torch.dtype,
    block_size: int = 128,
):
    assert x.is_contiguous() and x.dim() == 2, "Input must be 2D and contiguous"
    M, N = x.shape
    x_fp8 = torch.empty((M, N), dtype=dtype, device=x.device)

    scales_shape = (
        triton.cdiv(M, block_size) + batch_size,
        N,
    )  # M dim add batchsize.
    x_scales = torch.empty(scales_shape, dtype=torch.float32, device=x.device)
    grid = (triton.cdiv(M, block_size) + seg_lens.shape[0], triton.cdiv(N, block_size))
    quant_fp8_blockwise_segment_m_kernel[grid](
        x,
        x_fp8,
        x_scales,
        N,
        batch_size,
        seg_indptr,
        scales_seg_indptr,
        block_size,
        torch.finfo(dtype).max,
    )
    return x_fp8, x_scales


def quantize_mxfp8_impl(
    x: torch.Tensor, out_dtype: torch.dtype, block_size: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.is_contiguous(), "The x tensor must be contiguous"

    row_length = x.shape[-1]
    num_rows = x.numel() // row_length
    assert (
        row_length % block_size == 0
    ), "The last dimension of the x tensor must be divisible by the block size ."

    x_2d_view = x.view(num_rows, row_length)

    # Compute scale
    scale_M, scale_N = num_rows, row_length // block_size

    y = torch.empty(num_rows, row_length, dtype=out_dtype, device=x.device)
    scale_inv_shape = x.shape[:-1] + (scale_N,)
    scale_inv = torch.empty(scale_inv_shape, dtype=torch.uint8, device=x.device)
    scale_inv_2d_view = scale_inv.view(num_rows, scale_N)

    scale_stride_M, scale_stride_N = scale_inv_2d_view.stride(0), scale_inv_2d_view.stride(1)

    BLOCK_X = 64
    BLOCK_Y = 64
    GROUP_Y = block_size
    max_fp8 = torch.finfo(out_dtype).max
    grid = lambda META: (triton.cdiv(num_rows, META["BLOCK_Y"]) * triton.cdiv(row_length, META["BLOCK_X"]),)
    quantize_mxfp8_kernel[grid](
        x_2d_view,
        y,
        x_2d_view.stride(0),
        x_2d_view.stride(1),
        num_rows,
        row_length,
        scale_inv_2d_view,
        scale_stride_M,
        scale_stride_N,
        scale_M,
        scale_N,
        max_fp8,
        BLOCK_X,
        BLOCK_Y,
        GROUP_Y,
        block_size,
    )

    return y.view_as(x), scale_inv


def dequantize_mxfp8_impl(
    x: torch.Tensor, out_dtype: torch.dtype, block_size: int, scale_inv: torch.Tensor
) -> torch.Tensor:
    assert x.is_contiguous(), "The x tensor must be contiguous"
    assert scale_inv.is_contiguous(), "The scale_inv tensor must be contiguous"

    row_length = x.shape[-1]
    num_rows = x.numel() // row_length
    assert (
        row_length % block_size == 0
    ), "The last dimension of the x tensor must be divisible by the block size ."

    x_2d_view = x.view(num_rows, row_length)

    scale_inv_2d_view = scale_inv.view(num_rows, row_length // block_size)

    scale_M, scale_N = scale_inv_2d_view.shape
    out = torch.empty(x.shape, dtype=out_dtype, device=x.device)

    BLOCK_X = 64
    BLOCK_Y = 64
    GROUP_Y = 4

    grid = lambda META: (triton.cdiv(num_rows, META["BLOCK_Y"]) * triton.cdiv(row_length, META["BLOCK_X"]),)
    dequantize_mxfp8_kernel[grid](
        x_2d_view,
        out,
        x_2d_view.stride(0),
        x_2d_view.stride(1),
        num_rows,
        row_length,
        scale_inv_2d_view,
        scale_inv_2d_view.stride(0),
        scale_inv_2d_view.stride(1),
        scale_M,
        scale_N,
        BLOCK_X,
        BLOCK_Y,
        GROUP_Y,
        block_size,
    )

    return out.view_as(x)
