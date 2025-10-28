from typing import Tuple

import torch
import triton
from torch.library import triton_op, wrap_triton

from primus_turbo.triton.quantize.quant_blockwise import (
    quant_fp8_blockwise_kernel,
    quant_fp8_blockwise_segment_m_kernel,
)

"""
Blockwise Quantization Kernel
"""


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
