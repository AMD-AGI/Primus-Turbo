###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from typing import Optional, Tuple

import torch
import triton
from torch.library import triton_op, wrap_triton

from primus_turbo.pytorch.core.low_precision import MXScalingRecipe, check_mxfp4_support
from primus_turbo.pytorch.ops.hadamard_transform import get_rht_matrix
from primus_turbo.triton.quantization.quant_blockwise import (
    quant_fp8_blockwise_for_weight_kernel,
    quant_fp8_blockwise_kernel,
    quant_fp8_blockwise_segment_m_kernel,
)
from primus_turbo.triton.quantization.quantization_mxfp4 import (
    dequantize_mxfp4_kernel,
    quantize_mxfp4_kernel,
)
from primus_turbo.triton.quantization.quantization_mxfp8 import (
    dequantize_mxfp8_kernel,
    quantize_mxfp8_kernel,
)


def ceil_div(a, b):
    return (a + b - 1) // b


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


@triton_op("primus_turbo::quant_fp8_blockwise_for_weight_impl", mutates_args=())
def quant_fp8_blockwise_for_weight_impl(
    w: torch.Tensor,
    dtype: torch.dtype,
    block_size: int = 128,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantization for fp8 blockwise (weight).

    Quantizes a 2D or 3D weight tensor using blockwise scales along both axes.
    Assumes `w` is contiguous and 2D or 3D.

    Returns:
        w_fp8: FP8-quantized weight tensor.
        w_scales: Per-block scale tensor in float32.
    """

    assert w.dim() in (2, 3)
    if not w.is_contiguous():
        w = w.contiguous()

    ori_dims = w.dim()
    if ori_dims == 2:
        B, M, N = 1, *w.shape
        w = w.unsqueeze(0)
    else:
        B, M, N = w.shape
    w_fp8 = torch.empty((B, M, N), dtype=dtype, device=w.device)
    w_scales = torch.empty(
        (B, ceil_div(M, block_size), ceil_div(N, block_size)),
        dtype=torch.float32,
        device=w.device,
    )
    grid = (B, triton.cdiv(M, block_size), triton.cdiv(N, block_size))
    wrap_triton(quant_fp8_blockwise_for_weight_kernel)[grid](
        w,
        w_fp8,
        w_scales,
        M,
        N,
        block_size,
        torch.finfo(dtype).max,
    )

    if ori_dims == 2:
        w_fp8 = w_fp8.squeeze(0)
        w_scales = w_scales.squeeze(0)
    return w_fp8, w_scales


@quant_fp8_blockwise_for_weight_impl.register_fake
def quant_fp8_blockwise_for_weight_impl_meta(
    w: torch.Tensor,
    dtype: torch.dtype,
    block_size: int = 128,
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert w.dim() in (2, 3)
    ori_dims = w.dim()
    if ori_dims == 2:
        B, M, N = 1, *w.shape
        w = w.unsqueeze(0)
    else:
        B, M, N = w.shape
    w_fp8 = torch.empty((B, M, N), dtype=dtype, device=w.device)
    w_scales = torch.empty(
        (B, ceil_div(M, block_size), ceil_div(N, block_size)),
        dtype=torch.float32,
        device=w.device,
    )
    if ori_dims == 2:
        w_fp8 = w_fp8.squeeze(0)
        w_scales = w_scales.squeeze(0)
    return w_fp8, w_scales


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


def padding_size(n: int, padding_align_size: int) -> int:
    return (n + padding_align_size - 1) // padding_align_size * padding_align_size - n


def quantize_mxfp8_impl(
    x: torch.Tensor,
    out_dtype: torch.dtype,
    axis: int,
    block_size: int,
    padding_align_size: Optional[int] = None,
    scaling_recipe: Optional[MXScalingRecipe] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.is_contiguous(), "The x tensor must be contiguous."
    assert x.dim() == 2, "The x must be 2D tensor."
    assert axis in (0, 1), "The axis must be 0 or 1."

    num_rows, row_length = x.size()

    scaling_recipe = MXScalingRecipe() if scaling_recipe is None else scaling_recipe

    trans = axis == 0

    if not trans:
        padded_num_rows, padded_row_length = num_rows, row_length
        if padding_align_size is not None:
            padded_row_length += padding_size(row_length, padding_align_size)

        assert (
            padded_row_length % block_size == 0
        ), f"The last dimension of the x tensor must be divisible by the block size but got {padded_row_length} % {block_size} != 0."

        if scaling_recipe.use_2d_block:
            assert (
                num_rows % block_size == 0
            ), f"The first dimension of the x tensor must be divisible by the block size when use 2D block but got {num_rows} % {block_size} != 0."

        scale_m, scale_n = padded_num_rows, padded_row_length // block_size

        y = torch.empty((padded_num_rows, padded_row_length), dtype=out_dtype, device=x.device)
        scale_inv = torch.empty(scale_m, scale_n, dtype=torch.uint8, device=x.device)
    else:
        padded_num_rows, padded_row_length = num_rows, row_length
        if padding_align_size is not None:
            padded_num_rows += padding_size(num_rows, padding_align_size)

        assert (
            padded_num_rows % block_size == 0
        ), f"The first dimension of the x tensor must be divisible by the block size but got {padded_num_rows} % {block_size} != 0."

        if scaling_recipe.use_2d_block:
            assert (
                padded_row_length % block_size == 0
            ), f"The last dimension of the x tensor must be divisible by the block size when use 2D block but got {padded_row_length} % {block_size} != 0."

        scale_m, scale_n = padded_num_rows // block_size, padded_row_length

        y = torch.empty((padded_row_length, padded_num_rows), dtype=out_dtype, device=x.device)
        scale_inv = torch.empty(scale_n, scale_m, dtype=torch.uint8, device=x.device)

    scale_stride_M, scale_stride_N = scale_inv.stride(0), scale_inv.stride(1)

    BLOCK_X = 64
    BLOCK_Y = 64
    GROUP_Y = block_size
    grid = lambda META: (
        triton.cdiv(padded_num_rows, META["BLOCK_Y"]) * triton.cdiv(padded_row_length, META["BLOCK_X"]),
    )
    quantize_mxfp8_kernel[grid](
        x,
        y,
        x.stride(0),
        x.stride(1),
        y.stride(0),
        y.stride(1),
        num_rows,
        row_length,
        padded_num_rows,
        padded_row_length,
        scale_inv,
        scale_stride_M,
        scale_stride_N,
        scale_m,
        scale_n,
        BLOCK_X,
        BLOCK_Y,
        GROUP_Y,
        trans,
        scaling_recipe.use_2d_block,
        block_size,
    )
    scale_inv = scale_inv.view(dtype=torch.float8_e8m0fnu)

    return y, scale_inv


def dequantize_mxfp8_impl(
    x: torch.Tensor, out_dtype: torch.dtype, axis: int, block_size: int, scale_inv: torch.Tensor
) -> torch.Tensor:
    assert x.is_contiguous(), "The x tensor must be contiguous."
    assert x.dim() == 2, "The x must be 2D tensor."
    assert scale_inv.dim() == 2, "The scale_inv must be 2D tensor."
    assert scale_inv.is_contiguous(), "The scale_inv tensor must be contiguous."
    assert axis in (0, 1), "The axis must be 0 or 1."
    SUPPORTED_OUT_DTYPES = [torch.float16, torch.bfloat16, torch.float32]
    assert (
        out_dtype in SUPPORTED_OUT_DTYPES
    ), f"The out dtype must be one of {SUPPORTED_OUT_DTYPES} but got {out_dtype}."

    trans = axis == 0

    num_rows, row_length = x.size()
    assert (
        row_length % block_size == 0
    ), "The last dimension of the x tensor must be divisible by the block size."

    # NOTE: triton can't canonicalize torch.float8_e8m0fnu, so we need to reinterpret it to torch.uint8.
    scale_inv = scale_inv.view(torch.uint8)

    scale_m, scale_n = scale_inv.size()
    if not trans:
        y = torch.empty((num_rows, row_length), dtype=out_dtype, device=x.device)
    else:
        y = torch.empty((row_length, num_rows), dtype=out_dtype, device=x.device)

    BLOCK_X = 64
    BLOCK_Y = 64
    GROUP_Y = 4
    grid = lambda META: (triton.cdiv(num_rows, META["BLOCK_Y"]) * triton.cdiv(row_length, META["BLOCK_X"]),)
    dequantize_mxfp8_kernel[grid](
        x,
        y,
        x.stride(0),
        x.stride(1),
        y.stride(0),
        y.stride(1),
        num_rows,
        row_length,
        scale_inv,
        scale_inv.stride(0),
        scale_inv.stride(1),
        scale_m,
        scale_n,
        BLOCK_X,
        BLOCK_Y,
        GROUP_Y,
        trans,
        block_size,
    )

    return y


def quantize_mxfp4_impl(
    x: torch.Tensor,
    out_dtype: torch.dtype,
    axis: int,
    block_size: int,
    padding_align_size: Optional[int] = None,
    scaling_recipe: Optional[MXScalingRecipe] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # NOTE: quantize fp4 kernel use the ISA which only avaiable on cdna4.
    mxfp4_support, reason = check_mxfp4_support()
    assert mxfp4_support, reason

    assert x.is_contiguous(), "The x tensor must be contiguous."
    assert x.dim() == 2, "The A must be 2D tensor."
    assert axis in (
        0,
        1,
    ), "The axis must be 0 or 1."
    assert (
        out_dtype == torch.float4_e2m1fn_x2
    ), f"The out dtype expect torch.float4_e2m1fn_x2 but got {out_dtype}"

    scaling_recipe = MXScalingRecipe() if scaling_recipe is None else scaling_recipe

    trans = axis == 0

    num_rows, row_length = x.size()

    if not trans:
        padded_num_rows, padded_row_length = num_rows, row_length
        if padding_align_size is not None:
            padded_row_length += padding_size(row_length, padding_align_size)

        assert (
            padded_row_length % block_size == 0
        ), f"The last dimension of the x tensor must be divisible by the block size but got {padded_row_length} % {block_size} != 0."

        if scaling_recipe.use_2d_block:
            assert (
                num_rows % block_size == 0
            ), f"The first dimension of the x tensor must be divisible by the block size when use 2D block but got {num_rows} % {block_size} != 0."

        scale_m, scale_n = padded_num_rows, padded_row_length // block_size

        y = torch.empty((padded_num_rows, padded_row_length // 2), dtype=torch.uint8, device=x.device)
        scale_inv = torch.empty(scale_m, scale_n, dtype=torch.uint8, device=x.device)
    else:
        padded_num_rows, padded_row_length = num_rows, row_length
        if padding_align_size is not None:
            padded_num_rows += padding_size(num_rows, padding_align_size)

        assert (
            padded_num_rows % block_size == 0
        ), f"The first dimension of the x tensor must be divisible by the block size but got {padded_num_rows} % {block_size} != 0."

        if scaling_recipe.use_2d_block:
            assert (
                padded_row_length % block_size == 0
            ), f"The last dimension of the x tensor must be divisible by the block size when use 2D block but got {padded_row_length} % {block_size} != 0."

        scale_m, scale_n = padded_num_rows // block_size, padded_row_length

        y = torch.empty((padded_row_length, padded_num_rows // 2), dtype=torch.uint8, device=x.device)
        scale_inv = torch.empty(scale_n, scale_m, dtype=torch.uint8, device=x.device)

    philox_seed, philox_offset = scaling_recipe.philox_seed, scaling_recipe.philox_offset
    if scaling_recipe.use_sr:
        if philox_seed is None:
            philox_seed = torch.randint(0, 2**31 - 1, (1,), device="cpu").item()
        if philox_offset is None:
            philox_offset = torch.randint(0, 2**31 - 1, (1,), device="cpu").item()

    if scaling_recipe.use_rht:
        hadamard_matrix = get_rht_matrix(x.dtype, x.device)

    BLOCK_X = 64
    BLOCK_Y = 64
    GROUP_Y = block_size
    grid = lambda META: (
        triton.cdiv(padded_num_rows, META["BLOCK_Y"]) * triton.cdiv(padded_row_length, META["BLOCK_X"]),
    )
    quantize_mxfp4_kernel[grid](
        x,
        y,
        x.stride(0),
        x.stride(1),
        y.stride(0),
        y.stride(1),
        num_rows,
        row_length,
        padded_num_rows,
        padded_row_length,
        scale_inv,
        scale_inv.stride(0),
        scale_inv.stride(1),
        scale_m,
        scale_n,
        hadamard_matrix if scaling_recipe.use_rht else None,
        hadamard_matrix.size(0) if scaling_recipe.use_rht else 0,
        philox_seed,
        philox_offset,
        BLOCK_X=BLOCK_X,
        BLOCK_Y=BLOCK_Y,
        GROUP_Y=GROUP_Y,
        TRANS=trans,
        USE_2D_BLOCK=scaling_recipe.use_2d_block,
        USE_SR=scaling_recipe.use_sr,
        USE_RHT=scaling_recipe.use_rht,
        MXFP4_BLOCK_SIZE=block_size,
    )

    y = y.view(torch.float4_e2m1fn_x2)
    scale_inv = scale_inv.view(dtype=torch.float8_e8m0fnu)

    return y, scale_inv


def dequantize_mxfp4_impl(
    x: torch.Tensor,
    out_dtype: torch.dtype,
    axis: int,
    block_size: int,
    scale_inv: torch.Tensor,
) -> torch.Tensor:
    assert x.is_contiguous(), "The x tensor must be contiguous."
    assert x.dim() == 2, "The x must be 2D tensor."
    assert scale_inv.dim() == 2, "The scale_inv must be 2D tensor."
    assert scale_inv.is_contiguous(), "The scale_inv tensor must be contiguous."
    assert axis in (
        0,
        1,
    ), "The axis must be 0 or 1."
    assert x.dtype == torch.float4_e2m1fn_x2, f"The x dtype must be torch.float4_e2m1fn_x2 but got {x.dtype}."
    SUPPORTED_OUT_DTYPES = [torch.float16, torch.bfloat16, torch.float32]
    assert (
        out_dtype in SUPPORTED_OUT_DTYPES
    ), f"The out dtype must be one of {SUPPORTED_OUT_DTYPES} but got {out_dtype}."

    trans = axis == 0

    num_rows, row_length = x.size()
    # NOTE: x is packed in last dimension
    row_length = row_length * 2
    assert (
        row_length % block_size == 0
    ), "The last dimension of the x tensor must be divisible by the block size."

    # NOTE: triton can't canonicalize torch.float8_e8m0fnu, so we need to reinterpret it to torch.uint8.
    scale_inv = scale_inv.view(torch.uint8)

    scale_m, scale_n = scale_inv.size()
    if not trans:
        y = torch.empty((num_rows, row_length), dtype=out_dtype, device=x.device)
    else:
        y = torch.empty((row_length, num_rows), dtype=out_dtype, device=x.device)

    BLOCK_X = 64
    BLOCK_Y = 64
    GROUP_Y = 4
    grid = lambda META: (triton.cdiv(num_rows, META["BLOCK_Y"]) * triton.cdiv(row_length, META["BLOCK_X"]),)
    dequantize_mxfp4_kernel[grid](
        x.view(torch.uint8),
        y,
        x.stride(0),
        x.stride(1),
        y.stride(0),
        y.stride(1),
        num_rows,
        row_length,
        scale_inv,
        scale_inv.stride(0),
        scale_inv.stride(1),
        scale_m,
        scale_n,
        BLOCK_X=BLOCK_X,
        BLOCK_Y=BLOCK_Y,
        GROUP_Y=GROUP_Y,
        TRANS=trans,
        MXFP4_BLOCK_SIZE=block_size,
    )

    return y
