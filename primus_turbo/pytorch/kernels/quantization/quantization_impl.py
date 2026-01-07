###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from typing import Optional, Tuple, Union

import torch
import triton
from torch.library import triton_op, wrap_triton

from primus_turbo.pytorch.core.low_precision import MXScalingRecipe, check_mxfp4_support
from primus_turbo.pytorch.kernels.quantization.hadamard_transform import get_rht_matrix
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
    with_trans: bool = False,
    scaling_recipe: Optional[MXScalingRecipe] = None,
    scaling_recipe_for_trans: Optional[MXScalingRecipe] = None,
) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    assert x.is_contiguous(), "The x tensor must be contiguous."
    assert x.dim() == 2, "The x must be 2D tensor."

    scaling_recipe = MXScalingRecipe() if scaling_recipe is None else scaling_recipe
    if with_trans:
        scaling_recipe_for_trans = MXScalingRecipe()
    else:
        scaling_recipe_for_trans = scaling_recipe

    if not with_trans:
        assert axis in (0, 1), "The axis must be 0 or 1 when with_trans is False."
    else:
        assert axis is None, "The axis must be None when with_trans is True."

    num_rows, row_length = x.size()

    use_rowwise = axis == 1
    use_colwise = axis == 0
    if with_trans:
        use_rowwise, use_colwise = True, True

    y_rowwise, y_colwise = None, None
    scale_inv_rowwise, scale_inv_colwise = None, None

    num_rows_padding_size, row_length_padding_size = 0, 0
    if padding_align_size is not None:
        num_rows_padding_size = padding_size(num_rows, padding_align_size)
        row_length_padding_size = padding_size(row_length, padding_align_size)

    if use_rowwise:
        padded_num_rows, padded_row_length = num_rows, row_length
        padded_row_length += row_length_padding_size

        assert (
            padded_row_length % block_size == 0
        ), f"The last dimension of the x tensor must be divisible by the block size but got {padded_row_length} % {block_size} != 0."

        if scaling_recipe.use_2d_block:
            assert (
                num_rows % block_size == 0
            ), f"The first dimension of the x tensor must be divisible by the block size when use 2D block but got {num_rows} % {block_size} != 0."

        scale_inv_rowwise_n_rows, scale_inv_rowwise_n_cols = padded_num_rows, padded_row_length // block_size

        y_rowwise = torch.empty((padded_num_rows, padded_row_length), dtype=out_dtype, device=x.device)
        scale_inv_rowwise = torch.empty(
            scale_inv_rowwise_n_rows, scale_inv_rowwise_n_cols, dtype=torch.uint8, device=x.device
        )

        if row_length_padding_size > 0 or num_rows_padding_size > 0:
            BLOCK_X = 64
            BLOCK_Y = 64
            GROUP_Y = block_size
            grid = lambda META: (
                triton.cdiv(padded_num_rows, META["BLOCK_Y"])
                * triton.cdiv(padded_row_length, META["BLOCK_X"]),
            )
            quantize_mxfp8_kernel[grid](
                x,
                y_rowwise,
                None,
                x.stride(0),
                x.stride(1),
                y_rowwise.stride(0),
                y_rowwise.stride(1),
                None,
                None,
                num_rows,
                row_length,
                padded_num_rows,
                padded_row_length,
                scale_inv_rowwise,
                scale_inv_colwise,
                scale_inv_rowwise.stride(0),
                scale_inv_rowwise.stride(1),
                None,
                None,
                scale_inv_rowwise_n_rows,
                scale_inv_rowwise_n_cols,
                None,
                None,
                BLOCK_X=BLOCK_X,
                BLOCK_Y=BLOCK_Y,
                GROUP_Y=GROUP_Y,
                USE_ROWWISE=use_rowwise,
                USE_COLWISE=False,
                ROWWISE_USE_2D_BLOCK=scaling_recipe.use_2d_block,
                COLWISE_USE_2D_BLOCK=False,
                MXFP8_BLOCK_SIZE=block_size,
            )

    if use_colwise:
        padded_num_rows, padded_row_length = num_rows, row_length
        padded_num_rows += num_rows_padding_size

        assert (
            padded_num_rows % block_size == 0
        ), f"The first dimension of the x tensor must be divisible by the block size but got {padded_num_rows} % {block_size} != 0."

        if scaling_recipe_for_trans.use_2d_block:
            assert (
                padded_row_length % block_size == 0
            ), f"The last dimension of the x tensor must be divisible by the block size when use 2D block but got {padded_row_length} % {block_size} != 0."

        scale_inv_colwise_n_rows, scale_inv_colwise_n_cols = padded_num_rows // block_size, padded_row_length

        y_colwise = torch.empty((padded_row_length, padded_num_rows), dtype=out_dtype, device=x.device)
        scale_inv_colwise = torch.empty(
            scale_inv_colwise_n_cols, scale_inv_colwise_n_rows, dtype=torch.uint8, device=x.device
        )

        if num_rows_padding_size > 0 or row_length_padding_size > 0:
            BLOCK_X = 64
            BLOCK_Y = 64
            GROUP_Y = block_size
            grid = lambda META: (
                triton.cdiv(padded_num_rows, META["BLOCK_Y"])
                * triton.cdiv(padded_row_length, META["BLOCK_X"]),
            )
            quantize_mxfp8_kernel[grid](
                x,
                None,
                y_colwise,
                x.stride(0),
                x.stride(1),
                None,
                None,
                y_colwise.stride(0),
                y_colwise.stride(1),
                num_rows,
                row_length,
                padded_num_rows,
                padded_row_length,
                scale_inv_rowwise,
                scale_inv_colwise,
                None,
                None,
                scale_inv_colwise.stride(0),
                scale_inv_colwise.stride(1),
                None,
                None,
                scale_inv_colwise_n_rows,
                scale_inv_colwise_n_cols,
                BLOCK_X=BLOCK_X,
                BLOCK_Y=BLOCK_Y,
                GROUP_Y=GROUP_Y,
                USE_ROWWISE=False,
                USE_COLWISE=use_colwise,
                ROWWISE_USE_2D_BLOCK=False,
                COLWISE_USE_2D_BLOCK=scaling_recipe_for_trans.use_2d_block,
                MXFP8_BLOCK_SIZE=block_size,
            )

    if num_rows_padding_size == 0 and row_length_padding_size == 0:
        BLOCK_X = 64
        BLOCK_Y = 64
        GROUP_Y = block_size
        grid = lambda META: (
            triton.cdiv(num_rows, META["BLOCK_Y"]) * triton.cdiv(row_length, META["BLOCK_X"]),
        )
        quantize_mxfp8_kernel[grid](
            x,
            y_rowwise,
            y_colwise,
            x.stride(0),
            x.stride(1),
            y_rowwise.stride(0) if y_rowwise is not None else None,
            y_rowwise.stride(1) if y_rowwise is not None else None,
            y_colwise.stride(0) if y_colwise is not None else None,
            y_colwise.stride(1) if y_colwise is not None else None,
            num_rows,
            row_length,
            num_rows,
            row_length,
            scale_inv_rowwise,
            scale_inv_colwise,
            scale_inv_rowwise.stride(0) if scale_inv_rowwise is not None else None,
            scale_inv_rowwise.stride(1) if scale_inv_rowwise is not None else None,
            scale_inv_colwise.stride(0) if scale_inv_colwise is not None else None,
            scale_inv_colwise.stride(1) if scale_inv_colwise is not None else None,
            scale_inv_rowwise_n_rows if scale_inv_rowwise is not None else None,
            scale_inv_rowwise_n_cols if scale_inv_rowwise is not None else None,
            scale_inv_colwise_n_rows if scale_inv_colwise is not None else None,
            scale_inv_colwise_n_cols if scale_inv_colwise is not None else None,
            BLOCK_X=BLOCK_X,
            BLOCK_Y=BLOCK_Y,
            GROUP_Y=GROUP_Y,
            USE_ROWWISE=use_rowwise,
            USE_COLWISE=use_colwise,
            ROWWISE_USE_2D_BLOCK=scaling_recipe.use_2d_block,
            COLWISE_USE_2D_BLOCK=scaling_recipe_for_trans.use_2d_block,
            MXFP8_BLOCK_SIZE=block_size,
        )

    return_list = []
    if use_rowwise:
        return_list += [y_rowwise, scale_inv_rowwise.view(dtype=torch.float8_e8m0fnu)]
    if use_colwise:
        return_list += [y_colwise, scale_inv_colwise.view(dtype=torch.float8_e8m0fnu)]

    return tuple(return_list)


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

    use_rowwise = axis == 1

    num_rows, row_length = x.size()
    assert (
        row_length % block_size == 0
    ), "The last dimension of the x tensor must be divisible by the block size."

    # NOTE: triton can't canonicalize torch.float8_e8m0fnu, so we need to reinterpret it to torch.uint8.
    scale_inv = scale_inv.view(torch.uint8)

    scale_m, scale_n = scale_inv.size()
    if use_rowwise:
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
        BLOCK_X=BLOCK_X,
        BLOCK_Y=BLOCK_Y,
        GROUP_Y=GROUP_Y,
        USE_ROWWISE=use_rowwise,
        MXFP8_BLOCK_SIZE=block_size,
    )

    return y


def quantize_mxfp4_impl(
    x: torch.Tensor,
    out_dtype: torch.dtype,
    axis: int,
    block_size: int,
    padding_align_size: Optional[int] = None,
    with_trans: bool = False,
    scaling_recipe: Optional[MXScalingRecipe] = None,
    scaling_recipe_for_trans: Optional[MXScalingRecipe] = None,
) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    # NOTE: quantize fp4 kernel use the ISA which only avaiable on cdna4.
    mxfp4_support, reason = check_mxfp4_support()
    assert mxfp4_support, reason

    scaling_recipe = MXScalingRecipe() if scaling_recipe is None else scaling_recipe
    if with_trans:
        scaling_recipe_for_trans = MXScalingRecipe()
    else:
        scaling_recipe_for_trans = scaling_recipe

    if not with_trans:
        assert axis in (0, 1), "The axis must be 0 or 1 when with_trans is False."
    else:
        assert axis is None, "The axis must be None when with_trans is True."

    assert (
        out_dtype == torch.float4_e2m1fn_x2
    ), f"The out dtype expect torch.float4_e2m1fn_x2 but got {out_dtype}"

    num_rows, row_length = x.size()

    use_rowwise = axis == 1
    use_colwise = axis == 0
    if with_trans:
        use_rowwise, use_colwise = True, True

    y_rowwise, y_colwise = None, None
    scale_inv_rowwise, scale_inv_colwise = None, None

    num_rows_padding_size, row_length_padding_size = 0, 0
    if padding_align_size is not None:
        num_rows_padding_size = padding_size(num_rows, padding_align_size)
        row_length_padding_size = padding_size(row_length, padding_align_size)

    if scaling_recipe.use_rht or scaling_recipe_for_trans.use_rht:
        hadamard_matrix = get_rht_matrix(x.dtype, x.device)

    rowwise_philox_seed, rowwise_philox_offset = scaling_recipe.philox_seed, scaling_recipe.philox_offset
    if use_rowwise:
        padded_num_rows, padded_row_length = num_rows, row_length
        padded_row_length += row_length_padding_size

        assert (
            padded_row_length % block_size == 0
        ), f"The last dimension of the x tensor must be divisible by the block size but got {padded_row_length} % {block_size} != 0."

        if scaling_recipe.use_2d_block:
            assert (
                num_rows % block_size == 0
            ), f"The first dimension of the x tensor must be divisible by the block size when use 2D block but got {num_rows} % {block_size} != 0."

        if scaling_recipe.use_sr:
            if rowwise_philox_seed is None:
                rowwise_philox_seed = torch.randint(0, 2**31 - 1, (1,), device="cpu").item()
            if rowwise_philox_offset is None:
                rowwise_philox_offset = torch.randint(0, 2**31 - 1, (1,), device="cpu").item()

        scale_inv_rowwise_n_rows, scale_inv_rowwise_n_cols = padded_num_rows, padded_row_length // block_size

        y_rowwise = torch.empty((padded_num_rows, padded_row_length // 2), dtype=torch.uint8, device=x.device)
        scale_inv_rowwise = torch.empty(
            scale_inv_rowwise_n_rows, scale_inv_rowwise_n_cols, dtype=torch.uint8, device=x.device
        )

        if num_rows_padding_size > 0 or row_length_padding_size > 0:
            BLOCK_X = 64
            BLOCK_Y = 64
            GROUP_Y = block_size
            grid = lambda META: (
                triton.cdiv(padded_num_rows, META["BLOCK_Y"])
                * triton.cdiv(padded_row_length, META["BLOCK_X"]),
            )
            quantize_mxfp4_kernel[grid](
                x,
                y_rowwise,
                None,
                x.stride(0),
                x.stride(1),
                y_rowwise.stride(0),
                y_rowwise.stride(1),
                None,
                None,
                num_rows,
                row_length,
                padded_num_rows,
                padded_row_length,
                scale_inv_rowwise,
                None,
                scale_inv_rowwise.stride(0),
                scale_inv_rowwise.stride(1),
                None,
                None,
                scale_inv_rowwise_n_rows,
                scale_inv_rowwise_n_cols,
                None,
                None,
                hadamard_matrix if scaling_recipe.use_rht else None,
                hadamard_matrix.size(0) if scaling_recipe.use_rht else 0,
                rowwise_philox_seed,
                rowwise_philox_offset,
                None,
                None,
                BLOCK_X=BLOCK_X,
                BLOCK_Y=BLOCK_Y,
                GROUP_Y=GROUP_Y,
                USE_ROWWISE=use_rowwise,
                USE_COLWISE=False,
                ROWWISE_USE_2D_BLOCK=scaling_recipe.use_2d_block,
                COLWISE_USE_2D_BLOCK=False,
                ROWWISE_USE_SR=scaling_recipe.use_sr,
                COLWISE_USE_SR=False,
                ROWWISE_USE_RHT=scaling_recipe.use_rht,
                COLWISE_USE_RHT=False,
                MXFP4_BLOCK_SIZE=block_size,
            )

    colwise_philox_seed, colwise_philox_offset = (
        scaling_recipe_for_trans.philox_seed,
        scaling_recipe_for_trans.philox_offset,
    )
    if use_colwise:
        padded_num_rows, padded_row_length = num_rows, row_length
        padded_num_rows += num_rows_padding_size

        assert (
            padded_num_rows % block_size == 0
        ), f"The first dimension of the x tensor must be divisible by the block size but got {padded_num_rows} % {block_size} != 0."

        if scaling_recipe_for_trans.use_2d_block:
            assert (
                padded_row_length % block_size == 0
            ), f"The last dimension of the x tensor must be divisible by the block size when use 2D block but got {padded_row_length} % {block_size} != 0."

        if scaling_recipe_for_trans.use_sr:
            if colwise_philox_seed is None:
                colwise_philox_seed = torch.randint(0, 2**31 - 1, (1,), device="cpu").item()
            if colwise_philox_offset is None:
                colwise_philox_offset = torch.randint(0, 2**31 - 1, (1,), device="cpu").item()

        scale_inv_colwise_n_rows, scale_inv_colwise_n_cols = padded_num_rows // block_size, padded_row_length

        y_colwise = torch.empty((padded_row_length, padded_num_rows // 2), dtype=torch.uint8, device=x.device)
        scale_inv_colwise = torch.empty(
            scale_inv_colwise_n_cols, scale_inv_colwise_n_rows, dtype=torch.uint8, device=x.device
        )

        if num_rows_padding_size > 0 or row_length_padding_size > 0:
            BLOCK_X = 64
            BLOCK_Y = 64
            GROUP_Y = block_size
            grid = lambda META: (
                triton.cdiv(padded_num_rows, META["BLOCK_Y"])
                * triton.cdiv(padded_row_length, META["BLOCK_X"]),
            )
            quantize_mxfp4_kernel[grid](
                x,
                None,
                y_colwise,
                x.stride(0),
                x.stride(1),
                None,
                None,
                y_colwise.stride(0),
                y_colwise.stride(1),
                num_rows,
                row_length,
                padded_num_rows,
                padded_row_length,
                scale_inv_rowwise,
                scale_inv_colwise,
                None,
                None,
                scale_inv_colwise.stride(0),
                scale_inv_colwise.stride(1),
                None,
                None,
                scale_inv_colwise_n_rows,
                scale_inv_colwise_n_cols,
                hadamard_matrix if scaling_recipe_for_trans.use_rht else None,
                hadamard_matrix.size(0) if scaling_recipe_for_trans.use_rht else 0,
                None,
                None,
                colwise_philox_seed,
                colwise_philox_offset,
                BLOCK_X=BLOCK_X,
                BLOCK_Y=BLOCK_Y,
                GROUP_Y=GROUP_Y,
                USE_ROWWISE=False,
                USE_COLWISE=use_colwise,
                ROWWISE_USE_2D_BLOCK=False,
                COLWISE_USE_2D_BLOCK=scaling_recipe_for_trans.use_2d_block,
                ROWWISE_USE_SR=False,
                COLWISE_USE_SR=scaling_recipe_for_trans.use_sr,
                ROWWISE_USE_RHT=False,
                COLWISE_USE_RHT=scaling_recipe_for_trans.use_rht,
                MXFP4_BLOCK_SIZE=block_size,
            )

    if num_rows_padding_size == 0 and row_length_padding_size == 0:
        BLOCK_X = 64
        BLOCK_Y = 64
        GROUP_Y = block_size
        grid = lambda META: (
            triton.cdiv(padded_num_rows, META["BLOCK_Y"]) * triton.cdiv(padded_row_length, META["BLOCK_X"]),
        )
        quantize_mxfp4_kernel[grid](
            x,
            y_rowwise,
            y_colwise,
            x.stride(0),
            x.stride(1),
            y_rowwise.stride(0) if y_rowwise is not None else None,
            y_rowwise.stride(1) if y_rowwise is not None else None,
            y_colwise.stride(0) if y_colwise is not None else None,
            y_colwise.stride(1) if y_colwise is not None else None,
            num_rows,
            row_length,
            padded_num_rows,
            padded_row_length,
            scale_inv_rowwise,
            scale_inv_colwise,
            scale_inv_rowwise.stride(0) if scale_inv_rowwise is not None else None,
            scale_inv_rowwise.stride(1) if scale_inv_rowwise is not None else None,
            scale_inv_colwise.stride(0) if scale_inv_colwise is not None else None,
            scale_inv_colwise.stride(1) if scale_inv_colwise is not None else None,
            scale_inv_rowwise_n_rows if scale_inv_rowwise is not None else None,
            scale_inv_rowwise_n_cols if scale_inv_rowwise is not None else None,
            scale_inv_colwise_n_rows if scale_inv_colwise is not None else None,
            scale_inv_colwise_n_cols if scale_inv_colwise is not None else None,
            hadamard_matrix if scaling_recipe.use_rht or scaling_recipe_for_trans.use_rht else None,
            hadamard_matrix.size(0) if scaling_recipe.use_rht or scaling_recipe_for_trans.use_rht else 0,
            rowwise_philox_seed,
            rowwise_philox_offset,
            colwise_philox_seed,
            colwise_philox_offset,
            BLOCK_X=BLOCK_X,
            BLOCK_Y=BLOCK_Y,
            GROUP_Y=GROUP_Y,
            USE_ROWWISE=use_rowwise,
            USE_COLWISE=use_colwise,
            ROWWISE_USE_2D_BLOCK=scaling_recipe.use_2d_block,
            COLWISE_USE_2D_BLOCK=scaling_recipe_for_trans.use_2d_block,
            ROWWISE_USE_SR=scaling_recipe.use_sr,
            COLWISE_USE_SR=scaling_recipe_for_trans.use_sr,
            ROWWISE_USE_RHT=scaling_recipe.use_rht,
            COLWISE_USE_RHT=scaling_recipe_for_trans.use_rht,
            MXFP4_BLOCK_SIZE=block_size,
        )

    return_list = []
    if use_rowwise:
        return_list += [
            y_rowwise.view(torch.float4_e2m1fn_x2),
            scale_inv_rowwise.view(dtype=torch.float8_e8m0fnu),
        ]
    if use_colwise:
        return_list += [
            y_colwise.view(torch.float4_e2m1fn_x2),
            scale_inv_colwise.view(dtype=torch.float8_e8m0fnu),
        ]

    return tuple(return_list)


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

    use_rowwise = axis == 1

    num_rows, row_length = x.size()
    # NOTE: x is packed in last dimension
    row_length = row_length * 2
    assert (
        row_length % block_size == 0
    ), "The last dimension of the x tensor must be divisible by the block size."

    # NOTE: triton can't canonicalize torch.float8_e8m0fnu, so we need to reinterpret it to torch.uint8.
    scale_inv = scale_inv.view(torch.uint8)

    scale_m, scale_n = scale_inv.size()
    if use_rowwise:
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
        USE_ROWWISE=use_rowwise,
        MXFP4_BLOCK_SIZE=block_size,
    )

    return y
