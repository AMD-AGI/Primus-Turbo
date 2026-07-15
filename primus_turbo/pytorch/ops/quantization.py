###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from typing import Optional, Tuple

import torch

from primus_turbo.pytorch.core.low_precision import (
    MXFP4_BLOCK_SIZE,
    MXFP8_BLOCK_SIZE,
    ScalingGranularity,
    ScalingRecipe,
    float4_e2m1fn_x2,
)
from primus_turbo.pytorch.kernels.quantization.quantization_impl import (
    dequantize_fp8_rowwise_impl,
    dequantize_fp8_tensorwise_impl,
    dequantize_mxfp4_impl,
    dequantize_mxfp8_impl,
    grouped_dequantize_mxfp8_impl,
    grouped_quantize_mxfp4_impl,
    grouped_quantize_mxfp8_impl,
    quant_fp8_blockwise_for_weight_impl,
    quant_fp8_blockwise_impl,
    quantize_fp8_rowwise_impl,
    quantize_fp8_tensorwise_impl,
    quantize_mxfp4_impl,
    quantize_mxfp8_impl,
)

__all__ = [
    "quantize_fp8",
    "quantize_fp8_with_trans",
    "dequantize_fp8",
    "quantize_fp4",
    "quantize_fp4_with_trans",
    "dequantize_fp4",
    "grouped_quantize_fp8",
    "grouped_quantize_fp8_with_trans",
    "grouped_dequantize_fp8",
    "grouped_quantize_fp4",
    "grouped_dequantize_fp4",
]


def quantize_fp8(
    x: torch.Tensor,
    out_dtype: torch.dtype,
    granularity: ScalingGranularity,
    *,
    block_size: Optional[int] = None,
    axis: Optional[int] = None,
    scaling_recipe: Optional[ScalingRecipe] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    FP8 Quantize

    NOTE:
        For ROWWISE quantization:
            1. The axis must be specified.

        For MXFP8 quantization:
            1. The x must be 2D tensor.
            2. The axis means direction of quantization. The 0 means along column direction and 1 means along row direction.
            3. The block size must be 32.
    """
    if granularity == ScalingGranularity.TENSORWISE:
        return quantize_fp8_tensorwise_impl(x, out_dtype)

    elif granularity == ScalingGranularity.ROWWISE:
        return quantize_fp8_rowwise_impl(x, out_dtype, axis)
    elif granularity == ScalingGranularity.BLOCKWISE:
        assert block_size is not None, "block_size must be specified for BLOCKWISE quantization"
        if scaling_recipe is not None and scaling_recipe.use_2d_block:
            # 2D block (for weight): ignores axis; scales along both dims.
            return quant_fp8_blockwise_for_weight_impl(x, out_dtype, block_size=block_size)
        assert axis is not None, "axis must be specified for 1D BLOCKWISE quantization"

        return quant_fp8_blockwise_impl(x, out_dtype, axis=axis, block_size=block_size)
    elif granularity == ScalingGranularity.MX_BLOCKWISE:
        assert block_size == MXFP8_BLOCK_SIZE, (
            f"The block size must be {MXFP8_BLOCK_SIZE} for MXFP8 quantization"
        )

        return quantize_mxfp8_impl(
            x,
            out_dtype,
            axis,
            block_size,
            False,
            scaling_recipe,
        )
    else:
        raise NotImplementedError(f"Unknown granularity {granularity}")


def quantize_fp8_with_trans(
    x: torch.Tensor,
    out_dtype: torch.dtype,
    granularity: ScalingGranularity,
    *,
    block_size: Optional[int] = None,
    scaling_recipe: Optional[ScalingRecipe] = None,
    scaling_recipe_for_trans: Optional[ScalingRecipe] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    FP8 Quantize with trans

    NOTE:
        For MXFP8 quantization:
            1. The x may be a 2D ``[M, N]`` tensor or a 3D ``[G, M, N]`` batched
               (per-group) tensor. The MX grouped GEMM weight path calls this with
               3D ``(G, N, K)`` weights.
            2. Both row-wise and col-wise outputs are produced in one pass; no
               ``axis`` argument is taken (it is implied by the dual direction).
            3. The block size must be 32.
            4. The return value is x_rowwise, x_scale_inv_rowwise, x_colwise and x_scale_inv_colwise.
    """
    if granularity == ScalingGranularity.MX_BLOCKWISE:
        assert block_size == MXFP8_BLOCK_SIZE, (
            f"The block size must be {MXFP8_BLOCK_SIZE} for MXFP8 quantization"
        )
        return quantize_mxfp8_impl(
            x,
            out_dtype,
            None,
            block_size,
            True,
            scaling_recipe,
            scaling_recipe_for_trans,
        )
    else:
        raise NotImplementedError(f"Unknown granularity {granularity}")


def grouped_quantize_fp8(
    x: torch.Tensor,
    out_dtype: torch.dtype,
    granularity: ScalingGranularity,
    group_lens: torch.Tensor,
    group_offs: torch.Tensor,
    *,
    block_size: Optional[int] = None,
    axis: Optional[int] = None,
    scaling_recipe: Optional[ScalingRecipe] = None,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """
    FP8 Grouped Quantize with trans
    """
    if granularity == ScalingGranularity.MX_BLOCKWISE:
        assert block_size == MXFP8_BLOCK_SIZE, (
            f"The block size must be {MXFP8_BLOCK_SIZE} for MXFP8 quantization"
        )

        return grouped_quantize_mxfp8_impl(
            x,
            out_dtype,
            axis,
            block_size,
            group_lens,
            group_offs,
            False,
            scaling_recipe,
        )
    else:
        raise NotImplementedError(f"Unknown granularity {granularity}")


def grouped_quantize_fp8_with_trans(
    x: torch.Tensor,
    out_dtype: torch.dtype,
    granularity: ScalingGranularity,
    group_lens: torch.Tensor,
    group_offs: torch.Tensor,
    *,
    block_size: Optional[int] = None,
    scaling_recipe: Optional[ScalingRecipe] = None,
    scaling_recipe_for_trans: Optional[ScalingRecipe] = None,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """
    FP8 Grouped Quantize with trans
    """
    if granularity == ScalingGranularity.MX_BLOCKWISE:
        assert block_size == MXFP8_BLOCK_SIZE, (
            f"The block size must be {MXFP8_BLOCK_SIZE} for MXFP8 quantization"
        )

        return grouped_quantize_mxfp8_impl(
            x,
            out_dtype,
            None,
            block_size,
            group_lens,
            group_offs,
            True,
            scaling_recipe,
            scaling_recipe_for_trans,
        )
    else:
        raise NotImplementedError(f"Unknown granularity {granularity}")


def dequantize_fp8(
    x: torch.Tensor,
    out_dtype: torch.dtype,
    granularity: ScalingGranularity,
    *,
    block_size: Optional[int] = None,
    axis: Optional[int] = None,
    scale_inv: torch.Tensor,
    scaling_recipe: Optional[ScalingRecipe] = None,
):
    """
    FP8 DeQuantize

    NOTE:
        For ROWWISE quantization:
            1. The axis must be specified.

        For MXFP8 quantization:
            1. The x must be 2D tensor.
            2. The axis means direction of de-quantization. The 0 means along column direction and 1 means along row direction.
            3. The block size must be 32.
    """
    if granularity == ScalingGranularity.TENSORWISE:
        return dequantize_fp8_tensorwise_impl(x, out_dtype, scale_inv)
    elif granularity == ScalingGranularity.ROWWISE:
        if axis is None:
            raise ValueError("axis must be specified for rowwise FP8 de-quantization")

        return dequantize_fp8_rowwise_impl(x, out_dtype, axis, scale_inv)
    elif granularity == ScalingGranularity.MX_BLOCKWISE:
        assert block_size == MXFP8_BLOCK_SIZE, (
            f"The block size must be {MXFP8_BLOCK_SIZE} for MXFP8 quantization"
        )

        return dequantize_mxfp8_impl(x, out_dtype, axis, block_size, scale_inv)
    else:
        raise NotImplementedError(f"Unknown granularity {granularity}")


def grouped_dequantize_fp8(
    x: torch.Tensor,
    out_dtype: torch.dtype,
    granularity: ScalingGranularity,
    group_offs: torch.Tensor,
    group_offs_padded: torch.Tensor,
    *,
    block_size: Optional[int] = None,
    axis: Optional[int] = None,
    scale_inv: torch.Tensor,
    scaling_recipe: Optional[ScalingRecipe] = None,
    total_M: Optional[int] = None,
) -> torch.Tensor:
    """
    FP8 Grouped DeQuantize
    """
    if granularity == ScalingGranularity.MX_BLOCKWISE:
        assert block_size == MXFP8_BLOCK_SIZE, (
            f"The block size must be {MXFP8_BLOCK_SIZE} for MXFP8 quantization"
        )
        return grouped_dequantize_mxfp8_impl(
            x, out_dtype, axis, block_size, scale_inv, group_offs, group_offs_padded, total_M
        )
    else:
        raise NotImplementedError(f"Unknown granularity {granularity}")


def quantize_fp4(
    x: torch.Tensor,
    out_dtype: torch.dtype,
    granularity: ScalingGranularity,
    *,
    block_size: Optional[int] = None,
    axis: Optional[int] = None,
    scaling_recipe: Optional[ScalingRecipe] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    FP4 Quantize (single direction).

    NOTE:
        For MXFP4 quantization:
            1. The x may be a 2D ``[M, N]`` tensor or a 3D ``[G, M, N]`` batched
               (per-group) tensor (e.g. the MX grouped GEMM weight ``[G, N, K]``).
               3D inputs are quantized per-group by the single-direction kernel,
               which walks each group along ``blockIdx.z``.
            2. The axis means direction of quantization. For 2D, ``axis == 1`` is
               row-wise and ``axis == 0`` is col-wise (transpose). For 3D, the last
               dim (``axis == 2``) is row-wise and ``axis == 1`` is col-wise.
            3. The block size must be 32.
    """
    if granularity == ScalingGranularity.MX_BLOCKWISE:
        assert block_size == MXFP4_BLOCK_SIZE, (
            f"The block size must be {MXFP4_BLOCK_SIZE} for MXFP4 quantization"
        )
        assert axis is not None, "axis must be specified for MXFP4 quantization"

        return quantize_mxfp4_impl(
            x,
            out_dtype,
            axis,
            block_size,
            with_trans=False,
            scaling_recipe=scaling_recipe,
        )
    else:
        raise NotImplementedError(f"Unknown granularity {granularity}")


def grouped_quantize_fp4(
    x: torch.Tensor,
    out_dtype: torch.dtype,
    granularity: ScalingGranularity,
    group_lens: torch.Tensor,
    group_offs: torch.Tensor,
    *,
    block_size: Optional[int] = None,
    axis: Optional[int] = None,
    scaling_recipe: Optional[ScalingRecipe] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """FP4 Grouped Quantize (single direction).

    ``x`` is a 2D packed-M grouped activation ``[total_m, N]`` (groups along M via
    ``group_lens`` / ``group_offs``). The FP4 grouped path only exposes the fused
    dual quantizer, so we run it once and keep the requested direction:

      * row-wise (``axis`` == last dim): tight-M layout (row i == input row i);
        the returned padded group_lens/offs equal the (tight) originals.
      * col-wise (``axis`` == 0): 128-padded per-group M layout; the returned
        padded group_lens/offs are the col-wise padded offsets from the kernel.

    Returns ``(data, scale_inv, group_lens_padded, group_offs_padded)`` to mirror
    :func:`grouped_quantize_fp8`.
    """
    assert out_dtype == float4_e2m1fn_x2, "The out_dtype must be float4_e2m1fn_x2 for MXFP4 quantization"
    if granularity == ScalingGranularity.MX_BLOCKWISE:
        assert block_size == MXFP4_BLOCK_SIZE, (
            f"The block size must be {MXFP4_BLOCK_SIZE} for MXFP4 quantization"
        )
        assert x.ndim == 2, "grouped MXFP4 quantize expects a 2D packed-M tensor"
        assert axis in (0, 1), "axis must be 0 (colwise) or 1 (rowwise) for grouped MXFP4"

        return grouped_quantize_mxfp4_impl(
            x,
            out_dtype,
            axis,
            block_size,
            group_lens,
            group_offs,
            False,
            scaling_recipe,
        )
    else:
        raise NotImplementedError(f"Unknown granularity {granularity}")


def grouped_quantize_fp4_with_trans(
    x: torch.Tensor,
    out_dtype: torch.dtype,
    granularity: ScalingGranularity,
    group_lens: torch.Tensor,
    group_offs: torch.Tensor,
    *,
    block_size: Optional[int] = None,
    scaling_recipe: Optional[ScalingRecipe] = None,
    scaling_recipe_for_trans: Optional[ScalingRecipe] = None,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """FP4 Grouped Quantize with trans (fused rowwise + colwise).

    ``x`` is a 2D packed-M grouped activation ``[total_m, N]`` (groups along M via
    ``group_lens`` / ``group_offs``). One bf16 read emits both operands:

      * row-wise FP4 [total_m, N_pad/2] + scale [total_m, N_pad/32] in the tight
        (un-padded) M layout (row i == input row i) -- the fwd/dgrad operand;
      * col-wise FP4 [N, M_pad_col/2] + scale [N, M_pad_col/32] in the 128-padded
        per-group M layout -- the variable-K wgrad operand.

    Returns ``(rowwise_out, rowwise_scale, colwise_out, colwise_scale,
    group_lens_padded_row, group_offs_padded_row,
    group_lens_padded_col, group_offs_padded_col)`` to mirror
    :func:`grouped_quantize_fp8_with_trans`. Row-wise is tight-M, so its padded
    layout equals the original ``group_lens`` / ``group_offs``.
    """
    assert out_dtype == float4_e2m1fn_x2, "The out_dtype must be float4_e2m1fn_x2 for MXFP4 quantization"
    if granularity == ScalingGranularity.MX_BLOCKWISE:
        assert block_size == MXFP4_BLOCK_SIZE, (
            f"The block size must be {MXFP4_BLOCK_SIZE} for MXFP4 quantization"
        )
        assert x.ndim == 2, "grouped MXFP4 quantize expects a 2D packed-M tensor"

        return grouped_quantize_mxfp4_impl(
            x,
            out_dtype,
            None,
            block_size,
            group_lens,
            group_offs,
            True,
            scaling_recipe,
            scaling_recipe_for_trans,
        )
    else:
        raise NotImplementedError(f"Unknown granularity {granularity}")


def quantize_fp4_with_trans(
    x: torch.Tensor,
    out_dtype: torch.dtype,
    granularity: ScalingGranularity,
    *,
    block_size: Optional[int] = None,
    axis: Optional[int] = None,
    scaling_recipe: Optional[ScalingRecipe] = None,
    scaling_recipe_for_trans: Optional[ScalingRecipe] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    FP4 Quantize with trans

    NOTE:
        For MXFP4 quantization:
            1. The x may be a 2D ``[M, N]`` tensor or a 3D ``[G, M, N]`` batched
               (per-group) tensor. The MX grouped GEMM weight path calls this with
               3D ``(G, N, K)`` weights; each group is walked by ``blockIdx.z``.
            2. Both row-wise and col-wise outputs are produced in one pass; no
               ``axis`` argument is taken (it is implied by the dual direction).
            3. The block size must be 32.
            4. The return value is x_rowwise, x_scale_inv_rowwise, x_colwise and x_scale_inv_colwise.
    """
    if granularity == ScalingGranularity.MX_BLOCKWISE:
        assert block_size == MXFP4_BLOCK_SIZE, (
            f"The block size must be {MXFP4_BLOCK_SIZE} for MXFP4 quantization"
        )
        return quantize_mxfp4_impl(
            x,
            out_dtype,
            axis,
            block_size,
            with_trans=True,
            scaling_recipe=scaling_recipe,
            scaling_recipe_for_trans=scaling_recipe_for_trans,
        )
    else:
        raise NotImplementedError(f"Unknown granularity {granularity}")


def dequantize_fp4(
    x: torch.Tensor,
    out_dtype: torch.dtype,
    granularity: ScalingGranularity,
    *,
    block_size: Optional[int] = None,
    axis: Optional[int] = None,
    scale_inv: torch.Tensor,
    scaling_recipe: Optional[ScalingRecipe] = None,
) -> torch.Tensor:
    """
    FP4 DeQuantize

    NOTE:
        For MXFP4 quantization:
            1. The x may be a 2D ``[M, N]`` tensor, or a 3D ``[G, M, N]`` batched
               tensor. 3D inputs are flattened to 2D ([G*M, ...]) in the kernel and
               reshaped back, mirroring :func:`dequantize_fp8`.
            2. The axis means direction of de-quantization. For 2D, 0 means the column
               direction and 1 the row direction. For 3D, 1 means the column direction
               and 2 the row direction.
            3. The block size must be 32.
    """
    if granularity == ScalingGranularity.MX_BLOCKWISE:
        assert block_size == MXFP4_BLOCK_SIZE, (
            f"The block size must be {MXFP4_BLOCK_SIZE} for MXFP4 quantization"
        )

        return dequantize_mxfp4_impl(x, out_dtype, axis, block_size, scale_inv)
    else:
        raise NotImplementedError(f"Unknown granularity {granularity}")


def grouped_dequantize_fp4(
    x: torch.Tensor,
    out_dtype: torch.dtype,
    granularity: ScalingGranularity,
    *,
    block_size: Optional[int] = None,
    axis: Optional[int] = None,
    scale_inv: torch.Tensor,
    scaling_recipe: Optional[ScalingRecipe] = None,
) -> torch.Tensor:
    """FP4 Grouped DeQuantize (row-wise only)."""
    if granularity == ScalingGranularity.MX_BLOCKWISE:
        assert block_size == MXFP4_BLOCK_SIZE, (
            f"The block size must be {MXFP4_BLOCK_SIZE} for MXFP4 quantization"
        )

        return dequantize_mxfp4_impl(x, out_dtype, axis, block_size, scale_inv)
    else:
        raise NotImplementedError(f"Unknown granularity {granularity}")
