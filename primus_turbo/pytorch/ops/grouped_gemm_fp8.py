###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
from typing import Union

import torch

from primus_turbo.pytorch.core.low_precision import (
    Float8QuantConfig,
    Format,
    ScalingGranularity,
    float8_e4m3,
    float8_e5m2,
)
from primus_turbo.pytorch.kernels.grouped_gemm.grouped_gemm_fp8_impl import (
    grouped_gemm_compute_offs,
    grouped_gemm_fp8_csrc_impl,
    grouped_gemm_fp8_variable_k_csrc_impl,
)
from primus_turbo.pytorch.kernels.quantization.quantization_impl import (
    quant_fp8_blockwise_for_weight_impl,
    quant_fp8_blockwise_impl,
)
from primus_turbo.pytorch.ops.quantization import quantize_fp8

__all__ = [
    "grouped_gemm_fp8",
]


def _needs_padding_for_blockwise(group_offs: torch.Tensor, block_size: int = 128) -> bool:
    """Check if any group boundary is not aligned to block_size."""
    # group_offs[1:] are the boundaries (excluding the initial 0)
    # Check if all boundaries are divisible by block_size
    boundaries = group_offs[1:-1]  # Exclude first (0) and last (total)
    if boundaries.numel() == 0:
        return False
    return not torch.all(boundaries % block_size == 0).item()


def _pad_for_blockwise_variable_k(
    tensor: torch.Tensor,
    group_lens: torch.Tensor,
    group_offs: torch.Tensor,
    block_size: int = 128,
):
    """
    Pad tensor along the K dimension to align group boundaries to block_size.

    Args:
        tensor: Input tensor of shape [total_K, width]
        group_lens: Group lengths tensor of shape [B]
        group_offs: Group offsets tensor of shape [B+1]
        block_size: Block size for alignment (default 128)

    Returns:
        padded_tensor: Padded tensor with aligned group boundaries
        padded_group_lens: New group lengths (each padded to multiple of block_size)
        padded_group_offs: New group offsets (all aligned to block_size)
    """
    device = tensor.device
    dtype = tensor.dtype
    total_K, width = tensor.shape

    # Calculate padded group_lens (each group K padded to multiple of block_size)
    padded_group_lens = ((group_lens + block_size - 1) // block_size) * block_size

    # Calculate new group_offs
    padded_group_offs = torch.zeros_like(group_offs)
    padded_group_offs[1:] = torch.cumsum(padded_group_lens, dim=0)

    total_padded_K = padded_group_offs[-1].item()

    # Create padded tensor (zeros for padding regions)
    padded_tensor = torch.zeros(total_padded_K, width, dtype=dtype, device=device)

    # Compute cumulative padding before each group
    cumulative_padding = padded_group_offs[:-1] - group_offs[:-1]  # [B]

    # Create row indices and find group assignment using searchsorted
    row_indices = torch.arange(total_K, device=device, dtype=torch.int64)
    group_ids = torch.searchsorted(group_offs[1:], row_indices, right=True)

    # Calculate destination indices: src_idx + cumulative_padding[group_id]
    dst_indices = row_indices + cumulative_padding[group_ids]

    # Scatter source rows to destination (fully vectorized)
    padded_tensor[dst_indices] = tensor

    return padded_tensor, padded_group_lens, padded_group_offs


class GroupedGemmFP8BlockFunc(torch.autograd.Function):
    @staticmethod
    def get_fp8_dtype(format: Format, is_fwd_stage: bool):
        if format == Format.E4M3:
            return float8_e4m3
        elif format == Format.E5M2:
            return float8_e5m2
        else:
            raise ValueError(f"Unsupported FP8 format: {format}")

    @staticmethod
    def forward(
        ctx,
        a: torch.Tensor,
        b: torch.Tensor,
        group_lens: torch.Tensor,  # [B,] int64
        group_offs: torch.Tensor,  # [B+1,] int64
        trans_b: bool,
        config: Float8QuantConfig,
        num_cu: int | None,
    ):
        assert config.granularity == ScalingGranularity.BLOCKWISE
        assert config.block_size in [128], "Only block_size 128 is supported currently."
        assert a.ndim == 2, "Input tensor must be 2-dimensional."
        assert b.ndim == 3, "Weight tensor must be 3-dimensional."
        assert group_lens.size(0) == b.size(0), "group_lens size must match b size(0)."
        out_dtype = a.dtype
        assert out_dtype in [torch.float16, torch.bfloat16]

        a_dtype = GroupedGemmFP8BlockFunc.get_fp8_dtype(config.format, True)
        b_dtype = GroupedGemmFP8BlockFunc.get_fp8_dtype(config.format, True)
        a_fp8_row, a_scale_inv_row = quant_fp8_blockwise_impl(
            a, a_dtype, axis=1, block_size=config.block_size
        )
        b_fp8, b_scale_inv = quant_fp8_blockwise_for_weight_impl(b, b_dtype, block_size=config.block_size)

        out = grouped_gemm_fp8_csrc_impl(
            a_fp8_row,
            b_fp8,
            a_scale_inv_row,
            b_scale_inv,
            group_lens,
            group_offs,
            trans_a=False,
            trans_b=trans_b,
            out_dtype=out_dtype,
            granularity=config.granularity,
            num_cu=num_cu,
        )

        ctx.save_for_backward(a, b_fp8, b_scale_inv, group_lens, group_offs)
        ctx.trans_a = False
        ctx.trans_b = trans_b
        ctx.config = config
        ctx.out_dtype = out_dtype
        ctx.num_cu = num_cu

        return out

    @staticmethod
    def backward(ctx, grad_out):

        a, b_fp8, b_scale_inv, group_lens, group_offs = ctx.saved_tensors
        block_size = ctx.config.block_size
        grad_out_dtype = GroupedGemmFP8BlockFunc.get_fp8_dtype(ctx.config.format, False)
        a_dtype = GroupedGemmFP8BlockFunc.get_fp8_dtype(ctx.config.format, False)

        # Quantize grad_out in row-wise for dgrad
        grad_out_fp8_row, grad_out_scale_inv_row = quant_fp8_blockwise_impl(
            grad_out, grad_out_dtype, axis=1, block_size=block_size
        )

        # grad_a: grad_out @ b^T
        grad_a = grouped_gemm_fp8_csrc_impl(
            grad_out_fp8_row,
            b_fp8,
            grad_out_scale_inv_row,
            b_scale_inv,
            group_lens,
            group_offs,
            trans_a=False,
            trans_b=not ctx.trans_b,
            out_dtype=ctx.out_dtype,
            granularity=ctx.config.granularity,
            num_cu=ctx.num_cu,
        )

        # For grad_b, we need col-wise quantization with padding for alignment
        needs_padding = _needs_padding_for_blockwise(group_offs, block_size)

        if needs_padding:
            # Pad tensors to align group boundaries to block_size
            a_padded, padded_group_lens, padded_group_offs = _pad_for_blockwise_variable_k(
                a, group_lens, group_offs, block_size
            )
            grad_out_padded, _, _ = _pad_for_blockwise_variable_k(
                grad_out, group_lens, group_offs, block_size
            )
            # Quantize padded tensors
            a_fp8_col, a_scale_inv_col = quant_fp8_blockwise_impl(
                a_padded, a_dtype, axis=0, block_size=block_size
            )
            grad_out_fp8_col, grad_out_scale_inv_col = quant_fp8_blockwise_impl(
                grad_out_padded, grad_out_dtype, axis=0, block_size=block_size
            )
            var_k_group_lens = padded_group_lens
            var_k_group_offs = padded_group_offs
        else:
            # No padding needed, use original tensors
            a_fp8_col, a_scale_inv_col = quant_fp8_blockwise_impl(a, a_dtype, axis=0, block_size=block_size)
            grad_out_fp8_col, grad_out_scale_inv_col = quant_fp8_blockwise_impl(
                grad_out, grad_out_dtype, axis=0, block_size=block_size
            )
            var_k_group_lens = group_lens
            var_k_group_offs = group_offs

        lhs, rhs = (grad_out_fp8_col, a_fp8_col) if ctx.trans_b else (a_fp8_col, grad_out_fp8_col)
        lhs_scale, rhs_scale = (
            (grad_out_scale_inv_col, a_scale_inv_col)
            if ctx.trans_b
            else (a_scale_inv_col, grad_out_scale_inv_col)
        )

        grad_b = grouped_gemm_fp8_variable_k_csrc_impl(
            lhs,
            rhs,
            lhs_scale,
            rhs_scale,
            var_k_group_lens,
            var_k_group_offs,
            trans_a=True,
            trans_b=False,
            out_dtype=ctx.out_dtype,
            granularity=ctx.config.granularity,
            num_cu=ctx.num_cu,
        )

        return grad_a, grad_b, None, None, None, None, None


class GroupedGemmFP8RowFunc(torch.autograd.Function):
    @staticmethod
    def get_fp8_dtype(format: Format, is_fwd_stage: bool):
        if format == Format.E4M3:
            return float8_e4m3
        elif format == Format.E5M2:
            return float8_e5m2
        else:
            raise ValueError(f"Unsupported FP8 format: {format}")

    @staticmethod
    def forward(
        ctx,
        a: torch.Tensor,
        b: torch.Tensor,
        group_lens: torch.Tensor,  # [B,] int64
        group_offs: torch.Tensor,  # [B+1,] int64
        trans_b: bool,
        config: Float8QuantConfig,
        num_cu: int | None,
    ):
        assert config.granularity == ScalingGranularity.ROWWISE
        assert a.ndim == 2, "Input tensor must be 3-dimensions."
        assert b.ndim == 3, "Weight tensor must be 3-dimensional."
        out_dtype = a.dtype
        assert out_dtype in [torch.float16, torch.bfloat16]

        a_dtype = GroupedGemmFP8RowFunc.get_fp8_dtype(config.format, True)
        b_dtype = GroupedGemmFP8RowFunc.get_fp8_dtype(config.format, True)
        a_fp8_row, a_scale_inv_row = quantize_fp8(a, a_dtype, config.granularity, axis=-1)
        b_fp8_row, b_scale_inv_row = quantize_fp8(
            b, b_dtype, config.granularity, axis=(-1 if trans_b else -2)
        )
        out = grouped_gemm_fp8_csrc_impl(
            a_fp8_row,
            b_fp8_row,
            a_scale_inv_row,
            b_scale_inv_row,
            group_lens,
            group_offs,
            trans_a=False,
            trans_b=trans_b,
            out_dtype=out_dtype,
            granularity=config.granularity,
            num_cu=num_cu,
        )

        # we need a/b do col quant for backward.
        a_fp8_col, a_scale_inv_col = quantize_fp8(a, a_dtype, config.granularity, axis=-2)
        b_fp8_col, b_scale_inv_col = quantize_fp8(
            b, b_dtype, config.granularity, axis=(-2 if trans_b else -1)
        )

        ctx.save_for_backward(a_fp8_col, b_fp8_col, a_scale_inv_col, b_scale_inv_col, group_lens, group_offs)
        ctx.trans_a = False
        ctx.trans_b = trans_b
        ctx.config = config
        ctx.out_dtype = out_dtype
        ctx.num_cu = num_cu
        return out

    @staticmethod
    def backward(ctx, grad_out):
        a_fp8_col, b_fp8_col, a_scale_inv_col, b_scale_inv_col, group_lens, group_offs = ctx.saved_tensors

        # For grad_a
        grad_out_dtype = GroupedGemmFP8RowFunc.get_fp8_dtype(ctx.config.format, False)
        grad_out_fp8_row, grad_out_scale_inv_row = quantize_fp8(
            grad_out, grad_out_dtype, ctx.config.granularity, axis=-1
        )

        grad_a = grouped_gemm_fp8_csrc_impl(
            grad_out_fp8_row,
            b_fp8_col,
            grad_out_scale_inv_row,
            b_scale_inv_col,
            group_lens,
            group_offs,
            trans_a=False,
            trans_b=not ctx.trans_b,
            out_dtype=ctx.out_dtype,
            granularity=ctx.config.granularity,
            num_cu=ctx.num_cu,
        )

        # For grad_b
        grad_out_fp8_col, grad_out_scale_inv_col = quantize_fp8(
            grad_out, grad_out_dtype, ctx.config.granularity, axis=-2
        )

        lhs, rhs = (grad_out_fp8_col, a_fp8_col) if ctx.trans_b else (a_fp8_col, grad_out_fp8_col)
        lhs_scale, rhs_scale = (
            (grad_out_scale_inv_col, a_scale_inv_col)
            if ctx.trans_b
            else (a_scale_inv_col, grad_out_scale_inv_col)
        )

        grad_b = grouped_gemm_fp8_variable_k_csrc_impl(
            lhs,
            rhs,
            lhs_scale,
            rhs_scale,
            group_lens,
            group_offs,
            trans_a=True,
            trans_b=False,
            out_dtype=ctx.out_dtype,
            granularity=ctx.config.granularity,
            num_cu=ctx.num_cu,
        )

        return grad_a, grad_b, None, None, None, None, None


class GroupedGemmFP8TensorFunc(torch.autograd.Function):

    @staticmethod
    def get_fp8_dtype(format: Format, is_fwd_stage: bool):
        if format == Format.E4M3:
            return float8_e4m3
        elif format == Format.E5M2:
            return float8_e5m2
        else:
            raise ValueError(f"Unsupported FP8 format: {format}")

    @staticmethod
    def forward(
        ctx,
        a: torch.Tensor,
        b: torch.Tensor,
        group_lens: torch.Tensor,  # [B,] int64
        group_offs: torch.Tensor,  # [B+1,] int64
        trans_b: bool,
        config: Float8QuantConfig,
        num_cu: int | None,
    ):

        assert config.granularity == ScalingGranularity.TENSORWISE
        assert a.ndim == 2, "Input tensor must be 3-dimensions."
        assert b.ndim == 3, "Weight tensor must be 3-dimensional."
        a_dtype = GroupedGemmFP8TensorFunc.get_fp8_dtype(config.format, True)
        b_dtype = GroupedGemmFP8TensorFunc.get_fp8_dtype(config.format, True)
        a_fp8, a_scale_inv = quantize_fp8(a, a_dtype, config.granularity)
        b_fp8, b_scale_inv = quantize_fp8(b, b_dtype, config.granularity)

        out = grouped_gemm_fp8_csrc_impl(
            a_fp8,
            b_fp8,
            a_scale_inv,
            b_scale_inv,
            group_lens,
            group_offs,
            trans_a=False,
            trans_b=trans_b,
            out_dtype=a.dtype,
            granularity=config.granularity,
            num_cu=num_cu,
        )

        ctx.save_for_backward(a_fp8, b_fp8, a_scale_inv, b_scale_inv, group_lens, group_offs)
        ctx.trans_a = False
        ctx.trans_b = trans_b
        ctx.config = config
        ctx.out_dtype = a.dtype
        ctx.num_cu = num_cu
        return out

    @staticmethod
    def backward(ctx, grad_out):
        a_fp8, b_fp8, a_scale_inv, b_scale_inv, group_lens, group_offs = ctx.saved_tensors

        # For grad_a
        grad_out_dtype = GroupedGemmFP8TensorFunc.get_fp8_dtype(ctx.config.format, False)
        grad_out_fp8, grad_out_scale_inv = quantize_fp8(grad_out, grad_out_dtype, ctx.config.granularity)
        grad_a = grouped_gemm_fp8_csrc_impl(
            grad_out_fp8,
            b_fp8,
            grad_out_scale_inv,
            b_scale_inv,
            group_lens,
            group_offs,
            trans_a=False,
            trans_b=not ctx.trans_b,
            out_dtype=ctx.out_dtype,
            granularity=ctx.config.granularity,
            num_cu=ctx.num_cu,
        )

        # For grad_b
        lhs, rhs = (grad_out_fp8, a_fp8) if ctx.trans_b else (a_fp8, grad_out_fp8)
        lhs_scale, rhs_scale = (
            (grad_out_scale_inv, a_scale_inv) if ctx.trans_b else (a_scale_inv, grad_out_scale_inv)
        )
        grad_b = grouped_gemm_fp8_variable_k_csrc_impl(
            lhs,
            rhs,
            lhs_scale,
            rhs_scale,
            group_lens,
            group_offs,
            trans_a=True,
            trans_b=False,
            out_dtype=ctx.out_dtype,
            granularity=ctx.config.granularity,
            num_cu=ctx.num_cu,
        )

        return grad_a, grad_b, None, None, None, None, None


def grouped_gemm_fp8(
    a: torch.Tensor,
    b: torch.Tensor,
    group_lens: torch.Tensor,
    group_offs: torch.Tensor | None = None,
    trans_b: bool = True,
    config: Union[Float8QuantConfig, None] = None,
    num_cu: int | None = None,
) -> torch.Tensor:
    """ """
    supported_dtypes = [torch.bfloat16, torch.float16]
    assert a.dtype in supported_dtypes, f"Unsupported dtype {a.dtype}, expected one of {supported_dtypes}"
    assert b.dtype in supported_dtypes, f"Unsupported dtype {b.dtype}, expected one of {supported_dtypes}"

    if group_offs is None:
        group_offs = grouped_gemm_compute_offs(group_lens)
    if config is None:
        config = Float8QuantConfig()

    args = (a, b, group_lens, group_offs, trans_b, config, num_cu)

    if config.granularity == ScalingGranularity.TENSORWISE:
        return GroupedGemmFP8TensorFunc.apply(*args)
    elif config.granularity == ScalingGranularity.ROWWISE:
        return GroupedGemmFP8RowFunc.apply(*args)
    elif config.granularity == ScalingGranularity.BLOCKWISE:
        return GroupedGemmFP8BlockFunc.apply(*args)
    else:
        raise ValueError(f"Unsupported FP8 ScalingGranularity: {config.granularity}")


"""
TODO: MXFP8, MXFP4
"""
