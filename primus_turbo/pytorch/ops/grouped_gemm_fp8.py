###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
from typing import Union

import torch

from primus_turbo.pytorch.core.backend import BackendType
from primus_turbo.pytorch.core.low_precision import (
    Float8QuantConfig,
    Format,
    ScalingGranularity,
    float8_e4m3,
    float8_e5m2,
)
from primus_turbo.pytorch.kernels.grouped_gemm.grouped_gemm_fp8_impl import (
    grouped_gemm_compute_offs,
    grouped_gemm_fp8_impl,
    grouped_gemm_fp8_variable_k_impl,
)
from primus_turbo.pytorch.core.low_precision import ScalingRecipe
from primus_turbo.pytorch.kernels.quantization.quantization_impl import (
    quant_fp8_blockwise_for_weight_impl,
    quant_fp8_blockwise_impl,
    quant_fp8_blockwise_segment_m_impl,
)
from primus_turbo.pytorch.ops.quantization import (
    MX_BLOCK_SIZE,
    quantize_fp8,
    quantize_fp8_with_trans,
)

MXFP8_PADDING_ALIGN_SIZE = 128

__all__ = [
    "grouped_gemm_fp8",
]


def _get_fp8_dtype(format: Format, is_fwd_stage: bool):
    if format == Format.E4M3:
        return float8_e4m3
    elif format == Format.E5M2:
        return float8_e5m2
    elif format == Format.HYBRID:
        return float8_e4m3 if is_fwd_stage else float8_e5m2
    else:
        raise ValueError(f"Unsupported FP8 format: {format}")


def _regroup_b_colwise_for_dgrad(
    b_fp8_col_2d: torch.Tensor,
    b_scale_inv_col_2d: torch.Tensor,
    group_num: int,
    n: int,
    k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert flattened B^T quantization to grouped (G, K, N) layout for dgrad."""
    assert b_fp8_col_2d.size(1) >= group_num * n
    assert n % MX_BLOCK_SIZE == 0

    scale_cols = n // MX_BLOCK_SIZE
    assert b_scale_inv_col_2d.size(0) == k
    assert b_scale_inv_col_2d.size(1) >= group_num * scale_cols

    # Transposed view; the transpose is folded into the .contiguous() / copy_().
    b_fp8_col_view = (
        b_fp8_col_2d[:, : group_num * n]
        .reshape(k, group_num, n)
        .transpose(0, 1)
    )
    b_scale_inv_col_view = (
        b_scale_inv_col_2d[:, : group_num * scale_cols]
        .reshape(k, group_num, scale_cols)
        .transpose(0, 1)
    )
    if n % MXFP8_PADDING_ALIGN_SIZE == 0:
        return b_fp8_col_view.contiguous(), b_scale_inv_col_view.contiguous()

    # Tail-pad N to MXFP8_PADDING_ALIGN_SIZE (e.g. N=2880 -> 2944).
    n_padded = (
        (n + MXFP8_PADDING_ALIGN_SIZE - 1) // MXFP8_PADDING_ALIGN_SIZE
    ) * MXFP8_PADDING_ALIGN_SIZE
    scale_cols_padded = n_padded // MX_BLOCK_SIZE
    b_fp8_col_padded = b_fp8_col_view.new_zeros((group_num, k, n_padded))
    b_fp8_col_padded[:, :, :n].copy_(b_fp8_col_view)
    b_scale_inv_col_padded = b_scale_inv_col_view.new_zeros((group_num, k, scale_cols_padded))
    b_scale_inv_col_padded[:, :, :scale_cols].copy_(b_scale_inv_col_view)
    return b_fp8_col_padded, b_scale_inv_col_padded


def _ensure_contiguous_grad_out(grad_out: torch.Tensor) -> torch.Tensor:
    # Upstream reductions sometimes produce expanded zero-stride views.
    return grad_out if grad_out.is_contiguous() else grad_out.contiguous()


class FP8GroupedGemmBlockFunc(torch.autograd.Function):

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

        a_dtype = _get_fp8_dtype(config.format, True)
        b_dtype = _get_fp8_dtype(config.format, True)

        a_fp8_row, a_scale_inv_row = quant_fp8_blockwise_impl(
            a, a_dtype, axis=1, block_size=config.block_size
        )

        b_fp8, b_scale_inv = quant_fp8_blockwise_for_weight_impl(b, b_dtype, block_size=config.block_size)

        out = grouped_gemm_fp8_impl(
            a_fp8_row,
            b_fp8,
            a_scale_inv_row,
            b_scale_inv,
            group_lens,
            group_offs,
            trans_a=False,
            trans_b=trans_b,
            out_dtype=out_dtype,
            granularity=config.granularity.value,
            num_cu=num_cu,
            default_backend=BackendType.TRITON.value,
        )

        a_fp8_col, a_scale_inv_col, _, _ = quant_fp8_blockwise_segment_m_impl(
            a, a_dtype, config.block_size, group_lens, group_offs
        )

        ctx.save_for_backward(
            a_fp8_col,
            a_scale_inv_col,
            b_fp8,
            b_scale_inv,
            group_lens,
            group_offs,
        )
        ctx.trans_a = False
        ctx.trans_b = trans_b
        ctx.config = config
        ctx.out_dtype = out_dtype
        ctx.num_cu = num_cu

        return out

    @staticmethod
    def backward(ctx, grad_out):
        grad_out = _ensure_contiguous_grad_out(grad_out)

        (
            a_fp8_col,
            a_scale_inv_col,
            b_fp8,
            b_scale_inv,
            group_lens,
            group_offs,
        ) = ctx.saved_tensors
        block_size = ctx.config.block_size
        grad_out_dtype = _get_fp8_dtype(ctx.config.format, False)

        # Quantize grad_out in row-wise for dgrad
        grad_out_fp8_row, grad_out_scale_inv_row = quant_fp8_blockwise_impl(
            grad_out, grad_out_dtype, axis=1, block_size=block_size
        )

        # grad_a: grad_out @ b^T
        grad_a = grouped_gemm_fp8_impl(
            grad_out_fp8_row,
            b_fp8,
            grad_out_scale_inv_row,
            b_scale_inv,
            group_lens,
            group_offs,
            trans_a=False,
            trans_b=not ctx.trans_b,
            out_dtype=ctx.out_dtype,
            granularity=ctx.config.granularity.value,
            num_cu=ctx.num_cu,
            default_backend=BackendType.TRITON.value,
        )

        # Quantize grad_out with segment padding for wgrad (colwise quantization)
        grad_out_fp8_col, grad_out_scale_inv_col, var_k_group_lens, var_k_group_offs = (
            quant_fp8_blockwise_segment_m_impl(
                grad_out,
                grad_out_dtype,
                block_size,
                group_lens,
                group_offs,
            )
        )

        grad_b = grouped_gemm_fp8_variable_k_impl(
            a_fp8_col,
            grad_out_fp8_col,
            a_scale_inv_col,
            grad_out_scale_inv_col,
            var_k_group_lens,
            var_k_group_offs,
            trans_a=not ctx.trans_a,
            trans_b=False,
            trans_c=ctx.trans_b,
            out_dtype=ctx.out_dtype,
            granularity=ctx.config.granularity.value,
            num_cu=ctx.num_cu,
            default_backend=BackendType.TRITON.value,
        )

        return grad_a, grad_b, None, None, None, None, None


class FP8GroupedGemmRowFunc(torch.autograd.Function):

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

        a_dtype = _get_fp8_dtype(config.format, True)
        b_dtype = _get_fp8_dtype(config.format, True)
        a_fp8_row, a_scale_inv_row = quantize_fp8(a, a_dtype, config.granularity, axis=-1)
        b_fp8_row, b_scale_inv_row = quantize_fp8(
            b, b_dtype, config.granularity, axis=(-1 if trans_b else -2)
        )
        out = grouped_gemm_fp8_impl(
            a_fp8_row,
            b_fp8_row,
            a_scale_inv_row,
            b_scale_inv_row,
            group_lens,
            group_offs,
            trans_a=False,
            trans_b=trans_b,
            out_dtype=out_dtype,
            granularity=config.granularity.value,
            num_cu=num_cu,
            default_backend=BackendType.TRITON.value,
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
        grad_out = _ensure_contiguous_grad_out(grad_out)
        a_fp8_col, b_fp8_col, a_scale_inv_col, b_scale_inv_col, group_lens, group_offs = ctx.saved_tensors

        # For grad_a
        grad_out_dtype = _get_fp8_dtype(ctx.config.format, False)
        grad_out_fp8_row, grad_out_scale_inv_row = quantize_fp8(
            grad_out, grad_out_dtype, ctx.config.granularity, axis=-1
        )

        grad_a = grouped_gemm_fp8_impl(
            grad_out_fp8_row,
            b_fp8_col,
            grad_out_scale_inv_row,
            b_scale_inv_col,
            group_lens,
            group_offs,
            trans_a=False,
            trans_b=not ctx.trans_b,
            out_dtype=ctx.out_dtype,
            granularity=ctx.config.granularity.value,
            num_cu=ctx.num_cu,
            default_backend=BackendType.TRITON.value,
        )

        # For grad_b
        grad_out_fp8_col, grad_out_scale_inv_col = quantize_fp8(
            grad_out, grad_out_dtype, ctx.config.granularity, axis=-2
        )

        grad_b = grouped_gemm_fp8_variable_k_impl(
            a_fp8_col,
            grad_out_fp8_col,
            a_scale_inv_col,
            grad_out_scale_inv_col,
            group_lens,
            group_offs,
            trans_a=not ctx.trans_a,
            trans_b=False,
            trans_c=ctx.trans_b,
            out_dtype=ctx.out_dtype,
            granularity=ctx.config.granularity.value,
            num_cu=ctx.num_cu,
            default_backend=BackendType.TRITON.value,
        )

        return grad_a, grad_b, None, None, None, None, None


class FP8GroupedGemmTensorFunc(torch.autograd.Function):

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
        a_dtype = _get_fp8_dtype(config.format, True)
        b_dtype = _get_fp8_dtype(config.format, True)
        a_fp8, a_scale_inv = quantize_fp8(a, a_dtype, config.granularity)
        b_fp8, b_scale_inv = quantize_fp8(b, b_dtype, config.granularity)

        out = grouped_gemm_fp8_impl(
            a_fp8,
            b_fp8,
            a_scale_inv,
            b_scale_inv,
            group_lens,
            group_offs,
            trans_a=False,
            trans_b=trans_b,
            out_dtype=a.dtype,
            granularity=config.granularity.value,
            num_cu=num_cu,
            default_backend=BackendType.TRITON.value,
            maybe_pre_sync=True,
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
        grad_out = _ensure_contiguous_grad_out(grad_out)
        a_fp8, b_fp8, a_scale_inv, b_scale_inv, group_lens, group_offs = ctx.saved_tensors

        # For grad_a
        grad_out_dtype = _get_fp8_dtype(ctx.config.format, False)
        grad_out_fp8, grad_out_scale_inv = quantize_fp8(grad_out, grad_out_dtype, ctx.config.granularity)
        grad_a = grouped_gemm_fp8_impl(
            grad_out_fp8,
            b_fp8,
            grad_out_scale_inv,
            b_scale_inv,
            group_lens,
            group_offs,
            trans_a=False,
            trans_b=not ctx.trans_b,
            out_dtype=ctx.out_dtype,
            granularity=ctx.config.granularity.value,
            num_cu=ctx.num_cu,
            default_backend=BackendType.TRITON.value,
        )

        # For grad_b
        grad_b = grouped_gemm_fp8_variable_k_impl(
            a_fp8,
            grad_out_fp8,
            a_scale_inv,
            grad_out_scale_inv,
            group_lens,
            group_offs,
            trans_a=not ctx.trans_a,
            trans_b=False,
            trans_c=ctx.trans_b,
            out_dtype=ctx.out_dtype,
            granularity=ctx.config.granularity.value,
            num_cu=ctx.num_cu,
            default_backend=BackendType.TRITON.value,
        )

        return grad_a, grad_b, None, None, None, None, None


class GroupedGemmFP8MXFunc(torch.autograd.Function):
    """MXFP8 grouped GEMM autograd (NT layout, MX_BLOCKWISE).

    Forward / dgrad / wgrad all land on the turbo MXFP8 backends.
    """

    @staticmethod
    def forward(
        ctx,
        a: torch.Tensor,
        b: torch.Tensor,
        group_lens: torch.Tensor,  # [G,] int64
        group_offs: torch.Tensor,  # [G+1,] int64
        trans_b: bool,
        config: Float8QuantConfig,
        num_cu: int | None,
    ):
        assert config.granularity == ScalingGranularity.MX_BLOCKWISE
        assert a.ndim == 2, "A must be 2D [total_M, K]."
        assert b.ndim == 3, "B must be 3D [G, N, K] (NT layout)."
        assert trans_b, "MX_BLOCKWISE grouped GEMM only supports NT layout (trans_b=True)."
        out_dtype = a.dtype
        assert out_dtype in [torch.float16, torch.bfloat16]

        G, N, K = b.shape
        # Caller must pass 128-aligned per-group M_g; misalignment is silent.
        a_dtype = b_dtype = _get_fp8_dtype(config.format, True)

        # A row-wise (fwd LHS) + col-wise (saved for wgrad).
        a_fp8_row, a_scale_inv_row, a_fp8_col, a_scale_inv_col = quantize_fp8_with_trans(
            a, a_dtype, config.granularity, block_size=MX_BLOCK_SIZE
        )

        # B row-wise (fwd) + col-wise (dgrad NT(dC, B^T)).
        weight_2d_block_recipe = ScalingRecipe(use_2d_block=True)
        b_fp8_row_2d, b_scale_inv_row_2d, b_fp8_col_2d, b_scale_inv_col_2d = quantize_fp8_with_trans(
            b.reshape(G * N, K),
            b_dtype,
            config.granularity,
            block_size=MX_BLOCK_SIZE,
            scaling_recipe=weight_2d_block_recipe,
            scaling_recipe_for_trans=weight_2d_block_recipe,
        )
        b_fp8_row = b_fp8_row_2d.reshape(G, N, -1)
        b_scale_inv_row = b_scale_inv_row_2d.reshape(G, N, -1)
        # Col-quant regroup is deferred to backward to keep fwd lean.

        # Forward: NT(A_row, B_row).
        out = grouped_gemm_fp8_impl(
            a_fp8_row,
            b_fp8_row,
            a_scale_inv_row,
            b_scale_inv_row,
            group_lens,
            group_offs,
            trans_a=False,
            trans_b=True,
            out_dtype=out_dtype,
            granularity=config.granularity.value,
            num_cu=num_cu,
            default_backend=BackendType.TURBO.value,
        )

        ctx.save_for_backward(
            a_fp8_col,
            a_scale_inv_col,
            b_fp8_col_2d,
            b_scale_inv_col_2d,
            group_lens,
            group_offs,
        )
        ctx.config = config
        ctx.out_dtype = out_dtype
        ctx.num_cu = num_cu
        # G/N/K recoverable from saved tensors; trans_b is fixed True (NT only).
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        grad_out = _ensure_contiguous_grad_out(grad_out)
        (
            a_fp8_col,
            a_scale_inv_col,
            b_fp8_col_2d,
            b_scale_inv_col_2d,
            group_lens,
            group_offs,
        ) = ctx.saved_tensors

        # grad_out: row-wise (dgrad) + col-wise transposed (wgrad).
        grad_out_fp8_row, grad_out_scale_inv_row, grad_out_t_fp8, grad_out_t_scale_inv = quantize_fp8_with_trans(
            grad_out,
            _get_fp8_dtype(ctx.config.format, False),
            ctx.config.granularity,
            block_size=MX_BLOCK_SIZE,
        )

        # Regroup B's flat col-quant into per-group (G, K, N) for dgrad.
        b_fp8_col, b_scale_inv_col = _regroup_b_colwise_for_dgrad(
            b_fp8_col_2d,
            b_scale_inv_col_2d,
            group_lens.size(0),
            grad_out.size(1),
            b_fp8_col_2d.size(0),
        )

        # dgrad: dA = NT(dC, B^T); b_fp8_col is (G, K, N).
        grad_a = grouped_gemm_fp8_impl(
            grad_out_fp8_row,
            b_fp8_col,
            grad_out_scale_inv_row,
            b_scale_inv_col,
            group_lens,
            group_offs,
            trans_a=False,
            trans_b=True,
            out_dtype=ctx.out_dtype,
            granularity=ctx.config.granularity.value,
            num_cu=ctx.num_cu,
            default_backend=BackendType.TURBO.value,
        )

        # wgrad: dB = dC^T @ A via fixed-NT variable-K turbo; col-quant
        # tensors are already transposed, so all trans_* are False.
        grad_b = grouped_gemm_fp8_variable_k_impl(
            grad_out_t_fp8,
            a_fp8_col,
            grad_out_t_scale_inv,
            a_scale_inv_col,
            group_lens,
            group_offs,
            trans_a=False,
            trans_b=False,
            trans_c=False,
            out_dtype=ctx.out_dtype,
            granularity=ctx.config.granularity.value,
            num_cu=ctx.num_cu,
            default_backend=BackendType.TURBO.value,
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
        return FP8GroupedGemmTensorFunc.apply(*args)
    elif config.granularity == ScalingGranularity.ROWWISE:
        return FP8GroupedGemmRowFunc.apply(*args)
    elif config.granularity == ScalingGranularity.BLOCKWISE:
        return FP8GroupedGemmBlockFunc.apply(*args)
    elif config.granularity == ScalingGranularity.MX_BLOCKWISE:
        return GroupedGemmFP8MXFunc.apply(*args)
    else:
        raise ValueError(f"Unsupported FP8 ScalingGranularity: {config.granularity}")
