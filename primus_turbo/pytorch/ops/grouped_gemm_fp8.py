###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
from typing import Optional, Union

import torch

from primus_turbo.pytorch.core.backend import BackendType
from primus_turbo.pytorch.core.low_precision import (
    Float8QuantConfig,
    Format,
    ScalingGranularity,
    ScalingRecipe,
    float8_e4m3,
    float8_e5m2,
)
from primus_turbo.pytorch.core.quantized_tensor import (
    QuantizedTensor,
    QuantizedTensorPair,
    check_quantized_tensor,
)
from primus_turbo.pytorch.core.utils import is_gfx942
from primus_turbo.pytorch.kernels.grouped_gemm.grouped_gemm_fp8_impl import (
    grouped_gemm_fp8_impl,
    grouped_gemm_fp8_variable_k_impl,
)
from primus_turbo.pytorch.kernels.grouped_gemm.grouped_gemm_utils import (
    group_offs_from_lens,
)
from primus_turbo.pytorch.kernels.quantization.quantization_impl import (
    quant_fp8_blockwise_for_weight_impl,
    quant_fp8_blockwise_segment_m_row_col_impl,
)
from primus_turbo.pytorch.ops.quantization import (
    grouped_quantize_fp8_with_trans,
    quantize_fp8_with_trans,
)

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


def _ensure_contiguous_grad_out(grad_out: torch.Tensor) -> torch.Tensor:
    # Some upstream reductions can produce expanded zero-stride grad_out views.
    # Custom grouped GEMM kernels expect dense layouts.
    return grad_out if grad_out.is_contiguous() else grad_out.contiguous()


def _deter_use_nt_layout_gemm_in_bwd(trans_a: bool, trans_b: bool):
    if is_gfx942():
        return False

    # NOTE: the non-NT layout gemm is not optimized for mi350/mi450.
    # Force to use NT layout GEMM in backward for now.
    return trans_a == False and trans_b == True


class FP8GroupedGemmBlockFunc(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx,
        a: torch.Tensor,
        b: torch.Tensor,
        group_lens: torch.Tensor,  # [B,] int64
        group_offs: torch.Tensor,  # [B + 1,] int64
        trans_b: bool,
        out_dtype: torch.dtype,
        config: Float8QuantConfig,
        num_cu: int | None,
    ):
        assert config.granularity == ScalingGranularity.BLOCKWISE
        assert config.block_size in [128], "Only block_size 128 is supported currently."
        assert a.ndim == 2, "Input tensor must be 2-dimensional."
        assert b.ndim == 3, "Weight tensor must be 3-dimensional."
        assert group_lens.size(0) == b.size(0), "group_lens size must match b size(0)."
        assert out_dtype in [torch.float16, torch.bfloat16]

        a_dtype = _get_fp8_dtype(config.format, True)
        b_dtype = _get_fp8_dtype(config.format, True)

        # One bf16 read of `a` → row-wise (fwd) + segment-padded col-wise (bwd wgrad).
        # Row scales are pre-shuffled to the persistent GEMM's scale order. gemm_other_dim
        # = fwd-GEMM N lets the quant pick the HIP fast path on small GEMMs.
        gemm_n = b.size(-2) if trans_b else b.size(-1)
        a_fp8_row, a_fp8_col, a_scale_inv_row, a_scale_inv_col, _, _ = (
            quant_fp8_blockwise_segment_m_row_col_impl(
                a, a_dtype, config.block_size, group_lens, group_offs, gemm_other_dim=gemm_n
            )
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

        # One bf16 read of grad_out → row-wise (dgrad) + segment-padded col-wise (wgrad).
        # gemm_other_dim = bwd-GEMM K lets the quant pick the HIP fast path on small GEMMs.
        gemm_k = b_fp8.size(-1) if ctx.trans_b else b_fp8.size(-2)
        (
            grad_out_fp8_row,
            grad_out_fp8_col,
            grad_out_scale_inv_row,
            grad_out_scale_inv_col,
            var_k_group_lens,
            var_k_group_offs,
        ) = quant_fp8_blockwise_segment_m_row_col_impl(
            grad_out, grad_out_dtype, block_size, group_lens, group_offs, gemm_other_dim=gemm_k
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

        return (
            grad_a,  # a
            grad_b,  # b
            None,  # group_lens
            None,  # group_offs
            None,  # trans_b
            None,  # out_dtype
            None,  # config
            None,  # num_cu
        )


class FP8GroupedGemmRowFunc(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx,
        a: Union[torch.Tensor, QuantizedTensor],
        b: Union[torch.Tensor, QuantizedTensor],
        a_colwise: Optional[QuantizedTensor],
        b_colwise: Optional[QuantizedTensor],
        group_lens: torch.Tensor,  # [B,] int64
        group_offs: torch.Tensor,  # [B + 1,] int64
        trans_b: bool,
        out_dtype: torch.dtype,
        config: Float8QuantConfig,
        num_cu: int | None,
    ):
        assert config.granularity == ScalingGranularity.ROWWISE

        # --- A side: [total_m, k] grouped activation, row-wise scale on axis=-1 (K) ---
        if isinstance(a, QuantizedTensor):
            assert a._is_grouped_tensor, "A QuantizedTensor input must be a grouped tensor"
            check_quantized_tensor(a, config, axis=-1)
            quantized_a = a
            group_offs = a.group_offs
        else:
            a_dtype = _get_fp8_dtype(config.format, True)
            quantized_a = QuantizedTensor.quantize(
                a,
                a_dtype,
                config.granularity,
                axis=-1,
                block_size=config.block_size,
                group_lens=group_lens,
            )

        # --- B side: 3D weight, row axis is K-direction, col axis is N-direction ---
        # trans_b=True  -> layout [G, N, K]: K is axis=-1, N is axis=-2
        # trans_b=False -> layout [G, K, N]: K is axis=-2, N is axis=-1
        b_row_axis = -1 if trans_b else -2
        b_col_axis = -2 if trans_b else -1
        if isinstance(b, QuantizedTensor):
            assert not b._is_grouped_tensor, "B QuantizedTensor input must not be a grouped tensor"
            check_quantized_tensor(b, config, axis=b_row_axis)
            quantized_b = b
        else:
            b_dtype = _get_fp8_dtype(config.format, True)
            quantized_b = QuantizedTensor.quantize(
                b,
                b_dtype,
                config.granularity,
                axis=b_row_axis,
                block_size=config.block_size,
            )

        out = grouped_gemm_fp8_impl(
            quantized_a.qdata,
            quantized_b.qdata,
            quantized_a.scale_inv,
            quantized_b.scale_inv,
            group_lens,
            group_offs,
            trans_a=False,
            trans_b=trans_b,
            out_dtype=out_dtype,
            granularity=config.granularity.value,
            num_cu=num_cu,
            default_backend=BackendType.TRITON.value,
        )

        # Col-wise trans cache for backward. If the caller pre-quantized this
        # and passed it via ``a_colwise`` / ``b_colwise``, reuse it directly; otherwise
        # derive it (dequantize + re-quantize along the other axis), mirroring
        # FP8GemmRowFunction in gemm_fp8.py.
        if a_colwise is not None:
            quantized_a_colwise = a_colwise
        else:
            quantized_a_colwise = QuantizedTensor.quantize(
                quantized_a.dequantize(),
                quantized_a.real_dtype,
                config.granularity,
                axis=-2,
                block_size=config.block_size,
                group_lens=group_lens,
            )

        if b_colwise is not None:
            quantized_b_colwise = b_colwise
        else:
            quantized_b_colwise = QuantizedTensor.quantize(
                quantized_b.dequantize(),
                quantized_b.real_dtype,
                config.granularity,
                axis=b_col_axis,
                block_size=config.block_size,
            )

        ctx.save_for_backward(
            quantized_a_colwise.qdata,
            quantized_b_colwise.qdata,
            quantized_a_colwise.scale_inv,
            quantized_b_colwise.scale_inv,
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
        a_fp8_col, b_fp8_col, a_scale_inv_col, b_scale_inv_col, group_lens, group_offs = ctx.saved_tensors

        grad_out_dtype = _get_fp8_dtype(ctx.config.format, False)

        # grad_out row-wise (axis=-1) for grad_a
        quantized_grad_out = QuantizedTensor.quantize(
            grad_out,
            grad_out_dtype,
            ctx.config.granularity,
            axis=-1,
            block_size=ctx.config.block_size,
            group_lens=group_lens,
        )

        grad_a = grouped_gemm_fp8_impl(
            quantized_grad_out.qdata,
            b_fp8_col,
            quantized_grad_out.scale_inv,
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

        # grad_out col-wise (axis=-2) for grad_b
        quantized_grad_out_t = QuantizedTensor.quantize(
            grad_out,
            grad_out_dtype,
            ctx.config.granularity,
            axis=-2,
            block_size=ctx.config.block_size,
            group_lens=group_lens,
        )

        grad_b = grouped_gemm_fp8_variable_k_impl(
            a_fp8_col,
            quantized_grad_out_t.qdata,
            a_scale_inv_col,
            quantized_grad_out_t.scale_inv,
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

        return (
            grad_a,  # a
            grad_b,  # b
            None,  # a_colwise
            None,  # b_colwise
            None,  # group_lens
            None,  # group_offs
            None,  # trans_b
            None,  # out_dtype
            None,  # config
            None,  # num_cu
        )


class FP8GroupedGemmTensorFunc(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx,
        a: Union[torch.Tensor, QuantizedTensor],
        b: Union[torch.Tensor, QuantizedTensor],
        a_colwise: Optional[QuantizedTensor],  # not used
        b_colwise: Optional[QuantizedTensor],
        group_lens: torch.Tensor,  # [B,] int64
        group_offs: torch.Tensor,  # [B + 1,] int64
        trans_b: bool,
        out_dtype: torch.dtype,
        config: Float8QuantConfig,
        num_cu: int | None,
    ):
        use_nt_layout_gemm_in_bwd = _deter_use_nt_layout_gemm_in_bwd(False, trans_b)

        assert config.granularity == ScalingGranularity.TENSORWISE

        if isinstance(a, QuantizedTensor):
            assert a._is_grouped_tensor, "A QuantizedTensor input must be a grouped tensor"
            check_quantized_tensor(a, config)
            quantized_a = a
            group_offs = a.group_offs
        else:
            a_dtype = _get_fp8_dtype(config.format, True)
            quantized_a = QuantizedTensor.quantize(
                a,
                a_dtype,
                config.granularity,
                axis=-1,
                block_size=config.block_size,
                group_lens=group_lens,
            )

        if isinstance(b, QuantizedTensor):
            assert not b._is_grouped_tensor, "B QuantizedTensor input must not be a grouped tensor"
            check_quantized_tensor(b, config)
            quantized_b = b
        else:
            b_dtype = _get_fp8_dtype(config.format, True)
            quantized_b = QuantizedTensor.quantize(
                b,
                b_dtype,
                config.granularity,
                axis=-1,
                block_size=config.block_size,
            )

        if use_nt_layout_gemm_in_bwd:
            if b_colwise is not None and isinstance(b_colwise, QuantizedTensor):
                quantized_b_colwise = b_colwise
            else:
                quantized_b_colwise = quantized_b.transpose(-1, -2).contiguous()

        out = grouped_gemm_fp8_impl(
            quantized_a.qdata,
            quantized_b.qdata,
            quantized_a.scale_inv,
            quantized_b.scale_inv,
            group_lens,
            group_offs,
            trans_a=False,
            trans_b=trans_b,
            out_dtype=out_dtype,
            granularity=config.granularity.value,
            num_cu=num_cu,
            default_backend=BackendType.TRITON.value,
            maybe_pre_sync=True,
        )

        if use_nt_layout_gemm_in_bwd:
            ctx.save_for_backward(
                quantized_a.qdata,
                quantized_b_colwise.qdata,
                quantized_a.scale_inv,
                quantized_b_colwise.scale_inv,
                group_lens,
                group_offs,
            )
        else:
            ctx.save_for_backward(
                quantized_a.qdata,
                quantized_b.qdata,
                quantized_a.scale_inv,
                quantized_b.scale_inv,
                group_lens,
                group_offs,
            )
        ctx.trans_a = False
        ctx.trans_b = trans_b
        ctx.use_nt_layout_gemm_in_bwd = use_nt_layout_gemm_in_bwd
        ctx.config = config
        ctx.out_dtype = out_dtype
        ctx.num_cu = num_cu

        return out

    @staticmethod
    def backward(ctx, grad_out):
        grad_out = _ensure_contiguous_grad_out(grad_out)
        a_fp8, b_fp8, a_scale_inv, b_scale_inv, group_lens, group_offs = ctx.saved_tensors

        grad_out_dtype = _get_fp8_dtype(ctx.config.format, False)
        quantized_grad_out = QuantizedTensor.quantize(
            grad_out,
            grad_out_dtype,
            ctx.config.granularity,
            axis=-1,
            block_size=ctx.config.block_size,
            group_lens=group_lens,
        )

        if ctx.use_nt_layout_gemm_in_bwd:
            # b_fp8 is the per-group (K, N) transpose cache; grad_a runs as NT.
            grad_a = grouped_gemm_fp8_impl(
                quantized_grad_out.qdata,
                b_fp8,
                quantized_grad_out.scale_inv,
                b_scale_inv,
                group_lens,
                group_offs,
                trans_a=False,
                trans_b=True,
                out_dtype=ctx.out_dtype,
                granularity=ctx.config.granularity.value,
                num_cu=ctx.num_cu,
                default_backend=BackendType.TRITON.value,
            )
        else:
            grad_a = grouped_gemm_fp8_impl(
                quantized_grad_out.qdata,
                b_fp8,
                quantized_grad_out.scale_inv,
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

        grad_b = grouped_gemm_fp8_variable_k_impl(
            a_fp8,
            quantized_grad_out.qdata,
            a_scale_inv,
            quantized_grad_out.scale_inv,
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

        return (
            grad_a,  # a
            grad_b,  # b
            None,  # a_colwise
            None,  # b_colwise
            None,  # group_lens
            None,  # group_offs
            None,  # trans_b
            None,  # out_dtype
            None,  # config
            None,  # num_cu
        )


class FP8GroupedGemmMXFunc(torch.autograd.Function):
    """MXFP8 grouped GEMM autograd (MX_BLOCKWISE), Triton backend.

    Same interface as the hip path; only the backend differs
    (default_backend=TRITON).  A / grad_out use grouped dual-quant (padded
    per-group M, dense E8M0 scale); B uses per-group dual-quant.  fwd / dgrad
    read the padded layout (group_offs_padded_rowwise); the output is
    over-allocated to the padded rows, group_offs_out packs each group tight,
    and the caller slices [:total_m].  wgrad output (G, N, K) is
    padding-independent.  When hip MX kernels land, only default_backend
    needs to flip to TURBO.
    """

    @staticmethod
    def forward(ctx, a, b, group_lens, group_offs, trans_b, out_dtype, config, num_cu):
        assert config.granularity == ScalingGranularity.MX_BLOCKWISE
        assert a.ndim == 2 and b.ndim == 3
        assert out_dtype in [torch.float16, torch.bfloat16]
        assert trans_b, "MXFP8 grouped GEMM only supports trans_b=True (NT layout)."

        a_dtype = b_dtype = _get_fp8_dtype(config.format, True)

        # A: fused grouped dual-quant + per-group M zero-pad (rowwise 32 / colwise
        # 128), padded layouts computed on GPU (no D2H sync).
        (
            a_fp8_row,
            a_scale_row,
            a_fp8_col,
            a_scale_col,
            _,
            group_offs_padded_rowwise,
            _,
            _,
        ) = grouped_quantize_fp8_with_trans(
            a,
            a_dtype,
            config.granularity,
            group_lens,
            group_offs,
            block_size=config.block_size,
            scaling_recipe=ScalingRecipe(),
            scaling_recipe_for_trans=ScalingRecipe(),
        )
        # B: per-group dual-quant (rowwise (G,N,K) for fwd, colwise (G,K,N) for dgrad).
        b_fp8_row, b_scale_row, b_fp8_col, b_scale_col = quantize_fp8_with_trans(
            b,
            b_dtype,
            config.granularity,
            block_size=config.block_size,
            scaling_recipe=ScalingRecipe(use_2d_block=True),
            scaling_recipe_for_trans=ScalingRecipe(use_2d_block=True),
        )

        total_m = int(a.size(0))
        # fwd: read rowwise-padded layout (group_offs_padded_rowwise); the output
        # is over-allocated to the padded rows, group_offs_out packs each group
        # tight, then slice [:total_m] back to the user-visible shape.
        out = grouped_gemm_fp8_impl(
            a_fp8_row,
            b_fp8_row,
            a_scale_row,
            b_scale_row,
            group_lens,
            group_offs_padded_rowwise,
            trans_a=False,
            trans_b=True,
            out_dtype=out_dtype,
            granularity=ScalingGranularity.MX_BLOCKWISE.value,
            num_cu=num_cu,
            default_backend=BackendType.TRITON.value,
            group_offs_out=group_offs,
        )
        out = out[:total_m]

        ctx.save_for_backward(a_fp8_col, a_scale_col, b_fp8_col, b_scale_col, group_lens, group_offs)
        ctx.config = config
        ctx.out_dtype = out_dtype
        ctx.num_cu = num_cu
        ctx.total_m = total_m
        return out

    @staticmethod
    def backward(ctx, grad_out):
        grad_out = _ensure_contiguous_grad_out(grad_out)
        (a_fp8_col, a_scale_col, b_fp8_col, b_scale_col, group_lens, group_offs) = ctx.saved_tensors
        grad_out_dtype = _get_fp8_dtype(ctx.config.format, False)

        (
            grad_out_fp8_row,
            grad_out_scale_row,
            grad_out_t_fp8,
            grad_out_t_scale,
            _,
            group_offs_padded_rowwise,
            group_lens_padded_colwise,
            group_offs_padded_colwise,
        ) = grouped_quantize_fp8_with_trans(
            grad_out,
            grad_out_dtype,
            ctx.config.granularity,
            group_lens,
            group_offs,
            block_size=ctx.config.block_size,
            scaling_recipe=ScalingRecipe(),
            scaling_recipe_for_trans=ScalingRecipe(),
        )

        # dgrad: grad_a = grad_out @ b_col^T  (same single NT op as fwd)
        grad_a = grouped_gemm_fp8_impl(
            grad_out_fp8_row,
            b_fp8_col,
            grad_out_scale_row,
            b_scale_col,
            group_lens,
            group_offs_padded_rowwise,
            trans_a=False,
            trans_b=True,
            out_dtype=ctx.out_dtype,
            granularity=ScalingGranularity.MX_BLOCKWISE.value,
            num_cu=ctx.num_cu,
            default_backend=BackendType.TRITON.value,
            group_offs_out=group_offs,
        )
        grad_a = grad_a[: ctx.total_m]

        # wgrad: grad_b[g] = grad_out_col[g] @ a_col[g]^T  (variable-K over colwise-128 M_g)
        grad_b = grouped_gemm_fp8_variable_k_impl(
            grad_out_t_fp8,
            a_fp8_col,
            grad_out_t_scale,
            a_scale_col,
            group_lens_padded_colwise,
            group_offs_padded_colwise,
            trans_a=False,
            trans_b=False,
            trans_c=False,
            out_dtype=ctx.out_dtype,
            granularity=ScalingGranularity.MX_BLOCKWISE.value,
            num_cu=ctx.num_cu,
            default_backend=BackendType.TRITON.value,
        )
        # NT-only: wgrad already produces grad_b as (G, N, K) matching b.
        return grad_a, grad_b, None, None, None, None, None, None


@torch._dynamo.disable(
    recursive=True,
    reason=(
        "Grouped FP8 GEMM constructs (Grouped)QuantizedTensor wrapper subclasses "
        "inside its autograd.Function.forward and reads their inner tensors "
        "(data / scale_inv / group_lens / group_offs). Dynamo cannot recover Python "
        "sources for those graph-internal inner tensors, tripping gb0116 "
        "('SourcelessBuilder.create cannot wrap FakeTensor'). "
    ),
)
def grouped_gemm_fp8(
    a: Union[torch.Tensor, QuantizedTensor, QuantizedTensorPair],
    b: Union[torch.Tensor, QuantizedTensor, QuantizedTensorPair],
    group_lens: torch.Tensor,
    group_offs: Union[torch.Tensor, None] = None,
    trans_b: bool = True,
    out_dtype: Union[torch.dtype, None] = None,
    config: Union[Float8QuantConfig, None] = None,
    num_cu: int | None = None,
) -> torch.Tensor:
    """Grouped GEMM with FP8 quantization.

    This function automatically quantizes input tensors to FP8 based on the config,
    performs grouped matrix multiplication, and returns the result in the original dtype.

    Args:
        a: Input tensor A with shape [bs * m, k] (float16 or bfloat16).
            Can also be a pre-quantized :class:`QuantizedTensor` (grouped), or
            a :class:`QuantizedTensorPair` carrying both ``data`` (row-wise) and
            the backward-direction ``data_colwise`` (col-wise) for ROWWISE granularity.
        b: Input tensor B with shape [bs, k, n] or [bs, n, k] if trans_b (float16 or bfloat16).
            Same pre-quantized variants as ``a`` are accepted.
        group_lens: Group lengths tensor [bs] (int64)
        trans_b: Whether B is transposed (default: True)
        out_dtype: Output dtype (default: None, inferred from input dtypes)
        config: FP8 quantization config. If None, uses default (TENSORWISE, E4M3, DYNAMIC)
        num_cu: Cap on the number of compute units the grouped GEMM may use
            (limits the persistent-kernel grid). If None, uses all CUs on the device.

    Returns:
        Output tensor with shape [m, n] (same dtype as input)
    """
    if config is None:
        config = Float8QuantConfig()

    if group_offs is None:
        group_offs = group_offs_from_lens(group_lens)
    if isinstance(a, QuantizedTensorPair):
        a_data, a_data_colwise = a.data_rowwise, a.data_colwise
    else:
        a_data, a_data_colwise = a, None

    if isinstance(b, QuantizedTensorPair):
        b_data, b_data_colwise = b.data_rowwise, b.data_colwise
    else:
        b_data, b_data_colwise = b, None

    if out_dtype is None:
        out_dtype = torch.promote_types(a_data.dtype, b_data.dtype)

    if config.granularity == ScalingGranularity.TENSORWISE:
        # TENSORWISE has a single scalar scale (no col-wise trans cache needed);
        # the inner ``data_colwise`` is ignored if provided.
        return FP8GroupedGemmTensorFunc.apply(
            a_data,
            b_data,
            a_data_colwise,
            b_data_colwise,
            group_lens,
            group_offs,
            trans_b,
            out_dtype,
            config,
            num_cu,
        )
    elif config.granularity == ScalingGranularity.ROWWISE:
        return FP8GroupedGemmRowFunc.apply(
            a_data,
            b_data,
            a_data_colwise,
            b_data_colwise,
            group_lens,
            group_offs,
            trans_b,
            out_dtype,
            config,
            num_cu,
        )
    elif config.granularity == ScalingGranularity.BLOCKWISE:
        # BLOCKWISE only accepts raw tensors today; preserve existing assertion
        # behaviour in ``FP8GroupedGemmBlockFunc.forward``.
        return FP8GroupedGemmBlockFunc.apply(a, b, group_lens, group_offs, trans_b, out_dtype, config, num_cu)
    elif config.granularity == ScalingGranularity.MX_BLOCKWISE:
        # MXFP8: raw tensors; quant + grouped GEMM via the Triton MX backend.
        return FP8GroupedGemmMXFunc.apply(a, b, group_lens, group_offs, trans_b, out_dtype, config, num_cu)
    else:
        raise ValueError(f"Unsupported FP8 ScalingGranularity: {config.granularity}")
