###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""MXFP4 grouped GEMM (MX_BLOCKWISE, Triton backend, gfx950).

The MXFP4 training recipe (https://arxiv.org/pdf/2509.25149) applies a 16-point Random Hadamard
Transform (RHT) only to the wgrad operands (the col-wise grad_out + the cached
col-wise A transpose); fwd and dgrad stay plain MX. Because the Hadamard is
orthogonal it cancels inside each GEMM **only when both contracted operands
share it**, so the recipes are paired exactly like the dense ``gemm_fp4``:

    fwd   : C    = A_row(rht=F)         @ B_row(rht=F)^T          (contract K)
    dgrad : dA   = gradO_row(rht=F)     @ B_col(rht=F)^T          (contract N)
    wgrad : dB   = gradO_col(rht=T)     @ A_col(rht=T)^T          (contract M_g)
"""

from typing import Optional, Union

import torch

from primus_turbo.pytorch.core.backend import BackendType
from primus_turbo.pytorch.core.low_precision import (
    MXFP4_BLOCK_SIZE,
    Float4QuantConfig,
    ScalingGranularity,
    ScalingRecipe,
    check_mxfp4_support,
    float4_e2m1fn_x2,
)
from primus_turbo.pytorch.core.quantized_tensor import (
    QuantizedTensor,
    QuantizedTensorPair,
    check_quantized_tensor,
)
from primus_turbo.pytorch.kernels.grouped_gemm.grouped_gemm_fp4_impl import (
    grouped_gemm_fp4_impl,
    grouped_gemm_fp4_variable_k_impl,
)
from primus_turbo.pytorch.kernels.grouped_gemm.grouped_gemm_utils import (
    group_offs_from_lens,
)
from primus_turbo.pytorch.ops.quantization import grouped_quantize_fp4_with_trans, quantize_fp4_with_trans

__all__ = ["grouped_gemm_fp4"]


def _ensure_contiguous_grad_out(grad_out: torch.Tensor) -> torch.Tensor:
    # Some upstream reductions can produce expanded zero-stride grad_out views.
    # Custom grouped GEMM kernels expect dense layouts.
    return grad_out if grad_out.is_contiguous() else grad_out.contiguous()


class FP4GroupedGemmMXFunc(torch.autograd.Function):
    """MXFP4 grouped GEMM autograd (MX_BLOCKWISE, NT-only, Triton backend)."""

    @staticmethod
    def forward(
        ctx,
        a: Union[torch.Tensor, QuantizedTensor],
        b: Union[torch.Tensor, QuantizedTensor],
        a_t: Optional[QuantizedTensor],
        b_t: Optional[QuantizedTensor],
        group_lens: torch.Tensor,
        group_offs: torch.Tensor,
        trans_b: bool,
        out_dtype: torch.dtype,
        config: Float4QuantConfig,
        num_cu: int | None,
    ):
        assert config.granularity == ScalingGranularity.MX_BLOCKWISE
        assert a.ndim == 2 and b.ndim == 3
        assert out_dtype in (torch.float16, torch.bfloat16)
        assert trans_b, "MXFP4 grouped GEMM only supports trans_b=True (NT layout)."
        assert not config.use_preshuffle, "Triton MXFP4 grouped GEMM does not use preshuffle."
        supported, reason = check_mxfp4_support()
        assert supported, reason

        N, K = int(b.shape[-2]), int(b.shape[-1])
        # MX scales cover one 32-element block, so the contraction must be a
        # 32-multiple. N/K need not be 128-multiples: the quantizer zero-pads the
        # contraction to MXFP4_PADDING_ALIGN_SIZE (=128) and the GEMM runs over
        # that padded length
        assert K % MXFP4_BLOCK_SIZE == 0, f"K must be a multiple of {MXFP4_BLOCK_SIZE} (got {K})."
        assert N % MXFP4_BLOCK_SIZE == 0, f"N must be a multiple of {MXFP4_BLOCK_SIZE} (got {N})."
        total_m = int(a.size(0))

        a_scaling_recipe = ScalingRecipe()
        a_t_scaling_recipe = ScalingRecipe(
            use_rht=True,
        )
        if not isinstance(a, QuantizedTensor):
            a_row, a_row_scale, a_col, a_col_scale, _, group_offs_padded_rowwise, _, _ = (
                grouped_quantize_fp4_with_trans(
                    a,
                    float4_e2m1fn_x2,
                    config.granularity,
                    group_lens,
                    group_offs,
                    block_size=MXFP4_BLOCK_SIZE,
                    scaling_recipe=a_scaling_recipe,
                    scaling_recipe_for_trans=a_t_scaling_recipe,
                )
            )
        else:
            quantized_a = a
            check_quantized_tensor(quantized_a, config, axis=-1, scaling_recipe=a_scaling_recipe)
            if a_t is None:
                quantized_a_t = QuantizedTensor.quantize(
                    quantized_a.dequantize(),
                    quantized_a.real_dtype,
                    config.granularity,
                    axis=-2,
                    block_size=config.block_size,
                    scaling_recipe=a_t_scaling_recipe,
                    group_lens=group_lens,
                )
            else:
                assert isinstance(a_t, QuantizedTensor)
                quantized_a_t = a_t

            a_row, a_row_scale = quantized_a.qdata, quantized_a.scale_inv
            a_col, a_col_scale = quantized_a_t.qdata, quantized_a_t.scale_inv

            group_offs_padded_rowwise = quantized_a.group_offs

        # --- B: 3D weight (G, N, K). row-wise (rht=F) is the fwd operand; col-wise
        b_scaling_recipe = ScalingRecipe(use_2d_block=True)
        b_t_scaling_recipe = ScalingRecipe(use_2d_block=True)
        if not isinstance(b, QuantizedTensor):
            b_row, b_row_scale, b_col, b_col_scale = quantize_fp4_with_trans(
                b,
                float4_e2m1fn_x2,
                ScalingGranularity.MX_BLOCKWISE,
                block_size=MXFP4_BLOCK_SIZE,
                scaling_recipe=b_scaling_recipe,
                scaling_recipe_for_trans=b_t_scaling_recipe,
            )
        else:
            quantized_b = b
            check_quantized_tensor(quantized_b, config, axis=-1, scaling_recipe=b_scaling_recipe)

            if b_t is None:
                quantized_b_t = QuantizedTensor.quantize(
                    quantized_b.dequantize(),
                    quantized_b.real_dtype,
                    config.granularity,
                    axis=-2,
                    block_size=config.block_size,
                    scaling_recipe=b_t_scaling_recipe,
                )
            else:
                assert isinstance(b_t, QuantizedTensor)
                quantized_b_t = b_t

            b_row, b_row_scale = quantized_b.qdata, quantized_b.scale_inv
            b_col, b_col_scale = quantized_b_t.qdata, quantized_b_t.scale_inv

        total_m = int(a.size(0))
        out = grouped_gemm_fp4_impl(
            a_row,
            b_row,
            a_row_scale,
            b_row_scale,
            group_lens,
            group_offs_padded_rowwise,
            trans_a=False,
            trans_b=True,
            out_dtype=out_dtype,
            granularity=ScalingGranularity.MX_BLOCKWISE.value,
            num_cu=num_cu,
            default_backend=BackendType.FLYDSL.value,
            group_offs_out=group_offs,
        )
        out = out[:total_m]

        ctx.save_for_backward(a_col, a_col_scale, b_col, b_col_scale, group_lens, group_offs)
        ctx.total_m = total_m
        ctx.config = config
        ctx.out_dtype = out_dtype
        ctx.num_cu = num_cu
        return out

    @staticmethod
    def backward(ctx, grad_out):
        grad_out = _ensure_contiguous_grad_out(grad_out)
        a_col, a_col_scale, b_col, b_col_scale, group_lens, group_offs = ctx.saved_tensors

        # --- grad_out: fused grouped dual-quant in one bf16 read
        grad_out_scaling_recipe = ScalingRecipe(
            use_sr=ctx.config.use_gradient_sr,
        )
        grad_out_t_scaling_recipe = ScalingRecipe(
            use_sr=ctx.config.use_gradient_sr,
            use_rht=True,
        )
        (
            grad_out_fp4_row,
            grad_out_scale_row,
            grad_out_t_fp4,
            grad_out_t_scale,
            _,
            group_offs_padded_rowwise,
            group_lens_padded_colwise,
            group_offs_padded_colwise,
        ) = grouped_quantize_fp4_with_trans(
            grad_out,
            float4_e2m1fn_x2,
            ctx.config.granularity,
            group_lens,
            group_offs,
            block_size=ctx.config.block_size,
            scaling_recipe=grad_out_scaling_recipe,
            scaling_recipe_for_trans=grad_out_t_scaling_recipe,
        )

        # --- dgrad: grad_a = gradO_row(rht=T) @ B_col(rht=T)^T, contract N -> [total_m, K] ---
        grad_a = grouped_gemm_fp4_impl(
            grad_out_fp4_row,
            b_col,
            grad_out_scale_row,
            b_col_scale,
            group_lens,
            group_offs_padded_rowwise,
            trans_a=False,
            trans_b=True,
            out_dtype=ctx.out_dtype,
            granularity=ScalingGranularity.MX_BLOCKWISE.value,
            num_cu=ctx.num_cu,
            default_backend=BackendType.FLYDSL.value,
            group_offs_out=group_offs,
        )
        grad_a = grad_a[: ctx.total_m]

        # --- wgrad: grad_b[g] = gradO_col(rht=T) @ A_col(rht=T)^T, contract M_g -> [G, N, K] ---
        grad_b = grouped_gemm_fp4_variable_k_impl(
            grad_out_t_fp4,
            a_col,
            grad_out_t_scale,
            a_col_scale,
            group_lens_padded_colwise,
            group_offs_padded_colwise,
            trans_a=False,
            trans_b=False,
            trans_c=False,
            out_dtype=ctx.out_dtype,
            granularity=ScalingGranularity.MX_BLOCKWISE.value,
            num_cu=ctx.num_cu,
            default_backend=BackendType.FLYDSL.value,
        )

        return (
            grad_a,
            grad_b,  # b
            None,  # a_t
            None,  # b_t
            None,  # group_lens
            None,  # group_offs
            None,  # trans_b
            None,  # out_dtype
            None,  # config
            None,  # num_cu
        )


@torch._dynamo.disable(
    recursive=True,
    reason=(
        "Grouped MXFP4 GEMM composes the dense quantizer per direction and reads "
        "packed FP4 / E8M0 inner tensors inside its autograd.Function.forward; "
        "Dynamo cannot recover Python sources for those graph-internal tensors."
    ),
)
def grouped_gemm_fp4(
    a: Union[torch.Tensor, QuantizedTensor, QuantizedTensorPair],
    b: Union[torch.Tensor, QuantizedTensor, QuantizedTensorPair],
    group_lens: torch.Tensor,
    group_offs: Union[torch.Tensor, None] = None,
    trans_b: bool = True,
    out_dtype: Union[torch.dtype, None] = None,
    config: Union[Float4QuantConfig, None] = None,
    num_cu: int | None = None,
) -> torch.Tensor:
    """Grouped GEMM with MXFP4 (E2M1) quantization, supporting autograd.

    Args:
        a: Activation [total_m, K] (float16 / bfloat16), grouped along M. Can also
            be a pre-quantized grouped :class:`QuantizedTensor`, or a
            :class:`QuantizedTensorPair` carrying both the row-wise ``data`` and the
            col-wise (rht=True) backward operand ``data_t``.
        b: Weight [G, N, K] (trans_b=True, NT layout only). Same pre-quantized
            variants as ``a`` are accepted (``data_t`` is the col-wise dgrad operand).
        group_lens: Per-group row counts [G] (int64).
        group_offs: Optional per-group offsets [G+1]; derived from group_lens if None.
        trans_b: Must be True (NT).
        out_dtype: Output dtype (defaults to promote(a, b)).
        config: Float4QuantConfig (MX_BLOCKWISE). use_preshuffle must be False.
        num_cu: Optional cap on compute units for the persistent grid.

    Returns:
        Output [total_m, N].
    """
    if config is None:
        config = Float4QuantConfig()
    if group_offs is None:
        group_offs = group_offs_from_lens(group_lens)

    if isinstance(a, QuantizedTensorPair):
        a_data, a_data_t = a.data, a.data_t
    else:
        a_data, a_data_t = a, None

    if isinstance(b, QuantizedTensorPair):
        b_data, b_data_t = b.data, b.data_t
    else:
        b_data, b_data_t = b, None

    if out_dtype is None:
        out_dtype = torch.promote_types(a_data.dtype, b_data.dtype)

    assert a_data.ndim == 2, "a must be 2D [total_m, K]"
    assert b_data.ndim == 3, "b must be 3D [G, N, K]"

    if config.granularity == ScalingGranularity.MX_BLOCKWISE:
        return FP4GroupedGemmMXFunc.apply(
            a_data,
            b_data,
            a_data_t,
            b_data_t,
            group_lens,
            group_offs,
            trans_b,
            out_dtype,
            config,
            num_cu,
        )
    raise ValueError(f"Unsupported FP4 ScalingGranularity: {config.granularity}")
