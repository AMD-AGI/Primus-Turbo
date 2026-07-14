###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""MXFP4 grouped GEMM (MX_BLOCKWISE, Triton backend, gfx950).

Mirrors :class:`FP8GroupedGemmMXFunc` but for E2M1 FP4. The MXFP4 training
recipe (https://arxiv.org/pdf/2509.25149) keeps the forward operands plain MX
and applies a 16-point Random Hadamard Transform (RHT) to the backward operands
(grad_out + the cached col-wise transposes). Because the Hadamard is orthogonal
it cancels inside each GEMM **only when both contracted operands share it**, so
the recipes are paired exactly like the dense ``gemm_fp4``:

    fwd   : C    = A_row(rht=F)         @ B_row(rht=F)^T          (contract K)
    dgrad : dA   = gradO_row(rht=T)     @ B_col(rht=T)^T          (contract N)
    wgrad : dB   = gradO_col(rht=T)     @ A_col(rht=T)^T          (contract M_g)

Both grouped activations ``A`` (fwd) and ``grad_out`` (bwd) are quantized by the
fused C++ ``grouped_quantize_mxfp4_dual`` kernel: one bf16 read emits both the
tight-M row-wise operand (fwd/dgrad) and the 128-padded per-group col-wise
operand (variable-K wgrad), with the per-group M zero-pad + GPU-computed offsets
folded into the quant pass (no bf16 scatter, no extra read, no D2H sync ->
CUDA-graph capturable). ``A`` uses rht=F row-wise; ``grad_out`` uses rht=T both
directions (+ optional gradient SR).

The weight ``B`` ([G, N, K]) is dual-quantized in a single batched
``quantize_fp4_with_trans`` call (rht=F row-wise for fwd, rht=T col-wise for
dgrad): the FP4 ``quantize_mxfp4_dual`` kernel walks each group along
``blockIdx.z``, so there is no Python per-group loop and no full-weight
transpose/``contiguous`` copy. This mirrors the MXFP8 weight path
(``FP8GroupedGemmMXFunc`` -> ``quantize_fp8_with_trans`` on the 3D ``(G, N, K)``).
"""

from typing import Union

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
from primus_turbo.pytorch.kernels.grouped_gemm.grouped_gemm_fp4_impl import (
    grouped_gemm_fp4_impl,
    grouped_gemm_fp4_variable_k_impl,
)
from primus_turbo.pytorch.kernels.grouped_gemm.grouped_gemm_utils import (
    group_offs_from_lens,
)
from primus_turbo.pytorch.kernels.quantization.quantization_impl import (
    grouped_quantize_mxfp4_impl,
)
from primus_turbo.pytorch.ops.quantization import quantize_fp4_with_trans

__all__ = ["grouped_gemm_fp4"]


def _quant_weight_dual(b: torch.Tensor):
    """Dual MXFP4 quant of the 3D weight ``b`` (G, N, K) in one batched launch.

    Returns (b_row, b_row_scale, b_col, b_col_scale):
      b_row      (G, N, K_pad/2)  rht=False  -> fwd B operand   (2D-block, axis K)
      b_col      (G, K, N_pad/2)  rht=True   -> dgrad B operand (2D-block, axis N)

    Both directions come from a single batched ``quantize_fp4_with_trans`` call
    over the 3D weight -- the FP4 ``quantize_mxfp4_dual`` kernel walks each group
    with ``blockIdx.z``, so there is no Python per-group loop and no full-weight
    transpose/contiguous copy. This mirrors the MXFP8 grouped GEMM weight path
    (``FP8GroupedGemmMXFunc`` -> ``quantize_fp8_with_trans`` on ``(G, N, K)``).
    """
    return quantize_fp4_with_trans(
        b,
        float4_e2m1fn_x2,
        ScalingGranularity.MX_BLOCKWISE,
        block_size=MXFP4_BLOCK_SIZE,
        scaling_recipe=ScalingRecipe(use_2d_block=True, use_sr=False, use_rht=False),
        scaling_recipe_for_trans=ScalingRecipe(use_2d_block=True, use_sr=False, use_rht=True),
    )


class FP4GroupedGemmMXFunc(torch.autograd.Function):
    """MXFP4 grouped GEMM autograd (MX_BLOCKWISE, NT-only, Triton backend)."""

    @staticmethod
    def forward(ctx, a, b, group_lens, group_offs, trans_b, out_dtype, config, num_cu):
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
        # that padded length (the packed last dim already reflects it), while free
        # dims are handled by the kernel's `% N` wrap + `c_mask`.
        assert K % MXFP4_BLOCK_SIZE == 0, f"K must be a multiple of {MXFP4_BLOCK_SIZE} (got {K})."
        assert N % MXFP4_BLOCK_SIZE == 0, f"N must be a multiple of {MXFP4_BLOCK_SIZE} (got {N})."
        total_m = int(a.size(0))

        # --- A: fused grouped dual-quant in one bf16 read. rowwise (rht=F) is the
        # tight-M fwd operand; colwise (rht=T) is the 128-padded-M wgrad operand.
        # The per-group M zero-pad and its GPU-computed offsets (``padded_offs``,
        # the col layout used by the variable-K wgrad) are folded into the quant
        # pass -- no bf16 scatter, no extra read, no D2H sync. ---
        a_row, a_row_s, a_col, a_col_s, _, padded_offs = grouped_quantize_mxfp4_impl(
            a,
            MXFP4_BLOCK_SIZE,
            group_lens,
            group_offs,
            rowwise_recipe=ScalingRecipe(use_2d_block=False, use_sr=False, use_rht=False),
            colwise_recipe=ScalingRecipe(use_2d_block=False, use_sr=False, use_rht=True),
        )
        b_row, b_row_s, b_col, b_col_s = _quant_weight_dual(b)

        # b_row is FP4-packed (G, N, K/2): the backend derives the (128-padded)
        # contraction K = b.shape[-1]*2 and the free dim N = b.shape[-2]. a_row is
        # the tight-M layout, so group_offs doubles as group_offs_out (no slice).
        out = grouped_gemm_fp4_impl(
            a_row,
            b_row,
            a_row_s,
            b_row_s,
            group_lens,
            group_offs,
            trans_a=False,
            trans_b=True,
            out_dtype=out_dtype,
            granularity=ScalingGranularity.MX_BLOCKWISE.value,
            num_cu=num_cu,
            default_backend=BackendType.FLYDSL.value,
            group_offs_out=group_offs,
        )

        # ``group_lens`` is re-saved so backward can fuse grad_out's dual-quant the
        # same way (its per-group 128-pad layout matches ``padded_offs``).
        ctx.save_for_backward(a_col, a_col_s, b_col, b_col_s, group_lens, group_offs, padded_offs)
        ctx.N = N
        ctx.total_m = total_m
        ctx.config = config
        ctx.out_dtype = out_dtype
        ctx.num_cu = num_cu
        return out

    @staticmethod
    def backward(ctx, grad_out):
        a_col, a_col_s, b_col, b_col_s, group_lens, group_offs, padded_offs = ctx.saved_tensors
        N = ctx.N
        out_dtype, num_cu = ctx.out_dtype, ctx.num_cu
        sr = ctx.config.use_gradient_sr

        grad_out = grad_out.contiguous().view(ctx.total_m, N)

        # --- grad_out: fused grouped dual-quant in one bf16 read (symmetric to A).
        # rowwise (rht=T) is the tight-M dgrad operand; colwise (rht=T) is the
        # 128-padded-M wgrad operand (same layout as ``padded_offs`` / a_col). SR
        # (use_gradient_sr) applies to both directions. No bf16 scatter / D2H. ---
        go_row, go_row_s, go_col, go_col_s, _, _ = grouped_quantize_mxfp4_impl(
            grad_out,
            MXFP4_BLOCK_SIZE,
            group_lens,
            group_offs,
            rowwise_recipe=ScalingRecipe(use_2d_block=False, use_sr=sr, use_rht=True),
            colwise_recipe=ScalingRecipe(use_2d_block=False, use_sr=sr, use_rht=True),
        )

        # --- dgrad: grad_a = gradO_row(rht=T) @ B_col(rht=T)^T, contract N -> [total_m, K] ---
        # b_col is FP4-packed (G, K, N/2): the backend derives the (128-padded)
        # contraction from the packed last dim and the free dim K = b.shape[-2].
        grad_a = grouped_gemm_fp4_impl(
            go_row,
            b_col,
            go_row_s,
            b_col_s,
            group_lens,
            group_offs,
            trans_a=False,
            trans_b=True,
            out_dtype=out_dtype,
            granularity=ScalingGranularity.MX_BLOCKWISE.value,
            num_cu=num_cu,
            default_backend=BackendType.FLYDSL.value,
            group_offs_out=group_offs,
        )

        # --- wgrad: grad_b[g] = gradO_col(rht=T) @ A_col(rht=T)^T, contract M_g -> [G, N, K] ---
        # OUT_M = go_col.shape[0] (=N), OUT_N = a_col.shape[0] (=K), G = group count,
        # all derived in the backend; padded_offs are the per-group M offsets.
        grad_b = grouped_gemm_fp4_variable_k_impl(
            go_col,
            a_col,
            go_col_s,
            a_col_s,
            group_lens,
            padded_offs,
            trans_a=False,
            trans_b=False,
            trans_c=False,
            out_dtype=out_dtype,
            granularity=ScalingGranularity.MX_BLOCKWISE.value,
            num_cu=num_cu,
            default_backend=BackendType.FLYDSL.value,
        )

        # forward args: (a, b, group_lens, group_offs, trans_b, out_dtype, config, num_cu)
        return grad_a, grad_b, None, None, None, None, None, None


@torch._dynamo.disable(
    recursive=True,
    reason=(
        "Grouped MXFP4 GEMM composes the dense quantizer per direction and reads "
        "packed FP4 / E8M0 inner tensors inside its autograd.Function.forward; "
        "Dynamo cannot recover Python sources for those graph-internal tensors."
    ),
)
def grouped_gemm_fp4(
    a: torch.Tensor,
    b: torch.Tensor,
    group_lens: torch.Tensor,
    group_offs: Union[torch.Tensor, None] = None,
    trans_b: bool = True,
    out_dtype: Union[torch.dtype, None] = None,
    config: Union[Float4QuantConfig, None] = None,
    num_cu: int | None = None,
) -> torch.Tensor:
    """Grouped GEMM with MXFP4 (E2M1) quantization, supporting autograd.

    Args:
        a: Activation [total_m, K] (float16 / bfloat16), grouped along M.
        b: Weight [G, N, K] (trans_b=True, NT layout only).
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
    if out_dtype is None:
        out_dtype = torch.promote_types(a.dtype, b.dtype)

    assert a.ndim == 2, "a must be 2D [total_m, K]"
    assert b.ndim == 3, "b must be 3D [G, N, K]"

    if config.granularity == ScalingGranularity.MX_BLOCKWISE:
        return FP4GroupedGemmMXFunc.apply(a, b, group_lens, group_offs, trans_b, out_dtype, config, num_cu)
    raise ValueError(f"Unsupported FP4 ScalingGranularity: {config.granularity}")
