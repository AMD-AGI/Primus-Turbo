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
from primus_turbo.pytorch.kernels.quantization.quantization_impl import (
    quant_fp8_blockwise_for_weight_impl,
    quant_fp8_blockwise_impl,
    quant_fp8_blockwise_segment_m_impl,
)
from primus_turbo.pytorch.ops.quantization import quantize_fp8
from primus_turbo.triton.grouped_gemm.grouped_gemm_mxfp8_kernel import (
    grouped_gemm_mxfp8_triton_kernel,
)
from primus_turbo.triton.grouped_gemm.grouped_gemm_mxfp8_variable_k_kernel import (
    grouped_gemm_mxfp8_variable_k_triton_kernel,
)
from primus_turbo.triton.quantization.mxfp8_quant_kernels import (
    MXFP8WeightPrequant,
    quant_mxfp8_colwise_jagged,
    quant_mxfp8_dual_jagged,
    quant_mxfp8_rowwise,
    quant_mxfp8_weight_dgrad,
    quant_mxfp8_weight_fwd,
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
            default_backend=BackendType.CK.value,
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
            default_backend=BackendType.CK.value,
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
            default_backend=BackendType.CK.value,
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
            default_backend=BackendType.CK.value,
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
            default_backend=BackendType.CK.value,
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
            default_backend=BackendType.CK.value,
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
            default_backend=BackendType.CK.value,
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
            default_backend=BackendType.CK.value,
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
            default_backend=BackendType.CK.value,
        )

        return grad_a, grad_b, None, None, None, None, None


class FP8GroupedGemmMXFunc(torch.autograd.Function):
    """MX-FP8 grouped GEMM autograd Function (fwd + dgrad + wgrad all mxfp8).

    Forward: quantize A rowwise (scales along K) and B with scales along K
    (N-first layout), call ``grouped_gemm_mxfp8_triton_kernel`` which emits
    native ``v_mfma_scale_f32_32x32x64_f8f6f4`` on gfx950.

    A's rowwise + colwise quant are produced in a single HBM read via
    ``quant_mxfp8_dual`` so the save-for-backward cost is amortised. The
    same dual pattern is used for ``grad_out`` in the backward pass.

    Backward:
      - dgrad: re-quantise grad_out rowwise and reuse the forward kernel with
        ``trans_b=True``; reduction axis is the original N.
      - wgrad: call the variable-K MX-FP8 kernel with scales grouped along M
        for both A and grad_out (the reduction axis).

    Under ``torch.no_grad()`` the backward-save quant steps are skipped.

    Constraints:
      - ``trans_b=False`` only (matches the shipping gpt_oss_20B path).
      - Every per-expert segment length must be a multiple of 32 so no MX
        scale group spans two expert groups. The target training shape
        (M_total=65536, G=32 → M_g=2048) satisfies this trivially.
    """

    @staticmethod
    def forward(
        ctx,
        a: torch.Tensor,
        b: torch.Tensor,
        group_lens: torch.Tensor,
        group_offs: torch.Tensor,
        trans_b: bool,
        config: Float8QuantConfig,
        num_cu: int | None,
        b_fp8_fwd_prequant: torch.Tensor | None = None,
        b_scale_fwd_prequant: torch.Tensor | None = None,
        b_fp8_dgrad_prequant: torch.Tensor | None = None,
        b_scale_dgrad_prequant: torch.Tensor | None = None,
    ):
        assert config.granularity == ScalingGranularity.MX_BLOCKWISE
        assert a.ndim == 2, "A must be 2D [M_total, K]."
        assert b.ndim == 3, "B must be 3D [G, K, N] (trans_b=False) or [G, N, K] (trans_b=True)."
        assert trans_b is False, "MX_BLOCKWISE currently only supports trans_b=False."
        out_dtype = a.dtype
        assert out_dtype in [torch.float16, torch.bfloat16]
        use_prequant = b_fp8_fwd_prequant is not None

        # ``torch.is_grad_enabled()`` reads False inside autograd.Function.forward
        # by design — check input ``requires_grad`` to detect training vs
        # inference. Under ``torch.no_grad()`` autograd never calls backward,
        # so inference pays only the forward rowwise quant + forward kernel.
        need_bwd = a.requires_grad or b.requires_grad

        # Forward A quant. If bwd is needed, use the fused dual-jagged quant
        # which produces both rowwise (for fwd kernel) + jagged colwise (for
        # wgrad save) in a single HBM read of A. When inference-only, just
        # the rowwise quant.
        if need_bwd:
            a_fp8, a_scale, a_fp8_col, a_scale_col, scale_offs = quant_mxfp8_dual_jagged(
                a, group_offs, group_lens,
            )
        else:
            a_fp8, a_scale = quant_mxfp8_rowwise(a)
        if use_prequant:
            b_fp8_fwd = b_fp8_fwd_prequant
            b_scale_fwd = b_scale_fwd_prequant
        else:
            b_fp8_fwd, b_scale_fwd = quant_mxfp8_weight_fwd(b)

        out = grouped_gemm_mxfp8_triton_kernel(
            a_fp8,
            b_fp8_fwd,
            a_scale,
            b_scale_fwd,
            group_offs,
            trans_b=False,
            out_dtype=out_dtype,
        )

        if need_bwd:
            # (a_fp8_col, a_scale_col, scale_offs already produced above by the
            # fused quant_mxfp8_dual_jagged — same HBM read as rowwise A.)
            # B dgrad-layout quant — skipped if the caller provided a prequant.
            if use_prequant:
                b_fp8_dgrad = b_fp8_dgrad_prequant
                b_scale_dgrad = b_scale_dgrad_prequant
            else:
                b_fp8_dgrad, b_scale_dgrad = quant_mxfp8_weight_dgrad(b)
            ctx.save_for_backward(
                a_fp8_col, a_scale_col,
                b_fp8_dgrad, b_scale_dgrad,
                group_lens, group_offs, scale_offs,
            )
        ctx.trans_b = trans_b
        ctx.config = config
        ctx.out_dtype = out_dtype
        ctx.num_cu = num_cu
        ctx.b_shape = b.shape
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        grad_out = _ensure_contiguous_grad_out(grad_out)
        out_dtype = ctx.out_dtype
        (
            a_fp8_col, a_scale_col,
            b_fp8_dgrad, b_scale_dgrad,
            group_lens, group_offs, scale_offs,
        ) = ctx.saved_tensors
        # Prequant tensors (if any) aren't differentiated through; return
        # None for them at the end of backward.

        # dgrad uses rowwise grad_out (scales along reduction axis = original N)
        # — no cross-group issue. wgrad uses jagged colwise grad_out (scales
        # grouped along reduction axis = M, per-expert layout). Fuse both into
        # a single HBM read of grad_out.
        (
            grad_out_fp8_row, grad_out_scale_row,
            grad_out_fp8_col, grad_out_scale_col,
            _go_scale_offs,
        ) = quant_mxfp8_dual_jagged(grad_out, group_offs, group_lens)

        # ── dgrad: dA = grad_out @ B^T via mxfp8 kernel with trans_b=True.
        grad_a = grouped_gemm_mxfp8_triton_kernel(
            grad_out_fp8_row,
            b_fp8_dgrad,
            grad_out_scale_row,
            b_scale_dgrad,
            group_offs,
            trans_b=True,
            out_dtype=out_dtype,
        )

        # ── wgrad: dB[g] = A^T @ grad_out per group via MX-FP8 variable-K kernel
        # with jagged scale layout. Uses group_offs (M-space) + scale_offs (scale-space).
        grad_b = grouped_gemm_mxfp8_variable_k_triton_kernel(
            a_fp8_col,
            grad_out_fp8_col,
            a_scale_col,
            grad_out_scale_col,
            group_offs,
            scale_offs,
            out_dtype=out_dtype,
        )

        # Signature: (a, b, group_lens, group_offs, trans_b, config, num_cu,
        #             b_fp8_fwd_prequant, b_scale_fwd_prequant,
        #             b_fp8_dgrad_prequant, b_scale_dgrad_prequant)
        return grad_a, grad_b, None, None, None, None, None, None, None, None, None


def grouped_gemm_fp8(
    a: torch.Tensor,
    b: Union[torch.Tensor, MXFP8WeightPrequant],
    group_lens: torch.Tensor,
    group_offs: torch.Tensor | None = None,
    trans_b: bool = True,
    config: Union[Float8QuantConfig, None] = None,
    num_cu: int | None = None,
) -> torch.Tensor:
    """Grouped GEMM with optional FP8 / MX-FP8 quant.

    ``b`` may be either a bf16/fp16 tensor (quantised internally every call)
    or, for the ``MX_BLOCKWISE`` path, a :class:`MXFP8WeightPrequant` wrapper
    obtained from :func:`prequant_mxfp8_weights`. The wrapper lifts the B
    quant out of the hot forward path — useful for gradient-accumulation
    training loops where weights are constant across micro-batches.
    """
    supported_dtypes = [torch.bfloat16, torch.float16]
    assert a.dtype in supported_dtypes, f"Unsupported dtype {a.dtype}, expected one of {supported_dtypes}"

    # Unwrap prequant container (only valid for MX_BLOCKWISE).
    b_prequant = None
    if isinstance(b, MXFP8WeightPrequant):
        if config is None or config.granularity != ScalingGranularity.MX_BLOCKWISE:
            raise ValueError(
                "MXFP8WeightPrequant can only be used with "
                "ScalingGranularity.MX_BLOCKWISE; got config.granularity="
                f"{getattr(config, 'granularity', None)}"
            )
        b_prequant = b
        b = b_prequant.b_bf16

    assert b.dtype in supported_dtypes, f"Unsupported dtype {b.dtype}, expected one of {supported_dtypes}"

    if group_offs is None:
        group_offs = grouped_gemm_compute_offs(group_lens)
    if config is None:
        config = Float8QuantConfig()

    if config.granularity == ScalingGranularity.TENSORWISE:
        return FP8GroupedGemmTensorFunc.apply(a, b, group_lens, group_offs, trans_b, config, num_cu)
    elif config.granularity == ScalingGranularity.ROWWISE:
        return FP8GroupedGemmRowFunc.apply(a, b, group_lens, group_offs, trans_b, config, num_cu)
    elif config.granularity == ScalingGranularity.BLOCKWISE:
        return FP8GroupedGemmBlockFunc.apply(a, b, group_lens, group_offs, trans_b, config, num_cu)
    elif config.granularity == ScalingGranularity.MX_BLOCKWISE:
        if b_prequant is not None:
            return FP8GroupedGemmMXFunc.apply(
                a, b, group_lens, group_offs, trans_b, config, num_cu,
                b_prequant.b_fp8_fwd, b_prequant.b_scale_fwd,
                b_prequant.b_fp8_dgrad, b_prequant.b_scale_dgrad,
            )
        return FP8GroupedGemmMXFunc.apply(
            a, b, group_lens, group_offs, trans_b, config, num_cu,
            None, None, None, None,
        )
    else:
        raise ValueError(f"Unsupported FP8 ScalingGranularity: {config.granularity}")
