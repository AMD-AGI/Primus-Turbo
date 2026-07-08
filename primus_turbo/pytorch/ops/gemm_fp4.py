###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from typing import Optional, Union

import torch

from primus_turbo.flydsl.quant.mxfp4_quant_kernel import (
    dual2_eligible,
    dual_eligible,
    flydsl_dual_quant,
    flydsl_dual_quant2,
)
from primus_turbo.pytorch.core.backend import BackendType
from primus_turbo.pytorch.core.low_precision import (
    Float4QuantConfig,
    Format,
    ScalingGranularity,
    ScalingRecipe,
    check_mxfp4_support,
)
from primus_turbo.pytorch.core.quantized_tensor import (
    QuantizedTensor,
    QuantizedTensorPair,
    check_quantized_tensor,
)
from primus_turbo.pytorch.kernels.gemm.gemm_fp4_impl import gemm_fp4_impl
from primus_turbo.pytorch.kernels.quantization.quantization_impl import (
    quantize_mxfp4_impl,
)

__all__ = ["gemm_fp4"]


def _quantize_mxfp4_row_col(x, fp4_dtype, config, row_recipe, col_recipe):
    """Fused rowwise + colwise MXFP4 quant in a single bf16 read.

    Returns ``(row_qt, col_qt)`` QuantizedTensors equivalent to two separate
    ``QuantizedTensor.quantize`` calls (``axis=1`` with ``row_recipe`` and
    ``axis=0`` with ``col_recipe``) but issued as one dual quant kernel -- one
    activation read and one launch instead of two, and the colwise output uses
    the dual kernel's coalesced LDS store. Output layouts are identical to the
    per-direction calls. This optimizes quantization independently; it does not
    fuse quant into the GEMM.
    """
    # FlyDSL fused dual (bit-exact vs C++ for 2d=F / SR-off / unshuffled recipes
    # with 128/256-aligned dims) -- one compiled launch (~3us dispatch) instead of
    # the C++ op (~6us). Falls back to the C++ dual for every other case (e.g. the
    # 2d-block weight cast). This optimizes quant only; no quant<->gemm fusion.
    use_flydsl = (
        x.dtype == torch.bfloat16
        and x.dim() == 2
        and x.is_contiguous()
        and dual_eligible(x.size(0), x.size(1), row_recipe, col_recipe)
    )
    if use_flydsl:
        row_data, row_scale, col_data, col_scale = flydsl_dual_quant(
            x,
            fp4_dtype,
            row_recipe.use_rht,
            col_recipe.use_rht,
            row_recipe.use_2d_block,
            col_recipe.use_2d_block,
        )
    else:
        row_data, row_scale, col_data, col_scale = quantize_mxfp4_impl(
            x,
            fp4_dtype,
            None,
            config.block_size,
            with_trans=True,
            scaling_recipe=row_recipe,
            scaling_recipe_for_trans=col_recipe,
        )
    return _make_row_col_qt(
        x.size(),
        x.dtype,
        fp4_dtype,
        config,
        row_recipe,
        col_recipe,
        row_data,
        row_scale,
        col_data,
        col_scale,
    )


def _make_row_col_qt(
    x_size,
    x_dtype,
    fp4_dtype,
    config,
    row_recipe,
    col_recipe,
    row_data,
    row_scale,
    col_data,
    col_scale,
):
    """Wrap fused row/col quant outputs into the (row_qt, col_qt) pair."""
    row_qt = QuantizedTensor(
        row_data,
        row_scale,
        shape=x_size,
        orig_dtype=x_dtype,
        dest_dtype=fp4_dtype,
        granularity=config.granularity,
        block_size=config.block_size,
        scaling_recipe=row_recipe,
        requires_grad=False,
        quantized_axis=1,
    )
    col_qt = QuantizedTensor(
        col_data,
        col_scale,
        shape=x_size,
        orig_dtype=x_dtype,
        dest_dtype=fp4_dtype,
        granularity=config.granularity,
        block_size=config.block_size,
        scaling_recipe=col_recipe,
        requires_grad=False,
        quantized_axis=0,
    )
    return row_qt, col_qt


def _quantize_mxfp4_row_col_ab(a, b, fp4_dtype, config, recipes):
    """Merged A+B rowwise+colwise MXFP4 quant in one kernel launch (quant-only,
    no quant<->GEMM fusion). Returns ``(a_row_qt, a_col_qt, b_row_qt, b_col_qt)``,
    or ``None`` if not eligible so the caller falls back to two separate duals."""
    a_row_rec, a_t_rec, b_row_rec, b_t_rec = recipes
    ok = (
        a.dtype == torch.bfloat16
        and b.dtype == torch.bfloat16
        and a.dim() == 2
        and b.dim() == 2
        and a.is_contiguous()
        and b.is_contiguous()
        and dual2_eligible(
            a.size(0),
            a.size(1),
            b.size(0),
            b.size(1),
            a_row_rec,
            a_t_rec,
            b_row_rec,
            b_t_rec,
        )
    )
    if not ok:
        return None
    a_recipes = (a_row_rec.use_rht, a_t_rec.use_rht, a_row_rec.use_2d_block, a_t_rec.use_2d_block)
    b_recipes = (b_row_rec.use_rht, b_t_rec.use_rht, b_row_rec.use_2d_block, b_t_rec.use_2d_block)
    (a_ro, a_rs, a_co, a_cs), (b_ro, b_rs, b_co, b_cs) = flydsl_dual_quant2(
        a, b, fp4_dtype, a_recipes, b_recipes
    )
    a_row_qt, a_col_qt = _make_row_col_qt(
        a.size(), a.dtype, fp4_dtype, config, a_row_rec, a_t_rec, a_ro, a_rs, a_co, a_cs
    )
    b_row_qt, b_col_qt = _make_row_col_qt(
        b.size(), b.dtype, fp4_dtype, config, b_row_rec, b_t_rec, b_ro, b_rs, b_co, b_cs
    )
    return a_row_qt, a_col_qt, b_row_qt, b_col_qt


class FP4GemmMXFunction(torch.autograd.Function):
    """
    MXFP4 scaling recipe reference: https://arxiv.org/pdf/2509.25149
    """

    @staticmethod
    def get_fp4_dtype(format: Format):
        if format == Format.E2M1_X2:
            return torch.float4_e2m1fn_x2
        else:
            raise ValueError(f"Unsupported FP4 format: {format}")

    @staticmethod
    def forward(
        ctx,
        a: Union[torch.Tensor, QuantizedTensor],
        b: Union[torch.Tensor, QuantizedTensor],
        a_t: Optional[QuantizedTensor],
        b_t: Optional[QuantizedTensor],
        trans_a: bool,
        trans_b: bool,
        out_dtype: torch.dtype,
        config: Float4QuantConfig,
    ):
        supported_mxfp4_backend, reason = check_mxfp4_support()
        assert supported_mxfp4_backend, reason

        preshuffle = config.use_preshuffle
        fp4_dtype = FP4GemmMXFunction.get_fp4_dtype(config.format)

        # Recipes for the four cast directions. The rowwise (axis=1) casts feed
        # the fwd GEMM; the colwise (axis=0) + RHT casts are the backward
        # (a_t / b_t) operands. When a raw tensor is available and no external
        # transpose was supplied, both casts share a single bf16 read via the
        # dual quantizer (see ``_quantize_mxfp4_row_col``).
        a_scaling_recipe = ScalingRecipe(
            use_2d_block=False,
            use_sr=False,
            use_rht=False,
            shuffle_scale=preshuffle,
            shuffle_out=False,
        )
        a_t_scaling_recipe = ScalingRecipe(
            use_2d_block=False,
            use_sr=False,
            use_rht=True,
            shuffle_scale=preshuffle,
            shuffle_out=preshuffle,
        )
        b_scaling_recipe = ScalingRecipe(
            use_2d_block=True,
            use_sr=False,
            use_rht=False,
            shuffle_scale=preshuffle,
            shuffle_out=preshuffle,
        )
        b_t_scaling_recipe = ScalingRecipe(
            use_2d_block=True,
            use_sr=False,
            use_rht=True,
            shuffle_scale=preshuffle,
            shuffle_out=preshuffle,
        )

        # ---- Merged A+B dual: when both operands are raw bf16 and eligible, run
        # all four casts (A row/col + B row/col) in ONE grid launch, saving a host
        # dispatch per fwd step. Quant-only; no quant<->gemm fusion. ----
        quantized_a_t = None
        quantized_b_t = None
        a_fp4 = None
        b_fp4 = None
        if (
            not isinstance(a, QuantizedTensor)
            and a_t is None
            and not isinstance(b, QuantizedTensor)
            and b_t is None
        ):
            merged = _quantize_mxfp4_row_col_ab(
                a,
                b,
                fp4_dtype,
                config,
                (a_scaling_recipe, a_t_scaling_recipe, b_scaling_recipe, b_t_scaling_recipe),
            )
            if merged is not None:
                a_fp4, quantized_a_t, b_fp4, quantized_b_t = merged

        # ---- A: rowwise fwd operand (+ fused colwise/RHT a_t when possible) ----
        if a_fp4 is None:
            if not isinstance(a, QuantizedTensor) and a_t is None:
                a_fp4, quantized_a_t = _quantize_mxfp4_row_col(
                    a, fp4_dtype, config, a_scaling_recipe, a_t_scaling_recipe
                )
            elif isinstance(a, QuantizedTensor):
                check_quantized_tensor(a, config, scaling_recipe=a_scaling_recipe)
                a_fp4 = a
            else:
                a_fp4 = QuantizedTensor.quantize(
                    a,
                    fp4_dtype,
                    config.granularity,
                    block_size=config.block_size,
                    scaling_recipe=a_scaling_recipe,
                    axis=1,
                )

        # ---- B: rowwise fwd operand (+ fused colwise/RHT b_t when possible) ----
        if b_fp4 is None:
            if not isinstance(b, QuantizedTensor) and b_t is None:
                b_fp4, quantized_b_t = _quantize_mxfp4_row_col(
                    b, fp4_dtype, config, b_scaling_recipe, b_t_scaling_recipe
                )
            elif isinstance(b, QuantizedTensor):
                check_quantized_tensor(b, config, scaling_recipe=b_scaling_recipe)
                b_fp4 = b
            else:
                b_fp4 = QuantizedTensor.quantize(
                    b,
                    fp4_dtype,
                    config.granularity,
                    block_size=config.block_size,
                    scaling_recipe=b_scaling_recipe,
                    axis=1,
                )

        # NT layout
        out = gemm_fp4_impl(
            a_fp4.qdata,
            a_fp4.scale_inv,
            False,
            b_fp4.qdata,
            b_fp4.scale_inv,
            True,
            out_dtype,
            False,
            granularity=config.granularity.value,
            default_backend=(BackendType.AITER if preshuffle else BackendType.HIPBLASLT).value,
            preshuffled=preshuffle,
        )

        # Backward needs a col-wise (axis=0) version of A/B with an RHT recipe.
        # If the caller pre-quantized this and passed it via ``a_t`` / ``b_t``,
        # reuse it directly; otherwise derive it.
        #
        # Caution: do NOT derive from ``a_fp4.dequantize()`` when
        # ``shuffle_scale=True`` is in the recipe -- ``dequantize_fp4``
        # treats ``scale_inv`` as canonical-layout and a shuffled scale
        # gives garbage. When a raw high-precision tensor was passed in,
        # re-quantize from it directly; this matches what
        # ``_quantize_mxfp4_dual`` would do under one fused op.
        # ``quantized_a_t`` / ``quantized_b_t`` are already set when the dual
        # fast path ran above; otherwise derive the colwise/RHT transpose here.
        if quantized_a_t is None:
            if a_t is not None:
                quantized_a_t = a_t
            else:
                a_for_t = a if not isinstance(a, QuantizedTensor) else a_fp4.dequantize()
                quantized_a_t = QuantizedTensor.quantize(
                    a_for_t,
                    a_fp4.real_dtype,
                    config.granularity,
                    block_size=config.block_size,
                    axis=0,
                    scaling_recipe=a_t_scaling_recipe,
                )

        if quantized_b_t is None:
            if b_t is not None:
                quantized_b_t = b_t
            else:
                b_for_t = b if not isinstance(b, QuantizedTensor) else b_fp4.dequantize()
                quantized_b_t = QuantizedTensor.quantize(
                    b_for_t,
                    b_fp4.real_dtype,
                    config.granularity,
                    block_size=config.block_size,
                    axis=0,
                    scaling_recipe=b_t_scaling_recipe,
                )
        ctx.save_for_backward(
            quantized_a_t.qdata, quantized_a_t.scale_inv, quantized_b_t.qdata, quantized_b_t.scale_inv
        )

        ctx.trans_a = trans_a
        ctx.trans_b = trans_b
        ctx.out_dtype = out_dtype
        ctx.config = config
        ctx.a_fp4_dtype = a_fp4.real_dtype
        ctx.b_fp4_dtype = b_fp4.real_dtype

        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        a_fp4_t, a_t_scale_inv, b_fp4_t, b_t_scale_inv = ctx.saved_tensors
        grad_out_dtype = FP4GemmMXFunction.get_fp4_dtype(
            ctx.config.format,
        )

        grad_out = grad_out.view(grad_out.shape[0], -1).contiguous()

        preshuffle = ctx.config.use_preshuffle
        default_backend = (BackendType.AITER if preshuffle else BackendType.HIPBLASLT).value

        # grad_out feeds two GEMMs: rowwise (dgrad, grad_a) and colwise (wgrad,
        # grad_b). Both casts use the same RHT/SR recipe on the same grad_out, so
        # fuse them into one bf16 read via the dual quantizer.
        grad_out_scaling_recipe = ScalingRecipe(
            use_2d_block=False,
            use_sr=ctx.config.use_gradient_sr,
            use_rht=True,
            shuffle_scale=preshuffle,
            shuffle_out=False,
        )
        grad_out_t_scaling_recipe = ScalingRecipe(
            use_2d_block=False,
            use_sr=ctx.config.use_gradient_sr,
            use_rht=True,
            shuffle_scale=preshuffle,
            shuffle_out=False,
        )
        quantized_grad_out, quantized_grad_out_t = _quantize_mxfp4_row_col(
            grad_out,
            grad_out_dtype,
            ctx.config,
            grad_out_scaling_recipe,
            grad_out_t_scaling_recipe,
        )

        # NOTE: convert NN layout to NT layout because MXFP4 only supports NT layout on hipblaslt.
        grad_a = gemm_fp4_impl(
            quantized_grad_out.qdata,
            quantized_grad_out.scale_inv,
            False,
            b_fp4_t,
            b_t_scale_inv,
            True,
            ctx.out_dtype,
            False,
            granularity=ctx.config.granularity.value,
            default_backend=default_backend,
            preshuffled=preshuffle,
        )

        # NOTE: convert TN layout to NT layout because MXFP4 only supports NT layout on hipblaslt.
        grad_b = gemm_fp4_impl(
            quantized_grad_out_t.qdata,
            quantized_grad_out_t.scale_inv,
            False,
            a_fp4_t,
            a_t_scale_inv,
            True,
            ctx.out_dtype,
            False,
            granularity=ctx.config.granularity.value,
            default_backend=default_backend,
            preshuffled=preshuffle,
        )

        # Grads correspond to forward args:
        #   (a, b, a_t, b_t, trans_a, trans_b, out_dtype, config)
        return grad_a, grad_b, None, None, None, None, None, None


@torch._dynamo.disable(
    recursive=True,
    reason=(
        "FP4 GEMM constructs QuantizedTensor wrapper subclasses inside its "
        "autograd.Function.forward and reads their inner tensors (data / scale_inv). "
        "Dynamo cannot recover Python sources for those graph-internal inner tensors, "
    ),
)
def gemm_fp4(
    a: Union[torch.Tensor, QuantizedTensor, QuantizedTensorPair],
    b: Union[torch.Tensor, QuantizedTensor, QuantizedTensorPair],
    trans_a: bool = False,
    trans_b: bool = False,
    out_dtype: Union[torch.dtype, None] = None,
    config: Union[Float4QuantConfig, None] = None,
) -> torch.Tensor:
    """General matrix multiplication (GEMM) with FP4 quantization, supporting autograd.

    Automatically quantizes inputs to FP4 format during forward and backward passes
    to accelerate training and inference. When ``a`` or ``b`` is already a
    :class:`QuantizedTensor`, its quantized data / scale is reused directly,
    skipping the forward-direction quantization. If a :class:`QuantizedTensorPair`
    wrapper is passed instead, the optional ``data_t`` field is also forwarded
    and reused as the col-wise / RHT transpose cache for backward.

    Pre-quantized input contract:
        When passing a pre-quantized :class:`QuantizedTensor` (or
        :class:`QuantizedTensorPair`), the caller's :class:`ScalingRecipe`
        must match what ``FP4GemmMXFunction`` constructs internally; this
        is checked by :func:`check_quantized_tensor` via strict equality.
        Under the AITER backend the recipe includes
        ``shuffle_scale`` / ``shuffle_out`` flags derived from
        Recommended pattern::

            a_recipe = ScalingRecipe(
                use_2d_block=False, use_sr=False, use_rht=False,
                shuffle_scale=preshuffle, shuffle_out=False,
            )
            b_recipe = ScalingRecipe(
                use_2d_block=True, use_sr=False, use_rht=False,
                shuffle_scale=preshuffle, shuffle_out=preshuffle,
            )

    Args:
        a: Input matrix a with shape (M, K), must be 2D tensor. The A matrix should be activaton.
            Can also be a pre-quantized :class:`QuantizedTensor` (forward only)
            or a :class:`QuantizedTensorPair` carrying both ``data`` and the
            backward-direction ``data_t``.
        b: Input matrix b with shape (K, N) or (N, K), must be 2D tensor. The B matrix should be weight.
            Same pre-quantized variants as ``a`` are accepted.
        trans_a: Whether to transpose matrix a
        trans_b: Whether to transpose matrix b, if True b shape is (N, K)
        out_dtype: Output data type, defaults to None (auto-inferred)
        config: FP4 quantization config

    Returns:
        torch.Tensor: Output matrix with shape (M, N)

    Scaling Granularity (config.granularity):
        - MX_BLOCKWISE

    FP4 Format (config.format):
        - E2M1_X2

    Example::

        >>> # Basic usage
        >>> a = torch.randn(128, 512, device='cuda')
        >>> b = torch.randn(512, 256, device='cuda')
        >>> out = gemm_fp4(a, b)
        >>>
        >>> # ROWWISE quantization
        >>> config = Float4QuantConfig()
        >>> out = gemm_fp4(a, b, trans_b=True, config=config)

    """
    if config is None:
        config = Float4QuantConfig()

    if isinstance(a, QuantizedTensorPair):
        a_data, a_data_t = a.data, a.data_t
    else:
        a_data, a_data_t = a, None

    if isinstance(b, QuantizedTensorPair):
        b_data, b_data_t = b.data, b.data_t
    else:
        b_data, b_data_t = b, None

    assert a_data.ndim == 2, "Only 2D tensors are supported"
    assert b_data.ndim == 2, "Only 2D tensors are supported"

    if out_dtype is None:
        out_dtype = torch.promote_types(a_data.dtype, b_data.dtype)

    if config.granularity == ScalingGranularity.MX_BLOCKWISE:
        return FP4GemmMXFunction.apply(
            a_data, b_data, a_data_t, b_data_t, trans_a, trans_b, out_dtype, config
        )
    else:
        raise ValueError(f"Unsupported FP4 ScalingGranularity: {config.granularity}")
