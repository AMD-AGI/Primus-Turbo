###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Dense SwiGLU MLP on dense MXFP4 ``kernel_gemm_4w`` + ``StoreCSwiGLU``.

P1: MLP-up GEMM fuses SiLU(gate)*up into the same store slot as the unfused
    FlyDSL GEMM. fc2 and both backwards stay on ``gemm_fp4`` (plain
    ``kernel_gemm_4w``). Router ``probs`` are ones.

P2: same GEMM writes the row/col MXFP4 pair fc2 / wgrad consume
    (``StoreCSwiGLUQuant``). ``l1`` stays BF16 for the dSwiGLU kernel. Gated by
    ``fuse_act_quant`` / ``PRIMUS_TURBO_DENSE_MLP_FUSE_QUANT``.

P3: fc2 dgrad fuses dSwiGLU + dual-quant of ``grad_l1``
    (``StoreCdSwiGLUQuadQuant``). Gated by ``fuse_dglu`` /
    ``PRIMUS_TURBO_DENSE_MLP_FUSE_DGLU`` (requires P2).

Do not route this through G=1 grouped GEMM: that path is graph-correct and a
TTT loss on Llama-3.1-8B.
"""

import os
from typing import Union

import torch

from primus_turbo.flydsl.gemm.gemm_mxfp4_kernel import dense_glu_epi_quant_supported
from primus_turbo.flydsl.utils.swiglu_kernel import swiglu_backward_dense_flydsl
from primus_turbo.pytorch.core.backend import BackendType
from primus_turbo.pytorch.core.low_precision import (
    MXFP4_BLOCK_SIZE,
    Float4QuantConfig,
    ScalingGranularity,
    ScalingRecipe,
    check_mxfp4_support,
    float4_e2m1fn_x2,
)
from primus_turbo.pytorch.kernels.gemm.gemm_fp4_impl import (
    gemm_fp4_dglu_quant_impl,
    gemm_fp4_glu_bf16_impl,
    gemm_fp4_glu_quant_impl,
    gemm_fp4_impl,
)
from primus_turbo.pytorch.ops.gemm_fp4 import _bgrad_gemm_fp4_impl_wrapper, gemm_fp4
from primus_turbo.pytorch.ops.quantization import quantize_fp4_with_trans
from primus_turbo.pytorch.ops.utils import (
    _ensure_contiguous_grad_out,
    _get_dummy_wgrad,
    _setup_fused_grad_accum,
)

__all__ = ["dense_mlp_fp4"]

_SUPPORTED_ACTIVATIONS = ("silu",)
_PROBS_ONES: dict = {}


def _probs_ones(M: int, device) -> torch.Tensor:
    key = (M, str(device))
    t = _PROBS_ONES.get(key)
    if t is None:
        t = torch.ones(M, device=device, dtype=torch.float32)
        _PROBS_ONES[key] = t
    return t


def _quantize_weight(w, config: Float4QuantConfig):
    recipe = ScalingRecipe(use_2d_block=True)
    return quantize_fp4_with_trans(
        w,
        float4_e2m1fn_x2,
        ScalingGranularity.MX_BLOCKWISE,
        block_size=MXFP4_BLOCK_SIZE,
        scaling_recipe=recipe,
        scaling_recipe_for_trans=recipe,
    )


def _dswiglu(dact: torch.Tensor, l1: torch.Tensor, probs: torch.Tensor) -> torch.Tensor:
    """SwiGLU gradient on the FlyDSL kernel the MoE path already uses.

    ``probs`` are ones for a dense MLP, so the routing-weight form is off.
    """
    del probs
    return swiglu_backward_dense_flydsl(dact, l1, clamp=False)


class FP4DenseGluMXFunc(torch.autograd.Function):
    """MXFP4 dense GEMM + SwiGLU (P1). Returns ``act``; saves ``l1`` for backward."""

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        w1: torch.Tensor,
        trans_w1: bool,
        activation: str,
        out_dtype: torch.dtype,
        config: Float4QuantConfig,
        fuse_wgrad_accum_pattern: Union[None, str],
    ):
        assert activation in _SUPPORTED_ACTIVATIONS
        assert trans_w1, "MXFP4 dense GLU is NT-only"
        assert config.granularity == ScalingGranularity.MX_BLOCKWISE
        assert not config.use_preshuffle
        supported, reason = check_mxfp4_support()
        assert supported, reason

        fuse_w1_accum, w1_main_grad = _setup_fused_grad_accum(w1, fuse_wgrad_accum_pattern)
        ctx.w1_grad_shape = tuple(w1.shape)
        ctx.w1_grad_dtype = w1.dtype

        x_scaling_recipe = ScalingRecipe()
        x_t_scaling_recipe = ScalingRecipe(use_rht=True)
        x_row, x_row_scale, x_col, x_col_scale = quantize_fp4_with_trans(
            x,
            float4_e2m1fn_x2,
            ScalingGranularity.MX_BLOCKWISE,
            block_size=MXFP4_BLOCK_SIZE,
            scaling_recipe=x_scaling_recipe,
            scaling_recipe_for_trans=x_t_scaling_recipe,
        )
        w1_row, w1_row_scale, w1_col, w1_col_scale = _quantize_weight(w1, config)

        M = int(x.shape[0])
        probs = _probs_ones(M, x.device)
        l1, act = gemm_fp4_glu_bf16_impl(
            x_row,
            x_row_scale,
            w1_row,
            w1_row_scale,
            probs,
            out_dtype,
        )

        ctx.save_for_backward(x_col, x_col_scale, w1_col, w1_col_scale, l1, probs)
        ctx.out_dtype = out_dtype
        ctx.config = config
        ctx.fuse_w1_accum = fuse_w1_accum
        ctx.w1_main_grad = w1_main_grad
        return act

    @staticmethod
    def backward(ctx, dact):
        dact = _ensure_contiguous_grad_out(dact)
        x_col, x_col_scale, w1_col, w1_col_scale, l1, probs = ctx.saved_tensors
        sr = ctx.config.use_gradient_sr
        default_backend = BackendType.HIPBLASLT.value
        preshuffle = ctx.config.use_preshuffle

        dl1 = _dswiglu(dact, l1, probs)
        g_row, g_row_scale, g_col, g_col_scale = quantize_fp4_with_trans(
            dl1,
            float4_e2m1fn_x2,
            ctx.config.granularity,
            block_size=ctx.config.block_size,
            scaling_recipe=ScalingRecipe(use_sr=sr),
            scaling_recipe_for_trans=ScalingRecipe(use_sr=sr, use_rht=True),
        )

        grad_x = gemm_fp4_impl(
            g_row,
            g_row_scale,
            False,
            w1_col,
            w1_col_scale,
            True,
            ctx.out_dtype,
            False,
            granularity=ctx.config.granularity.value,
            default_backend=default_backend,
            preshuffled=preshuffle,
        )
        grad_w1 = _bgrad_gemm_fp4_impl_wrapper(
            g_col,
            g_col_scale,
            False,
            x_col,
            x_col_scale,
            True,
            ctx.out_dtype,
            False,
            granularity=ctx.config.granularity.value,
            default_backend=default_backend,
            preshuffled=preshuffle,
            inplace_add_to_out=ctx.fuse_w1_accum,
            out=ctx.w1_main_grad,
        )
        if ctx.fuse_w1_accum:
            grad_w1 = _get_dummy_wgrad(ctx.w1_grad_shape, ctx.w1_grad_dtype)
        return grad_x, grad_w1, None, None, None, None, None


class FP4DenseMlpQuantMXFunc(torch.autograd.Function):
    """P2/P3: MXFP4 dense GEMM + SwiGLU + dual-quant of ``act``, then fc2.

    ``act`` never hits BF16 HBM. ``l1`` stays BF16. fc2 and both wgrads stay on
    plain ``kernel_gemm_4w``. P3 replaces the dSwiGLU kernel + ``dL1`` quant with
    ``gemm_fp4_dglu_quant_impl`` when ``fuse_dglu`` is set.
    """

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        trans_w1: bool,
        trans_w2: bool,
        activation: str,
        out_dtype: torch.dtype,
        config: Float4QuantConfig,
        fuse_wgrad_accum_pattern: Union[None, str],
        fuse_dglu: bool,
        x_row_pre: Union[None, torch.Tensor],
        x_row_scale_pre: Union[None, torch.Tensor],
        x_col_pre: Union[None, torch.Tensor],
        x_col_scale_pre: Union[None, torch.Tensor],
    ):
        assert activation in _SUPPORTED_ACTIVATIONS
        assert trans_w1 and trans_w2, "MXFP4 dense MLP is NT-only"
        assert config.granularity == ScalingGranularity.MX_BLOCKWISE
        assert not config.use_preshuffle
        supported, reason = check_mxfp4_support()
        assert supported, reason

        fuse_w1_accum, w1_main_grad = _setup_fused_grad_accum(w1, fuse_wgrad_accum_pattern)
        fuse_w2_accum, w2_main_grad = _setup_fused_grad_accum(w2, fuse_wgrad_accum_pattern)
        ctx.w1_grad_shape = tuple(w1.shape)
        ctx.w2_grad_shape = tuple(w2.shape)
        ctx.w1_grad_dtype = w1.dtype
        ctx.w2_grad_dtype = w2.dtype

        x_scaling_recipe = ScalingRecipe()
        x_t_scaling_recipe = ScalingRecipe(use_rht=True)
        if x_row_pre is not None:
            x_row, x_row_scale, x_col, x_col_scale = (
                x_row_pre,
                x_row_scale_pre,
                x_col_pre,
                x_col_scale_pre,
            )
        else:
            x_row, x_row_scale, x_col, x_col_scale = quantize_fp4_with_trans(
                x,
                float4_e2m1fn_x2,
                ScalingGranularity.MX_BLOCKWISE,
                block_size=MXFP4_BLOCK_SIZE,
                scaling_recipe=x_scaling_recipe,
                scaling_recipe_for_trans=x_t_scaling_recipe,
            )
        w1_row, w1_row_scale, w1_col, w1_col_scale = _quantize_weight(w1, config)
        w2_row, w2_row_scale, w2_col, w2_col_scale = _quantize_weight(w2, config)

        M = int(x.shape[0])
        two_i = int(w1.shape[0])
        I = two_i // 2
        K = int(x.shape[1])
        assert dense_glu_epi_quant_supported(K, I, M, out_dtype), (
            f"dense glu-quant epilogue does not cover K={K} I={I} M={M} dtype={out_dtype}"
        )
        probs = _probs_ones(M, x.device)
        l1, act_row, act_row_scale, act_col, act_col_scale = gemm_fp4_glu_quant_impl(
            x_row,
            x_row_scale,
            w1_row,
            w1_row_scale,
            probs,
            out_dtype,
            False,
            False,
        )
        out = gemm_fp4_impl(
            act_row,
            act_row_scale,
            False,
            w2_row,
            w2_row_scale,
            True,
            out_dtype,
            False,
            granularity=config.granularity.value,
            default_backend=BackendType.HIPBLASLT.value,
            preshuffled=False,
        )

        ctx.save_for_backward(
            x_col,
            x_col_scale,
            w1_col,
            w1_col_scale,
            act_col,
            act_col_scale,
            w2_col,
            w2_col_scale,
            l1,
            probs,
        )
        ctx.out_dtype = out_dtype
        ctx.config = config
        ctx.fuse_w1_accum = fuse_w1_accum
        ctx.fuse_w2_accum = fuse_w2_accum
        ctx.w1_main_grad = w1_main_grad
        ctx.w2_main_grad = w2_main_grad
        ctx.fuse_dglu = bool(fuse_dglu)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        grad_out = _ensure_contiguous_grad_out(grad_out)
        (
            x_col,
            x_col_scale,
            w1_col,
            w1_col_scale,
            act_col,
            act_col_scale,
            w2_col,
            w2_col_scale,
            l1,
            probs,
        ) = ctx.saved_tensors
        sr = ctx.config.use_gradient_sr
        default_backend = BackendType.HIPBLASLT.value
        preshuffle = ctx.config.use_preshuffle

        g_row, g_row_scale, g_col, g_col_scale = quantize_fp4_with_trans(
            grad_out,
            float4_e2m1fn_x2,
            ctx.config.granularity,
            block_size=ctx.config.block_size,
            scaling_recipe=ScalingRecipe(use_sr=sr),
            scaling_recipe_for_trans=ScalingRecipe(use_sr=sr, use_rht=True),
        )
        grad_w2 = _bgrad_gemm_fp4_impl_wrapper(
            g_col,
            g_col_scale,
            False,
            act_col,
            act_col_scale,
            True,
            ctx.out_dtype,
            False,
            granularity=ctx.config.granularity.value,
            default_backend=default_backend,
            preshuffled=preshuffle,
            inplace_add_to_out=ctx.fuse_w2_accum,
            out=ctx.w2_main_grad,
        )
        if ctx.fuse_w2_accum:
            grad_w2 = _get_dummy_wgrad(ctx.w2_grad_shape, ctx.w2_grad_dtype)

        if ctx.fuse_dglu:
            gl_row, gl_row_scale, gl_col, gl_col_scale = gemm_fp4_dglu_quant_impl(
                g_row,
                g_row_scale,
                w2_col,
                w2_col_scale,
                l1,
                probs,
                ctx.out_dtype,
                sr,
                sr,
            )
        else:
            dact = gemm_fp4_impl(
                g_row,
                g_row_scale,
                False,
                w2_col,
                w2_col_scale,
                True,
                ctx.out_dtype,
                False,
                granularity=ctx.config.granularity.value,
                default_backend=default_backend,
                preshuffled=preshuffle,
            )
            dl1 = _dswiglu(dact, l1, probs)
            gl_row, gl_row_scale, gl_col, gl_col_scale = quantize_fp4_with_trans(
                dl1,
                float4_e2m1fn_x2,
                ctx.config.granularity,
                block_size=ctx.config.block_size,
                scaling_recipe=ScalingRecipe(use_sr=sr),
                scaling_recipe_for_trans=ScalingRecipe(use_sr=sr, use_rht=True),
            )
        grad_x = gemm_fp4_impl(
            gl_row,
            gl_row_scale,
            False,
            w1_col,
            w1_col_scale,
            True,
            ctx.out_dtype,
            False,
            granularity=ctx.config.granularity.value,
            default_backend=default_backend,
            preshuffled=preshuffle,
        )
        grad_w1 = _bgrad_gemm_fp4_impl_wrapper(
            gl_col,
            gl_col_scale,
            False,
            x_col,
            x_col_scale,
            True,
            ctx.out_dtype,
            False,
            granularity=ctx.config.granularity.value,
            default_backend=default_backend,
            preshuffled=preshuffle,
            inplace_add_to_out=ctx.fuse_w1_accum,
            out=ctx.w1_main_grad,
        )
        if ctx.fuse_w1_accum:
            grad_w1 = _get_dummy_wgrad(ctx.w1_grad_shape, ctx.w1_grad_dtype)
        return grad_x, grad_w1, grad_w2, None, None, None, None, None, None, None, None, None, None, None


def _fuse_act_quant(explicit: Union[bool, None]) -> bool:
    if explicit is not None:
        return bool(explicit)
    return os.environ.get("PRIMUS_TURBO_DENSE_MLP_FUSE_QUANT", "0") == "1"


def _fuse_dglu(explicit: Union[bool, None]) -> bool:
    if explicit is not None:
        return bool(explicit)
    return os.environ.get("PRIMUS_TURBO_DENSE_MLP_FUSE_DGLU", "0") == "1"


@torch._dynamo.disable(
    recursive=True,
    reason=(
        "Dense MXFP4 MLP attaches main_grad views and unsqueezes Parameters; "
        "Dynamo cannot recover those aliases."
    ),
)
def dense_mlp_fp4(
    x: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    *,
    trans_w1: bool = True,
    trans_w2: bool = True,
    out_dtype: Union[None, torch.dtype] = None,
    config: Union[Float4QuantConfig, None] = None,
    num_cu: int | None = None,
    fuse_wgrad_accum_pattern: Union[None, str] = None,
    activation: Union[None, str] = None,
    fuse_act_quant: Union[bool, None] = None,
    fuse_dglu: Union[bool, None] = None,
    x_prequant: Union[None, tuple] = None,
) -> torch.Tensor:
    """``fc2(silu(gate(x @ w1^T)) * up)`` for a single dense expert.

    Args:
        x: [M, K] activations.
        w1: [2I, K] fused gate||up weight (NT). Must be the Parameter so DDP
            overlap_grad_reduce hooks run.
        w2: [K_out, I] down-projection weight (NT).
        fuse_act_quant: P2 path. ``None`` reads ``PRIMUS_TURBO_DENSE_MLP_FUSE_QUANT``.
        fuse_dglu: P3 path. ``None`` reads ``PRIMUS_TURBO_DENSE_MLP_FUSE_DGLU``.
            Ignored unless ``fuse_act_quant`` is on.
    """
    del num_cu
    if activation is None:
        activation = "silu"
    if config is None:
        config = Float4QuantConfig()
    if out_dtype is None:
        out_dtype = x.dtype

    fused = _fuse_act_quant(fuse_act_quant)
    assert x_prequant is None or fused, (
        "x_prequant needs fuse_act_quant=True (or PRIMUS_TURBO_DENSE_MLP_FUSE_QUANT=1): the "
        "unfused path quantizes x itself and would ignore the pre-quantized pair."
    )
    if fused:
        x_row_pre = x_rs_pre = x_col_pre = x_cs_pre = None
        if x_prequant is not None:
            x_row_pre, x_rs_pre, x_col_pre, x_cs_pre = x_prequant
        return FP4DenseMlpQuantMXFunc.apply(
            x,
            w1,
            w2,
            trans_w1,
            trans_w2,
            activation,
            out_dtype,
            config,
            fuse_wgrad_accum_pattern,
            _fuse_dglu(fuse_dglu),
            x_row_pre,
            x_rs_pre,
            x_col_pre,
            x_cs_pre,
        )

    act = FP4DenseGluMXFunc.apply(
        x,
        w1,
        trans_w1,
        activation,
        out_dtype,
        config,
        fuse_wgrad_accum_pattern,
    )
    return gemm_fp4(
        act,
        w2,
        trans_a=False,
        trans_b=trans_w2,
        out_dtype=out_dtype,
        config=config,
        fuse_bgrad_accum_pattern=fuse_wgrad_accum_pattern,
    )
