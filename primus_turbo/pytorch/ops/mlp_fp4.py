###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Dense SwiGLU MLP in MXFP4, the single-expert counterpart of grouped_mlp_fp4.

    fwd  fc1 : l1, act_row, act_col = glu(x_row, w1_row)      contract K
         fc2 : out                  = gemm(act_row, w2_row)   contract I
    bwd  fc2 : dl1_row, dl1_col     = dglu(gradO_row, w2_col) contract K_out
         fc1 : dx                   = gemm(dl1_row, w1_col)   contract 2I

Both GLU epilogues quantize their own output, so neither ``act`` nor ``grad_l1``
is ever materialized in ``out_dtype``: they feed nothing but the quantizer, and a
round trip through HBM at bf16 would cost more than the GEMM saves. ``l1`` is the
one bf16 tensor kept, because the backward's dSwiGLU needs it.

Do not route this through the G=1 grouped GEMM: that path is graph-correct and a
TTT loss on Llama-3.1-8B.
"""

from typing import Optional, Union

import torch

from primus_turbo.flydsl.gemm.gemm_mxfp4_kernel import dense_glu_epi_quant_supported
from primus_turbo.flydsl.utils.gemm_epilogue_helper import SUPPORTED_ACTIVATIONS
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
from primus_turbo.pytorch.kernels.gemm.gemm_fp4_impl import (
    gemm_fp4_dglu_quant_impl,
    gemm_fp4_glu_quant_impl,
    gemm_fp4_impl,
)
from primus_turbo.pytorch.ops.gemm_fp4 import _bgrad_gemm_fp4_impl_wrapper
from primus_turbo.pytorch.ops.quantization import quantize_fp4_with_trans
from primus_turbo.pytorch.ops.utils import (
    _ensure_contiguous_grad_out,
    _get_dummy_wgrad,
    _setup_fused_grad_accum,
)

__all__ = ["mlp_fp4"]

_PROBS_ONES: dict = {}


def _probs_ones(M: int, device) -> torch.Tensor:
    """The routing weight the GLU epilogues always scale by; a dense MLP has none."""
    key = (M, str(device))
    t = _PROBS_ONES.get(key)
    if t is None:
        t = torch.ones(M, device=device, dtype=torch.float32)
        _PROBS_ONES[key] = t
    return t


def _quantize_weight(
    w: Union[torch.Tensor, QuantizedTensor],
    w_t: Optional[QuantizedTensor],
    config: Float4QuantConfig,
):
    """(row-wise, col-wise) MX operands for one weight.

    Neither carries the RHT -- that is the wgrad pair's alone, so a cached ``w_t``
    is taken as given; only its absence forces the col-wise pass.
    """
    recipe = ScalingRecipe(use_2d_block=True)
    if not isinstance(w, QuantizedTensor):
        return quantize_fp4_with_trans(
            w,
            float4_e2m1fn_x2,
            ScalingGranularity.MX_BLOCKWISE,
            block_size=MXFP4_BLOCK_SIZE,
            scaling_recipe=recipe,
            scaling_recipe_for_trans=recipe,
            scale_rounding_mode=config.scale_rounding_mode,
        )

    check_quantized_tensor(w, config, axis=-1, scaling_recipe=recipe)
    if w_t is None:
        w_t = QuantizedTensor.quantize(
            w.dequantize(),
            w.real_dtype,
            config.granularity,
            axis=-2,
            block_size=config.block_size,
            scaling_recipe=recipe,
            scale_rounding_mode=config.scale_rounding_mode,
        )
    else:
        assert isinstance(w_t, QuantizedTensor)
        check_quantized_tensor(w_t, config, axis=-2, scaling_recipe=recipe)
    return w.qdata, w.scale_inv, w_t.qdata, w_t.scale_inv


class FP4MLPMXFunc(torch.autograd.Function):
    """MXFP4 dense MLP autograd (MX_BLOCKWISE, NT-only, FlyDSL backend)."""

    @staticmethod
    def forward(
        ctx,
        x: Union[torch.Tensor, QuantizedTensor],
        w1: Union[torch.Tensor, QuantizedTensor],
        w2: Union[torch.Tensor, QuantizedTensor],
        x_t: Optional[QuantizedTensor],
        w1_t: Optional[QuantizedTensor],
        w2_t: Optional[QuantizedTensor],
        trans_w1: bool,
        trans_w2: bool,
        activation: str,
        clamp_limit: Union[None, float],
        out_dtype: torch.dtype,
        config: Float4QuantConfig,
        num_cu: Optional[int],
        fuse_wgrad_accum_pattern: Union[None, str],
    ):
        del num_cu
        assert activation in SUPPORTED_ACTIVATIONS, f"activation must be one of {SUPPORTED_ACTIVATIONS}"
        assert trans_w1 and trans_w2, "MXFP4 dense MLP is NT-only"
        assert config.granularity == ScalingGranularity.MX_BLOCKWISE
        assert not config.use_preshuffle, "the FlyDSL dense MLP takes raw (non-preshuffled) scales"
        supported, reason = check_mxfp4_support()
        assert supported, reason

        # Each weight has its own accumulation buffer, so these cannot be shared.
        fuse_w1_accum, w1_main_grad = _setup_fused_grad_accum(w1, fuse_wgrad_accum_pattern)
        fuse_w2_accum, w2_main_grad = _setup_fused_grad_accum(w2, fuse_wgrad_accum_pattern)
        ctx.w1_grad_shape = tuple(w1.shape)
        ctx.w2_grad_shape = tuple(w2.shape)
        ctx.w1_grad_dtype = w1.dtype
        ctx.w2_grad_dtype = w2.dtype

        # x's col-wise half is a wgrad operand, so it is the one that carries the RHT.
        x_scaling_recipe = ScalingRecipe()
        x_t_scaling_recipe = ScalingRecipe(use_rht=True)
        if not isinstance(x, QuantizedTensor):
            x_row, x_row_scale, x_col, x_col_scale = quantize_fp4_with_trans(
                x,
                float4_e2m1fn_x2,
                ScalingGranularity.MX_BLOCKWISE,
                block_size=MXFP4_BLOCK_SIZE,
                scaling_recipe=x_scaling_recipe,
                scaling_recipe_for_trans=x_t_scaling_recipe,
                scale_rounding_mode=config.scale_rounding_mode,
            )
        else:
            check_quantized_tensor(x, config, axis=-1, scaling_recipe=x_scaling_recipe)
            x_row, x_row_scale = x.qdata, x.scale_inv
            if x_t is None:
                x_t = QuantizedTensor.quantize(
                    x.dequantize(),
                    x.real_dtype,
                    config.granularity,
                    axis=-2,
                    block_size=config.block_size,
                    scaling_recipe=x_t_scaling_recipe,
                    scale_rounding_mode=config.scale_rounding_mode,
                )
            else:
                assert isinstance(x_t, QuantizedTensor)
                check_quantized_tensor(x_t, config, axis=-2, scaling_recipe=x_t_scaling_recipe)
            x_col, x_col_scale = x_t.qdata, x_t.scale_inv

        w1_row, w1_row_scale, w1_col, w1_col_scale = _quantize_weight(w1, w1_t, config)
        w2_row, w2_row_scale, w2_col, w2_col_scale = _quantize_weight(w2, w2_t, config)

        M = int(x.shape[0])
        I = int(w1.shape[0]) // 2
        K = int(x.shape[1])
        assert dense_glu_epi_quant_supported(K, I, M, out_dtype), (
            f"dense glu-quant epilogue does not cover K={K} I={I} M={M} dtype={out_dtype}"
        )
        probs = _probs_ones(M, x_row.device)
        l1, act_row, act_row_scale, act_col, act_col_scale = gemm_fp4_glu_quant_impl(
            x_row,
            x_row_scale,
            w1_row,
            w1_row_scale,
            probs,
            out_dtype,
            False,
            False,
            config.scale_rounding_mode,
            activation,
            clamp_limit,
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
        ctx.activation = activation
        ctx.clamp_limit = clamp_limit
        ctx.fuse_w1_accum = fuse_w1_accum
        ctx.fuse_w2_accum = fuse_w2_accum
        ctx.w1_main_grad = w1_main_grad
        ctx.w2_main_grad = w2_main_grad
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
            scale_rounding_mode=ctx.config.scale_rounding_mode,
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
            ctx.config.scale_rounding_mode,
            ctx.activation,
            ctx.clamp_limit,
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


@torch._dynamo.disable(
    recursive=True,
    reason=(
        "The MXFP4 MLP attaches main_grad views and unsqueezes Parameters; "
        "Dynamo cannot recover those aliases."
    ),
)
def mlp_fp4(
    x: Union[torch.Tensor, QuantizedTensor, QuantizedTensorPair],
    w1: Union[torch.Tensor, QuantizedTensor, QuantizedTensorPair],
    w2: Union[torch.Tensor, QuantizedTensor, QuantizedTensorPair],
    trans_w1: bool = True,
    trans_w2: bool = True,
    out_dtype: Union[None, torch.dtype] = None,
    config: Union[Float4QuantConfig, None] = None,
    num_cu: int | None = None,
    fuse_wgrad_accum_pattern: Union[None, str] = None,
    activation: Union[None, str] = None,
    clamp_limit: Union[None, float] = None,
) -> torch.Tensor:
    """Dense MLP in MXFP4: ``fc2(f(gate) * up)``.

    Args:
        x: [M, K] activations. May instead be a pre-quantized
            :class:`QuantizedTensor`, or a :class:`QuantizedTensorPair` whose
            ``data_t`` is the col-wise (rht=True) wgrad operand.
        w1: [2I, K] fc1 weight, gate rows first then up.
        w2: [K_out, I] fc2 weight. Both are NT, as MXFP4 has no other layout, so
            ``trans_w1`` and ``trans_w2`` must both be True; anything else is
            rejected rather than silently transposed. Both accept the same
            pre-quantized variants as ``x``, except that a weight's ``data_t`` is
            the col-wise dgrad operand and so carries no RHT.
        out_dtype: output dtype, inferred from the operands when None.
        config: FP4 quantization config. MX_BLOCKWISE only, and
            ``use_preshuffle`` must be False: the FlyDSL epilogues take raw scales.
        num_cu: accepted for signature parity with the grouped op; unused.
        fuse_wgrad_accum_pattern: ``"megatron"`` accumulates each wgrad into the
            weight's ``main_grad`` instead of returning it.
        activation: gate function, ``"silu"`` by default.
        clamp_limit: saturate the gate's inputs at +/- this before the activation,
            as the grouped op does; None leaves them unclamped.

    Returns:
        torch.Tensor: [M, K_out] output.
    """
    if config is None:
        config = Float4QuantConfig()
    if activation is None:
        activation = "silu"

    if isinstance(x, QuantizedTensorPair):
        x_data, x_data_t = x.data, x.data_t
    else:
        x_data, x_data_t = x, None

    if isinstance(w1, QuantizedTensorPair):
        w1_data, w1_data_t = w1.data, w1.data_t
    else:
        w1_data, w1_data_t = w1, None

    if isinstance(w2, QuantizedTensorPair):
        w2_data, w2_data_t = w2.data, w2.data_t
    else:
        w2_data, w2_data_t = w2, None

    if out_dtype is None:
        assert w1_data.dtype == w2_data.dtype, "w1 and w2 must have the same dtype"
        out_dtype = torch.promote_types(x_data.dtype, w1_data.dtype)

    if config.granularity != ScalingGranularity.MX_BLOCKWISE:
        raise ValueError(f"Unsupported FP4 ScalingGranularity: {config.granularity}")

    return FP4MLPMXFunc.apply(
        x_data,
        w1_data,
        w2_data,
        x_data_t,
        w1_data_t,
        w2_data_t,
        trans_w1,
        trans_w2,
        activation,
        clamp_limit,
        out_dtype,
        config,
        num_cu,
        fuse_wgrad_accum_pattern,
    )
