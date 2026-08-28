###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""MXFP4 grouped MoE MLP with SwiGLU fused into both GEMM epilogues.

Follows the recipe ``grouped_gemm_fp4`` establishes: the 16-point RHT rides the
wgrad operands only, and every GEMM is NT.

    fwd  fc1 : l1, act      = glu(x_row, w1_row)        contract K
         fc2 : out          = act_row @ w2_row^T        contract I
    bwd  fc2 : dl1, dprobs  = dglu(gradO_row, w2_col)   contract K_out
         fc1 : dx           = dl1_row @ w1_col^T        contract 2I
         wgrad              : the col-wise (rht=T) operands, contract M
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
    grouped_gemm_fp4_dglu_impl,
    grouped_gemm_fp4_glu_impl,
    grouped_gemm_fp4_impl,
    grouped_gemm_fp4_variable_k_accum_impl,
    grouped_gemm_fp4_variable_k_impl,
)
from primus_turbo.pytorch.kernels.grouped_gemm.grouped_gemm_utils import (
    group_offs_from_lens,
)
from primus_turbo.pytorch.ops.quantization import (
    grouped_quantize_fp4_with_trans,
    quantize_fp4_with_trans,
)
from primus_turbo.pytorch.ops.utils import (
    _ensure_contiguous_grad_out,
    _get_dummy_wgrad,
    _setup_fused_grad_accum,
)

__all__ = ["grouped_mlp_fp4"]


_SUPPORTED_ACTIVATIONS = ("silu",)


def _wgrad_grouped_gemm_fp4_impl_wrapper(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scales: torch.Tensor,
    b_scales: torch.Tensor,
    group_lens: torch.Tensor,
    group_offs: torch.Tensor,
    out_dtype: torch.dtype,
    num_cu: int | None,
    inplace_add_to_out: bool,
    out: Optional[torch.Tensor],
) -> torch.Tensor:
    """Variable-K wgrad, accumulating into ``out`` when asked to.

    Returns a dummy buffer in that case: the framework's own accumulation stands
    down, but Megatron needs a tensor rather than None for its backward hooks.
    """
    inputs = (a, b, a_scales, b_scales, group_lens, group_offs)
    options = dict(
        trans_a=False,
        trans_b=False,
        trans_c=False,
        out_dtype=out_dtype,
        granularity=ScalingGranularity.MX_BLOCKWISE.value,
        num_cu=num_cu,
        default_backend=BackendType.FLYDSL.value,
    )
    if not inplace_add_to_out:
        return grouped_gemm_fp4_variable_k_impl(*inputs, **options)

    assert out is not None, "out should not be None when inplace_add_to_out is True"
    grouped_gemm_fp4_variable_k_accum_impl(*inputs, out=out, **options)
    return _get_dummy_wgrad(out.shape, out_dtype)


def _quantize_weight(
    w: Union[torch.Tensor, QuantizedTensor],
    w_t: Optional[QuantizedTensor],
    config: Float4QuantConfig,
):
    """(row-wise, col-wise) MX operands for one 3D expert weight.

    Neither carries the RHT -- that is the wgrad pair's alone, so a cached
    ``w_t`` is taken as given; only its absence forces the col-wise pass.
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
        )

    assert not w._is_grouped_tensor, "an expert weight must not be a grouped tensor"
    check_quantized_tensor(w, config, axis=-1, scaling_recipe=recipe)
    if w_t is None:
        w_t = QuantizedTensor.quantize(
            w.dequantize(),
            w.real_dtype,
            config.granularity,
            axis=-2,
            block_size=config.block_size,
            scaling_recipe=recipe,
        )
    else:
        assert isinstance(w_t, QuantizedTensor)
    return w.qdata, w.scale_inv, w_t.qdata, w_t.scale_inv


class FP4GroupedMLPMXFunc(torch.autograd.Function):
    """MXFP4 grouped MoE MLP autograd (MX_BLOCKWISE, NT-only, FlyDSL backend)."""

    @staticmethod
    def forward(
        ctx,
        x: Union[torch.Tensor, QuantizedTensor],
        probs: torch.Tensor,
        w1: Union[torch.Tensor, QuantizedTensor],
        w2: Union[torch.Tensor, QuantizedTensor],
        x_t: Optional[QuantizedTensor],
        w1_t: Optional[QuantizedTensor],
        w2_t: Optional[QuantizedTensor],
        group_lens: torch.Tensor,  # [G,] int64
        group_offs: torch.Tensor,  # [G + 1,] int64
        trans_w1: bool,
        trans_w2: bool,
        activation: str,
        out_dtype: torch.dtype,
        config: Float4QuantConfig,
        num_cu: int | None,
        fuse_wgrad_accum_pattern: Union[None, str] = None,
    ):
        assert activation in _SUPPORTED_ACTIVATIONS, (
            f"Unsupported activation: {activation!r}, expected one of {_SUPPORTED_ACTIVATIONS}"
        )
        # MXFP4 has no non-NT layout, so the weights can only be given as
        # w1 [G, 2I, K] / w2 [G, K_out, I], which is what both flags being set means.
        assert trans_w1 and trans_w2, (
            "MXFP4 grouped MLP is NT-only: trans_w1 and trans_w2 must both be True, "
            f"got trans_w1={trans_w1}, trans_w2={trans_w2}."
        )
        assert config.granularity == ScalingGranularity.MX_BLOCKWISE
        assert not config.use_preshuffle, "MXFP4 grouped MLP does not use preshuffle."
        assert out_dtype in (torch.float16, torch.bfloat16)
        supported, reason = check_mxfp4_support()
        assert supported, reason

        assert x.ndim == 2 and w1.ndim == 3 and w2.ndim == 3
        K, two_i = int(x.shape[-1]), int(w1.shape[-2])
        assert two_i % 2 == 0, f"fc1 width must be even (gate||up), got {two_i}"
        I = two_i // 2
        assert int(w1.shape[-1]) == K, f"w1 must be [G, 2I, {K}], got {tuple(w1.shape)}"
        assert int(w2.shape[-1]) == I, f"w2 must be [G, K_out, {I}], got {tuple(w2.shape)}"
        for name, dim in (("K", K), ("2I", two_i), ("I", I), ("K_out", int(w2.shape[-2]))):
            assert dim % MXFP4_BLOCK_SIZE == 0, (
                f"{name} must be a multiple of {MXFP4_BLOCK_SIZE} (got {dim})."
            )

        # Each weight has its own accumulation buffer, so these cannot be shared.
        fuse_w1_accum, w1_main_grad = _setup_fused_grad_accum(w1, fuse_wgrad_accum_pattern)
        fuse_w2_accum, w2_main_grad = _setup_fused_grad_accum(w2, fuse_wgrad_accum_pattern)

        # x's col-wise half is a wgrad operand, so it is the one that carries the RHT.
        x_scaling_recipe = ScalingRecipe()
        x_t_scaling_recipe = ScalingRecipe(use_rht=True)
        if not isinstance(x, QuantizedTensor):
            x_row, x_row_scale, x_col, x_col_scale, _, offs_row, _, _ = grouped_quantize_fp4_with_trans(
                x,
                float4_e2m1fn_x2,
                ScalingGranularity.MX_BLOCKWISE,
                group_lens,
                group_offs,
                block_size=MXFP4_BLOCK_SIZE,
                scaling_recipe=x_scaling_recipe,
                scaling_recipe_for_trans=x_t_scaling_recipe,
            )
        else:
            check_quantized_tensor(x, config, axis=-1, scaling_recipe=x_scaling_recipe)
            x_row, x_row_scale = x.qdata, x.scale_inv
            offs_row = x.group_offs
            if x_t is None:
                x_t = QuantizedTensor.quantize(
                    x.dequantize(),
                    x.real_dtype,
                    config.granularity,
                    axis=-2,
                    block_size=config.block_size,
                    scaling_recipe=x_t_scaling_recipe,
                    group_lens=group_lens,
                )
            else:
                assert isinstance(x_t, QuantizedTensor)
            x_col, x_col_scale = x_t.qdata, x_t.scale_inv

        w1_row, w1_row_scale, w1_col, w1_col_scale = _quantize_weight(w1, w1_t, config)
        w2_row, w2_row_scale, w2_col, w2_col_scale = _quantize_weight(w2, w2_t, config)

        # The activation is quantized inside the GLU epilogue: it feeds nothing but
        # the quantizer, so materializing it in out_dtype would be an [M, I] round trip
        # through HBM. The row-wise pair is tight-M, so its group table is ``group_offs``.
        l1, act_row, act_row_scale, act_col, act_col_scale = grouped_gemm_fp4_glu_impl(
            x_row,
            w1_row,
            x_row_scale,
            w1_row_scale,
            group_lens,
            offs_row,
            trans_a=False,
            trans_b=True,
            out_dtype=out_dtype,
            num_cu=num_cu,
            probs=probs,
            config=config,
            out_row_scaling_recipe=ScalingRecipe(),
            out_col_scaling_recipe=ScalingRecipe(use_rht=True),
            activation=activation,
        )

        out = grouped_gemm_fp4_impl(
            act_row,
            w2_row,
            act_row_scale,
            w2_row_scale,
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

        ctx.save_for_backward(
            x_col,
            x_col_scale,
            act_col,
            act_col_scale,
            w1_col,
            w1_col_scale,
            w2_col,
            w2_col_scale,
            l1,
            probs,
            group_lens,
            group_offs,
        )
        ctx.activation = activation
        ctx.config = config
        ctx.out_dtype = out_dtype
        ctx.num_cu = num_cu
        ctx.fuse_w1_accum = fuse_w1_accum
        ctx.fuse_w2_accum = fuse_w2_accum
        # Off save_for_backward: the wgrad writes these in place, which would bump
        # the version counter saved tensors are checked against.
        ctx.w1_main_grad = w1_main_grad
        ctx.w2_main_grad = w2_main_grad
        return out

    @staticmethod
    def backward(ctx, grad_out):
        grad_out = _ensure_contiguous_grad_out(grad_out)
        (
            x_col,
            x_col_scale,
            act_col,
            act_col_scale,
            w1_col,
            w1_col_scale,
            w2_col,
            w2_col_scale,
            l1,
            probs,
            group_lens,
            group_offs,
        ) = ctx.saved_tensors

        sr = ctx.config.use_gradient_sr
        (
            go_row,
            go_row_scale,
            go_col,
            go_col_scale,
            _,
            go_offs_row,
            go_lens_col,
            go_offs_col,
        ) = grouped_quantize_fp4_with_trans(
            grad_out,
            float4_e2m1fn_x2,
            ScalingGranularity.MX_BLOCKWISE,
            group_lens,
            group_offs,
            block_size=ctx.config.block_size,
            scaling_recipe=ScalingRecipe(use_sr=sr),
            scaling_recipe_for_trans=ScalingRecipe(use_sr=sr, use_rht=True),
        )

        # grad_w2 = gradO_col(rht=T) @ act_col(rht=T)^T, contracting M.
        grad_w2 = _wgrad_grouped_gemm_fp4_impl_wrapper(
            go_col,
            act_col,
            go_col_scale,
            act_col_scale,
            go_lens_col,
            go_offs_col,
            ctx.out_dtype,
            ctx.num_cu,
            ctx.fuse_w2_accum,
            ctx.w2_main_grad,
        )

        # dgrad against w2_col, contracting K_out; the epilogue turns it into the
        # pre-activation gradient and quantizes that, so neither the gradient wrt act
        # nor the [M, 2I] one wrt l1 reaches HBM. The row-wise pair is tight-M, so its
        # group table is ``group_offs``; the col-wise pair shares gradO's, since the
        # two have the same M and group_lens.
        grad_probs, gl_row, gl_row_scale, gl_col, gl_col_scale = grouped_gemm_fp4_dglu_impl(
            go_row,
            w2_col,
            go_row_scale,
            w2_col_scale,
            group_lens,
            go_offs_row,
            trans_a=False,
            trans_b=True,
            out_dtype=ctx.out_dtype,
            num_cu=ctx.num_cu,
            probs=probs,
            intermediate=l1,
            config=ctx.config,
            out_row_scaling_recipe=ScalingRecipe(use_sr=sr),
            out_col_scaling_recipe=ScalingRecipe(use_sr=sr, use_rht=True),
            activation=ctx.activation,
        )
        gl_offs_row, gl_lens_col, gl_offs_col = group_offs, go_lens_col, go_offs_col

        # grad_x = grad_l1 @ w1_col^T, contracting 2I.
        grad_x = grouped_gemm_fp4_impl(
            gl_row,
            w1_col,
            gl_row_scale,
            w1_col_scale,
            group_lens,
            gl_offs_row,
            trans_a=False,
            trans_b=True,
            out_dtype=ctx.out_dtype,
            granularity=ScalingGranularity.MX_BLOCKWISE.value,
            num_cu=ctx.num_cu,
            default_backend=BackendType.FLYDSL.value,
            group_offs_out=group_offs,
        )

        # grad_w1 = grad_l1_col(rht=T) @ x_col(rht=T)^T, contracting M.
        grad_w1 = _wgrad_grouped_gemm_fp4_impl_wrapper(
            gl_col,
            x_col,
            gl_col_scale,
            x_col_scale,
            gl_lens_col,
            gl_offs_col,
            ctx.out_dtype,
            ctx.num_cu,
            ctx.fuse_w1_accum,
            ctx.w1_main_grad,
        )

        return (
            grad_x,  # x
            grad_probs,  # probs
            grad_w1,  # w1
            grad_w2,  # w2
            None,  # x_t
            None,  # w1_t
            None,  # w2_t
            None,  # group_lens
            None,  # group_offs
            None,  # trans_w1
            None,  # trans_w2
            None,  # activation
            None,  # out_dtype
            None,  # config
            None,  # num_cu
            None,  # fuse_wgrad_accum_pattern
        )


@torch._dynamo.disable(
    recursive=True,
    reason=(
        "Grouped MXFP4 MLP composes the grouped quantizer per direction and reads "
        "packed FP4 / E8M0 inner tensors inside its autograd.Function.forward; "
        "Dynamo cannot recover Python sources for those graph-internal tensors."
    ),
)
def grouped_mlp_fp4(
    x: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    group_lens: torch.Tensor,
    probs: torch.Tensor,
    group_offs: torch.Tensor | None = None,
    trans_w1: bool = False,
    trans_w2: bool = False,
    out_dtype: Union[None, torch.dtype] = None,
    config: Union[Float4QuantConfig, None] = None,
    num_cu: int | None = None,
    fuse_wgrad_accum_pattern: Union[None, str] = None,
    activation: Union[None, str] = None,
) -> torch.Tensor:
    """MoE expert MLP in MXFP4: ``fc2(silu(gate) * up * probs)`` over ``group_lens``.

    Args:
        x: [total_m, K] activations, grouped along M. May instead be a
            pre-quantized grouped :class:`QuantizedTensor`, or a
            :class:`QuantizedTensorPair` whose ``data_t`` is the col-wise
            (rht=True) wgrad operand.
        w1: [G, 2I, K] fc1 weight, gate rows first then up.
        w2: [G, K_out, I] fc2 weight. Both are NT, as MXFP4 has no other layout,
            so ``trans_w1`` and ``trans_w2`` must both be True; anything else is
            rejected rather than silently transposed. Both accept the same
            pre-quantized variants as ``x``, except that a weight's ``data_t`` is
            the col-wise dgrad operand and so carries no RHT.
        probs: [total_m] float32 routing probabilities. Required -- the fused
            epilogues always scale by it and reduce its gradient.

    Returns:
        [total_m, K_out] in ``out_dtype``.
    """
    if config is None:
        config = Float4QuantConfig()

    assert activation in _SUPPORTED_ACTIVATIONS, (
        f"Unsupported activation: {activation!r}, expected one of {_SUPPORTED_ACTIVATIONS}"
    )
    assert probs is not None, "probs is required: the fused GLU epilogues always scale by it"

    if group_offs is None:
        group_offs = group_offs_from_lens(group_lens)

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

    return FP4GroupedMLPMXFunc.apply(
        x_data,
        probs,
        w1_data,
        w2_data,
        x_data_t,
        w1_data_t,
        w2_data_t,
        group_lens,
        group_offs,
        trans_w1,
        trans_w2,
        activation,
        out_dtype,
        config,
        num_cu,
        fuse_wgrad_accum_pattern,
    )
