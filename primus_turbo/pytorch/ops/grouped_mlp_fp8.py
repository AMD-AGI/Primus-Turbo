###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from typing import Optional, Union

import torch

from primus_turbo.pytorch.core.backend import BackendType
from primus_turbo.pytorch.core.low_precision import (
    MXFP8_BLOCK_SIZE,
    Float8QuantConfig,
    ScalingGranularity,
    ScalingRecipe,
    check_mxfp8_support,
)
from primus_turbo.pytorch.core.quantized_tensor import (
    QuantizedTensor,
    QuantizedTensorPair,
    check_quantized_tensor,
)
from primus_turbo.pytorch.core.utils import is_gfx950
from primus_turbo.pytorch.kernels.grouped_gemm.grouped_gemm_fp8_impl import (
    grouped_gemm_fp8_dglu_impl,
    grouped_gemm_fp8_glu_impl,
    grouped_gemm_fp8_impl,
    grouped_gemm_fp8_mx_dglu_impl,
    grouped_gemm_fp8_mx_glu_impl,
    grouped_gemm_fp8_variable_k_accum_impl,
    grouped_gemm_fp8_variable_k_impl,
)
from primus_turbo.pytorch.kernels.grouped_gemm.grouped_gemm_utils import (
    group_offs_from_lens,
)
from primus_turbo.pytorch.ops.quantization import (
    grouped_quantize_fp8_with_trans,
    quantize_fp8_with_trans,
)
from primus_turbo.pytorch.ops.utils import (
    _ensure_contiguous_grad_out,
    _get_dummy_wgrad,
    _get_fp8_dtype,
    _setup_fused_grad_accum,
)

__all__ = [
    "grouped_mlp_fp8",
]


_SUPPORTED_ACTIVATIONS = ("silu", "gelu")

_SUPPORTED_CLAMPABLE_ACTIVATIONS = ("silu",)


def _check_activation(activation: str, clamp_limit: Union[None, float]) -> Union[None, float]:
    assert activation in _SUPPORTED_ACTIVATIONS, (
        f"Unsupported activation: {activation!r}, expected one of {_SUPPORTED_ACTIVATIONS}"
    )
    if clamp_limit is None:
        return None
    assert activation in _SUPPORTED_CLAMPABLE_ACTIVATIONS, (
        f"clamp_limit is only supported for activation in {_SUPPORTED_CLAMPABLE_ACTIVATIONS}, got {activation!r}"
    )
    clamp_limit = float(clamp_limit)
    assert clamp_limit > 0.0, f"clamp_limit must be positive, got {clamp_limit}"
    return clamp_limit


# Pad the fp8 grouped-MLP contraction/feature dims to 128. gpt-oss-20b runs
# H = I = 2880, and 2880 % 128 == 64: an unpadded fp8 GEMM splits every
# cache line across two L1->L2 requests (+50% traffic) for identical MFMA math.
# Padding H (fc1/fc2 K and the fc2-output N) and the fc2 contraction I up to
# the next multiple recovers the aligned access; real-shape recovery keeps the
# stored tensors tight, so this is copy-free on the padded quantiser buffers.
_FP8_PAD_ALIGN = 128


def _default_gemm_backend() -> int:
    return BackendType.FLYDSL.value if is_gfx950() else BackendType.TRITON.value


def _grouped_gemm_fp8_variable_k_impl_wrapper(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scales: torch.Tensor,
    b_scales: torch.Tensor,
    group_lens: torch.Tensor,
    group_offs: torch.Tensor,
    trans_a: bool,
    trans_b: bool,
    trans_c: bool,
    out_dtype: torch.dtype,
    granularity: int,
    num_cu: Optional[int],
    default_backend: int,
    inplace_add_to_out: bool = False,
    out: Optional[torch.Tensor] = None,
    m_real: Optional[int] = None,
    n_real: Optional[int] = None,
) -> Optional[torch.Tensor]:
    """Run the variable-K wgrad GEMM, accumulating into ``out`` when asked to.

    Returns the weight gradient for autograd, or a dummy buffer when the wgrad went
    straight into ``out``: forward already flagged the weight, so the training
    framework's own accumulation step stands down. Megatron still expects a tensor
    rather than None there, so its backward hooks stay on the main thread; the
    contents are never read. It is handed back in the weight's own dtype, since a
    mismatch would make autograd allocate and cast a full-size copy.
    """
    inputs = (a, b, a_scales, b_scales, group_lens, group_offs)
    options = dict(
        trans_a=trans_a,
        trans_b=trans_b,
        trans_c=trans_c,
        out_dtype=out_dtype,
        granularity=granularity,
        num_cu=num_cu,
        default_backend=default_backend,
        m_real=m_real,
        n_real=n_real,
    )

    if not inplace_add_to_out:
        return grouped_gemm_fp8_variable_k_impl(*inputs, **options)

    assert out is not None, "out should not be None when inplace_add_to_out is True"
    grouped_gemm_fp8_variable_k_accum_impl(*inputs, out=out, **options)

    return _get_dummy_wgrad(out.shape, out_dtype)


class FP8GroupedMLPTensorFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: Union[torch.Tensor, QuantizedTensor],
        probs: torch.Tensor,
        w1: Union[torch.Tensor, QuantizedTensor],
        w2: Union[torch.Tensor, QuantizedTensor],
        x_t: Optional[QuantizedTensor],  # not used
        w1_t: Optional[QuantizedTensor],  # not used
        w2_t: Optional[QuantizedTensor],  # not used
        group_lens: torch.Tensor,  # [B,] int64
        group_offs: torch.Tensor,  # [B + 1,] int64
        trans_w1: bool,
        trans_w2: bool,
        activation: str,
        clamp_limit: Union[None, float],
        out_dtype: torch.dtype,
        config: Float8QuantConfig,
        num_cu: int | None,
        fuse_wgrad_accum_pattern: Union[None, str] = None,
    ):
        clamp_limit = _check_activation(activation, clamp_limit)

        # Each weight carries its own accumulation buffer, so the two wgrads
        # cannot share one: resolve them separately while the parameter objects
        # are still in hand.
        fuse_w1_accum, w1_main_grad = _setup_fused_grad_accum(w1, fuse_wgrad_accum_pattern)
        fuse_w2_accum, w2_main_grad = _setup_fused_grad_accum(w2, fuse_wgrad_accum_pattern)

        assert config.granularity == ScalingGranularity.TENSORWISE

        # Also the dtype the fc1 activation is quantised to before fc2.
        x_dtype = _get_fp8_dtype(config.format, True)

        if isinstance(x, QuantizedTensor):
            assert x._is_grouped_tensor, "A QuantizedTensor input must be a grouped tensor"
            check_quantized_tensor(x, config)
            quantized_x = x
            group_offs = x.group_offs
        else:
            quantized_x = QuantizedTensor.quantize(
                x,
                x_dtype,
                config.granularity,
                axis=-1,
                block_size=config.block_size,
                group_lens=group_lens,
                pad_align_last=_FP8_PAD_ALIGN,
            )

        if isinstance(w1, QuantizedTensor):
            assert not w1._is_grouped_tensor, "w1 QuantizedTensor input must not be a grouped tensor"
            check_quantized_tensor(w1, config)
            quantized_w1 = w1
        else:
            w1_dtype = _get_fp8_dtype(config.format, True)
            quantized_w1 = QuantizedTensor.quantize(
                w1,
                w1_dtype,
                config.granularity,
                axis=-1,
                block_size=config.block_size,
                pad_align_last=_FP8_PAD_ALIGN,
            )

        if isinstance(w2, QuantizedTensor):
            assert not w2._is_grouped_tensor, "w2 QuantizedTensor input must not be a grouped tensor"
            check_quantized_tensor(w2, config)
            quantized_w2 = w2
        else:
            w2_dtype = _get_fp8_dtype(config.format, True)
            quantized_w2 = QuantizedTensor.quantize(
                w2,
                w2_dtype,
                config.granularity,
                axis=-1,
                block_size=config.block_size,
                pad_align_penultimate=_FP8_PAD_ALIGN,
                pad_align_last=_FP8_PAD_ALIGN,
            )

        # The activation is quantised inside the GLU op: it feeds nothing but the
        # quantiser, so it stages there rather than here, where a later kernel can fold
        # the conversion into the epilogue and drop the [M, I] round trip for good.
        fc1_out, act_fp8, act_scale_inv = grouped_gemm_fp8_glu_impl(
            quantized_x.qdata,
            quantized_w1.qdata,
            quantized_x.scale_inv,
            quantized_w1.scale_inv,
            group_lens,
            group_offs,
            trans_a=False,
            trans_b=trans_w1,
            out_dtype=out_dtype,
            num_cu=num_cu,
            probs=probs,
            config=config,
            out_quant_dtype=x_dtype,
            out_row_scaling_recipe=ScalingRecipe(),
            out_col_scaling_recipe=ScalingRecipe(),
            activation=activation,
            clamp_limit=clamp_limit,
            # Pad the fused activation's I -> Ip so fc2's contraction matches w2's
            # padded I; copy-free (the quantiser writes the padded buffer directly).
            k_align=_FP8_PAD_ALIGN,
        )

        # padN+padK recovers the tight fc2 output (N = H) from the padded w2, and
        # routes fc2 through FLYDSL so the padded contraction is actually consumed.
        h_real = quantized_x.shape[-1]
        # Tight I (fc2 contraction), for dglu/grad_w2 to recover from padded Ip.
        i_real = quantized_w2.shape[-1]
        fc2_n_pitch = quantized_w2.qdata.shape[-2] if trans_w2 else quantized_w2.qdata.shape[-1]
        fc2_out = grouped_gemm_fp8_impl(
            act_fp8,
            quantized_w2.qdata,
            act_scale_inv,
            quantized_w2.scale_inv,
            group_lens,
            group_offs,
            trans_a=False,
            trans_b=trans_w2,
            out_dtype=out_dtype,
            granularity=config.granularity.value,
            num_cu=num_cu,
            default_backend=_default_gemm_backend(),
            maybe_pre_sync=True,
            n_real=h_real if h_real != fc2_n_pitch else None,
        )

        ctx.save_for_backward(
            quantized_x.qdata,
            act_fp8,
            quantized_w1.qdata,
            quantized_w2.qdata,
            quantized_x.scale_inv,
            act_scale_inv,
            quantized_w1.scale_inv,
            quantized_w2.scale_inv,
            fc1_out,
            probs,
            group_lens,
            group_offs,
        )
        ctx.trans_w1 = trans_w1
        ctx.trans_w2 = trans_w2
        ctx.activation = activation
        ctx.clamp_limit = clamp_limit
        ctx.config = config
        ctx.out_dtype = out_dtype
        ctx.num_cu = num_cu
        ctx.h_real = h_real
        ctx.i_real = i_real
        ctx.fuse_w1_accum = fuse_w1_accum
        ctx.fuse_w2_accum = fuse_w2_accum
        # Kept off save_for_backward on purpose: the wgrad GEMM writes into these
        # buffers in place, which would bump the version counter that saved tensors
        # are checked against.
        ctx.w1_main_grad = w1_main_grad
        ctx.w2_main_grad = w2_main_grad

        return fc2_out

    @staticmethod
    def backward(ctx, grad_out):
        grad_out = _ensure_contiguous_grad_out(grad_out)
        (
            x_fp8,
            act_fp8,
            w1_fp8,
            w2_fp8,
            x_scale_inv,
            act_scale_inv,
            w1_scale_inv,
            w2_scale_inv,
            fc1_out,
            probs,
            group_lens,
            group_offs,
        ) = ctx.saved_tensors

        grad_out_dtype = _get_fp8_dtype(ctx.config.format, False)
        quantized_grad_out = QuantizedTensor.quantize(
            grad_out,
            grad_out_dtype,
            ctx.config.granularity,
            axis=-1,
            block_size=ctx.config.block_size,
            group_lens=group_lens,
            pad_align_last=_FP8_PAD_ALIGN,
        )
        # H is padded on the fc2-output side (grad_out last, w2 penult, x/w1
        # contraction); real recovers tight H on every GEMM whose output or wgrad
        # feature is H. I stays tight in M1, so its real is left None.
        h_real = ctx.h_real
        i_real = ctx.i_real
        default_backend = _default_gemm_backend()
        go_pitch = quantized_grad_out.qdata.shape[-1]

        # grad_w2 = act^T @ grad_out, both operands already in hand.
        grad_w2 = _grouped_gemm_fp8_variable_k_impl_wrapper(
            act_fp8,
            quantized_grad_out.qdata,
            act_scale_inv,
            quantized_grad_out.scale_inv,
            group_lens,
            group_offs,
            trans_a=True,
            trans_b=False,
            trans_c=ctx.trans_w2,
            out_dtype=ctx.out_dtype,
            granularity=ctx.config.granularity.value,
            num_cu=ctx.num_cu,
            default_backend=default_backend,
            inplace_add_to_out=ctx.fuse_w2_accum,
            out=ctx.w2_main_grad,
            m_real=h_real if h_real != go_pitch else None,
            # a = act, padded on its I axis; n_real recovers the tight I (grad_w2 N).
            n_real=i_real if i_real != act_fp8.shape[-1] else None,
        )

        # fc2 dgrad (grad_out @ w2^T) with the activation gradient fused into its
        # epilogue, giving the pre-activation gradient directly, quantised. The probs
        # scaling and grad_probs both ride along inside that epilogue.
        grad_probs, grad_fc1_out_fp8, grad_fc1_out_scale_inv = grouped_gemm_fp8_dglu_impl(
            quantized_grad_out.qdata,
            w2_fp8,
            quantized_grad_out.scale_inv,
            w2_scale_inv,
            group_lens,
            group_offs,
            trans_a=False,
            trans_b=not ctx.trans_w2,
            out_dtype=ctx.out_dtype,
            num_cu=ctx.num_cu,
            probs=probs,
            intermediate=fc1_out,
            config=ctx.config,
            out_quant_dtype=grad_out_dtype,
            out_row_scaling_recipe=ScalingRecipe(),
            out_col_scaling_recipe=ScalingRecipe(),
            activation=ctx.activation,
            clamp_limit=ctx.clamp_limit,
            # w2 (b) is padded on its I axis to w2_fp8.shape[-1]; i_real recovers the
            # tight I so grad_fc1_out stays [M, 2I] and the padded Ip rides as n_stride.
            i_real=i_real if i_real != w2_fp8.shape[-1] else None,
        )

        # grad_x = grad_fc1_out @ w1^T; output feature N = H (padded on w1), recovered.
        gx_trans_b = not ctx.trans_w1
        gx_n_pitch = w1_fp8.shape[-2] if gx_trans_b else w1_fp8.shape[-1]
        grad_x = grouped_gemm_fp8_impl(
            grad_fc1_out_fp8,
            w1_fp8,
            grad_fc1_out_scale_inv,
            w1_scale_inv,
            group_lens,
            group_offs,
            trans_a=False,
            trans_b=gx_trans_b,
            out_dtype=ctx.out_dtype,
            granularity=ctx.config.granularity.value,
            num_cu=ctx.num_cu,
            default_backend=default_backend,
            n_real=h_real if h_real != gx_n_pitch else None,
        )

        # grad_w1 = x^T @ grad_fc1_out
        grad_w1 = _grouped_gemm_fp8_variable_k_impl_wrapper(
            x_fp8,
            grad_fc1_out_fp8,
            x_scale_inv,
            grad_fc1_out_scale_inv,
            group_lens,
            group_offs,
            trans_a=True,
            trans_b=False,
            trans_c=ctx.trans_w1,
            out_dtype=ctx.out_dtype,
            granularity=ctx.config.granularity.value,
            num_cu=ctx.num_cu,
            default_backend=default_backend,
            inplace_add_to_out=ctx.fuse_w1_accum,
            out=ctx.w1_main_grad,
            m_real=None,
            n_real=h_real if h_real != x_fp8.shape[-1] else None,
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
            None,  # clamp_limit
            None,  # out_dtype
            None,  # config
            None,  # num_cu
            None,  # fuse_wgrad_accum_pattern
        )


def _quantize_weight(
    w: Union[torch.Tensor, QuantizedTensor],
    w_t: Optional[QuantizedTensor],
    config: Float8QuantConfig,
):
    """(row-wise, col-wise) MXFP8 operands for one 3D expert weight.

    Both halves take the per-32x32 tile scale, which is weight-only; sharing one amax
    across the tile keeps the forward and dgrad operands consistent. A cached ``w_t``
    is taken as given; only its absence forces the col-wise pass.
    """
    recipe = ScalingRecipe(use_2d_block=True)
    w_dtype = _get_fp8_dtype(config.format, True)
    if not isinstance(w, QuantizedTensor):
        return quantize_fp8_with_trans(
            w,
            w_dtype,
            ScalingGranularity.MX_BLOCKWISE,
            block_size=MXFP8_BLOCK_SIZE,
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


class FP8GroupedMLPMXFunc(torch.autograd.Function):
    """MXFP8 grouped MoE MLP autograd (MX_BLOCKWISE, NT-only, FlyDSL backend)."""

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
        clamp_limit: Union[None, float],
        out_dtype: torch.dtype,
        config: Float8QuantConfig,
        num_cu: int | None,
        fuse_wgrad_accum_pattern: Union[None, str] = None,
    ):
        clamp_limit = _check_activation(activation, clamp_limit)
        # Both flags set is what w1 [G, 2I, K] / w2 [G, K_out, I] means.
        assert trans_w1 and trans_w2, (
            "MXFP8 grouped MLP is NT-only: trans_w1 and trans_w2 must both be True, "
            f"got trans_w1={trans_w1}, trans_w2={trans_w2}."
        )
        assert config.mxfp8_scaling(), (
            f"MXFP8 grouped MLP needs MX_BLOCKWISE granularity with E8M0 scales, got {config}"
        )
        assert out_dtype == torch.bfloat16, (
            f"the fused MXFP8 GLU epilogues are bfloat16-only, got {out_dtype}"
        )
        supported, reason = check_mxfp8_support()
        assert supported, reason

        assert x.ndim == 2 and w1.ndim == 3 and w2.ndim == 3
        K, two_i = int(x.shape[-1]), int(w1.shape[-2])
        assert two_i % 2 == 0, f"fc1 width must be even (gate||up), got {two_i}"
        I = two_i // 2
        assert int(w1.shape[-1]) == K, f"w1 must be [G, 2I, {K}], got {tuple(w1.shape)}"
        assert int(w2.shape[-1]) == I, f"w2 must be [G, K_out, {I}], got {tuple(w2.shape)}"
        for name, dim in (("K", K), ("2I", two_i), ("I", I), ("K_out", int(w2.shape[-2]))):
            assert dim % MXFP8_BLOCK_SIZE == 0, (
                f"{name} must be a multiple of {MXFP8_BLOCK_SIZE} (got {dim})."
            )

        # Each weight has its own accumulation buffer, so these cannot be shared.
        fuse_w1_accum, w1_main_grad = _setup_fused_grad_accum(w1, fuse_wgrad_accum_pattern)
        fuse_w2_accum, w2_main_grad = _setup_fused_grad_accum(w2, fuse_wgrad_accum_pattern)

        M = int(probs.shape[0])
        x_dtype = _get_fp8_dtype(config.format, True)
        if not isinstance(x, QuantizedTensor):
            x_row, x_row_scale, x_col, x_col_scale, _, offs_row, _, _ = grouped_quantize_fp8_with_trans(
                x,
                x_dtype,
                ScalingGranularity.MX_BLOCKWISE,
                group_lens,
                group_offs,
                block_size=MXFP8_BLOCK_SIZE,
            )
        else:
            assert x._is_grouped_tensor, "a QuantizedTensor input must be a grouped tensor"
            check_quantized_tensor(x, config, axis=-1)
            x_row, x_row_scale = x.qdata, x.scale_inv
            offs_row = x.group_offs
            if x_t is None:
                x_t = QuantizedTensor.quantize(
                    x.dequantize(),
                    x.real_dtype,
                    config.granularity,
                    axis=-2,
                    block_size=config.block_size,
                    group_lens=group_lens,
                )
            else:
                assert isinstance(x_t, QuantizedTensor)
            x_col, x_col_scale = x_t.qdata, x_t.scale_inv

        w1_row, w1_row_scale, w1_col, w1_col_scale = _quantize_weight(w1, w1_t, config)
        w2_row, w2_row_scale, w2_col, w2_col_scale = _quantize_weight(w2, w2_t, config)

        # The activation is quantised inside the epilogue: it feeds nothing but the
        # quantiser, so staging it in out_dtype would be an [M, I] round trip through HBM.
        l1, act_row, act_row_scale, act_col, act_col_scale = grouped_gemm_fp8_mx_glu_impl(
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
            out_col_scaling_recipe=ScalingRecipe(),
            activation=activation,
            clamp_limit=clamp_limit,
        )

        # act_row shares x_row's padded rows, so it reads under ``offs_row`` and writes
        # tight; the output is over-allocated to those padded rows and sliced back.
        out = grouped_gemm_fp8_impl(
            act_row,
            w2_row,
            act_row_scale,
            w2_row_scale,
            group_lens,
            offs_row,
            trans_a=False,
            trans_b=True,
            out_dtype=out_dtype,
            granularity=config.granularity.value,
            num_cu=num_cu,
            default_backend=BackendType.FLYDSL.value,
            group_offs_out=group_offs,
        )[:M]

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
        ctx.clamp_limit = clamp_limit
        ctx.config = config
        ctx.out_dtype = out_dtype
        ctx.num_cu = num_cu
        ctx.fuse_w1_accum = fuse_w1_accum
        ctx.fuse_w2_accum = fuse_w2_accum
        # Off save_for_backward: the wgrad writes these in place, which would bump the
        # version counter saved tensors are checked against.
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

        M = int(probs.shape[0])
        grad_out_dtype = _get_fp8_dtype(ctx.config.format, False)
        (
            go_row,
            go_row_scale,
            go_col,
            go_col_scale,
            _,
            go_offs_row,
            go_lens_col,
            go_offs_col,
        ) = grouped_quantize_fp8_with_trans(
            grad_out,
            grad_out_dtype,
            ScalingGranularity.MX_BLOCKWISE,
            group_lens,
            group_offs,
            block_size=ctx.config.block_size,
        )
        default_backend = BackendType.FLYDSL.value

        # grad_w2 = gradO_col @ act_col^T, contracting M.
        grad_w2 = _grouped_gemm_fp8_variable_k_impl_wrapper(
            go_col,
            act_col,
            go_col_scale,
            act_col_scale,
            go_lens_col,
            go_offs_col,
            trans_a=False,
            trans_b=False,
            trans_c=False,
            out_dtype=ctx.out_dtype,
            granularity=ctx.config.granularity.value,
            num_cu=ctx.num_cu,
            default_backend=default_backend,
            inplace_add_to_out=ctx.fuse_w2_accum,
            out=ctx.w2_main_grad,
        )

        # dgrad against w2_col, contracting K_out; the epilogue turns it into the
        # pre-activation gradient and quantises that, so neither reaches HBM.
        grad_probs, gl_row, gl_row_scale, gl_col, gl_col_scale = grouped_gemm_fp8_mx_dglu_impl(
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
            out_row_scaling_recipe=ScalingRecipe(),
            out_col_scaling_recipe=ScalingRecipe(),
            activation=ctx.activation,
            clamp_limit=ctx.clamp_limit,
        )
        # grad_x = grad_l1 @ w1_col^T, contracting 2I. grad_l1 reuses gradO's tables --
        # the two have the same M and group_lens.
        grad_x = grouped_gemm_fp8_impl(
            gl_row,
            w1_col,
            gl_row_scale,
            w1_col_scale,
            group_lens,
            go_offs_row,
            trans_a=False,
            trans_b=True,
            out_dtype=ctx.out_dtype,
            granularity=ctx.config.granularity.value,
            num_cu=ctx.num_cu,
            default_backend=default_backend,
            group_offs_out=group_offs,
        )[:M]

        # grad_w1 = grad_l1_col @ x_col^T, contracting M.
        grad_w1 = _grouped_gemm_fp8_variable_k_impl_wrapper(
            gl_col,
            x_col,
            gl_col_scale,
            x_col_scale,
            go_lens_col,
            go_offs_col,
            trans_a=False,
            trans_b=False,
            trans_c=False,
            out_dtype=ctx.out_dtype,
            granularity=ctx.config.granularity.value,
            num_cu=ctx.num_cu,
            default_backend=default_backend,
            inplace_add_to_out=ctx.fuse_w1_accum,
            out=ctx.w1_main_grad,
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
            None,  # clamp_limit
            None,  # out_dtype
            None,  # config
            None,  # num_cu
            None,  # fuse_wgrad_accum_pattern
        )


@torch._dynamo.disable(
    recursive=True,
    reason=(
        "Grouped FP8 MLP constructs (Grouped)QuantizedTensor wrapper subclasses "
        "inside its autograd.Function.forward and reads their inner tensors "
        "(x / w1 / w2 / scale_inv / group_lens / group_offs). Dynamo cannot recover Python "
        "sources for those graph-internal inner tensors, tripping gb0116 "
        "('SourcelessBuilder.create cannot wrap FakeTensor'). "
    ),
)
def grouped_mlp_fp8(
    x: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    group_lens: torch.Tensor,
    probs: torch.Tensor,
    group_offs: torch.Tensor | None = None,
    trans_w1: bool = False,
    trans_w2: bool = False,
    out_dtype: Union[None, torch.dtype] = None,
    config: Union[Float8QuantConfig, None] = None,
    num_cu: int | None = None,
    fuse_wgrad_accum_pattern: Union[None, str] = None,
    activation: Union[None, str] = None,
    clamp_limit: Union[None, float] = None,
) -> torch.Tensor:
    """Grouped FP8 MLP: ``fc2(f(gate) * up * probs)`` with fc1's GLU fused into its epilogue.

    Args:
        activation: the GLU gate, one of ``("silu", "gelu")``. "gelu" is the tanh
            approximation, i.e. ``F.gelu(approximate="tanh")``.
        clamp_limit: DeepSeek-V4's pre-multiplication clamp bound ``L``, or None for no
            clamp. With it the activation is ``silu(min(gate, L)) * clamp(up, -L, L)``,
            whose backward is straight-through. Supported for "silu" only.
    """
    if config is None:
        config = Float8QuantConfig()

    clamp_limit = _check_activation(activation, clamp_limit)

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

    if config.granularity == ScalingGranularity.TENSORWISE:
        # TENSORWISE has a single scalar scale (no col-wise trans cache needed);
        # the inner ``data_t`` is ignored if provided.
        return FP8GroupedMLPTensorFunc.apply(
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
            clamp_limit,
            out_dtype,
            config,
            num_cu,
            fuse_wgrad_accum_pattern,
        )
    elif config.granularity == ScalingGranularity.MX_BLOCKWISE:
        return FP8GroupedMLPMXFunc.apply(
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
            clamp_limit,
            out_dtype,
            config,
            num_cu,
            fuse_wgrad_accum_pattern,
        )
    else:
        raise ValueError(f"Unsupported FP8 ScalingGranularity: {config.granularity}")
