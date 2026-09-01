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
    ScalingGranularity,
    ScalingRecipe,
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
    grouped_gemm_fp8_variable_k_accum_impl,
    grouped_gemm_fp8_variable_k_impl,
)
from primus_turbo.pytorch.kernels.grouped_gemm.grouped_gemm_utils import (
    group_offs_from_lens,
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


_SUPPORTED_ACTIVATIONS = ("silu",)


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
        out_dtype: torch.dtype,
        config: Float8QuantConfig,
        num_cu: int | None,
        fuse_wgrad_accum_pattern: Union[None, str] = None,
    ):
        assert activation in _SUPPORTED_ACTIVATIONS, (
            f"Unsupported activation: {activation!r}, expected one of {_SUPPORTED_ACTIVATIONS}"
        )

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
) -> torch.Tensor:
    if config is None:
        config = Float8QuantConfig()

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
            out_dtype,
            config,
            num_cu,
            fuse_wgrad_accum_pattern,
        )
    else:
        raise ValueError(f"Unsupported FP8 ScalingGranularity: {config.granularity}")
