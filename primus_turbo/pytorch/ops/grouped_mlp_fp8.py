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
)
from primus_turbo.pytorch.core.quantized_tensor import (
    QuantizedTensor,
    QuantizedTensorPair,
    check_quantized_tensor,
)
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
                pad_align_last=128,
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
                **({"pad_align_last": 128} if trans_w1 else {"pad_align_penultimate": 128}),
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
                pad_align_last=128,
                pad_align_penultimate=128,
            )

        # Keep the fc1 gate||up layout unpadded, while fc2 pads both its
        # contraction and output dimensions. Real extents below preserve the
        # model-visible shapes through forward, dgrad, and wgrad.
        x_real = quantized_x.shape[-1]
        x_pitch = quantized_x.qdata.shape[-1]
        w1_k_real = quantized_w1.shape[-1] if trans_w1 else quantized_w1.shape[-2]
        w1_k_pitch = quantized_w1.qdata.shape[-1] if trans_w1 else quantized_w1.qdata.shape[-2]
        w1_out_real = quantized_w1.shape[-2] if trans_w1 else quantized_w1.shape[-1]
        w1_out_pitch = quantized_w1.qdata.shape[-2] if trans_w1 else quantized_w1.qdata.shape[-1]
        assert x_real == w1_k_real and x_pitch == w1_k_pitch
        assert w1_out_real == w1_out_pitch and w1_out_real % 2 == 0, (
            "fused GLU requires an unpadded even gate||up dimension"
        )
        act_real = w1_out_real // 2

        w2_k_real = quantized_w2.shape[-1] if trans_w2 else quantized_w2.shape[-2]
        w2_k_pitch = quantized_w2.qdata.shape[-1] if trans_w2 else quantized_w2.qdata.shape[-2]
        out_real = quantized_w2.shape[-2] if trans_w2 else quantized_w2.shape[-1]
        out_pitch = quantized_w2.qdata.shape[-2] if trans_w2 else quantized_w2.qdata.shape[-1]
        assert act_real == w2_k_real, f"fc2 K mismatch: activation={act_real}, weight={w2_k_real}"

        fc1_act, fc1_out = grouped_gemm_fp8_glu_impl(
            quantized_x.qdata,
            quantized_w1.qdata,
            quantized_x.scale_inv,
            quantized_w1.scale_inv,
            group_lens,
            group_offs,
            trans_a=False,
            trans_b=trans_w1,
            out_dtype=out_dtype,
            granularity=config.granularity.value,
            num_cu=num_cu,
            activation=activation,
            probs=probs,
            k_real=(x_real if x_real != x_pitch else None),
        )

        quantized_act = QuantizedTensor.quantize(
            fc1_act,
            x_dtype,
            config.granularity,
            axis=-1,
            block_size=config.block_size,
            group_lens=group_lens,
            pad_align_last=128,
        )
        assert quantized_act.shape[-1] == act_real
        assert quantized_act.qdata.shape[-1] == w2_k_pitch

        fc2_out = grouped_gemm_fp8_impl(
            quantized_act.qdata,
            quantized_w2.qdata,
            quantized_act.scale_inv,
            quantized_w2.scale_inv,
            group_lens,
            group_offs,
            trans_a=False,
            trans_b=trans_w2,
            out_dtype=out_dtype,
            granularity=config.granularity.value,
            num_cu=num_cu,
            default_backend=BackendType.FLYDSL.value,
            maybe_pre_sync=True,
            n_real=(out_real if out_real != out_pitch else None),
        )

        ctx.save_for_backward(
            quantized_x.qdata,
            quantized_act.qdata,
            quantized_w1.qdata,
            quantized_w2.qdata,
            quantized_x.scale_inv,
            quantized_act.scale_inv,
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
        ctx.x_real = x_real
        ctx.x_pitch = x_pitch
        ctx.act_real = act_real
        ctx.act_pitch = w2_k_pitch
        ctx.out_real = out_real
        ctx.out_pitch = out_pitch
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
            **({"pad_align_last": 128} if ctx.out_pitch != ctx.out_real else {}),
        )
        assert quantized_grad_out.shape[-1] == ctx.out_real
        assert quantized_grad_out.qdata.shape[-1] == ctx.out_pitch

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
            default_backend=BackendType.FLYDSL.value,
            inplace_add_to_out=ctx.fuse_w2_accum,
            out=ctx.w2_main_grad,
            m_real=(ctx.out_real if ctx.out_real != ctx.out_pitch else None),
            n_real=(ctx.act_real if ctx.act_real != ctx.act_pitch else None),
        )

        # fc2 dgrad (grad_out @ w2^T) with the activation gradient fused into its
        # epilogue, giving the pre-activation gradient directly. The probs
        # scaling and grad_probs both ride along inside that epilogue.
        grad_fc1_out, grad_probs = grouped_gemm_fp8_dglu_impl(
            quantized_grad_out.qdata,
            w2_fp8,
            quantized_grad_out.scale_inv,
            w2_scale_inv,
            group_lens,
            group_offs,
            trans_a=False,
            trans_b=not ctx.trans_w2,
            out_dtype=ctx.out_dtype,
            granularity=ctx.config.granularity.value,
            num_cu=ctx.num_cu,
            activation=ctx.activation,
            probs=probs,
            intermediate=fc1_out,
            n_real=(ctx.act_real if ctx.act_real != ctx.act_pitch else None),
        )

        quantized_grad_fc1_out = QuantizedTensor.quantize(
            grad_fc1_out,
            grad_out_dtype,
            ctx.config.granularity,
            axis=-1,
            block_size=ctx.config.block_size,
            group_lens=group_lens,
        )

        # grad_x = grad_fc1_out @ w1^T
        grad_x = grouped_gemm_fp8_impl(
            quantized_grad_fc1_out.qdata,
            w1_fp8,
            quantized_grad_fc1_out.scale_inv,
            w1_scale_inv,
            group_lens,
            group_offs,
            trans_a=False,
            trans_b=not ctx.trans_w1,
            out_dtype=ctx.out_dtype,
            granularity=ctx.config.granularity.value,
            num_cu=ctx.num_cu,
            default_backend=BackendType.FLYDSL.value,
            n_real=(ctx.x_real if ctx.x_real != ctx.x_pitch else None),
        )

        # grad_w1 = x^T @ grad_fc1_out
        grad_w1 = _grouped_gemm_fp8_variable_k_impl_wrapper(
            x_fp8,
            quantized_grad_fc1_out.qdata,
            x_scale_inv,
            quantized_grad_fc1_out.scale_inv,
            group_lens,
            group_offs,
            trans_a=True,
            trans_b=False,
            trans_c=ctx.trans_w1,
            out_dtype=ctx.out_dtype,
            granularity=ctx.config.granularity.value,
            num_cu=ctx.num_cu,
            default_backend=BackendType.FLYDSL.value,
            inplace_add_to_out=ctx.fuse_w1_accum,
            out=ctx.w1_main_grad,
            n_real=(ctx.x_real if ctx.x_real != ctx.x_pitch else None),
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
