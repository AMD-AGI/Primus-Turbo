###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import os
from typing import Optional

import torch

from primus_turbo.pytorch.core.low_precision import (
    Float8QuantConfig,
    ScalingGranularity,
)
from primus_turbo.pytorch.core.utils import get_device_compute_capability
from primus_turbo.pytorch.kernels.attention.attention_csrc_impl import (
    attention_aiter_csrc_backward_impl,
    attention_aiter_csrc_forward_impl,
)
from primus_turbo.pytorch.kernels.attention.attention_triton_impl import (
    attention_aiter_triton_backward_impl,
    attention_aiter_triton_forward_impl,
    attention_triton_backward_impl,
    attention_triton_forward_impl,
)
from primus_turbo.pytorch.ops.attention.attention_utils import (
    block_scaling_node,
    get_p_scale,
)

__all__ = ["flash_attn_func", "flash_attn_fp8_func"]


class AiterFlashAttnFunc(torch.autograd.Function):
    @staticmethod
    def _resolve_is_v3_atomic_fp32(is_v3_atomic_fp32: Optional[bool]) -> bool:
        if is_v3_atomic_fp32 in [True, False]:
            return is_v3_atomic_fp32
        val = os.getenv("PRIMUS_TURBO_ATTN_V3_ATOMIC_FP32", "1")
        return val == "1" if val in ("0", "1") else True

    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        bias,
        alibi_slopes,
        deterministic,
        return_lse,
        return_softmax,
        is_grad_enabled,
        is_v3_atomic_fp32: Optional[bool] = None,
        how_v3_bf16_cvt: Optional[int] = 1,
        sink: Optional[torch.Tensor] = None,
    ):
        # MI355 (gfx950): better perf when is_v3_atomic_fp32=False
        # Controlled by env var PRIMUS_TURBO_ATTN_V3_ATOMIC_FP32
        is_v3_atomic_fp32 = AiterFlashAttnFunc._resolve_is_v3_atomic_fp32(is_v3_atomic_fp32)

        # Avoid aiter print warning when how_v3_bf16_cvt!=0 in gfx950.
        if get_device_compute_capability() >= (9, 5):
            how_v3_bf16_cvt = 0

        is_grad = is_grad_enabled and any(x.requires_grad for x in [q, k, v])

        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)

        head_size_q_og = q.size(3)
        head_size_v_og = v.size(3)
        if head_size_q_og % 8 != 0:
            q = torch.nn.functional.pad(q, [0, 8 - head_size_q_og % 8])
            k = torch.nn.functional.pad(k, [0, 8 - head_size_q_og % 8])
        if head_size_v_og % 8 != 0:
            v = torch.nn.functional.pad(v, [0, 8 - head_size_v_og % 8])

        # Use Triton backend when sink is provided, as C++ backend doesn't support sink feature
        use_triton = sink is not None
        rng_state = None
        philox_seed = 0
        philox_offset = 0

        if not use_triton:
            out_padded, softmax_lse, S_dmask, rng_state = attention_aiter_csrc_forward_impl(
                q,
                k,
                v,
                dropout_p,
                softmax_scale,
                causal=causal,
                window_size_left=int(window_size[0]),
                window_size_right=int(window_size[1]),
                bias=bias,
                alibi_slopes=alibi_slopes,
                return_lse=True,
                return_softmax=return_softmax and dropout_p > 0,
            )
        else:
            out_padded, softmax_lse, S_dmask, philox_seed, philox_offset = (
                attention_aiter_triton_forward_impl(
                    q,
                    k,
                    v,
                    dropout_p,
                    softmax_scale,
                    causal=causal,
                    window_size_left=int(window_size[0]),
                    window_size_right=int(window_size[1]),
                    bias=bias,
                    alibi_slopes=alibi_slopes,
                    return_lse=True,
                    return_softmax=return_softmax and dropout_p > 0,
                    max_seqlen_q=q.shape[1],
                    max_seqlen_k=k.shape[1],
                    sink=sink,
                )
            )

        if is_grad:
            if use_triton:
                ctx.save_for_backward(q, k, v, out_padded, softmax_lse, sink)
            else:
                ctx.save_for_backward(q, k, v, out_padded, softmax_lse, rng_state)
            ctx.use_triton = use_triton
            ctx.dropout_p = dropout_p
            ctx.softmax_scale = softmax_scale
            ctx.causal = causal
            ctx.window_size = window_size
            ctx.bias = bias
            ctx.alibi_slopes = alibi_slopes
            ctx.deterministic = deterministic
            ctx.head_size_q_og = head_size_q_og
            ctx.head_size_v_og = head_size_v_og
            ctx.is_v3_atomic_fp32 = is_v3_atomic_fp32
            ctx.how_v3_bf16_cvt = how_v3_bf16_cvt
            ctx.philox_seed = philox_seed
            ctx.philox_offset = philox_offset

        out = out_padded[..., :head_size_v_og]

        result = [out]
        if return_lse:
            result.append(softmax_lse)
        if return_softmax:
            result.append(S_dmask)

        return result[0] if len(result) == 1 else tuple(result)

    @staticmethod
    def backward(ctx, dout, *args):
        use_triton = ctx.use_triton
        head_size_q_og = ctx.head_size_q_og
        head_size_v_og = ctx.head_size_v_og
        bias = ctx.bias
        dbias = torch.empty_like(bias) if bias is not None else None

        if use_triton:
            q, k, v, out_padded, softmax_lse, sink = ctx.saved_tensors
            # dsink must be zeros as kernel accumulates gradients via atomic adds
            dsink = torch.zeros_like(sink, dtype=torch.float32) if sink is not None else None
        else:
            q, k, v, out_padded, softmax_lse, rng_state = ctx.saved_tensors
            sink = None
            dsink = None

        dout_padded = dout
        v_padded = v

        if head_size_v_og % 8 != 0:
            dout_padded = torch.nn.functional.pad(dout, [0, 8 - head_size_v_og % 8])
        if head_size_q_og != head_size_v_og:
            v_padded = torch.nn.functional.pad(v_padded, [0, head_size_q_og - head_size_v_og])
            out_padded = torch.nn.functional.pad(out_padded, [0, head_size_q_og - head_size_v_og])
            dout_padded = torch.nn.functional.pad(dout, [0, head_size_q_og - head_size_v_og])

        dq, dk, dv_padded = torch.zeros_like(q), torch.empty_like(k), torch.empty_like(v_padded)

        if use_triton:
            attention_aiter_triton_backward_impl(
                dout_padded,
                q,
                k,
                v_padded,
                out_padded,
                softmax_lse,
                dq,
                dk,
                dv_padded,
                dbias,
                ctx.softmax_scale,
                ctx.alibi_slopes,
                ctx.causal,
                None,  # cu_seqlens_q
                None,  # cu_seqlens_k
                q.shape[1],  # max_seqlen_q
                k.shape[1],  # max_seqlen_k
                ctx.dropout_p,
                ctx.philox_seed,
                ctx.philox_offset,
                True,  # USE_INT64_STRIDES
                sink=sink,
                dsink=dsink,
            )
        else:
            attention_aiter_csrc_backward_impl(
                dout_padded,
                q,
                k,
                v_padded,
                out_padded,
                softmax_lse,
                dq,
                dk,
                dv_padded,
                dbias,
                ctx.dropout_p,
                ctx.softmax_scale,
                ctx.causal,
                int(ctx.window_size[0]),
                int(ctx.window_size[1]),
                ctx.bias,
                ctx.alibi_slopes,
                ctx.deterministic,
                rng_state,
                ctx.is_v3_atomic_fp32,
                ctx.how_v3_bf16_cvt,
            )

        dq = dq[..., :head_size_q_og]  # We could have padded the head dimension
        dk = dk[..., :head_size_q_og]
        dv = dv_padded[..., :head_size_v_og]
        return dq, dk, dv, None, None, None, None, dbias, None, None, None, None, None, None, None, dsink


class TritonFlashAttnFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        bias,
        alibi_slopes,
        return_lse,
        return_softmax,
        is_grad_enabled,
        use_fp8,
    ):
        is_grad = is_grad_enabled and any(x.requires_grad for x in [q, k, v])
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)

        q, q_descale = block_scaling_node(q, use_fp8)
        k, k_descale = block_scaling_node(k, use_fp8)
        v, v_descale = block_scaling_node(v, use_fp8)
        p_scale = get_p_scale(use_fp8)

        output, softmax_lse, exp_scores = attention_triton_forward_impl(
            q,
            k,
            v,
            p_scale,
            q_descale,
            k_descale,
            v_descale,
            dropout_p,
            softmax_scale,
            causal,
            window_size[0],
            window_size[1],
            bias,
            alibi_slopes,
            return_softmax,
            use_fp8,
        )

        if is_grad:
            # q, k, v should be fp8 when set use_fp8 to True
            ctx.save_for_backward(
                q, k, v, output, softmax_lse, alibi_slopes, bias, q_descale, k_descale, v_descale
            )

            ctx.sm_scale = softmax_scale
            ctx.p_scale = p_scale
            ctx.causal = causal
            ctx.use_fp8 = use_fp8
            ctx.cu_seqlens_q = 0
            ctx.cu_seqlens_k = 0
            ctx.max_seqlens_q = q.shape[1]
            ctx.max_seqlens_k = k.shape[1]

        result = [output]
        if return_lse:
            result.append(softmax_lse)
        if return_softmax:
            result.append(exp_scores)
        return result[0] if len(result) == 1 else tuple(result)

    @staticmethod
    def backward(ctx, do, *args):
        (q, k, v, o, softmax_lse, alibi_slopes, bias, q_descale, k_descale, v_descale) = ctx.saved_tensors
        assert bias is None, "Currently bias is not supported by fa backward function."
        assert do.dtype is torch.bfloat16, f"do should be bfloat16 but get {do.dtype}"

        dq, dk, dv = attention_triton_backward_impl(
            do,
            q,
            k,
            v,
            o,
            q_descale,
            k_descale,
            v_descale,
            ctx.p_scale,
            softmax_lse,
            None,
            None,
            None,
            ctx.cu_seqlens_q,
            ctx.cu_seqlens_k,
            ctx.max_seqlens_q,
            ctx.max_seqlens_k,
            ctx.sm_scale,
            ctx.causal,
            -1,
            -1,
            alibi_slopes,
            ctx.use_fp8,
        )

        return dq, dk, dv, None, None, None, None, None, None, None, None, None, None


def flash_attn_func(
    q,
    k,
    v,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    bias=None,
    alibi_slopes=None,
    deterministic=False,
    return_lse=False,
    return_attn_probs=False,
    sink: Optional[torch.Tensor] = None,
):
    return AiterFlashAttnFunc.apply(
        q,
        k,
        v,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        bias,
        alibi_slopes,
        deterministic,
        return_lse,
        return_attn_probs,
        torch.is_grad_enabled(),
        None,  # is_v3_atomic_fp32
        1,  # how_v3_bf16_cvt
        sink,
    )


def flash_attn_fp8_func(
    q,
    k,
    v,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    bias=None,
    alibi_slopes=None,
    deterministic=False,
    return_lse=False,
    return_attn_probs=False,
    fp8_config: Optional[Float8QuantConfig] = None,
):
    # Default config: blockwise with block_size=64
    if fp8_config is None:
        fp8_config = Float8QuantConfig(
            granularity=ScalingGranularity.BLOCKWISE,
            block_size=64,
        )

    # Check if config is supported
    if fp8_config.granularity != ScalingGranularity.BLOCKWISE:
        raise ValueError(
            f"flash_attn_fp8_func only supports BLOCKWISE granularity, " f"but got {fp8_config.granularity}"
        )
    if fp8_config.block_size != 64:
        raise ValueError(
            f"flash_attn_fp8_func only supports block_size=64, " f"but got block_size={fp8_config.block_size}"
        )

    return TritonFlashAttnFunc.apply(
        q,
        k,
        v,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        bias,
        alibi_slopes,
        return_lse,
        return_attn_probs,
        torch.is_grad_enabled(),
        True,
    )
