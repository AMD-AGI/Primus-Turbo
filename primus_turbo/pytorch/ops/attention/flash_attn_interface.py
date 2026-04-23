###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from typing import Optional

import torch

from primus_turbo.pytorch.core.low_precision import (
    Float8QuantConfig,
    ScalingGranularity,
)
from primus_turbo.pytorch.core.utils import get_device_compute_capability
from primus_turbo.pytorch.kernels.attention.attention_aiter_impl import (
    attention_aiter_backward_impl,
    attention_aiter_forward_impl,
)
from primus_turbo.pytorch.kernels.attention.attention_triton_impl import (
    attention_triton_backward_impl,
    attention_triton_forward_impl,
)
from primus_turbo.pytorch.ops.attention.attention_utils import (
    _infer_qkv_format,
    _resolve_is_v3_atomic_fp32_from_env,
    block_scaling_node,
    get_p_scale,
)

__all__ = ["flash_attn_func", "flash_attn_fp8_func"]


class AiterFlashAttnFunc(torch.autograd.Function):

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
        how_v3_bf16_cvt: Optional[int] = 1,
        sink: Optional[torch.Tensor] = None,
        qkv_format: Optional[str] = "bshd",
    ):
        assert not (deterministic and sink is not None), (
            "deterministic and sink cannot be enabled together currently; "
            "please set deterministic=False or sink=None"
        )
        # MI355 (gfx950): better perf when is_v3_atomic_fp32=False
        # Controlled by env var PRIMUS_TURBO_ATTN_V3_ATOMIC_FP32
        is_v3_atomic_fp32 = _resolve_is_v3_atomic_fp32_from_env()

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

        enable_sink = sink is not None
        if enable_sink:
            # NOTE: Sink attention is not supported for sbhd format. Fallback to BSHD.
            q = q.contiguous()
            k = k.contiguous()
            v = v.contiguous()

        out_padded, softmax_lse, S_dmask, rng_state = attention_aiter_forward_impl(
            q=q,
            k=k,
            v=v,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size_left=int(window_size[0]),
            window_size_right=int(window_size[1]),
            bias=bias,
            alibi_slopes=alibi_slopes,
            return_lse=True,
            return_softmax=return_softmax and dropout_p > 0,
            max_seqlen_q=q.size(1),
            max_seqlen_k=k.size(1),
            sink=sink,
            qkv_format=qkv_format if not enable_sink else "bshd",
        )

        if enable_sink:
            if qkv_format == "sbhd":
                # permute back to SBHD
                out_padded = out_padded.permute(1, 0, 2, 3).contiguous()
            elif qkv_format == "bhsd":
                out_padded = out_padded.permute(0, 2, 1, 3).contiguous()
            else:
                assert qkv_format == "bshd"

        if is_grad:
            ctx.save_for_backward(q, k, v, out_padded, softmax_lse, rng_state)
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
            ctx.sink = sink
            ctx.qkv_format = qkv_format

        out = out_padded[..., :head_size_v_og]

        result = [out]
        if return_lse:
            result.append(softmax_lse)
        if return_softmax:
            result.append(S_dmask)

        return result[0] if len(result) == 1 else tuple(result)

    @staticmethod
    def backward(ctx, dout, *args):
        head_size_q_og = ctx.head_size_q_og
        head_size_v_og = ctx.head_size_v_og

        q, k, v, out_padded, softmax_lse, rng_state = ctx.saved_tensors

        enable_sink = ctx.sink is not None
        qkv_format = ctx.qkv_format

        if enable_sink and qkv_format in ("sbhd", "bhsd"):
            dout = dout.contiguous()

        dout_padded = dout
        if head_size_v_og % 8 != 0:
            dout_padded = torch.nn.functional.pad(dout, [0, 8 - head_size_v_og % 8])

        bwd_qkv_format = qkv_format if not enable_sink else "bshd"
        batch_size, seq_len, num_heads_q, head_dim_qk = q.size()
        _, _, num_heads_k, head_dim_k = k.size()
        _, _, num_heads_v, head_dim_v = v.size()

        if bwd_qkv_format == "sbhd":
            dq = torch.ones(
                (seq_len, batch_size, num_heads_q, head_dim_qk), dtype=q.dtype, device=q.device
            ).permute(1, 0, 2, 3)
            dk = torch.empty(
                (seq_len, batch_size, num_heads_k, head_dim_k), dtype=k.dtype, device=k.device
            ).permute(1, 0, 2, 3)
            dv_padded = torch.empty(
                (seq_len, batch_size, num_heads_v, head_dim_v), dtype=v.dtype, device=v.device
            ).permute(1, 0, 2, 3)
        elif bwd_qkv_format == "bhsd":
            dq = torch.ones(
                (batch_size, num_heads_q, seq_len, head_dim_qk), dtype=q.dtype, device=q.device
            ).permute(0, 2, 1, 3)
            dk = torch.empty(
                (batch_size, num_heads_k, seq_len, head_dim_k), dtype=k.dtype, device=k.device
            ).permute(0, 2, 1, 3)
            dv_padded = torch.empty(
                (batch_size, num_heads_v, seq_len, head_dim_v), dtype=v.dtype, device=v.device
            ).permute(0, 2, 1, 3)
        else:
            dq = torch.ones((batch_size, seq_len, num_heads_q, head_dim_qk), dtype=q.dtype, device=q.device)
            dk = torch.empty((batch_size, seq_len, num_heads_k, head_dim_k), dtype=k.dtype, device=k.device)
            dv_padded = torch.empty(
                (batch_size, seq_len, num_heads_v, head_dim_v), dtype=v.dtype, device=v.device
            )
        dbias = torch.empty_like(ctx.bias) if ctx.bias is not None else None
        dsink = torch.zeros_like(ctx.sink, dtype=torch.float32) if ctx.sink is not None else None

        _ = attention_aiter_backward_impl(
            dout=dout_padded,
            q=q,
            k=k,
            v=v,
            out=out_padded,
            softmax_lse=softmax_lse,
            dq=dq,
            dk=dk,
            dv=dv_padded,
            dropout_p=ctx.dropout_p,
            softmax_scale=ctx.softmax_scale,
            causal=ctx.causal,
            window_size_left=int(ctx.window_size[0]),
            window_size_right=int(ctx.window_size[1]),
            bias=ctx.bias,
            alibi_slopes=ctx.alibi_slopes,
            deterministic=ctx.deterministic,
            rng_state=rng_state,
            is_v3_atomic_fp32=ctx.is_v3_atomic_fp32,
            how_v3_bf16_cvt=ctx.how_v3_bf16_cvt,
            dbias=dbias,
            dsink=dsink,
            sink=ctx.sink,
            qkv_format=bwd_qkv_format,
        )

        if enable_sink:
            if qkv_format == "sbhd":
                dq = dq.permute(1, 0, 2, 3).contiguous()
                dk = dk.permute(1, 0, 2, 3).contiguous()
                dv_padded = dv_padded.permute(1, 0, 2, 3).contiguous()
            elif qkv_format == "bhsd":
                dq = dq.permute(0, 2, 1, 3).contiguous()
                dk = dk.permute(0, 2, 1, 3).contiguous()
                dv_padded = dv_padded.permute(0, 2, 1, 3).contiguous()
            else:
                assert qkv_format == "bshd"

        dq = dq[..., :head_size_q_og]
        dk = dk[..., :head_size_q_og]
        dv = dv_padded[..., :head_size_v_og]

        return (
            dq,
            dk,
            dv,
            None,
            None,
            None,
            None,
            dbias,
            None,
            None,
            None,
            None,
            None,
            None,
            dsink,
            None,
        )


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
        q, k, v, o, softmax_lse, alibi_slopes, bias, q_descale, k_descale, v_descale = ctx.saved_tensors
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
    qkv_format = _infer_qkv_format(q, k, v)

    result = AiterFlashAttnFunc.apply(
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
        1,  # how_v3_bf16_cvt
        sink,
        qkv_format,
    )

    return result


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
    qkv_format = _infer_qkv_format(q, k, v)

    if qkv_format == "bhsd":
        q = q.permute(0, 2, 1, 3).contiguous()
        k = k.permute(0, 2, 1, 3).contiguous()
        v = v.permute(0, 2, 1, 3).contiguous()
    elif qkv_format == "sbhd":
        q = q.permute(1, 0, 2, 3).contiguous()
        k = k.permute(1, 0, 2, 3).contiguous()
        v = v.permute(1, 0, 2, 3).contiguous()

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

    o = TritonFlashAttnFunc.apply(
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

    if qkv_format == "sbhd":
        o = o.permute(1, 0, 2, 3).contiguous()
    elif qkv_format == "bhsd":
        o = o.permute(0, 2, 1, 3).contiguous()

    return o
