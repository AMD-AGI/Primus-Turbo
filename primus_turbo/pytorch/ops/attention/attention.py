###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from typing import Optional

import torch

from primus_turbo.pytorch.kernels.attention.attention_csrc_impl import (
    attention_aiter_csrc_backward_impl,
    attention_aiter_csrc_forward_impl,
)
from primus_turbo.pytorch.kernels.attention.attention_triton_impl import (
    attention_triton_backward_impl,
    attention_triton_forward_impl,
)
from primus_turbo.pytorch.ops.attention.attention_cp_dispatcher import (
    dispatch_attention_cp_functions,
)
from primus_turbo.pytorch.ops.utils.attention_utils import (
    block_scaling_node,
    quant_v_get_p_scale,
)

__all__ = ["attention", "attention_fp8_blockwise"]


class AttentionCKFunction(torch.autograd.Function):
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
        is_v3_atomic_fp32: Optional[bool] = True,
        how_v3_bf16_cvt: Optional[int] = 1,
    ):
        is_grad = is_grad_enabled and any(x.requires_grad for x in [q, k, v])
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)
        head_size_q_og = q.size(3)
        head_size_v_og = v.size(3)
        # todo fix if head_size_v_og!=head_size_q_og, no padding
        if head_size_q_og != head_size_v_og:
            v = torch.nn.functional.pad(v, [0, head_size_q_og - head_size_v_og])
        if head_size_q_og % 8 != 0:
            q = torch.nn.functional.pad(q, [0, 8 - head_size_q_og % 8])
            k = torch.nn.functional.pad(k, [0, 8 - head_size_q_og % 8])
        if head_size_v_og % 8 != 0:
            v = torch.nn.functional.pad(v, [0, 8 - head_size_v_og % 8])

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
            ctx.is_v3_atomic_fp32 = is_v3_atomic_fp32
            ctx.how_v3_bf16_cvt = how_v3_bf16_cvt
        out = out_padded[..., :head_size_v_og]

        result = [out]
        if return_lse:
            result.append(softmax_lse)
        if return_softmax:
            result.append(S_dmask)

        return result[0] if len(result) == 1 else tuple(result)

    @staticmethod
    def backward(ctx, dout, *args):
        q, k, v, out, softmax_lse, rng_state = ctx.saved_tensors
        dq, dk, dv = torch.zeros_like(q), torch.empty_like(k), torch.empty_like(v)
        bias = ctx.bias
        dbias = torch.empty_like(bias) if bias is not None else None
        head_size_q_og = ctx.head_size_q_og
        head_size_v_og = dout.size(3)
        dout_padded = dout
        if head_size_v_og % 8 != 0:
            dout_padded = torch.nn.functional.pad(dout, [0, 8 - head_size_v_og % 8])
        if head_size_q_og != head_size_v_og:
            dout_padded = torch.nn.functional.pad(dout, [0, head_size_q_og - head_size_v_og])
        attention_aiter_csrc_backward_impl(
            dout_padded,
            q,
            k,
            v,
            out,
            softmax_lse,
            dq,
            dk,
            dv,
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
        dv = dv[..., :head_size_v_og]
        return dq, dk, dv, None, None, None, None, dbias, None, None, None, None, None


class AttentionTritonFunction(torch.autograd.Function):
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

        q, q_scale = block_scaling_node(q, use_fp8)
        k, k_scale = block_scaling_node(k, use_fp8)
        v, v_scale, p_scale = quant_v_get_p_scale(v, use_fp8)

        output, softmax_lse, exp_scores = attention_triton_forward_impl(
            q,
            k,
            v,
            p_scale,
            q_scale,
            k_scale,
            v_scale,
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
            ctx.save_for_backward(q, k, v, output, softmax_lse, alibi_slopes, bias, q_scale, k_scale, v_scale)

            ctx.sm_scale = softmax_scale
            ctx.p_scale = p_scale
            ctx.causal = causal
            ctx.use_fp8 = use_fp8
            ctx.cu_seqlens_q = torch.tensor(0, device="cuda")
            ctx.cu_seqlens_k = torch.tensor(0, device="cuda")
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
        (q, k, v, o, softmax_lse, alibi_slopes, bias, q_scale, k_scale, v_scale) = ctx.saved_tensors
        assert bias is None, "Currently bias is not supported by fa backward function."
        assert do.dtype is torch.bfloat16, f"do should be bfloat16 but get {do.dtype}"

        dq, dk, dv = attention_triton_backward_impl(
            do,
            q,
            k,
            v,
            o,
            q_scale,
            k_scale,
            v_scale,
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


def attention(
    q,
    k,
    v,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite context window
    bias=None,
    alibi_slopes=None,
    deterministic=True,
    return_lse=False,
    return_attn_probs=False,
    backend_type: str = "ck",  # 'ck', 'triton'
    cp_param_bundle=None,
):
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)

    if cp_param_bundle is not None:  # CP
        assert "cp_group" in cp_param_bundle
        assert "cp_comm_type" in cp_param_bundle
        return dispatch_attention_cp_functions(
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
            backend_type,
            False,
            cp_param_bundle["cp_group"],
            cp_param_bundle["cp_comm_type"],
        )

    if backend_type == "ck":
        return AttentionCKFunction.apply(
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
        )
    elif backend_type == "triton":
        return AttentionTritonFunction.apply(
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
            False,
        )
    else:
        raise NotImplementedError(f"backend_type {backend_type} not supported")


def attention_fp8_blockwise(
    q,
    k,
    v,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite context window
    bias=None,
    alibi_slopes=None,
    deterministic=False,
    return_lse=False,
    return_attn_probs=False,
    backend_type: str = "triton",  # for now 'triton' only
    cp_param_bundle=None,
):
    assert backend_type == "triton", "attention_fp8_blockwise only support triton backend"
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)

    if cp_param_bundle is not None:  # CP

        assert "cp_group" in cp_param_bundle
        assert "cp_comm_type" in cp_param_bundle
        return dispatch_attention_cp_functions(
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
            backend_type,
            True,
            cp_param_bundle["cp_group"],
            cp_param_bundle["cp_comm_type"],
        )

    return AttentionTritonFunction.apply(
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
