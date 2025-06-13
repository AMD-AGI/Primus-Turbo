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
from primus_turbo.triton.attention.attention_kernel import (
    FIXED_BLOCK_M,
    FIXED_BLOCK_N,
    block_scaling_node,
)


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


def attention_ck(
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
    use_fp8=False,
):
    """dropout_p should be set to 0.0 during evaluation
    Supports multi-query and grouped-query attention (MQA/GQA) by passing in KV with fewer heads
    than Q. Note that the number of heads in Q must be divisible by the number of heads in KV.
    For example, if Q has 6 heads and K, V have 2 heads, head 0, 1, 2 of Q will attention to head
    0 of K, V, and head 3, 4, 5 of Q will attention to head 1 of K, V.

    If causal=True, the causal mask is aligned to the bottom right corner of the attention matrix.
    For example, if seqlen_q = 2 and seqlen_k = 5, the causal mask (1 = keep, 0 = masked out) is:
        1 1 1 1 0
        1 1 1 1 1
    If seqlen_q = 5 and seqlen_k = 2, the causal mask is:
        0 0
        0 0
        0 0
        1 0
        1 1
    If the row of the mask is all zero, the output will be zero.

    If window_size != (-1, -1), implements sliding window local attention. Query at position i
    will only attend to keys between
    [i + seqlen_k - seqlen_q - window_size[0], i + seqlen_k - seqlen_q + window_size[1]] inclusive.

    Arguments:
        q: (batch_size, seqlen, nheads, headdim_q)
        k: (batch_size, seqlen, nheads_k, headdim_q)
        v: (batch_size, seqlen, nheads_k, headdim_v)
        dropout_p: float. Dropout probability.
        softmax_scale: float. The scaling of QK^T before applying softmax.
            Default to 1 / sqrt(headdim_q).
        causal: bool. Whether to apply causal attention mask (e.g., for auto-regressive modeling).
        window_size: (left, right). If not (-1, -1), implements sliding window local attention.
        bias: (seqlen_q, seqlen_k)
        alibi_slopes: (nheads,) or (batch_size, nheads), fp32. A bias of
            (-alibi_slope * |i + seqlen_k - seqlen_q - j|)
            is added to the attention score of query i and key j.
        deterministic: bool. Whether to use the deterministic implementation of the backward pass,
            which is slightly slower and uses more memory. The forward pass is always deterministic.
        return_attn_probs: bool. Whether to return the attention probabilities. This option is for
           testing only. The returned probabilities are not guaranteed to be correct
           (they might not have the right scaling).
    Return:
        out: (batch_size, seqlen, nheads, headdim_v).
        softmax_lse [optional, if return_attn_probs=True]: (batch_size, nheads, seqlen). The
            logsumexp of each row of the matrix QK^T * scaling (e.g., log of the softmax
            normalization factor).
        S_dmask [optional, if return_attn_probs=True]: (batch_size, nheads, seqlen, seqlen).
            The output of softmax (possibly with different scaling). It also encodes the dropout
            pattern (negative means that location was dropped, nonnegative means it was kept).
    """
    assert use_fp8 is False, "attention_ck does not support fp8"
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

        if use_fp8:
            # online quant
            range_v = torch.max(torch.abs(v))
            float8_fw = torch.float8_e4m3fnuz
            dtype_max = torch.finfo(float8_fw).max
            v_scale = dtype_max / range_v
            p_scale = dtype_max

            def check_and_convert(t, scale):
                finfo = torch.finfo(float8_fw)
                return (
                    (t * scale).clamp(min=finfo.min, max=finfo.max).to(dtype=float8_fw)
                    if t.dtype != float8_fw
                    else t
                )

            q, q_scale = block_scaling_node(q, FIXED_BLOCK_M)
            k, k_scale = block_scaling_node(k, FIXED_BLOCK_N)
            v = check_and_convert(v, v_scale)
        else:
            use_fp8 = False
            q_scale = torch.tensor([1.0], device=q.device)
            k_scale = torch.tensor([1.0], device=q.device)
            v_scale = torch.tensor([1.0], device=q.device)
            p_scale = 1.0

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


def attention_triton(
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
    use_fp8=True,
):

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
        use_fp8,
    )
