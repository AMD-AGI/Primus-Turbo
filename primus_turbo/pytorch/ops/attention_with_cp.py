import torch

from primus_turbo.pytorch.kernels.attention.attention_triton_impl import (
    attention_triton_backward_impl,
    attention_triton_forward_impl,
)
from primus_turbo.pytorch.ops.utils.attention_utils import (
    All2AllAttentionCommunicator,
    blockwise_scaling_qkv_to_fp8,
)


class AttentionTritonFunctionCPA2A(torch.autograd.Function):
    """
    QKV split by attention heads and a2a
    Refer the paper `DeepSpeed Ulysses <https://arxiv.org/abs/2309.14509>` for detail.
    """

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
        is_grad,
        use_fp8,
        cp_group,
    ):
        q_heads = q.shape[-2]
        kv_heads = k.shape[-2]
        cp_size = cp_group.size()
        assert q_heads % cp_size == 0
        assert kv_heads % cp_size == 0

        attention_communicator = All2AllAttentionCommunicator(cp_group)
        # original shape
        q_shape = q.shape
        k_shape = k.shape
        v_shape = v.shape

        # bshd
        seq_dim = 1
        # todo: overlap with a2a
        # [b, s // n, h, d] -> [b, s // n, n, h // n, d] -> [n, b, s // n, h // n, d]
        q, k, v = [
            x.view(*x.shape[:-2], cp_size, x.shape[-2] // cp_size, x.shape[-1]).movedim(-3, 0).contiguous()
            for x in [q, k, v]
        ]

        send_tensors = [q, k, v]
        q_local_heads = torch.empty_like(q)
        k_local_heads = torch.empty_like(k)
        v_local_heads = torch.empty_like(v)

        attention_communicator.data_exchange_over_cp_groups_async(
            send_tensors, [q_local_heads, k_local_heads, v_local_heads], 0
        )
        attention_communicator.wait_data_exchange_done()

        # [n, b, s // n, h // n, d] -> [b, n, s // n, h // n, d]
        q_local_heads, k_local_heads, v_local_heads = [
            x.movedim(0, seq_dim).contiguous() for x in [q_local_heads, k_local_heads, v_local_heads]
        ]

        # reshape to bshd
        q_local_heads = q_local_heads.view(q_shape[0], q_shape[1] * cp_size, -1, q_shape[3])
        k_local_heads = k_local_heads.view(k_shape[0], k_shape[1] * cp_size, -1, k_shape[3])
        v_local_heads = v_local_heads.view(v_shape[0], v_shape[1] * cp_size, -1, v_shape[3])

        # do local attention (todo: also could be done in parallel)
        q_local_heads, k_local_heads, v_local_heads, p_scale, q_scale, k_scale, v_scale = (
            blockwise_scaling_qkv_to_fp8(q_local_heads, k_local_heads, v_local_heads, use_fp8)
        )

        output_local_heads, softmax_lse, exp_scores = attention_triton_forward_impl(
            q_local_heads,
            k_local_heads,
            v_local_heads,
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

        # attention result all2all
        # [b, s, h // n, d] -> [b, n, s // n, h // n, d] -> [n, b, s // n, h // n, d]
        output_local_heads_a2a = (
            output_local_heads.view(
                output_local_heads.shape[0],
                cp_size,
                output_local_heads.shape[1] // cp_size,
                *output_local_heads.shape[2:],
            )
            .movedim(seq_dim, 0)
            .contiguous()
        )

        output_local_tokens = torch.empty_like(output_local_heads_a2a)
        attention_communicator.data_exchange_over_cp_groups_async(
            [output_local_heads_a2a], [output_local_tokens], -1
        )
        attention_communicator.wait_data_exchange_done()

        # [n, b, s // n, h // n, d] -> [b, s // n, n, h // n, d]
        output_local_tokens = output_local_tokens.movedim(0, -3).contiguous()
        # [b, s // n, n, h // n, d] -> [b, s // n, h, d]
        output_local_tokens = output_local_tokens.view(
            *output_local_tokens.shape[:-3],
            output_local_tokens.shape[-3] * output_local_tokens.shape[-2],
            output_local_tokens.shape[-1],
        )

        # save_ctx for backward
        if is_grad:
            # q, k, v should be fp8 when set use_fp8 to True
            ctx.save_for_backward(
                q_local_heads,
                k_local_heads,
                v_local_heads,
                output_local_heads,
                softmax_lse,
                alibi_slopes,
                bias,
                q_scale,
                k_scale,
                v_scale,
            )
            ctx.sm_scale = softmax_scale
            ctx.p_scale = p_scale
            ctx.causal = causal
            ctx.use_fp8 = use_fp8
            ctx.cu_seqlens_q = torch.tensor(0, device="cuda")
            ctx.cu_seqlens_k = torch.tensor(0, device="cuda")
            ctx.max_seqlens_q = q.shape[1]
            ctx.max_seqlens_k = k.shape[1]
            ctx.attention_communicator = attention_communicator
            ctx.q_shape = q_shape
            ctx.k_shape = k_shape
            ctx.v_shape = v_shape
            ctx.seq_dim = seq_dim
            ctx.cp_group = cp_group

        result = [output_local_tokens]
        if return_lse:
            result.append(softmax_lse)
        if return_softmax:
            result.append(exp_scores)
        return result[0] if len(result) == 1 else tuple(result)

    @staticmethod
    def backward(ctx, dout, *args):
        (
            q_local_heads,
            k_local_heads,
            v_local_heads,
            output_local_heads,
            softmax_lse,
            alibi_slopes,
            bias,
            q_scale,
            k_scale,
            v_scale,
        ) = ctx.saved_tensors
        assert bias is None, "Currently bias is not supported by fa backward function."
        assert dout.dtype is torch.bfloat16, f"dout should be bfloat16 but get {dout.dtype}"
        attention_communicator = ctx.attention_communicator

        # all2all o_grad
        dout_local_heads = torch.empty_like(output_local_heads)
        cp_size = ctx.cp_group.size()
        seq_dim = ctx.seq_dim

        # [b, s // n, h, d] -> [b, s // n, n , h // n, d] -> [n, b, s // n, h // n, d]
        dout = (
            dout.view(*dout.shape[:-2], cp_size, dout.shape[-2] // cp_size, dout.shape[-1])
            .movedim(-3, 0)
            .contiguous()
        )

        attention_communicator.data_exchange_over_cp_groups_async([dout], [dout_local_heads], -1)
        attention_communicator.wait_data_exchange_done()

        # [n, b, s // n, h // n, d] -> [b, n, s // n, h // n, d]
        dout_local_heads = dout_local_heads.view(*dout.shape).movedim(0, seq_dim).contiguous()
        # [b, n, s // n, h // n, d] -> [b, s, h // n, d]
        dout_local_heads = dout_local_heads.view(
            *dout_local_heads.shape[:seq_dim],
            dout_local_heads.shape[seq_dim] * dout_local_heads.shape[seq_dim + 1],
            *dout_local_heads.shape[seq_dim + 2 :],
        )
        # local backward function
        dq_local_heads, dk_local_heads, dv_local_heads = attention_triton_backward_impl(
            dout_local_heads,
            q_local_heads,
            k_local_heads,
            v_local_heads,
            output_local_heads,
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

        # [b, s, h // n, d] -> [b, n, s // n, h // n, d] -> [n, b, s // n, h // n, d]
        dq_local_heads, dk_local_heads, dv_local_heads = [
            x.view(x.shape[0], cp_size, x.shape[1] // cp_size, *x.shape[2:]).movedim(seq_dim, 0).contiguous()
            for x in [dq_local_heads, dk_local_heads, dv_local_heads]
        ]

        # all2all d_{q/k/v}
        dq_local_tokens = torch.empty_like(dq_local_heads, dtype=torch.bfloat16)
        dk_local_tokens = torch.empty_like(dk_local_heads, dtype=torch.bfloat16)
        dv_local_tokens = torch.empty_like(dv_local_heads, dtype=torch.bfloat16)

        attention_communicator.data_exchange_over_cp_groups_async(
            [dq_local_heads, dk_local_heads, dv_local_heads],
            [dq_local_tokens, dk_local_tokens, dv_local_tokens],
            -1,
        )
        attention_communicator.wait_data_exchange_done()

        # [n, b, s // n, h // n, d] -> [b, s // n, n, h // n, d] -> [b, s // n, h, d]
        dq_local_tokens, dk_local_tokens, dv_local_tokens = [
            x.movedim(0, -3).contiguous() for x in [dq_local_tokens, dk_local_tokens, dv_local_tokens]
        ]

        dq_local_tokens = dq_local_tokens.view(ctx.q_shape)
        dk_local_tokens = dk_local_tokens.view(ctx.k_shape)
        dv_local_tokens = dv_local_tokens.view(ctx.v_shape)

        return (
            dq_local_tokens,
            dk_local_tokens,
            dv_local_tokens,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def dispatch_attention_triton_functions(
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
    is_grad_enabled,
    fp8,
    cp_group,
    cp_stream,
    cp_comm_type,
):
    if cp_comm_type == "all2all":
        return AttentionTritonFunctionCPA2A.apply(
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
            is_grad_enabled,
            fp8,
            cp_group,
        )
    else:
        raise NotImplementedError(f"not supported cp_comm_type {cp_comm_type} yet")
