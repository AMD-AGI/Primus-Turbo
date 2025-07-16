import functools
from typing import Tuple

import torch

from primus_turbo.pytorch.kernels.attention.attention_triton_impl import (
    attention_triton_backward_impl,
    attention_triton_forward_impl,
)
from primus_turbo.pytorch.ops.utils.attention_utils import (
    All2AllAttentionCommunicator,
    block_scaling_node,
    quant_v_get_p_scale,
)


def reshape_qkv_tensors_before_cp_a2a(
    send_tensor: torch.Tensor, recv_tensor: torch.Tensor, cp_size
) -> Tuple[torch.Tensor, torch.Tensor]:
    """reshape qkv tensors before all2all communication"""
    # [b, s // n, h, d] -> [b, s // n, n, h // n, d] -> [n, b, s // n, h // n, d]
    send_tensor = (
        send_tensor.view(
            *send_tensor.shape[:-2], cp_size, send_tensor.shape[-2] // cp_size, send_tensor.shape[-1]
        )
        .movedim(-3, 0)
        .contiguous()
    )
    recv_tensor = recv_tensor.view(*send_tensor.shape)
    return send_tensor, recv_tensor


def reshape_qkv_tensors_after_cp_a2a(
    recv_tensor: torch.Tensor, original_shape, cp_size, seq_dim
) -> torch.Tensor:
    """reshape qkv input tensors after all2all communication"""
    # [n, b, s // n, h // n, d] -> [b, n, s // n, h // n, d]
    recv_tensor = recv_tensor.movedim(0, seq_dim).contiguous()
    recv_tensor = recv_tensor.view(original_shape[0], original_shape[1] * cp_size, -1, original_shape[3])
    return recv_tensor


def reshape_and_quant_qkv_local_heads_after_cp_a2a(
    recv_tensor: torch.Tensor, original_shape, cp_size, seq_dim, use_fp8
) -> Tuple[torch.Tensor, torch.Tensor]:
    """reshape and quant qkv local heads tensor after all2all communication"""
    # reshape and do quant
    recv_tensor = reshape_qkv_tensors_after_cp_a2a(recv_tensor, original_shape, cp_size, seq_dim)
    recv_tensor, recv_tensor_scale = block_scaling_node(recv_tensor, use_fp8=use_fp8)
    return recv_tensor, recv_tensor_scale


def reshape_output_tensor_before_cp_a2a(
    output_local_heads: torch.Tensor, recv_tensor: torch.Tensor, cp_size, seq_dim
) -> torch.Tensor:
    """reshape output local heads tensor before all2all communication"""
    # [b, s, h, d] -> [b, s, h // n, d] -> [b, n, s // n, h // n, d] -> [n, b, s // n, h // n, d]
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
    recv_tensor = recv_tensor.view(*output_local_heads_a2a.shape)
    return output_local_heads_a2a, recv_tensor


def reshape_output_tensor_after_cp_a2a(output_local_tokens: torch.Tensor) -> torch.Tensor:
    """reshape output local tokens tensor after all2all communication"""
    # [n, b, s // n, h // n, d] -> [b, s // n, n, h // n, d]
    output_local_tokens = output_local_tokens.movedim(0, -3).contiguous()
    # [b, s // n, n, h // n, d] -> [b, s // n, h, d]
    output_local_tokens = output_local_tokens.view(
        *output_local_tokens.shape[:-3],
        output_local_tokens.shape[-3] * output_local_tokens.shape[-2],
        output_local_tokens.shape[-1],
    )
    return output_local_tokens


def reshape_output_grad_after_cp_a2a(dout_local_heads: torch.Tensor, seq_dim) -> torch.Tensor:
    """reshape output grad tensor after all2all communication"""
    # [n, b, s // n, h // n, d] -> [b, n, s // n, h // n, d]
    dout_local_heads = dout_local_heads.movedim(0, seq_dim).contiguous()
    # [b, n, s // n, h // n, d] -> [b, s, h // n, d]
    dout_local_heads = dout_local_heads.view(
        *dout_local_heads.shape[:seq_dim],
        dout_local_heads.shape[seq_dim] * dout_local_heads.shape[seq_dim + 1],
        *dout_local_heads.shape[seq_dim + 2 :],
    )

    return dout_local_heads


def reshape_qkv_grad_before_cp_a2a(
    send_tensor: torch.Tensor, recv_tensor: torch.Tensor, cp_size, seq_dim
) -> Tuple[torch.Tensor, torch.Tensor]:
    """reshape qkv grad tensors before all2all communication"""
    # [b, s, h // n, d] -> [b, n, s // n, h // n, d] -> [n, b, s // n, h // n, d]
    send_tensor = (
        send_tensor.view(
            send_tensor.shape[0], cp_size, send_tensor.shape[1] // cp_size, *send_tensor.shape[2:]
        )
        .movedim(seq_dim, 0)
        .contiguous()
    )
    recv_tensor = recv_tensor.view(*send_tensor.shape)
    return send_tensor, recv_tensor


def reshape_qkv_grad_after_cp_a2a(recv_tensor: torch.Tensor, original_shape) -> torch.Tensor:
    """reshape qkv grad tensors after all2all communication"""
    # [n, b, s // n, h // n, d] -> [b, s // n, n, h // n, d] -> [b, s // n, h, d]
    recv_tensor = recv_tensor.movedim(0, -3).contiguous()
    recv_tensor = recv_tensor.view(original_shape)
    return recv_tensor


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

        before_all2all_funcs = [functools.partial(reshape_qkv_tensors_before_cp_a2a, cp_size=cp_size)] * 3
        after_all2all_funcs = [
            functools.partial(
                reshape_and_quant_qkv_local_heads_after_cp_a2a,
                original_shape=q_shape,
                cp_size=cp_size,
                seq_dim=seq_dim,
                use_fp8=use_fp8,
            ),
            functools.partial(
                reshape_and_quant_qkv_local_heads_after_cp_a2a,
                original_shape=k_shape,
                cp_size=cp_size,
                seq_dim=seq_dim,
                use_fp8=use_fp8,
            ),
            functools.partial(
                reshape_qkv_tensors_after_cp_a2a, original_shape=v_shape, cp_size=cp_size, seq_dim=seq_dim
            ),  # no quant for v
        ]

        q_local_heads_bundle, k_local_heads_bundle, v_local_heads = (
            attention_communicator.data_exchange_over_cp_groups(
                [q, k, v], before_all2all_funcs=before_all2all_funcs, after_all2all_funcs=after_all2all_funcs
            )
        )

        q_local_heads, q_scale = q_local_heads_bundle
        k_local_heads, k_scale = k_local_heads_bundle

        # cal v_scale and p_scale
        v_local_heads, v_scale, p_scale = quant_v_get_p_scale(v_local_heads, use_fp8)

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

        (output_local_tokens,) = attention_communicator.data_exchange_over_cp_groups(
            [output_local_heads],
            before_all2all_funcs=[
                functools.partial(reshape_output_tensor_before_cp_a2a, cp_size=cp_size, seq_dim=seq_dim)
            ],
            after_all2all_funcs=[reshape_output_tensor_after_cp_a2a],
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
            ctx.max_seqlens_q = q_local_heads.shape[1]
            ctx.max_seqlens_k = k_local_heads.shape[1]
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
        cp_size = ctx.cp_group.size()
        seq_dim = ctx.seq_dim

        # [b, s // n, h, d] -> [b, s // n, n , h // n, d] -> [n, b, s // n, h // n, d]
        dout = (
            dout.view(*dout.shape[:-2], cp_size, dout.shape[-2] // cp_size, dout.shape[-1])
            .movedim(-3, 0)
            .contiguous()
        )

        after_all2all_funcs = [functools.partial(reshape_output_grad_after_cp_a2a, seq_dim=seq_dim)]

        (dout_local_heads,) = attention_communicator.data_exchange_over_cp_groups(
            [dout], before_all2all_funcs=None, after_all2all_funcs=after_all2all_funcs
        )

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
        before_all2all_funcs = [
            functools.partial(reshape_qkv_grad_before_cp_a2a, cp_size=cp_size, seq_dim=seq_dim)
        ] * 3
        after_all2all_funcs = [
            functools.partial(reshape_qkv_grad_after_cp_a2a, original_shape=x_shape)
            for x_shape in [ctx.q_shape, ctx.k_shape, ctx.v_shape]
        ]
        # all2all d_{q/k/v}
        dq_local_tokens, dk_local_tokens, dv_local_tokens = (
            attention_communicator.data_exchange_over_cp_groups(
                [dq_local_heads, dk_local_heads, dv_local_heads],
                before_all2all_funcs=before_all2all_funcs,
                after_all2all_funcs=after_all2all_funcs,
            )
        )

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


def dispatch_attention_cp_functions(
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
    backend_type,
    fp8,
    cp_group,
    cp_comm_type,
):
    if backend_type == "triton":
        if cp_comm_type == "a2a":
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
            raise NotImplementedError(
                f"not supported backend_type {backend_type} cp_comm_type {cp_comm_type} yet"
            )
    elif backend_type == "ck":
        if cp_comm_type == "a2a":
            pass
        else:
            raise NotImplementedError(
                f"not supported backend_type {backend_type} cp_comm_type {cp_comm_type} yet"
            )
    else:
        raise NotImplementedError(
            f"not supported backend_type {backend_type} cp_comm_type {cp_comm_type} yet"
        )
