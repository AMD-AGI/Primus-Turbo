from typing import Optional

import torch

from primus_turbo.triton.attention.attention_kernel import (
    attention_block_backward_triton_impl,
    attention_block_forward_triton_impl,
)


def attention_triton_forward_impl(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    p_scale: float,
    q_scale: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    dropout_p: float,
    softmax_scale: float,
    causal: bool,
    window_size_left: int,
    window_size_right: int,
    bias: Optional[torch.Tensor],
    alibi_slopes: Optional[torch.Tensor],
    return_softmax: bool,
    use_fp8,
):

    assert (
        window_size_left == -1 and window_size_right == -1
    ), "in triton attn kernel, window_size_left and window_size_right must be -1."

    output, softmax_lse, exp_scores = attention_block_forward_triton_impl(
        q,
        k,
        v,
        p_scale,
        q_scale,
        k_scale,
        v_scale,
        softmax_scale,
        alibi_slopes,
        causal,
        bias,
        dropout_p,
        "bshd",
        0,
        0,
        q.shape[1],
        k.shape[1],
        return_softmax,
        True,
        use_fp8,
    )

    return output, softmax_lse, exp_scores


# q k v should be dtype=torch.bf16
def attention_triton_backward_impl(
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    q_scale: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    p_scale: float,
    softmax_lse: torch.Tensor,
    dq: Optional[torch.Tensor],
    dk: Optional[torch.Tensor],
    dv: Optional[torch.Tensor],
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    softmax_scale: float,
    causal: bool,
    window_size_left: int,
    window_size_right: int,
    alibi_slopes: Optional[torch.Tensor],
    use_fp8,
):
    assert (
        window_size_left == -1 and window_size_right == -1
    ), "in triton attn kernel, window_size_left and window_size_right must be -1."

    dq, dk, dv = attention_block_backward_triton_impl(
        dout,
        q,
        k,
        v,
        out,
        q_scale,
        k_scale,
        v_scale,
        p_scale,
        softmax_lse,
        dq,
        dk,
        dv,
        softmax_scale,
        alibi_slopes,
        causal,
        "bshd",
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        True,
        use_fp8,
    )
    return dq, dk, dv
