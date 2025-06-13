from typing import Optional, Tuple

import torch

_torch_custom_op_wrapper = torch.library.custom_op

from primus_turbo.triton.attention.attention_kernel import (
    attention_block_backward_triton_impl,
    attention_block_forward_triton_impl,
)


@_torch_custom_op_wrapper("primus_turbo::attention_triton_forward_impl", mutates_args=(), device_types="cuda")
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
    use_fp8: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

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


@attention_triton_forward_impl.register_fake
def _attention_triton_forward_impl_fake(
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
    use_fp8: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    o_shape = list(q.shape)
    o_shape[-1] = v.shape[-1]  # output shape should match v's head dim
    o = torch.empty(
        o_shape,
        device=q.device,
        dtype=torch.bfloat16 if use_fp8 else q.dtype,
        requires_grad=True,
    )

    batch_q, max_seqlen_q, nheads_q, head_size_q = q.shape
    batch_k, max_seqlen_k, nheads_k, head_size_k = k.shape

    if return_softmax:
        exp_scores = torch.zeros(
            (batch_q, nheads_q, max_seqlen_q, max_seqlen_k), device=q.device, dtype=torch.float32
        )
    else:
        exp_scores = torch.empty([], device=q.device, dtype=torch.float32)

    softmax_lse = torch.empty((batch_q, nheads_q, max_seqlen_q * 2), device=q.device, dtype=torch.float32)

    return o, softmax_lse, exp_scores


@_torch_custom_op_wrapper(
    "primus_turbo::attention_triton_backward_impl", mutates_args=("dq", "dk", "dv"), device_types="cuda"
)
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
    use_fp8: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    assert (
        window_size_left == -1 and window_size_right == -1
    ), "in triton attn kernel, window_size_left and window_size_right must be -1."

    # 调用attention_block_backward_triton_impl函数，计算dq、dk、dv
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
    # 返回dq、dk、dv
    return dq, dk, dv


@attention_triton_backward_impl.register_fake
def _attention_triton_backward_impl_fake(
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
    use_fp8: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dq_out, dk_out, dv_out = (
        torch.empty_like(q, dtype=torch.bfloat16),
        torch.empty_like(k, dtype=torch.bfloat16),
        torch.empty_like(v, dtype=torch.bfloat16),
    )
    return dq_out, dk_out, dv_out
