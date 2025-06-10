import torch
from aiter.ops.triton.mha import _flash_attn_forward, _flash_attn_backward, cast_to_fp8
from typing import Optional
from primus_turbo.triton.attention.attention_kernel import (
    attention_block_forward_triton_impl,
    attention_block_backward_triton_impl,
    get_f8_fwd_dtype,
    block_scaling_node,
    FIXED_BLOCK_M,
    FIXED_BLOCK_N,
)


def attention_triton_forward_impl(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dropout_p: float,
    softmax_scale: float,
    causal: bool,
    window_size_left: int,
    window_size_right: int,
    bias: Optional[torch.Tensor],
    alibi_slopes: Optional[torch.Tensor],
    return_lse: bool,
    return_softmax: bool,
    use_fp8,
):

    assert (
        window_size_left == -1 and window_size_right == -1
    ), "in triton attn kernel, window_size_left and window_size_right must be -1."

    (q_scale, k_scale, v_scale) = None
    if use_fp8:
        # online quant
        range_v = torch.max(torch.abs(v))
        float8_fw = get_f8_fwd_dtype()
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
        "bhsd",
        0,
        0,
        q.shape[1],
        k.shape[1],
        return_softmax,
        return_lse,
        True,
        use_fp8,
    )

    return output, softmax_lse, exp_scores, q_scale, k_scale, v_scale, p_scale


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
) -> torch.Tensor:
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
        "bhsd",
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        True,
        use_fp8,
    )
