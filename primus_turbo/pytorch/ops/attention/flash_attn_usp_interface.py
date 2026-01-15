###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from typing import Optional, Tuple

import torch
import torch.distributed as dist
from aiter.ops.mha import _flash_attn_backward as _aiter_flash_attn_backward
from aiter.ops.mha import _flash_attn_forward as _aiter_flash_attn_forward

from primus_turbo.pytorch.core.low_precision import (
    Float8QuantConfig,
    ScalingGranularity,
)
from primus_turbo.pytorch.kernels.attention.attention_triton_impl import (
    attention_triton_backward_impl,
    attention_triton_forward_impl,
)
from primus_turbo.pytorch.ops.attention.attention_utils import (
    block_scaling_node,
    get_p_scale,
)

from .usp.attention_a2a_helper import get_attention_cp_a2a_helper
from .usp.attention_ring import ring_attn_bwd, ring_attn_fwd


def _aiter_csrc_attn_fwd(
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
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Adapter for ring attention forward using aiter csrc."""
    return _aiter_flash_attn_forward(
        q,
        k,
        v,
        dropout_p,
        softmax_scale,
        causal,
        window_size_left,
        window_size_right,
        0,  # sink_size
        bias,
        alibi_slopes,
        None,  # q_descale
        None,  # k_descale
        None,  # v_descale
        return_lse,
        return_softmax,
    )


class AttentionCKFunctionCPA2A(torch.autograd.Function):
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
        deterministic,
        return_lse,
        return_softmax,
        is_grad_enabled,
        ulysses_group,
        ring_group,
        is_v3_atomic_fp32: Optional[bool] = True,
        how_v3_bf16_cvt: Optional[int] = 1,
    ):
        assert bias is None
        is_grad = is_grad_enabled and any(x.requires_grad for x in [q, k, v])

        n = ulysses_group.size()
        b, s, h_q, d_qk = q.shape
        _, _, h_kv, d_v = v.shape
        s = s * n
        assert h_q % n == 0
        assert h_kv % n == 0
        # bshd only
        seq_dim = 1
        attn_helper = get_attention_cp_a2a_helper(b, s, h_q, h_kv, d_qk, d_v, seq_dim, n)

        qkv = attn_helper.combine_qkv_before_a2a(q, k, v)
        qkv_out = torch.empty_like(qkv)
        torch.distributed.all_to_all_single(qkv_out, qkv, group=ulysses_group, async_op=False)
        q_local_heads, k_local_heads, v_local_heads = attn_helper.splits_qkv_after_a2a(qkv_out)

        if d_qk % 8 != 0:
            q_local_heads = torch.nn.functional.pad(q_local_heads, [0, 8 - d_qk % 8])
            k_local_heads = torch.nn.functional.pad(k_local_heads, [0, 8 - d_qk % 8])
        if d_v % 8 != 0:
            v_local_heads = torch.nn.functional.pad(v_local_heads, [0, 8 - d_v % 8])

        assert not return_softmax
        assert dropout_p == 0.0
        out_padded, softmax_lse, S_dmask, rng_state = ring_attn_fwd(
            ring_group,
            _aiter_csrc_attn_fwd,
            q_local_heads,
            k_local_heads,
            v_local_heads,
            softmax_scale=softmax_scale,
            dropout_p=dropout_p,
            causal=causal,
            window_size_left=window_size[0],
            window_size_right=window_size[1],
            bias=bias,
            alibi_slopes=alibi_slopes,
            return_lse=True,
            return_softmax=return_softmax and dropout_p > 0,
        )

        if is_grad:
            ctx.save_for_backward(
                q_local_heads, k_local_heads, v_local_heads, out_padded, softmax_lse, rng_state
            )
            ctx.dropout_p = dropout_p
            ctx.softmax_scale = softmax_scale
            ctx.causal = causal
            ctx.window_size = window_size
            ctx.bias = bias
            ctx.alibi_slopes = alibi_slopes
            ctx.deterministic = deterministic
            ctx.d_qk = d_qk
            ctx.is_v3_atomic_fp32 = is_v3_atomic_fp32
            ctx.how_v3_bf16_cvt = how_v3_bf16_cvt
            ctx.attn_helper = attn_helper
            ctx.ulysses_group = ulysses_group
            ctx.ring_group = ring_group
            ctx.seq_dim = seq_dim

        output_local_heads = out_padded[..., :d_v]
        output_local_heads = attn_helper.reshape_o_before_a2a(output_local_heads)
        output_local_tokens = torch.empty_like(output_local_heads)
        torch.distributed.all_to_all_single(
            output_local_tokens, output_local_heads, group=ulysses_group, async_op=False
        )
        output_local_tokens = attn_helper.reshape_o_after_a2a(output_local_tokens)

        result = [output_local_tokens]
        if return_lse:
            result.append(softmax_lse)
        if return_softmax:
            result.append(S_dmask)

        return result[0] if len(result) == 1 else tuple(result)

    @staticmethod
    def backward(ctx, dout, *args):
        (
            q_local_heads,
            k_local_heads,
            v_local_heads,
            output_padded,
            softmax_lse,
            rng_state,
        ) = ctx.saved_tensors
        attn_helper = ctx.attn_helper

        dout = attn_helper.reshape_do_before_a2a(dout)

        dout_local_heads = torch.empty_like(dout)
        torch.distributed.all_to_all_single(dout_local_heads, dout, group=ctx.ulysses_group)

        dout_local_heads = attn_helper.reshape_do_after_a2a(dout_local_heads)

        dbias = None

        d_qk = ctx.d_qk
        d_v = dout_local_heads.size(3)
        dout_padded = dout_local_heads
        if d_v % 8 != 0:
            dout_padded = torch.nn.functional.pad(dout_local_heads, [0, 8 - d_v % 8])
        if d_qk != d_v:
            v_local_heads = torch.nn.functional.pad(v_local_heads, [0, d_qk - d_v])
            output_padded = torch.nn.functional.pad(output_padded, [0, d_qk - d_v])
            dout_padded = torch.nn.functional.pad(dout_local_heads, [0, d_qk - d_v])

        dq, dk, dv = ring_attn_bwd(
            ctx.ring_group,
            _aiter_flash_attn_backward,
            dout_padded,
            q_local_heads,
            k_local_heads,
            v_local_heads,
            output_padded,
            softmax_lse,
            dbias=dbias,
            dropout_p=ctx.dropout_p,
            softmax_scale=ctx.softmax_scale,
            causal=ctx.causal,
            window_size_left=ctx.window_size[0],
            window_size_right=ctx.window_size[1],
            bias=ctx.bias,
            alibi_slopes=ctx.alibi_slopes,
            deterministic=ctx.deterministic,
            rng_state=rng_state,
            is_v3_atomic_fp32=ctx.is_v3_atomic_fp32,
            how_v3_bf16_cvt=ctx.how_v3_bf16_cvt,
        )

        dq = dq[..., :d_qk]  # We could have padded the head dimension
        dk = dk[..., :d_qk]
        dv = dv[..., :d_v]

        dqkv = attn_helper.combine_dqkv_before_a2a(dq, dk, dv)
        dqkv_out = torch.empty_like(dqkv)
        torch.distributed.all_to_all_single(dqkv_out, dqkv, group=ctx.ulysses_group)
        dq_local_tokens, dk_local_tokens, dv_local_tokens = attn_helper.split_dqkv_after_a2a(dqkv_out)

        return (
            dq_local_tokens,
            dk_local_tokens,
            dv_local_tokens,
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
            None,
            None,
            None,
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
        ulysses_group,
        ring_group,
    ):
        assert bias is None
        assert (
            dist.get_world_size(ring_group) == 1
        ), "currently ring attention not supported for fp8, since triton implementation does not use the standard flash attention interface"

        n = ulysses_group.size()
        b, s, h_q, d_qk = q.shape
        _, _, h_kv, d_v = v.shape
        s = s * n
        assert h_q % n == 0
        assert h_kv % n == 0
        # bshd only
        seq_dim = 1
        attn_helper = get_attention_cp_a2a_helper(b, s, h_q, h_kv, d_qk, d_v, seq_dim, n)

        qkv = attn_helper.combine_qkv_before_a2a(q, k, v)
        qkv_out = torch.empty_like(qkv)
        torch.distributed.all_to_all_single(qkv_out, qkv, group=ulysses_group, async_op=False)
        q_local_heads, k_local_heads, v_local_heads = attn_helper.splits_qkv_after_a2a(qkv_out)

        q_local_heads, q_descale = block_scaling_node(q_local_heads, use_fp8=use_fp8)
        k_local_heads, k_descale = block_scaling_node(k_local_heads, use_fp8=use_fp8)
        v_local_heads, v_descale = block_scaling_node(v_local_heads, use_fp8=use_fp8)
        p_scale = get_p_scale(use_fp8)

        output_local_heads, softmax_lse, exp_scores = attention_triton_forward_impl(
            q_local_heads,
            k_local_heads,
            v_local_heads,
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
                q_descale,
                k_descale,
                v_descale,
            )
            ctx.sm_scale = softmax_scale
            ctx.p_scale = p_scale
            ctx.causal = causal
            ctx.use_fp8 = use_fp8
            ctx.cu_seqlens_q = torch.tensor(0, device="cuda")
            ctx.cu_seqlens_k = torch.tensor(0, device="cuda")
            ctx.max_seqlens_q = q_local_heads.shape[1]
            ctx.max_seqlens_k = k_local_heads.shape[1]
            ctx.attn_helper = attn_helper
            ctx.seq_dim = seq_dim
            ctx.ulysses_group = ulysses_group

        output_local_heads = attn_helper.reshape_o_before_a2a(output_local_heads)
        output_local_tokens = torch.empty_like(output_local_heads)
        torch.distributed.all_to_all_single(
            output_local_tokens, output_local_heads, group=ulysses_group, async_op=False
        )
        output_local_tokens = attn_helper.reshape_o_after_a2a(output_local_tokens)

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
            q_descale,
            k_descale,
            v_descale,
        ) = ctx.saved_tensors
        assert bias is None, "Currently bias is not supported by fa backward function."
        assert dout.dtype is torch.bfloat16, f"dout should be bfloat16 but get {dout.dtype}"
        attn_helper = ctx.attn_helper

        dout = attn_helper.reshape_do_before_a2a(dout)
        dout_local_heads = torch.empty_like(dout)
        torch.distributed.all_to_all_single(dout_local_heads, dout, group=ctx.ulysses_group)
        dout_local_heads = attn_helper.reshape_do_after_a2a(dout_local_heads)

        dq_local_heads, dk_local_heads, dv_local_heads = attention_triton_backward_impl(
            dout_local_heads,
            q_local_heads,
            k_local_heads,
            v_local_heads,
            output_local_heads,
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

        dqkv = attn_helper.combine_dqkv_before_a2a(dq_local_heads, dk_local_heads, dv_local_heads)
        dqkv_out = torch.empty_like(dqkv)
        torch.distributed.all_to_all_single(dqkv_out, dqkv, group=ctx.ulysses_group)
        dq_local_tokens, dk_local_tokens, dv_local_tokens = attn_helper.split_dqkv_after_a2a(dqkv_out)

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
            None,
        )


def flash_attn_usp_func(
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
    ulysses_group=None,
    ring_group=None,
):
    assert (
        ulysses_group and ring_group
    ), "ulysses_group and ring_group must be set while calling usp attention"
    return AttentionCKFunctionCPA2A.apply(
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
        ulysses_group,
        ring_group,
    )


def flash_attn_fp8_usp_func(
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
    ulysses_group=None,
    ring_group=None,
):
    assert (
        ulysses_group and ring_group
    ), "ulysses_group and ring_group must be set while calling usp attention"

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
        torch.is_grad_enabled(),
        True,
        ulysses_group,
        ring_group,
    )
