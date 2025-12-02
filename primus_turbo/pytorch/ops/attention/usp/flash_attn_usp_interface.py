from functools import lru_cache
from typing import Optional, Tuple

import torch

from primus_turbo.pytorch.kernels.attention.attention_csrc_impl import (  # attention_aiter_csrc_backward_impl,
    attention_aiter_csrc_backward_impl,
    attention_aiter_csrc_forward_impl,
)

from .attention_ring import ring_attn_bwd, ring_attn_fwd


class AttentionCPA2AHelper:
    """AttentionCPA2AHelper: a helper to transpose tensor for CP A2A"""

    def __init__(self, b, s, h_q, h_kv, d_qk, d_v, seq_dim, n):
        assert seq_dim == 1, "only_support bshd yet"
        self.seq_dim = seq_dim

        self.qkv_shape_traits = ((n, b, s, h_q, d_qk), (n, b, s, h_kv, d_qk), (n, b, s, h_kv, d_v))

        self.o_shape_traits = (n, b, s, h_q, d_v)

        self.combine_splits = (
            b * s * h_q * d_qk // n // n,
            b * s * h_kv * d_qk // n // n,
            b * s * h_kv * d_v // n // n,
        )

    def combine_qkv_before_a2a(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Combine and reshape qkv before all2all

        Args:
            q (torch.Tensor): query tensor (b, s // n, h_q, d_qk)
            k (torch.Tensor): key tensor (b, s // n, h_kv, d_qk)
            v (torch.Tensor): value tensor (b, s // n, h_kv, d_v)

        Returns:
            qkv (torch.Tensor): qkv combined tensor (n, -1)
        """
        # [b, s // n, h, d] -> [b, s // n, n, h // n, d] -> [n, b, s // n, h // n, d] -> [n, -1]
        q, k, v = (
            x.view(b, s // n, n, h // n, d).movedim(-3, 0).contiguous().view(n, -1)
            for x, (n, b, s, h, d) in zip((q, k, v), self.qkv_shape_traits)
        )

        qkv = torch.cat((q, k, v), dim=1).contiguous()
        return qkv

    def splits_qkv_after_a2a(self, qkv: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Split and reshape qkv before all2all

        Args:
            qkv (torch.Tensor): qkv tensor of local heads (n, -1)

        Returns:
            q_local_heads, k_local_heads, v_local_heads (Tuple[torch.Tensor, torch.Tensor, torch.Tensor]): (b, s, h // n, d)
        """
        q, k, v = torch.split(qkv, self.combine_splits, dim=1)
        # [n, b, s // n, h // n, d] -> [b, n, s // n, h // n, d] -> [b, s, h // n, d]
        q, k, v = (
            x.view(n, b, s // n, h // n, d).movedim(0, 1).contiguous().view(b, s, h // n, d)
            for x, (n, b, s, h, d) in zip((q, k, v), self.qkv_shape_traits)
        )
        return q, k, v

    def reshape_o_before_a2a(self, o: torch.Tensor) -> torch.Tensor:
        """Reshape output before all2all

        Args:
            o (torch.Tensor): output of local heads (b, s, h // n, d)

        Returns:
            o_reshaped (torch.Tensor): (n, b, s // n, h // n, d)
        """

        # [b, s, h // n, d] -> [b, n, s // n, h // n, d] -> [n, b, s // n, h // n, d]
        n, b, s, h, d = self.o_shape_traits
        o = o.view(b, n, s // n, h // n, d).movedim(1, 0).contiguous()
        return o

    def reshape_o_after_a2a(self, o: torch.Tensor) -> torch.Tensor:
        """Reshape output after all2all

        Args:
            o (torch.Tensor): output of local seq (n, b, s // n, h // n, d)

        Returns:
            o_reshaped (torch.Tensor): (b, s // n, h, d)
        """
        n, b, s, h, d = self.o_shape_traits
        # [n, b, s // n, h // n, d] -> [b, s // n, n, h // n, d] -> [b, s // n, h, d]
        o = o.movedim(0, -3).contiguous().view(b, s // n, h, d)

        return o

    def reshape_do_before_a2a(self, d_o: torch.Tensor) -> torch.Tensor:
        """Reshape output grad before all2all

        Args:
            d_o (torch.Tensor): output grad of local seq (b, s // n, h, d)

        Returns:
            d_o_reshaped torch.Tensor: (n, b, s // n, h // n, d)
        """
        # [b, s // n, h, d] -> [b, s // n, n , h // n, d] -> [n, b, s // n, h // n, d]
        n, b, s, h, d = self.o_shape_traits
        d_o = d_o.view(b, s // n, n, h // n, d).movedim(-3, 0).contiguous()
        return d_o

    def reshape_do_after_a2a(self, d_o: torch.Tensor) -> torch.Tensor:
        """Reshape output grad after all2all

        Args:
            d_o (torch.Tensor): output grad of local head (n, b, s // n, h // n, d)

        Returns:
            d_o_reshaped torch.Tensor: (b, s, h // n, d)
        """
        # [n, b, s // n, h // n, d] -> [b, n, s // n, h // n, d] -> [b, s, h // n, d]
        n, b, s, h, d = self.o_shape_traits
        d_o = d_o.movedim(0, 1).contiguous().view(b, s, h // n, d)
        return d_o

    def combine_dqkv_before_a2a(self, dq: torch.Tensor, dk: torch.Tensor, dv: torch.Tensor) -> torch.Tensor:
        """Combine qkv tensor of local heads before a2a

        Args:
            dq (torch.Tensor): dq local heads (b, s, h // n, d)
            dk (torch.Tensor): dk local heads (b, s, h // n, d)
            dv (torch.Tensor): dv local heads (b, s, h // n, d)

        Returns:
            d_qkv torch.Tensor: dqkv of local heads (n, -1)
        """

        # [b, s, h // n, d] -> [b, n, s // n, h // n, d] -> [n, b, s // n, h // n, d] -> [n, -1]
        dq, dk, dv = (
            x.view(b, n, s // n, h // n, d).movedim(1, 0).contiguous().view(n, -1)
            for x, (n, b, s, h, d) in zip((dq, dk, dv), self.qkv_shape_traits)
        )
        dqkv = torch.cat((dq, dk, dv), dim=1).contiguous()

        return dqkv

    def split_dqkv_after_a2a(self, dqkv: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Combine qkv tensor of local seq after a2a

        Args:
            dqkv (torch.Tensor): dqkv of local seq (n, -1)

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: dq, dk, dv of local seq (b, s // n, h, d)
        """
        # [n, b, s // n, h // n, d] -> [b, s // n, n, h // n, d] -> [b, s // n, h, d]
        dq, dk, dv = torch.split(dqkv, self.combine_splits, dim=1)
        dq, dk, dv = (
            x.view(n, b, s // n, h // n, d).movedim(0, -3).contiguous().view(b, s // n, h, d)
            for x, (n, b, s, h, d) in zip((dq, dk, dv), self.qkv_shape_traits)
        )
        return dq, dk, dv


@lru_cache
def get_attention_cp_a2a_helper(b, s, h_q, h_kv, d_qk, d_v, seq_dim, n):
    attn_helper = AttentionCPA2AHelper(b, s, h_q, h_kv, d_qk, d_v, seq_dim, n)
    return attn_helper


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
            attention_aiter_csrc_forward_impl,
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
        attn_helper: AttentionCPA2AHelper = ctx.attn_helper

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
            attention_aiter_csrc_backward_impl,
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
    deterministic=True,
    return_lse=False,
    return_attn_probs=False,
    ulysses_group=None,
    ring_group=None,
):
    assert ulysses_group and ring_group
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
