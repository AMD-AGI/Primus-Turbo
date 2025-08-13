###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from functools import lru_cache
from typing import Optional, Tuple

import torch

from primus_turbo.pytorch.kernels.attention.attention_csrc_impl import (
    attention_aiter_csrc_backward_impl,
    attention_aiter_csrc_forward_impl,
)
from primus_turbo.pytorch.kernels.attention.attention_triton_impl import (
    attention_triton_backward_impl,
    attention_triton_forward_impl,
)
from primus_turbo.pytorch.ops.utils.attention_utils import (
    block_scaling_node,
    quant_v_get_p_scale,
)


@lru_cache
def get_attention_cp_a2a_helper(b, s, h_q, h_kv, d_qk, d_v, seq_dim, n):
    attn_helper = AttentionCPA2AHelper(b, s, h_q, h_kv, d_qk, d_v, seq_dim, n)
    return attn_helper


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
        assert bias is None

        n = cp_group.size()
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
        torch.distributed.all_to_all_single(qkv_out, qkv, group=cp_group, async_op=False)
        q_local_heads, k_local_heads, v_local_heads = attn_helper.splits_qkv_after_a2a(qkv_out)

        q_local_heads, q_scale = block_scaling_node(q_local_heads, use_fp8=use_fp8)
        k_local_heads, k_scale = block_scaling_node(k_local_heads, use_fp8=use_fp8)
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
            ctx.attn_helper = attn_helper
            ctx.seq_dim = seq_dim
            ctx.cp_group = cp_group

        output_local_heads = attn_helper.reshape_o_before_a2a(output_local_heads)
        output_local_tokens = torch.empty_like(output_local_heads)
        torch.distributed.all_to_all_single(
            output_local_tokens, output_local_heads, group=cp_group, async_op=False
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
            q_scale,
            k_scale,
            v_scale,
        ) = ctx.saved_tensors
        assert bias is None, "Currently bias is not supported by fa backward function."
        assert dout.dtype is torch.bfloat16, f"dout should be bfloat16 but get {dout.dtype}"
        attn_helper = ctx.attn_helper

        dout = attn_helper.reshape_do_before_a2a(dout)
        dout_local_heads = torch.empty_like(dout)
        torch.distributed.all_to_all_single(dout_local_heads, dout, group=ctx.cp_group)
        dout_local_heads = attn_helper.reshape_do_after_a2a(dout_local_heads)

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

        dqkv = attn_helper.combine_dqkv_before_a2a(dq_local_heads, dk_local_heads, dv_local_heads)
        dqkv_out = torch.empty_like(dqkv)
        torch.distributed.all_to_all_single(dqkv_out, dqkv, group=ctx.cp_group)
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
        cp_group,
        is_v3_atomic_fp32: Optional[bool] = True,
        how_v3_bf16_cvt: Optional[int] = 1,
    ):
        assert bias is None
        is_grad = is_grad_enabled and any(x.requires_grad for x in [q, k, v])

        n = cp_group.size()
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
        torch.distributed.all_to_all_single(qkv_out, qkv, group=cp_group, async_op=False)
        q_local_heads, k_local_heads, v_local_heads = attn_helper.splits_qkv_after_a2a(qkv_out)

        head_size_q_og = q_local_heads.size(3)
        head_size_v_og = v_local_heads.size(3)
        # todo fix if head_size_v_og!=head_size_q_og, no padding
        if head_size_q_og != head_size_v_og:
            v_local_heads = torch.nn.functional.pad(v_local_heads, [0, head_size_q_og - head_size_v_og])
        if head_size_q_og % 8 != 0:
            q_local_heads = torch.nn.functional.pad(q_local_heads, [0, 8 - head_size_q_og % 8])
            k_local_heads = torch.nn.functional.pad(k_local_heads, [0, 8 - head_size_q_og % 8])
        if head_size_v_og % 8 != 0:
            v_local_heads = torch.nn.functional.pad(v_local_heads, [0, 8 - head_size_v_og % 8])

        out_padded, softmax_lse, S_dmask, rng_state = attention_aiter_csrc_forward_impl(
            q_local_heads,
            k_local_heads,
            v_local_heads,
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
            ctx.head_size_q_og = head_size_q_og
            ctx.is_v3_atomic_fp32 = is_v3_atomic_fp32
            ctx.how_v3_bf16_cvt = how_v3_bf16_cvt
            ctx.attn_helper = attn_helper
            ctx.cp_group = cp_group
            ctx.seq_dim = seq_dim

        output_local_heads = out_padded[..., :head_size_v_og]
        output_local_heads = attn_helper.reshape_o_before_a2a(output_local_heads)
        output_local_tokens = torch.empty_like(output_local_heads)
        torch.distributed.all_to_all_single(
            output_local_tokens, output_local_heads, group=cp_group, async_op=False
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
        torch.distributed.all_to_all_single(dout_local_heads, dout, group=ctx.cp_group)

        dout_local_heads = attn_helper.reshape_do_after_a2a(dout_local_heads)

        dq, dk, dv = (
            torch.zeros_like(q_local_heads),
            torch.empty_like(k_local_heads),
            torch.empty_like(v_local_heads),
        )
        dbias = None

        head_size_q_og = ctx.head_size_q_og
        head_size_v_og = dout_local_heads.size(3)
        dout_padded = dout_local_heads
        if head_size_v_og % 8 != 0:
            dout_padded = torch.nn.functional.pad(dout_local_heads, [0, 8 - head_size_v_og % 8])
        if head_size_q_og != head_size_v_og:
            dout_padded = torch.nn.functional.pad(dout_local_heads, [0, head_size_q_og - head_size_v_og])

        attention_aiter_csrc_backward_impl(
            dout_padded,
            q_local_heads,
            k_local_heads,
            v_local_heads,
            output_padded,
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

        dqkv = attn_helper.combine_dqkv_before_a2a(dq, dk, dv)
        dqkv_out = torch.empty_like(dqkv)
        torch.distributed.all_to_all_single(dqkv_out, dqkv, group=ctx.cp_group)
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
        )
