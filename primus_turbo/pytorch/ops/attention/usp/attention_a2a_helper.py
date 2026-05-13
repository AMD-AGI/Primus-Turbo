###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from functools import lru_cache
from typing import Tuple

import torch


class AttentionCPA2AHelper:
    """AttentionCPA2AHelper: a helper to transpose tensor for CP A2A"""

    def __init__(self, b, s, h_q, h_kv, d_qk, d_v, seq_dim, n):
        # seq_dim semantics:
        #   0 -> sbhd (input local layout [s/n, b, h, d])
        #   1 -> bshd (input local layout [b, s/n, h, d])
        #   2 -> bhsd (input local layout [b, h, s/n, d])
        assert seq_dim in (0, 1, 2), f"unsupported seq_dim: {seq_dim}"
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
            q (torch.Tensor): query tensor of local seq with full heads
                seq_dim==1: (b, s // n, h_q,    d_qk)
                seq_dim==0: (s // n, b, h_q,    d_qk)
                seq_dim==2: (b, h_q,    s // n, d_qk)
            k (torch.Tensor): key tensor with the analogous layout
            v (torch.Tensor): value tensor with the analogous layout

        Returns:
            qkv (torch.Tensor): qkv combined tensor (n, -1)
        """
        if self.seq_dim == 1:
            # [b, s // n, h, d] -> [b, s // n, n, h // n, d] -> [n, b, s // n, h // n, d] -> [n, -1]
            q, k, v = (
                x.view(b, s // n, n, h // n, d).movedim(-3, 0).contiguous().view(n, -1)
                for x, (n, b, s, h, d) in zip((q, k, v), self.qkv_shape_traits)
            )
        elif self.seq_dim == 2:
            # [b, h, s // n, d] -> [b, n, h // n, s // n, d] -> [n, b, h // n, s // n, d] -> [n, -1]
            q, k, v = (
                x.view(b, n, h // n, s // n, d).movedim(1, 0).contiguous().view(n, -1)
                for x, (n, b, s, h, d) in zip((q, k, v), self.qkv_shape_traits)
            )
        else:
            # [s // n, b, h, d] -> [s // n, b, n, h // n, d] -> [n, s // n, b, h // n, d] -> [n, -1]
            q, k, v = (
                x.view(s // n, b, n, h // n, d).movedim(-3, 0).contiguous().view(n, -1)
                for x, (n, b, s, h, d) in zip((q, k, v), self.qkv_shape_traits)
            )

        qkv = torch.cat((q, k, v), dim=1).contiguous()
        return qkv

    def splits_qkv_after_a2a(self, qkv: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Split and reshape qkv after all2all

        Args:
            qkv (torch.Tensor): qkv tensor of local heads (n, -1)

        Returns:
            q_local_heads, k_local_heads, v_local_heads (Tuple[torch.Tensor, torch.Tensor, torch.Tensor]):
                seq_dim==1: (b, s,     h // n, d)
                seq_dim==0: (s, b,     h // n, d)
                seq_dim==2: (b, h // n, s,     d)
        """
        q, k, v = torch.split(qkv, self.combine_splits, dim=1)
        if self.seq_dim == 1:
            # [n, b, s // n, h // n, d] -> [b, n, s // n, h // n, d] -> [b, s, h // n, d]
            q, k, v = (
                x.view(n, b, s // n, h // n, d).movedim(0, 1).contiguous().view(b, s, h // n, d)
                for x, (n, b, s, h, d) in zip((q, k, v), self.qkv_shape_traits)
            )
        elif self.seq_dim == 2:
            # [n, b, h // n, s // n, d] -> [b, h // n, n, s // n, d] -> [b, h // n, s, d]
            q, k, v = (
                x.view(n, b, h // n, s // n, d).movedim(0, 2).contiguous().view(b, h // n, s, d)
                for x, (n, b, s, h, d) in zip((q, k, v), self.qkv_shape_traits)
            )
        else:
            # [n, s // n, b, h // n, d] -> [s, b, h // n, d]
            q, k, v = (
                x.view(n, s // n, b, h // n, d).contiguous().view(s, b, h // n, d)
                for x, (n, b, s, h, d) in zip((q, k, v), self.qkv_shape_traits)
            )
        return q, k, v

    def reshape_o_before_a2a(self, o: torch.Tensor) -> torch.Tensor:
        """Reshape output before all2all

        Args:
            o (torch.Tensor): output of local heads
                seq_dim==1: (b, s,     h // n, d)
                seq_dim==0: (s, b,     h // n, d)
                seq_dim==2: (b, h // n, s,     d)

        Returns:
            o_reshaped (torch.Tensor): (n, b, ..., ..., d) ready for all_to_all_single
        """

        n, b, s, h, d = self.o_shape_traits
        if self.seq_dim == 1:
            # [b, s, h // n, d] -> [b, n, s // n, h // n, d] -> [n, b, s // n, h // n, d]
            o = o.view(b, n, s // n, h // n, d).movedim(1, 0).contiguous()
        elif self.seq_dim == 2:
            # [b, h // n, s, d] -> [b, h // n, n, s // n, d] -> [n, b, h // n, s // n, d]
            # Use reshape (not view) because when d_v < d_qk, input may be non-contiguous
            # along the s/d boundary after `out_padded[..., :d_v].transpose(1, 2)`.
            o = o.reshape(b, h // n, n, s // n, d).movedim(2, 0).contiguous()
        else:
            # [s, b, h // n, d] -> [n, s // n, b, h // n, d]
            o = o.view(n, s // n, b, h // n, d).contiguous()
        return o

    def reshape_o_after_a2a(self, o: torch.Tensor) -> torch.Tensor:
        """Reshape output after all2all

        Args:
            o (torch.Tensor): output of local seq (n, b, ..., ..., d)

        Returns:
            o_reshaped (torch.Tensor):
                seq_dim==1: (b, s // n, h,     d)
                seq_dim==0: (s // n, b, h,     d)
                seq_dim==2: (b, h,     s // n, d)
        """
        n, b, s, h, d = self.o_shape_traits
        if self.seq_dim == 1:
            # [n, b, s // n, h // n, d] -> [b, s // n, n, h // n, d] -> [b, s // n, h, d]
            o = o.movedim(0, -3).contiguous().view(b, s // n, h, d)
        elif self.seq_dim == 2:
            # [n, b, h // n, s // n, d] -> [b, n, h // n, s // n, d] -> [b, h, s // n, d]
            o = o.movedim(0, 1).contiguous().view(b, h, s // n, d)
        else:
            # [n, s // n, b, h // n, d] -> [s // n, b, n, h // n, d] -> [s // n, b, h, d]
            o = o.movedim(0, 2).contiguous().view(s // n, b, h, d)

        return o

    def reshape_do_before_a2a(self, d_o: torch.Tensor) -> torch.Tensor:
        """Reshape output grad before all2all

        Args:
            d_o (torch.Tensor): output grad of local seq
                seq_dim==1: (b, s // n, h,     d)
                seq_dim==0: (s // n, b, h,     d)
                seq_dim==2: (b, h,     s // n, d)

        Returns:
            d_o_reshaped torch.Tensor: (n, b, ..., ..., d) ready for all_to_all_single
        """
        n, b, s, h, d = self.o_shape_traits
        if self.seq_dim == 1:
            # [b, s // n, h, d] -> [b, s // n, n , h // n, d] -> [n, b, s // n, h // n, d]
            d_o = d_o.view(b, s // n, n, h // n, d).movedim(-3, 0).contiguous()
        elif self.seq_dim == 2:
            # [b, h, s // n, d] -> [b, n, h // n, s // n, d] -> [n, b, h // n, s // n, d]
            d_o = d_o.view(b, n, h // n, s // n, d).movedim(1, 0).contiguous()
        else:
            # [s // n, b, h, d] -> [s // n, b, n, h // n, d] -> [n, s // n, b, h // n, d]
            d_o = d_o.view(s // n, b, n, h // n, d).movedim(-3, 0).contiguous()
        return d_o

    def reshape_do_after_a2a(self, d_o: torch.Tensor) -> torch.Tensor:
        """Reshape output grad after all2all

        Args:
            d_o (torch.Tensor): output grad of local head (n, b, ..., ..., d)

        Returns:
            d_o_reshaped torch.Tensor:
                seq_dim==1: (b, s,     h // n, d)
                seq_dim==0: (s, b,     h // n, d)
                seq_dim==2: (b, h // n, s,     d)
        """
        n, b, s, h, d = self.o_shape_traits
        if self.seq_dim == 1:
            # [n, b, s // n, h // n, d] -> [b, n, s // n, h // n, d] -> [b, s, h // n, d]
            d_o = d_o.movedim(0, 1).contiguous().view(b, s, h // n, d)
        elif self.seq_dim == 2:
            # [n, b, h // n, s // n, d] -> [b, h // n, n, s // n, d] -> [b, h // n, s, d]
            d_o = d_o.movedim(0, 2).contiguous().view(b, h // n, s, d)
        else:
            # [n, s // n, b, h // n, d] -> [s, b, h // n, d]
            d_o = d_o.contiguous().view(s, b, h // n, d)
        return d_o

    def combine_dqkv_before_a2a(self, dq: torch.Tensor, dk: torch.Tensor, dv: torch.Tensor) -> torch.Tensor:
        """Combine qkv tensor of local heads before a2a

        Args:
            dq (torch.Tensor): dq local heads
            dk (torch.Tensor): dk local heads
            dv (torch.Tensor): dv local heads
                seq_dim==1: (b, s,     h // n, d)
                seq_dim==0: (s, b,     h // n, d)
                seq_dim==2: (b, h // n, s,     d)

        Returns:
            d_qkv torch.Tensor: dqkv of local heads (n, -1)
        """

        if self.seq_dim == 1:
            # [b, s, h // n, d] -> [b, n, s // n, h // n, d] -> [n, b, s // n, h // n, d] -> [n, -1]
            dq, dk, dv = (
                x.view(b, n, s // n, h // n, d).movedim(1, 0).contiguous().view(n, -1)
                for x, (n, b, s, h, d) in zip((dq, dk, dv), self.qkv_shape_traits)
            )
        elif self.seq_dim == 2:
            # [b, h // n, s, d] -> [b, h // n, n, s // n, d] -> [n, b, h // n, s // n, d] -> [n, -1]
            # Use reshape (not view) for the first step because when d_v < d_qk, dv may be
            # non-contiguous along the s/d boundary after `dv[..., :d_v].transpose(1, 2)`.
            dq, dk, dv = (
                x.reshape(b, h // n, n, s // n, d).movedim(2, 0).contiguous().view(n, -1)
                for x, (n, b, s, h, d) in zip((dq, dk, dv), self.qkv_shape_traits)
            )
        else:
            # [s, b, h // n, d] -> [n, s // n, b, h // n, d] -> [n, -1]
            dq, dk, dv = (
                x.view(n, s // n, b, h // n, d).contiguous().view(n, -1)
                for x, (n, b, s, h, d) in zip((dq, dk, dv), self.qkv_shape_traits)
            )
        dqkv = torch.cat((dq, dk, dv), dim=1).contiguous()

        return dqkv

    def split_dqkv_after_a2a(self, dqkv: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Split qkv tensor of local seq after a2a

        Args:
            dqkv (torch.Tensor): dqkv of local seq (n, -1)

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: dq, dk, dv of local seq
                seq_dim==1: (b, s // n, h,     d)
                seq_dim==0: (s // n, b, h,     d)
                seq_dim==2: (b, h,     s // n, d)
        """
        dq, dk, dv = torch.split(dqkv, self.combine_splits, dim=1)
        if self.seq_dim == 1:
            # [n, b, s // n, h // n, d] -> [b, s // n, n, h // n, d] -> [b, s // n, h, d]
            dq, dk, dv = (
                x.view(n, b, s // n, h // n, d).movedim(0, -3).contiguous().view(b, s // n, h, d)
                for x, (n, b, s, h, d) in zip((dq, dk, dv), self.qkv_shape_traits)
            )
        elif self.seq_dim == 2:
            # [n, b, h // n, s // n, d] -> [b, n, h // n, s // n, d] -> [b, h, s // n, d]
            dq, dk, dv = (
                x.view(n, b, h // n, s // n, d).movedim(0, 1).contiguous().view(b, h, s // n, d)
                for x, (n, b, s, h, d) in zip((dq, dk, dv), self.qkv_shape_traits)
            )
        else:
            # [n, s // n, b, h // n, d] -> [s // n, b, n, h // n, d] -> [s // n, b, h, d]
            dq, dk, dv = (
                x.view(n, s // n, b, h // n, d).movedim(0, 2).contiguous().view(s // n, b, h, d)
                for x, (n, b, s, h, d) in zip((dq, dk, dv), self.qkv_shape_traits)
            )
        return dq, dk, dv


@lru_cache
def get_attention_cp_a2a_helper(b, s, h_q, h_kv, d_qk, d_v, seq_dim, n):
    attn_helper = AttentionCPA2AHelper(b, s, h_q, h_kv, d_qk, d_v, seq_dim, n)
    return attn_helper
