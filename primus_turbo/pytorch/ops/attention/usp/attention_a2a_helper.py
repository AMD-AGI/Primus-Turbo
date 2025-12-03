from functools import lru_cache
from typing import Tuple

import torch


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
