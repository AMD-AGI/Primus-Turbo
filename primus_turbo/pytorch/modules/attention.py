import torch

from typing import Optional
from primus_turbo.pytorch.ops.attention import attention_ck, attention_triton

__all__ = ["CoreAttention"]


class CoreAttention(torch.nn.Module):
    def __init__(
        self,
        attention_type: str = "ck",  # 'ck', 'triton'
        dropout_p=0.0,
        softmax_scale=None,
        causal=False,
        window_size=(-1, -1),  # -1 means infinite context window
        alibi_slopes=None,
        deterministic=True,
        return_lse=False,
        return_attn_probs=False,
        use_fp8=False,
    ):
        super().__init__()

        assert not (
            use_fp8 and attention_type == "ck"
        ), "When use_fp8 is True, attention_type cannot be 'ck'."

        self.dropout_p = dropout_p
        self.softmax_scale = softmax_scale
        self.causal = causal
        self.window_size = window_size
        self.alibi_slopes = alibi_slopes
        self.return_lse = return_lse
        self.return_attn_probs = return_attn_probs
        self.deterministic = deterministic
        self.use_fp8 = use_fp8

        if attention_type == "ck":
            self.attention_fn = attention_ck
        elif attention_type == "triton":
            self.attention_fn = attention_triton
        else:
            raise ValueError(f"Unknown attention type: {attention_type}")

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ):

        return self.attention_fn(
            q,
            k,
            v,
            bias,
            dropout_p=self.dropout_p,
            softmax_scale=self.softmax_scale,
            causal=self.causal,
            window_size=self.window_size,
            alibi_slopes=self.alibi_slopes,
            deterministic=self.deterministic,
            return_lse=self.return_lse,
            return_attn_probs=self.return_attn_probs,
            use_fp8=self.use_fp8,
        )
