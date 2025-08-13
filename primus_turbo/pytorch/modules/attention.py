###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from typing import Optional

import torch

from primus_turbo.pytorch.ops.attention import attention, attention_fp8_blockwise

__all__ = ["TurboAttention"]


class TurboAttention(torch.nn.Module):
    def __init__(
        self,
        dropout_p=0.0,
        softmax_scale=None,
        causal=False,
        window_size=(-1, -1),  # -1 means infinite context window
        alibi_slopes=None,
        deterministic=False,
        return_lse=False,
        return_attn_probs=False,
        use_fp8=False,
        backend_type: str = "ck",  # 'ck', 'triton'
    ):
        super().__init__()

        assert not (use_fp8 and backend_type == "ck"), "When use_fp8 is True, attention_type cannot be 'ck'."

        self.dropout_p = dropout_p
        self.softmax_scale = softmax_scale
        self.causal = causal
        self.window_size = window_size
        self.alibi_slopes = alibi_slopes
        self.return_lse = return_lse
        self.return_attn_probs = return_attn_probs
        self.deterministic = deterministic
        self.backend_type = backend_type

        if backend_type == "ck" and use_fp8 == False:
            self.attention_fn = attention
        elif backend_type == "triton":
            self.attention_fn = attention_fp8_blockwise
        else:
            raise ValueError(f"Unknown attention type: {backend_type}")

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
            dropout_p=self.dropout_p,
            softmax_scale=self.softmax_scale,
            causal=self.causal,
            window_size=self.window_size,
            bias=bias,
            alibi_slopes=self.alibi_slopes,
            deterministic=self.deterministic,
            return_lse=self.return_lse,
            return_attn_probs=self.return_attn_probs,
            backend_type=self.backend_type,
        )
