###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from typing import Optional

import torch

from primus_turbo.pytorch.core.low_precision import Float8QuantConfig
from primus_turbo.pytorch.ops.attention import (
    flash_attn_fp8_func,
    flash_attn_fp8_usp_func,
    flash_attn_func,
    flash_attn_usp_func,
)

__all__ = ["TurboAttention"]


class TurboAttention(torch.nn.Module):
    def __init__(
        self,
        dropout_p=0.0,
        softmax_scale=None,
        causal=False,
        window_size=(-1, -1),
        alibi_slopes=None,
        deterministic=False,
        return_lse=False,
        return_attn_probs=False,
        fp8_config: Optional[Float8QuantConfig] = None,
        ulysses_group=None,
        ring_group=None,
    ):
        super().__init__()

        self.dropout_p = dropout_p
        self.softmax_scale = softmax_scale
        self.causal = causal
        self.window_size = window_size
        self.alibi_slopes = alibi_slopes
        self.return_lse = return_lse
        self.return_attn_probs = return_attn_probs
        self.deterministic = deterministic
        self.fp8_config = fp8_config
        self.ulysses_group = ulysses_group
        self.ring_group = ring_group

        self.attention_fn = self.get_attention_func()

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ):
        kwargs = dict(
            dropout_p=self.dropout_p,
            softmax_scale=self.softmax_scale,
            causal=self.causal,
            window_size=self.window_size,
            bias=bias,
            alibi_slopes=self.alibi_slopes,
            deterministic=self.deterministic,
            return_lse=self.return_lse,
            return_attn_probs=self.return_attn_probs,
        )
        if self.fp8_config is not None:
            kwargs["fp8_config"] = self.fp8_config

        if self.ulysses_group is not None:
            kwargs["ulysses_group"] = self.ulysses_group
            kwargs["ring_group"] = self.ring_group

        return self.attention_fn(q, k, v, **kwargs)

    def get_attention_func(self):
        if self.fp8_config is not None:
            if self.ulysses_group is not None:
                return flash_attn_fp8_usp_func
            return flash_attn_fp8_func
        if self.ulysses_group is not None:
            return flash_attn_usp_func
        return flash_attn_func
