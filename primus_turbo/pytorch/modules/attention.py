###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from typing import Optional

import os
import torch

from primus_turbo.pytorch.core.low_precision import Float8QuantConfig
from primus_turbo.pytorch.ops.attention import (
    flash_attn_fp8_func,
    flash_attn_fp8_usp_func,
    flash_attn_func,
    flash_attn_usp_func,
)

__all__ = ["TurboAttention"]


# Opt out of having torch.compile trace INTO attention.
#
# Inductor gains nothing here and costs a great deal: the forward is either a prebuilt ASM
# kernel or a hand-written Triton one, the backward is a single hand-tuned Triton kernel, and
# letting inductor re-lower them means it must trace an autograd.Function containing a raw
# Triton launch -- which it then tries to recompile through its own pipeline and fails. The
# surrounding model still compiles; attention just becomes a graph break, which is the right
# boundary for a kernel that is already hand-optimised.
#
# Off by default so nothing changes for callers who have this working today. Set
# PRIMUS_TURBO_ATTN_NO_COMPILE=1 to take the boundary.
def _maybe_no_compile(fn):
    if os.environ.get("PRIMUS_TURBO_ATTN_NO_COMPILE", "") in ("", "0"):
        return fn
    try:
        import torch._dynamo

        return torch._dynamo.disable(fn)
    except Exception:  # noqa: BLE001 -- never break the caller over a diagnostic switch
        return fn



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


    @_maybe_no_compile
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
