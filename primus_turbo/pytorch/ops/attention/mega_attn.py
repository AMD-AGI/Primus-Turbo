###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Causal self-attention whose forward runs as a persistent kernel (one CTA per CU claiming work
items); the backward is the FlyDSL sbhd backward. The forward is bitwise equal to ``flash_attn_func``'s
FlyDSL sbhd path."""

import torch

from primus_turbo.flydsl.mega_attn.attn_fwd_persistent import attn_fwd_persistent
from primus_turbo.pytorch.kernels.attention.attention_flydsl_impl import (
    flash_attn_sbhd_flydsl_backward_impl,
)

__all__ = ["mega_attn_func"]


class MegaAttnFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v):
        out, lse = attn_fwd_persistent(q, k, v)
        ctx.save_for_backward(q, k, v, out, lse)
        return out

    @staticmethod
    def backward(ctx, dout):
        q, k, v, out, lse = ctx.saved_tensors
        Sq, B, Hq, D = q.shape
        dq, dk, dv = flash_attn_sbhd_flydsl_backward_impl(
            dout, q, k, v, out, lse.view(B, Sq, Hq).permute(0, 2, 1), softmax_scale=D**-0.5, causal=True
        )
        return dq, dk, dv


def mega_attn_func(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """``q`` [s, b, hq, d], ``k`` / ``v`` [s, b, hkv, d] contiguous bf16 (sbhd), causal, scale 1/sqrt(d)."""
    return MegaAttnFunction.apply(q, k, v)
