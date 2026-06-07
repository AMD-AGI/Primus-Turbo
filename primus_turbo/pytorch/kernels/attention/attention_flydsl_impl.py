###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""FlyDSL attention kernel-impl wrappers (gfx1250 / RDNA4).

Thin bridge between Primus-Turbo's attention autograd layer and the FlyDSL
kernels in ``primus_turbo.flydsl.attention``:
  - forward : vendored ``flash_attn_varlen_d192_gfx1250`` (always emits LSE).
  - backward: net-new ``flash_attn_varlen_bwd_d192_gfx1250``.

Only the DeepSeek-V3 MLA head dims are supported (D_qk=192, D_v=128, bf16,
varlen THD, plain MHA / GQA). ``flydsl_attn_varlen_supported`` gates the
dispatch so unsupported configs fall back to the aiter / Triton path.
"""

from __future__ import annotations

import torch

from primus_turbo.pytorch.core.utils import is_gfx1250

HEAD_DIM_QK = 192
HEAD_DIM_V = 128


def flydsl_attn_varlen_supported(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dropout_p: float,
    window_size,
    bias,
    alibi_slopes,
) -> bool:
    """True iff the gfx1250 FlyDSL FMHA can handle this varlen config."""
    return (
        is_gfx1250()
        and q.dtype == torch.bfloat16
        and q.shape[-1] == HEAD_DIM_QK
        and v.shape[-1] == HEAD_DIM_V
        and dropout_p == 0.0
        and tuple(window_size[:2]) == (-1, -1)
        and bias is None
        and alibi_slopes is None
    )


def attention_flydsl_varlen_forward_impl(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    softmax_scale: float,
    causal: bool,
):
    """Returns (out [total_q, H, 128], lse [total_q, H]).

    LSE is always computed (the backward consumes it), regardless of whether
    the caller asked for it.
    """
    from primus_turbo.flydsl.attention.fmha_gfx1250 import (
        flash_attn_varlen_d192_gfx1250,
    )

    out, lse = flash_attn_varlen_d192_gfx1250(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        softmax_scale=softmax_scale,
        causal=causal,
        return_lse=True,
    )
    return out, lse


def attention_flydsl_varlen_backward_impl(
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    softmax_lse: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    softmax_scale: float,
    causal: bool,
):
    """Returns (dq, dk, dv) in the THD layout / dtype of q, k, v."""
    from primus_turbo.flydsl.attention.attention_bwd_kernel import (
        flash_attn_varlen_bwd_d192_gfx1250,
    )

    return flash_attn_varlen_bwd_d192_gfx1250(
        dout,
        q,
        k,
        v,
        out,
        softmax_lse,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        softmax_scale=softmax_scale,
        causal=causal,
    )
