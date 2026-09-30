###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Launch and validation helpers for GPT-OSS fused QK RMSNorm + RoPE."""

from __future__ import annotations

from typing import Sequence

import torch

from primus_turbo.flydsl.rope.qk_rmsnorm_rope_kernel import (
    QK_RMSNORM_ROPE_HEAD_DIM,
    flydsl_qkv_rmsnorm_rope_backward,
    flydsl_qkv_rmsnorm_rope_forward,
)


def qk_rmsnorm_rope_shape_error(
    qkv: torch.Tensor,
    q_gamma: torch.Tensor,
    k_gamma: torch.Tensor,
    freqs: torch.Tensor,
    split_sizes: Sequence[int],
) -> str | None:
    """Return why this input is unsupported, or ``None`` for the GPT-OSS contract."""
    D = QK_RMSNORM_ROPE_HEAD_DIM
    if qkv.ndim != 4:
        return f"qkv must be [S,B,NG,(NPG+2)*D], got {tuple(qkv.shape)}"
    if qkv.dtype != torch.bfloat16 or not qkv.is_contiguous():
        return f"qkv must be contiguous bfloat16, got dtype={qkv.dtype} contiguous={qkv.is_contiguous()}"
    if len(split_sizes) != 3:
        return f"split_sizes must be [q,k,v], got {list(split_sizes)}"
    q_size, k_size, v_size = split_sizes
    if q_size % D or k_size != D or v_size != D:
        return f"expected q multiple of {D} and k=v={D}, got {list(split_sizes)}"
    if qkv.shape[-1] != q_size + k_size + v_size:
        return f"packed width {qkv.shape[-1]} != q+k+v {q_size + k_size + v_size}"
    if q_gamma.shape != (D,) or k_gamma.shape != (D,):
        return f"q/k gamma must both be [{D}], got {tuple(q_gamma.shape)}/{tuple(k_gamma.shape)}"
    if q_gamma.dtype != torch.bfloat16 or k_gamma.dtype != torch.bfloat16:
        return f"q/k gamma must be bfloat16, got {q_gamma.dtype}/{k_gamma.dtype}"
    if freqs.shape != (qkv.shape[0], 1, 1, D):
        return f"freqs must be [{qkv.shape[0]},1,1,{D}], got {tuple(freqs.shape)}"
    if freqs.dtype != torch.float32 or not freqs.is_contiguous():
        return f"freqs must be contiguous float32, got dtype={freqs.dtype} contiguous={freqs.is_contiguous()}"
    return None


def qk_rmsnorm_rope_fwd_impl(qkv, q_gamma, k_gamma, freqs, split_sizes, eps):
    why = qk_rmsnorm_rope_shape_error(qkv, q_gamma, k_gamma, freqs, split_sizes)
    if why is not None:
        raise ValueError(f"fused_qkv_rmsnorm_rope: unsupported input ({why})")
    return flydsl_qkv_rmsnorm_rope_forward(
        qkv, q_gamma, k_gamma, freqs, split_sizes, eps
    )


def qk_rmsnorm_rope_bwd_impl(
    dq, dk, dv, qkv, q_gamma, k_gamma, freqs, q_rstd, k_rstd, split_sizes
):
    dqkv, dqg_partial, dkg_partial = flydsl_qkv_rmsnorm_rope_backward(
        dq.contiguous(),
        dk.contiguous(),
        dv.contiguous(),
        qkv,
        q_gamma,
        k_gamma,
        freqs,
        q_rstd,
        k_rstd,
        split_sizes,
    )
    return (
        dqkv,
        dqg_partial.sum(dim=0).to(q_gamma.dtype),
        dkg_partial.sum(dim=0).to(k_gamma.dtype),
    )
