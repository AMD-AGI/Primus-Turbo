###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""DeepSeek-V4 single-latent sparse-MLA attention as a multi-backend training op.

Public API (``sparse_mla_func``) mirrors the DSV4 sparse-MLA convention:

* ``q``            : ``[T, H, 576]`` bf16  (512 latent + 64 rope).
* ``kv``           : ``[num_kv, 1, 576]`` bf16 single MQA latent (K == V);
                     ``[num_kv, 576]`` is also accepted.
* ``topk_indices`` : ``[T, topk]`` int32, per-token selected kv rows (-1 padded;
                     ``topk`` padded to a multiple of 64).
* ``attn_sink``    : ``[H]`` fp32 or ``None``.
* returns ``o`` : ``[T, H, 512]`` bf16 (and ``lse`` ``[T, H]`` fp32 when
  ``return_lse``).

The backend (flydsl fast default / triton oracle-fallback) is resolved once in
the forward and pinned through ``ctx.backend`` so fwd and bwd never mix backends.
"""

from typing import Optional

import torch

from primus_turbo.pytorch.core.backend import (
    BackendType,
    GlobalBackendManager,
    PrecisionType,
)
from primus_turbo.pytorch.kernels.attention.sparse_mla_impl import (
    SparseMlaBwdDispatcher,
    SparseMlaFwdDispatcher,
    resolve_sparse_mla_fwd_backend,
)

__all__ = ["sparse_mla_func"]

# Sparse-MLA selects its backend on the bf16/fp16/fp32 precision bucket.
_SPARSE_ATTN_PRECISION = PrecisionType.BF16_FP16_FP32


class SparseMLAFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, kv, topk_indices, attn_sink, kv_lora_rank, scale, return_lse, is_grad_enabled):
        is_grad = is_grad_enabled and any(x is not None and x.requires_grad for x in (q, kv, attn_sink))

        backend = resolve_sparse_mla_fwd_backend(
            GlobalBackendManager.get_sparse_attn_backend(_SPARSE_ATTN_PRECISION),
            q=q,
            kv=kv,
            topk_indices=topk_indices,
            attn_sink=attn_sink,
            kv_lora_rank=kv_lora_rank,
            scale=scale,
        )
        o, lse = SparseMlaFwdDispatcher.dispatch(
            BackendType.FLYDSL,
            backend,
            q=q,
            kv=kv,
            topk_indices=topk_indices,
            attn_sink=attn_sink,
            kv_lora_rank=kv_lora_rank,
            scale=scale,
        )

        if is_grad:
            ctx.save_for_backward(q, kv, o, lse, topk_indices, attn_sink)
            ctx.kv_lora_rank = kv_lora_rank
            ctx.scale = scale
            ctx.backend = backend

        return (o, lse) if return_lse else o

    @staticmethod
    def backward(ctx, do, *args):
        q, kv, o, lse, topk_indices, attn_sink = ctx.saved_tensors
        dq, dkv, d_sink = SparseMlaBwdDispatcher.dispatch(
            BackendType.FLYDSL,
            ctx.backend,  # pinned to the forward's backend
            q=q,
            kv=kv,
            o=o,
            do=do.contiguous(),
            topk_indices=topk_indices,
            lse=lse,
            attn_sink=attn_sink,
            kv_lora_rank=ctx.kv_lora_rank,
            scale=ctx.scale,
        )
        # inputs: q, kv, topk_indices, attn_sink, kv_lora_rank, scale, return_lse, is_grad_enabled
        return dq, dkv, None, d_sink, None, None, None, None


def sparse_mla_func(
    q: torch.Tensor,
    kv: torch.Tensor,
    topk_indices: torch.Tensor,
    attn_sink: Optional[torch.Tensor] = None,
    kv_lora_rank: int = 512,
    scale: Optional[float] = None,
    return_lse: bool = False,
):
    """DeepSeek-V4 single-latent sparse-MLA attention (fwd + bwd autograd).

    Arguments:
        q: (T, H, 576) bf16 query (512 latent + 64 rope cols).
        kv: (num_kv, 1, 576) or (num_kv, 576) bf16 single latent (K == V).
        topk_indices: (T, topk) int32 per-token kv-row selection (-1 padded).
        attn_sink: (H,) fp32 per-head sink logit, or None.
        kv_lora_rank: latent width (512).
        scale: QK scale; defaults to 1 / sqrt(576).
        return_lse: also return the softmax LSE (T, H) fp32.

    Returns:
        o: (T, H, 512) bf16 (and lse (T, H) fp32 when return_lse).
    """
    return SparseMLAFunc.apply(
        q,
        kv,
        topk_indices,
        attn_sink,
        kv_lora_rank,
        scale,
        return_lse,
        torch.is_grad_enabled(),
    )
