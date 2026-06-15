###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""DeepSeek-V4 CSA attention autograd entry point (ported from Primus).

CSA (``compress_ratio == 4``) fuses local sliding-window attention, a
per-query top-K sparse branch, and a shared per-head softmax sink into a
single online softmax.

Public API:

* :func:`csa_attention` — CSA from a pre-gathered top-K tensor
  (``[B, Sq, K_topk, D]``). Drop-in for :func:`eager_csa_attention`.
* :func:`csa_attention_from_pool` — CSA that gathers the sparse keys
  in-kernel from the compressed pool and produces ``dpool`` directly via
  atomic scatter-add in the backward (no materialised gathered tensor).

Each routes through a :class:`torch.autograd.Function` wrapping the Triton
FWD + Triton BWD (the BWD re-materialises the joint softmax from the saved
fp32 LSE). When ``K_topk == 0`` (degenerate Indexer state) both functions
short-circuit to :func:`hca_attention`: CSA's local-SWA branch is
bit-identical to dense+SWA+sink in that limit.
"""

from __future__ import annotations

from typing import Optional

import torch

from primus_turbo.pytorch.ops.attention.hca_attention import hca_attention
from primus_turbo.triton.attention.deepseek import (
    _launch_csa_attention_bwd,
    _launch_csa_attention_fwd,
    _launch_csa_attention_pool_bwd,
    _launch_csa_attention_pool_fwd,
)

__all__ = [
    "CSAAttentionFn",
    "CSAPoolAttentionFn",
    "csa_attention",
    "csa_attention_from_pool",
]


class CSAAttentionFn(torch.autograd.Function):
    """Triton FWD + Triton BWD for the CSA fused attention path."""

    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        q: torch.Tensor,  # [B, H, Sq, D]
        k_local: torch.Tensor,  # [B, H, Sq, D]
        v_local: torch.Tensor,  # [B, H, Sq, D]
        gathered: torch.Tensor,  # [B, Sq, K_topk, D]
        sparse_mask: torch.Tensor,  # [B, Sq, K_topk]
        sink: Optional[torch.Tensor],  # [H] or None
        swa_window: int,
        attn_dropout: float,
        training: bool,
        scale: float,
    ) -> torch.Tensor:
        if attn_dropout > 0.0 and training:
            raise NotImplementedError(
                "csa_attention does not implement in-kernel attention "
                "dropout (V4 trains with attn_dropout=0). Got "
                f"attn_dropout={attn_dropout}, training={training}."
            )

        out, lse = _launch_csa_attention_fwd(
            q,
            k_local,
            v_local,
            gathered,
            sparse_mask,
            sink=sink,
            swa_window=swa_window,
            scale=scale,
        )
        ctx.save_for_backward(q, k_local, v_local, gathered, sparse_mask, out, lse, sink)
        ctx.swa_window = int(swa_window)
        ctx.attn_dropout = float(attn_dropout)
        ctx.training_mode = bool(training)
        ctx.scale = float(scale)
        ctx.sink_was_none = sink is None
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):  # type: ignore[override]
        q, k_local, v_local, gathered, sparse_mask, out, lse, sink = ctx.saved_tensors

        sink_arg = None if ctx.sink_was_none else sink

        if not grad_out.is_contiguous():
            grad_out = grad_out.contiguous()

        dq, dk_local, dv_local, dgathered, dsink = _launch_csa_attention_bwd(
            q,
            k_local,
            v_local,
            gathered,
            sparse_mask,
            out,
            grad_out,
            lse,
            sink=sink_arg,
            swa_window=ctx.swa_window,
            scale=ctx.scale,
        )

        if not ctx.needs_input_grad[0]:
            dq = None
        if not ctx.needs_input_grad[1]:
            dk_local = None
        if not ctx.needs_input_grad[2]:
            dv_local = None
        if not ctx.needs_input_grad[3]:
            dgathered = None
        # sparse_mask (index 4) is not learnable -> no gradient.
        if not ctx.needs_input_grad[5] or ctx.sink_was_none:
            dsink = None

        # Forward signature: (q, k_local, v_local, gathered, sparse_mask,
        # sink, swa_window, attn_dropout, training, scale).
        return dq, dk_local, dv_local, dgathered, None, dsink, None, None, None, None


class CSAPoolAttentionFn(torch.autograd.Function):
    """Triton CSA attention with in-kernel topk gather and pool scatter-add."""

    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        q: torch.Tensor,
        k_local: torch.Tensor,
        v_local: torch.Tensor,
        pool: torch.Tensor,
        topk_idxs: torch.Tensor,
        sink: Optional[torch.Tensor],
        swa_window: int,
        attn_dropout: float,
        training: bool,
        scale: float,
    ) -> torch.Tensor:
        if attn_dropout > 0.0 and training:
            raise NotImplementedError(
                "csa_attention_from_pool does not implement in-kernel "
                "attention dropout (V4 trains with attn_dropout=0). Got "
                f"attn_dropout={attn_dropout}, training={training}."
            )

        out, lse = _launch_csa_attention_pool_fwd(
            q,
            k_local,
            v_local,
            pool,
            topk_idxs,
            sink=sink,
            swa_window=swa_window,
            scale=scale,
        )
        ctx.save_for_backward(q, k_local, v_local, pool, topk_idxs, out, lse, sink)
        ctx.swa_window = int(swa_window)
        ctx.attn_dropout = float(attn_dropout)
        ctx.training_mode = bool(training)
        ctx.scale = float(scale)
        ctx.sink_was_none = sink is None
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):  # type: ignore[override]
        q, k_local, v_local, pool, topk_idxs, out, lse, sink = ctx.saved_tensors
        sink_arg = None if ctx.sink_was_none else sink

        if not grad_out.is_contiguous():
            grad_out = grad_out.contiguous()

        dq, dk_local, dv_local, dpool, dsink = _launch_csa_attention_pool_bwd(
            q,
            k_local,
            v_local,
            pool,
            topk_idxs,
            out,
            grad_out,
            lse,
            sink=sink_arg,
            swa_window=ctx.swa_window,
            scale=ctx.scale,
        )

        if not ctx.needs_input_grad[0]:
            dq = None
        if not ctx.needs_input_grad[1]:
            dk_local = None
        if not ctx.needs_input_grad[2]:
            dv_local = None
        if not ctx.needs_input_grad[3]:
            dpool = None
        if not ctx.needs_input_grad[5] or ctx.sink_was_none:
            dsink = None

        # Forward signature: (q, k_local, v_local, pool, topk_idxs, sink,
        # swa_window, attn_dropout, training, scale).
        return dq, dk_local, dv_local, dpool, None, dsink, None, None, None, None


def csa_attention(
    q: torch.Tensor,  # [B, H, Sq, D]
    k_local: torch.Tensor,  # [B, H, Sq, D]
    v_local: torch.Tensor,  # [B, H, Sq, D]
    gathered: torch.Tensor,  # [B, Sq, K_topk, D]
    *,
    sink: Optional[torch.Tensor],  # [H] or None
    swa_window: int,
    sparse_mask: torch.Tensor,  # [B, Sq, K_topk]
    attn_dropout: float,
    training: bool,
    scale: float,
) -> torch.Tensor:
    """Triton-backed DeepSeek-V4 CSA fused attention (pre-gathered top-K).

    Drop-in replacement for :func:`eager_csa_attention`. When
    ``gathered.shape[2] == 0`` (degenerate Indexer state) the wrapper
    short-circuits to :func:`hca_attention`: CSA's local SWA branch is
    bit-identical to dense+SWA+sink in that limit.

    Returns ``[B, H, Sq, D]`` in ``v_local.dtype``.
    """
    K_topk = gathered.shape[2]
    if K_topk == 0:
        return hca_attention(
            q,
            k_local,
            v_local,
            sink=sink,
            swa_window=swa_window,
            additive_mask=None,
            attn_dropout=attn_dropout,
            training=training,
            scale=scale,
        )

    return CSAAttentionFn.apply(
        q,
        k_local,
        v_local,
        gathered,
        sparse_mask,
        sink,
        swa_window,
        attn_dropout,
        training,
        scale,
    )


def csa_attention_from_pool(
    q: torch.Tensor,  # [B, H, Sq, D]
    k_local: torch.Tensor,  # [B, H, Sq, D]
    v_local: torch.Tensor,  # [B, H, Sq, D]
    pool: torch.Tensor,  # [B, P, D]
    *,
    topk_idxs: torch.Tensor,  # [B, Sq, K_topk], -1 masks a slot
    sink: Optional[torch.Tensor],  # [H] or None
    swa_window: int,
    attn_dropout: float,
    training: bool,
    scale: float,
) -> torch.Tensor:
    """Triton-backed CSA attention that gathers sparse keys in-kernel.

    ``pool`` is the compressed-pool tensor before per-query top-K gather.
    ``topk_idxs`` drives the sparse branch directly; negative entries are
    masked. The backward kernel emits ``dpool`` via atomic scatter-add,
    avoiding the materialised ``[B, Sq, K_topk, D]`` gathered tensor.

    Returns ``[B, H, Sq, D]`` in ``v_local.dtype``.
    """
    K_topk = topk_idxs.shape[2]
    if K_topk == 0:
        return hca_attention(
            q,
            k_local,
            v_local,
            sink=sink,
            swa_window=swa_window,
            additive_mask=None,
            attn_dropout=attn_dropout,
            training=training,
            scale=scale,
        )

    return CSAPoolAttentionFn.apply(
        q,
        k_local,
        v_local,
        pool,
        topk_idxs,
        sink,
        swa_window,
        attn_dropout,
        training,
        scale,
    )
