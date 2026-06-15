###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Eager-Python references for the V4 attention kernels (ported from Primus).

These are the dtype / semantic ground truth for the Triton kernels:

* :func:`eager_hca_attention` — single-key-axis attention with optional
  per-head learned softmax sink, optional sliding window, and optional
  ``[Sq, Sk]`` additive bias. Covers ``compress_ratio == 0`` (dense +
  SWA + sink) and ``compress_ratio == 128`` (HCA — caller
  pre-concatenates the compressed pool to the local keys and supplies
  the joint-softmax additive bias).
* :func:`eager_csa_attention` — fused local-SWA + per-query top-K
  sparse attention with shared per-head sink and joint softmax.
  Covers ``compress_ratio == 4`` (CSA). The caller is responsible for
  the per-query top-K gather; the function takes the gathered
  ``[B, Sq, K, head_dim]`` tensor directly.

dtype contract: every matmul / einsum runs on tensor cores in the input
dtype (bf16 in production, fp32 accumulator inside); the *only* fp32 step
is the softmax block, which upcasts internally and returns fp32. The
caller-side ``probs.to(v.dtype)`` puts probs back on the bf16 path before
the V-matmul.
"""

from __future__ import annotations

from typing import Optional

import torch

__all__ = [
    "sliding_window_causal_mask",
    "eager_hca_attention",
    "eager_csa_attention",
]


# ---------------------------------------------------------------------------
# Mask helper (inlined from primus.backends...sliding_window_kv)
# ---------------------------------------------------------------------------


def sliding_window_causal_mask(
    seq_len: int,
    window: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return a ``[seq_len, seq_len]`` additive attention mask.

    Position ``j`` is allowed for query ``i`` iff ``j <= i`` (causal) and
    ``i - j < window`` (sliding window). Disallowed positions get ``-inf``;
    allowed positions get ``0``. A ``window`` of ``0`` or ``>= seq_len``
    degenerates to the standard causal mask.
    """
    q = torch.arange(seq_len, device=device).unsqueeze(1)
    k = torch.arange(seq_len, device=device).unsqueeze(0)
    dist = q - k
    if window <= 0 or window >= seq_len:
        allowed = dist >= 0
    else:
        allowed = (dist >= 0) & (dist < window)
    return torch.where(allowed, 0.0, float("-inf")).to(dtype)


def _build_local_attention_mask(
    seq_len: int,
    swa_window: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    window = swa_window if swa_window > 0 else seq_len
    return sliding_window_causal_mask(seq_len, window, device=device, dtype=dtype)


# ---------------------------------------------------------------------------
# Internal softmax-with-sink
# ---------------------------------------------------------------------------


def _softmax_with_sink(
    logits: torch.Tensor,
    sink: Optional[torch.Tensor],
) -> torch.Tensor:
    """Numerically stable softmax with optional per-head learned sink column.

    ``logits`` shape ``[B, H, ..., Sk]`` — head axis at dim=1. When ``sink``
    (shape ``[H]``) is given it is joined as a virtual key column with
    notional value zero, then dropped after softmax. This is the only fp32
    step: ``logits`` may arrive bf16; we upcast and return fp32.
    """
    logits_fp32 = logits.float()
    if sink is None:
        logits_fp32 = logits_fp32 - logits_fp32.amax(dim=-1, keepdim=True).detach()
        return logits_fp32.softmax(dim=-1)

    ndim = logits_fp32.dim()
    num_heads = sink.shape[0]
    view_shape = [1] * ndim
    view_shape[1] = num_heads
    view_shape[-1] = 1
    target_shape = list(logits_fp32.shape[:-1]) + [1]
    sink_col = sink.float().view(*view_shape).expand(*target_shape)
    logits_aug = torch.cat([logits_fp32, sink_col], dim=-1)
    logits_aug = logits_aug - logits_aug.amax(dim=-1, keepdim=True).detach()
    probs = logits_aug.softmax(dim=-1)
    return probs[..., :-1]


# ---------------------------------------------------------------------------
# Public reference ops
# ---------------------------------------------------------------------------


def eager_hca_attention(
    q: torch.Tensor,  # [B, H, Sq, D]
    k: torch.Tensor,  # [B, H, Sk, D]
    v: torch.Tensor,  # [B, H, Sk, D]
    *,
    sink: Optional[torch.Tensor],  # [H] or None
    swa_window: int,
    additive_mask: Optional[torch.Tensor],  # [Sq, Sk] or None
    attn_dropout: float,
    training: bool,
    scale: float,
) -> torch.Tensor:
    """Eager-Python V4 dense / HCA attention. Returns ``[B, H, Sq, D]`` in ``v.dtype``.

    Mask resolution:

    * ``additive_mask is not None`` — used directly, ``swa_window`` ignored.
      (HCA caller pre-builds ``cat([local_mask, hca_mask])``.)
    * ``additive_mask is None`` and ``swa_window > 0`` — SWA-causal mask
      built internally (requires ``Sq == Sk``).
    * ``additive_mask is None`` and ``swa_window <= 0`` — full causal mask.
    """
    Sq = q.shape[2]
    Sk = k.shape[2]

    if additive_mask is None:
        if Sq != Sk:
            raise ValueError(
                "eager_hca_attention requires `additive_mask` when Sq != Sk; "
                f"got Sq={Sq}, Sk={Sk}."
            )
        mask = _build_local_attention_mask(Sq, swa_window, device=q.device, dtype=q.dtype)
    else:
        mask = additive_mask

    logits = torch.matmul(q, k.transpose(-2, -1)) * scale
    logits = logits + mask
    probs = _softmax_with_sink(logits, sink)
    if attn_dropout > 0.0 and training:
        probs = torch.nn.functional.dropout(probs, p=attn_dropout)
    return torch.matmul(probs.to(v.dtype), v)


def eager_csa_attention(
    q: torch.Tensor,  # [B, H, Sq, D]
    k_local: torch.Tensor,  # [B, H, Sq, D]
    v_local: torch.Tensor,  # [B, H, Sq, D]
    gathered: torch.Tensor,  # [B, Sq, K, D] — pre-gathered per-query top-K
    *,
    sink: Optional[torch.Tensor],  # [H] or None
    swa_window: int,
    sparse_mask: torch.Tensor,  # [B, Sq, K] additive (broadcasts over H)
    attn_dropout: float,
    training: bool,
    scale: float,
) -> torch.Tensor:
    """Eager-Python V4 CSA fused attention (joint local SWA + sparse top-K).

    The local SWA mask is built internally from ``swa_window``; the caller
    pre-builds ``sparse_mask`` to flag indexer-dropped slots (``-inf`` for
    ``topk_idx == -1``). Returns ``[B, H, Sq, D]`` in ``v_local.dtype``.
    """
    B, H, Sq, D = q.shape
    K = gathered.shape[2]

    local_mask = _build_local_attention_mask(Sq, swa_window, device=q.device, dtype=q.dtype)

    local_logits = torch.matmul(q, k_local.transpose(-2, -1)) * scale
    local_logits = local_logits + local_mask

    gathered_h = gathered.unsqueeze(1).expand(B, H, Sq, K, D)
    sparse_logits = torch.einsum("bhsd,bhskd->bhsk", q, gathered_h) * scale
    sparse_logits = sparse_logits + sparse_mask.unsqueeze(1)

    joint_logits = torch.cat([local_logits, sparse_logits], dim=-1)
    probs = _softmax_with_sink(joint_logits, sink)

    if attn_dropout > 0.0 and training:
        probs = torch.nn.functional.dropout(probs, p=attn_dropout)

    probs_local = probs[..., :Sq].to(v_local.dtype)
    probs_sparse = probs[..., Sq:].to(v_local.dtype)

    out_local = torch.matmul(probs_local, v_local)
    out_sparse = torch.einsum(
        "bhsk,bhskd->bhsd",
        probs_sparse,
        gathered_h.to(v_local.dtype),
    )

    return out_local + out_sparse
