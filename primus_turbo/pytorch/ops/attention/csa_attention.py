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

* :func:`csa_attention_from_pool` — CSA that gathers the sparse keys
  in-kernel from the compressed pool and produces ``dpool`` directly via
  atomic scatter-add in the backward (no materialised gathered tensor).

It routes through a :class:`torch.autograd.Function` wrapping the FlyDSL /
Triton FWD + Triton BWD (the BWD re-materialises the joint softmax from the
saved fp32 LSE). When ``K_topk == 0`` (degenerate Indexer state) it
short-circuits to :func:`hca_attention`: CSA's local-SWA branch is
bit-identical to dense+SWA+sink in that limit.
"""

from __future__ import annotations

from typing import Optional

import torch

# Importing the dispatcher registers the CSA forward custom_op
# (``deepseek_csa_pool_attn_fwd``).
import primus_turbo.pytorch.kernels.attention.deepseek_attn_impl  # noqa: F401
from primus_turbo.pytorch.core.backend import BackendType
from primus_turbo.pytorch.kernels.attention.deepseek_attn_impl import (
    deepseek_csa_pool_attn_bwd,
)
from primus_turbo.pytorch.ops.attention.hca_attention import hca_attention

__all__ = [
    "CSAPoolAttentionFn",
    "csa_attention_from_pool",
]

# Sentinel for "no per-call backend override" passed into the custom_op.
_NO_BACKEND_OVERRIDE = 0


def _backend_to_int(backend: Optional[BackendType]) -> int:
    return _NO_BACKEND_OVERRIDE if backend is None else int(backend.value)


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
        backend_override: int,
    ) -> torch.Tensor:
        if attn_dropout > 0.0 and training:
            raise NotImplementedError(
                "csa_attention_from_pool does not implement in-kernel "
                "attention dropout (V4 trains with attn_dropout=0). Got "
                f"attn_dropout={attn_dropout}, training={training}."
            )

        out, lse = torch.ops.primus_turbo.deepseek_csa_pool_attn_fwd(
            q,
            k_local,
            v_local,
            pool,
            topk_idxs,
            sink,
            int(swa_window),
            float(scale),
            BackendType.TRITON.value,
            int(backend_override),
        )
        ctx.save_for_backward(q, k_local, v_local, pool, topk_idxs, out, lse, sink)
        ctx.swa_window = int(swa_window)
        ctx.attn_dropout = float(attn_dropout)
        ctx.training_mode = bool(training)
        ctx.scale = float(scale)
        ctx.sink_was_none = sink is None
        ctx.backend_override = int(backend_override)
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):  # type: ignore[override]
        q, k_local, v_local, pool, topk_idxs, out, lse, sink = ctx.saved_tensors
        sink_arg = None if ctx.sink_was_none else sink

        if not grad_out.is_contiguous():
            grad_out = grad_out.contiguous()

        dq, dk_local, dv_local, dpool, dsink = deepseek_csa_pool_attn_bwd(
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
            backend_override=ctx.backend_override,
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
        # swa_window, attn_dropout, training, scale, backend_override).
        return dq, dk_local, dv_local, dpool, None, dsink, None, None, None, None, None


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
    backend: Optional[BackendType] = None,
) -> torch.Tensor:
    """CSA attention that gathers sparse keys in-kernel (Triton or FlyDSL).

    ``pool`` is the compressed-pool tensor before per-query top-K gather.
    ``topk_idxs`` drives the sparse branch directly; negative entries are
    masked. The backward kernel emits ``dpool`` via atomic scatter-add,
    avoiding the materialised ``[B, Sq, K_topk, D]`` gathered tensor.

    ``backend`` selects the forward kernel (default Triton; the CSA FlyDSL
    two-pass sparse branch is a later optimization round, design §4.7).
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
            backend=backend,
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
        _backend_to_int(backend),
    )
