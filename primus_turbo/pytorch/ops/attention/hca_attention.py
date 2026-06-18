###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""DeepSeek-V4 dense / HCA attention autograd entry point (ported from Primus).

Public API:

* :func:`hca_attention` — dense (``compress_ratio == 0``) / HCA
  (``compress_ratio == 128``) attention. Drop-in for
  :func:`eager_hca_attention`. Dense and HCA share one Triton kernel;
  HCA is selected via the split-mask branch (``hca_local_seqlen > 0``).
* :class:`HCAAttentionFn` — :class:`torch.autograd.Function` wrapping the
  Triton FWD + Triton BWD (the BWD re-materialises softmax from the saved
  fp32 LSE rather than storing the ``[Sq, Sk]`` probability matrix).

dtype contract (matches :func:`eager_hca_attention`): Q/K/V matmuls run in
input dtype with fp32 accumulator; the online-softmax accumulator and the
saved LSE are fp32; output is in ``v.dtype``.
"""

from __future__ import annotations

from typing import Optional

import torch

# Importing the dispatcher registers the ``primus_turbo::deepseek_attn_fwd``
# custom_op (Triton + FlyDSL backends) used by the forward below.
import primus_turbo.pytorch.kernels.attention.deepseek_attn_impl  # noqa: F401
from primus_turbo.pytorch.core.backend import BackendType
from primus_turbo.pytorch.kernels.attention.deepseek_attn_impl import deepseek_attn_bwd

__all__ = [
    "HCAAttentionFn",
    "hca_attention",
]

# Sentinel for "no per-call backend override" passed into the custom_op.
_NO_BACKEND_OVERRIDE = 0


def _backend_to_int(backend: Optional[BackendType]) -> int:
    return _NO_BACKEND_OVERRIDE if backend is None else int(backend.value)


class HCAAttentionFn(torch.autograd.Function):
    """Triton FWD + Triton BWD for DeepSeek-V4 dense / HCA attention."""

    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        q: torch.Tensor,  # [B, H, Sq, D]
        k: torch.Tensor,  # [B, K_H, Sk, D]
        v: torch.Tensor,  # [B, K_H, Sk, D]
        sink: Optional[torch.Tensor],  # [H] or None
        additive_mask: Optional[torch.Tensor],  # [Sq, Sk] or None
        swa_window: int,
        attn_dropout: float,
        training: bool,
        scale: float,
        hca_local_seqlen: int,
        backend_override: int,
    ) -> torch.Tensor:
        if attn_dropout > 0.0 and training:
            raise NotImplementedError(
                "hca_attention does not implement in-kernel attention dropout "
                "(V4 trains with attn_dropout=0). Got "
                f"attn_dropout={attn_dropout}, training={training}."
            )

        # Forward dispatch (Triton default; FlyDSL when selected and supported).
        # register_fake mirrors (out like q, lse [B, H, Sq] fp32) for torch.compile.
        out, lse = torch.ops.primus_turbo.deepseek_attn_fwd(
            q,
            k,
            v,
            sink,
            additive_mask,
            int(swa_window),
            float(scale),
            int(hca_local_seqlen),
            BackendType.TRITON.value,
            int(backend_override),
        )
        ctx.save_for_backward(q, k, v, out, lse, sink, additive_mask)
        ctx.swa_window = int(swa_window)
        ctx.attn_dropout = float(attn_dropout)
        ctx.training_mode = bool(training)
        ctx.scale = float(scale)
        ctx.hca_local_seqlen = int(hca_local_seqlen)
        ctx.sink_was_none = sink is None
        ctx.mask_was_none = additive_mask is None
        ctx.backend_override = int(backend_override)
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):  # type: ignore[override]
        q, k, v, out, lse, sink, additive_mask = ctx.saved_tensors

        sink_arg = None if ctx.sink_was_none else sink
        mask_arg = None if ctx.mask_was_none else additive_mask

        if not grad_out.is_contiguous():
            grad_out = grad_out.contiguous()

        dq, dk, dv, dsink = deepseek_attn_bwd(
            q,
            k,
            v,
            out,
            grad_out,
            lse,
            sink=sink_arg,
            additive_mask=mask_arg,
            swa_window=ctx.swa_window,
            scale=ctx.scale,
            hca_local_seqlen=ctx.hca_local_seqlen,
            backend_override=ctx.backend_override,
        )

        if not ctx.needs_input_grad[0]:
            dq = None
        if not ctx.needs_input_grad[1]:
            dk = None
        if not ctx.needs_input_grad[2]:
            dv = None
        if not ctx.needs_input_grad[3] or ctx.sink_was_none:
            dsink = None

        # Forward signature: (q, k, v, sink, additive_mask, swa_window,
        # attn_dropout, training, scale, hca_local_seqlen, backend_override)
        return dq, dk, dv, dsink, None, None, None, None, None, None, None


def hca_attention(
    q: torch.Tensor,  # [B, H, Sq, D]
    k: torch.Tensor,  # [B, K_H, Sk, D]   K_H in {1, H}
    v: torch.Tensor,  # [B, K_H, Sk, D]
    *,
    sink: Optional[torch.Tensor],  # [H] or None
    swa_window: int,
    additive_mask: Optional[torch.Tensor],  # [Sq, Sk] or None
    attn_dropout: float,
    training: bool,
    scale: float,
    hca_local_seqlen: int = 0,
    backend: Optional[BackendType] = None,
) -> torch.Tensor:
    """DeepSeek-V4 dense / HCA attention (Triton or FlyDSL backend).

    Drop-in replacement for :func:`eager_hca_attention`. The MQA case
    (``K_H == 1``) is detected from ``k.shape[1]``; the kernel broadcasts
    the single shared K / V head across the query heads.

    Dispatch contract:

    * ``compress_ratio == 0`` (dense): pass ``swa_window > 0`` and
      ``additive_mask=None`` so the kernel applies the SWA-causal mask.
    * ``compress_ratio == 128`` (HCA): pre-concatenate pool keys after
      local keys, pass the pool-only ``[Sq, P]`` additive mask, set
      ``hca_local_seqlen=Sq``, and keep ``swa_window > 0``.

    The forward kernel is selected by the attention dispatcher: the default
    backend is ``TRITON``; pass ``backend=BackendType.FLYDSL`` (or set
    ``set_attention_backend`` / ``PRIMUS_TURBO_ATTENTION_BACKEND``) to force
    FlyDSL where it is supported (gfx950, ``D == 512``, structured SWA / HCA
    masks). When FlyDSL cannot handle the inputs the dispatcher falls back to
    Triton. The backward stays on Triton in this version (design §5.4).

    Returns ``[B, H, Sq, D]`` in ``v.dtype``.
    """
    return HCAAttentionFn.apply(
        q,
        k,
        v,
        sink,
        additive_mask,
        swa_window,
        attn_dropout,
        training,
        scale,
        hca_local_seqlen,
        _backend_to_int(backend),
    )
