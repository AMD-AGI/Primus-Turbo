###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""FlyDSL flash-attention forward/backward for gfx1250 (MI455X), dense [b, s, h, d] tensors.

bf16, head_dim 128, GQA (Hq a multiple of Hkv), bottom-right causal or full attention, no
dropout / bias / alibi / sink / sliding window. The kernels address BSHD; any other byte
order is made contiguous first. Mirrors the ``attention_triton_impl`` layer -- dispatch and
autograd wiring belong to the caller.
"""

from typing import Optional

import torch

from primus_turbo.pytorch.kernels.attention.attention_flydsl_impl import (
    _eager_custom_op,
)


def _interface():
    # Imported on first use: the package is wave32 WMMA code for gfx1250 only.
    from primus_turbo.flydsl.attention.gfx1250 import interface

    return interface


def flydsl_gfx1250_unsupported_reason(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool
) -> Optional[str]:
    """Why the gfx1250 kernels cannot take these [b, s, h, d]-shaped tensors, or None."""
    try:
        iface = _interface()
    except ImportError as exc:  # flydsl missing or incompatible
        return f"flydsl gfx1250 attention is unavailable: {exc}"
    return iface.unsupported_reason(q.shape, k.shape, v.shape, q.dtype, causal)


@_eager_custom_op("primus_turbo::flash_attn_flydsl_gfx1250_forward")
def _forward(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, softmax_scale: float, causal: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    return _interface().flash_attn_fwd(q, k, v, softmax_scale, causal)


@_forward.register_fake
def _(q, k, v, softmax_scale, causal):
    b, sq, hq, _ = q.shape
    return q.new_empty((b, sq, hq, v.shape[-1])), q.new_empty((b, hq, sq), dtype=torch.float32)


@_eager_custom_op("primus_turbo::flash_attn_flydsl_gfx1250_backward")
def _backward(
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    softmax_scale: float,
    causal: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return _interface().flash_attn_bwd(dout, q, k, v, out, lse, softmax_scale, causal)


@_backward.register_fake
def _(dout, q, k, v, out, lse, softmax_scale, causal):
    return torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)


def _scale(softmax_scale: Optional[float], head_dim: int) -> float:
    return float(head_dim**-0.5 if softmax_scale is None else softmax_scale)


def flash_attn_flydsl_gfx1250_forward_impl(q, k, v, softmax_scale=None, causal=True):
    """Returns (out [b, sq, hq, d] contiguous, lse [b, hq, sq] fp32 natural log)."""
    q, k, v = (t.contiguous() for t in (q, k, v))
    return _forward(q, k, v, _scale(softmax_scale, q.shape[-1]), bool(causal))


def flash_attn_flydsl_gfx1250_backward_impl(dout, q, k, v, out, lse, softmax_scale=None, causal=True):
    """Returns (dq, dk, dv), contiguous [b, s, h, d]; dk/dv have k's head count.

    q/k/v/out/lse are what the forward saw and returned; only dout may arrive strided.
    """
    return _backward(dout.contiguous(), q, k, v, out, lse, _scale(softmax_scale, q.shape[-1]), bool(causal))
