###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""HipKittens attention impl layer, gfx950 only.

The same position in the stack as ``attention_aiter_impl`` / ``attention_flydsl_impl``:
plain forward and backward entry points over the backend's own code, with dispatch and
autograd wiring left to the caller. The kernels and everything specific to them live under
``primus_turbo.hipkittens.attention.gfx950``; this module is what the dispatcher imports.

Everything here is a thin pass-through, so the eligibility rules and the padding stay in one
place rather than being restated at the dispatcher.
"""

from typing import Optional, Tuple

import torch

__all__ = [
    "hipkittens_attn_supported_impl",
    "flash_attn_sbhd_hipkittens_forward_impl",
    "flash_attn_sbhd_hipkittens_backward_impl",
]


def _layer():
    """Imported lazily and per call site.

    The extension is built in every arch configuration but carries gfx950 device code only,
    so on another card it imports and registers its ops and then has no kernel to launch --
    which is what the is_gfx950() check inside the layer is for. The import still stays lazy
    because PRIMUS_TURBO_BUILD_HIPKITTENS=0 can skip the extension entirely, and a dispatcher
    that cannot be imported at all is a worse failure than a backend that reports it cannot
    take the call.
    """
    from primus_turbo.hipkittens.attention import gfx950

    return gfx950


def hipkittens_attn_supported_impl(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    **kwargs,
) -> Tuple[bool, str]:
    """Whether these kernels take this call, and if not, why.

    ``q``/``k``/``v`` are SBHD. Returns the reason as well as the verdict so a pinned backend
    can report what it refused instead of silently handing the work to another one.
    """
    try:
        layer = _layer()
    except ImportError:
        return False, "hipkittens extension is not built (gfx950-only)"
    return layer.hipkittens_attn_supported(q, k, v, **kwargs)


def flash_attn_sbhd_hipkittens_forward_impl(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale: Optional[float] = None,
    causal: bool = True,
    window_size: Tuple[int, int] = (-1, -1),
    return_lse: bool = True,
):
    """SBHD forward. Returns ``(out, lse)``, or just ``out`` when ``return_lse`` is False."""
    out, lse = _layer().hipkittens_attn_forward(
        q, k, v, softmax_scale=softmax_scale, causal=causal, window_size=window_size
    )
    return (out, lse) if return_lse else out


def flash_attn_sbhd_hipkittens_backward_impl(
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    softmax_scale: Optional[float] = None,
    causal: bool = True,
    window_size: Tuple[int, int] = (-1, -1),
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """SBHD backward. Returns ``(dq, dk, dv)`` in the layouts of q, k and v."""
    return _layer().hipkittens_attn_backward(
        dout, q, k, v, out, lse,
        softmax_scale=softmax_scale, causal=causal, window_size=window_size,
    )
