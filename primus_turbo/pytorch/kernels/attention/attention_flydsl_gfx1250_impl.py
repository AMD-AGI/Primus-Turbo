###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""FlyDSL flash-attention forward/backward for gfx1250 (MI455X), dense [b, s, h, d] tensors.

bf16, head_dim 128, GQA (Hq / Hkv of 1, 2, 4, 8 or 16), Sq a multiple of 64 and Skv of 32,
q and k each at most 1 GiB, bottom-right causal or full attention, no dropout / bias / alibi /
sink / sliding window; ``interface.unsupported_reason`` of the kernel package has the exact
rules. The kernels address BSHD; any other byte order is made contiguous first. Mirrors the
``attention_triton_impl`` layer -- dispatch and autograd wiring belong to the caller.

The kernels need a flydsl that satisfies ``FLYDSL_REQUIREMENT`` (``>=0.3.4.1,<0.3.5``, in
``primus_turbo/flydsl/attention/gfx1250/flydsl_version.py``). Under any other flydsl the gate
declines every call before any kernel module is imported, and it also declines every call
when the kernel package fails to import; either reason is logged once. A declined call goes
to the dispatcher's fallback, which on gfx1250 tries AITER first, and AITER's CK backward
fails there: a training call this gate declines needs the TRITON backend pinned
(``PRIMUS_TURBO_ATTN_BACKEND=TRITON``).
"""

from typing import Optional

import torch

from primus_turbo.common.logger import logger
from primus_turbo.pytorch.kernels.attention.attention_flydsl_impl import (
    _eager_custom_op,
)

# The kernel package's interface module once imported or, when it cannot be, the reason (a
# str); None until the first gate call. Resolved once: Python does not cache a failed import,
# so a broken flydsl would otherwise be imported again on every attention call.
_INTERFACE = None


def _import_interface():
    # Imported on first use: the package is wave32 WMMA code for gfx1250 only.
    from primus_turbo.flydsl.attention.gfx1250 import interface

    return interface


def _load_interface():
    """The interface module, or why the kernels cannot be used (a str); never raises."""
    try:
        # A stdlib-only module: asking it imports no kernel code.
        from primus_turbo.flydsl.attention.gfx1250.flydsl_version import (
            flydsl_unavailable_reason,
        )

        reason = flydsl_unavailable_reason()
        return _import_interface() if reason is None else reason
    except Exception as exc:  # noqa: BLE001 -- a flydsl that cannot load the kernels is a decline
        return f"the gfx1250 attention kernels failed to import: {type(exc).__name__}: {exc}"


@torch.compiler.assume_constant_result
def _unavailable_reason() -> Optional[str]:
    """None when the kernels are importable, else why not; resolved on the first call.

    Marked constant for torch.compile, whose Dynamo then calls it instead of tracing it (the
    gate runs inside compiled regions): the answer cannot change within a process, and the
    import, its failure and the log line are not traceable.
    """
    global _INTERFACE
    if _INTERFACE is None:
        _INTERFACE = _load_interface()
        if isinstance(_INTERFACE, str):
            logger.warning(f"gfx1250 FlyDSL attention is disabled: {_INTERFACE}", once=True)
    return _INTERFACE if isinstance(_INTERFACE, str) else None


def _interface():
    """The kernel package's interface module; ImportError with the reason when it cannot load."""
    reason = _unavailable_reason()
    if reason is not None:
        raise ImportError(reason)
    return _INTERFACE


def flydsl_gfx1250_unsupported_reason(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool
) -> Optional[str]:
    """Why the gfx1250 kernels cannot take these [b, s, h, d]-shaped tensors, or None.

    A missing flydsl, one outside ``flydsl_version.FLYDSL_REQUIREMENT`` (found before any
    kernel module is imported) or a kernel package that fails to import is a reason too:
    the gate declines, it never raises.
    """
    reason = _unavailable_reason()
    if reason is not None:
        return reason
    return _INTERFACE.unsupported_reason(q.shape, k.shape, v.shape, q.dtype, causal)


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
