###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""HipKittens flash-attention forward/backward for gfx1250 (MI455X), head dim 128, dense [b, s, h, d]
tensors: what the HIPKITTENS backend runs on gfx1250.

The HIP kernels (``primus_turbo/hipkittens/gfx1250_attn``, ``impl.hk_attn_fwd`` / ``hk_attn_bwd``) take a
subset of what the FlyDSL gfx1250 head-dim-128 kernels take (bf16, causal, GQA groups of 4, sequence
multiples of 64 / 128, ...), with the same outputs: contiguous BSHD out / dq / dk / dv and lse
[B, Hq, Sq] fp32 natural log. The backend therefore accepts exactly the FlyDSL gfx1250 head-dim-128
calls, and each custom op below runs its direction on HK when ``hk_unsupported_reason`` allows and on
the FlyDSL gfx1250 kernels otherwise, logging one warning per reason. The decision is made inside the
op, at run time, so a compiled graph holds one op per direction and never traces the gate. Cases that
run FlyDSL: deterministic=True backward (HK adds dQ with atomics; FlyDSL's deterministic two-kernel
backward runs), shapes outside HK's rules, code objects that are not built / not pinned by
``manifest.json``, a kernel module that is missing or fails to import, and the mixed-mode settings
``PRIMUS_TURBO_HK_FWD=flydsl`` / ``PRIMUS_TURBO_HK_BWD=flydsl`` (read at import, like the forward
module choice ``pp`` / ``v0``; the lse convention is shared, so the directions can mix).

Importing this module imports neither the HK package nor the FlyDSL kernels; the first call does.
"""

from typing import Optional

import torch

from primus_turbo.common.logger import logger
from primus_turbo.pytorch.kernels.attention.attention_flydsl_gfx1250_impl import (
    _interface as _flydsl_interface,
)
from primus_turbo.pytorch.kernels.attention.attention_flydsl_impl import (
    _eager_custom_op,
)

__all__ = [
    "flash_attn_hk_gfx1250_backward_impl",
    "flash_attn_hk_gfx1250_forward_impl",
    "hk_gfx1250_unsupported_reason",
]

_DIRECTION_NAME = {"fwd": "forward", "bwd": "backward"}

# The HK package's impl module once imported or, when it cannot be, the reason (a str); None until
# the first call. Resolved once: Python does not cache a failed import.
_IMPL = None


def _import_impl():
    from primus_turbo.hipkittens.gfx1250_attn import impl

    return impl


def _load_impl():
    """The HK impl module, or why it cannot be used (a str); never raises."""
    global _IMPL
    if _IMPL is None:
        try:
            _IMPL = _import_impl()
        except Exception as exc:  # noqa: BLE001 -- an HK package that cannot load is a fallback
            _IMPL = f"the gfx1250 HipKittens attention package failed to import: {type(exc).__name__}: {exc}"
    return _IMPL


def hk_gfx1250_unsupported_reason(
    direction: str,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool,
    deterministic: bool = False,
    softmax_scale: Optional[float] = None,
) -> Optional[str]:
    """Why HK cannot run ``direction`` ("fwd" / "bwd") of this call, or None; never raises. A call it
    declines runs on the FlyDSL gfx1250 kernels, which the backend's gate already accepted."""
    impl = _load_impl()
    if isinstance(impl, str):
        return impl
    return impl.hk_unsupported_reason(direction, q, k, v, causal, deterministic, softmax_scale=softmax_scale)


def _fallback_warning(direction, reason) -> None:
    name = _DIRECTION_NAME[direction]
    logger.warning(f"HIPKITTENS attention on gfx1250: {name} runs on FlyDSL ({reason})", once=True)


def _decide(direction, q, k, v, causal, deterministic=False, softmax_scale=None) -> bool:
    """True to run HK; else logs (once per reason) that FlyDSL runs this direction."""
    reason = hk_gfx1250_unsupported_reason(direction, q, k, v, causal, deterministic, softmax_scale)
    if reason is not None:
        _fallback_warning(direction, reason)
        return False
    logger.info(f"HIPKITTENS attention on gfx1250: {_IMPL.describe(direction)}", once=True)
    return True


# hk_attn_fwd / hk_attn_bwd raise NotImplementedError only before anything is launched (what the gate
# cannot see from q / k / v: do / o / lse alignment, a pinned code object replaced since the gate's
# check), so that call still runs on FlyDSL.


def _run_forward(q, k, v, softmax_scale, causal):
    if _decide("fwd", q, k, v, causal, softmax_scale=softmax_scale):
        try:
            return _IMPL.hk_attn_fwd(q, k, v, softmax_scale, causal)
        except NotImplementedError as exc:
            _fallback_warning("fwd", exc)
    return _flydsl_interface().flash_attn_fwd(q, k, v, softmax_scale, causal)


def _run_backward(dout, q, k, v, out, lse, softmax_scale, causal, deterministic):
    if _decide("bwd", q, k, v, causal, deterministic, softmax_scale):
        try:
            return _IMPL.hk_attn_bwd(dout, q, k, v, out, lse, softmax_scale, causal, False)
        except NotImplementedError as exc:
            _fallback_warning("bwd", exc)
    return _flydsl_interface().flash_attn_bwd(dout, q, k, v, out, lse, softmax_scale, causal, deterministic)


@_eager_custom_op("primus_turbo::flash_attn_hk_gfx1250_forward")
def _forward(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, softmax_scale: float, causal: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    return _run_forward(q, k, v, softmax_scale, causal)


@_forward.register_fake
def _(q, k, v, softmax_scale, causal):
    b, sq, hq, _ = q.shape
    return q.new_empty((b, sq, hq, v.shape[-1])), q.new_empty((b, hq, sq), dtype=torch.float32)


@_eager_custom_op("primus_turbo::flash_attn_hk_gfx1250_backward")
def _backward(
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    softmax_scale: float,
    causal: bool,
    deterministic: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return _run_backward(dout, q, k, v, out, lse, softmax_scale, causal, deterministic)


@_backward.register_fake
def _(dout, q, k, v, out, lse, softmax_scale, causal, deterministic):
    return torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)


def _scale(softmax_scale: Optional[float], head_dim: int) -> float:
    return float(head_dim**-0.5 if softmax_scale is None else softmax_scale)


def flash_attn_hk_gfx1250_forward_impl(q, k, v, softmax_scale=None, causal=True):
    """Returns (out [b, sq, hq, d] contiguous, lse [b, hq, sq] fp32 natural log), from HK or FlyDSL."""
    q, k, v = (t.contiguous() for t in (q, k, v))
    return _forward(q, k, v, _scale(softmax_scale, q.shape[-1]), bool(causal))


def flash_attn_hk_gfx1250_backward_impl(
    dout, q, k, v, out, lse, softmax_scale=None, causal=True, deterministic=False
):
    """Returns (dq, dk, dv), contiguous [b, s, h, d]; dk/dv have k's head count.

    q/k/v/out/lse are what the forward saw and returned (either family's); only dout may arrive
    strided. deterministic=True runs FlyDSL's deterministic backward.
    """
    return _backward(
        dout.contiguous(),
        q,
        k,
        v,
        out,
        lse,
        _scale(softmax_scale, q.shape[-1]),
        bool(causal),
        bool(deterministic),
    )
