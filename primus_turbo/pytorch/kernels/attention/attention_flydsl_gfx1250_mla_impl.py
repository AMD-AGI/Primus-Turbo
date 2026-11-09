###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""FlyDSL flash attention for gfx1250 (MI455X) at the DeepSeek-V3 MLA head dims.

q/k head dim 192, v head dim 128, bf16, MHA, bottom-right causal, any softmax scale; no dropout,
bias, alibi, sink or sliding window. The kernels take contiguous [b, s, h, d] tensors. b > 1 SBHD
storage (Megatron's layout) is handed over as [1, s, b*h, d], the same bytes with the batch folded
into the heads, so it is not copied. Dispatch and autograd wiring belong to the caller.

The backward runs on the HipKittens kernels (``attention_hipkittens_gfx1250_mla_impl.hk_mla_bwd``)
when PRIMUS_TURBO_MLA_BWD_IMPL is ``hk`` (``mla_bwd_impl``) and they take the call; a call they
cannot take runs on FlyDSL, with one warning per reason.
"""

from typing import Optional, Tuple

import torch

import primus_turbo.common.envs as envs
from primus_turbo.common.logger import logger
from primus_turbo.pytorch.kernels.attention import (
    attention_hipkittens_gfx1250_mla_impl as hk,
)
from primus_turbo.pytorch.kernels.attention.attention_flydsl_impl import (
    _eager_custom_op,
)

D_QK = 192
D_V = 128


def _fwd():
    # Imported on first use: gfx1250-only kernels, never loaded on other archs.
    from primus_turbo.flydsl.attention.gfx1250_mla import fwd

    return fwd


def _bwd():
    from primus_turbo.flydsl.attention.gfx1250_mla import bwd

    return bwd


def is_mla_head_dims(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> bool:
    """Whether q/k/v carry the DeepSeek-V3 MLA head dims (qk 192, v 128)."""
    return q.shape[-1] == D_QK and k.shape[-1] == D_QK and v.shape[-1] == D_V


def flydsl_gfx1250_mla_unsupported_reason(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool
) -> Optional[str]:
    """Why the gfx1250 MLA kernels cannot take these [b, s, h, d]-shaped tensors, or None.

    Covers the backward too: a training step needs both passes.
    """
    if k.device != q.device or v.device != q.device:
        return f"q/k/v must be on one device, got {q.device}/{k.device}/{v.device}"
    if q.dtype != torch.bfloat16 or k.dtype != q.dtype or v.dtype != q.dtype:
        return f"dtype must be bfloat16, got {q.dtype}/{k.dtype}/{v.dtype}"
    if any(t.dim() != 4 for t in (q, k, v)):
        return "q/k/v must be 4-D [b, s, h, d]"
    if not is_mla_head_dims(q, k, v):
        return f"head dims must be qk {D_QK} / v {D_V}, got {q.shape[-1]}/{k.shape[-1]}/{v.shape[-1]}"
    b, sq, h, _ = q.shape
    _, skv, hkv, _ = k.shape
    if k.shape[0] != b or v.shape[:3] != k.shape[:3]:
        return f"k {tuple(k.shape)} / v {tuple(v.shape)} must match each other and q's batch {b}"
    if h != hkv:
        return f"MHA only (heads_q == heads_kv), got {h} / {hkv}"
    if not causal:
        return "causal only"
    return _bwd().shape_unsupported_reason(b, sq, skv, h)


@_eager_custom_op("primus_turbo::flash_attn_flydsl_gfx1250_mla_forward")
def _forward_op(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, softmax_scale: float, causal: bool
) -> Tuple[torch.Tensor, torch.Tensor]:
    return _fwd().flash_attn_mla_fwd(q, k, v, softmax_scale, causal)


@_forward_op.register_fake
def _forward_op_fake(q, k, v, softmax_scale, causal):
    b, sq, h, _ = q.shape
    return q.new_empty((b, sq, h, D_V)), q.new_empty((b, h, sq), dtype=torch.float32)


@_eager_custom_op("primus_turbo::flash_attn_flydsl_gfx1250_mla_backward")
def _backward_op(
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    softmax_scale: float,
    causal: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return _bwd().flash_attn_mla_bwd(dout, q, k, v, out, lse, softmax_scale, causal)


@_backward_op.register_fake
def _backward_op_fake(dout, q, k, v, out, lse, softmax_scale, causal):
    return torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)


@_eager_custom_op("primus_turbo::flash_attn_hk_gfx1250_mla_backward")
def _backward_hk_op(
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    softmax_scale: float,
    causal: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return hk.hk_mla_bwd(dout, q, k, v, out, lse, softmax_scale, causal)


_backward_hk_op.register_fake(_backward_op_fake)


@torch.compiler.assume_constant_result
def mla_bwd_impl() -> str:
    """The backward kernels, "flydsl" or "hk": PRIMUS_TURBO_MLA_BWD_IMPL through envs, so at the first MLA backward,
    not at import (an invalid value raises ValueError there). torch.compile runs this while tracing the backward,
    with the forward: a compiled graph keeps the setting it was traced with."""
    return envs.PRIMUS_TURBO_MLA_BWD_IMPL


@torch.compiler.assume_constant_result  # runs while torch.compile traces, which cannot trace a logger call
def _log_fallback(reason: str) -> bool:
    logger.warning(f"PRIMUS_TURBO_MLA_BWD_IMPL=hk, but this backward runs on FlyDSL: {reason}", once=True)
    return True


def _backward_kernels(q, k, v, causal: bool):
    """``_backward_hk_op`` when the setting is "hk" and the HipKittens kernels take the call, else ``_backward_op``."""
    if mla_bwd_impl() != "hk":
        return _backward_op
    reason = hk.hk_mla_unsupported_reason(q, k, v, causal)
    if reason is None:
        return _backward_hk_op
    _log_fallback(reason)
    return _backward_op


def _scale(softmax_scale: Optional[float]) -> float:
    return float(D_QK**-0.5 if softmax_scale is None else softmax_scale)


def _to_kernel_layout(t: torch.Tensor, fold: bool) -> torch.Tensor:
    """[b, s, h, d] in any byte order -> contiguous [b, s, h, d], or [1, s, b*h, d] when folding."""
    if fold:
        b, s, h, d = t.shape
        return t.transpose(0, 1).contiguous().view(1, s, b * h, d)
    return t.contiguous()


def _from_kernel_layout(x: torch.Tensor, batch: int) -> torch.Tensor:
    """Inverse of _to_kernel_layout; a fold comes back as a [b, s, h, d] view of SBHD storage."""
    if x.shape[0] == batch:
        return x
    _, s, bh, d = x.shape
    return x.view(s, batch, bh // batch, d).transpose(0, 1)


def flash_attn_flydsl_gfx1250_mla_forward_impl(q, k, v, softmax_scale=None, causal=True):
    """Returns (out [b, sq, h, 128], lse [b, h, sq] fp32 natural log, saved).

    ``saved`` is the kernel-layout (q, k, v, out, lse) the backward impl takes; out and lse are
    views of its last two.
    """
    b, sq, h, _ = q.shape
    fold = b > 1 and q.transpose(0, 1).is_contiguous()
    qk, kk, vk = (_to_kernel_layout(t, fold) for t in (q, k, v))
    out, lse = _forward_op(qk, kk, vk, _scale(softmax_scale), bool(causal))
    return _from_kernel_layout(out, b), lse.view(b, h, sq), (qk, kk, vk, out, lse)


def flash_attn_flydsl_gfx1250_mla_backward_impl(dout, q, k, v, out, lse, softmax_scale=None, causal=True):
    """q/k/v/out/lse are the forward impl's ``saved``; dout is [b, sq, h, 128] in any byte order.

    Returns (dq, dk, dv) shaped like the forward's q/k/v.
    """
    batch = dout.shape[0]
    fold = q.shape[0] != batch  # the forward folded the batch into the heads
    causal = bool(causal)
    grads = _backward_kernels(q, k, v, causal)(
        _to_kernel_layout(dout, fold), q, k, v, out, lse, _scale(softmax_scale), causal
    )
    return tuple(_from_kernel_layout(g, batch) for g in grads)
