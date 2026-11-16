###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""FlyDSL attention for the DeepSeek-V3 MLA head dims (qk 192 / v 128) on gfx950 (MI355X).

Dense causal self-attention, bf16, Hq == Hkv, on contiguous SBHD or contiguous BSHD: the kernels are
compiled for either byte order, so neither needs a copy. Forward and backward share the LSE layout of
the other gfx950 FlyDSL kernels ([B*Sq, Hq] fp32, natural log), so the op layer saves and hands it over
the same way. Dispatch and autograd wiring belong to the caller.
"""

import math
from typing import Optional, Tuple

import torch

from primus_turbo.pytorch.kernels.attention.attention_flydsl_impl import (
    _eager_custom_op,
)

D_QK = 192
D_V = 128
# The forward runs 256-row q blocks; the backward tiles q by 32.
_SEQ_MULTIPLE = 256
_I32_MAX = 2**31 - 1


def _kernels():
    # Imported on first use: gfx950-only kernels, never loaded on other archs.
    from primus_turbo.flydsl.attention import gfx950_mla

    return gfx950_mla


def mla_bshd_native(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> bool:
    """Whether these [b, s, h, d]-shaped tensors are contiguous as they stand (bshd bytes).

    Then the bshd kernels read them in place. Anything else goes through the [s, b, h, d] view,
    which is free for sbhd storage and a copy for any other byte order.
    """
    return all(t.is_contiguous() for t in (q, k, v))


def is_mla_head_dims(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> bool:
    return q.shape[-1] == D_QK and k.shape[-1] == D_QK and v.shape[-1] == D_V


def flydsl_gfx950_mla_unsupported_reason(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool,
    window_size=(-1, -1),
    softmax_scale: Optional[float] = None,
    deterministic: bool = False,
) -> Optional[str]:
    """Why the gfx950 MLA kernels cannot take these [b, s, h, d]-shaped tensors, or None.

    Covers the backward too: a training step needs both passes.
    """
    if any(t.dim() != 4 for t in (q, k, v)):
        return "q/k/v must be 4-D [b, s, h, d]"
    if not is_mla_head_dims(q, k, v):
        return f"head dims {q.shape[-1]}/{v.shape[-1]} are not {D_QK}/{D_V}"
    if not all(t.dtype == torch.bfloat16 for t in (q, k, v)):
        return "bf16 only"
    if not causal:
        return "causal only"
    if int(window_size[0]) >= 0 or int(window_size[1]) > 0:
        return f"no sliding window, got {tuple(window_size)}"
    if softmax_scale is not None and abs(float(softmax_scale) - 1.0 / math.sqrt(D_QK)) > 1e-6:
        return "the forward bakes softmax_scale = 1/sqrt(192)"
    if deterministic:
        return "dQ is accumulated with atomics; no deterministic path"
    B, Sq, Hq, _ = q.shape
    if k.shape[0] != B or v.shape[:3] != k.shape[:3]:
        return f"k {tuple(k.shape)} / v {tuple(v.shape)} must match each other and q's batch {B}"
    Skv, Hkv = k.shape[1], k.shape[2]
    if Sq != Skv:
        return "self-attention only (Sq == Skv)"
    if Hq != Hkv:
        return "Hq == Hkv only"
    if Sq % _SEQ_MULTIPLE != 0:
        return f"sequence length must be a multiple of {_SEQ_MULTIPLE}"
    if B * Sq * Hq * D_QK > _I32_MAX:
        return "q is too large for the int32 buffer offsets"
    return None


@_eager_custom_op("primus_turbo::flash_attn_flydsl_gfx950_mla_forward")
def _forward_op(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    return _kernels().mla_fwd_sbhd(q, k, v, causal=True, return_lse=True)


@_forward_op.register_fake
def _forward_op_fake(q, k, v):
    Sq, B, Hq, _ = q.shape
    return q.new_empty((Sq, B, Hq, D_V)), q.new_empty((B * Sq, Hq), dtype=torch.float32)


@_eager_custom_op("primus_turbo::flash_attn_flydsl_gfx950_mla_backward")
def _backward_op(
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    softmax_scale: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return _kernels().mla_bwd_sbhd(dout, q, k, v, out, lse, softmax_scale=softmax_scale)


@_backward_op.register_fake
def _backward_op_fake(dout, q, k, v, out, lse, softmax_scale):
    return torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)


def flash_attn_flydsl_gfx950_mla_forward_impl(q, k, v):
    """q/k [Sq,B,H,192], v [Sq,B,H,128] (sbhd). Returns O [Sq,B,H,128] and LSE [B*Sq,H] fp32."""
    q, k, v = (t.contiguous() for t in (q, k, v))
    return _forward_op(q, k, v)


def flash_attn_flydsl_gfx950_mla_backward_impl(dout, q, k, v, out, lse, softmax_scale=None):
    """sbhd tensors as in the forward, ``lse`` [B*Sq,H] as the forward returned it."""
    scale = 1.0 / math.sqrt(D_QK) if softmax_scale is None else float(softmax_scale)
    return _backward_op(dout.contiguous(), q, k, v, out, lse, scale)


@_eager_custom_op("primus_turbo::flash_attn_flydsl_gfx950_mla_bshd_forward")
def _forward_bshd_op(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    return _kernels().mla_fwd_bshd(q, k, v, causal=True, return_lse=True)


@_forward_bshd_op.register_fake
def _forward_bshd_op_fake(q, k, v):
    B, Sq, Hq, _ = q.shape
    return q.new_empty((B, Sq, Hq, D_V)), q.new_empty((B * Sq, Hq), dtype=torch.float32)


@_eager_custom_op("primus_turbo::flash_attn_flydsl_gfx950_mla_bshd_backward")
def _backward_bshd_op(
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    softmax_scale: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return _kernels().mla_bwd_bshd(dout, q, k, v, out, lse, softmax_scale=softmax_scale)


@_backward_bshd_op.register_fake
def _backward_bshd_op_fake(dout, q, k, v, out, lse, softmax_scale):
    return torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)


def flash_attn_flydsl_gfx950_mla_forward_bshd_impl(q, k, v):
    """q/k [B,Sq,H,192], v [B,Sq,H,128], contiguous. Returns O [B,Sq,H,128] and LSE [B*Sq,H] fp32."""
    return _forward_bshd_op(q, k, v)


def flash_attn_flydsl_gfx950_mla_backward_bshd_impl(dout, q, k, v, out, lse, softmax_scale=None):
    """bshd tensors as in the forward, ``lse`` [B*Sq,H] as the forward returned it."""
    scale = 1.0 / math.sqrt(D_QK) if softmax_scale is None else float(softmax_scale)
    return _backward_bshd_op(dout.contiguous(), q, k, v, out, lse, scale)
