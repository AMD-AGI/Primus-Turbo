###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""FlyDSL flash attention for gfx1250 (MI455X) at DeepSeek-V3 MLA head dims.

q/k head dim 192, v head dim 128, bf16, MHA (Hq == Hkv), bottom-right causal, runtime softmax
scale; no dropout / bias / alibi / sink / sliding window. Full (non-causal) attention is refused:
the forward's non-causal variant at head dim 192 spills VGPRs to scratch (8 VGPRs, 36 B), and a
spilling FlyDSL build is not launched.
Forward: ``primus_turbo/flydsl/attention/gfx1250_mla_fwd``; backward:
``primus_turbo/flydsl/attention/gfx1250_mla_bwd``. Mirrors ``attention_flydsl_gfx1250_impl``
(head dim 128) -- dispatch and autograd wiring belong to the caller.

Layout. The kernels address contiguous BSHD ``[B, S, H, D]``. Megatron hands over
``[b, s, h, d]`` views of contiguous SBHD storage; those bytes ARE the contiguous BSHD tensor
``[1, s, b*h, d]`` (attention is independent per (batch, head), and the kv head of folded q head
``b*Hq + h`` is ``b*Hkv + h // G``), so for b > 1 the batch is folded into the heads and nothing
is copied. Outputs come back as the same kind of view, lse as a ``[b, h, s]`` view of the
kernel's ``[1, b*h, s]``. b == 1 is contiguous BSHD either way; any other byte order is copied.
"""

from typing import Optional, Tuple

import torch

from primus_turbo.pytorch.kernels.attention.attention_flydsl_impl import (
    _eager_custom_op,
)

D_QK = 192
D_V = 128


def is_mla_head_dims(q: torch.Tensor, k: Optional[torch.Tensor], v: Optional[torch.Tensor]) -> bool:
    """The (192, 128) head-dim pair this backend owns; every other pair keeps its old route."""
    return (
        k is not None and v is not None and q.shape[-1] == D_QK and k.shape[-1] == D_QK and v.shape[-1] == D_V
    )


def _bwd_kernels():
    # Imported on first use (never on other archs): wave32 WMMA code for gfx1250 that needs
    # flydsl 0.3.4.x; the package raises ImportError on any other flydsl.
    from primus_turbo.flydsl.attention.gfx1250_mla_bwd import impl

    return impl


def _fwd_interface():
    from primus_turbo.flydsl.attention.gfx1250_mla_fwd import interface

    return interface


def _flydsl_unavailable_reason() -> Optional[str]:
    # stdlib-only module, imported here so importing this adapter never imports flydsl
    from primus_turbo.flydsl.attention.gfx1250_mla_version import (
        flydsl_unavailable_reason,
    )

    return flydsl_unavailable_reason()


def flydsl_gfx1250_mla_unsupported_reason(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool
) -> Optional[str]:
    """Why the gfx1250 MLA kernels cannot take these [b, s, h, d]-shaped tensors, or None.

    Covers forward AND backward (the autograd path always needs both). The sequence rules are
    the backward's: k_dkdv64 consumes query pairs of 32 rows and kv blocks of
    DKDV_NW * BLOCK_KV (two waves of BLOCK_KV rows each), the dQ kernels (k_dqg96 / k_dqg) query
    tiles of a multiple of DQ_BQW and kv steps of KV_STEP, k_delta rows in groups of ROWS_DELTA.
    A missing flydsl or one outside ``gfx1250_mla_version.FLYDSL_REQUIREMENT`` is a reason too
    (checked before any kernel module is imported), so the call falls back to another backend.
    """
    if q.dtype != torch.bfloat16 or k.dtype != q.dtype or v.dtype != q.dtype:
        return f"dtype must be bfloat16, got {q.dtype}/{k.dtype}/{v.dtype}"
    if any(t.dim() != 4 for t in (q, k, v)):
        return "q/k/v must be 4-D [b, s, h, d]"
    if not is_mla_head_dims(q, k, v):
        return f"head dims must be qk {D_QK} / v {D_V}, got {q.shape[-1]}/{k.shape[-1]}/{v.shape[-1]}"
    b, sq, hq, _ = q.shape
    bk, skv, hkv, _ = k.shape
    if bk != b or tuple(v.shape[:3]) != (bk, skv, hkv):
        return f"k {tuple(k.shape)} / v {tuple(v.shape)} must match each other and q's batch {b}"
    if hq != hkv:
        return f"MHA only (heads_q == heads_kv), got {hq} / {hkv}"
    if not causal:
        return "causal only: the non-causal forward variant spills at head dim 192"
    reason = _flydsl_unavailable_reason()
    if reason is not None:
        return reason
    try:
        kern = _bwd_kernels()._k
    except ImportError as exc:  # a flydsl that passes the version check but cannot load the kernels
        return f"flydsl gfx1250 MLA attention is unavailable: {exc}"
    q_mult = max(64, kern.DQ_BQW)
    kv_mult = max(kern.KV_STEP, kern.BLOCK_KV * kern.DKDV_NW)
    if sq == 0 or sq % q_mult:
        return f"seqlen_q must be a positive multiple of {q_mult}, got {sq}"
    if skv == 0 or skv % kv_mult:
        return f"seqlen_kv must be a positive multiple of {kv_mult}, got {skv}"
    if sq > skv:
        # Bottom-right causal leaves the first sq - skv queries with no key at all.
        return f"causal needs seqlen_q <= seqlen_kv, got {sq} > {skv}"
    if (b * sq * hq) % kern.ROWS_DELTA:
        return f"batch*seqlen_q*heads_q must be a multiple of {kern.ROWS_DELTA}"
    # Byte extents the backward computes in 32-bit arithmetic or hard-codes as descriptor sizes.
    if (
        b * sq * hq * D_QK * 2 > (1 << 30)
        or b * skv * hkv * D_QK * 2 >= (1 << 31)
        or b * hq * sq * 4 > (1 << 28)
    ):
        return "q/k/lse too large for the backward's 32-bit descriptor extents"
    return None


@_eager_custom_op("primus_turbo::flash_attn_flydsl_gfx1250_mla_forward")
def _forward(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, softmax_scale: float, causal: bool
) -> Tuple[torch.Tensor, torch.Tensor]:
    return _fwd_interface().flash_attn_fwd(q, k, v, softmax_scale, causal)


@_forward.register_fake
def _(q, k, v, softmax_scale, causal):
    b, sq, hq, _ = q.shape
    return q.new_empty((b, sq, hq, v.shape[-1])), q.new_empty((b, hq, sq), dtype=torch.float32)


@_eager_custom_op("primus_turbo::flash_attn_flydsl_gfx1250_mla_backward")
def _backward(
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    softmax_scale: float,
    causal: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return _bwd_kernels().flydsl_attn_bwd(dout, q, k, v, out, lse, softmax_scale, causal)


@_backward.register_fake
def _(dout, q, k, v, out, lse, softmax_scale, causal):
    return torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)


def _scale(softmax_scale: Optional[float]) -> float:
    return float(D_QK**-0.5 if softmax_scale is None else softmax_scale)


def sbhd_fold(q: torch.Tensor) -> bool:
    """True when the batch can be folded into the heads: b > 1 and SBHD storage."""
    return q.shape[0] > 1 and q.transpose(0, 1).is_contiguous()


def to_kernel_layout(t: torch.Tensor, fold: bool) -> torch.Tensor:
    """[b, s, h, d] in any byte order -> the contiguous tensor the kernels take."""
    if fold:
        s, b, h, d = t.shape[1], t.shape[0], t.shape[2], t.shape[3]
        return t.transpose(0, 1).contiguous().view(1, s, b * h, d)
    return t.contiguous()


def from_kernel_layout(x: torch.Tensor, fold: bool, batch: int) -> torch.Tensor:
    """The kernels' [1, s, b*h, d] (fold) / [b, s, h, d] back to a [b, s, h, d] tensor or view."""
    if fold:
        _, s, bh, d = x.shape
        return x.view(s, batch, bh // batch, d).transpose(0, 1)
    return x


def flash_attn_flydsl_gfx1250_mla_forward_impl(q, k, v, softmax_scale=None, causal=True):
    """Returns (out [b, sq, hq, 128], lse [b, hq, sq] fp32 natural log, saved, fold).

    ``saved`` = the kernel-layout (q, k, v, out, lse) the backward impl takes, ``fold`` the
    layout flag it needs; out/lse are views of saved[3]/saved[4].
    """
    fold = sbhd_fold(q)
    b, sq, hq, _ = q.shape
    qk, kk, vk = (to_kernel_layout(t, fold) for t in (q, k, v))
    ok, lsek = _forward(qk, kk, vk, _scale(softmax_scale), bool(causal))
    return from_kernel_layout(ok, fold, b), lsek.view(b, hq, sq), (qk, kk, vk, ok, lsek), fold


def flash_attn_flydsl_gfx1250_mla_backward_impl(
    dout, q, k, v, out, lse, fold, batch, softmax_scale=None, causal=True
):
    """q/k/v/out/lse = the forward impl's ``saved`` (kernel layout); dout is [b, sq, hq, 128] in
    any byte order. Returns (dq, dk, dv) shaped like the forward's q/k/v (views of SBHD storage
    when ``fold``)."""
    dq, dk, dv = _backward(
        to_kernel_layout(dout, fold), q, k, v, out, lse, _scale(softmax_scale), bool(causal)
    )
    return tuple(from_kernel_layout(g, fold, batch) for g in (dq, dk, dv))
