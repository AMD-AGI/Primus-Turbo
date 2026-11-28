###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Launch, validation and backend selection for fused QK RMSNorm + RoPE.

Two backends with the same numerics:

* FlyDSL (gfx942 / gfx950): GPT-OSS head_dim 64 and Qwen3 head_dim 128, weighted norms, RoPE, no gate, shapes
  its row tiling covers.
* Triton (any arch): also weightless norms, NoPE (no RoPE), a gate block per query group (Megatron's
  ``attention_output_gate``), any heads per group and any power-of-2 head_dim >= 16.

Without an explicit backend, FlyDSL takes every input it supports and Triton the rest.
"""

from __future__ import annotations

import weakref
from typing import Optional, Sequence

import torch
import triton

from primus_turbo.pytorch.core.backend import BackendType
from primus_turbo.triton.rope.qk_rmsnorm_rope_kernel import (
    qk_rmsnorm_rope_bwd_kernel,
    qk_rmsnorm_rope_fwd_kernel,
)

try:
    from primus_turbo.flydsl.rope.qk_rmsnorm_rope_kernel import (
        QK_RMSNORM_ROPE_HEAD_DIMS,
        _check_row_tileable,
        flydsl_qkv_rmsnorm_rope_backward,
        flydsl_qkv_rmsnorm_rope_forward,
    )

    _HAVE_FLYDSL = True
except ImportError:
    QK_RMSNORM_ROPE_HEAD_DIMS = (64, 128)
    _HAVE_FLYDSL = False

_SUPPORTED_ARCHS = ("gfx942", "gfx950")
_BACKENDS = (BackendType.FLYDSL, BackendType.TRITON)

# Triton: query groups of one token per program (the largest that divides NG) and warps per program
_TRITON_GROUPS_PER_PROG = (4, 2, 1)
_TRITON_NUM_WARPS = 2


def _pow2(n: int) -> bool:
    return n > 0 and n & (n - 1) == 0


def _layout(qkv, split_sizes):
    """``(NPG, D, gated)`` of ``[q, k, v]`` / ``[q, gate, k, v]`` widths, or a reason string."""
    if len(split_sizes) not in (3, 4):
        return f"split_sizes must be [q,k,v] or [q,gate,k,v], got {list(split_sizes)}"
    gated = len(split_sizes) == 4
    q_size, k_size, v_size = split_sizes[0], split_sizes[-2], split_sizes[-1]
    D = k_size
    if D <= 0 or q_size <= 0 or q_size % D or v_size != D:
        return f"expected q multiple of {D} and k=v={D}, got {list(split_sizes)}"
    if gated and split_sizes[1] != q_size:
        return f"the gate width must equal the q width, got {list(split_sizes)}"
    if qkv.shape[-1] != sum(split_sizes):
        return f"packed width {qkv.shape[-1]} != the split sum {sum(split_sizes)}"
    return q_size // D, D, gated


def _common_error(qkv, q_gamma, k_gamma, freqs, split_sizes):
    if qkv.ndim != 4:
        return f"qkv must be [S,B,NG,(NPG+2)*D] or [S,B,NG,(2*NPG+2)*D], got {tuple(qkv.shape)}"
    if qkv.dtype != torch.bfloat16 or not qkv.is_contiguous():
        return f"qkv must be contiguous bfloat16, got dtype={qkv.dtype} contiguous={qkv.is_contiguous()}"
    if not qkv.is_cuda:
        return "qkv must be a CUDA tensor"
    if min(qkv.shape[:3]) <= 0:
        return f"qk_rmsnorm_rope: S/B/NG must be positive, got {tuple(qkv.shape[:3])}"
    layout = _layout(qkv, split_sizes)
    if isinstance(layout, str):
        return layout
    _, D, _ = layout
    if (q_gamma is None) != (k_gamma is None):
        return "q_gamma and k_gamma must both be given or both be None"
    for g in (q_gamma, k_gamma):
        if g is None:
            continue
        if g.shape != (D,):
            return f"q/k gamma must both be [{D}], got {tuple(q_gamma.shape)}/{tuple(k_gamma.shape)}"
        if g.dtype != torch.bfloat16 or not g.is_contiguous():
            return "q/k gamma must be contiguous bfloat16"
        if g.device != qkv.device:
            return "qkv, q/k gamma, and freqs must be on the same device"
    if freqs is not None:
        if freqs.shape != (qkv.shape[0], 1, 1, D):
            return f"freqs must be [{qkv.shape[0]},1,1,{D}], got {tuple(freqs.shape)}"
        if freqs.dtype != torch.float32 or not freqs.is_contiguous():
            return f"freqs must be contiguous float32, got dtype={freqs.dtype} contiguous={freqs.is_contiguous()}"
        if freqs.device != qkv.device:
            return "qkv, q/k gamma, and freqs must be on the same device"
    return None


def _flydsl_error(qkv, q_gamma, k_gamma, freqs, split_sizes):
    if not _HAVE_FLYDSL:
        return "the FlyDSL backend is not installed"
    npg, D, gated = _layout(qkv, split_sizes)
    if gated:
        return "the FlyDSL backend has no gate block"
    if q_gamma is None:
        return "the FlyDSL backend needs q/k gamma"
    if freqs is None:
        return "the FlyDSL backend needs freqs (RoPE)"
    if D not in QK_RMSNORM_ROPE_HEAD_DIMS:
        return f"head_dim (k width) must be one of {QK_RMSNORM_ROPE_HEAD_DIMS}, got {list(split_sizes)}"
    try:
        _check_row_tileable(qkv.shape[0], qkv.shape[1], qkv.shape[2], npg, D)
    except ValueError as exc:
        return str(exc)
    arch = str(torch.cuda.get_device_properties(qkv.device).gcnArchName).split(":", 1)[0]
    if arch not in _SUPPORTED_ARCHS:
        return f"fused QK RMSNorm + RoPE (FlyDSL) requires one of {_SUPPORTED_ARCHS}, got {arch}"
    return None


def _triton_error(qkv, q_gamma, k_gamma, freqs, split_sizes):
    _, D, _ = _layout(qkv, split_sizes)
    if not (_pow2(D) and D >= 16):
        return (
            f"head_dim (k width) must be a power of 2 >= 16 for the Triton backend, got {list(split_sizes)}"
        )
    return None


def select_backend(qkv, q_gamma, k_gamma, freqs, split_sizes, backend: Optional[BackendType] = None):
    """``(backend, None)`` for the backend that runs this input, or ``(None, reason)``."""
    why = _common_error(qkv, q_gamma, k_gamma, freqs, split_sizes)
    if why is not None:
        return None, why
    if backend is not None and backend not in _BACKENDS:
        return None, f"backend must be one of {[b.name for b in _BACKENDS]}, got {backend}"
    reasons = []
    for b in (backend,) if backend is not None else _BACKENDS:
        check = _flydsl_error if b == BackendType.FLYDSL else _triton_error
        why = check(qkv, q_gamma, k_gamma, freqs, split_sizes)
        if why is None:
            return b, None
        reasons.append(why)
    return None, "; ".join(reasons)


def qk_rmsnorm_rope_shape_error(
    qkv: torch.Tensor,
    q_gamma: Optional[torch.Tensor],
    k_gamma: Optional[torch.Tensor],
    freqs: Optional[torch.Tensor],
    split_sizes: Sequence[int],
    backend: Optional[BackendType] = None,
) -> str | None:
    """Return why this input is unsupported, or ``None`` when a backend accepts it."""
    return select_backend(qkv, q_gamma, k_gamma, freqs, split_sizes, backend)[1]


# Megatron passes the same rotary table to every layer: build the fp32 cos / sin once per table and stream.
_ROTARY_TABLE_CACHE = {}


def _cos_sin(freqs):
    S, D = freqs.shape[0], freqs.shape[-1]
    key = (freqs.device.index, torch.cuda.current_stream(freqs.device).cuda_stream)
    cached = _ROTARY_TABLE_CACHE.get(key)
    if cached is not None and cached[0]() is freqs and cached[1] == freqs._version:
        return cached[2], cached[3]
    cosine = freqs.cos().reshape(S, D)
    sine = freqs.sin().reshape(S, D)
    _ROTARY_TABLE_CACHE[key] = (weakref.ref(freqs), freqs._version, cosine, sine)
    return cosine, sine


def _groups_per_prog(NG: int) -> int:
    return next(gp for gp in _TRITON_GROUPS_PER_PROG if NG % gp == 0)


def _triton_fwd(qkv, q_gamma, k_gamma, freqs, npg, D, gated, eps):
    S, B, NG, _ = qkv.shape
    gp = _groups_per_prog(NG)
    q = torch.empty(S, B, NG * npg, D, device=qkv.device, dtype=qkv.dtype)
    k = torch.empty(S, B, NG, D, device=qkv.device, dtype=qkv.dtype)
    cos, sin = _cos_sin(freqs) if freqs is not None else (qkv, qkv)
    has_w = q_gamma is not None
    with torch.cuda.device(qkv.device):
        qk_rmsnorm_rope_fwd_kernel[(S * B * NG // gp,)](
            qkv, q_gamma if has_w else qkv, k_gamma if has_w else qkv, cos, sin, q, k, B, float(eps),
            NG=NG, NPG=npg, NPG_P2=triton.next_power_of_2(npg), D=D, HALF=D // 2, GATED=gated, HAS_W=has_w,
            ROT=freqs is not None, GP=gp, num_warps=_TRITON_NUM_WARPS,
        )  # fmt: skip
    return q, k


def _rows_ready(t, shape, like):
    if t is None:
        return torch.zeros(shape, device=like.device, dtype=like.dtype)
    # the kernel reads the heads of one (s, b) as one contiguous run
    return t if t.stride(-1) == 1 and t[0, 0].is_contiguous() else t.contiguous()


def _triton_bwd(dq, dk, dv, dgate, qkv, q_gamma, k_gamma, freqs, npg, D, gated, eps):
    S, B, NG, _ = qkv.shape
    gp = _groups_per_prog(NG)
    dq = _rows_ready(dq, (S, B, NG * npg, D), qkv)
    dk = _rows_ready(dk, (S, B, NG, D), qkv)
    dv = _rows_ready(dv, (S, B, NG, D), qkv)
    dgate = _rows_ready(dgate, (S, B, NG, npg * D), qkv) if gated else dv
    dqkv = torch.empty_like(qkv)
    has_w = q_gamma is not None
    programs = S * B * NG // gp
    if has_w:
        dqg = torch.empty(programs, D, device=qkv.device, dtype=torch.float32)
        dkg = torch.empty(programs, D, device=qkv.device, dtype=torch.float32)
    else:
        dqg = dkg = qkv
    cos, sin = _cos_sin(freqs) if freqs is not None else (qkv, qkv)
    with torch.cuda.device(qkv.device):
        qk_rmsnorm_rope_bwd_kernel[(programs,)](
            qkv, dq, dk, dgate, dv, q_gamma if has_w else qkv, k_gamma if has_w else qkv, cos, sin, dqkv,
            dqg, dkg, B, float(eps),
            dq.stride(0), dq.stride(1), dk.stride(0), dk.stride(1),
            dgate.stride(0), dgate.stride(1), dv.stride(0), dv.stride(1),
            NG=NG, NPG=npg, NPG_P2=triton.next_power_of_2(npg), D=D, HALF=D // 2, GATED=gated, HAS_W=has_w,
            ROT=freqs is not None, GP=gp, num_warps=_TRITON_NUM_WARPS,
        )  # fmt: skip
    if not has_w:
        return dqkv, None, None
    return dqkv, dqg.sum(dim=0).to(q_gamma.dtype), dkg.sum(dim=0).to(k_gamma.dtype)


def qk_rmsnorm_rope_fwd_impl(qkv, q_gamma, k_gamma, freqs, split_sizes, eps, backend=None):
    """``(backend, q, k, v, gate, saved)``: gate is None without a gate block; ``saved`` goes to the backward."""
    chosen, why = select_backend(qkv, q_gamma, k_gamma, freqs, split_sizes, backend)
    if chosen is None:
        raise ValueError(f"fused_qkv_rmsnorm_rope: unsupported input ({why})")
    npg, D, gated = _layout(qkv, split_sizes)
    if chosen == BackendType.FLYDSL:
        q, k, v, q_rstd, k_rstd = flydsl_qkv_rmsnorm_rope_forward(
            qkv, q_gamma, k_gamma, freqs.float().contiguous(), split_sizes, eps
        )
        return chosen, q, k, v, None, (q_rstd, k_rstd)
    q, k = _triton_fwd(qkv, q_gamma, k_gamma, freqs, npg, D, gated, eps)
    v = qkv[..., -D:]
    gate = qkv[..., npg * D : 2 * npg * D] if gated else None
    return chosen, q, k, v, gate, ()


def qk_rmsnorm_rope_bwd_impl(
    backend, dq, dk, dv, dgate, qkv, q_gamma, k_gamma, freqs, saved, split_sizes, eps
):
    """``(dqkv, dq_gamma, dk_gamma)``; the gamma gradients are None without weights."""
    npg, D, gated = _layout(qkv, split_sizes)
    if backend == BackendType.FLYDSL:
        S, B, NG, _ = qkv.shape
        q_rstd, k_rstd = saved
        dq, dk, dv = (
            torch.zeros(shape, device=qkv.device, dtype=qkv.dtype) if g is None else g.contiguous()
            for g, shape in ((dq, (S, B, NG * npg, D)), (dk, (S, B, NG, D)), (dv, (S, B, NG, D)))
        )
        dqkv, dqg_partial, dkg_partial = flydsl_qkv_rmsnorm_rope_backward(
            dq, dk, dv, qkv, q_gamma, k_gamma, freqs, q_rstd, k_rstd, split_sizes
        )
        return dqkv, dqg_partial.sum(dim=0).to(q_gamma.dtype), dkg_partial.sum(dim=0).to(k_gamma.dtype)
    return _triton_bwd(dq, dk, dv, dgate, qkv, q_gamma, k_gamma, freqs, npg, D, gated, eps)
