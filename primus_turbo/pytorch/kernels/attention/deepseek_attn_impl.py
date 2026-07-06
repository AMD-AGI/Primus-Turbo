###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""DeepSeek-V4 attention forward dispatcher (Triton / FlyDSL backends).

This mirrors the GEMM FP8 dispatcher (``kernels/gemm/gemm_fp8_impl.py``):
each backend is a :class:`KernelBackend` with ``can_handle`` + ``execute``,
the per-op :class:`AutoKernelDispatcher` declares ``_backends`` + ``make_key``,
and the public entry is a ``@torch.library.custom_op`` (with ``register_fake``)
that resolves the backend via :meth:`AutoKernelDispatcher.dispatch`.

Two forward custom_ops cover V4's attention input forms:

* ``deepseek_attn_fwd``          — dense (``compress_ratio == 0``) and HCA
  (``compress_ratio == 128``, split-mask); see ``ops/attention/hca_attention.py``.
* ``deepseek_csa_pool_attn_fwd`` — CSA (``compress_ratio == 4``) that gathers the
  sparse keys in-kernel from the compressed pool.

Backend contract (matches the dispatcher base): ``can_handle`` must return
``False`` (never raise) for unsupported inputs so the dispatcher can fall back.
The default backend is ``TRITON``; a user can force ``FLYDSL`` per-call (the
``ops`` layer ``backend=`` keyword) or process-wide (``set_attention_backend`` /
``PRIMUS_TURBO_ATTENTION_BACKEND``). When ``FLYDSL`` cannot handle the inputs
(e.g. gfx942, ``D != 512``, dense additive bias) it returns ``False`` and the
dispatcher falls back to Triton.

The FlyDSL forward backend first version covers the dense / HCA split-mask
paths (design §4); CSA FlyDSL (the two-pass head-block sparse branch, design
§4.7) and the FlyDSL backward (design §5.4) are later optimization rounds, so
their FlyDSL ``can_handle`` returns ``False`` and they fall back to Triton.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch

# FlyDSL is an optional backend (design §3.3): the installed ``flydsl`` package
# may be absent or ABI-mismatched. Import lazily so a broken/missing FlyDSL never
# breaks the default Triton attention path — ``can_handle`` then reports the
# FlyDSL backend as unavailable and the dispatcher falls back to Triton.
try:
    from primus_turbo.flydsl.attention.deepseek_attn_fwd_kernel import (
        csa_pool_attention_fwd_flydsl_kernel,
        hca_attention_fwd_flydsl_kernel,
    )
    from primus_turbo.flydsl.attention.deepseek_attn_bwd_kernel import (
        hca_attention_bwd_flydsl_kernel,
    )

    _FLYDSL_AVAILABLE = True
except Exception:  # noqa: BLE001 - ImportError or any flydsl ABI/load failure
    hca_attention_fwd_flydsl_kernel = None
    csa_pool_attention_fwd_flydsl_kernel = None
    hca_attention_bwd_flydsl_kernel = None
    _FLYDSL_AVAILABLE = False

from primus_turbo.pytorch.core.backend import (
    AutoKernelDispatcher,
    BackendEntry,
    BackendType,
    GlobalBackendManager,
    KernelBackend,
    PrecisionType,
    TuneCache,
)
from primus_turbo.pytorch.core.utils import get_device_compute_capability
from primus_turbo.triton.attention.deepseek import (
    _launch_csa_attention_pool_fwd,
    _launch_hca_attention_bwd,
    _launch_hca_attention_fwd,
)

_torch_custom_op_wrapper = torch.library.custom_op

# Attention runs in bf16 / fp16 (input dtype matmuls, fp32 online softmax), so
# the backend selection bucket is the BF16_FP16_FP32 precision.
_ATTN_PRECISION = PrecisionType.BF16_FP16_FP32

# Sentinel for "no per-call backend override" in the custom_op int arg
# (BackendType enum values start at 1, so 0 is safe).
_NO_BACKEND_OVERRIDE = 0


def _resolve_user_backend(backend_override: int) -> Optional[BackendType]:
    """Per-call override (``ops`` ``backend=`` keyword) wins over the global /
    env attention backend; ``0`` means "no override -> consult the manager"."""
    if backend_override != _NO_BACKEND_OVERRIDE:
        return BackendType(backend_override)
    return GlobalBackendManager.get_attention_backend(_ATTN_PRECISION)


def _fallback_if_unsupported(dispatcher_cls, user_backend, kwargs):
    """Attention dispatch contract (design §3.1): an explicitly selected backend
    (e.g. FLYDSL) that cannot handle the inputs falls back to the default
    (Triton) silently rather than raising. Returns the user backend if it can
    handle the inputs, else ``None`` (so :meth:`dispatch` uses the default)."""
    if user_backend is None:
        return None
    entry = dispatcher_cls._backends.get(user_backend)
    if entry is None or not entry.impl.can_handle(**kwargs):
        return None
    return user_backend


# ───────────────────────────────────────────────────────────────────────────
# dense / HCA forward
# ───────────────────────────────────────────────────────────────────────────


class DeepSeekAttnTritonBackend(KernelBackend):
    """Triton dense / HCA forward (the existing ``_launch_hca_attention_fwd``)."""

    @staticmethod
    def can_handle(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        sink: Optional[torch.Tensor],
        additive_mask: Optional[torch.Tensor],
        swa_window: int,
        scale: float,
        hca_local_seqlen: int,
    ) -> bool:
        # Triton is the general-purpose backend; the launcher validates shapes.
        return True

    @staticmethod
    def execute(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        sink: Optional[torch.Tensor],
        additive_mask: Optional[torch.Tensor],
        swa_window: int,
        scale: float,
        hca_local_seqlen: int,
    ):
        return _launch_hca_attention_fwd(
            q,
            k,
            v,
            sink=sink,
            swa_window=swa_window,
            additive_mask=additive_mask,
            scale=scale,
            hca_local_seqlen=hca_local_seqlen,
        )


class DeepSeekAttnFlyDSLBackend(KernelBackend):
    """FlyDSL dense / sliding-window-causal (SWA) forward (ported from Primus).

    Gating: gfx950 only (``ds_read_*_tr`` / 16 B G2S), bf16 with
    ``q.dtype == k.dtype == v.dtype``, ``D == 512`` (the V4-Flash head_dim the
    ported SWA kernel specialises), ``K_H in {1, H}``, ``swa_window > 0``,
    ``Sq == Sk``, ``scale == 1/sqrt(D)``, and *no* sink / additive bias / HCA
    split-mask (the ported SWA kernel's scope is "no sink, no additive mask, no
    HCA" — those inputs fall back to Triton). HCA forward stays on Triton.
    """

    @staticmethod
    def can_handle(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        sink: Optional[torch.Tensor],
        additive_mask: Optional[torch.Tensor],
        swa_window: int,
        scale: float,
        hca_local_seqlen: int,
    ) -> bool:
        supported = True
        # A missing / ABI-mismatched flydsl package keeps this backend disabled
        # so the dispatcher silently uses Triton.
        supported &= _FLYDSL_AVAILABLE
        # ds_read_*_tr / 16 B global-to-LDS exist only on gfx950 (CDNA4).
        supported &= get_device_compute_capability() >= (9, 5)
        supported &= q.dim() == 4 and k.dim() == 4 and v.dim() == 4
        # The ported SWA forward kernel is bf16-only.
        supported &= q.dtype == torch.bfloat16
        supported &= q.dtype == k.dtype == v.dtype
        supported &= q.shape[-1] == 512
        # MQA (K_H == 1) or MHA (K_H == H).
        K_H = k.shape[1]
        supported &= K_H == 1 or K_H == q.shape[1]
        # SWA-only: no sink, no additive bias, no HCA split-mask.
        supported &= sink is None
        supported &= additive_mask is None
        supported &= int(hca_local_seqlen) == 0
        supported &= int(swa_window) > 0
        # Sliding-window forward requires Sq == Sk.
        supported &= k.shape[2] == q.shape[2]
        # FlyDSL packs the flat (``view(-1)``) tensor length as int32, so a
        # per-tensor element count of >= 2**31 overflows the arg codec. Fall
        # back to Triton for those (very long) shapes.
        supported &= q.numel() < 2**31
        # The kernel hard-codes scale == 1/sqrt(D).
        supported &= math.isclose(float(scale), 1.0 / math.sqrt(q.shape[-1]), rel_tol=1e-4)
        return supported

    @staticmethod
    def execute(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        sink: Optional[torch.Tensor],
        additive_mask: Optional[torch.Tensor],
        swa_window: int,
        scale: float,
        hca_local_seqlen: int,
    ):
        return hca_attention_fwd_flydsl_kernel(
            q,
            k,
            v,
            sink=sink,
            swa_window=swa_window,
            additive_mask=additive_mask,
            scale=scale,
            hca_local_seqlen=hca_local_seqlen,
        )


class DeepSeekAttnFwdDispatcher(AutoKernelDispatcher):
    _backends = {
        BackendType.TRITON: BackendEntry(DeepSeekAttnTritonBackend),
        BackendType.FLYDSL: BackendEntry(DeepSeekAttnFlyDSLBackend),
    }
    _cache = TuneCache(1024)

    @classmethod
    def make_key(cls, q, k, v, sink, additive_mask, swa_window, hca_local_seqlen, **kwargs):
        B, H, Sq, D = q.shape
        K_H, Sk = k.shape[1], k.shape[2]
        kind = "hca" if hca_local_seqlen > 0 else "dense"
        return (
            B,
            H,
            K_H,
            Sq,
            Sk,
            D,
            q.dtype,
            int(swa_window),
            sink is not None,
            additive_mask is not None,
            int(hca_local_seqlen),
            kind,
        )


@_torch_custom_op_wrapper("primus_turbo::deepseek_attn_fwd", mutates_args=(), device_types="cuda")
def deepseek_attn_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    sink: Optional[torch.Tensor],
    additive_mask: Optional[torch.Tensor],
    swa_window: int,
    scale: float,
    hca_local_seqlen: int,
    default_backend: int,
    backend_override: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    default_backend_enum = BackendType(default_backend)
    user_backend_enum = _resolve_user_backend(backend_override)
    kwargs = dict(
        q=q,
        k=k,
        v=v,
        sink=sink,
        additive_mask=additive_mask,
        swa_window=int(swa_window),
        scale=float(scale),
        hca_local_seqlen=int(hca_local_seqlen),
    )
    user_backend_enum = _fallback_if_unsupported(DeepSeekAttnFwdDispatcher, user_backend_enum, kwargs)
    return DeepSeekAttnFwdDispatcher.dispatch(default_backend_enum, user_backend_enum, **kwargs)


@deepseek_attn_fwd.register_fake
def _deepseek_attn_fwd_meta(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    sink: Optional[torch.Tensor],
    additive_mask: Optional[torch.Tensor],
    swa_window: int,
    scale: float,
    hca_local_seqlen: int,
    default_backend: int,
    backend_override: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    B, H, Sq, _ = q.shape
    out = torch.empty_like(q)
    lse = torch.empty((B, H, Sq), dtype=torch.float32, device=q.device)
    return out, lse


# ───────────────────────────────────────────────────────────────────────────
# dense / HCA backward (dispatcher selects backend per fwd / bwd, design §3.3)
# ───────────────────────────────────────────────────────────────────────────


class DeepSeekAttnBwdTritonBackend(KernelBackend):
    """Triton dense / HCA backward (the existing ``_launch_hca_attention_bwd``)."""

    @staticmethod
    def can_handle(q, k, v, out, dout, lse, sink, additive_mask, swa_window, scale, hca_local_seqlen) -> bool:
        return True

    @staticmethod
    def execute(q, k, v, out, dout, lse, sink, additive_mask, swa_window, scale, hca_local_seqlen):
        return _launch_hca_attention_bwd(
            q,
            k,
            v,
            out,
            dout,
            lse,
            sink=sink,
            swa_window=swa_window,
            additive_mask=additive_mask,
            scale=scale,
            hca_local_seqlen=hca_local_seqlen,
        )


class DeepSeekAttnBwdFlyDSLBackend(KernelBackend):
    """FlyDSL dense / SWA / HCA split-mask backward.

    Two paths, both bf16 + MQA (``K_H == 1``) + ``D == 512`` on gfx950:

    * Dense / SWA / causal (``additive_mask is None`` and
      ``hca_local_seqlen == 0``, ``Sq == Sk``).
    * HCA split-mask (``additive_mask`` and ``hca_local_seqlen > 0``): the
      joint-softmax backward composed from the local SWA dq/dkv kernels plus
      the ported HCA pool dq/dkv kernels. Pool kernel constraints: ``B == 1``,
      ``hca_local_seqlen == Sq``, ``hca_local_seqlen % 64 == 0`` and
      ``0 < pool_size <= 32`` (``pool_size = Sk - hca_local_seqlen``).

    The CSA backward has its own dispatcher. Anything outside these envelopes
    returns ``False`` and the dispatcher falls back to Triton.
    """

    @staticmethod
    def can_handle(q, k, v, out, dout, lse, sink, additive_mask, swa_window, scale, hca_local_seqlen) -> bool:
        supported = _FLYDSL_AVAILABLE
        supported &= get_device_compute_capability() >= (9, 5)
        supported &= q.dim() == 4 and k.dim() == 4 and v.dim() == 4
        # The ported SWA / HCA backward kernels are bf16-only and MQA-only.
        supported &= q.dtype == torch.bfloat16
        supported &= q.dtype == k.dtype == v.dtype
        supported &= q.shape[-1] == 512
        supported &= k.shape[1] == 1
        supported &= int(swa_window) > 0
        if not supported:
            return False

        B, HQ, Sq, D = q.shape
        Sk = k.shape[2]
        is_hca = additive_mask is not None and int(hca_local_seqlen) > 0
        if is_hca:
            pool_size = int(Sk) - int(hca_local_seqlen)
            ok = B == 1
            ok &= int(hca_local_seqlen) == Sq
            ok &= int(hca_local_seqlen) % 64 == 0
            ok &= 0 < pool_size <= 32
            ok &= additive_mask.dim() == 2
            ok &= tuple(additive_mask.shape) == (Sq, pool_size)
            return bool(ok)

        # Dense / SWA only: no additive bias, no HCA split-mask pool branch.
        ok = additive_mask is None
        ok &= int(hca_local_seqlen) == 0
        ok &= Sk == Sq
        return bool(ok)

    @staticmethod
    def execute(q, k, v, out, dout, lse, sink, additive_mask, swa_window, scale, hca_local_seqlen):
        return hca_attention_bwd_flydsl_kernel(
            q,
            k,
            v,
            out,
            dout,
            lse,
            sink=sink,
            swa_window=swa_window,
            additive_mask=additive_mask,
            scale=scale,
            hca_local_seqlen=hca_local_seqlen,
        )


class DeepSeekAttnBwdDispatcher(AutoKernelDispatcher):
    _backends = {
        BackendType.TRITON: BackendEntry(DeepSeekAttnBwdTritonBackend),
        BackendType.FLYDSL: BackendEntry(DeepSeekAttnBwdFlyDSLBackend),
    }
    _cache = TuneCache(1024)

    @classmethod
    def make_key(cls, q, k, v, sink, additive_mask, swa_window, hca_local_seqlen, **kwargs):
        B, H, Sq, D = q.shape
        K_H, Sk = k.shape[1], k.shape[2]
        kind = "hca" if hca_local_seqlen > 0 else "dense"
        return (
            B,
            H,
            K_H,
            Sq,
            Sk,
            D,
            q.dtype,
            int(swa_window),
            sink is not None,
            additive_mask is not None,
            int(hca_local_seqlen),
            kind,
        )


def deepseek_attn_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    dout: torch.Tensor,
    lse: torch.Tensor,
    *,
    sink: Optional[torch.Tensor],
    additive_mask: Optional[torch.Tensor],
    swa_window: int,
    scale: float,
    hca_local_seqlen: int,
    backend_override: int = _NO_BACKEND_OVERRIDE,
):
    """Backward dispatch entry (Triton default; FlyDSL when selected/supported).

    Called directly from the autograd ``Function.backward`` (the backward is not
    a ``custom_op`` — the autograd Function is the torch.compile boundary).
    Returns ``(dq, dk, dv, dsink)``; an explicitly selected FlyDSL backend that
    cannot handle the inputs falls back to Triton (no raise; design §3.1)."""
    user_backend_enum = _resolve_user_backend(int(backend_override))
    kwargs = dict(
        q=q,
        k=k,
        v=v,
        out=out,
        dout=dout,
        lse=lse,
        sink=sink,
        additive_mask=additive_mask,
        swa_window=int(swa_window),
        scale=float(scale),
        hca_local_seqlen=int(hca_local_seqlen),
    )
    user_backend_enum = _fallback_if_unsupported(DeepSeekAttnBwdDispatcher, user_backend_enum, kwargs)
    return DeepSeekAttnBwdDispatcher.dispatch(BackendType.TRITON, user_backend_enum, **kwargs)


# ───────────────────────────────────────────────────────────────────────────
# CSA forward (from compressed pool, in-kernel gather)
# ───────────────────────────────────────────────────────────────────────────


class DeepSeekCSAPoolAttnTritonBackend(KernelBackend):
    @staticmethod
    def can_handle(q, k_local, v_local, pool, topk_idxs, sink, swa_window, scale) -> bool:
        return True

    @staticmethod
    def execute(q, k_local, v_local, pool, topk_idxs, sink, swa_window, scale):
        return _launch_csa_attention_pool_fwd(
            q,
            k_local,
            v_local,
            pool,
            topk_idxs,
            sink=sink,
            swa_window=swa_window,
            scale=scale,
        )


class DeepSeekCSAPoolAttnFlyDSLBackend(KernelBackend):
    """CSA-from-pool FlyDSL forward (in-kernel gather, design §4.7).

    Gathers the per-query top-K keys from the compressed ``pool`` via
    ``topk_idxs`` inside the kernel (no materialised ``[B, Sq, K_topk, D]``
    tensor). Gating: gfx950, bf16 (``q.dtype == k_local.dtype == v_local.dtype
    == pool.dtype``), ``D == 512``, ``K_H in {1, H}``, ``swa_window > 0``,
    ``K_topk > 0`` and ``scale == 1/sqrt(D)``.
    """

    @staticmethod
    def can_handle(q, k_local, v_local, pool, topk_idxs, sink, swa_window, scale) -> bool:
        supported = _FLYDSL_AVAILABLE
        supported &= get_device_compute_capability() >= (9, 5)
        supported &= q.dim() == 4 and k_local.dim() == 4 and v_local.dim() == 4 and pool.dim() == 3
        supported &= q.dtype == torch.bfloat16
        supported &= q.dtype == k_local.dtype == v_local.dtype == pool.dtype
        supported &= q.shape[-1] == 512
        K_H = k_local.shape[1]
        supported &= K_H == 1 or K_H == q.shape[1]
        supported &= topk_idxs.dim() == 3 and topk_idxs.shape[2] > 0
        supported &= int(swa_window) > 0
        supported &= math.isclose(float(scale), 1.0 / math.sqrt(q.shape[-1]), rel_tol=1e-4)
        # FlyDSL packs the flat (``view(-1)``) tensor length as int32. From-pool
        # avoids the gathered buffer, so only q / pool / topk_idxs are checked.
        supported &= q.numel() < 2**31
        supported &= pool.numel() < 2**31
        supported &= topk_idxs.numel() < 2**31
        return supported

    @staticmethod
    def execute(q, k_local, v_local, pool, topk_idxs, sink, swa_window, scale):
        return csa_pool_attention_fwd_flydsl_kernel(
            q,
            k_local,
            v_local,
            pool,
            topk_idxs,
            sink,
            int(swa_window),
            float(scale),
        )


class DeepSeekCSAPoolAttnFwdDispatcher(AutoKernelDispatcher):
    _backends = {
        BackendType.TRITON: BackendEntry(DeepSeekCSAPoolAttnTritonBackend),
        BackendType.FLYDSL: BackendEntry(DeepSeekCSAPoolAttnFlyDSLBackend),
    }
    _cache = TuneCache(1024)

    @classmethod
    def make_key(cls, q, k_local, v_local, pool, topk_idxs, sink, swa_window, **kwargs):
        B, H, Sq, D = q.shape
        P = pool.shape[1]
        K_topk = topk_idxs.shape[2]
        return (
            B,
            H,
            Sq,
            D,
            K_topk,
            P,
            q.dtype,
            pool.dtype,
            topk_idxs.dtype,
            int(swa_window),
            sink is not None,
            "csa_pool",
        )


@_torch_custom_op_wrapper("primus_turbo::deepseek_csa_pool_attn_fwd", mutates_args=(), device_types="cuda")
def deepseek_csa_pool_attn_fwd(
    q: torch.Tensor,
    k_local: torch.Tensor,
    v_local: torch.Tensor,
    pool: torch.Tensor,
    topk_idxs: torch.Tensor,
    sink: Optional[torch.Tensor],
    swa_window: int,
    scale: float,
    default_backend: int,
    backend_override: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    default_backend_enum = BackendType(default_backend)
    user_backend_enum = _resolve_user_backend(backend_override)
    kwargs = dict(
        q=q,
        k_local=k_local,
        v_local=v_local,
        pool=pool,
        topk_idxs=topk_idxs,
        sink=sink,
        swa_window=int(swa_window),
        scale=float(scale),
    )
    user_backend_enum = _fallback_if_unsupported(
        DeepSeekCSAPoolAttnFwdDispatcher, user_backend_enum, kwargs
    )
    return DeepSeekCSAPoolAttnFwdDispatcher.dispatch(default_backend_enum, user_backend_enum, **kwargs)


@deepseek_csa_pool_attn_fwd.register_fake
def _deepseek_csa_pool_attn_fwd_meta(
    q: torch.Tensor,
    k_local: torch.Tensor,
    v_local: torch.Tensor,
    pool: torch.Tensor,
    topk_idxs: torch.Tensor,
    sink: Optional[torch.Tensor],
    swa_window: int,
    scale: float,
    default_backend: int,
    backend_override: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    B, H, Sq, _ = q.shape
    out = torch.empty_like(q)
    lse = torch.empty((B, H, Sq), dtype=torch.float32, device=q.device)
    return out, lse


__all__ = [
    "deepseek_attn_fwd",
    "deepseek_attn_bwd",
    "deepseek_csa_pool_attn_fwd",
    "DeepSeekAttnFwdDispatcher",
    "DeepSeekAttnBwdDispatcher",
    "DeepSeekCSAPoolAttnFwdDispatcher",
]
