###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Unified multi-backend selection for dense flash-attention (bf16).

Mirrors the GEMM multi-backend convention (``kernels/gemm/gemm_impl.py``):
``KernelBackend`` subclasses (``can_handle`` / ``execute``) registered in an
``AutoKernelDispatcher`` and selected by ``GlobalBackendManager`` /
``PRIMUS_TURBO_ATTN_BACKEND`` / autotune.

Attention differs from GEMM in that a forward+backward pair must run on the
*same* backend (their saved-tensor / LSE conventions differ). Rather than one
generic autograd.Function threading a backend enum through ctx, the op layer
(``ops/attention/flash_attn_interface.py``) resolves the backend once via
``resolve_flash_attn_backend`` and routes to a per-backend ``autograd.Function``
-- so consistency holds by construction and the existing aiter autograd path is
untouched. ``execute`` here runs the forward only; it exists so autotune can
time each backend. FP8 (triton) stays on its own ``flash_attn_fp8_func`` path.
"""

import math
from typing import Optional

import torch

from primus_turbo.pytorch.core.backend import (
    AutoKernelDispatcher,
    BackendEntry,
    BackendType,
    KernelBackend,
    TuneCache,
)
from primus_turbo.pytorch.core.utils import get_device_compute_capability
from primus_turbo.pytorch.kernels.attention.attention_aiter_impl import (
    attention_aiter_forward_impl,
    attention_aiter_varlen_forward_impl,
)
from primus_turbo.pytorch.kernels.attention.attention_flydsl_impl import (
    flash_attn_sbhd_flydsl_forward_impl,
    flash_attn_varlen_flydsl_forward_impl,
)

_GFX950 = (9, 5)


def _scale_ok(softmax_scale: Optional[float], head_dim: int) -> bool:
    """FlyDSL bakes softmax_scale = 1/sqrt(D); accept only None or that value."""
    return softmax_scale is None or abs(softmax_scale - 1.0 / math.sqrt(head_dim)) < 1e-6


def _sink_ok(sink: Optional[torch.Tensor], num_heads_q: int) -> bool:
    """FlyDSL folds a learned per-q-head attention sink into the softmax denominator; it
    must be an fp32 [Hq] tensor. None (no sink) is always fine; a malformed sink falls
    back to aiter."""
    return sink is None or (sink.dtype == torch.float32 and sink.numel() == num_heads_q)


def _gqa_group_ok(num_heads_q: int, num_heads_kv: int) -> bool:
    """FlyDSL deterministic dkdv backward (BLOCK_SIZE=256, BLOCK_Q=64) needs the GQA
    group size G = Hq // Hkv to be a power of two in [8, 256]: its LDS-staged (delta,
    lse) load uses LD_VEC = BLOCK_Q // (BLOCK_SIZE // G) = 64 // (256 // G), which must
    be >= 2 (G >= 8) for the cooperative vector store. MHA (G==1) and small/non-power-of-2
    groups fall back to aiter."""
    if num_heads_kv <= 0 or num_heads_q % num_heads_kv != 0:
        return False
    g = num_heads_q // num_heads_kv
    return 8 <= g <= 256 and (g & (g - 1)) == 0


def _flydsl_common_ok(
    q: torch.Tensor,
    causal: bool,
    window_size,
    softmax_scale: Optional[float],
    dropout_p: float,
    bias: Optional[torch.Tensor],
    alibi_slopes: Optional[torch.Tensor],
) -> bool:
    """Shared FlyDSL eligibility gate (gfx950, causal, D in {64,128}, bf16, ...).

    Both D=64 and D=128 are supported: the fwd handles D in {64,128}, and the
    deterministic 16x16 backward's LDS pack is now D-dependent -- D64 packs two
    64-wide rows per 128-wide block, D128 uses one 128-wide row per block. Other
    head dims fall back to aiter.

    ``sink`` is validated separately (``_sink_ok``) where Hq is known: FlyDSL folds a
    learned per-head sink into the softmax denominator, so a valid sink stays on FlyDSL.
    """
    head_dim = q.shape[-1]
    return (
        get_device_compute_capability() >= _GFX950
        and bool(causal)
        and q.dtype == torch.bfloat16
        and head_dim in (64, 128)
        and int(window_size[1]) in (0, -1)
        and _scale_ok(softmax_scale, head_dim)
        and dropout_p == 0.0
        and bias is None
        and alibi_slopes is None
    )


# =============================================================================
# Dense (bshd / sbhd / bhsd) forward backends
# =============================================================================


class DenseAttnFwdAiterBackend(KernelBackend):
    @staticmethod
    def can_handle(q: torch.Tensor, **kwargs) -> bool:
        return q.dtype in (torch.float16, torch.bfloat16) and q.ndim == 4

    @staticmethod
    def execute(
        q,
        k,
        v,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        bias,
        alibi_slopes,
        sink,
        qkv_format,
        **kwargs,
    ):
        return attention_aiter_forward_impl(
            q=q,
            k=k,
            v=v,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size_left=int(window_size[0]),
            window_size_right=int(window_size[1]),
            bias=bias,
            alibi_slopes=alibi_slopes,
            return_lse=True,
            return_softmax=False,
            max_seqlen_q=q.size(1),
            max_seqlen_k=k.size(1),
            sink=sink,
            qkv_format=qkv_format,
        )


class DenseAttnFwdFlydslBackend(KernelBackend):
    @staticmethod
    def can_handle(
        q,
        k=None,
        dropout_p=0.0,
        softmax_scale=None,
        causal=True,
        window_size=(-1, -1),
        bias=None,
        alibi_slopes=None,
        sink=None,
        qkv_format="bshd",
        **kwargs,
    ) -> bool:
        # FlyDSL dense forward is SBHD-native: only the no-copy sbhd layout is eligible.
        if k is None or not _gqa_group_ok(q.shape[2], k.shape[2]) or not _sink_ok(sink, q.shape[2]):
            return False
        return qkv_format == "sbhd" and _flydsl_common_ok(
            q, causal, window_size, softmax_scale, dropout_p, bias, alibi_slopes
        )

    @staticmethod
    def execute(q, k, v, softmax_scale, causal, window_size, return_lse=True, **kwargs):
        # Logical [b,s,h,d] with sbhd storage -> [s,b,h,d] contiguous (no copy).
        q_s = q.permute(1, 0, 2, 3)
        k_s = k.permute(1, 0, 2, 3)
        v_s = v.permute(1, 0, 2, 3)
        return flash_attn_sbhd_flydsl_forward_impl(
            q_s,
            k_s,
            v_s,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            return_lse=return_lse,
        )


_DENSE_FWD_BACKENDS = {
    BackendType.FLYDSL: BackendEntry(DenseAttnFwdFlydslBackend),
    BackendType.AITER: BackendEntry(DenseAttnFwdAiterBackend),
}


class FlashAttnDenseDispatcher(AutoKernelDispatcher):
    _backends = _DENSE_FWD_BACKENDS
    _cache = TuneCache(1024)

    @classmethod
    def make_key(cls, q, k, causal=True, window_size=(-1, -1), qkv_format="bshd", sink=None, **kwargs):
        b, s, hq, d = q.shape
        hkv = k.shape[2]
        return (b, s, hq, hkv, d, q.dtype, bool(causal), tuple(window_size), qkv_format, sink is not None)


# =============================================================================
# Varlen (thd) forward backends
# =============================================================================


def _uniform_cu_seqlens(cu_seqlens_q, cu_seqlens_k) -> bool:
    """True when every per-batch segment length is equal (FlyDSL varlen needs it)."""
    for cu in (cu_seqlens_q, cu_seqlens_k):
        seg = cu[1:] - cu[:-1]
        if seg.numel() == 0 or not bool((seg == seg[0]).all().item()):
            return False
    return True


class VarlenAttnFwdAiterBackend(KernelBackend):
    @staticmethod
    def can_handle(q: torch.Tensor, **kwargs) -> bool:
        return q.dtype in (torch.float16, torch.bfloat16) and q.ndim == 3

    @staticmethod
    def execute(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        bias,
        alibi_slopes,
        **kwargs,
    ):
        return attention_aiter_varlen_forward_impl(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size_left=int(window_size[0]),
            window_size_right=int(window_size[1]),
            bias=bias,
            alibi_slopes=alibi_slopes,
            return_lse=True,
            return_softmax=False,
        )


class VarlenAttnFwdFlydslBackend(KernelBackend):
    @staticmethod
    def can_handle(
        q,
        k=None,
        cu_seqlens_q=None,
        cu_seqlens_k=None,
        dropout_p=0.0,
        softmax_scale=None,
        causal=True,
        window_size=(-1, -1),
        bias=None,
        alibi_slopes=None,
        sink=None,
        **kwargs,
    ) -> bool:
        # varlen THD: q [total, Hq, D], k [total, Hkv, D].
        if k is None or not _gqa_group_ok(q.shape[1], k.shape[1]) or not _sink_ok(sink, q.shape[1]):
            return False
        if not _flydsl_common_ok(q, causal, window_size, softmax_scale, dropout_p, bias, alibi_slopes):
            return False
        return _uniform_cu_seqlens(cu_seqlens_q, cu_seqlens_k)

    @staticmethod
    def execute(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        softmax_scale,
        causal,
        window_size,
        return_lse=True,
        **kwargs,
    ):
        return flash_attn_varlen_flydsl_forward_impl(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            return_lse=return_lse,
        )


_VARLEN_FWD_BACKENDS = {
    BackendType.FLYDSL: BackendEntry(VarlenAttnFwdFlydslBackend),
    BackendType.AITER: BackendEntry(VarlenAttnFwdAiterBackend),
}


class FlashAttnVarlenDispatcher(AutoKernelDispatcher):
    _backends = _VARLEN_FWD_BACKENDS
    _cache = TuneCache(1024)

    @classmethod
    def make_key(
        cls, q, k, causal=True, window_size=(-1, -1), max_seqlen_q=0, max_seqlen_k=0, sink=None, **kwargs
    ):
        total_q, hq, d = q.shape
        total_k, hkv, _ = k.shape
        return (
            total_q,
            total_k,
            hq,
            hkv,
            d,
            q.dtype,
            bool(causal),
            tuple(window_size),
            int(max_seqlen_q),
            int(max_seqlen_k),
            sink is not None,
        )


# =============================================================================
# Backend resolution (used by the op layer to pick fwd == bwd backend)
# =============================================================================


def resolve_flash_attn_backend(varlen: bool, user_backend: Optional[BackendType], **kwargs) -> BackendType:
    """Resolve the dense/varlen flash-attn backend enum (default FLYDSL, else AITER)."""
    dispatcher = FlashAttnVarlenDispatcher if varlen else FlashAttnDenseDispatcher
    return dispatcher.resolve(BackendType.FLYDSL, user_backend, **kwargs)
