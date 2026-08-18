###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Multi-backend selection for DeepSeek-V4 single-latent sparse-MLA attention.

Mirrors the flash-attn multi-backend layer (``attention_impl.py``): per-backend
``KernelBackend`` (can_handle / execute) registered in an ``AutoKernelDispatcher``
and selected by ``GlobalBackendManager`` / ``PRIMUS_TURBO_ATTN_BACKEND`` /
autotune. FLYDSL is the fast default (gfx950); TRITON is the reference oracle and
non-gfx950 fallback.

As with flash-attn, a forward+backward pair must run on the *same* backend. The
op layer (``ops/attention/sparse_mla_interface.py``) resolves the backend once in
the forward via ``resolve_sparse_mla_fwd_backend``, stores it in ``ctx.backend``,
and pins that enum for the backward -- so consistency holds by construction.
``execute`` runs one pass (fwd or bwd); it exists so autotune can time backends.
"""

from typing import Optional

import torch

from primus_turbo.pytorch.core.backend import (
    AutoKernelDispatcher,
    BackendEntry,
    BackendType,
    KernelBackend,
    TuneCache,
)
from primus_turbo.pytorch.core.utils import is_gfx950

# NOTE: the flydsl/triton sparse-MLA kernels are imported lazily inside execute()
# (not at module top) so that importing this package stays cheap and does not pull
# in the gfx950-only flydsl sparse kernels on non-gfx950 hosts / at test collection.

_DSV4_QK_DIM = 576  # kv_lora_rank(512) + rope(64)


def _shape_ok(q: torch.Tensor, topk_indices: torch.Tensor) -> bool:
    """DSV4 sparse-MLA shape gate shared by both backends: q head-dim 576, bf16,
    num_heads multiple of 32, topk (padded) multiple of 32."""
    _, num_heads, d_qk = q.shape
    topk = topk_indices.shape[1]
    return d_qk == _DSV4_QK_DIM and q.dtype == torch.bfloat16 and num_heads % 32 == 0 and topk % 32 == 0


# =============================================================================
# Forward backends
# =============================================================================


class SparseMlaFwdFlydslBackend(KernelBackend):
    @staticmethod
    def can_handle(q, topk_indices, **kwargs) -> bool:
        return is_gfx950() and _shape_ok(q, topk_indices)

    @staticmethod
    def execute(q, kv, topk_indices, attn_sink, kv_lora_rank, scale, **kwargs):
        from primus_turbo.flydsl.attention.sparse_mla_fwd import sparse_mla_fwd_flydsl

        return sparse_mla_fwd_flydsl(
            q, kv, topk_indices, attn_sink=attn_sink, kv_lora_rank=kv_lora_rank, scale=scale
        )


class SparseMlaFwdTritonBackend(KernelBackend):
    @staticmethod
    def can_handle(q, topk_indices, **kwargs) -> bool:
        return _shape_ok(q, topk_indices)

    @staticmethod
    def execute(q, kv, topk_indices, attn_sink, kv_lora_rank, scale, **kwargs):
        from primus_turbo.triton.attention.sparse_mla import sparse_mla_fwd_triton

        return sparse_mla_fwd_triton(
            q, kv, topk_indices, attn_sink=attn_sink, kv_lora_rank=kv_lora_rank, scale=scale
        )


_SPARSE_MLA_FWD_BACKENDS = {
    BackendType.FLYDSL: BackendEntry(SparseMlaFwdFlydslBackend),
    BackendType.TRITON: BackendEntry(SparseMlaFwdTritonBackend),
}


class SparseMlaFwdDispatcher(AutoKernelDispatcher):
    _backends = _SPARSE_MLA_FWD_BACKENDS
    _cache = TuneCache(1024)

    @classmethod
    def make_key(cls, q, kv, topk_indices, **kwargs):
        total_tokens, num_heads, d_qk = q.shape
        num_kv = kv.shape[0]
        topk = topk_indices.shape[1]
        return (total_tokens, num_heads, d_qk, num_kv, topk, q.dtype)


# =============================================================================
# Backward backends
# =============================================================================


class SparseMlaBwdFlydslBackend(KernelBackend):
    @staticmethod
    def can_handle(q, topk_indices, **kwargs) -> bool:
        return is_gfx950() and _shape_ok(q, topk_indices)

    @staticmethod
    def execute(q, kv, o, do, topk_indices, lse, attn_sink, kv_lora_rank, scale, **kwargs):
        from primus_turbo.flydsl.attention.sparse_mla_bwd import sparse_mla_bwd_flydsl

        return sparse_mla_bwd_flydsl(
            q, kv, o, do, topk_indices, lse, attn_sink=attn_sink, kv_lora_rank=kv_lora_rank, scale=scale
        )


class SparseMlaBwdTritonBackend(KernelBackend):
    @staticmethod
    def can_handle(q, topk_indices, **kwargs) -> bool:
        return _shape_ok(q, topk_indices)

    @staticmethod
    def execute(q, kv, o, do, topk_indices, lse, attn_sink, kv_lora_rank, scale, **kwargs):
        from primus_turbo.triton.attention.sparse_mla import sparse_mla_bwd_triton

        return sparse_mla_bwd_triton(
            q, kv, o, do, topk_indices, lse, attn_sink=attn_sink, kv_lora_rank=kv_lora_rank, scale=scale
        )


_SPARSE_MLA_BWD_BACKENDS = {
    BackendType.FLYDSL: BackendEntry(SparseMlaBwdFlydslBackend),
    BackendType.TRITON: BackendEntry(SparseMlaBwdTritonBackend),
}


class SparseMlaBwdDispatcher(AutoKernelDispatcher):
    _backends = _SPARSE_MLA_BWD_BACKENDS
    _cache = TuneCache(1024)

    @classmethod
    def make_key(cls, q, kv, topk_indices, **kwargs):
        total_tokens, num_heads, d_qk = q.shape
        num_kv = kv.shape[0]
        topk = topk_indices.shape[1]
        return (total_tokens, num_heads, d_qk, num_kv, topk, q.dtype)


# =============================================================================
# Backend resolution (fwd resolves once; bwd is pinned to ctx.backend)
# =============================================================================


def resolve_sparse_mla_fwd_backend(user_backend: Optional[BackendType], **kwargs) -> BackendType:
    """Resolve the sparse-MLA backend enum (default FLYDSL, else TRITON oracle).

    PRIMUS_TURBO_ATTN_BACKEND names one backend for both attention dispatchers, which do not
    carry the same set -- sparse-MLA has no aiter path. A name this one does not have is
    taken as no preference rather than an error, so pinning aiter for flash-attention does
    not take sparse-MLA down with it.
    """
    if user_backend is not None and user_backend not in SparseMlaFwdDispatcher._backends:
        user_backend = None
    return SparseMlaFwdDispatcher.resolve(BackendType.FLYDSL, user_backend, **kwargs)
