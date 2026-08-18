###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Multi-backend selection for DeepSeek-V4 single-latent sparse-MLA attention.

Mirrors the GEMM convention (``kernels/gemm/gemm_impl.py``): ``KernelBackend`` subclasses
registered in an ``AutoKernelDispatcher``, wrapped in a custom op that the autograd
Function calls once per pass.

Forward and backward dispatch independently, which is only sound because both backends
save and consume the same ``o`` / ``lse``: either backward is correct on either forward's
output. Pin a backend if one implementation is wanted for both passes.

FLYDSL is the fast path (gfx950), TRITON the reference oracle and non-gfx950 fallback;
the caller passes which to prefer as ``default_backend``.
"""

from typing import Optional, Tuple

import torch

from primus_turbo.flydsl.attention.sparse_mla_bwd import sparse_mla_bwd_flydsl
from primus_turbo.flydsl.attention.sparse_mla_fwd import sparse_mla_fwd_flydsl
from primus_turbo.pytorch.core.backend import (
    AutoKernelDispatcher,
    BackendChoice,
    BackendEntry,
    BackendType,
    GlobalBackendManager,
    KernelBackend,
    PrecisionType,
    TuneCache,
)
from primus_turbo.pytorch.core.utils import is_gfx950
from primus_turbo.triton.attention.sparse_mla import (
    sparse_mla_bwd_triton,
    sparse_mla_fwd_triton,
)

_DSV4_QK_DIM = 576  # kv_lora_rank(512) + rope(64)


# =============================================================================
# Forward backends
# =============================================================================


class SparseMlaFwdFlydslBackend(KernelBackend):
    @staticmethod
    def can_handle(q, topk_indices, **kwargs) -> bool:
        _, num_heads, d_qk = q.shape
        return (
            is_gfx950()
            and q.dtype == torch.bfloat16
            and d_qk == _DSV4_QK_DIM
            and num_heads % 32 == 0
            and topk_indices.shape[1] % 32 == 0
        )

    @staticmethod
    def execute(q, kv, topk_indices, attn_sink, kv_lora_rank, scale, **kwargs):
        return sparse_mla_fwd_flydsl(
            q, kv, topk_indices, attn_sink=attn_sink, kv_lora_rank=kv_lora_rank, scale=scale
        )


class SparseMlaFwdTritonBackend(KernelBackend):
    @staticmethod
    def can_handle(q, topk_indices, **kwargs) -> bool:
        _, num_heads, d_qk = q.shape
        return (
            q.dtype == torch.bfloat16
            and d_qk == _DSV4_QK_DIM
            and num_heads % 32 == 0
            and topk_indices.shape[1] % 32 == 0
        )

    @staticmethod
    def execute(q, kv, topk_indices, attn_sink, kv_lora_rank, scale, **kwargs):
        return sparse_mla_fwd_triton(
            q, kv, topk_indices, attn_sink=attn_sink, kv_lora_rank=kv_lora_rank, scale=scale
        )


# =============================================================================
# Backward backends
# =============================================================================


class SparseMlaBwdFlydslBackend(KernelBackend):
    @staticmethod
    def can_handle(q, topk_indices, **kwargs) -> bool:
        _, num_heads, d_qk = q.shape
        return (
            is_gfx950()
            and q.dtype == torch.bfloat16
            and d_qk == _DSV4_QK_DIM
            and num_heads % 32 == 0
            and topk_indices.shape[1] % 32 == 0
        )

    @staticmethod
    def execute(q, kv, o, do, topk_indices, lse, attn_sink, kv_lora_rank, scale, **kwargs):
        return sparse_mla_bwd_flydsl(
            q, kv, o, do, topk_indices, lse, attn_sink=attn_sink, kv_lora_rank=kv_lora_rank, scale=scale
        )


class SparseMlaBwdTritonBackend(KernelBackend):
    @staticmethod
    def can_handle(q, topk_indices, **kwargs) -> bool:
        _, num_heads, d_qk = q.shape
        return (
            q.dtype == torch.bfloat16
            and d_qk == _DSV4_QK_DIM
            and num_heads % 32 == 0
            and topk_indices.shape[1] % 32 == 0
        )

    @staticmethod
    def execute(q, kv, o, do, topk_indices, lse, attn_sink, kv_lora_rank, scale, **kwargs):
        return sparse_mla_bwd_triton(
            q, kv, o, do, topk_indices, lse, attn_sink=attn_sink, kv_lora_rank=kv_lora_rank, scale=scale
        )


# =============================================================================
# Dispatchers
# =============================================================================


class SparseMlaFwdDispatcher(AutoKernelDispatcher):
    _backends = {
        BackendType.FLYDSL: BackendEntry(SparseMlaFwdFlydslBackend),
        BackendType.TRITON: BackendEntry(SparseMlaFwdTritonBackend),
    }
    _cache = TuneCache(1024)

    @classmethod
    def make_key(cls, q, kv, topk_indices, attn_sink=None, **kwargs):
        # A sink changes which kernel each backend builds, so it belongs in the key.
        return (*q.shape, kv.shape[0], topk_indices.shape[1], q.dtype, attn_sink is not None)


class SparseMlaBwdDispatcher(AutoKernelDispatcher):
    _backends = {
        BackendType.FLYDSL: BackendEntry(SparseMlaBwdFlydslBackend),
        BackendType.TRITON: BackendEntry(SparseMlaBwdTritonBackend),
    }
    _cache = TuneCache(1024)

    @classmethod
    def make_key(cls, q, kv, topk_indices, attn_sink=None, **kwargs):
        return (*q.shape, kv.shape[0], topk_indices.shape[1], q.dtype, attn_sink is not None)


# =============================================================================
# Custom ops
# =============================================================================


@torch.library.custom_op("primus_turbo::sparse_mla_fwd_impl", mutates_args=(), device_types="cuda")
def sparse_mla_fwd_impl(
    q: torch.Tensor,
    kv: torch.Tensor,
    topk_indices: torch.Tensor,
    attn_sink: Optional[torch.Tensor],
    kv_lora_rank: int,
    scale: Optional[float],
    default_backend: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sparse-MLA forward: o [T, H, kv_lora_rank] and a sink-inclusive lse [T, H] fp32."""
    return SparseMlaFwdDispatcher.dispatch(
        BackendChoice(BackendType(default_backend)),
        GlobalBackendManager.get_sparse_attn_backend(PrecisionType.BF16_FP16_FP32),
        q=q,
        kv=kv,
        topk_indices=topk_indices,
        attn_sink=attn_sink,
        kv_lora_rank=kv_lora_rank,
        scale=scale,
    )


@sparse_mla_fwd_impl.register_fake
def sparse_mla_fwd_impl_meta(
    q: torch.Tensor,
    kv: torch.Tensor,
    topk_indices: torch.Tensor,
    attn_sink: Optional[torch.Tensor],
    kv_lora_rank: int,
    scale: Optional[float],
    default_backend: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    total_tokens, num_heads, _ = q.shape
    return (
        q.new_empty((total_tokens, num_heads, kv_lora_rank)),
        q.new_empty((total_tokens, num_heads), dtype=torch.float32),
    )


@torch.library.custom_op("primus_turbo::sparse_mla_bwd_impl", mutates_args=(), device_types="cuda")
def sparse_mla_bwd_impl(
    q: torch.Tensor,
    kv: torch.Tensor,
    o: torch.Tensor,
    do: torch.Tensor,
    topk_indices: torch.Tensor,
    lse: torch.Tensor,
    attn_sink: Optional[torch.Tensor],
    kv_lora_rank: int,
    scale: Optional[float],
    default_backend: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sparse-MLA backward: dq, dkv, and dsink [H] fp32, empty without a sink since the op
    schema returns a fixed number of tensors. dkv is viewed back to the caller's rank --
    both kernels shape it after unsqueezing a 2-D kv, and autograd rejects a grad that is
    not the shape of its input."""
    dq, dkv, dsink = SparseMlaBwdDispatcher.dispatch(
        BackendChoice(BackendType(default_backend)),
        GlobalBackendManager.get_sparse_attn_backend(PrecisionType.BF16_FP16_FP32),
        q=q,
        kv=kv,
        o=o,
        do=do,
        topk_indices=topk_indices,
        lse=lse,
        attn_sink=attn_sink,
        kv_lora_rank=kv_lora_rank,
        scale=scale,
    )
    dsink = dsink if dsink is not None else q.new_empty((0,), dtype=torch.float32)
    return dq, dkv.view_as(kv), dsink


@sparse_mla_bwd_impl.register_fake
def sparse_mla_bwd_impl_meta(
    q: torch.Tensor,
    kv: torch.Tensor,
    o: torch.Tensor,
    do: torch.Tensor,
    topk_indices: torch.Tensor,
    lse: torch.Tensor,
    attn_sink: Optional[torch.Tensor],
    kv_lora_rank: int,
    scale: Optional[float],
    default_backend: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dsink_shape = (q.shape[1],) if attn_sink is not None else (0,)
    return (
        torch.empty_like(q),
        torch.empty_like(kv),
        q.new_empty(dsink_shape, dtype=torch.float32),
    )
