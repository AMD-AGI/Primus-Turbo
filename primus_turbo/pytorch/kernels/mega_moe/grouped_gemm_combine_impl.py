###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Fused grouped GEMM + combine PUSH (+ optional topk reduce) as a torch custom op.

Mirrors ``dispatch_grouped_gemm_impl.py``: a ``torch.library.custom_op`` makes the
FlyDSL combine kernel ``torch.compile``-traceable (opaque mutating op + fake meta),
and an ``AutoKernelDispatcher`` selects / auto-tunes the backend. The kernel runs
grouped BF16 GEMM into the caller-owned ``l2y`` [pool_capacity, N], pushes each
finished row cross-rank into the origin rank's combine buffer, and (when reduce
args are supplied) sums each token's topk rows into ``output`` — all mutated in
place; see ``primus_turbo.flydsl.mega.grouped_gemm_combine_bf16_kernel``.
"""

from typing import Optional

import torch

from primus_turbo.pytorch.core.backend import (
    AutoKernelDispatcher,
    BackendEntry,
    BackendType,
    GlobalBackendManager,
    KernelBackend,
    TuneCache,
)

_SUPPORTED_DTYPES = (torch.bfloat16,)


def _flydsl_kernel():
    """Lazy import — keep this module importable when FlyDSL is absent."""
    from primus_turbo.flydsl.mega.grouped_gemm_combine_bf16_kernel import (
        grouped_gemm_combine_bf16,
    )

    return grouped_gemm_combine_bf16


class GroupedGEMMCombineFlyDSLBackend(KernelBackend):
    """FlyDSL fused grouped BF16 GEMM + combine PUSH (+ optional topk reduce)."""

    @staticmethod
    def can_handle(
        act: torch.Tensor,
        weight: torch.Tensor,
        layout: str,
        **kwargs,
    ) -> bool:
        supported = True
        supported &= act.dim() == 2 and weight.dim() == 3
        supported &= act.dtype in _SUPPORTED_DTYPES and weight.dtype in _SUPPORTED_DTYPES
        supported &= layout in ("nt", "nn", "tn")
        # K (contraction) must match between act and weight per layout
        act_k = act.shape[0] if layout == "tn" else act.shape[1]
        w_k = weight.shape[2] if layout == "nt" else weight.shape[1]
        supported &= act_k == w_k
        return supported

    @staticmethod
    def execute(
        act: torch.Tensor,
        weight: torch.Tensor,
        l2y: torch.Tensor,
        tile_to_group: torch.Tensor,
        sb_l2: torch.Tensor,
        origin_rank: torch.Tensor,
        origin_slot: torch.Tensor,
        comb_addrs: torch.Tensor,
        num_tile_blocks: torch.Tensor,
        output: Optional[torch.Tensor],
        comb_local: Optional[torch.Tensor],
        barrier_local: Optional[torch.Tensor],
        barrier_addrs: Optional[torch.Tensor],
        topk_indices: Optional[torch.Tensor],
        num_tokens_per_rank: Optional[torch.Tensor],
        combine_slots: int,
        topk: int,
        num_experts: int,
        rank: int,
        layout: str,
        BM: int,
        BN: int,
        num_combine_cu: int,
        num_reduce_cu: int,
        nt_vmcnt: int,
        waves_per_eu: int,
        agpr_alloc: int,
        autotune: bool,
        **kwargs,
    ):
        kernel = _flydsl_kernel()
        kernel(
            act,
            weight,
            l2y,
            tile_to_group,
            sb_l2,
            origin_rank,
            origin_slot,
            comb_addrs,
            int(combine_slots),
            num_tile_blocks,
            output=output,
            comb_local=comb_local,
            barrier_local=barrier_local,
            barrier_addrs=barrier_addrs,
            topk_indices=topk_indices,
            num_tokens_per_rank=num_tokens_per_rank,
            topk=int(topk),
            num_experts=int(num_experts),
            rank=int(rank),
            layout=layout,
            BM=int(BM),
            BN=int(BN),
            num_combine_cu=int(num_combine_cu),
            num_reduce_cu=int(num_reduce_cu),
            nt_vmcnt=int(nt_vmcnt),
            waves_per_eu=int(waves_per_eu),
            agpr_alloc=int(agpr_alloc),
            autotune=bool(autotune),
            autotune_reset=sb_l2.zero_,
        )
        return None


_GROUPED_GEMM_COMBINE_BACKENDS = {
    # autotune is kernel-internal; skip framework-level backend profiling
    BackendType.FLYDSL: BackendEntry(GroupedGEMMCombineFlyDSLBackend, autotune=False),
}


class GroupedGEMMCombineKernelDispatcher(AutoKernelDispatcher):
    _backends = _GROUPED_GEMM_COMBINE_BACKENDS
    _cache = TuneCache(1024)

    @classmethod
    def make_key(
        cls, act, weight, layout, combine_slots, num_reduce_cu, topk, num_experts, rank, BM, BN, **kwargs
    ):
        G = weight.shape[0]
        n_idx, k_idx = (1, 2) if layout == "nt" else (2, 1)
        N, K = weight.shape[n_idx], weight.shape[k_idx]
        pool_capacity = act.shape[1] if layout == "tn" else act.shape[0]
        return (
            G,
            N,
            K,
            pool_capacity,
            BM,
            BN,
            int(combine_slots),
            int(num_reduce_cu),
            int(topk),
            int(num_experts),
            int(rank),
            layout,
            act.dtype,
        )


_torch_custom_op_wrapper = torch.library.custom_op

# Caller-owned buffers the kernel writes on this rank. l2y holds the grouped-GEMM
# output (and is the combine PUSH source); sb_l2 (scoreboard) is bumped by the GEMM
# role; output/comb_local/barrier_local are the reduce path's local targets.
# Cross-rank writes via comb_addrs/barrier_addrs are raw-pointer untracked.
_MUTATED_ARGS = ("l2y", "sb_l2", "output", "comb_local", "barrier_local")


@_torch_custom_op_wrapper(
    "primus_turbo::grouped_gemm_combine_impl",
    mutates_args=_MUTATED_ARGS,
    device_types="cuda",
)
def grouped_gemm_combine_impl(
    act: torch.Tensor,
    weight: torch.Tensor,
    l2y: torch.Tensor,
    tile_to_group: torch.Tensor,
    sb_l2: torch.Tensor,
    origin_rank: torch.Tensor,
    origin_slot: torch.Tensor,
    comb_addrs: torch.Tensor,
    num_tile_blocks: torch.Tensor,
    output: Optional[torch.Tensor],
    comb_local: Optional[torch.Tensor],
    barrier_local: Optional[torch.Tensor],
    combine_slots: int,
    default_backend: int,
    barrier_addrs: Optional[torch.Tensor] = None,
    topk_indices: Optional[torch.Tensor] = None,
    num_tokens_per_rank: Optional[torch.Tensor] = None,
    topk: int = 1,
    num_experts: int = 0,
    rank: int = 0,
    layout: str = "nt",
    BM: int = 256,
    BN: int = 256,
    num_combine_cu: int = 32,
    num_reduce_cu: int = 0,
    nt_vmcnt: int = 3,
    waves_per_eu: int = 2,
    agpr_alloc: int = 0,
    autotune: bool = False,
) -> None:
    """Fused grouped BF16 GEMM + combine PUSH (+ optional topk reduce), in place.

    The grouped-GEMM result is written into the caller-owned ``l2y`` [pool_capacity,
    N] (also the combine PUSH source); ``sb_l2`` must be zeroed before each call.
    Supplying ``output`` + ``num_reduce_cu > 0`` enables the 3-role topk reduce,
    writing the final result into ``output``. Read ``l2y``/``output`` after the
    call. Cross-rank writes via ``comb_addrs``/``barrier_addrs`` are raw-pointer
    and untracked by autograd.
    """
    default_backend_enum = BackendType(default_backend)

    # kernel autotune: explicit flag OR global auto-tune, never under capture
    do_autotune = autotune or GlobalBackendManager.auto_tune_enabled()
    do_autotune = do_autotune and not AutoKernelDispatcher._is_graph_capturing()

    kwargs = dict(
        act=act,
        weight=weight,
        l2y=l2y,
        tile_to_group=tile_to_group,
        sb_l2=sb_l2,
        origin_rank=origin_rank,
        origin_slot=origin_slot,
        comb_addrs=comb_addrs,
        num_tile_blocks=num_tile_blocks,
        output=output,
        comb_local=comb_local,
        barrier_local=barrier_local,
        barrier_addrs=barrier_addrs,
        topk_indices=topk_indices,
        num_tokens_per_rank=num_tokens_per_rank,
        combine_slots=combine_slots,
        topk=topk,
        num_experts=num_experts,
        rank=rank,
        layout=layout,
        BM=BM,
        BN=BN,
        num_combine_cu=num_combine_cu,
        num_reduce_cu=num_reduce_cu,
        nt_vmcnt=nt_vmcnt,
        waves_per_eu=waves_per_eu,
        agpr_alloc=agpr_alloc,
        autotune=do_autotune,
    )

    GroupedGEMMCombineKernelDispatcher.dispatch(default_backend_enum, None, **kwargs)


@grouped_gemm_combine_impl.register_fake
def grouped_gemm_combine_impl_meta(
    act: torch.Tensor,
    weight: torch.Tensor,
    l2y: torch.Tensor,
    tile_to_group: torch.Tensor,
    sb_l2: torch.Tensor,
    origin_rank: torch.Tensor,
    origin_slot: torch.Tensor,
    comb_addrs: torch.Tensor,
    num_tile_blocks: torch.Tensor,
    output: Optional[torch.Tensor],
    comb_local: Optional[torch.Tensor],
    barrier_local: Optional[torch.Tensor],
    combine_slots: int,
    default_backend: int,
    barrier_addrs: Optional[torch.Tensor] = None,
    topk_indices: Optional[torch.Tensor] = None,
    num_tokens_per_rank: Optional[torch.Tensor] = None,
    topk: int = 1,
    num_experts: int = 0,
    rank: int = 0,
    layout: str = "nt",
    BM: int = 256,
    BN: int = 256,
    num_combine_cu: int = 32,
    num_reduce_cu: int = 0,
    nt_vmcnt: int = 3,
    waves_per_eu: int = 2,
    agpr_alloc: int = 0,
    autotune: bool = False,
) -> None:
    assert act.dim() == 2, f"act must be 2D, got {act.shape}"
    assert weight.dim() == 3, f"weight must be 3D, got {weight.shape}"
    assert l2y.dim() == 2, f"l2y must be 2D, got {l2y.shape}"
    assert act.dtype in _SUPPORTED_DTYPES, f"act must be bf16, got {act.dtype}"
    assert weight.dtype in _SUPPORTED_DTYPES, f"weight must be bf16, got {weight.dtype}"
    assert layout in ("nt", "nn", "tn"), f"unknown layout {layout}"
    return None
