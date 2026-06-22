###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Fused MoE dispatch-prologue as a torch custom op + multi-backend dispatcher.

Mirrors ``dispatch_grouped_gemm_impl.py``: a ``torch.library.custom_op`` makes the
FlyDSL prologue kernel ``torch.compile``-traceable (opaque mutating op + fake
meta), and an ``AutoKernelDispatcher`` selects/auto-tunes the backend. The kernel
builds the whole EP dispatch plan (counts / pool layout / comm tasks /
tile_to_group / expected / origin tables) over caller-owned (symmetric) buffers
mutated in place; see ``primus_turbo.flydsl.mega.mega_moe_prologue``.
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

# topk_idx may be int32/int64; routing weight float; everything else metadata int32
_SUPPORTED_INDEX_DTYPES = (torch.int32, torch.int64)


def _flydsl_kernel():
    """Lazy import — keep this module importable when FlyDSL is absent."""
    from primus_turbo.flydsl.mega.mega_moe_prologue import mega_moe_prologue

    return mega_moe_prologue


class MegaMoePrologueFlyDSLBackend(KernelBackend):
    """FlyDSL persistent fused MoE dispatch-prologue (table build + scatter)."""

    @staticmethod
    def can_handle(
        topk_idx: torch.Tensor,
        num_topk: int,
        num_experts: int,
        world_size: int,
        block_m: int,
        pool_capacity: int,
        **kwargs,
    ) -> bool:
        supported = True
        supported &= topk_idx.dim() == 2 and topk_idx.shape[-1] == int(num_topk)
        supported &= topk_idx.dtype in _SUPPORTED_INDEX_DTYPES
        supported &= int(num_experts) % int(world_size) == 0
        supported &= int(pool_capacity) % int(block_m) == 0
        return supported

    @staticmethod
    def execute(
        topk_idx: torch.Tensor,
        topk_w,
        send_local: torch.Tensor,
        within_expert_counter: torch.Tensor,
        c_buffer_ptrs: torch.Tensor,
        signal_ptrs: torch.Tensor,
        origin_rank_ptrs: torch.Tensor,
        origin_slot_ptrs: torch.Tensor,
        start_per_expert: torch.Tensor,
        source_offset_per_expert: torch.Tensor,
        pool_base: torch.Tensor,
        destination: torch.Tensor,
        start: torch.Tensor,
        count: torch.Tensor,
        source_offset_out: torch.Tensor,
        tile_to_group: torch.Tensor,
        expected: torch.Tensor,
        source_tokens: torch.Tensor,
        source_topk_slot: torch.Tensor,
        source_weight: torch.Tensor,
        zero_topk_weights: torch.Tensor,
        origin_rank: torch.Tensor,
        origin_slot: torch.Tensor,
        meta_scalars: torch.Tensor,
        grid_barrier_state: torch.Tensor,
        token_rank_table: torch.Tensor,
        dedup_src_row_ptrs: torch.Tensor,
        dedup_src_row: torch.Tensor,
        source_dedup: torch.Tensor,
        profile: torch.Tensor,
        num_tokens: int,
        num_topk: int,
        num_experts: int,
        world_size: int,
        rank: int,
        experts_per_rank: int,
        block_m: int,
        pool_capacity: int,
        dedup: bool,
        no_cpu_sync: bool,
        **kwargs,
    ):
        kernel = _flydsl_kernel()
        # launch_cache=None: each call is a normal launch (the kernel self-resets
        # its counters, so re-launch is idempotent and autotune-profile safe).
        kernel(
            topk_idx,
            topk_w,
            send_local=send_local,
            within_expert_counter=within_expert_counter,
            c_buffer_ptrs=c_buffer_ptrs,
            signal_ptrs=signal_ptrs,
            origin_rank_ptrs=origin_rank_ptrs,
            origin_slot_ptrs=origin_slot_ptrs,
            start_per_expert=start_per_expert,
            source_offset_per_expert=source_offset_per_expert,
            pool_base=pool_base,
            destination=destination,
            start=start,
            count=count,
            source_offset_out=source_offset_out,
            tile_to_group=tile_to_group,
            expected=expected,
            source_tokens=source_tokens,
            source_topk_slot=source_topk_slot,
            source_weight=source_weight,
            zero_topk_weights=zero_topk_weights,
            origin_rank=origin_rank,
            origin_slot=origin_slot,
            meta_scalars=meta_scalars,
            grid_barrier_state=grid_barrier_state,
            token_rank_table=token_rank_table,
            dedup_src_row_ptrs=dedup_src_row_ptrs,
            dedup_src_row=dedup_src_row,
            source_dedup=source_dedup,
            profile=profile,
            num_tokens=int(num_tokens),
            num_topk=int(num_topk),
            num_experts=int(num_experts),
            world_size=int(world_size),
            rank=int(rank),
            experts_per_rank=int(experts_per_rank),
            block_m=int(block_m),
            pool_capacity=int(pool_capacity),
            dedup=bool(dedup),
            no_cpu_sync=bool(no_cpu_sync),
        )
        return None


_MEGA_MOE_PROLOGUE_BACKENDS = {
    # Lone backend; framework profiling off (cross-rank handshake would deadlock
    # world>1, and no second backend to compare). Dispatcher still supports it.
    BackendType.FLYDSL: BackendEntry(MegaMoePrologueFlyDSLBackend, autotune=False),
}


class MegaMoePrologueKernelDispatcher(AutoKernelDispatcher):
    _backends = _MEGA_MOE_PROLOGUE_BACKENDS
    _cache = TuneCache(1024)

    @classmethod
    def make_key(
        cls,
        topk_idx,
        num_tokens,
        num_topk,
        num_experts,
        world_size,
        rank,
        block_m,
        pool_capacity,
        dedup,
        **kwargs,
    ):
        return (
            int(num_tokens),
            int(num_topk),
            int(num_experts),
            int(world_size),
            int(rank),
            int(block_m),
            int(pool_capacity),
            bool(dedup),
            topk_idx.dtype,
        )


_torch_custom_op_wrapper = torch.library.custom_op

# Buffers the kernel writes on this rank (direct or own peer-pointer row).
# c_buffer/signal are cross-rank scratch reached only via *_ptrs -> untracked.
_MUTATED_ARGS = (
    "send_local",
    "within_expert_counter",
    "start_per_expert",
    "source_offset_per_expert",
    "pool_base",
    "destination",
    "start",
    "count",
    "source_offset_out",
    "tile_to_group",
    "expected",
    "source_tokens",
    "source_topk_slot",
    "source_weight",
    "origin_rank",
    "origin_slot",
    "meta_scalars",
    "grid_barrier_state",
    "token_rank_table",
    "dedup_src_row",
    "source_dedup",
    "profile",
)


@_torch_custom_op_wrapper(
    "primus_turbo::mega_moe_prologue_impl",
    mutates_args=_MUTATED_ARGS,
    device_types="cuda",
)
def mega_moe_prologue_impl(
    topk_idx: torch.Tensor,
    topk_w: Optional[torch.Tensor],
    send_local: torch.Tensor,
    within_expert_counter: torch.Tensor,
    c_buffer_ptrs: torch.Tensor,
    signal_ptrs: torch.Tensor,
    origin_rank_ptrs: torch.Tensor,
    origin_slot_ptrs: torch.Tensor,
    start_per_expert: torch.Tensor,
    source_offset_per_expert: torch.Tensor,
    pool_base: torch.Tensor,
    destination: torch.Tensor,
    start: torch.Tensor,
    count: torch.Tensor,
    source_offset_out: torch.Tensor,
    tile_to_group: torch.Tensor,
    expected: torch.Tensor,
    source_tokens: torch.Tensor,
    source_topk_slot: torch.Tensor,
    source_weight: torch.Tensor,
    zero_topk_weights: torch.Tensor,
    origin_rank: torch.Tensor,
    origin_slot: torch.Tensor,
    meta_scalars: torch.Tensor,
    grid_barrier_state: torch.Tensor,
    token_rank_table: torch.Tensor,
    dedup_src_row_ptrs: torch.Tensor,
    dedup_src_row: torch.Tensor,
    source_dedup: torch.Tensor,
    profile: torch.Tensor,
    num_tokens: int,
    num_topk: int,
    num_experts: int,
    world_size: int,
    rank: int,
    block_m: int,
    pool_capacity: int,
    default_backend: int,
    dedup: bool = False,
    no_cpu_sync: bool = True,
    autotune: bool = False,
) -> None:
    """Fused MoE dispatch-prologue: build the full EP dispatch plan in place.

    All metadata buffers are mutated in place over caller-owned (symmetric)
    memory; the caller MUST keep the buffers alive and may read them after the
    call. Cross-rank writes via ``*_ptrs`` tables are raw-pointer and untracked
    by autograd. Returns nothing — the plan lives in the mutated buffers.
    """
    default_backend_enum = BackendType(default_backend)
    experts_per_rank = int(num_experts) // int(world_size)

    # kernel autotune: explicit flag OR global auto-tune, never under capture
    do_autotune = autotune or GlobalBackendManager.auto_tune_enabled()
    do_autotune = do_autotune and not AutoKernelDispatcher._is_graph_capturing()

    kwargs = dict(
        topk_idx=topk_idx,
        topk_w=topk_w,
        send_local=send_local,
        within_expert_counter=within_expert_counter,
        c_buffer_ptrs=c_buffer_ptrs,
        signal_ptrs=signal_ptrs,
        origin_rank_ptrs=origin_rank_ptrs,
        origin_slot_ptrs=origin_slot_ptrs,
        start_per_expert=start_per_expert,
        source_offset_per_expert=source_offset_per_expert,
        pool_base=pool_base,
        destination=destination,
        start=start,
        count=count,
        source_offset_out=source_offset_out,
        tile_to_group=tile_to_group,
        expected=expected,
        source_tokens=source_tokens,
        source_topk_slot=source_topk_slot,
        source_weight=source_weight,
        zero_topk_weights=zero_topk_weights,
        origin_rank=origin_rank,
        origin_slot=origin_slot,
        meta_scalars=meta_scalars,
        grid_barrier_state=grid_barrier_state,
        token_rank_table=token_rank_table,
        dedup_src_row_ptrs=dedup_src_row_ptrs,
        dedup_src_row=dedup_src_row,
        source_dedup=source_dedup,
        profile=profile,
        num_tokens=num_tokens,
        num_topk=num_topk,
        num_experts=num_experts,
        world_size=world_size,
        rank=rank,
        experts_per_rank=experts_per_rank,
        block_m=block_m,
        pool_capacity=pool_capacity,
        dedup=dedup,
        no_cpu_sync=no_cpu_sync,
        autotune=do_autotune,
    )

    MegaMoePrologueKernelDispatcher.dispatch(default_backend_enum, None, **kwargs)


@mega_moe_prologue_impl.register_fake
def mega_moe_prologue_impl_meta(
    topk_idx: torch.Tensor,
    topk_w: Optional[torch.Tensor],
    send_local: torch.Tensor,
    within_expert_counter: torch.Tensor,
    c_buffer_ptrs: torch.Tensor,
    signal_ptrs: torch.Tensor,
    origin_rank_ptrs: torch.Tensor,
    origin_slot_ptrs: torch.Tensor,
    start_per_expert: torch.Tensor,
    source_offset_per_expert: torch.Tensor,
    pool_base: torch.Tensor,
    destination: torch.Tensor,
    start: torch.Tensor,
    count: torch.Tensor,
    source_offset_out: torch.Tensor,
    tile_to_group: torch.Tensor,
    expected: torch.Tensor,
    source_tokens: torch.Tensor,
    source_topk_slot: torch.Tensor,
    source_weight: torch.Tensor,
    zero_topk_weights: torch.Tensor,
    origin_rank: torch.Tensor,
    origin_slot: torch.Tensor,
    meta_scalars: torch.Tensor,
    grid_barrier_state: torch.Tensor,
    token_rank_table: torch.Tensor,
    dedup_src_row_ptrs: torch.Tensor,
    dedup_src_row: torch.Tensor,
    source_dedup: torch.Tensor,
    profile: torch.Tensor,
    num_tokens: int,
    num_topk: int,
    num_experts: int,
    world_size: int,
    rank: int,
    block_m: int,
    pool_capacity: int,
    default_backend: int,
    dedup: bool = False,
    no_cpu_sync: bool = True,
    autotune: bool = False,
) -> None:
    assert topk_idx.dim() == 2, f"topk_idx must be 2D [T,K], got {topk_idx.shape}"
    assert topk_idx.shape[-1] == int(num_topk), "topk_idx last dim must equal num_topk"
    assert topk_idx.dtype in _SUPPORTED_INDEX_DTYPES, f"topk_idx dtype {topk_idx.dtype} unsupported"
    assert int(num_experts) % int(world_size) == 0, "num_experts must be divisible by world_size"
    assert int(pool_capacity) % int(block_m) == 0, "pool_capacity must be a multiple of block_m"
    return None
