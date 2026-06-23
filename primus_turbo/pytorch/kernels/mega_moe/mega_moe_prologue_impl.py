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

from primus_turbo.flydsl.mega.mega_moe_prologue_kernel import mega_moe_prologue
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
        peer_ptrs: torch.Tensor,
        origin_rank: torch.Tensor,
        origin_slot: torch.Tensor,
        meta_scalars: torch.Tensor,
        grid_barrier_state: torch.Tensor,
        profile: torch.Tensor,
        scoreboard: torch.Tensor,
        barrier_local: torch.Tensor,
        num_tokens: int,
        num_topk: int,
        num_experts: int,
        world_size: int,
        rank: int,
        experts_per_rank: int,
        block_m: int,
        pool_capacity: int,
        no_cpu_sync: bool,
        num_cu: int = 64,
        **kwargs,
    ):
        # The primitive allocates + returns its plan output tables internally; pass them
        # straight through. (launch_cache=None: each call is a normal launch; the kernel
        # self-resets its counters, so re-launch is idempotent.) scoreboard + barrier_local
        # are reset inside the kernel (sb_l2 / comb stay host-zeroed).
        plan, ttg, exp, _orank, _oslot, _npb, _mnt = mega_moe_prologue(
            topk_idx,
            topk_w,
            peer_ptrs=peer_ptrs,
            origin_rank=origin_rank,
            origin_slot=origin_slot,
            meta_scalars=meta_scalars,
            grid_barrier_state=grid_barrier_state,
            profile=profile,
            scoreboard=scoreboard,
            barrier_local=barrier_local,
            num_tokens=int(num_tokens),
            num_topk=int(num_topk),
            num_experts=int(num_experts),
            world_size=int(world_size),
            rank=int(rank),
            experts_per_rank=int(experts_per_rank),
            block_m=int(block_m),
            pool_capacity=int(pool_capacity),
            no_cpu_sync=bool(no_cpu_sync),
            num_cu=int(num_cu),
        )
        # plan = (dst_rank, dst_offset, count, src_offset, src_tokens, topk_slot, weight)
        return (*plan, ttg, exp)


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
        num_cu=64,
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
            int(num_cu),
            topk_idx.dtype,
        )


_torch_custom_op_wrapper = torch.library.custom_op

# In-place buffers the kernel writes on this rank: the symmetric origin tables
# (written cross-rank via *_ptrs), the device meta scalars, and the barrier/profile
# scratch. The plan output tables are allocated + RETURNED (not mutated inputs).
_MUTATED_ARGS = (
    "origin_rank",
    "origin_slot",
    "meta_scalars",
    "grid_barrier_state",
    "profile",
    "scoreboard",
    "barrier_local",
)

# return order mirrors the FlyDSL primitive: the flat dispatch plan tensors
#   (dst_rank, dst_offset, count, src_offset, src_tokens, topk_slot, weight)
# followed by tile_to_group + expected. origin tables are mutated in place; the
# scalar returns (num_pool_blocks / max_num_token) are static -> derived by callers.
_RET = tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]


@_torch_custom_op_wrapper(
    "primus_turbo::mega_moe_prologue_impl",
    mutates_args=_MUTATED_ARGS,
    device_types="cuda",
)
def mega_moe_prologue_impl(
    topk_idx: torch.Tensor,
    topk_w: Optional[torch.Tensor],
    peer_ptrs: torch.Tensor,
    origin_rank: torch.Tensor,
    origin_slot: torch.Tensor,
    meta_scalars: torch.Tensor,
    grid_barrier_state: torch.Tensor,
    profile: torch.Tensor,
    scoreboard: torch.Tensor,
    barrier_local: torch.Tensor,
    num_tokens: int,
    num_topk: int,
    num_experts: int,
    world_size: int,
    rank: int,
    block_m: int,
    pool_capacity: int,
    default_backend: int,
    no_cpu_sync: bool = True,
    autotune: bool = False,
    num_cu: int = 64,
) -> _RET:
    """Fused MoE dispatch-prologue: build the full EP dispatch plan.

    Allocates + RETURNS the dispatch plan output tables (dst_rank / dst_offset /
    count / src_offset / src_tokens / topk_slot / weight + tile_to_group / expected),
    mirroring the FlyDSL primitive. The symmetric origin tables + device meta scalars
    are mutated in place (origin writes cross-rank via raw ``*_ptrs``, untracked).
    scoreboard + barrier_local are reset inside the kernel before its final cross-rank
    barrier; sb_l2 / comb stay host-zeroed (comb is large -> faster as a full-grid memset).
    """
    default_backend_enum = BackendType(default_backend)
    experts_per_rank = int(num_experts) // int(world_size)

    # kernel autotune: explicit flag OR global auto-tune, never under capture
    do_autotune = autotune or GlobalBackendManager.auto_tune_enabled()
    do_autotune = do_autotune and not AutoKernelDispatcher._is_graph_capturing()

    kwargs = dict(
        topk_idx=topk_idx,
        topk_w=topk_w,
        peer_ptrs=peer_ptrs,
        origin_rank=origin_rank,
        origin_slot=origin_slot,
        meta_scalars=meta_scalars,
        grid_barrier_state=grid_barrier_state,
        profile=profile,
        scoreboard=scoreboard,
        barrier_local=barrier_local,
        num_tokens=num_tokens,
        num_topk=num_topk,
        num_experts=num_experts,
        world_size=world_size,
        rank=rank,
        experts_per_rank=experts_per_rank,
        block_m=block_m,
        pool_capacity=pool_capacity,
        no_cpu_sync=no_cpu_sync,
        autotune=do_autotune,
        num_cu=num_cu,
    )

    return MegaMoePrologueKernelDispatcher.dispatch(default_backend_enum, None, **kwargs)


@mega_moe_prologue_impl.register_fake
def mega_moe_prologue_impl_meta(
    topk_idx: torch.Tensor,
    topk_w: Optional[torch.Tensor],
    peer_ptrs: torch.Tensor,
    origin_rank: torch.Tensor,
    origin_slot: torch.Tensor,
    meta_scalars: torch.Tensor,
    grid_barrier_state: torch.Tensor,
    profile: torch.Tensor,
    scoreboard: torch.Tensor,
    barrier_local: torch.Tensor,
    num_tokens: int,
    num_topk: int,
    num_experts: int,
    world_size: int,
    rank: int,
    block_m: int,
    pool_capacity: int,
    default_backend: int,
    no_cpu_sync: bool = True,
    autotune: bool = False,
    num_cu: int = 64,
) -> _RET:
    assert topk_idx.dim() == 2, f"topk_idx must be 2D [T,K], got {topk_idx.shape}"
    assert topk_idx.shape[-1] == int(num_topk), "topk_idx last dim must equal num_topk"
    assert topk_idx.dtype in _SUPPORTED_INDEX_DTYPES, f"topk_idx dtype {topk_idx.dtype} unsupported"
    assert int(num_experts) % int(world_size) == 0, "num_experts must be divisible by world_size"
    assert int(pool_capacity) % int(block_m) == 0, "pool_capacity must be a multiple of block_m"
    E, P, M = int(num_experts), int(pool_capacity), int(pool_capacity) // int(block_m)

    def i32(n):
        return torch.empty(n, dtype=torch.int32, device=topk_idx.device)

    return (
        # dst_rank, dst_offset, count, src_offset
        i32(E),
        i32(E),
        i32(E),
        i32(E),
        i32(P),
        i32(P),  # src_tokens, topk_slot
        torch.empty(P, dtype=torch.float32, device=topk_idx.device),  # weight
        i32(M),
        i32(M),  # tile_to_group, expected
    )
