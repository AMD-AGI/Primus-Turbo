###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Heuristic config selection for the Mega MoE kernel.

This is the Python counterpart of DeepGEMM's
``csrc/jit_kernels/heuristics/mega_moe.hpp``.  The C++ side keeps its
own copy that the kernel launcher reads; the Python version here is
exposed for introspection, autotuning sweeps and debugging without
having to round-trip through the C extension.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = ["MegaMoEConfig", "get_mega_moe_config"]


@dataclass
class MegaMoEConfig:
    """Selected mega-MoE tile / pipeline configuration."""

    # Block tiling.
    block_m: int
    block_n: int
    block_k: int
    load_block_m: int
    load_block_n: int
    store_block_m: int

    # SF (scaling factor) tile sizes.
    sf_block_m: int
    sf_block_n: int

    # Token-pool capacity (after top-k fan-out) and SF padded variant.
    num_max_pool_tokens: int
    num_padded_sf_pool_tokens: int

    # LDS swizzle mode for activations / weights (analogous to DG's TMA
    # swizzle mode; 0 means no swizzle and the kernel falls back to the
    # default layout).
    swizzle_acts_mode: int
    swizzle_weights_mode: int

    # Persistent-CTA wave granularity.
    num_experts_per_wave: int

    # Pipeline depth + total LDS bytes per CTA.
    num_stages: int
    smem_size: int

    # Warp partition inside one persistent CTA.
    num_dispatch_threads: int
    num_non_epilogue_threads: int
    num_epilogue_threads: int

    def __str__(self) -> str:  # pragma: no cover - debug only
        return (
            "MegaMoEConfig("
            f"block_m={self.block_m}, block_n={self.block_n}, block_k={self.block_k}, "
            f"load_block_m={self.load_block_m}, load_block_n={self.load_block_n}, "
            f"store_block_m={self.store_block_m}, "
            f"sf_block_m={self.sf_block_m}, sf_block_n={self.sf_block_n}, "
            f"num_max_pool_tokens={self.num_max_pool_tokens}, "
            f"num_padded_sf_pool_tokens={self.num_padded_sf_pool_tokens}, "
            f"swizzle_acts_mode={self.swizzle_acts_mode}, "
            f"swizzle_weights_mode={self.swizzle_weights_mode}, "
            f"num_experts_per_wave={self.num_experts_per_wave}, "
            f"num_stages={self.num_stages}, smem_size={self.smem_size}, "
            f"num_dispatch_threads={self.num_dispatch_threads}, "
            f"num_non_epilogue_threads={self.num_non_epilogue_threads}, "
            f"num_epilogue_threads={self.num_epilogue_threads})"
        )


def _pick_block_m(num_expected_tokens_per_expert: float) -> tuple[int, int]:
    """Choose block_m and store_block_m based on per-expert token load.

    Mirrors the bucketing logic from DeepGEMM's
    ``get_block_config_for_mega_moe`` but tuned for AMD MFMA tiles
    (placeholders; revisit when the kernel is implemented).
    """

    if num_expected_tokens_per_expert <= 8.5:
        return 16, 8
    if num_expected_tokens_per_expert <= 16.5:
        return 32, 16
    if num_expected_tokens_per_expert <= 32.5:
        return 64, 32
    if num_expected_tokens_per_expert <= 64.5:
        return 96, 16
    if num_expected_tokens_per_expert <= 96.5:
        return 128, 32
    return 192, 32


def _pick_num_experts_per_wave(
    num_experts_per_rank: int,
    num_tokens: int,
    num_topk: int,
    intermediate_hidden: int,
    block_m: int,
    block_n: int,
    num_cus: int,
) -> int:
    expected_tokens_per_expert = max(num_tokens, 1) * num_topk / max(num_experts_per_rank, 1)
    if expected_tokens_per_expert < 1:
        return num_experts_per_rank

    imbalance_factor = 2
    num_m_blocks = max(1, math.ceil(math.ceil(expected_tokens_per_expert) / block_m))
    num_n_blocks = max(1, (2 * intermediate_hidden) // block_n)
    blocks_per_expert = num_m_blocks * num_n_blocks
    if blocks_per_expert == 0:
        return num_experts_per_rank

    num_per_wave = max(1, math.ceil(imbalance_factor * num_cus / blocks_per_expert))
    num_per_wave = min(num_per_wave, num_experts_per_rank)
    # Round up to a divisor of num_experts_per_rank.
    while num_per_wave < num_experts_per_rank and num_experts_per_rank % num_per_wave != 0:
        num_per_wave += 1
    return num_per_wave


def get_mega_moe_config(
    *,
    num_ranks: int,
    num_experts: int,
    num_experts_per_rank: int,
    num_max_tokens_per_rank: int,
    num_tokens: int,
    num_topk: int,
    hidden: int,
    intermediate_hidden: int,
    num_padded_sf_pool_tokens: int,
    num_max_pool_tokens: int,
    num_cus: int = 304,
) -> MegaMoEConfig:
    """Return the mega-MoE config selected for the given shape."""

    expected_tokens_per_expert = num_tokens * num_ranks * num_topk / max(num_experts, 1)
    block_m, store_block_m = _pick_block_m(expected_tokens_per_expert)
    block_n = 128
    block_k = 128
    load_block_m = max(1, block_m // 2)
    load_block_n = block_n

    num_per_wave = _pick_num_experts_per_wave(
        num_experts_per_rank=num_experts_per_rank,
        num_tokens=num_tokens,
        num_topk=num_topk,
        intermediate_hidden=intermediate_hidden,
        block_m=block_m,
        block_n=block_n,
        num_cus=num_cus,
    )

    return MegaMoEConfig(
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
        load_block_m=load_block_m,
        load_block_n=load_block_n,
        store_block_m=store_block_m,
        sf_block_m=block_m,
        sf_block_n=block_n,
        num_max_pool_tokens=num_max_pool_tokens,
        num_padded_sf_pool_tokens=num_padded_sf_pool_tokens,
        # FP8 activations + FP4 weights (unpacked to 8-bit in LDS) both
        # use 128-byte swizzle.  Matches DG's SM100 reference.
        swizzle_acts_mode=128,
        swizzle_weights_mode=128,
        num_experts_per_wave=num_per_wave,
        num_stages=4,
        smem_size=0,
        num_dispatch_threads=128,
        num_non_epilogue_threads=128,
        num_epilogue_threads=256,
    )
