###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Backend dispatch + symmetric buffer layout helpers for Mega MoE.

This module is the Python-side counterpart of DeepGEMM's
``csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp``.  It owns:

* the symmetric-buffer layout calculation (offsets / total bytes),
* the weight pre-processing helpers (interleave gate/up, SF transpose),
* the actual kernel launcher dispatch (currently a no-op shell that
  raises ``NotImplementedError`` when invoked on shapes that the
  device kernel does not yet support).

Function names mirror DeepGEMM's ``deep_gemm.mega`` API surface 1:1 so
existing call sites in DeepSeek-style MoE pipelines can switch
backends with minimal diff.  The kernel-side implementation is
intentionally a stub.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch

__all__ = [
    "SymmBufferLayout",
    "fp8_fp4_mega_moe_impl",
    "get_symm_buffer_size_for_mega_moe",
    "get_token_alignment_for_mega_moe",
    "transform_l1_weights_for_mega_moe",
    "transform_l2_weights_for_mega_moe",
]


# ---------------------------------------------------------------------------
# Internal helpers — C extension thin wrappers
# ---------------------------------------------------------------------------


def _cpp_extension():
    return torch.ops.primus_turbo_cpp_extension


def get_token_alignment_for_mega_moe() -> int:
    """Return the per-rank token count alignment required by the layout.

    Mirrors DeepGEMM's ``deep_gemm._C.get_token_alignment_for_mega_moe``.
    """

    return int(_cpp_extension().get_token_alignment_for_mega_moe())


# ---------------------------------------------------------------------------
# Symmetric buffer layout
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SymmBufferLayout:
    """Per-region offsets (in bytes) into the symmetric memory buffer.

    The layout is computed once by the C++ helper and mirrored here as
    plain Python integers so the rest of the frontend can slice it
    cheaply without re-entering the C extension.
    """

    total_bytes: int
    num_max_pool_tokens: int
    num_padded_sf_pool_tokens: int

    workspace_offset: int
    input_x_offset: int
    input_x_sf_offset: int
    input_topk_idx_offset: int
    input_topk_weights_offset: int
    l1_pool_x_offset: int
    l1_pool_x_sf_offset: int
    l1_pool_weights_offset: int
    l2_pool_x_offset: int
    l2_pool_x_sf_offset: int
    combine_buffer_offset: int


def get_symm_buffer_size_for_mega_moe(
    *,
    num_ranks: int,
    num_experts: int,
    num_max_tokens_per_rank: int,
    num_topk: int,
    hidden: int,
    intermediate_hidden: int,
    use_fp8_dispatch: bool = True,
) -> SymmBufferLayout:
    """Compute the symmetric memory layout for one mega-MoE allocation.

    Mirrors DeepGEMM's ``deep_gemm._C.get_symm_buffer_size_for_mega_moe``.
    DG returns ``(num_bytes, slice_callback)``; here we return the
    full ``SymmBufferLayout`` so the Python ``SymmBuffer`` wrapper can
    do the slicing itself (the C++ binding lives behind
    ``torch.library``).
    """

    raw = _cpp_extension().get_symm_buffer_size_for_mega_moe(
        int(num_ranks),
        int(num_experts),
        int(num_max_tokens_per_rank),
        int(num_topk),
        int(hidden),
        int(intermediate_hidden),
        bool(use_fp8_dispatch),
    )
    values = raw.tolist()
    return SymmBufferLayout(
        total_bytes=values[0],
        num_max_pool_tokens=values[1],
        num_padded_sf_pool_tokens=values[2],
        workspace_offset=values[3],
        input_x_offset=values[4],
        input_x_sf_offset=values[5],
        input_topk_idx_offset=values[6],
        input_topk_weights_offset=values[7],
        l1_pool_x_offset=values[8],
        l1_pool_x_sf_offset=values[9],
        l1_pool_weights_offset=values[10],
        l2_pool_x_offset=values[11],
        l2_pool_x_sf_offset=values[12],
        combine_buffer_offset=values[13],
    )


# ---------------------------------------------------------------------------
# Weight pre-processing
# ---------------------------------------------------------------------------


def _interleave_gate_up(
    weights: torch.Tensor,
    *,
    granularity: int = 8,
) -> torch.Tensor:
    """Interleave (gate | up) along the N axis into (gate, up) blocks.

    Matches the layout DeepGEMM uses for SwiGLU fusion:
    ``[gate: 0..g-1, up: 0..g-1, gate: g..2g-1, up: g..2g-1, ...]``.
    """

    if weights.dim() < 2:
        raise ValueError(f"_interleave_gate_up: expected >=2D tensor, got {weights.shape}")
    n = weights.size(1)
    if n % 2 != 0:
        raise ValueError(f"_interleave_gate_up: N must be even, got {n}")

    half = n // 2
    g, _, *rest = weights.shape
    gate = weights[:, :half].reshape(g, half // granularity, granularity, *rest)
    up = weights[:, half:].reshape(g, half // granularity, granularity, *rest)
    interleaved = torch.stack([gate, up], dim=2).reshape(g, n, *rest)
    return torch.empty_like(weights).copy_(interleaved)


def _transpose_sf_for_mfma(sf: torch.Tensor) -> torch.Tensor:
    """Permute SFs so MFMA / UTCCP-style loads see a contiguous layout."""

    if sf.dim() != 3:
        raise ValueError(f"_transpose_sf_for_mfma: expected 3D SF tensor, got {sf.shape}")
    num_groups, mn, packed_sf_k = sf.shape
    if mn % 128 != 0:
        raise ValueError(f"_transpose_sf_for_mfma: SF mn dim must be %128 == 0, got {mn}")

    permuted = (
        sf.reshape(num_groups, -1, 4, 32, packed_sf_k).transpose(2, 3).reshape(num_groups, mn, packed_sf_k)
    )
    return torch.empty_like(sf).copy_(permuted)


def transform_l1_weights_for_mega_moe(
    l1_weights: Tuple[torch.Tensor, torch.Tensor]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pre-process L1 (up-projection) weights into the kernel layout."""

    w, sf = l1_weights
    w = _interleave_gate_up(w)
    sf = _transpose_sf_for_mfma(sf)
    return w, sf


def transform_l2_weights_for_mega_moe(
    l2_weights: Tuple[torch.Tensor, torch.Tensor]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pre-process L2 (down-projection) weights into the kernel layout."""

    w, sf = l2_weights
    sf = _transpose_sf_for_mfma(sf)
    return w, sf


# ---------------------------------------------------------------------------
# Kernel launcher
# ---------------------------------------------------------------------------


def fp8_fp4_mega_moe_impl(
    y: torch.Tensor,
    l1_weights: Tuple[torch.Tensor, torch.Tensor],
    l2_weights: Tuple[torch.Tensor, torch.Tensor],
    *,
    sym_buffer: torch.Tensor,
    sym_buffer_ptrs: list[int],
    rank_idx: int,
    num_max_tokens_per_rank: int,
    num_experts: int,
    num_topk: int,
    num_tokens: int,
    hidden: int,
    intermediate_hidden: int,
    cumulative_local_expert_recv_stats: Optional[torch.Tensor] = None,
    recipe: Tuple[int, int, int] = (1, 1, 32),
    activation: str = "swiglu",
    activation_clamp: Optional[float] = None,
    fast_math: bool = True,
) -> None:
    """Forward to the C++ ``fp8_fp4_mega_moe`` launcher.

    The kernel itself is currently a no-op stub: it validates arguments
    and returns without writing to ``y``.  Once the device kernel lands
    this function becomes the single entry point used by the autograd
    op and module wrapper.
    """

    if activation != "swiglu":
        raise NotImplementedError(
            f"fp8_fp4_mega_moe currently supports activation='swiglu', got {activation!r}"
        )
    if len(recipe) != 3:
        raise ValueError(f"fp8_fp4_mega_moe: recipe must be a length-3 tuple, got {recipe}")
    clamp = float("inf") if activation_clamp is None else float(activation_clamp)
    if clamp < 0:
        raise ValueError(f"activation_clamp must be non-negative, got {activation_clamp}")

    l1_w, l1_sf = l1_weights
    l2_w, l2_sf = l2_weights

    _cpp_extension().fp8_fp4_mega_moe(
        y,
        l1_w,
        l1_sf,
        l2_w,
        l2_sf,
        cumulative_local_expert_recv_stats,
        sym_buffer,
        list(sym_buffer_ptrs),
        int(rank_idx),
        int(num_max_tokens_per_rank),
        int(num_experts),
        int(num_topk),
        int(num_tokens),
        int(hidden),
        int(intermediate_hidden),
        [int(recipe[0]), int(recipe[1]), int(recipe[2])],
        str(activation),
        clamp,
        bool(fast_math),
    )
