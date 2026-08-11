###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Heuristic for the CK WS kernel's ``ws_local_per_xcd`` parameter.

The CK WS kernel takes a single ``ws_local_per_xcd`` integer:

    local_per_xcd = 0                            -> global-only
    local_per_xcd = ceil(total_tiles/num_xcds)   -> per-xcd-only
    in-between                                   -> hierarchical

The high-level API exposes a string ``ws_mode`` (``"auto" | "per-xcd" |
"global" | "hierarchical"``); this module resolves it to the integer the
kernel wants. ``"auto"`` applies the empirical-best policy derived from
the 288-shape Primus-Turbo MoE bench:

    standard kernel (fwd / dgrad):  tpc < 4 -> per-xcd, else global
    variable-K kernel (wgrad):      tpc < 2 -> per-xcd, else global

(``tpc`` = total_tiles / num_cus, the "tiles per CU" workload density.)

This module must stay in sync with ``NUM_XCDS_WS`` in
``csrc/kernels/grouped_gemm/grouped_gemm_kernel_ws.hpp``.
"""

from __future__ import annotations

CK_NUM_XCDS = 8

_KERNEL_STANDARD = "standard"
_KERNEL_VARIABLE_K = "variable_k"

# Empirical "auto" thresholds: tpc below this -> per-xcd, else global.
_AUTO_TPC_THRESHOLD = {
    _KERNEL_STANDARD: 4.0,
    _KERNEL_VARIABLE_K: 2.0,
}


def resolve_ck_ws_local_per_xcd(
    ws_mode: str,
    total_tiles: int,
    num_cus: int,
    *,
    kernel_kind: str = _KERNEL_STANDARD,
) -> int:
    """Map ``ws_mode`` (string) to ``ws_local_per_xcd`` (int) for the CK WS kernel.

    Args:
        ws_mode: ``"auto"``, ``"per-xcd"``, ``"global"``, or ``"hierarchical"``.
        total_tiles: total persistent-loop tile count (sum across groups).
        num_cus: grid-x dim the kernel will launch with (e.g. 256 on MI355X).
        kernel_kind: ``"standard"`` (fwd / dgrad) or ``"variable_k"`` (wgrad).

    Returns:
        The integer ``ws_local_per_xcd`` value to pass to the kernel.
    """
    if total_tiles <= 0 or num_cus <= 0:
        return 0

    per_xcd_full = (total_tiles + CK_NUM_XCDS - 1) // CK_NUM_XCDS

    if ws_mode == "global":
        return 0
    if ws_mode == "per-xcd":
        return per_xcd_full
    if ws_mode == "hierarchical":
        # ~50% per-xcd, ~50% global tail.
        # Edge case: for total_tiles < CK_NUM_XCDS * 2 (= 16), the integer
        # division floors to 0 and the ``max(1, ...)`` clamp yields
        # local_per_xcd=1, meaning phase 1 produces only one claim per XCD.
        # That is functionally equivalent to per-xcd mode for such tiny
        # shapes, which is fine -- the kernel bounds-checks local_per_xcd
        # against the remaining tile count.
        return max(1, (total_tiles // 2) // CK_NUM_XCDS)
    if ws_mode == "auto":
        threshold = _AUTO_TPC_THRESHOLD.get(kernel_kind, _AUTO_TPC_THRESHOLD[_KERNEL_STANDARD])
        if total_tiles / num_cus < threshold:
            return per_xcd_full
        return 0
    raise ValueError(
        f"unknown ws_mode={ws_mode!r}; expected one of 'auto', 'per-xcd', 'global', 'hierarchical'"
    )


# Tile shape constants for total_tiles computation. These mirror the CK
# kernel's BlockM/BlockN -- keep in sync with the tile partitioner template
# parameters in grouped_gemm_kernel_ws.hpp.
CK_TILE_M = 256
CK_TILE_N = 256


def approximate_ck_standard_total_tiles(total_m: int, num_groups: int, n: int) -> int:
    """Sync-free LOWER bound on the standard CK kernel's total_tiles.

    Exact formula (would require reading per-group sizes from the GPU):

        exact      = sum_g ceil(M_g / TILE_M) * num_n_tiles

    This approximation uses only ``total_m = sum(group_lens) = a.size(0)``
    -- tensor metadata, no GPU sync:

        lower_bnd  = ceil(total_m / TILE_M) * num_n_tiles
        upper_bnd  = lower_bnd + (num_groups - 1) * num_n_tiles

    The returned value is the LOWER bound. It is exact when group sizes
    are uniform and underestimates by at most
    ``(num_groups - 1) * num_n_tiles`` for non-uniform groups.

    Why the lower bound (not midpoint or upper)? The resolved
    ``ws_local_per_xcd`` drives the kernel's phase-1 budget.
    Overestimating ``total_tiles`` -> per-xcd budget too large -> CTAs
    burn atomic claims past the real total (correctness-safe via the
    kernel's bounds check, but every wasted claim costs ~1 us of
    serialized atomic time). Underestimating -> phase 1 covers fewer
    tiles than exist -> phase 2 picks up the remainder via the global
    counter (no wasted atomics, just a small phase-balance shift).
    Bias toward "do less in phase 1" is cheap; bias toward "do more"
    wastes work.

    ``num_groups`` is accepted for documentation (it appears in the
    upper-bound formula above) but unused in the returned value.
    """
    del num_groups  # accepted for documentation only
    num_n_tiles = (n + CK_TILE_N - 1) // CK_TILE_N
    return ((total_m + CK_TILE_M - 1) // CK_TILE_M) * num_n_tiles


def compute_ck_variable_k_total_tiles(num_groups: int, k: int, n: int) -> int:
    """Total tiles for the variable-K CK kernel (wgrad).

    Output is [G, K, N] partitioned into [TILE_M, TILE_N] blocks (K plays
    the M role and N plays the N role). Variable-K's tile count depends
    only on integer shapes (not per-group values), so this is exact and
    sync-free.
    """
    num_m_tiles = (k + CK_TILE_M - 1) // CK_TILE_M
    num_n_tiles = (n + CK_TILE_N - 1) // CK_TILE_N
    return num_groups * num_m_tiles * num_n_tiles
