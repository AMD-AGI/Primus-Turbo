###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Heuristic for the Triton WS kernel's ``ws_local_per_xcd`` parameter.

The Triton WS kernel takes a single ``ws_local_per_xcd`` integer that
selects the per-XCD phase-1 tile budget:

    local_per_xcd = 0                            -> global-only
    local_per_xcd = ceil(total_tiles/num_xcds)   -> per-xcd-only
    in-between                                   -> hierarchical

The high-level API exposes a string ``ws_mode`` (``"auto" | "per-xcd" |
"global" | "hierarchical"``); this module resolves it to the integer the
kernel wants. ``"auto"`` applies the empirical-best policy derived from
the 288-shape Primus-Turbo MoE bench:

    standard kernel (fwd / dgrad):
        tpc < 16  -> hierarchical adaptive split
        tpc >= 16  -> global

    variable-K kernel (wgrad):
        tpc < 64  -> per-xcd
        tpc >= 64  -> global

(``tpc`` = total_tiles / num_sms.)
"""

from __future__ import annotations

# MI355X / MI350 chiplet count.
NUM_XCDS = 8

_KERNEL_STANDARD = "standard"
_KERNEL_VARIABLE_K = "variable_k"


def resolve_triton_ws_local_per_xcd(
    ws_mode: str,
    total_tiles: int,
    num_sms: int,
    *,
    num_xcds: int = NUM_XCDS,
    kernel_kind: str = _KERNEL_STANDARD,
) -> int:
    """Map ``ws_mode`` (string) to ``ws_local_per_xcd`` (int) for the Triton WS kernel.

    Args:
        ws_mode: ``"auto"``, ``"per-xcd"``, ``"global"``, or ``"hierarchical"``.
        total_tiles: total persistent-loop tile count (sum across groups).
        num_sms: persistent grid size (typically the compute-unit count).
        num_xcds: chiplet count (8 on MI355X / MI350).
        kernel_kind: ``"standard"`` (fwd / dgrad) or ``"variable_k"`` (wgrad).

    Returns:
        The integer ``local_per_xcd`` value to pass to the kernel.
    """
    if total_tiles <= 0 or num_sms <= 0:
        return 0

    per_xcd_full = (total_tiles + num_xcds - 1) // num_xcds

    if ws_mode == "global":
        return 0
    if ws_mode == "per-xcd":
        return per_xcd_full

    tpc = total_tiles / max(num_sms, 1)

    if ws_mode == "auto":
        if kernel_kind == _KERNEL_VARIABLE_K:
            # wgrad: per-xcd wins on ~56% of shapes; global only takes over on
            # very dense shapes.
            if tpc < 64:
                return per_xcd_full
            return 0
        # standard kernel (fwd / dgrad):
        if tpc >= 16:
            return 0
        # sparse standard-kernel shapes: fall through to hierarchical adaptive split

    if ws_mode in ("hierarchical", "auto"):
        # Adaptive split: at very sparse (tpc < 4) local_frac = 1 -> all
        # phase 1; ramp down as tpc grows; floor at 0.5.
        local_frac = max(0.5, 1.0 - max(0.0, tpc - 4.0) * 0.05)
        return max(1, int(total_tiles * local_frac) // num_xcds)

    raise ValueError(
        f"unknown ws_mode={ws_mode!r}; expected one of 'auto', 'per-xcd', 'global', 'hierarchical'"
    )


# Tile shape constants for total_tiles computation. Default Triton block
# sizes mirror grouped_gemm_kernel_ws.py -- keep in sync.
TRITON_BLOCK_M = 128
TRITON_BLOCK_N = 128


def approximate_triton_total_tiles(
    total_m: int, num_groups: int, n: int, *, block_m: int = TRITON_BLOCK_M, block_n: int = TRITON_BLOCK_N
) -> int:
    """Sync-free LOWER bound on the standard Triton kernel's total_tiles.

    Exact formula (would require reading per-group sizes from the GPU):

        exact     = sum_g ceil(M_g / block_m) * num_pid_n

    This approximation uses only ``total_m = sum(group_lens) = a.size(0)``
    -- tensor metadata, no GPU sync:

        lower_bnd = ceil(total_m / block_m) * num_pid_n
        upper_bnd = lower_bnd + (G - 1) * num_pid_n

    Lower bound is exact for uniform group sizes and underestimates by
    at most ``(G - 1) * num_pid_n`` for non-uniform groups.

    Why the lower bound (not midpoint or upper): the resolved
    ``ws_local_per_xcd`` drives phase-1 budgeting. Overestimating
    causes CTAs to burn atomic claims past the real ``total_tiles``;
    underestimating shifts work to phase 2 (global counter) with no
    wasted atomics. Bias toward underestimating is cheap.

    ``num_groups`` is accepted for documentation (it appears in the
    upper-bound formula above) but unused in the returned value.
    """
    del num_groups  # accepted for documentation only
    num_pid_n = (n + block_n - 1) // block_n
    return ((total_m + block_m - 1) // block_m) * num_pid_n


def compute_triton_variable_k_total_tiles(
    num_groups: int, k: int, n: int, *, block_m: int = TRITON_BLOCK_M, block_n: int = TRITON_BLOCK_N
) -> int:
    """Total persistent tiles for the variable-K Triton kernel (wgrad).

    Output is [G, K, N] partitioned into [block_m, block_n] blocks. Uses
    only integer shapes -- exact and sync-free.
    """
    return num_groups * ((k + block_m - 1) // block_m) * ((n + block_n - 1) // block_n)
