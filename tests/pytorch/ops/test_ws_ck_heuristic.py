###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Unit tests for primus_turbo.pytorch.kernels.grouped_gemm.ws_ck_heuristic.

Pure Python -- no GPU required. Pins the policy boundaries so future
re-tuning is intentional rather than accidental.
"""

import pytest

from primus_turbo.pytorch.kernels.grouped_gemm.ws_ck_heuristic import (
    CK_NUM_XCDS,
    CK_TILE_M,
    CK_TILE_N,
    approximate_ck_standard_total_tiles,
    compute_ck_variable_k_total_tiles,
    resolve_ck_ws_local_per_xcd,
)

# ---------------------------------------------------------------------------
# resolve_ck_ws_local_per_xcd
# ---------------------------------------------------------------------------


def test_degenerate_inputs_return_zero():
    """Non-positive total_tiles or num_cus short-circuits to global mode."""
    assert resolve_ck_ws_local_per_xcd("auto", 0, 256) == 0
    assert resolve_ck_ws_local_per_xcd("auto", -1, 256) == 0
    assert resolve_ck_ws_local_per_xcd("auto", 512, 0) == 0
    assert resolve_ck_ws_local_per_xcd("per-xcd", 0, 256) == 0


def test_explicit_global_returns_zero():
    """Explicit 'global' always returns 0 regardless of shape."""
    for total_tiles in (1, 16, 512, 4096):
        for num_cus in (32, 256):
            assert resolve_ck_ws_local_per_xcd("global", total_tiles, num_cus) == 0


def test_explicit_per_xcd_returns_ceil_divided_by_xcds():
    """Explicit 'per-xcd' returns ceil(total_tiles / CK_NUM_XCDS)."""
    # 512 / 8 = 64 exactly
    assert resolve_ck_ws_local_per_xcd("per-xcd", 512, 256) == 64
    # 513 / 8 = 64.125 -> ceil = 65
    assert resolve_ck_ws_local_per_xcd("per-xcd", 513, 256) == 65
    # 1 / 8 -> ceil = 1
    assert resolve_ck_ws_local_per_xcd("per-xcd", 1, 256) == 1


def test_explicit_hierarchical_clamps_to_at_least_one():
    """'hierarchical' should never return 0 for positive total_tiles."""
    # tiny total -> (1 // 2) // 8 = 0 -> clamped to 1
    assert resolve_ck_ws_local_per_xcd("hierarchical", 1, 256) == 1
    # 800 // 2 = 400; 400 // 8 = 50
    assert resolve_ck_ws_local_per_xcd("hierarchical", 800, 256) == 50


def test_unknown_ws_mode_raises():
    """An unrecognized ws_mode string is a programming error."""
    with pytest.raises(ValueError, match="unknown ws_mode"):
        resolve_ck_ws_local_per_xcd("bogus", 512, 256)
    with pytest.raises(ValueError, match="unknown ws_mode"):
        resolve_ck_ws_local_per_xcd("Global", 512, 256)  # case-sensitive


# ---------------------------------------------------------------------------
# 'auto' heuristic behavior (standard kernel: fwd / dgrad)
# ---------------------------------------------------------------------------


def test_auto_standard_sparse_picks_per_xcd():
    """tpc < 4 on the standard kernel -> per-xcd."""
    # tpc = 512 / 256 = 2.0 (< 4) -> per-xcd
    assert resolve_ck_ws_local_per_xcd("auto", 512, 256, kernel_kind="standard") == 64
    # tpc = 3 (just under threshold)
    assert resolve_ck_ws_local_per_xcd("auto", 768, 256, kernel_kind="standard") == 96


def test_auto_standard_dense_picks_global():
    """tpc >= 4 on the standard kernel -> global (returns 0)."""
    # tpc = 4 exactly -> global
    assert resolve_ck_ws_local_per_xcd("auto", 1024, 256, kernel_kind="standard") == 0
    # tpc = 8 -> global
    assert resolve_ck_ws_local_per_xcd("auto", 2048, 256, kernel_kind="standard") == 0


def test_auto_unknown_kernel_kind_falls_back_to_standard_threshold():
    """Unknown kernel_kind uses the standard-kernel threshold (safe fallback)."""
    # 512 / 256 = 2 < 4 -> per-xcd (standard behavior)
    assert resolve_ck_ws_local_per_xcd("auto", 512, 256, kernel_kind="not-a-kernel-kind") == 64


# ---------------------------------------------------------------------------
# 'auto' heuristic behavior (variable-K kernel: wgrad)
# ---------------------------------------------------------------------------


def test_auto_variable_k_sparse_picks_per_xcd():
    """tpc < 2 on the variable-K kernel -> per-xcd."""
    # tpc = 256 / 256 = 1.0 (< 2) -> per-xcd
    assert resolve_ck_ws_local_per_xcd("auto", 256, 256, kernel_kind="variable_k") == 32


def test_auto_variable_k_dense_picks_global():
    """tpc >= 2 on the variable-K kernel -> global (returns 0)."""
    # tpc = 2 exactly -> global
    assert resolve_ck_ws_local_per_xcd("auto", 512, 256, kernel_kind="variable_k") == 0
    # tpc = 4 -> global
    assert resolve_ck_ws_local_per_xcd("auto", 1024, 256, kernel_kind="variable_k") == 0


# ---------------------------------------------------------------------------
# total_tiles helpers
# ---------------------------------------------------------------------------


def test_approximate_total_tiles_uniform():
    """Uniform group sizes -> lower bound is exact."""
    # total_m = 8192, TILE_M = 256 -> 32 m-tiles
    # n = 2048, TILE_N = 256 -> 8 n-tiles
    # total = 32 * 8 = 256
    assert approximate_ck_standard_total_tiles(8192, num_groups=8, n=2048) == 256


def test_approximate_total_tiles_ceil_division():
    """Non-multiples round up (ceil-div)."""
    # total_m = 300, TILE_M = 256 -> 2 m-tiles; n = 200, TILE_N = 256 -> 1 n-tile
    assert approximate_ck_standard_total_tiles(300, 1, 200) == 2


def test_compute_variable_k_total_tiles():
    """Variable-K total = G * ceil(K/TILE_M) * ceil(N/TILE_N) (exact)."""
    # G=8, K=2048, N=2048, TILE_M=TILE_N=256 -> 8 * 8 * 8 = 512
    assert compute_ck_variable_k_total_tiles(8, 2048, 2048) == 512
    # G=4, K=300, N=200, TILE_M=256, TILE_N=256 -> 4 * ceil(300/256) * ceil(200/256)
    #                                            = 4 * 2 * 1 = 8
    assert compute_ck_variable_k_total_tiles(4, 300, 200) == 8


def test_num_xcds_constant_matches_mi355x():
    """Sanity-check the chiplet count exposed by the module."""
    assert CK_NUM_XCDS == 8


def test_tile_shape_constants_are_256():
    """CK tile shape constants mirror the C++ TilePartitioner template
    parameters -- if the C++ side changes, these must be updated in tandem."""
    assert CK_TILE_M == 256
    assert CK_TILE_N == 256
