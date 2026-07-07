###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Unit tests for primus_turbo.triton.grouped_gemm.ws_triton_heuristic.

Pure Python -- no GPU required. Pins the policy boundaries so future
re-tuning is intentional rather than accidental.
"""

import pytest

from primus_turbo.triton.grouped_gemm.ws_triton_heuristic import (
    NUM_XCDS,
    approximate_triton_total_tiles,
    compute_triton_variable_k_total_tiles,
    resolve_triton_ws_local_per_xcd,
)

# ---------------------------------------------------------------------------
# resolve_triton_ws_local_per_xcd
# ---------------------------------------------------------------------------


def test_degenerate_inputs_return_zero():
    """Non-positive total_tiles or num_sms short-circuits to global mode."""
    assert resolve_triton_ws_local_per_xcd("auto", 0, 256) == 0
    assert resolve_triton_ws_local_per_xcd("auto", -1, 256) == 0
    assert resolve_triton_ws_local_per_xcd("auto", 512, 0) == 0
    assert resolve_triton_ws_local_per_xcd("per-xcd", 0, 256) == 0


def test_explicit_global_returns_zero():
    """Explicit 'global' always returns 0 regardless of shape."""
    for total_tiles in (1, 16, 512, 4096):
        for num_sms in (32, 256):
            assert resolve_triton_ws_local_per_xcd("global", total_tiles, num_sms) == 0


def test_explicit_per_xcd_returns_ceil_divided_by_xcds():
    """Explicit 'per-xcd' returns ceil(total_tiles / num_xcds)."""
    # 512 / 8 = 64 exactly
    assert resolve_triton_ws_local_per_xcd("per-xcd", 512, 256) == 64
    # 513 / 8 = 64.125 -> ceil = 65
    assert resolve_triton_ws_local_per_xcd("per-xcd", 513, 256) == 65
    # 1 / 8 -> ceil = 1
    assert resolve_triton_ws_local_per_xcd("per-xcd", 1, 256) == 1


def test_explicit_hierarchical_clamps_to_at_least_one():
    """'hierarchical' should never return 0 for positive total_tiles."""
    # tiny total -> (1 // 2) // 8 = 0 -> clamped to 1
    assert resolve_triton_ws_local_per_xcd("hierarchical", 1, 256) == 1
    # tpc = 800/256 ~= 3.125 -> local_frac = 1.0 (under the 4-tpc shoulder)
    # local_per_xcd = max(1, int(800 * 1.0) // 8) = 100
    assert resolve_triton_ws_local_per_xcd("hierarchical", 800, 256) == 100
    # tpc = 50 (16000/256 - large), local_frac clamped to 0.5
    # local_per_xcd = max(1, int(16000 * 0.5) // 8) = 1000
    assert resolve_triton_ws_local_per_xcd("hierarchical", 16000, 256) == 1000


def test_unknown_ws_mode_raises():
    """An unrecognized ws_mode string is a programming error."""
    with pytest.raises(ValueError, match="unknown ws_mode"):
        resolve_triton_ws_local_per_xcd("bogus", 512, 256)
    with pytest.raises(ValueError, match="unknown ws_mode"):
        resolve_triton_ws_local_per_xcd("Global", 512, 256)  # case-sensitive


# ---------------------------------------------------------------------------
# 'auto' heuristic behavior (standard kernel: fwd / dgrad)
# ---------------------------------------------------------------------------


def test_auto_standard_dense_picks_global():
    """tpc >= 16 on the standard kernel -> global (returns 0)."""
    # tpc = 4096 / 256 = 16 exactly -> global
    assert resolve_triton_ws_local_per_xcd("auto", 4096, 256, kernel_kind="standard") == 0
    # tpc = 32 -> definitely global
    assert resolve_triton_ws_local_per_xcd("auto", 8192, 256, kernel_kind="standard") == 0


def test_auto_standard_sparse_picks_adaptive():
    """tpc < 16 on the standard kernel -> adaptive hierarchical split."""
    # tpc = 800 / 256 ~= 3.125 (below the 4-tpc shoulder -> local_frac = 1.0)
    result = resolve_triton_ws_local_per_xcd("auto", 800, 256, kernel_kind="standard")
    # local_per_xcd = max(1, (800 * 1.0) // 8) = 100
    assert result == 100
    # tpc = 12 (= 3072 / 256), in the ramp region
    # local_frac = max(0.5, 1.0 - (12-4)*0.05) = max(0.5, 0.6) = 0.6
    # local_per_xcd = max(1, int(3072 * 0.6) // 8) = max(1, 1843 // 8) = 230
    result = resolve_triton_ws_local_per_xcd("auto", 3072, 256, kernel_kind="standard")
    assert result == 230


def test_auto_standard_floor_at_half():
    """local_frac is floored at 0.5 -- never below."""
    # tpc = 50 -> would be 1.0 - 46*0.05 = -1.3, but clamp to 0.5
    # (this case would have been picked off by the tpc>=16 short-circuit,
    # but exercising the formula directly via hierarchical mode:)
    result = resolve_triton_ws_local_per_xcd("hierarchical", 16000, 256)
    # local_per_xcd = (16000 // 2) // 8 = 1000 (hierarchical helper is the
    # same formula but always with local_frac = 0.5)
    assert result == 1000


# ---------------------------------------------------------------------------
# 'auto' heuristic behavior (variable-K kernel: wgrad)
# ---------------------------------------------------------------------------


def test_auto_variable_k_sparse_picks_per_xcd():
    """tpc < 64 on the variable-K kernel -> per-xcd."""
    # tpc = 1024 / 256 = 4 (< 64) -> per-xcd
    assert resolve_triton_ws_local_per_xcd("auto", 1024, 256, kernel_kind="variable_k") == 128
    # tpc = 63 (just under threshold)
    assert resolve_triton_ws_local_per_xcd("auto", 16128, 256, kernel_kind="variable_k") == 2016


def test_auto_variable_k_dense_picks_global():
    """tpc >= 64 on the variable-K kernel -> global (returns 0)."""
    # tpc = 64 exactly -> global
    assert resolve_triton_ws_local_per_xcd("auto", 16384, 256, kernel_kind="variable_k") == 0
    # tpc = 128
    assert resolve_triton_ws_local_per_xcd("auto", 32768, 256, kernel_kind="variable_k") == 0


# ---------------------------------------------------------------------------
# total_tiles helpers
# ---------------------------------------------------------------------------


def test_approximate_total_tiles_uniform():
    """Uniform group sizes -> lower bound is exact."""
    # total_m = 16384, block_m = 128 -> 128 m-tiles
    # n = 4096, block_n = 128 -> 32 n-tiles
    # total = 128 * 32 = 4096
    assert approximate_triton_total_tiles(16384, num_groups=8, n=4096) == 4096


def test_approximate_total_tiles_respects_block_size():
    """block_m / block_n kwargs override the defaults."""
    # total_m = 1024, block_m = 256 -> 4 m-tiles; n = 512, block_n = 256 -> 2 n-tiles
    assert approximate_triton_total_tiles(1024, 1, 512, block_m=256, block_n=256) == 8


def test_compute_variable_k_total_tiles():
    """Variable-K total = G * ceil(K/bm) * ceil(N/bn) (exact)."""
    # G=8, K=2048, N=2048, block_m=block_n=128 -> 8 * 16 * 16 = 2048
    assert compute_triton_variable_k_total_tiles(8, 2048, 2048) == 2048
    # G=4, K=300, N=200, block_m=128, block_n=128 -> 4 * ceil(300/128) * ceil(200/128)
    #                                              = 4 * 3 * 2 = 24
    assert compute_triton_variable_k_total_tiles(4, 300, 200) == 24


def test_num_xcds_constant_matches_mi355x():
    """Sanity-check the chiplet count exposed by the module."""
    assert NUM_XCDS == 8
