###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Index-decode tests for the FlyDSL dense-TN 4-wave split-K window.

The kernel turns a dispatch id into a (slice, tile) pair with a fixed-point reciprocal
divide, so the decode is a bijection only while that reciprocal is exact over the whole
id range. These run on the host: they check the constants the kernel emits, not a GEMM.
"""

import pytest

from primus_turbo.flydsl.gemm.gemm_fp8_kernel import (
    _TN4_BLOCK_K,
    _TN4_RECT,
    _TN4_SPLIT_S,
    _TN4_SQUARE,
    _dense_tn_split,
    _exact_recip,
    _tn4_phases,
)

# CU counts of the parts this kernel is built for; the planner keys its window off them.
CU_COUNTS = (256, 304)
MAX_S = max(_TN4_SPLIT_S)


def _recip(d, xmax):
    """The (shift, mul) pair ``_dense_tn_slice_div`` emits, or None for its shift path."""
    return None if d & (d - 1) == 0 else _exact_recip(d, xmax)


def _decode(win_off, nwin, s):
    """Replica of the kernel's dispatch-id decode, using the constants it emits."""
    rcp = _recip(nwin, nwin * s - 1)
    sid = win_off // nwin if rcp is None else (win_off * rcp[1]) >> rcp[0]
    return sid, win_off - sid * nwin


def test_exact_recip_rounds_down_over_its_declared_range():
    """The reciprocal must match ``//`` for every dividend up to the bound it was given."""
    for d in range(1, 1025):
        xmax = MAX_S * d  # the decode's widest dividend is nwin * s - 1
        shift, mul = _exact_recip(d, xmax)
        assert xmax * mul < (1 << 31), f"the emitted i32 multiply would overflow at d={d}"
        for x in range(xmax + 1):
            assert (x * mul) >> shift == x // d, f"d={d} x={x} shift={shift} mul={mul}"


@pytest.mark.parametrize("s", _TN4_SPLIT_S)
def test_split_decode_is_a_bijection(s):
    """Every id in the window must land on its own (slice, tile), and cover all of them."""
    for nwin in range(1, 513):
        seen = set()
        for win_off in range(nwin * s):
            sid, tile = _decode(win_off, nwin, s)
            assert 0 <= sid < s, f"slice {sid} out of range for nwin={nwin} s={s}"
            assert 0 <= tile < nwin, f"tile {tile} out of range for nwin={nwin} s={s}"
            seen.add((sid, tile))
        assert len(seen) == nwin * s, f"decode collided for nwin={nwin} s={s}"


@pytest.mark.parametrize("ncu", CU_COUNTS)
@pytest.mark.parametrize("geom", [_TN4_SQUARE, _TN4_RECT], ids=["square", "rect"])
def test_planner_windows_decode_cleanly(geom, ncu):
    """Sweep the tile counts the planner can see and check every window it settles on."""
    phases = _tn4_phases(geom)
    windows = {_dense_tn_split(tiles, 4096 // _TN4_BLOCK_K, ncu, phases) for tiles in range(1, 8 * ncu)}
    windows.discard(None)
    assert windows, "the planner produced no split-K window to check"
    for _, nwin, s in sorted(windows):
        seen = {_decode(off, nwin, s) for off in range(nwin * s)}
        assert len(seen) == nwin * s, f"decode collided on planner window nwin={nwin} s={s}"


def test_gpt_oss_20b_attn_out_wgrad_window():
    """Regression lock on the shape that first showed the defect.

    gpt-oss-20B attn_out wgrad at mbs 1 is C[2880, 4096], which is 12x16 = 192 of the 256x256
    macro tile, so the planner splits all 192 tiles 4 ways. A 16-bit reciprocal reads sid = 3
    from win_off >= 512, which drops tile 191 of slices 2 and 3 and leaves the fold reading a
    workspace band nothing wrote.
    """
    nwin, s = 192, 4
    tiles = -(-2880 // _TN4_SQUARE.bm) * -(-4096 // _TN4_SQUARE.bn)
    assert tiles == nwin
    k_iters = 4096 // _TN4_BLOCK_K
    assert _dense_tn_split(tiles, k_iters, 256, _tn4_phases(_TN4_SQUARE)) == (0, nwin, s)
    assert all(_decode(off, nwin, s) == (off // nwin, off % nwin) for off in range(nwin * s))
    # The pre-fix constants, spelled out so the failure mode stays legible.
    stale = [off for off in range(nwin * s) if (off * (-(-(1 << 16) // nwin))) >> 16 != off // nwin]
    assert stale == [575, 767], "a 16-bit reciprocal should misdecode exactly these two ids"
