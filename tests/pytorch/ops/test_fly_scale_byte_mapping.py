###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""The shift/mask fast path of ``fly_scale_byte`` (csrc/include/primus_turbo/mxfp4_emit.hpp) equals the generic
mapping for the 256-wide N tile (nt 4) and interleave 0 / 4. Python ports of both, checked over every kblk, is_b,
ilv, a dense row range and rows near 16384 (the mapping is periodic in row beyond the group / block structure)."""

import pytest


def _generic(row, kblk, is_b, nt, ilv, k128):
    ku = 2 if (k128 // 2) % 2 == 0 else 1
    nw, kk = 2 * ku, k128 // 2
    kdw, g = kblk // 4, kblk % 4
    kh, rem = kdw // nw, kdw % nw
    u, lo = rem // 2, rem % 2
    gspan = 16 * nt
    grp, loc = row // gspan, row % gspan
    if is_b:
        blk, off = grp // 4, grp % 4
        r_region, wi = off // 2, blk * 2 + off % 2
    else:
        wi, r_region = grp // 2, grp % 2
    if ilv:
        r, t = loc // ilv, loc % ilv
    else:
        t, r = loc // 16, loc % 16
    last = r_region * 2 + lo
    base = ((wi * kk + kh * ku) * 64 + r) * 4
    return (base + u * 256 + g * 64 + last) * 4 + t


def _fast(row, kblk, is_b, ilv, k128):
    kk = k128 >> 1
    ku_shift = 1 if (kk & 1) == 0 else 0
    nw_shift = ku_shift + 1
    kdw, g = kblk >> 2, kblk & 3
    kh, rem = kdw >> nw_shift, kdw & ((1 << nw_shift) - 1)
    u, lo = rem >> 1, rem & 1
    grp, loc = row >> 6, row & 63
    if is_b:
        r_region, wi = (grp & 3) >> 1, (grp >> 2) * 2 + (grp & 1)
    else:
        wi, r_region = grp >> 1, grp & 1
    r = (loc >> 2) if ilv else (loc & 15)
    t = (loc & 3) if ilv else (loc >> 4)
    last = r_region * 2 + lo
    base = ((wi * kk + (kh << ku_shift)) * 64 + r) * 4
    return (base + u * 256 + g * 64 + last) * 4 + t


@pytest.mark.parametrize("k", [512, 768, 1024, 3072, 8192, 9216, 12288, 16384])
def test_fly_scale_byte_fast_path_matches_generic(k):
    k128 = k // 128
    rows = list(range(1024)) + list(range(16384 - 64, 16384))
    for is_b in (0, 1):
        for ilv in (0, 4):
            for row in rows:
                for kblk in range(k128 * 4):
                    assert _generic(row, kblk, is_b, 4, ilv, k128) == _fast(row, kblk, is_b, ilv, k128), (
                        k,
                        is_b,
                        ilv,
                        row,
                        kblk,
                    )
