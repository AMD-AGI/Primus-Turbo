#!/usr/bin/env python3
"""Print the pipeline chunk plan / batch cut the host would choose for each scored cell."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import primus_turbo.flydsl.attention.flash_attn_bwd as b

CELLS = [
    ("d128_70b_b2", 2, 64, 8, 8192, 128, -1),
    ("d128_8b_b2", 2, 32, 8, 8192, 128, -1),
    ("d64_gptoss_b4", 4, 64, 8, 8192, 64, -1),
    ("d64_b2", 2, 64, 8, 8192, 64, -1),
    ("d64_swa_b4", 4, 64, 8, 8192, 64, 128),
    ("d128_b1", 1, 64, 8, 8192, 128, -1),
]

for arm in sys.argv[1:] or ["base"]:
    if arm != "base":
        import _probe_bwdcfg

        _probe_bwdcfg.patch_all(arm)
    print(f"===== {arm}")
    for tag, B, Hq, Hkv, S, D, W in CELLS:
        bkv = b._fuse_blockkv_for(S, D, W)
        qs = b._qsplit_for(S, W, D)
        wgs = (S // bkv) * Hkv * B
        plan = b._pipe_chunks(B, qs, bkv, S, D, True, wgs=wgs)
        nq = plan[-1][2] or qs
        fb = b._dq_fold_bytes(B, S, Hq, D, S, bkv, qs, nq)
        uniform = len(plan) > 1 and len({c[2] for c in plan}) == 1
        grid = b._dq_grid_cut(wgs, B, nq, bkv, fb) if uniform else 1
        tail = b._dq_tail_cut(wgs, B, nq, bkv) if grid == 1 else 1
        print(
            f"{tag:14s} bkv={bkv:3d} qs={qs} wgs={wgs:5d} plan={plan} "
            f"chunk_fold={fb / (1 << 30):.2f}GiB grid_cut={grid} tail_cut={tail}"
        )
        nbt = B // grid if grid > 1 else None
        cfb = b._dq_fold_bytes(nbt or B, S, Hq, D, S, bkv, qs, plan[0][2])
        sl = b._fold_slices(B, S, qs, plan[0][1], plan[0][2], cfb, bkv, bat_lo=0, n_bat=nbt)
        print(f"    {len(sl)} slices of chunk 0: {[(s[1], s[2]) for s in sl]}")
