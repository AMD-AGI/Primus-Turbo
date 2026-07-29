"""Diagnostic: is the fwd/dgrad NT config race deterministic?

Clears the cfg cache and re-races each campaign shape N times, printing the winner plus the
per-point cand/base ratios the race saw. Deterministic selection needs every candidate to sit
on the same side of the adoption margin on every repeat. Read-only w.r.t. production code.
"""
import statistics
import sys

import torch

import primus_turbo.pytorch  # noqa: F401
import primus_turbo.flydsl.grouped_gemm.mxfp8_grouped_kernel as MK
from primus_turbo.flydsl.utils.gemm_helper import _robust_ab_ratio, _robust_time
from primus_turbo.pytorch.core.low_precision import float8_e4m3

DEV = "cuda"
G = 32
REPEAT = int(sys.argv[1]) if len(sys.argv) > 1 else 3
# (label, K, N) -- the three static NT shapes the 8 fwd/dgrad campaign configs collapse onto
SHAPES = [("fwd/dgrad down", 2944, 2944), ("fwd gate_up", 2944, 5760), ("dgrad gate_up", 5760, 2944)]


def mk_args(K, N, M):
    a8 = torch.randint(0, 127, (M, K), device=DEV, dtype=torch.int8)
    b8 = torch.randint(0, 127, (G * N, K), device=DEV, dtype=torch.int8)
    out = torch.empty((M, N), device=DEV, dtype=torch.bfloat16)
    a_raw = torch.randint(120, 128, (M, K // 32), device=DEV, dtype=torch.uint8).view(torch.int32).reshape(-1)
    b_raw = torch.randint(120, 128, (G * N, K // 32), device=DEV, dtype=torch.uint8).view(torch.int32).reshape(-1)
    stream = torch.cuda.current_stream()
    a_sp, b_sp, a_blocks, a_ngrp = MK._get_grouped_mx_workspace(M, N, K // 128, G, DEV, stream)
    go = (torch.arange(0, G + 1, dtype=torch.int64, device=DEV) * (M // G)).view(torch.int32)
    n_blocks = (N + 256 - 1) // 256
    return (
        a8, b8, out, a_raw, b_raw, a_sp, b_sp, go, go, M,
        a_ngrp * 64, N, a_blocks, a_ngrp, ((M + 255) // 256 + G) * n_blocks, stream,
    )


for label, K, N in SHAPES:
    print(f"\n===== {label}  K={K} N={N} =====", flush=True)
    args = mk_args(K, N, 131072)
    cands = MK._gnt_nt_candidates(N)
    points = [MK._canon_nt_targs(args, K, G, N, pm, skew) for pm, skew in MK._GNT_PM_CANON]
    base = MK._get_nt_launch(K, G, N, *cands[0], 0, 0, False, False)
    _robust_time(base, points[0][0])
    hist, ratios = [], {c: [[] for _ in points] for c in cands[1:]}
    for r in range(REPEAT):
        MK._GNT_CFG_CACHE.clear()
        best, best_ratio = cands[0], 1.0
        for cfg in cands[1:]:
            ln = MK._get_nt_launch(K, G, N, *cfg, 0, 0, False, False)
            rs = [_robust_ab_ratio(base, ln, t) for t, _ in points]
            for i, v in enumerate(rs):
                ratios[cfg][i].append(v)
            gm = (rs[0] * rs[1]) ** 0.5
            if max(rs) < MK._GNT_AT_MARGIN and gm < best_ratio:
                best, best_ratio = cfg, gm
        hist.append(best)
        print(f"  repeat {r}: winner={best}", flush=True)
    for cfg in cands[1:]:
        cells = []
        for i, (pm, skew) in enumerate(MK._GNT_PM_CANON):
            v = ratios[cfg][i]
            sp = "skew" if skew else "bal "
            cells.append(f"pm{pm}/{sp} {statistics.median(v):.4f} [{min(v):.4f},{max(v):.4f}]")
        print(f"  cand {cfg}: " + " | ".join(cells), flush=True)
    print(f"  ==> {'STABLE' if len(set(hist)) == 1 else '*** FLIPPED ***'}  {set(hist)}", flush=True)
    del args, points
    torch.cuda.empty_cache()
