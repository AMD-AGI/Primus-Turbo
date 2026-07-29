"""Diagnostic: (group_m, num_xcd, group_n) tile-swizzle sweep for the mxfp8 variable-K wgrad
on the real campaign workload (M_total=131072, G=32), all three token distributions, timed
interleaved. The swizzle is a pure WG->tile bijection (bit-identical), so this only ranks L2
residency / dispatch order. Read-only w.r.t. production code.
"""
import statistics

import torch

import primus_turbo.pytorch  # noqa: F401
import primus_turbo.flydsl.grouped_gemm.mxfp8_grouped_kernel as MK
from primus_turbo.flydsl.utils.gemm_helper import ceildiv
from primus_turbo.pytorch.core.low_precision import float8_e4m3

DEV = "cuda"
G, M, U = 32, 131072, 512
PROJS = {"gate_up": (2944, 5760), "down": (2944, 2944)}
PACK = 4
GRID = [(4, 1, 0), (8, 1, 0), (2, 1, 0), (16, 1, 0), (8, 1, 4)]  # (group_m, num_xcd, group_n)


def f8(*s):
    t = torch.empty(s, dtype=float8_e4m3, device=DEV)
    t.view(torch.uint8).random_(0, 64)
    return t


def _alloc(w):
    NU = M // U
    s = sum(w)
    raw = [max(1, round(NU * wi / s)) for wi in w]
    raw[raw.index(max(raw))] += NU - sum(raw)
    return raw


DISTS = {
    "balanced": [M // U // G] * G,
    "moderate": _alloc([1.0 / (i + 1) ** 1.1 for i in range(G)]),
    "heavy": _alloc([1.0 / (i + 1) ** 2.2 for i in range(G)]),
}


def offs(units):
    o = [0]
    for u in units:
        o.append(o[-1] + u * U)
    return torch.tensor(o, dtype=torch.int64, device=DEV)


def interleaved(fns, warmup=10, reps=13):
    for _ in range(warmup):
        for f in fns:
            f()
    torch.cuda.synchronize()
    acc = [[] for _ in fns]
    for _ in range(reps):
        for i, f in enumerate(fns):
            e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
            e0.record()
            f()
            e1.record()
            torch.cuda.synchronize()
            acc[i].append(e0.elapsed_time(e1))
    return [statistics.median(x) for x in acc]


for proj, (OUT_M, OUT_N) in PROJS.items():
    print(f"\n===== wgrad {proj}  OUT_M={OUT_M} OUT_N={OUT_N} =====", flush=True)
    lhs, rhs = f8(OUT_M, M), f8(OUT_N, M)
    l_s = torch.full((OUT_M, M // 32), 127, dtype=torch.uint8, device=DEV)
    r_s = torch.full((OUT_N, M // 32), 127, dtype=torch.uint8, device=DEV)
    K128 = M // 128
    stream = torch.cuda.current_stream()
    a_sp, b_sp = MK._get_grouped_wgrad_workspace(OUT_M, OUT_N, K128, G, PACK, DEV, stream)
    a_ngrp = ceildiv(OUT_M, 64)
    b_ngrp = ((OUT_N + 255) // 256) * 4
    n_ck = K128 // MK._PRESHUF_KT + G
    a_blocks = a_ngrp * n_ck
    pre_grid = a_blocks + b_ngrp * n_ck
    out = torch.empty((G, OUT_M, OUT_N), dtype=torch.bfloat16, device=DEV)
    base = (
        lhs.view(torch.int8),
        rhs.view(torch.int8),
        out,
        l_s.view(torch.int32).reshape(-1),
        r_s.view(torch.int32).reshape(-1),
        a_sp,
        b_sp,
    )
    lns = {}
    for gm, xcd, gn in GRID:
        for pre in (True, False):
            lns[(gm, xcd, gn, pre)] = MK._compile_grouped_mxfp8_wgrad_fused(
                OUT_M, OUT_N, G, 256, 256, gm, xcd, gn, 0, 0, False, pack=PACK, preshuffle=pre
            )
    for dn, units in DISTS.items():
        go = offs(units).view(torch.int32)
        args = base + (go, M, K128, n_ck, a_blocks, pre_grid, stream)
        lns[(GRID[0][0], GRID[0][1], GRID[0][2], True)](*args)
        torch.cuda.synchronize()
        ts = interleaved([(lambda ln=lns[(g, x, n, False)]: ln(*args)) for g, x, n in GRID])
        b0 = ts[0]
        print(
            f"  {dn:9s} | " + "  ".join(f"gm{g}x{x}n{n}:{v:7.3f}({b0/v:5.3f}x)" for (g, x, n), v in zip(GRID, ts)),
            flush=True,
        )
    del lhs, rhs, l_s, r_s, out
    torch.cuda.empty_cache()
