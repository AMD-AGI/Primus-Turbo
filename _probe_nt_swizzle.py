"""Diagnostic: (group_m, num_xcd, group_n) L2-swizzle sweep for the mxfp8 NT kernel on the
REAL campaign workload (M=131072, G=32), balanced and heavy routing, timed interleaved; plus
a determinism check of the first-call config race (cleared cfg cache, repeated).

The swizzle is a pure WG->tile bijection (bit-identical output), so this only ranks L2
residency. Read-only w.r.t. production code.
"""
import statistics
import sys

import torch

import primus_turbo.pytorch  # noqa: F401
import primus_turbo.flydsl.grouped_gemm.mxfp8_grouped_kernel as MK
from primus_turbo.pytorch.core.low_precision import float8_e4m3

DEV = "cuda"
G, M, U = 32, 131072, 512
MODE = sys.argv[1] if len(sys.argv) > 1 else "all"

GRID = [
    (256, 4, 4, 0),  # base
    (256, 8, 4, 0),
    (256, 4, 8, 0),
    (256, 1, 4, 0),
    (256, 2, 4, 0),
    (256, 16, 4, 0),
    (256, 4, 2, 0),
    (256, 4, 1, 0),
    (256, 8, 8, 0),
    (256, 4, 4, 4),
]


def f8(*s):
    t = torch.empty(s, dtype=float8_e4m3, device=DEV)
    t.view(torch.uint8).random_(0, 64)
    return t


def sc(*s):
    return torch.full(s, 127, dtype=torch.uint8, device=DEV)


def _alloc(w):
    NU = M // U
    s = sum(w)
    raw = [max(1, round(NU * wi / s)) for wi in w]
    raw[raw.index(max(raw))] += NU - sum(raw)
    return raw


DISTS = {
    "balanced": [M // U // G] * G,
    "heavy": _alloc([1.0 / (i + 1) ** 2.2 for i in range(G)]),
}


def offs(units):
    o = [0]
    for u in units:
        o.append(o[-1] + u * U)
    return torch.tensor(o, dtype=torch.int64, device=DEV)


def nt_args(N, K, o, a, a_s, w, w_s):
    M_pad = a.shape[0]
    stream = torch.cuda.current_stream()
    a_sp, b_sp, a_blocks, a_ngrp = MK._get_grouped_mx_workspace(M_pad, N, K // 128, G, a.device, stream)
    go = o.view(torch.int32)
    return (
        a.view(torch.int8),
        w.view(torch.int8),
        torch.empty((M_pad, N), dtype=torch.bfloat16, device=a.device),
        a_s.view(torch.int32).reshape(-1),
        w_s.view(torch.int32).reshape(-1),
        a_sp,
        b_sp,
        go,
        go,
        M_pad,
        a_ngrp * 64,
        N,
        a_blocks,
        a_ngrp,
        ((M_pad + 255) // 256 + G) * ((N + 255) // 256),
        stream,
    )


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


def sweep(N, K):
    print(f"\n===== swizzle sweep  N={N} K={K} =====", flush=True)
    a, w = f8(M, K), f8(G, N, K)
    a_s, w_s = sc(M, K // 32), sc(G, N, K // 32)
    res = {}
    for dn, units in DISTS.items():
        args = nt_args(N, K, offs(units), a, a_s, w, w_s)
        MK._get_nt_launch(K, G, N, *GRID[0], 0, 0, False, False, preshuffle=True)(*args)
        torch.cuda.synchronize()
        lns = [MK._get_nt_launch(K, G, N, *c, 0, 0, False, False, preshuffle=False) for c in GRID]
        res[dn] = interleaved([(lambda ln=ln: ln(*args)) for ln in lns])
        del args
    b = {d: v[0] for d, v in res.items()}
    for i, c in enumerate(GRID):
        rb, rh = b["balanced"] / res["balanced"][i], b["heavy"] / res["heavy"][i]
        print(
            f"  gm={c[1]:2d} xcd={c[2]} gn={c[3]} | bal {res['balanced'][i]:7.3f}ms ({rb:5.3f}x)"
            f" | heavy {res['heavy'][i]:7.3f}ms ({rh:5.3f}x) | gmean {(rb*rh)**0.5:5.3f}x",
            flush=True,
        )
    del a, w, a_s, w_s
    torch.cuda.empty_cache()


def picks(trials=3):
    print("\n===== race determinism (cleared cfg cache) =====", flush=True)
    for N, K, tag in ((2944, 2944, "fwd_down/dgrad_down"), (2944, 5760, "dgrad_gate_up"), (5760, 2944, "fwd_gate_up")):
        a, w = f8(M, K), f8(G, N, K)
        a_s, w_s = sc(M, K // 32), sc(G, N, K // 32)
        o = offs(DISTS["balanced"])
        got = []
        for _ in range(trials):
            MK._GNT_CFG_CACHE.clear()
            MK._GNT_AT_CACHE.clear()
            MK.grouped_gemm_mxfp8_flydsl_kernel(a, a_s, w, w_s, o, N, K, num_cu=-1)
            torch.cuda.synchronize()
            got.append(next(iter(MK._GNT_CFG_CACHE.values())))
            torch.cuda.empty_cache()
        print(f"  N={N} K={K:4d} ({tag:20s}) -> {got}  stable={len(set(got))==1}", flush=True)
        del a, w, a_s, w_s
        torch.cuda.empty_cache()


if MODE in ("all", "sweep"):
    sweep(2944, 2944)
if MODE in ("all", "picks"):
    picks()
