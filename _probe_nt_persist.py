"""Diagnostic: mxfp8 NT persistent vs non-persistent on the real campaign workload.

Non-persistent pays the per-WG O(G) group-scan prologue once per tile; persistent amortises
it over ~tiles/CU tiles but pays the scf.for tile-loop penalty. Interleaved timing, both
token distributions, all three NT shapes of the bench.
"""
import statistics

import torch

import primus_turbo.pytorch  # noqa: F401
import primus_turbo.flydsl.grouped_gemm.mxfp8_grouped_kernel as MK
from primus_turbo.pytorch.core.low_precision import float8_e4m3

DEV = "cuda"
G, M, U = 32, 131072, 512
CFGS = [(256, 4, 4, 0), (256, 8, 4, 0), (256, 4, 8, 0)]


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


DISTS = {"balanced": [M // U // G] * G, "heavy": _alloc([1.0 / (i + 1) ** 2.2 for i in range(G)])}


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


for N, K in ((2944, 2944), (2944, 5760), (5760, 2944)):
    print(f"\n===== N={N} K={K} =====", flush=True)
    a, w = f8(M, K), f8(G, N, K)
    a_s, w_s = sc(M, K // 32), sc(G, N, K // 32)
    for dn, units in DISTS.items():
        args = nt_args(N, K, offs(units), a, a_s, w, w_s)
        MK._get_nt_launch(K, G, N, *CFGS[0], 0, 0, False, False, preshuffle=True)(*args)
        torch.cuda.synchronize()
        tags, lns = [], []
        for c in CFGS:
            for p in (False, True):
                tags.append(f"gm{c[1]}x{c[2]}{'P' if p else 'N'}")
                lns.append(MK._get_nt_launch(K, G, N, *c, 0, 0, False, p, preshuffle=False))
        ts = interleaved([(lambda ln=ln: ln(*args)) for ln in lns])
        base = ts[0]
        print(f"  {dn:9s} | " + "  ".join(f"{t_}:{v:7.3f}({base/v:5.3f}x)" for t_, v in zip(tags, ts)), flush=True)
        del args
    del a, w, a_s, w_s
    torch.cuda.empty_cache()
