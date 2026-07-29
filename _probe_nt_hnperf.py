"""Price the half-N boundary-body variants, interleaved, on the campaign NT shapes.

hn_mode 0 = b1 g2s dropped + partial drain (RACY, reference only)
hn_mode 1 = b1 g2s kept, original drains (round-4 form)
hn_mode 2 = no boundary body at all (full quadrants everywhere)
hn_mode 4 = b1 g2s dropped + vmcnt(0) drains in the boundary body (det-clean)
"""
import statistics

import torch

import primus_turbo.flydsl.grouped_gemm.mxfp8_grouped_kernel as MK
import primus_turbo.pytorch  # noqa: F401
from primus_turbo.pytorch.core.low_precision import float8_e4m3

DEV = "cuda"
G, M, U = 32, 131072, 512
CFG = (256, 4, 4, 0)
MODES = (1, 0, 4, 5)


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


DISTS = {"balanced": [M // U // G] * G, "heavy": _alloc([1.0 / (i + 1) ** 2.2 for i in range(G)])}


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


for N, K in ((2944, 2944), (2944, 5760), (5760, 2944)):
    print(f"\n===== N={N} K={K} =====", flush=True)
    a, w = f8(M, K), f8(G, N, K)
    a_s = torch.full((M, K // 32), 127, dtype=torch.uint8, device=DEV)
    w_s = torch.full((G, N, K // 32), 127, dtype=torch.uint8, device=DEV)
    for dn, units in DISTS.items():
        o = offs(units)
        stream = torch.cuda.current_stream()
        a_sp, b_sp, a_blocks, a_ngrp = MK._get_grouped_mx_workspace(M, N, K // 128, G, DEV, stream)
        args = (
            a.view(torch.int8),
            w.view(torch.int8),
            torch.empty((M, N), dtype=torch.bfloat16, device=DEV),
            a_s.view(torch.int32).reshape(-1),
            w_s.view(torch.int32).reshape(-1),
            a_sp,
            b_sp,
            o.view(torch.int32),
            o.view(torch.int32),
            M,
            a_ngrp * 64,
            N,
            a_blocks,
            a_ngrp,
            ((M + 255) // 256 + G) * ((N + 255) // 256),
            stream,
        )
        MK._get_nt_launch(K, G, N, *CFG, 0, 0, False, False, preshuffle=True)(*args)
        torch.cuda.synchronize()
        lns = [MK._get_nt_launch(K, G, N, *CFG, 0, 0, False, False, preshuffle=False, hn_mode=h) for h in MODES]
        ts = interleaved([(lambda ln=ln: ln(*args)) for ln in lns])
        base = ts[0]
        print(f"  {dn:9s} | " + "  ".join(f"h{h}:{t:7.3f}({base/t:6.4f}x)" for h, t in zip(MODES, ts)), flush=True)
        del args
    del a, w, a_s, w_s
    torch.cuda.empty_cache()
