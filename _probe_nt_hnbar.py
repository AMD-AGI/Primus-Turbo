"""Price the boundary-body phase compression (hn_phase) on the three NT campaign shapes.

Interleaved A/B in one process (drift-immune): base = hn_phase False (4 phases, two of
them hollow in the b0-only body), cand = hn_phase True (2-phase skeleton). Shapes:
(N=2944,K=2944) fwd/dgrad down, (N=2944,K=5760) dgrad gate_up, (N=5760,K=2944) fwd gate_up.
"""
import statistics
import sys

import torch

import primus_turbo.flydsl.grouped_gemm.mxfp8_grouped_kernel as MK
import primus_turbo.pytorch  # noqa: F401
from primus_turbo.pytorch.core.low_precision import float8_e4m3

DEV = "cuda"
G, M, U = 32, 131072, 512
CFG = (256, 4, 4, 0)
REPS = int(sys.argv[1]) if len(sys.argv) > 1 else 13


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


def interleaved(fns, warmup=10, reps=REPS):
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
    print(f"\n===== N={N} K={K} half_n={(N % 256 != 0) and (N % 256 <= 128)} =====", flush=True)
    a, w = f8(M, K), f8(G, N, K)
    a_s = torch.full((M, K // 32), 127, dtype=torch.uint8, device=DEV)
    w_s = torch.full((G, N, K // 32), 127, dtype=torch.uint8, device=DEV)
    for dn, units in DISTS.items():
        o = offs(units)
        stream = torch.cuda.current_stream()
        a_sp, b_sp, a_blocks, a_ngrp = MK._get_grouped_mx_workspace(M, N, K // 128, G, DEV, stream)
        outs = [torch.zeros((M, N), dtype=torch.bfloat16, device=DEV) for _ in range(2)]

        def mk(hp, out):
            args = (
                a.view(torch.int8),
                w.view(torch.int8),
                out,
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
            MK._get_nt_launch(K, G, N, *CFG, 0, 0, False, False, preshuffle=True, hn_phase=hp)(*args)
            torch.cuda.synchronize()
            ln = MK._get_nt_launch(K, G, N, *CFG, 0, 0, False, False, preshuffle=False, hn_phase=hp)
            return lambda: ln(*args)

        fns = [mk(False, outs[0]), mk(True, outs[1])]
        ts = interleaved(fns)
        for f in fns:
            f()
        torch.cuda.synchronize()
        mism = int((outs[0] != outs[1]).sum())
        print(
            f"  {dn:9s} | base {ts[0]:7.3f}ms  cand {ts[1]:7.3f}ms  cand/base {ts[1]/ts[0]:6.4f}"
            f"  mismatch={mism}",
            flush=True,
        )
        del outs
    del a, w, a_s, w_s
    torch.cuda.empty_cache()
