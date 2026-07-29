"""Which half-N variant races? Same shape/grid, only the boundary body differs.

hn_mode 0 = current tree (b1 g2s dropped, vmcnt counts lowered by N_LDS_STEPS_B)
hn_mode 1 = round-4 form  (b1 g2s kept, original counts, only ds_read+MFMA skipped)
hn_mode 2 = no boundary body at all (full body for every tile)
Byte-compares 6 repeats per variant and reports differing column blocks + magnitude.
"""
import sys

import torch

import primus_turbo.flydsl.grouped_gemm.mxfp8_grouped_kernel as MK
import primus_turbo.pytorch  # noqa: F401
from primus_turbo.pytorch.core.low_precision import float8_e4m3

DEV = "cuda"
G, M = 32, 131072
CFG = (256, 4, 4, 0)
N, K = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) > 2 else (2944, 2944)
REPS = 6


def f8(*s):
    t = torch.empty(s, dtype=float8_e4m3, device=DEV)
    t.view(torch.uint8).random_(0, 64)
    return t


a, w = f8(M, K), f8(G, N, K)
a_s = torch.full((M, K // 32), 127, dtype=torch.uint8, device=DEV)
w_s = torch.full((G, N, K // 32), 127, dtype=torch.uint8, device=DEV)
o = torch.arange(0, G + 1, dtype=torch.int64, device=DEV) * (M // G)
stream = torch.cuda.current_stream()
a_sp, b_sp, a_blocks, a_ngrp = MK._get_grouped_mx_workspace(M, N, K // 128, G, DEV, stream)
c = torch.zeros((M, N), dtype=torch.bfloat16, device=DEV)
args = (
    a.view(torch.int8),
    w.view(torch.int8),
    c,
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
gold = c.clone()

print(f"N={N} K={K} reps={REPS}", flush=True)
for hm in (1, 4, 5):
    ln = MK._get_nt_launch(K, G, N, *CFG, 0, 0, False, False, preshuffle=False, hn_mode=hm)
    reps = []
    for _ in range(REPS):
        c.zero_()
        ln(*args)
        torch.cuda.synchronize()
        reps.append(c.clone())
    bad = 0
    info = ""
    for i in range(1, REPS):
        ne = reps[0] != reps[i]
        nr = int(ne.any(dim=1).sum())
        if nr:
            bad += 1
            cb = sorted(set(int(v) for v in ne.any(dim=0).nonzero().flatten() // 256))
            mx = (reps[0].float() - reps[i].float()).abs().max()
            info = f" cols={cb} maxabs={mx:.1f} rows={nr}"
    dg = reps[0] != gold
    ndg = int(dg.any(dim=1).sum())
    cbg = sorted(set(int(v) for v in dg.any(dim=0).nonzero().flatten() // 256)) if ndg else []
    print(f"  hn_mode={hm}: nondet {bad}/{REPS-1} pairs{info} | vs gold rows={ndg} cols={cbg}", flush=True)
