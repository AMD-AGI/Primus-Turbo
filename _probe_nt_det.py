"""Determinism check for mxfp8 NT at the campaign TIMING shape (M=131072, G=32).

The bench det gate runs a small g=2 shape; this repeats the same byte-exact comparisons at
the scored shape: full-vs-full, preshuffle=False vs full, and back-to-back repeats. Also
reports how many rows/cols differ so a padding artefact can be told from a real race.
"""
import torch

import primus_turbo.flydsl.grouped_gemm.mxfp8_grouped_kernel as MK
import primus_turbo.pytorch  # noqa: F401
from primus_turbo.pytorch.core.low_precision import float8_e4m3

DEV = "cuda"
G, M, U = 32, 131072, 512
CFG = (256, 4, 4, 0)


def f8(*s):
    t = torch.empty(s, dtype=float8_e4m3, device=DEV)
    t.view(torch.uint8).random_(0, 64)
    return t


def report(tag, x, y):
    eq = bool(torch.equal(x, y))
    d = (x != y).any(dim=1)
    print(f"  {tag:28s} equal={eq} rows_diff={int(d.sum())}/{x.shape[0]} first={int(d.nonzero()[0]) if int(d.sum()) else -1}")


for N, K in ((2944, 2944),):
    a, w = f8(M, K), f8(G, N, K)
    a_s = torch.full((M, K // 32), 127, dtype=torch.uint8, device=DEV)
    w_s = torch.full((G, N, K // 32), 127, dtype=torch.uint8, device=DEV)
    o = torch.arange(0, G + 1, dtype=torch.int64, device=DEV) * (M // G)
    stream = torch.cuda.current_stream()
    a_sp, b_sp, a_blocks, a_ngrp = MK._get_grouped_mx_workspace(M, N, K // 128, G, DEV, stream)
    c = torch.empty((M, N), dtype=torch.bfloat16, device=DEV)
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
    full = MK._get_nt_launch(K, G, N, *CFG, 0, 0, False, False, preshuffle=True)
    only = MK._get_nt_launch(K, G, N, *CFG, 0, 0, False, False, preshuffle=False)

    full(*args)
    torch.cuda.synchronize()
    r1 = c.clone()
    full(*args)
    torch.cuda.synchronize()
    r2 = c.clone()
    report("full vs full", r1, r2)
    c.zero_()
    only(*args)
    torch.cuda.synchronize()
    r3 = c.clone()
    report("preshuf=False vs full", r3, r1)
    c.zero_()
    only(*args)
    torch.cuda.synchronize()
    report("preshuf=False x2", c, r3)
    print(f"  mean|full|={r1.float().abs().mean():.4f} nan={int(r1.isnan().sum())}")
