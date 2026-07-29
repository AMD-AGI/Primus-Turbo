"""Localise the mxfp8 NT nondeterminism seen at the campaign timing shape.

Reports, per shape, repeat-vs-repeat byte diffs with the differing column-blocks and the
error magnitude. N=2944/5760 emit the half-N boundary body (_HALF_N), N=3072/5632 are
256-aligned and emit only the full body, so the two tell apart "boundary body" from a
pre-existing race in the full body.
"""
import torch

import primus_turbo.flydsl.grouped_gemm.mxfp8_grouped_kernel as MK
import primus_turbo.pytorch  # noqa: F401
from primus_turbo.pytorch.core.low_precision import float8_e4m3

DEV = "cuda"
G, M = 32, 131072
CFG = (256, 4, 4, 0)


def f8(*s):
    t = torch.empty(s, dtype=float8_e4m3, device=DEV)
    t.view(torch.uint8).random_(0, 64)
    return t


def diff(x, y, N):
    ne = x != y
    rows = ne.any(dim=1)
    nr = int(rows.sum())
    if nr == 0:
        return "identical"
    cb = ne.any(dim=0).nonzero().flatten() // 256
    blocks = sorted(set(int(v) for v in cb))
    mx = (x.float() - y.float()).abs().max()
    ref = x.float().abs().mean()
    return f"rows={nr}/{x.shape[0]} col_blocks={blocks} maxabs={mx:.3f} mean|x|={ref:.2f}"


for N, K in ((2944, 2944), (3072, 2944), (5760, 2944), (5632, 2944)):
    hn = (N % 256 != 0) and (N % 256 <= 128)
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
    only = MK._get_nt_launch(K, G, N, *CFG, 0, 0, False, False, preshuffle=False)
    reps = []
    for _ in range(4):
        c.zero_()
        only(*args)
        torch.cuda.synchronize()
        reps.append(c.clone())
    print(f"\n== N={N} K={K} half_n={hn} n_blocks={-(-N // 256)} ==", flush=True)
    for i in range(1, 4):
        print(f"  rep0 vs rep{i}: {diff(reps[0], reps[i], N)}", flush=True)
    del a, w, a_s, w_s, c, args, reps
    torch.cuda.empty_cache()
