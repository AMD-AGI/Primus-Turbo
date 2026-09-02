#!/usr/bin/env python3
"""Sweep the dense-fp8 tile/band geometry for the proj unit's GEMM shapes.

The shipped candidate tables offer two to four arms per regime and never vary GROUP_M /
group_n / num_xcd for the shapes this campaign runs. This widens the tables, lets the
existing race build every arm, and prints the whole timing distribution: builds all
happen before any timing, so the ranking is a clean palindrome over pre-built kernels
(pitfalls/02: a sweep that recompiles between arms cannot be palindrome-corrected).
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import primus_turbo.flydsl.gemm.gemm_fp8_kernel as GK  # noqa: E402

DEV = "cuda"
F8 = torch.float8_e4m3fn
REC = []


def _time(fn, iters=20, reps=2):
    torch.cuda.synchronize()
    best = float("inf")
    for _ in range(reps):
        e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
        e0.record()
        for _ in range(iters):
            fn()
        e1.record()
        torch.cuda.synchronize()
        best = min(best, e0.elapsed_time(e1) / iters)
    return best


def _pick(cands, args):
    n = len(cands)
    ts = [float("inf")] * n
    order = list(range(n))
    for _ in range(300):
        cands[0][2](*args)
    torch.cuda.synchronize()
    for r in range(4):
        for i in order if r % 2 == 0 else order[::-1]:
            ts[i] = min(ts[i], _time(lambda c=cands[i][2]: c(*args)))
    best = min(order, key=ts.__getitem__)
    REC.append((sorted(zip([round(t * 1000, 2) for t in ts], [str(c[1]) for c in cands])), cands[best][1]))
    return cands[best]


GK._pick_dense_candidate = _pick

XCD = [1, 2, 4, 8]
GM = [1, 2, 4, 8]

# 8-wave NT: (BLOCK_M, GROUP_M, num_xcd, AGPR)
NT8 = [(256, gm, x, 32) for gm in GM for x in XCD] + [(256, 4, 8, 44), (256, 2, 1, 44)]
# 8-wave NN: (BLOCK_M, GROUP_M, group_n, num_xcd, AGPR)
NN8 = [(256, gm, gn, x, 44) for gm in (1, 2, 4) for gn in (0, 4, 8) for x in (4, 8)]
# whole-loop NT bands: (GROUP_M, num_xcd)
NT4 = [(gm, x) for gm in GM for x in XCD]
# whole-loop NN bands: (GROUP_M, group_n, num_xcd)
NN4 = [(gm, gn, x) for gm in (1, 2, 4, 8) for gn in (0, 8) for x in (2, 8)]


def q(*shape):
    return (torch.randn(*shape, device=DEV) * 0.3).to(F8)


CASES = {
    "qkv_fwd": ((32768, 2880), (5120, 2880), False, True),
    "o_fwd": ((32768, 4096), (2880, 4096), False, True),
    "o_dg": ((32768, 2880), (2880, 4096), False, False),
    "qkv_dg": ((32768, 5120), (5120, 2880), False, False),
}


def main():
    which = sys.argv[1]
    GK._NT_CANDIDATES = {k: NT8 for k in GK._NT_CANDIDATES}
    GK._NN_CANDIDATES = {k: NN8 for k in GK._NN_CANDIDATES}
    GK._NT4_BANDS = NT4
    GK._NN4_BANDS = NN4
    ash, bsh, ta, tb = CASES[which]
    a, b = q(*ash), q(*bsh)
    sa = torch.ones(1, device=DEV, dtype=torch.float32)
    REC.clear()
    out = GK.gemm_fp8_tensorwise_flydsl_kernel(a, sa, b, sa, trans_a=ta, trans_b=tb)
    torch.cuda.synchronize()
    print(f"== {which} a{tuple(ash)} b{tuple(bsh)} -> {tuple(out.shape)}")
    for ts, best in REC:
        for t, c in ts:
            print(f"    {t:8.2f}  {c}")
        print(f"    shipped-race pick = {best}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
