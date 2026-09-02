#!/usr/bin/env python3
"""Round-8 probe: is the fused GLU/dGLU swizzle race picking the deployment arm?

The race scores at M_total = 32768 (1024 rows/expert); the deployed MLP runs 131072
(4096 rows/expert). `group_m` is an L2 blocking knob whose optimum moves with the
M-tile count, so time every arm at the M that ships.
"""
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import _bench_gptoss_stepfuse as BE
import primus_turbo.flydsl.grouped_gemm.grouped_gemm_fp8_glu_kernel as GLU

DEV = "cuda"


def _mk_step():
    leaves = BE._mlp_leaves(BE.M, BE.G)
    lens = BE._lens(BE.M, BE.G, False)
    cot = torch.randn(BE.M, BE.K, device=DEV, dtype=torch.bfloat16)
    x, w1, w2, probs = (t.detach().requires_grad_(True) for t in leaves)

    def step():
        for a in (x, w1, w2):
            a.grad = None
        BE._mlp(x, w1, w2, probs, lens).backward(cot)

    return step, (x, w1, w2)


def _time(fn, warms, reps):
    for _ in range(warms):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / reps * 1e3


_SHIP_NT, _SHIP_NN = GLU._GLU_NT_CANDS, GLU._GLU_NN_CANDS


def _set(nt, nn):
    GLU._GLU_NT_CANDS = nt
    GLU._GLU_NN_CANDS = nn
    GLU._GROUPED_GLU_CACHE.clear()
    GLU._GROUPED_DGLU_CACHE.clear()


def sweep(which, warms=4, reps=12):
    step, grads = _mk_step()
    arms = [(x, g) for x in (2, 4, 8) for g in (0, 2, 4, 8, 16)]
    if which == "nt":
        sel = lambda a: ((a,), _SHIP_NN)  # noqa: E731
    else:
        sel = lambda a: (_SHIP_NT, (a,))  # noqa: E731

    out = {}
    ref = None
    for a in arms:
        _set(*sel(a))
        step()  # build + race + discard arm, post-compile
        t = _time(step, warms, reps)
        g = tuple(x.grad.clone() for x in grads)
        if ref is None:
            ref = g
            bad = ""
        else:
            bad = "" if all(torch.equal(p, q) for p, q in zip(ref, g)) else "  DIFF"
        out[a] = t
        print(f"{which} xcd={a[0]} gm={a[1]:2d}  {t:.4f} ms{bad}", flush=True)
    best = min(out, key=out.get)
    base = out[_SHIP_NT[0] if which == "nt" else _SHIP_NN[0]]
    print(f"{which} best {best} {out[best]:.4f}  ship-arm0 {base:.4f}  ratio {out[best]/base:.4f}", flush=True)


if __name__ == "__main__":
    sweep(sys.argv[1] if len(sys.argv) > 1 else "nn")
