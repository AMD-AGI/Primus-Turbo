#!/usr/bin/env python3
"""Round-8 probe: the scored proj unit, in isolation, for tracing and A/B arms.

Mirrors `_bench_gptoss_stepfuse.py`'s proj part exactly (same shapes, same call
sequence, same cotangents) so a dispatch census taken here describes the scored unit.
"""
import json
import os
import statistics as st
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from primus_turbo.pytorch.ops.gemm_fp8 import gemm_fp8  # noqa: E402
from primus_turbo.pytorch.ops.normalization import rmsnorm, rmsnorm_residual  # noqa: E402

DEV, BF = "cuda", torch.bfloat16
S, B, H = 8192, 4, 2880
NTOK = S * B
HQ, HKV, HD = 64, 8, 64
QKV = HQ * HD + 2 * HKV * HD
AO = HQ * HD


def proj_leaves():
    torch.manual_seed(11)
    mk = lambda *s: torch.randn(*s, device=DEV, dtype=BF).requires_grad_()  # noqa: E731
    return {
        "x": mk(NTOK, H),
        "res": mk(NTOK, H),
        "g_in": mk(H),
        "x2": mk(NTOK, H),
        "g_post": mk(H),
        "w_qkv": mk(QKV, H),
        "ao": mk(NTOK, AO),
        "w_o": mk(H, AO),
        "gq": mk(HD),
        "gk": mk(HD),
    }


def proj_cots():
    r = lambda *s: torch.randn(*s, device=DEV, dtype=BF)  # noqa: E731
    return [r(NTOK, QKV), r(NTOK, H), r(S, B, HQ, HD), r(S, B, HKV, HD), r(NTOK, H)]


def proj_unit(lv, cots):
    h, _ = rmsnorm_residual(lv["x"], lv["res"], lv["g_in"])
    qkv = gemm_fp8(h, lv["w_qkv"], trans_b=True)
    h2 = rmsnorm(lv["x2"], lv["g_post"])
    o = gemm_fp8(lv["ao"], lv["w_o"], trans_b=True)
    q = rmsnorm(qkv[:, :AO].view(S, B, HQ, HD), lv["gq"])
    k = rmsnorm(qkv[:, AO : AO + HKV * HD].view(S, B, HKV, HD), lv["gk"])
    torch.autograd.backward([qkv, o, q, k, h2], cots)


def timed(fn, warms=5, reps=20):
    for _ in range(warms):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
        e0.record()
        fn()
        e1.record()
        torch.cuda.synchronize()
        ts.append(e0.elapsed_time(e1))
    return st.median(ts)


def make_step():
    lv, cots = proj_leaves(), proj_cots()

    def step():
        for v in lv.values():
            v.grad = None
        proj_unit(lv, cots)

    return step


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "time"
    if mode == "trace":
        # Few, clean iterations: the analyzer segments on rmsnorm_fwd_residual_kernel.
        step = make_step()
        for _ in range(6):
            step()
        torch.cuda.synchronize()
        for _ in range(3):
            step()
        torch.cuda.synchronize()
        return 0
    if mode == "time":
        step = make_step()
        out = [round(timed(step), 4) for _ in range(int(sys.argv[2]) if len(sys.argv) > 2 else 3)]
        print(json.dumps({"proj_ms": out}))
        return 0
    raise SystemExit(f"unknown mode {mode}")


if __name__ == "__main__":
    sys.exit(main())
