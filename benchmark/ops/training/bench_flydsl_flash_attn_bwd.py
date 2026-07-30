#!/usr/bin/env python3
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# Square-causal backward perf for primus_turbo.flydsl.attention.flash_attn_bwd,
# hot-steady (continuous warmup, no sleep) against the H100 FA-v3 reference.
# Times flydsl_varlen_backward, i.e. odo + dq + dkdv plus the split-K dK/dV reduce,
# so the number cannot drift from the deployed configuration.
#
#   HIP_VISIBLE_DEVICES=0 python3 bench_flydsl_flash_attn_bwd.py
import math
import os
import time

import torch

from primus_turbo.flydsl.attention.flash_attn_bwd import flydsl_varlen_backward

DEV = "cuda"
DT = torch.bfloat16
D = 64

HQ, HKV = 128, 16
SQUARE_S = [2048, 4096, 8192, 16384]
# H100 FA-v3 square-causal bwd reference (B=1, D=64, TF/s).
H100_SQ = {2048: 226, 4096: 377, 8192: 466, 16384: 508}
B = 1
WARMS = 3.0
REPS = 7
IT = 20
NODE = os.uname().nodename


def _causal_frac(Sq, Skv):
    return 1.0 - (Sq - 1) / (2.0 * Skv)


def _time(fn):
    t0 = time.time()
    n = 0
    while time.time() - t0 < WARMS:  # continuous full-load warmup, no sleep
        fn()
        n += 1
        if n % 20 == 0:
            torch.cuda.synchronize()
    torch.cuda.synchronize()
    ts = []
    for _ in range(REPS):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(IT):
            fn()
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e) / IT)
    ts.sort()
    return ts[len(ts) // 2]


def bench_one(S):
    scale = 1.0 / math.sqrt(D)
    torch.manual_seed(0)
    q = torch.randn(B * S, HQ, D, device=DEV, dtype=DT)
    k = torch.randn(B * S, HKV, D, device=DEV, dtype=DT)
    v = torch.randn(B * S, HKV, D, device=DEV, dtype=DT)
    out = torch.randn(B * S, HQ, D, device=DEV, dtype=DT)
    dout = torch.randn(B * S, HQ, D, device=DEV, dtype=DT)
    lse = torch.randn(B, HQ, S, device=DEV, dtype=torch.float32)

    def r():
        flydsl_varlen_backward(dout, q, k, v, out, lse, B, S, S, HQ, HKV, D, scale)

    ms = _time(r)
    # 5 backward GEMMs: dP, dS@K, dS^T@Q, P^T@dO and the recomputed S.
    flop = 10.0 * B * HQ * S * S * D * _causal_frac(S, S)
    return ms, flop / 1e12 / (ms / 1e3)


def main():
    print(
        f"device={torch.cuda.get_device_name(0)} arch={torch.cuda.get_device_properties(0).gcnArchName}",
        flush=True,
    )
    print(f"\n===== Backward hd64 THD  B={B}  deterministic  hw-exp  [node {NODE}] =====", flush=True)
    print(
        f"{'S':>6} {'H100_bwd':>9} {'1.4xtgt(MI350)':>15} {'MI355':>7} {'MI350(/1.2)':>12} {'xH100':>6} {'verdict':>8}",
        flush=True,
    )
    npass = 0
    for S in SQUARE_S:
        _ms, tf = bench_one(S)
        h = H100_SQ[S]
        tgt = h * 1.40
        mi350 = tf / 1.2
        xh = mi350 / h
        if xh >= 1.4:
            verdict = "PASS"
            npass += 1
        else:
            verdict = f"{(xh / 1.4 - 1) * 100:+.1f}%"
        print(
            f"{S:6d} {h:9d} {tgt:15.0f} {tf:7.0f} {mi350:12.0f} {xh:6.2f} {verdict:>8}",
            flush=True,
        )
    print(f"  -> {npass}/4 clear 1.4x H100 (MI350-equivalent)", flush=True)


if __name__ == "__main__":
    main()
