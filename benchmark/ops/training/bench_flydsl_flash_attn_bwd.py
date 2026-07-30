#!/usr/bin/env python3
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# Backward perf for primus_turbo.flydsl.attention.flash_attn_bwd, hot-steady
# (continuous warmup, no sleep). Two tables: the square-causal acceptance shapes
# against the H100 FA-v3 TF/s reference, and Meta's 20 configs (10 shapes x
# {full-causal, SWA}) against the faster of H100's two deterministic backends.
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

# Meta's 20 configs: 10 shapes x {full-causal, SWA(W)}, B=4, most rectangular.
# H100 = min(FA-3, FA-2) DETERMINISTIC backward wall time, ms, measured on the same
# shapes (see the aiter reproducer's landscape table); FA-2 is H100's faster
# deterministic backend on SWA. MI350-equivalent = MI355 x 1.2.
META_B = 4
META = [
    #  Hq  Hkv     Sq    Skv     W   H100 full  H100 SWA
    (128, 16, 2048, 16384, 2048, 24.78, 7.88),
    (128, 16, 4096, 16384, 2048, 41.37, 14.04),
    (128, 16, 8192, 16384, 2048, 66.79, 26.50),
    (128, 16, 16384, 16384, 2048, 86.79, 48.39),
    (48, 6, 4096, 4096, 2047, 2.651, 4.625),
    (48, 6, 4096, 8192, 2047, 7.166, 6.113),
    (48, 6, 4096, 12288, 2047, 11.643, 6.274),
    (48, 6, 4096, 16384, 2047, 15.143, 6.438),
    (64, 8, 1024, 1024, 2047, 0.515, 0.488),
    (64, 8, 1024, 16384, 2047, 7.514, 2.385),
]


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


def _auto_it(fn):
    """One timed call sets the iteration count so every block lands near 60 ms."""
    torch.cuda.synchronize()
    t = time.time()
    fn()
    torch.cuda.synchronize()
    return max(2, min(30, int(60.0 / max((time.time() - t) * 1e3, 0.05))))


def bench_meta(Hq, Hkv, Sq, Skv, W):
    scale = 1.0 / math.sqrt(D)
    torch.manual_seed(0)
    q = torch.randn(META_B * Sq, Hq, D, device=DEV, dtype=DT)
    k = torch.randn(META_B * Skv, Hkv, D, device=DEV, dtype=DT)
    v = torch.randn(META_B * Skv, Hkv, D, device=DEV, dtype=DT)
    out = torch.randn(META_B * Sq, Hq, D, device=DEV, dtype=DT)
    dout = torch.randn(META_B * Sq, Hq, D, device=DEV, dtype=DT)
    lse = torch.randn(META_B, Hq, Sq, device=DEV, dtype=torch.float32)

    def r():
        flydsl_varlen_backward(dout, q, k, v, out, lse, META_B, Sq, Skv, Hq, Hkv, D, scale, W)

    global IT
    keep, IT = IT, _auto_it(r)
    ms = _time(r)
    IT = keep
    return ms


def meta20():
    print(f"\n===== Backward, Meta 20 configs  B={META_B}  vs best H100 det (ms) =====", flush=True)
    print(
        f"{'Hq':>4} {'Sq':>6} {'Skv':>6} {'W':>5} | {'ours':>7} {'MI350':>7} {'H100':>7} {'xH100':>6} {'ok':>4}"
        f" | {'ours':>7} {'MI350':>7} {'H100':>7} {'xH100':>6} {'ok':>4}",
        flush=True,
    )
    print(f"{'':>24} | {'--- full causal ---':^36} | {'------- SWA -------':^36}", flush=True)
    npass = 0
    for Hq, Hkv, Sq, Skv, W, h_full, h_swa in META:
        cells = []
        for w, h100 in ((-1, h_full), (W, h_swa)):
            ms = bench_meta(Hq, Hkv, Sq, Skv, w)
            mi350 = ms * 1.2
            xh = h100 / mi350
            npass += xh >= 1.4
            cells.append(f"{ms:7.3f} {mi350:7.3f} {h100:7.3f} {xh:6.2f} {'PASS' if xh >= 1.4 else '--':>4}")
        print(f"{Hq:4d} {Sq:6d} {Skv:6d} {W:5d} | {cells[0]} | {cells[1]}", flush=True)
    print(f"  -> {npass}/20 clear 1.4x best-H100-deterministic (MI350-equivalent)", flush=True)


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
    meta20()


if __name__ == "__main__":
    main()
