#!/usr/bin/env python3
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# Ragged / block-causal (packed document-masking) backward perf for the flydsl hd64
# THD path, hot-steady (continuous warmup, no sleep). This is a SEPARATE table from
# the uniform 20-config acceptance bench: the ragged grid tiles by max_seqlen, so
# skewed segment layouts pay early-exit waste on the short segments (the standard
# packed-varlen cost) -- ragged is correctness-first, not on the 1.4x-H100 metric.
# Drives the impl layer (fwd once for out/lse, then times the ragged backward), so
# the number matches the deployed dispatch. No H100 reference for ragged shapes.
#
#   HIP_VISIBLE_DEVICES=0 python3 bench_flydsl_flash_attn_varlen_bwd.py
import math
import os
import time

import torch

from primus_turbo.pytorch.kernels.attention.attention_flydsl_impl import (
    flash_attn_varlen_flydsl_backward_impl,
    flash_attn_varlen_flydsl_forward_impl,
)

DEV = "cuda"
DT = torch.bfloat16
D = 64
HQ, HKV = 64, 8  # GQA G=8 (flydsl requires G a power of two >= 8)
WARMS = 3.0
REPS = 7
NODE = os.uname().nodename

# (tag, segment layout). A uniform layout is included as the zero-waste baseline; the
# rest are ragged (document packing) with increasing segment-length skew.
CONFIGS = [
    ("uniform", [2048, 2048, 2048, 2048]),
    ("mild", [1024, 2048, 4096, 1024]),
    ("skew", [512, 2048, 1024, 4096]),
    ("longtail", [4096, 512, 256, 128]),
]
WINDOWS = [(-1, -1), (2048, 0)]


def _attended_frac(S, win):
    """Fraction of the SxS score block a query row attends: bottom-right causal, minus
    what a left window of `win` prunes. Continuous with the causal fraction at win>=S-1."""
    if win < 0 or win >= S - 1:
        return (S + 1) / (2.0 * S)
    return (win * (win + 1) / 2.0 + (S - win) * (win + 1)) / (S * S)


def _build_cu(segs):
    cu = torch.zeros(len(segs) + 1, device=DEV, dtype=torch.int32)
    cu[1:] = torch.cumsum(torch.tensor(segs, device=DEV, dtype=torch.int32), 0)
    return cu, max(segs), int(cu[-1].item())


def _time(fn, it):
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
        for _ in range(it):
            fn()
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e) / it)
    ts.sort()
    return ts[len(ts) // 2]


def _auto_it(fn):
    """One timed call sets the iteration count so every block lands near 60 ms."""
    fn()  # first call carries the JIT compile; do not time it
    torch.cuda.synchronize()
    t = time.time()
    fn()
    torch.cuda.synchronize()
    return max(2, min(30, int(60.0 / max((time.time() - t) * 1e3, 0.05))))


def bench_case(segs, win):
    scale = 1.0 / math.sqrt(D)
    cu, maxs, total = _build_cu(segs)
    torch.manual_seed(0)
    q = torch.randn(total, HQ, D, device=DEV, dtype=DT)
    k = torch.randn(total, HKV, D, device=DEV, dtype=DT)
    v = torch.randn(total, HKV, D, device=DEV, dtype=DT)
    do = torch.randn(total, HQ, D, device=DEV, dtype=DT)
    ws = (win, 0) if win >= 0 else (-1, -1)
    out, lse = flash_attn_varlen_flydsl_forward_impl(
        q, k, v, cu, cu, maxs, maxs, softmax_scale=scale, causal=True, window_size=ws, return_lse=True
    )

    def r():
        flash_attn_varlen_flydsl_backward_impl(
            do, q, k, v, out, lse, cu, cu, maxs, maxs, softmax_scale=scale, causal=True, window_size=ws
        )

    ms = _time(r, _auto_it(r))
    # 5 backward GEMMs at D wide; block-diagonal, so flop sums over segments.
    flop = sum(10.0 * HQ * s * s * D * _attended_frac(s, win) for s in segs)
    return ms, flop / 1e12 / (ms / 1e3)


def main():
    print(
        f"device={torch.cuda.get_device_name(0)} arch={torch.cuda.get_device_properties(0).gcnArchName}",
        flush=True,
    )
    print(
        f"\n===== Backward ragged/block-causal hd64 THD  HQ/HKV={HQ}/{HKV}  hw-exp  [node {NODE}] =====",
        flush=True,
    )
    print(
        f"{'tag':>9} {'segs':>26} {'total':>6} | "
        f"{'full ms':>8} {'full TF':>8} | {'SWA ms':>8} {'SWA TF':>8}",
        flush=True,
    )
    for tag, segs in CONFIGS:
        cells = []
        for wl, _wr in WINDOWS:
            ms, tf = bench_case(segs, wl)
            cells.append(f"{ms:8.3f} {tf:8.0f}")
        total = sum(segs)
        print(f"{tag:>9} {str(segs):>26} {total:6d} | {cells[0]} | {cells[1]}", flush=True)
    print(
        "  note: ragged tiles by max_seqlen -> skewed layouts pay early-exit waste; TF/s is\n"
        "  effective (block-diagonal attended-fraction), not a peak-utilization figure.",
        flush=True,
    )


if __name__ == "__main__":
    main()
