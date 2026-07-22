#!/usr/bin/env python3
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# Perf reproduction for the integrated FlyDSL hd64 flash-attention BACKWARD
# (primus_turbo.flydsl.attention.flash_attn_bwd), Meta 20-config acceptance grid.
#
# Reproduces the meta_aiter_attn bwd table: B=4, hw (52 dB, fast_exp2=False) and
# fast (35 dB, fast_exp2=True), full-causal + causal-SWA, MI355X (gfx950).
# ratio = H100_bestdet_ms / (MI355_ms * 1.2)   (1.2 = MI355->MI350 clock derate);
# the >=1.4x line certifies MI350 would clear 1.4x H100 FA-v3 bwd.
#
# Hot-steady protocol (matches meta _bench_all): WARMS s continuous full-load
# warmup (no sleep) + median of REPS. Times odo + dq + dkdv (the full bwd).
#
#   HIP_VISIBLE_DEVICES=0 python3 _bench_flydsl_flash_bwd.py
import math
import os
import time

import torch

from primus_turbo.flydsl.attention.flash_attn_bwd import (
    _blockkv_for,
    _prescale_lse,
    _qsplit_for,
    build_flash_attn_bwd_dkdv_module,
    build_flash_attn_bwd_dq_module,
    build_flash_attn_bwd_odo_module,
)

DEV = "cuda"
DT = torch.bfloat16
D = 64
MI = 1.2  # MI355 -> MI350 clock derate

# label, Hq, Hkv, Sq, Skv, W, H100_full_ms, H100_swa_ms  (H100 best-det bwd refs)
SHAPES = [
    ("Hq128 Sq2048  Skv16384", 128, 16, 2048, 16384, 2048, 24.78, 7.88),
    ("Hq128 Sq4096  Skv16384", 128, 16, 4096, 16384, 2048, 41.37, 14.04),
    ("Hq128 Sq8192  Skv16384", 128, 16, 8192, 16384, 2048, 66.79, 26.50),
    ("Hq128 Sq16384 Skv16384", 128, 16, 16384, 16384, 2048, 86.79, 48.39),
    ("Hq48  Sq4096  Skv4096 ", 48, 6, 4096, 4096, 2047, 2.65, 4.62),
    ("Hq48  Sq4096  Skv8192 ", 48, 6, 4096, 8192, 2047, 7.17, 6.11),
    ("Hq48  Sq4096  Skv12288", 48, 6, 4096, 12288, 2047, 11.64, 6.27),
    ("Hq48  Sq4096  Skv16384", 48, 6, 4096, 16384, 2047, 15.14, 6.44),
    ("Hq64  Sq1024  Skv1024 ", 64, 8, 1024, 1024, 2047, 0.52, 0.49),
    ("Hq64  Sq1024  Skv16384", 64, 8, 1024, 16384, 2047, 7.51, 2.38),
]

B = int(os.environ.get("B", "4"))
WARMS = float(os.environ.get("WARMS", "3.0"))
REPS = int(os.environ.get("REPS", "7"))
IT = int(os.environ.get("IT", "50"))
NODE = os.environ.get("NODE", os.uname().nodename)


def _time(fn):
    t0 = time.time()
    while time.time() - t0 < WARMS:  # continuous full-load warmup, no sleep
        for _ in range(30):
            fn()
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


def bench_one(Hq, Hkv, Sq, Skv, wl, fast):
    scale = 1.0 / math.sqrt(D)
    qsp = _qsplit_for(Sq)
    bkv = _blockkv_for(Sq)
    common = dict(
        num_heads=Hq,
        head_dim=D,
        causal=True,
        dtype_str="bf16",
        sm_scale=scale,
        num_kv_heads=Hkv,
        window_left=wl,
    )
    dq_l = build_flash_attn_bwd_dq_module(fast_exp2=fast, **common)
    dkdv_l = build_flash_attn_bwd_dkdv_module(q_split=qsp, fast_exp2=fast, block_kv=bkv, **common)
    odo_l = build_flash_attn_bwd_odo_module(num_heads=Hq, head_dim=D, num_kv_heads=Hkv, sm_scale=scale)
    st = torch.cuda.current_stream()
    torch.manual_seed(0)
    q = torch.randn(B * Sq, Hq, D, device=DEV, dtype=DT)
    k = torch.randn(B * Skv, Hkv, D, device=DEV, dtype=DT)
    v = torch.randn(B * Skv, Hkv, D, device=DEV, dtype=DT)
    do = torch.randn(B * Sq, Hq, D, device=DEV, dtype=DT)
    out = torch.randn(B * Sq, Hq, D, device=DEV, dtype=DT)
    lse = torch.randn(B, Hq, Sq, device=DEV, dtype=torch.float32)
    delta = torch.empty(B, Hq, Sq, device=DEV, dtype=torch.float32)
    lse_s = _prescale_lse(lse, fast).reshape(-1)
    dq = torch.empty_like(q)
    k16 = torch.empty(1, device=DEV, dtype=DT)
    wk = torch.empty(B, qsp, Skv, Hkv, D, device=DEV, dtype=DT)
    wv = torch.empty_like(wk)
    qf, kf, vf, dof, df, of = (
        q.reshape(-1),
        k.reshape(-1),
        v.reshape(-1),
        do.reshape(-1),
        delta.reshape(-1),
        out.reshape(-1),
    )

    def r_all():
        odo_l(of, dof, df, B, Sq, st)
        dq_l(qf, kf, vf, dof, lse_s, df, dq.reshape(-1), k16, B, Sq, Skv, st)
        dkdv_l(qf, kf, vf, dof, lse_s, df, wk.reshape(-1), wv.reshape(-1), B, Sq, Skv, st)

    return _time(r_all)


def run_table(fast):
    tag = "fast(35dB)" if fast else "hw(52dB)"
    print(f"\n===== {tag}  B={B}  ratio=H100_bestdet_ms/(MI355_ms*1.2)  [node {NODE}] =====", flush=True)
    print(f"{'shape':22s} {'mode':4s} {'ms':>7} {'H100ms':>7} {'ratio':>6} {'1.4x':>4}", flush=True)
    npass = 0
    full_r, swa_r = [], []
    for label, Hq, Hkv, Sq, Skv, W, h_full, h_swa in SHAPES:
        for mode, href in (("full", h_full), ("swa", h_swa)):
            wl = W if mode == "swa" else -1
            ms = bench_one(Hq, Hkv, Sq, Skv, wl, fast)
            ratio = href / (ms * MI)
            ok = ratio >= 1.4
            npass += ok
            (full_r if mode == "full" else swa_r).append(ratio)
            print(
                f"{label:22s} {mode:4s} {ms:7.2f} {href:7.2f} {ratio:6.2f} {'YES' if ok else 'no':>4}",
                flush=True,
            )
    print(
        f"  -> {npass}/20 clear 1.4x | full {min(full_r):.2f}-{max(full_r):.2f} | "
        f"swa {min(swa_r):.2f}-{max(swa_r):.2f}",
        flush=True,
    )


def main():
    print(
        f"device={torch.cuda.get_device_name(0)} arch={torch.cuda.get_device_properties(0).gcnArchName}",
        flush=True,
    )
    for fast in (False, True):
        run_table(fast)


if __name__ == "__main__":
    main()
