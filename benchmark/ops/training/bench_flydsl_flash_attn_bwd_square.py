#!/usr/bin/env python3
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# Square-causal acceptance for the integrated FlyDSL hd64 flash-attention BACKWARD
# (primus_turbo.flydsl.attention.flash_attn_bwd), B=1, Sq=Skv=S, Hq=128 Hkv=16 D=64.
# Prints both exp paths: hw (52.6 dB, fast_exp2=False) and fast (35 dB, production
# default, fast_exp2=True), in the same acceptance format as the forward bench.
#
# conv TF/s = 10*B*Hq*S*S*D*(1-(S-1)/(2S)) / full_bwd_ms(odo+dq+dkdv)  (5 bwd GEMMs).
# MI350-equivalent = MI355 convTF / 1.2 ; xH100 = MI350 / H100_bwd ; PASS if >= 1.4x.
# Hot-steady timing (continuous warmup + median).
#
#   HIP_VISIBLE_DEVICES=0 python3 bench_flydsl_flash_attn_bwd_square.py
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
HQ, HKV = 128, 16
SQUARE_S = [2048, 4096, 8192, 16384]
H100_SQ = {2048: 226, 4096: 377, 8192: 466, 16384: 508}  # H100 FA-v3 bwd square, TF/s
B = int(os.environ.get("B", "1"))
WARMS = float(os.environ.get("WARMS", "3.0"))
REPS = int(os.environ.get("REPS", "7"))
IT = int(os.environ.get("IT", "50"))
NODE = os.environ.get("NODE", os.uname().nodename)


def conv_flop(S):
    return 10.0 * B * HQ * S * S * D * (1.0 - (S - 1) / (2.0 * S))


def _time(fn):
    t0 = time.time()
    while time.time() - t0 < WARMS:
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


def bench_one(S, fast):
    scale = 1.0 / math.sqrt(D)
    qsp = _qsplit_for(S)
    bkv = _blockkv_for(S)
    common = dict(
        num_heads=HQ,
        head_dim=D,
        causal=True,
        dtype_str="bf16",
        sm_scale=scale,
        num_kv_heads=HKV,
        window_left=-1,
    )
    dq_l = build_flash_attn_bwd_dq_module(fast_exp2=fast, **common)
    dkdv_l = build_flash_attn_bwd_dkdv_module(q_split=qsp, fast_exp2=fast, block_kv=bkv, **common)
    odo_l = build_flash_attn_bwd_odo_module(num_heads=HQ, head_dim=D, num_kv_heads=HKV, sm_scale=scale)
    st = torch.cuda.current_stream()
    torch.manual_seed(0)
    q = torch.randn(B * S, HQ, D, device=DEV, dtype=DT)
    k = torch.randn(B * S, HKV, D, device=DEV, dtype=DT)
    v = torch.randn(B * S, HKV, D, device=DEV, dtype=DT)
    do = torch.randn(B * S, HQ, D, device=DEV, dtype=DT)
    out = torch.randn(B * S, HQ, D, device=DEV, dtype=DT)
    lse = torch.randn(B, HQ, S, device=DEV, dtype=torch.float32)
    delta = torch.empty(B, HQ, S, device=DEV, dtype=torch.float32)
    lse_s = _prescale_lse(lse, fast).reshape(-1)
    dq = torch.empty_like(q)
    k16 = torch.empty(1, device=DEV, dtype=DT)
    wk = torch.empty(B, qsp, S, HKV, D, device=DEV, dtype=DT)
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
        odo_l(of, dof, df, B, S, st)
        dq_l(qf, kf, vf, dof, lse_s, df, dq.reshape(-1), k16, B, S, S, st)
        dkdv_l(qf, kf, vf, dof, lse_s, df, wk.reshape(-1), wv.reshape(-1), B, S, S, st)

    ms = _time(r_all)
    return conv_flop(S) / 1e12 / (ms / 1e3)


def run_table(fast):
    tag = "fast(35dB, production default)" if fast else "hw(52.6dB)"
    print(f"\n===== Backward hd64 THD  B={B}  square-causal  {tag}  [node {NODE}] =====", flush=True)
    print(
        f"{'S':>6} {'H100_bwd':>9} {'1.4xtgt(MI350)':>15} {'MI355':>7} {'MI350(/1.2)':>12} {'xH100':>6} {'verdict':>8}",
        flush=True,
    )
    npass = 0
    for S in SQUARE_S:
        tf = bench_one(S, fast)
        h = H100_SQ[S]
        tgt = h * 1.40
        mi350 = tf / 1.2
        xh = mi350 / h
        if xh >= 1.4:
            verdict = "PASS"
            npass += 1
        else:
            verdict = f"{(xh / 1.4 - 1) * 100:+.1f}%"
        print(f"{S:6d} {h:9d} {tgt:15.0f} {tf:7.0f} {mi350:12.0f} {xh:6.2f} {verdict:>8}", flush=True)
    print(f"  -> {npass}/4 clear 1.4x H100 (MI350-equivalent)", flush=True)


def main():
    print(
        f"device={torch.cuda.get_device_name(0)} arch={torch.cuda.get_device_properties(0).gcnArchName}",
        flush=True,
    )
    for fast in (False, True):
        run_table(fast)


if __name__ == "__main__":
    main()
