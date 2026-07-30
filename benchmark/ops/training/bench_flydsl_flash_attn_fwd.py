#!/usr/bin/env python3
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# Square-causal forward perf for primus_turbo.flydsl.attention.flash_attn_fwd,
# hot-steady (continuous warmup, no sleep) against the H100 FA-v3 reference.
#
#   HIP_VISIBLE_DEVICES=0 python3 bench_flydsl_flash_attn_fwd.py
import os
import time

import torch

from primus_turbo.flydsl.attention.flash_attn_fwd import build_flash_attn_dualwave_swp_module

DEV = "cuda"
DT = torch.bfloat16
D = 64

HQ, HKV = 128, 16
SQUARE_S = [2048, 4096, 8192, 16384]
# H100 FA-v3 square-causal fwd reference (B=1, D=64, TF/s).
H100_SQ = {2048: 298, 4096: 445, 8192: 486, 16384: 522}
B = 1
WARMS = 3.0
REPS = 9
IT = 20
NODE = os.uname().nodename


def _causal_frac(Sq, Skv):
    return 1.0 - (Sq - 1) / (2.0 * Skv)


def cu_uniform(nb, S, device):
    return torch.arange(0, (nb + 1) * S, S, device=device, dtype=torch.int32)


def _time(fn):
    t0 = time.time()
    n = 0
    while time.time() - t0 < WARMS:  # continuous full-load warmup, no sleep
        fn()
        n += 1
        if n % 40 == 0:
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
    mod = build_flash_attn_dualwave_swp_module(
        num_heads=HQ,
        head_dim=D,
        causal=True,
        dtype_str="bf16",
        num_kv_heads=HKV,
        varlen=True,
        cross_seqlen=False,  # square Sq==Skv
        waves_per_eu=2,
        dualwave_swp_setprio=True,
        dualwave_swp_enable_stagger=False,  # stagger OFF
        dualwave_swp_lazy_rescale=True,
        block_m=128,  # 4-wave
    )
    st = torch.cuda.current_stream()
    torch.manual_seed(0)
    q = torch.randn(B * S, HQ, D, device=DEV, dtype=DT)
    k = torch.randn(B * S, HKV, D, device=DEV, dtype=DT)
    v = torch.randn(B * S, HKV, D, device=DEV, dtype=DT)
    out = torch.empty_like(q)
    cu_q = cu_uniform(B, S, DEV)
    cu_k = cu_uniform(B, S, DEV)

    def r():
        mod(q, k, v, out, B, S, seq_len_kv=S, cu_seqlens_q=cu_q, cu_seqlens_kv=cu_k, stream=st)

    ms = _time(r)
    flop = 4.0 * B * HQ * S * S * D * _causal_frac(S, S)
    tf = flop / 1e12 / (ms / 1e3)
    return ms, tf


def main():
    print(
        f"device={torch.cuda.get_device_name(0)} arch={torch.cuda.get_device_properties(0).gcnArchName}",
        flush=True,
    )
    print(f"\n===== Forward hd64 THD  B={B}  4-wave stagger-off  hw-exp  [node {NODE}] =====", flush=True)
    print(
        f"{'S':>6} {'H100_fwd':>9} {'1.4xtgt(MI350)':>15} {'MI355':>7} {'MI350(/1.2)':>12} {'xH100':>6} {'verdict':>8}",
        flush=True,
    )
    npass = 0
    for S in SQUARE_S:
        ms, tf = bench_one(S)
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
