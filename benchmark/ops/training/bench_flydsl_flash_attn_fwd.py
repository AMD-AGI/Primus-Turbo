#!/usr/bin/env python3
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# Forward perf for primus_turbo.flydsl.attention.flash_attn_fwd, hot-steady
# (continuous warmup, no sleep). Two tables: the square-causal acceptance shapes
# against the H100 FA-v3 TF/s reference, and Meta's 20 configs (10 shapes x
# {full-causal, SWA}) against the measured H100 FA-3 wall time.
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

# Meta's 20 configs: 10 shapes x {full-causal, SWA(W)}, B=4, most rectangular.
# H100 = FA-3 forward wall time, ms, measured on the same shapes (see the aiter
# reproducer's landscape table). MI350-equivalent = MI355 x 1.2.
META_B = 4
META = [
    #  Hq  Hkv     Sq    Skv     W   H100 full  H100 SWA
    (128, 16, 2048, 16384, 2048, 8.39, 1.24),
    (128, 16, 4096, 16384, 2048, 15.80, 2.51),
    (128, 16, 8192, 16384, 2048, 26.72, 5.09),
    (128, 16, 16384, 16384, 2048, 35.69, 9.55),
    (48, 6, 4096, 4096, 2047, 0.872, 0.739),
    (48, 6, 4096, 8192, 2047, 2.503, 0.923),
    (48, 6, 4096, 12288, 2047, 4.197, 0.924),
    (48, 6, 4096, 16384, 2047, 5.645, 0.928),
    (64, 8, 1024, 1024, 2047, 0.144, 0.120),
    (64, 8, 1024, 16384, 2047, 2.152, 0.329),
]


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


def _auto_it(fn):
    """One timed call sets the iteration count so every block lands near 40 ms."""
    torch.cuda.synchronize()
    t = time.time()
    fn()
    torch.cuda.synchronize()
    return max(3, min(50, int(40.0 / max((time.time() - t) * 1e3, 0.05))))


def bench_meta(Hq, Hkv, Sq, Skv, W):
    mod = build_flash_attn_dualwave_swp_module(
        num_heads=Hq,
        head_dim=D,
        causal=True,
        dtype_str="bf16",
        num_kv_heads=Hkv,
        varlen=True,
        cross_seqlen=Sq != Skv,
        waves_per_eu=2,
        dualwave_swp_setprio=True,
        dualwave_swp_enable_stagger=False,
        dualwave_swp_lazy_rescale=True,
        window_left=W,
        block_m=128,
    )
    st = torch.cuda.current_stream()
    torch.manual_seed(0)
    q = torch.randn(META_B * Sq, Hq, D, device=DEV, dtype=DT)
    k = torch.randn(META_B * Skv, Hkv, D, device=DEV, dtype=DT)
    v = torch.randn(META_B * Skv, Hkv, D, device=DEV, dtype=DT)
    out = torch.empty_like(q)
    cu_q = cu_uniform(META_B, Sq, DEV)
    cu_k = cu_uniform(META_B, Skv, DEV)

    def r():
        mod(q, k, v, out, META_B, Sq, seq_len_kv=Skv, cu_seqlens_q=cu_q, cu_seqlens_kv=cu_k, stream=st)

    global IT
    keep, IT = IT, _auto_it(r)
    ms = _time(r)
    IT = keep
    return ms


def meta20():
    print(f"\n===== Forward, Meta 20 configs  B={META_B}  vs H100 FA-3 (ms) =====", flush=True)
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
    print(f"  -> {npass}/20 clear 1.4x H100 FA-3 (MI350-equivalent)", flush=True)


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
    meta20()


if __name__ == "__main__":
    main()
