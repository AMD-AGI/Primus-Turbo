"""Round-11 (gpt_oss FP8 kernel-only ceiling) tight verify:
two num_slots-lever extensions on top of R9/R10.

(A) Down-B4-M2048 fwd+dA RCR finer scan around slots=192.
    R9 shipped slots=200 (+4.99% kernel; never tested slots=192).
    The R3 var-K wgrad rule for the same Down-B4 family uses
    slots=192 so it's plausible that 192 also wins on the RCR
    persistent grid. Test slots ∈ {184, 188, 192, 196, 200, 256}
    on the R2 dispatcher cell (gm=16, xcds=2) at 1.4 ws/CU.

(B) GateUP-B4-M4096 dA-via-H4 RCR. R8 cell (gm=1, xcds=4),
    tiles_m=16 m_total=16384 → 16×11×4 = 704 tile-steps / 256 ≈
    2.75 ws/CU. Same density as the R10-FALSIFIED GateUP-B4-M2048
    fwd cell. Predicted FALSIFIED but systematic verify with the
    same slot grid.

Methodology: in-process direct grouped_rcr_dscale call (R9 per-call
num_slots arg). 1500-iter × 7-trial × 5-seed p20.
"""
from __future__ import annotations

import os
import statistics
import sys

os.environ.setdefault("PRIMUS_TURBO_HIPKITTEN_PATH", "/workspace/code/HipKittens")
sys.path.insert(0, "/workspace/code/Primus-Turbo")

import torch
from primus_turbo.pytorch.kernels.hipkitten import loader as _hk_loader

hk = _hk_loader.load_fp8()
fn = hk.grouped_rcr_dscale


def _quantize(t):
    amax = t.abs().max()
    s = (amax / 240.0).clamp(min=1e-6).to(torch.float32)
    q = (t.to(torch.float32) / s).clamp(-240.0, 240.0).to(torch.float8_e4m3fn)
    return q, s


def _bench(B, M, N, K, gm, xcd, num_slots, seed,
           warmup=80, iters=1500, trials=7):
    torch.manual_seed(seed)
    g_lens = torch.full((B,), M, dtype=torch.int64, device="cuda")
    g_offs = torch.zeros(B + 1, dtype=torch.int64, device="cuda")
    g_offs[1:] = torch.cumsum(g_lens, dim=0)
    a_bf = torch.randn((B * M, K), dtype=torch.bfloat16, device="cuda")
    b_bf = torch.randn((B, N, K), dtype=torch.bfloat16, device="cuda")
    a, sa = _quantize(a_bf)
    b, sb = _quantize(b_bf)
    out = torch.empty((B * M, N), dtype=torch.bfloat16, device="cuda")

    def _call():
        fn(a, b, out, sa, sb, g_offs, gm, m_per_group=M, num_xcds=xcd,
           num_slots=num_slots)

    for _ in range(warmup):
        _call()
    torch.cuda.synchronize()
    p20s = []
    for _ in range(trials):
        se = torch.cuda.Event(enable_timing=True)
        ee = torch.cuda.Event(enable_timing=True)
        times = []
        for _ in range(iters):
            torch.cuda.synchronize()
            se.record()
            _call()
            ee.record()
            torch.cuda.synchronize()
            times.append(se.elapsed_time(ee))
        times.sort()
        p20s.append(times[len(times) // 5])
    return statistics.median(p20s), min(p20s), max(p20s)


SHAPES = [
    # name                  B  M     N     K     gm  xcd  reference
    ("Down_B4_M2048_fwd",   4, 2048, 2880, 2880, 16, 2),    # R9 cell, 1.4 ws/CU; R9 baseline=200
    ("GateUP_B4_M4096_dA",  4, 4096, 2880, 5760, 1,  4),    # R8 cell, 2.75 ws/CU; predict FALSE
]
NS = [184, 188, 192, 196, 200, 256]
SEEDS = (42, 137, 2024, 7, 1234)

for name, B, M, N, K, gm, xcd in SHAPES:
    m_total = B * M
    flops = 2 * B * M * N * K
    ts = (M // 256) * ((N + 255) // 256) * B
    ws = ts / 256.0
    print(f"\n{'=' * 70}")
    print(f"  {name}  (B={B}, M={M}, N={N}, K={K}, m_total={m_total})")
    print(f"  cell = (gm={gm}, xcds={xcd})  ws/slot ≈ {ws:.2f}")
    print(f"{'=' * 70}\n")

    all_results = {}
    for ns in NS:
        all_results[ns] = {}
        for seed in SEEDS:
            med, lo, hi = _bench(B, M, N, K, gm, xcd, ns, seed)
            all_results[ns][seed] = (med, lo, hi, flops / (med * 1e9))

    print(f"  baseline ns=256 (NUM_CUS default):")
    for seed in SEEDS:
        med, lo, hi, tf = all_results[256][seed]
        print(f"    seed={seed:>4}  TF={tf:6.1f}  ms_med={med:.4f}")

    print(f"\n  candidates vs ns=256:")
    for ns in NS:
        if ns == 256:
            continue
        seeds_pos = []
        for seed in SEEDS:
            tf = all_results[ns][seed][3]
            tf_b = all_results[256][seed][3]
            delta = (tf - tf_b) / tf_b * 100
            seeds_pos.append(delta)
        seed_med = statistics.median(seeds_pos)
        seed_min = min(seeds_pos)
        seed_max = max(seeds_pos)
        seed_spread = seed_max - seed_min
        all_pos = all(d > 0 for d in seeds_pos)
        pos_count = sum(1 for d in seeds_pos if d > 0)
        per_seed_str = " ".join(f"{d:+.2f}" for d in seeds_pos)
        if all_pos and seed_med > 0.5 and seed_spread < seed_med:
            verdict = "WIN-ROBUST"
        elif seed_med > 0 and all_pos:
            verdict = "WIN-LIGHT"
        elif abs(seed_med) < 0.5:
            verdict = "TIE"
        else:
            verdict = "LOSS"
        print(f"    ns={ns:>3}  med Δ={seed_med:+.3f}pp  spread={seed_spread:.2f}pp  pos={pos_count}/{len(SEEDS)}  per-seed=[{per_seed_str}]  {verdict}")
