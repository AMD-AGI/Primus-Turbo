"""Round-12 (gpt_oss FP8 kernel-only ceiling) probe:
num_slots lever audit on Down-B4-M4096 fwd+dgrad RCR.

This is the LAST untested cell at the 2.75 ws/CU density tier:
  * R9/R11 found slots=200/196 wins +5% on Down-B4-M2048 (1.4 ws/CU, k=2880).
  * R10 falsified slot reduction on GateUP-B4-M2048 fwd
    (2.75 ws/CU, k=2880) — slots ∈ {184..220} all LOSS -12 to -16%.
  * R10 falsified slot reduction on GateUP-B4-M4096 dA-via-H4
    (2.75 ws/CU, k=5760) — slots ∈ {184..200} all LOSS -13 to -24%.

Down-B4-M4096 fwd/dgrad sits at 2.75 ws/CU AND k=2880 — same density
as the falsified GateUP-B4-M2048-fwd, but at the SHORTER per-tile K
of 22.5 K-blocks (vs GateUP-B4-M4096-dA-via-H4 with k=5760 → 45 K-blocks).

If the falsification threshold is purely density-driven (ws/CU > 2.5),
this should also FALSIFY. If the per-tile prologue/epilogue amortisation
fraction matters — i.e. shorter K → larger fraction → lever still works —
this could WIN.

Methodology: in-process direct grouped_rcr_dscale call (R9 per-call
num_slots arg). 1500-iter × 7-trial × 5-seed p20. Mirror of R11.
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


def _bit_eq(B, M, N, K, gm, xcd, ns_a, ns_b, seed=42):
    """Check max_abs_diff between two num_slots values."""
    torch.manual_seed(seed)
    g_lens = torch.full((B,), M, dtype=torch.int64, device="cuda")
    g_offs = torch.zeros(B + 1, dtype=torch.int64, device="cuda")
    g_offs[1:] = torch.cumsum(g_lens, dim=0)
    a_bf = torch.randn((B * M, K), dtype=torch.bfloat16, device="cuda")
    b_bf = torch.randn((B, N, K), dtype=torch.bfloat16, device="cuda")
    a, sa = _quantize(a_bf)
    b, sb = _quantize(b_bf)
    out_a = torch.empty((B * M, N), dtype=torch.bfloat16, device="cuda")
    out_b = torch.empty((B * M, N), dtype=torch.bfloat16, device="cuda")
    fn(a, b, out_a, sa, sb, g_offs, gm, m_per_group=M, num_xcds=xcd, num_slots=ns_a)
    fn(a, b, out_b, sa, sb, g_offs, gm, m_per_group=M, num_xcds=xcd, num_slots=ns_b)
    diff = (out_a.float() - out_b.float()).abs().max().item()
    return diff


SHAPES = [
    # name                 B  M     N     K     gm xcd  reference
    ("Down_B4_M4096_fwd",  4, 4096, 2880, 2880,  1,  4),  # R2 cell, 2.75 ws/CU, k=2880
]
NS = [184, 188, 192, 196, 200, 208, 220, 256]
SEEDS = (42, 137, 2024, 7, 1234)

for name, B, M, N, K, gm, xcd in SHAPES:
    m_total = B * M
    flops = 2 * B * M * N * K
    ts = (M // 256) * ((N + 255) // 256) * B
    ws = ts / 256.0
    print(f"\n{'=' * 70}")
    print(f"  {name}  (B={B}, M={M}, N={N}, K={K}, m_total={m_total})")
    print(f"  cell = (gm={gm}, xcds={xcd})  ws/slot ≈ {ws:.2f}  k_iters={K // 128}.{(K % 128) * 10 // 128}")
    print(f"{'=' * 70}\n")

    print("  Bit-equivalence check (vs ns=0 default):")
    for ns in (192, 196, 200):
        d = _bit_eq(B, M, N, K, gm, xcd, 0, ns)
        print(f"    ns=0 vs ns={ns:>3}  max_abs_diff={d}")

    all_results = {}
    for ns in NS:
        all_results[ns] = {}
        for seed in SEEDS:
            med, lo, hi = _bench(B, M, N, K, gm, xcd, ns, seed)
            all_results[ns][seed] = (med, lo, hi, flops / (med * 1e9))

    print(f"\n  baseline ns=256 (NUM_CUS default):")
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
