"""Round-10 (gpt_oss FP8 kernel-only ceiling) tight A/B verify:
GateUP_B4_M2048 fwd + dgrad-via-H4 — num_slots lever sweep.

Background:
  R9 (commit b00082d) added the per-call ``num_slots`` arg to the FP8
  grouped RCR binding (mirror of var-K's R3 num_slots wiring) and
  shipped Down-B4-M2048 fwd+dgrad rule with num_slots=200, netting
  +5% kernel TFLOPS on that one cell (1.4 ws/CU sparsity).

  R9's next-round suggestion #1 was to extend the audit to the GateUP
  family. Two shapes share m_total=8192 with Down-B4-M2048:

    * GateUP_B4_M2048 fwd RCR  (R23 cell: gm=1, xcds=4)
        per-group: 8 (tiles_m) × 22 (tiles_n) = 176 tile-steps
        × 4 groups = 704 tile-steps over 256 CUs ≈ 2.75 ws/CU
        (2× denser than Down-B4-M2048 fwd; K-loop 2880 same)

    * GateUP_B4_M2048 dgrad-via-H4 RCR  (R7 cell: gm=8, xcds=8/None)
        per-group: 8 (tiles_m) × 11 (tiles_n=K_fwd/256) = 88 tile-steps
        × 4 groups = 352 tile-steps over 256 CUs ≈ 1.4 ws/CU
        (IDENTICAL sparsity to Down-B4-M2048 fwd; K-loop 5760 = 2×
         deeper since K_dgrad = N_fwd = 5760)

  The dgrad shape is the cleanest test of whether the R9 num_slots lever
  generalises to the GateUP family at the same sparsity. The fwd shape
  tests whether the lever fires at 2× higher density.

Methodology: in-process direct ``grouped_rcr_dscale`` call (R9 lever
allows per-call num_slots arg, no env / no subprocess needed).
1500-iter × 7-trial × 5-seed p20 (mirror of R7/R8/R9 tight-verify).

Candidate space: slots ∈ {196, 200, 208, 220, 256} — the R9-anchored
peaks (200/208/220) plus 196 (XCD-aligned 8×24.5) and 256 baseline.
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
    torch.manual_seed(seed)
    g_lens = torch.full((B,), M, dtype=torch.int64, device="cuda")
    g_offs = torch.zeros(B + 1, dtype=torch.int64, device="cuda")
    g_offs[1:] = torch.cumsum(g_lens, dim=0)
    a_bf = torch.randn((B * M, K), dtype=torch.bfloat16, device="cuda")
    b_bf = torch.randn((B, N, K), dtype=torch.bfloat16, device="cuda")
    a, sa = _quantize(a_bf)
    b, sb = _quantize(b_bf)
    out_a = torch.zeros((B * M, N), dtype=torch.bfloat16, device="cuda")
    out_b = torch.zeros((B * M, N), dtype=torch.bfloat16, device="cuda")
    fn(a, b, out_a, sa, sb, g_offs, gm, m_per_group=M, num_xcds=xcd, num_slots=ns_a)
    fn(a, b, out_b, sa, sb, g_offs, gm, m_per_group=M, num_xcds=xcd, num_slots=ns_b)
    torch.cuda.synchronize()
    return (out_a.float() - out_b.float()).abs().max().item()


# Two shapes — both gpt_oss-GateUP-B4-M2048, m_total=8192 (small-grid).
SHAPES = [
    # name                   B  M    N     K     gm  xcd_or_0
    ("GateUP_B4_M2048_fwd",  4, 2048, 5760, 2880, 1,  4),       # R23 cell, 2.75 ws/CU
    ("GateUP_B4_M2048_dgrad",4, 2048, 2880, 5760, 8,  0),       # R7 cell (xcds=None=0=default 8), 1.4 ws/CU
]
NUM_SLOTS_CANDIDATES = [196, 200, 208, 220, 256]
SEEDS = (42, 137, 2024, 7, 1234)

for name, B, M, N, K, gm, xcd in SHAPES:
    m_total = B * M
    flops = 2 * B * M * N * K
    tile_steps = (M // 256) * ((N + 255) // 256) * B
    ws_per_slot = tile_steps / 256.0
    print(f"\n{'=' * 70}")
    print(f"  {name}  (B={B}, M={M}, N={N}, K={K}, m_total={m_total})")
    print(f"  cell = (gm={gm}, xcds={xcd})  [current rule]")
    print(f"  tile-steps = {tile_steps}, ws/slot ≈ {ws_per_slot:.2f}")
    print(f"{'=' * 70}")

    print(f"\n[bit-eq] num_slots=0 (default) vs candidates:")
    for ns in NUM_SLOTS_CANDIDATES:
        if ns == 256:
            continue
        diff = _bit_eq(B, M, N, K, gm, xcd, 0, ns, seed=42)
        print(f"  ns=0 vs ns={ns:>3}:  max_abs_diff = {diff:.4e}")

    print(f"\nPer-seed bench (1500-iter × 7-trial p20):")
    all_results = {}
    for ns in NUM_SLOTS_CANDIDATES:
        all_results[ns] = {}
        for seed in SEEDS:
            med, lo, hi = _bench(B, M, N, K, gm, xcd, ns, seed)
            tf = flops / (med * 1e9)
            all_results[ns][seed] = (med, lo, hi, tf)

    print(f"\n  baseline ns=256 (legacy NUM_CUS default):")
    for seed in SEEDS:
        med, lo, hi, tf = all_results[256][seed]
        print(f"    seed={seed:>4}  ms_med={med:.4f}  TF={tf:6.1f}  spread={(hi-lo)/med*100:.2f}pp")

    print(f"\n  candidates vs ns=256:")
    for ns in NUM_SLOTS_CANDIDATES:
        if ns == 256:
            continue
        seeds_pos = []
        for seed in SEEDS:
            med, lo, hi, tf = all_results[ns][seed]
            med_b, _, _, tf_b = all_results[256][seed]
            delta = (tf - tf_b) / tf_b * 100
            seeds_pos.append(delta)
            print(f"    ns={ns:>3}  seed={seed:>4}  TF={tf:6.1f}  Δ={delta:+.3f}pp  spread={(hi-lo)/med*100:.2f}pp")
        seed_med = statistics.median(seeds_pos)
        seed_min = min(seeds_pos)
        seed_max = max(seeds_pos)
        seed_spread = seed_max - seed_min
        all_pos = all(d > 0 for d in seeds_pos)
        pos_count = sum(1 for d in seeds_pos if d > 0)
        if all_pos and seed_med > 0.5 and seed_spread < seed_med:
            verdict = "WIN-ROBUST"
        elif seed_med > 0 and all_pos:
            verdict = "WIN-LIGHT"
        elif abs(seed_med) < 0.5:
            verdict = "TIE"
        else:
            verdict = "LOSS"
        print(f"    ns={ns:>3}  seed-med Δ={seed_med:+.3f}pp  spread={seed_spread:.2f}pp  pos={pos_count}/{len(SEEDS)}  verdict={verdict}")
        print()

print()
print("=" * 70)
print("Verdict gate: WIN-ROBUST = every-seed Δ > 0 AND seed-med > 0.5pp")
print("              AND seed-spread < seed-med.")
