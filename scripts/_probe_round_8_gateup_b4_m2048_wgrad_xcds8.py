"""Round-8 (gpt_oss FP8 kernel-only ceiling) tight A/B verify:
gpt_oss-GateUP-B4-M2048 wgrad var-K dB — re-test with the **xcds=8
column** that R3 (commit 0b14c83 series, Primus-Turbo this run)
and R35 (the predecessor) BOTH skipped.

Background:
  R3 (current binding's GateUP-B4-M2048 wgrad rule) selected
  (gm=1, xcds=4) over R35's (2, 2) by +0.52% with med/spread=21.7×.
  The candidate set was 6 cells: (1,4)/(2,4)/(2,2)/(4,2)/(4,4)/(1,2)
  — ALL xcds∈{2,4}. R35's predecessor 7-cell neighbor probe was
  also xcds∈{1,2,4}.

  Per the R7 lesson (commit 1d526e2 — found (gm=8, xcds=8) on
  GateUP-B4-M2048 dgrad RCR by adding the missing xcds=8 column to
  R34's xcds=4-only sweep), this var-K rule has the same
  candidate-set hole.

  R9's sibling probe on B4-M4096 var-K (m_total=16384, same N=5760
  K=2880, identical 968 tile-step / ~4 wave-step grid) tested
  xcds=8 cells:
    (1, 8)  -1.94/-2.13/-2.17 LOSS
    (8, 8)  -2.91/-2.71/-2.99 LOSS
    (16, 8) -2.95/-3.41/-3.22 LOSS
  All LOSS. But B4-M2048's K-loop depth per tile-step is half
  (M_per_g/KBLOCK = 2048/128 = 16 vs 4096/128 = 32 blocks),
  changing the per-tile compute density. Possible the M-tier
  flips the optimum (R7 just found this exact pattern on RCR side).

Anchor:
  m_total = 8192, N_fwd = 5760, K_fwd = 2880, B = 4 groups
  per-group output = [5760, 2880] ⇒ tiles_n=22, tiles_k=11
  → 242 tile-steps × 4 = 968 / 256 ≈ 3.78 wave-steps per slot

Methodology: 1500-iter × 7-trial × 3-seed p20 (mirror of R7 / R17).
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
fn = hk.grouped_variable_k_crr_dscale


def _quantize(t):
    amax = t.abs().max()
    s = (amax / 240.0).clamp(min=1e-6).to(torch.float32)
    q = (t.to(torch.float32) / s).clamp(-240.0, 240.0).to(torch.float8_e4m3fn)
    return q, s


def _build_inputs(B, M_per_g, N_fwd, K_fwd, seed):
    torch.manual_seed(seed)
    g_lens = torch.full((B,), M_per_g, dtype=torch.int64, device="cuda")
    g_offs = torch.zeros(B + 1, dtype=torch.int64, device="cuda")
    g_offs[1:] = torch.cumsum(g_lens, dim=0)
    m_total = B * M_per_g
    x_bf = torch.randn((m_total, K_fwd), dtype=torch.bfloat16, device="cuda")
    grad_out_bf = torch.randn((m_total, N_fwd), dtype=torch.bfloat16, device="cuda")
    x_q, sa = _quantize(x_bf)
    grad_q, sb = _quantize(grad_out_bf)
    out = torch.empty((B, N_fwd, K_fwd), dtype=torch.bfloat16, device="cuda")
    return grad_q, x_q, sb, sa, g_offs, out


def _bench(B, M_per_g, N_fwd, K_fwd, gm, xcd, seed, warmup=80, iters=1500, trials=7):
    grad_q, x_q, sb, sa, g_offs, out = _build_inputs(B, M_per_g, N_fwd, K_fwd, seed)

    def _call():
        fn(grad_q, x_q, out, sb, sa, g_offs,
           group_m=gm, num_xcds=xcd, num_slots=0)

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


def _bit_eq(B, M_per_g, N_fwd, K_fwd, gm_a, xcd_a, gm_b, xcd_b, seed=0):
    grad_q, x_q, sb, sa, g_offs, _ = _build_inputs(B, M_per_g, N_fwd, K_fwd, seed)
    out_a = torch.zeros((B, N_fwd, K_fwd), dtype=torch.bfloat16, device="cuda")
    out_b = torch.zeros((B, N_fwd, K_fwd), dtype=torch.bfloat16, device="cuda")
    fn(grad_q, x_q, out_a, sb, sa, g_offs, group_m=gm_a, num_xcds=xcd_a, num_slots=0)
    fn(grad_q, x_q, out_b, sb, sa, g_offs, group_m=gm_b, num_xcds=xcd_b, num_slots=0)
    torch.cuda.synchronize()
    return (out_a.float() - out_b.float()).abs().max().item()


B, M_per_g, N_fwd, K_fwd = 4, 2048, 5760, 2880
flops = 2 * B * M_per_g * N_fwd * K_fwd

print(f"========= GateUP_B4_M2048 wgrad var-K dB — round-8 xcds=8 column verify =========")
print(f"  B={B}, M_per_g={M_per_g}, N_fwd={N_fwd}, K_fwd={K_fwd}, m_total={B*M_per_g}")
print(f"  per-group output [N_fwd, K_fwd] = [{N_fwd}, {K_fwd}] (tiles_n=22, tiles_k=11)")
print(f"  Currently: R3 (gm=1, xcds=4). Re-test with xcds=8 column added.")
print(f"  ITERS=1500 × TRIALS=7 × SEEDS={{42, 137, 2024}}")
print()

# Bit-eq sanity for all candidates vs R3.
print("[bit-eq] (1, 4) [R3] vs candidate cells (max abs diff in bf16):")
for (gm, xcd) in [(1, 8), (2, 8), (4, 8), (8, 8), (16, 8), (32, 8), (1, 0)]:
    diff = _bit_eq(B, M_per_g, N_fwd, K_fwd, 1, 4, gm, xcd, seed=42)
    print(f"  (1, 4) vs ({gm:>2}, {xcd}):  max_abs_diff = {diff:.4e}")
print()

CELLS = [
    ("R3", 1, 4),  # baseline
    # xcds=8 column (the missing piece)
    ("xcds=8", 1, 8),
    ("xcds=8", 2, 8),
    ("xcds=8", 4, 8),
    ("xcds=8", 8, 8),
    ("xcds=8", 16, 8),
    ("xcds=8", 32, 8),
    # default-equivalent (xcds=0 → BSNX=8)
    ("default", 1, 0),  # gm=1 with default xcds (=8)
]

print("Per-seed bench (median ms / TF / Δ vs R3):")
all_results = {}
for label, gm, xcd in CELLS:
    all_results[(gm, xcd)] = {}
    for seed in (42, 137, 2024):
        med, lo, hi = _bench(B, M_per_g, N_fwd, K_fwd, gm=gm, xcd=xcd, seed=seed)
        tf = flops / (med * 1e9)
        all_results[(gm, xcd)][seed] = (med, lo, hi, tf)

print(f"\n  baseline R3 (gm=1, xcd=4):")
for seed in (42, 137, 2024):
    med, lo, hi, tf = all_results[(1, 4)][seed]
    print(f"    seed={seed:>4}  ms_med={med:.4f}  [{lo:.4f}, {hi:.4f}]  TF={tf:6.1f}  spread={(hi-lo)/med*100:.2f}pp")

print(f"\n  candidates vs R3:")
for label, gm, xcd in CELLS:
    if (gm, xcd) == (1, 4):
        continue
    seeds_pos = []
    for seed in (42, 137, 2024):
        med, lo, hi, tf = all_results[(gm, xcd)][seed]
        med_base, _, _, tf_base = all_results[(1, 4)][seed]
        delta = (tf - tf_base) / tf_base * 100
        seeds_pos.append(delta)
        print(f"    ({gm:>2}, {xcd}) seed={seed:>4}  ms_med={med:.4f}  TF={tf:6.1f}  Δ={delta:+.2f}pp  spread={(hi-lo)/med*100:.2f}pp")
    seed_med = statistics.median(seeds_pos)
    seed_min = min(seeds_pos)
    seed_max = max(seeds_pos)
    seed_spread = seed_max - seed_min
    all_pos = all(d > 0 for d in seeds_pos)
    if all_pos and seed_med > 0.5 and seed_spread < seed_med:
        verdict = "WIN-ROBUST"
    elif seed_med > 0 and all_pos:
        verdict = "WIN-LIGHT"
    elif abs(seed_med) < 0.5:
        verdict = "TIE"
    else:
        verdict = "LOSS"
    print(f"    ({gm:>2}, {xcd}) {label:>15}  seed-med Δ={seed_med:+.2f}pp  per-seed-spread={seed_spread:.2f}pp  all_pos={all_pos}  verdict={verdict}")
    print()

print()
print("=========================================================")
print("Verdict gate: WIN-ROBUST = every-seed Δ > 0 AND seed-med Δ > 0.5pp")
print("              AND seed-spread < seed-med (signal > noise).")
print("Anything else = TIE / LOSS / FALSIFIED.")
