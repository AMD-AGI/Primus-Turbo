"""Round-8 (gpt_oss FP8 kernel-only ceiling) tight A/B verify:
GateUP_B4_M4096 dgrad (RCR via H4) — confirm full xcds=8 column on
the current binding. R8's comment only documents (8, 8) as -1.80%;
R7 found xcds=8 wins on the M=2048 sibling so the M=4096 boundary
deserves a tight verify with the full column.

Background:
  R8 set (gm=1, xcds=4) for this shape. The R8 probe tested at least
  (8, 8) which lost -1.80%, but the FULL xcds=8 column (gm ∈ {1, 4,
  16, 32} at xcds=8) was not exhaustively documented. Per the R7
  pattern (m_total=8192 → xcds=8 wins for dgrad), m_total=16384 sits
  in between — maybe there's a borderline cell.

Anchor:
  m=4096, n=K_fwd=2880, k=N_fwd=5760, m_total=16384,
  tiles_m=16, tiles_n=11, k=5760
  → 176 tile-steps × 4 groups = 704 / 256 ≈ 2.75 wave-steps per slot

Methodology: 1500-iter × 7-trial × 3-seed p20 (mirror of R7).
"""
from __future__ import annotations

import os
import statistics
import sys

os.environ.setdefault("PRIMUS_TURBO_HIPKITTEN_PATH", "/workspace/code/HipKittens")
sys.path.insert(0, "/workspace/code/Primus-Turbo")

import torch
from primus_turbo.pytorch.kernels.hipkitten import loader as _hk_loader
from primus_turbo.triton.utils.fp8_transpose import fp8_transpose_3d  # noqa: E402

hk = _hk_loader.load_fp8()
fn = hk.grouped_rcr_dscale


def _quantize(t):
    amax = t.abs().max()
    s = (amax / 240.0).clamp(min=1e-6).to(torch.float32)
    q = (t.to(torch.float32) / s).clamp(-240.0, 240.0).to(torch.float8_e4m3fn)
    return q, s


def _bench(B, M, N_fwd, K_fwd, gm, xcd, seed, warmup=80, iters=1500, trials=7):
    torch.manual_seed(seed)
    g_lens = torch.full((B,), M, dtype=torch.int64, device="cuda")
    g_offs = torch.zeros(B + 1, dtype=torch.int64, device="cuda")
    g_offs[1:] = torch.cumsum(g_lens, dim=0)
    grad_bf = torch.randn((B * M, N_fwd), dtype=torch.bfloat16, device="cuda")
    b_orig_bf = torch.randn((B, N_fwd, K_fwd), dtype=torch.bfloat16, device="cuda")
    a, sa = _quantize(grad_bf)
    b_orig, sb = _quantize(b_orig_bf)
    b_t = fp8_transpose_3d(b_orig)
    out = torch.empty((B * M, K_fwd), dtype=torch.bfloat16, device="cuda")

    def _call():
        fn(a, b_t, out, sa, sb, g_offs, gm, m_per_group=M, num_xcds=xcd)

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


B, M, N_fwd, K_fwd = 4, 4096, 5760, 2880
flops = 2 * B * M * N_fwd * K_fwd

print(f"========= GateUP_B4_M4096 dgrad (RCR via H4) — round-8 xcds=8 column verify =========")
print(f"  m={M}, n=K_fwd={K_fwd}, k=N_fwd={N_fwd}, m_total={B*M}")
print(f"  tiles_m={M//256}, tiles_n={K_fwd//256}, k={N_fwd}")
print(f"  Currently: R8 (gm=1, xcd=4). Re-test with xcds=8 column added.")
print()

CELLS = [
    ("R8", 1, 4),  # baseline
    # xcds=8 column — was R8's only cell tested explicitly?
    ("xcds=8", 1, 8),
    ("xcds=8", 4, 8),
    ("xcds=8", 8, 8),  # R8 documented as -1.80%
    ("xcds=8", 16, 8),
    ("xcds=8", 32, 8),
    # default (xcds=0 → BSNX=8)
    ("default", 4, 0),
]

print("Per-seed bench (median ms / TF / Δ vs R8):")
all_results = {}
for label, gm, xcd in CELLS:
    all_results[(gm, xcd)] = {}
    for seed in (42, 137, 2024):
        med, lo, hi = _bench(B, M, N_fwd, K_fwd, gm=gm, xcd=xcd, seed=seed)
        tf = flops / (med * 1e9)
        all_results[(gm, xcd)][seed] = (med, lo, hi, tf)

print(f"\n  baseline R8 (gm=1, xcd=4):")
for seed in (42, 137, 2024):
    med, lo, hi, tf = all_results[(1, 4)][seed]
    print(f"    seed={seed:>4}  ms_med={med:.4f}  [{lo:.4f}, {hi:.4f}]  TF={tf:6.1f}  spread={(hi-lo)/med*100:.2f}pp")

print(f"\n  candidates vs R8:")
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
