"""Round-9 (gpt_oss FP8 kernel-only ceiling) tight A/B verify:
GateUP wgrad var-K dB — num_slots lever sweep across the 3 GateUP
shapes that R2 (commit 2d3946f1 series, current Primus run, 2026-05-07)
analytically excluded but never tight-verified.

Background:
  R2 wired ``num_slots=192`` for var-K wgrad on the
  ``k==2880 AND n==2880 AND m_total<=16384`` predicate (i.e.
  gpt_oss-Down B=4 family). Documented evidence:
    Down-B4-M2048 (484 tile-steps, 1.89 ws/slot):     +6.14% slots=192
    Down-B4-M4096 (484 tile-steps, 1.89 ws/slot):     +5.22% slots=192
    GateUP-B32-M4096 (7744 tile-steps, 30+ ws/slot):  -17.34% slots=192
  R2 then ANALYTICALLY argued GateUP-B4-M2048 (3.78 ws/slot) is
  "already moderately amortised", and excluded it from the rule
  by gating on n==2880 (excludes GateUP n=5760). The other GateUP
  shapes — B4-M4096 (3.78 ws/slot, K_loop deeper), B32-M2048
  (15 ws/slot) — were never probed.

  Per the R8 next-round suggestion: confirm the analytic exclusion
  on the current binding by tight-verifying num_slots ∈ {128, 160,
  192, 224, 256} on each GateUP wgrad shape. The R2 lever changes
  per-tile prologue/MFMA ratio; if the var-K kernel rebuild has
  shifted the optimum or if the GateUP tile geometry shows a
  different wave-step / num_slots interaction than Down, this is
  worth catching.

Anchors (all use a==2880, b==5760, identical N×K geometry):
  GateUP_B4_M2048 wgrad: m_total=8192,  ~3.78 ws/slot, K_loop=16
  GateUP_B4_M4096 wgrad: m_total=16384, ~3.78 ws/slot, K_loop=32
  GateUP_B32_M2048 wgrad: m_total=65536, ~15 ws/slot, K_loop=16
  per-group output [5760, 2880] ⇒ 22×11 = 242 tile-steps × B groups

Methodology: 1500-iter × 7-trial × 3-seed p20 (mirror of R7 / R8).
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


def _bench(B, M_per_g, N_fwd, K_fwd, gm, xcd, num_slots, seed,
           warmup=80, iters=1500, trials=7):
    grad_q, x_q, sb, sa, g_offs, out = _build_inputs(B, M_per_g, N_fwd, K_fwd, seed)

    def _call():
        fn(grad_q, x_q, out, sb, sa, g_offs,
           group_m=gm, num_xcds=xcd, num_slots=num_slots)

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


def _bit_eq(B, M_per_g, N_fwd, K_fwd, gm, xcd, ns_a, ns_b, seed=0):
    grad_q, x_q, sb, sa, g_offs, _ = _build_inputs(B, M_per_g, N_fwd, K_fwd, seed)
    out_a = torch.zeros((B, N_fwd, K_fwd), dtype=torch.bfloat16, device="cuda")
    out_b = torch.zeros((B, N_fwd, K_fwd), dtype=torch.bfloat16, device="cuda")
    fn(grad_q, x_q, out_a, sb, sa, g_offs, group_m=gm, num_xcds=xcd, num_slots=ns_a)
    fn(grad_q, x_q, out_b, sb, sa, g_offs, group_m=gm, num_xcds=xcd, num_slots=ns_b)
    torch.cuda.synchronize()
    return (out_a.float() - out_b.float()).abs().max().item()


# Three GateUP wgrad shapes (var-K dB), keyed by current rule cell:
#   - B=4 M=2048: R3 (gm=1, xcds=4)  m_total=8192
#   - B=4 M=4096: R9-A (gm=4, xcds=4) m_total=16384  (m_total<32768 branch)
#   - B=32 M=2048: R31 (gm=1, xcds=4) m_total=65536  (m_total>=32768 branch)
N_FWD = 5760
K_FWD = 2880
SHAPES = [
    ("GateUP_B4_M2048",  4,  2048, 1, 4),
    ("GateUP_B4_M4096",  4,  4096, 4, 4),
    ("GateUP_B32_M2048", 32, 2048, 1, 4),
]
NUM_SLOTS_CANDIDATES = [128, 160, 192, 224, 256]  # 0 = kernel default = 256

for name, B, M_per_g, gm, xcd in SHAPES:
    m_total = B * M_per_g
    flops = 2 * B * M_per_g * N_FWD * K_FWD
    print(f"\n=========================================================")
    print(f"  {name}  (B={B}, M_per_g={M_per_g}, m_total={m_total})")
    print(f"  cell = (gm={gm}, xcds={xcd})  [current rule]")
    print(f"  per-group output [{N_FWD}, {K_FWD}] ⇒ 22×11 = 242 tile-steps × {B} = "
          f"{242*B} tile-steps over 256 CUs ≈ {242*B/256:.2f} wave-steps/slot")
    print(f"=========================================================")

    print("\n[bit-eq] (gm, xcd, ns=256) [baseline] vs candidate num_slots:")
    for ns in NUM_SLOTS_CANDIDATES:
        if ns == 256:
            continue
        diff = _bit_eq(B, M_per_g, N_FWD, K_FWD, gm, xcd, 256, ns, seed=42)
        print(f"  ns=256 vs ns={ns:>3}:  max_abs_diff = {diff:.4e}")

    print("\nPer-seed bench (median ms / TF / Δ vs ns=256):")
    all_results = {}
    for ns in NUM_SLOTS_CANDIDATES:
        all_results[ns] = {}
        for seed in (42, 137, 2024):
            med, lo, hi = _bench(B, M_per_g, N_FWD, K_FWD, gm, xcd, ns, seed)
            tf = flops / (med * 1e9)
            all_results[ns][seed] = (med, lo, hi, tf)

    print(f"\n  baseline ns=256:")
    for seed in (42, 137, 2024):
        med, lo, hi, tf = all_results[256][seed]
        print(f"    seed={seed:>4}  ms_med={med:.4f}  TF={tf:6.1f}  spread={(hi-lo)/med*100:.2f}pp")

    print(f"\n  candidates vs ns=256:")
    for ns in NUM_SLOTS_CANDIDATES:
        if ns == 256:
            continue
        seeds_pos = []
        for seed in (42, 137, 2024):
            med, lo, hi, tf = all_results[ns][seed]
            med_base, _, _, tf_base = all_results[256][seed]
            delta = (tf - tf_base) / tf_base * 100
            seeds_pos.append(delta)
            print(f"    ns={ns:>3}  seed={seed:>4}  TF={tf:6.1f}  Δ={delta:+.2f}pp  spread={(hi-lo)/med*100:.2f}pp")
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
        print(f"    ns={ns:>3}  seed-med Δ={seed_med:+.2f}pp  per-seed-spread={seed_spread:.2f}pp  all_pos={all_pos}  verdict={verdict}")
        print()

print("\n=========================================================")
print("Verdict gate: WIN-ROBUST = every-seed Δ > 0 AND seed-med Δ > 0.5pp")
print("              AND seed-spread < seed-med (signal > noise).")
