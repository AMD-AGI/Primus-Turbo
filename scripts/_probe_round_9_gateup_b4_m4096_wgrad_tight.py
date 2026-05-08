"""Round-9 (gpt_oss FP8 kernel-only ceiling) tight-2 verify:
GateUP_B4_M4096 wgrad var-K dB — num_slots tight scan around ns=224.

Background:
  Round-9 first probe (_probe_round_9_gateup_wgrad_num_slots.py)
  found ns=224 borderline WIN-LIGHT at +0.31% seed-med vs ns=256
  baseline (3/3 seeds positive: +0.09/+0.31/+0.35; spread 0.26pp).
  This sits below the standard 0.5pp WIN-ROBUST threshold but with
  every-seed positive and tight spread. Need a tighter scan to:
    (a) confirm or falsify the WIN-LIGHT signal at ns=224
    (b) check if the optimum is even closer to ns=256 (e.g. ns=232,
        ns=240, ns=248)

Anchor:
  m_total=16384, B=4, M_per_g=4096, N_fwd=5760, K_fwd=2880
  per-group output [5760, 2880] ⇒ 22×11 = 242 tile-steps × 4 = 968
  tile-steps over 256 CUs ≈ 3.78 wave-steps per slot, K_loop=32.
  Current rule: R9-A (gm=4, xcds=4) [m_total<32768 GateUP branch]

Methodology: 1500-iter × 7-trial × 5-seed p20 (more seeds than R7's 3;
mirror of R45's tight-2 verify methodology).
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


B, M_per_g, N_FWD, K_FWD = 4, 4096, 5760, 2880
m_total = B * M_per_g
flops = 2 * B * M_per_g * N_FWD * K_FWD
gm, xcd = 4, 4

# Tight scan around ns=224, plus ns=256 baseline. Add ns=232/240/248 to
# check whether the optimum sits between 224 and 256.
NS_CANDIDATES = [224, 232, 240, 248, 256]
SEEDS = (42, 137, 2024, 7, 1234)  # 5 seeds, two new (7, 1234)

print(f"=========================================================")
print(f"  GateUP_B4_M4096 wgrad var-K dB — round-9 tight-2 verify")
print(f"  m_total={m_total}, gm={gm}, xcds={xcd}")
print(f"  ns ∈ {NS_CANDIDATES}, seeds = {SEEDS}, 1500-iter × 7-trial p20")
print(f"=========================================================")

print("\nPer-seed bench (median ms / TF):")
all_results = {}
for ns in NS_CANDIDATES:
    all_results[ns] = {}
    for seed in SEEDS:
        med, lo, hi = _bench(B, M_per_g, N_FWD, K_FWD, gm, xcd, ns, seed)
        tf = flops / (med * 1e9)
        all_results[ns][seed] = (med, lo, hi, tf)

print(f"\n  baseline ns=256:")
for seed in SEEDS:
    med, lo, hi, tf = all_results[256][seed]
    print(f"    seed={seed:>4}  ms_med={med:.4f}  TF={tf:6.1f}  spread={(hi-lo)/med*100:.2f}pp")

print(f"\n  candidates vs ns=256:")
for ns in NS_CANDIDATES:
    if ns == 256:
        continue
    seeds_pos = []
    for seed in SEEDS:
        med, lo, hi, tf = all_results[ns][seed]
        med_base, _, _, tf_base = all_results[256][seed]
        delta = (tf - tf_base) / tf_base * 100
        seeds_pos.append(delta)
        print(f"    ns={ns:>3}  seed={seed:>4}  TF={tf:6.1f}  Δ={delta:+.3f}pp  spread={(hi-lo)/med*100:.2f}pp")
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
    pos_count = sum(1 for d in seeds_pos if d > 0)
    print(f"    ns={ns:>3}  seed-med Δ={seed_med:+.3f}pp  spread={seed_spread:.2f}pp  pos={pos_count}/{len(SEEDS)}  verdict={verdict}")
    print()

print("=========================================================")
print("Verdict gate: WIN-ROBUST = every-seed Δ > 0 AND seed-med > 0.5pp")
print("              AND seed-spread < seed-med.")
