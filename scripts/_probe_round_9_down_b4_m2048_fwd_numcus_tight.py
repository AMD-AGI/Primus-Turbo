"""Round-9 (gpt_oss FP8 kernel-only ceiling) tight-2 verify of R4's
TK_RCR_NUM_CUS lever on Down-B4-M2048 fwd.

Background:
  R4 (commit pre-1d526e2 in this Primus run, 2026-05-07) probed
  TK_RCR_NUM_CUS slots ∈ {128, 160, 192, 208, 224, 240, 256} on
  Down-B4-M2048 fwd (the sparsest fwd shape, ~1.5 wave-steps/CU)
  using 250-iter × 7-trial × 3-seed methodology and reported:
    slots=192   1590.1 T   -0.69%
    slots=208   1624.6 T   +1.47%   ← within ±2% noise per R4
    slots=224   1587.3 T   -0.86%
    slots=256   1601.1 T   baseline
  R4 declared the +1.47% "within noise" and shipped no rule.

  Per-shape sibling regressions in R4:
    Down-B4-M4096 fwd     slots=208 -17.58%  LOSS (huge)
    GateUP-B32-M4096 fwd  slots=240 -12.90%  LOSS (saturated)

  This round: tight-verify R4's borderline +1.47% on the SAME shape
  with the standard R7/R8 1500-iter × 7-trial × 5-seed methodology
  to either (a) confirm a real WIN-ROBUST that R4's coarser
  methodology missed, or (b) definitively kill the +1.47% as
  noise.

  Critical caveat: TK_RCR_NUM_CUS is process-static cached on first
  read inside the HK kernel binding (see kernel_fp8_layouts.cpp
  line 7406-7412 — `static const int rcr_slots = []() {...}();`).
  This forces subprocess-per-slot probe (cannot toggle in-process).

  IF this confirms a robust WIN, the next step is HK kernel surgery
  to thread num_slots through the grouped_rcr_dscale binding as a
  per-call argument (mirror of var-K's num_slots, which IS per-call).
  Until that surgery happens the lever cannot be wired through a
  Python dispatcher rule — but tight-verifying first is cheap and
  tells us whether the surgery is worth attempting.

Anchor:
  Down-B4-M2048 fwd RCR. B=4, M=2048, N=2880, K=2880.
  Current rule: R2 (gm=16, xcds=2). Persistent grid 96 tile-steps
  per group × 4 groups = 384 tile-steps over 256 CUs ≈ 1.5
  wave-steps/CU. Sparsest fwd shape in the metric.

Methodology: subprocess-per-slot, each child runs 1500-iter ×
7-trial × p20 × 5-seed direct grouped_rcr_dscale call, prints
RESULT line for parent collection.
"""
from __future__ import annotations

import os
import statistics
import subprocess
import sys

SLOT_VALUES = [196, 200, 204, 208, 212, 216, 220, 256]  # 256 = baseline
SEEDS = [42, 137, 2024, 7, 1234]

CHILD_SOURCE = r'''
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


def _bench(B, M, N, K, gm, xcd, seed, warmup=80, iters=1500, trials=7):
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
        fn(a, b, out, sa, sb, g_offs, gm, m_per_group=M, num_xcds=xcd)

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


B, M, N, K = 4, 2048, 2880, 2880
gm, xcd = 16, 2  # R2 dispatcher cell for Down-B4-M2048 fwd
flops = 2 * B * M * N * K
seeds = [int(s) for s in sys.argv[1].split(",")]
results = {}
for seed in seeds:
    med, lo, hi = _bench(B, M, N, K, gm, xcd, seed)
    tf = flops / (med * 1e9)
    results[seed] = (med, lo, hi, tf)
for seed in seeds:
    med, lo, hi, tf = results[seed]
    print(f"RESULT seed={seed} med_ms={med:.5f} lo={lo:.5f} hi={hi:.5f} tf={tf:.2f}",
          flush=True)
'''


def run_one_slot(slots):
    env = os.environ.copy()
    env.setdefault("PRIMUS_TURBO_HIPKITTEN_PATH", "/workspace/code/HipKittens")
    if slots != 256:
        env["TK_RCR_NUM_CUS"] = str(slots)
    elif "TK_RCR_NUM_CUS" in env:
        del env["TK_RCR_NUM_CUS"]
    seed_arg = ",".join(str(s) for s in SEEDS)
    proc = subprocess.run(
        [sys.executable, "-c", CHILD_SOURCE, seed_arg],
        env=env, capture_output=True, text=True, timeout=600,
    )
    out = (proc.stdout or "") + (proc.stderr or "")
    seed_data = {}
    for line in out.splitlines():
        if line.startswith("RESULT "):
            tokens = line.split()
            d = {k: float(v) for k, v in (t.split("=") for t in tokens[1:])}
            seed_data[int(d["seed"])] = (d["med_ms"], d["lo"], d["hi"], d["tf"])
    if not seed_data:
        print(f"  [slots={slots}] FAILED. Tail: {out[-400:]}", flush=True)
    return seed_data


print("=" * 70)
print("  Down-B4-M2048 fwd RCR — round-9 tight verify of R4 TK_RCR_NUM_CUS lever")
print(f"  B=4 M=2048 N=2880 K=2880  cell=(gm=16, xcds=2) [R2 dispatcher]")
print(f"  slots ∈ {SLOT_VALUES}  seeds={SEEDS}")
print(f"  per-cell: 1500-iter × 7-trial p20 × {len(SEEDS)}-seed via subprocess")
print("=" * 70)

baseline = run_one_slot(256)
all_results = {256: baseline}
for slots in SLOT_VALUES:
    if slots == 256:
        continue
    print(f"\n[slot={slots}]")
    all_results[slots] = run_one_slot(slots)

print("\n  baseline (slots=256):")
for seed in SEEDS:
    med, lo, hi, tf = baseline[seed]
    print(f"    seed={seed:>4}  ms_med={med:.4f}  TF={tf:6.1f}  spread={(hi-lo)/med*100:.2f}pp")

print("\n  candidates vs slots=256:")
for slots in SLOT_VALUES:
    if slots == 256:
        continue
    cell = all_results[slots]
    seeds_pos = []
    for seed in SEEDS:
        med, lo, hi, tf = cell[seed]
        med_b, _, _, tf_b = baseline[seed]
        delta = (tf - tf_b) / tf_b * 100
        seeds_pos.append(delta)
        print(f"    slots={slots:>3}  seed={seed:>4}  TF={tf:6.1f}  Δ={delta:+.3f}pp  spread={(hi-lo)/med*100:.2f}pp")
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
    print(f"    slots={slots:>3}  seed-med Δ={seed_med:+.3f}pp  spread={seed_spread:.2f}pp  pos={pos_count}/{len(SEEDS)}  verdict={verdict}")
    print()

print()
print("=" * 70)
print("Verdict gate: WIN-ROBUST = every-seed Δ > 0 AND seed-med > 0.5pp")
print("              AND seed-spread < seed-med.")
