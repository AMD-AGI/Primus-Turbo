"""Round-7 (gpt_oss FP8 kernel-only ceiling) tight A/B verify:
GateUP_B4_M2048 dgrad (RCR via H4) — currently on BINDING DEFAULT
(group_m=4, num_xcds=None → kernel BLOCK_SWIZZLE_NUM_XCDS=8). This is
the ONLY gpt_oss dgrad shape without a dispatcher rule.

R34 audit (`config.py` lines 2084-2089) recorded "default best; do not
add a rule for tiles_m=8 + m_total<65536" with the candidate table:

  default (4, 8)  baseline                              keep default
  (4, 4)          -1.26%  spread 2.69pp  TIE
  (8, 4)          -1.54%  spread 3.45pp  TIE
  (16, 4)         -2.51%  spread 3.50pp  TIE

That probe used 200-iter × 7-trial × p20 × 3 seeds methodology. R4's
later audit (in the round-4 falsification note) marked the result
"NOT ROBUST" but did not re-test on the post-R5/R10dm/R35-rebuilt
binary. This round re-probes with TIGHTER methodology (1500-iter ×
7-trial × 3-seed p20, mirror of R17 fwd RCR drift verify) on the
CURRENT binding (commit 2d3946f1) to resolve drift.

Anchor:
  m=2048, n=K_fwd=2880, k=N_fwd=5760, m_total=8192, tiles_m=8,
  tiles_n=11, k=5760.
Default behavior: HipKittenConfig(group_m=4, num_xcds=None, kernel=None)
=> kernel runs with BLOCK_SWIZZLE_NUM_XCDS=8.
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
assert fn is not None


def _quantize(t):
    amax = t.abs().max()
    s = (amax / 240.0).clamp(min=1e-6).to(torch.float32)
    q = (t.to(torch.float32) / s).clamp(-240.0, 240.0).to(torch.float8_e4m3fn)
    return q, s


def _bench(B, M, N_fwd, K_fwd, gm, xcd, seed, warmup=80, iters=1500, trials=7):
    """Mirror dgrad path: grad@b_T after H4 reroute. Effective RCR call:
        a [m_total, N_fwd], b_T [bs, K_fwd, N_fwd], output [m_total, K_fwd].
        Dispatcher key: m=M, n=K_fwd, k=N_fwd, m_total=B*M.
    xcd=0 in the binding maps to the kernel default BLOCK_SWIZZLE_NUM_XCDS=8."""
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
    avg_m = M

    def _call():
        fn(a, b_t, out, sa, sb, g_offs, gm, m_per_group=avg_m, num_xcds=xcd)

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


def _bit_eq(B, M, N_fwd, K_fwd, gm_a, xcd_a, gm_b, xcd_b, seed=0):
    torch.manual_seed(seed)
    g_lens = torch.full((B,), M, dtype=torch.int64, device="cuda")
    g_offs = torch.zeros(B + 1, dtype=torch.int64, device="cuda")
    g_offs[1:] = torch.cumsum(g_lens, dim=0)
    grad_bf = torch.randn((B * M, N_fwd), dtype=torch.bfloat16, device="cuda")
    b_orig_bf = torch.randn((B, N_fwd, K_fwd), dtype=torch.bfloat16, device="cuda")
    a, sa = _quantize(grad_bf)
    b_orig, sb = _quantize(b_orig_bf)
    b_t = fp8_transpose_3d(b_orig)
    out_a = torch.zeros((B * M, K_fwd), dtype=torch.bfloat16, device="cuda")
    out_b = torch.zeros((B * M, K_fwd), dtype=torch.bfloat16, device="cuda")
    fn(a, b_t, out_a, sa, sb, g_offs, gm_a, m_per_group=M, num_xcds=xcd_a)
    fn(a, b_t, out_b, sa, sb, g_offs, gm_b, m_per_group=M, num_xcds=xcd_b)
    torch.cuda.synchronize()
    return (out_a.float() - out_b.float()).abs().max().item()


def _flops(B, M, N, K):
    return 2 * B * M * N * K


B, M, N_fwd, K_fwd = 4, 2048, 5760, 2880
flops = _flops(B, M, N_fwd, K_fwd)

print(f"========= GateUP_B4_M2048 dgrad (RCR via H4) — round-7 tight verify =========")
print(f"  m={M}, n=K_fwd={K_fwd}, k=N_fwd={N_fwd}, m_total={B*M}, "
      f"tiles_m={M//256}, tiles_n={K_fwd//256}, k={N_fwd}")
print(f"  ITERS=1500 × TRIALS=7 × SEEDS={{42, 137, 2024}}")
print(f"  Currently: BINDING DEFAULT (gm=4, xcds=0 → kernel BSNX=8).")
print()

# Bit-eq sanity
print("[bit-eq] (gm=4, xcd=0) [default] vs candidate cells (max abs diff in bf16):")
for (gm, xcd) in [(4, 4), (8, 4), (16, 4), (1, 4), (1, 8), (8, 8), (16, 8), (1, 2), (4, 2)]:
    diff = _bit_eq(B, M, N_fwd, K_fwd, 4, 0, gm, xcd, seed=42)
    print(f"  (4, 0) vs ({gm:>2}, {xcd}):  max_abs_diff = {diff:.4e}")
print()

# Candidate cells. Default = (gm=4, xcds=0 → kernel default 8).
CELLS = [
    ("default", 4, 0),  # baseline; xcds=0 means kernel auto-pick BSNX=8
    ("R34-tested", 4, 4),
    ("R34-tested", 8, 4),
    ("R34-tested", 16, 4),
    ("new-A", 1, 4),
    ("new-B", 1, 8),
    ("new-C", 8, 8),
    ("new-D", 16, 8),
    ("new-E", 1, 2),
    ("new-F", 4, 2),
]

print("Per-seed bench (median ms / TF / Δ vs default):")
all_results = {}
for label, gm, xcd in CELLS:
    all_results[(gm, xcd)] = {}
    for seed in (42, 137, 2024):
        med, lo, hi = _bench(B, M, N_fwd, K_fwd, gm=gm, xcd=xcd, seed=seed)
        tf = flops / (med * 1e9)
        all_results[(gm, xcd)][seed] = (med, lo, hi, tf)

print(f"\n  baseline default (gm=4, xcd=0 → BSNX=8):")
for seed in (42, 137, 2024):
    med, lo, hi, tf = all_results[(4, 0)][seed]
    print(f"    seed={seed:>4}  ms_med={med:.4f}  [{lo:.4f}, {hi:.4f}]  TF={tf:6.1f}  spread={(hi-lo)/med*100:.2f}pp")

print(f"\n  candidates vs default:")
for label, gm, xcd in CELLS:
    if (gm, xcd) == (4, 0):
        continue
    seeds_pos = []
    for seed in (42, 137, 2024):
        med, lo, hi, tf = all_results[(gm, xcd)][seed]
        med_base, _, _, tf_base = all_results[(4, 0)][seed]
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
