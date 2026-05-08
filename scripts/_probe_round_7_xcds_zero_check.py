"""Round-7 sanity probe: confirm xcds=0 (binding default → kernel
BSNX=8) is bit + perf equivalent to xcds=8 (explicit) for GateUP_B4_
M2048 dgrad. The R7 rule produces ``HipKittenConfig(group_m=8,
num_xcds=None)`` which the impl maps to ``xcds_arg=0``; the original
probe tested xcds=8 explicitly. The R67 / binding-default comments
state these are equivalent, but the metric did NOT show the probe's
+2.11% lift — so we need to verify."""
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


B, M, N_fwd, K_fwd = 4, 2048, 5760, 2880
flops = 2 * B * M * N_fwd * K_fwd

print(f"========= xcds=0 vs xcds=8 equivalence — round-7 sanity =========")
print(f"  shape: GateUP_B4_M2048 dgrad (m={M}, n=K_fwd={K_fwd}, k=N_fwd={N_fwd}, m_total={B*M})")
print()

# Bit-eq across xcds=0 vs xcds=8 at gm=4 and gm=8.
for gm in [4, 8]:
    diff = _bit_eq(B, M, N_fwd, K_fwd, gm, 0, gm, 8, seed=42)
    print(f"  bit-eq gm={gm}: (xcds=0) vs (xcds=8) -> max_abs_diff = {diff:.4e}")

print()
print("Per-cell tight bench (1500 iters × 7 trials × 3 seeds):")
CELLS = [
    ("default", 4, 0),
    ("default-explicit", 4, 8),
    ("R7", 8, 0),
    ("R7-explicit", 8, 8),
]
all_results = {}
for label, gm, xcd in CELLS:
    all_results[(gm, xcd)] = {}
    for seed in (42, 137, 2024):
        med, lo, hi = _bench(B, M, N_fwd, K_fwd, gm=gm, xcd=xcd, seed=seed)
        tf = flops / (med * 1e9)
        all_results[(gm, xcd)][seed] = (med, lo, hi, tf)

for label, gm, xcd in CELLS:
    print(f"\n  ({gm:>2}, {xcd}) [{label}]:")
    for seed in (42, 137, 2024):
        med, lo, hi, tf = all_results[(gm, xcd)][seed]
        print(f"    seed={seed:>4}  ms_med={med:.4f}  [{lo:.4f}, {hi:.4f}]  TF={tf:6.1f}  spread={(hi-lo)/med*100:.2f}pp")
    seed_meds = [all_results[(gm, xcd)][s][0] for s in (42, 137, 2024)]
    print(f"    cross-seed median ms = {statistics.median(seed_meds):.4f}  TF = {flops/(statistics.median(seed_meds)*1e9):6.1f}")

print()
print("Δ (xcds=0) vs (xcds=8) at each gm:")
for gm in [4, 8]:
    s0 = [all_results[(gm, 0)][s][0] for s in (42, 137, 2024)]
    s8 = [all_results[(gm, 8)][s][0] for s in (42, 137, 2024)]
    med0 = statistics.median(s0)
    med8 = statistics.median(s8)
    delta = (med8 - med0) / med0 * 100
    print(f"  gm={gm}: ms_med xcd=0 = {med0:.4f}  vs  xcd=8 = {med8:.4f}  Δ = {delta:+.2f}pp")
