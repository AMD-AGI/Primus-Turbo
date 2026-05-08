"""Round-9 in-process verify of the new per-call num_slots arg added to
grouped_rcr_dscale (HK surgery this round).

Calls grouped_rcr_dscale TWICE on the same input — once with num_slots=0
(default → NUM_CUS=256) and once with num_slots=200 (the WIN-ROBUST cell
from the subprocess probe). Verifies:
  (a) bit-equivalent output (max_abs_diff = 0)
  (b) per-call kernel time differs as expected (slots=200 faster on
      Down-B4-M2048 fwd, same shape as the subprocess probe)

If both pass, the surgery is wired correctly.
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
    fn(a, b, out_a, sa, sb, g_offs, gm, m_per_group=M, num_xcds=xcd,
       num_slots=ns_a)
    fn(a, b, out_b, sa, sb, g_offs, gm, m_per_group=M, num_xcds=xcd,
       num_slots=ns_b)
    torch.cuda.synchronize()
    return (out_a.float() - out_b.float()).abs().max().item()


B, M, N, K = 4, 2048, 2880, 2880
gm, xcd = 16, 2  # R2 cell for Down-B4-M2048 fwd
flops = 2 * B * M * N * K

print("=" * 70)
print("  Round-9 in-process per-call num_slots wiring verify")
print(f"  Down-B4-M2048 fwd RCR (B={B} M={M} N={N} K={K}, gm={gm}, xcds={xcd})")
print("=" * 70)

# Bit-eq check first
print("\n[bit-eq] num_slots=0 (default) vs num_slots=200, num_slots=208, num_slots=256:")
for ns in (200, 208, 220, 256):
    diff = _bit_eq(B, M, N, K, gm, xcd, 0, ns, seed=42)
    print(f"  ns=0 vs ns={ns:>3}:  max_abs_diff = {diff:.4e}")
print()

# Bench: confirm per-call num_slots produces the same lift the
# subprocess probe found.
print("Bench (1500-iter × 7-trial × 3-seed p20):")
results = {}
for ns in (0, 200, 208, 256):
    results[ns] = {}
    for seed in (42, 137, 2024):
        med, lo, hi = _bench(B, M, N, K, gm, xcd, ns, seed)
        tf = flops / (med * 1e9)
        results[ns][seed] = (med, lo, hi, tf)

print("\n  baseline ns=0 (default = legacy NUM_CUS=256):")
for seed in (42, 137, 2024):
    med, lo, hi, tf = results[0][seed]
    print(f"    seed={seed:>4}  ms_med={med:.4f}  TF={tf:6.1f}")

print("\n  candidates vs ns=0:")
for ns in (200, 208, 256):
    print(f"  ns={ns}:")
    for seed in (42, 137, 2024):
        med, lo, hi, tf = results[ns][seed]
        med_b, _, _, tf_b = results[0][seed]
        delta = (tf - tf_b) / tf_b * 100
        print(f"    seed={seed:>4}  ms_med={med:.4f}  TF={tf:6.1f}  Δ={delta:+.2f}pp")

print("\n  Expected: ns=200 / ns=208 should match the subprocess probe's +5%")
print("  Expected: ns=256 should match ns=0 (both → NUM_CUS=256, identical)")
