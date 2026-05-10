#!/usr/bin/env python3
"""R46 drift re-audit on GateUP-B4-M2048 dgrad-via-H4 RCR rule (METHODOLOGY-CORRECTED).

R45 forward-pointer (PRIMARY): the R45 dgrad column ran (gm, xcd) sweep
with slots=0/cs=0 — uninterpretable because the production rule
(config.py:3022-3029) bundles R10 (slots=200) + R15 (chunk_size=24) +
R16 (gm=1, xcds=None=8). With those load-bearing levers FIXED to
production, the (gm, xcd) optimum may have shifted on the current
binding.

Production rule under test:
    HipKittenConfig(group_m=1, num_xcds=None, num_slots=200, chunk_size=24)
    (xcds=None → kernel BLOCK_SWIZZLE_NUM_XCDS=8 default)

Hypothesis: kernel codegen drift since R16 may have shifted the (gm,
xcds) optimum. R16's wide sweep ranking was:
    (1, default=8) > (12,8) ≈ (16,8) ≈ (24,8) > (2,8) > (4,8) > (8,8) [baseline]
on tiles_n=11 + k=5760 + tiles_m=8 + m_total=8192.

Cells (gm, xcd) per R45 forward-pointer:
  (1, 8)*  — current production rule (baseline; xcds=8 == None default)
  (1, 4)   — alt chiplet partition (R10 had it; R16 kept xcds=None)
  (1, 2)   — narrower chiplet partition (probe)
  (1, 1)   — single-XCD control (catastrophic-loss expected)
  (2, 8)   — gm=2 with same xcds=8 (R16 column shows -1.74 % at R16
             time; may have drifted)
  (4, 8)   — gm=4 with same xcds=8 (R16 shows -1.83 % at R16 time)
  (8, 8)   — gm=8 with same xcds=8 (R16 baseline before retune;
             R16 docs +0 % vs gm=8)

Methodology: 5 seeds × 2000-iter p20 per cell, kernel-only timing via
direct grouped_rcr_dscale call with monkey-patched (gm, xcd, slots=200,
chunk_size=24) — load-bearing R10/R15/R16 levers held FIXED. Probe
runs only time_dgrad (R46 hypothesis is dgrad-only; the fwd RCR rule
on this shape is config.py:1418-1482 with slots=0/cs=0 default and was
already R45-FALSIFIED).

SHIP gate: best != baseline AND lift >= 1.0 % AND signal > spread.

Outcome (per R45 BACKUP plan): if no cell wins >= 1.0 % over (1, 8)
at production slots/cs, R44+R45+R46 establish three consecutive
empirical drift FALSIFICATIONS on the most-tuned dispatcher rules in
the gpt_oss FP8 suite. Justifies the R10-R46 audit summary marking the
dispatcher (gm, xcds) lever class feature-complete on the structural
~700 plateau.
"""
import os
import sys
import statistics
import time

os.environ.setdefault("PRIMUS_TURBO_HIPKITTEN_PATH", "/workspace/code/HipKittens")
sys.path.insert(0, "/workspace/code/Primus-Turbo")
sys.path.insert(0, "/workspace/code/Primus-Turbo/scripts")

import torch
import primus_turbo.pytorch as turbo  # noqa: F401
from primus_turbo.pytorch.core.backend import BackendType, PrecisionType
from primus_turbo.pytorch.kernels.grouped_gemm.grouped_gemm_fp8_impl import (
    grouped_gemm_compute_offs,
    grouped_gemm_fp8_impl,
)
from primus_turbo.pytorch.ops.quantization import quantize_fp8
from primus_turbo.pytorch.core.low_precision import ScalingGranularity
from primus_turbo.pytorch.kernels import hipkitten as hipkit_module
import _metric_hk_ratio as hk_ratio

_FP8_DTYPE = torch.float8_e4m3fn
_GRAN = ScalingGranularity.TENSORWISE

# Held FIXED — production R10/R15/R16 levers (config.py:3022-3029)
SLOTS = 200
CHUNK = 24


def _bench_p20(fn, warmup=20, iters=2000):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    se = torch.cuda.Event(enable_timing=True)
    ee = torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(iters):
        torch.cuda.synchronize()
        se.record()
        fn()
        ee.record()
        torch.cuda.synchronize()
        times.append(se.elapsed_time(ee))
    times.sort()
    return times[len(times) // 5]


def _patch_hk_dscale(gm, xcds):
    hk = hipkit_module.load_fp8()
    orig = hk.grouped_rcr_dscale

    def wrapped(*args, **kwargs):
        if len(args) >= 7:
            args = args[:6]
        kwargs["group_m"] = gm
        kwargs["num_xcds"] = xcds
        kwargs["num_slots"] = SLOTS
        kwargs["chunk_size"] = CHUNK
        return orig(*args, **kwargs)

    object.__setattr__(hk, "grouped_rcr_dscale", wrapped)
    return orig


def _restore_hk_dscale(orig):
    hk = hipkit_module.load_fp8()
    object.__setattr__(hk, "grouped_rcr_dscale", orig)


def time_dgrad(B, M, N, K):
    g_lens = torch.full((B,), M, dtype=torch.int64, device="cuda")
    g_offs = grouped_gemm_compute_offs(g_lens)
    grad_out = torch.randn((B * M, N), dtype=torch.bfloat16, device="cuda")
    b = torch.randn((B, N, K), dtype=torch.bfloat16, device="cuda")
    g_out_fp8, g_out_s = quantize_fp8(grad_out, _FP8_DTYPE, _GRAN)
    b_fp8, b_s = quantize_fp8(b, _FP8_DTYPE, _GRAN)

    def _call():
        return grouped_gemm_fp8_impl(
            g_out_fp8, b_fp8, g_out_s, b_s, g_lens, g_offs,
            trans_a=False, trans_b=False, out_dtype=torch.bfloat16,
            granularity=_GRAN.value, num_cu=None,
            default_backend=BackendType.HIPKITTEN.value,
        )

    flops = 2.0 * (B * M) * N * K
    with hk_ratio.force_grouped_gemm_backend(BackendType.HIPKITTEN, PrecisionType.FP8):
        ms = _bench_p20(_call)
    return flops / (ms * 1e9), ms


def run(label, time_fn, B, M, N, K, cells, baseline_cell, seeds=(42, 137, 2024, 99, 1234)):
    print(f"\n=== {label} (B={B}, M={M}, N={N}, K={K}) ===")
    print(f"  {'cell':>10}  {'med ms':>8}  {'min ms':>8}  {'max ms':>8}  {'spread%':>7}  {'TFLOPS':>8}  {'delta%':>7}")
    flops = 2.0 * (B * M) * N * K
    results = {}
    for cell in cells:
        gm, xcd = cell
        orig = _patch_hk_dscale(gm, xcd)
        seed_meds = []
        for seed in seeds:
            torch.manual_seed(seed)
            t, ms = time_fn(B, M, N, K)
            seed_meds.append(ms)
        med = statistics.median(seed_meds)
        results[cell] = (med, min(seed_meds), max(seed_meds))
        _restore_hk_dscale(orig)

    base_ms = results[baseline_cell][0]
    for cell in cells:
        med, lo, hi = results[cell]
        tflops = flops / (med * 1e9)
        spread_pp = (hi - lo) / med * 100
        delta_pp = (base_ms - med) / base_ms * 100
        marker = " *base" if cell == baseline_cell else ""
        print(f"  {str(cell):>10}  {med:>8.4f}  {lo:>8.4f}  {hi:>8.4f}  {spread_pp:>6.2f}%  {tflops:>8.1f}  {delta_pp:+6.2f}%{marker}")
    best_cell = min(cells, key=lambda c: results[c][0])
    best_med = results[best_cell][0]
    lift = (base_ms - best_med) / base_ms * 100
    print(f"  BEST: cell={best_cell}  ({lift:+.2f}% over baseline {baseline_cell})")
    return results, best_cell, lift


if __name__ == "__main__":
    print(f"[probe] R46 GateUP-B4-M2048 dgrad-via-H4 (gm, xcds) drift re-audit METHODOLOGY-CORRECTED")
    print(f"[probe] holding slots={SLOTS}, chunk_size={CHUNK} FIXED (R10/R15 production levers)")
    print(f"[probe] device: {torch.cuda.get_device_name(0)}")

    # (1, 8) is current production (config.py:3022-3029, xcds=None==8 default)
    cells = [(1, 8), (1, 4), (1, 2), (1, 1), (2, 8), (4, 8), (8, 8)]
    baseline = (1, 8)

    t0 = time.monotonic()
    res_dg, best_dg, lift_dg = run(
        "GateUP-B4-M2048 dgrad-via-H4", time_dgrad,
        B=4, M=2048, N=5760, K=2880,
        cells=cells, baseline_cell=baseline,
    )
    print(f"\n[probe] total wall {time.monotonic()-t0:.1f}s")

    flops = 2.0 * 4 * 2048 * 5760 * 2880
    print(f"\n[probe] SUMMARY:")
    print(f"  baseline (1,8) [production]: {flops / (res_dg[(1, 8)][0] * 1e9):.1f} T")
    print(f"  best     {best_dg}: {flops / (res_dg[best_dg][0] * 1e9):.1f} T  ({lift_dg:+.2f}%)")

    if best_dg != baseline:
        med, lo, hi = res_dg[best_dg]
        spread_pp = (hi - lo) / med * 100
        ship = (lift_dg >= 1.0 and lift_dg > spread_pp)
        print(f"\n[probe] SHIP gate: best={best_dg} (lift {lift_dg:.2f}%, spread {spread_pp:.2f}%)")
        print(f"        DECISION: {'SHIP' if ship else 'FALSIFIED-noise'}")
    else:
        print(f"\n[probe] baseline still optimal — DECISION: FALSIFIED")
