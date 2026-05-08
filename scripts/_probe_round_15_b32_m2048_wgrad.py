#!/usr/bin/env python3
"""Round 15 — tight verify (16, 4) vs (8, 4) on Down-B32-M2048 wgrad.

R4 audit reported (gm=16, xcds=4) was +0.27% over R1-current's (8, 4) but
NOT ROBUST (50T spread). Re-test under tight 1500-iter × 7-trial × 3-seed.

Also test (4, 4) which R30 originally picked for the entire Down-B32 family
before R1-current split it by m_total.
"""
import os
import sys
os.environ.setdefault("PRIMUS_TURBO_HIPKITTEN_PATH", "/workspace/code/HipKittens")
sys.path.insert(0, "/workspace/code/Primus-Turbo")

import statistics
import time

import torch
import primus_turbo.pytorch as turbo  # noqa: F401
from primus_turbo.pytorch.kernels import hipkitten as hipkit_module
from primus_turbo.pytorch.ops.quantization import quantize_fp8
from primus_turbo.pytorch.core.low_precision import ScalingGranularity

_FP8_DTYPE = torch.float8_e4m3fn
_GRAN = ScalingGranularity.TENSORWISE


def _bench(fn, warmup=10, iters=1500):
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


def make_call(B, M, slots, gm, xcds):
    g_lens = torch.full((B,), M, dtype=torch.int64, device="cuda")
    from primus_turbo.pytorch.kernels.grouped_gemm.grouped_gemm_fp8_impl import (
        grouped_gemm_compute_offs,
    )
    g_offs = grouped_gemm_compute_offs(g_lens)
    a = torch.randn((B * M, 2880), dtype=torch.bfloat16, device="cuda")
    grad = torch.randn((B * M, 2880), dtype=torch.bfloat16, device="cuda")
    a_col, a_s = quantize_fp8(a, _FP8_DTYPE, _GRAN, axis=-2)
    g_col, g_s = quantize_fp8(grad, _FP8_DTYPE, _GRAN, axis=-2)
    hk = hipkit_module.load_fp8()
    out = torch.empty((B, 2880, 2880), dtype=torch.bfloat16, device="cuda")

    def _call():
        hk.grouped_variable_k_crr_dscale(
            g_col, a_col, out, g_s, a_s, g_offs,
            group_m=gm, num_xcds=xcds, num_slots=slots,
        )
    return _call


def run_one(B, M, gm, xcds, slots=0, n_seeds=3, n_trials=7, n_iters=1500):
    print(f"\n=== Tight verify Down-B{B}_M{M} wgrad var-K @ slots={slots}, "
          f"(gm={gm}, xcds={xcds}) ===")
    flops = 2.0 * (B * M) * 2880 * 2880
    SEEDS = [42, 137, 2024][:n_seeds]
    seed_meds = []
    for seed in SEEDS:
        torch.manual_seed(seed)
        call = make_call(B, M, slots, gm, xcds)
        trial_p20s = []
        for _ in range(n_trials):
            trial_p20s.append(_bench(call, warmup=10, iters=n_iters))
        med = statistics.median(trial_p20s)
        lo = min(trial_p20s)
        hi = max(trial_p20s)
        tflops = flops / (med * 1e9)
        seed_meds.append(med)
        print(f"    seed={seed}  med={med:.4f}ms  lo={lo:.4f}  hi={hi:.4f}  TF={tflops:.1f}")
    overall_med = statistics.median(seed_meds)
    overall_lo = min(seed_meds)
    overall_hi = max(seed_meds)
    spread_pp = (overall_hi - overall_lo) / overall_med * 100
    overall_tflops = flops / (overall_med * 1e9)
    print(f"    OVERALL: med={overall_med:.4f}ms (TF={overall_tflops:.1f})  "
          f"spread={spread_pp:.3f}pp")
    return overall_med, overall_tflops


if __name__ == "__main__":
    print("[probe] Round 15 tight verify B32 wgrad")
    print(f"[probe] HIP_VISIBLE_DEVICES={os.environ.get('HIP_VISIBLE_DEVICES', '<unset>')}")
    print(f"[probe] device: {torch.cuda.get_device_name(0)}")
    t0 = time.monotonic()

    # Down-B32-M2048 wgrad — current R1-current cell
    print("\n>>> Down-B32-M2048 wgrad (m_total=65536), slots=0 (default 256)")
    cur = run_one(32, 2048, 8, 4, slots=0)  # current R1-current
    alt1 = run_one(32, 2048, 16, 4, slots=0)  # R4-noted alternative
    alt2 = run_one(32, 2048, 4, 4, slots=0)  # R30 original cell
    alt3 = run_one(32, 2048, 12, 4, slots=0)  # interpolation
    alt4 = run_one(32, 2048, 8, 8, slots=0)  # xcds=default
    alt5 = run_one(32, 2048, 8, 2, slots=0)  # xcds=2
    print("\n=== Down-B32-M2048 wgrad summary ===")
    print(f"  cur (8, 4): {cur[1]:.1f} TF")
    print(f"  alt1 (16, 4): {alt1[1]:.1f} TF  Δ = {(cur[0]-alt1[0])/cur[0]*100:+.2f}%")
    print(f"  alt2 (4, 4): {alt2[1]:.1f} TF  Δ = {(cur[0]-alt2[0])/cur[0]*100:+.2f}%")
    print(f"  alt3 (12, 4): {alt3[1]:.1f} TF  Δ = {(cur[0]-alt3[0])/cur[0]*100:+.2f}%")
    print(f"  alt4 (8, 8): {alt4[1]:.1f} TF  Δ = {(cur[0]-alt4[0])/cur[0]*100:+.2f}%")
    print(f"  alt5 (8, 2): {alt5[1]:.1f} TF  Δ = {(cur[0]-alt5[0])/cur[0]*100:+.2f}%")

    print(f"\n[probe] wall {time.monotonic()-t0:.1f}s")
