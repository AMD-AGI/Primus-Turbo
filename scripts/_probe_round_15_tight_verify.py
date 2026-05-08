#!/usr/bin/env python3
"""Round 15 — tight verify (1, 4) vs (1, 2) on Down-B4-M4096 wgrad at slots=192.

Probe coarse data (1500-iter × 5-rep × 3-seed) suggests (gm=1, xcds=4) wins
+0.85% over R10's (1, 2) cell ON THE SLOTS=192 GRID. Verify with the
1500-iter × 7-trial × 3-seed methodology used by R10 / R11.

Also test the sibling shape Down-B4-M2048 wgrad (m_total=8192) to ensure
the rule split (M=2048 keeps (1, 2), M=4096 changes to (1, 4)) holds OR
both shapes prefer same cell.

Bit-equivalence: (gm, xcds) are pure persistent-grid scheduling knobs;
documented bit-equivalent in R10/R11/R30/R31 etc. Verify max_abs_diff=0.
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
    return times[len(times) // 5]  # p20


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
            grad, a, out, g_s, a_s, g_offs,
            group_m=gm, num_xcds=xcds, num_slots=slots,
        )
    return _call, out, grad, a, g_s, a_s, g_offs


def run_one(B, M, gm, xcds, slots=192, n_seeds=3, n_trials=7, n_iters=1500):
    print(f"\n=== Tight verify Down-B{B}_M{M} wgrad var-K @ slots={slots}, "
          f"(gm={gm}, xcds={xcds}) ===")
    print(f"    seeds={n_seeds}, trials/seed={n_trials}, iters/trial={n_iters}")
    flops = 2.0 * (B * M) * 2880 * 2880
    SEEDS = [42, 137, 2024][:n_seeds]
    seed_meds = []
    seed_data = []
    for seed in SEEDS:
        torch.manual_seed(seed)
        call, _, _, _, _, _, _ = make_call(B, M, slots, gm, xcds)
        trial_p20s = []
        for _ in range(n_trials):
            trial_p20s.append(_bench(call, warmup=10, iters=n_iters))
        med = statistics.median(trial_p20s)
        lo = min(trial_p20s)
        hi = max(trial_p20s)
        tflops = flops / (med * 1e9)
        seed_data.append((seed, med, lo, hi, tflops))
        seed_meds.append(med)
        print(f"    seed={seed}  med={med:.4f}ms  lo={lo:.4f}  hi={hi:.4f}  TF={tflops:.1f}")
    overall_med = statistics.median(seed_meds)
    overall_lo = min(seed_meds)
    overall_hi = max(seed_meds)
    spread_pp = (overall_hi - overall_lo) / overall_med * 100
    overall_tflops = flops / (overall_med * 1e9)
    print(f"    OVERALL: med={overall_med:.4f}ms (TF={overall_tflops:.1f})  "
          f"lo={overall_lo:.4f}  hi={overall_hi:.4f}  spread={spread_pp:.3f}pp")
    return overall_med, overall_lo, overall_hi, overall_tflops


def correctness_check(B, M, gm_a, xcds_a, gm_b, xcds_b, slots=192):
    """Bit-equivalence check between two (gm, xcds) cells."""
    print(f"\n=== Correctness check Down-B{B}_M{M} wgrad: "
          f"(gm={gm_a}, xcds={xcds_a}) vs (gm={gm_b}, xcds={xcds_b}) @ slots={slots} ===")
    for seed in [42, 137]:
        torch.manual_seed(seed)
        call_a, out_a, grad, a, g_s, a_s, g_offs = make_call(B, M, slots, gm_a, xcds_a)
        # Run cell A
        out_a.zero_()
        call_a()
        out_a_clone = out_a.clone()
        # Run cell B with same inputs
        hk = hipkit_module.load_fp8()
        out_b = torch.empty_like(out_a)
        out_b.zero_()
        hk.grouped_variable_k_crr_dscale(
            grad, a, out_b, g_s, a_s, g_offs,
            group_m=gm_b, num_xcds=xcds_b, num_slots=slots,
        )
        max_abs = (out_a_clone.float() - out_b.float()).abs().max().item()
        print(f"    seed={seed}  max_abs_diff={max_abs}  bit_eq={'True' if max_abs == 0 else 'False'}")


if __name__ == "__main__":
    print("[probe] Round 15 tight verify — (1, 4) vs (1, 2) at slots=192")
    print(f"[probe] HIP_VISIBLE_DEVICES={os.environ.get('HIP_VISIBLE_DEVICES', '<unset>')}")
    print(f"[probe] device: {torch.cuda.get_device_name(0)}")
    t0 = time.monotonic()

    # Anchor: Down-B4-M4096 wgrad
    print("\n>>> Down-B4-M4096 wgrad, m_total=16384")
    res_b4_m4096_12 = run_one(4, 4096, 1, 2)
    res_b4_m4096_14 = run_one(4, 4096, 1, 4)
    res_b4_m4096_18 = run_one(4, 4096, 1, 8)

    # Sibling: Down-B4-M2048 wgrad (R11)
    print("\n>>> Down-B4-M2048 wgrad, m_total=8192")
    res_b4_m2048_12 = run_one(4, 2048, 1, 2)
    res_b4_m2048_14 = run_one(4, 2048, 1, 4)

    # Compare
    print("\n=== Summary (slots=192) ===")
    print(f"  Down-B4-M4096:  (1, 2) {res_b4_m4096_12[3]:.1f} TF  vs  "
          f"(1, 4) {res_b4_m4096_14[3]:.1f} TF  vs  (1, 8) {res_b4_m4096_18[3]:.1f} TF")
    print(f"     Δ(1,4) = {(res_b4_m4096_12[0] - res_b4_m4096_14[0])/res_b4_m4096_12[0]*100:+.2f}%")
    print(f"  Down-B4-M2048:  (1, 2) {res_b4_m2048_12[3]:.1f} TF  vs  "
          f"(1, 4) {res_b4_m2048_14[3]:.1f} TF")
    print(f"     Δ(1,4) = {(res_b4_m2048_12[0] - res_b4_m2048_14[0])/res_b4_m2048_12[0]*100:+.2f}%")

    # Bit-equivalence
    correctness_check(4, 4096, 1, 2, 1, 4)

    print(f"\n[probe] wall {time.monotonic()-t0:.1f}s")
