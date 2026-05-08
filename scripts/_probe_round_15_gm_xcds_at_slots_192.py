#!/usr/bin/env python3
"""Round 15 (gpt_oss FP8 kernel-only ceiling) — (gm, xcds) at slots=192.

R11 sweep was at slots=256 (no R3 lever). With slots=192 in production for
Down-B4 wgrad, the optimum (gm, xcds) cell may have shifted due to the
different persistent-grid topology (192 slots × 1.4 wave-steps/slot vs 256
slots × 1.05 wave-steps/slot).

Probe sweeps {(1, 2), (1, 4), (1, 8), (2, 2), (4, 2), (8, 2), (16, 2),
(32, 2), (16, 4), (32, 4)} at slots=192 fixed on:
  * Down-B4-M2048 wgrad (R11 + R3 cell)
  * Down-B4-M4096 wgrad (R10 + R3 cell)

If any cell beats (1, 2) by >= 0.5% with median > spread, ship it.
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


def _bench(fn, warmup=10, iters=200):
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


def make_call(B, M, N, K, slots, gm, xcds):
    g_lens = torch.full((B,), M, dtype=torch.int64, device="cuda")
    from primus_turbo.pytorch.kernels.grouped_gemm.grouped_gemm_fp8_impl import (
        grouped_gemm_compute_offs,
    )
    g_offs = grouped_gemm_compute_offs(g_lens)
    a = torch.randn((B * M, K), dtype=torch.bfloat16, device="cuda")
    grad = torch.randn((B * M, N), dtype=torch.bfloat16, device="cuda")
    a_col, a_s = quantize_fp8(a, _FP8_DTYPE, _GRAN, axis=-2)
    g_col, g_s = quantize_fp8(grad, _FP8_DTYPE, _GRAN, axis=-2)
    hk = hipkit_module.load_fp8()
    out = torch.empty((B, N, K), dtype=torch.bfloat16, device="cuda")

    def _call():
        hk.grouped_variable_k_crr_dscale(
            grad, a, out, g_s, a_s, g_offs,
            group_m=gm, num_xcds=xcds, num_slots=slots,
        )
    return _call


def run_shape(B, M, slots=192):
    SHAPE = f"B{B}_M{M}_N2880_K2880"
    flops = 2.0 * (B * M) * 2880 * 2880
    print(f"\n=== Down-{SHAPE} wgrad var-K @ slots={slots}, multi-(gm, xcds) ===")
    print(f"  {'cfg':>10}  {'med ms':>9}  {'min ms':>9}  {'TFLOPS':>8}  {'Δ%':>6}")
    cells = [(1, 2), (1, 4), (1, 8), (2, 2), (4, 2), (8, 2), (16, 2),
             (32, 2), (16, 4), (32, 4), (1, 16)]
    SEEDS = [42, 137, 2024]
    REPEATS = 5
    results = {}
    for (gm, xc) in cells:
        seed_meds = []
        for seed in SEEDS:
            torch.manual_seed(seed)
            call = make_call(B, M, 2880, 2880, slots, gm, xc)
            ms_per_iter = []
            for _ in range(REPEATS):
                ms_per_iter.append(_bench(call, warmup=10, iters=200))
            seed_meds.append(statistics.median(ms_per_iter))
        results[(gm, xc)] = (statistics.median(seed_meds),
                              min(seed_meds), max(seed_meds))

    base = results[(1, 2)][0]
    base_tflops = flops / (base * 1e9)
    # Sort by speed (median ms) ascending
    sorted_cells = sorted(results.items(), key=lambda x: x[1][0])
    for (gm, xc), (med, lo, hi) in sorted_cells:
        tflops = flops / (med * 1e9)
        delta_pp = (base - med) / base * 100
        marker = " *cur(R11)" if (gm, xc) == (1, 2) else ""
        cell = f"({gm}, {xc})"
        print(f"  {cell:>10}  {med:>9.4f}  {lo:>9.4f}  {tflops:>8.1f}  {delta_pp:+6.2f}{marker}")
    return results


if __name__ == "__main__":
    print(f"[probe] Round 15 (gm, xcds) sweep at slots=192 on Down-B4 wgrad")
    print(f"[probe] HIP_VISIBLE_DEVICES={os.environ.get('HIP_VISIBLE_DEVICES', '<unset>')}")
    print(f"[probe] device: {torch.cuda.get_device_name(0)}")
    t0 = time.monotonic()
    run_shape(B=4, M=2048, slots=192)
    run_shape(B=4, M=4096, slots=192)
    print(f"[probe] wall {time.monotonic()-t0:.1f}s")
