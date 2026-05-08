#!/usr/bin/env python3
"""Round 15 (gpt_oss FP8 kernel-only ceiling) — fine-grained slots sweep.

R3 shipped slots=192 for the Down-B4 wgrad family (m_total<=16384, a==2880,
b==2880). R2's coarse sweep was {32, 64, 96, 128, 160, 192, 256, 384}. The
optimum landed at slots=192 by ~+6% over slots=256. But the gap between
192 and 256 was not finely sampled — slots ∈ {200, 208, 216, 224, 240}
might hold a better optimum.

This probe:
  * Sweeps slots ∈ {160, 176, 184, 192, 200, 208, 216, 224, 240, 256}
  * On the worst metric shape: Down-B4-M2048 wgrad (m_total=8192)
  * Sibling: Down-B4-M4096 wgrad (m_total=16384) — verify same optimum
  * Methodology: 200-iter × 5-trial p20 × 3 seeds, kernel-only direct call
  * Uses the current R11 (gm=1, xcds=2) cell

If a finer slots value beats 192 by >= 0.5% on median across seeds AND
spread is < median lift, ship it. Otherwise, falsification.
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
    return times[len(times) // 5]  # p20


def make_wgrad_call(B, M, N, K, slots, gm, xcds):
    """Build a kernel-only var-K wgrad call with the given slots/gm/xcds."""
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
    return _call, B, M, N, K


def run_shape(B, M, N, K, gm=1, xcds=2):
    SHAPE = f"B{B}_M{M}_N{N}_K{K}"
    flops = 2.0 * (B * M) * N * K  # kernel FLOPs

    # Gold reference (slots=192, current rule) — establish baseline
    SLOTS_LIST = [160, 176, 184, 192, 200, 208, 216, 224, 240, 256]

    print(f"\n=== Down-{SHAPE} wgrad var-K @ (gm={gm}, xcds={xcds}) ===")
    print(f"  {'slots':>6}  {'med ms':>8}  {'min ms':>8}  {'max ms':>8}  {'TFLOPS':>8}  {'Δ%':>6}")
    results = {}
    SEEDS = [42, 137, 2024]
    REPEATS = 5
    for slots in SLOTS_LIST:
        seed_meds = []
        for seed in SEEDS:
            torch.manual_seed(seed)
            call, _, _, _, _ = make_wgrad_call(B, M, N, K, slots, gm, xcds)
            ms_per_iter = []
            for _ in range(REPEATS):
                ms_per_iter.append(_bench(call, warmup=10, iters=200))
            seed_meds.append(statistics.median(ms_per_iter))
        med_med = statistics.median(seed_meds)
        results[slots] = (med_med, min(seed_meds), max(seed_meds))

    # Find baseline and print
    base_ms = results[192][0]
    base_tflops = flops / (base_ms * 1e9)
    for slots, (med, lo, hi) in results.items():
        tflops = flops / (med * 1e9)
        delta_pp = (base_ms - med) / base_ms * 100  # +% means faster
        marker = " *base" if slots == 192 else ""
        print(f"  {slots:>6}  {med:>8.4f}  {lo:>8.4f}  {hi:>8.4f}  {tflops:>8.1f}  {delta_pp:+6.2f}{marker}")
    return results


if __name__ == "__main__":
    print(f"[probe] Round 15 fine-grained slots sweep on Down-B4 wgrad")
    print(f"[probe] HIP_VISIBLE_DEVICES={os.environ.get('HIP_VISIBLE_DEVICES', '<unset>')}")
    print(f"[probe] device: {torch.cuda.get_device_name(0)}")

    # Anchor (worst gpt_oss wgrad shape per metric)
    t0 = time.monotonic()
    run_shape(B=4, M=2048, N=2880, K=2880, gm=1, xcds=2)
    # Sibling (also benefits from R3 slots=192)
    run_shape(B=4, M=4096, N=2880, K=2880, gm=1, xcds=2)
    print(f"[probe] wall {time.monotonic()-t0:.1f}s")
