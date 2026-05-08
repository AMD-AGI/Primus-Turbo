#!/usr/bin/env python3
"""Round 17 (gpt_oss FP8 kernel-only ceiling) — Down-B4-M2048 fwd RCR
(gm, xcds) drift re-sweep.

The current rule (R2 of this run, line 2058 of config.py) is (gm=16, xcds=2)
based on a tight verify done before R3 (slots=192 for var-K) and R4-R16
landed. Multiple kernel rebuilds since then may have shifted the optimum
(see R30/R31/R32/R45/R50 kernel-rebuild-drift commentary in config.py).

Probe sweeps an 8×4 grid:
  gm ∈ {1, 2, 4, 8, 16, 32}, xcds ∈ {2, 4, 8}
plus the current rule (16, 2) and a defensive (1, 2) baseline.

If any cell beats (16, 2) by >= 0.5pp with median > spread × 5 (R15
robustness threshold) and 3/3 seeds positive, run a tight verify.
"""
import os
import sys
import statistics
import time

os.environ.setdefault("PRIMUS_TURBO_HIPKITTEN_PATH", "/workspace/code/HipKittens")
sys.path.insert(0, "/workspace/code/Primus-Turbo")

import torch  # noqa: E402

import primus_turbo.pytorch as turbo  # noqa: F401  E402
from primus_turbo.pytorch.core.low_precision import ScalingGranularity  # noqa: E402
from primus_turbo.pytorch.kernels import hipkitten as hipkit_module  # noqa: E402
from primus_turbo.pytorch.kernels.grouped_gemm.grouped_gemm_fp8_impl import (  # noqa: E402
    grouped_gemm_compute_offs,
)
from primus_turbo.pytorch.ops.quantization import quantize_fp8  # noqa: E402

_FP8_DTYPE = torch.float8_e4m3fnuz
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


def make_call(B, M, N, K, gm, xcds):
    g_lens = torch.full((B,), M, dtype=torch.int64, device="cuda")
    g_offs = grouped_gemm_compute_offs(g_lens)
    a = torch.randn((B * M, K), dtype=torch.bfloat16, device="cuda")
    b = torch.randn((B, N, K), dtype=torch.bfloat16, device="cuda")
    a_fp8, a_s = quantize_fp8(a, _FP8_DTYPE, _GRAN)
    b_fp8, b_s = quantize_fp8(b, _FP8_DTYPE, _GRAN)
    out = torch.empty((B * M, N), dtype=torch.bfloat16, device="cuda")
    hk = hipkit_module.load_fp8()

    def _call():
        hk.grouped_rcr_dscale(
            a_fp8, b_fp8, out, a_s, b_s, g_offs, gm,
            m_per_group=M, num_xcds=xcds,
        )
    return _call


def run_shape(name, B, M, N, K, baseline_cfg=(16, 2)):
    flops = 2.0 * (B * M) * N * K
    print(f"\n=== {name} fwd RCR (B={B} M={M} N={N} K={K}) ===")
    print(f"  {'cfg':>10}  {'med ms':>9}  {'min ms':>9}  {'TFLOPS':>8}  {'spread':>7}  {'Δpp vs cur':>10}")
    cells = [
        (1, 2), (1, 4), (1, 8),
        (2, 2), (2, 4), (2, 8),
        (4, 2), (4, 4), (4, 8),
        (8, 2), (8, 4), (8, 8),
        (16, 2), (16, 4), (16, 8),
        (32, 2), (32, 4),
    ]
    SEEDS = [42, 137, 2024]
    REPEATS = 3
    results = {}
    for (gm, xc) in cells:
        seed_meds = []
        for seed in SEEDS:
            torch.manual_seed(seed)
            call = make_call(B, M, N, K, gm, xc)
            ms_per_iter = []
            for _ in range(REPEATS):
                ms_per_iter.append(_bench(call, warmup=10, iters=200))
            seed_meds.append(statistics.median(ms_per_iter))
        results[(gm, xc)] = (
            statistics.median(seed_meds),
            min(seed_meds),
            max(seed_meds),
        )

    base_med = results[baseline_cfg][0]
    sorted_cells = sorted(results.items(), key=lambda x: x[1][0])
    for (gm, xc), (med, lo, hi) in sorted_cells:
        tflops = flops / (med * 1e9)
        spread = hi - lo
        delta_pp = (base_med - med) / base_med * 100
        marker = " *cur" if (gm, xc) == baseline_cfg else ""
        cell = f"({gm}, {xc})"
        print(f"  {cell:>10}  {med:>9.4f}  {lo:>9.4f}  {tflops:>8.1f}  {spread:>7.4f}  {delta_pp:+9.2f}{marker}")
    return results


if __name__ == "__main__":
    print(f"[probe] Round 17: Down-B4-M2048 fwd RCR (gm, xcds) drift re-sweep")
    print(f"[probe] HIP_VISIBLE_DEVICES={os.environ.get('HIP_VISIBLE_DEVICES', '<unset>')}")
    print(f"[probe] device: {torch.cuda.get_device_name(0)}")
    t0 = time.monotonic()
    run_shape("gpt_oss-Down-B4-M2048", B=4, M=2048, N=2880, K=2880, baseline_cfg=(16, 2))
    run_shape("gpt_oss-Down-B4-M4096", B=4, M=4096, N=2880, K=2880, baseline_cfg=(1, 4))
    run_shape("gpt_oss-GateUP-B4-M2048", B=4, M=2048, N=5760, K=2880, baseline_cfg=(1, 4))
    run_shape("gpt_oss-Down-B32-M2048", B=32, M=2048, N=2880, K=2880, baseline_cfg=(16, 4))
    run_shape("gpt_oss-Down-B32-M4096", B=32, M=4096, N=2880, K=2880, baseline_cfg=(4, 8))
    print(f"[probe] wall {time.monotonic()-t0:.1f}s")
