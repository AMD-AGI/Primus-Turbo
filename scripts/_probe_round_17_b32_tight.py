#!/usr/bin/env python3
"""R17 tight verify — Down-B32 fwd RCR drift candidates.

Coarse sweep (200-iter × 3-repeat × 3-seed) flagged:
  Down-B32-M2048: (32, 4) +0.58pp vs current (16, 4)
  Down-B32-M4096: (4, 4)  +0.50pp vs default (4, 8)
                  (8, 4)  +0.40pp vs default (4, 8)

Tight verify with 1500-iter × 7-repeat × 3-seed p20 (R7/R10 convention).
Ship rule iff median win > 0.5pp AND median > spread × 5 (R15 robustness).
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


def _bench(fn, warmup=20, iters=1500):
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
    # p20
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


def tight_verify(name, B, M, N, K, cells, baseline):
    flops = 2.0 * (B * M) * N * K
    print(f"\n=== {name} TIGHT VERIFY (B={B} M={M} N={N} K={K}) ===")
    SEEDS = [42, 137, 2024]
    REPEATS = 7
    results = {}
    for cell in cells:
        gm, xc = cell
        seed_meds = []
        for seed in SEEDS:
            torch.manual_seed(seed)
            call = make_call(B, M, N, K, gm, xc)
            ms_per_iter = []
            for _ in range(REPEATS):
                ms_per_iter.append(_bench(call, warmup=20, iters=1500))
            seed_meds.append(statistics.median(ms_per_iter))
        results[cell] = (
            statistics.median(seed_meds),
            min(seed_meds),
            max(seed_meds),
            seed_meds,
        )

    base_med = results[baseline][0]
    print(f"  {'cfg':>10}  {'med ms':>9}  {'TFLOPS':>8}  {'spread':>7}  {'med/spread':>10}  {'Δpp':>6}  {'seeds':>20}")
    sorted_cells = sorted(results.items(), key=lambda x: x[1][0])
    for (gm, xc), (med, lo, hi, seeds) in sorted_cells:
        tflops = flops / (med * 1e9)
        spread = hi - lo
        med_over_spread = (med / spread) if spread > 0 else 999.0
        delta_pp = (base_med - med) / base_med * 100
        marker = " *base" if (gm, xc) == baseline else ""
        cell_str = f"({gm}, {xc})"
        seed_str = "/".join(f"{s:.4f}" for s in seeds)
        print(f"  {cell_str:>10}  {med:>9.4f}  {tflops:>8.1f}  {spread:>7.4f}  {med_over_spread:>10.1f}  {delta_pp:+6.2f}{marker}  {seed_str}")
    return results


if __name__ == "__main__":
    print(f"[probe] R17 tight verify: B32 M2048/M4096 fwd RCR drift")
    print(f"[probe] device: {torch.cuda.get_device_name(0)}")
    t0 = time.monotonic()

    # Down-B32-M2048: tight-verify (16, 4) vs (32, 4) and (16, 2) and (32, 2)
    tight_verify(
        "gpt_oss-Down-B32-M2048", 32, 2048, 2880, 2880,
        cells=[(16, 4), (32, 4), (16, 2), (32, 2)],
        baseline=(16, 4),
    )

    # Down-B32-M4096: default is (gm=4, xcds=None=8). Tight-verify against
    # (4, 4), (8, 4), (1, 4).
    tight_verify(
        "gpt_oss-Down-B32-M4096", 32, 4096, 2880, 2880,
        cells=[(4, 8), (4, 4), (8, 4), (1, 4)],
        baseline=(4, 8),
    )

    print(f"\n[probe] wall {time.monotonic()-t0:.1f}s")
