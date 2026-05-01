#!/usr/bin/env python3
"""Round-1 FP8 grouped probe: DSV3-GateUP (gm, xcd) sweep.

Shapes (tiles_n=16, tiles_m ∈ {8, 16}, k=7168, m_total ∈ {32768, 65536, 131072}):
  - DeepSeek-V3-GateUP-B16-M2048  (B=16, M_per=2048, N=4096, K=7168) ratio 1.024
  - DeepSeek-V3-GateUP-B16-M4096  (B=16, M_per=4096, N=4096, K=7168) ratio 1.033
  - DeepSeek-V3-GateUP-B32-M2048  (B=32, M_per=2048, N=4096, K=7168) ratio 1.017
  - DeepSeek-V3-GateUP-B32-M4096  (B=32, M_per=4096, N=4096, K=7168) ratio 1.060

Default is (gm=4, xcd=None=8). Sweep a 7-cell candidate set. We're looking for
a kernel-level win >=0.5pp (anchored against measurement noise band ~0.2-0.3pp
at REPEATS=5 / ITERS=80 p20).

Only prints the summary — saves no cache. Run on pinned GPU (HIP_VISIBLE_DEVICES).
"""
from __future__ import annotations

import os
import statistics
import sys
import time

os.environ.setdefault("PRIMUS_TURBO_HIPKITTEN_PATH", "/workspace/code/HipKittens")

# Ensure Primus can find HipKittens modules
_SCRIPTS = os.path.abspath("/workspace/code/Primus-Turbo/scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

import torch  # noqa: E402

from primus_turbo.pytorch.core.backend import BackendType, PrecisionType  # noqa: E402
from primus_turbo.pytorch.core.low_precision import (  # noqa: E402
    Float8QuantConfig,
    Format,
    ScalingGranularity,
)
import primus_turbo.pytorch as turbo  # noqa: E402
import _metric_hk_ratio as hk_ratio  # noqa: E402

from primus_turbo.pytorch.kernels.hipkitten import config as hk_config  # noqa: E402


DSV3_GATEUP_SHAPES = [
    ("DeepSeek-V3-GateUP-B16-M2048", 16, 2048, 4096, 7168),
    ("DeepSeek-V3-GateUP-B16-M4096", 16, 4096, 4096, 7168),
    ("DeepSeek-V3-GateUP-B32-M2048", 32, 2048, 4096, 7168),
    ("DeepSeek-V3-GateUP-B32-M4096", 32, 4096, 4096, 7168),
]

CANDIDATES = [
    # (gm, xcd)
    (4, None),  # current default
    (4, 4),
    (4, 2),
    (2, 4),
    (2, 8),
    (1, 4),
    (1, 2),
    (8, 4),
]

ITERS = 80
REPEATS = 5


def bench_grouped_fp8(B, M, N, K, gm, xcd, backend=BackendType.HIPKITTEN) -> float:
    """Run fwd of grouped_gemm_fp8 on (B, M, N, K) with (gm, xcd) override."""
    torch.manual_seed(0)
    cfg = Float8QuantConfig(format=Format.E4M3, granularity=ScalingGranularity.TENSORWISE)
    a = torch.randn((B * M, K), dtype=torch.bfloat16, device="cuda")
    b = torch.randn((B, N, K), dtype=torch.bfloat16, device="cuda")
    group_lens = torch.full((B,), M, dtype=torch.int64, device="cuda")

    # Override the config.select_default_config for this shape only
    orig_select = hk_config.select_default_config

    def patched_select(m, n, k, layout, dtype, m_total=None):
        if (
            dtype == "fp8"
            and layout == "rcr"
            and n == 4096
            and k == 7168
            and m_total == B * M
        ):
            return hk_config.HipKittenConfig(
                layout=layout,
                group_m=gm,
                num_xcds=xcd,
                kernel=None,
            )
        return orig_select(m, n, k, layout, dtype, m_total=m_total)

    hk_config.select_default_config = patched_select
    try:
        with hk_ratio.force_grouped_gemm_backend(backend, PrecisionType.FP8):
            # Warmup
            for _ in range(10):
                _ = turbo.ops.grouped_gemm_fp8(a, b, group_lens, trans_b=True, config=cfg)
            torch.cuda.synchronize()

            # Timed
            tflops_samples = []
            for _ in range(REPEATS):
                torch.cuda.synchronize()
                t0 = time.perf_counter_ns()
                for _ in range(ITERS):
                    _ = turbo.ops.grouped_gemm_fp8(a, b, group_lens, trans_b=True, config=cfg)
                torch.cuda.synchronize()
                elapsed_s = (time.perf_counter_ns() - t0) / 1e9 / ITERS
                # 2 * M_total * N * K FLOPS per GEMM
                flops = 2 * (B * M) * N * K
                tflops = flops / elapsed_s / 1e12
                tflops_samples.append(tflops)
            # Use p20 (min-ish) - safer than median for microbench
            tflops_samples.sort()
            p20 = tflops_samples[0]  # with REPEATS=5, use min as conservative
            return p20
    finally:
        hk_config.select_default_config = orig_select


def main():
    if not torch.cuda.is_available():
        print("CUDA not available", file=sys.stderr)
        return 1

    print(f"[probe_dsv3_gateup_fp8_round1] ITERS={ITERS} REPEATS={REPEATS}")
    print(f"{'shape':42s}  {'config':>12s}  {'tflops':>10s}  {'Δ vs (4,None)':>14s}")
    for name, B, M, N, K in DSV3_GATEUP_SHAPES:
        baseline = None
        lines = []
        for gm, xcd in CANDIDATES:
            try:
                tflops = bench_grouped_fp8(B, M, N, K, gm, xcd)
            except Exception as e:
                lines.append((gm, xcd, float("nan"), str(e)))
                continue
            if (gm, xcd) == (4, None):
                baseline = tflops
            lines.append((gm, xcd, tflops, ""))
        for gm, xcd, tf, err in lines:
            cfg_str = f"(gm={gm},xcd={xcd})"
            if baseline and baseline > 0 and not err:
                delta_pct = (tf - baseline) / baseline * 100.0
                delta_str = f"{delta_pct:+.2f}%"
            else:
                delta_str = "?"
            if err:
                cfg_str = f"(gm={gm},xcd={xcd})  ERR:{err[:30]}"
            print(f"{name:42s}  {cfg_str:>12s}  {tf:>10.2f}  {delta_str:>14s}")
        print()


if __name__ == "__main__":
    raise SystemExit(main())
