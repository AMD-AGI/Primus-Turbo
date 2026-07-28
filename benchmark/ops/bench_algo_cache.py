###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Benchmark hipBLASLt algorithm cache impact on FP8 GEMM dispatch latency.

Measures cold-cache (first call, includes heuristic lookup) vs warm-cache
(subsequent calls, heuristic skipped) latency for model-realistic GEMM shapes.

Usage:
    cd benchmark/ops
    python bench_algo_cache.py [--warmup 5] [--repeat 50] [--output results.csv]
"""

import argparse
from datetime import datetime

import pandas as pd
import torch
from config import (
    BATCH_SIZE_LIST,
    DenseModelConfigs,
    gen_gemm_test_cases,
    get_platform_info,
)
from tabulate import tabulate

from primus_turbo.pytorch.core.backend import BackendType, GlobalBackendManager
from primus_turbo.pytorch.core.low_precision import (
    Float8QuantConfig,
    Format,
    ScalingGranularity,
)
from primus_turbo.pytorch.ops import gemm_fp8

DEVICE = "cuda:0"
DTYPE = torch.bfloat16


def bench_gemm_cache(m, n, k, config, warmup_iters, repeat):
    """Return (cold_ms, warm_median_ms, warm_min_ms) for a single GEMM shape."""
    a = torch.randn(m, k, dtype=DTYPE, device=DEVICE)
    b = torch.randn(n, k, dtype=DTYPE, device=DEVICE)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    gemm_fp8(a, b, False, True, DTYPE, config)
    end.record()
    torch.cuda.synchronize()
    cold_ms = start.elapsed_time(end)

    for _ in range(warmup_iters):
        gemm_fp8(a, b, False, True, DTYPE, config)
    torch.cuda.synchronize()

    timings = []
    for _ in range(repeat):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        gemm_fp8(a, b, False, True, DTYPE, config)
        e.record()
        torch.cuda.synchronize()
        timings.append(s.elapsed_time(e))

    timings.sort()
    warm_median = timings[len(timings) // 2]
    warm_min = timings[0]

    flops = 2 * m * n * k
    warm_tflops = flops / (warm_median * 1e-3) / 1e12
    cold_tflops = flops / (cold_ms * 1e-3) / 1e12

    return cold_ms, warm_median, warm_min, cold_tflops, warm_tflops


def benchmark_algo_cache(warmup_iters, repeat, output_csv):
    platform, gpu_name = get_platform_info()

    GlobalBackendManager.set_gemm_backend(BackendType.HIPBLASLT)
    GlobalBackendManager.set_auto_tune(False)

    config = Float8QuantConfig(granularity=ScalingGranularity.TENSORWISE, format=Format.E4M3)

    print(f"\nhipBLASLt Algo Cache Benchmark")
    print(f"Platform: {platform}, GPU: {gpu_name}")
    print(f"warmup={warmup_iters}, repeat={repeat}\n")

    _a = torch.randn(64, 64, dtype=DTYPE, device=DEVICE)
    _b = torch.randn(64, 64, dtype=DTYPE, device=DEVICE)
    gemm_fp8(_a, _b, False, True, DTYPE, config)
    torch.cuda.synchronize()
    del _a, _b

    rows = []
    test_id = 0

    for model_name, model_config in DenseModelConfigs.items():
        test_cases = gen_gemm_test_cases(model_config)
        for mbs in BATCH_SIZE_LIST:
            for shape in test_cases:
                test_id += 1
                m = shape[0] * mbs
                n = shape[1]
                k = shape[2]

                try:
                    cold, warm_med, warm_min, cold_tflops, warm_tflops = bench_gemm_cache(
                        m, n, k, config, warmup_iters, repeat
                    )
                    speedup = cold / warm_med if warm_med > 0 else float("inf")
                    overhead_saved_us = (cold - warm_med) * 1000

                    rows.append(
                        {
                            "TestID": test_id,
                            "Case": model_name,
                            "MBS": mbs,
                            "M": m,
                            "N": n,
                            "K": k,
                            "Cold (ms)": f"{cold:.3f}",
                            "Warm med (ms)": f"{warm_med:.3f}",
                            "Warm min (ms)": f"{warm_min:.3f}",
                            "Cold TFLOPS": f"{cold_tflops:.2f}",
                            "Warm TFLOPS": f"{warm_tflops:.2f}",
                            "Overhead saved (us)": f"{overhead_saved_us:.0f}",
                            "Speedup": f"{speedup:.1f}x",
                        }
                    )
                except Exception as e:
                    print(f"  FAILED ({model_name} MBS={mbs} {m}x{n}x{k}): {e}")
                    rows.append(
                        {
                            "TestID": test_id,
                            "Case": model_name,
                            "MBS": mbs,
                            "M": m,
                            "N": n,
                            "K": k,
                            "Cold (ms)": "ERROR",
                            "Warm med (ms)": "ERROR",
                            "Warm min (ms)": "ERROR",
                            "Cold TFLOPS": "0.00",
                            "Warm TFLOPS": "0.00",
                            "Overhead saved (us)": "0",
                            "Speedup": "N/A",
                        }
                    )

    results = pd.DataFrame(rows)
    print("\nResults:")
    print(tabulate(results, headers="keys", tablefmt="grid", showindex=False))

    overhead_vals = results["Overhead saved (us)"].apply(lambda x: float(x) if x != "0" else 0)
    print(f"\nMean overhead saved per GEMM call: {overhead_vals.mean():.0f} us")

    if output_csv:
        filename = output_csv
    else:
        timestamp = datetime.now().strftime("%Y%m%d")
        filename = f"algo_cache_{timestamp}_{gpu_name}.csv"
    results.to_csv(filename, index=False)
    print(f"Results saved to {filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark hipBLASLt algo cache")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=50)
    parser.add_argument("--output", "-o", type=str, default=None, help="Output CSV filename")
    args = parser.parse_args()

    benchmark_algo_cache(args.warmup, args.repeat, args.output)
