#!/usr/bin/env python3
"""Benchmark CK FP8 Grouped GEMM with M-aware tile selection."""

import json
import torch
import torch.utils.benchmark as benchmark

from primus_turbo.pytorch.core.backend import BackendType, GlobalBackendManager, PrecisionType
from primus_turbo.pytorch.core.low_precision import Float8QuantConfig, Format, ScalingGranularity
from primus_turbo.pytorch.ops import grouped_gemm_fp8

CASES = [
    # Small M (M-aware should trigger 128x128 for RowColQuant)
    {"label": "DSv3 B=8 M=512 N=4096 K=7168",      "B": 8,  "M": 512,  "N": 4096, "K": 7168},
    {"label": "DSv3 B=16 M=512 N=4096 K=7168",      "B": 16, "M": 512,  "N": 4096, "K": 7168},
    {"label": "DSv3 B=8 M=1024 N=4096 K=7168",      "B": 8,  "M": 1024, "N": 4096, "K": 7168},
    {"label": "DSv3-Down B=8 M=512 N=7168 K=2048",  "B": 8,  "M": 512,  "N": 7168, "K": 2048},
    {"label": "DSv2L B=2 M=512 N=2816 K=2048",      "B": 2,  "M": 512,  "N": 2816, "K": 2048},
    {"label": "DSv2L B=8 M=512 N=2816 K=2048",      "B": 8,  "M": 512,  "N": 2816, "K": 2048},
    {"label": "Qwen3-30B B=8 M=512 N=4096 K=2048",  "B": 8,  "M": 512,  "N": 4096, "K": 2048},
    {"label": "Kimi-K2 B=12 M=512 N=4096 K=7168",   "B": 12, "M": 512,  "N": 4096, "K": 7168},
    {"label": "Kimi-K2 B=24 M=512 N=4096 K=7168",   "B": 24, "M": 512,  "N": 4096, "K": 7168},
    # Large M (should NOT regress)
    {"label": "DSv3 B=8 M=4096 N=4096 K=7168",      "B": 8,  "M": 4096, "N": 4096, "K": 7168},
    {"label": "DSv3 B=8 M=16384 N=4096 K=7168",     "B": 8,  "M": 16384,"N": 4096, "K": 7168},
    {"label": "DSv2L B=2 M=8192 N=2816 K=2048",     "B": 2,  "M": 8192, "N": 2816, "K": 2048},
    {"label": "Kimi-K2 B=12 M=8192 N=4096 K=7168",  "B": 12, "M": 8192, "N": 4096, "K": 7168},
]


def run_case(B, M, N, K, quant_granularity):
    device = "cuda"
    dtype = torch.bfloat16

    x = torch.randn((B * M, K), dtype=dtype, device=device)
    w = torch.randn((B, N, K), dtype=dtype, device=device)
    group_lens = torch.full((B,), M, dtype=torch.long, device=device)

    config = Float8QuantConfig(
        format=Format.E4M3,
        granularity=quant_granularity,
        block_size=128 if quant_granularity == ScalingGranularity.BLOCKWISE else None,
    )

    fn = lambda: grouped_gemm_fp8(x, w, group_lens, trans_b=True, config=config)

    for _ in range(5):
        fn()
    torch.cuda.synchronize()

    timer = benchmark.Timer(stmt="fn()", globals={"fn": fn})
    measurement = timer.timeit(50)
    mean_ms = measurement.mean * 1e3
    tflops = 2 * B * M * N * K / (mean_ms * 1e-3) / 1e12
    return round(mean_ms, 3), round(tflops, 2)


def main():
    GlobalBackendManager.set_grouped_gemm_backend(BackendType.CK, PrecisionType.FP8)
    gpu = torch.cuda.get_device_name(0)
    print(f"GPU: {gpu}")

    results = {}
    for granularity_name, granularity in [
        ("RowColQuant", ScalingGranularity.ROWWISE),
        ("ABQuantGrouped", ScalingGranularity.BLOCKWISE),
    ]:
        print(f"\n=== {granularity_name} ===")
        print(f"{'Case':<42} {'ms':>8} {'TFLOPS':>8}")
        print("-" * 62)

        results[granularity_name] = []
        for case in CASES:
            B, M, N, K = case["B"], case["M"], case["N"], case["K"]
            ms, tf = run_case(B, M, N, K, granularity)
            print(f"{case['label']:<42} {ms:>8.3f} {tf:>8.2f}")
            results[granularity_name].append({
                "label": case["label"], "B": B, "M": M, "N": N, "K": K,
                "mean_ms": ms, "tflops": tf,
            })

    out_path = "benchmark/baselines/grouped_gemm_fp8_optimized_ck.json"
    with open(out_path, "w") as f:
        json.dump({"operator": "grouped_gemm_fp8_ck", "gpu": gpu, "results": results}, f, indent=2)
    print(f"\nResults saved to {out_path}")

    GlobalBackendManager.set_grouped_gemm_backend(None, PrecisionType.FP8)


if __name__ == "__main__":
    main()
