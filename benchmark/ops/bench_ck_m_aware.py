#!/usr/bin/env python3
"""Benchmark CK Grouped GEMM with M-aware tile selection (128x128 vs 256x256)."""

import json
import torch
import torch.utils.benchmark as benchmark

import primus_turbo.pytorch as turbo
from primus_turbo.pytorch.core.backend import BackendType, GlobalBackendManager, PrecisionType

CASES = [
    # Small M cases (should benefit from 128x128 tiles)
    {"label": "DSv3 B=8 M=512 N=4096 K=7168",     "B": 8,  "M": 512,   "N": 4096,  "K": 7168},
    {"label": "DSv3 B=8 M=1024 N=4096 K=7168",    "B": 8,  "M": 1024,  "N": 4096,  "K": 7168},
    {"label": "DSv3 B=16 M=512 N=4096 K=7168",    "B": 16, "M": 512,   "N": 4096,  "K": 7168},
    {"label": "DSv3 B=16 M=1024 N=4096 K=7168",   "B": 16, "M": 1024,  "N": 4096,  "K": 7168},
    {"label": "DSv3 B=32 M=512 N=4096 K=7168",    "B": 32, "M": 512,   "N": 4096,  "K": 7168},
    {"label": "DSv3-Down B=8 M=512 N=7168 K=2048", "B": 8,  "M": 512,   "N": 7168,  "K": 2048},
    {"label": "DSv3-Down B=8 M=1024 N=7168 K=2048","B": 8,  "M": 1024,  "N": 7168,  "K": 2048},
    {"label": "DSv2L B=2 M=512 N=2816 K=2048",    "B": 2,  "M": 512,   "N": 2816,  "K": 2048},
    {"label": "DSv2L B=2 M=1024 N=2816 K=2048",   "B": 2,  "M": 1024,  "N": 2816,  "K": 2048},
    {"label": "DSv2L B=4 M=512 N=2816 K=2048",    "B": 4,  "M": 512,   "N": 2816,  "K": 2048},
    {"label": "DSv2L B=4 M=1024 N=2816 K=2048",   "B": 4,  "M": 1024,  "N": 2816,  "K": 2048},
    {"label": "DSv2L B=8 M=512 N=2816 K=2048",    "B": 8,  "M": 512,   "N": 2816,  "K": 2048},
    {"label": "Qwen3-30B B=4 M=512 N=4096 K=2048", "B": 4,  "M": 512,   "N": 4096,  "K": 2048},
    {"label": "Qwen3-30B B=8 M=512 N=4096 K=2048", "B": 8,  "M": 512,   "N": 4096,  "K": 2048},
    {"label": "Qwen3-30B B=16 M=512 N=4096 K=2048","B": 16, "M": 512,   "N": 4096,  "K": 2048},
    {"label": "Kimi-K2 B=12 M=512 N=4096 K=7168",  "B": 12, "M": 512,   "N": 4096,  "K": 7168},
    {"label": "Kimi-K2 B=24 M=512 N=4096 K=7168",  "B": 24, "M": 512,   "N": 4096,  "K": 7168},
    {"label": "Kimi-K2 B=48 M=512 N=4096 K=7168",  "B": 48, "M": 512,   "N": 4096,  "K": 7168},
    {"label": "MoE-1T B=7 M=512 N=3840 K=8192",    "B": 7,  "M": 512,   "N": 3840,  "K": 8192},
    {"label": "MoE-1T B=14 M=512 N=3840 K=8192",   "B": 14, "M": 512,   "N": 3840,  "K": 8192},
    {"label": "MoE-1T B=28 M=512 N=3840 K=8192",   "B": 28, "M": 512,   "N": 3840,  "K": 8192},

    # Large M cases (should NOT regress, still use 256x256)
    {"label": "DSv3 B=8 M=4096 N=4096 K=7168",    "B": 8,  "M": 4096,  "N": 4096,  "K": 7168},
    {"label": "DSv3 B=8 M=16384 N=4096 K=7168",   "B": 8,  "M": 16384, "N": 4096,  "K": 7168},
    {"label": "DSv3 B=32 M=4096 N=4096 K=7168",   "B": 32, "M": 4096,  "N": 4096,  "K": 7168},
    {"label": "DSv2L B=2 M=8192 N=2816 K=2048",   "B": 2,  "M": 8192,  "N": 2816,  "K": 2048},
    {"label": "DSv2L B=2 M=16384 N=2816 K=2048",  "B": 2,  "M": 16384, "N": 2816,  "K": 2048},
    {"label": "Qwen3-30B B=4 M=8192 N=4096 K=2048","B": 4,  "M": 8192,  "N": 4096,  "K": 2048},
    {"label": "Kimi-K2 B=12 M=8192 N=4096 K=7168", "B": 12, "M": 8192,  "N": 4096,  "K": 7168},
]


def run_case(B, M, N, K):
    device = "cuda"
    dtype = torch.bfloat16
    x = torch.randn((B * M, K), dtype=dtype, device=device)
    w = torch.randn((B, N, K), dtype=dtype, device=device)
    group_lens = torch.full((B,), M, dtype=torch.long, device=device)

    fn = lambda: turbo.ops.grouped_gemm(x, w, group_lens, trans_b=True)

    for _ in range(5):
        fn()
    torch.cuda.synchronize()

    timer = benchmark.Timer(stmt="fn()", globals={"fn": fn})
    measurement = timer.timeit(50)
    mean_ms = measurement.mean * 1e3
    tflops = 2 * B * M * N * K / (mean_ms * 1e-3) / 1e12
    return round(mean_ms, 3), round(tflops, 2)


def main():
    GlobalBackendManager.set_grouped_gemm_backend(BackendType.CK, PrecisionType.BF16_FP16_FP32)

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"{'Case':<42} {'ms':>8} {'TFLOPS':>8}")
    print("-" * 62)

    results = []
    for case in CASES:
        B, M, N, K = case["B"], case["M"], case["N"], case["K"]
        ms, tf = run_case(B, M, N, K)
        print(f"{case['label']:<42} {ms:>8.3f} {tf:>8.2f}")
        results.append({"label": case["label"], "B": B, "M": M, "N": N, "K": K,
                        "mean_ms": ms, "tflops": tf})

    with open("benchmark/baselines/grouped_gemm_optimized_ck.json", "w") as f:
        json.dump({"operator": "grouped_gemm_ck_optimized", "gpu": "MI300X",
                   "results": results}, f, indent=2)
    print(f"\nResults saved to benchmark/baselines/grouped_gemm_optimized_ck.json")


if __name__ == "__main__":
    main()
