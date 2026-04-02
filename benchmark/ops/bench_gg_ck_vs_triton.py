#!/usr/bin/env python3
"""Quick CK vs Triton comparison for Grouped GEMM on representative configs."""

import os
import torch
import torch.utils.benchmark as benchmark

import primus_turbo.pytorch as turbo
from primus_turbo.pytorch.core.backend import BackendType, GlobalBackendManager, PrecisionType

CASES = [
    {"name": "DSv3-GateUP B=8 M=512",   "B": 8,  "M": 512,   "N": 4096, "K": 7168},
    {"name": "DSv3-GateUP B=8 M=4096",   "B": 8,  "M": 4096,  "N": 4096, "K": 7168},
    {"name": "DSv3-GateUP B=8 M=16384",  "B": 8,  "M": 16384, "N": 4096, "K": 7168},
    {"name": "DSv3-GateUP B=32 M=512",   "B": 32, "M": 512,   "N": 4096, "K": 7168},
    {"name": "DSv3-GateUP B=32 M=4096",  "B": 32, "M": 4096,  "N": 4096, "K": 7168},
    {"name": "DSv2-Lite B=2 M=512",      "B": 2,  "M": 512,   "N": 2816, "K": 2048},
    {"name": "DSv2-Lite B=2 M=4096",     "B": 2,  "M": 4096,  "N": 2816, "K": 2048},
    {"name": "Mixtral-8x7B B=1 M=4096",  "B": 1,  "M": 4096,  "N": 28672,"K": 4096},
    {"name": "Mixtral-8x7B B=1 M=16384", "B": 1,  "M": 16384, "N": 28672,"K": 4096},
    {"name": "Qwen3-30B B=4 M=512",      "B": 4,  "M": 512,   "N": 4096, "K": 2048},
    {"name": "Qwen3-30B B=4 M=4096",     "B": 4,  "M": 4096,  "N": 4096, "K": 2048},
    {"name": "Kimi-K2 B=12 M=512",       "B": 12, "M": 512,   "N": 4096, "K": 7168},
    {"name": "Kimi-K2 B=12 M=8192",      "B": 12, "M": 8192,  "N": 4096, "K": 7168},
]

def run_case(B, M, N, K, backend_name):
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
    return mean_ms, tflops

def main():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"{'Case':<35} {'CK ms':>8} {'CK TF':>8} {'Tri ms':>8} {'Tri TF':>8} {'Speedup':>8}")
    print("-" * 90)

    for case in CASES:
        B, M, N, K = case["B"], case["M"], case["N"], case["K"]
        if B == 1:
            print(f"{case['name']:<35} {'(B=1 → hipBLASLt, skip)':>50}")
            continue

        GlobalBackendManager.set_grouped_gemm_backend(BackendType.CK, PrecisionType.BF16_FP16_FP32)
        ck_ms, ck_tf = run_case(B, M, N, K, "CK")

        GlobalBackendManager.set_grouped_gemm_backend(BackendType.TRITON, PrecisionType.BF16_FP16_FP32)
        tri_ms, tri_tf = run_case(B, M, N, K, "TRITON")

        speedup = ck_ms / tri_ms
        marker = "✓" if speedup > 1.0 else "✗"
        print(f"{case['name']:<35} {ck_ms:>7.2f}  {ck_tf:>7.1f}  {tri_ms:>7.2f}  {tri_tf:>7.1f}  {speedup:>6.2f}x {marker}")

    os.environ.pop("PRIMUS_TURBO_GROUPED_GEMM_BACKEND", None)

if __name__ == "__main__":
    main()
