#!/usr/bin/env python3
"""Quick hipBLASLt vs Triton comparison for GEMM on representative configs."""

import torch
import torch.utils.benchmark as benchmark

import primus_turbo.pytorch as turbo
from primus_turbo.pytorch.core.backend import BackendType, GlobalBackendManager, PrecisionType

CASES = [
    {"name": "Llama2-7B M=4096 N=12288",   "M": 4096,  "N": 12288, "K": 4096},
    {"name": "Llama2-7B M=4096 N=4096",     "M": 4096,  "N": 4096,  "K": 4096},
    {"name": "Llama2-70B M=4096 N=8192",    "M": 4096,  "N": 8192,  "K": 28672},
    {"name": "Llama3.1-8B M=8192 N=28672",  "M": 8192,  "N": 28672, "K": 4096},
    {"name": "Llama3.1-405B M=8192 N=18432", "M": 8192, "N": 18432, "K": 16384},
    {"name": "Llama3.1-405B M=8192 N=106496","M": 8192, "N": 106496,"K": 16384},
    {"name": "Llama3.1-405B M=8192 K=53248", "M": 8192, "N": 16384, "K": 53248},
    {"name": "Qwen2.5-72B M=32K N=10240",   "M": 32768, "N": 10240, "K": 8192},
    {"name": "Qwen2.5-72B M=8192 K=29568",  "M": 8192,  "N": 8192,  "K": 29568},
    {"name": "Mistral-7B M=16384 K=14336",  "M": 16384, "N": 4096,  "K": 14336},
]

def run_case(M, N, K, backend_type):
    device = "cuda"
    dtype = torch.bfloat16
    a = torch.randn((M, K), dtype=dtype, device=device)
    b = torch.randn((N, K), dtype=dtype, device=device)
    GlobalBackendManager.set_gemm_backend(backend_type, PrecisionType.BF16_FP16_FP32)

    fn = lambda: turbo.ops.gemm(a, b, trans_b=True)
    for _ in range(5):
        fn()
    torch.cuda.synchronize()

    timer = benchmark.Timer(stmt="fn()", globals={"fn": fn})
    measurement = timer.timeit(50)
    mean_ms = measurement.mean * 1e3
    tflops = 2 * M * N * K / (mean_ms * 1e-3) / 1e12
    return mean_ms, tflops

def main():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"{'Case':<40} {'HBLt ms':>8} {'HBLt TF':>8} {'Tri ms':>8} {'Tri TF':>8} {'Tri/HBLt':>8}")
    print("-" * 95)

    for case in CASES:
        M, N, K = case["M"], case["N"], case["K"]
        hb_ms, hb_tf = run_case(M, N, K, BackendType.HIPBLASLT)
        tri_ms, tri_tf = run_case(M, N, K, BackendType.TRITON)
        ratio = tri_tf / hb_tf
        marker = ">" if ratio > 0.95 else "<"
        print(f"{case['name']:<40} {hb_ms:>7.2f}  {hb_tf:>7.1f}  {tri_ms:>7.2f}  {tri_tf:>7.1f}  {ratio:>6.2f}x {marker}")

    GlobalBackendManager.set_gemm_backend(None, PrecisionType.BF16_FP16_FP32)

if __name__ == "__main__":
    main()
