"""Benchmark CK FP8 GEMM M-aware tile selection.

Compares FP8 GEMM performance for small-M shapes where M-aware
tile selection switches from 256x* to 128x128x128.
"""

import json
import time
import torch
from primus_turbo.pytorch.core.backend import BackendType, GlobalBackendManager
from primus_turbo.pytorch.core.low_precision import (
    Float8QuantConfig,
    Format,
    ScalingGranularity,
)
from primus_turbo.pytorch.ops import gemm_fp8

CASES = [
    # (M, N, K, description) - shapes where M-aware should trigger
    # Small M with N%256==0: 256x256 → few tiles
    (256, 4096, 2048,  "small-M n%256"),
    (512, 4096, 2048,  "small-M n%256"),
    (1024, 4096, 2048, "small-M n%256"),
    (2048, 4096, 2048, "med-M n%256"),
    # Small M with N%128==0 but N%256!=0
    (256, 3968, 2048,  "small-M n%128"),
    (512, 3968, 2048,  "small-M n%128"),
    (1024, 3968, 2048, "small-M n%128"),
    # Larger N to confirm benefit
    (256, 8192, 4096,  "small-M large-N"),
    (512, 8192, 4096,  "small-M large-N"),
    (1024, 8192, 4096, "small-M large-N"),
    # Large M baseline (M-aware should NOT trigger)
    (8192, 4096, 2048, "large-M baseline"),
    (16384, 4096, 2048, "large-M baseline"),
    # LLM inference shapes (prefill)
    (128, 4096, 4096,  "llm prefill bs=1"),
    (256, 14336, 4096, "llm prefill Llama70B gate"),
    (512, 14336, 4096, "llm prefill Llama70B gate"),
]


def bench_one(m, n, k, granularity, warmup=5, iters=20):
    device = "cuda:0"
    a = torch.randn(m, k, dtype=torch.bfloat16, device=device)
    b = torch.randn(k, n, dtype=torch.bfloat16, device=device)

    config = Float8QuantConfig(granularity=granularity, format=Format.E4M3)

    GlobalBackendManager.set_gemm_backend(BackendType.CK)
    GlobalBackendManager.set_auto_tune(False)

    for _ in range(warmup):
        _ = gemm_fp8(a, b, False, False, torch.bfloat16, config)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(iters):
        _ = gemm_fp8(a, b, False, False, torch.bfloat16, config)
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / iters

    flops = 2.0 * m * n * k
    tflops = flops / elapsed / 1e12
    ms = elapsed * 1000

    GlobalBackendManager.reset()
    return ms, tflops


def main():
    results = []
    for m, n, k, desc in CASES:
        m_tiles_256 = (m + 255) // 256
        n_tiles_256 = n // 256 if n % 256 == 0 else n // 128

        for gran_name, gran in [
            ("tensorwise", ScalingGranularity.TENSORWISE),
            ("rowwise", ScalingGranularity.ROWWISE),
        ]:
            ms, tflops = bench_one(m, n, k, gran)
            triggered = "YES" if m_tiles_256 * n_tiles_256 < 304 else "NO"
            entry = {
                "m": m, "n": n, "k": k, "desc": desc,
                "granularity": gran_name,
                "ms": round(ms, 4), "tflops": round(tflops, 2),
                "m_aware_triggered": triggered,
            }
            results.append(entry)
            print(
                f"M={m:>5} N={n:>5} K={k:>5} {gran_name:>12} "
                f"| {ms:7.3f} ms  {tflops:7.2f} TFLOPS | m_aware={triggered}"
            )

    with open("benchmark/baselines/gemm_fp8_m_aware_ck.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {len(results)} results to benchmark/baselines/gemm_fp8_m_aware_ck.json")


if __name__ == "__main__":
    main()
