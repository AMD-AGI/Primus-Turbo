"""Tight perf measurement; one warmup, multiple runs for stability."""

import time

import torch

import primus_turbo  # noqa: F401
import primus_turbo.pytorch  # noqa: F401

DEVICE = torch.device("cuda")
DTYPE_FP8 = torch.float8_e4m3fn
DTYPE_OUT = torch.bfloat16


def quantize_mx(x, axis):
    return torch.ops.primus_turbo_cpp_extension.quantize_mxfp8(
        x, DTYPE_FP8, axis, False, False, False
    )


def bench_grouped(G, lens_list, n=8192, k=2048, iters=300, repeats=3):
    torch.manual_seed(2)
    total_m = sum(lens_list)
    a_hp = torch.randn(total_m, k, device=DEVICE, dtype=torch.bfloat16) * 0.5
    b_hp = torch.randn(G, n, k, device=DEVICE, dtype=torch.bfloat16) * 0.5
    oa = quantize_mx(a_hp, axis=1)
    a_fp8, a_s = oa[0], oa[1]
    ob = quantize_mx(b_hp.reshape(G * n, k), axis=1)
    b_fp8 = ob[0].reshape(G, n, k)
    b_s = ob[1].reshape(G, n, -1)
    lens_t = torch.tensor(lens_list, dtype=torch.int64, device=DEVICE)
    offs_t = torch.ops.primus_turbo_cpp_extension.grouped_gemm_compute_offs(lens_t)
    for _ in range(40):
        torch.ops.primus_turbo_cpp_extension.turbo_grouped_gemm_fp8(
            a_fp8, b_fp8, a_s, b_s, lens_t, offs_t, False, True, DTYPE_OUT, "MX_BLOCKWISE"
        )
    torch.cuda.synchronize()
    best = float("inf")
    for r in range(repeats):
        s = time.perf_counter()
        for _ in range(iters):
            torch.ops.primus_turbo_cpp_extension.turbo_grouped_gemm_fp8(
                a_fp8, b_fp8, a_s, b_s, lens_t, offs_t, False, True, DTYPE_OUT, "MX_BLOCKWISE"
            )
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - s) / iters * 1e6
        best = min(best, elapsed)
    flops = 2 * total_m * n * k
    tflops = flops / best / 1e6
    print(f"GG  G={G:1d} {str(lens_list[:3])[:25]:25s} k={k:5d}: {best:7.2f} us  {tflops:6.1f} TFLOPS")
    return tflops


def bench_single(M, N, K, iters=300, repeats=3):
    torch.manual_seed(2)
    a_hp = torch.randn(M, K, device=DEVICE, dtype=torch.bfloat16) * 0.5
    b_hp = torch.randn(N, K, device=DEVICE, dtype=torch.bfloat16) * 0.5
    oa = quantize_mx(a_hp, axis=1)
    a_fp8, a_s = oa[0], oa[1]
    ob = quantize_mx(b_hp, axis=1)
    b_fp8, b_s = ob[0], ob[1]
    for _ in range(40):
        torch.ops.primus_turbo_cpp_extension.turbo_gemm_fp8(
            a_fp8, a_s, b_fp8, b_s, DTYPE_OUT, False, True, False, "MX_BLOCKWISE"
        )
    torch.cuda.synchronize()
    best = float("inf")
    for r in range(repeats):
        s = time.perf_counter()
        for _ in range(iters):
            torch.ops.primus_turbo_cpp_extension.turbo_gemm_fp8(
                a_fp8, a_s, b_fp8, b_s, DTYPE_OUT, False, True, False, "MX_BLOCKWISE"
            )
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - s) / iters * 1e6
        best = min(best, elapsed)
    flops = 2 * M * N * K
    tflops = flops / best / 1e6
    print(f"S   M={M:5d} N={N:5d} K={K:5d}    : {best:7.2f} us  {tflops:6.1f} TFLOPS")
    return tflops


print("=== Single GEMM ===")
for M, N, K in [
    (8192, 8192, 2048),
    (8192, 8192, 4096),
    (8192, 8192, 8192),
    (4096, 8192, 8192),
    (2048, 8192, 8192),
]:
    bench_single(M, N, K)

print("\n=== Grouped GEMM ===")
for cfg in [
    (1, [8192]),
    (2, [8192] * 2),
    (4, [8192] * 4),
    (4, [4096] * 4),
    (4, [2048] * 4),
    (4, [1024] * 4),
    (4, [512] * 4),
    (8, [2048] * 8),
]:
    bench_grouped(*cfg)
