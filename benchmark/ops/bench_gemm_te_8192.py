###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Simple TE FP8 Tensorwise GEMM Benchmark for M=N=K=8192."""

import torch
import torch.utils.benchmark as benchmark
import transformer_engine as te
from transformer_engine.common.recipe import Float8CurrentScaling, Format


def main():
    # Configuration
    M, N, K = 8192, 8192, 8192
    dtype = torch.bfloat16
    device = "cuda"

    fp8_recipe = Float8CurrentScaling(fp8_format=Format.E4M3)

    print(f"{'='*60}")
    print(f"TE FP8 Tensorwise GEMM Benchmark")
    print(f"M={M}, N={N}, K={K}")
    print(f"{'='*60}")

    # Create tensors and layer
    x = torch.randn((M, K), dtype=dtype, device=device, requires_grad=True)
    layer = te.pytorch.Linear(K, N, bias=False, params_dtype=dtype).to(device)

    # Forward pass
    with te.pytorch.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
        out = layer(x)
    grad_out = torch.randn_like(out)

    # Benchmark functions
    def fwd_func():
        with te.pytorch.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
            return layer(x)

    def fwd_bwd_func():
        with te.pytorch.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
            out = layer(x)
        out.backward(grad_out)

    # Warmup
    print("\nWarming up...")
    for _ in range(20):
        fwd_bwd_func()
    torch.cuda.synchronize()

    # Benchmark
    print("Benchmarking...")
    fwd_timer = benchmark.Timer(stmt="fn()", globals={"fn": fwd_func})
    fwd_bwd_timer = benchmark.Timer(stmt="fn()", globals={"fn": fwd_bwd_func})
    fwd_measurement = fwd_timer.timeit(100)
    fwd_bwd_measurement = fwd_bwd_timer.timeit(100)

    # Calculate TFLOPS
    fwd_total_flops = 2 * M * N * K
    bwd_total_flops = 2 * fwd_total_flops

    fwd_mean_time_ms = fwd_measurement.mean * 1e3
    fwd_bwd_mean_time_ms = fwd_bwd_measurement.mean * 1e3
    bwd_mean_time_ms = fwd_bwd_mean_time_ms - fwd_mean_time_ms
    fwd_tflops = fwd_total_flops / (fwd_mean_time_ms * 1e-3) / 1e12
    bwd_tflops = bwd_total_flops / (bwd_mean_time_ms * 1e-3) / 1e12

    print(f"\n{'='*60}")
    print(f"Results:")
    print(f"{'='*60}")
    print(f"Forward:  {fwd_mean_time_ms:.3f} ms | {fwd_tflops:.2f} TFLOPS")
    print(f"Backward: {bwd_mean_time_ms:.3f} ms | {bwd_tflops:.2f} TFLOPS")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
