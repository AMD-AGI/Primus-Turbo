###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Simple FP8 Tensorwise GEMM Benchmark for M=N=K=8192."""

import torch
import torch.utils.benchmark as benchmark

import primus_turbo.pytorch as turbo
from primus_turbo.pytorch.core.low_precision import (
    Float8QuantConfig,
    Format,
    ScalingGranularity,
)


def compute_snr(ref, test):
    """Compute Signal-to-Noise Ratio in dB."""
    ref = ref.float()
    test = test.float()
    noise = test - ref
    signal_power = (ref**2).mean()
    noise_power = (noise**2).mean()
    if noise_power == 0:
        return float("inf")
    return 10 * torch.log10(signal_power / noise_power).item()


def gemm_ref(a, b, trans_b=False):
    """Reference GEMM using PyTorch."""
    if trans_b:
        return a @ b.T
    return a @ b


def check_correctness(a, b, out, grad_out, trans_b, snr_threshold=25):
    """Check correctness using SNR."""
    # Forward check
    out_ref = gemm_ref(a.detach(), b.detach(), trans_b=trans_b)
    out_snr = compute_snr(out_ref, out.detach())

    # Backward check
    a_ref = a.detach().clone().requires_grad_()
    b_ref = b.detach().clone().requires_grad_()
    out_ref = gemm_ref(a_ref, b_ref, trans_b=trans_b)
    out_ref.backward(grad_out)
    out.backward(grad_out, retain_graph=True)
    da_snr = compute_snr(a_ref.grad, a.grad)
    db_snr = compute_snr(b_ref.grad, b.grad)

    a.grad = None
    b.grad = None

    correct = all(snr > snr_threshold for snr in [out_snr, da_snr, db_snr])
    status = "PASS" if correct else "FAIL"
    print(
        f"Correctness: {status} (out={out_snr:.1f}dB, da={da_snr:.1f}dB, db={db_snr:.1f}dB) threshold={snr_threshold}dB"
    )
    return correct


def main():
    # Configuration
    M, N, K = 8192, 8192, 8192
    dtype = torch.bfloat16
    device = "cuda"
    trans_b = True

    config = Float8QuantConfig(
        format=Format.E4M3,
        granularity=ScalingGranularity.TENSORWISE,
    )

    print(f"{'='*60}")
    print(f"FP8 Tensorwise GEMM Benchmark")
    print(f"M={M}, N={N}, K={K}")
    print(f"{'='*60}")

    # Create tensors
    b_shape = (N, K) if trans_b else (K, N)
    a = torch.randn((M, K), dtype=dtype, device=device, requires_grad=True)
    b = torch.randn(b_shape, dtype=dtype, device=device, requires_grad=True)

    # Forward pass
    out = turbo.ops.gemm_fp8(a, b, trans_b=trans_b, config=config)
    grad_out = torch.randn_like(out)

    # Correctness check
    print("\nChecking correctness...")
    check_correctness(a, b, out, grad_out, trans_b)

    # Benchmark functions
    fwd_func = lambda: turbo.ops.gemm_fp8(a, b, trans_b=trans_b, config=config)
    out = fwd_func()
    bwd_func = lambda: out.backward(grad_out, retain_graph=True)

    # Warmup
    print("\nWarming up...")
    for _ in range(20):
        fwd_func()
        bwd_func()
    torch.cuda.synchronize()

    # Benchmark
    print("Benchmarking...")
    fwd_timer = benchmark.Timer(stmt="fn()", globals={"fn": fwd_func})
    bwd_timer = benchmark.Timer(stmt="fn()", globals={"fn": bwd_func})
    fwd_measurement = fwd_timer.timeit(100)
    bwd_measurement = bwd_timer.timeit(100)

    # Calculate TFLOPS
    fwd_total_flops = 2 * M * N * K
    bwd_total_flops = 2 * fwd_total_flops

    fwd_mean_time_ms = fwd_measurement.mean * 1e3
    bwd_mean_time_ms = bwd_measurement.mean * 1e3
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
