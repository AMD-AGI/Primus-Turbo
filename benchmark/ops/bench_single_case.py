#!/usr/bin/env python3
###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Primus-Turbo Single Case Grouped GEMM Benchmark."""

import argparse
import torch
import torch.utils.benchmark as benchmark
from config import (
    check_allclose,
    gen_grouped_gemm_group_lens,
    get_platform_info,
    grouped_gemm_ref,
)
from tabulate import tabulate

import primus_turbo.pytorch as turbo


def check_grouped_gemm_correctness(x, w, group_lens, out, grad_out, dtype):
    """Check correctness of BF16 grouped GEMM forward and backward."""
    out_ref = grouped_gemm_ref(x.detach(), w.detach(), group_lens, trans_b=True)
    fwd_correct = check_allclose(out.detach(), out_ref, dtype)

    x_ref = x.detach().clone().requires_grad_()
    w_ref = w.detach().clone().requires_grad_()
    out_ref = grouped_gemm_ref(x_ref, w_ref, group_lens, trans_b=True)
    out_ref.backward(grad_out)
    out.backward(grad_out, retain_graph=True)
    bwd_x_correct = check_allclose(x.grad, x_ref.grad, dtype)
    bwd_w_correct = check_allclose(w.grad, w_ref.grad, dtype)

    x.grad = None
    w.grad = None

    correct = fwd_correct and bwd_x_correct and bwd_w_correct
    status = "PASS" if correct else "FAIL"
    print(f"Correctness Check: {status} (fwd={fwd_correct}, bwd_x={bwd_x_correct}, bwd_w={bwd_w_correct})")

    return correct


def profile_grouped_gemm(B, M, N, K, dtype, warmup=20, repeat=100):
    """Profile BF16 Grouped GEMM."""
    device = "cuda"
    x = torch.randn((B * M, K), dtype=dtype, device=device, requires_grad=True)
    w = torch.randn((B, N, K), dtype=dtype, device=device, requires_grad=True)
    group_lens = gen_grouped_gemm_group_lens(B, M, balance=True).to(device)

    out = turbo.ops.grouped_gemm(x, w, group_lens, trans_b=True)
    grad_out = torch.randn_like(out)
    correct = check_grouped_gemm_correctness(x, w, group_lens, out, grad_out, dtype)

    fwd_func = lambda: turbo.ops.grouped_gemm(x, w, group_lens, trans_b=True)
    bwd_func = lambda: out.backward(grad_out, retain_graph=True)
    out = fwd_func()
    bwd_func()

    fwd_total_flops = 2 * B * M * N * K
    bwd_total_flops = 2 * fwd_total_flops

    # Warmup
    for _ in range(warmup):
        fwd_func()
        bwd_func()
    torch.cuda.synchronize()

    # Benchmark
    fwd_timer = benchmark.Timer(stmt="fn()", globals={"fn": fwd_func})
    bwd_timer = benchmark.Timer(stmt="fn()", globals={"fn": bwd_func})
    fwd_measurement = fwd_timer.timeit(repeat)
    bwd_measurement = bwd_timer.timeit(repeat)

    fwd_mean_time_ms = fwd_measurement.mean * 1e3
    bwd_mean_time_ms = bwd_measurement.mean * 1e3
    fwd_tflops = fwd_total_flops / (fwd_mean_time_ms * 1e-3) / 1e12
    bwd_tflops = bwd_total_flops / (bwd_mean_time_ms * 1e-3) / 1e12

    return fwd_mean_time_ms, fwd_tflops, bwd_mean_time_ms, bwd_tflops, correct


def profile_backward_kernels_separately(B, M, N, K, dtype, warmup=20, repeat=100):
    """分别测量 backward 的两个 kernel: grad_x 和 grad_w"""
    from primus_turbo.pytorch.kernels.grouped_gemm.grouped_gemm_csrc_impl import (
        grouped_gemm_impl,
        grouped_gemm_variable_k_impl,
    )
    from primus_turbo.pytorch.core.backend import BackendType
    
    device = "cuda"
    
    # 模拟 forward 的输入
    x = torch.randn((B * M, K), dtype=dtype, device=device)
    w = torch.randn((B, N, K), dtype=dtype, device=device)  # trans_b=True
    group_lens = gen_grouped_gemm_group_lens(B, M, balance=True).to(device)
    group_offs = torch.zeros(B + 1, dtype=torch.int64, device=device)
    group_offs[1:] = torch.cumsum(group_lens, dim=0)
    
    # grad_out shape = (B * M, N)
    grad_out = torch.randn((B * M, N), dtype=dtype, device=device)
    
    # ============ grad_x kernel ============
    # grad_x = grad_out @ w  (使用 grouped_gemm_impl, trans_b=False)
    # Shape: (B*M, N) @ (B, N, K) -> (B*M, K)
    def grad_x_func():
        return grouped_gemm_impl(
            grad_out, w, group_lens, group_offs,
            trans_a=False, trans_b=False,  # w was trans_b=True, so now trans_b=False
            num_cu=None,
            default_backend=BackendType.CK.value,
        )
    
    # ============ grad_w kernel ============
    # grad_w = x^T @ grad_out  (使用 grouped_gemm_variable_k_impl)
    # Shape: (K, B*M) @ (B*M, N) -> (B, K, N) then transpose to (B, N, K)
    def grad_w_func():
        return grouped_gemm_variable_k_impl(
            x, grad_out, group_lens, group_offs,
            trans_a=True, trans_b=False, trans_c=True,
            num_cu=None,
            default_backend=BackendType.CK.value,
        )
    
    # Warmup
    for _ in range(warmup):
        grad_x_func()
        grad_w_func()
    torch.cuda.synchronize()
    
    # Benchmark grad_x
    grad_x_timer = benchmark.Timer(stmt="fn()", globals={"fn": grad_x_func})
    grad_x_measurement = grad_x_timer.timeit(repeat)
    grad_x_time_ms = grad_x_measurement.mean * 1e3
    
    # Benchmark grad_w
    grad_w_timer = benchmark.Timer(stmt="fn()", globals={"fn": grad_w_func})
    grad_w_measurement = grad_w_timer.timeit(repeat)
    grad_w_time_ms = grad_w_measurement.mean * 1e3
    
    # 计算 TFLOPS
    flops_per_gemm = 2 * B * M * N * K
    grad_x_tflops = flops_per_gemm / (grad_x_time_ms * 1e-3) / 1e12
    grad_w_tflops = flops_per_gemm / (grad_w_time_ms * 1e-3) / 1e12
    
    return grad_x_time_ms, grad_x_tflops, grad_w_time_ms, grad_w_tflops


def main():
    parser = argparse.ArgumentParser(description="Benchmark single Grouped GEMM case")
    parser.add_argument("--B", type=int, default=10, help="Batch/Group count")
    parser.add_argument("--M", type=int, default=1024, help="M dimension (per group)")
    parser.add_argument("--N", type=int, default=5120, help="N dimension")
    parser.add_argument("--K", type=int, default=1536, help="K dimension")
    parser.add_argument("--warmup", type=int, default=20, help="Warmup iterations")
    parser.add_argument("--repeat", type=int, default=100, help="Benchmark iterations")
    parser.add_argument("--detailed", action="store_true", help="Show detailed backward breakdown")
    args = parser.parse_args()

    B, M, N, K = args.B, args.M, args.N, args.K
    dtype = torch.bfloat16

    platform, gpu_name = get_platform_info()

    print("=" * 80)
    print("Primus-Turbo Single Case Grouped GEMM Benchmark")
    print("=" * 80)
    print(f"Platform: {platform}")
    print(f"GPU: {gpu_name}")
    print(f"B={B}, M={M}, N={N}, K={K}, dtype=bf16")
    print(f"Warmup: {args.warmup}, Repeat: {args.repeat}")
    print("=" * 80)

    fwd_time_ms, fwd_tflops, bwd_time_ms, bwd_tflops, correct = profile_grouped_gemm(
        B, M, N, K, dtype, warmup=args.warmup, repeat=args.repeat
    )

    print("\n" + "=" * 80)
    print("Results")
    print("=" * 80)

    results = [
        ["Forward", f"{fwd_time_ms:.3f} ms", f"{fwd_tflops:.2f} TFLOPS"],
        ["Backward (total)", f"{bwd_time_ms:.3f} ms", f"{bwd_tflops:.2f} TFLOPS"],
    ]
    print(tabulate(results, headers=["Pass", "Time", "Performance"], tablefmt="grid"))
    print(f"\nCorrectness: {'PASS' if correct else 'FAIL'}")

    # Detailed backward breakdown
    if args.detailed:
        print("\n" + "=" * 80)
        print("Backward Kernel Breakdown (分开测量)")
        print("=" * 80)
        
        grad_x_time, grad_x_tflops, grad_w_time, grad_w_tflops = profile_backward_kernels_separately(
            B, M, N, K, dtype, warmup=args.warmup, repeat=args.repeat
        )
        
        results = [
            ["grad_x (grouped_gemm)", f"{grad_x_time:.3f} ms", f"{grad_x_tflops:.2f} TFLOPS"],
            ["grad_w (variable_k)", f"{grad_w_time:.3f} ms", f"{grad_w_tflops:.2f} TFLOPS"],
            ["Total (grad_x + grad_w)", f"{grad_x_time + grad_w_time:.3f} ms", "-"],
        ]
        print(tabulate(results, headers=["Kernel", "Time", "Performance"], tablefmt="grid"))
        
        # 调和平均
        harmonic_mean = 2 / (1/grad_x_tflops + 1/grad_w_tflops)
        print(f"\nBackward 调和平均 TFLOPS: {harmonic_mean:.2f}")
        print(f"理论 Backward 时间: {grad_x_time + grad_w_time:.3f} ms")
        print(f"实际 Backward 时间: {bwd_time_ms:.3f} ms")
        print(f"差异 (overhead): {bwd_time_ms - (grad_x_time + grad_w_time):.3f} ms")

    # FLOPs breakdown
    fwd_total_flops = 2 * B * M * N * K
    bwd_total_flops = 4 * B * M * N * K  # grad_x + grad_w
    print(f"\nForward FLOPs: {fwd_total_flops / 1e12:.4f} TFLOPs")
    print(f"Backward FLOPs: {bwd_total_flops / 1e12:.4f} TFLOPs")


if __name__ == "__main__":
    main()

