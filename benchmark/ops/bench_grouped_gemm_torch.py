###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""PyTorch Grouped GEMM Baseline Benchmark using torch._grouped_mm."""

import argparse
from datetime import datetime

import pandas as pd
import torch
import torch.utils.benchmark as benchmark
from config import (
    check_allclose,
    gen_grouped_gemm_group_lens,
    gen_grouped_gemm_test_cases,
    get_platform_info,
    grouped_gemm_ref,
)
from tabulate import tabulate


def check_grouped_gemm_correctness(x, w, group_lens, out, grad_out, dtype):
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

    status = "PASS" if (fwd_correct and bwd_x_correct and bwd_w_correct) else "FAIL"
    print(f"Correctness Check: {status} (fwd={fwd_correct}, bwd_x={bwd_x_correct}, bwd_w={bwd_w_correct})")

    return fwd_correct and bwd_x_correct and bwd_w_correct


def profile_grouped_gemm(B, M, N, K, dtype):
    device = "cuda"
    x = torch.randn((B * M, K), dtype=dtype, device=device, requires_grad=True)
    w = torch.randn((B, N, K), dtype=dtype, device=device, requires_grad=True)
    group_lens = gen_grouped_gemm_group_lens(B, M, balance=True).to(device)
    offsets = torch.cumsum(group_lens, dim=0, dtype=torch.int32)

    out = torch._grouped_mm(x, w.transpose(-2, -1), offs=offsets)
    grad_out = torch.randn_like(out)
    correct = check_grouped_gemm_correctness(x, w, group_lens, out, grad_out, dtype)

    fwd_func = lambda: torch._grouped_mm(x, w.transpose(-2, -1), offs=offsets)
    bwd_func = lambda: out.backward(grad_out, retain_graph=True)
    out = fwd_func()
    bwd_func()

    fwd_total_flops = 2 * B * M * N * K
    bwd_total_flops = 2 * fwd_total_flops

    for _ in range(20):
        fwd_func()
        bwd_func()
    torch.cuda.synchronize()

    fwd_timer = benchmark.Timer(stmt="fn()", globals={"fn": fwd_func})
    bwd_timer = benchmark.Timer(stmt="fn()", globals={"fn": bwd_func})
    fwd_measurement = fwd_timer.timeit(100)
    bwd_measurement = bwd_timer.timeit(100)

    fwd_mean_time_ms = fwd_measurement.mean * 1e3
    bwd_mean_time_ms = bwd_measurement.mean * 1e3
    fwd_tflops = fwd_total_flops / (fwd_mean_time_ms * 1e-3) / 1e12
    bwd_tflops = bwd_total_flops / (bwd_mean_time_ms * 1e-3) / 1e12
    print(f"Forward  Mean time: {fwd_mean_time_ms:.3f} ms | TFLOPS: {fwd_tflops:.2f}")
    print(f"Backward Mean time: {bwd_mean_time_ms:.3f} ms | TFLOPS: {bwd_tflops:.2f}")
    return fwd_mean_time_ms, fwd_tflops, bwd_mean_time_ms, bwd_tflops, correct


def benchmark_grouped_gemm_torch(output_csv=None):
    platform, gpu_name = get_platform_info()
    test_cases = gen_grouped_gemm_test_cases()
    rows = []
    test_id = 0
    for case in test_cases:
        test_id += 1
        B = case["B"]
        M = case["M"]
        N = case["N"]
        K = case["K"]
        dtype = case["dtype"]

        print(f"\n{'='*60}")
        print(f"TestID: {test_id}, Case: {case['Case']}, B: {B}, M: {M}, N: {N}, K: {K}, dtype: bf16")
        print(f"{'='*60}")

        try:
            fwd_time_ms, fwd_tflops, bwd_time_ms, bwd_tflops, correct = profile_grouped_gemm(
                B=B,
                M=M,
                N=N,
                K=K,
                dtype=dtype,
            )
            rows.append(
                {
                    "TestID": test_id,
                    "Platform": platform,
                    "GPU": gpu_name,
                    "Case": case["Case"],
                    "B": B,
                    "M": M,
                    "N": N,
                    "K": K,
                    "Dtype": "bf16",
                    "Check": "PASS" if correct else "FAIL",
                    "Forward Time (ms)": f"{fwd_time_ms:.2f}",
                    "Forward TFLOPS": f"{fwd_tflops:.2f}",
                    "Backward Time (ms)": f"{bwd_time_ms:.2f}",
                    "Backward TFLOPS": f"{bwd_tflops:.2f}",
                }
            )

        except Exception as e:
            print(f"Failed to run {case}: {str(e)}")
            rows.append(
                {
                    "TestID": test_id,
                    "Platform": platform,
                    "GPU": gpu_name,
                    "Case": case["Case"],
                    "B": B,
                    "M": M,
                    "N": N,
                    "K": K,
                    "Dtype": "bf16",
                    "Check": "ERROR",
                    "Forward Time (ms)": "ERROR",
                    "Forward TFLOPS": "0.00",
                    "Backward Time (ms)": "ERROR",
                    "Backward TFLOPS": "0.00",
                }
            )

    results = pd.DataFrame(rows)
    print("\nFinal Results:")
    print(tabulate(results, headers="keys", tablefmt="grid", showindex=False))

    avg_fwd_tflops = results["Forward TFLOPS"].astype(float).mean()
    avg_bwd_tflops = results["Backward TFLOPS"].astype(float).mean()
    print(f"\nAverage Forward TFLOPS: {avg_fwd_tflops:.2f}")
    print(f"Average Backward TFLOPS: {avg_bwd_tflops:.2f}")

    if output_csv:
        filename = output_csv
    else:
        timestamp = datetime.now().strftime("%Y%m%d")
        filename = f"grouped_gemm_torch_benchmark_{timestamp}_{gpu_name}.csv"
    results.to_csv(filename, index=False)
    print(f"Results saved to {filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark PyTorch Grouped GEMM (Baseline)")
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output CSV filename. Default: grouped_gemm_torch_benchmark_{date}_{gpu}.csv",
    )
    args = parser.parse_args()
    benchmark_grouped_gemm_torch(output_csv=args.output)
