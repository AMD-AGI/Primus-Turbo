###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""GMM (grouped_gemm) BF16 Grouped GEMM Baseline Benchmark.

Reference: https://github.com/tgale96/grouped_gemm
"""

import argparse
from datetime import datetime

import grouped_gemm.ops as gmm_ops
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


def check_grouped_gemm_gmm_correctness(a, b, batch_sizes, grad_out, dtype):
    """Check correctness of gmm forward and backward against reference."""
    out = gmm_ops.gmm(a, b, batch_sizes, trans_b=True)

    B = b.shape[0]
    group_lens = batch_sizes.to(device=a.device, dtype=torch.int64)
    out_ref = grouped_gemm_ref(a.detach(), b.detach(), group_lens, trans_b=True)
    fwd_correct = check_allclose(out.detach(), out_ref, dtype)

    a_ref = a.detach().clone().requires_grad_()
    b_ref = b.detach().clone().requires_grad_()
    out_ref = grouped_gemm_ref(a_ref, b_ref, group_lens, trans_b=True)
    out_ref.backward(grad_out)
    out.backward(grad_out, retain_graph=True)
    bwd_correct = check_allclose(a.grad, a_ref.grad, dtype)

    a.grad = None

    correct = fwd_correct and bwd_correct
    status = "PASS" if correct else "FAIL"
    print(f"Correctness Check: {status} (fwd={fwd_correct}, bwd={bwd_correct})")

    return correct


def profile_grouped_gemm_gmm(B, M, N, K, dtype, balance=True, num_topk=None):
    """Profile BF16 Grouped GEMM using grouped_gemm (gmm) library."""
    device = "cuda"
    # batch_sizes must be on CPU for gmm
    batch_sizes = gen_grouped_gemm_group_lens(B, M, balance=balance, num_topk=num_topk)
    total_m = batch_sizes.sum().item()

    a = torch.randn((total_m, K), dtype=dtype, device=device, requires_grad=True)
    b = torch.randn((B, N, K), dtype=dtype, device=device, requires_grad=True)

    out = gmm_ops.gmm(a, b, batch_sizes, trans_b=True)
    grad_out = torch.randn_like(out)

    correct = check_grouped_gemm_gmm_correctness(a, b, batch_sizes, grad_out, dtype)

    def fwd_func():
        return gmm_ops.gmm(a, b, batch_sizes, trans_b=True)

    def fwd_bwd_func():
        out = gmm_ops.gmm(a, b, batch_sizes, trans_b=True)
        out.backward(grad_out)
        a.grad = None
        b.grad = None

    fwd_total_flops = 2 * B * M * N * K
    bwd_total_flops = 2 * fwd_total_flops

    for _ in range(100):
        fwd_bwd_func()
    torch.cuda.synchronize()

    fwd_timer = benchmark.Timer(stmt="fn()", globals={"fn": fwd_func})
    fwd_measurement = fwd_timer.timeit(200)

    fwd_bwd_timer = benchmark.Timer(stmt="fn()", globals={"fn": fwd_bwd_func})
    fwd_bwd_measurement = fwd_bwd_timer.timeit(200)

    fwd_mean_time_ms = fwd_measurement.mean * 1e3
    fwd_bwd_mean_time_ms = fwd_bwd_measurement.mean * 1e3
    bwd_mean_time_ms = fwd_bwd_mean_time_ms - fwd_mean_time_ms

    fwd_tflops = fwd_total_flops / (fwd_mean_time_ms * 1e-3) / 1e12
    bwd_tflops = bwd_total_flops / (bwd_mean_time_ms * 1e-3) / 1e12
    print(f"Forward  Mean time: {fwd_mean_time_ms:.3f} ms | TFLOPS: {fwd_tflops:.2f}")
    print(f"Backward Mean time: {bwd_mean_time_ms:.3f} ms | TFLOPS: {bwd_tflops:.2f}")

    return fwd_mean_time_ms, fwd_tflops, bwd_mean_time_ms, bwd_tflops, correct


def benchmark_grouped_gemm_gmm(output_csv=None):
    platform, gpu_name = get_platform_info()

    test_cases = gen_grouped_gemm_test_cases()

    rows = []
    test_id = 0
    dtype = torch.bfloat16

    for case in test_cases:
        test_id += 1
        B, M, N, K = case["B"], case["M"], case["N"], case["K"]

        balance = case.get("balance", True)
        balance_str = "balanced" if balance else "unbalanced"
        print(f"\n{'='*60}")
        print(f"TestID: {test_id}, Case: {case['Case']}, B: {B}, M: {M}, N: {N}, K: {K}, dtype: bf16, {balance_str}")
        print(f"{'='*60}")

        try:
            fwd_time_ms, fwd_tflops, bwd_time_ms, bwd_tflops, correct = profile_grouped_gemm_gmm(
                B=B, M=M, N=N, K=K, dtype=dtype,
                balance=balance,
                num_topk=case.get("num_topk"),
            )

            row = {
                "TestID": test_id,
                "Platform": platform,
                "GPU": gpu_name,
                "Case": case["Case"],
                "B": B,
                "M": M,
                "N": N,
                "K": K,
                "Dtype": "bf16",
                "Balance": "Y" if balance else "N",
                "Check": "PASS" if correct else "FAIL",
                "Forward Time (ms)": f"{fwd_time_ms:.2f}",
                "Forward TFLOPS": f"{fwd_tflops:.2f}",
                "Backward Time (ms)": f"{bwd_time_ms:.2f}",
                "Backward TFLOPS": f"{bwd_tflops:.2f}",
            }
            rows.append(row)

        except Exception as e:
            import traceback

            print(f"Failed: {str(e)}")
            traceback.print_exc()
            row = {
                "TestID": test_id,
                "Platform": platform,
                "GPU": gpu_name,
                "Case": case["Case"],
                "B": B,
                "M": M,
                "N": N,
                "K": K,
                "Dtype": "bf16",
                "Balance": "Y" if balance else "N",
                "Check": "ERROR",
                "Forward Time (ms)": "ERROR",
                "Forward TFLOPS": "0.00",
                "Backward Time (ms)": "ERROR",
                "Backward TFLOPS": "0.00",
            }
            rows.append(row)

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
        filename = f"grouped_gemm_gmm_bf16_{timestamp}_{gpu_name}.csv"
    results.to_csv(filename, index=False)
    print(f"Results saved to {filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark grouped_gemm (gmm) BF16 Grouped GEMM")
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output CSV filename",
    )
    args = parser.parse_args()
    benchmark_grouped_gemm_gmm(output_csv=args.output)
