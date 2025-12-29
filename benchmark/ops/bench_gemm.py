###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import argparse
import re
from datetime import datetime

import pandas as pd
import torch
import torch.utils.benchmark as benchmark
from config import BATCH_SIZE_LIST, DenseModelConfigs, gen_gemm_test_cases
from tabulate import tabulate

import primus_turbo.pytorch as turbo


def gemm_ref(a, b, trans_b=True):
    """Reference GEMM using PyTorch native matmul."""
    b_mat = b.T if trans_b else b
    return a @ b_mat


def check_allclose(out, out_ref, dtype, rtol=None, atol=None):
    """Check if two tensors are close within tolerance."""
    if rtol is None or atol is None:
        if dtype == torch.float32:
            rtol, atol = 1e-4, 1e-4
        elif dtype == torch.float16:
            rtol, atol = 1e-2, 1e-2
        else:  # bfloat16
            rtol, atol = 1e-2, 1e-2
    return torch.allclose(out, out_ref, rtol=rtol, atol=atol)


def check_gemm_correctness(a, b, out, grad_out, trans_b, dtype):
    """Check correctness of GEMM forward and backward against PyTorch reference."""
    # Forward check
    out_ref = gemm_ref(a.detach(), b.detach(), trans_b=trans_b)
    fwd_correct = check_allclose(out.detach(), out_ref, dtype)

    # Backward check
    a_ref = a.detach().clone().requires_grad_()
    b_ref = b.detach().clone().requires_grad_()
    out_ref = gemm_ref(a_ref, b_ref, trans_b=trans_b)
    out_ref.backward(grad_out)
    out.backward(grad_out, retain_graph=True)
    bwd_a_correct = check_allclose(a.grad, a_ref.grad, dtype)
    bwd_b_correct = check_allclose(b.grad, b_ref.grad, dtype)

    # Reset gradients
    a.grad = None
    b.grad = None

    status = "PASS" if (fwd_correct and bwd_a_correct and bwd_b_correct) else "FAIL"
    print(f"Correctness Check: {status} (fwd={fwd_correct}, bwd_a={bwd_a_correct}, bwd_b={bwd_b_correct})")

    return fwd_correct and bwd_a_correct and bwd_b_correct


def profile_gemm(M, N, K, dtype, trans_b):
    device = "cuda"
    b_shape = (N, K) if trans_b else (K, N)
    a = torch.randn((M, K), dtype=dtype, device=device, requires_grad=True)
    b = torch.randn(b_shape, dtype=dtype, device=device, requires_grad=True)

    # Prepare gradient output
    out = turbo.ops.gemm(a, b, trans_b=trans_b)
    grad_out = torch.randn_like(out)

    # Correctness check
    correct = check_gemm_correctness(a, b, out, grad_out, trans_b, dtype)

    # Forward and backward functions
    fwd_func = lambda: turbo.ops.gemm(a, b, trans_b=trans_b)
    bwd_func = lambda: out.backward(grad_out, retain_graph=True)
    out = fwd_func()
    bwd_func()

    # Calculate FLOPs
    fwd_total_flops = 2 * M * N * K
    bwd_total_flops = 2 * fwd_total_flops

    # Warmup
    warmup = 20
    for _ in range(warmup):
        fwd_func()
        bwd_func()
    torch.cuda.synchronize()

    # Benchmark
    fwd_timer = benchmark.Timer(
        stmt="fn()",
        globals={"fn": fwd_func},
    )
    bwd_timer = benchmark.Timer(
        stmt="fn()",
        globals={"fn": bwd_func},
    )
    fwd_measurement = fwd_timer.timeit(100)
    bwd_measurement = bwd_timer.timeit(100)

    fwd_mean_time_ms = fwd_measurement.mean * 1e3
    bwd_mean_time_ms = bwd_measurement.mean * 1e3
    fwd_tflops = fwd_total_flops / (fwd_mean_time_ms * 1e-3) / 1e12
    bwd_tflops = bwd_total_flops / (bwd_mean_time_ms * 1e-3) / 1e12
    print(f"Forward  Mean time: {fwd_mean_time_ms:.3f} ms | TFLOPS: {fwd_tflops:.2f}")
    print(f"Backward Mean time: {bwd_mean_time_ms:.3f} ms | TFLOPS: {bwd_tflops:.2f}")

    return fwd_mean_time_ms, fwd_tflops, bwd_mean_time_ms, bwd_tflops, correct


def benchmark_gemm(output_csv=None):
    # Get GPU name
    full_name = torch.cuda.get_device_name(0)
    match = re.search(r"(MI\d+)", full_name)
    gpu_name = match.group(1) if match else full_name.split()[-1]

    # List to collect results
    rows = []

    test_id = 0
    dtype = torch.bfloat16
    trans_b = True
    for model_name, model_config in DenseModelConfigs.items():
        test_cases = gen_gemm_test_cases(model_config)
        for MBS in BATCH_SIZE_LIST:
            for shape in test_cases:
                test_id += 1

                M = shape[0] * MBS
                N = shape[1]
                K = shape[2]

                print(f"\n{'='*60}")
                print(
                    f"TestID: {test_id}, Case: {model_name}, MBS: {MBS}, M: {M}, N: {N}, K: {K}, dtype: {dtype}"
                )
                print(f"{'='*60}")

                try:
                    fwd_time_ms, fwd_tflops, bwd_time_ms, bwd_tflops, correct = profile_gemm(
                        M, N, K, dtype, trans_b
                    )
                    rows.append(
                        {
                            "TestID": test_id,
                            "GPU": gpu_name,
                            "Case": model_name,
                            "MBS": MBS,
                            "M": M,
                            "N": N,
                            "K": K,
                            "Check": "PASS" if correct else "FAIL",
                            "Forward Time (ms)": f"{fwd_time_ms:.2f}",
                            "Forward TFLOPS": f"{fwd_tflops:.2f}",
                            "Backward Time (ms)": f"{bwd_time_ms:.2f}",
                            "Backward TFLOPS": f"{bwd_tflops:.2f}",
                        }
                    )
                except Exception as e:
                    print(f"Failed: {str(e)}")
                    rows.append(
                        {
                            "TestID": test_id,
                            "GPU": gpu_name,
                            "Case": model_name,
                            "MBS": MBS,
                            "M": M,
                            "N": N,
                            "K": K,
                            "Check": "ERROR",
                            "Forward Time (ms)": "ERROR",
                            "Forward TFLOPS": "0.00",
                            "Backward Time (ms)": "ERROR",
                            "Backward TFLOPS": "0.00",
                        }
                    )

    # Create DataFrame from collected results
    results = pd.DataFrame(rows)

    # Print results
    print("\nFinal Results:")
    print(tabulate(results, headers="keys", tablefmt="grid", showindex=False))

    # Calculate and print average TFLOPS
    avg_fwd_tflops = results["Forward TFLOPS"].astype(float).mean()
    avg_bwd_tflops = results["Backward TFLOPS"].astype(float).mean()
    print(f"\nAverage Forward TFLOPS: {avg_fwd_tflops:.2f}")
    print(f"Average Backward TFLOPS: {avg_bwd_tflops:.2f}")

    # Generate filename with timestamp and hardware info
    if output_csv:
        filename = output_csv
    else:
        timestamp = datetime.now().strftime("%Y%m%d")
        filename = f"gemm_benchmark_result_{timestamp}_{gpu_name}.csv"

    # Save to CSV
    results.to_csv(filename, index=False)
    print(f"Results saved to {filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark GEMM operations")
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output CSV filename. If not specified, uses default naming: gemm_benchmark_result_{date}_{gpu}.csv",
    )
    args = parser.parse_args()
    benchmark_gemm(output_csv=args.output)
