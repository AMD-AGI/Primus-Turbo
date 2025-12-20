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
from config import generate_grouped_gemm_group_lens
from tabulate import tabulate

from primus_turbo.pytorch.ops import grouped_gemm


def grouped_gemm_ref(a, b, seg_lens, trans_b=True):
    """Reference grouped GEMM using PyTorch native matmul."""
    seg_lens = seg_lens.cpu().numpy()
    out = []
    start = 0
    for i, size in enumerate(seg_lens):
        rhs = b[i, :, :].t() if trans_b else b[i, :, :]
        out.append(a[start : start + size, :] @ rhs)
        start += size
    return torch.cat(out)


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


def check_grouped_gemm_correctness(x, w, group_lens, out, grad_out, dtype):
    """Check correctness of grouped GEMM forward and backward against PyTorch reference."""
    # Forward check
    out_ref = grouped_gemm_ref(x.detach(), w.detach(), group_lens, trans_b=True)
    fwd_correct = check_allclose(out.detach(), out_ref, dtype)

    # Backward check
    x_ref = x.detach().clone().requires_grad_()
    w_ref = w.detach().clone().requires_grad_()
    out_ref = grouped_gemm_ref(x_ref, w_ref, group_lens, trans_b=True)
    out_ref.backward(grad_out)
    out.backward(grad_out, retain_graph=True)
    bwd_x_correct = check_allclose(x.grad, x_ref.grad, dtype)
    bwd_w_correct = check_allclose(w.grad, w_ref.grad, dtype)

    # Reset gradients
    x.grad = None
    w.grad = None

    status = "PASS" if (fwd_correct and bwd_x_correct and bwd_w_correct) else "FAIL"
    print(f"Correctness Check: {status} (fwd={fwd_correct}, bwd_x={bwd_x_correct}, bwd_w={bwd_w_correct})")

    return fwd_correct and bwd_x_correct and bwd_w_correct


M_SIZE_LIST = [512, 1024, 2048, 4096, 8192, 16384]
EP_SIZE_LIST = [32, 16, 8]


def _generate_moe_test_cases(
    name_prefix: str,
    n_routed_experts: int,
    moe_intermediate_size: int,
    hidden_size: int,
):
    test_cases = []
    shapes_dict = {
        f"{name_prefix}-GateUP": (2 * moe_intermediate_size, hidden_size),
        f"{name_prefix}-Down": (hidden_size, moe_intermediate_size),
    }

    for ep in EP_SIZE_LIST:
        if n_routed_experts % ep != 0:
            continue
        B = n_routed_experts // ep
        if B < 1:
            continue
        for M in M_SIZE_LIST:
            for name, (N, K) in shapes_dict.items():
                for dtype in [torch.bfloat16]:
                    test_cases.append(
                        {
                            "Case": name,
                            "B": B,
                            "M": M,
                            "N": N,
                            "K": K,
                            "dtype": dtype,
                        }
                    )
    return test_cases


def generate_deepseekv3_test_cases():
    return _generate_moe_test_cases(
        "DSV3", n_routed_experts=256, moe_intermediate_size=2048, hidden_size=7168
    )


def generate_deepseekv2_test_cases():
    return _generate_moe_test_cases(
        "DSV2", n_routed_experts=160, moe_intermediate_size=1536, hidden_size=5120
    )


def generate_deepseekv2_lite_test_cases():
    return _generate_moe_test_cases(
        "DSV2-Lite", n_routed_experts=64, moe_intermediate_size=1408, hidden_size=2048
    )


def generate_grok_v2_test_cases():
    # https://huggingface.co/xai-org/grok-2/blob/main/config.json
    return _generate_moe_test_cases(
        "Grok-V2", n_routed_experts=8, moe_intermediate_size=16384, hidden_size=8192
    )


def bench_grouped_gemm(B, M, N, K, dtype):
    device = "cuda"
    # Prepare inputs
    x = torch.randn((B * M, K), dtype=dtype, device=device, requires_grad=True)
    w = torch.randn((B, N, K), dtype=dtype, device=device, requires_grad=True)
    group_lens = generate_grouped_gemm_group_lens(B, M, balance=True).to(device)

    # Prepare gradient output
    out = grouped_gemm(x, w, group_lens, trans_b=True)
    grad_out = torch.randn_like(out)

    # Correctness check
    correct = check_grouped_gemm_correctness(x, w, group_lens, out, grad_out, dtype)

    # Forward and backward functions
    fwd_func = lambda: grouped_gemm(x, w, group_lens, trans_b=True)
    bwd_func = lambda: out.backward(grad_out, retain_graph=True)
    out = fwd_func()
    bwd_func()

    # Calculate FLOPs
    fwd_total_flops = 2 * B * M * N * K
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


def benchmark_grouped_gemm(output_csv=None):
    # Get GPU name
    full_name = torch.cuda.get_device_name(0)
    match = re.search(r"(MI\d+)", full_name)
    gpu_name = match.group(1) if match else full_name.split()[-1]

    # Generate test cases
    dsv2_lite_test_cases = generate_deepseekv2_lite_test_cases()
    dsv2_test_cases = generate_deepseekv2_test_cases()
    dsv3_test_cases = generate_deepseekv3_test_cases()
    grok_v2_test_cases = generate_grok_v2_test_cases()
    test_cases = dsv2_lite_test_cases + dsv2_test_cases + dsv3_test_cases + grok_v2_test_cases

    # List to collect results
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
        print(f"TestID: {test_id}, Case: {case['Case']}, B: {B}, M: {M}, N: {N}, K: {K}, dtype: {dtype}")
        print(f"{'='*60}")

        try:
            # Run benchmark
            fwd_time_ms, fwd_tflops, bwd_time_ms, bwd_tflops, correct = bench_grouped_gemm(
                B=B,
                M=M,
                N=N,
                K=K,
                dtype=dtype,
            )

            # Add to results list
            rows.append(
                {
                    "TestID": test_id,
                    "GPU": gpu_name,
                    "Case": case["Case"],
                    "B": B,
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
            print(f"Failed to run {case}: {str(e)}")
            rows.append(
                {
                    "TestID": test_id,
                    "GPU": gpu_name,
                    "Case": case["Case"],
                    "B": B,
                    "M": M,
                    "N": N,
                    "K": K,
                    "Check": "ERROR",
                    "Forward Time (ms)": "Failed",
                    "Forward TFLOPS": "N/A",
                    "Backward Time (ms)": "Failed",
                    "Backward TFLOPS": "N/A",
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
        filename = f"grouped_gemm_benchmark_result_{timestamp}_{gpu_name}.csv"

    # Save to CSV
    results.to_csv(filename, index=False)
    print(f"Results saved to {filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark Grouped GEMM operations")
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output CSV filename. If not specified, uses default naming: grouped_gemm_benchmark_result_{date}_{gpu}.csv",
    )
    args = parser.parse_args()
    benchmark_grouped_gemm(output_csv=args.output)
