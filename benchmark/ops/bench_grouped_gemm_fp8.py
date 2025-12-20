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

from primus_turbo.pytorch.core.low_precision import (
    Float8QuantConfig,
    Format,
    ScaleDtype,
    ScalingGranularity,
)
from primus_turbo.pytorch.ops import grouped_gemm_fp8

# Mapping from CLI argument to Float8QuantConfig
GRANULARITY_CONFIG_MAP = {
    "tensorwise": Float8QuantConfig(format=Format.E4M3, granularity=ScalingGranularity.TENSORWISE),
    "rowwise": Float8QuantConfig(format=Format.E4M3, granularity=ScalingGranularity.ROWWISE),
    "blockwise": Float8QuantConfig(
        format=Format.E4M3,
        granularity=ScalingGranularity.BLOCKWISE,
        block_size=128,
    ),
    "mx": Float8QuantConfig(
        format=Format.E4M3,
        granularity=ScalingGranularity.MX_BLOCKWISE,
        block_size=32,
        scale_dtype=ScaleDtype.E8M0,
    ),
}

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
        B = n_routed_experts // ep
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


def bench_grouped_gemm_fp8(B, M, N, K, ori_dtype, config):
    device = "cuda"
    trans_b = True
    # Prepare inputs
    group_lens = generate_grouped_gemm_group_lens(B, M, balance=True).to(device)
    b_shape = (B, N, K) if trans_b else (B, K, N)
    a = torch.randn((B * M, K), dtype=ori_dtype, device=device, requires_grad=True)
    b = torch.randn(b_shape, dtype=ori_dtype, device=device, requires_grad=True)

    # Prepare gradient output
    out = grouped_gemm_fp8(a, b, group_lens, trans_b=trans_b, config=config)
    grad_out = torch.randn_like(out)

    # Forward and backward functions
    fwd_func = lambda: grouped_gemm_fp8(a, b, group_lens, trans_b=trans_b, config=config)
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
    return fwd_mean_time_ms, fwd_tflops, bwd_mean_time_ms, bwd_tflops


def benchmark_grouped_gemm_fp8(granularity_name="tensorwise"):
    # Get GPU name
    full_name = torch.cuda.get_device_name(0)
    match = re.search(r"(MI\d+)", full_name)
    gpu_name = match.group(1) if match else full_name.split()[-1]

    # Get config from granularity name
    config = GRANULARITY_CONFIG_MAP[granularity_name]

    # Generate test cases
    dsv2_lite_test_cases = generate_deepseekv2_lite_test_cases()
    dsv2_test_cases = generate_deepseekv2_test_cases()
    dsv3_test_cases = generate_deepseekv3_test_cases()
    test_cases = dsv2_lite_test_cases + dsv2_test_cases + dsv3_test_cases

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
        print(
            f"TestID: {test_id}, Case: {case['Case']}, B: {B}, M: {M}, N: {N}, K: {K}, granularity: {granularity_name}"
        )
        print(f"{'='*60}")

        try:
            # Run benchmark
            fwd_time_ms, fwd_tflops, bwd_time_ms, bwd_tflops = bench_grouped_gemm_fp8(
                B=B,
                M=M,
                N=N,
                K=K,
                ori_dtype=dtype,
                config=config,
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
                    "granularity": granularity_name,
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
                    "granularity": granularity_name,
                    "Forward Time (ms)": "N/A",
                    "Forward TFLOPS": "N/A",
                    "Backward Time (ms)": "N/A",
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

    # Generate filename with timestamp, hardware and granularity info
    timestamp = datetime.now().strftime("%Y%m%d")
    filename = f"grouped_gemm_fp8_{granularity_name}_benchmark_result_{timestamp}_{gpu_name}.csv"

    # Save to CSV
    results.to_csv(filename, index=False)
    print(f"Results saved to {filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark FP8 Grouped GEMM operations")
    parser.add_argument(
        "--granularity",
        type=str,
        choices=["tensorwise", "rowwise", "blockwise", "mx"],
        default="tensorwise",
        help="Scaling granularity (default: tensorwise)",
    )
    args = parser.parse_args()
    benchmark_grouped_gemm_fp8(args.granularity)
