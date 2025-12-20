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
from config import MBS_LIST, ModelConfigs, gen_gemm_test_cases
from tabulate import tabulate

from primus_turbo.pytorch.core.low_precision import (
    Float8QuantConfig,
    Format,
    ScaleDtype,
    ScalingGranularity,
)
from primus_turbo.pytorch.ops import gemm_fp8

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


def profile_gemm_fp8(M, N, K, ori_dtype, config, trans_b):
    device = "cuda"
    b_shape = (N, K) if trans_b else (K, N)
    a = torch.randn((M, K), dtype=ori_dtype, device=device, requires_grad=True)
    b = torch.randn(b_shape, dtype=ori_dtype, device=device, requires_grad=True)

    out = gemm_fp8(a, b, trans_b=trans_b, config=config)
    grad_out = torch.randn_like(out)

    # Forward and backward functions
    fwd_func = lambda: gemm_fp8(a, b, trans_b=trans_b, config=config)
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
    return fwd_mean_time_ms, fwd_tflops, bwd_mean_time_ms, bwd_tflops


def benchmark_gemm_fp8(granularity_name="tensorwise"):
    # Get GPU name
    full_name = torch.cuda.get_device_name(0)
    match = re.search(r"(MI\d+)", full_name)
    gpu_name = match.group(1) if match else full_name.split()[-1]

    # Get config from granularity name
    config = GRANULARITY_CONFIG_MAP[granularity_name]

    # List to collect results
    rows = []

    test_id = 0
    ori_dtype = torch.bfloat16
    trans_b = True
    for model_name, model_config in ModelConfigs.items():
        test_cases = gen_gemm_test_cases(model_config)
        for MBS in MBS_LIST:
            for shape in test_cases:
                test_id += 1

                M = shape[0] * MBS
                N = shape[1]
                K = shape[2]

                print(f"\n{'='*60}")
                print(
                    f"TestID: {test_id}, Case: {model_name}, MBS: {MBS}, M: {M}, N: {N}, K: {K}, granularity: {granularity_name}"
                )
                print(f"{'='*60}")

                fwd_time_ms, fwd_tflops, bwd_time_ms, bwd_tflops = profile_gemm_fp8(
                    M, N, K, ori_dtype, config, trans_b
                )
                # Add to results list
                rows.append(
                    {
                        "TestID": test_id,
                        "GPU": gpu_name,
                        "Case": model_name,
                        "MBS": MBS,
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
    filename = f"gemm_fp8_{granularity_name}_benchmark_result_{timestamp}_{gpu_name}.csv"

    # Save to CSV
    results.to_csv(filename, index=False)
    print(f"Results saved to {filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark FP8 GEMM operations")
    parser.add_argument(
        "--granularity",
        type=str,
        choices=["tensorwise", "rowwise", "blockwise", "mx"],
        default="tensorwise",
        help="Scaling granularity (default: tensorwise)",
    )
    args = parser.parse_args()
    benchmark_gemm_fp8(args.granularity)
