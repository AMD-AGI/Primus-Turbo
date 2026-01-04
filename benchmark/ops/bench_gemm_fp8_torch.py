###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""PyTorch FP8 GEMM Baseline Benchmark using torchao."""

import argparse
from datetime import datetime

import pandas as pd
import torch
import torch.utils.benchmark as benchmark
from config import (
    BATCH_SIZE_LIST,
    DenseModelConfigs,
    compute_snr,
    gemm_ref,
    gen_gemm_test_cases,
    get_platform_info,
)
from tabulate import tabulate
from torchao.float8 import Float8LinearConfig
from torchao.float8.float8_linear import matmul_with_hp_or_float8_args
from torchao.float8.float8_training_tensor import LinearMMConfig, ScaledMMConfig

# Default config for FP8 GEMM (use_fast_accum=False for fp32 accumulation, matching turbo)
DEFAULT_MM_CONFIG = LinearMMConfig(
    ScaledMMConfig(emulate=False, use_fast_accum=False, fp8_output=False, pad_inner_dim=False),
    ScaledMMConfig(emulate=False, use_fast_accum=False, fp8_output=False, pad_inner_dim=False),
    ScaledMMConfig(emulate=False, use_fast_accum=False, fp8_output=False, pad_inner_dim=False),
)
DEFAULT_CONFIG = Float8LinearConfig()


def _gemm_fp8_impl(a, b):
    """FP8 GEMM implementation using torchao."""
    return matmul_with_hp_or_float8_args.apply(a, b.t(), DEFAULT_MM_CONFIG, DEFAULT_CONFIG)


gemm_fp8_torch = None


def get_compiled_gemm():
    """Get a compiled FP8 GEMM function."""
    global gemm_fp8_torch
    torch._dynamo.reset()
    torch._functorch.config.donated_buffer = False
    gemm_fp8_torch = torch.compile(_gemm_fp8_impl)
    return gemm_fp8_torch


def check_gemm_fp8_correctness(a, b, out, grad_out):
    """Check correctness of FP8 GEMM forward and backward using SNR."""
    snr_threshold = 20  # Match turbo's threshold for E4M3

    out_ref = gemm_ref(a.detach(), b.detach(), trans_b=True)
    out_snr = compute_snr(out_ref, out.detach())

    a_ref = a.detach().clone().requires_grad_()
    b_ref = b.detach().clone().requires_grad_()
    out_ref = gemm_ref(a_ref, b_ref, trans_b=True)
    out_ref.backward(grad_out)
    out.backward(grad_out, retain_graph=True)
    da_snr = compute_snr(a_ref.grad, a.grad)
    db_snr = compute_snr(b_ref.grad, b.grad)

    a.grad = None
    b.grad = None

    correct = all(snr > snr_threshold for snr in [out_snr, da_snr, db_snr])
    status = "PASS" if correct else "FAIL"
    print(
        f"Correctness Check: {status} (out={out_snr:.1f}, da={da_snr:.1f}, db={db_snr:.1f}) threshold={snr_threshold}"
    )

    return correct


def profile_gemm_fp8(M, N, K, dtype):
    device = "cuda"
    a = torch.randn((M, K), dtype=dtype, device=device, requires_grad=True)
    b = torch.randn((N, K), dtype=dtype, device=device, requires_grad=True)

    compiled_fn = get_compiled_gemm()

    out = compiled_fn(a, b)
    grad_out = torch.randn_like(out)
    correct = check_gemm_fp8_correctness(a, b, out, grad_out)

    fwd_func = lambda: compiled_fn(a, b)
    bwd_func = lambda: out.backward(grad_out, retain_graph=True)
    out = fwd_func()
    bwd_func()

    fwd_total_flops = 2 * M * N * K
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


def benchmark_gemm_fp8_torch(granularity_name="tensorwise", output_csv=None):
    platform, gpu_name = get_platform_info()
    rows = []
    test_id = 0
    dtype = torch.bfloat16

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
                    f"TestID: {test_id}, Case: {model_name}, MBS: {MBS}, "
                    f"M: {M}, N: {N}, K: {K}, granularity: {granularity_name}"
                )
                print(f"{'='*60}")

                try:
                    fwd_time_ms, fwd_tflops, bwd_time_ms, bwd_tflops, correct = profile_gemm_fp8(
                        M, N, K, dtype
                    )
                    rows.append(
                        {
                            "TestID": test_id,
                            "Platform": platform,
                            "GPU": gpu_name,
                            "Case": model_name,
                            "MBS": MBS,
                            "M": M,
                            "N": N,
                            "K": K,
                            "Granularity": granularity_name,
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
                            "Platform": platform,
                            "GPU": gpu_name,
                            "Case": model_name,
                            "MBS": MBS,
                            "M": M,
                            "N": N,
                            "K": K,
                            "Granularity": granularity_name,
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
        filename = f"gemm_fp8_torch_{granularity_name}_benchmark_{timestamp}_{gpu_name}.csv"
    results.to_csv(filename, index=False)
    print(f"Results saved to {filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark PyTorch FP8 GEMM (Baseline)")
    parser.add_argument(
        "--granularity",
        type=str,
        choices=["tensorwise"],
        default="tensorwise",
        help="Scaling granularity (default: tensorwise). Currently only tensorwise is supported.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output CSV filename. Default: gemm_fp8_torch_{granularity}_benchmark_{date}_{gpu}.csv",
    )
    args = parser.parse_args()
    benchmark_gemm_fp8_torch(granularity_name=args.granularity, output_csv=args.output)
