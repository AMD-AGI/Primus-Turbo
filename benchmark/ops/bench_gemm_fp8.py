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

import primus_turbo.pytorch as turbo
from primus_turbo.pytorch.core.low_precision import (
    Float8QuantConfig,
    Format,
    ScaleDtype,
    ScalingGranularity,
)

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


def compute_snr(ref, actual):
    """Compute Signal-to-Noise Ratio (SNR) in dB."""
    err = ref.to(torch.float64) - actual.to(torch.float64)
    return 20 * torch.log10(ref.to(torch.float64).norm() / err.norm()).item()


def gemm_ref(a, b, trans_b=True):
    """Reference GEMM using PyTorch native matmul."""
    b_mat = b.T if trans_b else b
    return a @ b_mat


def check_gemm_fp8_correctness(a, b, out, grad_out, trans_b, fp8_format):
    """Check correctness of FP8 GEMM forward and backward using SNR."""
    snr_threshold = 25 if fp8_format == Format.E4M3 else 20

    out_ref = gemm_ref(a.detach(), b.detach(), trans_b=trans_b)
    out_snr = compute_snr(out_ref, out.detach())

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
        f"Correctness Check: {status} (out={out_snr:.1f}, da={da_snr:.1f}, db={db_snr:.1f}) threshold={snr_threshold}"
    )

    return correct


def profile_gemm_fp8(M, N, K, ori_dtype, config, trans_b):
    device = "cuda"
    b_shape = (N, K) if trans_b else (K, N)
    a = torch.randn((M, K), dtype=ori_dtype, device=device, requires_grad=True)
    b = torch.randn(b_shape, dtype=ori_dtype, device=device, requires_grad=True)

    out = turbo.ops.gemm_fp8(a, b, trans_b=trans_b, config=config)
    grad_out = torch.randn_like(out)
    correct = check_gemm_fp8_correctness(a, b, out, grad_out, trans_b, config.format)

    fwd_func = lambda: turbo.ops.gemm_fp8(a, b, trans_b=trans_b, config=config)
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


def benchmark_gemm_fp8(granularity_name="tensorwise", output_csv=None):
    full_name = torch.cuda.get_device_name(0)
    match = re.search(r"(MI\d+)", full_name)
    gpu_name = match.group(1) if match else full_name.split()[-1]
    config = GRANULARITY_CONFIG_MAP[granularity_name]

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
                    f"TestID: {test_id}, Case: {model_name}, MBS: {MBS}, "
                    f"M: {M}, N: {N}, K: {K}, granularity: {granularity_name}"
                )
                print(f"{'='*60}")

                fwd_time_ms, fwd_tflops, bwd_time_ms, bwd_tflops, correct = profile_gemm_fp8(
                    M, N, K, ori_dtype, config, trans_b
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
                        "Granularity": granularity_name,
                        "Check": "PASS" if correct else "FAIL",
                        "Forward Time (ms)": f"{fwd_time_ms:.2f}",
                        "Forward TFLOPS": f"{fwd_tflops:.2f}",
                        "Backward Time (ms)": f"{bwd_time_ms:.2f}",
                        "Backward TFLOPS": f"{bwd_tflops:.2f}",
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
        filename = f"gemm_fp8_{granularity_name}_benchmark_result_{timestamp}_{gpu_name}.csv"

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
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output CSV filename. Default: gemm_fp8_{granularity}_benchmark_result_{date}_{gpu}.csv",
    )
    args = parser.parse_args()
    benchmark_gemm_fp8(granularity_name=args.granularity, output_csv=args.output)
