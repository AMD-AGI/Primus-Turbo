###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Primus-Turbo Grouped GEMM Benchmark (BF16 and FP8)."""

import argparse
from datetime import datetime

import pandas as pd
import torch
import torch.utils.benchmark as benchmark
from config import (
    check_allclose,
    compute_snr,
    gen_grouped_gemm_group_lens,
    gen_grouped_gemm_test_cases,
    get_platform_info,
    grouped_gemm_ref,
)
from tabulate import tabulate

import primus_turbo.pytorch as turbo
from primus_turbo.pytorch.core.low_precision import (
    Float8QuantConfig,
    Format,
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
}


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


def check_grouped_gemm_fp8_correctness(a, b, out, grad_out, group_lens, fp8_format):
    """Check correctness of FP8 grouped GEMM forward and backward using SNR."""
    snr_threshold = 25 if fp8_format == Format.E4M3 else 20

    out_ref = grouped_gemm_ref(a.detach(), b.detach(), group_lens, trans_b=True)
    out_snr = compute_snr(out_ref, out.detach())

    a_ref = a.detach().clone().requires_grad_()
    b_ref = b.detach().clone().requires_grad_()
    out_ref = grouped_gemm_ref(a_ref, b_ref, group_lens, trans_b=True)
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


def profile_grouped_gemm(B, M, N, K, dtype):
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


def profile_grouped_gemm_fp8(B, M, N, K, dtype, config):
    """Profile FP8 Grouped GEMM."""
    device = "cuda"
    group_lens = gen_grouped_gemm_group_lens(B, M, balance=True).to(device)
    a = torch.randn((B * M, K), dtype=dtype, device=device, requires_grad=True)
    b = torch.randn((B, N, K), dtype=dtype, device=device, requires_grad=True)

    out = turbo.ops.grouped_gemm_fp8(a, b, group_lens, trans_b=True, config=config)
    grad_out = torch.randn_like(out)
    correct = check_grouped_gemm_fp8_correctness(a, b, out, grad_out, group_lens, config.format)

    fwd_func = lambda: turbo.ops.grouped_gemm_fp8(a, b, group_lens, trans_b=True, config=config)
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


def benchmark_grouped_gemm_turbo(dtype_name="bf16", granularity_name="tensorwise", output_csv=None):
    platform, gpu_name = get_platform_info()

    is_fp8 = dtype_name == "fp8"
    config = GRANULARITY_CONFIG_MAP[granularity_name] if is_fp8 else None

    test_cases = gen_grouped_gemm_test_cases()

    rows = []
    test_id = 0
    for case in test_cases:
        test_id += 1
        B, M, N, K = case["B"], case["M"], case["N"], case["K"]
        dtype = case["dtype"]

        print(f"\n{'='*60}")
        if is_fp8:
            print(
                f"TestID: {test_id}, Case: {case['Case']}, B: {B}, M: {M}, N: {N}, K: {K}, "
                f"dtype: fp8, granularity: {granularity_name}"
            )
        else:
            print(f"TestID: {test_id}, Case: {case['Case']}, B: {B}, M: {M}, N: {N}, K: {K}, dtype: bf16")
        print(f"{'='*60}")

        try:
            if is_fp8:
                fwd_time_ms, fwd_tflops, bwd_time_ms, bwd_tflops, correct = profile_grouped_gemm_fp8(
                    B=B, M=M, N=N, K=K, dtype=dtype, config=config
                )
            else:
                fwd_time_ms, fwd_tflops, bwd_time_ms, bwd_tflops, correct = profile_grouped_gemm(
                    B=B, M=M, N=N, K=K, dtype=dtype
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
                "Dtype": dtype_name,
            }
            if is_fp8:
                row["Granularity"] = granularity_name
            row.update(
                {
                    "Check": "PASS" if correct else "FAIL",
                    "Forward Time (ms)": f"{fwd_time_ms:.2f}",
                    "Forward TFLOPS": f"{fwd_tflops:.2f}",
                    "Backward Time (ms)": f"{bwd_time_ms:.2f}",
                    "Backward TFLOPS": f"{bwd_tflops:.2f}",
                }
            )
            rows.append(row)

        except Exception as e:
            print(f"Failed: {str(e)}")
            row = {
                "TestID": test_id,
                "Platform": platform,
                "GPU": gpu_name,
                "Case": case["Case"],
                "B": B,
                "M": M,
                "N": N,
                "K": K,
                "Dtype": dtype_name,
            }
            if is_fp8:
                row["Granularity"] = granularity_name
            row.update(
                {
                    "Check": "ERROR",
                    "Forward Time (ms)": "ERROR",
                    "Forward TFLOPS": "0.00",
                    "Backward Time (ms)": "ERROR",
                    "Backward TFLOPS": "0.00",
                }
            )
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
        if is_fp8:
            filename = f"grouped_gemm_turbo_fp8_{granularity_name}_{timestamp}_{gpu_name}.csv"
        else:
            filename = f"grouped_gemm_turbo_bf16_{timestamp}_{gpu_name}.csv"
    results.to_csv(filename, index=False)
    print(f"Results saved to {filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark Primus-Turbo Grouped GEMM operations")
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["bf16", "fp8"],
        default="bf16",
        help="Data type: bf16 or fp8 (default: bf16)",
    )
    parser.add_argument(
        "--granularity",
        type=str,
        choices=["tensorwise", "rowwise", "blockwise"],
        default="tensorwise",
        help="FP8 scaling granularity (only used when dtype=fp8, default: tensorwise)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output CSV filename",
    )
    args = parser.parse_args()
    benchmark_grouped_gemm_turbo(
        dtype_name=args.dtype, granularity_name=args.granularity, output_csv=args.output
    )
