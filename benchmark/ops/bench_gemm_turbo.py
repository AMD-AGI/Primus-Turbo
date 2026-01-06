###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Primus-Turbo GEMM Benchmark (BF16 and FP8)."""

import argparse
from datetime import datetime

import pandas as pd
import torch
import torch.utils.benchmark as benchmark
from config import (
    BATCH_SIZE_LIST,
    DenseModelConfigs,
    check_allclose,
    compute_snr,
    gemm_ref,
    gen_gemm_test_cases,
    get_platform_info,
)
from tabulate import tabulate

import primus_turbo.pytorch as turbo
from primus_turbo.pytorch.core.low_precision import (
    Float4QuantConfig,
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
    "mxfp8": Float8QuantConfig(
        format=Format.E4M3,
        granularity=ScalingGranularity.MX_BLOCKWISE,
        block_size=32,
        scale_dtype=ScaleDtype.E8M0,
    ),
    "mxfp4": Float4QuantConfig(
        format=Format.E2M1_X2,
        granularity=ScalingGranularity.MX_BLOCKWISE,
        block_size=32,
        scale_dtype=ScaleDtype.E8M0,
    ),
}


def check_gemm_correctness(a, b, out, grad_out, trans_b, dtype):
    """Check correctness of BF16 GEMM forward and backward."""
    out_ref = gemm_ref(a.detach(), b.detach(), trans_b=trans_b)
    fwd_correct = check_allclose(out.detach(), out_ref, dtype)

    a_ref = a.detach().clone().requires_grad_()
    b_ref = b.detach().clone().requires_grad_()
    out_ref = gemm_ref(a_ref, b_ref, trans_b=trans_b)
    out_ref.backward(grad_out)
    out.backward(grad_out, retain_graph=True)
    bwd_a_correct = check_allclose(a.grad, a_ref.grad, dtype)
    bwd_b_correct = check_allclose(b.grad, b_ref.grad, dtype)

    a.grad = None
    b.grad = None

    correct = fwd_correct and bwd_a_correct and bwd_b_correct
    status = "PASS" if correct else "FAIL"
    print(f"Correctness Check: {status} (fwd={fwd_correct}, bwd_a={bwd_a_correct}, bwd_b={bwd_b_correct})")

    return correct


def check_gemm_correctness_by_snr(a, b, out, grad_out, trans_b, snr_threshold):
    """Check correctness of GEMM forward and backward using SNR."""
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


def profile_gemm(M, N, K, dtype, trans_b):
    """Profile BF16 GEMM."""
    device = "cuda"
    b_shape = (N, K) if trans_b else (K, N)
    a = torch.randn((M, K), dtype=dtype, device=device, requires_grad=True)
    b = torch.randn(b_shape, dtype=dtype, device=device, requires_grad=True)

    out = turbo.ops.gemm(a, b, trans_b=trans_b)
    grad_out = torch.randn_like(out)
    correct = check_gemm_correctness(a, b, out, grad_out, trans_b, dtype)

    fwd_func = lambda: turbo.ops.gemm(a, b, trans_b=trans_b)
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


def profile_gemm_fp8(M, N, K, dtype, config, trans_b):
    """Profile FP8 GEMM."""
    device = "cuda"
    b_shape = (N, K) if trans_b else (K, N)
    a = torch.randn((M, K), dtype=dtype, device=device, requires_grad=True)
    b = torch.randn(b_shape, dtype=dtype, device=device, requires_grad=True)

    out = turbo.ops.gemm_fp8(a, b, trans_b=trans_b, config=config)
    grad_out = torch.randn_like(out)
    fp8_snr_threshold = 25 if config.format == Format.E4M3 else 20
    correct = check_gemm_correctness_by_snr(a, b, out, grad_out, trans_b, fp8_snr_threshold)

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


def profile_gemm_fp4(M, N, K, dtype, config, trans_b):
    """Profile FP4 GEMM."""
    device = "cuda"
    b_shape = (N, K) if trans_b else (K, N)
    a = torch.randn((M, K), dtype=dtype, device=device, requires_grad=True)
    b = torch.randn(b_shape, dtype=dtype, device=device, requires_grad=True)

    out = turbo.ops.gemm_fp4(a, b, trans_b=trans_b, config=config)
    grad_out = torch.randn_like(out)
    fp4_snr_threshold = 10
    correct = check_gemm_correctness_by_snr(a, b, out, grad_out, trans_b, fp4_snr_threshold)

    fwd_func = lambda: turbo.ops.gemm_fp4(a, b, trans_b=trans_b, config=config)
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


def benchmark_gemm_turbo(dtype_name="bf16", granularity_name="tensorwise", output_csv=None):
    platform, gpu_name = get_platform_info()

    is_fp8 = dtype_name == "fp8"
    is_fp4 = dtype_name == "fp4"

    if is_fp8:
        assert granularity_name in [
            "tensorwise",
            "rowwise",
            "blockwise",
            "mxfp8",
        ], "FP8 only supports granularity: tensorwise, rowwise, blockwise or mxfp8"
    if is_fp4:
        assert granularity_name in ["mxfp4"], "FP4 only supports granularity: mxfp4"

    config = GRANULARITY_CONFIG_MAP[granularity_name] if is_fp8 or is_fp4 else None

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
                if is_fp8:
                    print(
                        f"TestID: {test_id}, Case: {model_name}, MBS: {MBS}, "
                        f"M: {M}, N: {N}, K: {K}, dtype: fp8, granularity: {granularity_name}"
                    )
                elif is_fp4:
                    print(
                        f"TestID: {test_id}, Case: {model_name}, MBS: {MBS}, "
                        f"M: {M}, N: {N}, K: {K}, dtype: fp4, granularity: {granularity_name}"
                    )
                else:
                    print(
                        f"TestID: {test_id}, Case: {model_name}, MBS: {MBS}, "
                        f"M: {M}, N: {N}, K: {K}, dtype: bf16"
                    )
                print(f"{'='*60}")

                try:
                    if is_fp8:
                        fwd_time_ms, fwd_tflops, bwd_time_ms, bwd_tflops, correct = profile_gemm_fp8(
                            M, N, K, dtype, config, trans_b
                        )
                    elif is_fp4:
                        fwd_time_ms, fwd_tflops, bwd_time_ms, bwd_tflops, correct = profile_gemm_fp4(
                            M, N, K, dtype, config, trans_b
                        )
                    else:
                        fwd_time_ms, fwd_tflops, bwd_time_ms, bwd_tflops, correct = profile_gemm(
                            M, N, K, dtype, trans_b
                        )

                    row = {
                        "TestID": test_id,
                        "Platform": platform,
                        "GPU": gpu_name,
                        "Case": model_name,
                        "MBS": MBS,
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
                        "Case": model_name,
                        "MBS": MBS,
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
            filename = f"gemm_turbo_fp8_{granularity_name}_{timestamp}_{gpu_name}.csv"
        else:
            filename = f"gemm_turbo_bf16_{timestamp}_{gpu_name}.csv"
    results.to_csv(filename, index=False)
    print(f"Results saved to {filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark Primus-Turbo GEMM operations")
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["bf16", "fp8", "fp4"],
        default="bf16",
        help="Data type: bf16, fp8 or fp4 (default: bf16)",
    )
    parser.add_argument(
        "--granularity",
        type=str,
        choices=["tensorwise", "rowwise", "blockwise", "mxfp8", "mxfp4"],
        default="tensorwise",
        help="FP8 scaling granularity (only used when dtype=fp8 or fp4, default: tensorwise)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output CSV filename",
    )
    args = parser.parse_args()
    benchmark_gemm_turbo(dtype_name=args.dtype, granularity_name=args.granularity, output_csv=args.output)
