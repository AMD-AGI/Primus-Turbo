###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Primus-Turbo Grouped GEMM Benchmark (BF16 and FP8)."""

import argparse
import os
from datetime import datetime

import pandas as pd
import torch
import torch.utils.benchmark as benchmark
from config import (
    check_allclose,
    compute_snr,
    expand_grouped_gemm_cases_by_balance,
    filter_grouped_gemm_test_cases,
    gen_grouped_gemm_group_lens,
    gen_grouped_gemm_test_cases,
    get_grouped_gemm_preset_cases,
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


def _format_balance_label(balance: bool) -> str:
    return "balanced" if balance else "imbalanced"


def _format_check_status(correct):
    if correct is None:
        return "SKIP"
    return "PASS" if correct else "FAIL"


def _format_metric(value):
    return "" if value is None else f"{value:.2f}"


def check_grouped_gemm_correctness(x, w, group_lens, out, grad_out, dtype, run_backward=True):
    """Check correctness of BF16 grouped GEMM forward and backward."""
    out_ref = grouped_gemm_ref(x.detach(), w.detach(), group_lens, trans_b=True)
    fwd_correct = check_allclose(out.detach(), out_ref, dtype)

    if not run_backward:
        status = "PASS" if fwd_correct else "FAIL"
        print(f"Correctness Check: {status} (fwd={fwd_correct})")
        return fwd_correct

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


def check_grouped_gemm_fp8_correctness(a, b, out, grad_out, group_lens, fp8_format, run_backward=True):
    """Check correctness of FP8 grouped GEMM forward and backward using SNR."""
    snr_threshold = 25 if fp8_format == Format.E4M3 else 20

    out_ref = grouped_gemm_ref(a.detach(), b.detach(), group_lens, trans_b=True)
    out_snr = compute_snr(out_ref, out.detach())

    if not run_backward:
        correct = out_snr > snr_threshold
        status = "PASS" if correct else "FAIL"
        print(f"Correctness Check: {status} (out={out_snr:.1f}) threshold={snr_threshold}")
        return correct

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


def profile_grouped_gemm(B, M, N, K, dtype, *, balance=True, check_correctness=True, run_backward=True):
    """Profile BF16 Grouped GEMM."""
    device = "cuda"
    x = torch.randn((B * M, K), dtype=dtype, device=device, requires_grad=run_backward)
    w = torch.randn((B, N, K), dtype=dtype, device=device, requires_grad=run_backward)
    group_lens = gen_grouped_gemm_group_lens(B, M, balance=balance).to(device)

    out = turbo.ops.grouped_gemm(x, w, group_lens, trans_b=True)
    grad_out = torch.randn_like(out) if run_backward else None
    correct = None
    if check_correctness:
        correct = check_grouped_gemm_correctness(
            x, w, group_lens, out, grad_out, dtype, run_backward=run_backward
        )

    fwd_func = lambda: turbo.ops.grouped_gemm(x, w, group_lens, trans_b=True)
    if run_backward:
        bwd_func = lambda: out.backward(grad_out, retain_graph=True)
        out = fwd_func()
        bwd_func()

    fwd_total_flops = 2 * B * M * N * K
    bwd_total_flops = 2 * fwd_total_flops if run_backward else None

    for _ in range(20):
        fwd_func()
        if run_backward:
            bwd_func()
    torch.cuda.synchronize()

    fwd_timer = benchmark.Timer(stmt="fn()", globals={"fn": fwd_func})
    fwd_measurement = fwd_timer.timeit(100)

    fwd_mean_time_ms = fwd_measurement.mean * 1e3
    fwd_tflops = fwd_total_flops / (fwd_mean_time_ms * 1e-3) / 1e12
    print(f"Forward  Mean time: {fwd_mean_time_ms:.3f} ms | TFLOPS: {fwd_tflops:.2f}")
    bwd_mean_time_ms = None
    bwd_tflops = None
    if run_backward:
        bwd_timer = benchmark.Timer(stmt="fn()", globals={"fn": bwd_func})
        bwd_measurement = bwd_timer.timeit(100)
        bwd_mean_time_ms = bwd_measurement.mean * 1e3
        bwd_tflops = bwd_total_flops / (bwd_mean_time_ms * 1e-3) / 1e12
        print(f"Backward Mean time: {bwd_mean_time_ms:.3f} ms | TFLOPS: {bwd_tflops:.2f}")

    return fwd_mean_time_ms, fwd_tflops, bwd_mean_time_ms, bwd_tflops, correct


def profile_grouped_gemm_fp8(
    B, M, N, K, dtype, config, *, balance=True, check_correctness=True, run_backward=True
):
    """Profile FP8 Grouped GEMM."""
    device = "cuda"
    group_lens = gen_grouped_gemm_group_lens(B, M, balance=balance).to(device)
    a = torch.randn((B * M, K), dtype=dtype, device=device, requires_grad=run_backward)
    b = torch.randn((B, N, K), dtype=dtype, device=device, requires_grad=run_backward)

    out = turbo.ops.grouped_gemm_fp8(a, b, group_lens, trans_b=True, config=config)
    grad_out = torch.randn_like(out) if run_backward else None
    correct = None
    if check_correctness:
        correct = check_grouped_gemm_fp8_correctness(
            a, b, out, grad_out, group_lens, config.format, run_backward=run_backward
        )

    fwd_func = lambda: turbo.ops.grouped_gemm_fp8(a, b, group_lens, trans_b=True, config=config)
    if run_backward:
        bwd_func = lambda: out.backward(grad_out, retain_graph=True)
        out = fwd_func()
        bwd_func()

    fwd_total_flops = 2 * B * M * N * K
    bwd_total_flops = 2 * fwd_total_flops if run_backward else None

    for _ in range(20):
        fwd_func()
        if run_backward:
            bwd_func()
    torch.cuda.synchronize()

    fwd_timer = benchmark.Timer(stmt="fn()", globals={"fn": fwd_func})
    fwd_measurement = fwd_timer.timeit(100)

    fwd_mean_time_ms = fwd_measurement.mean * 1e3
    fwd_tflops = fwd_total_flops / (fwd_mean_time_ms * 1e-3) / 1e12
    print(f"Forward  Mean time: {fwd_mean_time_ms:.3f} ms | TFLOPS: {fwd_tflops:.2f}")
    bwd_mean_time_ms = None
    bwd_tflops = None
    if run_backward:
        bwd_timer = benchmark.Timer(stmt="fn()", globals={"fn": bwd_func})
        bwd_measurement = bwd_timer.timeit(100)
        bwd_mean_time_ms = bwd_measurement.mean * 1e3
        bwd_tflops = bwd_total_flops / (bwd_mean_time_ms * 1e-3) / 1e12
        print(f"Backward Mean time: {bwd_mean_time_ms:.3f} ms | TFLOPS: {bwd_tflops:.2f}")

    return fwd_mean_time_ms, fwd_tflops, bwd_mean_time_ms, bwd_tflops, correct


def benchmark_grouped_gemm_turbo(
    dtype_name="bf16",
    granularity_name="tensorwise",
    output_csv=None,
    *,
    preset="all",
    case_names=None,
    b_values=None,
    m_values=None,
    n_values=None,
    k_values=None,
    balance_mode="balanced",
    run_backward=True,
    check_correctness=True,
    reuse_b_quant_upper_bound=False,
):
    platform, gpu_name = get_platform_info()

    is_fp8 = dtype_name == "fp8"
    config = GRANULARITY_CONFIG_MAP[granularity_name] if is_fp8 else None

    if reuse_b_quant_upper_bound and not (is_fp8 and granularity_name == "tensorwise"):
        raise ValueError("--reuse-b-quant-upper-bound only supports fp8 tensorwise benchmark runs")

    prev_reuse_env = os.environ.get("PRIMUS_TURBO_FP8_TW_REUSE_B")
    try:
        if reuse_b_quant_upper_bound:
            os.environ["PRIMUS_TURBO_FP8_TW_REUSE_B"] = "1"
        else:
            os.environ.pop("PRIMUS_TURBO_FP8_TW_REUSE_B", None)

        if preset == "all":
            test_cases = gen_grouped_gemm_test_cases()
        else:
            test_cases = get_grouped_gemm_preset_cases(preset)
        test_cases = expand_grouped_gemm_cases_by_balance(test_cases, balance_mode)
        test_cases = filter_grouped_gemm_test_cases(
            test_cases,
            case_names=case_names,
            b_values=b_values,
            m_values=m_values,
            n_values=n_values,
            k_values=k_values,
        )

        if not test_cases:
            raise ValueError("No grouped GEMM test cases selected. Please loosen the filters.")

        print(
            f"Selected {len(test_cases)} grouped GEMM cases | preset={preset} | balance={balance_mode} | "
            f"forward_only={not run_backward} | skip_correctness={not check_correctness} | "
            f"reuse_b_quant_upper_bound={reuse_b_quant_upper_bound}"
        )

        rows = []
        test_id = 0
        for case in test_cases:
            test_id += 1
            B, M, N, K = case["B"], case["M"], case["N"], case["K"]
            dtype = case["dtype"]
            balance = case.get("balance", True)

            print(f"\n{'='*60}")
            if is_fp8:
                print(
                    f"TestID: {test_id}, Case: {case['Case']}, B: {B}, M: {M}, N: {N}, K: {K}, "
                    f"dtype: fp8, granularity: {granularity_name}, balance: {_format_balance_label(balance)}"
                )
            else:
                print(
                    f"TestID: {test_id}, Case: {case['Case']}, B: {B}, M: {M}, N: {N}, K: {K}, "
                    f"dtype: bf16, balance: {_format_balance_label(balance)}"
                )
            print(f"{'='*60}")

            try:
                if is_fp8:
                    fwd_time_ms, fwd_tflops, bwd_time_ms, bwd_tflops, correct = profile_grouped_gemm_fp8(
                        B=B,
                        M=M,
                        N=N,
                        K=K,
                        dtype=dtype,
                        config=config,
                        balance=balance,
                        check_correctness=check_correctness,
                        run_backward=run_backward,
                    )
                else:
                    fwd_time_ms, fwd_tflops, bwd_time_ms, bwd_tflops, correct = profile_grouped_gemm(
                        B=B,
                        M=M,
                        N=N,
                        K=K,
                        dtype=dtype,
                        balance=balance,
                        check_correctness=check_correctness,
                        run_backward=run_backward,
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
                    "Balance": _format_balance_label(balance),
                    "Dtype": dtype_name,
                }
                if is_fp8:
                    row["Granularity"] = granularity_name
                    row["Reuse B Quant"] = "upper_bound" if reuse_b_quant_upper_bound else "off"
                row.update(
                    {
                        "Check": _format_check_status(correct),
                        "Forward Time (ms)": _format_metric(fwd_time_ms),
                        "Forward TFLOPS": _format_metric(fwd_tflops),
                        "Backward Time (ms)": _format_metric(bwd_time_ms),
                        "Backward TFLOPS": _format_metric(bwd_tflops),
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
                    "Balance": _format_balance_label(balance),
                    "Dtype": dtype_name,
                }
                if is_fp8:
                    row["Granularity"] = granularity_name
                    row["Reuse B Quant"] = "upper_bound" if reuse_b_quant_upper_bound else "off"
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

        avg_fwd_tflops = pd.to_numeric(results["Forward TFLOPS"], errors="coerce").mean()
        avg_bwd_tflops = pd.to_numeric(results["Backward TFLOPS"], errors="coerce").mean()
        if not pd.isna(avg_fwd_tflops):
            print(f"\nAverage Forward TFLOPS: {avg_fwd_tflops:.2f}")
        if not pd.isna(avg_bwd_tflops):
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
    finally:
        if prev_reuse_env is None:
            os.environ.pop("PRIMUS_TURBO_FP8_TW_REUSE_B", None)
        else:
            os.environ["PRIMUS_TURBO_FP8_TW_REUSE_B"] = prev_reuse_env


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
    parser.add_argument(
        "--preset",
        type=str,
        choices=["all", "smoke", "tuning", "acceptance"],
        default="all",
        help="Grouped GEMM case preset (default: all)",
    )
    parser.add_argument(
        "--case",
        nargs="*",
        default=None,
        help="Filter case names, e.g. --case DeepSeek-V3-GateUP Kimi-K2-GateUP",
    )
    parser.add_argument("--B", dest="b_values", nargs="*", type=int, default=None, help="Filter B values")
    parser.add_argument("--M", dest="m_values", nargs="*", type=int, default=None, help="Filter M values")
    parser.add_argument("--N", dest="n_values", nargs="*", type=int, default=None, help="Filter N values")
    parser.add_argument("--K", dest="k_values", nargs="*", type=int, default=None, help="Filter K values")
    parser.add_argument(
        "--balance",
        type=str,
        choices=["balanced", "imbalanced", "both"],
        default="balanced",
        help="Group length distribution to benchmark (default: balanced)",
    )
    parser.add_argument(
        "--forward-only",
        action="store_true",
        help="Benchmark forward only and skip backward timing",
    )
    parser.add_argument(
        "--skip-correctness",
        action="store_true",
        help="Skip correctness/SNR checks to speed up tuning",
    )
    parser.add_argument(
        "--reuse-b-quant-upper-bound",
        action="store_true",
        help="Enable tensorwise FP8 weight-quant reuse upper bound for benchmarking",
    )
    args = parser.parse_args()
    benchmark_grouped_gemm_turbo(
        dtype_name=args.dtype,
        granularity_name=args.granularity,
        output_csv=args.output,
        preset=args.preset,
        case_names=args.case,
        b_values=args.b_values,
        m_values=args.m_values,
        n_values=args.n_values,
        k_values=args.k_values,
        balance_mode=args.balance,
        run_backward=not args.forward_only,
        check_correctness=not args.skip_correctness,
        reuse_b_quant_upper_bound=args.reuse_b_quant_upper_bound,
    )
