###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import pandas as pd
import torch
import torch.utils.benchmark as benchmark
from tabulate import tabulate

from primus_turbo.pytorch.core.float8 import (
    Float8QuantConfig,
    Format,
    ScalingGranularity,
)
from primus_turbo.pytorch.ops import grouped_gemm_fp8
from tests.pytorch.ref.gemm_ref import (
    generate_grouped_gemm_group_lens,
    grouped_gemm_ref,
)
from tests.pytorch.test_utils import compute_snr

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


def bench_grouped_gemm_fp8(B, M, N, K, ori_dtype, format, granularity, trans_b, balance):
    device = "cuda"
    # Prepare inputs
    group_lens = generate_grouped_gemm_group_lens(B, M, balance=balance).to(device)
    b_shape = (B, N, K) if trans_b else (B, K, N)
    a = torch.randn((B * M, K), dtype=ori_dtype, device=device, requires_grad=True)
    b = torch.randn(b_shape, dtype=ori_dtype, device=device, requires_grad=True)
    a_ref = a.detach().clone().requires_grad_(True)
    b_ref = b.detach().clone().requires_grad_(True)

    config = Float8QuantConfig(format=format, granularity=granularity)

    # Reference forward pass
    out_ref = grouped_gemm_ref(a_ref, b_ref, group_lens, trans_b=trans_b)
    grad_out = torch.randn_like(out_ref)
    out_ref.backward(grad_out, retain_graph=True)

    # Forward pass for implementation
    fwd_func = lambda: grouped_gemm_fp8(a, b, group_lens, trans_b=trans_b, config=config)
    bwd_func = lambda: out.backward(grad_out, retain_graph=True)
    out = fwd_func()
    bwd_func()

    # Compute SNRs
    out_snr = compute_snr(out_ref, out)
    a_grad_snr = compute_snr(a_ref.grad, a.grad)
    b_grad_snr = compute_snr(b_ref.grad, b.grad)

    if out_snr <= 20:
        print(f"out_snr too low: {out_snr}")
    if a_grad_snr <= 20:
        print(f"x_grad_snr too low: {a_grad_snr}")
    if b_grad_snr <= 20:
        print(f"w_grad_snr too low: {b_grad_snr}")

    assert out_snr > 20, "out_snr too low"
    assert a_grad_snr > 20, "x_grad_snr too low"
    assert b_grad_snr > 20, "w_grad_snr too low"
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


if __name__ == "__main__":
    dsv2_lite_test_cases = generate_deepseekv2_lite_test_cases()
    dsv2_test_cases = generate_deepseekv2_test_cases()
    dsv3_test_cases = generate_deepseekv3_test_cases()
    test_cases = dsv2_lite_test_cases + dsv2_test_cases + dsv3_test_cases

    # DataFrame to store results
    results = pd.DataFrame(
        columns=[
            "TestID",
            "Case",
            "B",
            "M",
            "N",
            "K",
            "format",
            "granularity",
            "Forward Time (ms)",
            "Forward TFLOPS",
            "Backward Time (ms)",
            "Backward TFLOPS",
        ]
    )
    test_id = 0
    format = Format.E4M3
    granularity = ScalingGranularity.TENSORWISE
    for case in test_cases:
        B = case["B"]
        M = case["M"]
        N = case["N"]
        K = case["K"]
        dtype = case["dtype"]

        print(f"\n{'='*50}")
        print(f"Testing Case: {case}")
        print(f"{'='*50}")

        trans_b = True
        balance = True
        test_id += 1
        try:
            # Run benchmark
            fwd_time_ms, fwd_tflops, bwd_time_ms, bwd_tflops = bench_grouped_gemm_fp8(
                B=B,
                M=M,
                N=N,
                K=K,
                ori_dtype=dtype,
                format=format,
                granularity=granularity,
                trans_b=trans_b,
                balance=balance,
            )
            # Add to results table
            new_row = {
                "TestID": test_id,
                "Case": case["Case"],
                "B": B,
                "M": M,
                "N": N,
                "K": K,
                "format": format,
                "granularity": granularity,
                "Forward Time (ms)": f"{fwd_time_ms:.2f}",
                "Forward TFLOPS": f"{fwd_tflops:.2f}",
                "Backward Time (ms)": f"{bwd_time_ms:.2f}",
                "Backward TFLOPS": f"{bwd_tflops:.2f}",
            }
            results = pd.concat([results, pd.DataFrame([new_row])], ignore_index=True)
        except Exception as e:
            print(f"Failed to run {case}: {str(e)}")
            new_row = {
                "TestID": test_id,
                "Case": case["Case"],
                "B": B,
                "M": M,
                "N": N,
                "K": K,
                "format": format,
                "granularity": granularity,
                "Forward Time (ms)": "N/A",
                "Forward TFLOPS": "N/A",
                "Backward Time (ms)": "N/A",
                "Backward TFLOPS": "N/A",
            }
            results = pd.concat([results, pd.DataFrame([new_row])], ignore_index=True)

    # Print results
    print("\nFinal Results:")
    print(tabulate(results, headers="keys", tablefmt="grid", showindex=False))

    # Save to CSV
    results.to_csv("grouped_gemm_fp8_benchmark_results.csv", index=False)
    print("Results saved to grouped_gemm_fp8_benchmark_results.csv")
