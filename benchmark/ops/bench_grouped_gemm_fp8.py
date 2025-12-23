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

M_SIZE_LIST = [512, 1024, 2048, 4096, 8192, 16384]
EP_SIZE_LIST = [32, 16, 8]


def compute_snr(ref, actual):
    """Compute Signal-to-Noise Ratio (SNR) in dB."""
    err = ref.to(torch.float64) - actual.to(torch.float64)
    return 20 * torch.log10(ref.to(torch.float64).norm() / err.norm()).item()


def grouped_gemm_ref(a, b, group_lens, trans_b=True):
    """Reference grouped GEMM using PyTorch native matmul."""
    outputs = []
    offset = 0
    for i, glen in enumerate(group_lens.tolist()):
        a_slice = a[offset : offset + glen]
        b_mat = b[i].T if trans_b else b[i]
        outputs.append(a_slice @ b_mat)
        offset += glen
    return torch.cat(outputs, dim=0)


def check_grouped_gemm_fp8_correctness(a, b, out, grad_out, group_lens, trans_b, fp8_format):
    """Check correctness of FP8 grouped GEMM forward and backward using SNR."""
    snr_threshold = 25 if fp8_format == Format.E4M3 else 20

    out_ref = grouped_gemm_ref(a.detach(), b.detach(), group_lens, trans_b=trans_b)
    out_snr = compute_snr(out_ref, out.detach())

    a_ref = a.detach().clone().requires_grad_()
    b_ref = b.detach().clone().requires_grad_()
    out_ref = grouped_gemm_ref(a_ref, b_ref, group_lens, trans_b=trans_b)
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


def profile_grouped_gemm_fp8(B, M, N, K, ori_dtype, config):
    device = "cuda"
    trans_b = True
    group_lens = generate_grouped_gemm_group_lens(B, M, balance=True).to(device)
    b_shape = (B, N, K) if trans_b else (B, K, N)
    a = torch.randn((B * M, K), dtype=ori_dtype, device=device, requires_grad=True)
    b = torch.randn(b_shape, dtype=ori_dtype, device=device, requires_grad=True)

    out = turbo.ops.grouped_gemm_fp8(a, b, group_lens, trans_b=trans_b, config=config)
    grad_out = torch.randn_like(out)
    correct = check_grouped_gemm_fp8_correctness(a, b, out, grad_out, group_lens, trans_b, config.format)

    fwd_func = lambda: turbo.ops.grouped_gemm_fp8(a, b, group_lens, trans_b=trans_b, config=config)
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


def benchmark_grouped_gemm_fp8(granularity_name="tensorwise", output_csv=None):
    full_name = torch.cuda.get_device_name(0)
    match = re.search(r"(MI\d+)", full_name)
    gpu_name = match.group(1) if match else full_name.split()[-1]
    config = GRANULARITY_CONFIG_MAP[granularity_name]

    test_cases = (
        generate_deepseekv2_lite_test_cases()
        + generate_deepseekv2_test_cases()
        + generate_deepseekv3_test_cases()
    )

    rows = []
    test_id = 0
    for case in test_cases:
        test_id += 1
        B, M, N, K = case["B"], case["M"], case["N"], case["K"]
        dtype = case["dtype"]

        print(f"\n{'='*60}")
        print(
            f"TestID: {test_id}, Case: {case['Case']}, B: {B}, M: {M}, N: {N}, K: {K}, "
            f"granularity: {granularity_name}"
        )
        print(f"{'='*60}")

        try:
            fwd_time_ms, fwd_tflops, bwd_time_ms, bwd_tflops, correct = profile_grouped_gemm_fp8(
                B=B, M=M, N=N, K=K, ori_dtype=dtype, config=config
            )
            rows.append(
                {
                    "TestID": test_id,
                    "GPU": gpu_name,
                    "Case": case["Case"],
                    "B": B,
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
                    "Granularity": granularity_name,
                    "Check": "ERROR",
                    "Forward Time (ms)": "N/A",
                    "Forward TFLOPS": "N/A",
                    "Backward Time (ms)": "N/A",
                    "Backward TFLOPS": "N/A",
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
        filename = f"grouped_gemm_fp8_{granularity_name}_benchmark_result_{timestamp}_{gpu_name}.csv"

    results.to_csv(filename, index=False)
    print(f"Results saved to {filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark FP8 Grouped GEMM operations")
    parser.add_argument(
        "--granularity",
        type=str,
        choices=["tensorwise", "rowwise", "blockwise"],
        default="tensorwise",
        help="Scaling granularity (default: tensorwise)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output CSV filename. Default: grouped_gemm_fp8_{granularity}_benchmark_result_{date}_{gpu}.csv",
    )
    args = parser.parse_args()
    benchmark_grouped_gemm_fp8(granularity_name=args.granularity, output_csv=args.output)
