###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import argparse
import os
from datetime import datetime

import pandas as pd
from tabulate import tabulate

# GPU Configuration
AMD_GPUS = ["MI325"]
NV_GPUS = ["H200"]

AMD_PLATFORM = "ROCm"
NV_PLATFORM = "CUDA"

# AMD Benchmark Ops
AMD_BENCHMARK_OPS = {
    "Attention": {
        "PyTorch": "attention_torch_benchmark.csv",
        "Turbo(Aiter/CK)": "attention_benchmark.csv",
    },
    "Attention-FP8": {
        "Turbo(Triton)": "attention_fp8_benchmark.csv",
    },
    "GEMM": {
        "PyTorch": "gemm_torch_benchmark.csv",
        "TE": "gemm_te_benchmark.csv",
        "Turbo(Hipblaslt)": "gemm_hipblaslt_benchmark.csv",
    },
    "GEMM-FP8-Tensorwise": {
        "PyTorch": "gemm_fp8_tensorwise_torch_benchmark.csv",
        "TE": "gemm_fp8_tensorwise_te_benchmark.csv",
        "Turbo(Hipblaslt)": "gemm_fp8_tensorwise_hipblaslt_benchmark.csv",
        "Turbo(CK)": "gemm_fp8_tensorwise_ck_benchmark.csv",
        "Turbo(AutoTune)": "gemm_fp8_tensorwise_autotune_benchmark.csv",
    },
    "GEMM-FP8-Rowwise": {
        "Turbo(CK)": "gemm_fp8_rowwise_ck_benchmark.csv",
    },
    "GEMM-FP8-Blockwise": {
        "Turbo(CK)": "gemm_fp8_blockwise_ck_benchmark.csv",
    },
    "GEMM-MXFP8": {
        "Turbo(Hipblaslt)": "gemm_mxfp8_hipblaslt_benchmark.csv",
    },
    "Grouped-GEMM": {
        "PyTorch": "grouped_gemm_torch_benchmark.csv",
        "TE": "grouped_gemm_te_benchmark.csv",
        "Turbo(Hipblaslt)": "grouped_gemm_hipblaslt_benchmark.csv",
        "Turbo(CK)": "grouped_gemm_ck_benchmark.csv",
        "Turbo(AutoTune)": "grouped_gemm_autotune_benchmark.csv",
    },
    "Grouped-GEMM-FP8-Tensorwise": {
        "TE": "grouped_gemm_fp8_tensorwise_te_benchmark.csv",
        "Turbo(Hipblaslt)": "grouped_gemm_fp8_tensorwise_hipblaslt_benchmark.csv",
        "Turbo(CK)": "grouped_gemm_fp8_tensorwise_ck_benchmark.csv",
        "Turbo(AutoTune)": "grouped_gemm_fp8_tensorwise_autotune_benchmark.csv",
    },
    "Grouped-GEMM-FP8-Rowwise": {
        "Turbo(CK)": "grouped_gemm_fp8_rowwise_ck_benchmark.csv",
    },
    "Grouped-GEMM-FP8-Blockwise": {
        "Turbo(CK)": "grouped_gemm_fp8_blockwise_ck_benchmark.csv",
    },
}

# NVIDIA Benchmark Ops
NV_BENCHMARK_OPS = {
    "Attention": {
        "PyTorch": "attention_torch_benchmark.csv",
    },
    "GEMM": {
        "PyTorch": "gemm_torch_benchmark.csv",
        "TE": "gemm_te_benchmark.csv",
    },
    "GEMM-FP8-Tensorwise": {
        "PyTorch": "gemm_fp8_tensorwise_torch_benchmark.csv",
        "TE": "gemm_fp8_tensorwise_te_benchmark.csv",
    },
    "Grouped-GEMM": {
        "PyTorch": "grouped_gemm_torch_benchmark.csv",
        "TE": "grouped_gemm_te_benchmark.csv",
    },
    "Grouped-GEMM-FP8-Tensorwise": {
        "TE": "grouped_gemm_fp8_tensorwise_te_benchmark.csv",
    },
}


def get_benchmark_ops(gpu):
    if gpu in AMD_GPUS:
        return AMD_BENCHMARK_OPS
    elif gpu in NV_GPUS:
        return NV_BENCHMARK_OPS
    else:
        return {}


def get_expected_platform(gpu):
    if gpu in AMD_GPUS:
        return AMD_PLATFORM
    elif gpu in NV_GPUS:
        return NV_PLATFORM
    else:
        return None


def get_avg_tflops(data_dir, csv_filename, expected_gpu=None, expected_platform=None):
    csv_path = os.path.join(data_dir, csv_filename)
    if not os.path.exists(csv_path):
        return 0, 0

    df = pd.read_csv(csv_path)

    if expected_gpu and "GPU" in df.columns:
        for gpu in df["GPU"].unique():
            if not gpu.startswith(expected_gpu):
                print(f"Warning: GPU mismatch in {csv_path}: expected '{expected_gpu}*', found '{gpu}'")

    if expected_platform and "Platform" in df.columns:
        for plat in df["Platform"].unique():
            if plat != expected_platform:
                print(
                    f"Warning: Platform mismatch in {csv_path}: expected '{expected_platform}', found '{plat}'"
                )

    fwd_tflops = pd.to_numeric(df["Forward TFLOPS"], errors="coerce").mean()
    bwd_tflops = pd.to_numeric(df["Backward TFLOPS"], errors="coerce").mean()

    return fwd_tflops if not pd.isna(fwd_tflops) else 0, bwd_tflops if not pd.isna(bwd_tflops) else 0


def generate_summary_table(data_dir, date_str, gpus, output_file=None):
    gpus_to_process = gpus if isinstance(gpus, list) else [gpus]

    summary_data = []
    idx = 1
    for gpu in gpus_to_process:
        benchmark_ops = get_benchmark_ops(gpu)
        if not benchmark_ops:
            print(f"Warning: Unknown GPU '{gpu}', skipping...")
            continue

        gpu_data_dir = os.path.join(data_dir, gpu)
        if not os.path.exists(gpu_data_dir):
            print(f"Warning: Data directory '{gpu_data_dir}' not found, skipping {gpu}...")
            continue

        csv_files = [f for f in os.listdir(gpu_data_dir) if f.endswith(".csv")]
        if not csv_files:
            print(f"Warning: No CSV files in '{gpu_data_dir}', skipping {gpu}...")
            continue

        expected_platform = get_expected_platform(gpu)

        for op_name, frameworks in benchmark_ops.items():
            for framework_name, csv_filename in frameworks.items():
                fwd_tflops, bwd_tflops = get_avg_tflops(
                    gpu_data_dir, csv_filename, expected_gpu=gpu, expected_platform=expected_platform
                )
                summary_data.append(
                    {
                        "#": idx,
                        "Op": op_name,
                        "GPU:Framework:Stage": f"{gpu}:{framework_name}:Fwd",
                        date_str: f"{fwd_tflops:.2f}",
                    }
                )
                idx += 1
                summary_data.append(
                    {
                        "#": idx,
                        "Op": op_name,
                        "GPU:Framework:Stage": f"{gpu}:{framework_name}:Bwd",
                        date_str: f"{bwd_tflops:.2f}",
                    }
                )
                idx += 1

    if not summary_data:
        print(f"\nWarning: No benchmark data found for GPU: {', '.join(gpus_to_process)}")
        return pd.DataFrame()

    summary_df = pd.DataFrame(summary_data)

    print(f"\n{'='*80}")
    print(f"  Summary Table - GPU: {', '.join(gpus_to_process)}")
    print(f"{'='*80}\n")
    print(tabulate(summary_df, headers="keys", tablefmt="grid", showindex=False))

    if output_file:
        summary_df.to_csv(output_file, index=False)
        print(f"\n[Saved to {output_file}]")

    return summary_df


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize daily benchmark results")
    parser.add_argument(
        "--data-dir", type=str, required=True, help="Date directory containing GPU subdirectories"
    )
    parser.add_argument(
        "--gpu",
        type=str,
        nargs="+",
        default=["MI325", "H200"],
        help="GPU(s) to summarize (default: MI325 H200)",
    )
    parser.add_argument(
        "--date", type=str, default=None, help="Date string for the summary table (default: today)"
    )
    parser.add_argument("-o", "--output", type=str, default=None, help="Output CSV file path")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    date_str = args.date or datetime.now().strftime("%Y-%m-%d")
    generate_summary_table(args.data_dir, date_str, args.gpu, args.output)
