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

# Benchmark results: Op -> Backend -> csv filename
BENCHMARK_TABLES = {
    "Attention": {
        "Aiter/CK": "attention_benchmark.csv",
    },
    "Attention-FP8": {
        "Triton": "attention_fp8_benchmark.csv",
    },
    "GEMM": {
        "Hipblaslt": "gemm_hipblaslt_benchmark.csv",
    },
    "GEMM-FP8-Tensorwise": {
        "Hipblaslt": "gemm_fp8_tensorwise_hipblaslt_benchmark.csv",
        "CK": "gemm_fp8_tensorwise_ck_benchmark.csv",
        "AutoTune": "gemm_fp8_tensorwise_autotune_benchmark.csv",
    },
    "GEMM-FP8-Rowwise": {
        "CK": "gemm_fp8_rowwise_ck_benchmark.csv",
    },
    "GEMM-FP8-Blockwise": {
        "CK": "gemm_fp8_blockwise_ck_benchmark.csv",
    },
    # MXFP8 (MI355 only, not available on MI325)
    "GEMM-MXFP8": {
        "Hipblaslt": "gemm_mxfp8_hipblaslt_benchmark.csv",
    },
    "Grouped-GEMM": {
        "Hipblaslt": "grouped_gemm_hipblaslt_benchmark.csv",
        "CK": "grouped_gemm_ck_benchmark.csv",
        "AutoTune": "grouped_gemm_autotune_benchmark.csv",
    },
    "Grouped-GEMM-FP8-Tensorwise": {
        "Hipblaslt": "grouped_gemm_fp8_tensorwise_hipblaslt_benchmark.csv",
        "CK": "grouped_gemm_fp8_tensorwise_ck_benchmark.csv",
        "AutoTune": "grouped_gemm_fp8_tensorwise_autotune_benchmark.csv",
    },
    "Grouped-GEMM-FP8-Rowwise": {
        "CK": "grouped_gemm_fp8_rowwise_ck_benchmark.csv",
    },
    "Grouped-GEMM-FP8-Blockwise": {
        "CK": "grouped_gemm_fp8_blockwise_ck_benchmark.csv",
    },
}


def get_csv_path(data_dir, csv_filename):
    """Get full path to a CSV file."""
    return os.path.join(data_dir, csv_filename)


def get_avg_tflops(data_dir, csv_filename):
    """Get average Forward and Backward TFLOPS from a CSV file."""
    csv_path = get_csv_path(data_dir, csv_filename)
    if not os.path.exists(csv_path):
        return 0, 0

    df = pd.read_csv(csv_path)

    # Handle N/A values
    fwd_col = "Forward TFLOPS"
    bwd_col = "Backward TFLOPS"

    fwd_tflops = pd.to_numeric(df[fwd_col], errors="coerce").mean()
    bwd_tflops = pd.to_numeric(df[bwd_col], errors="coerce").mean()

    return fwd_tflops if not pd.isna(fwd_tflops) else 0, bwd_tflops if not pd.isna(bwd_tflops) else 0


def generate_summary_table(data_dir, date_str, output_file=None):
    """Generate a summary table with Op, Backend:Stage, and average TFLOPS."""
    # Build summary data: each row is Op, Backend:Stage, TFLOPS
    summary_data = []
    idx = 1
    for op_name, backends in BENCHMARK_TABLES.items():
        for backend_name, csv_filename in backends.items():
            fwd_tflops, bwd_tflops = get_avg_tflops(data_dir, csv_filename)
            # Forward row
            summary_data.append(
                {
                    "#": idx,
                    "Op": op_name,
                    "Backend:Stage": f"{backend_name}:Fwd",
                    date_str: f"{fwd_tflops:.2f}",
                }
            )
            idx += 1
            # Backward row
            summary_data.append(
                {
                    "#": idx,
                    "Op": op_name,
                    "Backend:Stage": f"{backend_name}:Bwd",
                    date_str: f"{bwd_tflops:.2f}",
                }
            )
            idx += 1

    summary_df = pd.DataFrame(summary_data)

    print(f"\n{'='*80}")
    print(f"  Summary Table")
    print(f"{'='*80}\n")
    print(tabulate(summary_df, headers="keys", tablefmt="grid", showindex=False))

    # Save to CSV if output file is specified
    if output_file:
        summary_df.to_csv(output_file, index=False)
        print(f"\n[Saved to {output_file}]")

    return summary_df


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize daily benchmark results")
    parser.add_argument(
        "--data-dir", type=str, required=True, help="Directory containing benchmark CSV files"
    )
    parser.add_argument(
        "--date", type=str, default=None, help="Date string for the summary table (default: today)"
    )
    parser.add_argument("-o", "--output", type=str, default=None, help="Output CSV file path")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    date_str = args.date or datetime.now().strftime("%Y-%m-%d")

    generate_summary_table(args.data_dir, date_str, args.output)
