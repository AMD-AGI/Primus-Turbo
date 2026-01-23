###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import argparse
import io
import re
from contextlib import redirect_stdout
from datetime import datetime
from types import SimpleNamespace

import pandas as pd
import torch
import torch.distributed as dist
from tabulate import tabulate
from test_intranode import test_main
from utils import get_deep_ep_backend, init_dist

from benchmark.ops.config import (
    BATCH_SIZE_LIST,
    gen_deepep_test_cases,
    get_platform_info,
)


def profile_intranode(
    args: SimpleNamespace,
    num_sms: int,
    local_rank: int,
    num_ranks: int,
    rank: int,
    buffer,
    group: dist.ProcessGroup,
):
    """
    Call test_main and extract dispatch/combine performance data from the output.

    Returns:
        tuple: (dispatch_time_ms, dispatch_bw, combine_time_ms, combine_bw, correct)
            - dispatch_time_ms: Dispatch time in milliseconds
            - dispatch_bw: Dispatch bandwidth (GB/s)
            - combine_time_ms: Combine time in milliseconds
            - combine_bw: Combine bandwidth (GB/s)
            - correct: Whether the test passed
    """

    # Capture stdout
    captured_output = io.StringIO()
    correct = True

    try:
        with redirect_stdout(captured_output):
            test_main(args, num_sms, local_rank, num_ranks, rank, buffer, group)
    except Exception as e:
        correct = False
        print(f"test_main failed with exception: {e}")

    output = captured_output.getvalue()

    # Parse Best dispatch lines
    # Format: [tuning] Best dispatch (FP8/BF16): SMs X, NVL chunk Y, Z.ZZ GB/s (NVL), t: W.WW us
    dispatch_pattern = r"\[tuning\] Best dispatch \((\w+)\): SMs (\d+), NVL chunk (\d+), ([\d.]+) GB/s \(NVL\), t: ([\d.]+) us"
    dispatch_matches = re.findall(dispatch_pattern, output)

    # Parse Best combine lines
    # Format: [tuning] Best combine: SMs X, NVL chunk Y: Z.ZZ GB/s (NVL), t: W.WW us
    combine_pattern = (
        r"\[tuning\] Best combine: SMs (\d+), NVL chunk (\d+): ([\d.]+) GB/s \(NVL\), t: ([\d.]+) us"
    )
    combine_matches = re.findall(combine_pattern, output)

    # Extract dispatch (forward) data - prefer FP8 over BF16
    dispatch_time_us = 0.0
    dispatch_bw = 0.0
    if dispatch_matches:
        # Prefer FP8 result, otherwise use the last match
        for match in dispatch_matches:
            dtype, sms, nvl_chunk, bw, time_us = match
            dispatch_bw = float(bw)
            dispatch_time_us = float(time_us)
            if dtype == "FP8":
                break

    # Extract combine (backward) data
    combine_time_us = 0.0
    combine_bw = 0.0
    if combine_matches:
        sms, nvl_chunk, bw, time_us = combine_matches[-1]  # Use the last match
        combine_bw = float(bw)
        combine_time_us = float(time_us)

    # Convert time units: us -> ms
    dispatch_time_ms = dispatch_time_us / 1000.0
    combine_time_ms = combine_time_us / 1000.0

    # Mark as failed if no data was parsed
    if dispatch_time_us == 0.0 or combine_time_us == 0.0:
        correct = False

    return dispatch_time_ms, dispatch_bw, combine_time_ms, combine_bw, correct


def benchmark_intranode(local_rank: int, num_local_ranks: int, args: argparse.Namespace):
    platform, gpu_name = get_platform_info()

    num_sms = int(args.num_sms)
    output_csv = args.output

    rows = []
    test_id = 0

    rank, num_ranks, group = init_dist(local_rank, num_local_ranks)

    deep_ep = get_deep_ep_backend(args.backend)

    buffer = deep_ep.Buffer(group, int(2e9), 0, low_latency_mode=False, explicitly_destroy=True)

    test_cases = gen_deepep_test_cases()
    for MBS in BATCH_SIZE_LIST:
        for model_name, num_tokens, hidden, num_experts, num_topk in test_cases:
            test_id += 1
            num_tokens *= MBS

            if local_rank == 0:
                print(f"\n{'='*60}")
                print(
                    f"TestID: {test_id}, Case: {model_name}, MBS: {MBS}, "
                    f"num_tokens: {num_tokens}, hidden: {hidden}, num_topk: {num_topk}, num_experts: {num_experts}"
                )
                print(f"{'='*60}")

            try:
                torch.manual_seed(rank)
                args = SimpleNamespace(
                    num_tokens=num_tokens,
                    hidden=hidden,
                    num_topk=num_topk,
                    num_experts=num_experts,
                    backend=args.backend,
                )

                dispatch_time_ms, dispatch_bw, combine_time_ms, combine_bw, correct = profile_intranode(
                    args, num_sms, local_rank, num_ranks, rank, buffer, group
                )

                row = {
                    "TestID": test_id,
                    "Platform": platform,
                    "GPU": gpu_name,
                    "Case": model_name,
                    "MBS": MBS,
                    "num_tokens": num_tokens,
                    "hidden": hidden,
                    "num_topk": num_topk,
                    "num_experts": num_experts,
                }
                row.update(
                    {
                        "Check": "PASS" if correct else "FAIL",
                        "Dispatch Time (ms)": f"{dispatch_time_ms:.2f}",
                        "Dispatch Bandwidth (GB/s)": f"{dispatch_bw:.2f}",
                        "Combine Time (ms)": f"{combine_time_ms:.2f}",
                        "Combine Bandwidth (GB/s)": f"{combine_bw:.2f}",
                    }
                )
                rows.append(row)

            except Exception as e:
                if local_rank == 0:
                    print(f"Failed: {str(e)}")
                row = {
                    "TestID": test_id,
                    "Platform": platform,
                    "GPU": gpu_name,
                    "Case": model_name,
                    "MBS": MBS,
                    "num_tokens": num_tokens,
                    "hidden": hidden,
                    "num_topk": num_topk,
                    "num_experts": num_experts,
                }
                row.update(
                    {
                        "Check": "ERROR",
                        "Dispatch Time (ms)": "ERROR",
                        "Dispatch Bandwidth (GB/s)": "0.00",
                        "Combine Time (ms)": "ERROR",
                        "Combine Bandwidth (GB/s)": "0.00",
                    }
                )
                rows.append(row)

    results = pd.DataFrame(rows)
    if local_rank == 0:
        print("\nFinal Results:")
        print(tabulate(results, headers="keys", tablefmt="grid", showindex=False))

    avg_dispatch_bw = results["Dispatch Bandwidth (GB/s)"].astype(float).mean()
    avg_combine_bw = results["Combine Bandwidth (GB/s)"].astype(float).mean()

    if local_rank == 0:
        print(f"\nAverage Dispatch Bandwidth: {avg_dispatch_bw:.2f}")
        print(f"Average Combine Bandwidth: {avg_combine_bw:.2f}")

    if output_csv:
        filename = output_csv
    else:
        timestamp = datetime.now().strftime("%Y%m%d")
        filename = f"deep_ep_intranode_benchmark_results_{timestamp}_{gpu_name}.csv"

    if local_rank == 0:

        results.to_csv(filename, index=False)

        print(f"Results saved to {filename}")

    # Destroy the buffer runtime and communication group
    buffer.destroy()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test intranode EP kernels")
    parser.add_argument(
        "--num-processes", type=int, default=8, help="Number of processes to spawn (default: 8)"
    )
    parser.add_argument("--backend", type=str, default="turbo", help="Backend to use (default: turbo)")
    parser.add_argument("--num-sms", type=int, default=64, help="Number of SMs to use (default: 32)")
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output CSV filename",
    )
    args = parser.parse_args()

    num_processes = args.num_processes
    torch.multiprocessing.spawn(benchmark_intranode, args=(num_processes, args), nprocs=num_processes)
