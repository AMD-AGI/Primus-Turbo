###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""hipBLASLt algorithm tuning benchmark.

Runs each dense-model GEMM shape (MBS=1) with level-2 AutoTune so
that the framework benchmarks all hipBLASLt heuristic algorithms and
prints the winner for each shape.  The tuning logs come from the
AutoKernelDispatcher logger.

By default all dtype/granularity combinations are tested.  Use --dtype
and --granularity to restrict to a single combination.

Usage:
    python benchmark/ops/bench_hipblaslt_algo_tuning.py                  # all combos
    python benchmark/ops/bench_hipblaslt_algo_tuning.py --dtype bf16     # bf16 only
    python benchmark/ops/bench_hipblaslt_algo_tuning.py --dtype fp8 --granularity tensorwise
    python benchmark/ops/bench_hipblaslt_algo_tuning.py --max-algos 5
"""

import argparse
import os
import sys
from datetime import datetime

os.environ.setdefault("PRIMUS_TURBO_LOG_LEVEL", "INFO")

import torch  # noqa: E402
from config import (  # noqa: E402
    DenseModelConfigs,
    gen_gemm_test_cases,
    get_platform_info,
)

import primus_turbo.pytorch as turbo  # noqa: E402
from primus_turbo.pytorch.core.backend import GlobalBackendManager  # noqa: E402
from primus_turbo.pytorch.core.low_precision import (  # noqa: E402
    Float8QuantConfig,
    Format,
    ScaleDtype,
    ScalingGranularity,
)

FP8_GRANULARITY_CONFIGS = {
    "tensorwise": Float8QuantConfig(format=Format.E4M3, granularity=ScalingGranularity.TENSORWISE),
    "rowwise": Float8QuantConfig(format=Format.E4M3, granularity=ScalingGranularity.ROWWISE),
    "blockwise": Float8QuantConfig(
        format=Format.E4M3,
        granularity=ScalingGranularity.BLOCKWISE,
        block_size=128,
    ),
}


def run_shapes(dtype_label, device, fp8_config=None):
    """Run all dense-model GEMM shapes for a given dtype/granularity."""
    for model_name, model_config in DenseModelConfigs.items():
        test_cases = gen_gemm_test_cases(model_config)
        for shape in test_cases:
            M, N, K = shape[0], shape[1], shape[2]

            print(f"{'='*60}")
            print(f"Case: {model_name}, M={M}, N={N}, K={K}, dtype={dtype_label}")
            print(f"{'='*60}")

            a = torch.randn((M, K), dtype=torch.bfloat16, device=device)
            b = torch.randn((N, K), dtype=torch.bfloat16, device=device)

            if fp8_config is not None:
                turbo.ops.gemm_fp8(a, b, trans_b=True, config=fp8_config)
                turbo.ops.gemm_fp8(a, b, trans_b=True, config=fp8_config)
            else:
                turbo.ops.gemm(a, b, trans_b=True)
                turbo.ops.gemm(a, b, trans_b=True)

            torch.cuda.synchronize()
            print()


def main():
    parser = argparse.ArgumentParser(description="hipBLASLt Algorithm Tuning Benchmark")
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["bf16", "fp8", "all"],
        default="all",
        help="Data type to test (default: all)",
    )
    parser.add_argument(
        "--granularity",
        type=str,
        choices=["tensorwise", "rowwise", "blockwise", "all"],
        default="all",
        help="FP8 scaling granularity (only used when dtype includes fp8, default: all)",
    )
    parser.add_argument(
        "--max-algos",
        type=int,
        default=100,
        help="Max hipBLASLt algorithms to try per shape (default: 10)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output log file (default: auto-generated name). Use 'stdout' to skip file output.",
    )
    args = parser.parse_args()

    os.environ["PRIMUS_TURBO_HIPBLASLT_TUNE_MAX_ALGOS"] = str(args.max_algos)
    GlobalBackendManager.set_auto_tune(2)

    platform, gpu_name = get_platform_info()
    device = "cuda"

    log_file = None
    if args.output != "stdout":
        filename = args.output or (
            f"hipblaslt_algo_tuning_{args.dtype}_{args.granularity}"
            f"_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{gpu_name}.log"
        )
        log_file = open(filename, "w")

        # Tee output to both stdout and file
        class Tee:
            def __init__(self, *streams):
                self.streams = streams

            def write(self, data):
                for s in self.streams:
                    s.write(data)
                    s.flush()

            def flush(self):
                for s in self.streams:
                    s.flush()

        sys.stdout = Tee(sys.__stdout__, log_file)
        sys.stderr = Tee(sys.__stderr__, log_file)

        # Point existing logger handlers at the new stderr so their
        # output is captured in the log file too.
        import logging

        for handler in logging.getLogger("primus_turbo").handlers:
            if isinstance(handler, logging.StreamHandler):
                handler.stream = sys.stdout

    print(f"Platform: {platform}, GPU: {gpu_name}")
    print(f"AutoTune level: 2, hipBLASLt algo cap: {args.max_algos}")
    print(f"Dtype: {args.dtype}, Granularity: {args.granularity}\n")

    run_bf16 = args.dtype in ("bf16", "all")
    run_fp8 = args.dtype in ("fp8", "all")

    if run_bf16:
        print(f"\n{'#'*60}")
        print(f"# BF16 GEMM")
        print(f"{'#'*60}\n")
        run_shapes("bf16", device)
        GlobalBackendManager.reset()
        GlobalBackendManager.set_auto_tune(2)

    if run_fp8:
        granularities = (
            list(FP8_GRANULARITY_CONFIGS.keys()) if args.granularity == "all" else [args.granularity]
        )
        for gran in granularities:
            print(f"\n{'#'*60}")
            print(f"# FP8 GEMM ({gran})")
            print(f"{'#'*60}\n")
            run_shapes(f"fp8-{gran}", device, fp8_config=FP8_GRANULARITY_CONFIGS[gran])
            GlobalBackendManager.reset()
            GlobalBackendManager.set_auto_tune(2)

    GlobalBackendManager.reset()
    print("Done.")

    if log_file is not None:
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        log_file.close()
        print(f"Log saved to {filename}")


if __name__ == "__main__":
    main()
