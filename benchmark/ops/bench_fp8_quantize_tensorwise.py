###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Benchmark FP8 tensorwise quantization: original 3-kernel vs fused 2-kernel.

The original path (quantize_fp8_tensorwise):
  1. reduce_row (abs-max)          -> amax       (HIP kernel)
  2. compute_scale_from_amax       -> scale/inv  (HIP kernel)
  3. quantize_tensorwise_impl      -> fp8_output (HIP kernel)

The fused path (quantize_fp8_tensorwise_fused):
  1. reduce_amax_and_compute_scale -> scale/inv  (HIP kernel, fused)
  2. quantize_tensorwise_impl      -> fp8_output (HIP kernel)

Usage:
    cd benchmark/ops
    python bench_fp8_quantize_tensorwise.py [--warmup 30] [--repeat 300]
"""

import argparse

import torch

import primus_turbo.pytorch as turbo
from primus_turbo.pytorch.core.low_precision import ScalingGranularity
from primus_turbo.pytorch.ops import quantize_fp8
from primus_turbo.pytorch.ops.quantization import quantize_fp8_fused

DEVICE = "cuda:0"
DTYPE = torch.bfloat16
FP8_DTYPE = turbo.float8_e4m3

SHAPES = [
    (4096, 3072),  # 12.6M
    (4096, 12288),  # 50.3M
    (12288, 4096),  # 50.3M
    (16384, 3072),  # 50.3M
    (8192, 8192),  # 67.1M
    (32768, 3072),  # 100.7M
    (1, 352321536),  # 352M  - full model param flat
]


def quantize_original(x):
    """Original 3-kernel path."""
    return quantize_fp8(x, FP8_DTYPE, granularity=ScalingGranularity.TENSORWISE)


def quantize_fused_path(x):
    """Fused 2-kernel path."""
    return quantize_fp8_fused(x, FP8_DTYPE, granularity=ScalingGranularity.TENSORWISE)


def benchmark_fn(fn, warmup, repeat, *args):
    """Return (median_us, min_us) using CUDA events."""
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(repeat)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(repeat)]

    for i in range(repeat):
        starts[i].record()
        fn(*args)
        ends[i].record()
    torch.cuda.synchronize()

    times = sorted(s.elapsed_time(e) * 1000 for s, e in zip(starts, ends))
    return times[len(times) // 2], times[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--repeat", type=int, default=300)
    args = parser.parse_args()

    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Warmup: {args.warmup}, Repeat: {args.repeat}")

    print(
        f"\n{'Shape':>20s}  {'Numel':>12s}  "
        f"{'Original':>12s}  {'Fused':>12s}  "
        f"{'Saved':>8s}  {'Speedup':>8s}  "
        f"{'Min orig':>10s}  {'Min fused':>10s}"
    )
    print(
        f"{'':>20s}  {'':>12s}  "
        f"{'3-kern us':>12s}  {'2-kern us':>12s}  "
        f"{'us':>8s}  {'':>8s}  "
        f"{'us':>10s}  {'us':>10s}"
    )
    print("-" * 110)

    for shape in SHAPES:
        x = torch.randn(*shape, dtype=DTYPE, device=DEVICE)
        numel = x.numel()

        orig_med, orig_min = benchmark_fn(quantize_original, args.warmup, args.repeat, x)
        fused_med, fused_min = benchmark_fn(quantize_fused_path, args.warmup, args.repeat, x)

        saved = orig_med - fused_med
        speedup = orig_med / fused_med if fused_med > 0 else float("inf")
        shape_str = f"{shape[0]}x{shape[1]}"
        print(
            f"{shape_str:>20s}  {numel:>12,d}  "
            f"{orig_med:>12.1f}  {fused_med:>12.1f}  "
            f"{saved:>8.1f}  {speedup:>7.2f}x  "
            f"{orig_min:>10.1f}  {fused_min:>10.1f}"
        )

    # Correctness check
    print("\nCorrectness check (largest shape):")
    x = torch.randn(*SHAPES[-1], dtype=DTYPE, device=DEVICE)
    out_orig, si_orig = quantize_original(x)
    out_fused, si_fused = quantize_fused_path(x)

    match = torch.equal(out_orig, out_fused)
    scale_close = torch.allclose(si_orig, si_fused, rtol=1e-5, atol=1e-8)
    print(f"  Output match:    {match}")
    print(f"  Scale_inv close: {scale_close} " f"(orig={si_orig.item():.8f}, fused={si_fused.item():.8f})")


if __name__ == "__main__":
    main()
