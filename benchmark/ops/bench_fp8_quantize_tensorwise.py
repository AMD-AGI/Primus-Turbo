###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Benchmark FP8 tensorwise quantization variants.

  1. Original (quantize_fp8_tensorwise):        3 kernels
  2. Fused (quantize_fp8_tensorwise_fused):      2 kernels
  3. Fused + separate amax:                      2 + 1 kernels (delayed scaling baseline)
  4. Fused with amax_out:                        1 kernel  (delayed scaling optimized)

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


def quantize_only_with_scale(x, scale):
    """Quantize-only with pre-computed scale (1 kernel). Lower bound for delayed."""
    fp8_out, scale_inv = torch.ops.primus_turbo_cpp_extension.quantize_fp8_tensorwise_fused(
        x, FP8_DTYPE, scale
    )
    return fp8_out, scale_inv


def quantize_fused_plus_separate_amax(x, scale, amax_buf):
    """Delayed scaling baseline: fused quantize with pre-computed scale + separate amax."""
    fp8_out, scale_inv = torch.ops.primus_turbo_cpp_extension.quantize_fp8_tensorwise_fused(
        x, FP8_DTYPE, scale
    )
    amax_buf.copy_(x.detach().abs().amax())
    return fp8_out, scale_inv


def quantize_fused_with_amax_out(x, scale, amax_buf):
    """Delayed scaling optimized: fused quantize + amax in a single kernel."""
    fp8_out, scale_inv = torch.ops.primus_turbo_cpp_extension.quantize_fp8_tensorwise_fused(
        x, FP8_DTYPE, scale, amax_buf
    )
    return fp8_out, scale_inv



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

    # --- Section 1: Dynamic scaling comparison (original vs fused) ---
    print("\n=== Dynamic Scaling: Original (3-kern) vs Fused (2-kern) ===")
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

    # --- Section 2: Delayed scaling full comparison ---
    print("\n=== Delayed Scaling: Full Comparison (all with pre-computed scale) ===")
    print(
        f"\n{'Shape':>20s}  {'Numel':>12s}  "
        f"{'Q-only':>10s}  {'Q+sepAmax':>10s}  {'Q&Amax':>10s}  {'Dynamic':>10s}  "
        f"{'Amax OH':>8s}  {'vs Dyn':>8s}"
    )
    print(
        f"{'':>20s}  {'':>12s}  "
        f"{'1kern us':>10s}  {'2+kern us':>10s}  {'1kern us':>10s}  {'2kern us':>10s}  "
        f"{'us':>8s}  {'us':>8s}"
    )
    print("-" * 105)

    fp8_max = torch.finfo(FP8_DTYPE).max
    for shape in SHAPES:
        x = torch.randn(*shape, dtype=DTYPE, device=DEVICE)
        numel = x.numel()
        scale = torch.tensor(fp8_max / x.abs().amax().item(), dtype=torch.float32, device=DEVICE)
        amax_buf = torch.zeros((), dtype=torch.float32, device=DEVICE)

        qonly_med, _ = benchmark_fn(
            quantize_only_with_scale, args.warmup, args.repeat, x, scale
        )
        sep_med, _ = benchmark_fn(
            quantize_fused_plus_separate_amax, args.warmup, args.repeat, x, scale, amax_buf
        )
        fused_med, _ = benchmark_fn(
            quantize_fused_with_amax_out, args.warmup, args.repeat, x, scale, amax_buf
        )
        dyn_med, _ = benchmark_fn(
            quantize_fused_path, args.warmup, args.repeat, x
        )

        amax_overhead = fused_med - qonly_med
        vs_dynamic = fused_med - dyn_med
        shape_str = f"{shape[0]}x{shape[1]}"
        print(
            f"{shape_str:>20s}  {numel:>12,d}  "
            f"{qonly_med:>10.1f}  {sep_med:>10.1f}  {fused_med:>10.1f}  {dyn_med:>10.1f}  "
            f"{amax_overhead:>8.1f}  {vs_dynamic:>8.1f}"
        )

    # --- Correctness check ---
    print("\nCorrectness check (largest shape):")
    x = torch.randn(*SHAPES[-1], dtype=DTYPE, device=DEVICE)
    out_orig, si_orig = quantize_original(x)
    out_fused, si_fused = quantize_fused_path(x)

    match = torch.equal(out_orig, out_fused)
    scale_close = torch.allclose(si_orig, si_fused, rtol=1e-5, atol=1e-8)
    print(f"  Output match:    {match}")
    print(f"  Scale_inv close: {scale_close} " f"(orig={si_orig.item():.8f}, fused={si_fused.item():.8f})")

    scale = torch.tensor(fp8_max / x.abs().amax().item(), dtype=torch.float32, device=DEVICE)
    amax_buf = torch.zeros((), dtype=torch.float32, device=DEVICE)
    out_amax, si_amax = quantize_fused_with_amax_out(x, scale, amax_buf)
    expected_amax = x.float().abs().amax()
    amax_match = torch.allclose(amax_buf, expected_amax, rtol=1e-5, atol=1e-8)
    print(f"  Amax_out match:  {amax_match} (got={amax_buf.item():.6f}, expected={expected_amax.item():.6f})")

    print(f"\n  ~114 quantize calls/iter in Flux 12B (57 layers x fwd+bwd)")
    print(f"  To close 65ms gap, need ~570us savings per call")


if __name__ == "__main__":
    main()
