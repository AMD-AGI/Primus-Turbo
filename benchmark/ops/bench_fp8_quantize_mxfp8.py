###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Benchmark FP8 quantization kernels (MXFP8, Tensorwise, Blockwise-128).

Compares:
  1. MXFP8 single-direction:  quantize_mxfp8(axis=1) -- rowwise
  2. MXFP8 single-direction:  quantize_mxfp8(axis=0) -- colwise
  3. MXFP8 dual (fwd+trans):  quantize_mxfp8_dual   -- both directions at once
  4. FP8 blockwise (128):     quant_fp8_blockwise_impl(block_size=128)
  5. Tensorwise:              quantize_fp8_tensorwise  -- scalar scale
  6. Tensorwise (fused):      quantize_fp8_tensorwise_fused -- fused amax+scale path

Shapes are derived from Flux 12B GEMM dimensions (activations and weights).

Requires MI350/MI355X (GFX950) -- MXFP8 quantization is GFX950-only.
Tensorwise works on gfx942+ but is run here for apples-to-apples comparison.

Usage:
    cd benchmark/ops
    python bench_fp8_quantize_mxfp8.py [--warmup 20] [--repeat 200]
"""

import argparse

import torch

import primus_turbo.pytorch as turbo
from primus_turbo.pytorch.core.low_precision import MXFP8_PADDING_ALIGN_SIZE

DEVICE = "cuda:0"
DTYPE = torch.bfloat16
FP8_DTYPE = turbo.float8_e4m3
PT_OPS = torch.ops.primus_turbo_cpp_extension

# The fused amax+scale tensorwise op ships with the fused-amax-scale feature; the
# benchmark still runs (and compares the other paths) when it is not available.
HAS_TW_FUSED = hasattr(PT_OPS, "quantize_fp8_tensorwise_fused")

FLUX_12B_QUANT_SHAPES = [
    # Activation shapes (M, K) from forward pass
    (16384, 3072),
    (16384, 12288),
    (32768, 3072),
    (32768, 12288),
    # Weight shapes (N, K) -- quantized along K (axis=1)
    (9216, 3072),
    (3072, 3072),
    (12288, 3072),
    (3072, 12288),
    # Gradient shapes from backward (same dims as activations)
    (16384, 9216),
    (32768, 9216),
    # Wgrad activation transpose shapes
    (3072, 16384),
    (3072, 32768),
    (12288, 16384),
    (12288, 32768),
]


def quant_mxfp8_rowwise(x):
    """MXFP8 quantize along K (axis=1), block-of-32, E8M0 scales."""
    return PT_OPS.quantize_mxfp8(x, FP8_DTYPE, 1, MXFP8_PADDING_ALIGN_SIZE, False, False, False)


def quant_mxfp8_colwise(x):
    """MXFP8 quantize along M (axis=0), block-of-32, E8M0 scales."""
    return PT_OPS.quantize_mxfp8(x, FP8_DTYPE, 0, MXFP8_PADDING_ALIGN_SIZE, False, False, False)


def quant_mxfp8_dual(x):
    """MXFP8 dual quantize (both row and col), block-of-32, E8M0 scales."""
    return PT_OPS.quantize_mxfp8_dual(
        x, FP8_DTYPE, MXFP8_PADDING_ALIGN_SIZE, False, False, False, False, False, False
    )


def quant_blockwise_128(x):
    """FP8 blockwise quantize, block_size=128, FP32 scales (for comparison)."""
    from primus_turbo.pytorch.kernels.quantization.quantization_impl import (
        quant_fp8_blockwise_impl,
    )

    return quant_fp8_blockwise_impl(x, FP8_DTYPE, axis=1, block_size=128)


def quant_tensorwise(x):
    """FP8 tensorwise quantize -- single scalar scale per tensor."""
    return PT_OPS.quantize_fp8_tensorwise(x, FP8_DTYPE, None)


def quant_tensorwise_fused(x):
    """FP8 tensorwise quantize (fused amax+scale, 2-kernel path)."""
    return PT_OPS.quantize_fp8_tensorwise_fused(x, FP8_DTYPE, None)


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

    times_us = sorted(s.elapsed_time(e) * 1000 for s, e in zip(starts, ends))
    return times_us[len(times_us) // 2], times_us[0]


def compute_bandwidth_gbps(rows, cols, time_us):
    """Compute effective bandwidth: read BF16 input + write FP8 output + write scales."""
    input_bytes = rows * cols * 2  # BF16
    output_bytes = rows * cols * 1  # FP8
    scale_bytes = rows * ((cols + 31) // 32) * 1  # E8M0 (uint8), one per 32 elements
    total_bytes = input_bytes + output_bytes + scale_bytes
    return total_bytes / (time_us * 1e-6) / 1e9


def main():
    parser = argparse.ArgumentParser(description="MXFP8 Quantization Benchmark")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeat", type=int, default=200)
    args = parser.parse_args()

    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Warmup: {args.warmup}, Repeat: {args.repeat}")

    print("\n=== FP8 Quantization: MXFP8 vs Tensorwise vs Blockwise-128 ===")
    print(
        f"\n{'Shape':>14s}  {'Numel':>10s}  "
        f"{'MX row':>9s}  {'MX col':>9s}  {'MX dual':>9s}  {'BW-128':>9s}  "
        f"{'TW':>9s}  {'TW-fused':>9s}  "
        f"{'Dual/Row':>9s}  {'BW GB/s':>9s}"
    )
    print(
        f"{'':>14s}  {'':>10s}  "
        f"{'us':>9s}  {'us':>9s}  {'us':>9s}  {'us':>9s}  "
        f"{'us':>9s}  {'us':>9s}  "
        f"{'ratio':>9s}  {'(row)':>9s}"
    )
    print("-" * 127)

    for rows, cols in FLUX_12B_QUANT_SHAPES:
        x = torch.randn(rows, cols, dtype=DTYPE, device=DEVICE)
        numel = x.numel()

        row_med, row_min = benchmark_fn(quant_mxfp8_rowwise, args.warmup, args.repeat, x)
        col_med, col_min = benchmark_fn(quant_mxfp8_colwise, args.warmup, args.repeat, x)
        dual_med, dual_min = benchmark_fn(quant_mxfp8_dual, args.warmup, args.repeat, x)
        bw128_med, bw128_min = benchmark_fn(quant_blockwise_128, args.warmup, args.repeat, x)
        tw_med, tw_min = benchmark_fn(quant_tensorwise, args.warmup, args.repeat, x)
        if HAS_TW_FUSED:
            tw_fused_med, tw_fused_min = benchmark_fn(quant_tensorwise_fused, args.warmup, args.repeat, x)
            tw_fused_str = f"{tw_fused_med:>9.1f}"
        else:
            tw_fused_str = f"{'N/A':>9s}"

        dual_vs_row = dual_med / row_med if row_med > 0 else float("inf")
        bw_gbps = compute_bandwidth_gbps(rows, cols, row_med)

        shape_str = f"{rows}x{cols}"
        print(
            f"{shape_str:>14s}  {numel:>10,d}  "
            f"{row_med:>9.1f}  {col_med:>9.1f}  {dual_med:>9.1f}  {bw128_med:>9.1f}  "
            f"{tw_med:>9.1f}  {tw_fused_str}  "
            f"{dual_vs_row:>8.2f}x  {bw_gbps:>9.1f}"
        )

    print("\nNotes:")
    print("  MX row/col: quantize_mxfp8 with axis=1/0 (block-of-32, E8M0 scales)")
    print("  MX dual:    quantize_mxfp8_dual (both directions in one kernel launch)")
    print("  BW-128:     quant_fp8_blockwise_impl with block_size=128 (FP32 scales)")
    print("  TW:         quantize_fp8_tensorwise (3-kernel: amax, scale, cast)")
    print("  TW-fused:   quantize_fp8_tensorwise_fused (2-kernel: fused amax+scale, cast)")
    if not HAS_TW_FUSED:
        print("  TW-fused shown as N/A: quantize_fp8_tensorwise_fused not available in this build")
    print("  Dual/Row:   ratio shows overhead of dual vs single-direction (ideally < 2.0x)")
    print("  BW GB/s:    effective bandwidth of rowwise MXFP8 quantization")


if __name__ == "__main__":
    main()
