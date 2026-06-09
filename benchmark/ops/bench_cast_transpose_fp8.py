###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Benchmark fused FP8 cast+transpose+amax vs separate quantize + .t().contiguous().

Compares:
  1. Unfused baseline: quantize_fp8_tensorwise + .t().contiguous()  -- 2 ops
  2. Triton fused: cast_transpose_fp8_triton                        -- 1 kernel

Usage:
    cd benchmark/ops
    python bench_cast_transpose_fp8.py [--warmup 30] [--repeat 300] [--gemm]
"""

import argparse

import torch

import primus_turbo.pytorch as turbo
from primus_turbo.pytorch.kernels.quantization.cast_transpose_fp8 import (
    cast_transpose_fp8_triton,
)

DEVICE = "cuda:0"
DTYPE = torch.bfloat16
FP8_DTYPE = turbo.float8_e4m3

SHAPES = [
    (4096, 3072),
    (4096, 12288),
    (12288, 4096),
    (16384, 3072),
    (8192, 8192),
    (32768, 3072),
    (3072, 12288),
]


def baseline_quantize_transpose(x, scale, amax_buf):
    """Unfused: C++ quantize_fp8_tensorwise + .t().contiguous()."""
    fp8_out, scale_inv = torch.ops.primus_turbo_cpp_extension.quantize_fp8_tensorwise(
        x, FP8_DTYPE, None
    )
    t_out = fp8_out.t().contiguous()
    amax_buf.fill_(x.float().abs().amax().item())
    return fp8_out, t_out, scale_inv


def triton_cast_transpose(x, scale, amax_buf):
    """Triton @triton_op fused: cast + transpose + amax in 1 kernel."""
    return cast_transpose_fp8_triton(x, FP8_DTYPE, scale, amax_out=amax_buf)


def benchmark_fn(fn, warmup, repeat, *args):
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


def run_performance_bench(args):
    fp8_max = torch.finfo(FP8_DTYPE).max

    print("\n=== Cast + Transpose + Amax: Unfused Baseline vs Triton Fused ===")
    print(
        f"\n{'Shape':>14s}  {'Numel':>10s}  "
        f"{'Unfused':>10s}  {'Triton':>10s}  {'Speedup':>8s}"
    )
    print(f"{'':>14s}  {'':>10s}  " f"{'us':>10s}  {'us':>10s}  {'':>8s}")
    print("-" * 70)

    for shape in SHAPES:
        x = torch.randn(*shape, dtype=DTYPE, device=DEVICE)
        numel = x.numel()
        scale = torch.tensor(fp8_max / x.abs().amax().item(), dtype=torch.float32, device=DEVICE)
        amax_buf = torch.zeros((), dtype=torch.float32, device=DEVICE)

        base_med, _ = benchmark_fn(baseline_quantize_transpose, args.warmup, args.repeat, x, scale, amax_buf)
        triton_med, _ = benchmark_fn(triton_cast_transpose, args.warmup, args.repeat, x, scale, amax_buf)

        speedup_tri = base_med / triton_med if triton_med > 0 else float("inf")

        shape_str = f"{shape[0]}x{shape[1]}"
        print(
            f"{shape_str:>14s}  {numel:>10,d}  "
            f"{base_med:>10.1f}  {triton_med:>10.1f}  {speedup_tri:>7.2f}x"
        )


def run_gemm_interaction_bench(args):
    """Test that fused kernel outputs don't degrade GEMM performance."""
    fp8_max = torch.finfo(FP8_DTYPE).max

    print("\n=== GEMM Interaction Test (quantize+transpose, then matmul) ===")
    print(
        f"\n{'Shape':>14s}  "
        f"{'Unfused':>10s}  {'Triton':>10s}  {'Delta':>10s}"
    )
    print(f"{'':>14s}  " f"{'us':>10s}  {'us':>10s}  {'us':>10s}")
    print("-" * 55)

    gemm_shapes = [
        (16384, 3072),
        (16384, 12288),
        (16384, 9216),
        (32768, 3072),
    ]

    for shape in gemm_shapes:
        M, N = shape
        x = torch.randn(M, N, dtype=DTYPE, device=DEVICE)
        scale = torch.tensor(fp8_max / x.abs().amax().item(), dtype=torch.float32, device=DEVICE)
        amax_buf = torch.zeros((), dtype=torch.float32, device=DEVICE)

        w = torch.randn(N, N, dtype=DTYPE, device=DEVICE).to(FP8_DTYPE).t()
        w_scale = torch.tensor(1.0, dtype=torch.float32, device=DEVICE)

        def bench_with_gemm(quant_fn, *qargs):
            def run():
                results = quant_fn(*qargs)
                fp8_out = results[0]
                _ = torch._scaled_mm(fp8_out, w, out_dtype=DTYPE, scale_a=scale, scale_b=w_scale)

            return run

        base_fn = bench_with_gemm(baseline_quantize_transpose, x, scale, amax_buf)
        triton_fn = bench_with_gemm(triton_cast_transpose, x, scale, amax_buf)

        base_med, _ = benchmark_fn(lambda: base_fn(), args.warmup, args.repeat)
        tri_med, _ = benchmark_fn(lambda: triton_fn(), args.warmup, args.repeat)

        shape_str = f"{M}x{N}"
        print(
            f"{shape_str:>14s}  "
            f"{base_med:>10.1f}  {tri_med:>10.1f}  "
            f"{tri_med - base_med:>+10.1f}"
        )


def run_correctness_check():
    fp8_max = torch.finfo(FP8_DTYPE).max

    print("\n=== Correctness Check (Triton fused vs unfused baseline) ===")
    for shape in SHAPES:
        x = torch.randn(*shape, dtype=DTYPE, device=DEVICE)
        scale = torch.tensor(fp8_max / x.abs().amax().item(), dtype=torch.float32, device=DEVICE)

        amax_buf_base = torch.zeros((), dtype=torch.float32, device=DEVICE)
        fp8_base, t_base, si_base = baseline_quantize_transpose(x, scale, amax_buf_base)

        amax_buf_triton = torch.zeros((), dtype=torch.float32, device=DEVICE)
        fp8_triton, t_triton, si_triton = triton_cast_transpose(x, scale, amax_buf_triton)

        expected_amax = x.float().abs().amax()

        tri_fp8 = torch.equal(fp8_base, fp8_triton)
        tri_t = torch.equal(t_base, t_triton)
        tri_si = torch.allclose(si_base, si_triton, rtol=1e-5)
        tri_amax = torch.allclose(amax_buf_triton, expected_amax, rtol=1e-3)

        tri_ok = tri_fp8 and tri_t and tri_si and tri_amax

        shape_str = f"{shape[0]}x{shape[1]}"
        print(f"  {shape_str:>14s}: {'PASS' if tri_ok else 'FAIL'}")
        if not tri_ok:
            print(f"    detail: fp8={tri_fp8} trans={tri_t} si={tri_si} amax={tri_amax}")
            if not tri_fp8:
                diff = (fp8_base.float() - fp8_triton.float()).abs()
                print(f"      fp8 diff: max={diff.max().item()}, count_ne={(diff > 0).sum().item()}")
            if not tri_amax:
                print(f"      amax: triton={amax_buf_triton.item():.6f} expected={expected_amax.item():.6f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--repeat", type=int, default=300)
    parser.add_argument("--gemm", action="store_true", help="Run GEMM interaction test")
    args = parser.parse_args()

    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Warmup: {args.warmup}, Repeat: {args.repeat}")

    run_correctness_check()
    run_performance_bench(args)
    if args.gemm:
        run_gemm_interaction_bench(args)


if __name__ == "__main__":
    main()
