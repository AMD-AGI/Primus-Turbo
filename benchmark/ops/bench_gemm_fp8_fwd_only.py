###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Test FP8 GEMM forward only (no backward) to compare kernel performance."""

import torch
import torch.utils.benchmark as benchmark

from primus_turbo.pytorch.core.low_precision import (
    Float8QuantConfig,
    Format,
    ScalingGranularity,
)
from primus_turbo.pytorch.kernels.gemm.gemm_fp8_impl import BackendType, gemm_fp8_impl
from primus_turbo.pytorch.ops.quantization import quantize_fp8


def main():
    M, N, K = 8192, 8192, 8192
    dtype = torch.bfloat16
    device = "cuda"

    config = Float8QuantConfig(
        format=Format.E4M3,
        granularity=ScalingGranularity.TENSORWISE,
    )

    a = torch.randn((M, K), dtype=dtype, device=device)
    b = torch.randn((N, K), dtype=dtype, device=device)  # trans_b=True shape

    # Pre-quantize (without transpose)
    a_dtype = torch.float8_e4m3fn
    b_dtype = torch.float8_e4m3fn

    print("Testing quantization kernels...")

    # Test 1: Original quantize (no transpose)
    def quant_no_trans():
        a_fp8, a_scale = quantize_fp8(a, a_dtype, config.granularity, with_trans=False)
        b_fp8, b_scale = quantize_fp8(b, b_dtype, config.granularity, with_trans=False)
        return a_fp8, a_scale, b_fp8, b_scale

    # Test 2: Quantize with transpose
    def quant_with_trans():
        a_fp8, a_scale, a_t, a_t_scale = quantize_fp8(a, a_dtype, config.granularity, with_trans=True)
        b_fp8, b_scale, b_t, b_t_scale = quantize_fp8(b, b_dtype, config.granularity, with_trans=True)
        return a_fp8, a_scale, b_fp8, b_scale

    # Warmup
    for _ in range(10):
        quant_no_trans()
        quant_with_trans()
    torch.cuda.synchronize()

    # Benchmark quantization
    timer1 = benchmark.Timer(stmt="fn()", globals={"fn": quant_no_trans})
    timer2 = benchmark.Timer(stmt="fn()", globals={"fn": quant_with_trans})

    m1 = timer1.timeit(50)
    m2 = timer2.timeit(50)

    print(f"Quantize (no trans):   {m1.mean * 1e3:.3f} ms")
    print(f"Quantize (with trans): {m2.mean * 1e3:.3f} ms")
    print(f"Overhead: {(m2.mean - m1.mean) * 1e3:.3f} ms ({(m2.mean / m1.mean - 1) * 100:.1f}%)")

    # Test GEMM kernel alone
    print("\nTesting GEMM kernel...")
    a_fp8, a_scale = quantize_fp8(a, a_dtype, config.granularity, with_trans=False)
    b_fp8, b_scale = quantize_fp8(b, b_dtype, config.granularity, with_trans=False)

    def gemm_only():
        return gemm_fp8_impl(
            a_fp8,
            a_scale,
            False,
            b_fp8,
            b_scale,
            True,  # trans_b=True
            dtype,
            False,
            granularity=config.granularity.value,
            default_backend=BackendType.HIPBLASLT.value,
        )

    for _ in range(10):
        gemm_only()
    torch.cuda.synchronize()

    timer3 = benchmark.Timer(stmt="fn()", globals={"fn": gemm_only})
    m3 = timer3.timeit(50)

    fwd_flops = 2 * M * N * K
    tflops = fwd_flops / (m3.mean) / 1e12
    print(f"GEMM kernel only: {m3.mean * 1e3:.3f} ms | {tflops:.2f} TFLOPS")


if __name__ == "__main__":
    main()
