"""Detailed breakdown of forward pass performance."""

import torch
import torch.utils.benchmark as benchmark

import primus_turbo.pytorch as turbo
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
    fp8_dtype = torch.float8_e4m3fn
    granularity = ScalingGranularity.TENSORWISE

    print("=" * 70)
    print(f"Detailed Forward Pass Breakdown - M=N=K={M}")
    print("=" * 70)

    a = torch.randn((M, K), dtype=dtype, device=device)
    b = torch.randn((N, K), dtype=dtype, device=device)  # trans_b=True shape

    # Warmup
    for _ in range(5):
        a_fp8, a_scale_inv, a_t_fp8, a_t_scale_inv = quantize_fp8(a, fp8_dtype, granularity, with_trans=True)
        b_fp8, b_scale_inv, b_t_fp8, b_t_scale_inv = quantize_fp8(b, fp8_dtype, granularity, with_trans=True)

    torch.cuda.synchronize()

    # 1. Quantize a (with transpose)
    a_quant_timer = benchmark.Timer(
        stmt="quantize_fp8(a, fp8_dtype, granularity, with_trans=True)",
        globals={"quantize_fp8": quantize_fp8, "a": a, "fp8_dtype": fp8_dtype, "granularity": granularity},
    )
    a_quant_time = a_quant_timer.timeit(100).mean * 1e3
    print(f"1. Quantize a (with trans): {a_quant_time:.4f} ms")

    # 2. Quantize b (with transpose)
    b_quant_timer = benchmark.Timer(
        stmt="quantize_fp8(b, fp8_dtype, granularity, with_trans=True)",
        globals={"quantize_fp8": quantize_fp8, "b": b, "fp8_dtype": fp8_dtype, "granularity": granularity},
    )
    b_quant_time = b_quant_timer.timeit(100).mean * 1e3
    print(f"2. Quantize b (with trans): {b_quant_time:.4f} ms")

    # Pre-quantize for GEMM benchmark
    a_fp8, a_scale_inv, a_t_fp8, a_t_scale_inv = quantize_fp8(a, fp8_dtype, granularity, with_trans=True)
    b_fp8, b_scale_inv, b_t_fp8, b_t_scale_inv = quantize_fp8(b, fp8_dtype, granularity, with_trans=True)

    # 3. GEMM (NT layout)
    def gemm_func():
        return gemm_fp8_impl(
            a_fp8,
            a_scale_inv,
            False,
            b_fp8,
            b_scale_inv,
            True,  # trans_b=True
            dtype,
            False,
            granularity=granularity.value,
            default_backend=BackendType.HIPBLASLT.value,
        )

    gemm_timer = benchmark.Timer(stmt="fn()", globals={"fn": gemm_func})
    gemm_time = gemm_timer.timeit(100).mean * 1e3
    print(f"3. GEMM (NT layout):        {gemm_time:.4f} ms")

    # Total estimate
    total_estimate = a_quant_time + b_quant_time + gemm_time
    print(f"\n--- Sum (1+2+3): {total_estimate:.4f} ms ---")

    # Full forward pass
    config = Float8QuantConfig(format=Format.E4M3, granularity=ScalingGranularity.TENSORWISE)
    a_grad = torch.randn((M, K), dtype=dtype, device=device, requires_grad=True)
    b_grad = torch.randn((N, K), dtype=dtype, device=device, requires_grad=True)

    full_fwd_timer = benchmark.Timer(
        stmt="turbo.ops.gemm_fp8(a, b, trans_b=True, config=config)",
        globals={"turbo": turbo, "a": a_grad, "b": b_grad, "config": config},
    )
    full_fwd_time = full_fwd_timer.timeit(100).mean * 1e3
    print(f"Full forward:   {full_fwd_time:.4f} ms")

    # Overhead
    overhead = full_fwd_time - total_estimate
    print(f"Overhead:       {overhead:.4f} ms")

    print("\n" + "=" * 70)
    print("Comparison with TE (reference):")
    print("  TE Forward:   ~0.456 ms")
    print(f"  Our Forward:  {full_fwd_time:.3f} ms")
    print(f"  Gap:          {full_fwd_time - 0.456:.3f} ms")
    print("=" * 70)


if __name__ == "__main__":
    main()
