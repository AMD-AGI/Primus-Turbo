"""Breakdown of forward pass time."""

import torch
import torch.utils.benchmark as benchmark

from primus_turbo.pytorch.core.low_precision import (
    Float8QuantConfig,
    Format,
    ScalingGranularity,
)
from primus_turbo.pytorch.kernels.gemm.gemm_fp8_impl import BackendType, gemm_fp8_impl
from primus_turbo.pytorch.ops.quantization import quantize_fp8
from primus_turbo.triton.quantization.quantization_tensorwise import fast_amax


def main():
    M, N, K = 8192, 8192, 8192
    dtype = torch.bfloat16
    device = "cuda"

    config = Float8QuantConfig(format=Format.E4M3, granularity=ScalingGranularity.TENSORWISE)
    a = torch.randn((M, K), dtype=dtype, device=device)
    b = torch.randn((N, K), dtype=dtype, device=device)  # trans_b shape

    fp8_dtype = torch.float8_e4m3fn
    fp8_max = torch.finfo(fp8_dtype).max

    print("=" * 60)
    print(f"Forward pass breakdown for M=N=K={M}")
    print("=" * 60)

    # 1. Fast amax (2x for a and b)
    def step_amax():
        amax_a = fast_amax(a)
        amax_b = fast_amax(b)
        return amax_a, amax_b

    # 2. Scale computation
    amax_a, amax_b = step_amax()

    def step_scale():
        scale_a = fp8_max / torch.clamp(amax_a, min=1e-12)
        scale_b = fp8_max / torch.clamp(amax_b, min=1e-12)
        return scale_a, scale_b

    # 3. Quantize with transpose (our fused kernel)
    def step_quant_with_trans():
        a_fp8, a_scale, a_t, _ = quantize_fp8(a, fp8_dtype, config.granularity, with_trans=True)
        b_fp8, b_scale, b_t, _ = quantize_fp8(b, fp8_dtype, config.granularity, with_trans=True)
        return a_fp8, a_scale, b_fp8, b_scale

    # 4. GEMM only
    a_fp8, a_scale, b_fp8, b_scale = step_quant_with_trans()

    def step_gemm():
        return gemm_fp8_impl(
            a_fp8,
            a_scale,
            False,
            b_fp8,
            b_scale,
            True,
            dtype,
            False,
            granularity=config.granularity.value,
            default_backend=BackendType.HIPBLASLT.value,
        )

    # 5. Full forward (quantize with trans + GEMM)
    def full_forward():
        a_fp8, a_scale, a_t, _ = quantize_fp8(a, fp8_dtype, config.granularity, with_trans=True)
        b_fp8, b_scale, b_t, _ = quantize_fp8(b, fp8_dtype, config.granularity, with_trans=True)
        return gemm_fp8_impl(
            a_fp8,
            a_scale,
            False,
            b_fp8,
            b_scale,
            True,
            dtype,
            False,
            granularity=config.granularity.value,
            default_backend=BackendType.HIPBLASLT.value,
        )

    # Warmup
    for _ in range(20):
        step_amax()
        step_scale()
        step_quant_with_trans()
        step_gemm()
        full_forward()
    torch.cuda.synchronize()

    # Benchmark
    t_amax = benchmark.Timer(stmt="fn()", globals={"fn": step_amax}).timeit(100)
    t_scale = benchmark.Timer(stmt="fn()", globals={"fn": step_scale}).timeit(100)
    t_quant = benchmark.Timer(stmt="fn()", globals={"fn": step_quant_with_trans}).timeit(100)
    t_gemm = benchmark.Timer(stmt="fn()", globals={"fn": step_gemm}).timeit(100)
    t_full = benchmark.Timer(stmt="fn()", globals={"fn": full_forward}).timeit(100)

    print(f"\nBreakdown:")
    print(f"  1. Fast amax (2x):           {t_amax.mean * 1e3:.4f} ms")
    print(f"  2. Scale computation:        {t_scale.mean * 1e3:.4f} ms")
    print(f"  3. Quant+trans (2x):         {t_quant.mean * 1e3:.4f} ms")
    print(f"  4. GEMM only:                {t_gemm.mean * 1e3:.4f} ms")
    print(f"  -----------------------------------")
    print(
        f"  Sum (1+2+3+4):               {(t_amax.mean + t_scale.mean + t_quant.mean + t_gemm.mean) * 1e3:.4f} ms"
    )
    print(f"  Full forward:                {t_full.mean * 1e3:.4f} ms")
    print(f"  -----------------------------------")
    print(f"  TE forward:                  0.447 ms (reference)")
    print(f"  Gap:                         {t_full.mean * 1e3 - 0.447:.4f} ms")


if __name__ == "__main__":
    main()
