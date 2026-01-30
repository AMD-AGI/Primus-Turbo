"""Benchmark quantization kernels only."""

import torch
import torch.utils.benchmark as benchmark

from primus_turbo.pytorch.core.low_precision import ScalingGranularity
from primus_turbo.pytorch.ops.quantization import quantize_fp8


def main():
    M, K = 8192, 8192
    dtype = torch.bfloat16
    device = "cuda"
    fp8_dtype = torch.float8_e4m3fn

    print("=" * 70)
    print(f"Quantization Kernel Benchmark - M={M}, K={K}")
    print("=" * 70)

    x = torch.randn((M, K), dtype=dtype, device=device)

    # Warmup
    for _ in range(5):
        x_fp8, x_scale_inv = quantize_fp8(x, fp8_dtype, ScalingGranularity.TENSORWISE, with_trans=False)
        x_fp8, x_t_fp8, x_scale_inv = quantize_fp8(
            x, fp8_dtype, ScalingGranularity.TENSORWISE, with_trans=True
        )

    torch.cuda.synchronize()

    # Benchmark without transpose (original C++ kernel)
    quant_timer = benchmark.Timer(
        stmt="quantize_fp8(x, fp8_dtype, granularity, with_trans=False)",
        globals={
            "quantize_fp8": quantize_fp8,
            "x": x,
            "fp8_dtype": fp8_dtype,
            "granularity": ScalingGranularity.TENSORWISE,
        },
    )
    quant_measurement = quant_timer.timeit(100)
    print(f"Quantize only (no trans):    {quant_measurement.mean * 1e3:.4f} ms")

    # Benchmark with transpose (new fused C++ kernel)
    quant_trans_timer = benchmark.Timer(
        stmt="quantize_fp8(x, fp8_dtype, granularity, with_trans=True)",
        globals={
            "quantize_fp8": quantize_fp8,
            "x": x,
            "fp8_dtype": fp8_dtype,
            "granularity": ScalingGranularity.TENSORWISE,
        },
    )
    quant_trans_measurement = quant_trans_timer.timeit(100)
    print(f"Quantize + transpose (fused): {quant_trans_measurement.mean * 1e3:.4f} ms")

    # Baseline: quant + explicit transpose
    def quant_then_transpose():
        x_fp8, x_scale_inv = quantize_fp8(x, fp8_dtype, ScalingGranularity.TENSORWISE, with_trans=False)
        x_t_fp8 = x_fp8.t().contiguous()
        return x_fp8, x_t_fp8, x_scale_inv

    baseline_timer = benchmark.Timer(stmt="fn()", globals={"fn": quant_then_transpose})
    baseline_measurement = baseline_timer.timeit(100)
    print(f"Quant + separate transpose:   {baseline_measurement.mean * 1e3:.4f} ms")

    # PyTorch amax baseline
    amax_timer = benchmark.Timer(stmt="torch.abs(x).amax()", globals={"torch": torch, "x": x})
    amax_measurement = amax_timer.timeit(100)
    print(f"PyTorch abs().amax():         {amax_measurement.mean * 1e3:.4f} ms")

    print("=" * 70)


if __name__ == "__main__":
    main()
