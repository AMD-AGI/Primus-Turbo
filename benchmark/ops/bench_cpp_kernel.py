"""Benchmark C++ fused cast+transpose kernel vs Triton vs TE."""

import torch
import torch.utils.benchmark as benchmark

import primus_turbo.pytorch as turbo
from primus_turbo.pytorch.core.low_precision import (
    Float8QuantConfig,
    Format,
    ScalingGranularity,
)


def main():
    M, N, K = 8192, 8192, 8192
    dtype = torch.bfloat16
    device = "cuda"
    trans_b = True

    config = Float8QuantConfig(
        format=Format.E4M3,
        granularity=ScalingGranularity.TENSORWISE,
    )

    print("=" * 70)
    print(f"C++ Fused Cast+Transpose Kernel Benchmark")
    print(f"M={M}, N={N}, K={K}")
    print("=" * 70)

    a = torch.randn((M, K), dtype=dtype, device=device, requires_grad=True)
    b_shape = (N, K) if trans_b else (K, N)
    b = torch.randn(b_shape, dtype=dtype, device=device, requires_grad=True)

    # Warmup
    for _ in range(5):
        out = turbo.ops.gemm_fp8(a, b, trans_b=trans_b, config=config)

    grad_out = torch.randn_like(out)

    # Forward
    def fwd_func():
        return turbo.ops.gemm_fp8(a, b, trans_b=trans_b, config=config)

    # Forward + Backward
    def fwd_bwd_func():
        out = turbo.ops.gemm_fp8(a, b, trans_b=trans_b, config=config)
        out.backward(grad_out)

    # Warmup
    for _ in range(20):
        fwd_bwd_func()
    torch.cuda.synchronize()

    # Benchmark
    fwd_timer = benchmark.Timer(stmt="fn()", globals={"fn": fwd_func})
    fwd_measurement = fwd_timer.timeit(100)

    fwd_bwd_timer = benchmark.Timer(stmt="fn()", globals={"fn": fwd_bwd_func})
    fwd_bwd_measurement = fwd_bwd_timer.timeit(100)

    fwd_total_flops = 2 * M * N * K
    bwd_total_flops = 2 * fwd_total_flops  # d_a + d_b

    fwd_mean_time_ms = fwd_measurement.mean * 1e3
    fwd_bwd_mean_time_ms = fwd_bwd_measurement.mean * 1e3
    bwd_mean_time_ms = fwd_bwd_mean_time_ms - fwd_mean_time_ms

    fwd_tflops = fwd_total_flops / (fwd_mean_time_ms * 1e-3) / 1e12
    bwd_tflops = bwd_total_flops / (bwd_mean_time_ms * 1e-3) / 1e12

    print(f"\nPrimus-Turbo (C++ kernel):")
    print(f"  Forward:  {fwd_mean_time_ms:.3f} ms | {fwd_tflops:.0f} TFLOPS")
    print(f"  Backward: {bwd_mean_time_ms:.3f} ms | {bwd_tflops:.0f} TFLOPS")

    # Compare with TE if available
    try:
        import transformer_engine as te
        from transformer_engine.common.recipe import Float8CurrentScaling
        from transformer_engine.common.recipe import Format as TEFormat

        fp8_recipe = Float8CurrentScaling(fp8_format=TEFormat.E4M3)

        x = torch.randn((M, K), dtype=dtype, device=device, requires_grad=True)
        layer = te.pytorch.Linear(K, N, bias=False, params_dtype=dtype).to(device)

        with te.pytorch.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
            out_te = layer(x)
        grad_out_te = torch.randn_like(out_te)

        def te_fwd_func():
            with te.pytorch.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
                return layer(x)

        def te_fwd_bwd_func():
            with te.pytorch.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
                out = layer(x)
            out.backward(grad_out_te)

        for _ in range(20):
            te_fwd_bwd_func()
        torch.cuda.synchronize()

        te_fwd_timer = benchmark.Timer(stmt="fn()", globals={"fn": te_fwd_func})
        te_fwd_measurement = te_fwd_timer.timeit(100)

        te_fwd_bwd_timer = benchmark.Timer(stmt="fn()", globals={"fn": te_fwd_bwd_func})
        te_fwd_bwd_measurement = te_fwd_bwd_timer.timeit(100)

        te_fwd_mean_time_ms = te_fwd_measurement.mean * 1e3
        te_fwd_bwd_mean_time_ms = te_fwd_bwd_measurement.mean * 1e3
        te_bwd_mean_time_ms = te_fwd_bwd_mean_time_ms - te_fwd_mean_time_ms

        te_fwd_tflops = fwd_total_flops / (te_fwd_mean_time_ms * 1e-3) / 1e12
        te_bwd_tflops = bwd_total_flops / (te_bwd_mean_time_ms * 1e-3) / 1e12

        print(f"\nTransformerEngine:")
        print(f"  Forward:  {te_fwd_mean_time_ms:.3f} ms | {te_fwd_tflops:.0f} TFLOPS")
        print(f"  Backward: {te_bwd_mean_time_ms:.3f} ms | {te_bwd_tflops:.0f} TFLOPS")

        print(f"\nSpeedup (Primus vs TE):")
        print(f"  Forward:  {te_fwd_mean_time_ms / fwd_mean_time_ms:.2f}x")
        print(f"  Backward: {te_bwd_mean_time_ms / bwd_mean_time_ms:.2f}x")

    except ImportError:
        print("\nTransformerEngine not available for comparison")

    print("=" * 70)


if __name__ == "__main__":
    main()
