"""Benchmark hipBLASLt FP8 GEMM for different layouts (TN, NN, NT)."""

import torch
import torch.utils.benchmark as benchmark

from primus_turbo.pytorch.core.low_precision import ScalingGranularity
from primus_turbo.pytorch.kernels.gemm.gemm_fp8_impl import BackendType, gemm_fp8_impl
from primus_turbo.pytorch.ops.quantization import quantize_fp8


def main():
    M, N, K = 8192, 8192, 8192
    dtype = torch.bfloat16
    device = "cuda"
    fp8_dtype = torch.float8_e4m3fn
    granularity = ScalingGranularity.TENSORWISE

    print("=" * 70)
    print(f"hipBLASLt FP8 GEMM Benchmark - M=N=K={M}")
    print("=" * 70)

    # Pre-quantize all tensors (exclude quantization time)
    a = torch.randn((M, K), dtype=dtype, device=device)
    b_nn = torch.randn((K, N), dtype=dtype, device=device)  # NN: (K, N)
    b_nt = torch.randn((N, K), dtype=dtype, device=device)  # NT: (N, K), trans_b=True

    # Quantize
    a_fp8, a_scale = quantize_fp8(a, fp8_dtype, granularity)
    a_t_fp8, a_t_scale = quantize_fp8(a.t().contiguous(), fp8_dtype, granularity)  # For TN
    b_nn_fp8, b_nn_scale = quantize_fp8(b_nn, fp8_dtype, granularity)
    b_nt_fp8, b_nt_scale = quantize_fp8(b_nt, fp8_dtype, granularity)
    b_t_fp8, b_t_scale = quantize_fp8(b_nt.t().contiguous(), fp8_dtype, granularity)  # Transposed NT for TN

    fwd_flops = 2 * M * N * K

    # Layout: NT (A @ B^T where B is (N, K))
    # trans_a=False, trans_b=True
    def gemm_nt():
        return gemm_fp8_impl(
            a_fp8,
            a_scale,
            False,
            b_nt_fp8,
            b_nt_scale,
            True,  # trans_b=True
            dtype,
            False,
            granularity=granularity.value,
            default_backend=BackendType.HIPBLASLT.value,
        )

    # Layout: NN (A @ B where B is (K, N))
    # trans_a=False, trans_b=False
    def gemm_nn():
        return gemm_fp8_impl(
            a_fp8,
            a_scale,
            False,
            b_nn_fp8,
            b_nn_scale,
            False,  # trans_b=False
            dtype,
            False,
            granularity=granularity.value,
            default_backend=BackendType.HIPBLASLT.value,
        )

    # Layout: TN (A^T @ B where A is (K, M), B is (K, N))
    # trans_a=True, trans_b=False
    # Note: For FP8 GEMM, trans_a=True may not be directly supported
    # We use A^T stored as (M, K) with trans_a=True
    def gemm_tn():
        return gemm_fp8_impl(
            a_t_fp8,
            a_t_scale,
            True,  # trans_a=True, A^T is (K, M)
            b_nn_fp8,
            b_nn_scale,
            False,  # trans_b=False
            dtype,
            False,
            granularity=granularity.value,
            default_backend=BackendType.HIPBLASLT.value,
        )

    # Warmup and test
    print("\nWarming up...")
    for _ in range(20):
        gemm_nt()
        gemm_nn()
        try:
            gemm_tn()
        except:
            pass
    torch.cuda.synchronize()

    # Benchmark NT
    print("\nBenchmarking...")
    t_nt = benchmark.Timer(stmt="fn()", globals={"fn": gemm_nt}).timeit(100)
    tflops_nt = fwd_flops / t_nt.mean / 1e12
    print(f"  NT (trans_a=F, trans_b=T): {t_nt.mean * 1e3:.3f} ms | {tflops_nt:.2f} TFLOPS")

    # Benchmark NN
    t_nn = benchmark.Timer(stmt="fn()", globals={"fn": gemm_nn}).timeit(100)
    tflops_nn = fwd_flops / t_nn.mean / 1e12
    print(f"  NN (trans_a=F, trans_b=F): {t_nn.mean * 1e3:.3f} ms | {tflops_nn:.2f} TFLOPS")

    # Benchmark TN (may fail)
    try:
        t_tn = benchmark.Timer(stmt="fn()", globals={"fn": gemm_tn}).timeit(100)
        tflops_tn = fwd_flops / t_tn.mean / 1e12
        print(f"  TN (trans_a=T, trans_b=F): {t_tn.mean * 1e3:.3f} ms | {tflops_tn:.2f} TFLOPS")
    except Exception as e:
        print(f"  TN (trans_a=T, trans_b=F): Not supported or error: {e}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
