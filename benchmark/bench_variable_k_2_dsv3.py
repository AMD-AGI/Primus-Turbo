#!/usr/bin/env python3
"""
Benchmark for ck_grouped_gemm_variable_k_2 (ColMajor output) for DeepSeek V3.

Measures performance of variable_k_2 for grad_b computation in MoE backward pass.
"""

import torch
import torch.utils.benchmark as benchmark

import primus_turbo.pytorch._C  # Import to register ops


def generate_group_lens(B: int, M: int) -> torch.Tensor:
    """Generate balanced group lengths."""
    base_size = M // B
    remainder = M % B
    group_lens = torch.full((B,), base_size, dtype=torch.int64)
    if remainder > 0:
        group_lens[:remainder] += 1
    return group_lens


def benchmark_variable_k(B: int, M: int, N: int, K: int, dtype: torch.dtype, use_v2: bool = False):
    """
    Benchmark variable_k or variable_k_2 for grad_b = x^T @ grad_y.

    Args:
        B: Batch size (number of groups)
        M: M dimension (token count)
        N: N dimension (output features)
        K: K dimension (hidden size)
        dtype: Data type (torch.bfloat16 or torch.float16)
        use_v2: Use variable_k_2 (ColMajor output) if True, else variable_k (RowMajor)

    Returns:
        (time_ms, tflops)
    """
    device = "cuda"

    # Prepare tensors for grad_b computation
    # grad_b = x^T @ grad_y, where x=[M,K], grad_y=[B,M,N]
    x = torch.randn((B * M, N), dtype=dtype, device=device)
    grad_y = torch.randn((B * M, K), dtype=dtype, device=device)
    group_lens = generate_group_lens(B, B * M).to(device)
    group_offs = torch.zeros(B + 1, dtype=torch.int64, device=device)
    group_offs[1:] = torch.cumsum(group_lens, dim=0)
    # print("x.shape: ", x.shape)
    # print("grad_y.shape: ", grad_y.shape)
    # print("group_offs: ", group_offs)
    if use_v2:
        # variable_k_2: ColMajor output [B, N, K]
        func = lambda: torch.ops.primus_turbo_cpp_extension.ck_grouped_gemm_variable_k_2(
            x, grad_y, group_lens, group_offs, True, False, None
        )
    else:
        # variable_k: RowMajor output [B, K, N]
        func = lambda: torch.ops.primus_turbo_cpp_extension.ck_grouped_gemm_variable_k(
            grad_y, x, group_lens, group_offs, True, False, None
        )

    # Warmup
    for _ in range(20):
        _ = func()
    torch.cuda.synchronize()

    # Benchmark
    timer = benchmark.Timer(stmt="fn()", globals={"fn": func})
    result = timer.timeit(100)
    time_ms = result.mean * 1000

    # Calculate TFLOPS
    # For grouped GEMM variable_k: x^T @ grad_y where x=[M,K], grad_y=[B,M,N]
    # Total tokens M are split into B groups, so total FLOPS = 2 * K * N * M
    total_flops = 2.0 * K * N * B * M
    tflops = total_flops / (time_ms * 1e-3) / 1e12

    return time_ms, tflops


def main():
    print("=" * 110)
    print("Benchmark: M=N=K=8192 cases for ck_grouped_gemm_variable_k vs variable_k_2")
    print("=" * 110)

    # Test M=N=K=8192 with different B (group count)
    configs = [
        # (name, B, M, N, K)
        ("8192x8192x8192-B1", 1, 8192, 8192, 8192),
        ("8192x8192x8192-B2", 2, 8192, 8192, 8192),
        ("8192x8192x8192-B4", 4, 8192, 8192, 8192),
        ("8192x8192x8192-B8", 8, 8192, 8192, 8192),
        ("8192x8192x8192-B16", 16, 8192, 8192, 8192),
        ("8192x8192x8192-B32", 32, 8192, 8192, 8192),
    ]

    dtypes = [("bf16", torch.bfloat16), ("fp16", torch.float16)]

    print(
        f"\n{'Case':<25} {'dtype':<8} {'Shape (B,M,N,K)':<20} "
        f"{'v1 ms':>10} {'v1 TFLOP':>10} {'v2 ms':>10} {'v2 TFLOP':>10} {'Speedup':>10}"
    )
    print("-" * 110)

    for config_name, B, M, N, K in configs:
        for dtype_name, dtype in dtypes:
            try:
                # Benchmark variable_k (v1)
                time_v1, tflops_v1 = benchmark_variable_k(B, M, N, K, dtype, use_v2=False)

                # Benchmark variable_k_2 (v2)
                time_v2, tflops_v2 = benchmark_variable_k(B, M, N, K, dtype, use_v2=True)

                speedup = time_v1 / time_v2
                shape_str = f"{B},{M},{N},{K}"

                print(
                    f"{config_name:<25} {dtype_name:<8} {shape_str:<20} "
                    f"{time_v1:>10.3f} {tflops_v1:>10.2f} {time_v2:>10.3f} {tflops_v2:>10.2f} {speedup:>10.3f}x"
                )

            except Exception as e:
                shape_str = f"{B},{M},{N},{K}"
                print(f"{config_name:<25} {dtype_name:<8} {shape_str:<20} ERROR")
                print(f"  Error: {e}")
                import traceback

                traceback.print_exc()

    print("=" * 110)
    print("v1 = variable_k (RowMajor output [B,K,N])")
    print("v2 = variable_k_2 (ColMajor output [B,N,K])")
    print("=" * 110)


if __name__ == "__main__":
    main()
