#!/usr/bin/env python3
"""
Benchmark for grouped_gemm_fp8_variable_k_csrc_impl (CK backend)
专门测试 variable K 版本，主要用于 backward pass 中的 grad_b 计算
"""

import torch
import pandas as pd
from torch.utils.benchmark import Timer
import argparse

from primus_turbo.pytorch.core.low_precision import (
    Float8QuantConfig,
    Format,
    ScalingGranularity,
    float8_e4m3,
)
from primus_turbo.pytorch.kernels.grouped_gemm.grouped_gemm_fp8_impl import (
    grouped_gemm_fp8_variable_k_csrc_impl,
    grouped_gemm_compute_offs,
)
from primus_turbo.pytorch.ops.quantization import quantize_fp8


def generate_grouped_gemm_group_lens(B: int, M: int, balance: float = 1.0):
    """
    生成 group_lens
    balance=1.0: 均匀分布
    balance<1.0: 不均匀分布
    """
    if balance == 1.0:
        group_lens = torch.full((B,), M, dtype=torch.int64)
    else:
        # Generate random lengths with some imbalance
        group_lens = torch.randint(int(M * balance), M + 1, (B,), dtype=torch.int64)
        # Adjust to ensure sum equals B*M
        total = group_lens.sum().item()
        target = B * M
        if total != target:
            diff = target - total
            group_lens[0] += diff
    
    return group_lens


def benchmark_grouped_gemm_fp8_variable_k_csrc(
    B: int,  # Number of groups
    M: int,  # Rows per group (average) - becomes K in output
    N: int,  # Output N dimension
    K: int,  # Input K dimension (variable across groups due to M)
    granularity: ScalingGranularity = ScalingGranularity.TENSORWISE,
    balance: float = 1.0,
    warmup: int = 20,
    repeat: int = 100,
    device: str = "cuda",
):
    """
    Benchmark grouped_gemm_fp8_variable_k_csrc_impl
    
    这个函数执行: output[B, K_i, N] = A[K_i, sum(M)]^T @ B[sum(M), N]
    其中 K_i 是每个 group 的 K 维度（由 group_lens 决定）
    
    注意：只支持 trans_a=True, trans_b=False
    
    Args:
        B: Number of groups
        M: Average rows per group (这会成为 A 的列数，即 output 的 K 维度)
        N: Output columns (B 的列数)
        K: A 的行数 = B 的行数 (在 grouped 场景下是 total_M)
        granularity: Scaling granularity
        balance: Load balance factor
        warmup: Warmup iterations
        repeat: Benchmark iterations
        device: Device to run on
    """
    
    # Generate group lengths
    group_lens = generate_grouped_gemm_group_lens(B, M, balance).to(device)
    total_M = group_lens.sum().item()
    group_offs = grouped_gemm_compute_offs(group_lens)
    
    # 在 variable_k 版本中：
    # A 是 [total_M, K] (实际上是 grad_out 或 activations)
    # B 是 [total_M, N] (实际上是 activations 或 grad_out)
    # Output 是 [B, K, N] (每个 group 有不同的 K 维度由 group_lens 决定)
    
    # 注意：这里的 K 实际是 total_M，因为在 backward 中：
    # grad_b = A^T @ B，其中 A 和 B 都是按 group 拆分的
    
    ori_dtype = torch.bfloat16
    
    # A: [total_M, K] - will be transposed, so effectively [K, total_M]
    a = torch.randn((total_M, K), dtype=ori_dtype, device=device)
    
    # B: [total_M, N]
    b = torch.randn((total_M, N), dtype=ori_dtype, device=device)
    
    # Quantize to FP8
    a_fp8, a_scale_inv = quantize_fp8(a, float8_e4m3, granularity)
    b_fp8, b_scale_inv = quantize_fp8(b, float8_e4m3, granularity)
    
    # Warmup
    for _ in range(warmup):
        out = grouped_gemm_fp8_variable_k_csrc_impl(
            a_fp8,
            b_fp8,
            a_scale_inv,
            b_scale_inv,
            group_lens,
            group_offs,
            trans_a=True,  # 必须是 True
            trans_b=False,  # 必须是 False
            out_dtype=ori_dtype,
            granularity=granularity,
            num_cu=None,
        )
    torch.cuda.synchronize()
    
    # Benchmark
    timer = Timer(
        stmt="grouped_gemm_fp8_variable_k_csrc_impl(a_fp8, b_fp8, a_scale_inv, b_scale_inv, group_lens, group_offs, True, False, ori_dtype, granularity, None)",
        globals={
            "grouped_gemm_fp8_variable_k_csrc_impl": grouped_gemm_fp8_variable_k_csrc_impl,
            "a_fp8": a_fp8,
            "b_fp8": b_fp8,
            "a_scale_inv": a_scale_inv,
            "b_scale_inv": b_scale_inv,
            "group_lens": group_lens,
            "group_offs": group_offs,
            "ori_dtype": ori_dtype,
            "granularity": granularity,
        },
    )
    
    measurement = timer.timeit(repeat)
    avg_time_ms = measurement.mean * 1e3
    
    # Calculate FLOPs
    # For each group i: output[i] = A[group_lens[i], K]^T @ B[group_lens[i], N]
    # FLOPs per group: 2 * group_lens[i] * K * N
    # Total FLOPs: sum over all groups
    total_flops = 2 * total_M * K * N
    tflops = (total_flops / (avg_time_ms * 1e-3)) / 1e12
    
    # Calculate bandwidth
    # Input: a_fp8 (total_M * K * 1 byte) + b_fp8 (total_M * N * 1 byte)
    # Output: out (B * M * N * 2 bytes for bf16, but varies per group)
    input_bytes = total_M * K + total_M * N
    output_bytes = total_M * N * 2  # Approximate
    total_bytes = input_bytes + output_bytes
    bandwidth_gb_s = (total_bytes / (avg_time_ms * 1e-3)) / 1e9
    
    return {
        "B": B,
        "M_avg": M,
        "M_total": total_M,
        "K": K,
        "N": N,
        "trans_a": "T",
        "trans_b": "N",
        "granularity": granularity.name,
        "balance": balance,
        "latency_ms": f"{avg_time_ms:.3f}",
        "TFLOPS": f"{tflops:.2f}",
        "bandwidth_GB_s": f"{bandwidth_gb_s:.2f}",
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark grouped_gemm_fp8_variable_k_csrc_impl")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run on")
    parser.add_argument("--warmup", type=int, default=20, help="Warmup iterations")
    parser.add_argument("--repeat", type=int, default=100, help="Benchmark iterations")
    parser.add_argument("--output", type=str, default="grouped_gemm_fp8_variable_k_csrc_benchmark.csv", 
                       help="Output CSV file")
    args = parser.parse_args()
    
    print("=" * 80)
    print("Benchmark: grouped_gemm_fp8_variable_k_csrc_impl (CK Backend)")
    print("Variable K version - Used for backward pass grad_b computation")
    print("=" * 80)
    print(f"Device: {args.device}")
    print(f"Warmup: {args.warmup}, Repeat: {args.repeat}")
    print(f"Note: Only trans_a=True, trans_b=False is supported")
    print()
    
    # Test configurations
    # 这些配置模拟 backward pass 中计算 grad_b 的场景
    configs = [
        # DeepSeek-V2 Lite style (Small scale)
        # grad_b = grad_out^T @ activations
        {"B": 8, "M": 512, "K": 2048, "N": 1408, "granularity": ScalingGranularity.TENSORWISE},
        
        # Medium scale
        {"B": 16, "M": 1024, "K": 4096, "N": 4096, "granularity": ScalingGranularity.TENSORWISE},
        
        # Large scale (MoE-like)
        {"B": 64, "M": 2048, "K": 8192, "N": 14336, "granularity": ScalingGranularity.TENSORWISE},
        
        # Very large K (long context scenario)
        {"B": 16, "M": 1024, "K": 16384, "N": 4096, "granularity": ScalingGranularity.TENSORWISE},
        
        # Wide output (large N)
        {"B": 32, "M": 1024, "K": 4096, "N": 28672, "granularity": ScalingGranularity.TENSORWISE},
        
        # Test imbalanced workload
        {"B": 16, "M": 1024, "K": 4096, "N": 4096, "granularity": ScalingGranularity.TENSORWISE, "balance": 0.5},
        {"B": 64, "M": 2048, "K": 8192, "N": 14336, "granularity": ScalingGranularity.TENSORWISE, "balance": 0.5},
    ]
    
    results = []
    
    for i, config in enumerate(configs, 1):
        print(f"[{i}/{len(configs)}] Testing B={config['B']}, M={config['M']}, K={config['K']}, N={config['N']}, "
              f"granularity={config['granularity'].name}, balance={config.get('balance', 1.0)}")
        
        try:
            result = benchmark_grouped_gemm_fp8_variable_k_csrc(
                B=config["B"],
                M=config["M"],
                K=config["K"],
                N=config["N"],
                granularity=config["granularity"],
                balance=config.get("balance", 1.0),
                warmup=args.warmup,
                repeat=args.repeat,
                device=args.device,
            )
            results.append(result)
            print(f"  -> {result['TFLOPS']} TFLOPS, {result['latency_ms']} ms, {result['bandwidth_GB_s']} GB/s")
        except Exception as e:
            print(f"  -> ERROR: {e}")
            import traceback
            traceback.print_exc()
        
        print()
    
    # Save results
    df = pd.DataFrame(results)
    df.to_csv(args.output, index=False)
    
    print("=" * 80)
    print("Summary")
    print("=" * 80)
    print(df.to_string(index=False))
    print()
    print(f"Results saved to: {args.output}")


if __name__ == "__main__":
    main()

