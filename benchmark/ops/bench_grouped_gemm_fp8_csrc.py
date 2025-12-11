#!/usr/bin/env python3
"""
Benchmark for grouped_gemm_fp8_csrc_impl (CK backend)
直接测试 C++ kernel 实现，不经过 autograd
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
    grouped_gemm_fp8_csrc_impl,
    grouped_gemm_compute_offs,
)
from primus_turbo.pytorch.ops.quantization import quantize_fp8


def generate_grouped_gemm_group_lens(B: int, M: int, balance: float = 1.0):
    """
    生成 group_lens，类似 bench_grouped_gemm_fp8.py
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
            # Distribute difference
            group_lens[0] += diff
    
    return group_lens


def benchmark_grouped_gemm_fp8_csrc(
    B: int,  # Number of groups
    M: int,  # Rows per group (average)
    N: int,  # Columns
    K: int,  # Inner dimension
    trans_b: bool = True,
    granularity: ScalingGranularity = ScalingGranularity.TENSORWISE,
    balance: float = 1.0,
    warmup: int = 20,
    repeat: int = 100,
    device: str = "cuda",
):
    """
    Benchmark grouped_gemm_fp8_csrc_impl
    
    Args:
        B: Number of groups
        M: Average rows per group
        N: Output columns
        K: Inner dimension (K)
        trans_b: Whether B is transposed
        granularity: Scaling granularity (TENSORWISE/ROWWISE/BLOCKWISE)
        balance: Load balance factor (1.0 = perfect balance)
        warmup: Warmup iterations
        repeat: Benchmark iterations
        device: Device to run on
    """
    
    # Generate group lengths
    group_lens = generate_grouped_gemm_group_lens(B, M, balance).to(device)
    total_M = group_lens.sum().item()
    group_offs = grouped_gemm_compute_offs(group_lens)
    
    # Create input tensors in BF16
    ori_dtype = torch.bfloat16
    a = torch.randn((total_M, K), dtype=ori_dtype, device=device)
    
    if trans_b:
        b = torch.randn((B, N, K), dtype=ori_dtype, device=device)
    else:
        b = torch.randn((B, K, N), dtype=ori_dtype, device=device)
    
    # Quantize to FP8
    a_fp8, a_scale_inv = quantize_fp8(a, float8_e4m3, granularity)
    
    if trans_b:
        b_fp8, b_scale_inv = quantize_fp8(b, float8_e4m3, granularity, axis=-1)
    else:
        b_fp8, b_scale_inv = quantize_fp8(b, float8_e4m3, granularity, axis=-2)
    
    # Warmup
    for _ in range(warmup):
        out = grouped_gemm_fp8_csrc_impl(
            a_fp8,
            b_fp8,
            a_scale_inv,
            b_scale_inv,
            group_lens,
            group_offs,
            trans_a=False,
            trans_b=trans_b,
            out_dtype=ori_dtype,
            granularity=granularity,
            num_cu=None,
        )
    torch.cuda.synchronize()
    
    # Benchmark
    timer = Timer(
        stmt="grouped_gemm_fp8_csrc_impl(a_fp8, b_fp8, a_scale_inv, b_scale_inv, group_lens, group_offs, False, trans_b, ori_dtype, granularity, None)",
        globals={
            "grouped_gemm_fp8_csrc_impl": grouped_gemm_fp8_csrc_impl,
            "a_fp8": a_fp8,
            "b_fp8": b_fp8,
            "a_scale_inv": a_scale_inv,
            "b_scale_inv": b_scale_inv,
            "group_lens": group_lens,
            "group_offs": group_offs,
            "trans_b": trans_b,
            "ori_dtype": ori_dtype,
            "granularity": granularity,
        },
    )
    
    measurement = timer.timeit(repeat)
    avg_time_ms = measurement.mean * 1e3
    
    # Calculate FLOPs
    total_flops = 2 * total_M * N * K
    tflops = (total_flops / (avg_time_ms * 1e-3)) / 1e12
    
    # Calculate bandwidth
    # Input: a_fp8 (total_M * K * 1 byte) + b_fp8 (B * N * K * 1 byte or B * K * N * 1 byte)
    # Output: out (total_M * N * 2 bytes for bf16)
    # Scales: depends on granularity
    input_bytes = total_M * K + B * N * K
    output_bytes = total_M * N * 2
    total_bytes = input_bytes + output_bytes
    bandwidth_gb_s = (total_bytes / (avg_time_ms * 1e-3)) / 1e9
    
    return {
        "B": B,
        "M_avg": M,
        "M_total": total_M,
        "N": N,
        "K": K,
        "trans_b": "T" if trans_b else "N",
        "granularity": granularity.name,
        "balance": balance,
        "latency_ms": f"{avg_time_ms:.3f}",
        "TFLOPS": f"{tflops:.2f}",
        "bandwidth_GB_s": f"{bandwidth_gb_s:.2f}",
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark grouped_gemm_fp8_csrc_impl")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run on")
    parser.add_argument("--warmup", type=int, default=20, help="Warmup iterations")
    parser.add_argument("--repeat", type=int, default=100, help="Benchmark iterations")
    parser.add_argument("--output", type=str, default="grouped_gemm_fp8_csrc_benchmark.csv", 
                       help="Output CSV file")
    args = parser.parse_args()
    
    print("=" * 80)
    print("Benchmark: grouped_gemm_fp8_csrc_impl (CK Backend)")
    print("=" * 80)
    print(f"Device: {args.device}")
    print(f"Warmup: {args.warmup}, Repeat: {args.repeat}")
    print()
    
    # Test configurations
    configs = [
        # DeepSeek-V2 Lite style (Small scale)
        {"B": 8, "M": 512, "N": 1408, "K": 2048, "trans_b": True, "granularity": ScalingGranularity.TENSORWISE},
        {"B": 8, "M": 512, "N": 1408, "K": 2048, "trans_b": False, "granularity": ScalingGranularity.TENSORWISE},
        
        # Medium scale
        {"B": 16, "M": 1024, "N": 4096, "K": 4096, "trans_b": True, "granularity": ScalingGranularity.TENSORWISE},
        {"B": 16, "M": 1024, "N": 4096, "K": 4096, "trans_b": False, "granularity": ScalingGranularity.TENSORWISE},
        
        # Large scale (MoE-like)
        {"B": 64, "M": 2048, "N": 14336, "K": 8192, "trans_b": True, "granularity": ScalingGranularity.TENSORWISE},
        {"B": 64, "M": 2048, "N": 14336, "K": 8192, "trans_b": False, "granularity": ScalingGranularity.TENSORWISE},
        
        
        # Test imbalanced workload
        {"B": 16, "M": 1024, "N": 4096, "K": 4096, "trans_b": True, "granularity": ScalingGranularity.TENSORWISE, "balance": 0.5},
        {"B": 16, "M": 1024, "N": 4096, "K": 4096, "trans_b": False, "granularity": ScalingGranularity.TENSORWISE, "balance": 0.5},
    ]
    
    results = []
    
    for i, config in enumerate(configs, 1):
        print(f"[{i}/{len(configs)}] Testing B={config['B']}, M={config['M']}, N={config['N']}, K={config['K']}, "
              f"trans_b={config['trans_b']}, granularity={config['granularity'].name}, "
              f"balance={config.get('balance', 1.0)}")
        
        try:
            result = benchmark_grouped_gemm_fp8_csrc(
                B=config["B"],
                M=config["M"],
                N=config["N"],
                K=config["K"],
                trans_b=config["trans_b"],
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

