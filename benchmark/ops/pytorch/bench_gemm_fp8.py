###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import pandas as pd
import torch
import torch.utils.benchmark as benchmark
from tabulate import tabulate

from primus_turbo.pytorch.core.low_precision import (
    Float8QuantConfig,
    Format,
    ScalingGranularity,
)
from primus_turbo.pytorch.ops import gemm_fp8

ModelConfigs = {
    "llama2-7b": {
        "seqlen": 4096,
        "hidden_size": 4096,
        "intermediate_size": 11008,
        "num_attention_heads": 32,
        "num_key_value_heads": 32,
        "head_dim": 128,
    },
    "llama2-70b": {
        "seqlen": 4096,
        "hidden_size": 8192,
        "intermediate_size": 28672,
        "num_attention_heads": 64,
        "num_key_value_heads": 8,
        "head_dim": 128,
    },
    "llama3.1-8b": {
        "seqlen": 8192,
        "hidden_size": 4096,
        "intermediate_size": 14336,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "head_dim": 128,
    },
    "llama3.1-405B": {
        "seqlen": 8192,
        "hidden_size": 16384,
        "intermediate_size": 53248,
        "num_attention_heads": 128,
        "num_key_value_heads": 8,
        "head_dim": 128,
    },
}


def gen_gemm_test_cases(model_config):
    seq = model_config["seqlen"]
    hidden_size = model_config["hidden_size"]
    intermediate_size = model_config["intermediate_size"]
    num_attention_heads = model_config["num_attention_heads"]
    num_key_value_heads = model_config["num_key_value_heads"]
    head_dim = model_config["head_dim"]

    # [[m, n, k]...]
    gemm_shape_list = []
    # attn qkv pass
    gemm_shape_list.append(
        [
            seq,
            int((num_attention_heads + 2 * num_key_value_heads) * head_dim),
            hidden_size,
        ]
    )
    # attn out
    gemm_shape_list.append([seq, hidden_size, hidden_size])
    # mlp gate+up
    gemm_shape_list.append([seq, int(2 * intermediate_size), hidden_size])
    # mlp down
    gemm_shape_list.append([seq, hidden_size, intermediate_size])
    return gemm_shape_list


def quantize_to_fp8(x, dtype=torch.float8_e4m3fn):
    amax = x.abs().max()
    scale = torch.finfo(dtype).max / amax.clamp(min=1e-12)
    x_fp8 = (x * scale).to(dtype)
    scale = scale.to(torch.float32)
    return x_fp8, scale.reciprocal().reshape(1)


def profile_gemm_fp8(M, N, K, ori_dtype, format, granularity, trans_b):
    device = "cuda"
    b_shape = (N, K) if trans_b else (K, N)
    a = torch.randn((M, K), dtype=ori_dtype, device=device, requires_grad=True)
    b = torch.randn(b_shape, dtype=ori_dtype, device=device, requires_grad=True)
    
    a_ref = a.detach().clone().requires_grad_(True)
    b_ref = b.detach().clone().requires_grad_(True)

    config = Float8QuantConfig(format=format, granularity=granularity)

    # Reference forward pass
    a_fp8, scale_a = quantize_to_fp8(a_ref)
    b_fp8, scale_b = quantize_to_fp8(b_ref)

    if trans_b:
        b_fp8_in = b_fp8.transpose(-1, -2)
    else:
        b_fp8_in = b_fp8

    fwd_func_ref = lambda: torch._scaled_mm(
        a_fp8, b_fp8_in, scale_a, scale_b, out_dtype=ori_dtype
    )
    out_ref = fwd_func_ref()
    grad_out = torch.randn_like(out_ref)
    bwd_func_ref = lambda: out_ref.backward(grad_out, retain_graph=True)

    ref_bwd_supported = True
    try:
        bwd_func_ref()
    except RuntimeError:
        ref_bwd_supported = False

    # Forward pass for implementation
    out = gemm_fp8(a, b, trans_b=trans_b, config=config)
    # Forward pass for implementation
    fwd_func = lambda: gemm_fp8(a, b, trans_b=trans_b, config=config)
    bwd_func = lambda: out.backward(grad_out, retain_graph=True)
    out = fwd_func()
    bwd_func()

    # Calculate FLOPs
    fwd_total_flops = 2 * M * N * K
    bwd_total_flops = 2 * fwd_total_flops

    # Warmup
    warmup = 20
    for _ in range(warmup):
        fwd_func()
        bwd_func()
        fwd_func_ref()
        if ref_bwd_supported:
            bwd_func_ref()
    torch.cuda.synchronize()

    # Benchmark
    fwd_timer = benchmark.Timer(
        stmt="fn()",
        globals={"fn": fwd_func},
    )
    bwd_timer = benchmark.Timer(
        stmt="fn()",
        globals={"fn": bwd_func},
    )
    fwd_ref_timer = benchmark.Timer(
        stmt="fn()",
        globals={"fn": fwd_func_ref},
    )

    fwd_measurement = fwd_timer.timeit(100)
    bwd_measurement = bwd_timer.timeit(100)
    fwd_ref_measurement = fwd_ref_timer.timeit(100)

    fwd_mean_time_ms = fwd_measurement.mean * 1e3
    bwd_mean_time_ms = bwd_measurement.mean * 1e3
    fwd_tflops = fwd_total_flops / (fwd_mean_time_ms * 1e-3) / 1e12
    bwd_tflops = bwd_total_flops / (bwd_mean_time_ms * 1e-3) / 1e12
    
    fwd_ref_mean_time_ms = fwd_ref_measurement.mean * 1e3
    fwd_ref_tflops = fwd_total_flops / (fwd_ref_mean_time_ms * 1e-3) / 1e12

    if ref_bwd_supported:
        bwd_ref_timer = benchmark.Timer(
            stmt="fn()",
            globals={"fn": bwd_func_ref},
        )
        bwd_ref_measurement = bwd_ref_timer.timeit(100)
        bwd_ref_mean_time_ms = bwd_ref_measurement.mean * 1e3
        bwd_ref_tflops = bwd_total_flops / (bwd_ref_mean_time_ms * 1e-3) / 1e12
    else:
        bwd_ref_mean_time_ms = float("nan")
        bwd_ref_tflops = float("nan")

    print(f"Forward  Mean time: {fwd_mean_time_ms:.3f} ms | TFLOPS: {fwd_tflops:.2f}")
    print(f"Backward Mean time: {bwd_mean_time_ms:.3f} ms | TFLOPS: {bwd_tflops:.2f}")
    print(f"Ref Forward  Mean time: {fwd_ref_mean_time_ms:.3f} ms | TFLOPS: {fwd_ref_tflops:.2f}")
    if ref_bwd_supported:
        print(f"Ref Backward Mean time: {bwd_ref_mean_time_ms:.3f} ms | TFLOPS: {bwd_ref_tflops:.2f}")

    return fwd_mean_time_ms, fwd_tflops, bwd_mean_time_ms, bwd_tflops, fwd_ref_mean_time_ms, fwd_ref_tflops, bwd_ref_mean_time_ms, bwd_ref_tflops


def benchmark_gemm_fp8():
    # DataFrame to store results
    results = pd.DataFrame(
        columns=[
            "TestID",
            "Case",
            "MBS",
            "M",
            "N",
            "K",
            "format",
            "granularity",
            "Forward Time (ms)",
            "Forward TFLOPS",
            "Backward Time (ms)",
            "Backward TFLOPS",
            "Ref Forward Time (ms)",
            "Ref Forward TFLOPS",
            "Ref Backward Time (ms)",
            "Ref Backward TFLOPS",
        ]
    )

    MBS_LIST = [1]
    test_id = 0
    ori_dtype = torch.bfloat16
    format = Format.E4M3
    granularity = ScalingGranularity.TENSORWISE
    trans_b = True
    for model_name, model_config in ModelConfigs.items():
        test_cases = gen_gemm_test_cases(model_config)
        for MBS in MBS_LIST:
            for shape in test_cases:
                test_id += 1

                M = shape[0] * MBS
                N = shape[1]
                K = shape[2]

                print(f"\n{'='*50}")
                print(
                    f"Testing Case: {model_name} with MBS={MBS}, M={M}, N={N}, K={K}, format={format}, granularity={granularity}"
                )
                print(f"{'='*50}")

                (
                    fwd_time_ms,
                    fwd_tflops,
                    bwd_time_ms,
                    bwd_tflops,
                    fwd_ref_time_ms,
                    fwd_ref_tflops,
                    bwd_ref_time_ms,
                    bwd_ref_tflops,
                ) = profile_gemm_fp8(
                    M, N, K, ori_dtype, format, granularity, trans_b
                )
                # Add to results table
                new_row = {
                    "TestID": test_id,
                    "Case": model_name,
                    "MBS": MBS,
                    "M": M,
                    "N": N,
                    "K": K,
                    "format": format,
                    "granularity": granularity,
                    "Forward Time (ms)": f"{fwd_time_ms:.2f}",
                    "Forward TFLOPS": f"{fwd_tflops:.2f}",
                    "Backward Time (ms)": f"{bwd_time_ms:.2f}",
                    "Backward TFLOPS": f"{bwd_tflops:.2f}",
                    "Ref Forward Time (ms)": f"{fwd_ref_time_ms:.2f}",
                    "Ref Forward TFLOPS": f"{fwd_ref_tflops:.2f}",
                    "Ref Backward Time (ms)": f"{bwd_ref_time_ms:.2f}",
                    "Ref Backward TFLOPS": f"{bwd_ref_tflops:.2f}",
                }
                results = pd.concat([results, pd.DataFrame([new_row])], ignore_index=True)

    # Print results
    print("\nFinal Results:")
    print(tabulate(results, headers="keys", tablefmt="grid", showindex=False))

    # Save to CSV
    results.to_csv("gemm_fp8_benchmark_results.csv", index=False)
    print("Results saved to gemm_fp8_benchmark_results.csv")


if __name__ == "__main__":
    benchmark_gemm_fp8()
