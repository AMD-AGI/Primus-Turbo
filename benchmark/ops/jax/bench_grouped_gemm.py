###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import time

import jax
import jax.numpy as jnp

from primus_turbo.jax.lax.grouped_gemm import grouped_gemm
from tests.jax.ref.gemm_ref import generate_grouped_gemm_group_lens, grouped_gemm_ref
from tests.jax.test_utils import compute_snr

# Configure JAX
jax.config.update("jax_enable_x64", True)

M_SIZE_LIST = [512, 1024, 2048, 4096, 8192, 16384]
EP_SIZE_LIST = [32, 16, 8]


def _generate_moe_test_cases(
    name_prefix: str,
    n_routed_experts: int,
    moe_intermediate_size: int,
    hidden_size: int,
):
    test_cases = []
    shapes_dict = {
        f"{name_prefix}-GateUP": (2 * moe_intermediate_size, hidden_size),
        f"{name_prefix}-Down": (hidden_size, moe_intermediate_size),
    }

    for ep in EP_SIZE_LIST:
        if n_routed_experts % ep != 0:
            continue
        B = n_routed_experts // ep
        if B < 1:
            continue
        for M in M_SIZE_LIST:
            for name, (N, K) in shapes_dict.items():
                for dtype in [jnp.bfloat16]:
                    test_cases.append(
                        {
                            "Case": name,
                            "B": B,
                            "M": M,
                            "N": N,
                            "K": K,
                            "dtype": dtype,
                        }
                    )
    return test_cases


def generate_deepseekv3_test_cases():
    return _generate_moe_test_cases(
        "DSV3", n_routed_experts=256, moe_intermediate_size=2048, hidden_size=7168
    )


def generate_deepseekv2_test_cases():
    return _generate_moe_test_cases(
        "DSV2", n_routed_experts=160, moe_intermediate_size=1536, hidden_size=5120
    )


def generate_deepseekv2_lite_test_cases():
    return _generate_moe_test_cases(
        "DSV2-Lite", n_routed_experts=64, moe_intermediate_size=1408, hidden_size=2048
    )


def generate_grok_v2_test_cases():
    # https://huggingface.co/xai-org/grok-2/blob/main/config.json
    return _generate_moe_test_cases(
        "Grok-V2", n_routed_experts=8, moe_intermediate_size=16384, hidden_size=8192
    )


def bench_grouped_gemm(B, M, N, K, dtype):
    # Prepare inputs
    key = jax.random.PRNGKey(0)
    key1, key2 = jax.random.split(key)

    x = jax.random.normal(key1, (B * M, K), dtype=dtype)
    w = jax.random.normal(key2, (B, N, K), dtype=dtype)
    group_lens = generate_grouped_gemm_group_lens(B, M, balance=True)

    print("group_lens: ", group_lens)

    # Device put to ensure data is on device
    x = jax.device_put(x)
    w = jax.device_put(w)
    group_lens = jax.device_put(group_lens)

    # Reference forward pass
    x_ref = x.astype(jnp.float32)
    w_ref = w.astype(jnp.float32)
    out_ref = grouped_gemm_ref(x_ref, w_ref, group_lens, trans_b=True)

    # Forward pass for implementation
    out = grouped_gemm(x, w, group_lens, transB=True)

    # Check forward pass
    out_snr = compute_snr(out_ref.astype(dtype), out)
    if out_snr <= 20:
        print(f"out_snr too low: {out_snr}")
    assert out_snr > 20, "out_snr too low"

    # Define loss functions for gradient computation
    def loss_fn(a, b):
        return jnp.sum(grouped_gemm(a, b, group_lens, transB=True))

    def loss_fn_ref(a, b):
        return jnp.sum(grouped_gemm_ref(a, b, group_lens, trans_b=True))

    # Compute gradients
    grad_fn = jax.grad(loss_fn, argnums=(0, 1))
    grad_fn_ref = jax.grad(loss_fn_ref, argnums=(0, 1))

    grad_x, grad_w = grad_fn(x, w)
    grad_x_ref, grad_w_ref = grad_fn_ref(x_ref, w_ref)

    # Check gradients
    x_grad_snr = compute_snr(grad_x_ref.astype(dtype), grad_x)
    w_grad_snr = compute_snr(grad_w_ref.astype(dtype), grad_w)

    if x_grad_snr <= 20:
        print(f"x_grad_snr too low: {x_grad_snr}")
    if w_grad_snr <= 20:
        print(f"w_grad_snr too low: {w_grad_snr}")
    assert x_grad_snr > 20, "x_grad_snr too low"
    assert w_grad_snr > 20, "w_grad_snr too low"

    # Calculate FLOPs
    fwd_total_flops = 2 * B * M * N * K
    bwd_total_flops = 2 * fwd_total_flops

    # JIT compile functions
    fwd_func = jax.jit(lambda: grouped_gemm(x, w, group_lens, transB=True))

    # For backward, we need to use value_and_grad
    def bwd_func_impl():
        _, grads = jax.value_and_grad(loss_fn, argnums=(0, 1))(x, w)
        return grads

    bwd_func = jax.jit(bwd_func_impl)

    # Warmup
    warmup = 20
    for _ in range(warmup):
        _ = fwd_func()
        _ = bwd_func()
    jax.block_until_ready(fwd_func())
    jax.block_until_ready(bwd_func())

    # Benchmark forward pass
    num_iters = 100
    start_time = time.perf_counter()
    for _ in range(num_iters):
        result = fwd_func()
    jax.block_until_ready(result)
    end_time = time.perf_counter()
    fwd_mean_time_ms = (end_time - start_time) * 1e3 / num_iters

    # Benchmark backward pass
    start_time = time.perf_counter()
    for _ in range(num_iters):
        result = bwd_func()
    jax.block_until_ready(result)
    end_time = time.perf_counter()
    bwd_mean_time_ms = (end_time - start_time) * 1e3 / num_iters

    fwd_tflops = fwd_total_flops / (fwd_mean_time_ms * 1e-3) / 1e12
    bwd_tflops = bwd_total_flops / (bwd_mean_time_ms * 1e-3) / 1e12

    print(f"Forward  Mean time: {fwd_mean_time_ms:.3f} ms | TFLOPS: {fwd_tflops:.2f}")
    print(f"Backward Mean time: {bwd_mean_time_ms:.3f} ms | TFLOPS: {bwd_tflops:.2f}")
    return fwd_mean_time_ms, fwd_tflops, bwd_mean_time_ms, bwd_tflops


if __name__ == "__main__":
    dsv2_lite_test_cases = generate_deepseekv2_lite_test_cases()
    dsv2_test_cases = generate_deepseekv2_test_cases()
    dsv3_test_cases = generate_deepseekv3_test_cases()
    grok_v2_test_cases = generate_grok_v2_test_cases()
    test_cases = dsv2_lite_test_cases + dsv2_test_cases + dsv3_test_cases + grok_v2_test_cases

    import pandas as pd
    from tabulate import tabulate

    # DataFrame to store results
    results = pd.DataFrame(
        columns=[
            "TestID",
            "Case",
            "B",
            "M",
            "N",
            "K",
            "dtype",
            "Forward Time (ms)",
            "Forward TFLOPS",
            "Backward Time (ms)",
            "Backward TFLOPS",
        ]
    )
    test_id = 0
    for case in test_cases:
        B = case["B"]
        M = case["M"]
        N = case["N"]
        K = case["K"]
        dtype = case["dtype"]
        print(f"\n{'='*50}")
        print(f"Testing Case: {case}")
        print(f"{'='*50}")
        test_id += 1
        try:
            # Run benchmark
            fwd_time_ms, fwd_tflops, bwd_time_ms, bwd_tflops = bench_grouped_gemm(
                B=B,
                M=M,
                N=N,
                K=K,
                dtype=dtype,
            )

            # Add to results table
            new_row = {
                "TestID": test_id,
                "Case": case["Case"],
                "B": B,
                "M": M,
                "N": N,
                "K": K,
                "dtype": dtype,
                "Forward Time (ms)": f"{fwd_time_ms:.2f}",
                "Forward TFLOPS": f"{fwd_tflops:.2f}",
                "Backward Time (ms)": f"{bwd_time_ms:.2f}",
                "Backward TFLOPS": f"{bwd_tflops:.2f}",
            }
            results = pd.concat([results, pd.DataFrame([new_row])], ignore_index=True)

        except Exception as e:
            print(f"Failed to run {case}: {str(e)}")
            new_row = {
                "TestID": test_id,
                "Case": case["Case"],
                "B": B,
                "M": M,
                "N": N,
                "K": K,
                "dtype": dtype,
                "Forward Time (ms)": "Failed",
                "Forward TFLOPS": "N/A",
                "Backward Time (ms)": "Failed",
                "Backward TFLOPS": "N/A",
            }
            results = pd.concat([results, pd.DataFrame([new_row])], ignore_index=True)

    # Print results
    print("\nFinal Results:")
    print(tabulate(results, headers="keys", tablefmt="grid", showindex=False))

    # Save to CSV
    results.to_csv("grouped_gemm_benchmark_results_jax.csv", index=False)
    print("Results saved to grouped_gemm_benchmark_results_jax.csv")
