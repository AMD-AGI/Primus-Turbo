import torch
import torch.utils.benchmark as benchmark

import primus_turbo.pytorch as pt
from tests.pytorch.ref.attention_ref import (
    AttnConfig,
    attention_vanilla_forward_pytorch_ref_impl,
)
from tests.test_utils import compute_snr

test_cases = [
    AttnConfig(seqlen_q=4096, seqlen_kv=4096, num_head_q=32, num_head_kv=32, head_dim_qk=128, head_dim_v=128),
    AttnConfig(seqlen_q=4096, seqlen_kv=4096, num_head_q=64, num_head_kv=8, head_dim_qk=128, head_dim_v=128),
    AttnConfig(seqlen_q=4096, seqlen_kv=4096, num_head_q=32, num_head_kv=8, head_dim_qk=128, head_dim_v=128),
    AttnConfig(seqlen_q=4096, seqlen_kv=4096, num_head_q=64, num_head_kv=8, head_dim_qk=128, head_dim_v=128),
    AttnConfig(seqlen_q=4096, seqlen_kv=4096, num_head_q=16, num_head_kv=16, head_dim_qk=192, head_dim_v=128),
    AttnConfig(
        seqlen_q=4096, seqlen_kv=4096, num_head_q=128, num_head_kv=128, head_dim_qk=192, head_dim_v=128
    ),
    AttnConfig(seqlen_q=4096, seqlen_kv=4096, num_head_q=32, num_head_kv=8, head_dim_qk=128, head_dim_v=128),
    AttnConfig(seqlen_q=4096, seqlen_kv=4096, num_head_q=48, num_head_kv=8, head_dim_qk=128, head_dim_v=128),
]


def bench_attention(batch, config, causal: bool, backend_type: str, use_fp8: bool, test_backward: bool):
    device = "cuda"
    dtype = torch.bfloat16
    seqlen_q, seqlen_kv, num_head_q, num_head_kv, head_dim_qk, head_dim_v = (
        config.seqlen_q,
        config.seqlen_kv,
        config.num_head_q,
        config.num_head_kv,
        config.head_dim_qk,
        config.head_dim_v,
    )
    q_layout = (batch, seqlen_q, num_head_q, head_dim_qk)
    k_layout = (batch, seqlen_kv, num_head_kv, head_dim_qk)
    v_layout = (batch, seqlen_kv, num_head_kv, head_dim_v)

    query = torch.randn(q_layout, device=device, dtype=dtype, requires_grad=True)
    key = torch.randn(k_layout, device=device, dtype=dtype, requires_grad=True)
    value = torch.randn(v_layout, device=device, dtype=dtype, requires_grad=True)
    query_ref = query.clone().detach().requires_grad_()
    key_ref = key.clone().detach().requires_grad_()
    value_ref = value.clone().detach().requires_grad_()

    sm_scale = query.shape[-1] ** (-0.5)

    o_ref = attention_vanilla_forward_pytorch_ref_impl(query_ref, key_ref, value_ref, sm_scale, causal)
    if use_fp8 == False:
        fn_forward = lambda: pt.ops.attention(
            query,
            key,
            value,
            dropout_p=0.0,
            softmax_scale=sm_scale,
            causal=causal,
            window_size=(-1, -1),
            bias=None,
            alibi_slopes=None,
            deterministic=False,
            return_lse=False,
            return_attn_probs=False,
            backend_type=backend_type,
        )
    else:
        fn_forward = lambda: pt.ops.attention_fp8_blockwise(
            query,
            key,
            value,
            dropout_p=0.0,
            softmax_scale=sm_scale,
            causal=causal,
            window_size=(-1, -1),
            bias=None,
            alibi_slopes=None,
            deterministic=False,
            return_lse=False,
            return_attn_probs=False,
            backend_type=backend_type,
        )

    # Forward pass
    o = fn_forward()
    grad_output = torch.randn_like(o)

    # Backward pass for reference
    o_ref.backward(grad_output, retain_graph=True)
    # Backward pass for implementation
    o.backward(grad_output, retain_graph=True)

    # Compute SNRs
    out_snr = compute_snr(o_ref, o)
    query_grad_snr = compute_snr(query_ref.grad, query.grad)
    key_grad_snr = compute_snr(key_ref.grad, key.grad)
    value_grad_snr = compute_snr(value_ref.grad, value.grad)

    # Verify SNRs meet requirements
    assert out_snr > 20, "out_snr too low"
    assert query_grad_snr > 15, "query_grad_snr too low"
    assert key_grad_snr > 15, "key_grad_snr too low"
    assert value_grad_snr > 15, "value_grad_snr too low"

    # Calculate FLOPs
    total_flops = (
        2 * batch * seqlen_q * seqlen_kv * num_head_q * head_dim_qk
        + 2 * batch * seqlen_q * seqlen_kv * num_head_q * head_dim_v
    )
    if causal:
        total_flops //= 2

    if test_backward:
        # Re-run forward pass to get fresh output
        o = fn_forward()
        do = torch.randn_like(o)
        fn = lambda: o.backward(do, retain_graph=True)
        total_flops *= 2.5  # Approximate factor for backward pass
    else:
        fn = fn_forward

    # Warmup
    warmup = 5
    for _ in range(warmup):
        _ = fn()
    torch.cuda.synchronize()

    # Benchmark
    timer = benchmark.Timer(
        stmt="fn()",
        globals={"fn": fn},
    )
    measurement = timer.timeit(100)
    mean_time_ms = measurement.mean * 1e3
    flops_per_sec = total_flops / (mean_time_ms * 1e-3) / 1e12  # TFLOPs/s

    print(f"Mean time: {mean_time_ms:.3f} ms | TFLOPS: {flops_per_sec:.2f}")
    return mean_time_ms, flops_per_sec, total_flops


if __name__ == "__main__":
    import pandas as pd
    from tabulate import tabulate

    # Define test configurations
    test_configs = [
        {"causal": False, "backend": "ck", "fp8": False, "test_backward": False},
        {"causal": True, "backend": "ck", "fp8": False, "test_backward": False},
        {"causal": False, "backend": "ck", "fp8": False, "test_backward": True},
        {"causal": True, "backend": "ck", "fp8": False, "test_backward": True},
        {"causal": False, "backend": "triton", "fp8": True, "test_backward": False},
        {"causal": True, "backend": "triton", "fp8": True, "test_backward": False},
        {"causal": False, "backend": "triton", "fp8": True, "test_backward": True},
        {"causal": True, "backend": "triton", "fp8": True, "test_backward": True},
    ]

    # DataFrame to store results
    results = pd.DataFrame(
        columns=[
            "Test id",
            "Causal",
            "Backend",
            "FP8",
            "Test Backward",
            "num_head_q",
            "num_head_kv",
            "head_dim_qk",
            "head_dim_v",
            "Time (ms)",
            "TFLOPS",
        ]
    )
    test_id = 0
    for test_case in test_cases:
        print(f"\n{'='*50}")
        print(f"Testing config: {test_case}")
        print(f"{'='*50}")
        for config in test_configs:
            test_id += 1
            try:
                # Run benchmark
                time_ms, tflops, _ = bench_attention(
                    batch=4,
                    config=test_case,
                    causal=config["causal"],
                    backend_type=config["backend"],
                    use_fp8=config["fp8"],
                    test_backward=config["test_backward"],
                )

                # Add to results table
                new_row = {
                    "Test id": test_id,
                    "Causal": config["causal"],
                    "Backend": config["backend"],
                    "FP8": config["fp8"],
                    "Test Backward": config["test_backward"],
                    "Time (ms)": f"{time_ms:.2f}",
                    "TFLOPS": f"{tflops:.2f}",
                    "num_head_q": test_case.num_head_q,
                    "num_head_kv": test_case.num_head_kv,
                    "head_dim_qk": test_case.head_dim_qk,
                    "head_dim_v": test_case.head_dim_v,
                }
                results = pd.concat([results, pd.DataFrame([new_row])], ignore_index=True)

            except Exception as e:
                print(f"Failed to run {config}: {str(e)}")
                new_row = {
                    "Test id": test_id,
                    "Causal": config["causal"],
                    "Backend": config["backend"],
                    "FP8": config["fp8"],
                    "Test Backward": config["test_backward"],
                    "Time (ms)": "Failed",
                    "TFLOPS": "N/A",
                    "num_head_q": test_case.num_head_q,
                    "num_head_kv": test_case.num_head_kv,
                    "head_dim_qk": test_case.head_dim_qk,
                    "head_dim_v": test_case.head_dim_v,
                }
                results = pd.concat([results, pd.DataFrame([new_row])], ignore_index=True)

    # Print results
    print("\nFinal Results:")
    print(tabulate(results, headers="keys", tablefmt="grid", showindex=False))

    # Save to CSV
    results.to_csv("attention_benchmark_results.csv", index=False)
    print("Results saved to attention_benchmark_results.csv")
