import torch
import torch.utils.benchmark as benchmark

from primus_turbo.pytorch.core.float8 import Format, MXQuantConfig
from primus_turbo.pytorch.ops import gemm_fp8_blockwise
from tests.test_utils import compute_snr

test_configs = [
    {
        "B": [4],
        "M": 4096,
        "N": 2048,
        "K": 7168,
        "ori_dtype": torch.float16,
        "dtype": Format.E4M3,
        "block_size": 128,
    },
    {
        "B": [2],
        "M": 2048,
        "N": 4096,
        "K": 2048,
        "ori_dtype": torch.float16,
        "dtype": Format.E5M2,
        "block_size": 128,
    },
]


def bench_fp8_linear(B, M, N, K, ori_dtype, dtype, block_size, test_backward):
    device = "cuda"
    config = MXQuantConfig(dtype=dtype, block_size=block_size)

    # Prepare inputs
    x = torch.randn((*B, M, K), dtype=ori_dtype, device=device, requires_grad=True)
    w = torch.randn((N, K), dtype=ori_dtype, device=device, requires_grad=True)
    x_ref = x.clone().detach().requires_grad_()
    w_ref = w.clone().detach().requires_grad_()

    fn_forward = lambda: gemm_fp8_blockwise(x, w, config)

    # Reference forward pass
    out_ref = x_ref @ w_ref.T

    grad_out = torch.randn_like(out_ref)
    out_ref.backward(grad_out, retain_graph=True)

    # Forward pass for implementation
    out = fn_forward()
    out.backward(grad_out, retain_graph=True)

    # Compute SNRs
    out_snr = compute_snr(out_ref, out)
    assert out_snr > 20, "out_snr too low"

    x_grad_snr = compute_snr(x_ref.grad, x.grad)
    w_grad_snr = compute_snr(w_ref.grad, w.grad)
    assert x_grad_snr > 20, "x_grad_snr too low"
    assert w_grad_snr > 20, "w_grad_snr too low"

    # Calculate FLOPs
    total_flops = 2 * M * N * K

    if test_backward:
        # Re-run forward pass to get fresh output
        o = fn_forward()
        do = torch.randn_like(o)
        fn = lambda: o.backward(do, retain_graph=True)
        total_flops *= 2  # Approximate factor for backward pass
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

    # DataFrame to store results
    results = pd.DataFrame(
        columns=[
            "Test id",
            "B",
            "M",
            "N",
            "K",
            "ori_dtype",
            "dtype",
            "block_size",
            "Test Backward",
            "Time (ms)",
            "TFLOPS",
        ]
    )
    test_id = 0
    for config in test_configs:
        B = config["B"]
        M = config["M"]
        N = config["N"]
        K = config["K"]
        ori_dtype = config["ori_dtype"]
        dtype = config["dtype"]
        block_size = config["block_size"]
        print(f"\n{'='*50}")
        print(f"Testing config: {config}")
        print(f"{'='*50}")
        for test_backward in [False, True]:
            test_id += 1
            try:
                # Run benchmark
                time_ms, tflops, _ = bench_fp8_linear(
                    B=B,
                    M=M,
                    N=N,
                    K=K,
                    ori_dtype=ori_dtype,
                    dtype=dtype,
                    block_size=block_size,
                    test_backward=test_backward,
                )

                # Add to results table
                new_row = {
                    "Test id": test_id,
                    "B": B,
                    "M": M,
                    "N": N,
                    "K": K,
                    "ori_dtype": ori_dtype,
                    "dtype": dtype,
                    "block_size": block_size,
                    "Test Backward": test_backward,
                    "Time (ms)": f"{time_ms:.2f}",
                    "TFLOPS": f"{tflops:.2f}",
                }
                results = pd.concat([results, pd.DataFrame([new_row])], ignore_index=True)

            except Exception as e:
                print(f"Failed to run {config}: {str(e)}")
                new_row = {
                    "Test id": test_id,
                    "B": B,
                    "M": M,
                    "N": N,
                    "K": K,
                    "ori_dtype": ori_dtype,
                    "dtype": dtype,
                    "block_size": block_size,
                    "Test Backward": test_backward,
                    "Time (ms)": "Failed",
                    "TFLOPS": "N/A",
                }
                results = pd.concat([results, pd.DataFrame([new_row])], ignore_index=True)

    # Print results
    print("\nFinal Results:")
    print(tabulate(results, headers="keys", tablefmt="grid", showindex=False))

    # Save to CSV
    results.to_csv("fp8_linear_benchmark_results.csv", index=False)
    print("Results saved to fp8_linear_benchmark_results.csv")
