###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import argparse
import os
from datetime import datetime

import pandas as pd
import torch
import torch.utils.benchmark as benchmark
from config import (
    BATCH_SIZE_LIST,
    compute_snr,
    gen_attention_test_cases,
    get_platform_info,
)
from tabulate import tabulate
from torch.nn.attention import SDPBackend, sdpa_kernel


# Disable FP32 atomic for better performance on gfx950
def _is_gfx950():
    """Check if current GPU is gfx950 using torch."""
    props = torch.cuda.get_device_properties(0)
    return props.major == 9 and props.minor == 5


if _is_gfx950():
    os.environ["PRIMUS_TURBO_ATTN_V3_ATOMIC_FP32"] = "0"

import primus_turbo.pytorch as turbo

# PyTorch SDPA backends for reference implementation
ATTN_BACKENDS = [
    SDPBackend.FLASH_ATTENTION,
    SDPBackend.EFFICIENT_ATTENTION,
    SDPBackend.MATH,
]


def attention_ref(q, k, v, sm_scale, causal):
    """Reference attention using PyTorch's scaled_dot_product_attention."""
    num_heads = q.shape[2]
    n_kv_heads = k.shape[2]
    n_rep = num_heads // n_kv_heads

    # BSHD -> BHSD
    q = q.transpose(1, 2).contiguous()
    k = k.transpose(1, 2).contiguous()
    v = v.transpose(1, 2).contiguous()

    with sdpa_kernel(ATTN_BACKENDS):
        o_ref = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, is_causal=causal, scale=sm_scale, enable_gqa=n_rep > 1
        )

    # BHSD -> BSHD
    return o_ref.transpose(1, 2)


def check_attention_correctness(q, k, v, q_ref, k_ref, v_ref, o, o_ref, grad_out, use_fp8):
    """Check correctness of attention forward and backward against PyTorch reference."""
    # Backward pass
    o_ref.backward(grad_out, retain_graph=True)
    o.backward(grad_out, retain_graph=True)

    # Compute SNRs
    out_snr = compute_snr(o_ref, o)
    dq_snr = compute_snr(q_ref.grad, q.grad)
    dk_snr = compute_snr(k_ref.grad, k.grad)
    dv_snr = compute_snr(v_ref.grad, v.grad)

    # SNR thresholds: bf16 requires higher SNR (40), fp8 allows lower (20)
    threshold = 20 if use_fp8 else 40

    correct = all(snr > threshold for snr in [out_snr, dq_snr, dk_snr, dv_snr])
    status = "PASS" if correct else "FAIL"
    print(
        f"Correctness Check: {status} (out={out_snr:.1f}, dq={dq_snr:.1f}, dk={dk_snr:.1f}, dv={dv_snr:.1f})"
    )

    # Reset gradients
    q.grad = None
    k.grad = None
    v.grad = None

    return correct


def profile_attention(batch, seqlen, num_head_q, num_head_kv, head_dim_qk, head_dim_v, causal, use_fp8):
    """Profile attention forward and backward performance."""
    device = "cuda"
    dtype = torch.bfloat16

    # Create tensors
    q = torch.randn((batch, seqlen, num_head_q, head_dim_qk), device=device, dtype=dtype, requires_grad=True)
    k = torch.randn((batch, seqlen, num_head_kv, head_dim_qk), device=device, dtype=dtype, requires_grad=True)
    v = torch.randn((batch, seqlen, num_head_kv, head_dim_v), device=device, dtype=dtype, requires_grad=True)
    q_ref = q.clone().detach().requires_grad_()
    k_ref = k.clone().detach().requires_grad_()
    v_ref = v.clone().detach().requires_grad_()

    sm_scale = head_dim_qk ** (-0.5)

    # Reference forward
    o_ref = attention_ref(q_ref, k_ref, v_ref, sm_scale, causal)

    # Define forward function
    if use_fp8:
        fwd_func = lambda: turbo.ops.flash_attn_fp8_func(
            q,
            k,
            v,
            dropout_p=0.0,
            softmax_scale=sm_scale,
            causal=causal,
            window_size=(-1, -1),
            bias=None,
            alibi_slopes=None,
            deterministic=False,
            return_lse=False,
            return_attn_probs=False,
        )
    else:
        fwd_func = lambda: turbo.ops.flash_attn_func(
            q,
            k,
            v,
            dropout_p=0.0,
            softmax_scale=sm_scale,
            causal=causal,
            window_size=(-1, -1),
            bias=None,
            alibi_slopes=None,
            deterministic=False,
            return_lse=False,
            return_attn_probs=False,
        )

    # Forward pass and correctness check
    out = fwd_func()
    grad_out = torch.randn_like(out)
    correct = check_attention_correctness(q, k, v, q_ref, k_ref, v_ref, out, o_ref, grad_out, use_fp8)

    # Prepare benchmark functions
    out = fwd_func()
    bwd_func = lambda: out.backward(grad_out, retain_graph=True)
    bwd_func()

    # Calculate FLOPs
    fwd_flops = 2 * batch * seqlen * seqlen * num_head_q * (head_dim_qk + head_dim_v)
    if causal:
        fwd_flops //= 2
    bwd_flops = fwd_flops * 2.5

    # Warmup
    for _ in range(20):
        fwd_func()
        bwd_func()
    torch.cuda.synchronize()

    # Benchmark
    fwd_time = benchmark.Timer(stmt="fn()", globals={"fn": fwd_func}).timeit(100).mean * 1e3
    bwd_time = benchmark.Timer(stmt="fn()", globals={"fn": bwd_func}).timeit(100).mean * 1e3
    fwd_tflops = fwd_flops / (fwd_time * 1e-3) / 1e12
    bwd_tflops = bwd_flops / (bwd_time * 1e-3) / 1e12

    print(f"Forward  Mean time: {fwd_time:.3f} ms | TFLOPS: {fwd_tflops:.2f}")
    print(f"Backward Mean time: {bwd_time:.3f} ms | TFLOPS: {bwd_tflops:.2f}")

    return fwd_time, fwd_tflops, bwd_time, bwd_tflops, correct


def benchmark_attention(output_csv=None, use_fp8=False):
    """Run attention benchmark."""
    platform, gpu_name = get_platform_info()

    # Generate test cases from config
    test_cases = gen_attention_test_cases()

    rows = []
    test_id = 0
    total_tests = 2 * len(BATCH_SIZE_LIST) * len(test_cases)
    print(f"Total tests: {total_tests}, FP8: {use_fp8}")

    for causal in [False, True]:
        for case in test_cases:
            num_head_q = case["num_head_q"]
            num_head_kv = case["num_head_kv"]
            head_dim_qk = case["head_dim_qk"]
            head_dim_v = case["head_dim_v"]
            seqlen = case["seqlen"]
            for batch in BATCH_SIZE_LIST:
                test_id += 1

                print(f"\n{'='*60}")
                print(
                    f"TestID: {test_id}, batch={batch}, seqlen={seqlen}, "
                    f"heads={num_head_q}/{num_head_kv}, dim={head_dim_qk}/{head_dim_v}, "
                    f"causal={causal}, fp8={use_fp8}"
                )
                print(f"{'='*60}")

                row = {
                    "TestID": test_id,
                    "Platform": platform,
                    "GPU": gpu_name,
                    "Batch": batch,
                    "SeqLen": seqlen,
                    "num_head_q": num_head_q,
                    "num_head_kv": num_head_kv,
                    "head_dim_qk": head_dim_qk,
                    "head_dim_v": head_dim_v,
                    "Causal": causal,
                }

                try:
                    fwd_time, fwd_tflops, bwd_time, bwd_tflops, correct = profile_attention(
                        batch, seqlen, num_head_q, num_head_kv, head_dim_qk, head_dim_v, causal, use_fp8
                    )
                    row.update(
                        {
                            "Check": "PASS" if correct else "FAIL",
                            "Forward Time (ms)": f"{fwd_time:.2f}",
                            "Forward TFLOPS": f"{fwd_tflops:.2f}",
                            "Backward Time (ms)": f"{bwd_time:.2f}",
                            "Backward TFLOPS": f"{bwd_tflops:.2f}",
                        }
                    )
                except Exception as e:
                    print(f"Failed: {str(e)}")
                    row.update(
                        {
                            "Check": "ERROR",
                            "Forward Time (ms)": "ERROR",
                            "Forward TFLOPS": "0.00",
                            "Backward Time (ms)": "ERROR",
                            "Backward TFLOPS": "0.00",
                        }
                    )

                rows.append(row)

    # Create DataFrame
    results = pd.DataFrame(rows)

    # Print results
    print("\nFinal Results:")
    print(tabulate(results, headers="keys", tablefmt="grid", showindex=False))

    # Print average TFLOPS
    avg_fwd = results["Forward TFLOPS"].astype(float).mean()
    avg_bwd = results["Backward TFLOPS"].astype(float).mean()
    print(f"\nAverage Forward TFLOPS: {avg_fwd:.2f}")
    print(f"Average Backward TFLOPS: {avg_bwd:.2f}")

    # Save to CSV
    if output_csv:
        filename = output_csv
    else:
        timestamp = datetime.now().strftime("%Y%m%d")
        prefix = "attention_fp8_benchmark_result" if use_fp8 else "attention_benchmark_result"
        filename = f"{prefix}_{timestamp}_{gpu_name}.csv"
    results.to_csv(filename, index=False)
    print(f"Results saved to {filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark Attention operations")
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output CSV filename. Default: attention[_fp8]_benchmark_result_{date}_{gpu}.csv",
    )
    parser.add_argument(
        "--fp8",
        action="store_true",
        help="Enable FP8 attention benchmark (default: disabled)",
    )
    args = parser.parse_args()
    benchmark_attention(output_csv=args.output, use_fp8=args.fp8)
