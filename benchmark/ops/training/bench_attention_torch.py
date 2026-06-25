###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""PyTorch Attention Baseline Benchmark using SDPA."""

import argparse
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

ATTN_BACKENDS = [
    SDPBackend.CUDNN_ATTENTION,
    SDPBackend.FLASH_ATTENTION,
    SDPBackend.EFFICIENT_ATTENTION,
    SDPBackend.MATH,
]


def attention_torch(q, k, v, sm_scale, causal, enable_gqa):
    with sdpa_kernel(ATTN_BACKENDS, set_priority=True):
        out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, is_causal=causal, scale=sm_scale, enable_gqa=enable_gqa
        )
    return out


def check_attention_correctness(q, k, v, out, grad_out, sm_scale, causal, enable_gqa):
    q_ref = q.detach().clone().requires_grad_()
    k_ref = k.detach().clone().requires_grad_()
    v_ref = v.detach().clone().requires_grad_()
    out_ref = attention_torch(q_ref, k_ref, v_ref, sm_scale, causal, enable_gqa)

    out_ref.backward(grad_out, retain_graph=True)
    out.backward(grad_out, retain_graph=True)

    out_snr = compute_snr(out_ref.detach(), out.detach())
    dq_snr = compute_snr(q_ref.grad, q.grad)
    dk_snr = compute_snr(k_ref.grad, k.grad)
    dv_snr = compute_snr(v_ref.grad, v.grad)

    q.grad = None
    k.grad = None
    v.grad = None

    threshold = 40
    correct = all(snr > threshold for snr in [out_snr, dq_snr, dk_snr, dv_snr])
    status = "PASS" if correct else "FAIL"
    print(
        f"Correctness Check: {status} (out={out_snr:.1f}, dq={dq_snr:.1f}, dk={dk_snr:.1f}, dv={dv_snr:.1f})"
    )

    return correct


def profile_attention(batch, seqlen, num_head_q, num_head_kv, head_dim_qk, head_dim_v, causal):
    device = "cuda"
    dtype = torch.bfloat16
    enable_gqa = num_head_q != num_head_kv

    q = torch.randn((batch, num_head_q, seqlen, head_dim_qk), device=device, dtype=dtype, requires_grad=True)
    k = torch.randn((batch, num_head_kv, seqlen, head_dim_qk), device=device, dtype=dtype, requires_grad=True)
    v = torch.randn((batch, num_head_kv, seqlen, head_dim_v), device=device, dtype=dtype, requires_grad=True)

    sm_scale = head_dim_qk ** (-0.5)

    out = attention_torch(q, k, v, sm_scale, causal, enable_gqa)
    grad_out = torch.randn_like(out)
    correct = check_attention_correctness(q, k, v, out, grad_out, sm_scale, causal, enable_gqa)

    fwd_func = lambda: attention_torch(q, k, v, sm_scale, causal, enable_gqa)
    bwd_func = lambda: out.backward(grad_out, retain_graph=True)
    out = fwd_func()
    bwd_func()

    fwd_flops = 2 * batch * seqlen * seqlen * num_head_q * (head_dim_qk + head_dim_v)
    if causal:
        fwd_flops //= 2
    bwd_flops = fwd_flops * 2.5

    for _ in range(20):
        fwd_func()
        bwd_func()
    torch.cuda.synchronize()

    fwd_time = benchmark.Timer(stmt="fn()", globals={"fn": fwd_func}).timeit(100).mean * 1e3
    bwd_time = benchmark.Timer(stmt="fn()", globals={"fn": bwd_func}).timeit(100).mean * 1e3
    fwd_tflops = fwd_flops / (fwd_time * 1e-3) / 1e12
    bwd_tflops = bwd_flops / (bwd_time * 1e-3) / 1e12

    print(f"Forward  Mean time: {fwd_time:.3f} ms | TFLOPS: {fwd_tflops:.2f}")
    print(f"Backward Mean time: {bwd_time:.3f} ms | TFLOPS: {bwd_tflops:.2f}")

    return fwd_time, fwd_tflops, bwd_time, bwd_tflops, correct


def benchmark_attention_torch(output_csv=None):
    platform, gpu_name = get_platform_info()
    test_cases = gen_attention_test_cases()

    rows = []
    test_id = 0

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
                    f"causal={causal}"
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
                        batch, seqlen, num_head_q, num_head_kv, head_dim_qk, head_dim_v, causal
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

    results = pd.DataFrame(rows)
    print("\nFinal Results:")
    print(tabulate(results, headers="keys", tablefmt="grid", showindex=False))

    avg_fwd = results["Forward TFLOPS"].astype(float).mean()
    avg_bwd = results["Backward TFLOPS"].astype(float).mean()
    print(f"\nAverage Forward TFLOPS: {avg_fwd:.2f}")
    print(f"Average Backward TFLOPS: {avg_bwd:.2f}")

    if output_csv:
        filename = output_csv
    else:
        timestamp = datetime.now().strftime("%Y%m%d")
        filename = f"attention_torch_benchmark_{timestamp}_{gpu_name}.csv"
    results.to_csv(filename, index=False)
    print(f"Results saved to {filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark PyTorch Attention (Baseline)")
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output CSV filename. Default: attention_torch_benchmark_{date}_{gpu}.csv",
    )
    args = parser.parse_args()
    benchmark_attention_torch(output_csv=args.output)
