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
        f"Correctness Check (SNR>thr={threshold} vs torch-ref): "
        f"{status} (out={out_snr:.1f}, dq={dq_snr:.1f}, dk={dk_snr:.1f}, dv={dv_snr:.1f})"
    )

    # Reset gradients
    q.grad = None
    k.grad = None
    v.grad = None

    return correct


def check_attention_determinism(fwd_func, q, k, v, grad_out):
    """Check deterministic for forward outputs and backward gradients (bitwise exact)."""
    # Forward: same input -> same output
    out1 = fwd_func()
    out2 = fwd_func()
    torch.cuda.synchronize()

    out_ok = torch.equal(out1, out2)
    out_max_abs_diff = (out1 - out2).abs().max().item()

    # Backward: same input + same grad_out -> same grads
    q.grad = None
    k.grad = None
    v.grad = None

    out1_bwd = fwd_func()
    out1_bwd.backward(grad_out, retain_graph=False)
    dq1 = q.grad.detach().clone()
    dk1 = k.grad.detach().clone()
    dv1 = v.grad.detach().clone()

    q.grad = None
    k.grad = None
    v.grad = None

    out2_bwd = fwd_func()
    out2_bwd.backward(grad_out, retain_graph=False)
    dq2 = q.grad.detach().clone()
    dk2 = k.grad.detach().clone()
    dv2 = v.grad.detach().clone()
    torch.cuda.synchronize()

    dq_ok = torch.equal(dq1, dq2)
    dk_ok = torch.equal(dk1, dk2)
    dv_ok = torch.equal(dv1, dv2)

    dq_max_abs_diff = (dq1 - dq2).abs().max().item()
    dk_max_abs_diff = (dk1 - dk2).abs().max().item()
    dv_max_abs_diff = (dv1 - dv2).abs().max().item()

    # Reset gradients
    q.grad = None
    k.grad = None
    v.grad = None

    return (
        out_ok,
        out_max_abs_diff,
        dq_ok,
        dq_max_abs_diff,
        dk_ok,
        dk_max_abs_diff,
        dv_ok,
        dv_max_abs_diff,
    )


def profile_attention(
    batch,
    seqlen,
    num_head_q,
    num_head_kv,
    head_dim_qk,
    head_dim_v,
    causal,
    use_fp8,
    deterministic,
):
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
            deterministic=deterministic,
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
            deterministic=deterministic,
            return_lse=False,
            return_attn_probs=False,
        )

    # Fixed grad_out for deterministic / correctness checks
    out_for_grad = fwd_func()
    grad_out = torch.randn_like(out_for_grad)

    det_out_ok = None
    det_out_max_abs_diff = None
    det_dq_ok = None
    det_dq_max_abs_diff = None
    det_dk_ok = None
    det_dk_max_abs_diff = None
    det_dv_ok = None
    det_dv_max_abs_diff = None
    if deterministic:
        (
            det_out_ok,
            det_out_max_abs_diff,
            det_dq_ok,
            det_dq_max_abs_diff,
            det_dk_ok,
            det_dk_max_abs_diff,
            det_dv_ok,
            det_dv_max_abs_diff,
        ) = check_attention_determinism(fwd_func, q, k, v, grad_out)

    # Forward pass and correctness check
    out = fwd_func()
    correct = check_attention_correctness(q, k, v, q_ref, k_ref, v_ref, out, o_ref, grad_out, use_fp8)

    # Print deterministic status only when deterministic is enabled
    if deterministic:
        determinism_ok = bool(det_out_ok) and bool(det_dq_ok) and bool(det_dk_ok) and bool(det_dv_ok)
        status = "PASS" if determinism_ok else "FAIL"
        print(
            "Deterministic Check (bitwise; max_abs_diff): "
            f"{status} (out={det_out_max_abs_diff}, dq={det_dq_max_abs_diff}, "
            f"dk={det_dk_max_abs_diff}, dv={det_dv_max_abs_diff})"
        )

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

    return (
        fwd_time,
        fwd_tflops,
        bwd_time,
        bwd_tflops,
        correct,
        det_out_ok,
        det_out_max_abs_diff,
        det_dq_ok,
        det_dq_max_abs_diff,
        det_dk_ok,
        det_dk_max_abs_diff,
        det_dv_ok,
        det_dv_max_abs_diff,
    )


def benchmark_attention(output_csv=None, use_fp8=False, deterministic=False):
    """Run attention benchmark."""
    platform, gpu_name = get_platform_info()

    # Generate test cases from config
    test_cases = gen_attention_test_cases()

    rows = []
    test_id = 0
    total_tests = 2 * len(BATCH_SIZE_LIST) * len(test_cases)
    print(f"Total tests: {total_tests}, FP8: {use_fp8}, deterministic: {deterministic}")

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
                    f"causal={causal}, fp8={use_fp8}, deterministic={deterministic}"
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
                    "Deterministic": deterministic,
                }

                try:
                    (
                        fwd_time,
                        fwd_tflops,
                        bwd_time,
                        bwd_tflops,
                        correct,
                        det_out_ok,
                        det_out_max_abs_diff,
                        det_dq_ok,
                        det_dq_max_abs_diff,
                        det_dk_ok,
                        det_dk_max_abs_diff,
                        det_dv_ok,
                        det_dv_max_abs_diff,
                    ) = profile_attention(
                        batch,
                        seqlen,
                        num_head_q,
                        num_head_kv,
                        head_dim_qk,
                        head_dim_v,
                        causal,
                        use_fp8,
                        deterministic,
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
                    if deterministic:
                        row["Deterministic Check"] = (
                            "PASS" if (det_out_ok and det_dq_ok and det_dk_ok and det_dv_ok) else "FAIL"
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
                    if deterministic:
                        row["Deterministic Check"] = "ERROR"

                rows.append(row)

    # Create DataFrame
    results = pd.DataFrame(rows)
    if deterministic and "Check" in results.columns and "Deterministic Check" in results.columns:
        cols = list(results.columns)
        cols.insert(cols.index("Check") + 1, cols.pop(cols.index("Deterministic Check")))
        results = results[cols]

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
        if deterministic:
            prefix = (
                "attention_deterministic_fp8_benchmark_result"
                if use_fp8
                else "attention_deterministic_benchmark_result"
            )
        else:
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
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Enable deterministic kernel mode (default: disabled).",
    )
    args = parser.parse_args()
    benchmark_attention(output_csv=args.output, use_fp8=args.fp8, deterministic=args.deterministic)
