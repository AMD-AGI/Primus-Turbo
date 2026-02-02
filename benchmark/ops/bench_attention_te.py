###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Benchmark Transformer Engine attention performance.

This script intentionally aligns with benchmark/ops/bench_attention_turbo.py:
- Same test case generator (config.gen_attention_test_cases)
- Same FLOPs model and TFLOPS reporting
- Same correctness check vs PyTorch SDPA reference (SNR)

Installation notes:
- NVIDIA TE: https://github.com/NVIDIA/TransformerEngine
- AMD TE: https://github.com/ROCm/TransformerEngine
"""

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

_platform, _ = get_platform_info()
if _platform == "CUDA":
    from transformer_engine.pytorch import DotProductAttention
else:
    from transformer_engine.pytorch.attention.dot_product_attention import (
        DotProductAttention,
    )

ATTENTION_ENV_VARS = [
    "NVTE_FUSED_ATTN",  # supports both NV and AMD
    "NVTE_FLASH_ATTN",
    "NVTE_FUSED_ATTN_AOTRITON",
    "NVTE_FUSED_ATTN_CK",
    "NVTE_UNFUSED_ATTN",
    "NVTE_CK_USES_BWD_V3",
    "NVTE_CK_USES_FWD_V3",
    "NVTE_CK_IS_V3_ATOMIC_FP32",
]


def cleanup_env():
    for var in ATTENTION_ENV_VARS:
        os.environ[var] = "0"


def setup_backend_env():
    cleanup_env()

    os.environ["NVTE_FUSED_ATTN"] = "1"
    os.environ["NVTE_FUSED_ATTN_CK"] = "1"
    os.environ["NVTE_CK_USES_BWD_V3"] = "1"
    os.environ["NVTE_CK_IS_V3_ATOMIC_FP32"] = "0"
    os.environ["NVTE_CK_USES_FWD_V3"] = "1"


def _is_gfx950():
    props = torch.cuda.get_device_properties(0)
    return props.major == 9 and props.minor == 5


if _is_gfx950():
    os.environ["PRIMUS_TURBO_ATTN_V3_ATOMIC_FP32"] = "0"

ATTN_BACKENDS = [
    SDPBackend.FLASH_ATTENTION,
    SDPBackend.EFFICIENT_ATTENTION,
    SDPBackend.MATH,
]


def attention_ref(q, k, v, sm_scale, causal):
    num_heads = q.shape[2]
    n_kv_heads = k.shape[2]
    n_rep = num_heads // n_kv_heads

    q = q.transpose(1, 2).contiguous()
    k = k.transpose(1, 2).contiguous()
    v = v.transpose(1, 2).contiguous()

    with sdpa_kernel(ATTN_BACKENDS):
        o_ref = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, is_causal=causal, scale=sm_scale, enable_gqa=n_rep > 1
        )
    # BHSD -> BSHD. for TE, the output is in BHSD format.
    return o_ref.transpose(1, 2).reshape(q.shape[0], q.shape[2], -1)


def check_attention_correctness(q, k, v, q_ref, k_ref, v_ref, o, o_ref, grad_out):
    o_ref.backward(grad_out, retain_graph=True)
    o.backward(grad_out, retain_graph=True)

    out_snr = compute_snr(o_ref, o)
    dq_snr = compute_snr(q_ref.grad, q.grad)
    dk_snr = compute_snr(k_ref.grad, k.grad)
    dv_snr = compute_snr(v_ref.grad, v.grad)

    threshold = 40

    correct = all(snr > threshold for snr in [out_snr, dq_snr, dk_snr, dv_snr])
    status = "PASS" if correct else "FAIL"
    print(
        f"Correctness Check (SNR>thr={threshold} vs torch-ref): "
        f"{status} (out={out_snr:.1f}, dq={dq_snr:.1f}, dk={dk_snr:.1f}, dv={dv_snr:.1f})"
    )

    q.grad = None
    k.grad = None
    v.grad = None

    return correct


def check_attention_determinism(fwd_func, q, k, v, grad_out):
    out1 = fwd_func()
    out2 = fwd_func()
    torch.cuda.synchronize()

    out_ok = torch.equal(out1, out2)
    out_max_abs_diff = (out1 - out2).abs().max().item()

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
    deterministic,
):
    device = "cuda"
    dtype = torch.bfloat16

    q = torch.randn(
        (batch, seqlen, num_head_q, head_dim_qk),
        device=device,
        dtype=dtype,
        requires_grad=True,
    )
    k = torch.randn(
        (batch, seqlen, num_head_kv, head_dim_qk),
        device=device,
        dtype=dtype,
        requires_grad=True,
    )
    v = torch.randn(
        (batch, seqlen, num_head_kv, head_dim_v),
        device=device,
        dtype=dtype,
        requires_grad=True,
    )
    q_ref = q.clone().detach().requires_grad_()
    k_ref = k.clone().detach().requires_grad_()
    v_ref = v.clone().detach().requires_grad_()

    sm_scale = head_dim_qk ** (-0.5)
    attn_mask_type = "causal" if causal else "no_mask"

    o_ref = attention_ref(q_ref, k_ref, v_ref, sm_scale, causal)

    fwd_func = DotProductAttention(
        num_head_q,
        (head_dim_qk, head_dim_v),
        num_gqa_groups=num_head_kv,
        attention_dropout=0.0,
        qkv_format="bshd",
        attn_mask_type=attn_mask_type,
        sequence_parallel=False,
        tp_size=1,
        tp_group=None,
        layer_number=1,
        attention_type="self",
        softmax_scale=sm_scale,
    ).to(dtype=dtype, device=device)

    fwd_call = lambda: fwd_func(
        q,
        k,
        v,
        qkv_format="bshd",
        max_seqlen_q=seqlen,
        max_seqlen_kv=seqlen,
        attn_mask_type=attn_mask_type,
        window_size=None,
    )

    out_for_grad = fwd_call()
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
        ) = check_attention_determinism(fwd_call, q, k, v, grad_out)

    out = fwd_call()
    correct = check_attention_correctness(q, k, v, q_ref, k_ref, v_ref, out, o_ref, grad_out)

    if deterministic:
        determinism_ok = bool(det_out_ok) and bool(det_dq_ok) and bool(det_dk_ok) and bool(det_dv_ok)
        status = "PASS" if determinism_ok else "FAIL"
        print(
            "Deterministic Check (bitwise; max_abs_diff): "
            f"{status} (out={det_out_max_abs_diff}, dq={det_dq_max_abs_diff}, "
            f"dk={det_dk_max_abs_diff}, dv={det_dv_max_abs_diff})"
        )

    out = fwd_call()
    out.backward(grad_out, retain_graph=True)

    fwd_flops = 2 * batch * seqlen * seqlen * num_head_q * (head_dim_qk + head_dim_v)
    if causal:
        fwd_flops //= 2
    bwd_flops = fwd_flops * 2.5

    for _ in range(20):
        out = fwd_call()
        out.backward(grad_out, retain_graph=True)
    torch.cuda.synchronize()

    fwd_time = (
        benchmark.Timer(
            stmt=(
                "fwd_func(q, k, v, qkv_format='bshd', max_seqlen_q=seqlen, "
                "max_seqlen_kv=seqlen, attn_mask_type=attn_mask_type, window_size=None)"
            ),
            globals={
                "fwd_func": fwd_func,
                "q": q,
                "k": k,
                "v": v,
                "seqlen": seqlen,
                "attn_mask_type": attn_mask_type,
            },
        )
        .timeit(100)
        .mean
        * 1e3
    )
    bwd_time = (
        benchmark.Timer(
            stmt=(
                "out = fwd_func(q, k, v, qkv_format='bshd', max_seqlen_q=seqlen, "
                "max_seqlen_kv=seqlen, attn_mask_type=attn_mask_type, window_size=None); "
                "out.backward(grad_out, retain_graph=True)"
            ),
            globals={
                "fwd_func": fwd_func,
                "q": q,
                "k": k,
                "v": v,
                "seqlen": seqlen,
                "attn_mask_type": attn_mask_type,
                "grad_out": grad_out,
            },
        )
        .timeit(100)
        .mean
        * 1e3
    )
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


def benchmark_attention(output_csv=None, deterministic=False):
    platform, gpu_name = get_platform_info()

    test_cases = gen_attention_test_cases()

    rows = []
    test_id = 0
    total_tests = 2 * len(BATCH_SIZE_LIST) * len(test_cases)
    print(f"Total tests: {total_tests}, ck: v3, deterministic: {deterministic}")

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
                    f"causal={causal}, ck=v3, deterministic={deterministic}"
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

    results = pd.DataFrame(rows)
    if deterministic and "Check" in results.columns and "Deterministic Check" in results.columns:
        cols = list(results.columns)
        cols.insert(cols.index("Check") + 1, cols.pop(cols.index("Deterministic Check")))
        results = results[cols]

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
        if deterministic:
            prefix = "attention_deterministic_benchmark_result_ck_v3"
        else:
            prefix = "attention_benchmark_result_ck_v3"
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
        help="Output CSV filename. Default: attention_benchmark_result_ck_v3_{date}_{gpu}.csv",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Enable deterministic kernel mode (default: disabled).",
    )
    args = parser.parse_args()
    setup_backend_env()
    benchmark_attention(output_csv=args.output, deterministic=args.deterministic)
