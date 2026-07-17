###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import argparse
import math
from datetime import datetime

import pandas as pd
import torch
import torch.utils.benchmark as benchmark
from config import compute_snr, get_platform_info
from tabulate import tabulate

# Import the package first so it fully initializes before the triton kernel modules pull in
# triton_knobs_helper (avoids a partial-init circular import via primus_turbo.pytorch.core.utils).
import primus_turbo.pytorch  # noqa: F401  (import-ordering side effect)
from primus_turbo.flydsl.attention.sparse_mla_bwd import sparse_mla_bwd_v4_flydsl
from primus_turbo.flydsl.attention.sparse_mla_fwd import sparse_mla_fwd_v4_flydsl
from primus_turbo.triton.attention.sparse_mla import (
    sparse_mla_bwd_v4_triton,
    sparse_mla_fwd_v4_triton,
)

# DeepSeek-V4 single-latent sparse-MLA fixed dims (kv_lora_rank + rope pad).
ROPE_DIM = 64
HEAD_DIM = 512
SWA_WINDOW = 128

# (variant, num_heads, index-topk cap). cr in {0, 4, 128} spans pure-SWA / random-pool /
# deterministic-pool (HCA) at seqlen S: topk = SWA + selected-pool ranks.
VARIANTS = {"flash": dict(num_heads=64, index_topk=512), "pro": dict(num_heads=128, index_topk=1024)}
COMPRESS_RATIOS = [0, 4, 128]

FWD_SNR_THRESHOLD = 40.0
BWD_SNR_THRESHOLD = 35.0


def sparse_mla_topk(variant, cr, seqlen):
    """Selected-rank count (SWA window + pool) for a (variant, compress-ratio) shape."""
    if cr == 0:
        return 0, 0, SWA_WINDOW
    if cr == 4:
        pool = max(seqlen // 4, 1)
        topk_pool = min(VARIANTS[variant]["index_topk"], pool)
        return pool, topk_pool, SWA_WINDOW + topk_pool
    pool = max(seqlen // cr, 1)
    return pool, 0, SWA_WINDOW + pool


def build_sparse_mla_inputs(cr, num_heads, seqlen, pool, topk_pool, seed=0):
    """Build the DSV4 sparse-MLA inputs: single-latent kv, per-token top-k indices (SWA band +
    optional pool), zero-padded rope cols, and a random sink / grad_out."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    dev, dt = "cuda", torch.bfloat16
    latent = torch.randn(seqlen, HEAD_DIM, generator=gen, device=dev, dtype=dt)
    q = torch.randn(seqlen, num_heads, HEAD_DIM, generator=gen, device=dev, dtype=dt)
    q = torch.cat([q, torch.zeros(seqlen, num_heads, ROPE_DIM, device=dev, dtype=dt)], -1).contiguous()
    sink = torch.randn(num_heads, generator=gen, device=dev, dtype=torch.float32) * 0.1
    grad_out = torch.randn(seqlen, num_heads, HEAD_DIM, generator=gen, device=dev, dtype=dt)

    tok = torch.arange(seqlen, device=dev).view(seqlen, 1)
    win = tok - SWA_WINDOW + 1 + torch.arange(SWA_WINDOW, device=dev).view(1, SWA_WINDOW)
    win = torch.where(win >= 0, win, torch.full_like(win, -1))
    if cr == 0:
        kv = latent.unsqueeze(1)
        topk = win
    else:
        p = torch.randn(pool, HEAD_DIM, generator=gen, device=dev, dtype=dt)
        kv = torch.cat([latent, p], 0).unsqueeze(1)
        if cr == 4:
            pool_topk = seqlen + torch.randint(0, pool, (seqlen, topk_pool), generator=gen, device=dev)
        else:
            ps = torch.arange(pool, device=dev).view(1, pool)
            pool_topk = torch.where(
                ((ps + 1) * cr - 1) <= tok, seqlen + ps, torch.full_like(ps.expand(seqlen, pool), -1)
            )
        topk = torch.cat([win, pool_topk], 1)
    pad = ((topk.shape[1] + 63) // 64) * 64 - topk.shape[1]
    if pad > 0:
        topk = torch.cat([topk, torch.full((seqlen, pad), -1, device=dev, dtype=topk.dtype)], 1)
    kv = torch.cat([kv, torch.zeros(kv.shape[0], 1, ROPE_DIM, device=dev, dtype=dt)], -1).contiguous()
    return q, kv, topk.to(torch.int32).contiguous(), sink, grad_out


def profile_sparse_mla(variant, cr, seqlen):
    """Profile flydsl sparse-MLA fwd/bwd; SNR is checked against the triton oracle."""
    num_heads = VARIANTS[variant]["num_heads"]
    pool, topk_pool, topk = sparse_mla_topk(variant, cr, seqlen)
    scale = 1.0 / math.sqrt(HEAD_DIM)
    q, kv, topk_idx, sink, grad_out = build_sparse_mla_inputs(cr, num_heads, seqlen, pool, topk_pool)

    fwd_func = lambda: sparse_mla_fwd_v4_flydsl(
        q, kv, topk_idx, attn_sink=sink, kv_lora_rank=HEAD_DIM, scale=scale
    )
    out, lse = fwd_func()
    out_ref, lse_ref = sparse_mla_fwd_v4_triton(
        q, kv, topk_idx, attn_sink=sink, kv_lora_rank=HEAD_DIM, scale=scale
    )
    fwd_snr = compute_snr(out_ref, out)

    bwd_func = lambda: sparse_mla_bwd_v4_flydsl(
        q, kv, out, grad_out, topk_idx, lse, attn_sink=sink, kv_lora_rank=HEAD_DIM, scale=scale
    )
    dq, dkv, dsink = bwd_func()
    dq_ref, dkv_ref, dsink_ref = sparse_mla_bwd_v4_triton(
        q, kv, out_ref, grad_out, topk_idx, lse_ref, attn_sink=sink, kv_lora_rank=HEAD_DIM, scale=scale
    )
    bwd_snr = min(compute_snr(dq_ref, dq), compute_snr(dkv_ref, dkv))
    if dsink is not None and dsink_ref is not None:
        bwd_snr = min(bwd_snr, compute_snr(dsink_ref, dsink))

    # QK + PV GEMMs over the selected topk ranks (both HEAD_DIM-wide); bwd ~2.5x fwd.
    fwd_flops = 2 * seqlen * num_heads * topk * (HEAD_DIM + HEAD_DIM)
    bwd_flops = fwd_flops * 2.5

    for _ in range(20):
        fwd_func()
        bwd_func()
    torch.cuda.synchronize()

    fwd_time = benchmark.Timer(stmt="fn()", globals={"fn": fwd_func}).timeit(100).mean * 1e3
    bwd_time = benchmark.Timer(stmt="fn()", globals={"fn": bwd_func}).timeit(100).mean * 1e3
    fwd_tflops = fwd_flops / (fwd_time * 1e-3) / 1e12
    bwd_tflops = bwd_flops / (bwd_time * 1e-3) / 1e12

    correct = fwd_snr > FWD_SNR_THRESHOLD and bwd_snr > BWD_SNR_THRESHOLD
    print(
        f"Correctness (SNR vs triton): {'PASS' if correct else 'FAIL'} (fwd={fwd_snr:.1f}, bwd={bwd_snr:.1f})"
    )
    print(f"Forward  Mean time: {fwd_time:.3f} ms | TFLOPS: {fwd_tflops:.2f}")
    print(f"Backward Mean time: {bwd_time:.3f} ms | TFLOPS: {bwd_tflops:.2f}")
    return fwd_time, fwd_tflops, fwd_snr, bwd_time, bwd_tflops, bwd_snr, correct


def benchmark_sparse_mla(output_csv=None, seqlen=4096):
    """Run the DSV4 sparse-MLA benchmark over the 6 (variant x compress-ratio) shapes."""
    platform, gpu_name = get_platform_info()

    rows = []
    test_id = 0
    total_tests = len(VARIANTS) * len(COMPRESS_RATIOS)
    print(f"Total tests: {total_tests}, seqlen: {seqlen}")

    for variant in VARIANTS:
        for cr in COMPRESS_RATIOS:
            test_id += 1
            num_heads = VARIANTS[variant]["num_heads"]
            _, _, topk = sparse_mla_topk(variant, cr, seqlen)
            print(f"\n{'=' * 60}")
            print(
                f"TestID: {test_id}, variant={variant}, cr={cr}, seqlen={seqlen}, heads={num_heads}, topk={topk}"
            )
            print(f"{'=' * 60}")

            row = {
                "TestID": test_id,
                "Platform": platform,
                "GPU": gpu_name,
                "Variant": variant,
                "CompressRatio": cr,
                "SeqLen": seqlen,
                "num_heads": num_heads,
                "topk": topk,
            }
            try:
                fwd_time, fwd_tflops, fwd_snr, bwd_time, bwd_tflops, bwd_snr, correct = profile_sparse_mla(
                    variant, cr, seqlen
                )
                row.update(
                    {
                        "Check": "PASS" if correct else "FAIL",
                        "Forward Time (ms)": f"{fwd_time:.2f}",
                        "Forward TFLOPS": f"{fwd_tflops:.2f}",
                        "Forward SNR": f"{fwd_snr:.1f}",
                        "Backward Time (ms)": f"{bwd_time:.2f}",
                        "Backward TFLOPS": f"{bwd_tflops:.2f}",
                        "Backward SNR": f"{bwd_snr:.1f}",
                    }
                )
            except Exception as e:
                print(f"Failed: {str(e)}")
                row.update(
                    {
                        "Check": "ERROR",
                        "Forward Time (ms)": "ERROR",
                        "Forward TFLOPS": "0.00",
                        "Forward SNR": "0.0",
                        "Backward Time (ms)": "ERROR",
                        "Backward TFLOPS": "0.00",
                        "Backward SNR": "0.0",
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
        filename = f"sparse_mla_benchmark_result_{timestamp}_{gpu_name}.csv"
    results.to_csv(filename, index=False)
    print(f"Results saved to {filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark DeepSeek-V4 sparse-MLA attention (flydsl)")
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output CSV filename. Default: sparse_mla_benchmark_result_{date}_{gpu}.csv",
    )
    parser.add_argument("--seqlen", type=int, default=4096, help="Sequence length (default: 4096)")
    args = parser.parse_args()
    benchmark_sparse_mla(output_csv=args.output, seqlen=args.seqlen)
