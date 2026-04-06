"""Benchmark Triton FP8 Attention for decode (small seqlen_q) workloads.

Measures the effect of BLOCK_N=128 vs BLOCK_N=64 for decode-like shapes
where seqlen_q is very small (1-64) and seqlen_k is large (cache).
"""

import json
import time

import torch

from primus_turbo.pytorch.ops.attention import flash_attn_fp8_func

CASES = [
    # (batch, seqlen_q, seqlen_k, nheads_q, nheads_kv, head_dim, desc)
    (1, 1, 2048, 32, 32, 128, "decode B=1 cache=2K"),
    (1, 1, 4096, 32, 32, 128, "decode B=1 cache=4K"),
    (1, 1, 8192, 32, 32, 128, "decode B=1 cache=8K"),
    (4, 1, 2048, 32, 32, 128, "decode B=4 cache=2K"),
    (4, 1, 4096, 32, 32, 128, "decode B=4 cache=4K"),
    (4, 1, 8192, 32, 32, 128, "decode B=4 cache=8K"),
    (8, 1, 4096, 32, 32, 128, "decode B=8 cache=4K"),
    (16, 1, 4096, 32, 32, 128, "decode B=16 cache=4K"),
    # GQA decode
    (4, 1, 4096, 32, 8, 128, "decode GQA B=4"),
    (4, 1, 4096, 64, 8, 128, "decode GQA B=4 64h"),
    # Short prefill (seqlen_q=4..64)
    (4, 4, 4096, 32, 32, 128, "short-pf sq=4"),
    (4, 16, 4096, 32, 32, 128, "short-pf sq=16"),
    (4, 32, 4096, 32, 32, 128, "short-pf sq=32"),
    (4, 64, 4096, 32, 32, 128, "short-pf sq=64"),
    # Reference: large prefill (unchanged path)
    (4, 256, 4096, 32, 32, 128, "prefill sq=256 (ref)"),
    (4, 1024, 4096, 32, 32, 128, "prefill sq=1024 (ref)"),
]


def bench_one(batch, seqlen_q, seqlen_k, nheads_q, nheads_kv, head_dim, warmup=10, iters=50):
    device = "cuda:0"
    q = torch.randn(batch, seqlen_q, nheads_q, head_dim, device=device, dtype=torch.bfloat16)
    k = torch.randn(batch, seqlen_k, nheads_kv, head_dim, device=device, dtype=torch.bfloat16)
    v = torch.randn(batch, seqlen_k, nheads_kv, head_dim, device=device, dtype=torch.bfloat16)

    for _ in range(warmup):
        _ = flash_attn_fp8_func(q, k, v, causal=True)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(iters):
        _ = flash_attn_fp8_func(q, k, v, causal=True)
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / iters

    flops = 4 * batch * nheads_q * seqlen_q * seqlen_k * head_dim
    tflops = flops / elapsed / 1e12
    ms = elapsed * 1000
    return ms, tflops


def main():
    results = []
    for batch, sq, sk, nhq, nhkv, hd, desc in CASES:
        ms, tflops = bench_one(batch, sq, sk, nhq, nhkv, hd)
        entry = {
            "batch": batch,
            "seqlen_q": sq,
            "seqlen_k": sk,
            "nheads_q": nhq,
            "nheads_kv": nhkv,
            "head_dim": hd,
            "desc": desc,
            "ms": round(ms, 4),
            "tflops": round(tflops, 2),
        }
        results.append(entry)
        print(
            f"B={batch:2d} sq={sq:5d} sk={sk:5d} nhq={nhq:2d} nhkv={nhkv:2d} hd={hd:3d} "
            f"| {ms:8.4f} ms  {tflops:7.2f} TFLOPS | {desc}"
        )

    with open("benchmark/baselines/attention_decode_fp8.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {len(results)} results")


if __name__ == "__main__":
    main()
