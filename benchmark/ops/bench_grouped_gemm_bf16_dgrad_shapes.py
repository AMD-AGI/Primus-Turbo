#!/usr/bin/env python3
"""Microbench grouped NN dgrad at explicit M×K×N shapes.

Compares:
  * grouped_gemm_bf16_only (bench_mega_moe baseline)
  * grouped_gemm_bf16_dgrad_flydsl_kernel (new standalone module)

Run (single GPU is enough for GEMM-only):
  PYTHONPATH=/perf_apps/xiaoming/MegaMoE python benchmark/ops/bench_grouped_gemm_bf16_dgrad_shapes.py
"""

from __future__ import annotations

import argparse
import sys
import time

import torch

# bench_mega_moe helpers live under benchmark/ops/training/
sys.path.insert(0, "benchmark/ops/training")
from bench_mega_moe import grouped_gemm_bf16_only  # noqa: E402

from primus_turbo.flydsl.grouped_gemm.grouped_gemm_bf16_dgrad_kernel import (  # noqa: E402
    grouped_gemm_bf16_dgrad_flydsl_kernel,
)

SHAPES = (
    (16384, 2048, 768),
    (16384, 768, 2048),
    (131072, 2048, 768),
)


def _bench_ms(fn, *, warmup: int, iters: int) -> float:
    torch.cuda.synchronize()
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return float(s.elapsed_time(e) / iters)


def _run_shape(M: int, K: int, N: int, *, G: int, bm: int, bn: int, group_m: int, warmup: int, iters: int):
    assert M % bm == 0 and N % bn == 0
    n_mblk = M // bm
    device = "cuda"
    pool = torch.randn(M, K, device=device, dtype=torch.bfloat16)
    weight = torch.randn(G, K, N, device=device, dtype=torch.bfloat16)
    tile_to_expert = (torch.arange(n_mblk, device=device, dtype=torch.int32) % G).contiguous()
    num_tile_blocks = torch.tensor([n_mblk], device=device, dtype=torch.int32)
    out_base = torch.empty(M, N, device=device, dtype=torch.bfloat16)
    out_new = torch.empty(M, N, device=device, dtype=torch.bfloat16)

    def _baseline():
        grouped_gemm_bf16_only(
            pool,
            weight,
            out_base,
            tile_to_expert,
            num_tile_blocks,
            layout="nn",
            BLOCK_M=bm,
            BLOCK_N=bn,
            GROUP_M=group_m,
            num_xcd=1,
        )

    def _new():
        nonlocal out_new
        out_new = grouped_gemm_bf16_dgrad_flydsl_kernel(
            pool,
            weight,
            tile_to_expert,
            num_tile_blocks,
            BLOCK_M=bm,
            BLOCK_N=bn,
            GROUP_M=group_m,
        )

    # warmup compiles
    _baseline()
    _new()
    ref = out_base.clone()
    snr = float(
        10.0
        * torch.log10(
            ref.float().pow(2).sum()
            / ((ref.float() - out_new.float()).pow(2).sum() + 1e-12)
        )
    )

    t_base = _bench_ms(_baseline, warmup=warmup, iters=iters)
    t_new = _bench_ms(_new, warmup=warmup, iters=iters)
    flops = 2.0 * M * K * N
    tf_base = flops / (t_base * 1e-3) / 1e12
    tf_new = flops / (t_new * 1e-3) / 1e12
    ratio = tf_new / tf_base if tf_base > 0 else float("nan")
    return {
        "M": M,
        "K": K,
        "N": N,
        "tf_base": tf_base,
        "tf_new": tf_new,
        "t_base": t_base,
        "t_new": t_new,
        "ratio": ratio,
        "snr": snr,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experts", type=int, default=32)
    ap.add_argument("--bm", type=int, default=256)
    ap.add_argument("--bn", type=int, default=256)
    ap.add_argument("--group-m", type=int, default=4)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=20)
    args = ap.parse_args()

    print(f"device={torch.cuda.get_device_name()}  G={args.experts}  BM={args.bm} BN={args.bn} GROUP_M={args.group_m}")
    print(f"{'shape (M×K×N)':<22} {'baseline':>10} {'new':>10} {'new/base':>10} {'SNR dB':>10}")
    print("-" * 66)
    for M, K, N in SHAPES:
        r = _run_shape(
            M,
            K,
            N,
            G=args.experts,
            bm=args.bm,
            bn=args.bn,
            group_m=args.group_m,
            warmup=args.warmup,
            iters=args.iters,
        )
        shape = f"{M}×{K}×{N}"
        print(
            f"{shape:<22} {r['tf_base']:10.1f} {r['tf_new']:10.1f} {r['ratio']:10.2f}× {r['snr']:10.1f}"
        )


if __name__ == "__main__":
    main()
