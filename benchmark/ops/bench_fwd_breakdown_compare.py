#!/usr/bin/env python3
"""fp8 vs bf16 mega MoE forward per-stage breakdown (EP8, DSv3 shapes).

Times analogous legs side-by-side on the same routing/weights:
  L1 dispatch+fc1 | SwiGLU (+ fp8 quant) | L2 combine | FULL op

Run (8 GPUs):
  PYTHONPATH=<repo> python benchmark/ops/bench_fwd_breakdown_compare.py \\
    --num-processes 8 --num-tokens 8192 --warmup 8 --iters 25
"""

import argparse
import datetime
import math
import os

import numpy as np
import torch
import torch.distributed as dist

import primus_turbo.pytorch  # noqa: F401
from primus_turbo.flydsl.mega import (
    dispatch_grouped_gemm_bf16_flydsl_kernel,
    grouped_gemm_combine_bf16_flydsl_kernel,
    swiglu_flydsl_kernel,
)
from primus_turbo.flydsl.mega.fp8 import (
    dispatch_grouped_gemm_mxfp8_flydsl_kernel,
    grouped_gemm_combine_mxfp8_flydsl_kernel,
    swiglu_mxfp8_flydsl_kernel,
)
from primus_turbo.pytorch.kernels.fused_mega_moe.fused_mega_moe_weight_prep_fp8 import prepare_w2_fp8
from primus_turbo.pytorch.kernels.fused_mega_moe.fused_mega_moe_weight_prep_fp8 import (
    _w1_fp8_cached,
    _w2_fp8_cached,
)
from primus_turbo.pytorch.ops.moe.fused_mega_moe import fused_mega_moe
from primus_turbo.pytorch.ops.moe.fused_mega_moe_fp8 import fused_mega_moe_fp8

_H_BF16_NUM_TILE_BLOCKS = 8


def _bench(fn, *, warmup, iters, group):
    for _ in range(warmup):
        torch.cuda.synchronize()
        group.barrier()
        fn()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        torch.cuda.synchronize()
        group.barrier()
        starts[i].record()
        fn()
        ends[i].record()
    torch.cuda.synchronize()
    ms = float(np.mean([s.elapsed_time(e) for s, e in zip(starts, ends)][1:]))
    t = torch.tensor([ms], device="cuda")
    dist.all_reduce(t, op=dist.ReduceOp.MAX, group=group)
    return float(t)


def worker(local_rank, world, args):
    ip = os.getenv("MASTER_ADDR", "127.0.0.1")
    port = int(os.getenv("MASTER_PORT", "8495"))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        "nccl",
        init_method=f"tcp://{ip}:{port}",
        world_size=world,
        rank=local_rank,
        timeout=datetime.timedelta(seconds=int(os.getenv("MEGA_BENCH_TIMEOUT_S", "600"))),
    )
    group = dist.new_group(list(range(world)))
    rank = dist.get_rank()

    H, I, E, K, T = args.hidden, args.inter, args.num_experts, args.num_topk, args.num_tokens
    BM, BN = args.bm, args.bn
    CC, DC, PC = args.combine_cu, args.dispatch_cu, args.preshuffle_cu
    epr = E // world

    torch.manual_seed(7 + rank)
    x = torch.randn((T, H), device="cuda", dtype=torch.bfloat16)
    g = torch.Generator(device="cuda").manual_seed(100 + rank)
    scores = torch.rand(T, E, generator=g, device="cuda").abs() + 1
    topk_w, topk_idx = torch.topk(scores.softmax(-1), K, dim=-1)
    topk_w, topk_idx = topk_w.float(), topk_idx.long()
    W1g = torch.randn((E, 2 * I, H), device="cuda", dtype=torch.bfloat16) * (2.0 / math.sqrt(H))
    W2g = torch.randn((E, H, I), device="cuda", dtype=torch.bfloat16) * (2.0 / math.sqrt(I))
    W1 = W1g[rank * epr : (rank + 1) * epr].contiguous()
    W2 = W2g[rank * epr : (rank + 1) * epr].contiguous()

    w1_fp8 = _w1_fp8_cached(W1)
    w2_fp8 = prepare_w2_fp8(W2)

    # ── fp8 path setup ──
    def _fp8_l1():
        return dispatch_grouped_gemm_mxfp8_flydsl_kernel(
            x, w1_fp8, group, topk_idx=topk_idx, topk_weights=topk_w,
            num_dispatch_cu=DC, num_preshuffle_cu=PC, BM=BM, BN=BN,
        )

    l1_fp8, handle_fp8, _, _ = _fp8_l1()
    ntb_fp8 = handle_fp8[11]
    act_fp8, act_a_sp = swiglu_mxfp8_flydsl_kernel(l1_fp8, ntb_fp8)
    torch.cuda.synchronize()
    group.barrier()

    # ── bf16 path setup (same x / routing / weights) ──
    def _bf16_l1_once():
        return dispatch_grouped_gemm_bf16_flydsl_kernel(
            x, W1, group, handle=None, topk_idx=topk_idx, topk_weights=topk_w,
            layout="nt", BM=BM, BN=BN,
        )

    l1_bf16, _, _, handle_bf16 = _bf16_l1_once()
    ntb_bf16 = handle_bf16[_H_BF16_NUM_TILE_BLOCKS]
    act_bf16 = swiglu_flydsl_kernel(l1_bf16, num_tile_blocks=ntb_bf16)
    topk_idx32 = topk_idx.to(torch.int32).contiguous().view(-1)
    topk_w_flat = topk_w.contiguous().view(-1)
    torch.cuda.synchronize()
    group.barrier()

    fp8, bf16 = {}, {}

    fp8["L1"] = _bench(_fp8_l1, warmup=args.warmup, iters=args.iters, group=group)
    bf16["L1"] = _bench(
        lambda: dispatch_grouped_gemm_bf16_flydsl_kernel(
            x, W1, group, handle=None, topk_idx=topk_idx, topk_weights=topk_w,
            layout="nt", BM=BM, BN=BN,
        ),
        warmup=args.warmup, iters=args.iters, group=group,
    )

    fp8["SwiGLU+quant"] = _bench(
        lambda: swiglu_mxfp8_flydsl_kernel(l1_fp8, ntb_fp8),
        warmup=args.warmup, iters=args.iters, group=group,
    )
    bf16["SwiGLU"] = _bench(
        lambda: swiglu_flydsl_kernel(l1_bf16, num_tile_blocks=ntb_bf16),
        warmup=args.warmup, iters=args.iters, group=group,
    )

    fp8["L2"] = _bench(
        lambda: grouped_gemm_combine_mxfp8_flydsl_kernel(
            None, w2_fp8, list(handle_fp8), group,
            topk_indices=topk_idx, topk_weights=topk_w,
            x_fp8=(act_fp8, act_a_sp), BM=BM, BN=BN, num_combine_cu=CC,
        )[0],
        warmup=args.warmup, iters=args.iters, group=group,
    )
    bf16["L2"] = _bench(
        lambda: grouped_gemm_combine_bf16_flydsl_kernel(
            act_bf16, W2, handle_bf16, topk_indices=topk_idx32, topk_weights=topk_w_flat,
            layout="nt", BM=BM, BN=BN,
        )[0],
        warmup=args.warmup, iters=args.iters, group=group,
    )

    fp8["FULL"] = _bench(
        lambda: fused_mega_moe_fp8(group, x, topk_idx, topk_w, W1, W2),
        warmup=args.warmup, iters=args.iters, group=group,
    )
    bf16["FULL"] = _bench(
        lambda: fused_mega_moe(group, x, topk_idx, topk_w, W1, W2),
        warmup=args.warmup, iters=args.iters, group=group,
    )

    if rank == 0:
        rows = [
            ("L1 dispatch+fc1", "L1", "L1"),
            ("SwiGLU (+ fp8 quant)", "SwiGLU+quant", "SwiGLU"),
            ("L2 fc2+combine", "L2", "L2"),
            ("FULL forward", "FULL", "FULL"),
        ]
        print(f"\n{'=' * 78}")
        print(f"[fwd breakdown fp8 vs bf16]  EP{world} T={T} H={H} I={I}  "
              f"fp8 L1={DC}/{PC} L2 cc={CC}  (max rank ms)")
        print(f"{'=' * 78}")
        print(f"  {'stage':<22} {'fp8':>8} {'bf16':>8} {'bf16/fp8':>9}  {'fp8%FULL':>8}")
        fp8_full = fp8["FULL"]
        for label, fk, bk in rows:
            fms, bms = fp8[fk], bf16[bk]
            ratio = bms / fms if fms > 0 else float("inf")
            pct = 100.0 * fms / fp8_full if fk != "FULL" else 100.0
            print(f"  {label:<22} {fms:7.3f} {bms:7.3f} {ratio:8.2f}x  {pct:7.1f}%")

        fp8_sum = fp8["L1"] + fp8["SwiGLU+quant"] + fp8["L2"]
        bf16_sum = bf16["L1"] + bf16["SwiGLU"] + bf16["L2"]
        print(f"\n  --- composition ---")
        print(f"  fp8  L1+SwiGLU+quant+L2 (isolated sum) = {fp8_sum:.3f} ms")
        print(f"  bf16 L1+SwiGLU+L2       (isolated sum) = {bf16_sum:.3f} ms")
        print(f"  fp8  FULL op                           = {fp8['FULL']:.3f} ms")
        print(f"  bf16 FULL op                           = {bf16['FULL']:.3f} ms")
        print(f"  FULL speedup bf16/fp8                  = {bf16['FULL'] / fp8['FULL']:.2f}x")
        print(f"  fp8  FULL - isolated_sum gap           = {fp8['FULL'] - fp8_sum:+.3f} ms")
        print(f"  bf16 FULL - isolated_sum gap           = {bf16['FULL'] - bf16_sum:+.3f} ms")

        print(f"\n  --- bottleneck (%% of fp8 FULL) ---")
        for label, fk in [("L1", "L1"), ("SwiGLU+quant", "SwiGLU+quant"), ("L2", "L2")]:
            print(f"  fp8 {label:<14} {100.0 * fp8[fk] / fp8_full:5.1f}%")
        for label, bk in [("L1", "L1"), ("SwiGLU", "SwiGLU"), ("L2", "L2")]:
            print(f"  bf16 {label:<13} {100.0 * bf16[bk] / bf16['FULL']:5.1f}%")

    dist.destroy_process_group()


def main():
    ap = argparse.ArgumentParser(description="fp8 vs bf16 mega MoE forward breakdown")
    ap.add_argument("--num-processes", type=int, default=8)
    ap.add_argument("--num-tokens", type=int, default=8192)
    ap.add_argument("--hidden", type=int, default=7168)
    ap.add_argument("--inter", type=int, default=2048)
    ap.add_argument("--num-experts", type=int, default=256)
    ap.add_argument("--num-topk", type=int, default=8)
    ap.add_argument("--bm", type=int, default=256)
    ap.add_argument("--bn", type=int, default=256)
    ap.add_argument("--dispatch-cu", type=int, default=16)
    ap.add_argument("--preshuffle-cu", type=int, default=16)
    ap.add_argument("--combine-cu", type=int, default=32)
    ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument("--iters", type=int, default=25)
    args = ap.parse_args()
    torch.multiprocessing.spawn(worker, args=(args.num_processes, args), nprocs=args.num_processes)


if __name__ == "__main__":
    main()
