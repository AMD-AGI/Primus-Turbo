#!/usr/bin/env python3
"""Sweep num_combine_cu for fp8 fwd L2 + FULL op (EP8, DSv3).

Run (8 GPUs):
  PYTHONPATH=<repo> python benchmark/ops/bench_fwd_combine_cu_sweep.py \\
    --num-processes 8 --num-tokens 8192 --warmup 8 --iters 20
"""

import argparse
import datetime
import math
import os

import numpy as np
import torch
import torch.distributed as dist

import primus_turbo.pytorch  # noqa: F401
from primus_turbo.flydsl.mega.fp8 import (
    dispatch_grouped_gemm_mxfp8_flydsl_kernel,
    grouped_gemm_combine_mxfp8_flydsl_kernel,
    swiglu_mxfp8_flydsl_kernel,
)
from primus_turbo.pytorch.kernels.fused_mega_moe.fused_mega_moe_backward_fp8_impl import prepare_w2_fp8
from primus_turbo.pytorch.kernels.fused_mega_moe.fused_mega_moe_backward_fp8_impl import _w1_fp8_cached
from primus_turbo.pytorch.ops.moe.fused_mega_moe_fp8 import fused_mega_moe_fp8


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
    epr = E // world
    candidates = [int(x) for x in args.combine_cu_list.split(",")]

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

    l1, handle, _, _ = dispatch_grouped_gemm_mxfp8_flydsl_kernel(
        x, w1_fp8, group, topk_idx=topk_idx, topk_weights=topk_w, BM=BM, BN=BN,
    )
    ntb = handle[11]
    act_fp8, act_a_sp = swiglu_mxfp8_flydsl_kernel(l1, ntb)
    torch.cuda.synchronize()
    group.barrier()

    full_ms = _bench(
        lambda: fused_mega_moe_fp8(group, x, topk_idx, topk_w, W1, W2),
        warmup=args.warmup, iters=args.iters, group=group,
    )

    rows = []
    for cc in candidates:
        l2_ms = _bench(
            lambda cc=cc: grouped_gemm_combine_mxfp8_flydsl_kernel(
                None, w2_fp8, list(handle), group,
                topk_indices=topk_idx, topk_weights=topk_w,
                x_fp8=(act_fp8, act_a_sp), BM=BM, BN=BN, num_combine_cu=cc,
            )[0],
            warmup=args.warmup, iters=args.iters, group=group,
        )
        rows.append((cc, l2_ms))

    if rank == 0:
        print(f"\n{'=' * 72}")
        print(f"[fwd L2 combine_cu sweep]  EP{world} T={T}  FULL(op, pinned cc=32)={full_ms:.3f} ms")
        print(f"{'=' * 72}")
        print(f"  {'combine_cu':>10}  {'L2 ms':>8}")
        best_cc, best_l2 = min(rows, key=lambda r: r[1])
        for cc, l2_ms in rows:
            mark = " <-- best L2" if cc == best_cc else ""
            print(f"  {cc:10d}  {l2_ms:8.3f}{mark}")
        print(f"\n  best L2: combine_cu={best_cc} @ {best_l2:.3f} ms")

    dist.destroy_process_group()


def main():
    ap = argparse.ArgumentParser(description="fp8 fwd L2 num_combine_cu sweep")
    ap.add_argument("--num-processes", type=int, default=8)
    ap.add_argument("--num-tokens", type=int, default=8192)
    ap.add_argument("--hidden", type=int, default=7168)
    ap.add_argument("--inter", type=int, default=2048)
    ap.add_argument("--num-experts", type=int, default=256)
    ap.add_argument("--num-topk", type=int, default=8)
    ap.add_argument("--bm", type=int, default=256)
    ap.add_argument("--bn", type=int, default=256)
    ap.add_argument("--combine-cu-list", type=str, default="24,32,40,48,56")
    ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument("--iters", type=int, default=20)
    args = ap.parse_args()
    torch.multiprocessing.spawn(worker, args=(args.num_processes, args), nprocs=args.num_processes)


if __name__ == "__main__":
    main()
