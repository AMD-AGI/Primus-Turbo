#!/usr/bin/env python3
"""Isolated swiglu_mxfp8_flydsl_kernel for rocprof / timing (EP8 DSv3 L1 output shape)."""

import argparse
import datetime
import math

import numpy as np
import torch
import torch.distributed as dist

import primus_turbo.pytorch  # noqa: F401
from primus_turbo.flydsl.mega.fp8 import (
    dispatch_grouped_gemm_mxfp8,
    dispatch_prologue,
    get_symm_buffer_for_mega_moe,
    quantize_grouped_weight_mxfp8_flydsl,
    swiglu_mxfp8_flydsl_kernel,
)
from primus_turbo.pytorch.kernels.fused_mega_moe.fused_mega_moe_weight_prep_fp8 import (
    prepare_dispatch_weight_fp8,
)


def _bench(fn, group, warmup, iters):
    for _ in range(warmup):
        torch.cuda.synchronize()
        group.barrier()
        fn()
    ss = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ee = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        torch.cuda.synchronize()
        group.barrier()
        ss[i].record()
        fn()
        ee[i].record()
    torch.cuda.synchronize()
    ms = float(np.mean([a.elapsed_time(b) for a, b in zip(ss, ee)][1:]))
    t = torch.tensor([ms], device="cuda")
    dist.all_reduce(t, op=dist.ReduceOp.MAX, group=group)
    return float(t)


def worker(local_rank, world, args):
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        "nccl",
        init_method=f"tcp://127.0.0.1:{args.port}",
        world_size=world,
        rank=local_rank,
        timeout=datetime.timedelta(seconds=600),
    )
    group = dist.new_group(list(range(world)))
    rank = dist.get_rank()
    T, H, I, E, K, BM, BN = args.num_tokens, args.hidden, args.inter, args.num_experts, args.num_topk, 256, 256
    epr = E // world
    torch.manual_seed(7 + rank)
    x = torch.randn(T, H, device="cuda", dtype=torch.bfloat16)
    g = torch.Generator(device="cuda").manual_seed(100 + rank)
    scores = torch.rand(T, E, generator=g, device="cuda").abs() + 1
    topk_w, topk_idx = torch.topk(scores.softmax(-1), K, dim=-1)
    topk_w, topk_idx = topk_w.float(), topk_idx.long()
    W1 = torch.randn(E, 2 * I, H, device="cuda", dtype=torch.bfloat16) * (2 / math.sqrt(H))
    W1 = W1[rank * epr : (rank + 1) * epr].contiguous()
    symm = get_symm_buffer_for_mega_moe(
        group, num_experts=E, num_max_tokens_per_rank=T, num_topk=K,
        hidden=H, intermediate_hidden=I, block_m=BM, block_n=BN, use_mxfp8=True,
    )
    sym_layout = symm.make_sym_layout()
    handle = tuple(
        dispatch_prologue(
            topk_idx, topk_w, sym_layout=sym_layout, num_tokens=T, num_topk=K,
            num_experts=E, world_size=world, rank=rank, experts_per_rank=epr, block_m=BM,
            num_max_pool_tokens=symm.num_max_pool_tokens,
        )
    )
    w1_fp8 = prepare_dispatch_weight_fp8(W1)
    w1q, w1s = w1_fp8[:2]
    l1 = dispatch_grouped_gemm_mxfp8(x, w1_fp8, handle, sym_layout, symm, BM=BM, BN=BN)
    ntb = symm.meta_scalars[1:2]
    torch.cuda.synchronize()
    group.barrier()

    def _run():
        swiglu_mxfp8_flydsl_kernel(l1, ntb)

    if args.profile_iters > 0:
        for _ in range(args.warmup):
            _run()
        torch.cuda.synchronize()
        group.barrier()
        for _ in range(args.profile_iters):
            _run()
    else:
        ms = _bench(_run, group, args.wwarmup, args.iters)
        if rank == 0:
            print(f"swiglu_mxfp8: {ms:.3f} ms (max rank, EP{world} T={T})")
    dist.destroy_process_group()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-processes", type=int, default=8)
    ap.add_argument("--port", type=int, default=9600)
    ap.add_argument("--num-tokens", type=int, default=8192)
    ap.add_argument("--hidden", type=int, default=7168)
    ap.add_argument("--inter", type=int, default=2048)
    ap.add_argument("--num-experts", type=int, default=256)
    ap.add_argument("--num-topk", type=int, default=8)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=15)
    ap.add_argument("--profile-iters", type=int, default=0, help="if >0, run N iters for rocprof (no timing)")
    args = ap.parse_args()
    torch.multiprocessing.spawn(worker, args=(args.num_processes, args), nprocs=args.num_processes)


if __name__ == "__main__":
    main()
