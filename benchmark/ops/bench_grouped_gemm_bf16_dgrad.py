###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Benchmark standalone bf16 grouped NN dgrad GEMM vs fused dispatch(dy)+GEMM.

Compares:
  * ``grouped_gemm_bf16_dgrad_flydsl_kernel`` — GEMM-only (new)
  * ``dispatch_grouped_gemm_bf16_flydsl_kernel(..., layout=\"nn\")`` — comm PUSH + GEMM (production L2 dgrad)

Run (8 GPUs, inside dev container):
  PYTHONPATH=/perf_apps/xiaoming/MegaMoE python benchmark/ops/bench_grouped_gemm_bf16_dgrad.py \\
      --num-processes 8 --num-tokens 8192
"""

import argparse
import datetime
import math
import os

import torch
import torch.distributed as dist

import primus_turbo.pytorch  # noqa: F401
from primus_turbo.flydsl.grouped_gemm.grouped_gemm_bf16_dgrad_kernel import (
    grouped_gemm_bf16_dgrad_flydsl_kernel,
)
from primus_turbo.flydsl.mega import dispatch_grouped_gemm_bf16_flydsl_kernel


def _routing(T, K, E, *, device, seed):
    g = torch.Generator(device=device).manual_seed(seed)
    scores = torch.rand(T, E, generator=g, device=device).abs() + 1
    w, idx = torch.topk(scores.softmax(-1), K, dim=-1)
    return idx.to(torch.int64), w.to(torch.float32)


def _global_weights(E, I, H, device):
    g = torch.Generator(device=device).manual_seed(1234)
    W1 = torch.randn((E, 2 * I, H), generator=g, device=device, dtype=torch.bfloat16) * (2.0 / math.sqrt(H))
    W2 = torch.randn((E, H, I), generator=g, device=device, dtype=torch.bfloat16) * (2.0 / math.sqrt(I))
    return W1, W2


def _snr_db(ref, out):
    ref, out = ref.float(), out.float()
    return float(10.0 * torch.log10(ref.pow(2).sum() / ((ref - out).pow(2).sum() + 1e-12)))


def _bench(fn, *, warmup, iters, group):
    torch.cuda.synchronize()
    group.barrier()
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    group.barrier()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return float(s.elapsed_time(e) / iters)


@torch.no_grad()
def profile(group, args):
    rank, world = group.rank(), group.size()
    H, I, E, K, T, BM, BN = args.hidden, args.inter, args.num_experts, args.num_topk, args.num_tokens, args.bm, args.bn
    epr = E // world

    torch.manual_seed(7 + rank)
    x = torch.randn((T, H), device="cuda", dtype=torch.bfloat16)
    topk_idx, topk_w = _routing(T, K, E, device="cuda", seed=100 + rank)
    W1g, W2g = _global_weights(E, I, H, "cuda")
    W1 = W1g[rank * epr : (rank + 1) * epr].contiguous()
    W2 = W2g[rank * epr : (rank + 1) * epr].contiguous()
    del W1g, W2g
    dy = torch.randn((T, H), device="cuda", dtype=torch.bfloat16)

    # Forward nt dispatch -> handle (tile_to_expert / num_tile_blocks routing metadata).
    _, _, _, handle = dispatch_grouped_gemm_bf16_flydsl_kernel(
        x, W1, group, handle=None, topk_idx=topk_idx, topk_weights=topk_w, layout="nt", BM=BM, BN=BN,
    )
    tile_to_expert = handle[5]
    num_tile_blocks = handle[8]

    # One fused dy-dispatch + GEMM to populate the dy pool and check correctness.
    out_fused, pool_dy, _, _ = dispatch_grouped_gemm_bf16_flydsl_kernel(
        dy, W2, group, handle=handle, layout="nn", BM=BM, BN=BN, GROUP_M=args.group_m,
    )
    pool_snapshot = pool_dy.clone()
    M_pad = pool_dy.shape[0]
    M_eff = int(num_tile_blocks.item()) * BM

    out_standalone = grouped_gemm_bf16_dgrad_flydsl_kernel(
        pool_dy, W2, tile_to_expert, num_tile_blocks, BLOCK_M=BM, BLOCK_N=BN, GROUP_M=args.group_m,
    )
    snr = _snr_db(out_fused[:M_eff], out_standalone[:M_eff])
    nan = bool((~torch.isfinite(out_standalone.float())).any())

    def _gemm_only():
        return grouped_gemm_bf16_dgrad_flydsl_kernel(
            pool_snapshot, W2, tile_to_expert, num_tile_blocks, BLOCK_M=BM, BLOCK_N=BN, GROUP_M=args.group_m,
        )

    def _fused():
        out, _, _, _ = dispatch_grouped_gemm_bf16_flydsl_kernel(
            dy, W2, group, handle=handle, layout="nn", BM=BM, BN=BN, GROUP_M=args.group_m,
        )
        return out

    t_gemm = _bench(_gemm_only, warmup=args.warmup, iters=args.iters, group=group)
    t_fused = _bench(_fused, warmup=args.warmup, iters=args.iters, group=group)

    flops = 2.0 * M_eff * I * H
    return {
        "snr": snr,
        "nan": float(nan),
        "t_gemm": t_gemm,
        "t_fused": t_fused,
        "flops": flops,
        "M_eff": M_eff,
        "M_pad": M_pad,
    }


def _amax(group, v):
    t = torch.tensor([v], device="cuda")
    dist.all_reduce(t, op=dist.ReduceOp.MAX, group=group)
    return float(t)


def _amin(group, v):
    t = torch.tensor([v], device="cuda")
    dist.all_reduce(t, op=dist.ReduceOp.MIN, group=group)
    return float(t)


def worker(local_rank, world, args):
    ip = os.getenv("MASTER_ADDR", "127.0.0.1")
    port = int(os.getenv("MASTER_PORT", "8493"))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        "nccl",
        init_method=f"tcp://{ip}:{port}",
        world_size=world,
        rank=local_rank,
        timeout=datetime.timedelta(seconds=int(os.getenv("MEGA_BENCH_TIMEOUT_S", "600"))),
    )
    torch.set_default_device("cuda")
    group = dist.new_group(list(range(world)))
    rank = dist.get_rank()
    try:
        r = profile(group, args)
        snr, nan = _amin(group, r["snr"]), _amax(group, r["nan"])
        t_gemm, t_fused = _amax(group, r["t_gemm"]), _amax(group, r["t_fused"])
        if rank == 0:
            tf = lambda ms: r["flops"] / (ms * 1e-3) / 1e12
            comm_ms = t_fused - t_gemm
            print(f"\n{'=' * 72}")
            print(
                f"[bf16 grouped NN dgrad  GEMM-only vs fused dispatch+GEMM]  EP{world} "
                f"T={args.num_tokens} H={args.hidden} I={args.inter} E={args.num_experts} K={args.num_topk}"
            )
            print(f"{'=' * 72}")
            print(f"  M_eff={r['M_eff']}  M_pad={r['M_pad']}  BM={args.bm} BN={args.bn} GROUP_M={args.group_m}")
            print(f"  grouped dgrad GEMM only : {t_gemm:8.3f} ms | {tf(t_gemm):8.1f} TFLOPS")
            print(f"  fused dispatch+GEMM     : {t_fused:8.3f} ms | {tf(t_fused):8.1f} TFLOPS")
            print(f"  implied comm (fused-gemm): {comm_ms:8.3f} ms  ({comm_ms / t_fused * 100:.1f}% of fused)")
            print(
                f"  [acc] standalone vs fused output: SNR={snr:.2f} dB  nan={bool(nan)}  "
                f"{'PASS' if snr >= 40.0 and not nan else 'FAIL'} (gate SNR>=40dB)"
            )
        torch.cuda.synchronize()
        group.barrier()
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="bf16 grouped NN dgrad GEMM-only vs fused dispatch+GEMM")
    ap.add_argument("--num-processes", type=int, default=8)
    ap.add_argument("--hidden", type=int, default=7168)
    ap.add_argument("--inter", type=int, default=2048)
    ap.add_argument("--num-experts", type=int, default=256)
    ap.add_argument("--num-topk", type=int, default=8)
    ap.add_argument("--num-tokens", type=int, default=8192)
    ap.add_argument("--bm", type=int, default=256)
    ap.add_argument("--bn", type=int, default=256)
    ap.add_argument("--group-m", type=int, default=4)
    ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument("--iters", type=int, default=20)
    args = ap.parse_args()
    torch.multiprocessing.spawn(worker, args=(args.num_processes, args), nprocs=args.num_processes)
