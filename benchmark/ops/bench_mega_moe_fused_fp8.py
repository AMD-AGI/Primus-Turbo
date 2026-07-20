###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""End-to-end forward validation for the fp8 mega MoE op.

Compares the fp8 forward ``mega_moe_fused_fp8`` (L1 fused mxfp8 dispatch+fc1 -> SwiGLU -> L2 fp8
combine) against the bf16 ``mega_moe_fused`` on identical inputs, with an SNR/cos gate, and reports
rough per-op latency. Forward-only (no_grad); the fp8 backward is not ported yet.

Run inside the dev container (8 GPUs):
  PYTHONPATH=<repo> python benchmark/ops/bench_mega_moe_fused_fp8.py --num-processes 8 --num-tokens 2048
"""

import argparse
import datetime
import math
import os

import numpy as np
import torch
import torch.distributed as dist

import primus_turbo.pytorch  # noqa: F401
from primus_turbo.pytorch.ops.moe.mega_moe_fused import mega_moe_fused
from primus_turbo.pytorch.ops.moe.mega_moe_fused_fp8 import mega_moe_fused_fp8


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
    sig = ref.pow(2).sum()
    noise = (ref - out).pow(2).sum()
    return float(10.0 * torch.log10(sig / (noise + 1e-12)))


def _cos(ref, out):
    r, o = ref.float().flatten(), out.float().flatten()
    return float(torch.dot(o, r) / (o.norm() * r.norm() + 1e-12))


def _bench(fn, *, warmup, iters, group):
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for _ in range(warmup):
        torch.cuda.synchronize(); group.barrier(); fn()
    for i in range(iters):
        torch.cuda.synchronize(); group.barrier()
        starts[i].record(); fn(); ends[i].record()
    torch.cuda.synchronize()
    return float(np.average([s.elapsed_time(e) for s, e in zip(starts, ends)][1:]))


@torch.no_grad()
def profile(group, args):
    rank, world = group.rank(), group.size()
    H, I, E, K, T = args.hidden, args.inter, args.num_experts, args.num_topk, args.num_tokens
    epr = E // world

    torch.manual_seed(7 + rank)
    x = torch.randn((T, H), device="cuda", dtype=torch.float32).bfloat16()
    topk_idx, topk_w = _routing(T, K, E, device="cuda", seed=100 + rank)
    W1g, W2g = _global_weights(E, I, H, "cuda")
    W1 = W1g[rank * epr : (rank + 1) * epr].contiguous()
    W2 = W2g[rank * epr : (rank + 1) * epr].contiguous()
    del W1g, W2g

    # fp8 first (optionally in isolation) so a NaN here can't be blamed on bf16-op interference
    y_fp8 = mega_moe_fused_fp8(group, x, topk_idx, topk_w, W1, W2)
    print(f"[rank{rank}] fp8 norm={float(y_fp8.float().norm()):.4e} "
          f"nan={bool(torch.isnan(y_fp8).any())} inf={bool(torch.isinf(y_fp8).any())}", flush=True)
    if args.only == "fp8":
        return {"snr": 0.0, "cos": 0.0, "t_bf16": 1.0, "t_fp8": 1.0}
    y_bf16 = mega_moe_fused(group, x, topk_idx, topk_w, W1, W2)
    print(f"[rank{rank}] bf16 norm={float(y_bf16.float().norm()):.4e} "
          f"nan={bool(torch.isnan(y_bf16).any())}", flush=True)
    snr = _snr_db(y_bf16, y_fp8)
    cos = _cos(y_bf16, y_fp8)

    t_bf16 = _bench(lambda: mega_moe_fused(group, x, topk_idx, topk_w, W1, W2),
                    warmup=args.warmup, iters=args.iters, group=group)
    t_fp8 = _bench(lambda: mega_moe_fused_fp8(group, x, topk_idx, topk_w, W1, W2),
                   warmup=args.warmup, iters=args.iters, group=group)
    return {"snr": snr, "cos": cos, "t_bf16": t_bf16, "t_fp8": t_fp8}


def _amin(group, v):
    t = torch.tensor([v], device="cuda"); dist.all_reduce(t, op=dist.ReduceOp.MIN, group=group); return float(t)


def _amax(group, v):
    t = torch.tensor([v], device="cuda"); dist.all_reduce(t, op=dist.ReduceOp.MAX, group=group); return float(t)


def worker(local_rank, world, args):
    ip = os.getenv("MASTER_ADDR", "127.0.0.1")
    port = int(os.getenv("MASTER_PORT", "8492"))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        "nccl", init_method=f"tcp://{ip}:{port}", world_size=world, rank=local_rank,
        timeout=datetime.timedelta(seconds=int(os.getenv("MEGA_BENCH_TIMEOUT_S", "600"))),
    )
    torch.set_default_device("cuda")
    group = dist.new_group(list(range(world)))
    rank = dist.get_rank()
    try:
        r = profile(group, args)
        snr, cos = _amin(group, r["snr"]), _amin(group, r["cos"])
        t_bf16, t_fp8 = _amax(group, r["t_bf16"]), _amax(group, r["t_fp8"])
        if rank == 0:
            print(f"\n{'='*72}")
            print(f"[mega MoE forward  fp8 vs bf16]  EP{world} T={args.num_tokens} H={args.hidden} "
                  f"I={args.inter} E={args.num_experts} K={args.num_topk}")
            print(f"{'='*72}")
            print(f"  bf16 forward : {t_bf16:8.3f} ms")
            print(f"  fp8  forward : {t_fp8:8.3f} ms | {t_bf16 / t_fp8:.2f}x vs bf16")
            print(f"  [acc] fp8 vs bf16: SNR={snr:.2f} dB  cos={cos:.5f}  "
                  f"{'PASS' if snr >= 18.0 else 'FAIL'} (gate SNR>=18dB)")
        torch.cuda.synchronize(); group.barrier()
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="mega MoE forward fp8 vs bf16 (SNR gate + latency)")
    ap.add_argument("--num-processes", type=int, default=8)
    ap.add_argument("--hidden", type=int, default=7168)
    ap.add_argument("--inter", type=int, default=2048)
    ap.add_argument("--num-experts", type=int, default=256)
    ap.add_argument("--num-topk", type=int, default=8)
    ap.add_argument("--num-tokens", type=int, default=2048)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=15)
    ap.add_argument("--only", choices=["both", "fp8"], default="both")
    args = ap.parse_args()
    torch.multiprocessing.spawn(worker, args=(args.num_processes, args), nprocs=args.num_processes)
