#!/usr/bin/env python3
"""Full mega MoE e2e timing: fwd+bwd, fwd-only, and bwd-only, across op variants.

Variants: ``fp8`` (single fused op), ``fp8_staged`` (the two-stage gate-up / gate-down split),
``bf16``. All legs use the same persistent leaf tensors so fp8 weight-quant caches hit across iters
(mirrors training). Cross-check: fwd+bwd ~= fwd-only + bwd-only.

Run (8 GPUs):
  PYTHONPATH=<repo> python benchmark/ops/bench_mega_moe_bwd_only.py \\
    --num-processes 8 --num-tokens 8192 --routing-mode both --warmup 8 --iters 25
  # staged-vs-fused fp8 split regression check:
  PYTHONPATH=<repo> python benchmark/ops/bench_mega_moe_bwd_only.py --only fp8_split
"""

import argparse
import datetime
import math
import os

import numpy as np
import torch
import torch.distributed as dist

import primus_turbo.pytorch  # noqa: F401
from primus_turbo.pytorch.ops.moe.fused_mega_moe import fused_mega_moe
from primus_turbo.pytorch.ops.moe.fused_mega_moe_fp8 import (
    fused_mega_moe_fp8,
    fused_mega_moe_fp8_stage1,
    fused_mega_moe_fp8_stage2,
)

_ROW_LABEL_W = 14
_COL_W = 11


def _fused_mega_moe_fp8_staged(group, x, topk_idx, topk_weights, W1, W2):
    """The two-stage fp8 split behind the single-op signature the bench drives."""
    l1, dispatch_weights, handle, state = fused_mega_moe_fp8_stage1(x, topk_idx, topk_weights, W1, group)
    return fused_mega_moe_fp8_stage2(
        l1, dispatch_weights, handle, state, topk_idx, topk_weights, W2, group
    )


_OPS = {
    "fp8": fused_mega_moe_fp8,
    "fp8_staged": _fused_mega_moe_fp8_staged,
    "bf16": fused_mega_moe,
}
# --only -> which legs to time, in print order. The first leg is the ratio denominator.
_LEG_SETS = {
    "both": ["fp8", "bf16"],
    "all": ["fp8", "fp8_staged", "bf16"],
    "fp8_split": ["fp8", "fp8_staged"],
    "fp8": ["fp8"],
    "fp8_staged": ["fp8_staged"],
    "bf16": ["bf16"],
}


def _routing(T, K, E, *, device, seed, mode):
    """Match ``bench_mega_moe_fp8.generate_inputs`` routing modes."""
    g = torch.Generator(device=device).manual_seed(seed)
    if mode == "load_balanced":
        scores = torch.rand(T, E, generator=g, device=device).abs() + 1
        topk_w, topk_idx = torch.topk(scores.softmax(-1), K, dim=-1)
    elif mode == "round_robin":
        topk_idx = torch.arange(T * K, device=device).view(T, K) % E
        topk_w = torch.rand(T, K, generator=g, device=device).softmax(-1)
    else:
        raise ValueError(f"unknown routing mode: {mode}")
    return topk_idx.to(torch.int64), topk_w.to(torch.float32)


def _global_weights(E, I, H, device):
    g = torch.Generator(device=device).manual_seed(1234)
    W1 = torch.randn((E, 2 * I, H), generator=g, device=device, dtype=torch.bfloat16) * (
        2.0 / math.sqrt(H)
    )
    W2 = torch.randn((E, H, I), generator=g, device=device, dtype=torch.bfloat16) * (
        2.0 / math.sqrt(I)
    )
    return W1, W2


def _leaf(t):
    return t.detach().clone().requires_grad_(True)


def _bench(fn, *, warmup, iters, group):
    """Back-to-back CUDA-event timing (same helper as bench_mega_moe_fused_fp8_bwd)."""
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for _ in range(warmup):
        torch.cuda.synchronize()
        group.barrier()
        fn()
    for i in range(iters):
        torch.cuda.synchronize()
        group.barrier()
        starts[i].record()
        fn()
        ends[i].record()
    torch.cuda.synchronize()
    return float(np.average([s.elapsed_time(e) for s, e in zip(starts, ends)][1:]))


def _bench_bwd_only(op, group, x, topk_idx, topk_w, W1, W2, grad_y, *, warmup, iters):
    """Time only ``y.backward(grad_y)``; forward runs outside the CUDA-event window."""
    xL, twL, W1L, W2L = _leaf(x), _leaf(topk_w), _leaf(W1), _leaf(W2)
    for _ in range(warmup):
        for t in (xL, twL, W1L, W2L):
            t.grad = None
        op(group, xL, topk_idx, twL, W1L, W2L).backward(grad_y)
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        for t in (xL, twL, W1L, W2L):
            t.grad = None
        y = op(group, xL, topk_idx, twL, W1L, W2L)
        torch.cuda.synchronize()
        group.barrier()
        starts[i].record()
        y.backward(grad_y)
        ends[i].record()
    torch.cuda.synchronize()
    return float(np.average([s.elapsed_time(e) for s, e in zip(starts, ends)][1:]))


def _profile_op(op, group, x, topk_idx, topk_w, W1, W2, grad_y, *, warmup, iters):
    """Return (t_fwd_bwd, t_fwd, t_bwd) for one op with persistent leaves."""
    xL, twL, W1L, W2L = _leaf(x), _leaf(topk_w), _leaf(W1), _leaf(W2)

    def _clear():
        for t in (xL, twL, W1L, W2L):
            t.grad = None

    def _fwd_bwd():
        _clear()
        op(group, xL, topk_idx, twL, W1L, W2L).backward(grad_y)

    def _fwd_only():
        _clear()
        op(group, xL, topk_idx, twL, W1L, W2L)

    t_fwd_bwd = _bench(_fwd_bwd, warmup=warmup, iters=iters, group=group)
    t_fwd = _bench(_fwd_only, warmup=warmup, iters=iters, group=group)
    t_bwd = _bench_bwd_only(op, group, x, topk_idx, topk_w, W1, W2, grad_y, warmup=warmup, iters=iters)
    return t_fwd_bwd, t_fwd, t_bwd


def _amax(group, v):
    t = torch.tensor([v], device="cuda")
    dist.all_reduce(t, op=dist.ReduceOp.MAX, group=group)
    return float(t)


def _profile_routing(group, args, *, x, W1, W2, grad_y, topk_idx, topk_w):
    """-> ``{leg: (t_fwd_bwd, t_fwd, t_bwd)}`` for every leg ``--only`` selected."""
    out = {}
    for leg in _LEG_SETS[args.only]:
        t = _profile_op(
            _OPS[leg], group, x, topk_idx, topk_w, W1, W2, grad_y,
            warmup=args.warmup, iters=args.iters,
        )
        out[leg] = tuple(_amax(group, v) for v in t)
    return out


def _print_routing_table(routing_mode, timings):
    w = _ROW_LABEL_W
    c = _COL_W
    sep = "-" * (w + 3 * c)
    print(f"\n{routing_mode:<{w}}{'fwd+bwd':>{c}}{'fwd-only':>{c}}{'bwd-only':>{c}}")
    print(sep)
    for leg, t in timings.items():
        print(f"{leg:<{w}}{t[0]:>{c}.3f}{t[1]:>{c}.3f}{t[2]:>{c}.3f}")
    legs = list(timings)
    base = timings[legs[0]]
    for leg in legs[1:]:
        ratio = tuple(o / b if b > 0 else 0.0 for b, o in zip(base, timings[leg]))
        label = f"{leg}/{legs[0]}"
        print(f"{label:<{w}}" + "".join(f"{r:>{c - 1}.2f}x" for r in ratio))
    for leg, t in timings.items():
        print(
            f"\n  cross-check {leg:<10}: fwd+bwd={t[0]:.3f} ms  "
            f"fwd+bwd_est={t[1] + t[2]:.3f} ms  (fwd-only + bwd-only)"
        )


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
    H, I, E, K, T = args.hidden, args.inter, args.num_experts, args.num_topk, args.num_tokens
    epr = E // world
    routing_modes = (
        ["load_balanced", "round_robin"] if args.routing_mode == "both"
        else [args.routing_mode]
    )

    torch.manual_seed(7 + rank)
    x = torch.randn((T, H), device="cuda", dtype=torch.bfloat16)
    W1g, W2g = _global_weights(E, I, H, "cuda")
    W1 = W1g[rank * epr : (rank + 1) * epr].contiguous()
    W2 = W2g[rank * epr : (rank + 1) * epr].contiguous()
    grad_y = torch.randn((T, H), device="cuda", dtype=torch.bfloat16)

    try:
        if rank == 0:
            sep = "=" * 80
            print(f"\n{sep}")
            print(
                f"[mega MoE e2e autograd timing]  EP{world} T={T} H={H} I={I} "
                f"E={E} K={K}  (persistent weights; max over ranks; ms)"
            )
            print(sep)

        for routing_mode in routing_modes:
            topk_idx, topk_w = _routing(T, K, E, device="cuda", seed=100 + rank, mode=routing_mode)
            timings = _profile_routing(
                group, args, x=x, W1=W1, W2=W2, grad_y=grad_y, topk_idx=topk_idx, topk_w=topk_w,
            )
            if rank == 0:
                _print_routing_table(routing_mode, timings)
            torch.cuda.synchronize()
            group.barrier()
    finally:
        dist.destroy_process_group()


def main():
    ap = argparse.ArgumentParser(
        description="mega MoE e2e fwd+bwd / fwd-only / bwd-only timing (fp8 / fp8_staged / bf16)"
    )
    ap.add_argument("--num-processes", type=int, default=8)
    ap.add_argument("--hidden", type=int, default=7168)
    ap.add_argument("--inter", type=int, default=2048)
    ap.add_argument("--num-experts", type=int, default=256)
    ap.add_argument("--num-topk", type=int, default=8)
    ap.add_argument("--num-tokens", type=int, default=8192)
    ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument("--iters", type=int, default=25)
    ap.add_argument("--only", choices=list(_LEG_SETS), default="both")
    ap.add_argument(
        "--routing-mode",
        choices=["load_balanced", "round_robin", "both"],
        default="both",
        help="top-k routing distribution (matches bench_mega_moe_fp8 --mode); "
             "'both' runs load_balanced then round_robin",
    )
    args = ap.parse_args()
    torch.multiprocessing.spawn(worker, args=(args.num_processes, args), nprocs=args.num_processes)


if __name__ == "__main__":
    main()
