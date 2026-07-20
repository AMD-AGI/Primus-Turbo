###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Isolated fc2 (L2) latency benchmark for the fp8 combine.

Builds the symm workspace + prologue, runs the fp8 L1 (dispatch+fc1) + SwiGLU to get a real
``act`` in the pool, then times the fp8 L2 path on that ``act``:
  grouped_gemm_combine_fp8  (fp8 GEMM + mxfp8 epilogue + fp8 PUSH + fp8-dequant reduce) -> y bf16.
Reports latency + a finite/NaN check. (For the fp8-vs-bf16 accuracy/speed comparison at the
whole-forward level, use bench_mega_moe_fused_fp8.py, which uses the bf16 mega_moe_fused as the
reference -- this repo carries no bf16 combine kernel under flydsl/mega/fp8/.)

Run inside the dev container (8 GPUs):
  PYTHONPATH=<repo> python benchmark/ops/bench_grouped_gemm_combine_fp8_l2.py --num-processes 8 --num-tokens 8192
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
    dispatch_grouped_gemm_mxfp8,
    dispatch_prologue,
    get_symm_buffer_for_mega_moe,
    grouped_gemm_combine_fp8,
    prepare_w2_fp8,
    quantize_grouped_weight_mxfp8,
    swiglu,
)


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


def _bench(fn, *, warmup, iters, group, reset):
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]

    def _iter(s=None, e=None):
        torch.cuda.synchronize(); group.barrier()
        reset()
        torch.cuda.synchronize(); group.barrier()
        if s is None:
            fn(); return
        s.record(); fn(); e.record()

    for _ in range(warmup):
        _iter()
    for i in range(iters):
        _iter(starts[i], ends[i])
    torch.cuda.synchronize()
    return float(np.average([s.elapsed_time(e) for s, e in zip(starts, ends)][1:]))


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

    symm = get_symm_buffer_for_mega_moe(
        group, num_experts=E, num_max_tokens_per_rank=T, num_topk=K, hidden=H,
        intermediate_hidden=I, block_m=BM, block_n=BN, use_mxfp8=True,
    )
    sym_layout = symm.make_sym_layout()
    handle = tuple(dispatch_prologue(
        topk_idx, topk_w, sym_layout=sym_layout, num_tokens=T, num_topk=K, num_experts=E,
        world_size=world, rank=symm.rank, experts_per_rank=epr, block_m=BM,
        num_max_pool_tokens=symm.num_max_pool_tokens,
    ))
    w1q, w1s = quantize_grouped_weight_mxfp8(W1)

    # L1 (fp8) + SwiGLU -> act (the real L2 input), once
    torch.cuda.synchronize(); group.barrier()
    symm.scoreboard.zero_()
    torch.cuda.synchronize(); group.barrier()
    l1 = dispatch_grouped_gemm_mxfp8(x, None, w1q, w1s, handle, sym_layout, symm, BM=BM, BN=BN)
    act = swiglu(l1).contiguous()

    tw_f32 = topk_w.to(torch.float32)
    w2_fp8 = prepare_w2_fp8(W2)  # weight prep at the caller (op layer); combine is pure compute

    def _reset_l2():
        symm.sb_l2.zero_()
        symm.barrier_local.fill_(-1)

    def _fp8():
        return grouped_gemm_combine_fp8(
            act, w2_fp8, list(handle), group, topk_indices=topk_idx, topk_weights=tw_f32,
            BM=BM, BN=BN, num_combine_cu=48,
        )

    # correctness smoke: fp8 L2 output must be finite
    torch.cuda.synchronize(); group.barrier(); _reset_l2(); torch.cuda.synchronize(); group.barrier()
    y_fp8 = _fp8()
    torch.cuda.synchronize(); group.barrier()
    nan = bool((~torch.isfinite(y_fp8.float())).any())

    t_fp8 = _bench(_fp8, warmup=args.warmup, iters=args.iters, group=group, reset=_reset_l2)
    symm.destroy()
    return {"t_fp8": t_fp8, "nan": float(nan)}


def _amax(group, v):
    t = torch.tensor([v], device="cuda"); dist.all_reduce(t, op=dist.ReduceOp.MAX, group=group); return float(t)


def _amin(group, v):
    t = torch.tensor([v], device="cuda"); dist.all_reduce(t, op=dist.ReduceOp.MIN, group=group); return float(t)


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
        t_fp8 = _amax(group, r["t_fp8"])
        nan = _amax(group, r["nan"])
        if rank == 0:
            print(f"\n{'='*72}")
            print(f"[fc2 (L2 combine) fp8]  EP{world} T={args.num_tokens} H={args.hidden} "
                  f"I={args.inter} E={args.num_experts} K={args.num_topk} BM={args.bm} BN={args.bn}")
            print(f"{'='*72}")
            print(f"  fp8 L2 combine : {t_fp8:8.3f} ms   finite={not bool(nan)}")
        torch.cuda.synchronize(); group.barrier()
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="isolated fc2 (L2 combine) fp8 vs bf16")
    ap.add_argument("--num-processes", type=int, default=8)
    ap.add_argument("--hidden", type=int, default=7168)
    ap.add_argument("--inter", type=int, default=2048)
    ap.add_argument("--num-experts", type=int, default=256)
    ap.add_argument("--num-topk", type=int, default=8)
    ap.add_argument("--num-tokens", type=int, default=8192)
    ap.add_argument("--bm", type=int, default=256)
    ap.add_argument("--bn", type=int, default=256)
    ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument("--iters", type=int, default=20)
    args = ap.parse_args()
    torch.multiprocessing.spawn(worker, args=(args.num_processes, args), nprocs=args.num_processes)
