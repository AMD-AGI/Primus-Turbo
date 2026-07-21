###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Backward STEP1 (dispatch(dy) + fc2 dgrad) fp8 -- isolated correctness + latency.

Sets up the forward symm + prologue (real fwd->bwd order: run the fp8 L1 first), then runs STEP1
(``_mxfp8_step1_dispatch_dgrad``): fp8 dispatch(dy) PUSH + grouped mxfp8 fc2-dgrad NT GEMM ->
``grad_swiglu`` [P, I] + the dispatched-dy fp8 pool. Correctness: dequant STEP1's OWN dispatched-dy
pool -> ``dl2`` [P,H] bf16, then per local expert ``grad_swiglu_ref = dl2 @ w2[g]`` -- isolates the
fork GEMM (dispatch-push correctness is trusted / covered elsewhere). SNR gate on the real
dispatched rows. Also times the STEP1 call.

Run inside the dev container (8 GPUs):
  PYTHONPATH=<repo> python benchmark/ops/bench_step1_dispatch_dgrad_fp8.py --num-processes 8 --num-tokens 8192
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
    quantize_grouped_weight_mxfp8,
)
from primus_turbo.pytorch.ops.moe.mega_moe_fused_fp8 import _mxfp8_step1_dispatch_dgrad

_H_GROUP_OFFS = 10


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

    # forward L1 first (match the real fwd->bwd order; also establishes symm/L2 state)
    w1q, w1s = quantize_grouped_weight_mxfp8(W1)
    torch.cuda.synchronize(); group.barrier()
    symm.scoreboard.zero_()
    torch.cuda.synchronize(); group.barrier()
    dispatch_grouped_gemm_mxfp8(x, None, w1q, w1s, handle, sym_layout, symm, BM=BM, BN=BN)

    dy = torch.randn((T, H), device="cuda", dtype=torch.bfloat16)

    def _step1():
        return _mxfp8_step1_dispatch_dgrad(dy, W2, group, handle, BM, BN)

    # correctness: STEP1 once, then per-group bf16 ref over STEP1's own dispatched-dy pool
    grad_swiglu, pool_handle = _step1()
    grad_swiglu = grad_swiglu.clone()
    pool_fp8, pool_scale = pool_handle
    P, Hh = pool_fp8.shape
    offs = handle[_H_GROUP_OFFS]
    c_m = int(offs[-1].item())
    pf = pool_fp8.to(torch.float32).view(P, Hh // 32, 32)
    ps = pool_scale.reshape(P, Hh // 32).view(torch.uint8).to(torch.int32)
    sc = torch.exp2((ps - 127).to(torch.float32)).view(P, Hh // 32, 1)
    dl2 = (pf * sc).view(P, Hh).to(torch.bfloat16)
    ref = torch.zeros_like(grad_swiglu)
    for g in range(epr):
        lo, hi = int(offs[g].item()), int(offs[g + 1].item())
        if hi > lo:
            ref[lo:hi] = (dl2[lo:hi].float() @ W2[g].float()).to(torch.bfloat16)
    snr = _snr_db(ref[:c_m], grad_swiglu[:c_m])
    nan = bool((~torch.isfinite(grad_swiglu.float())).any())

    t_step1 = _bench(_step1, warmup=args.warmup, iters=args.iters, group=group)
    M_eff = c_m
    flops = 2.0 * M_eff * I * H  # dgrad GEMM: [P,H] @ [H,I]
    symm.destroy()
    return {"snr": snr, "nan": float(nan), "t_step1": t_step1, "flops": flops, "M_eff": M_eff}


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
        snr, nan = _amin(group, r["snr"]), _amax(group, r["nan"])
        t_step1 = _amax(group, r["t_step1"])
        if rank == 0:
            tf = r["flops"] / (t_step1 * 1e-3) / 1e12
            print(f"\n{'='*72}")
            print(f"[backward STEP1  dispatch(dy)+fc2 dgrad  fp8]  EP{world} T={args.num_tokens} "
                  f"H={args.hidden} I={args.inter} E={args.num_experts} K={args.num_topk} BM={args.bm} BN={args.bn}")
            print(f"{'='*72}")
            print(f"  STEP1 fp8    : {t_step1:8.3f} ms | {tf:8.1f} TFLOPS  (M_eff={r['M_eff']})")
            print(f"  [acc] grad_swiglu fp8 vs per-group bf16 ref: SNR={snr:.2f} dB  nan={bool(nan)}  "
                  f"{'PASS' if snr >= 18.0 and not nan else 'FAIL'} (gate SNR>=18dB)")
        torch.cuda.synchronize(); group.barrier()
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="backward STEP1 (dispatch(dy)+fc2 dgrad) fp8 isolate")
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
