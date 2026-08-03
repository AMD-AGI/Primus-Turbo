#!/usr/bin/env python3
"""Per-stage fp8 mega MoE forward breakdown (EP8, DSv3 shapes).

Times each leg of ``mega_moe_forward_fp8_impl`` in isolation after shared L1 setup:
  L1 dispatch+fc1 | SwiGLU bf16 | mxfp8 quant | SwiGLU+quant fused | w2 prep | L2 combine
plus FULL op (no-grad + autograd leaves) for cross-check.

Run (8 GPUs):
  PYTHONPATH=<repo> python benchmark/ops/bench_fwd_breakdown_fp8.py \\
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
from primus_turbo.flydsl.mega import swiglu_flydsl_kernel
from primus_turbo.flydsl.mega.fp8 import (
    dispatch_grouped_gemm_mxfp8_flydsl_kernel,
    grouped_gemm_combine_mxfp8_flydsl_kernel,
    quantize_rowwise_mxfp8_flydsl,
    swiglu_mxfp8_flydsl_kernel,
)
from primus_turbo.pytorch.kernels.fused_mega_moe.weight_prep_fp8 import prepare_w2_fp8
from primus_turbo.pytorch.ops.moe.mega_moe_fused_fp8 import mega_moe_fused_fp8


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
    port = int(os.getenv("MASTER_PORT", "8494"))
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

    # persistent leaves for FULL autograd timing (matches bench_mega_moe_bwd_only)
    x_nograd, tw_nograd = x, topk_w
    xL = x.detach().clone().requires_grad_(True)
    twL = topk_w.detach().clone().requires_grad_(True)
    W1L = W1.detach().clone().requires_grad_(True)
    W2L = W2.detach().clone().requires_grad_(True)

    from primus_turbo.pytorch.kernels.fused_mega_moe.mega_moe_forward_fp8_impl import (
        _w1_fp8_cached,
        _w2_fp8_cached,
    )

    w1q, w1s = _w1_fp8_cached(W1L)
    w2_fp8 = prepare_w2_fp8(W2)  # one-shot prep (cached on W2L in FULL)

    def _l1():
        return dispatch_grouped_gemm_mxfp8_flydsl_kernel(
            x, w1q, w1s, group, topk_idx=topk_idx, topk_weights=topk_w,
            num_dispatch_cu=DC, num_preshuffle_cu=PC, BM=BM, BN=BN,
        )

    # warmup L1 once for downstream stages (uses live symm)
    l1, handle, _, _ = _l1()
    ntb = handle[11]
    torch.cuda.synchronize()
    group.barrier()

    act_fp8, act_a_sp = swiglu_mxfp8_flydsl_kernel(l1, ntb)
    act_bf16 = swiglu_flydsl_kernel(l1, ntb)
    torch.cuda.synchronize()
    group.barrier()

    r = {}
    r["L1_dispatch_fc1"] = _bench(_l1, warmup=args.warmup, iters=args.iters, group=group)
    r["SwiGLU_bf16"] = _bench(lambda: swiglu_flydsl_kernel(l1, ntb), warmup=args.warmup, iters=args.iters, group=group)
    r["quant_rowwise"] = _bench(
        lambda: quantize_rowwise_mxfp8_flydsl(act_bf16, preshuffle=True),
        warmup=args.warmup, iters=args.iters, group=group,
    )
    r["SwiGLU+quant_fused"] = _bench(
        lambda: swiglu_mxfp8_flydsl_kernel(l1, ntb),
        warmup=args.warmup, iters=args.iters, group=group,
    )
    r["w2_prep_cached"] = _bench(lambda: _w2_fp8_cached(W2L), warmup=args.warmup, iters=args.iters, group=group)
    r["L2_combine_xfp8"] = _bench(
        lambda: grouped_gemm_combine_mxfp8_flydsl_kernel(
            None, w2_fp8, list(handle), group,
            topk_indices=topk_idx, topk_weights=topk_w,
            x_fp8=(act_fp8, act_a_sp), BM=BM, BN=BN, num_combine_cu=CC,
        )[0],
        warmup=args.warmup, iters=args.iters, group=group,
    )
    r["FULL_no_grad"] = _bench(
        lambda: mega_moe_fused_fp8(group, x_nograd, topk_idx, tw_nograd, W1, W2),
        warmup=args.warmup, iters=args.iters, group=group,
    )
    r["FULL_autograd"] = _bench(
        lambda: mega_moe_fused_fp8(group, xL, topk_idx, twL, W1L, W2L),
        warmup=args.warmup, iters=args.iters, group=group,
    )

    if rank == 0:
        w = 22
        print(f"\n{'=' * 72}")
        print(f"[fp8 fwd breakdown]  EP{world} T={T} H={H} I={I}  L1={DC}/{PC} L2 cc={CC}  (max rank ms)")
        print(f"{'=' * 72}")
        for k in (
            "L1_dispatch_fc1", "SwiGLU_bf16", "quant_rowwise", "SwiGLU+quant_fused",
            "w2_prep_cached", "L2_combine_xfp8",
            "FULL_no_grad", "FULL_autograd",
        ):
            print(f"  {k:<22} {r[k]:7.3f} ms")

        sum_iso = r["L1_dispatch_fc1"] + r["SwiGLU+quant_fused"] + r["L2_combine_xfp8"]
        unfused_prep = r["SwiGLU_bf16"] + r["quant_rowwise"]
        print(f"\n  --- composition ---")
        print(f"  L1 + fused_swiglu + L2(x_fp8)     = {sum_iso:.3f} ms  (isolated sum)")
        print(f"  FULL no-grad (op)                 = {r['FULL_no_grad']:.3f} ms")
        print(f"  FULL autograd (bwd_only fwd col)  = {r['FULL_autograd']:.3f} ms")
        print(f"  autograd overhead                 = {r['FULL_autograd'] - r['FULL_no_grad']:+.3f} ms")
        print(f"  swiglu_bf16 + quant (unfused)     = {unfused_prep:.3f} ms")
        print(f"  swiglu_mxfp8 fused                = {r['SwiGLU+quant_fused']:.3f} ms  "
              f"(delta {r['SwiGLU+quant_fused'] - unfused_prep:+.3f} vs unfused)")
        print(f"  swiglu as %% of FULL no-grad      = {100.0 * r['SwiGLU+quant_fused'] / r['FULL_no_grad']:.1f}%")
        print(f"  L1 as %% of FULL no-grad          = {100.0 * r['L1_dispatch_fc1'] / r['FULL_no_grad']:.1f}%")
        print(f"  L2 as %% of FULL no-grad          = {100.0 * r['L2_combine_xfp8'] / r['FULL_no_grad']:.1f}%")
        gap = r["FULL_no_grad"] - sum_iso
        print(f"  FULL - isolated_sum gap           = {gap:+.3f} ms  (op overhead / L1 re-prologue)")

    dist.destroy_process_group()


def main():
    ap = argparse.ArgumentParser(description="fp8 mega MoE forward per-stage breakdown")
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
