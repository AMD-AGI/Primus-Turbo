#!/usr/bin/env python3
"""Isolated latency for rowcol_dual_quant vs split rowwise/colwise on real grad_l1.

Run (8 GPUs, inside dev container):
  PYTHONPATH=<repo> python benchmark/ops/_bench_dual_quant.py \\
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
from primus_turbo.flydsl.mega.fp8 import (
    dispatch_grouped_gemm_mxfp8,
    dispatch_prologue,
    get_symm_buffer_for_mega_moe,
    quantize_grouped_weight_mxfp8,
)
from primus_turbo.flydsl.mega import swiglu_backward_flydsl_kernel
from primus_turbo.flydsl.mega.fp8.quant_colwise_trans_flydsl import (
    colwise_grouped_meta,
    colwise_quant_mxfp8_grouped_flydsl,
    rowcol_dual_quant_mxfp8_grouped_flydsl,
)
from primus_turbo.flydsl.mega.fp8.quant_flydsl import quantize_rowwise_mxfp8_flydsl
from primus_turbo.pytorch.kernels.mega_moe.mega_moe_backward_fp8_impl import (
    _DW_FP8_FORMAT,
    _dispatch_l2_dgrad_mxfp8_flydsl_kernel,
)


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
    return float(t.item())


def worker(local_rank, world, args):
    ip = os.getenv("MASTER_ADDR", "127.0.0.1")
    port = int(os.getenv("MASTER_PORT", "8600"))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        "nccl", init_method=f"tcp://{ip}:{port}", world_size=world, rank=local_rank,
        timeout=datetime.timedelta(seconds=180),
    )
    torch.set_default_device("cuda")
    group = dist.new_group(list(range(world)))
    rank = dist.get_rank()
    T, H, I, E, K = args.num_tokens, 7168, 2048, 256, 8
    epr = E // world
    BM, BN = 256, 256

    g = torch.Generator(device="cuda").manual_seed(100 + rank)
    scores = torch.rand(T, E, generator=g, device="cuda").abs() + 1
    topk_w, topk_idx = torch.topk(scores.softmax(-1), K, dim=-1)
    topk_idx, topk_w = topk_idx.to(torch.int64), topk_w.to(torch.float32)

    g2 = torch.Generator(device="cuda").manual_seed(1234)
    W1 = torch.randn((epr, 2 * I, H), generator=g2, device="cuda", dtype=torch.bfloat16) * (2.0 / math.sqrt(H))
    W2 = torch.randn((epr, H, I), generator=g2, device="cuda", dtype=torch.bfloat16) * (2.0 / math.sqrt(I))
    x = torch.randn((T, H), device="cuda", dtype=torch.bfloat16)
    dy = torch.randn((T, H), device="cuda", dtype=torch.bfloat16)

    symm = get_symm_buffer_for_mega_moe(
        group, num_experts=E, num_max_tokens_per_rank=T, num_topk=K, hidden=H,
        intermediate_hidden=I, block_m=BM, block_n=BN, use_mxfp8=True,
    )
    sl = symm.make_sym_layout()
    handle = tuple(dispatch_prologue(
        topk_idx, topk_w, sym_layout=sl, num_tokens=T, num_topk=K, num_experts=E,
        world_size=world, rank=symm.rank, experts_per_rank=epr, block_m=BM,
        num_max_pool_tokens=symm.num_max_pool_tokens,
    ))
    w1q, w1s = quantize_grouped_weight_mxfp8(W1)
    torch.cuda.synchronize()
    group.barrier()
    l1 = dispatch_grouped_gemm_mxfp8(x, None, w1q, w1s, handle, sl, symm, BM=BM, BN=BN)
    dw = symm.weight_recv_buf.clone()
    grad_swiglu, _ = _dispatch_l2_dgrad_mxfp8_flydsl_kernel(dy, W2, group, handle, BM, BN)
    grad_l1, _ = swiglu_backward_flydsl_kernel(
        grad_swiglu, l1, symm.meta_scalars[1:2], scale=dw, return_gate=True,
    )
    meta = colwise_grouped_meta(handle[9], handle[10])
    m_used = int(handle[10][-1].item())

    w, n = args.warmup, args.iters
    t_dual = _bench(
        lambda: rowcol_dual_quant_mxfp8_grouped_flydsl(grad_l1, _DW_FP8_FORMAT, meta=meta),
        warmup=w, iters=n, group=group,
    )
    t_row = _bench(
        lambda: quantize_rowwise_mxfp8_flydsl(grad_l1.contiguous(), preshuffle=True),
        warmup=w, iters=n, group=group,
    )
    t_col = _bench(
        lambda: colwise_quant_mxfp8_grouped_flydsl(grad_l1, _DW_FP8_FORMAT, meta=meta),
        warmup=w, iters=n, group=group,
    )
    t_split = _bench(
        lambda: (
            quantize_rowwise_mxfp8_flydsl(grad_l1.contiguous(), preshuffle=True),
            colwise_quant_mxfp8_grouped_flydsl(grad_l1, _DW_FP8_FORMAT, meta=meta),
        ),
        warmup=w, iters=n, group=group,
    )

    if rank == 0:
        print(f"[dual quant bench] EP{world} T={T}  grad_l1={tuple(grad_l1.shape)}  m_used={m_used}")
        print(f"  dual (1 read)     : {t_dual:7.3f} ms")
        print(f"  split rowwise     : {t_row:7.3f} ms")
        print(f"  split colwise     : {t_col:7.3f} ms")
        print(f"  split row+col (2x): {t_split:7.3f} ms  (delta vs dual {t_split - t_dual:+.3f})")

    symm.destroy()
    dist.destroy_process_group()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-processes", type=int, default=8)
    ap.add_argument("--num-tokens", type=int, default=8192)
    ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument("--iters", type=int, default=25)
    args = ap.parse_args()
    if args.num_processes == 1:
        worker(0, 1, args)
    else:
        torch.multiprocessing.spawn(worker, args=(args.num_processes, args), nprocs=args.num_processes)
