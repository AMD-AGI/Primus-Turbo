#!/usr/bin/env python3
"""L1 (dispatch+fc1) serial-prefix + fused-kernel decomposition (EP8, DSv3).

Measures on one stream (no side-stream overlap):
  prologue | fused [x quant + comm | preshuffle | GEMM] | full L1 (flydsl entry)

The fused kernel legs (comm / preshuffle / GEMM) overlap inside one grid; isolate with env:
  PT_DISPATCH_PUSH_ONLY=1  comm PUSH only (preshuffle+gemm idle)
  PT_DISPATCH_GEMM_ONLY=1  preshuffle+GEMM only (comm idle)
Either isolation makes the output WRONG; they are timing probes only.

Run (8 GPUs):
  PYTHONPATH=<repo> python benchmark/ops/bench_l1_decomp_fp8.py \\
    --num-processes 8 --num-tokens 8192 --warmup 10 --iters 30
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
    dispatch_grouped_gemm_mxfp8_flydsl_kernel,
    dispatch_prologue,
    get_symm_buffer_for_mega_moe,
    quantize_rowwise_mxfp8_flydsl,
)
from primus_turbo.pytorch.kernels.mega_moe.mega_moe_forward_fp8_impl import _w1_fp8_cached


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
    port = int(os.getenv("MASTER_PORT", "8496"))
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
    DC, PC = args.dispatch_cu, args.preshuffle_cu
    epr = E // world

    torch.manual_seed(7 + rank)
    x = torch.randn((T, H), device="cuda", dtype=torch.bfloat16)
    g = torch.Generator(device="cuda").manual_seed(100 + rank)
    scores = torch.rand(T, E, generator=g, device="cuda").abs() + 1
    topk_w, topk_idx = torch.topk(scores.softmax(-1), K, dim=-1)
    topk_w, topk_idx = topk_w.float(), topk_idx.long()
    W1g = torch.randn((E, 2 * I, H), device="cuda", dtype=torch.bfloat16) * (2.0 / math.sqrt(H))
    W1 = W1g[rank * epr : (rank + 1) * epr].contiguous()
    w1q, w1s = _w1_fp8_cached(W1)

    symm = get_symm_buffer_for_mega_moe(
        group, num_experts=E, num_max_tokens_per_rank=T, num_topk=K, hidden=H,
        intermediate_hidden=I, block_m=BM, block_n=BN, use_mxfp8=True,
    )
    sym_layout = symm.make_sym_layout()

    def _prologue():
        return tuple(
            dispatch_prologue(
                topk_idx, topk_w, sym_layout=sym_layout, num_tokens=T, num_topk=K,
                num_experts=E, world_size=world, rank=rank, experts_per_rank=epr,
                block_m=BM, num_max_pool_tokens=symm.num_max_pool_tokens,
            )
        )

    handle = _prologue()
    torch.cuda.synchronize()
    group.barrier()

    r = {}
    r["prologue"] = _bench(_prologue, warmup=args.warmup, iters=args.iters, group=group)
    r["x_quant"] = _bench(
        lambda: quantize_rowwise_mxfp8_flydsl(x),
        warmup=args.warmup, iters=args.iters, group=group,
    )

    def _fused():
        dispatch_grouped_gemm_mxfp8(
            x, w1q, w1s, handle, sym_layout, symm,
            num_dispatch_cu=DC, num_preshuffle_cu=PC, BM=BM, BN=BN,
        )

    r["fused_kernel"] = _bench(_fused, warmup=args.warmup, iters=args.iters, group=group)

    def _full_l1():
        dispatch_grouped_gemm_mxfp8_flydsl_kernel(
            x, w1q, w1s, group, topk_idx=topk_idx, topk_weights=topk_w,
            num_dispatch_cu=DC, num_preshuffle_cu=PC, BM=BM, BN=BN,
        )

    r["full_l1_flydsl"] = _bench(_full_l1, warmup=args.warmup, iters=args.iters, group=group)

    for leg, env_var in (("push_only", "PT_DISPATCH_PUSH_ONLY"), ("gemm_only", "PT_DISPATCH_GEMM_ONLY")):
        os.environ[env_var] = "1"
        try:
            r[leg] = _bench(_fused, warmup=max(4, args.warmup // 2), iters=max(10, args.iters // 2), group=group)
        except Exception as exc:
            r[leg] = float("nan")
            if rank == 0:
                print(f"  [warn] {leg} failed: {exc}")
        finally:
            os.environ.pop(env_var, None)

    if rank == 0:
        serial_sum = r["prologue"] + r["fused_kernel"]
        print(f"\n{'=' * 72}")
        print(f"[L1 decomp fp8]  EP{world} T={T} H={H} I={I}  dispatch/ps CU={DC}/{PC}  (max rank ms)")
        print(f"{'=' * 72}")
        print(f"  {'prologue':<22} {r['prologue']:7.3f} ms")
        print(f"  {'fused kernel':<22} {r['fused_kernel']:7.3f} ms  (x quant + comm|preshuffle|GEMM)")
        print(f"  {'  of which x quant':<22} {r['x_quant']:7.3f} ms  (rowwise mxfp8, host launch)")
        print(f"  {'serial sum (above)':<22} {serial_sum:7.3f} ms")
        print(f"  {'full L1 (flydsl entry)':<22} {r['full_l1_flydsl']:7.3f} ms  (prologue+quant+kernel / call)")
        print(f"  gap full - serial_sum   {r['full_l1_flydsl'] - serial_sum:+.3f} ms")
        if not math.isnan(r.get("push_only", float("nan"))):
            print(f"\n  --- in-grid isolation (env) ---")
            print(f"  {'PUSH_ONLY (comm)':<22} {r['push_only']:7.3f} ms")
            print(f"  {'GEMM_ONLY (ps+gemm)':<22} {r['gemm_only']:7.3f} ms")

    symm.destroy()
    dist.destroy_process_group()


def main():
    ap = argparse.ArgumentParser(description="L1 dispatch+fc1 decomposition")
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
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=30)
    args = ap.parse_args()
    torch.multiprocessing.spawn(worker, args=(args.num_processes, args), nprocs=args.num_processes)


if __name__ == "__main__":
    main()
