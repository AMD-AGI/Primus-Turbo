###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Backward STEP3 (fc1 dgrad + combine) fp8 -- smoke correctness + latency.

Replicates the backward up to STEP3 on real mega-pool data: forward L1 -> (l1, dispatch_weights),
STEP1 (dispatch(dy)+fc2 dgrad), STEP2 swiglu_backward -> grad_l1 + grad_gate. Then STEP3
(``_l1_dgrad_combine_mxfp8_flydsl_kernel``): fp8 fc1-dgrad + combine PUSH + unweighted reduce + gate
scatter -> dx [T, H] + grad_topk_weights [T, K].

SMOKE only: dx / grad_topk_weights finite + correctly shaped + runs cross-rank (no deadlock/NaN).
Rigorous dx SNR is validated later by the full-backward e2e gradcheck vs a torch dense-MoE autograd
reference -- an isolated bf16 dx reference would need a bf16 combine on the fp8 SymLayout stack (not
present; the fp8 subpackage carries no bf16 combine kernel).

Run inside the dev container (8 GPUs):
  PYTHONPATH=<repo> python benchmark/ops/bench_step3_fp8.py --num-processes 8 --num-tokens 8192
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
from primus_turbo.pytorch.kernels.mega_moe.mega_moe_backward_fp8_impl import (
    _dispatch_l2_dgrad_mxfp8_flydsl_kernel,
    _l1_dgrad_combine_mxfp8_flydsl_kernel,
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


def _bench(fn, *, warmup, iters, group):
    # BACK-TO-BACK timing (host custom-op dispatch overlaps GPU); single-call event-bracket timing
    # would count host dispatch/autotune-lookup as GPU-idle and inflate fast fp8 kernels.
    torch.cuda.synchronize(); group.barrier()
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize(); group.barrier()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record(); torch.cuda.synchronize()
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
    torch.cuda.synchronize(); group.barrier()
    symm.scoreboard.zero_()
    torch.cuda.synchronize(); group.barrier()
    l1 = dispatch_grouped_gemm_mxfp8(x, None, w1q, w1s, handle, sym_layout, symm, BM=BM, BN=BN)
    dispatch_weights = symm.weight_recv_buf.clone()

    dy = torch.randn((T, H), device="cuda", dtype=torch.bfloat16)
    grad_swiglu, _ = _dispatch_l2_dgrad_mxfp8_flydsl_kernel(dy, W2, group, handle, BM, BN)
    grad_l1, grad_gate = swiglu_backward_flydsl_kernel(grad_swiglu, l1, get_symm_buffer_for_mega_moe().meta_scalars[1:2], scale=dispatch_weights, return_gate=True)

    def _step3():
        return _l1_dgrad_combine_mxfp8_flydsl_kernel(
            grad_l1, W1, group, handle, BM, BN, grad_gate=grad_gate, topk_idx=topk_idx,
            num_tokens=T, num_topk=K,
        )

    dx, grad_topk_weights = _step3()
    dx_ok = tuple(dx.shape) == (T, H) and bool(torch.isfinite(dx.float()).all())
    dtw_ok = tuple(grad_topk_weights.shape) == (T, K) and bool(torch.isfinite(grad_topk_weights.float()).all())
    dx_norm = float(dx.float().norm())

    t_step3 = _bench(_step3, warmup=args.warmup, iters=args.iters, group=group)
    m_pad = int(handle[10][-1].item())
    flops = 2.0 * m_pad * (2 * I) * H  # fc1 dgrad GEMM: [P,2I] @ [2I,H]
    symm.destroy()
    return {"dx_ok": float(dx_ok), "dtw_ok": float(dtw_ok), "dx_norm": dx_norm,
            "t_step3": t_step3, "flops": flops, "m_pad": m_pad}


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
        dx_ok, dtw_ok = _amin(group, r["dx_ok"]), _amin(group, r["dtw_ok"])
        dx_norm = _amax(group, r["dx_norm"])
        t_step3 = _amax(group, r["t_step3"])
        if rank == 0:
            tf = r["flops"] / (t_step3 * 1e-3) / 1e12
            ok = dx_ok >= 1.0 and dtw_ok >= 1.0
            print(f"\n{'='*72}")
            print(f"[backward STEP3  fc1 dgrad+combine  fp8]  EP{world} T={args.num_tokens} "
                  f"H={args.hidden} I={args.inter} E={args.num_experts} K={args.num_topk}")
            print(f"{'='*72}")
            print(f"  STEP3 fp8    : {t_step3:8.3f} ms | {tf:8.1f} TFLOPS  (M_pool={r['m_pad']})")
            print(f"  [smoke] dx [T,H] finite={bool(dx_ok>=1.0)} (norm={dx_norm:.3e}) | "
                  f"grad_topk [T,K] finite={bool(dtw_ok>=1.0)}  {'PASS' if ok else 'FAIL'} "
                  f"(rigorous dx SNR -> e2e backward gradcheck)")
        torch.cuda.synchronize(); group.barrier()
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="backward STEP3 (fc1 dgrad + combine) fp8 smoke")
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
