###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Backward STEP1 (dispatch(dy) + fc2 dgrad) -- fp8 vs bf16 latency, apples-to-apples.

STEP1 is the *fused* cross-rank dispatch(dy) PUSH + grouped fc2-dgrad GEMM (grad_swiglu =
dispatched_dy @ w2, NN, contract H -> [P, I]). Both legs run the SAME fused op:
  * fp8 : ``_dispatch_l2_dgrad_mxfp8_flydsl_kernel`` (fp8 PUSH, byte-halved comm + mxfp8 ~2x-compute GEMM),
          on the fp8 mega stack (SymLayout + two-heap symm).
  * bf16: ``dispatch_grouped_gemm_bf16_flydsl_kernel(dy, w2, handle=h, layout="nn")`` -- exactly
          the L2-dgrad the bf16 ``mega_moe_fused`` backward uses (bf16 PUSH + bf16 GEMM), on the
          bf16 mega stack.
The two stacks are separate symm globals; we run fp8 fully, ``destroy()`` it, then run bf16 (no
same-process coexistence). fp8 correctness is spot-checked vs a per-group torch ref (SNR).

Run inside the dev container (8 GPUs):
  PYTHONPATH=<repo> python benchmark/ops/bench_step1_fp8_vs_bf16.py --num-processes 8 --num-tokens 8192
"""

import argparse
import datetime
import math
import os

import numpy as np
import torch
import torch.distributed as dist

import primus_turbo.pytorch  # noqa: F401
from primus_turbo.flydsl.mega import dispatch_grouped_gemm_bf16_flydsl_kernel
from primus_turbo.flydsl.mega.fp8 import (
    dispatch_grouped_gemm_mxfp8,
    dispatch_prologue,
    get_symm_buffer_for_mega_moe,
    quantize_grouped_weight_mxfp8,
)
from primus_turbo.pytorch.kernels.mega_moe.mega_moe_backward_fp8_impl import _dispatch_l2_dgrad_mxfp8_flydsl_kernel

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
    dy = torch.randn((T, H), device="cuda", dtype=torch.bfloat16)

    # ─────────────── fp8 STEP1 (fp8 mega stack) ───────────────
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
    dispatch_grouped_gemm_mxfp8(x, None, w1q, w1s, handle, sym_layout, symm, BM=BM, BN=BN)

    def _fp8():
        return _dispatch_l2_dgrad_mxfp8_flydsl_kernel(dy, W2, group, handle, BM, BN)

    grad_swiglu_fp8, pool_handle = _fp8()
    grad_swiglu_fp8 = grad_swiglu_fp8.clone()
    offs = handle[_H_GROUP_OFFS]
    c_m = int(offs[-1].item())
    # per-group bf16 ref over STEP1's OWN dispatched-dy pool -> fp8 grad_swiglu correctness
    pool_fp8, pool_scale = pool_handle
    P, Hh = pool_fp8.shape
    pf = pool_fp8.to(torch.float32).view(P, Hh // 32, 32)
    ps = pool_scale.reshape(P, Hh // 32).view(torch.uint8).to(torch.int32)
    sc = torch.exp2((ps - 127).to(torch.float32)).view(P, Hh // 32, 1)
    dl2 = (pf * sc).view(P, Hh).to(torch.bfloat16)
    ref = torch.zeros_like(grad_swiglu_fp8)
    for g in range(epr):
        lo, hi = int(offs[g].item()), int(offs[g + 1].item())
        if hi > lo:
            ref[lo:hi] = (dl2[lo:hi].float() @ W2[g].float()).to(torch.bfloat16)
    snr = _snr_db(ref[:c_m], grad_swiglu_fp8[:c_m])
    nan = bool((~torch.isfinite(grad_swiglu_fp8.float())).any())

    t_fp8 = _bench(_fp8, warmup=args.warmup, iters=args.iters, group=group)
    M_eff = c_m
    symm.destroy()  # free the fp8 symm before building the bf16 stack (no coexistence)
    torch.cuda.synchronize(); group.barrier()

    # ─────────────── bf16 STEP1 (bf16 mega stack) ───────────────
    # bf16 forward dispatch (nt, handle=None auto-prologue) -> bf16 handle for the nn dgrad reuse.
    _l1b, _, _dwb, hbf = dispatch_grouped_gemm_bf16_flydsl_kernel(
        x, W1, group, handle=None, topk_idx=topk_idx, topk_weights=topk_w, layout="nt", BM=BM, BN=BN,
    )

    def _bf16():  # L2 dgrad = dispatch(dy) PUSH + grouped fc2-dgrad GEMM (exactly the bf16 bwd step)
        return dispatch_grouped_gemm_bf16_flydsl_kernel(dy, W2, group, handle=hbf, layout="nn", BM=BM, BN=BN)

    _ = _bf16()
    t_bf16 = _bench(_bf16, warmup=args.warmup, iters=args.iters, group=group)

    flops = 2.0 * M_eff * I * H  # dgrad GEMM [P,H] @ [H,I]
    return {"snr": snr, "nan": float(nan), "t_fp8": t_fp8, "t_bf16": t_bf16, "flops": flops, "M_eff": M_eff}


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
        t_fp8, t_bf16 = _amax(group, r["t_fp8"]), _amax(group, r["t_bf16"])
        if rank == 0:
            tf = lambda ms: r["flops"] / (ms * 1e-3) / 1e12
            print(f"\n{'='*72}")
            print(f"[backward STEP1  dispatch(dy)+fc2 dgrad  fp8 vs bf16]  EP{world} T={args.num_tokens} "
                  f"H={args.hidden} I={args.inter} E={args.num_experts} K={args.num_topk}")
            print(f"{'='*72}")
            print(f"  STEP1 fp8    : {t_fp8:8.3f} ms | {tf(t_fp8):8.1f} TFLOPS  (M_pool={r['M_eff']})")
            print(f"  STEP1 bf16   : {t_bf16:8.3f} ms | {tf(t_bf16):8.1f} TFLOPS")
            print(f"  fp8/bf16     : {t_fp8 / t_bf16:.3f}x  ({'fp8 faster' if t_fp8 < t_bf16 else 'fp8 SLOWER'})  "
                  f"[fused dispatch PUSH + fc2-dgrad GEMM; fp8 halves PUSH bytes + ~2x GEMM]")
            print(f"  [acc] grad_swiglu fp8 vs per-group bf16 ref: SNR={snr:.2f} dB  nan={bool(nan)}  "
                  f"{'PASS' if snr >= 18.0 and not nan else 'FAIL'} (gate SNR>=18dB)")
        torch.cuda.synchronize(); group.barrier()
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="backward STEP1 (dispatch(dy)+fc2 dgrad) fp8 vs bf16")
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
