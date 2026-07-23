###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""End-to-end forward+BACKWARD validation for the fp8 mega MoE op.

Runs the full fp8 autograd path (forward L1 dispatch+fc1 -> SwiGLU -> L2 combine; backward STEP1
dispatch(dy)+fc2 dgrad -> SwiGLU^T -> dW2 -> STEP3 fc1 dgrad+combine -> dW1) and grad-checks every
gradient (dx, d_topk_w, dW1, dW2) against the bf16 ``mega_moe_fused`` reference on identical inputs,
with an SNR gate. Reports rough fwd+bwd latency.

NOTE: the STEP1 dispatch(dy) and STEP3 combine gates self-reset via a device epoch (no host
rendezvous), which removed the old large-T reset-race stall -- validated stable through T=8192.

Run inside the dev container (8 GPUs):
  PYTHONPATH=<repo> python benchmark/ops/bench_mega_moe_fused_fp8_bwd.py --num-processes 8 --num-tokens 2048
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
    return float(10.0 * torch.log10(ref.pow(2).sum() / ((ref - out).pow(2).sum() + 1e-12)))


def _leaf(t):
    return t.detach().clone().requires_grad_(True)


def _run_once(fp8, group, x, topk_idx, topk_w, W1, W2, grad_y):
    """One fwd+bwd; returns (y, dx, d_topk_w, dW1, dW2) with fresh leaf inputs."""
    xL, twL, W1L, W2L = _leaf(x), _leaf(topk_w), _leaf(W1), _leaf(W2)
    op = mega_moe_fused_fp8 if fp8 else mega_moe_fused
    y = op(group, xL, topk_idx, twL, W1L, W2L)
    y.backward(grad_y)
    return y.detach(), xL.grad, twL.grad, W1L.grad, W2L.grad


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


def profile(group, args):
    rank, world = group.rank(), group.size()
    H, I, E, K, T = args.hidden, args.inter, args.num_experts, args.num_topk, args.num_tokens
    epr = E // world

    torch.manual_seed(7 + rank)
    x = torch.randn((T, H), device="cuda", dtype=torch.bfloat16)
    topk_idx, topk_w = _routing(T, K, E, device="cuda", seed=100 + rank)
    W1g, W2g = _global_weights(E, I, H, "cuda")
    W1 = W1g[rank * epr : (rank + 1) * epr].contiguous()
    W2 = W2g[rank * epr : (rank + 1) * epr].contiguous()
    del W1g, W2g
    grad_y = torch.randn((T, H), device="cuda", dtype=torch.bfloat16)

    # fp8 first (isolation): a NaN/hang here can't be blamed on bf16-op interference.
    y8, dx8, dtw8, dW18, dW28 = _run_once(True, group, x, topk_idx, topk_w, W1, W2, grad_y)
    fin = all(bool(torch.isfinite(t.float()).all()) for t in (y8, dx8, dtw8, dW18, dW28))
    shapes_ok = (
        tuple(dx8.shape) == (T, H) and tuple(dtw8.shape) == (T, K)
        and tuple(dW18.shape) == W1.shape and tuple(dW28.shape) == W2.shape
    )
    if args.only == "fp8":
        return {"fin": float(fin), "shapes": float(shapes_ok),
                "snr_dx": 0.0, "snr_dtw": 0.0, "snr_dw1": 0.0, "snr_dw2": 0.0,
                "t_fp8": 1.0, "t_bf16": 1.0}

    _yb, dxb, dtwb, dW1b, dW2b = _run_once(False, group, x, topk_idx, topk_w, W1, W2, grad_y)
    snr = {
        "snr_dx": _snr_db(dxb, dx8), "snr_dtw": _snr_db(dtwb, dtw8),
        "snr_dw1": _snr_db(dW1b, dW18), "snr_dw2": _snr_db(dW2b, dW28),
    }

    # PERSISTENT leaf weights (created ONCE) so the op's version-keyed weight-quant caches HIT across
    # timing iters -- mirrors real training (weights change only on optim.step; the fp8 preps are
    # reused across a grad-accum window). Fresh-leaf-per-iter would re-quantize every weight each
    # call (a bench artifact that hugely inflates the fp8 time). x/topk_w persist too for stable timing.
    xf, twf, W1f, W2f = _leaf(x), _leaf(topk_w), _leaf(W1), _leaf(W2)
    xb, twb, W1b, W2b = _leaf(x), _leaf(topk_w), _leaf(W1), _leaf(W2)

    def _fwd_bwd_fp8():
        for t in (xf, twf, W1f, W2f):
            t.grad = None
        mega_moe_fused_fp8(group, xf, topk_idx, twf, W1f, W2f).backward(grad_y)

    def _fwd_bwd_bf16():
        for t in (xb, twb, W1b, W2b):
            t.grad = None
        mega_moe_fused(group, xb, topk_idx, twb, W1b, W2b).backward(grad_y)

    # time each fully (its own warmup absorbs the use_mxfp8<->bf16 symm realloc on first call)
    t_fp8 = _bench(_fwd_bwd_fp8, warmup=args.warmup, iters=args.iters, group=group)
    t_bf16 = _bench(_fwd_bwd_bf16, warmup=args.warmup, iters=args.iters, group=group)
    return {"fin": float(fin), "shapes": float(shapes_ok), "t_fp8": t_fp8, "t_bf16": t_bf16, **snr}


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
        fin, shapes = _amin(group, r["fin"]), _amin(group, r["shapes"])
        snr_dx = _amin(group, r["snr_dx"]); snr_dtw = _amin(group, r["snr_dtw"])
        snr_dw1 = _amin(group, r["snr_dw1"]); snr_dw2 = _amin(group, r["snr_dw2"])
        t_fp8 = _amax(group, r["t_fp8"]); t_bf16 = _amax(group, r["t_bf16"])
        if rank == 0:
            print(f"\n{'='*72}")
            print(f"[mega MoE fwd+bwd  fp8 e2e]  EP{world} T={args.num_tokens} H={args.hidden} "
                  f"I={args.inter} E={args.num_experts} K={args.num_topk}")
            print(f"{'='*72}")
            print(f"  [smoke] all grads finite={bool(fin>=1.0)}  shapes_ok={bool(shapes>=1.0)}")
            if args.only != "fp8":
                print(f"  bf16 fwd+bwd : {t_bf16:8.3f} ms")
                print(f"  fp8  fwd+bwd : {t_fp8:8.3f} ms | {t_bf16 / t_fp8:.2f}x vs bf16")
                gate = 15.0
                ok = min(snr_dx, snr_dtw, snr_dw1, snr_dw2) >= gate
                print(f"  [grad SNR vs bf16]  dx={snr_dx:.1f}  d_topk_w={snr_dtw:.1f}  "
                      f"dW1={snr_dw1:.1f}  dW2={snr_dw2:.1f} dB  "
                      f"{'PASS' if ok else 'FAIL'} (gate>={gate:.0f}dB)")
            else:
                print(f"  fp8  fwd+bwd : {t_fp8:8.3f} ms")
        torch.cuda.synchronize(); group.barrier()
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="mega MoE fwd+bwd fp8 e2e gradcheck vs bf16")
    ap.add_argument("--num-processes", type=int, default=8)
    ap.add_argument("--hidden", type=int, default=7168)
    ap.add_argument("--inter", type=int, default=2048)
    ap.add_argument("--num-experts", type=int, default=256)
    ap.add_argument("--num-topk", type=int, default=8)
    ap.add_argument("--num-tokens", type=int, default=2048)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--only", choices=["both", "fp8"], default="both")
    args = ap.parse_args()
    torch.multiprocessing.spawn(worker, args=(args.num_processes, args), nprocs=args.num_processes)
