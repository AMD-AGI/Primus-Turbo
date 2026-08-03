###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""End-to-end forward+BACKWARD validation for the fp8 mega MoE op.

Runs the full fp8 autograd path (forward L1 dispatch+fc1 -> SwiGLU -> L2 combine; backward STEP1
dispatch(dy)+fc2 dgrad -> SwiGLU^T -> dW2 -> STEP3 fc1 dgrad+combine -> dW1) and grad-checks the
forward output and every gradient (dx, d_topk_w, dW1, dW2) against an ANALYTIC reference, with an
SNR gate. Reports rough fwd+bwd latency against the bf16 op.

The reference is analytic, NOT the bf16 ``fused_mega_moe`` op, because the bf16 backward combine
loses a race that inflates a handful of dx rows by 4-9x on a run-to-run varying set of ranks. That
made the old bf16-referenced dx gate report 3-19 dB at random while the fp8 path itself measures a
flat 21.9 dB against truth. bf16 is still used for the LATENCY comparison, which the race does not
affect. Expect roughly 19-23 dB: that is the mxfp8 E5M2 gradient encoding floor, not slack.

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
import torch.nn.functional as F

import primus_turbo.pytorch  # noqa: F401
from primus_turbo.pytorch.ops.moe.fused_mega_moe import fused_mega_moe
from primus_turbo.pytorch.ops.moe.fused_mega_moe_fp8 import fused_mega_moe_fp8

_ACT_CLAMP = 10.0  # matches ACTIVATION_CLAMP in flydsl/mega/swiglu_kernel.py


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
    op = fused_mega_moe_fp8 if fp8 else fused_mega_moe
    y = op(group, xL, topk_idx, twL, W1L, W2L)
    y.backward(grad_y)
    return y.detach(), xL.grad, twL.grad, W1L.grad, W2L.grad


def _act(l1, I):
    """SwiGLU exactly as the kernel does it: gate is the FIRST I of the 2I columns, both clamped."""
    gate = l1[:, :I].float().clamp(-_ACT_CLAMP, _ACT_CLAMP)
    up = l1[:, I:].float().clamp(-_ACT_CLAMP, _ACT_CLAMP)
    return (F.silu(gate) * up).to(torch.bfloat16)


def _expert_rows(topk_idx, e):
    """(token ids, flat slot ids) of the top-k slots routed to expert `e`."""
    flat = topk_idx.reshape(-1)
    sel = (flat == e).nonzero(as_tuple=True)[0]
    return sel // topk_idx.shape[1], sel


def _truth_local(x, topk_idx, topk_w, grad_y, W1g, W2g, I):
    """Analytic y, dx and d_topk_w for THIS rank's tokens, through the full global expert set.

    One backward per expert, so only a single expert's activations are ever live.
    """
    xs = x.detach().clone().requires_grad_(True)
    tw = topk_w.detach().clone().requires_grad_(True)
    y = torch.zeros(x.shape[0], x.shape[1], dtype=torch.float32, device=x.device)
    for e in torch.unique(topk_idx).tolist():
        toks, sel = _expert_rows(topk_idx, e)
        o = (_act(xs[toks] @ W1g[e].T, I) @ W2g[e].T).float() * tw.reshape(-1)[sel].unsqueeze(1)
        y.index_add_(0, toks, o.detach())
        (o * grad_y[toks].float()).sum().backward()
    return y, xs.grad.detach().float(), tw.grad.detach().float()


def _truth_weights(xg, idxg, twg, dyg, W1g, W2g, I, lo, hi):
    """Analytic dW1/dW2 for this rank's LOCAL experts [lo, hi), fed by tokens from EVERY rank."""
    W1l = W1g[lo:hi].detach().clone().requires_grad_(True)
    W2l = W2g[lo:hi].detach().clone().requires_grad_(True)
    for e in range(lo, hi):
        toks, sel = _expert_rows(idxg, e)
        if toks.numel() == 0:
            continue
        o = (_act(xg[toks] @ W1l[e - lo].T, I) @ W2l[e - lo].T).float()
        (o * twg.reshape(-1)[sel].unsqueeze(1).float() * dyg[toks].float()).sum().backward()
    return W1l.grad.detach().float(), W2l.grad.detach().float()


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
    lo, hi = rank * epr, (rank + 1) * epr
    W1 = W1g[lo:hi].contiguous()
    W2 = W2g[lo:hi].contiguous()
    grad_y = torch.randn((T, H), device="cuda", dtype=torch.bfloat16)

    # fp8 first (isolation): a NaN/hang here can't be blamed on bf16-op interference.
    y8, dx8, dtw8, dW18, dW28 = _run_once(True, group, x, topk_idx, topk_w, W1, W2, grad_y)
    fin = all(bool(torch.isfinite(t.float()).all()) for t in (y8, dx8, dtw8, dW18, dW28))
    shapes_ok = (
        tuple(dx8.shape) == (T, H) and tuple(dtw8.shape) == (T, K)
        and tuple(dW18.shape) == W1.shape and tuple(dW28.shape) == W2.shape
    )
    if args.only == "fp8":
        return {"fin": float(fin), "shapes": float(shapes_ok), "snr_y": 0.0,
                "snr_dx": 0.0, "snr_dtw": 0.0, "snr_dw1": 0.0, "snr_dw2": 0.0,
                "t_fp8": 1.0, "t_bf16": 1.0}

    # Analytic reference. dx/d_topk_w need this rank's tokens through every expert; dW1/dW2 need
    # this rank's experts fed by every rank's tokens, hence the all-gather.
    y_t, dx_t, dtw_t = _truth_local(x, topk_idx, topk_w, grad_y, W1g, W2g, I)

    def _ag(t):
        out = [torch.empty_like(t) for _ in range(world)]
        dist.all_gather(out, t.contiguous(), group=group)
        return torch.cat(out, dim=0)

    dW1_t, dW2_t = _truth_weights(_ag(x), _ag(topk_idx), _ag(topk_w), _ag(grad_y), W1g, W2g, I, lo, hi)
    snr = {
        "snr_y": _snr_db(y_t, y8), "snr_dx": _snr_db(dx_t, dx8), "snr_dtw": _snr_db(dtw_t, dtw8),
        "snr_dw1": _snr_db(dW1_t, dW18), "snr_dw2": _snr_db(dW2_t, dW28),
    }
    # free the reference before the timing loops allocate their own leaves
    del W1g, W2g, y_t, dx_t, dtw_t, dW1_t, dW2_t, y8, dx8, dtw8, dW18, dW28
    torch.cuda.empty_cache()

    # PERSISTENT leaf weights (created ONCE) so the op's version-keyed weight-quant caches HIT across
    # timing iters -- mirrors real training (weights change only on optim.step; the fp8 preps are
    # reused across a grad-accum window). Fresh-leaf-per-iter would re-quantize every weight each
    # call (a bench artifact that hugely inflates the fp8 time). x/topk_w persist too for stable timing.
    xf, twf, W1f, W2f = _leaf(x), _leaf(topk_w), _leaf(W1), _leaf(W2)
    xb, twb, W1b, W2b = _leaf(x), _leaf(topk_w), _leaf(W1), _leaf(W2)

    def _fwd_bwd_fp8():
        for t in (xf, twf, W1f, W2f):
            t.grad = None
        fused_mega_moe_fp8(group, xf, topk_idx, twf, W1f, W2f).backward(grad_y)

    def _fwd_bwd_bf16():
        for t in (xb, twb, W1b, W2b):
            t.grad = None
        fused_mega_moe(group, xb, topk_idx, twb, W1b, W2b).backward(grad_y)

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
        snr_y = _amin(group, r["snr_y"])
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
                ok = min(snr_y, snr_dx, snr_dtw, snr_dw1, snr_dw2) >= gate
                print(f"  [SNR vs analytic ref]  y={snr_y:.1f}  dx={snr_dx:.1f}  "
                      f"d_topk_w={snr_dtw:.1f}  dW1={snr_dw1:.1f}  dW2={snr_dw2:.1f} dB  "
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
