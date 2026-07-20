###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Self-contained smoke + latency harness for the ported fused MXFP8 dispatch+fc1 (L1).

This validates ONLY the vendored fp8 stack under ``primus_turbo.flydsl.mega.fp8`` (no bf16
kernels, no source ``mega_utils``). It:

  1. builds the mxfp8 symmetric workspace (vendored SymLayout/scoreboard/two-heap),
  2. runs the dispatch prologue -> handle,
  3. quantizes x (rowwise mxfp8) + W1 (grouped mxfp8) ONCE,
  4. launches ``dispatch_grouped_gemm_mxfp8`` (fused clean-push -> preshuffle -> mxfp8 NT GEMM),
  5. checks correctness with a PURE-TORCH dequant grouped-GEMM over the kernel's OWN dispatched
     pool (``symm.pool_fp8`` / ``symm.pool_scale``) -> cos gate. This gates the fused GEMM math +
     preshuffle + E8M0 scale handling. (It reads the kernel-dispatched pool, so it does not
     independently re-derive the cross-rank dispatch; that parity is a follow-up vs bf16.)
  6. times the fused kernel (CUDA events).

Run inside the dev container (8 GPUs):
  PYTHONPATH=<repo> python benchmark/ops/bench_dispatch_grouped_gemm_mxfp8_l1.py \
      --num-processes 8 --num-tokens 2048
"""

import argparse
import datetime
import math
import os

import numpy as np
import torch
import torch.distributed as dist

import primus_turbo.pytorch  # noqa: F401  (registers ops / core)
from primus_turbo.flydsl.mega.fp8 import (
    dispatch_grouped_gemm_mxfp8,
    dispatch_prologue,
    get_symm_buffer_for_mega_moe,
    quantize_grouped_weight_mxfp8,
    quantize_rowwise_mxfp8,
    quantize_rowwise_mxfp8_flydsl,
)

_H_TILE_TO_EXPERT = 7
_H_EXPECTED = 8
_MXFP8_BLOCK = 32


def _generate_routing(num_tokens, num_topk, num_experts, *, device, seed):
    """Load-balanced top-k routing (matches the source bench's load_balanced mode)."""
    g = torch.Generator(device=device).manual_seed(seed)
    scores = torch.rand(num_tokens, num_experts, generator=g, device=device).abs() + 1
    topk_weight, topk_idx = torch.topk(scores.softmax(-1), num_topk, dim=-1)
    return topk_idx.to(torch.int64), topk_weight.to(torch.float32)


def _global_weights(E, I, H, device):
    """Deterministic global expert W1 [E, 2I, H] (identical on every rank, then sliced)."""
    g = torch.Generator(device=device).manual_seed(1234)
    return torch.randn((E, 2 * I, H), generator=g, device=device, dtype=torch.bfloat16) * (2.0 / math.sqrt(H))


def _dequant_mxfp8(q, s_raw, block=_MXFP8_BLOCK):
    """Dequant a rowwise-along-last-dim mxfp8 tensor: q (fp8) * 2^(s_raw - 127).

    ``q`` [..., K] fp8, ``s_raw`` [..., K//block] raw E8M0 bytes. Returns fp32 [..., K]."""
    *lead, K = q.shape
    qf = q.float().view(*lead, K // block, block)
    scale = torch.exp2(s_raw.view(torch.uint8).float() - 127.0).unsqueeze(-1)
    return (qf * scale).view(*lead, K)


def _bench(fn, *, warmup, iters, group, reset=None):
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]

    def _iter(s=None, e=None):
        torch.cuda.synchronize(); group.barrier()
        if reset is not None:
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
    times = np.array([s.elapsed_time(e) for s, e in zip(starts, ends)])[1:]
    return float(np.average(times))


def profile(group, args):
    rank, world = group.rank(), group.size()
    BM, BN, H, I = args.bm, args.bn, args.hidden, args.inter
    E, K, T, ndcu, pscu = args.num_experts, args.num_topk, args.num_tokens, args.num_dispatch_cu, args.num_preshuffle_cu
    epr = E // world
    N = 2 * I

    torch.manual_seed(7 + rank)
    x = torch.randn((T, H), device="cuda", dtype=torch.float32).bfloat16()
    topk_idx, topk_weight = _generate_routing(T, K, E, device="cuda", seed=100 + rank)
    W1 = _global_weights(E, I, H, "cuda")[rank * epr : (rank + 1) * epr].contiguous()

    symm = get_symm_buffer_for_mega_moe(
        group, num_experts=E, num_max_tokens_per_rank=T, num_topk=K, hidden=H,
        intermediate_hidden=I, block_m=BM, block_n=BN, use_mxfp8=True,
    )
    sym_layout = symm.make_sym_layout()
    handle = tuple(
        dispatch_prologue(
            topk_idx, topk_weight, sym_layout=sym_layout, num_tokens=T, num_topk=K,
            num_experts=E, world_size=world, rank=rank, experts_per_rank=epr,
            block_m=BM, num_max_pool_tokens=symm.num_max_pool_tokens, no_cpu_sync=True,
        )
    )
    tile_to_expert = handle[_H_TILE_TO_EXPERT]
    num_tile_blocks = symm.meta_scalars[1:2]
    symm.assert_capacity()

    # WEIGHTS are static (module-owned + version-keyed in production) -> quantize ONCE, out of the
    # timed L1 step. TOKENS are activation-dependent -> re-quantized EVERY forward; that token quant
    # now lives INSIDE dispatch_grouped_gemm_mxfp8 (pass bf16 x + xs=None -> one global rowwise
    # mxfp8 quant, then the clean-push pipeline). We still time the quant alone to show its share.
    w1q, w1s = quantize_grouped_weight_mxfp8(W1)

    def _quant_tokens():  # for the breakdown only (the op does this internally on the bf16 path)
        xq, xs = quantize_rowwise_mxfp8_flydsl(x)
        return xq, xs.view(torch.float8_e8m0fnu)

    real_tiles = int(num_tile_blocks[0].item())
    M_eff = real_tiles * BM

    def _l1_step():  # the REAL per-forward cost: token quant (inside) + fused dispatch+GEMM
        return dispatch_grouped_gemm_mxfp8(
            x, None, w1q, w1s, handle, sym_layout, symm, BM=BM, BN=BN,
            num_dispatch_cu=ndcu, num_preshuffle_cu=pscu,
        )

    # ── correctness: single L1 step, then torch dequant-GEMM over the kernel's dispatched pool ──
    torch.cuda.synchronize(); group.barrier()
    symm.scoreboard.zero_()
    torch.cuda.synchronize(); group.barrier()
    out = _l1_step()
    torch.cuda.synchronize(); group.barrier()

    # torch reference over the pool the kernel just dispatched (pool_fp8 / pool_scale raw E8M0)
    A = _dequant_mxfp8(symm.pool_fp8[:M_eff], symm.pool_scale[:M_eff])             # [M_eff, H] f32
    Wd = _dequant_mxfp8(w1q, w1s)                                                  # [G, N, H]  f32
    t2e = tile_to_expert[:real_tiles].to(torch.long)
    row_expert = t2e.repeat_interleave(BM)                                         # [M_eff]
    ref = torch.empty((M_eff, N), device="cuda", dtype=torch.float32)
    for g in torch.unique(row_expert).tolist():
        m = row_expert == g
        ref[m] = A[m] @ Wd[g].t()
    o = out[:M_eff].float()
    cos = float(torch.dot(o.flatten(), ref.flatten()) / (o.norm() * ref.norm() + 1e-12))
    rel = float((o - ref).norm() / (ref.norm() + 1e-12))
    # free the large fp32 reference temporaries before timing (at T=8192 these are ~8GB).
    del A, Wd, ref, o, out

    # ── latency. One fused-containing loop only (t_l1 = token quant + fused): back-to-back fused
    # launches at large T can race the cross-rank scoreboard reset (late XGMI writes landing after
    # the zero-barrier), so we keep a single fused bench loop -- the real per-forward cost. The
    # token quant is also timed alone (local, no scoreboard) to show its share; fused = L1 - quant.
    t_quant = _bench(_quant_tokens, warmup=args.warmup, iters=args.iters, group=group)
    t_l1 = _bench(_l1_step, warmup=args.warmup, iters=args.iters,
                  reset=lambda: symm.scoreboard.zero_(), group=group)

    flops = 2.0 * M_eff * N * H
    symm.destroy()
    return {"cos": cos, "rel": rel, "quant_ms": t_quant, "l1_ms": t_l1, "flops": flops, "M_eff": M_eff}


def _all_max(group, v):
    t = torch.tensor([v], device="cuda"); dist.all_reduce(t, op=dist.ReduceOp.MAX, group=group); return float(t)


def _all_min(group, v):
    t = torch.tensor([v], device="cuda"); dist.all_reduce(t, op=dist.ReduceOp.MIN, group=group); return float(t)


def worker(local_rank, world, args):
    ip = os.getenv("MASTER_ADDR", "127.0.0.1")
    port = int(os.getenv("MASTER_PORT", "8492"))
    torch.cuda.set_device(local_rank)
    timeout_s = int(os.getenv("MEGA_BENCH_TIMEOUT_S", "600"))
    dist.init_process_group(
        "nccl", init_method=f"tcp://{ip}:{port}", world_size=world, rank=local_rank,
        timeout=datetime.timedelta(seconds=timeout_s),
    )
    torch.set_default_device("cuda")
    group = dist.new_group(list(range(world)))
    rank = dist.get_rank()
    try:
        r = profile(group, args)
        cos, rel = _all_min(group, r["cos"]), _all_max(group, r["rel"])
        quant = _all_max(group, r["quant_ms"])
        l1 = _all_max(group, r["l1_ms"])
        if rank == 0:
            tf = lambda ms: r["flops"] / (ms * 1e-3) / 1e12
            print(f"\n{'='*72}")
            print(f"[fused mxfp8 dispatch+fc1 L1]  EP{world} T={args.num_tokens} H={args.hidden} "
                  f"I={args.inter} E={args.num_experts} K={args.num_topk} BM={args.bm} BN={args.bn}")
            print(f"{'='*72}")
            print(f"  token_quant  : {quant:8.3f} ms  (rowwise mxfp8, per-forward)")
            print(f"  fused        : {l1 - quant:8.3f} ms | {tf(l1 - quant):8.1f} TFLOPS  (kernel only, = L1 - quant)")
            print(f"  L1 total     : {l1:8.3f} ms | {tf(l1):8.1f} TFLOPS  (token quant + fused; M_eff={r['M_eff']})")
            print(f"  [acc] fp8 fused vs torch dequant-GEMM ref: cos={cos:.5f} rel={rel:.4f}  "
                  f"{'PASS' if cos >= 0.99 and rel <= 0.05 else 'FAIL'}")
        torch.cuda.synchronize(); group.barrier()
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Fused mxfp8 dispatch+fc1 (L1) smoke + latency")
    ap.add_argument("--num-processes", type=int, default=8)
    ap.add_argument("--hidden", type=int, default=7168)   # DeepSeek-V3
    ap.add_argument("--inter", type=int, default=2048)
    ap.add_argument("--num-experts", type=int, default=256)
    ap.add_argument("--num-topk", type=int, default=8)
    ap.add_argument("--num-tokens", type=int, default=2048)  # smaller than DSv3 8192 for a fast smoke
    ap.add_argument("--bm", type=int, default=256)
    ap.add_argument("--bn", type=int, default=256)
    ap.add_argument("--num-dispatch-cu", type=int, default=16)
    ap.add_argument("--num-preshuffle-cu", type=int, default=16)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=30)
    args = ap.parse_args()
    torch.multiprocessing.spawn(worker, args=(args.num_processes, args), nprocs=args.num_processes)
