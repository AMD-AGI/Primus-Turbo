###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""BF16 mega MoE per-stage latency bench (EP, intra-node) -- the bf16 reference legs split out
of ``bench_mega_moe_fp8.py`` into their OWN process so the fp8 and bf16 stacks can never poison
each other (e.g. the large-K fc1_dgrad_combine bf16 combine OOB must not take down the fp8 stage bench).

Same CLI/stages as ``bench_mega_moe_fp8.py`` (compare the two runs' numbers offline):

  --stage l1                 L1 = dispatch + fc1 (NT)
  --stage l2                 L2 = fc2 + combine (NT weighted)
  --stage fwd                full forward (mega_moe_fused)
  --stage dispatch_fc2_dgrad backward dispatch(dy) + fc2-dgrad (NN)
  --stage fc2_wgrad          backward dW2 (variable-K wgrad, Triton)
  --stage fc1_wgrad          backward dW1 (variable-K wgrad, Triton)
  --stage fc1_dgrad_combine  backward fc1 dgrad + combine (NT)
  --stage bwd                all backward stages

The wgrad stages time the bf16 Triton variable-K GEMM on SYNTHETIC bf16 operands (correct shapes
+ the real bf16 handle's group_lens/offs); GEMM latency is value-independent, so this is a faithful
latency reference without re-deriving the fp8-coupled operands.

Run (8 GPUs): PYTHONPATH=<repo> python benchmark/ops/training/bench_mega_moe_bf16.py --num-tokens 8192 --stage both
"""

import argparse
import datetime
import math
import os
import sys

import numpy as np
import torch
import torch.distributed as dist

_HERE = os.path.dirname(__file__)
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..", "..")))

from config import get_platform_info  # noqa: E402

import primus_turbo.pytorch  # noqa: E402,F401
from primus_turbo.flydsl.mega import (  # noqa: E402  (shipped bf16 kernels)
    dispatch_grouped_gemm_bf16_flydsl_kernel,
    grouped_gemm_combine_bf16_flydsl_kernel,
    swiglu_flydsl_kernel,
)
from primus_turbo.pytorch.core.backend import BackendType  # noqa: E402
from primus_turbo.pytorch.kernels.grouped_gemm.grouped_gemm_impl import (  # noqa: E402
    grouped_gemm_variable_k_impl,
)
from primus_turbo.pytorch.ops.moe.mega_moe_fused import mega_moe_fused  # noqa: E402

# bf16 dispatch-handle layout (see mega_moe_backward_impl): 5=tile_to_expert, 6=num_tokens_per_expert
# (group_lens), 7=its prefix (group_offs), 8=num_tile_blocks.
_H_GROUP_LENS = 6
_H_GROUP_OFFS = 7
_H_NUM_TILE_BLOCKS = 8


def generate_inputs(rank, world, *, T, H, I, E, K, mode, device="cuda"):
    """One rank's local MoE inputs: x, this rank's L1/L2 expert shard, top-k routing (per-rank seed).
    load_balanced = rand-softmax top-k; round_robin = arange%E (deterministic, zero pad)."""
    epr = E // world
    g = torch.Generator(device=device).manual_seed(1234 + rank)
    x = torch.randn((T, H), generator=g, device=device, dtype=torch.float32).bfloat16()
    l1 = torch.randn((epr, 2 * I, H), generator=g, device=device, dtype=torch.bfloat16) * (2.0 / math.sqrt(H))
    l2 = torch.randn((epr, H, I), generator=g, device=device, dtype=torch.bfloat16) * (2.0 / math.sqrt(I))
    if mode == "load_balanced":
        scores = torch.rand(T, E, generator=g, device=device).abs() + 1
        topk_w, topk_idx = torch.topk(scores.softmax(-1), K, dim=-1)
    elif mode == "round_robin":
        topk_idx = (torch.arange(T * K, device=device).view(T, K) % E)
        topk_w = torch.rand(T, K, generator=g, device=device).softmax(-1)
    else:
        raise ValueError(f"unknown routing mode: {mode}")
    return x, l1, l2, topk_idx.to(torch.int64), topk_w.to(torch.float32)


def _bench(fn, *, warmup, iters, group):
    """Per-call CUDA-event latency; sync+barrier before each call serializes ranks."""
    ev_s = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ev_e = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for _ in range(warmup):
        torch.cuda.synchronize(); group.barrier(); fn()
    for i in range(iters):
        torch.cuda.synchronize(); group.barrier()
        ev_s[i].record(); fn(); ev_e[i].record()
    torch.cuda.synchronize()
    return float(np.average([s.elapsed_time(e) for s, e in zip(ev_s, ev_e)][1:]))


def _bench_b2b(fn, *, warmup, iters, group):
    """Back-to-back latency (N calls / N) for the fast fused kernels (source test_dw2_bench method)."""
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


def _amax(group, v):
    t = torch.tensor([v], device="cuda"); dist.all_reduce(t, op=dist.ReduceOp.MAX, group=group); return float(t)


def _amin(group, v):
    t = torch.tensor([v], device="cuda"); dist.all_reduce(t, op=dist.ReduceOp.MIN, group=group); return float(t)


def _bf16_l1_handle(x, W1, group, topk_idx, topk_w, BM, BN):
    """One bf16 L1 dispatch (handle=None auto-prologue) -> (l1_out, handle). Handle carries the
    group_lens/offs/num_tile_blocks the wgrad + combine stages need."""
    l1_out, _pool, _dw, handle = dispatch_grouped_gemm_bf16_flydsl_kernel(
        x, W1, group, handle=None, topk_idx=topk_idx, topk_weights=topk_w, layout="nt", BM=BM, BN=BN,
    )
    return l1_out, handle


def _handle_dims(handle, BM):
    m_eff = int(handle[_H_NUM_TILE_BLOCKS].item()) * BM
    m_pad = int(handle[_H_GROUP_OFFS][-1].item())
    return m_eff, m_pad


def profile_l1(group, args, mode):
    rank, world = group.rank(), group.size()
    H, I, E, K, T, BM, BN = args.hidden, args.inter, args.num_experts, args.num_topk, args.num_tokens, args.bm, args.bn
    x, W1, _W2, topk_idx, topk_w = generate_inputs(rank, world, T=T, H=H, I=I, E=E, K=K, mode=mode)

    def _bf16():  # FULL per-forward L1 (shipped bf16): handle=None re-runs the prologue each call
        return dispatch_grouped_gemm_bf16_flydsl_kernel(
            x, W1, group, handle=None, topk_idx=topk_idx, topk_weights=topk_w, layout="nt", BM=BM, BN=BN,
        )

    out, hbf = _bf16_l1_handle(x, W1, group, topk_idx, topk_w, BM, BN)
    m_eff, m_pad = _handle_dims(hbf, BM)
    fin = bool(torch.isfinite(out.float()).all())
    t_bf16 = _bench(_bf16, warmup=args.warmup, iters=args.iters, group=group)
    flops = 2.0 * m_eff * (2 * I) * H
    return {"bf16_ms": t_bf16, "flops": flops, "M_eff": m_eff, "m_pad": m_pad, "fin": float(fin)}


def profile_l2(group, args, mode):
    rank, world = group.rank(), group.size()
    H, I, E, K, T, BM, BN = args.hidden, args.inter, args.num_experts, args.num_topk, args.num_tokens, args.bm, args.bn
    x, W1, W2, topk_idx, topk_w = generate_inputs(rank, world, T=T, H=H, I=I, E=E, K=K, mode=mode)

    l1_bf, hbf = _bf16_l1_handle(x, W1, group, topk_idx, topk_w, BM, BN)
    act_bf = swiglu_flydsl_kernel(l1_bf, num_tile_blocks=hbf[_H_NUM_TILE_BLOCKS])
    tidx32 = topk_idx.to(torch.int32).contiguous().view(-1)
    topk_w_flat = topk_w.contiguous().view(-1)
    m_eff, m_pad = _handle_dims(hbf, BM)

    def _bf16():  # shipped bf16 combine (nt), reused back-to-back (self-managed cross-rank state)
        return grouped_gemm_combine_bf16_flydsl_kernel(
            act_bf, W2, hbf, topk_indices=tidx32, topk_weights=topk_w_flat, layout="nt", BM=BM, BN=BN,
        )

    y = _bf16()
    fin = bool(torch.isfinite(y[0].float()).all()) if isinstance(y, tuple) else bool(torch.isfinite(y.float()).all())
    t_bf16 = _bench(_bf16, warmup=args.warmup, iters=args.iters, group=group)
    flops = 2.0 * m_eff * H * I
    return {"bf16_ms": t_bf16, "flops": flops, "M_eff": m_eff, "m_pad": m_pad, "fin": float(fin)}


def profile_fwd(group, args, mode):
    rank, world = group.rank(), group.size()
    H, I, E, K, T = args.hidden, args.inter, args.num_experts, args.num_topk, args.num_tokens
    x, W1, W2, topk_idx, topk_w = generate_inputs(rank, world, T=T, H=H, I=I, E=E, K=K, mode=mode)
    with torch.no_grad():
        def _bf16():
            return mega_moe_fused(group, x, topk_idx, topk_w, W1, W2)
        y = _bf16()
        fin = bool(torch.isfinite(y.float()).all())
        t_bf16 = _bench(_bf16, warmup=args.warmup, iters=args.iters, group=group)
    return {"bf16_ms": t_bf16, "fin": float(fin)}


def profile_dispatch_fc2_dgrad(group, args, mode):
    rank, world = group.rank(), group.size()
    H, I, E, K, T, BM, BN = args.hidden, args.inter, args.num_experts, args.num_topk, args.num_tokens, args.bm, args.bn
    x, W1, W2, topk_idx, topk_w = generate_inputs(rank, world, T=T, H=H, I=I, E=E, K=K, mode=mode)
    g = torch.Generator(device="cuda").manual_seed(4321 + rank)
    dy = torch.randn((T, H), generator=g, device="cuda", dtype=torch.bfloat16)

    # bf16 forward dispatch (nt) -> handle; then the L2-dgrad = dispatch(dy) + fc2-dgrad (nn)
    _l1, hbf = _bf16_l1_handle(x, W1, group, topk_idx, topk_w, BM, BN)
    m_eff, m_pad = _handle_dims(hbf, BM)

    def _bf16():  # L2 dgrad = dispatch(dy) PUSH + grouped fc2-dgrad GEMM (exactly the bf16 bwd step)
        return dispatch_grouped_gemm_bf16_flydsl_kernel(dy, W2, group, handle=hbf, layout="nn", BM=BM, BN=BN)

    y = _bf16()
    fin = bool(torch.isfinite(y[0].float()).all())
    t_bf16 = _bench_b2b(_bf16, warmup=args.warmup, iters=args.iters, group=group)
    flops = 2.0 * m_eff * I * H
    return {"bf16_ms": t_bf16, "flops": flops, "M_eff": m_eff, "m_pad": m_pad, "fin": float(fin)}


def _wgrad_profile(group, args, mode, *, kdim, ndim, flops_k):
    """Shared bf16 variable-K wgrad latency: dW = a^T @ b (trans_a) over the pool, per group.
    Synthetic bf16 operands a[m_pad, kdim] / b[m_pad, ndim] on the real bf16 handle's group_lens/offs
    (GEMM latency is value-independent). ``flops_k`` = the reduced K used in the FLOP count."""
    rank, world = group.rank(), group.size()
    H, I, E, K, T, BM, BN = args.hidden, args.inter, args.num_experts, args.num_topk, args.num_tokens, args.bm, args.bn
    x, W1, _W2, topk_idx, topk_w = generate_inputs(rank, world, T=T, H=H, I=I, E=E, K=K, mode=mode)
    _l1, hbf = _bf16_l1_handle(x, W1, group, topk_idx, topk_w, BM, BN)
    group_lens, group_offs = hbf[_H_GROUP_LENS], hbf[_H_GROUP_OFFS]
    _m_eff, m_pad = _handle_dims(hbf, BM)
    gen = torch.Generator(device="cuda").manual_seed(99 + rank)
    a = torch.randn((m_pad, kdim), generator=gen, device="cuda", dtype=torch.bfloat16)
    b = torch.randn((m_pad, ndim), generator=gen, device="cuda", dtype=torch.bfloat16)

    def _bf16():  # bf16 variable-K wgrad (shipped Triton): a^T @ b per group -> [G, kdim, ndim]
        return grouped_gemm_variable_k_impl(
            a, b, group_lens, group_offs, trans_a=True, trans_b=False, trans_c=False,
            num_cu=None, default_backend=BackendType.TRITON.value,
        )

    dW = _bf16()
    fin = bool(torch.isfinite(dW.float()).all())
    t_bf16 = _bench_b2b(_bf16, warmup=args.warmup, iters=args.iters, group=group)
    flops = 2.0 * m_pad * kdim * ndim
    return {"bf16_ms": t_bf16, "flops": flops, "m_pad": m_pad, "fin": float(fin)}


def profile_fc2_wgrad(group, args, mode):
    # dW2 = dispatch_l2_grad^T @ act_weighted : a[m_pad,H] , b[m_pad,I] -> [G,H,I]
    return _wgrad_profile(group, args, mode, kdim=args.hidden, ndim=args.inter, flops_k=None)


def profile_fc1_wgrad(group, args, mode):
    # dW1 = grad_l1^T @ pool_x : a[m_pad,2I] , b[m_pad,H] -> [G,2I,H]
    return _wgrad_profile(group, args, mode, kdim=2 * args.inter, ndim=args.hidden, flops_k=None)


def profile_fc1_dgrad_combine(group, args, mode):
    rank, world = group.rank(), group.size()
    H, I, E, K, T, BM, BN = args.hidden, args.inter, args.num_experts, args.num_topk, args.num_tokens, args.bm, args.bn
    x, W1, _W2, topk_idx, topk_w = generate_inputs(rank, world, T=T, H=H, I=I, E=E, K=K, mode=mode)

    # bf16 stack: shipped dispatch for a handle + a realistic grad_l1 replica, then combine(nt).
    l1_bf, hbf = _bf16_l1_handle(x, W1, group, topk_idx, topk_w, BM, BN)
    grad_l1_bf = l1_bf.contiguous()                 # [P, 2I]
    w1t_bf = W1.transpose(1, 2).contiguous()        # [G, H, 2I] (NT-reuse)
    tidx32 = topk_idx.to(torch.int32).contiguous().view(-1)
    m_eff, m_pad = _handle_dims(hbf, BM)

    def _bf16():  # shipped bf16 combine (nt = w1^T reuse). NOTE: OOBs at large T (K=2I) -- isolated here.
        dx, _ = grouped_gemm_combine_bf16_flydsl_kernel(
            grad_l1_bf, w1t_bf, hbf, topk_indices=tidx32, layout="nt", BM=BM, BN=BN,
        )
        return dx

    dx = _bf16()
    fin = bool(torch.isfinite(dx.float()).all())
    t_bf16 = _bench_b2b(_bf16, warmup=args.warmup, iters=args.iters, group=group)
    flops = 2.0 * m_pad * (2 * I) * H
    return {"bf16_ms": t_bf16, "flops": flops, "M_eff": m_eff, "m_pad": m_pad, "fin": float(fin)}


_STAGES = {
    "l1": (profile_l1, "L1 (dispatch+fc1)"),
    "l2": (profile_l2, "L2 (fc2+combine)"),
    "fwd": (profile_fwd, "FULL forward (L1+SwiGLU+L2)"),
    "dispatch_fc2_dgrad": (profile_dispatch_fc2_dgrad, "bwd dispatch(dy)+fc2-dgrad"),
    "fc2_wgrad": (profile_fc2_wgrad, "bwd fc2 wgrad (dW2, variable-K)"),
    "fc1_wgrad": (profile_fc1_wgrad, "bwd fc1 wgrad (dW1, variable-K)"),
    "fc1_dgrad_combine": (profile_fc1_dgrad_combine, "bwd fc1_dgrad_combine"),
}
_FWD_STAGES = ("l1", "l2", "fwd")
_BWD_STAGES = ("dispatch_fc2_dgrad", "fc2_wgrad", "fc1_wgrad", "fc1_dgrad_combine")


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
    _platform, gpu = get_platform_info() if rank == 0 else (None, None)
    modes = ["load_balanced", "round_robin"] if args.mode == "both" else [args.mode]
    stages = (list(_FWD_STAGES) if args.stage == "both"
              else list(_BWD_STAGES) if args.stage == "bwd"
              else [args.stage])
    try:
        for mode in modes:
            hdr = (f"{gpu}  EP{world} T={args.num_tokens} H={args.hidden} I={args.inter} "
                   f"E={args.num_experts} K={args.num_topk}  routing={mode}")
            for st in stages:
                profile_fn, label = _STAGES[st]
                r = profile_fn(group, args, mode)
                bf16_ms = _amax(group, r["bf16_ms"])
                fin = _amin(group, r.get("fin", 1.0))
                if rank == 0:
                    line = f"\n{'='*80}\n[mega MoE bf16  {label}]  {hdr}\n{'='*80}"
                    if "flops" in r:
                        tf = r["flops"] / (bf16_ms * 1e-3) / 1e12
                        line += f"\n  bf16 : {bf16_ms:8.3f} ms | {tf:8.1f} TFLOPS  (M_pool={r.get('m_pad', '-')})"
                    else:
                        line += f"\n  bf16 : {bf16_ms:8.3f} ms"
                    line += f"\n  [acc] finite={bool(fin >= 1.0)}"
                    print(line)
                torch.cuda.synchronize(); group.barrier()
    finally:
        dist.destroy_process_group()


def _build_parser():
    ap = argparse.ArgumentParser(description="mega MoE BF16 per-stage latency bench (training)")
    ap.add_argument("--stage",
                    choices=list(_STAGES) + ["both", "bwd"], default="both",
                    help="which stage(s): 'both' = l1+l2+fwd; 'bwd' = all backward stages "
                         "(fc1_dgrad_combine bf16 OOBs at large T -- runs last)")
    ap.add_argument("--mode", choices=["load_balanced", "round_robin", "both"], default="both")
    ap.add_argument("--num-processes", type=int, default=8)
    ap.add_argument("--hidden", type=int, default=7168)   # DeepSeek-V3
    ap.add_argument("--inter", type=int, default=2048)
    ap.add_argument("--num-experts", type=int, default=256)
    ap.add_argument("--num-topk", type=int, default=8)
    ap.add_argument("--num-tokens", type=int, default=8192)
    ap.add_argument("--bm", type=int, default=256)
    ap.add_argument("--bn", type=int, default=256)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=15)
    return ap


if __name__ == "__main__":
    args = _build_parser().parse_args()
    torch.multiprocessing.spawn(worker, args=(args.num_processes, args), nprocs=args.num_processes)
