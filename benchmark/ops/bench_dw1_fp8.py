###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Backward dW1 (fc1 weight grad) MXFP8 variable-K wgrad -- isolated correctness + latency.

Replicates the backward up to dW1 on real mega-pool data: forward L1 -> (l1, dispatch_weights) and
CLONE the forward-dispatched fc1-input pool (pool_x, native rowwise-fp8) BEFORE STEP1 overwrites it;
STEP1 (dispatch(dy)+fc2 dgrad); STEP2 swiglu_backward -> grad_l1. Then dW1 = grad_l1^T @ pool_x
(variable-K over the pool) BOTH ways on the SAME tensors -- fp8 (``_mxfp8_variable_k_wgrad_dw1``,
colwise-quant grad_l1 + requant the fp8 pool colwise) vs bf16 (``grouped_gemm_variable_k_impl`` on
the dequant'd pool) -- and gates by SNR. dW1's fp8 path is LOCAL (reuses the forward pool, no
cross-rank re-dispatch of saved_x). NOTE: this isolates the wgrad GEMM (both on the same local pool);
the production bf16 dW1 additionally re-dispatches saved_x cross-rank, which fp8 avoids.

Run inside the dev container (8 GPUs):
  PYTHONPATH=<repo> python benchmark/ops/bench_dw1_fp8.py --num-processes 8 --num-tokens 8192
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
from primus_turbo.pytorch.core.backend import BackendType
from primus_turbo.pytorch.kernels.grouped_gemm.grouped_gemm_impl import (
    grouped_gemm_variable_k_impl,
)
from primus_turbo.pytorch.kernels.mega_moe.mega_moe_backward_fp8_impl import (
    _mxfp8_step1_dispatch_dgrad,
    _mxfp8_variable_k_wgrad_dw1,
)

_H_GROUP_LENS = 9
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

    # forward L1 -> l1 + dispatch_weights, then CLONE the fc1-input pool BEFORE STEP1 overwrites it
    w1q, w1s = quantize_grouped_weight_mxfp8(W1)
    torch.cuda.synchronize(); group.barrier()
    symm.scoreboard.zero_()
    torch.cuda.synchronize(); group.barrier()
    l1 = dispatch_grouped_gemm_mxfp8(x, None, w1q, w1s, handle, sym_layout, symm, BM=BM, BN=BN)
    dispatch_weights = symm.weight_recv_buf.clone()
    Pp, Hp = symm.pool_fp8.shape
    pool_x_fp8 = (symm.pool_fp8.clone(), symm.pool_scale.reshape(Pp, Hp // 32).clone())

    dy = torch.randn((T, H), device="cuda", dtype=torch.bfloat16)
    grad_swiglu, _ = _mxfp8_step1_dispatch_dgrad(dy, W2, group, handle, BM, BN)
    grad_l1, _, _ = swiglu_backward_flydsl_kernel(
        grad_swiglu, l1, get_symm_buffer_for_mega_moe().meta_scalars[1:2],
        scale=dispatch_weights, return_gate=True, return_act_w=True,
    )
    group_lens, group_offs = handle[_H_GROUP_LENS], handle[_H_GROUP_OFFS]

    # bf16 reference operand: dequant the fc1-input pool once
    pf = pool_x_fp8[0].to(torch.float32).view(Pp, Hp // 32, 32)
    ps = pool_x_fp8[1].reshape(Pp, Hp // 32).view(torch.uint8).to(torch.int32)
    sc = torch.exp2((ps - 127).to(torch.float32)).view(Pp, Hp // 32, 1)
    pool_x_bf = (pf * sc).view(Pp, Hp).to(torch.bfloat16)

    def _fp8():  # colwise-quant grad_l1 + requant fp8 pool colwise -> mxfp8 variable-K wgrad (LOCAL)
        return _mxfp8_variable_k_wgrad_dw1(grad_l1, pool_x_fp8, group_lens, group_offs)

    def _bf16():  # bf16 variable-K wgrad on the dequant'd pool
        return grouped_gemm_variable_k_impl(
            grad_l1, pool_x_bf, group_lens, group_offs,
            trans_a=True, trans_b=False, trans_c=False, num_cu=None,
            default_backend=BackendType.TRITON.value,
        )

    dW1_fp8 = _fp8()
    dW1_bf = _bf16()
    assert tuple(dW1_fp8.shape) == (epr, 2 * I, H), f"dW1 fp8 shape {tuple(dW1_fp8.shape)} != {(epr, 2 * I, H)}"
    snr = _snr_db(dW1_bf, dW1_fp8)
    nan = bool((~torch.isfinite(dW1_fp8.float())).any())

    t_dw1 = _bench(_fp8, warmup=args.warmup, iters=args.iters, group=group)
    t_dw1_bf = _bench(_bf16, warmup=args.warmup, iters=args.iters, group=group)
    m_pad = int(group_offs[-1].item())
    flops = 2.0 * m_pad * (2 * I) * H
    symm.destroy()
    return {"snr": snr, "nan": float(nan), "t_dw1": t_dw1, "t_dw1_bf": t_dw1_bf, "flops": flops, "m_pad": m_pad}


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
        t_dw1, t_dw1_bf = _amax(group, r["t_dw1"]), _amax(group, r["t_dw1_bf"])
        if rank == 0:
            tf = lambda ms: r["flops"] / (ms * 1e-3) / 1e12
            print(f"\n{'='*72}")
            print(f"[backward dW1  fc1 wgrad (variable-K, LOCAL)  fp8 vs bf16]  EP{world} T={args.num_tokens} "
                  f"H={args.hidden} I={args.inter} E={args.num_experts} K={args.num_topk}")
            print(f"{'='*72}")
            print(f"  dW1 fp8      : {t_dw1:8.3f} ms | {tf(t_dw1):8.1f} TFLOPS  (M_pool={r['m_pad']})")
            print(f"  dW1 bf16     : {t_dw1_bf:8.3f} ms | {tf(t_dw1_bf):8.1f} TFLOPS  (GEMM-only, same local pool)")
            print(f"  fp8/bf16     : {t_dw1 / t_dw1_bf:.3f}x  ({'fp8 faster' if t_dw1 < t_dw1_bf else 'fp8 SLOWER'})  "
                  f"[GEMM-only; fp8 additionally avoids the bf16 cross-rank re-dispatch of saved_x]")
            print(f"  [acc] dW1 fp8 vs bf16: SNR={snr:.2f} dB  nan={bool(nan)}  "
                  f"{'PASS' if snr >= 15.0 and not nan else 'FAIL'} (gate SNR>=15dB)")
        torch.cuda.synchronize(); group.barrier()
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="backward dW1 (fc1 wgrad, LOCAL) fp8 vs bf16")
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
