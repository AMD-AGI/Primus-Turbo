###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Backward dW2 (fc2 weight grad) MXFP8 variable-K wgrad -- isolated correctness + latency.

Replicates the backward up to dW2 on real mega-pool data: forward L1 -> (l1, dispatch_weights),
STEP1 (dispatch(dy)+fc2 dgrad) -> grad_swiglu + the dispatched-dy fp8 pool, STEP2 swiglu_backward
-> act_weighted. Then computes dW2 = dispatch_l2_grad^T @ act_weighted (variable-K over the pool)
BOTH ways on the SAME tensors -- fp8 (``_mxfp8_variable_k_wgrad``, requant the fp8 pool colwise +
colwise-quant act) vs bf16 (``grouped_gemm_variable_k_impl`` on the dequant'd pool) -- and gates by
SNR. dW2 is the large backward GEMM (H*I over the pool).

Run inside the dev container (8 GPUs):
  PYTHONPATH=<repo> python benchmark/ops/bench_dw2_fp8.py --num-processes 8 --num-tokens 8192
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
    colwise_grouped_meta,
    colwise_quant_mxfp8_grouped_flydsl,
    colwise_requant_mxfp8_grouped_fp8in_flydsl,
    dispatch_grouped_gemm_mxfp8,
    dispatch_prologue,
    get_symm_buffer_for_mega_moe,
    quantize_grouped_weight_mxfp8,
    swiglu_backward,
)
from primus_turbo.pytorch.core.backend import BackendType
from primus_turbo.pytorch.core.low_precision import ScalingGranularity
from primus_turbo.pytorch.kernels.grouped_gemm.grouped_gemm_fp8_impl import (
    grouped_gemm_fp8_variable_k_impl,
)
from primus_turbo.pytorch.kernels.grouped_gemm.grouped_gemm_impl import (
    grouped_gemm_variable_k_impl,
)
from primus_turbo.pytorch.kernels.mega_moe.mega_moe_backward_fp8_impl import (
    _DW_FP8_FORMAT,
    _mxfp8_step1_dispatch_dgrad,
    _mxfp8_variable_k_wgrad,
)

_H_GROUP_LENS = 9
_H_GROUP_OFFS = 10


def _routing(T, K, E, *, device, seed):
    import os
    if os.environ.get("PT_BALANCED_ROUTING", "0") == "1":
        # balanced round-robin: each token -> K distinct experts, every expert gets exactly T*K/E
        # tokens/rank -> ZERO pool padding (m_pad minimal). Isolates GEMM efficiency from load
        # imbalance / padding-tail effects.
        j = torch.arange(K, device=device).reshape(1, K)
        base = (torch.arange(T, device=device).reshape(T, 1) * K + j) % E
        w = torch.full((T, K), 1.0 / K, device=device, dtype=torch.float32)
        return base.to(torch.int64), w
    g = torch.Generator(device=device).manual_seed(seed)
    gate = torch.randn(T, E, generator=g, device=device)
    w0, idx = torch.sigmoid(gate).topk(K, dim=-1)
    w = (w0 / (w0.sum(-1, keepdim=True) + 1e-20)).to(torch.float32)
    return idx.to(torch.int64), w


def _global_weights(E, I, H, device):
    g = torch.Generator(device=device).manual_seed(1234)
    W1 = torch.randn((E, 2 * I, H), generator=g, device=device, dtype=torch.bfloat16) * (2.0 / math.sqrt(H))
    W2 = torch.randn((E, H, I), generator=g, device=device, dtype=torch.bfloat16) * (2.0 / math.sqrt(I))
    return W1, W2


def _snr_db(ref, out):
    ref, out = ref.float(), out.float()
    return float(10.0 * torch.log10(ref.pow(2).sum() / ((ref - out).pow(2).sum() + 1e-12)))


def _bench(fn, *, warmup, iters, group):
    # BACK-TO-BACK timing (matches the source test_dw2_bench): enqueue `iters` calls between two
    # events and divide. Timing each call individually (event-bracketed single fn) would count the
    # host-side custom-op dispatch / autotune-lookup as GPU-idle and inflate fast fp8 kernels.
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

    # mirror the source test_dw2_bench RNG sequence EXACTLY (seed 123+rank; draw x, w1, w2, gate in
    # this order on the GLOBAL RNG) so the routing -> per-expert token partition (m_pad) reproduces
    # the source's 66048 pool -> apples-to-apples dW2 GEMM throughput vs the source bench.
    torch.manual_seed(123 + rank)
    x = torch.randn(T, H, device="cuda", dtype=torch.bfloat16)
    W1 = torch.randn(epr, 2 * I, H, device="cuda", dtype=torch.bfloat16) * 0.05
    W2 = torch.randn(epr, H, I, device="cuda", dtype=torch.bfloat16) * 0.05
    gate = torch.randn(T, E, device="cuda")
    topk_w0, topk_idx = torch.sigmoid(gate).topk(K, dim=-1)
    topk_w = (topk_w0 / (topk_w0.sum(-1, keepdim=True) + 1e-20)).to(torch.float32)

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

    # forward L1 -> l1 + dispatch_weights (the STEP2 inputs)
    w1q, w1s = quantize_grouped_weight_mxfp8(W1)
    torch.cuda.synchronize(); group.barrier()
    symm.scoreboard.zero_()
    torch.cuda.synchronize(); group.barrier()
    l1 = dispatch_grouped_gemm_mxfp8(x, None, w1q, w1s, handle, sym_layout, symm, BM=BM, BN=BN)
    dispatch_weights = symm.weight_recv_buf.clone()

    dy = torch.randn((T, H), device="cuda", dtype=torch.bfloat16)
    # STEP1 (dispatch(dy)+fc2 dgrad) -> grad_swiglu + dispatched-dy fp8 pool
    grad_swiglu, pool_handle = _mxfp8_step1_dispatch_dgrad(dy, W2, group, handle, BM, BN)
    # STEP2 (swiglu^T, re-inject routing weight) -> act_weighted (the dW2 b operand)
    _, _, act_weighted = swiglu_backward(
        grad_swiglu, l1, scale=dispatch_weights, return_gate=True, return_act_w=True,
    )
    group_lens, group_offs = handle[_H_GROUP_LENS], handle[_H_GROUP_OFFS]

    # dequant the SAME dispatched-dy pool -> dispatch_l2_grad (bf16 wgrad `a` operand)
    pool_fp8, pool_scale = pool_handle
    P, Hh = pool_fp8.shape
    pf = pool_fp8.to(torch.float32).view(P, Hh // 32, 32)
    ps = pool_scale.reshape(P, Hh // 32).view(torch.uint8).to(torch.int32)
    sc = torch.exp2((ps - 127).to(torch.float32)).view(P, Hh // 32, 1)
    dl2 = (pf * sc).view(P, Hh).to(torch.bfloat16)

    def _fp8():  # requant fp8 pool colwise + colwise-quant act -> mxfp8 variable-K wgrad
        return _mxfp8_variable_k_wgrad(pool_handle, act_weighted, group_lens, group_offs)

    def _bf16():  # bf16 variable-K wgrad on the dequant'd pool
        return grouped_gemm_variable_k_impl(
            dl2, act_weighted, group_lens, group_offs,
            trans_a=True, trans_b=False, trans_c=False, num_cu=None,
            default_backend=BackendType.TRITON.value,
        )

    # BREAKDOWN: pre-quantize the two operands ONCE (outside the timed loop) so we can time the
    # ISOLATED fp8 variable-K GEMM (the ~2x-vs-bf16 win) apart from the requant(pool)/quant(act)
    # overhead that the `full` fp8 wgrad pays per call. Mirrors the source test_dw2_bench breakdown.
    meta0 = colwise_grouped_meta(group_lens, group_offs)
    a_t, a_ts, lens_pc, offs_pc = colwise_requant_mxfp8_grouped_fp8in_flydsl(
        pool_fp8, pool_scale, _DW_FP8_FORMAT, meta=meta0
    )
    b_t, b_ts, _, _ = colwise_quant_mxfp8_grouped_flydsl(act_weighted, _DW_FP8_FORMAT, meta=meta0)

    def _gemm():  # isolated fp8 variable-K GEMM on pre-quantized operands (no requant/quant)
        return grouped_gemm_fp8_variable_k_impl(
            a_t, b_t, a_ts.view(torch.float8_e8m0fnu), b_ts.view(torch.float8_e8m0fnu),
            lens_pc.to(torch.int64), offs_pc.to(torch.int64),
            trans_a=False, trans_b=False, trans_c=False, out_dtype=torch.bfloat16,
            granularity=ScalingGranularity.MX_BLOCKWISE.value, num_cu=None,
            default_backend=BackendType.FLYDSL.value,
        )

    def _req():  # requant the fp8 pool colwise (dW2 `a` operand producer fusion)
        return colwise_requant_mxfp8_grouped_fp8in_flydsl(pool_fp8, pool_scale, _DW_FP8_FORMAT, meta=meta0)

    def _qnt():  # colwise-quant act_weighted (dW2 `b` operand)
        return colwise_quant_mxfp8_grouped_flydsl(act_weighted, _DW_FP8_FORMAT, meta=meta0)

    dW2_fp8 = _fp8()
    dW2_bf = _bf16()
    assert tuple(dW2_fp8.shape) == (epr, H, I), f"dW2 fp8 shape {tuple(dW2_fp8.shape)} != {(epr, H, I)}"
    snr = _snr_db(dW2_bf, dW2_fp8)
    nan = bool((~torch.isfinite(dW2_fp8.float())).any())

    t_dw2 = _bench(_fp8, warmup=args.warmup, iters=args.iters, group=group)
    t_dw2_bf = _bench(_bf16, warmup=args.warmup, iters=args.iters, group=group)
    t_gemm = _bench(_gemm, warmup=args.warmup, iters=args.iters, group=group)
    t_req = _bench(_req, warmup=args.warmup, iters=args.iters, group=group)
    t_qnt = _bench(_qnt, warmup=args.warmup, iters=args.iters, group=group)
    m_pad = int(group_offs[-1].item())
    flops = 2.0 * m_pad * H * I
    symm.destroy()
    return {"snr": snr, "nan": float(nan), "t_dw2": t_dw2, "t_dw2_bf": t_dw2_bf,
            "t_gemm": t_gemm, "t_req": t_req, "t_qnt": t_qnt, "flops": flops, "m_pad": m_pad}


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
        t_dw2, t_dw2_bf = _amax(group, r["t_dw2"]), _amax(group, r["t_dw2_bf"])
        t_gemm, t_req, t_qnt = _amax(group, r["t_gemm"]), _amax(group, r["t_req"]), _amax(group, r["t_qnt"])
        if rank == 0:
            tf = lambda ms: r["flops"] / (ms * 1e-3) / 1e12
            print(f"\n{'='*72}")
            print(f"[backward dW2  fc2 wgrad (variable-K)  fp8 vs bf16]  EP{world} T={args.num_tokens} "
                  f"H={args.hidden} I={args.inter} E={args.num_experts} K={args.num_topk}")
            print(f"{'='*72}")
            print(f"  dW2 fp8 FULL : {t_dw2:8.3f} ms | {tf(t_dw2):8.1f} TFLOPS  (M_pool={r['m_pad']})  [requant+quant+GEMM]")
            print(f"  dW2 bf16 ref : {t_dw2_bf:8.3f} ms | {tf(t_dw2_bf):8.1f} TFLOPS")
            print(f"  fp8/bf16     : FULL {t_dw2 / t_dw2_bf:.3f}x  |  GEMM-only {t_gemm / t_dw2_bf:.3f}x  "
                  f"({'fp8 faster' if t_dw2 < t_dw2_bf else 'fp8 SLOWER'} full)")
            print(f"  breakdown    : requant(pool)={t_req:.3f}  quant(act)={t_qnt:.3f}  "
                  f"GEMM={t_gemm:.3f} ms ({tf(t_gemm):.0f} TFLOPS)  [sum={t_req + t_qnt + t_gemm:.3f}]")
            print(f"  [acc] dW2 fp8 vs bf16: SNR={snr:.2f} dB  nan={bool(nan)}  "
                  f"{'PASS' if snr >= 15.0 and not nan else 'FAIL'} (gate SNR>=15dB)")
        torch.cuda.synchronize(); group.barrier()
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="backward dW2 (fc2 wgrad) fp8 vs bf16")
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
