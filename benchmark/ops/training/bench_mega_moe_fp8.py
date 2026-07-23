###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Benchmark the MXFP8 mega MoE stages (fp8-only), EP, intra-node.

The BF16 reference legs were SPLIT OUT into ``bench_mega_moe_bf16.py`` (run it separately and
compare offline) so the fp8 and bf16 stacks can never poison each other -- e.g. the large-K STEP3
bf16 combine OOB must not take down the fp8 stage bench. This file runs ONLY the fp8 legs; per-stage
correctness gates that need a reference use a self-contained torch/e2e reference, not the bf16 kernel.

Built stage by stage. Stage 1 (``--stage l1``): L1 = fused dispatch + fc1.
  * fp8  : vendored fp8 stack ``dispatch_grouped_gemm_mxfp8`` (mxfp8 clean-push + preshuffle + NT GEMM)
  * bf16 : shipped ``dispatch_grouped_gemm_bf16_flydsl_kernel(layout="nt")``
Both timed as the FULL per-forward L1 (prologue + dispatch + fc1) so the comparison is apples-to-
apples (the bf16 nt kernel cannot be safely reused back-to-back, so each call re-runs its prologue;
the fp8 leg matches by rebuilding its handle each call, and its comm gates self-reset on device via
the epoch bump -- exactly one training forward). fp8 correctness gated vs a torch dequant
grouped-GEMM over the kernel's own pool.

Stage ``--stage fwd``: the FULL forward (L1 + SwiGLU + L2) via the actual ops -- fp8
``mega_moe_fused_fp8`` vs shipped bf16 ``mega_moe_fused`` on identical inputs. Each op is
self-contained (own prologue/dispatch/combine/reset per call) so this is the true per-forward
wall-time; persistent W1/W2 make the fp8 op's version-keyed weight-quant cache HIT (as in training).
Accuracy = SNR(fp8 y vs bf16 y), gate >=18 dB. NOTE round_robin trips the fp8 L2 combine cross-rank
race (NaN/GPU fault); use ``--mode load_balanced`` for the clean full-forward number.

Backward stage ``--stage dispatch_fc2_dgrad``: the fused cross-rank dispatch(dy) PUSH + grouped
fc2-dgrad GEMM (grad_swiglu = dispatched_dy @ w2, NN). fp8 ``_dispatch_l2_dgrad_mxfp8_flydsl_kernel`` (fp8 PUSH
halves comm bytes + mxfp8 ~2x GEMM, comm hidden under the GEMM) vs the bf16 backward's own L2-dgrad
``dispatch_grouped_gemm_bf16_flydsl_kernel(dy, w2, layout='nn')``. Fast fused kernels -> back-to-back
timing; accuracy = grad_swiglu SNR vs a per-group bf16 ref. (Not part of ``both``.)

Backward stage ``--stage fc2_wgrad``: the fc2 weight grad dW2 (MXFP8 variable-K wgrad) --
dW2 = dispatch_l2_grad^T @ act_weighted over the pool. fp8 ``_mxfp8_variable_k_wgrad`` (colwise
requant the fp8 pool + colwise-quant act -> fp8 GEMM) vs shipped bf16 Triton
``grouped_gemm_variable_k_impl``. Reports FULL vs GEMM-only + requant/quant/GEMM breakdown (the
isolated fp8 GEMM is ~2x, per-call quant/requant dilute FULL to ~parity). SNR gate. (Not in ``both``.)

Backward stage ``--stage fc1_wgrad``: the fc1 weight grad dW1 (MXFP8 variable-K wgrad, LOCAL) --
dW1 = grad_l1^T @ pool_x over the forward-dispatched fc1-input pool (reused, NO cross-rank
re-dispatch). fp8 ``_mxfp8_variable_k_wgrad_dw1`` (colwise-quant grad_l1 + colwise-requant fp8
pool_x -> fp8 GEMM) vs shipped bf16 Triton ``grouped_gemm_variable_k_impl``. Same FULL/GEMM-only +
breakdown + SNR gate as fc2_wgrad. (Not in ``both``.)

Backward stage ``--stage fc1_dgrad_combine``: STEP3 = fc1 dgrad (grad_l1 @ w1^T) + combine + reduce
+ grad_gate scatter -> dx [T,H]. fp8 ``grouped_gemm_combine_mxfp8_flydsl_kernel(grad_gate=...)`` (fp8-PUSH, fp8 stack) vs the
shipped bf16 ``grouped_gemm_combine_bf16_flydsl_kernel(layout='nn')`` (separate bf16 stack -- the
fp8/bf16 dispatch handles are not interchangeable), kernel-only timing -> LATENCY compare (dx SNR
via e2e gradcheck). The fp8-PUSH combine can intermittently spin-deadlock at large T (~5-15%) and
round_robin trips it -- use load_balanced, re-run w/ fresh MASTER_PORT if it hangs. (Not in ``both``.)

Run inside the dev container (8 GPUs):
  PYTHONPATH=<repo> python benchmark/ops/training/bench_mega_moe_fp8.py --num-tokens 8192 --stage l1
  PYTHONPATH=<repo> python benchmark/ops/training/bench_mega_moe_fp8.py --num-tokens 8192 --stage fwd --mode load_balanced
  PYTHONPATH=<repo> python benchmark/ops/training/bench_mega_moe_fp8.py --num-tokens 8192 --stage dispatch_fc2_dgrad --mode load_balanced
  PYTHONPATH=<repo> python benchmark/ops/training/bench_mega_moe_fp8.py --num-tokens 8192 --stage fc2_wgrad --mode load_balanced
  PYTHONPATH=<repo> python benchmark/ops/training/bench_mega_moe_fp8.py --num-tokens 8192 --stage fc1_wgrad --mode load_balanced
  PYTHONPATH=<repo> python benchmark/ops/training/bench_mega_moe_fp8.py --num-tokens 8192 --stage fc1_dgrad_combine --mode load_balanced
"""

import argparse
import datetime
import math
import os
import sys

import numpy as np
import torch
import torch.distributed as dist

# config (get_platform_info) is same-dir; repo root for primus_turbo.
_HERE = os.path.dirname(__file__)
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..", "..")))

from config import get_platform_info  # noqa: E402

import primus_turbo.pytorch  # noqa: E402,F401
from primus_turbo.flydsl.mega import (  # noqa: E402  (SwiGLU fwd/bwd; shared with the bf16 stack)
    swiglu_backward_flydsl_kernel,
    swiglu_flydsl_kernel,
)
from primus_turbo.flydsl.mega.fp8 import (  # noqa: E402  (vendored fp8 stack)
    colwise_grouped_meta,
    colwise_quant_mxfp8_grouped_flydsl,
    colwise_requant_mxfp8_grouped_fp8in_flydsl,
    dispatch_grouped_gemm_mxfp8,
    dispatch_prologue,
    get_symm_buffer_for_mega_moe,
    grouped_gemm_combine_mxfp8_flydsl_kernel,
    quantize_grouped_weight_mxfp8,
    quantize_rowwise_mxfp8_flydsl,
)
from primus_turbo.pytorch.kernels.mega_moe.weight_prep_fp8 import prepare_w2_fp8  # noqa: E402
from primus_turbo.pytorch.core.backend import BackendType  # noqa: E402
from primus_turbo.pytorch.core.low_precision import ScalingGranularity  # noqa: E402
from primus_turbo.pytorch.kernels.grouped_gemm.grouped_gemm_fp8_impl import (  # noqa: E402
    grouped_gemm_fp8_variable_k_impl,
)
from primus_turbo.pytorch.kernels.mega_moe.mega_moe_backward_fp8_impl import (  # noqa: E402  (fp8 bwd stages)
    _DW_FP8_FORMAT,
    _dispatch_l2_dgrad_mxfp8_flydsl_kernel,
    _mxfp8_variable_k_wgrad,
    _mxfp8_variable_k_wgrad_dw1,
)
from primus_turbo.pytorch.ops.moe.mega_moe_fused_fp8 import mega_moe_fused_fp8  # noqa: E402  (fp8 op)

_MXFP8_BLOCK = 32
_H_TILE_TO_EXPERT = 7
_H_GROUP_LENS = 9
_H_GROUP_OFFS = 10


def generate_inputs(rank, world, *, T, H, I, E, K, mode, device="cuda"):
    """One rank's local MoE inputs: x, this rank's L1/L2 expert shard, and top-k routing in the
    requested ``mode`` (matches the source mega_utils.generate_routing):
      * load_balanced : rand-softmax top-k (~balanced with variance -> some pool padding)
      * round_robin   : arange(T*K) % E (deterministic; exactly T*K/E per expert -> zero padding)
    Seeded per rank."""
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


def _dequant_mxfp8(q, s_raw, block=_MXFP8_BLOCK):
    *lead, K = q.shape
    qf = q.float().view(*lead, K // block, block)
    scale = torch.exp2(s_raw.view(torch.uint8).float() - 127.0).unsqueeze(-1)
    return (qf * scale).view(*lead, K)


def _bench(fn, *, warmup, iters, group, reset=None):
    """Per-call CUDA-event latency. Optional ``reset`` runs OUTSIDE the timed window each iter (the
    epoch-self-reset kernels need none). sync+barrier before each call serializes ranks (safe
    back-to-back even for the cross-rank fp8 kernels)."""
    ev_s = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ev_e = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]

    def _one(s=None, e=None):
        torch.cuda.synchronize(); group.barrier()
        if reset is not None:
            reset()
        torch.cuda.synchronize(); group.barrier()
        if s is None:
            fn(); return
        s.record(); fn(); e.record()

    for _ in range(warmup):
        _one()
    for i in range(iters):
        _one(ev_s[i], ev_e[i])
    torch.cuda.synchronize()
    return float(np.average([s.elapsed_time(e) for s, e in zip(ev_s, ev_e)][1:]))


def _bench_b2b(fn, *, warmup, iters, group):
    """Back-to-back latency (N calls / N). For the fast fused backward kernels: single-call
    event-bracket timing (``_bench``) counts host custom-op dispatch / autotune-lookup as GPU-idle
    and inflates them -- back-to-back (source test_dw2_bench methodology) is the correct measure."""
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


def _snr_db(ref, out):
    ref, out = ref.float(), out.float()
    return float(10.0 * torch.log10(ref.pow(2).sum() / ((ref - out).pow(2).sum() + 1e-12)))


def _amax(group, v):
    t = torch.tensor([v], device="cuda"); dist.all_reduce(t, op=dist.ReduceOp.MAX, group=group); return float(t)


def _amin(group, v):
    t = torch.tensor([v], device="cuda"); dist.all_reduce(t, op=dist.ReduceOp.MIN, group=group); return float(t)


def profile_l1(group, args, mode):
    """Stage 1: L1 = dispatch + fc1, fp8 (vendored) vs bf16 (shipped). Both FULL per-forward."""
    rank, world = group.rank(), group.size()
    H, I, E, K, T, BM, BN = args.hidden, args.inter, args.num_experts, args.num_topk, args.num_tokens, args.bm, args.bn
    epr = E // world
    N = 2 * I

    x, W1, _W2, topk_idx, topk_w = generate_inputs(rank, world, T=T, H=H, I=I, E=E, K=K, mode=mode)

    symm = get_symm_buffer_for_mega_moe(
        group, num_experts=E, num_max_tokens_per_rank=T, num_topk=K, hidden=H,
        intermediate_hidden=I, block_m=BM, block_n=BN, use_mxfp8=True,
    )
    sym_layout = symm.make_sym_layout()
    w1q, w1s = quantize_grouped_weight_mxfp8(W1)  # static weight quant (module-owned in production)

    def _prologue():
        return tuple(dispatch_prologue(
            topk_idx, topk_w, sym_layout=sym_layout, num_tokens=T, num_topk=K, num_experts=E,
            world_size=world, rank=rank, experts_per_rank=epr, block_m=BM,
            num_max_pool_tokens=symm.num_max_pool_tokens,
        ))

    def _fp8():  # FULL per-forward L1: prologue + fused dispatch+GEMM (epoch self-reset on device)
        h = _prologue()
        return dispatch_grouped_gemm_mxfp8(x, None, w1q, w1s, h, sym_layout, symm, BM=BM, BN=BN)

    # ── correctness (fp8): one L1, then torch dequant grouped-GEMM over the dispatched pool ──
    handle = _prologue()
    torch.cuda.synchronize(); group.barrier()
    out = dispatch_grouped_gemm_mxfp8(x, None, w1q, w1s, handle, sym_layout, symm, BM=BM, BN=BN)
    torch.cuda.synchronize(); group.barrier()
    real_tiles = int(symm.meta_scalars[1].item())
    M_eff = real_tiles * BM
    A = _dequant_mxfp8(symm.pool_fp8[:M_eff], symm.pool_scale[:M_eff])
    Wd = _dequant_mxfp8(w1q, w1s)
    row_expert = handle[_H_TILE_TO_EXPERT][:real_tiles].to(torch.long).repeat_interleave(BM)
    ref = torch.empty((M_eff, N), device="cuda", dtype=torch.float32)
    for gi in torch.unique(row_expert).tolist():
        m = row_expert == gi
        ref[m] = A[m] @ Wd[gi].t()
    o = out[:M_eff].float()
    cos = float(torch.dot(o.flatten(), ref.flatten()) / (o.norm() * ref.norm() + 1e-12))
    rel = float((o - ref).norm() / (ref.norm() + 1e-12))
    m_pad = int(handle[_H_GROUP_OFFS][-1].item())
    del A, Wd, ref, o, out

    t_fp8 = _bench(_fp8, warmup=args.warmup, iters=args.iters, group=group)
    flops = 2.0 * M_eff * N * H
    symm.destroy()
    return {"cos": cos, "rel": rel, "fp8_ms": t_fp8, "flops": flops, "M_eff": M_eff, "m_pad": m_pad}


def profile_l2(group, args, mode):
    """Stage 2: L2 = fc2 GEMM + combine PUSH + weighted top-k reduce, fp8 (vendored) vs bf16 (shipped).

    Both need the pool populated + a SwiGLU activation, so each leg runs its own L1 -> swiglu -> act
    first (setup, untimed), then times ONLY the L2 combine per-forward:
      * fp8  : the fp8 combine (self-resets its epoch flags on device -- no host reset cost)
      * bf16 : shipped combine reused back-to-back (self-managed cross-rank state; timing-valid)."""
    rank, world = group.rank(), group.size()
    H, I, E, K, T, BM, BN = args.hidden, args.inter, args.num_experts, args.num_topk, args.num_tokens, args.bm, args.bn
    epr = E // world

    x, W1, W2, topk_idx, topk_w = generate_inputs(rank, world, T=T, H=H, I=I, E=E, K=K, mode=mode)
    topk_w_f32 = topk_w.to(torch.float32)

    # ── fp8 leg: L1 -> swiglu -> act (setup), prepare w2, then time the fp8 combine ──
    symm = get_symm_buffer_for_mega_moe(
        group, num_experts=E, num_max_tokens_per_rank=T, num_topk=K, hidden=H,
        intermediate_hidden=I, block_m=BM, block_n=BN, use_mxfp8=True,
    )
    sym_layout = symm.make_sym_layout()
    handle = tuple(dispatch_prologue(
        topk_idx, topk_w, sym_layout=sym_layout, num_tokens=T, num_topk=K, num_experts=E,
        world_size=world, rank=rank, experts_per_rank=epr, block_m=BM,
        num_max_pool_tokens=symm.num_max_pool_tokens,
    ))
    w1q, w1s = quantize_grouped_weight_mxfp8(W1)
    torch.cuda.synchronize(); group.barrier()
    l1 = dispatch_grouped_gemm_mxfp8(x, None, w1q, w1s, handle, sym_layout, symm, BM=BM, BN=BN)
    act = swiglu_flydsl_kernel(l1, symm.meta_scalars[1:2]).contiguous()  # [M, I] bf16 (contiguous: combine reads flat)
    w2_fp8 = prepare_w2_fp8(W2)               # static weight prep (module-owned in production)
    real_tiles = int(symm.meta_scalars[1].item())
    M_eff = real_tiles * BM
    m_pad = int(handle[_H_GROUP_OFFS][-1].item())

    def _fp8():  # fp8 L2 combine kernel (GEMM + PUSH + reduce); self-resets the L2 flags internally
        y, _ = grouped_gemm_combine_mxfp8_flydsl_kernel(
            act, w2_fp8, list(handle), group, topk_indices=topk_idx, topk_weights=topk_w_f32,
            BM=BM, BN=BN, num_combine_cu=48,
        )
        return y

    y = _fp8()
    torch.cuda.synchronize(); group.barrier()
    fin = bool(torch.isfinite(y.float()).all())
    y_norm = float(y.float().norm())
    t_fp8 = _bench(_fp8, warmup=args.warmup, iters=args.iters, group=group)
    flops = 2.0 * M_eff * H * I  # fc2 GEMM: [M,I] @ [I,H] -> [M,H]  (N=H, K=I)
    symm.destroy()
    return {"fin": float(fin), "y_norm": y_norm, "fp8_ms": t_fp8, "flops": flops,
            "M_eff": M_eff, "m_pad": m_pad}


def profile_fwd(group, args, mode):
    """Full forward: L1 (dispatch+fc1) + SwiGLU + L2 (fc2+combine), via the actual ops --
    fp8 ``mega_moe_fused_fp8`` vs the shipped bf16 ``mega_moe_fused`` on identical inputs.

    Each op is self-contained (own prologue + dispatch + reset + combine per call), so this is the
    true per-forward wall-time. Persistent W1/W2 -> the fp8 op's version-keyed weight-quant cache
    HITS (mirrors training). Accuracy = SNR(fp8 y vs bf16 y). NOTE: round_robin trips the fp8 L2
    combine's cross-rank race (NaN/GPU fault) -- load_balanced (real routing) is clean."""
    rank, world = group.rank(), group.size()
    H, I, E, K, T = args.hidden, args.inter, args.num_experts, args.num_topk, args.num_tokens
    x, W1, W2, topk_idx, topk_w = generate_inputs(rank, world, T=T, H=H, I=I, E=E, K=K, mode=mode)

    with torch.no_grad():
        def _fp8():
            return mega_moe_fused_fp8(group, x, topk_idx, topk_w, W1, W2)

        y_fp8 = _fp8()
        fin = bool(torch.isfinite(y_fp8.float()).all())
        t_fp8 = _bench(_fp8, warmup=args.warmup, iters=args.iters, group=group)
    # full-forward ms only (mixed-stage TFLOPS not meaningful; fp8-vs-bf16 SNR -> bench_mega_moe_fused_fp8).
    return {"fin": float(fin), "fp8_ms": t_fp8}


def profile_dispatch_fc2_dgrad(group, args, mode):
    """Backward dispatch(dy) + fc2-dgrad = fused cross-rank dispatch(dy) PUSH + grouped fc2-dgrad
    GEMM (grad_swiglu = dispatched_dy @ w2, NN, contract H -> [P, I]). Both legs the SAME fused op:
      * fp8  : ``_dispatch_l2_dgrad_mxfp8_flydsl_kernel`` (fp8 PUSH byte-halved comm + mxfp8 ~2x GEMM), fp8 stack
      * bf16 : ``dispatch_grouped_gemm_bf16_flydsl_kernel(dy, w2, layout='nn')`` -- exactly the
               L2-dgrad the bf16 ``mega_moe_fused`` backward uses, bf16 stack.
    Separate symm globals: run fp8 fully, ``destroy()``, then bf16 (no coexistence). Fast fused
    kernels -> back-to-back timing. Accuracy: grad_swiglu vs per-group bf16 ref over this stage's own
    dispatched-dy pool (SNR)."""
    rank, world = group.rank(), group.size()
    H, I, E, K, T, BM, BN = args.hidden, args.inter, args.num_experts, args.num_topk, args.num_tokens, args.bm, args.bn
    epr = E // world

    x, W1, W2, topk_idx, topk_w = generate_inputs(rank, world, T=T, H=H, I=I, E=E, K=K, mode=mode)
    g = torch.Generator(device="cuda").manual_seed(4321 + rank)
    dy = torch.randn((T, H), generator=g, device="cuda", dtype=torch.bfloat16)

    # ─────────────── fp8 dispatch(dy)+fc2-dgrad (fp8 mega stack) ───────────────
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
    dispatch_grouped_gemm_mxfp8(x, None, w1q, w1s, handle, sym_layout, symm, BM=BM, BN=BN)  # fwd L1 (setup)

    def _fp8():
        return _dispatch_l2_dgrad_mxfp8_flydsl_kernel(dy, W2, group, handle, BM, BN)

    grad_swiglu_fp8, pool_handle = _fp8()
    grad_swiglu_fp8 = grad_swiglu_fp8.clone()
    offs = handle[_H_GROUP_OFFS]
    c_m = int(offs[-1].item())
    # per-group bf16 ref over this stage's OWN dispatched-dy pool -> fp8 grad_swiglu correctness
    pool_fp8, pool_scale = pool_handle
    P, Hh = pool_fp8.shape
    pf = pool_fp8.to(torch.float32).view(P, Hh // 32, 32)
    ps = pool_scale.reshape(P, Hh // 32).view(torch.uint8).to(torch.int32)
    sc = torch.exp2((ps - 127).to(torch.float32)).view(P, Hh // 32, 1)
    dl2 = (pf * sc).view(P, Hh).to(torch.bfloat16)
    ref = torch.zeros_like(grad_swiglu_fp8)
    for gi in range(epr):
        lo, hi = int(offs[gi].item()), int(offs[gi + 1].item())
        if hi > lo:
            ref[lo:hi] = (dl2[lo:hi].float() @ W2[gi].float()).to(torch.bfloat16)
    snr = _snr_db(ref[:c_m], grad_swiglu_fp8[:c_m])
    nan = bool((~torch.isfinite(grad_swiglu_fp8.float())).any())

    t_fp8 = _bench_b2b(_fp8, warmup=args.warmup, iters=args.iters, group=group)
    M_eff = c_m
    symm.destroy()
    flops = 2.0 * M_eff * I * H  # dgrad GEMM [P,H] @ [H,I]
    return {"snr": snr, "nan": float(nan), "fp8_ms": t_fp8, "flops": flops, "M_eff": M_eff}


def profile_fc2_wgrad(group, args, mode):
    """Backward fc2 wgrad (dW2), MXFP8 variable-K wgrad. Replays the backward up to dW2 on the real
    mega pool: fwd L1 -> (l1, dispatch_weights); dispatch(dy)+fc2-dgrad -> grad_swiglu + the
    dispatched-dy fp8 pool; swiglu_backward -> act_weighted. Then dW2 = dispatch_l2_grad^T @
    act_weighted (variable-K over the pool) BOTH ways:
      * fp8  : ``_mxfp8_variable_k_wgrad`` (colwise-requant the fp8 pool + colwise-quant act -> fp8 GEMM)
      * bf16 : ``grouped_gemm_variable_k_impl`` (Triton) on the dequant'd pool.
    Back-to-back timing; SNR gate. Also reports FULL vs GEMM-only + requant/quant/GEMM breakdown --
    the isolated fp8 GEMM is the ~2x win, while the per-call colwise requant/quant dilute it to
    ~parity in FULL (dW2 is compute-bound; the fp8 value there is grad bytes/bandwidth, not latency)."""
    rank, world = group.rank(), group.size()
    H, I, E, K, T, BM, BN = args.hidden, args.inter, args.num_experts, args.num_topk, args.num_tokens, args.bm, args.bn
    epr = E // world

    x, W1, W2, topk_idx, topk_w = generate_inputs(rank, world, T=T, H=H, I=I, E=E, K=K, mode=mode)
    g = torch.Generator(device="cuda").manual_seed(4321 + rank)
    dy = torch.randn((T, H), generator=g, device="cuda", dtype=torch.bfloat16)

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
    # fwd L1 -> l1 + dispatch_weights (the swiglu_backward inputs)
    w1q, w1s = quantize_grouped_weight_mxfp8(W1)
    torch.cuda.synchronize(); group.barrier()
    l1 = dispatch_grouped_gemm_mxfp8(x, None, w1q, w1s, handle, sym_layout, symm, BM=BM, BN=BN)
    dispatch_weights = symm.weight_recv_buf.clone()

    # dispatch(dy)+fc2-dgrad -> grad_swiglu + dispatched-dy fp8 pool; swiglu_backward -> act_weighted
    grad_swiglu, pool_handle = _dispatch_l2_dgrad_mxfp8_flydsl_kernel(dy, W2, group, handle, BM, BN)
    _, _, act_weighted = swiglu_backward_flydsl_kernel(
        grad_swiglu, l1, symm.meta_scalars[1:2], scale=dispatch_weights, return_gate=True, return_act_w=True,
    )
    group_lens, group_offs = handle[_H_GROUP_LENS], handle[_H_GROUP_OFFS]
    pool_fp8, pool_scale = pool_handle

    def _fp8():  # requant fp8 pool colwise + colwise-quant act -> mxfp8 variable-K wgrad
        return _mxfp8_variable_k_wgrad(pool_handle, act_weighted, group_lens, group_offs)

    # BREAKDOWN: pre-quantize both operands ONCE (outside the timed loop) -> time the ISOLATED fp8
    # variable-K GEMM apart from the per-call requant(pool)/quant(act) the FULL fp8 wgrad pays.
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

    def _req():  # requant the fp8 pool colwise (dW2 `a` operand producer)
        return colwise_requant_mxfp8_grouped_fp8in_flydsl(pool_fp8, pool_scale, _DW_FP8_FORMAT, meta=meta0)

    def _qnt():  # colwise-quant act_weighted (dW2 `b` operand)
        return colwise_quant_mxfp8_grouped_flydsl(act_weighted, _DW_FP8_FORMAT, meta=meta0)

    def _meta():  # grouped meta (one total_M_pad D2H); shared by requant/quant inside the FULL op
        return colwise_grouped_meta(group_lens, group_offs)

    dW2_fp8 = _fp8()
    assert tuple(dW2_fp8.shape) == (epr, H, I), f"dW2 fp8 shape {tuple(dW2_fp8.shape)} != {(epr, H, I)}"
    nan = bool((~torch.isfinite(dW2_fp8.float())).any())

    t_fp8 = _bench_b2b(_fp8, warmup=args.warmup, iters=args.iters, group=group)
    t_gemm = _bench_b2b(_gemm, warmup=args.warmup, iters=args.iters, group=group)
    t_req = _bench_b2b(_req, warmup=args.warmup, iters=args.iters, group=group)
    t_qnt = _bench_b2b(_qnt, warmup=args.warmup, iters=args.iters, group=group)
    t_meta = _bench_b2b(_meta, warmup=args.warmup, iters=args.iters, group=group)
    m_pad = int(group_offs[-1].item())
    flops = 2.0 * m_pad * H * I  # wgrad GEMM: [P,H]^T @ [P,I] -> [H,I] over the pool
    symm.destroy()
    return {"nan": float(nan), "fp8_ms": t_fp8, "gemm_ms": t_gemm,
            "req_ms": t_req, "qnt_ms": t_qnt, "meta_ms": t_meta, "flops": flops, "m_pad": m_pad}


def profile_fc1_wgrad(group, args, mode):
    """Backward fc1 wgrad (dW1), MXFP8 variable-K wgrad, LOCAL. Replays backward up to dW1 on the
    real mega pool: fwd L1 -> (l1, dispatch_weights) + CLONE the forward-dispatched fc1-input pool
    (pool_x, native rowwise-fp8) BEFORE STEP1 overwrites symm.pool_fp8; dispatch(dy)+fc2-dgrad ->
    grad_swiglu; swiglu_backward -> grad_l1. Then dW1 = grad_l1^T @ pool_x (variable-K) BOTH ways:
      * fp8  : ``_mxfp8_variable_k_wgrad_dw1`` (colwise-quant grad_l1 + colwise-requant fp8 pool_x -> fp8 GEMM)
      * bf16 : ``grouped_gemm_variable_k_impl`` (Triton) on the dequant'd pool.
    LOCAL: reuses the forward-dispatched pool, so unlike the bf16 fused path there is NO cross-rank
    re-dispatch of saved_x -> the real fp8 advantage is LARGER than the GEMM-only ratio here (which
    compares both on the same local pool). Back-to-back timing; FULL vs GEMM-only + breakdown; SNR gate."""
    rank, world = group.rank(), group.size()
    H, I, E, K, T, BM, BN = args.hidden, args.inter, args.num_experts, args.num_topk, args.num_tokens, args.bm, args.bn
    epr = E // world

    x, W1, W2, topk_idx, topk_w = generate_inputs(rank, world, T=T, H=H, I=I, E=E, K=K, mode=mode)
    g = torch.Generator(device="cuda").manual_seed(4321 + rank)
    dy = torch.randn((T, H), generator=g, device="cuda", dtype=torch.bfloat16)

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
    # fwd L1 -> l1 + dispatch_weights; CLONE the fc1-input pool BEFORE STEP1 overwrites symm.pool_fp8
    w1q, w1s = quantize_grouped_weight_mxfp8(W1)
    torch.cuda.synchronize(); group.barrier()
    l1 = dispatch_grouped_gemm_mxfp8(x, None, w1q, w1s, handle, sym_layout, symm, BM=BM, BN=BN)
    dispatch_weights = symm.weight_recv_buf.clone()
    Pp, Hp = symm.pool_fp8.shape
    pool_x_fp8 = (symm.pool_fp8.clone(), symm.pool_scale.reshape(Pp, Hp // 32).clone())

    grad_swiglu, _ = _dispatch_l2_dgrad_mxfp8_flydsl_kernel(dy, W2, group, handle, BM, BN)
    grad_l1, _, _ = swiglu_backward_flydsl_kernel(
        grad_swiglu, l1, symm.meta_scalars[1:2], scale=dispatch_weights, return_gate=True, return_act_w=True,
    )
    group_lens, group_offs = handle[_H_GROUP_LENS], handle[_H_GROUP_OFFS]

    def _fp8():  # colwise-quant grad_l1 + colwise-requant fp8 pool_x -> mxfp8 variable-K wgrad (LOCAL)
        return _mxfp8_variable_k_wgrad_dw1(grad_l1, pool_x_fp8, group_lens, group_offs)

    # BREAKDOWN: pre-quantize both operands ONCE -> isolate the fp8 variable-K GEMM from the per-call
    # colwise quant(grad_l1)/requant(pool_x). (dW1: `a`=grad_l1 quant, `b`=pool_x requant.)
    pool_fp8, pool_scale = pool_x_fp8
    meta0 = colwise_grouped_meta(group_lens, group_offs)
    a_t, a_ts, lens_pc, offs_pc = colwise_quant_mxfp8_grouped_flydsl(grad_l1, _DW_FP8_FORMAT, meta=meta0)
    b_t, b_ts, _, _ = colwise_requant_mxfp8_grouped_fp8in_flydsl(pool_fp8, pool_scale, _DW_FP8_FORMAT, meta=meta0)

    def _gemm():  # isolated fp8 variable-K GEMM on pre-quantized operands
        return grouped_gemm_fp8_variable_k_impl(
            a_t, b_t, a_ts.view(torch.float8_e8m0fnu), b_ts.view(torch.float8_e8m0fnu),
            lens_pc.to(torch.int64), offs_pc.to(torch.int64),
            trans_a=False, trans_b=False, trans_c=False, out_dtype=torch.bfloat16,
            granularity=ScalingGranularity.MX_BLOCKWISE.value, num_cu=None,
            default_backend=BackendType.FLYDSL.value,
        )

    def _qnt():  # colwise-quant grad_l1 (dW1 `a` operand)
        return colwise_quant_mxfp8_grouped_flydsl(grad_l1, _DW_FP8_FORMAT, meta=meta0)

    def _req():  # requant the fp8 pool_x colwise (dW1 `b` operand)
        return colwise_requant_mxfp8_grouped_fp8in_flydsl(pool_fp8, pool_scale, _DW_FP8_FORMAT, meta=meta0)

    def _meta():  # grouped meta (one total_M_pad D2H); shared inside the FULL op
        return colwise_grouped_meta(group_lens, group_offs)

    dW1_fp8 = _fp8()
    assert tuple(dW1_fp8.shape) == (epr, 2 * I, H), f"dW1 fp8 shape {tuple(dW1_fp8.shape)} != {(epr, 2 * I, H)}"
    nan = bool((~torch.isfinite(dW1_fp8.float())).any())

    t_fp8 = _bench_b2b(_fp8, warmup=args.warmup, iters=args.iters, group=group)
    t_gemm = _bench_b2b(_gemm, warmup=args.warmup, iters=args.iters, group=group)
    t_qnt = _bench_b2b(_qnt, warmup=args.warmup, iters=args.iters, group=group)
    t_req = _bench_b2b(_req, warmup=args.warmup, iters=args.iters, group=group)
    t_meta = _bench_b2b(_meta, warmup=args.warmup, iters=args.iters, group=group)
    m_pad = int(group_offs[-1].item())
    flops = 2.0 * m_pad * (2 * I) * H  # wgrad GEMM: [P,2I]^T @ [P,H] -> [2I,H] over the pool
    symm.destroy()
    return {"nan": float(nan), "fp8_ms": t_fp8, "gemm_ms": t_gemm,
            "req_ms": t_req, "qnt_ms": t_qnt, "meta_ms": t_meta, "flops": flops, "m_pad": m_pad}


def profile_fc1_dgrad_combine(group, args, mode):
    """Backward STEP3 = fc1 dgrad (grad_l1 @ w1^T) + combine + reduce + grad_gate scatter -> dx [T,H].
    fp8 vs bf16 on SEPARATE symm stacks (the fp8 / bf16 dispatch handles are NOT interchangeable --
    bf16 combine reads recv_* at handle[9..12] where the fp8 handle holds group_lens/offs), so this
    is a LATENCY comparison (dx SNR is gated by the e2e backward gradcheck, not here):
      * fp8  : fp8 stack -- real backward to grad_l1 + grad_gate, then ``grouped_gemm_combine_mxfp8_flydsl_kernel(grad_gate=...)``
               (fp8 fc1-dgrad + fp8-PUSH combine; w1^T prepped once). Kernel-only (reset outside window).
      * bf16 : bf16 stack -- shipped ``dispatch...bf16`` for a handle + a realistic grad_l1 replica,
               then ``grouped_gemm_combine_bf16_flydsl_kernel(layout='nn')``, same pool/routing.
    NOTE: the fp8-PUSH combine has an intermittent cross-rank reduce-flag race at large T (~5-15%
    spin-deadlock) + round_robin trips it like the forward L2 -- use load_balanced, re-run w/ fresh
    MASTER_PORT if it hangs."""
    rank, world = group.rank(), group.size()
    H, I, E, K, T, BM, BN = args.hidden, args.inter, args.num_experts, args.num_topk, args.num_tokens, args.bm, args.bn
    epr = E // world

    x, W1, W2, topk_idx, topk_w = generate_inputs(rank, world, T=T, H=H, I=I, E=E, K=K, mode=mode)
    topk_w_f32 = topk_w.to(torch.float32)
    g = torch.Generator(device="cuda").manual_seed(4321 + rank)
    dy = torch.randn((T, H), generator=g, device="cuda", dtype=torch.bfloat16)

    # ── fp8 leg: fp8 stack -- real backward to grad_l1/grad_gate, then fp8 STEP3 combine bwd ──
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
    # dispatch/combine gates self-reset on device (epoch) -> no host scoreboard/flag reset.
    l1 = dispatch_grouped_gemm_mxfp8(x, None, w1q, w1s, handle, sym_layout, symm, BM=BM, BN=BN)
    dispatch_weights = symm.weight_recv_buf.clone()
    grad_swiglu, _ = _dispatch_l2_dgrad_mxfp8_flydsl_kernel(dy, W2, group, handle, BM, BN)
    grad_l1, grad_gate = swiglu_backward_flydsl_kernel(grad_swiglu, l1, symm.meta_scalars[1:2], scale=dispatch_weights, return_gate=True)

    w1t_fp8 = prepare_w2_fp8(W1.transpose(1, 2).contiguous())  # w1^T fp8 prep (static weight, once)
    tidx64 = topk_idx.contiguous().view(-1)                    # fp8 combine-bwd takes int64
    m_pad = int(handle[_H_GROUP_OFFS][-1].item())
    flops = 2.0 * m_pad * (2 * I) * H  # fc1 dgrad GEMM: [P,2I] @ [2I,H] -> [P,H]

    def _reset_fp8():  # epoch self-reset (device) -> no host flag reset needed
        pass

    def _fp8():  # fp8 fc1-dgrad + fp8-PUSH combine (kernel only); grad_gate=... selects the bwd role
        dx, _ = grouped_gemm_combine_mxfp8_flydsl_kernel(
            grad_l1, w1t_fp8, list(handle), group, topk_indices=tidx64, grad_gate=grad_gate,
            BM=BM, BN=BN, num_combine_cu=16,
        )
        return dx

    torch.cuda.synchronize(); group.barrier(); _reset_fp8(); torch.cuda.synchronize(); group.barrier()
    dx_fp8 = _fp8()
    torch.cuda.synchronize(); group.barrier()
    fin = bool(torch.isfinite(dx_fp8.float()).all())
    dx_norm = float(dx_fp8.float().norm())
    t_fp8 = _bench(_fp8, warmup=args.warmup, iters=args.iters, group=group, reset=_reset_fp8)
    if rank == 0:  # emit fp8 STEP3 NOW so it survives a bf16-ref-leg abort at large T (see below)
        _tf = flops / (t_fp8 * 1e-3) / 1e12
        print(f"  [STEP3 fp8] {t_fp8:.3f} ms | {_tf:.1f} TFLOPS  dx finite={fin} (norm={dx_norm:.3e}; "
              f"M_pool={m_pad})", flush=True)
    symm.destroy()
    # bf16 STEP3 reference -> bench_mega_moe_bf16.py (split out; its large-K combine OOBs at large T).
    return {"fin": float(fin), "dx_norm": dx_norm, "fp8_ms": t_fp8, "flops": flops, "m_pad": m_pad}


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
    platform, gpu = get_platform_info() if rank == 0 else (None, None)
    modes = ["load_balanced", "round_robin"] if args.mode == "both" else [args.mode]
    try:
        for mode in modes:
            hdr = (f"{gpu}  EP{world} T={args.num_tokens} H={args.hidden} I={args.inter} "
                   f"E={args.num_experts} K={args.num_topk}  routing={mode}")
            if args.stage in ("l1", "both"):
                r = profile_l1(group, args, mode)
                cos, rel = _amin(group, r["cos"]), _amax(group, r["rel"])
                fp8_ms = _amax(group, r["fp8_ms"])
                if rank == 0:
                    tf = lambda ms: r["flops"] / (ms * 1e-3) / 1e12
                    print(f"\n{'='*80}\n[mega MoE L1 (dispatch+fc1)  fp8]  {hdr}\n{'='*80}")
                    print(f"  fp8  L1 : {fp8_ms:8.3f} ms | {tf(fp8_ms):8.1f} TFLOPS  (full per-forward; M_eff={r['M_eff']}, m_pad={r['m_pad']})")
                    print(f"  [acc] fp8 vs torch dequant-GEMM: cos={cos:.5f} rel={rel:.4f}  "
                          f"{'PASS' if cos >= 0.99 and rel <= 0.05 else 'FAIL'}  (bf16 ref -> bench_mega_moe_bf16.py)")
                torch.cuda.synchronize(); group.barrier()

            if args.stage in ("l2", "both"):
                r = profile_l2(group, args, mode)
                fin = _amin(group, r["fin"])
                fp8_ms = _amax(group, r["fp8_ms"])
                if rank == 0:
                    tf = lambda ms: r["flops"] / (ms * 1e-3) / 1e12
                    print(f"\n{'='*80}\n[mega MoE L2 (fc2+combine)  fp8]  {hdr}\n{'='*80}")
                    print(f"  fp8  L2 : {fp8_ms:8.3f} ms | {tf(fp8_ms):8.1f} TFLOPS  (kernel-only fc2 GEMM+PUSH+reduce; M_eff={r['M_eff']}, m_pad={r['m_pad']})")
                    print(f"  [acc] fp8 y finite={bool(fin >= 1.0)} (norm={r['y_norm']:.3e})  "
                          f"(rigorous SNR -> test_forward_mxfp8; bf16 ref -> bench_mega_moe_bf16.py)")
                torch.cuda.synchronize(); group.barrier()

            if args.stage in ("fwd", "both"):
                r = profile_fwd(group, args, mode)
                fin = _amin(group, r["fin"])
                fp8_ms = _amax(group, r["fp8_ms"])
                if rank == 0:
                    print(f"\n{'='*80}\n[mega MoE FULL forward (L1+SwiGLU+L2)  fp8]  {hdr}\n{'='*80}")
                    print(f"  fp8  fwd : {fp8_ms:8.3f} ms  (mega_moe_fused_fp8, full per-forward)")
                    print(f"  [acc] fp8 y finite={bool(fin >= 1.0)}  "
                          f"(fp8-vs-bf16 SNR -> bench_mega_moe_fused_fp8; bf16 ms -> bench_mega_moe_bf16.py)")
                torch.cuda.synchronize(); group.barrier()

            if args.stage == "dispatch_fc2_dgrad":
                r = profile_dispatch_fc2_dgrad(group, args, mode)
                snr, nan = _amin(group, r["snr"]), _amax(group, r["nan"])
                fp8_ms = _amax(group, r["fp8_ms"])
                if rank == 0:
                    tf = lambda ms: r["flops"] / (ms * 1e-3) / 1e12
                    print(f"\n{'='*80}\n[mega MoE bwd dispatch(dy)+fc2-dgrad  fp8]  {hdr}\n{'='*80}")
                    print(f"  fp8  : {fp8_ms:8.3f} ms | {tf(fp8_ms):8.1f} TFLOPS  (fused dispatch(dy) PUSH + fc2-dgrad GEMM; M_pool={r['M_eff']})")
                    print(f"  [acc] grad_swiglu SNR(fp8 vs per-group bf16 ref)={snr:.2f} dB  nan={bool(nan >= 1.0)}  "
                          f"{'PASS' if snr >= 18.0 and nan < 1.0 else 'FAIL'} (gate SNR>=18dB; bf16 ms -> bench_mega_moe_bf16.py)")
                torch.cuda.synchronize(); group.barrier()

            if args.stage == "fc2_wgrad":
                r = profile_fc2_wgrad(group, args, mode)
                nan = _amax(group, r["nan"])
                fp8_ms = _amax(group, r["fp8_ms"])
                gemm_ms, req_ms, qnt_ms = _amax(group, r["gemm_ms"]), _amax(group, r["req_ms"]), _amax(group, r["qnt_ms"])
                meta_ms = _amax(group, r["meta_ms"])
                if rank == 0:
                    tf = lambda ms: r["flops"] / (ms * 1e-3) / 1e12
                    print(f"\n{'='*80}\n[mega MoE bwd fc2 wgrad (dW2, variable-K)  fp8]  {hdr}\n{'='*80}")
                    print(f"  fp8  FULL : {fp8_ms:8.3f} ms | {tf(fp8_ms):8.1f} TFLOPS  (meta+requant+quant+GEMM; M_pool={r['m_pad']})")
                    print(f"  breakdown: meta={meta_ms:.3f}  requant(pool)={req_ms:.3f}  quant(act)={qnt_ms:.3f}  "
                          f"GEMM={gemm_ms:.3f} ms ({tf(gemm_ms):.0f} TFLOPS)  [sum={meta_ms + req_ms + qnt_ms + gemm_ms:.3f}]")
                    print(f"  [acc] dW2 fp8 finite={not bool(nan >= 1.0)}  "
                          f"(fp8-vs-bf16 SNR -> e2e gradcheck; bf16 ref -> bench_mega_moe_bf16.py)")
                torch.cuda.synchronize(); group.barrier()

            if args.stage == "fc1_wgrad":
                r = profile_fc1_wgrad(group, args, mode)
                nan = _amax(group, r["nan"])
                fp8_ms = _amax(group, r["fp8_ms"])
                gemm_ms, req_ms, qnt_ms = _amax(group, r["gemm_ms"]), _amax(group, r["req_ms"]), _amax(group, r["qnt_ms"])
                meta_ms = _amax(group, r["meta_ms"])
                if rank == 0:
                    tf = lambda ms: r["flops"] / (ms * 1e-3) / 1e12
                    print(f"\n{'='*80}\n[mega MoE bwd fc1 wgrad (dW1, variable-K, LOCAL)  fp8]  {hdr}\n{'='*80}")
                    print(f"  fp8  FULL : {fp8_ms:8.3f} ms | {tf(fp8_ms):8.1f} TFLOPS  (meta+quant+requant+GEMM; M_pool={r['m_pad']})")
                    print(f"  breakdown: meta={meta_ms:.3f}  quant(grad_l1)={qnt_ms:.3f}  requant(pool_x)={req_ms:.3f}  "
                          f"GEMM={gemm_ms:.3f} ms ({tf(gemm_ms):.0f} TFLOPS)  [sum={meta_ms + qnt_ms + req_ms + gemm_ms:.3f}]")
                    print(f"  [acc] dW1 fp8 finite={not bool(nan >= 1.0)}  (fp8-vs-bf16 SNR -> e2e gradcheck; "
                          f"bf16 ref -> bench_mega_moe_bf16.py). fp8 dW1 is LOCAL (reuses fwd pool).")
                torch.cuda.synchronize(); group.barrier()

            if args.stage == "fc1_dgrad_combine":
                r = profile_fc1_dgrad_combine(group, args, mode)
                fin = _amin(group, r["fin"])
                fp8_ms = _amax(group, r["fp8_ms"])
                if rank == 0:
                    tf = lambda ms: r["flops"] / (ms * 1e-3) / 1e12
                    print(f"\n{'='*80}\n[mega MoE bwd fc1 dgrad+combine (STEP3)  fp8]  {hdr}\n{'='*80}")
                    print(f"  fp8  : {fp8_ms:8.3f} ms | {tf(fp8_ms):8.1f} TFLOPS  (fc1-dgrad GEMM + fp8-PUSH combine + reduce, kernel-only; M_pool={r['m_pad']})")
                    print(f"  [acc] fp8 dx finite={bool(fin >= 1.0)} (norm={r['dx_norm']:.3e})  "
                          f"(rigorous dx SNR -> e2e gradcheck; bf16 ref -> bench_mega_moe_bf16.py)")
                    print(f"  note: epoch self-reset removed the old large-T STEP3 deadlock; use load_balanced "
                          f"(round_robin is a pathological all-to-few case)")
                torch.cuda.synchronize(); group.barrier()
    finally:
        dist.destroy_process_group()


def _build_parser():
    ap = argparse.ArgumentParser(
        description="mega MoE MXFP8 per-stage bench (fp8-only; bf16 reference -> bench_mega_moe_bf16.py)")
    ap.add_argument("--stage",
                    choices=["l1", "l2", "fwd", "dispatch_fc2_dgrad", "fc2_wgrad", "fc1_wgrad",
                             "fc1_dgrad_combine", "both"],
                    default="both",
                    help="which stage to bench (dispatch_fc2_dgrad / fc2_wgrad / fc1_wgrad / "
                         "fc1_dgrad_combine are backward; not in 'both')")
    ap.add_argument("--mode", choices=["load_balanced", "round_robin", "both"], default="both",
                    help="routing distribution(s) to bench")
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
