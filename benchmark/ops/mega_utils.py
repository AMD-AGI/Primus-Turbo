###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Shared helpers for the mega-MoE EP benchmarks (dispatch+GEMM / GEMM+combine).

Single source of truth for: routing/weight generation, the dense-GEMM roofline,
the CUDA-event bench helper, the prologue-driven input builders for both kernels,
and the per-stage metric/print template so both benchmarks print identically.
"""

import math
from types import SimpleNamespace

import numpy as np
import torch

from primus_turbo.flydsl.mega.dispatch_grouped_gemm_bf16_kernel import (
    dispatch_only,
    dispatch_prologue,
    grouped_gemm_bf16_only,
)
from primus_turbo.flydsl.mega.gemm_bf16_kernel import (
    _compile_dense_nt,
    _get_compiled_dense,
)
from primus_turbo.flydsl.mega.swiglu_kernel import swiglu
from primus_turbo.flydsl.mega.symm_buffer import get_symm_buffer_for_mega_moe

# re-export so benchmarks can import everything mega from one place
__all__ = [
    "generate_routing",
    "global_weights",
    "dense_gemm_peak_ms",
    "bench",
    "generate_input",
    "compute_stage_metrics",
    "print_header",
    "print_stage",
    "dispatch_only",
    "dispatch_prologue",
    "grouped_gemm_bf16_only",
    "get_symm_buffer_for_mega_moe",
    "swiglu",
]


# --------------------------------------------------------------------------- #
# Routing + weights
# --------------------------------------------------------------------------- #
def generate_routing(num_tokens, num_topk, num_experts, mode, *, device="cuda", seed=0):
    """(topk_idx[T,K] int64, topk_weight[T,K] f32) for load_balanced / round_robin modes."""
    g = torch.Generator(device=device).manual_seed(seed)
    if mode == "load_balanced":
        scores = torch.rand(num_tokens, num_experts, generator=g, device=device).abs() + 1
        topk_weight, topk_idx = torch.topk(scores.softmax(-1), num_topk, dim=-1)
        topk_idx = topk_idx.to(torch.int64)
        topk_weight = topk_weight.to(torch.float32)
    elif mode == "round_robin":
        topk_idx = (
            torch.arange(num_tokens * num_topk, device=device).view(num_tokens, num_topk) % num_experts
        ).to(torch.int64)
        topk_weight = (
            torch.rand(num_tokens, num_topk, generator=g, device=device).softmax(-1).to(torch.float32)
        )
    else:
        raise ValueError(f"unknown routing mode: {mode}")
    return topk_idx, topk_weight


def global_weights(E, I, H, device):
    """Deterministic global expert weights (identical on every rank, then sliced)."""
    g = torch.Generator(device=device).manual_seed(1234)
    W1 = torch.randn((E, 2 * I, H), generator=g, device=device, dtype=torch.bfloat16) * (2.0 / math.sqrt(H))
    W2 = torch.randn((E, H, I), generator=g, device=device, dtype=torch.bfloat16) * (2.0 / math.sqrt(I))
    return W1, W2


# --------------------------------------------------------------------------- #
# Bench helper: warmup, one L2 flush before timing, then CUDA-event timing.
# Cross-rank variants (group set) barrier around an optional scoreboard reset.
# --------------------------------------------------------------------------- #
_L2_FLUSH_BUF = None


def _l2_flush():
    global _L2_FLUSH_BUF
    if _L2_FLUSH_BUF is None:
        _L2_FLUSH_BUF = torch.empty(256 * 1024 * 1024 // 4, dtype=torch.int32, device="cuda")
    _L2_FLUSH_BUF.zero_()


def bench(fn, *, warmup=20, iters=30, reset=None, group=None):
    """Mean ms/call (CUDA events).

    Cross-rank discipline (group set): barrier BEFORE reset so every rank has
    finished the previous fn (no in-flight peer signals), zero, then a second
    barrier so all state is clean before any rank's fn pushes/signals a peer.
    Skipping the pre-reset barrier races the scoreboard and can deadlock."""
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]

    def _iter(start_event=None, end_event=None):
        if group is not None:
            torch.cuda.synchronize()
            group.barrier()  # all ranks done with previous fn
        if reset is not None:
            reset()
        if group is not None:
            torch.cuda.synchronize()
            group.barrier()  # all state clean before any fn
        if start_event is None:
            fn()
            return
        start_event.record()
        fn()
        end_event.record()

    for _ in range(warmup):
        _iter()
    _l2_flush()
    for i in range(iters):
        _iter(start_events[i], end_events[i])
    torch.cuda.synchronize()

    times = np.array([s.elapsed_time(e) for s, e in zip(start_events, end_events)])[1:]
    return np.average(times)


def dense_gemm_peak_ms(M, N, K, BM, BN, iters, *, group_m_cands=(4,)):
    """Dense NT bf16 GEMM (gemm_bf16_kernel) of the SAME M x N x K as the grouped
    GEMM -> the single-weight compute roofline. autotune sweeps GROUP_M {1,4,8};
    default off uses the single best (GROUP_M=4). Returns (best_ms, best_group_m)."""
    a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16) / 8
    b = torch.randn(N, K, device="cuda", dtype=torch.bfloat16) / 8  # NT: B [N,K]
    c = torch.empty(M, N, device="cuda", dtype=torch.bfloat16)
    dense_args = (a.view(-1), b.view(-1), c.view(-1), M, N, torch.cuda.current_stream())
    best_ms, best_group_m = float("inf"), None
    for group_m in group_m_cands:
        launch = _compile_dense_nt(K=K, BLOCK_M=BM, BLOCK_N=BN, GROUP_M=group_m, num_xcd=8)
        compiled = _get_compiled_dense(launch, dense_args)
        ms = bench(lambda: compiled(*dense_args), iters=iters)
        if ms < best_ms:
            best_ms, best_group_m = ms, group_m
    del a, b, c
    return best_ms, best_group_m


# --------------------------------------------------------------------------- #
# Input builders (consistent with the EP test): build the SymmBuffer, run the
# prologue to produce the dispatch handle, then fill the real pool/activation.
# `kind` selects which kernel's inputs to materialize.
# --------------------------------------------------------------------------- #
def _build_symm_and_plan(group, *, T, H, I, E, K, BM, BN, mode, base_seed):
    """Shared prologue: SymmBuffer + routing + dispatch handle (no pool fill yet)."""
    rank = group.rank()
    torch.manual_seed(base_seed + rank)
    x = torch.randn((T, H), device="cuda", dtype=torch.float32).bfloat16()
    topk_idx, topk_weight = generate_routing(T, K, E, mode, device="cuda", seed=100 + rank)

    # one symmetric allocation for every cross-rank + scratch buffer (production arena)
    symm = get_symm_buffer_for_mega_moe(
        group,
        num_experts=E,
        num_max_tokens_per_rank=T,
        num_topk=K,
        hidden=H,
        intermediate_hidden=I,
        block_m=BM,
        block_n=BN,
    )
    # prologue -> dispatch handle tables (same path as the test); resets scoreboard +
    # barrier_local in-kernel and ends with a cross-rank barrier.
    handle, tile_to_expert, expected, _orank, _oslot, _npb, _mnt = dispatch_prologue(
        topk_idx,
        topk_weight,
        sym_layout=symm.make_sym_layout(),
        num_tokens=T,
        num_topk=K,
        num_experts=E,
        world_size=symm.world,
        rank=symm.rank,
        experts_per_rank=E // symm.world,
        block_m=symm.block_m,
        pool_capacity=symm.pool_capacity,
        no_cpu_sync=True,
    )
    num_tile_blocks = symm.meta_scalars[1:2]  # device real-tile count
    symm.assert_capacity()  # fail loudly rather than silently drop rows
    return symm, x, topk_idx, topk_weight, handle, tile_to_expert, expected, num_tile_blocks


def generate_input(group, *, kind, mode, T, H, I, E, K, BM, BN, W1, W2, num_dispatch_cu=16, base_seed=7):
    """Build the real inputs for one of the mega kernels over the production SymmBuffer.

    kind="dispatch": pool is filled by a single dispatch PUSH (real A for the L1 GEMM).
    kind="combine":  prologue + dispatch + L1 GEMM + SwiGLU -> act (real L2 input), and
                     l2_token_buffer is filled once so combine_only has real rows."""
    group.rank()
    symm, x, topk_idx, topk_weight, handle, tile_to_expert, expected, num_tile_blocks = _build_symm_and_plan(
        group, T=T, H=H, I=I, E=E, K=K, BM=BM, BN=BN, mode=mode, base_seed=base_seed
    )

    if kind == "dispatch":
        # L1 GEMM output (2*inter wide) has no slot in the arena -> local scratch
        l1_out = torch.empty((symm.pool_capacity, 2 * I), dtype=torch.bfloat16, device="cuda")
        destination, _start, count, _src_offset, _src_tokens, _topk_slot, _weight = handle
        # fill the pool (real A) via dispatch_only (peers synced by the prologue)
        torch.cuda.synchronize()
        group.barrier()
        dispatch_only(x, handle, symm.pool, symm.pool_ptrs, num_dispatch_cu=num_dispatch_cu)
        torch.cuda.synchronize()
        group.barrier()
        return SimpleNamespace(
            symm=symm,
            x=x,
            handle=handle,
            l1_out=l1_out,
            tile_to_expert=tile_to_expert,
            expected=expected,
            num_tile_blocks=num_tile_blocks,
            destination=destination,
            count=count,
            topk_idx=topk_idx,
            topk_weight=topk_weight,
            W1=W1,
            W2=W2,
        )

    if kind == "combine":
        # sb_l2 is a same-rank scoreboard accumulator -> zero it (no barrier needed).
        symm.sb_l2.zero_()
        # cross-rank dispatch PUSH (fills the pool), then grouped L1 GEMM (NT):
        # pool[M,H] @ W1[g,2I,H] -> l1_out[M,2I] (local scratch, no arena slot)
        torch.cuda.synchronize()
        group.barrier()
        dispatch_only(x, handle, symm.pool, symm.pool_ptrs, num_dispatch_cu=32)
        torch.cuda.synchronize()
        group.barrier()
        l1_out = torch.empty((symm.pool_capacity, 2 * I), dtype=torch.bfloat16, device="cuda")
        grouped_gemm_bf16_only(symm.pool, W1, l1_out, tile_to_expert, num_tile_blocks, BM=BM, BN=BN)
        # fused SwiGLU activation -> act (L2 GEMM input)
        act = swiglu(l1_out)
        # fill l2_token_buffer once with the real rows (for combine_only)
        grouped_gemm_bf16_only(act, W2, symm.l2_token_buffer, tile_to_expert, num_tile_blocks, BM=BM, BN=BN)
        torch.cuda.synchronize()
        group.barrier()
        return SimpleNamespace(
            symm=symm,
            act=act,
            num_tile_blocks=num_tile_blocks,
            topk_idx=topk_idx,
            topk_weight=topk_weight,
            tile_to_expert=tile_to_expert,
            W1=W1,
            W2=W2,
        )

    raise ValueError(f"unknown kind: {kind}")


# --------------------------------------------------------------------------- #
# Metric + print template: identical layout for dispatch and combine, fwd/bwd.
# --------------------------------------------------------------------------- #
def compute_stage_metrics(*, gemm_ms, dense_ms, dense_gm, comm_ms, fused_ms, flops, xgmi):
    """Derive the standard per-stage metrics shared by both benchmarks.

    grouped/dense = grouped GEMM vs the dense single-weight roofline.
    hidden  = serial(gemm+comm) - fused      (comm time hidden under GEMM)
    speedup = serial / fused                 (fused vs serial)
    roofline= max(gemm,comm) / fused         (overlap floor: the slower leg)"""
    serial_ms = gemm_ms + comm_ms
    return SimpleNamespace(
        gemm_ms=gemm_ms,
        gemm_tf=flops / (gemm_ms * 1e-3) / 1e12,
        dense_ms=dense_ms,
        dense_tf=flops / (dense_ms * 1e-3) / 1e12,
        dense_gm=dense_gm,
        comm_ms=comm_ms,
        comm_bw=xgmi / (comm_ms * 1e-3) / 1e9,
        fused_ms=fused_ms,
        fused_tf=flops / (fused_ms * 1e-3) / 1e12,
        grouped_eff_pct=(flops / (gemm_ms * 1e-3)) / (flops / (dense_ms * 1e-3)) * 100.0,
        serial_ms=serial_ms,
        hidden_ms=serial_ms - fused_ms,
        speedup=serial_ms / fused_ms,
        roofline_pct=max(gemm_ms, comm_ms) / fused_ms * 100.0,
    )


def print_header(tag, gpu_name, world, args, mode):
    """Common '===' header line shared by both benchmarks."""
    print(
        f"\n{'='*72}\n[{tag}] {gpu_name} EP{world} T={args.num_tokens} H={args.hidden} "
        f"I={args.inter} E={args.num_experts} K={args.num_topk} mode={mode} "
        f"(max over ranks)\n{'='*72}"
    )


def print_stage(m, *, comm_label, comm_unit, comm_tag, comm_extra="", fused_extra="", sub_header=None):
    """Print one stage (forward or backward) in the shared 4-line layout.

    comm_label : 'dispatch_only' | 'combine_only'   (left column)
    comm_unit  : e.g. 'GB/s (XGMI, nodeup)' | 'GB/s (XGMI)'
    comm_tag   : 'disp' | 'comb'                     (roofline formula text)
    comm_extra / fused_extra : kernel-specific suffixes (e.g. CU sweep strings)."""
    if sub_header is not None:
        print(f"  {'-'*68}  {sub_header}")
    print(
        f"  dense_gemm   : {m.dense_ms:8.3f} ms | {m.dense_tf:7.1f} TFLOPS "
        f"(single-weight roofline, GROUP_M={m.dense_gm})"
    )
    print(
        f"  gemm_only    : {m.gemm_ms:8.3f} ms | {m.gemm_tf:7.1f} TFLOPS | "
        f"grouped/dense = {m.grouped_eff_pct:.1f}%"
    )
    print(f"  {comm_label:<13}: {m.comm_ms:8.3f} ms | {m.comm_bw:7.1f} {comm_unit}{comm_extra}")
    print(
        f"  fused        : {m.fused_ms:8.3f} ms | {m.fused_tf:7.1f} TFLOPS | "
        f"roofline (max(gemm,{comm_tag})/fused) = {m.roofline_pct:.1f}% | "
        f"speedup vs serial = {m.speedup:.2f}x{fused_extra}"
    )
