###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Benchmark the fused BF16 grouped GEMM + combine mega kernel (EP, intra-node).

Variants on the same prologue-generated data (so it matches
tests/pytorch/ops/test_mega_moe_dispatch_combine_grouped_gemm.py exactly):

  * gemm_only    -- grouped L2 GEMM over the SwiGLU activation (compute peak)  [TFLOPS]
  * combine_only -- cross-rank combine PUSH only                        [XGMI GB/s]
  * fused        -- 3-role grouped L2 GEMM + combine PUSH + weighted topk reduce -> y;
                    this IS e2e forward step 4 (mega_moe_fused).               [TFLOPS]

Backward (always profiled) reproduces ``mega_moe_fused.backward`` STEP 3 (NN): L1 dgrad
``grad_l1[M,2I] @ w1[g,2I,H] -> grad_pool[M,H]`` + combine PUSH + dx reduce (3-role, the
reduce unweighted -- the routing weight rides ``grad_l1``). Same metric set as the forward.

The inputs (act / weight / tile_to_group / origin_rank / origin_slot / combine
buffers) are built over the production ``SymmBuffer`` (``get_symm_buffer_for_mega_moe``)
by running ``mega_moe_prologue_impl`` + ``dispatch_grouped_gemm_impl`` + SwiGLU --
the exact same path as the EP test.

Run inside dev_primus (8 GPUs):
  PYTHONPATH=<...>/Primus-Turbo python benchmark/ops/bench_grouped_gemm_comine.py \
      --num-processes 8 [--mode load_balanced|round_robin|both]
"""

import argparse
import math
import os
import sys
from datetime import datetime
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from config import get_platform_info
from tabulate import tabulate

# repo root (primus_turbo) + the test module (single source of data generation)
_HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..")))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..", "tests", "pytorch", "ops")))

from test_mega_moe_dispatch_combine_grouped_gemm import (  # noqa: E402
    _global_weights,
    generate_routing,
)

from primus_turbo.flydsl.mega.dispatch_grouped_gemm_bf16_kernel import (  # noqa: E402
    grouped_gemm_bf16_only,
)
from primus_turbo.flydsl.mega.gemm_bf16_kernel import (  # noqa: E402
    _compile_dense_nt,
    _get_compiled_dense,
)
from primus_turbo.flydsl.mega.grouped_gemm_combine_bf16_kernel import (  # noqa: E402
    combine_only,
    grouped_gemm_combine_bf16,
)
from primus_turbo.flydsl.mega.mega_moe_epilogue import swiglu  # noqa: E402
from primus_turbo.pytorch.core.backend import BackendType  # noqa: E402
from primus_turbo.pytorch.core.symm_mem import SymmetricMemory  # noqa: E402
from primus_turbo.pytorch.kernels.mega_moe.dispatch_grouped_gemm_impl import (  # noqa: E402
    dispatch_grouped_gemm_impl,
)
from primus_turbo.pytorch.kernels.mega_moe.mega_moe_prologue_impl import (  # noqa: E402
    mega_moe_prologue_impl,
)

# single-allocation symmetric buffer shared with the production forward + the EP test
from primus_turbo.pytorch.ops.moe.mega_moe_fused import (  # noqa: E402
    get_symm_buffer_for_mega_moe,
)

_FLYDSL = BackendType.FLYDSL.value


class _SymmSig:
    """Symmetric SIGNAL buffer in UNCACHED memory (per-slot flags for the 3-role reduce).

    ``.local`` = this rank's uncached signal-pad view; ``.ptrs`` = per-rank signal-pad
    base ptrs (for cross-rank raises). Uncached so relaxed cross-rank reads stay fresh."""

    def __init__(self, group, shape, dtype):
        nbytes = math.prod(shape) * dtype.itemsize
        self.sm = SymmetricMemory(group, alloc_size=16, signal_pad_size=max(1024, nbytes))
        self.rank = group.rank()
        self.local = self.sm.get_signal_pad(self.rank, shape, dtype)
        self.ptrs = self.sm.signal_pad_ptrs_dev


def _dense_gemm_peak_ms(M, N, K, BM, BN, iters, *, group_m_cands=(4,)):
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
        ms = _bench(lambda: compiled(*dense_args), iters=iters)
        if ms < best_ms:
            best_ms, best_group_m = ms, group_m
    del a, b, c
    return best_ms, best_group_m


# --------------------------------------------------------------------------- #
# Bench helper: warmup, one L2 flush before the sync, then CUDA-event timing.
# Cross-rank variants barrier each iter (and reset the local scoreboard before it).
# --------------------------------------------------------------------------- #
_L2_FLUSH_BUF = None


def _l2_flush():
    global _L2_FLUSH_BUF
    if _L2_FLUSH_BUF is None:
        _L2_FLUSH_BUF = torch.empty(256 * 1024 * 1024 // 4, dtype=torch.int32, device="cuda")
    _L2_FLUSH_BUF.zero_()


def _bench(fn, *, reset=None, group=None, warmup=5, iters=30):
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


# --------------------------------------------------------------------------- #
# Input builder (consistent with the test): get the symmetric buffer, run the
# prologue + dispatch + SwiGLU so ``act`` is the real L2 input, then fill l2y once.
# --------------------------------------------------------------------------- #
def build_inputs(group, *, T, H, I, E, K, BM, BN, mode, W1, W2, base_seed=7):
    """Build the real L2-GEMM input by running the production prologue + dispatch +
    SwiGLU over the cached symmetric buffer (same path as the EP test), then fill
    ``l2_token_buffer`` once so combine_only has real rows."""
    rank = group.rank()
    torch.manual_seed(base_seed + rank)
    x = torch.randn((T, H), device="cuda", dtype=torch.float32).bfloat16()
    topk_idx, topk_w = generate_routing(T, K, E, mode, device="cuda", seed=100 + rank)

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

    # 1) prologue: build the cross-rank dispatch plan; returns the plan tables.
    # scoreboard + barrier_local are reset in-kernel by the prologue; sb_l2 + comb stay host-zeroed.
    plan, tile_to_group, expected = mega_moe_prologue_impl(
        topk_idx,
        topk_w,
        symm.buffer_base,
        symm.buffer_offsets,
        symm.origin_rank,
        symm.origin_slot,
        symm.meta_scalars,
        symm.grid_barrier_state,
        symm.profile,
        symm.scoreboard,
        symm.barrier_local,
        T,
        K,
        E,
        symm.world,
        symm.rank,
        symm.block_m,
        symm.pool_capacity,
        _FLYDSL,
        no_cpu_sync=True,
    )
    num_tile_blocks = symm.meta_scalars[1:2]  # device real-tile count
    symm.assert_capacity()  # fail loudly rather than silently drop rows

    # sb_l2 is a same-rank scoreboard accumulator -> zero it (no barrier needed). comb is
    # fully overwritten by the combine PUSH with no dropped tokens, so it needs no host zero/fence.
    symm.sb_l2.zero_()

    # 2) cross-rank dispatch PUSH + grouped L1 GEMM (NT): pool[M,H] @ W1[g,2I,H] -> l1_out
    l1_out = dispatch_grouped_gemm_impl(
        x,
        plan,
        symm.pool,
        symm.pool_ptrs,
        W1,
        tile_to_group,
        symm.scoreboard,
        symm.scoreboard_ptrs,
        expected,
        num_tile_blocks,
        symm.sb_consume,
        E,
        _FLYDSL,
        layout="nt",
        BM=BM,
        BN=BN,
    )

    # 3) fused SwiGLU activation -> act (L2 GEMM input)
    act = swiglu(
        l1_out,
        symm.act,
        symm.intermediate_hidden,
        symm.pool_capacity,
        num_tile_blocks=num_tile_blocks,
        BM=BM,
    )

    # 4) fill l2_token_buffer once with the real rows (for combine_only)
    grouped_gemm_bf16_only(act, W2, symm.l2_token_buffer, tile_to_group, num_tile_blocks, BM=BM, BN=BN)
    torch.cuda.synchronize()
    group.barrier()
    return SimpleNamespace(
        symm=symm,
        act=act,
        num_tile_blocks=num_tile_blocks,
        topk_idx=topk_idx,
        topk_w=topk_w,
        W1=W1,
        W2=W2,
        tile_to_group=tile_to_group,
    )


def profile_combine(group, args, mode, W1, W2):
    """Forward = e2e step 4 (NT): grouped L2 GEMM + combine PUSH + weighted topk reduce -> y.
    Backward = e2e step 3 (NN, always profiled): grad_l1 @ w1 (L1 dgrad) + combine PUSH +
    dx reduce. Both report the 3-role fused kernel (``grouped_gemm_combine_bf16`` with a
    reduce role), so ``fused`` == the actual e2e combine stage. The fused reduce is PERF ONLY:
    the per-slot ready flags are pre-zeroed each iter (no real cross-rank wait)."""
    rank = group.rank()
    BM, BN, H, I, num_combine_cu = args.bm, args.bn, args.hidden, args.inter, args.num_combine_cu
    inp = build_inputs(
        group,
        T=args.num_tokens,
        H=H,
        I=I,
        E=args.num_experts,
        K=args.num_topk,
        BM=BM,
        BN=BN,
        mode=mode,
        W1=W1,
        W2=W2,
    )
    symm, act, num_tile_blocks = inp.symm, inp.act, inp.num_tile_blocks
    tile_to_group = inp.tile_to_group  # prologue-returned expert-id-per-tile map
    flag_sm = None  # 3-role barrier symm buffer (freed in finally)
    try:
        real_tiles = int(symm.meta_scalars[1].item())
        M_eff = real_tiles * BM
        comb_addrs = symm.comb_addrs
        slots = symm.combine_slots
        group_m_cands = (1, 4, 8) if args.autotune else (args.dense_group_m,)
        # combine CUs: autotune sweeps {16,32,48,64,96}; default off uses --num-combine-cu (best).
        cu_cands = (16, 32, 48, 64, 96) if args.autotune else (num_combine_cu,)

        # XGMI combine bytes per rank = remote rows (origin_rank != rank, valid) x H x 2
        origin = symm.origin_rank
        remote_rows = int(((origin != rank) & (origin >= 0)).sum().item())
        xgmi_bytes = remote_rows * H * 2

        # per-slot ready flags in the UNCACHED signal pad (shared by fwd + bwd reduce).
        # 0 == ready (role 2 waits while flag < 0); pre-zeroed each iter -> no real cross-rank
        # wait. topk tables drive the per-token reduce.
        flag_sm = _SymmSig(group, (slots,), torch.int32)
        topk_idx_flat = inp.topk_idx.to(torch.int32).contiguous().view(-1)
        topk_w_flat = inp.topk_w.to(torch.float32).contiguous().view(-1)
        tokens_per_rank = torch.full((group.size(),), args.num_tokens, dtype=torch.int32, device="cuda")

        def _reset_fused():
            symm.sb_l2.zero_()
            flag_sm.local.zero_()

        # ---- forward (e2e step 4, NT): L2 GEMM N=H, K=I ; reduce -> y[T,H] (weighted) ----
        N, K = H, I
        flops = 2.0 * M_eff * N * K
        y_out = torch.empty(args.num_tokens, H, dtype=torch.bfloat16, device="cuda")

        # L2 has small K=I -> short contraction; GROUP_M=8 reuses the per-expert weight
        # across more M-tiles than the GROUP_M=4 L1 default (best for this shape).
        t_gemm = _bench(
            lambda: grouped_gemm_bf16_only(
                act, inp.W2, symm.l2_token_buffer, tile_to_group, num_tile_blocks, BM=BM, BN=BN, GROUP_M=8
            ),
            iters=args.iters,
        )
        # dense single-weight GEMM of the same M_eff x N x K = the grouped-GEMM roofline
        t_dense, dense_group_m = _dense_gemm_peak_ms(
            M_eff, N, K, BM, BN, args.iters, group_m_cands=group_m_cands
        )
        # combine_only and the fused kernel share the same CU candidates.
        comb_sweep = {}
        for cu in cu_cands:
            comb_sweep[cu] = _bench(
                lambda cu=cu: combine_only(
                    symm.l2_token_buffer,
                    symm.origin_rank,
                    symm.origin_slot,
                    comb_addrs,
                    slots,
                    num_tile_blocks,
                    BM=BM,
                    num_combine_cu=cu,
                ),
                group=group,
                iters=args.iters,
            )
        best_comb_cu = min(comb_sweep, key=comb_sweep.get)
        t_comb = comb_sweep[best_comb_cu]

        # fused = 3-role (GEMM + combine PUSH + weighted topk reduce -> y) == e2e forward step 4
        fused_sweep = {}
        for cu in cu_cands:
            fused_sweep[cu] = _bench(
                lambda cu=cu: grouped_gemm_combine_bf16(
                    act,
                    inp.W2,
                    symm.l2_token_buffer,
                    tile_to_group,
                    symm.sb_l2,
                    symm.origin_rank,
                    symm.origin_slot,
                    slots,
                    num_tile_blocks,
                    comb_addrs=comb_addrs,
                    comb_local=symm.comb,
                    output=y_out,
                    barrier_local=flag_sm.local,
                    barrier_addrs=flag_sm.ptrs,
                    topk_indices=topk_idx_flat,
                    num_tokens_per_rank=tokens_per_rank,
                    topk_weights=topk_w_flat,
                    topk=args.num_topk,
                    num_experts=args.num_experts,
                    rank=rank,
                    BM=BM,
                    BN=BN,
                    num_combine_cu=cu,
                    num_reduce_cu=args.num_reduce_cu,
                ),
                reset=_reset_fused,
                group=group,
                iters=args.iters,
            )
        best_fused_cu = min(fused_sweep, key=fused_sweep.get)
        t_fused = fused_sweep[best_fused_cu]

        # ---- backward (e2e step 3, NN): L1 dgrad grad_l1[M,2I] @ w1[g,2I,H] -> grad_pool[M,H],
        #      combine PUSH (H-wide) + dx reduce (unweighted -> dx[T,H]) == mega_moe_fused STEP 3.
        bwd_N, bwd_K = H, 2 * I  # weight [G, K=2I, N=H]
        bwd_flops = 2.0 * M_eff * bwd_N * bwd_K
        bwd_xgmi = remote_rows * H * 2  # H-wide push, equals the forward
        grad_l1 = torch.randn(symm.pool_capacity, bwd_K, device="cuda", dtype=torch.bfloat16) / 8
        dx_out = torch.empty(args.num_tokens, H, dtype=torch.bfloat16, device="cuda")

        t_bwd_gemm = _bench(
            lambda: grouped_gemm_bf16_only(
                grad_l1,
                inp.W1,
                symm.l2_token_buffer,
                tile_to_group,
                num_tile_blocks,
                layout="nn",
                BM=BM,
                BN=BN,
                GROUP_M=8,
            ),
            iters=args.iters,
        )
        t_bwd_dense, bwd_dense_gm = _dense_gemm_peak_ms(
            M_eff, bwd_N, bwd_K, BM, BN, args.iters, group_m_cands=group_m_cands
        )
        bwd_comb_sweep = {}
        for cu in cu_cands:
            bwd_comb_sweep[cu] = _bench(
                lambda cu=cu: combine_only(
                    symm.l2_token_buffer,
                    symm.origin_rank,
                    symm.origin_slot,
                    comb_addrs,
                    slots,
                    num_tile_blocks,
                    BM=BM,
                    num_combine_cu=cu,
                ),
                group=group,
                iters=args.iters,
            )
        best_bwd_comb_cu = min(bwd_comb_sweep, key=bwd_comb_sweep.get)
        # bwd fused = 3-role NN (GEMM + combine PUSH + unweighted dx reduce); weight rides grad_l1
        bwd_fused_sweep = {}
        for cu in cu_cands:
            bwd_fused_sweep[cu] = _bench(
                lambda cu=cu: grouped_gemm_combine_bf16(
                    grad_l1,
                    inp.W1,
                    symm.l2_token_buffer,
                    tile_to_group,
                    symm.sb_l2,
                    symm.origin_rank,
                    symm.origin_slot,
                    slots,
                    num_tile_blocks,
                    comb_addrs=comb_addrs,
                    comb_local=symm.comb,
                    output=dx_out,
                    barrier_local=flag_sm.local,
                    barrier_addrs=flag_sm.ptrs,
                    topk_indices=topk_idx_flat,
                    num_tokens_per_rank=tokens_per_rank,
                    topk_weights=None,
                    topk=args.num_topk,
                    num_experts=args.num_experts,
                    rank=rank,
                    layout="nn",
                    BM=BM,
                    BN=BN,
                    num_combine_cu=cu,
                    num_reduce_cu=args.num_reduce_cu,
                ),
                reset=_reset_fused,
                group=group,
                iters=args.iters,
            )
        best_bwd_cu = min(bwd_fused_sweep, key=bwd_fused_sweep.get)
        del grad_l1
    finally:
        if flag_sm is not None:
            flag_sm.sm.destroy()
        symm.destroy()  # always free the symmetric buffer (re-allocated per mode)
    # raw per-rank timings + work; rank 0 aggregates across ranks (bottleneck = max latency)
    return {
        "gemm_only_ms": t_gemm,
        "dense_gemm_only_ms": t_dense,
        "dense_gm": dense_group_m,
        "combine_only_ms": t_comb,
        "comb_sweep": comb_sweep,
        "comb_cu": best_comb_cu,
        "fused_ms": t_fused,
        "fused_sweep": fused_sweep,
        "fused_cu": best_fused_cu,
        "flops": flops,
        "xgmi_bytes": xgmi_bytes,
        "bwd": {
            "gemm_only_ms": t_bwd_gemm,
            "dense_gemm_only_ms": t_bwd_dense,
            "dense_gm": bwd_dense_gm,
            "combine_only_ms": bwd_comb_sweep[best_bwd_comb_cu],
            "comb_sweep": bwd_comb_sweep,
            "comb_cu": best_bwd_comb_cu,
            "fused_ms": bwd_fused_sweep[best_bwd_cu],
            "fused_sweep": bwd_fused_sweep,
            "fused_cu": best_bwd_cu,
            "flops": bwd_flops,
            "xgmi_bytes": bwd_xgmi,
        },
    }


def benchmark_combine(local_rank, world, args):
    ip = os.getenv("MASTER_ADDR", "127.0.0.1")
    port = int(os.getenv("MASTER_PORT", "8483"))
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", init_method=f"tcp://{ip}:{port}", world_size=world, rank=local_rank)
    torch.set_default_device("cuda")
    torch.cuda.set_device(local_rank)
    group = dist.new_group(list(range(world)))
    rank = dist.get_rank()
    platform, gpu_name = get_platform_info()

    # this rank's expert weights, built once (sliced from the deterministic global set)
    experts_per_rank = args.num_experts // world
    W1_global, W2_global = _global_weights(args.num_experts, args.inter, args.hidden, "cuda")
    W1 = W1_global[rank * experts_per_rank : (rank + 1) * experts_per_rank].contiguous()
    W2 = W2_global[rank * experts_per_rank : (rank + 1) * experts_per_rank].contiguous()
    del W1_global, W2_global

    modes = ["load_balanced", "round_robin"] if args.mode == "both" else [args.mode]
    rows = []
    try:
        for mode in modes:
            result = profile_combine(group, args, mode, W1, W2)
            gathered = [None] * world
            dist.all_gather_object(gathered, (rank, result), group=group)
            if rank != 0:
                torch.cuda.synchronize()
                group.barrier()
                continue

            per_rank = [g[1] for g in gathered]
            # distributed bottleneck = slowest rank (max latency); work ~ uniform (mean)
            rank_max = lambda key: max(d[key] for d in per_rank)
            rank_mean = lambda key: sum(d[key] for d in per_rank) / len(per_rank)

            def _sweep_str(key):  # "cu->ms" for each swept config (max over ranks)
                cus = sorted(per_rank[0][key].keys())
                return " ".join(f"{cu}->{max(d[key][cu] for d in per_rank):.3f}" for cu in cus)

            gemm_ms, comb_ms, fused_ms = (
                rank_max("gemm_only_ms"),
                rank_max("combine_only_ms"),
                rank_max("fused_ms"),
            )
            dense_ms = rank_max("dense_gemm_only_ms")
            flops, xgmi = rank_mean("flops"), rank_mean("xgmi_bytes")
            gemm_tf = flops / (gemm_ms * 1e-3) / 1e12
            dense_tf = flops / (dense_ms * 1e-3) / 1e12
            fused_tf = flops / (fused_ms * 1e-3) / 1e12
            comb_bw = xgmi / (comb_ms * 1e-3) / 1e9
            # grouped GEMM vs the dense single-weight roofline (is grouped at peak?)
            grouped_eff_pct = gemm_tf / dense_tf * 100.0
            # fused vs running gemm then comm serially, and vs the overlap floor:
            #   hidden  = serial(gemm+comm) - fused      (comm time hidden under GEMM)
            #   speedup = serial / fused                 (fused vs serial, e.g. 1.2x)
            #   roofline= max(gemm,comb) / fused         (overlap floor: the slower leg)
            serial_ms = gemm_ms + comb_ms
            hidden_ms = serial_ms - fused_ms
            speedup = serial_ms / fused_ms
            roofline_pct = max(gemm_ms, comb_ms) / fused_ms * 100.0
            print(
                f"\n{'='*72}\n[combine] {gpu_name} EP{world} T={args.num_tokens} H={args.hidden} "
                f"I={args.inter} E={args.num_experts} K={args.num_topk} mode={mode} "
                f"(max over ranks)\n{'='*72}"
            )
            print(
                f"  dense_gemm   : {dense_ms:8.3f} ms | {dense_tf:7.1f} TFLOPS (single-weight roofline, "
                f"GROUP_M={per_rank[0]['dense_gm']})"
            )
            print(
                f"  gemm_only    : {gemm_ms:8.3f} ms | {gemm_tf:7.1f} TFLOPS | "
                f"grouped/dense = {grouped_eff_pct:.1f}%"
            )
            print(
                f"  combine_only : {comb_ms:8.3f} ms | {comb_bw:7.1f} GB/s (XGMI) | "
                f"combine_cu: {_sweep_str('comb_sweep')} ms (best={per_rank[0]['comb_cu']})"
            )
            # fused now = the 3-role kernel (GEMM + combine PUSH + weighted topk reduce -> y),
            # i.e. the actual e2e forward step 4 (PERF ONLY: reduce flags pre-set ready).
            print(
                f"  fused : {fused_ms:8.3f} ms | {fused_tf:7.1f} TFLOPS | "
                f"hid {hidden_ms:.3f}/{comb_ms:.3f} ms comm | speedup vs serial = {speedup:.2f}x | "
                f"roofline (max(gemm,comb)/fused) = {roofline_pct:.1f}% | "
                f"num_reduce_cu={args.num_reduce_cu} | "
                f"combine_cu: {_sweep_str('fused_sweep')} ms (best={per_rank[0]['fused_cu']})"
            )

            # backward STEP3 (NN, = mega_moe_fused STEP 3): L1 dgrad GEMM + combine + dx reduce.
            # Same metric set as the forward; reduce is unweighted (weight rides grad_l1).
            bwd_max = lambda key: max(d["bwd"][key] for d in per_rank)
            bwd_mean = lambda key: sum(d["bwd"][key] for d in per_rank) / len(per_rank)

            def _bwd_sweep_str(key):
                cus = sorted(per_rank[0]["bwd"][key].keys())
                return " ".join(f"{cu}->{max(d['bwd'][key][cu] for d in per_rank):.3f}" for cu in cus)

            bwd_gemm_ms = bwd_max("gemm_only_ms")
            bwd_dense_ms = bwd_max("dense_gemm_only_ms")
            bwd_comb_ms = bwd_max("combine_only_ms")
            bwd_fused_ms = bwd_max("fused_ms")
            bwd_flops, bwd_xgmi = bwd_mean("flops"), bwd_mean("xgmi_bytes")
            bwd_gemm_tf = bwd_flops / (bwd_gemm_ms * 1e-3) / 1e12
            bwd_dense_tf = bwd_flops / (bwd_dense_ms * 1e-3) / 1e12
            bwd_fused_tf = bwd_flops / (bwd_fused_ms * 1e-3) / 1e12
            bwd_comb_bw = bwd_xgmi / (bwd_comb_ms * 1e-3) / 1e9
            bwd_grouped_eff = bwd_gemm_tf / bwd_dense_tf * 100.0
            bwd_serial_ms = bwd_gemm_ms + bwd_comb_ms
            bwd_hidden_ms = bwd_serial_ms - bwd_fused_ms
            bwd_speedup = bwd_serial_ms / bwd_fused_ms
            bwd_roofline = max(bwd_gemm_ms, bwd_comb_ms) / bwd_fused_ms * 100.0
            print(f"  {'-'*68}  backward STEP3 (NN, = mega_moe_fused STEP 3)")
            print(
                f"  dense_gemm   : {bwd_dense_ms:8.3f} ms | {bwd_dense_tf:7.1f} TFLOPS (single-weight roofline, "
                f"GROUP_M={per_rank[0]['bwd']['dense_gm']})"
            )
            print(
                f"  gemm_only    : {bwd_gemm_ms:8.3f} ms | {bwd_gemm_tf:7.1f} TFLOPS | "
                f"grouped/dense = {bwd_grouped_eff:.1f}%"
            )
            print(
                f"  combine_only : {bwd_comb_ms:8.3f} ms | {bwd_comb_bw:7.1f} GB/s (XGMI) | "
                f"combine_cu: {_bwd_sweep_str('comb_sweep')} ms (best={per_rank[0]['bwd']['comb_cu']})"
            )
            print(
                f"  fused : {bwd_fused_ms:8.3f} ms | {bwd_fused_tf:7.1f} TFLOPS | "
                f"hid {bwd_hidden_ms:.3f}/{bwd_comb_ms:.3f} ms comm | speedup vs serial = {bwd_speedup:.2f}x | "
                f"roofline (max(gemm,comb)/fused) = {bwd_roofline:.1f}% | "
                f"num_reduce_cu={args.num_reduce_cu} | "
                f"combine_cu: {_bwd_sweep_str('fused_sweep')} ms (best={per_rank[0]['bwd']['fused_cu']})"
            )

            rows.append(
                {
                    "Platform": platform,
                    "GPU": gpu_name,
                    "EP": world,
                    "Mode": mode,
                    "T": args.num_tokens,
                    "H": args.hidden,
                    "I": args.inter,
                    "E": args.num_experts,
                    "K": args.num_topk,
                    "dense_gemm (ms)": f"{dense_ms:.3f}",
                    "dense_gemm (TFLOPS)": f"{dense_tf:.1f}",
                    "gemm_only (ms)": f"{gemm_ms:.3f}",
                    "gemm_only (TFLOPS)": f"{gemm_tf:.1f}",
                    "grouped/dense": f"{grouped_eff_pct:.1f}%",
                    "combine_only (ms)": f"{comb_ms:.3f}",
                    "combine_only (XGMI GB/s)": f"{comb_bw:.1f}",
                    "fused (ms)": f"{fused_ms:.3f}",
                    "fused (TFLOPS)": f"{fused_tf:.1f}",
                    "comm_hidden (ms)": f"{hidden_ms:.3f}",
                    "speedup (vs serial)": f"{speedup:.2f}x",
                    "roofline (max(gemm,comb)/fused)": f"{roofline_pct:.1f}%",
                    "bwd dense_gemm (TFLOPS)": f"{bwd_dense_tf:.1f}",
                    "bwd gemm_only (ms)": f"{bwd_gemm_ms:.3f}",
                    "bwd gemm_only (TFLOPS)": f"{bwd_gemm_tf:.1f}",
                    "bwd grouped/dense": f"{bwd_grouped_eff:.1f}%",
                    "bwd combine_only (ms)": f"{bwd_comb_ms:.3f}",
                    "bwd combine_only (XGMI GB/s)": f"{bwd_comb_bw:.1f}",
                    "bwd fused (ms)": f"{bwd_fused_ms:.3f}",
                    "bwd fused (TFLOPS)": f"{bwd_fused_tf:.1f}",
                    "bwd speedup (vs serial)": f"{bwd_speedup:.2f}x",
                    "bwd roofline (max(gemm,comb)/fused)": f"{bwd_roofline:.1f}%",
                }
            )
            torch.cuda.synchronize()
            group.barrier()

        if rank == 0 and rows:
            results = pd.DataFrame(rows)
            print("\nFinal Results:")
            print(tabulate(results, headers="keys", tablefmt="grid", showindex=False))
            out_file = args.output or f"grouped_gemm_combine_{datetime.now():%Y%m%d}_{gpu_name}.csv"
            results.to_csv(out_file, index=False)
            print(f"Results saved to {out_file}")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Benchmark fused BF16 grouped GEMM + combine")
    ap.add_argument("--num-processes", type=int, default=8)
    ap.add_argument("--hidden", type=int, default=7168)  # DeepSeek-V3
    ap.add_argument("--inter", type=int, default=2048)
    ap.add_argument("--num-experts", type=int, default=256)
    ap.add_argument("--num-topk", type=int, default=8)
    ap.add_argument("--num-tokens", type=int, default=8192)
    ap.add_argument("--bm", type=int, default=256)
    ap.add_argument("--bn", type=int, default=256)
    # 64 combine CUs is the near-optimum for the fused path in both routing modes
    # (autotune sweep {16,32,48,64,96}); default off uses this single best value.
    ap.add_argument("--num-combine-cu", type=int, default=64)
    # GROUP_M for the dense roofline reference; default = best, autotune sweeps {1,4,8}.
    ap.add_argument("--dense-group-m", type=int, default=4)
    ap.add_argument(
        "--autotune",
        action="store_true",
        help="sweep combine CUs {16,32,48,64,96} + dense GROUP_M {1,4,8}; "
        "default off uses --num-combine-cu / --dense-group-m (best)",
    )
    ap.add_argument(
        "--num-reduce-cu",
        type=int,
        default=32,
        help="role-2 topk-reduce dedicated blocks for the fused 3-role kernel",
    )
    ap.add_argument("--mode", choices=["load_balanced", "round_robin", "both"], default="both")
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--output", "-o", type=str, default=None)
    args = ap.parse_args()
    torch.multiprocessing.spawn(benchmark_combine, args=(args.num_processes, args), nprocs=args.num_processes)
