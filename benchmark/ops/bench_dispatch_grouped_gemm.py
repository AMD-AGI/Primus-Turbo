###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Benchmark the fused BF16 dispatch + grouped GEMM mega kernel (EP, intra-node).

Three variants on the same prologue-generated data (so it matches
tests/pytorch/ops/test_mega_moe_dispatch_combine_grouped_gemm.py exactly):

  * gemm_only     -- grouped L1 GEMM over the dispatched pool (compute peak)   [TFLOPS]
  * dispatch_only -- cross-rank dispatch PUSH only                       [XGMI GB/s]
  * fused         -- dispatch PUSH + grouped L1 GEMM (overlap)                [TFLOPS]

The test data (pool / dispatch plan / tile_to_group / expected / scoreboard) is
built by the production ``SymmBuffer`` + ``mega_moe_prologue_impl`` -- the exact
same path as the end-to-end test.

Run inside dev_primus (8 GPUs):
  PYTHONPATH=<...>/Primus-Turbo python benchmark/ops/bench_dispatch_grouped_gemm.py \
      --num-processes 8 [--mode load_balanced|round_robin|both]
"""

import argparse
import math
import os
import sys
from datetime import datetime
from types import SimpleNamespace

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
    dispatch_grouped_gemm_bf16,
    dispatch_only,
    grouped_gemm_bf16_only,
)
from primus_turbo.flydsl.mega.gemm_bf16_kernel import (  # noqa: E402
    _compile_dense_nt,
    _get_compiled_dense,
)
from primus_turbo.pytorch.core.backend import BackendType  # noqa: E402
from primus_turbo.pytorch.core.symm_mem import SymmetricMemory  # noqa: E402
from primus_turbo.pytorch.kernels.mega_moe.mega_moe_prologue_impl import (  # noqa: E402
    mega_moe_prologue_impl,
)

# single-allocation symmetric buffer shared with the production mega_fused_moe forward
from primus_turbo.pytorch.ops.moe.mega_fused_moe import (  # noqa: E402
    get_symm_buffer_for_mega_moe,
)

_FLYDSL = BackendType.FLYDSL.value


class _Symm:
    """One extra HIP-IPC symmetric buffer: local zero-copy view + peer pointer table.

    Used only for the optional backward dgrad pool (2*inter wide), which has no slot
    in the production ``SymmBuffer`` arena."""

    def __init__(self, group, shape, dtype):
        nbytes = math.prod(shape) * dtype.itemsize
        self.sm = SymmetricMemory(group, nbytes)
        self.rank = group.rank()
        self.local = self.sm.get_buffer(self.rank, shape, dtype)
        self.ptrs = torch.tensor(self.sm.buffer_ptrs, dtype=torch.int64, device="cuda")


def _dense_gemm_peak_ms(M, N, K, BM, BN, iters, *, group_m_cands=(4,)):
    """Dense NT bf16 GEMM (gemm_bf16_kernel) of the SAME M x N x K as the grouped
    GEMM -> the single-weight compute roofline. autotune sweeps GROUP_M {1,4,8};
    default off uses the single best (GROUP_M=4). Returns (best_ms, best_group_m)."""
    a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16) / 8
    b = torch.randn(N, K, device="cuda", dtype=torch.bfloat16) / 8  # NT: B [N,K]
    c = torch.empty(M, N, device="cuda", dtype=torch.bfloat16)
    dargs = (a.view(-1), b.view(-1), c.view(-1), M, N, torch.cuda.current_stream())
    best, best_gm = float("inf"), None
    for gm in group_m_cands:
        launch = _compile_dense_nt(K=K, BLOCK_M=BM, BLOCK_N=BN, GROUP_M=gm, num_xcd=8)
        compiled = _get_compiled_dense(launch, dargs)
        t = _bench(lambda: compiled(*dargs), iters=iters)
        if t < best:
            best, best_gm = t, gm
    del a, b, c
    return best, best_gm


# --------------------------------------------------------------------------- #
# Bench helper: warmup, one L2 flush before the sync, then CUDA-event timing.
# Cross-rank variants barrier each iter (and reset the scoreboard before it).
# --------------------------------------------------------------------------- #
_L2_FLUSH_BUF = None


def _l2_flush():
    global _L2_FLUSH_BUF
    if _L2_FLUSH_BUF is None:
        _L2_FLUSH_BUF = torch.empty(256 * 1024 * 1024 // 4, dtype=torch.int32, device="cuda")
    _L2_FLUSH_BUF.zero_()


def _bench(fn, *, reset=None, group=None, warmup=5, iters=30):
    """Mean ms/call (CUDA events); cross-rank: barrier, reset scoreboard, barrier (else the fused scoreboard races/deadlocks)."""
    e0 = torch.cuda.Event(enable_timing=True)
    e1 = torch.cuda.Event(enable_timing=True)

    def _iter(measure):
        if group is not None:
            torch.cuda.synchronize()
            group.barrier()  # all ranks done with previous fn
        if reset is not None:
            reset()
        if group is not None:
            torch.cuda.synchronize()
            group.barrier()  # all scoreboards clean before any fn
        if not measure:
            fn()
            return 0.0
        e0.record()
        fn()
        e1.record()
        torch.cuda.synchronize()
        return e0.elapsed_time(e1)

    for _ in range(warmup):
        _iter(False)
    _l2_flush()
    total = sum(_iter(True) for _ in range(iters))
    return total / iters


# --------------------------------------------------------------------------- #
# Test-data generator (consistent with the test): build the SymmBuffer, run the
# prologue to produce the dispatch plan, then dispatch once to fill the pool.
# --------------------------------------------------------------------------- #
def generate(group, *, T, H, I, E, K, BM, BN, num_dispatch_cu, mode, W1, W2, base_seed=7):
    rank = group.rank()
    torch.manual_seed(base_seed + rank)
    x = torch.randn((T, H), device="cuda", dtype=torch.float32).bfloat16()
    topk_idx, topk_w = generate_routing(T, K, E, mode, device="cuda", seed=100 + rank)

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
    # L1 GEMM output (2*inter wide) has no slot in the arena -> local scratch
    acc1 = torch.empty((symm.pool_capacity, 2 * I), dtype=torch.bfloat16, device="cuda")

    # prologue -> dispatch plan tables (same path as the test/forward); the prologue
    # resets scoreboard + barrier_local in-kernel and ends with a cross-rank barrier.
    (
        destination,
        start,
        count,
        source_offset_out,
        source_tokens,
        source_topk_slot,
        source_weight,
        tile_to_group,
        expected,
    ) = mega_moe_prologue_impl(
        topk_idx,
        topk_w,
        symm.peer_ptrs,
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
    # DeepEP-style dispatch handle (kernels read plan[:5]); num_tasks == num_experts
    plan = (destination, start, count, source_offset_out, source_tokens, source_topk_slot, source_weight)
    num_tile_blocks = symm.meta_scalars[1:2]  # device real-tile count
    symm.assert_capacity()  # fail loudly rather than silently drop rows

    # fill the pool (real A for gemm_only) via dispatch_only (peers synced by the prologue)
    torch.cuda.synchronize()
    group.barrier()
    dispatch_only(x, plan, symm.pool, symm.pool_ptrs, num_dispatch_cu=num_dispatch_cu)
    torch.cuda.synchronize()
    group.barrier()
    return SimpleNamespace(
        symm=symm,
        x=x,
        plan=plan,
        acc1=acc1,
        tile_to_group=tile_to_group,
        expected=expected,
        num_tile_blocks=num_tile_blocks,
        destination=destination,
        count=count,
        source_offset_out=source_offset_out,
        num_tasks=E,
        W1=W1,
        W2=W2,
    )


def profile_dispatch(group, args, mode, W1, W2):
    rank = group.rank()
    BM, BN, H, I, num_dispatch_cu = args.bm, args.bn, args.hidden, args.inter, args.num_dispatch_cu
    d = generate(
        group,
        T=args.num_tokens,
        H=H,
        I=I,
        E=args.num_experts,
        K=args.num_topk,
        BM=BM,
        BN=BN,
        num_dispatch_cu=num_dispatch_cu,
        mode=mode,
        W1=W1,
        W2=W2,
    )
    symm, x, plan = d.symm, d.x, d.plan
    pool = symm.pool
    try:
        # GEMM work = real (padded) pool rows x N x K ; L1 NT: N=2I, K=H
        real_tiles = int(d.num_tile_blocks[0].item())
        M_eff, N, Kd = real_tiles * BM, 2 * I, H
        flops = 2.0 * M_eff * N * Kd
        # XGMI push bytes per rank = remote rows (dest != rank) x hidden x 2 (bf16)
        dest_cpu = d.destination.cpu()
        cnt_cpu = d.count.cpu()
        remote_rows = int(cnt_cpu[dest_cpu != rank].sum().item())
        xgmi_bytes = remote_rows * H * 2

        def _sb_reset():
            symm.scoreboard.zero_()

        t_gg = _bench(
            lambda: grouped_gemm_bf16_only(
                pool, d.W1, d.acc1, d.tile_to_group, d.num_tile_blocks, BM=BM, BN=BN
            ),
            iters=args.iters,
        )
        # dense single-weight GEMM of the same M_eff x N x K = the grouped-GEMM roofline
        gm_cands = (1, 4, 8) if args.autotune else (args.dense_group_m,)
        t_dense, dense_gm = _dense_gemm_peak_ms(M_eff, N, Kd, BM, BN, args.iters, group_m_cands=gm_cands)
        t_disp = _bench(
            lambda: dispatch_only(x, plan, pool, symm.pool_ptrs, num_dispatch_cu=num_dispatch_cu),
            group=group,
            iters=args.iters,
        )
        # fused: full XGMI push + grouped GEMM (overlap)
        t_fused = _bench(
            lambda: dispatch_grouped_gemm_bf16(
                x,
                plan,
                pool,
                symm.pool_ptrs,
                d.W1,
                d.acc1,
                d.tile_to_group,
                symm.scoreboard,
                symm.scoreboard_ptrs,
                d.expected,
                d.num_tile_blocks,
                BM=BM,
                BN=BN,
                num_dispatch_cu=num_dispatch_cu,
            ),
            reset=_sb_reset,
            group=group,
            iters=args.iters,
        )

        # ── backward NN throughput (timing only): dispatch a random dgrad[T,2I] (mirrors
        # the forward x[T,H]) into a 2I-wide pool, then NN grouped GEMM @ W1[g,2I,H] ->
        # bwd_out[M,H]. Same comm plan/scoreboard and FLOPs as the forward L1.
        bwd = None
        if args.backward:
            T, Kb, Nb = args.num_tokens, 2 * I, H
            dgrad = (torch.randn(T, Kb, device="cuda", dtype=torch.float32) / 8).bfloat16()
            bwd_pool = _Symm(group, (symm.pool_capacity, Kb), torch.bfloat16)
            bwd_out = torch.empty(symm.pool_capacity, Nb, device="cuda", dtype=torch.bfloat16)
            try:
                torch.cuda.synchronize()
                group.barrier()
                dispatch_only(dgrad, plan, bwd_pool.local, bwd_pool.ptrs, num_dispatch_cu=num_dispatch_cu)
                torch.cuda.synchronize()
                group.barrier()
                t_bwd_gg = _bench(
                    lambda: grouped_gemm_bf16_only(
                        bwd_pool.local,
                        d.W1,
                        bwd_out,
                        d.tile_to_group,
                        d.num_tile_blocks,
                        layout="nn",
                        BM=BM,
                        BN=BN,
                    ),
                    iters=args.iters,
                )
                t_bwd_fused = _bench(
                    lambda: dispatch_grouped_gemm_bf16(
                        dgrad,
                        plan,
                        bwd_pool.local,
                        bwd_pool.ptrs,
                        d.W1,
                        bwd_out,
                        d.tile_to_group,
                        symm.scoreboard,
                        symm.scoreboard_ptrs,
                        d.expected,
                        d.num_tile_blocks,
                        layout="nn",
                        BM=BM,
                        BN=BN,
                        num_dispatch_cu=num_dispatch_cu,
                    ),
                    reset=_sb_reset,
                    group=group,
                    iters=args.iters,
                )
            finally:
                bwd_pool.sm.destroy()
            bwd = {"gg_ms": t_bwd_gg, "fused_ms": t_bwd_fused}
    finally:
        symm.destroy()  # always free symmetric buffers
    # raw per-rank timings + work; rank 0 aggregates across ranks (bottleneck = max latency)
    return {
        "gemm_only_ms": t_gg,
        "dense_gemm_only_ms": t_dense,
        "dense_gm": dense_gm,
        "dispatch_only_ms": t_disp,
        "fused_ms": t_fused,
        "flops": flops,
        "xgmi_bytes": xgmi_bytes,
        "M_eff": M_eff,
        "remote_rows": remote_rows,
        "bwd": bwd,
    }


def benchmark_dispatch(local_rank, world, args):
    ip = os.getenv("MASTER_ADDR", "127.0.0.1")
    port = int(os.getenv("MASTER_PORT", "8481"))
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", init_method=f"tcp://{ip}:{port}", world_size=world, rank=local_rank)
    torch.set_default_device("cuda")
    torch.cuda.set_device(local_rank)
    group = dist.new_group(list(range(world)))
    rank = dist.get_rank()
    platform, gpu_name = get_platform_info()

    # this rank's expert weights, built once (sliced from the deterministic global set)
    epr = args.num_experts // world
    W1g, W2g = _global_weights(args.num_experts, args.inter, args.hidden, "cuda")
    W1 = W1g[rank * epr : (rank + 1) * epr].contiguous()
    W2 = W2g[rank * epr : (rank + 1) * epr].contiguous()
    del W1g, W2g

    modes = ["load_balanced", "round_robin"] if args.mode == "both" else [args.mode]
    rows = []
    try:
        for mode in modes:
            r = profile_dispatch(group, args, mode, W1, W2)
            allp = [None] * world
            dist.all_gather_object(allp, (rank, r), group=group)
            if rank == 0:
                data = [p[1] for p in allp]
                # distributed bottleneck = slowest rank (max latency); work ~ uniform (mean)
                mx = lambda k: max(d[k] for d in data)
                mean = lambda k: sum(d[k] for d in data) / len(data)

                gg_ms, disp_ms, fused_ms = mx("gemm_only_ms"), mx("dispatch_only_ms"), mx("fused_ms")
                dense_ms = mx("dense_gemm_only_ms")
                flops, xgmi = mean("flops"), mean("xgmi_bytes")
                gg_tf = flops / (gg_ms * 1e-3) / 1e12
                dense_tf = flops / (dense_ms * 1e-3) / 1e12
                fused_tf = flops / (fused_ms * 1e-3) / 1e12
                disp_bw = xgmi / (disp_ms * 1e-3) / 1e9
                # grouped GEMM vs the dense single-weight roofline (is grouped at peak?)
                grouped_eff_pct = gg_tf / dense_tf * 100.0
                # fused vs running gemm then comm serially, and how close to the pure-gemm
                # roofline (comm fully hidden -> fused==gemm -> 100%):
                #   hidden  = serial(gemm+comm) - fused      (comm time hidden under GEMM)
                #   speedup = serial / fused                 (fused vs serial, e.g. 1.2x)
                #   roofline= gemm / fused                   (100% = comm fully hidden)
                serial_ms = gg_ms + disp_ms
                hidden_ms = serial_ms - fused_ms
                speedup = serial_ms / fused_ms
                roofline_pct = gg_ms / fused_ms * 100.0
                print(
                    f"\n{'='*72}\n[dispatch] {gpu_name} EP{world} T={args.num_tokens} H={args.hidden} "
                    f"I={args.inter} E={args.num_experts} K={args.num_topk} mode={mode} "
                    f"(max over ranks)\n{'='*72}"
                )
                print(
                    f"  dense_gemm   : {dense_ms:8.3f} ms | {dense_tf:7.1f} TFLOPS (single-weight roofline, "
                    f"GROUP_M={data[0]['dense_gm']})"
                )
                print(
                    f"  gemm_only    : {gg_ms:8.3f} ms | {gg_tf:7.1f} TFLOPS | "
                    f"grouped/dense = {grouped_eff_pct:.1f}%"
                )
                print(f"  dispatch_only: {disp_ms:8.3f} ms | {disp_bw:7.1f} GB/s (XGMI, full push)")
                print(
                    f"  fused        : {fused_ms:8.3f} ms | {fused_tf:7.1f} TFLOPS | "
                    f"roofline (gemm/fused) = {roofline_pct:.1f}% | speedup vs serial = {speedup:.2f}x"
                )
                if data[0]["bwd"] is not None:  # backward dgrad (NN), same FLOPs
                    bgg = max(d["bwd"]["gg_ms"] for d in data)
                    bfu = max(d["bwd"]["fused_ms"] for d in data)
                    bgg_tf = flops / (bgg * 1e-3) / 1e12
                    bfu_tf = flops / (bfu * 1e-3) / 1e12
                    print(f"  bwd gemm(nn) : {bgg:8.3f} ms | {bgg_tf:7.1f} TFLOPS (NN dgrad layout)")
                    print(
                        f"  bwd fused(nn): {bfu:8.3f} ms | {bfu_tf:7.1f} TFLOPS | "
                        f"roofline (gemm/fused) = {bgg/bfu*100:.1f}%"
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
                        "gemm_only (ms)": f"{gg_ms:.3f}",
                        "gemm_only (TFLOPS)": f"{gg_tf:.1f}",
                        "grouped/dense": f"{grouped_eff_pct:.1f}%",
                        "dispatch_only (ms)": f"{disp_ms:.3f}",
                        "dispatch_only (XGMI GB/s)": f"{disp_bw:.1f}",
                        "fused (ms)": f"{fused_ms:.3f}",
                        "fused (TFLOPS)": f"{fused_tf:.1f}",
                        "comm_hidden (ms)": f"{hidden_ms:.3f}",
                        "speedup (vs serial)": f"{speedup:.2f}x",
                        "roofline (gemm/fused)": f"{roofline_pct:.1f}%",
                    }
                )
            torch.cuda.synchronize()
            group.barrier()

        if rank == 0 and rows:
            results = pd.DataFrame(rows)
            print("\nFinal Results:")
            print(tabulate(results, headers="keys", tablefmt="grid", showindex=False))
            fn = args.output or f"dispatch_grouped_gemm_{datetime.now():%Y%m%d}_{gpu_name}.csv"
            results.to_csv(fn, index=False)
            print(f"Results saved to {fn}")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Benchmark fused BF16 dispatch + grouped GEMM")
    ap.add_argument("--num-processes", type=int, default=8)
    ap.add_argument("--hidden", type=int, default=7168)  # DeepSeek-V3
    ap.add_argument("--inter", type=int, default=2048)
    ap.add_argument("--num-experts", type=int, default=256)
    ap.add_argument("--num-topk", type=int, default=8)
    ap.add_argument("--num-tokens", type=int, default=8192)
    ap.add_argument("--bm", type=int, default=256)
    ap.add_argument("--bn", type=int, default=256)
    # 16 dispatch CUs already saturate the intra-node XGMI dispatch (~350 GB/s); more only
    # steal CUs from the GEMM (single-rank HBM wants ~96, but XGMI is ~15x slower so it
    # saturates with far fewer CUs).
    ap.add_argument("--num-dispatch-cu", type=int, default=16)
    # GROUP_M for the dense roofline reference; default = best, autotune sweeps {1,4,8}.
    ap.add_argument("--dense-group-m", type=int, default=4)
    ap.add_argument(
        "--autotune",
        action="store_true",
        help="sweep dense GROUP_M {1,4,8}; default off uses --dense-group-m (best)",
    )
    ap.add_argument(
        "--backward",
        action="store_true",
        help="also benchmark the backward dgrad (NN) fused dispatch + grouped GEMM",
    )
    ap.add_argument("--mode", choices=["load_balanced", "round_robin", "both"], default="both")
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--output", "-o", type=str, default=None)
    args = ap.parse_args()
    torch.multiprocessing.spawn(
        benchmark_dispatch, args=(args.num_processes, args), nprocs=args.num_processes
    )
