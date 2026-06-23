###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Benchmark the fused BF16 dispatch + grouped GEMM mega kernel (EP, intra-node).

Forward variants on the same prologue-generated data (so it matches
tests/pytorch/ops/test_mega_moe_dispatch_combine_grouped_gemm.py exactly):

  * gemm_only     -- grouped L1 GEMM over the dispatched pool (compute peak)   [TFLOPS]
  * dispatch_only -- cross-rank dispatch PUSH only                       [XGMI GB/s]
  * fused         -- dispatch PUSH + grouped L1 GEMM (overlap)                [TFLOPS]

Backward (always profiled) reproduces the e2e backward STEP1 of
``mega_moe_fused.backward`` -- the fused dispatch + L2 dgrad (= dispatch_grouped_0):
dispatch dy[T,H] + grouped NN GEMM ``pool[M,H] @ w2[g,H,I] -> d_swiglu[M,I]``.
Same metric set as the forward (dense roofline / gemm_only / dispatch_only / fused).

The inputs (pool / dispatch plan / tile_to_group / expected / scoreboard) are built
by the production ``SymmBuffer`` + ``mega_moe_prologue_impl`` -- the exact same path
as the end-to-end test.

Run inside dev_primus (8 GPUs):
  PYTHONPATH=<...>/Primus-Turbo python benchmark/ops/bench_dispatch_grouped_gemm.py \
      --num-processes 8 [--mode load_balanced|round_robin|both]
"""

import argparse
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
from primus_turbo.pytorch.kernels.mega_moe.mega_moe_prologue_impl import (  # noqa: E402
    mega_moe_prologue_impl,
)

# single-allocation symmetric buffer shared with the production mega_moe_fused forward
from primus_turbo.pytorch.ops.moe.mega_moe_fused import (  # noqa: E402
    get_symm_buffer_for_mega_moe,
)

_FLYDSL = BackendType.FLYDSL.value


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
# Cross-rank variants barrier each iter (and reset the scoreboard before it).
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
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

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
        start_event.record()
        fn()
        end_event.record()
        torch.cuda.synchronize()
        return start_event.elapsed_time(end_event)

    for _ in range(warmup):
        _iter(False)
    _l2_flush()
    total_ms = sum(_iter(True) for _ in range(iters))
    return total_ms / iters


# --------------------------------------------------------------------------- #
# Input builder (consistent with the test): build the SymmBuffer, run the
# prologue to produce the dispatch plan, then dispatch once to fill the pool.
# --------------------------------------------------------------------------- #
def build_inputs(group, *, T, H, I, E, K, BM, BN, num_dispatch_cu, mode, W1, W2, base_seed=7):
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
    l1_out = torch.empty((symm.pool_capacity, 2 * I), dtype=torch.bfloat16, device="cuda")

    # prologue -> dispatch plan tables (same path as the test/forward); the prologue
    # resets scoreboard + barrier_local in-kernel and ends with a cross-rank barrier.
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
    # DeepEP-style dispatch plan (kernels read plan[:5]); num_tasks == num_experts
    plan = tuple(plan)
    destination, _start, count, _src_offset, _src_tokens, _topk_slot, _weight = plan
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
        l1_out=l1_out,
        tile_to_group=tile_to_group,
        expected=expected,
        num_tile_blocks=num_tile_blocks,
        destination=destination,
        count=count,
        W1=W1,
        W2=W2,
    )


def profile_dispatch(group, args, mode, W1, W2):
    rank = group.rank()
    BM, BN, H, I, num_dispatch_cu = args.bm, args.bn, args.hidden, args.inter, args.num_dispatch_cu
    inp = build_inputs(
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
    symm, x, plan = inp.symm, inp.x, inp.plan
    pool = symm.pool
    try:
        # GEMM work = real (padded) pool rows x N x K ; L1 NT: N=2I, K=H
        real_tiles = int(inp.num_tile_blocks[0].item())
        M_eff, N, K = real_tiles * BM, 2 * I, H
        flops = 2.0 * M_eff * N * K
        # XGMI push bytes per rank = remote rows (dest != rank) x hidden x 2 (bf16)
        dest_cpu, count_cpu = inp.destination.cpu(), inp.count.cpu()
        remote_rows = int(count_cpu[dest_cpu != rank].sum().item())
        xgmi_bytes = remote_rows * H * 2

        def _reset_scoreboard():
            symm.scoreboard.zero_()

        t_gemm = _bench(
            lambda: grouped_gemm_bf16_only(
                pool, inp.W1, inp.l1_out, inp.tile_to_group, inp.num_tile_blocks, BM=BM, BN=BN
            ),
            iters=args.iters,
        )
        # dense single-weight GEMM of the same M_eff x N x K = the grouped-GEMM roofline
        group_m_cands = (1, 4, 8) if args.autotune else (args.dense_group_m,)
        t_dense, dense_group_m = _dense_gemm_peak_ms(
            M_eff, N, K, BM, BN, args.iters, group_m_cands=group_m_cands
        )
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
                inp.W1,
                inp.l1_out,
                inp.tile_to_group,
                symm.scoreboard,
                symm.scoreboard_ptrs,
                inp.expected,
                inp.num_tile_blocks,
                symm.sb_consume,
                BM=BM,
                BN=BN,
                num_dispatch_cu=num_dispatch_cu,
            ),
            reset=_reset_scoreboard,
            group=group,
            iters=args.iters,
        )

        # ── backward STEP1 (= e2e dispatch_grouped_0): dispatch dy[T,H] + L2 dgrad
        # pool[M,H] @ w2[g,H,I] (NN) -> d_swiglu[M,I]. Same metric set as the forward
        # (dense roofline / gemm_only / dispatch_only / fused). N=I, K=H; dispatch bytes
        # equal the forward (dy is H-wide like x).
        N_bwd = I  # backward STEP1 output width
        flops_bwd = 2.0 * M_eff * N_bwd * K
        dy = (torch.randn(args.num_tokens, H, device="cuda", dtype=torch.float32) / 8).bfloat16()
        d_swiglu = torch.empty(symm.pool_capacity, N_bwd, device="cuda", dtype=torch.bfloat16)

        # fill the pool with dy once for the gemm-only baseline
        torch.cuda.synchronize()
        group.barrier()
        dispatch_only(dy, plan, pool, symm.pool_ptrs, num_dispatch_cu=num_dispatch_cu)
        torch.cuda.synchronize()
        group.barrier()
        t_bwd_gemm = _bench(
            lambda: grouped_gemm_bf16_only(
                pool, inp.W2, d_swiglu, inp.tile_to_group, inp.num_tile_blocks, layout="nn", BM=BM, BN=BN
            ),
            iters=args.iters,
        )
        t_bwd_dense, bwd_dense_group_m = _dense_gemm_peak_ms(
            M_eff, N_bwd, K, BM, BN, args.iters, group_m_cands=group_m_cands
        )
        t_bwd_disp = _bench(
            lambda: dispatch_only(dy, plan, pool, symm.pool_ptrs, num_dispatch_cu=num_dispatch_cu),
            group=group,
            iters=args.iters,
        )
        t_bwd_fused = _bench(
            lambda: dispatch_grouped_gemm_bf16(
                dy,
                plan,
                pool,
                symm.pool_ptrs,
                inp.W2,
                d_swiglu,
                inp.tile_to_group,
                symm.scoreboard,
                symm.scoreboard_ptrs,
                inp.expected,
                inp.num_tile_blocks,
                symm.sb_consume,
                layout="nn",
                BM=BM,
                BN=BN,
                num_dispatch_cu=num_dispatch_cu,
            ),
            reset=_reset_scoreboard,
            group=group,
            iters=args.iters,
        )
        bwd = {
            "gemm_only_ms": t_bwd_gemm,
            "dense_gemm_only_ms": t_bwd_dense,
            "dense_gm": bwd_dense_group_m,
            "dispatch_only_ms": t_bwd_disp,
            "fused_ms": t_bwd_fused,
            "flops": flops_bwd,
        }
    finally:
        symm.destroy()  # always free symmetric buffers
    # raw per-rank timings + work; rank 0 aggregates across ranks (bottleneck = max latency)
    return {
        "gemm_only_ms": t_gemm,
        "dense_gemm_only_ms": t_dense,
        "dense_gm": dense_group_m,
        "dispatch_only_ms": t_disp,
        "fused_ms": t_fused,
        "flops": flops,
        "xgmi_bytes": xgmi_bytes,
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
    experts_per_rank = args.num_experts // world
    W1_global, W2_global = _global_weights(args.num_experts, args.inter, args.hidden, "cuda")
    W1 = W1_global[rank * experts_per_rank : (rank + 1) * experts_per_rank].contiguous()
    W2 = W2_global[rank * experts_per_rank : (rank + 1) * experts_per_rank].contiguous()
    del W1_global, W2_global

    modes = ["load_balanced", "round_robin"] if args.mode == "both" else [args.mode]
    rows = []
    try:
        for mode in modes:
            result = profile_dispatch(group, args, mode, W1, W2)
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

            gemm_ms, disp_ms, fused_ms = (
                rank_max("gemm_only_ms"),
                rank_max("dispatch_only_ms"),
                rank_max("fused_ms"),
            )
            dense_ms = rank_max("dense_gemm_only_ms")
            flops, xgmi = rank_mean("flops"), rank_mean("xgmi_bytes")
            gemm_tf = flops / (gemm_ms * 1e-3) / 1e12
            dense_tf = flops / (dense_ms * 1e-3) / 1e12
            fused_tf = flops / (fused_ms * 1e-3) / 1e12
            disp_bw = xgmi / (disp_ms * 1e-3) / 1e9
            # grouped GEMM vs the dense single-weight roofline (is grouped at peak?)
            grouped_eff_pct = gemm_tf / dense_tf * 100.0
            # fused vs running gemm then comm serially, and vs the overlap floor:
            #   hidden  = serial(gemm+comm) - fused      (comm time hidden under GEMM)
            #   speedup = serial / fused                 (fused vs serial, e.g. 1.2x)
            #   roofline= max(gemm,disp) / fused         (overlap floor: the slower leg)
            serial_ms = gemm_ms + disp_ms
            hidden_ms = serial_ms - fused_ms
            speedup = serial_ms / fused_ms
            roofline_pct = max(gemm_ms, disp_ms) / fused_ms * 100.0
            print(
                f"\n{'='*72}\n[dispatch] {gpu_name} EP{world} T={args.num_tokens} H={args.hidden} "
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
            print(f"  dispatch_only: {disp_ms:8.3f} ms | {disp_bw:7.1f} GB/s (XGMI, full push)")
            print(
                f"  fused        : {fused_ms:8.3f} ms | {fused_tf:7.1f} TFLOPS | "
                f"roofline (max(gemm,disp)/fused) = {roofline_pct:.1f}% | speedup vs serial = {speedup:.2f}x"
            )

            # backward STEP1 (= dispatch_grouped_0): same metric set as the forward.
            # N=I, K=H -> flops_bwd; dispatch bytes equal the forward (dy is H-wide).
            bwd_max = lambda key: max(d["bwd"][key] for d in per_rank)
            bwd_gemm_ms, bwd_disp_ms, bwd_fused_ms = (
                bwd_max("gemm_only_ms"),
                bwd_max("dispatch_only_ms"),
                bwd_max("fused_ms"),
            )
            bwd_dense_ms = bwd_max("dense_gemm_only_ms")
            bwd_flops = sum(d["bwd"]["flops"] for d in per_rank) / len(per_rank)
            bwd_gemm_tf = bwd_flops / (bwd_gemm_ms * 1e-3) / 1e12
            bwd_dense_tf = bwd_flops / (bwd_dense_ms * 1e-3) / 1e12
            bwd_fused_tf = bwd_flops / (bwd_fused_ms * 1e-3) / 1e12
            bwd_disp_bw = xgmi / (bwd_disp_ms * 1e-3) / 1e9
            bwd_grouped_eff = bwd_gemm_tf / bwd_dense_tf * 100.0
            bwd_speedup = (bwd_gemm_ms + bwd_disp_ms) / bwd_fused_ms
            bwd_roofline = max(bwd_gemm_ms, bwd_disp_ms) / bwd_fused_ms * 100.0
            print(f"  {'-'*68}  backward STEP1 (NN, = dispatch_grouped_0)")
            print(
                f"  dense_gemm   : {bwd_dense_ms:8.3f} ms | {bwd_dense_tf:7.1f} TFLOPS (single-weight roofline, "
                f"GROUP_M={per_rank[0]['bwd']['dense_gm']})"
            )
            print(
                f"  gemm_only    : {bwd_gemm_ms:8.3f} ms | {bwd_gemm_tf:7.1f} TFLOPS | "
                f"grouped/dense = {bwd_grouped_eff:.1f}%"
            )
            print(f"  dispatch_only: {bwd_disp_ms:8.3f} ms | {bwd_disp_bw:7.1f} GB/s (XGMI, full push)")
            print(
                f"  fused        : {bwd_fused_ms:8.3f} ms | {bwd_fused_tf:7.1f} TFLOPS | "
                f"roofline (max(gemm,disp)/fused) = {bwd_roofline:.1f}% | speedup vs serial = {bwd_speedup:.2f}x"
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
                    "dispatch_only (ms)": f"{disp_ms:.3f}",
                    "dispatch_only (XGMI GB/s)": f"{disp_bw:.1f}",
                    "fused (ms)": f"{fused_ms:.3f}",
                    "fused (TFLOPS)": f"{fused_tf:.1f}",
                    "comm_hidden (ms)": f"{hidden_ms:.3f}",
                    "speedup (vs serial)": f"{speedup:.2f}x",
                    "roofline (max(gemm,disp)/fused)": f"{roofline_pct:.1f}%",
                    "bwd dense_gemm (TFLOPS)": f"{bwd_dense_tf:.1f}",
                    "bwd gemm_only (ms)": f"{bwd_gemm_ms:.3f}",
                    "bwd gemm_only (TFLOPS)": f"{bwd_gemm_tf:.1f}",
                    "bwd grouped/dense": f"{bwd_grouped_eff:.1f}%",
                    "bwd dispatch_only (ms)": f"{bwd_disp_ms:.3f}",
                    "bwd dispatch_only (XGMI GB/s)": f"{bwd_disp_bw:.1f}",
                    "bwd fused (ms)": f"{bwd_fused_ms:.3f}",
                    "bwd fused (TFLOPS)": f"{bwd_fused_tf:.1f}",
                    "bwd speedup (vs serial)": f"{bwd_speedup:.2f}x",
                    "bwd roofline (max(gemm,disp)/fused)": f"{bwd_roofline:.1f}%",
                }
            )
            torch.cuda.synchronize()
            group.barrier()

        if rank == 0 and rows:
            results = pd.DataFrame(rows)
            print("\nFinal Results:")
            print(tabulate(results, headers="keys", tablefmt="grid", showindex=False))
            out_file = args.output or f"dispatch_grouped_gemm_{datetime.now():%Y%m%d}_{gpu_name}.csv"
            results.to_csv(out_file, index=False)
            print(f"Results saved to {out_file}")
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
    ap.add_argument("--mode", choices=["load_balanced", "round_robin", "both"], default="both")
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--output", "-o", type=str, default=None)
    args = ap.parse_args()
    torch.multiprocessing.spawn(
        benchmark_dispatch, args=(args.num_processes, args), nprocs=args.num_processes
    )
