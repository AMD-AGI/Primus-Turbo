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

The test data (pool / comm tasks / tile_to_group / expected / scoreboard) is built
by ``MegaFusedMoE._prologue`` -- the exact same path as the test.

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
    MegaFusedMoE,
    _global_weights,
    _Symm,
    generate_routing,
)

from primus_turbo.flydsl.mega.dispatch_grouped_gemm_bf16_kernel import (  # noqa: E402
    dispatch_dedup_only,
    dispatch_grouped_gemm_bf16,
    dispatch_only,
    grouped_gemm_bf16_only,
)
from primus_turbo.flydsl.mega.gemm_bf16_kernel import (  # noqa: E402
    _compile_dense_nt,
    _get_compiled_dense,
)


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
# Test-data generator (consistent with the test): build MegaFusedMoE, run the
# prologue to produce the dispatch plan, then dispatch once to fill the pool.
# --------------------------------------------------------------------------- #
def generate(group, *, T, H, I, E, K, BM, BN, comm_blocks, mode, W1, W2, dedup=False, base_seed=7):
    rank = group.rank()
    torch.manual_seed(base_seed + rank)
    x = torch.randn((T, H), device="cuda", dtype=torch.float32).bfloat16()
    topk_idx, topk_w = generate_routing(T, K, E, mode, device="cuda", seed=100 + rank)

    mega = MegaFusedMoE(
        group,
        num_tokens=T,
        hidden=H,
        inter=I,
        num_experts=E,
        num_topk=K,
        W1=W1,
        W2=W2,
        BM=BM,
        BN=BN,
        dedup=dedup,
    )

    res = mega._prologue(topk_idx, topk_w)  # <-- the dispatch plan (same as the test)
    mega.assert_capacity()  # fail loudly rather than silently drop rows
    mega._reset()
    torch.cuda.synchronize()
    group.barrier()
    torch.cuda.synchronize()
    # fill the pool (real A for gemm_only) via dispatch_only -- avoids mega._dispatch's
    # autotune (which sweeps comm_blocks and can hit the scoreboard deadlock).
    cc = res.comm_tasks
    dispatch_only(
        x,
        cc.dest,
        cc.start,
        cc.cnt,
        cc.srcoff,
        cc.src_tokens,
        cc.num_comm,
        mega._sm_pool.local,
        mega._sm_pool.ptrs,
        comm_blocks=comm_blocks,
    )
    torch.cuda.synchronize()
    group.barrier()
    return SimpleNamespace(mega=mega, x=x, res=res, W1=W1, W2=W2)


def profile_dispatch(group, args, mode, W1, W2):
    rank = group.rank()
    BM, BN, H, I, comm_blocks = args.bm, args.bn, args.hidden, args.inter, args.comm_blocks
    d = generate(
        group,
        T=args.num_tokens,
        H=H,
        I=I,
        E=args.num_experts,
        K=args.num_topk,
        BM=BM,
        BN=BN,
        comm_blocks=comm_blocks,
        mode=mode,
        W1=W1,
        W2=W2,
        dedup=args.dedup,
    )
    mega, x, res = d.mega, d.x, d.res
    pool = mega._sm_pool.local
    try:
        # GEMM work = real (padded) pool rows x N x K ; L1 NT: N=2I, K=H
        real_tiles = int(mega.wp.buffers["meta_scalars"][1].item())
        M_eff, N, Kd = real_tiles * BM, 2 * I, H
        flops = 2.0 * M_eff * N * Kd
        # XGMI push bytes per rank = remote rows (dest != rank) x hidden x 2 (bf16)
        comm = res.comm_tasks
        dest_cpu = comm.dest.cpu()
        cnt_cpu = comm.cnt.cpu()
        remote_rows = int(cnt_cpu[dest_cpu != rank].sum().item())
        xgmi_bytes = remote_rows * H * 2
        # dedup (opt-in): only PRIMARY remote rows actually cross XGMI (secondaries skipped)
        xgmi_bytes_dedup = None
        if args.dedup:
            srcoff_cpu = comm.srcoff.cpu()
            sd_cpu = res.source_dedup.cpu()
            prim_remote = 0
            for t in range(comm.num_comm):
                if int(dest_cpu[t]) != rank:
                    o = int(srcoff_cpu[t])
                    c = int(cnt_cpu[t])
                    prim_remote += int((sd_cpu[o : o + c] == 0).sum().item())
            xgmi_bytes_dedup = prim_remote * H * 2

        def _sb_reset():
            mega._sm_scoreboard.local.zero_()

        def _fused_reset():
            mega._sm_scoreboard.local.zero_()
            mega.sb_copy.zero_()

        t_gg = _bench(
            lambda: grouped_gemm_bf16_only(
                pool, d.W1, mega.acc1, res.tile_to_group, mega._num_tile_blocks, BM=BM, BN=BN
            ),
            iters=args.iters,
        )
        # dense single-weight GEMM of the same M_eff x N x K = the grouped-GEMM roofline
        gm_cands = (1, 4, 8) if args.autotune else (args.dense_group_m,)
        t_dense, dense_gm = _dense_gemm_peak_ms(M_eff, N, Kd, BM, BN, args.iters, group_m_cands=gm_cands)
        t_disp = _bench(
            lambda: dispatch_only(
                x,
                comm.dest,
                comm.start,
                comm.cnt,
                comm.srcoff,
                comm.src_tokens,
                comm.num_comm,
                pool,
                mega._sm_pool.ptrs,
                comm_blocks=comm_blocks,
            ),
            group=group,
            iters=args.iters,
        )
        # fused WITHOUT dedup (baseline: full XGMI push + GEMM) -- the default path
        t_fused_nd = _bench(
            lambda: dispatch_grouped_gemm_bf16(
                x,
                comm.dest,
                comm.start,
                comm.cnt,
                comm.srcoff,
                comm.src_tokens,
                comm.num_comm,
                pool,
                mega._sm_pool.ptrs,
                d.W1,
                mega.acc1,
                res.tile_to_group,
                mega._sm_scoreboard.local,
                mega._sm_scoreboard.ptrs,
                res.expected,
                mega._num_tile_blocks,
                BM=BM,
                BN=BN,
                comm_blocks=comm_blocks,
            ),
            reset=_sb_reset,
            group=group,
            iters=args.iters,
        )
        # dedup paths (opt-in via --dedup): 2-role dispatch_dd + 3-role fused (dispatch ‖ permute ‖ gemm).
        # permute role CUs: autotune sweeps {16,32}, default off uses --permute-blocks (=32).
        disp_dd_sweep, fused_dd_sweep = {}, {}
        t_disp_dd = best_pb = best_fpb = None
        if args.dedup:
            pb_cands = (16, 32) if args.autotune else (args.permute_blocks,)
            for pb in pb_cands:
                disp_dd_sweep[pb] = _bench(
                    lambda pb=pb: dispatch_dedup_only(
                        x,
                        comm.dest,
                        comm.start,
                        comm.cnt,
                        comm.srcoff,
                        comm.src_tokens,
                        comm.num_comm,
                        pool,
                        mega._sm_pool.ptrs,
                        mega._sm_scoreboard.local,
                        mega._sm_scoreboard.ptrs,
                        res.expected,
                        mega._num_tile_blocks,
                        res.source_dedup,
                        res.dedup_src_row,
                        mega.sb_copy,
                        BM=BM,
                        dispatch_blocks=comm_blocks,
                        permute_blocks=pb,
                    ),
                    reset=_fused_reset,
                    group=group,
                    iters=args.iters,
                )
            best_pb = min(disp_dd_sweep, key=disp_dd_sweep.get)
            t_disp_dd = disp_dd_sweep[best_pb]
            for pb in pb_cands:
                fused_dd_sweep[pb] = _bench(
                    lambda pb=pb: dispatch_grouped_gemm_bf16(
                        x,
                        comm.dest,
                        comm.start,
                        comm.cnt,
                        comm.srcoff,
                        comm.src_tokens,
                        comm.num_comm,
                        pool,
                        mega._sm_pool.ptrs,
                        d.W1,
                        mega.acc1,
                        res.tile_to_group,
                        mega._sm_scoreboard.local,
                        mega._sm_scoreboard.ptrs,
                        res.expected,
                        mega._num_tile_blocks,
                        BM=BM,
                        BN=BN,
                        comm_blocks=comm_blocks,
                        permute_blocks=pb,
                        source_dedup=res.source_dedup,
                        dedup_src_row=res.dedup_src_row,
                        sb_copy=mega.sb_copy,
                    ),
                    reset=_fused_reset,
                    group=group,
                    iters=args.iters,
                )
            best_fpb = min(fused_dd_sweep, key=fused_dd_sweep.get)
        # headline fused = dedup best when enabled, else the no-dedup path
        t_fused = fused_dd_sweep[best_fpb] if args.dedup else t_fused_nd

        # ── backward NN throughput (timing only): dispatch a random dgrad[T,2I] (mirrors
        # the forward x[T,H]) into a 2I-wide pool, then NN grouped GEMM @ W1[g,2I,H] ->
        # bwd_out[M,H]. Same comm plan/scoreboard and FLOPs as the forward L1.
        bwd = None
        if args.backward:
            T, Kb, Nb = args.num_tokens, 2 * I, H
            dgrad = (torch.randn(T, Kb, device="cuda", dtype=torch.float32) / 8).bfloat16()
            bwd_pool = _Symm(group, (mega.pool_capacity, Kb), torch.bfloat16)
            bwd_out = torch.empty(mega.pool_capacity, Nb, device="cuda", dtype=torch.bfloat16)
            try:
                torch.cuda.synchronize()
                group.barrier()
                dispatch_only(
                    dgrad,
                    comm.dest,
                    comm.start,
                    comm.cnt,
                    comm.srcoff,
                    comm.src_tokens,
                    comm.num_comm,
                    bwd_pool.local,
                    bwd_pool.ptrs,
                    comm_blocks=comm_blocks,
                )
                torch.cuda.synchronize()
                group.barrier()
                t_bwd_gg = _bench(
                    lambda: grouped_gemm_bf16_only(
                        bwd_pool.local,
                        d.W1,
                        bwd_out,
                        res.tile_to_group,
                        mega._num_tile_blocks,
                        layout="nn",
                        BM=BM,
                        BN=BN,
                    ),
                    iters=args.iters,
                )
                t_bwd_fused = _bench(
                    lambda: dispatch_grouped_gemm_bf16(
                        dgrad,
                        comm.dest,
                        comm.start,
                        comm.cnt,
                        comm.srcoff,
                        comm.src_tokens,
                        comm.num_comm,
                        bwd_pool.local,
                        bwd_pool.ptrs,
                        d.W1,
                        bwd_out,
                        res.tile_to_group,
                        mega._sm_scoreboard.local,
                        mega._sm_scoreboard.ptrs,
                        res.expected,
                        mega._num_tile_blocks,
                        layout="nn",
                        BM=BM,
                        BN=BN,
                        comm_blocks=comm_blocks,
                    ),
                    reset=_sb_reset,
                    group=group,
                    iters=args.iters,
                )
            finally:
                bwd_pool.sm.destroy()
            bwd = {"gg_ms": t_bwd_gg, "fused_ms": t_bwd_fused}
    finally:
        mega.destroy()  # always free symmetric buffers
    # raw per-rank timings + work; rank 0 aggregates across ranks (bottleneck = max latency)
    return {
        "dispatch_dedup_ms": t_disp_dd,
        "disp_dd_pb": best_pb,
        "disp_dd_sweep": disp_dd_sweep,
        "fused_dd_pb": best_fpb,
        "fused_dd_sweep": fused_dd_sweep,
        "fused_nodedup_ms": t_fused_nd,
        "xgmi_bytes_dedup": xgmi_bytes_dedup,
        "gemm_only_ms": t_gg,
        "dense_gemm_only_ms": t_dense,
        "dense_gm": dense_gm,
        "dispatch_only_ms": t_disp,
        "fused_ms": t_fused,
        "flops": flops,
        "xgmi_bytes": xgmi_bytes,
        "M_eff": M_eff,
        "remote_rows": remote_rows,
        "dedup": args.dedup,
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

                def _sweep_str(key):  # "pb->ms" for each swept config (max over ranks)
                    pbs = sorted(data[0][key].keys())
                    return " ".join(f"{pb}->{max(d[key][pb] for d in data):.3f}" for pb in pbs)

                dedup_on = data[0]["dedup"]
                gg_ms, disp_ms, fused_ms = mx("gemm_only_ms"), mx("dispatch_only_ms"), mx("fused_ms")
                dense_ms = mx("dense_gemm_only_ms")
                fused_nd_ms = mx("fused_nodedup_ms")
                flops, xgmi = mean("flops"), mean("xgmi_bytes")
                gg_tf = flops / (gg_ms * 1e-3) / 1e12
                dense_tf = flops / (dense_ms * 1e-3) / 1e12
                fused_tf = flops / (fused_ms * 1e-3) / 1e12
                fused_nd_tf = flops / (fused_nd_ms * 1e-3) / 1e12
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
                    f"dedup={dedup_on} (max over ranks)\n{'='*72}"
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
                # headline fused: no-dedup by default; dedup is opt-in (--dedup)
                tag = "fused (dedup)" if dedup_on else "fused        "
                print(
                    f"  {tag}: {fused_ms:8.3f} ms | {fused_tf:7.1f} TFLOPS | "
                    f"roofline (gemm/fused) = {roofline_pct:.1f}% | speedup vs serial = {speedup:.2f}x"
                )
                if dedup_on:
                    disp_dd_ms = mx("dispatch_dedup_ms")
                    xgmi_dd = mean("xgmi_bytes_dedup")
                    disp_dd_bw = xgmi_dd / (disp_dd_ms * 1e-3) / 1e9  # XGMI-only bytes / total time
                    print(
                        f"  dispatch_dd  : {disp_dd_ms:8.3f} ms | {disp_dd_bw:7.1f} GB/s (XGMI primaries) | "
                        f"dedup push = {xgmi_dd/max(xgmi,1)*100:.0f}% of full | vs dispatch_only = {disp_ms/disp_dd_ms:.2f}x | "
                        f"permute_blocks: {_sweep_str('disp_dd_sweep')} ms (best={data[0]['disp_dd_pb']})"
                    )
                    print(
                        f"  fused_nodedup: {fused_nd_ms:8.3f} ms | {fused_nd_tf:7.1f} TFLOPS | "
                        f"dedup vs nodedup = {fused_nd_ms/fused_ms:.2f}x | "
                        f"permute_blocks: {_sweep_str('fused_dd_sweep')} ms (best={data[0]['fused_dd_pb']})"
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
                        "dedup": dedup_on,
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
    # 16 comm CUs already saturate the intra-node XGMI dispatch (~350 GB/s); more only
    # steal CUs from the GEMM (single-rank HBM wants ~96, but XGMI is ~15x slower so it
    # saturates with far fewer CUs).
    ap.add_argument("--comm-blocks", type=int, default=16)
    # permute (role-2) CUs for the dedup path; default = best, autotune sweeps {16,32}.
    ap.add_argument("--permute-blocks", type=int, default=32)
    # GROUP_M for the dense roofline reference; default = best, autotune sweeps {1,4,8}.
    ap.add_argument("--dense-group-m", type=int, default=4)
    ap.add_argument(
        "--autotune",
        action="store_true",
        help="sweep permute_blocks {16,32} + dense GROUP_M {1,4,8}; "
        "default off uses --permute-blocks / --dense-group-m (best)",
    )
    ap.add_argument(
        "--dedup",
        action="store_true",
        help="benchmark the token-dedup paths (dispatch_dd + 3-role fused); default off",
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
