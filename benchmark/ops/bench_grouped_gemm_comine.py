###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Benchmark the fused BF16 grouped GEMM + combine mega kernel (EP, intra-node).

Three variants on the same prologue-generated data (so it matches
tests/pytorch/ops/test_mega_moe_dispatch_combine_grouped_gemm.py exactly):

  * gemm_only    -- grouped L2 GEMM over the SwiGLU activation (compute peak)  [TFLOPS]
  * combine_only -- cross-rank combine PUSH only                        [XGMI GB/s]
  * fused        -- grouped L2 GEMM + combine PUSH (overlap)                  [TFLOPS]

The test data (act / weight / tile_to_group / origin_rank / origin_slot / combine
buffers) is built by ``MegaFusedMoE._prologue`` + dispatch + SwiGLU -- the exact
same path as the test.

Run inside dev_primus (8 GPUs):
  PYTHONPATH=<...>/Primus-Turbo python benchmark/ops/bench_grouped_gemm_comine.py \
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
    _SymmSig,
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
            group.barrier()  # all state clean before any fn
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
# prologue + dispatch + SwiGLU so ``act`` is the real L2 input, then fill l2y once.
# --------------------------------------------------------------------------- #
def generate(group, *, T, H, I, E, K, BM, BN, mode, W1, W2, base_seed=7):
    rank = group.rank()
    torch.manual_seed(base_seed + rank)
    x = torch.randn((T, H), device="cuda", dtype=torch.float32).bfloat16()
    topk_idx, topk_w = generate_routing(T, K, E, mode, device="cuda", seed=100 + rank)

    mega = MegaFusedMoE(
        group, num_tokens=T, hidden=H, inter=I, num_experts=E, num_topk=K, W1=W1, W2=W2, BM=BM, BN=BN
    )
    mega._topk_idx = topk_idx  # kept for the optional 3-role reduce

    res = mega._prologue(topk_idx, topk_w)  # <-- the dispatch plan (same as the test)
    mega.assert_capacity()  # fail loudly rather than silently drop rows
    mega._reset()
    torch.cuda.synchronize()
    group.barrier()
    torch.cuda.synchronize()
    mega._dispatch(x, res)  # fill pool -> acc1 (autotuned, same as test)
    act = mega._swiglu()  # L2 GEMM input
    grouped_gemm_bf16_only(
        act, W2, mega.l2y, res.tile_to_group, mega._num_tile_blocks, BM=BM, BN=BN
    )  # fill l2y (real rows for combine_only)
    torch.cuda.synchronize()
    group.barrier()
    return SimpleNamespace(mega=mega, act=act, res=res, W2=W2)


def profile_combine(group, args, mode, W1, W2):
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
        mode=mode,
        W1=W1,
        W2=W2,
    )
    mega, act, res = d.mega, d.act, d.res
    sm_bar = None  # 3-role barrier symm buffer (freed in finally)
    try:
        # GEMM work = real (padded) pool rows x N x K ; L2 NT: N=H, K=I
        real_tiles = int(mega.wp.buffers["meta_scalars"][1].item())
        M_eff, N, Kd = real_tiles * BM, H, I
        flops = 2.0 * M_eff * N * Kd
        # XGMI combine bytes per rank = remote rows (origin_rank != rank, valid) x H x 2
        origin = res.origin_rank
        remote_rows = int(((origin != rank) & (origin >= 0)).sum().item())
        xgmi_bytes = remote_rows * H * 2

        comb_addrs = mega._sm_comb.ptrs
        slots = mega.combine_slots

        # L2 has small K=I -> short contraction; GROUP_M=8 reuses the per-expert weight
        # across more M-tiles than the GROUP_M=4 L1 default (best for this shape).
        t_gg = _bench(
            lambda: grouped_gemm_bf16_only(
                act, d.W2, mega.l2y, res.tile_to_group, mega._num_tile_blocks, BM=BM, BN=BN, GROUP_M=8
            ),
            iters=args.iters,
        )
        # dense single-weight GEMM of the same M_eff x N x K = the grouped-GEMM roofline
        gm_cands = (1, 4, 8) if args.autotune else (args.dense_group_m,)
        t_dense, dense_gm = _dense_gemm_peak_ms(M_eff, N, Kd, BM, BN, args.iters, group_m_cands=gm_cands)
        # combine CUs: autotune sweeps {16,32,64}; default off uses the single best
        # (--comm-blocks=32). combine_only and the fused kernel share the same candidate.
        cu_cands = (16, 32, 48, 64, 96) if args.autotune else (comm_blocks,)
        cb_sweep = {}
        for cu in cu_cands:
            cb_sweep[cu] = _bench(
                lambda cu=cu: combine_only(
                    mega.l2y,
                    res.origin_rank,
                    res.origin_slot,
                    comb_addrs,
                    slots,
                    mega._num_tile_blocks,
                    BM=BM,
                    num_combine_cu=cu,
                ),
                group=group,
                iters=args.iters,
            )
        best_cb_cu = min(cb_sweep, key=cb_sweep.get)
        t_cb = cb_sweep[best_cb_cu]
        fused_sweep = {}
        for cu in cu_cands:
            fused_sweep[cu] = _bench(
                lambda cu=cu: grouped_gemm_combine_bf16(
                    act,
                    d.W2,
                    mega.l2y,
                    res.tile_to_group,
                    mega.sb_l2,
                    res.origin_rank,
                    res.origin_slot,
                    comb_addrs,
                    slots,
                    mega._num_tile_blocks,
                    BM=BM,
                    BN=BN,
                    num_combine_cu=cu,
                ),
                reset=mega.sb_l2.zero_,
                group=group,
                iters=args.iters,
            )
        best_fused_cu = min(fused_sweep, key=fused_sweep.get)
        t_fused = fused_sweep[best_fused_cu]

        # ── backward NN (dgrad), timing only. L2 dgrad: dact[M,I] = dy[M,H] @ W2[g,H,I]
        # (NN, weight [G,K=H,N=I] -> reuses W2 as-is). Same dispatch plan / scoreboard /
        # combine buffers as the forward; combine pushes the I-wide dact rows (I<H -> fits).
        bwd = {}
        if args.backward:
            M_pool = mega.pool_capacity
            dy = torch.randn(M_pool, H, device="cuda", dtype=torch.bfloat16) / 8
            bwd_l2y = torch.empty(M_pool, I, dtype=torch.bfloat16, device="cuda")
            bwd_flops = 2.0 * M_eff * I * H  # same dims as forward L2, B transposed
            bwd_xgmi = remote_rows * I * 2
            t_bgg = _bench(
                lambda: grouped_gemm_bf16_only(
                    dy,
                    d.W2,
                    bwd_l2y,
                    res.tile_to_group,
                    mega._num_tile_blocks,
                    layout="nn",
                    BM=BM,
                    BN=BN,
                    GROUP_M=8,
                ),
                iters=args.iters,
            )
            bwd_fused_sweep = {}
            for cu in cu_cands:
                bwd_fused_sweep[cu] = _bench(
                    lambda cu=cu: grouped_gemm_combine_bf16(
                        dy,
                        d.W2,
                        bwd_l2y,
                        res.tile_to_group,
                        mega.sb_l2,
                        res.origin_rank,
                        res.origin_slot,
                        comb_addrs,
                        slots,
                        mega._num_tile_blocks,
                        layout="nn",
                        BM=BM,
                        BN=BN,
                        num_combine_cu=cu,
                    ),
                    reset=mega.sb_l2.zero_,
                    group=group,
                    iters=args.iters,
                )
            best_bwd_cu = min(bwd_fused_sweep, key=bwd_fused_sweep.get)
            bwd = {
                "bwd_gemm_ms": t_bgg,
                "bwd_fused_ms": bwd_fused_sweep[best_bwd_cu],
                "bwd_fused_sweep": bwd_fused_sweep,
                "bwd_cu": best_bwd_cu,
                "bwd_flops": bwd_flops,
                "bwd_xgmi_bytes": bwd_xgmi,
            }
            del dy, bwd_l2y

        # optional 3-role (GEMM + combine + in-kernel topk reduce). PERF ONLY -- correctness is
        # NOT expected: origin_slot is source-order (not token-major) and the fused reduce is
        # compiler-blocked, so we pre-set the barrier ready each iter to keep role 2 from spinning.
        t_fused3 = None
        if args.reduce_cu > 0:
            T = args.num_tokens
            out3 = torch.empty(T, H, dtype=torch.bfloat16, device="cuda")
            # per-slot completion flags in the UNCACHED signal pad -> relaxed cross-rank reads stay fresh
            sm_bar = _SymmSig(group, (slots,), torch.int32)
            tki = mega._topk_idx.to(torch.int32).contiguous().view(-1)
            ntok = torch.full((group.size(),), T, dtype=torch.int32, device="cuda")

            def _reset3():
                mega.sb_l2.zero_()
                sm_bar.local.fill_(1)  # flags ready -> role 2 never blocks

            t_fused3 = _bench(
                lambda: grouped_gemm_combine_bf16(
                    act,
                    d.W2,
                    mega.l2y,
                    res.tile_to_group,
                    mega.sb_l2,
                    res.origin_rank,
                    res.origin_slot,
                    comb_addrs,
                    slots,
                    mega._num_tile_blocks,
                    output=out3,
                    comb_local=mega._sm_comb.local,
                    barrier_local=sm_bar.local,
                    barrier_addrs=sm_bar.ptrs,
                    topk_indices=tki,
                    num_tokens_per_rank=ntok,
                    topk=args.num_topk,
                    num_experts=args.num_experts,
                    rank=rank,
                    BM=BM,
                    BN=BN,
                    num_combine_cu=comm_blocks,
                    num_reduce_cu=args.reduce_cu,
                ),
                reset=_reset3,
                group=group,
                iters=args.iters,
            )
    finally:
        if sm_bar is not None:
            sm_bar.sm.destroy()
        mega.destroy()  # always free symmetric buffers
    # raw per-rank timings + work; rank 0 aggregates across ranks (bottleneck = max latency)
    return {
        "gemm_only_ms": t_gg,
        "dense_gemm_only_ms": t_dense,
        "dense_gm": dense_gm,
        "combine_only_ms": t_cb,
        "cb_sweep": cb_sweep,
        "cb_cu": best_cb_cu,
        "fused_ms": t_fused,
        "fused_sweep": fused_sweep,
        "fused_cu": best_fused_cu,
        "fused3_ms": t_fused3,
        "flops": flops,
        "xgmi_bytes": xgmi_bytes,
        "M_eff": M_eff,
        "remote_rows": remote_rows,
        **bwd,
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
    epr = args.num_experts // world
    W1g, W2g = _global_weights(args.num_experts, args.inter, args.hidden, "cuda")
    W1 = W1g[rank * epr : (rank + 1) * epr].contiguous()
    W2 = W2g[rank * epr : (rank + 1) * epr].contiguous()
    del W1g, W2g

    modes = ["load_balanced", "round_robin"] if args.mode == "both" else [args.mode]
    rows = []
    try:
        for mode in modes:
            r = profile_combine(group, args, mode, W1, W2)
            allp = [None] * world
            dist.all_gather_object(allp, (rank, r), group=group)
            if rank == 0:
                data = [p[1] for p in allp]
                # distributed bottleneck = slowest rank (max latency); work ~ uniform (mean)
                mx = lambda k: max(d[k] for d in data)
                mean = lambda k: sum(d[k] for d in data) / len(data)

                def _sweep_str(key):  # "cu->ms" for each swept config (max over ranks)
                    cus = sorted(data[0][key].keys())
                    return " ".join(f"{cu}->{max(d[key][cu] for d in data):.3f}" for cu in cus)

                gg_ms, cb_ms, fused_ms = mx("gemm_only_ms"), mx("combine_only_ms"), mx("fused_ms")
                dense_ms = mx("dense_gemm_only_ms")
                flops, xgmi = mean("flops"), mean("xgmi_bytes")
                gg_tf = flops / (gg_ms * 1e-3) / 1e12
                dense_tf = flops / (dense_ms * 1e-3) / 1e12
                fused_tf = flops / (fused_ms * 1e-3) / 1e12
                cb_bw = xgmi / (cb_ms * 1e-3) / 1e9
                # grouped GEMM vs the dense single-weight roofline (is grouped at peak?)
                grouped_eff_pct = gg_tf / dense_tf * 100.0
                # fused vs running gemm then comm serially, and how close to the pure-gemm
                # roofline (comm fully hidden -> fused==gemm -> 100%):
                #   hidden  = serial(gemm+comm) - fused      (comm time hidden under GEMM)
                #   speedup = serial / fused                 (fused vs serial, e.g. 1.2x)
                #   roofline= gemm / fused                   (100% = comm fully hidden)
                serial_ms = gg_ms + cb_ms
                hidden_ms = serial_ms - fused_ms
                speedup = serial_ms / fused_ms
                roofline_pct = gg_ms / fused_ms * 100.0
                print(
                    f"\n{'='*72}\n[combine] {gpu_name} EP{world} T={args.num_tokens} H={args.hidden} "
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
                print(
                    f"  combine_only : {cb_ms:8.3f} ms | {cb_bw:7.1f} GB/s (XGMI) | "
                    f"combine_cu: {_sweep_str('cb_sweep')} ms (best={data[0]['cb_cu']})"
                )
                print(
                    f"  fused        : {fused_ms:8.3f} ms | {fused_tf:7.1f} TFLOPS | "
                    f"hid {hidden_ms:.3f}/{cb_ms:.3f} ms comm | speedup vs serial = {speedup:.2f}x | "
                    f"roofline (gemm/fused) = {roofline_pct:.1f}% | "
                    f"combine_cu: {_sweep_str('fused_sweep')} ms (best={data[0]['fused_cu']})"
                )
                fused3_row = {}
                if data[0]["fused3_ms"] is not None:
                    f3_ms = mx("fused3_ms")
                    f3_tf = flops / (f3_ms * 1e-3) / 1e12
                    print(
                        f"  fused3 (+r2) : {f3_ms:8.3f} ms | {f3_tf:7.1f} TFLOPS | "
                        f"vs fused = {f3_ms / fused_ms:.2f}x (reduce_cu={args.reduce_cu}, PERF ONLY)"
                    )
                    fused3_row = {
                        "fused3 (ms)": f"{f3_ms:.3f}",
                        "fused3 (TFLOPS)": f"{f3_tf:.1f}",
                        "fused3/fused": f"{f3_ms / fused_ms:.2f}x",
                    }
                bwd_row = {}
                if data[0].get("bwd_gemm_ms") is not None:
                    bgg_ms, bf_ms = mx("bwd_gemm_ms"), mx("bwd_fused_ms")
                    bflops, bxgmi = mean("bwd_flops"), mean("bwd_xgmi_bytes")
                    bgg_tf, bf_tf = bflops / (bgg_ms * 1e-3) / 1e12, bflops / (bf_ms * 1e-3) / 1e12
                    b_roof = bgg_ms / bf_ms * 100.0  # 100% = combine fully hidden under GEMM
                    print(f"  bwd gemm(nn) : {bgg_ms:8.3f} ms | {bgg_tf:7.1f} TFLOPS (NN dgrad layout)")
                    print(
                        f"  bwd fused(nn): {bf_ms:8.3f} ms | {bf_tf:7.1f} TFLOPS | "
                        f"roofline (gemm/fused) = {b_roof:.1f}% | "
                        f"combine_cu: {_sweep_str('bwd_fused_sweep')} ms (best={data[0]['bwd_cu']})"
                    )
                    bwd_row = {
                        "bwd gemm(nn) (ms)": f"{bgg_ms:.3f}",
                        "bwd gemm(nn) (TFLOPS)": f"{bgg_tf:.1f}",
                        "bwd fused(nn) (ms)": f"{bf_ms:.3f}",
                        "bwd fused(nn) (TFLOPS)": f"{bf_tf:.1f}",
                        "bwd roofline": f"{b_roof:.1f}%",
                    }
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
                        "combine_only (ms)": f"{cb_ms:.3f}",
                        "combine_only (XGMI GB/s)": f"{cb_bw:.1f}",
                        "fused (ms)": f"{fused_ms:.3f}",
                        "fused (TFLOPS)": f"{fused_tf:.1f}",
                        "comm_hidden (ms)": f"{hidden_ms:.3f}",
                        "speedup (vs serial)": f"{speedup:.2f}x",
                        "roofline (gemm/fused)": f"{roofline_pct:.1f}%",
                        **fused3_row,
                        **bwd_row,
                    }
                )
            torch.cuda.synchronize()
            group.barrier()

        if rank == 0 and rows:
            results = pd.DataFrame(rows)
            print("\nFinal Results:")
            print(tabulate(results, headers="keys", tablefmt="grid", showindex=False))
            fn = args.output or f"grouped_gemm_combine_{datetime.now():%Y%m%d}_{gpu_name}.csv"
            results.to_csv(fn, index=False)
            print(f"Results saved to {fn}")
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
    ap.add_argument("--comm-blocks", type=int, default=64)
    # GROUP_M for the dense roofline reference; default = best, autotune sweeps {1,4,8}.
    ap.add_argument("--dense-group-m", type=int, default=4)
    ap.add_argument(
        "--autotune",
        action="store_true",
        help="sweep combine CUs {16,32,48,64,96} + dense GROUP_M {1,4,8}; "
        "default off uses --comm-blocks / --dense-group-m (best)",
    )
    ap.add_argument(
        "--reduce-cu",
        type=int,
        default=0,
        help="role-2 topk-reduce blocks; >0 enables the 3-role fused bench (perf only)",
    )
    ap.add_argument(
        "--backward",
        action="store_true",
        help="also bench the backward NN dgrad (gemm_only + fused combine), timing only",
    )
    ap.add_argument("--mode", choices=["load_balanced", "round_robin", "both"], default="both")
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--output", "-o", type=str, default=None)
    args = ap.parse_args()
    torch.multiprocessing.spawn(benchmark_combine, args=(args.num_processes, args), nprocs=args.num_processes)
