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

The inputs (act / weight / tile_to_expert / origin_rank / origin_slot / combine
buffers) are built over the production ``SymmBuffer`` (``get_symm_buffer_for_mega_moe``)
by running ``dispatch_prologue`` + ``dispatch_grouped_gemm_impl`` + SwiGLU --
the exact same path as the EP test.

Run inside dev_primus (8 GPUs):
  PYTHONPATH=<...>/Primus-Turbo python benchmark/ops/bench_grouped_gemm_comine.py \
      --num-processes 8 [--mode load_balanced|round_robin|both]
"""

import argparse
import os
import sys
from datetime import datetime

import pandas as pd
import torch
import torch.distributed as dist
from config import get_platform_info
from tabulate import tabulate

# repo root (primus_turbo) on the path so the shared mega_utils + kernels import
_HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..")))

from mega_utils import (  # noqa: E402
    bench,
    compute_stage_metrics,
    dense_gemm_peak_ms,
    generate_input,
    global_weights,
    grouped_gemm_bf16_only,
    print_header,
    print_stage,
)

from primus_turbo.flydsl.mega.grouped_gemm_combine_bf16_kernel import (  # noqa: E402
    combine_only,
    grouped_gemm_combine_bf16,
)


def profile_combine(group, args, mode, W1, W2):
    """Forward = e2e step 4 (NT): grouped L2 GEMM + combine PUSH + weighted topk reduce -> y.
    Backward = e2e step 3 (NN, always profiled): grad_l1 @ w1 (L1 dgrad) + combine PUSH +
    dx reduce. Both report the 3-role fused kernel (``grouped_gemm_combine_bf16`` with a
    reduce role), so ``fused`` == the actual e2e combine stage. The fused reduce is PERF ONLY:
    the per-slot ready flags are pre-zeroed each iter (no real cross-rank wait)."""
    rank = group.rank()
    BM, BN, H, I, num_combine_cu = args.bm, args.bn, args.hidden, args.inter, args.num_combine_cu
    inp = generate_input(
        group,
        kind="combine",
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
    tile_to_expert = inp.tile_to_expert  # prologue-returned expert-id-per-tile map
    try:
        real_tiles = int(symm.meta_scalars[1].item())
        M_eff = real_tiles * BM
        group_m_cands = (1, 4, 8) if args.autotune else (args.dense_group_m,)
        # combine CUs: autotune sweeps {16,32,48,64,96}; default off uses --num-combine-cu (best).
        cu_cands = (16, 32, 48, 64, 96) if args.autotune else (num_combine_cu,)

        # XGMI combine bytes per rank = remote rows (origin_rank != rank, valid) x H x 2
        origin = symm.origin_rank
        remote_rows = int(((origin != rank) & (origin >= 0)).sum().item())
        xgmi_bytes = remote_rows * H * 2

        # per-slot ready flags = symm.barrier_local (UNCACHED signal heap, shared fwd + bwd reduce).
        # 0 == ready (role 2 waits while flag < 0); pre-zeroed each iter -> no real cross-rank
        # wait. topk tables drive the per-token reduce.
        topk_idx_flat = inp.topk_idx.to(torch.int32).contiguous().view(-1)
        topk_w_flat = inp.topk_weight.to(torch.float32).contiguous().view(-1)

        def _reset_fused():
            symm.sb_l2.zero_()
            symm.barrier_local.zero_()

        # ---- forward (e2e step 4, NT): L2 GEMM N=H, K=I ; reduce -> y[T,H] (weighted) ----
        N, K = H, I
        flops = 2.0 * M_eff * N * K

        # L2 has small K=I -> short contraction; GROUP_M=8 reuses the per-expert weight
        # across more M-tiles than the GROUP_M=4 L1 default (best for this shape).
        t_gemm = bench(
            lambda: grouped_gemm_bf16_only(
                act, inp.W2, symm.l2_token_buffer, tile_to_expert, num_tile_blocks, BM=BM, BN=BN, GROUP_M=8
            ),
            iters=args.iters,
        )
        # dense single-weight GEMM of the same M_eff x N x K = the grouped-GEMM roofline
        t_dense, dense_group_m = dense_gemm_peak_ms(
            M_eff, N, K, BM, BN, args.iters, group_m_cands=group_m_cands
        )
        # combine_only and the fused kernel share the same CU candidates.
        comb_sweep = {}
        for cu in cu_cands:
            comb_sweep[cu] = bench(
                lambda cu=cu: combine_only(
                    group,
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
            fused_sweep[cu] = bench(
                lambda cu=cu: grouped_gemm_combine_bf16(
                    act,
                    inp.W2,
                    inp.handle,
                    group,
                    topk_indices=topk_idx_flat,
                    topk_weights=topk_w_flat,
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
        grad_l1 = torch.randn(symm.num_max_pool_tokens, bwd_K, device="cuda", dtype=torch.bfloat16) / 8

        t_bwd_gemm = bench(
            lambda: grouped_gemm_bf16_only(
                grad_l1,
                inp.W1,
                symm.l2_token_buffer,
                tile_to_expert,
                num_tile_blocks,
                layout="nn",
                BM=BM,
                BN=BN,
                GROUP_M=8,
            ),
            iters=args.iters,
        )
        t_bwd_dense, bwd_dense_gm = dense_gemm_peak_ms(
            M_eff, bwd_N, bwd_K, BM, BN, args.iters, group_m_cands=group_m_cands
        )
        bwd_comb_sweep = {}
        for cu in cu_cands:
            bwd_comb_sweep[cu] = bench(
                lambda cu=cu: combine_only(
                    group,
                    BM=BM,
                    num_combine_cu=cu,
                ),
                group=group,
                iters=args.iters,
            )
        best_bwd_comb_cu = min(bwd_comb_sweep, key=bwd_comb_sweep.get)
        # bwd fused = 3-role NN (GEMM + combine PUSH + unweighted dx reduce); weight rides grad_l1.
        # GEMM-bound -> default to the low --bwd-combine-cu (more CUs for the GEMM); autotune sweeps.
        bwd_cu_cands = cu_cands if args.autotune else (args.bwd_combine_cu,)
        bwd_fused_sweep = {}
        for cu in bwd_cu_cands:
            bwd_fused_sweep[cu] = bench(
                lambda cu=cu: grouped_gemm_combine_bf16(
                    grad_l1,
                    inp.W1,
                    inp.handle,
                    group,
                    topk_indices=topk_idx_flat,
                    topk_weights=None,
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
    W1_global, W2_global = global_weights(args.num_experts, args.inter, args.hidden, "cuda")
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

            flops, xgmi = rank_mean("flops"), rank_mean("xgmi_bytes")
            # forward metrics in the shared layout; combine appends CU-sweep details
            m = compute_stage_metrics(
                gemm_ms=rank_max("gemm_only_ms"),
                dense_ms=rank_max("dense_gemm_only_ms"),
                dense_gm=per_rank[0]["dense_gm"],
                comm_ms=rank_max("combine_only_ms"),
                fused_ms=rank_max("fused_ms"),
                flops=flops,
                xgmi=xgmi,
            )
            print_header("combine", gpu_name, world, args, mode)
            # fused = the 3-role kernel (GEMM + combine PUSH + weighted topk reduce -> y),
            # i.e. the actual e2e forward step 4 (PERF ONLY: reduce flags pre-set ready).
            print_stage(
                m,
                comm_label="combine_only",
                comm_unit="GB/s (XGMI)",
                comm_tag="comb",
                comm_extra=f" | combine_cu: {_sweep_str('comb_sweep')} ms (best={per_rank[0]['comb_cu']})",
                fused_extra=(
                    f" | hid {m.hidden_ms:.3f}/{m.comm_ms:.3f} ms comm | "
                    f"num_reduce_cu={args.num_reduce_cu} | "
                    f"combine_cu: {_sweep_str('fused_sweep')} ms (best={per_rank[0]['fused_cu']})"
                ),
            )

            # backward STEP3 (NN, = mega_moe_fused STEP 3): L1 dgrad GEMM + combine + dx reduce.
            # Same metric set as the forward; reduce is unweighted (weight rides grad_l1).
            bwd_max = lambda key: max(d["bwd"][key] for d in per_rank)
            bwd_mean = lambda key: sum(d["bwd"][key] for d in per_rank) / len(per_rank)

            def _bwd_sweep_str(key):
                cus = sorted(per_rank[0]["bwd"][key].keys())
                return " ".join(f"{cu}->{max(d['bwd'][key][cu] for d in per_rank):.3f}" for cu in cus)

            mb = compute_stage_metrics(
                gemm_ms=bwd_max("gemm_only_ms"),
                dense_ms=bwd_max("dense_gemm_only_ms"),
                dense_gm=per_rank[0]["bwd"]["dense_gm"],
                comm_ms=bwd_max("combine_only_ms"),
                fused_ms=bwd_max("fused_ms"),
                flops=bwd_mean("flops"),
                xgmi=bwd_mean("xgmi_bytes"),
            )
            print_stage(
                mb,
                comm_label="combine_only",
                comm_unit="GB/s (XGMI)",
                comm_tag="comb",
                sub_header="backward STEP3 (NN, = mega_moe_fused STEP 3)",
                comm_extra=(
                    f" | combine_cu: {_bwd_sweep_str('comb_sweep')} ms "
                    f"(best={per_rank[0]['bwd']['comb_cu']})"
                ),
                fused_extra=(
                    f" | hid {mb.hidden_ms:.3f}/{mb.comm_ms:.3f} ms comm | "
                    f"num_reduce_cu={args.num_reduce_cu} | "
                    f"combine_cu: {_bwd_sweep_str('fused_sweep')} ms (best={per_rank[0]['bwd']['fused_cu']})"
                ),
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
                    "dense_gemm (ms)": f"{m.dense_ms:.3f}",
                    "dense_gemm (TFLOPS)": f"{m.dense_tf:.1f}",
                    "gemm_only (ms)": f"{m.gemm_ms:.3f}",
                    "gemm_only (TFLOPS)": f"{m.gemm_tf:.1f}",
                    "grouped/dense": f"{m.grouped_eff_pct:.1f}%",
                    "combine_only (ms)": f"{m.comm_ms:.3f}",
                    "combine_only (XGMI GB/s)": f"{m.comm_bw:.1f}",
                    "fused (ms)": f"{m.fused_ms:.3f}",
                    "fused (TFLOPS)": f"{m.fused_tf:.1f}",
                    "comm_hidden (ms)": f"{m.hidden_ms:.3f}",
                    "speedup (vs serial)": f"{m.speedup:.2f}x",
                    "roofline (max(gemm,comb)/fused)": f"{m.roofline_pct:.1f}%",
                    "bwd dense_gemm (TFLOPS)": f"{mb.dense_tf:.1f}",
                    "bwd gemm_only (ms)": f"{mb.gemm_ms:.3f}",
                    "bwd gemm_only (TFLOPS)": f"{mb.gemm_tf:.1f}",
                    "bwd grouped/dense": f"{mb.grouped_eff_pct:.1f}%",
                    "bwd combine_only (ms)": f"{mb.comm_ms:.3f}",
                    "bwd combine_only (XGMI GB/s)": f"{mb.comm_bw:.1f}",
                    "bwd fused (ms)": f"{mb.fused_ms:.3f}",
                    "bwd fused (TFLOPS)": f"{mb.fused_tf:.1f}",
                    "bwd speedup (vs serial)": f"{mb.speedup:.2f}x",
                    "bwd roofline (max(gemm,comb)/fused)": f"{mb.roofline_pct:.1f}%",
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
        default=0,
        help="role-2 topk-reduce dedicated blocks (0 = reduce on empty GEMM blocks = best)",
    )
    # backward STEP3 is GEMM-bound (K=2I) -> fewer combine CUs leaves more CUs for the GEMM,
    # the combine still hides (XGMI-BW-bound, ~16 CU saturates). Forward STEP4 stays at 64.
    ap.add_argument("--bwd-combine-cu", type=int, default=20)
    ap.add_argument("--mode", choices=["load_balanced", "round_robin", "both"], default="both")
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--output", "-o", type=str, default=None)
    args = ap.parse_args()
    torch.multiprocessing.spawn(benchmark_combine, args=(args.num_processes, args), nprocs=args.num_processes)
