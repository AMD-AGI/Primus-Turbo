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

The inputs (pool / dispatch handle / tile_to_expert / expected / scoreboard) are built
by the production ``SymmBuffer`` + ``dispatch_prologue`` -- the exact same path
as the end-to-end test.

Run inside dev_primus (8 GPUs):
  PYTHONPATH=<...>/Primus-Turbo python benchmark/ops/bench_dispatch_grouped_gemm.py \
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
    check_accuracy,
    compute_stage_metrics,
    dense_gemm_peak_ms,
    dispatch_only,
    generate_input,
    global_weights,
    grouped_gemm_bf16_only,
    grouped_gemm_tn_wgrad_only,
    print_header,
    print_stage,
)

import primus_turbo.pytorch  # noqa: E402,F401  (fully init pytorch first: dodges the
from primus_turbo.flydsl.mega.dispatch_grouped_gemm_bf16_kernel import (  # noqa: E402
    dispatch_grouped_gemm_bf16,
)
from primus_turbo.pytorch.ops import grouped_gemm as _turbo_gg  # noqa: E402

# symm_buffer<->pytorch<->swiglu circular import that the mega kernels otherwise trigger)


def profile_dispatch(group, args, mode, W1, W2):
    rank = group.rank()
    BM, BN, H, I, num_dispatch_cu = args.bm, args.bn, args.hidden, args.inter, args.num_dispatch_cu
    inp = generate_input(
        group,
        kind="dispatch",
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
    symm, x, handle = inp.symm, inp.x, inp.handle
    pool = symm.pool

    # one synced cross-rank op (barrier, barrier) so the accuracy probes run in a
    # clean state, isolated from the timing loops. The dispatch scoreboard is a
    # never-reset parity double-buffer, so no signal reset is needed here.
    def _synced(fn):
        torch.cuda.synchronize()
        group.barrier()
        out = fn()
        torch.cuda.synchronize()
        group.barrier()
        return out

    acc = {}
    try:
        # GEMM work = real (padded) pool rows x N x K ; L1 NT: N=2I, K=H
        real_tiles = int(inp.num_tile_blocks[0].item())
        M_eff, N, K = real_tiles * BM, 2 * I, H
        flops = 2.0 * M_eff * N * K
        # BM-padded per-expert row counts: pool[:M_eff] is laid out expert-major with
        # each expert padded to a BM multiple; padding rows are zero -> contract to 0.
        # These let turbo grouped_gemm reproduce the fused GEMM in the SAME pool layout.
        experts_per_rank = args.num_experts // group.size()
        t2e = inp.tile_to_expert[:real_tiles].to(torch.int64)
        counts = torch.bincount(t2e, minlength=experts_per_rank)[:experts_per_rank]
        padded_group_lens = (counts * BM).to(torch.int64)  # sum == M_eff
        # XGMI push bytes per rank = remote rows (dest != rank) x hidden x 2 (bf16)
        dest_cpu, count_cpu = inp.destination.cpu(), inp.count.cpu()
        remote_rows = int(count_cpu[dest_cpu != rank].sum().item())
        xgmi_bytes = remote_rows * H * 2

        t_gemm = bench(
            lambda: grouped_gemm_bf16_only(
                pool, inp.W1, inp.l1_out, inp.tile_to_expert, inp.num_tile_blocks, BM=BM, BN=BN
            ),
            iters=args.iters,
        )
        # dense single-weight GEMM of the same M_eff x N x K = the grouped-GEMM roofline
        group_m_cands = (1, 4, 8) if args.autotune else (args.dense_group_m,)
        t_dense, dense_group_m = dense_gemm_peak_ms(
            M_eff, N, K, BM, BN, args.iters, group_m_cands=group_m_cands
        )
        t_disp = bench(
            lambda: dispatch_only(x, handle, symm, num_dispatch_cu=num_dispatch_cu),
            iters=args.iters,
        )
        # fused: full XGMI push + grouped GEMM (overlap)
        t_fused = bench(
            lambda: dispatch_grouped_gemm_bf16(
                x,
                inp.W1,
                group,
                handle=handle,
                layout="nt",
                BM=BM,
                BN=BN,
                num_dispatch_cu=num_dispatch_cu,
            ),
            iters=args.iters,
        )

        # accuracy: fused dispatch+GEMM(NT) vs the turbo grouped_gemm reference over the
        # same dispatched pool (pool currently holds x). turbo gg reproduces the per-expert
        # GEMM in the SAME BM-padded pool layout via padded_group_lens, so we can compare
        # the real (non-padding) rows directly. NT -> trans_b=True, b=W1 [G, 2I, H].
        ref_fwd = _synced(
            lambda: _turbo_gg(pool[:M_eff].contiguous(), inp.W1, padded_group_lens, trans_b=True)
        )
        fwd_fused = _synced(
            lambda: dispatch_grouped_gemm_bf16(
                x, inp.W1, group, handle=handle, layout="nt", BM=BM, BN=BN, num_dispatch_cu=num_dispatch_cu
            ),
        )[0]
        acc["fwd"] = check_accuracy(group, "fwd fused (nt)", fwd_fused[:M_eff], ref_fwd[:M_eff])

        # ── backward STEP1 (= e2e dispatch_grouped_0): dispatch dy[T,H] + L2 dgrad
        # pool[M,H] @ w2[g,H,I] (NN) -> d_swiglu[M,I]. Same metric set as the forward
        # (dense roofline / gemm_only / dispatch_only / fused). N=I, K=H; dispatch bytes
        # equal the forward (dy is H-wide like x).
        N_bwd = I  # backward STEP1 output width
        flops_bwd = 2.0 * M_eff * N_bwd * K
        dy = (torch.randn(args.num_tokens, H, device="cuda", dtype=torch.float32) / 8).bfloat16()
        d_swiglu = torch.empty(symm.num_max_pool_tokens, N_bwd, device="cuda", dtype=torch.bfloat16)

        # fill the pool with dy once for the gemm-only baseline
        torch.cuda.synchronize()
        group.barrier()
        dispatch_only(dy, handle, symm, num_dispatch_cu=num_dispatch_cu)
        torch.cuda.synchronize()
        group.barrier()
        t_bwd_gemm = bench(
            lambda: grouped_gemm_bf16_only(
                pool, inp.W2, d_swiglu, inp.tile_to_expert, inp.num_tile_blocks, layout="nn", BM=BM, BN=BN
            ),
            iters=args.iters,
        )
        t_bwd_dense, bwd_dense_group_m = dense_gemm_peak_ms(
            M_eff, N_bwd, K, BM, BN, args.iters, group_m_cands=group_m_cands
        )
        t_bwd_disp = bench(
            lambda: dispatch_only(dy, handle, symm, num_dispatch_cu=num_dispatch_cu),
            iters=args.iters,
        )
        t_bwd_fused = bench(
            lambda: dispatch_grouped_gemm_bf16(
                dy,
                inp.W2,
                group,
                handle=handle,
                layout="nn",
                BM=BM,
                BN=BN,
                num_dispatch_cu=num_dispatch_cu,
            ),
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

        # accuracy: fused dispatch+GEMM(NN) vs the turbo grouped_gemm reference over the
        # same dispatched pool (pool currently holds dy). NN -> trans_b=False, b=W2 [G, H, I].
        ref_bwd = _synced(
            lambda: _turbo_gg(pool[:M_eff].contiguous(), inp.W2, padded_group_lens, trans_b=False)
        )
        bwd_fused = _synced(
            lambda: dispatch_grouped_gemm_bf16(
                dy, inp.W2, group, handle=handle, layout="nn", BM=BM, BN=BN, num_dispatch_cu=num_dispatch_cu
            ),
        )[0]
        acc["bwd"] = check_accuracy(group, "bwd STEP1 fused (nn)", bwd_fused[:M_eff], ref_bwd[:M_eff])

        # group metadata shared by the variable-K wgrads: block_m-padded per-expert
        # boundaries (padded rows contract to 0, matching mega_moe_fused).
        group_offs = torch.zeros(experts_per_rank + 1, dtype=torch.int32, device="cuda")
        group_offs[1:] = padded_group_lens.to(torch.int32).cumsum(0)
        cap = symm.num_max_pool_tokens
        ext_handle = handle  # full prologue tuple (tile_to_expert / expected / group_offs inside)

        # ── backward wgrad dW1 (TN, variable-K) — the e2e _grouped_variable_k_gemm
        # hotspot. dW1 = pool(x)^T @ grad_l1: a/b swapped vs grad-first — lhs = pool(x)
        # (feature H, the dispatched operand), rhs = grad_l1 (gate+up, feature 2I,
        # resident). OUT_M = H, OUT_N = 2I. The gemm_only path matches mega_moe_fused's
        # local grouped TN wgrad (no comm in e2e dW1); the fused column is the
        # hypothetical dispatch+wgrad overlap story.
        OUT_M_w1, OUT_N_w1 = H, 2 * I  # dW1: lhs feature = H (pool x), rhs feature = 2I (grad_l1)
        flops_wg1 = 2.0 * M_eff * OUT_M_w1 * OUT_N_w1
        x_pool = (torch.randn(cap, OUT_M_w1, device="cuda", dtype=torch.float32) / 8).bfloat16()
        grad_pool_w1 = (torch.randn(cap, OUT_N_w1, device="cuda", dtype=torch.float32) / 8).bfloat16()
        # trans_c: dW1 stored transposed [G, OUT_N, OUT_M] = [G, 2I, H] = W1-native layout
        dW1 = torch.empty(experts_per_rank, OUT_N_w1, OUT_M_w1, device="cuda", dtype=torch.bfloat16)
        # gemm_only: resident lhs/rhs pools (no comm) = the e2e local grouped TN wgrad
        t_wg1_gemm = bench(
            lambda: grouped_gemm_tn_wgrad_only(
                x_pool, grad_pool_w1, group_offs, dW1, BLOCK_M=BM, BLOCK_N=BN, trans_c=True
            ),
            iters=args.iters,
        )
        # dense roofline of the same total FLOPs (one [H,2I] GEMM contracting M_eff rows)
        t_wg1_dense, wg1_dense_group_m = dense_gemm_peak_ms(
            OUT_M_w1, OUT_N_w1, M_eff, BM, BN, args.iters, group_m_cands=group_m_cands
        )
        # comm baseline = dispatch the H-wide lhs (matches the fused dispatch width)
        t_wg1_disp = bench(
            lambda: dispatch_only(x_pool, handle, symm, num_dispatch_cu=num_dispatch_cu),
            iters=args.iters,
        )
        t_wg1_fused = bench(
            lambda: dispatch_grouped_gemm_bf16(
                x_pool,
                grad_pool_w1,
                group,
                handle=ext_handle,
                layout="tn",
                num_dispatch_cu=num_dispatch_cu,
                BM=BM,
                BN=BN,
                trans_c=True,
            ),
            iters=args.iters,
            group=group,
        )
        wgrad1 = {
            "gemm_only_ms": t_wg1_gemm,
            "dense_gemm_only_ms": t_wg1_dense,
            "dense_gm": wg1_dense_group_m,
            "dispatch_only_ms": t_wg1_disp,
            "fused_ms": t_wg1_fused,
            "flops": flops_wg1,
        }

        # accuracy: fused dispatch+wgrad(TN) vs a pure-torch per-expert wgrad reference
        # over the same dispatched pool (turbo grouped_gemm has no wgrad / trans_a path).
        # The fused path pushes x_pool over XGMI, so the reference wgrads the DISPATCHED
        # pool (not resident x_pool). trans_c stores dW1[e] = grad[e]^T @ x_pool[e] = [2I, H].
        ref_dW1 = torch.zeros_like(dW1)  # empty groups -> 0 (matches the fused padded output)
        offs_cpu = group_offs.tolist()

        def _ref_wg1():
            dispatch_only(x_pool, handle, symm, num_dispatch_cu=num_dispatch_cu)
            torch.cuda.synchronize()
            group.barrier()
            for e in range(experts_per_rank):
                s, t = offs_cpu[e], offs_cpu[e + 1]
                if t > s:  # padded rows are zero -> contract to 0 (skip is equivalent)
                    ref_dW1[e] = (grad_pool_w1[s:t].float().T @ pool[s:t].float()).to(dW1.dtype)

        _synced(_ref_wg1)
        wg1_fused = _synced(
            lambda: dispatch_grouped_gemm_bf16(
                x_pool,
                grad_pool_w1,
                group,
                handle=ext_handle,
                layout="tn",
                num_dispatch_cu=num_dispatch_cu,
                BM=BM,
                BN=BN,
                trans_c=True,
            ),
        )[0]
        acc["wgrad1"] = check_accuracy(group, "wgrad dW1 fused (tn)", wg1_fused, ref_dW1)
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
        "wgrad1": wgrad1,
        "acc": acc,
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
    W1_global, W2_global = global_weights(args.num_experts, args.inter, args.hidden, "cuda")
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

            flops, xgmi = rank_mean("flops"), rank_mean("xgmi_bytes")
            # fused accuracy verdicts (global worst-rank, identical on every rank)
            acc0 = per_rank[0]["acc"]
            fmt_acc = lambda v: "n/a" if v is None else f"{v[0]:.5f}/{'PASS' if v[2] else 'FAIL'}"
            # forward metrics in the shared layout
            m = compute_stage_metrics(
                gemm_ms=rank_max("gemm_only_ms"),
                dense_ms=rank_max("dense_gemm_only_ms"),
                dense_gm=per_rank[0]["dense_gm"],
                comm_ms=rank_max("dispatch_only_ms"),
                fused_ms=rank_max("fused_ms"),
                flops=flops,
                xgmi=xgmi,
            )
            print_header("dispatch", gpu_name, world, args, mode)
            print_stage(m, comm_label="dispatch_only", comm_unit="GB/s (XGMI, nodeup)", comm_tag="disp")

            # backward STEP1 (= dispatch_grouped_0): same metric set as the forward.
            # N=I, K=H -> flops_bwd; dispatch bytes equal the forward (dy is H-wide).
            bwd_max = lambda key: max(d["bwd"][key] for d in per_rank)
            bwd_flops = sum(d["bwd"]["flops"] for d in per_rank) / len(per_rank)
            mb = compute_stage_metrics(
                gemm_ms=bwd_max("gemm_only_ms"),
                dense_ms=bwd_max("dense_gemm_only_ms"),
                dense_gm=per_rank[0]["bwd"]["dense_gm"],
                comm_ms=bwd_max("dispatch_only_ms"),
                fused_ms=bwd_max("fused_ms"),
                flops=bwd_flops,
                xgmi=xgmi,
            )
            print_stage(
                mb,
                comm_label="dispatch_only",
                comm_unit="GB/s (XGMI, nodeup)",
                comm_tag="disp",
                sub_header="backward STEP1 (NN, = dispatch_grouped_0)",
            )

            # backward wgrad dW1 (TN, variable-K): OUT_M=2I, OUT_N=H.
            w1_max = lambda key: max(d["wgrad1"][key] for d in per_rank)
            w1_flops = sum(d["wgrad1"]["flops"] for d in per_rank) / len(per_rank)
            mw1 = compute_stage_metrics(
                gemm_ms=w1_max("gemm_only_ms"),
                dense_ms=w1_max("dense_gemm_only_ms"),
                dense_gm=per_rank[0]["wgrad1"]["dense_gm"],
                comm_ms=w1_max("dispatch_only_ms"),
                fused_ms=w1_max("fused_ms"),
                flops=w1_flops,
                xgmi=xgmi,
            )
            print_stage(
                mw1,
                comm_label="dispatch_only",
                comm_unit="GB/s (XGMI, nodeup)",
                comm_tag="disp",
                sub_header="backward wgrad dW1 (TN, = dispatch + variable-K wgrad)",
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
                    "dispatch_only (ms)": f"{m.comm_ms:.3f}",
                    "dispatch_only (XGMI GB/s)": f"{m.comm_bw:.1f}",
                    "fused (ms)": f"{m.fused_ms:.3f}",
                    "fused (TFLOPS)": f"{m.fused_tf:.1f}",
                    "fused acc (cos/ok)": fmt_acc(acc0.get("fwd")),
                    "comm_hidden (ms)": f"{m.hidden_ms:.3f}",
                    "speedup (vs serial)": f"{m.speedup:.2f}x",
                    "roofline (max(gemm,disp)/fused)": f"{m.roofline_pct:.1f}%",
                    "bwd dense_gemm (TFLOPS)": f"{mb.dense_tf:.1f}",
                    "bwd gemm_only (ms)": f"{mb.gemm_ms:.3f}",
                    "bwd gemm_only (TFLOPS)": f"{mb.gemm_tf:.1f}",
                    "bwd grouped/dense": f"{mb.grouped_eff_pct:.1f}%",
                    "bwd dispatch_only (ms)": f"{mb.comm_ms:.3f}",
                    "bwd dispatch_only (XGMI GB/s)": f"{mb.comm_bw:.1f}",
                    "bwd fused (ms)": f"{mb.fused_ms:.3f}",
                    "bwd fused (TFLOPS)": f"{mb.fused_tf:.1f}",
                    "bwd fused acc (cos/ok)": fmt_acc(acc0.get("bwd")),
                    "bwd speedup (vs serial)": f"{mb.speedup:.2f}x",
                    "bwd roofline (max(gemm,disp)/fused)": f"{mb.roofline_pct:.1f}%",
                    "wgrad1 dense_gemm (TFLOPS)": f"{mw1.dense_tf:.1f}",
                    "wgrad1 gemm_only (ms)": f"{mw1.gemm_ms:.3f}",
                    "wgrad1 gemm_only (TFLOPS)": f"{mw1.gemm_tf:.1f}",
                    "wgrad1 grouped/dense": f"{mw1.grouped_eff_pct:.1f}%",
                    "wgrad1 dispatch_only (ms)": f"{mw1.comm_ms:.3f}",
                    "wgrad1 fused (ms)": f"{mw1.fused_ms:.3f}",
                    "wgrad1 fused (TFLOPS)": f"{mw1.fused_tf:.1f}",
                    "wgrad1 fused acc (cos/ok)": fmt_acc(acc0.get("wgrad1")),
                    "wgrad1 speedup (vs serial)": f"{mw1.speedup:.2f}x",
                    "wgrad1 roofline (max(gemm,disp)/fused)": f"{mw1.roofline_pct:.1f}%",
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
