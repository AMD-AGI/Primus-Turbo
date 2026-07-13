###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Benchmark the fused BF16 dispatch + grouped GEMM mega kernel (EP, intra-node)."""

import argparse
import os
import statistics
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Optional

import pandas as pd
import torch
import torch.distributed as dist
from config import gen_moe_test_cases, get_platform_info
from tabulate import tabulate

# repo root (primus_turbo) on the path (config + mega_utils are same-dir)
_HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..", "..")))

from mega_utils import (  # noqa: E402
    BF16_BYTES,
    aggregate_stage_metrics,
    all_ranks_ok,
    apply_case,
    bench,
    check_accuracy,
    checks_verdict,
    dense_gemm_peak_ms,
    dispatch_only,
    generate_input,
    grouped_gemm_bf16_only,
    grouped_gemm_variable_k_only,
    print_header,
    print_stage,
    stage_columns,
    sync_ranks,
    turbo_csv_row,
)

# import primus_turbo.pytorch first to dodge the mega kernels' circular import
import primus_turbo.pytorch  # noqa: E402,F401
from primus_turbo.flydsl.mega import (  # noqa: E402
    dispatch_grouped_gemm_bf16_flydsl_kernel,
)
from primus_turbo.pytorch.ops import grouped_gemm as turbo_grouped_gemm  # noqa: E402


@dataclass
class StageMetrics:
    """Raw per-rank timings + work for one stage (fields match compute_stage_metrics kwargs)."""

    gemm_ms: float
    dense_ms: float
    dense_gm: int
    comm_ms: float
    fused_ms: float
    flops: float


@dataclass
class StageSpec:
    """Declarative spec for one benchmarked stage; fused_fn is reused for timing + accuracy probe."""

    name: str
    flops: float
    dense_dims: tuple[int, int, int]
    gemm_fn: Callable
    dispatch_fn: Callable
    fused_fn: Callable
    ref_fn: Callable
    acc_slice: slice
    fused_group: Optional[Any] = None


@dataclass
class RunContext:
    """Shared per-run state built by _make_context: config, input/symm buffers, derived geometry."""

    # collective + CLI config
    group: Any
    args: Any
    rank: int
    # dispatched input namespace + symmetric buffers (own x / handle / pool / cap)
    inp: Any
    symm: Any
    # derived geometry: expert-major pool, each expert padded to a BM multiple
    M_eff: int  # total padded pool rows this rank GEMMs
    experts_per_rank: int
    padded_group_lens: Any  # per-expert BM-padded row counts, sum == M_eff
    group_offs: Any  # cumulative padded boundaries [experts_per_rank + 1]


class StageRunner:
    """Binds per-run context so each stage passes only its own knobs; run does the 5-step template."""

    def __init__(self, group, args, synced_fn, group_m_cands):
        self.group = group
        self.args = args
        self.synced_fn = synced_fn
        self.group_m_cands = group_m_cands

    def run(self, spec):
        """Time the 5-step template for spec, then gate accuracy; returns (StageMetrics, AccuracyCheck)."""
        args = self.args
        dense_m, dense_n, dense_k = spec.dense_dims
        t_gemm = bench(spec.gemm_fn, iters=args.iters)
        t_dense, dense_gm = dense_gemm_peak_ms(
            dense_m, dense_n, dense_k, args.bm, args.bn, args.iters, group_m_cands=self.group_m_cands
        )
        t_disp = bench(spec.dispatch_fn, iters=args.iters)
        t_fused = bench(spec.fused_fn, iters=args.iters)
        metrics = StageMetrics(
            gemm_ms=t_gemm,
            dense_ms=t_dense,
            dense_gm=dense_gm,
            comm_ms=t_disp,
            fused_ms=t_fused,
            flops=spec.flops,
        )
        # accuracy: fused vs ref over the SAME dispatched pool (both under a synced bracket)
        ref_out = self.synced_fn(spec.ref_fn)
        out = self.synced_fn(spec.fused_fn)[0]
        check = check_accuracy(self.group, spec.name, out[spec.acc_slice], ref_out[spec.acc_slice])
        return metrics, check


def _make_context(group, args):
    """Build input + symm buffers and precompute shared geometry (weights built in generate_input)."""
    inp = generate_input(
        group,
        kind="dispatch",
        T=args.num_tokens,
        H=args.hidden,
        I=args.inter,
        E=args.num_experts,
        K=args.num_topk,
        BLOCK_M=args.bm,
        BLOCK_N=args.bn,
        num_dispatch_cu=args.num_dispatch_cu,
    )
    experts_per_rank = args.num_experts // group.size()
    # BM-padded expert-major per-expert row counts (padding rows zero) so turbo gg matches the layout
    real_tiles = int(inp.num_tile_blocks[0].item())
    M_eff = real_tiles * args.bm
    tile_experts = inp.tile_to_expert[:real_tiles].to(torch.int64)
    counts = torch.bincount(tile_experts, minlength=experts_per_rank)[:experts_per_rank]
    padded_group_lens = (counts * args.bm).to(torch.int64)  # sum == M_eff
    # block_m-padded per-expert boundaries for the variable-K wgrads
    group_offs = torch.zeros(experts_per_rank + 1, dtype=torch.int32, device="cuda")
    group_offs[1:] = padded_group_lens.to(torch.int32).cumsum(0)
    return RunContext(
        group=group,
        args=args,
        rank=group.rank(),
        inp=inp,
        symm=inp.symm,
        M_eff=M_eff,
        experts_per_rank=experts_per_rank,
        padded_group_lens=padded_group_lens,
        group_offs=group_offs,
    )


def _make_fused_call(ctx, lhs, rhs, layout, *, trans_c=False):
    """Build the fused dispatch+GEMM call; stages differ only in operands / layout / trans_c."""
    args = ctx.args
    return lambda: dispatch_grouped_gemm_bf16_flydsl_kernel(
        lhs,
        rhs,
        ctx.group,
        handle=ctx.inp.handle,
        layout=layout,
        BM=args.bm,
        BN=args.bn,
        trans_c=trans_c,
    )


def _make_dispatch_call(ctx, operand):
    """Build the dispatch-only call (the comm baseline for one stage)."""
    return lambda: dispatch_only(operand, ctx.inp.handle, ctx.symm, num_dispatch_cu=ctx.args.num_dispatch_cu)


def _stage_fwd(runner, ctx):
    """forward (NT): N=2I, K=H; returns (metrics, check, xgmi_bytes)."""
    inp, args = ctx.inp, ctx.args
    pool = ctx.symm.dispatch_token_pool
    M_eff, N_fwd, K = ctx.M_eff, 2 * args.inter, args.hidden
    flops = 2.0 * M_eff * N_fwd * K
    # XGMI push bytes per rank = remote rows (dest != rank) x hidden x bf16
    dest_cpu, count_cpu = inp.destination.cpu(), inp.count.cpu()
    remote_rows = int(count_cpu[dest_cpu != ctx.rank].sum().item())
    xgmi_bytes = remote_rows * args.hidden * BF16_BYTES

    spec = StageSpec(
        name="fwd fused (nt)",
        flops=flops,
        dense_dims=(M_eff, N_fwd, K),
        gemm_fn=lambda: grouped_gemm_bf16_only(
            pool,
            inp.W1,
            inp.l1_out,
            inp.tile_to_expert,
            inp.num_tile_blocks,
            BLOCK_M=args.bm,
            BLOCK_N=args.bn,
        ),
        dispatch_fn=_make_dispatch_call(ctx, inp.x),
        fused_fn=_make_fused_call(ctx, inp.x, inp.W1, "nt"),
        ref_fn=lambda: turbo_grouped_gemm(
            pool[:M_eff].contiguous(), inp.W1, ctx.padded_group_lens, trans_b=True
        ),
        acc_slice=slice(0, M_eff),
    )
    metrics, check = runner.run(spec)
    return metrics, check, xgmi_bytes


def _stage_bwd_dgrad(runner, ctx):
    """backward dgrad (NN): dispatch dy + L2 dgrad pool[M,H] @ w2 -> d_swiglu[M,I]; N=I, K=H."""
    inp, args, symm = ctx.inp, ctx.args, ctx.symm
    pool = symm.dispatch_token_pool
    M_eff, N_bwd, K = ctx.M_eff, args.inter, args.hidden
    flops = 2.0 * M_eff * N_bwd * K
    dy = torch.ones(args.num_tokens, args.hidden, device="cuda", dtype=torch.bfloat16)
    d_swiglu = torch.empty(symm.num_max_pool_tokens, N_bwd, device="cuda", dtype=torch.bfloat16)
    # fill the pool with dy once so the gemm-only baseline contracts the right rows
    sync_ranks(ctx.group)
    dispatch_only(dy, inp.handle, symm, num_dispatch_cu=args.num_dispatch_cu)
    sync_ranks(ctx.group)

    # accuracy: fused dispatch+GEMM(NN) vs turbo grouped_gemm. NN -> trans_b=False.
    spec = StageSpec(
        name="bwd dgrad fused (nn)",
        flops=flops,
        dense_dims=(M_eff, N_bwd, K),
        gemm_fn=lambda: grouped_gemm_bf16_only(
            pool,
            inp.W2,
            d_swiglu,
            inp.tile_to_expert,
            inp.num_tile_blocks,
            layout="nn",
            BLOCK_M=args.bm,
            BLOCK_N=args.bn,
        ),
        dispatch_fn=_make_dispatch_call(ctx, dy),
        fused_fn=_make_fused_call(ctx, dy, inp.W2, "nn"),
        ref_fn=lambda: turbo_grouped_gemm(
            pool[:M_eff].contiguous(), inp.W2, ctx.padded_group_lens, trans_b=False
        ),
        acc_slice=slice(0, M_eff),
    )
    return runner.run(spec)


def _stage_bwd_wgrad(runner, ctx):
    """backward wgrad dW1 (TN, variable-K): dW1 = pool(x)^T @ grad_l1; OUT_M=H, OUT_N=2I."""
    inp, args, symm = ctx.inp, ctx.args, ctx.symm
    pool = symm.dispatch_token_pool
    cap = symm.num_max_pool_tokens
    M_out, N_out = args.hidden, 2 * args.inter  # dW1: lhs feature = H (pool x), rhs feature = 2I (grad_l1)
    flops = 2.0 * ctx.M_eff * M_out * N_out
    x_pool = torch.ones(cap, M_out, device="cuda", dtype=torch.bfloat16)
    grad_pool = torch.ones(cap, N_out, device="cuda", dtype=torch.bfloat16)
    # trans_c: dW1 stored transposed [G, N_out, M_out] = [G, 2I, H] = W1-native layout
    dW1 = torch.empty(ctx.experts_per_rank, N_out, M_out, device="cuda", dtype=torch.bfloat16)

    ref_dW1 = torch.zeros_like(dW1)  # empty groups -> 0 (matches the fused padded output)
    offs_cpu = ctx.group_offs.tolist()

    def _ref():
        dispatch_only(x_pool, inp.handle, symm, num_dispatch_cu=args.num_dispatch_cu)
        sync_ranks(ctx.group)
        for expert in range(ctx.experts_per_rank):
            start, end = offs_cpu[expert], offs_cpu[expert + 1]
            if end > start:  # padded rows are zero -> contract to 0 (skip is equivalent)
                ref_dW1[expert] = (grad_pool[start:end].float().T @ pool[start:end].float()).to(dW1.dtype)
        return ref_dW1

    spec = StageSpec(
        name="wgrad dW1 fused (tn)",
        flops=flops,
        # dense roofline of the same total FLOPs (one [H,2I] GEMM contracting M_eff rows)
        dense_dims=(M_out, N_out, ctx.M_eff),
        gemm_fn=lambda: grouped_gemm_variable_k_only(
            x_pool, grad_pool, ctx.group_offs, dW1, BLOCK_M=args.bm, BLOCK_N=args.bn, trans_c=True
        ),
        dispatch_fn=_make_dispatch_call(ctx, x_pool),
        fused_fn=_make_fused_call(ctx, x_pool, grad_pool, "tn", trans_c=True),
        ref_fn=_ref,
        acc_slice=slice(None),
        fused_group=ctx.group,  # wgrad TN scoreboard needs the per-iter barrier discipline
    )
    return runner.run(spec)


def profile_dispatch_grouped_gemm(group, args):
    ctx = _make_context(group, args)

    # synced bracket isolates accuracy probes from timing; scoreboard is never-reset parity
    def _synced(run_fn):
        sync_ranks(group)
        out = run_fn()
        sync_ranks(group)
        return out

    group_m_cands = (1, 4, 8) if args.autotune else (args.dense_group_m,)
    runner = StageRunner(group, args, _synced, group_m_cands)
    checks = {}
    try:
        fwd_metrics, checks["fwd"], xgmi_bytes = _stage_fwd(runner, ctx)
        bwd_metrics, checks["bwd"] = _stage_bwd_dgrad(runner, ctx)
        wgrad_metrics, checks["wgrad"] = _stage_bwd_wgrad(runner, ctx)
    finally:
        ctx.symm.destroy()  # always free symmetric buffers
    # raw per-rank timings + work; rank 0 aggregates across ranks (bottleneck = max latency)
    return {
        "stages": {"fwd": fwd_metrics, "bwd": bwd_metrics, "wgrad": wgrad_metrics},
        "xgmi_bytes": xgmi_bytes,
        "checks": checks,
    }


# Skip: moe_intermediate_size not a multiple of GEMM tile (BN=256) -> autotune fails. TODO(mega): add tail N tile.
UNSUPPORTED = {"DeepSeek-V2-Lite", "MoE-1T"}


def _report_case(platform, gpu_name, world, args, case_name, per_rank):
    """rank-0: aggregate one case's per-rank results -> (rich_row, csv_row); prints the detail block."""
    # distributed bottleneck = slowest rank (max latency); work ~ uniform (mean)
    xgmi = statistics.mean([res["xgmi_bytes"] for res in per_rank])
    rank0_checks = per_rank[0]["checks"]

    agg_fwd = aggregate_stage_metrics(per_rank, "fwd", xgmi)
    agg_bwd = aggregate_stage_metrics(per_rank, "bwd", xgmi)
    agg_wgrad = aggregate_stage_metrics(per_rank, "wgrad", xgmi)

    print_header("dispatch", gpu_name, world, args, case=case_name)
    print_stage(agg_fwd, comm_label="dispatch_only", comm_unit="GB/s (XGMI, nodeup)", comm_tag="disp")
    print_stage(
        agg_bwd,
        comm_label="dispatch_only",
        comm_unit="GB/s (XGMI, nodeup)",
        comm_tag="disp",
        sub_header="backward dgrad (NN, = dispatch_grouped_0)",
    )
    print_stage(
        agg_wgrad,
        comm_label="dispatch_only",
        comm_unit="GB/s (XGMI, nodeup)",
        comm_tag="disp",
        sub_header="backward wgrad dW1 (TN, = dispatch + variable-K wgrad)",
    )
    rich_row = {
        "Platform": platform,
        "GPU": gpu_name,
        "Case": case_name,
        "EP": world,
        "T": args.num_tokens,
        "H": args.hidden,
        "I": args.inter,
        "E": args.num_experts,
        "K": args.num_topk,
        **stage_columns(
            "",
            agg_fwd,
            rank0_checks.get("fwd"),
            comm_label="dispatch_only",
            comm_short="disp",
            dense_ms=True,
            xgmi=True,
            hidden=True,
        ),
        **stage_columns(
            "bwd ",
            agg_bwd,
            rank0_checks.get("bwd"),
            comm_label="dispatch_only",
            comm_short="disp",
            xgmi=True,
        ),
        **stage_columns(
            "wgrad ",
            agg_wgrad,
            rank0_checks.get("wgrad"),
            comm_label="dispatch_only",
            comm_short="disp",
        ),
    }
    # saved CSV follows the gemm_turbo convention; Backward = dgrad + wgrad
    csv_row = turbo_csv_row(
        platform,
        gpu_name,
        world,
        args,
        case=case_name,
        check=checks_verdict(rank0_checks),
        fwd=agg_fwd,
        bwd_stages=[agg_bwd, agg_wgrad],
    )
    return rich_row, csv_row


def benchmark_dispatch_grouped_gemm(local_rank, world, args):
    master_addr = os.getenv("MASTER_ADDR", "127.0.0.1")
    port = int(os.getenv("MASTER_PORT", "8481"))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        "nccl", init_method=f"tcp://{master_addr}:{port}", world_size=world, rank=local_rank
    )
    torch.set_default_device("cuda")
    torch.cuda.set_device(local_rank)
    group = dist.new_group(list(range(world)))
    rank = dist.get_rank()
    platform, gpu_name = get_platform_info()

    try:
        rich_rows, csv_rows = [], []
        # sweep every MoE model (gemm_turbo-style); unsupported cases are skipped
        for case in gen_moe_test_cases(args.models):
            name = case["Case"]
            if name in UNSUPPORTED:
                if rank == 0:
                    print(f"[skip] {name}: unsupported by mega dispatch (see UNSUPPORTED TODO)")
                continue
            apply_case(args, case)
            sync_ranks(group)
            ok, result = True, None
            try:
                result = profile_dispatch_grouped_gemm(group, args)
            except Exception as e:  # noqa: BLE001  probe: skip cases the kernel can't run
                ok = False
                if rank == 0:
                    print(f"[skip] {name}: {e!r}")
            # collective agreement so every rank skips a failed case together (no hang)
            if not all_ranks_ok(group, ok):
                sync_ranks(group)
                continue
            per_rank = [None] * world  # all_gather_object fills per_rank[i] from rank i
            dist.all_gather_object(per_rank, result, group=group)
            if rank == 0:
                rich_row, csv_row = _report_case(platform, gpu_name, world, args, name, per_rank)
                rich_rows.append(rich_row)
                csv_rows.append(csv_row)
            sync_ranks(group)

        if rank == 0 and csv_rows:
            print("\nFinal Results:")
            print(tabulate(pd.DataFrame(rich_rows), headers="keys", tablefmt="grid", showindex=False))
            out_file = args.output or f"dispatch_grouped_gemm_{datetime.now():%Y%m%d}_{gpu_name}.csv"
            pd.DataFrame(csv_rows).to_csv(out_file, index=False)
            print(f"Results saved to {out_file}")
        sync_ranks(group)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark fused BF16 dispatch + grouped GEMM")
    parser.add_argument("--num-processes", type=int, default=8)
    # H/I/E/K come from each MoE model case (config.gen_moe_test_cases); no CLI knob
    parser.add_argument("--num-tokens", type=int, default=8192)
    parser.add_argument("--bm", type=int, default=256)
    parser.add_argument("--bn", type=int, default=256)
    # 16 CUs already saturate intra-node XGMI dispatch; more just steal CUs from the GEMM
    parser.add_argument("--num-dispatch-cu", type=int, default=16)
    # GROUP_M for the dense roofline reference; default = best, autotune sweeps {1,4,8}.
    parser.add_argument("--dense-group-m", type=int, default=4)
    parser.add_argument(
        "--autotune",
        action="store_true",
        help="sweep dense GROUP_M {1,4,8}; default off uses --dense-group-m (best)",
    )
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--output", "-o", type=str, default=None)
    # restrict the sweep to these MoE model names (default = all)
    parser.add_argument("--models", nargs="+", default=None)
    args = parser.parse_args()
    torch.multiprocessing.spawn(
        benchmark_dispatch_grouped_gemm, args=(args.num_processes, args), nprocs=args.num_processes
    )
