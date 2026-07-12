###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Benchmark the fused BF16 grouped GEMM + combine mega kernel (EP, intra-node)."""

import argparse
import os
import statistics
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable

import pandas as pd
import torch
import torch.distributed as dist
from config import gen_moe_test_cases, get_platform_info
from tabulate import tabulate

# repo root (primus_turbo) on the path so the shared mega_utils + kernels import
_HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..")))

from mega_utils import (  # noqa: E402
    BF16_BYTES,
    aggregate_stage_metrics,
    apply_case,
    bench,
    check_accuracy,
    checks_verdict,
    combine_only,
    dense_gemm_peak_ms,
    generate_input,
    grouped_gemm_bf16_only,
    print_header,
    print_stage,
    stage_columns,
    sync_ranks,
    topk_reduce_only,
    turbo_csv_row,
)

import primus_turbo.pytorch  # noqa: E402,F401  # load full pkg first to break the circular import
from primus_turbo.flydsl.mega import (  # noqa: E402
    grouped_gemm_combine_bf16_flydsl_kernel,
)


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
    """Declarative spec for one benchmarked stage; fused_fn is reused for timing + accuracy check."""

    name: str
    flops: float
    dense_dims: tuple[int, int, int]
    gemm_fn: Callable
    comm_fn: Callable
    fused_fn: Callable
    ref_fn: Callable


@dataclass
class RunContext:
    """Shared per-run state built by _make_context: config, input/symm buffers, derived geometry + tables."""

    # collective + CLI config
    group: Any
    args: Any
    rank: int
    # input namespace + symmetric buffers (own act / handle / l2 buffers, read off inp/symm)
    inp: Any
    symm: Any
    # derived geometry + tables (computed once, shared by both stages)
    M_eff: int  # total padded pool rows this rank GEMMs
    xgmi_bytes: int  # combine push bytes per rank (same fwd/bwd, H-wide)
    num_tokens: int
    topk_idx_flat: Any  # int32 [T*K], drives the per-token reduce
    topk_w_flat: Any  # f32 [T*K], forward routing weights
    reduce_ready: Any  # standalone reduce ready flags (0 == ready)


class StageRunner:
    """Binds per-run context so each stage passes only its own knobs; run does the shared template."""

    def __init__(self, group, args, synced_fn, group_m_cands):
        self.group = group
        self.args = args
        self.synced_fn = synced_fn
        self.group_m_cands = group_m_cands

    def run(self, spec):
        """Time gemm / dense roofline / combine_only / fused, then gate accuracy under a synced bracket."""
        args = self.args
        t_gemm = bench(spec.gemm_fn, iters=args.iters)
        t_dense, dense_gm = dense_gemm_peak_ms(
            *spec.dense_dims, args.bm, args.bn, args.iters, group_m_cands=self.group_m_cands
        )
        t_comb = bench(spec.comm_fn, group=self.group, iters=args.iters)
        t_fused = bench(spec.fused_fn, group=self.group, iters=args.iters)
        metrics = StageMetrics(
            gemm_ms=t_gemm,
            dense_ms=t_dense,
            dense_gm=dense_gm,
            comm_ms=t_comb,
            fused_ms=t_fused,
            flops=spec.flops,
        )
        # accuracy: fused vs ref over the SAME symm state (both under a synced bracket).
        # ref_fn is the decoupled combine path (same handle/buffers), so any never-reset
        # scoreboard drift cancels -> the isolated fused_fn output is directly comparable.
        ref_out = self.synced_fn(spec.ref_fn)
        out = self.synced_fn(spec.fused_fn)[0]
        check = check_accuracy(self.group, spec.name, out, ref_out)
        return metrics, check


def _make_context(group, args):
    """Build input + symm buffers and precompute shared geometry + topk tables (weights built in generate_input)."""
    inp = generate_input(
        group,
        kind="combine",
        T=args.num_tokens,
        H=args.hidden,
        I=args.inter,
        E=args.num_experts,
        K=args.num_topk,
        BLOCK_M=args.bm,
        BLOCK_N=args.bn,
    )
    symm = inp.symm
    rank = group.rank()
    real_tiles = int(inp.num_tile_blocks.item())
    M_eff = real_tiles * args.bm
    # combine push bytes per rank = remote rows (origin_rank != rank, valid) x H x bf16
    origin = symm.pool_src_rank
    remote_rows = int(((origin != rank) & (origin >= 0)).sum().item())
    xgmi_bytes = remote_rows * args.hidden * BF16_BYTES
    topk_idx_flat = inp.topk_idx.to(torch.int32).contiguous().view(-1)
    topk_w_flat = inp.topk_weight.to(torch.float32).contiguous().view(-1)
    reduce_ready = torch.zeros(int(symm.num_combine_slots), dtype=torch.int32, device="cuda")
    return RunContext(
        group=group,
        args=args,
        rank=rank,
        inp=inp,
        symm=symm,
        M_eff=M_eff,
        xgmi_bytes=xgmi_bytes,
        num_tokens=int(symm.num_tokens),
        topk_idx_flat=topk_idx_flat,
        topk_w_flat=topk_w_flat,
        reduce_ready=reduce_ready,
    )


def _make_fused_call(ctx, lhs, rhs, *, layout="nt", topk_weights):
    """Build the fused GEMM + combine PUSH + topk-reduce call (3-role); stages differ in operands / layout / weights."""
    args = ctx.args
    return lambda: grouped_gemm_combine_bf16_flydsl_kernel(
        lhs,
        rhs,
        ctx.inp.handle,
        topk_indices=ctx.topk_idx_flat,
        topk_weights=topk_weights,
        layout=layout,
        BM=args.bm,
        BN=args.bn,
    )


def _make_combine_call(ctx, bm, cu):
    """Build the combine-only call (the comm baseline) at a given CU count."""
    return lambda: combine_only(ctx.group, handle=ctx.inp.handle, BLOCK_M=bm, num_combine_cu=cu)


def _stage_fwd(runner, ctx):
    """forward (e2e step 4, NT): L2 GEMM N=H, K=I; 3-role fused -> weighted y[T,H]."""
    inp, args, symm = ctx.inp, ctx.args, ctx.symm
    N, K = args.hidden, args.inter
    flops = 2.0 * ctx.M_eff * N * K
    ref_y = torch.empty(ctx.num_tokens, args.hidden, device="cuda", dtype=torch.bfloat16)

    # reference: decoupled gemm_only(nt) + combine_only + weighted topk_reduce, over the
    # SAME handle/buffers as fused_fn -> never-reset scoreboard drift cancels between them.
    def _ref_fwd():
        grouped_gemm_bf16_only(
            inp.act,
            inp.W2,
            symm.l2_token_buffer,
            inp.tile_to_expert,
            inp.num_tile_blocks,
            BLOCK_M=args.bm,
            BLOCK_N=args.bn,
            GROUP_M=8,
        )
        sync_ranks(ctx.group)
        combine_only(ctx.group, handle=ctx.inp.handle, BLOCK_M=args.bm, num_combine_cu=args.num_combine_cu)
        sync_ranks(ctx.group)
        ctx.reduce_ready.zero_()  # 0 == ready (reduce_only does not spin)
        topk_reduce_only(
            ref_y,
            symm.combine_token_buffer,
            ctx.reduce_ready,
            ctx.topk_idx_flat,
            symm.num_tokens_per_rank,
            int(symm.num_combine_slots),
            topk=int(symm.num_topk),
            num_experts=int(symm.num_experts),
            rank=ctx.rank,
            topk_weights=ctx.topk_w_flat,  # weighted (forward routing weights)
        )
        return ref_y

    # L2 has small K=I -> GROUP_M=8 reuses the per-expert weight across more M-tiles (best here)
    spec = StageSpec(
        name="fwd fused (nt)",
        flops=flops,
        dense_dims=(ctx.M_eff, N, K),
        gemm_fn=lambda: grouped_gemm_bf16_only(
            inp.act,
            inp.W2,
            symm.l2_token_buffer,
            inp.tile_to_expert,
            inp.num_tile_blocks,
            BLOCK_M=args.bm,
            BLOCK_N=args.bn,
            GROUP_M=8,
        ),
        comm_fn=_make_combine_call(ctx, args.bm, args.num_combine_cu),
        fused_fn=_make_fused_call(ctx, inp.act, inp.W2, topk_weights=ctx.topk_w_flat),
        ref_fn=_ref_fwd,
    )
    return runner.run(spec)


def _stage_bwd(runner, ctx):
    """backward (e2e step 3, NN): L1 dgrad grad_l1[M,2I] @ w1 -> grad_pool[M,H]; 3-role fused -> dx[T,H]."""
    inp, args, symm = ctx.inp, ctx.args, ctx.symm
    N, K = args.hidden, 2 * args.inter  # weight [G, K=2I, N=H]
    flops = 2.0 * ctx.M_eff * N * K
    grad_l1 = torch.randn(symm.num_max_pool_tokens, K, device="cuda", dtype=torch.bfloat16) / 8
    ref_dx = torch.empty(ctx.num_tokens, args.hidden, device="cuda", dtype=torch.bfloat16)

    # reference: decoupled gemm_only(nn) + combine_only + unweighted topk_reduce (weight rides grad_l1)
    def _ref_bwd():
        grouped_gemm_bf16_only(
            grad_l1,
            inp.W1,
            symm.l2_token_buffer,
            inp.tile_to_expert,
            inp.num_tile_blocks,
            layout="nn",
            BLOCK_M=args.bm,
            BLOCK_N=args.bn,
            GROUP_M=8,
        )
        sync_ranks(ctx.group)
        combine_only(ctx.group, handle=ctx.inp.handle, BLOCK_M=args.bm, num_combine_cu=args.bwd_combine_cu)
        sync_ranks(ctx.group)
        ctx.reduce_ready.zero_()  # 0 == ready (reduce_only does not spin)
        topk_reduce_only(
            ref_dx,
            symm.combine_token_buffer,
            ctx.reduce_ready,
            ctx.topk_idx_flat,
            symm.num_tokens_per_rank,
            int(symm.num_combine_slots),
            topk=int(symm.num_topk),
            num_experts=int(symm.num_experts),
            rank=ctx.rank,
            topk_weights=None,  # unweighted (weight rides grad_l1)
        )
        return ref_dx

    # GEMM-bound (K=2I) -> fewer combine CUs (--bwd-combine-cu) leaves more for the GEMM
    spec = StageSpec(
        name="bwd dgrad fused (nn)",
        flops=flops,
        dense_dims=(ctx.M_eff, N, K),
        gemm_fn=lambda: grouped_gemm_bf16_only(
            grad_l1,
            inp.W1,
            symm.l2_token_buffer,
            inp.tile_to_expert,
            inp.num_tile_blocks,
            layout="nn",
            BLOCK_M=args.bm,
            BLOCK_N=args.bn,
            GROUP_M=8,
        ),
        comm_fn=_make_combine_call(ctx, args.bm, args.bwd_combine_cu),
        fused_fn=_make_fused_call(ctx, grad_l1, inp.W1, layout="nn", topk_weights=None),
        ref_fn=_ref_bwd,
    )
    return runner.run(spec)


def profile_grouped_gemm_combine(group, args):
    """Forward = e2e step 4 (NT weighted); backward = e2e step 3 (NN unweighted). Both report the 3-role fused kernel."""
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
        fwd_metrics, checks["fwd"] = _stage_fwd(runner, ctx)
        bwd_metrics, checks["bwd"] = _stage_bwd(runner, ctx)
        xgmi_bytes = ctx.xgmi_bytes
    finally:
        ctx.symm.destroy()  # always free the symmetric buffer
    # raw per-rank timings + work; rank 0 aggregates across ranks (bottleneck = max latency)
    return {
        "stages": {"fwd": fwd_metrics, "bwd": bwd_metrics},
        "xgmi_bytes": xgmi_bytes,
        "checks": checks,
    }


# Fast-skip flydsl tiling gaps: moe_intermediate_size not a multiple of GEMM tile (BN=256) -> autotune fails. TODO(mega): add tail N tile.
UNSUPPORTED = {"DeepSeek-V2-Lite", "MoE-1T"}


def _report_case(platform, gpu_name, world, args, case_name, per_rank):
    """rank-0: aggregate one case's per-rank results -> (rich_row, csv_row); prints the detail block."""
    rank0_checks = per_rank[0]["checks"]
    # distributed bottleneck = slowest rank (max latency); work ~ uniform (mean)
    xgmi = statistics.mean([res["xgmi_bytes"] for res in per_rank])
    agg_fwd = aggregate_stage_metrics(per_rank, "fwd", xgmi)
    agg_bwd = aggregate_stage_metrics(per_rank, "bwd", xgmi)

    print_header("combine", gpu_name, world, args, case=case_name)
    # fused = 3-role (GEMM + combine PUSH + weighted topk reduce -> y) == e2e forward step 4
    print_stage(agg_fwd, comm_label="combine_only", comm_unit="GB/s (XGMI)", comm_tag="comb")
    print_stage(
        agg_bwd,
        comm_label="combine_only",
        comm_unit="GB/s (XGMI)",
        comm_tag="comb",
        sub_header="backward dgrad (NN, = mega_moe_fused STEP 3)",
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
            comm_label="combine_only",
            comm_short="comb",
            dense_ms=True,
            xgmi=True,
            hidden=True,
        ),
        **stage_columns(
            "bwd ",
            agg_bwd,
            rank0_checks.get("bwd"),
            comm_label="combine_only",
            comm_short="comb",
            xgmi=True,
        ),
    }
    # saved CSV follows the gemm_turbo convention; Backward = dgrad (NN)
    csv_row = turbo_csv_row(
        platform,
        gpu_name,
        world,
        args,
        case=case_name,
        check=checks_verdict(rank0_checks),
        fwd=agg_fwd,
        bwd_stages=[agg_bwd],
    )
    return rich_row, csv_row


def benchmark_combine(local_rank, world, args):
    """Run ONE model case (args geometry already applied). Runs in its own spawn so a
    crash/deadlock is isolated to this model and never wedges the rest of the sweep.
    rank 0 appends its row to the shared CSV (header written for the first row)."""
    master_addr = os.getenv("MASTER_ADDR", "127.0.0.1")
    port = int(os.getenv("MASTER_PORT", "8483"))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        "nccl", init_method=f"tcp://{master_addr}:{port}", world_size=world, rank=local_rank
    )
    torch.set_default_device("cuda")
    torch.cuda.set_device(local_rank)
    group = dist.new_group(list(range(world)))
    rank = dist.get_rank()
    platform, gpu_name = get_platform_info()
    name = args.case_name

    try:
        result = profile_grouped_gemm_combine(group, args)
        per_rank = [None] * world  # all_gather_object fills per_rank[i] from rank i
        dist.all_gather_object(per_rank, result, group=group)
        if rank == 0:
            _, csv_row = _report_case(platform, gpu_name, world, args, name, per_rank)
            out_file = args.output
            header = (not os.path.exists(out_file)) or os.path.getsize(out_file) == 0
            pd.DataFrame([csv_row]).to_csv(out_file, mode="a", header=header, index=False)
            print(f"[{name}] appended to {out_file}")
        sync_ranks(group)
    finally:
        dist.destroy_process_group()


def _join_with_timeout(ctx, timeout_s):
    """Block until every child of ``ctx`` exits; terminate + raise on timeout (deadlock guard).

    ctx.join(timeout) returns True when all done and re-raises on a child error; a
    hung (deadlocked) case never returns, so we cap total wall-clock and kill it."""
    deadline = time.monotonic() + timeout_s
    while True:
        if ctx.join(timeout=5):  # raises ProcessExited/RaisedException on child failure
            return
        if time.monotonic() >= deadline:
            for p in ctx.processes:
                if p.is_alive():
                    p.terminate()
            for p in ctx.processes:
                p.join(10)
            raise TimeoutError(f"case exceeded {timeout_s}s (killed)")


def run_sweep(args):
    """Drive the per-model sweep: one fresh 8-proc spawn per MoE model so a crash or
    deadlock is isolated. Each model appends its row to a single shared CSV."""
    cases = gen_moe_test_cases(args.models)
    # resolve the output path once so per-model appends land in the same file
    out_file = args.output or f"grouped_gemm_combine_{datetime.now():%Y%m%d}_{get_platform_info()[1]}.csv"
    args.output = out_file
    if os.path.exists(out_file):  # fresh file so appends don't mix with a stale run
        os.remove(out_file)

    base_port = int(os.getenv("MASTER_PORT", "8483"))
    nprocs = args.num_processes
    for idx, case in enumerate(cases):
        name = case["Case"]
        if name in UNSUPPORTED:
            print(f"[skip] {name}: unsupported by mega combine (see UNSUPPORTED TODO)")
            continue
        apply_case(args, case)
        args.case_name = name
        os.environ["MASTER_PORT"] = str(base_port + idx)  # unique port avoids TIME_WAIT reuse
        print(f"\n{'#' * 72}\n# combine sweep [{idx + 1}/{len(cases)}]: {name}\n{'#' * 72}")
        try:
            ctx = torch.multiprocessing.spawn(
                benchmark_combine, args=(nprocs, args), nprocs=nprocs, join=False
            )
            _join_with_timeout(ctx, args.case_timeout)
        except Exception as e:  # noqa: BLE001  isolate a per-model crash/deadlock; keep sweeping
            print(f"[skip] {name}: subprocess failed: {e!r}")

    if os.path.exists(out_file) and os.path.getsize(out_file) > 0:
        df = pd.read_csv(out_file)
        print("\nFinal Results:")
        print(tabulate(df, headers="keys", tablefmt="grid", showindex=False))
        print(f"Results saved to {out_file}")
    else:
        print("No supported models produced results.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark fused BF16 grouped GEMM + combine")
    parser.add_argument("--num-processes", type=int, default=8)
    # H/I/E/K come from each MoE model case (config.gen_moe_test_cases); no CLI knob
    parser.add_argument("--num-tokens", type=int, default=8192)
    parser.add_argument("--bm", type=int, default=256)
    parser.add_argument("--bn", type=int, default=256)
    # forward combine_only CU count; 64 is the near-optimum for the fused path
    parser.add_argument("--num-combine-cu", type=int, default=64)
    # GROUP_M for the dense roofline reference; default = best, autotune sweeps {1,4,8}
    parser.add_argument("--dense-group-m", type=int, default=4)
    parser.add_argument(
        "--autotune",
        action="store_true",
        help="sweep dense GROUP_M {1,4,8}; default off uses --dense-group-m (best)",
    )
    # bwd dgrad GEMM-bound; must be multiple of 8 (XCD count) or ~25% XGMI BW loss
    parser.add_argument("--bwd-combine-cu", type=int, default=16)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--output", "-o", type=str, default=None)
    # per-model wall-clock guard; a model exceeding it is terminated and skipped (deadlock safety)
    parser.add_argument("--case-timeout", type=int, default=900)
    # restrict the sweep to these MoE model names (default = all)
    parser.add_argument("--models", nargs="+", default=None)
    args = parser.parse_args()
    run_sweep(args)
