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
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable

import pandas as pd
import torch
import torch.distributed as dist
from config import get_platform_info
from tabulate import tabulate

# repo root (primus_turbo) on the path so the shared mega_utils + kernels import
_HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..")))

# Megatron-LM lives under the sibling Primus repo, not on the default path
_MEGATRON_LM = os.path.abspath(os.path.join(_HERE, "..", "..", "..", "Primus", "third_party", "Megatron-LM"))
if _MEGATRON_LM not in sys.path:
    sys.path.insert(0, _MEGATRON_LM)

from mega_utils import (  # noqa: E402
    BF16_BYTES,
    aggregate_stage_metrics,
    bench,
    check_accuracy,
    combine_only,
    dense_gemm_peak_ms,
    generate_input,
    grouped_gemm_bf16_only,
    print_header,
    print_stage,
    stage_columns,
    sync_ranks,
    topk_reduce_only,
)
from megatron.core.fusions.fused_bias_swiglu import (  # noqa: E402
    weighted_bias_swiglu_impl,
)

import primus_turbo.pytorch  # noqa: E402,F401  # load full pkg first to break the circular import
import primus_turbo.pytorch as turbo  # noqa: E402
from primus_turbo.flydsl.mega.grouped_gemm_combine_bf16_kernel import (  # noqa: E402
    grouped_gemm_combine_bf16,
)
from primus_turbo.pytorch.ops import grouped_gemm as _turbo_gg  # noqa: E402


def make_turbo_reference(group, *, num_experts, num_topk, hidden, inter):
    """Turbo (DeepEP) EP-MoE full forward = ground-truth y; dispatcher built once, W1/W2 passed per call."""
    dispatcher = turbo.modules.DeepEPTokenDispatcher(
        num_experts=num_experts,
        router_topk=num_topk,
        ep_group=group,
        permute_fusion=True,
        deepep_num_use_cu=80,
    )

    def turbo_reference(x, topk_idx, gate_logits, W1, W2):
        permuted_hidden, tokens_per_expert, permuted_probs = dispatcher.token_dispatch(
            x, gate_logits, indices=topk_idx
        )
        group_lens = tokens_per_expert.to(device=x.device, dtype=torch.int64)
        fc1_out = _turbo_gg(permuted_hidden, W1, group_lens, trans_b=True)
        act = weighted_bias_swiglu_impl(fc1_out, None, permuted_probs.unsqueeze(-1)).to(x.dtype)
        fc2_out = _turbo_gg(act, W2, group_lens, trans_b=True)
        return dispatcher.token_combine(fc2_out)

    return turbo_reference


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
    gate_logits: Any  # [T, E] probs for the turbo reference
    turbo_reference: Any


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
        # accuracy: fused vs ref over a clean synced bracket
        ref_out = self.synced_fn(spec.ref_fn)
        out = self.synced_fn(spec.fused_fn)[0]
        check = check_accuracy(self.group, spec.name, out, ref_out)
        return metrics, check


def _make_context(group, args, turbo_reference):
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
    real_tiles = int(symm.meta_scalars[1].item())
    M_eff = real_tiles * args.bm
    # combine push bytes per rank = remote rows (origin_rank != rank, valid) x H x bf16
    origin = symm.pool_src_rank
    remote_rows = int(((origin != rank) & (origin >= 0)).sum().item())
    xgmi_bytes = remote_rows * args.hidden * BF16_BYTES
    topk_idx_flat = inp.topk_idx.to(torch.int32).contiguous().view(-1)
    topk_w_flat = inp.topk_weight.to(torch.float32).contiguous().view(-1)
    reduce_ready = torch.zeros(int(symm.num_combine_slots), dtype=torch.int32, device="cuda")
    gate_logits = torch.zeros(args.num_tokens, args.num_experts, dtype=torch.float32, device="cuda")
    gate_logits.scatter_(1, inp.topk_idx, inp.topk_weight)
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
        gate_logits=gate_logits,
        turbo_reference=turbo_reference,
    )


def _make_fused_call(ctx, lhs, rhs, *, layout="nt", topk_weights):
    """Build the fused GEMM + combine PUSH + topk-reduce call (3-role); stages differ in operands / layout / weights."""
    args = ctx.args
    return lambda: grouped_gemm_combine_bf16(
        lhs,
        rhs,
        ctx.inp.handle,
        topk_indices=ctx.topk_idx_flat,
        topk_weights=topk_weights,
        layout=layout,
        BM=args.bm,
        BN=args.bn,
    )


def _make_combine_call(group, bm, cu):
    """Build the combine-only call (the comm baseline) at a given CU count."""
    return lambda: combine_only(group, BLOCK_M=bm, num_combine_cu=cu)


def _stage_fwd(runner, ctx):
    """forward (e2e step 4, NT): L2 GEMM N=H, K=I; 3-role fused -> weighted y[T,H]."""
    inp, args, symm = ctx.inp, ctx.args, ctx.symm
    N, K = args.hidden, args.inter
    flops = 2.0 * ctx.M_eff * N * K
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
        comm_fn=_make_combine_call(ctx.group, args.bm, args.num_combine_cu),
        fused_fn=_make_fused_call(ctx, inp.act, inp.W2, topk_weights=ctx.topk_w_flat),
        ref_fn=lambda: ctx.turbo_reference(inp.x, inp.topk_idx, ctx.gate_logits, inp.W1, inp.W2),
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
        combine_only(ctx.group, BLOCK_M=args.bm, num_combine_cu=args.bwd_combine_cu)
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
        name="bwd STEP3 fused (nn)",
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
        comm_fn=_make_combine_call(ctx.group, args.bm, args.bwd_combine_cu),
        fused_fn=_make_fused_call(ctx, grad_l1, inp.W1, layout="nn", topk_weights=None),
        ref_fn=_ref_bwd,
    )
    return runner.run(spec)


def profile_combine(group, args, turbo_reference):
    """Forward = e2e step 4 (NT weighted); backward = e2e step 3 (NN unweighted). Both report the 3-role fused kernel."""
    ctx = _make_context(group, args, turbo_reference)

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


def benchmark_combine(local_rank, world, args):
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

    # turbo (DeepEP) full-forward baseline = the accuracy ground truth; dispatcher built once here
    turbo_reference = make_turbo_reference(
        group,
        num_experts=args.num_experts,
        num_topk=args.num_topk,
        hidden=args.hidden,
        inter=args.inter,
    )
    try:
        result = profile_combine(group, args, turbo_reference)
        # all_gather_object fills per_rank[i] from rank i (already rank-ordered)
        per_rank = [None] * world
        dist.all_gather_object(per_rank, result, group=group)
        if rank == 0:
            rank0_checks = per_rank[0]["checks"]
            # distributed bottleneck = slowest rank (max latency); work ~ uniform (mean)
            xgmi = statistics.mean([res["xgmi_bytes"] for res in per_rank])
            agg_fwd = aggregate_stage_metrics(per_rank, "fwd", xgmi)
            agg_bwd = aggregate_stage_metrics(per_rank, "bwd", xgmi)

            print_header("combine", gpu_name, world, args)
            # fused = 3-role (GEMM + combine PUSH + weighted topk reduce -> y) == e2e forward step 4
            print_stage(agg_fwd, comm_label="combine_only", comm_unit="GB/s (XGMI)", comm_tag="comb")
            print_stage(
                agg_bwd,
                comm_label="combine_only",
                comm_unit="GB/s (XGMI)",
                comm_tag="comb",
                sub_header="backward STEP3 (NN, = mega_moe_fused STEP 3)",
            )
            row = {
                "Platform": platform,
                "GPU": gpu_name,
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
            results = pd.DataFrame([row])
            print("\nFinal Results:")
            print(tabulate(results, headers="keys", tablefmt="grid", showindex=False))
            out_file = args.output or f"grouped_gemm_combine_{datetime.now():%Y%m%d}_{gpu_name}.csv"
            results.to_csv(out_file, index=False)
            print(f"Results saved to {out_file}")
        sync_ranks(group)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark fused BF16 grouped GEMM + combine")
    parser.add_argument("--num-processes", type=int, default=8)
    parser.add_argument("--hidden", type=int, default=7168)  # DeepSeek-V3
    parser.add_argument("--inter", type=int, default=2048)
    parser.add_argument("--num-experts", type=int, default=256)
    parser.add_argument("--num-topk", type=int, default=8)
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
    # backward STEP3 is GEMM-bound (K=2I) -> fewer combine CUs leaves more for the GEMM.
    # MUST be a multiple of 8 (XCD count): non-multiples spread blocks unevenly across
    # the 8 XCDs' XGMI paths -> ~25% BW loss (20->252 GB/s vs 16->339, 24->327).
    parser.add_argument("--bwd-combine-cu", type=int, default=16)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--output", "-o", type=str, default=None)
    args = parser.parse_args()
    torch.multiprocessing.spawn(benchmark_combine, args=(args.num_processes, args), nprocs=args.num_processes)
