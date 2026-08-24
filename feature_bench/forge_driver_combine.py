###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""forge-loop measurement driver for grouped_gemm_combine (EP8, BF16).

Two cases, one per layout the operator is used with:
    combine_nt   forward L2         act @ w2   -> combine + topk reduce, gated
    combine_nn   backward L1 dgrad  g  @ w1    -> combine + gate-grad scatter

The combine operand is a real activation: the driver runs one dispatch L1 pass
plus SwiGLU to produce it, exactly as the production pipeline does. Only the
combine kernel is timed.

Correctness defers to benchmark/ops/training/bench_mega_moe.py's in-process
reference; nothing has to be recorded before a campaign starts. See
forge_harness's docstring for what that replaced and why.

Usage:
    python forge_driver_combine.py                                 # SNR
    python forge_driver_combine.py --warmup 10 --iters 30 --bench-mode
    python forge_driver_combine.py --profile-run
"""

from __future__ import annotations

import argparse
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))

import forge_harness as fh  # noqa: E402

fh.configure_env()

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402

import primus_turbo.pytorch  # noqa: E402,F401
from primus_turbo.flydsl.mega import (  # noqa: E402
    dispatch_grouped_gemm_bf16_flydsl_kernel,
    grouped_gemm_combine_bf16_flydsl_kernel,
    swiglu_flydsl_kernel,
)
from primus_turbo.flydsl.mega.symm_buffer import BLOCK_M as POOL_BLOCK_M  # noqa: E402
from primus_turbo.flydsl.mega.symm_buffer import (  # noqa: E402
    get_symm_buffer_for_mega_moe,
)

LAYOUTS = ("nt", "nn")
CASES = {layout: f"combine_{layout}" for layout in LAYOUTS}
# Correctness comes from benchmark/ops/training/bench_mega_moe.py; this maps its
# stage keys onto the case ids this driver reports.
REF_STAGES = (("grouped_gemm_combine", {"fwd": CASES["nt"], "bwd": CASES["nn"]}),)


def build_context(shape, group, rank, world):
    """Symmetric buffer, a real L1+SwiGLU activation, and the per-layout operands."""
    tokens = shape["num_tokens"]
    hidden, inter = shape["hidden"], shape["inter"]
    experts, topk = shape["num_experts"], shape["num_topk"]
    experts_per_rank = experts // world

    symm = get_symm_buffer_for_mega_moe(
        group,
        num_experts=experts,
        num_max_tokens_per_rank=tokens,
        num_topk=topk,
        hidden=hidden,
        intermediate_hidden=inter,
    )

    fh.seed_rank(rank)
    x = torch.randn(tokens, hidden, device="cuda", dtype=torch.bfloat16)
    w1 = torch.randn(experts_per_rank, 2 * inter, hidden, device="cuda", dtype=torch.bfloat16)
    w1.mul_(1.0 / math.sqrt(hidden))
    w2 = torch.randn(experts_per_rank, hidden, inter, device="cuda", dtype=torch.bfloat16)
    w2.mul_(1.0 / math.sqrt(inter))
    topk_idx, topk_weight = fh.deepseek_routing(tokens, experts, topk)
    topk_indices_flat = topk_idx.contiguous().view(-1)
    topk_weights_flat = topk_weight.to(torch.float32).contiguous().view(-1)

    # One real L1 pass builds both the handle and the activation combine consumes.
    dist.barrier(group)
    l1_out, _, _, handle = dispatch_grouped_gemm_bf16_flydsl_kernel(
        x, w1, group, handle=None, topk_idx=topk_idx, topk_weights=topk_weight, layout="nt"
    )
    num_tile_blocks, *_tables = handle
    act = swiglu_flydsl_kernel(l1_out, num_tile_blocks=num_tile_blocks)
    del l1_out
    torch.cuda.synchronize()

    # nn is the backward L1 dgrad: no routing weights, but it scatters the gate
    # grad. Both operands are indexed by pool row, so both must be keyed off the
    # row's source token -- see pool_keyed_operand for why a randn is unusable.
    grad_l1 = fh.pool_keyed_operand(
        symm, 2 * inter, scale=1.0 / math.sqrt(hidden), tokens=tokens, topk=topk, world=world
    )
    grad_gate = fh.pool_keyed_operand(
        symm, 1, scale=1.0, tokens=tokens, topk=topk, world=world, dtype=torch.float32
    ).view(-1)

    operands = {
        "nt": (act, w2, topk_weights_flat, None),
        "nn": (grad_l1, w1, None, grad_gate),
    }
    return symm, handle, operands, topk_indices_flat


def scrub(symm):
    """Reset the staging buffers combine accumulates into, before each replay.

    Combine reduces into the symmetric combine buffer and only touches the slots
    its routing selects, so leftovers from the previous run would leak into the
    result. The epoch flags stay untouched -- they are parity counters.
    """

    def _scrub(out):
        out.zero_()
        symm.combine_token_buffer.zero_()
        symm.l2_token_buffer.zero_()

    return _scrub


def make_step(layout, operands, handle, topk_indices_flat, bn):
    lhs, rhs, weights, gate = operands[layout]

    def step():
        return grouped_gemm_combine_bf16_flydsl_kernel(
            lhs,
            rhs,
            handle,
            topk_indices=topk_indices_flat,
            topk_weights=weights,
            grad_gate=gate,
            layout=layout,
            BM=POOL_BLOCK_M,
            BN=bn,
        )[0]

    return step


def run_correctness(args, group, rank, world):
    """Gate both layouts against bench_mega_moe.py's in-process reference.

    Not the golden projection this driver used to write: it does not reproduce
    on this build. See forge_harness's docstring.
    """
    snrs = fh.reference_snrs(
        group, rank, REF_STAGES, model=args.ref_model, tokens=args.ref_tokens, iters=args.ref_iters
    )
    fh.report_snr(snrs, rank, group)
    return 0


def run_bench(args, group, rank, world):
    _symm, handle, operands, topk_indices_flat = build_context(fh.BENCH_SHAPE, group, rank, world)
    for layout in LAYOUTS:
        step = make_step(layout, operands, handle, topk_indices_flat, args.bn)
        rounds = []
        for _ in range(max(args.repeat, 1)):
            result = fh.cuda_graph_bench(
                step,
                warmup=args.warmup,
                iters=args.iters,
                group=group,
                # No scrub here. Not for timing (the scrub sits outside the
                # timed window) but for the bench stage's wall-clock budget: a
                # 7.6 GB pool memset before all 90 replays is minutes of it.
                dirty=None,
                verify=lambda out: bool(torch.isfinite(out.reshape(-1)[:4096].float()).all()),
            )
            rounds.append(result["times_ms"])
            del result
        fh.report_case(CASES[layout], rounds, rank, group)
    return 0


def run_profile(args, group, rank, world):
    _symm, handle, operands, topk_indices_flat = build_context(fh.BENCH_SHAPE, group, rank, world)
    step = make_step("nt", operands, handle, topk_indices_flat, args.bn)
    for _ in range(3):
        step()
        torch.cuda.synchronize()
        dist.barrier(group)
    for _ in range(3):
        step()
    torch.cuda.synchronize()
    dist.barrier(group)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    fh.add_common_args(parser)
    args = parser.parse_args()

    fh.relaunch_under_torchrun()
    group, rank, world = fh.init_dist()
    if world != fh.WORLD:
        raise ValueError(f"this driver requires EP{fh.WORLD}, got EP{world}")
    try:
        if args.profile_run:
            return run_profile(args, group, rank, world)
        if args.bench_mode:
            return run_bench(args, group, rank, world)
        return run_correctness(args, group, rank, world)
    finally:
        fh.shutdown_dist()


if __name__ == "__main__":
    sys.exit(main())
