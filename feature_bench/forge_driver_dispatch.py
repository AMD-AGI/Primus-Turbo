###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""forge-loop measurement driver for dispatch_grouped_gemm (EP8, BF16).

Three cases, one per layout the operator is used with:
    dispatch_nt   forward L1        x @ w1          -> pool activations
    dispatch_nn   backward L2 dgrad dy @ w2         -> pool grad
    dispatch_tn   backward dW1      pool(x)^T @ g   -> per-expert weight grad

Usage (forge-loop drives these three, the driver launches its own 8 ranks):
    python forge_driver_dispatch.py                                # SNR
    python forge_driver_dispatch.py --warmup 10 --iters 30 --bench-mode
    python forge_driver_dispatch.py --profile-run

One-off, before the campaign starts, on the pristine kernel:
    python forge_driver_dispatch.py --make-golden && git add forge_golden_dispatch.pt
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
    dispatch_prologue_flydsl_kernel,
)
from primus_turbo.flydsl.mega.symm_buffer import BLOCK_M as POOL_BLOCK_M  # noqa: E402
from primus_turbo.flydsl.mega.symm_buffer import (  # noqa: E402
    get_symm_buffer_for_mega_moe,
)

GOLDEN = "dispatch"
LAYOUTS = ("nt", "nn", "tn")
CASES = {layout: f"dispatch_{layout}" for layout in LAYOUTS}


def build_context(shape, group, rank, world):
    """Symmetric buffer + dispatch prologue handle + the per-layout operands."""
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
    pool_rows = int(symm.num_max_pool_tokens)

    fh.seed_rank(rank)
    x = torch.randn(tokens, hidden, device="cuda", dtype=torch.bfloat16)
    topk_idx, topk_weight = fh.deepseek_routing(tokens, experts, topk)

    def scaled(shape_, scale):
        return torch.randn(shape_, device="cuda", dtype=torch.bfloat16).mul_(scale)

    operands = {
        # nt: x @ w1, w1 is [G, 2I, H]
        "nt": (x, scaled((experts_per_rank, 2 * inter, hidden), 1.0 / math.sqrt(hidden))),
        # nn: dgrad, w2 is [G, H, I]
        "nn": (
            scaled((tokens, hidden), 1.0),
            scaled((experts_per_rank, hidden, inter), 1.0 / math.sqrt(inter)),
        ),
    }

    num_tile_blocks, grouped_meta, dispatch_meta, combine_meta = dispatch_prologue_flydsl_kernel(
        topk_idx,
        topk_weight,
        sym_buffer=symm.get_sym_buffer(),
        num_tokens=tokens,
        num_topk=topk,
        num_experts=experts,
        num_ranks=world,
        rank=rank,
        experts_per_rank=experts_per_rank,
        block_m=POOL_BLOCK_M,
        num_max_pool_tokens=pool_rows,
        hidden=hidden,
        num_max_tokens_per_rank=tokens,
    )
    # Mirrors the dispatch launcher: the prologue leaves pool_src_slot unset.
    recv_dst_rank, recv_start_row, recv_count, _, dedup_key_row = combine_meta
    combine_meta = (recv_dst_rank, recv_start_row, recv_count, symm.pool_src_slot.clone(), dedup_key_row)
    handle = (num_tile_blocks, grouped_meta, dispatch_meta, combine_meta)
    torch.cuda.synchronize()

    # tn is wgrad: its rhs is indexed by pool row, so it can only be built after
    # the prologue has recorded where each row came from. See pool_keyed_operand.
    operands["tn"] = (
        x,
        fh.pool_keyed_operand(
            symm, 2 * inter, scale=1.0 / math.sqrt(hidden), tokens=tokens, topk=topk, world=world
        ),
    )
    return symm, handle, operands


def make_step(layout, operands, group, handle, bn):
    act, rhs = operands[layout]

    def step():
        return dispatch_grouped_gemm_bf16_flydsl_kernel(
            act,
            rhs,
            group,
            handle=handle,
            layout=layout,
            BM=POOL_BLOCK_M,
            BN=bn,
            # dW1 is stored W1-native [G, 2I, H]; the untransposed store is slower
            trans_c=(layout == "tn"),
        )[0]

    return step


def scrub(symm):
    """Reset everything the kernel only partially overwrites, before each replay.

    The output is sized to the whole pool but only the active rows are written,
    and the pool itself keeps the previous run's residue -- both make the result
    depend on history unless they are zeroed. The epoch flags are deliberately
    left alone: they are parity counters, not scratch. weight_recv_buf is left
    alone too -- the prologue fills it and nothing in this kernel reads it back,
    so scrubbing it is pure cost today and destroys an input the day something
    does.
    """

    def _scrub(out):
        out.zero_()
        symm.dispatch_token_pool.zero_()

    return _scrub


def run_correctness(args, group, rank, world):
    """Compare every layout against the golden projection (or write it)."""
    symm, handle, operands = build_context(fh.CORRECT_SHAPE, group, rank, world)
    golden = None if args.make_golden else fh.load_golden(GOLDEN, rank, CASES.values())

    projections, snrs = {}, {}
    for layout in LAYOUTS:
        case = CASES[layout]
        step = make_step(layout, operands, group, handle, args.bn)
        # Graph-captured, same as bench: the output is then a fixed buffer we can
        # scrub, so the pool rows the kernel never writes read as a stable zero
        # instead of whatever the last allocation left there.
        result = fh.cuda_graph_bench(
            step,
            warmup=2,
            iters=2,
            group=group,
            dirty=scrub(symm),
            # Whole tensor, not a prefix: after the scrub, a replay that wrote
            # nothing is all zeros, and the leading rows may legitimately be
            # padding. Cheap at the correctness shape.
            verify=lambda out: bool(out.float().abs().sum() > 0),
        )
        out = result["out"]

        proj = fh.project(out)
        if args.make_golden:
            projections[case] = proj
        else:
            snrs[case] = fh.compare(golden[case], proj)
        # Drop the graph and its output before capturing the next layout.
        del out, result

    if args.make_golden:
        fh.save_golden(GOLDEN, rank, projections, group)
    else:
        fh.report_snr(snrs, rank, group)
    return 0


def run_bench(args, group, rank, world):
    _symm, handle, operands = build_context(fh.BENCH_SHAPE, group, rank, world)
    for layout in LAYOUTS:
        step = make_step(layout, operands, group, handle, args.bn)
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
    """One representative case, target kernel only, a handful of iterations."""
    _symm, handle, operands = build_context(fh.BENCH_SHAPE, group, rank, world)
    step = make_step("nt", operands, group, handle, args.bn)
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
