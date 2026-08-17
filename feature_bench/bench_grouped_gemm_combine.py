###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Time the fused grouped GEMM + combine (sender-side dedup, the only mode).

Same DeepSeek-V3 BF16 MoE shape as bench_dispatch_grouped_gemm.py, but for the
L2 leg: grouped GEMM -> combine PUSH -> topk reduce. The sender folds a token's
local routes into a single push, so the reduce sees one slot per source token.

Both layouts the operator is used with are covered (see docs/README_Mega_MoE.md):
    nt  forward L2          act @ w2      -> y, weighted top-k reduce
    nn  backward L1 dgrad   grad_l1 @ w1  -> dx, plus the gate-grad scatter

The prologue + dispatch run once to build a real handle and real activations; the
timed region is only the fused combine kernel. Only rank 0's numbers are printed.

Run:
    python feature_bench/bench_grouped_gemm_combine.py
    python feature_bench/bench_grouped_gemm_combine.py --layouts nt --profile
"""

import argparse
import math
import os
import sys

import numpy as np
import torch
import torch.distributed as dist

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..")))

import primus_turbo.flydsl.mega.grouped_gemm_combine_bf16_kernel as ggc  # noqa: E402
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

DSV3 = dict(hidden=7168, inter=2048, num_experts=256, num_topk=8)
# Kimi-K2 is this same H/I with E=384; override to reach it without the full bench.
DSV3["num_experts"] = int(os.environ.get("TURBO_BENCH_E", DSV3["num_experts"]))
NUM_GROUPS = 8
GROUP_TOPK = 4
ROUTING_SCALE = 2.5
ALL_LAYOUTS = ("nt", "nn")
# Handle slots, mirrored from the production callers (fused_mega_moe_*_impl.py).
# Local constants on purpose: reading the kernel's private names would break on a
# legal refactor of that file.
_H_SOURCE_SLOT_KIND = 13
_H_NUM_TILE_BLOCKS = 8
_H_DEDUP_KEY_ROW = 20
_MIN_HANDLE_LEN = 21


def generate_deepseek_v3_routing(num_tokens):
    """Generate DeepSeek-V3-style group-limited sigmoid top-k routing."""
    num_experts = DSV3["num_experts"]
    num_topk = DSV3["num_topk"]
    scores = torch.sigmoid(torch.randn(num_tokens, num_experts, device="cuda", dtype=torch.float32))
    grouped = scores.view(num_tokens, NUM_GROUPS, num_experts // NUM_GROUPS)
    group_scores = grouped.topk(num_topk // GROUP_TOPK, dim=-1).values.sum(dim=-1)
    selected_groups = group_scores.topk(GROUP_TOPK, dim=-1, sorted=False).indices
    group_mask = torch.zeros_like(group_scores, dtype=torch.bool)
    group_mask.scatter_(1, selected_groups, True)
    expert_mask = (
        group_mask.unsqueeze(-1)
        .expand(num_tokens, NUM_GROUPS, num_experts // NUM_GROUPS)
        .reshape(num_tokens, num_experts)
    )
    topk_weight, topk_idx = torch.topk(scores.masked_fill(~expert_mask, float("-inf")), num_topk, dim=-1)
    topk_weight = topk_weight / (topk_weight.sum(dim=-1, keepdim=True) + 1e-20)
    return topk_idx.to(torch.int64), (topk_weight * ROUTING_SCALE).to(torch.float32)


def generate_uniform_routing(num_tokens):
    """Plain softmax top-k over all experts: fewer per-(token, rank) duplicates."""
    logits = torch.randn(num_tokens, DSV3["num_experts"], device="cuda")
    topk_weight, topk_idx = torch.topk(logits.softmax(-1), DSV3["num_topk"], dim=-1)
    return topk_idx.to(torch.int64), topk_weight.to(torch.float32)


def bench(fn, *, warmup, iters, presync=None):
    """Return mean milliseconds per call from CUDA events."""
    torch.cuda.synchronize()
    for _ in range(warmup):
        fn()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for index in range(iters):
        if presync is not None:
            # back-to-back launches let the ranks drift; the drift then shows up
            # inside the kernel as time spent waiting on peer pushes
            presync()
        starts[index].record()
        fn()
        ends[index].record()
    torch.cuda.synchronize()
    return float(np.mean([start.elapsed_time(end) for start, end in zip(starts, ends)][1:]))


def _worker(local_rank, world, args):
    master_addr = os.getenv("MASTER_ADDR", "127.0.0.1")
    port = int(os.getenv("MASTER_PORT", "29593"))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        "nccl", init_method=f"tcp://{master_addr}:{port}", world_size=world, rank=local_rank
    )
    torch.set_default_device("cuda")
    torch.cuda.set_device(local_rank)
    group = dist.group.WORLD
    rank = dist.get_rank()

    T = args.num_tokens
    H, I = DSV3["hidden"], DSV3["inter"]
    E, K = DSV3["num_experts"], DSV3["num_topk"]
    experts_per_rank = E // world
    if world != NUM_GROUPS:
        raise ValueError(f"this DeepSeek-V3 bench requires EP{NUM_GROUPS}, got EP{world}")
    layouts = [name.strip() for name in args.layouts.split(",") if name.strip()]
    for name in layouts:
        if name not in ALL_LAYOUTS:
            raise ValueError(f"unknown layout {name}, pick from {ALL_LAYOUTS}")

    try:
        symm = get_symm_buffer_for_mega_moe(
            group,
            num_experts=E,
            num_max_tokens_per_rank=T,
            num_topk=K,
            hidden=H,
            intermediate_hidden=I,
        )
        pool_rows = int(symm.num_max_pool_tokens)
        torch.manual_seed(1234 + rank)
        x = torch.randn(T, H, device="cuda", dtype=torch.bfloat16)
        w1 = torch.randn(experts_per_rank, 2 * I, H, device="cuda", dtype=torch.bfloat16)
        w1.mul_(1.0 / math.sqrt(H))
        w2 = torch.randn(experts_per_rank, H, I, device="cuda", dtype=torch.bfloat16)
        w2.mul_(1.0 / math.sqrt(I))
        if args.routing == "uniform":
            topk_idx, topk_weight = generate_uniform_routing(T)
        else:
            topk_idx, topk_weight = generate_deepseek_v3_routing(T)
        topk_indices_flat = topk_idx.contiguous().view(-1)
        topk_weights_flat = topk_weight.to(torch.float32).contiguous().view(-1)

        # real handle + real activations from one L1 pass
        dist.barrier(group)
        l1_out, _, _, handle = dispatch_grouped_gemm_bf16_flydsl_kernel(
            x, w1, group, handle=None, topk_idx=topk_idx, topk_weights=topk_weight, layout="nt"
        )
        # Fail loudly rather than silently reading the wrong tensor if the ABI moves.
        if len(handle) < _MIN_HANDLE_LEN:
            raise RuntimeError(f"handle has {len(handle)} entries, expected >= {_MIN_HANDLE_LEN}")
        act = swiglu_flydsl_kernel(l1_out, num_tile_blocks=handle[_H_NUM_TILE_BLOCKS])
        del l1_out
        torch.cuda.synchronize()
        active_rows = int(handle[_H_NUM_TILE_BLOCKS][0].item()) * POOL_BLOCK_M

        # Dedup's contract: combine pushes one row per (src_rank, src_token) that
        # dispatch sent, i.e. the exact inverse volume. Only the EP-wide sums are
        # comparable -- kind counts what this rank SENT, key_row what it pushes back
        # for what it RECEIVED, and those two sets differ per rank.
        key_row = handle[_H_DEDUP_KEY_ROW].view(-1, K)
        volume = torch.tensor(
            [
                int((handle[_H_SOURCE_SLOT_KIND] != 0).sum().item()),  # dispatch sent
                int((key_row[:, 0] >= 0).sum().item()),  # combine pushes back
                int((key_row >= 0).sum().item()),  # un-deduped combine would push
            ],
            device="cuda",
            dtype=torch.int64,
        )
        dist.all_reduce(volume, group=group)
        dispatch_rows, combine_rows, naive_rows = (int(v) for v in volume.tolist())

        ggc._COMBINE_DEDUP_NPASS = args.npass

        # nn is the backward L1 dgrad: no routing weights, but it scatters the gate grad
        grad_l1 = torch.randn(pool_rows, 2 * I, device="cuda", dtype=torch.bfloat16)
        grad_l1.mul_(1.0 / math.sqrt(H))
        grad_gate = torch.randn(pool_rows, device="cuda", dtype=torch.float32)

        def run(layout):
            if layout == "nt":
                lhs, rhs, weights, gate = act, w2, topk_weights_flat, None
            else:
                lhs, rhs, weights, gate = grad_l1, w1, None, grad_gate
            return grouped_gemm_combine_bf16_flydsl_kernel(
                lhs,
                rhs,
                handle,
                topk_indices=topk_indices_flat,
                topk_weights=weights,
                grad_gate=gate,
                layout=layout,
                BM=POOL_BLOCK_M,
                BN=args.bn,
            )[0]

        def presync():
            torch.cuda.synchronize()
            dist.barrier(group)

        sync = None if args.back_to_back else presync
        if rank == 0:
            print(
                f"\nDeepSeek-V3 grouped_gemm_combine (rank 0)  "
                f"T={T} H={H} I={I} E={E} K={K} EP={world} active_rows={active_rows} "
                f"BM={POOL_BLOCK_M} BN={args.bn} npass={args.npass} routing={args.routing} "
                f"sync_each={not args.back_to_back}"
            )
            row_mb = H * 2 / 1024 / 1024 / world
            print(
                f"  push volume (EP{world} total): combine {combine_rows} rows vs "
                f"dispatch {dispatch_rows} rows, un-deduped {naive_rows} rows "
                f"-> {combine_rows * row_mb:.0f} MB/rank, "
                f"{100.0 * (1.0 - combine_rows / max(naive_rows, 1)):.1f}% saved"
            )
            header = f"{'layout':>6} {'ms':>9}"
            print(header)
            print("-" * len(header))

        for layout in layouts:
            dist.barrier(group)
            samples = []
            for _ in range(args.rounds):
                dist.barrier(group)
                samples.append(
                    bench(lambda lo=layout: run(lo), warmup=args.warmup, iters=args.iters, presync=sync)
                )
            dist.barrier(group)

            if args.profile:
                from torch.profiler import ProfilerActivity, profile

                for _ in range(5):
                    run(layout)
                torch.cuda.synchronize()
                dist.barrier(group)
                with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
                    for _ in range(args.profile_iters):
                        if sync is not None:
                            sync()
                        run(layout)
                    torch.cuda.synchronize()
                if rank == 0:
                    rows = [
                        (e.key, e.device_time_total / max(e.count, 1), e.count)
                        for e in prof.key_averages()
                        if e.device_time_total > 0 and "kernel" in e.key and "nccl" not in e.key
                    ]
                    rows.sort(key=lambda r: -r[1])
                    for key, us, count in rows[:2]:
                        print(f"  [prof {layout}] {key[:46]:46s} {us:9.1f} us x{count}", flush=True)
                dist.barrier(group)

            if rank == 0:
                print(f"{layout:>6} {float(np.median(samples)):>9.4f}", flush=True)
    finally:
        try:
            get_symm_buffer_for_mega_moe().destroy()
        except RuntimeError:
            pass
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-processes", type=int, default=8)
    parser.add_argument("--num-tokens", type=int, default=8192)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--bn", type=int, default=256)
    parser.add_argument("--npass", type=int, default=2)
    parser.add_argument("--layouts", default=",".join(ALL_LAYOUTS), help="subset of nt,nn")
    parser.add_argument("--routing", choices=("dsv3", "uniform"), default="dsv3")
    parser.add_argument("--back-to-back", action="store_true", help="drop the per-iteration rank re-sync")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--profile-iters", type=int, default=30)
    args = parser.parse_args()
    torch.multiprocessing.spawn(_worker, args=(args.num_processes, args), nprocs=args.num_processes)


if __name__ == "__main__":
    main()
