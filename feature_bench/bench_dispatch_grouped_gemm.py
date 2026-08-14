###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Time the fused dispatch + grouped GEMM (sender-side dedup, the only mode).

The fixed workload is the DeepSeek-V3 BF16 MoE shape with 8192 tokens per EP
rank. The dispatch prologue runs once to build a shared handle, so the timing
covers only the fused dispatch + grouped-GEMM kernel.

All three layouts the operator is used with are covered (see docs/README_Mega_MoE.md):
    nt  forward L1          x @ w1        -> pool activations
    nn  backward L2 dgrad   dy @ w2       -> pool grad
    tn  backward dW1        pool(x)^T @ grad_l1

Only rank 0's numbers are printed; the ranks differ by <1% anyway.

Run:
    python feature_bench/bench_dispatch_grouped_gemm.py
    python feature_bench/bench_dispatch_grouped_gemm.py --layouts nt --profile
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
# The disk cache hashes the top-level kernel but not the inlined tile helpers,
# so disable it unless the caller opts in.
os.environ.setdefault("FLYDSL_RUNTIME_ENABLE_CACHE", "0")

import primus_turbo.pytorch  # noqa: E402,F401
from primus_turbo.flydsl.mega import (  # noqa: E402
    dispatch_grouped_gemm_bf16_flydsl_kernel,
    dispatch_prologue_flydsl_kernel,
)
from primus_turbo.flydsl.mega.symm_buffer import BLOCK_M as POOL_BLOCK_M  # noqa: E402
from primus_turbo.flydsl.mega.symm_buffer import (  # noqa: E402
    get_symm_buffer_for_mega_moe,
)

DSV3 = dict(hidden=7168, inter=2048, num_experts=256, num_topk=8)
NUM_GROUPS = 8
GROUP_TOPK = 4
ROUTING_SCALE = 2.5
ALL_LAYOUTS = ("nt", "nn", "tn")


def generate_deepseek_v3_routing(num_tokens, *, device="cuda"):
    """Generate DeepSeek-V3-style group-limited sigmoid top-k routing."""
    num_experts = DSV3["num_experts"]
    num_topk = DSV3["num_topk"]
    scores = torch.sigmoid(torch.randn(num_tokens, num_experts, device=device, dtype=torch.float32))
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
    port = int(os.getenv("MASTER_PORT", "29592"))
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
        x = (
            torch.ones(T, H, device="cuda", dtype=torch.bfloat16)
            if args.fill == "ones"
            else torch.randn(T, H, device="cuda", dtype=torch.bfloat16)
        )
        if args.routing == "uniform":
            # plain softmax top-k over all experts: fewer per-(token,rank) duplicates
            logits = torch.randn(T, E, device="cuda")
            topk_weight, topk_idx = torch.topk(logits.softmax(-1), K, dim=-1)
            topk_idx = topk_idx.to(torch.int64)
            topk_weight = topk_weight.to(torch.float32)
        else:
            topk_idx, topk_weight = generate_deepseek_v3_routing(T)

        # per-layout (activation, rhs); rhs is the weight for nt/nn, the pool-side
        # activation for the tn wgrad
        # all-ones operands draw much less MFMA switching power than random data,
        # which shows up as higher clocks on the big wgrad GEMM
        def fill(shape, scale):
            if args.fill == "ones":
                return torch.ones(shape, device="cuda", dtype=torch.bfloat16)
            return torch.randn(shape, device="cuda", dtype=torch.bfloat16).mul_(scale)

        def make_operands(layout):
            if layout == "nt":
                return x, fill((experts_per_rank, 2 * I, H), 1.0 / math.sqrt(H))
            if layout == "nn":
                return fill((T, H), 1.0), fill((experts_per_rank, H, I), 1.0 / math.sqrt(I))
            return x, fill((pool_rows, 2 * I), 1.0 / math.sqrt(H))

        prologue = tuple(
            dispatch_prologue_flydsl_kernel(
                topk_idx,
                topk_weight,
                sym_buffer=symm.get_sym_buffer(),
                num_tokens=T,
                num_topk=K,
                num_experts=E,
                num_ranks=world,
                rank=rank,
                experts_per_rank=experts_per_rank,
                block_m=POOL_BLOCK_M,
                num_max_pool_tokens=symm.num_max_pool_tokens,
                hidden=H,
                num_max_tokens_per_rank=T,
            )
        )
        handle = prologue[:11] + (symm.pool_src_slot.clone(),) + prologue[11:]
        active_rows = int(handle[7][0].item()) * POOL_BLOCK_M

        def presync():
            torch.cuda.synchronize()
            dist.barrier(group)

        sync = None if args.back_to_back else presync
        if rank == 0:
            print(
                f"\nDeepSeek-V3 dispatch_grouped_gemm (rank 0)  "
                f"T={T} H={H} I={I} E={E} K={K} EP={world} active_rows={active_rows} "
                f"BM={POOL_BLOCK_M} BN={args.bn} "
                f"routing={args.routing} fill={args.fill} sync_each={not args.back_to_back}"
            )
            header = f"{'layout':>6} {'ms':>9}"
            print(header)
            print("-" * len(header))

        for layout in layouts:
            act, rhs = make_operands(layout)

            def run(layout=layout, act=act, rhs=rhs):
                return dispatch_grouped_gemm_bf16_flydsl_kernel(
                    act,
                    rhs,
                    group,
                    handle=handle,
                    layout=layout,
                    BM=POOL_BLOCK_M,
                    BN=args.bn,
                    # dW1 is stored W1-native [G, 2I, H]; the untransposed store is slower
                    trans_c=(layout == "tn"),
                )[0]

            dist.barrier(group)
            samples = []
            for _ in range(args.rounds):
                dist.barrier(group)
                samples.append(bench(run, warmup=args.warmup, iters=args.iters, presync=sync))
            dist.barrier(group)

            if args.profile:
                from torch.profiler import ProfilerActivity, profile

                for _ in range(5):
                    run()
                torch.cuda.synchronize()
                dist.barrier(group)
                with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
                    for _ in range(args.profile_iters):
                        if sync is not None:
                            sync()
                        run()
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

            del act, rhs
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
    parser.add_argument("--layouts", default=",".join(ALL_LAYOUTS), help="subset of nt,nn,tn")
    parser.add_argument("--routing", choices=("dsv3", "uniform"), default="dsv3")
    parser.add_argument("--fill", choices=("randn", "ones"), default="randn")
    parser.add_argument("--back-to-back", action="store_true", help="drop the per-iteration rank re-sync")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--profile-iters", type=int, default=30)
    args = parser.parse_args()
    torch.multiprocessing.spawn(_worker, args=(args.num_processes, args), nprocs=args.num_processes)


if __name__ == "__main__":
    main()
