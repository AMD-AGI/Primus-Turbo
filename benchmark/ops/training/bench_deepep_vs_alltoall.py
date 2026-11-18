###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Intranode DeepEP against the RCCL all-to-all of Megatron's alltoall token dispatcher.

Shape: the DeepSeek-V3 MoE layer (hidden 7168 BF16, top-8 of 256 experts, routing limited to the
top 4 of 8 expert groups, random router logits). DeepEP is timed as training calls it: layout,
notify and dispatch without a cached handle, then combine. The all-to-all sends one copy of a token
per expert and includes the all-gather of the counts and the device-to-host copy of the splits that
Megatron does before it; neither side includes the local permute.

Timing: a 512 MB flush before every iteration, the slowest rank of each iteration, median of 30
iterations, mean of 3 rounds.

    python benchmark/ops/training/bench_deepep_vs_alltoall.py --tokens 4096 8192 --num-cus 32 64 128
"""

import argparse
import os
import statistics

import torch
import torch.distributed as dist

HIDDEN, NUM_EXPERTS, TOPK, N_GROUP, TOPK_GROUP = 7168, 256, 8, 8, 4
WARMUP, ITERS, ROUNDS = 5, 30, 3


def route(num_tokens, gen):
    scores = torch.randn((num_tokens, NUM_EXPERTS), generator=gen, device="cuda").sigmoid()
    grouped = scores.view(num_tokens, N_GROUP, -1)
    group_scores = grouped.topk(2, dim=-1).values.sum(-1)
    keep = torch.zeros_like(group_scores).scatter_(1, group_scores.topk(TOPK_GROUP, dim=-1).indices, 1)
    masked = (grouped * keep.unsqueeze(-1)).view(num_tokens, NUM_EXPERTS)
    weights, idx = masked.topk(TOPK, dim=-1)
    return idx.to(torch.int64), (weights / weights.sum(-1, keepdim=True)).float()


def timed(fn, flush):
    rounds = []
    for _ in range(ROUNDS):
        for _ in range(WARMUP):
            fn()
        times = []
        for _ in range(ITERS):
            flush.zero_()
            torch.cuda.synchronize()
            dist.barrier()
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            fn()
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))
        slowest = torch.tensor(times, device="cuda")
        dist.all_reduce(slowest, op=dist.ReduceOp.MAX)
        rounds.append(statistics.median(slowest.tolist()))
    return sum(rounds) / len(rounds) * 1000


def bench_deepep(buffer, x, topk_idx, topk_weights, num_ranks, num_cus, flush):
    """DeepEP layout + dispatch (no cached handle) and combine at ``num_cus`` CUs."""
    from primus_turbo.pytorch.deep_ep import Buffer

    Buffer.set_num_sms(num_cus)
    dispatch_config = Buffer.get_dispatch_config(num_ranks)
    combine_config = Buffer.get_combine_config(num_ranks)
    state = {}

    def dispatch():
        per_rank, _, per_expert, in_rank, _ = buffer.get_dispatch_layout(topk_idx, NUM_EXPERTS)
        recv_x, _, recv_weights, _, handle, _ = buffer.dispatch(
            x,
            topk_idx=topk_idx,
            topk_weights=topk_weights,
            num_tokens_per_rank=per_rank,
            is_token_in_rank=in_rank,
            num_tokens_per_expert=per_expert,
            config=dispatch_config,
        )
        state.update(recv_x=recv_x, recv_weights=recv_weights, handle=handle)

    def combine():
        buffer.combine(
            state["recv_x"],
            state["handle"],
            topk_weights=state["recv_weights"],
            config=combine_config,
        )

    t_dispatch = timed(dispatch, flush)
    dispatch()
    t_combine = timed(combine, flush)
    return t_dispatch, t_combine, state["recv_x"].shape[0]


def bench_alltoall(x, topk_idx, local_rank, num_ranks, flush):
    """Megatron's alltoall dispatcher: count all_gather, splits to the host, all_to_all_single."""
    experts_per_rank = NUM_EXPERTS // num_ranks
    flat = topk_idx.flatten()
    permuted = x.index_select(0, flat.argsort(stable=True) // TOPK)
    tokens_per_expert = torch.bincount(flat, minlength=NUM_EXPERTS).to(torch.int64)
    combined = torch.empty_like(permuted)
    state = {}

    def dispatch():
        counts = torch.empty((num_ranks, NUM_EXPERTS), dtype=torch.int64, device="cuda")
        dist.all_gather_into_tensor(counts, tokens_per_expert)
        local = counts[:, local_rank * experts_per_rank : (local_rank + 1) * experts_per_rank]
        input_splits = tokens_per_expert.view(num_ranks, experts_per_rank).sum(1).cpu().tolist()
        output_splits = local.sum(1).cpu().tolist()
        out = torch.empty((sum(output_splits), HIDDEN), dtype=x.dtype, device="cuda")
        dist.all_to_all_single(out, permuted, output_splits, input_splits)
        state.update(out=out, input_splits=input_splits, output_splits=output_splits)

    def combine():
        dist.all_to_all_single(combined, state["out"], state["input_splits"], state["output_splits"])

    t_dispatch = timed(dispatch, flush)
    t_combine = timed(combine, flush)
    return t_dispatch, t_combine, state["out"].shape[0]


def run(local_rank, num_ranks, args):
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29533")
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        "nccl", rank=local_rank, world_size=num_ranks, device_id=torch.device("cuda", local_rank)
    )
    from primus_turbo.pytorch.deep_ep import Buffer

    flush = torch.empty(512 * 1024 * 1024 // 4, dtype=torch.int32, device="cuda")
    hidden_bytes = HIDDEN * 2
    Buffer.set_num_sms(max(args.num_cus))
    num_nvl_bytes = 0
    for config in (Buffer.get_dispatch_config(num_ranks), Buffer.get_combine_config(num_ranks)):
        num_nvl_bytes = max(config.get_nvl_buffer_size_hint(hidden_bytes, num_ranks), num_nvl_bytes)
    num_nvl_bytes = max(
        Buffer.get_token_buffer_size_hint(max(args.tokens), hidden_bytes, num_ranks, TOPK), num_nvl_bytes
    )
    buffer = Buffer(dist.group.WORLD, num_nvl_bytes, 0, explicitly_destroy=True)
    results = []

    for num_tokens in args.tokens:
        gen = torch.Generator(device="cuda").manual_seed(1234 + local_rank)
        x = torch.randn((num_tokens, HIDDEN), dtype=torch.bfloat16, device="cuda", generator=gen)
        topk_idx, topk_weights = route(num_tokens, gen)
        for num_cus in args.num_cus:
            timing = bench_deepep(buffer, x, topk_idx, topk_weights, num_ranks, num_cus, flush)
            results.append((num_tokens, f"DeepEP, {num_cus} CUs", *timing))
        timing = bench_alltoall(x, topk_idx, local_rank, num_ranks, flush)
        results.append((num_tokens, "RCCL all-to-all", *timing))

    rows = torch.tensor([r[4] for r in results], device="cuda")
    dist.all_reduce(rows, op=dist.ReduceOp.MAX)
    if local_rank == 0:
        print(f"{torch.cuda.get_device_name()}, EP{num_ranks}")
        print("tokens/rank | method | notify+dispatch (us) | combine (us) | rows received per GPU (max)")
        for (num_tokens, name, t_dispatch, t_combine, _), received in zip(results, rows.tolist()):
            print(f"{num_tokens} | {name} | {t_dispatch:.0f} | {t_combine:.0f} | {received}")
    buffer.destroy()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--num-processes", type=int, default=8)
    parser.add_argument("--tokens", type=int, nargs="+", default=[4096, 8192], help="tokens per rank")
    parser.add_argument("--num-cus", type=int, nargs="+", default=[32, 64, 128], help="CUs for DeepEP")
    args = parser.parse_args()
    torch.multiprocessing.spawn(run, args=(args.num_processes, args), nprocs=args.num_processes)
