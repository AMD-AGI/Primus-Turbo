"""
Test signal-based pipeline EP overlap.

Usage:
  torchrun --standalone --nproc_per_node=8 tests/pytorch/cco/test_pipeline_ep.py \
      --num-tokens 4096 --hidden 7168 --num-topk 8 --num-experts 256 --num-groups 8
"""

import argparse
import time

import torch
import torch.distributed as dist
from utils import init_dist, inplace_unique, indices_to_map

from primus_turbo.pytorch.cco.pipeline_ep import (
    PipelineEPConfig,
    pipeline_ep_preprocess,
    pipeline_ep_dispatch,
    pipeline_ep_recv,
    pipeline_ep_full,
    get_grouped_gemm_schedule,
    _reset_signals,
)


def _reference_dispatch(x, topk_idx, topk_weights, group, num_experts):
    """Simple reference: all_gather tokens, then locally permute."""
    rank = group.rank()
    num_ranks = group.size()
    num_tokens, hidden = x.shape
    num_topk = topk_idx.shape[1]
    E_per_R = num_experts // num_ranks

    all_x = torch.empty(num_tokens * num_ranks, hidden, dtype=x.dtype, device=x.device)
    dist.all_gather_into_tensor(all_x, x.contiguous(), group=group)

    all_topk_idx = torch.empty(num_tokens * num_ranks, num_topk, dtype=topk_idx.dtype, device=x.device)
    dist.all_gather_into_tensor(all_topk_idx, topk_idx.contiguous(), group=group)

    all_topk_weights = torch.empty(num_tokens * num_ranks, num_topk, dtype=topk_weights.dtype, device=x.device)
    dist.all_gather_into_tensor(all_topk_weights, topk_weights.contiguous(), group=group)

    local_expert_start = rank * E_per_R
    local_expert_end = (rank + 1) * E_per_R

    expert_token_lists = [[] for _ in range(E_per_R)]
    total_tokens = num_tokens * num_ranks

    for t in range(total_tokens):
        for k in range(num_topk):
            e = all_topk_idx[t, k].item()
            if local_expert_start <= e < local_expert_end:
                le = e - local_expert_start
                expert_token_lists[le].append(t)

    expert_counts = torch.tensor(
        [len(lst) for lst in expert_token_lists], dtype=torch.int32, device=x.device
    )
    expert_offsets = torch.zeros(E_per_R, dtype=torch.int32, device=x.device)
    if E_per_R > 1:
        expert_offsets[1:] = expert_counts[:-1].cumsum(0).to(torch.int32)
    total_permuted = expert_counts.sum().item()

    expert_output = torch.zeros(total_permuted, hidden, dtype=x.dtype, device=x.device)
    next_pos = [0] * E_per_R
    for le in range(E_per_R):
        for t in expert_token_lists[le]:
            row = expert_offsets[le].item() + next_pos[le]
            expert_output[row] = all_x[t]
            next_pos[le] += 1

    return expert_output, expert_counts, expert_offsets


def test_correctness(args, local_rank, num_ranks, rank, group):
    num_tokens = args.num_tokens
    hidden = args.hidden
    num_topk = args.num_topk
    num_experts = args.num_experts
    num_groups = args.num_groups

    config = PipelineEPConfig(
        num_experts=num_experts,
        num_groups=num_groups,
        expert_alignment=1,
    )

    torch.manual_seed(42 + rank)
    x = torch.randn(num_tokens, hidden, dtype=torch.bfloat16, device="cuda")

    scores = torch.randn(num_tokens, num_experts, dtype=torch.float32, device="cuda").abs() + 1
    topk_idx = torch.topk(scores, num_topk, dim=-1, largest=True, sorted=False)[1]
    topk_weights = torch.ones(num_tokens, num_topk, dtype=torch.float32, device="cuda") * rank

    if rank == 0:
        print(f"[config] N={num_tokens} H={hidden} K={num_topk} E={num_experts} G={num_groups} R={num_ranks}")

    # ── Reference ─────────────────────────────────────────────
    ref_output, ref_counts, ref_offsets = _reference_dispatch(
        x, topk_idx, topk_weights, group, num_experts
    )
    group.barrier()

    # ── Pipeline EP ───────────────────────────────────────────
    handle = pipeline_ep_preprocess(topk_idx, topk_weights, x.shape, group, config)
    group.barrier()

    _reset_signals(handle)
    events = pipeline_ep_dispatch(x, handle)
    torch.cuda.synchronize()
    group.barrier()

    pipe_output = pipeline_ep_recv(handle, wait_for_signal=False)
    torch.cuda.synchronize()

    # ── Compare ───────────────────────────────────────────────
    E_per_R = num_experts // num_ranks
    E_per_G = config.experts_per_group

    all_ok = True
    for le in range(E_per_R):
        ref_start = ref_offsets[le].item()
        ref_end = ref_start + ref_counts[le].item()
        ref_slice = ref_output[ref_start:ref_end]

        pipe_start = handle.recv_expert_offsets[le].item()
        pipe_end = pipe_start + handle.recv_expert_counts[le].item()
        pipe_slice = pipe_output[pipe_start:pipe_end]

        if ref_slice.shape[0] != pipe_slice.shape[0]:
            print(f"[rank {rank}] expert {le}: count mismatch ref={ref_slice.shape[0]} pipe={pipe_slice.shape[0]}")
            all_ok = False
            continue

        if ref_slice.shape[0] == 0:
            continue

        ref_key = ref_slice.float().sum(dim=1)
        pipe_key = pipe_slice.float().sum(dim=1)
        ref_sorted = ref_slice[ref_key.argsort()]
        pipe_sorted = pipe_slice[pipe_key.argsort()]

        if not torch.equal(ref_sorted, pipe_sorted):
            diff = (ref_sorted.float() - pipe_sorted.float()).abs().max().item()
            n_mismatch = (ref_sorted != pipe_sorted).any(dim=1).sum().item()
            print(f"[rank {rank}] expert {le}: max diff = {diff}, "
                  f"mismatched rows = {n_mismatch}/{ref_slice.shape[0]}")
            all_ok = False

    status = "PASS" if all_ok else "FAIL"
    print(f"[rank {rank}] correctness: {status}", flush=True)

    # ── Pipeline full (with overlap) ──────────────────────────
    group.barrier()
    pipe_output2, handle2, recv_events = pipeline_ep_full(x, topk_idx, topk_weights, group, config)
    torch.cuda.synchronize()
    group.barrier()

    gemm_schedule = get_grouped_gemm_schedule(handle2)
    if rank == 0:
        print(f"[rank {rank}] pipeline_ep_full: output shape = {pipe_output2.shape}")
        for lg, (counts, offsets) in enumerate(gemm_schedule):
            print(f"  local_group {lg}: counts={counts.tolist()}, offsets={offsets.tolist()}")


def test_loop(local_rank, num_local_ranks, args):
    rank, num_ranks, group = init_dist(local_rank, num_local_ranks)
    torch.manual_seed(rank)
    test_correctness(args, local_rank, num_ranks, rank, group)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-processes", type=int, default=8)
    parser.add_argument("--num-tokens", type=int, default=256)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--num-topk", type=int, default=8)
    parser.add_argument("--num-experts", type=int, default=64)
    parser.add_argument("--num-groups", type=int, default=8)
    args = parser.parse_args()

    torch.multiprocessing.spawn(
        test_loop, args=(args.num_processes, args), nprocs=args.num_processes
    )
