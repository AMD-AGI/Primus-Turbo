"""
This is the same test_intranode.py test in DeepEP's repo.
OMP_NUM_THREADS=8 torchrun --standalone --nproc_per_node=8 bench/test_intranode.py \
    --num-tokens 4096 --hidden 7168 --num-topk 8 --num-experts 256
"""

import argparse
import time

import torch
import torch.distributed as dist
from utils import init_dist, inplace_unique, per_token_cast_to_fp8, bench_kineto, indices_to_map, permute, indices_to_multihot

from primus_turbo.pytorch.cco import _fused_dispatch_permute


def permute_ref(ref_recv_x, ref_recv_topk_idx, ref_recv_topk_weights, num_local_experts, num_out_tokens):
    local_routing, local_probs = indices_to_multihot(
        num_local_experts, ref_recv_topk_idx, ref_recv_topk_weights)
    permuted_x, _, _, _, _ = permute(
        ref_recv_x, local_routing, num_out_tokens=num_out_tokens)
    return permuted_x


def compute_recv_and_permuted_x(rank_prefix_matrix, dispatch_to_expert_map, hidden_size, num_ranks, current_rank, num_out_tokens):
    """
    Generate recv_x from rank_prefix_matrix, then compute permuted_x via dispatch_to_expert_map.

    recv_x generation: rank_prefix_matrix[i][current_rank] rows with value i, replicated hidden_size times.
    permuted_x is recv_x reordered according to dispatch_to_expert_map.
    """
    num_recv_tokens = rank_prefix_matrix[num_ranks - 1][current_rank].item()

    recv_x = torch.empty((num_recv_tokens, hidden_size),
                         dtype=torch.bfloat16, device="cuda")
    start = 0
    for i in range(num_ranks):
        end = rank_prefix_matrix[i][current_rank].item()
        recv_x[start:end, :] = i
        start = end

    d2e = dispatch_to_expert_map[:num_recv_tokens]
    valid_mask = d2e >= 0

    permuted_x = torch.empty(
        (num_out_tokens, hidden_size), dtype=torch.bfloat16, device="cuda")
    src_tokens = torch.arange(
        num_recv_tokens, device="cuda").unsqueeze(1).expand_as(d2e)
    valid_dst = d2e[valid_mask]
    valid_src = src_tokens[valid_mask]

    assert valid_dst.max(
    ) < num_out_tokens, f"valid_dst.max()={valid_dst.max()}, num_out_tokens={num_out_tokens}"
    assert valid_src.max() < num_recv_tokens
    permuted_x[valid_dst] = recv_x[valid_src]

    return recv_x, permuted_x


def test_main(
    args: argparse.Namespace,
    num_sms: int,
    local_rank: int,
    num_ranks: int,
    rank: int,
    group: dist.ProcessGroup,
):
    # Settings
    num_tokens, hidden = args.num_tokens, args.hidden
    num_topk, num_experts = args.num_topk, args.num_experts

    assert num_experts % num_ranks == 0
    if local_rank == 0:
        print(
            f"[config] num_tokens={num_tokens}, hidden={hidden}, num_topk={num_topk}",
            flush=True,
        )

    # Random data
    x = torch.ones((num_tokens, hidden),
                   dtype=torch.bfloat16, device="cuda") * rank
    x_pure_rand = torch.randn((num_tokens, hidden),
                              dtype=torch.bfloat16, device="cuda")
    x_e4m3 = per_token_cast_to_fp8(x)
    x_e4m3 = (x_e4m3[0], x_e4m3[1].T.contiguous(
    ).T) if x_e4m3 is not None else None
    scores = torch.randn((num_tokens, num_experts),
                         dtype=torch.float32, device="cuda").abs() + 1
    topk_idx = torch.topk(scores, num_topk, dim=-1,
                          largest=True, sorted=False)[1]
    routing_map, _ = indices_to_map(topk_idx, None, num_tokens, num_experts)
    topk_weights = torch.ones((num_tokens, num_topk),
                              dtype=torch.float32, device="cuda") * rank
    topk_weights_pure_rand = torch.randn(
        (num_tokens, num_topk), dtype=torch.float32, device="cuda")
    rank_idx = topk_idx // (num_experts // num_ranks)
    rank_idx.masked_fill_(topk_idx == -1, -1)
    inplace_unique(rank_idx, num_ranks)

    # Expert meta
    num_tokens_per_expert = torch.zeros(
        (num_experts,), dtype=torch.int, device="cuda")
    for i in range(num_experts):
        num_tokens_per_expert[i] = (topk_idx == i).sum()
    gbl_num_tokens_per_expert = num_tokens_per_expert.clone()
    dist.all_reduce(gbl_num_tokens_per_expert, group=group)

    # Rank layout meta
    num_tokens_per_rank = torch.empty(
        (num_ranks,), dtype=torch.int, device="cuda")
    token_idx_in_rank = torch.full(
        (num_ranks, num_tokens), -1, dtype=torch.long, device="cuda")
    for i in range(num_ranks):
        num_tokens_per_rank[i] = (rank_idx == i).sum()
        token_sel = (rank_idx == i).max(dim=-1)[0]
        count = token_sel.sum().item()
        tokens = torch.sort(token_sel.to(torch.int), descending=True)[1]
        tokens[:count] = torch.sort(tokens[:count])[0]
        token_idx_in_rank[i][tokens[:count]] = torch.arange(
            count, dtype=torch.long, device="cuda")
    token_idx_in_rank = token_idx_in_rank.T.contiguous().to(torch.int)
    is_token_in_rank = token_idx_in_rank >= 0
    gbl_num_tokens_per_rank = num_tokens_per_rank.clone()
    dist.all_reduce(gbl_num_tokens_per_rank, group=group)

    def check_data(check_x, rank_prefix_matrix):
        assert torch.allclose(check_x.amin(dim=1), check_x.amax(dim=1))
        check_start = 0
        for i in range(num_ranks):
            check_end = rank_prefix_matrix[i][rank].item()
            assert (check_x[check_start:check_end,
                    :].int() - i).sum().item() == 0
            check_start = check_end

    (
        ref_permuted_x,
        ref_recv_x_scales,
        ref_recv_topk_idx,
        ref_recv_topk_weights,
        ref_moe_recv_counter,
        ref_moe_recv_expert_counter,
        ref_rank_prefix_matrix,
        ref_channel_prefix_matrix,
        ref_dispatch_to_expert_map,
        ref_send_head
    ), handle = _fused_dispatch_permute(
        x, group, topk_idx, topk_weights, num_experts, num_sms=num_sms
    )
    group.barrier()

    time.sleep(1)

    (ref_recv_x, ref_num_tokens_per_rank, ref_num_tokens_per_expert,
     ref_rank_prefix_matrix, ref_channel_prefix_matrix, ref_is_token_in_rank, ref_expert_prefix, ref_channel_expert_prefix) = handle

    num_moe_recv_tokens = ref_moe_recv_counter.item()
    recv_num_tokens_per_expert_list = ref_moe_recv_expert_counter.tolist()

    # split out to num_moe_recv_tokens
    ref_recv_topk_idx = ref_recv_topk_idx[:num_moe_recv_tokens, :]
    ref_recv_topk_weights = ref_recv_topk_weights[:num_moe_recv_tokens, :]

    assert (
        gbl_num_tokens_per_expert.view(num_ranks, -1)[rank].tolist()
        == recv_num_tokens_per_expert_list
    )

    assert torch.allclose(ref_num_tokens_per_rank, num_tokens_per_rank)
    assert torch.allclose(ref_num_tokens_per_expert, num_tokens_per_expert)
    assert torch.allclose(ref_is_token_in_rank, is_token_in_rank)

    # Check `topk_idx`
    assert (ref_recv_topk_idx.eq(-1) | ((ref_recv_topk_idx >= 0) & (ref_recv_topk_idx <
            (num_experts // num_ranks)))).sum().item() == ref_recv_topk_idx.numel()
    for i, count in enumerate(recv_num_tokens_per_expert_list):
        assert ref_recv_topk_idx.eq(i).sum().item() == count

    # Check `topk_weights`
    ref_recv_topk_weights[ref_recv_topk_idx.eq(-1)] = (
        ref_recv_topk_weights.amax(dim=1, keepdim=True).expand_as(
            ref_recv_topk_weights
        )[ref_recv_topk_idx.eq(-1)]
    )
    check_data(ref_recv_topk_weights, ref_rank_prefix_matrix)

    num_out_tokens = sum(recv_num_tokens_per_expert_list)
    num_experts_per_rank = num_experts // num_ranks

    # compute permuted recv_x
    recv_x, permuted_x = compute_recv_and_permuted_x(
        ref_rank_prefix_matrix, ref_dispatch_to_expert_map, hidden, num_ranks, rank, num_out_tokens)

    # assert torch.allclose(ref_recv_x[:num_moe_recv_tokens, :], recv_x), f"{ref_recv_x[:num_moe_recv_tokens, 0]} {recv_x[:num_moe_recv_tokens, 0]}"
    assert torch.allclose(ref_permuted_x[:num_out_tokens, :], permuted_x), f"{ref_permuted_x[:num_out_tokens, 0]} {permuted_x[:num_out_tokens, 0]}"

    if local_rank == 0:
        print("dispatch_permute correctness checks passed", flush=True)

    # benchmark performance
    nvl_recv_bytes = num_moe_recv_tokens * hidden * 2

    kwargs = {"x": x, "group": group,  "topk_idx": topk_idx,
              "topk_weights": topk_weights, "num_experts": num_experts, "num_sms": num_sms}

    for num_max_send_tokens in range(2, 33, 2):
        t, notify_t = bench_kineto(lambda: _fused_dispatch_permute(
            **kwargs, num_max_send_tokens=num_max_send_tokens), ("fused_dispatch_permute", "notify_dispatch"), suppress_kineto_output=True)

        if local_rank == 0:
            print(f"SMs {num_sms}: num_max_send_tokens={num_max_send_tokens}: {nvl_recv_bytes / 1e9 / t:.2f} GB/s (NVL), {t * 1e6:.2f} us + {notify_t * 1e6:.2f} us",
                  flush=True,
                  )


def test_loop(local_rank: int, num_local_ranks: int, args: argparse.Namespace):
    rank, num_ranks, group = init_dist(local_rank, num_local_ranks)

    torch.manual_seed(rank)

    for i in (48,):
        test_main(args, i, local_rank, num_ranks, rank, group)
        if local_rank == 0:
            print("", flush=True)

    # Destroy the buffer runtime and communication group
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test intranode EP kernels")
    parser.add_argument(
        "--num-processes", type=int, default=8, help="Number of processes to spawn (default: 8)"
    )
    parser.add_argument("--num-tokens", type=int, default=4096,
                        help="Number of tokens (default: 4096)")
    parser.add_argument("--hidden", type=int, default=7168,
                        help="Hidden dimension size (default: 7168)")
    parser.add_argument("--num-topk", type=int, default=8,
                        help="Number of top-k experts (default: 8)")
    parser.add_argument("--num-experts", type=int, default=256,
                        help="Number of experts (default: 256)")
    args = parser.parse_args()

    num_processes = args.num_processes
    torch.multiprocessing.spawn(test_loop, args=(
        num_processes, args), nprocs=num_processes)
