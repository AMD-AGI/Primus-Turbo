"""
This is the same test_intranode.py test in DeepEP's repo.
OMP_NUM_THREADS=8 torchrun --standalone --nproc_per_node=8 bench/test_intranode.py \
    --num-tokens 4096 --hidden 7168 --num-topk 8 --num-experts 256
"""

import argparse
import time

import torch
import torch.distributed as dist
from utils import init_dist, inplace_unique, per_token_cast_to_fp8, bench_kineto

from primus_turbo.pytorch.cco import _fused_dispatch


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
        ref_rank_prefix_matrix,
        ref_channel_prefix_matrix,
        ref_send_head,
        ref_num_tokens_per_rank,
        ref_num_tokens_per_expert,
        ref_is_token_in_rank,
        num_moe_recv_tokens,
        recv_num_tokens_per_expert,
        ref_recv_x,
        ref_recv_topk_idx,
        ref_recv_topk_weights,
    ) = _fused_dispatch(
        x, group.group_name, topk_idx, topk_weights, num_experts, num_sms=num_sms
    )
    group.barrier()

    time.sleep(1)

    num_moe_recv_tokens = num_moe_recv_tokens.item()
    recv_num_tokens_per_expert_list = recv_num_tokens_per_expert.tolist()

    assert (
        gbl_num_tokens_per_expert.view(num_ranks, -1)[rank].tolist()
        == recv_num_tokens_per_expert_list
    )

    assert torch.allclose(ref_num_tokens_per_rank, num_tokens_per_rank)
    assert torch.allclose(ref_num_tokens_per_expert, num_tokens_per_expert)
    assert torch.allclose(ref_is_token_in_rank, is_token_in_rank)

    # check recv_x
    # check_data(ref_recv_x[:num_moe_recv_tokens, :], ref_rank_prefix_matrix)

    ref_recv_topk_idx = ref_recv_topk_idx[:num_moe_recv_tokens]
    ref_recv_topk_weights = ref_recv_topk_weights[:num_moe_recv_tokens]

    # Check `topk_idx`
    # assert (
    #     ref_recv_topk_idx.eq(-1)
    #     | ((ref_recv_topk_idx >= 0) & (ref_recv_topk_idx < (num_experts // num_ranks)))
    # ).sum().item() == ref_recv_topk_idx.numel()
    # for i, count in enumerate(recv_num_tokens_per_expert_list):
    #     assert ref_recv_topk_idx.eq(i).sum().item() == count

    # # Check `topk_weights`
    # ref_recv_topk_weights[ref_recv_topk_idx.eq(-1)] = ref_recv_topk_weights.amax(
    #     dim=1, keepdim=True
    # ).expand_as(ref_recv_topk_weights)[ref_recv_topk_idx.eq(-1)]
    # check_data(ref_recv_topk_weights, ref_rank_prefix_matrix)

    # benchmark performance
    nvl_recv_bytes = num_moe_recv_tokens * hidden * 2

    kwargs = {"x": x, "group_name": group.group_name, "topk_idx": topk_idx,
              "topk_weights": topk_weights, "num_experts": num_experts, "num_sms": num_sms}
    t, notify_t, layout_t = bench_kineto(lambda: _fused_dispatch(
        **kwargs), ("intranode::dispatch", "intranode::notify_dispatch", "get_dispatch_layout"))

    if local_rank == 0:
        print(f"SMs {num_sms}: {nvl_recv_bytes / 1e9 / t:.2f} GB/s (NVL), {t * 1e6:.2f} us + {notify_t * 1e6:.2f} us + {layout_t * 1e6:.2f} us",
              flush=True,
              )


def test_loop(local_rank: int, num_local_ranks: int, args: argparse.Namespace):
    rank, num_ranks, group = init_dist(local_rank, num_local_ranks)

    torch.manual_seed(rank)

    for i in (64,):
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
