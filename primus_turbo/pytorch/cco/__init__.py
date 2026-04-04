from typing import Optional, Tuple, Union

import torch
import torch.distributed as dist

import primus_turbo.pytorch._C as cpp_extensions

from primus_turbo.pytorch.cco.symm_mem import get_symm_mem_workspace

from primus_turbo.pytorch.cco.metadata_preprocessing_triton import metadata_preprocessing


def _fused_dispatch_with_permute(x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
                                 group: dist.ProcessGroup,
                                 routing_map: torch.Tensor,
                                 topk_idx: Optional[torch.Tensor] = None,
                                 topk_weights: Optional[torch.Tensor] = None,
                                 num_experts: Optional[int] = None,
                                 expert_alignment: int = 1,
                                 *,
                                 num_sms: int = None,
                                 handle=None,
                                 num_max_send_tokens: int = 2,
                                 ):
    x, x_scales = x if isinstance(x, tuple) else (x, None)
    assert x.dim() == 2

    cached_mode = handle is not None
    rank = group.rank()
    num_ranks = group.size()

    num_tokens, hidden_size = x.shape
    num_worst_tokens = num_tokens * num_ranks

    workspace_size = int(1e9) + num_worst_tokens * hidden_size * \
        num_ranks * x.element_size() + num_ranks * num_ranks * 4

    symm_mem = get_symm_mem_workspace(group, workspace_size)

    buffer_ptrs_dev: torch.Tensor = symm_mem.buffer_ptrs_dev
    barrier_signal_ptrs_dev: torch.Tensor = symm_mem.signal_pad_ptrs_dev

    assert topk_idx is not None

    num_topk = topk_idx.shape[1]
    num_permuted_tokens = num_worst_tokens * num_topk

    handle = metadata_preprocessing(routing_map,
                                    group,
                                    num_of_tokens_per_rank=num_tokens,
                                    num_ranks_per_node=num_ranks,
                                    num_experts_per_rank=num_experts // num_ranks,
                                    local_rank=rank,
                                    num_of_tokens_per_chunk=32,
                                    max_num_of_tokens=num_worst_tokens,
                                    num_permuted_tokens=num_permuted_tokens,
                                    pad_multiple=0,
                                    enable_permute=True,
                                    fuse_permute_dispatch=False,
                                    non_blocking=True)

    torch.cuda.synchronize()

    if cached_mode:
        pass
    else:
        assert topk_idx is not None and topk_weights is not None and num_experts is not None

        (num_tokens_per_rank,
         num_tokens_per_rdma_rank,
         num_tokens_per_expert,
         is_token_in_rank) = cpp_extensions.get_dispatch_layout(topk_idx, num_experts, num_ranks, 1)

        moe_recv_counter = torch.empty(1, dtype=torch.int32, device=x.device)
        moe_recv_expert_counter = torch.empty(
            num_experts // num_ranks, dtype=torch.int32, device=x.device)

        (permuted_x,
         recv_x_scales,
         recv_topk_idx,
         recv_topk_weights,
         moe_recv_counter,
         moe_recv_expert_counter,
         rank_prefix_matrix,
         channel_prefix_matrix,
         recv_channel_prefix_matrix,
         recv_src_idx,
         send_head) = cpp_extensions.intranode_dispatch_with_permute(x,
                                                                     x_scales,
                                                                     handle.row_id_map,
                                                                     topk_idx,
                                                                     topk_weights,
                                                                     num_tokens_per_rank,
                                                                     is_token_in_rank,
                                                                     num_tokens_per_expert,
                                                                     0,
                                                                     None,
                                                                     None,
                                                                     buffer_ptrs_dev,
                                                                     barrier_signal_ptrs_dev,
                                                                     moe_recv_counter,
                                                                     moe_recv_expert_counter,
                                                                     expert_alignment,
                                                                     num_worst_tokens,
                                                                     num_permuted_tokens,
                                                                     rank,
                                                                     num_ranks,
                                                                     num_sms,
                                                                     num_max_send_tokens)

    offset = rank_prefix_matrix.nbytes + num_sms // 2 * \
        num_ranks * torch.int32.itemsize * 3
    recv_x = symm_mem.get_buffer(
        rank, [num_worst_tokens, hidden_size], x.dtype, offset // x.element_size())
    offset += recv_x.nbytes
    # offset += recv_x.nbytes
    # recv_topk_idx = symm_mem.get_buffer(
    #     rank, recv_topk_idx.shape, recv_topk_idx.dtype, offset // topk_idx.element_size())
    # offset += recv_topk_idx.nbytes
    # recv_topk_weights = symm_mem.get_buffer(
    #     rank, recv_topk_weights.shape, recv_topk_weights.dtype, offset // recv_topk_weights.element_size())

    return (permuted_x,
            recv_x_scales,
            recv_topk_idx,
            recv_topk_weights,
            moe_recv_counter,
            moe_recv_expert_counter,
            rank_prefix_matrix,
            channel_prefix_matrix,
            recv_channel_prefix_matrix,
            recv_src_idx,
            send_head), handle, recv_x
