from typing import NamedTuple, Optional, Tuple, Union

import torch
import torch.distributed as dist

import primus_turbo.pytorch._C as cpp_extensions

from primus_turbo.pytorch.cco.symm_mem import get_symm_mem_workspace
from primus_turbo.pytorch.cco.pipeline_ep import (
    PipelineEPConfig,
    PipelineEPHandle,
    pipeline_ep_preprocess,
    pipeline_ep_dispatch,
    pipeline_ep_recv,
    pipeline_ep_full,
    get_grouped_gemm_schedule,
)


def _fused_dispatch_permute(x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
                            group: dist.ProcessGroup,
                            topk_idx: Optional[torch.Tensor] = None,
                            topk_weights: Optional[torch.Tensor] = None,
                            num_experts: Optional[int] = None,
                            expert_alignment: int = 1,
                            *,
                            num_sms: int = None,
                            handle=None,
                            num_max_send_tokens: int = 1,
                            ):
    x, x_scales = x if isinstance(x, tuple) else (x, None)
    assert x.dim() == 2

    cached_mode = handle is not None
    rank = group.rank()
    num_ranks = group.size()

    num_tokens, hidden_size = x.shape
    num_worst_tokens = num_tokens * num_ranks

    num_experts_per_rank = num_experts // num_ranks
    num_channels = num_sms // 2

    workspace_size = (int(1e9)
                      + num_worst_tokens * hidden_size * num_ranks * x.element_size()
                      + num_ranks * num_ranks * 4)

    symm_mem = get_symm_mem_workspace(group, workspace_size)

    buffer_ptrs_dev: torch.Tensor = symm_mem.buffer_ptrs_dev
    barrier_signal_ptrs_dev: torch.Tensor = symm_mem.signal_pad_ptrs_dev

    assert num_experts is not None

    assert topk_idx is not None
    num_topk = topk_idx.shape[1]
    if cached_mode:
        raise NotImplementedError("cached fused_dispatch_permute handle is not supported yet")
    else:
        moe_recv_counter = torch.empty(
            (1,), dtype=torch.int32, device=x.device)
        moe_recv_expert_counter_padded = torch.empty(
            (num_experts_per_rank,), dtype=torch.int32, device=x.device)

        # Phase 1+2: get_dispatch_layout + notify_dispatch_permute
        (num_tokens_per_rank,
         num_tokens_per_expert,
         is_token_in_rank,
         moe_recv_counter,
         moe_recv_expert_counter,
         rank_prefix_matrix,
         channel_prefix_matrix,
         expert_offsets,
         moe_recv_expert_counter_padded) = cpp_extensions.fused_dispatch_permute_preprocess(
            topk_idx,
            buffer_ptrs_dev,
            barrier_signal_ptrs_dev,
            moe_recv_counter,
            moe_recv_expert_counter_padded,
            num_experts,
            expert_alignment,
            num_worst_tokens,
            rank,
            num_ranks,
            num_sms)

        # Phase 3: fused_dispatch_permute — receiver atomically materialises expert-major rows
        (permuted_x,
         recv_topk_idx,
         recv_topk_weights,
         dispatch_to_expert_map) = cpp_extensions.fused_dispatch_permute(
            x,
            x_scales,
            topk_idx,
            topk_weights,
            is_token_in_rank,
            channel_prefix_matrix,
            moe_recv_expert_counter,
            buffer_ptrs_dev,
            num_worst_tokens,
            num_worst_tokens * num_topk,
            num_experts,
            rank,
            num_ranks,
            num_sms,
            num_max_send_tokens)

    metadata_ints = max(num_ranks * num_experts_per_rank, num_channels * num_ranks * 3)
    offset = rank_prefix_matrix.nbytes + metadata_ints * 4
    recv_x = symm_mem.get_buffer(rank, [num_worst_tokens, hidden_size],
                                 x.dtype, storage_offset=offset // x.element_size())
    handle = (recv_x, num_tokens_per_rank, num_tokens_per_expert,
              rank_prefix_matrix, channel_prefix_matrix, is_token_in_rank, None, None)

    return ((permuted_x,
             None,
            recv_topk_idx,
            recv_topk_weights,
            moe_recv_counter,
            moe_recv_expert_counter,
            rank_prefix_matrix,
            channel_prefix_matrix,
            dispatch_to_expert_map,
            None),
            handle)
