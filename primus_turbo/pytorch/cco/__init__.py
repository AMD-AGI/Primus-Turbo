from typing import NamedTuple, Optional, Tuple, Union

import torch
import torch.distributed as dist

import primus_turbo.pytorch._C as cpp_extensions
from primus_turbo.pytorch.cco.symm_mem import get_symm_mem_workspace

# Persistent ``recv_x`` buffers keyed by ``(device, dtype, rows, hidden)``.
#
# The C++ dispatch kernels need a ~3.5 GB scratch buffer for the permuted
# output. Allocating it fresh per call is functionally correct but triggers
# a PyTorch HIP caching-allocator race in overlap benchmarks: the allocator
# can mark an iteration's block free while a consumer gemm on a different
# stream is still reading it, leading to HSA_STATUS_ERROR_EXCEPTION (0x1016
# — memory aperture violation) a few dozen pipelined iterations in. Caching
# the buffer here keeps the storage alive for the lifetime of the process,
# eliminating the free/reuse race without changing the callers.
_RECV_X_CACHE: dict = {}


def _get_recv_x_buffer(device: torch.device, dtype: torch.dtype, rows: int, hidden: int) -> torch.Tensor:
    key = (device, dtype, rows, hidden)
    buf = _RECV_X_CACHE.get(key)
    if buf is None or buf.size(0) < rows or buf.size(1) != hidden:
        buf = torch.empty((rows, hidden), dtype=dtype, device=device)
        _RECV_X_CACHE[key] = buf
    return buf


def _fused_dispatch_permute(
    x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
    group: dist.ProcessGroup,
    group_tail_idx: torch.Tensor,
    topk_idx: Optional[torch.Tensor] = None,
    topk_weights: Optional[torch.Tensor] = None,
    num_experts: Optional[int] = None,
    expert_alignment: int = 1,
    *,
    num_sms: int = None,
    handle=None,
    num_max_send_tokens: int = 4,
):
    with torch.profiler.record_function("fused_dispatch_permute"):
        x, x_scales = x if isinstance(x, tuple) else (x, None)
        assert x.dim() == 2

        cached_mode = handle is not None
        rank = group.rank()
        num_ranks = group.size()

        num_tokens, hidden_size = x.shape
        num_worst_tokens = num_tokens * num_ranks

        num_experts_per_rank = num_experts // num_ranks
        num_channels = num_sms // 2

        workspace_size = (
            int(1e9)
            + num_worst_tokens * hidden_size * num_ranks * x.element_size()
            + num_ranks * num_ranks * 4
        )

        symm_mem = get_symm_mem_workspace(group, workspace_size)

        buffer_ptrs_dev: torch.Tensor = symm_mem.buffer_ptrs_dev
        barrier_signal_ptrs_dev: torch.Tensor = symm_mem.signal_pad_ptrs_dev

        assert num_experts is not None

        assert topk_idx is not None
        num_topk = topk_idx.shape[1]
        if cached_mode:
            raise NotImplementedError("cached fused_dispatch_permute handle is not supported yet")
        else:

            moe_recv_counter = torch.empty((1,), dtype=torch.int32, device=x.device)
            moe_recv_expert_counter_padded = torch.empty(
                (num_experts_per_rank,), dtype=torch.int32, device=x.device
            )

            # Phase 1+2: get_dispatch_layout + notify_dispatch_permute
            (
                num_tokens_per_rank,
                num_tokens_per_expert,
                is_token_in_rank,
                moe_recv_counter,
                moe_recv_expert_counter,
                rank_prefix_matrix,
                channel_prefix_matrix,
                expert_offsets,
                moe_recv_expert_counter_padded,
            ) = cpp_extensions.fused_dispatch_permute_preprocess(
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
                num_sms,
            )

            recv_x_buffer = _get_recv_x_buffer(x.device, x.dtype, num_worst_tokens * num_topk, hidden_size)

            # Phase 3: fused_dispatch_permute — receiver atomically materialises expert-major rows
            (permuted_x, recv_topk_idx, recv_topk_weights, dispatch_to_expert_map) = (
                cpp_extensions.fused_dispatch_permute(
                    x,
                    x_scales,
                    topk_idx,
                    topk_weights,
                    is_token_in_rank,
                    channel_prefix_matrix,
                    moe_recv_expert_counter,
                    buffer_ptrs_dev,
                    group_tail_idx,
                    num_worst_tokens,
                    num_worst_tokens * num_topk,
                    num_experts,
                    rank,
                    num_ranks,
                    num_sms,
                    num_max_send_tokens,
                    recv_x_buffer,
                )
            )

        total_permuted_tokens = num_worst_tokens * num_topk
        metadata_ints = num_channels * num_ranks * 3 + num_experts_per_rank + total_permuted_tokens
        offset = rank_prefix_matrix.nbytes + metadata_ints * 4
        recv_x = symm_mem.get_buffer(
            rank, [num_worst_tokens, hidden_size], x.dtype, storage_offset=offset // x.element_size()
        )
        handle = (
            recv_x,
            num_tokens_per_rank,
            num_tokens_per_expert,
            rank_prefix_matrix,
            channel_prefix_matrix,
            is_token_in_rank,
            None,
            None,
        )

        return (
            (
                permuted_x,
                None,
                recv_topk_idx,
                recv_topk_weights,
                moe_recv_counter,
                moe_recv_expert_counter,
                rank_prefix_matrix,
                channel_prefix_matrix,
                dispatch_to_expert_map,
                None,
            ),
            handle,
        )


def _expert_grouped_dispatch_permute(
    x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
    group: dist.ProcessGroup,
    group_tail_idx: torch.Tensor,
    topk_idx: Optional[torch.Tensor] = None,
    topk_weights: Optional[torch.Tensor] = None,
    num_experts: Optional[int] = None,
    num_experts_per_group: Optional[int] = None,
    expert_alignment: int = 1,
    *,
    num_sms: int = None,
    handle=None,
    num_max_send_tokens: int = 4,
):
    """Pipelined expert-major fused dispatch + permute.

    Extends ``_fused_dispatch_permute`` by sending tokens in expert-group
    order. The producer sends tokens whose primary local expert lies in
    group ``g`` during phase ``g``; a grouped-GEMM consumer polling
    ``group_tail_idx[e]`` can therefore start on experts of group 0 as soon
    as phase 0 completes, overlapping the remaining dispatch phases with
    compute.

    ``num_experts_per_group`` controls the pipeline granularity. When it
    equals ``num_experts // num_ranks`` (single group) the kernel degenerates
    to a plain expert-major dispatch with identical timing to
    ``_fused_dispatch_permute``.
    """
    with torch.profiler.record_function("expert_grouped_dispatch_permute"):
        x, x_scales = x if isinstance(x, tuple) else (x, None)
        assert x.dim() == 2
        assert num_experts is not None
        assert topk_idx is not None

        cached_mode = handle is not None
        rank = group.rank()
        num_ranks = group.size()

        num_tokens, hidden_size = x.shape
        num_worst_tokens = num_tokens * num_ranks

        num_experts_per_rank = num_experts // num_ranks
        if num_experts_per_group is None:
            num_experts_per_group = num_experts_per_rank
        assert 1 <= num_experts_per_group <= num_experts_per_rank
        num_channels = num_sms // 2

        workspace_size = (
            int(1e9)
            + num_worst_tokens * hidden_size * num_ranks * x.element_size()
            + num_ranks * num_ranks * 4
        )

        symm_mem = get_symm_mem_workspace(group, workspace_size)

        buffer_ptrs_dev: torch.Tensor = symm_mem.buffer_ptrs_dev
        barrier_signal_ptrs_dev: torch.Tensor = symm_mem.signal_pad_ptrs_dev

        num_topk = topk_idx.shape[1]
        if cached_mode:
            raise NotImplementedError("cached expert_grouped_dispatch_permute handle is not supported yet")

        moe_recv_counter = torch.empty((1,), dtype=torch.int32, device=x.device)
        moe_recv_expert_counter_padded = torch.empty(
            (num_experts_per_rank,), dtype=torch.int32, device=x.device
        )

        (
            num_tokens_per_rank,
            num_tokens_per_expert,
            is_token_in_rank,
            moe_recv_counter,
            moe_recv_expert_counter,
            rank_prefix_matrix,
            channel_prefix_matrix,
            expert_offsets,
            moe_recv_expert_counter_padded,
        ) = cpp_extensions.fused_dispatch_permute_preprocess(
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
            num_sms,
        )

        recv_x_buffer = _get_recv_x_buffer(x.device, x.dtype, num_worst_tokens * num_topk, hidden_size)

        (permuted_x, recv_topk_idx, recv_topk_weights, dispatch_to_expert_map) = (
            cpp_extensions.expert_grouped_dispatch_permute(
                x,
                x_scales,
                topk_idx,
                topk_weights,
                is_token_in_rank,
                channel_prefix_matrix,
                moe_recv_expert_counter,
                buffer_ptrs_dev,
                group_tail_idx,
                num_worst_tokens,
                num_worst_tokens * num_topk,
                num_experts,
                num_experts_per_group,
                rank,
                num_ranks,
                num_sms,
                num_max_send_tokens,
                recv_x_buffer,
            )
        )

        total_permuted_tokens = num_worst_tokens * num_topk
        metadata_ints = num_channels * num_ranks * 3 + num_experts_per_rank + total_permuted_tokens
        offset = rank_prefix_matrix.nbytes + metadata_ints * 4
        recv_x = symm_mem.get_buffer(
            rank,
            [num_worst_tokens, hidden_size],
            x.dtype,
            storage_offset=offset // x.element_size(),
        )
        handle = (
            recv_x,
            num_tokens_per_rank,
            num_tokens_per_expert,
            rank_prefix_matrix,
            channel_prefix_matrix,
            is_token_in_rank,
            None,
            None,
        )

        return (
            (
                permuted_x,
                None,
                recv_topk_idx,
                recv_topk_weights,
                moe_recv_counter,
                moe_recv_expert_counter,
                rank_prefix_matrix,
                channel_prefix_matrix,
                dispatch_to_expert_map,
                None,
            ),
            handle,
        )
