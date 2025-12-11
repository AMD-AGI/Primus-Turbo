###############################################################################
# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Modification Copyright© 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from typing import Optional, Tuple, Union

import deep_ep
import torch
import torch.distributed as dist
import torch.distributed.distributed_c10d as c10d

import primus_turbo.pytorch as turbo

BufferType = Union[turbo.deep_ep.Buffer, deep_ep.Buffer]

_buffer: BufferType = None


def get_hidden_bytes(x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]) -> int:
    """Calculate the number of hidden bytes for a tensor.

    Args:
        x (torch.Tensor): Input tensor or FP8 input tensor with scale tensor

    Returns:
        int: Number of hidden bytes
    """
    inp = x if isinstance(x, torch.Tensor) else x[0]
    return inp.size(1) * max(inp.element_size(), 2)


def get_buffer(
    group: dist.ProcessGroup,
    hidden_bytes: int,
    BufferClass: BufferType,
    extra_kwargs: dict,
    autotune_config=None,
) -> BufferType:
    global _buffer

    num_nvl_bytes, num_rdma_bytes = 0, 0
    for config in autotune_config or (
        BufferClass.get_dispatch_config(group.size()),
        BufferClass.get_combine_config(group.size()),
    ):
        num_nvl_bytes = max(config.get_nvl_buffer_size_hint(hidden_bytes, group.size()), num_nvl_bytes)
        try:
            num_rdma_bytes = max(config.get_rdma_buffer_size_hint(hidden_bytes, group.size()), num_rdma_bytes)
        except:
            pass

    if (
        _buffer is None
        or _buffer.group != group
        or _buffer.num_nvl_bytes < num_nvl_bytes
        or _buffer.num_rdma_bytes < num_rdma_bytes
    ):
        _buffer = BufferClass(
            group,
            num_nvl_bytes,
            num_rdma_bytes,
            **extra_kwargs,
        )
    return _buffer


def _moe_dispatch_backend_impl(
    buffer,
    x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
    handle: Optional[Tuple] = None,
    topk_idx: Optional[torch.Tensor] = None,
    token_weights: Optional[torch.Tensor] = None,
    num_experts: Optional[int] = None,
    async_finish: bool = False,
    allocate_on_comm_stream: bool = False,
    num_worst_tokens: int = 0,
):

    # forward dispatch need to calculate layout
    if handle is None:
        assert topk_idx is not None
        assert token_weights is not None
        (
            num_tokens_per_rank,
            num_tokens_per_rdma_rank,
            num_tokens_per_expert,
            is_token_in_rank,
            previous_event,
        ) = buffer.get_dispatch_layout(
            topk_idx,
            num_experts,
            previous_event=previous_event,
            async_finish=async_finish,
            allocate_on_comm_stream=allocate_on_comm_stream,
        )

        (
            recv_x,
            recv_token_indices,
            recv_token_probs,
            tokens_per_expert,
            handle,
            after_event_overlap,
        ) = buffer.dispatch(
            x,
            topk_idx=topk_idx,
            topk_weights=token_weights,
            num_tokens_per_rank=num_tokens_per_rank,
            num_tokens_per_rdma_rank=num_tokens_per_rdma_rank,
            is_token_in_rank=is_token_in_rank,
            num_tokens_per_expert=num_tokens_per_expert,
            previous_event=previous_event,
            async_finish=async_finish,
            allocate_on_comm_stream=allocate_on_comm_stream,
            num_worst_tokens=num_worst_tokens,
        )
    else:
        # backward dispatch use existing handle
        recv_x, recv_token_indices, recv_token_probs, tokens_per_expert, handle, after_event_overlap = (
            buffer.dispatch(
                x,
                handle=handle,
                previous_event=previous_event,
                async_finish=async_finish,
                allocate_on_comm_stream=allocate_on_comm_stream,
            )
        )

    return recv_x, recv_token_indices, recv_token_probs, tokens_per_expert, handle, after_event_overlap


def moe_dispatch_impl(
    x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
    group_name: str,
    handle: Optional[Tuple] = None,
    topk_idx: Optional[torch.Tensor] = None,
    token_weight: Optional[torch.Tensor] = None,
    num_experts: Optional[int] = None,
    async_finish=False,
    allocate_on_comm_stream=False,
    num_worst_tokens=0,
    backend: str = "turbo",
):
    group = c10d._resolve_process_group(group_name)
    if backend == "turbo":
        buffer = get_buffer(group, get_hidden_bytes(x), turbo.deep_ep.Buffer, {}, autotune_config=None)
    elif backend == "uccl":
        buffer = get_buffer(group, get_hidden_bytes(x), deep_ep.Buffer, {}, autotune_config=None)
    else:
        raise ValueError(f"Invalid backend: {backend}")

    return _moe_dispatch_backend_impl(
        buffer,
        x,
        handle,
        topk_idx,
        token_weight,
        num_experts,
        async_finish,
        allocate_on_comm_stream,
        num_worst_tokens,
    )
