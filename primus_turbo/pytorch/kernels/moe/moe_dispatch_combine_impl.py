###############################################################################
# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Modification Copyright© 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from typing import Optional, Tuple, Union

import deep_ep as uccl_ep
import torch
import torch.distributed as dist

import primus_turbo.pytorch.deep_ep as turbo_ep
from primus_turbo.pytorch.core.backend import BackendType

BufferType = Union[turbo_ep.Buffer, uccl_ep.Buffer]
ConfigType = Union[turbo_ep.Config, uccl_ep.Config]
EventHandleType = Union[turbo_ep.utils.EventHandle, uccl_ep.EventHandle]
EventOverlapType = Union[turbo_ep.utils.EventOverlap, uccl_ep.EventOverlap]

_buffer: Optional[BufferType] = None
_buffer_config: Tuple = None


def get_hidden_bytes(x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]) -> int:
    """Calculate the number of hidden bytes for a tensor.

    Args:
        x (torch.Tensor): Input tensor or FP8 input tensor with scale tensor

    Returns:
        int: Number of hidden bytes
    """
    inp = x if isinstance(x, torch.Tensor) else x[0]
    return inp.size(1) * max(inp.element_size(), 2)


def get_backend_classes(
    backend: Optional[BackendType] = None,
) -> Tuple[BufferType, EventOverlapType, EventHandleType]:
    if backend is None:
        return turbo_ep.Buffer, turbo_ep.utils.EventOverlap, turbo_ep.utils.EventHandle
    elif backend == BackendType.UCCL:
        return uccl_ep.Buffer, uccl_ep.EventOverlap, uccl_ep.EventHandle
    else:
        raise ValueError(f"Invalid backend: {backend.name}")


def set_buffer_global_config(
    num_use_cu: int = 32,
    autotune_config: Optional[Tuple[ConfigType, ConfigType]] = None,
):
    global _buffer_config
    _buffer_config = (num_use_cu, autotune_config)


def get_buffer(
    group: dist.ProcessGroup,
    hidden_bytes: int,
    BufferClass: BufferType,
    extra_kwargs: dict,
) -> BufferType:
    global _buffer, _buffer_config

    num_nvl_bytes, num_rdma_bytes = 0, 0
    ep_size = group.size()

    num_use_cu, autotune_config = _buffer_config
    BufferClass.set_num_sms(num_use_cu)

    dispatch_config, combine_config = autotune_config or (
        BufferClass.get_dispatch_config(ep_size),
        BufferClass.get_combine_config(ep_size),
    )

    for config in (dispatch_config, combine_config):
        num_nvl_bytes = max(config.get_nvl_buffer_size_hint(hidden_bytes, group.size()), num_nvl_bytes)
        try:
            num_rdma_bytes = max(config.get_rdma_buffer_size_hint(hidden_bytes, group.size()), num_rdma_bytes)
        except:
            pass

    # Allocate buffer if not existed or not enough buffer size
    if (
        _buffer is None
        or not isinstance(_buffer, BufferClass)
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


def _moe_dispatch_multiple_backends_impl(
    buffer: BufferType,
    EventOverlapClass: EventOverlapType,
    EventHandleClass: EventHandleType,
    x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
    handle: Optional[Tuple] = None,
    topk_idx: Optional[torch.Tensor] = None,
    token_weights: Optional[torch.Tensor] = None,
    num_experts: Optional[int] = None,
    async_finish: bool = False,
    allocate_on_comm_stream: bool = False,
    num_worst_tokens: int = 0,
):
    previous_event = None
    if async_finish:
        previous_event = EventOverlapClass(EventHandleClass())

    # forward dispatch need to calculate layout
    if handle is None:
        assert topk_idx is not None
        assert token_weights is not None
        (
            num_tokens_per_rank,
            num_tokens_per_rdma_rank,
            num_tokens_per_expert,
            is_token_in_rank,
            event,
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
            previous_event=event,
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

    # Make sure current stream is synchronized
    if async_finish:
        after_event_overlap.current_stream_wait()

    return recv_x, recv_token_indices, recv_token_probs, tokens_per_expert, handle


def _moe_combine_multiple_backends_impl(
    buffer: BufferType,
    EventOverlapClass: EventOverlapType,
    EventHandleClass: EventHandleType,
    x: torch.Tensor,
    handle: Tuple,
    topk_weights: Optional[torch.Tensor] = None,
    async_finish: bool = False,
    allocate_on_comm_stream: bool = False,
):
    previous_event = None
    if async_finish:
        previous_event = EventOverlapClass(EventHandleClass())

    combined_x, combined_topk_weights, after_event_overlap = buffer.combine(
        x,
        handle=handle,
        topk_weights=None if topk_weights is None else topk_weights.float(),
        async_finish=async_finish,
        allocate_on_comm_stream=allocate_on_comm_stream,
        previous_event=previous_event,
    )

    # Make sure current stream is synchronized
    if async_finish:
        after_event_overlap.current_stream_wait()

    return combined_x, combined_topk_weights


def moe_dispatch_impl(
    x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
    group: dist.ProcessGroup,
    handle: Optional[Tuple] = None,
    topk_idx: Optional[torch.Tensor] = None,
    token_weight: Optional[torch.Tensor] = None,
    num_experts: Optional[int] = None,
    async_finish=False,
    allocate_on_comm_stream=False,
    num_worst_tokens=0,
    default_backend: Optional[BackendType] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Tuple]:
    BuferCls, EventOverlapCls, EventHandleCls = get_backend_classes(default_backend)
    buffer = get_buffer(group, get_hidden_bytes(x), BuferCls, {})

    return _moe_dispatch_multiple_backends_impl(
        buffer,
        EventOverlapCls,
        EventHandleCls,
        x,
        handle,
        topk_idx,
        token_weight,
        num_experts,
        async_finish,
        allocate_on_comm_stream,
        num_worst_tokens,
    )


def moe_combine_impl(
    x: torch.Tensor,
    group: dist.ProcessGroup,
    handle: Tuple,
    topk_weights: Optional[torch.Tensor] = None,
    async_finish: bool = False,
    allocate_on_comm_stream: bool = False,
    default_backend: Optional[BackendType] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    BuferCls, EventOverlapCls, EventHandleCls = get_backend_classes(default_backend)
    buffer = get_buffer(group, get_hidden_bytes(x), BuferCls, {})
    return _moe_combine_multiple_backends_impl(
        buffer,
        EventOverlapCls,
        EventHandleCls,
        x,
        handle,
        topk_weights,
        async_finish,
        allocate_on_comm_stream,
    )
