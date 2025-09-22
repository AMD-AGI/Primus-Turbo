###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################


from typing import List, Optional, Tuple, Union

import torch
import torch.distributed as dist

from primus_turbo.pytorch.deep_ep import Buffer, Config
from primus_turbo.pytorch.deep_ep.utils import EventHandle, EventOverlap

__all__ = [
    "init_deepep_buffer",
    "deepep_is_initialized",
    "DeepEPDispatch",
    "DeepEPCombine",
    "deepep_dispatch",
    "deepep_combine",
]

_buffer = None


def init_deepep_buffer(
    group: dist.ProcessGroup,
    hidden_bytes: int,
    use_default_stream_as_comm_stream: bool = True,
    autotune_config: Optional[Tuple[Config, Config]] = None,
    num_use_cu: Optional[int] = None,
) -> Buffer:
    global _buffer

    if num_use_cu:
        Buffer.set_num_sms(num_use_cu)

    num_nvl_bytes, num_rdma_bytes = 0, 0
    ep_size = group.size()
    dispatch_config, combine_config = autotune_config or (
        Buffer.get_dispatch_config(ep_size),
        Buffer.get_combine_config(ep_size),
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
        or _buffer.group != group
        or _buffer.num_nvl_bytes < num_nvl_bytes
        or _buffer.num_rdma_bytes < num_rdma_bytes
    ):
        _buffer = Buffer(
            group,
            num_nvl_bytes,
            num_rdma_bytes,
            use_default_stream_as_comm_stream=use_default_stream_as_comm_stream,
        )
    return _buffer


def deepep_is_initialized():
    global _buffer
    return _buffer is not None


class DeepEPDispatch(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx,
        x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        token_indices: torch.Tensor,
        token_weights: torch.Tensor,
        num_experts: int,
        async_finish=False,
        allocate_on_comm_stream=False,
        use_cuda_num_token_per_expert: bool = True,
        num_worst_tokens: int = 0,
    ) -> Tuple[
        Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        torch.Tensor,
        torch.Tensor,
        Union[List, torch.Tensor],
        Tuple,
    ]:
        global _buffer
        assert deepep_is_initialized(), "Please check DeepEP buffer has been initialized."

        previous_event = None
        if async_finish:
            previous_event = EventOverlap(EventHandle())

        # Calculate layout before actual dispatch
        (
            num_tokens_per_rank,
            num_tokens_per_rdma_rank,
            num_tokens_per_expert,
            is_token_in_rank,
            previous_event,
        ) = _buffer.get_dispatch_layout(
            token_indices,
            num_experts,
            previous_event=previous_event,
            async_finish=async_finish,
            allocate_on_comm_stream=allocate_on_comm_stream,
        )

        # Do MoE dispatch
        # NOTES: the CPU will wait for GPU's signal to arrive,
        # so this is not compatible with CUDA graph
        (
            recv_x,
            recv_token_indices,
            recv_token_probs,
            tokens_per_expert,
            handle,
            after_event_overlap,
        ) = _buffer.dispatch(
            x,
            topk_idx=token_indices,
            topk_weights=token_weights,  # DeepEP only supports float32 probs
            num_tokens_per_rank=num_tokens_per_rank,
            num_tokens_per_rdma_rank=num_tokens_per_rdma_rank,
            is_token_in_rank=is_token_in_rank,
            num_tokens_per_expert=num_tokens_per_expert,
            previous_event=previous_event,
            async_finish=async_finish,
            allocate_on_comm_stream=allocate_on_comm_stream,
            num_recv_tokens_per_expert_as_cuda=use_cuda_num_token_per_expert,
            num_worst_tokens=num_worst_tokens,
        )

        # Make sure current stream is synchronized
        if async_finish:
            after_event_overlap.current_stream_wait()

        ctx.handle = handle
        ctx.async_finish = async_finish
        ctx.allocate_on_comm_stream = allocate_on_comm_stream

        if not use_cuda_num_token_per_expert:
            assert isinstance(tokens_per_expert, list)
            tokens_per_expert = torch.tensor(tokens_per_expert)

        return (recv_x, recv_token_indices, recv_token_probs, tokens_per_expert, handle)

    @staticmethod
    def backward(ctx, grad_output, grad_token_indices, grad_token_probs, grad_tokens_per_expert, grad_handle):
        global _buffer
        handle = ctx.handle
        previous_event = None
        if ctx.async_finish:
            previous_event = EventOverlap(EventHandle())
        grad_x, grad_token_probs, after_event = _buffer.combine(
            grad_output.contiguous(),
            handle,
            topk_weights=grad_token_probs.float(),
            previous_event=previous_event,
            async_finish=ctx.async_finish,
            allocate_on_comm_stream=ctx.allocate_on_comm_stream,
        )
        # Make sure current stream is synchronized
        if ctx.async_finish:
            after_event.current_stream_wait()
        return grad_x, None, grad_token_probs, None, None, None, None, None


class DeepEPCombine(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx, x: torch.Tensor, handle: Tuple, async_finish=False, allocate_on_comm_stream=False
    ) -> torch.Tensor:
        global _buffer
        assert deepep_is_initialized(), "Please check DeepEP buffer has been initialized."
        previous_event = None
        if async_finish:
            previous_event = EventOverlap(EventHandle())
        combined_x, _, after_event = _buffer.combine(
            x,
            handle=handle,
            async_finish=True,
            previous_event=previous_event,
            allocate_on_comm_stream=previous_event is not None,
        )
        # Make sure current stream is synchronized
        if async_finish:
            after_event.current_stream_wait()

        ctx.handle = handle

        ctx.async_finish = async_finish
        ctx.allocate_on_comm_stream = allocate_on_comm_stream
        return combined_x

    @staticmethod
    def backward(ctx, grad_x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]):
        global _buffer
        previous_event = None
        if ctx.async_finish:
            previous_event = EventOverlap(EventHandle())
        grad_x, _, _, _, _, after_event = _buffer.dispatch(
            grad_x,
            handle=ctx.handle,
            previous_event=previous_event,
            async_finish=ctx.async_finish,
            allocate_on_comm_stream=ctx.allocate_on_comm_stream,
        )
        # Make sure current stream is synchronized
        if ctx.async_finish:
            after_event.current_stream_wait()
        return grad_x, None, None, None


def deepep_dispatch(
    x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
    token_indices: torch.Tensor,
    token_weights: torch.Tensor,
    num_experts: int,
    async_finish=False,
    allocate_on_comm_stream=False,
    use_cuda_num_token_per_expert: bool = False,
    num_worst_tokens: int = 0,
):
    return DeepEPDispatch.apply(
        x,
        token_indices,
        token_weights,
        num_experts,
        async_finish,
        allocate_on_comm_stream,
        use_cuda_num_token_per_expert,
        num_worst_tokens,
    )


def deepep_combine(
    x,
    handle,
    async_finish=False,
    allocate_on_comm_stream=False,
):
    return DeepEPCombine.apply(x, handle, async_finish, allocate_on_comm_stream)
