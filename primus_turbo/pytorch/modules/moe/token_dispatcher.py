###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################


import os
import warnings
from abc import abstractmethod
from typing import Optional, Tuple

import torch
import torch.distributed as dist

import primus_turbo.pytorch as turbo
from primus_turbo.common.constants import ENV_EP_FORCE_CURRENT_STREAM
from primus_turbo.pytorch.deep_ep import Config
from primus_turbo.pytorch.kernels.moe.moe_dispatch_combine_impl import (
    clear_backend_instances,
    set_buffer_global_config,
)


class TokenDispatcher:
    def __init__(
        self,
        num_experts: int,
        router_topk: int,
        ep_group: dist.ProcessGroup,
        tp_group: Optional[dist.ProcessGroup],
        tp_ep_group: Optional[dist.ProcessGroup],
    ):

        self.ep_size = ep_group.size()
        # only use ep_group
        if tp_group is None and tp_ep_group is None:
            tp_group = dist.new_group([dist.get_rank()], backend=dist.get_backend(ep_group))
            tp_ep_group = ep_group
        else:
            assert tp_group and tp_ep_group, "tp_group or tp_ep_group is None"

        self.ep_group = ep_group
        self.tp_group = tp_group
        self.tp_ep_group = tp_ep_group

        self.ep_size = ep_group.size()
        self.tp_size = tp_group.size()
        self.tp_ep_size = self.ep_size * self.tp_size

        assert num_experts % self.ep_size == 0
        self.num_local_experts = num_experts // self.ep_size

        self.num_experts = num_experts * self.tp_size
        self.router_topk = router_topk * self.tp_size

    def token_dispatch(
        self,
        hidden_states: torch.Tensor,
        probs: torch.Tensor,
        routing_map: Optional[torch.Tensor] = None,
        indices: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden_states, probs = self._pre_dispatch(hidden_states, probs, routing_map, indices)
        dispatched_tokens, dispatched_probs = self._exec_dispatch(hidden_states, probs)
        dispatched_input, tokens_per_expert, permuted_probs = self._post_dispatch(
            dispatched_tokens, dispatched_probs
        )
        return dispatched_input, tokens_per_expert, permuted_probs

    def token_combine(self, hidden_states: torch.Tensor):
        output = self._pre_combine(hidden_states)
        combined_tokens = self._exec_combine(output)
        return self._post_combine(combined_tokens)

    @abstractmethod
    def _pre_dispatch(
        self,
        hidden_states: torch.Tensor,
        probs: torch.Tensor,
        routing_map: Optional[torch.Tensor] = None,
        indices: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    @abstractmethod
    def _exec_dispatch(self, hidden_states: torch.Tensor, probs: torch.Tensor):
        raise NotImplementedError

    @abstractmethod
    def _post_dispatch(
        self, hidden_states: torch.Tensor, probs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    @abstractmethod
    def _pre_combine(self, hidden_states: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def _exec_combine(self, hidden_states: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def _post_combine(self, hidden_states: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class DeepEPTokenDispatcher(TokenDispatcher):
    """
    Dispatch tokens to different experts, with backward pass to combine gradients back to the input.

    Two host-sync sources sit between dispatch and the first expert GEMM:
    DeepEP itself (CPU read of ``moe_recv_*_counter``) and ``moe_permute``
    (host copy of ``tokens_per_expert.sum()`` to size the permuted buffer).
    The ``deepep_num_worst_tokens`` / ``num_permuted_tokens`` pair removes
    them so the whole dispatch → permute → group-GEMM chain runs without a
    host sync (and is CUDA-graph capturable):

    1. Set ``deepep_num_worst_tokens > 0``: DeepEP allocates the worst-case
       receive buffer, returns ``dispatched_indices`` shaped
       ``[num_worst_tokens, num_topk]`` with ``-1`` padding, and emits no
       per-expert counter list (``tokens_per_expert`` returned by DeepEP is
       ``null``). This forces ``deepep_use_cuda_num_tokens_per_expert=True``
       so we read ``tokens_per_expert`` from the permute kernel (CUDA tensor)
       instead. Padding rows with ``-1`` are skipped by the cu permute kernel.
    2. Set ``num_permuted_tokens > 0``: ``moe_permute`` uses this caller-
       provided cap to size the permuted buffer, skipping the
       ``tokens_per_expert.sum().item()`` host copy.

    ``moe_permute`` itself derives ``num_dispatched_tokens`` on-device by
    scanning ``expert_map`` for non-padding rows and returns it as a
    1-element int32 CUDA tensor — no caller plumbing is needed. We forward
    that tensor straight into ``moe_unpermute``.

    Args:
        num_experts: the number of moe experts.
        router_topk: the number of experts to route to for each token.
        ep_group: the group to use for expert parallism.
        tp_group: the group to use for tensor parallism.
        tp_ep_group: the group to use for tensor-expert parallism.
        expert_capacity_factor: The capacity factor for each expert, None means no token will be dropped.
        permute_fusion: use permuate fusion kernel when ``permute_fusion`` is True.
        num_permuted_tokens: caller-provided upper bound on the number of permuted
            rows. ``> 0`` removes the ``tokens_per_expert.sum().item()`` host sync
            inside ``moe_permute`` (the cu kernel uses this as the allocation
            cap and emits an overflow flag if exceeded). ``0`` (default) falls
            back to the sync path. Requires ``deepep_use_cuda_num_tokens_per_expert=True``.
        deepep_use_comm_stream: When False, force all EP dispatch/combine kernels onto the current stream by setting PRIMUS_TURBO_EP_FORCE_CURRENT_STREAM=1.
        deepep_num_use_cu: number of cu deepep used.
        deepep_num_worst_tokens: ``> 0`` puts DeepEP in worst-case-allocation
            mode: dispatch returns a fixed ``[num_worst_tokens, ...]`` buffer
            with ``-1`` padding and no per-expert counter list, eliminating
            DeepEP's CPU read-back. Requires ``deepep_use_cuda_num_tokens_per_expert=True``.
            See ``DeepEP.dispatch`` for full details.
        deepep_use_cuda_num_tokens_per_expert: when True, ``token_dispatch``
            returns ``tokens_per_expert`` as a CUDA tensor (sourced from the
            permute kernel) instead of a host tensor. Required for the nosync
            path; also avoids the CPU sync in turbo group-gemm.
        deepep_autotune_config: use autotuned DeepEP config to initialize DeepEP buffer for better performance.

    """

    def __init__(
        self,
        num_experts: int,
        router_topk: int,
        ep_group: dist.ProcessGroup,
        tp_group: Optional[dist.ProcessGroup] = None,
        tp_ep_group: Optional[dist.ProcessGroup] = None,
        expert_capacity_factor: Optional[float] = None,
        permute_fusion: bool = False,
        num_permuted_tokens: int = 0,
        deepep_async_finish: bool = True,
        deepep_allocate_on_comm_stream: bool = True,
        deepep_use_comm_stream: bool = False,
        deepep_num_use_cu: int = 32,
        deepep_num_worst_tokens: int = 0,
        deepep_use_cuda_num_tokens_per_expert: Optional[bool] = False,
        deepep_autotune_config: Optional[Config] = None,
    ):
        super().__init__(num_experts, router_topk, ep_group, tp_group, tp_ep_group)

        if deepep_num_worst_tokens > 0 and not deepep_use_cuda_num_tokens_per_expert:
            raise ValueError(
                "Please set deepep_use_cuda_num_tokens_per_expert=True when use deepep_num_worst_tokens"
            )

        if num_permuted_tokens > 0 and not deepep_use_cuda_num_tokens_per_expert:
            # The host sync in moe_permute is only one of two; keeping
            # tokens_per_expert on CPU re-introduces the other one (DeepEP's
            # counter read-back). Surface this early so users don't silently
            # leave a sync in place.
            raise ValueError(
                "num_permuted_tokens > 0 requires deepep_use_cuda_num_tokens_per_expert=True "
                "to actually remove the host sync."
            )

        if not deepep_use_comm_stream and os.environ.get(ENV_EP_FORCE_CURRENT_STREAM) != "1":
            clear_backend_instances()
            os.environ[ENV_EP_FORCE_CURRENT_STREAM] = "1"

        self.capacity_factor = expert_capacity_factor

        # permute
        self.permute_fusion = permute_fusion
        self.num_permuted_tokens = num_permuted_tokens

        # deepep
        self.deepep_async_finish = deepep_async_finish
        self.deepep_allocate_on_comm_stream = deepep_allocate_on_comm_stream
        self.deepep_use_cuda_num_tokens_per_expert = deepep_use_cuda_num_tokens_per_expert
        self.deepep_num_worst_tokens = deepep_num_worst_tokens

        set_buffer_global_config(
            num_use_cu=deepep_num_use_cu,
            autotune_config=deepep_autotune_config,
        )

    def _pre_dispatch(self, hidden_states, probs, routing_map=None, token_indices=None):
        self.hidden_shape = hidden_states.shape

        # reshape tokens, organize probs to [num_local_tokens, world_size, num_local_experts]
        hidden_states = hidden_states.view(-1, self.hidden_shape[-1])
        num_tokens = hidden_states.shape[0]

        probs = (
            probs.reshape(num_tokens, self.ep_size, 1, self.num_local_experts)
            .expand(-1, -1, self.tp_size, -1)
            .reshape(num_tokens, self.tp_ep_size, self.num_local_experts)
        ).contiguous()

        probs = probs.reshape(num_tokens, self.num_experts)

        # 1. token_indices is None, probs is unsorted with shape [num_tokens, num_experts]
        # call topk to get token_idx and token_probs
        if token_indices is None:
            token_probs, token_indices = torch.topk(probs, self.router_topk, dim=-1)
        else:
            # 2. token_indices is not None
            # call gather to get token_probs if token_probs unsorted, otherwise skip
            token_probs = probs.gather(1, token_indices)

        self.token_indices = token_indices

        # Mask the indices of dropped tokens with -1
        if self.capacity_factor is not None:
            mask = token_probs == 0
            self.token_indices = self.token_indices.masked_fill(mask, -1)

        return hidden_states, token_probs

    def _exec_dispatch(self, hidden_states, token_probs):
        # DeepEP only supports float32 probs
        if token_probs.dtype != torch.float32:
            if token_probs.dtype in [torch.bfloat16, torch.float16]:
                warnings.warn("DeepEP only supports float32 probs!")
            token_probs = token_probs.float()  # downcast or upcast

        hidden_states, dispatched_indices, dispatched_probs, tokens_per_expert, handle = (
            turbo.ops.moe_dispatch(
                hidden_states,
                token_indices=self.token_indices,
                token_probs=token_probs,
                num_experts=self.num_experts,
                group=self.tp_ep_group,
                async_finish=self.deepep_async_finish,
                allocate_on_comm_stream=self.deepep_allocate_on_comm_stream,
                num_worst_tokens=self.deepep_num_worst_tokens,
            )
        )

        self.handle = handle
        self.tokens_per_expert = tokens_per_expert
        self.dispatched_indices = dispatched_indices

        return hidden_states, dispatched_probs

    def _post_dispatch(self, hidden_states, dispatched_probs):
        # ``moe_permute`` accepts ``num_permuted_tokens`` as a non-negative
        # caller-provided cap (no host sync) or ``-1`` (sync via
        # ``tokens_per_expert.sum().item()``). Translate the dispatcher's
        # ``0 = unspecified`` convention to the op's ``-1``.
        num_permuted_tokens = self.num_permuted_tokens if self.num_permuted_tokens > 0 else -1

        self.hidden_shape_before_permute = hidden_states.shape
        assert dispatched_probs.dtype == torch.float32, "DeepEP only supports float32 probs"

        # ``moe_permute`` produces ``num_dispatched_token_tensor`` itself by
        # scanning ``dispatched_indices`` for ``-1`` padding on-device — we
        # cache it on ``self`` so ``_pre_combine`` can hand the very same
        # tensor to ``moe_unpermute`` without a host round-trip.
        (
            hidden_states,
            self.row_id_map,
            tokens_per_expert,
            _,
            self.num_dispatched_token_tensor,
            _,
            permuted_probs,
        ) = turbo.ops.moe_permute(
            hidden_states,
            self.dispatched_indices,
            num_local_experts=self.num_local_experts,
            num_topk=self.router_topk,
            num_permuted_tokens=num_permuted_tokens,
            probs=dispatched_probs,
        )

        if not self.deepep_use_cuda_num_tokens_per_expert:
            if self.tokens_per_expert is not None and self.tokens_per_expert.numel() > 0:
                tokens_per_expert = self.tokens_per_expert
            else:
                tokens_per_expert = tokens_per_expert.cpu()

        self.tokens_per_expert = None
        return hidden_states, tokens_per_expert, permuted_probs

    def _pre_combine(self, hidden_states):
        hidden_states, _ = turbo.ops.moe_unpermute(
            hidden_states,
            self.row_id_map,
            self.num_dispatched_token_tensor,
            restore_shape=self.hidden_shape_before_permute,
            num_local_experts=self.num_local_experts,
        )
        return hidden_states

    def _exec_combine(self, hidden_states):
        hidden_states = turbo.ops.moe_combine(
            hidden_states,
            self.tp_ep_group,
            self.handle,
            async_finish=self.deepep_async_finish,
            allocate_on_comm_stream=self.deepep_allocate_on_comm_stream,
        )
        # Release the handle after combine operation
        self.handle = None
        return hidden_states

    def _post_combine(self, hidden_states):
        return hidden_states.view(self.hidden_shape)
