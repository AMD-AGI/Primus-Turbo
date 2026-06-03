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
        if tp_group is None and tp_ep_group is None:
            # No-TP: reuse ep_group instead of leaking a per-rank singleton group.
            self.tp_group = None
            self.tp_ep_group = ep_group
            self.tp_size = 1
        else:
            assert tp_group and tp_ep_group, "tp_group or tp_ep_group is None"
            self.tp_group = tp_group
            self.tp_ep_group = tp_ep_group
            self.tp_size = tp_group.size()

        self.ep_group = ep_group
        self.tp_ep_size = self.ep_size * self.tp_size

        assert num_experts % self.ep_size == 0
        self.num_local_experts = num_experts // self.ep_size

        self.tp_expanded_num_experts = num_experts * self.tp_size
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
    """Dispatch tokens to experts (autograd combines gradients on backward).

    ``tokens_per_expert`` comes from ``moe_permute`` so the count reflects
    ``pad_multiple`` padding.

    Fully nosync / CUDA-graph capturable requires all three:
    ``deepep_num_worst_tokens > 0``, ``permute_max_token_num > 0``,
    ``deepep_use_cuda_num_tokens_per_expert=True``.

    Args:
        expert_capacity_factor: caller must pre-zero probs of routes to drop;
            this class only translates zero-prob into DeepEP's ``-1`` sentinel
            (mirrors Megatron's ``_DeepepManager``; capacity itself is upstream).
        permute_fusion: deprecated; ``moe_permute`` is always fused.
        pad_multiple: pad per-expert permuted count to this multiple
            (set to grouped-GEMM ``BLOCK_M``).
        permute_max_token_num: caller-provided cap; ``> 0`` removes the sum-item host sync.
        deepep_use_comm_stream: when False, pin EP kernels to the current stream.
        deepep_num_worst_tokens: ``> 0`` enables worst-case allocation mode.
        deepep_use_cuda_num_tokens_per_expert: keep ``tokens_per_expert`` on device.
    """

    # C++ helper caches the env once per process; first dispatcher wins, later
    # conflicting flags only warn (must not silently swap the cached backend).
    _ep_force_current_stream_locked: bool = False

    def __init__(
        self,
        num_experts: int,
        router_topk: int,
        ep_group: dist.ProcessGroup,
        tp_group: Optional[dist.ProcessGroup] = None,
        tp_ep_group: Optional[dist.ProcessGroup] = None,
        expert_capacity_factor: Optional[float] = None,
        permute_fusion: Optional[bool] = None,
        pad_multiple: int = 0,
        permute_max_token_num: int = -1,
        deepep_async_finish: bool = True,
        deepep_allocate_on_comm_stream: bool = True,
        deepep_use_comm_stream: bool = False,
        deepep_num_use_cu: int = 32,
        deepep_num_worst_tokens: int = 0,
        deepep_use_cuda_num_tokens_per_expert: Optional[bool] = False,
        deepep_autotune_config: Optional[Config] = None,
    ):
        super().__init__(num_experts, router_topk, ep_group, tp_group, tp_ep_group)

        if permute_fusion is not None:
            warnings.warn(
                "`permute_fusion` is deprecated and ignored: moe_permute is always "
                "fused now. Drop the argument to silence this warning.",
                DeprecationWarning,
                stacklevel=2,
            )

        if not deepep_use_comm_stream:
            if DeepEPTokenDispatcher._ep_force_current_stream_locked:
                if os.environ.get(ENV_EP_FORCE_CURRENT_STREAM) != "1":
                    warnings.warn(
                        "deepep_use_comm_stream=False requested on a later dispatcher, but a "
                        "prior dispatcher already locked the current-stream setting. The C++ "
                        "helper caches the env value once per process — set "
                        f"{ENV_EP_FORCE_CURRENT_STREAM}=1 before constructing any dispatcher "
                        "to apply this setting consistently.",
                        stacklevel=2,
                    )
            else:
                if os.environ.get(ENV_EP_FORCE_CURRENT_STREAM) != "1":
                    clear_backend_instances()
                    os.environ[ENV_EP_FORCE_CURRENT_STREAM] = "1"
                DeepEPTokenDispatcher._ep_force_current_stream_locked = True

        self.capacity_factor = expert_capacity_factor

        self.pad_multiple = pad_multiple
        self.permute_max_token_num = permute_max_token_num

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

        hidden_states = hidden_states.view(-1, self.hidden_shape[-1])
        num_tokens = hidden_states.shape[0]

        # Inputs must be in the pre-TP-expansion expert space; the reshape below
        # replicates across tp_size, so already-expanded inputs silently corrupt.
        original_num_experts = self.ep_size * self.num_local_experts
        assert probs.dim() == 2 and probs.shape == (num_tokens, original_num_experts), (
            f"probs must have shape [num_tokens={num_tokens}, "
            f"ep_size*num_local_experts={original_num_experts}], got {tuple(probs.shape)}"
        )
        if routing_map is not None:
            assert routing_map.shape == (num_tokens, original_num_experts), (
                f"routing_map must have shape [num_tokens={num_tokens}, "
                f"ep_size*num_local_experts={original_num_experts}], "
                f"got {tuple(routing_map.shape)}"
            )
        if token_indices is not None:
            assert token_indices.dim() == 2 and token_indices.shape[0] == num_tokens, (
                f"token_indices must be 2D with shape[0]={num_tokens}, " f"got {tuple(token_indices.shape)}"
            )

        probs = (
            probs.reshape(num_tokens, self.ep_size, 1, self.num_local_experts)
            .expand(-1, -1, self.tp_size, -1)
            .reshape(num_tokens, self.tp_expanded_num_experts)
        ).contiguous()

        if token_indices is not None:
            if self.tp_size > 1:
                # Remap original-expert indices into the TP-replicated layout.
                ep_idx = token_indices // self.num_local_experts
                local_e = token_indices % self.num_local_experts
                base = ep_idx * (self.tp_size * self.num_local_experts) + local_e
                tp_offsets = (
                    torch.arange(self.tp_size, device=token_indices.device, dtype=token_indices.dtype)
                    * self.num_local_experts
                )
                token_indices = (base.unsqueeze(-1) + tp_offsets).reshape(num_tokens, -1)
            token_probs = probs.gather(1, token_indices)
        elif routing_map is not None:
            warnings.warn(
                "DeepEP only accepts the topk_idx format ([num_tokens, router_topk]); "
                "converting from routing_map incurs an extra TP expand + stable sort + "
                "gather per dispatch. For hot paths, precompute token_indices in the "
                "router and pass it directly to avoid this overhead.",
                stacklevel=2,
            )
            routing_map = (
                routing_map.reshape(num_tokens, self.ep_size, 1, self.num_local_experts)
                .expand(-1, -1, self.tp_size, -1)
                .reshape(num_tokens, self.tp_expanded_num_experts)
            ).contiguous()
            # Stable descending sort puts True first; mask short rows with -1.
            _, sorted_expert_ids = routing_map.to(torch.int8).sort(dim=-1, descending=True, stable=True)
            token_indices = sorted_expert_ids[:, : self.router_topk]
            valid = routing_map.gather(1, token_indices).to(torch.bool)
            token_probs = probs.gather(1, token_indices).masked_fill(~valid, 0.0)
            token_indices = token_indices.masked_fill(~valid, -1)
        else:
            token_probs, token_indices = torch.topk(probs, self.router_topk, dim=-1)

        self.token_indices = token_indices

        if self.capacity_factor is not None:
            mask = token_probs == 0
            self.token_indices = self.token_indices.masked_fill(mask, -1)

        return hidden_states, token_probs

    def _exec_dispatch(self, hidden_states, token_probs):
        if token_probs.dtype != torch.float32:
            if token_probs.dtype in (torch.bfloat16, torch.float16):
                warnings.warn("DeepEP only supports float32 probs!")
            token_probs = token_probs.float()

        # Discard DeepEP's tokens_per_expert; moe_permute's count includes pad_multiple.
        hidden_states, dispatched_indices, dispatched_probs, _, handle = turbo.ops.moe_dispatch(
            hidden_states,
            token_indices=self.token_indices,
            token_probs=token_probs,
            num_experts=self.tp_expanded_num_experts,
            group=self.tp_ep_group,
            async_finish=self.deepep_async_finish,
            allocate_on_comm_stream=self.deepep_allocate_on_comm_stream,
            num_worst_tokens=self.deepep_num_worst_tokens,
        )

        self.handle = handle
        self.dispatched_indices = dispatched_indices

        return hidden_states, dispatched_probs

    def _post_dispatch(self, hidden_states, dispatched_probs):
        # Both dispatcher and moe_permute use -1 to mean "unspecified" (sync path).
        self.hidden_shape_before_permute = hidden_states.shape
        assert dispatched_probs.dtype == torch.float32, "DeepEP only supports float32 probs"

        # Cache num_dispatched_token_tensor for _pre_combine (avoids a host round-trip).
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
            pad_multiple=self.pad_multiple,
            num_permuted_tokens=self.permute_max_token_num,
            probs=dispatched_probs,
            probs_layout="topk",  # dispatched_probs is [num_dispatched, router_topk]
        )

        if not self.deepep_use_cuda_num_tokens_per_expert:
            tokens_per_expert = tokens_per_expert.cpu()

        return hidden_states, tokens_per_expert, permuted_probs

    def _pre_combine(self, hidden_states):
        hidden_states, _ = turbo.ops.moe_unpermute(
            hidden_states,
            self.row_id_map,
            self.num_dispatched_token_tensor,
            restore_shape=self.hidden_shape_before_permute,
            num_local_experts=self.num_local_experts,
            pad_multiple=self.pad_multiple,
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
        # Clear per-dispatch state; a second combine without re-dispatch must fail loudly.
        self.handle = None
        self.row_id_map = None
        self.dispatched_indices = None
        self.token_indices = None
        self.num_dispatched_token_tensor = None
        return hidden_states

    def _post_combine(self, hidden_states):
        return hidden_states.view(self.hidden_shape)
