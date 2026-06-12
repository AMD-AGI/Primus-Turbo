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
from primus_turbo.pytorch.deep_ep import Config
from primus_turbo.pytorch.kernels.moe.moe_dispatch_combine_impl import (
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
    Args:
        `num_experts`: the number of moe experts
        `router_topk`: the number of experts to route to for each token.
        `ep_group`: the group to use for expert parallism.
        `tp_group`: the group to use for tensor parallism.
        `tp_ep_group`: the group to use for tensor-expert parallism.
        `expert_capacity_factor`: The capacity factor for each expert, None means no token will be dropped
        `permute_fusion`: use permuate fusion kernel when permute_fusion is True
        `permute_max_token_num`: use max_token_num can elimite host sync in permute when set deepep_use_cuda_num_tokens_per_expert=True
        `deepep_use_comm_stream`: DeepEP will use current stream as communication stream when deepep_use_comm_stream is False
        `deepep_num_use_cu`: number of cu deepep used
        `deepep_num_worst_tokens`: number of worst tokens for deepep, see DeepEP for more detail.
        `deepep_use_cuda_num_tokens_per_expert`: DeepEPTokenDispatcher will return num_tokens_per_expert by cuda tensor instead of cpu tensor, this may elimate groumlp cpu sync when use turbo's groupgemm.
        `deepep_autotune_config`: use autotuned DeepEP config to initialize DeepEP buffer for better performance.

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
        permute_max_token_num: int = 0,
        deepep_async_finish: bool = True,
        deepep_allocate_on_comm_stream: bool = True,
        deepep_use_comm_stream: Optional[bool] = False,
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

        self.capacity_factor = expert_capacity_factor

        # permute
        self.permute_fusion = permute_fusion
        self.permute_max_token_num = permute_max_token_num

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
        # Reset per-iteration overflow state.
        self._overflow_warned = False

        # Pre-compute a static dispatch budget so token_permute uses a fixed num_out_tokens,
        # enabling CUDA graph capture without device-to-host synchronisation.
        # MOE_EXPERT_RANK_CAPACITY_FACTOR is set in config_MI355X_1x8x1_... and flows through
        # gpt_oss_20B-pretrain-fp8.yaml via ${oc.env:MOE_EXPERT_RANK_CAPACITY_FACTOR,null}.
        _rcf_str = os.environ.get("MOE_EXPERT_RANK_CAPACITY_FACTOR", "")
        _rank_capacity_factor = float(_rcf_str) if _rcf_str else None
        if _rank_capacity_factor is not None:
            _num_tokens = hidden_states.shape[0]         # bs x seq_len (e.g. 4 x 8192 = 32768)
            _topk = self.router_topk                     # e.g. 4 (includes tp_size factor)
            _pad = 256                                   # align to 256 tokens
            budget = int(_num_tokens * _topk * _rank_capacity_factor)
            budget += (-budget) % _pad                   # round up e.g. to 157440
            self._num_permuted_tokens = budget
            if not getattr(self, '_budget_printed', False):
                print(
                    f"[DeepEPTokenDispatcher] _exec_dispatch: budget={budget} tokens "
                    f"(num_tokens={_num_tokens}, topk={_topk}, factor={_rank_capacity_factor})",
                    flush=True,
                )
                self._budget_printed = True
        else:
            self._num_permuted_tokens = None

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

        # turbo.ops.moe_dispatch dynamically allocates recv_x, dispatched_indices,
        # and dispatched_probs.  Inside a captured graph those allocations are
        # replayed at capture-time addresses with potentially wrong sizes.
        # Copy into fixed-shape persistent buffers so the graph always sees [budget, H].
        if self._num_permuted_tokens is not None:
            hidden_size = hidden_states.shape[-1]
            if not hasattr(self, '_dispatch_buf') or self._dispatch_buf is None:
                self._dispatch_buf = torch.zeros(
                    (self._num_permuted_tokens, hidden_size),
                    dtype=hidden_states.dtype, device=hidden_states.device,
                )
                self._dispatch_indices_buf = torch.full(
                    (self._num_permuted_tokens,), -1,
                    dtype=dispatched_indices.dtype, device=hidden_states.device,
                )
                self._dispatch_probs_buf = torch.zeros(
                    (self._num_permuted_tokens,),
                    dtype=dispatched_probs.dtype, device=hidden_states.device,
                )
                print(
                    f"[DeepEPTokenDispatcher] Persistent dispatch buffers allocated: "
                    f"({self._num_permuted_tokens}, {hidden_size})",
                    flush=True,
                )

            actual = hidden_states.shape[0]
            self._dispatch_buf[:actual].copy_(hidden_states)
            self._dispatch_buf[actual:].zero_()
            self._dispatch_indices_buf[:actual].copy_(dispatched_indices)
            self._dispatch_indices_buf[actual:].fill_(-1)
            self._dispatch_probs_buf[:actual].copy_(dispatched_probs)
            self._dispatch_probs_buf[actual:].zero_()

            self.dispatched_indices = self._dispatch_indices_buf
            return self._dispatch_buf, self._dispatch_probs_buf

        self.dispatched_indices = dispatched_indices
        return hidden_states, dispatched_probs

    def _post_dispatch(self, hidden_states, dispatched_probs):
        _device_initiated = os.environ.get("PRIMUS_DEVICE_INITIATED_GEMM", "0") == "1"

        _rcf_str = os.environ.get("MOE_EXPERT_RANK_CAPACITY_FACTOR", "")
        _rank_capacity_factor = float(_rcf_str) if _rcf_str else None

        if self.permute_max_token_num > 0:
            num_out_tokens = self.permute_max_token_num
        elif (
            _rank_capacity_factor is not None
            and hasattr(self, '_num_permuted_tokens')
            and self._num_permuted_tokens is not None
        ):
            # Rank-level capacity pre-allocation: static budget, no host-device sync.
            # self._num_permuted_tokens was computed in _exec_dispatch() before the kernel call.
            num_out_tokens = self._num_permuted_tokens
            if not getattr(self, '_budget_post_printed', False):
                print(
                    f"[DeepEPTokenDispatcher] _post_dispatch: using budget num_out_tokens={num_out_tokens} "
                    f"(recv_x.shape[0]={hidden_states.shape[0]}, "
                    f"factor={_rank_capacity_factor})",
                    flush=True,
                )
                self._budget_post_printed = True
            if self.handle is not None and len(self.handle) > 0:
                overflow_flag = self.handle[-1]
                if overflow_flag is not None:
                    try:
                        is_over_budget = bool(overflow_flag.item())
                    except Exception:
                        is_over_budget = False
                    if is_over_budget and not getattr(self, '_overflow_warned', False):
                        import warnings
                        warnings.warn(
                            f"[MoE Rank Capacity Overflow] Actual dispatched tokens exceeded "
                            f"pre-allocated budget of {self._num_permuted_tokens} tokens "
                            f"(moe_expert_rank_capacity_factor={_rank_capacity_factor}). "
                            f"Increase moe_expert_rank_capacity_factor to suppress. "
                            f"Overflow breaks CUDA graph replay when a graph is active.",
                            stacklevel=3,
                        )
                        self._overflow_warned = True
        elif _device_initiated and self.tokens_per_expert.numel() > 0:
            # Device-initiated GEMM without capacity pre-allocation.
            # Reached only when MOE_EXPERT_RANK_CAPACITY_FACTOR is unset.
            num_out_tokens = hidden_states.shape[0]
        elif self.tokens_per_expert.numel() > 0:
            num_out_tokens = self.tokens_per_expert.sum().item()  # host-device sync (eager mode)
        else:
            # will case cpu sync at permute phase
            num_out_tokens = -1

        self.dispatched_routing_map, dispatched_probs = turbo.ops.indices_to_multihot(
            self.dispatched_indices, dispatched_probs, self.num_local_experts, fused=self.permute_fusion
        )

        self.hidden_shape_before_permute = hidden_states.shape
        assert dispatched_probs.dtype == torch.float32, "DeepEP only supports float32 probs"
        hidden_states, permuted_probs, self.reversed_mapping_for_combine, tokens_per_expert = (
            turbo.ops.token_permute(
                hidden_states,
                num_out_tokens=num_out_tokens,
                routing_map=self.dispatched_routing_map,
                probs=dispatched_probs,
                fused=self.permute_fusion,
                return_tokens_per_expert=self.deepep_use_cuda_num_tokens_per_expert or num_out_tokens == -1,
            )
        )

        if not self.deepep_use_cuda_num_tokens_per_expert:
            if self.tokens_per_expert is not None:
                tokens_per_expert = self.tokens_per_expert
            else:
                tokens_per_expert = tokens_per_expert.cpu()

        self.tokens_per_expert = None
        return hidden_states, tokens_per_expert, permuted_probs

    def _pre_combine(self, hidden_states):
        hidden_states = turbo.ops.token_unpermute(
            hidden_states,
            self.reversed_mapping_for_combine,
            restore_shape=self.hidden_shape_before_permute,
            routing_map=self.dispatched_routing_map,
            fused=self.permute_fusion,
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
