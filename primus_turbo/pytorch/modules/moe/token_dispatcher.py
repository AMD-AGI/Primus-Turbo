###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################


from typing import Optional, Tuple

import torch
import torch.distributed as dist

import primus_turbo.pytorch as turbo
from primus_turbo.pytorch.deep_ep import Config


class TurboTokenPermuter(torch.nn.Module):
    """
    This module permute dispatched tokens to match the order of experts.
    """

    def forward(
        self,
        tokens: torch.Tensor,
        token_probs: Optional[torch.Tensor] = None,
        routing_map: Optional[torch.Tensor] = None,
        topk_indices: Optional[torch.Tensor] = None,
        num_out_tokens: int = -1,
        permute_fusion: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        sort tokens to match the order of experts for MoE routing.

        Args:
            tokens (torch.Tensor): dispatched tokens by DeepEP
            token_probs (torch.Tensor): dispatched token_probs by DeepEP
            routing_map(torch.Tensor): The token to expert mapping tensor. shape [num_tokens, num_experts]
            topk_indices(torch.Tensor): The token to expert mapping tensor. shape [num_tokens, topK]
            num_out_tokens(int): the number output tokens of permutation for permute fusion kernel
            permute_fusion(bool): use permute fusion kernel when permute_fusion is True

            See primus_turbo/pytorch/ops/permutation.py::token_permute for more details
        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - permuted_kens: the permuted tokens
                - permuted_probs: the permuted token_probs
                - row_id_map: The tensor of a mapping table for sorted indices used to unpermute the tokens,
        """

        if routing_map is None and topk_indices is None:
            raise ValueError("routing_map or topk_indices must be set")
        if permute_fusion:
            return turbo.ops.token_permute(
                tokens,
                probs=token_probs,
                routing_map=routing_map,
                topk_indices=topk_indices,
                num_out_tokens=num_out_tokens,
            )
        num_tokens, _ = tokens.shape
        num_experts = routing_map.shape[1]

        # mask [num_tokens, num_experts] -> [num_experts, num_tokens]
        routing_map = routing_map.bool().T.contiguous()

        # Create a dense expert-to-token mapping from the sparse token-to-expert mapping
        token_indices = (
            torch.arange(num_tokens, device=routing_map.device).unsqueeze(0).expand(num_experts, -1)
        )
        sorted_indices = token_indices.masked_select(routing_map)

        permuted_probs = None
        if token_probs is not None:
            permuted_probs = token_probs.T.contiguous().masked_select(routing_map)

        # use the mapping to permute the tokens
        permuted_input = tokens.index_select(0, sorted_indices)

        return permuted_input, permuted_probs, sorted_indices


class TurboTokenUnpermuter(torch.nn.Module):
    """
    This module unpermute tokens to for DeepEP combine.
    """

    def forward(
        self,
        permuted_tokens: torch.Tensor,
        sorted_indices: torch.Tensor,
        restore_shape: torch.Size,
        probs: torch.Tensor = None,
        routing_map: torch.Tensor = None,
        drop_and_pad: bool = False,
        permute_fusion: bool = True,
    ):
        """
        unpermute tokens according to the given sorted_indices

        Args:
            permute_fusion(bool): use permute fusion kernel when permute_fusion is True

            See primus_turbo/pytorch/ops/permutation.py::token_unpermute for more details
        """

        if permute_fusion:
            return turbo.ops.token_unpermute(
                permuted_tokens, sorted_indices, merging_probs=probs, restore_shape=restore_shape
            )

        _, hidden = restore_shape
        input_dtype = permuted_tokens.dtype

        if probs is not None:
            assert routing_map is not None, "Mask must be provided to permute the probs."
            if drop_and_pad:
                num_experts = routing_map.size(1)
                num_permuted_tokens = sorted_indices.size(0)
                capacity = num_permuted_tokens // num_experts
                num_unpermuted_tokens = probs.size(0)

                # [num_unpermuted_tokens, num_experts] -> num_experts * num_unpermuted_tokens
                probs_T_1D = probs.T.contiguous().view(-1)

                # get 1D indices of the probs selected by routing_map
                indices_dim0 = torch.arange(num_experts, device=routing_map.device).unsqueeze(-1)
                indices_dim1 = sorted_indices.view(num_experts, capacity)
                indices_1D = (indices_dim0 * num_unpermuted_tokens + indices_dim1).view(-1)

                # get probs from indices
                permuted_probs = probs_T_1D.index_select(0, indices_1D)
            else:
                permuted_probs = probs.T.contiguous().masked_select(routing_map.T.contiguous())
            # Here may promote permuted_tokens to higher precision (fp32/fp64) if probs is in
            # higher precision due to moe_router_dtype being enabled. This can lead to
            # additional GPU memory usage. Use --moe-permute-fusion flag to avoid this extra memory
            # allocation.
            permuted_tokens = permuted_tokens * permuted_probs.unsqueeze(-1)

        # Create an output tensor filled with zeros
        output_tokens = torch.zeros(restore_shape, dtype=permuted_tokens.dtype, device=permuted_tokens.device)
        # Scatter add the permuted_input back to the original positions
        output_tokens.scatter_add_(0, sorted_indices.unsqueeze(1).expand(-1, hidden), permuted_tokens)
        return output_tokens.to(dtype=input_dtype)


class TurboDeepEPTokenDispatcher(torch.nn.Module):
    """
    Dispatch tokens to different experts, with backward pass to combine gradients back to the input.
    Args:
        `ep_group`: the group to use for expert parallism.
        `router_topk`: the number of experts to route to for each token.
        `num_experts`: the number of moe experts
        `hidden_size`: transformer hidden size.
        `dtype`: dtype of tokens which mainly used for DeepEP to initialize DeepEP buffer.
        `deepep_use_comm_stream`: DeepEP will use current stream as communication stream when deepep_use_comm_stream is False
        `deepep_use_cuda_num_tokens_per_expert`: TurboDeepEPTokenDispatcher will return num_tokens_per_expert by cuda tensor instead of cpu tensor, this may elimate groumlp cpu sync when use turbo's groupgemm.
        `deepep_autotune_config`: use autotuned DeepEP config to initialize DeepEP buffer for better performance.
        `permute_fusion`: use permuate fusion kernel when permute_fusion is True
    """

    cuda_dtoh_stream = None

    def __init__(
        self,
        ep_group: dist.ProcessGroup,
        router_topk: int,
        num_experts: int,
        hidden_size: int,
        dtype: torch.dtype,
        deepep_use_comm_stream: Optional[bool] = False,
        deepep_num_use_cu: int = 32,
        deepep_use_cuda_num_tokens_per_expert: Optional[bool] = False,
        deepep_autotune_config: Optional[Config] = None,
        permute_fusion: bool = True,
    ):
        super().__init__()
        assert num_experts % ep_group.size() == 0
        self.router_topk = router_topk
        self.num_experts = num_experts
        self.hidden_size = hidden_size
        self.dtype = dtype
        self.num_local_experts = num_experts // ep_group.size()
        self.deepep_use_cuda_num_tokens_per_expert = deepep_use_cuda_num_tokens_per_expert
        self.permute_fusion = permute_fusion
        hidden_bytes = hidden_size * dtype.itemsize

        turbo.ops.init_deepep_buffer(
            group=ep_group,
            hidden_bytes=hidden_bytes,
            use_default_stream_as_comm_stream=not deepep_use_comm_stream,
            num_use_cu=deepep_num_use_cu,
            autotune_config=deepep_autotune_config,
        )

        self.token_permuter = TurboTokenPermuter()

        if deepep_use_cuda_num_tokens_per_expert and TurboDeepEPTokenDispatcher.cuda_dtoh_stream is None:
            TurboDeepEPTokenDispatcher.cuda_dtoh_stream = torch.cuda.Stream()

    @classmethod
    def maybe_cpu_sync(cls):
        if cls.cuda_dtoh_stream is not None:
            cls.cuda_dtoh_stream.synchronize()

    def forward(
        self,
        hidden_states: torch.Tensor,
        token_probs: torch.Tensor,
        routing_map: Optional[torch.Tensor] = None,
        token_indices: Optional[torch.Tensor] = None,
        deepep_num_worst_tokens: int = 0,
        permute_max_token_num: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Tuple]:
        """
        Args:
            `hidden_states`: Input hidden states to be processed
            `token_probs`: Routing probabilities for each token-expert pair
            `routing_map`: shape [num_tokens, num_experts], map indicating which expert each token should be routed to
            `token_indices`: shape [num_tokens, router_topk], which are the routed expert indices.
            `deepep_num_worst_tokens`: the worst number of tokens to receive, see 'deep_ep/buffer.py::dispatch' for more details.
            `permute_max_token_num`: the max tokens of permute fusion kernel.
        """

        if deepep_num_worst_tokens > 0 and not self.deepep_use_cuda_num_tokens_per_expert:
            raise ValueError(
                "Please set deepep_use_cuda_num_tokens_per_expert=True when use deepep_num_worst_tokens"
            )

        if permute_max_token_num > 0 and not self.permute_fusion:
            raise ValueError("Please set permute_fusion=True when use permute_max_token_num")

        # check hidden_states, token_probs shape or dtype
        tokens = hidden_states.view(-1, self.hidden_size)
        num_tokens = tokens.shape[0]

        # the shape of token_probs maybe [num_tokens, topk] or [num_tokens, num_experts]
        token_probs = token_probs.reshape(num_tokens, -1)
        probs_sorted = token_probs.shape[-1] == self.router_topk

        # 1. token_indices is None
        # call topk to get token_idx and token_probs, ensure token_probs unsorted
        if token_indices is None:
            assert (
                not probs_sorted
            ), "token_probs is sorted by topk, need unsorted token_probs when token_indices is None"
            token_probs, token_indices = torch.topk(token_probs, self.router_topk, dim=-1)

            probs_sorted = True

        # 2. token_indices is not None
        # call gather to get token_probs if token_probs unsorted, otherwise skip
        if not probs_sorted:
            token_probs = token_probs.gather(1, token_indices)

        if token_probs.dtype != torch.float32 and token_probs.dtype in [torch.bfloat16, torch.float16]:
            if dist.get_rank() == 0:
                print("DeepEP only supports float32 token_probs!")
            token_probs = token_probs.float()  # downcast or upcast

        (dispatched_tokens, dispatched_indices, dispatched_probs, num_tokens_per_expert, dispatch_handle) = (
            turbo.ops.deepep_dispatch(
                tokens,
                token_indices=token_indices,
                token_weights=token_probs,
                num_experts=self.num_experts,
                num_worst_tokens=deepep_num_worst_tokens,
                use_cuda_num_token_per_expert=self.deepep_use_cuda_num_tokens_per_expert,
            )
        )

        # TODO: try use topk_indices instead of routing_map can eliminate indices_to_multihot
        dispatched_routing_map, dispatched_probs = turbo.ops.fused_indices_to_multihot(
            dispatched_indices, dispatched_probs, self.num_local_experts
        )
        if permute_max_token_num > 0:
            num_out_tokens = permute_max_token_num
        else:
            num_out_tokens = torch.sum(num_tokens_per_expert)
            if num_out_tokens.device.type == "cuda":
                self.cuda_dtoh_stream.wait_stream(torch.cuda.current_stream())
                num_out_tokens.record_stream(self.cuda_dtoh_stream)
                with self.cuda_dtoh_stream:
                    num_out_tokens_cpu = torch.empty_like(
                        num_out_tokens, dtype=num_out_tokens.dtype, device="cpu", pin_memory=True
                    )
                    num_out_tokens_cpu.copy_(num_out_tokens, non_blocking=True)

                num_out_tokens = num_out_tokens_cpu

        permuted_tokens, permuted_probs, reversed_mapping_for_combine = self.token_permuter(
            dispatched_tokens,
            token_probs=dispatched_probs,
            routing_map=dispatched_routing_map,
            num_out_tokens=num_out_tokens,
            permute_fusion=self.permute_fusion,
        )
        handle = (
            reversed_mapping_for_combine,
            dispatched_tokens.shape,
            dispatched_routing_map,
            dispatch_handle,
            self.permute_fusion,
        )
        return permuted_tokens, num_tokens_per_expert, permuted_probs, handle


class TurboDeepEPTokenCombiner(torch.nn.Module):
    """
    Combine tokens from different experts, with backward pass to dispatch gradients back to the input.
    Note: the initialization of DeepEP buffer is finised in TurboDeepEPTokenDispatcher, we don't need init agin in Combiner
    """

    def __init__(self):
        super().__init__()
        self.token_unpermuter = TurboTokenUnpermuter()

    def forward(
        self, hidden_states: torch.Tensor, handle: Tuple, bias: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            `hidden_states`: Input hidden states to be processed
            `handle`: the handle of dispatch and permute, which is the last output of TurboDeepEPTokenDispatcher
        """

        assert bias is None

        # unpack deepep and permute handle
        (
            reversed_mapping_for_combine,
            hidden_shape_before_permute,
            dispatched_routing_map,
            dispatch_handle,
            permute_fusion,
        ) = handle

        unpermuted_tokens = self.token_unpermuter(
            hidden_states,
            reversed_mapping_for_combine,
            restore_shape=hidden_shape_before_permute,
            routing_map=dispatched_routing_map,
            permute_fusion=permute_fusion,
        )
        combined_token = turbo.ops.deepep_combine(unpermuted_tokens, handle=dispatch_handle)

        return combined_token, None
