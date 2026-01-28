# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################


import os
from dataclasses import dataclass
from functools import lru_cache
from itertools import product

import torch
import torch.distributed as dist
from torch.testing._internal.common_distributed import MultiProcessTestCase
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    run_tests,
)

import primus_turbo.pytorch as turbo


@dataclass
class TokenDispatcherTestConfig:
    # random data generation
    num_tokens: int
    hidden_size: int
    dtype: torch.dtype
    # DeepEPTokenDispatcher init
    router_topk: int
    num_experts: int
    permute_fusion: bool
    deepep_use_cuda_num_tokens_per_expert: bool
    expert_capacity_factor: float

    # DeepEPTokenDispatcher forward
    deepep_num_worst_tokens: int
    permute_max_token_num: int


@lru_cache
def get_token_dispatcher_config():
    num_tokens_list = [4096]
    hidden_size_list = [4096]
    dtype_list = [torch.bfloat16]
    router_topk_list = [8]
    num_experts_list = [256]
    permute_fusion_list = [True]
    deepep_use_cuda_num_tokens_per_expert_list = [False, True]
    expert_capacity_factor_list = [None, 0.5]
    for (
        num_tokens,
        hidden_size,
        dtype,
        router_topk,
        num_experts,
        permute_fusion,
        deepep_use_cuda_num_tokens_per_expert,
        expert_capacity_factor,
    ) in product(
        num_tokens_list,
        hidden_size_list,
        dtype_list,
        router_topk_list,
        num_experts_list,
        permute_fusion_list,
        deepep_use_cuda_num_tokens_per_expert_list,
        expert_capacity_factor_list,
    ):
        deepep_num_worst_tokens_list = [0, num_tokens * 8]
        permute_max_token_num_list = [0, num_tokens * 8 * router_topk]

        for deepep_num_worst_tokens, permute_max_token_num in product(
            deepep_num_worst_tokens_list, permute_max_token_num_list
        ):
            if deepep_num_worst_tokens > 0 and not deepep_use_cuda_num_tokens_per_expert:
                continue

            yield TokenDispatcherTestConfig(
                num_tokens=num_tokens,
                hidden_size=hidden_size,
                dtype=dtype,
                router_topk=router_topk,
                num_experts=num_experts,
                deepep_use_cuda_num_tokens_per_expert=deepep_use_cuda_num_tokens_per_expert,
                permute_fusion=permute_fusion,
                deepep_num_worst_tokens=deepep_num_worst_tokens,
                expert_capacity_factor=expert_capacity_factor,
                permute_max_token_num=permute_max_token_num,
            )


@instantiate_parametrized_tests
class TokenDispatcherTestBase(MultiProcessTestCase):
    def setUp(self) -> None:
        super().setUp()
        self._spawn_processes()

    @property
    def world_size(self) -> int:
        return torch.cuda.device_count()

    @property
    def device(self) -> torch.device:
        return torch.device("cuda", self.rank)

    def _init_process(self):
        torch.cuda.set_device(self.device)
        store = dist.FileStore(self.file_name, self.world_size)
        dist.init_process_group(
            backend="nccl",
            world_size=self.world_size,
            rank=self.rank,
            store=store,
        )
        torch.manual_seed(42 + self.rank)
        os.environ["WORLD_SIZE"] = str(self.world_size)
        os.environ["LOCAL_WORLD_SIZE"] = str(self.world_size)
        os.environ["LOCAL_RANK"] = str(self.rank)

    def test_token_dispatcher_dropless(self):
        self._init_process()
        ep_group = dist.group.WORLD

        for cfg in get_token_dispatcher_config():
            dispatcher = turbo.modules.DeepEPTokenDispatcher(
                cfg.num_experts,
                cfg.router_topk,
                ep_group,
                permute_fusion=cfg.permute_fusion,
                deepep_use_cuda_num_tokens_per_expert=cfg.deepep_use_cuda_num_tokens_per_expert,
                deepep_num_worst_tokens=cfg.deepep_num_worst_tokens,
                permute_max_token_num=cfg.permute_max_token_num,
                expert_capacity_factor=cfg.expert_capacity_factor,
            )

            hidden_states = torch.randn((cfg.num_tokens, cfg.hidden_size), dtype=cfg.dtype, device="cuda")
            ans = hidden_states
            hidden_states.requires_grad = True

            probs = (
                torch.ones((cfg.num_tokens, cfg.num_experts), dtype=torch.float32, device="cuda")
                / cfg.router_topk
            )

            permuted_local_hidden_states, tokens_per_expert, permuted_probs = dispatcher.token_dispatch(
                hidden_states,
                probs,
            )

            permuted_local_hidden_states = permuted_local_hidden_states * permuted_probs.unsqueeze(-1)

            permuted_local_hidden_states = permuted_local_hidden_states.to(ans.dtype)

            restored_hidden_states = dispatcher.token_combine(permuted_local_hidden_states)

            assert torch.allclose(
                restored_hidden_states, ans
            ), f"Restored hidden states do not match original hidden states, {restored_hidden_states} {ans}"

            # check if the grad of the hidden states is same as the hidden states
            torch.autograd.backward(restored_hidden_states, hidden_states)
            assert torch.allclose(
                hidden_states.grad, ans
            ), "Restored hidden states do not match original hidden states"

            expected_token_per_expert_device = "cuda" if cfg.deepep_use_cuda_num_tokens_per_expert else "cpu"
            assert tokens_per_expert.device.type == expected_token_per_expert_device


if __name__ == "__main__":
    run_tests()
