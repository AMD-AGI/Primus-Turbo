# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################


from dataclasses import dataclass

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
    num_tokens: int
    hidden_size: int
    dtype: torch.dtype
    router_topk: int
    num_experts: int


_deepep_token_dispatcher_config = [
    TokenDispatcherTestConfig(
        num_tokens=4096, hidden_size=4096, dtype=torch.bfloat16, router_topk=8, num_experts=16
    )
]


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

    def _init_dispatcher(self, kwargs):
        ep_group = dist.group.WORLD
        return turbo.modules.TurboDeepEPTokenDispatcher(ep_group, **kwargs)

    def test_token_dispatcher_dropless(self):
        self._init_process()
        ep_group = dist.group.WORLD

        for cfg in _deepep_token_dispatcher_config:
            dispatcher = turbo.modules.TurboDeepEPTokenDispatcher(
                ep_group, cfg.router_topk, cfg.num_experts, cfg.hidden_size, cfg.dtype
            )
            combiner = turbo.modules.TurboDeepEPTokenCombiner()

            hidden_states = torch.randn((cfg.num_tokens, cfg.hidden_size), dtype=cfg.dtype, device="cuda")
            ans = hidden_states
            hidden_states.requires_grad = True

            probs = (
                torch.ones((cfg.num_tokens, cfg.num_experts), dtype=torch.float32, device="cuda")
                / cfg.router_topk
            )

            (permuted_local_hidden_states, tokens_per_expert, permuted_probs, handle) = dispatcher(
                hidden_states, probs, token_indices=None
            )

            permuted_local_hidden_states = permuted_local_hidden_states * permuted_probs.unsqueeze(-1)

            permuted_local_hidden_states = permuted_local_hidden_states.to(hidden_states.dtype)

            restored_hidden_states, _ = combiner(permuted_local_hidden_states, handle)

            assert torch.allclose(
                restored_hidden_states, ans
            ), f"Restored hidden states do not match original hidden states, {restored_hidden_states} {ans}"

            # check if the grad of the hidden states is same as the hidden states
            torch.autograd.backward(restored_hidden_states, hidden_states)
            assert torch.allclose(
                hidden_states.grad, ans
            ), "Restored hidden states do not match original hidden states"


if __name__ == "__main__":
    run_tests()
