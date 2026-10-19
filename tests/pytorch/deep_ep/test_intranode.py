###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# The reference flow this test drives is derived from DeepEP
# (https://github.com/deepseek-ai/DeepEP); see tests/pytorch/ref/deep_ep_ref.py.
#
# See LICENSE for license information.
###############################################################################

import torch
import torch.distributed as dist
from torch.testing._internal.common_distributed import (
    MultiProcessTestCase,
    skip_if_lt_x_gpu,
)
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
)

import primus_turbo.pytorch as pt
from tests.pytorch.ref.deep_ep_ref import (
    get_dispatch_layout_ref,
    tune_and_verify_intranode,
)


@instantiate_parametrized_tests
class DeepEPIntranodeTestCase(MultiProcessTestCase):
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
        buffer = pt.deep_ep.Buffer(dist.group.WORLD, int(1e9))
        return buffer

    @skip_if_lt_x_gpu(2)
    @parametrize("num_tokens", [4096])
    @parametrize("hidden", [4096])
    @parametrize("num_topk", [8])
    @parametrize("num_experts", [128])
    @parametrize("num_sms", [24])
    def test_intranode(self, num_tokens: int, hidden: int, num_topk: int, num_experts: int, num_sms: int):
        # Random data
        buffer = self._init_process()
        group = dist.group.WORLD
        rank = self.rank
        num_ranks = group.size()
        torch.manual_seed(42 + rank)

        tune_and_verify_intranode(
            num_sms, num_tokens, hidden, num_topk, num_experts, rank, num_ranks, rank, buffer, group
        )

    @skip_if_lt_x_gpu(2)
    def test_dispatch_layout(self):
        buffer = self._init_process()
        num_ranks = dist.group.WORLD.size()
        torch.manual_seed(42 + self.rank)
        # (num_tokens, num_topk, num_experts, routing); top-k 20 exercises the per-expert fallback.
        cases = [
            (4099, 8, 128, "random"),
            (4099, 8, 128, "masked"),
            (32768, 8, 128, "even"),
            (1, 8, 128, "random"),
            (0, 8, 128, "random"),
            (777, 1, 64, "random"),
            (777, 9, 256, "masked"),
            (777, 20, 256, "random"),
        ]
        for num_tokens, num_topk, num_experts, routing in cases:
            if routing == "even":
                topk_idx = torch.arange(num_tokens * num_topk, device="cuda").view(num_tokens, num_topk)
                topk_idx = topk_idx % num_experts
            else:
                scores = torch.randn((num_tokens, num_experts), device="cuda").abs() + 1
                topk_idx = torch.topk(scores, num_topk, dim=-1, sorted=False)[1]
                if routing == "masked":
                    topk_idx.masked_fill_(torch.rand(topk_idx.shape, device="cuda") < 0.2, -1)

            num_tokens_per_rank, _, num_tokens_per_expert, is_token_in_rank, _ = buffer.get_dispatch_layout(
                topk_idx, num_experts
            )
            ref_per_rank, _, ref_per_expert, ref_in_rank = get_dispatch_layout_ref(
                topk_idx, num_experts, num_ranks=num_ranks
            )
            case = (num_tokens, num_topk, num_experts, routing)
            assert torch.equal(num_tokens_per_rank, ref_per_rank), case
            assert torch.equal(num_tokens_per_expert, ref_per_expert), case
            assert torch.equal(is_token_in_rank, ref_in_rank), case


if __name__ == "__main__":
    run_tests()
