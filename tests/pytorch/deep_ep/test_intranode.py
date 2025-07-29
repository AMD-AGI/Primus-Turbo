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
from tests.pytorch.ref.deep_ep_ref import tune_and_verify_intranode


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


if __name__ == "__main__":
    run_tests()
