from typing import Dict, List

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
from tests.test_utils import get_tolerances

_backend_streams: Dict[int, List[torch.cuda.Stream]] = {}


def get_backend_stream(size=1, priority=0, prefix=""):
    global _backend_streams

    key = (priority, prefix)
    if key not in _backend_streams or len(_backend_streams[key]) < size:
        _backend_streams[key] = [torch.cuda.Stream(priority=priority) for _ in range(size)]

    return _backend_streams[key][:size]


def native_torch_all_gather_matmul(A_shard: torch.Tensor, Bs: List[torch.Tensor], gather_dim: int, group):

    A_shard_flat = A_shard.movedim(gather_dim, 0)
    leading_dims = [group.size()] + list(A_shard_flat.shape[:-1])
    A_shard_flat = A_shard_flat.flatten(0, -2)

    def unflatten(t: torch.Tensor) -> torch.Tensor:
        return t.view(*leading_dims, -1).flatten(0, 1).movedim(0, gather_dim)

    A_out = A_shard_flat.new_empty(
        A_shard_flat.shape[0] * group.size(),
        A_shard_flat.shape[1],
    )

    torch.distributed.all_gather_into_tensor(A_out, A_shard_flat, group=group)

    outputs = []

    for B in Bs:
        outputs.append(torch.mm(A_out, B))

    return unflatten(A_out), [unflatten(output) for output in outputs]


@instantiate_parametrized_tests
class FusedAllGatherMatmulTestBase(MultiProcessTestCase):
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

        self.gemm_streams = [torch.cuda.current_stream()]
        self.comm_streams = get_backend_stream(size=self.world_size - 1, priority=0, prefix="comm")

        self.copy_streams = get_backend_stream(size=1, priority=0, prefix="copy")

    @skip_if_lt_x_gpu(2)
    @parametrize("gather_dim", [0, 1])
    @parametrize("dtype", [torch.bfloat16, torch.float16])
    def test_fused_all_gather_matmul(self, gather_dim: int, dtype: torch.dtype) -> None:
        self._init_process()

        BATCH = 8
        M = 64
        N = 16
        K = 32
        group = dist.group.WORLD
        rank = self.rank

        torch.manual_seed(42 + rank)
        A_shard = torch.rand(BATCH, M // self.world_size, K, dtype=dtype, device="cuda")
        Bs = [torch.rand(K, N, dtype=dtype, device="cuda") for _ in range(3)]

        ag_output_0, mm_outputs_0 = native_torch_all_gather_matmul(
            A_shard, Bs, gather_dim=gather_dim, group=group
        )
        ag_output_1, mm_outputs_1 = pt.ops.fused_all_gather_matmul(
            A_shard,
            Bs,
            layouts=["NN" for _ in range(len(Bs))],
            gather_dim=gather_dim,
            group_name=group.group_name,
            gemm_streams=self.gemm_streams,
            comm_streams=self.comm_streams,
            copy_streams=self.copy_streams,
            comm_method="pipeline",
        )

        torch.testing.assert_close(ag_output_0, ag_output_1, atol=0.0, rtol=0.0)
        assert ag_output_0.stride() == ag_output_1.stride()
        for mm_output_0, mm_output_1 in zip(mm_outputs_0, mm_outputs_1):
            torch.testing.assert_close(mm_output_0, mm_output_1, **get_tolerances(dtype))

    @skip_if_lt_x_gpu(2)
    @parametrize("M,K,N", [(8192, 8192, 10240), (8192, 8192, 57344), (8192, 8192, 8192), (8192, 8192, 28672)])
    @parametrize("batch_size", [1, 4])
    @parametrize("dtype", [torch.bfloat16])
    def test_llama3_70b_fused_all_gather_matmul(self, dtype, batch_size, M, K, N) -> None:
        self._init_process()
        group = dist.group.WORLD
        rank = self.rank

        torch.manual_seed(42 + rank)
        A_shard = torch.rand(batch_size * M // self.world_size, K, dtype=dtype, device="cuda")
        Bs = [torch.rand(K, N, dtype=dtype, device="cuda")]

        ag_output_0, mm_outputs_0 = native_torch_all_gather_matmul(A_shard, Bs, gather_dim=0, group=group)
        ag_output_1, mm_outputs_1 = pt.ops.fused_all_gather_matmul(
            A_shard,
            Bs,
            layouts=["NN"],
            gather_dim=0,
            group_name=group.group_name,
            gemm_streams=self.gemm_streams,
            comm_streams=self.comm_streams,
            copy_streams=self.copy_streams,
            comm_method="pipeline",
            num_splits=4,
        )

        torch.testing.assert_close(ag_output_0, ag_output_1, atol=0.0, rtol=0.0)
        for mm_output_0, mm_output_1 in zip(mm_outputs_0, mm_outputs_1):
            torch.testing.assert_close(mm_output_0, mm_output_1, **get_tolerances(dtype))


if __name__ == "__main__":
    run_tests()
