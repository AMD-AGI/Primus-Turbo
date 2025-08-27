###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import itertools
from typing import Dict, List, Optional

import torch
import torch.distributed as dist
from torch.testing._internal.common_distributed import (
    MultiProcessTestCase,
    skip_if_lt_x_gpu,
)
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    run_tests,
)

import primus_turbo.pytorch as pt
from tests.test_utils import get_tolerances

_backend_streams: Dict[int, List[torch.cuda.Stream]] = {}


def get_llama3_70b_cfg():
    Ms = [8192]
    Ns = [10240, 57344, 8192, 28672]
    Ks = [8192]
    Bs = [1]
    dtypes = [torch.bfloat16]

    for m, n, k, batch_size, dtype in itertools.product(Ms, Ns, Ks, Bs, dtypes):
        yield m, n, k, batch_size, dtype


def get_llama3_70b_fp8_cfg():
    Ms = [8192]
    Ns = [10240, 57344, 8192, 28672]
    Ks = [8192]
    Bs = [1]
    dtypes = [torch.bfloat16]
    scale_dtypes = [pt.float8_e4m3]
    out_dtypes = [torch.bfloat16]
    scale_modes = ["tensor-wise", "row-wise-sharded", "row-wise-replicated"]
    for m, n, k, batch_size, dtype, scale_dtype, out_dtype, scale_mode in itertools.product(
        Ms, Ns, Ks, Bs, dtypes, scale_dtypes, out_dtypes, scale_modes
    ):
        yield m, n, k, batch_size, dtype, scale_dtype, out_dtype, scale_mode


def get_backend_stream(size=1, priority=0, prefix=""):
    global _backend_streams

    key = (priority, prefix)
    if key not in _backend_streams or len(_backend_streams[key]) < size:
        _backend_streams[key] = [torch.cuda.Stream(priority=priority) for _ in range(size)]

    return _backend_streams[key][:size]


def native_torch_all_gather_matmul(
    A_shard: torch.Tensor,
    Bs: List[torch.Tensor],
    gather_dim: int,
    group,
    scale_mode: Optional[str] = None,
    out_dtype=None,
    A_scale=None,
    B_scales=None,
):

    if scale_mode is not None:
        if scale_mode == "row-wise-sharded":
            A_scale_shard_flat = A_scale.movedim(gather_dim, 0).flatten(0, -2)
            A_scale = A_scale_shard_flat.new_empty(
                (A_scale_shard_flat.shape[0] * group.size(), A_scale_shard_flat.shape[1])
            )
            torch.distributed.all_gather_into_tensor(A_scale, A_scale_shard_flat)
        elif len(A_scale.shape) > 2:
            A_scale = A_scale.flatten(0, -2)
        kwargs = [
            {
                "scale_a": A_scale,
                "scale_b": B_scales[i],
                "scale_result": None,
                "use_fast_accum": None,
                "out_dtype": out_dtype,
            }
            for i, _ in enumerate(Bs)
        ]
        fn = torch.ops.aten._scaled_mm
    else:
        fn = torch.mm
        kwargs = [{} for _ in Bs]

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

    for i, B in enumerate(Bs):
        out = torch.empty((A_out.shape[0], B.shape[-1]), dtype=out_dtype, device="cuda")
        fn(A_out, B, out=out, **kwargs[i])
        outputs.append(out)

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
    def test_fused_all_gather_matmul(self) -> None:
        self._init_process()

        BATCH = 8
        M = 64
        N = 16
        K = 32
        group = dist.group.WORLD
        rank = self.rank

        torch.manual_seed(42 + rank)

        for gather_dim, dtype in itertools.product([0, 1], [torch.bfloat16, torch.float16]):
            A_shard = torch.rand(BATCH, M // self.world_size, K, dtype=dtype, device="cuda")
            Bs = [torch.rand(K, N, dtype=dtype, device="cuda") for _ in range(3)]

            ag_output_0, mm_outputs_0 = native_torch_all_gather_matmul(
                A_shard,
                Bs,
                gather_dim=gather_dim,
                group=group,
                out_dtype=dtype,
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

            if self.rank == 0:
                print(f"test_fused_all_gather_matmul_{gather_dim}_{dtype} Pass")

    @skip_if_lt_x_gpu(2)
    def test_llama3_70b_fused_all_gather_matmul(self) -> None:
        self._init_process()
        group = dist.group.WORLD
        rank = self.rank

        torch.manual_seed(42 + rank)

        for M, N, K, batch_size, dtype in get_llama3_70b_cfg():

            A_shard = torch.rand(batch_size * M // self.world_size, K, dtype=dtype, device="cuda")
            Bs = [torch.rand(K, N, dtype=dtype, device="cuda")]

            ag_output_0, mm_outputs_0 = native_torch_all_gather_matmul(
                A_shard,
                Bs,
                gather_dim=0,
                group=group,
                out_dtype=dtype,
            )
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

            if self.rank == 0:
                print(f"test_llama3_70b_fused_all_gather_matmul_{M}_{N}_{K}_{batch_size}_{dtype} Pass")

    @skip_if_lt_x_gpu(2)
    def test_fused_all_gather_scaled_matmul(self) -> None:
        self._init_process()
        BATCH = 8
        M = 64
        N = 16
        K = 32
        group = dist.group.WORLD
        rank = self.rank

        torch.manual_seed(42 + rank)

        for gather_dim, dtype, scale_dtype, out_dtype, scale_mode in itertools.product(
            [0, 1],
            [torch.bfloat16],
            [pt.float8_e4m3],
            [torch.bfloat16],
            ["tensor-wise", "row-wise-sharded", "row-wise-replicated"],
        ):
            if gather_dim == 0:
                leading_dims = (BATCH // self.world_size, M)
            elif gather_dim == 1:
                leading_dims = (BATCH, M // self.world_size)
            else:
                raise AssertionError(f"Invalid scale_mode: {scale_mode}")

            A_shard = torch.rand(*leading_dims, K, dtype=dtype, device="cuda").to(scale_dtype)
            Bs = [torch.rand(N, K, dtype=dtype, device="cuda").to(scale_dtype) for _ in range(1)]
            if scale_mode == "tensor-wise":
                A_scale = torch.rand((1,), device="cuda")
                B_scales = [torch.rand((1,), device="cuda")]
            elif scale_mode == "row-wise-sharded":
                A_scale = torch.rand((*leading_dims, 1), device="cuda")
                B_scales = [torch.rand((1, B.shape[0]), device="cuda") for B in Bs]
            elif scale_mode == "row-wise-replicated":
                A_scale = torch.full((BATCH, M, 1), 0.1, device="cuda")
                B_scales = [torch.full((1, B.shape[0]), 0.1, device="cuda") for B in Bs]
            else:
                raise AssertionError(f"Invalid scale_mode: {scale_mode}")

            ag_output_0, mm_outputs_0 = native_torch_all_gather_matmul(
                A_shard,
                [B.T for B in Bs],
                gather_dim=gather_dim,
                group=group,
                scale_mode=scale_mode,
                A_scale=A_scale,
                B_scales=B_scales,
                out_dtype=out_dtype,
            )
            ag_output_1, mm_outputs_1 = pt.ops.fused_all_gather_scaled_matmul(
                A_shard,
                [B.T for B in Bs],
                ["NN" for _ in Bs],
                A_scale,
                B_scales,
                gather_dim=gather_dim,
                group_name=group.group_name,
                biases=[None] * len(Bs),
                result_scales=[None] * len(Bs),
                use_fast_accum=[None] * len(Bs),
                out_dtypes=[out_dtype for B in Bs],
                gemm_streams=self.gemm_streams,
                comm_streams=self.comm_streams,
                copy_streams=self.copy_streams,
                comm_method="pipeline",
            )

            torch.testing.assert_close(ag_output_0, ag_output_1, atol=0.0, rtol=0.0)
            assert ag_output_0.stride() == ag_output_1.stride()
            for mm_output_0, mm_output_1 in zip(mm_outputs_0, mm_outputs_1):
                torch.testing.assert_close(mm_output_0, mm_output_1, **get_tolerances(out_dtype))

            if self.rank == 0:
                print(
                    f"test_fused_all_gather_scaled_matmul_{gather_dim}_{dtype}_{scale_dtype}_{out_dtype}_{scale_mode} Pass"
                )

    @skip_if_lt_x_gpu(2)
    def test_llama3_70b_fused_all_gather_scaled_matmul(
        self,
    ) -> None:
        self._init_process()
        group = dist.group.WORLD
        rank = self.rank

        torch.manual_seed(42 + rank)

        for M, N, K, batch_size, dtype, scale_dtype, out_dtype, scale_mode in get_llama3_70b_fp8_cfg():
            A_shard = torch.rand(batch_size * M // group.size(), K, dtype=dtype, device="cuda").to(
                scale_dtype
            )
            Bs = [torch.rand(N, K, dtype=dtype, device="cuda").to(scale_dtype)]

            if scale_mode == "tensor-wise":
                A_scale = torch.rand((1,), device="cuda")
                B_scales = [torch.rand((1,), device="cuda")]
            elif scale_mode == "row-wise-sharded":
                A_scale = torch.rand((batch_size * M // self.world_size, 1), device="cuda")
                B_scales = [torch.rand((1, B.shape[0]), device="cuda") for B in Bs]
            elif scale_mode == "row-wise-replicated":
                A_scale = torch.full((batch_size * M, 1), 0.1, device="cuda")
                B_scales = [torch.full((1, B.shape[0]), 0.1, device="cuda") for B in Bs]
            else:
                raise AssertionError(f"Invalid scale_mode: {scale_mode}")

            ag_output_0, mm_outputs_0 = native_torch_all_gather_matmul(
                A_shard,
                [B.T for B in Bs],
                gather_dim=0,
                group=group,
                scale_mode=scale_mode,
                A_scale=A_scale,
                B_scales=B_scales,
                out_dtype=out_dtype,
            )
            ag_output_1, mm_outputs_1 = pt.ops.fused_all_gather_scaled_matmul(
                A_shard,
                [B.T for B in Bs],
                ["NN" for _ in Bs],
                A_scale,
                B_scales,
                gather_dim=0,
                group_name=group.group_name,
                biases=[None] * len(Bs),
                result_scales=[None] * len(Bs),
                use_fast_accum=[None] * len(Bs),
                out_dtypes=[out_dtype for B in Bs],
                gemm_streams=self.gemm_streams,
                comm_streams=self.comm_streams,
                copy_streams=self.copy_streams,
                comm_method="pipeline",
            )

            torch.testing.assert_close(ag_output_0, ag_output_1, atol=0.0, rtol=0.0)
            for mm_output_0, mm_output_1 in zip(mm_outputs_0, mm_outputs_1):
                torch.testing.assert_close(mm_output_0, mm_output_1, **get_tolerances(out_dtype))


if __name__ == "__main__":
    run_tests()
