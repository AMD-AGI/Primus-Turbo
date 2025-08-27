###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import itertools
from typing import Dict, List

import torch
import torch.distributed as dist
from torch._C._distributed_c10d import ReduceOp
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
    Ns = [8192]
    Ks = [8192, 28672]
    Bs = [1, 4]
    dtypes = [torch.bfloat16]
    comm_methods = ["tile", "pipeline"]
    for m, n, k, batch_size, dtype, comm_method in itertools.product(Ms, Ns, Ks, Bs, dtypes, comm_methods):
        yield m, n, k, batch_size, dtype, comm_method


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


def native_torch_matmul_reduce_scatter(
    A: torch.Tensor,
    B: torch.Tensor,
    scatter_dim: int,
    reduce_op: str,
    group: dist.group.WORLD,
):
    output = torch.matmul(A, B)
    output_flat = output.movedim(scatter_dim, 0).contiguous()
    rs_out = output.new_empty(
        output_flat.shape[0] // group.size(),
        *output_flat.shape[1:],
    )

    if reduce_op == "avg":
        reduce_op = ReduceOp.AVG
    elif reduce_op == "sum":
        reduce_op = ReduceOp.SUM
    else:
        raise ValueError(f"Only avg or sum be supported, but provided reduce_op is {reduce_op}")
    torch.distributed.reduce_scatter_tensor(rs_out, output_flat, reduce_op, group)

    rs_out_flat = rs_out.movedim(0, scatter_dim)
    return rs_out_flat


def native_torch_matmul_reduce_scatter_a2a(
    A: torch.Tensor,
    B: torch.Tensor,
    scatter_dim: int,
    reduce_op: str,
    group: dist.group.WORLD,
):
    rs_out_shape = [*A.shape[:-1], B.shape[0]]
    rs_out_shape[scatter_dim] //= group.size()

    A_flat = A.movedim(scatter_dim, 0)
    leading_dims = [group.size()] + list(A_flat.shape[:-1])
    leading_dims[1] //= group.size()
    A_flat = A_flat.flatten(0, -2)

    output = torch.matmul(A_flat, B)

    output_a2a = torch.empty_like(output)
    torch.distributed.all_to_all_single(output_a2a, output)
    output_a2a = output_a2a.view(*leading_dims, -1)

    if reduce_op == "avg":
        rs_output = torch.mean(output_a2a.float(), dim=0).to(A.dtype)
    else:
        rs_output = torch.sum(output_a2a.float(), dim=0).to(A.dtype)
    rs_output = rs_output.movedim(0, scatter_dim)

    return rs_output


def generate_data(shape, dtype, device="cuda", scale=0.01, bias=0.0):
    return (torch.rand(*shape, dtype=dtype, device=device) * 2 - 1) * scale + bias


@instantiate_parametrized_tests
class FusedMatmulReduceScatterTestBase(MultiProcessTestCase):
    def setUp(self) -> None:
        super().setUp()
        self._spawn_processes()

    @property
    def world_size(self) -> int:
        return torch.cuda.device_count()

    @property
    def device(self) -> torch.device:
        return torch.device("cuda", self.rank)

    def _init_process(
        self,
    ):
        torch.cuda.set_device(self.device)
        store = dist.FileStore(self.file_name, self.world_size)
        dist.init_process_group(
            backend="nccl",
            world_size=self.world_size,
            rank=self.rank,
            store=store,
        )
        torch.manual_seed(42 + self.rank)

    def init_streams(self, comm_method):
        if comm_method == "pipeline":
            self.gemm_streams = [torch.cuda.current_stream()]
            self.comm_streams = get_backend_stream(size=self.world_size, priority=0, prefix="comm")
        else:
            self.gemm_streams = []
            self.comm_streams = []

    @skip_if_lt_x_gpu(2)
    def test_fused_matmul_reduce_scatter(self) -> None:
        self._init_process()
        for comm_method, dtype, reduce_op, scatter_dim in itertools.product(
            ["pipeline"], [torch.bfloat16, torch.float16], ["sum", "avg"], [0, 1]
        ):
            self.init_streams(comm_method)

            BATCH = 8
            M = 256
            N = 256
            K = 512
            group = dist.group.WORLD
            rank = self.rank

            A = generate_data(
                shape=(BATCH, M, K),
                dtype=dtype,
                device="cuda",
                scale=0.01 * (rank + 1),
            )
            B = generate_data(
                shape=(K, N),
                dtype=dtype,
                device="cuda",
                scale=0.01 * (rank + 1),
            )

            rs_output_a2a = native_torch_matmul_reduce_scatter_a2a(A, B, scatter_dim, reduce_op, group)
            rs_output_native = native_torch_matmul_reduce_scatter(A, B, scatter_dim, reduce_op, group)
            rs_output_turbo = pt.ops.fused_matmul_reduce_scatter(
                A,
                B,
                layout="NN",
                reduce_op=reduce_op,
                scatter_dim=scatter_dim,
                group_name=group.group_name,
                gemm_streams=self.gemm_streams,
                comm_streams=self.comm_streams,
                comm_method=comm_method,
            )

            assert (
                rs_output_native.stride() == rs_output_turbo.stride()
            ), f"rs_output_native stride {rs_output_native.stride()} is not equal to rs_output_turbo stride {rs_output_turbo.stride()}"
            diff = (rs_output_native - rs_output_turbo).abs().max()
            diff_mask = rs_output_native != rs_output_turbo
            print(
                f"[rank {torch.distributed.get_rank()}]native vs turbo: diff_mask-{diff_mask.sum()}, diff_max-{diff}, value_range-({rs_output_native.min()}, {rs_output_native.max()})"
            )
            diff = (rs_output_turbo - rs_output_a2a).abs().max()
            diff_mask = rs_output_turbo != rs_output_a2a
            print(
                f"[rank {torch.distributed.get_rank()}]a2a vs turbo: diff_mask-{diff_mask.sum()}, diff_max-{diff}, value_range-({rs_output_a2a.min()}, {rs_output_a2a.max()})"
            )
            torch.testing.assert_close(rs_output_turbo, rs_output_a2a, **get_tolerances(dtype))

    @skip_if_lt_x_gpu(2)
    def test_llama3_70b_fused_matmul_reduce_scatter(self) -> None:
        self._init_process()
        group = dist.group.WORLD
        rank = self.rank
        scatter_dim = 0
        reduce_op = "sum"

        for M, N, K, batch_size, dtype, comm_method in get_llama3_70b_cfg():
            self.init_streams(comm_method)
            A = generate_data(
                shape=(batch_size * M, K // group.size()),
                dtype=dtype,
                device="cuda",
                scale=0.01 * (rank + 1),
            )
            B = generate_data(
                shape=(K // group.size(), N),
                dtype=dtype,
                device="cuda",
                scale=0.01 * (rank + 1),
            )

            rs_output_native = native_torch_matmul_reduce_scatter(A, B, scatter_dim, reduce_op, group)
            rs_output_turbo = pt.ops.fused_matmul_reduce_scatter(
                A,
                B,
                layout="NN",
                reduce_op=reduce_op,
                scatter_dim=scatter_dim,
                group_name=group.group_name,
                gemm_streams=self.gemm_streams,
                comm_streams=self.comm_streams,
                comm_method=comm_method,
            )

            assert rs_output_native.stride() == rs_output_turbo.stride()
            diff = (rs_output_native - rs_output_turbo).abs().max()
            diff_mask = rs_output_native != rs_output_turbo
            print(
                f"[rank {torch.distributed.get_rank()}]native vs turbo: diff_mask-{diff_mask.sum()}, diff_max-{diff}, value_range-({rs_output_turbo.min()}, {rs_output_turbo.max()})"
            )
            torch.testing.assert_close(
                rs_output_native.float(), rs_output_turbo.float(), **get_tolerances(dtype)
            )


if __name__ == "__main__":
    run_tests()
