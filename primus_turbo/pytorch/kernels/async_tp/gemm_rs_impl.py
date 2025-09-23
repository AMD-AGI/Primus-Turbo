###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from functools import lru_cache
from typing import Any, List

import torch
import torch.distributed.distributed_c10d as c10d
import triton
from hip import hip

from primus_turbo.triton.async_tp.gemm_rs_kernel import (
    kernel_gemm_rs_producer_fuse_scatter,
)
from primus_turbo.triton.reduce.reduce_kernel import kernel_consumer_reduce_async_tp

from .amd_symmetric_memory import get_amd_symm_mem_workspace
from .common_ops import hip_check


def matmul_fuse_scatter(a, b, scatter_bufs_ptr, rank, num_ranks, transpose_weight):
    # Check constraints.
    if transpose_weight:
        assert a.shape[1] == b.shape[0], "Incompatible dimensions"
        K, N = b.shape
        stride_bk, stride_bn = b.stride(0), b.stride(1)
    else:
        assert a.shape[1] == b.shape[1], "Incompatible dimensions"
        N, K = b.shape
        stride_bk, stride_bn = b.stride(1), b.stride(0)
    assert a.is_contiguous(), "Matrix A must be contiguous"
    M, K = a.shape

    alignment = 256
    assert M % alignment == 0 and N % alignment == 0 and K % alignment == 0

    # Allocates output.
    def grid(META):
        return (triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),)

    compiled = kernel_gemm_rs_producer_fuse_scatter[grid](
        a,
        b,
        scatter_bufs_ptr,  #
        rank,
        num_ranks,
        M,
        N,
        K,  #
        a.stride(0),
        a.stride(1),  #
        stride_bk,
        stride_bn,  #
        N,
        1,  #
    )
    return compiled


def ring_reduce_after_scatter(
    rank,
    num_ranks,
    reduce_op,
    scatter_out,  # [M, N]
    output,
    stream,
):
    M, N = scatter_out.shape
    M_per_rank = M // num_ranks
    if output is None:
        output = torch.empty((M_per_rank, N), dtype=scatter_out.dtype, device=scatter_out.device)

    REDUCE_AVG = True if reduce_op == "avg" else False

    def grid(META):
        return (triton.cdiv(M_per_rank * N, META["BLOCK_SIZE"]),)

    with torch.cuda.stream(stream):
        kernel_consumer_reduce_async_tp[grid](
            scatter_out,
            output,
            M_per_rank,
            N,
            rank=rank,
            num_ranks=num_ranks,
            REDUCE_AVG=REDUCE_AVG,
            BLOCK_SIZE=2048,
            num_warps=2,
        )

    return output


def _tiled_fused_matmul_scatter_out_impl(
    input: torch.Tensor,
    weight: torch.Tensor,
    group_name: str,
    reduce_op: str,
    *,
    output: torch.Tensor,
    rs_output: torch.Tensor,
    out_dtype: torch.dtype,
    stream: torch.cuda.Stream
):

    M = input.shape[0]
    N = weight.shape[0]

    group = c10d._resolve_process_group(group_name)
    rank = group.rank()
    num_ranks = group.size()

    p2p_workspace_size_req = M * N * input.element_size()
    symm_mem = get_amd_symm_mem_workspace(group_name, min_size=p2p_workspace_size_req)
    scatter_bufs = [symm_mem.get_buffer(i, [M, N], out_dtype) for i in range(num_ranks)]
    scatter_bufs_ptr = get_scatter_buf_ptrs((t.data_ptr() for t in scatter_bufs))

    symm_mem.barrier()
    matmul_fuse_scatter(input, weight, scatter_bufs_ptr, rank, num_ranks, transpose_weight=False)
    symm_mem.barrier()

    scatter_out = scatter_bufs[rank][:M]

    rs_output = ring_reduce_after_scatter(
        rank,
        num_ranks,
        reduce_op,
        scatter_out,  # [M, N]
        rs_output,
        stream,
    )

    if output is not None:
        output.copy_(scatter_out)
    return rs_output


@lru_cache
def get_scatter_buf_ptrs(scatter_bufs_ptr_cpu):
    return torch.tensor(
        list(scatter_bufs_ptr_cpu),
        device=torch.cuda.current_device(),
        requires_grad=False,
    )


_local_mm_out_bufs_cache = {}


def get_local_mm_out_bufs(M, N, num_splits, dtype, device):
    key = (M, N, num_splits, str(dtype), str(device))
    if key not in _local_mm_out_bufs_cache:
        _local_mm_out_bufs_cache[key] = [
            torch.empty((M // num_splits, N), dtype=dtype, device=device) for _ in range(num_splits)
        ]
    return _local_mm_out_bufs_cache[key]


@lru_cache
def get_events(num_splits):
    return [torch.cuda.Event() for _ in range(num_splits)]


@lru_cache
def extract_indices(chunk_idx, num_ranks, m_per_rank, m_per_chunk):
    start_indices = torch.arange(num_ranks, device=torch.cuda.current_device()) * m_per_rank
    offset = m_per_chunk * chunk_idx + torch.arange(m_per_chunk, device=torch.cuda.current_device())
    row_indices = start_indices[:, None] + offset[None, :]
    row_indices = row_indices.flatten()
    return row_indices


def _pipeline_matmul_scatter_out_impl(
    mm_out_op: torch._ops.OpOverload,
    input: torch.Tensor,
    weight: torch.Tensor,
    group_name: str,
    reduce_op: str,
    num_splits: int,
    enable_sdma: bool,
    gemm_stream_pool: List[torch.cuda.Stream],
    comm_stream_pool: List[torch.cuda.Stream],
    output: torch.Tensor,
    rs_output: torch.Tensor,
    out_dtype: torch.dtype,
):
    M = input.shape[0]
    N = weight.shape[1]

    group = c10d._resolve_process_group(group_name)
    rank = group.rank()
    num_ranks = group.size()
    if enable_sdma:
        hip_memcpy_kind = hip.hipMemcpyKind.hipMemcpyDeviceToDeviceNoCU
    else:
        hip_memcpy_kind = hip.hipMemcpyKind.hipMemcpyDeviceToDevice

    p2p_workspace_size_req = M * N * input.element_size()
    symm_mem = get_amd_symm_mem_workspace(group_name, min_size=p2p_workspace_size_req)
    scatter_bufs = [symm_mem.get_buffer(i, [M, N], out_dtype) for i in range(num_ranks)]

    m_per_rank = M // num_ranks
    m_per_chunk = m_per_rank // num_splits
    local_tensor_buffers = get_local_mm_out_bufs(M, N, num_splits, out_dtype, input.device)

    gemm_events = get_events(num_splits)
    current_stream = torch.cuda.current_stream()
    stream_pool = comm_stream_pool + gemm_stream_pool
    symm_mem.barrier()

    for stream in stream_pool:
        stream.wait_stream(current_stream)

    for chunk_idx in range(num_splits):
        gemm_stream = gemm_stream_pool[chunk_idx % len(gemm_stream_pool)]
        with gemm_stream:
            row_indices = extract_indices(chunk_idx, num_ranks, m_per_rank, m_per_chunk)
            chunk_input = input.index_select(0, row_indices)
            mm_out_op(chunk_input, weight, out=local_tensor_buffers[chunk_idx])
            gemm_events[chunk_idx].record(gemm_stream)

        rank_orders = [(i + chunk_idx) % num_ranks for i in range(num_ranks)]
        for idx, remote_rank in enumerate(rank_orders):
            comm_stream = comm_stream_pool[idx % len(comm_stream_pool)]
            comm_stream.wait_event(gemm_events[chunk_idx])

            data_elem_size = local_tensor_buffers[chunk_idx].element_size()

            M_dst_start_pos = rank * m_per_rank + chunk_idx * m_per_chunk
            M_src_start_pos = remote_rank * m_per_chunk

            src_ptr = local_tensor_buffers[chunk_idx].data_ptr() + M_src_start_pos * N * data_elem_size
            dst_ptr = scatter_bufs[remote_rank].data_ptr() + M_dst_start_pos * N * data_elem_size

            nbytes = m_per_chunk * N * data_elem_size
            cp_res = hip.hipMemcpyAsync(
                dst_ptr,
                src_ptr,
                nbytes,
                hip_memcpy_kind,
                comm_stream.cuda_stream,
            )
            hip_check(cp_res)

    for stream in stream_pool:
        current_stream.wait_stream(stream)
    symm_mem.barrier()

    scatter_out = scatter_bufs[rank][:M]

    rs_output = ring_reduce_after_scatter(
        rank,
        num_ranks,
        reduce_op,
        scatter_out,  # [M, N]
        rs_output,
        stream,
    )

    if output is not None:
        output.copy_(scatter_out)
    return rs_output


def _pipeline_scaled_matmul_scatter_out_impl(
    mm_out_op: torch._ops.OpOverload,
    A: torch.Tensor,
    B: torch.Tensor,
    A_scale: torch.Tensor,
    kwargs: dict[str, Any],
    group_name: str,
    reduce_op: str,
    num_splits: int,
    enable_sdma: bool,
    gemm_stream_pool: List[torch.cuda.Stream],
    comm_stream_pool: List[torch.cuda.Stream],
    output: torch.Tensor,
    rs_output: torch.Tensor,
    out_dtype: torch.dtype,
):
    M = A.shape[0]
    N = B.shape[1]

    group = c10d._resolve_process_group(group_name)
    rank = group.rank()
    num_ranks = group.size()
    if enable_sdma:
        hip_memcpy_kind = hip.hipMemcpyKind.hipMemcpyDeviceToDeviceNoCU
    else:
        hip_memcpy_kind = hip.hipMemcpyKind.hipMemcpyDeviceToDevice

    p2p_workspace_size_req = M * N * out_dtype.itemsize
    symm_mem = get_amd_symm_mem_workspace(group_name, min_size=p2p_workspace_size_req)
    scatter_bufs = [symm_mem.get_buffer(i, [M, N], out_dtype) for i in range(num_ranks)]

    m_per_rank = M // num_ranks
    m_per_chunk = m_per_rank // num_splits
    local_tensor_buffers = get_local_mm_out_bufs(M, N, num_splits, out_dtype, A.device)

    gemm_events = get_events(num_splits)
    current_stream = torch.cuda.current_stream()
    stream_pool = comm_stream_pool + gemm_stream_pool
    symm_mem.barrier()

    for stream in stream_pool:
        stream.wait_stream(current_stream)

    for chunk_idx in range(num_splits):
        gemm_stream = gemm_stream_pool[chunk_idx % len(gemm_stream_pool)]
        with gemm_stream:
            row_indices = extract_indices(chunk_idx, num_ranks, m_per_rank, m_per_chunk)
            A_shard = A.index_select(0, row_indices)
            if A_scale is None:
                mm_out_op(A_shard, B, out=local_tensor_buffers[chunk_idx])
            else:
                if A_scale.numel() > 1:
                    A_shard_scale = A_scale.index_select(0, row_indices)
                    A_shard_scale = (
                        A_shard_scale if A_shard_scale.data_ptr() % 16 == 0 else A_shard_scale.clone()
                    )
                else:
                    A_shard_scale = A_scale
                mm_out_op(A_shard, B, scale_a=A_shard_scale, **kwargs, out=local_tensor_buffers[chunk_idx])
            gemm_events[chunk_idx].record(gemm_stream)

        rank_orders = [(i + chunk_idx) % num_ranks for i in range(num_ranks)]
        for idx, remote_rank in enumerate(rank_orders):
            comm_stream = comm_stream_pool[idx % len(comm_stream_pool)]
            comm_stream.wait_event(gemm_events[chunk_idx])

            data_elem_size = local_tensor_buffers[chunk_idx].element_size()

            M_dst_start_pos = rank * m_per_rank + chunk_idx * m_per_chunk
            M_src_start_pos = remote_rank * m_per_chunk

            src_ptr = local_tensor_buffers[chunk_idx].data_ptr() + M_src_start_pos * N * data_elem_size
            dst_ptr = scatter_bufs[remote_rank].data_ptr() + M_dst_start_pos * N * data_elem_size

            nbytes = m_per_chunk * N * data_elem_size
            cp_res = hip.hipMemcpyAsync(
                dst_ptr,
                src_ptr,
                nbytes,
                hip_memcpy_kind,
                comm_stream.cuda_stream,
            )
            hip_check(cp_res)

    for stream in stream_pool:
        current_stream.wait_stream(stream)
    symm_mem.barrier()

    scatter_out = scatter_bufs[rank][:M]

    rs_output = ring_reduce_after_scatter(
        rank,
        num_ranks,
        reduce_op,
        scatter_out,  # [M, N]
        rs_output,
        stream,
    )

    if output is not None:
        output.copy_(scatter_out)
    return rs_output
