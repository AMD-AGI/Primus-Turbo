###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from functools import lru_cache
from itertools import cycle
from typing import Any, Callable, List, Optional, Tuple

import torch
import torch.distributed.distributed_c10d as c10d
from hip import hip
from torch.distributed._symmetric_memory import (
    _check_and_verify_fp8_all_gather_scale_mode,
    _ScaleMode,
)

from .amd_symmetric_memory import get_amd_symm_mem_workspace
from .common_ops import batch_wait_eq_sys, ipc_create_tensor_lists


def hip_check(call_result):
    err = call_result[0]
    result = call_result[1:]
    if len(result) == 1:
        result = result[0]
    if isinstance(err, hip.hipError_t) and err != hip.hipError_t.hipSuccess:
        raise RuntimeError(str(err))
    return result


def wait_all_gather(rank, num_ranks, barrier_ptr: int):
    batch_wait_eq_sys[(1,)](rank, barrier_ptr, num_ranks)


@lru_cache
def get_cpu_split_indices(group_name: str, num_splits: int) -> torch.Tensor:
    group = c10d._resolve_process_group(group_name)
    indices = []
    for i in range(num_splits):
        tmp = []
        for r in range(group.size()):
            if r == group.rank():
                continue
            tmp.append(r * num_splits + i)
        indices.append(tmp)
    return indices


@lru_cache
def get_device_indices(group_name: str, num_splits: int) -> torch.Tensor:
    group = c10d._resolve_process_group(group_name)
    indices = []
    for i in range(num_splits):
        tmp = []
        for r in range(group.size()):
            if r == group.rank():
                continue
            tmp.append(r * num_splits + i)
        indices.append(torch.tensor(tmp, device="cuda"))

    return indices


@lru_cache
def get_one_tensor() -> torch.Tensor:
    one = torch.ones((1,), dtype=torch.int32, device="cuda")
    return one


@lru_cache
def get_comm_event():
    comm_event = torch.cuda.Event()
    return comm_event


@lru_cache
def get_barriers_tensors(group_name: str, num_splits: int) -> List[torch.Tensor]:
    group = c10d._resolve_process_group(group_name)
    m_chunk_num = group.size() * num_splits
    barriers = ipc_create_tensor_lists(group, [m_chunk_num], torch.int32)
    barriers[group.rank()].fill_(0)
    return barriers


def pipelined_all_gather_copy_send(
    dst_ptrs: list[list[int]],
    src_ptrs: list[list[int]],
    send_n_bytes: list[int],
    stream_generator,
    barrier_ptrs,
    rank,
    num_ranks,
    comm_kind_type,
):

    one = get_one_tensor()
    num_ag = len(send_n_bytes)

    comm_event = get_comm_event()

    def _skip_myrank(cur_rank):
        return cur_rank != rank

    for dst_rank in filter(_skip_myrank, range(num_ranks)):
        dst_offset = rank if rank < dst_rank else rank - 1
        ag_stream = next(stream_generator)

        for i in range(num_ag):
            hip_check(
                hip.hipMemcpyAsync(
                    dst_ptrs[dst_rank][i] + dst_offset * send_n_bytes[i],
                    src_ptrs[i],
                    send_n_bytes[i],
                    comm_kind_type,
                    ag_stream.cuda_stream,
                )
            )

            dst_ptrs[dst_rank][i] += send_n_bytes[i] * (num_ranks - 1)

        hip_check(
            hip.hipMemcpyAsync(
                barrier_ptrs[dst_rank] + rank * 4,
                one.data_ptr(),
                one.nbytes,
                comm_kind_type,
                ag_stream.cuda_stream,
            )
        )
        comm_event.record(ag_stream)
        barrier_ptrs[dst_rank] += num_ranks * 4

    class _CustomHandle:

        @staticmethod
        def wait():
            torch.cuda.current_stream().wait_event(comm_event)
            wait_all_gather(rank, num_ranks, barrier_ptrs[rank])

    return _CustomHandle


@lru_cache
def get_symm_shard_buf_and_chunk(
    shard_info: Tuple[Tuple[torch.Size, torch.dtype]],
    group_name: int,
    p2p_workspace_size_req: int,
    num_splits: int,
):
    symm_mem = get_amd_symm_mem_workspace(group_name, min_size=p2p_workspace_size_req)
    local_shard_buf_chunk = [[] for _ in range(num_splits)]
    shard_buf = [[] for _ in range(symm_mem.world_size)]
    offset = 0
    for shape, dtype in shard_info:
        if shape is None:
            continue

        buf_shape = [shape[0] * (symm_mem.world_size - 1)] + list(shape[1:])

        for rank in range(symm_mem.world_size):
            buf = symm_mem.get_buffer(rank, buf_shape, dtype, offset // dtype.itemsize)
            shard_buf[rank].append(buf)
            if rank == symm_mem.rank:
                for i, s in enumerate(buf.chunk(num_splits)):
                    local_shard_buf_chunk[i].append(s)

        offset += buf.nbytes

    return symm_mem, shard_buf, local_shard_buf_chunk


def _pipelined_multi_all_gather_and_consume(
    shard: list[torch.Tensor],
    shard_consumer: Callable[[list[torch.Tensor], int], None],
    ag_out: list[torch.Tensor],
    mm_out: list[torch.Tensor],
    group_name: str,
    comm_stream_pool: List[torch.cuda.Stream],
    copy_stream_pool: List[torch.cuda.Stream],
    gemm_stream_pool: List[torch.cuda.Stream],
    ag_out_needed: bool = True,
    num_splits: int = 2,
    enable_sdma: bool = False,
    skip_copy_local_ag_out: bool = False,
) -> None:
    comm_kind_type = (
        hip.hipMemcpyKind.hipMemcpyDeviceToDeviceNoCU
        if enable_sdma
        else hip.hipMemcpyKind.hipMemcpyDeviceToDevice
    )
    group = c10d._resolve_process_group(group_name)
    rank = group.rank()
    num_ranks = group.size()
    p2p_workspace_size_req = 0
    for t in shard:
        p2p_workspace_size_req += t.nbytes * (num_ranks - 1)

    shard_info = (None if s is None else (s.shape, s.dtype) for s in shard)

    _, shard_buf, local_shard_buf_chunk = get_symm_shard_buf_and_chunk(
        shard_info, group_name, p2p_workspace_size_req, num_splits
    )
    shard_buf_ptrs = [[buf.data_ptr() for buf in bufs] for bufs in shard_buf]
    shard_stride = [s.nbytes // num_splits for s in shard]
    barrier_tensors = get_barriers_tensors(group_name, num_splits)
    barrier_ptrs = [t.data_ptr() for t in barrier_tensors]
    cpu_indices = get_cpu_split_indices(group_name, num_splits)
    stream_pool = gemm_stream_pool + comm_stream_pool + copy_stream_pool
    gemm_stream_generator = cycle(gemm_stream_pool)
    comm_stream_generator = cycle(comm_stream_pool)
    copy_stream_generator = cycle(copy_stream_pool)

    shard_ptrs = [t.data_ptr() for t in shard]
    mm_out_info = [(o.data_ptr(), o.nbytes // num_splits // num_ranks) for o in mm_out]
    saved_tmp_mm_out = []
    ag_out_cp_info = [(ag_o.data_ptr(), ag_o.nbytes // num_splits // num_ranks) for ag_o in ag_out]

    current_stream = torch.cuda.current_stream()

    for st in stream_pool:
        st.wait_stream(current_stream)

    # 2. firstly compute local A_shard
    fist_gemm_stream = next(gemm_stream_generator)
    with fist_gemm_stream:
        local_chunked_outputs = [o.chunk(num_ranks)[rank] for o in mm_out]
        shard_consumer(shard, out=local_chunked_outputs)

    # 3. pipelined ag gemm
    for step in range(num_splits):
        gemm_stream = next(gemm_stream_generator)
        copy_stream = next(copy_stream_generator)

        handle = pipelined_all_gather_copy_send(
            dst_ptrs=shard_buf_ptrs,
            src_ptrs=shard_ptrs,
            send_n_bytes=shard_stride,
            stream_generator=comm_stream_generator,
            barrier_ptrs=barrier_ptrs,
            rank=rank,
            num_ranks=num_ranks,
            comm_kind_type=comm_kind_type,
        )

        with gemm_stream:
            handle.wait()
            if ag_out_needed:
                copy_stream.wait_stream(gemm_stream)
                for j, (ag_ptr, ag_stride) in enumerate(ag_out_cp_info):
                    chunk_size = local_shard_buf_chunk[step][j].nbytes // (num_ranks - 1)
                    ptr = local_shard_buf_chunk[step][j].data_ptr()

                    for i, idx in enumerate(cpu_indices[step]):
                        dst_ptr = ag_ptr + idx * ag_stride
                        src_ptr = ptr + i * chunk_size

                        hip_check(
                            hip.hipMemcpyAsync(
                                dst_ptr,
                                src_ptr,
                                chunk_size,
                                hip.hipMemcpyKind.hipMemcpyDeviceToDevice,
                                copy_stream.cuda_stream,
                            )
                        )

            tmp_outputs = shard_consumer(local_shard_buf_chunk[step])
            saved_tmp_mm_out.append(tmp_outputs)
            for (o_ptr, o_stride), t_o in zip(mm_out_info, tmp_outputs):
                chunk_out_stride = t_o.nbytes // (num_ranks - 1)

                copy_stream.wait_stream(gemm_stream)
                for i, idx in enumerate(cpu_indices[step]):
                    dst_ptr = o_ptr + idx * o_stride
                    src_ptr = t_o.data_ptr() + i * chunk_out_stride
                    hip_check(
                        hip.hipMemcpyAsync(
                            dst_ptr,
                            src_ptr,
                            chunk_out_stride,
                            hip.hipMemcpyKind.hipMemcpyDeviceToDevice,
                            copy_stream.cuda_stream,
                        )
                    )

        barrier_ptrs[rank] += num_ranks * 4
        for i in range(len(shard_stride)):
            shard_ptrs[i] += shard_stride[i]

        if ag_out_needed and not skip_copy_local_ag_out:
            for j, ((ag_ptr, ag_stride), s) in enumerate(zip(ag_out_cp_info, shard)):
                for i in range(num_splits):
                    idx = rank * num_splits + i
                    dst_ptr = ag_ptr + ag_stride * idx
                    src_ptr = s.data_ptr() + i * shard_stride[j]
                    hip_check(
                        hip.hipMemcpyAsync(
                            dst_ptr,
                            src_ptr,
                            shard_stride[j],
                            hip.hipMemcpyKind.hipMemcpyDeviceToDevice,
                            copy_stream.cuda_stream,
                        )
                    )

    for st in stream_pool:
        torch.cuda.current_stream().wait_stream(st)


def _fused_all_gather_matmul_impl(
    mm_out_op: torch._ops.OpOverload,
    A_shard: torch.Tensor,
    Bs: list[torch.Tensor],
    layouts: list[str],
    A_scale: Optional[torch.Tensor],
    kwargs_list: list[dict[str, Any]],
    out_dtypes: list[Optional[torch.dtype]],
    gather_dim: int,
    group_name: str,
    return_A: bool,
    comm_method: str,
    num_splits: int,
    enable_sdma: bool,
    comm_stream_pool: List[torch.cuda.Stream],
    copy_stream_pool: List[torch.cuda.Stream],
    gemm_stream_pool: List[torch.cuda.Stream],
    skip_copy_local_ag_out: bool,
    *,
    A_out: Optional[torch.Tensor] = None,
    mm_out: Optional[List[torch.Tensor]] = None,
) -> tuple[Optional[torch.Tensor], list[torch.Tensor]]:
    if A_shard.dim() < 2:
        raise ValueError("A_shard must be a matrix")
    for B in Bs:
        if B.dim() != 2:
            raise ValueError("B must be a matrix")
    if len(out_dtypes) != len(Bs):
        raise ValueError("len(out_types) must be the same as len(Bs)")
    if len(kwargs_list) != len(Bs):
        raise ValueError("len(kwargs_list) must be the same as len(Bs)")
    if gather_dim < 0 or gather_dim >= A_shard.dim():
        raise ValueError("Invalid gather_dim")

    if len(layouts) != len(Bs):
        raise ValueError("len(layouts) must be the same as len(Bs)")

    if layouts[0][0] != "N":
        raise ValueError("layout must be NN or NT")

    Bs = [B.T if layout[1] == "T" else B for B, layout in zip(Bs, layouts)]

    group = c10d._resolve_process_group(group_name)

    if comm_method == "pipeline":
        multi_all_gather_consume_fn = _pipelined_multi_all_gather_and_consume
        multi_all_gather_consume_kwargs = {"num_splits": num_splits, "enable_sdma": enable_sdma}
    elif comm_method == "ring_exchange":
        raise NotImplementedError("not support yet!")

    # Move the gather_dim to the front and flatten the tensor into a 2D matrix.
    # The flattened tensor doesn't need to be contiguous (for computation
    # efficiency), as _pipelined_all_gather_and_consume guarantees that shards
    # passed to shard_consumer are contiguous.
    A_shard_flat = A_shard.movedim(gather_dim, 0)
    leading_dims = [group.size()] + list(A_shard_flat.shape[:-1])
    A_shard_flat = A_shard_flat.flatten(0, -2)

    # Helper function for reverting the above transformation
    def unflatten(t: torch.Tensor) -> torch.Tensor:
        return t.view(*leading_dims, -1).flatten(0, 1).movedim(0, gather_dim)

    if A_out is None:
        A_out = A_shard_flat.new_empty(
            A_shard_flat.shape[0] * group.size(),
            A_shard_flat.shape[1],
        )

    if mm_out is None:
        mm_out = [
            A_out.new_empty(A_out.shape[0], B.shape[1], dtype=out_dtype or B.dtype)
            for B, out_dtype in zip(Bs, out_dtypes)
        ]

    scale_mode = _check_and_verify_fp8_all_gather_scale_mode(
        shard=A_shard, scale=A_scale, gather_dim=gather_dim, group_size=group.size()
    )

    # Computing block-wise matmul along the first dim of A

    if scale_mode == _ScaleMode.ROW_WISE_SHARDED:
        assert A_scale is not None
        A_scale_shard = A_scale.movedim(gather_dim, 0).flatten(0, -2)
        A_scale_flat = A_scale_shard.new_empty(
            A_scale_shard.shape[0] * group.size(),
            A_scale_shard.shape[1],
        )

        def row_wise_sharded_consumer(
            shard: list[torch.Tensor], out: Optional[list[torch.Tensor]] = None
        ) -> None:
            if out is None:
                out = [
                    A_out.new_empty((shard[0].shape[0], B.shape[1]), dtype=out_dtype or B.dtype)
                    for B, out_dtype in zip(Bs, out_dtypes)
                ]

            for B, o, kwargs in zip(Bs, out, kwargs_list):
                mm_out_op(
                    shard[0],
                    B,
                    scale_a=shard[1],
                    **kwargs,
                    out=o,
                )
            return out

        multi_all_gather_consume_fn(
            [A_shard_flat, A_scale_shard],
            row_wise_sharded_consumer,
            [A_out, A_scale_flat],
            mm_out,
            group_name,
            comm_stream_pool,
            copy_stream_pool,
            gemm_stream_pool,
            return_A,
            skip_copy_local_ag_out=skip_copy_local_ag_out,
            **multi_all_gather_consume_kwargs,
        )
    elif scale_mode == _ScaleMode.ROW_WISE_REPLICATED:
        assert A_scale is not None
        A_scale_shard = A_scale.movedim(gather_dim, 0).flatten(0, -2).chunk(group.size())[group.rank()]
        A_scale_flat = A_scale_shard.new_empty(
            A_scale_shard.shape[0] * group.size(),
            A_scale_shard.shape[1],
        )

        def row_wise_replicated_consumer(
            shard: list[torch.Tensor], out: Optional[list[torch.Tensor]] = None
        ) -> None:
            if out is None:
                out = [
                    A_out.new_empty((shard[0].shape[0], B.shape[1]), dtype=out_dtype or B.dtype)
                    for B, out_dtype in zip(Bs, out_dtypes)
                ]

            for B, o, kwargs in zip(Bs, out, kwargs_list):
                mm_out_op(
                    shard[0],
                    B,
                    scale_a=shard[1],
                    **kwargs,
                    out=o,
                )
            return out

        multi_all_gather_consume_fn(
            [A_shard_flat, A_scale_shard],
            row_wise_replicated_consumer,
            [A_out, A_scale_flat],
            mm_out,
            group_name,
            comm_stream_pool,
            copy_stream_pool,
            gemm_stream_pool,
            return_A,
            skip_copy_local_ag_out=skip_copy_local_ag_out,
            **multi_all_gather_consume_kwargs,
        )
    else:
        if scale_mode == _ScaleMode.TENSOR_WISE:
            assert A_scale is not None
            for kwargs in kwargs_list:
                kwargs["scale_a"] = A_scale
        else:
            assert scale_mode == _ScaleMode.UNSCALED

        def default_consumer(shard: list[torch.Tensor], out: Optional[List[torch.Tensor]] = None) -> None:
            if out is None:
                out = [
                    A_out.new_empty((shard[0].shape[0], B.shape[1]), dtype=out_dtype or B.dtype)
                    for B, out_dtype in zip(Bs, out_dtypes)
                ]

            for B, o, kwargs in zip(Bs, out, kwargs_list):
                mm_out_op(
                    shard[0],
                    B,
                    **kwargs,
                    out=o,
                )
            return out

        multi_all_gather_consume_fn(
            [A_shard_flat],
            default_consumer,
            [A_out],
            mm_out,
            group_name,
            comm_stream_pool,
            copy_stream_pool,
            gemm_stream_pool,
            return_A,
            skip_copy_local_ag_out=skip_copy_local_ag_out,
            **multi_all_gather_consume_kwargs,
        )

    A = unflatten(A_out) if return_A else None
    return A, [unflatten(output) for output in mm_out]
