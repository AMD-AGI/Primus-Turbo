from functools import lru_cache
from typing import List, Optional

import pyrocshmem
import torch
import torch.distributed.distributed_c10d as c10d
import triton
import triton.language as tl
from hip import hip
from triton_dist.kernels.amd.common_ops import load_acquire_system, thread_idx
from triton_dist.utils import HIP_CHECK

from .amd_symmetric_memory import get_amd_symm_mem_workspace


@triton.jit
def batch_wait_eq_sys(rank, barrier_ptr, n_elements: tl.constexpr, value):
    barrier_ptr = barrier_ptr.to(tl.pointer_type(tl.int32))
    tid = thread_idx(axis=0)
    if tid == 0:
        for i in tl.static_range(n_elements):
            while i != rank and load_acquire_system(barrier_ptr + i) != value:
                pass

    tl.debug_barrier()


def wait_all_gather(rank, num_ranks, barrier_ptr: int, value):
    batch_wait_eq_sys[(1,)](rank, barrier_ptr, num_ranks, value)


@lru_cache
def get_barriers_tensors(group_name: str, num_splits: int) -> List[torch.Tensor]:
    group = c10d._resolve_process_group(group_name)
    m_chunk_num = group.size() * num_splits
    barriers = pyrocshmem.hipipc_create_tensor_list(group, [m_chunk_num], torch.int32)
    barriers[group.rank()].fill_(0)
    return barriers


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
def get_one_tensor() -> torch.Tensor:
    one = torch.ones((1,), dtype=torch.int32, device="cuda")
    return one


@lru_cache
def get_comm_event():
    comm_event = torch.cuda.Event()
    return comm_event


def _pipeline_fused_all_gather_matmul_impl(
    input: torch.Tensor,
    weight: List[torch.Tensor],
    mm_op: torch._ops.OpOverload,
    group_name: str,
    comm_stream_pool: List[torch.cuda.Stream],
    copy_stream_pool: List[torch.cuda.Stream],
    gemm_stream_pool: List[torch.cuda.Stream],
    num_splits: int,
    *,
    outputs: List[torch.Tensor],
    enable_sdma: bool = False,
    skip_copy_local_A: bool = False,
    return_A=True,
    A_out: Optional[torch.Tensor] = None
):

    local_M, K = input.shape
    group = c10d._resolve_process_group(group_name)
    rank = group.rank()
    num_ranks = group.size()
    if enable_sdma:
        hip_memcpy_kind = hip.hipMemcpyKind.hipMemcpyDeviceToDeviceNoCU
    else:
        hip_memcpy_kind = hip.hipMemcpyKind.hipMemcpyDeviceToDevice

    p2p_workspace_size_req = input.nbytes * (num_ranks - 1)
    symm_mem = get_amd_symm_mem_workspace(group_name, min_size=p2p_workspace_size_req)

    # num_rank x (num_splits, num_ranks)
    barrier_tensors = get_barriers_tensors(group_name, num_splits)
    cpu_indices = get_cpu_split_indices(group_name, num_splits)
    one = get_one_tensor()

    stream_pool = gemm_stream_pool + comm_stream_pool + copy_stream_pool

    input_buffer_shape = [local_M * (num_ranks - 1), K]
    input_ptr = input.data_ptr()

    A_out_ptr = A_out.data_ptr() if return_A else None
    output_ptrs = [o.data_ptr() for o in outputs]
    # (num_ranks, num_splits, -1) --> num_splits x (num_ranks, -1)
    input_buf_ptrs = [symm_mem._buffers[i].data_ptr() for i in range(num_ranks)]
    barrier_ptrs = [t.data_ptr() for t in barrier_tensors]
    local_input_buf = symm_mem.get_buffer(rank, input_buffer_shape, input.dtype)
    local_input_buf_chunk = local_input_buf.chunk(num_splits)

    local_input_buf_chunk_stride = local_input_buf.nbytes // (num_ranks - 1) // num_splits

    # (num_splits, num_ranks - 1, M_PER_CHUNK)
    input_stride = input.nbytes // num_splits
    input_buf_stride = p2p_workspace_size_req // num_splits
    barrier_stride = num_ranks * 4
    A_out_stride = A_out.nbytes // num_splits // num_ranks if return_A else 0
    output_strides = [o.nbytes // num_splits // num_ranks for o in outputs]

    current_stream = torch.cuda.current_stream()

    barrier_tensors[rank].fill_(0)
    symm_mem.barrier()

    comm_event = get_comm_event()

    for st in stream_pool:
        st.wait_stream(current_stream)

    with gemm_stream_pool[0]:
        for w, o in zip(weight, outputs):
            chunked_output = o.chunk(num_ranks)[rank]
            mm_op.out(input, w, out=chunked_output)

    for step in range(num_splits):
        idx = 0
        gemm_stream = gemm_stream_pool[(1 + step) % len(gemm_stream_pool)]
        copy_stream = copy_stream_pool[step % len(copy_stream_pool)]
        for dst_rank in range(num_ranks):
            if dst_rank == rank:
                continue

            dst_offset = rank if rank < dst_rank else rank - 1
            ag_stream = comm_stream_pool[idx % len(comm_stream_pool)]

            HIP_CHECK(
                hip.hipMemcpyAsync(
                    input_buf_ptrs[dst_rank] + dst_offset * input_stride,
                    input_ptr,
                    input_stride,
                    hip_memcpy_kind,
                    ag_stream.cuda_stream,
                )
            )
            HIP_CHECK(
                hip.hipMemcpyAsync(
                    barrier_ptrs[dst_rank] + rank * 4,
                    one.data_ptr(),
                    one.nbytes,
                    hip_memcpy_kind,
                    ag_stream.cuda_stream,
                )
            )
            comm_event.record(ag_stream)

            idx += 1
            input_buf_ptrs[dst_rank] += input_buf_stride
            barrier_ptrs[dst_rank] += barrier_stride

        with gemm_stream:
            wait_all_gather(rank, num_ranks, barrier_ptrs[rank], 1)
            if return_A:
                copy_stream.wait_stream(gemm_stream)
                for i, idx in enumerate(cpu_indices[step]):
                    dst_ptr = A_out_ptr + idx * A_out_stride
                    src_ptr = local_input_buf_chunk[step].data_ptr() + i * local_input_buf_chunk_stride
                    HIP_CHECK(
                        hip.hipMemcpyAsync(
                            dst_ptr,
                            src_ptr,
                            local_input_buf_chunk_stride,
                            hip_memcpy_kind,
                            copy_stream.cuda_stream,
                        )
                    )

            for w, o_ptr, o_stride in zip(weight, output_ptrs, output_strides):
                chunk_out = mm_op(local_input_buf_chunk[step], w)
                chunk_out_stride = chunk_out.nbytes // (num_ranks - 1)

                copy_stream.wait_stream(gemm_stream)
                for i, idx in enumerate(cpu_indices[step]):
                    dst_ptr = o_ptr + idx * o_stride
                    src_ptr = chunk_out.data_ptr() + i * chunk_out_stride
                    HIP_CHECK(
                        hip.hipMemcpyAsync(
                            dst_ptr, src_ptr, chunk_out_stride, hip_memcpy_kind, copy_stream.cuda_stream
                        )
                    )

        input_ptr += input_stride
        barrier_ptrs[rank] += barrier_stride

    if return_A and not skip_copy_local_A:
        for i in range(num_splits):
            idx = rank * num_splits + i
            dst_ptr = A_out_ptr + A_out_stride * idx
            src_ptr = input.data_ptr() + i * input_stride
            HIP_CHECK(
                hip.hipMemcpyAsync(dst_ptr, src_ptr, input_stride, hip_memcpy_kind, copy_stream.cuda_stream)
            )

    for st in stream_pool:
        torch.cuda.current_stream().wait_stream(st)
