import torch
import triton
import triton.language as tl

import primus_turbo


@triton.jit
def batch_wait_eq_sys(rank, barrier_ptr, n_elements: tl.constexpr):
    barrier_ptr = barrier_ptr.to(tl.pointer_type(tl.int32))
    for i in range(n_elements):
        while i != rank and tl.atomic_cas(barrier_ptr + i, 1, 0, scope="sys", sem="acq_rel") != 1:
            pass

    tl.debug_barrier()


@triton.jit
def barrier_all_ipc(rank, num_ranks, comm_buf_base_ptrs):
    for i in range(num_ranks):
        remote_base_ptr = tl.load(comm_buf_base_ptrs + i).to(tl.pointer_type(tl.int32))
        while tl.atomic_cas(remote_base_ptr + rank, 0, 1, scope="sys", sem="release") != 0:
            pass

    for i in range(num_ranks):
        local_base_ptr = tl.load(comm_buf_base_ptrs + rank).to(tl.pointer_type(tl.int32))
        while tl.atomic_cas(local_base_ptr + i, 1, 0, scope="sys", sem="acquire") != 1:
            pass

    tl.debug_barrier()


def barrier_all_on_stream(
    rank,
    num_ranks,
    sync_bufs_ptr,
    stream,
):
    with torch.cuda.stream(stream):
        barrier_all_ipc[(1,)](rank, num_ranks, sync_bufs_ptr)


def ipc_create_tensor_lists(group: torch.distributed.ProcessGroup, shape: list[int], dtype: torch.dtype):
    return primus_turbo.pytorch._C.rendezvous_shmem(group.group_name, shape, dtype)
