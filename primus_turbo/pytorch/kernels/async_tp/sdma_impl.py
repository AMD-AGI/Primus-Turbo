###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""AG-GEMM and GEMM-RS on ``SdmaCommunicator`` (``comm_method="sdma"``).

Layer L2 of ``docs/design/collective_gemm_mi455.html`` in its GEMM-agnostic form: the GEMM is any
``mm_out_op`` run per chunk on the caller's stream, and the stream waits per chunk for its flag.
The SDMA engines move the data, so the GEMM keeps every CU and no extra streams are needed.

* AG-GEMM: each rank puts its shards, in ``num_splits`` row chunks, into every peer's bank; the
  local shard is multiplied first while those copies run, then each peer's chunks in the order
  they are sent (rank ``r`` is sent to ``r + 1`` first, so this rank hears from ``rank - 1`` first).
* GEMM-RS: each rank multiplies the rows owned by each peer, chunk by chunk, and puts every
  partial into that peer's bank as soon as it is computed; its own rows come last, then the
  owner sums the ``world`` partials in rank order (fp32 accumulation): bitwise reproducible.
"""

from typing import Callable, List, Optional

import torch
import torch.distributed as dist
import torch.distributed.distributed_c10d as c10d

from primus_turbo.pytorch.dist.sdma_comm import SdmaCommunicator

_ALIGN = 256
_comms: dict = {}


def _align(n: int, a: int = _ALIGN) -> int:
    return (n + a - 1) // a * a


def get_sdma_communicator(group: dist.ProcessGroup, bytes_per_rank: int, max_chunks: int) -> SdmaCommunicator:
    """The group's communicator, rebuilt (collectively: every rank asks for the same sizes) when a
    call needs a larger bank or more flags than it has."""
    key = group.group_name
    comm = _comms.get(key)
    if comm is not None and (
        comm.bank_bytes < _align(group.size() * bytes_per_rank) or comm.max_chunks < max_chunks
    ):
        bytes_per_rank = max(bytes_per_rank, comm.bank_bytes // group.size())
        max_chunks = max(max_chunks, comm.max_chunks)
        comm.destroy()
        comm = None
    if comm is None:
        comm = SdmaCommunicator(group, bytes_per_rank, max_chunks=max_chunks)
        _comms[key] = comm
    return comm


def _row_chunks(rows: int, num_splits: int) -> List[tuple]:
    if rows == 0:
        return [(0, 0)]
    size = -(-rows // max(1, min(num_splits, rows)))
    return [(r0, min(size, rows - r0)) for r0 in range(0, rows, size)]


def _sdma_multi_all_gather_and_consume(
    shard: List[torch.Tensor],
    shard_consumer: Callable[..., List[torch.Tensor]],
    ag_out: List[torch.Tensor],
    mm_out: List[torch.Tensor],
    group_name: str,
    comm_stream_pool: List[torch.cuda.Stream],
    copy_stream_pool: List[torch.cuda.Stream],
    gemm_stream_pool: List[torch.cuda.Stream],
    ag_out_needed: bool = True,
    num_splits: int = 2,
    skip_copy_local_ag_out: bool = False,
    **_,
) -> None:
    """Same contract as ``ag_gemm_impl._pipelined_multi_all_gather_and_consume``: ``shard`` are 2-D
    row shards gathered rank-major into ``ag_out``; ``shard_consumer(chunks, out=...)`` multiplies
    matching row chunks of every shard into ``out``, row slices of each ``mm_out``. Runs on the
    current stream; the stream pools are unused."""
    group = c10d._resolve_process_group(group_name)
    rank, world = group.rank(), group.size()
    shard = [s.contiguous() for s in shard]
    rows = shard[0].shape[0]
    chunks = _row_chunks(rows, num_splits)
    row_bytes = [s[0].numel() * s.element_size() if rows else 0 for s in shard]
    # bank layout: region j holds [world, rows, ...] of shard j
    region = []
    offset = 0
    for s in shard:
        region.append(offset)
        offset += _align(world * s.numel() * s.element_size())
    comm = get_sdma_communicator(group, -(-offset // world), len(chunks) * len(shard))

    comm.begin()
    banks = [comm.bank((world, *s.shape), s.dtype, offset=off) for s, off in zip(shard, region)]
    for c, (r0, nr) in enumerate(chunks):
        for k in range(1, world):
            peer = (rank + k) % world
            for j, s in enumerate(shard):
                comm.put(
                    peer,
                    region[j] + (rank * rows + r0) * row_bytes[j],
                    s.data_ptr() + r0 * row_bytes[j],
                    nr * row_bytes[j],
                    flag=c * len(shard) + j,
                )
    comm.release()

    def rows_of(t: torch.Tensor, src: int, r0: int, nr: int) -> torch.Tensor:
        return t[src * rows + r0 : src * rows + r0 + nr]

    shard_consumer(shard, out=[o[rank * rows : (rank + 1) * rows] for o in mm_out])
    if ag_out_needed and not skip_copy_local_ag_out:
        for a, s in zip(ag_out, shard):
            a[rank * rows : (rank + 1) * rows].copy_(s)
    for k in range(1, world):
        src = (rank - k) % world
        for c, (r0, nr) in enumerate(chunks):
            for j in range(len(shard)):
                comm.wait_flag(src, c * len(shard) + j)
            pieces = [b[src, r0 : r0 + nr] for b in banks]
            shard_consumer(pieces, out=[rows_of(o, src, r0, nr) for o in mm_out])
            if ag_out_needed:
                for a, p in zip(ag_out, pieces):
                    rows_of(a, src, r0, nr).copy_(p)
    comm.wait(len(chunks) * len(shard))


def _sdma_matmul_reduce_scatter_impl(
    mm_out_op: torch._ops.OpOverload,
    input: torch.Tensor,
    weight: torch.Tensor,
    group_name: str,
    reduce_op: str,
    num_splits: int,
    output: Optional[torch.Tensor],
    rs_output: torch.Tensor,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """``rs_output`` = reduce over ranks of rows ``[rank * m, (rank + 1) * m)`` of ``input @ weight``.
    ``output``, if given (``[M, N]``), receives the ``world`` partials of this rank's rows, rank-major,
    as the pipeline method's scatter buffer."""
    group = c10d._resolve_process_group(group_name)
    rank, world = group.rank(), group.size()
    M, N = input.shape[0], weight.shape[1]
    if M % world:
        raise ValueError(f"M ({M}) must be divisible by the group size ({world})")
    m = M // world
    input = input.contiguous()
    chunks = _row_chunks(m, num_splits)
    row_bytes = N * torch.empty((), dtype=out_dtype).element_size()
    comm = get_sdma_communicator(group, m * row_bytes, len(chunks))
    # partials headed to peers must stay intact until their copies finish: one slot per (peer, chunk)
    staging = input.new_empty((M, N), dtype=out_dtype)

    comm.begin()
    bank = comm.bank((world, m, N), out_dtype)
    for k in range(1, world):
        dst = (rank + k) % world
        for c, (r0, nr) in enumerate(chunks):
            rows = slice(dst * m + r0, dst * m + r0 + nr)
            mm_out_op(input[rows], weight, out=staging[rows])
            comm.put(dst, (rank * m + r0) * row_bytes, staging[rows].data_ptr(), nr * row_bytes, flag=c)
            comm.release()
    mine = slice(rank * m, (rank + 1) * m)
    mm_out_op(input[mine], weight, out=staging[mine])
    bank[rank].copy_(staging[mine])
    comm.wait(len(chunks))
    if output is not None:
        output.copy_(bank.view(M, N))
    red = bank.sum(0, dtype=torch.float32)
    if reduce_op == "avg":
        red.div_(world)
    rs_output.copy_(red.view(rs_output.shape))
    return rs_output
