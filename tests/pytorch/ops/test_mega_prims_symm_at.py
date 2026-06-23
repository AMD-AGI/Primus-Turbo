###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Single-GPU correctness test for ``primus_turbo.flydsl.mega.prims.symm_at``.

No multi-rank needed: two buffers on one device stand in for two ranks' symmetric
heaps. The base table holds both data_ptrs; the kernel runs as "rank 0" and uses
``symm_at`` to translate its local address to a chosen ``dst_rank``, then writes a
known pattern through the translated address. We then check the pattern landed in
the right buffer (translation) and not the other.

Run inside dev_primus:
  PYTHONPATH=<...>/Primus-Turbo python tests/pytorch/ops/test_mega_prims_symm_at.py
"""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl.expr.buffer_ops import buffer_store, create_buffer_resource

from primus_turbo.flydsl.mega.prims import (
    addr_buffer_resource,
    heap_base,
    symm_at,
    symm_at_offset,
    tensor_base,
)

_BIAS = 100  # written value at index i is i + _BIAS


# plain-Python helper (NOT @flyc.kernel): the compile-time use_offset branch lives
# here so it folds at trace time -- the kernel body's `if` becomes scf.if and would
# not export `peer`.
def _peer_addr(peer_base_res, local_tensor, my_rank, dst_rank, use_offset):
    if use_offset:
        # offset form: peer_base[dst] + 0 (sub-buffer at heap start)
        return symm_at_offset(peer_base_res, fx.Int32(dst_rank), 0)
    # pointer-translation form: local + (peer_base[dst] - my_base)
    my_base = heap_base(peer_base_res, fx.Int32(my_rank))
    local = tensor_base(local_tensor)
    return symm_at(local, peer_base_res, my_base, fx.Int32(dst_rank))


@functools.lru_cache(maxsize=None)
def _build(n, my_rank, dst_rank, use_offset):
    """Compile a kernel that writes ``i + _BIAS`` into peer ``dst_rank``[i] via symm_at."""

    @flyc.kernel(known_block_size=[n, 1, 1])
    def k(PEER_BASE: fx.Tensor, LOCAL: fx.Tensor):
        ti = fx.thread_idx.x
        peer_base_res = create_buffer_resource(PEER_BASE, max_size=True)
        peer = _peer_addr(peer_base_res, LOCAL, my_rank, dst_rank, use_offset)
        peer_res = addr_buffer_resource(peer, num_records_bytes=n * 4)
        buffer_store(ti + fx.Int32(_BIAS), peer_res, ti)

    @flyc.jit
    def launch(PEER_BASE, LOCAL, stream: fx.Stream = fx.Stream(None)):
        k(PEER_BASE, LOCAL).launch(grid=(1, 1, 1), block=(n, 1, 1), stream=stream)

    return launch


def _run_case(my_rank, dst_rank, use_offset, n=64):
    buf_a = torch.zeros(n, dtype=torch.int32, device="cuda")  # "rank 0" heap
    buf_b = torch.zeros(n, dtype=torch.int32, device="cuda")  # "rank 1" heap
    table = torch.tensor([buf_a.data_ptr(), buf_b.data_ptr()], dtype=torch.int64, device="cuda")
    local = buf_a if my_rank == 0 else buf_b
    bufs = (buf_a, buf_b)

    _build(n, my_rank, dst_rank, use_offset)(table, local, stream=torch.cuda.current_stream())
    torch.cuda.synchronize()

    expected = torch.arange(n, dtype=torch.int32, device="cuda") + _BIAS
    dst, other = bufs[dst_rank], bufs[1 - dst_rank]
    ok_dst = torch.equal(dst, expected)
    ok_other = torch.equal(other, torch.zeros(n, dtype=torch.int32, device="cuda"))
    tag = "offset" if use_offset else "translate"
    print(
        f"[symm_at {tag}] my_rank={my_rank} dst_rank={dst_rank} "
        f"dst_written={ok_dst} other_untouched={ok_other}"
    )
    assert ok_dst, f"dst buffer not written correctly:\n{dst}\nvs\n{expected}"
    assert ok_other, f"other buffer was clobbered:\n{other}"


def main():
    torch.cuda.set_device(0)
    # translation form
    _run_case(my_rank=0, dst_rank=1, use_offset=False)  # cross: rank0 -> rank1
    _run_case(my_rank=0, dst_rank=0, use_offset=False)  # self identity: rank0 -> rank0
    _run_case(my_rank=1, dst_rank=0, use_offset=False)  # cross: rank1 -> rank0
    # offset form
    _run_case(my_rank=0, dst_rank=1, use_offset=True)
    _run_case(my_rank=0, dst_rank=0, use_offset=True)
    print("[prims.symm_at] ALL PASS")


if __name__ == "__main__":
    main()
