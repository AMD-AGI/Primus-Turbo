###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Tests for the single-arg ``symm_at`` over a self-describing ``SymmTensor``.

Two parts:
  1. single-process device test -- two buffers each with a faked 256B header stand
     in for two ranks; validates the in-kernel ``symm_at`` translation (no dist).
  2. EP8 ring test -- each rank allocates a SymmTensor, writes a rank-stamped
     pattern into its right neighbor via symm_at, then verifies it received the
     expected pattern from its left neighbor.

Run inside dev_primus:
  PYTHONPATH=<...>/Primus-Turbo python tests/pytorch/ops/test_mega_symm_tensor.py
"""

import argparse
import functools
import os

import pytest
import torch

try:
    import flydsl.compiler as flyc
    import flydsl.expr as fx
    from flydsl.expr.buffer_ops import buffer_store

    from primus_turbo.flydsl.mega.prims import addr_buffer_resource
    from primus_turbo.flydsl.mega.symm_tensor import SYMM_HDR_BYTES, SymmTensor, symm_at

    _HAVE_FLYDSL = True
    _IMPORT_ERR = None
except Exception as e:  # pragma: no cover
    _HAVE_FLYDSL = False
    _IMPORT_ERR = e

import torch.distributed as dist

_BIAS = 100  # written value at payload index i is i + _BIAS + rank


@functools.lru_cache(maxsize=None)
def _build(n):
    """Kernel: write ``i + _BIAS + rank_stamp`` into peer ``dst``'s payload via symm_at."""

    @flyc.kernel(known_block_size=[n, 1, 1])
    def k(PAYLOAD: fx.Tensor, dst_rank: fx.Int32, rank_stamp: fx.Int32):
        ti = fx.thread_idx.x
        peer = symm_at(PAYLOAD, dst_rank)
        peer_res = addr_buffer_resource(peer, num_records_bytes=n * 4)
        buffer_store(ti + fx.Int32(_BIAS) + rank_stamp, peer_res, ti)

    @flyc.jit
    def launch(PAYLOAD, dst_rank, rank_stamp, stream: fx.Stream = fx.Stream(None)):
        k(PAYLOAD, dst_rank, rank_stamp).launch(grid=(1, 1, 1), block=(n, 1, 1), stream=stream)

    return launch


# ---------------------------------------------------------------------------
# 1. single-process device test (no distributed)
# ---------------------------------------------------------------------------
def _single_proc_device_test(n=64, world=2):
    """Fake two ranks with two allocations, each = [256B header][payload]."""
    hdr_i64 = SYMM_HDR_BYTES // 8
    elem_off = SYMM_HDR_BYTES // 4  # int32 payload offset

    allocs = [torch.zeros(hdr_i64 + n, dtype=torch.int64, device="cuda") for _ in range(world)]
    bases = [a.data_ptr() for a in allocs]
    for a in allocs:
        a[:world].copy_(torch.tensor(bases, dtype=torch.int64, device="cuda"))

    # int32 payload views at byte offset 256
    payloads = [a.view(torch.int32)[elem_off : elem_off + n] for a in allocs]

    for my_rank in range(world):
        for dst_rank in range(world):
            for p in payloads:
                p.zero_()
            _build(n)(payloads[my_rank], dst_rank, my_rank * 1000, stream=torch.cuda.current_stream())
            torch.cuda.synchronize()
            expected = torch.arange(n, dtype=torch.int32, device="cuda") + _BIAS + my_rank * 1000
            ok_dst = torch.equal(payloads[dst_rank], expected)
            ok_other = all(
                torch.equal(payloads[r], torch.zeros(n, dtype=torch.int32, device="cuda"))
                for r in range(world)
                if r != dst_rank
            )
            print(f"[device symm_at] my={my_rank} dst={dst_rank} dst_ok={ok_dst} others_clean={ok_other}")
            assert ok_dst, f"dst payload wrong:\n{payloads[dst_rank]}\nvs\n{expected}"
            assert ok_other, "another payload was clobbered"
    print("[device symm_at] ALL PASS")


# ---------------------------------------------------------------------------
# 2. multi-rank ring test
# ---------------------------------------------------------------------------
def _run_ring(local_rank, world, n):
    ip = os.getenv("MASTER_ADDR", "127.0.0.1")
    port = int(os.getenv("MASTER_PORT", "8417"))
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", init_method=f"tcp://{ip}:{port}", world_size=world, rank=local_rank)
    torch.set_default_device("cuda")
    group = dist.new_group(list(range(world)))
    rank = dist.get_rank()

    # payload is zero-initialized at allocation (hipMemset); do NOT pre-zero here --
    # caching our own zeros would mask the remote write under gfx950 stale-cache reads
    st = SymmTensor.empty(group, (n,), torch.int32)
    st.barrier()

    dst = (rank + 1) % world  # write to right neighbor
    src = (rank - 1 + world) % world
    _build(n)(st.tensor, dst, rank * 1000, stream=torch.cuda.current_stream())
    torch.cuda.synchronize()  # our remote store retired
    dist.barrier(group)  # host-side strong fence: all ranks finished writing
    torch.cuda.synchronize()

    # we should have received src's pattern in our own payload
    expected = torch.arange(n, dtype=torch.int32, device="cuda") + _BIAS + src * 1000
    ok = torch.equal(st.tensor, expected)
    gathered = [None] * world
    dist.all_gather_object(gathered, (rank, src, ok), group=group)
    if rank == 0:
        for r, s, o in gathered:
            print(f"[ring symm_at] rank={r} recv_from={s} ok={o}")
        assert all(o for _, _, o in gathered), "ring receive mismatch"
        print(f"[ring symm_at] EP{world} ALL PASS")

    st.barrier()
    st.destroy()
    dist.barrier(group)
    dist.destroy_process_group()


# ---------------------------------------------------------------------------
# entry points
# ---------------------------------------------------------------------------
_WORLD = 8


@pytest.mark.skipif(not _HAVE_FLYDSL, reason="flydsl not importable (run inside dev_primus)")
def test_device_symm_at():
    torch.cuda.set_device(0)
    _single_proc_device_test()


@pytest.mark.skipif(not _HAVE_FLYDSL, reason="flydsl not importable (run inside dev_primus)")
@pytest.mark.skipif(torch.cuda.device_count() < _WORLD, reason=f"needs {_WORLD} GPUs for EP{_WORLD}")
def test_ring_symm_at():
    torch.multiprocessing.spawn(_run_ring, args=(_WORLD, 64), nprocs=_WORLD)


def main():
    if not _HAVE_FLYDSL:
        raise SystemExit(f"flydsl not importable: {_IMPORT_ERR}")
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-processes", type=int, default=_WORLD)
    ap.add_argument("--n", type=int, default=64)
    args = ap.parse_args()

    torch.cuda.set_device(0)
    _single_proc_device_test(n=args.n)
    if torch.cuda.device_count() >= args.num_processes:
        torch.multiprocessing.spawn(_run_ring, args=(args.num_processes, args.n), nprocs=args.num_processes)
    else:
        print(f"[ring symm_at] SKIP (need {args.num_processes} GPUs)")


if __name__ == "__main__":
    main()
