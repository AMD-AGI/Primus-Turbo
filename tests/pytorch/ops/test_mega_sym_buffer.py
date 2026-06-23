###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""EP8 tests for the DeepGEMM-style FlyDSL ``SymBuffer`` (one delta table / arena).

Two parts:
  1. arena multi-buffer: one SymBuffer arena, two sub-buffers (IN, OUT); a single
     kernel uses the SAME ``.offsets`` table to map BOTH an input ptr and an output
     ptr to peers -> proves one table serves the whole arena.
  2. all_gather rebuilt on SymBuffer (push): each rank writes its input chunk into
     slot ``self`` of every peer's output via ``sym_map``. Checked vs NCCL + perf.

Run inside dev_primus (8 GPUs):
  PYTHONPATH=<...>/Primus-Turbo python tests/pytorch/ops/test_mega_sym_buffer.py \
      --num-processes 8 --elems 1048576 --iters 50
"""

import argparse
import functools
import os

import pytest
import torch
import torch.distributed as dist

try:
    import flydsl.compiler as flyc
    import flydsl.expr as fx
    from flydsl.expr.buffer_ops import buffer_load, buffer_store, create_buffer_resource

    from primus_turbo.flydsl.mega.prims import addr_buffer_resource
    from primus_turbo.flydsl.mega.sym_buffer import SymBuffer, sym_map_tensor

    _HAVE_FLYDSL = True
    _IMPORT_ERR = None
except Exception as e:  # pragma: no cover
    _HAVE_FLYDSL = False
    _IMPORT_ERR = e

_BIG = 100_000_000  # per-rank pattern offset (distinct ranks, no int32 overflow)
_VEC = 4  # 4 x i32 = 16B per thread
_BLOCK = 256
_WORLD = 8


# ---------------------------------------------------------------------------
# 1. arena multi-buffer kernel: one offsets table maps two sub-buffers
# ---------------------------------------------------------------------------
@functools.lru_cache(maxsize=None)
def _build_copy2(world, n, block):
    """For each peer p: push local IN chunk into slot ``self`` of peer's OUTA and OUTB.

    OUTA gets the value, OUTB gets value+7. Both destinations are mapped with the SAME
    offsets table, exercising the arena property (OUTA and OUTB are distinct sub-buffers
    of one SymBuffer; IN is a read-only source sub-buffer)."""
    n_vec = n // _VEC
    ny = (n_vec + block - 1) // block

    @flyc.kernel(known_block_size=[block, 1, 1])
    def k(IN: fx.Tensor, OUTA: fx.Tensor, OUTB: fx.Tensor, OFFSETS: fx.Tensor, self_rank: fx.Int32):
        peer = fx.block_idx.x
        vg = fx.block_idx.y * fx.Int32(block) + fx.thread_idx.x
        elem = vg * fx.Int32(_VEC)
        mask = elem < fx.Int32(n)

        off_res = create_buffer_resource(OFFSETS, max_size=True)
        in_res = create_buffer_resource(IN, max_size=True)
        val = buffer_load(in_res, elem, vec_width=_VEC, dtype=fx.T.i32())

        # same offsets table maps two distinct sub-buffer ptrs (OUTA, OUTB) to the peer
        peer_a = sym_map_tensor(OUTA, off_res, peer)
        peer_b = sym_map_tensor(OUTB, off_res, peer)
        slot = self_rank * fx.Int32(n) + elem
        buffer_store(val, addr_buffer_resource(peer_a, num_records_bytes=world * n * 4), slot, mask=mask)
        buffer_store(
            val + fx.Int32(7), addr_buffer_resource(peer_b, num_records_bytes=world * n * 4), slot, mask=mask
        )

    @flyc.jit
    def launch(IN, OUTA, OUTB, OFFSETS, self_rank, stream: fx.Stream = fx.Stream(None)):
        k(IN, OUTA, OUTB, OFFSETS, self_rank).launch(grid=(world, ny, 1), block=(block, 1, 1), stream=stream)

    return launch


# ---------------------------------------------------------------------------
# 2. all_gather (push) on SymBuffer
# ---------------------------------------------------------------------------
@functools.lru_cache(maxsize=None)
def _build_allgather(world, n, block):
    n_vec = n // _VEC
    ny = (n_vec + block - 1) // block

    @flyc.kernel(known_block_size=[block, 1, 1])
    def k(IN: fx.Tensor, OUT: fx.Tensor, OFFSETS: fx.Tensor, self_rank: fx.Int32):
        peer = fx.block_idx.x
        vg = fx.block_idx.y * fx.Int32(block) + fx.thread_idx.x
        elem = vg * fx.Int32(_VEC)
        mask = elem < fx.Int32(n)

        off_res = create_buffer_resource(OFFSETS, max_size=True)
        in_res = create_buffer_resource(IN, max_size=True)
        val = buffer_load(in_res, elem, vec_width=_VEC, dtype=fx.T.i32())

        peer_out = sym_map_tensor(OUT, off_res, peer)
        peer_res = addr_buffer_resource(peer_out, num_records_bytes=world * n * 4)
        buffer_store(val, peer_res, self_rank * fx.Int32(n) + elem, mask=mask)

    @flyc.jit
    def launch(IN, OUT, OFFSETS, self_rank, stream: fx.Stream = fx.Stream(None)):
        k(IN, OUT, OFFSETS, self_rank).launch(grid=(world, ny, 1), block=(block, 1, 1), stream=stream)

    return launch


def _bench(fn, iters, group):
    fn()
    torch.cuda.synchronize()
    dist.barrier(group)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters * 1e3  # us/iter


def _run(local_rank, world, args):
    ip = os.getenv("MASTER_ADDR", "127.0.0.1")
    port = int(os.getenv("MASTER_PORT", "8421"))
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", init_method=f"tcp://{ip}:{port}", world_size=world, rank=local_rank)
    torch.set_default_device("cuda")
    group = dist.new_group(list(range(world)))
    rank = dist.get_rank()
    n = args.elems

    # one arena, sub-buffers: read-only source IN[n] + two outputs OUTA/OUTB[world*n]
    arena = SymBuffer.create(group, (n + 2 * world * n) * 4 + 4096)
    inp = arena.alloc((n,), torch.int32)  # this rank's input chunk (read-only source)
    outa = arena.alloc((world * n,), torch.int32)
    outb = arena.alloc((world * n,), torch.int32)
    inp.copy_(torch.arange(n, dtype=torch.int32, device="cuda") + rank * _BIG)
    arena.barrier()

    exp_in = torch.empty(world * n, dtype=torch.int32, device="cuda")
    for p in range(world):
        exp_in[p * n : (p + 1) * n] = torch.arange(n, dtype=torch.int32, device="cuda") + p * _BIG

    # ---- part 1: one offsets table maps two distinct sub-buffers (OUTA, OUTB) ----
    _build_copy2(world, n, _BLOCK)(inp, outa, outb, arena.offsets, rank, stream=torch.cuda.current_stream())
    torch.cuda.synchronize()
    dist.barrier(group)
    torch.cuda.synchronize()
    ok_a = torch.equal(outa, exp_in)
    ok_b = torch.equal(outb, exp_in + 7)
    res = [None] * world
    dist.all_gather_object(res, (rank, ok_a, ok_b), group=group)
    if rank == 0:
        assert all(a and b for _, a, b in res), f"arena multi-buffer FAILED: {res}"
        print(f"[sym_buffer arena] one offsets table maps OUTA+OUTB sub-buffers: PASS (world={world})")

    out = outa  # reuse OUTA for the all_gather part
    out.zero_()
    arena.barrier()

    # ---- part 2: all_gather rebuilt on SymBuffer ----
    launch = _build_allgather(world, n, _BLOCK)

    def _allgather():
        launch(inp, out, arena.offsets, rank, stream=torch.cuda.current_stream())

    _allgather()
    torch.cuda.synchronize()
    dist.barrier(group)
    torch.cuda.synchronize()

    ref = torch.empty(world * n, dtype=torch.int32, device="cuda")
    dist.all_gather_into_tensor(ref, inp.contiguous(), group=group)
    ok_pat = torch.equal(out, exp_in)
    ok_nccl = torch.equal(out, ref)
    res2 = [None] * world
    dist.all_gather_object(res2, (rank, ok_pat, ok_nccl), group=group)
    if rank == 0:
        assert all(a and b for _, a, b in res2), f"all_gather FAILED: {res2}"
        print(f"[sym_buffer all_gather] correctness PASS (N={n} int32 = {n*4/1e6:.1f} MB/rank)")

    # ---- perf vs NCCL ----
    if args.iters > 0:
        bytes_per_rank = (world - 1) * n * 4
        t_fly = _bench(_allgather, args.iters, group)
        ref_in = inp.contiguous()

        def _nccl():
            dist.all_gather_into_tensor(ref, ref_in, group=group)

        t_nccl = _bench(_nccl, args.iters, group)
        if rank == 0:
            bw_fly = bytes_per_rank / (t_fly * 1e-6) / 1e9
            bw_nccl = bytes_per_rank / (t_nccl * 1e-6) / 1e9
            print(
                f"[perf] SymBuffer sym_map: {t_fly:8.2f} us  {bw_fly:7.1f} GB/s/rank   |   "
                f"NCCL: {t_nccl:8.2f} us  {bw_nccl:7.1f} GB/s/rank   |   fly/nccl = {t_nccl/t_fly:.2f}x"
            )

    dist.barrier(group)
    arena.destroy()
    dist.destroy_process_group()


def _argparser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-processes", type=int, default=_WORLD)
    ap.add_argument("--elems", type=int, default=1 << 20)
    ap.add_argument("--iters", type=int, default=50)
    return ap


@pytest.mark.skipif(not _HAVE_FLYDSL, reason="flydsl not importable (run inside dev_primus)")
@pytest.mark.skipif(torch.cuda.device_count() < _WORLD, reason=f"needs {_WORLD} GPUs")
def test_sym_buffer():
    args = _argparser().parse_args(["--elems", str(1 << 18), "--iters", "20"])
    torch.multiprocessing.spawn(_run, args=(_WORLD, args), nprocs=_WORLD)


def main():
    if not _HAVE_FLYDSL:
        raise SystemExit(f"flydsl not importable: {_IMPORT_ERR}")
    args = _argparser().parse_args()
    torch.multiprocessing.spawn(_run, args=(args.num_processes, args), nprocs=args.num_processes)


if __name__ == "__main__":
    main()
