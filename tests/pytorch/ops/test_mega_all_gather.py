###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""8-process all_gather built from FlyDSL ``symm_at`` over symmetric memory.

PUSH design (mirrors the proven ring test: remote write + local read + host fence):
  - output is a symmetric tensor of shape [world * N]; input is a local [N] tensor.
  - each rank copies its local input into slot ``self`` of EVERY peer's output via
    ``symm_at(out_payload, peer)`` -- a vectorized 4xi32 (16B) buffer store.
  - after a host barrier each rank reads its own output (written by all peers).

Correctness is checked against ``torch.distributed.all_gather_into_tensor`` (and an
analytic pattern); performance is compared to the same NCCL collective.

Run inside dev_primus (8 GPUs):
  PYTHONPATH=<...>/Primus-Turbo python tests/pytorch/ops/test_mega_all_gather.py \
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
    from primus_turbo.flydsl.mega.symm_tensor import SymmTensor, symm_at

    _HAVE_FLYDSL = True
    _IMPORT_ERR = None
except Exception as e:  # pragma: no cover
    _HAVE_FLYDSL = False
    _IMPORT_ERR = e

_BIG = 100_000_000  # per-rank pattern offset (keeps ranks distinct, no int32 overflow)
_VEC = 4  # 4 x i32 = 16B per thread
_BLOCK = 256


@functools.lru_cache(maxsize=None)
def _build(world, n, block):
    """Compile a push all_gather kernel for [N]-int32 chunks over ``world`` ranks."""
    assert n % _VEC == 0, "elems per rank must be a multiple of 4"
    n_vec = n // _VEC  # number of 16B groups per chunk
    ny = (n_vec + block - 1) // block

    @flyc.kernel(known_block_size=[block, 1, 1])
    def k(IN: fx.Tensor, OUT: fx.Tensor, self_rank: fx.Int32):
        peer = fx.block_idx.x  # destination rank
        vg = fx.block_idx.y * fx.Int32(block) + fx.thread_idx.x
        elem = vg * fx.Int32(_VEC)  # scalar element offset within the chunk
        mask = elem < fx.Int32(n)

        in_res = create_buffer_resource(IN, max_size=True)
        val = buffer_load(in_res, elem, vec_width=_VEC, dtype=fx.T.i32())

        peer_out = symm_at(OUT, peer)  # peer's output base (i64)
        peer_res = addr_buffer_resource(peer_out, num_records_bytes=world * n * 4)
        out_off = self_rank * fx.Int32(n) + elem  # slot `self` in peer's output
        buffer_store(val, peer_res, out_off, mask=mask)

    @flyc.jit
    def launch(IN, OUT, self_rank, stream: fx.Stream = fx.Stream(None)):
        k(IN, OUT, self_rank).launch(grid=(world, ny, 1), block=(block, 1, 1), stream=stream)

    return launch


def _bench(fn, iters, group):
    fn()  # warmup + compile
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
    port = int(os.getenv("MASTER_PORT", "8419"))
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", init_method=f"tcp://{ip}:{port}", world_size=world, rank=local_rank)
    torch.set_default_device("cuda")
    group = dist.new_group(list(range(world)))
    rank = dist.get_rank()
    n = args.elems

    # local input + symmetric output [world*N]
    inp = torch.arange(n, dtype=torch.int32, device="cuda") + rank * _BIG
    out_st = SymmTensor.empty(group, (world * n,), torch.int32)
    launch = _build(world, n, _BLOCK)

    def _allgather():
        launch(inp, out_st.tensor, rank, stream=torch.cuda.current_stream())

    # ---- correctness ----
    _allgather()
    torch.cuda.synchronize()
    dist.barrier(group)  # host-side ordering fence (data written before read)
    torch.cuda.synchronize()

    expected = torch.empty(world * n, dtype=torch.int32, device="cuda")
    for p in range(world):
        expected[p * n : (p + 1) * n] = torch.arange(n, dtype=torch.int32, device="cuda") + p * _BIG
    ok_pattern = torch.equal(out_st.tensor, expected)

    # cross-check vs NCCL all_gather
    ref = torch.empty(world * n, dtype=torch.int32, device="cuda")
    dist.all_gather_into_tensor(ref, inp, group=group)
    ok_nccl = torch.equal(out_st.tensor, ref)

    res = [None] * world
    dist.all_gather_object(res, (rank, ok_pattern, ok_nccl), group=group)
    if rank == 0:
        for r, op, on in res:
            print(f"[all_gather] rank={r} pattern_ok={op} matches_nccl={on}")
        assert all(op and on for _, op, on in res), "all_gather correctness FAILED"
        print(f"[all_gather] correctness PASS (world={world}, N={n} int32 = {n*4/1e6:.1f} MB/rank)")

    # ---- perf: FlyDSL symm_at vs NCCL ----
    if args.iters > 0:
        bytes_per_rank = (world - 1) * n * 4  # inbound/outbound XGMI traffic per rank

        t_fly = _bench(_allgather, args.iters, group)

        def _nccl():
            dist.all_gather_into_tensor(ref, inp, group=group)

        t_nccl = _bench(_nccl, args.iters, group)

        if rank == 0:
            bw_fly = bytes_per_rank / (t_fly * 1e-6) / 1e9
            bw_nccl = bytes_per_rank / (t_nccl * 1e-6) / 1e9
            print(
                f"[perf] FlyDSL symm_at: {t_fly:8.2f} us  {bw_fly:7.1f} GB/s/rank   |   "
                f"NCCL: {t_nccl:8.2f} us  {bw_nccl:7.1f} GB/s/rank   |   "
                f"fly/nccl = {t_nccl/t_fly:.2f}x"
            )

    dist.barrier(group)
    out_st.destroy()
    dist.destroy_process_group()


# ---------------------------------------------------------------------------
# entry points
# ---------------------------------------------------------------------------
_WORLD = 8


def _argparser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-processes", type=int, default=_WORLD)
    ap.add_argument("--elems", type=int, default=1 << 20)  # int32 per rank (4 MB)
    ap.add_argument("--iters", type=int, default=50)
    return ap


@pytest.mark.skipif(not _HAVE_FLYDSL, reason="flydsl not importable (run inside dev_primus)")
@pytest.mark.skipif(torch.cuda.device_count() < _WORLD, reason=f"needs {_WORLD} GPUs")
def test_all_gather():
    args = _argparser().parse_args(["--elems", str(1 << 18), "--iters", "20"])
    torch.multiprocessing.spawn(_run, args=(_WORLD, args), nprocs=_WORLD)


def main():
    if not _HAVE_FLYDSL:
        raise SystemExit(f"flydsl not importable: {_IMPORT_ERR}")
    args = _argparser().parse_args()
    torch.multiprocessing.spawn(_run, args=(args.num_processes, args), nprocs=args.num_processes)


if __name__ == "__main__":
    main()
