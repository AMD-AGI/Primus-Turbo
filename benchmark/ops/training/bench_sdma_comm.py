###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""SdmaCommunicator vs RCCL: all-gather / all-to-all / reduce-scatter latency and per-GPU egress.

Egress = bytes this rank sends to its peers / time, the figure the MI455X design doc compares
against the ~1.4 TB/s an 8-GPU all-to-all reaches. ``--no-rccl`` runs on a gloo group only.

    torchrun --nproc_per_node 8 benchmark/ops/training/bench_sdma_comm.py
    torchrun --nnodes 2 --nproc_per_node 4 ... benchmark/ops/training/bench_sdma_comm.py --mib 1 4 16 64
"""

import argparse
import os
from functools import partial

import torch
import torch.distributed as dist

from primus_turbo.pytorch.dist.sdma_comm import SdmaCommunicator


def _time_ms(fn, iters: int, warmup: int = 3) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mib", type=float, nargs="+", default=[0.25, 1, 4, 16, 64], help="MiB per peer")
    parser.add_argument("--chunks", type=int, nargs="+", default=[1, 2, 4], help="puts per peer")
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--no-rccl", action="store_true")
    args = parser.parse_args()

    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    dist.init_process_group("gloo" if args.no_rccl else "cpu:gloo,cuda:nccl")
    group = dist.group.WORLD
    rank, n = dist.get_rank(), dist.get_world_size()
    dtype = torch.bfloat16

    comm = SdmaCommunicator(group, int(max(args.mib) * 2**20), max_chunks=max(args.chunks))
    if rank == 0:
        print(f"world={n} engines/GPU={len(comm.engines)} backend={comm.symm.backend}")
        print(
            f"{'op':<15}{'MiB/peer':>9}{'chunks':>7}{'sdma us':>10}{'egress GB/s':>13}"
            f"{'rccl us':>10}{'egress GB/s':>13}"
        )
    for mib in args.mib:
        numel = int(mib * 2**20) // dtype.itemsize
        egress = (n - 1) * numel * dtype.itemsize
        ag_in = torch.randn(numel, device="cuda", dtype=dtype)
        a2a_in = torch.randn(n, numel, device="cuda", dtype=dtype)
        ag_out = torch.empty(n, numel, device="cuda", dtype=dtype)
        a2a_out = torch.empty(n, numel, device="cuda", dtype=dtype)
        rs_out = torch.empty(numel, device="cuda", dtype=dtype)
        cases = [
            ("all_gather", comm.all_gather, ag_in, partial(dist.all_gather_into_tensor, ag_out, ag_in)),
            ("all_to_all", comm.all_to_all, a2a_in, partial(dist.all_to_all_single, a2a_out, a2a_in)),
            (
                "reduce_scatter",
                comm.reduce_scatter,
                a2a_in,
                partial(dist.reduce_scatter_tensor, rs_out, a2a_in),
            ),
        ]
        for name, sdma_op, sdma_in, rccl_fn in cases:
            dist.barrier()
            t_rccl = float("nan") if args.no_rccl else _time_ms(rccl_fn, args.iters)
            for chunks in args.chunks:
                dist.barrier()
                t_sdma = _time_ms(partial(sdma_op, sdma_in, num_chunks=chunks), args.iters)
                if rank == 0:
                    print(
                        f"{name:<15}{mib:>9.2f}{chunks:>7}{t_sdma * 1e3:>10.1f}{egress / t_sdma / 1e6:>13.1f}"
                        f"{t_rccl * 1e3:>10.1f}{egress / t_rccl / 1e6:>13.1f}"
                    )
    comm.destroy()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
