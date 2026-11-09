###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""All-to-all over XGMI from kernel loads/stores: push (remote store) vs pull (remote load).

Every rank moves ``--mib-per-peer`` to (push) or from (pull) each peer at once, using
``--cus`` workgroups split evenly over the peers. Sizes the CU budget of the Ulysses
GemmA2A / A2AGemm communication blocks and picks push or pull.

    torchrun --nproc_per_node 8 benchmark/ops/training/bench_xgmi_read_write.py
"""

import argparse
import os
import statistics

import torch
import torch.distributed as dist
import triton
import triton.language as tl

from primus_turbo.pytorch.core.symm_mem import SymmetricMemory


@triton.jit
def _a2a_copy_kernel(
    ptrs,
    rank,
    num_ranks: tl.constexpr,
    peer_numel,
    progs_per_peer,
    PULL: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    step = pid // progs_per_peer + 1
    peer = (rank + step) % num_ranks
    chunk = pid % progs_per_peer
    local = tl.load(ptrs + rank).to(tl.pointer_type(tl.bfloat16))
    remote = tl.load(ptrs + peer).to(tl.pointer_type(tl.bfloat16))
    # buffer = [send region: n slots][recv region: n slots], slot = peer_numel
    if PULL:
        src = remote + rank * peer_numel
        dst = local + num_ranks * peer_numel + peer * peer_numel
    else:
        src = local + peer * peer_numel
        dst = remote + num_ranks * peer_numel + rank * peer_numel
    offs = tl.arange(0, BLOCK)
    for start in range(chunk * BLOCK, peer_numel, progs_per_peer * BLOCK):
        x = tl.load(src + start + offs, cache_modifier=".cg")
        tl.store(dst + start + offs, x, cache_modifier=".cs")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mib-per-peer", type=float, nargs="+", default=[4.0, 6.0, 24.0])
    parser.add_argument("--cus", type=int, nargs="+", default=[7, 14, 28, 56, 112])
    parser.add_argument("--iters", type=int, default=20)
    args = parser.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    group = dist.group.WORLD
    rank, n = dist.get_rank(), dist.get_world_size()

    max_numel = int(max(args.mib_per_peer) * 2**20 // 2)
    symm = SymmetricMemory(group, 2 * n * max_numel * 2)
    ptrs = torch.tensor(symm.buffer_ptrs, dtype=torch.int64, device="cuda")
    block = 4096

    rows = []
    for mib in args.mib_per_peer:
        peer_numel = int(mib * 2**20 // 2) // block * block
        for cus in args.cus:
            progs_per_peer = max(1, cus // (n - 1))
            grid = (progs_per_peer * (n - 1),)
            result = {}
            for pull in (False, True):
                times = []
                for it in range(args.iters + 3):
                    torch.cuda.synchronize()
                    dist.barrier()
                    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    start.record()
                    _a2a_copy_kernel[grid](
                        ptrs, rank, n, peer_numel, progs_per_peer, PULL=pull, BLOCK=block, num_warps=8
                    )
                    end.record()
                    torch.cuda.synchronize()
                    if it >= 3:
                        times.append(start.elapsed_time(end) * 1e3)
                t = torch.tensor([statistics.median(times)], device="cuda", dtype=torch.float64)
                dist.all_reduce(t, op=dist.ReduceOp.MAX)
                result["pull" if pull else "push"] = t.item()
            rows.append((mib, grid[0], result["push"], result["pull"], peer_numel * 2 * (n - 1)))

    if rank == 0:
        print(
            f"ranks={n}  (max over ranks of per-rank median, us; GB/s = bytes leaving or entering one rank)"
        )
        print(f"{'MiB/peer':>9}{'CUs':>6}{'push us':>10}{'push GB/s':>11}{'pull us':>10}{'pull GB/s':>11}")
        for mib, cus, push, pull, nbytes in rows:
            print(
                f"{mib:>9.1f}{cus:>6}{push:>10.1f}{nbytes / push / 1e3:>11.0f}"
                f"{pull:>10.1f}{nbytes / pull / 1e3:>11.0f}"
            )
    dist.barrier()
    symm.destroy()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
