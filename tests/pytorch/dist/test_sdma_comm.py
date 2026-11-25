###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""SdmaCommunicator: all-gather / all-to-all / reduce-scatter on SDMA engines.

Every rank derives every rank's input from (rank, iteration), so each checks its results bitwise
against a reference built locally; the process group is gloo-only, which keeps the test independent
of RCCL. Calls run back to back without host syncs, cycling both banks and every signal slot.

    pytest tests/pytorch/dist/test_sdma_comm.py
    torchrun --nproc_per_node 8 tests/pytorch/dist/test_sdma_comm.py
    torchrun --nnodes 2 --nproc_per_node 4 ... tests/pytorch/dist/test_sdma_comm.py   # across trays
"""

import os
import subprocess
import sys
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist

ITERS = 5
# elements per rank (per destination for all-to-all / reduce-scatter)
NUMELS = [1, 1000, 1 << 18, 9 << 19]
DTYPES = [torch.bfloat16, torch.float32, torch.uint8]


def _data(seed: int, shape, dtype: torch.dtype) -> torch.Tensor:
    g = torch.Generator(device="cuda").manual_seed(seed)
    if dtype.is_floating_point:
        return torch.randn(shape, generator=g, device="cuda", dtype=torch.float32).to(dtype)
    return torch.randint(0, 256, shape, generator=g, device="cuda", dtype=torch.int32).to(dtype)


def _seed(src: int, it: int, case: int) -> int:
    return (case * 1000 + it) * 1000 + src


def _reduce(rows: torch.Tensor) -> torch.Tensor:
    if rows.dtype in (torch.bfloat16, torch.float16):
        return rows.sum(0, dtype=torch.float32).to(rows.dtype)
    return rows.sum(0)


def _reversed_gather(comm, inp: torch.Tensor) -> torch.Tensor:
    """All-gather into row ``world - 1 - src`` through the raw primitive: two puts per peer behind
    two release points, tail (flag 1) before head (flag 0), the head produced on the stream after
    the first release; per-peer ``wait_flag`` before the final ``wait``. The head is empty for a
    1-byte input."""
    rank, world = comm.rank, comm.world_size
    nbytes = inp.numel() * inp.element_size()
    head = nbytes // 2
    staged = torch.zeros_like(inp)
    comm.begin()
    view = comm.bank((world, *inp.shape), inp.dtype)
    dst = (world - 1 - rank) * nbytes
    view[world - 1 - rank].copy_(inp)
    peers = [p for p in range(world) if p != rank]
    for peer in peers:
        comm.put(peer, dst + head, inp.data_ptr() + head, nbytes - head, flag=1)
    comm.release()
    staged.copy_(inp)
    for peer in peers:
        comm.put(peer, dst, staged.data_ptr(), head, flag=0)
    for peer in reversed(peers):
        comm.wait_flag(peer, 1)
    comm.wait(2)
    return view


def run(group, failures):
    from primus_turbo.pytorch.dist.sdma_comm import SdmaCommunicator

    rank, world = group.rank(), group.size()
    max_bytes = max(NUMELS) * max(d.itemsize for d in DTYPES)
    comm = SdmaCommunicator(group, max_bytes)
    case = 0
    for dtype in DTYPES:
        for numel in NUMELS:
            case += 1
            tag = f"[rank {rank} {dtype} numel={numel}]"
            got_ag, got_a2a, got_rs, got_put = [], [], [], []
            for it in range(ITERS):
                chunks = 1 + it % comm.max_chunks
                ag_in = _data(_seed(rank, it, case), (numel,), dtype)
                a2a_in = _data(_seed(rank, it, case + 500), (world, numel), dtype)
                # a view of the bank is valid until the next-but-one call: copy it out on the stream
                got_ag.append(comm.all_gather(ag_in, num_chunks=chunks).clone())
                if it % 2:
                    out = torch.empty((world, numel), dtype=dtype, device="cuda")
                    got_a2a.append(comm.all_to_all(a2a_in, out=out, num_chunks=chunks))
                else:
                    got_a2a.append(comm.all_to_all(a2a_in, num_chunks=chunks).clone())
                if dtype.is_floating_point:
                    got_rs.append(comm.reduce_scatter(a2a_in, num_chunks=chunks))
                got_put.append(_reversed_gather(comm, ag_in).clone())
            torch.cuda.synchronize()
            for it in range(ITERS):
                want_ag = torch.stack([_data(_seed(src, it, case), (numel,), dtype) for src in range(world)])
                if not torch.equal(got_ag[it], want_ag):
                    failures.append(
                        f"{tag} all_gather iter {it}: {(got_ag[it] != want_ag).sum().item()} wrong"
                    )
                want_a2a = torch.stack(
                    [_data(_seed(src, it, case + 500), (world, numel), dtype)[rank] for src in range(world)]
                )
                if not torch.equal(got_a2a[it], want_a2a):
                    bad = (got_a2a[it] != want_a2a).sum().item()
                    failures.append(f"{tag} all_to_all iter {it}: {bad} wrong")
                if dtype.is_floating_point and not torch.equal(got_rs[it], _reduce(want_a2a)):
                    failures.append(f"{tag} reduce_scatter iter {it}: not bitwise equal")
                if not torch.equal(got_put[it], want_ag.flip(0)):
                    failures.append(
                        f"{tag} put iter {it}: {(got_put[it] != want_ag.flip(0)).sum().item()} wrong"
                    )
    comm.destroy()


def main():
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    dist.init_process_group("gloo", timeout=timedelta(minutes=5))
    group = dist.group.WORLD
    failures = []
    run(group, failures)
    gathered = [None] * group.size()
    dist.all_gather_object(gathered, failures, group=group)
    failures = [f for per_rank in gathered for f in per_rank]
    if dist.get_rank() == 0:
        print(f"world={group.size()}: " + ("PASS" if not failures else "FAIL"))
        for f in failures[:50]:
            print("  " + f)
    dist.barrier()
    dist.destroy_process_group()
    sys.exit(1 if failures else 0)


@pytest.mark.parametrize("world_size", [2, 4, 8])
def test_sdma_comm(world_size):
    if torch.cuda.device_count() < world_size:
        pytest.skip(f"needs {world_size} GPUs")
    cmd = [sys.executable, "-m", "torch.distributed.run", f"--nproc_per_node={world_size}", __file__]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-4000:]


if __name__ == "__main__":
    main()
