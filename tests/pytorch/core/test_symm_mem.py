###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""SymmetricMemory over its heap backends (legacy HIP IPC, fabric handles).

Every rank writes a rank-tagged pattern into its slot of every peer's buffer and signal pad, then
checks what landed in its own; the signal-pad GPU barrier exercises system-scope atomics on peer
memory. A group spanning hosts (the trays of an MI455X UALink pod) runs the fabric backend only;
``auto`` must pick it there and ``ipc`` must refuse on every rank.

The pytest entry launches this file under torchrun on the local GPUs:

    pytest tests/pytorch/core/test_symm_mem.py
    torchrun --nproc_per_node 8 tests/pytorch/core/test_symm_mem.py
    torchrun --nnodes 2 --nproc_per_node 4 ... tests/pytorch/core/test_symm_mem.py   # across hosts
"""

import os
import subprocess
import sys
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist

N = 1 << 16


def _check_heap(symm, backend, uncached, failures):
    rank, world = symm.rank, symm.world_size
    tag = f"[rank {rank} {backend} uncached={uncached}]"
    if symm.backend != backend:
        failures.append(f"{tag} backend {symm.backend}")
    if len(symm.buffer_ptrs) != world or 0 in symm.buffer_ptrs or len(set(symm.buffer_ptrs)) != world:
        failures.append(f"{tag} bad buffer ptrs {symm.buffer_ptrs}")
    local = symm.get_buffer(rank, (world, N), torch.int32)
    if local.count_nonzero().item() != 0:
        failures.append(f"{tag} heap not zeroed")
    dist.barrier(group=symm.group)
    pad_words = symm.signal_pad_size // 4
    for peer in range(world):
        symm.get_buffer(peer, (world, N), torch.int32)[rank].copy_(
            torch.arange(N, device="cuda", dtype=torch.int32) + (rank * world + peer) * N
        )
        symm.get_signal_pad(peer, (pad_words,), torch.int32)[world + rank] = rank * world + peer + 1
    torch.cuda.synchronize()
    dist.barrier(group=symm.group)
    for src in range(world):
        want = torch.arange(N, device="cuda", dtype=torch.int32) + (src * world + rank) * N
        if not torch.equal(local[src], want):
            failures.append(f"{tag} slot of rank {src}: {(local[src] != want).sum().item()} words wrong")
    got = symm.get_signal_pad(rank, (pad_words,), torch.int32)[world : 2 * world].tolist()
    if got != [src * world + rank + 1 for src in range(world)]:
        failures.append(f"{tag} signal pad {got}")
    for _ in range(4):
        symm.barrier()
    torch.cuda.synchronize()
    if symm.get_signal_pad(rank, (pad_words,), torch.int32)[:world].count_nonzero().item() != 0:
        failures.append(f"{tag} barrier left its counters set")


def run(group):
    from primus_turbo.pytorch.core.symm_mem import (
        SymmetricMemory,
        _os_instance_id,
        resolve_symm_mem_backend,
    )

    ids = [None] * group.size()
    dist.all_gather_object(ids, _os_instance_id(), group=group)
    multi_host = len(set(ids)) > 1
    failures = []

    want_auto = "fabric" if multi_host else "ipc"
    if resolve_symm_mem_backend(group, "auto") != want_auto:
        failures.append(f"[rank {group.rank()}] auto did not resolve to {want_auto}")
    if multi_host:
        try:
            resolve_symm_mem_backend(group, "ipc")
            failures.append(f"[rank {group.rank()}] ipc accepted a group spanning hosts")
        except RuntimeError:
            pass

    backends = ["fabric"] if multi_host else ["ipc", "fabric"]
    for backend in backends:
        for uncached in (False, True):
            for _ in range(2):
                symm = SymmetricMemory(
                    group,
                    group.size() * N * 4,
                    signal_pad_size=1024,
                    uncached_buffer=uncached,
                    uncached_signal_pad=uncached,
                    backend=backend,
                )
                _check_heap(symm, backend, uncached, failures)
                torch.cuda.synchronize()
                dist.barrier(group=group)
                symm.destroy()

    gathered = [None] * group.size()
    dist.all_gather_object(gathered, failures, group=group)
    return [f for per_rank in gathered for f in per_rank], backends, multi_host


def main():
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    # handle exchange and barriers only: gloo keeps the test independent of RCCL
    dist.init_process_group("gloo", timeout=timedelta(minutes=5))
    group = dist.group.WORLD
    failures, backends, multi_host = run(group)
    if dist.get_rank() == 0:
        where = "across hosts" if multi_host else "on one host"
        print(f"world={group.size()} {where} backends={backends}: " + ("PASS" if not failures else "FAIL"))
        for f in failures:
            print("  " + f)
    dist.barrier()
    dist.destroy_process_group()
    sys.exit(1 if failures else 0)


@pytest.mark.parametrize("world_size", [2, 4, 8])
def test_symm_mem(world_size):
    if torch.cuda.device_count() < world_size:
        pytest.skip(f"needs {world_size} GPUs")
    cmd = [sys.executable, "-m", "torch.distributed.run", f"--nproc_per_node={world_size}", __file__]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-4000:]


if __name__ == "__main__":
    main()
