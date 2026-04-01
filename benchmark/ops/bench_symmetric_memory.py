###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Symmetric-memory intranode allgather benchmark.

This script compares:
1) primus_turbo.pytorch.cco.symm_mem.SymmetricMemory
2) torch.distributed._symmetric_memory

Both backends run the same Triton allgather kernel for fair comparison.
"""

import argparse
import csv
import socket
from dataclasses import dataclass
from typing import Callable

import torch
import torch.distributed as dist
import triton
import triton.language as tl

from primus_turbo.pytorch.cco.pyhip_runtime_wrapper import HIPRuntimeLibrary, hipMemcpyKindEnum
from primus_turbo.pytorch.cco.symm_mem import SymmetricMemory as TurboSymmetricMemory

try:
    import torch.distributed._symmetric_memory as torch_symm_mem
except Exception:
    torch_symm_mem = None


def _pick_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return int(s.getsockname()[1])


def _ptr_to_int(ptr) -> int:
    if ptr is None:
        return 0
    if hasattr(ptr, "value"):
        return 0 if ptr.value is None else int(ptr.value)
    return int(ptr)


@triton.jit
def _intranode_allgather_kernel(
    src_ptr,
    dst_ptrs_ptr,
    numel,
    rank,
    BLOCK: tl.constexpr,
):
    dst_rank = tl.program_id(axis=0)
    pid = tl.program_id(axis=1)

    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    vals = tl.load(src_ptr + offs, mask=mask, other=0.0)

    dst_base = tl.load(dst_ptrs_ptr + dst_rank).to(tl.pointer_type(tl.float32))
    dst_ptr = dst_base + rank * numel + offs
    tl.store(dst_ptr, vals, mask=mask)


@dataclass
class BackendResources:
    name: str
    ptrs_dev: torch.Tensor
    local_ptr: int
    destroy: Callable[[], None]


def _get_group_name(group: dist.ProcessGroup) -> str:
    group_name = getattr(group, "group_name", None)
    if callable(group_name):
        group_name = group_name()
    if not isinstance(group_name, str):
        raise RuntimeError("Cannot resolve ProcessGroup.group_name for torch symmetric memory.")
    return group_name


def _build_turbo_backend(
    group: dist.ProcessGroup, workspace_bytes: int, signal_pad_size: int
) -> BackendResources:
    symm = TurboSymmetricMemory(group, workspace_bytes, signal_pad_size=signal_pad_size)
    local_ptr = _ptr_to_int(symm.buffer_ptrs[symm.rank])
    ptrs_dev = symm.buffer_ptrs_dev.contiguous()

    return BackendResources(
        name="symm_mem.py",
        ptrs_dev=ptrs_dev,
        local_ptr=local_ptr,
        destroy=symm.destroy,
    )


def _build_torch_backend(group: dist.ProcessGroup, workspace_bytes: int) -> BackendResources:
    if torch_symm_mem is None:
        raise RuntimeError("torch.distributed._symmetric_memory is not available in current torch.")

    group_name = _get_group_name(group)
    symm = torch_symm_mem.get_symm_mem_workspace(group_name, workspace_bytes)
    ptrs = []
    for rank in range(group.size()):
        rank_buf = symm.get_buffer(rank, [workspace_bytes], torch.uint8)
        ptrs.append(int(rank_buf.data_ptr()))
    ptrs_dev = torch.tensor(ptrs, dtype=torch.int64, device="cuda")

    return BackendResources(
        name="torch._symmetric_memory",
        ptrs_dev=ptrs_dev,
        local_ptr=ptrs[group.rank()],
        destroy=lambda: None,
    )


def _copy_local_allgather_result(
    lib: HIPRuntimeLibrary, local_ptr: int, world_size: int, numel: int
) -> torch.Tensor:
    recv = torch.empty((world_size, numel), dtype=torch.float32, device="cuda")
    slot_nbytes = numel * recv.element_size()
    stream_ptr = torch.cuda.current_stream().cuda_stream

    for src_rank in range(world_size):
        src_ptr = local_ptr + src_rank * slot_nbytes
        dst_ptr = int(recv[src_rank].data_ptr())
        lib.hipMemcpyAsync(
            dst_ptr,
            src_ptr,
            slot_nbytes,
            hipMemcpyKindEnum.hipMemcpyDeviceToDevice,
            stream_ptr,
        )
    torch.cuda.synchronize()
    return recv


def _run_case(
    backend: BackendResources,
    lib: HIPRuntimeLibrary,
    rank: int,
    world_size: int,
    workspace_bytes: int,
    numel: int,
    warmup: int,
    iters: int,
    block_size: int,
    do_check: bool,
    group: dist.ProcessGroup,
) -> tuple[float, float, bool]:
    src = torch.full((numel,), float(rank + 1), dtype=torch.float32, device="cuda")
    grid = (world_size, triton.cdiv(numel, block_size))

    # Clear local workspace; each rank owns and clears its local segment.
    lib.hipMemset(backend.local_ptr, 0, workspace_bytes)
    torch.cuda.synchronize()
    dist.barrier(group=group)

    for _ in range(warmup):
        _intranode_allgather_kernel[grid](
            src,
            backend.ptrs_dev,
            numel,
            rank,
            BLOCK=block_size,
        )
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        _intranode_allgather_kernel[grid](
            src,
            backend.ptrs_dev,
            numel,
            rank,
            BLOCK=block_size,
        )
    end.record()
    end.synchronize()

    avg_ms = start.elapsed_time(end) / iters
    per_rank_bytes = world_size * numel * src.element_size()
    bw_gbps = per_rank_bytes / (avg_ms * 1e-3) / 1e9

    check_ok = True
    if do_check:
        dist.barrier(group=group)
        recv = _copy_local_allgather_result(lib, backend.local_ptr, world_size, numel)
        expected = (
            torch.arange(1, world_size + 1, dtype=torch.float32, device="cuda")
            .view(world_size, 1)
            .expand(world_size, numel)
        )
        check_ok = torch.allclose(recv, expected, rtol=0.0, atol=0.0)
    dist.barrier(group=group)
    return avg_ms, bw_gbps, check_ok


def _worker(rank: int, world_size: int, args: argparse.Namespace):
    torch.cuda.set_device(rank)
    dist.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size,
        init_method=f"tcp://127.0.0.1:{args.master_port}",
    )
    group = dist.new_group(list(range(world_size)))
    lib = HIPRuntimeLibrary()

    numel_list = [int(x.strip()) for x in args.numel_list.split(",") if x.strip()]
    max_numel = max(numel_list)
    workspace_bytes = world_size * max_numel * torch.tensor([], dtype=torch.float32).element_size()

    backends = [_build_turbo_backend(group, workspace_bytes, args.signal_pad_size)]
    if args.skip_torch_symm:
        if rank == 0:
            print("Skip torch symmetric memory benchmark (--skip-torch-symm enabled).")
    else:
        backends.append(_build_torch_backend(group, workspace_bytes))

    all_rows = []
    try:
        elem_size = torch.tensor([], dtype=torch.float32).element_size()
        for numel in numel_list:
            nbytes = numel * elem_size
            if rank == 0:
                print(f"\n=== nbytes={nbytes} ===")
            per_case = {}
            for backend in backends:
                avg_ms, bw_gbps, check_ok = _run_case(
                    backend=backend,
                    lib=lib,
                    rank=rank,
                    world_size=world_size,
                    workspace_bytes=workspace_bytes,
                    numel=numel,
                    warmup=args.warmup,
                    iters=args.iters,
                    block_size=args.block_size,
                    do_check=args.check,
                    group=group,
                )

                gathered = [None] * world_size
                dist.all_gather_object(
                    gathered,
                    {
                        "rank": rank,
                        "avg_ms": avg_ms,
                        "bw_gbps": bw_gbps,
                        "check_ok": check_ok,
                    },
                    group=group,
                )

                max_ms = max(item["avg_ms"] for item in gathered)
                min_ms = min(item["avg_ms"] for item in gathered)
                mean_ms = sum(item["avg_ms"] for item in gathered) / world_size
                min_bw = min(item["bw_gbps"] for item in gathered)
                check_pass = all(item["check_ok"] for item in gathered)

                row = {
                    "nbytes": nbytes,
                    "backend": backend.name,
                    "latency_ms(max_rank)": max_ms,
                    "latency_ms(mean_rank)": mean_ms,
                    "latency_ms(min_rank)": min_ms,
                    "per_rank_write_bw_gbps(min_rank)": min_bw,
                    "check": "PASS" if check_pass else "FAIL",
                }
                all_rows.append(row)
                per_case[backend.name] = row

                if rank == 0:
                    print(
                        f"[{backend.name}] "
                        f"lat(max/mean/min)={max_ms:.4f}/{mean_ms:.4f}/{min_ms:.4f} ms, "
                        f"bw(min-rank)={min_bw:.2f} GB/s, check={row['check']}"
                    )

            if rank == 0 and "symm_mem.py" in per_case and "torch._symmetric_memory" in per_case:
                t0 = per_case["symm_mem.py"]["latency_ms(max_rank)"]
                t1 = per_case["torch._symmetric_memory"]["latency_ms(max_rank)"]
                speedup = t1 / t0 if t0 > 0 else float("inf")
                print(f"speedup(symm_mem.py vs torch._symmetric_memory): {speedup:.3f}x")

        if rank == 0:
            print("\n=== Summary ===")
            for row in all_rows:
                print(
                    f"nbytes={row['nbytes']:>10}, "
                    f"backend={row['backend']:<24}, "
                    f"lat(max)={row['latency_ms(max_rank)']:.4f} ms, "
                    f"bw(min)={row['per_rank_write_bw_gbps(min_rank)']:.2f} GB/s, "
                    f"check={row['check']}"
                )

            if args.output is not None:
                with open(args.output, "w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
                    writer.writeheader()
                    writer.writerows(all_rows)
                print(f"\nSaved results to {args.output}")
    finally:
        for backend in backends:
            backend.destroy()
        dist.barrier(group=group)
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark intranode allgather on two symmetric-memory backends."
    )
    parser.add_argument(
        "--num-processes",
        type=int,
        default=min(8, torch.cuda.device_count()),
        help="Number of ranks to spawn (default: min(8, local_gpu_count)).",
    )
    parser.add_argument(
        "--numel-list",
        type=str,
        default="262144,524288,1048576",
        help="Comma-separated allgather element counts for float32 tensor.",
    )
    parser.add_argument("--warmup", type=int, default=30, help="Warmup iterations per case.")
    parser.add_argument("--iters", type=int, default=100, help="Measured iterations per case.")
    parser.add_argument("--block-size", type=int, default=256, help="Triton BLOCK size.")
    parser.add_argument(
        "--signal-pad-size",
        type=int,
        default=1024,
        help="Signal pad size used by symm_mem.py backend.",
    )
    parser.add_argument(
        "--skip-torch-symm",
        action="store_true",
        help="Only benchmark symm_mem.py backend.",
    )
    parser.add_argument(
        "--no-check",
        action="store_true",
        help="Disable correctness check.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Optional CSV path for benchmark summary.",
    )
    args = parser.parse_args()
    args.check = not args.no_check

    if args.num_processes <= 0:
        raise ValueError("--num-processes must be > 0")
    if args.num_processes > torch.cuda.device_count():
        raise ValueError(
            f"--num-processes={args.num_processes} exceeds visible GPU count={torch.cuda.device_count()}"
        )

    args.master_port = _pick_free_port()
    torch.multiprocessing.spawn(
        _worker,
        args=(args.num_processes, args),
        nprocs=args.num_processes,
        join=True,
    )


if __name__ == "__main__":
    main()
