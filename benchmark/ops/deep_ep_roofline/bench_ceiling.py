###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

# What the XGMI links and HBM move without DeepEP's protocol, as the ceiling for the FlyDSL DeepEP kernels.
# Launch with torchrun, one process per GPU (docs/flydsl_deepep_mi455_roofline.md).
#
# copy kernel (copy_kernel.py): the V2 dispatch's data path alone. NUM_THREADS per block, one warp per row, rows of
#   HIDDEN bf16 (one hidden row each), each loaded once and stored to every peer, into a symmetric heap
#   region per source rank. No slots, counts, flags or receiver copy.
#   tdm_uncached  Tensor Data Mover load to LDS and store with SCOPE_SYS into the uncached heap: what V2 does
#   vec_uncached  warp copy through VGPRs (buffer loads / stores, SCOPE_SYS), uncached heap: V2's non-TDM path
#   tdm_cached    as tdm_uncached into an ordinary (cached) hipMalloc heap
#   tdm_self      tdm_uncached into this rank's own heap: HBM, no XGMI
# references: hipmemcpy (hipMemcpyAsync D2D to the peer heap, one call per peer; the runtime picks the
#   engine), sdma (the same with hipMemcpyDeviceToDeviceNoCU: copy engines only), rccl_a2a
#   (all_to_all_single of equal splits, default RCCL settings), torch_copy (local copy_, HBM).
# direction: "bi" every rank sends to every peer at once (the dispatch / combine pattern; a GPU barrier,
#   untimed, starts the ranks together); "uni" rank 0 sends, the others idle.
# Reported bytes ("link_bytes"): what one sending rank moves to its peers (for tdm_self / torch_copy: the
#   bytes copied). GB/s = link_bytes / time.
#
# env: HIDDEN (7168), BLOCKS_LIST (8,16,32,48,64,80,96,128,160,192,224,256), ROWS_LIST (8192,32768; rows per
#      peer for the block sweep), SIZE_BLOCKS (80,256), SIZE_ROWS (256,1024,2048,4096,8192,16384,32768),
#      METHODS (all copy-kernel methods), REFS (1: the references too), COPY_WARPS (copy_kernel.py), CHECK (1),
#      ITERS, ROUNDS, FLUSH_MB, OUT_JSON, TAG
import os

import torch
import torch.distributed as dist
from common import RANK, WORLD, Results, bench, finish, init

group = init()

from copy_kernel import COPY_WARPS, NUM_THREADS, launch_copy  # noqa: E402

from primus_turbo.flydsl.deep_ep.launch_cache import run_launcher  # noqa: E402
from primus_turbo.pytorch.core.pyhip_runtime_wrapper import hipMemcpyKindEnum  # noqa: E402
from primus_turbo.pytorch.core.symm_mem import SymmetricMemory  # noqa: E402

H = int(os.environ.get("HIDDEN", 7168))
ROW_BYTES = H * 2
BLOCKS_LIST = [
    int(b) for b in os.environ.get("BLOCKS_LIST", "8,16,32,48,64,80,96,128,160,192,224,256").split(",")
]
ROWS_LIST = [int(r) for r in os.environ.get("ROWS_LIST", "8192,32768").split(",")]
SIZE_BLOCKS = [int(b) for b in os.environ.get("SIZE_BLOCKS", "80,256").split(",") if b]
SIZE_ROWS = [
    int(r) for r in os.environ.get("SIZE_ROWS", "256,1024,2048,4096,8192,16384,32768").split(",") if r
]
CHECK = os.environ.get("CHECK", "1") == "1"
REFS = os.environ.get("REFS", "1") == "1"
MAX_ROWS = max(ROWS_LIST + SIZE_ROWS)
REGION_BYTES = MAX_ROWS * ROW_BYTES
PEERS = WORLD - 1


def make_src(r, n):
    """Rows of rank r: a pattern every rank can rebuild, to check what landed."""
    words = n * ROW_BYTES // 4
    v = (torch.arange(words, dtype=torch.int64, device="cuda") * 2654435761 + (r + 1) * 97531) & 0x7FFFFFFF
    return v.to(torch.int32).view(torch.uint8).view(n, ROW_BYTES)


results = Results(
    dict(
        bench="ceiling",
        H=H,
        blocks_list=BLOCKS_LIST,
        rows_list=ROWS_LIST,
        size_blocks=SIZE_BLOCKS,
        size_rows=SIZE_ROWS,
        num_threads=NUM_THREADS,
        copy_warps=COPY_WARPS,
        tag=os.environ.get("TAG", ""),
    )
)
if RANK == 0:
    print(
        f"ceiling EP{WORLD} row {ROW_BYTES} B, blocks {BLOCKS_LIST}, rows {ROWS_LIST}, size sweep {SIZE_BLOCKS} x "
        f"{SIZE_ROWS}, {NUM_THREADS} threads per block",
        flush=True,
    )

heaps = {
    "uncached": SymmetricMemory(group, WORLD * REGION_BYTES, uncached_buffer=True),
    "cached": SymmetricMemory(group, WORLD * REGION_BYTES),
}
src = make_src(RANK, MAX_ROWS)
stream = torch.cuda.current_stream()
lib = heaps["uncached"].lib
METHODS = {
    "tdm_uncached": ("uncached", 1, 0),
    "vec_uncached": ("uncached", 0, 0),
    "tdm_cached": ("cached", 1, 0),
    "tdm_self": ("uncached", 1, 1),
}
if os.environ.get("METHODS"):
    METHODS = {m: METHODS[m] for m in os.environ["METHODS"].split(",")}


def copy_fn(method, blocks, rows, direction):
    heap, use_tdm, to_self = METHODS[method]
    symm = heaps[heap]

    def fn():
        if direction == "bi" or RANK == 0:
            run_launcher(
                launch_copy,
                src,
                symm.buffer_ptrs_dev,
                rows,
                RANK,
                WORLD,
                blocks,
                ROW_BYTES,
                REGION_BYTES,
                use_tdm,
                to_self,
                stream,
            )

    return fn


def check_landed(method, blocks, rows):
    """Every peer's rows arrived intact in this rank's heap (for tdm_self: our own rows)."""
    heap, _, to_self = METHODS[method]
    symm = heaps[heap]
    local = symm.get_buffer(RANK, (WORLD, MAX_ROWS, ROW_BYTES), torch.uint8)
    local.zero_()
    torch.cuda.synchronize()
    dist.barrier()
    copy_fn(method, blocks, rows, "bi")()
    torch.cuda.synchronize()
    dist.barrier()
    sources = [RANK] if to_self else [(RANK - j) % WORLD for j in range(1, WORLD)]
    ok = all(torch.equal(local[s, :rows], make_src(s, rows)) for s in sources)
    oks = [None] * WORLD
    dist.all_gather_object(oks, ok)
    return all(oks)


def barrier():
    heaps["cached"].barrier()


def add(method, direction, blocks, rows, stats):
    link = rows * ROW_BYTES * (1 if method in ("tdm_self", "torch_copy") else PEERS)
    us = stats["us"] if direction == "bi" else stats["rank_us"][0]
    results.add(
        method=method,
        direction=direction,
        blocks=blocks,
        rows=rows,
        link_bytes=link,
        gbps=link / us / 1e3,
        **stats,
    )


if CHECK:
    for method in METHODS:
        results.add(
            method=method,
            op="check",
            blocks=BLOCKS_LIST[-1],
            rows=MAX_ROWS,
            ok=check_landed(method, BLOCKS_LIST[-1], MAX_ROWS),
        )

# ---- block sweep, every method ----
for rows in ROWS_LIST:
    for method in METHODS:
        directions = ("bi",) if method == "tdm_self" else ("bi", "uni")
        for direction in directions:
            if RANK == 0:
                print(f"== {method} {direction} rows {rows}", flush=True)
            for blocks in BLOCKS_LIST:
                add(
                    method,
                    direction,
                    blocks,
                    rows,
                    bench(copy_fn(method, blocks, rows, direction), pre=barrier),
                )

# ---- size sweep: fixed cost and asymptote of the V2 data path ----
for blocks in SIZE_BLOCKS:
    if RANK == 0:
        print(f"== tdm_uncached bi size sweep, {blocks} blocks", flush=True)
    for rows in SIZE_ROWS:
        add(
            "tdm_uncached",
            "bi",
            blocks,
            rows,
            bench(copy_fn("tdm_uncached", blocks, rows, "bi"), pre=barrier),
        )

# ---- references ----
peer_ptrs = heaps["uncached"].buffer_ptrs


MEMCPY_KINDS = {
    "hipmemcpy": hipMemcpyKindEnum.hipMemcpyDeviceToDevice,
    "sdma": hipMemcpyKindEnum.hipMemcpyDeviceToDeviceNoCU,
}


# one stream per peer past EP2: on one stream the copies to the peers would run one after another
peer_streams = [torch.cuda.Stream() for _ in range(PEERS)] if PEERS > 1 else [stream]
fork = torch.cuda.Event()
joins = [torch.cuda.Event() for _ in peer_streams]


def hipmemcpy_fn(kind, rows, direction):
    nbytes = rows * ROW_BYTES

    def fn():
        if direction == "bi" or RANK == 0:
            fork.record(stream)
            for j, (peer_stream, join) in enumerate(zip(peer_streams, joins), start=1):
                peer_stream.wait_event(fork)
                dst = peer_ptrs[(RANK + j) % WORLD] + RANK * REGION_BYTES
                lib.hipMemcpyAsync(dst, src.data_ptr(), nbytes, MEMCPY_KINDS[kind], peer_stream.cuda_stream)
                join.record(peer_stream)
                stream.wait_event(join)

    return fn


for kind in MEMCPY_KINDS if REFS else ():
    for rows in sorted(set(ROWS_LIST + SIZE_ROWS)):
        for direction in ("bi", "uni"):
            add(kind, direction, 0, rows, bench(hipmemcpy_fn(kind, rows, direction), pre=barrier))

for rows in sorted(set(ROWS_LIST + SIZE_ROWS)) if REFS else ():
    inp = torch.empty(WORLD * rows, H, dtype=torch.bfloat16, device="cuda")
    out = torch.empty_like(inp)
    add("rccl_a2a", "bi", 0, rows, bench(lambda out=out, inp=inp: dist.all_to_all_single(out, inp)))
    del inp, out

dst_local = torch.empty_like(src)
for rows in ROWS_LIST if REFS else ():
    add("torch_copy", "bi", 0, rows, bench(lambda rows=rows: dst_local[:rows].copy_(src[:rows])))

finish(results)
