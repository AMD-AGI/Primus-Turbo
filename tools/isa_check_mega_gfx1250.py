###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Compile-only ISA check of the arch-sensitive primitives the BF16 Mega MoE EP path relies on.

Nothing is launched and no GPU is touched: FlyDSL runs with ``COMPILE_ONLY=1`` against the arch
given on the command line, and the final ISA of two probe kernels is dumped and inspected.

* ``mega_prims_probe`` -- ``read_clock``, ``wait_mem_all``, device / system cache policies on
  buffer loads and stores, system-scope flag ``ld`` / ``st`` / ``atomic_add`` through a peer
  address mapped by ``SymBuffer``.
* ``mega_ep_tiles_probe`` -- the real ``dispatch_bf16_tile``, ``combine_bf16_tile`` and
  ``topk_reduce_bf16_tile`` bodies, each behind its own block-index branch.

Usage (inside a container that has FlyDSL and the ROCm device libs)::

    python tools/isa_check_mega_gfx1250.py --arch gfx1250 --out /tmp/mega_isa
    python tools/isa_check_mega_gfx1250.py --arch gfx950 --out /tmp/mega_isa_gfx950

On gfx1250 the script fails if a gfx9-only instruction or cache-policy spelling survives, or if a
required gfx12 form is missing. On gfx950 it only prints the summary, which must keep the
``sc1 nt`` / ``sc0 sc1 nt`` policies and ``s_memrealtime`` the EP path always had there.
"""

import argparse
import collections
import os
import re
import sys
from pathlib import Path


def _parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--arch", default="gfx1250", choices=("gfx1250", "gfx950", "gfx942"))
    parser.add_argument("--out", default="/tmp/mega_isa_check", help="FLYDSL_DUMP_DIR for the probe kernels")
    return parser.parse_args()


ARGS = _parse_args()
# Set before FlyDSL is imported: the arch is read while the kernels are traced.
os.environ["COMPILE_ONLY"] = "1"
os.environ["FLYDSL_GPU_ARCH"] = ARGS.arch
os.environ["FLYDSL_COMPILE_ARCH"] = ARGS.arch
os.environ["FLYDSL_DUMP_IR"] = "1"
os.environ["FLYDSL_DUMP_DIR"] = ARGS.out
os.environ["FLYDSL_RUNTIME_ENABLE_CACHE"] = "0"

import flydsl.compiler as flyc  # noqa: E402
import flydsl.expr as fx  # noqa: E402

from primus_turbo.flydsl.mega.bf16.ep_intranode import (  # noqa: E402
    combine_bf16_tile,
    dispatch_bf16_tile,
    topk_reduce_bf16_tile,
)
from primus_turbo.flydsl.mega.bf16.symm_buffer import SymBuffer, Workspace  # noqa: E402
from primus_turbo.flydsl.utils.arch import (  # noqa: E402
    cpol_device_stream,
    cpol_system_stream,
)
from primus_turbo.flydsl.utils.buffer_ops import (  # noqa: E402
    buffer_load,
    buffer_store,
    create_buffer_resource_from_addr,
)
from primus_turbo.flydsl.utils.prims import (  # noqa: E402
    atomic_add,
    cast,
    ld,
    read_clock,
    st,
    wait_mem_all,
)

_BLOCK = 512
# A DeepSeek-V3-like EP8 layout; the values only size the probe, nothing is allocated.
_NUM_RANKS = 8
_NUM_EXPERTS = 256
_NUM_MAX_TOKENS_PER_RANK = 256
_NUM_TOPK = 8
_HIDDEN = 7168
# Fake heap bases: SymBuffer only turns them into a device table at launch, which never happens.
_FAKE_HEAPS = [(i + 1) << 40 for i in range(_NUM_RANKS)]


@flyc.kernel(known_block_size=[_BLOCK, 1, 1])
def mega_prims_probe(sym: SymBuffer, dst_rank: fx.Int32):
    lane_word = fx.thread_idx.x * fx.Int32(4)
    base = sym.get_base_ptr()
    peer = sym.map(base, dst_rank)
    local_res = create_buffer_resource_from_addr(base, num_records_bytes=_BLOCK * 16)
    peer_res = create_buffer_resource_from_addr(peer, num_records_bytes=_BLOCK * 16)
    start = read_clock()
    device_cpol, system_cpol = cpol_device_stream(), cpol_system_stream()
    payload = buffer_load(local_res, lane_word, vec_width=4, dtype=fx.T.i32, cache_modifier=device_cpol)
    buffer_store(payload, peer_res, lane_word, cache_modifier=system_cpol)
    wait_mem_all()
    st(peer, fx.Int32(_BLOCK * 4), fx.Int64(1), scope="sys")
    atomic_add(peer, fx.Int32(_BLOCK * 4 + 1), fx.Int64(1), scope="sys")
    seen = ld(base, fx.Int32(_BLOCK * 4), scope="sys", dtype=fx.T.i64)
    echoed = buffer_load(peer_res, lane_word, vec_width=4, dtype=fx.T.i32, cache_modifier=system_cpol)
    elapsed = cast(read_clock() - start + seen, fx.T.i32)
    buffer_store(echoed, local_res, lane_word)
    buffer_store(elapsed, local_res, fx.Int32(_BLOCK * 4) + fx.thread_idx.x)


@flyc.kernel(known_block_size=[_BLOCK, 1, 1])
def mega_ep_tiles_probe(sym: SymBuffer, rt: fx.Int32):
    thread_index = fx.thread_idx.x
    block_index, _b, _c = fx.block_idx
    # hoist every workspace / resource before the dynamic branches (rewriter can't carry Workspace)
    workspace = Workspace(
        sym.get_base_ptr(), _NUM_RANKS, _NUM_EXPERTS, _NUM_MAX_TOKENS_PER_RANK, _NUM_TOPK, _HIDDEN
    )
    base = sym.get_base_ptr()
    table = base + fx.Int64(1 << 32)

    def res(slot):
        return create_buffer_resource_from_addr(table + fx.Int64(slot << 24), num_records_bytes=1 << 24)

    reduce_flag_base = workspace.get_reduce_flag_ptr()
    combine_token_res = create_buffer_resource_from_addr(
        workspace.get_combine_token_buffer_ptr(),
        num_records_bytes=int(workspace.num_max_routes) * _HIDDEN * 2,
    )
    epoch = cast(rt, fx.T.i64)
    if block_index < fx.Int32(_NUM_RANKS):
        dispatch_bf16_tile(
            sym,
            workspace,
            thread_index=thread_index,
            block_index=block_index,
            hidden_size=_HIDDEN,
            input_res=res(0),
            expert_send_dst_rank_res=res(1),
            expert_send_count_res=res(2),
            expert_send_offset_res=res(3),
            dispatched_token_idx_res=res(4),
            dispatch_parity=rt,
            dispatch_chunk_counter_ptr=table,
            num_chunks=1,
            num_ranks=_NUM_RANKS,
            num_experts_per_rank=_NUM_EXPERTS // _NUM_RANKS,
            rank=0,
        )
    elif block_index < fx.Int32(2 * _NUM_RANKS):
        combine_bf16_tile(
            sym,
            workspace,
            thread_index=thread_index,
            segment_idx=block_index,
            first_fold_row=rt,
            num_fold_rows=rt,
            recv_dst_rank_res=res(5),
            pool_row_to_route_res=res(6),
            pool_row_to_recv_token_res=res(7),
            recv_token_to_pool_rows_res=res(8),
            l2_ptr=table,
            epoch=epoch,
            reduce_bank=rt,
        )
    else:
        topk_reduce_bf16_tile(
            thread_index=thread_index,
            reduce_block_idx=block_index,
            num_reduce_warps=_BLOCK // 64,
            num_topk=_NUM_TOPK,
            hidden=_HIDDEN,
            num_experts=_NUM_EXPERTS,
            num_experts_per_rank=_NUM_EXPERTS // _NUM_RANKS,
            num_max_routes=int(workspace.num_max_routes),
            rank=0,
            combine_token_res=combine_token_res,
            output_res=res(9),
            topk_indices_res=res(10),
            num_tokens_res=res(11),
            reduce_flag_base=reduce_flag_base,
            reduce_bank=rt,
            epoch=epoch,
        )


# One kernel per jit: the dump directory is named after the kernel only when the module holds one.
@flyc.jit
def _compile_prims_probe(sym, rt: fx.Int32, stream: fx.Stream):
    mega_prims_probe(sym, rt).launch(grid=(1, 1, 1), block=(_BLOCK, 1, 1), stream=stream)


@flyc.jit
def _compile_ep_tiles_probe(sym, rt: fx.Int32, stream: fx.Stream):
    mega_ep_tiles_probe(sym, rt).launch(grid=(3 * _NUM_RANKS, 1, 1), block=(_BLOCK, 1, 1), stream=stream)


# ── ISA inspection ───────────────────────────────────────────────────────────

_MEM_OP = re.compile(r"^\s*((?:buffer|global|flat|scratch)_[a-z0-9_]+)\b(.*)$")
_GFX9_ONLY = ("s_memrealtime", "s_waitcnt ", "s_waitcnt\t")
# wait_mem_all is inline asm, so its wait prints as written (`0`); compiler-inserted waits print `0x0`
_GFX1250_REQUIRED = (r"s_sendmsg_rtn_b64", r"s_wait_storecnt 0\b")


def _policy(operands: str) -> str:
    """The cache-policy modifiers of one memory instruction, in either the gfx12 or the gfx9 spelling."""
    words = re.split(r"[\s,]+", operands)
    policy = [w for w in words if w.startswith(("th:", "scope:")) or w in ("sc0", "sc1", "nt")]
    return " ".join(policy) or "-"


def _summarize(isa: str):
    histogram = collections.Counter()
    for line in isa.splitlines():
        line = line.split(";")[0]
        match = _MEM_OP.match(line)
        if match:
            histogram[(match.group(1), _policy(match.group(2)))] += 1
    return histogram


def _check(kernel: str, isa: str, arch: str) -> list:
    problems = []
    if arch != "gfx1250":
        return problems
    for token in _GFX9_ONLY:
        if token in isa:
            problems.append(f"{kernel}: gfx9-only `{token.strip()}` in gfx1250 ISA")
    for pattern in _GFX1250_REQUIRED:
        if not re.search(pattern, isa):
            problems.append(f"{kernel}: missing `{pattern}`")
    for (op, policy), _ in _summarize(isa).items():
        # the EP path never asks for TH 3 (LU on loads, WB on stores): seeing it means a gfx9 literal leaked
        if "TH_LOAD_LU" in policy or "TH_STORE_WB" in policy or "th:0x3" in policy:
            problems.append(f"{kernel}: {op} carries {policy}, a gfx9 cache-policy literal")
    if "SCOPE_SYS" not in isa:
        problems.append(f"{kernel}: no system-scope memory op")
    return problems


def main():
    sym = SymBuffer(_FAKE_HEAPS, 0)
    _compile_prims_probe(sym, 1, None)
    _compile_ep_tiles_probe(sym, 1, None)
    problems = []
    for kernel in ("mega_prims_probe", "mega_ep_tiles_probe"):
        candidates = sorted(Path(ARGS.out).glob(f"*{kernel}*/*_final_isa.s"))
        if not candidates:
            problems.append(f"{kernel}: no final ISA under {ARGS.out}")
            continue
        isa = candidates[-1].read_text()
        print(f"== {kernel} ({ARGS.arch}): {candidates[-1]}")
        for (op, policy), count in sorted(_summarize(isa).items()):
            print(f"  {count:4d}  {op:32s} {policy}")
        problems += _check(kernel, isa, ARGS.arch)
    for problem in problems:
        print(f"FAIL {problem}")
    print("OK" if not problems else f"{len(problems)} problem(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
