###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

# The V2 dispatch's data path alone, as a FlyDSL kernel (bench_ceiling.py): NUM_THREADS per block, one
# warp per row, each row loaded once and stored to every peer (or to this rank: to_self) into the region of
# the source rank in a symmetric heap. use_tdm: Tensor Data Mover through LDS, stores with SCOPE_SYS, as V2
# on gfx1250; else the warp copy through VGPRs of V2's non-TDM path.
#
# COPY_WARPS=n: only n warps per block copy, each with MAX_LDS_BYTES / n of LDS for its row (the V2 combine
# pushes with 22 warps at hidden 7168, a whole row per warp; 32 warps move a row in two pieces). Read once per
# process, so give each value its own FLYDSL_RUNTIME_CACHE_DIR.
import os

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith, range_constexpr
from flydsl.expr.rocdl import tdm_ops

from primus_turbo.flydsl.deep_ep.common import SYS_STORE, WARP_SIZE, lds_base, tensor_addr, wait_mem_all
from primus_turbo.flydsl.deep_ep.intranode_kernel import (
    NUM_THREADS,
    NUM_WARPS,
    _addr,
    _load_ptr,
    _uniform_i64,
)
from primus_turbo.flydsl.deep_ep.intranode_v2_kernel import MAX_LDS_BYTES, _align, _fanout_row, _tdm_row
from primus_turbo.flydsl.utils.buffer_ops import create_buffer_resource, create_buffer_resource_from_addr
from primus_turbo.flydsl.utils.prims import _i64, _readfirstlane_i32

COPY_WARPS = min(int(os.environ.get("COPY_WARPS", NUM_WARPS)), NUM_WARPS)


def _make_copy(world, num_blocks, row_bytes, region_bytes, use_tdm, to_self):
    row_buffer_bytes = min(_align(row_bytes, 128), MAX_LDS_BYTES // COPY_WARPS // 128 * 128)
    lds_bytes = COPY_WARPS * row_buffer_bytes if use_tdm else 0
    targets = (0,) if to_self else tuple(range(1, world))

    @flyc.kernel(known_block_size=[NUM_THREADS, 1, 1])
    def copy_kernel(SRC: fx.Tensor, DST_PTRS: fx.Tensor, num_rows: fx.Int32, rank: fx.Int32):
        thread_id = fx.thread_idx.x
        warp_id = thread_id // fx.Int32(WARP_SIZE)
        ptrs_res = create_buffer_resource(DST_PTRS, max_size=True)
        src_base = tensor_addr(SRC)
        region = _i64(rank) * fx.Int64(region_bytes)
        dst_bases = [_load_ptr(ptrs_res, (rank + fx.Int32(j)) % fx.Int32(world)) + region for j in targets]
        row_lds = None
        if fx.const_expr(use_tdm):
            row_lds = fx.arith.ArithValue(lds_base(), signed=True) + _readfirstlane_i32(warp_id) * fx.Int32(
                row_buffer_bytes
            )
        row = fx.block_idx.x * fx.Int32(COPY_WARPS) + warp_id
        row = arith.select(warp_id < fx.Int32(COPY_WARPS), row, num_rows)
        while row < num_rows:
            if fx.const_expr(use_tdm):
                for p in range_constexpr(0, row_bytes, row_buffer_bytes):
                    piece = min(row_buffer_bytes, row_bytes - p)
                    tdm_ops.tensor_wait(0)
                    tdm_ops.tensor_load_2d(
                        _tdm_row(row_lds, _uniform_i64(_addr(src_base, p, row, row_bytes)), piece)
                    )
                    tdm_ops.tensor_wait(0)
                    for j in range_constexpr(len(dst_bases)):
                        tdm_ops.tensor_store_2d(
                            _tdm_row(row_lds, _uniform_i64(_addr(dst_bases[j], p, row, row_bytes)), piece),
                            cache_policy=SYS_STORE,
                        )
            else:
                dst_resources = [
                    create_buffer_resource_from_addr(
                        _uniform_i64(_addr(dst_bases[j], 0, row, row_bytes)), num_records_bytes=row_bytes
                    )
                    for j in range(len(dst_bases))
                ]
                src_res = create_buffer_resource_from_addr(
                    _uniform_i64(_addr(src_base, 0, row, row_bytes)), num_records_bytes=row_bytes
                )
                _fanout_row(dst_resources, src_res, row_bytes)
            row = row + fx.Int32(num_blocks * COPY_WARPS)
        if fx.const_expr(use_tdm):
            tdm_ops.tensor_wait(0)
        wait_mem_all()

    return copy_kernel, lds_bytes


@flyc.jit
def launch_copy(
    src,
    dst_ptrs,
    num_rows: fx.Int32,
    rank: fx.Int32,
    world: fx.Constexpr[int],
    num_blocks: fx.Constexpr[int],
    row_bytes: fx.Constexpr[int],
    region_bytes: fx.Constexpr[int],
    use_tdm: fx.Constexpr[int],
    to_self: fx.Constexpr[int],
    stream: fx.Stream,
):
    kernel, lds_bytes = _make_copy(world, num_blocks, row_bytes, region_bytes, use_tdm, to_self)
    kernel(src, dst_ptrs, num_rows, rank).launch(
        grid=(num_blocks, 1, 1), block=(NUM_THREADS, 1, 1), stream=stream, smem=lds_bytes
    )
