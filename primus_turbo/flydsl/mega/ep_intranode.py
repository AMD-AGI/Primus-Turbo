###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Intra-node EP comm layer (FlyDSL): precision-agnostic XGMI push + scoreboard handshake."""

import flydsl.expr as fx
from flydsl.compiler.ast_rewriter import InsertEmptyYieldForSCFFor
from flydsl.expr.buffer_ops import (
    buffer_load,
    buffer_store,
    create_buffer_resource_from_addr,
)

from primus_turbo.flydsl.common.tile_spec import _emit_if_then
from primus_turbo.flydsl.mega.prims import atomic_add

_VEC = 16  # fp8 bytes per lane per push step (b128 XGMI-wide)
_WARP = 64  # gfx950 wavefront size
_BLOCK_THREADS = 512  # 8 waves (2 x 4): tile-spec block size


def _peer_addr(local_base, offsets_resource, dst_rank):
    """Peer base addr = local base + per-peer i64 delta."""
    delta = buffer_load(offsets_resource, dst_rank, vec_width=1, dtype=fx.T.i64())
    return local_base + delta


def _emit_for(stop, body):
    """Runtime for i in range(stop) driving scf.for directly (body-only rewrite)."""
    InsertEmptyYieldForSCFFor.scf_for_dispatch(
        fx.Int32(0), stop, fx.Int32(1), lambda iv, _names: body(fx.arith.ArithValue(iv, signed=True))
    )


# Byte-agnostic warp-per-token XGMI push (shared by fp8 + bf16).
def _make_dispatch_tile(
    *,
    thread_index,
    n_warps,
    hidden_i32,
    cols_per_warp_i32,
    vec_i32,
    chunk_count,
    pool_record_bytes,
    input_resource,
    expert_send_dst_rank_resource,
    expert_send_dst_row_resource,
    expert_send_count_resource,
    expert_send_offset_resource,
    dispatched_token_idx_resource,
    pool_address_resource,
    signal=False,
    scoreboard_address_resource=None,
    block_m=0,
    pool_base=None,
    pool_offsets_resource=None,
    scoreboard_base=None,
    scoreboard_offsets_resource=None,
):
    """One block pushes a token slice of one task to its peer pool; signal=True adds
    the release fence + scoreboard signal."""
    warp_id = thread_index // fx.Int32(_WARP)
    lane_id = thread_index % fx.Int32(_WARP)

    def load_task(task_index):
        # read all per-task metadata once
        destination_rank = buffer_load(
            expert_send_dst_rank_resource, task_index, vec_width=1, dtype=fx.T.i32()
        )
        dest_row_start = buffer_load(expert_send_dst_row_resource, task_index, vec_width=1, dtype=fx.T.i32())
        source_offset = buffer_load(expert_send_offset_resource, task_index, vec_width=1, dtype=fx.T.i32())
        token_count = buffer_load(expert_send_count_resource, task_index, vec_width=1, dtype=fx.T.i32())
        # peer pool base: delta-map when given, else [world] ptr table
        if pool_base is None:
            pool_address = buffer_load(pool_address_resource, destination_rank, vec_width=1, dtype=fx.T.i64())
        else:
            pool_address = _peer_addr(pool_base, pool_offsets_resource, destination_rank)
        peer_pool = create_buffer_resource_from_addr(pool_address, num_records_bytes=pool_record_bytes)
        return destination_rank, dest_row_start, source_offset, token_count, peer_pool

    def copy_slice(dest_row_start, source_offset, peer_pool, tok_lo, tok_hi):
        # warp-per-token copy of [tok_lo, tok_hi)
        local_count = (tok_hi - tok_lo - warp_id + fx.Int32(n_warps - 1)) // fx.Int32(n_warps)

        def _row(i):
            row_index = tok_lo + warp_id + i * fx.Int32(n_warps)

            def _push_row():
                source_row = buffer_load(
                    dispatched_token_idx_resource, source_offset + row_index, vec_width=1, dtype=fx.T.i32()
                )
                dest_row = dest_row_start + row_index
                chunk_values = []
                for chunk_index in fx.range_constexpr(chunk_count):
                    column = fx.Int32(chunk_index * cols_per_warp_i32) + lane_id * fx.Int32(vec_i32)
                    chunk_values.append(
                        buffer_load(
                            input_resource,
                            source_row * fx.Int32(hidden_i32) + column,
                            vec_width=vec_i32,
                            dtype=fx.T.i32(),
                        )
                    )
                for chunk_index in fx.range_constexpr(chunk_count):
                    column = fx.Int32(chunk_index * cols_per_warp_i32) + lane_id * fx.Int32(vec_i32)
                    buffer_store(
                        chunk_values[chunk_index], peer_pool, dest_row * fx.Int32(hidden_i32) + column
                    )

            _push_row()

        _emit_for(local_count, _row)

    def dispatch_tile(task_index, sub, n_sub):
        destination_rank, dest_row_start, source_offset, token_count, peer_pool = load_task(task_index)
        if n_sub == 1:
            tok_lo = fx.Int32(0)
            tok_hi = token_count
        else:
            slice_tokens = (token_count + fx.Int32(n_sub - 1)) // fx.Int32(n_sub)
            tok_lo = sub * slice_tokens
            tok_hi = fx.arith.select(tok_lo + slice_tokens < token_count, tok_lo + slice_tokens, token_count)
        copy_slice(dest_row_start, source_offset, peer_pool, tok_lo, tok_hi)
        if signal:
            fx.rocdl.s_waitcnt(0)
            fx.gpu.barrier()

            def _signal():
                # peer scoreboard base: delta-map when given, else ptr table
                if scoreboard_base is None:
                    scoreboard_address = buffer_load(
                        scoreboard_address_resource, destination_rank, vec_width=1, dtype=fx.T.i64()
                    )
                else:
                    scoreboard_address = _peer_addr(
                        scoreboard_base, scoreboard_offsets_resource, destination_rank
                    )
                first_block = (dest_row_start + tok_lo) // fx.Int32(block_m)
                last_block = (dest_row_start + tok_hi - fx.Int32(1)) // fx.Int32(block_m)
                _emit_for(
                    last_block - first_block + fx.Int32(1),
                    lambda bo: atomic_add(scoreboard_address, first_block + bo, fx.Int32(1), scope="sys"),
                )

            _emit_if_then(thread_index == fx.Int32(0), _signal)

    return dispatch_tile


# bf16 push geometry (2 bytes/element); rows pushed as i32 words.
def _bf16_push_geom(hidden_size):
    hidden_bytes = hidden_size * 2
    assert (
        hidden_bytes % (_WARP * _VEC) == 0
    ), "hidden*2 must be a multiple of 1024 bytes -> hidden % 512 == 0"
    vec_i32 = _VEC // 4
    hidden_i32 = hidden_bytes // 4
    n_warps = _BLOCK_THREADS // _WARP
    cols_per_warp_i32 = _WARP * vec_i32
    chunk_count = hidden_i32 // cols_per_warp_i32
    return vec_i32, hidden_i32, n_warps, cols_per_warp_i32, chunk_count


def dispatch_bf16_tile(
    *,
    thread_index,
    hidden_size,
    num_max_pool_tokens,
    input_resource,
    expert_send_dst_rank_resource,
    expert_send_dst_row_resource,
    expert_send_count_resource,
    expert_send_offset_resource,
    dispatched_token_idx_resource,
    pool_address_resource,
    signal=False,
    scoreboard_address_resource=None,
    block_m=0,
    pool_base=None,
    pool_offsets_resource=None,
    scoreboard_base=None,
    scoreboard_offsets_resource=None,
):
    """bf16 comm PUSH closure (wraps the shared byte push with bf16 geometry).
    Peer addressing: pass [world] absolute ptr tables, OR the two-heap delta path
    (pool_base/pool_offsets_resource + scoreboard_base/scoreboard_offsets_resource)."""
    vec_i32, hidden_i32, n_warps, cols_per_warp_i32, chunk_count = _bf16_push_geom(hidden_size)
    pool_record_bytes = num_max_pool_tokens * hidden_size * 2
    return _make_dispatch_tile(
        thread_index=thread_index,
        n_warps=n_warps,
        hidden_i32=hidden_i32,
        cols_per_warp_i32=cols_per_warp_i32,
        vec_i32=vec_i32,
        chunk_count=chunk_count,
        pool_record_bytes=pool_record_bytes,
        input_resource=input_resource,
        expert_send_dst_rank_resource=expert_send_dst_rank_resource,
        expert_send_dst_row_resource=expert_send_dst_row_resource,
        expert_send_count_resource=expert_send_count_resource,
        expert_send_offset_resource=expert_send_offset_resource,
        dispatched_token_idx_resource=dispatched_token_idx_resource,
        pool_address_resource=pool_address_resource,
        signal=signal,
        scoreboard_address_resource=scoreboard_address_resource,
        block_m=block_m,
        pool_base=pool_base,
        pool_offsets_resource=pool_offsets_resource,
        scoreboard_base=scoreboard_base,
        scoreboard_offsets_resource=scoreboard_offsets_resource,
    )
