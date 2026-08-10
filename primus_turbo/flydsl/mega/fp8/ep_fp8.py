###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""The COMM role of the fused mxfp8 dispatch+GEMM kernel: ``dispatch_fp8_copy_tile``.

Warp-per-token geometry; hidden % 1024 == 0 (fp8 b128 push) and % 128 (MXFP8).
"""

import flydsl.expr as fx
from flydsl.compiler.ast_rewriter import InsertEmptyYieldForSCFFor
from flydsl.expr.buffer_ops import (
    buffer_load,
    buffer_store,
    create_buffer_resource_from_addr,
)

from primus_turbo.flydsl.mega.ep_intranode import _BLOCK_THREADS, _WARP
from primus_turbo.flydsl.mega.fp8.prims import atomic_add, l2_writeback
from primus_turbo.flydsl.utils.gemm_helper import emit_if_then

_VEC_I32 = 4  # 4 x i32 = 16B / lane (b128 XGMI)


def _peer_addr(local_base, offsets_resource, dst_rank):
    return local_base + buffer_load(offsets_resource, dst_rank, vec_width=1, dtype=fx.T.i64())


def _emit_for(stop, body):
    InsertEmptyYieldForSCFFor.scf_for_dispatch(
        fx.Int32(0), stop, fx.Int32(1), lambda iv, _names: body(fx.arith.ArithValue(iv, signed=True))
    )


def dispatch_fp8_copy_tile(
    *,
    thread_index,
    hidden_size,
    num_max_pool_tokens,
    xq_resource,  # pre-quant fp8 tokens int32 [T, hidden//4]
    xs_resource,  # raw E8M0 scales int32 [T, hidden//128]
    expert_send_dst_rank_resource,
    expert_send_dst_row_resource,
    expert_send_count_resource,
    expert_send_offset_resource,
    dispatched_token_idx_resource,
    pool_fp8_base,  # sym_layout.pool_fp8_ptr
    pool_scale_base,  # RAW: sym_layout.pool_scale_ptr ; BROADCAST: sym_layout.pool_scale_ps_ptr
    pool_offsets_resource,
    dispatch_flag_base,
    dispatch_flag_offsets_resource,
    bank,
    world_size,
):
    """Push one comm task's PRE-QUANTIZED fp8 rows (16B/lane b128) and RAW E8M0 scales into the
    peer's ``pool_fp8`` / ``pool_scale``, then release and signal its scoreboard.

    Tokens are quantized once on the source rather than in-push, which keeps both stores coalesced
    and the push XGMI-bound. Blocks take multiple tasks each so the preshuffle/gemm roles overlap;
    the raw scale is transposed to ScaleS2R on the destination by the preshuffle role."""
    assert hidden_size % 1024 == 0, f"fp8 token push needs hidden % 1024 == 0, got {hidden_size}"
    n_warps = _BLOCK_THREADS // _WARP
    hidden_i32 = hidden_size // 4  # fp8 row: hidden bytes -> i32 words
    cols_per_warp_i32 = _WARP * _VEC_I32  # 256
    chunk_count = hidden_i32 // cols_per_warp_i32  # = hidden // 1024
    scale_i32 = hidden_size // 128  # raw E8M0 row: hidden//32 bytes -> i32 words (=K128); 4 micro-blocks/word
    assert scale_i32 <= _WARP, f"raw scale row {scale_i32} i32 > warp {_WARP} (hidden > 8192 unsupported)"
    pool_tok_bytes = num_max_pool_tokens * hidden_size  # fp8 pool records (bytes)
    pool_scale_bytes = num_max_pool_tokens * (hidden_size // 32)  # RAW E8M0 pool records (bytes)

    warp_id = thread_index // fx.Int32(_WARP)
    lane_id = thread_index % fx.Int32(_WARP)

    def load_task(task_index):
        destination_rank = buffer_load(expert_send_dst_rank_resource, task_index, vec_width=1, dtype=fx.T.i32())
        dest_row_start = buffer_load(expert_send_dst_row_resource, task_index, vec_width=1, dtype=fx.T.i32())
        source_offset = buffer_load(expert_send_offset_resource, task_index, vec_width=1, dtype=fx.T.i32())
        token_count = buffer_load(expert_send_count_resource, task_index, vec_width=1, dtype=fx.T.i32())
        pool_addr = _peer_addr(pool_fp8_base, pool_offsets_resource, destination_rank)
        peer_pool = create_buffer_resource_from_addr(pool_addr, num_records_bytes=pool_tok_bytes)
        pscale_addr = _peer_addr(pool_scale_base, pool_offsets_resource, destination_rank)
        peer_pscale = create_buffer_resource_from_addr(pscale_addr, num_records_bytes=pool_scale_bytes)
        return destination_rank, dest_row_start, source_offset, token_count, peer_pool, peer_pscale

    def copy_rows(dest_row_start, source_offset, peer_pool, peer_pscale, token_count):
        # warp-strided over the task's rows: warp w takes rows w, w+n_warps, ...
        local_count = (token_count - warp_id + fx.Int32(n_warps - 1)) // fx.Int32(n_warps)

        def _row(i):
            row_index = warp_id + i * fx.Int32(n_warps)
            source_row = buffer_load(
                dispatched_token_idx_resource, source_offset + row_index, vec_width=1, dtype=fx.T.i32()
            )
            dest_row = dest_row_start + row_index
            vals = []
            for c in fx.range_constexpr(chunk_count):
                col = fx.Int32(c * cols_per_warp_i32) + lane_id * fx.Int32(_VEC_I32)
                vals.append(
                    buffer_load(
                        xq_resource, source_row * fx.Int32(hidden_i32) + col, vec_width=_VEC_I32, dtype=fx.T.i32()
                    )
                )
            for c in fx.range_constexpr(chunk_count):
                col = fx.Int32(c * cols_per_warp_i32) + lane_id * fx.Int32(_VEC_I32)
                buffer_store(vals[c], peer_pool, dest_row * fx.Int32(hidden_i32) + col)
            def _one_scale():
                sv = buffer_load(
                    xs_resource, source_row * fx.Int32(scale_i32) + lane_id, vec_width=1, dtype=fx.T.i32()
                )
                buffer_store(sv, peer_pscale, dest_row * fx.Int32(scale_i32) + lane_id)

            emit_if_then(lane_id < fx.Int32(scale_i32), _one_scale)

        _emit_for(local_count, _row)


    def dispatch_tile(task_index):
        # A block owns whole tasks -- the caller rotates it over them -- so no task is split.
        dst_rank, dest_row_start, source_offset, token_count, peer_pool, peer_pscale = load_task(task_index)
        copy_rows(dest_row_start, source_offset, peer_pool, peer_pscale, token_count)
        fx.rocdl.s_waitcnt(0)
        # Device-scope release before the flag: without it the peer can observe the signal ahead of
        # the rows it announces.
        l2_writeback()
        fx.gpu.barrier()

        def _signal():
            # One +1 to the peer's dispatch_flag[bank + local_expert]. The task table is dense
            # [local_expert][dst_rank], so each dst expert receives exactly num_ranks of them and
            # reaches the cumulative expected -- no host-reset scoreboard needed.
            df_address = _peer_addr(dispatch_flag_base, dispatch_flag_offsets_resource, dst_rank)
            local_expert = task_index // fx.Int32(world_size)
            atomic_add(df_address, bank + local_expert, fx.Int64(1), scope="sys")

        emit_if_then(thread_index == fx.Int32(0), _signal)

    return dispatch_tile
