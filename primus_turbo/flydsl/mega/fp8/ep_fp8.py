###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""fp8 EP comm closures for the FUSED mxfp8 dispatch+GEMM kernel (3-stage pipeline).

* ``dispatch_fp8_copy_tile``: the COMM role. One block CLEAN-pushes a comm task's
  PRE-QUANTIZED fp8 token rows (16B/lane b128, coalesced) + their RAW E8M0 block scales
  into the peer ``pool_fp8`` / ``pool_scale`` regions over XGMI, then (``signal``) drains
  with a device-scope L2 write-back and signals the peer per-pool-block scoreboard. No
  in-push quant (tokens are quantized once on the source) -> the push saturates XGMI.
* ``preshuffle_a_scale_tile``: the PRESHUFFLE role's per-block transpose. Cooperatively
  repacks ONE pool-block's raw E8M0 A-scale into the ScaleS2R broadcast layout so the gemm
  role reads it with a single coalesced load (fast MMA), done once per block (non-redundant).

Geometry: warp-per-token; hidden % 1024 == 0 (fp8 b128 push) and % 128 (MXFP8).
"""

import flydsl.expr as fx
from flydsl.compiler.ast_rewriter import InsertEmptyYieldForSCFFor
from flydsl.expr.buffer_ops import (
    buffer_load,
    buffer_store,
    create_buffer_resource_from_addr,
)

from primus_turbo.flydsl.mega.fp8.quant_flydsl import (
    _e8m0_broadcast_i32,
    _preshuffle_a_idx,
)
from primus_turbo.flydsl.mega.prims import atomic_add, l2_writeback
from primus_turbo.flydsl.utils.gemm_helper import _emit_if_then, ceildiv

_WARP = 64
_BLOCK_THREADS = 512
_VEC_I32 = 4  # 4 x i32 = 16B / lane (b128 XGMI)


def pool_scale_broadcast_i32(num_max_pool_tokens, hidden_size):
    """Element count of the pool_scale region in the ScaleS2R broadcast layout-1
    (int32): ceildiv(P, 64) * K128 * 256. This replaces the raw P*(hidden//32) bytes."""
    return ceildiv(num_max_pool_tokens, 64) * (hidden_size // 128) * 256


def preshuffle_a_scale_tile(
    scale_raw_res, scale_ps_res, m_row_base, block_m_rows, K128, thread_index, num_threads,
    cache_modifier=0, read_cache_modifier=0,
):
    """Cooperatively preshuffle ONE M-tile's raw E8M0 A-scale (``block_m_rows`` x K128 i32,
    rows [m_row_base, m_row_base+block_m_rows)) from the raw pool_scale into the ScaleS2R
    broadcast layout ``scale_ps`` (both LOCAL). Each (row, word) reads 1 raw i32 (=4
    micro-blocks) and scatters 4 broadcast i32. Used by the clean-push fused kernel's gemm
    role so the comm can push the RAW scale (coalesced XGMI) yet the MMA reads it preshuffled.

    ``cache_modifier`` (default 0): pass 16 (sc1 write-through to HBM) when the SAME
    workgroup both writes ``scale_ps`` and reads it back for its own GEMM (fused-preshuffle
    2-stage): write-through keeps the scratch coherent even if a concurrent workgroup's
    device-wide ``buffer_inv sc1`` drops the L2 line (data is already in HBM), so no
    device-wide ``l2_writeback`` handoff is needed.

    Requires ``block_m_rows * K128 % num_threads == 0`` (256*K128 % 512 == 0 for K128 even)."""
    total = block_m_rows * K128
    assert total % num_threads == 0, f"preshuffle tile {total} not divisible by {num_threads}"
    for it in range(total // num_threads):
        idx = thread_index + fx.Int32(it * num_threads)
        r = idx // fx.Int32(K128)
        k = idx % fx.Int32(K128)
        row = m_row_base + r
        word = buffer_load(
            scale_raw_res, row * fx.Int32(K128) + k, vec_width=1, dtype=fx.T.i32(),
            cache_modifier=read_cache_modifier,
        )
        for g in range(4):
            b = k * fx.Int32(4) + fx.Int32(g)
            byte = (fx.arith.ArithValue(word) >> fx.Int32(8 * g)) & fx.Int32(0xFF)
            buffer_store(
                _e8m0_broadcast_i32(byte), scale_ps_res, _preshuffle_a_idx(row, b, K128),
                cache_modifier=cache_modifier,
            )


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
    xq_resource,  # local pre-quantized fp8 tokens viewed int32 [T, hidden//4]
    xs_resource,  # local raw E8M0 scales viewed int32 [T, hidden//128]
    expert_send_dst_rank_resource,
    expert_send_dst_row_resource,
    expert_send_count_resource,
    expert_send_offset_resource,
    dispatched_token_idx_resource,
    pool_fp8_base,  # sym_layout.pool_fp8_ptr
    pool_scale_base,  # RAW: sym_layout.pool_scale_ptr ; BROADCAST: sym_layout.pool_scale_ps_ptr
    pool_offsets_resource,
    signal=False,
    scoreboard_base=None,
    scoreboard_offsets_resource=None,
    block_m=0,
    push_scale=True,
    fence=True,
    tok_unroll=1,
):
    """CLEAN fp8 comm PUSH closure (no in-push quant): copy a comm task's PRE-QUANTIZED
    fp8 tokens (16B/lane b128, coalesced) into the peer ``pool_fp8``, plus their RAW E8M0
    scales (coalesced) into the peer ``pool_scale``, then (``signal``) drain with a
    device-scope L2 write-back and signal the peer per-pool-block scoreboard. Mirrors
    ``dispatch_fp8_push`` (saturates XGMI) but with the fused kernel's multi-task-per-block
    distribution so the preshuffle/gemm roles can overlap. The preshuffle role transposes
    the raw scale to the ScaleS2R broadcast layout on the dest."""
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

    def copy_slice(dest_row_start, source_offset, peer_pool, peer_pscale, tok_lo, tok_hi):
        local_count = (tok_hi - tok_lo - warp_id + fx.Int32(n_warps - 1)) // fx.Int32(n_warps)

        def _row(i):
            row_index = tok_lo + warp_id + i * fx.Int32(n_warps)
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
            if push_scale:
                def _one_scale():
                    sv = buffer_load(
                        xs_resource, source_row * fx.Int32(scale_i32) + lane_id, vec_width=1, dtype=fx.T.i32()
                    )
                    buffer_store(sv, peer_pscale, dest_row * fx.Int32(scale_i32) + lane_id)

                _emit_if_then(lane_id < fx.Int32(scale_i32), _one_scale)

        _emit_for(local_count, _row)

    def copy_slice_unrolled(dest_row_start, source_offset, peer_pool, peer_pscale, tok_lo, tok_hi):
        # tok_unroll>1: each warp processes `tok_unroll` token rows per loop iteration -> the
        # b128 loads of all U rows are issued before any store, deepening in-flight XGMI traffic
        # (memory-level parallelism) to better hide push latency. Same warp-strided row set as
        # copy_slice; loads use a clamped in-range index (buffer-safe redundant read for the tail)
        # and the store is guarded, so it is numerically identical to copy_slice.
        U = tok_unroll
        local_count = (tok_hi - tok_lo - warp_id + fx.Int32(n_warps - 1)) // fx.Int32(n_warps)
        n_iter = (local_count + fx.Int32(U - 1)) // fx.Int32(U)

        def _batch(j):
            recs = []
            for u in range(U):
                i = j * fx.Int32(U) + fx.Int32(u)
                valid = i < local_count
                i_c = fx.arith.select(valid, i, fx.Int32(0))
                row_index = tok_lo + warp_id + i_c * fx.Int32(n_warps)
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
                sval = None
                if push_scale:
                    sval = buffer_load(
                        xs_resource, source_row * fx.Int32(scale_i32) + lane_id, vec_width=1, dtype=fx.T.i32()
                    )
                recs.append((valid, dest_row, vals, sval))
            # zero-arg closures + immediate _emit_if_then call => closure capture is safe (matches
            # the copy_slice/_one_scale idiom). Branch fns MUST take 0 args: ReplaceIfWithDispatch
            # injects result_names into any arg-accepting branch fn.
            def _emit_store(valid, dest_row, vals, sval):
                def _store():
                    for c in fx.range_constexpr(chunk_count):
                        col = fx.Int32(c * cols_per_warp_i32) + lane_id * fx.Int32(_VEC_I32)
                        buffer_store(vals[c], peer_pool, dest_row * fx.Int32(hidden_i32) + col)
                    if push_scale:
                        def _one_scale():
                            buffer_store(sval, peer_pscale, dest_row * fx.Int32(scale_i32) + lane_id)

                        _emit_if_then(lane_id < fx.Int32(scale_i32), _one_scale)

                _emit_if_then(valid, _store)

            for valid, dest_row, vals, sval in recs:
                _emit_store(valid, dest_row, vals, sval)

        _emit_for(n_iter, _batch)

    def dispatch_tile(task_index, sub, n_sub):
        dst_rank, dest_row_start, source_offset, token_count, peer_pool, peer_pscale = load_task(task_index)
        if n_sub == 1:
            tok_lo = fx.Int32(0)
            tok_hi = token_count
        else:
            slice_tokens = (token_count + fx.Int32(n_sub - 1)) // fx.Int32(n_sub)
            tok_lo = sub * slice_tokens
            tok_hi = fx.arith.select(tok_lo + slice_tokens < token_count, tok_lo + slice_tokens, token_count)
        if tok_unroll == 1:
            copy_slice(dest_row_start, source_offset, peer_pool, peer_pscale, tok_lo, tok_hi)
        else:
            copy_slice_unrolled(dest_row_start, source_offset, peer_pool, peer_pscale, tok_lo, tok_hi)
        if signal:
            fx.rocdl.s_waitcnt(0)
            if fence:
                l2_writeback()
            fx.gpu.barrier()

            def _signal():
                scoreboard_address = _peer_addr(scoreboard_base, scoreboard_offsets_resource, dst_rank)
                first_block = (dest_row_start + tok_lo) // fx.Int32(block_m)
                last_block = (dest_row_start + tok_hi - fx.Int32(1)) // fx.Int32(block_m)
                _emit_for(
                    last_block - first_block + fx.Int32(1),
                    lambda bo: atomic_add(scoreboard_address, first_block + bo, fx.Int32(1), scope="sys"),
                )

            _emit_if_then(thread_index == fx.Int32(0), _signal)

    return dispatch_tile
