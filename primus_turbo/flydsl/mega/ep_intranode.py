from typing import Optional

import flydsl.expr as fx
from flydsl._mlir.dialects import vector as _vector
from flydsl.compiler.ast_rewriter import ASTRewriter
from flydsl.expr import arith, const_expr, range_constexpr
from flydsl.expr.buffer_ops import (
    buffer_load,
    buffer_store,
    create_buffer_resource_from_addr,
)

from primus_turbo.flydsl.mega.prims import (
    _wait_mem,
    atomic_add,
    copy_warp,
    ld,
    read_clock,
    spin_timed_out,
    st,
)
from primus_turbo.flydsl.mega.sym_layout import SymLayout, map_signal, sym_map

_WARP = 64
_BLOCK_THREADS = 512
_PVEC = 8
_NUM_WARPS = _BLOCK_THREADS // _WARP
_L1_BYPASS = 1  # buffer cache_modifier: skip L1 (read fresh L2Y after l2_invalidate)


@ASTRewriter.transform
def dispatch_bf16_tile(
    sym_layout: SymLayout,
    thread_index: fx.Int32,
    hidden_size: int,
    num_max_pool_tokens: int,
    input_res: fx.ArithValue,
    expert_send_dst_rank_res: fx.ArithValue,
    expert_send_dst_row_res: fx.ArithValue,
    expert_send_count_res: fx.ArithValue,
    expert_send_offset_res: fx.ArithValue,
    dispatched_token_idx_res: fx.ArithValue,
    task_index: fx.ArithValue,
    signal: bool = False,
    block_m: int = 0,
    disp_parity: Optional[fx.Int32] = None,
    num_ranks: int = 0,
):
    hidden_bytes = hidden_size * 2
    assert hidden_bytes % 1024 == 0, "hidden*2 must be a multiple of 1024 bytes -> hidden % 512 == 0"
    hidden_i32 = hidden_bytes // 4  # row stride in i32 words

    warp_id = thread_index // fx.Int32(_WARP)

    dst_rank = buffer_load(expert_send_dst_rank_res, task_index, vec_width=1, dtype=fx.T.i32())
    dest_row_start = buffer_load(expert_send_dst_row_res, task_index, vec_width=1, dtype=fx.T.i32())
    source_offset = buffer_load(expert_send_offset_res, task_index, vec_width=1, dtype=fx.T.i32())
    token_count = buffer_load(expert_send_count_res, task_index, vec_width=1, dtype=fx.T.i32())
    pool_address = sym_map(sym_layout, sym_layout.pool_ptr, dst_rank)

    local_count = (token_count - warp_id + fx.Int32(_NUM_WARPS - 1)) // fx.Int32(_NUM_WARPS)

    for i in range(local_count):
        row_index = warp_id + i * fx.Int32(_NUM_WARPS)
        source_row = buffer_load(
            dispatched_token_idx_res, source_offset + row_index, vec_width=1, dtype=fx.T.i32()
        )
        dest_row = dest_row_start + row_index
        # dst = peer pool (base addr), src = local input (resource); offsets in i32 words
        copy_warp(
            pool_address,
            input_res,
            hidden_bytes,
            dst_off=dest_row * fx.Int32(hidden_i32),
            src_off=source_row * fx.Int32(hidden_i32),
        )

    if const_expr(signal):
        fx.rocdl.s_waitcnt(0)
        fx.gpu.barrier()
        if thread_index == fx.Int32(0):
            dispatch_flag_address = map_signal(sym_layout, sym_layout.dispatch_flag_ptr, dst_rank)
            bank = (
                fx.Int32(0)
                if disp_parity is None
                else disp_parity * fx.Int32(int(sym_layout.num_max_pool_blocks))
            )
            # DeepEP parity: every (source,expert) task (incl. zero-token) bumps the dst expert
            # counter +1 -> each expert accumulates exactly `world` per launch (host-predictable).
            local_expert = task_index // fx.Int32(num_ranks)
            atomic_add(dispatch_flag_address, bank + local_expert, fx.Int64(1), scope="sys")


@ASTRewriter.transform
def combine_bf16_tile(
    sym_layout: SymLayout,
    thread_index: fx.Int32,
    block_m: fx.ArithValue,
    block_m_size: int,
    grad_gate_res: Optional[fx.ArithValue] = None,
    signal: bool = False,
    epoch: Optional[fx.Int64] = None,
    bank_offset: Optional[fx.Int32] = None,
    with_gate: bool = False,
):
    out_features = int(sym_layout.hidden)
    n_slots = int(sym_layout.combine_slots)
    num_pool_tokens = int(sym_layout.num_max_pool_tokens)
    comb_records = n_slots * out_features * 2  # bf16 peer combine-buffer bound (bytes)
    gate_records = n_slots * 4  # f32 gate slots per peer

    assert block_m_size % _NUM_WARPS == 0, "block_m must be a multiple of num_warps (8)"
    cols_per_step = _WARP * _PVEC
    num_full_chunks = out_features // cols_per_step
    tail_cols = out_features % cols_per_step
    rows_per_warp = block_m_size // _NUM_WARPS
    lane_id = thread_index % fx.Int32(_WARP)
    warp_id = thread_index // fx.Int32(_WARP)
    chunk_base = warp_id * fx.Int32(rows_per_warp)
    l2_ptr = sym_layout.l2_token_buffer_ptr
    row_bytes = out_features * 2
    row_words = out_features // 2  # i32 words per row (bf16 -> 2 elems / word)
    full_bytes = num_full_chunks * cols_per_step * 2  # the b128-aligned prefix copy_warp moves

    origin_rank_res = create_buffer_resource_from_addr(
        sym_layout.origin_rank_ptr, num_records_bytes=num_pool_tokens * 4
    )
    origin_slot_res = create_buffer_resource_from_addr(
        sym_layout.origin_slot_ptr, num_records_bytes=num_pool_tokens * 4
    )
    if tail_cols:  # masked remainder needs whole-buffer resources (only when out_features % 512)
        l2_token_buffer_res = create_buffer_resource_from_addr(
            l2_ptr, num_records_bytes=num_pool_tokens * row_bytes
        )

    base_row = block_m * fx.Int32(block_m_size) + chunk_base
    for j in range_constexpr(rows_per_warp):
        row = base_row + fx.Int32(j)
        origin = buffer_load(origin_rank_res, row, vec_width=1, dtype=fx.T.i32())
        if origin >= fx.Int32(0):
            slot = buffer_load(origin_slot_res, row, vec_width=1, dtype=fx.T.i32())
            comb_addr = map_signal(sym_layout, sym_layout.comb_ptr, origin)
            # dst = peer comb (base addr), src = local L2Y (base addr); offsets in i32 words
            copy_warp(
                comb_addr,
                l2_ptr,
                full_bytes,
                dst_off=slot * fx.Int32(row_words),
                src_off=row * fx.Int32(row_words),
            )
            if const_expr(tail_cols):
                oob_index = fx.Int32(n_slots) * fx.Int32(out_features)
                slot_base = slot * fx.Int32(out_features)
                row_off = row * fx.Int32(out_features)
                peer = create_buffer_resource_from_addr(comb_addr, num_records_bytes=comb_records)
                col = fx.Int32(num_full_chunks * cols_per_step) + lane_id * fx.Int32(_PVEC)
                in_tail = (lane_id * fx.Int32(_PVEC)) < fx.Int32(tail_cols)
                safe_col = arith.select(in_tail, col, fx.Int32(out_features - _PVEC))
                tail_value = buffer_load(
                    l2_token_buffer_res,
                    row_off + safe_col,
                    vec_width=_PVEC,
                    dtype=fx.T.bf16(),
                    cache_modifier=_L1_BYPASS,
                )
                dst = arith.select(in_tail, slot_base + col, oob_index)
                buffer_store(tail_value, peer, dst)
            if const_expr(with_gate):
                gate_value = buffer_load(grad_gate_res, row, vec_width=1, dtype=fx.T.f32())
                gate_addr = sym_map(sym_layout, sym_layout.combine_gate_ptr, origin)
                gate_peer = create_buffer_resource_from_addr(gate_addr, num_records_bytes=gate_records)
                buffer_store(gate_value, gate_peer, slot)
            if const_expr(signal):
                bank = fx.Int32(0) if bank_offset is None else bank_offset
                barrier_addr = map_signal(sym_layout, sym_layout.reduce_flag_ptr, origin)
                _wait_mem()
                st(barrier_addr, bank + slot, epoch, scope="sys")


@ASTRewriter.transform
def topk_reduce_bf16_tile(
    signal: bool,
    apply_weights: bool,
    with_gate: bool,
    thread_index: fx.Int32,
    base_pid: fx.Int32,
    total_warps: fx.Int32,
    topk: int,
    out_features: int,
    num_experts: int,
    rank: int,
    comb_local_res: fx.ArithValue,
    output_res: fx.ArithValue,
    topk_indices_res: fx.ArithValue,
    num_tokens_res: fx.ArithValue,
    barrier_base: fx.ArithValue,
    reduce_bank: fx.Int32,
    topk_weights_res: fx.ArithValue,
    gate_local_res: Optional[fx.ArithValue],
    d_topk_w_res: Optional[fx.ArithValue],
    epoch: fx.Int64,
):
    f32_vec = fx.T.VectorType.get([_PVEC], fx.T.f32())
    bf16_vec = fx.T.VectorType.get([_PVEC], fx.T.bf16())
    num_vec_chunks = out_features // _PVEC
    lane_id = thread_index % fx.Int32(_WARP)
    warp_id = thread_index // fx.Int32(_WARP)
    global_warp_id = base_pid * fx.Int32(_NUM_WARPS) + warp_id
    num_tokens = buffer_load(num_tokens_res, fx.Int32(rank), vec_width=1, dtype=fx.T.i32())
    token = global_warp_id
    while token < num_tokens:
        if const_expr(signal):
            # Wait each slot's flag == epoch. Loop MUST stay inline (rewriter needs the control flow).
            for j in range_constexpr(topk):
                slot = token * fx.Int32(topk) + fx.Int32(j)
                topk_index = buffer_load(topk_indices_res, slot, vec_width=1, dtype=fx.T.i64())
                if topk_index >= fx.Int64(0):
                    if topk_index < fx.Int64(num_experts):
                        if lane_id == fx.Int32(0):
                            spin_start = read_clock()
                            flag = ld(barrier_base, reduce_bank + slot, scope="agent", dtype=fx.T.i64())
                            while flag != epoch:
                                fx.rocdl.s_sleep(fx.Int32(1))
                                if spin_timed_out(spin_start):
                                    # rank is a compile-time constant, baked into the format string
                                    fx.printf(
                                        "[MEGA rank="
                                        + str(rank)
                                        + " topk_reduce] combine reduce-flag STUCK: "
                                        "GEMM has not written this expert's rows; token={} slot={} expert={} "
                                        "reduce_flag_index={} (seen_flag={} expected_epoch={})\n",
                                        token,
                                        slot,
                                        topk_index,
                                        reduce_bank + slot,
                                        flag,
                                        epoch,
                                    )
                                    spin_start = read_clock()
                                flag = ld(barrier_base, reduce_bank + slot, scope="agent", dtype=fx.T.i64())
            _wait_mem()

        token_row_off = token * fx.Int32(topk) * fx.Int32(out_features)
        vec_idx = lane_id
        while vec_idx < fx.Int32(num_vec_chunks):
            col = vec_idx * fx.Int32(_PVEC)
            topk_vals = []
            for j in range_constexpr(topk):
                row_off = token_row_off + fx.Int32(j * out_features) + col
                topk_vals.append(buffer_load(comb_local_res, row_off, vec_width=_PVEC, dtype=fx.T.bf16()))
            if const_expr(apply_weights):
                weights = [
                    buffer_load(
                        topk_weights_res, token * fx.Int32(topk) + fx.Int32(j), vec_width=1, dtype=fx.T.f32()
                    )
                    for j in range_constexpr(topk)
                ]
            acc = None
            for j in range_constexpr(topk):
                term = fx.arith.extf(f32_vec, topk_vals[j])
                if const_expr(apply_weights):
                    term = fx.arith.mulf(term, _vector.broadcast(f32_vec, weights[j]))
                acc = term if acc is None else fx.arith.addf(acc, term)
            buffer_store(fx.arith.trunc_f(bf16_vec, acc), output_res, token * fx.Int32(out_features) + col)
            vec_idx = vec_idx + fx.Int32(_WARP)
        if const_expr(signal and with_gate):
            for j in range_constexpr(topk):
                slot = token * fx.Int32(topk) + fx.Int32(j)
                topk_index = buffer_load(topk_indices_res, slot, vec_width=1, dtype=fx.T.i64())
                if lane_id == fx.Int32(0):
                    gate_v = buffer_load(gate_local_res, slot, vec_width=1, dtype=fx.T.f32())
                    zero_f = fx.Float32(0.0)
                    v1 = fx.arith.select(topk_index < fx.Int64(num_experts), gate_v, zero_f)
                    d_val = fx.arith.select(topk_index >= fx.Int64(0), v1, zero_f)
                    buffer_store(d_val, d_topk_w_res, slot)
        token = token + total_warps
