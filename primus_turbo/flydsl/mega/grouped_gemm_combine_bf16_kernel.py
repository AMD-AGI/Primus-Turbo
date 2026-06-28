###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Fused grouped BF16 GEMM + cross-rank combine PUSH + intra-node topk reduce (FlyDSL).

Three-role single kernel, partitioned by ``block_index`` so the comm roles get CUs first:
  * role 1 COMBINE PUSH  ``[0, num_combine_cu)``: spin on the local scoreboard, then push each
    finished L2Y row to ``comb_addrs[origin][slot]`` and raise ``barrier_addrs[origin][slot]``.
  * role 2 TOPK REDUCE: warp-per-token, spin on each non-dropped slot's flag, then sum the
    ``topk`` combine rows into ``output[token]`` (token-major). Runs on TWO sources of blocks:
    the optional dedicated region ``[num_combine_cu, +num_reduce_cu)`` (``num_reduce_cu`` blocks,
    default 0), PLUS every EMPTY grouped-GEMM block (``block_m >= real_tiles``). The empty blocks
    have high ``block_index`` -> scheduled after the real GEMM tiles -> they land on the CUs freed
    as the GEMM winds down, so the reduce overlaps the push tail instead of spinning on a few
    dedicated CUs. Both sources grid-stride the full token range; coverage needs >= 1 empty tile
    (guaranteed by ``pool_mult >= 2``). Running both sources together double-reduces (idempotent
    but wasteful), so the default is empty-blocks-only (``num_reduce_cu == 0``).
  * role 3 GROUPED GEMM  ``[gemm_base, ...)`` (``gemm_base = num_combine_cu + num_reduce_cu``):
    one tile per block (NT forward A=act[M,K],B=weight[G,N,K]; NN dgrad B=weight[G,K,N]; TN wgrad
    A=act[K,M]) -> L2Y[M,N], bumping the per-pool-block scoreboard. The combine PUSH reads L2Y
    M-major, so every layout works (it just changes how A/B are read).

With no ``output`` supplied the reduce is off and this degenerates to the two-role GEMM + combine
kernel. K % BLOCK_K == 0; out_features a multiple of 8 (b128 vec)."""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl._mlir.dialects import vector as _vector
from flydsl.compiler.ast_rewriter import ASTRewriter
from flydsl.expr import arith, range_constexpr
from flydsl.expr.buffer_ops import (
    buffer_load,
    buffer_store,
    create_buffer_resource,
    create_buffer_resource_from_addr,
    extract_base_index,
)
from flydsl.expr.typing import AddressSpace, PointerType

# GEMM tile + LDS struct shared with the dispatch path (identical compute).
from primus_turbo.flydsl.mega.gemm_bf16_kernel import (
    _make_shared_storage,
    gemm_bf16_nn_tile,
    gemm_bf16_nt_tile,
    gemm_bf16_tn_tile,
)

# per-tile GEMM closure by layout (NT forward, NN dgrad, TN wgrad); all grouped via b_group_base
_GEMM_TILE = {"nt": gemm_bf16_nt_tile, "nn": gemm_bf16_nn_tile, "tn": gemm_bf16_tn_tile}
from primus_turbo.flydsl.common.tile_spec import _emit_if_then

# scalar/atomic prims over a raw i64 base + element offset (scope-selectable, monotonic).
# scope="agent" = device-wide relaxed (local scoreboard / flag); scope="sys" = system
# (cross-rank flag publish, pairs with the agent-scope consumer load on uncached signal mem).
# _mem_fence = cheap fence (s_waitcnt drain + compiler barrier); pairs with relaxed atomics.
from primus_turbo.flydsl.mega.prims import _mem_fence, atomic_add, ld, st

# single SymLayout struct (two-heap delta tables) names every symmetric sub-buffer;
# the active workspace is fetched by the host wrappers (no-group call -> current buffer).
from primus_turbo.flydsl.mega.sym_layout import SymLayout
from primus_turbo.flydsl.mega.symm_buffer import get_symm_buffer_for_mega_moe
from primus_turbo.flydsl.utils.gemm_helper import make_value_attrs

_WARP = 64  # wavefront (gfx950)
# 8 waves (wave_m x wave_n = 2 x 4) -- the tile block size
_BLOCK_THREADS = 512

# warp-per-rank combine: 8 warps stream 8 peers at once -> saturates XGMI with ~16 CUs.
_PVEC = 8  # bf16 elems/lane/step (16 B = b128)
# 8 warps -> 8 concurrent peer streams per block
_NUM_WARPS = _BLOCK_THREADS // _WARP


# combine PUSH closure: each warp streams its row chunk to the peer comb[origin][slot]
# (+flag if set). Peer addresses are local_base + per-peer DELTA[origin] from the SymLayout's
# two delta tables: comb/barrier live in the SIGNAL heap (signal_delta_res), combine_gate in
# the MAIN heap (main_delta_res). One i64 delta load + add == the old absolute-base table load.
def combine_bf16_tile(
    *,
    thread_index,
    block_m_size,
    out_features,
    comb_records,
    n_slots,
    l2y_resource,
    origin_rank_res,
    origin_slot_res,
    comb_base,
    signal_delta_res,
    barrier_base=None,
    enable_barrier=False,
    grad_gate_res=None,
    gate_base=None,
    main_delta_res=None,
    gate_records=0,
    with_gate=False,
):
    assert block_m_size % _NUM_WARPS == 0, "block_m must be a multiple of num_warps (8)"
    # 64 lanes * 8 = 512 elems/step
    cols_per_step = _WARP * _PVEC
    num_full_chunks = out_features // cols_per_step
    tail_cols = out_features % cols_per_step
    rows_per_warp = block_m_size // _NUM_WARPS
    lane = thread_index % fx.Int32(_WARP)
    warp_id = thread_index // fx.Int32(_WARP)
    oob_index = fx.Int32(n_slots) * fx.Int32(out_features)
    # this warp's contiguous chunk start
    chunk_base = warp_id * fx.Int32(rows_per_warp)

    def push_block(block_m):
        base_row = block_m * fx.Int32(block_m_size) + chunk_base
        # 8 warps each stream 32 contiguous rows
        for j in range(rows_per_warp):
            row = base_row + fx.Int32(j)
            origin = buffer_load(origin_rank_res, row, vec_width=1, dtype=fx.T.i32())
            # skip padding rows entirely
            origin_ok = origin >= fx.Int32(0)

            def _emit_row():
                slot = buffer_load(origin_slot_res, row, vec_width=1, dtype=fx.T.i32())
                comb_addr = comb_base + buffer_load(signal_delta_res, origin, vec_width=1, dtype=fx.T.i64())
                peer = create_buffer_resource_from_addr(comb_addr, num_records_bytes=comb_records)
                slot_base = slot * fx.Int32(out_features)
                row_off = row * fx.Int32(out_features)
                # load all chunks first, then store all -> wide in-flight writes
                chunk_values = []
                for c in range_constexpr(num_full_chunks):
                    col = fx.Int32(c * cols_per_step) + lane * fx.Int32(_PVEC)
                    chunk_values.append(
                        # NOTE: cache_modifier=1 (CPOL glc) = L1-bypass read. Defensive only -- the
                        # write-through C store already puts L2Y in L2; this guards a stale resident
                        # L1 line (write-through doesn't invalidate other CUs' L1) and is free (L2Y
                        # is read-once). Tests pass without it.
                        buffer_load(
                            l2y_resource, row_off + col, vec_width=_PVEC, dtype=fx.T.bf16(), cache_modifier=1
                        )
                    )
                for c in range_constexpr(num_full_chunks):
                    col = fx.Int32(c * cols_per_step) + lane * fx.Int32(_PVEC)
                    buffer_store(chunk_values[c], peer, slot_base + col)
                if tail_cols:  # masked partial last chunk
                    col = fx.Int32(num_full_chunks * cols_per_step) + lane * fx.Int32(_PVEC)
                    in_tail = (lane * fx.Int32(_PVEC)) < fx.Int32(tail_cols)
                    # clamp the load to an in-row column (out-of-tail would fault); store masked to OOB
                    safe_col = arith.select(in_tail, col, fx.Int32(out_features - _PVEC))
                    tail_value = buffer_load(
                        # NOTE: cache_modifier=1 (CPOL glc) = L1-bypass read. Defensive only -- the
                        # write-through C store already puts L2Y in L2; guards a stale resident L1
                        # line and is free (read-once). Tests pass without it.
                        l2y_resource,
                        row_off + safe_col,
                        vec_width=_PVEC,
                        dtype=fx.T.bf16(),
                        cache_modifier=1,
                    )
                    dst = arith.select(in_tail, slot_base + col, oob_index)
                    buffer_store(tail_value, peer, dst)
                if with_gate:
                    # scatter the per-row gate gradient (d_topk_w) to origin[slot] (slot =
                    # token*topk+k); same value/slot across lanes (idempotent, like the flag
                    # store below). 1 f32, ~free vs the hidden-wide push.
                    gate_value = buffer_load(grad_gate_res, row, vec_width=1, dtype=fx.T.f32())
                    gate_addr = gate_base + buffer_load(main_delta_res, origin, vec_width=1, dtype=fx.T.i64())
                    gate_peer = create_buffer_resource_from_addr(gate_addr, num_records_bytes=gate_records)
                    buffer_store(gate_value, gate_peer, slot)
                if enable_barrier:
                    # drain payload stores (vmcnt), then publish the flag via a relaxed atomic
                    # store (monotonic, sys scope -> cross-rank visible; pairs with consumer load).
                    barrier_addr = barrier_base + buffer_load(
                        signal_delta_res, origin, vec_width=1, dtype=fx.T.i64()
                    )
                    _mem_fence()
                    st(barrier_addr, slot, fx.Int32(1), scope="sys")

            _emit_if_then(origin_ok, _emit_row)

    return push_block


# Plain Python (NOT ASTRewriter-transformed): emits the per-token topk accumulation.
# Keeping the compile-time apply_weights branch here avoids a runtime scf.if inside the
# rewritten reduce (the rewriter doesn't export vars assigned in a folded if-branch).
def _reduce_acc(apply_weights, topk, f32_vec, topk_vals, topk_weights_res, token):
    if apply_weights:
        w0 = buffer_load(topk_weights_res, token * fx.Int32(topk), vec_width=1, dtype=fx.T.f32())
        acc = fx.arith.mulf(fx.arith.extf(f32_vec, topk_vals[0]), _vector.broadcast(f32_vec, w0))
        for j in range(topk - 1):
            w = buffer_load(
                topk_weights_res, token * fx.Int32(topk) + fx.Int32(j + 1), vec_width=1, dtype=fx.T.f32()
            )
            acc = fx.arith.addf(
                acc, fx.arith.mulf(fx.arith.extf(f32_vec, topk_vals[j + 1]), _vector.broadcast(f32_vec, w))
            )
    else:
        acc = fx.arith.extf(f32_vec, topk_vals[0])
        for j in range(topk - 1):
            acc = fx.arith.addf(acc, fx.arith.extf(f32_vec, topk_vals[j + 1]))
    return acc


# role 2: warp-per-token reduce of token-major combine [max_tokens, topk, hidden] -> output[token].
# wait_flags=True (fused) spins on each slot's cross-rank flag first; False (standalone, data
# already resident) skips the wait -> pure streaming reduce. apply_weights=True scales each
# topk row by topk_weights[token*topk+k] (routing weight) -> weighted reduce. Built per variant.
def _make_topk_reduce(wait_flags, apply_weights=False, with_gate=False):
    def _topk_reduce(
        thread_index,
        base_pid,
        total_warps,
        topk,
        out_features,
        num_experts,
        rank,
        comb_local_res,
        output_res,
        topk_indices_res,
        num_tokens_res,
        barrier_base,
        topk_weights_res,
        gate_local_res,
        d_topk_w_res,
    ):
        # barrier_base: i64 base (SymLayout signal-heap ptr) or a tensor base index; ld/st
        # scope="agent" read/reset the local per-slot flags on uncached signal memory.
        # total_warps: runtime worker-warp count (const for static roles, dynamic for empty blocks)
        f32_vec = fx.T.VectorType.get([_PVEC], fx.T.f32())
        bf16_vec = fx.T.VectorType.get([_PVEC], fx.T.bf16())
        num_vec_chunks = out_features // _PVEC  # b128 vec chunks per row
        lane = thread_index % fx.Int32(_WARP)
        warp_id = thread_index // fx.Int32(_WARP)
        global_warp_id = base_pid * fx.Int32(_NUM_WARPS) + warp_id
        num_tokens = buffer_load(num_tokens_res, fx.Int32(rank), vec_width=1, dtype=fx.T.i32())

        token = global_warp_id
        while token < num_tokens:
            if wait_flags:
                # lane 0 spins on each non-dropped slot's flag via relaxed atomic load
                # (L1-bypass -> always fresh; no acquire fence). valid iff 0 <= topk_index < num_experts.
                for j in range_constexpr(topk):
                    slot = token * fx.Int32(topk) + fx.Int32(j)
                    topk_index = buffer_load(topk_indices_res, slot, vec_width=1, dtype=fx.T.i64())
                    if topk_index >= fx.Int64(0):
                        if topk_index < fx.Int64(num_experts):
                            if lane == fx.Int32(0):
                                flag = ld(barrier_base, slot, scope="agent")
                                while flag < fx.Int32(0):
                                    fx.rocdl.s_sleep(fx.Int32(1))
                                    flag = ld(barrier_base, slot, scope="agent")
                _mem_fence()  # order payload reads after the flag (s_waitcnt drain)

            token_row_off = token * fx.Int32(topk) * fx.Int32(out_features)  # token's first combine row
            vec_idx = lane
            while vec_idx < fx.Int32(num_vec_chunks):
                col = vec_idx * fx.Int32(_PVEC)
                # issue all topk loads FIRST (independent -> in-flight) then reduce -> hides latency
                topk_vals = []
                for j in range_constexpr(topk):
                    row_off = token_row_off + fx.Int32(j * out_features) + col
                    topk_vals.append(buffer_load(comb_local_res, row_off, vec_width=_PVEC, dtype=fx.T.bf16()))
                # plain-Python helper: compile-time apply_weights branch lives outside the
                # rewritten kernel, so only the result acc crosses back into kernel scope.
                acc = _reduce_acc(apply_weights, topk, f32_vec, topk_vals, topk_weights_res, token)
                buffer_store(
                    fx.arith.trunc_f(bf16_vec, acc), output_res, token * fx.Int32(out_features) + col
                )
                vec_idx = vec_idx + fx.Int32(_WARP)
            if wait_flags:
                # consumed -> reset each non-dropped slot's flag to 0 (NOT -1) so the next
                # combine needs no host barrier_local.zero_. 0 is the no-wait sentinel: a 2nd
                # reducer (num_reduce_cu>0 double-reduce) sees 0 and proceeds (never spins),
                # so this is deadlock-safe unlike a -1 reset. Forward re-arms -1 via the prologue.
                for j in range_constexpr(topk):
                    slot = token * fx.Int32(topk) + fx.Int32(j)
                    topk_index = buffer_load(topk_indices_res, slot, vec_width=1, dtype=fx.T.i64())
                    if topk_index >= fx.Int64(0):
                        if topk_index < fx.Int64(num_experts):
                            if lane == fx.Int32(0):
                                st(barrier_base, slot, fx.Int32(0), scope="agent")
                    if with_gate:
                        # d_topk_w[slot] = gate_local[slot] for valid routes else 0, into a fresh
                        # buffer (folds the host combine_gate * (topk_idx>=0) mul + clone).
                        if lane == fx.Int32(0):
                            gate_v = buffer_load(gate_local_res, slot, vec_width=1, dtype=fx.T.f32())
                            zero_f = fx.Float32(0.0)
                            v1 = fx.arith.select(topk_index < fx.Int64(num_experts), gate_v, zero_f)
                            d_val = fx.arith.select(topk_index >= fx.Int64(0), v1, zero_f)
                            buffer_store(d_val, d_topk_w_res, slot)
            token = token + total_warps

    return ASTRewriter.transform(_topk_reduce)


@functools.lru_cache(maxsize=None)
def _get_topk_reduce(wait_flags, apply_weights, with_gate=False):
    """Build (once per variant) the reduce role: wait_flags toggles the cross-rank flag
    spin (fused vs standalone); apply_weights toggles the weighted reduce; with_gate also
    folds d_topk_w = masked combine_gate into a fresh local output."""
    return _make_topk_reduce(wait_flags, apply_weights, with_gate)


@functools.lru_cache(maxsize=256)
def _compile(
    out_features,
    hidden_size,
    num_max_pool_tokens,
    BLOCK_M,
    BLOCK_N,
    num_combine_cu,
    num_reduce_cu,
    combine_slots,
    topk,
    num_experts,
    rank,
    num_ranks,
    nt_vmcnt=3,
    waves_per_eu=2,
    agpr_alloc=0,
    out_fp16=False,
    layout="nt",
    apply_weights=False,
    with_gate=False,
):
    K = hidden_size
    gemm_tile = _GEMM_TILE[layout]
    assert out_features % BLOCK_N == 0, "out_features must be a multiple of BLOCK_N"
    assert num_max_pool_tokens % BLOCK_M == 0, "num_max_pool_tokens must be a multiple of BLOCK_M"
    assert out_features % _PVEC == 0, "out_features must be a multiple of 8 (bf16 vec)"
    assert topk >= 1, "topk must be >= 1"
    SharedStorage = _make_shared_storage(BLOCK_M, BLOCK_N)
    n_blocks = out_features // BLOCK_N
    worst_case_tiles = num_max_pool_tokens // BLOCK_M
    gemm_grid_blocks = worst_case_tiles * n_blocks  # compile-time empty+real GEMM block count
    comb_records = combine_slots * out_features * 2  # bf16 peer combine-buffer bound (bytes)
    gate_records = combine_slots * 4  # f32 gate slots per peer (backward d_topk_w scatter)
    delta_records = num_ranks * 8  # i64[num_ranks] per-peer delta table (bytes)
    pool_records = num_max_pool_tokens * 4  # i32[num_max_pool_tokens] origin tables (bytes)
    dedicated_reduce_warps = num_reduce_cu * _NUM_WARPS  # const worker-warp count for [combine,reduce)
    gemm_base = num_combine_cu + num_reduce_cu
    topk_reduce = _get_topk_reduce(True, apply_weights, with_gate)

    @flyc.kernel(known_block_size=[_BLOCK_THREADS, 1, 1])
    def grouped_gemm_combine_kernel(
        ACT: fx.Tensor,
        WEIGHTS: fx.Tensor,
        TILE_TO_GROUP: fx.Tensor,
        NUM_TILE_BLOCKS: fx.Tensor,
        OUTPUT: fx.Tensor,
        TOPK_INDICES: fx.Tensor,
        NUM_TOKENS_PER_RANK: fx.Tensor,
        TOPK_WEIGHTS: fx.Tensor,
        GRAD_GATE: fx.Tensor,
        D_TOPK_W: fx.Tensor,
        sym_layout: SymLayout,
        c_n: fx.Int32,
    ):
        thread_index = fx.thread_idx.x
        block_index, _b, _c = fx.block_idx
        combine_cu = fx.Int32(num_combine_cu)
        reduce_cu = fx.Int32(num_reduce_cu)
        lds = fx.SharedAllocator().allocate(SharedStorage).peek()

        # ── all cross-rank/scratch buffers come from the single SymLayout ──
        # local region bases (i64); origin tables + comb/gate locals built as buffer resources
        sb_l2_base = sym_layout.sb_l2_ptr
        comb_base = sym_layout.comb_ptr
        barrier_base = sym_layout.barrier_local_ptr
        # L2Y = the GEMM scratch (l2_token_buffer region): role 3 writes it (C), role 1 reads
        # it (combine PUSH source). bf16 view for the GEMM C; buffer resource for the push read.
        l2y_ptr_ty = PointerType.get(
            elem_ty=fx.BFloat16.ir_type, address_space=AddressSpace.Global, alignment=16
        )
        l2y_tensor = fx.make_view(
            fx.inttoptr(l2y_ptr_ty, sym_layout.l2_token_buffer_ptr),
            fx.make_layout(num_max_pool_tokens * out_features, 1),
        )
        l2y_resource = create_buffer_resource_from_addr(
            sym_layout.l2_token_buffer_ptr, num_records_bytes=num_max_pool_tokens * out_features * 2
        )
        # two per-peer DELTA tables (SIGNAL heap: comb/barrier; MAIN heap: combine_gate)
        signal_delta_res = create_buffer_resource_from_addr(
            sym_layout.signal_offsets_ptr, num_records_bytes=delta_records
        )
        origin_rank_res = create_buffer_resource_from_addr(
            sym_layout.origin_rank_ptr, num_records_bytes=pool_records
        )
        origin_slot_res = create_buffer_resource_from_addr(
            sym_layout.origin_slot_ptr, num_records_bytes=pool_records
        )
        comb_local_res = create_buffer_resource_from_addr(comb_base, num_records_bytes=comb_records)

        group_resource = create_buffer_resource(TILE_TO_GROUP, max_size=True)
        num_tile_blocks_res = create_buffer_resource(NUM_TILE_BLOCKS, max_size=True)
        output_res = create_buffer_resource(OUTPUT, max_size=True)
        topk_indices_res = create_buffer_resource(TOPK_INDICES, max_size=True)
        num_tokens_res = create_buffer_resource(NUM_TOKENS_PER_RANK, max_size=True)
        topk_weights_res = create_buffer_resource(TOPK_WEIGHTS, max_size=True)
        real_tiles = buffer_load(num_tile_blocks_res, fx.Int32(0), vec_width=1, dtype=fx.T.i32())
        # gate scatter (push to peer combine_gate) + local gate read + fresh d_topk_w (reduce fold)
        grad_gate_res = create_buffer_resource(GRAD_GATE, max_size=True) if with_gate else None
        gate_base = sym_layout.combine_gate_ptr if with_gate else None
        main_delta_res = (
            create_buffer_resource_from_addr(sym_layout.offsets_ptr, num_records_bytes=delta_records)
            if with_gate
            else None
        )
        gate_local_res = (
            create_buffer_resource_from_addr(gate_base, num_records_bytes=gate_records) if with_gate else None
        )
        d_topk_w_res = create_buffer_resource(D_TOPK_W, max_size=True) if with_gate else None

        push_block = combine_bf16_tile(
            thread_index=thread_index,
            block_m_size=BLOCK_M,
            out_features=out_features,
            comb_records=comb_records,
            n_slots=combine_slots,
            l2y_resource=l2y_resource,
            origin_rank_res=origin_rank_res,
            origin_slot_res=origin_slot_res,
            comb_base=comb_base,
            signal_delta_res=signal_delta_res,
            barrier_base=barrier_base,
            enable_barrier=True,  # role 1 raises per-slot flags for the reduce
            grad_gate_res=grad_gate_res,
            gate_base=gate_base,
            main_delta_res=main_delta_res,
            gate_records=gate_records,
            with_gate=with_gate,
        )

        # comm roles fill the FRONT [0, gemm_base) -> overlap under the MFMA-bound GEMM at the back
        if block_index < combine_cu:
            # ── role 1: COMBINE PUSH (grid-stride pool blocks) ──
            local_count = (real_tiles - block_index + combine_cu - fx.Int32(1)) // combine_cu
            for tile_iter in range(local_count):
                block_m = block_index + tile_iter * combine_cu
                if thread_index == fx.Int32(0):
                    signal_count = ld(sb_l2_base, block_m, scope="agent")
                    while signal_count < fx.Int32(n_blocks):
                        fx.rocdl.s_sleep(fx.Int32(2))
                        signal_count = ld(sb_l2_base, block_m, scope="agent")
                    # single consumer of this gate -> reset it to 0 for the next launch
                    # (all n_blocks GEMM increments already landed; folds out the host zero)
                    st(sb_l2_base, block_m, fx.Int32(0), scope="agent")
                fx.gpu.barrier()  # broadcast thread-0's poll result to the whole block
                push_block(block_m)
            # drain cross-rank PUSH stores before exit so the host barrier + reduce see landed data
            fx.rocdl.s_waitcnt(0)
        else:
            if block_index < combine_cu + reduce_cu:
                # ── role 2a: DEDICATED TOPK REDUCE (no-op when num_reduce_cu == 0) ──
                topk_reduce(
                    thread_index,
                    block_index - combine_cu,
                    fx.Int32(dedicated_reduce_warps),
                    topk,
                    out_features,
                    num_experts,
                    rank,
                    comb_local_res,
                    output_res,
                    topk_indices_res,
                    num_tokens_res,
                    barrier_base,
                    topk_weights_res,
                    gate_local_res,
                    d_topk_w_res,
                )
            else:
                # ── role 3: GROUPED GEMM (one tile -> L2Y, signal scoreboard) ──
                # naive M-major order is REQUIRED: each block_m's n_blocks tiles complete
                # contiguously AND in numeric block_m order, so the combine role (which
                # grid-strides block_m numerically) can stream-push each block_m mid-kernel.
                # Any reuse swizzle (GROUP_M groups, or XCD remap) reorders block_m completion
                # -> combine stalls on late tiles -> serializes into a 1.5x tail. Proven dead
                # ends; the fused GEMM trades weight-L2 reuse for combine overlap on purpose.
                gemm_tile_index = block_index - fx.Int32(gemm_base)
                block_m = gemm_tile_index // fx.Int32(n_blocks)
                block_n = gemm_tile_index % fx.Int32(n_blocks)
                if block_m < real_tiles:
                    # c_m = num_max_pool_tokens compile-time const folds the epilogue bound (~4% faster)
                    c_m_const = fx.Int32(num_max_pool_tokens)
                    group_index = buffer_load(group_resource, block_m, vec_width=1, dtype=fx.T.i32())
                    group_base = group_index * fx.Int32(K) * c_n
                    gemm_tile(
                        ACT,
                        WEIGHTS,
                        l2y_tensor,
                        c_m_const,
                        c_n,
                        lds,
                        block_m,
                        block_n,
                        K=K,
                        BLOCK_M=BLOCK_M,
                        BLOCK_N=BLOCK_N,
                        out_fp16=out_fp16,
                        nt_vmcnt=nt_vmcnt,
                        b_group_base=group_base,
                        # NOTE: cache_modifier=16 (CPOL sc1) = write-through C store -> L2Y lands in
                        # L2 so the role-1 push (different CU, same kernel) reads fresh, not stale L1.
                        c_cache_modifier=16,
                    )
                    # drain L2Y stores to L2 (coherence point): CDNA L1 is write-through, so
                    # vmcnt(0) means the rows are in L2 -- the role-1 push reads them L1-bypass.
                    fx.rocdl.s_waitcnt(0)
                    fx.gpu.barrier()  # all waves' stores landed before the signal
                    _emit_if_then(
                        thread_index == fx.Int32(0),
                        lambda: atomic_add(sb_l2_base, block_m, fx.Int32(1), scope="agent"),
                    )
                else:
                    # role 2b: empty GEMM tiles reduce on freed CUs (overlaps the push tail)
                    empty_ordinal = gemm_tile_index - real_tiles * fx.Int32(n_blocks)
                    total_empty_warps = (
                        fx.Int32(gemm_grid_blocks) - real_tiles * fx.Int32(n_blocks)
                    ) * fx.Int32(_NUM_WARPS)
                    topk_reduce(
                        thread_index,
                        empty_ordinal,
                        total_empty_warps,
                        topk,
                        out_features,
                        num_experts,
                        rank,
                        comb_local_res,
                        output_res,
                        topk_indices_res,
                        num_tokens_res,
                        barrier_base,
                        topk_weights_res,
                        gate_local_res,
                        d_topk_w_res,
                    )

    @flyc.jit
    def launch(
        ACT,
        WEIGHTS,
        TILE_TO_GROUP,
        NUM_TILE_BLOCKS,
        OUTPUT,
        TOPK_INDICES,
        NUM_TOKENS_PER_RANK,
        TOPK_WEIGHTS,
        GRAD_GATE,
        D_TOPK_W,
        sym_layout,
        c_n: int,
        stream: fx.Stream = fx.Stream(None),
    ):
        grid_size = gemm_base + worst_case_tiles * n_blocks
        grouped_gemm_combine_kernel(
            ACT,
            WEIGHTS,
            TILE_TO_GROUP,
            NUM_TILE_BLOCKS,
            OUTPUT,
            TOPK_INDICES,
            NUM_TOKENS_PER_RANK,
            TOPK_WEIGHTS,
            GRAD_GATE,
            D_TOPK_W,
            sym_layout,
            c_n,
            value_attrs=make_value_attrs(waves_per_eu, agpr_alloc, "512,512"),
        ).launch(grid=(grid_size, 1, 1), block=(_BLOCK_THREADS, 1, 1), stream=stream)

    return launch


# combine-PUSH-only launcher (no GEMM) -- raw combine bandwidth on a pre-filled L2Y pool.
@functools.lru_cache(maxsize=256)
def _compile_combine_only(
    out_features,
    num_max_pool_tokens,
    BLOCK_M,
    num_combine_cu,
    combine_slots,
    num_ranks,
    waves_per_eu=2,
    with_gate=False,
):
    assert num_max_pool_tokens % BLOCK_M == 0, "num_max_pool_tokens must be a multiple of BLOCK_M"
    assert out_features % _PVEC == 0, "out_features must be a multiple of 8 (bf16 vec)"
    comb_records = combine_slots * out_features * 2
    gate_records = combine_slots * 4  # f32 gate slots per peer
    delta_records = num_ranks * 8
    pool_records = num_max_pool_tokens * 4

    @flyc.kernel(known_block_size=[_BLOCK_THREADS, 1, 1])
    def combine_only_k(
        NUM_TILE_BLOCKS: fx.Tensor,
        GRAD_GATE: fx.Tensor,
        sym_layout: SymLayout,
    ):
        thread_index = fx.thread_idx.x
        block_index, _b, _c = fx.block_idx
        combine_cu = fx.Int32(num_combine_cu)
        # L2Y = the SymLayout l2_token_buffer (pre-filled GEMM scratch) -> combine PUSH source
        l2y_resource = create_buffer_resource_from_addr(
            sym_layout.l2_token_buffer_ptr, num_records_bytes=num_max_pool_tokens * out_features * 2
        )
        signal_delta_res = create_buffer_resource_from_addr(
            sym_layout.signal_offsets_ptr, num_records_bytes=delta_records
        )
        origin_rank_res = create_buffer_resource_from_addr(
            sym_layout.origin_rank_ptr, num_records_bytes=pool_records
        )
        origin_slot_res = create_buffer_resource_from_addr(
            sym_layout.origin_slot_ptr, num_records_bytes=pool_records
        )
        num_tile_blocks_res = create_buffer_resource(NUM_TILE_BLOCKS, max_size=True)
        real_tiles = buffer_load(num_tile_blocks_res, fx.Int32(0), vec_width=1, dtype=fx.T.i32())
        grad_gate_res = create_buffer_resource(GRAD_GATE, max_size=True) if with_gate else None
        gate_base = sym_layout.combine_gate_ptr if with_gate else None
        main_delta_res = (
            create_buffer_resource_from_addr(sym_layout.offsets_ptr, num_records_bytes=delta_records)
            if with_gate
            else None
        )

        push_block = combine_bf16_tile(
            thread_index=thread_index,
            block_m_size=BLOCK_M,
            out_features=out_features,
            comb_records=comb_records,
            n_slots=combine_slots,
            l2y_resource=l2y_resource,
            origin_rank_res=origin_rank_res,
            origin_slot_res=origin_slot_res,
            comb_base=sym_layout.comb_ptr,
            signal_delta_res=signal_delta_res,
            grad_gate_res=grad_gate_res,
            gate_base=gate_base,
            main_delta_res=main_delta_res,
            gate_records=gate_records,
            with_gate=with_gate,
        )

        local_count = (real_tiles - block_index + combine_cu - fx.Int32(1)) // combine_cu
        for tile_iter in range(local_count):
            push_block(block_index + tile_iter * combine_cu)

    @flyc.jit
    def launch(
        NUM_TILE_BLOCKS,
        GRAD_GATE,
        sym_layout,
        stream: fx.Stream = fx.Stream(None),
    ):
        combine_only_k(
            NUM_TILE_BLOCKS,
            GRAD_GATE,
            sym_layout,
            value_attrs=make_value_attrs(waves_per_eu, 0, "512,512"),
        ).launch(grid=(num_combine_cu, 1, 1), block=(_BLOCK_THREADS, 1, 1), stream=stream)

    return launch


# topk-reduce-only launcher (no GEMM/combine) -- reduces a pre-filled token-major combine buffer.
@functools.lru_cache(maxsize=256)
def _compile_reduce_only(
    out_features, combine_slots, num_reduce_cu, topk, num_experts, rank, waves_per_eu=2, apply_weights=False
):
    assert out_features % _PVEC == 0, "out_features must be a multiple of 8 (bf16 vec)"
    assert topk >= 1, "topk must be >= 1"
    topk_reduce = _get_topk_reduce(False, apply_weights)

    @flyc.kernel(known_block_size=[_BLOCK_THREADS, 1, 1])
    def reduce_only_k(
        OUTPUT: fx.Tensor,
        COMB_LOCAL: fx.Tensor,
        BARRIER_LOCAL: fx.Tensor,
        TOPK_INDICES: fx.Tensor,
        NUM_TOKENS_PER_RANK: fx.Tensor,
        TOPK_WEIGHTS: fx.Tensor,
    ):
        thread_index = fx.thread_idx.x
        block_index, _b, _c = fx.block_idx
        comb_local_res = create_buffer_resource(COMB_LOCAL, max_size=True)
        output_res = create_buffer_resource(OUTPUT, max_size=True)
        topk_indices_res = create_buffer_resource(TOPK_INDICES, max_size=True)
        num_tokens_res = create_buffer_resource(NUM_TOKENS_PER_RANK, max_size=True)
        topk_weights_res = create_buffer_resource(TOPK_WEIGHTS, max_size=True)
        # tensor base index -> prims.ld/st (agent scope) read/reset the local per-slot flags
        barrier_base = extract_base_index(BARRIER_LOCAL, address_space=1)
        topk_reduce(
            thread_index,
            block_index,
            fx.Int32(num_reduce_cu * _NUM_WARPS),
            topk,
            out_features,
            num_experts,
            rank,
            comb_local_res,
            output_res,
            topk_indices_res,
            num_tokens_res,
            barrier_base,
            topk_weights_res,
            None,  # gate_local_res (with_gate=False -> unused)
            None,  # d_topk_w_res
        )

    @flyc.jit
    def launch(
        OUTPUT,
        COMB_LOCAL,
        BARRIER_LOCAL,
        TOPK_INDICES,
        NUM_TOKENS_PER_RANK,
        TOPK_WEIGHTS,
        stream: fx.Stream = fx.Stream(None),
    ):
        reduce_only_k(
            OUTPUT,
            COMB_LOCAL,
            BARRIER_LOCAL,
            TOPK_INDICES,
            NUM_TOKENS_PER_RANK,
            TOPK_WEIGHTS,
            value_attrs=make_value_attrs(waves_per_eu, 0, "512,512"),
        ).launch(grid=(num_reduce_cu, 1, 1), block=(_BLOCK_THREADS, 1, 1), stream=stream)

    return launch


# autotune candidates (num_combine_cu, nt_vmcnt, agpr_alloc, waves_per_eu); sweep the low end.
_COMBINE_CANDIDATES = [
    (16, 3, 0, 2),
    (24, 3, 0, 2),
    (32, 3, 0, 2),
    (48, 3, 0, 2),
    (64, 3, 0, 2),
    (96, 3, 0, 2),
    (128, 3, 0, 2),
    (192, 3, 0, 2),
    (256, 3, 0, 2),
    (32, 4, 0, 2),
    (32, 2, 0, 2),
    (64, 4, 0, 2),
    (32, 3, 0, 1),
]
_COMBINE_AUTOTUNE_CACHE: dict = {}


# host-side wrappers
def grouped_gemm_combine_bf16(
    x,
    l2_weights,
    handle,
    group,
    *,
    topk_indices,
    topk_weights=None,
    grad_gate=None,
    layout="nt",
    BM=256,
    BN=256,
    num_combine_cu=64,  # best for fused 3-role (e2e sweep 64<48<96)
    num_reduce_cu=0,  # best: reduce on empty GEMM blocks, no dedicated region
    autotune=False,
):
    """Fused grouped BF16 GEMM + combine PUSH + topk reduce, three-role.

    Every buffer except the operands (``x`` activation, ``l2_weights``) comes from the
    active ``get_symm_buffer_for_mega_moe()`` (the workspace the caller built
    for this ``group``), handed to the kernel as a single ``SymLayout``. In particular the
    GEMM output / combine PUSH source is the SymLayout ``l2_token_buffer`` (no ``l2y`` arg),
    and the per-token reduce result ``output`` [num_tokens, N] is allocated here and returned.

    ``layout`` selects the GEMM tile: NT (forward) ``x`` [M,K] / ``l2_weights`` [G,N,K];
    NN (dgrad) ``x`` [M,K] / ``l2_weights`` [G,K,N]; TN (wgrad) ``x`` [K,M] / [G,K,N]. The
    GEMM writes ``l2_token_buffer`` [M,N] M-major (N == hidden); the SymLayout's
    ``origin_rank``/``origin_slot`` route finished rows into the peer combine buffer. The L2
    scoreboard (``sb_l2``) is read/reset in-kernel (folded host zero).

    Role 1 (combine push) runs on ``num_combine_cu`` front blocks (default 64). Role 2 (topk
    reduce) reads the token-major combine buffer, gated per slot by the SymLayout
    ``barrier_local`` flags role 1 raises cross-rank, sums the ``topk`` rows of each of this
    rank's tokens into a freshly-allocated ``output`` [num_tokens, N]. The reduce runs on the
    EMPTY grouped-GEMM blocks (freed CUs as the GEMM winds down) plus an optional dedicated
    region of ``num_reduce_cu`` blocks (default 0; non-zero double-reduces). ``topk_indices``
    [num_tokens*topk] int32 (< 0 == dropped) drives the per-token loop; ``topk_weights``
    [num_tokens*topk] f32 (optional) scales each topk row -> weighted reduce (else unweighted).

    ``grad_gate`` [num_max_pool_tokens] f32 (given): role 1 also scatters the per-row gate gradient
    to the peer ``combine_gate`` (backward d_topk_w); the reduce folds it into a fresh masked
    ``d_topk_w`` [combine_slots] f32.

    Returns ``(output, d_topk_w)``: ``output`` [num_tokens, N] bf16 is the reduce result;
    ``d_topk_w`` is the masked gate-grad fold (``None`` unless ``grad_gate`` given).
    ``autotune=True`` sweeps the num_combine_cu candidates per shape."""
    assert layout in ("nt", "nn", "tn"), f"unknown layout {layout}"
    assert x.dtype == torch.bfloat16 and l2_weights.dtype == torch.bfloat16
    assert topk_indices is not None, "topk reduce needs topk_indices"
    # dispatch prologue tuple: slot [7] = expert id per BM tile
    tile_to_expert = handle[7]
    # active symmetric workspace (built earlier by get_symm_buffer_for_mega_moe for `group`);
    # the single SymLayout names every symmetric sub-buffer (two-heap delta tables)
    symm = get_symm_buffer_for_mega_moe()
    sym_layout = symm.make_sym_layout()
    if layout == "tn":  # A is [K, M] (K-major)
        hidden_size, num_max_pool_tokens = x.shape
    else:  # A is [M, K]
        num_max_pool_tokens, hidden_size = x.shape
    if layout == "nt":  # weight [G, N, K]
        G, N, K = l2_weights.shape
    else:  # NN / TN: weight [G, K, N]
        G, K, N = l2_weights.shape
    assert K == hidden_size, f"weight K={K} != activation K={hidden_size}"
    out_features = N
    c_n = out_features
    # the GEMM output IS the SymLayout l2_token_buffer [num_max_pool_tokens, hidden] -> N == hidden
    assert out_features == int(
        sym_layout.hidden
    ), f"out_features {out_features} != SymLayout hidden {int(sym_layout.hidden)}"
    assert num_max_pool_tokens == int(
        sym_layout.num_max_pool_tokens
    ), "x rows must match SymLayout pool capacity"
    if num_combine_cu is None:
        num_combine_cu = 64  # best for fused 3-role (bench fused3 ~2.9ms; e2e sweep 64<48<96)

    device = x.device
    # dims from the SymLayout (single source of truth for the symmetric buffers)
    combine_slots = int(sym_layout.combine_slots)
    num_ranks = int(sym_layout.num_ranks)
    rank = int(sym_layout.rank_idx)
    topk = int(sym_layout.num_topk)
    num_experts = int(sym_layout.num_experts)
    assert topk >= 1 and num_experts > 0, "topk reduce needs topk>=1 and num_experts>0"
    num_tile_blocks = symm.meta_scalars[1:2]  # device real-tile count
    dummy = num_tile_blocks  # placeholder for unused optional kernel tensors

    apply_weights = topk_weights is not None  # weighted (fwd) vs unweighted (bwd dx) reduce
    with_gate = grad_gate is not None

    act_flat = x.contiguous().view(-1)
    if layout == "nt":  # weight [G, N, K]
        weight_flat = l2_weights.reshape(G * N, K).contiguous().view(-1)
    else:  # NN / TN: weight [G, K, N]
        weight_flat = l2_weights.reshape(G * K, N).contiguous().view(-1)
    # reduce result: allocated here [num_tokens, N] and returned
    num_tokens = int(symm.num_tokens)
    output = torch.empty(num_tokens, out_features, dtype=torch.bfloat16, device=device)
    output_d = output.view(-1)
    topk_indices_d = topk_indices.contiguous().view(-1)
    num_tokens_d = symm.num_tokens_per_rank
    topk_weights_d = topk_weights.contiguous().view(-1) if apply_weights else dummy
    grad_gate_d = grad_gate.contiguous().view(-1) if with_gate else dummy
    # d_topk_w fold (backward): the reduce reads the scattered combine_gate and writes a fresh
    # masked d_topk_w [combine_slots] f32 (replaces the host combine_gate*(topk_idx>=0) mul +
    # clone). Off (dummy) when no gate.
    d_topk_w = torch.empty(combine_slots, dtype=torch.float32, device=device) if with_gate else None
    d_topk_w_d = d_topk_w if with_gate else dummy

    pos_args = (
        act_flat,
        weight_flat,
        tile_to_expert,
        num_tile_blocks,
        output_d,
        topk_indices_d,
        num_tokens_d,
        topk_weights_d,
        grad_gate_d,
        d_topk_w_d,
        sym_layout,
        c_n,
    )
    finite_view = output  # finite-check view for autotune

    if autotune:
        key = (
            out_features,
            hidden_size,
            num_max_pool_tokens,
            BM,
            BN,
            int(combine_slots),
            int(num_reduce_cu),
            int(topk),
            int(num_experts),
            int(rank),
            int(num_ranks),
            layout,
            bool(apply_weights),
            bool(with_gate),
        )
        cached = _COMBINE_AUTOTUNE_CACHE.get(key)
        if cached is None:
            cached = _autotune(
                pos_args,
                finite_view,
                out_features,
                hidden_size,
                num_max_pool_tokens,
                BM,
                BN,
                int(combine_slots),
                int(num_reduce_cu),
                int(topk),
                int(num_experts),
                int(rank),
                int(num_ranks),
                symm.sb_l2,
                layout,
                apply_weights,
                with_gate,
            )
            _COMBINE_AUTOTUNE_CACHE[key] = cached
        launch, _cfg = cached
    else:
        launch = _compile(
            out_features,
            hidden_size,
            num_max_pool_tokens,
            BM,
            BN,
            int(num_combine_cu),
            int(num_reduce_cu),
            int(combine_slots),
            int(topk),
            int(num_experts),
            int(rank),
            int(num_ranks),
            layout=layout,
            apply_weights=apply_weights,
            with_gate=with_gate,
        )
    launch(*pos_args, stream=torch.cuda.current_stream())
    # output = per-token reduce result; d_topk_w = masked gate-grad fold (None unless grad_gate)
    return output, d_topk_w


def _autotune(
    pos_args,
    finite_view,
    out_features,
    hidden_size,
    num_max_pool_tokens,
    BM,
    BN,
    combine_slots,
    num_reduce_cu,
    topk,
    num_experts,
    rank,
    num_ranks,
    sb_l2,
    layout="nt",
    apply_weights=False,
    with_gate=False,
):
    """Bench the num_combine_cu candidates with a per-iter scoreboard reset; return (launch, cfg)."""
    stream = torch.cuda.current_stream()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    best_us, best = float("inf"), None
    for num_combine_cu, nt_vmcnt, agpr, waves in _COMBINE_CANDIDATES:
        try:
            launch = _compile(
                out_features,
                hidden_size,
                num_max_pool_tokens,
                BM,
                BN,
                int(num_combine_cu),
                int(num_reduce_cu),
                combine_slots,
                int(topk),
                int(num_experts),
                int(rank),
                int(num_ranks),
                int(nt_vmcnt),
                int(waves),
                int(agpr),
                layout=layout,
                apply_weights=apply_weights,
                with_gate=with_gate,
            )
            launch(*pos_args, stream=stream)
            torch.cuda.synchronize()
            if not torch.isfinite(finite_view.view(-1)[:1024].float()).all().item():
                continue
            for _ in range(2):

                launch(*pos_args, stream=stream)
            torch.cuda.synchronize()
            us_total = 0.0
            for _ in range(20):

                start_event.record()
                launch(*pos_args, stream=stream)
                end_event.record()
                torch.cuda.synchronize()
                us_total += start_event.elapsed_time(end_event) * 1000.0
            us = us_total / 20
            if us < best_us:
                best_us, best = us, (launch, (num_combine_cu, nt_vmcnt, agpr, waves))
        except Exception:
            continue
    if best is None:
        raise RuntimeError("grouped_gemm_combine_bf16 autotune found no working cfg")
    return best


def combine_only(
    group,
    *,
    BM=256,
    num_combine_cu=None,
    grad_gate=None,
):
    """Combine-PUSH only (no GEMM) -- pushes the pre-filled SymLayout ``l2_token_buffer``
    rows to the peer combine buffers. For measuring raw combine bandwidth. The whole
    symmetric workspace (L2Y source, origin tables, combine buffer, peer delta tables,
    optional gate-grad buffer) is fetched from the active ``get_symm_buffer_for_mega_moe()``.

    ``grad_gate`` [num_max_pool_tokens] f32 (given): also scatter the per-row gate gradient to
    the peer ``combine_gate`` (backward d_topk_w); one f32/row by lane 0, negligible."""
    symm = get_symm_buffer_for_mega_moe()
    sym_layout = symm.make_sym_layout()
    out_features = int(sym_layout.hidden)
    num_max_pool_tokens = int(sym_layout.num_max_pool_tokens)
    if num_combine_cu is None:
        num_combine_cu = num_max_pool_tokens // BM
    num_tile_blocks = symm.meta_scalars[1:2]
    with_gate = grad_gate is not None
    launch = _compile_combine_only(
        out_features,
        num_max_pool_tokens,
        int(BM),
        int(num_combine_cu),
        int(sym_layout.combine_slots),
        int(sym_layout.num_ranks),
        with_gate=with_gate,
    )
    # dummy satisfies the fixed kernel signature when gate is off (gated by const_expr)
    grad_gate_arg = grad_gate.contiguous().view(-1) if with_gate else num_tile_blocks
    launch(
        num_tile_blocks,
        grad_gate_arg,
        sym_layout,
        stream=torch.cuda.current_stream(),
    )
    return symm.l2_token_buffer


def topk_reduce_only(
    output,
    comb_local,
    barrier_local,
    topk_indices,
    num_tokens_per_rank,
    combine_slots,
    *,
    topk,
    num_experts,
    rank=0,
    num_reduce_cu=32,
    topk_weights=None,
):
    """Topk-reduce only (no GEMM, no combine) -- sums the pre-filled token-major combine
    buffer ``comb_local`` [combine_slots, N] into ``output`` [num_tokens, N]. ``barrier_local``
    [combine_slots] must already mark every needed slot ready (>= 0). ``topk_weights``
    [num_tokens*topk] f32 (optional) scales each topk row. For unit testing role 2."""
    assert topk >= 1 and num_experts > 0, "topk reduce needs topk>=1 and num_experts>0"
    out_features = output.size(1)
    apply_weights = topk_weights is not None
    launch = _compile_reduce_only(
        out_features,
        int(combine_slots),
        int(num_reduce_cu),
        int(topk),
        int(num_experts),
        int(rank),
        apply_weights=apply_weights,
    )
    topk_weights_d = topk_weights.contiguous().view(-1) if apply_weights else num_tokens_per_rank
    launch(
        output.view(-1),
        comb_local.contiguous().view(-1),
        barrier_local,
        topk_indices.contiguous().view(-1),
        num_tokens_per_rank,
        topk_weights_d,
        stream=torch.cuda.current_stream(),
    )
    return output
