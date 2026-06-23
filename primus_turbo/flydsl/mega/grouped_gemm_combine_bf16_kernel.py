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
from flydsl._mlir.dialects import llvm as _llvm
from flydsl._mlir.dialects import vector as _vector
from flydsl.compiler.ast_rewriter import ASTRewriter
from flydsl.expr import arith, range_constexpr
from flydsl.expr.buffer_ops import (
    _unwrap_value,
    buffer_load,
    buffer_store,
    create_buffer_resource,
    create_buffer_resource_from_addr,
)

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
from primus_turbo.flydsl.mega.mega_group_tile_spec import (
    _elem_ptr_i32,
    _elem_ptr_i32_from_addr,
)

# cheap fence = s_waitcnt drain + compiler barrier; pairs with relaxed atomics (no acquire/release).
from primus_turbo.flydsl.mega.mega_moe_prologue_kernel import _mem_fence
from primus_turbo.flydsl.utils.gemm_helper import make_value_attrs

_WARP = 64  # wavefront (gfx950)
# 8 waves (wave_m x wave_n = 2 x 4) -- the tile block size
_BLOCK_THREADS = 512

# warp-per-rank combine: 8 warps stream 8 peers at once -> saturates XGMI with ~16 CUs.
_PVEC = 8  # bf16 elems/lane/step (16 B = b128)
# 8 warps -> 8 concurrent peer streams per block
_NUM_WARPS = _BLOCK_THREADS // _WARP

_I4 = 4
_ORD = _llvm.AtomicOrdering
# device-scope scoreboard atomics (the coherence domain, not an ordering)
_SCOPE = "agent"


def _atomic_add_relaxed(tensor, idx, val):
    """Relaxed (agent-scope) atomic int32 add into ``tensor[idx]`` (scoreboard done-count)."""
    ptr = _elem_ptr_i32(tensor, idx)
    _llvm.atomicrmw(
        _llvm.AtomicBinOp.add, ptr, _unwrap_value(val), _ORD.monotonic, syncscope=_SCOPE, alignment=_I4
    )


def _ld_relaxed(tensor, idx):
    """Relaxed (agent-scope) atomic int32 load of ``tensor[idx]`` for spin polling."""
    ptr = _elem_ptr_i32(tensor, idx)
    op = _llvm.LoadOp(fx.T.i32(), ptr, ordering=_ORD.monotonic, syncscope=_SCOPE, alignment=_I4)
    return fx.arith.ArithValue(op.result, signed=True)


def _st_relaxed_addr(addr_i64, idx, val):
    """Relaxed system-scope atomic int32 store to ``(addr_i64)[idx]`` -- cross-rank flag
    publish (monotonic, sys scope; pairs with the consumer's relaxed-atomic load)."""
    ptr = _elem_ptr_i32_from_addr(addr_i64, idx)
    _llvm.StoreOp(
        _unwrap_value(val), ptr, ordering=_ORD.monotonic, syncscope=None, alignment=_I4
    )  # syncscope=None == system scope


def _st_relaxed(tensor, idx, val):
    """Relaxed (agent-scope) atomic int32 store to ``tensor[idx]`` (local scoreboard reset)."""
    ptr = _elem_ptr_i32(tensor, idx)
    _llvm.StoreOp(_unwrap_value(val), ptr, ordering=_ORD.monotonic, syncscope=_SCOPE, alignment=_I4)


# combine PUSH closure: each warp streams its row chunk to comb_addrs[origin][slot] (+flag if set).
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
    comb_addr_res,
    barrier_addr_res=None,
    enable_barrier=False,
    grad_gate_res=None,
    gate_addr_res=None,
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
                comb_addr = buffer_load(comb_addr_res, origin, vec_width=1, dtype=fx.T.i64())
                peer = create_buffer_resource_from_addr(comb_addr, num_records_bytes=comb_records)
                slot_base = slot * fx.Int32(out_features)
                row_off = row * fx.Int32(out_features)
                # load all chunks first, then store all -> wide in-flight writes
                chunk_values = []
                for c in range_constexpr(num_full_chunks):
                    col = fx.Int32(c * cols_per_step) + lane * fx.Int32(_PVEC)
                    chunk_values.append(
                        buffer_load(l2y_resource, row_off + col, vec_width=_PVEC, dtype=fx.T.bf16())
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
                        l2y_resource, row_off + safe_col, vec_width=_PVEC, dtype=fx.T.bf16()
                    )
                    dst = arith.select(in_tail, slot_base + col, oob_index)
                    buffer_store(tail_value, peer, dst)
                if with_gate:
                    # scatter the per-row gate gradient (d_topk_w) to origin[slot] (slot =
                    # token*topk+k); same value/slot across lanes (idempotent, like the flag
                    # store below). 1 f32, ~free vs the hidden-wide push.
                    gate_value = buffer_load(grad_gate_res, row, vec_width=1, dtype=fx.T.f32())
                    gate_addr = buffer_load(gate_addr_res, origin, vec_width=1, dtype=fx.T.i64())
                    gate_peer = create_buffer_resource_from_addr(gate_addr, num_records_bytes=gate_records)
                    buffer_store(gate_value, gate_peer, slot)
                if enable_barrier:
                    # drain payload stores (vmcnt), then publish the flag via a relaxed atomic
                    # store (monotonic, sys scope -> cross-rank visible; pairs with consumer load).
                    barrier_addr = buffer_load(barrier_addr_res, origin, vec_width=1, dtype=fx.T.i64())
                    _mem_fence()
                    _st_relaxed_addr(barrier_addr, slot, fx.Int32(1))

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
        barrier_local,
        topk_weights_res,
        gate_local_res,
        d_topk_w_res,
    ):
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
                    topk_index = buffer_load(topk_indices_res, slot, vec_width=1, dtype=fx.T.i32())
                    if topk_index >= fx.Int32(0):
                        if topk_index < fx.Int32(num_experts):
                            if lane == fx.Int32(0):
                                flag = _ld_relaxed(barrier_local, slot)
                                while flag < fx.Int32(0):
                                    fx.rocdl.s_sleep(fx.Int32(1))
                                    flag = _ld_relaxed(barrier_local, slot)
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
                    topk_index = buffer_load(topk_indices_res, slot, vec_width=1, dtype=fx.T.i32())
                    if topk_index >= fx.Int32(0):
                        if topk_index < fx.Int32(num_experts):
                            if lane == fx.Int32(0):
                                _st_relaxed(barrier_local, slot, fx.Int32(0))
                    if with_gate:
                        # d_topk_w[slot] = gate_local[slot] for valid routes else 0, into a fresh
                        # buffer (folds the host combine_gate * (topk_idx>=0) mul + clone).
                        if lane == fx.Int32(0):
                            gate_v = buffer_load(gate_local_res, slot, vec_width=1, dtype=fx.T.f32())
                            zero_f = fx.Float32(0.0)
                            v1 = fx.arith.select(topk_index < fx.Int32(num_experts), gate_v, zero_f)
                            d_val = fx.arith.select(topk_index >= fx.Int32(0), v1, zero_f)
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
    pool_capacity,
    BLOCK_M,
    BLOCK_N,
    num_combine_cu,
    num_reduce_cu,
    combine_slots,
    topk,
    num_experts,
    rank,
    nt_vmcnt=3,
    waves_per_eu=2,
    agpr_alloc=0,
    out_fp16=False,
    layout="nt",
    apply_weights=False,
    reduce_active=False,
    with_gate=False,
):
    K = hidden_size
    gemm_tile = _GEMM_TILE[layout]
    assert out_features % BLOCK_N == 0, "out_features must be a multiple of BLOCK_N"
    assert pool_capacity % BLOCK_M == 0, "pool_capacity must be a multiple of BLOCK_M"
    assert out_features % _PVEC == 0, "out_features must be a multiple of 8 (bf16 vec)"
    assert topk >= 1, "topk must be >= 1"
    SharedStorage = _make_shared_storage(BLOCK_M, BLOCK_N)
    n_blocks = out_features // BLOCK_N
    worst_case_tiles = pool_capacity // BLOCK_M
    gemm_grid_blocks = worst_case_tiles * n_blocks  # compile-time empty+real GEMM block count
    comb_records = combine_slots * out_features * 2  # bf16 peer combine-buffer bound (bytes)
    gate_records = combine_slots * 4  # f32 gate slots per peer (backward d_topk_w scatter)
    # role 1 raises per-slot flags whenever any reducer (dedicated or empty-block) runs
    with_barrier = reduce_active
    dedicated_reduce_warps = num_reduce_cu * _NUM_WARPS  # const worker-warp count for [combine,reduce)
    gemm_base = num_combine_cu + num_reduce_cu
    topk_reduce = _get_topk_reduce(True, apply_weights, with_gate)

    @flyc.kernel(known_block_size=[_BLOCK_THREADS, 1, 1])
    def combine_grouped(
        ACT: fx.Tensor,
        WEIGHTS: fx.Tensor,
        L2Y: fx.Tensor,
        TILE_TO_GROUP: fx.Tensor,
        SB_L2: fx.Tensor,
        ORIGIN_RANK: fx.Tensor,
        ORIGIN_SLOT: fx.Tensor,
        COMB_ADDRS: fx.Tensor,
        BARRIER_ADDRS: fx.Tensor,
        NUM_TILE_BLOCKS: fx.Tensor,
        OUTPUT: fx.Tensor,
        COMB_LOCAL: fx.Tensor,
        BARRIER_LOCAL: fx.Tensor,
        TOPK_INDICES: fx.Tensor,
        NUM_TOKENS_PER_RANK: fx.Tensor,
        TOPK_WEIGHTS: fx.Tensor,
        GRAD_GATE: fx.Tensor,
        GATE_ADDRS: fx.Tensor,
        GATE_LOCAL: fx.Tensor,
        D_TOPK_W: fx.Tensor,
        c_n: fx.Int32,
    ):
        thread_index = fx.thread_idx.x
        block_index, _b, _c = fx.block_idx
        combine_cu = fx.Int32(num_combine_cu)
        reduce_cu = fx.Int32(num_reduce_cu)
        lds = fx.SharedAllocator().allocate(SharedStorage).peek()

        group_resource = create_buffer_resource(TILE_TO_GROUP, max_size=True)
        num_tile_blocks_res = create_buffer_resource(NUM_TILE_BLOCKS, max_size=True)
        l2y_resource = create_buffer_resource(L2Y, max_size=True)
        origin_rank_res = create_buffer_resource(ORIGIN_RANK, max_size=True)
        origin_slot_res = create_buffer_resource(ORIGIN_SLOT, max_size=True)
        comb_addr_res = create_buffer_resource(COMB_ADDRS, max_size=True)
        barrier_addr_res = create_buffer_resource(BARRIER_ADDRS, max_size=True)
        comb_local_res = create_buffer_resource(COMB_LOCAL, max_size=True)
        output_res = create_buffer_resource(OUTPUT, max_size=True)
        topk_indices_res = create_buffer_resource(TOPK_INDICES, max_size=True)
        num_tokens_res = create_buffer_resource(NUM_TOKENS_PER_RANK, max_size=True)
        topk_weights_res = create_buffer_resource(TOPK_WEIGHTS, max_size=True)
        real_tiles = buffer_load(num_tile_blocks_res, fx.Int32(0), vec_width=1, dtype=fx.T.i32())
        grad_gate_res = create_buffer_resource(GRAD_GATE, max_size=True) if with_gate else None
        gate_addr_res = create_buffer_resource(GATE_ADDRS, max_size=True) if with_gate else None
        # local gate buffer (read) + fresh d_topk_w (write) for the reduce's masked-gate fold
        gate_local_res = create_buffer_resource(GATE_LOCAL, max_size=True) if with_gate else None
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
            comb_addr_res=comb_addr_res,
            barrier_addr_res=barrier_addr_res,
            enable_barrier=with_barrier,
            grad_gate_res=grad_gate_res,
            gate_addr_res=gate_addr_res,
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
                    signal_count = _ld_relaxed(SB_L2, block_m)
                    while signal_count < fx.Int32(n_blocks):
                        fx.rocdl.s_sleep(fx.Int32(2))
                        signal_count = _ld_relaxed(SB_L2, block_m)
                    # single consumer of this gate -> reset it to 0 for the next launch
                    # (all n_blocks GEMM increments already landed; folds out the host zero)
                    _st_relaxed(SB_L2, block_m, fx.Int32(0))
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
                    BARRIER_LOCAL,
                    topk_weights_res,
                    gate_local_res,
                    d_topk_w_res,
                )
            else:
                # ── role 3: GROUPED GEMM (one tile -> L2Y, signal scoreboard) ──
                gemm_tile_index = block_index - fx.Int32(gemm_base)
                block_m = gemm_tile_index // fx.Int32(n_blocks)
                block_n = gemm_tile_index % fx.Int32(n_blocks)
                if block_m < real_tiles:
                    # c_m = pool_capacity compile-time const folds the epilogue bound (~4% faster)
                    c_m_const = fx.Int32(pool_capacity)
                    group_index = buffer_load(group_resource, block_m, vec_width=1, dtype=fx.T.i32())
                    group_base = group_index * fx.Int32(K) * c_n
                    gemm_tile(
                        ACT,
                        WEIGHTS,
                        L2Y,
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
                    )
                    # drain L2Y stores to L2 (coherence point)
                    fx.rocdl.s_waitcnt(0)
                    fx.gpu.barrier()  # all waves' stores landed before the signal
                    _emit_if_then(
                        thread_index == fx.Int32(0), lambda: _atomic_add_relaxed(SB_L2, block_m, fx.Int32(1))
                    )
                else:
                    # role 2b: empty GEMM tiles reduce on freed CUs (overlaps the push tail)
                    if reduce_active:
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
                            BARRIER_LOCAL,
                            topk_weights_res,
                            gate_local_res,
                            d_topk_w_res,
                        )

    @flyc.jit
    def launch(
        ACT,
        WEIGHTS,
        L2Y,
        TILE_TO_GROUP,
        SB_L2,
        ORIGIN_RANK,
        ORIGIN_SLOT,
        COMB_ADDRS,
        BARRIER_ADDRS,
        NUM_TILE_BLOCKS,
        OUTPUT,
        COMB_LOCAL,
        BARRIER_LOCAL,
        TOPK_INDICES,
        NUM_TOKENS_PER_RANK,
        TOPK_WEIGHTS,
        GRAD_GATE,
        GATE_ADDRS,
        GATE_LOCAL,
        D_TOPK_W,
        c_n: int,
        stream: fx.Stream = fx.Stream(None),
    ):
        grid_size = gemm_base + worst_case_tiles * n_blocks
        combine_grouped(
            ACT,
            WEIGHTS,
            L2Y,
            TILE_TO_GROUP,
            SB_L2,
            ORIGIN_RANK,
            ORIGIN_SLOT,
            COMB_ADDRS,
            BARRIER_ADDRS,
            NUM_TILE_BLOCKS,
            OUTPUT,
            COMB_LOCAL,
            BARRIER_LOCAL,
            TOPK_INDICES,
            NUM_TOKENS_PER_RANK,
            TOPK_WEIGHTS,
            GRAD_GATE,
            GATE_ADDRS,
            GATE_LOCAL,
            D_TOPK_W,
            c_n,
            value_attrs=make_value_attrs(waves_per_eu, agpr_alloc, "512,512"),
        ).launch(grid=(grid_size, 1, 1), block=(_BLOCK_THREADS, 1, 1), stream=stream)

    return launch


# combine-PUSH-only launcher (no GEMM) -- raw combine bandwidth on a pre-filled L2Y pool.
@functools.lru_cache(maxsize=256)
def _compile_combine_only(
    out_features, pool_capacity, BLOCK_M, num_combine_cu, combine_slots, waves_per_eu=2, with_gate=False
):
    assert pool_capacity % BLOCK_M == 0, "pool_capacity must be a multiple of BLOCK_M"
    assert out_features % _PVEC == 0, "out_features must be a multiple of 8 (bf16 vec)"
    comb_records = combine_slots * out_features * 2
    gate_records = combine_slots * 4  # f32 gate slots per peer

    @flyc.kernel(known_block_size=[_BLOCK_THREADS, 1, 1])
    def combine_only_k(
        L2Y: fx.Tensor,
        ORIGIN_RANK: fx.Tensor,
        ORIGIN_SLOT: fx.Tensor,
        COMB_ADDRS: fx.Tensor,
        NUM_TILE_BLOCKS: fx.Tensor,
        GRAD_GATE: fx.Tensor,
        GATE_ADDRS: fx.Tensor,
    ):
        thread_index = fx.thread_idx.x
        block_index, _b, _c = fx.block_idx
        combine_cu = fx.Int32(num_combine_cu)
        l2y_resource = create_buffer_resource(L2Y, max_size=True)
        origin_rank_res = create_buffer_resource(ORIGIN_RANK, max_size=True)
        origin_slot_res = create_buffer_resource(ORIGIN_SLOT, max_size=True)
        comb_addr_res = create_buffer_resource(COMB_ADDRS, max_size=True)
        num_tile_blocks_res = create_buffer_resource(NUM_TILE_BLOCKS, max_size=True)
        real_tiles = buffer_load(num_tile_blocks_res, fx.Int32(0), vec_width=1, dtype=fx.T.i32())
        grad_gate_res = create_buffer_resource(GRAD_GATE, max_size=True) if with_gate else None
        gate_addr_res = create_buffer_resource(GATE_ADDRS, max_size=True) if with_gate else None

        push_block = combine_bf16_tile(
            thread_index=thread_index,
            block_m_size=BLOCK_M,
            out_features=out_features,
            comb_records=comb_records,
            n_slots=combine_slots,
            l2y_resource=l2y_resource,
            origin_rank_res=origin_rank_res,
            origin_slot_res=origin_slot_res,
            comb_addr_res=comb_addr_res,
            grad_gate_res=grad_gate_res,
            gate_addr_res=gate_addr_res,
            gate_records=gate_records,
            with_gate=with_gate,
        )

        local_count = (real_tiles - block_index + combine_cu - fx.Int32(1)) // combine_cu
        for tile_iter in range(local_count):
            push_block(block_index + tile_iter * combine_cu)

    @flyc.jit
    def launch(
        L2Y,
        ORIGIN_RANK,
        ORIGIN_SLOT,
        COMB_ADDRS,
        NUM_TILE_BLOCKS,
        GRAD_GATE,
        GATE_ADDRS,
        stream: fx.Stream = fx.Stream(None),
    ):
        combine_only_k(
            L2Y,
            ORIGIN_RANK,
            ORIGIN_SLOT,
            COMB_ADDRS,
            NUM_TILE_BLOCKS,
            GRAD_GATE,
            GATE_ADDRS,
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
            BARRIER_LOCAL,
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
    act_bf16,
    weight_bf16,
    l2y,
    tile_to_group,
    sb_l2,
    origin_rank,
    origin_slot,
    combine_slots,
    mblk_dev,
    *,
    comb_addrs=None,
    comb_local=None,
    output=None,
    barrier_local=None,
    barrier_addrs=None,
    topk_indices=None,
    num_tokens_per_rank=None,
    topk_weights=None,
    grad_gate=None,
    gate_addrs=None,
    gate_local=None,
    topk=1,
    num_experts=0,
    rank=0,
    layout="nt",
    BM=256,
    BN=256,
    num_combine_cu=None,
    num_reduce_cu=None,
    nt_vmcnt=3,
    waves_per_eu=2,
    agpr_alloc=0,
    autotune=False,
    autotune_reset=None,
):
    """Fused grouped BF16 GEMM + combine PUSH (+ optional topk reduce), three-role.

    The combine PUSH destination ``combine_output`` [combine_slots, N] bf16 (and, when
    ``grad_gate`` is given, the gate-grad ``combine_gate`` [combine_slots] f32) are returned.
    With ``comb_local`` / ``gate_local`` given they ARE those buffers (EP: this rank's
    symmetric receive buffers); otherwise they are allocated here. The peer pointer tables
    ``comb_addrs`` / ``barrier_addrs`` / ``gate_addrs`` (EP, cross-rank) default to self tables
    built from the local buffers -> single-rank (world=1) self-push when omitted.

    ``layout`` selects the GEMM tile: NT (forward) ``act`` [M,K] / ``weight`` [G,N,K];
    NN (dgrad) ``act`` [M,K] / ``weight`` [G,K,N]; TN (wgrad) ``act`` [K,M] / ``weight``
    [G,K,N]. The combine PUSH reads ``l2y`` [M,N] M-major, so every layout works (it only
    changes how A/B are read). ``l2y`` [M,N] bf16 is the local GEMM output. Per-row
    ``origin_rank`` / ``origin_slot`` route finished rows into ``comb_addrs[origin][slot]``.
    ``sb_l2`` (L2 scoreboard) must be zeroed before the call.

    Role 1 (combine push) runs on ``num_combine_cu`` front blocks (default 64). Supplying
    ``output`` enables role 2 (topk reduce): it reads the token-major ``combine_output``, gated
    per slot by ``barrier_local`` [combine_slots] which role 1 raises (through ``barrier_addrs``),
    sums the ``topk`` rows of each of this rank's tokens, and writes ``output`` [num_tokens, N].
    The reduce runs on the EMPTY grouped-GEMM blocks (freed CUs as the GEMM winds down) plus an
    optional dedicated region of ``num_reduce_cu`` blocks; ``num_reduce_cu`` defaults to 0
    (empty-blocks only — a non-zero value double-reduces, idempotent but wasteful). For the
    token-major reduce, ``origin_slot`` must encode ``token*topk + k`` (NOT the source-order slot
    used by the two-role path). ``topk_indices`` [num_tokens*topk] int32 (< 0 == dropped) and
    ``num_tokens_per_rank`` [world] int32 drive the per-token loop; ``rank`` selects the row.
    ``topk_weights`` [num_tokens*topk] f32 (optional) scales each topk row -> weighted reduce.

    ``grad_gate`` [pool_capacity] f32 (given): role 1 also scatters the per-row gate gradient
    to ``gate_addrs[origin][slot]`` (backward d_topk_w); one f32/row, negligible vs the
    hidden-wide combine push.

    Returns ``(combine_output, combine_gate)`` — each is ``None`` when the caller owns the
    buffer (``comb_local`` / ``gate_local`` given, EP) or the gate is off; only buffers
    allocated here are returned. With ``output`` omitted this degenerates to the two-role
    GEMM + combine kernel. ``autotune=True`` sweeps the num_combine_cu candidates per shape."""
    assert layout in ("nt", "nn", "tn"), f"unknown layout {layout}"
    assert act_bf16.dtype == torch.bfloat16 and weight_bf16.dtype == torch.bfloat16
    if layout == "tn":  # A is [K, M] (K-major)
        hidden_size, pool_capacity = act_bf16.shape
    else:  # A is [M, K]
        pool_capacity, hidden_size = act_bf16.shape
    if layout == "nt":  # weight [G, N, K]
        G, N, K = weight_bf16.shape
    else:  # NN / TN: weight [G, K, N]
        G, K, N = weight_bf16.shape
    assert K == hidden_size, f"weight K={K} != activation K={hidden_size}"
    out_features = N
    c_n = out_features
    if num_combine_cu is None:
        num_combine_cu = 64  # best for fused 3-role (bench fused3 ~2.9ms; e2e sweep 64<48<96)

    device = act_bf16.device

    def _self_table(buf):
        return torch.tensor([buf.data_ptr()], dtype=torch.int64, device=device)

    # combine PUSH destination: caller-owned receive buffer (EP) or allocated here (standalone).
    # comb_addrs (peer push table) defaults to a self table -> world=1 self-push. Only the
    # internally-allocated buffers are returned (returning a caller input would alias it).
    if comb_local is not None:
        combine_output, combine_output_ret = comb_local, None
    else:
        combine_output = torch.empty(combine_slots, out_features, dtype=torch.bfloat16, device=device)
        combine_output_ret = combine_output
    if comb_addrs is None:
        comb_addrs = _self_table(combine_output)
    # gate-grad scatter destination (backward d_topk_w), present iff grad_gate given
    with_gate = grad_gate is not None
    combine_gate_ret = None
    if with_gate:
        if gate_local is not None:
            combine_gate = gate_local
        else:
            combine_gate = torch.empty(combine_slots, dtype=torch.float32, device=device)
            combine_gate_ret = combine_gate
        if gate_addrs is None:
            gate_addrs = _self_table(combine_gate)
    else:
        combine_gate = None

    # reduce active iff output given; num_reduce_cu only sizes the optional dedicated region (0 = empty-block only)
    assert not (output is None and num_reduce_cu), "num_reduce_cu set but output is None"
    reduce_active = output is not None
    if reduce_active:
        assert all(
            t is not None for t in (barrier_local, topk_indices, num_tokens_per_rank)
        ), "topk reduce needs barrier_local/topk_indices/num_tokens_per_rank"
        assert topk >= 1 and num_experts > 0, "topk reduce needs topk>=1 and num_experts>0"
        if num_reduce_cu is None:
            num_reduce_cu = 0
        if barrier_addrs is None:
            barrier_addrs = _self_table(barrier_local)
    else:
        num_reduce_cu = 0
        topk = 1
    apply_weights = reduce_active and topk_weights is not None

    act_flat = act_bf16.contiguous().view(-1)
    if layout == "nt":  # weight [G, N, K]
        weight_flat = weight_bf16.reshape(G * N, K).contiguous().view(-1)
    else:  # NN / TN: weight [G, K, N]
        weight_flat = weight_bf16.reshape(G * K, N).contiguous().view(-1)
    l2y_flat = l2y.contiguous().view(-1)
    # the reduce reads the same buffer the push fills (comb_local == combine_output)
    output_d = output.view(-1) if reduce_active else l2y_flat
    comb_local_d = combine_output.view(-1)
    barrier_local_d = barrier_local if reduce_active else mblk_dev
    barrier_addrs_d = barrier_addrs if reduce_active else comb_addrs
    topk_indices_d = topk_indices.contiguous().view(-1) if reduce_active else mblk_dev
    num_tokens_d = num_tokens_per_rank if reduce_active else mblk_dev
    topk_weights_d = topk_weights.contiguous().view(-1) if apply_weights else mblk_dev
    # gate scatter (backward d_topk_w): grad_gate given -> role 1 also pushes -> gate_addrs[origin][slot]
    grad_gate_d = grad_gate.contiguous().view(-1) if with_gate else mblk_dev
    gate_addrs_d = gate_addrs if with_gate else comb_addrs
    # d_topk_w fold (backward, reduce active): the reduce reads the scattered gate buffer and
    # writes a fresh masked d_topk_w [combine_slots] f32 (replaces the host combine_gate*(topk_idx>=0)
    # mul + clone). Returned in the combine_gate slot. Off (dummy) for forward / non-reduce.
    fold_gate = with_gate and reduce_active
    gate_local_d = combine_gate.view(-1) if fold_gate else mblk_dev
    d_topk_w = torch.empty(combine_slots, dtype=torch.float32, device=device) if fold_gate else None
    d_topk_w_d = d_topk_w if fold_gate else mblk_dev
    if fold_gate:
        combine_gate_ret = d_topk_w

    pos_args = (
        act_flat,
        weight_flat,
        l2y_flat,
        tile_to_group,
        sb_l2,
        origin_rank,
        origin_slot,
        comb_addrs,
        barrier_addrs_d,
        mblk_dev,
        output_d,
        comb_local_d,
        barrier_local_d,
        topk_indices_d,
        num_tokens_d,
        topk_weights_d,
        grad_gate_d,
        gate_addrs_d,
        gate_local_d,
        d_topk_w_d,
        c_n,
    )

    if autotune:
        key = (
            out_features,
            hidden_size,
            pool_capacity,
            BM,
            BN,
            int(combine_slots),
            int(num_reduce_cu),
            int(topk),
            int(num_experts),
            int(rank),
            layout,
            bool(apply_weights),
            bool(with_gate),
        )
        cached = _COMBINE_AUTOTUNE_CACHE.get(key)
        if cached is None:
            cached = _autotune(
                pos_args,
                l2y_flat,
                out_features,
                hidden_size,
                pool_capacity,
                BM,
                BN,
                int(combine_slots),
                int(num_reduce_cu),
                int(topk),
                int(num_experts),
                int(rank),
                sb_l2,
                autotune_reset,
                layout,
                apply_weights,
                reduce_active,
                with_gate,
            )
            _COMBINE_AUTOTUNE_CACHE[key] = cached
        launch, _cfg = cached
    else:
        launch = _compile(
            out_features,
            hidden_size,
            pool_capacity,
            BM,
            BN,
            int(num_combine_cu),
            int(num_reduce_cu),
            int(combine_slots),
            int(topk),
            int(num_experts),
            int(rank),
            int(nt_vmcnt),
            int(waves_per_eu),
            int(agpr_alloc),
            layout=layout,
            apply_weights=apply_weights,
            reduce_active=reduce_active,
            with_gate=with_gate,
        )
    launch(*pos_args, stream=torch.cuda.current_stream())
    # return only internally-allocated buffers; None when the caller owns them (EP)
    return combine_output_ret, combine_gate_ret


def _autotune(
    pos_args,
    finite_view,
    out_features,
    hidden_size,
    pool_capacity,
    BM,
    BN,
    combine_slots,
    num_reduce_cu,
    topk,
    num_experts,
    rank,
    sb_l2,
    reset,
    layout="nt",
    apply_weights=False,
    reduce_active=False,
    with_gate=False,
):
    """Bench the num_combine_cu candidates with a per-iter scoreboard reset; return (launch, cfg)."""
    if reset is None:
        reset = sb_l2.zero_
    stream = torch.cuda.current_stream()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    best_us, best = float("inf"), None
    for num_combine_cu, nt_vmcnt, agpr, waves in _COMBINE_CANDIDATES:
        try:
            launch = _compile(
                out_features,
                hidden_size,
                pool_capacity,
                BM,
                BN,
                int(num_combine_cu),
                int(num_reduce_cu),
                combine_slots,
                int(topk),
                int(num_experts),
                int(rank),
                int(nt_vmcnt),
                int(waves),
                int(agpr),
                layout=layout,
                apply_weights=apply_weights,
                reduce_active=reduce_active,
                with_gate=with_gate,
            )
            reset()
            launch(*pos_args, stream=stream)
            torch.cuda.synchronize()
            if not torch.isfinite(finite_view.view(-1)[:1024].float()).all().item():
                continue
            for _ in range(2):
                reset()
                launch(*pos_args, stream=stream)
            torch.cuda.synchronize()
            us_total = 0.0
            for _ in range(20):
                reset()
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
    l2y,
    origin_rank,
    origin_slot,
    comb_addrs,
    combine_slots,
    mblk_dev,
    *,
    BM=256,
    num_combine_cu=None,
    grad_gate=None,
    gate_addrs=None,
):
    """Combine-PUSH only (no GEMM) -- pushes the pre-filled ``l2y`` rows to the peer
    combine buffers. For measuring raw combine bandwidth.

    ``grad_gate`` [pool_capacity] f32 + ``gate_addrs`` [world] i64 (both given): also
    scatter the per-row gate gradient to ``gate_addrs[origin][slot]`` (backward d_topk_w);
    one f32/row by lane 0, negligible vs the hidden-wide combine push."""
    out_features = l2y.size(1)
    pool_capacity = l2y.size(0)
    if num_combine_cu is None:
        num_combine_cu = pool_capacity // BM
    l2y_flat = l2y.contiguous().view(-1)
    with_gate = grad_gate is not None and gate_addrs is not None
    launch = _compile_combine_only(
        out_features,
        pool_capacity,
        int(BM),
        int(num_combine_cu),
        int(combine_slots),
        with_gate=with_gate,
    )
    # dummies satisfy the fixed kernel signature when gate is off (gated by const_expr)
    grad_gate_arg = grad_gate if with_gate else mblk_dev
    gate_addrs_arg = gate_addrs if with_gate else comb_addrs
    launch(
        l2y_flat,
        origin_rank,
        origin_slot,
        comb_addrs,
        mblk_dev,
        grad_gate_arg,
        gate_addrs_arg,
        stream=torch.cuda.current_stream(),
    )
    return l2y


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
