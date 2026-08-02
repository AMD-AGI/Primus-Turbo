###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Fused BF16 GEMM (mxfp8 epilogue quant) + FP8 combine PUSH + FP8-dequant reduce (FlyDSL).

EXPERIMENTAL DEAD-END (not wired into any forward path). Bit-correct (cos 0.9996 vs bf16 fused)
but ~0.76x (slower) than `grouped_gemm_combine_bf16`, and has an intermittent reduce-flag liveness
stall under back-to-back timing calls. Kept only as a reference for the exhausted fp8-L2-combine
approach. See NOTES_mxfp8_fused_gemm_combine_perf.md: the mxfp8 quant of the L2 GEMM output is
expensive compute wherever placed (combine / separate role / this epilogue) and exceeds the combine
byte-savings, so fp8 gives no fused-L2 win. Production L2 = bf16 fused; use fp8 at L1 only.

3-role L2 down-proj pipeline. The GEMM epilogue quantizes its f32 MFMA accumulators to
mxfp8 (per-1x32 E8M0) IN-REGISTER via a 32-lane butterfly amax (a 32x32 MFMA tile == one
32-col block) and writes LOCAL fp8 L2Y directly. So combine is a pure XGMI-bound fp8 copy
(the byte lever pays, few CUs) while the quant rides the GEMM's own CUs (off the combine
critical path):

  * role COMBINE ``[0, ncomb)``: spin ``sb_l2`` (GEMM done), l2_invalidate, read LOCAL fp8
    L2Y, push fp8 (payload + E8M0) to the peer packed ``comb``, raise ``barrier_local`` flag.
  * role REDUCE (empty GEMM blocks + optional dedicated): dequant topk fp8 rows -> ``output``.
  * role GEMM ``[gemm_base, ...)``: mxfp8 NT tile (``emit_gemm_mxfp8_nt_tile``) with a CShuffle
    mxfp8-quant epilogue -> LOCAL fp8 L2Y (write-through sc1) + E8M0; release-st per
    ``(block_m, block_n)`` combine_flag slot (no cross-WG atomic).

LOCAL fp8 L2Y buffers (``L2Y_FP8`` uint8 [pool*H], ``L2Y_SCALE`` uint8 [pool*H/32]); peer
``comb`` holds the packed fp8 payload + E8M0 scale (fits the bf16 comb region).
"""

import functools
import os

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, range_constexpr, rocdl
import torch
from flydsl._mlir.dialects import vector as _vector
from flydsl.compiler.ast_rewriter import ASTRewriter
from flydsl.expr.buffer_ops import (
    buffer_load,
    buffer_store,
    create_buffer_resource,
    create_buffer_resource_from_addr,
)
from flydsl.expr.rocdl import cvt_pk_f32_fp8
from flydsl.expr.typing import Vector as Vec

from primus_turbo.flydsl.mega.fp8.combine_config import _BLOCK_THREADS, _NUM_WARPS, _WARP
from primus_turbo.flydsl.mega.prims import cast
from primus_turbo.flydsl.mega.fp8.prims import (
    _wait_mem,
    l2_invalidate,
    ld,
    read_clock,
    spin_timed_out,
    st,
)
from primus_turbo.flydsl.mega.fp8.sym_layout import SymLayout
from primus_turbo.flydsl.mega.fp8.symm_buffer import get_symm_buffer_for_mega_moe
from primus_turbo.flydsl.mega.fp8.dispatch_grouped_gemm_mxfp8_kernel import _host_rendezvous
from primus_turbo.flydsl.mega.fp8.gemm_mxfp8_tile import (
    BLOCK_K as _MXFP8_BLOCK_K,
    emit_gemm_mxfp8_nt_tile,
)
from primus_turbo.flydsl.mega.fp8.gemm_helper import (
    StoreCQuantMxfp8CShuffle,
    _emit_if_then,
    make_value_attrs,
    xcd_remap_pid,
)
from primus_turbo.flydsl.grouped_gemm.gemm_fp8_grouped_kernel import _grouped_block_mn

# GEMM launch attrs (nt_vmcnt / waves_per_eu); included in flyc compile cache key below.
_COMBINE_NT_VMCNT = 3
_COMBINE_WAVES_PER_EU = 2
# Persistent GEMM optional (usually slower when total tiles >> num CUs — one-WG-per-tile
# keeps higher parallelism).  XCD / group_m swizzle apply in both modes.
_COMBINE_PERSISTENT_GEMM = os.environ.get("PT_COMBINE_PERSISTENT_GEMM", "0") != "0"
_COMBINE_GEMM_CU = int(os.environ.get("PT_COMBINE_GEMM_CU", "256"))
_COMBINE_NUM_XCD = int(os.environ.get("PT_COMBINE_NUM_XCD", "1"))
_COMBINE_GROUP_M = int(os.environ.get("PT_COMBINE_GROUP_M", "0"))
_COMBINE_GROUP_N = int(os.environ.get("PT_COMBINE_GROUP_N", "0"))
_COMBINE_TILE_SWIZZLE = _COMBINE_NUM_XCD > 1 or _COMBINE_GROUP_M > 0 or _COMBINE_GROUP_N > 0
# Launch only real_tiles*n_blocks GEMM blocks (+ separate tail-reduce reservation), not worst_case.
_COMBINE_REAL_TILES_GRID = os.environ.get("PT_COMBINE_REAL_TILES_GRID", "1") != "0"
# backward STEP3: optional split GEMM+push from reduce (host barrier between). Off by default
# after rowwise-quant fix; enable with PT_COMBINE_SPLIT_REDUCE=1 if co-scheduling races return.
_SPLIT_BWD_REDUCE = os.environ.get("PT_COMBINE_SPLIT_REDUCE", "0") == "1"
# top-k reduce payload load width in i32 words per lane. 4 -> b128 (matches the PUSH path); 1 ->
# the original dword path. Words are 4-aligned, so a group of 4 stays inside one E8M0 32-block:
# one scale word + one shift serves all 4. Falls back to 1 when H4 is not a multiple of _WARP*VW.
_REDUCE_VW = int(os.environ.get("PT_COMBINE_REDUCE_VW", "4"))


@functools.lru_cache(maxsize=4)
def _make_epoch_bump(add_combine, add_reduce):
    """Single-block device kernel: flip the flag parity, bump combine/reduce expected[new_parity].

    Mirrors the bf16 combine's ``_make_epoch_bump``. Launched on the combine stream just before the
    main kernel so the flags self-reset (no host synchronize()+barrier() rendezvous, no cross-call
    reset race): the flag banks are never zeroed; each call spins on the cumulative per-bank
    ``expected`` instead."""

    @flyc.kernel(known_block_size=[_BLOCK_THREADS, 1, 1])
    def epoch_bump_kernel(PARITY: fx.Tensor, COMBINE_EXP: fx.Tensor, REDUCE_EXP: fx.Tensor):
        if fx.thread_idx.x == fx.Int32(0):
            parity_res = create_buffer_resource(PARITY, max_size=True)
            combine_res = create_buffer_resource(COMBINE_EXP, max_size=True)
            reduce_res = create_buffer_resource(REDUCE_EXP, max_size=True)
            new_parity = buffer_load(parity_res, fx.Int32(0), vec_width=1, dtype=fx.T.i64()) ^ fx.Int64(1)
            buffer_store(new_parity, parity_res, fx.Int32(0))
            idx = cast(new_parity, fx.T.i32())
            new_combine = buffer_load(combine_res, idx, vec_width=1, dtype=fx.T.i64()) + fx.Int64(add_combine)
            buffer_store(new_combine, combine_res, idx)
            new_reduce = buffer_load(reduce_res, idx, vec_width=1, dtype=fx.T.i64()) + fx.Int64(add_reduce)
            buffer_store(new_reduce, reduce_res, idx)

    return epoch_bump_kernel


# ─────────── role COMBINE: read LOCAL fp8 L2Y -> push peer packed comb (pure copy) + flags ───────────
def combine_copy_fp8_tile(
    *, thread_index, block_m_size, hidden, comb_records, H4, SC, payload_i32_total,
    l2y_fp8_res, l2y_scale_res, origin_rank_res, origin_slot_res, comb_base, signal_delta_res, barrier_base,
    reduce_bank, expected_reduce,
    with_gate=False, grad_gate_res=None, gate_base=None, main_delta_res=None, gate_records=0,
):
    """FP8 combine PUSH: read local fp8 L2Y row -> push packed fp8 payload + E8M0 to the peer
    ``comb[slot]``, raise the sys-scope flag. ``with_gate`` (backward STEP3) additionally scatters
    the per-row gate gradient ``grad_gate[row]`` to the origin peer's ``combine_gate[slot]`` (MAIN
    heap ``main_delta``), mirroring the bf16 ``combine_bf16_tile`` gate path -- 1 f32, ~free vs the
    hidden-wide fp8 push."""
    rows_per_warp = block_m_size // _NUM_WARPS
    lane = thread_index % fx.Int32(_WARP)
    warp_id = thread_index // fx.Int32(_WARP)
    chunk_base = warp_id * fx.Int32(rows_per_warp)
    cols_per_step = _WARP * 4  # 256 i32 payload words/step (b128 copy)
    num_full = H4 // cols_per_step

    def _make_row_emitter(row, origin):
        # factory, not a closure over the loop var: `_emit_if_then` branch fns MUST take 0 args
        # (ReplaceIfWithDispatch injects result_names into any arg-accepting branch fn).
        def _emit_row():
            slot = buffer_load(origin_slot_res, row, vec_width=1, dtype=fx.T.i32())
            delta = buffer_load(signal_delta_res, origin, vec_width=1, dtype=fx.T.i64())
            peer = create_buffer_resource_from_addr(comb_base + delta, num_records_bytes=comb_records)
            slot_base = slot * fx.Int32(H4)
            row_base = row * fx.Int32(H4)
            vals = []
            for c in range(num_full):
                col = fx.Int32(c * cols_per_step) + lane * fx.Int32(4)
                vals.append(buffer_load(l2y_fp8_res, row_base + col, vec_width=4, dtype=fx.T.i32()))
            for c in range(num_full):
                col = fx.Int32(c * cols_per_step) + lane * fx.Int32(4)
                buffer_store(vals[c], peer, slot_base + col)

            def _emit_scale():
                sv = buffer_load(l2y_scale_res, row * fx.Int32(SC) + lane, vec_width=1, dtype=fx.T.i32())
                buffer_store(sv, peer, fx.Int32(payload_i32_total) + slot * fx.Int32(SC) + lane)

            _emit_if_then(lane < fx.Int32(SC), _emit_scale)
            if with_gate:
                # scatter the per-row gate gradient (d_topk_w) to origin[slot] in the MAIN-heap
                # combine_gate; same value/slot across lanes (idempotent, like the flag store).
                gate_value = buffer_load(grad_gate_res, row, vec_width=1, dtype=fx.T.f32())
                gate_addr = gate_base + buffer_load(main_delta_res, origin, vec_width=1, dtype=fx.T.i64())
                gate_peer = create_buffer_resource_from_addr(gate_addr, num_records_bytes=gate_records)
                buffer_store(gate_value, gate_peer, slot)
            # epoch flag: write the cumulative reduce target into the peer's reduce_flag bank
            # (never reset; the reduce spins on == expected_reduce). reduce_bank uses OUR parity,
            # which equals the peer's by lockstep, so it lands in the peer's current bank.
            barrier_addr = barrier_base + delta
            _wait_mem()
            st(barrier_addr, reduce_bank + slot, expected_reduce, scope="sys")

        return _emit_row

    def push_block(block_m):
        base_row = block_m * fx.Int32(block_m_size) + chunk_base
        for j in range(rows_per_warp):
            row = base_row + fx.Int32(j)
            origin = buffer_load(origin_rank_res, row, vec_width=1, dtype=fx.T.i32())
            _emit_if_then(origin >= fx.Int32(0), _make_row_emitter(row, origin))

    return push_block


# ─────────── role REDUCE: fp8-dequant topk sum (weighted|unweighted) + optional gate fold ───────────
def _make_topk_reduce_fp8(hidden, topk, combine_slots, apply_weights, with_gate):
    """FP8-dequant weighted top-k sum -> output. ``apply_weights`` (forward L2) multiplies each expert
    term by the routing weight; else (backward STEP3) the weight was already folded into the input
    upstream. ``with_gate`` (backward) additionally folds the gate gradient
    ``d_topk_w[slot] = combine_gate[slot]`` (route-masked). Mirrors the unified bf16
    ``topk_reduce_bf16_tile(apply_weights, with_gate)`` -- one reduce for both fwd and bwd."""
    H4 = hidden // 4
    payload_i32_total = combine_slots * H4
    SC = hidden // 128
    VW = _REDUCE_VW if H4 % (_WARP * _REDUCE_VW) == 0 else 1
    steps_per_lane = H4 // (_WARP * VW)

    def _reduce(thread_index, base_pid, total_warps, num_experts, rank, comb_base, comb_records,
                output_res, topk_indices_res, num_tokens_res, barrier_base, reduce_bank, expected_reduce,
                topk_weights_res, gate_local_res, d_topk_w_res):
        _v2 = fx.T.VectorType.get([2], fx.T.f32())
        f32_v4 = fx.T.VectorType.get([4], fx.T.f32())
        bf16_v4 = fx.T.VectorType.get([4], fx.T.bf16())
        lane = thread_index % fx.Int32(_WARP)
        warp_id = thread_index // fx.Int32(_WARP)
        global_warp_id = base_pid * fx.Int32(_NUM_WARPS) + warp_id
        num_tokens = buffer_load(num_tokens_res, fx.Int32(rank), vec_width=1, dtype=fx.T.i32())
        comb_res = create_buffer_resource_from_addr(comb_base, num_records_bytes=comb_records)

        token = global_warp_id
        while token < num_tokens:
            valid = []
            for jj in fx.range_constexpr(topk):
                slot = token * fx.Int32(topk) + fx.Int32(jj)
                topk_index = buffer_load(topk_indices_res, slot, vec_width=1, dtype=fx.T.i64())
                valid.append((topk_index >= fx.Int64(0)) & (topk_index < fx.Int64(num_experts)))
            for jj in fx.range_constexpr(topk):
                slot = token * fx.Int32(topk) + fx.Int32(jj)
                topk_index = buffer_load(topk_indices_res, slot, vec_width=1, dtype=fx.T.i64())
                if topk_index >= fx.Int64(0):
                    if topk_index < fx.Int64(num_experts):
                        if lane == fx.Int32(0):
                            spin_start = read_clock()
                            flag = ld(barrier_base, reduce_bank + slot, scope="agent", dtype=fx.T.i64())
                            while flag != expected_reduce:
                                fx.rocdl.s_sleep(fx.Int32(1))
                                if spin_timed_out(spin_start):
                                    fx.printf("MEGA fp8 ep reduce flag timeout: rank={} token={} slot={}\n",
                                              fx.Int32(rank), token, slot)
                                    spin_start = read_clock()
                                flag = ld(barrier_base, reduce_bank + slot, scope="agent", dtype=fx.T.i64())
            _wait_mem()

            zero_vec = fx.arith.constant_vector(0.0, f32_v4)
            for k in fx.range_constexpr(steps_per_lane):
                # lane owns VW consecutive payload words -> b128 per lane, 1024 B per warp step.
                w = lane * fx.Int32(VW) + fx.Int32(k * _WARP * VW)
                sword_idx = w // fx.Int32(32)
                shift = fx.Int32(8) * ((w // fx.Int32(8)) % fx.Int32(4))
                acc = [fx.arith.constant_vector(0.0, f32_v4) for _ in range_constexpr(VW)]
                for jj in fx.range_constexpr(topk):
                    slot = token * fx.Int32(topk) + fx.Int32(jj)
                    pv = buffer_load(
                        comb_res, slot * fx.Int32(H4) + w, vec_width=VW, dtype=fx.T.i32(),
                    )
                    words = [Vec(pv)[v].ir_value() for v in range_constexpr(VW)] if VW > 1 else [pv]
                    sw = buffer_load(
                        comb_res, fx.Int32(payload_i32_total) + slot * fx.Int32(SC) + sword_idx,
                        vec_width=1, dtype=fx.T.i32(),
                    )
                    e8 = (fx.arith.ArithValue(sw) >> shift) & fx.Int32(0xFF)
                    sf = (fx.arith.ArithValue(e8) << fx.Int32(23)).bitcast(fx.T.f32())
                    coef = fx.arith.ArithValue(sf)  # UNWEIGHTED (bwd): coef = scale
                    if const_expr(apply_weights):   # WEIGHTED (fwd L2): coef = scale * topk_weight
                        wj = buffer_load(topk_weights_res, slot, vec_width=1, dtype=fx.T.f32())
                        coef = fx.arith.ArithValue(sf) * fx.arith.ArithValue(wj)
                    coef_v = _vector.broadcast(f32_v4, fx.arith._to_raw(coef))
                    for v in range_constexpr(VW):
                        pw = words[v]
                        lo = cvt_pk_f32_fp8(res=_v2, src=pw, word_sel=False)
                        hi = cvt_pk_f32_fp8(res=_v2, src=pw, word_sel=True)
                        deq = _vector.shuffle(lo, hi, [0, 1, 2, 3])
                        term = fx.arith.mulf(deq, coef_v)
                        term = fx.arith.select(valid[jj], term, zero_vec)
                        acc[v] = fx.arith.addf(acc[v], term)
                for v in range_constexpr(VW):
                    buffer_store(
                        fx.arith.trunc_f(bf16_v4, acc[v]), output_res,
                        token * fx.Int32(hidden) + (w + fx.Int32(v)) * fx.Int32(4),
                    )

            if const_expr(with_gate):
                # d_topk_w[slot] = combine_gate[slot] for valid routes else 0 (backward gate grad).
                for jj in fx.range_constexpr(topk):
                    slot = token * fx.Int32(topk) + fx.Int32(jj)
                    topk_index = buffer_load(topk_indices_res, slot, vec_width=1, dtype=fx.T.i64())
                    if lane == fx.Int32(0):
                        gate_v = buffer_load(gate_local_res, slot, vec_width=1, dtype=fx.T.f32())
                        zero_f = fx.Float32(0.0)
                        v1 = fx.arith.select(topk_index < fx.Int64(num_experts), gate_v, zero_f)
                        d_val = fx.arith.select(topk_index >= fx.Int64(0), v1, zero_f)
                        buffer_store(d_val, d_topk_w_res, slot)
            # epoch flags self-reset (double-banked, cumulative expected) -> NO consuming store.
            token = token + total_warps

    return ASTRewriter.transform(_reduce)


_FP8_COMBINE_COMPILED: dict = {}


@functools.lru_cache(maxsize=64)
def _compile(
    out_features, hidden_size, num_max_pool_tokens, BLOCK_M, BLOCK_N, num_combine_cu, num_reduce_cu,
    combine_slots, topk, num_experts, rank, num_ranks, apply_weights, with_gate,
    num_groups=0, nt_vmcnt=_COMBINE_NT_VMCNT, waves_per_eu=_COMBINE_WAVES_PER_EU, agpr_alloc=0,
    real_tiles_grid=_COMBINE_REAL_TILES_GRID,
    persistent_gemm=_COMBINE_PERSISTENT_GEMM, num_persistent_cu=_COMBINE_GEMM_CU,
    num_xcd=_COMBINE_NUM_XCD, group_m=_COMBINE_GROUP_M, group_n=_COMBINE_GROUP_N,
    tile_swizzle=_COMBINE_TILE_SWIZZLE,
    force_no_reduce=False,
):
    """Unified fp8 combine: mxfp8 GEMM (CShuffle mxfp8-quant epilogue -> local fp8 pool) + FP8 combine
    PUSH (+ optional gate scatter) + fp8-dequant top-k reduce. One kernel for BOTH:
      * forward L2   (apply_weights=True,  with_gate=False): ``act @ w2``, K=I, WEIGHTED reduce -> y.
      * backward STEP3 (apply_weights=False, with_gate=True): ``grad_l1 @ w1^T``, K=2I, UNWEIGHTED
        reduce (routing weight folded upstream) + gate scatter / d_topk_w fold -> dx.
    Mirrors the unified bf16 ``grouped_gemm_combine`` (apply_weights/with_gate constexpr flags)."""
    K = hidden_size
    assert out_features % BLOCK_N == 0
    assert num_max_pool_tokens % BLOCK_M == 0
    assert BLOCK_N % 256 == 0, "epilogue quant assumes N_TILES_B=1 (BLOCK_N a multiple of 256)"
    _mx_a_lds = (BLOCK_M // 2) * _MXFP8_BLOCK_K
    _mx_b_lds = (BLOCK_N // 2) * _MXFP8_BLOCK_K
    _mx_cshuf_n = _BLOCK_THREADS // 64 * 16 * (BLOCK_N // 128 * 16)

    @fx.struct
    class _SharedStorage:
        A_lds_cur_0: fx.Array[fx.Float8E4M3FN, _mx_a_lds, 16]
        A_lds_cur_1: fx.Array[fx.Float8E4M3FN, _mx_a_lds, 16]
        A_lds_next_0: fx.Array[fx.Float8E4M3FN, _mx_a_lds, 16]
        A_lds_next_1: fx.Array[fx.Float8E4M3FN, _mx_a_lds, 16]
        B_lds_cur_0: fx.Array[fx.Float8E4M3FN, _mx_b_lds, 16]
        B_lds_cur_1: fx.Array[fx.Float8E4M3FN, _mx_b_lds, 16]
        B_lds_next_0: fx.Array[fx.Float8E4M3FN, _mx_b_lds, 16]
        B_lds_next_1: fx.Array[fx.Float8E4M3FN, _mx_b_lds, 16]
        C_lds_shuffle: fx.Array[fx.BFloat16, _mx_cshuf_n, 16]

    n_blocks = out_features // BLOCK_N
    worst_case_tiles = num_max_pool_tokens // BLOCK_M
    gemm_grid_blocks = worst_case_tiles * n_blocks
    comb_records = combine_slots * out_features * 2
    gate_records = combine_slots * 4  # f32 gate slots per peer (backward d_topk_w scatter)
    delta_records = num_ranks * 8
    pool_records = num_max_pool_tokens * 4
    dedicated_reduce_warps = num_reduce_cu * _NUM_WARPS
    gemm_base = num_combine_cu + num_reduce_cu
    H4 = out_features // 4
    SC = out_features // 128
    payload_i32_total = combine_slots * H4
    _env_no_reduce = os.environ.get("PT_COMBINE_NO_REDUCE", "0") == "1"
    # PT_COMBINE_GEMM_ONLY (isolation): combine PUSH does 0 tiles + reduce compiled out -> the kernel
    # runs ONLY the mxfp8 fc1-dgrad GEMM role (+ CShuffle fp8 epilogue). Measures the GEMM-role wall
    # (no cross-rank comm, no reduce) -> INCORRECT output, timing only.
    _gemm_only = os.environ.get("PT_COMBINE_GEMM_ONLY", "0") == "1"
    # PT_COMBINE_PUSH_ONLY (isolation): combine PUSH runs but SKIPS the GEMM-done sb_l2 gate (GEMM +
    # reduce idle), so it just XGMI-copies whatever's in the local fp8 L2Y to the peer comb + flags.
    # Measures the combine-PUSH wall (cross-rank byte cost) -> INCORRECT output, timing only.
    _push_only = os.environ.get("PT_COMBINE_PUSH_ONLY", "0") == "1"
    _no_reduce = force_no_reduce or _env_no_reduce or _gemm_only or _push_only
    reduce_fp8 = _make_topk_reduce_fp8(out_features, topk, combine_slots, apply_weights, with_gate)
    # Tail-reduce reservation when pool capacity exceeds real tiles (worst_case > real_tiles).
    max_tail_blocks = (worst_case_tiles - 1) * n_blocks

    @flyc.kernel(known_block_size=[_BLOCK_THREADS, 1, 1])
    def kern(
        ACT: fx.Tensor, WEIGHTS: fx.Tensor, L2Y_FP8: fx.Tensor, L2Y_SCALE: fx.Tensor,
        TILE_TO_GROUP: fx.Tensor, NUM_TILE_BLOCKS: fx.Tensor, OUTPUT: fx.Tensor, TOPK_INDICES: fx.Tensor,
        NUM_TOKENS_PER_RANK: fx.Tensor, TOPK_WEIGHTS: fx.Tensor, GRAD_GATE: fx.Tensor, D_TOPK_W: fx.Tensor,
        A_SCALE: fx.Tensor, B_SCALE: fx.Tensor,
        COMBINE_PARITY: fx.Tensor, COMBINE_EXPECTED: fx.Tensor, REDUCE_EXPECTED: fx.Tensor,
        sym_layout: SymLayout, c_n: fx.Int32,
    ):
        thread_index = fx.thread_idx.x
        block_index, _b, _c = fx.block_idx
        combine_cu = fx.Int32(num_combine_cu)
        reduce_cu = fx.Int32(num_reduce_cu)
        lds = fx.SharedAllocator().allocate(_SharedStorage).peek()

        combine_flag_base = sym_layout.combine_flag_ptr
        comb_base = sym_layout.comb_ptr
        reduce_flag_base = sym_layout.reduce_flag_ptr
        gate_base = sym_layout.combine_gate_ptr
        # ---- epoch: parity picks the flag bank; expected[parity] is the cumulative spin target ----
        combine_parity_res = create_buffer_resource(COMBINE_PARITY, max_size=True)
        combine_expected_res = create_buffer_resource(COMBINE_EXPECTED, max_size=True)
        reduce_expected_res = create_buffer_resource(REDUCE_EXPECTED, max_size=True)
        parity = cast(
            buffer_load(combine_parity_res, fx.Int32(0), vec_width=1, dtype=fx.T.i64()), fx.T.i32()
        )
        combine_bank = parity * fx.Int32(worst_case_tiles * n_blocks)
        n_blocks_i32 = fx.Int32(n_blocks)
        reduce_bank = parity * fx.Int32(combine_slots)
        expected_combine = buffer_load(combine_expected_res, parity, vec_width=1, dtype=fx.T.i64())
        expected_reduce = buffer_load(reduce_expected_res, parity, vec_width=1, dtype=fx.T.i64())
        l2y_fp8_res = create_buffer_resource(L2Y_FP8, max_size=True)
        l2y_scale_res = create_buffer_resource(L2Y_SCALE, max_size=True)
        signal_delta_res = create_buffer_resource_from_addr(
            sym_layout.signal_offsets_ptr, num_records_bytes=delta_records
        )
        main_delta_res = create_buffer_resource_from_addr(
            sym_layout.offsets_ptr, num_records_bytes=delta_records
        )
        origin_rank_res = create_buffer_resource_from_addr(sym_layout.origin_rank_ptr, num_records_bytes=pool_records)
        origin_slot_res = create_buffer_resource_from_addr(sym_layout.origin_slot_ptr, num_records_bytes=pool_records)
        gate_local_res = create_buffer_resource_from_addr(gate_base, num_records_bytes=gate_records)

        group_resource = create_buffer_resource(TILE_TO_GROUP, max_size=True)
        num_tile_blocks_res = create_buffer_resource(NUM_TILE_BLOCKS, max_size=True)
        output_res = create_buffer_resource(OUTPUT, max_size=True)
        topk_indices_res = create_buffer_resource(TOPK_INDICES, max_size=True)
        num_tokens_res = create_buffer_resource(NUM_TOKENS_PER_RANK, max_size=True)
        topk_weights_res = create_buffer_resource(TOPK_WEIGHTS, max_size=True)
        grad_gate_res = create_buffer_resource(GRAD_GATE, max_size=True)
        d_topk_w_res = create_buffer_resource(D_TOPK_W, max_size=True)
        real_tiles = buffer_load(num_tile_blocks_res, fx.Int32(0), vec_width=1, dtype=fx.T.i32())

        if block_index < combine_cu:
            push_block = combine_copy_fp8_tile(
                thread_index=thread_index, block_m_size=BLOCK_M, hidden=out_features, comb_records=comb_records,
                H4=H4, SC=SC, payload_i32_total=payload_i32_total, l2y_fp8_res=l2y_fp8_res,
                l2y_scale_res=l2y_scale_res, origin_rank_res=origin_rank_res, origin_slot_res=origin_slot_res,
                comb_base=comb_base, signal_delta_res=signal_delta_res, barrier_base=reduce_flag_base,
                reduce_bank=reduce_bank, expected_reduce=expected_reduce,
                with_gate=with_gate, grad_gate_res=grad_gate_res, gate_base=gate_base,
                main_delta_res=main_delta_res, gate_records=gate_records,
            )
            local_count = (
                fx.Int32(0) if _gemm_only
                else (real_tiles - block_index + combine_cu - fx.Int32(1)) // combine_cu
            )
            for tile_iter in range(local_count):
                block_m = block_index + tile_iter * combine_cu
                if not _push_only:  # PUSH_ONLY skips the GEMM-done gate + acquire (GEMM idle) -> pure push
                    if thread_index == fx.Int32(0):
                        spin_start = read_clock()
                        flag_base = combine_bank + block_m * n_blocks_i32
                        waiting = fx.Int32(1)
                        while waiting != fx.Int32(0):
                            ready = fx.Int32(0)
                            for bn in range_constexpr(n_blocks):
                                sig = ld(
                                    combine_flag_base, flag_base + fx.Int32(bn),
                                    scope="agent", dtype=fx.T.i64(),
                                )
                                if sig == expected_combine:
                                    ready = ready + fx.Int32(1)
                            if ready == n_blocks_i32:
                                waiting = fx.Int32(0)
                            else:
                                fx.rocdl.s_sleep(fx.Int32(2))
                                if spin_timed_out(spin_start):
                                    fx.printf(
                                        "MEGA fp8 ep combine gate timeout: block={} ready={}\n",
                                        block_m, ready,
                                    )
                                    spin_start = read_clock()
                    fx.gpu.barrier()
                    l2_invalidate()
                push_block(block_m)
            fx.rocdl.s_waitcnt(0)
        else:
            role_idx = block_index - fx.Int32(gemm_base)

            def _do_gemm_tile(block_m, block_n):
                c_m_const = fx.Int32(num_max_pool_tokens)
                group_index = buffer_load(group_resource, block_m, vec_width=1, dtype=fx.T.i32())
                n_tiles_a = BLOCK_M // 64
                n_tiles_b = BLOCK_N // 128
                c_idx_fn = lambda i, j: i * n_tiles_b + j
                store_c = StoreCQuantMxfp8CShuffle(
                    L2Y_FP8, L2Y_SCALE, c_m_const, out_features,
                    c_idx_fn, n_tiles_a, n_tiles_b,
                    fx.BFloat16, lds.C_lds_shuffle, thread_index // fx.Int32(64),
                )
                emit_gemm_mxfp8_nt_tile(
                    ACT, A_SCALE, WEIGHTS, B_SCALE, lds, block_m, block_n,
                    K=K, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, G=num_groups, group_idx=group_index,
                    c_m=c_m_const, c_n=c_n, nt_vmcnt=nt_vmcnt, scale_pack=4,
                    store_c=store_c, full_barrier=False, b_n_slabs=False,
                )
                fx.rocdl.s_waitcnt(0)
                rocdl.s_barrier()
                flag_off = combine_bank + block_m * n_blocks_i32 + block_n
                _emit_if_then(
                    thread_index == fx.Int32(0),
                    lambda: st(combine_flag_base, flag_off, expected_combine, scope="agent"),
                )

            def _map_gemm_tile(gemm_tile_index):
                total_gemm_tiles = real_tiles * fx.Int32(n_blocks)
                tt = xcd_remap_pid(gemm_tile_index, total_gemm_tiles, num_xcd)
                m_end_rows = real_tiles * fx.Int32(BLOCK_M)
                return _grouped_block_mn(
                    tt, fx.Int32(0), m_end_rows, n_blocks, BLOCK_M, group_m, group_n
                )

            def _persistent_gemm_tile(local):
                total_gemm_tiles = real_tiles * fx.Int32(n_blocks)
                tt = xcd_remap_pid(local, total_gemm_tiles, num_xcd)
                m_end_rows = real_tiles * fx.Int32(BLOCK_M)
                return _grouped_block_mn(
                    tt, fx.Int32(0), m_end_rows, n_blocks, BLOCK_M, group_m, group_n
                )

            def _run_persistent_gemm(persistent_pid):
                total_gemm_tiles = real_tiles * fx.Int32(n_blocks)
                nsms = fx.Int32(num_persistent_cu)
                for t in range(persistent_pid, total_gemm_tiles, nsms):
                    bm, bn = _persistent_gemm_tile(t)
                    _do_gemm_tile(bm, bn)

            if const_expr(not _no_reduce) and block_index < combine_cu + reduce_cu:
                reduce_fp8(
                    thread_index, block_index - combine_cu, fx.Int32(dedicated_reduce_warps),
                    num_experts, rank, comb_base, comb_records, output_res, topk_indices_res,
                    num_tokens_res, reduce_flag_base, reduce_bank, expected_reduce,
                    topk_weights_res, gate_local_res, d_topk_w_res,
                )
            elif const_expr(persistent_gemm):
                if role_idx < fx.Int32(num_persistent_cu):
                    if not _push_only:
                        _run_persistent_gemm(role_idx)
                elif const_expr(not _no_reduce) and const_expr(num_reduce_cu == 0):
                    tail_idx = role_idx - fx.Int32(num_persistent_cu)
                    n_empty_tiles = fx.Int32(worst_case_tiles) - real_tiles
                    max_tail = n_empty_tiles * fx.Int32(n_blocks)
                    if tail_idx < max_tail:
                        reduce_fp8(
                            thread_index, tail_idx, max_tail * fx.Int32(_NUM_WARPS),
                            num_experts, rank, comb_base, comb_records, output_res, topk_indices_res,
                            num_tokens_res, reduce_flag_base, reduce_bank, expected_reduce,
                            topk_weights_res, gate_local_res, d_topk_w_res,
                        )
            else:
                role_idx = block_index - fx.Int32(gemm_base)
                if const_expr(real_tiles_grid):
                    real_gemm_blocks = real_tiles * fx.Int32(n_blocks)
                    if role_idx < real_gemm_blocks:
                        gemm_tile_index = role_idx
                        if const_expr(tile_swizzle):
                            block_m, block_n = _map_gemm_tile(gemm_tile_index)
                        else:
                            block_m = gemm_tile_index // fx.Int32(n_blocks)
                            block_n = gemm_tile_index % fx.Int32(n_blocks)
                        if not _push_only:
                            _do_gemm_tile(block_m, block_n)
                    elif const_expr(not _no_reduce) and const_expr(num_reduce_cu == 0):
                        tail_idx = role_idx - real_gemm_blocks
                        n_empty_tiles = fx.Int32(worst_case_tiles) - real_tiles
                        max_tail = n_empty_tiles * fx.Int32(n_blocks)
                        if tail_idx < max_tail:
                            reduce_fp8(
                                thread_index, tail_idx, max_tail * fx.Int32(_NUM_WARPS),
                                num_experts, rank, comb_base, comb_records, output_res, topk_indices_res,
                                num_tokens_res, reduce_flag_base, reduce_bank, expected_reduce,
                                topk_weights_res, gate_local_res, d_topk_w_res,
                            )
                else:
                    gemm_tile_index = role_idx
                    if const_expr(tile_swizzle):
                        block_m, block_n = _map_gemm_tile(gemm_tile_index)
                    else:
                        block_m = gemm_tile_index // fx.Int32(n_blocks)
                        block_n = gemm_tile_index % fx.Int32(n_blocks)
                    if const_expr(_no_reduce):
                        if not _push_only and block_m < real_tiles:
                            _do_gemm_tile(block_m, block_n)
                    elif block_m < real_tiles:
                        _do_gemm_tile(block_m, block_n)
                    elif const_expr(num_reduce_cu == 0):
                        empty_ordinal = gemm_tile_index - real_tiles * fx.Int32(n_blocks)
                        total_empty_warps = (
                            fx.Int32(gemm_grid_blocks) - real_tiles * fx.Int32(n_blocks)
                        ) * fx.Int32(_NUM_WARPS)
                        reduce_fp8(
                            thread_index, empty_ordinal, total_empty_warps, num_experts, rank, comb_base,
                            comb_records, output_res, topk_indices_res, num_tokens_res, reduce_flag_base,
                            reduce_bank, expected_reduce, topk_weights_res, gate_local_res, d_topk_w_res,
                        )

    @flyc.jit
    def launch(
        ACT, WEIGHTS, L2Y_FP8, L2Y_SCALE, TILE_TO_GROUP, NUM_TILE_BLOCKS, OUTPUT, TOPK_INDICES,
        NUM_TOKENS_PER_RANK, TOPK_WEIGHTS, GRAD_GATE, D_TOPK_W, A_SCALE, B_SCALE,
        COMBINE_PARITY, COMBINE_EXPECTED, REDUCE_EXPECTED, sym_layout, c_n: int,
        real_tiles_host: int,
        tail_blocks_host: int,
        stream: fx.Stream = fx.Stream(None),
    ):
        if const_expr(persistent_gemm):
            grid_size = gemm_base + num_persistent_cu + (0 if _no_reduce else max_tail_blocks)
        elif const_expr(real_tiles_grid):
            if const_expr(not _no_reduce and num_reduce_cu == 0):
                grid_size = gemm_base + real_tiles_host * n_blocks + tail_blocks_host
            else:
                grid_size = gemm_base + real_tiles_host * n_blocks
        else:
            grid_size = gemm_base + worst_case_tiles * n_blocks
        # bump epoch on device (combine += n_blocks, reduce += 1) before the kernel; same-stream visible
        _make_epoch_bump(1, 1)(COMBINE_PARITY, COMBINE_EXPECTED, REDUCE_EXPECTED).launch(
            grid=(1, 1, 1), block=(_BLOCK_THREADS, 1, 1), stream=stream
        )
        kern(
            ACT, WEIGHTS, L2Y_FP8, L2Y_SCALE, TILE_TO_GROUP, NUM_TILE_BLOCKS, OUTPUT, TOPK_INDICES,
            NUM_TOKENS_PER_RANK, TOPK_WEIGHTS, GRAD_GATE, D_TOPK_W, A_SCALE, B_SCALE,
            COMBINE_PARITY, COMBINE_EXPECTED, REDUCE_EXPECTED, sym_layout, c_n,
            value_attrs=make_value_attrs(waves_per_eu, agpr_alloc, "512,512"),
        ).launch(grid=(grid_size, 1, 1), block=(_BLOCK_THREADS, 1, 1), stream=stream)

    return launch


@functools.lru_cache(maxsize=32)
def _compile_reduce_only(
    out_features, combine_slots, topk, num_experts, rank, num_ranks,
    apply_weights, with_gate, reduce_blocks,
):
    """Standalone fp8 topk reduce (+ optional gate fold) for split backward STEP3."""
    reduce_fp8 = _make_topk_reduce_fp8(out_features, topk, combine_slots, apply_weights, with_gate)
    comb_records = combine_slots * out_features * 2
    gate_records = combine_slots * 4

    @flyc.kernel(known_block_size=[_BLOCK_THREADS, 1, 1])
    def reduce_kern(
        OUTPUT: fx.Tensor, TOPK_INDICES: fx.Tensor, NUM_TOKENS_PER_RANK: fx.Tensor,
        TOPK_WEIGHTS: fx.Tensor, GRAD_GATE: fx.Tensor, D_TOPK_W: fx.Tensor,
        COMBINE_PARITY: fx.Tensor, REDUCE_EXPECTED: fx.Tensor, sym_layout: SymLayout,
    ):
        thread_index = fx.thread_idx.x
        block_index, _b, _c = fx.block_idx
        combine_parity_res = create_buffer_resource(COMBINE_PARITY, max_size=True)
        reduce_expected_res = create_buffer_resource(REDUCE_EXPECTED, max_size=True)
        parity = cast(
            buffer_load(combine_parity_res, fx.Int32(0), vec_width=1, dtype=fx.T.i64()), fx.T.i32()
        )
        reduce_bank = parity * fx.Int32(combine_slots)
        expected_reduce = buffer_load(reduce_expected_res, parity, vec_width=1, dtype=fx.T.i64())
        reduce_flag_base = sym_layout.reduce_flag_ptr
        comb_base = sym_layout.comb_ptr
        gate_base = sym_layout.combine_gate_ptr
        gate_local_res = create_buffer_resource_from_addr(gate_base, num_records_bytes=gate_records)
        output_res = create_buffer_resource(OUTPUT, max_size=True)
        topk_indices_res = create_buffer_resource(TOPK_INDICES, max_size=True)
        num_tokens_res = create_buffer_resource(NUM_TOKENS_PER_RANK, max_size=True)
        topk_weights_res = create_buffer_resource(TOPK_WEIGHTS, max_size=True)
        grad_gate_res = create_buffer_resource(GRAD_GATE, max_size=True)
        d_topk_w_res = create_buffer_resource(D_TOPK_W, max_size=True)
        reduce_fp8(
            thread_index, block_index, fx.Int32(reduce_blocks),
            num_experts, rank, comb_base, comb_records, output_res, topk_indices_res,
            num_tokens_res, reduce_flag_base, reduce_bank, expected_reduce,
            topk_weights_res, gate_local_res, d_topk_w_res,
        )

    @flyc.jit
    def launch_reduce(
        OUTPUT, TOPK_INDICES, NUM_TOKENS_PER_RANK, TOPK_WEIGHTS, GRAD_GATE, D_TOPK_W,
        COMBINE_PARITY, REDUCE_EXPECTED, sym_layout,
        stream: fx.Stream = fx.Stream(None),
    ):
        reduce_kern(
            OUTPUT, TOPK_INDICES, NUM_TOKENS_PER_RANK, TOPK_WEIGHTS, GRAD_GATE, D_TOPK_W,
            COMBINE_PARITY, REDUCE_EXPECTED, sym_layout,
        ).launch(grid=(reduce_blocks, 1, 1), block=(_BLOCK_THREADS, 1, 1), stream=stream)

    return launch_reduce


_FP8_REDUCE_ONLY_COMPILED: dict = {}

_L2Y_FP8_SCRATCH: dict = {}


def grouped_gemm_combine_mxfp8_flydsl_kernel(
    x, weights_fp8, handle, group, *, topk_indices, topk_weights=None, grad_gate=None,
    x_fp8=None, x_fp8_rowwise=None, BM=256, BN=256, num_combine_cu=None, num_reduce_cu=0,
):
    """Unified fp8 grouped mxfp8 GEMM (mxfp8-quant epilogue) + FP8 combine PUSH + FP8-dequant reduce.
    ONE entry for BOTH directions (mirrors bf16 ``grouped_gemm_combine_bf16_flydsl_kernel``); the role
    is inferred from the optional args:
      * forward L2  (pass ``topk_weights``, no ``grad_gate``): ``x @ w2``, K=I, WEIGHTED reduce
        -> ``y`` [num_tokens, H] bf16 (returned ``d_topk_w`` is None).
      * backward STEP3 (pass ``grad_gate``, no ``topk_weights``): ``grad_l1 @ w1^T``, K=2I, UNWEIGHTED
        reduce (routing weight folded upstream) + gate scatter -> ``dx`` [num_tokens, H] bf16 and
        ``d_topk_w`` [combine_slots] f32.

    PURE COMPUTE: the weight comes in ALREADY prepared as ``weights_fp8 = (weight_flat, b_sp)`` (build
    once with the op-layer ``prepare_w2_fp8``; fwd on ``w2`` [G,H,I], bwd on ``w1^T`` [G,H,2I]; op-layer
    version-keyed) -- NO weight quant/preshuffle and NO caching here.

    Forward L2 requires ``x_fp8=(act_fp8, a_sp)`` from ``swiglu_mxfp8_flydsl_kernel``; there is
    no internal A quant on the forward path.

    Backward STEP3 requires ``x_fp8_rowwise=(q_row, a_sp)`` from
    ``swiglu_bwd_rowcol_dual_quant_mxfp8_flydsl`` (fused rowwise quant of ``grad_l1``). Pass
    ``x=None``; M/K/device come from ``q_row.shape``.

    Forward L2 and backward STEP3 both pass ``x=None`` when ``x_fp8`` / ``x_fp8_rowwise`` is given.

    Self-resetting: the combine_flag / reduce_flag epoch gates are double-banked + device epoch-bumped,
    so NO host flag reset / rendezvous. Always returns ``(output, d_topk_w)`` (``d_topk_w`` is None in
    the forward role)."""
    if "PT_FP8_NUM_REDUCE_CU" in os.environ:
        num_reduce_cu = int(os.environ["PT_FP8_NUM_REDUCE_CU"])
    apply_weights = topk_weights is not None
    with_gate = grad_gate is not None
    weight_flat, b_sp = weights_fp8
    tile_to_expert = handle[7]
    symm = get_symm_buffer_for_mega_moe()
    sym_layout = symm.make_sym_layout()
    if x_fp8 is not None and x_fp8_rowwise is not None:
        raise ValueError("x_fp8 and x_fp8_rowwise are mutually exclusive")
    if with_gate:
        assert x_fp8_rowwise is not None, "backward STEP3 requires pre-quantized grad_l1 via x_fp8_rowwise"
        aq, a_sp = x_fp8_rowwise
        M, K = aq.shape  # K = 2I for fc1^T dgrad
        dev = aq.device
    else:
        assert x_fp8 is not None, "forward L2 requires pre-quantized activation via x_fp8"
        aq, a_sp = x_fp8
        M, K = aq.shape  # K = I for fc2
        dev = aq.device
    H = int(sym_layout.hidden)
    G = int(sym_layout.num_experts_per_rank)
    assert M == int(sym_layout.num_max_pool_tokens)
    out_features = H
    combine_slots = int(sym_layout.combine_slots)
    num_ranks = int(sym_layout.num_ranks)
    rank = int(sym_layout.rank_idx)
    topk = int(sym_layout.num_topk)
    num_experts = int(sym_layout.num_experts)
    num_tile_blocks = symm.meta_scalars[1:2]
    num_tokens = int(symm.num_tokens)

    sk = (M, H, dev)
    scratch = _L2Y_FP8_SCRATCH.get(sk)
    if scratch is None:
        l2y_fp8 = torch.empty(M * H, dtype=torch.uint8, device=dev)
        l2y_scale = torch.empty(M * (H // 32), dtype=torch.uint8, device=dev)
        _L2Y_FP8_SCRATCH[sk] = scratch = (l2y_fp8, l2y_scale)
    l2y_fp8, l2y_scale = scratch

    output = torch.empty(num_tokens, out_features, dtype=torch.bfloat16, device=dev)
    topk_indices_d = topk_indices.contiguous().view(-1)
    # per-role tensors: an unused slot gets a size-1 dummy so the kernel's create_buffer_resource
    # has a valid (never-indexed) tensor. TOPK_WEIGHTS only read when apply_weights; GRAD_GATE /
    # D_TOPK_W only touched when with_gate.
    _dummy_f32 = torch.empty(1, dtype=torch.float32, device=dev)
    topk_weights_arg = topk_weights.contiguous().view(-1) if apply_weights else _dummy_f32
    if with_gate:
        d_topk_w = torch.empty(combine_slots, dtype=torch.float32, device=dev)
        grad_gate_arg = grad_gate.contiguous().view(-1)
    else:
        d_topk_w, grad_gate_arg = _dummy_f32, _dummy_f32

    act_flat = aq.view(torch.int8).reshape(-1)

    real_tiles_host = int(num_tile_blocks.item())
    worst_case_tiles_host = M // BM
    tail_blocks_host = max(0, worst_case_tiles_host - real_tiles_host) * (out_features // BN)

    def _run_with_cu(cu: int) -> None:
        stream = torch.cuda.current_stream()
        gemm_push_args = (
            act_flat, weight_flat, l2y_fp8, l2y_scale, tile_to_expert, num_tile_blocks, output.view(-1),
            topk_indices_d, symm.num_tokens_per_rank, topk_weights_arg, grad_gate_arg, d_topk_w, a_sp, b_sp,
            symm._combine_parity, symm._combine_expected, symm._reduce_expected,
            sym_layout, out_features, real_tiles_host, tail_blocks_host, stream,
        )
        use_split = with_gate and _SPLIT_BWD_REDUCE and not any(
            os.environ.get(k, "0") == "1"
            for k in ("PT_COMBINE_GEMM_ONLY", "PT_COMBINE_PUSH_ONLY", "PT_COMBINE_NO_REDUCE")
        )

        if use_split:
            launch_gp = _compile(
                out_features, K, M, BM, BN, int(cu), int(num_reduce_cu),
                int(combine_slots), int(topk), int(num_experts), int(rank), int(num_ranks),
                apply_weights, with_gate, num_groups=int(G), force_no_reduce=True,
            )
            gp_ck = (
                out_features, K, M, BM, BN, int(cu), int(num_reduce_cu),
                int(combine_slots), int(topk), int(num_experts), int(rank), int(num_ranks), int(G),
                apply_weights, with_gate, True,
                _COMBINE_NT_VMCNT, _COMBINE_WAVES_PER_EU, _COMBINE_REAL_TILES_GRID,
                _COMBINE_PERSISTENT_GEMM, _COMBINE_GEMM_CU, _COMBINE_NUM_XCD, _COMBINE_GROUP_M,
                _COMBINE_GROUP_N, _COMBINE_TILE_SWIZZLE,
            )
            if torch.cuda.is_current_stream_capturing():
                launch_gp(*gemm_push_args)
            else:
                compiled_gp = _FP8_COMBINE_COMPILED.get(gp_ck)
                if compiled_gp is None:
                    compiled_gp = flyc.compile(launch_gp, *gemm_push_args)
                    _FP8_COMBINE_COMPILED[gp_ck] = compiled_gp
                compiled_gp(*gemm_push_args)

            _host_rendezvous(group)

            reduce_blocks = (num_tokens + _NUM_WARPS - 1) // _NUM_WARPS
            launch_r = _compile_reduce_only(
                out_features, int(combine_slots), int(topk), int(num_experts), int(rank), int(num_ranks),
                apply_weights, with_gate, reduce_blocks,
            )
            reduce_args = (
                output.view(-1), topk_indices_d, symm.num_tokens_per_rank, topk_weights_arg,
                grad_gate_arg, d_topk_w, symm._combine_parity, symm._reduce_expected, sym_layout, stream,
            )
            r_ck = (out_features, int(combine_slots), int(topk), int(num_experts), int(rank),
                    int(num_ranks), apply_weights, with_gate, reduce_blocks)
            if torch.cuda.is_current_stream_capturing():
                launch_r(*reduce_args)
            else:
                compiled_r = _FP8_REDUCE_ONLY_COMPILED.get(r_ck)
                if compiled_r is None:
                    compiled_r = flyc.compile(launch_r, *reduce_args)
                    _FP8_REDUCE_ONLY_COMPILED[r_ck] = compiled_r
                compiled_r(*reduce_args)
            return

        launch = _compile(
            out_features, K, M, BM, BN, int(cu), int(num_reduce_cu),
            int(combine_slots), int(topk), int(num_experts), int(rank), int(num_ranks),
            apply_weights, with_gate, num_groups=int(G),
        )
        ck = (out_features, K, M, BM, BN, int(cu), int(num_reduce_cu),
              int(combine_slots), int(topk), int(num_experts), int(rank), int(num_ranks), int(G),
              apply_weights, with_gate, False,
              _COMBINE_NT_VMCNT, _COMBINE_WAVES_PER_EU,
              _COMBINE_REAL_TILES_GRID,
              _COMBINE_PERSISTENT_GEMM, _COMBINE_GEMM_CU, _COMBINE_NUM_XCD, _COMBINE_GROUP_M, _COMBINE_GROUP_N,
              _COMBINE_TILE_SWIZZLE)
        if torch.cuda.is_current_stream_capturing():
            launch(*gemm_push_args)
        else:
            compiled = _FP8_COMBINE_COMPILED.get(ck)
            if compiled is None:
                compiled = flyc.compile(launch, *gemm_push_args)
                _FP8_COMBINE_COMPILED[ck] = compiled
            compiled(*gemm_push_args)

    # Shipped per-role CU split when the caller does not pin one (fwd L2 / bwd STEP3).
    _run_with_cu(int(num_combine_cu) if num_combine_cu is not None else (32 if apply_weights else 24))
    return output, (d_topk_w if with_gate else None)
