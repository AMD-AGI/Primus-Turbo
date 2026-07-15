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
  * role GEMM ``[gemm_base, ...)``: bf16 NT tile with a quantizing epilogue (``StoreCQuantFp8``)
    -> LOCAL fp8 L2Y (write-through sc1) + E8M0; bump ``sb_l2``.

LOCAL fp8 L2Y buffers (``L2Y_FP8`` uint8 [pool*H], ``L2Y_SCALE`` uint8 [pool*H/32]); peer
``comb`` holds the packed fp8 payload + E8M0 scale (fits the bf16 comb region).
"""

import functools
import os

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr
import torch
from flydsl._mlir.dialects import vector as _vector
from flydsl.compiler.ast_rewriter import ASTRewriter
from flydsl.expr.buffer_ops import (
    buffer_load,
    buffer_store,
    create_buffer_resource,
    create_buffer_resource_from_addr,
)
from flydsl.expr.rocdl import cvt_pk_f32_fp8, cvt_pk_fp8_f32
from flydsl.expr.typing import Vector as Vec

from primus_turbo.flydsl.mega.gemm_bf16_kernel import BLOCK_K as _BF16_BLOCK_K
from primus_turbo.flydsl.mega.gemm_bf16_kernel import GEMM_TILE, _make_shared_storage
from primus_turbo.flydsl.mega.grouped_gemm_combine_bf16_kernel import _BLOCK_THREADS, _NUM_WARPS, _WARP
from primus_turbo.flydsl.mega.prims import (
    _wait_mem,
    atomic_add,
    l2_invalidate,
    ld,
    read_clock,
    spin_timed_out,
    st,
)
from primus_turbo.flydsl.mega.sym_layout import SymLayout
from primus_turbo.flydsl.mega.symm_buffer import get_symm_buffer_for_mega_moe
from primus_turbo.flydsl.mega.fp8.gemm_mxfp8_tile import (
    BLOCK_K as _MXFP8_BLOCK_K,
    gemm_mxfp8_nt_tile,
)
from primus_turbo.flydsl.utils.gemm_helper import (
    StoreCQuantMxfp8CShuffle,
    StoreCQuantMxfp8CShuffle32,
    _emit_if_then,
    make_value_attrs,
)

_WT = 16  # cache_modifier sc1 = write-through


# ─────────── GEMM epilogue: quantize f32 accumulators -> LOCAL mxfp8 L2Y (32-lane amax) ───────────
class StoreCQuantFp8:
    """Quantizing epilogue store (drop-in for StoreCBf16.store). The MFMA 32x32x16 acc has
    lane n=lane%32 owning column ``base_col+n`` and 16 rows via ``acc[r]``; a 32x32 tile spans
    exactly one 32-col mxfp8 block, so the per-row block amax = a 32-lane butterfly reduction
    of ``acc[r]``. Writes fp8 payload byte (row*H+col) + E8M0 byte (row*H/32 + col//32)."""

    def __init__(self, l2y_fp8_res, l2y_scale_res, hidden):
        self.fp8 = l2y_fp8_res
        self.scale = l2y_scale_res
        self.H = hidden
        self.lane = fx.thread_idx.x % 64

    def store(self, c_frag, base_row, base_col):
        n = self.lane % fx.Int32(32)
        grp = self.lane // fx.Int32(32)
        for ti in fx.range_constexpr(len(c_frag)):
            acc = Vec(c_frag[ti])
            for r in fx.range_constexpr(16):
                v = fx.arith.ArithValue(fx.arith._to_raw(acc[r]))
                av = fx.arith.ArithValue((v.bitcast(fx.T.i32()) & fx.Int32(0x7FFFFFFF)).bitcast(fx.T.f32()))
                for sh in (1, 2, 4, 8, 16):  # 32-lane butterfly amax (within each 32-lane half)
                    peer = fx.arith.ArithValue(av.shuffle_xor(sh, _WARP))
                    av = fx.arith.ArithValue(fx.arith.maximumf(av, peer))
                amax_bits = av.bitcast(fx.T.i32())
                t = amax_bits + fx.Int32(1 << 19)
                exp = ((t >> fx.Int32(23)) & fx.Int32(0x1FF)) - fx.Int32(127 + 8)
                exp = fx.arith.select(exp < fx.Int32(-127), fx.Int32(-127), exp)
                exp = fx.arith.select(exp > fx.Int32(128), fx.Int32(128), exp)
                biased = fx.arith.ArithValue(exp) + fx.Int32(127)
                scale = (biased << fx.Int32(23)).bitcast(fx.T.f32())
                inv = fx.Float32(1.0) / fx.arith.ArithValue(scale)
                q = fx.arith.ArithValue(v) * fx.arith.ArithValue(inv)
                neglim = fx.arith.ArithValue(fx.arith._to_raw(fx.Float32(-448.0)))
                poslim = fx.arith.ArithValue(fx.arith._to_raw(fx.Float32(448.0)))
                q = fx.arith.ArithValue(fx.arith.minimumf(fx.arith.maximumf(q, neglim), poslim))
                w = cvt_pk_fp8_f32(fx.T.i32(), q, q, fx.Int32(0), False)
                fp8_b = (fx.arith.ArithValue(w) & fx.Int32(0xFF)).trunci(fx.T.i8())
                row = base_row + fx.Int32(ti * 32) + (fx.Int32(r // 4) * fx.Int32(8)) + grp * fx.Int32(4) + fx.Int32(r % 4)
                col = base_col + n
                buffer_store(fp8_b, self.fp8, row * fx.Int32(self.H) + col, cache_modifier=_WT)

                def _emit_scale():
                    sb = fx.arith.ArithValue(biased).trunci(fx.T.i8())
                    buffer_store(sb, self.scale, row * fx.Int32(self.H // 32) + col // fx.Int32(32), cache_modifier=_WT)

                _emit_if_then(n == fx.Int32(0), _emit_scale)


# ─────────── role COMBINE: read LOCAL fp8 L2Y -> push peer packed comb (pure copy) + flags ───────────
def combine_copy_fp8_tile(
    *, thread_index, block_m_size, hidden, comb_records, H4, SC, payload_i32_total,
    l2y_fp8_res, l2y_scale_res, origin_rank_res, origin_slot_res, comb_base, signal_delta_res, barrier_base,
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

    def push_block(block_m):
        base_row = block_m * fx.Int32(block_m_size) + chunk_base
        for j in range(rows_per_warp):
            row = base_row + fx.Int32(j)
            origin = buffer_load(origin_rank_res, row, vec_width=1, dtype=fx.T.i32())

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
                barrier_addr = barrier_base + delta
                _wait_mem()
                st(barrier_addr, slot, fx.Int32(1), scope="sys")

            _emit_if_then(origin >= fx.Int32(0), _emit_row)

    return push_block


# ─────────── role REDUCE: fp8-dequant weighted topk sum (+ wait flags) ───────────
def _make_topk_reduce_fp8(hidden, topk, combine_slots):
    H4 = hidden // 4
    payload_i32_total = combine_slots * H4
    SC = hidden // 128
    words_per_lane = H4 // _WARP

    def _reduce(thread_index, base_pid, total_warps, num_experts, rank, comb_base, comb_records,
                output_res, topk_indices_res, num_tokens_res, barrier_base, topk_weights_res):
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
            for jj in fx.range_constexpr(topk):
                slot = token * fx.Int32(topk) + fx.Int32(jj)
                topk_index = buffer_load(topk_indices_res, slot, vec_width=1, dtype=fx.T.i64())
                if topk_index >= fx.Int64(0):
                    if topk_index < fx.Int64(num_experts):
                        if lane == fx.Int32(0):
                            spin_start = read_clock()
                            flag = ld(barrier_base, slot, scope="agent")
                            while flag < fx.Int32(0):
                                fx.rocdl.s_sleep(fx.Int32(1))
                                if spin_timed_out(spin_start):
                                    fx.printf("MEGA fp8 ep reduce flag timeout: rank={} token={} slot={}\n",
                                              fx.Int32(rank), token, slot)
                                    spin_start = read_clock()
                                flag = ld(barrier_base, slot, scope="agent")
            _wait_mem()

            for k in fx.range_constexpr(words_per_lane):
                w = lane + fx.Int32(k * _WARP)
                sword_idx = w // fx.Int32(32)
                shift = fx.Int32(8) * ((w // fx.Int32(8)) % fx.Int32(4))
                acc = fx.arith.constant_vector(0.0, f32_v4)
                for jj in fx.range_constexpr(topk):
                    slot = token * fx.Int32(topk) + fx.Int32(jj)
                    pw = buffer_load(comb_res, slot * fx.Int32(H4) + w, vec_width=1, dtype=fx.T.i32())
                    sw = buffer_load(
                        comb_res, fx.Int32(payload_i32_total) + slot * fx.Int32(SC) + sword_idx,
                        vec_width=1, dtype=fx.T.i32(),
                    )
                    e8 = (fx.arith.ArithValue(sw) >> shift) & fx.Int32(0xFF)
                    sf = (fx.arith.ArithValue(e8) << fx.Int32(23)).bitcast(fx.T.f32())
                    wj = buffer_load(topk_weights_res, slot, vec_width=1, dtype=fx.T.f32())
                    coef = fx.arith.ArithValue(sf) * fx.arith.ArithValue(wj)
                    lo = cvt_pk_f32_fp8(res=_v2, src=pw, word_sel=False)
                    hi = cvt_pk_f32_fp8(res=_v2, src=pw, word_sel=True)
                    deq = _vector.shuffle(lo, hi, [0, 1, 2, 3])
                    term = fx.arith.mulf(deq, _vector.broadcast(f32_v4, fx.arith._to_raw(coef)))
                    acc = fx.arith.addf(acc, term)
                buffer_store(fx.arith.trunc_f(bf16_v4, acc), output_res, token * fx.Int32(hidden) + w * fx.Int32(4))

            for jj in fx.range_constexpr(topk):
                slot = token * fx.Int32(topk) + fx.Int32(jj)
                topk_index = buffer_load(topk_indices_res, slot, vec_width=1, dtype=fx.T.i64())
                if topk_index >= fx.Int64(0):
                    if topk_index < fx.Int64(num_experts):
                        if lane == fx.Int32(0):
                            st(barrier_base, slot, fx.Int32(-1), scope="agent")
            token = token + total_warps

    return ASTRewriter.transform(_reduce)


# ─────── role REDUCE (backward STEP3): UNWEIGHTED fp8-dequant topk sum + gate-grad fold ───────
def _make_topk_reduce_fp8_bwd(hidden, topk, combine_slots):
    """The backward STEP3 reduce: same fp8-dequant topk sum as ``_make_topk_reduce_fp8`` but
    UNWEIGHTED (the routing weight was already folded into ``grad_l1`` upstream) AND it folds the
    gate gradient: ``d_topk_w[slot] = combine_gate[slot]`` masked by a valid route (mirrors the
    bf16 ``_make_topk_reduce(apply_weights=False, with_gate=True)``)."""
    H4 = hidden // 4
    payload_i32_total = combine_slots * H4
    SC = hidden // 128
    words_per_lane = H4 // _WARP

    def _reduce(thread_index, base_pid, total_warps, num_experts, rank, comb_base, comb_records,
                output_res, topk_indices_res, num_tokens_res, barrier_base, gate_local_res, d_topk_w_res):
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
            for jj in fx.range_constexpr(topk):
                slot = token * fx.Int32(topk) + fx.Int32(jj)
                topk_index = buffer_load(topk_indices_res, slot, vec_width=1, dtype=fx.T.i64())
                if topk_index >= fx.Int64(0):
                    if topk_index < fx.Int64(num_experts):
                        if lane == fx.Int32(0):
                            spin_start = read_clock()
                            flag = ld(barrier_base, slot, scope="agent")
                            while flag < fx.Int32(0):
                                fx.rocdl.s_sleep(fx.Int32(1))
                                if spin_timed_out(spin_start):
                                    fx.printf("MEGA fp8 ep bwd reduce flag timeout: rank={} token={} slot={}\n",
                                              fx.Int32(rank), token, slot)
                                    spin_start = read_clock()
                                flag = ld(barrier_base, slot, scope="agent")
            _wait_mem()

            for k in fx.range_constexpr(words_per_lane):
                w = lane + fx.Int32(k * _WARP)
                sword_idx = w // fx.Int32(32)
                shift = fx.Int32(8) * ((w // fx.Int32(8)) % fx.Int32(4))
                acc = fx.arith.constant_vector(0.0, f32_v4)
                for jj in fx.range_constexpr(topk):
                    slot = token * fx.Int32(topk) + fx.Int32(jj)
                    pw = buffer_load(comb_res, slot * fx.Int32(H4) + w, vec_width=1, dtype=fx.T.i32())
                    sw = buffer_load(
                        comb_res, fx.Int32(payload_i32_total) + slot * fx.Int32(SC) + sword_idx,
                        vec_width=1, dtype=fx.T.i32(),
                    )
                    e8 = (fx.arith.ArithValue(sw) >> shift) & fx.Int32(0xFF)
                    sf = (fx.arith.ArithValue(e8) << fx.Int32(23)).bitcast(fx.T.f32())  # UNWEIGHTED coef = scale
                    lo = cvt_pk_f32_fp8(res=_v2, src=pw, word_sel=False)
                    hi = cvt_pk_f32_fp8(res=_v2, src=pw, word_sel=True)
                    deq = _vector.shuffle(lo, hi, [0, 1, 2, 3])
                    term = fx.arith.mulf(deq, _vector.broadcast(f32_v4, fx.arith._to_raw(sf)))
                    acc = fx.arith.addf(acc, term)
                buffer_store(fx.arith.trunc_f(bf16_v4, acc), output_res, token * fx.Int32(hidden) + w * fx.Int32(4))

            for jj in fx.range_constexpr(topk):
                slot = token * fx.Int32(topk) + fx.Int32(jj)
                topk_index = buffer_load(topk_indices_res, slot, vec_width=1, dtype=fx.T.i64())
                if topk_index >= fx.Int64(0):
                    if topk_index < fx.Int64(num_experts):
                        if lane == fx.Int32(0):
                            st(barrier_base, slot, fx.Int32(-1), scope="agent")
                if lane == fx.Int32(0):
                    # d_topk_w[slot] = combine_gate[slot] for valid routes else 0 (folds the host
                    # combine_gate * (topk_idx>=0) mask into a fresh buffer).
                    gate_v = buffer_load(gate_local_res, slot, vec_width=1, dtype=fx.T.f32())
                    zero_f = fx.Float32(0.0)
                    v1 = fx.arith.select(topk_index < fx.Int64(num_experts), gate_v, zero_f)
                    d_val = fx.arith.select(topk_index >= fx.Int64(0), v1, zero_f)
                    buffer_store(d_val, d_topk_w_res, slot)
            token = token + total_warps

    return ASTRewriter.transform(_reduce)


_FP8_COMBINE_COMPILED: dict = {}


@functools.lru_cache(maxsize=64)
def _compile(
    out_features, hidden_size, num_max_pool_tokens, BLOCK_M, BLOCK_N, num_combine_cu, num_reduce_cu,
    combine_slots, topk, num_experts, rank, num_ranks, nt_vmcnt=3, waves_per_eu=2, agpr_alloc=0,
    num_groups=0,
):
    _num_groups = num_groups
    K = hidden_size
    gemm_tile = GEMM_TILE["nt"]
    assert out_features % BLOCK_N == 0
    assert num_max_pool_tokens % BLOCK_M == 0
    assert BLOCK_N % 256 == 0, "epilogue quant assumes N_TILES_B=1 (BLOCK_N a multiple of 256)"
    # PT_FP8_COMBINE_CSHUF (default on): quantize the L2Y in the CShuffle epilogue (4/2-lane
    # amax + coalesced fp8 store) instead of StoreCQuantFp8's 32-lane butterfly + scattered
    # byte stores. Needs a per-wave C_lds_shuffle (8 waves x 32x32 bf16 = 16 KB) added to the
    # bf16 GEMM's LDS (128 KB @ BLOCK=256 -> 144 KB < 160 KB).
    _cshuf = os.environ.get("PT_FP8_COMBINE_CSHUF", "1") == "1"
    _a_lds = (BLOCK_M // 2) * _BF16_BLOCK_K
    _b_lds = (BLOCK_N // 2) * _BF16_BLOCK_K
    _cshuf_n = _BLOCK_THREADS // 64 * 32 * 32  # 8 waves x 32x32 bf16

    @fx.struct
    class _SharedStorageCShuf:
        A_lds_cur_0: fx.Array[fx.BFloat16, _a_lds, 16]
        A_lds_cur_1: fx.Array[fx.BFloat16, _a_lds, 16]
        A_lds_next_0: fx.Array[fx.BFloat16, _a_lds, 16]
        A_lds_next_1: fx.Array[fx.BFloat16, _a_lds, 16]
        B_lds_cur_0: fx.Array[fx.BFloat16, _b_lds, 16]
        B_lds_cur_1: fx.Array[fx.BFloat16, _b_lds, 16]
        B_lds_next_0: fx.Array[fx.BFloat16, _b_lds, 16]
        B_lds_next_1: fx.Array[fx.BFloat16, _b_lds, 16]
        C_lds_shuffle: fx.Array[fx.BFloat16, _cshuf_n, 16]

    # PT_FP8_COMBINE_GEMM (DEFAULT "mxfp8"): run the L2 fc2 GEMM in fp8 (gemm_mxfp8_nt_tile, ~2x
    # compute) -- the shipped fc2+combine path: mxfp8 GEMM + fp8 combine PUSH. The CU sweep showed
    # the fused L2 is GEMM-role-bound (more combine CUs hurt), so a faster GEMM frees the critical
    # path. Uses the mxfp8 tile + CShuffle mxfp8-quant epilogue; inputs become pre-quantized fp8
    # act/w2 + preshuffled scales. Set PT_FP8_COMBINE_GEMM=bf16 to force the (slower) bf16-GEMM path
    # for comparison.
    _mxgemm = os.environ.get("PT_FP8_COMBINE_GEMM", "mxfp8") == "mxfp8"
    # PT_COMBINE_NO_REDUCE (debug/isolation, default off): compile out the topk reduce role
    # (dedicated + empty-block) so the kernel measures ONLY the GEMM (produce local fp8 L2Y) +
    # combine PUSH (XGMI copy to peer). Isolates the fp8 byte-lever on the produce+transmit half
    # from the cross-rank reduce/dequant + producer-consumer sync. Requires num_reduce_cu==0.
    _no_reduce = os.environ.get("PT_COMBINE_NO_REDUCE", "0") == "1"
    _mx_a_lds = (BLOCK_M // 2) * _MXFP8_BLOCK_K  # fp8 elems/buffer
    _mx_b_lds = (BLOCK_N // 2) * _MXFP8_BLOCK_K
    _mx_cshuf_n = _BLOCK_THREADS // 64 * 16 * (BLOCK_N // 128 * 16)  # 8 waves x 16 x Cc bf16

    @fx.struct
    class _SharedStorageMxGemm:
        A_lds_cur_0: fx.Array[fx.Float8E4M3FN, _mx_a_lds, 16]
        A_lds_cur_1: fx.Array[fx.Float8E4M3FN, _mx_a_lds, 16]
        A_lds_next_0: fx.Array[fx.Float8E4M3FN, _mx_a_lds, 16]
        A_lds_next_1: fx.Array[fx.Float8E4M3FN, _mx_a_lds, 16]
        B_lds_cur_0: fx.Array[fx.Float8E4M3FN, _mx_b_lds, 16]
        B_lds_cur_1: fx.Array[fx.Float8E4M3FN, _mx_b_lds, 16]
        B_lds_next_0: fx.Array[fx.Float8E4M3FN, _mx_b_lds, 16]
        B_lds_next_1: fx.Array[fx.Float8E4M3FN, _mx_b_lds, 16]
        C_lds_shuffle: fx.Array[fx.BFloat16, _mx_cshuf_n, 16]

    if _mxgemm:
        SharedStorage = _SharedStorageMxGemm
    else:
        SharedStorage = _SharedStorageCShuf if _cshuf else _make_shared_storage(BLOCK_M, BLOCK_N)
    n_blocks = out_features // BLOCK_N
    worst_case_tiles = num_max_pool_tokens // BLOCK_M
    gemm_grid_blocks = worst_case_tiles * n_blocks
    comb_records = combine_slots * out_features * 2
    delta_records = num_ranks * 8
    pool_records = num_max_pool_tokens * 4
    dedicated_reduce_warps = num_reduce_cu * _NUM_WARPS
    gemm_base = num_combine_cu + num_reduce_cu
    H4 = out_features // 4
    SC = out_features // 128
    payload_i32_total = combine_slots * H4
    l2y_fp8_bytes = num_max_pool_tokens * out_features
    l2y_scale_bytes = num_max_pool_tokens * (out_features // 32)
    reduce_fp8 = _make_topk_reduce_fp8(out_features, topk, combine_slots)

    @flyc.kernel(known_block_size=[_BLOCK_THREADS, 1, 1])
    def kern(
        ACT: fx.Tensor, WEIGHTS: fx.Tensor, L2Y_FP8: fx.Tensor, L2Y_SCALE: fx.Tensor,
        TILE_TO_GROUP: fx.Tensor, NUM_TILE_BLOCKS: fx.Tensor, OUTPUT: fx.Tensor, TOPK_INDICES: fx.Tensor,
        NUM_TOKENS_PER_RANK: fx.Tensor, TOPK_WEIGHTS: fx.Tensor, A_SCALE: fx.Tensor, B_SCALE: fx.Tensor,
        sym_layout: SymLayout, c_n: fx.Int32,
    ):
        thread_index = fx.thread_idx.x
        block_index, _b, _c = fx.block_idx
        combine_cu = fx.Int32(num_combine_cu)
        reduce_cu = fx.Int32(num_reduce_cu)

        lds = fx.SharedAllocator().allocate(SharedStorage).peek()

        sb_l2_base = sym_layout.sb_l2_ptr
        comb_base = sym_layout.comb_ptr
        barrier_base = sym_layout.barrier_local_ptr
        l2y_fp8_res = create_buffer_resource(L2Y_FP8, max_size=True)
        l2y_scale_res = create_buffer_resource(L2Y_SCALE, max_size=True)
        signal_delta_res = create_buffer_resource_from_addr(
            sym_layout.signal_offsets_ptr, num_records_bytes=delta_records
        )
        origin_rank_res = create_buffer_resource_from_addr(sym_layout.origin_rank_ptr, num_records_bytes=pool_records)
        origin_slot_res = create_buffer_resource_from_addr(sym_layout.origin_slot_ptr, num_records_bytes=pool_records)

        group_resource = create_buffer_resource(TILE_TO_GROUP, max_size=True)
        num_tile_blocks_res = create_buffer_resource(NUM_TILE_BLOCKS, max_size=True)
        output_res = create_buffer_resource(OUTPUT, max_size=True)
        topk_indices_res = create_buffer_resource(TOPK_INDICES, max_size=True)
        num_tokens_res = create_buffer_resource(NUM_TOKENS_PER_RANK, max_size=True)
        topk_weights_res = create_buffer_resource(TOPK_WEIGHTS, max_size=True)
        real_tiles = buffer_load(num_tile_blocks_res, fx.Int32(0), vec_width=1, dtype=fx.T.i32())

        if block_index < combine_cu:
            push_block = combine_copy_fp8_tile(
                thread_index=thread_index, block_m_size=BLOCK_M, hidden=out_features, comb_records=comb_records,
                H4=H4, SC=SC, payload_i32_total=payload_i32_total, l2y_fp8_res=l2y_fp8_res,
                l2y_scale_res=l2y_scale_res, origin_rank_res=origin_rank_res, origin_slot_res=origin_slot_res,
                comb_base=comb_base, signal_delta_res=signal_delta_res, barrier_base=barrier_base,
            )
            local_count = (real_tiles - block_index + combine_cu - fx.Int32(1)) // combine_cu
            for tile_iter in range(local_count):
                block_m = block_index + tile_iter * combine_cu
                if thread_index == fx.Int32(0):
                    spin_start = read_clock()
                    sig = ld(sb_l2_base, block_m, scope="agent")
                    while sig < fx.Int32(n_blocks):
                        fx.rocdl.s_sleep(fx.Int32(2))
                        if spin_timed_out(spin_start):
                            fx.printf("MEGA fp8 ep combine gate timeout: block={} sig={}\n", block_m, sig)
                            spin_start = read_clock()
                        sig = ld(sb_l2_base, block_m, scope="agent")
                    st(sb_l2_base, block_m, fx.Int32(0), scope="agent")
                fx.gpu.barrier()
                l2_invalidate()
                push_block(block_m)
            fx.rocdl.s_waitcnt(0)
        else:
            # gemm_tile_index/block_m/block_n computed ONCE here (unconditional) so they always
            # carry a value before the dynamic ifs below (the scf.if rewriter cannot yield a
            # None-initialized var assigned only inside a branch). Negative for dedicated-reduce
            # blocks (block_index in [combine_cu, gemm_base)) -> harmless, unused on that path.
            gemm_tile_index = block_index - fx.Int32(gemm_base)
            block_m = gemm_tile_index // fx.Int32(n_blocks)
            block_n = gemm_tile_index % fx.Int32(n_blocks)

            def _do_gemm_tile(block_m, block_n):
                c_m_const = fx.Int32(num_max_pool_tokens)
                group_index = buffer_load(group_resource, block_m, vec_width=1, dtype=fx.T.i32())
                if const_expr(_mxgemm):
                    # fp8 L2 GEMM (gemm_mxfp8_nt_tile, preshuffled scales) + CShuffle mxfp8-quant
                    # epilogue. A_SCALE / B_SCALE are the preshuffled scale TENSORS (ScaleS2R /
                    # ScaleBComb build their own resources).
                    store_c = StoreCQuantMxfp8CShuffle(
                        L2Y_FP8, L2Y_SCALE, c_m_const, out_features,
                        lambda i, j: i * (BLOCK_N // 128) + j, BLOCK_M // 64, BLOCK_N // 128,
                        fx.BFloat16, lds.C_lds_shuffle, thread_index // fx.Int32(64),
                    )
                    gemm_mxfp8_nt_tile(
                        ACT, A_SCALE, WEIGHTS, B_SCALE, ACT, c_m_const, c_n, lds, block_m, block_n,
                        K=K, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, G=_num_groups, group_idx=group_index,
                        out_fp16=False, nt_vmcnt=nt_vmcnt, preshuffled=True, store_c=store_c,
                    )
                else:
                    group_base = group_index * fx.Int32(K) * c_n
                    if const_expr(_cshuf):
                        store_c = StoreCQuantMxfp8CShuffle32(
                            l2y_fp8_res, l2y_scale_res, out_features, lds.C_lds_shuffle,
                            thread_index // fx.Int32(64),
                        )
                    else:
                        store_c = StoreCQuantFp8(l2y_fp8_res, l2y_scale_res, out_features)
                    gemm_tile(
                        ACT, WEIGHTS, ACT, c_m_const, c_n, lds, block_m, block_n,
                        K=K, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, out_fp16=False, nt_vmcnt=nt_vmcnt,
                        b_group_base=group_base, store_c=store_c,
                    )
                fx.rocdl.s_waitcnt(0)
                fx.gpu.barrier()
                _emit_if_then(
                    thread_index == fx.Int32(0),
                    lambda: atomic_add(sb_l2_base, block_m, fx.Int32(1), scope="agent"),
                )

            if const_expr(_no_reduce):
                # ISOLATION (PT_COMBINE_NO_REDUCE): no reduce role. num_reduce_cu==0 so
                # gemm_base==combine_cu; all else-blocks are GEMM tiles (real -> produce local fp8
                # L2Y + signal; empty -> idle). Measures GEMM produce + combine PUSH only.
                if block_m < real_tiles:
                    _do_gemm_tile(block_m, block_n)
            else:
                if block_index < combine_cu + reduce_cu:
                    reduce_fp8(
                        thread_index, block_index - combine_cu, fx.Int32(dedicated_reduce_warps),
                        num_experts, rank, comb_base, comb_records, output_res, topk_indices_res,
                        num_tokens_res, barrier_base, topk_weights_res,
                    )
                else:
                    if block_m < real_tiles:
                        _do_gemm_tile(block_m, block_n)
                    else:
                        empty_ordinal = gemm_tile_index - real_tiles * fx.Int32(n_blocks)
                        total_empty_warps = (
                            fx.Int32(gemm_grid_blocks) - real_tiles * fx.Int32(n_blocks)
                        ) * fx.Int32(_NUM_WARPS)
                        reduce_fp8(
                            thread_index, empty_ordinal, total_empty_warps, num_experts, rank, comb_base,
                            comb_records, output_res, topk_indices_res, num_tokens_res, barrier_base, topk_weights_res,
                        )

    @flyc.jit
    def launch(
        ACT, WEIGHTS, L2Y_FP8, L2Y_SCALE, TILE_TO_GROUP, NUM_TILE_BLOCKS, OUTPUT, TOPK_INDICES,
        NUM_TOKENS_PER_RANK, TOPK_WEIGHTS, A_SCALE, B_SCALE, sym_layout, c_n: int,
        stream: fx.Stream = fx.Stream(None),
    ):
        grid_size = gemm_base + worst_case_tiles * n_blocks
        kern(
            ACT, WEIGHTS, L2Y_FP8, L2Y_SCALE, TILE_TO_GROUP, NUM_TILE_BLOCKS, OUTPUT, TOPK_INDICES,
            NUM_TOKENS_PER_RANK, TOPK_WEIGHTS, A_SCALE, B_SCALE, sym_layout, c_n,
            value_attrs=make_value_attrs(waves_per_eu, agpr_alloc, "512,512"),
        ).launch(grid=(grid_size, 1, 1), block=(_BLOCK_THREADS, 1, 1), stream=stream)

    return launch


@functools.lru_cache(maxsize=64)
def _compile_bwd(
    out_features, hidden_size, num_max_pool_tokens, BLOCK_M, BLOCK_N, num_combine_cu, num_reduce_cu,
    combine_slots, topk, num_experts, rank, num_ranks, num_groups, nt_vmcnt=3, waves_per_eu=2, agpr_alloc=0,
):
    """Backward STEP3 = mxfp8 fc1-dgrad GEMM (CShuffle mxfp8-quant epilogue -> local fp8 dx pool) +
    FP8 combine PUSH (+ gate scatter) + UNWEIGHTED fp8-dequant reduce (+ d_topk_w fold). The fp8
    analog of the current bf16-PUSH ``grouped_gemm_combine_mxfp8_bwd``, mirroring the production
    forward ``grouped_gemm_combine_fp8`` (always the mxfp8-GEMM path). ``ACT`` = grad_l1 fp8,
    ``WEIGHTS`` = w1^T fp8; the reduce is unweighted (routing weight folded into grad_l1 upstream)."""
    K = hidden_size
    assert out_features % BLOCK_N == 0
    assert num_max_pool_tokens % BLOCK_M == 0
    assert BLOCK_N % 256 == 0, "epilogue quant assumes N_TILES_B=1 (BLOCK_N a multiple of 256)"
    _mx_a_lds = (BLOCK_M // 2) * _MXFP8_BLOCK_K
    _mx_b_lds = (BLOCK_N // 2) * _MXFP8_BLOCK_K
    _mx_cshuf_n = _BLOCK_THREADS // 64 * 16 * (BLOCK_N // 128 * 16)

    @fx.struct
    class _SharedStorageMxGemm:
        A_lds_cur_0: fx.Array[fx.Float8E4M3FN, _mx_a_lds, 16]
        A_lds_cur_1: fx.Array[fx.Float8E4M3FN, _mx_a_lds, 16]
        A_lds_next_0: fx.Array[fx.Float8E4M3FN, _mx_a_lds, 16]
        A_lds_next_1: fx.Array[fx.Float8E4M3FN, _mx_a_lds, 16]
        B_lds_cur_0: fx.Array[fx.Float8E4M3FN, _mx_b_lds, 16]
        B_lds_cur_1: fx.Array[fx.Float8E4M3FN, _mx_b_lds, 16]
        B_lds_next_0: fx.Array[fx.Float8E4M3FN, _mx_b_lds, 16]
        B_lds_next_1: fx.Array[fx.Float8E4M3FN, _mx_b_lds, 16]
        C_lds_shuffle: fx.Array[fx.BFloat16, _mx_cshuf_n, 16]

    SharedStorage = _SharedStorageMxGemm
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
    _no_reduce = os.environ.get("PT_COMBINE_NO_REDUCE", "0") == "1"
    # PT_COMBINE_GEMM_ONLY (isolation): combine PUSH does 0 tiles + reduce compiled out -> the kernel
    # runs ONLY the mxfp8 fc1-dgrad GEMM role (+ CShuffle fp8 epilogue). Measures the GEMM-role wall
    # (no cross-rank comm, no reduce) -> INCORRECT output, timing only.
    _gemm_only = os.environ.get("PT_COMBINE_GEMM_ONLY", "0") == "1"
    # PT_COMBINE_PUSH_ONLY (isolation): combine PUSH runs but SKIPS the GEMM-done sb_l2 gate (GEMM +
    # reduce idle), so it just XGMI-copies whatever's in the local fp8 L2Y to the peer comb + flags.
    # Measures the combine-PUSH wall (cross-rank byte cost) -> INCORRECT output, timing only.
    _push_only = os.environ.get("PT_COMBINE_PUSH_ONLY", "0") == "1"
    _no_reduce = _no_reduce or _gemm_only or _push_only
    reduce_fp8_bwd = _make_topk_reduce_fp8_bwd(out_features, topk, combine_slots)

    @flyc.kernel(known_block_size=[_BLOCK_THREADS, 1, 1])
    def kern(
        ACT: fx.Tensor, WEIGHTS: fx.Tensor, L2Y_FP8: fx.Tensor, L2Y_SCALE: fx.Tensor,
        TILE_TO_GROUP: fx.Tensor, NUM_TILE_BLOCKS: fx.Tensor, OUTPUT: fx.Tensor, TOPK_INDICES: fx.Tensor,
        NUM_TOKENS_PER_RANK: fx.Tensor, GRAD_GATE: fx.Tensor, D_TOPK_W: fx.Tensor,
        A_SCALE: fx.Tensor, B_SCALE: fx.Tensor, sym_layout: SymLayout, c_n: fx.Int32,
    ):
        thread_index = fx.thread_idx.x
        block_index, _b, _c = fx.block_idx
        combine_cu = fx.Int32(num_combine_cu)
        reduce_cu = fx.Int32(num_reduce_cu)
        lds = fx.SharedAllocator().allocate(SharedStorage).peek()

        sb_l2_base = sym_layout.sb_l2_ptr
        comb_base = sym_layout.comb_ptr
        barrier_base = sym_layout.barrier_local_ptr
        gate_base = sym_layout.combine_gate_ptr
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
        grad_gate_res = create_buffer_resource(GRAD_GATE, max_size=True)
        d_topk_w_res = create_buffer_resource(D_TOPK_W, max_size=True)
        real_tiles = buffer_load(num_tile_blocks_res, fx.Int32(0), vec_width=1, dtype=fx.T.i32())

        if block_index < combine_cu:
            push_block = combine_copy_fp8_tile(
                thread_index=thread_index, block_m_size=BLOCK_M, hidden=out_features, comb_records=comb_records,
                H4=H4, SC=SC, payload_i32_total=payload_i32_total, l2y_fp8_res=l2y_fp8_res,
                l2y_scale_res=l2y_scale_res, origin_rank_res=origin_rank_res, origin_slot_res=origin_slot_res,
                comb_base=comb_base, signal_delta_res=signal_delta_res, barrier_base=barrier_base,
                with_gate=True, grad_gate_res=grad_gate_res, gate_base=gate_base,
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
                        sig = ld(sb_l2_base, block_m, scope="agent")
                        while sig < fx.Int32(n_blocks):
                            fx.rocdl.s_sleep(fx.Int32(2))
                            if spin_timed_out(spin_start):
                                fx.printf("MEGA fp8 ep bwd combine gate timeout: block={} sig={}\n", block_m, sig)
                                spin_start = read_clock()
                            sig = ld(sb_l2_base, block_m, scope="agent")
                        st(sb_l2_base, block_m, fx.Int32(0), scope="agent")
                    fx.gpu.barrier()
                    l2_invalidate()
                push_block(block_m)
            fx.rocdl.s_waitcnt(0)
        else:
            gemm_tile_index = block_index - fx.Int32(gemm_base)
            block_m = gemm_tile_index // fx.Int32(n_blocks)
            block_n = gemm_tile_index % fx.Int32(n_blocks)

            def _do_gemm_tile(block_m, block_n):
                c_m_const = fx.Int32(num_max_pool_tokens)
                group_index = buffer_load(group_resource, block_m, vec_width=1, dtype=fx.T.i32())
                store_c = StoreCQuantMxfp8CShuffle(
                    L2Y_FP8, L2Y_SCALE, c_m_const, out_features,
                    lambda i, j: i * (BLOCK_N // 128) + j, BLOCK_M // 64, BLOCK_N // 128,
                    fx.BFloat16, lds.C_lds_shuffle, thread_index // fx.Int32(64),
                )
                gemm_mxfp8_nt_tile(
                    ACT, A_SCALE, WEIGHTS, B_SCALE, ACT, c_m_const, c_n, lds, block_m, block_n,
                    K=K, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, G=num_groups, group_idx=group_index,
                    out_fp16=False, nt_vmcnt=nt_vmcnt, preshuffled=True, store_c=store_c,
                )
                fx.rocdl.s_waitcnt(0)
                fx.gpu.barrier()
                _emit_if_then(
                    thread_index == fx.Int32(0),
                    lambda: atomic_add(sb_l2_base, block_m, fx.Int32(1), scope="agent"),
                )

            if const_expr(_no_reduce):
                if not _push_only:  # PUSH_ONLY: GEMM role idle too (only the combine PUSH runs)
                    if block_m < real_tiles:
                        _do_gemm_tile(block_m, block_n)
            else:
                if block_index < combine_cu + reduce_cu:
                    reduce_fp8_bwd(
                        thread_index, block_index - combine_cu, fx.Int32(dedicated_reduce_warps),
                        num_experts, rank, comb_base, comb_records, output_res, topk_indices_res,
                        num_tokens_res, barrier_base, gate_local_res, d_topk_w_res,
                    )
                else:
                    if block_m < real_tiles:
                        _do_gemm_tile(block_m, block_n)
                    else:
                        empty_ordinal = gemm_tile_index - real_tiles * fx.Int32(n_blocks)
                        total_empty_warps = (
                            fx.Int32(gemm_grid_blocks) - real_tiles * fx.Int32(n_blocks)
                        ) * fx.Int32(_NUM_WARPS)
                        reduce_fp8_bwd(
                            thread_index, empty_ordinal, total_empty_warps, num_experts, rank, comb_base,
                            comb_records, output_res, topk_indices_res, num_tokens_res, barrier_base,
                            gate_local_res, d_topk_w_res,
                        )

    @flyc.jit
    def launch(
        ACT, WEIGHTS, L2Y_FP8, L2Y_SCALE, TILE_TO_GROUP, NUM_TILE_BLOCKS, OUTPUT, TOPK_INDICES,
        NUM_TOKENS_PER_RANK, GRAD_GATE, D_TOPK_W, A_SCALE, B_SCALE, sym_layout, c_n: int,
        stream: fx.Stream = fx.Stream(None),
    ):
        grid_size = gemm_base + worst_case_tiles * n_blocks
        kern(
            ACT, WEIGHTS, L2Y_FP8, L2Y_SCALE, TILE_TO_GROUP, NUM_TILE_BLOCKS, OUTPUT, TOPK_INDICES,
            NUM_TOKENS_PER_RANK, GRAD_GATE, D_TOPK_W, A_SCALE, B_SCALE, sym_layout, c_n,
            value_attrs=make_value_attrs(waves_per_eu, agpr_alloc, "512,512"),
        ).launch(grid=(grid_size, 1, 1), block=(_BLOCK_THREADS, 1, 1), stream=stream)

    return launch


_L2Y_FP8_SCRATCH: dict = {}
_L2_BSP_CACHE: dict = {}  # (w2 data_ptr, G, H, I) -> (fp8 w2 int8 flat, preshuffled b_sp) [static weights]


def prepare_w2_fp8(l2_weights):
    """Prepare the L2 fc2 weight for the fp8 combine: grouped mxfp8 quant (FlyDSL) + scale
    preshuffle (ScaleBComb layout) + int8 flat -> ``(weight_flat int8 [G*H*I], b_sp int32)``,
    exactly the two operands the mxfp8 combine GEMM consumes. Static per weight version, so a
    stateful holder (``MegaMoEFP8``) computes this ONCE per ``optim.step`` and passes it as
    ``w2_fp8`` -- the combine then does NO per-call weight quant OR preshuffle."""
    from primus_turbo.flydsl.mega.fp8.quant import quantize_grouped_weight_mxfp8_flydsl
    from primus_turbo.flydsl.mega.fp8.quant_flydsl import preshuffle_b_scale

    G, H, I = l2_weights.shape
    w2q, w2s = quantize_grouped_weight_mxfp8_flydsl(l2_weights)
    b_sp = preshuffle_b_scale(w2s, G, H, I)
    weight_flat = w2q.reshape(G * H, I).contiguous().view(torch.int8).reshape(-1)
    return weight_flat, b_sp


def grouped_gemm_combine_fp8(
    act, l2_weights, handle, group, *, topk_indices, topk_weights, BM=256, BN=256,
    num_combine_cu=48, num_reduce_cu=0, w2_fp8=None,
):
    """Fused grouped BF16 GEMM (mxfp8 epilogue quant) + FP8 combine PUSH + FP8-dequant reduce.

    ``act`` [M, I] bf16 (M = num_max_pool_tokens), ``l2_weights`` [G, H, I] bf16 (NT). The GEMM
    epilogue quantizes -> LOCAL fp8 L2Y; combine pure-copies fp8 -> peer; reduce dequants ->
    ``y`` [num_tokens, H] bf16. Caller inits ``symm.barrier_local``=-1 and ``symm.sb_l2``=0."""
    assert act.dtype == torch.bfloat16 and l2_weights.dtype == torch.bfloat16
    tile_to_expert = handle[7]
    symm = get_symm_buffer_for_mega_moe()
    sym_layout = symm.make_sym_layout()
    M, I = act.shape
    G, H, Iw = l2_weights.shape
    assert Iw == I and H == int(sym_layout.hidden) and M == int(sym_layout.num_max_pool_tokens)
    out_features = H
    combine_slots = int(sym_layout.combine_slots)
    num_ranks = int(sym_layout.num_ranks)
    rank = int(sym_layout.rank_idx)
    topk = int(sym_layout.num_topk)
    num_experts = int(sym_layout.num_experts)
    num_tile_blocks = symm.meta_scalars[1:2]
    num_tokens = int(symm.num_tokens)
    dev = act.device

    sk = (M, H, dev)
    scratch = _L2Y_FP8_SCRATCH.get(sk)
    if scratch is None:
        l2y_fp8 = torch.empty(M * H, dtype=torch.uint8, device=dev)
        l2y_scale = torch.empty(M * (H // 32), dtype=torch.uint8, device=dev)
        _L2Y_FP8_SCRATCH[sk] = scratch = (l2y_fp8, l2y_scale)
    l2y_fp8, l2y_scale = scratch

    output = torch.empty(num_tokens, out_features, dtype=torch.bfloat16, device=dev)
    topk_indices_d = topk_indices.contiguous().view(-1)
    topk_weights_d = topk_weights.contiguous().view(-1)

    _mxgemm = os.environ.get("PT_FP8_COMBINE_GEMM", "mxfp8") == "mxfp8"
    if _mxgemm:
        # fp8 L2 GEMM path: quantize act (rowwise mxfp8, preshuffled a_sp); w2 comes pre-prepared
        # from the caller (prepare_w2_fp8) or the internal version-keyed cache. ACT/WEIGHTS become
        # fp8 int8 views; A/B_SCALE = preshuffled i32.
        from primus_turbo.flydsl.mega.fp8.quant_flydsl import quantize_rowwise_mxfp8_flydsl

        aq, a_sp = quantize_rowwise_mxfp8_flydsl(act.contiguous(), preshuffle=True)
        act_flat = aq.view(torch.int8).reshape(-1)
        if w2_fp8 is not None:
            # caller (MegaMoEFP8) owns + version-maintains the FULL prepared w2 (quant + scale
            # preshuffle + int8 flat) via prepare_w2_fp8 -> no per-call quant OR preshuffle here.
            weight_flat, b_sp = w2_fp8
        else:
            # STATIC-weight cache. Key MUST include _version: in-place optim.step() updates keep the
            # same data_ptr, so a data_ptr-only key would return STALE fp8 w2 (wrong weights) in real
            # training. _version bumps on every in-place update -> invalidates. (id/data_ptr +
            # _version is the Rule-11 W1 pattern; real-training gain is grad-accum-only.) Caches BOTH
            # the quant and the preshuffle (prepare_w2_fp8) so neither is redone per call.
            _bk = (l2_weights.data_ptr(), getattr(l2_weights, "_version", 0), G, H, I)
            ent = _L2_BSP_CACHE.get(_bk)
            if ent is None:
                w2q_flat, b_sp = prepare_w2_fp8(l2_weights)
                _L2_BSP_CACHE[_bk] = ent = (w2q_flat, b_sp)
            weight_flat, b_sp = ent
        a_scale_arg, b_scale_arg, ng = a_sp, b_sp, int(G)
    else:
        act_flat = act.contiguous().view(-1)
        weight_flat = l2_weights.reshape(G * H, I).contiguous().view(-1)
        _dummy = torch.ones(1, dtype=torch.int32, device=dev)
        a_scale_arg, b_scale_arg, ng = _dummy, _dummy, 0

    launch = _compile(
        out_features, I, M, BM, BN, int(num_combine_cu), int(num_reduce_cu),
        int(combine_slots), int(topk), int(num_experts), int(rank), int(num_ranks), num_groups=ng,
    )
    args = (
        act_flat, weight_flat, l2y_fp8, l2y_scale, tile_to_expert, num_tile_blocks, output.view(-1),
        topk_indices_d, symm.num_tokens_per_rank, topk_weights_d, a_scale_arg, b_scale_arg,
        sym_layout, out_features, torch.cuda.current_stream(),
    )
    ck = (out_features, I, M, BM, BN, int(num_combine_cu), int(num_reduce_cu),
          int(combine_slots), int(topk), int(num_experts), int(rank), int(num_ranks), ng)
    if torch.cuda.is_current_stream_capturing():
        launch(*args)
    else:
        compiled = _FP8_COMBINE_COMPILED.get(ck)
        if compiled is None:
            compiled = flyc.compile(launch, *args)
            _FP8_COMBINE_COMPILED[ck] = compiled
        compiled(*args)
    return output


def grouped_gemm_combine_fp8_bwd(
    grad_l1, w1t, handle, group, *, topk_indices, grad_gate, BM=256, BN=256,
    num_combine_cu=16, num_reduce_cu=0, w1t_fp8=None,
):
    """Backward STEP3, FP8-PUSH: mxfp8 fc1-dgrad (``grad_l1 @ w1t^T``) with a CShuffle mxfp8-quant
    epilogue -> LOCAL fp8 dx pool -> FP8 combine PUSH (+ gate scatter) -> UNWEIGHTED fp8-dequant
    reduce (+ ``d_topk_w`` fold). The fp8-PUSH analog of ``grouped_gemm_combine_mxfp8_bwd`` (which
    pushes bf16 dx), mirroring the production forward ``grouped_gemm_combine_fp8``.

    ``grad_l1`` [M, 2I] bf16 (M = num_max_pool_tokens; routing weight already folded upstream),
    ``w1t`` [G, H, 2I] bf16 (w1 transposed, contraction 2I), ``grad_gate`` [M] f32. Returns
    ``(dx [num_tokens, H] bf16, d_topk_w [combine_slots] f32)``. Caller inits ``symm.barrier_local``
    =-1, ``symm.sb_l2``=0, ``symm.combine_gate``=0 (cross-rank barrier'd) before calling."""
    from primus_turbo.flydsl.mega.fp8.quant_flydsl import quantize_rowwise_mxfp8_flydsl

    assert grad_l1.dtype == torch.bfloat16 and w1t.dtype == torch.bfloat16
    tile_to_expert = handle[7]
    symm = get_symm_buffer_for_mega_moe()
    sym_layout = symm.make_sym_layout()
    M, K = grad_l1.shape  # K = 2I (fc1 gate||up contraction)
    G, H, Kw = w1t.shape
    assert Kw == K, f"w1^T K={Kw} != grad_l1 K={K}"
    assert H == int(sym_layout.hidden) and M == int(sym_layout.num_max_pool_tokens)
    out_features = H
    combine_slots = int(sym_layout.combine_slots)
    num_ranks = int(sym_layout.num_ranks)
    rank = int(sym_layout.rank_idx)
    topk = int(sym_layout.num_topk)
    num_experts = int(sym_layout.num_experts)
    num_tile_blocks = symm.meta_scalars[1:2]
    num_tokens = int(symm.num_tokens)
    dev = grad_l1.device

    sk = (M, H, dev)
    scratch = _L2Y_FP8_SCRATCH.get(sk)
    if scratch is None:
        l2y_fp8 = torch.empty(M * H, dtype=torch.uint8, device=dev)
        l2y_scale = torch.empty(M * (H // 32), dtype=torch.uint8, device=dev)
        _L2Y_FP8_SCRATCH[sk] = scratch = (l2y_fp8, l2y_scale)
    l2y_fp8, l2y_scale = scratch

    output = torch.empty(num_tokens, out_features, dtype=torch.bfloat16, device=dev)
    d_topk_w = torch.empty(combine_slots, dtype=torch.float32, device=dev)
    topk_indices_d = topk_indices.contiguous().view(-1)
    grad_gate_d = grad_gate.contiguous().view(-1)

    # act = grad_l1 rowwise mxfp8 (preshuffled a_sp); w1t prepared (grouped mxfp8 quant + preshuffle
    # b_sp). Mirror the forward mxgemm host path; E4M3 (dx PUSH format = the CShuffle epilogue's E4M3).
    aq, a_sp = quantize_rowwise_mxfp8_flydsl(grad_l1.contiguous(), preshuffle=True)
    act_flat = aq.view(torch.int8).reshape(-1)
    weight_flat, b_sp = w1t_fp8 if w1t_fp8 is not None else prepare_w2_fp8(w1t)  # prepare is shape-generic

    launch = _compile_bwd(
        out_features, K, M, BM, BN, int(num_combine_cu), int(num_reduce_cu),
        int(combine_slots), int(topk), int(num_experts), int(rank), int(num_ranks), int(G),
    )
    args = (
        act_flat, weight_flat, l2y_fp8, l2y_scale, tile_to_expert, num_tile_blocks, output.view(-1),
        topk_indices_d, symm.num_tokens_per_rank, grad_gate_d, d_topk_w, a_sp, b_sp,
        sym_layout, out_features, torch.cuda.current_stream(),
    )
    ck = (out_features, K, M, BM, BN, int(num_combine_cu), int(num_reduce_cu),
          int(combine_slots), int(topk), int(num_experts), int(rank), int(num_ranks), int(G), "bwd")
    if torch.cuda.is_current_stream_capturing():
        launch(*args)
    else:
        compiled = _FP8_COMBINE_COMPILED.get(ck)
        if compiled is None:
            compiled = flyc.compile(launch, *args)
            _FP8_COMBINE_COMPILED[ck] = compiled
        compiled(*args)
    return output, d_topk_w
