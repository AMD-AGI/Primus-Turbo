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
from types import SimpleNamespace

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl._mlir.dialects import llvm as _llvm
from flydsl._mlir.dialects import vector as _vector
from flydsl.compiler.ast_rewriter import ASTRewriter
from flydsl.expr import buffer_ops as _buffer_ops
from flydsl.expr import const_expr, range_constexpr, rocdl
from flydsl.expr.buffer_ops import (
    buffer_load,
    buffer_store,
    create_buffer_resource,
    create_buffer_resource_from_addr,
)
from flydsl.expr.rocdl import cvt_pk_f32_fp8
from flydsl.expr.typing import Vector as Vec

from primus_turbo.flydsl.mega.ep_intranode import _BLOCK_THREADS, _NUM_WARPS, _WARP
from primus_turbo.flydsl.mega.fp8.dispatch_grouped_gemm_mxfp8_kernel import (
    _H_NUM_TILE_BLOCKS,
    _H_ORIGIN_RANK,
    _H_ORIGIN_SLOT,
)
from primus_turbo.flydsl.mega.fp8.gemm_mxfp8_tile import (
    BLOCK_K as _MXFP8_BLOCK_K,
)
from primus_turbo.flydsl.mega.fp8.gemm_mxfp8_tile import (
    emit_gemm_mxfp8_nt_tile,
)
from primus_turbo.flydsl.mega.fp8.prims import (
    _wait_mem,
    l2_invalidate,
    ld,
    st,
)
from primus_turbo.flydsl.mega.fp8.symm_buffer import SymLayout, get_symm_buffer_for_mega_moe
from primus_turbo.flydsl.mega.prims import cast, read_clock, spin_timed_out
from primus_turbo.flydsl.utils.gemm_helper import emit_if_then, make_value_attrs, run_compiled

# GEMM launch attrs (nt_vmcnt / waves_per_eu); included in flyc compile cache key below.
_COMBINE_NT_VMCNT = 3
_COMBINE_WAVES_PER_EU = 2
# The GEMM role is one workgroup per tile, tiles mapped row-major, and the grid carries one block
# per REAL tile (plus the tail-reduce reservation) rather than the pool's worst case. Three
# alternatives were plumbed through here and none earned its keep: a persistent grid (a fixed CU
# count striding over the tiles) measured slower, because tiles far outnumber the CUs and
# one-WG-per-tile lets the scheduler refill them at a finer grain; the XCD / group_m L2-reuse
# swizzle the standalone grouped GEMM autotunes was never switched on, and at its shipped values
# (num_xcd=1, group_m=group_n=0) it computed exactly the row-major mapping anyway; and the
# worst-case grid only added blocks that exit on arrival. Bring any of them back per-shape if a
# measurement asks.
_REDUCE_VW = 4  # top-k reduce payload load width in i32 words per lane (4 -> b128)
# Cap on the empty blocks running the reduce. Matches the bf16 combine's default; the empty
# region is far smaller than this in practice, so the cap does not bind and every empty block
# joins in.
_COMBINE_REDUCE_CAP = 256


def wait_lgkmcnt(n):
    """Drain LDS/SMEM traffic to at most ``n`` outstanding ops."""
    _llvm.inline_asm(
        res=None,
        operands_=[],
        asm_string=f"s_waitcnt lgkmcnt({n})",
        constraints="",
        has_side_effects=True,
    )

class StoreCQuantMxfp8CShuffle:
    """CShuffle epilogue that QUANTIZES the f32 accumulators to per-1x32 E8M0 mxfp8 on the
    LDS re-read. Stages each 16-row sub-tile row-major into per-wave LDS (like the
    non-quantizing ``StoreCPerTensorCShuffle`` in ``flydsl/utils/gemm_helper.py``), then
    re-reads ``EPL`` CONTIGUOUS columns per lane -> the
    1x32-along-N block amax is a cheap 4-lane reduction (2 shuffles) instead of the direct
    MFMA layout's 32-lane butterfly (5 shuffles), and the fp8 payload is a COALESCED store
    instead of scattered bytes. Reuses ``StoreCQuantFp8``'s E8M0/cvt math.

    Requires Cc = n_tiles_b*16 == 32 (one 1x32 block == one shuffle-tile row; N_TILES_B=2,
    the mega L2 tile). Writes ``C_fp8`` [c_rows, c_cols] fp8 + ``C_scale`` [c_rows, c_cols//32]
    E8M0 byte. This measures the efficient in-GEMM mxfp8 quant placement for fp8-combine."""

    def __init__(self, C_fp8, C_scale, c_rows, c_cols, c_idx_fn, n_tiles_a, n_tiles_b, out_ty, c_lds, wave_id):
        self.c_rows = c_rows
        self.c_cols = c_cols
        self.lane_id = fx.thread_idx.x % 64
        self.wave_id = wave_id
        self.c_idx_fn = c_idx_fn
        self.n_tiles_a = n_tiles_a
        self.n_tiles_b = n_tiles_b
        self.out_ty = out_ty
        self.Cc = n_tiles_b * 16
        assert self.Cc == 32, f"StoreCQuantMxfp8CShuffle requires Cc==32 (N_TILES_B=2), got {self.Cc}"
        self.EPL = (16 * self.Cc) // 64  # = 8 contiguous cols/lane
        self.row_stride = self.Cc
        self.wave_lds_elems = 16 * self.row_stride
        self.c_lds = c_lds
        self.fp8 = _buffer_ops.create_buffer_resource(C_fp8, max_size=True)
        self.scale = _buffer_ops.create_buffer_resource(C_scale, max_size=True)
        # bf16 staging pointer (align 2 store, align 16 read).
        self._store_ptr_t = fx.PointerType.get(out_ty.ir_type, 2, 2)
        self._read_ptr_t = fx.PointerType.get(out_ty.ir_type, 2, 16)

    def store(self, c_frag, base_row, base_col):
        lds_base = fx.Int32(fx.ptrtoint(self.c_lds.ptr))
        wave_off = self.wave_id * self.wave_lds_elems
        for ti in range_constexpr(self.n_tiles_a):
            # stage this 16-row sub-tile row-major into per-wave LDS (bf16)
            for tj in range_constexpr(self.n_tiles_b):
                vec_f32 = Vec(c_frag[self.c_idx_fn(ti, tj)])
                lds_col = tj * 16 + self.lane_id % 16
                for i in range_constexpr(4):
                    lds_row = (self.lane_id // 16) * 4 + i
                    e = wave_off + lds_row * self.row_stride + lds_col
                    ptr = fx.inttoptr(self._store_ptr_t, lds_base + e * 2)
                    ptr.store(vec_f32[i].to(self.out_ty))
            wait_lgkmcnt(0)
            # re-read EPL=8 contiguous cols of one row per lane; 4 lanes cover one 1x32 block
            row_in = (self.lane_id * self.EPL) // self.Cc      # = lane//4
            col0 = (self.lane_id * self.EPL) % self.Cc         # = (lane%4)*8
            lane_e = wave_off + row_in * self.row_stride + col0
            rptr = fx.inttoptr(self._read_ptr_t, lds_base + lane_e * 2)
            vec = Vec(fx.make_view(rptr, fx.make_layout(self.EPL, 1)).load())  # 8 bf16
            f = [fx.arith.ArithValue(vec[j].to(fx.Float32)) for j in range_constexpr(self.EPL)]
            # within-lane |max| over the 8 owned values
            av = fx.arith.ArithValue(
                (fx.arith.ArithValue(f[0]).bitcast(fx.T.i32()) & fx.Int32(0x7FFFFFFF)).bitcast(fx.T.f32())
            )
            for j in range_constexpr(1, self.EPL):
                aj = fx.arith.ArithValue(
                    (f[j].bitcast(fx.T.i32()) & fx.Int32(0x7FFFFFFF)).bitcast(fx.T.f32())
                )
                av = fx.arith.ArithValue(fx.arith.maximumf(av, aj))
            # 4-lane amax (the 4 consecutive lanes owning this row's 32-col block)
            for sh in (1, 2):
                peer = fx.arith.ArithValue(av.shuffle_xor(sh, 64))
                av = fx.arith.ArithValue(fx.arith.maximumf(av, peer))
            # E8M0 scale (mirror StoreCQuantFp8), target 2^8
            amax_bits = av.bitcast(fx.T.i32())
            t = amax_bits + fx.Int32(1 << 19)
            exp = ((t >> fx.Int32(23)) & fx.Int32(0x1FF)) - fx.Int32(127 + 8)
            exp = fx.arith.select(exp < fx.Int32(-127), fx.Int32(-127), exp)
            exp = fx.arith.select(exp > fx.Int32(128), fx.Int32(128), exp)
            biased = fx.arith.ArithValue(exp) + fx.Int32(127)
            # 1/scale from the exponent bits, not 1.0/bits(biased << 23) -- see the same reciprocal
            # in quant.py. An all-zero 32-column block gives biased 0, where the float form divides
            # by zero and quantizes every element to 0*inf = NaN.
            inv = fx.arith.ArithValue((fx.Int32(254) - biased) << fx.Int32(23)).bitcast(fx.T.f32())
            neglim = fx.arith.ArithValue(fx.arith._to_raw(fx.Float32(-448.0)))
            poslim = fx.arith.ArithValue(fx.arith._to_raw(fx.Float32(448.0)))
            # cvt 8 vals -> 2 packed i32 (4 fp8/word), coalesced 64b store
            words = []
            for jw in range_constexpr(self.EPL // 4):
                q = [
                    fx.arith.ArithValue(
                        fx.arith.minimumf(fx.arith.maximumf(f[jw * 4 + k] * inv, neglim), poslim)
                    )
                    for k in range_constexpr(4)
                ]
                w = rocdl.cvt_pk_fp8_f32(fx.T.i32(), q[0], q[1], fx.Int32(0), False)
                w = rocdl.cvt_pk_fp8_f32(fx.T.i32(), q[2], q[3], w, True)
                words.append(w)
            base_row_i = base_row + ti * 16 + row_in
            gcol = base_col + col0
            fp8_idx = (base_row_i * fx.Int32(self.c_cols) + gcol) // fx.Int32(4)  # i32-word index
            _buffer_ops.buffer_store(
                Vec.from_elements(words, fx.Int32).ir_value(), self.fp8, fp8_idx, cache_modifier=16
            )
            # one E8M0 byte per 1x32 block: the lane owning col0==0 writes it
            def _emit_scale():
                sb = fx.arith.ArithValue(biased).trunci(fx.T.i8())
                _buffer_ops.buffer_store(
                    sb, self.scale, base_row_i * fx.Int32(self.c_cols // 32) + gcol // fx.Int32(32),
                    cache_modifier=16,
                )

            emit_if_then(col0 == fx.Int32(0), _emit_scale)


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
    ``comb[slot]``, raise the sys-scope flag. ``with_gate`` (backward L1 dgrad) additionally scatters
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
        # factory, not a closure over the loop var: `emit_if_then` branch fns MUST take 0 args
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

            emit_if_then(lane < fx.Int32(SC), _emit_scale)
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
            emit_if_then(origin >= fx.Int32(0), _make_row_emitter(row, origin))

    return push_block


# ─────────── role REDUCE: fp8-dequant topk sum (weighted|unweighted) + optional gate fold ───────────
def _make_topk_reduce_fp8(hidden, topk, combine_slots, apply_weights, with_gate):
    """FP8-dequant weighted top-k sum -> output. ``apply_weights`` (forward L2) multiplies each expert
    term by the routing weight; else (backward L1 dgrad) the weight was already folded into the input
    upstream. ``with_gate`` (backward) additionally folds the gate gradient
    ``d_topk_w[slot] = combine_gate[slot]`` (route-masked). Mirrors the unified bf16
    ``topk_reduce_bf16_tile(apply_weights, with_gate)`` -- one reduce for both fwd and bwd."""
    H4 = hidden // 4
    payload_i32_total = combine_slots * H4
    SC = hidden // 128
    # A lane loads VW consecutive payload words as one b128, matching the PUSH path. Words are
    # 4-aligned, so a group of 4 stays inside one E8M0 32-block and one scale word plus one shift
    # serves all 4 (``sword_idx`` / ``shift`` below); a narrower VW would need per-word scale
    # indexing. Divides because the fp8 push requires hidden % 1024 == 0 -- asserted here too, since
    # this file does not otherwise state the constraint it depends on.
    VW = _REDUCE_VW
    assert H4 % (_WARP * VW) == 0, f"fp8 reduce needs hidden % 1024 == 0, got hidden={hidden}"
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
                w = lane * fx.Int32(VW) + fx.Int32(k * _WARP * VW)  # 1024 B per warp step
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
    num_gemm_cu=None,
):
    """Unified fp8 combine: mxfp8 GEMM (CShuffle mxfp8-quant epilogue -> local fp8 pool) + FP8 combine
    PUSH (+ optional gate scatter) + fp8-dequant top-k reduce. One kernel for BOTH:
      * forward L2   (apply_weights=True,  with_gate=False): ``act @ w2``, K=I, WEIGHTED reduce -> y.
      * backward L1 dgrad (apply_weights=False, with_gate=True): ``grad_l1 @ w1^T``, K=2I, UNWEIGHTED
        reduce (routing weight folded upstream) + gate scatter / d_topk_w fold -> dx.
    Mirrors the unified bf16 ``grouped_gemm_combine`` (apply_weights/with_gate constexpr flags).

    The three roles are sized by their own argument, and 0 means the role is absent -- which is how
    the isolation builds the benches time are expressed, instead of a flag per combination:

      ``num_combine_cu``  blocks running the cross-rank PUSH.
      ``num_reduce_cu``   cap on how many of the grid's EMPTY blocks run the reduce, the rest exiting
                          on arrival; they stride over the empty tiles, so the cap only binds when
                          the pool is nearly full. 0 removes the role. Same meaning as the bf16
                          combine's argument of that name -- there is no separate reduce region in
                          the grid, here or there.
      ``num_gemm_cu``     None runs one workgroup per real tile, which is the only production shape:
                          the block count follows the routing, so it is derived, not tuned. 0 removes
                          the role. A positive count would mean a persistent grid, deleted as slower.

    Because the reduce rides the empty blocks rather than a reserved region, the grid is
    ``num_combine_cu + worst_case_tiles * n_blocks`` whatever the routing does -- so no launch ever
    reads the real tile count back from the device, which would drain the queue and kill CPU/GPU
    overlap. (The bf16 combine sizes its grid by the same expression.)

    Dropping a role changes the others by implication rather than by a second switch: with no GEMM
    role nothing ever sets the GEMM-done flag, so the PUSH must not wait on it, and the gate is
    compiled out. Combinations that measure something: (combine, 0, None) skips the reduce,
    (0, 0, None) is the GEMM alone, (combine, 0, 0) is the PUSH alone. All but the first give
    INCORRECT output -- they exist to attribute time, not to compute."""
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
    comb_records = combine_slots * out_features * 2
    gate_records = combine_slots * 4  # f32 gate slots per peer (backward d_topk_w scatter)
    delta_records = num_ranks * 8
    assert num_gemm_cu in (None, 0), (
        f"num_gemm_cu={num_gemm_cu}: the GEMM role is one workgroup per real tile, so its block count "
        "is derived (None) or the role is absent (0). A positive count would be the persistent grid, "
        "removed as slower than one-WG-per-tile."
    )
    assert num_reduce_cu >= 0, f"num_reduce_cu={num_reduce_cu}: a cap on the empty blocks, or 0 for off"
    _no_gemm = num_gemm_cu == 0
    gemm_base = num_combine_cu
    H4 = out_features // 4
    SC = out_features // 128
    payload_i32_total = combine_slots * H4
    reduce_fp8 = _make_topk_reduce_fp8(out_features, topk, combine_slots, apply_weights, with_gate)

    @flyc.kernel(known_block_size=[_BLOCK_THREADS, 1, 1])
    def kern(
        ACT: fx.Tensor, WEIGHTS: fx.Tensor, L2Y_FP8: fx.Tensor, L2Y_SCALE: fx.Tensor,
        TILE_TO_GROUP: fx.Tensor, NUM_TILE_BLOCKS: fx.Tensor, OUTPUT: fx.Tensor, TOPK_INDICES: fx.Tensor,
        NUM_TOKENS_PER_RANK: fx.Tensor, TOPK_WEIGHTS: fx.Tensor, GRAD_GATE: fx.Tensor, D_TOPK_W: fx.Tensor,
        A_SCALE: fx.Tensor, B_SCALE: fx.Tensor,
        ORIGIN_RANK: fx.Tensor, ORIGIN_SLOT: fx.Tensor,
        COMBINE_PARITY: fx.Tensor, COMBINE_EXPECTED: fx.Tensor, REDUCE_EXPECTED: fx.Tensor,
        sym_layout: SymLayout, c_n: fx.Int32,
    ):
        thread_index = fx.thread_idx.x
        block_index, _b, _c = fx.block_idx
        combine_cu = fx.Int32(num_combine_cu)
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
        # The pool-row -> (owning rank, topk slot) map has to come from this call's own snapshot, not
        # from the live symm region: every dispatch prologue resets that whole region to -1 and
        # refills only the rows it dispatches, so by the time a layer's backward runs, the region
        # describes the LAST layer's routing. Pushing this layer's rows through that map sends them
        # to the wrong slots and skips the right ones, which is what left peers' reduce spinning.
        # bf16 avoids this the same way, via its handle[12] pool_src_slot snapshot.
        origin_rank_res = create_buffer_resource(ORIGIN_RANK, max_size=True)
        origin_slot_res = create_buffer_resource(ORIGIN_SLOT, max_size=True)
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
            local_count = (real_tiles - block_index + combine_cu - fx.Int32(1)) // combine_cu
            for tile_iter in range(local_count):
                block_m = block_index + tile_iter * combine_cu
                # No GEMM role means no one ever sets the GEMM-done flag, so waiting on it would hang:
                # the PUSH then ships whatever the local L2Y already holds (incorrect, timing only).
                if not _no_gemm:
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
                    store_c=store_c,
                )
                # GEMM-done release. The second rendezvous is NOT extra ordering and must not be read
                # as such: diffing the emitted ISA with and without it shows the two builds identical
                # apart from these two instructions, and after the first `s_waitcnt(0)` vmcnt is
                # already 0 with no memory op between, so it has nothing to wait for and no wave left
                # to sync. What it buys is time. Without it the PUSH ships L2Y bytes this call's GEMM
                # has not written yet -- ~800 of 8192 output tokens per rank against the poison
                # amplifier in repro_fp8_combine_gate.py, torn at 32-column granularity inside tiles
                # rather than whole tiles missing, which is a propagation race, not a missing gate.
                # The `s_waitcnt` between the barriers is load-bearing: LLVM folds adjacent
                # `s_barrier`s.
                #
                # So this is a timing workaround for a gap in the release, not a fix for it, and it
                # could reopen under a different compiler or schedule. Ruled out by measurement, so
                # nobody repeats them: `buffer_wbl2` sc1 / sc0|sc1 on the release (worse, ~1100),
                # dropping sc1 from the C stores and releasing with wbl2 (worse still, ~1400),
                # `gpu.barrier` in place of the bare `s_barrier` (no change), sc0|sc1 on the C stores,
                # sc1 on the PUSH's L2Y loads, dropping the PUSH's `buffer_inv`, sys-scope flag
                # traffic, and an `s_waitcnt` before every flag read. Untried: making the flag store a
                # release atomic rather than a plain sc1 store.
                fx.rocdl.s_waitcnt(0)
                rocdl.s_barrier()
                fx.rocdl.s_waitcnt(0)
                rocdl.s_barrier()
                flag_off = combine_bank + block_m * n_blocks_i32 + block_n
                emit_if_then(
                    thread_index == fx.Int32(0),
                    lambda: st(combine_flag_base, flag_off, expected_combine, scope="agent"),
                )

            # The grid is sized to the pool's worst case, so a block owns a real tile (GEMM) or
            # falls in the empty region past them (reduce, capped at num_reduce_cu blocks that
            # stride over the leftovers). No reserved reduce region: same shape as bf16.
            role_idx = block_index - fx.Int32(gemm_base)
            real_gemm_blocks = real_tiles * fx.Int32(n_blocks)
            if role_idx < real_gemm_blocks:
                block_m = role_idx // fx.Int32(n_blocks)
                block_n = role_idx % fx.Int32(n_blocks)
                if not _no_gemm:
                    _do_gemm_tile(block_m, block_n)
            elif const_expr(num_reduce_cu > 0):
                empty_ordinal = role_idx - real_gemm_blocks
                if empty_ordinal < fx.Int32(num_reduce_cu):
                    n_empty_tiles = fx.Int32(worst_case_tiles) - real_tiles
                    n_reduce_tiles = n_empty_tiles * fx.Int32(n_blocks)
                    # stride = the blocks that actually showed up, so the cap only binds when the
                    # empty region is larger than it
                    active = fx.arith.select(
                        n_reduce_tiles < fx.Int32(num_reduce_cu),
                        n_reduce_tiles, fx.Int32(num_reduce_cu),
                    )
                    reduce_fp8(
                        thread_index, empty_ordinal, active * fx.Int32(_NUM_WARPS),
                        num_experts, rank, comb_base, comb_records, output_res, topk_indices_res,
                        num_tokens_res, reduce_flag_base, reduce_bank, expected_reduce,
                        topk_weights_res, gate_local_res, d_topk_w_res,
                    )

    @flyc.jit
    def launch(
        ACT, WEIGHTS, L2Y_FP8, L2Y_SCALE, TILE_TO_GROUP, NUM_TILE_BLOCKS, OUTPUT, TOPK_INDICES,
        NUM_TOKENS_PER_RANK, TOPK_WEIGHTS, GRAD_GATE, D_TOPK_W, A_SCALE, B_SCALE,
        ORIGIN_RANK, ORIGIN_SLOT,
        COMBINE_PARITY, COMBINE_EXPECTED, REDUCE_EXPECTED, sym_layout, c_n: int,
        stream: fx.Stream = fx.Stream(None),
    ):
        # Independent of the routing: the GEMM blocks and the empty blocks the reduce rides always
        # add up to worst_case_tiles * n_blocks, so no launch reads the real tile count back from
        # the device. Same expression as the bf16 combine's grid.
        grid_size = gemm_base + worst_case_tiles * n_blocks
        # bump epoch on device (combine += n_blocks, reduce += 1) before the kernel; same-stream visible
        _make_epoch_bump(1, 1)(COMBINE_PARITY, COMBINE_EXPECTED, REDUCE_EXPECTED).launch(
            grid=(1, 1, 1), block=(_BLOCK_THREADS, 1, 1), stream=stream
        )
        kern(
            ACT, WEIGHTS, L2Y_FP8, L2Y_SCALE, TILE_TO_GROUP, NUM_TILE_BLOCKS, OUTPUT, TOPK_INDICES,
            NUM_TOKENS_PER_RANK, TOPK_WEIGHTS, GRAD_GATE, D_TOPK_W, A_SCALE, B_SCALE,
            ORIGIN_RANK, ORIGIN_SLOT,
            COMBINE_PARITY, COMBINE_EXPECTED, REDUCE_EXPECTED, sym_layout, c_n,
            value_attrs=make_value_attrs(waves_per_eu, agpr_alloc, "512,512"),
        ).launch(grid=(grid_size, 1, 1), block=(_BLOCK_THREADS, 1, 1), stream=stream)

    return launch


_L2Y_FP8_SCRATCH: dict = {}


def _combine_ctx(handle, act_fp8):
    """Everything about a combine launch that the direction does not change.

    Kept in one place because the two entries would otherwise both carry it, and the handle reads
    below are not the kind of thing that should exist in two copies that can drift."""
    aq, a_sp = act_fp8
    M, K = aq.shape  # K = I for the fc2 forward, 2I for the fc1^T dgrad
    dev = aq.device
    symm = get_symm_buffer_for_mega_moe()
    sym_layout = symm.make_sym_layout()
    H = int(sym_layout.hidden)
    assert M == int(sym_layout.num_max_pool_tokens)

    sk = (M, H, dev)
    scratch = _L2Y_FP8_SCRATCH.get(sk)
    if scratch is None:
        l2y_fp8 = torch.empty(M * H, dtype=torch.uint8, device=dev)
        l2y_scale = torch.empty(M * (H // 32), dtype=torch.uint8, device=dev)
        _L2Y_FP8_SCRATCH[sk] = scratch = (l2y_fp8, l2y_scale)

    # Everything below that varies per call comes off the handle, never out of the live symm state.
    # A layer's backward runs long after later layers' prologues have rewritten the shared scratch,
    # so reading it there picks up another layer's dispatch: the tile count came back too large and
    # exposed tiles this call never filled (they still hold the prologue's out-of-range expert
    # sentinel, which indexed WEIGHTS one expert stride past its end), and the pool-row -> slot map
    # came back describing another layer's routing. The three have to move together -- fixing only
    # the count leaves the push sending rows to the wrong slots, which hangs peers' reduce.
    return SimpleNamespace(
        aq=aq, a_sp=a_sp, act_flat=aq.view(torch.int8).reshape(-1), M=M, K=K, dev=dev,
        symm=symm, sym_layout=sym_layout, out_features=H,
        G=int(sym_layout.num_experts_per_rank),
        combine_slots=int(sym_layout.combine_slots),
        num_ranks=int(sym_layout.num_ranks),
        rank=int(sym_layout.rank_idx),
        topk=int(sym_layout.num_topk),
        num_experts=int(sym_layout.num_experts),
        num_tokens=int(symm.num_tokens),
        tile_to_expert=handle[7],
        num_tile_blocks=handle[_H_NUM_TILE_BLOCKS],
        origin_rank=handle[_H_ORIGIN_RANK],
        origin_slot=handle[_H_ORIGIN_SLOT],
        l2y_fp8=scratch[0], l2y_scale=scratch[1],
        # An unused per-role slot gets this size-1 tensor so the kernel's create_buffer_resource has a
        # valid (never-indexed) one: TOPK_WEIGHTS is only read when apply_weights, GRAD_GATE and
        # D_TOPK_W only when with_gate.
        dummy_f32=torch.empty(1, dtype=torch.float32, device=dev),
    )


def _combine_launch(
    c, weights_fp8, *, topk_indices, topk_weights_arg, grad_gate_arg, d_topk_w,
    apply_weights, with_gate, BM, BN, combine_cu, num_reduce_cu, num_gemm_cu,
):
    """Compile (once per shape) and run the fused GEMM + PUSH + reduce; returns the output tensor.

    ``apply_weights`` / ``with_gate`` are forwarded, not branched on: they are constexpr in
    ``_compile`` and select the reduce's shape inside the one kernel both directions share."""
    weight_flat, b_sp = weights_fp8
    output = torch.empty(c.num_tokens, c.out_features, dtype=torch.bfloat16, device=c.dev)
    args = (
        c.act_flat, weight_flat, c.l2y_fp8, c.l2y_scale, c.tile_to_expert, c.num_tile_blocks,
        output.view(-1), topk_indices.contiguous().view(-1), c.symm.num_tokens_per_rank,
        topk_weights_arg, grad_gate_arg, d_topk_w, c.a_sp, b_sp, c.origin_rank, c.origin_slot,
        c.symm._combine_parity, c.symm._combine_expected, c.symm._reduce_expected,
        c.sym_layout, c.out_features, torch.cuda.current_stream(),
    )
    launch = _compile(
        c.out_features, c.K, c.M, BM, BN, int(combine_cu), int(num_reduce_cu),
        c.combine_slots, c.topk, c.num_experts, c.rank, c.num_ranks,
        apply_weights, with_gate, num_groups=c.G, num_gemm_cu=num_gemm_cu,
    )
    ck = (c.out_features, c.K, c.M, BM, BN, int(combine_cu), int(num_reduce_cu), num_gemm_cu,
          c.combine_slots, c.topk, c.num_experts, c.rank, c.num_ranks, c.G,
          apply_weights, with_gate, _COMBINE_NT_VMCNT, _COMBINE_WAVES_PER_EU)
    run_compiled(_FP8_COMBINE_COMPILED, ck, launch, *args)
    return output


def combine_l2_fwd_mxfp8_flydsl_kernel(
    weights_fp8, handle, *, topk_indices, topk_weights, x_fp8, BM=256, BN=256,
    num_combine_cu=32, num_reduce_cu=_COMBINE_REDUCE_CAP, num_gemm_cu=None,
):
    """Forward L2: ``act @ w2`` (K=I) + fp8 combine PUSH + WEIGHTED top-k reduce -> ``y`` [T, H] bf16.

    ``x_fp8 = (act_fp8, a_sp)`` comes pre-quantized from ``swiglu_mxfp8_flydsl_kernel``; there is no A
    quant here. The routing weight is applied by the reduce, which is what ``topk_weights`` is for.

    PURE COMPUTE: ``weights_fp8 = (weight_flat, b_sp)`` arrives ALREADY prepared (op-layer
    ``prepare_w2_fp8`` on ``w2`` [G,H,I], version-keyed there) -- no weight quant/preshuffle, no
    caching here. The combine_flag / reduce_flag epoch gates are double-banked and bumped on device,
    so there is no host flag reset or rendezvous.

    ``num_combine_cu`` / ``num_reduce_cu`` / ``num_gemm_cu`` size the three roles; 0 removes one, which
    is how the benches isolate a stage. See ``_compile`` for what each means."""
    c = _combine_ctx(handle, x_fp8)
    return _combine_launch(
        c, weights_fp8, topk_indices=topk_indices,
        topk_weights_arg=topk_weights.contiguous().view(-1),
        grad_gate_arg=c.dummy_f32, d_topk_w=c.dummy_f32,
        apply_weights=True, with_gate=False,
        BM=BM, BN=BN, combine_cu=num_combine_cu, num_reduce_cu=num_reduce_cu, num_gemm_cu=num_gemm_cu,
    )


def combine_l1_dgrad_mxfp8_flydsl_kernel(
    weights_fp8, handle, *, topk_indices, grad_gate, x_fp8_rowwise, BM=256, BN=256,
    num_combine_cu=24, num_reduce_cu=_COMBINE_REDUCE_CAP, num_gemm_cu=None,
):
    """Backward L1 dgrad: ``grad_l1 @ w1^T`` (K=2I) + fp8 combine PUSH + gate scatter + UNWEIGHTED
    top-k reduce -> ``(dx [T, H] bf16, d_topk_w [combine_slots] f32)``.

    ``x_fp8_rowwise = (q_row, a_sp)`` comes pre-quantized from
    ``swiglu_bwd_rowcol_dual_quant_mxfp8_flydsl``. The reduce is unweighted because the routing weight
    was folded upstream; ``grad_gate`` is scattered into ``d_topk_w`` alongside it.

    PURE COMPUTE: ``weights_fp8 = (weight_flat, b_sp)`` arrives ALREADY prepared (op-layer prep on
    ``w1^T`` [G,H,2I], version-keyed there) -- no weight quant/preshuffle, no caching here. The
    combine_flag / reduce_flag epoch gates are double-banked and bumped on device, so there is no host
    flag reset or rendezvous.

    ``num_combine_cu`` / ``num_reduce_cu`` / ``num_gemm_cu`` size the three roles; 0 removes one, which
    is how the benches isolate a stage. See ``_compile`` for what each means."""
    c = _combine_ctx(handle, x_fp8_rowwise)
    d_topk_w = torch.empty(c.combine_slots, dtype=torch.float32, device=c.dev)
    dx = _combine_launch(
        c, weights_fp8, topk_indices=topk_indices,
        topk_weights_arg=c.dummy_f32,
        grad_gate_arg=grad_gate.contiguous().view(-1), d_topk_w=d_topk_w,
        apply_weights=False, with_gate=True,
        BM=BM, BN=BN, combine_cu=num_combine_cu, num_reduce_cu=num_reduce_cu, num_gemm_cu=num_gemm_cu,
    )
    return dx, d_topk_w
