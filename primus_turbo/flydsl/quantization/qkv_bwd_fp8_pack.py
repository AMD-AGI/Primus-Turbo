###############################################################################
# SPDX-License-Identifier: Apache-2.0
#
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
###############################################################################
"""Fused GPT-OSS QKV-gradient pack + tensorwise FP8 quantize (+ transpose).

Replaces ``cat((dq,dk,dv),-1) -> quantize_fp8_tensorwise -> transpose_2d`` with

    pass A : global abs-amax over dq/dk/dv            (335 MB read)
    pass A2: amax -> scale / scale_inv                (1 block)
    pass B : one cast/scatter pass, both FP8 layouts  (335 MB read + 336 MB write)

The BF16 packed tensor is never materialised. ``packed_fp8`` bytes, ``scale_inv``
bits and ``packed_fp8_t`` bytes are identical to the reference sequence: the amax
is an exact (order-independent) max, the scale expression is copied from
``tensorwise_amax_scale_kernel``, and the cast is the same ``v_cvt_pk_fp8_f32``
after the same ``clamp(x * scale, +-448)``.

Any shape/dtype outside the fast path falls back to the reference sequence.
"""

import os as _os

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl._mlir.dialects import llvm as _llvm
from flydsl.expr import range_constexpr, rocdl
from flydsl.expr import buffer_ops as bo
from flydsl.expr import math as fm
from flydsl.expr.arith import ArithValue
from flydsl.expr.typing import T as _T
from flydsl.expr.typing import Vector as Vec

FP8_MAX = 448.0
AMAX_EPS = 1.0e-12

_CM = int(_os.environ.get("QKV_CM_ST", "16"))  # cache modifier on the row-major output store
# separate scope for the transposed store: it is a pure streaming write nothing
# re-reads (neither output stream can ever be L2-resident: ~32 MB total L2 across
# 8 XCDs vs 167.8 MB per stream), so nt (2) here leaves L2 entirely to the
# row-major write stream instead of having the two streams contend for it.
# Round-3 blocked 10-vs-10 ruler measurement: candidate_ms slow-mode median
# 0.19998 -> 0.19801 ms, score 2.3134 -> 2.3321 (+0.8%), pass-B spread collapsed
# from {124.2..138.3}us to {122.4..122.8}us. All 13 correctness cases + 3-repeat
# determinism pass identically at both scopes (output bytes are cache-scope
# invariant). See campaign goal.md P1.
# Round 11 added sc1 (bit4=16) on top of nt: 18 = sc1|nt. The transposed store fires
# only after a barrier with no concurrent loads (unlike the row-major store, which is
# interleaved with the next sub-tile's loads and wants bit1 UNSET -- see _CM below),
# so giving it its own coherence scope costs nothing and measured +0.58% alone
# (N=6, [2.58729,2.61282] vs the 2.59022 control) and +1.08% together with
# _TPR_Q_BIG=16 (N=8, [2.60903,2.62657], zero overlap with the N=12 control
# [2.57460,2.60452]). Round 12 re-confirms both alone and together at N=12.
_CM_STT = int(_os.environ.get("QKV_CM_STT", "18"))
_CM_LA = int(_os.environ.get("QKV_CM_LA", "3"))  # cache modifier on pass-A input loads
# nt (bit1=2) on pass-B's dq/dk/dv loads. Round 3/6's isolated-probe measurement (fixed
# out/out_t buffers, no allocation between reps) called this "neutral on the fused path"
# (124.9 -> 124.5us) or even a loss (round 2, pre-B_MAP). Round 7's BLOCKED ruler A/B
# (bench.sh, N=24 per side, alternating-paired baseline/candidate, fresh torch.empty()
# output allocation every call -- the real deployment regime) instead measures a huge,
# fully non-overlapping win: score median 2.286 -> 2.587 (+13.2%), candidate_ms
# 0.2036ms -> 0.1810ms. An 8-point bit sweep of every flag in isolation and combination
# (0,1,2,3,4,6,8,16 -- see goal.md P1' / round-7 report) shows the effect is caused by
# bit1 (value 2, "nt") ALONE: every value with bit1 set (2,3,6) scores ~2.59; every value
# without it (0,1,4,8,16) scores ~2.30, regardless of sc0/sc1/the undocumented bit2/swz
# bits. Mechanism hypothesis (not yet profiler-confirmed -- rocprofv3 aborts in this
# container): the isolated probe reuses the same fixed out/out_t buffers with zero
# allocator churn for its whole 40-rep loop, so whichever HBM/L2 mode a run's fixed
# addresses land in is "sticky" for that entire process. The real ruler -- like real
# training, where this kernel's inputs were just written by the attention backward and
# its outputs are consumed by the very next GEMM -- allocates fresh output tensors and
# alternates with the reference path's own very different allocation pattern every
# call, constantly perturbing the address/cache state. Cached pass-B loads (bit1=0) are
# apparently sensitive to that churn (landing in the slow mode almost every real call);
# nt loads (bit1=1) are not. Prefer plain nt (2) over sc0|nt (3): same measured result,
# one flag instead of two, matching the existing CM_STT=2 convention below.
_CM_LB = int(_os.environ.get("QKV_CM_LB", "2"))  # cache modifier on pass-B input loads
_B_MAP = int(_os.environ.get("QKV_B_MAP", "1"))  # pass-B block -> (row block, group) map
_FOLD_S = int(_os.environ.get("QKV_FOLD_S", "0"))  # fold the amax->scale reduce into pass B
# transposed-half attribution knob: 1 = LDS writes + barriers only, 2 = + corner-turn
# LDS reads (store masked off, no traffic), 3 = full. Diagnostic only.
_T_MODE = int(_os.environ.get("QKV_T_MODE", "3"))

# pass A -- global abs-amax
A_NB = int(_os.environ.get("QKV_A_NB", "1024"))  # workgroups == partial slots
A_NTH = int(_os.environ.get("QKV_A_NTH", "1024"))
A_VEC = int(_os.environ.get("QKV_A_VEC", "8"))  # 8 bf16 == 16 B per lane per load

# pass B -- fused cast / scatter
B_BM = int(_os.environ.get("QKV_B_BM", "128"))  # rows per workgroup tile
B_BK = 128  # logical packed cols per K|V sub-tile
B_NTH = int(_os.environ.get("QKV_B_NTH", "1024"))
# threads per row on the pure-Q sub-tiles. tpr*16 = contiguous output bytes a row
# gets per store and tpr*32 = contiguous input bytes it gets per load, so this is
# the burst-length knob. 8 reproduces the round-2 geometry.
#
# Round 11 ruler sweep (N=12 blocked, zero overlap): TPR_Q=8 -> 2.59022 (control),
# TPR_Q=16 -> 2.60506 (+0.57%), TPR_Q=32 -> 2.51446 (-2.93%): doubling the row-major
# store's contiguous run 128B->256B wins, but 256B->512B (plus LDS 16->64KB) loses.
# Round 12 adds TPR_Q=24 to bound the peak (still divisible: bm*tpr_q%nth==0 at
# BM=128/NTH=1024 requires tpr_q % 8 == 0, so 12/20 are not reachable without also
# changing BM/NTH; 24 is the nearest achievable neighbour between the two).
#
# _TPR_Q_BIG is the "wide burst" candidate; _fast_spec falls back to _TPR_Q_SMALL=8
# per-shape (not globally) whenever q_w does not divide evenly by _TPR_Q_BIG*16, so
# a wide default does not silently shrink the kernel's own shape coverage (e.g. the
# ruler's SMALL case q_w=16 would otherwise drop to the reference fallback at
# TPR_Q_BIG=16, still correct but no longer exercising the fused kernel).
_TPR_Q_BIG = int(_os.environ.get("QKV_TPR_Q_BIG", "16"))
_TPR_Q_SMALL = 8


def _select_tpr_q(q_w):
    """Per-shape burst-length choice: prefer the wide `_TPR_Q_BIG` candidate when
    `q_w` divides evenly by it (256B/512B.. contiguous runs); else fall back to the
    original round-2 `_TPR_Q_SMALL=8` geometry so odd/small shapes still take the
    fused fast path instead of the reference."""
    if q_w % (_TPR_Q_BIG * 16) == 0:
        return _TPR_Q_BIG
    return _TPR_Q_SMALL

_WR64 = (1, 2, 4, 8, 16, 32)


def _wave_max_f32(v, lane, IRI):
    """Butterfly max over a wave64 via ds_bpermute. max is order-independent, so the
    result is bit-identical to any other reduction order (incl. the C++ BlockReduce)."""
    I32 = fx.Int32
    F32 = fx.Float32
    for m in _WR64:
        idx = (lane ^ I32(m)) << I32(2)
        o = I32(rocdl.ds_bpermute(IRI, idx, v.bitcast(I32))).bitcast(F32)
        v = (v > o).select(v, o)
    return v


def _lds_wait():
    _llvm.inline_asm(
        res=None, operands_=[], asm_string="s_waitcnt lgkmcnt(0)", constraints="", has_side_effects=True
    )


def _sat(v):
    F32 = fx.Float32
    return fm.clampf(v, F32(-FP8_MAX), F32(FP8_MAX))


def _block_max_f32(t, acc, red, nw, IRI):
    """wave butterfly -> per-wave LDS slot -> every thread folds the nw slots."""
    I32 = fx.Int32
    F32 = fx.Float32
    lane = t & I32(63)
    w = t >> I32(6)
    acc = _wave_max_f32(acc, lane, IRI)
    # every lane of the wave writes the same value, so no divergent store is needed
    p = fx.add_offset(red.ptr, fx.make_int_tuple(w))
    fx.make_view(p, fx.make_layout(1, 1)).store(Vec.from_elements([acc], fx.Float32))
    _lds_wait()
    rocdl.s_barrier()
    out = F32(0.0)
    for i in range_constexpr(nw):
        pi = fx.add_offset(red.ptr, fx.make_int_tuple(I32(i)))
        vi = Vec(fx.make_view(pi, fx.make_layout(1, 1)).load())[0]
        out = (out > vi).select(out, vi)
    return out


def compile_amax(nq, nkv, elt):
    """Grid-strided abs-amax over dq (nq elems) + dk/dv (nkv elems each) -> f32[A_NB]."""
    nvq = nq // A_VEC
    nvk = nkv // A_VEC
    TILE = A_NB * A_NTH
    ITQ = (nvq + TILE - 1) // TILE
    ITK = (nvk + TILE - 1) // TILE
    NW = A_NTH // 64

    @flyc.kernel(known_block_size=[A_NTH, 1, 1])
    def kern(Q: fx.Tensor, K: fx.Tensor, V: fx.Tensor, P: fx.Tensor):
        I32 = fx.Int32
        F32 = fx.Float32
        IRI = fx.Int32.ir_type
        BF = elt.ir_type

        @fx.struct
        class Smem:
            red: fx.Array[fx.Float32, NW, 16]

        sm = fx.SharedAllocator().allocate(Smem).peek()
        red = sm.red
        t = fx.thread_idx.x
        bid = fx.block_idx.x
        base = bid * I32(A_NTH) + t

        acc = F32(0.0)
        for arg, nv, iters, nbytes in (
            (Q, nvq, ITQ, nq * 2),
            (K, nvk, ITK, nkv * 2),
            (V, nvk, ITK, nkv * 2),
        ):
            r = bo.create_buffer_resource(arg, max_size=False, num_records_bytes=I32(nbytes))
            exact = iters * TILE == nv
            for it in range_constexpr(iters):
                idx = base + I32(it * TILE)
                m = None if exact else (idx < I32(nv))
                v = bo.buffer_load(
                    r, idx * I32(A_VEC), vec_width=A_VEC, dtype=BF, mask=m, cache_modifier=_CM_LA
                )
                a = fm.absf(Vec(v).to(F32)).reduce("max")
                acc = (acc > a).select(acc, a)

        bmax = _block_max_f32(t, acc, red, NW, IRI)
        rp = bo.create_buffer_resource(P, max_size=False, num_records_bytes=I32(A_NB * 4))
        bo.buffer_store(bmax, rp, bid, mask=(t == I32(0)))

    @flyc.jit
    def launch(Q: fx.Tensor, K: fx.Tensor, V: fx.Tensor, P: fx.Tensor, stream: fx.Stream):
        kern(Q, K, V, P).launch(grid=(A_NB, 1, 1), block=(A_NTH, 1, 1), stream=stream)

    return launch


def compile_scale(nb):
    """partials[nb] -> (scale, scale_inv). Expression order copied verbatim from
    ``tensorwise_amax_scale_kernel`` so the f32 bits match."""
    NTH = 256
    VW = 4
    IT = (nb + NTH * VW - 1) // (NTH * VW)
    NW = NTH // 64

    @flyc.kernel(known_block_size=[NTH, 1, 1])
    def kern(P: fx.Tensor, S: fx.Tensor, SI: fx.Tensor):
        I32 = fx.Int32
        F32 = fx.Float32
        IRI = fx.Int32.ir_type

        @fx.struct
        class Smem:
            red: fx.Array[fx.Float32, NW, 16]

        sm = fx.SharedAllocator().allocate(Smem).peek()
        red = sm.red
        t = fx.thread_idx.x
        rp = bo.create_buffer_resource(P, max_size=False, num_records_bytes=I32(nb * 4))
        acc = F32(0.0)
        for it in range_constexpr(IT):
            idx = (t + I32(it * NTH)) * I32(VW)
            m = None if IT * NTH * VW == nb else (idx < I32(nb))
            v = bo.buffer_load(rp, idx, vec_width=VW, dtype=fx.Float32, mask=m)
            a = Vec(v).reduce("max")
            acc = (acc > a).select(acc, a)
        amax = _block_max_f32(t, acc, red, NW, IRI)
        eps = F32(AMAX_EPS)
        amax_c = (amax > eps).select(amax, eps)
        sc = F32(FP8_MAX) / amax_c
        si = F32(1.0) / sc
        ok = t == I32(0)
        rs = bo.create_buffer_resource(S, max_size=False, num_records_bytes=I32(4))
        rsi = bo.create_buffer_resource(SI, max_size=False, num_records_bytes=I32(4))
        bo.buffer_store(sc, rs, I32(0), mask=ok)
        bo.buffer_store(si, rsi, I32(0), mask=ok)

    @flyc.jit
    def launch(P: fx.Tensor, S: fx.Tensor, SI: fx.Tensor, stream: fx.Stream):
        kern(P, S, SI).launch(grid=(1, 1, 1), block=(NTH, 1, 1), stream=stream)

    return launch


def _piece_plain(C, rin_tag, rstride, wsrc, scoff, cw, nvec, dbase, sub_base, tpr):
    """One source tensor -> ``cw`` contiguous packed cols per thread.

    ``tpr`` threads cover one row of this piece, so one row gets ``tpr*cw`` bytes
    of contiguous output per store and ``tpr*cw*2`` bytes of contiguous input per
    load.  ``nth // tpr`` rows are done per pass and the piece loops until the
    workgroup's ``bm`` rows are covered.
    """
    I32 = fx.Int32
    rpp = C["nth"] // tpr  # rows per pass
    lr0 = C["t"] // I32(tpr)
    lcol = (C["t"] - lr0 * I32(tpr)) * I32(cw)
    for p in range_constexpr(C["BM"] // rpp):
        lr = lr0 + I32(p * rpp)
        src = (C["r0"] + lr) * I32(rstride) + C["g"] * I32(wsrc) + I32(scoff) + lcol
        ws = _emit_words(C, C["srcs"][rin_tag], src, nvec)
        dst = (C["r0"] + lr) * I32(C["PC"]) + C["g"] * I32(C["GW"]) + I32(dbase) + lcol
        bo.buffer_store(
            Vec.from_elements(ws, fx.Int32).ir_value(),
            C["ro"],
            dst,
            offset_is_bytes=True,
            cache_modifier=_CM,
        )
        C["lds_put"](C, ws, lr, lcol + I32(sub_base))


def _emit_words(C, rin, src, nvec):
    """nvec x (8 bf16) -> 2*nvec packed fp8 dwords, same clamp + v_cvt_pk_fp8_f32 as C++."""
    I32 = fx.Int32
    F32 = fx.Float32
    ws = []
    for j in range_constexpr(nvec):
        raw = bo.buffer_load(rin, src + I32(8 * j), vec_width=8, dtype=C["BF"], cache_modifier=_CM_LB)
        f = Vec(raw).to(F32) * C["scale"]
        for i in range_constexpr(2):
            w = I32(C["cvt"](C["IRI"], _sat(f[4 * i + 0]), _sat(f[4 * i + 1]), C["z"], 0))
            w = I32(C["cvt"](C["IRI"], _sat(f[4 * i + 2]), _sat(f[4 * i + 3]), w, 1))
            ws.append(w)
    return ws


def _emit_lds_put(C, ws, lrow, sub):
    """Scatter the packed dwords into the 16 B-strip XOR-swizzled LDS byte tile."""
    I32 = fx.Int32
    rb = (lrow >> I32(4)) & I32(C["STRIPS"] - 1)
    off = lrow * I32(C["LBK"]) + (((sub >> I32(4)) ^ rb) << I32(4)) + (sub & I32(15))
    bts = []
    for w in ws:
        for b in range_constexpr(4):
            bts.append(ArithValue((w >> I32(8 * b)) & I32(255)).trunci(_T.i8))
    p = fx.add_offset(C["tile"].ptr, fx.make_int_tuple(off))
    fx.make_view(p, fx.make_layout(len(bts), 1)).store(Vec.from_elements(bts, fx.Int8))


def _emit_t_writeback(C, gcol_base, width):
    """Corner-turn gather of 16 rows of one column -> one 16 B transposed store.

    ``BM//16`` threads cover one column (16 source rows each), so ``nth//(BM//16)``
    columns are turned per pass and the sub-tile's ``width`` columns need
    ``width / cols_per_pass`` passes.
    """
    I32 = fx.Int32
    _lds_wait()
    rocdl.s_barrier()
    tpc = C["BM"] // 16
    cpp = C["nth"] // tpc  # columns per pass
    lj0 = C["t"] // I32(tpc)
    li0 = (C["t"] - lj0 * I32(tpc)) * I32(16)
    rbr = (li0 >> I32(4)) & I32(C["STRIPS"] - 1)
    for p in range_constexpr(max(1, width // cpp) if _T_MODE >= 2 else 0):
        lj = lj0 + I32(p * cpp)
        scol = (((lj >> I32(4)) ^ rbr) << I32(4)) + (lj & I32(15))
        ows = []
        for d in range_constexpr(4):
            acc = None
            for k in range_constexpr(4):
                pk = fx.add_offset(
                    C["tile"].ptr, fx.make_int_tuple((li0 + I32(4 * d + k)) * I32(C["LBK"]) + scol)
                )
                b = Vec(fx.make_view(pk, fx.make_layout(1, 1)).load())[0]
                u = I32(ArithValue(b).extui(_T.i32)) & I32(255)
                acc = u if k == 0 else acc | (u << I32(8 * k))
            ows.append(acc)
        gc = C["g"] * I32(C["GW"]) + I32(gcol_base) + lj
        bo.buffer_store(
            Vec.from_elements(ows, fx.Int32).ir_value(),
            C["rot"],
            gc * I32(C["rows"]) + C["r0"] + li0,
            offset_is_bytes=True,
            cache_modifier=_CM_STT,
            mask=None if _T_MODE >= 3 else (C["pid"] < I32(0)),
        )
    _lds_wait()
    rocdl.s_barrier()


def _noop(*a, **k):
    return None


def _pack_scale(S, t, nth, red, IRI):
    """Either read the precomputed scale (S = f32[1]) or, with QKV_FOLD_S, redo the
    final amax -> scale reduce from the partials (S = f32[A_NB]) so pass A2's whole
    kernel launch disappears. max is order-independent, so the value is identical."""
    I32 = fx.Int32
    F32 = fx.Float32
    if not _FOLD_S:
        rs = bo.create_buffer_resource(S, max_size=False, num_records_bytes=I32(4))
        return F32(bo.buffer_load(rs, I32(0), vec_width=1, dtype=fx.Float32))
    nv = A_NB // nth
    rs = bo.create_buffer_resource(S, max_size=False, num_records_bytes=I32(A_NB * 4))
    acc = F32(0.0)
    for i in range_constexpr(nv):
        v = F32(bo.buffer_load(rs, t + I32(i * nth), vec_width=1, dtype=fx.Float32))
        acc = (acc > v).select(acc, v)
    amax = _block_max_f32(t, acc, red, nth // 64, IRI)
    eps = F32(AMAX_EPS)
    return F32(FP8_MAX) / (amax > eps).select(amax, eps)


def _pack_store_scale_inv(SI, scale, t, pid):
    """With QKV_FOLD_S nobody else writes scale_inv, so one workgroup does. Same
    ``1.0f / s`` expression as tensorwise_amax_scale_kernel."""
    if not _FOLD_S:
        return
    I32 = fx.Int32
    F32 = fx.Float32
    r = bo.create_buffer_resource(SI, max_size=False, num_records_bytes=I32(4))
    bo.buffer_store(F32(1.0) / scale, r, I32(0), mask=((t == I32(0)) & (pid == I32(0))))


def _pack_map(pid, NBM, groups, GRID):
    """(group, row-block) from the workgroup id. Plain python: the compile-time
    branch must not be rewritten into runtime control flow."""
    I32 = fx.Int32
    if _B_MAP == 0:
        g = pid // I32(NBM)
        return g, pid - g * I32(NBM)
    if _B_MAP == 2 and GRID % 8 == 0:
        x = pid & I32(7)
        pid = x * I32(GRID // 8) + (pid >> I32(3))
    if _B_MAP == 3 and GRID % 8 == 0:
        # XCD band: workgroup pid lands on XCD pid%8, so give each XCD a contiguous
        # row range -- its whole L2 working set is one linear span of dq/dk/dv.
        x = pid & I32(7)
        q = pid >> I32(3)
        b0 = q // I32(groups)
        return q - b0 * I32(groups), x * I32(NBM // 8) + b0
    if _B_MAP == 4 and GRID % 8 == 0:
        # XCD band, group-major inside the band
        x = pid & I32(7)
        q = pid >> I32(3)
        gg = q // I32(NBM // 8)
        return gg, x * I32(NBM // 8) + (q - gg * I32(NBM // 8))
    bmi = pid // I32(groups)
    return pid - bmi * I32(groups), bmi


def compile_pack(rows, groups, q_w, kv_w, elt, fuse_t=False, bm=B_BM, nth=B_NTH, tpr_q=_TPR_Q_BIG):
    """One pass over dq/dk/dv emitting the row-major FP8 (and, with ``fuse_t``, the
    transposed FP8 too).

    Tile = ``bm`` rows x one query group's ``q_w + 2*kv_w`` logical cols, walked as
    ``q_w / B_BK`` pure-Q sub-tiles plus one K|V sub-tile, so the source tensor of
    every load is a compile-time constant (no divergent gather, no base select).
    The transposed half stages the FP8 bytes in a 16 B-strip XOR-swizzled LDS tile
    (transpose_2d.cu's VEC-path swizzle) so both the store and the corner-turn
    gather are bank-conflict-free.

    ``tpr_q`` is an explicit parameter (not a module global) so the caller can pick
    it per-shape via ``_select_tpr_q`` and fold the choice into the compile cache key.
    """
    QT = groups * q_w  # dq row stride (elems)
    KT = groups * kv_w  # dk / dv row stride (elems)
    GW = q_w + 2 * kv_w  # logical cols per group
    PC = groups * GW  # packed cols
    BKQ = tpr_q * 16  # output cols a pure-Q sub-tile covers
    TPR_KV = (2 * kv_w) // 16  # 8 output B/thread on the K|V sub-tile
    NSQ = q_w // BKQ
    NBM = rows // bm
    GRID = NBM * groups
    LBK = max(BKQ, 2 * kv_w)  # LDS row stride (bytes) = widest sub-tile
    TB = bm * LBK
    STRIPS = LBK // 16
    assert q_w % BKQ == 0 and bm * tpr_q % nth == 0 and bm * TPR_KV % nth == 0
    if fuse_t:  # the corner turn walks `nth/(bm/16)` columns per pass
        assert BKQ * (bm // 16) % nth == 0 and (2 * kv_w) * (bm // 16) % nth == 0
    # compile-time walk plan: (LDS col base, width, ((emit, params), ...))
    PLAN = [
        (
            _st * BKQ,
            BKQ,
            ((_piece_plain, ("q", QT, q_w, _st * BKQ, 16, 2, _st * BKQ, 0, tpr_q)),),
        )
        for _st in range(NSQ)
    ]
    PLAN.append(
        (
            q_w,
            2 * kv_w,
            (
                (_piece_plain, ("k", KT, kv_w, 0, 8, 1, q_w, 0, TPR_KV)),
                (_piece_plain, ("v", KT, kv_w, 0, 8, 1, q_w + kv_w, kv_w, TPR_KV)),
            ),
        )
    )
    PLAN = tuple(PLAN)
    lds_put = _emit_lds_put if fuse_t else _noop
    t_write = _emit_t_writeback if fuse_t else _noop

    @flyc.kernel(known_block_size=[nth, 1, 1])
    def kern(
        Q: fx.Tensor,
        K: fx.Tensor,
        V: fx.Tensor,
        O: fx.Tensor,
        OT: fx.Tensor,
        S: fx.Tensor,
        SI: fx.Tensor,
    ):
        I32 = fx.Int32
        F32 = fx.Float32
        IRI = fx.Int32.ir_type
        BF = elt.ir_type
        z = I32(0)
        cvt = rocdl.cvt_pk_fp8_f32

        @fx.struct
        class Smem:
            tile: fx.Array[fx.Int8, TB, 16]
            red: fx.Array[fx.Float32, nth // 64, 16]

        sm = fx.SharedAllocator().allocate(Smem).peek()
        tile = sm.tile
        t = fx.thread_idx.x
        pid = fx.block_idx.x
        # map 0: group-major (one group's whole column range per contiguous pid run)
        # map 1: row-major (8 concurrent pids span one row block's full dq row)
        # map 2: map 1 after an XCD de-interleave so a full row block lands on one XCD
        g, bmi = _pack_map(pid, NBM, groups, GRID)
        r0 = bmi * I32(bm)

        scale = _pack_scale(S, t, nth, sm.red, IRI)
        _pack_store_scale_inv(SI, scale, t, pid)
        rq = bo.create_buffer_resource(Q, max_size=False, num_records_bytes=I32(rows * QT * 2))
        rk = bo.create_buffer_resource(K, max_size=False, num_records_bytes=I32(rows * KT * 2))
        rv = bo.create_buffer_resource(V, max_size=False, num_records_bytes=I32(rows * KT * 2))
        ro = bo.create_buffer_resource(O, max_size=False, num_records_bytes=I32(rows * PC))
        rot = bo.create_buffer_resource(OT, max_size=False, num_records_bytes=I32(rows * PC))

        C = {
            "BF": BF,
            "IRI": IRI,
            "z": z,
            "cvt": cvt,
            "scale": scale,
            "tile": tile,
            "t": t,
            "g": g,
            "r0": r0,
            "rot": rot,
            "pid": pid,
            "LBK": LBK,
            "BM": bm,
            "GW": GW,
            "STRIPS": STRIPS,
            "rows": rows,
        }
        C["srcs"] = {"q": rq, "k": rk, "v": rv}
        C["ro"] = ro
        C["PC"] = PC
        C["nth"] = nth
        C["lds_put"] = lds_put
        for cbase, cwidth, pieces in PLAN:
            for emit, params in pieces:
                emit(C, *params)
            t_write(C, cbase, cwidth)

    @flyc.jit
    def launch(
        Q: fx.Tensor,
        K: fx.Tensor,
        V: fx.Tensor,
        O: fx.Tensor,
        OT: fx.Tensor,
        S: fx.Tensor,
        SI: fx.Tensor,
        stream: fx.Stream,
    ):
        kern(Q, K, V, O, OT, S, SI).launch(grid=(GRID, 1, 1), block=(nth, 1, 1), stream=stream)

    return launch


# ---------------------------------------------------------------------------
# host side
# ---------------------------------------------------------------------------
_AMAX_C: dict = {}
_SCALE_C: dict = {}
_PACK_C: dict = {}
_WS: dict = {}

FUSE_T = True


def _elt(dtype):
    if dtype == torch.bfloat16:
        return fx.BFloat16
    if dtype == torch.float16:
        return fx.Float16
    return None


def _reference(dq, dk, dv, out_dtype):
    packed = torch.cat((dq, dk, dv), dim=-1).reshape(dq.shape[0] * dq.shape[1], -1)
    f8, scale_inv = torch.ops.primus_turbo_cpp_extension.quantize_fp8_tensorwise(
        packed, out_dtype, None, 1, 1
    )
    f8t = torch.ops.primus_turbo_cpp_extension.transpose_2d(f8, -2, -1)
    return f8, f8t, scale_inv


def _fast_spec(dq, dk, dv, out_dtype):
    if out_dtype != torch.float8_e4m3fn:
        return None
    if dq.ndim != 4 or dk.ndim != 4 or dv.ndim != 4:
        return None
    if dq.dtype != torch.bfloat16 or dk.dtype != dq.dtype or dv.dtype != dq.dtype:
        return None
    if not (dq.is_contiguous() and dk.is_contiguous() and dv.is_contiguous()):
        return None
    if not (dq.is_cuda and dk.is_cuda and dv.is_cuda):
        return None
    seq, batch, groups, q_w = (int(v) for v in dq.shape)
    if tuple(dk.shape) != tuple(dv.shape) or tuple(dk.shape[:3]) != (seq, batch, groups):
        return None
    kv_w = int(dk.shape[3])
    rows = seq * batch
    gw = q_w + 2 * kv_w
    pc = groups * gw
    tpr_q = _select_tpr_q(q_w)
    # fast-path geometry: full 16 B strips everywhere, 128-row tiles, 2*kv_w == B_BK
    if rows % B_BM or q_w % (tpr_q * 16) or (2 * kv_w) % 16 or gw % 16:
        return None
    if rows * pc >= 2**31 or rows * groups * q_w >= 2**31:
        return None
    if (rows * groups * q_w) % A_VEC or (rows * groups * kv_w) % A_VEC:
        return None
    return rows, groups, q_w, kv_w, pc, tpr_q


def qkv_bwd_fp8_pack_flydsl(dq, dk, dv, out_dtype):
    """Fused ``cat(dq,dk,dv,-1) -> tensorwise fp8 quantize -> transpose``.

    Returns freshly allocated ``(packed_fp8 [rows, cols], packed_fp8_t [cols, rows],
    scale_inv f32 scalar)``. Inputs are never written and no output aliases another.
    """
    spec = _fast_spec(dq, dk, dv, out_dtype)
    if spec is None:
        return _reference(dq, dk, dv, out_dtype)
    rows, groups, q_w, kv_w, pcols, tpr_q = spec
    dev = dq.device

    out = torch.empty((rows, pcols), dtype=out_dtype, device=dev)
    out_t = torch.empty((pcols, rows), dtype=out_dtype, device=dev)
    # flydsl cannot bind a 0-dim tensor (no stride==1 axis); keep the storage 1-D and
    # hand the ruler a 0-dim view of it.
    scale_inv_1d = torch.empty(1, dtype=torch.float32, device=dev)

    wkey = (dev.index,)
    ws = _WS.get(wkey)
    if ws is None:
        ws = (
            torch.empty(A_NB, dtype=torch.float32, device=dev),
            torch.empty(1, dtype=torch.float32, device=dev),
        )
        _WS[wkey] = ws
    partials, scale = ws

    stream = torch.cuda.current_stream()
    elt = _elt(dq.dtype)
    nq = rows * groups * q_w
    nkv = rows * groups * kv_w

    akey = (nq, nkv, dq.dtype)
    ac = _AMAX_C.get(akey)
    if ac is None:
        ac = flyc.compile(compile_amax(nq, nkv, elt), dq, dk, dv, partials, stream)
        _AMAX_C[akey] = ac
    ac(dq, dk, dv, partials, stream)

    sc = _SCALE_C.get(A_NB)
    if sc is None:
        sc = flyc.compile(compile_scale(A_NB), partials, scale, scale_inv_1d, stream)
        _SCALE_C[A_NB] = sc
    if not _FOLD_S:
        sc(partials, scale, scale_inv_1d, stream)
    sarg = partials if _FOLD_S else scale

    pkey = (rows, groups, q_w, kv_w, dq.dtype, FUSE_T, tpr_q, B_BM, B_NTH)
    pk = _PACK_C.get(pkey)
    if pk is None:
        pk = flyc.compile(
            compile_pack(rows, groups, q_w, kv_w, elt, fuse_t=FUSE_T, tpr_q=tpr_q),
            dq,
            dk,
            dv,
            out,
            out_t,
            sarg,
            scale_inv_1d,
            stream,
        )
        _PACK_C[pkey] = pk
    pk(dq, dk, dv, out, out_t, sarg, scale_inv_1d, stream)

    if not FUSE_T:
        out_t = torch.ops.primus_turbo_cpp_extension.transpose_2d(out, -2, -1)
    return out, out_t, scale_inv_1d.view(())
