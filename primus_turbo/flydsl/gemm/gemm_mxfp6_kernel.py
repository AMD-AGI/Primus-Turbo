###############################################################################
# SPDX-License-Identifier: Apache-2.0
#
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2025 FlyDSL Project Contributors
#
# Adapted from FlyDSL (https://github.com/ROCm/FlyDSL)
# Modified by the Primus-Turbo team.
#
# This file is distributed under the Apache License 2.0 (see LICENSE-APACHE),
# not the MIT license that covers the rest of Primus-Turbo (see LICENSE).
###############################################################################

"""4-wave MXFP6 (A6W6, both operands E2M3) dense NT GEMM for gfx950, derived from ``gemm_mxfp4_kernel.py``.

Two kernels share the operand formats and LDS geometry below: a single-tile kernel with the K loop unrolled
(``gemm_mxfp6_flydsl_kernel``, the tests' reference) and the persistent multi-tile kernel with a hardware K loop
(``gemm_mxfp6_persistent``, the one callers use; see its section further down).

C = A @ B^T, A [M, K] and B [N, K] in E2M3 with one E8M0 scale per 32 K.

Operand format ("plain-row planes"). The 16x16x128 f8f6f4 MFMA takes a 6-VGPR FP6 operand: a
lane holds 32 consecutive K values of one row, 6-bit codes packed little-endian into 24 bytes
(value i at bits 6i..6i+5). The shipped A6W6 kernel fills VGPRs 0-3 from one plane
(``ds_read_b128``) and 4-5 from another (``ds_read_b64``); the planes are simply bytes 0-15
and 16-23 of that 24-byte group (what ``__builtin_amdgcn_cvt_scalef32_2xpk16_fp6_f32``
emits and AITER's ``mxfp6_c0c1`` packer splits). Here the planes are plain row-major
tensors, so each one has FP4's addressing:
  * C0 ``[rows, K/2]`` uint8: bytes 16g..16g+15 of a row are bytes 0-15 of 32-group g;
  * C1 ``[rows, K/4]`` uint8: bytes 8g..8g+7 are bytes 16-23 of group g.
``kblk=True`` takes the same bytes K128-blocked (C0 ``[rows/16, K/128, 16, 64]``, C1
``[rows/32, K/128, 32, 32]``): each g2s instruction then reads one contiguous KiB instead of
16 (C0) / 32 (C1) partial cache lines; it is the format a packer should emit.
Scales are the MXFP4 kernel's packed E8M0 layout (geometry-only, format-independent), from
the MXFP4 preshuffle with the B interleave off (``preshuffle_mxfp6_scales``).

Geometry. 256x256 tile, 4 waves (2x2), a wave owns 128 rows x (2 x 64) columns: 8x8 MFMA
quads, 256 fp32 accumulators in AGPR. The K loop is a sequence of K128 *phases* (one MFMA per
quad per phase, 64 per wave); two phases make one K256 pair, which is the unit of the packed
scale layout (one dwordx4 scale load per operand per pair, VGPR-direct, two ping-pong sets).
LDS holds THREE K128 stages of both planes (48 KiB each, 144 KiB of the 160): FP4's K256 x 2
stages would need 192 KiB with the C1 plane added. Three stages buy a counter-exact pipeline
with two phases of g2s latency budget:

  phase b: MFMAs on block b (registers);   barrier (mid-phase, before the first refill):
           vmcnt(<g2s of block b+1 done>) lgkmcnt(0);  refill each fragment from block b+1's
           stage at its last use in this phase;  g2s block b+3 into block b's stage.

Every wait is derived from an issue-order model of the counters (no timing assumptions):
per-register ``lgkmcnt`` before each MFMA whose operand's two reads are not yet known
complete (2 LDS reads per fragment), ``vmcnt`` at the single per-phase barrier. The whole K
loop of the single-tile kernel is unrolled straight-line (K/128 phases); the persistent kernel
runs a hardware loop of 12 phases (lcm of 3 stages and 2 scale sets x 2 phases).

Bank swizzles (applied by permuting the g2s source, so the LDS-DMA write stays linear):
  * C0, 64-B rows, 16-B chunks: physical chunk = (g + r//2 + r//8) % 4 for row r (mod 16):
    a fragment's 8- and 16-lane groups hit distinct 16-B bank groups (conflict-free b128);
  * C1, 32-B rows, 16-B halves: physical half = (g//2) ^ (((r>>2) ^ (r>>3)) & 1): distinct
    over 32 lanes (two k-groups); 2-way at a 16-lane 128-B granule, which a 16-B g2s granule
    cannot avoid with a plain-row C1 (a packer format that interleaves row pairs could).
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm as _llvm
from flydsl.expr import arith, buffer_ops, const_expr, range_constexpr, rocdl
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

from primus_turbo.flydsl.gemm.gemm_mxfp4_kernel import (
    _MXFP4_PRESHUF_COMPILED,
    ScaleS2RPacked,
    StoreCPlain,
    _get_mxfp4_preshuffle_launch,
    _mxfp4_nt_config,
    _mxfp4_sc_unit,
    _raw,
    grouped_xcd_pid,
)
from primus_turbo.flydsl.utils.gemm_helper import (
    compile_with_scratch_out,
    make_fp8_rebased_tensor_and_srd,
    run_compiled,
)
from primus_turbo.flydsl.utils.prims import ceildiv, udiv, umod

# ── geometry ───────────────────────────────────────
_BM = _BN = 256
_NTA, _NTB = 8, 4  # A m-fragments per wave; B n-fragments per wave per half
_NQ = _NTA * _NTB  # quads per half
_NT = 2 * _NQ  # accumulator tuples (64 x 4 AGPRs)
_NSTAGE = 3  # K128 LDS stages
# Per-stage plane bytes. B holds both 128-column halves (L at +0, R at +half).
_A0_STAGE, _A1_STAGE = _BM * 64, _BM * 32
_B0_STAGE, _B1_STAGE = _BN * 64, _BN * 32
_B0_HALF, _B1_HALF = 128 * 64, 128 * 32
# Pinned VGPRs: two scale sets, then A fragments, then B fragments (6 VGPRs each).
_VSC = 8
_VFA = _VSC + 16
_VFB = _VFA + 6 * _NTA
_VEND = _VFB + 6 * 2 * _NTB
# g2s steps per wave per K128 block (1024 B per wave per step, 4 waves).
_NS_A0 = _A0_STAGE // 4096  # 4
_NS_A1 = _A1_STAGE // 4096  # 2
_NS_B0 = _B0_HALF // 4096  # 2 per half
_NS_B1 = _B1_HALF // 4096  # 1 per half
_SC_PAIR = 1024  # packed-scale bytes per (region, K256 pair) per wave region (64 lanes x 16 B)


def _d(a, b):
    return a // b if isinstance(a, int) else udiv(a, b)


def _m(a, b):
    return a % b if isinstance(a, int) else umod(a, b)


def c0_swz(r16):
    """C0 chunk rotation for row ``r16`` (= row % 16)."""
    return _d(r16, 2) + _d(r16, 8)


def c1_swz(r16):
    """C1 half flip (0/1) for row ``r16`` (= row % 16)."""
    return _m(_d(r16, 4) + _d(r16, 8), 2)


def c0_read_off(row, g):
    """LDS byte offset (within one stage) of lane (row, k-group g)'s 16 C0 bytes."""
    return row * 64 + _m(g + c0_swz(_m(row, 16)), 4) * 16


def c1_read_off(row, g):
    """LDS byte offset (within one stage) of lane (row, k-group g)'s 8 C1 bytes."""
    return row * 32 + _m(_d(g, 2) + c1_swz(_m(row, 16)), 2) * 16 + _m(g, 2) * 8


def c0_g2s_src(lane_id, wave_id, st, row_pitch, kblk=False):
    """Source byte offset (tile-relative, K-block 0) a lane fetches for C0 g2s step ``st``.
    The DMA writes lane l at stage byte (st*4 + wave)*1024 + 16 l: row (..)*16 + l//4,
    physical chunk l%4, which holds logical chunk (p - swz) % 4."""
    r16 = _d(lane_id, 4)
    row = (st * 4 + wave_id) * 16 + r16
    chunk = _m(_m(lane_id, 4) + 4 - _m(c0_swz(r16), 4), 4)
    if kblk:  # [rows/16, K/128, 16, 64]: a step's 16 rows x 64 B are one contiguous 1 KiB
        return (st * 4 + wave_id) * (16 * row_pitch) + r16 * 64 + chunk * 16
    return row * row_pitch + chunk * 16


def c1_g2s_src(lane_id, wave_id, st, row_pitch, kblk=False):
    """C1 counterpart: lane l writes row (st*4 + wave)*32 + l//2, physical half l%2."""
    rl = _d(lane_id, 2)
    row = (st * 4 + wave_id) * 32 + rl
    half = _m(_m(lane_id, 2) + c1_swz(_m(rl, 16)), 2)
    if kblk:  # [rows/32, K/128, 32, 32]
        return (st * 4 + wave_id) * (32 * row_pitch) + rl * 32 + half * 16
    return row * row_pitch + half * 16


# ── whole-K-loop asm emitter ─────────────────────────────────────────────────


def _phase_cells(mblk):
    """MFMA order within a phase: blocked-diagonal over (A rows, 8 columns) like the MXFP4
    emitter's `_MXFP4_MBLK`. Column c < 4 is BL n-fragment c, c >= 4 is BR n-fragment c-4."""
    bm, bn = mblk
    nib, ncb = _NTA // bm, 2 * _NTB // bn
    cells = []
    for D in range(nib + ncb - 1):
        for iib in range(nib):
            cb = D - iib
            if 0 <= cb < ncb:
                for di in range(bm):
                    for dj in range(bn):
                        col = cb * bn + dj
                        cells.append((iib * bm + di, col // _NTB, col % _NTB))
    return cells


_ASM_CACHE: dict = {}


def mxfp6_wholeloop_asm(NB, K2_0, K2_1, mblk=(8, 4), kblk=False, tacc=False):
    """(asm, constraints, struct type) for the straight-line K loop over ``NB`` K128 blocks."""
    key = (NB, K2_0, K2_1, tuple(mblk), kblk, tacc)
    kst0, kst1 = (1024, 1024) if kblk else (64, 32)  # K128-block stride of the C0 / C1 source
    if key in _ASM_CACHE:
        return _ASM_CACHE[key]
    assert NB % 2 == 0 and NB >= 2
    KP = NB // 2
    # operand numbering: outputs first
    o_sa0, o_sbr0, o_sa1, o_sbr1, o_ssa, o_ssb = range(_NT, _NT + 6)
    i = _NT + 6
    i_rd = dict(a0=i, a1=i + 1, b0=i + 2, b1=i + 3)
    i += 4
    i_mb = dict(a0=i, a1=i + 1, b0=i + 2, b1=i + 3)
    i += 4
    i_voa0 = list(range(i, i + _NS_A0))
    i += _NS_A0
    i_voa1 = list(range(i, i + _NS_A1))
    i += _NS_A1
    i_vob0 = list(range(i, i + _NS_B0))
    i += _NS_B0
    i_vob1 = list(range(i, i + _NS_B1))
    i += _NS_B1
    i_rs = dict(a0=i, a1=i + 1, b0=i + 2, b1=i + 3)
    i += 4
    i_srA, i_srB, i_scv, i_scsA, i_scsB = i, i + 1, i + 2, i + 3, i + 4
    i += 5
    n_in = i - (_NT + 6)

    def fa(ii):
        return _VFA + 6 * ii

    def fb(sl, ji):
        return _VFB + 6 * (sl * _NTB + ji)

    def acc(q):
        return f"a[{4 * q}:{4 * q + 3}]"

    # ---- counter model ----
    vm = []  # tags of issued vmem ops, in order
    nlg = [0]  # LDS reads issued
    lg_done = [-1]  # every read with index <= this is known complete
    rd_idx = {}  # fragment -> index of its last issued read

    def g2s(kb):
        buf = kb % _NSTAGE
        L = [
            f"s_mov_b32 ${o_sa0}, {kb * kst0}",
            f"s_mov_b32 ${o_sbr0}, {kb * kst0 + 128 * K2_0}",
            f"s_mov_b32 ${o_sa1}, {kb * kst1}",
            f"s_mov_b32 ${o_sbr1}, {kb * kst1 + 128 * K2_1}",
        ]
        G = []
        for st in range(_NS_A0):
            G.append((i_mb["a0"], buf * _A0_STAGE + st * 4096, i_voa0[st], i_rs["a0"], o_sa0))
        for st in range(_NS_A1):
            G.append((i_mb["a1"], buf * _A1_STAGE + st * 4096, i_voa1[st], i_rs["a1"], o_sa1))
        for sl, so in ((0, o_sa0), (1, o_sbr0)):
            for st in range(_NS_B0):
                G.append(
                    (i_mb["b0"], buf * _B0_STAGE + sl * _B0_HALF + st * 4096, i_vob0[st], i_rs["b0"], so)
                )
        for sl, so in ((0, o_sa1), (1, o_sbr1)):
            for st in range(_NS_B1):
                G.append(
                    (i_mb["b1"], buf * _B1_STAGE + sl * _B1_HALF + st * 4096, i_vob1[st], i_rs["b1"], so)
                )
        ops = []
        for mb, imm, vo, rs, so in G:
            ops.append(f"s_add_u32 m0, ${mb}, {imm}\nbuffer_load_dwordx4 ${vo}, ${rs}, ${so} offen lds")
        return L, ops

    def issue_g2s(line_list, op, kb):
        line_list.append(op)
        vm.append(("g2s", kb))

    def sc_load(p):
        t = p % 2
        a = _VSC + 8 * t
        L = [
            f"s_add_u32 ${o_ssa}, ${i_scsA}, {p * _SC_PAIR}",
            f"buffer_load_dwordx4 v[{a}:{a + 3}], ${i_scv}, ${i_srA}, ${o_ssa} offen",
            f"s_add_u32 ${o_ssb}, ${i_scsB}, {p * _SC_PAIR}",
            f"buffer_load_dwordx4 v[{a + 4}:{a + 7}], ${i_scv}, ${i_srB}, ${o_ssb} offen",
        ]
        vm.append(("sc", p))
        vm.append(("sc", p))
        return L

    def reads(frag, kb):
        buf = kb % _NSTAGE
        if frag[0] == "a":
            ii = frag[1]
            r = fa(ii)
            o0 = buf * _A0_STAGE + ii * 1024
            o1 = buf * _A1_STAGE + ii * 512
            b0, b1 = i_rd["a0"], i_rd["a1"]
        else:
            sl, ji = frag[1], frag[2]
            r = fb(sl, ji)
            o0 = buf * _B0_STAGE + sl * _B0_HALF + ji * 1024
            o1 = buf * _B1_STAGE + sl * _B1_HALF + ji * 512
            b0, b1 = i_rd["b0"], i_rd["b1"]
        L = [
            f"ds_read_b128 v[{r}:{r + 3}], ${b0} offset:{o0}",
            f"ds_read_b64 v[{r + 4}:{r + 5}], ${b1} offset:{o1}",
        ]
        nlg[0] += 2
        rd_idx[frag] = nlg[0] - 1
        return L

    def vm_wait(required):
        """vmcnt that guarantees every op tagged in ``required`` has completed."""
        idx = [k for k, t in enumerate(vm) if t in required]
        if not idx:
            return None
        return max(0, min(63, len(vm) - 1 - max(idx)))

    cells = _phase_cells(mblk)
    frags_all = [("a", ii) for ii in range(_NTA)] + [("b", sl, ji) for sl in range(2) for ji in range(_NTB)]

    L = ["s_waitcnt vmcnt(0) lgkmcnt(0)"]
    # ---- prologue: scales of pair 0, g2s of blocks 0..2, publish block 0, read it ----
    L += sc_load(0)
    for kb in range(min(_NSTAGE, NB)):
        sl_, ops = g2s(kb)
        L += sl_
        for op in ops:
            issue_g2s(L, op, kb)
    L.append(f"s_waitcnt vmcnt({vm_wait({('g2s', 0), ('sc', 0)})})")
    L.append("s_barrier")
    for f in frags_all:
        L += reads(f, 0)

    for b in range(NB):
        p, s = divmod(b, 2)
        t = p % 2
        if s == 0 and p + 1 < KP:
            L += sc_load(p + 1)
        refill = b + 1 < NB
        g2s_kb = b + _NSTAGE if b + _NSTAGE < NB else None
        last = {}
        for mi, (ii, sl, ji) in enumerate(cells):
            last[("a", ii)] = mi
            last[("b", sl, ji)] = mi
        bar_mi = min(last.values())  # first refill point
        rf_slots = set(last.values())
        g2s_ops = []
        if g2s_kb is not None:
            g2s_pre, g2s_ops = g2s(g2s_kb)
        else:
            g2s_kb = None
        free = [mi for mi in range(bar_mi + 1, len(cells)) if mi not in rf_slots]
        gap = max(len(free) // max(len(g2s_ops), 1), 1)
        gslots = {}
        for k, mi in enumerate(free):
            if k % gap == 0 and len(gslots) < len(g2s_ops):
                gslots[mi] = len(gslots)
        assert len(gslots) == len(g2s_ops)
        nxt = ("g2s", b + 1)
        sc_need = ("sc", (b + 1) // 2)
        for mi, (ii, sl, ji) in enumerate(cells):
            fA, fB = ("a", ii), ("b", sl, ji)
            need = max(rd_idx[fA], rd_idx[fB])
            if need > lg_done[0]:
                cnt = min(15, nlg[0] - 1 - need)
                L.append(f"s_waitcnt lgkmcnt({cnt})")
                lg_done[0] = nlg[0] - 1 - cnt
            q = sl * _NQ + ii * _NTB + ji
            oa, ob = ii % 4, ji
            sat = _VSC + 8 * t + (ii // 4) * 2 + s
            sbt = _VSC + 8 * t + 4 + sl * 2 + s
            src2 = "0" if b == 0 else acc(q)
            va, vb = f"v[{fa(ii)}:{fa(ii) + 5}]", f"v[{fb(sl, ji)}:{fb(sl, ji) + 5}]"
            if tacc:  # acc = C^T: swap the operands, their scales and op_sel halves
                osel = f"op_sel:[{ob & 1},{oa & 1},0] op_sel_hi:[{ob >> 1},{oa >> 1},0]"
                L.append(
                    f"v_mfma_scale_f32_16x16x128_f8f6f4 {acc(q)}, {vb}, {va}, {src2}, v{sbt}, v{sat} {osel} cbsz:2 blgp:2"
                )
            else:
                osel = f"op_sel:[{oa & 1},{ob & 1},0] op_sel_hi:[{oa >> 1},{ob >> 1},0]"
                L.append(
                    f"v_mfma_scale_f32_16x16x128_f8f6f4 {acc(q)}, {va}, {vb}, {src2}, v{sat}, v{sbt} {osel} cbsz:2 blgp:2"
                )
            if refill and mi == bar_mi:
                req = {nxt, sc_need} if (b + 1) < NB else set()
                w = vm_wait(req)
                L.append(f"s_waitcnt vmcnt({w if w is not None else 63}) lgkmcnt(0)")
                lg_done[0] = nlg[0] - 1
                L.append("s_barrier")
                if g2s_kb is not None:
                    L += g2s_pre
            if refill:
                for f in (fA, fB):
                    if last[f] == mi:
                        L += reads(f, b + 1)
            if mi in gslots:
                issue_g2s(L, g2s_ops[gslots[mi]], g2s_kb)
    L.append("s_waitcnt vmcnt(0) lgkmcnt(0)")
    asm = "\n".join(L)

    cons = (
        [f"={{a[{4 * q}:{4 * q + 3}]}}" for q in range(_NT)]
        + ["=&s"] * 6
        + ["v"] * 4  # LDS read bases
        + ["s"] * 4  # g2s M0 bases
        + ["v"] * (_NS_A0 + _NS_A1 + _NS_B0 + _NS_B1)  # g2s voffsets
        + ["s"] * 4  # operand SRDs
        + ["s", "s", "v", "s", "s"]  # scale SRDs, voffset, soffsets
        + [f"~{{v{r}}}" for r in range(_VSC, _VEND)]
        + ["~{scc}"]  # m0 is reserved (LLVM warns on clobbering it); nothing around the asm keeps state in it
    )
    st = "!llvm.struct<(" + ", ".join(["vector<4xf32>"] * _NT + ["i32"] * 6) + ")>"
    assert n_in == 4 + 4 + (_NS_A0 + _NS_A1 + _NS_B0 + _NS_B1) + 4 + 5
    _ASM_CACHE[key] = (asm, ",".join(cons), st)
    return _ASM_CACHE[key]


# ── kernel factory ───────────────────────────────────────────────────────────


def _build_mxfp6_gemm_kernel(*, K, mn, group_m=4, num_xcds=8, group_n=0, mblk=(8, 4), kblk=False, tacc=False):
    assert K % 256 == 0
    NB = K // 128
    K128 = K // 128
    K2_0, K2_1 = K // 2, K // 4
    M, N = mn
    assert M % _BM == 0 and N % _BN == 0, "single-tile kernel: M, N multiples of 256"
    n_pids = (M // _BM, N // _BN)
    asm, cons, st = mxfp6_wholeloop_asm(NB, K2_0, K2_1, mblk, kblk, tacc)

    _anns = {
        "A0": fx.Array[fx.Float8E4M3FN, _NSTAGE * _A0_STAGE, 16],
        "A1": fx.Array[fx.Float8E4M3FN, _NSTAGE * _A1_STAGE, 16],
        "B0": fx.Array[fx.Float8E4M3FN, _NSTAGE * _B0_STAGE, 16],
        "B1": fx.Array[fx.Float8E4M3FN, _NSTAGE * _B1_STAGE, 16],
    }
    SharedStorageFp6 = fx.struct(type("SharedStorageFp6_4w", (), {"__annotations__": _anns}))

    @flyc.kernel(known_block_size=[256, 1, 1])
    def kernel_gemm_mxfp6_4w(
        A0: fx.Tensor,
        A1: fx.Tensor,
        B0: fx.Tensor,
        B1: fx.Tensor,
        C: fx.Tensor,
        A_scale: fx.Tensor,
        B_scale: fx.Tensor,
        c_m: fx.Int32,
        c_n: fx.Int32,
    ):
        F8 = fx.Float8E4M3FN.ir_type
        lds = fx.SharedAllocator().allocate(SharedStorageFp6).peek()
        _cm, _cn = fx.Int32(M), fx.Int32(N)
        lane_id = umod(fx.thread_idx.x, 64)
        wave_id = udiv(fx.thread_idx.x, 64)
        wave_m = udiv(wave_id, 2)
        wave_n = umod(wave_id, 2)
        lane16 = umod(lane_id, 16)
        g = udiv(lane_id, 16)

        bm, bn = grouped_xcd_pid(
            fx.block_idx.x,
            c_m,
            c_n,
            _BM,
            _BN,
            group_m=group_m,
            num_xcds=num_xcds,
            group_n=group_n,
            n_pids=n_pids,
        )

        def _srd(t, pitch, row0, rows):
            base = arith.index_cast(T.index, row0) * arith.index(pitch)
            nrec = (arith.index_cast(T.index, rows) - arith.index_cast(T.index, row0)) * arith.index(pitch)
            _, r = make_fp8_rebased_tensor_and_srd(t, F8, base, nrec)
            return r

        row_a = bm * fx.Int32(_BM)
        row_b = bn * fx.Int32(_BN)
        rs_a0 = _srd(A0, K2_0, row_a, _cm)
        rs_a1 = _srd(A1, K2_1, row_a, _cm)
        rs_b0 = _srd(B0, K2_0, row_b, _cn)
        rs_b1 = _srd(B1, K2_1, row_b, _cn)

        def _lds(arr):
            return fx.Int32(fx.ptrtoint(arr.ptr))

        ra = wave_m * fx.Int32(_NTA * 16) + lane16
        rb = wave_n * fx.Int32(_NTB * 16) + lane16
        rd_a0 = _lds(lds.A0) + c0_read_off(ra, g)
        rd_a1 = _lds(lds.A1) + c1_read_off(ra, g)
        rd_b0 = _lds(lds.B0) + c0_read_off(rb, g)
        rd_b1 = _lds(lds.B1) + c1_read_off(rb, g)

        def _mb(arr):
            return rocdl.readfirstlane(T.i32, _lds(arr) + wave_id * fx.Int32(1024))

        mb = [_mb(lds.A0), _mb(lds.A1), _mb(lds.B0), _mb(lds.B1)]
        vo = (
            [fx.Int32(c0_g2s_src(lane_id, wave_id, s_, K2_0, kblk)) for s_ in range(_NS_A0)]
            + [fx.Int32(c1_g2s_src(lane_id, wave_id, s_, K2_1, kblk)) for s_ in range(_NS_A1)]
            + [fx.Int32(c0_g2s_src(lane_id, wave_id, s_, K2_0, kblk)) for s_ in range(_NS_B0)]
            + [fx.Int32(c1_g2s_src(lane_id, wave_id, s_, K2_1, kblk)) for s_ in range(_NS_B1)]
        )

        # Packed scales: identical layout to the MXFP4 kernel at the 256 tile with b_ilv=0.
        sa = ScaleS2RPacked(A_scale, n_pids[0] * 256, K, 4)
        sb = ScaleS2RPacked(B_scale, n_pids[1] * 256, K, 4)
        wia = bm * fx.Int32(2) + wave_m
        wib = bn * fx.Int32(2) + wave_n
        scs_a = rocdl.readfirstlane(T.i32, wia * fx.Int32(K128 * 512))
        scs_b = rocdl.readfirstlane(T.i32, wib * fx.Int32(K128 * 512))
        scv = lane_id * fx.Int32(16)

        ins = [
            rd_a0,
            rd_a1,
            rd_b0,
            rd_b1,
            *mb,
            *vo,
            rs_a0,
            rs_a1,
            rs_b0,
            rs_b1,
            sa.rsrc,
            sb.rsrc,
            scv,
            scs_a,
            scs_b,
        ]
        r = _llvm.inline_asm(ir.Type.parse(st), [_raw(x) for x in ins], asm, cons, has_side_effects=True)
        accs = [Vec(_llvm.extractvalue(ir.Type.parse("vector<4xf32>"), r, [q])) for q in range(_NT)]

        store_c = StoreCPlain(
            C, _cm, _cn, lambda i_, j_: i_ * _NTB + j_, _NTA, _NTB, fx.BFloat16, ilv=0, store_aux=2
        )
        base_row = row_a + wave_m * fx.Int32(_NTA * 16)
        col_l = row_b + wave_n * fx.Int32(_NTB * 16)
        if tacc:  # the MXFP4 kernel's TACCW epilogue: permlane16 swaps + dwordx4 stores
            store_c.store_tacc_wide(accs[:_NQ], base_row, col_l)
            store_c.store_tacc_wide(accs[_NQ:], base_row, col_l + fx.Int32(128))
        else:
            store_c.store(accs[:_NQ], base_row, col_l)
            store_c.store(accs[_NQ:], base_row, col_l + fx.Int32(128))

    _pt = {"passthrough": [["amdgpu-agpr-alloc", "256"]]}
    attrs = {"rocdl.flat_work_group_size": "256,256", "rocdl.waves_per_eu": 1, **_pt}
    grid = n_pids[0] * n_pids[1]

    @flyc.jit
    def launch_mxfp6(
        A0: fx.Tensor,
        A1: fx.Tensor,
        B0: fx.Tensor,
        B1: fx.Tensor,
        C: fx.Tensor,
        A_scale: fx.Tensor,
        B_scale: fx.Tensor,
        c_m: fx.Int32,
        c_n: fx.Int32,
        stream: fx.Stream,
    ):
        kernel_gemm_mxfp6_4w(A0, A1, B0, B1, C, A_scale, B_scale, c_m, c_n, value_attrs=attrs).launch(
            grid=(grid, 1, 1), block=(256, 1, 1), stream=stream
        )

    return launch_mxfp6


# ── host wrappers ────────────────────────────────────────────────────────────

_LAUNCH_CACHE: dict = {}
_COMPILED: dict = {}


def preshuffle_mxfp6_scales(a_scale, b_scale, M, N, K):
    """Canonical E8M0 ``[dim, K/32]`` -> the packed layout the MXFP6 kernels read: the
    MXFP4 preshuffle at the 256 tile with the B interleave off (b_ilv=0; these kernels have no
    folded C store, which is what turns the interleave on in the MXFP4 kernel)."""
    assert K % 256 == 0 and a_scale.shape[1] * 32 == K and b_scale.shape[1] * 32 == K
    k128 = K // 128
    sc_row = K // 32
    launch = _get_mxfp4_preshuffle_launch(b_ilv=0, sc_row=sc_row, src_unit=_mxfp4_sc_unit(sc_row), k128=k128)
    a_sp = torch.empty(ceildiv(M, 256) * 256 * k128, dtype=torch.int32, device=a_scale.device)
    b_sp = torch.empty(ceildiv(N, 256) * 256 * k128, dtype=torch.int32, device=b_scale.device)
    run_compiled(
        _MXFP4_PRESHUF_COMPILED,
        (0, sc_row, k128, M, N),
        launch,
        a_scale.contiguous().view(torch.int8),
        a_sp,
        b_scale.contiguous().view(torch.int8),
        b_sp,
        M,
        N,
        torch.cuda.current_stream(),
    )
    return a_sp, b_sp


def gemm_mxfp6_flydsl_kernel(
    a_c0, a_c1, b_c0, b_c1, a_sp, b_sp, *, out=None, mblk=(8, 4), swizzle=None, kblk=False, tacc=False
):
    """C[M, N] (bf16) = A @ B^T for E2M3 operands given as plain-row planes (module doc) and
    scales packed by ``preshuffle_mxfp6_scales``. ``kblk``: the planes are K128-blocked instead
    (C0 ``[rows/16, K/128, 16, 64]``, C1 ``[rows/32, K/128, 32, 32]``, same bytes per row and
    K128 block): every g2s instruction then reads one contiguous KiB (``kblk_planes``)."""
    M, N = a_c0.shape[0], b_c0.shape[0]
    K = a_c0.shape[1] * 2
    assert a_c1.shape == (M, K // 4) and b_c0.shape == (N, K // 2) and b_c1.shape == (N, K // 4)
    gm, gn, xcd = swizzle if swizzle is not None else _mxfp4_nt_config(M, N, K)
    key = (M, N, K, gm, gn, xcd, tuple(mblk), kblk, tacc)
    launch = _LAUNCH_CACHE.get(key)
    if launch is None:
        launch = _build_mxfp6_gemm_kernel(
            K=K, mn=(M, N), group_m=gm, num_xcds=xcd, group_n=gn, mblk=mblk, kblk=kblk, tacc=tacc
        )
        _LAUNCH_CACHE[key] = launch
    if out is None:
        out = torch.empty((M, N), dtype=torch.bfloat16, device=a_c0.device)
    args = (
        a_c0.view(torch.int8),
        a_c1.view(torch.int8),
        b_c0.view(torch.int8),
        b_c1.view(torch.int8),
        out,
        a_sp.view(torch.int32).reshape(-1),
        b_sp.view(torch.int32).reshape(-1),
        M,
        N,
        torch.cuda.current_stream(),
    )
    comp = _COMPILED.get(key)
    if comp is None:
        comp = compile_with_scratch_out(launch, args, out_index=4)
        _COMPILED[key] = comp
    comp(*args)
    return out


def kblk_planes(c0, c1):
    """Plain-row planes -> the K128-blocked planes ``kblk=True`` reads (same tensor shapes)."""
    R, K2 = c0.shape
    nb = K2 // 64
    b0 = c0.view(R // 16, 16, nb, 64).permute(0, 2, 1, 3).contiguous().view(R, K2)
    b1 = c1.view(R // 32, 32, nb, 32).permute(0, 2, 1, 3).contiguous().view(R, K2 // 2)
    return b0, b1


# ══ persistent multi-tile kernel (one asm: tile loop + hardware K loop) ══════════════════
#
# One workgroup walks TPW output tiles inside ONE inline-asm block. The K-block pipeline runs
# straight across tile boundaries: the last three K128 steps of tile t (which have no g2s of
# their own left) fill the stages with blocks 0..2 of tile t+1, and the step before loads
# t+1's first scale pair, so the next tile starts on resident data. The C store is folded
# into tile t's last K128 step (TACC accumulators: a lane holds 4 consecutive columns of one
# row -> v_cvt_pk_bf16_f32 x2 + one buffer_store_dwordx2 per accumulator), issued as the
# accumulators retire, and it is younger than t+1's prefetch, so t+1's waits only reach it
# from its third K128 step on. Operand/scale/C SRDs cover whole tensors; a tile is selected
# purely through SGPR soffsets derived in asm from a packed (bm | bn << 16) per tile.
# The K loop is a hardware loop over a 12-step period (3 stages x 2 scale sets x 2 steps),
# found by checking the generated per-step text for periodicity, so every static wait in the
# loop body is the one the counter model derived for each of its iterations.

_VST = _VEND  # store temps: 2 banks x (4 accvgpr reads + 2 packed dwords)
_VST_END = _VST + 24
_VD = _VST_END  # 16 deferred store units x 4 packed dwords
_VD_END = _VD + 64
_VPF = _VD_END  # L2-prefetch sink (results discarded)
_VBP = _VD + 16  # bias epilogue: 8 x 2 packed bf16 (ND <= 4 keeps the deferred bank below)
_VBF = _VD + 32  # bias epilogue: 32 fp32 columns, 4 per (half, n-fragment)
_BAR_DEFAULT = 8  # persistent kernel: per-step barrier after this MFMA (or the first refill, if earlier)
_STORE_AGE = 4  # MFMAs between an accumulator's last MFMA and its first accvgpr_read


def mxfp6_persistent_asm(
    NB,
    K2_0,
    K2_1,
    ldc_b,
    TPW,
    mblk=(8, 4),
    kloop=True,
    tune=(),
    nst=_NSTAGE,
    l2pf=0,
    bias=False,
    layout="kblk",
    guard=None,
):
    """``tune``: schedule options that leave the result unchanged -- "bar=k" (per-step barrier after MFMA k),
    "defer=n" (deferred C store units), "stmod=..." (C store cache modifiers, "none" for none), "rq=n" (queued
    refills, at most n fragments per MFMA), "g2srr" (stream-interleaved g2s), "wmaj" (wave-major g2s), "endstore"
    (C store after the last MFMA instead of folded into the last K128 step). ``nst``: K128 LDS stages (3 = 144
    KiB, the default; 2 also works). ``l2pf``: L2 prefetch distance in K128 steps (0 = off)."""
    key = (
        ("persist", NB, K2_0, K2_1, ldc_b, TPW, tuple(mblk), kloop, tuple(sorted(tune)), nst, l2pf, bias)
        + ((layout,) if layout != "kblk" else ())
        + ((("guard",) + tuple(guard)) if guard else ())
    )
    # l2pf = d > 0: at the end of each K128 step also touch block b + nst + d of the current tile
    # with plain buffer_load_dword (one lane per 128-B line, result discarded): an L2 prefetch
    # running d steps ahead of the LDS pipeline, which LDS capacity caps at nst stages. The
    # loads sit after the step's g2s, so only waits that already reach younger g2s cover them.
    assert 0 <= l2pf <= 3
    assert nst in (2, 3) and NB >= 2 * nst
    if key in _ASM_CACHE:
        return _ASM_CACHE[key]
    KP = NB // 2
    assert NB % 2 == 0 and KP % 2 == 0 and NB >= 6, "persistent path: K a multiple of 512, >= 768"
    assert layout in ("kblk", "aiter")
    if layout == "kblk":
        kst0 = kst1 = 1024  # kblk planes: K128-block stride of a 16- (C0) / 32-row (C1) group
        TS0, TS1 = 256 * K2_0, 256 * K2_1  # tile (256-row) stride of the C0 / C1 operand
        BR0, BR1 = 128 * K2_0, 128 * K2_1  # B's right half (rows 128..255) within a tile
        C1OFF = 0
        SCT, SCP = NB * 1024, _SC_PAIR  # packed-scale region stride per tile row / per K256 pair
    else:
        # AITER mxfp6_c0c1_256_padk2: one blob per operand, a 24 KiB [C0 16 KiB | C1 8 KiB] unit
        # per (256-row tile, K128 step), K/128 + 2 steps per tile row; scales 1 KiB per unit.
        assert l2pf == 0, "l2pf addresses kblk planes"
        kst0 = kst1 = 24576
        TS0 = TS1 = (NB + 2) * 24576
        BR0, BR1 = 8192, 4096
        C1OFF = 16384
        SCT, SCP = (NB + 2) * 1024, 2048
    st_mod = next((" " + a.split("=", 1)[1].replace("+", " ") for a in tune if a.startswith("stmod=")), " nt")
    st_mod = "" if st_mod.strip() == "none" else st_mod
    n = [0]

    def take(k):
        r = list(range(n[0], n[0] + k))
        n[0] += k
        return r

    oA0, oA1, oBL0, oBR0, oBL1, oBR1 = take(6)
    oscA, oscB = take(2)
    osC, osCp = take(2)
    onA0, onA1, onB0, onB1 = take(4)
    ot0, ot1, ot2 = take(3)
    okc, otc = take(2)
    o_slot = take(TPW + 1)
    n_out = n[0]
    i_rd = dict(zip(("a0", "a1", "b0", "b1"), take(4)))
    i_mb = dict(zip(("a0", "a1", "b0", "b1"), take(4)))
    i_voa0, i_voa1, i_vob0, i_vob1 = take(_NS_A0), take(_NS_A1), take(_NS_B0), take(_NS_B1)
    i_rs = dict(zip(("a0", "a1", "b0", "b1"), take(4)))
    i_srA, i_srB = take(2)
    (i_scv,) = take(1)
    i_wA, i_wB = take(2)
    (i_srC,) = take(1)
    i_vc = take(_NTA)
    take(TPW + 1)  # the packed (bm | bn << 16) per-tile slots: operands, read through o_slot
    i_pf = dict(zip(("a0", "a1", "b0", "b1"), take(4)))  # per-lane L2-prefetch line offsets
    if bias:
        # bf16 bias[N]: SRD over the whole vector, and this lane's column byte offset in a tile
        # ((wave_n * 64 + 4 * (lane // 16)) * 2; the tile's bn * 512 rides the soffset).
        i_bsrd, i_bvo = take(2)
    if guard:
        # Shape guard (AITER-ABI build): the launcher's M, N, K (it passes the 128-padded K) and
        # stride_D0 (elements) must equal what this code object was compiled for, else s_trap.
        i_guard = take(4)

    def fa(ii):
        return _VFA + 6 * ii

    def fb(sl, ji):
        return _VFB + 6 * (sl * _NTB + ji)

    def acc(q):
        return f"a[{4 * q}:{4 * q + 3}]"

    # ---- SALU helpers ----
    def unpack(slot):
        return [f"s_and_b32 ${ot0}, ${slot}, 0xffff", f"s_lshr_b32 ${ot1}, ${slot}, 16"]

    def running_from(bA0, bA1, bB0, bB1, j):
        """Running g2s soffsets for K128 block j of the tile whose bases are given."""
        return [
            f"s_add_u32 ${oA0}, {bA0}, {j * kst0}",
            f"s_add_u32 ${oA1}, {bA1}, {C1OFF + j * kst1}",
            f"s_add_u32 ${oBL0}, {bB0}, {j * kst0}",
            f"s_add_u32 ${oBR0}, {bB0}, {BR0 + j * kst0}",
            f"s_add_u32 ${oBL1}, {bB1}, {C1OFF + j * kst1}",
            f"s_add_u32 ${oBR1}, {bB1}, {C1OFF + BR1 + j * kst1}",
        ]

    def bases_into(dst, slot):
        """dst = (A0, A1, B0, B1) tile byte bases of the packed tile in ``slot``."""
        return unpack(slot) + [
            f"s_mul_i32 ${dst[0]}, ${ot0}, {TS0}",
            f"s_mul_i32 ${dst[1]}, ${ot0}, {TS1}",
            f"s_mul_i32 ${dst[2]}, ${ot1}, {TS0}",
            f"s_mul_i32 ${dst[3]}, ${ot1}, {TS1}",
        ]

    def advance():
        return [
            f"s_add_u32 ${r}, ${r}, {kst0 if r in (oA0, oBL0, oBR0) else kst1}"
            for r in (oA0, oA1, oBL0, oBR0, oBL1, oBR1)
        ]

    def tile_entry():
        """Current tile (slot 0): running g2s soffsets at block 3, scale soffsets at pair 1, C base
        (the previous tile's C base moves to osCp for its deferred stores)."""
        L = [f"s_mov_b32 ${osCp}, ${osC}"] + bases_into((onA0, onA1, onB0, onB1), f"{o_slot[0]}")
        L += running_from(f"${onA0}", f"${onA1}", f"${onB0}", f"${onB1}", nst)
        L += [
            f"s_mul_i32 ${oscA}, ${ot0}, {SCT}",
            f"s_add_u32 ${oscA}, ${oscA}, ${i_wA}",
            f"s_add_u32 ${oscA}, ${oscA}, {SCP}",
            f"s_mul_i32 ${oscB}, ${ot1}, {SCT}",
            f"s_add_u32 ${oscB}, ${oscB}, ${i_wB}",
            f"s_add_u32 ${oscB}, ${oscB}, {SCP}",
            f"s_mul_i32 ${osC}, ${ot0}, {256 * ldc_b}",
            f"s_mul_i32 ${ot2}, ${ot1}, 512",
            f"s_add_u32 ${osC}, ${osC}, ${ot2}",
        ]
        return L

    # ---- counter model ----
    sim = dict(vm=[], nlg=0, lg_done=-1, rd={})

    def g2s_ops(stage, tag):
        G = []
        for st in range(_NS_A0):
            G.append((i_mb["a0"], stage * _A0_STAGE + st * 4096, i_voa0[st], i_rs["a0"], oA0))
        for st in range(_NS_A1):
            G.append((i_mb["a1"], stage * _A1_STAGE + st * 4096, i_voa1[st], i_rs["a1"], oA1))
        for sl, so in ((0, oBL0), (1, oBR0)):
            for st in range(_NS_B0):
                G.append(
                    (i_mb["b0"], stage * _B0_STAGE + sl * _B0_HALF + st * 4096, i_vob0[st], i_rs["b0"], so)
                )
        for sl, so in ((0, oBL1), (1, oBR1)):
            for st in range(_NS_B1):
                G.append(
                    (i_mb["b1"], stage * _B1_STAGE + sl * _B1_HALF + st * 4096, i_vob1[st], i_rs["b1"], so)
                )
        if "g2srr" in tune:  # interleave the streams (A0 B0 A1 B1 ...) instead of stream-major
            by = {}
            for e in G:
                by.setdefault(e[0], []).append(e)
            lists = list(by.values())
            G = [l_[k] for k in range(max(len(x) for x in lists)) for l_ in lists if k < len(l_)]
        if "wmaj" in tune:
            # Wave-major fill: a wave's steps of one stream are 1 KiB apart in LDS, so they share
            # one M0 write and ride the instruction offset (which also offsets the global address;
            # the voffsets come in pre-subtracted). The LDS image is the same as step-major.
            out, seen = [], set()
            for mb, imm, vo, rs, so in G:
                key_ = (mb, so)
                first = key_ not in seen
                seen.add(key_)
                out.append((mb, imm, vo, rs, so, first))
            res = []
            cnt = {}
            for mb, imm, vo, rs, so, first in out:
                k_ = (mb, so)
                stp = cnt.get(k_, 0)
                cnt[k_] = stp + 1
                base = imm - stp * 4096  # stream base (stage + half) in LDS
                m0 = f"s_add_u32 m0, ${mb}, {base}\n" if first else ""
                res.append(
                    (f"{m0}buffer_load_dwordx4 ${vo}, ${rs}, ${so} offen offset:{stp * 1024} lds", tag)
                )
            return res
        return [
            (f"s_add_u32 m0, ${mb}, {imm}\nbuffer_load_dwordx4 ${vo}, ${rs}, ${so} offen lds", tag)
            for mb, imm, vo, rs, so in G
        ]

    def vm_issue(L, line, tag):
        L.append(line)
        sim["vm"].append(tag)

    def sc_load(L, setidx, sA, sB, tag):
        a = _VSC + 8 * setidx
        if layout == "aiter":
            # AITER's scale unit: lane l's 8 bytes at half * 512 + 8 l are the 8 m- (n-) fragments'
            # scales of one K128 step, the byte order this kernel's op_sel already uses. A: both
            # dwords (fragment groups 0-3, 4-7) per step -> set [g0s0 g1s0 g0s1 g1s1]; B: this
            # wave's dword (n-fragments wave_n*4..+3) of each half per step -> [BLs0 BLs1 BRs0 BRs1].
            vm_issue(L, f"buffer_load_dwordx2 v[{a}:{a + 1}], ${i_scv}, ${i_srA}, {sA} offen", tag)
            vm_issue(
                L, f"buffer_load_dwordx2 v[{a + 2}:{a + 3}], ${i_scv}, ${i_srA}, {sA} offen offset:1024", tag
            )
            for k_, o_ in enumerate((0, 1024, 512, 1536)):
                vm_issue(
                    L, f"buffer_load_dword v{a + 4 + k_}, ${i_scv}, ${i_srB}, {sB} offen offset:{o_}", tag
                )
            return
        vm_issue(L, f"buffer_load_dwordx4 v[{a}:{a + 3}], ${i_scv}, ${i_srA}, {sA} offen", tag)
        vm_issue(L, f"buffer_load_dwordx4 v[{a + 4}:{a + 7}], ${i_scv}, ${i_srB}, {sB} offen", tag)

    def reads(frag, stage):
        if frag[0] == "a":
            ii = frag[1]
            r, o0, o1 = fa(ii), stage * _A0_STAGE + ii * 1024, stage * _A1_STAGE + ii * 512
            b0, b1 = i_rd["a0"], i_rd["a1"]
        else:
            sl, ji = frag[1], frag[2]
            r = fb(sl, ji)
            o0 = stage * _B0_STAGE + sl * _B0_HALF + ji * 1024
            o1 = stage * _B1_STAGE + sl * _B1_HALF + ji * 512
            b0, b1 = i_rd["b0"], i_rd["b1"]
        sim["nlg"] += 2
        sim["rd"][frag] = sim["nlg"] - 1
        return [
            f"ds_read_b128 v[{r}:{r + 3}], ${b0} offset:{o0}",
            f"ds_read_b64 v[{r + 4}:{r + 5}], ${b1} offset:{o1}",
        ]

    def vm_wait(required):
        idx = [k for k, t in enumerate(sim["vm"]) if t in required]
        assert idx, required
        return max(0, min(63, len(sim["vm"]) - 1 - max(idx)))

    def store_lines(q, bank, t, dst=None):
        """Store unit for the n-fragment pair ending at accumulator ``q`` (odd ji): the MXFP4
        TACCW epilogue in asm -- two permlane16 swaps turn the pair into 8 consecutive bf16
        columns per lane, one buffer_store_dwordx4 per lane (64 B per row per instruction)."""
        sl, rem = divmod(q, _NQ)
        ii, ji = divmod(rem, _NTB)
        assert ji % 2 == 1
        qa, qb = q - 1, q
        T_ = _VST + 12 * bank
        P = T_ + 8 if dst is None else dst
        L = [f"v_accvgpr_read_b32 v{T_ + e}, a{4 * qa + e}" for e in range(4)]
        L += [f"v_accvgpr_read_b32 v{T_ + 4 + e}, a{4 * qb + e}" for e in range(4)]
        if bias:  # fp32 accumulator + fp32(bias), then the one RNE of v_cvt_pk_bf16_f32 (AITER's order)
            for h, jj in ((0, ji - 1), (4, ji)):
                f_ = _VBF + (sl * _NTB + jj) * 4
                L += [
                    f"v_pk_add_f32 v[{T_ + h}:{T_ + h + 1}], v[{T_ + h}:{T_ + h + 1}], v[{f_}:{f_ + 1}]",
                    f"v_pk_add_f32 v[{T_ + h + 2}:{T_ + h + 3}], v[{T_ + h + 2}:{T_ + h + 3}], v[{f_ + 2}:{f_ + 3}]",
                ]
        L += [
            f"v_cvt_pk_bf16_f32 v{P}, v{T_}, v{T_ + 1}",
            f"v_cvt_pk_bf16_f32 v{P + 1}, v{T_ + 2}, v{T_ + 3}",
            f"v_cvt_pk_bf16_f32 v{P + 2}, v{T_ + 4}, v{T_ + 5}",
            f"v_cvt_pk_bf16_f32 v{P + 3}, v{T_ + 6}, v{T_ + 7}",
            "s_nop 4",
            f"v_permlane16_swap_b32 v{P}, v{P + 2}",
            f"v_permlane16_swap_b32 v{P + 1}, v{P + 3}",
            "s_nop 4",
        ]
        st = (
            f"buffer_store_dwordx4 v[{P}:{P + 3}], ${i_vc[ii]}, ${i_srC}, ${osC} offen "
            f"offset:{sl * 256 + (ji // 2) * 64}{st_mod}"
        )
        return L, st

    cells = _phase_cells(mblk)
    frags_all = [("a", ii) for ii in range(_NTA)] + [("b", sl, ji) for sl in range(2) for ji in range(_NTB)]
    last = {}
    for mi, (ii, sl, ji) in enumerate(cells):
        last[("a", ii)] = mi
        last[("b", sl, ji)] = mi
    bar_mi = min(last.values())
    # "bar=k": put the per-step barrier after MFMA k instead of at the first refill, so the g2s
    # can spread over more of the step (the barrier must still precede every refill).
    # Default 8: the step's g2s then spread over 3/4 of its MFMAs instead of crowding the second half, where the
    # vector-memory issue queue is already loaded by the refills.
    bar_mi = min(bar_mi, next((int(a.split("=")[1]) for a in tune if a.startswith("bar=")), _BAR_DEFAULT))
    rf_slots = set(last.values())
    free = [mi for mi in range(bar_mi + 1, len(cells)) if mi not in rf_slots]
    n_g2s = _NS_A0 + _NS_A1 + 2 * _NS_B0 + 2 * _NS_B1
    gap = max(len(free) // n_g2s, 1)
    gslot = {}
    for k, mi in enumerate(free):
        if k % gap == 0 and len(gslot) < n_g2s:
            gslot[mi] = len(gslot)
    assert len(gslot) == n_g2s

    single = TPW == 1  # one tile per workgroup: no cross-tile prefetch, exit with the stores in flight
    # Deferred C store units: the last ND of the 32 (n-fragment pair) store units retire into
    # pinned VGPRs as packed bf16 and are stored during the next tile's first K128 steps, so a
    # tile's C write is not one burst that the next tile's first in-order vmcnt wait must drain.
    ND = next((int(a.split("=")[1]) for a in tune if a.startswith("defer=")), 16)
    ND = 0 if single else ND
    if bias:  # the deferred bank's upper 48 VGPRs hold the tile's bias (16 packed bf16 + 32 fp32)
        ND = min(ND, 4)
    assert 0 <= ND <= 16 and ND % 2 == 0
    NU = 2 * _NQ // 2  # 32 store units per wave per tile
    unit_q = []  # store-unit accumulator (odd ji), in retirement order
    for ii, sl, ji in cells:
        if ji % 2:
            unit_q.append(sl * _NQ + ii * _NTB + ji)
    assert len(unit_q) == NU
    deferred = unit_q[NU - ND :]

    def dstore(k, so):
        q = deferred[k]
        sl, rem = divmod(q, _NQ)
        ii, ji = divmod(rem, _NTB)
        d = _VD + 4 * k
        return (
            f"buffer_store_dwordx4 v[{d}:{d + 3}], ${i_vc[ii]}, ${i_srC}, ${so} offen "
            f"offset:{sl * 256 + (ji // 2) * 64}{st_mod}"
        )

    def gen_tile(t):
        """(entry lines, [lines per K128 step], exit lines) for tile ``t`` of the model."""
        E = tile_entry()
        if bias:  # this tile's 32 bias columns per lane, packed bf16 (s[ot2] = bn * 512 from tile_entry)
            for sl in range(2):
                for jj in range(_NTB):
                    p_ = _VBP + (sl * _NTB + jj) * 2
                    vm_issue(
                        E,
                        f"buffer_load_dwordx2 v[{p_}:{p_ + 1}], ${i_bvo}, ${i_bsrd}, ${ot2} offen offset:{sl * 256 + jj * 32}",
                        (t, "bias"),
                    )
        E.append(f"s_waitcnt vmcnt({vm_wait({(t, 'g2s', 0), (t, 'sc', 0)})})")
        E.append("s_barrier")
        sim["lg_done"] = sim["nlg"] - 1
        for f in frags_all:
            E += reads(f, 0)
        P_ = []
        bank = [0]
        # "rq=n": refills queue at their last use and issue at most n fragments (2 reads each)
        # per MFMA, spilling into the next step up to its barrier; an MFMA whose operand is
        # still queued forces the queue up to it first.
        rq_n = next((int(a.split("=")[1]) for a in tune if a.startswith("rq=")), 0)
        rq = []

        def rq_pop(L, upto=None, n=None):
            k = len(rq) if upto is not None else min(n, len(rq))
            if upto is not None:
                k = max(i + 1 for i, (f_, _s) in enumerate(rq) if f_ in upto)
            for f_, st_ in rq[:k]:
                L += reads(f_, st_)
            del rq[:k]

        for b in range(NB):
            L = []
            p, s = divmod(b, 2)
            tset = p % 2
            if s == 0:
                if p + 1 < KP:
                    sc_load(L, (p + 1) % 2, f"${oscA}", f"${oscB}", (t, "sc", p + 1))
                    L += [f"s_add_u32 ${oscA}, ${oscA}, {SCP}", f"s_add_u32 ${oscB}, ${oscB}, {SCP}"]
                elif not single:  # next tile's pair 0 into set 0 (KP even: the last pair runs on set 1)
                    L += unpack(f"{o_slot[1]}") + [
                        f"s_mul_i32 ${ot2}, ${ot0}, {SCT}",
                        f"s_add_u32 ${ot2}, ${ot2}, ${i_wA}",
                        f"s_mul_i32 ${ot1}, ${ot1}, {SCT}",
                        f"s_add_u32 ${ot1}, ${ot1}, ${i_wB}",
                    ]
                    sc_load(L, 0, f"${ot2}", f"${ot1}", (t + 1, "sc", 0))
            refill = b + 1 < NB
            cur_g2s = b + nst < NB
            stage = b % nst
            if cur_g2s:
                ops = g2s_ops(stage, (t, "g2s", b + nst))
                pre = []
            elif single:
                ops, pre = [], []
            else:
                j = b % nst  # block j of the next tile lands in stage j == b % nst
                ops = g2s_ops(stage, (t + 1, "g2s", j))
                pre = (
                    bases_into((onA0, onA1, onB0, onB1), f"{o_slot[1]}") if b == NB - nst else []
                ) + running_from(f"${onA0}", f"${onA1}", f"${onB0}", f"${onB1}", j)
            stq = []  # (ready_mi, q) store queue for the last step
            for mi, (ii, sl, ji) in enumerate(cells):
                fA, fB = ("a", ii), ("b", sl, ji)
                if any(f_ in (fA, fB) for f_, _s in rq):
                    rq_pop(L, upto=(fA, fB))
                need = max(sim["rd"][fA], sim["rd"][fB])
                if need > sim["lg_done"]:
                    cnt = min(15, sim["nlg"] - 1 - need)
                    L.append(f"s_waitcnt lgkmcnt({cnt})")
                    sim["lg_done"] = sim["nlg"] - 1 - cnt
                q = sl * _NQ + ii * _NTB + ji
                oa, ob = ii % 4, ji
                sat = _VSC + 8 * tset + ((ii // 4) * 2 + s if layout == "kblk" else s * 2 + ii // 4)
                sbt = _VSC + 8 * tset + 4 + sl * 2 + s
                src2 = "0" if b == 0 else acc(q)
                osel = f"op_sel:[{ob & 1},{oa & 1},0] op_sel_hi:[{ob >> 1},{oa >> 1},0]"
                L.append(
                    f"v_mfma_scale_f32_16x16x128_f8f6f4 {acc(q)}, v[{fb(sl, ji)}:{fb(sl, ji) + 5}], "
                    f"v[{fa(ii)}:{fa(ii) + 5}], {src2}, v{sbt}, v{sat} {osel} cbsz:2 blgp:2"
                )
                if mi == bar_mi:
                    if rq:  # this step's g2s overwrite the stage the queued reads read
                        rq_pop(L, n=len(rq))
                    if refill:
                        w = vm_wait({(t, "g2s", b + 1), (t, "sc", (b + 1) // 2)})
                        L.append(f"s_waitcnt vmcnt({w}) lgkmcnt(0)")
                    else:
                        L.append("s_waitcnt lgkmcnt(0)")
                    sim["lg_done"] = sim["nlg"] - 1
                    L.append("s_barrier")
                    L += pre
                    if b < ND // 2:  # the previous tile's deferred store units, two per step
                        for k in (2 * b, 2 * b + 1):
                            vm_issue(L, dstore(k, osCp), (t - 1, "st", "d", k))
                if refill:
                    for f in (fA, fB):
                        if last[f] == mi:
                            if rq_n:
                                rq.append((f, (b + 1) % nst))
                            else:
                                L += reads(f, (b + 1) % nst)
                if rq and mi > bar_mi:
                    rq_pop(L, n=rq_n)
                if mi in gslot and ops:
                    line, tag = ops[gslot[mi]]
                    vm_issue(L, line, tag)
                if b == NB - 1 and bias and mi == 0:  # bias -> fp32 before the first store unit retires
                    L.append(f"s_waitcnt vmcnt({vm_wait({(t, 'bias')})})")
                    for k in range(2 * _NTB):
                        p_, f_ = _VBP + 2 * k, _VBF + 4 * k
                        L += [
                            f"v_lshlrev_b32 v{f_}, 16, v{p_}",
                            f"v_and_b32 v{f_ + 1}, 0xffff0000, v{p_}",
                            f"v_lshlrev_b32 v{f_ + 2}, 16, v{p_ + 1}",
                            f"v_and_b32 v{f_ + 3}, 0xffff0000, v{p_ + 1}",
                        ]
                if b == NB - 1:
                    if ji % 2:
                        stq.append((mi + _STORE_AGE if "endstore" not in tune else 1 << 30, q))
                    while stq and stq[0][0] <= mi:
                        _, qq = stq.pop(0)
                        if qq in deferred:
                            sl_, _ = store_lines(qq, bank[0], t, dst=_VD + 4 * deferred.index(qq))
                            bank[0] ^= 1
                            L += sl_
                            continue
                        sl_, st_ = store_lines(qq, bank[0], t)
                        bank[0] ^= 1
                        L += sl_
                        vm_issue(L, st_, (t, "st", qq))
            if l2pf and b + nst + l2pf < NB:
                for nm, so in (("a0", oA0), ("a1", oA1), ("b0", oBL0), ("b1", oBL1)):
                    vm_issue(
                        L,
                        f"buffer_load_dword v{_VPF}, ${i_pf[nm]}, ${i_rs[nm]}, ${so} offen offset:{l2pf * 1024}",
                        (t, "pf", b),
                    )
            if cur_g2s:
                L += advance()
            if b == NB - 1:
                L += ["s_nop 15", "s_nop 15"]
                for _, qq in stq:
                    if qq in deferred:
                        sl_, _ = store_lines(qq, bank[0], t, dst=_VD + 4 * deferred.index(qq))
                        bank[0] ^= 1
                        L += sl_
                        continue
                    sl_, st_ = store_lines(qq, bank[0], t)
                    bank[0] ^= 1
                    L += sl_
                    vm_issue(L, st_, (t, "st", qq))
            P_.append(L)
        X = [f"s_mov_b32 ${o_slot[k]}, ${o_slot[k + 1]}" for k in range(TPW)]
        return E, P_, X

    # ---- prologue (tile 0's blocks 0..2 and scale pair 0), drained ----
    # osC starts out of range so the first tile's "previous tile" deferred stores are dropped.
    PRO = ["s_waitcnt vmcnt(0) lgkmcnt(0)", f"s_mov_b32 ${otc}, {TPW}", f"s_mov_b32 ${osC}, 0x80000000"]
    PRO += bases_into((onA0, onA1, onB0, onB1), f"{o_slot[0]}")
    PRO += [
        f"s_mul_i32 ${ot2}, ${ot0}, {SCT}",
        f"s_add_u32 ${ot2}, ${ot2}, ${i_wA}",
        f"s_mul_i32 ${ot1}, ${ot1}, {SCT}",
        f"s_add_u32 ${ot1}, ${ot1}, ${i_wB}",
    ]
    sc_load(PRO, 0, f"${ot2}", f"${ot1}", (0, "sc", 0))
    for j in range(nst):
        PRO += running_from(f"${onA0}", f"${onA1}", f"${onB0}", f"${onB1}", j)
        for line, tag in g2s_ops(j, (0, "g2s", j)):
            vm_issue(PRO, line, tag)
    if single:
        E1, P1, X1 = gen_tile(0)
        X1 = []
    else:
        PRO.append("s_waitcnt vmcnt(0)")
        gen_tile(0)
        E1, P1, X1 = gen_tile(1)
        E2, P2, X2 = gen_tile(2)
        assert (E1, P1, X1) == (E2, P2, X2), "tile body is not steady-state periodic"

    # ---- hardware K loop: largest run of 12-periodic steps ----
    txt = ["\n".join(p) for p in P1]
    best = (NB, 0, 0)  # (emitted steps, r, L)
    if kloop:
        for r in range(1, 13):
            L_ = 1
            while r + 12 * (L_ + 1) <= NB and all(
                txt[b] == txt[b - 12] for b in range(r + 12 * L_, r + 12 * L_ + 12)
            ):
                L_ += 1
            if L_ >= 2 and NB - 12 * (L_ - 1) < best[0]:
                best = (NB - 12 * (L_ - 1), r, L_)
    _, r, Lk = best
    body = []
    if Lk >= 2:
        body += sum(P1[:r], [])
        body += [f"s_mov_b32 ${okc}, {Lk}", "2:"] + sum(P1[r : r + 12], [])
        body += [f"s_sub_u32 ${okc}, ${okc}, 1", f"s_cmp_lg_u32 ${okc}, 0", "s_cbranch_scc1 2b"]
        body += sum(P1[r + 12 * Lk :], [])
    else:
        body += sum(P1, [])
    # Exit: every g2s (LDS DMA) must land before the workgroup's LDS is released; the C stores,
    # issued after all of them, may still be in flight.
    w_exit = vm_wait({t_ for t_ in sim["vm"] if t_[1] in ("g2s", "sc")})
    if single:
        A = PRO + E1 + body + [f"s_waitcnt vmcnt({w_exit}) lgkmcnt(0)"]
    else:
        A = PRO + ["1:"] + E1 + body + X1
        A += [f"s_sub_u32 ${otc}, ${otc}, 1", f"s_cmp_lg_u32 ${otc}, 0", "s_cbranch_scc1 1b"]
        A += [dstore(k, osC) for k in range(ND)]  # the last tile's deferred units
        A += [f"s_waitcnt vmcnt({min(63, w_exit + ND)}) lgkmcnt(0)"]
    if guard:
        G_ = []
        for ig, val in zip(i_guard, guard):
            G_ += [f"s_cmp_eq_u32 ${ig}, {val}", "s_cbranch_scc0 9f"]
        A = G_ + A + ["s_branch 8f", "9:", "s_trap 2", "8:"]
    asm = "\n".join(A)

    cons = (
        ["=&s"] * (n_out - (TPW + 1))
        + ["=s"] * (TPW + 1)
        + ["v"] * 4
        + ["s"] * 4
        + ["v"] * (_NS_A0 + _NS_A1 + _NS_B0 + _NS_B1)
        + ["s"] * 4
        + ["s", "s", "v", "s", "s", "s"]
        + ["v"] * _NTA
        + [str(o) for o in o_slot]
        + ["v"] * 4
        + (["s", "v"] if bias else [])
        + (["s"] * 4 if guard else [])
        + [f"~{{v{r_}}}" for r_ in range(_VSC, _VPF + 1)]
        + [f"~{{a{r_}}}" for r_ in range(256)]
        + ["~{scc}"]
    )
    st = "!llvm.struct<(" + ", ".join(["i32"] * n_out) + ")>"
    info = dict(kloop_r=r, kloop_n=Lk, steps_emitted=len(P1) - 12 * (Lk - 1) if Lk >= 2 else NB, lines=len(A))
    _ASM_CACHE[key] = (asm, ",".join(cons), st, info)
    return _ASM_CACHE[key]


def _build_mxfp6_persistent_kernel(
    *,
    K,
    mn,
    tpw,
    group_m=4,
    num_xcds=8,
    group_n=0,
    mblk=(8, 4),
    kloop=True,
    tune=(),
    nst=_NSTAGE,
    l2pf=0,
    bias=False,
    layout="kblk",
    abi="flydsl",
):
    assert K % 512 == 0
    NB = K // 128
    K128 = K // 128
    K2_0, K2_1 = K // 2, K // 4
    M, N = mn
    assert M % _BM == 0 and N % _BN == 0
    n_pids = (M // _BM, N // _BN)
    tiles = n_pids[0] * n_pids[1]
    assert tiles % tpw == 0
    grid = tiles // tpw
    ldc_b = N * 2
    asm, cons, st, _info = mxfp6_persistent_asm(
        NB,
        K2_0,
        K2_1,
        ldc_b,
        tpw,
        mblk,
        kloop,
        tune,
        nst,
        l2pf,
        bias,
        layout,
        guard=(M, N, K, N) if abi == "aiter" else None,
    )
    abi_aiter = abi == "aiter"
    assert not abi_aiter or (layout == "aiter" and tpw == 1), (
        "abi='aiter' is the layout='aiter', tpw=1 kernel"
    )

    def _rsrc(x, nbytes):  # SRD from a tensor (FlyDSL ABI) or a raw i64 address (AITER ABI)
        if abi_aiter:
            return buffer_ops.create_buffer_resource_from_addr(x, num_records_bytes=nbytes)
        return buffer_ops.create_buffer_resource(x, max_size=False, num_records_bytes=nbytes)

    aiter_l = layout == "aiter"
    NKP = NB + 2  # AITER's K128 steps per tile row (two guard steps)

    _anns = {
        "A0": fx.Array[fx.Float8E4M3FN, nst * _A0_STAGE, 16],
        "A1": fx.Array[fx.Float8E4M3FN, nst * _A1_STAGE, 16],
        "B0": fx.Array[fx.Float8E4M3FN, nst * _B0_STAGE, 16],
        "B1": fx.Array[fx.Float8E4M3FN, nst * _B1_STAGE, 16],
    }
    SharedStorageFp6P = fx.struct(type("SharedStorageFp6P_4w", (), {"__annotations__": _anns}))

    def _persist_body(A0, A1, B0, B1, C, A_scale, B_scale, c_m, c_n, Bias, kx=None):
        lds = fx.SharedAllocator().allocate(SharedStorageFp6P).peek()
        lane_id = umod(fx.thread_idx.x, 64)
        wave_id = udiv(fx.thread_idx.x, 64)
        wave_m = udiv(wave_id, 2)
        wave_n = umod(wave_id, 2)
        lane16 = umod(lane_id, 16)
        g = udiv(lane_id, 16)

        def _lds(arr):
            return fx.Int32(fx.ptrtoint(arr.ptr))

        ra = wave_m * fx.Int32(_NTA * 16) + lane16
        rb = wave_n * fx.Int32(_NTB * 16) + lane16
        if const_expr(aiter_l):
            # AITER units are already in MFMA lane order (16-row group: C0 1 KiB at 16 l, C1 512 B at
            # 8 l), and the g2s keeps them as they are: unswizzled, conflict-free linear reads.
            rd = [
                _lds(lds.A0) + wave_m * fx.Int32(8 * 1024) + lane_id * fx.Int32(16),
                _lds(lds.A1) + wave_m * fx.Int32(8 * 512) + lane_id * fx.Int32(8),
                _lds(lds.B0) + wave_n * fx.Int32(4 * 1024) + lane_id * fx.Int32(16),
                _lds(lds.B1) + wave_n * fx.Int32(4 * 512) + lane_id * fx.Int32(8),
            ]
            mb = [
                rocdl.readfirstlane(T.i32, _lds(a) + wave_id * fx.Int32(1024))
                for a in (lds.A0, lds.A1, lds.B0, lds.B1)
            ]
            _v = wave_id * fx.Int32(1024) + lane_id * fx.Int32(16)  # step st adds st * 4096
            vo = [fx.Int32(_v + fx.Int32(s_ * 4096)) for s_ in range(_NS_A0)]
            vo += [fx.Int32(_v + fx.Int32(s_ * 4096)) for s_ in range(_NS_A1)]
            vo += [fx.Int32(_v + fx.Int32(s_ * 4096)) for s_ in range(_NS_B0)]
            vo += [fx.Int32(_v + fx.Int32(s_ * 4096)) for s_ in range(_NS_B1)]
            _ab = n_pids[0] * NKP * 24576
            _bb = n_pids[1] * NKP * 24576
            rs = [_rsrc(A0, _ab), _rsrc(A0, _ab), _rsrc(B0, _bb), _rsrc(B0, _bb)]
            sa_rsrc = _rsrc(A_scale, n_pids[0] * NKP * 1024)
            sb_rsrc = _rsrc(B_scale, n_pids[1] * NKP * 1024)
            scv = lane_id * fx.Int32(8)
            wA = rocdl.readfirstlane(T.i32, wave_m * fx.Int32(512))
            wB = rocdl.readfirstlane(T.i32, wave_n * fx.Int32(4))
        else:
            rd = [
                _lds(lds.A0) + c0_read_off(ra, g),
                _lds(lds.A1) + c1_read_off(ra, g),
                _lds(lds.B0) + c0_read_off(rb, g),
                _lds(lds.B1) + c1_read_off(rb, g),
            ]
            wmaj = "wmaj" in tune
            if const_expr(wmaj):
                mb = [
                    rocdl.readfirstlane(T.i32, _lds(a) + wave_id * fx.Int32(1024 * ns_))
                    for a, ns_ in ((lds.A0, _NS_A0), (lds.A1, _NS_A1), (lds.B0, _NS_B0), (lds.B1, _NS_B1))
                ]
            else:
                mb = [
                    rocdl.readfirstlane(T.i32, _lds(a) + wave_id * fx.Int32(1024))
                    for a in (lds.A0, lds.A1, lds.B0, lds.B1)
                ]

            def _wm0(st, ns_, pitch):  # wave-major C0 source, minus the step's instruction offset
                r16 = udiv(lane_id, 4)
                chunk = umod(umod(lane_id, 4) + 4 - umod(c0_swz(r16), 4), 4)
                return (
                    (wave_id * fx.Int32(ns_) + fx.Int32(st)) * fx.Int32(16 * pitch)
                    + r16 * 64
                    + chunk * 16
                    - st * 1024
                )

            def _wm1(st, ns_, pitch):
                rl = udiv(lane_id, 2)
                half = umod(umod(lane_id, 2) + c1_swz(umod(rl, 16)), 2)
                return (
                    (wave_id * fx.Int32(ns_) + fx.Int32(st)) * fx.Int32(32 * pitch)
                    + rl * 32
                    + half * 16
                    - st * 1024
                )

            if const_expr(wmaj):
                vo = (
                    [fx.Int32(_wm0(s_, _NS_A0, K2_0)) for s_ in range(_NS_A0)]
                    + [fx.Int32(_wm1(s_, _NS_A1, K2_1)) for s_ in range(_NS_A1)]
                    + [fx.Int32(_wm0(s_, _NS_B0, K2_0)) for s_ in range(_NS_B0)]
                    + [fx.Int32(_wm1(s_, _NS_B1, K2_1)) for s_ in range(_NS_B1)]
                )
            else:
                vo = (
                    [fx.Int32(c0_g2s_src(lane_id, wave_id, s_, K2_0, True)) for s_ in range(_NS_A0)]
                    + [fx.Int32(c1_g2s_src(lane_id, wave_id, s_, K2_1, True)) for s_ in range(_NS_A1)]
                    + [fx.Int32(c0_g2s_src(lane_id, wave_id, s_, K2_0, True)) for s_ in range(_NS_B0)]
                    + [fx.Int32(c1_g2s_src(lane_id, wave_id, s_, K2_1, True)) for s_ in range(_NS_B1)]
                )
            rs = [
                buffer_ops.create_buffer_resource(A0, max_size=False, num_records_bytes=M * K2_0),
                buffer_ops.create_buffer_resource(A1, max_size=False, num_records_bytes=M * K2_1),
                buffer_ops.create_buffer_resource(B0, max_size=False, num_records_bytes=N * K2_0),
                buffer_ops.create_buffer_resource(B1, max_size=False, num_records_bytes=N * K2_1),
            ]
            sa = ScaleS2RPacked(A_scale, n_pids[0] * 256, K, 4)
            sb = ScaleS2RPacked(B_scale, n_pids[1] * 256, K, 4)
            scv = lane_id * fx.Int32(16)
            wA = rocdl.readfirstlane(T.i32, wave_m * fx.Int32(K128 * 512))
            wB = rocdl.readfirstlane(T.i32, wave_n * fx.Int32(K128 * 512))
            sa_rsrc, sb_rsrc = sa.rsrc, sb.rsrc
        srC = _rsrc(C, M * N * 2)
        rg = udiv(lane_id, 16)  # TACCW lane column offset: rg0 -> 0, rg1 -> 16, rg2 -> 8, rg3 -> 24
        col0 = wave_n * fx.Int32(_NTB * 16) + umod(rg, 2) * fx.Int32(16) + udiv(rg, 2) * fx.Int32(8)
        vc = [
            (wave_m * fx.Int32(_NTA * 16) + fx.Int32(ii * 16) + lane16) * fx.Int32(ldc_b) + col0 * fx.Int32(2)
            for ii in range(_NTA)
        ]
        slots = []
        for t_ in range_constexpr(tpw):
            if const_expr(abi_aiter):  # the launcher's 2D grid (N/256, M/256), linearised row-major
                pid = fx.block_idx.y * fx.Int32(n_pids[1]) + fx.block_idx.x
            else:
                pid = fx.block_idx.x + fx.Int32(t_ * grid)
            bm, bn = grouped_xcd_pid(
                pid, c_m, c_n, _BM, _BN, group_m=group_m, num_xcds=num_xcds, group_n=group_n, n_pids=n_pids
            )
            slots.append(rocdl.readfirstlane(T.i32, bm + bn * fx.Int32(65536)))
        slots.append(slots[-1])  # the last tile prefetches itself again (never consumed)
        # L2-prefetch lanes: lane l touches line l % 8 of g2s step (l // 8) % n_steps of its wave
        # (the same 1-KiB blocks the g2s reads; lanes past the steps repeat them).
        _ln = umod(lane_id, 8) * fx.Int32(128)
        _q = udiv(lane_id, 8)
        pf = [
            (umod(_q, 4) * fx.Int32(4) + wave_id) * fx.Int32(16 * K2_0) + _ln,
            (umod(_q, 2) * fx.Int32(4) + wave_id) * fx.Int32(32 * K2_1) + _ln,
            udiv(umod(_q, 4), 2) * fx.Int32(128 * K2_0)
            + (umod(_q, 2) * fx.Int32(4) + wave_id) * fx.Int32(16 * K2_0)
            + _ln,
            umod(_q, 2) * fx.Int32(128 * K2_1) + wave_id * fx.Int32(32 * K2_1) + _ln,
        ]
        ins = rd + mb + vo + rs + [sa_rsrc, sb_rsrc, scv, wA, wB, srC] + vc + slots + pf
        if const_expr(bias):
            ins = ins + [
                _rsrc(Bias, kx["bias_bytes"] if abi_aiter else N * 2),
                (wave_n * fx.Int32(_NTB * 16) + udiv(lane_id, 16) * fx.Int32(4)) * fx.Int32(2),
            ]
        if const_expr(abi_aiter):
            ins = ins + [rocdl.readfirstlane(T.i32, kx[k_]) for k_ in ("M", "N", "K", "stride_D0")]
        _llvm.inline_asm(ir.Type.parse(st), [_raw(x) for x in ins], asm, cons, has_side_effects=True)

    @flyc.kernel(known_block_size=[256, 1, 1])
    def kernel_gemm_mxfp6_persist(
        A0: fx.Tensor,
        A1: fx.Tensor,
        B0: fx.Tensor,
        B1: fx.Tensor,
        C: fx.Tensor,
        A_scale: fx.Tensor,
        B_scale: fx.Tensor,
        c_m: fx.Int32,
        c_n: fx.Int32,
    ):
        _persist_body(A0, A1, B0, B1, C, A_scale, B_scale, c_m, c_n, None)

    @flyc.kernel(known_block_size=[256, 1, 1])
    def kernel_gemm_mxfp6_persist_bias(
        A0: fx.Tensor,
        A1: fx.Tensor,
        B0: fx.Tensor,
        B1: fx.Tensor,
        C: fx.Tensor,
        A_scale: fx.Tensor,
        B_scale: fx.Tensor,
        c_m: fx.Int32,
        c_n: fx.Int32,
        Bias: fx.Tensor,
    ):
        _persist_body(A0, A1, B0, B1, C, A_scale, B_scale, c_m, c_n, Bias)

    # AITER ABI: the packed 0x180-byte KernelArgs of aiter's asm_gemm_a6w6.cu, field by field. Every
    # field owns 16 bytes: a pointer is (Int64, Int64 pad), a 4-byte value (Int32, Int32 pad, Int64 pad).

    @flyc.kernel(known_block_size=[256, 1, 1])
    def kernel_gemm_mxfp6_aiterabi(
        ptr_D: fx.Int64,
        _pad_ptr_D: fx.Int64,
        ptr_C: fx.Int64,
        _pad_ptr_C: fx.Int64,
        ptr_A: fx.Int64,
        _pad_ptr_A: fx.Int64,
        ptr_B: fx.Int64,
        _pad_ptr_B: fx.Int64,
        alpha: fx.Int32,
        _pad_alpha: fx.Int32,
        _pad2_alpha: fx.Int64,
        beta: fx.Int32,
        _pad_beta: fx.Int32,
        _pad2_beta: fx.Int64,
        stride_D0: fx.Int32,
        _pad_stride_D0: fx.Int32,
        _pad2_stride_D0: fx.Int64,
        stride_D1: fx.Int32,
        _pad_stride_D1: fx.Int32,
        _pad2_stride_D1: fx.Int64,
        stride_C0: fx.Int32,
        _pad_stride_C0: fx.Int32,
        _pad2_stride_C0: fx.Int64,
        stride_C1: fx.Int32,
        _pad_stride_C1: fx.Int32,
        _pad2_stride_C1: fx.Int64,
        stride_A0: fx.Int32,
        _pad_stride_A0: fx.Int32,
        _pad2_stride_A0: fx.Int64,
        stride_A1: fx.Int32,
        _pad_stride_A1: fx.Int32,
        _pad2_stride_A1: fx.Int64,
        stride_B0: fx.Int32,
        _pad_stride_B0: fx.Int32,
        _pad2_stride_B0: fx.Int64,
        stride_B1: fx.Int32,
        _pad_stride_B1: fx.Int32,
        _pad2_stride_B1: fx.Int64,
        M: fx.Int32,
        _pad_M: fx.Int32,
        _pad2_M: fx.Int64,
        N: fx.Int32,
        _pad_N: fx.Int32,
        _pad2_N: fx.Int64,
        K: fx.Int32,
        _pad_K: fx.Int32,
        _pad2_K: fx.Int64,
        ptr_ScaleA: fx.Int64,
        _pad_ptr_ScaleA: fx.Int64,
        ptr_ScaleB: fx.Int64,
        _pad_ptr_ScaleB: fx.Int64,
        stride_ScaleA0: fx.Int32,
        _pad_stride_ScaleA0: fx.Int32,
        _pad2_stride_ScaleA0: fx.Int64,
        stride_ScaleA1: fx.Int32,
        _pad_stride_ScaleA1: fx.Int32,
        _pad2_stride_ScaleA1: fx.Int64,
        stride_ScaleB0: fx.Int32,
        _pad_stride_ScaleB0: fx.Int32,
        _pad2_stride_ScaleB0: fx.Int64,
        stride_ScaleB1: fx.Int32,
        _pad_stride_ScaleB1: fx.Int32,
        _pad2_stride_ScaleB1: fx.Int64,
        log2_k_split: fx.Int32,
        _pad_log2_k_split: fx.Int32,
        _pad2_log2_k_split: fx.Int64,
    ):
        _persist_body(
            ptr_A,
            ptr_A,
            ptr_B,
            ptr_B,
            ptr_D,
            ptr_ScaleA,
            ptr_ScaleB,
            M,
            N,
            None,
            kx=dict(M=M, N=N, K=K, stride_D0=stride_D0, bias_bytes=stride_C1),
        )

    @flyc.kernel(known_block_size=[256, 1, 1])
    def kernel_gemm_mxfp6_aiterabi_bias(
        ptr_D: fx.Int64,
        _pad_ptr_D: fx.Int64,
        ptr_C: fx.Int64,
        _pad_ptr_C: fx.Int64,
        ptr_A: fx.Int64,
        _pad_ptr_A: fx.Int64,
        ptr_B: fx.Int64,
        _pad_ptr_B: fx.Int64,
        alpha: fx.Int32,
        _pad_alpha: fx.Int32,
        _pad2_alpha: fx.Int64,
        beta: fx.Int32,
        _pad_beta: fx.Int32,
        _pad2_beta: fx.Int64,
        stride_D0: fx.Int32,
        _pad_stride_D0: fx.Int32,
        _pad2_stride_D0: fx.Int64,
        stride_D1: fx.Int32,
        _pad_stride_D1: fx.Int32,
        _pad2_stride_D1: fx.Int64,
        stride_C0: fx.Int32,
        _pad_stride_C0: fx.Int32,
        _pad2_stride_C0: fx.Int64,
        stride_C1: fx.Int32,
        _pad_stride_C1: fx.Int32,
        _pad2_stride_C1: fx.Int64,
        stride_A0: fx.Int32,
        _pad_stride_A0: fx.Int32,
        _pad2_stride_A0: fx.Int64,
        stride_A1: fx.Int32,
        _pad_stride_A1: fx.Int32,
        _pad2_stride_A1: fx.Int64,
        stride_B0: fx.Int32,
        _pad_stride_B0: fx.Int32,
        _pad2_stride_B0: fx.Int64,
        stride_B1: fx.Int32,
        _pad_stride_B1: fx.Int32,
        _pad2_stride_B1: fx.Int64,
        M: fx.Int32,
        _pad_M: fx.Int32,
        _pad2_M: fx.Int64,
        N: fx.Int32,
        _pad_N: fx.Int32,
        _pad2_N: fx.Int64,
        K: fx.Int32,
        _pad_K: fx.Int32,
        _pad2_K: fx.Int64,
        ptr_ScaleA: fx.Int64,
        _pad_ptr_ScaleA: fx.Int64,
        ptr_ScaleB: fx.Int64,
        _pad_ptr_ScaleB: fx.Int64,
        stride_ScaleA0: fx.Int32,
        _pad_stride_ScaleA0: fx.Int32,
        _pad2_stride_ScaleA0: fx.Int64,
        stride_ScaleA1: fx.Int32,
        _pad_stride_ScaleA1: fx.Int32,
        _pad2_stride_ScaleA1: fx.Int64,
        stride_ScaleB0: fx.Int32,
        _pad_stride_ScaleB0: fx.Int32,
        _pad2_stride_ScaleB0: fx.Int64,
        stride_ScaleB1: fx.Int32,
        _pad_stride_ScaleB1: fx.Int32,
        _pad2_stride_ScaleB1: fx.Int64,
        log2_k_split: fx.Int32,
        _pad_log2_k_split: fx.Int32,
        _pad2_log2_k_split: fx.Int64,
    ):
        _persist_body(
            ptr_A,
            ptr_A,
            ptr_B,
            ptr_B,
            ptr_D,
            ptr_ScaleA,
            ptr_ScaleB,
            M,
            N,
            ptr_C,
            kx=dict(M=M, N=N, K=K, stride_D0=stride_D0, bias_bytes=stride_C1),
        )

    _pt = {"passthrough": [["amdgpu-agpr-alloc", "256"]]}
    attrs = {"rocdl.flat_work_group_size": "256,256", "rocdl.waves_per_eu": 1, **_pt}

    if abi_aiter:
        _k = kernel_gemm_mxfp6_aiterabi_bias if bias else kernel_gemm_mxfp6_aiterabi

        @flyc.jit
        def launch_mxfp6_aiterabi(
            ptr_D: fx.Int64,
            _pad_ptr_D: fx.Int64,
            ptr_C: fx.Int64,
            _pad_ptr_C: fx.Int64,
            ptr_A: fx.Int64,
            _pad_ptr_A: fx.Int64,
            ptr_B: fx.Int64,
            _pad_ptr_B: fx.Int64,
            alpha: fx.Int32,
            _pad_alpha: fx.Int32,
            _pad2_alpha: fx.Int64,
            beta: fx.Int32,
            _pad_beta: fx.Int32,
            _pad2_beta: fx.Int64,
            stride_D0: fx.Int32,
            _pad_stride_D0: fx.Int32,
            _pad2_stride_D0: fx.Int64,
            stride_D1: fx.Int32,
            _pad_stride_D1: fx.Int32,
            _pad2_stride_D1: fx.Int64,
            stride_C0: fx.Int32,
            _pad_stride_C0: fx.Int32,
            _pad2_stride_C0: fx.Int64,
            stride_C1: fx.Int32,
            _pad_stride_C1: fx.Int32,
            _pad2_stride_C1: fx.Int64,
            stride_A0: fx.Int32,
            _pad_stride_A0: fx.Int32,
            _pad2_stride_A0: fx.Int64,
            stride_A1: fx.Int32,
            _pad_stride_A1: fx.Int32,
            _pad2_stride_A1: fx.Int64,
            stride_B0: fx.Int32,
            _pad_stride_B0: fx.Int32,
            _pad2_stride_B0: fx.Int64,
            stride_B1: fx.Int32,
            _pad_stride_B1: fx.Int32,
            _pad2_stride_B1: fx.Int64,
            M: fx.Int32,
            _pad_M: fx.Int32,
            _pad2_M: fx.Int64,
            N: fx.Int32,
            _pad_N: fx.Int32,
            _pad2_N: fx.Int64,
            K: fx.Int32,
            _pad_K: fx.Int32,
            _pad2_K: fx.Int64,
            ptr_ScaleA: fx.Int64,
            _pad_ptr_ScaleA: fx.Int64,
            ptr_ScaleB: fx.Int64,
            _pad_ptr_ScaleB: fx.Int64,
            stride_ScaleA0: fx.Int32,
            _pad_stride_ScaleA0: fx.Int32,
            _pad2_stride_ScaleA0: fx.Int64,
            stride_ScaleA1: fx.Int32,
            _pad_stride_ScaleA1: fx.Int32,
            _pad2_stride_ScaleA1: fx.Int64,
            stride_ScaleB0: fx.Int32,
            _pad_stride_ScaleB0: fx.Int32,
            _pad2_stride_ScaleB0: fx.Int64,
            stride_ScaleB1: fx.Int32,
            _pad_stride_ScaleB1: fx.Int32,
            _pad2_stride_ScaleB1: fx.Int64,
            log2_k_split: fx.Int32,
            _pad_log2_k_split: fx.Int32,
            _pad2_log2_k_split: fx.Int64,
            stream: fx.Stream,
        ):
            _k(
                ptr_D,
                _pad_ptr_D,
                ptr_C,
                _pad_ptr_C,
                ptr_A,
                _pad_ptr_A,
                ptr_B,
                _pad_ptr_B,
                alpha,
                _pad_alpha,
                _pad2_alpha,
                beta,
                _pad_beta,
                _pad2_beta,
                stride_D0,
                _pad_stride_D0,
                _pad2_stride_D0,
                stride_D1,
                _pad_stride_D1,
                _pad2_stride_D1,
                stride_C0,
                _pad_stride_C0,
                _pad2_stride_C0,
                stride_C1,
                _pad_stride_C1,
                _pad2_stride_C1,
                stride_A0,
                _pad_stride_A0,
                _pad2_stride_A0,
                stride_A1,
                _pad_stride_A1,
                _pad2_stride_A1,
                stride_B0,
                _pad_stride_B0,
                _pad2_stride_B0,
                stride_B1,
                _pad_stride_B1,
                _pad2_stride_B1,
                M,
                _pad_M,
                _pad2_M,
                N,
                _pad_N,
                _pad2_N,
                K,
                _pad_K,
                _pad2_K,
                ptr_ScaleA,
                _pad_ptr_ScaleA,
                ptr_ScaleB,
                _pad_ptr_ScaleB,
                stride_ScaleA0,
                _pad_stride_ScaleA0,
                _pad2_stride_ScaleA0,
                stride_ScaleA1,
                _pad_stride_ScaleA1,
                _pad2_stride_ScaleA1,
                stride_ScaleB0,
                _pad_stride_ScaleB0,
                _pad2_stride_ScaleB0,
                stride_ScaleB1,
                _pad_stride_ScaleB1,
                _pad2_stride_ScaleB1,
                log2_k_split,
                _pad_log2_k_split,
                _pad2_log2_k_split,
                value_attrs=attrs,
            ).launch(grid=(n_pids[1], n_pids[0], 1), block=(256, 1, 1), stream=stream)

        return launch_mxfp6_aiterabi

    @flyc.jit
    def launch_mxfp6_persist(
        A0: fx.Tensor,
        A1: fx.Tensor,
        B0: fx.Tensor,
        B1: fx.Tensor,
        C: fx.Tensor,
        A_scale: fx.Tensor,
        B_scale: fx.Tensor,
        c_m: fx.Int32,
        c_n: fx.Int32,
        stream: fx.Stream,
    ):
        kernel_gemm_mxfp6_persist(A0, A1, B0, B1, C, A_scale, B_scale, c_m, c_n, value_attrs=attrs).launch(
            grid=(grid, 1, 1), block=(256, 1, 1), stream=stream
        )

    if bias:

        @flyc.jit
        def launch_mxfp6_persist_bias(
            A0: fx.Tensor,
            A1: fx.Tensor,
            B0: fx.Tensor,
            B1: fx.Tensor,
            C: fx.Tensor,
            A_scale: fx.Tensor,
            B_scale: fx.Tensor,
            c_m: fx.Int32,
            c_n: fx.Int32,
            Bias: fx.Tensor,
            stream: fx.Stream,
        ):
            kernel_gemm_mxfp6_persist_bias(
                A0, A1, B0, B1, C, A_scale, B_scale, c_m, c_n, Bias, value_attrs=attrs
            ).launch(grid=(grid, 1, 1), block=(256, 1, 1), stream=stream)

        return launch_mxfp6_persist_bias
    return launch_mxfp6_persist


def mxfp6_pick_tpw(M, N, K, ncu=None, tpw_max=16, kwork_max=49152):
    """Tiles per workgroup. Only a grid that is a whole number of CU rounds is worth walking
    persistently (a ragged one leaves the last round's CUs idle while others still walk); the
    tile count per WG is then the largest divisor of the rounds with tpw*K <= kwork_max -- a
    long-K walk keeps every CU in lockstep for its whole length, and two shorter rounds beat
    one long one there (3072-K shapes are best at one round). Otherwise 1: one tile per WG, no cross-tile prefetch."""
    tiles = (M // _BM) * (N // _BN)
    ncu = ncu or torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
    if tiles % ncu:
        return 1
    rounds = tiles // ncu
    return max(
        d for d in range(1, min(rounds, tpw_max) + 1) if rounds % d == 0 and (d == 1 or d * K <= kwork_max)
    )


def gemm_mxfp6_persistent(
    a_c0,
    a_c1,
    b_c0,
    b_c1,
    a_sp,
    b_sp,
    *,
    out=None,
    tpw=None,
    mblk=(8, 4),
    swizzle=None,
    kloop=True,
    tune=(),
    nst=_NSTAGE,
    l2pf=0,
    bias=None,
    layout="kblk",
    m=None,
    n=None,
    k=None,
):
    """Persistent kernel: kblk planes (see ``kblk_planes``), TPW tiles per workgroup in one asm,
    cross-tile prefetch, folded C store, hardware K loop. Bit-identical to the single-tile kernel.
    ``tune``, ``nst``, ``l2pf``: see ``mxfp6_persistent_asm``.

    ``bias``: optional bf16 [N], added in the folded C store as fp32(acc) + fp32(bias) with one RNE
    to bf16 -- AITER A6W6's bias epilogue (``gemm_a6w6_asm(..., 1.0, bias)``), bit for bit. None
    builds the unchanged no-bias kernel.

    ``layout="aiter"``: the operands are AITER's own A6W6 blobs (``mxfp6_c0c1_256_padk2``, what
    ``aiter.quant_mxfp6_gemm`` / Turbo's fmt-0 packer write): pass ``a_c0``/``b_c0`` = operand blobs,
    ``a_sp``/``b_sp`` = scale blobs, ``a_c1``/``b_c1`` = None, and the logical ``m``, ``n``, ``k``."""
    if layout == "aiter":
        assert m is not None and n is not None and k is not None
        M, N, K = m, n, k
        a_c1, b_c1 = a_c0, b_c0
    else:
        M, N = a_c0.shape[0], b_c0.shape[0]
        K = a_c0.shape[1] * 2
    gm, gn, xcd = swizzle if swizzle is not None else _mxfp4_nt_config(M, N, K)
    tpw = tpw or mxfp6_pick_tpw(M, N, K)
    has_bias = bias is not None
    if has_bias:
        assert bias.dtype == torch.bfloat16 and bias.numel() == N and bias.is_contiguous()
    key = (
        ("persist", M, N, K, gm, gn, xcd, tuple(mblk), tpw, kloop, tuple(sorted(tune)), nst, l2pf)
        + (("bias",) if has_bias else ())
        + ((layout,) if layout != "kblk" else ())
    )
    launch = _LAUNCH_CACHE.get(key)
    if launch is None:
        launch = _build_mxfp6_persistent_kernel(
            K=K,
            mn=(M, N),
            tpw=tpw,
            group_m=gm,
            num_xcds=xcd,
            group_n=gn,
            mblk=mblk,
            kloop=kloop,
            tune=tune,
            nst=nst,
            l2pf=l2pf,
            bias=has_bias,
            layout=layout,
        )
        _LAUNCH_CACHE[key] = launch
    if out is None:
        out = torch.empty((M, N), dtype=torch.bfloat16, device=a_c0.device)
    args = (
        (
            a_c0.view(torch.int8),
            a_c1.view(torch.int8),
            b_c0.view(torch.int8),
            b_c1.view(torch.int8),
            out,
            a_sp.view(torch.int32).reshape(-1),
            b_sp.view(torch.int32).reshape(-1),
            M,
            N,
        )
        + ((bias.view(-1),) if has_bias else ())
        + (torch.cuda.current_stream(),)
    )
    comp = _COMPILED.get(key)
    if comp is None:
        comp = compile_with_scratch_out(launch, args, out_index=4)
        _COMPILED[key] = comp
    comp(*args)
    return out


def aiter_kernel_args(A, B, As, Bs, out, K, bias=None, alpha=1.0):
    """The field values aiter's gemm_a6w6_asm (csrc/py_itfs_cu/asm_gemm_a6w6.cu) puts in its packed
    0x180-byte KernelArgs, in field order: (name, value). K is the padded K the launcher passes."""
    import struct

    M, N = out.shape
    return [
        ("ptr_D", out.data_ptr()),
        ("ptr_C", bias.data_ptr() if bias is not None else 0),
        ("ptr_A", A.data_ptr()),
        ("ptr_B", B.data_ptr()),
        ("alpha", struct.unpack("<i", struct.pack("<f", alpha))[0]),
        ("beta", 0),
        ("stride_D0", out.stride(0)),
        ("stride_D1", 0),
        ("stride_C0", 0),
        ("stride_C1", bias.numel() * 2 if bias is not None else 0),
        ("stride_A0", 0),
        ("stride_A1", 0),
        ("stride_B0", 0),
        ("stride_B1", 0),
        ("M", M),
        ("N", N),
        ("K", K),
        ("ptr_ScaleA", As.data_ptr()),
        ("ptr_ScaleB", Bs.data_ptr()),
        ("stride_ScaleA0", K // 32),
        ("stride_ScaleA1", 0),
        ("stride_ScaleB0", K // 32),
        ("stride_ScaleB1", 0),
        ("log2_k_split", 0),
    ]


def gemm_mxfp6_aiterabi(A, B, As, Bs, out, K, *, bias=None, mn=None, kmn=None, swizzle=None, mblk=(8, 4)):
    """The layout="aiter", tpw=1 kernel compiled with AITER's gemm_a6w6_asm ABI (packed 0x180-byte
    KernelArgs, 2D grid N/256 x M/256, bias via ptr_C, shape guard), launched here through FlyDSL with
    exactly the argument values the launcher would pass. ``mn`` / ``kmn`` = the (M, N) / K the code
    object is compiled for (default: the call's), so a mismatched launch can be provoked to test the guard.
    Kernel symbols: kernel_gemm_mxfp6_aiterabi_0 (no bias), kernel_gemm_mxfp6_aiterabi_bias_0 (bias)."""
    M, N = out.shape
    cM, cN = mn if mn is not None else (M, N)
    cK = kmn if kmn is not None else K
    gm, gn, xcd = swizzle if swizzle is not None else _mxfp4_nt_config(cM, cN, cK)
    has_bias = bias is not None
    key = ("aiterabi", cM, cN, cK, gm, gn, xcd, tuple(mblk), has_bias)
    launch = _LAUNCH_CACHE.get(key)
    if launch is None:
        launch = _build_mxfp6_persistent_kernel(
            K=cK,
            mn=(cM, cN),
            tpw=1,
            group_m=gm,
            num_xcds=xcd,
            group_n=gn,
            mblk=mblk,
            bias=has_bias,
            layout="aiter",
            abi="aiter",
        )
        _LAUNCH_CACHE[key] = launch
    vals = []
    for _nm, v in aiter_kernel_args(A, B, As, Bs, out, K, bias):
        vals += [v, 0] if _nm.startswith("ptr_") else [v, 0, 0]
    args = tuple(vals) + (torch.cuda.current_stream(),)
    comp = _COMPILED.get(key)
    if comp is None:
        comp = flyc.compile(launch, *args)
        _COMPILED[key] = comp
    comp(*args)
    return out
