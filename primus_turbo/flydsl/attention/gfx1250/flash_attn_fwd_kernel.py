###############################################################################
# SPDX-License-Identifier: Apache-2.0
#
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 FlyDSL Project Contributors
#
# Adapted from FlyDSL (https://github.com/ROCm/FlyDSL)
# Modified by the Primus-Turbo team.
#
# This file is distributed under the Apache License 2.0 (see LICENSE-APACHE),
# not the MIT license that covers the rest of Primus-Turbo (see LICENSE).
###############################################################################

# Adapted from aiter (https://github.com/ROCm/aiter, commit 6963ae9d), where it is
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc., under the MIT license.

"""Flash-attention forward kernel for gfx1250 (MI455X): bf16, head_dim 128, BSHD, GQA.

Vendored from aiter's gfx1250 FlyDSL prefill forward (github.com/ROCm/aiter, commit
6963ae9d) and tuned for Primus-Turbo (Llama-3.1-8B GQA training shapes). Written in the
FlyDSL layout-algebra style (TDM copy atoms + ``SharedAllocator``).

A workgroup computes ``BLOCK_M`` packed query rows of one kv head: the ``G = Hq / Hkv``
query heads that share the kv head are packed into the rows (head fastest), so every K/V
tile staged in LDS serves all of them. Each wave owns 32 packed rows (two 16-row WMMA
tiles) and streams the kv axis in 64-row tiles through a double-buffered LDS ring:
S^T = K Q^T, online softmax, O^T += V^T P^T. Masking is bottom-right causal, or none.

Two tilings share this code (``Tiling``); the interface picks one per call:
  - ``DEFAULT``: 8 waves (256 threads), BLOCK_M = 256, all 320 KB of LDS. Waves 0..3 and
    4..7 order their main-loop preamble differently (``WarpType``), so one wave of each
    SIMD pair drives memory while the other computes.
  - ``SMALL_GRID``: for grids with fewer 256-row workgroups than CUs, where the run time
    is that of the heaviest workgroup. 4 waves (128 threads) as 2 row waves x 2 d-halves:
    waves w and w + 2 both run QK^T and the softmax of the same 32 rows, and each
    accumulates and stores one half of the head dim, so BLOCK_M = 64. 160 KB of LDS (no
    floor on the K/V blocks) lets two workgroups share a CU; each d-half has its own Q and
    O staging copy.

The tilings share all source and module constants, so the ``Tiling`` tuple the kernel and
launcher closures capture is what tells their compiled kernels apart in flydsl's JIT cache
key (its closure part covers scalars and tuples, not objects). Everything a tiling changes
is passed down explicitly from that tuple.

Entry point: ``flash_attn_fwd``.
"""

import functools
from enum import IntEnum
from typing import NamedTuple

from .flydsl_version import require_flydsl

require_flydsl()

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch

# UNSTABLE(gfx1250): rocdl.{wave_id,mbcnt_lo,ballot,permlanex16,exp2,wmma_*,s_wait_*,sched_barrier}
# are ODS builders outside rocdl.__all__; no stable wrapper in 0.3.4.1.
from flydsl.expr import arith, gpu, rocdl
from flydsl.expr import math as fmath

# UNSTABLE(gfx1250): TDM (cdna5) tensor_wait; no stable export in 0.3.4.1.
from flydsl.expr.rocdl import tdm_ops
from flydsl.expr.typing import T

from . import buffer_ops
from .flash_attn_fwd_utils import (
    KManager16bV2,
    OManager16bV3,
    QManager16bV2,
    VManager16bV2,
)
from .flash_attn_utils import LOG2E, WAVE_SIZE, _run_compiled

# ============================================================================
# Constants
# ============================================================================

WMMA_M = 16  # query rows per WMMA tile
WMMA_N = 16  # kv rows per WMMA tile (the S^T = K Q^T output's kv axis)
WMMA_K = 32  # WMMA contraction depth (bf16 v_wmma_f32_16x16x32); d-tile width
WMMA_ROW_PER_WAVE = 2  # 16-row q tiles per row wave: wave i owns rows [32 i, 32 i + 32)
HEAD_DIM = 128  # qk and v head dim
N_BLOCK = 64  # kv rows per tile (columns of one QK^T tile)
N_KV_PP = 2  # K/V LDS ping-pong slots the main loop rotates through
LDS_BYTES = 320 * 1024  # LDS per workgroup on gfx1250, the only target of these kernels
ELEM_DTYPE = fx.BFloat16  # Q/K/V/P/O fragments

# gfx1250 expert scheduling mode 2 (DEP_MODE=2), on by default. The launcher passes the
# ``amdgpu-expert-scheduling-mode`` LLVM hint, so the kernel runs with the hardware VA_VDST /
# VM_VSRC dependency interlocks off and LLVM inserts the dependency waits itself. That is
# why every memory op of the forward is a plain intrinsic with SSA-visible operands, never
# opaque inline asm: LLVM must see each dependency it covers, such as the LDS read after
# a TDM write. The upstream kernel's notes report a rare failure under mode 2 (seq 8192,
# non-causal); test_forward_repeats_bitwise_at_seq_8192 in
# tests/pytorch/ops/test_attention_flydsl_gfx1250.py repeats that shape. False builds the
# kernels in mode 0.
ENABLE_SCHED_MODE2 = True

# Deferred O rescale (as in FlashAttention-4). Rescaling the running O accumulator by
# corr = exp(m_prev - m_new) is a full-width VALU pass every tile, but corr == 1 when the
# running max does not move. So m stays STALE while the tile's row max is within
# RESCALE_THRESHOLD logits of it: P = exp2(S - m_stale) accumulates against the
# un-rescaled O and denominator, which stays consistent. The per-lane test is made
# wave-uniform with a ballot (any lane over the threshold makes the whole wave rescale),
# so the wide multiply sits behind one non-divergent branch. exp2((s - m) * LOG2E) is
# e^(s - m), so 8.0 defers until the max would move by e^8 ~ 2981x, far below the e^88
# fp32 overflow.
RESCALE_THRESHOLD = 8.0

# Running-max seed: a finite big negative (not -inf), so a fully masked row keeps m finite
# and softmax's (m_prev - m_new) and fma(s, .., -m) never compute -inf - -inf (NaN).
# exp2(BIG_NEG - real) still underflows to 0. Masked scores stay -inf (p = 0).
BIG_NEG = -1.0e30


class WarpType(IntEnum):
    """Main-loop preamble ordering of a wave (compile-time). On gfx1250 wave i shares a SIMD
    with wave i + 4. LO_WARP waits for its K/V tile, reads K into registers, then issues the
    next tile's prefetch; HI_WARP issues the prefetch first. With both orders in one
    workgroup, one wave of each SIMD pair drives memory while its mate computes."""

    LO_WARP = 0
    HI_WARP = 1


class Tiling(NamedTuple):
    """Workgroup shape of one forward variant (compile-time)."""

    kernel_name: str  # device symbol, as profilers show it
    num_waves: int  # waves per workgroup
    num_row_waves: int  # waves with distinct q rows; d_split waves share each set and split the head dim
    lds_bytes: int  # LDS allocated per workgroup
    min_kv_blk_bytes: int  # floor of each K and each V LDS block
    # True: waves [0, num_waves / 2) run LO_WARP and the rest HI_WARP. False: every wave runs
    # LO_WARP; the kernel still contains the HI_WARP copy, behind a test that is always true,
    # which keeps the instructions of the tuned kernel.
    warp_specialized: bool

    @property
    def d_split(self):
        """Waves sharing one set of q rows; each accumulates HEAD_DIM // d_split of O."""
        return self.num_waves // self.num_row_waves

    @property
    def block_size(self):
        """Threads per workgroup."""
        return WAVE_SIZE * self.num_waves

    @property
    def block_m(self):
        """Packed q rows per workgroup."""
        return WMMA_M * WMMA_ROW_PER_WAVE * self.num_row_waves


DEFAULT = Tiling(
    kernel_name="k_flash_attn_fwd",
    num_waves=8,
    num_row_waves=8,
    lds_bytes=LDS_BYTES,
    min_kv_blk_bytes=64 * 1024,
    warp_specialized=True,
)
SMALL_GRID = Tiling(
    kernel_name="k_flash_attn_fwd_small_grid",
    num_waves=4,
    num_row_waves=2,
    lds_bytes=160 * 1024,
    min_kv_blk_bytes=0,
    warp_specialized=False,
)

# ============================================================================
# Small device helpers
# ============================================================================


def _warp_id():
    """Wave index within the workgroup."""
    return fx.Int32(rocdl.wave_id())


def _lane_id():
    """Lane index within the wave (wave32)."""
    return fx.Int32(rocdl.mbcnt_lo(T.i32, fx.Int32(-1).ir_value(), fx.Int32(0).ir_value()))


def _lpt_block_id(axis):
    """Longest-first (LPT) remap of the hardware block id.

    Hardware dispatches the linear id ``bx + gx*(by + gy*bz)`` in order, x fastest, and
    under causal a WG's KV-tile count rises with block x -- so the heaviest tiles went
    last in every (y, z) group. This bijection hands the first ``gy*gz`` dispatched WGs
    the highest x of every (y, z), then the next-highest, and so on. Every consumer of a
    block id goes through here, so all of them see the same remapped triple.
    """
    gx = fx.grid_dim.x
    gy = fx.grid_dim.y
    gz = fx.grid_dim.z
    lin = gpu.block_idx.x + gx * (gpu.block_idx.y + gy * gpu.block_idx.z)
    gyz = gy * gz
    rank = lin // gyz
    rem = lin - rank * gyz
    if axis == "x":
        return gx - fx.Int32(1) - rank
    if axis == "y":
        return rem % gy
    return rem // gy


def _packed_tile_indices(gqa_ratio, block_m, warp_idx, lane_idx):
    """Map this lane's rows in the packed ``(seq, q_head_in_group)`` tile to global
    indices; returns ``(kv_head, q_head_idx, seq_idx)`` where ``kv_head`` is a
    scalar ``fx.Int32`` (shared) and ``q_head_idx`` / ``seq_idx`` are length-R
    lists (one per q-WMMA-tile owned by this wave; R = WMMA_ROW_PER_WAVE).

    GQA head x seq packing:
      block_id x -> tile over one kv-head's ``(seq, q_head_in_group)`` plane
      block_id y -> kv_head
    ``q_head_in_group`` is the fast axis, so the ``% / //`` use the small (often
    power-of-two) ``gqa_ratio``. Each of the ``block_m`` rows is an independent
    query sharing this kv-head's K/V. The R tiles a (row) wave owns are contiguous:
    ``warp_row0 = block_x*block_m + warp_idx*(R*WMMA_M)`` and tile ``qt`` starts
    at ``warp_row0 + qt*WMMA_M``.
    """
    kv_head = _lpt_block_id("y")
    warp_row0 = _lpt_block_id("x") * block_m + warp_idx * (WMMA_ROW_PER_WAVE * WMMA_M)
    q_head_idx = []
    seq_idx = []
    for qt in range(WMMA_ROW_PER_WAVE):
        row_idx = warp_row0 + qt * WMMA_M + lane_idx % WMMA_M
        q_head_idx.append(kv_head * gqa_ratio + row_idx % gqa_ratio)
        seq_idx.append(row_idx // gqa_ratio)
    return kv_head, q_head_idx, seq_idx


# ============================================================================
# Compute stages: QK GEMM, online softmax, PV GEMM
# ============================================================================


def _wmma(a, b, c):
    """v_wmma_f32_16x16x32_bf16 (gfx1250, wave32): C[16x16 f32] = A[16x32] @ B[32x16] + C.
    Preserve the SSA-returning intrinsic and disable operand reuse.

    a/b: v16 bf16 fragments; c: v8 f32 accumulator; returns the v8 f32 result
    (raw MLIR value, feed straight back as ``c`` to accumulate)."""
    v8f32 = fx.Vector.make_type(8, fx.Float32)
    # modC defaults to WMMACModifier::none; omit it.
    # UNSTABLE(gfx1250): raw WMMA intrinsic (SSA-returning, reuse disabled); fx.rocdl.WMMA +
    # fx.gemm would change the fragment/ISA contract.
    return rocdl.wmma_f32_16x16x32_bf16(
        v8f32,
        fx.as_ir_value(a),
        fx.as_ir_value(b),
        fx.as_ir_value(c),
        reuseA=False,
        reuseB=False,
    ).result


def _qk_gemm(*, k_values, q_frags_list):
    """GEMM1: S^T = K @ Q^T for one resident KV tile, for all R q-WMMA-tiles this
    wave owns. K is **shared** across the q-tiles (loaded once), so each K fragment
    is shuffled once and fed into R independent WMMA chains.

    WMMA convention (gfx1250): S^T[kv,q] = K @ Q^T with **K = A-operand** (src_a)
    and **Q = B-operand** (src_b). Contract d in ``NDT = HEAD_DIM//WMMA_K`` tiles;
    produce ``NKV = N_BLOCK//WMMA_N`` kv-tiles. Accumulator layout: lane ``l``
    element ``si`` holds S^T[kv = kv_tile*WMMA_N + (l//16)*8 + si, q = l%16] (kv on
    the C-row / M axis, q on the C-col / N axis).

    ``q_frags_list`` is a length-R list; entry ``qt`` is that q-tile's NDT v16-bf16
    Q fragments. Returns ``s_acc_list``: a length-R list, each a list of NKV
    v8-f32 accumulators (== P^T for that q-tile).

    ``k_values`` is the already-burst-loaded flat ``(kv, dt, half)`` list of
    v8-bf16 ds_load results (from ``k_mgr.load_k_to_reg``, kept OUT of the WMMA stream so
    there is no wmma->ds_load issue bubble). Each K fragment is the two 16-col halves of a
    d-tile shuffled into a v16 fragment matching the Q frag layout.

    The ``s_wait_dscnt`` that drains the K ds_load burst is not here: under warp
    specialization the LO and HI warps drain at different points of the main-loop preamble.
    """
    R = len(q_frags_list)
    NKV = N_BLOCK // WMMA_N  # output kv tiles (WMMA_N kv rows each)
    NDT = len(q_frags_list[0])  # contraction d-tiles (== HEAD_DIM // WMMA_K)

    # Consume in (kv, dt, half) order: a (half=0, half=1) pair shuffles into a v16
    # K fragment (shared by all q-tiles); NDT d-tiles accumulate into one kv-tile's
    # s_acc, independently per q-tile.
    s_acc_list = [[None] * NKV for _ in range(R)]
    j = 0
    for kv in range(NKV):
        for dt in range(NDT):
            lo = k_values[j]
            hi = k_values[j + 1]
            j += 2
            k_frag = lo.shuffle(hi, list(range(16)))
            for qt in range(R):
                acc = s_acc_list[qt][kv] if dt > 0 else fx.Vector.filled(8, 0.0, fx.Float32)
                s_acc_list[qt][kv] = _wmma(k_frag, q_frags_list[qt][dt], acc)
    return s_acc_list


def _tree_reduce_multi(lists, op3, op2):
    """Balanced 3-way tree reduction of R independent lists in lockstep, returning one result
    per list. Per list the critical path is ~ceil(log3(N)) vs N-1 for a left-fold, and op3 =
    nested op2 so the backend fuses it (v_max3_f32 for max). Each layer's combines are emitted
    POSITION-MAJOR across the lists (list0[pos], list1[pos], ...) so the R independent ops sit
    adjacent in the IR -> the backend can dual-issue them and hide one row's cross-lane /
    latency bubble behind the other's work."""
    curs = [list(v) for v in lists]
    while max(len(c) for c in curs) > 1:
        nxts = [[] for _ in curs]
        idxs = [0] * len(curs)
        while any(idxs[k] < len(curs[k]) for k in range(len(curs))):
            for k in range(len(curs)):
                cur, i, n = curs[k], idxs[k], len(curs[k])
                if i >= n:
                    continue
                if n - i >= 3:
                    nxts[k].append(op3(cur[i], cur[i + 1], cur[i + 2]))
                    idxs[k] += 3
                elif n - i == 2:
                    nxts[k].append(op2(cur[i], cur[i + 1]))
                    idxs[k] += 2
                else:
                    nxts[k].append(cur[i])
                    idxs[k] += 1
        curs = nxts
    return [c[0] for c in curs]


def _softmax(*, s_list, m_prev_list, d_prev_list, lane_idx, kv_pos_base, q_max_list, kv_len):
    """Online-softmax update for one KV tile, for ALL R q-WMMA-tiles this wave owns.

    The R rows are independent (each owns its S, running m/d, and mask bound) but share
    the tile's K/V. Processing them together lets the rows' balanced max-tree and
    sum-tree reductions emit INTERLEAVED (position-major across rows, via
    ``_tree_reduce_multi``) so the backend can dual-issue row0/row1 combines and hide each
    other's cross-lane permlanex16 latency. ``s_list[r]`` already includes softmax_scale
    (folded into Q), so exp uses plain LOG2E.

    Layout (from ``_qk_gemm``): ``s_list[r]`` is a list of ``NKV = N_BLOCK//WMMA_N`` v8-f32
    accumulators; this lane owns query ``q = warp*16 + l%16`` and, in tile ``kvt``, the kv
    rows ``kvt*16 + (l//16)*8 + [0..8)`` (its half). The peer lane ``l^16`` holds the other
    8-row half of the same q, so the row max/sum reduce locally over (kvt, i) then across
    the ``l^16`` partner.

    Masking (per element, per row r, sequence-relative ``kv_pos = kv_pos_base + (l//16)*8 +
    kvt*16 + i``; all bounds fx.Int32): ``q_max_list[r]`` masks ``kv_pos > q_max`` (the
    causal edge ``q_seq + (kv_len - q_len)``; clamped to ``kv_len-1`` on the last tile to
    fold the tail); kv_len (only when q_max is None) masks ``kv_pos >= kv_len`` (the tail of
    the non-causal case). ``q_max_list`` None and kv_len None (an interior tile) mask
    nothing.

    Args: ``s_list``/``m_prev_list``/``d_prev_list``/``q_max_list`` are length-R lists
    (R = WMMA_ROW_PER_WAVE). m_prev/d_prev are fx.Float32 shared by the l<->l^16 pair.

    Returns 5 length-R lists ``(p, m_new, d_new, corr, do_rescale)`` -- per row: p = NKV v8
    **bf16** P^T = exp(S^T - m_new); m_new = updated running max, STALE (== m_prev) when the
    deferred-rescale ballot did not fire; d_new = corr*d_prev + rowsum(p);
    corr = exp(m_prev - m_new) (== 1 on the stale path); do_rescale = wave-uniform i1.
    """
    NKV = N_BLOCK // WMMA_N
    f32 = T.f32
    fast = arith.FastMathFlags.fast
    neg_inf = fx.Float32(float("-inf"))
    zero = fx.Float32(0.0)
    log2e = fx.Float32(LOG2E)

    def fmax(a, b):
        return arith.maxnumf(a, b, fastmath=fast)

    def fadd(a, b):
        with arith.fastmath(fast):
            return a + b

    # fast-math WITHOUT reassoc: LLVM's Reassociate pass otherwise re-linearizes the
    # sum tree back into a serial chain (max survives -- Reassociate ignores maxnum).
    _FF = arith.FastMathFlags
    _no_reassoc = _FF.nnan | _FF.ninf | _FF.nsz | _FF.arcp | _FF.contract | _FF.afn

    def fadd_t(a, b):
        with arith.fastmath(_no_reassoc):
            return a + b

    def fsub(a, b):
        with arith.fastmath(fast):
            return a - b

    def fmul(a, b):
        with arith.fastmath(fast):
            return a * b

    def exp2(x):  # native v_exp_f32 (no denorm range reduction), unlike fx.math.exp2
        return fx.Float32(rocdl.exp2(f32, x.ir_value()))

    # permlanex16 selectors: identity cross-16 gather (nibbles 0..15) => lane l<->l^16.
    sel_lo, sel_hi = fx.Int32(0x76543210).ir_value(), fx.Int32(0xFEDCBA98).ir_value()

    def peer(v):  # cross-lane reduce partner: lane l <-> l^16 (the other kv half)
        return fx.Float32(
            rocdl.permlanex16(
                f32,
                v.ir_value(),
                v.ir_value(),
                sel_lo,
                sel_hi,
                fi=False,
                bound_control=False,
            )
        )

    khalf = lane_idx // fx.Int32(WMMA_M)  # 0/1: which 8-row kv half this lane owns

    R = len(s_list)

    # ---- Pass 1 (all R rows): masked S values, flattened (kvt, i) order. Built for every
    # row first so the row max-trees below emit INTERLEAVED. ----
    s_masked_list = []
    for r in range(R):
        s = s_list[r]
        q_max = q_max_list[r]
        s_masked = []
        for kvt in range(NKV):
            svec = fx.Vector(s[kvt])
            for i in range(8):
                sval = svec[i]
                if q_max is not None or kv_len is not None:
                    kv_pos = kv_pos_base + khalf * fx.Int32(8) + fx.Int32(kvt * WMMA_N + i)
                    if q_max is not None:
                        ubound = q_max if kv_len is None else fx.min(q_max, kv_len - fx.Int32(1))
                        sval = (kv_pos > ubound).select(neg_inf, sval)
                    if kv_len is not None and q_max is None:
                        sval = (kv_pos >= kv_len).select(neg_inf, sval)
                s_masked.append(sval)
        s_masked_list.append(s_masked)

    # ---- Row max: the R rows' balanced max-trees emitted INTERLEAVED (position-major
    # across rows) so the backend dual-issues row0/row1 combines and hides the cross-lane
    # permlanex16 latency. ----
    m_new_list, corr_list, neg_m_list, do_rescale_list = [], [], [], []
    max3 = lambda a, b, c: fmax(fmax(a, b), c)
    local_max_list = _tree_reduce_multi(s_masked_list, max3, fmax)

    # ---- Per row: peer reduce + deferred-rescale decision + corr / neg_m. ----
    for r in range(R):
        m_prev = m_prev_list[r]
        row_max = fmax(local_max_list[r], peer(local_max_list[r]))
        m_full = fmax(m_prev, row_max)

        # Deferred O rescale (see RESCALE_THRESHOLD): keep m STALE while the running max
        # barely moves, so the caller SKIPS the wide `o_acc *= corr` multiply. The ballot
        # makes the per-lane test wave-uniform (non-divergent branch). `>` lowers to an
        # ordered compare, so a fully masked lane's -inf - -inf = NaN compares false and
        # never forces a rescale. Stale path: row_max - m_prev <= 8 -> p <= e^8, no overflow.
        need = fsub(row_max, m_prev) > fx.Float32(RESCALE_THRESHOLD)
        mask = rocdl.ballot(fx.Int32.ir_type, need)
        do_rescale = fx.Int32(mask) != fx.Int32(0)
        m_new = do_rescale.select(m_full, m_prev)

        # corr = exp(m_prev - m_new); neg_m = -(m_new * log2e) for the fused p exp.
        # m is seeded to BIG_NEG (finite), so m_prev/m_new never reach -inf: a fully
        # masked row (row_max=-inf) keeps m_new=BIG_NEG, giving corr=exp2(0)=1 and a
        # finite neg_m (p=exp2(-inf)=0). No (-inf)-(-inf) / -inf+inf, so no clamp needed.
        corr = exp2(fmul(fsub(m_prev, m_new), log2e))
        neg_m = fsub(zero, fmul(m_new, log2e))
        m_new_list.append(m_new)
        corr_list.append(corr)
        neg_m_list.append(neg_m)
        do_rescale_list.append(do_rescale)

    # ---- Pass 2 (all R rows): p = exp(S - m_new) (bf16, per tile) + flat p for the sum
    # tree. Built for every row first so the row sum-trees below emit INTERLEAVED. ----
    p_list, p_flat_list = [], []
    for r in range(R):
        neg_m, s_masked = neg_m_list[r], s_masked_list[r]
        p, p_flat, idx = [], [], 0
        for kvt in range(NKV):
            pe = []
            # exp2 argument as packed fma (v_pk_fma_f32), 2 elems per op.
            l2 = fx.Vector.from_elements([log2e], fx.Float32).broadcast_to(2)
            n2 = fx.Vector.from_elements([neg_m], fx.Float32).broadcast_to(2)
            for i in range(0, 8, 2):
                sv = fx.Vector.from_elements([s_masked[idx], s_masked[idx + 1]], fx.Float32)
                av = fmath.fma(sv, l2, n2)
                for e in range(2):
                    pj = exp2(av[e])
                    pe.append(pj)
                    p_flat.append(pj)
                idx += 2
            p.append(fx.Vector.from_elements(pe, fx.Float32).to(ELEM_DTYPE))
        p_list.append(p)
        p_flat_list.append(p_flat)

    # ---- Row sum: R rows' balanced sum-trees emitted INTERLEAVED. fadd_t (fast-math minus
    # reassoc) so LLVM's Reassociate does NOT re-linearize the tree into a serial chain. ----
    add3 = lambda a, b, c: fadd_t(fadd_t(a, b), c)
    if R == 2:
        # The two rows' trees have the same shape, so run them in lockstep as
        # one v2 tree (row0, row1) -> v_pk_add_f32; per-row association unchanged (bitwise).
        def vadd_t(a, b):
            with arith.fastmath(_no_reassoc):
                return a + b

        vadd3 = lambda a, b, c: vadd_t(vadd_t(a, b), c)
        leaves = [
            fx.Vector.from_elements([p_flat_list[0][i], p_flat_list[1][i]], fx.Float32)
            for i in range(len(p_flat_list[0]))
        ]
        (tot,) = _tree_reduce_multi([leaves], vadd3, vadd_t)
        local_sum_list = [tot[0], tot[1]]
    else:
        local_sum_list = _tree_reduce_multi(p_flat_list, add3, fadd_t)

    # fadd_t (no reassoc), so the order is corr*d + (own + peer) as written; with `fast`
    # LLVM re-associated it (e.g. (corr*d + peer) + own).
    d_new_list = []
    for r in range(R):
        d_new_list.append(
            fadd_t(
                fmul(corr_list[r], d_prev_list[r]),
                fadd_t(local_sum_list[r], peer(local_sum_list[r])),
            )
        )
    return p_list, m_new_list, d_new_list, corr_list, do_rescale_list


def _pv_gemm(*, v_values, p_list, v_hdim, o_acc_list):
    """GEMM2: O^T = V^T @ P^T for one resident KV tile, for all R q-WMMA-tiles this
    wave owns. V is **shared** across the q-tiles (transpose-loaded once), so each
    V fragment is shuffled once and fed into R independent WMMA chains.

    WMMA convention (gfx1250): D[M=d, N=q] with **A = V^T** (src_a, transpose-loaded
    via ds_load_tr16_b128) and **B = P^T** (src_b, the bf16 softmax output). Contract
    kv in ``nkt = N_BLOCK//WMMA_K`` tiles (K=32); produce ``d_tiles = v_hdim//WMMA_M``
    output d-tiles (M axis; ``v_hdim`` is the head-dim columns this wave owns). Lane
    ``l`` element ``si`` of tile ``dt`` holds O[q = l%16, d = dt*WMMA_M + (l//16)*8 + si]
    -- the O writer's fragment layout.

    ``p_list`` is a length-R list; entry ``qt`` is that q-tile's list of softmax
    kv-tiles (bf16 P^T B-operands). ``o_acc_list`` is a length-R list of running O
    accumulators (each ``d_tiles`` v8-f32, already rescaled by ``corr``). Returns
    ``out_list``: a length-R list of updated O accumulators.

    ``v_values`` is the already-burst-loaded flat ``(dt, kt, half)`` list of v8-bf16
    transpose-load results (from ``v_mgr.load_v_to_reg``, kept OUT of the WMMA stream so there is
    no wmma->ds_load issue bubble); a single ``s_wait_dscnt(0)`` drains that burst
    here before any WMMA. Online accumulation: each tile's PV adds onto the running
    o_acc. Each WMMA operand is a v16 bf16 fragment = two 16-wide halves shuffled:
    A from V-tiles (kv, kv+16), B from softmax tiles (p[2kt], p[2kt+1]).
    """
    R = len(p_list)
    d_tiles = v_hdim // WMMA_M  # output d-tiles (M axis, WMMA_M d rows each)
    nkt = N_BLOCK // WMMA_K  # kv contraction tiles (K=32 kv each)

    # Drain the whole V transpose burst once; every v_values entry is now resident.
    rocdl.s_wait_dscnt(0)

    out_list = [[None] * d_tiles for _ in range(R)]
    for dt in range(d_tiles):
        accs = [o_acc_list[qt][dt] for qt in range(R)]
        j = dt * nkt * 2
        for kt in range(nkt):
            # A-operand: V^T frag = two 16-kv transpose-load tiles -> v16 bf16
            # (shared across q-tiles).
            v_lo = v_values[j]
            v_hi = v_values[j + 1]
            j += 2
            v_frag = v_lo.shuffle(v_hi, list(range(16)))
            for qt in range(R):
                # B-operand: P^T frag = two consecutive softmax kv-tiles -> v16 bf16.
                p = p_list[qt]
                p_frag = p[2 * kt].shuffle(p[2 * kt + 1], list(range(16)))
                accs[qt] = _wmma(v_frag, p_frag, accs[qt])
        for qt in range(R):
            out_list[qt][dt] = accs[qt]
    return out_list


# ============================================================================
# Compute core
# ============================================================================


def _alloc_lds(nbytes):
    """Allocate the workgroup's LDS once and return its base (fx.Int32). Called once per
    kernel body before the warp-type dispatch so both ``_core_attention`` traces share the
    single SharedAllocator flydsl permits; K/V, Q, and the O epilogue all carve this base.
    """
    smem = fx.SharedAllocator().allocate(nbytes)
    return fx.Int32(fx.ptrtoint(smem.peek().ptr))


def _is_lo_warp(tiling, warp_idx):
    """Runtime test of the warp-type dispatch (see Tiling.warp_specialized)."""
    if tiling.warp_specialized:
        return warp_idx // fx.Int32(tiling.num_waves // 2) == fx.Int32(0)
    return warp_idx >= fx.Int32(0)


def _core_attention(
    *,
    tiling,  # compile-time Tiling
    causal,  # compile-time: mask keys past each query's (bottom-right) diagonal
    gqa_ratio,  # compile-time GQA group size = nheads_q // nheads_kv
    ptr_O,
    ptr_Q,
    ptr_K,
    ptr_V,
    ptr_LSE,
    softmax_scale,
    stride_q_seq,
    stride_k_seq,
    stride_v_seq,
    stride_o_seq,
    stride_q_head,
    stride_k_head,
    stride_v_head,
    stride_o_head,
    # LSE addressing (element strides + per-batch bound), resolved by the caller.
    stride_lse_seq,
    stride_lse_head,
    lse_base_elems,  # first element offset of this batch's LSE slab
    lse_num_records_bytes,  # buffer-resource bound (below the 0x7FFFFFFF drop)
    # Per-batch token ranges (fx.Int32), resolved by the caller:
    q_start,  # first Q token index of this batch in the global tensor
    q_len,  # valid Q tokens in this batch
    kv_start,  # first K/V token index of this batch
    kv_len,  # valid K/V tokens in this batch
    # Runtime fx.Int32 offset of the causal edge (the host passes 0); read only when causal.
    window_right,
    warp_idx,  # runtime fx.Int32 wave index
    warp_type,  # compile-time WarpType (LO_WARP / HI_WARP)
    lds_base,  # LDS base (fx.Int32), allocated once by the caller (_alloc_lds)
):
    """Attention of one workgroup's packed q rows over one batch's kv sequence.

    The caller resolves the per-batch token ranges (``q_start``/``q_len`` and
    ``kv_start``/``kv_len``) and the LSE slab.

    Warp-specialized: the caller dispatches on the runtime warp type and traces this body
    TWICE (once per compile-time ``warp_type``); the two instantiations differ only in the
    ``main_loop`` preamble ordering (see WarpType).
    """
    block_m = tiling.block_m
    d_split = tiling.d_split
    lane_idx = _lane_id()
    if d_split > 1:
        # D-split: row_w picks the q rows (QK + softmax, redundant across the waves that
        # share them), dh the part of d this wave accumulates and stores. The K/V TDM loads
        # stay split over all waves.
        row_w = warp_idx % fx.Int32(tiling.num_row_waves)
        dh = warp_idx // fx.Int32(tiling.num_row_waves)
    else:
        row_w, dh = warp_idx, None
    kv_head, q_head_idx, seq_idx = _packed_tile_indices(gqa_ratio, block_m, row_w, lane_idx)

    # K/V staging: N_KV_PP ping-pong slots ([K.pp0|V.pp0][K.pp1|V.pp1]), each K and V block
    # floored at tiling.min_kv_blk_bytes. O reuses a non-current slot; Q time-shares slot 1,
    # so the slot must also hold the Q staging footprint (one copy per d part, so no two
    # waves TDM-write the same bytes). slot_bytes is compile-time (no allocation; lds_base is
    # passed in).
    q_mgr = QManager16bV2(
        qk_hdim=HEAD_DIM,
        gqa_ratio=gqa_ratio,
        num_waves=tiling.num_row_waves,
        q_tiles_per_wave=WMMA_ROW_PER_WAVE,
    )
    k_mgr = KManager16bV2(qk_hdim=HEAD_DIM, n_block=N_BLOCK, num_waves=tiling.num_waves)
    v_mgr = VManager16bV2(v_hdim=HEAD_DIM, n_block=N_BLOCK, num_waves=tiling.num_waves)
    k_blk_bytes = max(k_mgr.get_lds_size_in_byte(), tiling.min_kv_blk_bytes)
    v_blk_bytes = max(v_mgr.get_lds_size_in_byte(), tiling.min_kv_blk_bytes)
    slot_bytes = max(k_blk_bytes + v_blk_bytes, d_split * q_mgr.get_lds_size_in_byte())
    assert N_KV_PP * slot_bytes <= tiling.lds_bytes, f"LDS ring {N_KV_PP}x{slot_bytes} exceeds allocation"

    def _k_lds_buf(pp):  # K base of ping-pong slot ``pp`` (int or fx.Int32; folds when const)
        if isinstance(pp, int):
            pp = fx.Int32(pp)
        return lds_base + pp * fx.Int32(slot_bytes)

    def _v_lds_buf(pp):  # V base of ping-pong slot ``pp`` (== K base + k_blk_bytes)
        if isinstance(pp, int):
            pp = fx.Int32(pp)
        return lds_base + pp * fx.Int32(slot_bytes) + fx.Int32(k_blk_bytes)

    # ---- Q staging TIME-SHARES slot 1: Q's LDS base = slot-1 base (kv_base +
    # slot_bytes), plus this wave's d-part copy under d-split. Q is loaded + drained into
    # VGPR in the prologue, then dead; the main loop's first slot-1 prefetch reuses the
    # region. Safe with zero new sync -- the prologue drains Q (part2) -> tensor_wait(0) ->
    # gpu.barrier() BEFORE the loop, and prologue K/V loads target slot 0. slot_bytes >= Q
    # footprint by construction (see above), so Q always fits in slot 1. ----
    q_lds_base = lds_base + fx.Int32(slot_bytes)
    if dh is not None:
        q_lds_base = q_lds_base + dh * fx.Int32(q_mgr.get_lds_size_in_byte())

    q_mgr.load_q_to_vgpr_part1(
        ptr_Q=ptr_Q,
        stride_q_seq=stride_q_seq,
        stride_q_head=stride_q_head,
        q_start=q_start,
        q_len=q_len,
        kv_head=kv_head,
        block_x=_lpt_block_id("x"),
        warp_idx=row_w,
        lane_idx=lane_idx,
        ptr_lds=q_lds_base,
    )

    # ---- This WG's KV tiles span relative kv [0, kv_len_wg). Packed row r maps to seq
    # r//gqa_ratio; under causal a query at seq s attends kv <= s + causal_off
    # (causal_off = kv_len - q_len, plus window_right), so kv_len_wg clips to the WG's max
    # query's limit and no tile fully past the diagonal runs. Non-causal: all kv (kv_len).
    block_x = _lpt_block_id("x")
    causal_off = kv_len - q_len
    if causal:
        wg_max_seq = (block_x * fx.Int32(block_m) + fx.Int32(block_m - 1)) // fx.Int32(gqa_ratio)
        wg_max_seq = fx.min(wg_max_seq, q_len - fx.Int32(1))
        kv_len_wg = wg_max_seq + causal_off + window_right + fx.Int32(1)
        kv_len_wg = fx.min(kv_len_wg, kv_len)
        kv_len_wg = fx.max(kv_len_wg, fx.Int32(1))
    else:
        kv_len_wg = kv_len

    # Tile range [start_tile, n_tiles): every WG starts at tile 0 (no left window edge).
    n_tiles = fx.ceildiv(kv_len_wg, fx.Int32(N_BLOCK))
    start_tile = fx.Int32(0)

    def _kv_valid(blk_row0):
        # How many rows of [blk_row0, blk_row0+N_BLOCK) are in-bounds, clamped to
        # the WG's effective KV length kv_len_wg (0..N_BLOCK). Past the end -> 0 (a
        # harmless clamped load that is never consumed).
        rem = fx.max(kv_len_wg - blk_row0, fx.Int32(0))
        return fx.min(rem, fx.Int32(N_BLOCK))

    # ---- Prologue (ordered for sched mode 2): build the first tile's K/V TDM copy views
    # in the Q global-load shadow, then run part2 (Q ds_load), then issue the K/V TDM
    # copies LAST -- so NOTHING runs between the loads and the prologue barrier below.
    #
    # Ping-pong parity is LOCAL to this WG's tile stream: the prologue always loads the
    # first tile (start_tile) into buffer 0, and the main loop selects buffers by the
    # 0-based LOCAL iteration index (not the absolute tile index).
    start_row0 = start_tile * fx.Int32(N_BLOCK)
    k_views = k_mgr.load_views(
        ptr_lds=_k_lds_buf(0),
        ptr_K=ptr_K,
        stride_k_seq=stride_k_seq,
        stride_k_head=stride_k_head,
        kv_head=kv_head,
        kv_row0=kv_start + start_row0,
        kv_valid=_kv_valid(start_row0),
    )
    v_views = v_mgr.load_views(
        ptr_lds=_v_lds_buf(0),
        ptr_V=ptr_V,
        stride_v_seq=stride_v_seq,
        stride_v_head=stride_v_head,
        kv_head=kv_head,
        kv_row0=kv_start + start_row0,
        kv_valid=_kv_valid(start_row0),
    )
    q_frags = q_mgr.load_q_to_vgpr_part2(scale=softmax_scale)
    for _v in k_views:
        fx.copy_atom_call(*_v)
    for _v in v_views:
        fx.copy_atom_call(*_v)
    tdm_ops.tensor_wait(0)
    gpu.barrier()

    # Loop init -- placed after the prologue barrier. Online-softmax seed + O
    # accumulators (iter_args) and loop bounds.
    #
    # Loop-carried state (scf.for_ iter_args): per q-tile the online-softmax running max
    # ``m`` and denom ``d`` (per-lane f32), followed by its ``d_tiles`` fp32 O accumulators.
    # Seed m=BIG_NEG (finite, see BIG_NEG), d=0, O=0: the first real tile's
    # corr=exp2(m_prev-m_new) underflows to 0 and zeroes the (already-zero) O before its PV
    # adds in -- the standard flash seed. Fully-masked leading tiles keep m finite, so
    # no NaN.
    d_tiles = HEAD_DIM // WMMA_M // d_split  # O tiles of the d part this wave owns
    v_hdim_w = HEAD_DIM // d_split  # head-dim columns of O this wave owns
    R = WMMA_ROW_PER_WAVE
    _QS = 2 + d_tiles  # per-q-tile carried state: [m, d, O_0 .. O_{d_tiles-1}]
    m_init = [fx.Float32(BIG_NEG) for _ in range(R)]
    d_init = [fx.Float32(0.0) for _ in range(R)]
    _init = []
    for qt in range(R):
        _init += fx.as_ir_value(
            [m_init[qt], d_init[qt]] + [fx.Vector.filled(8, 0.0, fx.Float32) for _ in range(d_tiles)]
        )

    # ---- ds_load LDS base pointers for both ping-pong buffers, carried as iter_args and
    # swapped curr<->next each iteration (buffer selected by pointer). Base count per mgr
    # is manager-defined -- carried generically. Under d-split a wave reads only the V
    # columns of its d part. ----
    k_lds_ld_curr = k_mgr.ds_load_ptrs(ptr_lds=_k_lds_buf(0), lane_idx=lane_idx)
    v_dh_off = None if dh is None else dh * fx.Int32(v_hdim_w * 2)  # bf16 byte offset

    def _v_ld_base(pp):
        return _v_lds_buf(pp) if v_dh_off is None else _v_lds_buf(pp) + v_dh_off

    v_lds_ld_curr = v_mgr.ds_load_ptrs(ptr_lds=_v_ld_base(0), lane_idx=lane_idx)
    k_lds_ld_next = k_mgr.ds_load_ptrs(ptr_lds=_k_lds_buf(1), lane_idx=lane_idx)
    v_lds_ld_next = v_mgr.ds_load_ptrs(ptr_lds=_v_ld_base(1), lane_idx=lane_idx)
    _NKB = len(k_lds_ld_curr)  # ds bases per K buffer
    _NVB = len(v_lds_ld_curr)
    _PTR_BASE = len(_init)
    _init = _init + k_lds_ld_curr + k_lds_ld_next + v_lds_ld_curr + v_lds_ld_next

    # ========================================================================
    # Main KV loop -- stream tiles [start_tile, n_tiles) through the N_KV_PP ping-pong
    # ring, one tile per iteration. The buffer is selected by the carried curr ds
    # pointers, swapped curr<->next at the end of each `main_loop`.
    #
    # Each `main_loop` call is one tile: it drains outstanding async + barriers (its KV
    # is then GUARANTEED resident -- start_tile from the prologue, every later tile from
    # the previous call's top-of-body bulk prefetch into the OTHER buffer), issues the
    # tile t+1 prefetch UP FRONT into the other buffer, then computes QK->softmax->PV on
    # its own buffer while that async copy runs, drained at the next call's top.
    # ========================================================================
    def main_loop(t, state, *, mask, kv_len):
        # mask (apply the causal mask) and kv_len (mask the kv tail) are compile-time per
        # sub-loop: the caller splits the tile stream into a mask-free clean region and a
        # boundary loop, and passes False / None where a sub-loop provably needs neither.
        #
        # Runtime ping-pong: this tile reads its curr buffer (carried curr pointers); the
        # tile t+1 prefetch writes the next buffer, and curr<->next are swapped in the yield.
        nxt_pp = (t - start_tile + fx.Int32(1)) % fx.Int32(2)

        kv_tile_start = t * fx.Int32(N_BLOCK)  # this tile's first (batch-relative) kv row

        # Unpack loop-carried state -- R independent per-q-tile (m, d, O) groups,
        # then the shared K/V ds pointers.
        m_prev = [fx.Float32(state[qt * _QS + 0]) for qt in range(R)]
        d_prev = [fx.Float32(state[qt * _QS + 1]) for qt in range(R)]
        o_acc = [[fx.Vector(state[qt * _QS + 2 + dt]) for dt in range(d_tiles)] for qt in range(R)]
        k_curr = list(state[_PTR_BASE + 0 * _NKB : _PTR_BASE + 1 * _NKB])
        k_next = list(state[_PTR_BASE + 1 * _NKB : _PTR_BASE + 2 * _NKB])
        _VB0 = _PTR_BASE + 2 * _NKB
        v_curr = list(state[_VB0 + 0 * _NVB : _VB0 + 1 * _NVB])
        v_next = list(state[_VB0 + 1 * _NVB : _VB0 + 2 * _NVB])

        # Warp-specialized preamble: same pieces, ordered per warp type so the SIMD-mate
        # pair (i / i+4) staggers K load vs prefetch. Correctness is warp-type-independent
        # (each wave reads its own resident K under the workgroup barrier).
        nxt = t + fx.Int32(1)
        nxt_row0 = nxt * fx.Int32(N_BLOCK)
        nxt_valid = _kv_valid(nxt_row0)

        def _addr_phase():
            # Pure (no memory op) -> hoistable: the TDM copy views for tile t+1's K/V
            # into the nxt_pp buffer.
            k_views = k_mgr.load_views(
                ptr_lds=_k_lds_buf(nxt_pp),
                ptr_K=ptr_K,
                stride_k_seq=stride_k_seq,
                stride_k_head=stride_k_head,
                kv_head=kv_head,
                kv_row0=kv_start + nxt_row0,
                kv_valid=nxt_valid,
            )
            v_views = v_mgr.load_views(
                ptr_lds=_v_lds_buf(nxt_pp),
                ptr_V=ptr_V,
                stride_v_seq=stride_v_seq,
                stride_v_head=stride_v_head,
                kv_head=kv_head,
                kv_row0=kv_start + nxt_row0,
                kv_valid=nxt_valid,
            )
            return (k_views, v_views)

        def _drain_barrier():
            tdm_ops.tensor_wait(0)
            rocdl.sched_barrier(0)
            gpu.barrier()
            rocdl.sched_barrier(0)

        def _prefetch(addr):
            # Skip t+1 prefetch on the last tile: a dead copy into the O-epilogue slot
            # races the epilogue O write across waves.
            @flyc.jit
            def _issue():
                if nxt < n_tiles:
                    k_views, v_views = addr
                    for _v in k_views:
                        fx.copy_atom_call(*_v)
                    for _v in v_views:
                        fx.copy_atom_call(*_v)

            _issue()

        # Address VALU up front (no barrier dependency) so it overlaps the drain; only
        # the async issue in _prefetch must stay after the barrier.
        addr = _addr_phase()
        if warp_type == WarpType.LO_WARP:
            _drain_barrier()
            k_values = k_mgr.load_k_to_reg(k_curr)
            _prefetch(addr)
        else:
            _drain_barrier()
            _prefetch(addr)
            k_values = k_mgr.load_k_to_reg(k_curr)

        # Fence the K burst out of the WMMA stream (no wmma<-ds_load bubble). No explicit
        # s_wait_dscnt: the K ds_load is SSA-visible, so LLVM inserts the dscnt wait itself
        # (mode 2), placed so the WMMA can issue as soon as its operands land.
        rocdl.sched_barrier(0)

        # ---- GEMM1: S^T = K @ Q^T for this KV tile (== P^T pre-softmax); consumes the
        # pre-loaded k_values (K burst already drained by the preamble above). ----
        s_list = _qk_gemm(k_values=k_values, q_frags_list=q_frags)

        # ---- Burst this tile's V transpose ds_loads (this wave's d part; out of the WMMA
        # stream), now that QK has consumed the K burst. Issued BEFORE softmax so the
        # ds_load latency hides under softmax's VALU; the values are consumed by _pv_gemm
        # after the rescale. ----
        rocdl.sched_barrier(0)
        v_values = v_mgr.load_v_to_reg(v_curr, d_tiles)
        rocdl.sched_barrier(0)

        # ---- Softmax: online update over this KV tile's kv axis, INDEPENDENTLY per
        # q-tile. Under causal this lane's query (tile qt) attends kv <= q_max =
        # seq+causal_off+window_right (batch-relative). A clean-region tile passes
        # q_max None and does zero per-element masking. kv_len is passed only on the last
        # tile (folds the OOB tail into the q_max clamp / standalone tail mask). K/V are
        # shared, but each q-tile has its own S and running m/d. All R q-tiles go through
        # ONE call so the rows' max/sum tree reductions emit INTERLEAVED (ILP). ----
        q_max_list = [seq_idx[qt] + causal_off + window_right if mask else None for qt in range(R)]
        p_list, m_new_list, d_new_list, corr_list, do_rescale_list = _softmax(
            s_list=s_list,
            m_prev_list=m_prev,
            d_prev_list=d_prev,
            lane_idx=lane_idx,
            kv_pos_base=kv_tile_start,
            q_max_list=q_max_list,
            kv_len=kv_len,
        )

        # ---- Rescale each q-tile's running O by its corr, then GEMM2 accumulates this
        # tile. The wide `o_acc *= corr` multiply (d_tiles*8 f32/lane) sits behind a
        # non-divergent scf.if on the wave-uniform do_rescale, so it runs only when the
        # running max actually moved; on the stale path corr == 1 and o_acc passes
        # through untouched. Each q-tile decides its own rescale. ----
        @flyc.jit
        def _maybe_rescale(o_vecs, corr_vec, do_rescale):
            result = o_vecs
            if do_rescale:
                result = [ov * corr_vec for ov in o_vecs]
            return result

        o_resc_list = []
        for qt in range(R):
            corr_vec = fx.Vector.from_elements([corr_list[qt]], fx.Float32).broadcast_to(8)
            o_vecs = list(o_acc[qt])
            o_resc_list.append(list(_maybe_rescale(o_vecs, corr_vec, do_rescale_list[qt])))

        o_new_list = _pv_gemm(
            v_values=v_values,
            p_list=p_list,
            v_hdim=v_hdim_w,
            o_acc_list=o_resc_list,
        )

        # Yield state -- R updated (m, d, O) groups, then the shared K/V ds pointers
        # swapped curr<->next (4 s_swap_b32).
        out = []
        for qt in range(R):
            out += fx.as_ir_value([m_new_list[qt], d_new_list[qt], *o_new_list[qt]])
        # Swap curr<->next ds bases (manager-defined count each).
        out += k_next + k_curr + v_next + v_curr
        return out

    # ---- Stream tiles [start_tile, n_tiles) through 2 sub-loops split at the causal
    # diagonal, so interior tiles fully inside it skip masking. clean_lo/clean_hi are
    # runtime split points, but the mask on/off per sub-loop is COMPILE-TIME (each loop
    # traces main_loop once with fixed None-ness). Ping-pong swap state threads
    # continuously through both; buffer parity is by LOCAL iteration index, so the
    # split leaves it intact.
    #   [clean_lo, clean_hi) clean, no mask
    #   [clean_hi, n_tiles)  causal boundary + kv_len tail (last tile)
    n_iter = n_tiles - start_tile
    n_last = n_tiles - fx.Int32(1)  # last tile always carries the kv_len tail

    # clean_hi = first tile that could need causal masking = the WG's earliest query's
    # diagonal tile ((min q_max + 1)//N_BLOCK). Kept <= n_last so the tail tile stays in
    # the boundary loop, and >= start_tile for a valid partition.
    if causal:
        wg_min_seq = (block_x * fx.Int32(block_m)) // fx.Int32(gqa_ratio)
        qmax_min = fx.max(wg_min_seq + causal_off + window_right, fx.Int32(0))
        clean_hi = (qmax_min + fx.Int32(1)) // fx.Int32(N_BLOCK)
    else:
        clean_hi = n_tiles
    clean_hi = fx.max(fx.min(clean_hi, n_last), start_tile)
    # The clean region starts at the first tile, clamped into [start_tile, clean_hi].
    clean_lo = fx.min(start_tile, clean_hi)

    @flyc.jit
    def _run_tiles(state, lo_i32, hi_i32, *, mask, kv_len):
        final_state = state
        for tile, carried in range(fx.Index(lo_i32), fx.Index(hi_i32), 1, init=state):
            next_state = main_loop(fx.Int32(tile), list(carried), mask=mask, kv_len=kv_len)
            final_state = yield next_state
        return final_state

    state = _run_tiles(_init, clean_lo, clean_hi, mask=False, kv_len=None)
    final = _run_tiles(state, clean_hi, n_tiles, mask=causal, kv_len=kv_len)

    # ========================================================================
    # Epilogue: normalize O by the running denom d, then reshape+store to VRAM.
    # ========================================================================
    # O staging reuses the NON-CURRENT K|V slot (plus this wave's d-part copy under
    # d-split). Local-parity ring: n_iter tiles occupy local indices 0..n_iter-1, so the
    # LAST tile lives in buffer (n_iter-1)%N_KV_PP and the free (non-current) slot is
    # n_iter%N_KV_PP. With the last iteration's dead prefetch skipped, no wave ever writes
    # that slot near the end: the last load into it was local tile n_iter-2 (issued during
    # local tile n_iter-3, consumed at n_iter-2), and the top-of-body barrier at local tile
    # n_iter-1 already synchronized every wave past that read. So the slot is idle here --
    # no cross-wave barrier needed. K/V/Q load via TDM (tensorcnt), so no async load is
    # inflight here. The O writer stages each q-tile and flushes the wave's R tiles
    # together on the last one.
    o_mgr = OManager16bV3(
        v_hdim=v_hdim_w,
        gqa_ratio=gqa_ratio,
        num_waves=tiling.num_row_waves,
        q_tiles_per_wave=R,
    )
    assert d_split * o_mgr.get_lds_size_in_byte() <= slot_bytes, (
        f"O staging {o_mgr.get_lds_size_in_byte()}B exceeds K|V slot {slot_bytes}B"
    )
    non_cur_pp = n_iter % fx.Int32(N_KV_PP)
    # O strides are in ELEMENTS (the O writer scales by _BF16_BYTES itself).
    o_lds_base = _k_lds_buf(non_cur_pp)
    if dh is not None:
        o_lds_base = o_lds_base + dh * fx.Int32(o_mgr.get_lds_size_in_byte())
    for qt in range(R):
        # Normalize this q-tile's O by its running denom d, then reshape+store to VRAM.
        # o_final[dt] lane l elem si = sum_kv P[q,kv] V[kv, dt*16+(l//16)*8+si]
        # (unnormalized); divide by the per-query denom d (peer-consistent across the
        # lane pair) to finish softmax. The O writer masks rows with seq >= q_len.
        d_final = fx.Float32(final[qt * _QS + 1])
        o_final = [fx.Vector(final[qt * _QS + 2 + dt]) for dt in range(d_tiles)]
        # Fully-masked row (d_final==0): 1/0=inf, o_final=0, 0*inf=NaN -> guard to O=0.
        inv = (d_final > fx.Float32(0.0)).select(fx.Float32(1.0) / d_final, fx.Float32(0.0))
        inv_vec = fx.Vector.from_elements([inv], fx.Float32).broadcast_to(8)
        # No dependency cover is placed between the last PV WMMA and this multiply: the
        # upstream kernel's notes report that adding one made the rare mode-2 failure (see
        # ENABLE_SCHED_MODE2) more frequent.
        o_norm = [o_final[dt] * inv_vec for dt in range(d_tiles)]
        if qt > 0:
            # No DS op is outstanding here (earlier q-tiles were only staged), so this wait is
            # free; it stays to keep the tuned instruction schedule.
            rocdl.s_wait_dscnt(0)
        o_mgr.store_o_to_vram(
            ptr_O=ptr_O,
            o_base_elems=fx.Int32(0) if dh is None else dh * fx.Int32(v_hdim_w),
            stride_o_seq=stride_o_seq,
            stride_o_head=stride_o_head,
            q_start=q_start,
            q_len=q_len,
            kv_head=kv_head,
            block_x=block_x,
            warp_idx=row_w,
            lane_idx=lane_idx,
            ptr_lds=o_lds_base,
            o_frags=o_norm,
            qtile=qt,
        )

    # ---- LSE store. LSE = m_final + ln(d_final) in the scaled-score domain
    # (softmax_scale is folded into Q, so S already carries it) -- matches
    # torch.logsumexp(scale * Q @ K^T, dim=kv). Each query q = warp*R*16 + qt*16 + l%16
    # is held identically by the lane pair (l, l^16); store once from the khalf==0
    # lanes (and, under d-split, from the d part 0 waves), masked by seq < q_len.
    # buffer_store redirects mask-drops to byte 0x7FFFFFFF, so lse_rsrc is bounded.
    # Emitted per q-tile.
    khalf0 = (lane_idx // fx.Int32(WMMA_M)) == fx.Int32(0)
    lse_rsrc = buffer_ops.create_buffer_resource(ptr_LSE, num_records_bytes=lse_num_records_bytes)
    for qt in range(R):
        m_final = fx.Float32(final[qt * _QS + 0])
        d_final = fx.Float32(final[qt * _QS + 1])
        # fx.log2 lowers to the HW v_log_f32 (base-2), so scale by ln2 (= 1/LOG2E)
        # to get the natural log for LSE = m + ln(d).
        ln_d = fx.log2(d_final) * fx.Float32(1.0 / LOG2E)
        lse_val = m_final + ln_d
        lse_mask = khalf0 & (seq_idx[qt] < q_len)
        if dh is not None:
            lse_mask = lse_mask & (dh == fx.Int32(0))
        lse_off_el = lse_base_elems + seq_idx[qt] * stride_lse_seq + q_head_idx[qt] * stride_lse_head
        # Pre-mask the offset (OOB rows -> 0x7fffffff) and pass mask=None so the
        # store maps 1:1 to a single buffer_store with masking already SSA-visible.
        lse_off_masked = lse_mask.select(lse_off_el * fx.Int32(4), fx.Int32(0x7FFFFFFF))
        buffer_ops.buffer_store(lse_val, lse_rsrc, lse_off_masked, mask=None, offset_is_bytes=True)


# ============================================================================
# Kernel, launcher and host entry
# ============================================================================


@functools.cache
def _launcher(tiling: Tiling, causal: bool, gqa_ratio: int):
    """The ``flyc.jit`` launcher of the forward kernel for one tiling, mask and GQA ratio.

    All three are compile-time: the kernel and the launcher capture them in their closures,
    which puts them in flydsl's JIT cache key, and ``gqa_ratio`` lets the per-lane ``% / //``
    fold to shift/and for powers of two. Every shape argument is a runtime Int32, so one
    launcher serves every (B, Sq, Skv, Hkv).
    """
    assert gqa_ratio >= 1, f"gqa_ratio must be >= 1, got {gqa_ratio}"

    # ptr_sink and window_left are unused, and window_right is always 0 (the causal edge
    # is the diagonal): the tuned kernel's signature is kept, since dropping or folding a
    # kernel argument changes the generated code.
    @flyc.kernel(name=tiling.kernel_name, known_block_size=[tiling.block_size, 1, 1])
    def k_flash_attn_fwd(
        ptr_O: fx.Tensor,
        ptr_Q: fx.Tensor,
        ptr_K: fx.Tensor,
        ptr_V: fx.Tensor,
        ptr_LSE: fx.Tensor,
        ptr_sink: fx.Tensor,
        softmax_scale: fx.Float32,
        stride_q_seq: fx.Int32,
        stride_k_seq: fx.Int32,
        stride_v_seq: fx.Int32,
        stride_o_seq: fx.Int32,
        stride_q_head: fx.Int32,
        stride_k_head: fx.Int32,
        stride_v_head: fx.Int32,
        stride_o_head: fx.Int32,
        stride_lse_seq: fx.Int32,
        stride_lse_head: fx.Int32,
        stride_lse_batch: fx.Int32,
        window_left: fx.Int32,
        window_right: fx.Int32,
        seq_len_q: fx.Int32,
        seq_len_k: fx.Int32,
    ):
        """Batched BSHD: uniform sequence lengths, token base batch * seq_len (batch =
        grid z). No per-call tensors besides q/k/v/o/lse, so the launch is CUDA-graph safe."""
        batch = _lpt_block_id("z")

        # LSE is [B, nheads_q, seq_q]: base = batch*stride_lse_batch; every valid
        # element offset is < base + stride_lse_batch (< the 0x7FFFFFFF drop).
        lse_base_elems = batch * stride_lse_batch
        lse_num_records_bytes = fx.Int64(lse_base_elems + stride_lse_batch) * fx.Int64(4)

        core_kw = {
            "tiling": tiling,
            "causal": causal,
            "gqa_ratio": gqa_ratio,
            "ptr_O": ptr_O,
            "ptr_Q": ptr_Q,
            "ptr_K": ptr_K,
            "ptr_V": ptr_V,
            "ptr_LSE": ptr_LSE,
            "softmax_scale": softmax_scale,
            "stride_q_seq": stride_q_seq,
            "stride_k_seq": stride_k_seq,
            "stride_v_seq": stride_v_seq,
            "stride_o_seq": stride_o_seq,
            "stride_q_head": stride_q_head,
            "stride_k_head": stride_k_head,
            "stride_v_head": stride_v_head,
            "stride_o_head": stride_o_head,
            "stride_lse_seq": stride_lse_seq,
            "stride_lse_head": stride_lse_head,
            "lse_base_elems": lse_base_elems,
            "lse_num_records_bytes": lse_num_records_bytes,
            "q_start": batch * seq_len_q,
            "q_len": seq_len_q,
            "kv_start": batch * seq_len_k,
            "kv_len": seq_len_k,
            "window_right": window_right,
        }
        lds_base = _alloc_lds(tiling.lds_bytes)
        warp_idx = _warp_id()
        if _is_lo_warp(tiling, warp_idx):
            _core_attention(warp_idx=warp_idx, warp_type=WarpType.LO_WARP, lds_base=lds_base, **core_kw)
        else:
            _core_attention(warp_idx=warp_idx, warp_type=WarpType.HI_WARP, lds_base=lds_base, **core_kw)

    @flyc.jit
    def _launch(
        ptr_O: fx.Tensor,
        ptr_Q: fx.Tensor,
        ptr_K: fx.Tensor,
        ptr_V: fx.Tensor,
        ptr_LSE: fx.Tensor,
        ptr_sink: fx.Tensor,
        softmax_scale: fx.Float32,
        stride_q_seq: fx.Int32,
        stride_k_seq: fx.Int32,
        stride_v_seq: fx.Int32,
        stride_o_seq: fx.Int32,
        stride_q_head: fx.Int32,
        stride_k_head: fx.Int32,
        stride_v_head: fx.Int32,
        stride_o_head: fx.Int32,
        stride_lse_seq: fx.Int32,
        stride_lse_head: fx.Int32,
        stride_lse_batch: fx.Int32,
        window_left: fx.Int32,
        window_right: fx.Int32,
        seq_len_q: fx.Int32,
        seq_len_k: fx.Int32,
        num_heads_kv: fx.Int32,
        batch_size: fx.Int32,
        stream: fx.Stream,
    ):
        # 3D grid: x = tiles over (seq, q_head_in_group) per kv-head,
        #          y = kv_head, z = batch. block = tiling.block_size threads.
        grid_x = fx.Index(fx.ceildiv(fx.Uint32(seq_len_q * gqa_ratio), fx.Uint32(tiling.block_m)))
        grid_y = fx.Index(num_heads_kv)
        grid_z = fx.Index(batch_size)

        launcher = k_flash_attn_fwd(
            ptr_O,
            ptr_Q,
            ptr_K,
            ptr_V,
            ptr_LSE,
            ptr_sink,
            softmax_scale,
            stride_q_seq,
            stride_k_seq,
            stride_v_seq,
            stride_o_seq,
            stride_q_head,
            stride_k_head,
            stride_v_head,
            stride_o_head,
            stride_lse_seq,
            stride_lse_head,
            stride_lse_batch,
            window_left,
            window_right,
            seq_len_q,
            seq_len_k,
        )
        launcher.launch(
            grid=(grid_x, grid_y, grid_z),
            block=(tiling.block_size, 1, 1),
            stream=stream,
        )

    _launch.compile_hints["llvm_options"] = {"amdgpu-expert-scheduling-mode": ENABLE_SCHED_MODE2}
    _launch.compile_hints["waves_per_eu"] = 2
    return _launch


def flash_attn_fwd(q, k, v, tiling=DEFAULT, *, softmax_scale, causal):
    """Returns (o [B, Sq, Hq, D] bf16, lse [B, Hq, Sq] fp32, natural log).

    q [B, Sq, Hq, D] and k/v [B, Skv, Hkv, D]: contiguous bf16 tensors of a problem that
    ``interface.unsupported_reason`` accepts (the caller checks). ``causal`` is bottom-right.
    ``tiling`` is DEFAULT or SMALL_GRID.
    """
    batch, seq_len_q, nheads_q, _ = q.shape
    seq_len_k, nheads_k = k.shape[1], k.shape[2]
    out = torch.empty((batch, seq_len_q, nheads_q, HEAD_DIM), dtype=q.dtype, device=q.device)
    lse = torch.empty((batch, nheads_q, seq_len_q), dtype=torch.float32, device=q.device)
    _run_compiled(
        _launcher(tiling, bool(causal), nheads_q // nheads_k),
        out,
        q,
        k,
        v,
        lse,
        q,  # ptr_sink: unused
        float(softmax_scale),
        q.stride(1),
        k.stride(1),
        v.stride(1),
        out.stride(1),
        q.stride(2),
        k.stride(2),
        v.stride(2),
        out.stride(2),
        lse.stride(2),
        lse.stride(1),
        lse.stride(0),
        0,  # window_left: unused
        0,  # window_right
        seq_len_q,
        seq_len_k,
        nheads_k,
        batch,
        torch.cuda.current_stream(),
    )
    return out, lse
