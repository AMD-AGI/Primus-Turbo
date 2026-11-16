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

"""gfx1250 MLA attention forward with scheduled tile regions (DeepSeek-V3 head dims qk 192 / v 128).

A workgroup is 4 waves (one per SIMD, one workgroup per CU: LDS_B bytes of LDS) running q blocks of 256 rows of one
(batch, head); wave w owns four 16-row q tiles, q tile i = block rows 64 i + 16 w .. + 15. A block's kv range is n
tiles of 64 rows: tiles [0, ncc) need no mask; the rest (seqlen_q == seqlen_kv: at most 4) are the diagonal tiles
ncc + k, in which q tiles i < k are fully masked (skipped), q tile k takes the causal mask and q tiles i > k none.

The per-tile math is generated code (fwd_sched): straight-line WMMA / LDS / softmax schedules on a static register
plan of 1024 VGPRs, as inline asm statements (regions INIT, INIT_C0, TILE_CL, TILE_D0..2, TILE_D3F, FINAL_A) whose
state lives in v[fwd_sched.BASE:1024]. This file is the rest: the schedule of q blocks, the TDM ring, the barriers,
the loop control and the output stores. Its own code must stay below BASE VGPRs between the regions of a block: a
region clobbers v[BASE:256), so LLVM keeps nothing there across one, and between two regions it takes the lowest
free VGPRs. This code needs v0..v38 in the build that was validated, so after any change here the final ISA must be
checked: outside the regions only the warm-up loads and the never-taken reserve block may name a VGPR >= BASE (the
numeric tests fail if another instruction does).

LDS: a 4-stage ring of kv tiles (a stage holds K 64 x 400 B and V 64 x 272 B), the Q block (256 x 400 B) and the
epilogue staging. Ring slot t of a block is its tile t, in stage (sb + t) % 4; the ring runs on across the blocks of
a workgroup. A slot takes 3 TDM ops (K columns 0..127 and 128..191, V), one each from waves 0..2; wave 3 issues a
NULL descriptor in their place, so every wave runs the same code and waits on static counts. Per block:
  prologue  the first block only: Q (waves 1, 2) and slots 0..2; B_0: every wave s_wait_tensorcnt 0, then the
            barrier
  tile t    issue slot t + 3 into the stage of tile t - 1; read the K / V fragments (the Q fragments stay in VGPRs
            from INIT on), then s_wait_dscnt 0, s_wait_tensorcnt 2 (slot t + 1 landed) and s_barrier_signal, the
            tile's math around them; s_barrier_wait. This barrier, B_{t+1}, makes tile t + 1 visible to every wave
            and frees the stage of tile t. Slots n..n + 2 past the last tile n - 1 load the next block's tiles 0..2
            (its ring slots 0..2: the same stages) when there is one, else 1-row copies of tile n - 1, so a wave
            issues one op in every tile.
  epilogue  after B_n: the next block's Q, then the output (staged at TAIL and in tile n - 1's stage)
No LDS read follows a tile's s_barrier_signal: right after B_n the next block's Q overwrites the Q area.

Grid: one q block per workgroup (npw == 0), or npw persistent workgroups on a static schedule: workgroup w runs
(q block, head-batch) pairs, x = w % 8, w / 8 = r nqb + i, R = npw / 8 / nqb: its step s is head-batch
hb = 8 (R s + r) + x, q block i on even steps and nqb - 1 - i on odd ones; the list ends at the first hb >= H B.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx

# UNSTABLE(gfx1250): the raw TDM load, the split barrier, MMRA fences, inline asm and disable_xdl_arb_stall have no
# stable API in FlyDSL 0.3.4.1.
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl.expr import range_constexpr, rocdl
from flydsl.expr.rocdl import tdm_ops

from primus_turbo.flydsl.utils.buffer_ops import buffer_store, create_buffer_resource_from_addr
from primus_turbo.flydsl.utils.prims import LOG2E

from . import SOURCE_DIGEST, fwd_sched
from .common import inline_region, lds_ptr, reserve_vgprs

D_QK, D_V = 192, 128
BM = 256  # q rows per block
QT = 4  # 16-row q tiles per wave
NB = 64  # kv rows per tile (ring stage)
NWAVES = 4
# The prologue fills ring slots 0..2 with the first block's own tiles, without looking at a next block: a workgroup
# that runs more than one block (the persistent grid: seqlen_q == seqlen_kv in whole q blocks, fwd.launch_grid)
# needs 3 tiles in every block, and has BM / NB of them at least.
assert BM // NB >= 3
XK, XV = fwd_sched.XK, fwd_sched.XV  # LDS row strides of K / Q and V (bytes)
K_B = NB * XK
V_B = NB * XV
STG_B = K_B + V_B  # a ring stage, K then V
NSTG = 4
Q_OFF = NSTG * STG_B  # the Q block
TAIL = Q_OFF + BM * XK  # the epilogue staging of q tiles 2, 3
# epilogue staging: SQT bytes per wave and q tile, O row r at XO r (16 rows), the lse of lane l at LSE_SLOT + 4 l;
# q tiles 0, 1 of the 4 waves in the ring stage of the block's last tile, q tiles 2, 3 at TAIL
XO = 272
LSE_SLOT = 16 * XO
SQT = LSE_SLOT + 32 * 4
LDS_B = TAIL + 2 * 4 * SQT
assert STG_B == fwd_sched.STG_B and SQT == fwd_sched.SQT_B
assert 2 * 4 * SQT <= STG_B and LDS_B <= 327680 and 2 * LDS_B > 327680
OOB = 0x80000000  # > every num_records (< 2^31) and + soffset < 2^32: dropped, no wrap
W0_A = 0x47510000  # TDM group 1 word 0: 128 columns, pad 72 elements -> 400-B rows
W0_B = 0x87110000  # 64 columns, pad 136 elements -> 400-B rows
W0_V = 0x07510000  # 128 columns, pad 8 elements -> 272-B rows
# One block per workgroup: KV_GROUP q blocks of KV_HEAD_CHUNK heads are dispatched together (powers of two)
KV_GROUP = 16
KV_HEAD_CHUNK = 16
RESERVE_V4 = 254  # common.reserve_vgprs: 16-byte values kept live (1016 VGPRs, plus the address)
# Code warm-up (_warm_code): the regions a block runs, 2 KB of code per load (32 lanes x one 64-byte line each), and
# this many 2 KB pieces of the kernel's own code before INIT_C0
WARM_REGIONS = ("INIT_C0", "TILE_CL", "TILE_D0", "TILE_D1", "TILE_D2", "TILE_D3F")
WARM_PIECE = 2048
WARM_OWN_PIECES = 2


def _w(x):
    """A 32-bit word as an i32 constant."""
    return fx.Int32(x - (1 << 32) if x >= 1 << 31 else x)


def _ir(v):
    return fx.as_ir_value(v)


def _sel(c, a, b):
    return c.select(a, b)


def _min(a, b):
    return _sel(a < b, a, b)


def _lds_fence(release):
    """Workgroup fence on LDS only (MMRA amdgpu-synchronize-as = local): the release waits for this wave's DS ops,
    not for its in-flight global stores."""
    op = llvm.FenceOp(
        llvm.AtomicOrdering.release if release else llvm.AtomicOrdering.acquire, syncscope="workgroup"
    )
    op.operation.attributes["llvm.mmra"] = ir.Attribute.parse(
        '#llvm.mmra_tag<"amdgpu-synchronize-as":"local">'
    )


def _bar_signal():
    """Retire this wave's LDS reads, release (workgroup, LDS), arrive at the barrier (no wait)."""
    rocdl.sched_barrier(0)
    rocdl.s_wait_dscnt(0)
    _lds_fence(True)
    rocdl.s_barrier_signal(-1)
    rocdl.sched_barrier(0)


def _bar_wait():
    """Wait at the workgroup barrier, acquire (workgroup, LDS)."""
    rocdl.sched_barrier(0)
    rocdl.s_barrier_wait(-1)
    _lds_fence(False)
    rocdl.sched_barrier(0)


def _wave_lds_order():
    """Cross-lane LDS data within one wave: its LDS ops execute in order; keep the compiler from moving loads above
    stores."""
    rocdl.sched_barrier(0)
    llvm.FenceOp(llvm.AtomicOrdering.seq_cst, syncscope="wavefront")
    rocdl.sched_barrier(0)


def _region(name, *operands):
    inline_region(fwd_sched, name, *operands)


def _tdm_g1(w0, tdim1, tile0, tile1, stride_el):
    """TDM group 1 of a 2-D tile load: w0 (element size and LDS row padding), dim 0 unbounded, tdim1 rows in bounds,
    a tile of tile0 columns (an int, or an Int32 already shifted to bits 16..31) x tile1 rows, the row stride
    stride_el (elements, Int64)."""
    tdim1 = fx.Int32(tdim1)
    return fx.Vector.from_elements(
        [
            _w(w0) if isinstance(w0, int) else w0,
            _w(0xFFFF0000),
            ((tdim1 & fx.Int32(0xFFFF)) << fx.Int32(16)) | fx.Int32(0x7FFF),
            (tdim1.shrui(fx.Int32(16)) & fx.Int32(0xFFFF))
            | (fx.Int32(tile0 << 16) if isinstance(tile0, int) else tile0),
            fx.Int32(tile1),
            fx.Int32(stride_el),
            fx.Int32(stride_el >> fx.Int64(32)) & fx.Int32(0xFFFF),
            fx.Int32(0),
        ],
        fx.Int32,
    )


def _tdm_g0(lds, ga, count):
    """TDM group 0: the LDS and global addresses; count 0 = the NULL descriptor (moves nothing)."""
    return fx.Vector.from_elements(
        [count, fx.Int32(lds), fx.Int32(ga), fx.Int32(ga >> fx.Int64(32)) | _w(0x80000000)], fx.Int32
    )


def _tdm_load(g0, g1):
    z4 = _ir(fx.Vector.filled(4, 0, fx.Int32))
    z8 = _ir(fx.Vector.filled(8, 0, fx.Int32))
    rocdl.tensor_load_to_lds(_ir(g0), _ir(g1), z4, z4, z8, cache_policy=0)


class _Problem:
    """The kernel arguments and the per-block helpers. ``z`` is an opaque 0 (derived from the block counter): every
    value is recomputed per block from ``arg + z``, so LLVM hoists nothing argument-derived out of the block loop and
    the kernel's own state fits in the SGPRs the regions leave it."""

    def __init__(self, a, z):
        self.a = {k: (v + z if isinstance(v, fx.Int32) else v) for k, v in a.items()}
        z64 = fx.Int64(z)
        a = self.a
        self.z = z
        self.cshift = a["Skv"] - a["Sq"]
        self.krow = fx.Int64(a["H"]) * fx.Int64(D_QK)  # row strides in elements
        self.vrow = fx.Int64(a["H"]) * fx.Int64(D_V)
        self.qaddr = fx.Int64(fx.ptrtoint(fx.get_iter(a["Q"]))) + z64
        self.kaddr = fx.Int64(fx.ptrtoint(fx.get_iter(a["K"]))) + z64
        self.vaddr = fx.Int64(fx.ptrtoint(fx.get_iter(a["V"]))) + z64
        self.oaddr = fx.Int64(fx.ptrtoint(fx.get_iter(a["O"]))) + z64
        self.laddr = fx.Int64(fx.ptrtoint(fx.get_iter(a["LSE"]))) + z64

    def lane(self):
        """The lane id, from v_mbcnt with the opaque addend: neither hoisted nor CSEd across blocks."""
        return fx.Int32(rocdl.mbcnt_lo(fx.Int32.ir_type, _ir(fx.Int32(-1)), _ir(self.z)))

    def ntiles(self, qb):
        Sq = self.a["Sq"]
        return (_min(qb * fx.Int32(BM) + fx.Int32(BM), Sq) + self.cshift + fx.Int32(NB - 1)).shrui(
            fx.Int32(6)
        )

    def kv_base(self, hh, bb):
        a = self.a
        row0 = fx.Int64(bb) * fx.Int64(a["Skv"]) * fx.Int64(a["H"]) + fx.Int64(hh)
        return self.kaddr + row0 * fx.Int64(2 * D_QK), self.vaddr + row0 * fx.Int64(2 * D_V)

    def slot_ops(self, wv, gk, gv):
        """This wave's TDM op of every ring slot (a slot's 3 ops split over waves 0..2: K columns 0..127, K columns
        128..191, V; wave 3 a NULL op): (count, LDS offset in a stage, global base of the block's tile 0, bytes per kv
        row, group-1 word 0, tile columns)."""
        is0, is1 = wv == fx.Int32(0), wv == fx.Int32(1)
        isv = wv == fx.Int32(2)
        cnt = _sel(wv < fx.Int32(3), fx.Int32(1), fx.Int32(0))
        lds_off = _sel(is0, fx.Int32(0), _sel(is1, fx.Int32(256), fx.Int32(K_B)))
        base = _sel(isv, gv, gk + _sel(is1, fx.Int64(256), fx.Int64(0)))
        stride = _sel(isv, self.vrow, self.krow)
        w0 = _sel(is0, _w(W0_A), _sel(is1, _w(W0_B), _w(W0_V)))
        tile0 = _sel(is1, fx.Int32(64 << 16), fx.Int32(128 << 16))
        return cnt, lds_off, base, stride, w0, tile0

    def issue_slot(self, lds0, op, t, n, sb, nx=None, nn=None, opn=None):
        """This wave's TDM op of ring slot t of a block (stage (sb + t) % 4; op = slot_ops of the block): its tile t,
        or past its last tile a 1-row copy of tile n - 1 into that (free) stage; with a next block (nx: nn tiles, its
        slot_ops opn) slot t >= n loads its tile t - n (1-row copies of its tile nn - 1 past it)."""
        cnt, lds_off, base, stride, w0, tile0 = op
        if nx is None:
            tt = _min(t, n - fx.Int32(1))
            real = t < n
        else:
            # t < n as the sign of t - n: next to the subtraction, a compare of t and n becomes a VALU
            # subtract-with-borrow and a v_readfirstlane in every tile's shell (~34 cycles)
            past = t - n
            before = past.shrui(fx.Int32(31)) != fx.Int32(0)
            own = before | (nx == fx.Int32(0))
            tt = _sel(own, _min(t, n - fx.Int32(1)), _min(past, nn - fx.Int32(1)))
            real = _sel(own, before, past < nn)
            base = _sel(own, base, opn[2])
        ga = base + fx.Int64(tt * fx.Int32(2 * NB)) * stride
        th = _sel(real, fx.Int32(NB), fx.Int32(1))
        st = lds0 + ((sb + t) & fx.Int32(NSTG - 1)) * fx.Int32(STG_B) + lds_off
        _tdm_load(_tdm_g0(st, ga, cnt), _tdm_g1(w0, NB, tile0, th, stride))

    def issue_q(self, lds0, wv, qb, hh, bb, on=None):
        """The block's Q: columns 0..127 by wave 1, 128..191 by wave 2 (NULL ops on the other waves, and on every
        wave unless on: two TDM ops per wave either way)."""
        a = self.a
        q0 = qb * fx.Int32(BM)
        row = (fx.Int64(bb) * fx.Int64(a["Sq"]) + fx.Int64(q0)) * fx.Int64(a["H"]) + fx.Int64(hh)
        gq = self.qaddr + row * fx.Int64(2 * D_QK)
        rows = a["Sq"] - q0
        w1, w2 = wv == fx.Int32(1), wv == fx.Int32(2)
        if on is not None:
            w1, w2 = w1 & on, w2 & on
        _tdm_load(
            _tdm_g0(lds0 + fx.Int32(Q_OFF), gq, _sel(w1, fx.Int32(1), fx.Int32(0))),
            _tdm_g1(W0_A, rows, 128, BM, self.krow),
        )
        _tdm_load(
            _tdm_g0(lds0 + fx.Int32(Q_OFF + 256), gq + fx.Int64(256), _sel(w2, fx.Int32(1), fx.Int32(0))),
            _tdm_g1(W0_B, rows, 64, BM, self.krow),
        )


def _remap(lin, gx, gy, gz):
    """(q block, head, batch) of dispatch rank ``lin`` after the longest-first remap.

    Causal work grows with the q block index, so the linear order is remapped to start the last q
    block of every (head, batch) first. When the grid divides evenly, windows of KV_GROUP ranks go
    out KV_HEAD_CHUNK heads at a time so a head's K/V is shared in L2.
    """
    gyz = gy * gz
    rank = lin // gyz
    rem = lin - rank * gyz
    lg, lhb = KV_GROUP.bit_length() - 1, KV_HEAD_CHUNK.bit_length() - 1
    r = (rank & fx.Int32(KV_GROUP - 1)) * gyz + rem
    c = r & fx.Int32(KV_GROUP * KV_HEAD_CHUNK - 1)
    grouped = ((gx & fx.Int32(KV_GROUP - 1)) | (gyz & fx.Int32(KV_HEAD_CHUNK - 1))) == fx.Int32(0)
    lin = grouped.select(
        ((rank & fx.Int32(-KV_GROUP)) + (c >> fx.Int32(lhb))) * gyz
        + ((r >> fx.Int32(lg + lhb)) << fx.Int32(lhb))
        + (c & fx.Int32(KV_HEAD_CHUNK - 1)),
        lin,
    )
    rank = lin // gyz
    rem = lin - rank * gyz
    return gx - fx.Int32(1) - rank, rem % gy, rem // gy


class _Sched:
    """The workgroup's blocks (q block, head, batch). Persistent (npw > 0): workgroup w runs step s as head-batch
    hb = 8 (R s + r) + w % 8 with r = (w / 8) / nqb, R = npw / 8 / nqb, q block i = (w / 8) % nqb on even steps and
    nqb - 1 - i on odd ones, while hb < H B; the divisions run once per workgroup, a step advances (head, batch) by
    8 R heads without one. Else one block per workgroup in the hardware dispatch order (_remap)."""

    def __init__(self, a):
        npw, nqb, H, B = a["npw"], a["nqb"], a["H"], a["B"]
        wg = fx.Int32(fx.block_idx.x)
        self.persist = npw > fx.Int32(0)
        nq = _sel(nqb > fx.Int32(0), nqb, fx.Int32(1))
        w8 = wg.shrui(fx.Int32(3))
        pr = w8 // nq
        self.pq = w8 - pr * nq
        p_r = npw.shrui(fx.Int32(3)) // nq
        self.p_r, self.pr = p_r, pr
        step = fx.Int32(8) * p_r
        self.st_b = step // H
        self.st_h = step - self.st_b * H
        self.n_hx = (H * B).shrui(fx.Int32(3))
        self.nqb, self.H = nqb, H
        hb = fx.Int32(8) * pr + (wg & fx.Int32(7))
        hb = _sel(pr < p_r, hb, fx.Int32(0))
        bb_p = hb // H
        hh_p = hb - bb_p * H
        lin = wg + nqb * (fx.Int32(fx.block_idx.y) + H * fx.Int32(fx.block_idx.z))
        qb_n, hh_n, bb_n = _remap(lin, nqb, H, B)
        p = self.persist
        self.first = (
            _sel(p, (pr < p_r) & (pr < self.n_hx), fx.Int32(1) == fx.Int32(1)),
            _sel(p, self.pq, qb_n),
            _sel(p, hh_p, hh_n),
            _sel(p, bb_p, bb_n),
        )

    def next(self, s, hh, bb):
        """(ok, q block, head, batch) of step s + 1 after step s at (head hh, batch bb)."""
        s1 = s + fx.Int32(1)
        ok = self.persist & (self.p_r * s1 + self.pr < self.n_hx)
        h2 = hh + self.st_h
        wrap = h2 >= self.H
        hh2 = _sel(wrap, h2 - self.H, h2)
        bb2 = bb + self.st_b + _sel(wrap, fx.Int32(1), fx.Int32(0))
        qb2 = _sel((s1 & fx.Int32(1)) == fx.Int32(1), self.nqb - fx.Int32(1) - self.pq, self.pq)
        return ok, qb2, hh2, bb2


def _warm_code(wv):
    """The kernel's code into L2 through the data path, at the kernel start, by wave 0 of every workgroup.

    When another kernel ran since this one's last launch (every call in training), its code is neither in the CUs'
    instruction caches nor in L2, and each line's first fetch comes from HBM behind the kernel's own K / V stream
    (thread trace: ~770 cycles per 256 bytes of INIT_C0; s_prefetch_inst and global_prefetch_b8 change nothing
    there). Loaded as data it is in L2 before the first block starts (the workgroup waits for its first TDM loads
    meanwhile), and the fetches hit L2. One global_load_b128 per WARM_PIECE bytes, its 32 lanes 64 bytes apart,
    counted back from each region's end label over the region's size bound (so never past the kernel's code: before
    a region is the kernel's own code), plus WARM_OWN_PIECES before INIT_C0. The loads' data goes to v[BASE:], which
    no code uses before the first region; _warm_code_landed() ends them.

    The pieces before INIT_C0 take that much of the kernel's own code to stand before it: 4.6 KB in the build that
    was validated, and the code object holds another 2 KB before the function. With a compiler that lays the kernel
    out otherwise, check that INIT_C0 still starts that far into the code object (a load before it is a fault)."""
    pieces = []
    for name in WARM_REGIONS:
        end = f"{fwd_sched.region_label(name)}_end"
        pieces += [
            (end, -WARM_PIECE * k) for k in range(1, -(-fwd_sched.region_bytes(name) // WARM_PIECE) + 1)
        ]
    first = fwd_sched.region_label(WARM_REGIONS[0])
    pieces += [(first, -WARM_PIECE * k) for k in range(1, WARM_OWN_PIECES + 1)]
    assert fwd_sched.BASE + 4 * len(pieces) <= 256
    lines, cur = ["s_getpc_b64 s[90:91]", "fwd_warm_pc:"], None
    for i, (label, off) in enumerate(pieces):
        if label != cur:
            # s[92:93]: the label's address (the loads in flight have read the previous one). A signed 64-bit add:
            # the compiler may place this block behind the regions (a negative label distance)
            lines += ["s_wait_xcnt 0x0"] if cur else []
            lines += [f"s_mov_b32 s94, {label}-fwd_warm_pc", "s_ashr_i32 s95, s94, 31"]
            lines.append("s_add_nc_u64 s[92:93], s[90:91], s[94:95]")
            cur = label
        r = fwd_sched.BASE + 4 * i
        lines.append(f"global_load_b128 v[{r}:{r + 3}], $0, s[92:93] offset:{off}")
    lines.append("s_wait_xcnt 0x0")
    clobbers = [f"~{{v{r}}}" for r in range(fwd_sched.BASE, fwd_sched.BASE + 4 * len(pieces))]
    clobbers += [f"~{{s{r}}}" for r in range(90, 96)] + ["~{scc}"]
    lane64 = (fx.Int32(fx.thread_idx.x) & fx.Int32(31)) * fx.Int32(64)

    @flyc.jit
    def warm(cond):
        if cond:
            llvm.inline_asm(
                res=None,
                operands_=[_ir(lane64)],
                asm_string="\n".join(lines),
                constraints=",".join(["v"] + clobbers),
                has_side_effects=True,
            )

    warm(wv == fx.Int32(0))


def _warm_code_landed():
    """_warm_code's loads have written their VGPRs: before the first region, and before the wave ends."""
    llvm.inline_asm(
        res=None, operands_=[], asm_string="s_wait_loadcnt 0x0", constraints="", has_side_effects=True
    )


def _attention(a):
    pb = _Problem(a, fx.Int32(0))
    lds0 = fx.Int32(fx.ptrtoint(fx.SharedAllocator().allocate(LDS_B).peek().ptr))
    rocdl.disable_xdl_arb_stall()
    wv = fx.Int32(rocdl.wave_id())
    _warm_code(wv)
    c2 = a["scale"] * fx.Float32(LOG2E)

    sc = _Sched(a)

    @flyc.jit
    def run():
        ok0, qb0, hh0, bb0 = sc.first
        if ok0:
            # the first block's prologue loads: Q (waves 1, 2), then its tiles 0..2 = ring slots 0..2 (waves 0..2;
            # past its last tile copies of that one, never a next block's tiles: the assertion on BM / NB above)
            pb.issue_q(lds0, wv, qb0, hh0, bb0)
            n0 = pb.ntiles(qb0)
            op0 = pb.slot_ops(wv, *pb.kv_base(hh0, bb0))
            for t in range_constexpr(3):
                pb.issue_slot(lds0, op0, fx.Int32(t), n0, fx.Int32(0))
            s = fx.Int32(0)
            sb = fx.Int32(0)
            more = fx.Int32(1)
            qb, hh, bb = qb0, hh0, bb0
            _warm_code_landed()
            while more != fx.Int32(0):
                # an opaque 0 from the block counter: the block's values are not hoisted out of the block loop
                z = s.shrui(fx.Int32(30))
                wvz = wv + z
                more, n, qb, hh, bb = _block(_Problem(a, z), sc, lds0 + z, wvz, c2, s, sb, qb, hh, bb)
                s = s + fx.Int32(1)
                sb = (sb + n) & fx.Int32(NSTG - 1)
            # past the last block: no TDM write (the 1-row copies) may land after the workgroup retires
            tdm_ops.tensor_wait(0)

    run()
    _warm_code_landed()  # (a workgroup without a block)
    reserve_vgprs(a["never"] != fx.Int32(0), lds0, RESERVE_V4)


def _block(pb, sc, lds0, wv, c2, s, sb, qb, hh, bb):
    """Block s (q block qb, head hh, batch bb) of the workgroup, its tile 0 in ring stage sb. Returns (1 if a next block
    follows else 0, this block's number of kv tiles, the next block's q block, head, batch)."""
    q0 = qb * fx.Int32(BM)
    n = pb.ntiles(qb)
    ncc = _min((q0 + pb.cshift + fx.Int32(1)).shrui(fx.Int32(6)), n)  # tiles [0, ncc) need no mask
    op = pb.slot_ops(wv, *pb.kv_base(hh, bb))
    lane = pb.lane()
    # B_0: Q and tiles 0..2 landed and visible; every wave's reads of the previous block's staging retired
    tdm_ops.tensor_wait(0)
    _bar_signal()
    _bar_wait()
    # per-lane LDS bases: Q B operands (q tiles 0, 1 from qa, 2, 3 from qb2, + 64 rows for q tiles 1, 3), K A
    # operands, V^T A operands (ds_load_tr16_b128)
    l15 = lane & fx.Int32(15)
    l4 = lane.shrui(fx.Int32(4))
    qa = lds0 + fx.Int32(Q_OFF) + (fx.Int32(16) * wv + l15) * fx.Int32(XK) + l4 * fx.Int32(16)
    qb2 = qa + fx.Int32(2 * 64 * XK)
    k_lane = l15 * fx.Int32(XK) + l4 * fx.Int32(16)
    v_lane = (l4 * fx.Int32(8) + (lane & fx.Int32(7))) * fx.Int32(XV) + (
        lane.shrui(fx.Int32(3)) & fx.Int32(1)
    ) * fx.Int32(16)
    # causal: element e of kv16 tile j of tile t (kv = 64 t + 16 j + 8 (lane / 16) + e) is masked for q tile i iff
    # 16 j + e > dl0 - 64 t + 64 i
    dl0 = q0 + fx.Int32(16) * wv + l15 + pb.cshift - l4 * fx.Int32(8)

    # Tiles [0, ncc). Tile t issues this wave's op of ring slot t + 3 into the stage of tile t - 1 (freed by barrier
    # B_t): the block's tile min(t + 3, n - 1) with all its 64 rows (past tile n - 1 a copy of it into the free stage,
    # never read), so group 1 is the same for every slot. Tile 0 runs with INIT in one region (INIT_C0: no previous
    # tile, so no prefix), tiles [1, ncc) as one generated loop (TILE_CL).
    cnt, lds_off, base, stride, w0, tile0 = op
    step = fx.Int64(2 * NB) * stride
    g1 = _tdm_g1(w0, NB, tile0, NB, stride)

    def slot_quad(t):
        """Group 0 of this wave's op of ring slot t + 3 (tile t's) for the tiles [0, ncc)."""
        lds = lds0 + ((sb + t + fx.Int32(3)) & fx.Int32(NSTG - 1)) * fx.Int32(STG_B) + lds_off
        return lds, (base + fx.Int64(_min(t + fx.Int32(3), n - fx.Int32(1))) * step) | fx.Int64(-(1 << 63))

    @flyc.jit
    def tiles_c(cond):
        if cond:
            st0 = lds0 + (sb & fx.Int32(NSTG - 1)) * fx.Int32(STG_B)
            lds3, addr3 = slot_quad(fx.Int32(0))
            q3 = fx.Vector.from_elements(
                [cnt, lds3, fx.Int32(addr3), fx.Int32(addr3 >> fx.Int64(32))], fx.Int32
            )
            _region("INIT_C0", qa, qb2, st0 + k_lane, st0 + fx.Int32(K_B) + v_lane, c2, q3, g1)
            _bar_wait()  # B_1

            @flyc.jit
            def loop(cond2):
                if cond2:
                    # the loop counts from 0: tile 1 + t, slot advance while 1 + t < n - 4
                    stage1 = lds0 + ((sb + fx.Int32(1)) & fx.Int32(NSTG - 1)) * fx.Int32(STG_B)
                    lds4, addr4 = slot_quad(fx.Int32(1))
                    _region(
                        "TILE_CL",
                        k_lane,
                        v_lane + fx.Int32(K_B),
                        c2,
                        stage1,
                        cnt,
                        lds4,
                        lds_off,
                        lds0 + fx.Int32(NSTG * STG_B),
                        ncc - fx.Int32(1),
                        n - fx.Int32(5),
                        addr4,
                        step,
                        g1,
                    )

            loop(ncc > fx.Int32(1))
        else:
            _region("INIT", qa, qb2, c2)

    tiles_c(ncc > fx.Int32(0))

    # the diagonal tiles ncc + k, k < 4 (seqlen_q == seqlen_kv); their slots n..n + 2 load the next block's tiles
    # 0..2 (the next block is known here on the static schedule)
    nx, qbx, hhx, bbx = sc.next(s, hh, bb)
    nnx = pb.ntiles(qbx)
    opx = pb.slot_ops(wv, *pb.kv_base(hhx, bbx))
    nxi = _sel(nx, fx.Int32(1), fx.Int32(0))
    for k in range_constexpr(3):

        @flyc.jit
        def diag(cond):
            if cond:
                t2 = ncc + fx.Int32(k)  # noqa: B023 (diag runs within its iteration)
                pb.issue_slot(lds0, op, t2 + fx.Int32(3), n, sb, nxi, nnx, opx)
                st = lds0 + ((sb + t2) & fx.Int32(NSTG - 1)) * fx.Int32(STG_B)
                _region(f"TILE_D{k}", st + k_lane, st + fx.Int32(K_B) + v_lane, dl0 - fx.Int32(NB) * t2, c2)  # noqa: B023
                _bar_wait()  # B_{t2+1}

        diag(ncc + fx.Int32(k) < n)

    stg = _staging(pb, lds0, wv, sb, n)

    # the last diagonal tile with FINAL (TILE_D3F), else FINAL alone after the block's last tile. The next block's Q
    # goes out before them (every wave's reads of the Q area retired at B_1; two TDM ops per wave, NULL ops past the
    # workgroup's last block, which TILE_D3F's tensorcnt wait leaves in flight)
    @flyc.jit
    def last(cond):
        if cond:
            t2 = ncc + fx.Int32(3)
            pb.issue_slot(lds0, op, t2 + fx.Int32(3), n, sb, nxi, nnx, opx)
            pb.issue_q(lds0, wv, qbx, hhx, bbx, on=nx)
            st = lds0 + ((sb + t2) & fx.Int32(NSTG - 1)) * fx.Int32(STG_B)
            _region(
                "TILE_D3F", st + k_lane, st + fx.Int32(K_B) + v_lane, dl0 - fx.Int32(NB) * t2, c2, *stg[2:]
            )
        else:
            pb.issue_q(lds0, wv, qbx, hhx, bbx, on=nx)
            _region("FINAL_A", *stg[2:])

    last(ncc + fx.Int32(3) < n)
    rocdl.sched_barrier(0)
    _store_out(pb, stg, wv, q0, hh, bb)
    return nxi, n, qbx, hhx, bbx


def _staging(pb, lds0, wv, sb, n):
    """This wave's staging slots of out / lse (q tiles 0, 1 in the stage of tile n - 1, q tiles 2, 3 at TAIL; O row r
    at XO r, the lse of lane l at LSE_SLOT + 4 l): (slot base a, slot base b, FINAL's operands wa, wb, la, lb)."""
    lane = pb.lane()
    sga = lds0 + ((sb + n - fx.Int32(1)) & fx.Int32(NSTG - 1)) * fx.Int32(STG_B) + wv * fx.Int32(SQT)
    sgb = lds0 + fx.Int32(TAIL) + wv * fx.Int32(SQT)
    wl = (lane & fx.Int32(15)) * fx.Int32(XO) + lane.shrui(fx.Int32(4)) * fx.Int32(16)
    wls = fx.Int32(LSE_SLOT) + fx.Int32(4) * lane
    return sga, sgb, sga + wl, sgb + wl, sga + wls, sgb + wls


def _store_out(pb, stg, wv, q0, hh, bb):
    """Copies this wave's staged out / lse (stg: _staging) to global memory: 8 row-pair buffer stores per q tile and
    one lse store."""
    _wave_lds_order()
    a = pb.a
    Sq, H = a["Sq"], a["H"]
    sga, sgb = stg[0], stg[1]
    # O, q tile i: a buffer resource based at its first row q0 + 64 i (num_records 0 past Sq: the 64 rows are all in
    # or all past Sq), soffset 2 j * row bytes (store j: rows 2 j, 2 j + 1), voffset the lane's row 16 wv + l / 16 and
    # 16-B column l % 16; lse: one 8-byte store of two rows per lane for the wave's 64 rows
    rsb = H * fx.Int32(D_V * 2)  # bytes per q row
    o_bytes = fx.Int64(a["B"]) * fx.Int64(Sq) * fx.Int64(H) * fx.Int64(D_V * 2)
    off0 = ((fx.Int64(bb) * fx.Int64(Sq) + fx.Int64(q0)) * fx.Int64(H) + fx.Int64(hh)) * fx.Int64(D_V * 2)
    ln = pb.lane()
    hl = ln.shrui(fx.Int32(4))
    rl = ln & fx.Int32(15)
    vo_l = (fx.Int32(16) * wv + hl) * rsb + rl * fx.Int32(16)
    rda = sga + hl * fx.Int32(XO) + rl * fx.Int32(16)
    rdb = sgb + hl * fx.Int32(XO) + rl * fx.Int32(16)
    v4 = fx.Vector.make_type(4, fx.Int32)
    v2 = fx.Vector.make_type(2, fx.Int32)
    lt = ln.shrui(fx.Int32(3))
    lk = ln & fx.Int32(7)
    lsrc = (
        _sel(lt < fx.Int32(2), sga, sgb)
        + (lt & fx.Int32(1)) * fx.Int32(4 * SQT)
        + fx.Int32(LSE_SLOT)
        + fx.Int32(8) * lk
    )
    lsev = llvm.load(v2, lds_ptr(lsrc))
    lq = q0 + fx.Int32(16) * wv + fx.Int32(64) * lt + fx.Int32(2) * lk
    la = ((bb * H + hh) * Sq + lq) * fx.Int32(4)
    lo = _sel(q0 + fx.Int32(64) * lt < Sq, la, _w(OOB))
    rocdl.sched_barrier(0)
    for i in range(QT):  # the q tile's 8 reads into registers of their own, then its stores
        ok = (q0 + fx.Int32(64 * i)) < Sq
        offi = off0 + fx.Int64(64 * i) * fx.Int64(rsb)
        base = pb.oaddr + _sel(ok, offi, fx.Int64(0))
        nrec = _sel(ok, fx.Int32(o_bytes - offi), fx.Int32(0))
        r_o = create_buffer_resource_from_addr(_ir(base), num_records_bytes=_ir(nrec))
        src = rda if i < 2 else rdb
        ws = [llvm.load(v4, lds_ptr(src + fx.Int32((i & 1) * 4 * SQT + 2 * j * XO))) for j in range(8)]
        for j, w in enumerate(ws):
            buffer_store(w, r_o, vo_l, soffset_bytes=fx.Int32(2 * j) * rsb, offset_is_bytes=True)
        rocdl.sched_barrier(0)
    # B H Sq 4 < 2^31 (the host's check)
    r_l = create_buffer_resource_from_addr(
        _ir(pb.laddr), num_records_bytes=_ir(a["B"] * H * Sq * fx.Int32(4))
    )
    buffer_store(lsev, r_l, lo, offset_is_bytes=True)
    rocdl.sched_barrier(0)


@flyc.kernel(name="mla_fwd_kernel", known_block_size=[NWAVES * 32, 1, 1])
def mla_fwd_kernel(
    O: fx.Tensor,
    Q: fx.Tensor,
    K: fx.Tensor,
    V: fx.Tensor,
    LSE: fx.Tensor,
    scale: fx.Float32,
    Sq: fx.Int32,
    Skv: fx.Int32,
    H: fx.Int32,
    B: fx.Int32,
    nqb: fx.Int32,
    npw: fx.Int32,
    never: fx.Int32,
):
    """Grid (nqb, H, B), or (npw, 1, 1) persistent (npw > 0). ``never`` is 0 (common.reserve_vgprs)."""
    _attention(
        dict(
            O=O, Q=Q, K=K, V=V, LSE=LSE, scale=scale, Sq=Sq, Skv=Skv, H=H, B=B, nqb=nqb, npw=npw, never=never
        )
    )


@flyc.jit
def launch_mla_fwd(
    O: fx.Tensor,
    Q: fx.Tensor,
    K: fx.Tensor,
    V: fx.Tensor,
    LSE: fx.Tensor,
    scale: fx.Float32,
    Sq: fx.Int32,
    Skv: fx.Int32,
    H: fx.Int32,
    B: fx.Int32,
    nqb: fx.Int32,
    npw: fx.Int32,
    stream: fx.Stream,
):
    _ = SOURCE_DIGEST  # FlyDSL disk-cache key
    persist = npw > fx.Int32(0)
    gx = fx.Index(_sel(persist, npw, nqb))
    gy = fx.Index(_sel(persist, fx.Int32(1), H))
    gz = fx.Index(_sel(persist, fx.Int32(1), B))
    mla_fwd_kernel(O, Q, K, V, LSE, scale, Sq, Skv, H, B, nqb, npw, fx.Int32(0)).launch(
        grid=(gx, gy, gz), block=(NWAVES * 32, 1, 1), stream=stream
    )


# Expert scheduling mode: the generated regions carry their own dependency waits (fwd_sched / asm) and LLVM
# inserts those of the kernel's own code. One wave per SIMD (1024 VGPRs). amdgpu-prealloc-sgpr-spill-vgprs: the
# VGPR lanes LLVM spills SGPRs to are allocated with the kernel's own VGPRs (below fwd_sched.BASE), not among the
# generated code's.
launch_mla_fwd.compile_hints["llvm_options"] = {
    "amdgpu-expert-scheduling-mode": True,
    "amdgpu-prealloc-sgpr-spill-vgprs": True,
}
launch_mla_fwd.compile_hints["waves_per_eu"] = 1
