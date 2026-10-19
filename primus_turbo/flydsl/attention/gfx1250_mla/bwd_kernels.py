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

"""MLA flash-attention backward kernels for gfx1250 (wave32 WMMA, bf16).

    k_delta_bshd  delta = rowsum(dO * O), and k_dkdv's row constants NL = -LOG2E*lse, ND = -scale*delta
    k_dkdv64      dV = P^T dO, dK = dS^T Q; two waves own a 64-row kv block and share one Q/dO ring
    k_dkdv        the same with one wave per 32-row kv block (small grids)
    k_dqg96       dQ = dS K; two waves of 48 queries share one K/V ring
    k_dqg         the same with one wave of 32 queries (head tiles, small grids)

q/k/dq/dk are [B, S, H, D_QK] and v/o/do/dv [B, S, H, D_V] bf16; lse (natural log), delta, NL and
ND are [B, Hq, Sq] fp32; Hq = G * Hkv. Causal is bottom-right. Q/dO (k_dkdv*) and K/V (k_dqg*)
stream through 3-stage Tensor Data Mover LDS rings, and every output element is written once (no
atomics).
"""

import flydsl.compiler as flyc
import flydsl.expr as fx

# UNSTABLE(gfx1250): the rocdl / tdm_ops calls other than the buffer views and copies (WMMA, the TDM
# atom and tensor_wait, ds_load_tr16, s_wait_dscnt, the split barrier, sched barriers, readfirstlane,
# raw_ptr_buffer_load, wave_id, disable_xdl_arb_stall), the raw LLVM LDS loads/stores and the
# workgroup-only gpu.barrier attribute have no stable API in FlyDSL 0.3.4.1.
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl.expr import arith, const_expr, range_constexpr, rocdl
from flydsl.expr.rocdl import tdm_ops

from primus_turbo.flydsl.utils.prims import LOG2E

from . import SOURCE_DIGEST
from .fwd_staging import lds_ptr, pow2_segments

D_QK = 192  # q/k head dim
D_V = 128  # v/o head dim
DV8_QK = D_QK // 8  # 8-element vectors per q/k row
DV8_V = D_V // 8  # 8-element vectors per v/o row
NDT_QK = D_QK // 32  # WMMA k-steps of S = Q K^T
NDT_V = D_V // 32  # WMMA k-steps of dP = dO V^T
NDT_MAX = max(NDT_QK, NDT_V)
NDO_QK = D_QK // 16  # 16-wide d tiles of dK / dQ
NDO_V = D_V // 16  # 16-wide d tiles of dV
NDO_MAX = max(NDO_QK, NDO_V)
NKV = 2  # 16-row kv sub-tiles per k_dkdv wave
NST = NKV * (NDO_V + NDO_QK)  # k_dkdv accumulators: dV, then dK
WAVE = 32
NEG = -3.0e38
TDM_DEPTH = 3  # LDS ring stages
BLOCK_KV = 32  # kv rows per k_dkdv wave
# TDM-staged LDS rows are padded by 16 B, so the 16 rows of a ds_load_tr16 start in 16 distinct
# 4-bank groups instead of all on bank 0.
XK_ROW_B = D_QK * 2 + 16
XV_ROW_B = D_V * 2 + 16
# TDM descriptor fields FlyDSL does not range-check: pad interval log2(w/2 dwords) - 1 in 3 bits,
# pad amount (row - w)/2 dwords - 1 in 7 bits.
assert all(4 <= w <= 512 and 1 <= (d + 8 - w) // 2 <= 128 for d in (D_QK, D_V) for _, w in pow2_segments(d))

# k_dkdv ring stage: the dO tile [32][D_V] at +0 and the Q tile [32][D_QK] at +QOFF.
QOFF = 32 * XV_ROW_B
QDO_B = 32 * (XV_ROW_B + XK_ROW_B)
TDM_OPS_QDO = len(pow2_segments(D_V)) + len(pow2_segments(D_QK))  # TDM ops per stage
TW_QDO = TDM_OPS_QDO * (TDM_DEPTH - 2)  # tensorcnt that retires all but the newest stage
NQE = 2 * NDT_QK  # carried Q readback entries per 16-query half
NHB = NQE + 2 * NDT_V  # ... plus the dO entries
NLD = 8  # carried NL/ND vec4 loads per iteration
DK0 = NKV * NDO_V  # first dK accumulator
# k_dkdv epilogue: per wave a row-major dV image [32][D_V] at +0 and a dK image [32][D_QK] at
# +EPI_DK in the dead ring, written back as whole 128-B lines.
EPI_DK = 32 * XV_ROW_B
EPI_W = 32 * XV_ROW_B + 32 * XK_ROW_B
EPI_LINE_B = 128
# k_dkdv64: wave w owns kv rows [32w, 32w + 32) of the 64-row block and moves rows [16w, 16w + 16)
# of every ring tile.
DKDV_NW = 2
BAR_KH = NKV // 2  # the full loop's ring barrier sits before kv sub-tile BAR_KH's WMMAs
KH0_SALU = 2  # SALU ops (the prefetch's address math) scheduled per kv sub-tile 0 WMMA
KB_SALU_MASK = 0x004  # sched_barrier mask: only SALU may cross
# Ring tiles split evenly over the waves, and dV / dK rows are whole 128-B lines.
assert 32 % DKDV_NW == 0 and (D_V * 2) % EPI_LINE_B == 0 and (D_QK * 2) % EPI_LINE_B == 0

DELTA_THREADS = 256
LANES_PER_ROW = D_V // 8
ROWS_PER_PASS = DELTA_THREADS // LANES_PER_ROW
ROWS_DELTA = 32  # queries per k_delta workgroup
PASSES_PER_WG = ROWS_DELTA // ROWS_PER_PASS


def _ir(v):
    return fx.as_ir_value(v)


def _bv(t, nb, dt, vec=1):
    """Buffer view of `t` as rows of `vec` elements of `dt`, with `nb` bytes in bounds."""
    nt = (1 << 31) // (vec * (dt.width // 8))
    b = fx.rocdl.make_buffer_tensor(t, num_records_bytes=fx.Int64(nb))
    return fx.Tensor(fx.make_view(fx.get_iter(b), fx.make_layout((nt, vec), (vec, 1))))


def _atom(dt, vec):
    return fx.make_copy_atom(fx.rocdl.BufferCopy(vec * dt.width), dt)


def _ldv(buf, tile, dt, vec):
    f = fx.make_rmem_tensor(fx.make_layout(vec, 1), dt)
    fx.copy_atom_call(_atom(dt, vec), fx.slice(buf, (tile, None)), f)
    return f.load()


def _ld1(buf, idx, dt):
    f = fx.make_rmem_tensor(fx.make_layout(1, 1), dt)
    fx.copy_atom_call(_atom(dt, 1), fx.slice(buf, (idx, None)), f)
    return f.load()[0]


def _st1(v, buf, idx, dt):
    f = fx.make_rmem_tensor(fx.make_layout(1, 1), dt)
    fx.memref_store_vec(fx.Vector.from_elements([v], dtype=dt), f)
    fx.copy_atom_call(_atom(dt, 1), f, fx.slice(buf, (idx, None)))


def _stv(vals, buf, tile, dt):
    n = len(vals)
    f = fx.make_rmem_tensor(fx.make_layout(n, 1), dt)
    fx.memref_store_vec(fx.Vector.from_elements(vals, dtype=dt), f)
    fx.copy_atom_call(_atom(dt, n), f, fx.slice(buf, (tile, None)))


def _lds_barrier():
    """Workgroup barrier fenced on LDS only, so in-flight global loads need not drain."""
    fx.gpu.barrier(address_spaces=ir.Attribute.parse("[#gpu.address_space<workgroup>]"))


def _exp2(x):
    return fx.math.exp2(x, fastmath=arith.FastMathFlags.afn)


def _tdm_rows(src, off, d, rows, valid, rs_el, lds_row0, lds_ty, num_warps=1):
    """TDM the [rows][d] bf16 tile at element `off` of `src` (row stride `rs_el`, `valid` rows in
    bounds) to LDS byte address `lds_row0` at row stride d*2 + 16, one op per power-of-two column
    segment. With num_warps > 1 every wave issues the same ops for its share of the rows, so each
    wave's tensorcnt counts only its own ops."""
    x_el = d + 8
    for c0, w in pow2_segments(d):
        g_off = off if c0 == 0 else off + fx.Int64(c0)
        lb = lds_row0 if c0 == 0 else lds_row0 + fx.Int32(c0 * 2)
        g_view = fx.Tensor(
            fx.make_view(fx.add_offset(fx.get_iter(src), g_off), fx.make_layout((rows, w), (d, 1)))
        )
        atom = fx.rocdl.make_tdm_atom(
            g_view,
            [valid, None],
            strides=[rs_el, None],
            num_warps=num_warps,
            pad_interval=w,
            pad_amount=x_el - w,
        )
        l_view = fx.Tensor(fx.make_view(fx.inttoptr(lds_ty, lb), fx.make_layout((rows, w), (x_el, 1))))
        fx.copy_atom_call(atom, g_view, l_view)


# ---------------------------------------------------------------------------------------- delta
@flyc.kernel(known_block_size=[DELTA_THREADS, 1, 1])
def k_delta_bshd(
    DO: fx.Tensor,
    O: fx.Tensor,
    DEL: fx.Tensor,
    LSE: fx.Tensor,
    NL: fx.Tensor,
    ND: fx.Tensor,
    scale: fx.Float32,
    S: fx.Int32,
    H: fx.Int32,
    n_rows: fx.Int32,
):
    """delta = sum_d dO * O, NL = -LOG2E * lse and ND = -scale * delta, all [B, H, S] fp32.

    Workgroup (sblk, bh) owns queries [32 sblk, 32 sblk + 32) of batch-major head bh, so each output
    is one 128-B line; 16 lanes sum a row and the row sums meet in LDS."""
    tid = fx.Int32(fx.thread_idx.x)
    sblk = fx.Int32(fx.block_idx.x)
    bh = fx.Int32(fx.block_idx.y)
    b = bh // H
    h = bh - b * H
    g_do = _bv(DO, n_rows * (D_V * 2), fx.BFloat16, 8)
    g_o = _bv(O, n_rows * (D_V * 2), fx.BFloat16, 8)
    g_delta = _bv(DEL, n_rows * 4, fx.Float32)
    g_lse = _bv(LSE, n_rows * 4, fx.Float32)
    g_nl = _bv(NL, n_rows * 4, fx.Float32)
    g_nd = _bv(ND, n_rows * 4, fx.Float32)
    smem = fx.SharedAllocator().allocate(ROWS_DELTA * 4)
    lds0 = fx.Int32(fx.ptrtoint(smem.peek().ptr))

    lane_in_row = tid % fx.Int32(LANES_PER_ROW)
    row_in_group = tid // fx.Int32(LANES_PER_ROW)
    s0 = sblk * fx.Int32(ROWS_DELTA)
    # pass u reads query s0 + u*ROWS_PER_PASS + row_in_group, i.e. BSHD row (b*S + s)*H + h
    tiles = [
        ((b * S + s0 + fx.Int32(u * ROWS_PER_PASS) + row_in_group) * H + h) * fx.Int32(DV8_V) + lane_in_row
        for u in range_constexpr(PASSES_PER_WG)
    ]
    do_vecs = [_ldv(g_do, tiles[u], fx.BFloat16, 8).ir_value() for u in range_constexpr(PASSES_PER_WG)]
    o_vecs = [_ldv(g_o, tiles[u], fx.BFloat16, 8).ir_value() for u in range_constexpr(PASSES_PER_WG)]

    for u in range_constexpr(PASSES_PER_WG):
        do8 = fx.Vector(do_vecs[u])
        o8 = fx.Vector(o_vecs[u])
        e0 = fx.Float32(0.0)
        e1 = fx.Float32(0.0)
        for c in range_constexpr(4):  # the lane's 8 elements in 2 FMA chains
            e0 = e0 + fx.Float32(do8[2 * c]) * fx.Float32(o8[2 * c])
            e1 = e1 + fx.Float32(do8[2 * c + 1]) * fx.Float32(o8[2 * c + 1])
        acc = e0 + e1
        for sft in range_constexpr(LANES_PER_ROW.bit_length() - 1):
            acc = acc + fx.gpu.shuffle_xor(acc, 1 << sft, WAVE)
        # the xor butterfly leaves the row sum in all 16 lanes of the row; they all store it
        r = fx.Int32(u * ROWS_PER_PASS) + row_in_group
        llvm.store(fx.as_ir_value(acc), lds_ptr(lds0 + r * fx.Int32(4)))
    _lds_barrier()
    # Lane l of every wave reads row l; only wave 0 stores, the other waves store to index n_rows,
    # past the end of every buffer.
    l32 = tid % fx.Int32(ROWS_DELTA)
    v = fx.Float32(llvm.load(fx.Float32.ir_type, lds_ptr(lds0 + l32 * fx.Int32(4))))
    dst = bh * S + s0 + l32
    w0 = tid < fx.Int32(ROWS_DELTA)
    _st1(v, g_delta, w0.select(dst, n_rows), fx.Float32)
    _st1(v * (fx.Float32(0.0) - scale), g_nd, w0.select(dst, n_rows), fx.Float32)
    lse_r = _ld1(g_lse, dst, fx.Float32)
    _st1(lse_r * fx.Float32(-LOG2E), g_nl, w0.select(dst, n_rows), fx.Float32)


@flyc.jit
def launch_delta(
    DO,
    O,
    DEL,
    LSE,
    NL,
    ND,
    scale: fx.Float32,
    S: fx.Int32,
    H: fx.Int32,
    n_rows: fx.Int32,
    nsblk: fx.Int32,
    nbh: fx.Int32,
    stream: fx.Stream,
):
    _ = SOURCE_DIGEST  # FlyDSL disk-cache key
    # grid = (S / ROWS_DELTA query blocks, B * H heads)
    k_delta_bshd(DO, O, DEL, LSE, NL, ND, scale, S, H, n_rows).launch(
        grid=(nsblk, nbh, 1), block=(DELTA_THREADS, 1, 1), stream=stream
    )


# ----------------------------------------------------------------------------------------- dkdv
def _dkdv_impl(
    Q, K, V, DO, NL, ND, DV_, DK, scale, Sq, Skv, Hq, Hkv, G, nqt, cshift, causal, B_, nw=1, HG=None
):
    """dK/dV of BLOCK_KV kv rows per wave: every (query pair, q head) of the kv head streams through
    the Q/dO ring into register accumulators (the G q heads of a kv head reduce there too).

    nw = 1 (k_dkdv): grid (Hkv, Skv/32, B), no barriers. nw = DKDV_NW (k_dkdv64): grid
    (HG, Skv/64, B*Hkv/HG); the waves share one ring and take the same barriers (B = _lds_barrier,
    each wave's TDM retired by tensor_wait first): masked iteration B, TDM, wait, B, read; prologue
    B, TDM x2, wait, B, read, TDM; full iteration tr16, WMMAs below BAR_KH, wait, B, read the next
    stage, TDM, the other WMMAs; exit wait, B, epilogue.
    """
    if const_expr(nw == 1):
        lane = fx.Int32(fx.thread_idx.x)
    else:
        lane = fx.Int32(fx.thread_idx.x) % fx.Int32(WAVE)
    if const_expr(HG is None):
        hkv = fx.Int32(fx.block_idx.x)
        bat = fx.Int32(fx.block_idx.z)
    else:
        # head group: z*HG + x is the batch-major kv head bat*Hkv + hkv
        _hb = fx.Int32(fx.block_idx.z) * HG + fx.Int32(fx.block_idx.x)
        bat = _hb // Hkv
        hkv = _hb - bat * Hkv
    bid = fx.Int32(fx.block_idx.y)
    row = lane % fx.Int32(16)
    half = lane // fx.Int32(16)
    kv0 = bid * fx.Int32(BLOCK_KV)
    kv0g = kv0  # the workgroup's first kv row (loop bounds)
    if const_expr(nw > 1):
        wv = fx.Int32(rocdl.wave_id())  # wave index, an SGPR
        kv0g = bid * fx.Int32(nw * BLOCK_KV)
        kv0 = kv0g + wv * fx.Int32(BLOCK_KV)  # this wave's kv rows

    # True byte extents: an over-read returns 0 instead of touching the next page.
    nk_b = B_ * Skv * Hkv * fx.Int32(D_QK * 2)  # k / dk
    nv_b = B_ * Skv * Hkv * fx.Int32(D_V * 2)  # v / dv
    nl_b = B_ * Hq * Sq * fx.Int32(4)  # NL / ND
    g_k = _bv(K, nk_b, fx.BFloat16, 8)
    g_v = _bv(V, nv_b, fx.BFloat16, 8)
    g_nl = _bv(NL, nl_b, fx.Float32)
    g_nd = _bv(ND, nl_b, fx.Float32)

    rs_k = Hkv * fx.Int32(DV8_QK)  # vectors between consecutive k rows
    base_k = bat * Skv * rs_k + hkv * fx.Int32(DV8_QK)
    rs_v = Hkv * fx.Int32(DV8_V)
    base_v = bat * Skv * rs_v + hkv * fx.Int32(DV8_V)

    # The Q/dO ring; the epilogue images reuse it.
    smem = fx.SharedAllocator().allocate(max(TDM_DEPTH * QDO_B, nw * EPI_W))
    _lds0 = fx.Int32(fx.ptrtoint(smem.peek().ptr))
    v8b = fx.Vector.make_type(8, fx.BFloat16)
    v8f = fx.Vector.make_type(8, fx.Float32)

    def gfrag(buf, base, rs, r, dt):
        t = base + (r + row) * rs + half + fx.Int32(dt * 4)
        return _ldv(buf, t, fx.BFloat16, 8).shuffle(
            _ldv(buf, t + fx.Int32(2), fx.BFloat16, 8), list(range(16))
        )

    # NL/ND loads: voffset 32*half (a loop-invariant VGPR) + an SGPR soffset.
    rsrc_nl = rocdl.get_buffer_rsrc(fx.get_iter(g_nl))
    rsrc_nd = rocdl.get_buffer_rsrc(fx.get_iter(g_nd))
    voff_l = half * fx.Int32(32)
    v4f = fx.Vector.make_type(4, fx.Float32)
    c1_sm = scale * fx.Float32(LOG2E)

    def _ldl(qt, gh):
        """NL and ND of query pair qt, q head hkv*G + gh: per 16-query half hh, NL u0, NL u1, ND u0,
        ND u1, the 8 query rows hh*16 + half*8 + [0, 8) of this lane's S/dP C elements."""
        qh = hkv * G + gh
        q0 = qt * fx.Int32(32)
        base_l = (bat * Hq + qh) * Sq
        sb = fx.Int32(rocdl.readfirstlane(fx.Int32.ir_type, ((base_l + q0) * fx.Int32(4)).ir_value()))
        out = []
        for hh in range_constexpr(2):
            for rs_ in (rsrc_nl, rsrc_nd):
                for u in range_constexpr(2):
                    so = sb + fx.Int32(hh * 64 + u * 16)
                    out.append(fx.Vector(rocdl.raw_ptr_buffer_load(v4f, rs_, voff_l, so)))
        return out

    # Q/dO ring: TDM_DEPTH stages of QDO_B bytes. The tile of query pair qt starts at element
    # ((bat*Sq + 32*qt)*Hq + qh)*D with Sq - 32*qt >= 32 rows in bounds (qt is clamped).
    _lds_bf_ty = fx.PointerType.get(
        elem_ty=fx.BFloat16.ir_type, address_space=fx.AddressSpace.Shared, alignment=16
    )
    _q_rs_v = Hq * fx.Int32(D_V)  # elements between consecutive do rows
    _q_rs_k = Hq * fx.Int32(D_QK)

    def _tdm_qdo_prep(qt, gh, stage_off):
        """The scalar address math of one stage's TDM (emitted where called)."""
        qh = hkv * G + gh
        q0 = qt * fx.Int32(32)
        row0 = fx.Int64(bat * Sq + q0) * fx.Int64(Hq) + fx.Int64(qh)
        off_v = row0 * fx.Int64(D_V)
        off_k = row0 * fx.Int64(D_QK)
        valid = Sq - q0
        lb_v = _lds0 + stage_off
        lb_k = _lds0 + stage_off + fx.Int32(QOFF)
        return (off_v, off_k, valid, lb_v, lb_k)

    def _tdm_qdo_issue(p):
        off_v, off_k, valid, lb_v, lb_k = p
        _tdm_rows(DO, off_v, D_V, 32, valid, _q_rs_v, lb_v, _lds_bf_ty, nw)
        _tdm_rows(Q, off_k, D_QK, 32, valid, _q_rs_k, lb_k, _lds_bf_ty, nw)

    def _tdm_qdo(qt, gh, stage_off):
        _tdm_qdo_issue(_tdm_qdo_prep(qt, gh, stage_off))

    # K and V fragments of this kv tile serve every query and q head.
    kf = [
        [gfrag(g_k, base_k, rs_k, kv0 + fx.Int32(kh * 16), dt) for dt in range(NDT_QK)] for kh in range(NKV)
    ]
    vf = [[gfrag(g_v, base_v, rs_v, kv0 + fx.Int32(kh * 16), dt) for dt in range(NDT_V)] for kh in range(NKV)]
    lane_r = (lane // fx.Int32(16)) * fx.Int32(8) + lane % fx.Int32(8)
    lane_c = ((lane // fx.Int32(8)) % fx.Int32(2)) * fx.Int32(8)
    # Lane parts of the DS addresses; every per-op offset folds into the DS immediate.
    lb_tr_v = lane_r * fx.Int32(XV_ROW_B) + lane_c * fx.Int32(2)  # dO tr16
    lb_rd_v = row * fx.Int32(XV_ROW_B) + half * fx.Int32(16)  # dO readback
    lb_tr_k = lane_r * fx.Int32(XK_ROW_B) + lane_c * fx.Int32(2)  # Q tr16
    lb_rd_k = row * fx.Int32(XK_ROW_B) + half * fx.Int32(16)  # Q readback

    def _rd_bases(stage_off):
        """(Q base, dO base) of a stage's readback."""
        b_v = _lds0 + stage_off + lb_rd_v
        return (_lds0 + stage_off + lb_rd_k, b_v)

    # Bottom-right causal: query pairs below qp_start see none of the workgroup's kv rows, pairs from
    # qp_start + nmaskp on see all of them. The bounds use the workgroup's rows, so every wave runs
    # the same trip counts; inside the masked pairs each wave masks with its own kv0.
    nqt2 = nqt // fx.Int32(2)  # query tiles come in pairs

    def _clampqt(t):
        """Clamp a prefetched query pair into [0, nqt2 - 1]."""
        t = (t < nqt2).select(t, nqt2 - fx.Int32(1))
        return (t < fx.Int32(0)).select(fx.Int32(0), t)

    _c = kv0g - cshift
    qp_start = ((_c < fx.Int32(0)).select(fx.Int32(0), _c)) // fx.Int32(32)
    qp_start = (causal != fx.Int32(0)).select(qp_start, fx.Int32(0))
    nqp_eff = nqt2 - qp_start

    # Iterations run query pair outer, q head inner: the pairs that straddle the workgroup's
    # diagonal come first and are masked in their own loop (qloop_mask), so the hot loop
    # (qloop_full) carries no mask.
    def _rdqd(stage_off, rbase=None):
        """The 2*NHB ds_load_b128 of one stage's S/dP B operands: entry hh*NHB + dt*2 + u is Q row
        hh*16 + row, bytes half*16 + dt*64 + u*32; entry hh*NHB + NQE + dt*2 + u the same for dO."""
        if const_expr(rbase is None):
            rbase = _rd_bases(stage_off)
        out = []
        for hh in range_constexpr(2):
            for rb_, img, xrow, ndt in (
                (rbase[0], QOFF, XK_ROW_B, NDT_QK),
                (rbase[1], 0, XV_ROW_B, NDT_V),
            ):
                for dt in range_constexpr(ndt):
                    for u in range_constexpr(2):
                        out.append(
                            fx.Vector(
                                llvm.load(
                                    v8b, lds_ptr(rb_ + fx.Int32(hh * 16 * xrow + img + dt * 64 + u * 32))
                                )
                            )
                        )
        return out

    def _body(
        acc,
        pre,
        qt,
        gh,
        do_mask,
        qt_n,
        gh_n,
        carry=True,
        cur_off=None,
        nxt_off=None,
        pf_qt=None,
        pf_gh=None,
        qd=None,
        rb_off=None,
        pf3=None,
    ):
        q0 = qt * fx.Int32(32)

        # carry (qloop_full): this iteration's B operands arrive in qd, read back one iteration early.
        # nw = 1 issues the TDM of iteration it+2 at the top, into the stage it-1 last read; k_dkdv64
        # issues the TDM of iteration it+3 into stage it%3 after its ring barrier. Not carry
        # (qloop_mask): load the tile into stage 0, wait.
        rb_base = None
        if const_expr(carry and nw > 1):
            rocdl.sched_barrier(0)
            nxt = _ldl(qt_n, gh_n)
        elif const_expr(carry):
            # TDM first, so the NL/ND soffset SGPRs are not recycled by the TDM descriptor math
            rocdl.sched_barrier(0)
            _tdm_qdo(pf_qt, pf_gh, nxt_off)
            rocdl.sched_barrier(0)
            nxt = _ldl(qt_n, gh_n)
        else:
            nxt = None
            cur_off = fx.Int32(0)
            pre = _ldl(qt, gh)
            if const_expr(nw > 1):
                rocdl.sched_barrier(0)
                _lds_barrier()  # WAR: every wave's reads of stage 0 retired
                rocdl.sched_barrier(0)
            _tdm_qdo(qt, gh, cur_off)
            rocdl.sched_barrier(0)
            tdm_ops.tensor_wait(0)
            rocdl.sched_barrier(0)
            if const_expr(nw > 1):
                _lds_barrier()  # RAW: every wave's half of the tile landed
                rocdl.sched_barrier(0)
        lds_do = _lds0 + cur_off

        ops = qd if const_expr(carry) else _rdqd(cur_off)
        # S = Q K^T and dP = dO V^T put kv rows on lanes and query rows on elements, so the bf16 pack
        # of P (dS) per kv sub-tile is directly the B operand of dV^T += dO^T P (dK^T += Q^T dS).
        p_half = [[] for _ in range_constexpr(NKV)]
        ds_half = [[] for _ in range_constexpr(NKV)]
        for hh in range_constexpr(2):
            qp = [(ops[hh * NHB + dt * 2], ops[hh * NHB + dt * 2 + 1]) for dt in range_constexpr(NDT_QK)]
            dp = [
                (ops[hh * NHB + NQE + dt * 2], ops[hh * NHB + NQE + dt * 2 + 1])
                for dt in range_constexpr(NDT_V)
            ]
            qfr = [qp[dt][0].shuffle(qp[dt][1], list(range(16))) for dt in range_constexpr(NDT_QK)]
            dfr = [dp[dt][0].shuffle(dp[dt][1], list(range(16))) for dt in range_constexpr(NDT_V)]
            # p = exp2(s*scale*LOG2E + NL), ds = p * (dp*scale + ND), query rows qb + si of this lane
            nl = [pre[hh * 4 + si // 4][si % 4] for si in range(8)]
            nd = [pre[hh * 4 + 2 + si // 4][si % 4] for si in range(8)]
            qb = q0 + fx.Int32(hh * 16) + half * fx.Int32(8)
            for kh in range_constexpr(NKV):
                s_acc = _ir(fx.Vector.filled(8, 0.0, fx.Float32))
                p_acc = _ir(fx.Vector.filled(8, 0.0, fx.Float32))
                for dt in range_constexpr(NDT_MAX):
                    if const_expr(dt < NDT_QK):
                        s_acc = rocdl.wmma_f32_16x16x32_bf16(
                            v8f, _ir(qfr[dt]), _ir(kf[kh][dt]), s_acc, reuseA=False, reuseB=False
                        ).result
                    if const_expr(dt < NDT_V):
                        p_acc = rocdl.wmma_f32_16x16x32_bf16(
                            v8f, _ir(dfr[dt]), _ir(vf[kh][dt]), p_acc, reuseA=False, reuseB=False
                        ).result
                sv, pv_ = fx.Vector(s_acc), fx.Vector(p_acc)
                kvl = kv0 + fx.Int32(kh * 16) + row  # this lane's kv row
                tt = [fx.Float32(fx.fma(sv[si], c1_sm, nl[si])) for si in range(8)]
                if const_expr(do_mask):
                    tt = [
                        ((kvl > qb + fx.Int32(si) + cshift) & (causal != fx.Int32(0))).select(
                            fx.Float32(NEG), tt[si]
                        )
                        for si in range(8)
                    ]
                pf = [_exp2(tt[si]) for si in range(8)]
                p_half[kh].append([x.to(fx.BFloat16) for x in pf])
                ds_half[kh].append(
                    [(pf[si] * fx.Float32(fx.fma(pv_[si], scale, nd[si]))).to(fx.BFloat16) for si in range(8)]
                )
        a_pk = [
            fx.Vector.from_elements(p_half[kh][0] + p_half[kh][1], dtype=fx.BFloat16)
            for kh in range_constexpr(NKV)
        ]
        a_dsk = [
            fx.Vector.from_elements(ds_half[kh][0] + ds_half[kh][1], dtype=fx.BFloat16)
            for kh in range_constexpr(NKV)
        ]
        # One DS phase per iteration: the dO/Q tr16 loads and, with carry, the next stage's readback.
        # Its bases are computed before the fence so their latency hides under the S/dP phase.
        tr_v = lds_do + lb_tr_v
        tr_k = lds_do + lb_tr_k
        if const_expr(carry):
            rb_base = _rd_bases(rb_off)
        rocdl.sched_barrier(0)

        # 32 staged queries: rows lane_r and lane_r + 16 concatenate into one v16 operand
        def tr(base, rowb):
            return fx.Vector(rocdl.ds_load_tr16_b128(v8b, lds_ptr(base))).shuffle(
                fx.Vector(rocdl.ds_load_tr16_b128(v8b, lds_ptr(base + fx.Int32(16 * rowb)))),
                list(range(16)),
            )

        if const_expr(carry):
            # DS issue order = WMMA consumption order: dO (dV) first, then Q (dK)
            b_do = [tr(tr_v + fx.Int32(dtile * 32), XV_ROW_B) for dtile in range_constexpr(NDO_V)]
            rocdl.sched_barrier(0)
            b_q = [tr(tr_k + fx.Int32(QOFF + dtile * 32), XK_ROW_B) for dtile in range_constexpr(NDO_QK)]
            rocdl.sched_barrier(0)
        else:
            b_do = []
            b_q = []
            for dtile in range_constexpr(NDO_MAX):
                if const_expr(dtile < NDO_V):
                    b_do.append(tr(tr_v + fx.Int32(dtile * 32), XV_ROW_B))
                if const_expr(dtile < NDO_QK):
                    b_q.append(tr(tr_k + fx.Int32(QOFF + dtile * 32), XK_ROW_B))
        if const_expr(carry and nw == 1):
            # Two stages are in flight and TDM ops retire in order: TW_QDO retires the one read next.
            rocdl.sched_barrier(0)
            tdm_ops.tensor_wait(TW_QDO)
            rocdl.sched_barrier(0)
            rb = _rdqd(rb_off, rb_base)
        rocdl.sched_barrier(0)
        tdm_p = None
        if const_expr(carry and nw > 1):
            # the prefetch's (tile it+3 into stage it%3) address math, scheduled between kh0 WMMAs
            pq, pg = pf3()
            tdm_p = _tdm_qdo_prep(pq, pg, cur_off)
        new = [None] * NST
        for kh in range_constexpr(NKV):
            if const_expr(carry and nw > 1 and kh == BAR_KH):
                # The ring barrier, after the kh0 WMMAs (they hide the tr16 drain): tensor_wait
                # retires this wave's half of stage (it+1)%3, the barrier publishes both halves and
                # retires every wave's reads of stage it%3; then the readback and the prefetch.
                rocdl.sched_barrier(0)
                tdm_ops.tensor_wait(TW_QDO)
                rocdl.sched_barrier(0)
                _lds_barrier()
                rocdl.sched_barrier(0)
                rb = _rdqd(rb_off, rb_base)
                rocdl.sched_barrier(KB_SALU_MASK)  # readback before the TDM; SALU may cross
                _tdm_qdo_issue(tdm_p)
                rocdl.sched_barrier(0)
            a_p = a_pk[kh]
            a_ds = a_dsk[kh]
            # dV^T += dO^T P and dK^T += Q^T dS: a C fragment holds 8 consecutive d of one kv row,
            # one ds_store_b128 in the epilogue. B (P or dS) is reused across the dtiles.
            for dtile in range_constexpr(NDO_V):
                new[kh * NDO_V + dtile] = rocdl.wmma_f32_16x16x32_bf16(
                    v8f, _ir(b_do[dtile]), _ir(a_p), acc[kh * NDO_V + dtile], reuseA=False, reuseB=(dtile > 0)
                ).result
            for dtile in range_constexpr(NDO_QK):
                new[DK0 + kh * NDO_QK + dtile] = rocdl.wmma_f32_16x16x32_bf16(
                    v8f,
                    _ir(b_q[dtile]),
                    _ir(a_ds),
                    acc[DK0 + kh * NDO_QK + dtile],
                    reuseA=False,
                    reuseB=(dtile > 0),
                ).result
            if const_expr(carry and nw > 1 and kh + 1 == BAR_KH):
                # {1 WMMA, KH0_SALU SALU} per kh0 WMMA: the prefetch's scalar math rides the run
                for _ in range_constexpr(BAR_KH * (NDO_V + NDO_QK)):
                    rocdl.sched_group_barrier(0x008, 1, 0)
                    rocdl.sched_group_barrier(0x004, KH0_SALU, 0)
        if const_expr(carry):
            # keeps the NL/ND back-edge copies, and their s_wait_loadcnt, at the end of the body
            rocdl.sched_barrier(0)
            return new + [_ir(v) for v in nxt] + [_ir(v) for v in rb]
        return new

    # Division-free loop counters: ii = qc*G + gc advances by the wrap gc + 1 == G -> (qc + 1, 0).
    def _wrap(qc, gc):
        g1 = gc + fx.Int32(1)
        same_q = g1 < G
        return fx.Int32(same_q.select(qc, qc + fx.Int32(1))), fx.Int32(same_q.select(g1, fx.Int32(0)))

    @flyc.jit
    def qloop_mask(state, n, qt0):
        final = state
        for it, carried in range(fx.Index(fx.Int32(0)), fx.Index(n), 1, init=state):
            st = list(carried)
            qi, gh = fx.Int32(st[NST]), fx.Int32(st[NST + 1])
            qn, gn = _wrap(qi, gh)
            res = _body(st[:NST], None, qt0 + qi, gh, do_mask=True, qt_n=None, gh_n=None, carry=False) + [
                fx.as_ir_value(qn),
                fx.as_ir_value(gn),
            ]
            final = yield res
        return final

    @flyc.jit
    def qloop_full(state, n, qt0):
        final = state
        for it, carried in range(fx.Index(fx.Int32(0)), fx.Index(n), 1, init=state):
            ii = fx.Int32(it)
            jj = ii + fx.Int32(1)
            st = list(carried)
            # carried: accumulators, NL/ND at [NST, NST + NLD), B operands, then (cur, qi, gh)
            cur, qi, gh = fx.Int32(st[-3]), fx.Int32(st[-2]), fx.Int32(st[-1])
            qn, gn = _wrap(qi, gh)  # iteration ii + 1
            live = jj < n  # else jj clamps to n - 1 == ii
            qj = fx.Int32(live.select(qn, qi))
            gj = fx.Int32(live.select(gn, gh))
            # stage (ii + 1) % 3: the next iteration's, read back during this one
            ncur = fx.Int32((cur == fx.Int32(2 * QDO_B)).select(fx.Int32(0), cur + fx.Int32(QDO_B)))
            if const_expr(nw > 1):
                # prefetch tile min(ii + 3, n - 1) into stage ii % 3 == cur, after the ring barrier
                def _pf3(qn=qn, gn=gn, qj=qj, gj=gj, ii=ii):
                    q2, g2 = _wrap(qn, gn)  # iteration ii + 2
                    live2 = (ii + fx.Int32(2)) < n
                    q2c = fx.Int32(live2.select(q2, qj))
                    g2c = fx.Int32(live2.select(g2, gj))
                    q3, g3 = _wrap(q2, g2)  # iteration ii + 3
                    live3 = (ii + fx.Int32(3)) < n
                    return (qt0 + fx.Int32(live3.select(q3, q2c)), fx.Int32(live3.select(g3, g2c)))

                res = _body(
                    st[:NST],
                    [fx.Vector(v) for v in st[NST : NST + NLD]],
                    qt0 + qi,
                    gh,
                    do_mask=False,
                    qt_n=qt0 + qj,
                    gh_n=gj,
                    cur_off=cur,
                    qd=[fx.Vector(v) for v in st[NST + NLD : -3]],
                    rb_off=ncur,
                    pf3=_pf3,
                )
            else:
                q2, g2 = _wrap(qn, gn)  # iteration ii + 2
                live2 = (ii + fx.Int32(2)) < n  # else it clamps to n - 1 == jj
                pf_qt = qt0 + fx.Int32(live2.select(q2, qj))
                pf_gh = fx.Int32(live2.select(g2, gj))
                # prefetch stage (ii + 2) % 3 == (ii - 1) % 3, the one before cur
                nxo = fx.Int32((cur == fx.Int32(0)).select(fx.Int32(2 * QDO_B), cur - fx.Int32(QDO_B)))
                res = _body(
                    st[:NST],
                    [fx.Vector(v) for v in st[NST : NST + NLD]],
                    qt0 + qi,
                    gh,
                    do_mask=False,
                    qt_n=qt0 + qj,
                    gh_n=gj,
                    cur_off=cur,
                    nxt_off=nxo,
                    pf_qt=pf_qt,
                    pf_gh=pf_gh,
                    qd=[fx.Vector(v) for v in st[NST + NLD : -3]],
                    rb_off=ncur,
                )
            res = res + [fx.as_ir_value(ncur), fx.as_ir_value(qn), fx.as_ir_value(gn)]
            final = yield res
        return final

    def _tdm_prologue(qt0, n):
        """Fill stages 0 and 1 with iterations 0 and min(1, n - 1) of qloop_full (clamped to a legal
        query pair, so an empty loop still issues in-bounds tiles), retire stage 0 and read it back.
        nw > 1 then also fills stage 2 (iteration min(2, n - 1)) after the RAW barrier."""
        if const_expr(nw > 1):
            rocdl.sched_barrier(0)
            _lds_barrier()  # WAR: every wave's reads of stage 0 in qloop_mask retired
            rocdl.sched_barrier(0)
        _tdm_qdo(_clampqt(qt0), fx.Int32(0), fx.Int32(0))
        q1w, g1w = _wrap(fx.Int32(0), fx.Int32(0))
        two = fx.Int32(1) < n
        q1 = fx.Int32(two.select(q1w, fx.Int32(0)))
        g1 = fx.Int32(two.select(g1w, fx.Int32(0)))
        _tdm_qdo(_clampqt(qt0 + q1), g1, fx.Int32(QDO_B))
        rocdl.sched_barrier(0)
        tdm_ops.tensor_wait(TW_QDO)
        rocdl.sched_barrier(0)
        if const_expr(nw > 1):
            _lds_barrier()  # RAW: every wave's half of stage 0 landed
            rocdl.sched_barrier(0)
        rb0 = _rdqd(fx.Int32(0))
        if const_expr(nw > 1):
            q2w, g2w = _wrap(q1w, g1w)
            three = fx.Int32(2) < n
            q2 = fx.Int32(three.select(q2w, q1))
            g2 = fx.Int32(three.select(g2w, g1))
            rocdl.sched_barrier(0)  # readback before the TDM
            _tdm_qdo(_clampqt(qt0 + q2), g2, fx.Int32(2 * QDO_B))
            rocdl.sched_barrier(0)
        return [_ir(v) for v in rb0]

    # Query pair qt is fully unmasked iff its smallest query attends the workgroup's largest kv row:
    # qt >= ceil((kv0g + nw*BLOCK_KV - 1 - cshift) / 32); cshift = Skv - Sq can be negative.
    _u = kv0g + fx.Int32(nw * BLOCK_KV - 1) - cshift
    _qsf = (_u < fx.Int32(0)).select(fx.Int32(0), (_u + fx.Int32(31)) // fx.Int32(32))
    _qsf = (_qsf < nqt2).select(_qsf, nqt2)
    _nm = _qsf - qp_start
    _nm = (_nm < fx.Int32(0)).select(fx.Int32(0), _nm)
    _nm = (_nm < nqp_eff).select(_nm, nqp_eff)
    nmaskp = (causal != fx.Int32(0)).select(_nm, fx.Int32(0))

    init = [_ir(fx.Vector.filled(8, 0.0, fx.Float32)) for _ in range(NST)]
    # both loops start at ii = 0: (qi, gh) = (0, 0), ring stage offset 0
    _z2 = [fx.as_ir_value(fx.Int32(0)), fx.as_ir_value(fx.Int32(0))]
    _z3 = [fx.as_ir_value(fx.Int32(0))] + _z2
    out = qloop_mask(init + _z2, G * nmaskp, qp_start)
    # qloop_full's first NL/ND loads and its TDM prologue (clamped: issued even for an empty loop)
    _pc0 = _clampqt(qp_start + nmaskp)
    out = list(out)[:NST] + [_ir(v) for v in _ldl(_pc0, fx.Int32(0))]
    out = out + _tdm_prologue(qp_start + nmaskp, G * (nqp_eff - nmaskp)) + _z3
    out = qloop_full(out, G * (nqp_eff - nmaskp), qp_start + nmaskp)
    # retire the last (clamped, unread) prefetch before the epilogue overwrites the ring
    tdm_ops.tensor_wait(0)
    _epi0 = _lds0
    if const_expr(nw > 1):
        # every wave's TDM landed and ring reads retired before any image overwrites the ring
        rocdl.sched_barrier(0)
        _lds_barrier()
        rocdl.sched_barrier(0)
        _epi0 = _lds0 + wv * fx.Int32(EPI_W)

    # Epilogue: each dV^T / dK^T C fragment (kv row kh*16 + row, d = dtile*16 + half*8 + [0, 8)) goes
    # into this wave's row-major images as one ds_store_b128; the images are read back in line order
    # so each 8-lane run of a buffer_store_b128 writes one aligned 128-B line. Each load is followed
    # by its store. The images are wave-local, so no barrier.
    g_dv8 = _bv(DV_, nv_b, fx.BFloat16, 8)
    g_dk8 = _bv(DK, nk_b, fx.BFloat16, 8)
    wr_v = _epi0 + row * fx.Int32(XV_ROW_B) + half * fx.Int32(16)
    wr_k = _epi0 + fx.Int32(EPI_DK) + row * fx.Int32(XK_ROW_B) + half * fx.Int32(16)
    for kh in range_constexpr(NKV):
        for dtile in range_constexpr(NDO_MAX):
            if const_expr(dtile < NDO_V):
                ov = fx.Vector(out[kh * NDO_V + dtile])
                llvm.store(
                    fx.as_ir_value(
                        fx.Vector.from_elements(
                            [ov[si].to(fx.BFloat16) for si in range_constexpr(8)], dtype=fx.BFloat16
                        )
                    ),
                    lds_ptr(wr_v + fx.Int32(kh * 16 * XV_ROW_B + dtile * 32)),
                )
            if const_expr(dtile < NDO_QK):
                ok_ = fx.Vector(out[DK0 + kh * NDO_QK + dtile])
                llvm.store(
                    fx.as_ir_value(
                        fx.Vector.from_elements(
                            [ok_[si].to(fx.BFloat16) for si in range_constexpr(8)], dtype=fx.BFloat16
                        )
                    ),
                    lds_ptr(wr_k + fx.Int32(kh * 16 * XK_ROW_B + dtile * 32)),
                )
    rocdl.sched_barrier(0)
    lrow = lane // fx.Int32(8)  # 4 kv rows per store
    lcol = lane % fx.Int32(8)  # 16-B chunk of the 128-B line
    rd_v = _epi0 + lrow * fx.Int32(XV_ROW_B) + lcol * fx.Int32(16)
    rd_k = _epi0 + fx.Int32(EPI_DK) + lrow * fx.Int32(XK_ROW_B) + lcol * fx.Int32(16)
    for rb in range_constexpr(BLOCK_KV // 4):
        # vector index of (kv row kv0 + 4rb + lrow, head hkv, d = 0) in dv / dk
        gt_v = base_v + (kv0 + fx.Int32(rb * 4) + lrow) * rs_v + lcol
        gt_k = base_k + (kv0 + fx.Int32(rb * 4) + lrow) * rs_k + lcol
        for li in range_constexpr(D_V * 2 // EPI_LINE_B):
            vv = fx.Vector(llvm.load(v8b, lds_ptr(rd_v + fx.Int32(rb * 4 * XV_ROW_B + li * EPI_LINE_B))))
            _stv(
                [vv[e] for e in range_constexpr(8)],
                g_dv8,
                gt_v + fx.Int32(li * (EPI_LINE_B // 16)),
                fx.BFloat16,
            )
        for li in range_constexpr(D_QK * 2 // EPI_LINE_B):
            kk = fx.Vector(llvm.load(v8b, lds_ptr(rd_k + fx.Int32(rb * 4 * XK_ROW_B + li * EPI_LINE_B))))
            _stv(
                [kk[e] for e in range_constexpr(8)],
                g_dk8,
                gt_k + fx.Int32(li * (EPI_LINE_B // 16)),
                fx.BFloat16,
            )


@flyc.kernel(known_block_size=[32, 1, 1])
def k_dkdv(
    Q: fx.Tensor,
    K: fx.Tensor,
    V: fx.Tensor,
    DO: fx.Tensor,
    NL: fx.Tensor,
    ND: fx.Tensor,
    DV_: fx.Tensor,
    DK: fx.Tensor,
    scale: fx.Float32,
    Sq: fx.Int32,
    Skv: fx.Int32,
    Hq: fx.Int32,
    Hkv: fx.Int32,
    G: fx.Int32,
    nqt: fx.Int32,
    cshift: fx.Int32,
    causal: fx.Int32,
    B_: fx.Int32,
):
    _dkdv_impl(Q, K, V, DO, NL, ND, DV_, DK, scale, Sq, Skv, Hq, Hkv, G, nqt, cshift, causal, B_)


@flyc.jit
def launch_dkdv(
    Q,
    K,
    V,
    DO,
    NL,
    ND,
    DV_,
    DK,
    scale: fx.Float32,
    Sq: fx.Int32,
    Skv: fx.Int32,
    Hq: fx.Int32,
    Hkv: fx.Int32,
    G: fx.Int32,
    nqt: fx.Int32,
    cshift: fx.Int32,
    causal: fx.Int32,
    nblk: fx.Int32,
    nhkv: fx.Int32,
    nb: fx.Int32,
    stream: fx.Stream,
):
    _ = SOURCE_DIGEST  # FlyDSL disk-cache key
    k_dkdv(Q, K, V, DO, NL, ND, DV_, DK, scale, Sq, Skv, Hq, Hkv, G, nqt, cshift, causal, nb).launch(
        grid=(nhkv, nblk, nb), block=(32, 1, 1), stream=stream
    )


# Two waves per workgroup (one per SIMD) fetch 64 kv rows' worth of Q/dO once: half the TDM bytes
# and LDS writes per dK/dV FLOP of k_dkdv.
@flyc.kernel(known_block_size=[WAVE * DKDV_NW, 1, 1])
def k_dkdv64(
    Q: fx.Tensor,
    K: fx.Tensor,
    V: fx.Tensor,
    DO: fx.Tensor,
    NL: fx.Tensor,
    ND: fx.Tensor,
    DV_: fx.Tensor,
    DK: fx.Tensor,
    scale: fx.Float32,
    Sq: fx.Int32,
    Skv: fx.Int32,
    Hq: fx.Int32,
    Hkv: fx.Int32,
    G: fx.Int32,
    nqt: fx.Int32,
    cshift: fx.Int32,
    causal: fx.Int32,
    B_: fx.Int32,
    HG: fx.Int32,
):
    rocdl.disable_xdl_arb_stall()  # no WMMA multicycle arbitration stall (SCHED_MODE)
    _dkdv_impl(Q, K, V, DO, NL, ND, DV_, DK, scale, Sq, Skv, Hq, Hkv, G, nqt, cshift, causal, B_, DKDV_NW, HG)


@flyc.jit
def launch_dkdv64(
    Q,
    K,
    V,
    DO,
    NL,
    ND,
    DV_,
    DK,
    scale: fx.Float32,
    Sq: fx.Int32,
    Skv: fx.Int32,
    Hq: fx.Int32,
    Hkv: fx.Int32,
    G: fx.Int32,
    nqt: fx.Int32,
    cshift: fx.Int32,
    causal: fx.Int32,
    nblk: fx.Int32,
    hg: fx.Int32,
    ngz: fx.Int32,
    nb: fx.Int32,
    stream: fx.Stream,
):
    _ = SOURCE_DIGEST  # FlyDSL disk-cache key
    # grid = (hg kv heads of a head group, Skv/64 kv blocks, B*Hkv/hg head groups)
    k_dkdv64(Q, K, V, DO, NL, ND, DV_, DK, scale, Sq, Skv, Hq, Hkv, G, nqt, cshift, causal, nb, hg).launch(
        grid=(hg, nblk, ngz), block=(WAVE * DKDV_NW, 1, 1), stream=stream
    )


# ------------------------------------------------------------------------------------------- dq
# A dQ wave owns 16*nqw queries of one q head and streams every KV_STEP-row kv block through a
# 3-stage K/V ring; in iteration i stage i%3 holds tile i (dQ tr16), stage (i+1)%3 tile i+1 (read
# back into the next S/dP A operands) and stage (i+2)%3 receives tile i+2. The masked loop
# continues the full loop's ring.
KV_STEP = 32  # kv rows per k_dqg iteration
NKT = KV_STEP // 16
DQ_BQW = 32  # queries per k_dqg wave (64 would spill at D_QK = 192)
NQW = DQ_BQW // 16
DQ_BQW48 = 48  # queries per k_dqg96 wave
NQW48 = DQ_BQW48 // 16
DQ_NWAVE = 2  # k_dqg96 waves sharing one K/V ring
DQ_BQW96 = DQ_NWAVE * DQ_BQW48


def _wg_sync():
    """k_dqg96 workgroup sync: this wave's LDS reads retire (dscnt 0), then a split barrier inside
    workgroup release/acquire fences; sched_barrier(0) keeps every DS/TDM op on its side."""
    rocdl.sched_barrier(0)
    rocdl.s_wait_dscnt(0)
    fx.llvm.memory_fence(syncscope="workgroup", ordering=fx.AtomicOrdering.Release)
    rocdl.s_barrier_signal(-1)
    rocdl.s_barrier_wait(-1)
    fx.llvm.memory_fence(syncscope="workgroup", ordering=fx.AtomicOrdering.Acquire)
    rocdl.sched_barrier(0)


VOFF = KV_STEP * XK_ROW_B  # k_dqg ring stage: K [32][D_QK] at +0, V [32][D_V] at +VOFF
KV_B = KV_STEP * (XK_ROW_B + XV_ROW_B)
TDM_OPS_KV = len(pow2_segments(D_QK)) + len(pow2_segments(D_V))  # TDM ops per stage
DQT_TW = TDM_OPS_KV * (TDM_DEPTH - 2)  # tensorcnt that retires all but the newest stage
DQ_IMG_ROW_B = XK_ROW_B  # row stride of the dQ epilogue image
DQ_LINE_EL = 64  # bf16 per 128-B line
# K/V ring tiles split evenly over the k_dqg96 waves, and dQ rows are whole 128-B lines.
assert KV_STEP % DQ_NWAVE == 0 and D_QK % DQ_LINE_EL == 0
assert max(DQ_NWAVE * DQ_BQW48, DQ_BQW) * DQ_IMG_ROW_B <= TDM_DEPTH * KV_B, (
    "dQ images must fit the ring k_dqg allocates"
)
# Carried S/dP A operands in issue order: per (kt, dt) K u0, K u1, then V u0, V u1, the 16-B chunks
# of row kt*16 + row at bytes half*16 + dt*64 + u*32.
_RD_ORDER = [
    (kt, dt, w, u)
    for kt in range(NKT)
    for dt in range(NDT_MAX)
    for w in "kv"
    if dt < (NDT_QK if w == "k" else NDT_V)
    for u in range(2)
]
_RD_IDX = {e: i for i, e in enumerate(_RD_ORDER)}
NP = len(_RD_ORDER)


def _dqg_impl(
    Q,
    K,
    V,
    DO,
    O,
    LSE,
    DEL,
    DQ,
    scale,
    Sq,
    Skv,
    Hq,
    Hkv,
    G,
    nkvt,
    cshift,
    causal,
    q_off,
    ntile,
    nqw,
    nwave,
    HG,
):
    """dQ of 16*nqw queries per wave; `nwave` waves share the K/V ring (k_dqg: 1, k_dqg96: 2).

    The workgroup owns queries [q0g, q0g + nwave*16*nqw), q0g = q_off + bid*16*nqw*nwave; grid
    (HG, ntile, B*Hq/HG). O is not read; dropping it from the kernel arguments would shift the later
    arguments and change the tuned binaries."""
    bqw = 16 * nqw
    bqwg = bqw * nwave
    if const_expr(nwave == 1):
        lane = fx.Int32(fx.thread_idx.x)
    else:
        _tid = fx.Int32(fx.thread_idx.x)
        lane = _tid % fx.Int32(WAVE)
        wv = fx.Int32(rocdl.readfirstlane(fx.Int32.ir_type, (_tid // fx.Int32(WAVE)).ir_value()))
    # Workgroups go to the 8 XCDs round-robin, so x -> (x%8)*(HG/8) + x/8 keeps adjacent q heads on
    # one XCD (a bijection when HG % 8 == 0, else the identity).
    _nx = fx.Int32(8)
    _gx = fx.Int32(fx.block_idx.x)
    qh = (HG % _nx == fx.Int32(0)).select((_gx % _nx) * (HG // _nx) + _gx // _nx, _gx)
    # longest first: query tiles descend on grid.y
    bid = ntile - fx.Int32(1) - fx.Int32(fx.block_idx.y)
    # z*HG + (remapped x) is the batch-major q head bat*Hq + qh
    _hb = fx.Int32(fx.block_idx.z) * HG + qh
    bat = _hb // Hq
    qh = _hb - bat * Hq
    row = lane % fx.Int32(16)
    half = lane // fx.Int32(16)
    q0g = q_off + bid * fx.Int32(bqwg)
    if const_expr(nwave == 1):
        q0 = q0g
    else:
        q0 = q0g + wv * fx.Int32(bqw)
    hkv = qh // G

    # Without a batch argument the descriptors take fixed 1 GiB / 256 MiB extents; the host keeps
    # every tensor inside them.
    g_q = _bv(Q, 1 << 30, fx.BFloat16, 8)
    g_do = _bv(DO, 1 << 30, fx.BFloat16, 8)
    g_lse = _bv(LSE, 1 << 28, fx.Float32)
    g_del = _bv(DEL, 1 << 28, fx.Float32)
    g_dq8 = _bv(DQ, 1 << 30, fx.BFloat16, 8)

    rs_q = Hq * fx.Int32(DV8_QK)  # vectors between consecutive q rows
    base_q = bat * Sq * rs_q + qh * fx.Int32(DV8_QK)
    rs_o = Hq * fx.Int32(DV8_V)
    base_o = bat * Sq * rs_o + qh * fx.Int32(DV8_V)
    base_l = (bat * Hq + qh) * Sq

    smem = fx.SharedAllocator().allocate(TDM_DEPTH * KV_B)
    _lds0 = fx.Int32(fx.ptrtoint(smem.peek().ptr))
    v8b = fx.Vector.make_type(8, fx.BFloat16)
    v8f = fx.Vector.make_type(8, fx.Float32)

    def gfrag(buf, base, rs, r, dt):
        t = base + (r + row) * rs + half + fx.Int32(dt * 4)
        return _ldv(buf, t, fx.BFloat16, 8).shuffle(
            _ldv(buf, t + fx.Int32(2), fx.BFloat16, 8), list(range(16))
        )

    qf = [
        [gfrag(g_q, base_q, rs_q, q0 + fx.Int32(qh_ * 16), dt) for dt in range(NDT_QK)] for qh_ in range(nqw)
    ]
    dof = [
        [gfrag(g_do, base_o, rs_o, q0 + fx.Int32(qh_ * 16), dt) for dt in range(NDT_V)] for qh_ in range(nqw)
    ]
    q_glob = [q0 + fx.Int32(qh_ * 16) + row for qh_ in range(nqw)]
    lse_q = [_ld1(g_lse, base_l + q_glob[qh_], fx.Float32) for qh_ in range(nqw)]
    del_q = [_ld1(g_del, base_l + q_glob[qh_], fx.Float32) for qh_ in range(nqw)]
    # softmax row constants: p = exp2(s*scale*LOG2E - lse*LOG2E), ds = p * (dp*scale - delta*scale)
    _c1q = scale * fx.Float32(LOG2E)
    _nlq = [lse_q[qh_] * fx.Float32(-LOG2E) for qh_ in range(nqw)]
    _ndq = [del_q[qh_] * (fx.Float32(0.0) - scale) for qh_ in range(nqw)]
    lane_r = (lane // fx.Int32(16)) * fx.Int32(8) + lane % fx.Int32(8)
    lane_c = ((lane // fx.Int32(8)) % fx.Int32(2)) * fx.Int32(8)
    # Lane parts of the DS addresses; every per-op offset folds into the DS immediate.
    lb_tr = lane_r * fx.Int32(XK_ROW_B) + lane_c * fx.Int32(2)  # dQ B operand (K) tr16
    lb_rd_k = row * fx.Int32(XK_ROW_B) + half * fx.Int32(16)  # S/dP A readback, K rows
    lb_rd_v = row * fx.Int32(XV_ROW_B) + half * fx.Int32(16)  # S/dP A readback, V rows

    _lds_bf_ty = fx.PointerType.get(
        elem_ty=fx.BFloat16.ir_type, address_space=fx.AddressSpace.Shared, alignment=16
    )
    _kv_rs_k = Hkv * fx.Int32(D_QK)  # elements between consecutive k rows
    _kv_rs_v = Hkv * fx.Int32(D_V)

    def _tdm_kv(kv0p, stage_off):
        """K then V rows [kv0p, kv0p + 32) of (bat, hkv) into a ring stage; Skv - kv0p >= 32 rows in
        bounds. With nwave > 1 each wave moves its KV_STEP/nwave rows of every op."""
        row0 = fx.Int64(bat * Skv + kv0p) * fx.Int64(Hkv) + fx.Int64(hkv)
        off_k = row0 * fx.Int64(D_QK)
        off_v = row0 * fx.Int64(D_V)
        valid = Skv - kv0p
        lb_k = _lds0 + stage_off
        lb_v = _lds0 + stage_off + fx.Int32(VOFF)
        _tdm_rows(K, off_k, D_QK, KV_STEP, valid, _kv_rs_k, lb_k, _lds_bf_ty, nwave)
        _tdm_rows(V, off_v, D_V, KV_STEP, valid, _kv_rs_v, lb_v, _lds_bf_ty, nwave)

    def _rdkv(stage_off):
        """NP ds_load_b128 in _RD_ORDER: K/V row kt*16 + row, bytes half*16 + dt*64 + u*32."""
        rb_k = _lds0 + stage_off + lb_rd_k
        rb_v = _lds0 + stage_off + lb_rd_v
        out = []
        for kt, dt, w, u in _RD_ORDER:
            if w == "k":
                a = rb_k + fx.Int32(kt * 16 * XK_ROW_B + dt * 64 + u * 32)
            else:
                a = rb_v + fx.Int32(VOFF + kt * 16 * XV_ROW_B + dt * 64 + u * 32)
            out.append(fx.Vector(llvm.load(v8b, lds_ptr(a))))
        return out

    def _bks(cur):
        """dQ B operands: K^T of this stage, NDO_QK v16 (kv rows 0..15 | 16..31)."""
        tb = _lds0 + cur + lb_tr
        out = []
        for dtile in range_constexpr(NDO_QK):
            base = tb + fx.Int32(dtile * 32)
            out.append(
                fx.Vector(rocdl.ds_load_tr16_b128(v8b, lds_ptr(base))).shuffle(
                    fx.Vector(rocdl.ds_load_tr16_b128(v8b, lds_ptr(base + fx.Int32(16 * XK_ROW_B)))),
                    list(range(16)),
                )
            )
        return out

    def _smx(sv, pv_, qh_, kt, kv0, do_mask):
        """softmax/dS of one (kt, qh) -> 8 bf16."""
        tt = [fx.Float32(fx.fma(sv[si], _c1q, _nlq[qh_])) for si in range(8)]
        if const_expr(do_mask):
            tt = [
                (
                    (kv0 + fx.Int32(kt * 16) + half * fx.Int32(8) + fx.Int32(si) > q_glob[qh_] + cshift)
                    & (causal != fx.Int32(0))
                ).select(fx.Float32(NEG), tt[si])
                for si in range(8)
            ]
        pf = [_exp2(tt[si]) for si in range(8)]
        return [(pf[si] * fx.Float32(fx.fma(pv_[si], scale, _ndq[qh_]))).to(fx.BFloat16) for si in range(8)]

    def _sgb(n):
        # {1 WMMA, 2 VALU, 1 TRANS} per WMMA
        for _ in range_constexpr(n):
            rocdl.sched_group_barrier(0x008, 1, 0)
            rocdl.sched_group_barrier(0x002, 2, 0)
            rocdl.sched_group_barrier(0x400, 1, 0)

    def _sdp(pre, kt, dt, s_acc, p_acc):
        """k-step dt of S^T = K Q^T (dt < NDT_QK) and dP^T = V dO^T (dt < NDT_V), every qh."""
        if const_expr(dt < NDT_QK):
            kfr = pre[_RD_IDX[(kt, dt, "k", 0)]].shuffle(pre[_RD_IDX[(kt, dt, "k", 1)]], list(range(16)))
        if const_expr(dt < NDT_V):
            vfr = pre[_RD_IDX[(kt, dt, "v", 0)]].shuffle(pre[_RD_IDX[(kt, dt, "v", 1)]], list(range(16)))
        for qh_ in range_constexpr(nqw):
            if const_expr(dt < NDT_QK):
                s_acc[qh_] = rocdl.wmma_f32_16x16x32_bf16(
                    v8f, _ir(kfr), _ir(qf[qh_][dt]), s_acc[qh_], reuseA=False, reuseB=False
                ).result
            if const_expr(dt < NDT_V):
                p_acc[qh_] = rocdl.wmma_f32_16x16x32_bf16(
                    v8f, _ir(vfr), _ir(dof[qh_][dt]), p_acc[qh_], reuseA=False, reuseB=False
                ).result

    def _body_full(acc, pre, kv0, pf_kv0, cur, nxo, ncur):
        # Full loop: kt0 S/dP WMMAs; kt1 S/dP in NDT_MAX chunks {dtile j's WMMAs + softmax/dS of
        # (kt0, qh j)}; the DS phase; softmax/dS (kt1, qh 0); dQ in chunks {qh j's WMMAs +
        # softmax/dS of (kt1, qh j + 1)}. Chunks end in sched_barrier(0).
        rocdl.sched_barrier(0)
        _tdm_kv(pf_kv0, nxo)
        rocdl.sched_barrier(0)
        if const_expr(nwave > 1):
            # The dQ B operands as one tr16 burst beside the TDM, so it drains under the S/dP WMMAs
            # instead of at the barrier. The wait retires the carried readback `pre` first.
            rocdl.s_wait_dscnt(0)
            b_ks = _bks(cur)
            rocdl.sched_barrier(0)
        sp_acc = []
        ds_halves = [[] for _ in range_constexpr(nqw)]
        for kt in range_constexpr(NKT):
            s_acc = [_ir(fx.Vector.filled(8, 0.0, fx.Float32)) for _ in range_constexpr(nqw)]
            p_acc = [_ir(fx.Vector.filled(8, 0.0, fx.Float32)) for _ in range_constexpr(nqw)]
            for dt in range_constexpr(NDT_MAX):
                _sdp(pre, kt, dt, s_acc, p_acc)
                if const_expr(kt == 1):
                    if const_expr(dt < nqw):
                        s0, p0 = sp_acc[0]
                        ds_halves[dt].append(_smx(fx.Vector(s0[dt]), fx.Vector(p0[dt]), dt, 0, kv0, False))
                    _sgb(nqw * (int(dt < NDT_QK) + int(dt < NDT_V)))
                    rocdl.sched_barrier(0)
            if const_expr(kt == 0):
                rocdl.sched_barrier(0)
            sp_acc.append((s_acc, p_acc))
        s1, p1 = sp_acc[1]
        rocdl.sched_barrier(0)
        if const_expr(nwave == 1):
            b_ks = _bks(cur)
            rocdl.sched_barrier(0)
            tdm_ops.tensor_wait(DQT_TW)
            rocdl.sched_barrier(0)
            rb = _rdkv(ncur)
            rocdl.sched_barrier(0)
            ds_halves[0].append(_smx(fx.Vector(s1[0]), fx.Vector(p1[0]), 0, 1, kv0, False))
            rocdl.sched_barrier(0)
        else:
            # softmax/dS (kt1, qh 0), then this wave's half of stage ncur retires and the workgroup
            # syncs: both halves landed and every wave's reads of stage cur retired. Then the readback.
            ds_halves[0].append(_smx(fx.Vector(s1[0]), fx.Vector(p1[0]), 0, 1, kv0, False))
            rocdl.sched_barrier(0)
            tdm_ops.tensor_wait(DQT_TW)
            _wg_sync()
            rb = _rdkv(ncur)
            rocdl.sched_barrier(0)
        new = [None] * (nqw * NDO_QK)
        for qh_ in range_constexpr(nqw):
            a_ds = fx.Vector.from_elements(ds_halves[qh_][0] + ds_halves[qh_][1], dtype=fx.BFloat16)
            for dtile in range_constexpr(NDO_QK):
                # dQ^T = K^T dS^T: a C fragment holds 8 consecutive d of one query row
                new[qh_ * NDO_QK + dtile] = rocdl.wmma_f32_16x16x32_bf16(
                    v8f, _ir(b_ks[dtile]), _ir(a_ds), acc[qh_ * NDO_QK + dtile], reuseA=False, reuseB=False
                ).result
            if const_expr(qh_ + 1 < nqw):
                ds_halves[qh_ + 1].append(
                    _smx(fx.Vector(s1[qh_ + 1]), fx.Vector(p1[qh_ + 1]), qh_ + 1, 1, kv0, False)
                )
                _sgb(NDO_QK)
            rocdl.sched_barrier(0)
        return new + [_ir(v) for v in rb]

    def _body(acc, pre, kv0, do_mask, pf_kv0, cur, nxo, ncur):
        if const_expr(not do_mask):
            return _body_full(acc, pre, kv0, pf_kv0, cur, nxo, ncur)
        # Masked loop: S/dP WMMAs | DS phase | softmax (hides the DS latency) + dQ. The TDM of tile
        # i+2 goes into stage (i-1)%3, whose last reads were consumed by iteration i-1's WMMAs.
        rocdl.sched_barrier(0)
        _tdm_kv(pf_kv0, nxo)
        rocdl.sched_barrier(0)
        sp_acc = []
        for kt in range_constexpr(NKT):
            s_acc = [_ir(fx.Vector.filled(8, 0.0, fx.Float32)) for _ in range_constexpr(nqw)]
            p_acc = [_ir(fx.Vector.filled(8, 0.0, fx.Float32)) for _ in range_constexpr(nqw)]
            for dt in range_constexpr(NDT_MAX):
                _sdp(pre, kt, dt, s_acc, p_acc)
            sp_acc.append((s_acc, p_acc))

        def _softmax():
            ds_halves = [[] for _ in range_constexpr(nqw)]
            for kt in range_constexpr(NKT):
                s_acc, p_acc = sp_acc[kt]
                for qh_ in range_constexpr(nqw):
                    sv, pv_ = fx.Vector(s_acc[qh_]), fx.Vector(p_acc[qh_])
                    ds_halves[qh_].append(_smx(sv, pv_, qh_, kt, kv0, do_mask))
            return ds_halves

        # One DS phase: the dQ B operands (tr16 of this stage's K), then retire the older of the two
        # stages in flight (TDM retires in order) and read it back.
        rocdl.sched_barrier(0)
        b_ks = _bks(cur)
        rocdl.sched_barrier(0)
        if const_expr(nwave == 1):
            tdm_ops.tensor_wait(DQT_TW)
            rocdl.sched_barrier(0)
            rb = _rdkv(ncur)
            rocdl.sched_barrier(0)
            ds_halves = _softmax()
        else:
            # the softmax covers the tr16 latency before the drain and sync
            ds_halves = _softmax()
            rocdl.sched_barrier(0)
            tdm_ops.tensor_wait(DQT_TW)
            _wg_sync()
            rb = _rdkv(ncur)
            rocdl.sched_barrier(0)
        a_ds = [
            fx.Vector.from_elements(ds_halves[qh_][0] + ds_halves[qh_][1], dtype=fx.BFloat16)
            for qh_ in range_constexpr(nqw)
        ]
        new = [None] * (nqw * NDO_QK)
        for qh_ in range_constexpr(nqw):
            for dtile in range_constexpr(NDO_QK):
                new[qh_ * NDO_QK + dtile] = rocdl.wmma_f32_16x16x32_bf16(
                    v8f,
                    _ir(b_ks[dtile]),
                    _ir(a_ds[qh_]),
                    acc[qh_ * NDO_QK + dtile],
                    reuseA=False,
                    reuseB=False,
                ).result
        return new + [_ir(v) for v in rb]

    nacc = nqw * NDO_QK

    def _mkloop(do_mask):
        @flyc.jit
        def kvloop(state, n, i0, nlast):
            # carried: accumulators, the readback A operands, cur (ring offset of stage i%3)
            final = state
            for it, carried in range(fx.Index(fx.Int32(0)), fx.Index(n), 1, init=state):
                st = list(carried)
                cur = fx.Int32(st[-1])
                ii = fx.Int32(it) + i0
                kk = ii + fx.Int32(TDM_DEPTH - 1)
                kk = (kk < nlast).select(kk, nlast)  # min(i + 2, N - 1)
                nxo = fx.Int32(
                    (cur == fx.Int32(0)).select(  # (i + 2) % 3 == (i - 1) % 3
                        fx.Int32((TDM_DEPTH - 1) * KV_B), cur - fx.Int32(KV_B)
                    )
                )
                ncur = fx.Int32(
                    (cur == fx.Int32((TDM_DEPTH - 1) * KV_B)).select(fx.Int32(0), cur + fx.Int32(KV_B))
                )  # (i + 1) % 3
                res = _body(
                    st[:nacc],
                    [fx.Vector(v) for v in st[nacc : nacc + NP]],
                    ii * fx.Int32(KV_STEP),
                    do_mask,
                    kk * fx.Int32(KV_STEP),
                    cur,
                    nxo,
                    ncur,
                )
                final = yield res + [fx.as_ir_value(ncur)]
            return final

        return kvloop

    kvloop_full = _mkloop(False)
    kvloop_mask = _mkloop(True)

    # kv range of the workgroup's queries: every wave runs the same nfull full + (nkvt_eff - nfull)
    # masked steps, so the barrier counts match; do_mask zeroes the rows past a wave's own diagonal.
    _lim = (q0g + fx.Int32(bqwg) + cshift + fx.Int32(KV_STEP - 1)) // fx.Int32(KV_STEP)
    _lim = (_lim < fx.Int32(1)).select(fx.Int32(1), _lim)
    _lim = (_lim < nkvt).select(_lim, nkvt)
    nkvt_eff = (causal != fx.Int32(0)).select(_lim, nkvt)
    _t = q0g + cshift + fx.Int32(1)
    _nf = (_t < fx.Int32(0)).select(fx.Int32(0), _t // fx.Int32(KV_STEP))
    _nf = (_nf < nkvt_eff).select(_nf, nkvt_eff)
    nfull = (causal != fx.Int32(0)).select(_nf, nkvt_eff)
    nlast = nkvt_eff - fx.Int32(1)

    # prologue: stage s gets tile min(s, N - 1) for s = 0, 1; retire stage 0 and read it back
    _tdm_kv(fx.Int32(0), fx.Int32(0))
    for s_ in range_constexpr(1, TDM_DEPTH - 1):
        t1 = (fx.Int32(s_) < nlast).select(fx.Int32(s_), nlast)
        _tdm_kv(t1 * fx.Int32(KV_STEP), fx.Int32(s_ * KV_B))
    rocdl.sched_barrier(0)
    tdm_ops.tensor_wait(DQT_TW)
    if const_expr(nwave > 1):
        _wg_sync()  # both halves of stage 0 landed
    rocdl.sched_barrier(0)
    rb0 = _rdkv(fx.Int32(0))

    init = [_ir(fx.Vector.filled(8, 0.0, fx.Float32)) for _ in range(nacc)]
    out = kvloop_full(init + [_ir(v) for v in rb0] + [fx.as_ir_value(fx.Int32(0))], nfull, fx.Int32(0), nlast)
    out = kvloop_mask(list(out), nkvt_eff - nfull, nfull, nlast)
    # retire the last two (clamped, unread) prefetches: the LDS must not be written after release
    tdm_ops.tensor_wait(0)
    # Epilogue: stage the dQ^T fragments as a row-major image in the dead ring and store it back as
    # whole 128-B lines. nwave > 1: sync first, so no image overwrites a stage another wave reads.
    if const_expr(nwave > 1):
        _wg_sync()
        img = _lds0 + wv * fx.Int32(bqw * DQ_IMG_ROW_B)
    else:
        img = _lds0
    for qh_ in range_constexpr(nqw):
        for dtile in range_constexpr(NDO_QK):
            ov = fx.Vector(out[qh_ * NDO_QK + dtile])
            # lane (row, half) holds query row qh_*16 + row, d = dtile*16 + half*8 + [0, 8)
            llvm.store(
                fx.as_ir_value(
                    fx.Vector.from_elements(
                        [ov[si].to(fx.BFloat16) for si in range_constexpr(8)], dtype=fx.BFloat16
                    )
                ),
                lds_ptr(
                    img
                    + (fx.Int32(qh_ * 16) + row) * fx.Int32(DQ_IMG_ROW_B)
                    + fx.Int32(dtile * 32)
                    + half * fx.Int32(16)
                ),
            )
    lrow = lane // fx.Int32(8)  # 4 rows per store
    lcol = lane % fx.Int32(8)  # 16-B chunk of the 128-B line
    rd0 = img + lrow * fx.Int32(DQ_IMG_ROW_B) + lcol * fx.Int32(16)
    for rb in range_constexpr(bqw // 4):
        # vector index of (query row q0 + 4rb + lrow, head qh, d = 0) in dq
        gt = ((bat * Sq + q0 + fx.Int32(rb * 4) + lrow) * Hq + qh) * fx.Int32(DV8_QK) + lcol
        for li in range_constexpr(D_QK // DQ_LINE_EL):
            vv = fx.Vector(llvm.load(v8b, lds_ptr(rd0 + fx.Int32(rb * 4 * DQ_IMG_ROW_B + li * 128))))
            _stv(
                [vv[e] for e in range_constexpr(8)], g_dq8, gt + fx.Int32(li * (DQ_LINE_EL // 8)), fx.BFloat16
            )


@flyc.kernel(known_block_size=[32, 1, 1])
def k_dqg(
    Q: fx.Tensor,
    K: fx.Tensor,
    V: fx.Tensor,
    DO: fx.Tensor,
    O: fx.Tensor,
    LSE: fx.Tensor,
    DEL: fx.Tensor,
    DQ: fx.Tensor,
    scale: fx.Float32,
    Sq: fx.Int32,
    Skv: fx.Int32,
    Hq: fx.Int32,
    Hkv: fx.Int32,
    G: fx.Int32,
    nkvt: fx.Int32,
    cshift: fx.Int32,
    causal: fx.Int32,
    q_off: fx.Int32,
    ntile: fx.Int32,
    HG: fx.Int32,
):
    _dqg_impl(
        Q,
        K,
        V,
        DO,
        O,
        LSE,
        DEL,
        DQ,
        scale,
        Sq,
        Skv,
        Hq,
        Hkv,
        G,
        nkvt,
        cshift,
        causal,
        q_off,
        ntile,
        NQW,
        1,
        HG,
    )


@flyc.jit
def launch_dqg(
    Q,
    K,
    V,
    DO,
    O,
    LSE,
    DEL,
    DQ,
    scale: fx.Float32,
    Sq: fx.Int32,
    Skv: fx.Int32,
    Hq: fx.Int32,
    Hkv: fx.Int32,
    G: fx.Int32,
    nkvt: fx.Int32,
    cshift: fx.Int32,
    causal: fx.Int32,
    q_off: fx.Int32,
    ntile: fx.Int32,
    hg: fx.Int32,
    ngz: fx.Int32,
    stream: fx.Stream,
):
    _ = SOURCE_DIGEST  # FlyDSL disk-cache key
    # grid = (hg q heads of a head group, ntile 32-query tiles from q_off, B*Hq/hg head groups)
    k_dqg(
        Q, K, V, DO, O, LSE, DEL, DQ, scale, Sq, Skv, Hq, Hkv, G, nkvt, cshift, causal, q_off, ntile, hg
    ).launch(grid=(hg, ntile, ngz), block=(32, 1, 1), stream=stream)


@flyc.kernel(known_block_size=[DQ_NWAVE * WAVE, 1, 1])
def k_dqg96(
    Q: fx.Tensor,
    K: fx.Tensor,
    V: fx.Tensor,
    DO: fx.Tensor,
    O: fx.Tensor,
    LSE: fx.Tensor,
    DEL: fx.Tensor,
    DQ: fx.Tensor,
    scale: fx.Float32,
    Sq: fx.Int32,
    Skv: fx.Int32,
    Hq: fx.Int32,
    Hkv: fx.Int32,
    G: fx.Int32,
    nkvt: fx.Int32,
    cshift: fx.Int32,
    causal: fx.Int32,
    q_off: fx.Int32,
    ntile: fx.Int32,
    HG: fx.Int32,
):
    rocdl.disable_xdl_arb_stall()  # no WMMA multicycle arbitration stall (SCHED_MODE)
    _dqg_impl(
        Q,
        K,
        V,
        DO,
        O,
        LSE,
        DEL,
        DQ,
        scale,
        Sq,
        Skv,
        Hq,
        Hkv,
        G,
        nkvt,
        cshift,
        causal,
        q_off,
        ntile,
        NQW48,
        DQ_NWAVE,
        HG,
    )


@flyc.jit
def launch_dqg96(
    Q,
    K,
    V,
    DO,
    O,
    LSE,
    DEL,
    DQ,
    scale: fx.Float32,
    Sq: fx.Int32,
    Skv: fx.Int32,
    Hq: fx.Int32,
    Hkv: fx.Int32,
    G: fx.Int32,
    nkvt: fx.Int32,
    cshift: fx.Int32,
    causal: fx.Int32,
    q_off: fx.Int32,
    ntile: fx.Int32,
    hg: fx.Int32,
    ngz: fx.Int32,
    stream: fx.Stream,
):
    _ = SOURCE_DIGEST  # FlyDSL disk-cache key
    # grid = (hg q heads of a head group, ntile 96-query tiles from q_off, B*Hq/hg head groups)
    k_dqg96(
        Q, K, V, DO, O, LSE, DEL, DQ, scale, Sq, Skv, Hq, Hkv, G, nkvt, cshift, causal, q_off, ntile, hg
    ).launch(grid=(hg, ntile, ngz), block=(DQ_NWAVE * WAVE, 1, 1), stream=stream)
