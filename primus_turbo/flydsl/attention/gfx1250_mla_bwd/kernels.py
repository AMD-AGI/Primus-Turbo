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

"""gfx1250 FlyDSL flash-attention backward: k_delta, k_dkdv, k_dqg (wave32 WMMA, bf16).

    delta[b,h,s] = sum_d dO*O                                   k_delta_bshd
    dV[kv,:] = sum_q P^T dO ;  dK[kv,:] = sum_q dS^T Q           k_dkdv64 (two waves per 64-row kv
                                                                 block, one shared Q/dO ring)
    dQ[q,:]  = sum_kv dS K                                       k_dqg96 (two waves x 48 queries
                                                                 share one K/V ring) + k_dqg
                                                                 (32-query head tiles)

Head dims: D_QK for q/k/dq/dk (S = Q K^T contracts D_QK), D_V for v/o/do/dv (dP = dO V^T
contracts D_V); DeepSeek-V3 MLA is D_QK = 192 (128 nope + 64 rope), D_V = 128.
Layouts: q [B, Sq, Hq, D_QK], o/do [B, Sq, Hq, D_V], k [B, Skv, Hkv, D_QK], v [B, Skv, Hkv,
D_V], all bf16; lse/delta [B, Hq, Sq] fp32, natural log; dq/dk/dv bf16 like q/k/v. Causal is
bottom-right (query i sees keys j <= i + Skv - Sq). GQA is reduced inside k_dkdv (the G q
heads of one kv head stream through the same accumulators): no atomics, every output
element written once, deterministic.

k_dkdv stages Q/dO through a 3-stage Tensor-Data-Mover LDS ring (prefetch two iterations
ahead) and reads the next iteration's S/dP B operands back into VGPRs one iteration early;
k_dqg does the same with a 3-stage K/V ring. k_delta, k_dqg and the legacy one-wave k_dkdv
run one wave32 per workgroup with no s_barrier; k_dkdv64 runs two waves (one per SIMD) over ONE
Q/dO ring and synchronises it with one workgroup barrier per full-loop iteration (protocol at
_dkdv_impl); k_dqg96 runs two waves on one K/V ring (each TDMs half of every tile) with one
workgroup barrier per kv step (_wg_sync). The design history behind every choice is in
PROVENANCE.md.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir as _mlir_ir  # UNSTABLE(gfx1250): gpu.barrier memfence attribute
from flydsl._mlir.dialects import llvm as llvm_dialect  # UNSTABLE(gfx1250): LDS load/store
from flydsl.expr import const_expr, range_constexpr, rocdl
from flydsl.expr.rocdl import tdm_ops  # UNSTABLE(gfx1250): tensor_wait


def _module_knobs():
    """Every module-level knob of this file -- upper-case scalars and tuples of scalars -- as
    sorted (name, value) pairs; MODULE_KNOBS (end of file) is this, taken at import."""
    g = globals()

    def knob(v):
        scalar = (bool, int, float, str)
        return isinstance(v, scalar) or (isinstance(v, tuple) and all(isinstance(x, scalar) for x in v))

    return tuple(
        (n, g[n])
        for n in sorted(g)
        if n.isupper() and not n.startswith("_") and n != "MODULE_KNOBS" and knob(g[n])
    )


def create_llvm_ptr(value, address_space=1):
    """Raw LLVM pointer from an i32 address (1 = global, 3 = LDS); aiter kernels_common's."""
    space = {1: fx.AddressSpace.Global, 3: fx.AddressSpace.Shared}.get(address_space, address_space)
    pt = fx.PointerType.get(fx.Int32.ir_type, address_space=space, alignment=4)
    return fx.as_ir_value(fx.to_llvm_ptr(fx.inttoptr(pt, value)))


def _ir(v):
    """A typed flydsl value as a raw ir.Value (aiter tensor_shim._to_raw)."""
    return fx.as_ir_value(v)


D_QK = 192  # q/k head dim
D_V = 128  # v/o head dim
DV8_QK = D_QK // 8  # vec8 tiles per row: q, k, dq, dk
DV8_V = D_V // 8  # vec8 tiles per row: v, o, do, dv
NDT_QK = D_QK // 32  # WMMA k-steps contracting S = K Q^T
NDT_V = D_V // 32  # WMMA k-steps contracting dP = V dO^T
NDT_MAX = max(NDT_QK, NDT_V)
NDO_QK = D_QK // 16  # 16-wide output tiles across d: dK, dQ
NDO_V = D_V // 16  # 16-wide output tiles across d: dV
NDO_MAX = max(NDO_QK, NDO_V)
SAME_D = D_QK == D_V  # one shared address family when the row strides agree
NKV = 2  # 16-row kv sub-tiles per k_dkdv workgroup
NST = NKV * (NDO_V + NDO_QK)  # dV then dK accumulators carried by k_dkdv
WAVE = 32  # gfx1250 dispatches wave32
LOG2E = 1.4426950408889634
NEG = -3.0e38
TDM_DEPTH = 3  # Q/dO (k_dkdv) and K/V (k_dqg) LDS ring stages
BLOCK_KV = 32  # kv rows one k_dkdv workgroup owns
S_ROW_B = BLOCK_KV * 2 + 16  # the former padded P/dS LDS row (80 B); unused since r5.i3.g17
# Padded LDS rows of the TDM-staged tiles: D*2 + 16 bytes. An unpadded 256 B row puts every
# row on bank 0 (64-way conflict on the 16 rows of a tr16 read); +16 B makes row r start at
# dword 4r (D=128, 68-dword stride) or 36r (D=192, 100-dword stride) mod 64 -- 16 distinct
# 4-bank groups either way.
XK_ROW_B = D_QK * 2 + 16  # q / k rows (400 B at D_QK = 192)
XV_ROW_B = D_V * 2 + 16  # do / v rows (272 B at D_V = 128)


def _pow2_segments(d):
    """[(col0, width)] covering [0, d) with power-of-two widths, largest first. A TDM pad
    interval must be a power-of-two number of dwords, so a 192-wide row takes two ops
    (128 + 64), each padded to the same LDS row stride."""
    segs, c0 = [], 0
    while c0 < d:
        w = 1 << ((d - c0).bit_length() - 1)
        segs.append((c0, w))
        c0 += w
    return segs


for _d in (D_QK, D_V):
    assert _d % 32 == 0, _d
    for _c0, _w in _pow2_segments(_d):
        # descriptor fields: pad_interval = log2(w/2 dwords) - 1 in 3 bits, pad_amount =
        # (row - w)/2 dwords - 1 in 7 bits (flydsl tdm_ops.compute_padding_encoding)
        assert 4 <= _w <= 512 and 1 <= (_d + 8 - _w) // 2 <= 128, (_d, _w)

# k_dkdv LDS: the Q/dO ring in segment 0 -- per stage the [32][D_V] dO tile at +0 and the
# [32][D_QK] Q tile at +QOFF -- and the P/dS tiles one 64 KB segment up (LDS_SEG).
LDS_SEG = 65536
QOFF = 32 * XV_ROW_B
QDO_B = 32 * (XV_ROW_B + XK_ROW_B)  # 21504 B at 192/128 (17408 at 128/128)
TDM_OPS_QDO = len(_pow2_segments(D_V)) + len(_pow2_segments(D_QK))  # TDM ops per stage
TW_QDO = TDM_OPS_QDO * (TDM_DEPTH - 2)  # tensorcnt that retires all but the newest stage
NQE = 2 * NDT_QK  # carried Q readback entries per 16-query half (u = 0, 1)
NHB = NQE + 2 * NDT_V  # + dO entries
NLD = 8  # carried NL/ND vec4 entries per iteration: 2 hh x {NL, ND} x 2 (r5.i3.g17)
DK0 = NKV * NDO_V  # dK accumulators follow the dV ones
# k_dkdv64 epilogue (bwd_r9_a): per wave a ROW-major dV image [32 kv][D_V] at row stride XV_ROW_B at +0
# and a row-major dK image [32 kv][D_QK] at row stride XK_ROW_B at +EPI_DK, read back in whole 128-B
# line order (4 kv rows x one line per ds_load_b128 / buffer_store_b128)
EPI_DK = 32 * XV_ROW_B  # dK image offset in a wave's epilogue area (8704 B)
EPI_W = 32 * XV_ROW_B + 32 * XK_ROW_B  # one wave's dV + dK epilogue images (21504 B)
EPI_LINE_B = 128  # bytes per global line one 8-lane run stores
# k_dkdv64: DKDV_NW waves per workgroup; wave w owns kv rows [kv0g + 32w, kv0g + 32w + 32) of the
# workgroup's DKDV_NW*BLOCK_KV-row block. The waves share ONE Q/dO ring (TDM issued cooperatively,
# num_warps = DKDV_NW: wave w moves rows [16w, 16w + 16) of every 32-row tile); each keeps its own
# epilogue image at w*EPI_W inside the dead ring. r5.i3.g17: P/dS no longer go through LDS (they feed
# the dV/dK WMMAs from the S/dP accumulators), so S_ROW_B / PDS_B size nothing in k_dkdv64 now.
DKDV_NW = 2
PDS_B = 2 * 32 * S_ROW_B  # the former per-wave P + dS tiles (5120 B); unused since r5.i3.g17
BAR_KH = NKV // 2  # k_dkdv64 full loop: the ring barrier sits before kv sub-tile BAR_KH's WMMAs
# k_dkdv64 full loop: the Q/dO prefetch (tile it+3 -> stage it%3) is issued after BARRIER(it)
# behind the readback; its scalar math rides the kh0 WMMA run, KH0_SALU SALU per WMMA
# (sched_group_barrier), and SALU (only) may cross the fence between the readback and the TDM.
KH0_SALU = 2
KB_SALU_MASK = 0x004  # llvm.amdgcn.sched.barrier mask: SALU may be scheduled across
assert TDM_DEPTH * QDO_B <= LDS_SEG, "the Q/dO ring must stay inside LDS segment 0"
assert DKDV_NW * EPI_W <= TDM_DEPTH * QDO_B, "epilogue images must fit the dead ring"
assert (D_V * 2) % EPI_LINE_B == 0 and (D_QK * 2) % EPI_LINE_B == 0, "dV/dK rows are whole 128-B lines"
assert 32 % DKDV_NW == 0 and (DKDV_NW & (DKDV_NW - 1)) == 0, "TDM num_warps splits 32 rows evenly"

# Head-group traversal (bwd_r4_a) of k_dkdv64 / k_dqg96 / k_dqg. Their grids are (heads,
# tiles, batch) with the head fastest, so the dispatch order -- and with it how many kv (q)
# tiles of ONE head are resident at once, i.e. how often a Q/dO (K/V) tile streamed by one
# workgroup is still in L2 for the next -- depends on the head count per launch row. The
# e2e/Turbo fold launches Megatron's b2 h128 SBHD views as [1, s, 256, d]: 256 heads per row,
# half the resident tiles per head of the b2h128 launch over the same bytes (4.39 vs 3.93 ms
# for r3_a). With a head group hg (hg | nh) the launch grid is (hg, tiles, B*nh/hg) -- the
# same workgroup count -- and the kernel decodes (block_idx.z, block_idx.x) as the batch-major
# head index z*hg + x = bat*nh + h. hg = nh is r3_a's grid and mapping exactly; hg = 128 on
# the fold launch IS the b2h128 grid. Every workgroup still does the work of exactly one
# (bat, head, tile) of r3_a (bounds_proof.py R1), so outputs are bitwise identical.
HEAD_GROUP = 128


def head_group(nh, target=HEAD_GROUP):
    """Head group of a launch with nh heads per batch: nh itself (r3_a's grid) when nh <=
    target or target <= 0; else the largest multiple of 8 that divides nh and is <= target
    (a multiple of 8 keeps every head's workgroups on one XCD under the round-robin linear-id
    dispatch: grid.x % 8 == 0), nh when there is none."""
    if target <= 0 or nh <= target:
        return nh
    for d in range(target - target % 8, 7, -8):
        if nh % d == 0:
            return d
    return nh


DELTA_THREADS = 256
LANES_PER_ROW = D_V // 8
ROWS_PER_PASS = DELTA_THREADS // LANES_PER_ROW
ROWS_DELTA = 32
PASSES_PER_WG = ROWS_DELTA // ROWS_PER_PASS


# ---- shared helpers ------------------------------------------------------------------
def _bv(t, nb, dt, vec=1):
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
    """Store a `len(vals)`-wide vector of `dt` at vector-index `tile`. Mirrors _ldv."""
    n = len(vals)
    f = fx.make_rmem_tensor(fx.make_layout(n, 1), dt)
    fx.memref_store_vec(fx.Vector.from_elements(vals, dtype=dt), f)
    fx.copy_atom_call(_atom(dt, n), f, fx.slice(buf, (tile, None)))


def _lds_barrier():
    """Workgroup barrier whose fences cover LDS only (gpu.barrier memfence [workgroup]): the
    release side drains this wave's DS ops (s_wait_dscnt 0) but not its in-flight global
    loads (the LSE/delta prefetch), which a plain gpu.barrier() also waits for."""
    fx.gpu.barrier(address_spaces=_mlir_ir.Attribute.parse("[#gpu.address_space<workgroup>]"))


def _exp2(x):
    return fx.Float32(fx.rocdl.exp2(fx.Float32.ir_type, x.ir_value()))


def _tdm_rows(src, off, d, rows, valid, rs_el, lds_row0, lds_ty, num_warps=1):
    """TDM the [rows][d] bf16 tile at element `off` of `src` (row stride rs_el elements,
    outer extent `valid` rows) into LDS at byte address lds_row0, row stride d*2+16 bytes:
    one TDM op per power-of-two column segment, so len(_pow2_segments(d)) ops. num_warps > 1
    is the cooperative form: every wave of the workgroup issues the same ops and wave w moves
    rows [w*rows/num_warps, (w+1)*rows/num_warps) (FlyDSL splits the outer dim by wave_id; the
    per-wave outer extent clamp is valid - w*rows/num_warps). Used by k_dkdv64 (Q/dO ring) and
    k_dqg96 (K/V ring): each wave's tensorcnt then counts only its own len(_pow2_segments(d))
    ops (FlyDSL GFX1250 TDM lowering: per-wave global/LDS offset from rocdl.wave_id, the fwd's
    num_warps 8 pattern)."""
    x_el = d + 8
    for c0, w in _pow2_segments(d):
        g_off = off if c0 == 0 else off + fx.Int64(c0)
        lb = lds_row0 if c0 == 0 else lds_row0 + fx.Int32(c0 * 2)
        g_view = fx.Tensor(
            fx.make_view(fx.add_offset(fx.get_iter(src), g_off), fx.make_layout((rows, w), (d, 1)))
        )
        atom = fx.rocdl.cdna5.make_tdm_atom(
            g_view,
            [valid, None],
            strides=[rs_el, None],
            num_warps=num_warps,
            pad_interval=w,
            pad_amount=x_el - w,
        )
        l_view = fx.Tensor(fx.make_view(fx.inttoptr(lds_ty, lb), fx.make_layout((rows, w), (x_el, 1))))
        fx.copy_atom_call(atom, g_view, l_view)


# ================================================================== delta ==========
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
    """delta[b, h, s] = sum_d dO[b, s, h, d] * O[b, s, h, d], fp32; and k_dkdv64's per-row softmax
    constants NL = -LOG2E * lse, ND = -scale * delta, [B, H, S] fp32 each (r5.i3.g17): the same two
    fp32 products k_dkdv64 formed per lane before, now read per element (its S/dP C layout puts 8
    query rows in each lane).

    r6.i1.g21: workgroup (block_idx.x, block_idx.y) = (sblk, bh) owns the ROWS_DELTA consecutive
    queries s = sblk*ROWS_DELTA + [0, ROWS_DELTA) of the batch-major head bh = b*H + h, so its outputs
    are ONE contiguous ROWS_DELTA*4 = 128-B line of each [B, H, S] array. (Before: 32 consecutive
    heads of one (b, s), i.e. 32 one-lane 4-B stores S*4 B apart per array.) Each row is still read
    by 16 lanes x 16 B with the same lane -> d mapping, the same two FMA chains and the same 16-lane
    xor reduction, so every row sum is bitwise the incumbent's. The ROWS_DELTA sums meet in LDS and
    wave 0 writes DEL, ND and NL as one 128-B line each."""
    _ = MODULE_KNOBS  # cache key only (see MODULE_KNOBS)
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
    # pass u reads query s0 + u*ROWS_PER_PASS + row_in_group: BSHD row (b*S + s)*H + h, vec8 tiles
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
        for c in range_constexpr(4):  # 8 elements of the lane's vec8, 2 chains
            e0 = e0 + fx.Float32(do8[2 * c]) * fx.Float32(o8[2 * c])
            e1 = e1 + fx.Float32(do8[2 * c + 1]) * fx.Float32(o8[2 * c + 1])
        acc = e0 + e1
        for sft in range_constexpr(LANES_PER_ROW.bit_length() - 1):  # LANES_PER_ROW lanes/row
            acc = acc + fx.gpu.shuffle_xor(acc, 1 << sft, WAVE)
        # the xor butterfly leaves the same sum in all 16 lanes of the row: they all write it
        r = fx.Int32(u * ROWS_PER_PASS) + row_in_group
        llvm_dialect.store(fx.as_ir_value(acc), create_llvm_ptr(lds0 + r * fx.Int32(4), address_space=3))
    _lds_barrier()
    # lane l of every wave reads row l; only wave 0 (tid < ROWS_DELTA) stores -- the other waves'
    # stores go to index n_rows, past every buffer (the incumbent's predication idiom). dst is in
    # bounds for every lane, so the LSE read needs no clamp.
    l32 = tid % fx.Int32(ROWS_DELTA)
    v = fx.Float32(
        llvm_dialect.load(fx.Float32.ir_type, create_llvm_ptr(lds0 + l32 * fx.Int32(4), address_space=3))
    )
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
    # grid = (S/ROWS_DELTA query blocks, B*H batch-major heads) (r6.i1.g21)
    k_delta_bshd(DO, O, DEL, LSE, NL, ND, scale, S, H, n_rows).launch(
        grid=(nsblk, nbh, 1), block=(DELTA_THREADS, 1, 1), stream=stream
    )


# =================================================================== dkdv =========
def _dkdv_impl(
    Q, K, V, DO, NL, ND, DV_, DK, scale, Sq, Skv, Hq, Hkv, G, nqt, cshift, causal, B_, nw=1, HG=None
):
    """One wave owns a BLOCK_KV-row kv tile of one kv head and streams every (q head, q pair).

    nw (compile-time) waves per workgroup. nw = 1: the legacy k_dkdv, grid = (Hkv,
    Skv/BLOCK_KV, B), no barrier. nw = DKDV_NW: k_dkdv64, grid = (Hkv, Skv/(nw*BLOCK_KV), B);
    wave w (rocdl.wave_id, an SGPR) owns kv rows kv0g + w*BLOCK_KV + [0, BLOCK_KV) and every
    wave streams the SAME (q pair, q head) sequence -- the workgroup's -- out of ONE Q/dO ring.
    The accumulators persist across the G q heads that share this kv head, so the GQA
    reduction happens in registers -- atomic-free, written once.

    k_dkdv64 ring protocol. Every wave runs the same instruction stream: the wave index only
    offsets addresses (kv rows, P/dS tiles, epilogue image, its TDM half), no branch depends on
    it, and every trip count comes from the block ids and kernel args, so every wave takes the
    same barriers: 2 per masked iteration + 2 in the prologue + 1 per full iteration + 1 before
    the epilogue (bounds_proof.py K6 replays them). BARRIER = _lds_barrier(): LDS-only
    workgroup release fence (s_wait_dscnt 0: this wave's DS ops retired; TDM retirement is the
    explicit tensor_wait before it) + s_barrier_signal/wait + acquire fence.
      masked iteration: BARRIER (WAR: every wave's reads of stage 0 retired) -> TDM own half of
        the tile into stage 0 -> tensor_wait(0) -> BARRIER (RAW) -> readback/tr16 of stage 0.
      prologue: BARRIER (WAR vs the masked loop) -> TDM stages 0, 1 -> tensor_wait(TW_QDO)
        (own half of stage 0) -> BARRIER (RAW) -> readback of stage 0.
      full iteration ii: TDM into (ii+2)%3 at the top [WAR: its last readers -- readback in
        ii-2, tr16 in ii-1 -- retired before BARRIER(ii-1)]; tr16 of stage ii%3; dK/dV WMMAs of
        kv sub-tiles < BAR_KH; tensor_wait(TW_QDO) (own half of stage (ii+1)%3); BARRIER(ii);
        readback of stage (ii+1)%3 [RAW: every wave's half retired before BARRIER(ii)]; WMMAs
        of kv sub-tiles >= BAR_KH (they hide the readback latency).
      exit: tensor_wait(0) -> BARRIER (every wave's last TDM landed, every ring read retired)
        -> the epilogue images overwrite the ring.
    """
    if const_expr(nw == 1):
        lane = fx.Int32(fx.thread_idx.x)
    else:
        lane = fx.Int32(fx.thread_idx.x) % fx.Int32(WAVE)
    if const_expr(HG is None):
        hkv = fx.Int32(fx.block_idx.x)  # kv head (cheap axis fastest)
        bat = fx.Int32(fx.block_idx.z)  # batch
    else:
        # head group (HEAD_GROUP note): grid (HG, Skv/64, B*Hkv/HG), HG | Hkv (impl._plan);
        # z*HG + x is the batch-major kv head index bat*Hkv + hkv (bounds_proof.py R1)
        _hb = fx.Int32(fx.block_idx.z) * HG + fx.Int32(fx.block_idx.x)
        bat = _hb // Hkv
        hkv = _hb - bat * Hkv
    bid = fx.Int32(fx.block_idx.y)  # kv tile (nw = 1) / kv block (nw > 1)
    row = lane % fx.Int32(16)
    half = lane // fx.Int32(16)
    kv0 = bid * fx.Int32(BLOCK_KV)
    kv0g = kv0  # the workgroup's first kv row (loop bounds)
    if const_expr(nw > 1):
        wv = fx.Int32(rocdl.wave_id())  # wave in the workgroup, [0, nw), SGPR
        kv0g = bid * fx.Int32(nw * BLOCK_KV)
        kv0 = kv0g + wv * fx.Int32(BLOCK_KV)  # this wave's kv tile

    # TRUE byte extents on every k_dkdv descriptor: an over-read returns 0 instead of
    # walking into the next page, and a clamp that hit a LIVE access would crater SQNR.
    nk_b = B_ * Skv * Hkv * fx.Int32(D_QK * 2)  # k / dk bf16 [B, Skv, Hkv, D_QK]
    nv_b = nk_b if SAME_D else B_ * Skv * Hkv * fx.Int32(D_V * 2)  # v / dv [.., D_V]
    nl_b = B_ * Hq * Sq * fx.Int32(4)  # NL / ND fp32 [B, Hq, Sq] (k_delta, r5.i3.g17)
    g_k = _bv(K, nk_b, fx.BFloat16, 8)
    g_v = _bv(V, nv_b, fx.BFloat16, 8)
    g_nl = _bv(NL, nl_b, fx.Float32)
    g_nd = _bv(ND, nl_b, fx.Float32)

    rs_k = Hkv * fx.Int32(DV8_QK)  # vec8 tiles between consecutive k rows
    base_k = bat * Skv * rs_k + hkv * fx.Int32(DV8_QK)
    rs_v = rs_k if SAME_D else Hkv * fx.Int32(DV8_V)
    base_v = base_k if SAME_D else bat * Skv * rs_v + hkv * fx.Int32(DV8_V)

    # LDS: the Q/dO ring at offset 0 (segment 0) and nothing else. r5.i3.g17: P and dS go from
    # the S/dP accumulators straight into the dV/dK WMMAs as A operands (VGPRs), so there are no
    # P/dS tiles; the epilogue images reuse the dead ring.
    smem = fx.SharedAllocator().allocate(max(TDM_DEPTH * QDO_B, nw * EPI_W))
    _lds0 = fx.Int32(fx.ptrtoint(smem.peek().ptr))
    v8b = fx.Vector.make_type(8, fx.BFloat16)
    v8f = fx.Vector.make_type(8, fx.Float32)

    def gfrag(buf, base, rs, r, dt):
        t = base + (r + row) * rs + half + fx.Int32(dt * 4)
        return _ldv(buf, t, fx.BFloat16, 8).shuffle(
            _ldv(buf, t + fx.Int32(2), fx.BFloat16, 8), list(range(16))
        )

    # NL/ND prefetch address (r5.i3.g17): voffset = 32*half (loop-invariant VGPR) + soffset (SGPR).
    rsrc_nl = rocdl.get_buffer_rsrc(fx.get_iter(g_nl))
    rsrc_nd = rocdl.get_buffer_rsrc(fx.get_iter(g_nd))
    voff_l = half * fx.Int32(32)
    v4f = fx.Vector.make_type(4, fx.Float32)
    c1_sm = scale * fx.Float32(LOG2E)  # softmax exp2 multiplier, hoisted (see _body)

    def _ldl(qt, gh):
        """The NLD = 8 vec4 loads of query pair qt, q head hkv*G+gh (r5.i3.g17): per hh, NL u0,
        NL u1, ND u0, ND u1. Lane (row, half) holds the S/dP C elements of q rows hh*16 + half*8 +
        [0, 8), so it reads those 8 rows of NL and of ND (2 x 16 B each). Byte address 4*(base_l +
        q0 + hh*16 + half*8) + 16u split as voffset = 32*half + soffset = 4*(base_l + q0) + 64*hh
        + 16u (bounds_proof.py K3). (qt, gh) are wave-uniform: one form for both loops; in the
        full loop the loads are carried one iteration early."""
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

    # Q/dO staging through the Tensor Data Mover into a TDM_DEPTH-stage LDS ring. Stage s
    # is [s*QDO_B, (s+1)*QDO_B): the [32 q rows][D_V] dO tile at +0 (row stride XV_ROW_B),
    # the [32][D_QK] Q tile at +QOFF (row stride XK_ROW_B); element c of a row at byte 2c.
    # Each tile is TDM_OPS_QDO-many ops in total, one per power-of-two column segment
    # (_tdm_rows). Tile origin element ((bat*Sq + q0)*Hq + qh)*D, outer stride Hq*D, outer
    # extent Sq - q0 (>= 32 for every issued tile: Sq % 64 == 0 is asserted by impl.py and
    # qt is clamped to [0, nqt2-1]).
    _lds_bf_ty = fx.PointerType.get(
        elem_ty=fx.BFloat16.ir_type, address_space=fx.AddressSpace.Shared, alignment=16
    )
    _q_rs_v = Hq * fx.Int32(D_V)  # elements between consecutive do rows
    _q_rs_k = _q_rs_v if SAME_D else Hq * fx.Int32(D_QK)

    def _tdm_qdo_prep(qt, gh, stage_off):
        """The scalar address math of one stage's TDM (emitted where it is called)."""
        qh = hkv * G + gh
        q0 = qt * fx.Int32(32)
        row0 = fx.Int64(bat * Sq + q0) * fx.Int64(Hq) + fx.Int64(qh)
        off_v = row0 * fx.Int64(D_V)
        off_k = off_v if SAME_D else row0 * fx.Int64(D_QK)
        valid = Sq - q0
        lb_v = _lds0 + stage_off
        lb_k = _lds0 + stage_off + fx.Int32(QOFF)
        return (off_v, off_k, valid, lb_v, lb_k)

    def _tdm_qdo_issue(p):
        off_v, off_k, valid, lb_v, lb_k = p
        # nw > 1: cooperative, wave w moves rows [16w, 16w + 16) (own tensorcnt)
        _tdm_rows(DO, off_v, D_V, 32, valid, _q_rs_v, lb_v, _lds_bf_ty, nw)
        _tdm_rows(Q, off_k, D_QK, 32, valid, _q_rs_k, lb_k, _lds_bf_ty, nw)

    def _tdm_qdo(qt, gh, stage_off):
        _tdm_qdo_issue(_tdm_qdo_prep(qt, gh, stage_off))

    # K and V fragments of this kv tile are invariant over every query and every q head.
    kf = [
        [gfrag(g_k, base_k, rs_k, kv0 + fx.Int32(kh * 16), dt) for dt in range(NDT_QK)] for kh in range(NKV)
    ]
    vf = [[gfrag(g_v, base_v, rs_v, kv0 + fx.Int32(kh * 16), dt) for dt in range(NDT_V)] for kh in range(NKV)]
    lane_r = (lane // fx.Int32(16)) * fx.Int32(8) + lane % fx.Int32(8)
    lane_c = ((lane // fx.Int32(8)) % fx.Int32(2)) * fx.Int32(8)
    # Lane-only parts of the DS-phase address families (dO rows / Q rows); every per-op
    # difference is a compile-time constant ISel folds into the 16-bit DS immediate
    # (bounds_proof.py: max immediate < 65536).
    lb_tr_v = lane_r * fx.Int32(XV_ROW_B) + lane_c * fx.Int32(2)  # b_do tr16
    lb_rd_v = row * fx.Int32(XV_ROW_B) + half * fx.Int32(16)  # dO readback
    lb_tr_k = lb_tr_v if SAME_D else lane_r * fx.Int32(XK_ROW_B) + lane_c * fx.Int32(2)
    lb_rd_k = lb_rd_v if SAME_D else row * fx.Int32(XK_ROW_B) + half * fx.Int32(16)

    def _rd_bases(stage_off):
        """(Q base, dO base) of a stage's readback: rows are addressed base + constant."""
        b_v = _lds0 + stage_off + lb_rd_v
        return (b_v, b_v) if SAME_D else (_lds0 + stage_off + lb_rd_k, b_v)

    # Causal tile skip: under bottom-right causal, kv row j is attended only by queries
    # q >= j - cshift, so query pairs below qp_start contribute exactly zero to dK/dV. All loop
    # bounds use the WORKGROUP's kv rows [kv0g, kv0g + nw*BLOCK_KV) (block id only): a pair
    # below qp_start is fully masked for every wave, a pair from qp_start + nmaskp on fully
    # unmasked for every wave; inside the masked pairs each wave masks with its own kv0.
    nqt2 = nqt // fx.Int32(2)  # query tiles come in PAIRS

    def _clampqt(t):
        """Pin a prefetch's query pair into [0, nqt2-1]."""
        t = (t < nqt2).select(t, nqt2 - fx.Int32(1))
        return (t < fx.Int32(0)).select(fx.Int32(0), t)

    _c = kv0g - cshift
    qp_start = ((_c < fx.Int32(0)).select(fx.Int32(0), _c)) // fx.Int32(32)
    qp_start = (causal != fx.Int32(0)).select(qp_start, fx.Int32(0))
    nqp_eff = nqt2 - qp_start

    # Iteration order is query-pair-OUTER, q-head-INNER (qi = ii // G, gh = ii % G), which
    # puts every masked iteration at the front: exactly one query pair per q head straddles
    # this kv tile, so the mask runs in its own loop (qloop_mask) and the hot loop
    # (qloop_full) carries no mask at all.
    def _rdqd(stage_off, rbase=None):
        """The 2*NHB ds_load_b128 that read one stage's S/dP B operands. Entry
        hh*NHB + dt*2 + u is Q row hh*16+row, bytes half*16 + dt*64 + u*32; entry
        hh*NHB + NQE + dt*2 + u the same for dO. Every address is a base + a constant."""
        if const_expr(rbase is None):
            rbase = _rd_bases(stage_off)
        out = []
        for hh in range_constexpr(2):
            for rb_, img, xrow, ndt in (
                (rbase[0], QOFF, XK_ROW_B, NDT_QK),
                (rbase[1], 0, XV_ROW_B, NDT_V),
            ):  # Q then dO
                for dt in range_constexpr(ndt):
                    for u in range_constexpr(2):
                        out.append(
                            fx.Vector(
                                llvm_dialect.load(
                                    v8b,
                                    create_llvm_ptr(
                                        rb_ + fx.Int32(hh * 16 * xrow + img + dt * 64 + u * 32),
                                        address_space=3,
                                    ),
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

        # carry=True (qloop_full): this iteration's S/dP B operands arrive in VGPRs (qd),
        # read back from stage it%3 during iteration it-1 (or by the prologue).
        # nw = 1: the TDM for iteration it+2 goes into stage (it+2)%3 == (it-1)%3 at the top,
        # with NO wait: that stage's last readers (tr16 of it-1, readback in it-2) were
        # consumed by WMMAs of it-1, hence retired. The tensor wait sits at the readback below.
        # nw > 1 (k_dkdv64): no TDM at the top. The prefetch of tile it+3 into stage it%3 is
        # issued right after BARRIER(it), behind the readback (see there); only its scalar
        # address math is emitted in the kh0 run below, where SALU between WMMAs is free.
        # carry=False (qloop_mask): load its own tile into stage 0 and wait for it.
        rb_base = None
        if const_expr(carry and nw > 1):
            # the NL/ND prefetch for it+1 stays at the top: it needs ~1.4k cycles before
            # its back-edge copy
            rocdl.sched_barrier(0)
            nxt = _ldl(qt_n, gh_n)
        elif const_expr(carry):
            # TDM first, then the NL/ND prefetch: their soffset SGPRs are then not
            # recycled by the TDM descriptor SALU right after the loads.
            rocdl.sched_barrier(0)
            _tdm_qdo(pf_qt, pf_gh, nxt_off)
            rocdl.sched_barrier(0)
            nxt = _ldl(qt_n, gh_n)
        else:
            nxt = None
            cur_off = fx.Int32(0)
            pre = _ldl(qt, gh)
            if const_expr(nw > 1):
                # WAR: every wave's reads of stage 0 (previous masked iteration) retired
                rocdl.sched_barrier(0)
                _lds_barrier()
                rocdl.sched_barrier(0)
            _tdm_qdo(qt, gh, cur_off)
            rocdl.sched_barrier(0)
            tdm_ops.tensor_wait(0)
            rocdl.sched_barrier(0)
            if const_expr(nw > 1):
                _lds_barrier()  # RAW: every wave's half of the tile landed
                rocdl.sched_barrier(0)
        lds_do = _lds0 + cur_off

        # The S/dP B operands, read back from the TDM image: lane (row, half) reads row
        # hh*16+row, bytes half*16 + dt*64 (+32).
        ops = qd if const_expr(carry) else _rdqd(cur_off)
        # r5.i3.g17: S = Q K^T and dP = dO V^T -- the operands of S^T / dP^T swapped, same
        # registers (A and B share one K striping). Their C layout is lane = kv (row), element si =
        # q row hh*16 + half*8 + si, so the bf16 pack [hh0 | hh1] of P (dS) per kv sub-tile IS the
        # A operand of dV += P^T dO (dK += dS^T Q), element for element what the P/dS LDS round
        # trip's tr16 produced (k_dqg96 feeds its dS -> dQ the same way). No P/dS stores, no A tr16.
        # bwd_r9_a: the same register is the B operand of dV^T += dO^T P (dK^T += Q^T dS).
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
            # softmax/dS in k_dqg96's FMA form (_smx): p = exp2(fma(s, scale*LOG2E, -lse*LOG2E)),
            # ds = bf16(p * fma(dp, scale, -delta*scale)); the two per-row constants come per
            # element from k_delta (NL = -LOG2E*lse, ND = -scale*delta: the products the per-lane
            # form computed as nl_q / nd_q), q rows qb + si of this lane
            nl = [pre[hh * 4 + si // 4][si % 4] for si in range(8)]
            nd = [pre[hh * 4 + 2 + si // 4][si % 4] for si in range(8)]
            qb = q0 + fx.Int32(hh * 16) + half * fx.Int32(8)
            for kh in range_constexpr(NKV):
                s_acc = _ir(fx.Vector.filled(8, 0.0, fx.Float32))
                p_acc = _ir(fx.Vector.filled(8, 0.0, fx.Float32))
                for dt in range_constexpr(NDT_MAX):
                    if const_expr(dt < NDT_QK):  # S = Q K^T contracts D_QK
                        s_acc = rocdl.wmma_f32_16x16x32_bf16(
                            v8f, _ir(qfr[dt]), _ir(kf[kh][dt]), s_acc, reuseA=False, reuseB=False
                        ).result
                    if const_expr(dt < NDT_V):  # dP = dO V^T contracts D_V
                        p_acc = rocdl.wmma_f32_16x16x32_bf16(
                            v8f, _ir(dfr[dt]), _ir(vf[kh][dt]), p_acc, reuseA=False, reuseB=False
                        ).result
                sv, pv_ = fx.Vector(s_acc), fx.Vector(p_acc)
                # causal, BOTTOM-RIGHT: query q attends kv <= q + (Skv - Sq); kv = this lane's row.
                kvl = kv0 + fx.Int32(kh * 16) + row
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
        # ONE DS phase per iteration. Fence off the S/dP/softmax part, then issue the B-operand
        # tr16 loads and (carry) the tensor wait + readback of the NEXT iteration's stage. The A
        # operands (P^T, dS^T) are already in VGPRs, so the first dK/dV WMMA waits only for its
        # own B operand. The DS-phase bases are computed BEFORE the fence so their VALU->DS
        # latency hides under the S/dP phase.
        tr_v = lds_do + lb_tr_v
        tr_k = tr_v if SAME_D else lds_do + lb_tr_k
        if const_expr(carry):
            rb_base = _rd_bases(rb_off)
        rocdl.sched_barrier(0)

        # 32 queries staged, so the contraction is FULL: rows [lane_r] and [lane_r+16]
        # concatenate in lane into a v16 operand.
        def tr(base, rowb):
            return fx.Vector(rocdl.ds_load_tr16_b128(v8b, create_llvm_ptr(base, address_space=3))).shuffle(
                fx.Vector(
                    rocdl.ds_load_tr16_b128(v8b, create_llvm_ptr(base + fx.Int32(16 * rowb), address_space=3))
                ),
                list(range(16)),
            )

        # The B operands (dO, Q) are shared by BOTH kv sub-tiles.
        if const_expr(carry):
            # DS issue order = the dK/dV WMMA consumption order (b_do for dV, then b_q for dK),
            # each group fenced so the scheduler keeps it.
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
            # Outstanding TDM here: stage (it+1)%3 (issued in it-1) and (it+2)%3 (issued at
            # the top of this iteration), TDM_OPS_QDO ops each. Waiting down to TDM_OPS_QDO
            # retires the older stage, the one about to be read (TDM ops retire in order).
            rocdl.sched_barrier(0)
            tdm_ops.tensor_wait(TW_QDO)
            rocdl.sched_barrier(0)
            rb = _rdqd(rb_off, rb_base)
        rocdl.sched_barrier(0)
        tdm_p = None
        if const_expr(carry and nw > 1):
            # k_dkdv64: the counters and address math of the prefetch (tile it+3 into stage
            # it%3), emitted inside the kh0 run's scheduling region; the sched_group_barriers
            # after the kh0 WMMAs place them between those WMMAs.
            pq, pg = pf3()
            tdm_p = _tdm_qdo_prep(pq, pg, cur_off)
        new = [None] * NST
        for kh in range_constexpr(NKV):
            if const_expr(carry and nw > 1 and kh == BAR_KH):
                # k_dkdv64: the ring barrier, after the WMMAs of kv sub-tiles < BAR_KH (their
                # run hides the drain of this iteration's tr16 reads). tensor_wait retires
                # this wave's half of stage (it+1)%3; the barrier's release fence drains every
                # wave's DS reads of stage it%3 (WAR for the prefetch below); after it every
                # wave's half of (it+1)%3 is in LDS, so the readback may start. Then the
                # prefetch of tile it+3 into stage it%3 (read again only after BARRIER(it+2)),
                # in the same memory window as the readback: one WMMA<->memory switch.
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
            # bwd_r9_a: dV^T += dO^T P and dK^T += Q^T dS -- the operands of r5.i3.g17's dV/dK WMMAs
            # swapped, same registers (A and B share one K striping; the per-element sum is the same,
            # bitwise on this card: r5.i3.g17, r6.i2.g22). The C fragment then holds kv row kh*16 + row
            # x d = dtile*16 + half*8 + [0, 8): 8 consecutive d of one kv row, which the epilogue
            # stores into a row-major image with ONE ds_store_b128. B-operand reuse hint: within a
            # run, B (P or dS of kv sub-tile kh) is held across all dtiles, so instructions 2.. reuse it.
            for dtile in range_constexpr(NDO_V):  # dV^T += dO^T P
                new[kh * NDO_V + dtile] = rocdl.wmma_f32_16x16x32_bf16(
                    v8f, _ir(b_do[dtile]), _ir(a_p), acc[kh * NDO_V + dtile], reuseA=False, reuseB=(dtile > 0)
                ).result
            for dtile in range_constexpr(NDO_QK):  # dK^T += Q^T dS
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
            # Fence the dK/dV WMMAs off from the back-edge copies of the NL/ND prefetch,
            # so their s_wait_loadcnt sits at the END of the body.
            rocdl.sched_barrier(0)
            return new + [_ir(v) for v in nxt] + [_ir(v) for v in rb]
        return new

    # Division-free loop index: the flat iteration index ii == qc*G + gc (0 <= gc < G) is
    # carried in the scf.for state as two uniform i32 counters advanced by a wrap
    # gc+1 == G -> (qc+1, 0); the ring stage offset cur = (ii % 3) * QDO_B is a third.
    # bounds_proof.py replays the counters against the division form for every iteration.
    def _wrap(qc, gc):
        g1 = gc + fx.Int32(1)
        nw = g1 < G
        return fx.Int32(nw.select(qc, qc + fx.Int32(1))), fx.Int32(nw.select(g1, fx.Int32(0)))

    @flyc.jit
    def qloop_mask(state, n, qt0):
        final = state
        for it, carried in range(fx.Index(fx.Int32(0)), fx.Index(n), 1, init=state):
            st = list(carried)
            qi, gh = fx.Int32(st[NST]), fx.Int32(st[NST + 1])
            qn, gn = _wrap(qi, gh)
            res = _body(st[:NST], None, qt0 + qi, gh, True, None, None, False) + [
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
            # counters (cur, qi, gh) at st[-3:], NL/ND at st[NST:NST+NLD], carried B
            # operands at st[NST+NLD:-3].
            cur, qi, gh = fx.Int32(st[-3]), fx.Int32(st[-2]), fx.Int32(st[-1])
            qn, gn = _wrap(qi, gh)  # decomposition of ii+1
            live = jj < n  # else jj clamps to n-1 == ii
            qj = fx.Int32(live.select(qn, qi))
            gj = fx.Int32(live.select(gn, gh))
            # (ii+1)%3: next iteration's stage == this iteration's readback stage
            ncur = fx.Int32((cur == fx.Int32(2 * QDO_B)).select(fx.Int32(0), cur + fx.Int32(QDO_B)))
            if const_expr(nw > 1):
                # k_dkdv64: the prefetch tile is min(ii+3, n-1), issued after BARRIER(ii) into
                # stage ii%3 == cur; _body emits these counters inside its kh0 run.
                def _pf3(qn=qn, gn=gn, qj=qj, gj=gj, ii=ii):
                    q2, g2 = _wrap(qn, gn)  # decomposition of ii+2
                    live2 = (ii + fx.Int32(2)) < n  # else clamps to min(ii+1, n-1)
                    q2c = fx.Int32(live2.select(q2, qj))
                    g2c = fx.Int32(live2.select(g2, gj))
                    q3, g3 = _wrap(q2, g2)  # decomposition of ii+3
                    live3 = (ii + fx.Int32(3)) < n  # else clamps to min(ii+2, n-1)
                    return (qt0 + fx.Int32(live3.select(q3, q2c)), fx.Int32(live3.select(g3, g2c)))

                res = _body(
                    st[:NST],
                    [fx.Vector(v) for v in st[NST : NST + NLD]],
                    qt0 + qi,
                    gh,
                    False,
                    qt0 + qj,
                    gj,
                    True,
                    cur,
                    None,
                    None,
                    None,
                    [fx.Vector(v) for v in st[NST + NLD : -3]],
                    ncur,
                    pf3=_pf3,
                )
            else:
                q2, g2 = _wrap(qn, gn)  # decomposition of ii+2
                live2 = (ii + fx.Int32(2)) < n  # else kk clamps to n-1 == jj
                pf_qt = qt0 + fx.Int32(live2.select(q2, qj))
                pf_gh = fx.Int32(live2.select(g2, gj))
                # prefetch stage (ii+2)%3 == (ii-1)%3: the stage before cur
                nxo = fx.Int32((cur == fx.Int32(0)).select(fx.Int32(2 * QDO_B), cur - fx.Int32(QDO_B)))
                res = _body(
                    st[:NST],
                    [fx.Vector(v) for v in st[NST : NST + NLD]],
                    qt0 + qi,
                    gh,
                    False,
                    qt0 + qj,
                    gj,
                    True,
                    cur,
                    nxo,
                    pf_qt,
                    pf_gh,
                    [fx.Vector(v) for v in st[NST + NLD : -3]],
                    ncur,
                )
            res = res + [fx.as_ir_value(ncur), fx.as_ir_value(qn), fx.as_ir_value(gn)]
            final = yield res
        return final

    def _tdm_prologue(qt0, n):
        """Fill stages 0..1 for qloop_full's first two iterations: stage 0 gets iteration 0's
        tile (qt0, gh 0), stage 1 iteration min(1, n-1)'s, clamped to a legal query pair, so
        an empty or 1-iteration loop still issues only in-bounds tiles (never read). Then
        retire stage 0 and read back iteration 0's B operands. nw > 1 (k_dkdv64): then also
        stage 2 <- tile min(2, n-1) -- the prefetch the full loop's iteration 0 no longer
        issues -- after the RAW barrier, so only one stage is in flight at that barrier."""
        if const_expr(nw > 1):
            # WAR: every wave's reads of stage 0 in the masked loop retired
            rocdl.sched_barrier(0)
            _lds_barrier()
            rocdl.sched_barrier(0)
        _tdm_qdo(_clampqt(qt0), fx.Int32(0), fx.Int32(0))
        # j1 = max(min(1, n-1), 0) is 1 iff n > 1, and 1 == wrap(0, 0).
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
            # j2 = min(2, n-1) (>= 0): 2 == wrap(wrap(0, 0)) iff n > 2, else j1
            q2w, g2w = _wrap(q1w, g1w)
            three = fx.Int32(2) < n
            q2 = fx.Int32(three.select(q2w, q1))
            g2 = fx.Int32(three.select(g2w, g1))
            rocdl.sched_barrier(0)  # readback before the TDM (one region, R4 order)
            _tdm_qdo(_clampqt(qt0 + q2), g2, fx.Int32(2 * QDO_B))
            rocdl.sched_barrier(0)
        return [_ir(v) for v in rb0]

    # Query pair `qt` is fully unmasked iff this workgroup's LARGEST key index is attended
    # by the pair's SMALLEST query: kv0g + nw*BLOCK_KV-1 <= qt*32 + cshift, i.e.
    # qt >= ceil((kv0g + nw*BLOCK_KV-1 - cshift)/32). cshift = Skv - Sq can be negative.
    _u = kv0g + fx.Int32(nw * BLOCK_KV - 1) - cshift
    _qsf = (_u < fx.Int32(0)).select(fx.Int32(0), (_u + fx.Int32(31)) // fx.Int32(32))
    _qsf = (_qsf < nqt2).select(_qsf, nqt2)
    _nm = _qsf - qp_start
    _nm = (_nm < fx.Int32(0)).select(fx.Int32(0), _nm)
    _nm = (_nm < nqp_eff).select(_nm, nqp_eff)
    nmaskp = (causal != fx.Int32(0)).select(_nm, fx.Int32(0))

    init = [_ir(fx.Vector.filled(8, 0.0, fx.Float32)) for _ in range(NST)]
    # every loop starts at ii = 0: (qi, gh) = (0, 0), ring stage offset cur = 0.
    _z2 = [fx.as_ir_value(fx.Int32(0)), fx.as_ir_value(fx.Int32(0))]
    _z3 = [fx.as_ir_value(fx.Int32(0))] + _z2
    out = qloop_mask(init + _z2, G * nmaskp, qp_start)
    # The NL/ND prefetch and the TDM prologue sit BETWEEN the loops, for qloop_full's
    # first iteration (qi = 0, gh = 0) at query pair qp_start + nmaskp (clamped: the
    # prologue issues even when the full loop is empty).
    _pc0 = _clampqt(qp_start + nmaskp)
    out = list(out)[:NST] + [_ir(v) for v in _ldl(_pc0, fx.Int32(0))]
    out = out + _tdm_prologue(qp_start + nmaskp, G * (nqp_eff - nmaskp)) + _z3
    out = qloop_full(out, G * (nqp_eff - nmaskp), qp_start + nmaskp)
    # Retire the last iteration's (clamped) prefetch -- or the prologue's, when the loop ran
    # 0 times -- before the epilogue reuses the start of LDS, which overlaps the ring.
    tdm_ops.tensor_wait(0)
    _epi0 = _lds0
    if const_expr(nw > 1):
        # every wave's TDM landed and every wave's ring reads (incl. the dead final readback)
        # retired before any wave's image overwrites the ring; wave w's image at w*EPI_W
        rocdl.sched_barrier(0)
        _lds_barrier()
        rocdl.sched_barrier(0)
        _epi0 = _lds0 + wv * fx.Int32(EPI_W)

    # Epilogue (bwd_r9_a, r6.i2.g22's dq form): the dV^T / dK^T C fragments (lane (row, half): kv
    # row kh*16 + row, d = dtile*16 + half*8 + [0, 8)) go as 8 bf16 = ONE ds_store_b128 each into
    # this wave's ROW-major images -- dV [32 kv][D_V] at row stride XV_ROW_B at +0, dK [32 kv][D_QK]
    # at row stride XK_ROW_B at +EPI_DK -- which are then read back in whole-line order: lane l reads
    # kv row 4rb + l//8, bytes li*128 + (l%8)*16, and its buffer_store_b128 writes the same 16 B of
    # that dv/dk row, so each 8-lane run of a store is one aligned 128-B line and one store covers 4
    # whole lines (per wave 16 + 24 stores, 160 line requests; oe6: 40 stores of 16 rows x 32 B, 640
    # segments). Each load is followed by its own store (paced pairs; a 40-load burst before 40 stores
    # cost +22-27% in bwd_r6_c E1), per row block rb the dV lines then the dK lines (interleaving the two
    # images per rb gives 15 s_wait_xcnt at prod; dV then dK as two passes gave 21). The images are
    # wave-local and DS ops of one wave execute in order: no barrier.
    g_dv8 = _bv(DV_, nv_b, fx.BFloat16, 8)
    g_dk8 = _bv(DK, nk_b, fx.BFloat16, 8)
    wr_v = _epi0 + row * fx.Int32(XV_ROW_B) + half * fx.Int32(16)
    wr_k = _epi0 + fx.Int32(EPI_DK) + row * fx.Int32(XK_ROW_B) + half * fx.Int32(16)
    for kh in range_constexpr(NKV):
        for dtile in range_constexpr(NDO_MAX):
            if const_expr(dtile < NDO_V):
                ov = fx.Vector(out[kh * NDO_V + dtile])
                llvm_dialect.store(
                    fx.as_ir_value(
                        fx.Vector.from_elements(
                            [ov[si].to(fx.BFloat16) for si in range_constexpr(8)], dtype=fx.BFloat16
                        )
                    ),
                    create_llvm_ptr(wr_v + fx.Int32(kh * 16 * XV_ROW_B + dtile * 32), address_space=3),
                )
            if const_expr(dtile < NDO_QK):
                ok_ = fx.Vector(out[DK0 + kh * NDO_QK + dtile])
                llvm_dialect.store(
                    fx.as_ir_value(
                        fx.Vector.from_elements(
                            [ok_[si].to(fx.BFloat16) for si in range_constexpr(8)], dtype=fx.BFloat16
                        )
                    ),
                    create_llvm_ptr(wr_k + fx.Int32(kh * 16 * XK_ROW_B + dtile * 32), address_space=3),
                )
    rocdl.sched_barrier(0)
    lrow = lane // fx.Int32(8)  # 4 kv rows per store
    lcol = lane % fx.Int32(8)  # 16-B chunk of the 128-B line
    rd_v = _epi0 + lrow * fx.Int32(XV_ROW_B) + lcol * fx.Int32(16)
    rd_k = _epi0 + fx.Int32(EPI_DK) + lrow * fx.Int32(XK_ROW_B) + lcol * fx.Int32(16)
    for rb in range_constexpr(BLOCK_KV // 4):
        # vec8 index of (kv row kv0 + 4rb + lrow, head hkv, d = 0) in dv [B, Skv, Hkv, D_V] / dk
        gt_v = base_v + (kv0 + fx.Int32(rb * 4) + lrow) * rs_v + lcol
        gt_k = base_k + (kv0 + fx.Int32(rb * 4) + lrow) * rs_k + lcol
        for li in range_constexpr(D_V * 2 // EPI_LINE_B):
            vv = fx.Vector(
                llvm_dialect.load(
                    v8b,
                    create_llvm_ptr(rd_v + fx.Int32(rb * 4 * XV_ROW_B + li * EPI_LINE_B), address_space=3),
                )
            )
            _stv(
                [vv[e] for e in range_constexpr(8)],
                g_dv8,
                gt_v + fx.Int32(li * (EPI_LINE_B // 16)),
                fx.BFloat16,
            )
        for li in range_constexpr(D_QK * 2 // EPI_LINE_B):
            kk = fx.Vector(
                llvm_dialect.load(
                    v8b,
                    create_llvm_ptr(rd_k + fx.Int32(rb * 4 * XK_ROW_B + li * EPI_LINE_B), address_space=3),
                )
            )
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
    _ = MODULE_KNOBS  # cache key only (see MODULE_KNOBS)
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
    k_dkdv(Q, K, V, DO, NL, ND, DV_, DK, scale, Sq, Skv, Hq, Hkv, G, nqt, cshift, causal, nb).launch(
        grid=(nhkv, nblk, nb), block=(32, 1, 1), stream=stream
    )


# k_dkdv64: DKDV_NW = 2 waves per workgroup (one per SIMD at 1 wave/SIMD), 64 kv rows per
# Q/dO fetch, so the Q/dO TDM bytes and LDS writes per dK/dV FLOP halve. Same body as k_dkdv
# (each wave keeps its 887-VGPR register plan), one shared ring, barriers per _dkdv_impl.
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
    _ = MODULE_KNOBS  # cache key only (see MODULE_KNOBS)
    rocdl.disable_xdl_arb_stall()  # lock_simd (SCHED_MODE.DISABLE_XDL_ARB_STALL); PROVENANCE.md, round 5
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
    # grid = (hg kv heads of a head group, Skv/64 kv blocks, ngz = B*Hkv/hg head groups); kv block
    # ascending = longest first (causal); hg = Hkv is (Hkv, Skv/64, B), r3_a's grid
    k_dkdv64(Q, K, V, DO, NL, ND, DV_, DK, scale, Sq, Skv, Hq, Hkv, G, nqt, cshift, causal, nb, hg).launch(
        grid=(hg, nblk, ngz), block=(WAVE * DKDV_NW, 1, 1), stream=stream
    )


# ===================================================================== dqg ========
# k_dqg -- one wave owns DQ_BQW queries of one q head and streams every KV_STEP-row kv
# block through a 3-stage TDM K/V LDS ring. Global loop index i (full loop, then masked):
#   stage i%3        : tile i  -> K region read by the dQ tr16
#   stage (i+1)%3    : tile min(i+1, N-1) -> read back (ds_load_b128) into the carried S/dP
#                      A operands of iteration i+1
#   stage (i+2)%3    : TDM for tile min(i+2, N-1), issued at the top of iteration i
# N = nkvt_eff. The mask loop continues the SAME ring, so the full->mask handoff is just the
# carried state. No VMEM in the hot loop.
KV_STEP = 32  # kv rows per k_dqg iteration: one WMMA contraction of dS
NKT = KV_STEP // 16
# queries per k_dqg wave: at D_QK = 192, 64 needs 1024 VGPR + 148 spilled (the accumulators
# and Q/dO fragments grow 1.5x); 32 halves both (709 VGPR, no spill).
DQ_BQW = 64 if D_QK <= 128 else 32
NQW = DQ_BQW // 16
# k_dqg48: the same kernel at 48 queries per wave (NQW48 = 3 query sub-tiles). Per 32-row kv step
# it issues the same 64 DS loads and 20 KB of TDM as the 32-query kernel but 96 WMMA (S 36, dP 24,
# dQ 36) instead of 64. Sq % 48 != 0 in general, so the 32-query k_dqg covers the head [0, q_split)
# and k_dqg48 the rest; dq_split() is the single source of that split (impl._plan, bounds_proof).
DQ_BQW48 = 48
NQW48 = DQ_BQW48 // 16
# k_dqg96: DQ_NWAVE waves per workgroup, each the k_dqg48 body over its own 48 queries, consume
# ONE K/V TDM ring: 96 queries per K/V fetch. Wave w owns [q0 + 48w, q0 + 48w + 48), q0 =
# q_off + bid*96; the kv range runs to the last wave's diagonal (both waves run the same trip
# count; wave 0's extra steps are masked through do_mask). One barrier per kv step.
DQ_NWAVE = 2
DQ_BQW96 = DQ_NWAVE * DQ_BQW48
DQ_SPLITS = (0, 32, 64)  # q_split candidates: multiples of DQ_BQW; lcm(32, 96) = 96 > 64


def dq_split(sq, bqw=DQ_BQW96):
    """(q_split, n32, nm) for Sq = sq: k_dqg runs n32 = q_split/32 tiles over queries [0, q_split),
    the main dQ kernel (bqw 96: k_dqg96, the launched one; 48: k_dqg48) nm = (sq - q_split)/bqw
    tiles over [q_split, sq); q_split is the smallest of DQ_SPLITS with (sq - q_split) % bqw == 0
    (exists for every sq % 32 == 0: sq % 96 in {0, 32, 64}, sq % 48 in {0, 32, 16})."""
    assert DQ_BQW == 32, "the k_dqg48/k_dqg96 split assumes the 32-query head kernel (D_QK 192)"
    for qs in DQ_SPLITS:
        if qs <= sq and (sq - qs) % bqw == 0:
            return qs, qs // DQ_BQW, (sq - qs) // bqw
    raise ValueError(f"no k_dqg/main split for seqlen_q {sq} at {bqw} queries per tile")


# Workgroup sync of k_dqg96, one per kv step and one in the prologue. Every LDS read this wave
# issued before it (the dQ tr16 of stage i%3, the readback of i-1) has retired (dscnt 0), and
# every TDM op of this wave except the newest stage has retired (the caller's tensor_wait).
# Raw split barrier with nothing between signal and wait; sched_barrier(0) on both sides keeps
# every DS/TDM op on its side (checked on the compiled ISA during development).
# DQ_BARRIER_FENCE: workgroup release/acquire fences around signal/wait (gpu.barrier()'s
# semantics), so no IR pass moves an LDS access across the barrier either. Measured at compile
# time: the fences add no s_wait_tensorcnt 0 (the newest ring stage stays in flight) and no wait
# the kernel did not already have (prologue loadcnt 0 is needed by the softmax constants anyway).
DQ_BARRIER_FENCE = True


def _wg_sync():
    # UNSTABLE(gfx1250): s_wait_dscnt, raw s_barrier_signal/wait (id -1), llvm.fence
    rocdl.sched_barrier(0)
    rocdl.s_wait_dscnt(0)
    if const_expr(DQ_BARRIER_FENCE):
        llvm_dialect.fence(llvm_dialect.AtomicOrdering.release, syncscope="workgroup")
    rocdl.s_barrier_signal(-1)
    rocdl.s_barrier_wait(-1)
    if const_expr(DQ_BARRIER_FENCE):
        llvm_dialect.fence(llvm_dialect.AtomicOrdering.acquire, syncscope="workgroup")
    rocdl.sched_barrier(0)


VOFF = KV_STEP * XK_ROW_B  # ring stage: K [32][D_QK] at +0, V [32][D_V] at +VOFF
KV_B = KV_STEP * (XK_ROW_B + XV_ROW_B)  # one ring stage (21504 B at 192/128)
TDM_OPS_KV = len(_pow2_segments(D_QK)) + len(_pow2_segments(D_V))  # TDM ops per stage
DQT_TW = TDM_OPS_KV * (TDM_DEPTH - 2)  # tensorcnt that retires all but the newest stage
assert TDM_DEPTH * KV_B <= LDS_SEG
# dQ epilogue (r6.i2.g22): each wave stages its dQ tile through the dead K/V ring as a row-major
# image [BQW q rows][D_QK] bf16, row stride D_QK*2 + 16 = XK_ROW_B (rows 36r mod 64 dwords apart:
# 16 distinct 4-bank groups), wave w at w*BQW*DQ_IMG_ROW_B, so every buffer_store_b128 can write 4
# WHOLE 128-B dq lines (lane l: row l//8, bytes (l%8)*16 of one 128-B line) instead of 16 rows x 32 B.
DQ_IMG_ROW_B = XK_ROW_B
DQ_LINE_EL = 64  # bf16 per 128-B line
assert D_QK % DQ_LINE_EL == 0 and (D_QK * 2) % 128 == 0, "dq rows are whole 128-B lines"
assert DQ_NWAVE * DQ_BQW48 * DQ_IMG_ROW_B <= TDM_DEPTH * KV_B and DQ_BQW * DQ_IMG_ROW_B <= TDM_DEPTH * KV_B
# Carried S/dP A operands (read back one iteration early), in issue order: per (kt, dt)
# K u0, K u1 (dt < NDT_QK) then V u0, V u1 (dt < NDT_V) -- the 16 B chunks of row kt*16+row
# at bytes half*16 + dt*64 + u*32.
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
# Full-loop schedule ("ck"): kt0 S/dP WMMAs; kt1 S/dP in NDT_MAX chunks {the WMMAs of dtile j
# + softmax/dS (kt0, qh j)}; the DS phase (tr16 | tensor_wait | readback); softmax/dS
# (kt1, qh 0); dQ qh-major in chunks {NDO_QK dQ WMMAs of qh j + softmax/dS (kt1, qh j+1)}.
# Chunks end in sched_barrier(0); inside a chunk sched_group_barrier {1 WMMA, 2 VALU,
# 1 TRANS} per WMMA.
DQT_VT_SGB = (2, 1)


def _dqg_tdm_impl(
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
    nwave=1,
    HG=None,
):
    # nqw (compile-time): 16-query sub-tiles per wave (NQW 2 = k_dqg, NQW48 3 = k_dqg48/96).
    # nwave (compile-time): waves per workgroup sharing the K/V ring (1, or DQ_NWAVE = k_dqg96).
    # The workgroup owns queries [q0g, q0g + nwave*16*nqw), q0g = q_off + bid*BQWG, bid in
    # [0, ntile) (runtime); wave wv owns [q0, q0 + BQW), q0 = q0g + wv*BQW. nwave == 1 is the
    # single-wave kernel unchanged (q0 == q0g, lane == thread id, no barrier).
    NQW = nqw
    BQW = 16 * NQW
    BQWG = BQW * nwave
    if const_expr(nwave == 1):
        lane = fx.Int32(fx.thread_idx.x)
    else:
        _tid = fx.Int32(fx.thread_idx.x)
        lane = _tid % fx.Int32(WAVE)
        # wave index in the workgroup (threads [32w, 32w+32) form wave w), made uniform (SGPR)
        wv = fx.Int32(rocdl.readfirstlane(fx.Int32.ir_type, (_tid // fx.Int32(WAVE)).ir_value()))
    # XCD-major q-head remap: workgroups go to the 8 XCDs round-robin on the linear id, so
    # x -> (x%8)*(ngrp/8) + x/8 puts adjacent q heads (one kv head under GQA) on one XCD.
    # A bijection of [0, ngrp) whenever ngrp % 8 == 0, else the identity. ngrp = grid.x: Hq
    # (HG None), else the head group HG (HEAD_GROUP note; grid (HG, ntile, B*Hq/HG), HG | Hq).
    _nx = fx.Int32(8)
    _gx = fx.Int32(fx.block_idx.x)
    if const_expr(HG is None):
        ngrp = Hq
    else:
        ngrp = HG
    qh = (ngrp % _nx == fx.Int32(0)).select((_gx % _nx) * (ngrp // _nx) + _gx // _nx, _gx)
    # longest-first: the query tile is walked DESCENDING on grid.y (grid.y == ntile)
    bid = ntile - fx.Int32(1) - fx.Int32(fx.block_idx.y)
    if const_expr(HG is None):
        bat = fx.Int32(fx.block_idx.z)
    else:
        # z*HG + (remapped x) is the batch-major q head index bat*Hq + qh (bounds_proof.py R1)
        _hb = fx.Int32(fx.block_idx.z) * HG + qh
        bat = _hb // Hq
        qh = _hb - bat * Hq
    row = lane % fx.Int32(16)
    half = lane // fx.Int32(16)
    q0g = q_off + bid * fx.Int32(BQWG)
    if const_expr(nwave == 1):
        q0 = q0g
    else:
        q0 = q0g + wv * fx.Int32(BQW)
    hkv = qh // G

    # Fake 1 GiB / 256 MiB extents (impl.py asserts every tensor fits): never "fix" these to
    # the k_dkdv formula -- k_dqg has no batch-count argument.
    g_q = _bv(Q, 1 << 30, fx.BFloat16, 8)
    g_do = _bv(DO, 1 << 30, fx.BFloat16, 8)
    g_lse = _bv(LSE, 1 << 28, fx.Float32)
    g_del = _bv(DEL, 1 << 28, fx.Float32)
    g_dq8 = _bv(DQ, 1 << 30, fx.BFloat16, 8)  # vec8 view: the whole-line dq epilogue (r6.i2.g22)

    rs_q = Hq * fx.Int32(DV8_QK)  # vec8 tiles between consecutive q rows
    base_q = bat * Sq * rs_q + qh * fx.Int32(DV8_QK)
    rs_o = rs_q if SAME_D else Hq * fx.Int32(DV8_V)
    base_o = base_q if SAME_D else bat * Sq * rs_o + qh * fx.Int32(DV8_V)
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
        [gfrag(g_q, base_q, rs_q, q0 + fx.Int32(qh_ * 16), dt) for dt in range(NDT_QK)] for qh_ in range(NQW)
    ]
    dof = [
        [gfrag(g_do, base_o, rs_o, q0 + fx.Int32(qh_ * 16), dt) for dt in range(NDT_V)] for qh_ in range(NQW)
    ]
    q_glob = [q0 + fx.Int32(qh_ * 16) + row for qh_ in range(NQW)]
    lse_q = [_ld1(g_lse, base_l + q_glob[qh_], fx.Float32) for qh_ in range(NQW)]
    del_q = [_ld1(g_del, base_l + q_glob[qh_], fx.Float32) for qh_ in range(NQW)]
    # softmax row constants, hoisted out of the kv loops:
    #   p = exp2(fma(s, scale*LOG2E, -lse*LOG2E)); ds = bf16(p * fma(dp, scale, -delta*scale))
    _c1q = scale * fx.Float32(LOG2E)
    _nlq = [lse_q[qh_] * fx.Float32(-LOG2E) for qh_ in range(NQW)]
    _ndq = [del_q[qh_] * (fx.Float32(0.0) - scale) for qh_ in range(NQW)]
    lane_r = (lane // fx.Int32(16)) * fx.Int32(8) + lane % fx.Int32(8)
    lane_c = ((lane // fx.Int32(8)) % fx.Int32(2)) * fx.Int32(8)
    # lane-only parts of the DS address families; every per-op difference is a compile-time
    # immediate (bounds_proof.py: max immediate < 65536).
    lb_tr = lane_r * fx.Int32(XK_ROW_B) + lane_c * fx.Int32(2)  # dQ B operand (K) tr16
    lb_rd_k = row * fx.Int32(XK_ROW_B) + half * fx.Int32(16)  # S/dP A readback, K rows
    lb_rd_v = lb_rd_k if SAME_D else row * fx.Int32(XV_ROW_B) + half * fx.Int32(16)

    _lds_bf_ty = fx.PointerType.get(
        elem_ty=fx.BFloat16.ir_type, address_space=fx.AddressSpace.Shared, alignment=16
    )
    _kv_rs_k = Hkv * fx.Int32(D_QK)  # elements between consecutive k rows
    _kv_rs_v = _kv_rs_k if SAME_D else Hkv * fx.Int32(D_V)

    def _tdm_kv(kv0p, stage_off):
        """K then V tile [kv0p, kv0p+32) of (bat, hkv) into stage_off / +VOFF: TDM_OPS_KV
        ops. Origin element ((bat*Skv + kv0p)*Hkv + hkv)*D, row stride Hkv*D, outer extent
        Skv - kv0p >= 32 (kv0p <= (nkvt-1)*32, bounds_proof.py Q1). nwave > 1: each wave
        issues its KV_STEP/nwave-row block of every op (bounds_proof.py W1)."""
        row0 = fx.Int64(bat * Skv + kv0p) * fx.Int64(Hkv) + fx.Int64(hkv)
        off_k = row0 * fx.Int64(D_QK)
        off_v = off_k if SAME_D else row0 * fx.Int64(D_V)
        valid = Skv - kv0p
        lb_k = _lds0 + stage_off
        lb_v = _lds0 + stage_off + fx.Int32(VOFF)
        _tdm_rows(K, off_k, D_QK, KV_STEP, valid, _kv_rs_k, lb_k, _lds_bf_ty, nwave)
        _tdm_rows(V, off_v, D_V, KV_STEP, valid, _kv_rs_v, lb_v, _lds_bf_ty, nwave)

    def _rdkv(stage_off):
        """NP ds_load_b128 in _RD_ORDER: K/V row kt*16+row, bytes half*16 + dt*64 + u*32."""
        rb_k = _lds0 + stage_off + lb_rd_k
        rb_v = rb_k if SAME_D else _lds0 + stage_off + lb_rd_v
        out = []
        for kt, dt, w, u in _RD_ORDER:
            if w == "k":
                a = rb_k + fx.Int32(kt * 16 * XK_ROW_B + dt * 64 + u * 32)
            else:
                a = rb_v + fx.Int32(VOFF + kt * 16 * XV_ROW_B + dt * 64 + u * 32)
            out.append(fx.Vector(llvm_dialect.load(v8b, create_llvm_ptr(a, address_space=3))))
        return out

    def _bks(cur):
        """dQ GEMM B operands: K^T of this stage, NDO_QK v16 (rows kv 0..15 | 16..31)."""
        tb = _lds0 + cur + lb_tr
        out = []
        for dtile in range_constexpr(NDO_QK):
            base = tb + fx.Int32(dtile * 32)
            out.append(
                fx.Vector(rocdl.ds_load_tr16_b128(v8b, create_llvm_ptr(base, address_space=3))).shuffle(
                    fx.Vector(
                        rocdl.ds_load_tr16_b128(
                            v8b, create_llvm_ptr(base + fx.Int32(16 * XK_ROW_B), address_space=3)
                        )
                    ),
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

    def _sgb(nw):
        # {1 WMMA, DQT_VT_SGB[0] VALU, DQT_VT_SGB[1] TRANS} x nw
        for _ in range_constexpr(nw):
            rocdl.sched_group_barrier(0x008, 1, 0)
            if const_expr(DQT_VT_SGB[0] > 0):
                rocdl.sched_group_barrier(0x002, DQT_VT_SGB[0], 0)
            if const_expr(DQT_VT_SGB[1] > 0):
                rocdl.sched_group_barrier(0x400, DQT_VT_SGB[1], 0)

    def _sdp(pre, kt, dt, s_acc, p_acc):
        """k-step dt of S^T = K Q^T (dt < NDT_QK) and dP^T = V dO^T (dt < NDT_V), all qh."""
        if const_expr(dt < NDT_QK):
            kfr = pre[_RD_IDX[(kt, dt, "k", 0)]].shuffle(pre[_RD_IDX[(kt, dt, "k", 1)]], list(range(16)))
        if const_expr(dt < NDT_V):
            vfr = pre[_RD_IDX[(kt, dt, "v", 0)]].shuffle(pre[_RD_IDX[(kt, dt, "v", 1)]], list(range(16)))
        for qh_ in range_constexpr(NQW):
            if const_expr(dt < NDT_QK):
                s_acc[qh_] = rocdl.wmma_f32_16x16x32_bf16(
                    v8f, _ir(kfr), _ir(qf[qh_][dt]), s_acc[qh_], reuseA=False, reuseB=False
                ).result
            if const_expr(dt < NDT_V):
                p_acc[qh_] = rocdl.wmma_f32_16x16x32_bf16(
                    v8f, _ir(vfr), _ir(dof[qh_][dt]), p_acc[qh_], reuseA=False, reuseB=False
                ).result

    def _body_ck(acc, pre, kv0, pf_kv0, cur, nxo, ncur):
        # Full loop: the WMMA<->softmax interleave described at DQT_VT_SGB.
        rocdl.sched_barrier(0)
        _tdm_kv(pf_kv0, nxo)
        rocdl.sched_barrier(0)
        if const_expr(nwave > 1):
            # k_dqg96: the dQ B operands (K^T of stage cur) as ONE tr16 burst at the step top,
            # beside the TDM (one WMMA<->memory switch, as before), so they drain under the kt0/kt1
            # WMMAs instead of at the _wg_sync dscnt 0. Stage cur was published by the previous
            # step's _wg_sync and is next re-targeted by the TDM at the top of i+1, after this
            # step's _wg_sync (dscnt 0 + barrier): same data, same WMMA order. The explicit wait
            # retires the carried readback `pre` (issued a whole dQ run ago) so the backend never
            # charges the burst to the first kt0 WMMA.
            # UNSTABLE(gfx1250): s_wait_dscnt
            rocdl.s_wait_dscnt(0)
            b_ks = _bks(cur)
            rocdl.sched_barrier(0)
        sp_acc = []
        ds_halves = [[] for _ in range_constexpr(NQW)]
        for kt in range_constexpr(NKT):
            s_acc = [_ir(fx.Vector.filled(8, 0.0, fx.Float32)) for _ in range_constexpr(NQW)]
            p_acc = [_ir(fx.Vector.filled(8, 0.0, fx.Float32)) for _ in range_constexpr(NQW)]
            for dt in range_constexpr(NDT_MAX):
                _sdp(pre, kt, dt, s_acc, p_acc)
                if const_expr(kt == 1):
                    # chunk dt: this dtile's kt1 WMMAs + softmax/dS of (kt0, qh = dt)
                    if const_expr(dt < NQW):
                        s0, p0 = sp_acc[0]
                        ds_halves[dt].append(_smx(fx.Vector(s0[dt]), fx.Vector(p0[dt]), dt, 0, kv0, False))
                    _sgb(NQW * (int(dt < NDT_QK) + int(dt < NDT_V)))
                    rocdl.sched_barrier(0)
            if const_expr(kt == 0):
                rocdl.sched_barrier(0)  # kt0 S/dP WMMAs: their own region
            sp_acc.append((s_acc, p_acc))
        for j in range_constexpr(NDT_MAX, NQW):  # kt0 rows without a kt1 chunk (NQW > NDT)
            s0, p0 = sp_acc[0]
            ds_halves[j].append(_smx(fx.Vector(s0[j]), fx.Vector(p0[j]), j, 0, kv0, False))
        s1, p1 = sp_acc[1]
        # the DS phase
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
            # k_dqg96: softmax/dS (kt1, qh 0), then this wave's half of stage ncur retires
            # (tensor_wait) and the workgroup syncs (_wg_sync; the step-top tr16 burst drained
            # under the S/dP WMMAs, so its dscnt 0 finds nothing in flight): both halves of ncur
            # landed, every wave's reads of stage cur retired (the TDM at the top of i+1 targets
            # it). Only then the readback of ncur.
            ds_halves[0].append(_smx(fx.Vector(s1[0]), fx.Vector(p1[0]), 0, 1, kv0, False))
            rocdl.sched_barrier(0)
            tdm_ops.tensor_wait(DQT_TW)
            _wg_sync()
            rb = _rdkv(ncur)
            rocdl.sched_barrier(0)
        new = [None] * (NQW * NDO_QK)
        for qh_ in range_constexpr(NQW):
            a_ds = fx.Vector.from_elements(ds_halves[qh_][0] + ds_halves[qh_][1], dtype=fx.BFloat16)
            for dtile in range_constexpr(NDO_QK):
                # r6.i2.g22: dQ^T = K^T dS^T (A = the K^T tr16 tile, B = the dS pack: the same
                # registers, A/B mirror images; bitwise on this card), so the C fragment holds q row
                # qh*16 + lane%16 x 8 consecutive d -- what the whole-line epilogue stages
                new[qh_ * NDO_QK + dtile] = rocdl.wmma_f32_16x16x32_bf16(
                    v8f, _ir(b_ks[dtile]), _ir(a_ds), acc[qh_ * NDO_QK + dtile], reuseA=False, reuseB=False
                ).result
            if const_expr(qh_ + 1 < NQW):
                ds_halves[qh_ + 1].append(
                    _smx(fx.Vector(s1[qh_ + 1]), fx.Vector(p1[qh_ + 1]), qh_ + 1, 1, kv0, False)
                )
                _sgb(NDO_QK)
            rocdl.sched_barrier(0)
        return new + [_ir(v) for v in rb]

    def _body(acc, pre, kv0, do_mask, pf_kv0, cur, nxo, ncur):
        if const_expr(not do_mask):
            return _body_ck(acc, pre, kv0, pf_kv0, cur, nxo, ncur)
        # Masked loop: S/dP WMMAs | DS phase | softmax VALU (hides the DS latency) + dQ.
        # TDM for tile min(i+2, N-1) into stage (i-1)%3: no wait, that stage's last readers
        # (dQ tr16 of i-1, readback in i-2) were consumed by WMMAs of i-1.
        rocdl.sched_barrier(0)
        _tdm_kv(pf_kv0, nxo)
        rocdl.sched_barrier(0)
        sp_acc = []
        for kt in range_constexpr(NKT):
            s_acc = [_ir(fx.Vector.filled(8, 0.0, fx.Float32)) for _ in range_constexpr(NQW)]
            p_acc = [_ir(fx.Vector.filled(8, 0.0, fx.Float32)) for _ in range_constexpr(NQW)]
            for dt in range_constexpr(NDT_MAX):
                _sdp(pre, kt, dt, s_acc, p_acc)
            sp_acc.append((s_acc, p_acc))

        def _softmax():
            ds_halves = [[] for _ in range_constexpr(NQW)]
            for kt in range_constexpr(NKT):
                s_acc, p_acc = sp_acc[kt]
                for qh_ in range_constexpr(NQW):
                    sv, pv_ = fx.Vector(s_acc[qh_]), fx.Vector(p_acc[qh_])
                    ds_halves[qh_].append(_smx(sv, pv_, qh_, kt, kv0, do_mask))
            return ds_halves

        # ONE DS phase: dQ B operands (tr16 of this stage's K), then retire the stage of tile
        # min(i+1, N-1) (outstanding: that stage and the one issued above; TDM retires in
        # order, so tensor_wait(DQT_TW) retires exactly the older one) and read it back.
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
            # k_dqg96: the softmax VALU covers the tr16 latency before the drain + sync (as in
            # _body_ck); the readback of ncur follows the sync.
            ds_halves = _softmax()
            rocdl.sched_barrier(0)
            tdm_ops.tensor_wait(DQT_TW)
            _wg_sync()
            rb = _rdkv(ncur)
            rocdl.sched_barrier(0)
        a_ds = [
            fx.Vector.from_elements(ds_halves[qh_][0] + ds_halves[qh_][1], dtype=fx.BFloat16)
            for qh_ in range_constexpr(NQW)
        ]
        new = [None] * (NQW * NDO_QK)
        for qh_ in range_constexpr(NQW):
            for dtile in range_constexpr(NDO_QK):
                new[qh_ * NDO_QK + dtile] = rocdl.wmma_f32_16x16x32_bf16(  # dQ^T (r6.i2.g22)
                    v8f,
                    _ir(b_ks[dtile]),
                    _ir(a_ds[qh_]),
                    acc[qh_ * NDO_QK + dtile],
                    reuseA=False,
                    reuseB=False,
                ).result
        return new + [_ir(v) for v in rb]

    NACC = NQW * NDO_QK

    def _mkloop(do_mask):
        @flyc.jit
        def kvloop(state, n, i0, nlast):
            # carried: acc[NACC] + readback A operands[NP] + cur (ring offset of stage i%3)
            final = state
            for it, carried in range(fx.Index(fx.Int32(0)), fx.Index(n), 1, init=state):
                st = list(carried)
                cur = fx.Int32(st[-1])
                ii = fx.Int32(it) + i0
                kk = ii + fx.Int32(TDM_DEPTH - 1)
                kk = (kk < nlast).select(kk, nlast)  # min(i+2, N-1)
                nxo = fx.Int32(
                    (cur == fx.Int32(0)).select(  # (i+2)%3 == (i-1)%3
                        fx.Int32((TDM_DEPTH - 1) * KV_B), cur - fx.Int32(KV_B)
                    )
                )
                ncur = fx.Int32(
                    (cur == fx.Int32((TDM_DEPTH - 1) * KV_B)).select(fx.Int32(0), cur + fx.Int32(KV_B))
                )  # (i+1)%3
                res = _body(
                    st[:NACC],
                    [fx.Vector(v) for v in st[NACC : NACC + NP]],
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

    # kv range of the WORKGROUP tile [q0g, q0g + BQWG) (== the wave's when nwave == 1): every
    # wave runs the same nfull full + (nkvt_eff - nfull) masked steps, so the barrier counts
    # match; rows of an earlier wave past their own diagonal are masked by do_mask (ds = 0).
    _lim = (q0g + fx.Int32(BQWG) + cshift + fx.Int32(KV_STEP - 1)) // fx.Int32(KV_STEP)
    _lim = (_lim < fx.Int32(1)).select(fx.Int32(1), _lim)
    _lim = (_lim < nkvt).select(_lim, nkvt)
    nkvt_eff = (causal != fx.Int32(0)).select(_lim, nkvt)
    _t = q0g + cshift + fx.Int32(1)
    _nf = (_t < fx.Int32(0)).select(fx.Int32(0), _t // fx.Int32(KV_STEP))
    _nf = (_nf < nkvt_eff).select(_nf, nkvt_eff)
    nfull = (causal != fx.Int32(0)).select(_nf, nkvt_eff)
    nlast = nkvt_eff - fx.Int32(1)  # >= 0: nkvt_eff >= 1

    # prologue: stage s = tile min(s, N-1) for s = 0..1; retire stage 0, read it back.
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

    init = [_ir(fx.Vector.filled(8, 0.0, fx.Float32)) for _ in range(NACC)]
    out = kvloop_full(init + [_ir(v) for v in rb0] + [fx.as_ir_value(fx.Int32(0))], nfull, fx.Int32(0), nlast)
    out = kvloop_mask(list(out), nkvt_eff - nfull, nfull, nlast)
    # retire the last two (clamped, never read) prefetches before the wave ends: the
    # workgroup's LDS must not be written after it is released.
    tdm_ops.tensor_wait(0)
    # dQ epilogue (r6.i2.g22): stage the dQ^T fragments as a row-major image in the dead ring, read
    # it back in whole-line order, and store 4 whole 128-B dq lines per buffer_store_b128.
    # nwave > 1: one more workgroup sync -- every wave's TDM landed (each waited 0 above) and every
    # wave's ring reads retired (_wg_sync's dscnt 0) before any image overwrites the ring.
    if const_expr(nwave > 1):
        _wg_sync()
        img = _lds0 + wv * fx.Int32(BQW * DQ_IMG_ROW_B)
    else:
        img = _lds0
    for qh_ in range_constexpr(NQW):
        for dtile in range_constexpr(NDO_QK):
            ov = fx.Vector(out[qh_ * NDO_QK + dtile])
            # C of dQ^T: lane (row, half) holds q row qh_*16 + row, d = dtile*16 + half*8 + [0, 8)
            llvm_dialect.store(
                fx.as_ir_value(
                    fx.Vector.from_elements(
                        [ov[si].to(fx.BFloat16) for si in range_constexpr(8)], dtype=fx.BFloat16
                    )
                ),
                create_llvm_ptr(
                    img
                    + (fx.Int32(qh_ * 16) + row) * fx.Int32(DQ_IMG_ROW_B)
                    + fx.Int32(dtile * 32)
                    + half * fx.Int32(16),
                    address_space=3,
                ),
            )
    lrow = lane // fx.Int32(8)  # 4 rows per store
    lcol = lane % fx.Int32(8)  # 16-B chunk of the 128-B line
    rd0 = img + lrow * fx.Int32(DQ_IMG_ROW_B) + lcol * fx.Int32(16)
    for rb in range_constexpr(BQW // 4):
        # vec8 index of (q row q0 + 4rb + lrow, head qh, d = 0) in dq [B, Sq, Hq, D_QK]
        gt = ((bat * Sq + q0 + fx.Int32(rb * 4) + lrow) * Hq + qh) * fx.Int32(DV8_QK) + lcol
        for li in range_constexpr(D_QK // DQ_LINE_EL):
            vv = fx.Vector(
                llvm_dialect.load(
                    v8b, create_llvm_ptr(rd0 + fx.Int32(rb * 4 * DQ_IMG_ROW_B + li * 128), address_space=3)
                )
            )
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
    _ = MODULE_KNOBS  # cache key only (see MODULE_KNOBS)
    _dqg_tdm_impl(
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


@flyc.kernel(known_block_size=[32, 1, 1])
def k_dqg48(
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
):
    _ = MODULE_KNOBS  # cache key only (see MODULE_KNOBS)
    _dqg_tdm_impl(
        Q, K, V, DO, O, LSE, DEL, DQ, scale, Sq, Skv, Hq, Hkv, G, nkvt, cshift, causal, q_off, ntile, NQW48
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
    # grid = (hg q heads of a head group, ntile DQ_BQW-query tiles from q_off, ngz = B*Hq/hg
    # head groups); hg = Hq is (Hq, ntile, B), r3_a's grid
    k_dqg(
        Q, K, V, DO, O, LSE, DEL, DQ, scale, Sq, Skv, Hq, Hkv, G, nkvt, cshift, causal, q_off, ntile, hg
    ).launch(grid=(hg, ntile, ngz), block=(32, 1, 1), stream=stream)


@flyc.jit
def launch_dqg48(
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
    ngrp: fx.Int32,
    nb: fx.Int32,
    stream: fx.Stream,
):
    # grid = (Hq q heads, ntile DQ_BQW48-query tiles from q_off, B)
    k_dqg48(
        Q, K, V, DO, O, LSE, DEL, DQ, scale, Sq, Skv, Hq, Hkv, G, nkvt, cshift, causal, q_off, ntile
    ).launch(grid=(ngrp, ntile, nb), block=(32, 1, 1), stream=stream)


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
    _ = MODULE_KNOBS  # cache key only (see MODULE_KNOBS)
    rocdl.disable_xdl_arb_stall()  # lock_simd (SCHED_MODE.DISABLE_XDL_ARB_STALL); PROVENANCE.md, round 5
    _dqg_tdm_impl(
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
    # grid = (hg q heads of a head group, ntile DQ_BQW96-query tiles from q_off, ngz = B*Hq/hg
    # head groups), DQ_NWAVE waves per workgroup; hg = Hq is (Hq, ntile, B), r3_a's grid
    k_dqg96(
        Q, K, V, DO, O, LSE, DEL, DQ, scale, Sq, Skv, Hq, Hkv, G, nkvt, cshift, causal, q_off, ntile, hg
    ).launch(grid=(hg, ntile, ngz), block=(DQ_NWAVE * WAVE, 1, 1), stream=stream)


# FlyDSL keys its JIT cache by the launcher's and its kernels' source, their closure scalars, and the
# module globals a static walk of their top-level code finds; a constant read only inside a nested
# helper (TDM_DEPTH, KH0_SALU, DQ_BARRIER_FENCE, ...) is not found, so two builds that differ only in
# its value would share a cached binary. Every kernel above reads MODULE_KNOBS, which puts the value
# of every knob of this file into the key. It is taken at import: edit a knob in this file, not at
# run time (FlyDSL refuses a captured global that changes after the first compile anyway).
MODULE_KNOBS = _module_knobs()
