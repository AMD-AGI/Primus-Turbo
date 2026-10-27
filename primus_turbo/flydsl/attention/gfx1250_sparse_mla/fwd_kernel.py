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

"""DeepSeek-V4 sparse-MLA attention forward for gfx1250 (MI455X): wave32, WMMA 16x16x32.

Same contract as ``sparse_mla_fwd_triton``: q ``[T, H, 576]``, a single MQA latent kv
``[NKV, 1, 576]`` whose first 512 columns are both K and V (the 64 rope columns are a zero
pad and take no part), per-token top-k row indices (``-1`` = unused slot), optional per-head
sink. Returns o ``[T, H, 512]`` and a sink-inclusive natural-log lse ``[T, H]``.

A workgroup owns one token and ``32 * waves`` of its heads, one wave per SIMD. Each wave owns
32 heads as two 16-head groups, so every K / V fragment read from LDS feeds two WMMAs (the
kernel is LDS-bandwidth bound; ~950 VGPRs). Each wave runs the transposed problem
``S^T = K Q^T``, ``O^T += V^T P^T`` (see ``flash_attn_fwd_gfx1250``): a lane holds one head,
so the softmax is in-lane, and V^T comes from ``ds_load_tr16_b128`` reads of the K tile.

The top-k latent rows are gathered into a 3-stage LDS ring by the Tensor Data Mover (gather
mode: 8 x i32 or 16 x i16 row indices per op, read with scalar loads); every wave issues its
share of each tile. The loop is software-pipelined: iteration t computes the scores of tile
t+1 interleaved with the P*V of tile t, LDS operands are loaded a batch of WMMA steps ahead, and
``sched_barrier`` pins the batch boundaries against the machine scheduler.

The cr0 / cr128 bands (every token's top-k is its 128-row sliding window, plus for cr128 a
causal pool of T / 128 compressed rows) have their own kernel: a workgroup owns a block of
consecutive tokens, gathers their shared window and the pool to LDS once, and walks the tokens
with per-lane row offsets into the window and closed-form masks (no top-k reads).
"""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm, scf
from flydsl._mlir.dialects import rocdl as _rocdl
from flydsl.expr import const_expr, gpu, range_constexpr
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec
from flydsl.expr.utils.arith import _to_raw as _raw

from .helper import (
    LN2,
    LOG2E,
    WAVE,
    buf_load,
    buf_load_b128,
    buf_store,
    cat,
    exp2,
    fadd,
    fmax,
    fmul,
    fsub,
    lds_load,
    lds_load_tr16,
    lds_store,
    log2,
    make_rsrc,
    rcp,
    swap_halves,
    tensor_addr,
    to_bf16,
    uniform,
    vec_type,
    vmul_scalar,
    wmma_bf16,
)

_NEG_INF = float("-inf")
DV = 512  # latent width: K and V
DQK = 576  # q / kv row length (latent + zero rope pad)
# gfx12 expert scheduling: the backend models the XDL / TRANS / VALU pipes and runs the wave in
# hardware SCHED_MODE 2 (5-8% faster on these WMMA-bound kernels).
COMPILE_HINTS = {"llvm_options": {"amdgpu-expert-scheduling-mode": True}}

NBUF = 3  # TDM ring stages; deeper rings lose to the extra LDS write traffic
PREFETCH = 8  # cr0 / cr128 kernel with batch=0: LDS operand loads run this many WMMA steps ahead
PIN = 4  # cr0 / cr128 kernel with batch=0: sched_barrier every PIN steps
# Top-k kernel: LDS operands load in batches of BATCH steps, one batch ahead (2 * BATCH fragments
# live; 8 fits ~1008 VGPRs, 12 spills). Under expert scheduling LLVM 24 adds va_vdst(0) before a
# ds_load that overwrites a WMMA's A operand; batching cuts those waits (114 -> 69 per tile).
BATCH = 8
_RESCALE_LOG2 = 8.0  # cr4 path: lag of the running max (log2 units) before O is rescaled
# TDM group-1 dword 0: 2-byte elements, pad 4 dwords after every 256 (one 512-elem row)
_TDM_G1_CFG = (1 << 16) | (1 << 20) | (7 << 22) | (3 << 25)
ROW = DV * 2 + 16  # LDS row pitch (bytes) the padding above produces
_PF_L2 = (2 << 3) | 1  # global_prefetch cache policy: SCOPE_DEV (L2, not the WGP cache), non-speculative


def _v_i32(vals):
    return Vec.from_elements([fx.Int32(v) for v in vals], fx.Int32)


def _tdm_gather(lds_addr, kv_addr, words, n_idx, n_rows, row_len=DQK):
    """Gather the first 512 columns of ``n_idx`` rows of length ``row_len`` (indices packed in
    8 dwords: 8 x i32 or 16 x i16) to LDS."""
    hdr = (fx.Int32(-(1 << 30)) if n_idx == 8 else fx.Int32(-(1 << 31))) | fx.Int32(1)
    g0 = _v_i32([hdr, lds_addr, fx.Int32(kv_addr), fx.Int32(kv_addr >> fx.Int64(32)) | fx.Int32(-(1 << 31))])
    g1 = _v_i32(
        [
            _TDM_G1_CFG,
            row_len << 16,
            (n_rows & fx.Int32(0xFFFF)) << fx.Int32(16),
            ((n_rows >> fx.Int32(16)) & fx.Int32(0xFFFF)) | fx.Int32(DV << 16),
            n_idx,
            row_len,
            0,
            0,
        ]
    )
    # cache policy 0 (CU scope): the opus default of device scope bypasses L2 reuse
    _rocdl.tensor_load_to_lds(
        _raw(g0), _raw(g1), _raw(_v_i32(words[:4])), _raw(_v_i32(words[4:])), _raw(_v_i32([0] * 8))
    )


def _if_wave(cond, vals, then_fn, else_fn):
    """scf.if threading ``vals`` through both arms, for a wave-uniform ``cond``."""
    _v = [_raw(v) for v in vals]
    op = scf.IfOp(_raw(cond), [x.type for x in _v], has_else=True)

    def arm(region, body):
        if not region.blocks:
            region.blocks.append()
        blk = region.blocks[0]
        ops = list(blk.operations)
        if ops:
            ops[-1].erase()  # the builder's placeholder yield
        with ir.InsertionPoint(blk):
            scf.YieldOp([_raw(x) for x in body()])

    arm(op.regions[0], then_fn)
    arm(op.regions[1], lambda: (else_fn(), _v)[1])
    return list(op.results)


def _like(raw, ref):
    """Wrap an ``ir.Value`` as the FlyDSL type of ``ref``."""
    return Vec(raw) if isinstance(ref, Vec) else type(ref)(raw)


def _ballot(pred):
    """Lane mask of ``pred`` (raw i32, wave-uniform)."""
    m = _rocdl.ballot(res=ir.IntegerType.get_signless(32), pred=_raw(pred))
    return m if isinstance(m, ir.Value) else m.result


def _highest_lane_p1(pred):
    """1 + the highest lane of the wave whose ``pred`` holds (0 if none); wave-uniform."""
    i32 = ir.IntegerType.get_signless(32)
    m = _ballot(pred)
    no = llvm.ConstantOp(
        ir.IntegerType.get_signless(1), ir.IntegerAttr.get(ir.IntegerType.get_signless(1), 0)
    )
    lz = fx.Int32(llvm.call_intrinsic(i32, "llvm.ctlz.i32", [m, no.result], [], []))
    return fx.Int32(32) - lz


def _l2_prefetch(base, nbytes, tid, block):
    """Prefetch ``nbytes`` at ``base`` into L2 (SCOPE_DEV, non-speculative), one 128 B line per
    lane per instruction."""
    for k in range_constexpr(-(-nbytes // (block * 128))):
        off = (tid + k * block) * 128
        off = fx.Int32((off < fx.Int32(nbytes)).select(off, fx.Int32(nbytes - 128)))
        ptr = llvm.IntToPtrOp(ir.Type.parse("!llvm.ptr<1>"), _raw(base + fx.Int64(off))).result
        llvm.call_intrinsic(None, "llvm.amdgcn.global.prefetch", [ptr, _raw(fx.Int32(_PF_L2))], [], [])


def _sbuf_load_v8i32(rsrc, byte_off):
    """Eight consecutive i32 of a buffer into SGPRs (the TDM index operands must be scalar)."""
    rsrc = rsrc if isinstance(rsrc, ir.Value) else rsrc.result
    v4 = llvm.bitcast(vec_type(4, fx.Int32), llvm.ptrtoint(ir.IntegerType.get_signless(128), rsrc))
    return Vec(
        llvm.call_intrinsic(
            vec_type(8, fx.Int32),
            "llvm.amdgcn.s.buffer.load.v8i32",
            [v4, _raw(fx.Int32(byte_off)), _raw(fx.Int32(0))],
            [],
            [],
        )
    )


# Up to this many kv tiles per token, the per-token q load and pipeline fill are exposed:
# workgroups become persistent and load the next token's q under the last P*V.
_PERSIST_MAX_TILES = 8


@functools.lru_cache(maxsize=None)
def _num_cus(device):
    return torch.cuda.get_device_properties(device).multi_processor_count


def _default_config(num_heads, topk):
    hpw = 32 if num_heads % 32 == 0 else 16
    waves = min(num_heads // hpw, 4)
    block_k = 64 if waves == 4 and topk % 64 == 0 else 32
    return hpw, waves, block_k


def _run_steps(steps, batch, prefetch=0, pin=0):
    """Run ``(load, mma)`` steps. ``batch > 0``: batch b's WMMAs run while batch b+1's operands
    load into other registers, ``sched_barrier`` at batch edges, so a reload overwrites registers
    some WMMA read once per batch, not once per step (LLVM 24 puts a va_vdst(0) before each such
    load; see BATCH). ``batch == 0``: loads ``prefetch`` steps ahead, ``sched_barrier`` every ``pin``."""
    n = len(steps)
    if const_expr(batch > 0):
        cur = [steps[i][0]() for i in range_constexpr(min(batch, n))]
        for b0 in range_constexpr(0, n, batch):
            nxt = [steps[i][0]() for i in range_constexpr(b0 + batch, min(b0 + 2 * batch, n))]
            fx.rocdl.sched_barrier(0)
            for j in range_constexpr(len(cur)):
                steps[b0 + j][1](cur[j])
            fx.rocdl.sched_barrier(0)
            cur = nxt
        return
    q = [steps[i][0]() for i in range_constexpr(min(prefetch, n))]
    for i in range_constexpr(n):
        a = q.pop(0)
        if const_expr(i + prefetch < n):
            q.append(steps[i + prefetch][0]())
        steps[i][1](a)
        if const_expr((i + 1) % pin == 0):
            fx.rocdl.sched_barrier(0)


@functools.lru_cache(maxsize=64)
def build_sparse_mla_fwd_gfx1250(
    num_heads,
    topk,
    scale,
    has_sink,
    idx16=False,
    hpw=None,
    waves=None,
    block_k=None,
    persist=None,
    l2_prefetch=True,
):
    """Launcher for fixed (H, topk, scale, sink); see ``sparse_mla_fwd_gfx1250``.

    ``idx16`` packs 16-bit row indices (16 rows per TDM op); only valid when NKV <= 65536.
    ``persist`` (default: at most ``_PERSIST_MAX_TILES`` kv tiles per token) loops each
    workgroup over tokens; otherwise one token per workgroup, which after its last tile
    prefetches into L2 (``l2_prefetch``) the q rows and indices of the token that takes over its
    slot next. Only at one workgroup per CU: with two, the sibling already hides the prologue.
    """
    H = num_heads
    d_hpw, d_waves, d_bk = _default_config(H, topk)
    hpw = hpw or d_hpw
    WAVES = waves or d_waves
    BN = block_k or d_bk
    G = hpw // 16  # 16-head groups per wave
    HB = WAVES * hpw
    if hpw % 16 or H % HB:
        raise ValueError(f"num_heads {H} must split into {WAVES} waves of {hpw} heads")
    if topk % BN or BN % 32:
        raise ValueError(f"topk {topk} must be a multiple of the kv tile {BN}")
    BLOCK = WAVES * WAVE
    NT = topk // BN
    KT = BN // 16  # 16-row kv sub-tiles
    PC = BN // 32  # 32-row P*V reduction steps
    KS = DV // 32  # QK reduction steps
    DT = DV // 16  # d tiles of O^T
    TILE_B = BN * ROW
    LDS_B = NBUF * TILE_B
    # O goes out through a per-wave LDS stage of 16 head rows, when it fits at full occupancy.
    STG_B = 16 * ROW
    STAGE = (LDS_B + WAVES * STG_B) * (4 // WAVES) <= _LDS_MAX
    RPO = 16 if idx16 else 8  # rows per TDM op
    if (BN // RPO) % WAVES:
        raise ValueError(f"kv tile {BN} does not split into TDM ops over {WAVES} waves")
    OPW = BN // RPO // WAVES  # TDM ops per wave per tile (queue depth is 3)
    IW = OPW * (RPO // 8)  # index octets per wave per tile
    c_log2 = float(scale) * LOG2E
    if persist is None:
        persist = NT <= _PERSIST_MAX_TILES

    @flyc.kernel(known_block_size=[BLOCK, 1, 1])
    def sparse_mla_fwd_gfx1250_kernel(
        Q: fx.Tensor,
        KV: fx.Tensor,
        TOPK: fx.Tensor,
        SINK: fx.Tensor,
        O: fx.Tensor,  # noqa: E741
        LSE: fx.Tensor,
        NKV: fx.Int32,
        N_TOK: fx.Int32,
        NPROG: fx.Int32,
    ):
        tid = fx.Int32(fx.thread_idx.x)
        wave = uniform(tid // WAVE)
        lane = tid % WAVE
        l16 = lane % 16
        hi = lane // 16
        l8 = lane % 8
        bid = fx.Int32(fx.block_idx.x)
        if const_expr(not persist):
            # One token per workgroup, last token first: under a causal index layout the late
            # tokens walk the most tiles, and launching them first shortens the tail.
            bid = N_TOK - fx.Int32(1) - bid
        h_lane = [fx.Int32(fx.block_idx.y) * HB + wave * hpw + g * 16 + l16 for g in range_constexpr(G)]
        # Persistent: this workgroup owns tokens bid, bid + NPROG, ... (NPROG <= N_TOK).
        n_loc = (N_TOK - bid + NPROG - fx.Int32(1)) // NPROG

        def tok_of(j):
            # Past the last token, keep re-fetching it into buffers that are never read.
            return bid + fx.Int32((j < n_loc).select(j, n_loc - fx.Int32(1))) * NPROG

        arena = fx.SharedAllocator(static=False)
        arena.allocate(LDS_B + (WAVES * STG_B if STAGE else 0))
        lds0 = fx.Int32(fx.Int64(fx.ptrtoint(arena.base_ptr)))
        stg = lds0 + LDS_B + wave * STG_B

        rsrc_idx = make_rsrc(tensor_addr(TOPK), N_TOK * (topk * 4))
        kv_addr = tensor_addr(KV)
        v4i = vec_type(4, fx.Int32)

        # One token per workgroup: walk its tiles only up to the last one holding a valid slot
        # (lane c checks tiles c, c + 32, ...). Tile 0 always runs.
        n_tile = fx.Int32(NT)
        if const_expr(not persist):
            n_tile = fx.Int32(1)
            for c0 in range_constexpr(0, NT, WAVE):
                tt = fx.Int32(c0) + lane
                live = tt < fx.Int32(NT)
                ct = fx.Int32(live.select(tt, fx.Int32(0)))
                mx = fx.Int32(-1)
                for k in range_constexpr(BN // 4):
                    iv = Vec(buf_load(rsrc_idx, (bid * topk + ct * BN) * 4 + k * 16, v4i))
                    for e in range_constexpr(4):
                        mx = fx.Int32((iv[e] > mx).select(iv[e], mx))
                hit = _highest_lane_p1(fx.Int32(live.select(mx, fx.Int32(-1))) >= fx.Int32(0))
                n_tile = fx.Int32((hit > fx.Int32(0)).select(fx.Int32(c0) + hit, n_tile))
            n_tile = uniform(n_tile)

        # The TDM tile stream runs across tokens: tile g is tile g % NT of local token g // NT,
        # so the next token's first tiles land while the current one finishes.
        def load_rows(g):
            base = tok_of(g // NT) * topk + (g % NT) * BN + wave * (OPW * RPO)
            return [_sbuf_load_v8i32(rsrc_idx, (base + j * 8) * 4) for j in range_constexpr(IW)]

        def issue(t, octs):
            buf_base = lds0 + (t % NBUF) * TILE_B
            for k in range_constexpr(OPW):
                rows = []
                for h in range_constexpr(RPO // 8):
                    rows += [octs[k * (RPO // 8) + h][i] for i in range_constexpr(8)]
                if const_expr(idx16):
                    words = [
                        (rows[2 * i] & fx.Int32(0xFFFF)) | (rows[2 * i + 1] << fx.Int32(16))
                        for i in range_constexpr(8)
                    ]
                else:
                    words = rows
                # An unused slot (-1) is out of range: the row lands as zeros, masked below.
                _tdm_gather(buf_base + (wave * OPW + k) * (RPO * ROW), kv_addr, words, RPO, NKV)

        # The prologue gathers go out before the Q loads so the two overlap.
        pro = [load_rows(fx.Int32(s)) for s in range_constexpr(NBUF)]
        for s in range_constexpr(NBUF - 1):
            issue(fx.Int32(s), pro[s])

        # ---- Q fragments (B operand of S^T = K Q^T): lane holds head h_lane[g].
        def load_q(j):
            # Past the last token the range is empty: the loads return zeros without memory traffic.
            n_rec = fx.Int32((j < n_loc).select(fx.Int32(H * DQK * 2), fx.Int32(0)))
            rsrc_q = make_rsrc(tensor_addr(Q) + fx.Int64(tok_of(j)) * (H * DQK * 2), n_rec)
            out = []
            for g in range_constexpr(G):
                q_off = h_lane[g] * (DQK * 2) + hi * 16
                for ks in range_constexpr(KS):
                    out.append(
                        cat(
                            buf_load_b128(rsrc_q, q_off + ks * 64),
                            buf_load_b128(rsrc_q, q_off + ks * 64 + 32),
                        )
                    )
            return out

        q_first = load_q(fx.Int32(0))

        k_lane = l16 * ROW + hi * 16
        v_lane = (hi * 8 + l8) * ROW + (l16 // 8) * 16
        v8e = vec_type(8, fx.BFloat16)

        def valid_of(tok, t):
            # Validity of the kv rows this lane's score elements cover: kt*16 + hi*8 + i.
            out = []
            for kt in range_constexpr(KT):
                o = (tok * topk + t * BN + kt * 16 + hi * 8) * 4
                iv = cat(Vec(buf_load(rsrc_idx, o, v4i)), Vec(buf_load(rsrc_idx, o + 16, v4i)))
                out.append([iv[i] >= fx.Int32(0) for i in range_constexpr(8)])
            return out

        # A step is (load: () -> fragment, mma: fragment -> None).
        def qk_steps(kbase, q_frag):
            s = [[Vec.filled(8, 0.0, fx.Float32) for _ in range_constexpr(KT)] for _ in range_constexpr(G)]
            steps = []
            for kt in range_constexpr(KT):
                for ks in range_constexpr(KS):

                    def ld(kt=kt, ks=ks):
                        off = kbase + k_lane + (kt * 16 * ROW + ks * 64)
                        return cat(lds_load(off, v8e), lds_load(off + 32, v8e))

                    def mma(a, kt=kt, ks=ks):
                        for g in range_constexpr(G):
                            s[g][kt] = wmma_bf16(a, q_frag[g][ks], s[g][kt])

                    steps.append((ld, mma))
            return s, steps

        def pv_steps(vbase, p_frag, o):
            steps = []
            for dt in range_constexpr(DT):
                for pc in range_constexpr(PC):

                    def ld(dt=dt, pc=pc):
                        base = vbase + v_lane + (pc * 32 * ROW + dt * 32)
                        return cat(lds_load_tr16(base), lds_load_tr16(base + 16 * ROW))

                    def mma(a, dt=dt, pc=pc):
                        for g in range_constexpr(G):
                            o[g][dt] = wmma_bf16(a, p_frag[g][pc], o[g][dt])

                    steps.append((ld, mma))
            return steps

        def run_steps(steps):
            _run_steps(steps, BATCH)

        def softmax(s, valid, m_run, l_run):
            vals = []
            for kt in range_constexpr(KT):
                sv = vmul_scalar(s[kt], c_log2)
                for i in range_constexpr(8):
                    vals.append(fx.Float32(valid[kt][i].select(sv[i], fx.Float32(_NEG_INF))))
            tmax = vals[0]
            for v in vals[1:]:
                tmax = fmax(tmax, v)
            tmax = fmax(tmax, swap_halves(tmax))
            # The running max moves only once some head's tile max passes it by more than
            # _RESCALE_LOG2 (p stays <= 2^_RESCALE_LOG2); ``need`` is wave-uniform.
            need = fx.Int32(_ballot(fsub(tmax, m_run) > fx.Float32(_RESCALE_LOG2))) != fx.Int32(0)
            m_new = fx.Float32(need.select(fmax(m_run, tmax), m_run))
            m_use = fx.Float32((m_new == fx.Float32(_NEG_INF)).select(fx.Float32(0.0), m_new))
            alpha = exp2(fsub(m_run, m_use))
            ps = [exp2(fsub(v, m_use)) for v in vals]
            rs = ps[0]
            for p in ps[1:]:
                rs = fadd(rs, p)
            l_new = fadd(fmul(l_run, alpha), rs)
            packs = [
                to_bf16(Vec.from_elements(ps[kt * 8 : kt * 8 + 8], fx.Float32)) for kt in range_constexpr(KT)
            ]
            p_frags = [cat(packs[2 * pc], packs[2 * pc + 1]) for pc in range_constexpr(PC)]
            return m_new, l_new, alpha, p_frags, need

        # Loop state: m[G], l[G], o[G][DT], p[G][PC], then the next tile's index octets.
        NS = G * (2 + DT + PC)

        def pack_state(m, l, o, p):  # noqa: E741
            out = list(m) + list(l)
            for g in range_constexpr(G):
                out += list(o[g])
            for g in range_constexpr(G):
                out += list(p[g])
            return out

        def unpack_state(st):
            m = [st[g] for g in range_constexpr(G)]
            l = [st[G + g] for g in range_constexpr(G)]  # noqa: E741
            o = [[st[2 * G + g * DT + dt] for dt in range_constexpr(DT)] for g in range_constexpr(G)]
            p = [[st[2 * G + G * DT + g * PC + pc] for pc in range_constexpr(PC)] for g in range_constexpr(G)]
            return m, l, o, p

        NQ = G * KS  # q fragments carried across tokens

        def token(ji, tsm):
            tok = tok_of(ji)
            g0 = ji * NT  # stream index of this token's tile 0
            qf = [tsm[i] for i in range_constexpr(NQ)]
            q_frag = [qf[g * KS : (g + 1) * KS] for g in range_constexpr(G)]
            octs_in = [tsm[NQ + i] for i in range_constexpr(IW)]

            # ---- prologue: scores and P of tile 0.
            _rocdl.s_wait_tensorcnt((NBUF - 2) * OPW)
            gpu.barrier()
            s0, steps0 = qk_steps(lds0 + (g0 % NBUF) * TILE_B, q_frag)
            run_steps(steps0)
            v0 = valid_of(tok, fx.Int32(0))
            m, l, p = [], [], []  # noqa: E741
            for g in range_constexpr(G):
                mg, lg, _, pg, _ = softmax(s0[g], v0, fx.Float32(_NEG_INF), fx.Float32(0.0))
                m.append(mg)
                l.append(lg)
                p.append(pg)
            o = [[Vec.filled(8, 0.0, fx.Float32) for _ in range_constexpr(DT)] for _ in range_constexpr(G)]
            st = pack_state(m, l, o, p) + octs_in

            if const_expr(NT > 1):
                # A static trip count with the tiles past n_tile branched over: a dynamic bound
                # adds a loop bypass that tips the flash config into spills.
                for t, sm in range(fx.Index(0), fx.Index(NT - 1), fx.Index(1), init=st):
                    ti = fx.Int32(t)
                    gt = g0 + ti

                    def tile_step(sm=sm, ti=ti, gt=gt):
                        m, l, o, p = unpack_state(sm)  # noqa: E741
                        # Tile t+1 landed (tiles complete in issue order), and every wave is done
                        # with tile t-1's buffer, which the gather below overwrites.
                        _rocdl.s_wait_tensorcnt((NBUF - 3) * OPW)
                        gpu.barrier()
                        issue(gt + NBUF - 1, [sm[NS + i] for i in range_constexpr(IW)])
                        nxt = load_rows(gt + NBUF)
                        valid = valid_of(tok, ti + 1)
                        s_n, qk = qk_steps(lds0 + ((gt + 1) % NBUF) * TILE_B, q_frag)
                        pv = pv_steps(lds0 + (gt % NBUF) * TILE_B, p, o)
                        seq = []
                        for i in range_constexpr(max(len(qk), len(pv))):
                            if const_expr(i < len(qk)):
                                seq.append(qk[i])
                            if const_expr(i < len(pv)):
                                seq.append(pv[i])
                        run_steps(seq)
                        m2, l2, p2, a2 = [], [], [], []
                        need = None
                        for g in range_constexpr(G):
                            mg, lg, ag, pg, ng = softmax(s_n[g], valid, m[g], l[g])
                            m2.append(mg)
                            l2.append(lg)
                            p2.append(pg)
                            a2.append(ag)
                            need = ng if need is None else need | ng
                        flat = [x for g in range_constexpr(G) for x in o[g]]

                        def rescale():
                            return [
                                vmul_scalar(o[g][dt], a2[g])
                                for g in range_constexpr(G)
                                for dt in range_constexpr(DT)
                            ]

                        flat = [
                            _like(r, x) for r, x in zip(_if_wave(need, flat, rescale, lambda: None), flat)
                        ]
                        o = [flat[g * DT : (g + 1) * DT] for g in range_constexpr(G)]
                        return pack_state(m2, l2, o, p2) + nxt

                    if const_expr(persist):
                        new_st = tile_step()
                    else:
                        out = _if_wave(ti + fx.Int32(1) < n_tile, list(sm), tile_step, lambda: None)
                        new_st = [_like(r, s) for r, s in zip(out, sm)]
                    res = yield new_st
                st = [res[i] for i in range_constexpr(NS + IW)]

            m, l, o, p = unpack_state(st)  # noqa: E741
            nxt = None
            if const_expr(not persist and l2_prefetch and WAVES == 4):
                # Tokens launch last first and NPROG workgroups fill the GPU: token tok - NPROG
                # takes over this slot next. Warm L2 with its q rows and indices.
                nx = tok - NPROG
                nx = fx.Int32((nx >= fx.Int32(0)).select(nx, tok))
                _l2_prefetch(tensor_addr(Q) + fx.Int64(nx) * (H * DQK * 2), H * DQK * 2, tid, BLOCK)
                _l2_prefetch(tensor_addr(TOPK) + fx.Int64(nx) * (topk * 4), topk * 4, tid, BLOCK)
            if const_expr(persist):
                # Every wave is past tile NT-2, whose buffer takes the next token's tile 1; the next
                # token's q loads into fresh registers (this token's q is dead) under the last P*V.
                gpu.barrier()
                issue(g0 + NT + NBUF - 2, [st[NS + i] for i in range_constexpr(IW)])
                nxt = load_q(ji + 1) + load_rows(g0 + NT + NBUF - 1)
            run_steps(pv_steps(lds0 + ((g0 + n_tile - fx.Int32(1)) % NBUF) * TILE_B, p, o))

            rsrc_o = make_rsrc(tensor_addr(O) + fx.Int64(tok) * (H * DV * 2), H * DV * 2)
            rsrc_lse = make_rsrc(tensor_addr(LSE) + fx.Int64(tok) * (H * 4), H * 4)
            for g in range_constexpr(G):
                l_tot = fadd(l[g], swap_halves(l[g]))
                if const_expr(has_sink):
                    rsrc_sink = make_rsrc(tensor_addr(SINK), H * 4)
                    sink_log2 = fmul(fx.Float32(buf_load(rsrc_sink, h_lane[g] * 4, T.f32)), LOG2E)
                    mf = fmax(m[g], sink_log2)
                    af = exp2(fsub(m[g], mf))
                    l_out = fadd(fmul(l_tot, af), exp2(fsub(sink_log2, mf)))
                    o_scale = fmul(af, rcp(l_out))
                else:
                    mf = m[g]
                    l_out = l_tot
                    o_scale = fx.Float32((l_tot > fx.Float32(0.0)).select(rcp(l_tot), fx.Float32(0.0)))
                if const_expr(STAGE):
                    # Lane (head l16, half hi) holds d = dt * 16 + hi * 8 + [0, 8) of its head; the
                    # read-back gives each store half a head row. LDS ops of a wave run in order,
                    # so the next group's writes cannot pass this group's reads.
                    for dt in range_constexpr(DT):
                        o16 = to_bf16(vmul_scalar(o[g][dt], o_scale))
                        lds_store(o16, stg + l16 * ROW + hi * 16 + dt * 32)
                    o_off = (h_lane[g] - l16) * (DV * 2) + lane * 16
                    for r in range_constexpr(DT):
                        row = lds_load(stg + (r // 2) * ROW + (r % 2) * 512 + lane * 16, v4i)
                        buf_store(row, rsrc_o, o_off + (r // 2) * (DV * 2) + (r % 2) * 512)
                else:
                    o_off = h_lane[g] * (DV * 2) + hi * 16
                    for dt in range_constexpr(DT):
                        o16 = to_bf16(vmul_scalar(o[g][dt], o_scale))
                        buf_store(o16.bitcast(fx.Int32), rsrc_o, o_off + dt * 32)
                lse = fmul(fadd(mf, log2(l_out)), LN2)
                # Lanes 16..31 hold the same heads; push their store out of range.
                buf_store(lse, rsrc_lse, fx.Int32((hi == 0).select(h_lane[g] * 4, fx.Int32(0x7FFFFFF0))))
            return nxt

        if const_expr(persist):
            for j, tsm in range(fx.Index(0), fx.Index(n_loc), fx.Index(1), init=q_first + pro[NBUF - 1]):
                _ = yield token(fx.Int32(j), tsm)
        else:
            token(fx.Int32(0), q_first + pro[NBUF - 1])
        # The over-issued gathers must land before the workgroup's LDS is released.
        _rocdl.s_wait_tensorcnt(0)

    @flyc.jit
    def launch_sparse_mla_fwd_gfx1250(
        Q: fx.Tensor,
        KV: fx.Tensor,
        TOPK: fx.Tensor,
        SINK: fx.Tensor,
        O: fx.Tensor,  # noqa: E741
        LSE: fx.Tensor,
        n_tok: fx.Int32,
        NKV: fx.Int32,
        n_prog: fx.Int32,
        n_grid: fx.Int32,
        stream: fx.Stream = fx.Stream(None),  # noqa: B008
    ):
        sparse_mla_fwd_gfx1250_kernel(
            Q, KV, TOPK, SINK, O, LSE, NKV, n_tok, n_prog, value_attrs={"rocdl.waves_per_eu": 1}
        ).launch(grid=(fx.Index(n_grid), H // HB, 1), block=(BLOCK, 1, 1), stream=stream)

    compiled = {}

    def run(q, kv, topk_indices, sink, o, lse, stream=None):
        n_tok = int(q.shape[0])
        # One wave per SIMD: 4 // WAVES workgroups per CU (persistent: each loops over its tokens;
        # otherwise the L2 prefetch distance).
        n_prog = min(n_tok, _num_cus(q.device) * (4 // WAVES))
        n_grid = n_prog if persist else n_tok
        args = (
            q,
            kv,
            topk_indices,
            o if sink is None else sink,
            o,
            lse,
            n_tok,
            int(kv.shape[0]),
            n_prog,
            n_grid,
        )
        fn = compiled.get("fn")
        if fn is None:
            fn = flyc.compile[COMPILE_HINTS](launch_sparse_mla_fwd_gfx1250, *args, fx.Stream(stream))
            compiled["fn"] = fn
        return fn(*args, stream)

    return run


_SWA_W = 128  # cr0 band: slot s of token t is kv row t - 127 + s (rows < 0 unused)
_SWA_POOL_MAX = 128  # cr128 pool rows held in LDS next to the window
_LDS_MAX = 320 * 1024


@functools.lru_cache(maxsize=64)
def build_sparse_mla_fwd_gfx1250_swa(
    num_heads, scale, has_sink, block_t=16, block_k=None, pool_p=0, pool_cr=0, prefetch=PREFETCH, batch=6
):
    """Launcher for the cr0 / cr128 bands; see ``sparse_mla_fwd_gfx1250``.

    A workgroup owns ``block_t`` consecutive tokens and loads their shared kv window
    (``block_t + 127`` rows) to LDS once; token ``t0 + r`` reads its 128 slots at row offset
    ``r`` (per-lane fragment addresses), and the band mask is closed-form (no top-k reads).
    ``pool_p > 0`` adds cr128's causal pool: kv rows ``T + p`` (loaded once per workgroup after
    the window), visible to token t iff ``p < (t + 1) // pool_cr``.

    Four waves of 32 heads; with fewer than 128 heads the head groups repeat over token lanes
    (wave pairs on alternate tokens of the block) that share the window.

    ``batch``: LDS operands load in batches of that many steps (``_run_steps``); 6 beats 4 / 8
    without spills. 0 restores the ``prefetch`` / ``PIN`` stream.
    """
    H = num_heads
    hpw = 32
    WAVES = 4
    NHG = min(H // hpw, WAVES)  # head groups (waves) per token
    TL = WAVES // NHG  # token lanes
    # 64-row tiles only without a pool (with one, they spill; 64-head cr0 spills on 32-row ones)
    BN = block_k or (64 if pool_p == 0 else 32)
    G = hpw // 16
    HB = NHG * hpw
    if H % HB or WAVES % NHG:
        raise ValueError(f"num_heads {H} must split into groups of {NHG} waves of {hpw} heads")
    BT = block_t
    if BT % TL:
        raise ValueError(f"block of {BT} tokens does not split over {TL} token lanes")
    BLOCK = WAVES * WAVE
    NT = _SWA_W // BN
    KS = DV // 32
    DT = DV // 16
    OPS = -(-(BT + _SWA_W - 1) // (8 * WAVES))  # 8-row TDM gathers per wave
    WR = OPS * 8 * WAVES  # window rows in LDS
    NPP = -(-pool_p // 32) * 32  # pool rows in LDS (P*V steps of 32)
    if NPP % (8 * WAVES):
        raise ValueError("pool rows must split into 8-row gathers over the waves")
    if NPP % BN:
        raise ValueError(f"pool of {NPP} rows does not split into kv tiles of {BN}")
    # The pool sits first in LDS: its token-independent fragment addresses then fit the ds
    # immediate offset (beyond 64 KB each would be a hoisted VGPR).
    NTT = NT + NPP // BN  # kv tiles per token: window, then pool
    if (WR + NPP) * ROW > _LDS_MAX:
        raise ValueError(f"window of {WR} rows + pool of {NPP} rows exceed LDS")
    # Per-wave 16-head o staging after window + pool, so o leaves as whole 512-byte row halves
    # (per-head 32-byte pieces from the fragments run the store ~1.6x slower); without room,
    # direct fragment stores.
    STAGE = (WR + NPP + 16 * WAVES) * ROW <= _LDS_MAX
    c_log2 = float(scale) * LOG2E

    @flyc.kernel(known_block_size=[BLOCK, 1, 1])
    def sparse_mla_fwd_gfx1250_swa_kernel(
        Q: fx.Tensor,
        KV: fx.Tensor,
        SINK: fx.Tensor,
        O: fx.Tensor,  # noqa: E741
        LSE: fx.Tensor,
        NKV: fx.Int32,
        N_TOK: fx.Int32,
    ):
        tid = fx.Int32(fx.thread_idx.x)
        wave = uniform(tid // WAVE)
        lane = tid % WAVE
        l16 = lane % 16
        hi = lane // 16
        l8 = lane % 8
        t0 = fx.Int32(fx.block_idx.x) * BT
        tl = wave // NHG  # this wave's token lane
        h_lane = [
            fx.Int32(fx.block_idx.y) * HB + (wave % NHG) * hpw + g * 16 + l16 for g in range_constexpr(G)
        ]
        n_loc = fx.Int32((N_TOK - t0 < fx.Int32(BT)).select(N_TOK - t0, fx.Int32(BT)))

        def tok_of(j):
            return t0 + fx.Int32((j < n_loc).select(j, n_loc - fx.Int32(1)))

        arena = fx.SharedAllocator(static=False)
        arena.allocate((WR + NPP + (16 * WAVES if STAGE else 0)) * ROW)
        lds0 = fx.Int32(fx.Int64(fx.ptrtoint(arena.base_ptr)))

        # LDS row p < NPP = kv row N_TOK + p (pool; rows past it land as zeros); LDS row NPP + i =
        # kv row t0 - 127 + i (window; rows < 0 land as zeros).
        kv_addr = tensor_addr(KV)
        row0 = t0 - fx.Int32(_SWA_W - 1)
        for m in range_constexpr(OPS):
            r = (wave * OPS + m) * 8
            _tdm_gather(lds0 + (NPP + r) * ROW, kv_addr, [row0 + r + i for i in range_constexpr(8)], 8, NKV)
        for m in range_constexpr(NPP // (8 * WAVES)):
            r = (wave * (NPP // (8 * WAVES)) + m) * 8
            _tdm_gather(lds0 + r * ROW, kv_addr, [N_TOK + r + i for i in range_constexpr(8)], 8, NKV)

        def load_q(j):
            n_rec = fx.Int32((j < n_loc).select(fx.Int32(H * DQK * 2), fx.Int32(0)))
            rsrc_q = make_rsrc(tensor_addr(Q) + fx.Int64(tok_of(j)) * (H * DQK * 2), n_rec)
            out = []
            for g in range_constexpr(G):
                q_off = h_lane[g] * (DQK * 2) + hi * 16
                for ks in range_constexpr(KS):
                    out.append(
                        cat(
                            buf_load_b128(rsrc_q, q_off + ks * 64),
                            buf_load_b128(rsrc_q, q_off + ks * 64 + 32),
                        )
                    )
            return out

        q_first = load_q(tl)
        _rocdl.s_wait_tensorcnt(0)
        gpu.barrier()

        k_lane = l16 * ROW + hi * 16
        v_lane = (hi * 8 + l8) * ROW + (l16 // 8) * 16
        v8e = vec_type(8, fx.BFloat16)
        v4i = vec_type(4, fx.Int32)
        stg = lds0 + (fx.Int32(WR + NPP) + wave * 16) * ROW

        def valid_win(lim, s0, nkt):
            # Window slot s0 + kt * 16 + hi * 8 + i is a kv row iff it is >= lim = 127 - tok.
            out = []
            for kt in range_constexpr(nkt):
                b = hi * 8 + s0 + fx.Int32(kt * 16) - lim
                out.append([b >= fx.Int32(-i) for i in range_constexpr(8)])
            return out

        def valid_pool(npv, s0, nkt):
            # Pool slot s0 + kt * 16 + hi * 8 + i is visible iff it is < npv.
            out = []
            for kt in range_constexpr(nkt):
                b = npv - hi * 8 - s0 - fx.Int32(kt * 16)
                out.append([b > fx.Int32(i) for i in range_constexpr(8)])
            return out

        def qk_steps(kbase, nkt, q_frag):
            s = [[Vec.filled(8, 0.0, fx.Float32) for _ in range_constexpr(nkt)] for _ in range_constexpr(G)]
            steps = []
            for kt in range_constexpr(nkt):
                for ks in range_constexpr(KS):

                    def ld(kt=kt, ks=ks):
                        off = kbase + (kt * 16 * ROW + ks * 64)
                        return cat(lds_load(off, v8e), lds_load(off + 32, v8e))

                    def mma(a, kt=kt, ks=ks):
                        for g in range_constexpr(G):
                            s[g][kt] = wmma_bf16(a, q_frag[g][ks], s[g][kt])

                    steps.append((ld, mma))
            return s, steps

        def pv_steps(vbase, npc, p_frag, o):
            steps = []
            for dt in range_constexpr(DT):
                for pc in range_constexpr(npc):

                    def ld(dt=dt, pc=pc):
                        base = vbase + (pc * 32 * ROW + dt * 32)
                        return cat(lds_load_tr16(base), lds_load_tr16(base + 16 * ROW))

                    def mma(a, dt=dt, pc=pc):
                        for g in range_constexpr(G):
                            o[g][dt] = wmma_bf16(a, p_frag[g][pc], o[g][dt])

                    steps.append((ld, mma))
            return steps

        def run_steps(steps):
            _run_steps(steps, batch, prefetch, PIN)

        def softmax(s, valid, m_run, l_run):
            nkt = len(s)
            vals = []
            for kt in range_constexpr(nkt):
                sv = vmul_scalar(s[kt], c_log2)
                for i in range_constexpr(8):
                    vals.append(fx.Float32(valid[kt][i].select(sv[i], fx.Float32(_NEG_INF))))
            tmax = vals[0]
            for v in vals[1:]:
                tmax = fmax(tmax, v)
            tmax = fmax(tmax, swap_halves(tmax))
            m_new = fmax(m_run, tmax)
            m_use = fx.Float32((m_new == fx.Float32(_NEG_INF)).select(fx.Float32(0.0), m_new))
            alpha = exp2(fsub(m_run, m_use))
            ps = [exp2(fsub(v, m_use)) for v in vals]
            rs = ps[0]
            for p in ps[1:]:
                rs = fadd(rs, p)
            l_new = fadd(fmul(l_run, alpha), rs)
            packs = [
                to_bf16(Vec.from_elements(ps[kt * 8 : kt * 8 + 8], fx.Float32)) for kt in range_constexpr(nkt)
            ]
            return (
                m_new,
                l_new,
                alpha,
                [cat(packs[2 * pc], packs[2 * pc + 1]) for pc in range_constexpr(nkt // 2)],
            )

        NQ = G * KS
        KT = BN // 16
        PC = BN // 32
        NS = G * (2 + DT + PC)  # loop state: m[G], l[G], o[G][DT], p[G][PC]

        def pack_state(m, l, o, p):  # noqa: E741
            out = list(m) + list(l)
            for g in range_constexpr(G):
                out += list(o[g])
            for g in range_constexpr(G):
                out += list(p[g])
            return out

        def unpack_state(st):
            m = [st[g] for g in range_constexpr(G)]
            l = [st[G + g] for g in range_constexpr(G)]  # noqa: E741
            o = [[st[2 * G + g * DT + dt] for dt in range_constexpr(DT)] for g in range_constexpr(G)]
            p = [[st[2 * G + G * DT + g * PC + pc] for pc in range_constexpr(PC)] for g in range_constexpr(G)]
            return m, l, o, p

        def step(qk, pv, s_n, valid, m, l, o):  # noqa: E741
            # QK of the next tile interleaved with P*V of the current one, then the softmax.
            seq = []
            for i in range_constexpr(max(len(qk), len(pv))):
                if const_expr(i < len(qk)):
                    seq.append(qk[i])
                if const_expr(i < len(pv)):
                    seq.append(pv[i])
            run_steps(seq)
            p = []
            for g in range_constexpr(G):
                m[g], l[g], ag, pg = softmax(s_n[g], valid, m[g], l[g])
                p.append(pg)
                o[g] = [vmul_scalar(o[g][dt], ag) for dt in range_constexpr(DT)]
            return p

        def token(ji, qf):
            tok = tok_of(ji)
            # Window tiles start at the token's row offset; pool tiles (after them) are shared.
            kbw = lds0 + (fx.Int32(NPP) + tok - t0) * ROW + k_lane
            vbw = lds0 + (fx.Int32(NPP) + tok - t0) * ROW + v_lane
            lim = fx.Int32(_SWA_W - 1) - tok
            npv = (tok + fx.Int32(1)) // fx.Int32(max(pool_cr, 1))
            q_frag = [qf[g * KS : (g + 1) * KS] for g in range_constexpr(G)]

            # One loop over window + pool tiles (same shape): base and mask by a uniform select.
            def tile_base(x, win, pool):
                if const_expr(NPP == 0):
                    return win + x * (BN * ROW)
                return fx.Int32((x < fx.Int32(NT)).select(win + x * (BN * ROW), pool + (x - NT) * (BN * ROW)))

            def tile_valid(x):
                w = valid_win(lim, x * BN, KT)
                if const_expr(NPP == 0):
                    return w
                pz = valid_pool(npv, (x - NT) * BN, KT)
                iw = x < fx.Int32(NT)
                return [
                    [iw.select(w[kt][i], pz[kt][i]) for i in range_constexpr(8)] for kt in range_constexpr(KT)
                ]

            s0, steps0 = qk_steps(kbw, KT, q_frag)
            run_steps(steps0)
            v0 = valid_win(lim, fx.Int32(0), KT)
            m, l, p = [], [], []  # noqa: E741
            for g in range_constexpr(G):
                mg, lg, _, pg = softmax(s0[g], v0, fx.Float32(_NEG_INF), fx.Float32(0.0))
                m.append(mg)
                l.append(lg)
                p.append(pg)
            o = [[Vec.filled(8, 0.0, fx.Float32) for _ in range_constexpr(DT)] for _ in range_constexpr(G)]
            if const_expr(NPP == 0):
                # Window only: unrolled (one scf.for iteration per tile costs pro cr0 ~3%).
                for t in range_constexpr(NTT - 1):
                    s_n, qk = qk_steps(kbw + (t + 1) * BN * ROW, KT, q_frag)
                    pv = pv_steps(vbw + t * BN * ROW, PC, p, o)
                    p = step(qk, pv, s_n, valid_win(lim, fx.Int32((t + 1) * BN), KT), m, l, o)
            else:
                # With the pool: a loop (unrolled pool tiles spill).
                st = pack_state(m, l, o, p)
                for t, sm in range(fx.Index(0), fx.Index(NTT - 1), fx.Index(1), init=st):
                    ti = fx.Int32(t)
                    m, l, o, p = unpack_state(sm)  # noqa: E741
                    s_n, qk = qk_steps(tile_base(ti + 1, kbw, lds0 + k_lane), KT, q_frag)
                    pv = pv_steps(tile_base(ti, vbw, lds0 + v_lane), PC, p, o)
                    p = step(qk, pv, s_n, tile_valid(ti + 1), m, l, o)
                    res = yield pack_state(m, l, o, p)
                m, l, o, p = unpack_state([res[i] for i in range_constexpr(NS)])  # noqa: E741

            # This token's q is dead: the lane's next token loads under the last P*V.
            nxt = load_q(ji + TL)
            run_steps(pv_steps(tile_base(fx.Int32(NTT - 1), vbw, lds0 + v_lane), PC, p, o))

            rsrc_o = make_rsrc(tensor_addr(O) + fx.Int64(tok) * (H * DV * 2), H * DV * 2)
            rsrc_lse = make_rsrc(tensor_addr(LSE) + fx.Int64(tok) * (H * 4), H * 4)
            for g in range_constexpr(G):
                l_tot = fadd(l[g], swap_halves(l[g]))
                if const_expr(has_sink):
                    rsrc_sink = make_rsrc(tensor_addr(SINK), H * 4)
                    sink_log2 = fmul(fx.Float32(buf_load(rsrc_sink, h_lane[g] * 4, T.f32)), LOG2E)
                    mf = fmax(m[g], sink_log2)
                    af = exp2(fsub(m[g], mf))
                    l_out = fadd(fmul(l_tot, af), exp2(fsub(sink_log2, mf)))
                    o_scale = fmul(af, rcp(l_out))
                else:
                    mf = m[g]
                    l_out = l_tot
                    o_scale = fx.Float32((l_tot > fx.Float32(0.0)).select(rcp(l_tot), fx.Float32(0.0)))
                if const_expr(STAGE):
                    # Lane (head l16, half hi) holds d = dt * 16 + hi * 8 + [0, 8) of its head; the
                    # read-back gives each store half a head row. LDS ops of a wave run in order,
                    # so the next group's writes cannot pass this group's reads.
                    for dt in range_constexpr(DT):
                        o16 = to_bf16(vmul_scalar(o[g][dt], o_scale))
                        lds_store(o16, stg + l16 * ROW + hi * 16 + dt * 32)
                    o_off = (h_lane[g] - l16) * (DV * 2) + lane * 16
                    for r in range_constexpr(DT):
                        row = lds_load(stg + (r // 2) * ROW + (r % 2) * 512 + lane * 16, v4i)
                        buf_store(row, rsrc_o, o_off + (r // 2) * (DV * 2) + (r % 2) * 512)
                else:
                    o_off = h_lane[g] * (DV * 2) + hi * 16
                    for dt in range_constexpr(DT):
                        o16 = to_bf16(vmul_scalar(o[g][dt], o_scale))
                        buf_store(o16.bitcast(fx.Int32), rsrc_o, o_off + dt * 32)
                lse = fmul(fadd(mf, log2(l_out)), LN2)
                buf_store(lse, rsrc_lse, fx.Int32((hi == 0).select(h_lane[g] * 4, fx.Int32(0x7FFFFFF0))))
            return nxt

        for j, tsm in range(fx.Index(tl), fx.Index(n_loc), fx.Index(TL), init=q_first):
            _ = yield token(fx.Int32(j), [tsm[i] for i in range_constexpr(NQ)])

    @flyc.jit
    def launch_sparse_mla_fwd_gfx1250_swa(
        Q: fx.Tensor,
        KV: fx.Tensor,
        SINK: fx.Tensor,
        O: fx.Tensor,  # noqa: E741
        LSE: fx.Tensor,
        NKV: fx.Int32,
        n_tok: fx.Int32,
        n_blk: fx.Int32,
        stream: fx.Stream = fx.Stream(None),  # noqa: B008
    ):
        sparse_mla_fwd_gfx1250_swa_kernel(
            Q, KV, SINK, O, LSE, NKV, n_tok, value_attrs={"rocdl.waves_per_eu": 1}
        ).launch(grid=(fx.Index(n_blk), H // HB, 1), block=(BLOCK, 1, 1), stream=stream)

    compiled = {}

    def run(q, kv, sink, o, lse, stream=None):
        n_tok = int(q.shape[0])
        args = (q, kv, o if sink is None else sink, o, lse, int(kv.shape[0]), n_tok, -(-n_tok // BT))
        fn = compiled.get("fn")
        if fn is None:
            fn = flyc.compile[COMPILE_HINTS](launch_sparse_mla_fwd_gfx1250_swa, *args, fx.Stream(stream))
            compiled["fn"] = fn
        return fn(*args, stream)

    return run


def _swa_block_t(n_tok, num_heads, device):
    """Largest token block (16 / 8 / 4 tokens per token lane) that still gives every CU a
    workgroup; ``None`` when even 4 does not (a lane walks its tokens serially, so the GPU would
    idle) or the heads do not split into the 4-wave layout."""
    nhg = min(num_heads // 32, 4)
    if num_heads % 32 or num_heads % (32 * nhg) or 4 % nhg:
        return None
    for per_lane in (16, 8, 4):
        bt = per_lane * (4 // nhg)
        if -(-n_tok // bt) * (num_heads // (32 * nhg)) >= _num_cus(device):
            return bt
    return None


def sparse_mla_fwd_gfx1250(
    q, kv, topk_indices, attn_sink=None, kv_lora_rank=512, scale=None, stream=None, use_swa=True
):
    """o ``[T, H, 512]`` (q.dtype) and lse ``[T, H]`` fp32; see the module docstring."""
    assert kv_lora_rank == DV and q.shape[-1] == DQK, "gfx1250 sparse-MLA is the 512 + 64 latent layout"
    assert q.dtype == torch.bfloat16 and q.is_contiguous() and topk_indices.is_contiguous()
    T_, H, _ = q.shape
    topk = topk_indices.shape[1]
    if scale is None:
        scale = 1.0 / DQK**0.5
    kv = kv.reshape(kv.shape[0], DQK)
    assert kv.is_contiguous() and topk_indices.dtype == torch.int32
    if attn_sink is not None:
        assert attn_sink.dtype == torch.float32 and attn_sink.shape == (H,)
        attn_sink = attn_sink.contiguous()
    o = torch.empty(T_, H, DV, dtype=q.dtype, device=q.device)
    lse = torch.empty(T_, H, dtype=torch.float32, device=q.device)
    if T_ == 0:
        return o, lse
    # Closed-form bands (same tests as the backward): cr0 = SWA window only; cr128 = window + causal
    # pool of P = NKV - T rows (kv row T + p visible iff t >= cr * (p + 1) - 1, cr = T / P).
    # Shapes only, the indices are not read: other indices with these shapes (e.g. packed
    # sequences) get wrong results; see ``sparse_mla_func``.
    pool_p = kv.shape[0] - T_
    if pool_p > 0 and not (T_ % pool_p == 0 and T_ // pool_p >= 64 and topk >= _SWA_W + pool_p):
        pool_p = -1
    if pool_p == 0 and topk != _SWA_W:
        pool_p = -1
    bt = _swa_block_t(T_, H, q.device)
    # With 128 heads the pool forces 32-row tiles, which lose to the generic 64-row kernel
    # once the pool is long (T = 16K: +4%).
    pool_max = _SWA_POOL_MAX if H < 128 else _SWA_POOL_MAX // 2
    if use_swa and 0 <= pool_p <= pool_max and bt is not None:
        build_sparse_mla_fwd_gfx1250_swa(
            H,
            float(scale),
            attn_sink is not None,
            block_t=bt,
            pool_p=pool_p,
            pool_cr=T_ // pool_p if pool_p > 0 else 0,
        )(q, kv, attn_sink, o, lse, stream=stream)
        return o, lse
    hpw, waves, _ = _default_config(H, topk)
    # 16-bit indices halve the TDM op count; worth it where each wave issues several ops per tile.
    idx16 = kv.shape[0] <= 65536 and waves < 4
    run = build_sparse_mla_fwd_gfx1250(H, topk, float(scale), attn_sink is not None, idx16=idx16)
    run(q, kv, topk_indices, attn_sink, o, lse, stream=stream)
    return o, lse
