"""FlyDSL gfx1250 FP8 block-scaled (UE8M0) M-grouped NT GEMM, persistent (moegemm track).

Same math, operands and per-tile k order as grouped_gemm_fp8blk_nt_flydsl_gfx1250 (flygg), so the output is
bit-identical to it:  out[rows_g] = (a * 2^sa)[rows_g] @ (b[g] * 2^sb[g])^T, a [M_total, K] fp8, b [G, N, K] fp8,
sa [M_total, K/32] / sb [G, N, K/32] uint8 E8M0 (block-128 exponents repeated 4x), out bf16.

Why (moegemm profile, in-step trace c3b): flygg time per call fits rounds x (5.5 us + k_tiles x 0.57 us), i.e. a
~5.5 us per-tile fixed cost (scale planes + 2-stage prologue latency, C staging barriers + TDM store + wait 0,
workgroup launch; plus ~45 dead m-tiles x N/256 workgroups from the no-host-sync grid bound). Here:
  * grid = #CUs workgroups; workgroup bid walks output tiles t = bid, bid + nwg, ... < live tiles (counted on the
    device from the group offsets, so no dead workgroups);
  * all k-tiles of all its tiles form ONE TDM stream through an nb-stage LDS ring: the next tile's first k-tiles
    are loaded while the current tile finishes (no per-tile prologue bubble);
  * scale planes are streamed in chunks of CH k-tiles (double-buffered sets) instead of whole-K planes, so the
    stage ring + planes fit 320 KB;
  * the epilogue stores C straight from the accumulators (global_store_b128 per 16x16 fragment row-half) and
    does not wait for the stores; the per-k-step barrier is fenced on LDS only (bar=2), so it does not drain them.
Precondition: every group offset is a multiple of tile_m (the MegaMoE pool pads each expert to 256 rows), so every
tile is full; offs[G] <= M_total.
"""
from __future__ import annotations

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl.expr import const_expr, range_constexpr, rocdl
from flydsl.expr.rocdl import tdm_ops
from flydsl.expr.typing import Constexpr
from flydsl.expr.typing import T as _T

from .grouped_gemm_fp8blk_nt_flydsl_gfx1250 import _A_ID, WAVE, _make_lds_load

try:  # LDS-only barrier (gemmx/path bar=2): no s_wait_storecnt / loadcnt drain at the k-step barrier
    from flydsl._mlir import ir as _ir
    from flydsl._mlir.dialects import gpu as _gpu
except Exception:  # pragma: no cover
    _ir = _gpu = None


def _bar(mode):
    if mode == 2 and _gpu is not None:
        _gpu.barrier(address_spaces=_ir.Attribute.parse("[#gpu.address_space<workgroup>]"))
    else:
        fx.barrier()


@flyc.jit
def launch_ggp_fp8blk_nt_gfx1250(
    arg_c: fx.Pointer,
    arg_a: fx.Pointer,
    arg_b: fx.Pointer,
    arg_offs: fx.Pointer,
    arg_sa: fx.Pointer,
    arg_sb: fx.Pointer,
    i32_nwg: fx.Int32,
    stream: fx.Stream,
    i32_n: fx.Int32,
    i32_k: fx.Int32,
    n_groups: Constexpr[int],
    tile_m: Constexpr[int],
    tile_n: Constexpr[int],
    tile_k: Constexpr[int],
    m_warp: Constexpr[int],
    n_warp: Constexpr[int],
    num_buffers: Constexpr[int],
    a_id: Constexpr[int],
    b_id: Constexpr[int],
    chunk: Constexpr[int] = 16,
    bar: Constexpr[int] = 2,
):
    a_cls, b_cls = _A_ID[a_id], _A_ID[b_id]
    WMMA_M = WMMA_N = 16
    WMMA_K = 128
    NB = num_buffers
    CH = chunk
    num_waves = m_warp * n_warp
    block = num_waves * WAVE
    assert tile_k == WMMA_K and NB >= 2 and CH >= NB
    assert tile_m % (m_warp * WMMA_M) == 0 and tile_n % (n_warp * WMMA_N) == 0
    assert tile_m % num_waves == 0 and tile_n % num_waves == 0
    out_cls = fx.BFloat16
    LDS_PAD = 16
    A_ROW = tile_k + LDS_PAD
    B_ROW = tile_k + LDS_PAD
    STAGE_A = tile_m * A_ROW
    STAGE_B = tile_n * B_ROW
    PITCH = STAGE_A + STAGE_B
    MX_BLOCK = 32
    CW = CH * (tile_k // MX_BLOCK)  # scale bytes per row per chunk
    CSP = CW + 4  # padded LDS row
    CPL_A = tile_m * CSP
    CSET = (tile_m + tile_n) * CSP
    SC_OFF = NB * PITCH
    ARENA = SC_OFF + 2 * CSET
    assert ARENA <= 320 * 1024, f"LDS arena {ARENA} B exceeds 320 KiB"
    G = n_groups
    kernel_name = (f"moe_ggp_fp8blk_nt_gfx1250_t{tile_m}x{tile_n}x{tile_k}_w{m_warp}x{n_warp}_nb{NB}_ch{CH}"
                   f"_bar{bar}_a{a_id}{b_id}_g{G}")

    @flyc.kernel(name=kernel_name, known_block_size=[block, 1, 1])
    def kernel_ggp(
        arg_c: fx.Pointer,
        arg_a: fx.Pointer,
        arg_b: fx.Pointer,
        arg_offs: fx.Pointer,
        arg_sa: fx.Pointer,
        arg_sb: fx.Pointer,
        i32_nwg: fx.Int32,
        i32_n: fx.Int32,
        i32_k: fx.Int32,
    ):
        tid = fx.Int32(fx.thread_idx.x)
        bid_x, _, _ = fx.block_idx
        bid = fx.Int32(bid_x)
        n_tiles_n = i32_n // tile_n
        i32_ptr = fx.PointerType.get(elem_ty=fx.Int32.ir_type, address_space=fx.AddressSpace.Global, alignment=4)
        offs = fx.recast_iter(i32_ptr, arg_offs)
        zero = bid * fx.Int32(0)
        one = zero + fx.Int32(1)
        live_rows = rocdl.readfirstlane(_T.i32, offs[zero + fx.Int32(G)])
        total = (live_rows // tile_m) * n_tiles_n
        my_cnt = (total - bid + i32_nwg - 1) // i32_nwg
        my_cnt = (my_cnt > zero).select(my_cnt, zero)
        KT = i32_k // tile_k
        CPT = KT // CH  # chunks per tile
        lda64 = fx.Int64(i32_k)
        ldc64 = fx.Int64(i32_n)
        sk = i32_k // MX_BLOCK
        sk64 = fx.Int64(sk)

        def tile_params(t):
            """tile t -> (expert, first pool row, first n column); offsets are tile_m aligned"""
            mt = t // n_tiles_n
            pn = t % n_tiles_n
            m_row = mt * tile_m
            g = zero
            for i in range_constexpr(G):
                e = offs[zero + fx.Int32(i + 1)]
                g = g + (e <= m_row).select(one, zero)
            g = (g < fx.Int32(G)).select(g, fx.Int32(G - 1))
            return (rocdl.readfirstlane(_T.i32, g), rocdl.readfirstlane(_T.i32, m_row),
                    rocdl.readfirstlane(_T.i32, pn * tile_n))

        arena = fx.SharedAllocator(static=False)
        arena.allocate(ARENA)
        base_ptr = arena.base_ptr

        def _stage(s):
            return fx.add_offset(base_ptr, s * PITCH)

        def _view(ptr, cls, shape, stride, align=16):
            return fx.Tensor(
                fx.make_view(
                    fx.recast_iter(fx.PointerType.get(cls.ir_type, ptr.address_space, align), ptr),
                    fx.make_layout(shape, stride),
                )
            )

        def _gview(base, off, shape, stride):
            return fx.Tensor(fx.make_view(fx.add_offset(base, off), fx.make_layout(shape, stride)))

        def _cast(cls, p, align=16):
            return fx.recast_iter(fx.PointerType.get(cls.ir_type, p.address_space, align), p)

        gA = _gview(_cast(a_cls, arg_a), fx.Int64(0), (tile_m, tile_k), (i32_k, 1))
        gB = _gview(_cast(b_cls, arg_b), fx.Int64(0), (tile_n, tile_k), (i32_k, 1))
        atomA = fx.rocdl.make_tdm_atom(gA, [None, None], strides=[lda64, None], num_warps=num_waves,
                                       pad_interval=tile_k, pad_amount=LDS_PAD)
        atomB = fx.rocdl.make_tdm_atom(gB, [None, None], strides=[lda64, None], num_warps=num_waves,
                                       pad_interval=tile_k, pad_amount=LDS_PAD)
        gSA_base = _cast(fx.Uint8, arg_sa, 1)
        gSB_base = _cast(fx.Uint8, arg_sb, 1)

        def sA_of(s):
            return _view(_stage(s), a_cls, (tile_m, tile_k), (A_ROW, 1))

        def sB_of(s):
            return _view(fx.add_offset(_stage(s), STAGE_A), b_cls, (tile_n, tile_k), (B_ROW, 1))

        def issue(s, offa, offb):
            fx.copy(atomA, gA, sA_of(s), imm_offset=offa)
            fx.copy(atomB, gB, sB_of(s), imm_offset=offb)

        def issue_chunk(pset, g, m_row, blk_n, c):
            """scale bytes of k-tiles [c*CH, (c+1)*CH) of the tile -> chunk set pset (2 TDM ops)"""
            kcol = fx.Int64(c * CW)
            gsa = _gview(gSA_base, fx.Int64(m_row) * sk64 + kcol, (tile_m, CW), (sk, 1))
            gsb = _gview(gSB_base, (fx.Int64(g) * fx.Int64(i32_n) + fx.Int64(blk_n)) * sk64 + kcol, (tile_n, CW),
                         (sk, 1))
            asa = fx.rocdl.make_tdm_atom(gsa, [None, None], strides=[sk64, None], num_warps=num_waves,
                                         pad_interval=CW, pad_amount=4)
            asb = fx.rocdl.make_tdm_atom(gsb, [None, None], strides=[sk64, None], num_warps=num_waves,
                                         pad_interval=CW, pad_amount=4)
            poff = SC_OFF + pset * CSET
            fx.copy(asa, gsa, _view(fx.add_offset(base_ptr, poff), fx.Uint8, (tile_m, CW), (CSP, 1), 1))
            fx.copy(asb, gsb, _view(fx.add_offset(base_ptr, poff + CPL_A), fx.Uint8, (tile_n, CW), (CSP, 1), 1))

        mma_atom = fx.make_mma_atom(fx.rocdl.WMMAScale(WMMA_M, WMMA_N, WMMA_K, b_cls, a_cls, fx.Float32,
                                                       block_size=MX_BLOCK))
        warp_m = tile_m // m_warp
        warp_n = tile_n // n_warp
        m_reps = warp_m // WMMA_M
        n_reps = warp_n // WMMA_N
        lds_b128 = _make_lds_load(128)
        lds_b32 = _make_lds_load(32)
        wave = rocdl.readfirstlane(_T.i32, tid // WAVE)
        lane = tid % WAVE
        lane16 = lane % 16
        kgrp = lane // 16
        wmb = (wave // n_warp) * warp_m
        wnb = (wave % n_warp) * warp_n
        n_acc = m_reps * n_reps
        c_frags = [fx.make_rmem_tensor(WMMA_N // 2, fx.Float32) for _ in range_constexpr(n_acc)]
        lds_base = fx.Int32(fx.ptrtoint(base_ptr))

        def _operand(base, plane_off, row, row_bytes):
            b0 = plane_off + row * row_bytes + kgrp * 16
            v = [fx.Vector(lds_b128(base, b0 + 32 * j)) for j in range_constexpr(4)]
            v01 = v[0].shuffle(v[1], list(range(8)))
            v23 = v[2].shuffle(v[3], list(range(8)))
            reg = fx.make_rmem_tensor(16, fx.Int32)
            reg.store(v01.shuffle(v23, list(range(16))))
            return reg

        def compute(s, pset, j):
            base = fx.Int32(fx.ptrtoint(_stage(s)))
            poff = SC_OFF + pset * CSET
            kc = j * 4
            bf = [_operand(base, STAGE_A, wnb + ni * WMMA_N + lane16, B_ROW) for ni in range_constexpr(n_reps)]
            sbv = [lds_b32(lds_base, poff + CPL_A + (wnb + ni * WMMA_N + lane16) * CSP + kc)[0]
                   for ni in range_constexpr(n_reps)]
            for mi in range_constexpr(m_reps):
                af = _operand(base, 0, wmb + mi * WMMA_M + lane16, A_ROW)
                sav = lds_b32(lds_base, poff + (wmb + mi * WMMA_M + lane16) * CSP + kc)[0]
                for ni in range_constexpr(n_reps):
                    idx = mi * n_reps + ni
                    fx.gemm(mma_atom, c_frags[idx], bf[ni], af, c_frags[idx], scale_a=sbv[ni], scale_b=sav)

        def offs_of(g, m_row, blk_n, kt):
            k = fx.Int64(kt) * tile_k
            return fx.Int64(m_row) * lda64 + k, (fx.Int64(g) * fx.Int64(i32_n) + fx.Int64(blk_n)) * lda64 + k

        gC_base = _cast(out_cls, arg_c)

        # prologue: tile bid's chunk 0 scales, then k-tiles 0 .. NB-2 (KT >= CH >= NB, so all in tile bid)
        t0 = (bid < total).select(bid, zero)
        g0, mr0, bn0 = tile_params(t0)
        issue_chunk(0, g0, mr0, bn0, zero)
        for j in range_constexpr(NB - 1):
            oa, ob = offs_of(g0, mr0, bn0, zero + fx.Int32(j))
            issue(j, oa, ob)

        for it in range(my_cnt):
            t = bid + it * i32_nwg
            g, m_row, blk_n = tile_params(t)
            t_nx = (t + i32_nwg < total).select(t + i32_nwg, t)
            ng, nm_row, nblk_n = tile_params(t_nx)
            for cf in c_frags:
                cf.fill(0.0)
            for c in range(CPT):
                cs = it * CPT + c  # global chunk index of this workgroup's stream
                pset = cs % 2
                for j in range_constexpr(CH):
                    # ops newer than stage s: everything issued at steps s-NB+2 .. s-1 (2 per stage, +2 for
                    # the chunk planes issued at the j == 0 step)
                    newer = 2 * (NB - 2) + (2 if (j >= 1 and j <= NB - 2) else 0)
                    tdm_ops.tensor_wait(newer)
                    _bar(bar)
                    s = cs * CH + j  # global step
                    if const_expr(j == 0):
                        # next chunk's planes (this tile's chunk c+1, or the next tile's chunk 0); set 1-pset was
                        # last read at step s-1, every wave is past the barrier
                        last_c = c + 1 >= CPT
                        cg = last_c.select(ng, g)
                        cm = last_c.select(nm_row, m_row)
                        cn = last_c.select(nblk_n, blk_n)
                        cc = last_c.select(zero, c + 1)
                        issue_chunk(1 - pset, cg, cm, cn, cc)
                    f = c * CH + (j + NB - 1)  # local k-tile to prefetch (may be in the next tile)
                    nxt = f >= KT
                    ka, kb_ = offs_of(g, m_row, blk_n, f)
                    na, nb_ = offs_of(ng, nm_row, nblk_n, f - KT)
                    issue((s + NB - 1) % NB, nxt.select(na, ka), nxt.select(nb_, kb_))
                    compute(s % NB, pset, j)
            # epilogue: accumulators -> bf16 -> global, straight from VGPRs (stores not waited on)
            for mi in range_constexpr(m_reps):
                row = m_row + wmb + mi * WMMA_M + lane16
                for ni in range_constexpr(n_reps):
                    col = blk_n + wnb + ni * WMMA_N + kgrp * (WMMA_N // 2)
                    h = c_frags[mi * n_reps + ni].load().to(out_cls)
                    fx.ptr_store(h.bitcast(fx.Int8), _cast(fx.Int8, gC_base) + (fx.Int64(row) * ldc64 + fx.Int64(col)) * 2)
        tdm_ops.tensor_wait(0)

    kernel_ggp(arg_c, arg_a, arg_b, arg_offs, arg_sa, arg_sb, i32_nwg, i32_n, i32_k).launch(
        grid=(i32_nwg, 1, 1), block=(block, 1, 1), stream=stream
    )


launch_ggp_fp8blk_nt_gfx1250.compile_hints["llvm_options"] = {"amdgpu-expert-scheduling-mode": True}

_COMPILED: dict = {}


def _ptr(t):
    return flyc.from_c_void_p(fx.Int8, t.data_ptr(), assumed_align=16)


def flyggp_fp8blk_nt(a, b, sa, sb, offs_i32, out=None, tile=(256, 256, 128), m_warp=2, n_warp=2, num_buffers=3,
                     chunk=16, bar=2, nwg=None):
    """Persistent drop-in for flygg_fp8blk_nt (bit-identical output). Requires every offs_i32 entry to be a
    multiple of tile[0] (not checked on the host: that would sync) and K // 128 to be a multiple of ``chunk``."""
    assert sa.dtype == torch.uint8 and sb.dtype == torch.uint8 and sa.is_contiguous() and sb.is_contiguous()
    M, K = a.shape
    G, N, Kb = b.shape
    assert K == Kb and a.is_contiguous() and b.is_contiguous()
    tm, tn, tk = tile
    assert tk == 128 and N % tn == 0 and (K // tk) % chunk == 0 and chunk >= num_buffers
    if out is None:
        out = torch.empty((M, N), device=a.device, dtype=torch.bfloat16)
    a_id = 0 if a.dtype == torch.float8_e4m3fn else 1
    b_id = 0 if b.dtype == torch.float8_e4m3fn else 1
    if nwg is None:
        nwg = torch.cuda.get_device_properties(a.device).multi_processor_count
    nwg = max(1, min(nwg, (M // tm) * (N // tn)))
    args = (_ptr(out), _ptr(a), _ptr(b), _ptr(offs_i32), _ptr(sa), _ptr(sb), nwg,
            fx.Stream(torch.cuda.current_stream(device=a.device)), N, K, G, tm, tn, tk, m_warp, n_warp,
            num_buffers, a_id, b_id, chunk, bar)
    key = (G, tm, tn, tk, m_warp, n_warp, num_buffers, a_id, b_id, chunk, bar)
    c = _COMPILED.get(key)
    if c is None:
        _COMPILED[key] = flyc.compile(launch_ggp_fp8blk_nt_gfx1250, *args)
    else:
        c(*args)
    return out
