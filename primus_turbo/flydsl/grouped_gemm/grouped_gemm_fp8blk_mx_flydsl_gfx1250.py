"""mxgemm: FlyDSL gfx1250 FP8 block-scaled (UE8M0) M-grouped NT GEMM with a deep LDS ring and a cross-k-step
fragment pipeline ("mxgg"). Same math, operands, WMMA (v_wmma_scale_f32_16x16x128_f8f6f4, MX block 32 with the
1x128 / 128x128 exponents repeated 4x), per-element k order (ascending 128-k steps into one fp32 accumulator) and bf16
rounding as flygg (grouped_gemm_fp8blk_nt_flydsl_gfx1250.kernel_gg): C is bit-identical.

Why (mxgemm ledger E0-E2): V_WMMA_SCALE issues every 8.2 cycles from registers, but flygg k256 runs ~14-15 cycles per
WMMA: its 2-stage ring of 256-deep k-tiles keeps one 128 KB k-tile in flight and the next k-tile is never resident
while the current one is computed, so every k-tile starts with an exposed TDM wait plus a burst of 40 fragment loads.
Here:
  * 4-stage ring of 128-deep k-tiles (A 256 x 144 B, B 256 x 144 B, A-scale rows 256 x 4 B per stage). At the top of
    k-step kt the kernel waits for k-tile kt+1 (not kt), so k-tiles kt+2 and kt+3 (2 x 64 KB) stay in flight and
    stage kt+1 is resident during step kt;
  * the fragments of step kt+1 (all 8 B fragments, the first A fragment, the scale words) are read from LDS during
    the row groups of step kt (B double-buffered, A one row group ahead), each group pinned with sched_barrier: no
    k-step starts with a fragment-load bubble;
  * weight scales: one row per 16-row WMMA fragment (128x128-block scales, as flygg bblk), loaded once per tile.
Epilogue: as kernel_gg (C staged through LDS, one TDM store), via `epilogue(...)` so a functor (activation, 1x128
quant + UE8M0 scale write, sumsq) can be added later without touching the main loop.
Requires: aligned group offsets (multiples of tile_m: the caller's pool alignment), K % 128 == 0, K >= 384,
N % tile_n == 0, weight scales constant over 16-row groups of N.
"""
from __future__ import annotations

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl.expr import const_expr, range_constexpr, rocdl
from flydsl.expr.rocdl import tdm_ops
from flydsl.expr.typing import Constexpr
from flydsl.expr.typing import T as _T

from .grouped_gemm_fp8blk_nt_flydsl_gfx1250 import _make_lds_load

WAVE = 32
_A_ID = {0: fx.Float8E4M3FN, 1: fx.Float8E5M2}


@flyc.jit
def launch_mxgg_fp8blk_nt_gfx1250(
    arg_c: fx.Pointer,
    arg_a: fx.Pointer,
    arg_b: fx.Pointer,
    arg_offs: fx.Pointer,
    arg_sa: fx.Pointer,
    arg_sb: fx.Pointer,
    i32_mtiles: fx.Int32,
    stream: fx.Stream,
    i32_n: fx.Int32,
    i32_k: fx.Int32,
    n_groups: Constexpr[int],
    a_id: Constexpr[int],
    b_id: Constexpr[int],
    k_max: Constexpr[int] = 4096,
    num_buffers: Constexpr[int] = 4,
    abl: Constexpr[int] = 0,
):
    """abl (timing ablations only, WRONG results): bit0 = no per-stage A-scale TDM (constant scale word), bit2 = no
    TDM refill in the k loop (the prologue's k-tiles are re-read)."""
    a_cls, b_cls = _A_ID[a_id], _A_ID[b_id]
    tile_m = tile_n = 256
    tile_k = 128
    m_warp = n_warp = 2
    WMMA_M = WMMA_N = 16
    WMMA_K = 128
    NB = num_buffers
    assert NB >= 3
    LA = NB - 1  # k-tiles issued ahead of the one being computed
    num_waves = m_warp * n_warp
    block = num_waves * WAVE
    out_cls = fx.BFloat16
    LDS_PAD = 16
    A_ROW = tile_k + LDS_PAD
    B_ROW = tile_k + LDS_PAD
    STAGE_A = tile_m * A_ROW
    STAGE_B = tile_n * B_ROW
    ASC_OFF = STAGE_A + STAGE_B  # per-stage A scale rows: tile_m x 4 B (one 128-k step, MX32 bytes)
    PITCH = ASC_OFF + tile_m * 4
    MX_BLOCK = 32
    SK = k_max // MX_BLOCK
    SP = SK + 4
    B_PROWS = tile_n // WMMA_N
    PLANE_SB = (B_PROWS * SP + 15) // 16 * 16
    SC_OFF = NB * PITCH
    C_ROW = tile_n + 8
    ARENA = max(SC_OFF + PLANE_SB, tile_m * C_ROW * 2)
    assert ARENA <= 320 * 1024, f"LDS arena {ARENA} B exceeds 320 KiB"
    n_ops = 2 if abl & 1 else 3  # TDM ops per k-tile and wave: A, B, A scale rows
    G = n_groups
    kernel_name = f"moe_mxgg_fp8blk_nt_gfx1250_t256x256x128_nb{NB}_a{a_id}{b_id}_g{G}_k{k_max}" + (f"_abl{abl}" if abl else "")

    @flyc.kernel(name=kernel_name, known_block_size=[block, 1, 1])
    def kernel_mxgg(
        arg_c: fx.Pointer,
        arg_a: fx.Pointer,
        arg_b: fx.Pointer,
        arg_offs: fx.Pointer,
        arg_sa: fx.Pointer,
        arg_sb: fx.Pointer,
        i32_mtiles: fx.Int32,
        i32_n: fx.Int32,
        i32_k: fx.Int32,
    ):
        tid = fx.Int32(fx.thread_idx.x)
        bid_x, _, _ = fx.block_idx
        n_tiles_n = i32_n // tile_n
        bid = fx.Int32(bid_x)
        t = rocdl.readfirstlane(_T.i32, bid // n_tiles_n)
        pn = bid % n_tiles_n
        blk_n = pn * tile_n

        i32_ptr = fx.PointerType.get(elem_ty=fx.Int32.ir_type, address_space=fx.AddressSpace.Global, alignment=4)
        mt = fx.recast_iter(i32_ptr, arg_offs)
        # aligned offsets: m-tile t starts at pool row t * tile_m; its expert = #(offs[i+1] <= row), clamped; rows
        # past the last expert are dead (rows = 0: TDM clamps to nothing, no C store)
        zero = t * fx.Int32(0)
        one = zero + fx.Int32(1)
        m_row = t * fx.Int32(tile_m)
        g_idx = zero
        for i in range_constexpr(G):
            g_idx = g_idx + (mt[zero + fx.Int32(i + 1)] <= m_row).select(one, zero)
        live_end = mt[zero + fx.Int32(G)]
        rows = (live_end - m_row)
        rows = (rows < fx.Int32(tile_m)).select(rows, fx.Int32(tile_m))
        rows = (rows > zero).select(rows, zero)
        g_idx = (g_idx < fx.Int32(G)).select(g_idx, fx.Int32(G - 1))
        g_idx = rocdl.readfirstlane(_T.i32, g_idx)
        rows = rocdl.readfirstlane(_T.i32, rows)
        m_row = rocdl.readfirstlane(_T.i32, (rows > zero).select(m_row, zero))
        K_TILES = (rows > zero).select(i32_k // tile_k, fx.Int32(LA))

        lda64 = fx.Int64(i32_k)
        ldc64 = fx.Int64(i32_n)

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

        gA = _gview(_cast(a_cls, arg_a), fx.Int64(m_row) * lda64, (tile_m, tile_k), (i32_k, 1))
        gB = _gview(_cast(b_cls, arg_b), (fx.Int64(g_idx) * fx.Int64(i32_n) + fx.Int64(blk_n)) * lda64,
                    (tile_n, tile_k), (i32_k, 1))
        atomA = fx.rocdl.make_tdm_atom(gA, [rows, None], strides=[lda64, None], num_warps=num_waves,
                                       pad_interval=tile_k, pad_amount=LDS_PAD)
        atomB = fx.rocdl.make_tdm_atom(gB, [i32_n - blk_n, None], strides=[lda64, None], num_warps=num_waves,
                                       pad_interval=tile_k, pad_amount=LDS_PAD)
        sk = i32_k // MX_BLOCK
        sk64 = fx.Int64(sk)
        # A scale rows of one k-tile: tile_m x 4 B (row stride sk) -> LDS tile_m x 4 B contiguous
        gSA = _gview(_cast(fx.Uint8, arg_sa, 1), fx.Int64(m_row) * sk64, (tile_m, 4), (sk, 1))
        atomSA = fx.rocdl.make_tdm_atom(gSA, [rows, None], strides=[sk64, None], num_warps=num_waves)
        # B scale plane: one row per 16-row fragment, whole K (loaded once)
        gSB = _gview(_cast(fx.Uint8, arg_sb, 1), (fx.Int64(g_idx) * fx.Int64(i32_n) + fx.Int64(blk_n)) * sk64,
                     (B_PROWS, SK), (sk * WMMA_N, 1))
        atomSB = fx.rocdl.make_tdm_atom(gSB, [B_PROWS, sk], strides=[sk64 * WMMA_N, None], num_warps=num_waves,
                                        pad_interval=SK, pad_amount=4)
        sSB = _view(fx.add_offset(base_ptr, SC_OFF), fx.Uint8, (B_PROWS, SK), (SP, 1), 1)

        def sA_of(s):
            return _view(_stage(s), a_cls, (tile_m, tile_k), (A_ROW, 1))

        def sB_of(s):
            return _view(fx.add_offset(_stage(s), STAGE_A), b_cls, (tile_n, tile_k), (B_ROW, 1))

        def sSA_of(s):
            return _view(fx.add_offset(_stage(s), ASC_OFF), fx.Uint8, (tile_m, 4), (4, 1), 1)

        def issue(s, kt):
            kt64 = fx.Int64(kt)
            fx.copy(atomA, gA, sA_of(s), imm_offset=kt64 * tile_k)
            fx.copy(atomB, gB, sB_of(s), imm_offset=kt64 * tile_k)
            if const_expr(not (abl & 1)):
                fx.copy(atomSA, gSA, sSA_of(s), imm_offset=kt64 * 4)

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
        for cf in c_frags:
            cf.fill(0.0)
        lds_base = fx.Int32(fx.ptrtoint(base_ptr))

        # Per-lane LDS byte offsets (constant over the kernel). Every fragment / scale load of a k-step is
        # (address VGPR of that step) + immediate offset; the address VGPRs are formed once at the start of the step,
        # >= 4 WMMAs before their first use. (Expert scheduling waits for a VALU-written address with
        # depctr_va_vdst, an in-order counter: an address formed right before its load drains every WMMA in flight.)
        lane_a = (wmb + lane16) * A_ROW + kgrp * 16
        lane_b = STAGE_A + (wnb + lane16) * B_ROW + kgrp * 16
        lane_sa = ASC_OFF + (wmb + lane16) * 4
        sb_row0 = SC_OFF + (wnb // WMMA_N) * SP
        zl = lane * (i32_n >> 30)  # a VGPR zero the compiler cannot fold (N < 2^30): keeps the uniform B-scale
        # address a VALU value formed at the step start instead of an SGPR copied next to its load

        def addrs(kt):
            st = fx.Int32(fx.ptrtoint(_stage(kt % NB)))
            return dict(a=st + lane_a, b=st + lane_b, sa=st + lane_sa, sb=zl + (lds_base + sb_row0 + kt * 4))

        def _ld_frag(base, off, dst=None):
            v = [fx.Vector(lds_b128(base, off + 32 * j)) for j in range_constexpr(4)]
            v01 = v[0].shuffle(v[1], list(range(8)))
            v23 = v[2].shuffle(v[3], list(range(8)))
            reg = fx.make_rmem_tensor(16, fx.Int32) if dst is None else dst
            reg.store(v01.shuffle(v23, list(range(16))))
            return reg

        def frag_a(ad, mi, dst=None):
            return _ld_frag(ad["a"], mi * WMMA_M * A_ROW, dst)

        def frag_b(ad, ni, dst=None):
            return _ld_frag(ad["b"], ni * WMMA_N * B_ROW, dst)

        def scale_a(ad, mi):
            if const_expr(abl & 1):
                return lds_b32(ad["sa"], mi * WMMA_M * 4) * 0 + 0x7F7F7F7F
            return lds_b32(ad["sa"], mi * WMMA_M * 4)

        def scale_b(ad, ni):
            return lds_b32(ad["sb"], ni * SP)

        # Two carrier sets of the fragments a k-step starts with (8 B fragments, A fragments of row groups 0 and 1,
        # their A scale words, the 8 B scale words). Step kt computes from set kt%2 and fills set (kt+1)%2 for step
        # kt+1 (rmem tensors = registers with memory semantics across the runtime loop; the steady loop is unrolled
        # by two so the roles never swap inside an iteration: no phi copies).
        def _set():
            return dict(b=[fx.make_rmem_tensor(16, fx.Int32) for _ in range_constexpr(n_reps)],
                        a=[fx.make_rmem_tensor(16, fx.Int32) for _ in range_constexpr(2)],
                        sa=[fx.make_rmem_tensor(1, fx.Int32) for _ in range_constexpr(2)],
                        sb=[fx.make_rmem_tensor(1, fx.Int32) for _ in range_constexpr(n_reps)])

        SETS = [_set(), _set()]

        def preload(kt, dst):
            ad = addrs(kt)
            for ni in range_constexpr(n_reps):
                frag_b(ad, ni, dst["b"][ni])
                dst["sb"][ni].store(scale_b(ad, ni))
            for j in range_constexpr(2):
                frag_a(ad, j, dst["a"][j])
                dst["sa"][j].store(scale_a(ad, j))

        def _wmmas(src, afs, savs, sbv, mi, lo, hi):
            for ni in range_constexpr(lo, hi):
                idx = mi * n_reps + ni
                fx.gemm(mma_atom, c_frags[idx], src["b"][ni], afs[mi], c_frags[idx], scale_a=sbv[ni],
                        scale_b=savs[mi])

        def compute(kt, src, dst):
            """64 WMMAs of 128-k step kt from carrier set src. Row group mi = WMMAs ni 0..3, the A fragment of group
            mi+2 and (dst not None) the B fragment mi of step kt+1 from its resident stage, WMMAs ni 4..7. All loads sit
            mid-group: the registers they may reuse were last read >= 4 WMMAs earlier, so expert scheduling waits for
            those WMMAs only, never for the whole pipe. Groups 6 / 7 fetch step kt+1's A fragments 0 / 1 instead of
            mi+2, group 0 also its B scale words."""
            rocdl.sched_barrier(0)
            ad = addrs(kt)
            if const_expr(dst is not None):
                an = addrs(kt + 1)
            sbv = [src["sb"][ni].load()[0] for ni in range_constexpr(n_reps)]
            afs = [src["a"][0], src["a"][1]] + [None] * (m_reps - 2)
            savs = [src["sa"][0].load()[0], src["sa"][1].load()[0]] + [None] * (m_reps - 2)
            h = n_reps // 2
            for mi in range_constexpr(m_reps):
                rocdl.sched_barrier(0)
                _wmmas(src, afs, savs, sbv, mi, 0, h)
                rocdl.sched_barrier(0)
                if const_expr(mi + 2 < m_reps):
                    afs[mi + 2] = frag_a(ad, mi + 2)
                    savs[mi + 2] = scale_a(ad, mi + 2)[0]
                elif const_expr(dst is not None):
                    j = mi + 2 - m_reps
                    frag_a(an, j, dst["a"][j])
                    dst["sa"][j].store(scale_a(an, j))
                if const_expr(dst is not None):
                    frag_b(an, mi, dst["b"][mi])
                    if const_expr(mi == 0):
                        for ni in range_constexpr(n_reps):
                            dst["sb"][ni].store(scale_b(an, ni))
                rocdl.sched_barrier(0)
                _wmmas(src, afs, savs, sbv, mi, h, n_reps)
                rocdl.sched_barrier(0)

        def bar():
            """raw workgroup barrier, no LDS/VMEM drain: the only LDS loads outstanding here read stage kt (the
            prefetch), never the stage the TDM refills next (its reads were all consumed by WMMAs before)."""
            rocdl.sched_barrier(0)
            rocdl.s_barrier_signal(-1)
            rocdl.s_barrier_wait(-1)
            rocdl.sched_barrier(0)

        def step(kt, par, wait_ops, do_issue, prefetch):
            if const_expr(wait_ops is not None):
                tdm_ops.tensor_wait(wait_ops)
                bar()
            if const_expr(do_issue and not (abl & 4)):
                issue((kt + LA) % NB, kt + LA)
            compute(kt, SETS[par], SETS[1 - par] if prefetch else None)

        def epilogue(gC_base, m_row, blk_n, rows, ldc64):
            """C tile -> bf16 -> LDS staging (row pitch C_ROW, over the stages) -> one TDM store of the live rows.
            Functor hook point: between the accumulator load and the bf16 store (activation, 1x128 quant + UE8M0
            scale write, sumsq). Same stores and rounding as kernel_gg."""
            fx.barrier()
            sC = _view(base_ptr, out_cls, (tile_m, C_ROW), (C_ROW, 1))
            for mi in range_constexpr(m_reps):
                row = wmb + mi * WMMA_M + lane16
                for ni in range_constexpr(n_reps):
                    col = wnb + ni * WMMA_N + kgrp * (WMMA_N // 2)
                    h = c_frags[mi * n_reps + ni].load().to(out_cls)
                    fx.ptr_store(h.bitcast(fx.Int8), base_ptr + (row * C_ROW + col) * 2)
            fx.barrier()
            gtC = _gview(gC_base, fx.Int64(m_row) * ldc64 + fx.Int64(blk_n), (tile_m, C_ROW), (C_ROW, 1))
            atomC = fx.rocdl.make_tdm_atom(gtC, [rows, tile_n], strides=[ldc64, None], num_warps=num_waves)
            fx.copy(atomC, sC, gtC)
            tdm_ops.tensor_wait(0)

        # prologue: B scale plane, k-tiles 0 .. LA-1; wait for k-tile 0 (and the plane), read step 0's fragments
        fx.copy(atomSB, gSB, sSB)
        for j in range_constexpr(LA):
            issue(j, j)
        tdm_ops.tensor_wait((LA - 1) * n_ops)
        fx.barrier()
        preload(0, SETS[0])
        # steady state (steps that issue k-tile kt+LA): step kt waits for k-tile kt+1, so k-tiles kt+2 .. kt+LA-1
        # stay in flight; its barrier also frees stage (kt-1) % NB for k-tile kt+LA. Unrolled by two.
        n_steady = K_TILES - LA
        n_pair = n_steady // 2
        for kp in range(n_pair):
            step(kp * 2, 0, (LA - 2) * n_ops, True, True)
            step(kp * 2 + 1, 1, (LA - 2) * n_ops, True, True)
        if n_steady % 2 == 1:
            step(n_pair * 2, 0, (LA - 2) * n_ops, True, True)
            for ni in range_constexpr(n_reps):  # back to set 0 for the tail (once per tile)
                SETS[0]["b"][ni].store(SETS[1]["b"][ni].load())
                SETS[0]["sb"][ni].store(SETS[1]["sb"][ni].load())
            for j in range_constexpr(2):
                SETS[0]["a"][j].store(SETS[1]["a"][j].load())
                SETS[0]["sa"][j].store(SETS[1]["sa"][j].load())
        # tail: the last LA steps issue nothing; step kt still waits for k-tile kt+1 (LA-2-j newer k-tiles)
        for j in range_constexpr(LA):
            kt = n_steady + j
            if const_expr(j < LA - 1):
                step(kt, j % 2, (LA - 2 - j) * n_ops, False, True)
            else:
                step(kt, j % 2, None, False, False)
        epilogue(_cast(out_cls, arg_c), m_row, blk_n, rows, ldc64)

    kernel_mxgg(arg_c, arg_a, arg_b, arg_offs, arg_sa, arg_sb, i32_mtiles, i32_n, i32_k).launch(
        grid=(i32_mtiles * (i32_n // tile_n), 1, 1), block=(block, 1, 1), stream=stream
    )


launch_mxgg_fp8blk_nt_gfx1250.compile_hints["llvm_options"] = {"amdgpu-expert-scheduling-mode": True}


_COMPILED: dict = {}


def _ptr(t):
    return flyc.from_c_void_p(fx.Int8, t.data_ptr(), assumed_align=16)


def mxgg_fp8blk_nt(a, b, sa, sb, offs_i32, out=None, num_buffers=4, k_max=None, abl=0):
    """Drop-in for flygg_fp8blk_nt(..., aligned=True, bblk=1) at tile 256x256: a [M, K] fp8, b [G, N, K] fp8,
    sa [M, K/32] / sb [G, N, K/32] uint8 E8M0 (block-128 exponents repeated 4x), offs_i32 [G+1] int32 with every entry
    a multiple of 256 (not checked: would sync). Returns out [M, N] bf16 (rows past offs[G] are not written).
    Bit-identical to flygg (same WMMAs and k order)."""
    assert sa.dtype == torch.uint8 and sb.dtype == torch.uint8 and sa.is_contiguous() and sb.is_contiguous()
    M, K = a.shape
    G, N, Kb = b.shape
    assert K == Kb and a.is_contiguous() and b.is_contiguous()
    assert K % 128 == 0 and K // 128 >= num_buffers - 1 and N % 256 == 0 and M % 256 == 0
    assert sa.shape == (M, K // 32) and sb.shape == (G, N, K // 32)
    if k_max is None:  # B scale plane row = k_max / 32 bytes; its TDM pad interval must be a power of two
        k_max = 1 << (K - 1).bit_length()
    assert k_max >= K and (k_max & (k_max - 1)) == 0 and k_max >= 128
    if out is None:
        out = torch.empty((M, N), device=a.device, dtype=torch.bfloat16)
    for t in (a, b, sa, sb, offs_i32, out):  # never hand the kernel a null address
        assert t.data_ptr() != 0
    a_id = 0 if a.dtype == torch.float8_e4m3fn else 1
    b_id = 0 if b.dtype == torch.float8_e4m3fn else 1
    args = (_ptr(out), _ptr(a), _ptr(b), _ptr(offs_i32), _ptr(sa), _ptr(sb), M // 256,
            fx.Stream(torch.cuda.current_stream(device=a.device)), N, K, G, a_id, b_id, k_max, num_buffers, abl)
    key = (G, a_id, b_id, k_max, num_buffers, abl)
    c = _COMPILED.get(key)
    if c is None:
        _COMPILED[key] = flyc.compile(launch_mxgg_fp8blk_nt_gfx1250, *args)
    else:
        c(*args)
    return out


@flyc.jit
def launch_mxgk_fp8blk_nt_gfx1250(
    arg_c: fx.Pointer,
    arg_a: fx.Pointer,
    arg_b: fx.Pointer,
    arg_offs: fx.Pointer,
    arg_sa: fx.Pointer,
    arg_sb: fx.Pointer,
    i32_mtiles: fx.Int32,
    stream: fx.Stream,
    i32_n: fx.Int32,
    i32_k: fx.Int32,
    n_groups: Constexpr[int],
    a_id: Constexpr[int],
    b_id: Constexpr[int],
    k_max: Constexpr[int] = 4096,
    early: Constexpr[int] = 4,
):
    """mxgk (mxgemm E4): flygg k256's TDM traffic exactly (2-stage ring of 256-deep k-tiles, 256 B rows: half the TDM
    rows per FLOP of 128-deep tiles; whole-K scale planes, B plane one row per 16-row fragment) plus a fragment pipeline
    that crosses k-tiles. In the second 128-k half of k-tile kt the A fragments of row groups `early`..7 are read ahead;
    then (s_wait_dscnt 0, wait for k-tile kt+1, raw barrier) stage kt is no longer read by any wave, so k-tile kt+2 is
    issued into it right there, and row groups early..7 read k-tile kt+1's first fragments (8 B, A of group 0, scale
    words) from the other stage. Inside a k-tile, B of the second half is double-buffered (as flygg bpf). Same WMMAs,
    operands, k order and epilogue as kernel_gg: bit-identical. Needs aligned group offsets, K % 256 == 0, K >= 512."""
    a_cls, b_cls = _A_ID[a_id], _A_ID[b_id]
    tile_m = tile_n = 256
    tile_k = 256
    KI = 2  # 128-k steps per k-tile
    m_warp = n_warp = 2
    WMMA_M = WMMA_N = 16
    WMMA_K = 128
    num_waves = m_warp * n_warp
    block = num_waves * WAVE
    out_cls = fx.BFloat16
    LDS_PAD = 16
    A_ROW = tile_k + LDS_PAD
    B_ROW = tile_k + LDS_PAD
    STAGE_A = tile_m * A_ROW
    PITCH = STAGE_A + tile_n * B_ROW
    MX_BLOCK = 32
    SK = k_max // MX_BLOCK
    SP = SK + 4
    PLANE_SA = tile_m * SP
    B_PROWS = tile_n // WMMA_N
    PLANE_SB = (B_PROWS * SP + 15) // 16 * 16
    SC_OFF = 2 * PITCH
    C_ROW = tile_n + 8
    ARENA = max(SC_OFF + PLANE_SA + PLANE_SB, tile_m * C_ROW * 2)
    assert ARENA <= 320 * 1024, f"LDS arena {ARENA} B exceeds 320 KiB"
    G = n_groups
    m_reps = n_reps = 8
    assert 2 <= early <= 6 and n_reps % (m_reps - early) == 0
    BPG = n_reps // (m_reps - early)  # next-tile B fragments prefetched per row group after the early barrier
    kernel_name = f"moe_mxgk_fp8blk_nt_gfx1250_t256x256x256_a{a_id}{b_id}_g{G}_k{k_max}_e{early}"

    @flyc.kernel(name=kernel_name, known_block_size=[block, 1, 1])
    def kernel_mxgk(
        arg_c: fx.Pointer,
        arg_a: fx.Pointer,
        arg_b: fx.Pointer,
        arg_offs: fx.Pointer,
        arg_sa: fx.Pointer,
        arg_sb: fx.Pointer,
        i32_mtiles: fx.Int32,
        i32_n: fx.Int32,
        i32_k: fx.Int32,
    ):
        tid = fx.Int32(fx.thread_idx.x)
        bid_x, _, _ = fx.block_idx
        n_tiles_n = i32_n // tile_n
        bid = fx.Int32(bid_x)
        t = rocdl.readfirstlane(_T.i32, bid // n_tiles_n)
        blk_n = (bid % n_tiles_n) * tile_n
        i32_ptr = fx.PointerType.get(elem_ty=fx.Int32.ir_type, address_space=fx.AddressSpace.Global, alignment=4)
        mt = fx.recast_iter(i32_ptr, arg_offs)
        zero = t * fx.Int32(0)
        one = zero + fx.Int32(1)
        m_row = t * fx.Int32(tile_m)
        g_idx = zero
        for i in range_constexpr(G):
            g_idx = g_idx + (mt[zero + fx.Int32(i + 1)] <= m_row).select(one, zero)
        rows = mt[zero + fx.Int32(G)] - m_row
        rows = (rows < fx.Int32(tile_m)).select(rows, fx.Int32(tile_m))
        rows = (rows > zero).select(rows, zero)
        g_idx = (g_idx < fx.Int32(G)).select(g_idx, fx.Int32(G - 1))
        g_idx = rocdl.readfirstlane(_T.i32, g_idx)
        rows = rocdl.readfirstlane(_T.i32, rows)
        m_row = rocdl.readfirstlane(_T.i32, (rows > zero).select(m_row, zero))
        K_TILES = (rows > zero).select(i32_k // tile_k, fx.Int32(2))
        lda64 = fx.Int64(i32_k)
        ldc64 = fx.Int64(i32_n)

        arena = fx.SharedAllocator(static=False)
        arena.allocate(ARENA)
        base_ptr = arena.base_ptr

        def _stage(s):
            return fx.add_offset(base_ptr, s * PITCH)

        def _view(ptr, cls, shape, stride, align=16):
            return fx.Tensor(fx.make_view(fx.recast_iter(fx.PointerType.get(cls.ir_type, ptr.address_space, align), ptr),
                                          fx.make_layout(shape, stride)))

        def _gview(base, off, shape, stride):
            return fx.Tensor(fx.make_view(fx.add_offset(base, off), fx.make_layout(shape, stride)))

        def _cast(cls, p, align=16):
            return fx.recast_iter(fx.PointerType.get(cls.ir_type, p.address_space, align), p)

        gA = _gview(_cast(a_cls, arg_a), fx.Int64(m_row) * lda64, (tile_m, tile_k), (i32_k, 1))
        gB = _gview(_cast(b_cls, arg_b), (fx.Int64(g_idx) * fx.Int64(i32_n) + fx.Int64(blk_n)) * lda64,
                    (tile_n, tile_k), (i32_k, 1))
        atomA = fx.rocdl.make_tdm_atom(gA, [rows, None], strides=[lda64, None], num_warps=num_waves,
                                       pad_interval=tile_k, pad_amount=LDS_PAD)
        atomB = fx.rocdl.make_tdm_atom(gB, [i32_n - blk_n, None], strides=[lda64, None], num_warps=num_waves,
                                       pad_interval=tile_k, pad_amount=LDS_PAD)
        sk = i32_k // MX_BLOCK
        sk64 = fx.Int64(sk)
        gSA = _gview(_cast(fx.Uint8, arg_sa, 1), fx.Int64(m_row) * sk64, (tile_m, SK), (sk, 1))
        gSB = _gview(_cast(fx.Uint8, arg_sb, 1), (fx.Int64(g_idx) * fx.Int64(i32_n) + fx.Int64(blk_n)) * sk64,
                     (B_PROWS, SK), (sk * WMMA_N, 1))
        atomSA = fx.rocdl.make_tdm_atom(gSA, [rows, sk], strides=[sk64, None], num_warps=num_waves,
                                        pad_interval=SK, pad_amount=4)
        atomSB = fx.rocdl.make_tdm_atom(gSB, [(i32_n - blk_n + (WMMA_N - 1)) // WMMA_N, sk],
                                        strides=[sk64 * WMMA_N, None], num_warps=num_waves, pad_interval=SK,
                                        pad_amount=4)
        sSA = _view(fx.add_offset(base_ptr, SC_OFF), fx.Uint8, (tile_m, SK), (SP, 1), 1)
        sSB = _view(fx.add_offset(base_ptr, SC_OFF + PLANE_SA), fx.Uint8, (B_PROWS, SK), (SP, 1), 1)

        def issue(s, kt):
            kt64 = fx.Int64(kt)
            fx.copy(atomA, gA, _view(_stage(s), a_cls, (tile_m, tile_k), (A_ROW, 1)), imm_offset=kt64 * tile_k)
            fx.copy(atomB, gB, _view(fx.add_offset(_stage(s), STAGE_A), b_cls, (tile_n, tile_k), (B_ROW, 1)),
                    imm_offset=kt64 * tile_k)

        mma_atom = fx.make_mma_atom(fx.rocdl.WMMAScale(WMMA_M, WMMA_N, WMMA_K, b_cls, a_cls, fx.Float32,
                                                       block_size=MX_BLOCK))
        lds_b128 = _make_lds_load(128)
        lds_b32 = _make_lds_load(32)
        wave = rocdl.readfirstlane(_T.i32, tid // WAVE)
        lane = tid % WAVE
        lane16 = lane % 16
        kgrp = lane // 16
        wmb = (wave // n_warp) * 128
        wnb = (wave % n_warp) * 128
        c_frags = [fx.make_rmem_tensor(WMMA_N // 2, fx.Float32) for _ in range_constexpr(m_reps * n_reps)]
        for cf in c_frags:
            cf.fill(0.0)
        lds_base = fx.Int32(fx.ptrtoint(base_ptr))
        lane_a = (wmb + lane16) * A_ROW + kgrp * 16
        lane_b = STAGE_A + (wnb + lane16) * B_ROW + kgrp * 16
        lane_sa = SC_OFF + (wmb + lane16) * SP
        zl = lane * (i32_n >> 30)  # opaque VGPR zero (see mxgg)
        zs = i32_n >> 30  # opaque uniform zero: the unrolled loop must not see kt % 2 as a constant, else the
        # stage offset (139264 B) is folded into every load's immediate, overflows its 16 bits and costs a v_add +
        # full WMMA drain per load
        sb_row0 = SC_OFF + PLANE_SA + (wnb // WMMA_N) * SP

        def addrs(kt):
            """address VGPRs of k-tile kt (formed at the start of the step, >= 4 WMMAs before first use)"""
            st = fx.Int32(fx.ptrtoint(_stage((kt + zs) % 2)))
            return dict(a=st + lane_a, b=st + lane_b, sa=lds_base + lane_sa + kt * (KI * 4),
                        sb=zl + (lds_base + sb_row0 + kt * (KI * 4)))

        def _ld_frag(base, off, dst=None):
            v = [fx.Vector(lds_b128(base, off + 32 * j)) for j in range_constexpr(4)]
            v01 = v[0].shuffle(v[1], list(range(8)))
            v23 = v[2].shuffle(v[3], list(range(8)))
            reg = fx.make_rmem_tensor(16, fx.Int32) if dst is None else dst
            reg.store(v01.shuffle(v23, list(range(16))))
            return reg

        def frag_a(ad, ks, mi, dst=None):
            return _ld_frag(ad["a"], mi * WMMA_M * A_ROW + ks * WMMA_K, dst)

        def frag_b(ad, ks, ni, dst=None):
            return _ld_frag(ad["b"], ni * WMMA_N * B_ROW + ks * WMMA_K, dst)

        def scale_a(ad, ks, mi):
            return lds_b32(ad["sa"], mi * WMMA_M * SP + ks * 4)

        def scale_b(ad, ks, ni):
            return lds_b32(ad["sb"], ni * SP + ks * 4)

        def _set():
            return dict(b=[fx.make_rmem_tensor(16, fx.Int32) for _ in range_constexpr(n_reps)],
                        a=fx.make_rmem_tensor(16, fx.Int32), sa=fx.make_rmem_tensor(1, fx.Int32),
                        sb=[fx.make_rmem_tensor(1, fx.Int32) for _ in range_constexpr(n_reps)])

        SETS = [_set(), _set()]

        def preload(kt, dst):
            ad = addrs(kt)
            for ni in range_constexpr(n_reps):
                frag_b(ad, 0, ni, dst["b"][ni])
                dst["sb"][ni].store(scale_b(ad, 0, ni))
            frag_a(ad, 0, 0, dst["a"])
            dst["sa"].store(scale_a(ad, 0, 0))

        def _wmmas(bfr, afr, sav, sbv, mi, lo, hi):
            for ni in range_constexpr(lo, hi):
                idx = mi * n_reps + ni
                fx.gemm(mma_atom, c_frags[idx], bfr[ni], afr, c_frags[idx], scale_a=sbv[ni], scale_b=sav)

        def bar():
            rocdl.sched_barrier(0)
            rocdl.s_barrier_signal(-1)
            rocdl.s_barrier_wait(-1)
            rocdl.sched_barrier(0)

        def step(kt, src, dst):
            """k-tile kt (2 x 64 WMMAs) from carrier set src; dst None = last k-tile (no wait / issue / prefetch)."""
            rocdl.sched_barrier(0)
            ad = addrs(kt)
            if const_expr(dst is not None):
                an = addrs(kt + 1)
            h = n_reps // 2
            # ---- first 128-k half: B of the second half double-buffered, A one row group ahead
            sb0 = [src["sb"][ni].load()[0] for ni in range_constexpr(n_reps)]
            a0 = [src["a"]] + [None] * m_reps
            s0 = [src["sa"].load()[0]] + [None] * m_reps
            b1 = [None] * n_reps
            sb1 = [None] * n_reps
            for mi in range_constexpr(m_reps):
                rocdl.sched_barrier(0)
                _wmmas(src["b"], a0[mi], s0[mi], sb0, mi, 0, h)
                rocdl.sched_barrier(0)
                if const_expr(mi + 1 < m_reps):
                    a0[mi + 1] = frag_a(ad, 0, mi + 1)
                    s0[mi + 1] = scale_a(ad, 0, mi + 1)[0]
                else:  # second half's A of group 0
                    a0[m_reps] = frag_a(ad, 1, 0)
                    s0[m_reps] = scale_a(ad, 1, 0)[0]
                b1[mi] = frag_b(ad, 1, mi)
                if const_expr(mi == 0):
                    for ni in range_constexpr(n_reps):
                        sb1[ni] = scale_b(ad, 1, ni)[0]
                rocdl.sched_barrier(0)
                _wmmas(src["b"], a0[mi], s0[mi], sb0, mi, h, n_reps)
            # ---- second 128-k half
            a1 = [a0[m_reps]] + [None] * (m_reps - 1)
            s1 = [s0[m_reps]] + [None] * (m_reps - 1)
            for mi in range_constexpr(m_reps):
                rocdl.sched_barrier(0)
                if const_expr(dst is not None and mi == early):
                    # every wave has its remaining A fragments of stage kt in registers: drain them, wait for
                    # k-tile kt+1, barrier -> stage kt is free (k-tile kt+2 goes there) and stage kt+1 is complete
                    rocdl.s_wait_dscnt(0)
                    tdm_ops.tensor_wait(0)
                    bar()
                    if kt + 2 < K_TILES:
                        issue(kt % 2, kt + 2)
                _wmmas(b1, a1[mi], s1[mi], sb1, mi, 0, h)
                rocdl.sched_barrier(0)
                if const_expr(dst is None):
                    if const_expr(mi + 1 < m_reps):
                        a1[mi + 1] = frag_a(ad, 1, mi + 1)
                        s1[mi + 1] = scale_a(ad, 1, mi + 1)[0]
                else:
                    # before the early barrier: groups 0..early-2 read one A fragment ahead, group early-1 all the
                    # rest (early..7), so no wave reads stage kt after the barrier
                    if const_expr(mi + 1 <= early - 1):
                        a1[mi + 1] = frag_a(ad, 1, mi + 1)
                        s1[mi + 1] = scale_a(ad, 1, mi + 1)[0]
                    if const_expr(mi == early - 1):
                        for j in range_constexpr(early, m_reps):
                            a1[j] = frag_a(ad, 1, j)
                            s1[j] = scale_a(ad, 1, j)[0]
                if const_expr(dst is not None and mi >= early):
                    for q in range_constexpr(BPG):
                        frag_b(an, 0, (mi - early) * BPG + q, dst["b"][(mi - early) * BPG + q])
                    if const_expr(mi == early):
                        for ni in range_constexpr(n_reps):
                            dst["sb"][ni].store(scale_b(an, 0, ni))
                    if const_expr(mi + 1 == m_reps):
                        frag_a(an, 0, 0, dst["a"])
                        dst["sa"].store(scale_a(an, 0, 0))
                rocdl.sched_barrier(0)
                _wmmas(b1, a1[mi], s1[mi], sb1, mi, h, n_reps)

        def epilogue(gC_base):
            fx.barrier()
            sC = _view(base_ptr, out_cls, (tile_m, C_ROW), (C_ROW, 1))
            for mi in range_constexpr(m_reps):
                row = wmb + mi * WMMA_M + lane16
                for ni in range_constexpr(n_reps):
                    col = wnb + ni * WMMA_N + kgrp * (WMMA_N // 2)
                    hv = c_frags[mi * n_reps + ni].load().to(out_cls)
                    fx.ptr_store(hv.bitcast(fx.Int8), base_ptr + (row * C_ROW + col) * 2)
            fx.barrier()
            gtC = _gview(gC_base, fx.Int64(m_row) * ldc64 + fx.Int64(blk_n), (tile_m, C_ROW), (C_ROW, 1))
            atomC = fx.rocdl.make_tdm_atom(gtC, [rows, tile_n], strides=[ldc64, None], num_warps=num_waves)
            fx.copy(atomC, sC, gtC)
            tdm_ops.tensor_wait(0)

        # prologue: planes + k-tile 0, wait, read k-tile 0's first fragments, k-tile 1 in flight
        fx.copy(atomSA, gSA, sSA)
        fx.copy(atomSB, gSB, sSB)
        issue(0, 0)
        tdm_ops.tensor_wait(0)
        fx.barrier()
        preload(0, SETS[0])
        issue(1, 1)
        n_pre = K_TILES - 1  # k-tiles that prefetch the next one
        n_pair = n_pre // 2
        for kp in range(n_pair):
            step(kp * 2, SETS[0], SETS[1])
            step(kp * 2 + 1, SETS[1], SETS[0])
        if n_pre % 2 == 1:
            step(n_pair * 2, SETS[0], SETS[1])
            for ni in range_constexpr(n_reps):
                SETS[0]["b"][ni].store(SETS[1]["b"][ni].load())
                SETS[0]["sb"][ni].store(SETS[1]["sb"][ni].load())
            SETS[0]["a"].store(SETS[1]["a"].load())
            SETS[0]["sa"].store(SETS[1]["sa"].load())
        step(K_TILES - 1, SETS[0], None)
        epilogue(_cast(out_cls, arg_c))

    kernel_mxgk(arg_c, arg_a, arg_b, arg_offs, arg_sa, arg_sb, i32_mtiles, i32_n, i32_k).launch(
        grid=(i32_mtiles * (i32_n // tile_n), 1, 1), block=(block, 1, 1), stream=stream)


launch_mxgk_fp8blk_nt_gfx1250.compile_hints["llvm_options"] = {"amdgpu-expert-scheduling-mode": True}
_COMPILED_K: dict = {}


def mxgk_fp8blk_nt(a, b, sa, sb, offs_i32, out=None, early=4):
    """mxgk: drop-in for flygg_fp8blk_nt(..., tile=(256, 256, 256), num_buffers=2, aligned=True, bblk=1) (same TDM
    traffic, cross-k-tile fragment pipeline). Bit-identical. Needs K % 256 == 0, K >= 512, power-of-two K (scale plane
    TDM pad interval), group offsets multiples of 256."""
    assert sa.dtype == torch.uint8 and sb.dtype == torch.uint8 and sa.is_contiguous() and sb.is_contiguous()
    M, K = a.shape
    G, N, Kb = b.shape
    assert K == Kb and a.is_contiguous() and b.is_contiguous()
    assert K % 256 == 0 and K >= 512 and (K & (K - 1)) == 0 and N % 256 == 0 and M % 256 == 0
    assert sa.shape == (M, K // 32) and sb.shape == (G, N, K // 32)
    if out is None:
        out = torch.empty((M, N), device=a.device, dtype=torch.bfloat16)
    for t in (a, b, sa, sb, offs_i32, out):
        assert t.data_ptr() != 0
    a_id = 0 if a.dtype == torch.float8_e4m3fn else 1
    b_id = 0 if b.dtype == torch.float8_e4m3fn else 1
    args = (_ptr(out), _ptr(a), _ptr(b), _ptr(offs_i32), _ptr(sa), _ptr(sb), M // 256,
            fx.Stream(torch.cuda.current_stream(device=a.device)), N, K, G, a_id, b_id, K, early)
    key = (G, a_id, b_id, K, early)
    c = _COMPILED_K.get(key)
    if c is None:
        _COMPILED_K[key] = flyc.compile(launch_mxgk_fp8blk_nt_gfx1250, *args)
    else:
        c(*args)
    return out
