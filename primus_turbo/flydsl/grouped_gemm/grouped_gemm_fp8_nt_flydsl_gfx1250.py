"""FlyDSL gfx1250 FP8 tensorwise M-grouped NT GEMM (forward; dgrad with a transposed weight copy).

out[rows_g] = a[rows_g] @ b[g]^T * (sa * sb),  a: [M_total, K] fp8, b: [G, N, K] fp8, out: [M_total, N] bf16.
Derived from primus_turbo/flydsl/gemm/gemm_gfx1250_kernel.py (TDM + wave32 WMMA 16x16x128 fp8). The M axis is
cut per expert: grid = (M_total // tile_m + G) m-tiles x N tiles (an upper bound, no host sync); each m-tile
decodes (expert, first row, rows) from the group offsets with an unrolled scan, TDM clamps the ragged rows, and
dead tiles (past the last expert) run only the pipeline prologue with zero rows.
"""
from __future__ import annotations

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl._mlir._mlir_libs._mlirDialectsFlyROCDL import MmaOpGFX1250_WMMAType
from flydsl.expr import const_expr, range_constexpr, rocdl
from flydsl.expr.rocdl import tdm_ops
from flydsl.expr.typing import Constexpr
from flydsl.expr.typing import T as _T

WAVE = 32
_A_ID = {0: fx.Float8E4M3FN, 1: fx.Float8E5M2}


def _make_lds_load(bits):
    layout = fx.make_layout(bits // 32, 1)
    atom = fx.make_copy_atom(fx.UniversalCopy(bits), fx.Int32)
    ptr_ty = fx.PointerType.get(fx.Int32.ir_type, fx.AddressSpace.Shared, bits // 8)

    def load(base, byte_offset):
        ptr = fx.inttoptr(ptr_ty, fx.Int32(base) + fx.Int32(byte_offset))
        reg = fx.make_rmem_tensor(layout, fx.Int32)
        fx.copy_atom_call(atom, fx.Tensor(fx.make_view(ptr, layout)), reg)
        return reg.load()

    return load


@flyc.jit
def launch_gg_fp8_nt_gfx1250(
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
    tile_m: Constexpr[int],
    tile_n: Constexpr[int],
    tile_k: Constexpr[int],
    m_warp: Constexpr[int],
    n_warp: Constexpr[int],
    num_buffers: Constexpr[int],
    a_id: Constexpr[int],
    b_id: Constexpr[int],
):
    a_cls, b_cls = _A_ID[a_id], _A_ID[b_id]
    WMMA_M = WMMA_N = 16
    WMMA_K = 128
    num_waves = m_warp * n_warp
    block = num_waves * WAVE
    assert tile_m % (m_warp * WMMA_M) == 0 and tile_n % (n_warp * WMMA_N) == 0
    assert tile_k % WMMA_K == 0 and num_buffers >= 2
    assert tile_m % num_waves == 0 and tile_n % num_waves == 0
    k_iters = tile_k // WMMA_K
    out_cls = fx.BFloat16
    LDS_PAD = 16
    A_ROW = tile_k + LDS_PAD
    B_ROW = tile_k + LDS_PAD
    STAGE_A = tile_m * A_ROW
    STAGE_B = tile_n * B_ROW
    PITCH = STAGE_A + STAGE_B
    C_ROW = tile_n + 8
    ARENA = max(num_buffers * PITCH, tile_m * C_ROW * 2)
    assert ARENA <= 160 * 1024, f"LDS arena {ARENA} B exceeds 160 KiB"
    n_ops = 2
    G = n_groups

    kernel_name = f"moe_gg_fp8_nt_gfx1250_t{tile_m}x{tile_n}x{tile_k}_w{m_warp}x{n_warp}_nb{num_buffers}_a{a_id}{b_id}_g{G}"

    @flyc.kernel(name=kernel_name, known_block_size=[block, 1, 1])
    def kernel_gg(
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
        n_tiles_n = (i32_n + (tile_n - 1)) // tile_n
        bid = fx.Int32(bid_x)
        t = rocdl.readfirstlane(_T.i32, bid // n_tiles_n)
        pn = bid % n_tiles_n
        blk_n = pn * tile_n
        n_oob = i32_n - blk_n

        i32_ptr = fx.PointerType.get(elem_ty=fx.Int32.ir_type, address_space=fx.AddressSpace.Global, alignment=4)
        f32_ptr = fx.PointerType.get(elem_ty=fx.Float32.ir_type, address_space=fx.AddressSpace.Global, alignment=4)
        mt = fx.recast_iter(i32_ptr, arg_offs)
        # unrolled scan: which expert owns m-tile t, its first row and row count
        zero = t * fx.Int32(0)
        one = zero + fx.Int32(1)
        acc = zero
        found = zero
        g_idx = zero
        first_g = zero
        len_g = zero
        row_beg = zero
        prev = mt[zero]
        for i in range_constexpr(G):
            nxt = mt[zero + fx.Int32(i + 1)]
            li = nxt - prev
            bi = (li + fx.Int32(tile_m - 1)) // fx.Int32(tile_m)
            acc2 = acc + bi
            take = (t < acc2).select(one, zero) * (one - found)
            g_idx = g_idx + take * (fx.Int32(i) - g_idx)
            first_g = first_g + take * (acc - first_g)
            len_g = len_g + take * (li - len_g)
            row_beg = row_beg + take * (prev - row_beg)
            found = found + take
            acc = acc2
            prev = nxt
        len_g = len_g * found
        row_beg = row_beg * found
        local = t - first_g
        raw = len_g - local * fx.Int32(tile_m)
        rows = (raw < fx.Int32(tile_m)).select(raw, fx.Int32(tile_m))
        rows = (rows > zero).select(rows, zero)
        m_row = (rows > zero).select(row_beg + local * fx.Int32(tile_m), zero)
        g_idx = rocdl.readfirstlane(_T.i32, g_idx)
        rows = rocdl.readfirstlane(_T.i32, rows)
        m_row = rocdl.readfirstlane(_T.i32, m_row)
        K_TILES = (rows > zero).select(i32_k // tile_k, fx.Int32(num_buffers - 1))

        sa = fx.recast_iter(f32_ptr, arg_sa)
        sb = fx.recast_iter(f32_ptr, arg_sb)
        scale = sa[fx.Int32(0)] * sb[fx.Int32(0)]

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

        gA_base = _cast(a_cls, arg_a)
        gB_base = _cast(b_cls, arg_b)
        gC_base = _cast(out_cls, arg_c)
        gA = _gview(gA_base, fx.Int64(m_row) * lda64, (tile_m, tile_k), (i32_k, 1))
        gB = _gview(gB_base, (fx.Int64(g_idx) * fx.Int64(i32_n) + fx.Int64(blk_n)) * lda64, (tile_n, tile_k), (i32_k, 1))
        atomA = fx.rocdl.make_tdm_atom(gA, [rows, None], strides=[lda64, None], num_warps=num_waves,
                                       pad_interval=tile_k, pad_amount=LDS_PAD)
        atomB = fx.rocdl.make_tdm_atom(gB, [n_oob, None], strides=[lda64, None], num_warps=num_waves,
                                       pad_interval=tile_k, pad_amount=LDS_PAD)

        def sA_of(s):
            return _view(_stage(s), a_cls, (tile_m, tile_k), (A_ROW, 1))

        def sB_of(s):
            return _view(fx.add_offset(_stage(s), STAGE_A), b_cls, (tile_n, tile_k), (B_ROW, 1))

        def issue(s, kt):
            kt64 = fx.Int64(kt)
            fx.copy(atomA, gA, sA_of(s), imm_offset=kt64 * tile_k)
            fx.copy(atomB, gB, sB_of(s), imm_offset=kt64 * tile_k)

        mma_atom = fx.make_mma_atom(
            MmaOpGFX1250_WMMAType.get(WMMA_M, WMMA_N, WMMA_K, b_cls.ir_type, a_cls.ir_type, fx.Float32.ir_type,
                                      sign_a=False, sign_b=False, clamp=False)
        )
        warp_m = tile_m // m_warp
        warp_n = tile_n // n_warp
        m_reps = warp_m // WMMA_M
        n_reps = warp_n // WMMA_N
        lds_b128 = _make_lds_load(128)
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

        def _operand(base, plane_off, row, row_bytes, ks):
            b0 = plane_off + row * row_bytes + ks * WMMA_K + kgrp * 16
            v = [fx.Vector(lds_b128(base, b0 + 32 * j)) for j in range_constexpr(4)]
            v01 = v[0].shuffle(v[1], list(range(8)))
            v23 = v[2].shuffle(v[3], list(range(8)))
            reg = fx.make_rmem_tensor(16, fx.Int32)
            reg.store(v01.shuffle(v23, list(range(16))))
            return reg

        def compute(s):
            base = fx.Int32(fx.ptrtoint(_stage(s)))
            for ks in range_constexpr(k_iters):
                bf = [_operand(base, STAGE_A, wnb + ni * WMMA_N + lane16, B_ROW, ks) for ni in range_constexpr(n_reps)]
                for mi in range_constexpr(m_reps):
                    af = _operand(base, 0, wmb + mi * WMMA_M + lane16, A_ROW, ks)
                    for ni in range_constexpr(n_reps):
                        idx = mi * n_reps + ni
                        fx.gemm(mma_atom, c_frags[idx], bf[ni], af, c_frags[idx])

        for j in range_constexpr(num_buffers - 1):
            issue(j, j)
        n_steady = K_TILES - (num_buffers - 1)
        for kt in range(n_steady):
            tdm_ops.tensor_wait((num_buffers - 2) * n_ops)
            fx.barrier()
            nxt = kt + (num_buffers - 1)
            issue(nxt % num_buffers, nxt)
            compute(kt % num_buffers)
        for j in range_constexpr(num_buffers - 1):
            tdm_ops.tensor_wait((num_buffers - 2 - j) * n_ops)
            fx.barrier()
            compute((n_steady + j) % num_buffers)

        fx.barrier()
        sC = _view(base_ptr, out_cls, (tile_m, C_ROW), (C_ROW, 1))
        for mi in range_constexpr(m_reps):
            row = wmb + mi * WMMA_M + lane16
            for ni in range_constexpr(n_reps):
                col = wnb + ni * WMMA_N + kgrp * (WMMA_N // 2)
                h = (c_frags[mi * n_reps + ni].load() * scale).to(out_cls)
                fx.ptr_store(h.bitcast(fx.Int8), base_ptr + (row * C_ROW + col) * 2)
        fx.barrier()
        gtC = _gview(gC_base, fx.Int64(m_row) * ldc64 + fx.Int64(blk_n), (tile_m, C_ROW), (C_ROW, 1))
        atomC = fx.rocdl.make_tdm_atom(gtC, [rows, tile_n], strides=[ldc64, None], num_warps=num_waves)
        fx.copy(atomC, sC, gtC)
        tdm_ops.tensor_wait(0)

    kernel_gg(arg_c, arg_a, arg_b, arg_offs, arg_sa, arg_sb, i32_mtiles, i32_n, i32_k).launch(
        grid=(i32_mtiles * ((i32_n + (tile_n - 1)) // tile_n), 1, 1), block=(block, 1, 1), stream=stream
    )


launch_gg_fp8_nt_gfx1250.compile_hints["llvm_options"] = {"amdgpu-expert-scheduling-mode": True}

_COMPILED: dict = {}


def _ptr(t):
    return flyc.from_c_void_p(fx.Int8, t.data_ptr(), assumed_align=16)


def flygg_fp8_nt(a, b, sa, sb, offs_i32, out=None, tile=(256, 256, 128), m_warp=2, n_warp=2, num_buffers=2):
    """a [M, K] fp8, b [G, N, K] fp8 contiguous, offs_i32 [G+1] int32. Returns out [M, N] bf16 (rows past
    offs[G] are not written)."""
    M, K = a.shape
    G, N, Kb = b.shape
    assert K == Kb and a.is_contiguous() and b.is_contiguous()
    tm, tn, tk = tile
    assert K % tk == 0 and N % tn == 0 and K // tk >= num_buffers - 1
    if out is None:
        out = torch.empty((M, N), device=a.device, dtype=torch.bfloat16)
    a_id = 0 if a.dtype == torch.float8_e4m3fn else 1
    b_id = 0 if b.dtype == torch.float8_e4m3fn else 1
    mtiles = M // tm + G
    args = (_ptr(out), _ptr(a), _ptr(b), _ptr(offs_i32), _ptr(sa), _ptr(sb), mtiles,
            fx.Stream(torch.cuda.current_stream(device=a.device)), N, K, G, tm, tn, tk, m_warp, n_warp,
            num_buffers, a_id, b_id)
    key = (G, tm, tn, tk, m_warp, n_warp, num_buffers, a_id, b_id)
    c = _COMPILED.get(key)
    if c is None:
        _COMPILED[key] = flyc.compile(launch_gg_fp8_nt_gfx1250, *args)
    else:
        c(*args)
    return out
