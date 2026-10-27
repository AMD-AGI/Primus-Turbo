###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Gathered-operand variable-K grouped GEMM for gfx1250 Mega MoE wgrads.

One operand's reduction rows come through a row-index table (TDM gather), so the
wgrad can read the dispatched token buffer in pool order. Lives under
``mega.gfx1250.bf16`` so the shared ``grouped_gemm_bf16_kernel_gfx1250`` stays free
of Mega-specific gather paths.

Shared tile helpers and the dense / non-gather variable-K path stay in
``primus_turbo.flydsl.grouped_gemm.grouped_gemm_bf16_kernel_gfx1250``.
"""

from __future__ import annotations

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl._mlir import ir
from flydsl._mlir.dialects import arith as std_arith
from flydsl._mlir.dialects import arith as _std_arith
from flydsl._mlir.dialects import llvm as _llvm
from flydsl._mlir.dialects import rocdl as _rocdl
from flydsl._mlir.dialects import vector as _vector
from flydsl.expr import arith, const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.arith import _to_raw as _raw
from flydsl.expr.utils.arith import ArithValue
from flydsl.expr.rocdl import tdm_ops
from flydsl.expr.typing import Constexpr, T, Vector as Vec

# Shared helpers / constants from the gfx1250 grouped GEMM (non-gather).
from primus_turbo.flydsl.grouped_gemm.grouped_gemm_bf16_kernel_gfx1250 import (
    _F16,
    _GFX1250_LDS_BYTES,
    _TDM_SCOPE,
    _TDM_TH_LOAD,
    _TDM_TH_STORE,
    _WG_BARRIER,
    _as_ptr,
    _check_smem,
    _check_tdm_cache,
    _get_arch,
    _group_m_tile_decode,
    _i32_table,
    _make_lds_copy_ops,
    _nn_b_pad,
    _pipeline_fence,
    _require_gfx1250,
    _vk_auto_split_bar,
    _vk_lds_pad,
    _vk_wave_grid,
    _workgroup_barrier,
    _xcd_band_remap,
    build_expert_table,
    check_prebuilt,
)

# ---------------------------------------------------------------------------
# Gathered operand (variable-K)
# ---------------------------------------------------------------------------
# One operand's reduction rows can come through a row-index table (reduction row -> source
# row) as TDM gather loads, 16 rows per instruction with 16-bit indices or 8 with 32-bit
# ones, so a caller whose rows live in another order (Mega MoE's pool order over the
# dispatch buffer) need not materialize them. Built by hand rather than through
# tdm_ops.make_tensor_gather_descriptor, which wants the LDS side as a memref and a constant
# row stride; the field layout is CDNA5 ISA 4.10.3.2, the bits that builder packs.


class _GatherDesc:
    def __init__(self, dgroup0, dgroup1, dgroup2, dgroup3):
        self.dgroup0, self.dgroup1, self.dgroup2, self.dgroup3 = dgroup0, dgroup1, dgroup2, dgroup3


def _readlane_i32(v, lane: int):
    """Lane ``lane`` of a per-lane i32, as a wave-uniform SGPR value."""
    op = rocdl.readlane(res=T.i32, src=_raw(v), lane=_raw(fx.Int32(lane)))
    return ArithValue(op.result if hasattr(op, "result") else op)


def _vk_gather_desc(lds_addr, src_addr, rows, idx_bits, row_width, num_src_rows, row_stride, lds_pad):
    """One gather load of ``rows`` (wave-uniform i32 indices < ``num_src_rows``), each
    ``row_width`` bf16 elements from ``src_addr`` (i64 bytes) with ``row_stride`` (i32
    elements), into LDS rows ``row_width * 2 + lds_pad`` bytes apart from ``lds_addr`` (i32 bytes)."""
    eb = 2
    i32 = ir.IntegerType.get_signless(32)

    def c32(v):
        return fx.Int32(v).ir_value()

    interval_dw, amount_dw = row_width * eb // 4, lds_pad // 4
    assert interval_dw & (interval_dw - 1) == 0, "TDM pad interval must be a power of two dwords"
    g = _raw(src_addr)
    lo = _std_arith.TruncIOp(i32, g).result
    hi = _std_arith.TruncIOp(i32, _std_arith.ShRUIOp(g, _raw(fx.Int64(32))).result).result
    hi = _std_arith.OrIOp(hi, c32(1 << 31)).result
    pred = c32(1 | ((1 if idx_bits == 32 else 0) << 30) | (1 << 31))
    dgroup0 = _vector.from_elements(T.vec(4, T.i32), [pred, _raw(lds_addr), lo, hi])
    s0 = (eb.bit_length() - 1) << 16 | 1 << 20 | (interval_dw.bit_length() - 2) << 22 | (amount_dw - 1) << 25
    words = [
        s0,
        (row_width & 0xFFFF) << 16,
        (row_width >> 16) & 0xFFFF | (num_src_rows & 0xFFFF) << 16,
        (num_src_rows >> 16) & 0xFFFF | row_width << 16,
        len(rows),
    ]
    dgroup1 = _vector.from_elements(
        T.vec(8, T.i32), [c32(w) for w in words] + [_raw(fx.Int32(row_stride)), c32(0), c32(0)]
    )
    if idx_bits == 32:
        packed = [_raw(r) for r in rows]
    else:
        # indices are < num_src_rows < 2^16, so the high half needs no mask
        packed = [_raw(rows[2 * w] | (rows[2 * w + 1] << fx.Int32(16))) for w in range(len(rows) // 2)]
    packed += [c32(0)] * (8 - len(packed))
    dgroup2 = _vector.from_elements(T.vec(4, T.i32), packed[:4])
    dgroup3 = _vector.from_elements(T.vec(4, T.i32), packed[4:])
    return _GatherDesc(dgroup0, dgroup1, dgroup2, dgroup3)


@flyc.jit
def _launch_grouped_bf16_variable_k(
    arg_c: fx.Pointer,
    arg_a: fx.Pointer,
    arg_b: fx.Pointer,
    arg_etab: fx.Pointer,
    arg_ridx: fx.Pointer,
    stream: fx.Stream,
    i32_n: fx.Int32,
    i32_k: fx.Int32,
    G: Constexpr[int],
    N_BLOCKS_N: Constexpr[int],
    N_BLOCKS_K: Constexpr[int],
    C_GRP_ELEMS: Constexpr[int],
    tile_m: Constexpr[int],
    tile_n: Constexpr[int],
    tile_k: Constexpr[int],
    m_warp: Constexpr[int],
    n_warp: Constexpr[int],
    num_buffers: Constexpr[int],
    is_f16: Constexpr[int] = 0,
    group_m: Constexpr[int] = 16,
    num_xcd: Constexpr[int] = 1,
    xcd_band: Constexpr[int] = 32,
    # ---- A/B and measurement knobs; every default reproduces the shipped code ---
    wmma_b2b: Constexpr[int] = 0,
    frag_pipeline: Constexpr[int] = 1,
    # 1 = run the k-tile's last sub-step of WMMA inside the workgroup barrier's
    #     signal-to-wait window instead of before the signal. Same knob and same
    #     mechanism as the NT/NN launcher's `split_bar`; see `_split_fence` there
    #     and `_auto_split_bar` for the WG/CU boundary.
    split_bar: Constexpr[int] = 0,
    # ⚠️ **EXPERIMENT ONLY. Not a tuning knob. Do not ship a non-zero value.**
    #
    # > 0 pins the per-expert k-tile count to a compile-time constant instead of
    # deriving it from `m_len`, which unrolls the steady-state loop. **It is only
    # correct when every expert really has this many k-tiles**; a value that
    # disagrees with `m_len` reads rows past the expert or drops rows inside it. The
    # entry point refuses it unless the caller also passes
    # `i_am_running_an_experiment=True`.
    k_tiles_const: Constexpr[int] = 0,
    # > 0 overrides the output's row stride in elements, decoupling it from OUT_N --
    # which is otherwise the row stride of both `b` (read) and `out` (written), so a
    # stride-alignment effect cannot be attributed to one side.
    #
    # ⚠️ This is the one parameter here that can fault the device rather than return a
    # wrong answer: `c_off` is `g_idx * C_GRP_ELEMS + blk_n * ldc + blk_k`, so the
    # stride and the per-expert span must move TOGETHER or the store walks off the
    # allocation. `gpu_recovery` is 0 on this box, which makes that a reboot. The
    # entry point therefore computes `C_GRP_ELEMS` from this value and refuses any
    # `out` whose real shape does not match it.
    out_ldc: Constexpr[int] = 0,
    # ---- TDM cache policy; see `_check_tdm_cache` for the encoding -----------
    # `tdm_load_cache` goes on both A's and B's read descriptors, `tdm_store_cache`
    # on C's write descriptor. 0 = shipped (TH_RT, SCOPE_WGP). They change no
    # address, no extent and no byte that is moved -- only which cache level is
    # asked to keep the line -- which is why they are safe to expose, and why no
    # non-zero value ships: the hypothesised direction measured worse.
    tdm_load_cache: Constexpr[int] = 0,
    tdm_store_cache: Constexpr[int] = 0,
    # Byte pad on both transposed LDS stage rows. Must be a multiple of 16 (the
    # transpose read needs every row base 16-byte aligned). The entry point
    # resolves it from `_vk_lds_pad`; the default stays at the flat 16 the launcher
    # shipped with, so a direct launcher call keeps the historical behaviour and the
    # residue rule lives in exactly one place. A compile-time constant, and in the
    # launcher's cache key -- which is why it is a parameter and not a local.
    lds_pad: Constexpr[int] = 16,
    # Which `wave_id` bit selects the M half of the warp grid. 0 = shipped; see
    # `_vk_wave_grid`.
    wave_grid: Constexpr[int] = 0,
    # Persistent workgroups. 0 = shipped -- one workgroup per output tile,
    # grid = TOTAL_TILES, and the emitted code is byte-identical to before this
    # existed (the shipped path calls the body once with no loop around it). > 0
    # launches exactly `persist` workgroups, each walking `TOTAL_TILES // persist`
    # tiles.
    #
    # `persist` must DIVIDE TOTAL_TILES so the trip count is a compile-time constant
    # and no tail guard is needed, and must leave >= 2 tiles per workgroup -- with
    # one tile 'the accumulator was not cleared' is unobservable.
    persist: Constexpr[int] = 0,
    # MEASUREMENT ONLY: skips the operand loads, the LDS transpose reads and every
    # WMMA and goes straight to the epilogue with whatever the accumulators hold, so
    # the write path can be timed through the store code the kernel really uses
    # rather than a re-derived microkernel. The RESULT IS GARBAGE by construction
    # and the correctness gate refuses to judge it. `const_expr` and defaulted to 0,
    # so the shipped trace is unchanged.
    wpath_only: Constexpr[int] = 0,
    # 1 / 2 = A / B is gathered: its reduction row m is row `arg_ridx[m]` (i32) of a source
    # matrix of `gather_rows` rows with the same row stride. 0 = both dense (`arg_ridx` unread).
    gather: Constexpr[int] = 0,
    gather_rows: Constexpr[int] = 0,
):
    """``out[g][n, k] = sum_{m in g} A[m, n] * B[m, k]``.

    ``arg_a`` is ``A[M_total, OUT_M]``, ``arg_b`` is ``B[M_total, OUT_N]``, both
    token-major and **not** pre-transposed; ``arg_c`` is ``[G, OUT_M, OUT_N]``.
    ``tile_m`` tiles the output's first axis, ``tile_n`` its second, and ``tile_k``
    is the reduction chunk along tokens.
    """
    WMMA_M = WMMA_N = 16
    WMMA_K = 32
    WAVE = 32  # gfx1250 is wave32 -- the single most invasive difference vs gfx950
    EB = 2
    K_WS = tile_k // WMMA_K
    warp_tile_m = tile_m // m_warp
    warp_tile_n = tile_n // n_warp
    wmma_m_rep = warp_tile_m // WMMA_M
    wmma_n_rep = warp_tile_n // WMMA_N
    n_acc = wmma_m_rep * wmma_n_rep
    num_waves = m_warp * n_warp
    block = num_waves * WAVE
    TILES_PER_EXPERT = N_BLOCKS_N * N_BLOCKS_K
    TOTAL_TILES = G * TILES_PER_EXPERT
    PERSIST = persist if persist else TOTAL_TILES
    TILES_PER_WG = TOTAL_TILES // PERSIST
    if persist:
        if TOTAL_TILES % persist:
            raise ValueError(f"persist={persist} must divide TOTAL_TILES={TOTAL_TILES}")
        if TILES_PER_WG < 2:
            raise ValueError(
                f"persist={persist} leaves {TILES_PER_WG} tile(s) per workgroup; with one "
                f"tile 'the accumulator was not cleared between tiles' is unobservable"
            )
    NB1 = num_buffers - 1

    if tile_k % WMMA_K or warp_tile_m % WMMA_M or warp_tile_n % WMMA_N:
        raise ValueError(f"tile ({tile_m},{tile_n},{tile_k}) / warps ({m_warp},{n_warp}) not WMMA-aligned")
    if K_WS < 2:
        raise ValueError(f"tile_k={tile_k} needs at least 2 WMMA K-steps")
    if num_waves < 2:
        raise ValueError("need >= 2 waves so the two TDM jobs land on different waves")
    if num_buffers < 2:
        raise ValueError("num_buffers must be >= 2")
    if split_bar not in (0, 1):
        # The entry point spells "auto" as -1 and resolves it against WG/CU before
        # it gets here; `const_expr` would read a leaked -1 as truthy.
        raise ValueError(
            f"split_bar must be 0 or 1 by the time it reaches the launcher, got {split_bar}; "
            f"-1 (auto) is resolved by the entry point via _auto_split_bar()"
        )
    if split_bar and frag_pipeline == 2:
        # `frag_pipeline=2` prefetches only the A fragments, so its last sub-step's
        # operands are not a single ring slot the way the other two branches leave
        # them -- there is no one `state` to hand to the barrier window. Not built.
        raise ValueError("split_bar is not supported with frag_pipeline=2")
    _check_tdm_cache(tdm_load_cache, _TDM_TH_LOAD, "tdm_load_cache")
    _check_tdm_cache(tdm_store_cache, _TDM_TH_STORE, "tdm_store_cache")

    # LDS runs [tile_k rows of m] x [output-index columns]; the transpose read
    # needs every row base 16-byte aligned, so the pad is in bytes.
    if lds_pad <= 0 or lds_pad % 16:
        raise ValueError(f"lds_pad must be a positive multiple of 16, got {lds_pad}")
    LDS_PAD = lds_pad
    LDS_A_ROW = tile_m * EB + LDS_PAD  # A tile: columns are output axis 0
    LDS_B_ROW = tile_n * EB + LDS_PAD  # B tile: columns are output axis 1
    STAGE_A = ((tile_k * LDS_A_ROW + 15) // 16) * 16
    STAGE_B = ((tile_k * LDS_B_ROW + 15) // 16) * 16
    PITCH = ((STAGE_A + STAGE_B + 1023) // 1024) * 1024
    elem_cls = fx.Float16 if is_f16 else fx.BFloat16
    C_STORE_B = ((tile_m * tile_n * EB + 127) // 128) * 128
    ARENA_B = max(num_buffers * PITCH, C_STORE_B)
    # Cross-tile prefetch precondition, COMPILE-TIME. The epilogue stages C at the
    # arena base and the prefetch of the next tile's first stage targets buffer 1, so
    # the two are disjoint only while the staging fits inside buffer 0:
    #
    #     PREFETCH_OK  <=>  C_STORE_B <= PITCH
    #
    # ⚠️ This inequality CHANGES SIGN with tile_k: `PITCH` is proportional to tile_k
    # while `C_STORE_B` is tile_m * tile_n * EB and does not contain tile_k at all, so
    # verifying it on one tile says nothing about another. Where it does not hold the
    # prefetch lands inside the staging the epilogue is still using and the output is
    # garbage -- hence an assertion in code rather than a note. A tile that fails it
    # emits no prefetch, which is correct and merely no faster.
    PREFETCH_OK = bool(persist) and C_STORE_B <= PITCH
    if gather not in (0, 1, 2):
        raise ValueError(f"gather must be 0 (none), 1 (A) or 2 (B), got {gather}")
    IDX_BITS = 16 if gather_rows < 0xFFFF else 32
    GATHER_ROWS_PER_OP = 16 if IDX_BITS == 16 else 8
    GATHER_ROWS_PER_WAVE = tile_k // num_waves
    if gather:
        # Every wave issues the same number of gather ops per k-tile, and with two buffers each
        # fence drains them all, so the per-wave TDM budget (TDM_PW) never has to count them.
        if num_buffers != 2 or persist:
            raise ValueError("gather needs num_buffers == 2 and persist == 0")
        if GATHER_ROWS_PER_WAVE % GATHER_ROWS_PER_OP or GATHER_ROWS_PER_WAVE > WAVE:
            raise ValueError(
                f"gather splits tile_k={tile_k} rows over {num_waves} waves into whole ops of "
                f"{GATHER_ROWS_PER_OP}, at most one row per lane"
            )
    if ARENA_B > _GFX1250_LDS_BYTES:
        raise ValueError(f"LDS arena {ARENA_B} B exceeds the gfx1250 budget {_GFX1250_LDS_BYTES} B")
    _check_smem(ARENA_B, str(_get_arch()))

    @flyc.kernel(known_block_size=[block, 1, 1])
    def kernel_grouped_variable_k(
        arg_c: fx.Pointer,
        arg_a: fx.Pointer,
        arg_b: fx.Pointer,
        arg_etab: fx.Pointer,
        arg_ridx: fx.Pointer,
        i32_n: fx.Int32,
        i32_k: fx.Int32,
    ):
        if const_expr(wmma_b2b):
            rocdl.disable_xdl_arb_stall()
        # Scalar-argument naming: `i32_n` carries OUT_M (output axis 0, and A's row
        # width) and `i32_k` carries OUT_N (output axis 1, B's row width and the
        # output row stride). The names are the autograd ones -- OUT_M is the dout
        # width N, OUT_N is the activation width K -- kept so the kernel body reads
        # the same as the bring-up version it was validated in.
        ldn_b = fx.Int64(i32_n) * EB  # A[M, OUT_M] row stride, bytes
        ldk_b = fx.Int64(i32_k) * EB  # B[M, OUT_N] row stride, bytes
        # C's row stride normally IS OUT_N, which is also `b`'s. `out_ldc` separates
        # them; see the parameter comment for why and for the bounds obligation.
        ldc64 = fx.Int64(out_ldc) if const_expr(out_ldc) else fx.Int64(i32_k)

        tid = fx.Int32(fx.thread_idx.x)
        wave = rocdl.readfirstlane(T.i32, tid // WAVE)
        lane = tid % WAVE
        lane16 = lane % 16
        kgrp = lane // 16
        l8 = lane % 8
        hi8 = lane16 // 8
        # Both spellings are bijections onto the same m_warp x n_warp grid, so the
        # tile decomposition, the LDS bytes, the instruction stream and the output
        # are all identical -- what moves is *which pair of waves ends up co-resident
        # on one SIMD*, and therefore whether that pair asks the LDS for the same
        # fragment at the same time. 0 is what shipped. See `_vk_wave_grid`.
        wave_m = (wave % m_warp) if const_expr(wave_grid) else (wave // n_warp)
        wave_n = (wave // m_warp) if const_expr(wave_grid) else (wave % n_warp)

        pid = fx.Int32(fx.block_idx.x)

        # One output tile. Lifted into a function ONLY so the persistent path can call
        # it in a loop; the shipped path calls it once, so the trace -- and therefore
        # the ISA -- is exactly what it was before `persist` existed. The loop is a
        # plain `for _t in range(...)` statement rather than a conditional iterable
        # because flydsl rewrites that pattern into a device loop and does not
        # recognise a ternary in the iterator position (it silently falls back to
        # builtin `range`, which then fails on the k-loop below).
        def _one_output_tile(tile_raw):
            tile = _xcd_band_remap(tile_raw, TOTAL_TILES, num_xcd, xcd_band)
            g_idx = rocdl.readfirstlane(T.i32, tile // fx.Int32(TILES_PER_EXPERT))
            local = tile % fx.Int32(TILES_PER_EXPERT)
            blk_n_idx, blk_k_idx = _group_m_tile_decode(local, N_BLOCKS_N, N_BLOCKS_K, group_m)

            # (m_start, m_len) per expert. m_len is the *valid* token count, which is
            # `masked_k[g]` when the caller passed one -- the padded tail between
            # m_start+m_len and the next expert's m_start is then never addressed.
            et = _i32_table(arg_etab)
            e = g_idx * fx.Int32(2)
            m_start = rocdl.readfirstlane(T.i32, et[e])
            m_len = rocdl.readfirstlane(T.i32, et[e + fx.Int32(1)])

            blk_n = blk_n_idx * tile_m
            blk_k = blk_k_idx * tile_n
            n_valid = i32_n - blk_n
            k_valid = i32_k - blk_k

            # ---- CORRECTNESS: back the *read* descriptors off so their window never
            # leaves the row.
            #
            # A descriptor claims tile_m*EB bytes per row, but a short last block has
            # only n_valid*EB valid, and the extents are not symmetric: a dim-0 extent
            # bounds addressing, a dim-1 extent bounds addressing on a STORE, but a
            # dim-1 extent on a LOAD clamps the DATA only -- an over-claimed load
            # walks off the allocation and faults the device.
            #
            # Reading [N-tile_m, N) instead of [blk_n, blk_n+tile_m) puts the whole
            # window inside the row by construction, at the cost of re-reading
            # n_back columns another block already owns. The fragments then index LDS
            # column r + n_back for accumulator row r, so rows r < n_valid land
            # exactly on their own data and the rest read into the row padding (still
            # inside the arena) and are dropped by the store's dim-0 extent, which
            # *is* an addressing bound.
            #
            # This is what `grouped_gemm_bf16_variable_k_supported()` checks: there
            # has to be a whole tile to back off into.
            blk_n_eff = (blk_n > (i32_n - fx.Int32(tile_m))).select(i32_n - fx.Int32(tile_m), blk_n)
            blk_k_eff = (blk_k > (i32_k - fx.Int32(tile_n))).select(i32_k - fx.Int32(tile_n), blk_k)
            n_back = blk_n - blk_n_eff  # 0 except on a short last block
            k_back = blk_k - blk_k_eff

            arena = fx.SharedAllocator(static=False)
            arena.allocate(ARENA_B)
            base_ptr = arena.base_ptr

            def _bidx(p):
                return fx.Int64(fx.ptrtoint(p))

            def _buf_ptr(s):
                return fx.add_offset(base_ptr, s * PITCH)

            # Which PHYSICAL buffer a logical pipeline slot lands in.
            #
            # Shipped (`persist == 0`): identity, and spelled as the original
            # expression so the trace -- and the ISA hash -- are unchanged.
            #
            # Persistent: a CONSTANT +1, so every tile starts its k-loop at buffer 1
            # and reaches buffer 0 only at the second slot. The epilogue's C staging
            # always sits at the arena base, inside buffer 0, and does not move with
            # the tile index, so buffer 1 is the only buffer free while the epilogue
            # runs and the cross-tile prefetch has to land there.
            #
            # ⚠️ It must be a CONSTANT +1 and not an alternation: because the staging
            # is always at the arena base, alternating by tile parity sends every other
            # tile's prefetch straight into the staging it is supposed to avoid.
            BUF_SHIFT = 1 % num_buffers if persist else 0

            def _slot(kt):
                return ((kt + BUF_SHIFT) % num_buffers) if BUF_SHIFT else (kt % num_buffers)

            def _gv(base, off, shape, stride):
                return fx.Tensor(fx.make_view(fx.add_offset(base, off), fx.make_layout(shape, stride)))

            def _lv(ptr, shape, stride):
                return fx.Tensor(fx.make_view(ptr, fx.make_layout(shape, stride)))

            def _tdm(gt, rows, cols_b, stride, pad_interval):
                """dim 0 = expert token tail (rows), dim 1 = ragged output columns.

                CORRECTNESS: a TDM extent is measured from the copy's ``imm_offset``
                and **not** from the descriptor base, so ``issue`` rebuilds the
                descriptor per k-tile with the row offset folded in and copies with
                ``imm_offset=0``. Building it once and walking K with ``imm_offset``
                clamps nothing, and every expert whose token count is not a multiple
                of ``tile_k`` then pulls in a whole extra tile of the *next*
                expert's tokens.
                """
                return fx.rocdl.make_tdm_atom(
                    gt,
                    [rows, cols_b],
                    strides=[stride, None],
                    num_warps=1,
                    pad_interval=pad_interval,
                    pad_amount=LDS_PAD,
                    cache_modifier=tdm_load_cache,
                    early_timeout=True,
                )

            gA_base = fx.recast_iter(fx.Int8, arg_a)  # A [M_total, OUT_M]
            gB_base = fx.recast_iter(fx.Int8, arg_b)  # B [M_total, OUT_N]
            gC_base = fx.recast_iter(fx.PointerType.get(elem_cls.ir_type, arg_c.address_space), arg_c)

            if const_expr(gather):
                ridx = _i32_table(arg_ridx)
                has_rows = m_len > fx.Int32(0)
                last_row = has_rows.select(m_len - fx.Int32(1), fx.Int32(0))
                # an empty expert's m_start can be the table's end
                first_row = has_rows.select(m_start, fx.Int32(0))
                gather_row = wave * fx.Int32(GATHER_ROWS_PER_WAVE) + lane % fx.Int32(GATHER_ROWS_PER_WAVE)

                def _gather_src_rows(kt):
                    # rows past the expert re-read its last one, which is finite, against the
                    # dense operand's zero-filled rows
                    r = fx.Int32(kt) * fx.Int32(tile_k) + gather_row
                    r = (r > last_row).select(last_row, r)
                    return Vec.from_elements([ridx[first_row + r]], fx.Int32)

                # this lane's source row for the next k-tile issued, loaded one k-tile ahead
                gather_cell = fx.make_rmem_tensor(1, fx.Int32)
                gather_cell.store(_gather_src_rows(0))

                def _issue_gather(stage, src_base, col0, lds_row, width, row_stride, kt):
                    src_rows = Vec(gather_cell.load())[0]
                    gather_cell.store(_gather_src_rows(fx.Int32(kt) + fx.Int32(1)))
                    lds0 = fx.Int32(_bidx(stage)) + wave * fx.Int32(GATHER_ROWS_PER_WAVE * lds_row)
                    src = fx.Int64(fx.ptrtoint(src_base)) + fx.Int64(col0) * fx.Int64(EB)
                    per_op = GATHER_ROWS_PER_OP
                    for q in range_constexpr(GATHER_ROWS_PER_WAVE // per_op):
                        rows = [_readlane_i32(src_rows, q * per_op + i) for i in range_constexpr(per_op)]
                        lds_addr = lds0 + fx.Int32(q * per_op * lds_row)
                        desc = _vk_gather_desc(
                            lds_addr, src, rows, IDX_BITS, width, gather_rows, row_stride, LDS_PAD
                        )
                        tdm_ops.tensor_load_gather(desc)

            def issue(s, kt, dec=None):
                pa = _buf_ptr(s)
                # `dec` lets the caller name a DIFFERENT tile's decode than the one this
                # closure was created for. Phase 2b uses it from the epilogue to fetch
                # the NEXT tile's first stage; everyone else passes None and gets this
                # tile's, so the shipped trace is unchanged.
                ms, ml, bne, bke = dec if dec is not None else (m_start, m_len, blk_n_eff, blk_k_eff)
                m_off = fx.Int32(ms) + fx.Int32(kt) * fx.Int32(tile_k)
                rem = ml - fx.Int32(kt) * fx.Int32(tile_k)
                rem = (rem < fx.Int32(0)).select(fx.Int32(0), rem)
                m_off64 = fx.Int64(m_off)
                if const_expr(gather == 1):
                    _issue_gather(pa, gA_base, bne, LDS_A_ROW, tile_m, i32_n, kt)
                else:
                    if wave == 0 % num_waves:
                        gt = _gv(
                            gA_base,
                            m_off64 * ldn_b + fx.Int64(bne) * fx.Int64(EB),
                            (tile_k, tile_m * EB),
                            (ldn_b, 1),
                        )
                        # Full width: the backed-off window is entirely valid columns, so
                        # there is nothing left for a dim-1 extent to clamp.
                        atom = _tdm(gt, rem, fx.Int32(tile_m * EB), ldn_b, tile_m * EB)
                        fx.copy(atom, gt, _lv(pa, (tile_k, tile_m * EB), (LDS_A_ROW, 1)))
                if const_expr(gather == 2):
                    _issue_gather(fx.add_offset(pa, STAGE_A), gB_base, bke, LDS_B_ROW, tile_n, i32_k, kt)
                else:
                    if wave == 1 % num_waves:
                        gt = _gv(
                            gB_base,
                            m_off64 * ldk_b + fx.Int64(bke) * fx.Int64(EB),
                            (tile_k, tile_n * EB),
                            (ldk_b, 1),
                        )
                        atom = _tdm(gt, rem, fx.Int32(tile_n * EB), ldk_b, tile_n * EB)
                        lds_b = _lv(fx.add_offset(pa, STAGE_A), (tile_k, tile_n * EB), (LDS_B_ROW, 1))
                        fx.copy(atom, gt, lds_b)

            wmb = wave_m * warp_tile_m
            wnb = wave_n * warp_tile_n

            # Per-lane transpose-read bases. See the derivation in the section header;
            # both addends below are compile-time from here on.
            a_lane = (l8 + kgrp * fx.Int32(8)) * fx.Int32(LDS_A_ROW) + (
                wmb + hi8 * fx.Int32(8) + n_back
            ) * fx.Int32(EB)
            b_lane = (
                (l8 + kgrp * fx.Int32(8)) * fx.Int32(LDS_B_ROW)
                + (wnb + hi8 * fx.Int32(8) + k_back) * fx.Int32(EB)
                + fx.Int32(STAGE_A)
            )
            vec8 = ir.VectorType.get([8], elem_cls.ir_type)
            _PTR3 = ir.Type.parse("!llvm.ptr<3>")

            def _tr(addr, const_off):
                """One transpose read at ``addr + const_off``.

                ``const_off`` must stay a compile-time constant: ds_load_tr16_b128 has
                a 16-bit unsigned offset field, so a constant folds into the
                instruction and costs no VALU, while a computed address costs an add
                *and* a live VGPR. Folding the buffer pointer in here rather than
                hoisting ``addr`` puts that add on nearly every load in the tile.
                """
                ptr_val = _llvm.inttoptr(_PTR3, _raw(addr + fx.Int32(const_off)))
                return Vec(_rocdl.ds_load_tr16_b128(vec8, ptr_val))

            def _frag(addr, row_stride, k_row_off, col_off):
                base = k_row_off * row_stride + col_off
                v0 = _tr(addr, base)
                v1 = _tr(addr, base + 16 * row_stride)
                return v0.shuffle(v1, list(range(16))).bitcast(fx.Int32)

            wmma_atom = fx.make_mma_atom(fx.rocdl.WMMA(WMMA_M, WMMA_N, WMMA_K, elem_cls, fx.Float32))
            # 16x16 tile over 32 lanes -> 8 f32 per lane (gfx950/MFMA wave64 would be 4).
            c_frags = [fx.make_rmem_tensor(8, fx.Float32) for _ in range_constexpr(n_acc)]
            for cf in c_frags:
                cf.store(Vec.filled(8, 0.0, fx.Float32))

            def _rmem(n, v):
                t = fx.make_rmem_tensor(n, fx.Int32)
                t.store(v)
                return t

            DS_A = DS_B = 2  # two transpose reads per fragment
            KS_DS = wmma_m_rep * DS_A + wmma_n_rep * DS_B

            def _load_ks(addrs, ks):
                a_addr, b_addr = addrs
                act = [
                    _rmem(8, _frag(a_addr, LDS_A_ROW, ks * WMMA_K, wm * WMMA_M * EB))
                    for wm in range_constexpr(wmma_m_rep)
                ]
                wt = [
                    _rmem(8, _frag(b_addr, LDS_B_ROW, ks * WMMA_K, wn * WMMA_N * EB))
                    for wn in range_constexpr(wmma_n_rep)
                ]
                return act, wt

            def _mma_ks(state):
                act, wt = state
                for wm in range_constexpr(wmma_m_rep):
                    for wn_raw in range_constexpr(wmma_n_rep):
                        # serpentine the inner order to shorten accumulator revisit distance
                        wn = (wmma_n_rep - 1 - wn_raw) if (wm % 2 == 1) else wn_raw
                        idx = wm * wmma_n_rep + wn
                        # B is the instruction's A operand: the accumulator's fast dim is N.
                        fx.gemm(wmma_atom, c_frags[idx], wt[wn], act[wm], c_frags[idx])

            def compute_ktile(buf, prefetch_kt, defer_last=False):
                # One address VGPR per operand for the whole tile; every fragment load
                # is then that register plus a compile-time offset.
                #
                # `defer_last=True` runs every sub-step but the last and returns the
                # last one's operands instead of issuing them, so the caller can put
                # that burst inside the barrier window (`split_bar`). The full
                # `s_wait_dscnt(0)` on the final sub-step still runs, which is what
                # makes the deferral legal: the LDS stage is drained before the
                # barrier and the deferred op then reads registers only.
                addrs = (fx.Int32(buf) + a_lane, fx.Int32(buf) + b_lane)
                n_mma = K_WS - 1 if const_expr(defer_last) else K_WS
                # frag_pipeline=1 stages the next k-step's fragments while the current
                # ones feed WMMA. That is 2 x 12 fragments x 8 VGPRs live at once on top
                # of 256 VGPRs of accumulator, which puts this kernel at the 512 ceiling
                # and spills 3 -- the transpose reads need more address registers than
                # the plain b128 reads the NT kernel uses. frag_pipeline=0 keeps one
                # set, trading latency hiding for headroom.
                if const_expr(frag_pipeline == 2):
                    # Prefetch only the A fragments rather than all twelve. The
                    # staircase waits drain the loads the next WMMA really consumes, so
                    # the lever is issuing loads earlier, which needs VGPRs: holding 4
                    # prefetched fragments instead of 12 frees 64 of them.
                    cur_a, cur_b = _load_ks(addrs, 0)
                    for ks in range_constexpr(K_WS):
                        nxt_a = (
                            [
                                _rmem(8, _frag(addrs[0], LDS_A_ROW, (ks + 1) * WMMA_K, wm * WMMA_M * EB))
                                for wm in range_constexpr(wmma_m_rep)
                            ]
                            if const_expr(ks + 1 < K_WS)
                            else None
                        )
                        rocdl.s_wait_dscnt(wmma_m_rep * DS_A if const_expr(nxt_a is not None) else 0)
                        if const_expr(ks == 0 and prefetch_kt is not None):
                            rocdl.sched_barrier(0)
                            issue(_slot(prefetch_kt), prefetch_kt)
                            rocdl.sched_barrier(0)
                        _mma_ks((cur_a, cur_b))
                        if const_expr(ks + 1 < K_WS):
                            cur_a = nxt_a
                            cur_b = [
                                _rmem(8, _frag(addrs[1], LDS_B_ROW, (ks + 1) * WMMA_K, wn * WMMA_N * EB))
                                for wn in range_constexpr(wmma_n_rep)
                            ]
                            rocdl.s_wait_dscnt(0)
                elif const_expr(frag_pipeline):
                    cur = _load_ks(addrs, 0)
                    for ks in range_constexpr(K_WS):
                        nxt = _load_ks(addrs, ks + 1) if const_expr(ks + 1 < K_WS) else None
                        rocdl.s_wait_dscnt(KS_DS if const_expr(nxt is not None) else 0)
                        if const_expr(ks == 0 and prefetch_kt is not None):
                            rocdl.sched_barrier(0)
                            issue(_slot(prefetch_kt), prefetch_kt)
                            rocdl.sched_barrier(0)
                        if const_expr(ks < n_mma):
                            _mma_ks(cur)
                        if const_expr(nxt is not None):
                            cur = nxt
                else:
                    for ks in range_constexpr(K_WS):
                        cur = _load_ks(addrs, ks)
                        rocdl.s_wait_dscnt(0)
                        if const_expr(ks == 0 and prefetch_kt is not None):
                            rocdl.sched_barrier(0)
                            issue(_slot(prefetch_kt), prefetch_kt)
                            rocdl.sched_barrier(0)
                        if const_expr(ks < n_mma):
                            _mma_ks(cur)
                rocdl.sched_barrier(0)
                # The caller runs this inside the barrier window when deferring.
                return cur if const_expr(defer_last) else None

            def _vk_split_fence(tail, outstanding):
                """Run the deferred sub-step inside the barrier's signal-to-wait window.

                Identical in shape to the NT/NN launcher's `_split_fence`: what it
                adds is the work moved *into* the window, not the split itself. The
                `sched_barrier(0)` fences are load-bearing -- without them the
                compiler sinks the matrix ops back out of the window and the answer
                stays correct, so the loss would be silent.
                """
                tdm_ops.tensor_wait(outstanding)
                rocdl.sched_barrier(0)
                rocdl.s_barrier_signal(_WG_BARRIER)
                rocdl.sched_barrier(0)
                _mma_ks(tail)
                rocdl.sched_barrier(0)
                rocdl.s_barrier_wait(_WG_BARRIER)

            # ---- CORRECTNESS: the TDM outstanding budget is counted PER WAVE, not
            # per workgroup.
            #
            # A and B are issued by *different* waves (`wave == 0` / `wave == 1`), so
            # each issuing wave has exactly one outstanding TDM op per k-tile -- not
            # two. Counting both jobs here makes tensor_wait a **no-op** and the
            # pipeline then reads LDS the DMA has not filled yet; the symptom is most
            # of the output NaN rather than a loss of precision.
            TDM_PW = 1

            def _fence(outstanding):
                tdm_ops.tensor_wait(outstanding)
                gpu.barrier()

            # Reduction length is per-expert, so K_TILES is a runtime value. Clamp it up
            # to the pipeline depth: a short expert then runs a few extra k-tiles whose
            # reads are all past m_len, hence zeroed, hence harmless -- much cheaper
            # than a divergent short-path and it keeps the barriers matched.
            if const_expr(k_tiles_const):
                # EXPERIMENT ONLY -- see the parameter comment. A Python int here makes
                # `range(n_steady)` below a trace-time loop, so the steady state unrolls
                # exactly as the NT/NN kernels' does.
                K_TILES = max(k_tiles_const, NB1)
            else:
                k_tiles_raw = (m_len + fx.Int32(tile_k - 1)) // fx.Int32(tile_k)
                K_TILES = (k_tiles_raw < fx.Int32(NB1)).select(fx.Int32(NB1), k_tiles_raw)

            if const_expr(not wpath_only):
                for i in range_constexpr(NB1):
                    if PREFETCH_OK and i == 0:
                        # Slot 0 was already filled by the PREVIOUS tile's
                        # epilogue, into buffer 1. Only the workgroup's very first tile has
                        # nobody to have done that for it. `tile_raw = pid + _t*PERSIST` and
                        # `pid < PERSIST`, so `tile_raw < PERSIST` is exactly `_t == 0`, and
                        # it is uniform across the workgroup -- a scalar branch, not divergence.
                        if tile_raw < fx.Int32(PERSIST):
                            issue(_slot(0), 0)
                    else:
                        issue(_slot(i), i)
                # A Python int in the `k_tiles_const` case, which is what lets the steady
                # loop go through `range_constexpr` below. The Python int is not by
                # itself enough: flydsl rewrites `for ... in range(...)` into a dynamic
                # loop whatever the bound's type, so `range_constexpr` is the only
                # spelling that unrolls.
                n_steady = (K_TILES - NB1) if const_expr(k_tiles_const) else (K_TILES - fx.Int32(NB1))

                # The `split_bar` prologue fence: it moves to the bottom of the body,
                # because the sub-step it has to hold in its window belongs to the k-tile
                # that just ran. The barrier count is unchanged -- the loop gains one here
                # and gives one back on the last k-tile, where the epilogue's entry fence
                # is already the next rendezvous.
                if const_expr(split_bar):
                    _fence(outstanding=TDM_PW * (num_buffers - 2))

                def _steady_ktile(kt):
                    buf = _bidx(_buf_ptr(_slot(kt)))
                    if const_expr(split_bar):
                        tail = compute_ktile(buf, kt + NB1, defer_last=True)
                        # Guards the next k-tile's stage, so it carries the outstanding
                        # count that k-tile's own fence used to carry.
                        _vk_split_fence(tail, TDM_PW * (num_buffers - 2))
                    else:
                        _fence(outstanding=TDM_PW * (num_buffers - 2))
                        compute_ktile(buf, kt + NB1)

                # Two spellings of one loop. `range_constexpr` is reachable only on the
                # experiment-only path, so the shipped paths still emit exactly what
                # they emitted before this branch existed.
                if const_expr(k_tiles_const):
                    for kt in range_constexpr(n_steady):
                        _steady_ktile(kt)
                else:
                    for kt in range(n_steady):
                        _steady_ktile(kt)

                for j in range_constexpr(NB1):
                    kt = n_steady + j
                    buf = _bidx(_buf_ptr(_slot(kt)))
                    if const_expr(split_bar):
                        tail = compute_ktile(buf, None, defer_last=True)
                        if const_expr(j < num_buffers - 2):
                            _vk_split_fence(tail, TDM_PW * (num_buffers - 3 - j))
                        else:
                            _mma_ks(tail)  # no successor stage to guard
                    else:
                        _fence(outstanding=TDM_PW * (num_buffers - 2 - j))
                        compute_ktile(buf, None)

            # ---- epilogue: out[g] tile, both output axes clamped ------------------
            accs = [c_frags[idx].load() for idx in range_constexpr(n_acc)]
            if const_expr(gather):
                # an empty expert has no row to re-read: its gathered LDS rows can be stale
                zero_acc = Vec.filled(8, 0.0, fx.Float32)
                accs = [Vec((m_len > fx.Int32(0)).select(acc, zero_acc)) for acc in accs]
            _fence(outstanding=0)
            if PREFETCH_OK:
                # ---- fetch the NEXT tile's first stage, here -------------------------
                # WHERE: immediately after the entry fence, which is `tensor_wait(0) +
                # barrier`, so every wave is past the drain and buffer 1 -- this tile's
                # slot-0 buffer under the constant +1 rotation -- is provably free, and
                # disjoint from the C staging below, which sits in buffer 0.
                #
                # WHY HERE AND NOT LATER: the final `tensor_wait(0)` at the bottom waits
                # on everything this wave has outstanding, and issued here the prefetch
                # has the whole epilogue to retire behind, so that wait costs about
                # nothing and the overlap is still obtained.
                #
                # ⛔ The alternative repair -- issue it late and relax the final wait to
                # `tensor_wait(TDM_PW)` -- is WRONG. The store carries
                # `num_warps=num_waves`, so every wave has one outstanding while only
                # waves 0 and 1 issue the prefetch; the others would satisfy the relaxed
                # wait immediately and cross the seam barrier with the store still in
                # flight. See the TDM_PW comment for what that class of miscount costs.
                #
                # The index is CLAMPED rather than guarded: the last tile of a workgroup
                # has no successor, and `tile_raw + PERSIST` would decode an expert past
                # the table and read out of bounds. Clamping re-fetches the last tile's
                # own first stage into a buffer nobody reads -- one iteration's wasted
                # bandwidth, and no branch.
                nxt_raw = tile_raw + fx.Int32(PERSIST)
                nxt_raw = (nxt_raw > fx.Int32(TOTAL_TILES - 1)).select(fx.Int32(TOTAL_TILES - 1), nxt_raw)
                nxt = _xcd_band_remap(nxt_raw, TOTAL_TILES, num_xcd, xcd_band)
                g2 = rocdl.readfirstlane(T.i32, nxt // fx.Int32(TILES_PER_EXPERT))
                l2 = nxt % fx.Int32(TILES_PER_EXPERT)
                bn2_idx, bk2_idx = _group_m_tile_decode(l2, N_BLOCKS_N, N_BLOCKS_K, group_m)
                e2 = g2 * fx.Int32(2)
                ms2 = rocdl.readfirstlane(T.i32, et[e2])
                ml2 = rocdl.readfirstlane(T.i32, et[e2 + fx.Int32(1)])
                bn2 = bn2_idx * tile_m
                bk2 = bk2_idx * tile_n
                bne2 = (bn2 > (i32_n - fx.Int32(tile_m))).select(i32_n - fx.Int32(tile_m), bn2)
                bke2 = (bk2 > (i32_k - fx.Int32(tile_n))).select(i32_k - fx.Int32(tile_n), bk2)
                issue(_slot(0), 0, dec=(ms2, ml2, bne2, bke2))
            for wm in range_constexpr(wmma_m_rep):
                row_rel = wmb + wm * WMMA_M + lane16
                for wn in range_constexpr(wmma_n_rep):
                    col_rel = wnb + wn * WMMA_N + kgrp * 8
                    h = accs[wm * wmma_n_rep + wn].to(elem_cls)
                    fx.ptr_store(h.bitcast(fx.Int8), base_ptr + (row_rel * tile_n + col_rel) * EB)
            gpu.barrier()
            c_off = fx.Int64(g_idx) * fx.Int64(C_GRP_ELEMS) + fx.Int64(blk_n) * ldc64 + fx.Int64(blk_k)
            gtC = _gv(gC_base, c_off, (tile_m, tile_n), (tile_n, 1))
            # Store side: dim-0 bounds addressing and dim-1 bounds a store's addressing
            # too (sentinel-verified), so the backed-off rows written above are dropped
            # here rather than reaching memory.
            atomC = fx.rocdl.make_tdm_atom(
                gtC,
                [n_valid, k_valid],
                strides=[ldc64, None],
                num_warps=num_waves,
                cache_modifier=tdm_store_cache,
                early_timeout=False,
            )
            fx.copy(atomC, _lv(fx.recast_iter(elem_cls, base_ptr), (tile_m, tile_n), (tile_n, 1)), gtC)
            tdm_ops.tensor_wait(0)
            # ⛔⛔ THE SEAM BARRIER. Without it the persistent path is NOT REENTRANT,
            # and the cost of that is a device reset:
            #
            #  1. The shipped body ends `tensor_store_from_lds` -> `s_wait_tensorcnt 0`
            #     -> `s_endpgm`, with no barrier between the wait and the end, so
            #     `s_endpgm` was doing double duty as the inter-tile barrier.
            #     Persistence replaces it with a back-edge and that barrier vanishes.
            #  2. The two regions are THE SAME BYTES: the epilogue's C staging starts at
            #     the arena base, and so do the next tile's buffer-0 A+B stages.
            #  3. `s_wait_tensorcnt` is PER-WAVE, A is issued by wave 0 and B by wave 1,
            #     and the output store is cooperative across all `num_waves`. So a wave
            #     that has retired its own share can start the next tile's
            #     `tensor_load_to_lds` into those bytes while another wave's store is
            #     still reading them.
            #
            # Scope: persistent path only. On the shipped path this would be a barrier
            # immediately before `s_endpgm`, i.e. pure cost.
            #
            # ⚠️ NECESSARY, NOT PROVEN SUFFICIENT: it closes the seam race, but the hang
            # seen under deep `persist` values correlates with workgroup retirements
            # rather than with seam count, which is a separate quantity and has to be
            # validated separately.
            if persist:
                gpu.barrier()

        if persist:
            for _t in range(TILES_PER_WG):
                _one_output_tile(pid + _t * fx.Int32(PERSIST))
        else:
            _one_output_tile(pid)

    kernel_grouped_variable_k(arg_c, arg_a, arg_b, arg_etab, arg_ridx, i32_n, i32_k).launch(
        grid=(PERSIST, 1, 1), block=(block, 1, 1), stream=stream
    )


# The LLVM-level gfx1250 expert scheduler, NOT triton's
# TRITON_HIP_USE_EXPERT_SCHEDULING pass. Upstream's dense gfx1250 GEMM turns it on
# and our correctness gates pass with it on; the analogous triton pass was found to
# silently miscompile bf16 grouped GEMM here, so treat this flag as validated by
# the gates rather than by construction, and re-check numerics if it is changed.
_launch_grouped_bf16_variable_k.compile_hints["llvm_options"] = {
    "amdgpu-expert-scheduling-mode": True,
}

# Slots 0-7 (five pointers, the stream, i32_n, i32_k) are the only runtime
# arguments; the constexpr tail starts here. See `_nt_launch` for why the tail is
# the right cache key.
_VK_CONSTEXPR0 = 8

_VK_COMPILED: dict = {}


def _vk_launch(args: tuple):
    """Same pre-bound CallState shortcut as ``_nt_launch``, for the variable-K kernel."""
    key = args[_VK_CONSTEXPR0:]
    fn = _VK_COMPILED.get(key)
    if fn is None:
        fn = flyc.compile(_launch_grouped_bf16_variable_k, *args)
        if fn is None:  # COMPILE_ONLY: nothing was executed, nothing to cache
            return None
        _VK_COMPILED[key] = fn
    return fn(*args)


@functools.lru_cache(maxsize=64)
def _pick_variable_k_config(OUT_M: int, OUT_N: int, avg_m: int):
    """(tile_m, tile_n, tile_k, m_warp, n_warp, num_buffers) for the variable-K kernel.

    ``tile_m`` tiles the output's first axis and ``tile_n`` its second while
    ``tile_k`` is the reduction over tokens, so unlike the forward ``avg_m`` is the
    **reduction** dimension here: the output is fixed by the weights, and shrinking
    the tile for a short token run buys no occupancy while multiplying the output
    traffic. The 2x2 warp grid the NT kernel uses carries over.

    There is no ``tile_n = 192`` branch here, unlike ``_pick_config``: this kernel's
    LDS rows feed the TDM atom's ``pad_interval``, which flydsl requires to be a
    power of two, so a 192-wide output tile aborts inside MLIR rather than raising.
    """
    # The read descriptors back their base off by up to a full tile (see the kernel
    # body), so a narrow output axis has nothing to back off into.
    if OUT_M < 256 or OUT_N < 256:
        return (64, 64, 64, 2, 2, 2)
    # At avg_m <= 128 the shipped tile runs exactly ONE k-tile per expert
    # (`K_TILES = max(ceil(avg_m/tile_k), num_buffers-1)` is 1 when tile_k == 128 ==
    # avg_m), so the double buffer never reaches steady state and the whole TDM
    # latency is exposed. Halving tile_k gives it two k-tiles to overlap, and 2x2
    # drops to one wave per SIMD, which both removes the co-resident LDS phase
    # collision `_vk_wave_grid` is about and gives each wave the full 128x128
    # accumulator (spill 0, checked at the compile gate).
    #
    # Pinned to avg_m == 128 because the effect REVERSES above it, and that is the
    # only batch in the delivery matrix where avg_m == tile_k.
    if avg_m <= 128:
        return (256, 256, 64, 2, 2, 2)
    # 2x2 instead of 4x2. ONLY `m_warp` moves -- tile area, `tile_k` and
    # `num_buffers` are untouched, so this is the warp grid on its own. 4x2 puts
    # eight waves in a workgroup, two per SIMD, and splits the accumulator; 2x2 is
    # four waves, one per SIMD, each holding the whole 128x128 accumulator (spill 0,
    # checked at the compile gate -- `(256,256,128,4,4,2)` spills and is refused).
    #
    # ⚠️ MECHANISM NOT ESTABLISHED, and in particular `_vk_wave_grid` is NOT voided
    # at four waves: forcing the other spelling on this very tile costs about 10%,
    # which rules out SIMD co-residency as the story, since at 2x2 there is no
    # co-resident pair to mis-pair. The explanation left standing is the one written
    # down for the NN path (see `_nn_native_launch`): `issue()` assigns the
    # global->LDS copies by RAW WAVE ID, so the relabelling re-pairs which loader
    # wave feeds which output sub-tile. That is a hypothesis, recorded as one.
    return (256, 256, 128, 2, 2, 2)


def grouped_gemm_bf16_variable_k_supported(OUT_M: int, OUT_N: int, avg_m: int = 0) -> bool:
    """Whether ``grouped_gemm_bf16_variable_k_flydsl_kernel`` can serve this shape.

    The non-throwing form of the public entry's assertions, for a dispatcher that
    has to pick a backend before committing. The read descriptors back their base
    off by up to a whole tile so the claimed window stays inside the row whatever
    the remainder, which is why there has to be a whole tile to back off into.
    """
    tile_m, tile_n = _pick_variable_k_config(OUT_M, OUT_N, avg_m)[:2]
    return OUT_M >= tile_m and OUT_N >= tile_n


# Back-compat alias for the bring-up scripts, which know this kernel as "wgrad_tn".
wgrad_tn_ok = grouped_gemm_bf16_variable_k_supported


def grouped_gemm_bf16_variable_k_flydsl_kernel(
    a: torch.Tensor,
    b: torch.Tensor,
    group_k_offsets: torch.Tensor,
    masked_k: torch.Tensor = None,
    out_dtype: torch.dtype = torch.bfloat16,
    # 0 = pick from the shape (recommended). The gfx950 entry defaults these to
    # 256; passing 256 explicitly here gives the same tile, but leaving them at 0
    # lets `_pick_variable_k_config` also choose tile_k / warps / buffer count,
    # which is where most of the measured win is.
    BLOCK_M: int = 0,
    BLOCK_N: int = 0,
    # Reduction depth in tokens, and the 4-buffer-style TDM pipeline depth.
    # gfx1250-only knobs; 0 = from the shape.
    BLOCK_K: int = 0,
    m_warp: int = 0,
    n_warp: int = 0,
    num_buffers: int = 0,
    # Super-tile width in output blocks: co-resident workgroups share a column
    # slice. 16, not the gfx950 default of 4; re-measured after `split_bar` was
    # wired onto the same resource and it still holds.
    group_m: int = 16,
    # XCD band-cyclic remap. Off by default -- the MI455X XCD count is unconfirmed,
    # unlike MI355X's 8. Correctness does not depend on the value.
    num_xcd: int = 1,
    xcd_band: int = 32,
    wmma_b2b: int = 0,
    frag_pipeline: int = 1,
    # 1 = run the k-tile's last sub-step of WMMA inside the workgroup barrier's
    # signal-to-wait window. Same knob as on the NT/NN entry.
    #
    # -1 = resolve from the shape (recommended). This kernel does NOT use the NT/NN
    # rule: WG/CU is 1 on every tile it returns, so that rule would turn the knob on
    # everywhere and it measured 4/6 negative when it did. What governs here is
    #   (a) executed steady-loop iterations >= 11, because the fence's cost is per
    #       workgroup and its benefit is per iteration, and
    #   (b) both operands' global row strides being a multiple of 256 B.
    # Both are measured; see `_vk_auto_split_bar`.
    split_bar: int = -1,
    # ⚠️ EXPERIMENT ONLY -- see the launcher parameter of the same name. Pins the
    # per-expert k-tile count to a compile-time constant so the steady-state loop
    # unrolls. WRONG ANSWERS, not slow ones, if the constant disagrees with the real
    # per-expert row counts -- hence the acknowledgement flag below.
    k_tiles_const: int = 0,
    # > 0 over-allocates `out` along its last axis and writes with that row stride,
    # leaving the returned view logically [G, OUT_M, OUT_N]. Separates the write-side
    # stride from `b`'s read-side stride, which are otherwise the same number.
    # Requires `out` to be absent or to already have the padded shape.
    out_ldc: int = 0,
    i_am_running_an_experiment: bool = False,
    # A (start, len) table from a previous call on the same `group_k_offsets` and
    # `masked_k`. Both variable-K calls in one MoE layer share it, so passing it
    # back removes one of the two builds. Provenance-checked -- a table from a
    # different call raises rather than silently computing the wrong thing.
    expert_table: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
    # gfx950 parameters accepted for signature compatibility; see below.
    trans_c: bool = False,
    cap_cu: int = 0,
    # A/B only; see the NT entry for the encoding and for why no value here ships.
    tdm_load_cache: int = 0,
    tdm_store_cache: int = 0,
    # Byte pad on both transposed LDS stage rows; multiple of 16. 0 = auto, which
    # is `_vk_lds_pad(tile_m, tile_n)` -- the conflict-free residue class. Pass an
    # explicit value only to measure the alternative; see the launcher.
    lds_pad: int = 0,
    # -1 = auto (`_vk_wave_grid`). 0/1 force one of the two warp-grid spellings.
    wave_grid: int = -1,
    # Experiment only -- see the launcher parameter of the same name.
    # 0 = shipped. Must divide TOTAL_TILES and leave >= 2 tiles per workgroup; both
    # are checked in the launcher, which is where TOTAL_TILES is known.
    persist: int = 0,
    # MEASUREMENT ONLY -- see the launcher's `wpath_only`. 1 emits the output write
    # path alone and RETURNS GARBAGE. Never set it on a correctness or delivery run.
    wpath_only: int = 0,
    # [M_total] int32: reduction row m of `a` (or `b`) is row `a_index[m]` of the tensor
    # passed, which may then hold any number of rows. At most one of the two.
    a_index: torch.Tensor | None = None,
    b_index: torch.Tensor | None = None,
) -> torch.Tensor:
    """Variable-K grouped wgrad: ``out[g] = a[rows_g].T @ b[rows_g]``.

    Drop-in for the gfx950 entry point of the same name. ``a`` is
    ``[M_total, OUT_M]`` and ``b`` is ``[M_total, OUT_N]`` with the groups
    concatenated along the reduction (token) dim, ``group_k_offsets`` ``[G+1]``
    gives each group's row start, and ``masked_k`` ``[G]`` the per-group **valid**
    row count so a padded tail is never read. Returns ``[G, OUT_M, OUT_N]``.

    In autograd terms with ``a = dout[M, N]`` and ``b = x[M, K]`` this is
    ``db[g] = dout[rows_g].T @ x[rows_g]`` of shape ``[G, N, K]`` -- the forward
    weight's layout. It is **end to end**: both operands are read token-major and
    the transpose happens inside LDS, so nothing is hidden outside the timed region.

    Differences from the gfx950 entry point
    ---------------------------------------
    * ``BLOCK_M`` / ``BLOCK_N`` default to 0 ("pick from the shape") rather than
      256. Pass 256 to reproduce the gfx950 default exactly.
    * ``trans_c=True`` is **not implemented** -- the epilogue would need a
      transposed TDM store -- and neither is ``cap_cu``, since the kernel launches
      one workgroup per output tile with no persistent loop to bound. Both raise
      rather than being silently ignored.
    * Requires ``OUT_M >= tile_m`` and ``OUT_N >= tile_n`` (see
      ``grouped_gemm_bf16_variable_k_supported``), because the read descriptors back
      their base off by a whole tile instead of bounding reads with an SRD
      ``num_records`` the way gfx950 does.
    * ``num_xcd`` defaults to 1, not 8.
    """
    assert a.dim() == 2 and b.dim() == 2, f"a must be [M,OUT_M] and b [M,OUT_N], got {a.dim()}/{b.dim()}"
    assert a.dtype == b.dtype and a.dtype in _F16, f"16-bit float operands only, got {a.dtype}/{b.dtype}"
    assert a.is_contiguous() and b.is_contiguous(), "operands must be contiguous"
    M_total, OUT_M = a.shape
    M2, OUT_N = b.shape
    assert a_index is None or b_index is None, "only one operand can be gathered"
    gather, gather_rows, row_index = 0, 0, None
    if a_index is not None:
        gather, gather_rows, row_index, M_total = 1, a.shape[0], a_index, M2
    elif b_index is not None:
        gather, gather_rows, row_index = 2, b.shape[0], b_index
    else:
        assert M2 == M_total, f"token counts differ: a M={M_total} vs b M={M2}"
    if row_index is not None:
        assert row_index.dtype == torch.int32 and row_index.is_contiguous(), "row index: contiguous int32"
        assert row_index.numel() >= M_total, f"row index covers {row_index.numel()} of {M_total} rows"
    G = group_k_offsets.numel() - 1
    _require_gfx1250(a.device)
    if trans_c:
        raise NotImplementedError(
            "trans_c=True is not implemented on gfx1250: the epilogue stores the tile through a "
            "TDM copy whose layout is fixed at [OUT_M, OUT_N]. Transpose the result, or use the "
            "gfx950 kernel."
        )
    if cap_cu:
        raise NotImplementedError(
            f"cap_cu={cap_cu} is not implemented on gfx1250: the kernel launches one workgroup per "
            f"output tile and has no persistent loop to bound. Pass cap_cu=0 (whole device)."
        )
    if masked_k is not None:
        assert masked_k.numel() == G, f"masked_k len {masked_k.numel()} != G {G}"
    if out_ldc and not i_am_running_an_experiment:
        raise ValueError(
            f"out_ldc={out_ldc} is an EXPERIMENT-ONLY override. Unlike the other knobs "
            f"here it can fault the device rather than return a wrong answer, because "
            f"the store's row stride and its per-expert span have to move together. "
            f"Pass i_am_running_an_experiment=True if that is genuinely what you want."
        )
    if k_tiles_const and not i_am_running_an_experiment:
        raise ValueError(
            f"k_tiles_const={k_tiles_const} is an EXPERIMENT-ONLY override that pins the "
            f"per-expert k-tile count at compile time. It computes WRONG ANSWERS, not slow "
            f"ones, on any input whose per-expert row counts disagree with it. Pass "
            f"i_am_running_an_experiment=True if that is genuinely what you want."
        )

    _avg_m = max(1, M_total // max(G, 1))
    cfg = _pick_variable_k_config(OUT_M, OUT_N, _avg_m)
    tile_m = BLOCK_M or cfg[0]
    tile_n = BLOCK_N or cfg[1]
    tile_k = BLOCK_K or cfg[2]
    m_warp = m_warp or cfg[3]
    n_warp = n_warp or cfg[4]
    num_buffers = num_buffers or cfg[5]
    # The read descriptors back their base off by up to a full tile to keep the
    # claimed window inside the row, so there has to be a full tile to back off
    # into. grouped_gemm_bf16_variable_k_supported() is the non-throwing form.
    assert OUT_M >= tile_m, f"OUT_M={OUT_M} must be >= tile_m={tile_m} for the descriptor back-off"
    assert OUT_N >= tile_n, f"OUT_N={OUT_N} must be >= tile_n={tile_n} for the descriptor back-off"
    # Resolved against the effective tile for the same reason `split_bar` is: an
    # explicit BLOCK_M/BLOCK_N override has to be padded for the row stride it
    # actually produces, not for the one the picker would have chosen.
    lds_pad = lds_pad or _vk_lds_pad(tile_m, tile_n, 2)
    if wave_grid == -1:
        wave_grid = _vk_wave_grid(m_warp, n_warp)
    if wave_grid not in (0, 1):
        raise ValueError(f"wave_grid must be 0 or 1 (or -1 for auto), got {wave_grid}")
    if split_bar == -1:
        # Resolved against the effective tile, so an explicit BLOCK_*/num_buffers
        # override is judged on what it actually produces rather than on the pick.
        split_bar = _vk_auto_split_bar(
            (tile_m, tile_n, tile_k, m_warp, n_warp, num_buffers), OUT_M, OUT_N, _avg_m
        )

    # The store walks `g*OUT_M*ldc + blk_n*ldc + blk_k`, so the allocation has to be
    # OUT_M*ldc per expert, not OUT_M*OUT_N. Checked rather than assumed: this is the
    # one place in this entry where a mismatch is an out-of-bounds device write.
    c_ldc = out_ldc or OUT_N
    if out_ldc:
        if out_ldc < OUT_N:
            raise ValueError(f"out_ldc={out_ldc} must be >= OUT_N={OUT_N}")
        if out is None:
            out = torch.empty(G, OUT_M, out_ldc, device=a.device, dtype=out_dtype)
        if tuple(out.shape) != (G, OUT_M, out_ldc) or not out.is_contiguous():
            raise ValueError(
                f"out_ldc={out_ldc} requires a contiguous out of shape "
                f"{(G, OUT_M, out_ldc)}, got shape {tuple(out.shape)} "
                f"contiguous={out.is_contiguous()}. Refusing: the store would write "
                f"past the allocation."
            )
        assert out.numel() >= G * OUT_M * out_ldc, "out too small for the padded stride"
    elif out is None:
        out = torch.empty(G, OUT_M, OUT_N, device=a.device, dtype=out_dtype)
    if expert_table is not None:
        check_prebuilt(expert_table, group_k_offsets, 0, G, 2 * G, "expert_table", masked_k=masked_k)
        etab = expert_table.reshape(-1)
    else:
        etab = build_expert_table(group_k_offsets, masked_k).reshape(-1)

    _vk_launch(
        (
            _as_ptr(out),
            _as_ptr(a),
            _as_ptr(b),
            _as_ptr(etab),
            _as_ptr(etab if row_index is None else row_index),
            torch.cuda.current_stream(),
            OUT_M,
            OUT_N,
            G,
            (OUT_M + tile_m - 1) // tile_m,
            (OUT_N + tile_n - 1) // tile_n,
            OUT_M * c_ldc,
            tile_m,
            tile_n,
            tile_k,
            m_warp,
            n_warp,
            num_buffers,
            1 if a.dtype == torch.float16 else 0,
            group_m,
            num_xcd,
            xcd_band,
            wmma_b2b,
            frag_pipeline,
            split_bar,
            k_tiles_const,
            out_ldc,
            tdm_load_cache,
            tdm_store_cache,
            lds_pad,
            wave_grid,
            persist,
            wpath_only,
            gather,
            gather_rows,
        )
    )
    # The caller asked for [G, OUT_M, OUT_N]; the padding is this function's business.
    return out[:, :, :OUT_N] if out_ldc else out
