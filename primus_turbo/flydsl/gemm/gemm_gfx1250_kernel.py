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
"""General dense GEMM for gfx1250 (bf16 / f16 / fp8 / mxfp8, NT / NN / TN).

``gemm_bf16_kernel.py`` next to this file cannot run here: it emits
``BufferCopyLDS128b``, its MMA atom is ``Mfma32x32x16`` (gfx1250 has no MFMA),
and it assumes wave64. This is the gfx1250 counterpart -- TDM descriptor atoms
for global->LDS, wave32 WMMA, a multi-buffer K loop.

Atom shapes are narrow, and were probed against the backend rather than
assumed. Only K is restricted; the operand types are not (see ``_KINDS``):

    bf16 / f16   WMMA        16x16x32 only, and A and B must share a dtype
    fp8          WMMA        16x16x64 or 16x16x128
    mxfp8        WMMAScale   16x16x128, block 32

The atom wants both operands as ``[row, K]`` with K contiguous. NT has that;
NN and TN do not, so their K-major operand is staged in its natural layout and
transposed in LDS (see ``transpose_stage``).

16-bit operands ride FlyDSL's tiled-MMA, which derives the per-lane register
mapping from the atom. 8-bit operands cannot -- allocating an 8-bit register
fragment leaves an ``ub.poison !fly.ptr<f8E4M3FN, register>`` after rmem SSA
promotion -- so those carry packed ``i32`` with an explicit lane mapping, as
the gfx1250 fp8/fp4 kernels in ROCm/aiter do.

Attribution: the gfx1250 TDM/WMMA pipeline idiom here follows the Apache-2.0
gfx1250 GEMM kernels contributed to ROCm/aiter (``gemm_a8w8_gfx1250.py``,
``mxfp4_preshuffle_gfx1250_tdm.py``, ``tdm_ops_gfx1250.py``, each
"SPDX-License-Identifier: Apache-2.0 / Copyright (c) 2026 FlyDSL Project
Contributors"). No code is copied verbatim from those or from the
MIT-licensed remainder of that repository.
"""

from __future__ import annotations

from typing import NamedTuple

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl._mlir import ir as _ir
from flydsl._mlir._mlir_libs._mlirDialectsFlyROCDL import MmaOpGFX1250_WMMAType
from flydsl._mlir.dialects import llvm as _llvm
from flydsl.expr import const_expr, range_constexpr, rocdl
from flydsl.expr.arith import _to_raw as _raw
from flydsl.expr.rocdl import tdm_ops
from flydsl.expr.typing import Constexpr
from flydsl.expr.typing import T as _T

__all__ = [
    "WAVE",
    "autotune",
    "can_run",
    "feasible_configs",
    "gemm_gfx1250",
    "launch_gemm_gfx1250",
    "supported_dtypes",
]

# gfx1250 is wave32. Every lane/wave derivation below depends on this; it is the
# single constant most often carried over wrongly from a wave64 (CDNA) kernel.
WAVE = 32

# kind -> (A class, B class, bytes per element, WMMA K, is-mx-scaled)
_KINDS = {
    "bf16": (fx.BFloat16, fx.BFloat16, 2, 32, False),
    "f16": (fx.Float16, fx.Float16, 2, 32, False),
    "fp8": (fx.Float8E4M3FN, fx.Float8E4M3FN, 1, 128, False),
    "fp8_e5m2": (fx.Float8E5M2, fx.Float8E5M2, 1, 128, False),
    "fp8_e4m3_e5m2": (fx.Float8E4M3FN, fx.Float8E5M2, 1, 128, False),
    "fp8_e5m2_e4m3": (fx.Float8E5M2, fx.Float8E4M3FN, 1, 128, False),
    "mxfp8": (fx.Float8E4M3FN, fx.Float8E4M3FN, 1, 128, True),
    "mxfp8_e5m2": (fx.Float8E5M2, fx.Float8E5M2, 1, 128, True),
    "mxfp8_e4m3_e5m2": (fx.Float8E4M3FN, fx.Float8E5M2, 1, 128, True),
    "mxfp8_e5m2_e4m3": (fx.Float8E5M2, fx.Float8E4M3FN, 1, 128, True),
}

# Hardware forms not wired up here: fp6/fp4 (sub-byte packing), i8/i4 (i32
# accumulator), and the bf16/f16-accumulator and 32x16x128 fp4 shapes.

# (A dtype, B dtype) -> (unscaled kind, MX-scaled kind). Pairs absent here are
# rejected, which is how the 16-bit operands get their "must match" rule: the
# hardware has no bf16 x f16 WMMA, so there is no row for it.
_E4M3, _E5M2 = torch.float8_e4m3fn, torch.float8_e5m2
_FROM_TORCH = {
    (torch.bfloat16, torch.bfloat16): ("bf16", None),
    (torch.float16, torch.float16): ("f16", None),
    (_E4M3, _E4M3): ("fp8", "mxfp8"),
    (_E5M2, _E5M2): ("fp8_e5m2", "mxfp8_e5m2"),
    (_E4M3, _E5M2): ("fp8_e4m3_e5m2", "mxfp8_e4m3_e5m2"),
    (_E5M2, _E4M3): ("fp8_e5m2_e4m3", "mxfp8_e5m2_e4m3"),
}


def supported_dtypes():
    """Operand kinds this kernel can run on gfx1250."""
    return tuple(_KINDS)


def _make_lds_load(bits):
    """Return ``load(lds_byte_base, byte_offset) -> Vector[i32]`` for LDS reads.

    Typed ``i32`` throughout: the 8-bit paths carry operand fragments as packed
    words, so the element type only reappears inside the MMA atom.
    """
    layout = fx.make_layout(bits // 32, 1)
    atom = fx.make_copy_atom(fx.UniversalCopy(bits), fx.Int32)
    ptr_ty = fx.PointerType.get(fx.Int32.ir_type, fx.AddressSpace.Shared, bits // 8)

    def load(base, byte_offset):
        ptr = fx.inttoptr(ptr_ty, fx.Int32(base) + fx.Int32(byte_offset))
        reg = fx.make_rmem_tensor(layout, fx.Int32)
        fx.copy_atom_call(atom, fx.Tensor(fx.make_view(ptr, layout)), reg)
        return reg.load()

    return load


# @flyc.jit specialises on ints, not strings. Derived from _KINDS rather than
# listed by hand, which is how six kinds once shipped unreachable.
_KIND_ID = {name: i for i, name in enumerate(_KINDS)}
_KIND_BY_ID = {v: (k, *_KINDS[k]) for k, v in _KIND_ID.items()}

# MX block size: one E8M0 scale per 32 contiguous K elements. The gfx1250
# V_WMMA_SCALE atom only accepts 32 (or 16); 128 is not a legal atom block size.
MX_BLOCK = 32

# out dtype -> the kernel's Constexpr code.
_OUT_KIND = {torch.bfloat16: 0, torch.float16: 1, torch.float32: 2}

# The WMMA atom is 16x16 in M/N for every operand type on this target.
WMMA_N_ATOM = 16

# Operand layouts, named for how A and B are stored (out = op(A) @ op(B)):
#   NT  A[M, K], B[N, K]  -- both K-contiguous. The forward pass of a Linear.
#   NN  A[M, K], B[K, N]  -- dgrad: dX = dY @ W.
#   TN  A[K, M], B[K, N]  -- wgrad: dW = X^T @ dY.
# The WMMA atom wants both operands as [row, K] with K contiguous, so an
# operand stored K-major has to be transposed on its way into LDS.
LAYOUT_NT, LAYOUT_NN, LAYOUT_TN = 0, 1, 2


@flyc.jit
def launch_gemm_gfx1250(
    arg_c: fx.Pointer,
    arg_a: fx.Pointer,
    arg_b: fx.Pointer,
    arg_scale_a: fx.Pointer,
    arg_scale_b: fx.Pointer,
    i32_m: fx.Int32,
    stream: fx.Stream,
    i32_n: fx.Int32,
    i32_k: fx.Int32,
    i32_lda: fx.Int32,
    i32_ldb: fx.Int32,
    i32_ldc: fx.Int32,
    tile_m: Constexpr[int],
    tile_n: Constexpr[int],
    tile_k: Constexpr[int],
    m_warp: Constexpr[int],
    n_warp: Constexpr[int],
    num_buffers: Constexpr[int],
    kind_id: Constexpr[int],
    out_kind: Constexpr[int],
    group_m: Constexpr[int] = 8,
    layout: Constexpr[int] = LAYOUT_NT,
    accumulate: Constexpr[int] = 0,
):
    """NT GEMM: ``C[M, N] = A[M, K] @ B[N, K]^T`` on gfx1250.

    ``K`` must be a multiple of ``tile_k`` and ``N`` of ``tile_n``; ``M`` is
    unrestricted (the TDM descriptors clamp the ragged tile, zero-filling on the
    load and dropping on the store).
    """
    kind, a_cls, b_cls, elem_bytes, WMMA_K, is_mx = _KIND_BY_ID[kind_id]
    WMMA_M = WMMA_N = 16
    # fp8 has both a 16x16x64 and a 16x16x128 atom; take the wider one when
    # tile_k allows, and fall back to 64 for a K that is not a multiple of 128
    # (DSV4 Flash's dense_down K=10944 is one). WMMAScale has only the 128 form.
    if not is_mx and elem_bytes == 1 and tile_k % 128:
        WMMA_K = 64
    is_16bit = elem_bytes == 2

    num_waves = m_warp * n_warp
    block = num_waves * WAVE
    assert tile_m % (m_warp * WMMA_M) == 0, "tile_m must cover m_warp x 16"
    assert tile_n % (n_warp * WMMA_N) == 0, "tile_n must cover n_warp x 16"
    assert tile_k % WMMA_K == 0, f"tile_k={tile_k} must be a multiple of {WMMA_K}"
    assert num_buffers >= 2, "need at least a double buffer"
    assert tile_m % num_waves == 0 and tile_n % num_waves == 0, "TDM splits the tile's outer dim across warps"

    k_iters = tile_k // WMMA_K
    # 0 bf16, 1 f16, 2 f32. The accumulator is f32 already, so an f32 output
    # skips a conversion and only costs a wider C staging tile.
    out_cls = (fx.BFloat16, fx.Float16, fx.Float32)[out_kind]
    out_bytes = 4 if out_kind == 2 else 2

    # `num_buffers` stages, each an A tile then a B tile, K-major. The 16 B of
    # row padding is load-bearing: unpadded, a row is an exact multiple of the
    # 128 B bank stride, so all 16 lanes of a fragment read hit one bank. TDM
    # applies it on the load path, so it costs nothing to compute.
    LDS_PAD = 16 // elem_bytes
    A_ROW = tile_k + LDS_PAD
    B_ROW = tile_k + LDS_PAD
    STAGE_A = tile_m * A_ROW * elem_bytes
    STAGE_B = tile_n * B_ROW * elem_bytes

    # An operand stored K-major lands in a scratch tile in its natural layout
    # and is transposed into the [row, K] tile the MMA path above expects, so
    # that path stays byte-identical across all three layouts.
    a_tr = layout == LAYOUT_TN
    b_tr = layout in (LAYOUT_NN, LAYOUT_TN)
    TR_A_ROW = tile_m + LDS_PAD  # scratch row = the contiguous (M or N) extent
    TR_B_ROW = tile_n + LDS_PAD
    SCRATCH_A = tile_k * TR_A_ROW * elem_bytes if a_tr else 0
    SCRATCH_B = tile_k * TR_B_ROW * elem_bytes if b_tr else 0
    # Per-stage MX scale planes: one E8M0 byte per MX_BLOCK K elements.
    SC_ROW = tile_k // MX_BLOCK
    STAGE_SA = tile_m * SC_ROW if is_mx else 0
    STAGE_SB = tile_n * SC_ROW if is_mx else 0
    SCRATCH_A_OFF = STAGE_A + STAGE_B + STAGE_SA + STAGE_SB
    SCRATCH_B_OFF = SCRATCH_A_OFF + SCRATCH_A
    PITCH = SCRATCH_B_OFF + SCRATCH_B
    # The epilogue restages C through the same arena, so it has to fit both.
    C_BYTES = tile_m * (tile_n + 8) * out_bytes
    # beta=1 stages the existing C beside the new one and adds them in LDS.
    ARENA = max(num_buffers * PITCH, C_BYTES * (2 if accumulate else 1))
    assert ARENA == lds_bytes(kind, layout, tile_m, tile_n, tile_k, num_buffers, accumulate, out_bytes), (
        "lds_bytes() drifted from the arena this kernel actually lays out"
    )
    assert ARENA <= LDS_LIMIT, f"LDS arena {ARENA} B exceeds the {LDS_LIMIT} B gfx1250 budget"

    n_ops = 4 if is_mx else 2  # TDM ops issued per stage

    kernel_name = (
        f"pt_gemm_gfx1250_{kind}_t{tile_m}x{tile_n}x{tile_k}_w{m_warp}x{n_warp}_nb{num_buffers}_g{group_m}"
    )

    @flyc.kernel(name=kernel_name, known_block_size=[block, 1, 1])
    def kernel_gemm(
        arg_c: fx.Pointer,
        arg_a: fx.Pointer,
        arg_b: fx.Pointer,
        arg_scale_a: fx.Pointer,
        arg_scale_b: fx.Pointer,
        i32_m: fx.Int32,
        i32_n: fx.Int32,
        i32_k: fx.Int32,
        i32_lda: fx.Int32,
        i32_ldb: fx.Int32,
        i32_ldc: fx.Int32,
    ):
        K_TILES = i32_k // tile_k
        tid = fx.Int32(fx.thread_idx.x)
        bid_x, bid_y, _ = fx.block_idx

        # Grouped tile ordering. A plain row-major walk over the tile grid
        # streams the whole of B through L2 for every row of A; visiting
        # `group_m` M-tiles before advancing N keeps that B column resident.
        n_tiles_m = (i32_m + (tile_m - 1)) // tile_m
        n_tiles_n = (i32_n + (tile_n - 1)) // tile_n
        pid = fx.Int32(bid_x)
        if const_expr(group_m > 1):
            per_group = group_m * n_tiles_n
            first_m = (pid // per_group) * group_m
            rem = n_tiles_m - first_m
            gsize = (rem < group_m).select(rem, fx.Int32(group_m))
            in_g = pid % per_group
            pm = first_m + in_g % gsize
            pn = in_g // gsize
        else:
            pm = pid % n_tiles_m
            pn = pid // n_tiles_m
        blk_m = pm * tile_m
        blk_n = pn * tile_n

        m_oob = i32_m - blk_m  # valid A / C rows in this tile
        n_oob = i32_n - blk_n  # valid B rows in this tile

        lda64 = fx.Int64(i32_lda)
        ldb64 = fx.Int64(i32_ldb)
        ldc64 = fx.Int64(i32_ldc)

        arena = fx.SharedAllocator(static=False)
        arena.allocate(ARENA)
        base_ptr = arena.base_ptr  # i8*

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
            # torch allocations are at least 256 B aligned; the kernel-arg
            # pointers arrive as i8* (alignment 1), which recast_iter refuses
            # to widen on its own.
            return fx.recast_iter(fx.PointerType.get(cls.ir_type, p.address_space, align), p)

        gA_base = _cast(a_cls, arg_a)
        gB_base = _cast(b_cls, arg_b)
        gC_base = _cast(out_cls, arg_c)

        # A K-major operand is fetched as a [tile_k, X] tile (its natural,
        # contiguous global layout, which is all TDM can do -- its innermost
        # stride is fixed at 1) and transposed in LDS afterwards. The ragged
        # extent moves to the inner dim with it.
        if const_expr(a_tr):
            gA = _gview(gA_base, fx.Int64(blk_m), (tile_k, tile_m), (i32_lda, 1))
            atomA = fx.rocdl.make_tdm_atom(
                gA,
                [None, m_oob],
                strides=[lda64, None],
                num_warps=num_waves,
                pad_interval=tile_m,
                pad_amount=LDS_PAD,
            )
        else:
            gA = _gview(gA_base, fx.Int64(blk_m) * lda64, (tile_m, tile_k), (i32_lda, 1))
            atomA = fx.rocdl.make_tdm_atom(
                gA,
                [m_oob, None],
                strides=[lda64, None],
                num_warps=num_waves,
                pad_interval=tile_k,
                pad_amount=LDS_PAD,
            )
        if const_expr(b_tr):
            gB = _gview(gB_base, fx.Int64(blk_n), (tile_k, tile_n), (i32_ldb, 1))
            atomB = fx.rocdl.make_tdm_atom(
                gB,
                [None, n_oob],
                strides=[ldb64, None],
                num_warps=num_waves,
                pad_interval=tile_n,
                pad_amount=LDS_PAD,
            )
        else:
            gB = _gview(gB_base, fx.Int64(blk_n) * ldb64, (tile_n, tile_k), (i32_ldb, 1))
            atomB = fx.rocdl.make_tdm_atom(
                gB,
                [n_oob, None],
                strides=[ldb64, None],
                num_warps=num_waves,
                pad_interval=tile_k,
                pad_amount=LDS_PAD,
            )

        gSA = gSB = atomSA = atomSB = None
        if const_expr(is_mx):
            sk = i32_k // MX_BLOCK
            gSA_base = _cast(fx.Uint8, arg_scale_a, 1)
            gSB_base = _cast(fx.Uint8, arg_scale_b, 1)
            gSA = _gview(gSA_base, fx.Int64(blk_m) * fx.Int64(sk), (tile_m, SC_ROW), (sk, 1))
            gSB = _gview(gSB_base, fx.Int64(blk_n) * fx.Int64(sk), (tile_n, SC_ROW), (sk, 1))
            atomSA = fx.rocdl.make_tdm_atom(
                gSA, [m_oob, None], strides=[fx.Int64(sk), None], num_warps=num_waves
            )
            atomSB = fx.rocdl.make_tdm_atom(
                gSB, [n_oob, None], strides=[fx.Int64(sk), None], num_warps=num_waves
            )

        def sA_of(s):
            return _view(_stage(s), a_cls, (tile_m, tile_k), (A_ROW, 1))

        def sB_of(s):
            return _view(fx.add_offset(_stage(s), STAGE_A), b_cls, (tile_n, tile_k), (B_ROW, 1))

        def sSA_of(s):
            return _view(
                fx.add_offset(_stage(s), STAGE_A + STAGE_B),
                fx.Uint8,
                (tile_m, SC_ROW),
                (SC_ROW, 1),
                1,
            )

        def sSB_of(s):
            return _view(
                fx.add_offset(_stage(s), STAGE_A + STAGE_B + STAGE_SA),
                fx.Uint8,
                (tile_n, SC_ROW),
                (SC_ROW, 1),
                1,
            )

        def scratchA_of(s):
            return _view(fx.add_offset(_stage(s), SCRATCH_A_OFF), a_cls, (tile_k, tile_m), (TR_A_ROW, 1))

        def scratchB_of(s):
            return _view(fx.add_offset(_stage(s), SCRATCH_B_OFF), b_cls, (tile_k, tile_n), (TR_B_ROW, 1))

        def issue(s, kt):
            """Start the TDM fetch of K-tile ``kt`` into LDS stage ``s``."""
            kt64 = fx.Int64(kt)
            # A K-major tile advances by whole rows, not by K elements.
            if const_expr(a_tr):
                fx.copy(atomA, gA, scratchA_of(s), imm_offset=kt64 * tile_k * lda64 * elem_bytes)
            else:
                fx.copy(atomA, gA, sA_of(s), imm_offset=kt64 * (tile_k * elem_bytes))
            if const_expr(b_tr):
                fx.copy(atomB, gB, scratchB_of(s), imm_offset=kt64 * tile_k * ldb64 * elem_bytes)
            else:
                fx.copy(atomB, gB, sB_of(s), imm_offset=kt64 * (tile_k * elem_bytes))
            if const_expr(is_mx):
                fx.copy(atomSA, gSA, sSA_of(s), imm_offset=kt64 * SC_ROW)
                fx.copy(atomSB, gSB, sSB_of(s), imm_offset=kt64 * SC_ROW)

        # ---- LDS transpose (NN / TN only) -------------------------------
        # `ds_load_tr16_b128` transposes an 8x8 block of 16-bit elements across
        # 8 lanes (mapping measured, not documented): lane L receives, for
        # j = 0..7, the element at offset L%8 from the address lane j supplied.
        # So pointing lane L at `&src[k0 + L%8][x0]` hands it the eight
        # consecutive K values of column `x0 + L%8`, one contiguous store.
        def transpose_stage(src_view, src_row, dst_view, dst_row, n_x, elem_cls):
            """Move a [tile_k, n_x] scratch tile into a [n_x, tile_k] compute tile.

            Both transpose reads hand a lane one column of the source as a run
            of consecutive K, which stores out as a single contiguous write.
            The two differ in how much they move and in how the supplying lanes
            are grouped, and both mappings were measured rather than inferred:

            16-bit, `ds_load_tr16_b128`: 8 lanes transpose an 8x8 block. Lane L
            receives, for j = 0..7, the element at offset L%8 from the address
            lane j supplied, so pointing lane L at &src[k0 + L%8][x0] hands it
            the eight K values of column x0 + L%8.

            8-bit, `ds_load_tr8_b64`: 16 lanes transpose a 16(k) x 8(x) block,
            and the supplying lane for output j of lane L is
            blk*16 + (j<4 ? 0 : 8) + ((L//8)%2)*4 + j%4 -- an interleave, not a
            straight run. Permuting the K each lane supplies by
            [0,1,2,3, 8,9,10,11, 4,5,6,7, 12,13,14,15] undoes it, leaving each
            lane with eight consecutive K again.
            """
            lds_ptr_ty = _ir.Type.parse("!llvm.ptr<3>")
            src_b = fx.Int32(fx.ptrtoint(fx.get_iter(src_view)))
            dst_i8 = fx.recast_iter(
                fx.PointerType.get(fx.Int8.ir_type, fx.get_iter(dst_view).address_space, 1),
                fx.get_iter(dst_view),
            )
            blocks_x = n_x // 8
            k_per_block = 8 if is_16bit else 16
            lanes_per_group = 8 if is_16bit else 16
            n_groups = block // lanes_per_group
            n_blocks = (tile_k // k_per_block) * blocks_x
            assert n_blocks % n_groups == 0, (
                f"transpose: {n_blocks} blocks do not divide over {n_groups} lane groups"
            )
            grp = tid // lanes_per_group
            i = tid % lanes_per_group
            r = i % 8
            if const_expr(is_16bit):
                k_supply, k_store = r, 0
            else:
                q = i // 4
                k_supply = i + 4 * ((q & 1) - (q >> 1))
                k_store = (i // 8) * 8
            for b in range_constexpr(n_blocks // n_groups):
                idx = grp + b * n_groups
                k0 = (idx // blocks_x) * k_per_block
                x0 = (idx % blocks_x) * 8
                addr = src_b + ((k0 + k_supply) * src_row + x0) * elem_bytes
                ptr = _llvm.inttoptr(lds_ptr_ty, _raw(addr))
                if const_expr(is_16bit):
                    v = rocdl.ds_load_tr16_b128(_T.vec(8, elem_cls.ir_type), ptr)
                else:
                    v = rocdl.ds_load_tr8_b64(_T.vec(2, _T.i32), ptr)
                fx.ptr_store(
                    fx.Vector(v).bitcast(fx.Int8),
                    dst_i8 + ((x0 + r) * dst_row + k0 + k_store) * elem_bytes,
                )

        def transpose_if_needed(s):
            if const_expr(a_tr):
                transpose_stage(scratchA_of(s), TR_A_ROW, sA_of(s), A_ROW, tile_m, a_cls)
            if const_expr(b_tr):
                transpose_stage(scratchB_of(s), TR_B_ROW, sB_of(s), B_ROW, tile_n, b_cls)
            if const_expr(a_tr or b_tr):
                fx.barrier()

        # ---- MMA -------------------------------------------------------
        # Slot 1 is the N side, slot 2 the M side -- hence `b_cls` first, and
        # `fx.gemm(atom, acc, <B frag>, <A frag>, acc)` below. Getting this
        # backwards is invisible until A and B have different dtypes.
        if const_expr(is_mx):
            mma_atom = fx.make_mma_atom(
                fx.rocdl.WMMAScale(
                    WMMA_M,
                    WMMA_N,
                    WMMA_K,
                    b_cls,
                    a_cls,
                    fx.Float32,
                    block_size=MX_BLOCK,
                )
            )
        else:
            # `fx.rocdl.WMMA` passes one element type for both operands, which
            # hides the mixed e4m3/e5m2 forms the hardware does support; the
            # underlying atom type takes the two slots separately.
            mma_atom = fx.make_mma_atom(
                MmaOpGFX1250_WMMAType.get(
                    WMMA_M,
                    WMMA_N,
                    WMMA_K,
                    b_cls.ir_type,
                    a_cls.ir_type,
                    fx.Float32.ir_type,
                    sign_a=False,
                    sign_b=False,
                    clamp=False,
                )
            )

        # C is staged through LDS so the global write leaves as one OOB-clamped
        # TDM store. Row padding keeps the register->LDS scatter off one bank.
        C_ROW = tile_n + 8

        warp_m = tile_m // m_warp
        warp_n = tile_n // n_warp
        m_reps = warp_m // WMMA_M
        n_reps = warp_n // WMMA_N

        if const_expr(is_16bit):
            # --- 16-bit path: FlyDSL's tiled-MMA derives every fragment layout
            # from the atom, so nothing here hard-codes the lane mapping. ---
            tiled_mma = fx.make_tiled_mma(mma_atom, fx.make_layout((m_warp, n_warp, 1), (n_warp, 1, 0)))
            thr_mma = tiled_mma.get_slice(tid)
            tC_shape = fx.Tensor(fx.make_view(gC_base, fx.make_layout((tile_m, tile_n), (C_ROW, 1))))
            frag_A = thr_mma.make_fragment_A(sA_of(0))
            frag_B = thr_mma.make_fragment_B(sB_of(0))
            frag_C = thr_mma.make_fragment_C(tC_shape)
            frag_C.fill(0.0)

            s2r = fx.make_copy_atom(fx.UniversalCopy128b(), a_cls)
            s2r_b = fx.make_copy_atom(fx.UniversalCopy128b(), b_cls)
            thr_a = fx.make_tiled_copy_A(s2r, tiled_mma).get_slice(tid)
            thr_b = fx.make_tiled_copy_B(s2r_b, tiled_mma).get_slice(tid)
            frag_A_rt = thr_a.retile(frag_A)
            frag_B_rt = thr_b.retile(frag_B)

            def compute(s):
                """Consume LDS stage ``s`` (``s`` may be a runtime value)."""
                pA = thr_a.partition_S(sA_of(s))
                pB = thr_b.partition_S(sB_of(s))
                for ki in range_constexpr(k_iters):
                    fx.copy(s2r, pA[None, None, ki], frag_A_rt[None, None, ki])
                    fx.copy(s2r_b, pB[None, None, ki], frag_B_rt[None, None, ki])
                    fx.gemm(
                        tiled_mma,
                        frag_C,
                        frag_A[None, None, ki],
                        frag_B[None, None, ki],
                        frag_C,
                    )

            def epilogue(sC):
                r2s = fx.make_copy_atom(
                    fx.UniversalCopy32b() if out_bytes == 4 else fx.UniversalCopy16b(),
                    out_cls,
                )
                thr_c = fx.make_tiled_copy_C(r2s, tiled_mma).get_slice(tid)
                frag_out = fx.make_fragment_like(frag_C, out_cls.ir_type)
                frag_out.store(frag_C.load().to(out_cls))
                fx.copy(r2s, thr_c.retile(frag_out), thr_c.partition_S(sC))

        else:
            # 8-bit path. `make_fragment_A/B` cannot allocate an 8-bit
            # register tensor (rmem SSA promotion leaves an `ub.poison`
            # !fly.ptr<f8E4M3FN, register>), so operands are packed i32 with an
            # explicit lane mapping: for 16x16x128 on wave32, lane L holds row
            # L%16 and the four 16-byte K-chunks at (L//16)*16 + 32*j.
            lds_b128 = _make_lds_load(128)
            lds_b32 = _make_lds_load(32)
            wave = rocdl.readfirstlane(_T.i32, tid // WAVE)
            lane = tid % WAVE
            lane16 = lane % 16
            kgrp = lane // 16
            wmb = (wave // n_warp) * warp_m
            wnb = (wave % n_warp) * warp_n
            n_acc = m_reps * n_reps
            A_ROW_B = A_ROW * elem_bytes
            B_ROW_B = B_ROW * elem_bytes
            sc32 = SC_ROW // 4  # i32 words per scale row

            c_frags = [fx.make_rmem_tensor(WMMA_N // 2, fx.Float32) for _ in range_constexpr(n_acc)]
            for cf in c_frags:
                cf.fill(0.0)

            # A lane covers WMMA_K bytes of its row as 16-byte chunks strided
            # by 32, the two half-waves interleaving; that is WMMA_K//32 chunks
            # and WMMA_K//8 dwords per lane.
            n_chunks = WMMA_K // 32
            frag_dwords = WMMA_K // 8

            def _operand(base, plane_off, row, row_bytes, ks):
                b0 = plane_off + row * row_bytes + ks * WMMA_K + kgrp * 16
                v = [fx.Vector(lds_b128(base, b0 + 32 * j)) for j in range_constexpr(n_chunks)]
                if const_expr(n_chunks == 4):
                    lo = v[0].shuffle(v[1], list(range(8)))
                    hi = v[2].shuffle(v[3], list(range(8)))
                    packed = lo.shuffle(hi, list(range(16)))
                else:
                    packed = v[0].shuffle(v[1], list(range(8)))
                reg = fx.make_rmem_tensor(frag_dwords, fx.Int32)
                reg.store(packed)
                return reg

            def compute(s):
                """Consume LDS stage ``s`` (``s`` may be a runtime value).

                The accumulator alone is m_reps*n_reps*8 VGPRs -- 512 for the
                fastest 256x256 w2x2 profile, which is the whole register file.
                So only the B fragments (reused by every M row) are held live;
                each A row is loaded immediately before the MMAs that consume
                it. Loading all of both up front costs another
                (m_reps+n_reps)*16 registers and spills.
                """
                base = fx.Int32(fx.ptrtoint(_stage(s)))
                sa_off = STAGE_A + STAGE_B
                sb_off = sa_off + STAGE_SA
                for ks in range_constexpr(k_iters):
                    bf = [
                        _operand(base, STAGE_A, wnb + ni * WMMA_N + lane16, B_ROW_B, ks)
                        for ni in range_constexpr(n_reps)
                    ]
                    if const_expr(is_mx):
                        # One i32 = the four E8M0 bytes covering this k-step.
                        sb = [
                            lds_b32(
                                base,
                                sb_off + ((wnb + ni * WMMA_N + lane16) * sc32 + ks) * 4,
                            )[0]
                            for ni in range_constexpr(n_reps)
                        ]
                    for mi in range_constexpr(m_reps):
                        af = _operand(base, 0, wmb + mi * WMMA_M + lane16, A_ROW_B, ks)
                        if const_expr(is_mx):
                            sa = lds_b32(
                                base,
                                sa_off + ((wmb + mi * WMMA_M + lane16) * sc32 + ks) * 4,
                            )[0]
                        for ni in range_constexpr(n_reps):
                            idx = mi * n_reps + ni
                            # The atom's first operand slot is the N side and
                            # the second the M side; the accumulator then comes
                            # out row = lane%16, cols = (lane//16)*8 + 0..7.
                            if const_expr(is_mx):
                                fx.gemm(
                                    mma_atom,
                                    c_frags[idx],
                                    bf[ni],
                                    af,
                                    c_frags[idx],
                                    scale_a=sb[ni],
                                    scale_b=sa,
                                )
                            else:
                                fx.gemm(mma_atom, c_frags[idx], bf[ni], af, c_frags[idx])

            def epilogue(sC):
                for mi in range_constexpr(m_reps):
                    row = wmb + mi * WMMA_M + lane16
                    for ni in range_constexpr(n_reps):
                        col = wnb + ni * WMMA_N + kgrp * (WMMA_N // 2)
                        h = c_frags[mi * n_reps + ni].load().to(out_cls)
                        fx.ptr_store(h.bitcast(fx.Int8), base_ptr + (row * C_ROW + col) * out_bytes)

        # ---- K pipeline -------------------------------------------------
        # `num_buffers - 1` tiles stay in flight; a stage is consumed once the
        # tensor counter says only the later ones are still outstanding.
        for j in range_constexpr(num_buffers - 1):
            issue(j, j)

        n_steady = K_TILES - (num_buffers - 1)
        for kt in range(n_steady):
            # One barrier per K-tile does both jobs: it publishes the stage that
            # just landed, and it guarantees every wave is done reading the
            # stage that the `issue` below is about to overwrite (the one
            # consumed last iteration). The fetch is started *before* the MMA so
            # the TDM transfer overlaps the math rather than following it.
            #
            # The transpose is *not* worth running a stage ahead of the MMA.
            # That was tried: it needs a third buffer, which most tiles cannot
            # fit, and where it did fit it measured 1.02x / 0.97x. The transpose
            # is bound by LDS bandwidth, not by sitting in step with the MMA, so
            # reordering buys nothing -- only moving less data would.
            tdm_ops.tensor_wait((num_buffers - 2) * n_ops)
            fx.barrier()
            nxt = kt + (num_buffers - 1)
            issue(nxt % num_buffers, nxt)
            transpose_if_needed(kt % num_buffers)
            compute(kt % num_buffers)

        for j in range_constexpr(num_buffers - 1):
            tdm_ops.tensor_wait((num_buffers - 2 - j) * n_ops)
            fx.barrier()
            transpose_if_needed((n_steady + j) % num_buffers)
            compute((n_steady + j) % num_buffers)

        # ---- epilogue: registers -> LDS -> global (TDM, OOB-clamped) ----
        fx.barrier()
        sC = _view(base_ptr, out_cls, (tile_m, C_ROW), (C_ROW, 1))
        gtC = _gview(
            gC_base,
            fx.Int64(blk_m) * ldc64 + fx.Int64(blk_n),
            (tile_m, C_ROW),
            (C_ROW, 1),
        )
        # Clamp to whichever is smaller: the columns this tile still has left
        # in C, or the tile width (which also drops the C_ROW padding). The B
        # load already zero-fills its out-of-range rows, so a ragged N tile
        # accumulates zeros there and the store just drops them.
        n_store = (n_oob < tile_n).select(n_oob, fx.Int32(tile_n))
        atomC = fx.rocdl.make_tdm_atom(gtC, [m_oob, n_store], strides=[ldc64, None], num_warps=num_waves)
        if const_expr(accumulate):
            # beta=1. Add in the LDS domain rather than in registers: the tile
            # lands beside the new one and the two are summed as a flat byte
            # range, so this needs no knowledge of the C fragment layout and is
            # the same code for 16- and 8-bit. Doing it in registers meant
            # reading sC back through the tiled copy, whose reverse direction
            # does not partition the way the store does.
            sC_prev = _view(fx.add_offset(base_ptr, C_BYTES), out_cls, (tile_m, C_ROW), (C_ROW, 1))
            fx.copy(atomC, gtC, sC_prev)
            tdm_ops.tensor_wait(0)
        epilogue(sC)
        fx.barrier()
        if const_expr(accumulate):
            ld128 = _make_lds_load(128)
            n_chunk = C_BYTES // 16
            assert n_chunk % block == 0, f"C tile {n_chunk} chunks do not divide over {block} threads"
            cb = fx.Int32(fx.ptrtoint(base_ptr))
            pb = cb + C_BYTES
            for c in range_constexpr(n_chunk // block):
                off = (tid + c * block) * 16
                cur = fx.Vector(ld128(cb, off)).bitcast(out_cls)
                prv = fx.Vector(ld128(pb, off)).bitcast(out_cls)
                fx.ptr_store(
                    (cur.to(fx.Float32) + prv.to(fx.Float32)).to(out_cls).bitcast(fx.Int8), base_ptr + off
                )
            fx.barrier()
        fx.copy(atomC, sC, gtC)
        tdm_ops.tensor_wait(0)

    kernel_gemm(
        arg_c,
        arg_a,
        arg_b,
        arg_scale_a,
        arg_scale_b,
        i32_m,
        i32_n,
        i32_k,
        i32_lda,
        i32_ldb,
        i32_ldc,
    ).launch(
        grid=(
            ((i32_m + (tile_m - 1)) // tile_m) * ((i32_n + (tile_n - 1)) // tile_n),
            1,
            1,
        ),
        block=(block, 1, 1),
        stream=stream,
    )


launch_gemm_gfx1250.compile_hints["llvm_options"] = {
    "amdgpu-expert-scheduling-mode": True,
}


# Swept on gfx1250. A 256x256 tile leaves most of the GPU idle at 1024^3, so
# small problems take a smaller one.
#   -> ((tile_m, tile_n, tile_k), m_warp, n_warp, num_buffers, group_m)
_SMALL_TILES = 4096 * 4096  # M*N below this counts as "not enough tiles"

_DEFAULT_CFG = {
    # 16-bit operands: K=32 atom, 2 bytes/element
    "16bit": {
        "small": ((128, 128, 128), 2, 2, 2, 8),
        "large": ((256, 256, 64), 4, 2, 2, 1),
    },
    # 8-bit operands: K=128 atom, 1 byte/element
    "8bit": {
        "small": ((128, 128, 128), 2, 2, 3, 8),
        "large": ((256, 256, 128), 2, 2, 2, 4),
    },
}


def _atom_k(kind: str) -> int:
    """Smallest K a single MMA atom covers, i.e. the floor for tile_k."""
    elem_bytes, is_mx = _KINDS[kind][2], _KINDS[kind][4]
    if is_mx:
        return 128  # WMMAScale has no narrower form
    return 64 if elem_bytes == 1 else 32


LDS_LIMIT = 160 * 1024  # per CU on gfx1250


def lds_bytes(kind, layout_id, tile_m, tile_n, tile_k, num_buffers, accumulate=False, out_bytes=2):
    """LDS the kernel will ask for. The launch asserts against this too, so a
    caller pre-filtering configs cannot disagree with what actually compiles."""
    elem_bytes, is_mx = _KINDS[kind][2], _KINDS[kind][4]
    pad = 16 // elem_bytes
    stage = (tile_m + tile_n) * (tile_k + pad) * elem_bytes
    if is_mx:
        stage += (tile_m + tile_n) * (tile_k // MX_BLOCK)
    if layout_id == LAYOUT_TN:  # A staged twice
        stage += tile_k * (tile_m + pad) * elem_bytes
    if layout_id in (LAYOUT_NN, LAYOUT_TN):  # B staged twice
        stage += tile_k * (tile_n + pad) * elem_bytes
    # The epilogue restages C through the same arena, twice over for beta=1.
    c_bytes = tile_m * (tile_n + 8) * out_bytes
    return max(num_buffers * stage, c_bytes * (2 if accumulate else 1))


def acc_vgprs(tile_m, tile_n, m_warp, n_warp):
    """Accumulator registers per lane. 512 is the file; the operand fragments
    have to fit alongside."""
    return (tile_m // m_warp // WMMA_N_ATOM) * (tile_n // n_warp // WMMA_N_ATOM) * 8


# NN and TN stage the K-major operand twice (natural tile plus its transpose),
# so they need a smaller tile to stay inside the 160 KiB LDS budget: at
# 128x128x64 with two buffers, NN takes 106 KiB and TN 140 KiB.
_TRANSPOSED_CFG = {
    2: ((128, 128, 64), 2, 2, 2, 8),  # 16-bit: NN 106 KiB, TN 140 KiB
    1: ((128, 128, 128), 2, 2, 2, 8),  # 8-bit: same LDS at twice the K
}


def default_config(kind: str, M: int, N: int, K: int | None = None, layout: int = LAYOUT_NT):
    """``((tile_m, tile_n, tile_k), m_warp, n_warp, num_buffers, group_m)``.

    With ``K`` given the tile is fitted to the problem: ``tile_k`` shrinks until
    it divides ``K`` (a K-tail is not handled, unlike a ragged M or N), and
    ``tile_n`` shrinks so a narrow output does not pay for a wide tile.
    """
    if layout != LAYOUT_NT:
        (tile_m, tile_n, tile_k), m_warp, n_warp, nb, gm = _TRANSPOSED_CFG[_KINDS[kind][2]]
    else:
        family = "16bit" if _KINDS[kind][2] == 2 else "8bit"
        bucket = "small" if M * N < _SMALL_TILES else "large"
        (tile_m, tile_n, tile_k), m_warp, n_warp, nb, gm = _DEFAULT_CFG[family][bucket]

    if K is not None:
        floor_k = _atom_k(kind)
        while tile_k > floor_k and (K % tile_k or K // tile_k < nb):
            tile_k //= 2
        # A tile narrower than one WMMA per wave is pointless.
        floor_n = n_warp * WMMA_N_ATOM
        while tile_n > floor_n and N <= tile_n // 2:
            tile_n //= 2
    return (tile_m, tile_n, tile_k), m_warp, n_warp, nb, gm


# tile config -> flyc.CompiledFunction (fast dispatch; see gemm_gfx1250).
# The launch takes 12 runtime arguments (pointers, sizes, strides, stream)
# before the Constexpr block starts.
_N_RUNTIME_ARGS = 12
_COMPILED: dict = {}


def _ptr(t: torch.Tensor):
    return flyc.from_c_void_p(fx.Uint8, t.data_ptr())


# Tile candidates worth trying. Kept deliberately small: the expensive part of
# tuning is the ~5-10 s FlyDSL compile per config, so the analytic filter below
# has to do the elimination, not the benchmark.
_TILE_MN = (128, 256)
_WARPS = ((2, 2), (4, 2), (2, 4), (4, 4))
_BUFFERS = (2, 3)


def feasible_configs(kind, layout, M, N, K):
    """Every launch config that will compile for this problem, best first.

    Nothing here touches the GPU. It mirrors the kernel's own constraints --
    LDS budget, accumulator registers, the WMMA K granularity, and the block
    sizes the LDS transpose needs -- so the survivors all build.

    Ordered by operand reuse (MMAs per fragment load) purely so the list is
    deterministic. Do not read it as a ranking: measured, the winners were
    often the wave-heavy configs this metric puts last -- TN's best was w4x4,
    which reuse scores lowest of all. Benchmark the whole list.
    """
    layout_id = _LAYOUTS[layout] if isinstance(layout, str) else layout
    elem_bytes = _KINDS[kind][2]
    atom_k = _atom_k(kind)
    a_tr = layout_id == LAYOUT_TN
    b_tr = layout_id in (LAYOUT_NN, LAYOUT_TN)
    # The LDS transpose moves 8x8 blocks of 16-bit or 16(k)x8(x) of 8-bit, one
    # per lane group, and every group has to get the same number of them.
    k_block, lanes_per_group = (8, 8) if elem_bytes == 2 else (16, 16)

    out = []
    for tile_m in _TILE_MN:
        for tile_n in _TILE_MN:
            if N % tile_n and tile_n > 128:
                continue  # a ragged N tile is fine, a mostly-empty one is not
            for tile_k in (atom_k, atom_k * 2, atom_k * 4):
                if K % tile_k:
                    continue
                for m_warp, n_warp in _WARPS:
                    waves = m_warp * n_warp
                    if tile_m % (m_warp * WMMA_N_ATOM) or tile_n % (n_warp * WMMA_N_ATOM):
                        continue
                    if tile_m % waves or tile_n % waves:
                        continue
                    if acc_vgprs(tile_m, tile_n, m_warp, n_warp) > 512:
                        continue
                    bad = False
                    for tr, extent in ((a_tr, tile_m), (b_tr, tile_n)):
                        if not tr:
                            continue
                        if extent % 8 or tile_k % k_block:
                            bad = True
                        elif ((tile_k // k_block) * (extent // 8)) % ((waves * WAVE) // lanes_per_group):
                            bad = True
                    if bad:
                        continue
                    for nb in _BUFFERS:
                        if K // tile_k < nb:
                            continue
                        if lds_bytes(kind, layout_id, tile_m, tile_n, tile_k, nb) > LDS_LIMIT:
                            continue
                        mr = tile_m // m_warp // WMMA_N_ATOM
                        nr = tile_n // n_warp // WMMA_N_ATOM
                        reuse = mr * nr / (mr + nr)
                        out.append(((tile_m, tile_n, tile_k), m_warp, n_warp, nb, reuse))
    out.sort(key=lambda c: -c[4])
    return [c[:4] for c in out]


# Tuned configs, keyed on (kind, layout, N, K). M is left out: it is the token
# count, varies per step, and moves throughput far less than N and K do.
_TUNED: dict = {}


def autotune(a, b, *, layout="nt", out_dtype=torch.bfloat16, limit=None, iters=12, **kw):
    """Benchmark the feasible configs for this problem and remember the winner.

    Returns ``((tile_m, tile_n, tile_k), m_warp, n_warp, num_buffers)``.

    Every feasible config is measured by default. That is 24-56 of them, which
    the analytic filter has already cut down from thousands, and the result is
    cached per (kind, layout, N, K). Sampling only the head of the list was
    tried and picked worse configs than the hand-written defaults in half the
    cases: no cheap ordering heuristic found so far predicts the winner.
    `limit` caps the count for a quick, worse answer.
    """
    cfg, reason = _resolve(a, b, layout, None, None, None, None, None, None, None, None, out_dtype)
    if reason is not None:
        raise ValueError(reason)
    key = (cfg.kind, layout, cfg.N, cfg.K)
    if key in _TUNED:
        return _TUNED[key]

    best, best_cfg = 0.0, None
    cands = feasible_configs(cfg.kind, layout, cfg.M, cfg.N, cfg.K)
    for tile, mw, nw, nb in cands[:limit] if limit else cands:
        try:

            def call(tile=tile, mw=mw, nw=nw, nb=nb):
                return gemm_gfx1250(
                    a,
                    b,
                    layout=layout,
                    tile=tile,
                    m_warp=mw,
                    n_warp=nw,
                    num_buffers=nb,
                    out_dtype=out_dtype,
                    **kw,
                )

            for _ in range(3):
                call()
            torch.cuda.synchronize()
            start, end = torch.cuda.Event(True), torch.cuda.Event(True)
            start.record()
            for _ in range(iters):
                call()
            end.record()
            torch.cuda.synchronize()
            flops = 2 * cfg.M * cfg.N * cfg.K * iters / (start.elapsed_time(end) / 1e3)
        except Exception:
            continue  # a config that will not compile is not a candidate
        if flops > best:
            best, best_cfg = flops, (tile, mw, nw, nb)
    if best_cfg is None:
        raise RuntimeError(f"no feasible config for {key}")
    _TUNED[key] = best_cfg
    return best_cfg


_LAYOUTS = {"nt": LAYOUT_NT, "nn": LAYOUT_NN, "tn": LAYOUT_TN}


class _Config(NamedTuple):
    M: int
    N: int
    K: int
    kind: str
    is_mx: bool
    layout_id: int
    tile_m: int
    tile_n: int
    tile_k: int
    m_warp: int
    n_warp: int
    num_buffers: int
    group_m: int


def _resolve(a, b, layout, scale_a, scale_b, kind, tile, m_warp, n_warp, num_buffers, group_m, out_dtype):
    """Work out the launch configuration, or say why this call cannot run.

    Returns ``(config, None)`` or ``(None, reason)``. Both :func:`gemm_gfx1250`
    and :func:`can_run` go through here, so a caller asking "will this work?"
    and the call itself can never disagree.
    """
    if a.dim() != 2 or b.dim() != 2:
        return None, f"A and B must be 2-D, got {tuple(a.shape)}, {tuple(b.shape)}"
    if layout not in _LAYOUTS:
        return None, f"layout must be one of {sorted(_LAYOUTS)}, got {layout!r}"
    if a.stride(1) != 1 or b.stride(1) != 1:
        return None, "A and B must be row-major (innermost stride 1)"
    layout_id = _LAYOUTS[layout]
    # out is always [M, N]; the layout says how A and B are stored.
    if layout_id == LAYOUT_NT:  # A[M,K] @ B[N,K]^T -- a Linear forward
        (M, K), (N, Kb) = a.shape, b.shape
    elif layout_id == LAYOUT_NN:  # A[M,K] @ B[K,N] -- dgrad
        (M, K), (Kb, N) = a.shape, b.shape
    else:  # LAYOUT_TN: A[K,M]^T @ B[K,N] -- wgrad
        (Kb, M), (K2, N) = a.shape, b.shape
        K, Kb = Kb, K2
    if K != Kb:
        return None, f"K mismatch between A and B: {K} vs {Kb} (layout={layout})"

    if kind is None:
        pair = _FROM_TORCH.get((a.dtype, b.dtype))
        if pair is None:
            return None, f"unsupported operand dtypes {a.dtype} x {b.dtype}"
        kind = pair[scale_a is not None]
        if kind is None:
            return None, f"{a.dtype} has no MX-scaled form"
    if kind not in _KINDS:
        return None, f"unknown kind {kind!r}; expected one of {supported_dtypes()}"
    is_mx = _KINDS[kind][4]
    if layout_id != LAYOUT_NT and _KINDS[kind][4]:
        return None, f"{layout} has no MX-scaled form (scales are not transposed), got {kind}"

    d_tile, d_mw, d_nw, d_nb, d_gm = default_config(kind, M, N, K, layout_id)
    tile_m, tile_n, tile_k = tile or d_tile
    m_warp = d_mw if m_warp is None else m_warp
    n_warp = d_nw if n_warp is None else n_warp
    num_buffers = d_nb if num_buffers is None else num_buffers
    group_m = d_gm if group_m is None else group_m
    # An f32 output doubles the C staging tile, which can be what decides the
    # arena; shrink until it fits rather than failing the launch assert.
    ob = 4 if out_dtype == torch.float32 else 2
    while (
        tile_n > n_warp * WMMA_N_ATOM
        and lds_bytes(kind, layout_id, tile_m, tile_n, tile_k, num_buffers, False, ob) > LDS_LIMIT
    ):
        tile_n //= 2
    if K % tile_k:
        return None, f"K={K} must be a multiple of tile_k={tile_k}"
    if K // tile_k < num_buffers:
        return None, (
            f"the {num_buffers}-buffer pipeline needs >= {num_buffers} K-tiles, "
            f"got {K // tile_k} (K={K}, tile_k={tile_k})"
        )
    if is_mx:
        if scale_a is None or scale_b is None:
            return None, "mxfp8 needs scale_a and scale_b"
        for nm, sc, rows in (("scale_a", scale_a, M), ("scale_b", scale_b, N)):
            if tuple(sc.shape) != (rows, K // MX_BLOCK):
                return None, f"{nm} must be {(rows, K // MX_BLOCK)} uint8 E8M0, got {tuple(sc.shape)}"
    if out_dtype not in _OUT_KIND:
        return None, f"out_dtype must be bf16, f16 or f32, got {out_dtype}"
    return (
        _Config(
            M,
            N,
            K,
            kind,
            is_mx,
            layout_id,
            tile_m,
            tile_n,
            tile_k,
            m_warp,
            n_warp,
            num_buffers,
            group_m,
        ),
        None,
    )


def can_run(
    a,
    b,
    *,
    layout="nt",
    scale_a=None,
    scale_b=None,
    kind=None,
    tile=None,
    m_warp=None,
    n_warp=None,
    num_buffers=None,
    group_m=None,
    out_dtype=torch.bfloat16,
) -> bool:
    """Whether :func:`gemm_gfx1250` can serve this call. Does not touch the GPU."""
    _, reason = _resolve(
        a, b, layout, scale_a, scale_b, kind, tile, m_warp, n_warp, num_buffers, group_m, out_dtype
    )
    return reason is None


def gemm_gfx1250(
    a: torch.Tensor,
    b: torch.Tensor,
    out: torch.Tensor | None = None,
    *,
    layout: str = "nt",
    scale_a: torch.Tensor | None = None,
    scale_b: torch.Tensor | None = None,
    kind: str | None = None,
    tile: tuple[int, int, int] | None = None,
    m_warp: int | None = None,
    n_warp: int | None = None,
    num_buffers: int | None = None,
    group_m: int | None = None,
    accumulate: bool = False,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """``C = A @ B^T`` on gfx1250. NT layout, all operands row-major.

    ``a`` is ``(M, K)``, ``b`` is ``(N, K)``, the result is ``(M, N)``.

    ``accumulate`` makes this ``out += A @ B`` rather than ``out = A @ B``
    (the beta=1 epilogue a fused gradient accumulation needs); ``out`` must
    then be supplied.

    ``kind`` selects the operand path (see :func:`supported_dtypes`); it is
    inferred from ``a.dtype`` when omitted, except that fp8 inputs carrying
    ``scale_a``/``scale_b`` are treated as ``"mxfp8"``. MX scales are
    ``(M, K // 32)`` / ``(N, K // 32)`` uint8 E8M0, row-major.
    """
    cfg, reason = _resolve(
        a, b, layout, scale_a, scale_b, kind, tile, m_warp, n_warp, num_buffers, group_m, out_dtype
    )
    if reason is not None:
        raise ValueError(reason)
    if accumulate and out is None:
        raise ValueError("accumulate=True needs an out tensor to add into")
    if out is None:
        out = torch.empty((cfg.M, cfg.N), dtype=out_dtype, device=a.device)
    # The scale pointers are unread unless the kind is MX-scaled; the kernel
    # still needs something valid there.
    sa = scale_a if cfg.is_mx else a
    sb = scale_b if cfg.is_mx else a
    args = (
        _ptr(out),
        _ptr(a),
        _ptr(b),
        _ptr(sa),
        _ptr(sb),
        cfg.M,
        fx.Stream(torch.cuda.current_stream(device=a.device)),
        cfg.N,
        cfg.K,
        a.stride(0),
        b.stride(0),
        out.stride(0),
        cfg.tile_m,
        cfg.tile_n,
        cfg.tile_k,
        cfg.m_warp,
        cfg.n_warp,
        cfg.num_buffers,
        _KIND_ID[cfg.kind],
        _OUT_KIND[out.dtype],
        cfg.group_m,
        cfg.layout_id,
        1 if accumulate else 0,
    )
    # Calling @flyc.jit directly re-binds the signature and rebuilds its cache
    # key on every launch: ~69 us of host time, which dominates any GEMM below
    # roughly 4096^3. flyc.compile hoists that to the first call (~5 us
    # dispatch). Everything from tile_m on is a Constexpr baked into the
    # compiled kernel, so the key is that slice of the call -- taken from args
    # rather than restated, since omitting one silently reused another
    # layout's kernel.
    key = args[_N_RUNTIME_ARGS:]
    compiled = _COMPILED.get(key)
    if compiled is None:
        # flyc.compile() also runs this first call.
        _COMPILED[key] = flyc.compile(launch_gemm_gfx1250, *args)
    else:
        compiled(*args)
    return out
