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

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""MXFP4/MXFP6/MXFP8 A x MXFP4 B preshuffle GEMM (gfx950): per-32 E8M0 scales folded into
a scaled 16x16x128 fx.gemm; A streams global->LDS via double-buffered async DMA. Layout
matches the host preshuffle (shuffle_weight_w4(.,16) + shuffle_scale_w4).

FLUX copy of FlyDSL v0.2.4's kernels/gemm/mxfp4_preshuffle.py, used for the A6W4 forward
(FLUX_FP4_FPROP_CAST=a6w4) through `gemm_a6w4`. Installed as
primus_turbo/flydsl/gemm/gemm_a6w4_kernel.py."""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import fly
from flydsl._mlir.dialects import llvm as _llvm
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.arith import _to_raw
from flydsl.expr.typing import (
    BFloat16,
    Constexpr,
    Float4E2M1FN,
    Float6E2M3FN,
    Float8E4M3FN,
    Float16,
    Float32,
    Int8,
    Int32,
    T,
)
from flydsl.expr.typing import Vector as Vec

from primus_turbo.flydsl.utils.gemm_helper import block_mn, run_compiled, xcd_remap_pid

_A_ELEM = {"fp4": Float4E2M1FN, "fp6": Float6E2M3FN, "fp8": Float8E4M3FN}


def _scale_mma_atoms(a_dtype, b_first=False):
    """16 scaled-MFMA atoms keyed (opsel of A, opsel of B); A elem is fp4/fp6/fp8, B always fp4.
    b_first: B is the MFMA's first operand, so the accumulators hold C^T fragments."""
    elem_a = _A_ELEM[a_dtype]
    if b_first:
        return {
            (osa, osb): fx.make_mma_atom(
                fx.rocdl.cdna4.MFMA_Scale(16, 16, 128, Float4E2M1FN, elem_a, opsel_a=osb, opsel_b=osa)
            )
            for osa in range(4)
            for osb in range(4)
        }
    return {
        (osa, osb): fx.make_mma_atom(
            fx.rocdl.cdna4.MFMA_Scale(16, 16, 128, elem_a, Float4E2M1FN, opsel_a=osa, opsel_b=osb)
        )
        for osa in range(4)
        for osb in range(4)
    }


def _permlane16_swap(a_i32, b_i32):
    """v_permlane16_swap_b32 (a's odd 16-lane rows <-> b's even rows) with wait states on both
    sides: inline asm is invisible to the hazard recognizer, so neither the VALU write of an
    input right before it nor a VALU read of a result right after it gets padded otherwise."""
    r = _llvm.inline_asm(
        ir.Type.parse("!llvm.struct<(i32, i32)>"),
        [_to_raw(a_i32), _to_raw(b_i32)],
        "s_nop 1\n\tv_permlane16_swap_b32 $0, $1\n\ts_nop 1",
        "=v,=v,0,1",
        has_side_effects=False,
    )
    i32 = ir.IntegerType.get_signless(32)
    return _llvm.extractvalue(i32, r, [0]), _llvm.extractvalue(i32, r, [1])


def _bq_view(arg_bq_addr, row_elems, KH4, k_tiles, k_halves):
    """Preshuffled B view for one N-row tile; index [l//16, l%16, kt, half, None] -> i32[4]."""
    col_base = rocdl.readfirstlane(T.i32, row_elems * KH4)
    i32_ptr_ty = fx.PointerType.get(T.i32, address_space=fx.AddressSpace.Global, alignment=16)
    off_i64 = fx.Int64(col_base)
    base_iter = fx.inttoptr(i32_ptr_ty, arg_bq_addr + off_i64 * fx.Int64(4))
    shape = (4, 16, k_tiles, k_halves, 4)
    view = fx.Tensor(fx.make_view(base_iter, fx.make_layout(shape, (64, 4, k_halves * 256, 256, 1))))
    return fx.rocdl.make_buffer_tensor(view, max_size=False)


@flyc.jit
def launch_gemm(
    arg_c: fx.Pointer,
    arg_a: fx.Pointer,
    arg_b: fx.Pointer,
    arg_scale_a: fx.Pointer,
    arg_scale_b: fx.Pointer,
    i32_m: fx.Int32,
    i32_n: fx.Int32,
    stream: fx.Stream,
    N: Constexpr[int],
    K: Constexpr[int],
    tile_m: Constexpr[int],
    tile_n: Constexpr[int],
    tile_k: Constexpr[int],
    a_dtype: Constexpr[str],
    out_dtype: Constexpr[str],
    batch: Constexpr[int],
    a_row_stride: Constexpr[int],
    a_batch_stride: Constexpr[int],
    sca_row_stride: Constexpr[int],
    sca_batch_stride: Constexpr[int],
    c_row_stride: Constexpr[int],
    c_batch_stride: Constexpr[int],
    waves_per_eu: Constexpr[int],
    group_m: Constexpr[int],
    num_xcd: Constexpr[int],
    pad16: Constexpr[int],
    b_prefetch: Constexpr[int],
    sa_lds: Constexpr[int],
    epi: Constexpr[int],
    arg_bias: fx.Pointer,
    has_bias: Constexpr[bool],
    a_stages: Constexpr[int],
    a_pipe: Constexpr[int],
):
    """Direct @flyc.jit launcher. Operands are fx.Pointer (pass ptr_arg(t): raw data_ptr, no
    per-launch DLPack). Compile once with flyc.compile, then cf(*runtime). a_dtype fp4/fp6/fp8
    A x preshuffled MXFP4 B, e8m0 scales. batch>1 = strided-batched over grid.z. The
    a_/sca_/c_ row/batch strides make A/scale_a/C addressing caller-controlled; each <0 keeps
    the contiguous [B,M,*] bmn default, all set = the [M,B,*] mbn layout. waves_per_eu<=0 = unset.
    group_m<=0 = native 2D grid (bid_x, bid_y); group_m>0 = flat 1D grid banded by block_mn.
    num_xcd>1 (group_m>0 only) remaps the flat id XCD-major first, so each XCD walks one
    contiguous run of the band order instead of every num_xcd-th tile.
    pad16==0 = dense A-tile LDS layout (byte-identical to the pre-R6 kernel); pad16>0 = D.1 LDS
    row padding (goal.md D.1), see A_LDS_B/CHUNK_ROWS below.
    b_prefetch selects the K loop: 0 = one K-tile per iteration, B loaded in the iteration that
    consumes it; 1/2 = two K-tiles per iteration with B loaded one half-iteration ahead (U2,
    see the loop below), 1 = next B/scales issued ahead of the A DMA and every half drained
    with vmcnt(0), 2 = issued behind the A DMA so each half drains only the DMA. 1/2 need
    tile_k=128 and an even K-tile count; otherwise the b_prefetch=0 loop is used.
    sa_lds=1 (U2 loop, 4 A-scale words per chunk, i.e. tile_m=128): wave w DMAs the chunk's
    A-scale word w into a double-buffered LDS slot and every wave reads all four back, instead
    of every wave loading the same four words; ignored where those conditions do not hold.
    epi selects the C store (same output bytes): 0 = one buffer_store_short per element;
    1 = row-merged, a permlane16_swap per n-fragment pair puts one row's 32 columns in each
    32-lane half, same store count, each store two 64 B row segments instead of four 32 B
    ones; 2 = MFMA operands swapped so a lane holds 4 consecutive columns of one row,
    cvt_pk + permlane16_swap pack 8 of them into one buffer_store_dwordx4 (16 stores per wave
    instead of 128), bias (has_bias) loaded in the prologue as one 8 B load per n-fragment.
    a_stages=3 (sa_lds U2 loop, chunk count a multiple of 3) = a three-slot A LDS ring: K-tile
    t's half issues the DMA of A(t+2), so every A tile has two halves to land instead of one;
    2 (or those conditions unmet) = the double-buffered U2 loop.
    a_pipe>0 (a_stages=3 ring only): each half reads m-row mi's A fragment a_pipe m-rows ahead
    of mi's MFMAs instead of reading all A fragments before the first MFMA; 0 = the latter.
    """
    BM, BN, BK = tile_m, tile_n, tile_k
    if const_expr(out_dtype == "bf16"):
        out_elem = BFloat16
    else:
        out_elem = Float16

    # Row sizes + read_a fragment layout (i32 units): fp6/fp8 read two b128 halves -> i32[A_NDW], fp4 one -> i32[4].
    if const_expr(a_dtype == "fp4"):  # 2 codes/byte
        a_row_bytes, A_ROW_B = K // 2, BK // 2
        A_GK_I32, A_KH_I32, A_HI_OFF, A_NDW = 4, 16, 0, 4
    else:
        a_row_bytes, A_ROW_B = K, BK
        if const_expr(a_dtype == "fp8"):
            A_GK_I32, A_KH_I32, A_HI_OFF, A_NDW = 4, 32, 16, 8
        else:  # fp6
            A_GK_I32, A_KH_I32, A_HI_OFF, A_NDW = 8, 32, 4, 6

    # D.1 per-shape LDS row padding (R6, goal.md D.1): the R3 swizzle leaves a residual 2x LDS
    # bank conflict (SQ_LDS_IDX_ACTIVE/inst 8.00 vs a 4-cycle floor); inserting pad16*16 bytes of
    # unused LDS after every CHUNK_ROWS=1024//A_ROW_B rows (the DMA buffer_load_dwordx4...lds
    # cooperative load's per-wave contiguous-byte unit -- the only independently placeable
    # padding granularity, since each lane's LDS destination within one coop load is hardware-
    # fixed at M0+lane*16) eliminates it (measured: SQ_LDS_BANK_CONFLICT 66,060,288 -> 0 at
    # pad16=8, bitwise-identical output -- padding only moves WHERE A lives in LDS, not what data
    # flows). A_REAL_B (unpadded data volume) drives n_coop below and must NOT include padding
    # bytes, or the DMA would walk past the real A row. pad16==0 reduces this to the exact
    # pre-R6 dense formula (BM % CHUNK_ROWS == 0 for every tile_m used here, so the two are
    # numerically identical, not just approximately).
    CHUNK_ROWS = 1024 // A_ROW_B
    PAD_B = pad16 * 16
    CHUNK_B = CHUNK_ROWS * A_ROW_B + PAD_B
    CHUNK_I32 = CHUNK_B // 4
    assert BM % CHUNK_ROWS == 0, "D.1 padding requires CHUNK_ROWS | tile_m"
    assert 16 % CHUNK_ROWS == 0, "D.1 strength reduction requires CHUNK_ROWS | 16 (one mi-step)"
    A_REAL_B = BM * A_ROW_B  # real (unpadded) A-tile bytes -- drives n_coop, independent of pad16
    A_LDS_B = (BM // CHUNK_ROWS) * CHUNK_B  # padded LDS buffer bytes per A-tile (== A_REAL_B at pad16=0)
    A_ROW_I32 = A_ROW_B // 4
    # XOR16 swizzle (dma_a_to_lds write + read_a's lo/hi offsets both branch on this) is dtype-
    # agnostic: k_blk16 = A_ROW_B//16 and the (lo_blk ^ row%k_blk16) pair hold for fp6 unchanged.
    # fp6 was excluded here even though the formula is already correct for it (R3, goal.md B.1).
    swz_lds = True
    k_blk16 = A_ROW_B // 16
    if const_expr(pad16 > 0):
        assert k_blk16 == CHUNK_ROWS, "D.1 swizzle-key reuse requires k_blk16 == CHUNK_ROWS"
    K_HALF = K // 2
    KH4 = K_HALF // 4
    K_TILES = K // BK
    GY = N // BN  # N-tile count for the group_m raster band (R4); N and BN are both Constexpr
    k_halves = BK // 128  # 16x16x128 MFMA k-steps per K-tile
    # e8m0 scales are 256-K granular, B 128-K: tiles_per_chunk K-tiles share a word (hi/lo 16b = 128-K half).
    tiles_per_chunk = 256 // BK  # 1 for tile_k=256, 2 for tile_k=128
    m_chunks = BM // 16
    num_acc_n = (BN // 4) // 16  # 16-col n-subblocks per wave
    _scale_chunk_dw = (K // 32 // 4 // 2) * 64  # e8m0 stride (dwords), per shuffle_scale_w4
    _scale_k0_dw = 64
    n_coop = A_REAL_B // 256 // 16  # 16B cooperative loads per thread (unpadded data volume)
    n_pairs = max(1, num_acc_n // 2)
    m_pairs = max(1, m_chunks // 2)
    # U2 K loop (b_prefetch 1/2, described at the loop below): two K-tiles per 256-K scale chunk.
    u2 = b_prefetch > 0 and tiles_per_chunk == 2 and K_TILES % 2 == 0
    # sa_lds maps A-scale word w of a chunk to wave w, so it needs m_pairs == 4 waves.
    sa_lds_on = sa_lds > 0 and u2 and m_pairs == 4
    SC_LDS_B = 2 * m_pairs * 256  # two chunk slots x m_pairs words x 64 lanes x 4 B
    # a_stages=3 (sa_lds U2 loop, chunk count a multiple of 3): three A LDS slots, each its own
    # LDS symbol; the loop runs three chunks per iteration so every half addresses its slots
    # through a fixed symbol (see the loop below).
    a3 = a_stages == 3 and sa_lds_on and (K_TILES // 2) % 3 == 0

    # Scheduler counts per loop iter: MFMAs, A LDS reads/thread (fp6/fp8 2 per (mi,kh)), gmem loads.
    sched_mfma_total = k_halves * m_chunks * num_acc_n
    if const_expr(a_dtype == "fp4"):
        a_ds_per = 1
    else:
        a_ds_per = 2
    sched_num_ds_load = m_chunks * k_halves * a_ds_per
    sched_num_gmem = n_coop + num_acc_n * k_halves + m_pairs + n_pairs

    if const_expr(a3):

        @fx.struct
        class SharedA:
            a0: fx.Array[Int8, A_LDS_B, 16]
            a1: fx.Array[Int8, A_LDS_B, 16]
            a2: fx.Array[Int8, A_LDS_B, 16]
            sc: fx.Array[Int8, SC_LDS_B, 16]

    elif const_expr(sa_lds_on):

        @fx.struct
        class SharedA:
            a0: fx.Array[Int8, A_LDS_B, 16]
            a1: fx.Array[Int8, A_LDS_B, 16]
            sc: fx.Array[Int8, SC_LDS_B, 16]

    else:

        @fx.struct
        class SharedA:
            a0: fx.Array[Int8, A_LDS_B, 16]
            a1: fx.Array[Int8, A_LDS_B, 16]

    @flyc.kernel
    def kernel_gemm(
        arg_c: fx.Int64,
        arg_a: fx.Int64,
        arg_b: fx.Int64,
        arg_scale_a: fx.Int64,
        arg_scale_b: fx.Int64,
        i32_m: fx.Int32,
        i32_n: fx.Int32,
        arg_bias: fx.Int64,
    ):
        scale_atoms = _scale_mma_atoms(a_dtype, b_first=epi == 2)

        tid = fx.Int32(fx.thread_idx.x)
        bid_x, bid_y, bid_z = fx.block_idx
        wave = rocdl.readfirstlane(T.i32, tid // 64)
        lane = tid % 64
        lane_div_16 = lane // 16
        lane_mod_16 = lane % 16
        # Per-shape group_m L2 band raster (R4, goal.md SS C): group_m==0 keeps today's native
        # 2D grid, (pid_m, pid_n) = (bid_x, bid_y), byte-for-byte unchanged. group_m>0 decodes a
        # flat 1D grid id (launched as (gx*gy, 1, batch) below) through block_mn's 1D GROUP_M
        # band (GN=0) -- the same bijection gemm_fp8_kernel.py's dense NT/NN GEMMs already use
        # for L2 reuse. block_mn's arith.select(rem_m < GM, rem_m, GM) is the gsize =
        # min(gx-first_m, gm) clamp that keeps the M=M_ALIGN tail band correct for any group_m.
        # num_xcd>1: the dispatcher deals workgroups round-robin over the XCDs (each with its own
        # L2), so xcd_remap_pid (a bijection over the gx*gy launched ids) hands each XCD one
        # contiguous run of the band order before block_mn decodes it.
        if const_expr(group_m > 0):
            num_pid_m = (i32_m + (BM - 1)) // BM
            if const_expr(num_xcd > 1):
                pid_flat = xcd_remap_pid(bid_x, num_pid_m * GY, num_xcd)
            else:
                pid_flat = bid_x
            pid_m, pid_n = block_mn(pid_flat, num_pid_m, GY, group_m, 0)
        else:
            pid_m, pid_n = bid_x, bid_y
        bx_m = pid_m * BM
        by_n = pid_n * BN

        # Strided-batched: shift each base to batch bid_z (A/scale_a via explicit strides or
        # the contiguous default; B/scale_b stay batch-contiguous). batch==1 emits no batch math.
        if const_expr(batch > 1):
            a_rstride = fx.Int32(a_row_bytes if a_row_stride < 0 else a_row_stride)
            sca_rstride = fx.Int32(_scale_chunk_dw if sca_row_stride < 0 else sca_row_stride)
            bz = fx.Int64(bid_z)
            if const_expr(a_batch_stride < 0):
                arg_a = arg_a + bz * (fx.Int64(i32_m) * fx.Int64(a_row_bytes))
            else:
                arg_a = arg_a + bz * fx.Int64(a_batch_stride)
            arg_b = arg_b + bz * fx.Int64(N * (K // 2))
            if const_expr(sca_batch_stride < 0):
                sc_bstride = fx.Int64((i32_m + 31) // 32) * fx.Int64(_scale_chunk_dw) * fx.Int64(4)
                arg_scale_a = arg_scale_a + bz * sc_bstride
            else:
                arg_scale_a = arg_scale_a + bz * fx.Int64(sca_batch_stride)
            arg_scale_b = arg_scale_b + bz * fx.Int64((N // 32) * _scale_chunk_dw * 4)
        else:
            a_rstride = fx.Int32(a_row_bytes)
            sca_rstride = fx.Int32(_scale_chunk_dw)

        # A source bound, computed once: the last valid M row (ragged M OOB -> 0). R8 (goal.md
        # A.3 WAR-drain fix): the buffer descriptor itself is now rebuilt per-kt inside
        # dma_a_to_lds below (base pointer shifted by kt*A_ROW_B bytes, num_records_bytes shrunk
        # by the identical amount so base+num_records -- the absolute end of the checked window
        # -- never moves, hence no OOB change), so only _i8g/a_nrec (the kt-independent pieces)
        # are needed here; the old single module-level a_flat/a_flat_div (built once from
        # arg_a with no kt term, and relying on a per-lane `+ kt*A_ROW_B` inside the voffset to
        # reach the right K-tile) is gone.
        _i8g = fx.PointerType.get(T.i8, address_space=fx.AddressSpace.Global, alignment=16)
        if const_expr(batch > 1 and a_row_stride >= 0):
            a_nrec = fx.Int64(i32_m - fx.Int32(1)) * fx.Int64(a_rstride) + fx.Int64(a_row_bytes)
        else:
            a_nrec = fx.Int64(i32_m) * fx.Int64(a_row_bytes)
        lds = fx.SharedAllocator().allocate(SharedA).peek()
        # A-LDS modeled as i32 (16B = 4 i32): fx.copy is dtype-agnostic, only the MMA cares.
        sA0_i32 = fx.recast_iter(Int32, lds.a0.ptr)
        lds_db = fx.Int32(fx.ptrtoint(lds.a1.ptr)) - fx.Int32(
            fx.ptrtoint(lds.a0.ptr)
        )  # ping/pong byte stride
        lds_db_i32 = lds_db // 4
        lds_copy = fx.make_copy_atom(fx.UniversalCopy128b(), Int32)
        lds_copy64 = fx.make_copy_atom(fx.UniversalCopy64b(), Int32)
        dma_atom = fx.make_copy_atom(fx.rocdl.BufferCopyLDS128b(), 128)
        _i8s = fx.PointerType.get(Int8.ir_type, fx.AddressSpace.Shared, 512)
        sA0_i8 = fx.recast_iter(_i8s, lds.a0.ptr)
        if const_expr(a3):
            # Each slot keeps its own symbol, so an access through it carries that symbol's alias
            # scope: the waitcnt pass then lets the next slot's DMA stay in flight across a slot's
            # ds_reads instead of draining it (which it does for any DMA into the same symbol).
            a_slots = (lds.a0.ptr, lds.a1.ptr, lds.a2.ptr)
            slot_i32 = [fx.recast_iter(Int32, p) for p in a_slots]
            slot_i8 = [fx.recast_iter(_i8s, p) for p in a_slots]

        def _iter_of(parity):  # parity in {0,1} (runtime) -> i32 LDS iterator
            return fx.add_offset(sA0_i32, parity * lds_db_i32)

        def _lds_view(base_iter, off_i32):
            return fx.make_view(fx.add_offset(base_iter, off_i32), fx.make_layout(4, 1))

        # Async A: gmem->LDS DMA (buffer_load_lds); issued after B/scale loads to overlap the MFMAs.
        # D.1 (pad16>0): the per-wave LDS destination base/stride use the padded CHUNK_B instead
        # of the dense 1024B (64 lanes * 16B); `wave` is a readfirstlane'd per-wave scalar, so
        # this multiply runs on SALU, parallel to the VALU work -- source (gmem) addressing below
        # is completely unaffected by padding, since only the LDS destination placement changes.
        #
        # R8 (goal.md A.3 WAR-drain fix): the K-tile index used to land here as a per-lane VALU
        # add (`gmem_byte = ... + kt*A_ROW_B + ...`), so the per-lane voffset value was a *fresh*
        # SSA value every loop iteration even though `(bx_m+row)*a_rstride+col` on its own does
        # not depend on kt -- LLVM cannot treat "invariant + loop-varying-scalar" as invariant,
        # so it recomputed (and re-lived) those voffset VGPRs every iteration, and the register
        # allocator was then free to recycle the same physical VGPRs for the scale-load
        # destinations later in the same iteration -- the exact WAR hazard goal.md A.3 measured
        # (4 `s_waitcnt vmcnt(N)` drains between the A-DMA issue and the first MFMA, draining
        # 9-of-14 in-flight VMEM ops just to make those registers safe to reuse).
        #
        # Fix: move kt's contribution into the buffer descriptor itself. The SRD's base pointer
        # is shifted forward by kt*A_ROW_B bytes and num_records_bytes is shrunk by the identical
        # amount, so base+num_records (the absolute end of the hardware-checked window) is
        # byte-for-byte unchanged -- the exact same global bytes are reachable, with the exact
        # same OOB behavior, just addressed via a smaller per-lane offset from a later base. The
        # per-lane `gmem_byte` below is then `(bx_m+row)*a_rstride+col` with NO kt term at all: a
        # true loop invariant, computed from values that never change across K-tiles. The SRD
        # rebuild (inttoptr + make_view + make_buffer_tensor) is a handful of SALU ops -- one
        # 64-bit pointer add, one 64-bit subtract, and the fixed buffer-descriptor-pack sequence
        # -- done ONCE per call on scalar (uniform) registers, not per-lane and not per n_coop
        # fragment, versus the four wide per-lane VALU address chains (one per `n_coop` fragment)
        # this is meant to let the backend hoist out of the K loop entirely.
        def dma_a_to_lds(kt, parity, slot_i8=None):
            # slot_i8 (a3): the destination slot's own LDS symbol; otherwise buffer `parity` off a0.
            if const_expr(slot_i8 is not None):
                wave_b = CHUNK_B if pad16 > 0 else 64 * 16
                lds_ptr = fx.add_offset(slot_i8, rocdl.readfirstlane(T.i32, wave * wave_b))
            elif const_expr(pad16 > 0):
                base_off = rocdl.readfirstlane(T.i32, parity * lds_db + wave * CHUNK_B)
                lds_ptr = fx.add_offset(sA0_i8, base_off)
            else:
                base_off = rocdl.readfirstlane(T.i32, parity * lds_db + wave * (64 * 16))
                lds_ptr = fx.add_offset(sA0_i8, base_off)
            kt_byte = fx.Int64(kt) * fx.Int64(A_ROW_B)
            a_flat_kt = fx.rocdl.make_buffer_tensor(
                fx.Tensor(
                    fx.make_view(
                        fx.inttoptr(_i8g, arg_a + kt_byte),
                        fx.make_layout(65536 * a_row_bytes, 1),
                    )
                ),
                max_size=False,
                num_records_bytes=a_nrec - kt_byte,
            )
            a_flat_div_kt = fx.logical_divide(a_flat_kt, fx.make_layout(1, 1))
            for i in range_constexpr(n_coop):
                if const_expr(i > 0):
                    if const_expr(pad16 > 0):
                        lds_ptr = fx.add_offset(lds_ptr, fx.Int32(4 * CHUNK_B))
                    else:
                        lds_ptr = fx.add_offset(lds_ptr, fx.Int32(256 * 16))
                lin = (i * 256 + tid) * 16
                row = lin // A_ROW_B
                col = lin % A_ROW_B
                if const_expr(swz_lds):
                    col = col ^ ((row % k_blk16) * 16)
                gmem_byte = (bx_m + row) * a_rstride + col
                dst = fx.make_view(lds_ptr, fx.make_layout(1, 1))
                src = fx.slice(a_flat_div_kt, (None, gmem_byte))
                fx.copy(dma_atom, src, dst)

        def _read16(base_iter, off_i32):
            # ds_read_b128 straight into an i32[4] register fragment.
            t = fx.make_rmem_tensor(4, Int32)
            fx.copy(lds_copy, _lds_view(base_iter, off_i32), t)
            return t

        def _read8(base_iter, off_i32):
            # ds_read_b64 into an i32[2] register fragment.
            t = fx.make_rmem_tensor(2, Int32)
            fx.copy(lds_copy64, fx.make_view(fx.add_offset(base_iter, off_i32), fx.make_layout(2, 1)), t)
            return t

        def read_a(parity, base_iter=None, mis=None):
            # base_iter (a3): the slot's own LDS symbol; otherwise buffer `parity` off a0.
            # mis: read only these m-rows (the others' entries are None).
            if const_expr(base_iter is None):
                base_iter = _iter_of(parity)
            av = []
            # D.1 (pad16>0): row//CHUNK_ROWS, row%CHUNK_ROWS and the swizzle key row%k_blk16 are
            # all independent of mi -- row = mi*16 + lane_mod_16, and mi*16 is always a multiple
            # of CHUNK_ROWS (CHUNK_ROWS | 16, asserted above), so mi*16 contributes 0 to any
            # mod/div by CHUNK_ROWS, and k_blk16==CHUNK_ROWS makes the swizzle key the same
            # reduction. Hoist to ONE computation per read_a call instead of recomputing a fresh
            # div+mod+mul chain per mi (the archived D.1 probe's first cut did the latter and
            # measured +4.4% on single_l1 -- ISA inspection this round shows the hoisted and
            # per-mi forms compile to nearly identical code, i.e. LLVM already CSEs most of the
            # per-mi chain, so static instruction count was not the single_l1 cause; the
            # strength reduction is kept anyway per the directive since it is free and can only
            # help). Each mi then adds only a Python-int constant
            # (mi*(16//CHUNK_ROWS)*CHUNK_I32, baked in at trace time since mi is
            # range_constexpr-unrolled) -- one SSA add per mi.
            if const_expr(pad16 > 0):
                lane_hi = lane_mod_16 // CHUNK_ROWS
                lane_lo = lane_mod_16 % CHUNK_ROWS
                lane_chunk_off = lane_hi * CHUNK_I32 + lane_lo * A_ROW_I32
                swz_key = lane_lo
                mi_step = (16 // CHUNK_ROWS) * CHUNK_I32  # Python int: trace-time constant
            for mi in range_constexpr(m_chunks):
                if const_expr(mis is not None and mi not in mis):
                    av.extend([None] * k_halves)
                    continue
                if const_expr(pad16 > 0):
                    row_base = lane_chunk_off + mi * mi_step  # one SSA add of a trace-time const
                    key = swz_key
                else:
                    row = mi * 16 + lane_mod_16
                    row_base = row * A_ROW_I32
                    key = row % k_blk16
                for kh in range_constexpr(k_halves):
                    lo_blk = kh * (A_KH_I32 // 4) + lane_div_16 * (A_GK_I32 // 4)
                    if const_expr(swz_lds):
                        off = row_base + (lo_blk ^ key) * 4
                    else:
                        off = row_base + kh * A_KH_I32 + lane_div_16 * A_GK_I32
                    if const_expr(a_dtype == "fp4"):
                        av.append(_read16(base_iter, off))
                    else:
                        # fp6/fp8: pack two halves (64 K apart, f8f6f4 ABI) into i32[A_NDW].
                        if const_expr(swz_lds):
                            hi_off = row_base + ((lo_blk + A_HI_OFF // 4) ^ key) * 4
                        else:
                            hi_off = off + A_HI_OFF
                        lo = Vec(fx.memref_load_vec(_read16(base_iter, off)))
                        if const_expr(A_NDW == 6):
                            # fp6: only the hi half's first 8 B are data (then the 8 B zero pad).
                            hi = Vec(fx.memref_load_vec(_read8(base_iter, hi_off)))
                        else:
                            hi = Vec(fx.memref_load_vec(_read16(base_iter, hi_off)))
                        t = fx.make_rmem_tensor(A_NDW, Int32)
                        t.store(lo.shuffle(hi, list(range(A_NDW))))
                        av.append(t)
            return av

        n_col_base = by_n + wave * (BN // 4)
        bq_views = [
            _bq_view(arg_b, n_col_base + ni * 16, KH4, K_TILES, k_halves) for ni in range_constexpr(num_acc_n)
        ]
        b_copy = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), 32)
        bs_copy = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), 32)

        # e8m0 scale buffers bounded to real size (OOB rows read 0); scale_a to the last 32-row chunk.
        _i32g = fx.PointerType.get(T.i32, address_space=fx.AddressSpace.Global, alignment=4)
        _sc_layout = fx.make_layout(1 << 28, 1)
        _a_sc_chunks = (i32_m + 31) // 32
        if const_expr(batch > 1 and sca_row_stride >= 0):
            a_sc_nrec = (
                fx.Int64(_a_sc_chunks - 1) * fx.Int64(sca_rstride) + fx.Int64(_scale_chunk_dw)
            ) * fx.Int64(4)
        else:
            a_sc_nrec = fx.Int64(_a_sc_chunks) * fx.Int64(_scale_chunk_dw) * fx.Int64(4)
        b_sc_nrec = fx.Int64((N // 32) * _scale_chunk_dw * 4)
        sa_flat = fx.logical_divide(
            fx.rocdl.make_buffer_tensor(
                fx.Tensor(fx.make_view(fx.inttoptr(_i32g, arg_scale_a), _sc_layout)),
                max_size=False,
                num_records_bytes=a_sc_nrec,
            ),
            fx.make_layout(1, 1),
        )
        sb_flat = fx.logical_divide(
            fx.rocdl.make_buffer_tensor(
                fx.Tensor(fx.make_view(fx.inttoptr(_i32g, arg_scale_b), _sc_layout)),
                max_size=False,
                num_records_bytes=b_sc_nrec,
            ),
            fx.make_layout(1, 1),
        )
        a_sc_base = [(bx_m // 32 + mp) * sca_rstride for mp in range_constexpr(m_pairs)]
        nsb = by_n // 32 + wave * (BN // 128)
        b_sc_base = [(nsb + np) * _scale_chunk_dw for np in range_constexpr(n_pairs)]
        sc_lane = lane_div_16 * 16 + lane_mod_16

        if const_expr(sa_lds_on):
            # LDS slot s holds a chunk's m_pairs A-scale words, word mp at s*m_pairs*256 + mp*256 B.
            sc_dma_atom = fx.make_copy_atom(fx.rocdl.BufferCopyLDS32b(), 32)
            lds_copy32 = fx.make_copy_atom(fx.UniversalCopy32b(), Int32)
            sa_flat_i8 = fx.logical_divide(
                fx.rocdl.make_buffer_tensor(
                    fx.Tensor(fx.make_view(fx.inttoptr(_i8g, arg_scale_a), fx.make_layout(1 << 30, 1))),
                    max_size=False,
                    num_records_bytes=a_sc_nrec,
                ),
                fx.make_layout(1, 1),
            )
            sSC_i8 = fx.recast_iter(_i8s, lds.sc.ptr)
            sSC_i32 = fx.recast_iter(Int32, lds.sc.ptr)
            a_sc_base_w = (bx_m // 32 + wave) * sca_rstride

        def dma_sa_to_lds(chunk_kt, slot):
            # Wave w DMAs word w of the chunk (64 lanes x 4 B); drained like the A DMA it follows.
            lds_ptr = fx.add_offset(sSC_i8, rocdl.readfirstlane(T.i32, slot * (m_pairs * 256) + wave * 256))
            soff = rocdl.readfirstlane(T.i32, (a_sc_base_w + chunk_kt * _scale_k0_dw) * 4)
            src = fx.slice(sa_flat_i8, (None, sc_lane * 4))
            fx.copy(sc_dma_atom, src, fx.make_view(lds_ptr, fx.make_layout(1, 1)), soffset=soff)

        def read_sa(slot):
            sa = []
            for mp in range_constexpr(m_pairs):
                t = fx.make_rmem_tensor(1, Int32)
                off = slot * (m_pairs * 64) + mp * 64 + sc_lane
                fx.copy(lds_copy32, fx.make_view(fx.add_offset(sSC_i32, off), fx.make_layout(1, 1)), t)
                sa.append(Vec(fx.memref_load_vec(t))[0])
            return sa

        n_acc = m_chunks * num_acc_n

        def load_b(kt):
            # buffer_load_dwordx4 straight into i32[4] register fragments.
            ops = []
            for ni in range_constexpr(num_acc_n):
                for kh in range_constexpr(k_halves):
                    bf = fx.make_rmem_tensor(4, Int32)
                    fx.copy_atom_call(b_copy, bq_views[ni][lane_div_16, lane_mod_16, kt, kh, None], bf)
                    ops.append(bf)
            return ops

        def load_sc(chunk_kt):
            # (sa, sb) e8m0 words per m-/n-pair for one 256-K chunk (uniform base -> SGPR soffset).
            # With sa_lds the A words come from read_sa instead and sa is empty.
            koff = chunk_kt * _scale_k0_dw
            sa = (
                []
                if sa_lds_on
                else [
                    Vec(
                        fly.copy_atom_call_ssa(
                            [T.vec(1, T.i32)],
                            bs_copy,
                            sa_flat[
                                None,
                                rocdl.readfirstlane(T.i32, a_sc_base[mp] + koff) + sc_lane,
                            ],
                        )
                    )[0]
                    for mp in range_constexpr(m_pairs)
                ]
            )
            sb = [
                Vec(
                    fly.copy_atom_call_ssa(
                        [T.vec(1, T.i32)],
                        bs_copy,
                        sb_flat[
                            None,
                            rocdl.readfirstlane(T.i32, b_sc_base[np] + koff) + sc_lane,
                        ],
                    )
                )[0]
                for np in range_constexpr(n_pairs)
            ]
            return sa, sb

        def compute(accs, av, bv, sa_v, sb_v, scale_shift=None, mis=None):
            # mis: run only these m-rows' MFMAs.
            # tile_k=128: shift the active 128-K half of the shared 256-K word into the opsel's low bytes.
            if const_expr(scale_shift is not None):
                sa_v = [v.shrui(scale_shift) for v in sa_v]
                sb_v = [v.shrui(scale_shift) for v in sb_v]
            # kh OUTERMOST: consecutive MFMAs hit distinct accumulators (dense issue). Each
            # scaled MFMA = fx.gemm over rank-1 i32[4] A/B frags, e8m0 word on scale_a=/scale_b=.
            mi_set = list(range(m_chunks)) if mis is None else list(mis)
            c_frags = [None] * n_acc
            for mi in mi_set:
                for ni in range_constexpr(num_acc_n):
                    c_frags[mi * num_acc_n + ni] = fx.make_rmem_tensor(4, Float32)
                    c_frags[mi * num_acc_n + ni].store(Vec(accs[mi * num_acc_n + ni]))
            for kh in range_constexpr(k_halves):
                for ni in range_constexpr(num_acc_n):
                    np_i, in_b = ni // 2, ni % 2
                    for mi in mi_set:
                        mp_i, im = mi // 2, mi % 2
                        cf = c_frags[mi * num_acc_n + ni]
                        atom = scale_atoms[(kh * 2 + im, kh * 2 + in_b)]
                        if const_expr(epi == 2):
                            fx.gemm(
                                atom,
                                cf,
                                bv[ni * k_halves + kh],
                                av[mi * k_halves + kh],
                                cf,
                                scale_a=sb_v[np_i],
                                scale_b=sa_v[mp_i],
                            )
                        else:
                            fx.gemm(
                                atom,
                                cf,
                                av[mi * k_halves + kh],
                                bv[ni * k_halves + kh],
                                cf,
                                scale_a=sa_v[mp_i],
                                scale_b=sb_v[np_i],
                            )
            for mi in mi_set:
                for ni in range_constexpr(num_acc_n):
                    accs[mi * num_acc_n + ni] = c_frags[mi * num_acc_n + ni].load().ir_value()
            return accs

        def compute_a_pipe(accs, base_iter, bv, sa_v, sb_v, scale_shift=None):
            # a_pipe: m-row mi's A fragment is read a_pipe groups ahead of mi's MFMAs; the
            # sched_barrier(0) fences keep the groups in order, so the backend's lgkmcnt waits
            # before each group are partial instead of one full drain before the first MFMA.
            if const_expr(scale_shift is not None):
                sa_v = [v.shrui(scale_shift) for v in sa_v]
                sb_v = [v.shrui(scale_shift) for v in sb_v]
            frags = {}
            for mi in range_constexpr(min(a_pipe, m_chunks)):
                frags[mi] = read_a(None, base_iter, [mi])
            for mi in range_constexpr(m_chunks):
                if const_expr(mi + a_pipe < m_chunks):
                    frags[mi + a_pipe] = read_a(None, base_iter, [mi + a_pipe])
                rocdl.sched_barrier(0)
                accs = compute(accs, frags[mi], bv, sa_v, sb_v, None, [mi])
                rocdl.sched_barrier(0)
            return accs

        def hot_loop_scheduler():
            # Interleave the MFMAs with the tile's vmem + A-LDS loads: preload all hints, then issue MFMAs 1-by-1.
            rocdl.sched_vmem(sched_num_gmem)
            rocdl.sched_dsrd(sched_num_ds_load)
            for _ in range_constexpr(sched_mfma_total):
                rocdl.sched_mfma(1)
            rocdl.sched_barrier(0)

        accs_init = [Vec.filled(4, 0.0, Float32).ir_value() for _ in range_constexpr(n_acc)]

        # R7 (goal.md SS C "R6", scale-only half-pipeline): carry the 6 e8m0 scale words
        # (m_pairs sa + n_pairs sb, plain Int32 SSA scalars -- NumericMeta implements
        # __extract_to_ir_values__/__construct_from_ir_values__ natively, no make_rmem_tensor
        # round-trip needed) one K-tile ahead as `range(..., init=[...])` loop state alongside
        # `accs`, mirroring the existing double-buffered A-DMA prefetch-ahead-of-use shape. B
        # stays exactly as before (`load_b(kt)`, synchronous, same iteration): the full B+scale
        # pipeline (carrying 4 i32[4] B fragments too, via make_rmem_tensor/.store()/.load() per
        # fragment) was measured bitwise-correct and VGPR-safe (216->237, <256) but the
        # pack/unpack round-trip added ~20 VALU/iter that outweighed the latency-hiding gain on
        # 5/6 shapes (regressed up to +8%); only single_l2 (K=15360) netted positive. Scale words
        # are cheap scalars with no memref round-trip (216->221 VGPR only), so this isolates the
        # latency-hiding benefit from the B-pipeline's overhead. rocprofv3 confirms the mechanism
        # on both evidence shapes: SQ_WAIT_ANY -25%(fc2)/-28%(l2), and unlike the full pipeline
        # SQ_WAVE_CYCLES actually drops here too (fc2 -2.9%, l2 -4.3%) instead of staying flat,
        # because there isn't enough new VALU overhead to reabsorb the freed cycles. Card-6 ABBA
        # (N=30, trimmed) measured net non-negative on all 6 shapes: double_fc2 -1.36% (passes
        # the strict mean>1%+stdev<half-mean trust gate), single_l2 -0.85% (stdev 0.04, very
        # tight but just under the 1% bar), double_fc1 -0.77% (28/30 blocks negative), the
        # remaining three (double_qkv +0.22%, double_proj +0.19%, single_l1 +0.01%) flat at
        # noise level -- no shape regresses. (R8 then moved the A-DMA's kt*A_ROW_B into the
        # buffer descriptor base, see dma_a_to_lds.)
        #
        # U2 (campaign 2 R2, b_prefetch 1/2): the loop runs K_TILES/2 iterations, one per 256-K
        # scale chunk, as two halves. Half h reads A(kt0+h) from LDS buffer h, DMAs A(kt0+h+1)
        # into the other buffer and runs its MFMAs on B(kt0+h), which the previous half loaded:
        # half 0 consumes the B carried across the iteration boundary, half 1 the B half 0
        # loaded, half 1 loads the B the next iteration's half 0 consumes. That is the B
        # carry of R7's note without its cost: one carried B set per two K-tiles and no phi
        # copies. The chunk's scale words load once (half 0) instead of once per K-tile; half 0
        # uses their low 16 bits, half 1 the high 16 bits. b_prefetch=2 issues the next B and
        # scales behind the A DMA, so the end-of-half wait drains exactly the DMA: vmcnt(<loads
        # issued after it>). Two orderings are load-bearing (each measured >10% when broken):
        # the half's ds_reads must precede its DMA in program order (an LDS DMA ahead of a
        # ds_read makes the backend wait vmcnt(0) before the read), and the sched_barrier(0)
        # right after the DMA keeps the scheduler from hoisting the B/scale loads above it
        # (the backend then strengthens the partial wait to vmcnt(0)).
        # sa_lds: the next chunk's A-scale DMA joins half 0's A DMA group, so half 0's partial
        # wait drains it too; the chunk's words are read back at the top of the next iteration,
        # two barriers after the slot they overwrite was last read.
        N_CH = K_TILES // 2
        n_bfr = num_acc_n * k_halves
        n_sa_c = 0 if sa_lds_on else m_pairs  # carried A-scale words
        n_sc = n_sa_c + n_pairs

        def sched_half(n_vmem):
            # n_vmem = B/scale loads of the half. Hint counts must equal the region's opcode
            # counts, or the backend silently drops the whole group.
            if const_expr(b_prefetch == 2):
                # ds_reads and the DMA sit above the fence; this region is B/scales + MFMAs.
                rocdl.sched_vmem(n_vmem)
            else:
                rocdl.sched_vmem(n_coop + n_vmem)
                rocdl.sched_dsrd(sched_num_ds_load)
            for _ in range_constexpr(sched_mfma_total):
                rocdl.sched_mfma(1)
            rocdl.sched_barrier(0)

        def wait_vmcnt(n):
            # s_waitcnt vmcnt(n) expcnt(7) lgkmcnt(0); vmcnt splits into bits [3:0] and [15:14].
            rocdl.s_waitcnt((n & 0xF) | (7 << 4) | (((n >> 4) & 0x3) << 14))

        def drain_half(n_after_dma):
            if const_expr(b_prefetch == 2):
                wait_vmcnt(n_after_dma)
            else:
                rocdl.s_waitcnt(0)

        def sched_mfma_region():
            for _ in range_constexpr(sched_mfma_total):
                rocdl.sched_mfma(1)
            rocdl.sched_barrier(0)

        def clamp_kt(kt):
            # min(kt, K_TILES - 1) for kt <= K_TILES + 1 (the redundant tail prefetches).
            return kt - (kt // K_TILES) * (kt - (K_TILES - 1))

        def _frag(v):
            t = fx.make_rmem_tensor(4, Int32)
            t.store(Vec(v))
            return t

        if const_expr(a3):
            dma_a_to_lds(fx.Int32(0), None, slot_i8[0])
        else:
            dma_a_to_lds(fx.Int32(0), fx.Int32(0))
        if const_expr(sa_lds_on):
            dma_sa_to_lds(fx.Int32(0), fx.Int32(0))
        if const_expr(has_bias and epi == 2):
            # epi=2 bias: a lane's 4 consecutive columns per n-fragment as one 8 B load, issued
            # here so the prologue drain covers it instead of a wait after the K loop.
            bias_ptr_ty = fx.PointerType.get(
                out_elem.ir_type, address_space=fx.AddressSpace.Global, alignment=8
            )
            bias_flat4 = fx.logical_divide(
                fx.rocdl.make_buffer_tensor(
                    fx.Tensor(fx.make_view(fx.inttoptr(bias_ptr_ty, arg_bias), fx.make_layout(1 << 28, 1))),
                    max_size=False,
                    num_records_bytes=fx.Int64(N) * fx.Int64(2),
                ),
                fx.make_layout(4, 1),
            )
            bias_copy64 = fx.make_copy_atom(fx.rocdl.BufferCopy64b(), out_elem)
            col_t4 = (by_n + wave * (BN // 4) + lane_div_16 * 4) // 4
            bias_t = []
            for ni in range_constexpr(num_acc_n):
                bt = fx.make_rmem_tensor(4, out_elem)
                fx.copy(bias_copy64, bias_flat4[None, col_t4 + ni * 4], bt)
                bias_t.append(bt.load())
        sa_pf0, sb_pf0 = load_sc(fx.Int32(0))
        b_pf0 = load_b(fx.Int32(0)) if u2 else []
        if const_expr(a3):
            # A(1) stays in flight into the loop: the first half's drain retires it. A bare
            # s_barrier: gpu.barrier()'s workgroup release fence would drain every LDS DMA.
            rocdl.sched_barrier(0)
            dma_a_to_lds(fx.Int32(1), None, slot_i8[1])
            wait_vmcnt(n_coop)
            rocdl.s_barrier()
        else:
            rocdl.s_waitcnt(0)
            gpu.barrier()
        loop_init = accs_init + list(sa_pf0) + list(sb_pf0) + [b.load().ir_value() for b in b_pf0]
        if const_expr(a3):
            n_iters = N_CH // 3
        elif const_expr(u2):
            n_iters = N_CH
        else:
            n_iters = K_TILES
        for iv, state in range(fx.Index(0), fx.Index(n_iters), fx.Index(1), init=loop_init):
            state = list(state)
            accs = state[:n_acc]
            sa_v = state[n_acc : n_acc + n_sa_c]
            sb_v = state[n_acc + n_sa_c : n_acc + n_sc]
            if const_expr(a3):
                # a_stages=3: three chunks (six K-tiles) per iteration, so K-tile t's slot t%3 is a
                # fixed symbol in every half. Half t reads A(t), issues B(t+1) (and in half 0 the
                # next chunk's scales) and only then the DMA of A(t+2) into slot (t+2)%3, so the
                # waits on B(t+1) in half t+1 never retire that DMA (vmcnt is in order). Its drain
                # moves one half later: the end of half t retires A(t+1), issued in half t-1,
                # with vmcnt(<loads issued after it>): 4 B + 2 scales + 4 DMA + 1 A-scale DMA = 11
                # after half 0, 4 B + 4 DMA = 8 after half 1 (A(t+2) and the A-scale DMA of half 0).
                # Slot (t+2)%3 was last read by K-tile t-1, one barrier earlier. The halves end
                # with a bare s_barrier: the waits above already order LDS, and gpu.barrier()'s
                # workgroup release fence would also drain the A(t+2) DMA (vmcnt(0)).
                jj = fx.Int32(iv)
                bv0 = [_frag(v) for v in state[n_acc + n_sc :]]
                for c in range_constexpr(3):
                    ch = jj * 3 + c
                    t0 = ch * 2
                    nch = ch + 1
                    ch_pf = nch - nch // N_CH  # clamp the last chunk prefetch to N_CH-1
                    sc_slot = ch % 2
                    sa_v = read_sa(sc_slot)
                    # half 0: A(t0) in slot (2c)%3, carried B(t0), chunk ch's low scale half.
                    if const_expr(a_pipe == 0):
                        av = read_a(None, slot_i32[(2 * c) % 3])
                    bv1 = load_b(t0 + 1)
                    sa_next, sb_next = load_sc(ch_pf)
                    rocdl.sched_barrier(0)
                    dma_a_to_lds(clamp_kt(t0 + 2), None, slot_i8[(2 * c + 2) % 3])
                    dma_sa_to_lds(ch_pf, fx.Int32(1) - sc_slot)
                    rocdl.sched_barrier(0)
                    if const_expr(a_pipe > 0):
                        accs = compute_a_pipe(accs, slot_i32[(2 * c) % 3], bv0, list(sa_v), list(sb_v), None)
                    else:
                        accs = compute(accs, av, bv0, list(sa_v), list(sb_v), None)
                        sched_mfma_region()
                    wait_vmcnt(n_bfr + n_sc + n_coop + 1)
                    rocdl.s_barrier()
                    # half 1: A(t0+1) in slot (2c+1)%3, B(t0+1), chunk ch's high scale half.
                    if const_expr(a_pipe == 0):
                        av = read_a(None, slot_i32[(2 * c + 1) % 3])
                    bv2 = load_b(clamp_kt(t0 + 2))
                    rocdl.sched_barrier(0)
                    dma_a_to_lds(clamp_kt(t0 + 3), None, slot_i8[(2 * c + 3) % 3])
                    rocdl.sched_barrier(0)
                    if const_expr(a_pipe > 0):
                        accs = compute_a_pipe(
                            accs, slot_i32[(2 * c + 1) % 3], bv1, list(sa_v), list(sb_v), fx.Int32(16)
                        )
                    else:
                        accs = compute(accs, av, bv1, list(sa_v), list(sb_v), fx.Int32(16))
                        sched_mfma_region()
                    wait_vmcnt(n_bfr + n_coop)
                    rocdl.s_barrier()
                    bv0 = bv2
                    sb_v = sb_next
                carry = accs + list(sb_v) + [b.load().ir_value() for b in bv0]
            elif const_expr(u2):
                j = fx.Int32(iv)
                kt1 = j * 2 + 1
                nj = j + 1
                ch_pf = nj - nj // N_CH  # clamp the last chunk prefetch to N_CH-1
                nkt = kt1 + 1
                kt_pf = nkt - nkt // K_TILES  # clamp the last tile prefetch to K_TILES-1
                if const_expr(sa_lds_on):
                    sc_slot = j % 2
                    sa_v = read_sa(sc_slot)
                    if const_expr(b_prefetch != 2):
                        # Keeps these reads out of half 0's sched_dsrd count.
                        rocdl.sched_barrier(0)
                # half 0: A(kt0) in LDS buffer 0, carried B(kt0), chunk j's low scale half.
                av = read_a(fx.Int32(0))
                bv0 = [_frag(v) for v in state[n_acc + n_sc :]]
                if const_expr(b_prefetch == 2):
                    dma_a_to_lds(kt1, fx.Int32(1))
                    if const_expr(sa_lds_on):
                        dma_sa_to_lds(ch_pf, fx.Int32(1) - sc_slot)
                    rocdl.sched_barrier(0)
                    bv1 = load_b(kt1)
                    sa_next, sb_next = load_sc(ch_pf)
                else:
                    bv1 = load_b(kt1)
                    sa_next, sb_next = load_sc(ch_pf)
                    dma_a_to_lds(kt1, fx.Int32(1))
                    if const_expr(sa_lds_on):
                        dma_sa_to_lds(ch_pf, fx.Int32(1) - sc_slot)
                accs = compute(accs, av, bv0, list(sa_v), list(sb_v), None)
                sched_half(n_bfr + n_sc + (1 if sa_lds_on and b_prefetch != 2 else 0))
                drain_half(n_bfr + n_sc)
                gpu.barrier()
                # half 1: A(kt1) in LDS buffer 1, B(kt1), chunk j's high scale half.
                av = read_a(fx.Int32(1))
                if const_expr(b_prefetch == 2):
                    dma_a_to_lds(kt_pf, fx.Int32(0))
                    rocdl.sched_barrier(0)
                    bv2 = load_b(kt_pf)
                else:
                    bv2 = load_b(kt_pf)
                    dma_a_to_lds(kt_pf, fx.Int32(0))
                accs = compute(accs, av, bv1, list(sa_v), list(sb_v), fx.Int32(16))
                sched_half(n_bfr)
                drain_half(n_bfr)
                gpu.barrier()
                carry = accs + list(sa_next) + list(sb_next) + [b.load().ir_value() for b in bv2]
            else:
                kt = fx.Int32(iv)
                cur = kt % 2
                nxt = (kt + 1) % 2
                nkt = kt + 1
                pf_kt = nkt - nkt // K_TILES  # clamp last-iter prefetch to K_TILES-1
                scale_shift = None if tiles_per_chunk == 1 else (kt % tiles_per_chunk) * 16
                chunk_pf = pf_kt if tiles_per_chunk == 1 else pf_kt // tiles_per_chunk
                av = read_a(cur)
                bv = load_b(kt)
                sa_next, sb_next = load_sc(chunk_pf)  # prefetch kt+1's scale chunk ahead of compute
                dma_a_to_lds(pf_kt, nxt)  # A DMA after B/scale loads -> overlaps the MFMAs
                accs = compute(accs, av, bv, list(sa_v), list(sb_v), scale_shift)
                hot_loop_scheduler()
                rocdl.s_waitcnt(0)  # drain the A DMA before the barrier
                gpu.barrier()
                carry = accs + list(sa_next) + list(sb_next)
            results = yield carry
        accs = list(results)[:n_acc]

        # Epilogue via fx.copy: a lane owns 4 rows per (mi,ni) accm (row m*16+(l//16)*4+ii, col
        # base+l%16), c_stride apart; c_flat bounds ragged-M OOB and honors c_row/batch_stride.
        c_stride = N if c_row_stride < 0 else c_row_stride
        if const_expr(c_row_stride < 0):
            c_nrec = fx.Int64(i32_m) * fx.Int64(N) * fx.Int64(2)
        else:
            c_nrec = (fx.Int64(i32_m - fx.Int32(1)) * fx.Int64(c_stride) + fx.Int64(N)) * fx.Int64(2)
        c_addr = arg_c
        if const_expr(batch > 1):
            c_bstride = (
                fx.Int64(i32_m) * fx.Int64(N) * fx.Int64(2)
                if c_batch_stride < 0
                else fx.Int64(c_batch_stride)
            )
            c_addr = c_addr + fx.Int64(bid_z) * c_bstride
        c_ptr_ty = fx.PointerType.get(out_elem.ir_type, address_space=fx.AddressSpace.Global, alignment=2)
        c_flat = fx.logical_divide(
            fx.rocdl.make_buffer_tensor(
                fx.Tensor(fx.make_view(fx.inttoptr(c_ptr_ty, c_addr), fx.make_layout(1 << 28, 1))),
                max_size=False,
                num_records_bytes=c_nrec,
            ),
            fx.make_layout(1, 1),
        )
        c_copy = fx.make_copy_atom(fx.rocdl.BufferCopy16b(), out_elem)
        c_rstride = fx.Int32(c_stride)
        col_w = by_n + wave * (BN // 4) + lane_mod_16
        # Bias joins the bf16-rounded product in fp32 and rounds again: the same bytes as a
        # separate `out + bias`, which is what the caller would otherwise launch.
        if const_expr(has_bias and epi == 2):
            bias_f = [Vec(bias_t[ni]).to(Float32) for ni in range_constexpr(num_acc_n)]
        elif const_expr(has_bias):
            bias_flat = fx.logical_divide(
                fx.rocdl.make_buffer_tensor(
                    fx.Tensor(fx.make_view(fx.inttoptr(c_ptr_ty, arg_bias), fx.make_layout(1 << 28, 1))),
                    max_size=False,
                    num_records_bytes=fx.Int64(N) * fx.Int64(2),
                ),
                fx.make_layout(1, 1),
            )
            bias_f = []
            for ni in range_constexpr(num_acc_n):
                bf = fx.make_rmem_tensor(1, out_elem)
                fx.copy(c_copy, bias_flat[None, col_w + ni * 16], bf)
                b = bf.load().to(Float32)[0]
                bias_f.append(Vec.from_elements([b, b, b, b], Float32))
        if const_expr(epi == 2):
            # C^T fragments: lane l holds row l%16, columns (l//16)*4..+3 of each n-fragment. Per
            # n-fragment pair the two permlane16_swaps leave 8 consecutive columns in each lane,
            # at col_off 0/16/8/24 for lane group 0/1/2/3: one 16 B store per lane.
            c_ptr_ty16 = fx.PointerType.get(
                out_elem.ir_type, address_space=fx.AddressSpace.Global, alignment=16
            )
            c_flat8 = fx.logical_divide(
                fx.rocdl.make_buffer_tensor(
                    fx.Tensor(fx.make_view(fx.inttoptr(c_ptr_ty16, c_addr), fx.make_layout(1 << 28, 1))),
                    max_size=False,
                    num_records_bytes=c_nrec,
                ),
                fx.make_layout(8, 1),
            )
            c_copy128 = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), out_elem)
            col_l = by_n + wave * (BN // 4) + (lane_div_16 % 2) * 16 + (lane_div_16 // 2) * 8
            for mi in range_constexpr(m_chunks):
                row_t = bx_m + mi * 16 + lane_mod_16
                for p in range_constexpr(num_acc_n // 2):
                    va = Vec(accs[mi * num_acc_n + 2 * p])
                    vb = Vec(accs[mi * num_acc_n + 2 * p + 1])
                    if const_expr(has_bias):
                        va = va.to(out_elem).to(Float32) + bias_f[2 * p]
                        vb = vb.to(out_elem).to(Float32) + bias_f[2 * p + 1]
                    d_a0 = rocdl.cvt_pk_bf16_f32(va[0], va[1])
                    d_a1 = rocdl.cvt_pk_bf16_f32(va[2], va[3])
                    d_b0 = rocdl.cvt_pk_bf16_f32(vb[0], vb[1])
                    d_b1 = rocdl.cvt_pk_bf16_f32(vb[2], vb[3])
                    w0, w2 = _permlane16_swap(d_a0, d_b0)
                    w1, w3 = _permlane16_swap(d_a1, d_b1)
                    t8 = fx.make_rmem_tensor(8, out_elem)
                    t8.store(Vec.from_elements([w0, w1, w2, w3], Int32).bitcast(out_elem))
                    off = row_t * c_rstride + col_l + p * 32
                    fx.copy(c_copy128, t8, c_flat8[None, off // 8])
        elif const_expr(epi == 1):
            # Lane group g holds rows 4g..4g+3, two per dword (h). Swapping fragment 2p's odd groups
            # with fragment 2p+1's even groups leaves each 32-lane half two rows of 32 consecutive
            # columns: s0 rows 2h+e (lanes 0-31) / 8+2h+e (lanes 32-63), s1 the same plus 4.
            col_rm = by_n + wave * (BN // 4) + lane % 32
            row_rm = bx_m + (lane // 32) * 8
            for mi in range_constexpr(m_chunks):
                for p in range_constexpr(num_acc_n // 2):
                    va = Vec(accs[mi * num_acc_n + 2 * p]).to(out_elem)
                    vb = Vec(accs[mi * num_acc_n + 2 * p + 1]).to(out_elem)
                    if const_expr(has_bias):
                        va = (va.to(Float32) + bias_f[2 * p]).to(out_elem)
                        vb = (vb.to(Float32) + bias_f[2 * p + 1]).to(out_elem)
                    wa = va.bitcast(Int32)
                    wb = vb.bitcast(Int32)
                    for h in range_constexpr(2):
                        s0, s1 = _permlane16_swap(wa[h], wb[h])
                        for w, r0 in ((s0, 2 * h), (s1, 4 + 2 * h)):
                            pair = Vec.from_elements([fx.Int32(w)], Int32).bitcast(out_elem)
                            for e in range_constexpr(2):
                                cf = fx.make_rmem_tensor(1, out_elem)
                                cf.store(Vec.from_elements([pair[e]], out_elem))
                                off = (row_rm + mi * 16 + r0 + e) * c_rstride + col_rm + p * 32
                                fx.copy(c_copy, cf, c_flat[None, off])
        else:
            for mi in range_constexpr(m_chunks):
                row_m = bx_m + mi * 16 + lane_div_16 * 4
                for ni in range_constexpr(num_acc_n):
                    col = col_w + ni * 16
                    acc = Vec(accs[mi * num_acc_n + ni]).to(out_elem)
                    if const_expr(has_bias):
                        acc = (acc.to(Float32) + bias_f[ni]).to(out_elem)
                    for ii in range_constexpr(4):
                        cf = fx.make_rmem_tensor(1, out_elem)
                        cf.store(Vec.from_elements([acc[ii]], out_elem))
                        off = (row_m + ii) * c_rstride + col
                        fx.copy(c_copy, cf, c_flat[None, off])

    c_addr = fx.Int64(fx.ptrtoint(arg_c))
    a_addr = fx.Int64(fx.ptrtoint(arg_a))
    b_addr = fx.Int64(fx.ptrtoint(arg_b))
    sa_addr = fx.Int64(fx.ptrtoint(arg_scale_a))
    sb_addr = fx.Int64(fx.ptrtoint(arg_scale_b))
    bias_addr = fx.Int64(fx.ptrtoint(arg_bias))
    if const_expr(waves_per_eu > 0):
        wpe = waves_per_eu
    else:
        wpe = None
    gx = (i32_m + (BM - 1)) // BM
    gy = i32_n // BN
    # group_m>0: flatten to a 1D grid so kernel_gemm's block_mn decode (above) sees every
    # workgroup id in band order; group_m==0: today's native 2D grid, unchanged.
    if const_expr(group_m > 0):
        grid = (gx * gy, 1, batch)
    else:
        grid = (gx, gy, batch)
    kernel_gemm(
        c_addr,
        a_addr,
        b_addr,
        sa_addr,
        sb_addr,
        i32_m,
        i32_n,
        bias_addr,
        value_attrs={"rocdl.waves_per_eu": wpe},
    ).launch(grid=grid, block=(256, 1, 1), stream=stream)


# The caller pads M to a multiple of this and guarantees nothing more, so every tile_m
# chosen below must divide it.
M_ALIGN = 256
# (tile_m, tile_n, tile_k, waves_per_eu, group_m, pad16) per (N, K), from the FLUX-shape sweep on
# MI355X. group_m (R4, goal.md SS C/B.4): per-shape L2 band raster width for the flat-grid
# block_mn decode above; 0 = native 2D grid. Measured on a knob-parameterised probe copy
# (p4_ab.py / kvar.py) before landing here: single_l1 (N=21504) and single_l2 (K=15360) are
# the two shapes where banding measured positive, each at its own interior optimum (gm=1 and
# gm=gx both measure at or above the native grid, so gm=32 / gm=2 are not a dispatch-order
# artifact); the other four shapes measured neutral-to-worse at every gm tried, so they keep
# gm=0.
# pad16 (R6, goal.md D.1): per-shape LDS row padding, in units of 16 bytes, after every
# CHUNK_ROWS=8 A rows (see launch_gemm). Correctness is identical regardless of pad16 (padding
# only moves WHERE A lives in LDS, verified bitwise-identical output at pad16 in {0,8} on all six
# shapes, at both native M and M=M_ALIGN=256); the per-shape value below is purely a timing
# choice, picked by isolated (no concurrent GPU activity -- concurrent probes on other cards
# measurably confounded the smaller shapes), interleaved A/B/A/B median comparison, N>=34 blocks
# per shape, re-run twice: single_l2/double_fc2/double_proj measured a reproducible, directionally
# -100%-consistent win at pad16=8 (-2.2 to -3.0%, -0.86 to -1.1%, -1.3 to -1.6% respectively,
# excluding box-noise-contaminated samples per kb/flydsl/pitfalls.md's canary rule); single_l1
# measured a reproducible, 0-faster-out-of-34-blocks regression (+4.2 to +4.4%) that persists even
# with the strength-reduced read_a (ISA-confirmed near-identical to the un-reduced form, so the
# regression is not from extra per-mi address VALU, likely an occupancy/launch-pattern effect at
# single_l1's much larger grid -- next lever, not re-tried blindly this round); double_fc1 and
# double_qkv showed a small (+1.4%, ~0%) non-win once isolated from cross-probe interference (an
# earlier concurrent-cards measurement wrongly suggested a double_fc1 win -- see this round's
# ROUND_REPORT), so fc1 keeps pad16=0 rather than ship an unproven per-shape change.
# R9 (this round, goal.md "double_proj / double_qkv per-shape tile/launch" directive): swept
# tile_m/n/k in {(128,256,128) [current], (256,128,128), (256,256,128), (128,128,256)} x
# waves_per_eu{1,2} x group_m{0,1,2,3,4,6,8} x pad16{0,8} (pad16=0 only at tile_k=256, which
# trips the k_blk16==CHUNK_ROWS assert at pad16>0) for double_proj and double_qkv specifically,
# on top of R6/R7/R8's since-landed scale-prefetch + A-DMA SRD-base LICM. Every alternate tile
# shape measured worse than today's (128,256,128) for both shapes (256,128,128 and 256,256,128
# both regress sharply, 256,256,128 at waves_per_eu=2 catastrophically so -- n_acc doubles to 64,
# ~9x slower, consistent with an occupancy collapse at that accumulator count; 128,128,256 is a
# smaller but still consistent regression on both shapes, consistent with the campaign's existing
# tile_k=256 dead end now reconfirmed on these two shapes too) and no group_m banding beat 0 for
# either shape at any pad16/wpe (double_proj was already known-optimal; double_qkv's own gm scan
# at pad16=8 confirms the same gm=0 optimum). The one real move: holding tile/wpe/gm fixed at
# today's values, double_qkv's pad16 flips from a measured non-win in R6 (pre-R7/R8 codebase) to
# a reproducible win now -- the R7 scale-prefetch and R8 SRD-base LICM changes evidently shifted
# the balance between pad16's own extra chunk-address VALU cost and the LDS-bank-conflict cycles
# it removes (R6's own pitfalls note already flagged this exact trade-off as shape- and
# codebase-state-dependent, not a one-time verdict). Confirmed via: a direct min-of-30 (bench.py's
# own timed_min semantics) A/B on two separate cards (-8.2%, -11.3%); a 38/38-block, mean>1%+
# std<half-mean-significant isolated card-6 ABBA at N_REP=80 (-1.87% median, -1.90% mean, std
# 0.14, trim10% -1.89%) that isolates pure GPU pipe time from host dispatch, confirming the
# effect is real (not purely host-side) even though its magnitude under that burst-timing method
# is smaller than under the ruler's own per-call-sync timing; and the official card-7 bench.sh
# itself end to end (next section) -- all four measurements agree on direction and on
# significance, only disagreeing on exact magnitude depending on how much of the per-call host
# round-trip (torch.cuda.Event + synchronize every rep, exactly what bench.py's timed_min does)
# each methodology includes. ISA: scratch=0, spill=0, agpr=0, vgpr 219->219 (unchanged), sgpr
# 55->55 (unchanged) at both pad16 values for this shape; LDS 32768->36864B (+12.5%, the expected
# pad16=8 cost), still far under the 160 KiB/CU budget so occupancy is unaffected. Correctness
# (bitwise-identical regardless of pad16, since padding only moves WHERE A lives in LDS) was
# already established for all six shapes at pad16 in {0,8} in R3/R6; reconfirmed this round
# end-to-end for double_qkv specifically against the current (post-R7/R8) codebase.
# b_prefetch (7th field, campaign 2 R2): the U2 loop flips two of the verdicts above, so the
# loop and its configs were re-measured together (in-process A/B vs the round-0 kernel,
# bitwise identical): pad16=8 now wins on single_l1 and double_fc1, single_l1 is faster with
# waves_per_eu unset than at 2, and single_l2 (K=15360) prefers b_prefetch=1.
# sa_lds (8th field, campaign 2 R4): A-scale words DMA'd once per workgroup and shared through
# LDS (see launch_gemm), on top of the fp6 ds_read_b64 hi-half read it was measured with.
# epi (9th field, campaign 2 R5): C store per shape, picked by in-process A/B against epi=0
# (bitwise): the dwordx4 store with the prologue bias load (2) on five shapes (-0.7..-2.4%),
# the row-merged store (1) on double_fc1, where 2 measured +0.1..+0.6% and 1 -1.1..-1.7%.
# xcd (10th field, campaign 2 R5): num_xcd for the XCD-major remap above, with group_m re-swept
# on top of the new epilogues (in-process A/B, bitwise): xcd=8 with gm=5/6/5 on double_qkv/
# double_proj/double_fc1 (-3.7%, -1.2..-2.4%, -4.1..-4.6%); without the remap no gm beat the
# native grid there, and the remap with the native M-fastest order is +6..+18% on every M=8192
# shape. double_fc2 and both single shapes measured flat-to-worse with the remap at every gm.
# a_stages (11th field, campaign 2 R7): the three-slot A ring (see launch_gemm), in-process A/B
# against a_stages=2 (two replicates each, bitwise): double_qkv -0.3/-0.6%, double_proj
# -1.7/-2.8%, double_fc2 -1.3/-0.5%; single_l1 +1.4%, single_l2 +0.1..+0.3%, and double_fc1
# (epi=1) needs 264 VGPR + 8 AGPR with it (+11.5%), so those three keep 2.
# a_pipe (12th field, campaign 2 R7): with the ring, single_l1 reads each A fragment 4 m-rows
# ahead of its MFMAs. 2 ahead measured -0.5..-0.8% against a_stages=2 over three replicates (1
# ahead the same), where the ring alone is +1.3..+1.4%; 4 ahead then -0.6..-0.8% against 2 over
# four replicates (3 and 6 ahead in between; group_m 16..64 all slower than 32 on the ring). On
# the other shapes a_pipe measured +0.4..+0.6% (double_fc2), +1.8..+2.2% (double_proj), flat
# (double_qkv), +0.4..+0.8% (single_l2, ring) and +0.3..+1.9% (double_fc1, ring), so they keep 0.
_A6W4_CONFIGS = {
    (9216, 3072): (128, 256, 128, 1, 5, 8, 2, 1, 2, 8, 3, 0),  # double_qkv
    (3072, 3072): (128, 256, 128, 1, 6, 8, 2, 1, 2, 8, 3, 0),  # double_proj
    (12288, 3072): (128, 256, 128, 1, 5, 8, 2, 1, 1, 8, 2, 0),  # double_fc1
    (3072, 12288): (128, 256, 128, 1, 0, 8, 2, 1, 2, 1, 3, 0),  # double_fc2
    (21504, 3072): (128, 256, 128, 0, 32, 8, 2, 1, 2, 1, 3, 4),  # single_l1
    (3072, 15360): (128, 256, 128, 0, 2, 8, 1, 1, 2, 1, 2, 0),  # single_l2
}
_A6W4_DEFAULT = (128, 256, 128, 0, 0, 0, 2, 0, 0, 1, 2, 0)

# One flyc.compile'd launch_gemm artifact per (N, K). Every Constexpr argument launch_gemm takes
# (tile_m/n/k, a_dtype, out_dtype, batch, the five stride flags, waves_per_eu, group_m, num_xcd,
# pad16, b_prefetch, sa_lds, epi, a_stages, a_pipe) is a pure function of (n, k) through _A6W4_CONFIGS/
# _A6W4_DEFAULT above, while i32_m, i32_n, stream and the five operand pointers are all runtime
# args whose FlyDSL types (Int32 / Stream / Pointer) are
# annotated-runtime-type -> type-only in the compile cache key (kb/flydsl/compilation_pipeline.md
# S:_arg_cache_sig), so a single artifact is correct for every M at a given (N, K). Verified by
# compiling once at M=2048 and calling the same object at M=256/2048/8192/16384: SNR 55.588/
# 55.593/55.601/55.597, all finite (campaign probe p7_mindep.py); group_m's own grid/band math is
# rebuilt from i32_m on every call (both the host gx and the device num_pid_m), so this M-independence
# holds with group_m>0 too. Calling launch_gemm directly instead re-pays its @flyc.jit dispatch on
# every call (signature bind over 24 args + cache-key build + globals-drift check), measured at
# ~42.5 us/call here -- flat across all six shapes and dwarfing torch.empty's 1.1 us, so it is JIT
# dispatch, not allocation. gemm_mxfp4_kernel.py (the production MXFP4 GEMM) and gemm_fp8_kernel.py
# both already compile-cache for this reason; this is the same pattern applied to the one launcher
# in this file that still used the raw @flyc.jit closure on every call.
_A6W4_LAUNCH_CACHE: dict = {}


def gemm_a6w4(aq, wq, sa, sb, n, k, bias=None):
    """C[M, N] bf16 = A6 @ W4^T (+ bias) for the A6W4 forward.

    aq    uint8 [M, K]      FP8-padded packed FP6 (E2M3): 24 code bytes + 8 zero bytes per 32
    wq    uint8 [N, K/2]    E2M1, low nibble = even element, preshuffled (shuffle_weight_w4)
    sa    uint8 [M, K/32]   E8M0, shuffle_scale_w4 layout
    sb    uint8 [N, K/32]   E8M0, shuffle_scale_w4 layout
    bias  bf16 [N] or None  added to the bf16 product, rounding like a separate `c + bias`
    M is a multiple of M_ALIGN; N and K are the FLUX shapes (N % 256 == 0, K % 256 == 0).
    """
    import torch

    m = aq.shape[0]
    cfg = _A6W4_CONFIGS.get((n, k), _A6W4_DEFAULT)
    tm, tn, tk, wpe, gm, pad16, bpf, sa_lds, epi, xcd, a_stg, a_pipe = cfg
    c = torch.empty(m, n, dtype=torch.bfloat16, device=aq.device)
    ptr = lambda t: flyc.from_c_void_p(fx.Uint8, t.data_ptr())
    if bias is not None:
        bias = bias.to(torch.bfloat16).contiguous()
    args = (
        ptr(c),
        ptr(aq),
        ptr(wq),
        ptr(sa),
        ptr(sb),
        m,
        n,
        torch.cuda.current_stream(),
        n,
        k,
        tm,
        tn,
        tk,
        "fp6",
        "bf16",
        1,
        -1,
        -1,
        -1,
        -1,
        -1,
        -1,
        wpe,
        gm,
        xcd,
        pad16,
        bpf,
        sa_lds,
        epi,
        ptr(c if bias is None else bias),
        bias is not None,
        a_stg,
        a_pipe,
    )
    if torch.cuda.is_current_stream_capturing():
        # A flyc.compile'd object regresses under CUDA-graph capture (no recompile mid-capture);
        # fall back to the raw @flyc.jit closure, same rule gemm_mxfp4_kernel.py follows.
        launch_gemm(*args)
    else:
        run_compiled(_A6W4_LAUNCH_CACHE, (n, k, bias is not None), launch_gemm, *args)
    return c
