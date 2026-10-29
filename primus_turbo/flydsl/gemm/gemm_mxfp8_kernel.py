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

"""8-wave MXFP8 matmul (per-1x32 E8M0 block scaling) for AMD CDNA4 (gfx950).

Parallels the tensorwise FP8 path in ``gemm_fp8_kernel.py``, except the per-32-K-block
E8M0 scale is fed to ``v_mfma_scale_f32_16x16x128_f8f6f4`` every K-iteration, so the
epilogue is a plain FP32->BF16 store.

The MMA takes one i32 scale per operand holding 4 packed E8M0 bytes (one per 32-K block);
a 16x16x128 MFMA spans K=128, so one i32 scale per (row/col tile, K-iteration).

Scale tensor layout expected by this kernel (passed pre-packed from host):
  A_scale: int32 [M, K // 128]   (each i32 == 4 consecutive E8M0 bytes of a row)
  B_scale: int32 [N, K // 128]
i.e. the raw uint8 E8M0 [DIM, K//32] viewed little-endian as int32.
"""

import warnings

import torch

# isort: off
# @flyc.kernel resolves these as module globals, not closure cells.
from primus_turbo.flydsl.utils.gemm_helper import (
    block_mn,
    build_preshuffle_ab_kernel,
    compute_global_swizzle,
    G2SLoader,
    make_fp8_buffer_tensor_rebased,
    make_row_band_resource,
    make_value_attrs,
    MfmaScale16x16x128,
    _PRESHUF_KT,
    resolve_accum_out,
    S2RLoader,
    ScaleBComb,
    scale_opsel,
    ScaleS2R,
    _store_quadrants,
    StoreCPerTensor,
    StoreCPerTensorRowN,
    swizzle_128,
    wait_barrier,
    xcd_remap_pid,
)
from primus_turbo.flydsl.utils.prims import ceildiv
from primus_turbo.flydsl.utils import buffer_ops
from primus_turbo.flydsl.autotune import Compile, NoWorkingConfig, Op, Race, Refine
from primus_turbo.flydsl.autotune.engine import Entry
from primus_turbo.flydsl.autotune.policies import Point, Synthetic
from primus_turbo.flydsl.autotune.race import staged

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith, const_expr, range_constexpr, rocdl
from flydsl.expr.arith import _to_raw as _raw
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

# isort: on

# E8M0 scales per preshuffled dword: unpacked, a dword carries one K-iteration's byte.
_MX_SCALE_PACK = 4

# Splitting K for the TAIL tiles alone refills a dispatch's short last round at a
# partial-sum cost that scales with the tail, not with the whole output.
_MX_SPLIT_MIN_KI = 16  # K-iters per slice; below this the per-tile fixed cost eats the gain
_FOLD_BLK = 256
_FOLD_ROWS = 64  # C rows one fold workgroup owns

# `nt` on the C store: C is write-once, so caching it evicts the A/B band L2 keeps.
_MX_CSTORE_NT = 2

# A quadrant's accumulators go final early, so its store rides an MFMA shadow; past 2
# the store temporaries outlive an MFMA pair and VGPRs spill.
_MX_EPI_LAP = 2  # quadrants lapped into the last K-iteration (0 = all four at the end)
_MX_WAVE_MAP = 1  # wave_id -> (wave_m, wave_n): 1 walks the N bands per phase group

_MX_BPERM = 1  # pair B rows so a lane's two n-fragments land on adjacent output columns


def _mx_arm_key():
    """Build-affecting module knobs: two settings must not share a compiled launch."""
    return (_MX_EPI_LAP, _MX_WAVE_MAP, _MX_BPERM)


def _mx_pair_rows(row):
    """Involution inside a 32-row B block: row 16*t+m carries B row 2*m+t. Applied to the
    fetched row only, so the LDS row (and with it the bank-swizzle key) is untouched."""
    return (row - row % 32) + (row % 16) * 2 + (row % 32) // 16


def _mx_b_global_swizzle(lane_id, wave_id, K, n_rounds):
    """compute_global_swizzle(preshuffled=False) with the column-pair involution folded into
    the fetched B row. One wave's eight rows stay inside one 32-row block, so the block base
    is wave-uniform and the read set per workgroup is unchanged."""
    offsets = []
    n_waves = fx.block_dim.x // 64
    for rnd in range_constexpr(n_rounds):
        row = lane_id // 8 + wave_id * 8 + rnd * (n_waves * 8)
        col = (lane_id % 8) * 16
        _, c = swizzle_128(row, col)
        offsets.append(_mx_pair_rows(row) * K + c)
    return offsets


class _MxStoreCColPair(StoreCPerTensor):
    """C store for the paired-column B layout: a lane's two n-fragments are adjacent columns,
    so one packed dword per fragment row already fills a write granule -- no cross-lane swap,
    no half-dword extract. The fragment row offset is wave-uniform and rides soffset."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.n_tiles_b == 2, "one fragment pair per row run"
        assert self.col_safe, "a paired column run has no column mask"
        assert not self.trans and not self.beta_is_one, "written for the plain NT epilogue"

    def store(self, c_frag, base_row, base_col, prev=None):
        rsrc = make_row_band_resource(self.c_base, base_row, self.c_rows, self.c_cols, 2)
        lane_off = ((self.lane_id // 16) * 4 * self.c_cols + base_col + (self.lane_id % 16) * 2) * 2
        for ti in range_constexpr(self.n_tiles_a):
            vecs = [Vec(c_frag[self.c_idx_fn(ti, tj)]) for tj in range_constexpr(2)]
            for i in range_constexpr(4):
                lo, hi = vecs[0][i], vecs[1][i]
                if self.elem_fn is not None:
                    lo, hi = self.elem_fn(lo), self.elem_fn(hi)
                buffer_ops.buffer_store(
                    fx.Int32(self._pack(lo, hi)),
                    rsrc,
                    lane_off,
                    cache_modifier=self.store_aux,
                    soffset_bytes=_raw(fx.Int32(2 * (ti * 16 + i)) * self.c_cols),
                    offset_is_bytes=True,
                )


def _build_mxfp8_fold_kernel(k_split, BLOCK_M, BLOCK_N, GROUP_M, group_n, cstore_aux):
    """Build the split-K fold kernel: C[tile] = sum_j WS[j][tile] over the tail tiles.

    Slot ``j * n_tail + t`` is a dense tile block, so only the C write is strided, and tile
    ids continue the gemm's pid walk so both kernels agree without a tile map."""
    VEC = 4  # i32 per lane per access: one dwordx4 == 8 bf16
    ROW_VECS = BLOCK_N // (2 * VEC)
    N_VECS = _FOLD_ROWS * ROW_VECS
    N_SUB = BLOCK_M // _FOLD_ROWS
    TILE_I32 = BLOCK_M * BLOCK_N // 2
    assert N_VECS % _FOLD_BLK == 0

    @flyc.kernel(known_block_size=[_FOLD_BLK, 1, 1])
    def kernel_mxfp8_fold(
        WS: fx.Tensor,
        C: fx.Tensor,
        c_m: fx.Int32,
        c_n: fx.Int32,
        pid_off: fx.Int32,
        n_tail: fx.Int32,
    ):
        tile = fx.block_idx.x // N_SUB
        sub = fx.block_idx.x % N_SUB
        block_m, block_n = block_mn(
            pid_off + tile, ceildiv(c_m, BLOCK_M), ceildiv(c_n, BLOCK_N), GROUP_M, group_n
        )
        ws_rsrc = buffer_ops.create_buffer_resource(
            WS, max_size=False, num_records_bytes=n_tail * k_split * TILE_I32 * 4
        )
        c_rsrc = make_row_band_resource(
            buffer_ops.extract_base_index(C),
            block_m * BLOCK_M + sub * _FOLD_ROWS,
            c_m,
            c_n,
            2,
            span_rows=_FOLD_ROWS,
        )
        base = tile * TILE_I32 + sub * (_FOLD_ROWS * BLOCK_N // 2)
        for i in range_constexpr(N_VECS // _FOLD_BLK):
            v = fx.thread_idx.x + i * _FOLD_BLK
            parts = [
                Vec(
                    buffer_ops.buffer_load(
                        ws_rsrc,
                        base + j * (n_tail * TILE_I32) + v * VEC,
                        vec_width=VEC,
                        dtype=T.i32,
                    )
                )
                for j in range_constexpr(k_split)
            ]
            dws = []
            for d in range_constexpr(VEC):
                halves = []
                for shift in range_constexpr(2):
                    acc = None
                    for j in range_constexpr(k_split):
                        w = fx.Int32(parts[j][d])
                        w = (w << fx.Int32(16)) if shift == 0 else (w & fx.Int32(-65536))
                        f = Vec.from_elements([w], fx.Int32).bitcast(fx.Float32)[0]
                        acc = f if acc is None else acc + f
                    halves.append(acc)
                dws.append(fx.Int32(rocdl.cvt_pk_bf16_f32(halves[0], halves[1])))
            row = v // ROW_VECS
            col = (v % ROW_VECS) * (2 * VEC)
            buffer_ops.buffer_store(
                Vec.from_elements(dws, fx.Int32).ir_value(),
                c_rsrc,
                (row * c_n + block_n * BLOCK_N + col) * 2,
                cache_modifier=cstore_aux,
                offset_is_bytes=True,
            )

    return kernel_mxfp8_fold


def _build_mxfp8_nt_kernel(
    K: int,
    BLOCK_M: int = 256,
    BLOCK_N: int = 256,
    GROUP_M: int = 4,
    group_n: int = 0,  # 0 = 1D GROUP_M swizzle; >0 = 2D N-band (big-N L2 reuse)
    num_xcd: int = 8,
    waves_per_eu: int = 2,
    cbsz: int = 0,  # srcA fp8 format: 0=E4M3, 1=E5M2
    blgp: int = 0,  # srcB fp8 format: 0=E4M3, 1=E5M2
    out_fp16: bool = False,  # fp16 output (else bf16)
    beta_is_one: bool = False,  # epilogue accumulates (C += acc) instead of overwriting
    col_safe: bool = False,  # N % BLOCK_N == 0: every launched tile owns all its columns
    k_split: int = 1,  # >1: tail-round arm, C is the [s, n_tail, BM, BN] partial workspace
    cstore_aux: int = 0,  # C-store cache policy (see _MX_CSTORE_NT); autotuned per shape
    bperm: bool = False,  # B rows interleaved into column pairs (see _MX_BPERM)
):
    BLOCK_K = 128
    assert GROUP_M >= 1

    assert BLOCK_M >= 128 and BLOCK_N >= 256 and BLOCK_M % 128 == 0 and BLOCK_N % 256 == 0
    assert K % BLOCK_K == 0
    # A slice keeps the packed-scale byte order only if its K-group base is PACK-aligned.
    assert (K // BLOCK_K) % (k_split * _MX_SCALE_PACK) == 0 or k_split == 1

    K_ITERS = K // BLOCK_K // k_split

    N_TILES_A = BLOCK_M // 64
    N_TILES_B = BLOCK_N // 128
    N_ACCUMS = N_TILES_A * N_TILES_B
    assert N_ACCUMS > 0

    LDS_BLOCK_M = BLOCK_M // 2
    LDS_BLOCK_N = BLOCK_N // 2

    N_LDS_STEPS_A = LDS_BLOCK_M // 64
    N_LDS_STEPS_B = LDS_BLOCK_N // 64
    N_LDS_ROUNDS = max(N_LDS_STEPS_A, N_LDS_STEPS_B)

    a_lds_size = LDS_BLOCK_M * BLOCK_K
    b_lds_size = LDS_BLOCK_N * BLOCK_K

    # A scale tiles per MFMA wrapper call.
    SA_TILES = N_TILES_A

    @fx.struct
    class SharedStorage:
        A_lds_cur_0: fx.Array[fx.Float8E4M3FN, a_lds_size, 16]
        A_lds_cur_1: fx.Array[fx.Float8E4M3FN, a_lds_size, 16]
        A_lds_next_0: fx.Array[fx.Float8E4M3FN, a_lds_size, 16]
        A_lds_next_1: fx.Array[fx.Float8E4M3FN, a_lds_size, 16]
        B_lds_cur_0: fx.Array[fx.Float8E4M3FN, b_lds_size, 16]
        B_lds_cur_1: fx.Array[fx.Float8E4M3FN, b_lds_size, 16]
        B_lds_next_0: fx.Array[fx.Float8E4M3FN, b_lds_size, 16]
        B_lds_next_1: fx.Array[fx.Float8E4M3FN, b_lds_size, 16]

    @flyc.kernel(known_block_size=[512, 1, 1])
    def kernel_mxfp8_nt(
        A: fx.Tensor,
        B_T: fx.Tensor,
        C: fx.Tensor,
        A_scale: fx.Tensor,
        B_scale: fx.Tensor,
        c_m: fx.Int32,
        c_n: fx.Int32,
        pid_off: fx.Int32,
        pid_span: fx.Int32,
    ):
        F8_IR_t = fx.Float8E4M3FN.ir_type

        n_blocks = ceildiv(c_n, BLOCK_N)

        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        a_cur0 = lds.A_lds_cur_0
        a_cur1 = lds.A_lds_cur_1
        a_next0 = lds.A_lds_next_0
        a_next1 = lds.A_lds_next_1
        b_cur0 = lds.B_lds_cur_0
        b_cur1 = lds.B_lds_cur_1
        b_next0 = lds.B_lds_next_0
        b_next1 = lds.B_lds_next_1

        lane_id = fx.thread_idx.x % 64
        wave_id = fx.thread_idx.x // 64
        wave_m = wave_id // 4
        wave_n = wave_id % 4
        # Keep the SIMD-mate pair on different wave_m so they share B, not the heavier A.
        if const_expr(_MX_WAVE_MAP == 1):
            wave_n = wave_n ^ (wave_m * 2)
        # Tile-minor slice walk, so one XCD still sees the unsplit contiguous tile run.
        num_pid_m = ceildiv(c_m, BLOCK_M)
        if const_expr(k_split > 1):
            slot = xcd_remap_pid(fx.block_idx.x, pid_span * k_split, num_xcd)
            k_slice = slot // pid_span
            pid = pid_off + slot % pid_span
            k_elem_off = k_slice * (K_ITERS * BLOCK_K)
            k_scale_off = k_slice * (K_ITERS // _MX_SCALE_PACK)
        else:
            slot, k_elem_off, k_scale_off = 0, 0, 0
            pid = pid_off + xcd_remap_pid(fx.block_idx.x, pid_span, num_xcd)
        block_m, block_n = block_mn(pid, num_pid_m, n_blocks, GROUP_M, group_n)

        # Fold the per-tile row base (m_row*K, n_row*K) into the 64-bit SRD base; the running
        # k*BLOCK_K offset stays small int32, so there is no 2^31/2^32 cap.
        a_base = arith.index_cast(T.index, block_m * BLOCK_M) * arith.index(K)
        b_base = arith.index_cast(T.index, block_n * BLOCK_N) * arith.index(K)
        a_nrec = (
            arith.index_cast(T.index, c_m) - arith.index_cast(T.index, block_m * BLOCK_M)
        ) * arith.index(K)
        b_nrec = (
            arith.index_cast(T.index, c_n) - arith.index_cast(T.index, block_n * BLOCK_N)
        ) * arith.index(K)
        A0_gl_offset = k_elem_off
        A1_gl_offset = LDS_BLOCK_M * K + k_elem_off
        B0_gl_offset = k_elem_off
        B1_gl_offset = LDS_BLOCK_N * K + k_elem_off

        gA = make_fp8_buffer_tensor_rebased(A, F8_IR_t, a_base, a_nrec)
        gB = make_fp8_buffer_tensor_rebased(B_T, F8_IR_t, b_base, b_nrec)
        a_div = fx.logical_divide(gA, fx.make_layout(1, 1))
        b_div = fx.logical_divide(gB, fx.make_layout(1, 1))

        gl_off_a = compute_global_swizzle(lane_id, wave_id, K, N_LDS_ROUNDS, preshuffled=False)
        if const_expr(bperm):
            gl_off_b = _mx_b_global_swizzle(lane_id, wave_id, K, N_LDS_ROUNDS)
        else:
            gl_off_b = compute_global_swizzle(lane_id, wave_id, K, N_LDS_ROUNDS, preshuffled=False)

        mfma = MfmaScale16x16x128(N_TILES_A, N_TILES_B, cbsz=cbsz, blgp=blgp)

        a_g2s = G2SLoader(a_div, gl_off_a, N_LDS_STEPS_A, F8_IR_t, wave_id)
        b_g2s = G2SLoader(b_div, gl_off_b, N_LDS_STEPS_B, F8_IR_t, wave_id)
        a_s2r = S2RLoader(wave_m, N_TILES_A)
        b_s2r = S2RLoader(wave_n, N_TILES_B)

        sa_s2r = ScaleS2R(A_scale, c_m, K, SA_TILES, pack=_MX_SCALE_PACK)
        sb_s2r = ScaleBComb(B_scale, c_n, K, pack=_MX_SCALE_PACK)  # one dwordx4 = b0+b1 scales
        _out_ty = fx.Float16 if out_fp16 else fx.BFloat16
        # Either layout merges a lane's two n-fragments into one granule-filling run.
        if const_expr(bperm):
            _store_cls = _MxStoreCColPair
        elif const_expr(col_safe and _out_ty is fx.BFloat16 and N_TILES_B % 2 == 0):
            _store_cls = StoreCPerTensorRowN
        else:
            _store_cls = StoreCPerTensor
        # A k_split arm stores to a dense workspace slot, so it needs no column mask.
        if const_expr(k_split > 1):
            store_rows, store_cols = BLOCK_M, BLOCK_N
            store_base = buffer_ops.extract_base_index(C) + arith.index_cast(T.index, slot) * arith.index(
                BLOCK_M * BLOCK_N * 2
            )
        else:
            store_rows, store_cols, store_base = c_m, c_n, None
        store_c = _store_cls(
            None,
            None,
            C,
            store_rows,
            store_cols,
            mfma.idx,
            N_TILES_A,
            N_TILES_B,
            _out_ty,
            col_safe=col_safe,
            c_base=store_base,
            store_aux=0 if k_split > 1 else cstore_aux,
            beta_is_one=beta_is_one,
        )

        # Global row/col bases for the two M / N regions (region1 = +LDS half).
        wave_m_offset = wave_m * (N_TILES_A * 16)
        wave_n_offset = wave_n * (N_TILES_B * 16)
        sa_base0 = fx.Int32(block_m * BLOCK_M + wave_m_offset)
        sa_base1 = sa_base0 + fx.Int32(LDS_BLOCK_M)
        sb_base0 = fx.Int32(block_n * BLOCK_N + wave_n_offset)

        c00_frag = [mfma.zero_value] * N_ACCUMS
        c01_frag = [mfma.zero_value] * N_ACCUMS
        c10_frag = [mfma.zero_value] * N_ACCUMS
        c11_frag = [mfma.zero_value] * N_ACCUMS

        b_g2s.load(b_cur0, B0_gl_offset + 0 * BLOCK_K)
        a_g2s.load(a_cur0, A0_gl_offset + 0 * BLOCK_K)
        b_g2s.load(b_cur1, B1_gl_offset + 0 * BLOCK_K)
        a_g2s.load(a_cur1, A1_gl_offset + 0 * BLOCK_K)

        if wave_m == 1:
            rocdl.s_barrier()

        wait_barrier(N_LDS_STEPS_A + N_LDS_STEPS_B)

        b_g2s.load(b_next0, B0_gl_offset + 1 * BLOCK_K)
        a_g2s.load(a_next0, A0_gl_offset + 1 * BLOCK_K)
        b_g2s.load(b_next1, B1_gl_offset + 1 * BLOCK_K)

        # K_ITERS == 2 skips the main loop and the non-transpose S2RLoader emits no s_waitcnt
        # of its own, so k=1's b_next0/a_next0 have to land here.
        wait_barrier(N_LDS_STEPS_B if K_ITERS == 2 else N_LDS_STEPS_A + 2 * N_LDS_STEPS_B)

        # 1-deep scale prefetch, fired only on the K-group crossing (shared g2s vmcnt).
        sa0 = sa_s2r.load(sa_base0, 0, kbase=k_scale_off)
        sa1 = sa_s2r.load(sa_base1, 0, kbase=k_scale_off)
        sb_all = sb_s2r.load(sb_base0, 0, kbase=k_scale_off)
        sb0, sb1 = sb_all[0:2], sb_all[2:4]

        for k in range_constexpr(K_ITERS - 2):
            mfma.opsel = scale_opsel(k, _MX_SCALE_PACK)  # byte k % PACK of the packed dword
            next_grp = (k + 1) % _MX_SCALE_PACK == 0
            if const_expr(next_grp):
                sa0n = sa_s2r.load(sa_base0, k + 1, kbase=k_scale_off)

            b0_frag = b_s2r.load(b_cur0)
            a0_frag = a_s2r.load(a_cur0)
            a_g2s.load(a_next1, A1_gl_offset + (k + 1) * BLOCK_K)
            rocdl.s_barrier()

            rocdl.s_setprio(1)
            c00_frag = mfma.call(a0_frag, b0_frag, c00_frag, sa0, sb0)
            rocdl.s_setprio(0)
            rocdl.s_barrier()

            b1_frag = b_s2r.load(b_cur1)
            b_g2s.load(b_cur0, B0_gl_offset + (k + 2) * BLOCK_K)
            if const_expr(next_grp):
                sb_alln = sb_s2r.load(sb_base0, k + 1, kbase=k_scale_off)  # one dwordx4 = both B regions
            rocdl.s_barrier()

            rocdl.s_setprio(1)
            c01_frag = mfma.call(a0_frag, b1_frag, c01_frag, sa0, sb1)
            rocdl.s_setprio(0)
            rocdl.s_barrier()

            a1_frag = a_s2r.load(a_cur1)
            a_g2s.load(a_cur0, A0_gl_offset + (k + 2) * BLOCK_K)
            if const_expr(next_grp):
                sa1n = sa_s2r.load(sa_base1, k + 1, kbase=k_scale_off)
            rocdl.s_barrier()

            rocdl.s_setprio(1)
            c10_frag = mfma.call(a1_frag, b0_frag, c10_frag, sa1, sb0)
            rocdl.s_setprio(0)
            rocdl.s_barrier()

            b_g2s.load(b_cur1, B1_gl_offset + (k + 2) * BLOCK_K)
            wait_barrier(2 * N_LDS_STEPS_A + N_LDS_STEPS_B)

            rocdl.s_setprio(1)
            c11_frag = mfma.call(a1_frag, b1_frag, c11_frag, sa1, sb1)
            rocdl.s_setprio(0)
            rocdl.s_barrier()

            a_cur0, a_next0 = a_next0, a_cur0
            a_cur1, a_next1 = a_next1, a_cur1
            b_cur0, b_next0 = b_next0, b_cur0
            b_cur1, b_next1 = b_next1, b_cur1
            if const_expr(next_grp):
                sa0, sa1 = sa0n, sa1n
                sb_all = sb_alln
                sb0, sb1 = sb_all[0:2], sb_all[2:4]

        # Step k = K_ITERS - 2 (sa*/sb* hold scales[K_ITERS-2]; prefetch last iter)
        mfma.opsel = scale_opsel(K_ITERS - 2, _MX_SCALE_PACK)
        last_grp = (K_ITERS - 1) % _MX_SCALE_PACK == 0
        if const_expr(last_grp):
            sa0n = sa_s2r.load(sa_base0, K_ITERS - 1, kbase=k_scale_off)
            sa1n = sa_s2r.load(sa_base1, K_ITERS - 1, kbase=k_scale_off)
            sb_alln = sb_s2r.load(sb_base0, K_ITERS - 1, kbase=k_scale_off)

        b0_frag = b_s2r.load(b_cur0)
        a0_frag = a_s2r.load(a_cur0)
        rocdl.s_barrier()

        rocdl.s_setprio(1)
        c00_frag = mfma.call(a0_frag, b0_frag, c00_frag, sa0, sb0)
        rocdl.s_setprio(0)
        rocdl.s_barrier()

        b1_frag = b_s2r.load(b_cur1)
        rocdl.s_barrier()

        rocdl.s_setprio(1)
        c01_frag = mfma.call(a0_frag, b1_frag, c01_frag, sa0, sb1)
        rocdl.s_setprio(0)
        rocdl.s_barrier()

        a1_frag = a_s2r.load(a_cur1)
        a_g2s.load(a_next1, A1_gl_offset + (K_ITERS - 1) * BLOCK_K)
        rocdl.s_barrier()

        rocdl.s_setprio(1)
        c10_frag = mfma.call(a1_frag, b0_frag, c10_frag, sa1, sb0)
        rocdl.s_setprio(0)
        rocdl.s_barrier()

        b0_frag = b_s2r.load(b_next0)
        rocdl.s_barrier()

        rocdl.s_setprio(1)
        c11_frag = mfma.call(a1_frag, b1_frag, c11_frag, sa1, sb1)
        rocdl.s_setprio(0)
        rocdl.s_barrier()

        a_cur0, a_next0 = a_next0, a_cur0
        a_cur1, a_next1 = a_next1, a_cur1
        b_cur0, b_next0 = b_next0, b_cur0
        b_cur1, b_next1 = b_next1, b_cur1
        if const_expr(last_grp):
            sa0, sa1 = sa0n, sa1n
            sb_all = sb_alln
            sb0, sb1 = sb_all[0:2], sb_all[2:4]

        # Step k = K_ITERS - 1 (sa*/sb* already hold scales[K_ITERS-1])
        mfma.opsel = scale_opsel(K_ITERS - 1, _MX_SCALE_PACK)
        if const_expr(k_split > 1):
            base_row = wave_m_offset
            base_col = wave_n_offset
        else:
            base_row = block_m * BLOCK_M + wave_m_offset
            base_col = block_n * BLOCK_N + wave_n_offset
        lap = 0 if beta_is_one else min(_MX_EPI_LAP, 3)
        a0_frag = a_s2r.load(a_cur0)
        wait_barrier(0)

        rocdl.s_setprio(1)
        c00_frag = mfma.call(a0_frag, b0_frag, c00_frag, sa0, sb0)
        rocdl.s_setprio(0)
        rocdl.s_barrier()

        b1_frag = b_s2r.load(b_cur1)
        if const_expr(lap > 0):
            store_c.store(c00_frag, base_row, base_col)
            rocdl.sched_barrier(0)
        rocdl.s_barrier()

        rocdl.s_setprio(1)
        c01_frag = mfma.call(a0_frag, b1_frag, c01_frag, sa0, sb1)
        rocdl.s_setprio(0)
        rocdl.s_barrier()

        a1_frag = a_s2r.load(a_cur1)
        if const_expr(lap > 1):
            store_c.store(c01_frag, base_row, base_col + LDS_BLOCK_N)
            rocdl.sched_barrier(0)
        rocdl.s_barrier()

        rocdl.s_setprio(1)
        c10_frag = mfma.call(a1_frag, b0_frag, c10_frag, sa1, sb0)
        if const_expr(lap > 2):
            rocdl.s_setprio(0)
            store_c.store(c10_frag, base_row + LDS_BLOCK_M, base_col)
            rocdl.sched_barrier(0)
            rocdl.s_setprio(1)
        c11_frag = mfma.call(a1_frag, b1_frag, c11_frag, sa1, sb1)
        rocdl.s_setprio(0)
        rocdl.s_barrier()

        if const_expr(lap == 0):
            _store_quadrants(
                store_c,
                c00_frag,
                c01_frag,
                c10_frag,
                c11_frag,
                base_row,
                base_col,
                LDS_BLOCK_M,
                LDS_BLOCK_N,
            )
        else:
            if const_expr(lap < 2):
                store_c.store(c01_frag, base_row, base_col + LDS_BLOCK_N)
            if const_expr(lap < 3):
                store_c.store(c10_frag, base_row + LDS_BLOCK_M, base_col)
            store_c.store(c11_frag, base_row + LDS_BLOCK_M, base_col + LDS_BLOCK_N)

    # Bare kernel, not a launch: the fused host stub issues it and sizes grid/value_attrs from these.
    return kernel_mxfp8_nt, BLOCK_M, BLOCK_N, waves_per_eu


_BLOCK_M = 256  # fixed: the A-scale preshuffle layout is bm-dependent (see _MXFP8_NT_CANDIDATES)
_BLOCK_N = 256
_NCU = 256  # co-resident workgroups per dispatch round (1 WG/CU at this tile's occupancy)
_PRESHUF_BLK = 256  # preshuffle kernel block size (matches build_preshuffle_ab_kernel)

# (K, bm, gm, xcd, gn, cbsz, blgp, out_fp16, beta1) -> launch_mxfp8_fused (preshuffle+gemm jit)
_MXFP8_FUSED_CACHE: dict = {}
# (M, N, K128, device, stream) -> (a_sp, b_sp, a_blocks, a_ngrp, b_ngrp). The preshuffle writes
# a_sp/b_sp and the gemm reads them in stream order, so same-stream reuse is safe.
_MXFP8_WS_CACHE: dict = {}
_MXFP8_SPLIT_WS_CACHE: dict = {}  # (tail, k_split, device, stream, dtype) -> bf16 partial workspace

# NT tile candidates (BLOCK_M, GROUP_M, num_xcd); BLOCK_N fixed 256. BLOCK_M stays 256: the
# A-scale preshuffle layout is bm-dependent and built once per launch, so autotune only sweeps
# knobs that leave it unchanged (GROUP_M / num_xcd / group_n).
_MXFP8_NT_CANDIDATES = [
    (256, 4, 8),
    (256, 8, 8),
]


def _mx_col_safe(N):
    """Every launched tile owns all BLOCK_N of its columns, so the epilogue needs no column
    mask and the row-merged (64B-request) store is legal. Compile-time: it selects StoreC."""
    return N % _BLOCK_N == 0


def _mx_ksplit(M, N, K, out_dtype, beta_is_one):
    """Tail-round split-K plan -> (k_split, head_tiles, tail_tiles); k_split=1 = one arm.

    Both halves must be whole rounds, or the split just moves the idle round elsewhere, and
    the slice must stay long enough that the extra epilogue is not what it buys."""
    if beta_is_one or out_dtype != torch.bfloat16:
        return 1, 0, 0  # the fold writes bf16 and cannot re-add the caller's C
    if M % _BLOCK_M or N % _BLOCK_N:
        return 1, 0, 0  # a mask-clamped tile does not fit the dense tile-block workspace
    tiles = (M // _BLOCK_M) * (N // _BLOCK_N)
    tail = tiles % _NCU
    k_iters = K // 128
    if tail == 0 or tiles <= _NCU or _NCU % tail:
        return 1, 0, 0
    k_split = _NCU // tail
    if k_iters % (k_split * _MX_SCALE_PACK) or k_iters // k_split < _MX_SPLIT_MIN_KI:
        return 1, 0, 0
    return k_split, tiles - tail, tail


def _mx_nt_gn_cands(N):
    """NT 2D N-band widths for the stage-2 sweep (the gn=0 seed is raced separately). A band
    needs >= 2*gn N-blocks to create the cross-tile B reuse it exists for."""
    n_blocks = (N + _BLOCK_N - 1) // _BLOCK_N
    # No gn=32: its one win needs a tile that stage 1 does not pick at gn=0.
    return [g for g in (4, 8, 16) if n_blocks >= 2 * g]


def _compile_mxfp8_fused(
    K,
    bm,
    gm,
    xcd,
    gn=0,
    cbsz=0,
    blgp=0,
    out_fp16=False,
    beta_is_one=False,
    col_safe=False,
    k_split=1,
    cstore_aux=0,
):
    """Build the fused @flyc.jit ``launch_mxfp8_fused``: one host stub enqueues the scale
    preshuffle (raw E8M0 -> caller-owned a_sp / b_sp) and then the NT GEMM on the same stream.
    gn = 2D N-band width (0 = 1D swizzle); cbsz/blgp = per-operand fp8 format (0=E4M3, 1=E5M2).

    Never name this jit (or any module global) ``launch``: ``.launch()`` lands in co_names and
    the cache-key dependency walker would resolve it back to this JitFunction (infinite recursion).
    """
    K128 = K // 128
    # The split plan and the fold both decode tile ids at _BLOCK_M.
    assert k_split == 1 or bm == _BLOCK_M
    # Gated: a paired run has no column mask, and the scales must match the gemm's layout.
    bperm = bool(_MX_BPERM) and col_safe and not beta_is_one
    pre_kern, n_kt = build_preshuffle_ab_kernel(K128, pack=_MX_SCALE_PACK, widen=True, bperm=bperm)
    _nt = dict(
        K=K,
        BLOCK_M=bm,
        BLOCK_N=_BLOCK_N,
        GROUP_M=gm,
        group_n=gn,
        num_xcd=xcd,
        cbsz=cbsz,
        blgp=blgp,
        out_fp16=out_fp16,
        cstore_aux=cstore_aux,
        bperm=bperm,
    )
    gemm_kern, BM, BN, wpe = _build_mxfp8_nt_kernel(beta_is_one=beta_is_one, col_safe=col_safe, **_nt)
    split_kern = _build_mxfp8_nt_kernel(col_safe=True, k_split=k_split, **_nt)[0] if k_split > 1 else None
    fold_kern = _build_mxfp8_fold_kernel(k_split, bm, _BLOCK_N, gm, gn, cstore_aux) if k_split > 1 else None

    @flyc.jit
    def launch_mxfp8_fused(
        a8: fx.Tensor,
        b8: fx.Tensor,
        C: fx.Tensor,
        a_raw: fx.Tensor,
        b_raw: fx.Tensor,
        a_sp: fx.Tensor,
        b_sp: fx.Tensor,
        ws: fx.Tensor,
        c_m: fx.Int32,
        c_n: fx.Int32,
        a_blocks: fx.Int32,
        a_ngrp: fx.Int32,
        b_ngrp: fx.Int32,
        head: fx.Int32,
        tail: fx.Int32,
        stream: fx.Stream,
    ):
        # 1) scale preshuffle (raw E8M0 -> broadcast int32 in a_sp/b_sp)
        pre_kern(a_raw, b_raw, a_sp, b_sp, c_m, c_n, a_blocks, a_ngrp, b_ngrp).launch(
            grid=(a_blocks + b_ngrp * n_kt, 1, 1), block=(_PRESHUF_BLK, 1, 1), stream=stream
        )
        # 2) NT GEMM over tile ids [0, head) -- the whole grid unless the plan splits.
        gemm_kern(
            a8,
            b8,
            C,
            a_sp,
            b_sp,
            c_m,
            c_n,
            fx.Int32(0),
            head,
            value_attrs=make_value_attrs(wpe, 0, "512,512"),
        ).launch(grid=(head, 1, 1), block=(512, 1, 1), stream=stream)
        if const_expr(k_split > 1):
            # 3) tail round at K/k_split, one full round of workgroups into ws
            split_kern(
                a8,
                b8,
                ws,
                a_sp,
                b_sp,
                c_m,
                c_n,
                head,
                tail,
                value_attrs=make_value_attrs(wpe, 0, "512,512"),
            ).launch(grid=(tail * k_split, 1, 1), block=(512, 1, 1), stream=stream)
            fold_kern(ws, C, c_m, c_n, head, tail).launch(
                grid=(tail * (bm // _FOLD_ROWS), 1, 1), block=(_FOLD_BLK, 1, 1), stream=stream
            )

    return launch_mxfp8_fused


def _get_mxfp8_fused_launch(
    K,
    bm,
    gm,
    xcd,
    gn=0,
    cbsz=0,
    blgp=0,
    out_fp16=False,
    beta_is_one=False,
    col_safe=False,
    k_split=1,
    cstore_aux=0,
):
    fk = (
        K,
        bm,
        gm,
        xcd,
        gn,
        cbsz,
        blgp,
        out_fp16,
        beta_is_one,
        col_safe,
        k_split,
        cstore_aux,
    ) + _mx_arm_key()
    launch = _MXFP8_FUSED_CACHE.get(fk)
    if launch is None:
        launch = _compile_mxfp8_fused(
            K, bm, gm, xcd, gn, cbsz, blgp, out_fp16, beta_is_one, col_safe, k_split, cstore_aux
        )
        _MXFP8_FUSED_CACHE[fk] = launch
    return launch


def _get_mx_workspace(M, N, K128, device, stream):
    """Caller-owned flat int32 scale workspace (a_sp/b_sp) + the preshuffle launch dims, cached
    by (M, N, K128, device, stream). A has cdiv(M,64) groups, B cdiv(N,256)*4; a partial last A
    group covers general M (the preshuffle zeroes its OOB rows and StoreC drops them)."""
    key = (M, N, K128, device, stream)
    e = _MXFP8_WS_CACHE.get(key)
    if e is None:
        a_ngrp = ceildiv(M, 64)
        b_ngrp = ((N + 255) // 256) * 4
        a_blocks = a_ngrp * ceildiv(K128, _PRESHUF_KT)
        k128p = ceildiv(K128, _MX_SCALE_PACK)  # one dword per PACK K-groups
        a_sp = torch.empty(a_ngrp * k128p * 256, dtype=torch.int32, device=device)
        b_sp = torch.empty(b_ngrp * k128p * 256, dtype=torch.int32, device=device)
        e = (a_sp, b_sp, a_blocks, a_ngrp, b_ngrp)
        _MXFP8_WS_CACHE[key] = e
    return e


def _get_mx_split_ws(tail, k_split, device, stream, out_dtype):
    """Caller-owned split-K partial workspace: [k_split, tail, BLOCK_M, BLOCK_N] out_dtype
    (k_split*tail == NCU, so one round's worth of tile blocks). Written by the tail arm and
    read by the fold in stream order, same reuse rule as the scale workspace."""
    key = (tail, k_split, device, stream, out_dtype)
    ws = _MXFP8_SPLIT_WS_CACHE.get(key)
    if ws is None:
        ws = torch.empty(k_split * tail * _BLOCK_M * _BLOCK_N, dtype=out_dtype, device=device)
        _MXFP8_SPLIT_WS_CACHE[key] = ws
    return ws


# Every step holds its candidates to reproducible output: several configs the race can reach
# return different bytes on every call, and which settings are safe differs by shape (#570).
_MX_TILE = Race.interleaved(1, 20, warmup=2, visit=(0, 3), check="reproducible")
# adopt a band only if it beats the re-measured seed by >1.5%
_MX_BAND = Race.interleaved(4, 20, visit=(2, 3), margin=0.985, check="reproducible")
# The C-store and split plans, both shape-dependent in sign; the split plan is judged against the
# time the C-store step's pick won with, not a re-measure.
_MX_TAIL = Race.interleaved(3, 20, visit=(2, 3), margin=0.99, check="reproducible")
_MX_RACES = (_MX_TILE, _MX_BAND, _MX_TAIL)
# The same steps held to a finite output alone: the last resort when nothing reproduces.
_MX_UNCHECKED = tuple(r.checked_by("finite") for r in _MX_RACES)
_GEMM_MXFP8 = Op("gemm_mxfp8", compile=Compile(once=True))
# Untuned: the leading tile at NT's native 1D swizzle, one arm, default C-store policy.
_MXFP8_BASE = _MXFP8_NT_CANDIDATES[0] + (0, 1, 0)


def _mxfp8_entry(M, N, K, cfg, out_dtype, cbsz, blgp, beta_is_one, tuned=True):
    """``cfg``'s launch, ``extra`` = its (k_split, head, tail) plan. A split pick runs as one arm
    under beta=1 or a non-bf16 C (see _mx_ksplit)."""
    bm, gm, xcd, gn, ks, aux = cfg
    ks, head, tail = _mx_ksplit(M, N, K, out_dtype, beta_is_one) if ks > 1 else (1, 0, 0)
    if ks == 1:
        head = ceildiv(M, bm) * ceildiv(N, _BLOCK_N)
    launch = _get_mxfp8_fused_launch(
        K,
        bm,
        gm,
        xcd,
        gn,
        cbsz=cbsz,
        blgp=blgp,
        out_fp16=out_dtype == torch.float16,
        beta_is_one=beta_is_one,
        col_safe=_mx_col_safe(N),
        k_split=ks,
        cstore_aux=aux,
    )
    return Entry(cfg, launch, tuned=tuned, extra=(ks, head, tail))


def _mx_space(tiles, N, bks, races, skip_gn=()):
    """The NT search from the stage-1 ``tiles``: (BM, GM, XCD), then group_n bands against the
    re-measured incumbent (never a width in ``skip_gn``), then the C-store policy, then the
    tail-round split-K plan against one arm at a margin. ``races``: the (tile, band, tail) races."""
    tile, band, tail = races

    def gn_stage(prev):
        # The incumbent is re-measured with the bands' estimator.
        widths = sorted(set([0] + _mx_nt_gn_cands(N)) - {prev[3]} - set(skip_gn))
        return [prev] + [prev[:3] + (gn,) + prev[4:] for gn in widths]

    return staged(
        Refine(lambda _prev: tiles, tile),
        Refine(gn_stage if _mx_nt_gn_cands(N) else (lambda prev: [prev]), band),
        Refine(lambda prev: [prev, prev[:5] + (_MX_CSTORE_NT,)], tail, reuse_winner_time=True),
        Refine(
            lambda prev: [prev] + ([prev[:4] + (bks, prev[5])] if bks > 1 else []),
            tail,
            reuse_winner_time=True,
        ),
        base=_MXFP8_BASE,
    )


def _mx_search(pick, key, M, N, K, bks):
    """The NT pick, ``pick(key, space) -> Pick``, held to reproducible output with two fallbacks
    (#570): a correct config at any speed beats a fast one that returns different bytes, and a
    warning beats refusing a shape that ran before the check existed.

    1. The tiles at the seed width group_n=0.
    2. None of them reproduces: the tiles at every other width, under ``key + ("any-width",)``.
       The band step then starts from that pick and never goes back to the seed width.
    3. Nothing reproduces at any width: warn, and race on finite output alone under
       ``key + ("unchecked",)``, as the kernel did before the check existed.

    A key where nothing at the seed width builds raises at once. Steps 2 and 3 run under their
    own keys, so a width that does not build is remembered there and step 3 still runs."""
    tiles = [c + (0, 1, 0) for c in _MXFP8_NT_CANDIDATES]
    try:
        return pick(key, _mx_space(tiles, N, bks, _MX_RACES))
    except NoWorkingConfig as ex:
        if not ex.data_dependent:
            raise
    widths = _mx_nt_gn_cands(N)
    if widths:
        wide = [t[:3] + (gn,) + t[4:] for gn in widths for t in tiles]
        try:
            return pick(key + ("any-width",), _mx_space(wide, N, bks, _MX_RACES, skip_gn=(0,)))
        except NoWorkingConfig:
            pass
    warnings.warn(
        f"mxfp8 NT: no reproducible config for M={M} N={N} K={K}; ranking on time alone, "
        "so this shape may return different results from call to call",
        RuntimeWarning,
        stacklevel=2,
    )
    return pick(key + ("unchecked",), _mx_space(tiles, N, bks, _MX_UNCHECKED))


def _pick_mxfp8(key, out, args_for, M, N, K, out_dtype, cbsz, blgp):
    """The beta=0 Entry for (M,N,K), its config (BLOCK_M, GROUP_M, num_xcd, group_n, k_split,
    cstore_aux); ``args_for(entry, c)`` gives an entry's launch arguments onto C ``c``. Candidates
    time the fused stub; the preshuffle is a constant offset, so the gemm ranking holds.

    The search and its fallbacks: see _mx_space and _mx_search."""
    # Race on a scratch: each candidate is run dozens of times, which a beta=1 build
    # would fold into the caller's accumulation buffer.
    scratch = {}

    def probe():
        scratch["out"] = torch.empty_like(out)
        return [Point((), scratch["out"])]

    def build(cfg):
        e = _mxfp8_entry(M, N, K, cfg, out_dtype, cbsz, blgp, False)
        args = args_for(e, scratch["out"])
        return lambda: e.raw(*args)

    def pick(k, space):
        return _GEMM_MXFP8.pick(k, space, probe=Synthetic(probe), build=build)

    res = _mx_search(pick, key, M, N, K, _mx_ksplit(M, N, K, out_dtype, False)[0])
    return _mxfp8_entry(M, N, K, res.config, out_dtype, cbsz, blgp, False, tuned=res.tuned)


def gemm_mxfp8_flydsl_kernel(
    a: torch.Tensor,
    a_scale: torch.Tensor,
    b: torch.Tensor,
    b_scale: torch.Tensor,
    *,
    trans_a: bool = False,
    trans_b: bool = True,
    out_dtype: torch.dtype = torch.bfloat16,
    beta: float = 0.0,
    out: "torch.Tensor | None" = None,
) -> torch.Tensor:
    """MXFP8 (per-1x32 E8M0 block-scaled) dense GEMM, gfx950. Returns C [M,N].

    NT only (trans_a=False, trans_b=True): A [M,K], B [N,K], C = a @ b^T.

    ``beta=1.0`` accumulates into ``out`` (``out += a @ b^T``) in the epilogue instead
    of overwriting it, and therefore requires ``out``.

    ``a_scale`` / ``b_scale`` are the raw E8M0 block scales ([M, K//32] / [N, K//32],
    uint8/e8m0); the launch preshuffles them itself in the same stream-ordered dispatch.

    Constraints: K % 128 == 0 and K >= 256; M >= 1; N >= 1 (general M/N).
    """
    assert a.dim() == 2 and b.dim() == 2, "a, b must be 2D"
    assert out_dtype in (torch.bfloat16, torch.float16), "mxfp8 FlyDSL store emits bf16/fp16"
    # Per-operand fp8 format -> MFMA cbsz(srcA)/blgp(srcB): 0=E4M3, 1=E5M2.
    cbsz = 1 if a.dtype == torch.float8_e5m2 else 0
    blgp = 1 if b.dtype == torch.float8_e5m2 else 0

    if (not trans_a) and trans_b:
        M, K = a.shape
        N, Kb = b.shape
    else:
        raise NotImplementedError(
            "mxfp8 FlyDSL GEMM is NT only (trans_a=False, trans_b=True); "
            f"got trans_a={trans_a}, trans_b={trans_b}."
        )
    assert K == Kb, f"K mismatch: a {a.shape}, b {b.shape}"
    assert K % 128 == 0 and K >= 256, f"K must be a multiple of 128 and >= 256, got {K}"
    assert M >= 1, f"M must be >= 1, got {M}"
    assert N >= 1, f"N must be >= 1, got {N}"
    assert a_scale.shape[0] == M and b_scale.shape[0] == N, "scale rows must match a/b rows"
    assert a_scale.shape[1] == K // 32 and b_scale.shape[1] == K // 32, "raw E8M0 scales are [dim, K//32]"

    K128 = K // 128
    # Raw E8M0 bytes viewed little-endian as flat int32 [dim*K128].
    a_raw = (a_scale if a_scale.is_contiguous() else a_scale.contiguous()).view(torch.int32).reshape(-1)
    b_raw = (b_scale if b_scale.is_contiguous() else b_scale.contiguous()).view(torch.int32).reshape(-1)
    out = resolve_accum_out(out, beta, (M, N), a.device, out_dtype)
    beta_is_one = beta == 1.0
    # Keep 2D views: FlyDSL marshals each shape dim as int32, so a flat [M*K] view overflows
    # past 2^31. The kernel re-bases the i64 SRD, so the 2D shape is just metadata.
    a8 = a.contiguous().view(torch.int8)
    b8 = b.contiguous().view(torch.int8)
    stream = torch.cuda.current_stream()
    a_sp, b_sp, a_blocks, a_ngrp, b_ngrp = _get_mx_workspace(M, N, K128, a.device, stream)

    def args_for(entry, c):
        ks, head, tail = entry.extra
        ws = _get_mx_split_ws(tail, ks, c.device, stream, out_dtype) if ks > 1 else c
        return (a8, b8, c, a_raw, b_raw, a_sp, b_sp, ws, M, N, a_blocks, a_ngrp, b_ngrp, head, tail, stream)

    # The arm knobs are in the key: a config is only timed on the arm it runs.
    key = (M, N, K, out_dtype, cbsz, blgp) + _mx_arm_key()
    entry = _GEMM_MXFP8.entry(key, lambda: _pick_mxfp8(key, out, args_for, M, N, K, out_dtype, cbsz, blgp))
    if beta_is_one:
        entry = _GEMM_MXFP8.derived(
            key, True, entry, lambda e: _mxfp8_entry(M, N, K, e.config, out_dtype, cbsz, blgp, True, e.tuned)
        )
    _GEMM_MXFP8.launch(entry, args_for(entry, out))
    return out
