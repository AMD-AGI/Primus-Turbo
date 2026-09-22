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

Structurally parallels the tensorwise FP8 path in ``gemm_fp8_kernel.py``; the
difference vs that kernel:

  * tensorwise applies a single per-row (A) / per-col (B) FP32 scale in the
    epilogue, with the MFMA run un-scaled (identity scale operand).
  * mxfp8 carries a per-32-element-K-block E8M0 scale that MUST be fed to the
    ``v_mfma_scale_f32_16x16x128_f8f6f4`` instruction per K-iteration. The
    epilogue therefore becomes a plain FP32->BF16 store (all scaling already
    folded into the accumulator by the MMA).

Scale operand semantics (gfx950): the MMA takes one i32 scale per operand,
holding 4 packed E8M0 bytes -- one byte per 32-K block. A single
16x16x128 MFMA spans K=128 == 4 micro-blocks, so exactly one i32 scale per
(row/col tile, K-iteration).

Scale tensor layout expected by this kernel (passed pre-packed from host):
  A_scale: int32 [M, K // 128]   (each i32 == 4 consecutive E8M0 bytes of a row)
  B_scale: int32 [N, K // 128]
i.e. the raw uint8 E8M0 [DIM, K//32] viewed little-endian as int32.
"""

import torch

# isort: off
# Shared fp8 GEMM primitives from primus_turbo/flydsl/utils/gemm_helper.py (the
# tensorwise FlyDSL backend, #356). Must be importable as module globals
# (@flyc.kernel needs them as globals, not closure cells). NT only (compute-only
# PR), so the NN/TN transpose loaders (S2RLoaderTr / swizzle_nn) are not imported.
from primus_turbo.flydsl.utils.gemm_helper import (
    block_mn,
    build_preshuffle_ab_kernel,
    compile_with_scratch_out,
    compute_global_swizzle,
    G2SLoader,
    make_fp8_buffer_tensor_rebased,
    make_row_band_resource,
    make_value_attrs,
    MfmaScale16x16x128,
    _PRESHUF_KT,
    resolve_accum_out,
    _robust_time,
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

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith, buffer_ops, const_expr, range_constexpr, rocdl
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

    # scale-tile fanout per MFMA wrapper call (A sub-tiles / B sub-tiles per wave).
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
        # 1D GROUP_M super-row swizzle (group_n=0) or 2D N-band (group_n>0, big-N L2
        # reuse: cuts the B re-stream). XCD-aware remap. See block_mn / xcd_remap_pid.
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

        # i64 input re-base (same as tensorwise NT): fold the per-tile row base
        # (m_row*K, n_row*K) into the SRD base in T.index (64-bit); A/B_T are
        # K-contiguous (foldable), so the running k*BLOCK_K offset stays small int32
        # -> no 2^31/2^32 cap. Output StoreCPlain already re-bases per band in i64.
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

        # 2x2 config of accumulators
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

        # Same zero-iteration hazard as the dense fp8 NT kernel, with no incidental cover:
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

    # Bare kernel (NOT a launch): the fused factory issues it + the preshuffle kernel
    # from a single @flyc.jit host stub. BLOCK_M/BLOCK_N/waves_per_eu are returned so
    # that stub can size the grid + value_attrs.
    return kernel_mxfp8_nt, BLOCK_M, BLOCK_N, waves_per_eu


_BLOCK_M = 256  # fixed: the A-scale preshuffle layout is bm-dependent (see _MXFP8_NT_CANDIDATES)
_BLOCK_N = 256
_NCU = 256  # co-resident workgroups per dispatch round (1 WG/CU at this tile's occupancy)
_PRESHUF_BLK = 256  # preshuffle kernel block size (matches build_preshuffle_ab_kernel)

# (K, bm, gm, xcd, gn, cbsz, blgp, out_fp16, beta1) -> launch_mxfp8_fused (preshuffle+gemm jit)
_MXFP8_FUSED_CACHE: dict = {}
_MXFP8_AT_CACHE: dict = {}  # (M,N,K,bm,gm,xcd,gn,cbsz,blgp,out_fp16,beta1) -> [raw_launch, compiled_or_None]
# (M, N, K128, device, stream) -> (a_sp, b_sp, a_blocks, a_ngrp, b_ngrp). Caller-owned
# scale workspace (turbo-style): the fused stub's preshuffle writes a_sp/b_sp then the
# gemm reads them, in stream order, so reuse across same-shape calls on one stream is safe.
_MXFP8_WS_CACHE: dict = {}
_MXFP8_SPLIT_WS_CACHE: dict = {}  # (tail, k_split, device, stream, dtype) -> bf16 partial workspace

# Per-shape NT autotune candidates (BLOCK_M, GROUP_M, num_xcd); BLOCK_N fixed 256.
# BLOCK_M=128 doubles the tiles (fills the CUs on skinny/small shapes), 256 wins big
# square / B-streaming; GROUP_M is the per-XCD L2-reuse super-block depth.
# BLOCK_M fixed at 256 (n_tiles_a = 256//64 = 4): the A-scale preshuffle layout is
# bm-dependent and the fused stub preshuffles the raw E8M0 scales to int32 (one fixed
# config) before the gemm, so the gemm cannot re-pack per candidate -> BLOCK_M constant.
# autotune sweeps only GROUP_M / num_xcd / group_n (none of which change the layout).
_MXFP8_NT_CANDIDATES = [
    (256, 4, 8),
    (256, 8, 8),
]


_MXFP8_AUTOTUNE_CACHE: dict = {}  # (M,N,K,out_dtype,cbsz,blgp) -> (BLOCK_M, GROUP_M, num_xcd, group_n)


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
    """NT 2D N-band candidate widths for the autotune stage-2 sweep. The seed band
    (gn=0, NT's 1D swizzle) is measured separately as the baseline, so the final
    pick can never regress. Only offer a band when there are >= 2*gn 256-col
    N-blocks (else the band can't create the cross-tile B reuse it exists for).
    Winners are shape-dependent (NT: 7B GateUp gn16, 70B QKV gn8/16), so the
    per-shape bench picks rather than a single heuristic."""
    n_blocks = (N + _BLOCK_N - 1) // _BLOCK_N
    # gn=32 was probed and dropped: its only win (NT 7B_GateUp +1.7% over gn16) is
    # coupled to tile (256,4), but stage-1 picks the tile at the seed band and lands
    # on (256,8) there, so the gn=32 win isn't reliably reachable — not worth the
    # extra autotune compile on every N>=16384 shape. {4,8,16} captures the robust
    # wins. (A fuller tile x gn cross-sweep could reach it but costs far more.)
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
    """Build the turbo-style fused @flyc.jit ``launch_mxfp8_fused``: ONE host stub
    that enqueues the A+B scale preshuffle kernel and then the NT mxfp8 GEMM kernel
    on the same stream (single Python dispatch, no separate preshuffle launch, no
    CPU sync). The preshuffle repacks raw E8M0 (int32-viewed) into the caller-owned
    a_sp / b_sp workspace; the gemm reads it in stream order. gn = autotuned 2D
    N-band width (0 = 1D swizzle); cbsz/blgp = per-operand fp8 format (0=E4M3,1=E5M2).

    NOTE: never name this jit (or any module global) ``launch`` -- the ``.launch()``
    method attribute lands in co_names and the cache-key dependency walker would
    resolve a module-global ``launch`` back to this JitFunction (infinite recursion).
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
    """Caller-owned scale workspace (a_sp/b_sp) + the preshuffle launch dims, cached
    by (M, N, K128, device, stream). a_sp/b_sp are flat int32 buffers the fused stub's
    preshuffle writes and the gemm reads; sizing mirrors the A (layout-1) / combined-B
    (layout-3) repack (A: cdiv(M,64) groups; B: cdiv(N,256)*4 groups; each group*K128*256
    i32, packed K-groups). cdiv for A (not M//64): a partial last 64-row group covers
    general (non-64) M, the preshuffle masks its OOB rows to 0 and StoreC drops them."""
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


def _autotune_mxfp8(
    a8, b8, out_view, a_raw, b_raw, a_sp, b_sp, M, N, K, a_blocks, a_ngrp, b_ngrp, out_dtype, cbsz=0, blgp=0
):
    """First-call micro-bench of the candidates for (M,N,K); cache the fastest cfg by
    shape. Returns (BLOCK_M, GROUP_M, num_xcd, group_n, k_split, cstore_aux). Each candidate times the
    FUSED stub (preshuffle + gemm): the preshuffle config is fixed (same K128) so it is a
    constant offset across candidates and the gemm-config ranking is preserved.

    Two-stage for NT: stage 1 picks (BM,GM,XCD) at group_n=0 (1D swizzle); stage 2
    fixes that tile and sweeps the 2D N-band width group_n. gn=0 is measured in
    stage 1, so the staged pick can never regress vs the gn-less NT path — it only
    captures the big-/mid-N L2-reuse win on shapes where a band helps. Stage 3 races the
    tail-round split-K plan against the single arm at a protection margin, so a shape whose
    partial-sum traffic outweighs the refilled round keeps one arm."""
    key = (M, N, K, out_dtype, cbsz, blgp)
    _ofp16 = out_dtype == torch.float16
    cached = _MXFP8_AUTOTUNE_CACHE.get(key)
    if cached is not None:
        return cached
    cands = _MXFP8_NT_CANDIDATES
    stream = torch.cuda.current_stream()
    # Race on a scratch: each candidate is run dozens of times, which a beta=1 build
    # would fold into the caller's accumulation buffer.
    out_view = torch.empty_like(out_view)

    def _time_cfg(bm, gm, xcd, gn, ks=1, aux=0):
        try:
            ks, head, tail = _mx_ksplit(M, N, K, out_dtype, False) if ks > 1 else (1, 0, 0)
            ws, head = (
                (_get_mx_split_ws(tail, ks, out_view.device, stream, out_dtype), head)
                if ks > 1
                else (out_view, ceildiv(M, bm) * ceildiv(N, _BLOCK_N))
            )
            launch = _get_mxfp8_fused_launch(
                K,
                bm,
                gm,
                xcd,
                gn,
                cbsz=cbsz,
                blgp=blgp,
                out_fp16=_ofp16,
                col_safe=_mx_col_safe(N),
                k_split=ks,
                cstore_aux=aux,
            )
            args = (
                a8,
                b8,
                out_view,
                a_raw,
                b_raw,
                a_sp,
                b_sp,
                ws,
                M,
                N,
                a_blocks,
                a_ngrp,
                b_ngrp,
                head,
                tail,
                stream,
            )
            launch(*args)
            torch.cuda.synchronize()
            if not torch.isfinite(out_view.reshape(-1)[:1024].float()).all().item():
                return float("inf")
            return _robust_time(launch, args, warmup=2, reps=3, iters=20)
        except Exception:
            return float("inf")

    # Stage 1: best (BLOCK_M, GROUP_M, num_xcd) at the SEED band width gn=0 (NT's
    # native 1D swizzle); stage 2 sweeps the 2D N-band width on top.
    seed_gn = 0
    best_us = float("inf")
    best = None
    for bm, gm, xcd in cands:
        us = _time_cfg(bm, gm, xcd, seed_gn)
        if us < best_us:
            best_us = us
            best = (bm, gm, xcd, seed_gn)
    if best is None:
        raise RuntimeError(
            f"mxfp8 autotune: all candidates failed for M={M} N={N} K={K}. "
            f"Check FlyDSL compilation (MLIR dialect, gfx target, OOM)."
        )
    # Stage 2: sweep 2D N-band width; adopt only if it beats re-measured seed by >1.5%.
    # gn=0 wins by default (bgn=seed_gn); sweep only visits non-seed bands.
    gn_cands = _mx_nt_gn_cands(N)
    if gn_cands:
        bm, gm, xcd, _ = best

        def _robust(gn):
            return min(_time_cfg(bm, gm, xcd, gn) for _ in range(4))

        seed_us = _robust(seed_gn)  # re-measured seed baseline (same estimator as the bands)
        bgn, bus = seed_gn, seed_us
        for gn in sorted(set([0] + gn_cands) - {seed_gn}):
            us = _robust(gn)
            if us < bus and us < seed_us * 0.985:
                bgn, bus = gn, us
        best = (bm, gm, xcd, bgn)
    # Stage 3: C-store policy then split plan; both are shape-dependent in sign, so raced.
    bm, gm, xcd, gn = best
    seed_us = min(_time_cfg(bm, gm, xcd, gn) for _ in range(3))
    baux, bus = 0, seed_us
    nt_us = min(_time_cfg(bm, gm, xcd, gn, aux=_MX_CSTORE_NT) for _ in range(3))
    if nt_us < seed_us * 0.99:
        baux, bus = _MX_CSTORE_NT, nt_us
    bks = _mx_ksplit(M, N, K, out_dtype, False)[0]
    if bks > 1 and min(_time_cfg(bm, gm, xcd, gn, ks=bks, aux=baux) for _ in range(3)) >= bus * 0.99:
        bks = 1
    best = (bm, gm, xcd, gn, bks, baux)
    _MXFP8_AUTOTUNE_CACHE[key] = best
    return best


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

    ``a_scale`` / ``b_scale`` are the RAW E8M0 block scales ([M, K//32] / [N, K//32],
    uint8/e8m0): this kernel preshuffles them to the broadcast int32 layout itself,
    fused into the GEMM launch (turbo-style single dispatch -- one @flyc.jit host stub
    enqueues the A+B preshuffle kernel then the gemm kernel on the same stream, with a
    caller-owned int32 workspace, so there is no separate preshuffle launch and no CPU
    sync). The quant emits only raw E8M0 scales.

    Constraints: K % 128 == 0 and K >= 256; M >= 1; N >= 1 (general M/N: the A-scale
    preshuffle / gemm size A by cdiv(M,64) and bound the partial tail by the StoreC clamp).
    """
    assert a.dim() == 2 and b.dim() == 2, "a, b must be 2D"
    assert out_dtype in (torch.bfloat16, torch.float16), "mxfp8 FlyDSL store emits bf16/fp16"
    # Per-operand fp8 format -> MFMA cbsz(srcA)/blgp(srcB): 0=E4M3, 1=E5M2.
    cbsz = 1 if a.dtype == torch.float8_e5m2 else 0
    blgp = 1 if b.dtype == torch.float8_e5m2 else 0
    out_fp16 = out_dtype == torch.float16

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
    # Raw E8M0 byte scales viewed little-endian as flat int32 [dim*K128] (the preshuffle
    # kernel reads grow*K128+gk); contiguous straight out of quant, copy only if a view broke it.
    a_raw = (a_scale if a_scale.is_contiguous() else a_scale.contiguous()).view(torch.int32).reshape(-1)
    b_raw = (b_scale if b_scale.is_contiguous() else b_scale.contiguous()).view(torch.int32).reshape(-1)
    out = resolve_accum_out(out, beta, (M, N), a.device, out_dtype)
    beta_is_one = beta == 1.0
    # Keep 2D int8 views (NOT flat 1D): FlyDSL marshals each shape dim as int32, so a
    # 1D [M*K] view overflows when M*K > 2^31. The kernel addresses via i64 SRD re-base
    # (extract_base_index reads only the base ptr), so the 2D shape is just metadata.
    a8 = a.contiguous().view(torch.int8)
    b8 = b.contiguous().view(torch.int8)
    stream = torch.cuda.current_stream()
    a_sp, b_sp, a_blocks, a_ngrp, b_ngrp = _get_mx_workspace(M, N, K128, a.device, stream)
    # Per-shape cfg: first call benches GROUP_M/num_xcd/group_n (BLOCK_M fixed 256) on
    # the fused stub, caches the winner by (M,N,K).
    bm, gm, xcd, gn, ks, aux = _autotune_mxfp8(
        a8, b8, out, a_raw, b_raw, a_sp, b_sp, M, N, K, a_blocks, a_ngrp, b_ngrp, out_dtype, cbsz, blgp
    )
    ks, head, tail = _mx_ksplit(M, N, K, out_dtype, beta_is_one) if ks > 1 else (1, 0, 0)
    if ks > 1:
        ws = _get_mx_split_ws(tail, ks, a.device, stream, out_dtype)
    else:
        ws, head = out, ceildiv(M, bm) * ceildiv(N, _BLOCK_N)
    launch = _get_mxfp8_fused_launch(
        K,
        bm,
        gm,
        xcd,
        gn,
        cbsz=cbsz,
        blgp=blgp,
        out_fp16=out_fp16,
        beta_is_one=beta_is_one,
        col_safe=_mx_col_safe(N),
        k_split=ks,
        cstore_aux=aux,
    )
    args = (a8, b8, out, a_raw, b_raw, a_sp, b_sp, ws, M, N, a_blocks, a_ngrp, b_ngrp, head, tail, stream)
    # [raw, compiled]: raw for CUDA-graph capture (flyc.compile regresses under capture);
    # compiled for eager (skips per-call jit dispatch overhead). Mirrors _GROUPED_AT_CACHE.
    at_key = (M, N, K, bm, gm, xcd, gn, cbsz, blgp, out_fp16, beta_is_one, ks, aux) + _mx_arm_key()
    entry = _MXFP8_AT_CACHE.get(at_key)
    if entry is None:
        entry = [launch, None]
        _MXFP8_AT_CACHE[at_key] = entry
    raw, compiled = entry
    if torch.cuda.is_current_stream_capturing():
        raw(*args)
    else:
        if compiled is None:
            compiled = compile_with_scratch_out(raw, args)
            entry[1] = compiled
        compiled(*args)
    return out
