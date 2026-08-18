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

"""4-wave MXFP4 dense NT GEMM (per-32-K E8M0 block scaling) for AMD CDNA4 (gfx950).
A [M, K] fp4 (packed 2/byte), B [N, K] fp4, C = a @ b^T. One wave per SIMD lets the
256-AGPR file hold the N-sliced accumulator; swizzle/depth/epilogue are autotuned."""

import torch

# isort: off
from primus_turbo.flydsl.utils.gemm_helper import (
    G2SLoader,
    ceildiv,
    compile_with_scratch_out,
    make_fp8_rebased_tensor_and_srd,
    make_row_band_resource,
    resolve_accum_out,
    xcd_remap_pid,
)
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm as _llvm
from flydsl.expr import arith, buffer_ops, const_expr, range_constexpr, rocdl
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

# isort: on


def _raw(v):
    if not isinstance(v, ir.Value) and hasattr(v, "ir_value"):
        return v.ir_value()
    return v


# ── Device-side scale + fragment loaders / geometry ──────────────────────────


class ScaleS2RPacked:
    """Packed-scale buffer resource (one dword per (region, k) holding n_tiles E8M0,
    byte t = tile t). The production kernel uses only ``.rsrc`` (sized to cover the
    lane-contiguous scale tensor); the VGPR-direct loads address it from the asm."""

    def __init__(self, sp_tensor, dim, K, n_tiles):
        group_span = 16 * n_tiles
        nbytes = (dim // group_span) * (K // 128) * 64 * 4  # int32 records, 1/lane
        self.rsrc = buffer_ops.create_buffer_resource(sp_tensor, max_size=False, num_records_bytes=nbytes)


def _swz_fwd(c, d=0):
    """LDS bank-swizzle (bijection): rotate each block's rows so one ds_read spreads
    across all bank-groups, avoiding conflicts. ``d`` offsets the rotation so two
    parity-split regions read in the same ds_read stay on disjoint bank-groups."""
    ph = c // 8
    return ph * 8 + (c % 8 + ph + d) % 8


def fp4_g2s_offsets(lane_id, wave_id, K, n_steps, bytes_per_row, swizzle=False, ilv=0):
    """Per-lane gmem byte offsets for fp4 G2S into identity LDS slots (S2R reads back
    at the same address). ``swizzle`` pre-applies the inverse bank-swizzle; ``ilv``
    column-interleaves source rows so a lane owns adjacent columns, LDS image intact."""
    n_waves = fx.block_dim.x // 64
    lpr = bytes_per_row // 16  # lanes per row
    rows_per_step = 64 // lpr
    assert not ilv or (bytes_per_row == 128 and ilv == 4)
    offs = []
    for r in range_constexpr(n_steps):
        ph = lane_id // lpr  # physical row slot in this lane's LDS block
        row = ph + wave_id * rows_per_step + r * (n_waves * rows_per_step)
        chunk = (lane_id % lpr + lpr - row % lpr) % lpr if swizzle else lane_id % lpr
        src = row
        if ilv:
            q = row % 64
            src = (row // 64) * 64 + ilv * (q % 16) + q // 16
        offs.append(src * (K // 2) + chunk * 16)
    return offs


def fp4_g2s_offsets_split(lane_id, wave_id, K, n_steps, row_par, shift, ilv=0):
    """Per-lane gmem byte offsets for ONE parity region of the split fp4 G2S: a region
    holds alternate operand rows so a whole request stays inside one cache line. ``ilv``
    column-interleaves across both regions so a lane owns adjacent output columns."""
    n_waves = fx.block_dim.x // 64
    offs = []
    for r in range_constexpr(n_steps):
        ph = lane_id // 8  # physical row slot in this lane's 1024B block
        u = ph + wave_id * 8 + r * (n_waves * 8)  # region row
        if ilv:
            u = r * (n_waves * 8) + (ph + (wave_id % 2) * 8) * 2 + wave_id // 2
        k = (lane_id % 8 + 8 - ph) % 8  # physical 16B slot -> logical chunk
        offs.append((u * 2 + row_par) * (K // 2) + k * 16 + shift)
    return offs


class S2RLoaderFp4Split:
    """LDS->reg fp4 fragment loader for the parity-split cache-line-aligned layout so no
    g2s crosses a cache-line boundary or reads a buffer being refilled. ``skew``/``f_base``
    drive the odd-phase rotating ring; ``ilv`` column-interleaves across both regions."""

    def __init__(self, wave_idx, n_tiles, n_rows, par, skew, ilv=0):
        self.lane16 = fx.thread_idx.x % 16
        self.g = (fx.thread_idx.x % 64) // 16
        self.wave_idx = wave_idx
        self.n_tiles = n_tiles
        self.slot = (n_rows // 2) * 128  # bytes per ring slot (one region)
        self.par = par  # 128B phase of tile row 0
        self.skew = skew
        self.ilv = ilv
        assert not ilv or ilv == n_tiles

    def _phys(self, lds, u, k, slot_off):
        off = u * 128 + ((k + u % 8) % 8) * 16 + slot_off
        i8 = fx.recast_iter(fx.Uint8, fx.add_offset(lds.ptr, fx.make_int_tuple(off)))
        return fx.ptrtoint(i8)

    def _row(self):
        return self.wave_idx * (self.n_tiles * 16) + self.lane16

    def _pair(self, lds_e, lds_o, u, buf, s):
        ae = self._phys(lds_e, u, s * 4 + self.g, buf * self.slot)
        if const_expr(self.skew):
            ko = (4 + self.g) if s == 0 else self.g  # top of line c, bottom of line c+1
            ao = self._phys(lds_o, u, ko, 0)
        else:
            ao = self._phys(lds_o, u, s * 4 + self.g, buf * self.slot)
        return ae, ao

    def f_base(self, lds_e, lds_o, buf, s):
        """Per-lane LDS read address for ring slot ``buf``, 128-K sub-step ``s``."""
        ae, ao = self._pair(lds_e, lds_o, self._row() // 2, buf, s)
        return arith.select((self._row() + self.par) % 2 == fx.Int32(1), ao, ae)

    def f_base_ilv(self, lds_e, lds_o, buf, s):
        """Both regions' addresses, for the column-interleaved fragment. The lane reads
        region rows wave*ntb*8 + lane%16 (+ tile_stride per tile PAIR) of each region."""
        u = self.wave_idx * (self.n_tiles * 8) + self.lane16
        return self._pair(lds_e, lds_o, u, buf, s)

    def q_unit(self):
        """Odd-ring slot stride for this lane (0 on even-phase rows)."""
        if const_expr(bool(self.ilv)):
            return fx.Int32(self.slot)  # interleaved: every lane reads the odd region
        odd = (self._row() + self.par) % 2 == fx.Int32(1)
        return arith.select(odd, fx.Int32(self.slot), fx.Int32(0))

    @property
    def tile_stride(self):
        return (self.n_tiles * 4 if self.ilv else 8) * 128


def grouped_xcd_pid(pid, c_m, c_n, BLOCK_M, BLOCK_N, group_m=4, num_xcds=8, group_n=0):
    """Map block_idx -> (block_m, block_n) with XCD-aware remap + GROUP_M tiling for
    L2 locality. group_n>0 enables a 2D super-block (N-band) swizzle on top (locks an
    A-slab AND a B-slab into L2). Pure index math, a bijection for any tile count.

    The XCD remap must be exact: one workgroup owns one tile, so a mapping that
    collides drops whichever tile lost the collision and leaves it at whatever the
    output allocation happened to hold.
    """
    num_pid_m = ceildiv(c_m, BLOCK_M)
    num_pid_n = ceildiv(c_n, BLOCK_N)
    total = num_pid_m * num_pid_n
    pid_r = xcd_remap_pid(pid, total, num_xcds)

    if group_n and group_n > 0:
        band_tiles = num_pid_m * group_n  # tiles in one full band
        n_full_bands = num_pid_n // group_n
        full_region = n_full_bands * band_tiles  # pids covered by full bands
        in_full = pid_r < full_region
        band_id = pid_r // band_tiles
        local_f = pid_r % band_tiles
        nbase_f = band_id * group_n
        bw_f = fx.Int32(group_n)
        rem = num_pid_n - n_full_bands * group_n
        local_r = pid_r - full_region
        nbase_r = n_full_bands * group_n
        bw_r = arith.select(rem < fx.Int32(1), fx.Int32(1), rem)  # avoid /0 in dead branch
        local = arith.select(in_full, local_f, local_r)
        nbase = arith.select(in_full, nbase_f, nbase_r)
        bw = arith.select(in_full, bw_f, bw_r)
        num_in_group = group_m * bw
        group_id = local // num_in_group
        first_m = group_id * group_m
        gsz = num_pid_m - first_m
        gsz = arith.select(gsz < fx.Int32(group_m), gsz, fx.Int32(group_m))
        inner = local % num_in_group
        block_m = first_m + inner % gsz
        block_n = nbase + inner // gsz
        return block_m, block_n

    num_in_group = group_m * num_pid_n
    group_id = pid_r // num_in_group
    first_m = group_id * group_m
    gsz = num_pid_m - first_m
    gsz = arith.select(gsz < fx.Int32(group_m), gsz, fx.Int32(group_m))
    inner = pid_r % num_in_group
    block_m = first_m + inner % gsz
    block_n = inner // gsz
    return block_m, block_n


class S2RLoaderFp4:
    """LDS->reg fp4 fragment loader (identity LDS, bytes_per_row K-iter rows).

    A K-iter spans n_sub == BLOCK_K/128 128-K sub-blocks; each sub-block is one
    16x16x128 MFMA. The production whole-loop reads via ``base_addr`` (one address
    reg per region + a ds_read offset immediate)."""

    def __init__(self, wave_idx, n_tiles, row_stride, swizzle=False):
        self.lane16 = fx.thread_idx.x % 16
        self.g = (fx.thread_idx.x % 64) // 16
        self.wave_idx = wave_idx
        self.n_tiles = n_tiles
        self.row_stride = row_stride
        self.swizzle = swizzle

    def base_addr(self, lds_src, s=0):
        """Single base LDS address (tile 0, sub-block s). Per-tile fragments are at
        base + i*tile_stride -> the asm uses ONE address reg per region + a ds_read
        offset immediate."""
        off_nat = (self.wave_idx * (self.n_tiles * 16) + self.lane16) * self.row_stride + s * 64 + self.g * 16
        if const_expr(self.swizzle):  # tile i = base + i*tile_stride stays swz-correct
            cib = (off_nat % 1024) // 16  # (tile_stride is a 1024-multiple -> %1024 const)
            off = (off_nat // 1024) * 1024 + _swz_fwd(cib) * 16
        else:
            off = off_nat
        i8_iter = fx.recast_iter(fx.Uint8, fx.add_offset(lds_src.ptr, fx.make_int_tuple(off)))
        return fx.ptrtoint(i8_iter)

    @property
    def tile_stride(self):
        return 16 * self.row_stride


class StoreCPlain:
    """Plain FP32 accumulator -> BF16/FP16 store (scales folded in MMA), using an
    OOB-index redirect for the column-edge mask to avoid per-store EXEC save/restore.
    ``out_ty`` fp16 forces the narrow path; ``ilv`` maps a lane to adjacent columns."""

    def __init__(
        self, C, c_rows, c_cols, c_idx_fn, n_tiles_a, n_tiles_b, out_ty=None, ilv=0, beta_is_one=False
    ):
        assert not ilv or ilv == n_tiles_b
        self.c_rows = c_rows
        self.c_cols = c_cols
        self.lane_id = fx.thread_idx.x % 64
        self.c_idx_fn = c_idx_fn
        self.n_tiles_a = n_tiles_a
        self.n_tiles_b = n_tiles_b
        self.ilv = ilv
        self.out_ty = out_ty if out_ty is not None else fx.BFloat16
        self.beta_is_one = beta_is_one
        # int64 byte base: the store re-bases per row band (make_row_band_resource) so a
        # C whose flat rows*cols exceeds 2^31 (large-G wgrad grad_b [G,N,K]) addresses
        # correctly. Pass C as 2D so its shape packs within int32.
        self.c_base = buffer_ops.extract_base_index(C)

    def store(self, c_frag, base_row, base_col, n_valid=None):
        # n_valid drops whole out-of-range column spans via band SRD num_records (no per-store mask).
        c_rows = self.c_rows
        if n_valid is not None and n_valid % (16 * self.n_tiles_b) == 0:
            c_rows = arith.select(base_col < fx.Int32(n_valid), c_rows, base_row)
            n_valid = None
        rsrc = make_row_band_resource(self.c_base, base_row, c_rows, self.c_cols, 2)
        if n_valid is None and const_expr(not self.beta_is_one):
            # Fast path writes only; beta=1 needs the read-back, so it falls through below.
            self._store_rowaddr(c_frag, base_col, rsrc)
            return
        # beta=1 issues every read-back before the first store so the loads pipeline.
        addrs = []
        for ti in range_constexpr(self.n_tiles_a):
            row_local = ti * 16 + (self.lane_id // 16) * 4  # relative to base_row
            for tj in range_constexpr(self.n_tiles_b):
                col = base_col + self._col(tj)
                col_valid = (col < fx.Int32(n_valid)) if n_valid is not None else None
                for i in range_constexpr(4):
                    addrs.append(((row_local + i) * self.c_cols + col, col_valid))
        prev = []
        if const_expr(self.beta_is_one):
            prev = [
                buffer_ops.buffer_load(rsrc, off_e, vec_width=1, dtype=self.out_ty.ir_type, mask=valid)
                for off_e, valid in addrs
            ]
        n = 0
        for ti in range_constexpr(self.n_tiles_a):
            for tj in range_constexpr(self.n_tiles_b):
                vec_f32 = Vec(c_frag[self.c_idx_fn(ti, tj)])
                for i in range_constexpr(4):
                    off_e, col_valid = addrs[n]
                    val = vec_f32[i]
                    if const_expr(self.beta_is_one):
                        val = val + self.out_ty(prev[n]).to(fx.Float32)  # add in f32: one rounding
                    val = val.to(self.out_ty)
                    buffer_ops.buffer_store(val, rsrc, off_e * 2, mask=col_valid, offset_is_bytes=True)
                    n += 1

    def _pack_pair(self, x0, x1):
        if const_expr(self.out_ty is fx.Float16):
            halves = Vec.from_elements([x0.to(fx.Float16), x1.to(fx.Float16)], fx.Float16)
            return halves.bitcast(fx.Int32)[0]
        return rocdl.cvt_pk_bf16_f32(x0, x1)

    def _col(self, tj):
        """Column of N sub-block ``tj`` for this lane, relative to the wave's column base."""
        if const_expr(bool(self.ilv)):
            return self.ilv * (self.lane_id % 16) + tj
        return tj * 16 + self.lane_id % 16

    def fused_operands(self, base_row, base_col_l, base_col_r, n_valid=None):
        """SRDs + per-lane voffset for a C store emitted INSIDE the whole-loop asm,
        addressed like ``_store_rowaddr`` (one address per row/lane). Both halves share
        the row band and differ only in num_records dropping an all-padding R half."""
        rsrc = []
        for col in (base_col_l, base_col_r):
            rows = self.c_rows
            if n_valid is not None:
                assert n_valid % (16 * self.n_tiles_b) == 0
                rows = arith.select(col < fx.Int32(n_valid), rows, base_row)
            rsrc.append(make_row_band_resource(self.c_base, base_row, rows, self.c_cols, 2))
        row_b = self.c_cols * fx.Int32(2)
        voff = (base_col_l + self._col(0)) * fx.Int32(2) + (self.lane_id // 16) * (row_b * fx.Int32(4))
        return rsrc[0], rsrc[1], voff, rocdl.readfirstlane(T.i32, row_b)

    def _store_rowaddr(self, c_frag, base_col, rsrc):
        """Unmasked store: ONE address per (row, lane) shared by the row's N sub-blocks,
        which ride the store's offset immediate so the per-store address VALU disappears.
        Rows past c_rows still OOB-drop through the band SRD's num_records."""
        row_b = self.c_cols * fx.Int32(2)  # C row stride in bytes
        base = (base_col + self._col(0)) * fx.Int32(2) + (self.lane_id // 16) * (row_b * fx.Int32(4))
        step = 2 if self.ilv else 32  # interleaved: sub-blocks are adjacent columns
        for ti in range_constexpr(self.n_tiles_a):
            for i in range_constexpr(4):
                off = base + row_b * fx.Int32(ti * 16 + i)
                for tj in range_constexpr(self.n_tiles_b):
                    val = Vec(c_frag[self.c_idx_fn(ti, tj)])[i].to(self.out_ty)
                    buffer_ops.buffer_store(
                        val, rsrc, off if tj == 0 else off + tj * step, offset_is_bytes=True
                    )

    @staticmethod
    def _permlane16_swap(a_i32, b_i32):
        """v_permlane16_swap_b32 a, b -- swap 16-lane row-groups between two regs
        (both read+written, in place). Returns (a', b'). Row-group map:
            a'[rg0]=a[rg0] a'[rg1]=b[rg0] a'[rg2]=a[rg2] a'[rg3]=b[rg2]
            b'[rg0]=a[rg1] b'[rg1]=b[rg1] b'[rg2]=a[rg3] b'[rg3]=b[rg3]"""
        st = "!llvm.struct<(i32, i32)>"
        # The result is consumed by a buffer_store (VMEM), not a VALU op, so the
        # permlane16_swap->read VALU hazard s_nop is unnecessary here (saves ~64 exposed
        # nops on the store-bound epilogue).
        r = _llvm.inline_asm(
            ir.Type.parse(st),
            [_raw(a_i32), _raw(b_i32)],
            "v_permlane16_swap_b32 $0, $1",
            "=v,=v,0,1",
            has_side_effects=False,
        )
        i32t = ir.IntegerType.get_signless(32)
        return _llvm.extractvalue(i32t, r, [0]), _llvm.extractvalue(i32t, r, [1])

    def store_tacc_wide(self, c_frag, base_row, base_col):
        """TACC + permlane16_swap wide store: two adjacent N sub-blocks become one
        16-row x 32-col region per lane, written with a single buffer_store_dwordx4.
        With acc = C^T a lane holds 4 consecutive columns; two swaps make them 8 contiguous."""
        nta, ntb = self.n_tiles_a, self.n_tiles_b
        assert ntb % 2 == 0, "store_tacc_wide pairs N sub-blocks (ntb must be even)"
        # Re-base at this tile's row band in int64 (rows*cols may exceed 2^31); the per-lane
        # byte offset below is then intra-band int32.
        rsrc = make_row_band_resource(self.c_base, base_row, self.c_rows, self.c_cols, 2)
        rg = self.lane_id // 16
        col_off = (rg % 2) * 16 + (rg // 2) * 8  # rg0->0 rg1->16 rg2->8 rg3->24
        # Hoisted per-lane base byte offset relative to base_row: (base_col + lane%16*c_cols
        # + col_off)*2. Per-store delta (ti*16*c_cols + tj*16)*2 is a compile-time constant ->
        # one v_add per store instead of recomputing r*c_cols (matches AITER's voffset+imm).
        lane_base_e = base_col + (self.lane_id % 16) * self.c_cols + col_off

        def _off_e(ti, tj):
            return lane_base_e + fx.Int32(ti * 16 * self.c_cols + tj * 16)

        def _xpose(ti, tj):
            A = Vec(c_frag[self.c_idx_fn(ti, tj)])
            B = Vec(c_frag[self.c_idx_fn(ti, tj + 1)])
            d_a0 = self._pack_pair(A[0], A[1])
            d_a1 = self._pack_pair(A[2], A[3])
            d_b0 = self._pack_pair(B[0], B[1])
            d_b1 = self._pack_pair(B[2], B[3])
            v16, v18 = self._permlane16_swap(d_a0, d_b0)
            v17, v19 = self._permlane16_swap(d_a1, d_b1)
            return Vec.from_elements([v16, v17, v18, v19], fx.Int32).bitcast(self.out_ty)

        # Software-pipeline cvt+permlane (phase1) away from the stores (phase2) so the
        # permlane16_swap->store RAW hazard (~80 exposed s_nop) is filled by independent
        # permlane work of later tiles instead of stalls.
        slots = [(ti, 2 * p) for ti in range_constexpr(nta) for p in range_constexpr(ntb // 2)]
        # Read-backs first, then the cvt/permlane work, then the stores: the loads get
        # the whole phase-1 shuffle to hide behind instead of stalling their own store.
        prev = (
            [
                Vec(buffer_ops.buffer_load(rsrc, _off_e(ti, tj), vec_width=8, dtype=self.out_ty.ir_type))
                for (ti, tj) in slots
            ]
            if const_expr(self.beta_is_one)
            else []
        )
        vecs = [_xpose(ti, tj) for (ti, tj) in slots]
        for n, (vec_out, (ti, tj)) in enumerate(zip(vecs, slots)):
            if const_expr(self.beta_is_one):
                # The packed value is already out_ty, so widen both sides and add in f32
                # to keep the sum from rounding twice.
                vec_out = (vec_out.to(fx.Float32) + prev[n].to(fx.Float32)).to(self.out_ty)
            buffer_ops.buffer_store(vec_out, rsrc, _off_e(ti, tj) * 2, offset_is_bytes=True)


# ── Scaled MFMA whole-loop emitter ───────────────────────────────────────────


class MfmaScaleFp4:
    """16x16x128 f8f6f4 MFMA in fp4 mode (cbsz=4/blgp=4) with packed per-block E8M0
    scales (one packed-i32 scale operand per region, opsel selects the per-XDL byte).

    Only the production whole-loop path is provided: the entire K-loop is one
    inline-asm hardware loop (no per-iter FlyDSL boundary), 2 LDS buffers ping-pong
    (unroll-2), with NEXT-K in-place operand refill and VGPR-direct scales."""

    def __init__(self, n_tiles_a, n_tiles_b, packed=False, wlv=10, elgk=9, coop=False, tacc=False):
        self.res_ty = Vec.make_type(4, fx.Float32)
        self.zero_value = Vec.filled(4, 0.0, fx.Float32)
        self.n_tiles_a = n_tiles_a
        self.n_tiles_b = n_tiles_b
        self.packed = packed
        # tacc: swap MMA operands so the accumulator holds C^T (4 consecutive columns/lane) for a wide store.
        self.tacc = tacc
        self.wlv = wlv
        self.elgk = elgk
        self.coop = coop

    def idx(self, i, j):
        return i * self.n_tiles_b + j

    def call_mxfp4_wholeloop(
        self,
        a_base,
        bl_base,
        br_base,
        ts_a,
        ts_b,
        abase,
        blbase,
        brbase,
        gl_a,
        gl_b,
        rsrc_a,
        rsrc_b,
        kstep,
        scv,
        cL,
        cR,
        n_sub,
        nsa,
        nsb,
        nval,
        soff0,
        soff0_bl,
        soff0_br,
        sc_rb,
        sc_gb,
        sc_rsa,
        sc_rsb,
        sc_voff,
        sc_soff0,
        ki=None,
        sc_buf_stride=0,
        half_n=None,
        half_g2s=True,
        half_k=False,
        split=None,
        cst=None,
        cst_gap=0,
        cst_ilv=0,
        cst_nt=False,
        b_base_even=None,
        _cache={},  # noqa: B006 -- deliberate cross-call asm compile cache
    ):
        """WHOLE-LOOP bare-asm K-loop: one inline-asm hw-loop, unroll-2 ping-pong with
        VGPR-direct or COOP scales. half_n/half_k/split/cst/half_g2s select boundary/odd-K/
        parity-split/fused-store/live-R-half variants; returns (accL, accR)."""
        assert self.packed
        nta, ntb = self.n_tiles_a, self.n_tiles_b
        nq = nta * ntb
        NT = 2 * nq
        na, nb = nta * n_sub, ntb * n_sub
        ntmp = na + 2 * nb
        _NWc = 4  # n_waves (4-wave kernel)
        nbuf = len(a_base)  # A pool size (= 2, unroll-2 ping-pong)
        nbuf_b = len(bl_base)  # B pool size (= 2)
        _nscbuf = nbuf_b  # scale LDS pool (unused under VGPR-direct, but operand slots are reserved)
        NSET = 1
        _WLV = self.wlv  # vmcnt kept in flight at the phase barrier (deep g2s pipeline)
        _ELGK = self.elgk  # lgkmcnt left at the phase barrier (late refills stay in flight)
        # Cooperative LDS scale staging (vs per-wave VGPR-direct): 4 waves co-load the 4 groups once.
        _COOP = self.coop
        _TACC = self.tacc  # transposed accumulator: swap MMA operands -> acc = C^T
        _PINBASE = 8
        key = (
            nta,
            ntb,
            n_sub,
            nsa,
            nsb,
            ts_a,
            ts_b,
            nbuf,
            nbuf_b,
            (ki is None) or (ki >= 2),
            (ki is not None) and bool(ki & 1),
            self.wlv,
            self.elgk,
            self.coop,
            _TACC,
            ki,
            half_n is not None,
            half_g2s,
            half_k,
            split is not None and len(split[0]),
            cst is not None,
            cst_gap,
            cst_ilv,
            cst_nt,
        )
        _SPLIT = split is not None
        _CST = cst is not None
        _CILV = cst_ilv
        assert not _CILV or (_CST and _CILV == ntb and ntb == 4)
        _BSPL = bool(_CILV) and _SPLIT
        assert _BSPL == (b_base_even is not None)
        # The fused store needs a g2s-free tail phase to ride (unified vmcnt): a peel or odd KI.
        assert not _CST or (not self.coop and (ki is None or (half_k and ((ki & 1) or ki >= 4))))
        _RTPEEL = _CST and ki is None
        # 3-slot odd ring (skewed rows) rotates ds_read bases; a 2-slot ring is byte-exact/static.
        _ROT = _SPLIT and len(split[0]) == 3
        _NOD = len(split[0]) if _SPLIT else 0
        if _SPLIT:
            assert n_sub == 2 and nbuf == 2 and nbuf_b == 2 and nsa % 2 == 0 and nsb % 2 == 0
        # K-loop shape: half_k + even trip count peels the final unroll-2 iteration (the loop stops
        # two phases early); an odd trip count peels its phase A instead; an odd trailing 128-K
        # block with no peel runs as an MFMA-only tail.
        _KPEEL = half_k and not _COOP and (ki is not None) and (ki >= 4) and not (ki & 1)
        _OPEEL = _CST and half_k and not _COOP and (ki is not None) and (ki >= 5) and bool(ki & 1)
        _has_loop = (ki is None) or (ki >= 2)
        _has_tail = (ki is not None) and bool(ki & 1) and not _OPEEL
        # Accumulator init folded into the first MFMA's src2 immediate 0: the head phase is peeled
        # so every quad's opening MFMA writes instead of accumulating, which drops the 256-dword
        # AGPR pre-clear (one VALU per accumulator dword) from every tile's prologue.  Needs the
        # accumulators pinned to named AGPRs (_CST); a runtime trip count that skips the head
        # keeps an explicit clear on that branch alone (emit_acc_clear).
        _ZACC = _CST and (_has_loop or _has_tail)
        if key not in _cache:
            o_acc = list(range(NT))
            t_a = NT
            t_bl = t_a + na
            t_br = t_bl + nb  # ds_read temp outputs
            nsct = 4 * n_sub  # scale temps: A-g0, A-g1, BL, BR x n_sub
            t_sc = t_br + nb  # scale temp base
            _scextra = nsct  # VGPR-direct 2nd scale set (ping-pong)
            set_sz = ntmp + nsct + _scextra
            ntmp2 = NSET * set_sz
            _nvx = 19 if _ROT else 0  # split: 12 rotating ds_read bases + 2x3 ring offsets + scratch
            _nbase = NT + ntmp2
            o_nb = [[[_nbase + t * 4 + b * 2 + s for s in range(2)] for b in range(2)] for t in range(3)]
            o_nq = [[_nbase + 12 + t * 3 + j for j in range(3)] for t in range(2)]
            o_vtm = _nbase + 18
            o_cnt = NT + ntmp2 + _nvx  # =&s loop counter
            o_sa = o_cnt + 1
            o_sbl = o_sa + 1
            o_sbr = o_sbl + 1  # advancing gmem soffsets A/BL/BR
            o_ta = o_sbr + 1
            o_tbl = o_ta + 1
            o_tbr = o_tbl + 1  # buf1 (=+kstep) scratch soffsets
            o_sca = [o_tbr + 1 + g for g in range(4)]  # 4 scale soffsets (A-g0, A-g1, BL, BR)
            o_sct = o_sca[3] + 1  # scale scratch soffset
            o_pod = [
                [o_sct + 1 + t * 3 + j for j in range(3)] for t in range(3)
            ]  # 3 odd-ring g2s dests/operand
            o_stm = o_sct + 10
            nout = o_sct + 1 + (10 if _ROT else 0)
            o_csc = nout
            o_crw = [[nout + 1 + p * 4 + e for e in range(4)] for p in range(2)]
            if _CST:
                nout += 9
            _CDV = _PINBASE + NSET * (2 * nsct + 4 * ntmp)
            _NCDV = 18 if _CILV else 0
            nout += _NCDV
            o_npv = nout  # runtime peel: hw-loop bound = trip count - 2
            if _RTPEEL:
                nout += 1
            # scale temp accessors (group: 0=A-g0, 1=A-g1, 2=BL, 3=BR; slot=grp*n_sub+s)
            # _scb[0] = ping-pong scale-set base (0 or nsct), set per phase.
            _scb = [0]

            def sa_t(s, g):
                return t_sc + _scb[0] + g * n_sub + s

            def sbl_t(s):
                return t_sc + _scb[0] + 2 * n_sub + s

            def sbr_t(s):
                return t_sc + _scb[0] + 3 * n_sub + s

            # inputs (after outputs):
            i = nout
            i_ab = [[i + b * n_sub + s for s in range(n_sub)] for b in range(nbuf)]
            i += nbuf * n_sub  # A ds_read base
            i_blb = [[i + b * n_sub + s for s in range(n_sub)] for b in range(nbuf_b)]
            i += nbuf_b * n_sub
            i_brb = [[i + b * n_sub + s for s in range(n_sub)] for b in range(nbuf_b)]
            i += nbuf_b * n_sub
            i_g_ab = [i + b for b in range(nbuf)]
            i += nbuf  # g2s A LDS dest base (sgpr)
            i_g_blb = [i + b for b in range(nbuf_b)]
            i += nbuf_b
            i_g_brb = [i + b for b in range(nbuf_b)]
            i += nbuf_b
            i_gla = [i + s for s in range(nsa)]
            i += nsa  # gmem voffsets A
            i_glb = [i + s for s in range(nsb)]
            i += nsb  # gmem voffsets B
            i_rsa = i
            i += 1
            i_rsb = i
            i += 1  # rsrc
            i_kstep = i
            i += 1
            i += 1  # (legacy const scale dummy, reserved operand slot)
            i_nval = i
            i += 1
            i_sa0 = i
            i += 1
            i_sbl0 = i
            i += 1
            i_sbr0 = i
            i += 1  # soffset inits A/BL/BR (region base k=0)
            i_scrb = [i + b for b in range(_nscbuf)]
            i += _nscbuf  # scale LDS read base (A,B for buf0; coop ds_read source)
            i_scgb = [i + b for b in range(_nscbuf)]
            i += _nscbuf  # scale LDS g2s dest base (per-buf, coop g2s dest)
            i_scrsa = i
            i += 1
            i_scrsb = i
            i += 1  # scale rsrc (A_scale, B_scale)
            i_scvoff = i
            i += 1  # scale per-lane gmem voffset
            i_sca0 = [i + g for g in range(4)]
            i += 4  # scale soffset inits (A-g0, A-g1, BL, BR)
            i_hn = i
            i += 1 if half_n is not None else 0  # half-N variant selector
            i_od = [[i + t * _NOD + j for j in range(_NOD)] for t in range(3)]  # odd g2s dest
            i += 3 * _NOD
            i_qu = [i, i + 1]  # per-lane odd-ring slot stride (0 on aligned rows)
            i += 2 if _ROT else 0
            i_cl, i_cr, i_cvo, i_crb = i, i + 1, i + 2, i + 3  # fused store: SRDs, voff, row bytes
            i += 4 if _CST else 0
            i_ble = [[i + b * n_sub + s for s in range(n_sub)] for b in range(nbuf_b)]
            i += nbuf_b * n_sub if _BSPL else 0
            i_bre = [[i + b * n_sub + s for s in range(n_sub)] for b in range(nbuf_b)]
            i += nbuf_b * n_sub if _BSPL else 0
            if _ROT:
                f_ab, f_blb, f_brb = i_ab, i_blb, i_brb
                i_ab, i_blb, i_brb = o_nb[0], o_nb[1], o_nb[2]

            def b_rd(sl, buf, s, ji):
                """LDS read operands of B's N sub-block ``ji`` (base register, byte offset)."""
                if _BSPL:
                    odd, even = (i_blb, i_ble) if sl == 0 else (i_brb, i_bre)
                    return (odd if ji % 2 else even)[buf][s], (ji // 2) * ts_b
                return (i_blb if sl == 0 else i_brb)[buf][s], ji * ts_b

            def emit_ds(buf, off=0):
                # operands only; scales are VGPR-direct (emit_sc_vgpr in the loop).
                r = []
                for ii in range(nta):
                    for s in range(n_sub):
                        r.append(
                            f"ds_read_b128 ${t_a + ii * n_sub + s + off}, ${i_ab[buf][s]} offset:{ii * ts_a}"
                        )
                for sl, tb in ((0, t_bl), (1, t_br)):
                    for ji in range(ntb):
                        for s in range(n_sub):
                            bb, bo = b_rd(sl, buf, s, ji)
                            r.append(f"ds_read_b128 ${tb + ji * n_sub + s + off}, ${bb} offset:{bo}")
                return r

            def emit_g2s(buf, sa_op, sbl_op, sbr_op, half=False, only_rg=None, b_only=False):
                if _SPLIT:
                    # Two streams/operand: aligned rows into slot buf, skewed rows into the 3-slot ring head.
                    ne, nbe = nsa // 2, nsb // 2
                    od = o_pod if _ROT else i_od
                    r = []
                    for de, do, gl, rs, so, n in (
                        (i_g_ab[buf], od[0][buf], i_gla, i_rsa, sa_op, 0 if b_only else ne),
                        (i_g_blb[buf], od[1][buf], i_glb, i_rsb, sbl_op, nbe),
                        (i_g_brb[buf], od[2][buf], i_glb, i_rsb, sbl_op if half else sbr_op, nbe),
                    ):
                        for st in range(n):
                            for rg, dst in enumerate((de, do)):
                                if only_rg is not None and rg != only_rg:
                                    continue
                                r.append(
                                    f"s_add_u32 m0, ${dst}, {st * _NWc * 1024}\n"
                                    f"buffer_load_dwordx4 ${gl[rg * n + st]}, ${rs}, ${so} offen lds"
                                )
                    return r
                r = []
                for st in range(0 if b_only else nsa):
                    r.append(
                        f"s_add_u32 m0, ${i_g_ab[buf]}, {st * _NWc * 1024}\n"
                        f"buffer_load_dwordx4 ${i_gla[st]}, ${i_rsa}, ${sa_op} offen lds"
                    )
                for st in range(nsb):
                    r.append(
                        f"s_add_u32 m0, ${i_g_blb[buf]}, {st * _NWc * 1024}\n"
                        f"buffer_load_dwordx4 ${i_glb[st]}, ${i_rsb}, ${sbl_op} offen lds"
                    )
                for st in range(nsb):
                    r.append(
                        f"s_add_u32 m0, ${i_g_brb[buf]}, {st * _NWc * 1024}\n"
                        f"buffer_load_dwordx4 ${i_glb[st]}, ${i_rsb}, ${sbl_op if half else sbr_op} offen lds"
                    )
                return r

            def emit_rot():
                mv = []
                for t in range(2):
                    q = o_nq[t]
                    mv += [
                        f"v_mov_b32 ${o_vtm}, ${q[1]}",
                        f"v_mov_b32 ${q[1]}, ${q[0]}",
                        f"v_mov_b32 ${q[0]}, ${q[2]}",
                        f"v_mov_b32 ${q[2]}, ${o_vtm}",
                    ]
                sv = []
                for t in range(3):
                    p = o_pod[t]
                    sv += [
                        f"s_mov_b32 ${o_stm}, ${p[1]}",
                        f"s_mov_b32 ${p[1]}, ${p[0]}",
                        f"s_mov_b32 ${p[0]}, ${p[2]}",
                        f"s_mov_b32 ${p[2]}, ${o_stm}",
                    ]
                return ["\n".join(mv), "\n".join(sv)]

            def emit_bases(buf):
                jj = (2, 0) if buf == 0 else (0, 1)
                r = []
                for t, fr in enumerate((f_ab, f_blb, f_brb)):
                    q = o_nq[0 if t == 0 else 1]
                    for s in range(2):
                        r.append(f"v_add_u32 ${o_nb[t][buf][s]}, ${fr[buf][s]}, ${q[jj[s]]}")
                return r

            def mix_g2s(g2s, extra):
                if not extra:
                    return g2s
                out = []
                gap = max(len(g2s) // len(extra), 1)
                ei = 0
                for k, ln in enumerate(g2s):
                    out.append(ln)
                    if ei < len(extra) and k % gap == gap - 1:
                        out.append(extra[ei])
                        ei += 1
                return out + extra[ei:]

            def ds_line(buf, tt):
                # per-temp ds_read for the in-place refill stream. Scale temps
                # (tt >= t_sc) are VGPR-direct -> no LDS read (empty line).
                if tt < t_bl:
                    rel = tt - t_a
                    ii = rel // n_sub
                    s = rel % n_sub
                    return f"ds_read_b128 ${tt}, ${i_ab[buf][s]} offset:{ii * ts_a}"
                if tt < t_sc:
                    sl = 0 if tt < t_br else 1
                    rel = tt - (t_bl if sl == 0 else t_br)
                    bb, bo = b_rd(sl, buf, rel % n_sub, rel // n_sub)
                    return f"ds_read_b128 ${tt}, ${bb} offset:{bo}"
                return ""

            # fused C store folded into the g2s-free tail MFMA stream, paced _CRATE lines/MFMA per acc.
            _CAGE = 4
            _CRATE = 6 if _CILV else 2
            _CST_HAZ = ["s_nop 15", "s_nop 15"]
            _ARD = 8

            def cst_rows(ii, p):
                r = o_crw[p]
                if ii:
                    ls = [
                        f"s_mul_i32 ${o_csc}, ${i_crb}, {ii * 16}",
                        f"v_add_u32 ${r[0]}, ${o_csc}, ${i_cvo}",
                    ]
                else:
                    ls = [f"v_mov_b32 ${r[0]}, ${i_cvo}"]
                for e in range(1, 4):
                    ls.append(f"v_add_u32 ${r[e]}, ${i_crb}, ${r[e - 1]}")
                return ls

            # C is write-only and evicts shared operand lines; ``nt`` keeps operands resident.
            _CNT = " nt" if cst_nt else ""

            def cst_group(ii, sl, ji, p):
                # bf16 is the accumulator's high half, so the store sources the AGPR directly.
                q = sl * nq + ii * ntb + ji
                imm = (cst_gap if sl else 0) + ji * 32
                rs = i_cr if sl else i_cl
                return [
                    f"buffer_store_short_d16_hi a{4 * q + e}, ${o_crw[p][e]}, ${rs}, 0 offen"
                    + (f" offset:{imm}" if imm else "")
                    + _CNT
                    for e in range(4)
                ]

            def cst_wide(ii, sl, p, u):
                # Interleaved: each C row packs to a dwordx2 via v_accvgpr_read + v_cvt_pk_bf16_f32.
                q0 = sl * nq + ii * ntb
                imm = cst_gap if sl else 0
                rs = i_cr if sl else i_cl
                b = _CDV + (u & 1) * 8
                s0, s1 = _CDV + 16, _CDV + 17
                ls = []
                for e in range(4):
                    d = b + 2 * e
                    ls += [
                        f"v_accvgpr_read_b32 v{d}, a{4 * q0 + e}",
                        f"v_accvgpr_read_b32 v{s0}, a{4 * (q0 + 1) + e}",
                        f"v_cvt_pk_bf16_f32 v{d}, v{d}, v{s0}",
                        f"v_accvgpr_read_b32 v{d + 1}, a{4 * (q0 + 2) + e}",
                        f"v_accvgpr_read_b32 v{s1}, a{4 * (q0 + 3) + e}",
                        f"v_cvt_pk_bf16_f32 v{d + 1}, v{d + 1}, v{s1}",
                    ]
                for e in range(4):
                    d = b + 2 * e
                    ls.append(
                        f"buffer_store_dwordx2 v[{d}:{d + 1}], ${o_crw[p][e]}, ${rs}, 0 offen"
                        + (f" offset:{imm}" if imm else "")
                        + _CNT
                    )
                return ls

            class CstSched:
                """Paced FIFO of the accumulators the tail's MFMA stream has finished."""

                def __init__(self):
                    self.q = []  # finished store units, in MFMA order
                    self.ln = []  # store lines of the unit being drained
                    self.cur = None
                    self.pend = {}  # interleaved: (ii, sl) -> accumulators finished
                    self.u = 0  # interleaved: store-unit counter (data-VGPR bank)

                def done(self, mi, ii, sl, ji):
                    if _CILV:
                        n = self.pend.get((ii, sl), 0) + 1
                        self.pend[(ii, sl)] = n
                        if n == ntb:
                            self.q.append((mi, ii, sl, None))
                        return
                    self.q.append((mi, ii, sl, ji))

                def emit(self, mi, n=_CRATE):
                    out = []
                    while n > 0:
                        if not self.ln:
                            if not self.q or (mi is not None and self.q[0][0] + _CAGE > mi):
                                break
                            _, ii, sl, ji = self.q.pop(0)
                            if _CILV:
                                self.ln.append("s_waitcnt vmcnt(4)")
                            if ii != self.cur:
                                self.ln += cst_rows(ii, ii % 2)
                                self.cur = ii
                            if _CILV:
                                self.ln += cst_wide(ii, sl, ii % 2, self.u)
                                self.u += 1
                            else:
                                self.ln += cst_group(ii, sl, ji, ii % 2)
                        k = min(n, len(self.ln))
                        out += self.ln[:k]
                        self.ln = self.ln[k:]
                        n -= k
                    return out

                def flush(self):
                    return self.emit(None, 1 << 30)

            def emit_inplace(nxt_buf, g2sl, half=False, drop_s=False, refill=True, cstq=None, zero=False):
                # NEXT-K in-place refill, blocked-diagonal (4 A-rows x 8 N-cols); GAVOID: g2s in no-refill slots.
                bm, bn = 4, 8
                ncol = 2 * ntb
                nib = nta // bm
                ncb = ncol // bn
                quads = []
                for D in range(nib + ncb - 1):
                    for iib in range(nib):
                        cb = D - iib
                        if 0 <= cb < ncb:
                            for di in range(bm):
                                for dj in range(bn):
                                    ii = iib * bm + di
                                    col = cb * bn + dj
                                    if half and col // ntb:
                                        continue  # R half: padding columns
                                    quads.append((ii, col // ntb, col % ntb))
                nsub_e = n_sub - 1 if drop_s else n_sub
                cells = []
                for q in quads:
                    for s in range(nsub_e):
                        cells.append(q + (s,))
                mlist = []
                for ii, sl, ji, s in cells:
                    tb = t_bl if sl == 0 else t_br
                    sbfn = sbl_t if sl == 0 else sbr_t
                    q = sl * nq + ii * ntb + ji
                    oa, ob = ii % 4, ji
                    at = t_a + ii * n_sub + s
                    bt = tb + ji * n_sub + s
                    sat = sa_t(s, ii // 4)
                    sbt = sbfn(s)
                    # A quad's opening k sub-step (cells run s-ascending per quad) may take the
                    # src2 immediate 0 instead of its own accumulator: same result bit for bit,
                    # and it makes the pre-loop AGPR clear dead.
                    acc_in = "0" if (zero and s == 0) else f"${q}"
                    if _TACC:  # acc = C^T: src0<->src1, scales, op_sel[0]<->op_sel[1]
                        osel = f"op_sel:[{ob & 1},{oa & 1},0] op_sel_hi:[{(ob >> 1) & 1},{(oa >> 1) & 1},0]"
                        mline = (
                            f"v_mfma_scale_f32_16x16x128_f8f6f4 ${q}, ${bt}, ${at}, {acc_in}, "
                            f"${sbt}, ${sat} {osel} cbsz:4 blgp:4"
                        )
                    else:
                        osel = f"op_sel:[{oa & 1},{ob & 1},0] op_sel_hi:[{(oa >> 1) & 1},{(ob >> 1) & 1},0]"
                        mline = (
                            f"v_mfma_scale_f32_16x16x128_f8f6f4 ${q}, ${at}, ${bt}, {acc_in}, "
                            f"${sat}, ${sbt} {osel} cbsz:4 blgp:4"
                        )
                    mlist.append((mline, at, bt, sat, sbt))
                last = {}
                for mi, (_ml, at, bt, sat, sbt) in enumerate(mlist):
                    last[at] = mi
                    last[bt] = mi
                    last[sat] = mi
                    last[sbt] = mi
                mid = set(t for t in last if t_a <= t < t_sc)  # operands (scales VGPR-direct)
                _gset = {}
                if g2sl:
                    _rfslot = set()
                    _rf = set()
                    for mi, (ml, at, bt, sat, sbt) in enumerate(mlist):
                        for rt in (at, bt, sat, sbt):
                            if rt in mid and last[rt] == mi and rt not in _rf:
                                _rfslot.add(mi)
                                _rf.add(rt)
                    _free = [mi for mi in range(len(mlist)) if mi not in _rfslot]
                    _n = len(g2sl)
                    _fgap = max(len(_free) // max(_n, 1), 1)
                    for _k, _fi in enumerate(_free):
                        if (_k % _fgap == 0) and len(_gset) < _n:
                            _gset[_fi] = len(_gset)
                out = []
                gi = 0
                refilled = set()
                for mi, (ml, at, bt, sat, sbt) in enumerate(mlist):
                    out.append(ml)
                    if refill:
                        for rt in (at, bt, sat, sbt):
                            if rt in mid and last[rt] == mi and rt not in refilled:
                                out.append(ds_line(nxt_buf, rt))
                                refilled.add(rt)
                    if g2sl and mi in _gset and gi < len(g2sl):
                        out.append(g2sl[gi])
                        gi += 1
                    if cstq is not None:
                        _ii, _sl, _ji, _s = cells[mi]
                        if _s == nsub_e - 1:
                            cstq.done(mi, _ii, _sl, _ji)
                        out += cstq.emit(mi)
                while gi < len(g2sl):
                    out.append(g2sl[gi])
                    gi += 1
                if refill:
                    for tt in range(t_a, NT + set_sz):  # end drain: refill still-pending temps
                        if half and t_br <= tt < t_sc:
                            continue  # R half: the variant never reads these fragments
                        if tt not in refilled:
                            out.append(ds_line(nxt_buf, tt))
                if cstq is not None:
                    out += _CST_HAZ + cstq.flush()
                return out

            _ipend = f"s_waitcnt vmcnt({_WLV}) lgkmcnt({_ELGK})\ns_barrier"

            # VGPR-direct scale prefetch: emit_sc_vgpr loads scale dwords to the pinned set (no LDS/ds_read).
            _pbsc = _PINBASE  # scale VGPRs pinned first (PINSC), at PINBASE
            _scw = 2 * n_sub  # scale dwords per operand (2 region groups x n_sub subs)
            _scwx = {1: "", 2: "x2", 4: "x4"}.get(_scw, f"x{_scw}")  # buffer_load width suffix

            def emit_sc_vgpr(tb):
                p = _pbsc + tb
                return [
                    f"buffer_load_dword{_scwx} v[{p}:{p + _scw - 1}], ${i_scvoff}, ${i_scrsa}, ${o_sca[0]} offen",
                    f"buffer_load_dword{_scwx} v[{p + _scw}:{p + 2 * _scw - 1}], ${i_scvoff}, ${i_scrsb}, ${o_sca[2]} offen",
                ]

            _scvstep = 64 * (2 * n_sub) * 4  # lane-contig kk stride in bytes

            def _scv_adv():
                return [
                    f"s_add_u32 ${o_sca[0]}, ${o_sca[0]}, {_scvstep}",
                    f"s_add_u32 ${o_sca[2]}, ${o_sca[2]}, {_scvstep}",
                ]

            # COOP 2-deep pipeline: each wave loads one group to SC_lds, s_barrier, then ds_reads A+B.
            def emit_sc_coop_g2s(buf):
                # one wave -> one group (4 dwords/lane) into SC_lds[buf] slot=wave_id.
                return [
                    f"s_add_u32 m0, ${i_scgb[buf]}, 0\n"
                    f"buffer_load_dwordx4 ${i_scvoff}, ${i_scrsa}, ${o_sca[0]} offen lds"
                ]

            def emit_sc_coop_ds(tb, buf):
                # ds_read this wave's A (slot wave_m) + B (slot 2+wave_n) groups from
                # SC_lds[buf] into the pinned scale set at PINBASE+tb.
                p = _pbsc + tb
                off = buf * sc_buf_stride
                _o = f" offset:{off}" if off else ""
                return [
                    f"ds_read_b128 v[{p}:{p + _scw - 1}], ${i_scrb[0]}{_o}",
                    f"ds_read_b128 v[{p + _scw}:{p + 2 * _scw - 1}], ${i_scrb[1]}{_o}",
                ]

            # coop phase barrier: full drain so the 1-ahead scale ds_read (lgkm, shares
            # the counter with operand ds_reads) is guaranteed complete before the next
            # phase's first MFMA.
            _ipend_coop = "s_waitcnt vmcnt(0) lgkmcnt(0)\ns_barrier"

            # deferred tail of the operand prefill: last ring slot issued after k=0 scale loads (hides latency).
            _APRE = _SPLIT  # the split ring also defers A's last slot
            _NPRE = (nsa if _APRE else 0) + 2 * nsb

            def emit_g2s_pre():
                _bo = not _APRE
                _sb = [
                    f"s_sub_u32 ${o_ta}, ${i_sa0}, ${i_kstep}",
                    f"s_sub_u32 ${o_tbl}, ${i_sbl0}, ${i_kstep}",
                    f"s_sub_u32 ${o_tbr}, ${i_sbr0}, ${i_kstep}",
                ]
                if not _SPLIT:
                    return _sb + emit_g2s(1, o_ta, o_tbl, o_tbr, b_only=_bo)
                r = _sb + emit_g2s(1, o_ta, o_tbl, o_tbr, only_rg=0, b_only=_bo)
                if _NOD == 3:
                    r += [
                        f"s_sub_u32 ${o_ta}, ${i_sa0}, 128",
                        f"s_sub_u32 ${o_tbl}, ${i_sbl0}, 128",
                        f"s_sub_u32 ${o_tbr}, ${i_sbr0}, 128",
                    ]
                return r + emit_g2s(1, o_ta, o_tbl, o_tbr, only_rg=1, b_only=_bo)

            L = [
                f"s_mov_b32 ${o_cnt}, {2 if (_KPEEL or _OPEEL) else 0}",
                f"s_mov_b32 ${o_sa}, ${i_sa0}",
                f"s_mov_b32 ${o_sbl}, ${i_sbl0}",
                f"s_mov_b32 ${o_sbr}, ${i_sbr0}",
            ]
            for g in range(4):
                L.append(f"s_mov_b32 ${o_sca[g]}, ${i_sca0[g]}")
            if _RTPEEL:
                L.append(f"s_sub_u32 ${o_npv}, ${i_nval}, 2")
            if _ROT:
                for t in range(3):
                    for j in range(3):
                        L.append(f"s_mov_b32 ${o_pod[t][j]}, ${i_od[t][(j + 1) % 3]}")
                for t in range(2):
                    q = o_nq[t]
                    L.append(f"v_mov_b32 ${q[0]}, ${i_qu[t]}")
                    L.append(f"v_lshlrev_b32 ${q[1]}, 1, ${i_qu[t]}")
                    L.append(f"v_mov_b32 ${q[2]}, 0")
                L += emit_bases(0) + emit_bases(1)
            # in-place double-buffer prologue: read buf0 (k=0) into set0 before the loop.
            if _COOP:
                L += emit_g2s_pre()
                L.append("s_waitcnt vmcnt(0) lgkmcnt(0)")
                L.append("s_barrier")
                L += emit_ds(0, 0)
                L += emit_sc_coop_g2s(0)
                L += _scv_adv()
                if (ki is None) or (ki >= 2):
                    L += emit_sc_coop_g2s(1)
                    L += _scv_adv()
                L.append("s_waitcnt vmcnt(0)")
                L.append("s_barrier")
                L += emit_sc_coop_ds(0, 0)
                L.append("s_waitcnt lgkmcnt(0)")
            else:
                # VGPR-direct scale prologue: set A = phase-A iter0 scales.
                L += emit_sc_vgpr(0) + _scv_adv()
                L += emit_g2s_pre()
                L.append(f"s_waitcnt vmcnt({_NPRE}) lgkmcnt(0)")
                L.append("s_barrier")
                L += emit_ds(0, 0)
                L.append("s_waitcnt vmcnt(0) lgkmcnt(0)")
                L.append("s_barrier")
            if _has_loop:
                # unroll-2 body (phase A even-k, phase B odd-k); scale loads lead the mfma stream.
                def emit_phase_a(half, zero=False):
                    # phase A: consume set0; g2s P+2 -> LDS0, ds_read P+1 (LDS1) -> set1.
                    _scb[0] = 0
                    if _COOP:
                        _scA = emit_sc_coop_g2s(0) + emit_sc_coop_ds(nsct, 1)
                    else:
                        _scA = emit_sc_vgpr(nsct)
                    _gA = emit_g2s(0, o_sa, o_sbl, o_sbr, half and half_g2s)
                    if _ROT:  # rotate first (phase A's own g2s dest is the new head)
                        _gA = emit_rot() + mix_g2s(_gA, emit_bases(0))
                    return _scA + emit_inplace(1, _gA, half, zero=zero) + _scv_adv()

                def emit_bsoff():
                    # phase B's g2s soffsets (this pair's odd 256-K block)
                    return [
                        f"s_add_u32 ${o_ta}, ${o_sa}, ${i_kstep}",
                        f"s_add_u32 ${o_tbl}, ${o_sbl}, ${i_kstep}",
                        f"s_add_u32 ${o_tbr}, ${o_sbr}, ${i_kstep}",
                    ]

                def emit_acc_clear(lo, hi):
                    # Explicit clear for accumulators no src2-immediate MFMA reaches (see emit_head).
                    return [
                        f"v_accvgpr_write_b32 a{4 * q + e}, 0"
                        for q in range_constexpr(lo, hi)
                        for e in range_constexpr(4)
                    ]

                def emit_head(half):
                    """Peeled copy of the first phase A with the accumulators written rather than
                    accumulated, then a jump into the loop body at phase B.  The dynamic phase
                    order (A B A B ...) and the trip count are unchanged, so every soffset/ring
                    rotation still advances exactly once per phase."""
                    B = emit_phase_a(half, zero=True)
                    if half and _has_tail:
                        # The R half sits out this body but the shared tail still accumulates into it.
                        B += emit_acc_clear(nq, NT)
                    return B + [_ipend] + emit_bsoff() + ["s_branch 7f"]

                def emit_loop(lbl, half):
                    B = [f"{lbl}:"]
                    B += emit_phase_a(half)
                    B.append(_ipend)
                    B += emit_bsoff()
                    if _ZACC:
                        B.append("7:")  # loop entry for the peeled head phase A
                    # phase B: consume set1; g2s P+2 -> LDS1, ds_read P+1 (LDS0) -> set0.
                    _scb[0] = nsct
                    if _COOP:
                        _scB = emit_sc_coop_g2s(1) + emit_sc_coop_ds(0, 0)
                    else:
                        _scB = emit_sc_vgpr(0)
                    _gB = emit_g2s(1, o_ta, o_tbl, o_tbr, half and half_g2s)
                    if _ROT:
                        _gB = mix_g2s(_gB, emit_bases(1))
                    B += _scB + emit_inplace(0, _gB, half)
                    B += _scv_adv()
                    B.append(_ipend)
                    for _so in (o_sa, o_sbl, o_sbr):
                        B.append(f"s_add_u32 ${_so}, ${_so}, ${i_kstep}")
                        B.append(f"s_add_u32 ${_so}, ${_so}, ${i_kstep}")
                    B.append(f"s_add_u32 ${o_cnt}, ${o_cnt}, 2")
                    B.append(f"s_cmp_lt_u32 ${o_cnt}, ${o_npv if _RTPEEL else i_nval}")
                    B.append(f"s_cbranch_scc1 {lbl}b")
                    return B

                def emit_peel(half):
                    # Peeled last iteration: keeps phase A's refill + scale prefetch, drops the g2s (past K's end).
                    B = [f"s_waitcnt vmcnt(0) lgkmcnt({_ELGK})", "s_barrier"]
                    _scb[0] = 0
                    B += emit_sc_vgpr(nsct) + emit_inplace(1, [], half)
                    B.append(f"s_waitcnt vmcnt(0) lgkmcnt({_ELGK})")
                    _scb[0] = nsct
                    B += emit_inplace(0, [], half, drop_s=True, refill=False)
                    return B

                def emit_peel_fold(half, sw=0):
                    """Peeled last iteration with the C store folded into its MFMA stream, so each
                    accumulator is final after 3 MFMAs and its stores ride the later ones.
                    Per-accumulator MFMA order is unchanged, so C is bit-identical."""
                    nsl = 1 if half else 2
                    cq = CstSched()
                    sc_f, sc_t = (0, nsct) if sw == 0 else (nsct, 0)

                    def aw(ii, s):  # rolling A window: k sub-steps + the trailing block's
                        return t_a + (ii % 2) * (n_sub + 2) + s

                    def b1(sl, ji):  # trailing block's B, in the A slots left over
                        return t_a + 2 * (n_sub + 2) + sl * ntb + ji

                    def rd_a(ii, s):
                        buf, ss = (sw, s) if s < n_sub else (1 - sw, 0)
                        return f"ds_read_b128 ${aw(ii, s)}, ${i_ab[buf][ss]} offset:{ii * ts_a}"

                    def rd_b1(sl, ji):
                        bb, bo = b_rd(sl, 1 - sw, 0, ji)
                        return f"ds_read_b128 ${b1(sl, ji)}, ${bb} offset:{bo}"

                    def mfl(ii, sl, ji, s):
                        q = sl * nq + ii * ntb + ji
                        oa, ob = ii % 4, ji
                        se = s if s < n_sub else 0
                        scb = sc_f if s < n_sub else sc_t
                        at = aw(ii, s)
                        bt = (t_bl if sl == 0 else t_br) + ji * n_sub + se if s < n_sub else b1(sl, ji)
                        sat = t_sc + scb + (ii // 4) * n_sub + se
                        sbt = t_sc + scb + (2 + sl) * n_sub + se
                        if _TACC:  # acc = C^T (swap operands/scales/op_sel)
                            osel = (
                                f"op_sel:[{ob & 1},{oa & 1},0] op_sel_hi:[{(ob >> 1) & 1},{(oa >> 1) & 1},0]"
                            )
                            return (
                                f"v_mfma_scale_f32_16x16x128_f8f6f4 ${q}, ${bt}, ${at}, ${q}, "
                                f"${sbt}, ${sat} {osel} cbsz:4 blgp:4"
                            )
                        osel = f"op_sel:[{oa & 1},{ob & 1},0] op_sel_hi:[{(oa >> 1) & 1},{(ob >> 1) & 1},0]"
                        return (
                            f"v_mfma_scale_f32_16x16x128_f8f6f4 ${q}, ${at}, ${bt}, ${q}, "
                            f"${sat}, ${sbt} {osel} cbsz:4 blgp:4"
                        )

                    B = ["s_waitcnt vmcnt(0) lgkmcnt(0)", "s_barrier"]
                    B += emit_sc_vgpr(sc_t)
                    for sl in range(nsl):
                        for ji in range(ntb):
                            B.append(rd_b1(sl, ji))
                    for s in range(n_sub + 1):
                        B.append(rd_a(0, s))
                    mi = 0
                    for ii in range(nta):
                        B.append("s_waitcnt lgkmcnt(0)")
                        if ii == 0:
                            B.append("s_waitcnt vmcnt(0)")  # trailing scale; no store in flight
                        j = 0
                        for sl in range(nsl):
                            for ji in range(ntb):
                                for s in range(n_sub + 1):
                                    B.append(mfl(ii, sl, ji, s))
                                    if s == n_sub:
                                        cq.done(mi, ii, sl, ji)
                                    B += cq.emit(mi)
                                    mi += 1
                                    j += 1
                                    if j == _ARD and ii + 1 < nta:
                                        for ss in range(n_sub + 1):
                                            B.append(rd_a(ii + 1, ss))
                    B += _CST_HAZ + cq.flush()
                    return B

                def emit_peel_rt(half, lbl):
                    """Hw-loop stopped two phases early plus the peeled copy, for a runtime trip
                    count. The peel consumes k-blocks already in LDS, so it issues no g2s and
                    an in-flight store cannot serialise a wait through the unified vmcnt.
                    A trip count too small for the loop also skips the src2-immediate head, so
                    that branch clears the accumulators itself."""
                    B = [f"s_cmp_lt_u32 ${i_nval}, 4", f"s_cbranch_scc1 {'8' if _ZACC else lbl}f"]
                    if _ZACC:
                        B += emit_head(half)
                    B += emit_loop("2" if half else "1", half)
                    B.append(f"{lbl}:")
                    B.append(f"s_waitcnt vmcnt(0) lgkmcnt({_ELGK})")
                    B.append("s_barrier")
                    _scb[0] = 0
                    B += emit_sc_vgpr(nsct) + emit_inplace(1, [], half)
                    B.append(f"s_waitcnt vmcnt(0) lgkmcnt({_ELGK})")
                    _scb[0] = nsct
                    B += emit_inplace(0, [], half, refill=False, cstq=CstSched())
                    if _ZACC:
                        # kept out of line so the loop-taken path never fetches through it
                        B += (
                            ["s_branch 9f", "8:"]
                            + emit_acc_clear(0, nq if half else NT)
                            + [f"s_branch {lbl}b", "9:"]
                        )
                    return B

                def emit_peel_odd(half):
                    """Odd trip count: peel the last full pair's phase A so the trailing half
                    k-block merges into the phase behind it, giving the fused store the same
                    whole-peel window an even KI gets. The peeled phase A is an exact copy."""
                    return emit_phase_a(half) + emit_peel_fold(half, sw=1)

                # Boundary N-block variant: same drain/barrier sequence, R-half MFMAs dropped.
                if half_n is not None:
                    L.append(f"s_cmp_lg_u32 ${i_hn}, 0")
                    L.append("s_cbranch_scc1 3f")
                _peel = emit_peel_fold if _CST else emit_peel

                def emit_body(lbl, half):
                    if _RTPEEL:
                        return emit_peel_rt(half, lbl)
                    B = (emit_head(half) if _ZACC else []) + emit_loop("2" if half else "1", half)
                    if _KPEEL:
                        B += _peel(half)
                    elif _OPEEL:
                        B += emit_peel_odd(half)
                    return B

                L += emit_body("5", False)
                if half_n is not None:
                    L.append("s_branch 4f")
                    L.append("3:")
                    L += emit_body("6", True)
                    L.append("4:")

            if _has_tail:
                # odd-KI trailing phase-A (MFMA-only): operands + set0 scales already staged, just drain and run.
                L.append("s_waitcnt vmcnt(0) lgkmcnt(0)")
                _scb[0] = 0
                _nst = n_sub - 1 if half_k else n_sub
                _cq = CstSched() if _CST else None
                _mi = 0
                _bm, _bn = 4, 8  # match loop-body block (see emit_inplace)
                _ncol = 2 * ntb
                _nib = nta // _bm
                _ncb = _ncol // _bn
                for _D in range(_nib + _ncb - 1):
                    for _iib in range(_nib):
                        _cb = _D - _iib
                        if 0 <= _cb < _ncb:
                            for _di in range(_bm):
                                for _dj in range(_bn):
                                    for _s in range(_nst):
                                        _ii = _iib * _bm + _di
                                        _col = _cb * _bn + _dj
                                        _sl = _col // ntb
                                        _ji = _col % ntb
                                        _tb = t_bl if _sl == 0 else t_br
                                        _sbfn = sbl_t if _sl == 0 else sbr_t
                                        _q = _sl * nq + _ii * ntb + _ji
                                        _oa, _ob = _ii % 4, _ji
                                        _at = t_a + _ii * n_sub + _s
                                        _bt = _tb + _ji * n_sub + _s
                                        _sat = sa_t(_s, _ii // 4)
                                        _sbt = _sbfn(_s)
                                        # tail-only body (no loop ran): its opening sub-step is the
                                        # quad's first MFMA, so it can carry the src2 immediate 0.
                                        _ai = "0" if (_ZACC and not _has_loop and _s == 0) else f"${_q}"
                                        if _TACC:  # acc = C^T (swap operands/scales/op_sel)
                                            _osel = (
                                                f"op_sel:[{_ob & 1},{_oa & 1},0] "
                                                f"op_sel_hi:[{(_ob >> 1) & 1},{(_oa >> 1) & 1},0]"
                                            )
                                            L.append(
                                                f"v_mfma_scale_f32_16x16x128_f8f6f4 ${_q}, ${_bt}, "
                                                f"${_at}, {_ai}, ${_sbt}, ${_sat} {_osel} cbsz:4 blgp:4"
                                            )
                                        else:
                                            _osel = (
                                                f"op_sel:[{_oa & 1},{_ob & 1},0] "
                                                f"op_sel_hi:[{(_oa >> 1) & 1},{(_ob >> 1) & 1},0]"
                                            )
                                            L.append(
                                                f"v_mfma_scale_f32_16x16x128_f8f6f4 ${_q}, ${_at}, "
                                                f"${_bt}, {_ai}, ${_sat}, ${_sbt} {_osel} cbsz:4 blgp:4"
                                            )
                                        if _cq is not None:
                                            if _s == _nst - 1:
                                                _cq.done(_mi, _ii, _sl, _ji)
                                            L += _cq.emit(_mi)
                                        _mi += 1
                if _cq is not None:
                    L += _CST_HAZ + _cq.flush()
            # ── register pinning (PIN + PINSC): scales LOW (PINBASE), frags after ──
            # Bypasses the LLVM RA "Cannot decrease cascade number" crash and aligns
            # the scale literals to the PINBASE base that emit_sc_vgpr writes.
            _vtmp = ["=&v"] * ntmp2
            bv = _PINBASE
            for s in range(NSET):
                order = list(range(ntmp))
                _nsc2 = nsct * 2  # 2 ping-pong scale sets (VGPR-direct)
                for j in range(_nsc2):
                    _vtmp[s * set_sz + ntmp + j] = f"=&{{v{bv}}}"
                    bv += 1
                for j in order:  # frags: vector<4xi32> = 4 VGPR
                    _vtmp[s * set_sz + j] = f"=&{{v[{bv}:{bv + 3}]}}"
                    bv += 4
            _vtmp += ["=&v"] * _nvx  # split: rotating ds_read bases + ring offsets
            cons = ",".join(
                ([f"={{a[{4 * q}:{4 * q + 3}]}}" for q in o_acc] if _CST else ["=a"] * NT)
                + _vtmp
                + ["=&s"] * (12 + (10 if _ROT else 0))  # cnt+3soff+3tmp+4scsoff+1sctmp(+ring)
                + ((["=&s"] + ["=&v"] * 8) if _CST else [])  # fused store scratch
                + [f"=&{{v{_CDV + j}}}" for j in range(_NCDV)]  # wide store data pool
                + (["=&s"] if _RTPEEL else [])  # runtime peel loop bound
                + ["v"] * ((nbuf + 2 * nbuf_b) * n_sub)  # a(nbuf)/bl/br(nbuf_b) ds_read bases
                + ["s"] * (nbuf + 2 * nbuf_b)  # g2s dest bases
                + ["v"] * (nsa + nsb)  # voffsets
                + ["s", "s", "s", "v", "s"]  # rsrc_a, rsrc_b, kstep, scv, nval
                + ["s", "s", "s"]  # operand soffset inits A/BL/BR
                + ["v"] * _nscbuf  # scale LDS read base (reserved)
                + ["s"] * _nscbuf  # scale LDS g2s dest base (reserved)
                + ["s", "s"]  # scale rsrc A, B
                + ["v"]  # scale voffset
                + ["s", "s", "s", "s"]  # scale soffset inits (A-g0, A-g1, BL, BR)
                + (["s"] if half_n is not None else [])  # half-N variant selector
                + ["s"] * (3 * _NOD)  # odd-ring g2s dest bases
                + (["v"] * 2 if _ROT else [])  # odd-ring per-lane slot strides
                + (["s", "s", "v", "s"] if _CST else [])  # C SRDs (L,R), voffset, row bytes
                + (["v"] * (2 * nbuf_b * n_sub) if _BSPL else [])  # even-region B bases
                + ([] if _ZACC else [str(q) for q in o_acc])  # tied accs (src2 imm 0 instead)
            )
            st = (
                "!llvm.struct<("
                + ", ".join(
                    ["vector<4xf32>"] * NT
                    + (["vector<4xi32>"] * ntmp + ["i32"] * nsct + ["i32"] * _scextra) * NSET
                    + ["i32"] * _nvx
                    + ["i32"] * (12 + (10 if _ROT else 0))
                    + ["i32"] * (9 if _CST else 0)
                    + ["i32"] * _NCDV
                    + ["i32"] * (1 if _RTPEEL else 0)
                )
                + ")>"
            )
            _cache[key] = ("\n".join(L), cons, st)
        asm, cons, st = _cache[key]
        ins = []
        for b in range_constexpr(nbuf):  # A pool
            for s in range_constexpr(n_sub):
                ins.append(_raw(a_base[b][s]))
        for fr in (bl_base, br_base):  # B pool
            for b in range_constexpr(nbuf_b):
                for s in range_constexpr(n_sub):
                    ins.append(_raw(fr[b][s]))
        for b in range_constexpr(nbuf):  # g2s A dest
            ins.append(_raw(abase[b]))
        for fr in (blbase, brbase):  # g2s B dest
            for b in range_constexpr(nbuf_b):
                ins.append(_raw(fr[b]))
        for v in gl_a:
            ins.append(_raw(v))
        for v in gl_b:
            ins.append(_raw(v))
        ins.append(_raw(rsrc_a))
        ins.append(_raw(rsrc_b))
        ins.append(_raw(kstep))
        ins.append(_raw(scv))
        ins.append(_raw(nval))
        ins.append(_raw(soff0))
        ins.append(_raw(soff0_bl))
        ins.append(_raw(soff0_br))
        for b in range_constexpr(_nscbuf):
            ins.append(_raw(sc_rb[b]))  # scale LDS read base (reserved)
        for b in range_constexpr(_nscbuf):
            ins.append(_raw(sc_gb[b]))  # scale LDS g2s dest base (reserved)
        ins.append(_raw(sc_rsa))
        ins.append(_raw(sc_rsb))  # scale rsrc
        ins.append(_raw(sc_voff))  # scale voffset
        for g in range_constexpr(4):
            ins.append(_raw(sc_soff0[g]))  # scale soffset inits
        if half_n is not None:
            ins.append(_raw(half_n))
        if _SPLIT:
            for _fr in range_constexpr(3):
                for _j in range_constexpr(_NOD):
                    ins.append(_raw(split[_fr][_j]))  # odd-ring g2s dest bases
            if _ROT:
                ins.append(_raw(split[3]))
                ins.append(_raw(split[4]))  # odd-ring slot strides (A, B)
        if _CST:
            for _ci in range_constexpr(4):
                ins.append(_raw(cst[_ci]))  # C SRD L/R, per-lane voffset, row bytes
        if _BSPL:
            for _fr in b_base_even:  # even-region B bases (BL, BR)
                for _b in range_constexpr(nbuf_b):
                    for _s in range_constexpr(n_sub):
                        ins.append(_raw(_fr[_b][_s]))
        if not _ZACC:
            for q in range_constexpr(nq):
                ins.append(_raw(cL[q]))
            for q in range_constexpr(nq):
                ins.append(_raw(cR[q]))
        r = _llvm.inline_asm(ir.Type.parse(st), ins, asm, cons, has_side_effects=True)
        o = [Vec(_llvm.extractvalue(ir.Type.parse("vector<4xf32>"), r, [q])) for q in range_constexpr(nq * 2)]
        return o[:nq], o[nq:]


# ── Compile factory (NT, BLOCK_M=BLOCK_N=BLOCK_K=256) ─────────────────────────


def _build_mxfp4_gemm_kernel(
    *,
    K: int,
    group_m: int = 4,
    num_xcds: int = 8,
    group_n: int = 0,
    wlv: int = 10,
    elgk: int = 9,
    coop: bool = False,
    ksplit: int = 1,
    taccw: bool = False,
    out_fp16: bool = False,
    beta_is_one: bool = False,  # epilogue accumulates (C += acc) instead of overwriting
):
    BLOCK_M = 256
    BLOCK_N = 256
    BLOCK_K = 256
    # Split-K stores partials into a scratch row band that the host then reduces, so the
    # accumulate belongs to that reduce, not to this store.
    assert not (beta_is_one and ksplit > 1), "split-K accumulates in the host reduce, not the epilogue"
    # const_expr() resolves compile-time branches from LOCALS/params, not reliably from
    # module globals -> alias the per-shape epilogue selection to locals before traced use.
    _l_taccw = taccw  # autotune-selected per shape (never-regress epilogue variant axis)
    _l_tacc = _l_taccw  # TACCW needs the acc=C^T MMA operand swap
    swizzle = True
    assert BLOCK_K % 128 == 0 and K % BLOCK_K == 0
    # Split-K: each WG computes a K/ksplit slice into workspace[split], host reduces -- fills
    # CUs on few-tile large-K shapes. Only trip count + per-split K-start bases change; the asm is untouched.
    assert K % ksplit == 0, f"K={K} not divisible by ksplit={ksplit}"
    K_loop = K // ksplit
    assert K_loop % BLOCK_K == 0, f"K/ksplit={K_loop} not a multiple of {BLOCK_K}"

    NBB = const_expr(2)  # B/SC pool (unroll-2)
    NABUF = const_expr(2)  # A pool (unroll-2)
    OCC = const_expr(1)  # 1 wave/SIMD -> full 256-AGPR file for the accumulator

    KI = K_loop // BLOCK_K  # loop trip count = per-split 256-K blocks (full K when ksplit=1)
    N_SUB = BLOCK_K // 128
    BPR = BLOCK_K // 2  # packed-fp4 bytes per K-iter row in LDS
    KSTEP = BPR
    K2 = K // 2  # packed-fp4 gmem row stride (bytes) -- FULL K (rows span all K)
    _AB_SPLIT_STEP = K_loop // 2
    _SC_SPLIT_STEP = KI * (64 * (2 * N_SUB) * 4)

    N_TILES_A = BLOCK_M // 32  # 8: wave_m covers 128 M-rows
    LDS_BN_HALF = BLOCK_N // 2  # 128: slice width
    N_TILES_BH = LDS_BN_HALF // 32  # 4: wave_n covers 64 N-cols/slice

    LDS_ROW_STRIDE = BPR
    a_lds_size = BLOCK_M * LDS_ROW_STRIDE  # 256 rows
    bh_lds_size = LDS_BN_HALF * LDS_ROW_STRIDE  # 128 rows per B half

    _ROWS_PER_STEP = 64 // (BPR // 16) * (256 // 64)  # n_waves = 256//64 = 4
    N_LDS_STEPS_A = BLOCK_M // _ROWS_PER_STEP
    N_LDS_STEPS_BH = LDS_BN_HALF // _ROWS_PER_STEP

    _PRELL = const_expr(2)  # operand buffers prefilled (k=0..PRELL-1)
    _NSCBUF = const_expr(2)
    K128 = const_expr(K // 128)
    _SCBUF = 4 * 4 * (BLOCK_K // 128) * 64  # n_waves * groups * n_sub * 64 dwords
    _SCW = const_expr(4 * N_SUB * 64)  # dwords per wave-region per scale buffer

    _anns = {f"A_lds{i}": fx.Array[fx.Float8E4M3FN, a_lds_size, 16] for i in range_constexpr(NABUF)}
    for _b in range_constexpr(NBB):
        _anns[f"BL_lds{_b}"] = fx.Array[fx.Float8E4M3FN, bh_lds_size, 16]
    for _b in range_constexpr(NBB):
        _anns[f"BR_lds{_b}"] = fx.Array[fx.Float8E4M3FN, bh_lds_size, 16]
    for _b in range_constexpr(_NSCBUF):
        _anns[f"SC_lds{_b}"] = fx.Array[fx.Int32, _SCBUF, 16]
    SharedStorageFp4_4w = fx.struct(type("SharedStorageFp4_4w", (), {"__annotations__": _anns}))

    @flyc.kernel(known_block_size=[256, 1, 1])
    def kernel_gemm_4w(
        A: fx.Tensor,
        B_T: fx.Tensor,
        C: fx.Tensor,
        A_scale: fx.Tensor,
        B_scale: fx.Tensor,
        c_m: fx.Int32,
        c_n: fx.Int32,
    ):
        F8_IR_t = fx.Float8E4M3FN.ir_type
        lds = fx.SharedAllocator().allocate(SharedStorageFp4_4w).peek()
        A_buf = [getattr(lds, f"A_lds{i}") for i in range_constexpr(NABUF)]
        BL_buf = [getattr(lds, f"BL_lds{i}") for i in range_constexpr(NBB)]
        BR_buf = [getattr(lds, f"BR_lds{i}") for i in range_constexpr(NBB)]

        lane_id = fx.thread_idx.x % 64
        wave_id = fx.thread_idx.x // 64
        wave_m = wave_id // 2
        wave_n = wave_id % 2
        # ── Tile-INDEPENDENT setup (hoisted out of the per-tile body; every value
        # below depends only on the fixed LDS buffers / wave id / whole-tensor
        # resources, NOT block_m/n) ──
        mfma = MfmaScaleFp4(N_TILES_A, N_TILES_BH, packed=True, wlv=wlv, elgk=elgk, coop=coop, tacc=_l_tacc)

        gl_off_a = fp4_g2s_offsets(lane_id, wave_id, K, N_LDS_STEPS_A, BPR, swizzle=swizzle)
        gl_off_b = fp4_g2s_offsets(lane_id, wave_id, K, N_LDS_STEPS_BH, BPR, swizzle=swizzle)
        # Operand SRDs/loaders are rebased per-tile (_bind): the tile's row/col base exceeds int32 for large M*K/N*K.
        _ld: dict = {}

        def _bind(bm, bn):
            a_base_e = arith.index_cast(T.index, bm * fx.Int32(BLOCK_M)) * arith.index(K2)
            b_base_e = arith.index_cast(T.index, bn * fx.Int32(BLOCK_N)) * arith.index(K2)
            a_nrec = (
                arith.index_cast(T.index, c_m) - arith.index_cast(T.index, bm * fx.Int32(BLOCK_M))
            ) * arith.index(K2)
            b_nrec = (
                arith.index_cast(T.index, c_n) - arith.index_cast(T.index, bn * fx.Int32(BLOCK_N))
            ) * arith.index(K2)
            gA, _ld["rsrc_a"] = make_fp8_rebased_tensor_and_srd(A, F8_IR_t, a_base_e, a_nrec)
            gB, _ld["rsrc_b"] = make_fp8_rebased_tensor_and_srd(B_T, F8_IR_t, b_base_e, b_nrec)
            a_div = fx.logical_divide(gA, fx.make_layout(1, 1))
            b_div = fx.logical_divide(gB, fx.make_layout(1, 1))
            _ld["a_g2s"] = G2SLoader(a_div, gl_off_a, N_LDS_STEPS_A, F8_IR_t, wave_id)
            _ld["bl_g2s"] = G2SLoader(b_div, gl_off_b, N_LDS_STEPS_BH, F8_IR_t, wave_id)
            _ld["br_g2s"] = G2SLoader(b_div, gl_off_b, N_LDS_STEPS_BH, F8_IR_t, wave_id)

        a_s2r = S2RLoaderFp4(wave_m, N_TILES_A, LDS_ROW_STRIDE, swizzle=swizzle)
        b_s2r = S2RLoaderFp4(wave_n, N_TILES_BH, LDS_ROW_STRIDE, swizzle=swizzle)

        # A scale: 8 M-tiles span 2 x 64-row groups (4 tiles each). Packed -> ONE i32
        # per group. group-span-ceil the dim so the floor in the loader's record count
        # covers the edge scale group.
        _qm = ((c_m + 63) // 64) * 64
        _qn = ((c_n + 63) // 64) * 64
        sa_s2r = ScaleS2RPacked(A_scale, _qm, K, 4)
        sb_s2r = ScaleS2RPacked(B_scale, _qn, K, 4)
        # split-K writes partials to workspace C[ksplit*M, N] (row band split*M); the
        # StoreCPlain c_rows only bounds the SRD (not used in the index), so widen it.
        _c_store_rows = c_m if const_expr(ksplit == 1) else c_m * fx.Int32(ksplit)
        # bf16/fp16 output: only the f32->out_ty cast in the store differs. Both the narrow
        # scalar store (generic ``.to``) and the wide TACCW store serve either dtype.
        _out_ty = fx.Float16 if out_fp16 else fx.BFloat16
        store_c = StoreCPlain(
            C, _c_store_rows, c_n, mfma.idx, N_TILES_A, N_TILES_BH, _out_ty, beta_is_one=beta_is_one
        )

        wave_m_off = wave_m * (N_TILES_A * 16)  # 0 or 128
        wave_n_off = wave_n * (N_TILES_BH * 16)  # 0 or 64
        SC_buf = [getattr(lds, f"SC_lds{b}") for b in range_constexpr(_NSCBUF)]

        # LDS read/g2s-dest bases + scale resources: all derived from the fixed LDS
        # buffers / wave id, so identical for every output tile -> compute once.
        a_base6 = [
            [a_s2r.base_addr(A_buf[b], s) for s in range_constexpr(N_SUB)] for b in range_constexpr(NABUF)
        ]
        bl_base6 = [
            [b_s2r.base_addr(BL_buf[b], s) for s in range_constexpr(N_SUB)] for b in range_constexpr(NBB)
        ]
        br_base6 = [
            [b_s2r.base_addr(BR_buf[b], s) for s in range_constexpr(N_SUB)] for b in range_constexpr(NBB)
        ]

        def _gbase(buf):
            v = fx.Int32(fx.ptrtoint(buf.ptr)) + fx.Int32(wave_id) * fx.Int32(1024)
            return rocdl.readfirstlane(T.i32, v)

        abase6 = [_gbase(A_buf[b]) for b in range_constexpr(NABUF)]
        blbase6 = [_gbase(BL_buf[b]) for b in range_constexpr(NBB)]
        brbase6 = [_gbase(BR_buf[b]) for b in range_constexpr(NBB)]
        gl_a6 = [fx.Int32(gl_off_a[st]) for st in range_constexpr(N_LDS_STEPS_A)]
        gl_b6 = [fx.Int32(gl_off_b[st]) for st in range_constexpr(N_LDS_STEPS_BH)]
        scv6 = fx.Int32(0x7F7F7F7F)
        _scrb_lane = lane_id
        sc_rb6 = [
            fx.ptrtoint(
                fx.add_offset(
                    SC_buf[b].ptr, fx.make_int_tuple(fx.Int32(wave_id) * fx.Int32(_SCW) + _scrb_lane)
                )
            )
            for b in range_constexpr(_NSCBUF)
        ]
        sc_gb6 = [
            rocdl.readfirstlane(
                T.i32,
                fx.Int32(
                    fx.ptrtoint(
                        fx.add_offset(SC_buf[b].ptr, fx.make_int_tuple(fx.Int32(wave_id) * fx.Int32(_SCW)))
                    )
                ),
            )
            for b in range_constexpr(_NSCBUF)
        ]
        _scrsa_v = sa_s2r.rsrc
        _scrsb_v = sb_s2r.rsrc
        sc_voff6 = lane_id * fx.Int32(8 * N_SUB)

        _SCSLOT = const_expr(4 * 64 * 4)  # 1024 B per LDS scale slot
        if const_expr(coop):
            # scalar wave_id so the cond is SCC (not VCC) -> arith.select on the two
            # SGPR rsrc descriptors lowers to s_cselect -> result stays in SGPR (a
            # buffer rsrc MUST be scalar; a VGPR rsrc is an invalid buffer operand).
            _wid_s = rocdl.readfirstlane(T.i32, wave_id)
            _w_lt2 = _wid_s < fx.Int32(2)
            coop_rsa = arith.select(_w_lt2, _scrsa_v, _scrsb_v)
            sc_gb6 = [
                rocdl.readfirstlane(
                    T.i32,
                    fx.Int32(
                        fx.ptrtoint(
                            fx.add_offset(
                                SC_buf[b].ptr,
                                fx.make_int_tuple(fx.Int32(wave_id) * fx.Int32(_SCSLOT // 4)),
                            )
                        )
                    ),
                )
                for b in range_constexpr(_NSCBUF)
            ]
            sc_rb6 = [
                fx.ptrtoint(
                    fx.add_offset(
                        SC_buf[0].ptr,
                        fx.make_int_tuple(_slot * fx.Int32(_SCSLOT // 4) + lane_id * fx.Int32(4)),
                    )
                )
                for _slot in (wave_m, fx.Int32(2) + wave_n)
            ]
        else:
            coop_rsa = _scrsa_v

        def _scsoff(base, extra):
            grp = (base + fx.Int32(extra)) // fx.Int32(64)
            return rocdl.readfirstlane(
                T.i32, (grp * fx.Int32(K128) + fx.Int32(_PRELL * N_SUB)) * fx.Int32(256)
            )

        # ── Per-tile closures (block_m/block_n -> offsets; fill; compute; store) ──
        def _offs(_pid):
            bm, bn = grouped_xcd_pid(
                _pid, c_m, c_n, BLOCK_M, BLOCK_N, group_m=group_m, num_xcds=num_xcds, group_n=group_n
            )
            _bind(bm, bn)  # rebase the operand SRDs/loaders on this tile's A/B base (int64)
            a_off = fx.Int32(0)  # tile A row / B col bases folded into the SRDs; only br's
            bl_off = fx.Int32(0)  # LDS-half column shift survives as an int32-safe residual.
            br_off = fx.Int32(LDS_BN_HALF * K2)
            sa_b = fx.Int32(bm * BLOCK_M + wave_m_off)
            sbl_b = fx.Int32(bn * BLOCK_N + wave_n_off)
            sbr_b = fx.Int32(bn * BLOCK_N + LDS_BN_HALF + wave_n_off)
            return (bm, bn, a_off, bl_off, br_off, sa_b, sbl_b, sbr_b)

        def _fill(o):
            _, _, a_off, bl_off, br_off, _, _, _ = o
            for _pp in range_constexpr(0, _PRELL):
                if const_expr(KI > _pp):
                    _ld["a_g2s"].load(A_buf[_pp], a_off + _pp * KSTEP)
            for _pp in range_constexpr(0, _PRELL - 1):
                if const_expr(KI > _pp):
                    _ld["bl_g2s"].load(BL_buf[_pp], bl_off + _pp * KSTEP)
                    _ld["br_g2s"].load(BR_buf[_pp], br_off + _pp * KSTEP)

        def _compute(o, _split=None):
            _, _, a_off, bl_off, br_off, sa_b, sbl_b, sbr_b = o
            accL = [mfma.zero_value] * (N_TILES_A * N_TILES_BH)
            accR = [mfma.zero_value] * (N_TILES_A * N_TILES_BH)
            soff6_a = rocdl.readfirstlane(T.i32, a_off + fx.Int32(_PRELL * KSTEP))
            soff6_bl = rocdl.readfirstlane(T.i32, bl_off + fx.Int32(_PRELL * KSTEP))
            soff6_br = rocdl.readfirstlane(T.i32, br_off + fx.Int32(_PRELL * KSTEP))
            # VGPR-direct scale soffsets: A-group0 (_soa) and B (_sob) per the wave's
            # region-group id; A-group1 (+64 rows) and BR keep the packed-group soffset.
            _sc1 = _scsoff(sa_b, 64)
            _sc3 = _scsoff(sbr_b, 0)
            _wia = sa_b // fx.Int32(128)
            _wib = (sbl_b // fx.Int32(256)) * fx.Int32(2) + (sbl_b % fx.Int32(256)) // fx.Int32(64)
            _soa = rocdl.readfirstlane(T.i32, _wia * fx.Int32(K128) * fx.Int32(512))
            _sob = rocdl.readfirstlane(T.i32, _wib * fx.Int32(K128) * fx.Int32(512))
            sc_soff06 = [_soa, _sc1, _sob, _sc3]
            _sc_rsa_arg = _scrsa_v
            if const_expr(coop):
                # per-wave group soffset: waves 0/1 -> A region (2*bm + wave_id),
                # waves 2/3 -> B region (2*bn + (wave_id-2)). The g2s reads ONE group
                # at this soffset; the in-asm coop path uses sc_soff0[0] only.
                bm_t, bn_t = o[0], o[1]
                _coop_reg = arith.select(
                    wave_id < fx.Int32(2),
                    fx.Int32(2) * bm_t + wave_id,
                    fx.Int32(2) * bn_t + (wave_id - fx.Int32(2)),
                )
                _coop_soff = rocdl.readfirstlane(T.i32, _coop_reg * fx.Int32(K128) * fx.Int32(512))
                sc_soff06 = [_coop_soff, _sc1, _sob, _sc3]
                _sc_rsa_arg = coop_rsa
            if const_expr(ksplit > 1):
                # shift every scale soffset to this split's K-start (region is full-K
                # contiguous; per-256-K advance is _scvstep, KI blocks per split).
                _scsh = rocdl.readfirstlane(T.i32, _split * fx.Int32(_SC_SPLIT_STEP))
                sc_soff06 = [rocdl.readfirstlane(T.i32, _x + _scsh) for _x in sc_soff06]
            return mfma.call_mxfp4_wholeloop(
                a_base6,
                bl_base6,
                br_base6,
                a_s2r.tile_stride,
                b_s2r.tile_stride,
                abase6,
                blbase6,
                brbase6,
                gl_a6,
                gl_b6,
                _ld["rsrc_a"],
                _ld["rsrc_b"],
                fx.Int32(KSTEP),
                scv6,
                accL,
                accR,
                N_SUB,
                N_LDS_STEPS_A,
                N_LDS_STEPS_BH,
                fx.Int32((KI // 2) * 2),
                soff6_a,
                soff6_bl,
                soff6_br,
                sc_rb6,
                sc_gb6,
                _sc_rsa_arg,
                _scrsb_v,
                sc_voff6,
                sc_soff06,
                ki=KI,
                sc_buf_stride=(_SCBUF * 4),
            )

        def _store(o, accL, accR, _split=None):
            bm, bn = o[0], o[1]
            base_row = bm * BLOCK_M + wave_m_off
            if const_expr(ksplit > 1):
                base_row = base_row + _split * c_m  # write to workspace row band split*M
            base_col_l = bn * BLOCK_N + wave_n_off
            base_col_r = bn * BLOCK_N + LDS_BN_HALF + wave_n_off
            if const_expr(_l_taccw):
                store_c.store_tacc_wide(accL, base_row, base_col_l)
                store_c.store_tacc_wide(accR, base_row, base_col_r)
                return
            store_c.store(accL, base_row, base_col_l)
            store_c.store(accR, base_row, base_col_r)

        def _split_shift(o, _split):
            # add this split's K-start to the A/B operand gmem offsets (row stride full-K).
            bm, bn, a_off, bl_off, br_off, sa_b, sbl_b, sbr_b = o
            _sh = _split * fx.Int32(_AB_SPLIT_STEP)
            return (bm, bn, a_off + _sh, bl_off + _sh, br_off + _sh, sa_b, sbl_b, sbr_b)

        if const_expr(ksplit > 1):
            # split-K: grid = total_tiles*ksplit; bid -> (tile, split). Each WG computes a
            # K/ksplit partial of its tile into workspace row band split*M; host reduces.
            _bid = fx.block_idx.x
            _ntile = ceildiv(c_m, BLOCK_M) * ceildiv(c_n, BLOCK_N)
            _tile = _bid % _ntile
            _split = _bid // _ntile
            o = _split_shift(_offs(_tile), _split)
            _fill(o)
            accL, accR = _compute(o, _split)
            _store(o, accL, accR, _split)
        else:
            o = _offs(fx.block_idx.x)
            _fill(o)
            accL, accR = _compute(o)
            _store(o, accL, accR)

    # agpr-alloc=256 lets the backend place the 256-f32 accumulator in AGPR;
    # waves_per_eu=1 -> the full 512-VGPR file is one wave's (no spill).
    _pt = {"passthrough": [["amdgpu-agpr-alloc", "256"]]}
    gemm_value_attrs = {"rocdl.flat_work_group_size": "256,256", "rocdl.waves_per_eu": OCC, **_pt}

    # Return the BARE kernel (NOT a launch): the fused factory issues preshuffle + this GEMM from one host stub.
    return kernel_gemm_4w, BLOCK_M, BLOCK_N, ksplit, gemm_value_attrs


# ── Primus-Turbo host wrapper ────────────────────────────────────────────────

_MXFP4_LAUNCH_CACHE: dict = {}  # (K, gm, xcd, gn, wlv, elgk, coop, ksplit, taccw, out_fp16) -> fused launch
# (M, N, K, gm, xcd, gn, wlv, elgk, taccw, coop, out_fp16) -> [raw, compiled_or_None]
_MXFP4_AT_CACHE: dict = {}
_MXFP4_CFG_CACHE: dict = {}  # (M, N, K, out_fp16) -> (gm, gn, xcd, wlv, elgk, taccw, coop)


def _mxfp4_nt_config(M, N, K):
    """Per-shape (group_m, group_n, num_xcds) for the BN256 path (2D N-band L2
    swizzle). Mirrors the standalone production recommend_config BN256 branch:
    wide-N (nb>=96) bands nb//8; big-K down-projections band on top (K>=28672 ->
    16 narrow aligned bands; K>=11008 -> width-4 bands); else 1D GROUP_M swizzle."""
    nb = N // 256
    num_xcds = 8
    if nb >= 96:
        group_n = nb // 8
    elif K >= 28672:
        group_n = 2
        num_xcds = 16
    elif K >= 11008:
        group_n = 4
    else:
        group_n = 0
    group_m = 4
    return group_m, group_n, num_xcds


def _mxfp4_swizzle_candidates(M, N, K):
    """<=3 L2-swizzle configs for the timed autotune: the production heuristic pick plus up
    to two group_n neighbors (group_n is the dominant L2-residency axis). Keeping the
    heuristic pick in the set means autotune never regresses below it; the swizzle is a pure
    WG->tile bijection (correctness-invariant), so trimming the sweep only trades coverage."""
    gm, gn, xcd = _mxfp4_nt_config(M, N, K)
    cands = [(gm, gn, xcd)]
    for gn2 in (0, gn * 2 if gn else 4):
        c = (gm, gn2, xcd)
        if c not in cands:
            cands.append(c)
    return cands[:3]


def _autotune_mxfp4_config(M, N, K, args, out_fp16=False):
    """Pick (group_m, group_n, num_xcds) for this (M, N, K, out dtype) by a quick timed
    sweep over ``_mxfp4_swizzle_candidates`` (<=3) on the real operands; cached per shape
    and store dtype.

    The swizzle only remaps which workgroup computes which output tile, so every
    candidate is bit-identical -- we are purely chasing L2 residency / tail balance.
    Skipped (falls back to the static heuristic) during CUDA-graph capture (cannot
    time inside capture). Compiled winners are stashed in _MXFP4_AT_CACHE so the
    subsequent real launch reuses them with no recompile."""
    key = (M, N, K, out_fp16)
    cached = _MXFP4_CFG_CACHE.get(key)
    if cached is not None:
        return cached

    if torch.cuda.is_current_stream_capturing():
        cfg = (
            *_mxfp4_nt_config(M, N, K),
            10,
            9,
            False,  # taccw: off outside the timed autotune
            False,  # coop
        )
        _MXFP4_CFG_CACHE[key] = cfg
        return cfg

    _try_deepwl = K >= 8192
    _wl_opts = ((10, 9), (16, 15)) if _try_deepwl else ((10, 9),)
    compiled_cands = []
    for _wlv, _elgk in _wl_opts:
        for gm, gn, xcd in _mxfp4_swizzle_candidates(M, N, K):
            try:
                at_key = (M, N, K, gm, xcd, gn, _wlv, _elgk, False, False, out_fp16, False)
                entry = _MXFP4_AT_CACHE.get(at_key)
                if entry is None:
                    raw = _get_mxfp4_fused_launch(K, gm, xcd, gn, _wlv, _elgk, coop=False, out_fp16=out_fp16)
                    entry = [raw, compile_with_scratch_out(raw, args)]
                    _MXFP4_AT_CACHE[at_key] = entry
                compiled_cands.append(((gm, gn, xcd, _wlv, _elgk), entry[1]))
            except Exception:  # noqa: BLE001 -- a bad config must not break the GEMM
                continue
    for _ in range(5):
        for _, compiled in compiled_cands:
            compiled(*args)
    torch.cuda.synchronize()

    ITERS, REPS = 20, 8
    cand_t = {cfg: float("inf") for cfg, _ in compiled_cands}
    for _ in range(REPS):
        for cfg, compiled in compiled_cands:
            torch.cuda.synchronize()
            e0 = torch.cuda.Event(enable_timing=True)
            e1 = torch.cuda.Event(enable_timing=True)
            e0.record()
            for _ in range(ITERS):
                compiled(*args)
            e1.record()
            torch.cuda.synchronize()
            cand_t[cfg] = min(cand_t[cfg], e0.elapsed_time(e1))
    _WL_MARGIN = 1.02
    best, best_t = None, float("inf")
    for cfg, t in cand_t.items():
        _teff = t * _WL_MARGIN if cfg[3:5] != (10, 9) else t
        if _teff < best_t:
            best_t, best = _teff, cfg
    if best is None:
        best = (*_mxfp4_nt_config(M, N, K), 10, 9)
    # Time the {TACCW,COOP,both} epilogue twins against the plain winner, keep the fastest.
    best = (*best, False, False)  # append (taccw, coop) = (False, False)
    _try_var = best_t < float("inf")
    if _try_var:
        gm0, gn0, xcd0, w0, e0 = best[:5]
        try:
            df_compiled = _MXFP4_AT_CACHE[(M, N, K, gm0, xcd0, gn0, w0, e0, False, False, out_fp16, False)][1]
            variants = []  # (taccw, coop, compiled)
            for _cp, _tw in ((False, True), (True, False), (True, True)):
                vkey = (M, N, K, gm0, xcd0, gn0, w0, e0, _tw, _cp, out_fp16, False)
                ventry = _MXFP4_AT_CACHE.get(vkey)
                if ventry is None:
                    vraw = _get_mxfp4_fused_launch(
                        K, gm0, xcd0, gn0, w0, e0, taccw=_tw, coop=_cp, out_fp16=out_fp16
                    )
                    ventry = [vraw, compile_with_scratch_out(vraw, args)]
                    _MXFP4_AT_CACHE[vkey] = ventry
                variants.append((_tw, _cp, ventry[1]))
            for _ in range(5):  # warm every twin + the winner into the same L2/clock state
                df_compiled(*args)
                for _, _, _vc in variants:
                    _vc(*args)
            torch.cuda.synchronize()

            def _time(fn):
                _q0 = torch.cuda.Event(enable_timing=True)
                _q1 = torch.cuda.Event(enable_timing=True)
                torch.cuda.synchronize()
                _q0.record()
                for _ in range(ITERS):
                    fn(*args)
                _q1.record()
                torch.cuda.synchronize()
                return _q0.elapsed_time(_q1)

            df_t = float("inf")
            vt = [float("inf")] * len(variants)
            _VREPS = 16  # extra reps: the twin wins are small, so we need a tight min
            for _ in range(_VREPS):  # round-robin so every cand shares the same thermal window
                df_t = min(df_t, _time(df_compiled))
                for _i, (_, _, _vc) in enumerate(variants):
                    vt[_i] = min(vt[_i], _time(_vc))
            # Keep the fastest twin only if it beats the plain winner by > the noise floor.
            # Unlike the wl-depth axis (needs a wide 1% guard), COOP/TACCW are stable mechanism
            # swaps whose small wins are real and stack, so a tight 0.5% guard is safe (worst
            # case: pick a noise-equal twin -> no regression).
            _bi, _bt = -1, df_t * 0.995
            for _i in range(len(variants)):
                if vt[_i] < _bt:
                    _bt, _bi = vt[_i], _i
            if _bi >= 0:
                _tw, _cp, _ = variants[_bi]
                best = (gm0, gn0, xcd0, w0, e0, _tw, _cp)
        except Exception:  # noqa: BLE001 -- a bad twin must not break the GEMM
            pass
    _MXFP4_CFG_CACHE[key] = best
    return best


_MXFP4_KSPLIT_CACHE: dict = {}  # (M, N, K, out_fp16) -> chosen ksplit (timed, never regresses vs 1)


def _ksplit_candidates(M, N, K):
    """Split-K candidates for the timed ksplit autotune. Only FEW-TILE large-K shapes
    (the one-WG-per-tile grid leaves CUs idle) are worth splitting; the sweep always
    includes ksplit=1 and takes the global min, so a bad split can never regress a shape.
    ksplit must divide K//256 (whole 256-K blocks per split) and K//ksplit % 256 == 0."""
    tiles = ceildiv(M, 256) * ceildiv(N, 256)
    ncu = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
    if tiles >= ncu // 2 or K < 2048:
        return [1]  # already enough WGs to fill the CUs (or K too small to split)
    kb = K // 256
    splits = [s for s in (2, 3, 4, 6, 8, 12, 16) if kb % s == 0 and s <= kb and tiles * s <= ncu * 2]
    # <=3 configs: baseline + the two largest valid splits (max CU fill for few-tile large-K);
    # ksplit=1 stays in the set + the sweep takes the global min, so trimming never regresses.
    return [1, *splits[-2:]]


def _compile_mxfp4_fused(
    K, gm, xcd, gn, wlv=10, elgk=9, ksplit=1, taccw=False, coop=False, out_fp16=False, beta_is_one=False
):
    """Turbo/mxfp8-style fused @flyc.jit stub: ONE host dispatch enqueues the A scale
    preshuffle, the B scale preshuffle, then the NT GEMM on the same stream (no separate
    preshuffle launch, no CPU sync). The preshuffle kernels repack raw E8M0 (int32-viewed)
    into the caller-owned a_sp/b_sp packed-int32 workspace; the GEMM reads it in stream
    order. ksplit>1 writes K/ksplit partials into a [ksplit*M, N] workspace C (the host sums
    the row bands outside the stub)."""
    K128 = K // 128
    pre_ab = _build_mxfp4_preshuffle_kernel_ab()
    gemm_kern, BM, BN, _ks, gemm_value_attrs = _build_mxfp4_gemm_kernel(
        K=K,
        group_m=gm,
        num_xcds=xcd,
        group_n=gn,
        wlv=wlv,
        elgk=elgk,
        coop=coop,
        ksplit=ksplit,
        taccw=taccw,
        out_fp16=out_fp16,
        beta_is_one=beta_is_one,
    )
    _PGRID = _MXFP4_PRESHUF_FO * _MXFP4_PRESHUF_BLK  # threads-per-block * fan-out

    @flyc.jit
    def launch_mxfp4_fused(
        A: fx.Tensor,
        B_T: fx.Tensor,
        C: fx.Tensor,
        A_raw: fx.Tensor,
        B_raw: fx.Tensor,
        A_scale: fx.Tensor,
        B_scale: fx.Tensor,
        c_m: fx.Int32,
        c_n: fx.Int32,
        stream: fx.Stream,
    ):
        # Merge A + B scale preshuffle into ONE launch to drop a kernel launch/gap; grid = ceildiv(dim*K128, FO*BLK), dim*K128 % FO == 0.
        grid_a = ceildiv(c_m * fx.Int32(K128), _PGRID)
        grid_b = ceildiv(c_n * fx.Int32(K128), _PGRID)
        # Dense enforces M/N % 256 == 0, so the packed extent equals the real row count.
        pre_ab(A_raw, A_scale, B_raw, B_scale, c_m, c_n, c_m, c_n, fx.Int32(K128), grid_a).launch(
            grid=(grid_a + grid_b, 1, 1),
            block=(_MXFP4_PRESHUF_BLK, 1, 1),
            stream=stream,
        )
        # 3) NT GEMM (reads the just-written A_scale/B_scale ws; same stream => ordered).
        grid_x = ceildiv(c_m, BM) * ceildiv(c_n, BN)
        if const_expr(ksplit > 1):
            grid_x = grid_x * fx.Int32(ksplit)  # split-K: one WG per (tile, split)
        gemm_kern(A, B_T, C, A_scale, B_scale, c_m, c_n, value_attrs=gemm_value_attrs).launch(
            grid=(grid_x, 1, 1), block=(256, 1, 1), stream=stream
        )

    return launch_mxfp4_fused


def _get_mxfp4_fused_launch(
    K, gm, xcd, gn, wlv=10, elgk=9, ksplit=1, taccw=False, coop=False, out_fp16=False, beta_is_one=False
):
    # wlv/elgk pick the phase-barrier in-flight memory depth (autotuned per shape).
    # ksplit>1 compiles a K/ksplit slice kernel for few-tile large-K shapes.
    # coop/taccw are the never-regress scale-load / wide-store autotune axes.
    # out_fp16 selects the store cast dtype (fp16 vs bf16); fp16 implies taccw=False.
    lk = (K, gm, xcd, gn, wlv, elgk, coop, ksplit, taccw, out_fp16, beta_is_one)
    launch = _MXFP4_LAUNCH_CACHE.get(lk)
    if launch is None:
        launch = _compile_mxfp4_fused(
            K,
            gm,
            xcd,
            gn,
            wlv=wlv,
            elgk=elgk,
            ksplit=ksplit,
            taccw=taccw,
            coop=coop,
            out_fp16=out_fp16,
            beta_is_one=beta_is_one,
        )
        _MXFP4_LAUNCH_CACHE[lk] = launch
    return launch


# Scale preshuffle (separate FlyDSL kernel; mirrors the mxfp8 gemm's decoupling): the quant
# emits canonical E8M0 [DIM, K/32]; this repacks them into the lane-contiguous packed int32
# layout ScaleS2RPacked consumes, run once on-stream before the GEMM. Gather form: one thread
# per output dword; decoding the packed index (wi,kk,lane,last) + inverting the A/B group map
# gives the 4 source rows grp*64 + t*16 + r. Forward map of the deleted C++ preshuffle index.

_MXFP4_PRESHUF_BLK = 256
_MXFP4_PRESHUF_NG = 4  # g bytes packed by one thread
_MXFP4_PRESHUF_ND = 4  # (r_region, K sub-block) cells packed by one thread
_MXFP4_PRESHUF_FO = _MXFP4_PRESHUF_NG * _MXFP4_PRESHUF_ND  # output dwords per thread
_MXFP4_SCALE_WS: dict = {}  # (M, N, K, device) -> (a_sp, b_sp) packed int32 workspace


def _mxfp4_pack_cell(dws, n_sub, nd, ng):
    """Byte-transpose one preshuffle cell so each output dword gathers byte g across
    source rows. Returns ng lists of nd dwords, each contiguous in the packed layout
    so it stores as one vector."""
    I32 = fx.Int32
    out = []
    for g in range_constexpr(ng):
        sh = I32(g * 8)
        grp = []
        for last in range_constexpr(nd):
            p = I32(0)
            for t in range_constexpr(nd):
                p = p | (((dws[(last // n_sub) * nd + t][last % n_sub] >> sh) & I32(0xFF)) << I32(t * 8))
            grp.append(p)
        out.append(grp)
    return out


def _mxfp4_grp_from(wi, r_region, mode):
    # Inverse of compute_preshuffle_scale_index_mxfp4's group map. Plain Python helper
    # (NOT inside the @flyc.kernel body) so the mode branch is a trace-time Python if.
    if mode == 0:  # A: grp = 2*wi + r_region
        return 2 * wi + r_region
    # B: stride-2 groups with block interleave (g0 = 4*(wi//2)+(wi%2); grp = g0 + 2*r_region)
    return 4 * (wi // 2) + (wi % 2) + 2 * r_region


def _build_mxfp4_preshuffle_kernel_ab(b_ilv=0):
    # Merged A+B scale preshuffle: ONE grid repacks BOTH operands so the fused stub issues a
    # single preshuffle launch instead of two -> one fewer launch + gap per GEMM (bigger win
    # on small-M/N). Blocks [0, grid_a) do A (mode 0); [grid_a, ...) do B (mode 1); the A/B
    # group map, buffer resources and dim are segment-selected from the WG-uniform block index
    # (readfirstlane -> SGPR arith.select). Per-thread packing math = the single-operand map.
    n_sub = 2
    nd = _MXFP4_PRESHUF_ND
    n_rr = nd // n_sub
    NG = _MXFP4_PRESHUF_NG
    FO = _MXFP4_PRESHUF_FO
    assert not b_ilv or b_ilv == nd

    @flyc.kernel(known_block_size=[_MXFP4_PRESHUF_BLK, 1, 1])
    def kern(
        a_raw: fx.Tensor,
        a_out: fx.Tensor,
        b_raw: fx.Tensor,
        b_out: fx.Tensor,
        dim_a: fx.Int32,
        dim_b: fx.Int32,
        rd_a: fx.Int32,
        rd_b: fx.Int32,
        K128: fx.Int32,
        grid_a: fx.Int32,
    ):
        KK = K128 // n_sub  # K/256
        # workgroup-uniform block index -> SGPR so the segment cond is SCC (scalar), which
        # lets arith.select route the SGPR buffer descriptors (a VGPR rsrc is invalid).
        bid = rocdl.readfirstlane(T.i32, fx.block_idx.x)
        is_b = bid >= grid_a
        local_bid = arith.select(is_b, bid - grid_a, bid)
        dim = arith.select(is_b, dim_b, dim_a)
        rd = arith.select(is_b, rd_b, rd_a)
        # Per-segment source/dest resources (each carries its own num_records bound).
        a_rin = buffer_ops.create_buffer_resource(a_raw, max_size=False, num_records_bytes=rd_a * K128 * 4)
        a_rout = buffer_ops.create_buffer_resource(a_out, max_size=False, num_records_bytes=dim_a * K128 * 4)
        b_rin = buffer_ops.create_buffer_resource(b_raw, max_size=False, num_records_bytes=rd_b * K128 * 4)
        b_rout = buffer_ops.create_buffer_resource(b_out, max_size=False, num_records_bytes=dim_b * K128 * 4)
        rin = arith.select(is_b, b_rin, a_rin)
        rout = arith.select(is_b, b_rout, a_rout)

        gid = local_bid * _MXFP4_PRESHUF_BLK + fx.thread_idx.x
        total = dim * K128  # output int32 dwords for the active operand
        ok = gid < total // FO

        r = gid % 16
        e2 = gid // 16
        kk = e2 % KK
        wi = e2 // KK
        k128 = kk * n_sub  # the thread's two K sub-blocks are adjacent source dwords
        base = ((wi * KK + kk) * 64 + r) * nd

        dws = []
        for r_region in range_constexpr(n_rr):
            # both A/B group maps emitted then segment-selected at runtime (trace-time Python if).
            grp = arith.select(is_b, _mxfp4_grp_from(wi, r_region, 1), _mxfp4_grp_from(wi, r_region, 0))
            for t in range_constexpr(nd):
                loc = arith.select(is_b, r * b_ilv + t, t * 16 + r) if b_ilv else (t * 16 + r)
                row = grp * 64 + loc
                dws.append(
                    Vec(
                        buffer_ops.buffer_load(
                            rin, row * K128 + k128, vec_width=n_sub, dtype=T.i32, mask=ok & (row < rd)
                        )
                    )
                )
        words = _mxfp4_pack_cell(dws, n_sub, nd, NG)
        for g in range_constexpr(NG):
            buffer_ops.buffer_store(Vec.from_elements(words[g]), rout, base + g * 64, mask=ok)

    return kern


def _get_mxfp4_scale_ws(M, N, K, device):
    """Caller-owned packed-scale workspace (a_sp/b_sp), cached per (M, N, K, device). Sized
    to the ScaleS2RPacked extent (dim * K/128 int32); the preshuffle writes it and the GEMM
    reads it in stream order, so same-shape reuse on one stream is safe."""
    K128 = K // 128
    key = (M, N, K, device)
    e = _MXFP4_SCALE_WS.get(key)
    if e is None:
        a_sp = torch.empty(M * K128, dtype=torch.int32, device=device)
        b_sp = torch.empty(N * K128, dtype=torch.int32, device=device)
        _MXFP4_SCALE_WS[key] = e = (a_sp, b_sp)
    return e


def gemm_mxfp4_flydsl_kernel(
    a: torch.Tensor,
    a_scale: torch.Tensor,
    b: torch.Tensor,
    b_scale: torch.Tensor,
    *,
    trans_a: bool = False,
    trans_b: bool = True,
    out_dtype: torch.dtype = torch.bfloat16,
    trans_c: bool = False,
    beta: float = 0.0,
    out: "torch.Tensor | None" = None,
) -> torch.Tensor:
    """MXFP4 (per-32-K E8M0 block-scaled) dense GEMM, gfx950 (4-wave whole-loop).

    NT only (trans_a=False, trans_b=True): A [M, K] fp4, B [N, K] fp4, C = a @ b^T.
    ``a``/``b`` are fp4 (float4_e2m1fn_x2, 2 values/byte) so the K byte-stride is K//2.

    ``a_scale``/``b_scale`` are CANONICAL E8M0 block scales ([M, K/32] / [N, K/32],
    row-major, as emitted by the generic quant). This wrapper repacks them into the
    lane-contiguous packed int32 layout the whole-loop reads (VGPR-direct scale path)
    via a separate FlyDSL preshuffle kernel launched on the same stream right before
    the GEMM -- so the quant stays generic (no fused scale write). Mirrors the mxfp8
    GEMM's quant/preshuffle decoupling; ``a`` is always the A operand, ``b`` the B.

    Constraints: K % 256 == 0; M % 256 == 0; N % 256 == 0.

    The whole-loop bare-asm body is an unroll-2 ping-pong that processes K in PAIRS
    of BLOCK_K=256. An odd number of 256-K blocks (K % 512 == 256) is handled by a
    single MFMA-only phase-A tail emitted after the hardware loop (see
    ``call_mxfp4_wholeloop``'s ``ki``/tail path);     the loop runs ``KI//2`` full pairs
    and the trailing block is accumulated by the tail. K=256 (KI==1) omits the loop
    entirely and runs only the tail. No host-side K padding is required.

    ``beta=1.0`` accumulates into ``out`` (``out += a @ b^T``) instead of overwriting it,
    and therefore requires ``out``. It is incompatible with ``trans_c``, whose transpose
    is a post-kernel copy.
    """
    assert a.dim() == 2 and b.dim() == 2, "a, b must be 2D"
    assert out_dtype in (torch.bfloat16, torch.float16), "mxfp4 FlyDSL store emits bf16/fp16"
    out_fp16 = out_dtype == torch.float16
    assert not (beta == 1.0 and trans_c), (
        "beta=1.0 cannot be combined with trans_c: the transpose is a post-kernel copy, "
        "so the accumulation would land in a buffer the caller never sees."
    )
    if not ((not trans_a) and trans_b):
        raise NotImplementedError(
            "mxfp4 FlyDSL GEMM is NT only (trans_a=False, trans_b=True); "
            f"got trans_a={trans_a}, trans_b={trans_b}."
        )

    M, Kb_a = a.shape
    N, Kb_b = b.shape
    K = Kb_a * 2  # packed 2 fp4 / byte
    assert Kb_a == Kb_b, f"K mismatch: a {a.shape}, b {b.shape}"
    # K % 256: the unroll-2 whole-loop runs KI//2 pairs + an MFMA-only tail for the
    # odd trailing 256-block (see the docstring / call_mxfp4_wholeloop).
    assert K % 256 == 0, f"K must be a multiple of 256, got {K}"
    assert M % 256 == 0, f"M must be a multiple of 256, got {M}"
    assert N % 256 == 0, f"N must be a multiple of 256, got {N}"

    stream = torch.cuda.current_stream()
    # Fused (turbo/mxfp8-style) path: a single @flyc.jit stub enqueues the A+B scale preshuffle
    # then the GEMM on this stream -- one host dispatch, no separate launch/sync. The preshuffle
    # repacks canonical E8M0 into the caller-owned packed workspace (a_sp/b_sp); the quant stays
    # generic. Workspace cached per shape (stable across graph replays); the timed autotune
    # includes the fixed preshuffle so the config ranking is preserved.
    _capturing = torch.cuda.is_current_stream_capturing()
    a_sp, b_sp = _get_mxfp4_scale_ws(M, N, K, a.device)
    a_raw = a_scale.contiguous().view(torch.int32).reshape(-1)
    b_raw = b_scale.contiguous().view(torch.int32).reshape(-1)
    out = resolve_accum_out(out, beta, (M, N), a.device, out_dtype)
    beta_is_one = beta == 1.0
    # Keep the fp4 operands 2D (do NOT flatten): M*K/2 / N*K/2 exceed 2^31 int8s for
    # large M*K / N*K, which flydsl packs as an int32 dim (host CABI overflow). Both the
    # prologue G2S and the in-loop asm refill address off the rebased flat base
    # (make_fp8_rebased_tensor_and_srd), so the operand's own shape is irrelevant.
    a8 = a.contiguous().view(torch.int8)
    b8 = b.contiguous().view(torch.int8)

    # Fused stub args: (A, B_T, C, A_raw, B_raw, A_scale_ws, B_scale_ws, c_m, c_n, stream).
    # C stays 2D (StoreCPlain re-bases per row band from C's base + c_n); a 1D M*N view
    # overflows the CABI for large M*N.
    def _args_for(target):
        return (a8, b8, target, a_raw, b_raw, a_sp, b_sp, M, N, stream)

    # Both autotunes launch the GEMM dozens of times. Against a beta=1 build that would
    # fold every one of them into the caller's buffer, so tuning runs the beta=0 build
    # into a scratch and only the final launch touches `out`. Allocated lazily: a warm
    # cache never tunes and never needs it.
    _tune_ws = []

    def _tune_args():
        if not beta_is_one:
            return _args_for(out)
        if not _tune_ws:
            _tune_ws.append(torch.zeros_like(out))
        return _args_for(_tune_ws[0])

    def _tune_target():
        return _tune_args()[2]

    def _exec_plain(target=None, accum=False):
        # default one-WG-per-tile path (autotuned swizzle / pipe depth / scale-load / wide store).
        target = out if target is None else target
        cfg = _MXFP4_CFG_CACHE.get((M, N, K, out_fp16))
        if cfg is None:
            cfg = _autotune_mxfp4_config(M, N, K, _tune_args(), out_fp16)
        gm, gn, xcd, _wlv, _elgk, _tw, _coop = cfg
        launch = _get_mxfp4_fused_launch(
            K, gm, xcd, gn, _wlv, _elgk, taccw=_tw, coop=_coop, out_fp16=out_fp16, beta_is_one=accum
        )
        at_key = (M, N, K, gm, xcd, gn, _wlv, _elgk, _tw, _coop, out_fp16, accum)
        fused_args = _args_for(target)
        entry = _MXFP4_AT_CACHE.get(at_key)
        if entry is None:
            entry = [launch, None]
            _MXFP4_AT_CACHE[at_key] = entry
        raw, compiled = entry
        if _capturing:
            raw(*fused_args)
        else:
            if compiled is None:
                compiled = compile_with_scratch_out(raw, fused_args)
                entry[1] = compiled
            compiled(*fused_args)
        return target

    def _exec_split(ksplit, target=None, accum=False):
        # split-K: grid x ksplit fills the CUs on few-tile large-K shapes. Each split writes
        # its K/ksplit partial into an out_dtype workspace[ksplit*M, N]; host sums the ksplit
        # row bands (BW-bound reduce, faster than an atomic-fused reduce for these shapes).
        # The fused stub still preshuffles A/B into a_sp/b_sp before the split GEMM.
        gm, gn, xcd = _mxfp4_nt_config(M, N, K)
        ws = torch.empty((ksplit * M, N), dtype=out_dtype, device=a.device)
        cbuf = ws.view(-1)
        sk_args = (a8, b8, cbuf, a_raw, b_raw, a_sp, b_sp, M, N, stream)
        launch = _get_mxfp4_fused_launch(K, gm, xcd, gn, 10, 9, ksplit=ksplit, out_fp16=out_fp16)
        sk_key = (M, N, K, gm, xcd, gn, 10, 9, ksplit, out_fp16)
        entry = _MXFP4_AT_CACHE.get(sk_key)
        if entry is None:
            entry = [launch, None]
            _MXFP4_AT_CACHE[sk_key] = entry
        raw, compiled = entry
        if _capturing:
            raw(*sk_args)
        else:
            if compiled is None:
                compiled = compile_with_scratch_out(raw, sk_args)
                entry[1] = compiled
            compiled(*sk_args)
        reduced = ws.view(ksplit, M, N).sum(dim=0)
        if accum:
            # The splits went to a scratch, so the epilogue could not accumulate; the
            # reduce pass this path already runs absorbs it for free.
            return (out if target is None else target).add_(reduced)
        return reduced

    # ── Choose ksplit: cached timed pick > timed autotune.
    # The autotune times {plain, split+reduce} end-to-end on the real operands and takes the
    # global min, so split-K is used ONLY where it actually wins and never regresses a shape.
    # Skipped during graph capture (uses the cached pick).
    ks = _MXFP4_KSPLIT_CACHE.get((M, N, K, out_fp16))
    if ks is None:
        cands = _ksplit_candidates(M, N, K)
        if _capturing or len(cands) == 1:
            ks = 1  # cannot time inside capture / nothing to try
            if not _capturing:
                _MXFP4_KSPLIT_CACHE[(M, N, K, out_fp16)] = ks
        else:

            def _bench(fn):
                for _ in range(3):
                    fn()
                torch.cuda.synchronize()
                best = float("inf")
                for _ in range(5):
                    e0 = torch.cuda.Event(enable_timing=True)
                    e1 = torch.cuda.Event(enable_timing=True)
                    e0.record()
                    for _ in range(20):
                        fn()
                    e1.record()
                    torch.cuda.synchronize()
                    best = min(best, e0.elapsed_time(e1))
                return best

            times = {}
            tgt = _tune_target()
            for s in cands:
                try:
                    fn = (lambda: _exec_plain(tgt)) if s == 1 else (lambda s=s: _exec_split(s, tgt))
                    times[s] = _bench(fn)
                except Exception:  # noqa: BLE001 -- a bad variant must not break the GEMM
                    continue
            ks = min(times, key=times.get) if times else 1
            _MXFP4_KSPLIT_CACHE[(M, N, K, out_fp16)] = ks

    out2 = _exec_split(ks, out, beta_is_one) if ks > 1 else _exec_plain(out, beta_is_one)
    return out2.t().contiguous() if trans_c else out2
